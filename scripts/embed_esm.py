import os
import time
import numpy as np
import pandas as pd
import torch
import esm

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
# outputs go to the repo by default; on O2 set SPF_OUTPUT to the shared project
# folder so results are computed once and pulled down rather than recomputed
output_root = os.environ.get("SPF_OUTPUT", project_dir)
variants_path = f"{output_root}/results/variants.parquet"
embedding_dir = f"{output_root}/embeddings"
family = os.environ.get("SPF_FAMILY", "PF00018")
encoding_name = "ESM2-150M"
cache_tag = "esm150M"
esm_layer = 30
esm_batch_size = int(os.environ.get("SPF_BATCH", "16"))
drop_insertions = True

os.makedirs(embedding_dir, exist_ok = True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def cache_matches(path, expected_rows):
    return os.path.exists(path) and np.load(path, mmap_mode='r').shape[0] == expected_rows


def esm_encode_batch(sequences, batch_size=esm_batch_size):
    all_embeddings = []
    total_batches = (len(sequences) + batch_size - 1) // batch_size
    start_time = time.time()
    for i in range(0, len(sequences), batch_size):
        batch_num = i // batch_size + 1
        if batch_num % max(1, total_batches // 10) == 0 or batch_num == 1:
            elapsed = time.time() - start_time
            eta = (elapsed / batch_num) * (total_batches - batch_num) if batch_num > 1 else 0
            print(f"Batch {batch_num}/{total_batches} ({batch_num * 100 // total_batches}%) - ETA: {eta:.0f}s", flush=True)
        batch_seqs = [(str(j), s) for j, s in enumerate(sequences[i:i+batch_size])]
        _, _, batch_tokens = batch_converter(batch_seqs)
        batch_tokens = batch_tokens.to(device)
        with torch.no_grad():
            results = esm_model(batch_tokens, repr_layers = [esm_layer])
        token_reps = results["representations"][esm_layer]
        for k, seq in enumerate(batch_seqs):
            seq_len = len(seq[1])
            embedding = token_reps[k, 1:seq_len+1,:].mean(dim=0)
            all_embeddings.append(embedding.cpu().numpy())
    return np.stack(all_embeddings)


variants = pd.read_parquet(variants_path)
variants = variants[variants['pfam_acc'] == family].copy()
if drop_insertions:
    # one-hot, z-scales and blosum all need a fixed length, and an insertion makes
    # the sequence longer than wild type. dropped here so every encoding sees the
    # same rows and the comparison between them is not confounded.
    before = len(variants)
    variants = variants[~variants['has_insertion']]
    print(f"dropped {before - len(variants):,} insertion rows, {len(variants):,} remain")

datasets = sorted(variants['dataset'].unique())
pending = [d for d in datasets
           if not cache_matches(f"{embedding_dir}/{d}_{cache_tag}.npy",
                                int((variants['dataset'] == d).sum()))]
print(f"family {family}: {len(datasets)} datasets, {len(pending)} still to embed")
if not pending:
    print("nothing to do")
    raise SystemExit(0)

esm_model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
batch_converter = alphabet.get_batch_converter()
esm_model = esm_model.to(device)
esm_model.eval()

for n, dataset in enumerate(pending):
    rows = variants[variants['dataset'] == dataset]
    print(f"[{n + 1}/{len(pending)}] {dataset}: {len(rows):,} sequences", flush=True)
    embeddings = esm_encode_batch(rows['mutated_sequence'].astype(str).tolist())

    # written through a temporary name so an interrupted job never leaves a
    # half-written cache that would pass the row-count check
    path = f"{embedding_dir}/{dataset}_{cache_tag}.npy"
    np.save(path + ".tmp.npy", embeddings.astype(np.float32))
    os.replace(path + ".tmp.npy", path)

print()
print(f"Embeddings written to {embedding_dir}")
