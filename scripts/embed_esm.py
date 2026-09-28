import os
import time
import numpy as np
import pandas as pd
import torch
import esm

from encoding import esm_models, filter_rows

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
# outputs go to the repo by default; on O2 set SPF_OUTPUT to the shared project
# folder so results are computed once and pulled down rather than recomputed
output_root = os.environ.get("SPF_OUTPUT", project_dir)
variants_path = f"{output_root}/results/variants.parquet"
embedding_dir = f"{output_root}/embeddings"
family = os.environ.get("SPF_FAMILY", "PF00018")
encoding_name = os.environ.get("SPF_ESM", "ESM2-150M")
if encoding_name not in esm_models:
    raise SystemExit(f"unknown model {encoding_name!r}, expected one of {sorted(esm_models)}")
model_name, esm_layer, cache_tag, default_batch = esm_models[encoding_name]
# the larger models need a smaller batch to fit on one card, so the default follows
# the model and SPF_BATCH still overrides it
esm_batch_size = int(os.environ.get("SPF_BATCH", str(default_batch)))
row_filter = os.environ.get("SPF_ROWS", "drop_insertions")

os.makedirs(embedding_dir, exist_ok = True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"model {encoding_name} ({model_name}), layer {esm_layer}, batch {esm_batch_size}")
print(f"cache tag {cache_tag}, rows {row_filter}")


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
# the row set has to match whatever this embedding will be compared against, so it
# is named rather than assumed
variants = filter_rows(variants, row_filter)

datasets = sorted(variants['dataset'].unique())
pending = [d for d in datasets
           if not cache_matches(f"{embedding_dir}/{d}_{cache_tag}.npy",
                                int((variants['dataset'] == d).sum()))]
print(f"family {family}: {len(datasets)} datasets, {len(pending)} still to embed")
if not pending:
    print("nothing to do")
    raise SystemExit(0)

esm_model, alphabet = getattr(esm.pretrained, model_name)()
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
