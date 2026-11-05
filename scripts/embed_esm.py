import os
import time
import numpy as np
import pandas as pd

# torch and esm are imported only once there is something to embed. They cost a
# GPU-sized install and the better part of a minute, and nothing above the model load
# needs them, so a bad configuration or an already-complete shard fails or exits
# before paying for either.
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

# a newline-delimited file of dataset names, used instead of the family filter. A
# clan spans many families, so a clan-level run cannot be selected by SPF_FAMILY at
# all; make_clan_folds.py writes a list of exactly the datasets its folds need.
dataset_list = os.environ.get("SPF_DATASETS", "").strip()
# one array task takes every nth dataset from the list, largest first, so the big
# datasets are spread across tasks instead of landing in one
shards = int(os.environ.get("SPF_SHARDS",
                            os.environ.get("SLURM_ARRAY_TASK_COUNT", "1")))
shard_index = int(os.environ.get("SPF_SHARD_INDEX",
                                 os.environ.get("SLURM_ARRAY_TASK_ID", "1")))
if shards < 1 or shard_index < 1:
    raise SystemExit(f"shard {shard_index} of {shards} is not a shard")
plan_only = os.environ.get("SPF_DRYRUN", "") == "1"

os.makedirs(embedding_dir, exist_ok = True)

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
if dataset_list:
    if not os.path.exists(dataset_list):
        raise SystemExit(f"no dataset list at {dataset_list}")
    wanted = [line.strip() for line in open(dataset_list) if line.strip()]
    if not wanted:
        raise SystemExit(f"{dataset_list} is empty")
    present = set(variants['dataset'].unique())
    absent = [d for d in wanted if d not in present]
    if absent:
        print(f"{len(absent)} of {len(wanted)} listed datasets are not in "
              f"{os.path.basename(variants_path)} and are skipped, first few: "
              f"{absent[:3]}")
    variants = variants[variants['dataset'].isin(set(wanted) & present)].copy()
    selection = f"list {os.path.basename(dataset_list)}"
else:
    variants = variants[variants['pfam_acc'] == family].copy()
    selection = f"family {family}"
# the row set has to match whatever this embedding will be compared against, so it
# is named rather than assumed
variants = filter_rows(variants, row_filter)
if variants.empty:
    raise SystemExit("no variants selected, nothing to embed")

counts = variants.groupby('dataset').size()
# Shard the whole list, biggest first, before looking at the cache. Sharding the
# pending list instead meant each task strided into a list that the other tasks
# running beside it were shortening as they wrote, so the slices did not partition
# anything and datasets fell between them. This assignment depends only on the list
# and the shard number, so it is the same whenever a task starts and whatever else
# has finished.
ordered = sorted(counts.index, key = lambda name: (-int(counts[name]), name))
print(f"{selection}: {len(ordered)} datasets")
if shards > 1:
    ordered = ordered[(shard_index - 1) % shards::shards]
    print(f"shard {shard_index} of {shards}: {len(ordered)} datasets, "
          f"{int(counts[ordered].sum()):,} sequences assigned")

pending = [d for d in ordered
           if not cache_matches(f"{embedding_dir}/{d}_{cache_tag}.npy",
                                int(counts[d]))]
print(f"{len(pending)} of this shard's {len(ordered)} still to embed")

if not pending:
    print("nothing to do")
    raise SystemExit(0)
if plan_only:
    print(f"\ndry run, would embed {len(pending)} datasets "
          f"({int(counts[pending].sum()):,} sequences):")
    for name in pending[:10]:
        print(f"  {name}: {int(counts[name]):,}")
    if len(pending) > 10:
        print(f"  ... and {len(pending) - 10} more")
    raise SystemExit(0)

import torch
import esm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
