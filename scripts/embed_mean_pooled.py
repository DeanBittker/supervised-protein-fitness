import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
# outputs go to the repo by default; on O2 set SPF_OUTPUT to the shared project
# folder so results are computed once and pulled down rather than recomputed
output_root = os.environ.get("SPF_OUTPUT", project_dir)
data_root = os.environ.get("SPF_DATA", f"{project_dir}/data")
variants_path = f"{output_root}/results/variants.parquet"
embedding_dir = f"{output_root}/embeddings"
model_name = os.environ.get("SPF_MODEL", "facebook/esm2_t30_150M_UR50D")
sequence_column = "mutated_sequence"
batch_size = int(os.environ.get("SPF_BATCH", "32"))
max_datasets = int(os.environ.get("SPF_MAX_DATASETS", "0"))

os.makedirs(embedding_dir, exist_ok = True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"model: {model_name}")

variants = pd.read_parquet(variants_path, columns = ['dataset', sequence_column])
datasets = sorted(variants['dataset'].unique())
if max_datasets:
    datasets = datasets[:max_datasets]
print(f"datasets to embed: {len(datasets)}")

# one file per dataset, written atomically and skipped if present, so a preempted
# requeue job resumes instead of starting over
pending = [d for d in datasets if not os.path.exists(f"{embedding_dir}/{d}.npy")]
print(f"already embedded: {len(datasets) - len(pending)}   pending: {len(pending)}")
if not pending:
    print("nothing to do")
    raise SystemExit(0)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device).eval()

for n, dataset in enumerate(pending):
    sequences = variants.loc[variants['dataset'] == dataset, sequence_column].astype(str).tolist()
    pooled = []
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i + batch_size]
            encoded = tokenizer(batch, return_tensors = "pt", padding = True, truncation = True)
            encoded = {k: v.to(device) for k, v in encoded.items()}
            hidden = model(**encoded).last_hidden_state
            # mean pool over real residues only, so padding does not dilute the mean
            mask = encoded['attention_mask'].unsqueeze(-1).float()
            pooled.append(((hidden * mask).sum(1) / mask.sum(1)).cpu().numpy())
    embeddings = np.concatenate(pooled).astype(np.float32)

    temporary = f"{embedding_dir}/{dataset}.npy.tmp"
    np.save(temporary, embeddings)
    os.replace(temporary, f"{embedding_dir}/{dataset}.npy")
    print(f"  [{n + 1}/{len(pending)}] {dataset}: {embeddings.shape}", flush = True)

print()
print(f"Embeddings written to {embedding_dir}")
