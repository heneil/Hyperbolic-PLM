import os
from esm.data import ESMStructuralSplitDataset

root_path = os.path.expanduser("/data/neilhe2/esm_data")
ds = ESMStructuralSplitDataset(
    split_level="superfamily",
    cv_partition="4",
    split="train",
    root_path=root_path,
    download=False,
)

out_fasta = "structural_superfamily4_train_50k.fasta"
N = min(len(ds), 50_000)

with open(out_fasta, "w") as f:
    for i in range(N):
        obj = ds[i]
        seq = obj["seq"]
        # unique FASTA header (FastaBatchedDataset asserts uniqueness)
        f.write(f">{ds.names[i]}|idx={i}\n")
        f.write(seq + "\n")

print("Wrote", out_fasta, "with", N, "sequences")