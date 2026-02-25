#!/usr/bin/env python3
"""
Shuffle FASTA and split into train/eval/test.

Policy:
  - eval: fixed N sequences (default 100)
  - from remaining sequences:
      train: train_ratio (default 0.99)
      test: the rest

Usage:
  python scripts/split_fasta_train_eval_test.py /path/to/uniref.fasta /path/to/out --eval_n 100 --train_ratio 0.99 --seed 42
"""
import argparse
import pathlib
import random
import sys


def read_fasta(path: pathlib.Path):
    """Yield (desc, seq) for each record in FASTA. No external deps."""
    desc = None
    seq_chunks = []
    with open(path, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if desc is not None:
                    yield desc, "".join(seq_chunks)
                desc = line[1:].strip()
                seq_chunks = []
            else:
                seq_chunks.append(line.strip())
        if desc is not None:
            yield desc, "".join(seq_chunks)


def write_fasta(path: pathlib.Path, recs):
    with open(path, "w") as f:
        for desc, seq in recs:
            f.write(f">{desc}\n{seq}\n")


def main():
    parser = argparse.ArgumentParser(description="Random shuffle FASTA and split into train / eval / test.")
    parser.add_argument("fasta_file", type=pathlib.Path, help="Full FASTA file.")
    parser.add_argument("output_dir", type=pathlib.Path, help="Directory to write train/eval/test FASTA.")
    parser.add_argument("--eval_n", type=int, default=300, help="Number of eval sequences (default 300).")
    parser.add_argument("--train_ratio", type=float, default=0.99, help="Train ratio from remaining (default 0.99).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    if not (0.0 <= args.train_ratio <= 1.0):
        print("Error: --train_ratio must be in [0, 1].", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fasta_file = args.fasta_file.resolve()
    if not fasta_file.exists():
        print("Error: FASTA not found:", fasta_file, file=sys.stderr)
        sys.exit(1)

    records = list(read_fasta(fasta_file))
    n = len(records)
    if n == 0:
        print("Error: No records found in FASTA.", file=sys.stderr)
        sys.exit(1)

    rng = random.Random(args.seed)
    rng.shuffle(records)

    # 1) eval fixed N (cap at total)
    n_eval = min(args.eval_n, n)
    eval_records = records[:n_eval]
    remaining = records[n_eval:]
    n_rem = len(remaining)

    # 2) train ratio from remaining
    n_train = int(n_rem * args.train_ratio)
    # keep bounds sane
    n_train = max(0, min(n_train, n_rem))
    train_records = remaining[:n_train]

    # 3) test is the rest
    test_records = remaining[n_train:]

    train_path = args.output_dir / "train.fasta"
    eval_path = args.output_dir / "eval.fasta"
    test_path = args.output_dir / "test.fasta"

    write_fasta(train_path, train_records)
    write_fasta(eval_path, eval_records)
    write_fasta(test_path, test_records)

    print(
        f"Total: {n} -> train {len(train_records)}, eval {len(eval_records)}, test {len(test_records)}.\n"
        f"Written:\n  {train_path}\n  {eval_path}\n  {test_path}"
    )


if __name__ == "__main__":
    main()