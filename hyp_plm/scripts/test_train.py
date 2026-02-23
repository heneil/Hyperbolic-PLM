#!/usr/bin/env python3 -u
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import pathlib

import torch
import torch.nn.functional as F

from esm import Alphabet, FastaBatchedDataset, ESM2, MSATransformer


def create_parser():
    parser = argparse.ArgumentParser(
        description="Extract per-token representations and model outputs for sequences in a FASTA file"  # noqa
    )

    parser.add_argument(
        "fasta_file",
        type=pathlib.Path,
        help="FASTA file on which to extract representations",
    )
    parser.add_argument(
        "output_dir",
        type=pathlib.Path,
        help="output directory for extracted representations",
    )

    parser.add_argument("--toks_per_batch", type=int, default=4096, help="maximum batch size")
    parser.add_argument(
        "--repr_layers",
        type=int,
        default=[-1],
        nargs="+",
        help="layers indices from which to extract representations (0 to num_layers, inclusive)",
    )
    parser.add_argument(
        "--include",
        type=str,
        nargs="+",
        choices=["mean", "per_tok", "bos", "contacts"],
        help="specify which representations to return",
        required=False,
    )
    parser.add_argument(
        "--truncation_seq_length",
        type=int,
        default=1022,
        help="truncate sequences longer than the given value",
    )

    parser.add_argument("--nogpu", action="store_true", help="Do not use GPU even if available")
    return parser


def run(args):
    alphabet = Alphabet.from_architecture("ESM-1b")
    model = ESM2(alphabet=alphabet)
    print(model)
    model.train()
    if isinstance(model, MSATransformer):
        raise ValueError(
            "This script currently does not handle models with MSA input (MSA Transformer)."
        )
    if torch.cuda.is_available() and not args.nogpu:
        model = model.cuda()
        print("Transferred model to GPU")
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    dataset = FastaBatchedDataset.from_file(args.fasta_file)
    batches = dataset.get_batch_indices(args.toks_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=alphabet.get_batch_converter(args.truncation_seq_length), batch_sampler=batches
    )
    print(f"Read {args.fasta_file} with {len(dataset)} sequences")

    # args.output_dir.mkdir(parents=True, exist_ok=True)
    # return_contacts = "contacts" in args.include

    assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in args.repr_layers)
    repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in args.repr_layers]

    for batch_idx, (seq_labels, strs, toks) in enumerate(data_loader):
        print(
            f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)"
        )
        if torch.cuda.is_available() and not args.nogpu:
            toks = toks.to(device="cuda", non_blocking=True)

        toks_masked = toks.clone()
        mlm_labels = toks.clone()
        special = toks.eq(alphabet.padding_idx) | toks.eq(alphabet.cls_idx) | toks.eq(alphabet.eos_idx)
        p = torch.full(mlm_labels.shape, 0.15, device=toks.device)
        p.masked_fill_(special, 0.0)
        masked = torch.bernoulli(p).bool()
        mlm_labels[~masked] = -100
        replace_p = torch.full(mlm_labels.shape, 0.8, device=toks.device)
        replaced = torch.bernoulli(replace_p).bool() & masked
        toks_masked[replaced] = alphabet.mask_idx
        std = torch.tensor([alphabet.get_idx(t) for t in alphabet.standard_toks], device=toks.device)
        rand_ids = std[torch.randint(low=0, high=std.numel(), size=mlm_labels.shape, device=toks.device)]
        random = (torch.bernoulli(torch.full(mlm_labels.shape, 0.5, device=toks.device)).bool() & masked & ~replaced)
        toks_masked[random] = rand_ids[random]

        out = model(toks_masked)

        logits = out["logits"]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), mlm_labels.reshape(-1), ignore_index=-100)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        print(f"batch {batch_idx}: loss={loss.item():.4f}")


def main():
    parser = create_parser()
    args = parser.parse_args()
    run(args)

if __name__ == "__main__":
    main()
