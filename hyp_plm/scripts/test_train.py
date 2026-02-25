#!/usr/bin/env python3 -u
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import pathlib
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from tqdm import tqdm

from esm import Alphabet, FastaBatchedDataset, ESM2, MSATransformer
from esm.model.hyp_esm2 import LorentzESM2


def run_eval(model, eval_loader, alphabet, device):
    """Run MLM evaluation on eval_loader, return average loss. No grad."""
    model.eval()
    total_loss = 0.0
    n_toks = 0
    with torch.no_grad():
        for seq_labels, strs, toks in tqdm(eval_loader, desc="eval", leave=False):
            toks = toks.to(device=device, non_blocking=True)
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
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), mlm_labels.reshape(-1), ignore_index=-100, reduction="sum"
            )
            n = (mlm_labels.reshape(-1) != -100).sum().item()
            total_loss += loss.item()
            n_toks += n
    model.train()
    return total_loss / max(n_toks, 1)


def load_config(config_path):
    """Load YAML config and return a SimpleNamespace for run(). Paths relative to hyp_plm (parent of configs/)."""
    try:
        import yaml
    except ImportError:
        raise RuntimeError("Config is YAML; install PyYAML: pip install pyyaml")
    config_path = pathlib.Path(config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    root = config_path.parent.parent  # hyp_plm
    t = raw.get("train") or raw
    def p(s):
        if s is None or (isinstance(s, str) and s.strip() == ""):
            return None
        return (root / s).resolve()
    model_type = (t.get("model") or "esm2").strip().lower()
    if model_type not in ("esm2", "lorentz_esm2"):
        raise ValueError(f"config train.model 必须是 'esm2' 或 'lorentz_esm2'，当前为: {model_type!r}")
    cfg = SimpleNamespace(
        model=model_type,
        fasta_file=p(t.get("fasta_file")),
        output_dir=p(t.get("output_dir")) or (root / "output").resolve(),
        toks_per_batch=int(t.get("toks_per_batch", 4096)),
        truncation_seq_length=int(t.get("truncation_seq_length", 1022)),
        repr_layers=list(t.get("repr_layers", [-1])),
        nogpu=bool(t.get("nogpu", False)),
        eval_fasta=p(t.get("eval_fasta")) if t.get("eval_fasta") else None,
        eval_every=int(t.get("eval_every", 500)),
        eval_at_start=bool(t.get("eval_at_start", False)),
        test_fasta=p(t.get("test_fasta")) if t.get("test_fasta") else None,
        local_rank=int(os.environ.get("LOCAL_RANK", -1)),
        wandb_project=t.get("wandb_project") or "hyperbolic-plm",
        wandb_run_name=t.get("wandb_run_name") or None,
    )
    return cfg


def run(cfg):
    use_ddp = int(os.environ.get("LOCAL_RANK", -1)) >= 0 and torch.cuda.is_available() and not cfg.nogpu
    if use_ddp:
        if cfg.local_rank >= 0:
            local_rank = cfg.local_rank
        else:
            local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device("cuda", local_rank)
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda" if (torch.cuda.is_available() and not cfg.nogpu) else "cpu")

    if rank == 0:
        try:
            import wandb
            wandb.init(
                project=cfg.wandb_project,
                name=cfg.wandb_run_name,
                config={
                    "model": cfg.model,
                    "toks_per_batch": cfg.toks_per_batch,
                    "truncation_seq_length": cfg.truncation_seq_length,
                    "eval_every": cfg.eval_every,
                    "world_size": world_size,
                },
            )
        except Exception as e:
            wandb = None
            print(f"wandb 未启用: {e}")
    else:
        wandb = None

    alphabet = Alphabet.from_architecture("ESM-1b")
    if cfg.model == "esm2":
        model = ESM2(alphabet=alphabet)
    else:
        model = LorentzESM2(alphabet=alphabet)
    if rank == 0:
        print(model)
    model.train()
    if isinstance(model, MSATransformer):
        raise ValueError(
            "This script currently does not handle models with MSA input (MSA Transformer)."
        )
    model = model.to(device)
    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    elif device.type == "cuda":
        if rank == 0:
            print("Transferred model to GPU")

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    dataset = FastaBatchedDataset.from_file(cfg.fasta_file)
    batches_all = dataset.get_batch_indices(cfg.toks_per_batch, extra_toks_per_seq=1)

    if use_ddp:
        batches_this_rank = batches_all[rank::world_size]
    else:
        batches_this_rank = batches_all

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=alphabet.get_batch_converter(cfg.truncation_seq_length),
        batch_sampler=batches_this_rank,
    )
    if rank == 0:
        print(f"Read {cfg.fasta_file} with {len(dataset)} sequences (world_size={world_size}, batches per rank={len(batches_this_rank)})")

    eval_loader = None
    if cfg.eval_fasta is not None and cfg.eval_fasta.exists() and rank == 0:
        eval_dataset = FastaBatchedDataset.from_file(cfg.eval_fasta)
        eval_batches = eval_dataset.get_batch_indices(cfg.toks_per_batch, extra_toks_per_seq=1)
        eval_loader = torch.utils.data.DataLoader(
            eval_dataset,
            collate_fn=alphabet.get_batch_converter(cfg.truncation_seq_length),
            batch_sampler=eval_batches,
        )
        if rank == 0:
            print(f"Eval set: {cfg.eval_fasta} with {len(eval_dataset)} sequences, eval every {cfg.eval_every} batches")

    unwrapped = model.module if use_ddp else model
    num_layers = unwrapped.num_layers
    assert all(-(num_layers + 1) <= i <= num_layers for i in cfg.repr_layers)
    repr_layers = [(i + num_layers + 1) % (num_layers + 1) for i in cfg.repr_layers]

    if rank == 0 and eval_loader is not None and cfg.eval_at_start:
        eval_loss = run_eval(model, eval_loader, alphabet, device)
        print(f"[eval] batch 0 (before training): eval_loss={eval_loss:.4f}")
        if wandb is not None:
            wandb.log({"eval_loss": eval_loss, "batch": 0})

    for batch_idx, (seq_labels, strs, toks) in enumerate(data_loader):
        if rank == 0 and batch_idx % 10 == 0:
            print(
                f"Processing {batch_idx + 1} of {len(batches_this_rank)} batches ({toks.size(0)} sequences)"
            )
        toks = toks.to(device=device, non_blocking=True)

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
        if rank == 0:
            print(f"batch {batch_idx}: train_loss={loss.item():.4f}")
            if wandb is not None:
                wandb.log({"train_loss": loss.item(), "batch": batch_idx + 1})
        if rank == 0 and eval_loader is not None and (batch_idx + 1) % cfg.eval_every == 0:
            eval_loss = run_eval(model, eval_loader, alphabet, device)
            print(f"[eval] batch {batch_idx + 1}: eval_loss={eval_loss:.4f}")
            if wandb is not None:
                wandb.log({"eval_loss": eval_loss, "batch": batch_idx + 1})


def main():
    parser = argparse.ArgumentParser(description="Train with config file (YAML).")
    parser.add_argument("--config", type=pathlib.Path, default=pathlib.Path("configs/train.yaml"), help="Path to config YAML (default: configs/train.yaml).")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for DDP (set by torchrun).")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.local_rank >= 0:
        cfg.local_rank = args.local_rank
    run(cfg)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
