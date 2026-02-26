#!/usr/bin/env python3 -u
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import math
import os
import pathlib
import sys
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from tqdm import tqdm

from esm import Alphabet, FastaBatchedDataset, ESM2, MSATransformer
from esm.model.hyp_esm2 import LorentzESM2


def log_unused_parameters(model, step, rank):
    """在 backward 之后调用：打印 grad 为 None 的参数（未参与计算图）。仅当 DDP_DEBUG_UNUSED=1 时用。"""
    unwrapped = model.module if hasattr(model, "module") else model
    unused = [name for name, p in unwrapped.named_parameters() if p.requires_grad and p.grad is None]
    if rank == 0 and unused:
        print(f"[DDP debug] step {step}: {len(unused)} parameters with no gradient:")
        for name in unused:
            print(f"  - {name}")
    return unused


def print_parameter_index_map(model, rank):
    """打印 (index, name) 映射，便于根据 DDP 报错里的 parameter indices 查名字。"""
    unwrapped = model.module if hasattr(model, "module") else model
    for i, (name, _) in enumerate(unwrapped.named_parameters()):
        if rank == 0:
            print(f"  [{i}] {name}")


def run_eval(model, eval_loader, alphabet, device):
    """Run MLM evaluation on eval_loader. No grad. Returns (total_loss, n_toks) for potential all_reduce."""
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
    return total_loss, n_toks


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
        max_epochs=int(t.get("max_epochs", 1)),
        local_rank=int(os.environ.get("LOCAL_RANK", -1)),
        wandb_project=t.get("wandb_project") or "hyperbolic-plm",
        wandb_run_name=t.get("wandb_run_name") or None,
    )
    return cfg


def run(cfg):
    ddp_debug_unused = os.environ.get("DDP_DEBUG_UNUSED", "0") == "1"
    use_ddp = int(os.environ.get("LOCAL_RANK", -1)) >= 0 and torch.cuda.is_available() and not cfg.nogpu and not ddp_debug_unused

    if int(os.environ.get("LOCAL_RANK", -1)) >= 0 and torch.cuda.is_available() and not cfg.nogpu:
        if cfg.local_rank >= 0:
            local_rank = cfg.local_rank
        else:
            local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if ddp_debug_unused:
            # 多卡启动时只让 rank 0 跑单卡一步，不包 DDP，这样会报错时照常报错，同时能打印未参与梯度的参数
            if rank != 0:
                dist.destroy_process_group()
                return
            use_ddp = False
            world_size = 1
            device = torch.device("cuda", 0)
            if rank == 0:
                print("[DDP debug] DDP_DEBUG_UNUSED=1: running single-GPU (no DDP), one step, then log unused params and exit.")
        else:
            device = torch.device("cuda", local_rank)
            if world_size == 1:
                use_ddp = False  # 单卡不包 DDP，便于直接看未参与梯度的参数
    elif use_ddp:
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
    if os.environ.get("DDP_PRINT_PARAM_INDEX", "") == "1" and rank == 0:
        print("[DDP] Parameter index -> name (set DDP_PRINT_PARAM_INDEX=1 to see this):")
        print_parameter_index_map(model, rank)
    if use_ddp:
        find_unused = os.environ.get("DDP_FIND_UNUSED_PARAMETERS", "0") == "1"
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=find_unused,
        )
        if rank == 0 and find_unused:
            print("[DDP] DDP_FIND_UNUSED_PARAMETERS=1: find_unused_parameters=True (slight overhead).")
    elif device.type == "cuda":
        if rank == 0:
            print("Transferred model to GPU")

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    dataset = FastaBatchedDataset.from_file(cfg.fasta_file)
    batches_all = dataset.get_batch_indices(cfg.toks_per_batch, extra_toks_per_seq=1)

    if use_ddp:
        batches_this_rank = list(batches_all[rank::world_size])
        # 多卡时保证每 rank 每 epoch 的 batch 数相同，否则 global_step 不同步，eval 时 barrier 死锁
        max_batches = (len(batches_all) + world_size - 1) // world_size
        while len(batches_this_rank) < max_batches:
            batches_this_rank.append(batches_this_rank[0])
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
    if cfg.eval_fasta is not None and cfg.eval_fasta.exists():
        eval_dataset = FastaBatchedDataset.from_file(cfg.eval_fasta)
        eval_batches_all = eval_dataset.get_batch_indices(cfg.toks_per_batch, extra_toks_per_seq=1)
        if use_ddp:
            eval_batches_this_rank = list(eval_batches_all[rank::world_size])
            max_eval = (len(eval_batches_all) + world_size - 1) // world_size
            while len(eval_batches_this_rank) < max_eval:
                eval_batches_this_rank.append(eval_batches_this_rank[0])
            eval_loader = torch.utils.data.DataLoader(
                eval_dataset,
                collate_fn=alphabet.get_batch_converter(cfg.truncation_seq_length),
                batch_sampler=eval_batches_this_rank,
            )
        else:
            if rank == 0:
                eval_loader = torch.utils.data.DataLoader(
                    eval_dataset,
                    collate_fn=alphabet.get_batch_converter(cfg.truncation_seq_length),
                    batch_sampler=eval_batches_all,
                )
        if rank == 0:
            print(f"Eval set: {cfg.eval_fasta} with {len(eval_dataset)} sequences, eval every {cfg.eval_every} batches")

    unwrapped = model.module if use_ddp else model
    num_layers = unwrapped.num_layers
    assert all(-(num_layers + 1) <= i <= num_layers for i in cfg.repr_layers)
    repr_layers = [(i + num_layers + 1) % (num_layers + 1) for i in cfg.repr_layers]

    batches_per_epoch = len(batches_this_rank)
    if rank == 0:
        print(f"Total epochs: {cfg.max_epochs}, batches per epoch (this rank): {batches_per_epoch}")

    if rank == 0 and eval_loader is not None and cfg.eval_at_start:
        total_loss_e, n_toks_e = run_eval(model.module if use_ddp else model, eval_loader, alphabet, device)
        eval_nll_0 = total_loss_e / max(n_toks_e, 1)
        eval_ppl_0 = min(math.exp(eval_nll_0), 1e10)  # cap PPL for display, avoid inf
        print(f"[eval] step 0 (before training): eval_nll={eval_nll_0:.4f} eval_ppl={eval_ppl_0:.4f}")
        if wandb is not None:
            wandb.log({"eval_nll": eval_nll_0, "eval_ppl": eval_ppl_0, "batch": 0})

    global_step = 0
    for epoch in range(cfg.max_epochs):
        if rank == 0:
            print(f"--- Epoch {epoch + 1}/{cfg.max_epochs} ---")
        for batch_idx, (seq_labels, strs, toks) in enumerate(data_loader):
            global_step += 1
            if rank == 0 and batch_idx % 10 == 0:
                print(
                    f"Processing epoch {epoch + 1} batch {batch_idx + 1}/{batches_per_epoch} (step {global_step}, {toks.size(0)} sequences)"
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
            # 单卡或 DDP debug：第一步后打印未参与梯度的参数
            if global_step == 1 and rank == 0 and (not use_ddp or ddp_debug_unused):
                log_unused_parameters(model, global_step, rank)
            if ddp_debug_unused and global_step == 1:
                print("[DDP debug] Done. Exit after one step.")
                if dist.is_initialized():
                    dist.destroy_process_group()
                sys.exit(0)
            optimizer.step()
            if rank == 0:
                train_nll = loss.item()
                train_ppl = min(math.exp(train_nll), 1e10)  # cap PPL for display, avoid inf
                print(f"step {global_step} (epoch {epoch + 1} batch {batch_idx + 1}): train_nll={train_nll:.4f} train_ppl={train_ppl:.4f}")
                if wandb is not None:
                    wandb.log({"train_nll": train_nll, "train_ppl": train_ppl, "batch": global_step})
            # 到 eval 步：多卡时所有 rank 一起跑 eval（数据按 rank 切分），再 all_reduce 得全局 eval_loss
            if global_step % cfg.eval_every == 0 and eval_loader is not None:
                _eval_model = model.module if use_ddp else model
                total_loss_e, n_toks_e = run_eval(_eval_model, eval_loader, alphabet, device)
                if use_ddp:
                    total_loss_t = torch.tensor([total_loss_e], device=device, dtype=torch.float64)
                    n_toks_t = torch.tensor([n_toks_e], device=device, dtype=torch.int64)
                    dist.all_reduce(total_loss_t, op=dist.ReduceOp.SUM)
                    dist.all_reduce(n_toks_t, op=dist.ReduceOp.SUM)
                    eval_loss = (total_loss_t.item() / max(n_toks_t.item(), 1))
                else:
                    eval_loss = total_loss_e / max(n_toks_e, 1)
                if rank == 0:
                    eval_nll = eval_loss
                    eval_ppl = min(math.exp(eval_nll), 1e10)  # cap PPL for display, avoid inf
                    print(f"[eval] step {global_step}: eval_nll={eval_nll:.4f} eval_ppl={eval_ppl:.4f}")
                    if wandb is not None:
                        wandb.log({"eval_nll": eval_nll, "eval_ppl": eval_ppl, "batch": global_step})


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
