#!/usr/bin/env python3 -u
"""
对 test 数据集做 MLM 评测。训练只用 train + eval，本脚本仅读取 test.fasta 并汇报 loss/perplexity。
用法:
  python hyp_plm/scripts/evaluate_test.py --config hyp_plm/configs/base.yaml [--checkpoint path.pt] [--test_fasta path]
"""
import argparse
import pathlib
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from tqdm import tqdm

from esm import Alphabet, FastaBatchedDataset, ESM2, MSATransformer
from esm.model.hyp_esm2 import LorentzESM2


def run_mlm_eval(model, data_loader, alphabet, device):
    """在 data_loader 上跑 MLM 评测，返回平均 loss。"""
    model.eval()
    total_loss = 0.0
    n_toks = 0
    with torch.no_grad():
        for seq_labels, strs, toks in tqdm(data_loader, desc="test eval"):
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
            random = (
                torch.bernoulli(torch.full(mlm_labels.shape, 0.5, device=toks.device)).bool() & masked & ~replaced
            )
            toks_masked[random] = rand_ids[random]
            out = model(toks_masked)
            logits = out["logits"]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), mlm_labels.reshape(-1), ignore_index=-100, reduction="sum"
            )
            n = (mlm_labels.reshape(-1) != -100).sum().item()
            total_loss += loss.item()
            n_toks += n
    return total_loss / max(n_toks, 1)


def load_config(config_path):
    """读取 YAML，返回评测用配置。路径相对 hyp_plm（configs 的父目录）。"""
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

    return SimpleNamespace(
        model=model_type,
        test_fasta=p(t.get("test_fasta")),
        toks_per_batch=int(t.get("toks_per_batch", 4096)),
        truncation_seq_length=int(t.get("truncation_seq_length", 1022)),
        nogpu=bool(t.get("nogpu", False)),
    )


def main():
    parser = argparse.ArgumentParser(description="在 test 数据集上评测 MLM loss。")
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        required=True,
        help="训练用 YAML 配置（用于模型类型与 test_fasta 等）。",
    )
    parser.add_argument(
        "--checkpoint",
        type=pathlib.Path,
        default=None,
        help="模型权重 .pt 文件；不传则使用随机初始化（仅做流程测试）。",
    )
    parser.add_argument(
        "--test_fasta",
        type=pathlib.Path,
        default=None,
        help="覆盖 config 中的 test_fasta。",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    test_fasta = pathlib.Path(args.test_fasta).resolve() if args.test_fasta else cfg.test_fasta
    if not test_fasta or not test_fasta.exists():
        raise FileNotFoundError(f"test_fasta 不存在: {test_fasta}")

    device = torch.device("cuda" if (torch.cuda.is_available() and not cfg.nogpu) else "cpu")
    alphabet = Alphabet.from_architecture("ESM-1b")

    if cfg.model == "esm2":
        model = ESM2(alphabet=alphabet)
    else:
        model = LorentzESM2(alphabet=alphabet)

    if isinstance(model, MSATransformer):
        raise ValueError("本脚本不支持 MSA Transformer。")

    if args.checkpoint is not None:
        ckpt = pathlib.Path(args.checkpoint).resolve()
        if not ckpt.exists():
            raise FileNotFoundError(f"checkpoint 不存在: {ckpt}")
        state = torch.load(ckpt, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=False)
        print(f"已加载权重: {ckpt}")

    model = model.to(device)

    dataset = FastaBatchedDataset.from_file(test_fasta)
    batches = dataset.get_batch_indices(cfg.toks_per_batch, extra_toks_per_seq=1)
    loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=alphabet.get_batch_converter(cfg.truncation_seq_length),
        batch_sampler=batches,
    )
    print(f"Test 集: {test_fasta}, 序列数: {len(dataset)}, batch 数: {len(batches)}")

    test_loss = run_mlm_eval(model, loader, alphabet, device)
    perplexity = torch.exp(torch.tensor(test_loss)).item()
    print(f"[test] loss={test_loss:.4f}  perplexity={perplexity:.4f}")


if __name__ == "__main__":
    main()
