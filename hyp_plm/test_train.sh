# export CUDA_VISIBLE_DEVICES=6

CONFIG=configs/base.yaml
# 从 config 里读 n_gpus：1=单卡，>1 则用 torchrun 多卡
N_GPUS=$(python -c "
import yaml
from pathlib import Path
p = Path('$CONFIG')
c = yaml.safe_load(p.read_text()) if p.exists() else {}
t = c.get('train') or c
print(int(t.get('n_gpus', 1)))
")

if [ "$N_GPUS" -gt 1 ]; then
  torchrun --nproc_per_node="$N_GPUS" scripts/test_train.py --config "$CONFIG"
else
  python scripts/test_train.py --config "$CONFIG"
fi