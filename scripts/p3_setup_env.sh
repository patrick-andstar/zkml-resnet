#!/usr/bin/env bash
# P3 step 1: isolated export environment (torch/torchvision CPU + LiteRT Torch).
# Does not touch the conda base env or any /root/zk directory.
#
# Note on indexes: download.pytorch.org is reachable here, but it redirects the
# pure-python dependencies to files.pythonhosted.org, which is unreliable from
# this host (a 2 MB networkx wheel failed after 6 attempts). So the +cpu wheels
# come from download.pytorch.org while everything else comes from the aliyun
# mirror that is already configured for this machine.
set -uo pipefail

REPRO=/root/autodl-tmp/zkml-repro
VENV=$REPRO/env/py-export
export PIP_CACHE_DIR=$REPRO/env/pip-cache
export PYTHONNOUSERSITE=1
PIP="$VENV/bin/pip"
PY="$VENV/bin/python"

COMMON=(
  --timeout 60
  --retries 8
  --resume-retries 20
  --index-url https://download.pytorch.org/whl/cpu
  --extra-index-url http://mirrors.aliyun.com/pypi/simple
  --trusted-host mirrors.aliyun.com
)

echo "=== env python ==="
$PY -V

echo "=== install torch/torchvision (CPU wheels) ==="
time $PIP install "${COMMON[@]}" 'torch==2.14.1+cpu' 'torchvision==0.29.1+cpu' 2>&1 | tail -12

echo "=== verify ==="
$PY - <<'PYEOF'
import torch, torchvision
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("cuda_available", torch.cuda.is_available())
PYEOF

df -h /root/autodl-tmp | tail -1
