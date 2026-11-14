#!/usr/bin/env bash
# P3 step 2: install Google AI Edge LiteRT Torch (the current official
# PyTorch -> TFLite route) plus the flatbuffer `tflite` package, which is what
# the repo's python/converter.py uses to enumerate operators.
#
# torch/torchvision are pinned via a constraints file so a transitive
# requirement cannot silently swap the CPU build for a CUDA one or a version
# outside the range the plan allows.
set -uo pipefail

REPRO=/root/autodl-tmp/zkml-repro
VENV=$REPRO/env/py-export
export PIP_CACHE_DIR=$REPRO/env/pip-cache
export PYTHONNOUSERSITE=1
PIP="$VENV/bin/pip"
PY="$VENV/bin/python"

cat > /tmp/p3-constraints.txt <<'EOF'
torch==2.14.1
torchvision==0.29.1
EOF

COMMON=(
  --timeout 60
  --retries 8
  --resume-retries 20
  --index-url http://mirrors.aliyun.com/pypi/simple
  --trusted-host mirrors.aliyun.com
  -c /tmp/p3-constraints.txt
)

echo "=== install litert-torch + tflite ==="
time $PIP install "${COMMON[@]}" 'litert-torch' 'tflite' 2>&1 | tail -18

echo "=== import check ==="
$PY - <<'PYEOF'
import importlib, sys
for m in ("torch", "torchvision", "litert_torch", "ai_edge_litert", "tflite", "numpy"):
    try:
        mod = importlib.import_module(m)
        print("OK  ", m, getattr(mod, "__version__", "?"))
    except Exception as e:
        print("FAIL", m, type(e).__name__, e)
PYEOF

echo "=== which packages are installed ==="
$PIP list 2>/dev/null | grep -Ei "torch|litert|tensorflow|jax|numpy|tflite" || true
df -h /root/autodl-tmp | tail -1
