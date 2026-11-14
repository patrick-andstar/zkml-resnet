#!/bin/bash
# P3：用 ai-edge-litert 替代 tensorflow 跑仓库原始 converter.py（不改仓库脚本）
set -u
REPRO=/root/autodl-tmp/zkml-repro
PY=$REPRO/env/py-export/bin/python
SHIM=$REPRO/env/shim
mkdir -p "$SHIM" "$REPRO/logs/p3"

$PY -m pip install -q msgpack -i http://mirrors.aliyun.com/pypi/simple \
    --trusted-host mirrors.aliyun.com 2>&1 | tail -3

cat > "$SHIM/tensorflow.py" <<'PYEOF'
"""P3 兼容层：仓库 python/converter.py 顶部 `import tensorflow as tf`，
本机只装了 ai-edge-litert（无 TensorFlow）。converter.py 实际只用到
tf.lite.Interpreter 的 allocate_tensors / get_input_details / get_output_details /
set_tensor / invoke / get_tensor，且 ai-edge-litert 的 Interpreter 支持
experimental_preserve_all_tensors 参数，因此这里只做符号转发，不改任何逻辑。"""
from ai_edge_litert.interpreter import Interpreter


class _Lite:
    Interpreter = Interpreter


lite = _Lite()
__version__ = "shim(ai-edge-litert)"
PYEOF

echo "=== shim 就绪 ==="
PYTHONPATH="$SHIM" $PY -c "import tensorflow as tf; print('tf shim ok, Interpreter =', tf.lite.Interpreter)"

echo
echo "=== 原样运行 converter.py（默认参数，仅烟测） ==="
cd "$REPRO/src" || exit 1
PYTHONPATH="$SHIM" timeout 900 $PY python/converter.py \
    --model "$REPRO/assets/p3-model.tflite" \
    --model_output "$REPRO/logs/p3/smoke-model.msgpack" \
    --config_output "$REPRO/logs/p3/smoke-config.msgpack" \
    > "$REPRO/logs/p3/converter-smoke.log" 2>&1
RC=$?
echo "exit=$RC"
echo "--- 尾部 ---"
tail -24 "$REPRO/logs/p3/converter-smoke.log"
echo "--- 产物 ---"
ls -la "$REPRO/logs/p3/" 2>/dev/null
exit 0
