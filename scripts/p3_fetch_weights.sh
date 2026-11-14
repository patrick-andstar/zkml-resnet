#!/usr/bin/env bash
# Fetch the torchvision ResNet-18 checkpoint with resume support.
# torchvision's own downloader restarts from zero on interruption, which is
# what happened the first time; curl -C - resumes instead.
set -uo pipefail
REPRO=/root/autodl-tmp/zkml-repro
DIR=$REPRO/assets/torch-hub/hub/checkpoints
DEST=$DIR/resnet18-f37072fd.pth
URL=https://download.pytorch.org/models/resnet18-f37072fd.pth

mkdir -p "$DIR"
rm -f "$DEST"*.partial

echo "start $(date -u +%H:%M:%S)"
curl -sSL --retry 8 --retry-delay 2 --retry-all-errors -C - -o "$DEST" "$URL"
rc=$?
echo "curl_rc=$rc $(date -u +%H:%M:%S)"
ls -la "$DEST" 2>/dev/null
sha256sum "$DEST" 2>/dev/null
exit $rc
