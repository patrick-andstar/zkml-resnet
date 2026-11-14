#!/usr/bin/env bash
# Build the P2A measurement binary into $CARGO_TARGET_DIR (/root/autodl-tmp/zkml-repro/target).
set -uo pipefail
source /root/autodl-tmp/zkml-repro/scripts/env.sh
echo "== toolchain =="
rustc -V
cargo -V
echo "CARGO_TARGET_DIR=$CARGO_TARGET_DIR"
echo "== build verify_profile2 =="
cargo build --release --bin verify_profile2 2>&1 | tail -30
rc=${PIPESTATUS[0]}
echo "cargo_rc=$rc"
ls -la "$CARGO_TARGET_DIR/release/verify_profile2" 2>&1
exit $rc
