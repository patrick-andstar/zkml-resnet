#!/usr/bin/env bash
# Source this to get the isolated Rust toolchain used by the reproduction.
#
# We must stay inside $REPRO/src when running cargo: that is where
# rust-toolchain.toml lives, so rustup picks the pinned nightly from it.
# The build artifacts are redirected to $REPRO/target (one place only).
export REPRO=/root/autodl-tmp/zkml-repro
export RUSTUP_HOME=$REPRO/env/rustup
export CARGO_HOME=$REPRO/env/cargo
export CARGO_TARGET_DIR=$REPRO/target
export PATH="$CARGO_HOME/bin:$PATH"
export RUST_BACKTRACE=1
cd $REPRO/src || return 1
