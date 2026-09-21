#!/usr/bin/env bash
# Bootstrap a fresh RunPod GPU pod for the SP1 backend.  Idempotent; safe to re-run.
#
# Installs build deps, rustup stable, sp1up v6.4.0 (cargo-prove + the `succinct` guest toolchain), uv + Python 3.12;
# builds the host with GPU proving (`--features cuda`, run with SP1_PROVER=cuda) from the checkout this script is in,
# into a persistent target dir, and installs it as /workspace/bin/veritor-zk-host-cuda; then prints `info`.
# Groth16/PLONK wrapping (native-gnark, needs Go) is not built: the campaign starts with core/compressed.
#
# Run it through research so the log is a recorded run:
#     uv run research run --on vy-sp1 --project verity --source . --stage sp1.bootstrap -- bash backends/sp1/pod_bootstrap.sh
# Workloads then point the driver at the binary with VERITY_SP1_HOST=/workspace/bin/veritor-zk-host-cuda.
#
# Adapted from veritor's zk/campaign pod_bootstrap.sh (tools/research/src/research/pods/sh/); no credentials are placed
# on the pod and the repository is shipped as a snapshot, never cloned.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
export PATH="$HOME/.sp1/bin:$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"          # backends/sp1 of the shipped checkout
WORK=/workspace
BIN=$WORK/bin
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$WORK/sp1-target}"
mkdir -p "$BIN" "$CARGO_TARGET_DIR"

stage() { echo; echo "=== [$(date -u +%H:%M:%S)] $*"; }

stage "machine"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
nproc; free -g | head -2; df -h "$WORK" | tail -1

stage "apt"
apt-get update -qq
apt-get install -y -qq --no-install-recommends \
  build-essential pkg-config libssl-dev clang libclang-dev cmake protobuf-compiler libprotobuf-dev \
  git curl ca-certificates jq >/dev/null

stage "rustup stable"
if ! command -v rustup >/dev/null; then
  curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable >/dev/null
fi
rustup toolchain install stable --profile minimal >/dev/null
rustc --version

stage "sp1up v6.4.0"
if ! cargo prove --version 2>/dev/null | grep -q "6.4.0"; then
  curl -fsSL https://sp1up.succinct.xyz | bash >/dev/null
  "$HOME/.sp1/bin/sp1up" --version v6.4.0
fi
cargo prove --version
rustup toolchain list | grep succinct

stage "uv + python 3.12"
if ! command -v uv >/dev/null; then
  curl -fsSL https://astral.sh/uv/install.sh | sh >/dev/null
fi
uv python install 3.12 >/dev/null
uv python find 3.12

stage "build host: cuda  (source: $HERE, target: $CARGO_TARGET_DIR)"
cd "$HERE"
cargo build --release -p veritor-zk-host --features cuda 2>&1 | tail -3
install -m 0755 "$CARGO_TARGET_DIR/release/veritor-zk-host" "$BIN/veritor-zk-host-cuda"
sha256sum "$BIN/veritor-zk-host-cuda"

stage "guest identity"
"$BIN/veritor-zk-host-cuda" info

stage "done"
echo "VERITY_SP1_HOST=$BIN/veritor-zk-host-cuda"
