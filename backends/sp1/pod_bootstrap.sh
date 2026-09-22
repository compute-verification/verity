#!/usr/bin/env bash
# Bootstrap a fresh RunPod GPU pod for the SP1 backend.  Idempotent; safe to re-run.
#
# Installs build deps, rustup stable, sp1up v6.4.0 (cargo-prove + the `succinct` guest toolchain), uv + Python 3.12;
# builds the host with GPU proving (`--features cuda`, run with SP1_PROVER=cuda) from the checkout this script is in,
# into a persistent target dir, prints `info` (a sound build must reproduce the approved identity of
# python/verity_sp1/host.py, else the script fails and installs nothing), and installs it as /workspace/bin/veritor-zk-host-cuda.
# Groth16/PLONK wrapping (native-gnark, needs Go) is not built: the campaign starts with core/compressed.
#
# Run it through research so the log is a recorded run:
#     uv run research run --on vy-sp1 --project verity --source . --stage sp1.bootstrap -- bash backends/sp1/pod_bootstrap.sh
# Workloads then point the driver at the binary with VERITY_SP1_HOST=/workspace/bin/veritor-zk-host-cuda.
#
# Adapted from veritor's zk/campaign pod_bootstrap.sh (tools/research/src/research/pods/sh/); no credentials are placed
# on the pod and the repository is shipped as a snapshot, never cloned.
#
# SP1_HOST_FEATURES (default `cuda`) selects the host's cargo features; a DIAGNOSTIC build is e.g.
# SP1_HOST_FEATURES=cuda,unsound-skip-authentication, installed as veritor-zk-host-cuda-unsound-skip-authentication
# from its own target dir (likewise cuda,unsound-relation-only -> veritor-zk-host-cuda-unsound-relation-only).
# Such a guest is never the approved prover; vector_run.py needs --diagnostic (and --relation-only) to use it.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
export PATH="$HOME/.sp1/bin:$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"          # backends/sp1 of the shipped checkout
WORK=/workspace
BIN=$WORK/bin
FEATURES="${SP1_HOST_FEATURES:-cuda}"
SUFFIX="$(echo "$FEATURES" | tr ',' '-')"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$WORK/sp1-target$([ "$FEATURES" = cuda ] || echo "-$SUFFIX")}"
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

stage "build host: $FEATURES  (source: $HERE, target: $CARGO_TARGET_DIR)"
cd "$HERE"
cargo build --release -p veritor-zk-host --features "$FEATURES" 2>&1 | grep -v -E "^\s+(Compiling|Downloaded|Downloading) "
BUILT="$CARGO_TARGET_DIR/release/veritor-zk-host"

stage "guest identity"
INFO="$("$BUILT" info | grep '^{' | tail -1)"
echo "$INFO"
if [ "$FEATURES" = cuda ]; then
  # The sound build must reproduce the approved identity (python/verity_sp1/host.py APPROVED): checked
  # BEFORE install, so a source change that moved the guest never replaces the approved binary.
  APPROVED_ELF="$(grep -oE 'elf_sha256="[0-9a-f]{64}"' "$HERE/python/verity_sp1/host.py" | cut -d'"' -f2)"
  APPROVED_VK="$(grep -oE 'vk_hash="0x[0-9a-f]{64}"' "$HERE/python/verity_sp1/host.py" | cut -d'"' -f2)"
  GOT_ELF="$(echo "$INFO" | jq -r .elf_sha256)"
  GOT_VK="$(echo "$INFO" | jq -r .vk_hash)"
  if [ "$GOT_ELF" != "$APPROVED_ELF" ] || [ "$GOT_VK" != "$APPROVED_VK" ]; then
    echo "SOUND GUEST IDENTITY CHANGED: elf $GOT_ELF vk $GOT_VK (approved: elf $APPROVED_ELF vk $APPROVED_VK); not installed" >&2
    exit 1
  fi
  echo "approved identity reproduced: elf $GOT_ELF vk $GOT_VK"
fi

stage "install"
install -m 0755 "$BUILT" "$BIN/veritor-zk-host-$SUFFIX"
sha256sum "$BIN/veritor-zk-host-$SUFFIX"

stage "done"
echo "VERITY_SP1_HOST=$BIN/veritor-zk-host-$SUFFIX"
