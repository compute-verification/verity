#!/usr/bin/env bash
# Build the modified SP1 (TC_DOT chip) on a Linux CUDA pod: upstream succinctlabs/sp1 at v6.4.0 (f66b4bff5) plus the
# patch series in sp1-patches/, applied with fixed committer identities and the author dates so the commit hashes
# reproduce: 0001-0003 = ~/projects/sp1 branch tc-dot-precompile @ 3510b39a7 (the fp8 TC_DOT chip), 0004-0005 = lane
# sp1-tcdot's TC_DOT_BF16 chip, 0006 = its prover sharding change (FORK_HEAD, tree FORK_TREE).  OPERANDS=witness adds
# sp1-patches/witness-operands/0007-0011, the witness-operands arm (FORK_HEAD_WIT, tree FORK_TREE_WIT): TC_DOT_BF16
# takes its operands as free witness values from the input stream instead of reading them through SP1's memory
# argument, which is sound only for a statement existential in x and W (relation-bare is); 0009 is prover-only (a
# precompile shard's calls on one accumulator word share one local memory event); 0010 constrains subnormal GroupSum
# outputs, so only overflowing and non-finite steps stay outside the chip's contract; 0011 is prover-only (the chip's
# trace and byte lookups generated in parallel).  The checkout is linked
# at ./sp1, where Cargo.toml's [patch] points.  Then the forked sp1-gpu-server (the stock prebuilt server has neither
# TC_DOT chip), installed under $SERVER_ROOT/bin; sp1-sdk's cuda client runs $HOME/.sp1/bin/sp1-gpu-server.  Build
# the host with VERITY_TCDOT_FORK_HEAD set to the arm's pin, and for the witness arm with --features stream-operands.
# Idempotent.
#
#   bash backends/sp1/tcdot/build_fork.sh                        # fork + GPU server into ~/.sp1/bin
#   SERVER_ROOT=/workspace/tcdot-root bash backends/sp1/tcdot/build_fork.sh   # keep a stock server in ~/.sp1/bin
#   OPERANDS=witness bash backends/sp1/tcdot/build_fork.sh       # the witness-operands arm
#   SKIP_SERVER=1 bash backends/sp1/tcdot/build_fork.sh
#
# Keep the builds sequential: two concurrent builds race on the nested sp1-core-executor-runner build and leave an
# empty runner stub that breaks every CUDA proof ("failed sending input to child: Broken pipe").
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
export PATH="$HOME/.sp1/bin:$HOME/.cargo/bin:/usr/local/cuda/bin:/usr/local/go/bin:$HOME/.local/bin:$PATH"
export CUDACXX="${CUDACXX:-/usr/local/cuda/bin/nvcc}"
# sp1-gpu-sys compiles its kernels for these SMs only (default: all of 80 86 89 90 [100 120]); 80 = A100, 89 = RTX 4090.
export CUDA_ARCHS="${CUDA_ARCHS:-80}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${SP1_TCDOT_ROOT:-/workspace/sp1-tcdot}"
FORK="$ROOT/sp1-fork"
SERVER_ROOT="${SERVER_ROOT:-$HOME/.sp1}"
BASE=f66b4bff51d0ccff51d152e0f7f66b2ffedf3529
case "${OPERANDS:-memory}" in
  memory) PIN=""; EXTRA=() ;;
  witness) PIN=_WIT; EXTRA=("$HERE"/sp1-patches/witness-operands/*.patch) ;;
  *) echo "OPERANDS must be memory or witness"; exit 2 ;;
esac
EXPECT_HEAD="$(cat "$HERE/FORK_HEAD$PIN")"
EXPECT_TREE="$(cat "$HERE/FORK_TREE$PIN")"
mkdir -p "$ROOT"

stage() { echo; echo "=== [$(date -u +%H:%M:%S)] $*"; }

stage "machine"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
nproc; free -g | head -2; df -h /workspace | tail -1

stage "apt"
apt-get update -qq
apt-get install -y -qq --no-install-recommends \
  build-essential pkg-config libssl-dev clang libclang-dev cmake protobuf-compiler libprotobuf-dev \
  git curl ca-certificates jq golang-go time >/dev/null
go version

stage "rustup + sp1up v6.4.0"
if ! command -v rustup >/dev/null; then
  curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable >/dev/null
fi
if ! cargo prove --version 2>/dev/null | grep -q "f66b4bf"; then
  curl -fsSL https://sp1up.succinct.xyz | bash >/dev/null
  "$HOME/.sp1/bin/sp1up" --version v6.4.0
fi
cargo prove --version

stage "fork: upstream $BASE + $(ls "$HERE"/sp1-patches/*.patch "${EXTRA[@]}" | wc -l) patches"
if [ "$(git -C "$FORK" rev-parse HEAD 2>/dev/null || true)" != "$EXPECT_HEAD" ]; then
  rm -rf "$FORK"
  git init -q "$FORK"
  git -C "$FORK" remote add origin https://github.com/succinctlabs/sp1.git
  git -C "$FORK" fetch -q --depth 1 origin "$BASE"
  git -C "$FORK" checkout -q FETCH_HEAD
  GIT_COMMITTER_NAME="Daniel Reuter" GIT_COMMITTER_EMAIL="62895916+danielreuter@users.noreply.github.com" \
    git -C "$FORK" am -q --committer-date-is-author-date "$HERE"/sp1-patches/000[1-3]-*.patch
  GIT_COMMITTER_NAME="verity sp1-tcdot lane" GIT_COMMITTER_EMAIL="lane@verity.local" \
    git -C "$FORK" am -q --committer-date-is-author-date "$HERE"/sp1-patches/000[4-9]-*.patch "${EXTRA[@]}"
fi
GOT="$(git -C "$FORK" rev-parse HEAD)"
TREE="$(git -C "$FORK" rev-parse 'HEAD^{tree}')"
echo "fork HEAD $GOT tree $TREE (expected $EXPECT_HEAD / $EXPECT_TREE)"
[ "$TREE" = "$EXPECT_TREE" ] || { echo "FORK TREE MISMATCH"; exit 2; }
git -C "$FORK" log --oneline -6
ln -sfn "$FORK" "$HERE/sp1"

if [ -z "${SKIP_SERVER:-}" ]; then
  stage "forked sp1-gpu-server -> $SERVER_ROOT/bin"
  cd "$FORK"
  CARGO_TARGET_DIR="$ROOT/target-server" cargo install --locked --force --root "$SERVER_ROOT" --path sp1-gpu/crates/server 2>&1 \
    | grep -v -E "^\s+(Compiling|Downloaded|Downloading) "
  "$SERVER_ROOT/bin/sp1-gpu-server" --version
  sha256sum "$SERVER_ROOT/bin/sp1-gpu-server"
  echo "$GOT" > "$ROOT/server-fork-head"
fi
stage "done"
