#!/usr/bin/env bash
# SP1 with the core FRI query count raised to the value SP1's own formula gives for a $SEC_BITS-bit target (default 134:
# 175 queries; stock SP1_TARGET_BITS_OF_SECURITY 100: 124), otherwise unmodified, on a Linux CUDA pod.  The count is
# compiled into sp1-primitives (fri_params.rs core_fri_config: unique_decoding_queries(CORE_LOG_BLOWUP)) and the prebuilt
# sp1-gpu-server bakes it in.  Only core_fri_config changes: raising SP1_TARGET_BITS_OF_SECURITY itself also grows the
# recursion programs that every prover builds at start-up (compose/deferred, fixed reduce shape), which then panic with
# "fixed height is too small"; core proofs never use recursion.  Both sides are rebuilt with that one line changed:
#   - prover: sp1-gpu-server from succinctlabs/sp1 v6.6.0 (f5a5bbf6, the version of the release server the stock cells
#     ran), installed as $ROOT/home/.sp1/bin/sp1-gpu-server; run the host with HOME=$ROOT/home.  sp1-sdk's cuda client
#     replaces a server whose `--version` differs from sp1_primitives::SP1_CRATE_VERSION by the release download, so the
#     patched crate keeps version 6.6.0 and every run must check the server's sha256 afterwards.
#   - verifier/host: backends/sp1 unmodified (--locked), built with CARGO_HOME=$ROOT/cargo-home, a copy of the registry
#     whose crates.io sp1-primitives 6.6.0 has the same line changed; host features cuda,relation-bare, installed as
#     /workspace/bin/veritor-zk-host-cuda-relation-bare-sec$SEC_BITS.  Its embedded guest (ELF, vk) equals stock's.
# Needs the stock toolchain first (backends/sp1/pod_bootstrap.sh with SP1_HOST_FEATURES=cuda,relation-bare, which also
# fills ~/.cargo/registry with sp1-primitives 6.6.0).  Keep builds sequential (kb/sp1-prover.md).
#   CUDA_ARCHS=80 bash backends/sp1/sec128/build.sh          # 80 A100, 90 H100, 89 RTX 4090, 120 RTX 5090
set -euo pipefail
export PATH="$HOME/.sp1/bin:$HOME/.cargo/bin:/usr/local/cuda/bin:$HOME/.local/bin:$PATH"
export CUDACXX="${CUDACXX:-/usr/local/cuda/bin/nvcc}"
export CUDA_ARCHS="${CUDA_ARCHS:-80}"
SEC_BITS="${SEC_BITS:-134}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SP1_SRC="$(dirname "$HERE")"
ROOT="${SEC128_ROOT:-/workspace/sec128}"
TAG_COMMIT=f5a5bbf6a1cf6007315b410d88678d96fb399fb5   # succinctlabs/sp1 v6.6.0
QUERIES="$(python3 -c "import math; print(math.ceil(($SEC_BITS - 16) / -math.log2(0.5 + 0.25 / 2)))")"
STOCK_LINE='        unique_decoding_queries(CORE_LOG_BLOWUP),'
NEW_LINE="        $QUERIES, // verity sec128: unique_decoding_queries at a $SEC_BITS-bit target (stock: unique_decoding_queries(CORE_LOG_BLOWUP) = 124)"
mkdir -p "$ROOT" /workspace/bin
stage() { echo; echo "=== [$(date -u +%H:%M:%S)] $*"; }

patch_line() {  # FILE: replace the stock line (it occurs once, in core_fri_config), and fail unless exactly that line changed
  local f=$1
  cp "$f" "$f.orig"
  [ "$(grep -cxF "$STOCK_LINE" "$f.orig")" = 1 ] || { echo "stock line not found once in $f"; exit 2; }
  python3 - "$f" "$STOCK_LINE" "$NEW_LINE" <<'EOF'
import sys
p, old, new = sys.argv[1:]
s = open(p).read()
open(p, "w").write(s.replace(old + "\n", new + "\n", 1))
EOF
  [ "$(diff "$f.orig" "$f" | grep -c '^[<>]')" = 2 ] || { echo "unexpected diff in $f"; diff "$f.orig" "$f"; exit 2; }
  rm "$f.orig"
}

stage "machine"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
nproc; df -h /workspace | tail -1
command -v nvcc >/dev/null && nvcc --version | tail -1
# sp1-recursion-gnark-ffi (a dependency of the server) builds Go code whose go.mod needs go >= 1.24; apt's is older.
if ! /usr/local/go/bin/go version 2>/dev/null | grep -qE 'go1\.(2[4-9]|[3-9][0-9])'; then
  curl -fsSL https://go.dev/dl/go1.24.7.linux-amd64.tar.gz | tar -C /usr/local -xzf -
fi
export PATH="/usr/local/go/bin:$PATH"
go version
# sp1-gpu-sys (v6.6.0) needs CMake >= 3.24; Ubuntu 22.04 images ship 3.22.
if ! /usr/local/cmake/bin/cmake --version >/dev/null 2>&1; then
  mkdir -p /usr/local/cmake
  curl -fsSL https://github.com/Kitware/CMake/releases/download/v3.30.5/cmake-3.30.5-linux-x86_64.tar.gz \
    | tar -C /usr/local/cmake --strip-components=1 -xzf -
fi
export PATH="/usr/local/cmake/bin:$PATH"
cmake --version | head -1

stage "prover source: sp1 v6.6.0 $TAG_COMMIT + core_fri_config $QUERIES queries"
FORK="$ROOT/sp1-v6.6.0-coreq$QUERIES"
if [ ! -f "$FORK/.sec-done" ]; then
  rm -rf "$FORK"; git init -q "$FORK"
  git -C "$FORK" remote add origin https://github.com/succinctlabs/sp1.git
  git -C "$FORK" fetch -q --depth 1 origin "$TAG_COMMIT"
  git -C "$FORK" checkout -q FETCH_HEAD
  patch_line "$FORK/crates/primitives/src/fri_params.rs"
  touch "$FORK/.sec-done"
fi
git -C "$FORK" rev-parse HEAD
git -C "$FORK" diff --stat
git -C "$FORK" diff | tee "$ROOT/prover-source.diff"

stage "sp1-gpu-server (CUDA_ARCHS=$CUDA_ARCHS) -> $ROOT/home/.sp1/bin"
mkdir -p "$ROOT/home/.sp1/bin"
( cd "$FORK" && CARGO_TARGET_DIR="$ROOT/target-server" cargo install --locked --force --root "$ROOT/home/.sp1" \
    --path sp1-gpu/crates/server 2>&1 | grep -v -E "^\s+(Compiling|Downloaded|Downloading) " )
SERVER="$ROOT/home/.sp1/bin/sp1-gpu-server"
[ "$("$SERVER" --version | tr -d '[:space:]')" = 6.6.0 ] || { echo "server version is not 6.6.0"; exit 2; }
sha256sum "$SERVER" | tee "$ROOT/server.sha256"

stage "verifier/host: backends/sp1 unmodified, CARGO_HOME=$ROOT/cargo-home with sp1-primitives 6.6.0 patched in place"
# The guest ELF carries its registry source paths remapped from <cargo home> to /cargo (host/build.rs), so patching the
# crate in a copied cargo home (no [patch], lock file unchanged) keeps the embedded guest byte-identical to stock.
REG_REL="registry/src/index.crates.io-1949cf8c6b5b557f/sp1-primitives-6.6.0"
[ -d "$HOME/.cargo/$REG_REL" ] || { echo "sp1-primitives 6.6.0 not in the cargo registry: run pod_bootstrap.sh first"; exit 2; }
CH="$ROOT/cargo-home"
rm -rf "$CH"; mkdir -p "$CH"
cp -a "$HOME/.cargo/registry" "$CH/"
[ -d "$HOME/.cargo/git" ] && cp -a "$HOME/.cargo/git" "$CH/"
for f in config.toml config; do [ -f "$HOME/.cargo/$f" ] && cp -a "$HOME/.cargo/$f" "$CH/"; done
patch_line "$CH/$REG_REL/src/fri_params.rs"
cmp "$CH/$REG_REL/src/fri_params.rs" "$FORK/crates/primitives/src/fri_params.rs" \
  || { echo "patched registry fri_params.rs differs from the prover's"; exit 2; }
diff -r "$HOME/.cargo/$REG_REL" "$CH/$REG_REL" | tee "$ROOT/verifier-source.diff" || true
HOST="$ROOT/host-src"
rm -rf "$HOST"; mkdir -p "$HOST"
tar -C "$SP1_SRC" --exclude=./target --exclude=./tcdot --exclude=./sec128 --exclude=./sp1 -cf - . | tar -C "$HOST" -xf -

stage "host: cuda,relation-bare"
cd "$HOST"; rm -rf "$ROOT/target-host"   # a fresh guest build under the new cargo home
CARGO_HOME="$CH" CARGO_TARGET_DIR="$ROOT/target-host" cargo build --release --locked -p veritor-zk-host \
  --features cuda,relation-bare 2>&1 | grep -v -E "^\s+(Compiling|Downloaded|Downloading) "
BIN=/workspace/bin/veritor-zk-host-cuda-relation-bare-sec$SEC_BITS
install -m 0755 "$ROOT/target-host/release/veritor-zk-host" "$BIN"
sha256sum "$BIN"
SP1_PROVER=cpu RUST_LOG=error "$BIN" info > "$ROOT/host-info.log" 2>&1 || { tail -20 "$ROOT/host-info.log"; exit 3; }
grep '^{' "$ROOT/host-info.log" | tail -1 | tee "$ROOT/host-info.json"
if [ -x /workspace/bin/veritor-zk-host-cuda-relation-bare ]; then
  echo "stock host:"; SP1_PROVER=cpu RUST_LOG=error /workspace/bin/veritor-zk-host-cuda-relation-bare info | grep '^{' | tail -1
fi
stage "done"
echo "VERITY_SP1_HOST=$BIN HOME=$ROOT/home (server sha256 $(cut -c1-16 "$ROOT/server.sha256"))"
