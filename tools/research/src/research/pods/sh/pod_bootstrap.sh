#!/usr/bin/env bash
# Bootstrap a fresh RunPod pod for the SP1 measurement campaign.
#
# Idempotent; safe to re-run.  Installs: build deps, Go 1.24 (gnark FFI), rustup
# stable, sp1up v6.4.0 (cargo-prove + the `succinct` guest toolchain), uv +
# Python 3.12.  Builds the host twice from /workspace/veritor/zk/sp1:
#   bin/veritor-zk-host-cuda        --features cuda              (core/compressed on GPU)
#   bin/veritor-zk-host-cuda-gnark  --features cuda,native-gnark (adds groth16/plonk)
# then runs `info` and a GPU smoke proof.  Logs to /workspace/bootstrap.log.
#
# No credentials are placed on the pod.  The repository is synced from the
# laptop with `git archive` / rsync, not cloned.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
export PATH="$HOME/.sp1/bin:$HOME/.cargo/bin:/usr/local/go/bin:$HOME/.local/bin:$PATH"
WORK=/workspace
REPO=$WORK/veritor
mkdir -p "$WORK/bin" "$WORK/results"
cd "$WORK"

stage() { echo; echo "=== [$(date -u +%H:%M:%S)] $*"; }

stage "machine"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
nproc; free -g | head -2; df -h /workspace | tail -1

stage "apt"
apt-get update -qq
apt-get install -y -qq --no-install-recommends \
  build-essential pkg-config libssl-dev clang libclang-dev cmake protobuf-compiler libprotobuf-dev \
  git rsync curl ca-certificates jq >/dev/null

stage "go 1.24 (gnark ffi)"
if ! /usr/local/go/bin/go version 2>/dev/null | grep -q "go1.24"; then
  curl -fsSL https://go.dev/dl/go1.24.6.linux-amd64.tar.gz -o /tmp/go.tgz
  rm -rf /usr/local/go && tar -C /usr/local -xzf /tmp/go.tgz
fi
go version

stage "rustup stable"
if ! command -v rustup >/dev/null; then
  curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable >/dev/null
fi
rustup toolchain install stable --profile minimal >/dev/null
rustc --version

stage "sp1up v6.4.0"
if ! cargo prove --version 2>/dev/null | grep -q "f66b4bf"; then
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

stage "repo"
test -d "$REPO/zk/sp1" || { echo "sync the repository to $REPO first"; exit 2; }
cd "$REPO"
uv sync --frozen >/dev/null 2>&1 || uv sync >/dev/null
.venv/bin/python -c "import veritor; print('veritor ok')"

stage "build host: cuda"
cd "$REPO/zk/sp1"
cargo build --release -p veritor-zk-host --features cuda 2>&1 | tail -3
cp target/release/veritor-zk-host "$WORK/bin/veritor-zk-host-cuda"
sha256sum "$WORK/bin/veritor-zk-host-cuda"

stage "build host: cuda + native-gnark (groth16/plonk)"
if [ "${SKIP_GNARK:-0}" = "1" ]; then
  echo "gnark build: skipped (SKIP_GNARK=1; this campaign uses core/compressed only)"
elif CARGO_TARGET_DIR="$REPO/zk/sp1/target-gnark" cargo build --release -p veritor-zk-host \
     --features cuda,native-gnark 2>&1 | tail -3; then
  cp "$REPO/zk/sp1/target-gnark/release/veritor-zk-host" "$WORK/bin/veritor-zk-host-cuda-gnark"
  echo "gnark build: ok"
else
  echo "gnark build: FAILED (groth16/plonk stage unavailable on this pod; recorded)"
fi

stage "info (CPU client)"
"$WORK/bin/veritor-zk-host-cuda" info | tee "$WORK/results/info.json"

stage "gpu smoke: core proof of the test-vector batch"
cd "$REPO"
.venv/bin/python - <<'EOF'
import json, pathlib
vec = json.loads(pathlib.Path("zk/sp1/common/tests/vectors.json").read_text())
out = pathlib.Path("/workspace/results/smoke"); out.mkdir(parents=True, exist_ok=True)
# vectors.json carries a list of cross-language batches since the int8-q7 merge; take the first accepted one
batches = vec["batches"] if "batches" in vec else [vec["batch"]]
batch = next(b for b in batches if b["verdict"] is True)
(out / "statement.bin").write_bytes(bytes.fromhex(batch["statement"]))
(out / "witness.bin").write_bytes(bytes.fromhex(batch["witness"]))
print("smoke case: vectors.json batch", batch.get("name", ""), batch["obligations"], "obligations, digest", batch["statement_digest"][:12])
EOF
if [ -f /workspace/results/smoke/statement.bin ]; then
  SP1_PROVER=cuda RUST_LOG=info "$WORK/bin/veritor-zk-host-cuda" prove \
    --statement /workspace/results/smoke/statement.bin --witness /workspace/results/smoke/witness.bin \
    --out /workspace/results/smoke/proof-core.bin --mode core 2>&1 | tail -5 | tee /workspace/results/smoke/core.json
fi

stage "bootstrap complete"
echo BOOTSTRAP_OK
