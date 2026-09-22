#!/usr/bin/env bash
# [L1 RECORD 2026-09-10] Fresh-pod bootstrap for the record lane (GPU pod `record-gpu`, CPU pod `record-cpu`).
#
#   bash scripts/rec_pod_setup.sh driver      # uv-managed driver venv (Python 3.12: veritor, verity.ir, numpy)  [both pods]
#   bash scripts/rec_pod_setup.sh vllm        # pinned vLLM cu129 venv + SmolLM2-135M checkpoint @ 93efa2f0     [GPU pod]
#   bash scripts/rec_pod_setup.sh host        # sp1 6.4.0 toolchain + `cargo build --release -p veritor-zk-host --features cuda`
#                                                      # from THIS tree (zk/sp1 tree 416482656...) and REFUSE unless the embedded ELF is
#                                                      # the approved, archived 9fd00e60... (e2e/approved_guests.json); no new ELF.
#
# Everything is idempotent; each stage logs under $OUT. Env: REPO (/workspace/veritor), ENV (/workspace/rec-env), OUT (/workspace/rec/setup).
set -uo pipefail
REPO=${REPO:-/workspace/veritor}
ENV=${ENV:-/workspace/rec-env}
OUT=${OUT:-/workspace/rec/setup}
VENV=$ENV/venv-cu129
PYV=$VENV/bin/python
export HF_HOME=${HF_HOME:-$ENV/hf}
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$HOME/.sp1/bin:$PATH"
export CARGO_TERM_COLOR=never
APPROVED_ELF=9fd00e60fb6a74762de987cde211aca9de4576de844569da779acac6bf3863db
APPROVED_VK=0x0018b8e4e955ad4c348bcf57271643814dfcd68d575323410841c29289bd5797
mkdir -p "$ENV" "$OUT"
stage() { echo; echo "=== [$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

do_driver() {
  stage "driver venv (uv sync in $REPO)"
  cd "$REPO"
  command -v uv >/dev/null || (curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1)
  export PATH="$HOME/.local/bin:$PATH"
  if [ ! -x .venv/bin/python ]; then
    uv python install 3.12 >/dev/null 2>&1 || true
    uv sync --frozen >"$OUT/uv-sync.log" 2>&1 || uv sync >>"$OUT/uv-sync.log" 2>&1
  fi
  .venv/bin/python -c "import veritor, numpy, verity.ir, sys; print('driver venv ok', sys.version.split()[0], numpy.__version__)"
}

do_vllm() {
  stage "vLLM venv (pinned wheel d9105ea80 cu129)"
  if [ ! -x "$PYV" ]; then
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install -q -U pip uv
  fi
  mkdir -p "$ENV/wheels" && cd "$ENV/wheels"
  W="vllm-0.28.1rc1.dev472+gd9105ea80.cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
  [ -f "$W" ] || curl -fLo "$W" "https://wheels.vllm.ai/d9105ea8001e0a6d77a96327d17515bb5791fb36/vllm-0.28.1rc1.dev472%2Bgd9105ea80.cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
  echo "7aa52ac71500caf74643135a340eb2d03ae0386d4282fc0b23846940029fbbb3  $W" | sha256sum -c || return 1
  "$PYV" -c "import vllm" 2>/dev/null || "$VENV/bin/uv" pip install --python "$PYV" \
      --extra-index-url https://download.pytorch.org/whl/cu129 --index-strategy unsafe-best-match \
      "$ENV/wheels/$W" numpy safetensors huggingface_hub zstandard pytest
  "$PYV" -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"
  stage "checkpoint (HuggingFaceTB/SmolLM2-135M @ 93efa2f0)"
  REVISION=93efa2f097d58c2a74874c7e644dbc9b0cee75a2
  SNAP="$HF_HOME/hub/models--HuggingFaceTB--SmolLM2-135M/snapshots/$REVISION"
  if [ ! -f "$SNAP/model.safetensors" ]; then
    "$PYV" - <<PYEOF
from huggingface_hub import snapshot_download
print(snapshot_download("HuggingFaceTB/SmolLM2-135M", revision="$REVISION",
                        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"]))
PYEOF
  fi
  echo "80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1  $SNAP/model.safetensors" | sha256sum -c
  ln -sfn "$SNAP" "$ENV/ckpt"
}

do_host() {
  stage "toolchain: rustup + sp1up v6.4.0"
  if ! command -v cargo >/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal >"$OUT/toolchain.log" 2>&1
  fi
  export PATH="$HOME/.cargo/bin:$PATH"
  if ! command -v cargo-prove >/dev/null || ! cargo prove --version 2>/dev/null | grep -q "f66b4bf"; then
    curl -L https://sp1up.succinct.xyz | bash >>"$OUT/toolchain.log" 2>&1
    export PATH="$HOME/.sp1/bin:$PATH"
    sp1up --version v6.4.0 >>"$OUT/toolchain.log" 2>&1
  fi
  { echo "cargo_prove: $(cargo prove --version 2>&1 | head -1)"; echo "rustc_succinct: $(rustup run succinct rustc --version 2>&1 | tail -1 || true)";
    echo "rustc_stable: $(rustc --version)"; echo "cargo: $(cargo --version)"; echo "uname: $(uname -a)"; echo "nproc: $(nproc)"; } | tee "$OUT/toolchain.txt"
  stage "host: cargo build --release -p veritor-zk-host --features cuda (build.rs embeds the guest built from this tree)"
  cd "$REPO/zk/sp1"
  git -C "$REPO" rev-parse HEAD:zk/sp1 2>/dev/null || echo "(tree import; zk/sp1 tree id per laptop: 416482656ffad8a45b4589d933da04e6f9076dff)"
  T0=$(date +%s)
  cargo build --release -p veritor-zk-host --features cuda >"$OUT/host-build.stdout" 2>"$OUT/host-build.stderr" || { echo "HOST BUILD FAILED"; tail -40 "$OUT/host-build.stderr"; return 1; }
  echo "host build wall seconds: $(( $(date +%s) - T0 ))" | tee "$OUT/host-build-time.txt"
  HOST=$REPO/zk/sp1/target/release/veritor-zk-host
  sha256sum "$HOST" | tee "$OUT/host.sha256"
  SP1_PROVER=cpu "$HOST" info | tee "$OUT/host-info.json"
  ELF=$(SP1_PROVER=cpu "$HOST" info | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('elf_sha256',''))")
  VK=$(SP1_PROVER=cpu "$HOST" info | python3 -c "import json,sys; d=json.load(sys.stdin); print(str(d.get('vk_hash','')).lower())")
  if [ "$ELF" != "$APPROVED_ELF" ] || [ "$VK" != "$APPROVED_VK" ]; then
    echo "REFUSED: host embeds ELF $ELF vk $VK; the approved archived guest is $APPROVED_ELF / $APPROVED_VK"; mv "$HOST" "$HOST.REFUSED"; return 2
  fi
  # the standalone ELF the build produced must equal the archived approved artifact byte for byte
  ELF_PATH=$(find "$REPO/zk/sp1/target" -path "*succinct-zkvm-elf/release/veritor-zk-guest" -type f | head -1)
  [ -n "$ELF_PATH" ] && sha256sum "$ELF_PATH" | tee "$OUT/elf.sha256"
  cmp "$ELF_PATH" "$REPO/fixtures/results/GUEST-2026-09-09/build-1/veritor-zk-guest.elf" && echo "ELF == archived approved 9fd00e60 (byte-identical)"
  cp "$HOST" "$ENV/veritor-zk-host"
  cp "$REPO/fixtures/results/GUEST-2026-09-09/build-1/guest_build.json" "$ENV/veritor-zk-host.guest_build.json"
  echo "HOST-OK $ENV/veritor-zk-host"
}

case "${1:-all}" in
  driver) do_driver ;;
  vllm) do_vllm ;;
  host) do_host ;;
  all) do_driver; do_vllm; do_host ;;
  *) echo "usage: $0 driver|vllm|host|all"; exit 4 ;;
esac
echo "REC-SETUP-DONE ${1:-all} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
