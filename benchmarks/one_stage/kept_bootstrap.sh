#!/usr/bin/env bash
# The vLLM bootstrap (adapters/vllm/verity_vllm/ops/pod_bootstrap.sh) kept on the node under BOOT_ROOT, outside any run's
# directory, in a directory keyed by a digest of everything it is built from (key.txt): pod_bootstrap.sh (the vLLM
# wheel's pin and sha256, torchcodec's, the cu129 index), torch's version (the wheel's own pin), the taps' scripts and
# sources, the CUDA toolkit (its path and nvcc's version), the card's architecture, the checkpoint pins and the cases.
# A key that matches a finished directory reuses it; one that doesn't builds a new one. Its venv, wheel and checkpoints
# are built with no card (pod_bootstrap.sh --cpu); the CUDA headers and taps, which read the card's capability, under ON
# (a lease of the card, or taskset). One job builds a key at a time (flock on DIR.lock); a job reading DIR holds
# `flock -s DIR.lock` while it does. Prints DIR on its last line.
#
#   bash benchmarks/one_stage/kept_bootstrap.sh CARD CASES OUT [ON...]
#   BOOT_ROOT (/workspace/cache/one-stage-bootstrap), CUDA_HOME, ROW_TAPS, FA2_TAP_HDIMS (pod_bootstrap.sh's)
set -uo pipefail
CARD=${1:?the card index}; CASES=${2:?the checkpoint cases}; OUT=${3:?a directory for the records of the bootstrap}; shift 3
REPO=$(pwd); A=$REPO/adapters/vllm; OPS=$A/verity_vllm/ops; BOOT=$OPS/pod_bootstrap.sh
[ -f "$BOOT" ] || { echo "kept_bootstrap.sh: run from the tree's root" >&2; exit 2; }
ROOT=${BOOT_ROOT:-/workspace/cache/one-stage-bootstrap}
mkdir -p "$ROOT/wheels" "$OUT" || exit 2
arch=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i "$CARD" 2>/dev/null | head -1)
[ -n "$arch" ] || { echo "kept_bootstrap.sh: nvidia-smi names no card $CARD" >&2; exit 2; }
CUDA_HOME=$(python3 benchmarks/kernels/machine.py toolkit 12.9) \
  || { echo "kept_bootstrap.sh: no CUDA 12.9 toolkit (benchmarks/kernels/machine.py toolkit 12.9, above)" >&2; exit 2; }
export CUDA_HOME
nvcc=$("$CUDA_HOME/bin/nvcc" --version 2>/dev/null | tail -2 | tr '\n' ' ')
[ -n "$nvcc" ] || { echo "kept_bootstrap.sh: no nvcc under CUDA_HOME=$CUDA_HOME" >&2; exit 2; }
torch=$(python3 - "$BOOT" "$ROOT/wheels" <<'EOF'
import hashlib, re, subprocess, sys, zipfile
from pathlib import Path
text, wheels = Path(sys.argv[1]).read_text(), Path(sys.argv[2])
name, sha = re.search(r'^W="(.+)"$', text, re.M)[1], re.search(r"^WSHA=([0-9a-f]{64})$", text, re.M)[1]
url = re.search(r'"(https://wheels\.vllm\.ai/[^"]+\.whl)"', text)[1]
w = wheels / name
if not w.is_file() or hashlib.sha256(w.read_bytes()).hexdigest() != sha:
    part = w.with_name(w.name + ".part")
    subprocess.run(["curl", "-fL", "--retry", "5", "-o", str(part), url], check=True, stdout=subprocess.DEVNULL)
    if hashlib.sha256(part.read_bytes()).hexdigest() != sha:
        sys.exit(f"kept_bootstrap.sh: the sha256 of {name} is not {sha}")
    part.rename(w)
with zipfile.ZipFile(w) as z:
    meta = z.read(next(n for n in z.namelist() if n.endswith(".dist-info/METADATA"))).decode()
print(re.search(r"^Requires-Dist: (torch==\S+)", meta, re.M)[1])
EOF
) || exit 2
srcs=$(cd "$A" && find verity_vllm/ops/pod_fa2_tap.sh verity_vllm/ops/pod_fa3_tap.sh verity_vllm/ops/pod_norm_tap.sh \
  verity_vllm/ops/pod_router_tap.sh verity_vllm/ops/cuda_headers.sh verity_vllm/acquire/fa2_tap_src verity_vllm/acquire/fa3_tap_src \
  verity_vllm/acquire/norm_tap_src verity_vllm/acquire/router_tap_src manifests/checkpoints.json manifests/checkpoints-int8.json \
  -type f ! -name '*.pyc' ! -path '*/__pycache__/*' | LC_ALL=C sort | xargs sha256sum) || { echo "kept_bootstrap.sh: a key input is missing" >&2; exit 2; }
key=$(printf '%s\n' "wheel+pins $(sha256sum "$BOOT" | cut -c1-64) $torch" "$srcs" "cuda $(realpath "$CUDA_HOME") $nvcc" "arch $arch" \
  "cases $CASES row_taps=${ROW_TAPS:-1} hdims=${FA2_TAP_HDIMS:-}")
K=$(sha256sum <<< "$key" | cut -c1-16); D=$ROOT/$K
ready() { [ -f "$D/ready" ] && "$D/venv312/bin/python" -c 'import torch, vllm' > /dev/null 2>&1; }
exec 8> "$D.lock"; flock -s 8
ready || { flock -u 8; flock 8; }
if ! ready; then
  rm -rf "$D"; mkdir -p "$D" && ln -s ../wheels "$D/wheels" || exit 2
  echo "kept_bootstrap.sh: building $D (no card first, then the taps under ${*:-no prefix})"
  (cd "$A" && WORK=$D bash verity_vllm/ops/pod_bootstrap.sh --cpu --cases "$CASES" --out "$OUT/cpu") || exit 3
  (cd "$A" && WORK=$D "$@" bash verity_vllm/ops/pod_bootstrap.sh --gpu --cases "$CASES" --out "$OUT/gpu") || exit 3
  echo "$key" > "$D/key.txt" && touch "$D/ready"
  PYTHONPATH=$REPO/tools/research/src python3 -m research keep "$ROOT" --owner @benchmarks --expires never --priority keep \
    --regenerate "bash benchmarks/one_stage/kept_bootstrap.sh CARD CASES OUT" \
    --why "the one-stage runs' vLLM bootstraps, each under the digest of what it is built from (key.txt)" > "$OUT/keep.json" 2>&1 \
    || echo "no retention record on $ROOT: $(tail -1 "$OUT/keep.json")" >&2
else
  echo "kept_bootstrap.sh: reusing $D"
fi
cp "$D/key.txt" "$OUT/key.txt"
echo "$D"
