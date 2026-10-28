#!/usr/bin/env bash
# The whole encoded store through the staging decode on the demo card (full.py), in place and read only, from build.sh's
# lib, the weights manifest with digests memory-accounting held W to, and the store's layout; cwd = the source
# tree's root:
#
#   tar -C BUILD_RUN/out -cf /tmp/p2lib.tar .
#   research run --on vy-dc-h200-2 --project verity --campaign e2e-demo --source . --cwd source --timeout 90m \
#     --env GPU_LEASE_WHO=LANE [--env RECEIPTS=1 --env GEMM_LAYER= --env SAMPLE=0] \
#     --send /tmp/p2lib.tar --send h200-manifest.json --send store-layout.json \
#     -- bash benchmarks/torch_gpu/p2_stage/full.sh
#
# (h200-manifest.json: `e2e_h200.served_weights manifest --digests`; store-layout.json: `p2_v1/store.py
# layout`.) ENC is memory-accounting's finished store (mem69-v2: every segment's `.c` and `.json`, `setup.json`
# and `store.json`), only ever read: nothing here writes, moves or deletes anything under it.
#
# Outside the lease: the lib unpacked and held to its MANIFEST.sha256, the layout to LAYOUT_SHA512, the manifest to
# MANIFEST_SHA256 and to the store's own record of it, the store's setup and record copied out; with RECEIPTS=1, the
# receipts' and sha512_rows' kernels built for ARCH (default 90a). Under one untimed lease of at most LEASE_MIN minutes
# (default 20), refused if it could reach one of QUIET's periods: full.py, every tensor of LAYERS (default all 52
# layers and the head) staged and held to the manifest's SHA-256, with RECEIPTS=1 the bandwidth audit's pass beside it
# (`--receipts`, its σ, y and salts kept in scratch), then GEMM_LAYER's matrices (default 51; empty for none) through
# the exact W8A8 GEMM at each batch of M (default 16384). After the lease, niced on half the host's cores: check.py on the GEMM's
# samples and SAMPLE blocks of every tensor (default 128; 0 skips it) against P2's reference, and with RECEIPTS=1
# bw_ref.py, the CPU reference's check of the kept pass (every y and the root, BW_SAMPLE units' σ, default 100000).
# Each step's exit code is in $OUT/rc.txt; exits non-zero if any failed.
set -uo pipefail
O=${OUT:?}; IN=${RESEARCH_RUN_DIR:?}/inputs
D=benchmarks/torch_gpu/p2_stage
for v in ${VERITY_BENCH_VENV:-} /workspace/cp/venv312 /workspace/jobs/venv312; do
  [ -z "${PY:-}" ] && [ -x "$v/bin/python" ] && PY=$v/bin/python
done
PY=${PY:-python3}
"$PY" -c 'import torch' 2>/dev/null || { PATH=$HOME/.local/bin:$PATH uv sync -q --all-packages --extra torch > "$O/sync.log" 2>&1; PY=$PWD/.venv/bin/python; }
TV=$($PY -c 'import torch; print(torch.version.cuda or "")' 2>/dev/null)
for c in ${CUDA:-} /usr/local/cuda-13.0 "/usr/local/cuda-$TV" /usr/local/cuda; do [ -x "$c/bin/nvcc" ] && { CUDA=$c; break; }; done
ENC=${ENC:-/workspace/research/pous-store/mem69-v2}
SEED=${SEED:-text:verity/cmp92/h200/weights/v1}
# the deployment's coin of record and W's commitment under the frozen layout
SOURCE=${SOURCE:-dd56f549e504fa8a48db829588683baa66d5ab0c8c381665a87c3b53ffeb3ced}
COMMITMENT=${COMMITMENT:-9aaaf192b1a7c0962b6139b7bad5b6a2abc9809d889ed5da5890628d1799dfe4291ce318bc8ba410b41886e30d52278f89686f429839257eab1728ac5902706a}
# p2_v1/store.py's LAYOUT_SHA512, and h200-manifest.json's
LAYOUT_SHA512=${LAYOUT_SHA512:-45436c2a341cfef20219a071b9b27bd9013535b1de9001ae22d0bf3bd25a825e6aa7c82a60734047d1de59c38e572e126e3c7c64c78f1176e258cb8cb833b783}
MANIFEST_SHA256=${MANIFEST_SHA256:-28a29668627351e0118b02bbb4dddd0d3f25e2dfa6860b82f3048c0871792f27}
LAYERS=${LAYERS:-}
GEMM_LAYER=${GEMM_LAYER-51}
M=${M:-16384}
SAMPLE=${SAMPLE:-128}
LEASE_MIN=${LEASE_MIN:-20}
RECEIPTS=${RECEIPTS:-}
ARCH=${ARCH:-90a}
BW_SAMPLE=${BW_SAMPLE:-100000}
# the card's periods no lease may reach, each START/END in UTC, space-separated (none by default)
QUIET=${QUIET:-}
S=$(mktemp -d); trap 'rm -rf "$S"' EXIT
LAYOUT=$IN/store-layout.json; MANIFEST=$IN/h200-manifest.json
export O D PY S ENC LAYOUT MANIFEST MANIFEST_SHA256 LEASE_MIN ARCH QUIET PATH=$CUDA/bin:$PATH \
  P2_RECEIPTS_BUILD=$S/p2r SHA512_ROWS_BUILD=$S/sxr \
  PYTHONPATH=$PWD:$PWD/benchmarks/pous:$PWD/demos/h200${PYTHONPATH:+:$PYTHONPATH}
: > "$O/rc.txt"
step() { echo "$1 rc $2 at $(date -u +%H:%M:%SZ)" >> "$O/rc.txt"; }

mkdir -p "$S/lib" && tar -C "$S/lib" -xf "$IN/p2lib.tar" && (cd "$S/lib" && sha256sum -c --quiet MANIFEST.sha256) \
  && [ "$(sha512sum < "$LAYOUT" | cut -c1-128)" = "$LAYOUT_SHA512" ] \
  && [ "$(sha256sum < "$MANIFEST" | cut -c1-64)" = "$MANIFEST_SHA256" ]
step ship $?
cp "$S/lib/source.txt" "$S/lib/toolchain.txt" "$S/lib/elf.txt" "$S/lib/MANIFEST.sha256" "$O/"
for f in setup.json store.json; do [ -f "$ENC/$f" ] && cp "$ENC/$f" "$O/store-$f"; done
"$PY" - > "$O/store-check.txt" 2>&1 <<'EOF'
import glob, json, os, sys
enc, want = os.environ["ENC"], os.environ["MANIFEST_SHA256"]
got = {f: json.load(open(p)).get("weights_manifest_sha256") for f in ("setup.json", "store.json")
       if os.path.exists(p := os.path.join(enc, f))}
segs = (len(glob.glob(os.path.join(enc, "seg", "*.c"))), len(glob.glob(os.path.join(enc, "seg", "*.json"))))
print(json.dumps({"store": enc, "weights_manifest_sha256": got, "segment_files": segs}))
sys.exit(0 if any(got.values()) and all(v == want for v in got.values() if v) else 1)
EOF
step store $?
if [ -n "$RECEIPTS" ]; then
  "$PY" -c 'import os, sys; sys.path.insert(0, os.environ["D"]); import sketch as K
K.receipts_lib(os.environ["ARCH"]); K.SE.device_lib(os.environ["ARCH"])' > "$O/build.log" 2>&1
  step build $?
fi
grep -qE "^(ship|store|build) rc [1-9]" "$O/rc.txt" && { cat "$O/rc.txt" "$O/store-check.txt"; exit 2; }
(cd "$IN" && sha256sum p2lib.tar h200-manifest.json store-layout.json) > "$O/inputs.sha256"
git rev-parse HEAD > "$O/commit.txt" 2>&1
nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.max.sm,power.limit,memory.total,memory.used \
  --format=csv > "$O/gpus.txt" 2>&1
"$PY" -c 'import torch; print(torch.__version__, torch.version.cuda)' > "$O/python.txt" 2>&1
gpu-lease status > "$O/lease-before.txt" 2>&1
df -h "$S" "$ENC" > "$O/df.txt" 2>&1
COMMON=(--layout "$LAYOUT" --weights "$MANIFEST" --generator e2e_h200.served_weights --shape h200 --weights-seed "$SEED"
        --encoded "$ENC" --source "$SOURCE" --commitment "$COMMITMENT" --out "$O")
FULL=("${COMMON[@]}" --lib "$S/lib/lib" ${GEMM_LAYER:+--gemm-layer "$GEMM_LAYER" --m "$M"} ${LAYERS:+--layers "$LAYERS"}
      ${RECEIPTS:+--receipts --keep "$S/keep"})
export FULL_ARGS="${FULL[*]}"

nvidia-smi --query-gpu=timestamp,index,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used \
  --format=csv -l 1 > "$O/smi.csv" 2>/dev/null &
SMI=$!
echo "p2_stage full: asking for the GPU at $(date -u +%H:%M:%SZ)"
# shellcheck disable=SC2016
env -u GPU_LEASE_TIMED gpu-lease 1 --wait --max-min "$LEASE_MIN" -- bash -c '
  step() { echo "$1 rc $2 at $(date -u +%H:%M:%SZ)" >> "$O/rc.txt"; }
  echo "leased at $(date -u +%H:%M:%SZ)" >> "$O/rc.txt"
  now=$(date -u +%s)
  for w in $QUIET; do
    w0=$(date -u -d "${w%/*}" +%s); w1=$(date -u -d "${w#*/}" +%s)
    if [ $((now + 60 * LEASE_MIN)) -gt "$w0" ] && [ "$now" -lt "$w1" ]; then
      echo "a lease from now could reach the quiet period $w: released unused" >> "$O/rc.txt"
      step window 3; exit 3
    fi
  done
  nvidia-smi -i "$GPU_LEASE_UUID" -q -d CLOCK,PERFORMANCE,POWER,MEMORY > "$O/clocks.txt" 2>&1
  # shellcheck disable=SC2086
  "$PY" "$D/full.py" $FULL_ARGS > "$O/full.log" 2> "$O/full.err"
  step full $?
  nvidia-smi -i "$GPU_LEASE_UUID" -q -d CLOCK,PERFORMANCE,POWER,MEMORY > "$O/clocks-end.txt" 2>&1
  echo "released at $(date -u +%H:%M:%SZ)" >> "$O/rc.txt"'
step lease $?
kill "$SMI" 2>/dev/null

WORKERS=$(( ($(nproc) + 1) / 2 ))
if [ "$SAMPLE" != 0 ]; then
  nice -n 10 "$PY" "$D/check.py" "${COMMON[@]}" --sample "$SAMPLE" --workers "$WORKERS" > "$O/check.log" 2> "$O/check.err"
  step check $?
fi
if [ -n "$RECEIPTS" ]; then
  cp "$S/keep/pass.json" "$O/pass.json" 2>/dev/null
  nice -n 10 "$PY" "$D/bw_ref.py" --keep "$S/keep" --layout "$LAYOUT" --weights "$MANIFEST" --encoded "$ENC" \
    --source "$SOURCE" --commitment "$COMMITMENT" --out "$O" --sample "$BW_SAMPLE" --workers "$WORKERS" \
    > "$O/bw_ref.log" 2> "$O/bw_ref.err"
  step bw_ref $?
fi
tail -1 "$O/full.log"
[ -n "$RECEIPTS" ] && tail -1 "$O/bw_ref.log"
cat "$O/rc.txt"
! grep -qE " rc [1-9]" "$O/rc.txt"
