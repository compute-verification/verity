#!/usr/bin/env bash
# One staging run on vy-dc-h200, from build.sh's lib, the weights module, the weights manifest with
# digests memory-accounting held W to, and the store's fixture (its layout JSON, `p2_v1/store.py layout`,
# and its toy, `p2_v1/store-toy/`); cwd = the source tree's root:
#
#   tar -C BUILD_RUN/out -cf /tmp/p2lib.tar .
#   tar -C benchmarks/pous/p2_v1/store-toy -cf /tmp/store-toy.tar toy.json spot.json c-star.bin spot-c.bin
#   research run --on vy-dc-h200 --project verity --campaign e2e-demo --source . --cwd source --timeout 240m \
#     --env GPU_LEASE_WHO=LANE [--env ENCODED=1] [--env STEPS=toy,stage,gemm,check] \
#     --send /tmp/p2lib.tar --send examples/h200/e2e_h200/served_weights.py --send h200-manifest.json \
#     --send store-layout.json \
#     --send /tmp/store-toy.tar [--send SLICE/setup.json --send SLICE/00000.c --send SLICE/00000.json ...] \
#     -- bash benchmarks/torch_gpu/p2_stage/session.sh
#
# (served_weights.py writes the seeded stream; h200-manifest.json: its manifest with digests.)
#
# With ENCODED=1 the store is memory-accounting's encoded store, read only: the layer's segment files of it (each
# segment's `<s>.c` and `<s>.json`, and `setup.json`, copied off the node that holds it), put once into ENC/seg and kept
# with `research keep`; its salt is `setup.draw_salt(SOURCE, COMMITMENT)`, the deployment's coin and commitment.
# Otherwise the development store STORE under the public test salt SALT, which `encode` extends across runs.
#
# Outside the lease: the lib unpacked and held to its MANIFEST.sha256, the layout to the fixture's LAYOUT_SHA512, the
# manifest to MANIFEST_SHA256; layer LAYER's stream bytes generated once into SRC/w.bin (`weights write`, seed SEED) and
# kept. SRC, STORE and ENC (by default under /workspace/tmp/p2-stage) are kept with `research keep` under keep.json's
# owner, expiry and reasons. Under one untimed lease of at most LEASE_MIN minutes (default 15; LEASE_TIMED=1 asks for a timed one), in order:
# `bench`, memory-accounting's decode_bench.py unchanged on the lib; then layer.py's `toy`, `encode` (development store
# only, ENCODE_BUDGET seconds), `stage` and `gemm` at each batch of M (default 4096,16384). After the lease, `check`:
# check.py on every block of the layer and the GEMM samples, on half the host's cores. Each step's exit code is in
# $OUT/rc.txt; exits non-zero if any failed.
set -uo pipefail
O=${OUT:?}; IN=${RESEARCH_RUN_DIR:?}/inputs
D=benchmarks/torch_gpu/p2_stage
PY=${VERITY_BENCH_VENV:-/workspace/cp/venv312}/bin/python
CUDA=${CUDA:-/usr/local/cuda-13.0}
LAYER=${LAYER:-0}
SRC=${SRC:-/workspace/tmp/p2-stage/stream-l$LAYER}
STORE=${STORE:-/workspace/tmp/p2-stage/fixture-l$LAYER}
ENC=${ENC:-/workspace/tmp/p2-stage/mem69-v2-l$LAYER}
SEED=${SEED:-text:verity/cmp92/h200/weights/v1}
# public: sha512("not-a-secret: verity/p2-stage/test-salt")[:24], the development store's salt
SALT=${SALT:-011feb1ddfe7334099b02f79a43b4d1411bec106975fffe4}
# the deployment's coin of record and W's commitment under the frozen layout
SOURCE=${SOURCE:-dd56f549e504fa8a48db829588683baa66d5ab0c8c381665a87c3b53ffeb3ced}
COMMITMENT=${COMMITMENT:-9aaaf192b1a7c0962b6139b7bad5b6a2abc9809d889ed5da5890628d1799dfe4291ce318bc8ba410b41886e30d52278f89686f429839257eab1728ac5902706a}
# p2_v1/store.py's LAYOUT_SHA512, and h200-manifest.json's
LAYOUT_SHA512=${LAYOUT_SHA512:-45436c2a341cfef20219a071b9b27bd9013535b1de9001ae22d0bf3bd25a825e6aa7c82a60734047d1de59c38e572e126e3c7c64c78f1176e258cb8cb833b783}
MANIFEST_SHA256=${MANIFEST_SHA256:-28a29668627351e0118b02bbb4dddd0d3f25e2dfa6860b82f3048c0871792f27}
STEPS=${STEPS:-toy,stage,gemm,check}
M=${M:-4096,16384}
S=$(mktemp -d); trap 'rm -rf "$S"' EXIT
LAYOUT=$IN/store-layout.json; MANIFEST=$IN/h200-manifest.json
export O D PY S SRC STORE ENC SALT LAYER STEPS LAYOUT MANIFEST PATH=$CUDA/bin:$PATH \
  PYTHONPATH=$PWD:$PWD/benchmarks/pous:$S/tl${PYTHONPATH:+:$PYTHONPATH}
: > "$O/rc.txt"
step() { echo "$1 rc $2 at $(date -u +%H:%M:%SZ)" >> "$O/rc.txt"; }
keep() {  # PATH WHAT: `research keep` PATH under keep.json's owner and expiry, with WHAT's reason
  local owner expires why
  IFS=$'\t' read -r owner expires why < <("$PY" -c 'import json, sys
k = json.load(open(sys.argv[1]))
print(k["owner"], k["expires"], k["why"][sys.argv[2]].format(layer=sys.argv[3]), sep="\t")' "$D/keep.json" "$2" "$LAYER")
  research keep "$1" --owner "$owner" --expires "$expires" --priority normal --why "$why" > /dev/null 2>&1
}

mkdir -p "$S/lib" "$S/toy" "$S/tl/e2e_h200" && tar -C "$S/lib" -xf "$IN/p2lib.tar" \
  && (cd "$S/lib" && sha256sum -c --quiet MANIFEST.sha256) && tar -C "$S/toy" -xf "$IN/store-toy.tar" \
  && [ "$(sha512sum < "$LAYOUT" | cut -c1-128)" = "$LAYOUT_SHA512" ] \
  && [ "$(sha256sum < "$MANIFEST" | cut -c1-64)" = "$MANIFEST_SHA256" ]
step ship $?
grep -q "ship rc 0" "$O/rc.txt" || { cat "$O/rc.txt"; exit 2; }
cp "$S/lib/source.txt" "$S/lib/toolchain.txt" "$S/lib/elf.txt" "$S/lib/MANIFEST.sha256" "$O/"
cp "$IN/served_weights.py" "$S/tl/e2e_h200/served_weights.py" && touch "$S/tl/e2e_h200/__init__.py"
(cd "$IN" && sha256sum served_weights.py h200-manifest.json store-layout.json store-toy.tar) > "$O/inputs.sha256"
git rev-parse HEAD > "$O/commit.txt" 2>&1
nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.max.sm,power.limit --format=csv > "$O/gpus.txt" 2>&1
"$PY" -c 'import torch; print(torch.__version__, torch.version.cuda)' > "$O/python.txt" 2>&1

mkdir -p "$SRC"
keep "$SRC" src
if [ "${ENCODED:-0}" = 1 ]; then
  if [ -f "$IN/setup.json" ]; then
    mkdir -p "$ENC/seg" && cp "$IN/setup.json" "$ENC/" \
      && for f in "$IN"/[0-9][0-9][0-9][0-9][0-9].c "$IN"/[0-9][0-9][0-9][0-9][0-9].json; do
           ln -f "$f" "$ENC/seg/" 2>/dev/null || cp "$f" "$ENC/seg/"; done
  fi
  keep "$ENC" enc
  [ -f "$ENC/setup.json" ] && ls "$ENC/seg" > "$O/encoded-files.txt"
  step encoded $?
  STORE_ARGS=(--encoded "$ENC" --source "$SOURCE" --commitment "$COMMITMENT")
else
  mkdir -p "$STORE"
  keep "$STORE" store
  STORE_ARGS=(--store "$STORE" --salt "$SALT")
fi
df -h "$SRC" > "$O/df.txt" 2>&1
read -r LO HI < <("$PY" - <<'EOF'
import json, os, sys
sys.path.insert(0, os.environ["D"])
import layout as L
print(*L.Weights.of(json.load(open(os.environ["MANIFEST"]))).span(int(os.environ["LAYER"])))
EOF
)
if [ ! -e "$SRC/w.key" ]; then
  "$PY" -m e2e_h200.served_weights write --shape h200 --seed "$SEED" --start "$LO" --stop "$HI" --out "$SRC/w.bin" \
    --workers "$(nproc)" 2>> "$O/prepare.log" && echo "$SEED $LO $HI" > "$SRC/w.key"
  step prepare $?
else
  read -r KSEED KLO KHI < "$SRC/w.key"
  if [ "$KSEED" = "$SEED" ] && [ "$KLO" -le "$LO" ] && [ "$KHI" -ge "$HI" ]; then
    step prepare 0
  else
    echo "$SRC holds W '$(cat "$SRC/w.key")', not seed $SEED's [$LO, $HI): pass another SRC" >> "$O/rc.txt"
    step prepare 2
  fi
fi
echo "stream bytes [$LO, $HI) of seed $SEED" >> "$O/rc.txt"
grep -qE "^(encoded|prepare) rc [1-9]" "$O/rc.txt" && { cat "$O/rc.txt"; exit 2; }
COMMON=(--layout "$LAYOUT" --weights "$MANIFEST" --w "$SRC/w.bin" "${STORE_ARGS[@]}" --layer "$LAYER" --out "$O")
export COMMON_ARGS="${COMMON[*]}"

nvidia-smi --query-gpu=timestamp,index,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu --format=csv -l 1 \
  > "$O/smi.csv" 2>/dev/null &
SMI=$!
echo "p2_stage: steps $STEPS, asking for the GPU at $(date -u +%H:%M:%SZ)"
LAYER_STEPS=$(echo "$STEPS" | tr , '\n' | grep -E '^(toy|encode|stage|gemm)$' | paste -sd, -)
export LAYER_STEPS M ENCODE_BUDGET=${ENCODE_BUDGET:-780}
if [[ ,$STEPS, == *,bench,* || -n $LAYER_STEPS ]]; then
TIMED=(); [ "${LEASE_TIMED:-0}" = 1 ] && TIMED=(--timed)
# shellcheck disable=SC2016
env -u GPU_LEASE_TIMED gpu-lease 1 --wait --max-min "${LEASE_MIN:-15}" "${TIMED[@]}" -- bash -c '
  step() { echo "$1 rc $2 at $(date -u +%H:%M:%SZ)" >> "$O/rc.txt"; }
  echo "leased at $(date -u +%H:%M:%SZ)" >> "$O/rc.txt"
  nvidia-smi -i "$GPU_LEASE_UUID" -q -d CLOCK,PERFORMANCE,POWER > "$O/clocks.txt" 2>&1
  if [[ ,$STEPS, == *,bench,* ]]; then
    "$PY" benchmarks/pous/p2_v1/decode_bench.py --lib "$S/lib/lib" --build-dir "$S/jit" --blocks $((8 << 16)) --reps 10 \
      --out "$O/decode-bench.json" > "$O/decode-bench.log" 2>&1
    step bench $?
  fi
  if [ -n "$LAYER_STEPS" ]; then
    # shellcheck disable=SC2086
    "$PY" "$D/layer.py" $COMMON_ARGS --lib "$S/lib/lib" --toy "$S/toy" --steps "$LAYER_STEPS" --m "$M" \
      --encode-budget "$ENCODE_BUDGET" > "$O/layer.log" 2> "$O/layer.err"
    step layer $?
  fi
  nvidia-smi -i "$GPU_LEASE_UUID" -q -d CLOCK,PERFORMANCE,POWER > "$O/clocks-end.txt" 2>&1
  echo "released at $(date -u +%H:%M:%SZ)" >> "$O/rc.txt"'
step lease $?
fi
kill "$SMI" 2>/dev/null

if [[ ,$STEPS, == *,check,* ]]; then
  nice -n 10 "$PY" "$D/check.py" "${COMMON[@]}" --workers $(( ($(nproc) + 1) / 2 )) > "$O/check.log" 2> "$O/check.err"
  step check $?
fi
[ "${ENCODED:-0}" = 1 ] || cp "$STORE/c.json" "$O/store.json" 2>/dev/null
cat "$O/rc.txt"
! grep -qE " rc [1-9]" "$O/rc.txt"
