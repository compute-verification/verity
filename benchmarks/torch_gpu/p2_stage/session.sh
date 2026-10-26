#!/usr/bin/env bash
# One staging session on vy-dc-h200, from build.sh's lib, ALLP-IFACE's weights module and CMP-95 item 2's fixture (the
# store's layout JSON, `p2_v1/store.py layout`, and its toy, `p2_v1/store-toy/`); cwd = the source tree's root:
#
#   tar -C BUILD_RUN/out -cf /tmp/p2lib.tar .
#   tar -C benchmarks/pous/p2_v1/store-toy -cf /tmp/store-toy.tar toy.json spot.json c-star.bin spot-c.bin
#   research run --on vy-dc-h200 --project verity --campaign e2e-demo --source . --cwd source --timeout 240m \
#     --env GPU_LEASE_WHO=kernels-kstage [--env STEPS=toy,encode,stage,gemm,check] [--env ENCODE_BUDGET=S] \
#     --send /tmp/p2lib.tar --send WEIGHTS/weights.py --send store-layout.json --send /tmp/store-toy.tar \
#     -- bash benchmarks/torch_gpu/p2_stage/session.sh
#
# (WEIGHTS: ALLP-IFACE's verity_torch_loop, whose weights.py writes the seeded stream.)
#
# Outside the lease: the lib unpacked and held to its MANIFEST.sha256, the layout to the fixture's LAYOUT_SHA512; the
# weights stream's manifest, and layer LAYER's stream bytes generated once into SRC/w.bin (`weights write`, seed SEED)
# and kept with `research keep`, as is the development store STORE. Under one lease of at most LEASE_MIN minutes
# (default 25), in order: `bench`, memory-accounting's decode_bench.py unchanged on the lib; then layer.py's `toy`,
# `encode` (ENCODE_BUDGET seconds, default 1080), `stage` and `gemm`. After the lease, `check`: check.py on every encoded
# block and the GEMM samples. Each step's exit code is in $OUT/rc.txt; exits non-zero if any failed.
set -uo pipefail
O=${OUT:?}; IN=${RESEARCH_RUN_DIR:?}/inputs
D=benchmarks/torch_gpu/p2_stage
PY=${VERITY_BENCH_VENV:-/workspace/cp/venv312}/bin/python
CUDA=${CUDA:-/usr/local/cuda-13.0}
LAYER=${LAYER:-0}
SRC=${SRC:-/workspace/tmp/kernels-kstage/p2-stage-l$LAYER}
STORE=${STORE:-/workspace/tmp/kernels-kstage/p2-stage-fixture-l$LAYER}
SEED=${SEED:-text:verity/cmp92/h200/weights/v1}
# public: sha512("not-a-secret: verity/p2-stage/test-salt")[:24], the development store's salt
SALT=${SALT:-011feb1ddfe7334099b02f79a43b4d1411bec106975fffe4}
# p2_v1/store.py's LAYOUT_SHA512 (CMP-95 item 2)
LAYOUT_SHA512=${LAYOUT_SHA512:-45436c2a341cfef20219a071b9b27bd9013535b1de9001ae22d0bf3bd25a825e6aa7c82a60734047d1de59c38e572e126e3c7c64c78f1176e258cb8cb833b783}
STEPS=${STEPS:-toy,encode,stage,gemm,check}
S=$(mktemp -d); trap 'rm -rf "$S"' EXIT
LAYOUT=$IN/store-layout.json
export O D PY S SRC STORE SALT LAYER STEPS LAYOUT PATH=$CUDA/bin:$PATH PYTHONPATH=$PWD:$PWD/benchmarks/pous:$S/tl${PYTHONPATH:+:$PYTHONPATH}
: > "$O/rc.txt"
step() { echo "$1 rc $2 at $(date -u +%H:%M:%SZ)" >> "$O/rc.txt"; }

mkdir -p "$S/lib" "$S/toy" "$S/tl/verity_torch_loop" && tar -C "$S/lib" -xf "$IN/p2lib.tar" \
  && (cd "$S/lib" && sha256sum -c --quiet MANIFEST.sha256) && tar -C "$S/toy" -xf "$IN/store-toy.tar" \
  && [ "$(sha512sum < "$LAYOUT" | cut -c1-128)" = "$LAYOUT_SHA512" ]
step ship $?
grep -q "ship rc 0" "$O/rc.txt" || { cat "$O/rc.txt"; exit 2; }
cp "$S/lib/source.txt" "$S/lib/toolchain.txt" "$S/lib/elf.txt" "$S/lib/MANIFEST.sha256" "$O/"
cp "$IN/weights.py" "$S/tl/verity_torch_loop/weights.py" && touch "$S/tl/verity_torch_loop/__init__.py"
(cd "$IN" && sha256sum weights.py store-layout.json store-toy.tar) > "$O/inputs.sha256"
git rev-parse HEAD > "$O/commit.txt" 2>&1
nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.max.sm,power.limit --format=csv > "$O/gpus.txt" 2>&1
"$PY" -c 'import torch; print(torch.__version__, torch.version.cuda)' > "$O/python.txt" 2>&1

mkdir -p "$SRC" "$STORE"
research keep "$SRC" --owner @kernels --expires 2026-11-01 --priority normal \
  --why "KSTAGE (HASH-12): layer $LAYER's weights-stream bytes, the staging check's plaintext" > /dev/null 2>&1
research keep "$STORE" --owner @kernels --expires 2026-11-01 --priority normal \
  --why "KSTAGE (HASH-12): layer $LAYER's P2 development store in CMP-95 item 2's layout, encoded across leases" > /dev/null 2>&1
df -h "$STORE" > "$O/df.txt" 2>&1
"$PY" -m verity_torch_loop.weights manifest --shape h200 --seed "$SEED" --out "$SRC/manifest.json" 2> "$O/prepare.log" \
  || { step manifest $?; cat "$O/rc.txt" "$O/prepare.log"; exit 2; }
read -r LO HI < <("$PY" - <<'EOF'
import json, os, sys
sys.path.insert(0, os.environ["D"])
import layout as L
print(*L.Weights.of(json.load(open(os.environ["SRC"] + "/manifest.json"))).span(int(os.environ["LAYER"])))
EOF
)
if [ ! -e "$SRC/w.key" ]; then
  "$PY" -m verity_torch_loop.weights write --shape h200 --seed "$SEED" --start "$LO" --stop "$HI" --out "$SRC/w.bin" \
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
grep -q "prepare rc 0" "$O/rc.txt" || { cat "$O/rc.txt"; exit 2; }

nvidia-smi --query-gpu=timestamp,index,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu --format=csv -l 1 \
  > "$O/smi.csv" 2>/dev/null &
SMI=$!
echo "p2_stage: steps $STEPS, asking for the GPU at $(date -u +%H:%M:%SZ)"
LAYER_STEPS=$(echo "$STEPS" | tr , '\n' | grep -E '^(toy|encode|stage|gemm)$' | paste -sd, -)
export LAYER_STEPS ENCODE_BUDGET=${ENCODE_BUDGET:-1080}
if [[ ,$STEPS, == *,bench,* || -n $LAYER_STEPS ]]; then
# shellcheck disable=SC2016
GPU_LEASE_TIMED=1 gpu-lease 1 --wait --max-min "${LEASE_MIN:-25}" -- bash -c '
  step() { echo "$1 rc $2 at $(date -u +%H:%M:%SZ)" >> "$O/rc.txt"; }
  echo "leased at $(date -u +%H:%M:%SZ)" >> "$O/rc.txt"
  nvidia-smi -i "$GPU_LEASE_UUID" -q -d CLOCK,PERFORMANCE,POWER > "$O/clocks.txt" 2>&1
  if [[ ,$STEPS, == *,bench,* ]]; then
    "$PY" benchmarks/pous/p2_v1/decode_bench.py --lib "$S/lib/lib" --build-dir "$S/jit" --blocks $((8 << 16)) --reps 10 \
      --out "$O/decode-bench.json" > "$O/decode-bench.log" 2>&1
    step bench $?
  fi
  if [ -n "$LAYER_STEPS" ]; then
    "$PY" "$D/layer.py" --layout "$LAYOUT" --weights "$SRC/manifest.json" --w "$SRC/w.bin" --store "$STORE" \
      --lib "$S/lib/lib" --salt "$SALT" --out "$O" --toy "$S/toy" --layer "$LAYER" --steps "$LAYER_STEPS" \
      --encode-budget "$ENCODE_BUDGET" > "$O/layer.log" 2> "$O/layer.err"
    step layer $?
  fi
  nvidia-smi -i "$GPU_LEASE_UUID" -q -d CLOCK,PERFORMANCE,POWER > "$O/clocks-end.txt" 2>&1
  echo "released at $(date -u +%H:%M:%SZ)" >> "$O/rc.txt"'
step lease $?
fi
kill "$SMI" 2>/dev/null

if [[ ,$STEPS, == *,check,* ]]; then
  "$PY" "$D/check.py" --layout "$LAYOUT" --weights "$SRC/manifest.json" --w "$SRC/w.bin" --store "$STORE" --salt "$SALT" \
    --out "$O" --layer "$LAYER" > "$O/check.log" 2> "$O/check.err"
  step check $?
fi
cp "$STORE/c.json" "$O/store.json" 2>/dev/null
cat "$O/rc.txt"
! grep -qE " rc [1-9]" "$O/rc.txt"
