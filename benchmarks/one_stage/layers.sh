#!/usr/bin/env bash
# The proof system by layer on stand-in values and one H200: one Llama 3.2 1B decoder layer's W8A8 linears for one token
# (a6.py's default members, 23,552 ScaledMmS8Coordinate_v1 units at K=2048 and K=8192, seeded s8 values, never served),
# each row a6.py on a CUDA build of Flock's flock-circuit, its sessions proved with `prove --gpu`. The served rows are
# benchmarks/prove_path/run.sh's and a5_llama1b_h200.sh's, which `verity bench prove layers` prints:
#   flock     every unit, Flock's plain protocol: each statement a window of its own (Lean's plain scope)
#   zk        every unit, Flock with zero knowledge (--zk), one docket over the layer
#   sampled   SPOT's one-stage audit: LAYERS_K units under the stratified draw, --zk, the audit's ε at δ
# From a clone of the tree, on a machine with an H200 (scratch and builds outside the tree; results under
# $RESEARCH_RUN_DIR/out, which defaults to ./layers-run outside a research run):
#   bash benchmarks/one_stage/layers.sh zk
#   research run --on NODE --cwd clone --mem-gb 192 --timeout 300m -- bash -c 'set -euo pipefail; bash benchmarks/one_stage/layers.sh zk'
# A session takes its card under `gpu-lease 1` where that is installed (GPU_LEASE_WHO names the holder).
# env: LAYERS_K (sampled's k, default 32)  LAYERS_MEMBERS (a6.py's --members)  LAYERS_MEM_GB (default 192)
#      LAYERS_FLOCK_WORK (the CUDA build's directory, kept across runs; default a mktemp -d one)  SM (default 90)
#      LAYERS_LEASE_MIN (a session's lease, default 60)  LAYERS_CACHE (hidden_zk's build cache, a6.sh's A6_CACHE)
set -euo pipefail
[ -n "${LAYERS_CACHE:-}" ] && export A6_CACHE=$LAYERS_CACHE
ROW=${1:?"a row: flock, zk or sampled"}
case $ROW in
  flock) ARGS=(--plain) K= ;;
  zk) ARGS=() K=23552 ;;
  sampled) ARGS=() K=${LAYERS_K:-32} ;;
  *) echo "layers.sh: no row $ROW (flock, zk, sampled; the two-stage audit has no row on this workload)" >&2; exit 2 ;;
esac
[ -n "${LAYERS_MEMBERS:-}" ] && [ "$ROW" = zk ] && K=$(echo "$LAYERS_MEMBERS" | tr , '\n' | awk -F: '{s+=$2} END {print s}')
export RESEARCH_RUN_DIR=${RESEARCH_RUN_DIR:-$PWD/layers-run}
OUT=$RESEARCH_RUN_DIR/out
mkdir -p "$OUT"
FW=${LAYERS_FLOCK_WORK:-$(mktemp -d)}

echo "LAYERS $ROW: coins: each session's coin tree from its own local challenger (FLOCK_CHALLENGER=1: the Lean verifier's" \
  "IO.getRandomBytes, a test LMS key h=4 w=3), Lean's draw on a development challenger (one_stage a0.DEV_SCHEME);" \
  "parameters: development (flock-circuit's default Ligerito configs, δ=2^-20)" | tee "$OUT/layers.txt"

# a build directory is built once: its checkout keeps the patches' new files, which a second `git apply` refuses
KEY=$(find verity/protocols/proofs/flock/prover -type f ! -path '*/python/*' ! -path '*/tests/*' -print0 | sort -z \
  | xargs -0 sha256sum | sha256sum | cut -c1-16)
FW=$FW/sm${SM:-90}-$KEY
BIN=$FW/flock-gpu-link/flock/target/release/flock-circuit
if [ ! -x "$BIN" ]; then
  echo "$(date -u +%H:%M:%SZ) layers: CUDA build of flock-circuit (SM ${SM:-90}) in $FW"
  mkdir -p "$FW"
  RESEARCH_RUN_DIR=$OUT/gpu-build FLOCK_WORK=$FW MODE=build GPU=1 SM=${SM:-90} \
    bash verity/protocols/proofs/flock/prover/pod/60-circuit.sh || { rm -rf "$FW"; exit 1; }
fi
sha256sum "$BIN"
(export PATH=$HOME/.elan/bin:$PATH; cd verity/protocols/proofs/spot/verifier/lean && lake build spot-verifier spot-difftest) \
  > "$OUT/build-challenger.log" 2>&1 || { tail -40 "$OUT/build-challenger.log"; exit 7; }

exec env FLOCK_CHALLENGER=1 FLOCK_PROVE=$BIN FLOCK_SELFTEST=$BIN A6_MEM_GB=${LAYERS_MEM_GB:-192} A6_K=$K \
  A6_MEMBERS=${LAYERS_MEMBERS:-} A6_ARGS="--gpu --lease-min ${LAYERS_LEASE_MIN:-60} ${ARGS[*]:-}" \
  bash benchmarks/one_stage/a6.sh
