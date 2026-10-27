#!/usr/bin/env bash
# The served -h3 device path on a node, untimed: GPU 1's ship (build.sh, its SASS gate) built on the node's CPUs, then
# h3_screen.py's capture on one preemptible GPU, its check against the reference on the CPUs after the lease, and window.sh
# MODE=smoke with SCHEME (default pearl-c-sm120-v1-h3) and the served row's switches (whole step, deferred tile hashing, both
# arms graphed).
#   research run --on vy-nebius-2 --project verity --campaign pouw --source <tree> --cwd source --declared-output 'out/*' \
#     --env GPU_LEASE_WHO=LANE -- bash benchmarks/pouw/pearl_c_vllm/h3_window.sh
# STEPS: the steps to run, of build (the ship), capture (the screen's, on the GPU), check (the capture against the reference,
# on the CPUs) and smoke (on the GPU); default all four; screen is "build capture check", smoke "build smoke" and both all four.
# The smoke passes when window.sh does and smoke.json has the gates and no error.  Without build, SHIP_TAR names a ship tarred by
# an earlier build (SHIP_OUT: where build copies its tar); SCREEN_DIR (default scratch, deleted after) holds the capture's passes
# between capture and check, which deletes it; capture.json and screen.json are copied to out/screen.
# On node 1, whose GPUs only the queue lends, the GPU steps go in a GPU job and the rest in CPU jobs, so the lease holds the GPU
# steps alone: research run --queue --on vy-nebius-1 [--gpus 1 --preemptible] ... with STEPS=build SHIP_OUT=T, then
# STEPS="capture smoke" SHIP_TAR=T SCREEN_DIR=D under --gpus 1, then STEPS=check SCREEN_DIR=D.
set -uo pipefail
D=${RESEARCH_RUN_DIR:?}; OUT=$D/out; mkdir -p "$OUT/screen"
WORK=${WORK:-/workspace/pouw/mvp-e2e}; PY=$WORK/venv312/bin/python; CUDA=/usr/local/cuda-13.0
S=$(mktemp -d "$(dirname "$WORK")/tmp-h3-window.XXXXXX"); trap 'rm -rf "$S"' EXIT
HEAD=$(git rev-parse HEAD); echo "tree $HEAD" | tee "$OUT/tree.txt"
STEPS=${STEPS:-both}; SCHEME=${SCHEME:-pearl-c-sm120-v1-h3}
case $STEPS in both) STEPS="build capture check smoke" ;; screen) STEPS="build capture check" ;; smoke) STEPS="build smoke" ;; esac
has() { case " $STEPS " in *" $1 "*) return 0 ;; esac; return 1; }
SCREEN=${SCREEN_DIR:-$S/screen}
stamp() { echo "h3-window: $* at $(date -u +%H:%M:%SZ)"; }
# under the queue's lease (research run --queue --gpus 1, node 1's GPUs) the workload already holds its GPU
lease() { if [ -n "${GPU_LEASE_UUID:-}" ]; then while [ "$1" != -- ]; do shift; done; shift; "$@"; else gpu-lease "$@"; fi; }
export PYTHONPATH=$PWD:$PWD/tools/config:$PWD/integrations/vllm:$PWD/benchmarks/pouw

if has build; then
  stamp "ship build"
  CUDA=$CUDA bash benchmarks/pouw/pearl_c_sm120/build.sh "$S/ship-${HEAD:0:8}" > "$OUT/build.log" 2>&1 \
    || { stamp "BUILD_FAIL rc=$?"; tail -20 "$OUT/build.log"; exit 11; }
  tar -cf "$S/ship.tar" -C "$S" "ship-${HEAD:0:8}"
  [ -z "${SHIP_OUT:-}" ] || { mkdir -p "$(dirname "$SHIP_OUT")" && cp "$S/ship.tar" "$SHIP_OUT"; }
elif has capture || has smoke; then
  [ -f "${SHIP_TAR:-}" ] || { stamp "NO_SHIP: without build, SHIP_TAR names the ship"; exit 12; }
  cp "$SHIP_TAR" "$S/ship.tar" && tar -xf "$S/ship.tar" -C "$S"
fi
CUBIN=$(find "$S" -name pearl_c_sm120.cubin | head -1)
[ -z "$CUBIN" ] || (cd "$(dirname "$CUBIN")" && sha256sum pearl_c_sm120.cubin) | tee "$OUT/cubin.sha256"
rc=0

if has capture; then
  stamp "screen capture under gpu-lease 1"
  lease 1 --wait --preemptible --max-min 10 -- bash -c 'export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*} CUDA_DEVICE_ORDER=PCI_BUS_ID;
    exec timeout 540 "$0" benchmarks/pouw/pearl_c_vllm/h3_screen.py capture --kernel benchmarks/pouw/pearl_c_sm120 \
      --cubin "$1" --out "$2" --scheme "$3"' "$PY" "$CUBIN" "$SCREEN" "$SCHEME" > "$OUT/screen/capture.log" 2>&1
  crc=$?; stamp "capture exited $crc"; tail -5 "$OUT/screen/capture.log"
  cp "$SCREEN"/*.json "$OUT/screen/" 2>/dev/null
  [ "$crc" = 0 ] || rc=21
fi
if has check; then
  stamp "screen check on the CPUs"
  timeout 1500 "$PY" benchmarks/pouw/pearl_c_vllm/h3_screen.py check --out "$SCREEN" --jobs 32 > "$OUT/screen/check.log" 2>&1
  src=$?; stamp "check exited $src"; cat "$OUT/screen/check.log" | tail -8
  cp "$SCREEN"/*.json "$OUT/screen/" 2>/dev/null
  rm -rf "$SCREEN"
  [ "$src" = 0 ] || rc=21
fi

if has smoke; then
  export SCHEME GRAPHS=1 FP8_GRAPHS=1 SCHEDULE=serial WHOLE_STEP=1 WHOLE_DEFER=1 MODE=smoke
  stamp "window.sh MODE=smoke SCHEME=$SCHEME under gpu-lease 1"
  SHIP_TAR=$S/ship.tar OUT=$OUT/window lease 1 --wait --preemptible --max-min 25 -- timeout 1440 \
    bash benchmarks/pouw/pearl_c_vllm/window.sh
  wrc=$?; stamp "window.sh exited $wrc"
  rm -rf "$OUT/window/ship"
  [ "$wrc" = 0 ] && { "$PY" -c 'import json,sys; o=json.load(open(sys.argv[1])); sys.exit(0 if o.get("gate_ok") and "error" not in o else 1)' \
    "$OUT/window/smoke.json" || wrc=23; }
  stamp "smoke $([ "$wrc" = 0 ] && echo PASS || echo "FAIL ($wrc)")"
  [ "$wrc" = 0 ] || rc=$((rc ? rc : 22))
fi
exit "$rc"
