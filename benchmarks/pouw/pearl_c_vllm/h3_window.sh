#!/usr/bin/env bash
# The served -h3 device path on node 2, untimed: GPU 1's ship (build.sh, its SASS gate) built on the node's CPUs, then
# h3_screen.py's capture on one preemptible GPU, its check against the reference on the CPUs after the lease, and window.sh
# MODE=smoke with SCHEME=pearl-c-sm120-v1-h3 and the served row's switches (-h2's: rows form s, whole step, deferred tile
# hashing, both arms graphed).
#   research run --on vy-nebius-2 --project verity --campaign pouw --source <tree> --cwd source --declared-output 'out/*' \
#     --env GPU_LEASE_WHO=<id> -- bash benchmarks/pouw/pearl_c_vllm/h3_window.sh
# STEPS: screen, smoke or both (the default); the smoke passes when window.sh does and smoke.json has the gates and no error.  The
# screen's retained passes stay in scratch (deleted after); its capture.json and screen.json are copied to out/screen.
set -uo pipefail
D=${RESEARCH_RUN_DIR:?}; OUT=$D/out; mkdir -p "$OUT/screen"
WORK=/workspace/pouw/mvp-e2e; PY=$WORK/venv312/bin/python; CUDA=/usr/local/cuda-13.0
S=$(mktemp -d /workspace/pouw/tmp-h3-window.XXXXXX); trap 'rm -rf "$S"' EXIT
HEAD=$(git rev-parse HEAD); echo "tree $HEAD" | tee "$OUT/tree.txt"
STEPS=${STEPS:-both}
stamp() { echo "h3-window: $* at $(date -u +%H:%M:%SZ)"; }
export PYTHONPATH=$PWD:$PWD/catalog:$PWD/kernels:$PWD/experimental:$PWD/integrations/vllm:$PWD/benchmarks/pouw

stamp "ship build"
CUDA=$CUDA bash benchmarks/pouw/pearl_c_sm120/build.sh "$S/ship-${HEAD:0:8}" > "$OUT/build.log" 2>&1 \
  || { stamp "BUILD_FAIL rc=$?"; tail -20 "$OUT/build.log"; exit 11; }
tar -cf "$S/ship.tar" -C "$S" "ship-${HEAD:0:8}"
(cd "$S/ship-${HEAD:0:8}" && sha256sum pearl_c_sm120.cubin) | tee "$OUT/cubin.sha256"
rc=0

if [ "$STEPS" != smoke ]; then
  stamp "screen capture under gpu-lease 1"
  gpu-lease 1 --wait --preemptible --max-min 10 -- bash -c 'export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*} CUDA_DEVICE_ORDER=PCI_BUS_ID;
    exec timeout 540 "$0" benchmarks/pouw/pearl_c_vllm/h3_screen.py capture --kernel benchmarks/pouw/pearl_c_sm120 \
      --cubin "$1" --out "$2"' "$PY" "$S/ship-${HEAD:0:8}/pearl_c_sm120.cubin" "$S/screen" > "$OUT/screen/capture.log" 2>&1
  stamp "capture exited $?"; tail -5 "$OUT/screen/capture.log"
  stamp "screen check on the CPUs"
  timeout 1500 "$PY" benchmarks/pouw/pearl_c_vllm/h3_screen.py check --out "$S/screen" --jobs 32 > "$OUT/screen/check.log" 2>&1
  src=$?; stamp "check exited $src"; cat "$OUT/screen/check.log" | tail -8
  cp "$S/screen"/*.json "$OUT/screen/" 2>/dev/null
  [ "$src" = 0 ] || rc=21
fi

if [ "$STEPS" != screen ]; then
  export SCHEME=pearl-c-sm120-v1-h3 GRAPHS=1 FP8_GRAPHS=1 SCHEDULE=serial ROWS_FORM=s WHOLE_STEP=1 WHOLE_DEFER=1 MODE=smoke
  stamp "window.sh MODE=smoke SCHEME=$SCHEME under gpu-lease 1"
  SHIP_TAR=$S/ship.tar OUT=$OUT/window gpu-lease 1 --wait --preemptible --max-min 25 -- timeout 1440 \
    bash benchmarks/pouw/pearl_c_vllm/window.sh
  wrc=$?; stamp "window.sh exited $wrc"
  rm -rf "$OUT/window/ship"
  [ "$wrc" = 0 ] && { "$PY" -c 'import json,sys; o=json.load(open(sys.argv[1])); sys.exit(0 if o.get("gate_ok") and "error" not in o else 1)' \
    "$OUT/window/smoke.json" || wrc=23; }
  stamp "smoke $([ "$wrc" = 0 ] && echo PASS || echo "FAIL ($wrc)")"
  [ "$wrc" = 0 ] || rc=$((rc ? rc : 22))
fi
exit "$rc"
