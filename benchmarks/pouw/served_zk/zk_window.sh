#!/usr/bin/env bash
# served_zk's served hm96 path on node 2: GPU 1's ship (build.sh) and hm96_rows.cubin built on the node's CPUs, then
# window.sh with the served row's switches (-h2, rows form s, whole step, deferred tile hashing, both
# arms graphed) and STATEMENT, which adds e2e.py's pearlc-zk mode (served_zk/statement.py) and its gate.
#   research run --on vy-nebius-2 --project verity --campaign pouw --source <tree> --cwd source --declared-output 'out/*' \
#     --env GPU_LEASE_WHO=LANE [--env MODE=window] -- bash benchmarks/pouw/served_zk/zk_window.sh
# MODE: smoke (the default; untimed, both engines resident, the gates, each mode's first sampling calls) or window (the gates,
# the interleaved reps with pearlc-zk after each pearlc, and the retained verify passes, recorded at priority low for a day:
# their -h2 commitments are pouw-e2e's, not this run's, so `research retention gc` may take them after that).  LEASE: gpu-lease's arguments (default one preemptible GPU, untimed), e.g. "8 --wait
# --timed" in a booked window.  WRAP: words between gpu-lease's -- and the command, e.g. infra's host-thread pinning
# "env VY_PROVERS_CPUS=160-175 vy-provers taskset -c 160-175".  PREBUILT: a directory holding this head's ship.tar and
# hm96_rows.cubin, which MODE=build writes (the builds only, no lease) and the other modes then use, so a booked window
# spends nothing on building; its sums are checked again before the lease.
set -uo pipefail
D=${RESEARCH_RUN_DIR:?}; ID=${RESEARCH_RUN_ID:?}; OUT=$D/out; mkdir -p "$OUT"
WORK=/workspace/pouw/mvp-e2e; PASSES=$WORK/passes/$ID; CUDA=/usr/local/cuda-13.0
S=$(mktemp -d /workspace/pouw/tmp-served-zk-window.XXXXXX); trap 'rm -rf "$S"' EXIT
HEAD=$(git rev-parse HEAD); echo "tree $HEAD" | tee "$OUT/tree.txt"
stamp() { echo "served-zk: $* at $(date -u +%H:%M:%SZ)"; }
export MODE=${MODE:-smoke} KEEP_PRIORITY=low KEEP_EXPIRES=1d
B=${PREBUILT:-$S}

if [ "$MODE" = build ] || [ -z "${PREBUILT:-}" ]; then
  stamp "ship and hm96_rows.cubin build into $B"
  mkdir -p "$B" && rm -rf "$B/ship-${HEAD:0:8}"
  CUDA=$CUDA bash benchmarks/pouw/pearl_c_sm120/build.sh "$B/ship-${HEAD:0:8}" > "$OUT/build.log" 2>&1 \
    || { stamp "BUILD_FAIL rc=$?"; tail -20 "$OUT/build.log"; exit 11; }
  tar -cf "$B/ship.tar" -C "$B" "ship-${HEAD:0:8}"
  "$CUDA/bin/nvcc" -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -cubin -o "$B/hm96_rows.cubin" \
    benchmarks/pouw/served_zk/hm96_rows.cu 2> "$OUT/nvcc-rows.txt" || { stamp "ROWS_BUILD_FAIL"; tail -20 "$OUT/nvcc-rows.txt"; exit 12; }
  (cd "$B" && sha256sum ship.tar hm96_rows.cubin "ship-${HEAD:0:8}/pearl_c_sm120.cubin" > SHA256SUMS && echo "$HEAD" > HEAD)
  [ "$MODE" = build ] && { cat "$B/SHA256SUMS" | tee "$OUT/prebuilt.sha256"; stamp "built"; exit 0; }
fi
[ "$(cat "$B/HEAD" 2>/dev/null)" = "$HEAD" ] || { stamp "PREBUILT_FAIL: $B is not this head's build"; exit 13; }
(cd "$B" && sha256sum -c --quiet SHA256SUMS) || { stamp "PREBUILT_FAIL: $B/SHA256SUMS does not hold"; exit 13; }
cp "$B/SHA256SUMS" "$OUT/prebuilt.sha256"

export SCHEME=pearl-c-sm120-v1-h3 GRAPHS=1 FP8_GRAPHS=1 SCHEDULE=serial WHOLE_STEP=1 WHOLE_DEFER=1
export STATEMENT=$B/hm96_rows.cubin
read -r -a LEASE_ARGS <<< "${LEASE:-1 --wait --preemptible --max-min 25}"
read -r -a WRAP_ARGS <<< "${WRAP:-}"
stamp "window.sh MODE=$MODE under gpu-lease ${LEASE_ARGS[*]} -- ${WRAP_ARGS[*]}"
SHIP_TAR=$B/ship.tar OUT=$OUT/window gpu-lease "${LEASE_ARGS[@]}" -- "${WRAP_ARGS[@]}" timeout 1440 \
  bash benchmarks/pouw/pearl_c_vllm/window.sh
rc=$?; stamp "window.sh exited $rc"
rm -rf "$OUT/window/ship"
[ -d "$PASSES" ] && du -sh "$PASSES"
exit "$rc"
