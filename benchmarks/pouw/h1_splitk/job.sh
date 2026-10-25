#!/usr/bin/env bash
# H1 lanes in chunks along K (h1zy.PouwH1LaneZyChunk_v1, h1zy.chunks) over a 32 × 32 leaf's v1 tile: run.py's structural
# stage of every chunk (k_log, template evaluated against the word reference, peak RSS), check_unit on each chunk's
# template, or circuit-check on the chunk targets (run.py's docstring). Run from the source tree's root:
#   research run --on vy-nebius-cpu-4 --project verity --source . --cwd source --mem-gb G --mem-wait 7200 \
#     --declared-output 'out/*' --env SK_WHAT=stage --env SK_KS=6144,12288,14336,16384 --env SK_WHOLE=6144 \
#     --env SK_JOBS=6 -- bash benchmarks/pouw/h1_splitk/job.sh
# env: SK_WHAT (stage, check or cc)  SK_KS (the Ks whose chunks run; cc: in place of the sidecar's and the small 32 × 32
#      targets)  SK_WHOLE (Ks whose whole lane is staged too)
#      SK_ONLY (K:C,.. or K:whole: only those)  SK_JOBS (processes at a time; cc: circuit-check's -j)  SK_MEM_GB (cc's --mem-gb)
set -uo pipefail
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH
SRC=$(pwd)
OUT=${OUT:?run under research run, which sets OUT}
mkdir -p "${TMPDIR:-/tmp}" && W=$(mktemp -d "${TMPDIR:-/tmp}/h1-splitk.XXXXXX")
trap 'rm -rf "$W"' EXIT
export UV_PROJECT_ENVIRONMENT=$W/venv
UV="uv run -q --locked --all-packages --extra torch-cpu"
WHAT=${SK_WHAT:?stage, check or cc}
log() { echo "$(date -u +%H:%M:%SZ) $*" | tee -a "$OUT/job.log"; }
{ lscpu | grep -E 'Model name|^CPU\(s\)'; grep -E 'MemTotal|MemAvailable' /proc/meminfo; echo "mem ${RESEARCH_RUN_MEM_GB:-?} GB"; } \
  > "$OUT/host.txt" 2>&1
log "h1-splitk $WHAT: source ${RESEARCH_SOURCE_SHA:-$(git -C "$SRC" rev-parse HEAD 2>/dev/null)}"
uv sync -q --locked --all-packages --extra torch-cpu > "$OUT/sync.log" 2>&1 || { log "uv sync FAILED: $(tail -5 "$OUT/sync.log")"; exit 1; }
R=benchmarks/pouw/h1_splitk/run.py
if [ "$WHAT" = cc ]; then
  $UV python "$R" targets "$OUT/cc.targets" ${SK_KS:+--ks "$SK_KS"} ${SK_ONLY:+--only "$SK_ONLY"} | tee -a "$OUT/job.log"
  mapfile -t T < "$OUT/cc.targets"
  $UV circuit-check "${T[@]}" --out "$OUT/cc.json" -j "${SK_JOBS:-3}" --mem-gb "${SK_MEM_GB:-${RESEARCH_RUN_MEM_GB:-64}}" \
    > "$OUT/cc.log" 2>&1
  rc=$?
  tail -12 "$OUT/cc.log" | tee -a "$OUT/job.log"
else
  $UV python "$R" all "$WHAT" --ks "${SK_KS:-}" --whole "${SK_WHOLE:-}" ${SK_ONLY:+--only "$SK_ONLY"} --jobs "${SK_JOBS:-1}" \
    --out "$OUT" 2>&1 | tee -a "$OUT/job.log"
  rc=${PIPESTATUS[0]}
fi
log "h1-splitk $WHAT: done (exit $rc)"
exit "$rc"
