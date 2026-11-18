#!/usr/bin/env bash
# The card's refresh path on one card, in stages: each starts only after the one before it passed, and the first that
# fails is named, with every stage after it not run (cwd = the tree; `research run --cwd clone`). It runs inside the
# caller's lease of the card, which every stage shares:
#
#   research run --on vy-h100-1 --project verity --source TREE --cwd clone -- \
#     gpu-lease 1 --wait --cores 0-9 --max-min 120 -- env NOT_AFTER=2026-11-19T03:30Z \
#     bash benchmarks/poub_session/refresh_card.sh
#
#   base     rows.sh with STEPS=rows: session.cu and kv.cu built for the card, the base's kernels unchanged
#            (same_sass.py), the sign key, the SASS gate (the 64-row kernels, rs_pass_at's and rs_subtrees_at's among
#            them), then rows.py's device gates at ROWS over GIB; then kv_bench.py --gates-only on that kv.cu
#   offset   refresh_gate.py against refresh_gate.json, the CPU reference's vectors: the re-key's encode in the spare
#            area under 0xA5 poison, written in its range only and out of place, the roots of plans whose pieces read
#            at shifted store indices (rs_pass_at), padded and whole, every kept level written, a no-write control
#            refused, the openings across the pieces' edges (rs_subtrees_at) and the chained seed (rs_chain_head)
#   session  run.sh with SERVE unset and ROLE=both: PLAN's sessions on one registration, REFRESH's word each, the
#            prover's spare area REFRESH_GIB, the verifier's process beside the prover on this machine, which audits
#            nothing: a check of the card's refresh path against the verifier
#
# STAGES (default `base offset session`) picks which run. GIB, PLAN, REFRESH, LIFETIME, WINDOWS and WINDOW_MS are
# run.sh's (defaults 8, honest,honest, 0,1, 8, 7 and 12000: a window holds a class's re-key, N / L blocks decoded and
# encoded again); REFRESH_GIB (default 7) holds the classes windows 1 … WINDOWS − 1 re-key, GIB / LIFETIME each; PROVER_CPUS and VERIFIER_CPUS default to the lease's cores less two, and those two. No stage starts
# after NOT_AFTER (`date -d` syntax). The run is a trial (VERITY_BENCH_TRIAL=1) unless VERITY_BENCH_TRIAL is set
# empty. out/stages.txt has each stage's verdict; exit 0 when every stage passed, else the failed stage's own code.
set -uo pipefail
TREE=$PWD RS=benchmarks/poub_session
O=${RESEARCH_RUN_DIR:?}/out
mkdir -p "$O"
say() { echo "$(date -u +%H:%M:%SZ) $*" | tee -a "$O/stages.txt"; }
[ -n "${GPU_LEASE_UUID:-}" ] || {
  say "refused: refresh_card.sh runs inside the caller's lease of one card (gpu-lease 1 --wait --cores CPUS -- ...)"
  exit 3
}
export PY=${PY:-/workspace/venv312/bin/python} CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.9}
[ -x "$PY" ] || { say "refused: no torch venv at PY=$PY"; exit 3; }
[ -x "$CUDA_HOME/bin/nvcc" ] || { say "refused: CUDA_HOME=$CUDA_HOME has no bin/nvcc"; exit 3; }
export VERITY_BENCH_TRIAL=${VERITY_BENCH_TRIAL-1}
CORES=$(taskset -cp $$ | sed 's/.*: //')
read -r PCORES VCORES < <("$PY" -c 'import os; c = sorted(os.sched_getaffinity(0))
print(",".join(map(str, c[:-2] or c)), ",".join(map(str, c[-2:])))')
export GIB=${GIB:-8} PLAN=${PLAN:-honest,honest} REFRESH=${REFRESH:-0,1} LIFETIME=${LIFETIME:-8}
export WINDOWS=${WINDOWS:-7} WINDOW_MS=${WINDOW_MS:-12000} REFRESH_GIB=${REFRESH_GIB:-7}
export PROVER_CPUS=${PROVER_CPUS:-$PCORES} VERIFIER_CPUS=${VERIFIER_CPUS:-$VCORES}
B=$(mktemp -d /tmp/refresh-card.XXXXXX); trap 'rm -rf "$B"' EXIT
{ id -un; hostname; git -C "$TREE" rev-parse HEAD; echo "lease $GPU_LEASE_UUID cores $CORES: prover $PROVER_CPUS," \
  "verifier $VERIFIER_CPUS"; nvidia-smi --query-gpu=name,uuid,clocks.max.sm,memory.total --format=csv,noheader
} > "$O/machine.txt" 2>&1
STAGES=${STAGES:-base offset session}
failed=
stage() {  # stage NAME CMD...: the stage's verdict into out/stages.txt, or `not run` after a failure
  local name=$1 rc; shift
  if [ -n "$failed" ]; then say "$name: not run (stage $failed failed)"; return 0; fi
  if [[ " $STAGES " != *" $name "* ]]; then say "$name: not run (not in STAGES)"; return 0; fi
  if [ -n "${NOT_AFTER:-}" ] && [ "$(date -u +%s)" -gt "$(date -u -d "$NOT_AFTER" +%s)" ]; then
    say "$name: not run (past NOT_AFTER $NOT_AFTER)"; failed=$name; code=11; return 0
  fi
  say "$name: started"
  "$@"
  rc=$?
  if [ $rc = 0 ]; then say "$name: passed"; else say "$name: FAILED (exit $rc)"; failed=$name; code=$rc; fi
}
base() {
  OUT=$O/base BUILD_DIR=$B STEPS=rows CARDS=0 GIB=${BASE_GIB:-16} bash "$RS/rows.sh" || return $?
  "$PY" verity/protocols/resource_accounting/memory/poub/prover/word_codec/kv_bench.py --lib "p1024-masked=$B/kv.so" \
    --out "$O/base/kv" --gates-only --clock-max-mhz "$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits \
    -i "${GPU_LEASE_UUID%%,*}" | head -1)" > "$O/base/kv_bench.log" 2>&1
  local rc=$?
  tail -n 12 "$O/base/kv_bench.log"
  return $rc
}
offset() {
  [ -f "$B/session.so" ] || { echo "offset reads the base stage's build, and base did not run"; return 3; }
  "$PY" "$RS/refresh_gate.py" --lib "$B/session.so" --kv "$B/kv.so" --out "$O/refresh_gate.json" 2>&1 |
    tee "$O/refresh_gate.log"
}
session() {
  mkdir -p "$O/session"
  RESEARCH_RUN_DIR=$O/session ROLE=both PROVER_ARGS="--refresh-gib $REFRESH_GIB ${PROVER_ARGS:-}" \
    bash "$RS/run.sh" > "$O/session.log" 2>&1
  local rc=$?
  tail -n 30 "$O/session.log"
  return $rc
}
code=0
stage base base
stage offset offset
stage session session
if [ -n "$failed" ]; then say "end: stage $failed failed, exit $code"; else say "end: every stage in STAGES passed"; fi
exit "$code"
