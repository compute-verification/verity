#!/usr/bin/env bash
# flock-circuit's smallest proof, rope-head d64 at 4 instances, on CPU and the card, each honest record given to the Lean verifier;
# exits non-zero when either selftest fails (60-circuit.sh's SELFTEST FAILED line names it).
# usage, from the tree root under research run (OUT), the same tree and machine for both steps:
#   SM=90 [DRV=580] 62-circuit-gpu-smoke.sh prepare   no card: the toolkit (CUDA_TK, else the machine record's 13.3, else 13.3
#                                                     unpacked under SMOKE_WORK, no root), the input set,
#                                                     the prover's GPU build (DRV: the driver's major version, when nvidia-smi can't run)
#                                                     and the Lean verifier, all under SMOKE_WORK (default $HOME/flock-smoke)
#   SM=90 62-circuit-gpu-smoke.sh run                 the card: the selftest build (incremental on prepare's), the CPU and GPU
#                                                     selftests with records (the card's own gpu_proofs_match_cpu case), the Lean verdicts
# SM is the target's architecture as a compute capability, 90 for sm_90 (H100), 120 for sm_120 (RTX PRO 6000); GPU=0: CPU only.
set -uo pipefail
export LEAN_ABORT_ON_PANIC=1  # one_stage/settings.py's ABORT_ON_PANIC: a Lean panic ends the process
STEP=${1:?usage: 62-circuit-gpu-smoke.sh prepare|run}
: "${OUT:?OUT: the output directory of the research run}"
GPU=${GPU:-1}; [ "$GPU" = 1 ] && : "${SM:?SM: the architecture as a compute capability, as in the usage}"
W=${SMOKE_WORK:-$HOME/flock-smoke}; export FLOCK_WORK=$W; mkdir -p "$W"
VER=verity/protocols/proof/flock/verifier
. verity/protocols/proof/spot/verifier/one_stage/verifier_env.sh
case $STEP in
prepare)
  if [ "$GPU" = 1 ]; then
    . benchmarks/one_stage/toolkit.sh
    ls /usr/local | grep '^cuda' | tr '\n' ' ' | sed 's/^/TOOLKITS /'; echo
    cuda_13_3 "$OUT/toolkit.log" || { cat "$OUT/toolkit.log"; exit 3; }
    echo "CUDA_TK=$CUDA_TK"; "$CUDA_TK/bin/nvcc" --version | tail -1; echo "export CUDA_TK=$CUDA_TK" > "$W/env"
  else : > "$W/env"; fi
  uv sync -q --all-packages --extra torch-cpu || exit 4
  rm -rf "$W/gen" "$W/inputs"; mkdir -p "$W/gen" "$W/inputs"
  SET=$(uv run python -c "from verity_flock import subcircuit_templates as TM; from verity_flock.synthetic import generate; print(generate(TM.rope_head(64), 4, 20260926, '$W/gen')['path'])") || exit 5
  tar -cf "$W/inputs/rope.tar" -C "$(dirname "$SET")" "$(basename "$SET")"
  t0=$(date +%s)
  RESEARCH_RUN_DIR=$W/prep MODE=build GPU=$GPU SM=${SM:-} bash verity/protocols/proof/flock/prover/pod/60-circuit.sh > "$OUT/build.log" 2>&1
  rc=$?; echo "PREPARE build rc=$rc $(( $(date +%s) - t0 )) s"; [ $rc = 0 ] || { tail -30 "$OUT/build.log"; exit $rc; }
  (cd $VER && lake build verity-verify-proof 2>&1 | tail -1) && mkdir -p "$W/bin" && cp $VER/.lake/build/bin/verity-verify-proof "$W/bin/" || exit 7
  "$W/bin/verity-verify-proof" --help > /dev/null 2>&1; echo "PREPARE lean verifier rc=$? $W/bin/verity-verify-proof"
  echo "PREPARE done"
  ;;
run)
  [ -f "$W/env" ] && [ -x "$W/bin/verity-verify-proof" ] && [ -f "$W/inputs/rope.tar" ] || { echo "run: prepare first (SMOKE_WORK=$W)"; exit 2; }
  . "$W/env"
  R=$W/run; rm -rf "$R"; mkdir -p "$R"; cp -r "$W/inputs" "$R/"
  t0=$(date +%s)
  RESEARCH_RUN_DIR=$R MODE=selftest GPU=$GPU SM=${SM:-} RECORD=1 SELF_N=4 bash verity/protocols/proof/flock/prover/pod/60-circuit.sh > "$OUT/selftest.log" 2>&1
  rc=$?; t1=$(date +%s); echo "SMOKE build+selftest rc=$rc $((t1 - t0)) s"; tail -15 "$OUT/selftest.log"
  D=$R/out/rope; cp -r "$R/out" "$OUT/flock-out"
  for dev in cpu gpu; do
    [ -d "$D/records-$dev" ] || continue
    python3 verity/protocols/proof/flock/prover/agreement/selftest_records.py "$D/records-$dev" "$R/s-$dev" > /dev/null || exit 6
    # a selftest record is never accepted: the honest verdict is a scope refusal, reached once every check has passed
    $(verifier_env) "$W/bin/verity-verify-proof" verify --statement verity/flock-circuit+seed-injection --circuit "$D/stage/circuit.txt" --public "$D/stage/pub-4.bin" \
      $( [ -d "$D/stage/tables" ] && echo --tables "$D/stage/tables" ) --session "$R/s-$dev/honest" | grep '^VERDICT ' | sed "s/^/LEAN-$dev /"
  done
  grep -h '"case":"gpu_proofs_match_cpu"' "$D/selftest-gpu.txt" 2>/dev/null | cut -c1-300 | sed 's/^/MATCH /'
  echo "SMOKE minutes: build+selftest $(( (t1 - t0) / 60 )), lean verify $(( ($(date +%s) - t1) / 60 ))"
  exit $rc
  ;;
*) echo "usage: 62-circuit-gpu-smoke.sh prepare|run"; exit 2 ;;
esac
