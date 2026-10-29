#!/usr/bin/env bash
# A served H1 window's live units (VLLM-85's live@<s>) through sampled proofs: each round's statement proved by C-Flock --zk
# and verified by the Lean verifier against the verifier's own live Program, partition and registered reads (live.py's
# docstring has the checks). On a node that holds the window's dump (or reads it, --send) (`pouw_h1_gpu --commit --live-dump DIR`):
#   research run --on vy-nebius-cpu-5 --project verity --campaign pouw --source . --cwd source --mem-gb 128 \
#     --declared-output 'out/*' --env H1_LIVE=/workspace/research/runs/RUN/prover -- bash benchmarks/pouw/h1_prove/live.sh
#   1. hidden_zk/job.sh's builds (flock-circuit, the Lean flock-verify, a venv)
#   2. per window under H1_LIVE (each directory holding a window.json): live.py prepare (the verifier's files, the
#      openings checked, each round staged), one loopback --zk session per S line of its plan, each prover under the
#      firewall unless H1_FIREWALL=0, the Lean check of each C line, and live.py summarize: out/<window>/live-summary.json;
#      then the honest rounds' circuit, public files and sessions copied to H1_KEEP/<window> (default the run dir's
#      live-proofs/, outside out/) and live.py evidence over them: out/<window>/live-evidence.json, credit's live evidence
# env: H1_LIVE (a dump DIR, or one window's, or a .tgz of either: --send it, then H1_LIVE=inputs/NAME.tgz; relative: the
#      run dir's)  H1_MEM_GB (128, the job's --mem-gb)  H1_PAR (sessions and Lean checks at a
#      time; H1_MEM_GB / 60)  H1_FIREWALL (1)  HZ_CACHE: hidden_zk's
set -uo pipefail
export HZ_CONFIGS=live HZ_MEM_GB=${H1_MEM_GB:-128} HZ_RUNS=${HZ_RUNS:-1} HZ_WARM=${HZ_WARM:-0}
export HZ_CACHE=${HZ_CACHE:-/workspace/research/cache/pouw-hidden-zk}
# shellcheck source=../hidden_zk/job.sh
. benchmarks/pouw/hidden_zk/job.sh
export PYTHONPATH=$SRC:$SRC/verity/ml/flock/python:$SRC/integrations/vllm
H=$SRC/benchmarks/pouw/h1_prove
LIVE=${H1_LIVE:?}
case $LIVE in /*) ;; *) LIVE=$RESEARCH_RUN_DIR/$LIVE ;; esac
[ -f "$LIVE" ] && { mkdir -p "$W/live" && tar -xzf "$LIVE" -C "$W/live" && LIVE=$W/live; }
setup
FW=${H1_FIREWALL:-1}
if [ "$FW" = 1 ]; then
  export FC_FIREWALL_LEAN=${LEAN%/*}/flock-firewall
  (exec 8>"${LEAN%/.lake/*}.firewall.lock"; flock 8; [ -x "$FC_FIREWALL_LEAN" ] || (cd "${LEAN%/.lake/*}" && lake build flock-firewall)) \
    > "$OUT/build-firewall.log" 2>&1
  [ -x "$FC_FIREWALL_LEAN" ] || { tail -20 "$OUT/build-firewall.log"; log "no firewall frame beside $LEAN"; exit 1; }
  sha256sum "$FC_FIREWALL_LEAN" | tee -a "$OUT/binaries.sha256" | tee -a "$OUT/job.log"
fi
KEEP=${H1_KEEP:-$RESEARCH_RUN_DIR/live-proofs}
VLIMIT=$(declare -f vlimit)
rc=0
mapfile -t WINDOWS < <(find "$LIVE" -name window.json -printf '%h\n' | sort)
[ ${#WINDOWS[@]} -gt 0 ] || { log "no window.json under $LIVE"; exit 1; }
for L in "${WINDOWS[@]}"; do
  name=$(basename "$L") WK=$W/$name O=$OUT/$name
  mkdir -p "$WK" "$O/lean"
  log "live $name: prepare"
  "$PY" "$H/live.py" prepare --live "$L" --work "$WK" 2>&1 | tee -a "$OUT/job.log"
  [ "${PIPESTATUS[0]}" = 0 ] || { log "live $name: prepare FAILED"; cp "$WK/record.json" "$O/" 2>/dev/null; rc=1; continue; }
  cp "$WK/record.json" "$WK/plan.tsv" "$O/" && mkdir -p "$O/verifier" && cp "$WK"/verifier/* "$O/verifier/"
  if [ "$FW" = 1 ]; then
    export FC_FIREWALL=1 FC_HOST_WITNESS=1 FC_FIREWALL_STATEMENT=verity/flock-circuit
    export FC_FIREWALL_HELLO_ARGS="--partition $WK/verifier/partition.json --program $WK/verifier/program.json"
    vlimit() { :; }   # the firewall's Lean children can't start their threads under `ulimit -v` (job.sh)
  fi
  # each statement is one instance of a few thousand ANDs, padded to k_log 26 by --zk: a session (serve, prove and the
  # firewall's Lean) peaks near 55 GB and takes about 20 min on 32 CPUs, so they run PAR at a time under the run's cgroup
  PAR=${H1_PAR:-$(( MEM_GB / 60 ))}; [ "$PAR" -lt 1 ] && PAR=1
  log "live $name: prove, $PAR sessions at a time"
  while IFS=$'\t' read -r kind tag D n; do
    [ "$kind" = S ] || continue
    ln -sf "$WK/circuit.txt" "$D/circuit.txt"
    mkdir -p "$O/$tag"
    while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do wait -n; done
    ( N=$n; session "$D" "$O/$tag" || log "live $name: prove $tag FAILED: $(tail -2 "$O/$tag/prove.err")" ) &
  done < "$WK/plan.tsv"
  wait
  unset FC_FIREWALL FC_HOST_WITNESS FC_FIREWALL_STATEMENT FC_FIREWALL_HELLO_ARGS
  eval "$VLIMIT"
  BOUND="--partition $WK/verifier/partition.json --program $WK/verifier/program.json"
  cap=$(( STAGE_GB / PAR ))
  while IFS=$'\t' read -r kind tag D pub reg _expect; do
    [ "$kind" = C ] || continue
    s=$(ls -d "$D"/sessions/*/ 2>/dev/null | tr '\n' ' ')
    [ -n "$s" ] || { echo "no session under $D" > "$O/lean/$tag.err"; continue; }
    while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do wait -n; done
    $TIME "$O/lean/$tag.time" bash -c "$(vlimit $cap) exec $LEAN verify --zk --circuit $WK/circuit.txt \
      --statement verity/flock-circuit --public $pub $BOUND --registered $reg --session $s" > "$O/lean/$tag.out" 2> "$O/lean/$tag.err" &
  done < "$WK/plan.tsv"
  wait
  "$PY" "$H/live.py" summarize --work "$WK" --out "$O" 2>&1 | tee -a "$OUT/job.log"
  [ "${PIPESTATUS[0]}" = 0 ] || rc=1
  # the honest rounds' proof files outlive W: KEEP/<window> holds them, and live-evidence.json names them for credit
  K=$KEEP/$name
  mkdir -p "$K/verifier" "$K/st"
  cp "$WK/circuit.txt" "$WK/record.json" "$K/" && cp "$WK"/verifier/* "$K/verifier/"
  for st in "$WK"/st/honest-*; do [ -d "$st/sessions" ] && mkdir -p "$K/st/${st##*/}" && cp -r "$st/sessions" "$K/st/${st##*/}/"; done
  "$PY" "$H/live.py" evidence --work "$K" --out "$O/live-evidence.json" 2>&1 | tee -a "$OUT/job.log"
  [ "${PIPESTATUS[0]}" = 0 ] || rc=1
done
log "h1-live: done (exit $rc)"
exit "$rc"
