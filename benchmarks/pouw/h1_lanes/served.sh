#!/usr/bin/env bash
# One unit a served H1 window's own draw selected, proved as its 32 lanes, each a C-Flock --zk statement under the Lean
# firewall, and verified by the Lean verifier against the window's registered roots (served.py's docstring).
#   research run --on NODE --project verity --campaign C --source . --cwd source --mem-gb G --timeout 600m \
#     --declared-output 'out/*' --env H1L_MEM_GB=G --env H1S_CALLS=... [...] -- bash benchmarks/pouw/h1_lanes/served.sh
#   1. hidden_zk/job.sh's builds and the firewall's Lean frame (flock-firewall)
#   2. the drawn units (h1_prove/served.py: core's draw on the window's signed coin) and served.py extract: the drawn unit's
#      openings from the prover's saved window, into the job's scratch (private), never out/
#   3. served.py prepare: the lane statements and the negatives staged (WORK/plan.tsv)
#   4. one loopback --zk session per statement, H1L_PAR at a time, each lane's prover under the firewall (FC_FIREWALL=1:
#      the Lean session loop, its level-0 draw and Lean finish) unless H1S_FIREWALL=0; the negatives first, without it
#      (H1S_UNFIREWALLED: what they check is the verifier's refusal)
#   5. the plan's Lean checks, each with the lane Program, its partition, its public file and registered reads
#   6. served.py summarize: out/summary.json
# env: H1L_MEM_GB (the job's --mem-gb)
#      the window: H1S_CALLS (pouw-h1-calls.json)  H1S_SESSION (the gate's fetched session out/)  H1S_LEDGER (ledger heads,
#      space-separated)  H1S_PIN (the challenger key's public_sha512)  H1S_DESCRIPTOR (window-program.json.gz)  H1S_STATE
#      (the prover's h1-window.pkl)  H1S_PY (the H1 venv's python, which unpickles it)
#      or H1S_SMOKE=K: a stand-in window at K (served.py smoke), the harness's own check
#      or H1S_OPENED (an extract's private openings: one node's share of the lanes, H1L_LANES, the negatives skipped
#      with H1S_SKIP=tampered-zy,other-call-a); H1S_EXTRACT_ONLY=FILE: keep the extract's openings there (0600) and stop
#      H1L_K (4096)  H1L_PAR (8: sessions at a time)  H1L_LPAR (8: Lean checks at a time)  H1L_JOBS (8: prepare's stagers)
#      H1L_LANES (all, or 0,5,...)  H1S_FIREWALL (1)  H1S_UNFIREWALLED ("tampered-zy other-call-a")  H1S_KEEP (tags whose sessions out/ keeps: "lane0 tampered-zy")
#      H1S_FIREWALL_TGZ (`git archive --format=tar.gz REV verity/core/service/flock`: the firewall frame built from REV's
#      Lean, which must speak this tree's prover protocol; the verifier stays this tree's)
set -uo pipefail
export HZ_CONFIGS=h1-served HZ_MEM_GB=${H1L_MEM_GB:?} HZ_RUNS=1 HZ_WARM=0
export HZ_CACHE=${HZ_CACHE:-/workspace/research/cache/pouw-hidden-zk}
# shellcheck source=../hidden_zk/job.sh
. benchmarks/pouw/hidden_zk/job.sh
H=$SRC/benchmarks/pouw/h1_lanes
export PYTHONPATH=$SRC:$SRC/verity/ml/flock/python:$SRC/benchmarks/pouw/hidden_zk:$SRC/benchmarks/pouw/h1_prove
PAR=${H1L_PAR:-8} LPAR=${H1L_LPAR:-8} K=${H1L_K:-4096} FW=${H1S_FIREWALL:-1}
WK=$W/work OP=$W/opened.json
mkdir -p "$OUT/lean" "$WK"
setup
if [ "$FW" = 1 ] && [ -n "${H1S_FIREWALL_TGZ:-}" ]; then
  FD=$HZ_CACHE/firewall-$(sha256sum "$H1S_FIREWALL_TGZ" | cut -c1-16)
  export FC_FIREWALL_LEAN=$FD/.lake/build/bin/flock-firewall
  log "firewall frame from $H1S_FIREWALL_TGZ ($(sha256sum "$H1S_FIREWALL_TGZ" | cut -c1-16)), not this tree's"
  (exec 8>"$FD.lock"; flock 8; [ -x "$FC_FIREWALL_LEAN" ] || { rm -rf "$FD" && mkdir -p "$FD" \
    && tar -xzf "$H1S_FIREWALL_TGZ" -C "$FD" --strip-components=4 && cd "$FD" && lake build flock-firewall; }) \
    > "$OUT/build-firewall.log" 2>&1
elif [ "$FW" = 1 ]; then
  export FC_FIREWALL_LEAN=${LEAN%/*}/flock-firewall
  (exec 8>"${LEAN%/.lake/*}.firewall.lock"; flock 8; [ -x "$FC_FIREWALL_LEAN" ] || (cd "${LEAN%/.lake/*}" && lake build flock-firewall)) \
    > "$OUT/build-firewall.log" 2>&1
fi
if [ "$FW" = 1 ]; then
  [ -x "$FC_FIREWALL_LEAN" ] || { tail -20 "$OUT/build-firewall.log"; log "no firewall frame at $FC_FIREWALL_LEAN"; exit 1; }
  sha256sum "$FC_FIREWALL_LEAN" | tee -a "$OUT/binaries.sha256" | tee -a "$OUT/job.log"
fi

# ---- the drawn unit's openings ---------------------------------------------------------------------------------------------
if [ -n "${H1S_SMOKE:-}" ]; then
  K=$H1S_SMOKE
  log "h1-served: SMOKE, a stand-in window at K=$K (the harness's own check, no served window)"
  "$PY" "$H/served.py" smoke --k "$K" --out "$OP" > "$OUT/extract.out" 2> "$OUT/extract.err" || { log "smoke FAILED: $(tail -3 "$OUT/extract.err")"; exit 1; }
elif [ -n "${H1S_OPENED:-}" ]; then
  # an extract's private openings, moved into the scratch: a copy left in the run dir would be uploaded with its record
  install -m 600 "$H1S_OPENED" "$OP" || exit 1
  case $H1S_OPENED in "$RESEARCH_RUN_DIR"/*) rm -f "$H1S_OPENED" \
    "${RESEARCH_RUN_DIR%/runs/*}/requests/${RESEARCH_RUN_DIR##*/}/inputs/${H1S_OPENED##*/}" ;; esac
  log "h1-served: the openings of $(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['window'][:16], 'unit', d['unit']['unit'], 'K', d['k'])" "$OP")"
else
  HPP=$SRC:$SRC/integrations/vllm:$SRC/verity/ml/flock/python:$SRC/benchmarks/pouw/h1_prove
  ledgers=()
  for l in ${H1S_LEDGER:?}; do ledgers+=(--ledger "$l"); done
  PYTHONPATH=$HPP "${H1S_PY:?}" "$SRC/benchmarks/pouw/h1_prove/served.py" --calls "${H1S_CALLS:?}" --session "${H1S_SESSION:?}" \
    "${ledgers[@]}" --pin "${H1S_PIN:?}" --descriptor "${H1S_DESCRIPTOR:?}" --out "$OUT/drawn.json" > "$OUT/drawn.out" 2> "$OUT/drawn.err" \
    || { log "draw FAILED: $(tail -3 "$OUT/drawn.err") $(tail -3 "$OUT/drawn.out")"; exit 1; }
  log "drawn: $(head -c 1500 "$OUT/drawn.out")"
  PYTHONPATH=$HPP $TIME "$OUT/extract.time" "$H1S_PY" "$H/served.py" extract --calls "$H1S_CALLS" --drawn "$OUT/drawn.json" \
    --descriptor "$H1S_DESCRIPTOR" --state "${H1S_STATE:?}" --k "$K" --out "$OP" --public "$OUT/extract.json" \
    > "$OUT/extract.out" 2> "$OUT/extract.err" || { log "extract FAILED: $(tail -3 "$OUT/extract.err")"; exit 1; }
  log "extract: $(tail -1 "$OUT/extract.out" | cut -c1-600) $(cat "$OUT/extract.time")"
  if [ -n "${H1S_EXTRACT_ONLY:-}" ]; then
    install -m 600 "$OP" "$H1S_EXTRACT_ONLY" && log "openings kept at $H1S_EXTRACT_ONLY (private), sha256 $(sha256sum < "$OP" | cut -c1-64)"
    exit $?
  fi
fi

# ---- the statements ------------------------------------------------------------------------------------------------------
log "h1-served: K=$K tile=${H1_TILE:-v1} firewall $FW, sessions $PAR at a time, Lean $LPAR at a time"
$TIME "$OUT/prepare.time" "$PY" "$H/served.py" prepare --opened "$OP" --work "$WK" --jobs "${H1L_JOBS:-8}" \
  --lanes "${H1L_LANES:-all}" --skip "${H1S_SKIP:-}" --name "h1-served-${RESEARCH_RUN_ID:-local}" > "$OUT/prepare.out" 2> "$OUT/prepare.err"
rc=$?
tail -4 "$OUT/prepare.out" | cut -c1-2000 | tee -a "$OUT/job.log"
cp "$WK/plan.json" "$WK/prepare.json" "$OUT/" 2>/dev/null
[ $rc = 0 ] || { log "prepare FAILED (exit $rc): $(tail -5 "$OUT/prepare.err")"; exit 1; }
log "prepare: $(cat "$OUT/prepare.time")"
mkdir -p "$OUT/verifier" && cp "$WK"/verifier/* "$OUT/verifier/"

# ---- proofs: one --zk session per statement ------------------------------------------------------------------------------
VLIMIT=$(declare -f vlimit)
if [ "$FW" = 1 ]; then
  export FC_FIREWALL=1 FC_HOST_WITNESS=1 FC_FIREWALL_STATEMENT=verity/flock-circuit
  export FC_FIREWALL_HELLO_ARGS="--partition $WK/verifier/lane.partition.json --program $WK/verifier/lane.program.json"
  # the firewall's Lean processes are the prover's children, and under an address-space cap (ulimit -v) their runtime
  # can't create its threads: a firewalled session is bounded by the run's cgroup (--mem-gb) alone
  vlimit() { :; }
fi
N=1
export RAYON_NUM_THREADS=$(( TH / PAR > 0 ? TH / PAR : 1 ))
UNFW=" ${H1S_UNFIREWALLED:-tampered-zy other-call-a} "
while IFS=$'\t' read -r kind tag D; do
  [ "$kind" = S ] || continue
  mkdir -p "$OUT/$tag"
  while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do wait -n; done
  if [[ $UNFW == *" $tag "* ]]; then fw=0; else fw=$FW; fi
  log "prove $tag (firewall $fw)"
  ( [ "$fw" = 1 ] || unset FC_FIREWALL FC_HOST_WITNESS FC_FIREWALL_STATEMENT FC_FIREWALL_HELLO_ARGS
    session "$D" "$OUT/$tag" && log "proved $tag: $(cat "$OUT/$tag/prove.time")" \
      || log "prove $tag FAILED: $(tail -2 "$OUT/$tag/prove.err")" ) &
done < <(awk -F'\t' -v u="$UNFW" '$1 == "S" && index(u, " " $2 " ")' "$WK/plan.tsv"
         awk -F'\t' -v u="$UNFW" '$1 == "S" && !index(u, " " $2 " ")' "$WK/plan.tsv")
wait
unset FC_FIREWALL FC_HOST_WITNESS FC_FIREWALL_STATEMENT FC_FIREWALL_HELLO_ARGS
eval "$VLIMIT"
log "proved: $(ls -d "$WK"/st/*/sessions/*/ 2>/dev/null | wc -l) sessions"

# ---- Lean: the plan's checks -----------------------------------------------------------------------------------------------
cap=$(( STAGE_GB / LPAR ))
while IFS=$'\t' read -r kind tag D pub reg prog part _expect; do
  [ "$kind" = C ] || continue
  s=$(ls -d "$D"/sessions/*/ 2>/dev/null | tr '\n' ' ')
  [ -n "$s" ] || { echo "no session under $D" > "$OUT/lean/$tag.err"; continue; }
  while [ "$(jobs -rp | wc -l)" -ge "$LPAR" ]; do wait -n; done
  $TIME "$OUT/lean/$tag.time" bash -c "$(vlimit $cap) exec $LEAN verify --zk --circuit $D/circuit.txt \
    --statement verity/flock-circuit --public $pub --partition $part --program $prog --registered $reg --session $s" \
    > "$OUT/lean/$tag.out" 2> "$OUT/lean/$tag.err" &
done < "$WK/plan.tsv"
wait
for tag in ${H1S_KEEP:-lane0 tampered-zy}; do
  [ -d "$WK/st/$tag/sessions" ] && cp -r "$WK/st/$tag/sessions" "$OUT/$tag/" && cp "$WK/st/$tag/pin.txt" "$OUT/$tag/"
done

"$PY" "$H/served.py" summarize --work "$WK" --out "$OUT" 2>&1 | tee -a "$OUT/job.log"
rc=${PIPESTATUS[0]}
log "h1-served: done (exit $rc)"
exit "$rc"
