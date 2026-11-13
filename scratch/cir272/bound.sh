#!/usr/bin/env bash
# CIR-272: each block type of scratch/cir272/$CB_TYPES staged bound (benchmarks/served_units/stage.py UNIT N DIR --bind) and
# handed to the Lean verifier with the verifier's own partition object and Program, without the Program, and with another
# Definition's Program; with `session`, one CPU --zk session of it and `verify --zk` on that session the same three ways.
# CIR-259's bound.sh (r20261112-143157-9dd0's inputs, with r20261112-140245-1147's Lean-peak rule), but:
#   - its inputs are this directory of the source tree, not $RESEARCH_RUN_DIR/inputs;
#   - each type runs in its own process group under CB_TYPE_S seconds (default 2400); past it the group is stopped, the type
#     recorded as timed out ($OUT/TAG/timeout) and the loop goes on; a type's failure or refusal never ends the loop;
#   - no type starts once fewer than CB_TYPE_S + 300 seconds of CB_BUDGET_S (default 21000) are left: it is listed in
#     $OUT/not-reached.tsv;
#   - the job's threads are CB_THREADS (default 8): hidden_zk/job.sh's TH and RAYON_NUM_THREADS, and the BLAS pools'.
# CIR-272's second pass adds: a row's fifth column, its own limit in seconds (else CB_TYPE_S), which the budget check uses too;
# the list as run ($OUT/types.tsv, form.py) stages the Boolean version the census lookup names where it refuses the word form
# under --zk; the --zk session's serve and prove get --tables when the stage wrote tables/; each Lean check is stopped past
# CB_STAGE_GB as the stager is.
# $CB_TYPES: TAG<TAB>UNIT<TAB>N<TAB>WORDS[<TAB>LIMIT_S], WORDS of: session (the CPU prover's --zk session), more (the partition
# check also without the partition object, with the same Definition's Program at another count, and with a larger
# population's).
# A type whose Lean check may peak over CB_LEAN_GB (default the job's memory less 4; the peak taken as 4.42 x circuit.txt's
# GiB) is staged and recorded, its checks not run.
# The stager is stopped once its resident memory passes CB_STAGE_GB (default the job's memory); stage.rss holds its samples.
# Every type's records are under $OUT/TAG as it goes, and $OUT/TAG/done holds its end, so a run cut short keeps what it did.
# env: CB_MEM_GB (the job's --mem-gb), CB_PROVE_GB / CB_SERVE_GB (the session's two `ulimit -v`, default 85% and 50% of
# it), CB_CACHE (builds; default $HOME/.cache/verity-a11)
set -uo pipefail
SRC=$(pwd)
unset FLOCK_WORK
export HZ_CONFIGS=cir272 HZ_MEM_GB=${CB_MEM_GB:?} HZ_RUNS=1 HZ_WARM=0
export HZ_CACHE=${CB_CACHE:-$HOME/.cache/verity-a11}
mkdir -p "$HZ_CACHE" 2>/dev/null
[ -w "$HZ_CACHE" ] || unset HZ_CACHE
export PYTHONPATH=$SRC:$SRC/verity/protocols/flock/prover/python:$SRC/adapters/vllm:$SRC/tools/verity/circuit_check/src
source benchmarks/pouw/hidden_zk/job.sh
TH=${CB_THREADS:-8}
export RAYON_NUM_THREADS=$TH OMP_NUM_THREADS=$TH OPENBLAS_NUM_THREADS=$TH MKL_NUM_THREADS=$TH NUMEXPR_NUM_THREADS=$TH
PROVE_GB=${CB_PROVE_GB:-$(( MEM_GB * 85 / 100 ))}
SERVE_GB=${CB_SERVE_GB:-$(( MEM_GB * 50 / 100 ))}
TYPE_S=${CB_TYPE_S:-2400}
BUDGET_S=${CB_BUDGET_S:-21000}
export PATH=$HOME/.elan/bin:$HOME/.cargo/bin:$PATH
IN=$SRC/scratch/cir272
PYM=$SRC/.venv/bin/python
SU=$SRC/benchmarks/served_units
case "$W" in *served-units-time*) echo "scratch $W is under a prove_time.sh directory" >&2; exit 2 ;; esac
setup
log "scratch $W; cache ${HZ_CACHE:-scratch}; threads $TH; type limit ${TYPE_S}s, budget ${BUDGET_S}s; session limits: prove $PROVE_GB GB, serve $SERVE_GB GB; lean $(sha256sum "$LEAN" | cut -c1-16)"
export PATH=$(dirname "$LEAN"):$PATH
LEAN_GB=${CB_LEAN_GB:-$(( MEM_GB - 4 ))}

# job.sh's session, but serve and prove read the MUFU tables staged beside a statement that reads in slots (--tables)
session() {
  local D=$1 O=$2 port pin srv rc try up= tu= tb=
  pin=$(cat "$D/pin.txt")
  [ -L "$D/unit.placed" ] || [ -e "$D/unit.placed" ] && tu="--tmpl-unit $D/unit.placed"
  [ -d "$D/tables" ] && tb="--tables $D/tables"
  for try in 1 2 3 4 5; do
    port=$(free_port)
    $TIME "$O/serve.time" bash -c "$(vlimit $SERVE_GB) exec $BIN serve --zk --listen 127.0.0.1:$port --out $D/sessions \
      --instances $D/pub.bin --circuit $D/circuit.txt $tu $tb --pin $pin --operator loopback --draw subset:$N" > "$O/serve.log" 2>&1 &
    srv=$!
    until grep -q '^SERVING ' "$O/serve.log" 2>/dev/null; do
      kill -0 $srv 2>/dev/null || break
      sleep 1
    done
    grep -q '^SERVING ' "$O/serve.log" 2>/dev/null && { up=1; break; }
    log "  serve exited before SERVING (try $try): $(tail -1 "$O/serve.log")"; wait $srv 2>/dev/null
  done
  [ -n "$up" ] || { log "  serve never came up after $try tries"; return 1; }
  $TIME "$O/prove.time" bash -c "$(vlimit $PROVE_GB) exec $BIN prove --zk --verifier 127.0.0.1:$port --instances $D/inst.bin \
    --circuit $D/circuit.txt $tu $tb --runs $RUNS --warm $WARM" > "$O/prove.out" 2> "$O/prove.err"
  rc=$?
  kill $srv 2>/dev/null; wait $srv 2>/dev/null
  du -sb "$D/sessions"/*/ > "$O/sessions.du" 2>/dev/null
  return $rc
}

stage() {   # stage TAG UNIT N: $W/TAG/B, its record under $OUT/TAG/B
  local d=$W/$1/B o=$OUT/$1/B rc pid kid rss top=0 gb=${CB_STAGE_GB:-$MEM_GB}
  mkdir -p "$d" "$o"
  /usr/bin/time -v -o "$o/stage.time" "$PYM" "$SU/stage.py" "$2" "$3" "$d" --bind > "$o/stage.log" 2> "$o/stage.err" &
  pid=$!
  # the stager's resident memory every 20 s (stage.rss: time, KB resident over the stager and its children, KB staged so
  # far); past the job's own memory it is stopped, so the job stays inside what it asked for
  while kill -0 $pid 2>/dev/null; do
    sleep 20
    kid=$(pgrep -P $pid | head -1); [ -n "$kid" ] || continue
    rss=$(ps -o rss= -p "$kid" --ppid "$kid" 2>/dev/null | awk '{s += $1} END {print s + 0}')
    echo "$(date -u +%T) $rss $(du -sk "$d" 2>/dev/null | cut -f1)" >> "$o/stage.rss"
    [ "$rss" -gt "$top" ] && top=$rss
    if [ "$rss" -gt $(( gb * 1048576 )) ]; then
      echo "stopped: the stager's resident memory, $(( rss / 1048576 )) GB, passed this job's $gb GB" >> "$o/stage.err"
      kill "$kid"
      break
    fi
  done
  wait $pid
  rc=$?
  for f in stage.json pin.txt partition.json; do [ -f "$d/$f" ] && cp "$d/$f" "$o/"; done
  [ -f "$d/program.json" ] && { stat -c %s "$d/program.json" > "$o/program.bytes"; sha512sum "$d/program.json" | cut -d' ' -f1 > "$o/program.file.sha512"
    [ "$(stat -c %s "$d/program.json")" -lt 4000000 ] && cp "$d/program.json" "$o/"; }
  [ -f "$d/pub.bin" ] && head -c 400000 "$d/pub.bin" | head -1 > "$o/pub.head"
  [ -f "$d/circuit.txt" ] && sed -n 2p "$d/circuit.txt" | head -c 2000000 > "$o/META.line"
  ls -l "$d" > "$o/files.txt" 2>/dev/null
  log "$1: stage.py $2 $3 --bind: rc $rc; $(grep -E 'Elapsed|Maximum resident' "$o/stage.time" | sed 's/^\s*//' | paste -sd ';'); $(tail -1 "$o/stage.err" | cut -c1-400)"
  return $rc
}

lean() {    # lean TAG NAME ARGS...: one flock-verify command, its output, exit code and peak under $OUT/TAG/lean
  local o=$OUT/$1/lean n=$2 rc
  shift 2
  mkdir -p "$o"
  /usr/bin/time -v -o "$o/$n.time" "$LEAN" "$@" > "$o/$n.out" 2> "$o/$n.err" &
  local pid=$! kid rss gb=${CB_STAGE_GB:-$MEM_GB}
  # the check's resident memory every 10 s; past CB_STAGE_GB it is stopped, as the stager is
  while kill -0 $pid 2>/dev/null; do
    sleep 10
    kid=$(pgrep -P $pid | head -1); [ -n "$kid" ] || continue
    rss=$(ps -o rss= -p "$kid" 2>/dev/null | awk '{s += $1} END {print s + 0}')
    if [ "$rss" -gt $(( gb * 1048576 )) ]; then
      echo "stopped: the Lean check's resident memory, $(( rss / 1048576 )) GB, passed this job's $gb GB" | tee "$o/$n.stopped" >> "$o/$n.err"
      kill "$kid"
      break
    fi
  done
  wait $pid
  rc=$?
  echo $rc > "$o/$n.rc"
  log "  lean $n: exit $rc; $(grep -E 'Elapsed|Maximum resident' "$o/$n.time" | sed 's/^\s*//' | paste -sd ';'); $(grep -E '^(VERDICT|KS-FORM|setup:|\{)' "$o/$n.out" | tail -1 | cut -c1-900) $(tail -1 "$o/$n.err" | cut -c1-200)"
}

one() {     # one TAG UNIT N WORDS: the type's staging and Lean checks; its exit says how far it got (0: every step ran)
  set +m
  local tag=$1 unit=$2 n=$3 words=$4 T=$W/$1 others S tb B need d o sB
  mkdir -p "$T"
  stage "$tag" "$unit" "$n" || { log "$tag: staging refused: rc $?"; return 3; }
  "$PYM" -c "
import json
r = json.load(open('$T/B/stage.json'))
print('STAGED $tag', json.dumps({k: r.get(k) for k in ('definition', 'word', 'n', 'ands', 'k_log', 'circuit_bytes', 'statement', 'seconds', 'maxrss_gb', 'lookups', 'slot_types')}))" | tee -a "$OUT/job.log"
  # a Lean check's peak: at most 4.42 x circuit.txt's GiB over CIR-272's accepted circuits past 5 GiB (r20261112-215857-32d2,
  # r20261112-215921-f101); job.sh's 6 x + 4 GB is logged beside it
  need=$(awk -v b="$(stat -c %s "$T/B/circuit.txt")" 'BEGIN {g = 4.42 * b / 2^30; print (g == int(g)) ? g : int(g) + 1}')
  if [ "$need" -gt "$LEAN_GB" ]; then
    log "$tag: Lean not run: a check of its $(stat -c %s "$T/B/circuit.txt")-byte circuit.txt may peak at $need GB (4.42 x its GiB; job.sh's rule $(( $(stat -c %s "$T/B/circuit.txt") * 6 / 1024 / 1024 / 1024 + 4 )) GB), over this job's $LEAN_GB GB for Lean"
    echo "$need" > "$OUT/$tag/lean-not-run"
    return 4
  fi
  others="other-def"; [[ " $words " == *" more "* ]] && others="other-def other-count larger"
  CB_OTHERS=$others "$PYM" "$IN/others.py" "$T/B/stage.json" "$T" "$unit" > "$OUT/$tag/others.json" 2> "$OUT/$tag/others.err" || { log "$tag: others FAILED: $(tail -2 "$OUT/$tag/others.err")"; return 5; }
  log "$tag: other Programs: $(cut -c1-600 "$OUT/$tag/others.json")"
  S=$("$PYM" -c "import json; print(json.load(open('$OUT/$tag/B/pub.head'))['statement'])")
  tb=; [ -d "$T/B/tables" ] && tb=" --tables $T/B/tables"
  B="--statement $S --circuit $T/B/circuit.txt --public $T/B/pub.bin$tb"
  lean "$tag" B.statement.held $(echo statement $B) --partition "$T/B/partition.json" --program "$T/B/program.json"
  lean "$tag" B.statement.other-def $(echo statement $B) --partition "$T/B/partition.json" --program "$T/other-def/program.json"
  lean "$tag" B.statement.no-program $(echo statement $B) --partition "$T/B/partition.json"
  if [[ " $words " == *" more "* ]]; then
    lean "$tag" B.statement.no-partition $(echo statement $B)
    lean "$tag" B.statement.other-count $(echo statement $B) --partition "$T/B/partition.json" --program "$T/other-count/program.json"
    lean "$tag" B.statement.other-both $(echo statement $B) --partition "$T/other-count/partition.json" --program "$T/other-count/program.json"
    lean "$tag" B.statement.larger $(echo statement $B) --partition "$T/larger/partition.json" --program "$T/larger/program.json"
  fi
  lean "$tag" B.ks-form $(echo ks-form $B)
  if [[ " $words " == *" session "* ]]; then
    N=$n; d=$T/B; o=$OUT/$tag/B
    if session "$d" "$o"; then
      log "$tag: --zk session: $(cut -c1-200 "$o/prove.time"); serve $(cut -c1-200 "$o/serve.time"); $(grep -E '^(PROOF|RESULT|proved|TIMING)' "$o/prove.out" | tail -2 | cut -c1-300 | tr '\n' ';')"
    else
      log "$tag: --zk session FAILED: $(tail -2 "$o/prove.err" | cut -c1-300 | tr '\n' '|') $(tail -2 "$o/serve.log" | cut -c1-300 | tr '\n' '|')"
    fi
    (cd "$d" && find sessions -type f -printf '%s\t%p\n' | sort -k2) > "$o/session-files.tsv" 2>/dev/null
    sB=$(ls -d "$T/B"/sessions/*/ 2>/dev/null | tail -1)
    if [ -n "$sB" ]; then
      lean "$tag" B.verify.held $(echo verify --zk $B) --partition "$T/B/partition.json" --program "$T/B/program.json" --session "$sB"
      lean "$tag" B.verify.other-def $(echo verify --zk $B) --partition "$T/B/partition.json" --program "$T/other-def/program.json" --session "$sB"
      lean "$tag" B.verify.no-program $(echo verify --zk $B) --partition "$T/B/partition.json" --session "$sB"
    else
      return 6
    fi
  fi
  return 0
}

"$PYM" "$IN/names.py" "$IN/${CB_TYPES:?}" > "$OUT/names.jsonl" 2> "$OUT/names.err"; log "names: rc $?; $(wc -l < "$OUT/names.jsonl") rows"
# the list as run: a census type whose word form the lookup refuses under --zk is staged as the Boolean version it names
"$PYM" "$IN/form.py" "$IN/$CB_TYPES" "$OUT/names.jsonl" "$OUT/form.tsv" > "$OUT/types.tsv" 2> "$OUT/form.err" \
  || { log "form.py FAILED: $(tail -2 "$OUT/form.err")"; exit 1; }
log "form: $(wc -l < "$OUT/types.tsv") types, $(awk -F'\t' '$3 != $2' "$OUT/form.tsv" | wc -l) staged in the lookup's Boolean version"

: > "$OUT/not-reached.tsv"
while IFS=$'\t' read -r tag unit n words lim; do
  [ -n "$tag" ] || continue
  lim=${lim:-$TYPE_S}
  if [ $(( SECONDS + lim + 300 )) -gt "$BUDGET_S" ]; then
    printf '%s\t%s\tnot reached: %ss of the run'"'"'s %ss budget were left, under its %ss limit and 300s\n' \
      "$tag" "$unit" $(( BUDGET_S - SECONDS )) "$BUDGET_S" "$lim" >> "$OUT/not-reached.tsv"
    continue
  fi
  mkdir -p "$OUT/$tag"
  log "$tag: $unit N=$n ($words; limit ${lim}s)"
  t0=$SECONDS
  set -m
  one "$tag" "$unit" "$n" "$words" < /dev/null &
  p=$!
  set +m
  out=
  while kill -0 $p 2>/dev/null; do
    if [ $(( SECONDS - t0 )) -gt "$lim" ]; then
      kill -TERM -- -$p 2>/dev/null; sleep 15; kill -KILL -- -$p 2>/dev/null
      echo "timed out: the type's ${lim}s limit passed at $(( SECONDS - t0 ))s" > "$OUT/$tag/timeout"
      log "$tag: timed out after $(( SECONDS - t0 ))s"
      out=timeout
      break
    fi
    sleep 10
  done
  wait $p; code=$?
  "$PYM" "$IN/verdicts.py" "$OUT/$tag" | tee -a "$OUT/job.log"
  echo "{\"exit\": $code, \"seconds\": $(( SECONDS - t0 )), \"timeout\": $([ "$out" = timeout ] && echo true || echo false)}" > "$OUT/$tag/done"
  log "$tag: done: exit $code in $(( SECONDS - t0 ))s"
  rm -rf "${W:?}/$tag"
done < "$OUT/types.tsv"
log "done; $(wc -l < "$OUT/not-reached.tsv") not reached"
exit 0
