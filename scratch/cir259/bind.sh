#!/usr/bin/env bash
# One statement type of benchmarks/served_units/stage.py staged bound (--bind) and unbound at the same N, compared, and
# the Lean verifier's partition and form checks on both: with the verifier's own partition object and Program, without
# them, and with another Program handed in. Optionally one CPU --zk session of each and `verify --zk` on it.
# inputs/types.tsv: TAG<TAB>UNIT<TAB>N<TAB>WORDS, N a count or "g" (the rebound statement's count), WORDS of: rebound
# (stage `full/2^30 --bind` first, the statement rebound in place, and hold its pin to the directly bound one's when the
# counts agree), session (the CPU prover's --zk session and `verify --zk`; a flat statement without tables only),
# public or packed (a UNITS type's --public-weights, --public-weights --packed, on both).
# env: CB_MEM_GB (the job's --mem-gb), CB_CACHE (builds; default $HOME/.cache/verity-a11)
set -uo pipefail
SRC=$(pwd)
export HZ_CONFIGS=cir245 HZ_MEM_GB=${CB_MEM_GB:?} HZ_RUNS=1 HZ_WARM=0
export HZ_CACHE=${CB_CACHE:-$HOME/.cache/verity-a11}
mkdir -p "$HZ_CACHE" 2>/dev/null
[ -w "$HZ_CACHE" ] || unset HZ_CACHE
export PYTHONPATH=$SRC:$SRC/verity/protocols/flock/prover/python:$SRC/adapters/vllm:$SRC/tools/verity/circuit_check/src
source benchmarks/pouw/hidden_zk/job.sh
export PATH=$HOME/.elan/bin:$HOME/.cargo/bin:$PATH
IN=${RESEARCH_RUN_DIR:?}/inputs
PYM=$SRC/.venv/bin/python
SU=$SRC/benchmarks/served_units
setup
log "cache ${HZ_CACHE:-scratch}; lean $(sha256sum "$LEAN" | cut -c1-16)"

export PATH=$(dirname "$LEAN"):$PATH   # flock-read-tmpl, the template a table read is staged on, is built beside flock-verify
log "flock-read-tmpl: $(command -v flock-read-tmpl || echo none)"

stage() {   # stage TAG WHAT UNIT N [FLAG]: $W/TAG/WHAT, its record under $OUT/TAG/WHAT
  local d=$W/$1/$2 o=$OUT/$1/$2 rc
  mkdir -p "$d" "$o"
  /usr/bin/time -v -o "$o/stage.time" "$PYM" "$SU/stage.py" "$3" "$4" "$d" ${5:-} > "$o/stage.log" 2> "$o/stage.err"
  rc=$?
  for f in stage.json pin.txt partition.json; do [ -f "$d/$f" ] && cp "$d/$f" "$o/"; done
  [ -f "$d/program.json" ] && { stat -c %s "$d/program.json" > "$o/program.bytes"; sha512sum "$d/program.json" | cut -d' ' -f1 > "$o/program.file.sha512"
    [ "$(stat -c %s "$d/program.json")" -lt 4000000 ] && cp "$d/program.json" "$o/"; }
  [ -f "$d/pub.bin" ] && head -c 400000 "$d/pub.bin" | head -1 > "$o/pub.head"
  [ -f "$d/inst.bin" ] && head -c 400000 "$d/inst.bin" | head -1 > "$o/inst.head"
  [ -f "$d/circuit.txt" ] && sed -n 2p "$d/circuit.txt" | head -c 2000000 > "$o/META.line"
  log "$1/$2: stage.py $3 $4 ${5:-}: rc $rc; $(grep -E 'Elapsed|Maximum resident' "$o/stage.time" | sed 's/^\s*//' | paste -sd ';'); $(tail -1 "$o/stage.err" | cut -c1-300)"
  return $rc
}

lean() {    # lean TAG NAME ARGS...: one flock-verify command, its output, exit code and peak under $OUT/TAG/lean
  local o=$OUT/$1/lean n=$2 rc
  shift 2
  mkdir -p "$o"
  /usr/bin/time -v -o "$o/$n.time" "$LEAN" "$@" > "$o/$n.out" 2> "$o/$n.err"
  rc=$?
  echo $rc > "$o/$n.rc"
  log "  lean $n: exit $rc; $(grep -E 'Elapsed|Maximum resident' "$o/$n.time" | sed 's/^\s*//' | paste -sd ';'); $(grep -E '^(VERDICT|KS-FORM|setup:|\{)' "$o/$n.out" | tail -1 | cut -c1-900) $(tail -1 "$o/$n.err" | cut -c1-200)"
}

rc=0
while IFS=$'\t' read -r tag unit n words; do
  [ -n "$tag" ] || continue
  T=$W/$tag; mkdir -p "$T" "$OUT/$tag"
  log "$tag: $unit N=$n ($words)"
  extra=
  [[ " $words " == *" public "* ]] && extra="--public-weights"
  [[ " $words " == *" packed "* ]] && extra="--public-weights --packed"
  if [[ " $words " == *" rebound "* ]]; then
    stage "$tag" R "$unit" full/1073741824 --bind || { rc=1; continue; }
    nR=$("$PYM" -c "import json; print(json.load(open('$T/R/stage.json'))['n'])")
    [ "$n" = g ] && n=$nR
  fi
  stage "$tag" B "$unit" "$n" "--bind $extra"; rB=$?
  stage "$tag" U "$unit" "$n" "$extra"; rU=$?
  [ $rB = 0 ] && [ $rU = 0 ] || { log "$tag: staging refused: bound rc $rB, unbound rc $rU"; rc=1; rm -rf "${T:?}"; continue; }
  "$PYM" "$IN/compare.py" "$T" "$OUT/$tag/compare.json" 2> "$OUT/$tag/compare.err" | tee -a "$OUT/job.log" || { rc=1; log "$tag: compare FAILED: $(tail -2 "$OUT/$tag/compare.err")"; }
  "$PYM" "$IN/others.py" "$T/B/stage.json" "$T" "$unit" > "$OUT/$tag/others.json" 2> "$OUT/$tag/others.err" || { rc=1; log "$tag: others FAILED: $(tail -2 "$OUT/$tag/others.err")"; continue; }
  log "$tag: other Programs: $(cut -c1-600 "$OUT/$tag/others.json")"
  S=$("$PYM" -c "import json; print(json.load(open('$OUT/$tag/B/pub.head'))['statement'])")
  log "$tag: statement $S"
  tb=; [ -d "$T/B/tables" ] && tb=" --tables $T/B/tables"
  B="--statement $S --circuit $T/B/circuit.txt --public $T/B/pub.bin$tb" U="--statement $S --circuit $T/U/circuit.txt --public $T/U/pub.bin$tb"
  lean "$tag" B.statement.held $(echo statement $B) --partition "$T/B/partition.json" --program "$T/B/program.json"
  lean "$tag" B.statement.no-program $(echo statement $B) --partition "$T/B/partition.json"
  lean "$tag" B.statement.no-partition $(echo statement $B)
  lean "$tag" B.statement.other-count $(echo statement $B) --partition "$T/B/partition.json" --program "$T/other-count/program.json"
  lean "$tag" B.statement.other-def $(echo statement $B) --partition "$T/B/partition.json" --program "$T/other-def/program.json"
  lean "$tag" B.statement.other-both $(echo statement $B) --partition "$T/other-count/partition.json" --program "$T/other-count/program.json"
  lean "$tag" B.statement.larger $(echo statement $B) --partition "$T/larger/partition.json" --program "$T/larger/program.json"
  lean "$tag" U.statement $(echo statement $U)
  lean "$tag" U.statement.program $(echo statement $U) --program "$T/B/program.json"
  lean "$tag" U.statement.held $(echo statement $U) --partition "$T/B/partition.json" --program "$T/B/program.json"
  [ -d "$T/R" ] && lean "$tag" R.statement.held statement --statement $S --circuit "$T/R/circuit.txt" --public "$T/R/pub.bin" $tb --partition "$T/R/partition.json" --program "$T/R/program.json"
  lean "$tag" B.ks-form $(echo ks-form $B)
  lean "$tag" U.ks-form $(echo ks-form $U)
  if [[ " $words " == *" session "* ]]; then
    N=$n
    for w in B U; do
      d=$T/$w; o=$OUT/$tag/$w
      if session "$d" "$o"; then
        log "$tag/$w: --zk session: $(cut -c1-200 "$o/prove.time"); $(grep -E '^(PROOF|RESULT|proved|TIMING)' "$o/prove.out" | tail -2 | cut -c1-300 | tr '\n' ';')"
      else
        log "$tag/$w: --zk session FAILED: $(tail -2 "$o/prove.err" | cut -c1-300 | tr '\n' '|') $(tail -2 "$o/serve.log" | cut -c1-300 | tr '\n' '|')"; rc=1
      fi
      (cd "$d" && find sessions -type f -printf '%s\t%p\n' | sort -k2) > "$o/session-files.tsv" 2>/dev/null
    done
    sB=$(ls -d "$T/B"/sessions/*/ 2>/dev/null | tail -1); sU=$(ls -d "$T/U"/sessions/*/ 2>/dev/null | tail -1)
    if [ -n "$sB" ]; then
      lean "$tag" B.verify.held $(echo verify --zk $B) --partition "$T/B/partition.json" --program "$T/B/program.json" --session "$sB"
      lean "$tag" B.verify.no-program $(echo verify --zk $B) --partition "$T/B/partition.json" --session "$sB"
      lean "$tag" B.verify.other-count $(echo verify --zk $B) --partition "$T/B/partition.json" --program "$T/other-count/program.json" --session "$sB"
      lean "$tag" B.verify.other-def $(echo verify --zk $B) --partition "$T/B/partition.json" --program "$T/other-def/program.json" --session "$sB"
      lean "$tag" B.verify.other-both $(echo verify --zk $B) --partition "$T/other-count/partition.json" --program "$T/other-count/program.json" --session "$sB"
    fi
    if [ -n "$sU" ]; then
      lean "$tag" U.verify.no-program $(echo verify --zk $U) --session "$sU"
      lean "$tag" U.verify.program $(echo verify --zk $U) --program "$T/B/program.json" --session "$sU"
      lean "$tag" U.verify.other-def $(echo verify --zk $U) --program "$T/other-def/program.json" --session "$sU"
      lean "$tag" U.verify.held $(echo verify --zk $U) --partition "$T/B/partition.json" --program "$T/B/program.json" --session "$sU"
    fi
  fi
  "$PYM" "$IN/verdicts.py" "$OUT/$tag" | tee -a "$OUT/job.log"
  rm -rf "${T:?}"
done < "$IN/${CB_TYPES:-types.tsv}"
log "done, rc $rc"
exit $rc
