#!/usr/bin/env bash
# CIR-259 part 2's checks of the bound default, from the source tree's root (`--cwd source`, after `uv sync`):
#   1. pins: each PR_PIN_TYPES row (TAG<TAB>UNIT<TAB>N) staged four ways on this tree's libraries: stage.py --unbound (U),
#      origin/main's stage.py with no flag (M, inputs/stage_main.py), 31b1b56's with no flag (O, inputs/stage_31b1b56.py)
#      and stage.py with no flag (B); U, M and O must give one circuit.txt (pin.txt, its SHA-512) and the same inst.bin and
#      pub.bin, and B the same class_sha512 with its own pin, partition.json and program.json
#   2. SiluMul_v2{I=256} at N=2 staged by the default (bound) and by --unbound, each with a CPU --zk session; Lean's
#      statement and verify --zk with its own partition and Program, another Definition's Program and no Program (bound),
#      and with no Program (unbound)
#   3. served_units/job.sh on SU_UNITS at N=2 (bound): summarize.py's verdicts
#   4. zk_cost.py stage, prove --cpu (fw arm, one warm process) and verify on ZK_UNIT at N=2
# env: PR_MEM_GB (the job's --mem-gb)
set -uo pipefail
SRC=$(pwd)
unset FLOCK_WORK
export HZ_CONFIGS=cir259-parts HZ_MEM_GB=${PR_MEM_GB:?} HZ_RUNS=1 HZ_WARM=0
export HZ_CACHE=${PR_CACHE:-$HOME/.cache/verity-a11}
mkdir -p "$HZ_CACHE" 2>/dev/null
[ -w "$HZ_CACHE" ] || unset HZ_CACHE
export PYTHONPATH=$SRC:$SRC/verity/protocols/flock/prover/python:$SRC/adapters/vllm:$SRC/tools/verity/circuit_check/src:$SRC/benchmarks/numerical/python
source benchmarks/pouw/hidden_zk/job.sh
PROVE_GB=$(( MEM_GB * 85 / 100 )); SERVE_GB=$(( MEM_GB * 50 / 100 ))
export PATH=$HOME/.elan/bin:$HOME/.cargo/bin:$PATH
IN=${RESEARCH_RUN_DIR:?}/inputs
PYM=$SRC/.venv/bin/python
SU=$SRC/benchmarks/served_units
setup
log "source ${RESEARCH_SOURCE_SHA:-?}; scratch $W; lean $(sha256sum "$LEAN" | cut -c1-16)"
FW=$(dirname "$LEAN")/flock-firewall
[ -x "$FW" ] || log "no flock-firewall beside $LEAN"
rc=0

stage_as() {   # stage_as TAG WAY UNIT N STAGER FLAG...: $W/TAG/WAY, its record under $OUT/TAG/WAY
  local tag=$1 way=$2 unit=$3 n=$4 stager=$5 d=$W/$1/$2 o=$OUT/$1/$2 r
  shift 5
  mkdir -p "$d" "$o"
  "$PYM" -c 'import resource, subprocess, sys, json, time
t = time.time(); p = subprocess.run(sys.argv[2:], stdout=open(sys.argv[1] + "/stage.log", "w"), stderr=subprocess.STDOUT)
ru = resource.getrusage(resource.RUSAGE_CHILDREN)
json.dump({"exit": p.returncode, "wall_s": round(time.time() - t, 1), "maxrss_kb": ru.ru_maxrss}, open(sys.argv[1] + "/stage.time.json", "w"))
sys.exit(p.returncode)' "$o" "$PYM" "$stager" "$unit" "$n" "$d" "$@"
  r=$?
  for f in stage.json pin.txt partition.json; do [ -f "$d/$f" ] && cp "$d/$f" "$o/"; done
  (cd "$d" && sha512sum circuit.txt inst.bin pub.bin program.json partition.json 2>/dev/null) > "$o/files.sha512"
  ls -l "$d" > "$o/files.txt" 2>/dev/null
  log "$tag/$way: $(basename "$stager") $unit $n $*: rc $r; $(cat "$o/stage.time.json"); pin $(cut -c1-16 "$d/pin.txt" 2>/dev/null); $(tail -1 "$o/stage.log" | cut -c1-300)"
  return $r
}

lean() {    # lean TAG NAME ARGS...: one flock-verify command under $OUT/TAG/lean
  local o=$OUT/$1/lean n=$2 r
  shift 2
  mkdir -p "$o"
  "$LEAN" "$@" > "$o/$n.out" 2> "$o/$n.err"
  r=$?
  echo $r > "$o/$n.rc"; : > "$o/$n.time"
  log "  lean $1 $n: exit $r; $(grep -E '^(VERDICT|KS-FORM|setup:)' "$o/$n.out" | tail -1 | cut -c1-700) $(tail -1 "$o/$n.err" | cut -c1-200)"
}

# ---- 1. pins ---------------------------------------------------------------------------------------------------------------
pins() {   # pins FILE: each row of FILE staged the four ways and compared
while IFS=$'\t' read -r tag unit n; do
  [ -n "$tag" ] || continue
  log "pins $tag: $unit N=$n"
  stage_as "$tag" U "$unit" "$n" "$SU/stage.py" --unbound
  stage_as "$tag" M "$unit" "$n" "$IN/stage_main.py"
  stage_as "$tag" O "$unit" "$n" "$IN/stage_31b1b56.py"
  stage_as "$tag" B "$unit" "$n" "$SU/stage.py"
  "$PYM" - "$OUT/$tag" <<'EOF' | tee -a "$OUT/job.log"
import json, sys
from pathlib import Path
o = Path(sys.argv[1])
f = {w: dict(reversed(x.split()) for x in (o / w / "files.sha512").read_text().splitlines()) for w in "UMOB"}
s = {w: json.loads((o / w / "stage.json").read_text()) if (o / w / "stage.json").exists() else {} for w in "UMOB"}
same = all(f[w].get(k) == f["U"].get(k) and f["U"].get(k) for w in "MO" for k in ("circuit.txt", "inst.bin", "pub.bin"))
pins = {w: (o / w / "pin.txt").read_text().strip()[:16] if (o / w / "pin.txt").exists() else None for w in "UMOB"}
unb = not any(k in f[w] for w in "UMO" for k in ("program.json", "partition.json"))
bound = "program.json" in f["B"] and "partition.json" in f["B"] and pins["B"] != pins["U"]
cls = {w: str(s[w].get("class_sha512"))[:16] for w in "UMOB"}
print(f"PINS {o.name}: U=M=O {same} (pins {pins}); U, M, O unbound {unb}; B bound with its own pin {bound}; "
      f"class_sha512 {cls}; B bind {json.dumps(s['B'].get('bind'))[:300]}")
EOF
  rm -rf "${W:?}/$tag"
done < "$1"
}
pins "$IN/pins.tsv"

# ---- 2. SiluMul bound and unbound ------------------------------------------------------------------------------------------
SILU='census:SiluMul_v2{I=256}'
B_UNIT=${PR_B_UNIT:-$SILU}
for way in B U; do
  tag=silu-$way; flag=; [ $way = U ] && flag=--unbound
  stage_as "$tag" "$way" "$B_UNIT" 2 "$SU/stage.py" $flag || { rc=1; continue; }
  d=$W/$tag/$way
  S=$("$PYM" -c "import json; print(json.loads(open('$d/pub.bin', 'rb').readline())['statement'])")
  tb=; [ -d "$d/tables" ] && tb="--tables $d/tables"
  A="--statement $S --circuit $d/circuit.txt --public $d/pub.bin $tb"
  if [ $way = B ]; then
    CB_OTHERS=other-def "$PYM" "$IN/others.py" "$d/stage.json" "$W/$tag" "$B_UNIT" > "$OUT/$tag/others.json" 2> "$OUT/$tag/others.err" \
      || { log "$tag: others.py FAILED: $(tail -2 "$OUT/$tag/others.err")"; rc=1; }
    lean "$tag" statement.held statement $A --partition "$d/partition.json" --program "$d/program.json"
    lean "$tag" statement.other-def statement $A --partition "$d/partition.json" --program "$W/$tag/other-def/program.json"
    lean "$tag" statement.no-program statement $A --partition "$d/partition.json"
    lean "$tag" statement.no-partition statement $A --program "$d/program.json"
  else
    lean "$tag" statement.no-program statement $A
  fi
  N=2; o=$OUT/$tag/$way
  if session "$d" "$o"; then
    log "$tag: --zk session: $(grep -E '^(LIVE)' "$o/prove.out" | tail -1 | cut -c1-300)"
  else
    log "$tag: --zk session FAILED: $(tail -2 "$o/prove.err" | tr '\n' '|' | cut -c1-300) $(tail -2 "$o/serve.log" | tr '\n' '|' | cut -c1-300)"; rc=1
  fi
  s=$(ls -d "$d"/sessions/*/ 2>/dev/null | tail -1)
  if [ -n "$s" ]; then
    if [ $way = B ]; then
      lean "$tag" verify.held verify --zk $A --partition "$d/partition.json" --program "$d/program.json" --session "$s"
      lean "$tag" verify.other-def verify --zk $A --partition "$d/partition.json" --program "$W/$tag/other-def/program.json" --session "$s"
      lean "$tag" verify.no-program verify --zk $A --partition "$d/partition.json" --session "$s"
    else
      lean "$tag" verify.no-program verify --zk $A --session "$s"
    fi
  fi
  "$PYM" "$IN/verdicts.py" "$OUT/$tag" | tee -a "$OUT/job.log"
  rm -rf "${W:?}/$tag"
done

# ---- 3. served_units/job.sh ------------------------------------------------------------------------------------------------
mkdir -p "$OUT/su"
log "served_units/job.sh: SU_UNITS=${PR_SU_UNITS:-fp8-coord:K=256} N=2"
( RESEARCH_RUN_DIR=$OUT/su SU_UNITS=${PR_SU_UNITS:-fp8-coord:K=256} SU_N=2 SU_MEM_GB=$MEM_GB SU_CACHE=${HZ_CACHE:-} \
  FLOCK_PROVE=$BIN FLOCK_SELFTEST=$SBIN FLOCK_VERIFY=$LEAN HZ_PYTHON=$PY bash "$SU/job.sh" ) > "$OUT/su/job.out" 2>&1
log "served_units/job.sh: rc $?; $(grep -E '^\{' "$OUT/su/job.out" | tail -2 | cut -c1-900 | tr '\n' '|')"

# ---- 4. zk_cost.py -----------------------------------------------------------------------------------------------------------
Z=$W/zk; ZU=${PR_ZK_UNIT:-$SILU}
mkdir -p "$OUT/zk"
ZK() { FC_FIREWALL_LEAN=$FW RUST_MIN_STACK=$((64 << 20)) "$PYM" "$SRC/benchmarks/one_stage/zk_cost.py" "$@" --out "$Z"; }
ZK stage --units "$ZU" --sizes 2 --draws c > "$OUT/zk/stage.txt" 2>&1; log "zk_cost stage: rc $?; $(grep ZKCOST "$OUT/zk/stage.txt" | tail -1)"
"$PYM" -c "import json; [print('PLAN', {k: p.get(k) for k in ('unit', 'c', 'k', 'partition', 'program')}) for p in json.load(open('$Z/plan.json'))]" 2>&1 | tee -a "$OUT/job.log"
ZK prove --cpu --bin "$BIN" --arms fw --cold 0 --warm 0 --runs 1 > "$OUT/zk/prove.txt" 2>&1; log "zk_cost prove: rc $?; $(grep ZKCOST "$OUT/zk/prove.txt" | tail -1 | cut -c1-500)"
ZK verify --lean "$LEAN" > "$OUT/zk/verify.txt" 2>&1; zr=$?
log "zk_cost verify: rc $zr; $(grep ZKCOST "$OUT/zk/verify.txt" | cut -c1-500 | tr '\n' '|')"
[ $zr = 0 ] || rc=1
cp "$Z"/records/*.jsonl "$Z/plan.json" "$OUT/zk/" 2>/dev/null
for f in "$Z"/*/lean-verify-*.log "$Z"/points/*/lean-verify-*.log; do [ -f "$f" ] && cp "$f" "$OUT/zk/"; done
# a type whose staging peak isn't measured at this size goes last, so the rest is recorded if it runs out of memory
[ -f "$IN/pins-late.tsv" ] && pins "$IN/pins-late.tsv"
log "done, rc $rc"
exit $rc
