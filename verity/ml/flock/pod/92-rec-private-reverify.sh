# rec-private (lane rec-v0): the replay. Lean's tools as this tree builds them (flock-verify, flock-firewall, inner-fold,
# core-firewall), again, on what earlier runs of 87-rec-private.sh recorded, each verdict beside the run's own:
#   - every outer part's verify --zk (honest parts and the cheating provers' alike): the statement its sizes.txt names
#     (circuit.txt, pub.bin and the verifier's partition, program and registered reads beside it) and its recorded session;
#   - for a part released behind the firewall, core-firewall's decision on its recorded evidence and the record before it;
#   - V*'s fold of the wired circuit (InnerFold.lean --program, the setup Lean's verify runs): the wired cheater's (made-up)
#     and the honest wired statement's, with and without the program;
#   - the firewall's contract on the run's inner session (firewall_agree.py --sessions), on it with a coin the challenger did
#     not commit to (rec_private_coins.py), and on firewall_agree.py's challengers (--cases, once);
#   - the salts check, Python as it ran (rec_private_summary.salts on the run's recorded transcripts).
# Statements, inner sessions and V*'s points are read where the run left them ($FLOCK_WORK/data/rec-private/TAG).
# out/reverify/RUN/{PART,controls}/, out/reverify.json, and a REVERIFY line for each verdict.
# env: RUNS (run ids, comma-separated)  RESTORED (RUN=DIR,...: a run whose files were evicted, read from DIR/out, a copy
#      restored from its artifact by `research data fetch`)  PARTS (default every part)  JOBS=8 (verifies at once)  CPUS=16
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uo pipefail
if [ -z "${RI_PINNED:-}" ]; then
  . verity/ml/flock/pod/cpu-slices.sh
  hold_slices "" "${CPUS:-16}" || exit 1
  export RI_PINNED=$SLICE_CPUS RAYON_NUM_THREADS=$(echo $SLICE_CPUS | tr ',' '\n' | wc -l)
  exec taskset -c $SLICE_CPUS bash "$0" "$@"
fi
# rec-env.sh's directories and Lean build, without its prover build (nothing here proves)
REPO=$(pwd); O=${RESEARCH_RUN_DIR:-/tmp/rec-v0-job}/out; mkdir -p $O
FW=${FLOCK_WORK:-/workspace/jobs/rec-v0}; W=$FW/flock-circuit; L=$FW/cache/rec-v0; mkdir -p $L || exit 1
[ -w "$HOME" ] || { export HOME=$FW/home; mkdir -p $HOME; }
{ lscpu | grep -E 'Model name|^CPU\(s\)'; echo "pinned=$RI_PINNED"; free -g | head -2; cat /proc/loadavg; git rev-parse HEAD; } > $O/host.txt 2>&1
[ -x /usr/bin/time ] || { echo "no /usr/bin/time" >&2; exit 1; }
LSHA=$(cd verity/core/service/flock && find . -name .lake -prune -o -type f -print | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-16)
LEAN_BIN=$L/lean-$LSHA/flock-verify
(
  export PATH=$HOME/.elan/bin:$PATH
  command -v lake >/dev/null || curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none >/dev/null
  exec 9>$L/lean.lock; flock 9
  if [ ! -x $LEAN_BIN ]; then
    rm -rf $L/lean-src && mkdir -p $L/lean-src && cp -r verity/core/service/flock/. $L/lean-src/ \
      && (cd $L/lean-src && lake build flock-verify) && mkdir -p $L/lean-$LSHA && cp $L/lean-src/.lake/build/bin/flock-verify $LEAN_BIN
  fi
) > $O/build-lean.log 2>&1
[ -x $LEAN_BIN ] || { tail -20 $O/build-lean.log; exit 1; }
# Lean's tools beside it, as 87-rec-private.sh builds them
TSHA=$( (find verity/core/service/flock verity/core/service/firewall verity/core/protocols/interface verity/core/difftest/lean \
  -name .lake -prune -o -type f -print | LC_ALL=C sort | xargs sha256sum; cat verity/ml/flock/tests/lean/InnerFold.lean) | sha256sum | cut -c1-16)
T=$L/tools-$TSHA
(
  export PATH=$HOME/.elan/bin:$PATH
  exec 9>$L/tools.lock; flock 9
  [ -x $T/core-firewall ] && [ -x $T/inner-fold ] && [ -x $T/flock-firewall ] && exit 0
  C=$L/tools-src; rm -rf $C && mkdir -p $C/verity/core/service $C/verity/core/protocols $C/verity/core/difftest || exit 1
  cp -r verity/core/service/flock verity/core/service/firewall $C/verity/core/service/ && cp -r verity/core/protocols/interface $C/verity/core/protocols/ \
    && cp -r verity/core/difftest/lean $C/verity/core/difftest/ || exit 1
  find $C -name .lake -prune -exec rm -rf {} +
  [ -d $L/lean-src/.lake/build ] && cp -r $L/lean-src/.lake $C/verity/core/service/flock/
  cp verity/ml/flock/tests/lean/InnerFold.lean $C/verity/core/service/flock/ \
    && printf '\n[[lean_exe]]\nname = "inner-fold"\nroot = "InnerFold"\n' >> $C/verity/core/service/flock/lakefile.toml || exit 1
  (cd $C/verity/core/service/flock && lake build flock-firewall inner-fold) || exit 1
  (cd $C/verity/core/service/firewall && lake build core-firewall) || exit 1
  mkdir -p $T && cp $C/verity/core/service/flock/.lake/build/bin/inner-fold $C/verity/core/service/flock/.lake/build/bin/flock-firewall \
    $C/verity/core/service/firewall/.lake/build/bin/core-firewall $T/
) > $O/build-tools.log 2>&1
[ -x $T/core-firewall ] && [ -x $T/inner-fold ] && [ -x $T/flock-firewall ] || { tail -30 $O/build-tools.log; exit 1; }
sha256sum $LEAN_BIN $T/* > $O/binaries.sha256
export PATH=$W/py/bin:$PATH PYTHONPATH=$REPO:$REPO/verity/ml/flock/python:$REPO/verity/ml/flock/pod${PYTHONPATH:+:$PYTHONPATH}
ROOT=${RESEARCH_RUNS_ROOT:-/workspace/research/runs}

# outdir RUN: where RUN's out/ is read from (its RESTORED copy, else the run dir); data_of OUT: the run's data dir, from the
# path its L0's sizes.txt names
outdir() { local kv; for kv in ${RESTORED//,/ }; do [ "${kv%%=*}" = $1 ] && { echo ${kv#*=}/out; return; }; done; echo $ROOT/$1/out; }
data_of() { grep -m1 '/circuit.txt$' $1/outer/L0/sizes.txt | cut -f2 | sed -E 's#(/data/rec-private/[^/]+)/.*#\1#'; }
pub_of() { ls $1/pub-*.bin | head -1; }

one() {
  local R=$1 NAME=$2 D=$(outdir $1)/outer/$2 E=$O/reverify/$1/$2 C SESS VF=()
  mkdir -p $E
  C=$(dirname "$(grep -m1 '/circuit.txt$' $D/sizes.txt | cut -f2)")
  SESS=$(find $D/sessions -name session.json -printf '%h\n' 2>/dev/null | sort | head -1)
  { [ -s $C/circuit.txt ] && [ -n "$SESS" ]; } || { echo "$R $NAME: no statement or session"; return 0; }
  for f in program registered; do [ -f $C/$f.json ] && VF+=(--$f $C/$f.json); done
  /usr/bin/time -v -o $E/time-lean.txt $LEAN_BIN verify --zk --statement verity/flock-circuit --circuit $C/circuit.txt --public $C/pub.bin \
    --partition $C/partition.json "${VF[@]}" --session $SESS > $E/lean.log 2> $E/lean.err
  echo "$R $NAME lean exit $?"
  if [ -s $D/evidence.json ] && [ -s $D/record-before.jsonl ]; then
    $T/core-firewall check $D/params.json $D/record-before.jsonl $D/evidence.json > $E/core-firewall.txt 2> $E/core-firewall.err
    echo "$R $NAME core-firewall exit $?"
  fi
}

controls() {
  local R=$1 OD=$(outdir $1) E=$O/reverify/$1/controls DD
  DD=$(data_of $OD); mkdir -p $E
  local W0=$DD/w/stage MU=$DD/w/made-up-input
  if [ -s $MU/circuit.txt ]; then
    $T/inner-fold --plain-leaves --partition $MU/partition.json --program $MU/program.desc verity/flock-circuit $MU/circuit.txt \
      $(pub_of $MU) $DD/points-made-up.json os > $E/fold-made-up.json 2> $E/fold-made-up.err; echo $? > $E/fold-made-up.exit
    $T/inner-fold --plain-leaves --partition $W0/partition.json --program $W0/program.desc verity/flock-circuit $W0/circuit.txt \
      $(pub_of $W0) $DD/points-wired.json os > $E/fold-wired.json 2> $E/fold-wired.err; echo $? > $E/fold-wired.exit
    $T/inner-fold --plain-leaves --partition $W0/partition.json verity/flock-circuit $W0/circuit.txt \
      $(pub_of $W0) $DD/points-wired-noprogram.json os > $E/fold-wired-noprogram.json 2> $E/fold-wired-noprogram.err
    echo $? > $E/fold-wired-noprogram.exit
  fi
  if [ -d $DD/inner-honest/session ]; then
    python3 verity/ml/flock/verifier/firewall_agree.py --lean $T/flock-firewall --out $E/firewall-contract --sessions $DD/inner-honest \
      > $E/firewall-contract.txt 2>&1; echo $? > $E/firewall-contract.exit
    python3 verity/ml/flock/pod/rec_private_coins.py --lean $T/flock-firewall --session $DD/inner-honest --out $E/coins-contract \
      > $E/coins-contract.txt 2>&1; echo $? > $E/coins-contract.exit
  fi
  echo "$R controls done"
}
export -f one controls outdir data_of pub_of
export O LEAN_BIN T ROOT RESTORED=${RESTORED:-}
python3 verity/ml/flock/verifier/firewall_agree.py --lean $T/flock-firewall --out $O/reverify/firewall-cases --cases \
  > $O/reverify/firewall-cases.txt 2>&1; echo "firewall cases exit $?"
for R in ${RUNS//,/ }; do
  echo "controls $R"
  for D in $(outdir $R)/outer/*/; do
    N=$(basename $D)
    [ -z "${PARTS:-}" ] || [[ ",$PARTS," == *",$N,"* ]] || continue
    echo "one $R $N"
  done
done | xargs -P ${JOBS:-8} -L 1 bash -c '"$0" "$@"'

python3 - $O $ROOT ${RUNS//,/ } <<'EOF'
import json, os, sys
from pathlib import Path

import rec_private_summary as S

out, root, runs = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
restored = dict(kv.split("=", 1) for kv in os.environ.get("RESTORED", "").split(",") if kv)
first = lambda p: (S.text(p).strip().splitlines() or [""])[0][:300]


def release(p: Path):
    """core-firewall's decision: its verdict and reason, or what it printed instead."""
    try:
        v = json.loads(S.text(p / "core-firewall.txt").splitlines()[0])
        return f"{v['verdict']}" + (f": {v['reason']}" if v.get("reason") else "")
    except (IndexError, KeyError, ValueError):
        return first(p / "core-firewall.err") or None


def fold(d: Path, n: str):
    if not (d / f"fold-{n}.exit").exists():
        return None
    rc = int(S.text(d / f"fold-{n}.exit").strip() or -1)
    why = (S.text(d / f"fold-{n}.err").strip().splitlines() or [""])[-1]
    return "InnerFold accepted" if rc == 0 else f"InnerFold refused: {why}"


def contract(p: Path):
    r = json.loads(S.text(p) or "null")
    if not isinstance(r, dict):
        return None
    return {k: S.verdict("Lean", v["verdict"].get("accepted"), v["verdict"].get("why")) if isinstance(v, dict) and "verdict" in v else v
            for k, v in r.items() if k != "swapped"}


res = {}
for r in runs:
    o = Path(restored[r]) / "out" if r in restored else root / r / "out"
    e = out / "reverify" / r
    rows = res.setdefault(r, {})
    for d in sorted(p for p in e.glob("*/") if p.name != "controls"):
        rec = o / "outer" / d.name
        rows[d.name] = {"recorded": S.lean(rec / "lean.log")["verdict"], "now": S.lean(d / "lean.log")["verdict"],
                        "lean_wall": S.tv(d / "time-lean.txt", r"Elapsed \(wall clock\) time")}
        if (d / "core-firewall.txt").exists():
            rows[d.name + " core-firewall"] = {"recorded": release(rec), "now": release(d)}
    c = e / "controls"
    for n in ("made-up", "wired", "wired-noprogram"):
        if fold(c, n) is not None:
            rows[f"fold-{n}"] = {"recorded": fold(o / "vstar", n), "now": fold(c, n)}
    for n in ("coins-contract", "firewall-contract"):
        now = contract(c / n / "result.json")
        if now is not None or (c / f"{n}.txt").exists():
            rows[n] = {"recorded": contract(o / "inner" / n / "result.json"), "now": now or S.text(c / f"{n}.txt")[-400:]}
    sm = json.loads(S.text(o / "summary.json") or "null") or {}
    now = S.salts(o)
    rows["salts"] = {"recorded": {k: v for k, v in (sm.get("salts") or {}).items() if k != "transcripts"} or None,
                     "now": {k: v for k, v in now.items() if k != "transcripts"}}
cases = lambda p: json.loads(S.text(p / "firewall-cases" / "result.json") or "null")
res["firewall-cases"] = {"cases": {"recorded": next((v for r in runs if (v := cases(root / r / "out" / "inner")) is not None), None),
                                   "now": cases(out / "reverify")}}
(out / "reverify.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
for r, rows in res.items():
    for n, v in rows.items():
        print("REVERIFY", r, n, json.dumps(v, ensure_ascii=False)[:1500])
EOF
