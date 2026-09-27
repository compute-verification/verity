#!/usr/bin/env bash
# Non-producer file re-verification of a verity/flock-circuit cell: stage this host's own circuit, tables and public files from the
# input set (circuit_bench's staging, public_only), check them against the verifier pod's, then `flock-circuit replay` every
# recorded session (Server::from_record + finish under this host's statement, plus the step-0 coin derivation), and negatives.
#   B=<flock-circuit, built from this tree with --features sha512>  SETDIR=<input set dir>  REC=<verifier run's out/verifier>
#   TEMPLATE=gemm-coordinate  TAG=name  NEG=p0-256/sessions-s0
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uo pipefail
REPO=$(pwd); O=${RESEARCH_RUN_DIR:-/tmp/fc-replay}/out/$TAG; W=${WORKDIR:-/tmp/fc-work}/$TAG; rm -rf $W; mkdir -p $O $W
export PYTHONPATH=$REPO/packages/verity/src:$REPO/backends/numerical/python:$REPO/backends/flock/python:$REPO/integrations/vllm
sha256sum $B | tee $O/binary.sha256
python3 - "$TEMPLATE" "$SETDIR" "$REC" "$W" <<'PY' | tee $O/stage.txt || exit 1
import hashlib, json, sys
from argparse import Namespace
from pathlib import Path
from verity_numerical.bench.input_sets import InputSet
from verity_flock import circuit as FC, circuit_bench as CB
template, set_dir, rec, out = sys.argv[1], sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])
inp = InputSet.open(set_dir)
assert inp.subcircuit.template.id == template, f"input set {inp.name} is not a {template} set"
low = FC.lowering_for_set(inp)
(out / "circuit.txt").write_text(FC.compose(low, inp.subcircuit, statics=FC.set_statics(inp)))
pin = hashlib.sha512((out / "circuit.txt").read_bytes()).hexdigest()
vpin = hashlib.sha512((rec / "circuit.txt").read_bytes()).hexdigest()
tables_same = None
if low.units.cut:
    from verity_flock import ir_lower
    ir_lower.write_tables(out / "tables")
    tables_same = all((out / "tables" / f.name).read_bytes() == f.read_bytes() for f in (rec / "tables").iterdir())
print(json.dumps({"set": inp.name, "content_digest": inp.content_digest, "n": inp.n, "circuit_sha512": pin, "verifier_circuit_sha512": vpin,
                  "circuit_equals_verifier": pin == vpin, "tables_equal_verifier": tables_same, "ands_per_instance": CB.ands(low)}))
a = Namespace(per_proof=1 << 30)
for d in sorted(rec.glob("p*-*"), key=lambda p: int(p.name[1:].split("-")[0])):
    total = int(d.name.split("-")[1])
    subs, t = CB.stage_point(a, low, inp, total, out / d.name, public=True)
    same = [hashlib.sha256((out / d.name / f"instances-s{j}.bin").read_bytes()).digest() == hashlib.sha256((d / f"instances-s{j}.bin").read_bytes()).digest()
            for j in range(len(subs))]
    print(json.dumps({"point": d.name, "subbatches": subs, "stage_s": t, "public_files_equal_verifier": same}), flush=True)
    assert all(same), f"{d.name}: a staged public file differs from the verifier pod's"
assert pin == vpin, "the staged circuit differs from the verifier pod's"
open(out / "pin", "w").write(pin)
PY
PIN=$(cat $W/pin); TB=; [ -d $W/tables ] && TB="--tables $W/tables"
rp() { $B replay --instances $1 --circuit ${5:-$W/circuit.txt} --pin $PIN $TB --session $2 --proofs $3,$4 2>/dev/null | grep -E '^(REPLAY|REFUSED)'; }
: > $O/replay.jsonl; n=0; bad=0
for sd in $REC/p*/sessions-s*; do
  pt=$(basename $(dirname $sd)); sb=${sd##*sessions-s}; inst=$W/$pt/instances-s$sb.bin
  for ses in $sd/l*; do
    [ -f $ses/circuit.rep0.bin ] || continue
    line=$(rp $inst $ses/session.json $ses/circuit.rep0.bin $ses/circuit.rep1.bin); n=$((n+1))
    echo "{\"session\": \"$pt/s$sb/$(basename $ses)\", \"replay\": ${line#*	}}" >> $O/replay.jsonl
    echo "$line" | grep -q '"accepted":true' || bad=$((bad+1))
  done
done
echo "REPLAYED $n sessions, $bad not accepted" | tee $O/summary.txt
# negatives on one session: each must be refused
sd=$REC/$NEG; ses=$(ls -d $sd/l* | sed -n 4p); oth=$(ls -d $sd/l* | sed -n 5p); inst=$W/${NEG%%/*}/instances-s${NEG##*sessions-s}.bin
T=$W/neg; mkdir -p $T; : > $O/negatives.jsonl
neg() { local name=$1 sj=$2 p0=$3 p1=$4 i=${5:-$inst} c=${6:-}
  line=$(rp $i $sj $p0 $p1 $c); r=$(echo "$line" | grep -q '"accepted":true' && echo accepted || echo rejected)
  echo "{\"neg\": \"$name\", \"result\": \"$r\", \"line\": $(echo "$line" | cut -f2- | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()[:220]))')}" | tee -a $O/negatives.jsonl
  [ $r = rejected ] || bad=$((bad+1)); }
neg proofs_from_other_session $ses/session.json $oth/circuit.rep0.bin $oth/circuit.rep1.bin
neg reps_swapped $ses/session.json $ses/circuit.rep1.bin $ses/circuit.rep0.bin
python3 - $ses/session.json $T <<'PY'
import json, sys, copy
r = json.load(open(sys.argv[1])); t = sys.argv[2]
def w(n, x): json.dump(x, open(f"{t}/{n}.json", "w"))
def lo(h): return h[:-1] + format(int(h[-1], 16) ^ 1, "x")
a = copy.deepcopy(r); a["coin_seed"]["seed"] = lo(a["coin_seed"]["seed"]); w("coin_seed_changed", a)
a = copy.deepcopy(r); s = next(x for x in a["streams"] if x["rounds"] and x["rounds"][-1].get("coins")); c = s["rounds"][len(s["rounds"]) // 2]["coins"]; c[0] = lo(c[0]); w("mid_coin_changed", a)
a = copy.deepcopy(r); m = a["streams"][0]["rounds"][0]["msg"]; a["streams"][0]["rounds"][0]["msg"] = ("1" if m[0] == "0" else "0") + m[1:]; w("retained_round_bytes_changed", a)
a = copy.deepcopy(r); a["link"]["sigma"] = lo(a["link"]["sigma"]); w("link_sigma_changed", a)
a = copy.deepcopy(r); p = a["link"]["points"][0]; p[0] = lo(p[0]); w("link_point_changed", a)
a = copy.deepcopy(r); a["link"] = None; w("link_exchange_removed", a)
a = copy.deepcopy(r); a["config"]["require_link"] = False; w("record_config_require_link_false", a)
PY
for c in coin_seed_changed mid_coin_changed retained_round_bytes_changed link_sigma_changed link_point_changed link_exchange_removed; do
  neg $c $T/$c.json $ses/circuit.rep0.bin $ses/circuit.rep1.bin; done
# info (not judged): the record's own config block is not trusted; from_record replays under the verifier's configuration
line=$(rp $inst $T/record_config_require_link_false.json $ses/circuit.rep0.bin $ses/circuit.rep1.bin)
echo "{\"neg\": \"record_config_require_link_false\", \"result\": \"info: $(echo "$line" | grep -q '"accepted":true' && echo accepted || echo rejected) (the replay uses its own config)\"}" | tee -a $O/negatives.jsonl
o=$(ls $W/p*/instances-s*.bin | grep -v "$inst" | head -1); [ -n "$o" ] && neg other_point_public_file $ses/session.json $ses/circuit.rep0.bin $ses/circuit.rep1.bin $o
python3 -c "import sys;t=open(sys.argv[1]).read().splitlines(True);i=len(t)//2;t[i]=t[i].replace('1','2',1) if '1' in t[i] else t[i]+' ';open(sys.argv[2],'w').write(''.join(t))" $W/circuit.txt $T/circuit-other.txt
neg circuit_not_the_pinned_one $ses/session.json $ses/circuit.rep0.bin $ses/circuit.rep1.bin $inst $T/circuit-other.txt
echo "NEGATIVES $(grep -c '"result": "rejected"' $O/negatives.jsonl) of $(grep -vc '"result": "info' $O/negatives.jsonl) rejected (+1 info); total bad $bad" | tee -a $O/summary.txt
[ $bad = 0 ]
