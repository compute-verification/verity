#!/usr/bin/env bash
# Non-producer file re-verification of an IR cell's verifier sessions (verity/flock-ir-frame/v3; a key-count class cell with its
# own manifest and per-T netlists, `--class`): build flock-ir-frame (CPU),
# stage every point's instance files here from the captured input set (IR2, ir_frame.stage as the verifier pod does), check
# them and the lowering against the verifier pod's (a sampling file: its native words checked first, S1), then `flock-ir-frame replay` every accepted session, plus tampered-record
# negatives that must be rejected.
#   inputs/$TAG.tgz (out/verifier minus instance files)  inputs/$TAG.sha256 (its instance files)  inputs/set-$TAG.tgz (the set)
#   TEMPLATE=rope-head PER=1024 NEG=p1-256/sessions-s0
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uxo pipefail
REPO=$(pwd); RD=${RESEARCH_RUN_DIR:-/tmp/flock-ir-replay}; W=/workspace/verify-flock-ir; mkdir -p $W
TAG=${TAG:?}; TEMPLATE=${TEMPLATE:?}; PER=${PER:?}; NEG=${NEG:-}; O=$RD/out/$TAG; mkdir -p $O
MODE=build GPU=0 bash verity/ml/flock/pod/20-gpu-link.sh > $O/build.log 2>&1 || { tail -30 $O/build.log; exit 1; }
source $HOME/.cargo/env
read -r Q PER_ < /sys/fs/cgroup/cpu.max 2>/dev/null || { Q=max; PER_=100000; }
TH=$([ "$Q" = max ] && nproc || echo $(( Q / PER_ ))); export RAYON_NUM_THREADS=$TH
F=/workspace/flock-gpu-link/flock
(cd $F && cargo build --release -p flock-live --bin flock-ir-frame -j $TH) >> $O/build.log 2>&1 || { tail -40 $O/build.log; exit 1; }
B=$F/target/release/flock-ir-frame; sha256sum $B | tee $O/binary.sha256
[ -x $HOME/.local/bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
[ -x $W/py/bin/python ] || { $HOME/.local/bin/uv venv -q --python 3.12 $W/py && $HOME/.local/bin/uv pip install -q --python $W/py/bin/python numpy blake3; }
export PATH=$W/py/bin:$PATH PYTHONPATH=$REPO:$REPO/benchmarks/numerical/python:$REPO/verity/ml/flock/python:$REPO/integrations/vllm
S=$W/rec-$TAG; rm -rf $S; mkdir -p $S; tar xzf $RD/inputs/$TAG.tgz -C $S
SD=$W/set-$TAG; rm -rf $SD; mkdir -p $SD; tar xzf $RD/inputs/set-$TAG.tgz -C $SD
SET=$(dirname $(find $SD -name manifest.json | head -1))
I=$W/inst-$TAG; rm -rf $I; mkdir -p $I
python3 - "$TEMPLATE" "$SET" "$S" "$I" "$PER" <<'PY' | tee $O/stage.txt || exit 1
import hashlib, json, os, sys
from argparse import Namespace
from pathlib import Path
from verity_numerical.bench.input_sets import InputSet
from verity_flock import ir_bench
template, set_dir, rec, out, per = sys.argv[1], sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]), int(sys.argv[5])
inp = InputSet.open(set_dir)
assert inp.subcircuit.template.id == template, f"input set {inp.name} is not a {template} set"
from verity_numerical.bench import lowerings
tmod = lowerings.registry("C-Flock").module(template)
klass = None
if hasattr(tmod, "class_manifest") and len(set(tmod.key_counts(inp))) > 1:
    # a key-count class cell: this pod's own manifest (the pin) and per-T netlists, which must be the verifier pod's
    klass = ir_bench.KeyClass(tmod, inp, out)
    low, pin = klass.lowering(klass.T[0]), klass.pin
    for T in sorted(set(klass.T)):
        klass.net(T)
    same = (out / "class.json").read_bytes() == (rec / "class.json").read_bytes() and all(
        klass.net(T).read_bytes() == (rec / f"net-t{T}.txt").read_bytes() for T in sorted(set(klass.T)))
    (out / "net.txt").write_text(low.text)
else:
    low = ir_bench._lowering(template, inp)
    (out / "net.txt").write_text(low.text)
    pin = hashlib.sha256((out / "net.txt").read_bytes()).hexdigest()
    same = (out / "net.txt").read_bytes() == (rec / "net.txt").read_bytes()
print(json.dumps({"set": inp.name, "content_digest": inp.content_digest, "n": inp.n, "pin": pin,
                  "class": [klass.lo, klass.hi] if klass else None, "netlist_equals_verifier": same}))
if low.units.cut:
    from verity_flock import ir_lower
    ir_lower.write_tables(out / "tables")
a = Namespace(per_proof=per, template=template, klass=klass)
mod = ir_bench.frame_module(template) if hasattr(ir_bench, 'frame_module') else None
for d in sorted(rec.glob("p*-*"), key=lambda p: int(p.name[1:].split("-")[0])):
    total = int(d.name.split("-")[1])
    subs, t = ir_bench.stage_point(a, low, inp, total, out / d.name)
    extra = []
    for j in range(len(subs)):
        f = out / d.name / f"instances-s{j}.bin"
        args = mod.serve_args(f) if mod is not None and hasattr(mod, "serve_args") else []
        if klass:
            args = ["--class", str(out / "class.json"), *args]
            (out / d.name / f"net-s{j}.txt").write_bytes(klass.net(klass.T[subs[j][0]]).read_bytes())
        (out / d.name / f"args-s{j}").write_text(" ".join(args))
        extra.append(args)
    print(json.dumps({"point": d.name, "subbatches": subs, "stage_s": t, "serve_args": extra}), flush=True)
open(os.environ.get("PINFILE", str(out / "pin")), "w").write(pin)
PY
grep -q '"netlist_equals_verifier": true' $O/stage.txt || { echo "netlist differs from the verifier pod's"; exit 1; }
PIN=$(cat $I/pin); TB=; [ -d $I/tables ] && TB="--tables $I/tables"
(cd $I && sha256sum -c $RD/inputs/$TAG.sha256) | tee $O/instances.check || exit 1
: > $O/replay.jsonl; n=0; bad=0
rp() { local j=$(basename $1 .bin | sed "s/instances-s//"); local ax=$(cat ${1%instances-s*}args-s$j 2>/dev/null); local nt=${1%instances-s*}net-s$j.txt; [ -f $nt ] || nt=$I/net.txt; $B replay --instances $1 --netlist $nt --pin $PIN $TB $ax --session $2 --proofs $3,$4 2>/dev/null | grep '^REPLAY'; }
for sd in $S/p*/sessions-s*; do
  pt=$(basename $(dirname $sd)); sb=${sd##*sessions-s}; inst=$I/$pt/instances-s$sb.bin
  for ses in $sd/l*; do
    [ -f $ses/irf.rep0.bin ] || continue
    line=$(rp $inst $ses/session.json $ses/irf.rep0.bin $ses/irf.rep1.bin); n=$((n+1))
    echo "{\"session\": \"$pt/s$sb/$(basename $ses)\", \"replay\": ${line#REPLAY	}}" >> $O/replay.jsonl
    echo "$line" | grep -q '"accepted":true' || bad=$((bad+1))
  done
done
echo "REPLAYED $n sessions, $bad not accepted" | tee $O/summary.txt
[ -z "$NEG" ] && { [ $bad = 0 ]; exit; }
# negatives on one session (NEG): must-reject, except unused_high_bits_of_query_coin (must accept: Flock masks a query word's
# low bits) and the info coins an honest proof does not depend on (the first zerocheck coin, the last query coin)
sd=$S/$NEG; ses=$(ls -d $sd/l* | sed -n 4p); oth=$(ls -d $sd/l* | sed -n 5p)
inst=$I/${NEG%%/*}/instances-s${NEG##*sessions-s}.bin
T=$W/neg-$TAG; mkdir -p $T; : > $O/negatives.jsonl
neg() { local name=$1 sj=$2 p0=$3 p1=$4 i=${5:-$inst} want=${6:-rejected}
  line=$(rp $i $sj $p0 $p1)
  r=$(echo "$line" | grep -q '"accepted":true' && echo accepted || echo rejected)
  echo "{\"neg\": \"$name\", \"want\": \"$want\", \"result\": \"$r\", \"replay\": ${line#REPLAY	}}" | tee -a $O/negatives.jsonl | cut -c1-300
  [ $want = info ] || [ $r = $want ] || bad=$((bad+1)); }
neg proofs_from_other_session $ses/session.json $oth/irf.rep0.bin $oth/irf.rep1.bin
neg reps_swapped $ses/session.json $ses/irf.rep1.bin $ses/irf.rep0.bin
python3 - $ses/session.json $T <<'PY'
import json, sys, copy
r = json.load(open(sys.argv[1])); t = sys.argv[2]; tb = "irf"
def w(n, x): json.dump(x, open(f"{t}/{n}.json", "w"))
def lo(h): return h[:-1] + format(int(h[-1], 16) ^ 1, "x")
def hi(h): return format(int(h[0], 16) ^ 8, "x") + h[1:]
S = {s["stream"]: s for s in r["streams"]}
mid = len(S[f"{tb}/rep0"]["rounds"]) // 2
for name, stream, k, j, f in [("zerocheck_coin_low_bit", f"{tb}/rep0", 0, 0, lo), ("rep1_second_round_coin_low_bit", f"{tb}/rep1", 1, 5, lo),
                              ("rep0_mid_round_coin_low_bit", f"{tb}/rep0", mid, 0, lo), ("last_coin_low_bit", f"{tb}/rep1", -1, 0, lo),
                              ("unused_high_bits_of_query_coin", f"{tb}/rep1", -1, 0, hi)]:
    a = copy.deepcopy(r); s = next(x for x in a["streams"] if x["stream"] == stream); c = s["rounds"][k]["coins"]; c[j] = f(c[j]); w(name, a)
a = copy.deepcopy(r); p = a["link"]["points"][0]; p[0] = lo(p[0]); w("link_point_changed", a)
a = copy.deepcopy(r); a["link"]["publics_sha256"][tb] = "00" * 32; w("publics_digest_changed", a)
a = copy.deepcopy(r); s = a["streams"][0]; s["rounds"][0]["msg_sha256"] = "00" * 32; w("round_message_digest_changed", a)
a = copy.deepcopy(r); a["config"]["require_link"] = False; w("require_link_false", a)
PY
for n in zerocheck_coin_low_bit last_coin_low_bit; do neg $n $T/$n.json $ses/irf.rep0.bin $ses/irf.rep1.bin $inst info; done
for n in rep1_second_round_coin_low_bit rep0_mid_round_coin_low_bit link_point_changed publics_digest_changed round_message_digest_changed require_link_false; do
  neg $n $T/$n.json $ses/irf.rep0.bin $ses/irf.rep1.bin; done
neg unused_high_bits_of_query_coin $T/unused_high_bits_of_query_coin.json $ses/irf.rep0.bin $ses/irf.rep1.bin $inst accepted
if [ -s ${inst%instances-s*}args-s${NEG##*sessions-s} ]; then  # S1: a sampling file replays only once check_native passed
  line=$($B replay --instances $inst --netlist $I/net.txt --pin $PIN $TB --session $ses/session.json --proofs $ses/irf.rep0.bin,$ses/irf.rep1.bin 2>/dev/null | grep -E '^(REPLAY|REFUSED)')
  r=$(echo "$line" | grep -q '"accepted":true' && echo accepted || echo rejected)
  echo "{\"neg\": \"native_check_not_attested\", \"want\": \"rejected\", \"result\": \"$r\", \"line\": $(echo "$line" | cut -f2- | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()[:200]))')}" | tee -a $O/negatives.jsonl
  [ $r = rejected ] || bad=$((bad+1))
fi
o=$(ls $I/*/instances-s*.bin | grep -v "$inst" | tail -1); neg other_instance_file $ses/session.json $ses/irf.rep0.bin $ses/irf.rep1.bin $o
echo "NEGATIVES $(grep -c '"want": "rejected", "result": "rejected"' $O/negatives.jsonl) rejected as wanted, $(grep -c '"want": "accepted", "result": "accepted"' $O/negatives.jsonl) accepted as wanted, $(grep -c '"want": "info"' $O/negatives.jsonl) info, of $(wc -l < $O/negatives.jsonl); total bad $bad" | tee -a $O/summary.txt
[ $bad = 0 ]
