#!/usr/bin/env bash
# Non-producer file re-verification of a Flock cell's verifier sessions (verity/flock-pure-block/v2): build flock-pure-gpu
# (CPU), regenerate every point's instance files here, check them against the verifier pod's, then `flock-pure-gpu replay`
# every accepted session from its record and proof files, plus tampered-record negatives that must be rejected.
#   inputs/$TAG.tgz (out/verifier minus instance files)  inputs/$TAG.sha256 (its instance files)  REL=bf16-hopper PER=8192
#   NEG=p3-8192/sessions-s0 (where the negatives run; empty: none)
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uxo pipefail
REPO=$(pwd); RD=${RESEARCH_RUN_DIR:-/tmp/flock-replay}; O=$RD/out; W=/workspace/verify-flock-pure; mkdir -p $O $W
REL=${REL:-bf16-hopper}; PER=${PER:-8192}; TAG=${TAG:-sessions}; NEG=${NEG-p3-8192/sessions-s0}; O=$O/$TAG; mkdir -p $O
MODE=build GPU=0 bash backends/flock/pod/20-gpu-link.sh > $O/build.log 2>&1 || { tail -30 $O/build.log; exit 1; }
source $HOME/.cargo/env
read -r Q PER_ < /sys/fs/cgroup/cpu.max 2>/dev/null || { Q=max; PER_=100000; }
TH=$([ "$Q" = max ] && nproc || echo $(( Q / PER_ ))); export RAYON_NUM_THREADS=$TH
[ -x $HOME/.local/bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
[ -x $W/py/bin/python ] || { $HOME/.local/bin/uv venv -q --python 3.12 $W/py && $HOME/.local/bin/uv pip install -q --python $W/py/bin/python numpy blake3; }
export PATH=$W/py/bin:$PATH PYTHONPATH=$REPO/packages/verity/src:$REPO/backends/numerical/python:$REPO/backends/flock/python
if [ "$REL" = bf16-ampere ]; then
  BI=$W/bench-instances/v1; FZ=$REPO/fixtures/bench-instances/v1
  [ -f $BI/vu-k1536.x.u16 ] || python3 -m verity_numerical.bench.instances build --out $BI --seeds $FZ/seeds --procs $TH 2>&1 | tail -3
  python3 - $BI $FZ <<'PYEOF' | tee $O/bench-instances.txt || exit 4
import json, sys
from pathlib import Path
from verity_numerical.bench.instances import sha256_file
bi, fz = Path(sys.argv[1]), Path(sys.argv[2])
man = json.loads((fz / "manifest.json").read_text())
built = {n: f for t in man["tiers"].values() for n, f in t.get("files", {}).items() if not f.get("committed")}
bad = [n for n, f in built.items() if not (bi / n).is_file() or sha256_file(bi / n) != f["sha256"]]
assert not bad, f"rebuilt arrays differ from the committed manifest: {bad}"
for n in built:
    dst = fz / n
    dst.unlink(missing_ok=True)
    dst.symlink_to(bi / n)
print(f"{len(built)} rebuilt arrays match the committed manifest; linked into {fz}")
PYEOF
fi
B=/workspace/flock-gpu-link/flock/target/release/flock-pure-gpu; sha256sum $B | tee $O/binary.sha256
lscpu | grep -E 'Model name|^CPU\(s\)' | tee $O/host.txt; echo threads $TH | tee -a $O/host.txt
S=$W/rec-$TAG; rm -rf $S; mkdir -p $S; tar xzf $RD/inputs/$TAG.tgz -C $S
PIN=$(python3 -c "from verity_flock import lowering; print(lowering.PINS['$REL'])")
NET=$W/net-$REL.txt; python3 -c "from verity_flock import lowering; open('$NET','w').write(lowering.netlist('$REL'))"
echo "$PIN  $NET" | sha256sum -c | tee $O/netlist.check || exit 1
cmp $NET $S/net-$REL.txt && echo "netlist equals the verifier pod's" | tee -a $O/netlist.check
python3 - "$REL" "$PER" "$S" "$W/inst-$TAG" "$TH" <<'PY' | tee $O/instances.txt
import sys, json
from pathlib import Path
from verity_flock import instances, bench
rel, per, s, out, procs = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), int(sys.argv[5])
for d in sorted(s.glob("p*-*"), key=lambda p: int(p.name[1:].split("-")[0])):
    total = int(d.name.split("-")[1])
    for j, (lo, hi) in enumerate(bench.subbatches(total, per)):
        f = out / d.name / f"instances-s{j}.bin"; f.parent.mkdir(parents=True, exist_ok=True)
        h = instances.write(f, rel, total, procs, None, lo, hi)
        print(json.dumps({"file": f"{d.name}/instances-s{j}.bin", "range": [lo, hi], "roots": h["roots"]}), flush=True)
PY
(cd $W/inst-$TAG && sha256sum -c $RD/inputs/$TAG.sha256) | tee $O/instances.check || exit 1
: > $O/replay.jsonl; n=0; bad=0
for sd in $S/p*/sessions-s*; do
  pt=$(basename $(dirname $sd)); sb=${sd##*sessions-s}; inst=$W/inst-$TAG/$pt/instances-s$sb.bin
  for ses in $sd/l*; do
    [ -f $ses/pure.rep0.bin ] || continue
    line=$($B replay --instances $inst --netlist $NET --pin $PIN --session $ses/session.json --proofs $ses/pure.rep0.bin,$ses/pure.rep1.bin 2>/dev/null | grep '^REPLAY')
    rc=$?; n=$((n+1))
    echo "{\"session\": \"$pt/s$sb/$(basename $ses)\", \"replay\": ${line#REPLAY	}}" >> $O/replay.jsonl
    echo "$line" | grep -q '"accepted":true' || bad=$((bad+1))
  done
done
echo "REPLAYED $n sessions, $bad not accepted" | tee $O/summary.txt
[ -z "$NEG" ] && { [ $bad = 0 ]; exit; }
# negatives on one session (NEG): each must be rejected, except unused_high_bits_of_query_coin (Flock's query sampling
# masks the low bits of each squeezed word, ligerito.rs queries_from_words), which must be accepted. info: coins an honest
# proof does not depend on (the zerocheck challenge multiplies A∘B−C, zero for an honest witness), recorded, not judged
sd=$S/$NEG; ses=$(ls -d $sd/l* | sed -n 4p); oth=$(ls -d $sd/l* | sed -n 5p)
inst=$W/inst-$TAG/${NEG%%/*}/instances-s${NEG##*sessions-s}.bin
T=$W/neg-$TAG; mkdir -p $T
neg() { local name=$1 sj=$2 p0=$3 p1=$4 i=${5:-$inst} want=${6:-rejected}
  line=$($B replay --instances $i --netlist $NET --pin $PIN --session $sj --proofs $p0,$p1 2>/dev/null | grep '^REPLAY')
  r=$(echo "$line" | grep -q '"accepted":true' && echo accepted || echo rejected)
  echo "{\"neg\": \"$name\", \"want\": \"$want\", \"result\": \"$r\", \"replay\": ${line#REPLAY	}}" | tee -a $O/negatives.jsonl | cut -c1-300
  [ $want = info ] || [ $r = $want ] || bad=$((bad+1)); }
: > $O/negatives.jsonl
neg proofs_from_other_session $ses/session.json $oth/pure.rep0.bin $oth/pure.rep1.bin
neg reps_swapped $ses/session.json $ses/pure.rep1.bin $ses/pure.rep0.bin
python3 - $ses/session.json $T <<'PY'
import json, sys, copy
r = json.load(open(sys.argv[1])); t = sys.argv[2]
def w(n, x): json.dump(x, open(f"{t}/{n}.json", "w"))
def lo(h): return h[:-1] + format(int(h[-1], 16) ^ 1, "x")
def hi(h): return format(int(h[0], 16) ^ 8, "x") + h[1:]
S = {s["stream"]: s for s in r["streams"]}
for name, stream, k, j, f in [("zerocheck_coin_low_bit", "pure/rep0", 0, 0, lo), ("rep1_second_round_coin_low_bit", "pure/rep1", 1, 5, lo),
                              ("rep0_mid_round_coin_low_bit", "pure/rep0", 70, 0, lo), ("last_coin_low_bit", "pure/rep1", -1, 0, lo),
                              ("unused_high_bits_of_query_coin", "pure/rep1", -1, 0, hi)]:
    a = copy.deepcopy(r); s = next(x for x in a["streams"] if x["stream"] == stream); c = s["rounds"][k]["coins"]; c[j] = f(c[j]); w(name, a)
a = copy.deepcopy(r); p = a["link"]["points"][0]; p[0] = lo(p[0]); w("link_point_changed", a)
a = copy.deepcopy(r); a["link"]["publics_sha256"]["pure"] = "00" * 32; w("publics_digest_changed", a)
a = copy.deepcopy(r); s = a["streams"][0]; s["rounds"][0]["msg_sha256"] = "00" * 32; w("round_message_digest_changed", a)
a = copy.deepcopy(r); a["config"]["require_link"] = False; w("require_link_false", a)
PY
for n in zerocheck_coin_low_bit last_coin_low_bit; do neg $n $T/$n.json $ses/pure.rep0.bin $ses/pure.rep1.bin $inst info; done
for n in rep1_second_round_coin_low_bit rep0_mid_round_coin_low_bit link_point_changed publics_digest_changed round_message_digest_changed require_link_false; do
  neg $n $T/$n.json $ses/pure.rep0.bin $ses/pure.rep1.bin; done
neg unused_high_bits_of_query_coin $T/unused_high_bits_of_query_coin.json $ses/pure.rep0.bin $ses/pure.rep1.bin $inst accepted
o=$(ls $W/inst-$TAG/*/instances-s*.bin | grep -v "$inst" | tail -1); neg other_instance_file $ses/session.json $ses/pure.rep0.bin $ses/pure.rep1.bin $o
echo "NEGATIVES $(grep -c '"want": "rejected", "result": "rejected"' $O/negatives.jsonl) rejected as wanted, $(grep -c '"want": "accepted", "result": "accepted"' $O/negatives.jsonl) accepted as wanted, $(grep -c '"want": "info"' $O/negatives.jsonl) info, of $(wc -l < $O/negatives.jsonl); total bad $bad" | tee -a $O/summary.txt
[ $bad = 0 ]
