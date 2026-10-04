#!/usr/bin/env bash
# The first recursive session's inner side (rec_inner.py, verity_flock.rec_live): RoPE d64 staged as M0's statement, proved
# on the CPU by the unmodified prover (FC_COINS=os, non-zk) against the rec proxy, whose coin server sees only caps, message
# commitments and the schedule; then the recorded session replayed by upstream's verifier (flock-circuit replay) and by
# Lean's flock-verify (--coins os), and the same session with one coin changed, which both must reject. Run from the verity
# tree (research run --cwd source). Outputs under out/: host.txt, stage.json, time-prove.txt, prove.txt/.err, proxy.log,
# inner/ (session/, public.json, private.json), replay.txt, lean.log, forged-*.txt, summary.json.
# env: N=32 (instances: RoPE heads, one VU each)  CPUS=32  FLOCK_WORK (as 76-read-gadget.sh)
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uxo pipefail
if [ -z "${RI_PINNED:-}" ]; then
  . backends/flock/pod/cpu-slices.sh
  hold_slices "" "${CPUS:-32}" || exit 1
  export RI_PINNED=$SLICE_CPUS RAYON_NUM_THREADS=$(echo $SLICE_CPUS | tr ',' '\n' | wc -l)
  exec taskset -c $SLICE_CPUS bash "$0" "$@"
fi
export FC_COINS=os
REPO=$(pwd); O=${RESEARCH_RUN_DIR:-/tmp/rec-inner}/out; mkdir -p $O
[ -z "${FLOCK_WORK:-}" ] && [ ! -w /workspace ] && export FLOCK_WORK=/workspace/jobs/rec-v0
FW=${FLOCK_WORK:-/workspace}; W=$FW/flock-circuit; mkdir -p $W || exit 1
[ -w "$HOME" ] || { export HOME=$FW/home; mkdir -p $HOME; }
S=$FW/scratch/${RESEARCH_RUN_ID:-local-$$}; rm -rf $S; mkdir -p $S || exit 1
L=$FW/cache/rec-v0; mkdir -p $L || exit 1
unset CARGO_TARGET_DIR
trap 'rm -rf $S' EXIT
{ lscpu | grep -E 'Model name|^CPU\(s\)'; echo "pinned=$RI_PINNED rayon=$RAYON_NUM_THREADS"; free -g | head -2; cat /proc/loadavg; } > $O/host.txt 2>&1
[ -x /usr/bin/time ] || { apt-get install -y -qq time >/dev/null 2>&1 || { apt-get update -qq >/dev/null 2>&1 && apt-get install -y -qq time >/dev/null 2>&1; }; }
[ -x /usr/bin/time ] || { echo "no /usr/bin/time" >&2; exit 1; }

LSHA=$(cd backends/flock/verifier/lean && find . -name .lake -prune -o -type f -print | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-16)
LEAN_BIN=$L/lean-$LSHA/flock-verify
(
  export PATH=$HOME/.elan/bin:$PATH
  command -v lake >/dev/null || curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none >/dev/null
  exec 9>$L/lean.lock; flock 9
  if [ ! -x $LEAN_BIN ]; then
    rm -rf $L/lean-src && mkdir -p $L/lean-src && cp -r backends/flock/verifier/lean/. $L/lean-src/ \
      && (cd $L/lean-src && lake build flock-verify) && mkdir -p $L/lean-$LSHA && cp $L/lean-src/.lake/build/bin/flock-verify $LEAN_BIN
  fi
) > $O/build-lean.log 2>&1 &
LP=$!

SM=${SM:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d .)}; SM=${SM:-89}
KEY=$(cat backends/flock/pod/20-gpu-link.sh backends/flock/pod/60-circuit.sh backends/flock/*.patch backends/flock/cuda_*_patch.py \
  $(find backends/flock/cuda backends/flock/live -type f | LC_ALL=C sort) | sha256sum | cut -c1-16)
BIN=$W/bin-$KEY-g0-sm$SM; F=$FW/flock-gpu-link/flock
exec 8>$W/build.lock; flock 8
if [ -x $BIN/flock-circuit ] && [[ $(ldd $BIN/flock-circuit 2>&1) == *"not found"* ]]; then
  rm -rf $BIN || { echo "$BIN: built against another glibc, and this job can't remove it" >&2; exit 1; }
fi
if [ ! -x $BIN/flock-circuit ]; then
  RESEARCH_RUN_DIR=$O/build FLOCK_WORK=$FW MODE=build GPU=0 SM=$SM bash backends/flock/pod/60-circuit.sh > $O/build.log 2>&1 || { tail -30 $O/build.log; exit 1; }
  mkdir -p $BIN && cp $F/target/release/flock-circuit $BIN/flock-circuit
fi
exec 8>&-
B=$BIN/flock-circuit; sha256sum $B | tee $O/binary.sha256
UV=$(command -v uv || echo $HOME/.local/bin/uv)
[ -x $UV ] || { curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1; UV=$HOME/.local/bin/uv; }
[ -x $W/py/bin/python ] || { $UV venv -q --python 3.12 $W/py && $UV pip install -q --python $W/py/bin/python numpy blake3; } || exit 1
export PATH=$W/py/bin:$PATH PYTHONPATH=$REPO/packages/verity/src:$REPO/backends/numerical/python:$REPO/backends/flock/python:$REPO/integrations/vllm:$REPO/protocols/sampled_proofs

N=${N:-32}; D=$S/stage
python3 backends/flock/pod/rec_inner.py stage --n $N --sets $S/sets --out $D > $O/stage.json 2> $O/stage.err || { tail -20 $O/stage.err; exit 1; }
cat $O/stage.json
TB=(); [ -d $D/tables ] && TB=(--tables $D/tables)

python3 -m verity_flock.rec_live --listen 127.0.0.1:0 --out $O/inner > $O/proxy.log 2>&1 &
PP=$!
ADDR=
for i in $(seq 600); do ADDR=$(awk '/^SERVING/ {print $2}' $O/proxy.log); [ -n "$ADDR" ] && break; kill -0 $PP 2>/dev/null || break; sleep 0.5; done
[ -n "$ADDR" ] || { cat $O/proxy.log; exit 1; }
/usr/bin/time -v -o $O/time-prove.txt $B prove --verifier $ADDR --instances $D/inst-$N.bin --circuit $D/circuit.txt "${TB[@]}" \
  --runs 1 --warm 0 > $O/prove.txt 2> $O/prove.err
echo "prove exit $?"
wait $PP; echo "proxy exit $?"
grep -h panicked $O/prove.err | head -3
SESS=$O/inner/session
[ -f $SESS/session.json ] || { tail -20 $O/proxy.log $O/prove.err; exit 1; }
$B replay --sessions $SESS --instances $D/pub-$N.bin --circuit $D/circuit.txt "${TB[@]}" > $O/replay.txt 2>&1
wait $LP
$LEAN_BIN verify --statement verity/flock-circuit --circuit $D/circuit.txt --public $D/pub-$N.bin --coins os --session $SESS > $O/lean.log 2>&1
echo "lean exit $?"
# the same session with rep 0's last coin changed: both verifiers must reject it
FS=$S/forged; rm -rf $FS; cp -r $SESS $FS
python3 - $FS/session.json <<'EOF'
import json, sys
p = sys.argv[1]; r = json.load(open(p))
st = [s for s in r["streams"] if s["stream"].endswith("/rep0")][0]
rd = [x for x in st["rounds"] if x["coins"]][-1]
c = rd["coins"][-1]; rd["coins"][-1] = c[:-1] + ("0" if c[-1] != "0" else "1")
json.dump(r, open(p, "w"))
EOF
$B replay --sessions $FS --instances $D/pub-$N.bin --circuit $D/circuit.txt "${TB[@]}" > $O/forged-replay.txt 2>&1
$LEAN_BIN verify --statement verity/flock-circuit --circuit $D/circuit.txt --public $D/pub-$N.bin --coins os --session $FS > $O/forged-lean.txt 2>&1
echo "forged lean exit $?"
python3 - $O <<'EOF'
import json, re, sys
from pathlib import Path
o = Path(sys.argv[1])
up = lambda f: next((json.loads(l.split("\t", 1)[1])["verdict"] for l in (o / f).read_text().splitlines() if l.startswith("UPSTREAM")), None)
lean = lambda f: next((l[:400] for l in (o / f).read_text().splitlines() if "VERDICT" in l or "accepted" in l), (o / f).read_text()[-400:])
t = (o / "time-prove.txt").read_text()
live = [json.loads(l[5:]) for l in (o / "prove.txt").read_text().splitlines() if l.startswith("LIVE")]
pub = json.loads((o / "inner" / "public.json").read_text())
s = {"stage": json.loads((o / "stage.json").read_text()), "replay": up("replay.txt"), "lean": lean("lean.log"),
     "forged_replay": up("forged-replay.txt"), "forged_lean": lean("forged-lean.txt"),
     "prover_peak_gb": int(re.search(r"Maximum resident set size \(kbytes\): (\d+)", t).group(1)) / 2**20,
     "prove_wall": re.search(r"Elapsed \(wall clock\) time.*: (\S+)", t).group(1),
     "live": {k: live[0].get(k) for k in ("prove_s", "prove_total_s", "e2e_s", "proof_bytes", "wire_sent", "wire_recv", "k_log", "dense_m")} if live else None,
     "inner": {"rounds": len(pub["rounds"]), "message_rows": pub["message_rows"], "coins": sum(r["n"] for r in pub["rounds"]),
               "caps": sum(len(r["caps"]) for r in pub["rounds"]), "streams": len(pub["streams"])}}
(o / "summary.json").write_text(json.dumps(s, indent=1))
print(json.dumps(s))
EOF
true
