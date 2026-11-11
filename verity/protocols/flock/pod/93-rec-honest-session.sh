#!/usr/bin/env bash
# The recorded honest session `tests/test_honest_reqs.py` holds to Lean's `honestReqs`: 80-rec-inner.sh's inner session
# alone (RoPE d64 staged at N instances, proved on the CPU by the unmodified prover, FC_COINS=os, against rec_live under
# Lean's `flock-firewall live`), with the ZK server's stdin and stdout kept as they pass through a tee around the binary
# (rec_live unchanged): the schedule and every request rec_live relayed, in order, one hex line each, and each answer with
# the salts the ZK server drew for it, after the schedule's line and answer of serve's load check, which starts the binary
# once before the session. Run from the verity tree (research run --cwd source). Outputs under out/: fixture/
# (the tree to put as fixture/v1: firewall-in.txt, firewall-out.txt, and rec_live's public.json, the challenger's live-coin part's record),
# inner/ (rec_live's session), stage.json, statement.txt, schedule.json, proxy.log, prove.txt/.err, summary.json.
# Pins its own slices where it can take slice locks (node 1); on a vy-cpu pod, run it in a check slot (--pool), which pins it.
# env: N=1 (RoPE heads: 1 to 8 are m=25, the smallest fast100 takes)  CPUS=32  FLOCK_WORK (as 80-rec-inner.sh; where neither
#      /workspace nor /workspace/jobs is writable, /workspace/cache/flock, as 71-gemm-slowdown.sh's)
# Refused: the prover has no per-round OS coins (FC_COINS is tree, or unset for the seed), and `flock-firewall live`'s
# inner session takes a Hello that names no coins (`Flock.Firewall.step`), so no honest session can be recorded until the
# ZK server's inner session takes the challenger's coin tree.
echo "93-rec-honest-session: refused: the prover has no FC_COINS=os, and flock-firewall live's inner Hello names no coins" >&2
exit 2
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uxo pipefail
if [ -z "${RI_PINNED:-}" ]; then
  if mkdir -p "${SLICE_LOCKS:-/workspace/jobs/slices}" 2>/dev/null; then
    . verity/protocols/flock/pod/cpu-slices.sh
    hold_slices "" "${CPUS:-32}" || exit 1
    export RI_PINNED=$SLICE_CPUS RAYON_NUM_THREADS=$(echo $SLICE_CPUS | tr ',' '\n' | wc -l)
    exec taskset -c $SLICE_CPUS bash "$0" "$@"
  fi
  export RI_PINNED=slot RAYON_NUM_THREADS=$(nproc)
  [ -z "${FLOCK_WORK:-}" ] && [ ! -w /workspace ] && export FLOCK_WORK=/workspace/cache/flock
fi
export FC_COINS=os
. verity/protocols/flock/pod/rec-env.sh

N=${N:-1}; D=$S/stage; X=$O/fixture; mkdir -p $X
python3 verity/protocols/flock/prover/pod/rec_inner.py stage --n $N --sets $S/sets --out $D > $O/stage.json 2> $O/stage.err || { tail -20 $O/stage.err; exit 1; }
TB=(); [ -d $D/tables ] && TB=(--tables $D/tables)
$B statement --instances $D/pub-$N.bin --circuit $D/circuit.txt "${TB[@]}" > $O/statement.txt 2>&1 || { tail -5 $O/statement.txt; exit 1; }
python3 -m verity_flock.rec_live --schedule-for $O/statement.txt > $O/schedule.json || exit 1
wait $LP; [ -x $FIREWALL_BIN ] || { tail -20 $O/build-lean.log; exit 1; }
TEE=$S/flock-firewall-tee
printf '#!/usr/bin/env bash\ntee -a %q | %q "$@" | tee -a %q\n' $X/firewall-in.txt $FIREWALL_BIN $X/firewall-out.txt > $TEE
chmod +x $TEE
python3 -m verity_flock.rec_live --listen 127.0.0.1:0 --out $O/inner --schedule $O/schedule.json --draw $D/draw.json \
  --lean $TEE > $O/proxy.log 2>&1 &
PP=$!
ADDR=
for i in $(seq 600); do ADDR=$(awk '/^SERVING/ {print $2}' $O/proxy.log); [ -n "$ADDR" ] && break; kill -0 $PP 2>/dev/null || break; sleep 0.5; done
[ -n "$ADDR" ] || { cat $O/proxy.log; exit 1; }
/usr/bin/time -v -o $O/time-prove.txt $B prove --verifier $ADDR --instances $D/inst-$N.bin --circuit $D/circuit.txt "${TB[@]}" \
  --runs 1 --warm 0 > $O/prove.txt 2> $O/prove.err
echo "prove exit $?"
wait $PP; echo "proxy exit $?"
[ -f $O/inner/public.json ] || { tail -20 $O/proxy.log $O/prove.err; exit 1; }
cp $O/inner/public.json $X/
python3 - $O <<'EOF'
import json, re, sys
from pathlib import Path
o = Path(sys.argv[1])
x = o / "fixture"
ins, outs = (x / "firewall-in.txt").read_text().splitlines(), (x / "firewall-out.txt").read_text().splitlines()
# two firewalls: serve's load check of the schedule (its line alone), then the session
starts = [i for i, line in enumerate(ins) if line.startswith("{")]
assert starts == [0, 1] and ins[0] == ins[1], f"firewall starts at lines {starts}: not serve's load check and one session"
ins, outs = ins[1:], outs[1:]
reqs = [bytes.fromhex(h) for h in ins[1:]]
pub = json.loads((x / "public.json").read_text())
tags = {}
for r in reqs:
    tags[r[0]] = tags.get(r[0], 0) + 1
s = {"stage": json.loads((o / "stage.json").read_text()), "end": pub["end"], "rounds": len(pub["rounds"]),
     "requests": len(reqs), "answers": len(outs) - 1, "by_tag": tags, "request_bytes": sum(map(len, reqs)),
     "stops": [a for a in outs if '"stop"' in a][:1],
     "fixture_bytes": {f.name: f.stat().st_size for f in sorted(x.iterdir())},
     "prove_wall": re.search(r"Elapsed \(wall clock\) time.*: (\S+)", (o / "time-prove.txt").read_text()).group(1)}
(o / "summary.json").write_text(json.dumps(s, indent=1))
print(json.dumps(s))
sys.exit(0 if s["end"] == "finish" and s["answers"] == s["requests"] and not s["stops"] else 1)
EOF
