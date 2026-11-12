#!/usr/bin/env bash
# The first recursive session's inner side (rec_inner.py, verity_flock.rec_live): RoPE d64 staged as M0's statement, proved
# on the CPU by the unmodified prover (non-zk, under INNER_COINS) against the rec proxy, whose live-coin part (the challenger) sees
# only salted tops, message commitments and the schedule (the statement's public shape's) and answers Register with the
# verifier's full sample (stage/draw.json, rec_live.full_draw: every unit); every coin of the session is the challenger's coin
# tree, sampled at Hello and opened round by round (rec_live --coin-tree, from TREE_FROM). Then the recorded session replayed by
# upstream's verifier (flock-circuit replay), V*'s hashing over it (verity_flock.rec_vstar), and two forgeries (a changed
# challenge, a changed query coin) that both must reject. Lean's plain verify takes a coin tree of the session's own streams
# (--coins tree), not one with the inner phase's sparse and stacked streams added, so the session is not verified on its own
# here (pod 85's INNER_LEAN likewise). Run from the verity tree (research run
# --cwd source). Outputs under out/: host.txt, stage.json, statement.txt, schedule.json, time-prove.txt, prove.txt/.err,
# proxy.log, inner/ (session/, public.json, private.json, stage/:
# the statement's circuit.txt, public file and the verity/partition/v1 object it binds with its program (partition.json,
# program.json), claims.json: each rep's extra claims and circuit fold from Lean's
# InnerClaims.lean, which V*'s algebra reads), replay.txt, vstar.json, forged-*, summary.json, tree-source.txt.
# env: N=64 (instances: RoPE heads, one VU each; 64 at G=2 is m=27)  CPUS=32  FLOCK_WORK (as 76-read-gadget.sh)
#      TREE_FROM= (the challenger's device, HOST:PORT, each session's coin tree sampled and opened there, rec_live --tree-from;
#      TREE_SECRET_FILE= its session secret, hex; empty: this host's OS samples it, rec_live --tree-test-os, a test's coins)
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uxo pipefail
if [ -z "${RI_PINNED:-}" ]; then
  . verity/protocols/flock/pod/cpu-slices.sh
  hold_slices "" "${CPUS:-32}" || exit 1
  export RI_PINNED=$SLICE_CPUS RAYON_NUM_THREADS=$(echo $SLICE_CPUS | tr ',' '\n' | wc -l)
  exec taskset -c $SLICE_CPUS bash "$0" "$@"
fi
. verity/protocols/flock/pod/rec-env.sh
# the inner statement's prover, statement and replay: on the challenger's coin tree (rec_live.INNER_COIN_SPEC: the prover's
# own streams, then the inner phase's sparse and stacked ones), as pod 85's
INNER_COINS="FC_COINS=tree FC_COIN_STREAMS=sparse/rep0,sparse/rep1,stacked/rep0"
TREE=(--tree-test-os)
[ -z "${TREE_FROM:-}" ] || TREE=(--tree-from $TREE_FROM ${TREE_SECRET_FILE:+--tree-secret-file $TREE_SECRET_FILE})
echo "${TREE_FROM:-test-os}" > $O/tree-source.txt

N=${N:-64}; D=$S/stage
# InnerClaims.lean as an executable beside flock-verify (interpreted it takes about half an hour), cached by its sources
CSHA=$( (echo $LSHA; cat verity/protocols/flock/prover/tests/lean/InnerClaims.lean) | sha256sum | cut -c1-16)
CLAIMS_BIN=$L/claims-$CSHA/inner-claims
(
  export PATH=$HOME/.elan/bin:$PATH
  exec 9>$L/claims.lock; flock 9
  if [ ! -x $CLAIMS_BIN ]; then
    C=$L/claims-src; rm -rf $C && C=$(python3 verity/protocols/flock/pod/lake_closure.py copy verity/protocols/flock/verifier $C) && cp verity/protocols/flock/prover/tests/lean/InnerClaims.lean $C/ \
      && printf '\n[[lean_exe]]\nname = "inner-claims"\nroot = "InnerClaims"\n' >> $C/lakefile.toml \
      && (cd $C && lake build inner-claims) && mkdir -p $L/claims-$CSHA && cp $C/.lake/build/bin/inner-claims $CLAIMS_BIN
  fi
) > $O/build-claims.log 2>&1 &
CLP=$!
python3 verity/protocols/flock/prover/pod/rec_inner.py stage --n $N --sets $S/sets --out $D > $O/stage.json 2> $O/stage.err || { tail -20 $O/stage.err; exit 1; }
cat $O/stage.json
TB=(); [ -d $D/tables ] && TB=(--tables $D/tables)
PART=(--partition $D/partition.json)

# the schedule the proxy holds the session to, from the statement's public shape before the session (rec_live --schedule-for)
env $INNER_COINS $B statement --instances $D/pub-$N.bin --circuit $D/circuit.txt "${TB[@]}" > $O/statement.txt 2>&1 || { tail -5 $O/statement.txt; exit 1; }
python3 -m verity_flock.rec_live --schedule-for $O/statement.txt > $O/schedule.json || exit 1
# the session's send check is Lean's flock-firewall live, built beside the Lean verifier
wait $LP; [ -x $FIREWALL_BIN ] || { tail -20 $O/build-lean.log; exit 1; }
python3 -m verity_flock.rec_live --listen 127.0.0.1:0 --out $O/inner --schedule $O/schedule.json --draw $D/draw.json \
  --coin-tree "${TREE[@]}" --lean $FIREWALL_BIN > $O/proxy.log 2>&1 &
PP=$!
ADDR=
for i in $(seq 600); do ADDR=$(awk '/^SERVING/ {print $2}' $O/proxy.log); [ -n "$ADDR" ] && break; kill -0 $PP 2>/dev/null || break; sleep 0.5; done
[ -n "$ADDR" ] || { cat $O/proxy.log; exit 1; }
/usr/bin/time -v -o $O/time-prove.txt env $INNER_COINS $B prove --verifier $ADDR --instances $D/inst-$N.bin --circuit $D/circuit.txt "${TB[@]}" \
  --runs 1 --warm 0 > $O/prove.txt 2> $O/prove.err
echo "prove exit $?"
wait $PP; echo "proxy exit $?"
grep -h panicked $O/prove.err | head -3
SESS=$O/inner/session
[ -f $SESS/session.json ] || { tail -20 $O/proxy.log $O/prove.err; exit 1; }
env $INNER_COINS $B replay --instances $D/pub-$N.bin --circuit $D/circuit.txt "${TB[@]}" --sessions $SESS > $O/replay.txt 2>&1
python3 -m verity_flock.rec_vstar $O/inner > $O/vstar.json 2> $O/vstar.err; echo "vstar exit $?"
mkdir -p $O/inner/stage && cp $D/circuit.txt $D/pub-$N.bin $D/partition.json $D/program.json $D/draw.json $O/inner/stage/
wait $CLP
/usr/bin/time -v -o $O/time-claims.txt $CLAIMS_BIN "${PART[@]}" verity/flock-circuit $D/circuit.txt $D/pub-$N.bin $SESS tree > $O/inner/claims.json 2> $O/claims.err
echo "claims exit $?"
# forgeries upstream's replay and V*'s reference must reject: rep 0's first challenge, and the first query coin of rep 0's
# level 0 (its position moves); both go to replay through session.json and to V*'s reference through public.json
for F in challenge query; do
  FS=$S/forged-$F; rm -rf $FS; cp -r $O/inner $FS
  python3 verity/protocols/flock/prover/pod/rec_inner.py forge --what $F --inner $FS
  env $INNER_COINS $B replay --instances $D/pub-$N.bin --circuit $D/circuit.txt "${TB[@]}" --sessions $FS/session > $O/forged-$F-replay.txt 2>&1
  python3 -m verity_flock.rec_vstar $FS > $O/forged-$F-vstar.json 2>&1; echo "forged $F vstar exit $?"
done
python3 - $O <<'EOF'
import json, re, sys
from pathlib import Path
o = Path(sys.argv[1])
up = lambda f: next((json.loads(l.split("\t", 1)[1])["verdict"] for l in (o / f).read_text().splitlines() if l.startswith("UPSTREAM")), None)
t = (o / "time-prove.txt").read_text()
live = [json.loads(l[5:]) for l in (o / "prove.txt").read_text().splitlines() if l.startswith("LIVE")]
pub = json.loads((o / "inner" / "public.json").read_text())
s = {"stage": json.loads((o / "stage.json").read_text()), "replay": up("replay.txt"),
     "vstar": json.loads((o / "vstar.json").read_text() or "null"),
     "forged": {f: {"replay": up(f"forged-{f}-replay.txt"),
                    "vstar": (o / f"forged-{f}-vstar.json").read_text()[-300:]} for f in ("challenge", "query")},
     "prover_peak_gb": int(re.search(r"Maximum resident set size \(kbytes\): (\d+)", t).group(1)) / 2**20,
     "prove_wall": re.search(r"Elapsed \(wall clock\) time.*: (\S+)", t).group(1),
     "live": {k: live[0].get(k) for k in ("prove_s", "prove_total_s", "e2e_s", "proof_bytes", "wire_sent", "wire_recv", "k_log", "dense_m")} if live else None,
     "claims": (lambda c: {"m": c["m"], "k_log": c["k_log"], "extras": [len(r["extras"]) for r in c["reps"]],
                           "wall": re.search(r"Elapsed \(wall clock\) time.*: (\S+)", (o / "time-claims.txt").read_text()).group(1)})(
         json.loads((o / "inner" / "claims.json").read_text())) if (o / "inner" / "claims.json").stat().st_size else (o / "claims.err").read_text()[-400:],
     "inner": {"rounds": len(pub["rounds"]), "message_rows": pub["message_rows"], "coins": sum(r["n"] for r in pub["rounds"]),
               "tops": sum(len(r["tops"]) for r in pub["rounds"]), "level0": sum(r["level0"] for r in pub["rounds"]), "streams": len(pub["streams"])}}
(o / "summary.json").write_text(json.dumps(s, indent=1))
print(json.dumps(s))
EOF
true
