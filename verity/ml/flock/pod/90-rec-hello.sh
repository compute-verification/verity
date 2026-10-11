#!/usr/bin/env bash
# Scratch job (cursor/rec-c3-hello-integration-8d57, never a PR): main + #1793 + #1786 + #1798 (cursor/rec-session-hello-8c79).
# 88-rec-c3-inner.sh N=16 SPARSE=18,13,6,0 STACKED=1 with the relay that keeps the prover's Hello (R's root a round
# commitment on stacked/rep0), then over its inner session: the two records' Hellos, rec-transcript (innerCommits), rec-coins
# (check 5) on the relay's record and on the prover's, L3 staged (c3_e2e.py stage level3), proved by C-Flock --zk under the
# verifier's draw (serve --draw subset:N), replayed by upstream, and Lean's verify under --rec-inner (plain) with check 5's
# inputs on the prover's record (option (b)), with and without the inner transcript, the transcript with R's registered top
# (inner/registration.json, --rec-inner-registration) and refused without it or with one a bit off. Outputs under out/: honest/
# (88's), e2e/ (fixture/: the inner session's files as rec-c3-inner holds them, with registration.json), exits.txt.
#   research run --on vy-cpu-1 --mem-gb 40 --source . --cwd source --declared-output 'out/*' -- \
#     python3 tools/verity/check/slot.py --by rec-hello -- bash $0
set -uxo pipefail
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
R0=$RESEARCH_RUN_DIR; RID=${RESEARCH_RUN_ID:-local-$$}; mkdir -p $R0/out
K=/workspace/jobs/rec-hello/$RID; mkdir -p $K || exit 1
RESEARCH_RUN_DIR=$R0/out/honest RESEARCH_RUN_ID=$RID-honest KEEP=$K/honest bash verity/ml/flock/pod/88-rec-c3-inner.sh N=16 \
  SPARSE=18,13,6,0 STACKED=1 > $R0/out/88.log 2>&1
echo "88 exit $?" | tee -a $R0/out/exits.txt

export RI_PINNED=$(python3 -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0)))))')
export RAYON_NUM_THREADS=$(echo $RI_PINNED | tr ',' '\n' | wc -l)
. verity/ml/flock/pod/rec-env.sh
wait $LP
E=$O/e2e; mkdir -p $E; X=$O/exits.txt
H=$O/honest/out; ST=$K/honest/stage; IC=$ST/circuit.txt; IP=$ST/pub-16.bin; IQ=$ST/partition.json; IG=$ST/program.json
[ -f $H/inner/session/session.json ] && [ -f $IQ ] || { echo "no inner session or partition" | tee -a $X; exit 1; }
sha256sum $LEAN_BIN $B > $E/bins.sha256

python3 - $H/inner > $E/hello.json <<'EOF'
import json, sys
from pathlib import Path
from verity_flock import rec_live as RL
d = Path(sys.argv[1])
v, p = (json.loads((d / s / "session.json").read_text()) for s in ("session", "prover"))
pub = json.loads((d / "public.json").read_text())
print(json.dumps({"same_bytes": v["hello"] == p["hello"], "canonical": p["hello"] == RL.canonical(json.loads(p["hello"])),
                  "tables": json.loads(v["hello"])["tables"], "roots": sorted(pub["link"]["roots"]), "hello": v["hello"],
                  "stacked_round0_tops": [len(t) for t in next(r for r in pub["rounds"] if r["stream"] == "stacked/rep0")["tops"]],
                  "zero_coin_rounds": [[s["stream"], r["k"]] for s in v["streams"] for r in s["rounds"] if not r["n"]],
                  "registration": (reg := json.loads((d / "registration.json").read_text())) | {"registered": sorted(reg["registered"])},
                  "registered_top_is_r_binding_round_top": reg["registered"]["stacked"]["top"]
                  == next(r for r in pub["rounds"] if r["stream"] == "stacked/rep0" and r["k"] == 0)["tops"][0]}))
EOF
echo "hello exit $?" | tee -a $X

REG=$H/inner/registration.json; BADREG=$E/registration-bad.json
python3 -c '
import json, sys
r = json.load(open(sys.argv[1])); t = r["registered"]["stacked"]["top"]; c = bytearray.fromhex(t[0]); c[-1] ^= 1; t[0] = c.hex()
print(json.dumps(r))' $REG > $BADREG
mkdir -p $E/fixture && cp -r $H/inner/public.json $H/inner/private.json $H/inner/prover $H/inner/session $REG $H/fold.json $H/points.json $E/fixture/
for s in prover session; do
  $LEAN_BIN rec-transcript --rec-inner-transcript $H/inner/public.json --rec-inner-session $H/inner/$s --rec-inner-registration $REG \
    > $E/rec-transcript-$s.txt 2>&1
  echo "rec-transcript $s exit $?" | tee -a $X
done
$LEAN_BIN rec-transcript --rec-inner-transcript $H/inner/public.json --rec-inner-session $H/inner/prover > $E/rec-transcript-noreg.txt 2>&1
echo "rec-transcript prover-noreg exit $? (refused: 1)" | tee -a $X
$LEAN_BIN rec-transcript --rec-inner-transcript $H/inner/public.json --rec-inner-session $H/inner/prover --rec-inner-registration $BADREG \
  > $E/rec-transcript-badreg.txt 2>&1
echo "rec-transcript prover-badreg exit $? (refused: 1)" | tee -a $X
for s in session prover; do
  $LEAN_BIN rec-coins --rec-inner-circuit $IC --rec-inner-public $IP --rec-inner-partition $IQ --rec-inner-session $H/inner/$s \
    > $E/rec-coins-$s.json 2> $E/rec-coins-$s.err
  echo "rec-coins $s exit $?" | tee -a $X
done

V3=$S/v-l3; SALTS=$S/acc-salts-l3.bin
/usr/bin/time -v -o $E/time-stage-l3.txt python3 verity/ml/flock/pod/c3_e2e.py stage level3 --inner $H/inner --fold $H/fold.json \
  --points $H/points.json --circuit $IC --salts $SALTS --out $V3 > $E/stage-l3.json 2> $E/stage-l3.err
echo "stage l3 exit $?" | tee -a $X
C=$V3/L3
[ -f $C/circuit.txt ] || { tail -20 $E/stage-l3.err; exit 1; }

$LEAN_BIN rec-inner --circuit $IC --public $IP --partition $IQ --program $IG > $E/rec-inner-plain.json 2> $E/rec-inner.err
echo "rec-inner exit $?" | tee -a $X
$LEAN_BIN rec-statements --rec-inner $E/rec-inner-plain.json > $E/rec-statements.txt 2>&1
python3 - $E $C <<'EOF' > $E/own.txt
import json, sys
from pathlib import Path
from verity_flock import rec_bindings as RB
e, c = map(Path, sys.argv[1:])
(line,) = [json.loads(ln[9:]) for ln in (e / "rec-statements.txt").read_text().splitlines() if ln.startswith("RECVSTAR ")]
names = [s[0] for s in line["statements"]]
with (c / "circuit.txt").open() as f:
    unit = f.readline().split()[1]
ports = json.loads((c / "registered.json").read_text())["ports"]
(e / "rec-rows.json").write_text(json.dumps({"format": "verity/registered-reads/v0", "ports": {"acc0": ports["acc"]}}))
(e / "rec-bindings.json").write_text(json.dumps(RB.bindings(json.loads((c / "partition.json").read_text()),
                                                             json.loads((c / "program.json").read_text()))))
print(f"--rec-inner {e}/rec-inner-plain.json --rec-part {names.index(unit)} --rec-bindings {e}/rec-bindings.json --rec-rows {e}/rec-rows.json")
EOF
echo "own exit $?" | tee -a $X

D=$E/prove; mkdir -p $D
NI=$(python3 -c 'import json, sys; print(json.loads(open(sys.argv[1], "rb").readline())["instances"])' $C/pub.bin)
DRAW=${DRAW:-subset:$NI}; echo "$DRAW" > $D/draw.txt
free_port() { python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])'; }
PIN=$(sha512sum $C/circuit.txt | cut -d' ' -f1) PORT=$(free_port)
$B serve --zk --listen 127.0.0.1:$PORT --out $D/sessions --instances $C/pub.bin --circuit $C/circuit.txt --pin $PIN --draw $DRAW \
  --operator loopback > $D/serve.log 2>&1 &
SP=$!
for i in $(seq 1800); do grep -q SERVING $D/serve.log && break; kill -0 $SP 2>/dev/null || break; sleep 1; done
/usr/bin/time -v -o $D/time-prove.txt $B prove --zk --verifier 127.0.0.1:$PORT --instances $C/inst.bin --circuit $C/circuit.txt \
  --warm 0 --runs 1 > $D/prove.txt 2> $D/prove.err
echo "prove exit $?" | tee -a $X
kill $SP; wait $SP 2>/dev/null
SESS=$(find $D/sessions -name session.json -printf '%h\n' | sort | head -1)
[ -n "$SESS" ] || { tail -20 $D/serve.log $D/prove.err; exit 1; }
python3 -c 'import json, sys; print(json.dumps(json.load(open(sys.argv[1]))["unit_draw"]))' $SESS/session.json > $D/unit-draw.json
$B replay --zk --instances $C/pub.bin --circuit $C/circuit.txt --sessions $SESS > $D/replay.txt 2>&1; echo "replay exit $?" | tee -a $X

LV=(--zk --statement verity/flock-circuit --circuit $C/circuit.txt --public $C/pub.bin --partition $C/partition.json --program $C/program.json
    --registered $C/registered.json $(cat $E/own.txt))
IN=(--rec-inner-circuit $IC --rec-inner-public $IP --rec-inner-partition $IQ)
v() {  # v NAME ARGS...: Lean verify on the drawn session
  local n=$1; shift
  /usr/bin/time -v -o $D/time-lean-$n.txt $LEAN_BIN verify "${LV[@]}" "$@" --session $SESS > $D/lean-$n.log 2>&1
  echo "lean $n exit $?" | tee -a $X
}
v prover-transcript "${IN[@]}" --rec-inner-session $H/inner/prover --rec-inner-transcript $H/inner/public.json --rec-inner-registration $REG
v prover-transcript-noreg "${IN[@]}" --rec-inner-session $H/inner/prover --rec-inner-transcript $H/inner/public.json
v prover-transcript-badreg "${IN[@]}" --rec-inner-session $H/inner/prover --rec-inner-transcript $H/inner/public.json --rec-inner-registration $BADREG
v prover "${IN[@]}" --rec-inner-session $H/inner/prover
cp $C/registered.json $C/partition.json $D/; head -1 $C/circuit.txt | cut -c1-300 > $D/unit.txt; du -sb $C/* > $D/sizes.txt
for f in $D/lean-*.log; do echo "$(basename $f): $(grep -h '^VERDICT' $f | cut -c1-600)"; done > $E/verdicts.txt
for f in $D/lean-*.log; do echo "== $f"; grep -v "^VERDICT" $f | tail -3; done >> $E/verdicts.txt
cat $X
