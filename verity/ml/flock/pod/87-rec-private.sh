#!/usr/bin/env bash
# rec-private (lane rec-v0): a hidden netlist as the recursion's inner statement. The universal unit UniversalUnit_v1{CLS}
# (verity.ml.private_circuits.universal) runs a random netlist padded to the class, its op/src/out one row every instance
# opens (the registered program: rec_inner.program_unit, so three regions), x a row per instance, staged with plain leaves
# (rec_inner.py universal); its inner session is proved on the CPU (M0, FC_COINS=os) against the rec proxy opening a
# recursive session on the firewall's state
# (rec_live --firewall, the parts V* will prove in order), the firewall's contract (Lean flock-firewall) checks what the
# proxy let out, V* is staged over it (rec_vstage points, InnerFold.lean, rec_vstage stage under the class's cap), and each
# of V*'s statements is proved with --zk behind the outer firewall (FC_FIREWALL=1: the send check, released by the firewall
# as part NAME of the recursive session, --firewall-state), each release decided again by Lean's core-firewall
# (verity/firewall-release/v0) on the record before it, and each session verified by Lean's flock-verify verify --zk and
# replayed by the live build. The controls, each refused where it should be:
#   other        another netlist of the class (the registered one with an unread no-op gate changed: same outputs), proved
#                against the registered statement: V*'s algebra over the inner's claims at the registered statement's fold
#                (alg parts; outer sessions sent though the witness fails, FC_ZK_SELF_CHECK=skip)
#   made-up      --wired's cheater (universal.run_wired: layer 1 reads a made-up input): the verifier's Linking check on
#                the inner statement's public file, live (statement --links, before any session) and in V*'s fold
#                (InnerFold.lean --program, the setup Lean's verify runs) over a session recorded under no links; the
#                honest wired statement passes both, and neither passes the fold without the verifier's program
#   flipped-top  rec_inner.py forge --what top: V*'s reference (rec_vstar) refuses it by name; staged by a prover that skips
#                its own top checks (rec_inner.py flipped), the level-1 statement, refused by Lean's verify --zk
#   forged-top   rec_vstage --forge top: an opening that climbs to a node other than its committed top (the last level)
# Run from the verity tree (research run --cwd source). Big files stay under $FLOCK_WORK/data/rec-private/$TAG; out/ gets
# the records and summary.json.
# env: CLS=16,16,8  N=64 (instances, two a block: m = k_log + 5)  NW=8 (wired: per layer, 4 NW in all)  SEED=0  CPUS=32
#      FLOCK_WORK (as 80-rec-inner.sh)  TAG (default CLS-nN)  LEAN_FROM (a cache directory whose lean-<sha>/flock-verify
#      of the same sources is used)
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uxo pipefail
if [ -z "${RI_PINNED:-}" ]; then
  . verity/ml/flock/pod/cpu-slices.sh
  hold_slices "" "${CPUS:-32}" || exit 1
  export RI_PINNED=$SLICE_CPUS RAYON_NUM_THREADS=$(echo $SLICE_CPUS | tr ',' '\n' | wc -l)
  exec taskset -c $SLICE_CPUS bash "$0" "$@"
fi
export FC_COINS=os FLOCK_WORK=${FLOCK_WORK:-/workspace/jobs/rec-v0}
LSHA0=$(cd verity/core/service/flock && find . -name .lake -prune -o -type f -print | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-16)
FW0=$FLOCK_WORK
if [ ! -x $FW0/cache/rec-v0/lean-$LSHA0/flock-verify ]; then
  for c in ${LEAN_FROM:-} /workspace/jobs/*/cache /workspace/jobs/*/cache/*; do
    [ -x $c/lean-$LSHA0/flock-verify ] || continue
    mkdir -p $FW0/cache/rec-v0/lean-$LSHA0 && cp $c/lean-$LSHA0/flock-verify $FW0/cache/rec-v0/lean-$LSHA0/ && break
  done
fi
. verity/ml/flock/pod/rec-env.sh

CLS=${CLS:-16,16,8}; N=${N:-64}; NW=${NW:-8}; SEED=${SEED:-0}; TAG=${TAG:-${CLS//,/-}-n$N}
DD=$FW/data/rec-private/$TAG; rm -rf $DD; mkdir -p $DD $O/inner $O/vstar $O/outer || exit 1
command -v research >/dev/null && research keep $DD --owner @proofs --expires 2026-10-23 --priority low --why "rec-private $TAG" > /dev/null 2>&1
PY=verity/ml/flock/pod/rec_inner.py
FWS=$DD/firewall

# Lean's tools beside flock-verify, in one copy of the packages they need, cached by their sources: flock-firewall (the
# firewall's contract), inner-fold (tests/lean/InnerFold.lean, V*'s native side) and core-firewall (the release decision)
TSHA=$( (find verity/core/service/flock verity/core/service/firewall verity/core/protocols/interface verity/core/difftest/lean \
  -name .lake -prune -o -type f -print | LC_ALL=C sort | xargs sha256sum; cat verity/ml/flock/tests/lean/InnerFold.lean) | sha256sum | cut -c1-16)
T=$L/tools-$TSHA
(
  export PATH=$HOME/.elan/bin:$PATH
  command -v lake >/dev/null || curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none >/dev/null
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
) > $O/build-tools.log 2>&1 &
TP=$!

free_port() { python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])'; }
pub_of() { ls $1/pub-*.bin | head -1; }

# stage: the universal unit's statement and the other netlist's (u/), the wired circuit's honest and cheating ones (w/)
/usr/bin/time -v -o $O/time-stage-universal.txt python3 $PY universal --cls $CLS --n $N --seed $SEED --out $DD/u > $O/stage-universal.json 2> $O/stage-universal.err \
  || { tail -20 $O/stage-universal.err; exit 1; }
/usr/bin/time -v -o $O/time-stage-wired.txt python3 $PY wired --cls $CLS --n $NW --seed $SEED --out $DD/w > $O/stage-wired.json 2> $O/stage-wired.err \
  || { tail -20 $O/stage-wired.err; exit 1; }
cp $DD/u/stage/partition.json $O/
U=$DD/u/stage; PUB=$(pub_of $U)

# inner NAME STMT INST [firewall|-] [LINKS]: the inner session of statement STMT (its circuit and public file: the schedule of
# its public shape, Linking checked on that file under the verifier's LINKS when the circuit declares links) proved from the
# prover's file INST against the rec proxy, into $DD/inner-NAME; the live build's replay of it, and V*'s points from its coins
# (rec_vstage points) into $DD/points-NAME.json
inner() {
  local name=$1 st=$2 inst=$3 fw=${4:--} D=$O/inner/$1 I=$DD/inner-$1 pub=$(pub_of $2) FWA=() LKA=() ADDR= PP
  [ -n "${5:-}" ] && LKA=(--links $5)
  mkdir -p $D; rm -rf $I
  $B statement --instances $pub --circuit $st/circuit.txt "${LKA[@]}" > $D/statement.txt 2>&1 || { tail -5 $D/statement.txt; return 1; }
  python3 -m verity_flock.rec_live --schedule-for $D/statement.txt > $D/schedule.json || return 1
  [ $fw = firewall ] && FWA=(--firewall $FWS --parts $PARTS)
  python3 -m verity_flock.rec_live --listen 127.0.0.1:0 --out $I --schedule $D/schedule.json "${FWA[@]}" > $D/proxy.log 2>&1 &
  PP=$!
  for i in $(seq 600); do ADDR=$(awk '/^SERVING/ {print $2}' $D/proxy.log); [ -n "$ADDR" ] && break; kill -0 $PP 2>/dev/null || break; sleep 0.5; done
  [ -n "$ADDR" ] || { cat $D/proxy.log; return 1; }
  /usr/bin/time -v -o $D/time-prove.txt $B prove --verifier $ADDR --instances $inst --circuit $st/circuit.txt --runs 1 --warm 0 \
    > $D/prove.txt 2> $D/prove.err
  echo "$name inner prove exit $?"
  wait $PP; echo "$name proxy exit $?"
  [ -f $I/session/session.json ] || { tail -20 $D/proxy.log $D/prove.err; return 1; }
  $B replay --instances $pub --circuit $st/circuit.txt "${LKA[@]}" --sessions $I/session > $D/replay.txt 2>&1
  cp $I/public.json $D/ && du -sb $I/session/* $I/*.json > $D/sizes.txt
  python3 -m verity_flock.rec_vstage points $I $st/circuit.txt $pub > $DD/points-$name.json 2> $D/points.err || { tail -5 $D/points.err; return 1; }
}

# fold NAME STMT [--program FILE]: V*'s native side (InnerFold.lean: the extra claims and each rep's comb, from the statement
# and the session's coins) at $DD/points-NAME.json, into $DD/fold-NAME.json; its status
fold() {
  local name=$1 st=$2; shift 2
  /usr/bin/time -v -o $O/vstar/time-fold-$name.txt $T/inner-fold --plain-leaves --partition $st/partition.json "$@" verity/flock-circuit \
    $st/circuit.txt $(pub_of $st) $DD/points-$name.json os > $DD/fold-$name.json 2> $O/vstar/fold-$name.err
  local rc=$?; echo "fold $name exit $rc"; echo $rc > $O/vstar/fold-$name.exit
  return $rc
}

# outer NAME DIR fw|dev: one --zk session of V*'s statement DIR on loopback into $O/outer/NAME. fw: behind the outer firewall
# (the send check, its transcript), released as part NAME of recursive session $SID, and that release decided again by Lean's
# core-firewall on the record before it; dev: a forgery's, its proofs sent though its witness fails. Then Lean's verify --zk
# under the verifier's program, partition and registered reads, and the live build's replay --zk
outer() {
  local name=$1 C=$2 mode=$3 D=$O/outer/$1 PORT=$(free_port) PIN=$(sha512sum $2/circuit.txt | cut -d' ' -f1) SP SESS
  local ENV=(FC_ZK_SELF_CHECK=skip) EXTRA=()
  mkdir -p $D
  [ -f $C/inst.bin ] || { echo "$name: no statement under $C"; return 1; }
  $B serve --zk --listen 127.0.0.1:$PORT --out $D/sessions --instances $C/pub.bin --circuit $C/circuit.txt --pin $PIN \
    --operator loopback > $D/serve.log 2>&1 &
  SP=$!
  for i in $(seq 1800); do grep -q SERVING $D/serve.log && break; kill -0 $SP 2>/dev/null || break; sleep 1; done
  if [ $mode = fw ]; then
    cp $FWS/record.jsonl $D/record-before.jsonl
    ENV=(FC_FIREWALL=1 FC_HOST_WITNESS=1 FC_FIREWALL_TRANSCRIPT=$D/transcript)
    EXTRA=(--firewall-state $FWS --session $SID --part $name)
  fi
  env "${ENV[@]}" /usr/bin/time -v -o $D/time-prove.txt $B prove --zk --verifier 127.0.0.1:$PORT --instances $C/inst.bin \
    --circuit $C/circuit.txt --warm 0 --runs 1 "${EXTRA[@]}" > $D/prove.txt 2> $D/prove.err
  echo "$name prove exit $?"
  grep -h -m3 -e panicked -e FIREWALL-REFUSED -e PROVER-STOPPED $D/prove.err
  sleep 2; kill $SP; wait $SP 2>/dev/null
  du -sb $C/inst.bin $C/pub.bin $C/circuit.txt > $D/sizes.txt
  if [ $mode = fw ]; then
    python3 - $D $SID $name <<'EOF'
import json, sys
from pathlib import Path
d, sid, part = Path(sys.argv[1]), sys.argv[2], sys.argv[3]

def canonical_json(v):
    # Lean's Flock.canon: serde_json's escaping, so a non-ASCII character (the transcript's τ) is its UTF-8, not \u
    return json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()

ts = sorted((d / "transcript").glob("session-*.json"))
pubs = sorted((d / "transcript").glob("public-*.json"))
if len(ts) != 1 or len(pubs) != 1:
    raise SystemExit(f"{d}: {len(ts)} transcripts and {len(pubs)} public inputs, not one each")
now = int(ts[0].stat().st_mtime * 1000)
(d / "params.json").write_bytes(canonical_json(json.loads(pubs[0].read_text())))
(d / "evidence.json").write_bytes(canonical_json({"now_ms": now, "part": part, "session": sid, "transcript": json.loads(ts[0].read_text())}))
EOF
    [ -f $D/evidence.json ] && $T/core-firewall check $D/params.json $D/record-before.jsonl $D/evidence.json > $D/core-firewall.txt 2> $D/core-firewall.err
    echo "$name core-firewall exit $?"
  fi
  SESS=$(find $D/sessions -name session.json -printf '%h\n' 2>/dev/null | sort | head -1)
  [ -n "$SESS" ] || { echo "$name: no session"; return 1; }
  du -sb $SESS/*.bin >> $D/sizes.txt
  /usr/bin/time -v -o $D/time-lean.txt $LEAN_BIN verify --zk --statement verity/flock-circuit --circuit $C/circuit.txt --public $C/pub.bin \
    --partition $C/partition.json --program $C/program.json --registered $C/registered.json --session $SESS > $D/lean.log 2> $D/lean.err
  echo "$name lean exit $?"
  /usr/bin/time -v -o $D/time-replay.txt $B replay --zk --instances $C/pub.bin --circuit $C/circuit.txt --sessions $SESS > $D/replay.txt 2>&1
  echo "$name replay exit $?"
}

# the other netlist first: its inner session (against the registered statement) and V*'s fold over the registered statement
# give the claims, so the class's cap and the recursive session's parts are known before the honest session opens one
inner other $U $DD/u/other/inst-$N.bin || exit 1
wait $LP; echo "flock-verify build exit $?"; [ -x $LEAN_BIN ] || { tail -20 $O/build-lean.log; exit 1; }
wait $TP; echo "tools build exit $?"; [ -x $T/inner-fold ] && [ -x $T/core-firewall ] || { tail -30 $O/build-tools.log; exit 1; }
fold other $U || exit 1
python3 - $U/circuit.txt $N $DD/fold-other.json > $DD/cap.json <<'EOF' || exit 1
import json, sys
from pathlib import Path
from verity_flock import rec_shape as RS
cap = RS.Cap.of(Path(sys.argv[1]), int(sys.argv[2]), 2 + len(json.loads(Path(sys.argv[3]).read_text())["extras"]))
print(json.dumps({"cap": cap.spec(), "shape": [cap.shape.m, cap.shape.k_log, cap.shape.claims], "parts": list(RS.outer(cap)),
                  "statements": RS.outer(cap)}))
EOF
cp $DD/cap.json $O/
CAP=$(python3 -c "import json, sys; print(json.load(open(sys.argv[1]))['cap'])" $DD/cap.json)
PARTS=$(python3 -c "import json, sys; print(','.join(json.load(open(sys.argv[1]))['parts']))" $DD/cap.json)

# honest: the inner session through the firewall (a recursive session opened with PARTS), the firewall's contract on what it
# let out, the inner verified by Lean as far as it goes (plain leaves: refused until soundness covers them), V*'s fold
inner honest $U $U/inst-$N.bin firewall || exit 1
SID=$(python3 -c "import json, sys; print(json.load(open(sys.argv[1]))['session'])" $DD/inner-honest/firewall.json) || exit 1
echo $SID > $O/inner/honest/firewall-session.txt
python3 verity/ml/flock/verifier/firewall_agree.py --lean $T/flock-firewall --out $O/inner/firewall-contract --sessions $DD/inner-honest \
  > $O/inner/firewall-contract.txt 2>&1; echo "inner firewall contract exit $?"
$LEAN_BIN verify --statement verity/flock-circuit+plain-leaves --circuit $U/circuit.txt --public $PUB --partition $U/partition.json \
  --coins os --session $DD/inner-honest/session > $O/inner/honest/lean.log 2> $O/inner/honest/lean.err; echo "inner lean exit $?"
fold honest $U || exit 1
python3 -m verity_flock.rec_vstar $DD/inner-honest > $O/vstar/vstar-honest.json 2> $O/vstar/vstar-honest.err; echo "vstar honest exit $?"

# V*'s statements: honest (every level and the algebra, under the class's cap), the wrong top, and the other netlist's algebra
V=$DD/v; mkdir -p $V
vstage() {
  local name=$1; shift
  /usr/bin/time -v -o $O/vstar/time-$name.txt python3 -m verity_flock.rec_vstage stage "$@" --cap $CAP > $O/vstar/$name.json 2> $O/vstar/$name.err
  echo "vstage $name exit $?"
}
vstage honest $DD/inner-honest $DD/fold-honest.json $V --points $DD/points-honest.json --salts $V/acc-salts.bin
[ -f $V/acc-salts.bin ] || { tail -20 $O/vstar/honest.err; exit 1; }
vstage forged-top $DD/inner-honest $DD/fold-honest.json $V --points $DD/points-honest.json --salts $V/acc-salts.bin --forge top
vstage other $DD/inner-other $DD/fold-other.json $DD/vo --points $DD/points-other.json --salts $DD/vo-acc-salts.bin --levels none
(cd $V && du -sb $(find . -name '*.bin' -o -name circuit.txt | LC_ALL=C sort)) > $O/vstar/sizes.txt

# the flipped top: rec_inner.py forge --what top on a copy of the honest session; V*'s reference and staging refuse it by name,
# and a prover that skips its own checks stages the level it flipped
F=$DD/inner-flipped; mkdir -p $F && cp $DD/inner-honest/public.json $DD/inner-honest/private.json $F/ && ln -s $DD/inner-honest/session $F/session
python3 $PY forge --what top --inner $F > $O/vstar/forge-top.json || exit 1
python3 -m verity_flock.rec_vstar $F > $O/vstar/vstar-flipped.json 2> $O/vstar/vstar-flipped.err; echo "vstar flipped exit $?"
vstage flipped-checked $F $DD/fold-honest.json $DD/vf-checked --points $DD/points-honest.json --salts $V/acc-salts.bin --levels 1 --no-algebra
python3 $PY flipped --inner $F --fold $DD/fold-honest.json --points $DD/points-honest.json --salts $V/acc-salts.bin --out $DD/vf \
  > $O/vstar/flipped.json 2> $O/vstar/flipped.err; echo "flipped stage exit $?"

# the wired circuit: the honest statement's inner session under the verifier's links, V*'s fold with the verifier's program
# (Linking) and without it; the cheater's public file under the verifier's links (the live build's Linking refuses it before
# any session), its inner session under none (a verifier that checks no Linking records it), and V*'s fold of it with the program
W=$DD/w/stage
inner wired $W $(ls $W/inst-*.bin) - $W/links.json || exit 1
fold wired $W --program $W/program.desc
cp $DD/points-wired.json $DD/points-wired-noprogram.json
fold wired-noprogram $W
mkdir -p $O/inner/made-up
$B statement --instances $(pub_of $DD/w/made-up-input) --circuit $DD/w/made-up-input/circuit.txt --links $W/links.json \
  > $O/inner/made-up/own-links.txt 2>&1; echo "made-up statement under the verifier's links exit $?"
inner made-up $DD/w/made-up-input $(ls $DD/w/made-up-input/inst-*.bin) - $DD/w/no-links.json || exit 1
fold made-up $DD/w/made-up-input --program $DD/w/made-up-input/program.desc

# the outer sessions: honest through the firewall in the recursive session's order, then the controls'
for p in ${PARTS//,/ }; do
  case $p in L*) C=$V/$p ;; alg-p*) C=$V/alg/p${p#alg-p} ;; esac
  outer $p $C fw
done
LAST=$(ls -d $V/forged-top/L* | head -1)
outer forged-top-$(basename $LAST) $LAST dev
outer flipped-top-L1 $DD/vf/forged-flipped-top/L1 dev
for C in $(ls -d $DD/vo/alg/p* | sort -V); do outer other-alg-$(basename $C) $C dev; done
cp $FWS/record.jsonl $O/firewall-record.jsonl
bash verity/ml/flock/pod/88-rec-private-public.sh TAG=$TAG > $O/public.log 2>&1; echo "public views exit $?"

python3 - $O <<'EOF'
import json, re, sys
from pathlib import Path
o = Path(sys.argv[1])

def text(p):
    return p.read_text() if p.exists() else ""

def tv(p, key):
    m = re.search(key + r".*: (\S+)", text(p))
    return m.group(1) if m else None

def gb(p):
    rss = tv(p, r"Maximum resident set size \(kbytes\)")
    return round(int(rss) / 2**20, 3) if rss else None

def lean(p):
    t = text(p)
    at = t.find("VERDICT ")
    if at < 0:
        return {"accepted": None, "why": (t + text(p.with_suffix(".err")))[-500:]}
    v = json.JSONDecoder(strict=False).raw_decode(t[at + 8:])[0]
    return {**{k: v.get(k) for k in ("accepted", "why", "registered", "partition")}, "every_check_passed": passed(v)}

def passed(v):
    """A verdict after every check passed: accepted, or refused for its form only (`ProvedScope.check`, which runs last)."""
    return bool(v.get("accepted")) or str(v.get("why", "")).endswith(": a statement form the soundness proof does not cover")

def up(p):
    return next((json.loads(l.split("\t", 1)[1]) for l in text(p).splitlines() if l.startswith("UPSTREAM")), text(p)[-400:] or None)

def live(d):
    rows = [json.loads(l[5:]) for l in text(d / "prove.txt").splitlines() if l.startswith("LIVE")]
    served = [json.loads(l.split("\t", 1)[1]) for l in text(d / "serve.log").splitlines() if l.startswith("SESSION\t")]
    r = rows[0] if rows else {}
    return {"accepted": r.get("accepted"), "why": r.get("why"), "every_check_passed": passed(r) if r else None,
            "prove_s": r.get("prove_s"), "prove_total_s": r.get("prove_total_s"),
            "e2e_s": r.get("e2e_s"), "proof_bytes": r.get("proof_bytes"), "k_log": r.get("k_log"), "instances": r.get("instances"),
            "serve": [{k: s.get(k) for k in ("accepted", "why", "verify_s")} for s in served],
            "stops": [l for l in text(d / "prove.err").splitlines() if l.startswith(("FIREWALL-REFUSED", "PROVER-STOPPED"))][:4],
            "prove_wall": tv(d / "time-prove.txt", r"Elapsed \(wall clock\) time"), "prover_peak_gb": gb(d / "time-prove.txt")}

s = {"stage": {k: json.loads(text(o / f"stage-{k}.json") or "null") for k in ("universal", "wired")},
     "cap": json.loads(text(o / "cap.json") or "null"), "inner": {}, "fold": {}, "vstar": {}, "outer": {},
     "public": json.loads(text(o / "public" / "public.json") or "null")}
for d in sorted(p for p in (o / "inner").iterdir() if p.is_dir() and (p / "prove.txt").exists()):
    s["inner"][d.name] = {**live(d), "replay": up(d / "replay.txt"), "sizes": text(d / "sizes.txt")}
s["inner"]["honest"]["lean_plain_leaves"] = lean(o / "inner" / "honest" / "lean.log")
if "made-up" in s["inner"]:
    s["inner"]["made-up"]["under_the_verifiers_links"] = next(
        (json.loads(l.split("\t", 1)[1]) for l in text(o / "inner" / "made-up" / "own-links.txt").splitlines() if l.startswith("REFUSED\t")),
        text(o / "inner" / "made-up" / "own-links.txt")[-400:])
s["inner"]["honest"]["firewall_contract"] = {"result": json.loads(text(o / "inner" / "firewall-contract" / "result.json") or "null"),
                                             "agreement": text(o / "inner" / "firewall-contract" / "agreement.tsv")}
for f in sorted((o / "vstar").glob("fold-*.exit")):
    n = f.stem[5:]
    s["fold"][n] = {"exit": int(text(f).strip() or -1), "err": text(o / "vstar" / f"fold-{n}.err")[-600:],
                    "wall": tv(o / "vstar" / f"time-fold-{n}.txt", r"Elapsed \(wall clock\) time")}
for n in ("honest", "forged-top", "other", "flipped-checked", "flipped"):
    t = text(o / "vstar" / f"{n}.json")
    s["vstar"][n] = {"stage": json.loads(t.splitlines()[0]) if t.strip() else text(o / "vstar" / f"{n}.err")[-800:],
                     "wall": tv(o / "vstar" / f"time-{n}.txt", r"Elapsed \(wall clock\) time")}
for n in ("vstar-honest", "vstar-flipped"):
    s["vstar"][n] = (text(o / "vstar" / f"{n}.json") + text(o / "vstar" / f"{n}.err"))[-800:]
s["vstar"]["forge_top"] = json.loads(text(o / "vstar" / "forge-top.json") or "null")
for d in sorted(p for p in (o / "outer").iterdir() if p.is_dir()):
    s["outer"][d.name] = {"live": live(d), "lean": lean(d / "lean.log"), "lean_wall": tv(d / "time-lean.txt", r"Elapsed \(wall clock\) time"),
                          "replay": up(d / "replay.txt"), "core_firewall": text(d / "core-firewall.txt").strip() or text(d / "core-firewall.err")[-300:] or None,
                          "sizes": text(d / "sizes.txt")}
(o / "summary.json").write_text(json.dumps(s, indent=1))
print(json.dumps({k: {n: (v.get("lean"), v.get("core_firewall")) for n, v in s["outer"].items()} if k == "outer" else None for k in ("outer",)}))
EOF
true
