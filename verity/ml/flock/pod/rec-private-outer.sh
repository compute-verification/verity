# Sourced by 87-rec-private.sh and 89-rec-private-direct.sh after rec-env.sh: V*'s outer --zk sessions. Reads $O, $B,
# $LEAN_BIN, $T (Lean's tools: core-firewall), and for a session behind the firewall $FWS and $SID.

free_port() { python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])'; }

# release_check D SID PART: the release of part PART of recursive session SID decided again by Lean's core-firewall
# (verity/firewall-release/v0) from D's transcript (FC_FIREWALL_TRANSCRIPT) on the record before it (D/record-before.jsonl)
release_check() {
  local D=$1
  python3 - $D $2 $3 <<'EOF'
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
}

# outer NAME DIR fw|dev|plain: one --zk session of V*'s statement DIR on loopback into $O/outer/NAME. fw: behind the outer
# firewall (the send check, its transcript), released as part NAME of recursive session $SID, and that release decided again
# by Lean's core-firewall; dev: a forgery's, its proofs sent though its witness fails; plain: an honest statement's, outside
# any recursive session. Then Lean's verify --zk under the verifier's program, partition and registered reads, and the live
# build's replay --zk
outer() {
  local name=$1 C=$2 mode=$3 D=$O/outer/$1 PORT=$(free_port) PIN=$(sha512sum $2/circuit.txt | cut -d' ' -f1) SP SESS
  local ENV=(FC_ZK_SELF_CHECK=skip) EXTRA=() DRAW=()
  mkdir -p $D
  [ -f $C/inst.bin ] || { echo "$name: no statement under $C"; return 1; }
  # the verifier's draw (Lean refuses a session without one, D12): OUTER_DRAW=all (default) every unit, subset:n; none: no draw
  [ "${OUTER_DRAW:-all}" = all ] && DRAW=(--draw subset:$(python3 -c 'import json, sys; print(json.loads(open(sys.argv[1], "rb").readline())["instances"])' $C/pub.bin))
  $B serve --zk --listen 127.0.0.1:$PORT --out $D/sessions --instances $C/pub.bin --circuit $C/circuit.txt --pin $PIN \
    "${DRAW[@]}" --operator loopback > $D/serve.log 2>&1 &
  SP=$!
  for i in $(seq 1800); do grep -q SERVING $D/serve.log && break; kill -0 $SP 2>/dev/null || break; sleep 1; done
  if [ $mode = fw ]; then
    cp $FWS/record.jsonl $D/record-before.jsonl
    ENV=(FC_FIREWALL=1 FC_HOST_WITNESS=1 FC_FIREWALL_TRANSCRIPT=$D/transcript)
    EXTRA=(--firewall-state $FWS --session $SID --part $name)
  elif [ $mode = plain ]; then
    ENV=()
  fi
  env "${ENV[@]}" /usr/bin/time -v -o $D/time-prove.txt $B prove --zk --verifier 127.0.0.1:$PORT --instances $C/inst.bin \
    --circuit $C/circuit.txt --warm 0 --runs 1 "${EXTRA[@]}" > $D/prove.txt 2> $D/prove.err
  echo "$name prove exit $?"
  grep -h -m3 -e panicked -e FIREWALL-REFUSED -e PROVER-STOPPED $D/prove.err
  sleep 2; kill $SP; wait $SP 2>/dev/null
  du -sb $C/inst.bin $C/pub.bin $C/circuit.txt > $D/sizes.txt
  if [ $mode = fw ]; then
    release_check $D $SID $name; echo "$name core-firewall exit $?"
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
