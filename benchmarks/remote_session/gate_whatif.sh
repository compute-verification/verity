#!/usr/bin/env bash
# The what-if line beside a gate run's line of record (gate_whatif.py), read from recorded artifacts only, so on any
# node and with no gate key: the host run SUB (its directory as vy-auditor-1 ships it, out/ beside request.json) inside
# the verifier run's artifact RUN_ART, the ledger head and pins log the auditor run AUDITOR_ART read through the gate
# key (auditor.sh's out/ledger.jsonl and out/pins.jsonl), and the prover run PROVER_ART's out/prover.json. PERIOD and
# PIN are the audit's period and challenger key, COMMIT the verifier run's recorded commit; the gate sessions' peer is
# the record's (PEER overrides it), RTT_MS the round trips replayed. Writes $OUT/gate_whatif.json and $OUT/whatif.md,
# and exits 1 when a session replayed at its own round trip isn't its record.
#
#   research run --on vy-nebius-cpu-4 --project verity --source . --cwd clone --cpus 4 --mem-gb 16 --timeout 30m \
#     --store-read --env RUN_ART=art:... --env SUB=out/host/vID --env AUDITOR_ART=art:... --env PROVER_ART=art:... \
#     --env PERIOD=P --env PIN=SHA512 --env COMMIT=SHA -- bash benchmarks/remote_session/gate_whatif.sh
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
S=$(mktemp -d)
trap 'rm -rf "$S"' EXIT
: > "$OUT/fetch.log"
fetch() {
  uv run research data fetch "$@" >> "$OUT/fetch.log" 2>&1
  local rc=$?
  echo "fetch $1 rc $rc" >> "$OUT/fetch.log"
  [ $rc = 0 ] || { cat "$OUT/fetch.log"; exit 3; }
}
fetch "${RUN_ART:?}" --to "$S/run" --path "${SUB:?}/request.json" --path "$SUB/out/verifier.json" \
  --path "$SUB/out/challenger/*"
fetch "${AUDITOR_ART:?}" --to "$S/aud" --path out/ledger.jsonl --path out/pins.jsonl
fetch "${PROVER_ART:?}" --to "$S/prover" --path out/prover.json
uv run python "$HERE/gate_whatif.py" "$S/run/$SUB/out" --prover "$S/prover/out/prover.json" \
  --host-bytes "${HOST_BYTES:-1843153879040}" --pin "${PIN:?}" --period "${PERIOD:?}" --commit "${COMMIT:?}" \
  ${PEER:+--peer "$PEER"} --ledger "$S/aud/out/ledger.jsonl" --pins "$S/aud/out/pins.jsonl" \
  ${RTT_MS:+--rtt-ms $RTT_MS} --table "$OUT/whatif.md" > "$OUT/gate_whatif.json" 2> "$OUT/gate_whatif.err"
W=$?
cat "$OUT/fetch.log" "$OUT/gate_whatif.err" "$OUT/whatif.md"
exit $W
