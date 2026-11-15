#!/usr/bin/env bash
# The what-if line beside a gate run's line of record (gate_whatif.py), read from recorded artifacts only, so on any
# node and with no gate key: the host run SUB (its directory as the verifier host ships it, out/ beside request.json) inside
# the verifier run's artifact RUN_ART (SUB unset: out/ at the artifact's root, as a gate run before host runs ships it),
# the ledger head and pins log the auditor run AUDITOR_ART read through the gate key (auditor.sh's out/ledger.jsonl and
# out/pins.jsonl), and the prover run PROVER_ART's out/prover.json. PERIOD and
# PIN are the audit's period and challenger key, COMMIT the verifier run's recorded commit; the gate sessions' peer is
# the record's (PEER overrides it), RTT_MS the round trips replayed, PASS_SCALE and PASS_MS the pass models (each pass
# its own time times F, or every pass MS). Writes the line of record as auditor.sh does
# ($OUT/auditor_line.json, $OUT/credit.md) and the replay ($OUT/gate_whatif.json, $OUT/whatif.md), and exits 1 when a
# session replayed at its own round trip isn't its record.
#
#   OUT=OUT RUN_ART=art:... SUB=out/host/vID AUDITOR_ART=art:... PROVER_ART=art:... PERIOD=P PIN=SHA512 COMMIT=SHA \
#     bash benchmarks/remote_session/gate_whatif.sh      # from a clone's root
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
P=${SUB:+$SUB/}
fetch "${RUN_ART:?}" --to "$S/run" --path "${P}request.json" --path "${P}out/verifier.json" \
  --path "${P}out/challenger/*"
fetch "${AUDITOR_ART:?}" --to "$S/aud" --path out/ledger.jsonl --path out/pins.jsonl
fetch "${PROVER_ART:?}" --to "$S/prover" --path out/prover.json
MAIN=()
# main's first-parent line as auditor.sh reads it (main_line.sh), where this tree's auditor_line.py takes it (--main)
if uv run python "$HERE/auditor_line.py" --help | grep -q -- '--main FILE'; then
  # shellcheck source=benchmarks/remote_session/main_line.sh
  source "$HERE/main_line.sh" || exit 3
  main_line "$OUT/fetch.log" "$OUT/main.txt" || {
    main_refused fetch.log "no session's pin can be checked on it: no line was written" >> "$OUT/fetch.log"
    cat "$OUT/fetch.log"
    exit 3; }
  MAIN=(--main "$OUT/main.txt")
fi
uv run python "$HERE/auditor_line.py" "$S/run/${P}out" --host-bytes "${HOST_BYTES:-1843153879040}" --pin "${PIN:?}" \
  --period "${PERIOD:?}" --commit "${COMMIT:?}" ${PEER:+--peer "$PEER"} --ledger "$S/aud/out/ledger.jsonl" \
  --pins "$S/aud/out/pins.jsonl" "${MAIN[@]}" --table "$OUT/credit.md" > "$OUT/auditor_line.json" \
  2> "$OUT/auditor_line.err" || { cat "$OUT/fetch.log" "$OUT/auditor_line.err"; exit 3; }
uv run python "$HERE/gate_whatif.py" "$S/run/${P}out" --line "$OUT/auditor_line.json" \
  --prover "$S/prover/out/prover.json" ${RTT_MS:+--rtt-ms $RTT_MS} ${PASS_SCALE:+--pass-scale $PASS_SCALE} \
  ${PASS_MS:+--pass-ms $PASS_MS} --table "$OUT/whatif.md" \
  > "$OUT/gate_whatif.json" 2> "$OUT/gate_whatif.err"
W=$?
cat "$OUT/fetch.log" "$OUT/credit.md" "$OUT/gate_whatif.err" "$OUT/whatif.md"
exit $W
