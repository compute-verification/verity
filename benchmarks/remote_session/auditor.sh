#!/usr/bin/env bash
# The analysis of record for a gate run: auditor_line.py's credited line over the verifier run RUN, its dir on this
# machine (the CPU node with the lane's gate key), copied to scratch with any files the node evicted to the
# store restored (run-files.evicted.json: only verifier.json and the challenger's files, the line's inputs; so the job
# reads the store).
# PERIOD and PIN are the audit's period and challenger key, COMMIT the verifier run's recorded commit. Through the
# lane's gate key it reads the ledger (`ledger`), a head beside each session's, and the challenger host's pins log
# (`pins`). A gate that refuses `pins` serves no log: then no session is pinned, and the line credits
# zero. It reads main's first-parent history as gate.sh's probe does (main_line.sh: in a run's clone from the node's
# bare repository, never the source root's frozen copy) into $OUT/main.txt (`--main`): a pin off that line likewise
# credits zero, and a fetch that fails refuses (exit 3) with no line. Writes $OUT/auditor_line.json and $OUT/credit.md,
# each session's checks and the line, which the Lean verifier computes (`line_by` lean: auditor_line.py builds
# remote-audit-verifier, so lake must be on the PATH, and refuses the line where line.py's differs). RUN may instead be
# a host run's directory as the verifier host ships it (its gate's `fetch`, out/ beside request.json): the sessions'
# challenger peer is the record's (verifier.json's, else request.json's --challenger-peer), and PEER overrides it.
# LOCKDOWN, if set, is infra's record of the provider's lockdown of the gate node (auditor_line.py --lockdown).
# GATE is the challenger host as USER@HOST, by default research@ the address VY_CHALLENGER_1_HOST holds; with neither,
# nothing runs.
#
#   OUT=OUT RUN=DIR PERIOD=P PIN=SHA512 COMMIT=SHA bash benchmarks/remote_session/auditor.sh      # from a clone's root
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
GATE=${GATE:-research@${VY_CHALLENGER_1_HOST:?the address of the challenger host, or GATE as USER@HOST}}
KEY=${GATE_KEY:-$HOME/.ssh/vy-gate-memory-accounting}
S=$(mktemp -d)
trap 'rm -rf "$S"' EXIT
cp -a "${RUN:?}/out" "$S/" && chmod -R u+w "$S"
[ ! -f "$RUN/request.json" ] || cp "$RUN/request.json" "$S/"
: > "$OUT/fetch.log"
# the eviction record is RUN's own, or its run's when RUN is a host dir inside one (a fire's out/host/V), with paths
# under that run's dir
R=$(cd "$RUN" && pwd)
EV=$R
until [ -f "$EV/run-files.evicted.json" ] || [ -f "$EV/job.json" ] || [ "$EV" = / ]; do EV=$(dirname "$EV"); done
if [ -f "$EV/run-files.evicted.json" ]; then
  P=${R#"$EV"}; P=${P#/}; P=${P:+$P/}
  art=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['artifact'])" \
    "$EV/run-files.evicted.json")
  uv run research data fetch "$art" --to "$S/restore" --path "${P}out/verifier.json" --path "${P}out/challenger/*" \
    > "$OUT/restore.log" 2>&1
  echo "restore $art ${P:-./} rc $?" >> "$OUT/fetch.log"
  [ ! -d "$S/restore/${P}out" ] || cp -a "$S/restore/${P}out/." "$S/out/"
fi
gate() { ssh -o BatchMode=yes -o IdentitiesOnly=yes -i "$KEY" "$GATE" "$@"; }
gate ledger > "$OUT/ledger.jsonl"
L=$?
echo "ledger rc $L" >> "$OUT/fetch.log"
[ $L = 0 ] || exit 3
PINS=()
if gate pins > "$OUT/pins.jsonl"; then
  PINS=(--pins "$OUT/pins.jsonl")
  echo "pins rc 0, $(grep -c . "$OUT/pins.jsonl") lines" >> "$OUT/fetch.log"
else
  echo "pins rc $?: the gate serves no pins log ($(head -c 400 "$OUT/pins.jsonl"))" >> "$OUT/fetch.log"
  mv "$OUT/pins.jsonl" "$OUT/pins.refused"
fi
# shellcheck source=benchmarks/remote_session/main_line.sh
source "$HERE/main_line.sh" || exit 3
main_line "$OUT/fetch.log" "$OUT/main.txt" || {
  main_refused fetch.log "no session's pin can be checked on it: no line was written" >> "$OUT/fetch.log"
  cat "$OUT/fetch.log"
  exit 3; }
echo "main from $MAIN_SHOWN, $(grep -c . "$OUT/main.txt") first-parent commits" >> "$OUT/fetch.log"
uv run python "$HERE/auditor_line.py" "$S/out" --host-bytes "${HOST_BYTES:-1843153879040}" --pin "${PIN:?}" \
  --period "${PERIOD:?}" --commit "${COMMIT:?}" ${PEER:+--peer "$PEER"} --ledger "$OUT/ledger.jsonl" "${PINS[@]}" \
  --main "$OUT/main.txt" ${LOCKDOWN:+--lockdown "$LOCKDOWN"} \
  --table "$OUT/credit.md" > "$OUT/auditor_line.json" 2> "$OUT/auditor_line.err"
W=$?
cat "$OUT/fetch.log" "$OUT/credit.md"
exit $W
