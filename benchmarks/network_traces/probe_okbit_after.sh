# A probe, sourced by scan_zk.sh (SC_AFTER) once scan_zk.py has proved its rounds: a prover that commits the unread
# unit's o0 as 1 when the unit computes 0.
#
# The selftest build's `output_row_committed_false` flips instance 0's first output word in its committed output row,
# recommits the row under its salt and recomputes the frame roots: on the unread member that is o0 = 1 under the
# public salt, a public file whose o0 root is the root of the one row [1]. `honest` is the same member unchanged
# (o0 = 0). Each recorded session is then held by Lean under each registered file scan_zk.py wrote for the member
# (the verifier's own, and probe_known's three), with the selftest's statement, which flock-verify refuses once
# every check has passed: a verdict that names another check is that check's refusal.
# Reads: W, OUT, SRC, SBIN, LEAN, PY, log (scan_zk.sh's, job.sh's). Writes $OUT/probe-forged.
D=$W/sczk/unread/stage-ingress-0
P=$OUT/probe-forged
mkdir -p "$P"
if [ ! -f "$D/inst-1.bin" ]; then
  log "probe-forged: no unread member staged under $D"
else
  mkdir -p "$D/rec"
  for c in honest output_row_committed_false; do
    "$SBIN" selftest --zk --instances "$D/inst-1.bin" --circuit "$D/circuit.txt" --tables "$D/tables" --only "$c" \
      --record-dir "$D/rec" --draw subset:1 > "$P/selftest-$c.out" 2> "$P/selftest-$c.err"
    log "probe-forged: selftest $c exit $?: $(grep -E '^(NEG|POS|CASE|SELFTEST)' "$P/selftest-$c.out" | cut -c1-400 | tr '\n' ' ')"
    tail -c 600 "$P/selftest-$c.err" | tr '\n' ' ' | cut -c1-600
    echo
  done
  "$PY" "$SRC/verity/protocols/flock/agreement/selftest_records.py" "$D/rec" "$D/st" > "$P/records.log" 2>&1
  log "probe-forged: records $(cat "$P/records.log" | cut -c1-200)"
  for s in "$D"/st/*/; do
    [ -d "$s" ] || continue
    case=$(basename "$s")
    pub=$D/pub-1.bin
    [ -e "$s/pub.bin" ] && pub=$s/pub.bin
    cp "$s/case.json" "$P/case-$case.json"
    head -c 4000 "$pub" | head -1 | "$PY" -c 'import json, sys; h = json.loads(sys.stdin.readline()); fv = h["frame_v3"]; print(json.dumps({"o0_root": fv["roots"]["o0"], "o0_domain": fv["domain_ids"]["o0"]}))' \
      > "$P/public-$case.json" 2>> "$P/records.log"
    for reg in registered.json registered-no-known.json registered-known-0.json registered-const-root.json; do
      [ -f "$D/$reg" ] || continue
      tag=$case-${reg%.json}
      "$LEAN" verify --zk --statement verity/flock-circuit+seed-injection --circuit "$D/circuit.txt" --public "$pub" \
        --partition "$W/sczk/unread/partition.json" --program "$W/sczk/unread/program.json" --registered "$D/$reg" \
        --session "$s" > "$P/lean-$tag.out" 2> "$P/lean-$tag.err"
      log "probe-forged: lean $tag exit $?: $(grep '^VERDICT ' "$P/lean-$tag.out" | cut -c1-600) $(tail -c 400 "$P/lean-$tag.err" | tr '\n' ' ')"
    done
  done
  cp "$D"/registered*.json "$P/" 2>/dev/null
fi
