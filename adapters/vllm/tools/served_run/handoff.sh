# Sourced from the tree's root with D, OUT and log(): pack_handoff returns 1 when D holds no compose hand-off, 4 when it can't pack it.
pack_handoff() {
  local c=$D/private/compose why
  { [ -f "$c/inputs.txt" ] && [ -f "$c/compose.env" ]; } || { log "no compose hand-off in $c"; return 1; }
  why=$(python3 benchmarks/one_stage/served_verdicts.py witness < "$c/inputs.txt") \
    || { log "HANDOFF NOT PACKED: the list names the witness: $(echo "$why" | head -1 | cut -c1-300)"; return 4; }
  mkdir -p "$OUT/handoff"
  (cd "$D" && { cat private/compose/inputs.txt; echo private/compose/compose.env; } | sort -u \
     | tar --create --dereference --no-recursion --file="$OUT/handoff/compose.tar" --files-from=-) \
    || { log "HANDOFF NOT PACKED: tar failed"; return 4; }
  (cd "$OUT/handoff" && sha256sum compose.tar > compose.tar.sha256)
  log "HANDOFF packed: $(wc -l < "$c/inputs.txt") files, $(du -h "$OUT/handoff/compose.tar" | cut -f1); the serve dir $D stays on this node"
}
