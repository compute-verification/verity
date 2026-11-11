# The onsite product's own pieces of a served run, the ones its private subprotocols run: the wipe's, PoUS's and the
# bandwidth audit's. `benchmarks/one_stage/served_verdicts.sh` and `served_composite.sh` source this file where
# ONSITE_PIECES names it (demos/vllm/placement.toml's window run sets it), with the tree as cwd, and call:
#
#   onsite_piece_commands   served_verdicts.sh's defaults, each kept where its variable is set (empty: none):
#       WIPE_CMD       'bash benchmarks/pous/streamed_wipe/served_wipe.sh "$1" "$2" "$3"' where the tree has it
#       POUS_CMD       'bash benchmarks/pous/served_piece.sh "$@"' where the tree has it and this node holds the memory
#                      challenger's key (~/.ssh/vy-auditor2-pous); that piece runs the bandwidth session over the store it
#                      holds and writes its answer where the bandwidth piece's goes
#       BANDWIDTH_CMD  once POUS_CMD is set, '"$PY" benchmarks/one_stage/served_bandwidth.py collect ...', which collects
#                      it; with no PoUS piece, 'bash benchmarks/one_stage/served_bandwidth.sh "$1" "$2" "$3"' where the
#                      tree has it and this node holds the bandwidth verifier's key (~/.ssh/vy-auditor2-bw) and its
#                      registered store (BW_STORE's setup.json, default /workspace/research/pous-store/mem69-v2)
#   onsite_cases            served_composite.sh's: PoUS's and the wipe's cases (`served_case.sh OUT/pieces`, which writes
#                           each under SERVE_DIR/private) with their Lean answers (pous-run-protocol and wipe-protocol,
#                           built with served_composite.sh's `exe`; without both it writes no case), PoUS's under a lease
#                           of its own of CASE_LEASE_MIN minutes (default 60) and CASE_MEM_GB GiB of host memory (default
#                           170: its Python and its Lean each hold the whole re-encoded store), all within CASE_TIMEOUT
#                           (default 90m); each CASE line's protocol and verdict go to OUT/verdicts.txt.

onsite_piece_commands() {
  # shellcheck disable=SC2016
  [ -f "$TREE/benchmarks/pous/streamed_wipe/served_wipe.sh" ] \
    && WIPE_CMD=${WIPE_CMD-'bash benchmarks/pous/streamed_wipe/served_wipe.sh "$1" "$2" "$3"'}
  # shellcheck disable=SC2016
  [ -f "$TREE/benchmarks/pous/served_piece.sh" ] && [ -f "$HOME/.ssh/vy-auditor2-pous" ] \
    && POUS_CMD=${POUS_CMD-'bash benchmarks/pous/served_piece.sh "$@"'}
  # shellcheck disable=SC2016
  if [ -n "${POUS_CMD:-}" ]; then
    BANDWIDTH_CMD=${BANDWIDTH_CMD-'"$PY" benchmarks/one_stage/served_bandwidth.py collect --serve-dir "$1" --window "$2" --out "$3"'}
  elif [ -f "$TREE/benchmarks/one_stage/served_bandwidth.sh" ] && [ -f "$HOME/.ssh/vy-auditor2-bw" ] \
    && [ -f "${BW_STORE:-/workspace/research/pous-store/mem69-v2}/setup.json" ]; then
    BANDWIDTH_CMD=${BANDWIDTH_CMD-'bash benchmarks/one_stage/served_bandwidth.sh "$1" "$2" "$3"'}
  fi
  return 0
}

onsite_cases() {
  local rc
  POUS_EXE=$(exe pous-run-protocol verity/protocols/pous/verifier/check) \
    && WIPE_EXE=$(exe wipe-protocol verity/protocols/pous/device/lean) || return 0
  GPU_LEASE_WHO=${GPU_LEASE_WHO:-vllm-composed} LEASE_MIN=${CASE_LEASE_MIN:-60} MEM_GB=${CASE_MEM_GB:-170} PY=$PY \
    POUS_EXE=$POUS_EXE WIPE_EXE=$WIPE_EXE timeout -k 1m "${CASE_TIMEOUT:-90m}" bash benchmarks/pous/served_case.sh \
    "$OUT/pieces" > "$P/served_case.log" 2>&1
  rc=$?
  grep -h '^CASE {' "$P/served_case.log" | "$PY" -c '
import json, sys
for line in sys.stdin:
    c = json.loads(line[len("CASE "):])
    print("COMPOSITE CASE", c["protocol"], c["verdict"])' | tee -a "$OUT/verdicts.txt"
  say "COMPOSITE CASES: served_case.sh exit $rc$([ "$rc" = 0 ] || echo ', its log on the node')"
}
