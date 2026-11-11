# The onsite product's own pieces of a served run, the ones its private subprotocols run: the wipe's, PoUS's and the
# bandwidth audit's. `benchmarks/one_stage/served_verdicts.sh` and `served_composite.sh` source this file where
# ONSITE_PIECES names it (demos/vllm/placement.toml's window run sets it), with the tree as cwd, and call:
#
#   onsite_piece_commands   served_verdicts.sh's defaults, each kept where its variable is set (empty: none):
#       WIPE_CMD       'bash benchmarks/pous/streamed_wipe/served_wipe.sh "$1" "$2" "$3"' where the tree has it
#       POUS_CMD       'bash benchmarks/pous/served_piece.sh "$@"' where the tree has it and this node holds the memory
#                      challenger's key (~/.ssh/vy-auditor2-pous); that piece runs the bandwidth session over the store it
#                      holds and writes its answer where the bandwidth piece's goes, and each of the two cases is
#                      answered by its Lean (pous-run-protocol, remote-audit-p2-protocol), which the piece builds
#       POUS_TIMEOUT   with POUS_CMD, that piece's limit: PIECE_TIMEOUT's where set, else 120m, which leaves the Lean
#                      builds and pous-run-protocol's answer on a served case their time beside the two sessions
#       BANDWIDTH_CMD  once POUS_CMD is set, '"$PY" benchmarks/one_stage/served_bandwidth.py collect ...', which collects
#                      it; with no PoUS piece, 'bash benchmarks/one_stage/served_bandwidth.sh "$1" "$2" "$3"' where the
#                      tree has it and this node holds the bandwidth verifier's key (~/.ssh/vy-auditor2-bw) and its
#                      registered store (BW_STORE's setup.json, default /workspace/research/pous-store/mem69-v2)
#   onsite_cases            served_composite.sh's, before it composes: PoUS's and the wipe's cases as their pieces wrote
#                           them under SERVE_DIR/private (PoUS's in pous/wK, K its pous.json's window; the wipe's in
#                           wipe), each beside its Lean's answer (pous-run-protocol's, wipe-protocol's). It takes no
#                           card, encodes nothing, builds no Lean and writes no case and no answer. It reads each
#                           piece's public summary (OUT/pieces/pous/pous.json, OUT/pieces/wipe/summary.json: the
#                           window, the verdict and who decided) and the answer beside the case, never the case.
#                           OUT/verdicts.txt gets `COMPOSITE CASE PROTOCOL VERDICT` for a case served_composite.sh then
#                           composes, and `COMPOSITE NOT_WRITTEN PROTOCOL` where the piece wrote no case, no answer
#                           beside it, or an answer that isn't its Lean's or isn't of the verdict it published; why
#                           stays in SERVE_DIR/private/composite/onsite_cases.log. served_composite.sh composes every
#                           directory under SERVE_DIR/private that holds a case beside an answer, so where one of these
#                           two protocols' is not the case named here (another window's, one left from an earlier run,
#                           one answered by the Python alone), this says `COMPOSITE: not run` and exits 1 before any
#                           product is composed: of each protocol at most one case reaches composite.py, its piece's.

onsite_piece_commands() {
  # shellcheck disable=SC2016
  [ -f "$TREE/benchmarks/pous/streamed_wipe/served_wipe.sh" ] \
    && WIPE_CMD=${WIPE_CMD-'bash benchmarks/pous/streamed_wipe/served_wipe.sh "$1" "$2" "$3"'}
  # shellcheck disable=SC2016
  [ -f "$TREE/benchmarks/pous/served_piece.sh" ] && [ -f "$HOME/.ssh/vy-auditor2-pous" ] \
    && POUS_CMD=${POUS_CMD-'bash benchmarks/pous/served_piece.sh "$@"'}
  # shellcheck disable=SC2016
  if [ -n "${POUS_CMD:-}" ]; then
    export POUS_TIMEOUT=${POUS_TIMEOUT:-${PIECE_TIMEOUT:-120m}}
    BANDWIDTH_CMD=${BANDWIDTH_CMD-'"$PY" benchmarks/one_stage/served_bandwidth.py collect --serve-dir "$1" --window "$2" --out "$3"'}
  elif [ -f "$TREE/benchmarks/one_stage/served_bandwidth.sh" ] && [ -f "$HOME/.ssh/vy-auditor2-bw" ] \
    && [ -f "${BW_STORE:-/workspace/research/pous-store/mem69-v2}/setup.json" ]; then
    BANDWIDTH_CMD=${BANDWIDTH_CMD-'bash benchmarks/one_stage/served_bandwidth.sh "$1" "$2" "$3"'}
  fi
  return 0
}

onsite_cases() {
  local rc
  "$PY" - "$D/private" "$OUT/pieces" "$P" 2> "$P/onsite_cases.log" <<'EOF' | tee -a "$OUT/verdicts.txt"
import json, sys
from pathlib import Path

private, pieces, skip = (Path(a).resolve() for a in sys.argv[1:4])
VERDICTS = ("accept", "reject", "undecided")


def read(path):
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return doc if isinstance(doc, dict) else {}


def log(head, **doc):
    print(head + " " + json.dumps(doc, sort_keys=True), file=sys.stderr)


def pous(summary):
    k = summary.get("window")
    return private / "pous" / f"w{k}" if isinstance(k, int) else None


# protocol: its Lean, its piece's summary under OUT/pieces, and its piece's private directory by that summary
OWN = {"verity/pous/p2/v4/run": ("pous-run-protocol", "pous/pous.json", pous),
       "verity/wipe/v0": ("wipe-protocol", "wipe/summary.json", lambda _: private / "wipe")}

composed = {}  # what served_composite.sh composes: each directory holding a case beside an answer, by its protocol
for f in sorted(private.rglob("case.json")):
    d = f.parent
    if d != skip and skip not in d.parents and (d / "answer.json").is_file():
        composed.setdefault(str(read(d / "answer.json").get("protocol")), []).append(d)

refused = {}
for protocol, (lean, name, where) in OWN.items():
    summary = read(pieces / name)
    own = where(summary) if summary else None
    answer = read(own / "answer.json") if own else {}
    by = summary.get("decided_by")
    by = by.get("program") if isinstance(by, dict) else None
    why = None
    if not summary:
        why = f"the run has no piece of it: no {name} under {pieces}"
    elif own is None:
        why = f"{name} names no window"
    elif not (own / "case.json").is_file():
        why = f"its piece wrote no case in {own}"
    elif not (own / "answer.json").is_file():
        why = f"its piece wrote no answer beside its case in {own}"
    elif answer.get("protocol") != protocol or answer.get("verdict") not in VERDICTS:
        why = f"the answer in {own} is not one of {protocol}"
    elif answer["verdict"] != summary.get("verdict"):
        why = f"the answer in {own} is {answer['verdict']}, and its piece published {summary.get('verdict')}"
    elif by != lean:
        why = f"its piece's answer is {by}'s, not its Lean's ({lean}'s)"
    if why is None:
        log("CASE", piece=str(own), protocol=protocol, verdict=answer["verdict"])
        print("COMPOSITE CASE", protocol, answer["verdict"])
    else:
        log("NOT_WRITTEN", protocol=protocol, reason=why)
        print("COMPOSITE NOT_WRITTEN", protocol)
    others = [str(d) for d in composed.get(protocol, []) if why or d != own]
    if others:
        refused[protocol] = others
for protocol, others in refused.items():
    log("REFUSED", protocol=protocol, pieces=others)
    print(f"COMPOSITE: not run, the serve dir's private holds a case of {protocol} that is not its piece's with its "
          "Lean's answer, and every case there is composed (onsite_cases.log on the node)")
sys.exit(3 if refused else 0)
EOF
  rc=${PIPESTATUS[0]}
  case $rc in
    0) say "COMPOSITE CASES: PoUS's and the wipe's as their pieces wrote them, none made here" ;;
    3) exit 1 ;;
    *) say "COMPOSITE: not run, onsite's own cases weren't read (exit $rc, onsite_cases.log on the node)"; exit 1 ;;
  esac
}
