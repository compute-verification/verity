#!/usr/bin/env bash
# Remote's and onsite's Lean verdicts on a served run's cases, `served_verdicts.sh`'s step after its record, with the tree
# as cwd (a clone: it builds Lean in it):
#
#   bash benchmarks/one_stage/served_composite.sh SERVE_DIR OUT
#
# OUT is the run's public $OUT, and OUT/pieces its subprotocols' public pieces. First, under PRODUCT=onsite, onsite's own
# cases (PoUS's and the wipe's, each under SERVE_DIR/private beside its Lean's answer, as its piece wrote it), named by
# `onsite_cases` in the file ONSITE_PIECES names (`served_verdicts.sh`'s, which exports it). Then, where the tree has `served_cases.py`,
# sampled proofs' v2 case and the registration case from the window's records audit (SERVE_DIR/v3c-w*/audit; in a run
# without V3c whose H1 windows weren't tied, so with no SERVE_DIR/h1/log.jsonl, `served_verdicts.sh`'s step 3's,
# SERVE_DIR/sampled-proofs-served), the latter with the policy's α and Γ read off the case's own division
# (`--from-division`), or ALPHA and GAMMA where both are set, each under SERVE_DIR/private; each CASE line's protocol
# and verdict go to OUT/verdicts.txt, and α and Γ, which count the block's gates, stay in the case's private policy.json.
# Then `composite.py` remote and onsite, each with `--exe` its product's Lean (`remote`, `onsite`, built here), over the
# directories under SERVE_DIR/private that hold a case.json and an answer.json whose protocol is one of that product's
# subprotocols (composite.py's COMPONENTS), but for ONSITE_SKIP's, which `onsite_cases` sets to those of a PoUS or wipe
# case that is not its piece's with its Lean's answer, and, under H1's tie (a SERVE_DIR/h1/log.jsonl), for
# SERVE_DIR/private/pouw, whose period case reaches a product only tied to the record H1 named; each within
# COMPOSITE_TIMEOUT (default 60m) and, where a systemd user scope is available,
# COMPOSITE_MEM_GB GiB (default 170). A case's evidence and params, and an answer's query, digests and profile, are
# witness, so composite.py's private output and every log of this step are under SERVE_DIR/private/composite/PRODUCT, on
# the node; its public output (composite.json: the composite's commitment, each case's id, params digest and commitment,
# what is missing and each outcome; and its verdict lines), which it refuses to write when it shares bytes with any
# case's evidence, is OUT/composite/PRODUCT. OUT/verdicts.txt gets each case's protocol and verdict (the CASE lines),
# each case not written or left out by its protocol, each product's NOT_ESTABLISHED and VERDICT lines, and the host it composed on
# (the serve dir's: the served node's, or, where SERVED_NODE names another, the node a deferral's follow-on copied the
# serve dir's audit inputs to), headed COMPOSITE, and nothing else. Without composite.py in the tree, or under
# PRODUCT=onsite without ONSITE_PIECES's file or where its `onsite_cases` itself fails (its Python's exit isn't 0, or
# its list of the directories it left out can't be read), it says so (`COMPOSITE: not run`) and exits 1, since no
# product's Lean answered. It exits 1 only where the step itself fails, not for a piece's summary or answer that doesn't
# parse: that makes its protocol NOT_WRITTEN, its case is not composed, the step goes on, and onsite's NOT_ESTABLISHED
# line names the protocol as missing. It makes no verdict: exit 0 when each product's Lean answered, else 1.
# PRODUCT=remote (`served_verdicts.sh`'s) reads no PoUS or wipe case and no ONSITE_PIECES, and runs remote's Lean alone.
set -uo pipefail
D=${1:?usage: served_composite.sh SERVE_DIR OUT}
OUT=${2:?usage: served_composite.sh SERVE_DIR OUT}
TREE=$PWD
PY=${PY:-/workspace/cp/venv312/bin/python}
TPY=${TPY:-$TREE/.venv/bin/python}
[ -x "$TPY" ] || TPY=$PY
P=$D/private/composite
say() { echo "$*" | tee -a "$OUT/verdicts.txt"; }
PRODUCT=${PRODUCT:-onsite}
case $PRODUCT in
  onsite) products=(remote onsite) ;;
  remote) products=(remote) ;;
  *) say "COMPOSITE: not run, PRODUCT is onsite or remote, not $PRODUCT"; exit 1 ;;
esac
[ -f benchmarks/one_stage/composite.py ] || { say "COMPOSITE: not run, this tree has no benchmarks/one_stage/composite.py"; exit 1; }
if [ "$PRODUCT" = onsite ]; then
  if [ -z "${ONSITE_PIECES:-}" ]; then why="it is unset"; else why="$ONSITE_PIECES is not one"; fi
  [ -n "${ONSITE_PIECES:-}" ] && [ -f "$ONSITE_PIECES" ] \
    || { say "COMPOSITE: not run, onsite's own cases come from ONSITE_PIECES's file, and $why"; exit 1; }
  # shellcheck source=/dev/null
  . "$ONSITE_PIECES"
fi
mkdir -p "$P" && chmod 700 "$D/private" "$P"
exe() {  # Lean executable $1 of package dir $2, built, its path printed
  (cd "$TREE/$2" && lake build "$1") >> "$P/lean-build.log" 2>&1 \
    || { say "COMPOSITE: $1 doesn't build in $2 (its log on the node)" >&2; return 1; }
  echo "$TREE/$2/.lake/build/bin/$1"
}
scope=()
command -v systemd-run > /dev/null && systemd-run --user --scope --quiet --collect true > /dev/null 2>&1 \
  && scope=(systemd-run --user --scope --quiet --collect -p "MemoryMax=${COMPOSITE_MEM_GB:-170}G" -p MemorySwapMax=0 --)

[ "$PRODUCT" != onsite ] || onsite_cases

if [ -f benchmarks/one_stage/served_cases.py ]; then
  ag=(--from-division)
  [ -n "${ALPHA:-}" ] && [ -n "${GAMMA:-}" ] && ag=(--alpha "$ALPHA" --gamma "$GAMMA")
  where=(--serve-dir "$D")
  compgen -G "$D/v3c-w*/audit/window/case.json" > /dev/null || [ -f "$D/h1/log.jsonl" ] \
    || { [ -f "$D/sampled-proofs-served/window/case.json" ] && where=(--audit "$D/sampled-proofs-served"); }
  "$TPY" benchmarks/one_stage/served_cases.py "${where[@]}" --private "$D/private" "${ag[@]}" > "$P/served_cases.log" 2>&1
  rc=$?
  grep -hE '^(CASE|NOT_WRITTEN) \{' "$P/served_cases.log" | "$PY" -c '
import json, sys
for line in sys.stdin:
    head, _, body = line.partition(" ")
    c = json.loads(body)
    print("COMPOSITE", head, c["protocol"], *([c["verdict"]] if head == "CASE" else []))' \
    | tee -a "$OUT/verdicts.txt"
  say "COMPOSITE CASES: served_cases.py exit $rc$([ "$rc" = 0 ] || echo ', its log on the node')"
fi

bad=0
for x in "${products[@]}"; do
  e=$(exe "$x" "verity/protocols/$x/verifier") || { bad=1; continue; }
  cases=() skips=("$P" ${ONSITE_SKIP[@]+"${ONSITE_SKIP[@]}"})
  [ -f "$D/h1/log.jsonl" ] && skips+=("$D/private/pouw")
  while IFS= read -r c; do cases+=(--cases "$c"); done < <("$TPY" - "$D/private" "$x" "${skips[@]}" <<'EOF'
import json, sys
from pathlib import Path
sys.path.insert(0, "benchmarks/one_stage")
import composite as C
root, product, skips = Path(sys.argv[1]), sys.argv[2], [Path(s) for s in sys.argv[3:]]
ids = set(C.COMPONENTS[product])
for f in sorted(root.rglob("case.json")):
    d = f.parent
    if any(d == s or s in d.parents for s in skips) or not (d / "answer.json").is_file():
        continue
    try:
        protocol = json.loads((d / "answer.json").read_text())["protocol"]
    except (ValueError, KeyError, TypeError):
        continue
    if protocol in ids:
        print(d)
EOF
)
  "${scope[@]}" timeout -k 1m "${COMPOSITE_TIMEOUT:-60m}" "$TPY" benchmarks/one_stage/composite.py "$x" "${cases[@]}" \
    --private-out "$P/$x/private" --public-out "$OUT/composite/$x" --exe "$e" > "$P/$x.log" 2>&1
  r=$?
  grep -E '^(NOT_ESTABLISHED|VERDICT) ' "$P/$x.log" | sed "s/^/COMPOSITE $x: /" | tee -a "$OUT/verdicts.txt"
  case $r in
    0 | 1 | 2) ;;
    65) say "COMPOSITE $x: composite.py wrote no public line (exit 65: they share bytes with a case's evidence)"; bad=1 ;;
    *) say "COMPOSITE $x: composite.py exit $r, its log on the node"; bad=1 ;;
  esac
done
if [ "${SERVED_NODE:-$(hostname)}" = "$(hostname)" ]; then
  say "COMPOSITE: composed on $(hostname -s), which holds the serve dir, from the cases the served node wrote there, not from the parties that produced them"
else
  say "COMPOSITE: composed on $(hostname -s), not the served node $SERVED_NODE: from a prover-side copy of its serve dir's audit inputs and the cases written here from it, not from the parties that produced them"
fi
exit "$bad"
