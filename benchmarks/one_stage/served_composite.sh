#!/usr/bin/env bash
# Each composite's Lean verdict on a served run's cases (remote's, remote-network's and onsite's), `served_verdicts.sh`'s
# step after its record, with the tree as cwd (a clone: it builds Lean in it):
#
#   bash benchmarks/one_stage/served_composite.sh SERVE_DIR OUT
#
# OUT is the run's public $OUT, and OUT/pieces its subprotocols' public pieces. First PoUS's and the wipe's cases
# (`benchmarks/pous/served_case.sh OUT/pieces`, which writes each under SERVE_DIR/private) with their Lean answers
# (pous-run-protocol and wipe-protocol, built here; without both it writes no case), PoUS's under a lease of its own of
# CASE_LEASE_MIN minutes (default 60) and CASE_MEM_GB GiB of host memory (default 170: its Python and its Lean each hold
# the whole re-encoded store), all within CASE_TIMEOUT (default 90m). Then, where the tree has `served_cases.py`, the
# cases on the window's registration record R (its docstring), each under SERVE_DIR/private: sampled proofs' one-stage
# v1 case (onsite's), remote's sampled proofs case (a5's v2 case, which only a `--zk` audit of one drawn member writes,
# in the form `one-stage`) and the inference-only case from V3c's records audit (SERVE_DIR/v3c-w*/audit, and never an
# audit that opened the advice: `prove_path/run.sh`'s PP_RECORDS audit, SERVE_DIR/sampled-proofs-served in a serve dir an
# earlier `served_verdicts.sh` left, is a development check no case reads), the latter with the policy's α and Γ read
# off the case's own division (`--from-division`), or ALPHA and GAMMA where both are set, and its log the serve's H1 log
# through R's previous record; and PoUW's period case (`served_pouw.sh`'s SERVE_DIR/private/pouw/period) once every
# window of it names R as its Commit record, refused otherwise. Each CASE line's protocol and verdict go to
# OUT/verdicts.txt, and α and Γ, which count the block's gates, stay in the case's private policy.json. Then
# `composite.py` remote, remote-network and onsite,
# each with `--exe` its product's Lean (`remote`, `remote-network` and `onsite`, built here, remote-network's in onsite's
# package), over the directories under SERVE_DIR/private and OUT/pieces/certifier (the network certifier's piece, whose
# case its leak guard released there) that hold a case.json and an answer.json whose protocol is one of that product's
# subprotocols (composite.py's COMPONENTS), but for SERVE_DIR/private/pouw, whose period case only served_cases.py
# hands on, tied to R; each within COMPOSITE_TIMEOUT (default 60m) and, where a systemd user scope is available,
# COMPOSITE_MEM_GB GiB (default 170). A case's evidence and params, and an answer's query, digests and profile, are
# witness, so composite.py's private output and every log of this step are under SERVE_DIR/private/composite/PRODUCT, on
# the node; its public output (composite.json: the composite's commitment, each case's id, params digest and commitment,
# what is missing and each outcome; and its verdict lines), which it refuses to write when it shares bytes with any
# case's evidence, is OUT/composite/PRODUCT. OUT/verdicts.txt gets each case's protocol and verdict (the CASE lines),
# each case not written by its protocol, each product's NOT_ESTABLISHED and VERDICT lines, and the host it composed on
# (the serve dir's: the served node's, or, where SERVED_NODE names another, the node a deferral's follow-on copied the
# serve dir's audit inputs to), headed COMPOSITE, and nothing else. Without served_case.sh or composite.py
# in the tree it says so and exits 1, since no product's Lean answered. It makes no verdict: exit 0 when each product's
# Lean answered, else 1. PRODUCT=remote (`served_verdicts.sh`'s) writes neither PoUS's case nor the wipe's, needs no
# served_case.sh, and runs remote's Lean alone; PRODUCT=remote-network the same with remote-network's Lean beside it.
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
  onsite) products=(remote remote-network onsite) need=(benchmarks/pous/served_case.sh benchmarks/one_stage/composite.py) ;;
  remote) products=(remote) need=(benchmarks/one_stage/composite.py) ;;
  remote-network) products=(remote remote-network) need=(benchmarks/one_stage/composite.py) ;;
  *) say "COMPOSITE: not run, PRODUCT is onsite, remote or remote-network, not $PRODUCT"; exit 1 ;;
esac
pkg() { case $1 in remote-network) echo onsite ;; *) echo "$1" ;; esac; }  # the Lean package a product's executable is in
for f in "${need[@]}"; do
  [ -f "$f" ] || { say "COMPOSITE: not run, this tree has no $f"; exit 1; }
done
mkdir -p "$P" && chmod 700 "$D/private" "$P"
exe() {  # Lean executable $1 of package dir $2, built, its path printed
  (cd "$TREE/$2" && lake build "$1") >> "$P/lean-build.log" 2>&1 \
    || { say "COMPOSITE: $1 doesn't build in $2 (its log on the node)" >&2; return 1; }
  echo "$TREE/$2/.lake/build/bin/$1"
}
scope=()
command -v systemd-run > /dev/null && systemd-run --user --scope --quiet --collect true > /dev/null 2>&1 \
  && scope=(systemd-run --user --scope --quiet --collect -p "MemoryMax=${COMPOSITE_MEM_GB:-170}G" -p MemorySwapMax=0 --)

if [ "$PRODUCT" = onsite ] && POUS_EXE=$(exe pous-run-protocol verity/core/protocols/pous/check) \
  && WIPE_EXE=$(exe wipe-protocol verity/core/devices/memory_challenger/lean); then
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
fi

if [ -f benchmarks/one_stage/served_cases.py ]; then
  ag=(--from-division)
  [ -n "${ALPHA:-}" ] && [ -n "${GAMMA:-}" ] && ag=(--alpha "$ALPHA" --gamma "$GAMMA")
  "$TPY" benchmarks/one_stage/served_cases.py --serve-dir "$D" --private "$D/private" "${ag[@]}" > "$P/served_cases.log" 2>&1
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
  e=$(exe "$x" "verity/core/protocols/$(pkg "$x")") || { bad=1; continue; }
  cases=()
  while IFS= read -r c; do cases+=(--cases "$c"); done < <("$TPY" - "$x" "$P" "$D/private/pouw" "$D/private" \
    "$OUT/pieces/certifier" <<'EOF'
import json, sys
from pathlib import Path
sys.path.insert(0, "benchmarks/one_stage")
import composite as C
product, skips, roots = sys.argv[1], [Path(s) for s in sys.argv[2:4]], [Path(r) for r in sys.argv[4:]]
ids = set(C.COMPONENTS[product])
for f in (f for root in roots if root.is_dir() for f in sorted(root.rglob("case.json"))):
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
