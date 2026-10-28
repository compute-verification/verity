#!/usr/bin/env bash
# Remote's and onsite's Lean verdicts on a served run's cases, `served_verdicts.sh`'s step after its record, with the tree
# as cwd (a clone: it builds Lean in it):
#
#   bash benchmarks/one_stage/served_composite.sh SERVE_DIR OUT
#
# OUT is the run's public $OUT, and OUT/pieces its subprotocols' public pieces. First PoUS's and the wipe's cases
# (`benchmarks/pous/served_case.sh OUT/pieces`, which writes each under SERVE_DIR/private) with their Lean answers
# (pous-run-protocol and wipe-protocol, built here; without both it writes no case), PoUS's under a lease of its own of
# CASE_LEASE_MIN minutes (default 60) and CASE_MEM_GB GiB of host memory (default 170: its Python and its Lean each hold
# the whole re-encoded store), all within CASE_TIMEOUT (default 90m). Then, where the tree has `served_cases.py`,
# sampled proofs' v2 case and the registration case from the window's records audit (SERVE_DIR/v3c-w*/audit), the latter
# with the policy's α and Γ read off the case's own division (`--from-division`), or ALPHA and GAMMA where both are set,
# each under SERVE_DIR/private; each CASE line's protocol, verdict and `alpha_gamma` go to OUT/verdicts.txt. Then
# `composite.py` remote and onsite,
# each with `--exe` its product's Lean (`remote`, `onsite`, built here), over the directories under SERVE_DIR/private
# that hold a case.json and an answer.json whose protocol is one of that product's subprotocols (composite.py's
# COMPONENTS), each within COMPOSITE_TIMEOUT (default 60m) and, where a systemd user scope is available,
# COMPOSITE_MEM_GB GiB (default 170). A case's evidence and params, and an answer's query, digests and profile, are
# witness, so composite.py's private output and every log of this step are under SERVE_DIR/private/composite/PRODUCT, on
# the node; its public output (composite.json: the composite's commitment, each case's id, params digest and commitment,
# what is missing and each outcome; and its verdict lines), which it refuses to write when it shares bytes with any
# case's evidence, is OUT/composite/PRODUCT. OUT/verdicts.txt gets each case's protocol and verdict (the CASE lines),
# each case not written by its protocol, and each product's NOT_ESTABLISHED and VERDICT lines, headed COMPOSITE, and
# nothing else. Without served_case.sh or composite.py in the tree it says so and exits 1, since no product's Lean
# answered. It makes no verdict: exit 0 when each product's Lean answered, else 1.
set -uo pipefail
D=${1:?usage: served_composite.sh SERVE_DIR OUT}
OUT=${2:?usage: served_composite.sh SERVE_DIR OUT}
TREE=$PWD
PY=${PY:-/workspace/cp/venv312/bin/python}
TPY=${TPY:-$TREE/.venv/bin/python}
[ -x "$TPY" ] || TPY=$PY
P=$D/private/composite
say() { echo "$*" | tee -a "$OUT/verdicts.txt"; }
for f in benchmarks/pous/served_case.sh benchmarks/one_stage/composite.py; do
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

if POUS_EXE=$(exe pous-run-protocol verity/core/protocols/pous/check) \
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
    print("COMPOSITE", head, c["protocol"], *([c["verdict"]] if head == "CASE" else []),
          *(["(alpha_gamma " + str(c["alpha_gamma"]) + ")"] if c.get("alpha_gamma") else []))' \
    | tee -a "$OUT/verdicts.txt"
  say "COMPOSITE CASES: served_cases.py exit $rc$([ "$rc" = 0 ] || echo ', its log on the node')"
fi

bad=0
for x in remote onsite; do
  e=$(exe "$x" "verity/core/protocols/$x") || { bad=1; continue; }
  cases=()
  while IFS= read -r c; do cases+=(--cases "$c"); done < <("$TPY" - "$D/private" "$P" "$x" <<'EOF'
import json, sys
from pathlib import Path
sys.path.insert(0, "benchmarks/one_stage")
import composite as C
root, skip, product = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
ids = set(C.COMPONENTS[product])
for f in sorted(root.rglob("case.json")):
    d = f.parent
    if d == skip or skip in d.parents or not (d / "answer.json").is_file():
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
exit "$bad"
