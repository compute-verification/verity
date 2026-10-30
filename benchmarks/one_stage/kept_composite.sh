#!/usr/bin/env bash
# Remote's and onsite's Lean verdicts (`remote check`, `onsite check`) on a served run's kept records, through
# composite.py. Run in the tree, on a CPU node (the Lean builds), with OUT set:
#     bash benchmarks/one_stage/kept_composite.sh SRC [K]
# SRC is the served run's out/ (`served_verdicts.sh`'s $OUT), or a .tgz of the part of it read here: serve/windows.jsonl
# and serve/h1/, sampled-proofs/served/window/, and pieces/NAME/ for each kept piece. K is the window (default 0).
# It builds remote, onsite, pouw-check, pouw-period and sampled-proofs-one-stage-zk, then writes each case private in
# scratch (0700, files 0600; a case's evidence is witness):
#  - PoUW's: pouw_piece.py v1 (remote) and period (onsite, and remote once composite.py lists it among remote's
#    components) from the serve dir, each MISSING with why where the run holds no case of it. With pieces/pouw/ kept,
#    the period is built from it, H1_KEY_SHA512, main.txt (or MAIN_FILE) and the drawn calls' statements and live
#    evidence (POUW_STATEMENTS, POUW_LIVE, files on this node), its REFERENCE line credit's own verdict;
#  - sampled proofs v2's: v2_case.py on the kept window, with its own Lean answer;
#  - every kept piece directory pieces/NAME/ holding a case (case.json, or NAME.case.json as the wipe writes it) beside its
#    own answer.json (the wipe's, PoUS's once served_case.sh has written its case), entered as kept, onsite only.
# composite.py then composes and checks each product; $OUT/remote and $OUT/onsite get only its public output (verdict
# lines, commitments, params digests), and $OUT/kept_composite.txt every line printed here.
set -uo pipefail
OUT=${OUT:?set OUT}
SRC=${1:?SRC: the served run out/ dir or a .tgz of it}
K=${2:-0}
TREE=$PWD
export PATH=$HOME/.elan/bin:$HOME/.local/bin:$PATH
mkdir -p "$OUT"
say() { echo "$*" | tee -a "$OUT/kept_composite.txt"; }
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
if [ -f "$SRC" ]; then mkdir -p "$W/src" && tar xzf "$SRC" -C "$W/src" && SRC=$W/src; fi
PY=${PYTHON:-}
if [ -z "$PY" ]; then
  (uv venv -q "$W/venv" && VIRTUAL_ENV=$W/venv uv pip install -q numpy) > "$OUT/venv.log" 2>&1 || { say "VENV failed"; exit 3; }
  PY=$W/venv/bin/python
fi
export PYTHONPATH=$TREE:$TREE/benchmarks/numerical/python:$TREE/verity/core/difftest/python:$TREE/verity/ml/flock/python
say "TREE $(git -C "$TREE" rev-parse HEAD 2>/dev/null || echo '?') SRC $1 window $K"
build() {  # DIR TARGET
  local t0=$SECONDS
  (cd "$TREE/$1" && lake build "$2") > "$OUT/build-$2.log" 2>&1
  local rc=$?
  say "BUILD $2 (in $1): exit $rc in $((SECONDS - t0)) s"
  return $rc
}
build verity/core/protocols/remote remote || exit 3
build verity/core/protocols/onsite onsite || exit 3
build verity/core/protocols/pouw/lean/check pouw-check || exit 3
build verity/core/protocols/pouw/lean/check pouw-period || exit 3
build verity/core/service/sampled_proofs sampled-proofs-one-stage-zk || exit 3
B=.lake/build/bin
REMOTE=$TREE/verity/core/protocols/remote/$B/remote ONSITE=$TREE/verity/core/protocols/onsite/$B/onsite
POUW=$TREE/verity/core/protocols/pouw/lean/check/$B/pouw-check PERIOD=$TREE/verity/core/protocols/pouw/lean/check/$B/pouw-period
V2=$TREE/verity/core/service/sampled_proofs/$B/sampled-proofs-one-stage-zk
P=$W/cases
log() { "$@" 2>&1 | tee -a "$OUT/kept_composite.txt"; return "${PIPESTATUS[0]}"; }
log "$PY" benchmarks/one_stage/pouw_piece.py v1 --serve-run "$SRC/serve" --window "$K" --out "$P/remote/pouw" --exe "$POUW"
PERIOD_IN=()
if [ -d "$SRC/pieces/pouw" ]; then
  PERIOD_IN=(--pouw "$SRC/pieces/pouw" --key-sha512 "${H1_KEY_SHA512:?set H1_KEY_SHA512, the period key the gate pins}"
             --main "${MAIN_FILE:-$SRC/main.txt}" ${POUW_STATEMENTS:+--statements "$POUW_STATEMENTS"}
             ${POUW_LIVE:+--live-evidence "$POUW_LIVE"})
fi
log "$PY" benchmarks/one_stage/pouw_piece.py period --serve-run "$SRC/serve" --window "$K" --out "$P/period/pouw-period" \
  --exe "$PERIOD" "${PERIOD_IN[@]}"
REMOTE_CASES=(--cases "$P/remote")
"$PY" -c 'import sys; sys.path.insert(0, "benchmarks/one_stage"); import composite as C
sys.exit(0 if "verity/pouw/period/v1" in C.COMPONENTS["remote"] else 1)' && REMOTE_CASES+=(--cases "$P/period")
log "$PY" benchmarks/one_stage/v2_case.py --window "$SRC/sampled-proofs/served/window" --out "$P/v2" --scratch "$W/rebuild" \
  --exe "$V2"
for d in "$SRC"/pieces/*/; do
  name=$(basename "$d")
  [ "$name" = pouw ] && continue
  log "$PY" - "$d" "$P/onsite/$name" <<'EOF'
import json, sys
from pathlib import Path
sys.path.insert(0, "benchmarks/one_stage")
from composite import write_private
src, out = Path(sys.argv[1]), Path(sys.argv[2])
case = next((f for f in (src / "case.json", src / f"{src.name}.case.json") if f.is_file()), None)
if case is None or not (src / "answer.json").is_file():
    print(f"KEPT {src.name}: no case beside an answer, not entered")
    sys.exit(0)
c, a = json.loads(case.read_text()), json.loads((src / "answer.json").read_text())
write_private(out, {"case.json": json.dumps({**c, "blobs": c.get("blobs") or []}, sort_keys=True).encode(),
                    "answer.json": (json.dumps(a, sort_keys=True, separators=(",", ":")) + "\n").encode()})
print(f"KEPT {src.name}: {a['protocol']} {a['verdict']}, its kept case and answer")
EOF
done
log "$PY" benchmarks/one_stage/composite.py remote "${REMOTE_CASES[@]}" --cases "$P/v2" --private-out "$W/remote" \
  --public-out "$OUT/remote" --exe "$REMOTE"
say "REMOTE exit $? (0 accept, 1 reject, 2 undecided)"
log "$PY" benchmarks/one_stage/composite.py onsite --cases "$P/onsite" --cases "$P/period" --cases "$P/v2" \
  --private-out "$W/onsite" --public-out "$OUT/onsite" --exe "$ONSITE"
say "ONSITE exit $? (0 accept, 1 reject, 2 undecided)"
