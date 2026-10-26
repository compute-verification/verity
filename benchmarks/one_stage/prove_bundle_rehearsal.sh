#!/usr/bin/env bash
# M3's rehearsal (PRF-172) on the torch loop's fixture, in M3's slot with M3's binaries, before the delivered tree arrives:
# 0. the binaries: PB_BINARIES when it holds all three of this source's (a finished run's out/bin is evicted once custody
#    preserves it, and a run on the node has no store remote to restore it from), else built here (PB_BUILD_ONLY);
# 1. `research data put` then `fetch` round-trips in the node's own store (M3 reads its tree with PB_IN_ART, which a run
#    can fetch only from the node's local store: runs have no remote);
# 2. a registered step (prove_bundle.py --step): the fixture's step registered, K blocks drawn live over every
#    (slot, layout unit) by Lean's coin, a one-unit bundle a drawn block, delivered as a tree (PB_IN) and proved;
# 3. that tree with one block's step path tampered and another block's bundle under a third's index, the other blocks
#    removed: prove_bundle.py --step must refuse both before any proof (rc 1, nothing proved);
# 4. the single-bundle delivery (window/, bundle.{json,bin}, tampered/), K units drawn live over the window.
#   research run --on vy-nebius-cpu-4 --as proofs --project verity --source . --cwd clone -- \
#     env CHECK_SLOTS=/workspace/research/locks/m3/slots python3 tools/verity/check/slot.py --by proofs -- \
#     bash benchmarks/one_stage/prove_bundle_rehearsal.sh [PB_BINARIES=DIR] [PB_K=8]
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uo pipefail
REPO=$(pwd); O=${RESEARCH_RUN_DIR:?}/out; mkdir -p "$O"; K=${PB_K:-8}
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
log() { echo "[$(date -u +%H:%M:%SZ)] REHEARSAL $*" | tee -a "$O/rehearsal.log"; }
uv sync -q --all-packages --extra torch-cpu > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 5; }
PY=$REPO/.venv/bin/python
export PYTHONPATH=$REPO:$REPO/verity/ml/flock/python:$REPO/integrations/torch_loop:$REPO/benchmarks/e2e_demo:$REPO/benchmarks/numerical/python
STAMP=$(git rev-parse HEAD:verity/ml/flock HEAD:verity/core/service/flock | tr '\n' ' ')
B=${PB_BINARIES:-}
if [ -n "$B" ] && [ "$(cat "$B/stamp" 2>/dev/null)" = "$STAMP" ] && [ -x "$B/flock-circuit" ] && [ -x "$B/flock-verify" ] \
    && [ -x "$B/flock-firewall" ]; then
  log "binaries: $B (stamp $STAMP)"
else
  log "binaries: ${B:-none given} lacks this source's three (stamp $STAMP); building"
  OUT=$O/build bash benchmarks/one_stage/prove_bundle.sh PB_BUILD_ONLY=1 > "$O/build.log" 2>&1 || { tail -20 "$O/build.log"; exit 6; }
  B=$O/build/bin
  log "binaries: built in $B ($(grep -h 'build only' "$O/build/job.log" | tail -1))"
fi
run_pb() { OUT=$O/$1 bash benchmarks/one_stage/prove_bundle.sh "${@:2}" PB_BINARIES="$B" PB_FLIPS=1 PB_LEAN_JOBS=2; }

log "step: make (k $K, live over the step registration)"
"$PY" benchmarks/one_stage/prove_bundle.py make --step --k "$K" --lean "$B/flock-verify" --scratch "$T/s" --out "$T/step" \
  > "$O/step-make.log" 2>&1 || { tail -20 "$O/step-make.log"; exit 8; }
grep '^MAKE' "$O/step-make.log" | tee -a "$O/rehearsal.log"
art=$(research data put --kind m3-rehearsal-fetch-check --file "$T/step/step-registration.json" 2> "$O/fetch-check.log" | tail -1)
if [ -n "$art" ] && research data fetch "$art" --to "$T/f" >> "$O/fetch-check.log" 2>&1 \
    && cmp -s "$T/step/step-registration.json" "$(find "$T/f" -type f | head -1)"; then
  log "fetch ok: $art (put then fetched in the node's store, the same bytes)"
else
  log "fetch FAILED: $(tail -1 "$O/fetch-check.log")"
fi
t0=$(date +%s)
run_pb step PB_IN="$T/step"; rc_step=$?
log "step: the drawn blocks proved and verified in $(( $(date +%s) - t0 )) s wall (sessions one at a time, 2 Lean verifies beside)"
log "step: prove_bundle.sh rc $rc_step"

log "step negatives: a tampered step path, a block under another index"
cp -rL "$T/step" "$T/neg"
"$PY" - "$T/neg" > "$O/step-negatives-make.log" 2>&1 <<'EOF'
import json, shutil, sys
from pathlib import Path
d = Path(sys.argv[1]) / "blocks"
gs = sorted(int(p.name) for p in d.iterdir())
assert len(gs) >= 3, f"{gs}: fewer than 3 drawn blocks to tamper"
a, b, c = gs[:3]
doc = json.loads((d / str(a) / "bundle.json").read_text())
next(r for r in doc["units"][0]["rows"] if r["slot"] is not None)["slot"]["step_path"]["proof"][0] = "00" * 64
(d / str(a) / "bundle.json").write_text(json.dumps(doc))
for f in ("bundle.json", "bundle.bin"):
    shutil.copy(d / str(c) / f, d / str(b) / f)
for g in gs[3:] + [c]:
    shutil.rmtree(d / str(g))
print(f"REHEARSAL negatives: block {a} step path tampered, block {b} holds block {c}'s bundle, the rest removed")
EOF
mk=$?; tee -a "$O/rehearsal.log" < "$O/step-negatives-make.log"
rc_neg=x proved=x
if [ "$mk" = 0 ]; then
  "$PY" benchmarks/one_stage/prove_bundle.py --step "$T/neg" --bin "$B/flock-circuit" --lean "$B/flock-verify" \
    --firewall "$B/flock-firewall" --out "$T/neg-pb" > "$O/step-negatives.log" 2>&1; rc_neg=$?
  grep '^PROVEBUNDLE' "$O/step-negatives.log" | tee -a "$O/rehearsal.log"
  cp "$T/neg-pb/summary.json" "$O/step-negatives-summary.json" 2>/dev/null
  proved=$(find "$T/neg-pb" -name session.json 2>/dev/null | wc -l)
fi
if [ "$rc_neg" = 1 ] && [ "$proved" = 0 ] && grep -q '"refused before proving' "$O/step-negatives.log"; then
  log "step negatives: refused before proving (rc 1, 0 sessions)"; neg_ok=1
else
  log "step negatives: NOT refused as required (rc $rc_neg, $proved sessions)"; neg_ok=0
fi

log "single bundle: make (k $K, live over the window), delivered as window/ bundle.{json,bin} tampered/"
"$PY" benchmarks/one_stage/prove_bundle.py make --scratch "$T/w" --out "$T/w/b" --k "$K" --lean "$B/flock-verify" \
  > "$O/bundle-make.log" 2>&1 || { tail -20 "$O/bundle-make.log"; exit 8; }
mkdir -p "$T/in" && mv "$T/w/window" "$T/w/tampered" "$T/in/" && mv "$T/w/b/bundle.json" "$T/w/b/bundle.bin" "$T/in/"
"$PY" - "$T/in" <<'EOF' | tee -a "$O/rehearsal.log"
import json, sys
from pathlib import Path
d = Path(sys.argv[1]); keep = {m["k"] for m in json.load(open(d / "bundle.json"))["members"]}
gone = [p for p in (d / "window").glob("m*-private.bin") if int(p.name[1:].split("-")[0]) not in keep]
for p in gone:
    p.unlink()
print("REHEARSAL drawn members", sorted(keep), "private files removed", len(gone))
EOF
run_pb bundle PB_IN="$T/in"; rc_bundle=$?
log "single bundle: prove_bundle.sh rc $rc_bundle"
log "done: step rc $rc_step, step negatives ok $neg_ok, single bundle rc $rc_bundle"
[ "$rc_step" = 0 ] && [ "$neg_ok" = 1 ] && [ "$rc_bundle" = 0 ]
