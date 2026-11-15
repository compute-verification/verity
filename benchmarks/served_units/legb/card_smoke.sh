# card_smoke.sh: the shortest of proofs' launchers end to end on one tree: stage_prove.sh stages one small statement and
# proves it once (WARM=0 RUNS=1) on DEV (gpu: on CARD, its prover and verifier on CORES, its NUMA node's; cpu: the job's
# cores), then the tree's Flock Lean verifier, built in the clone, checks its kept session. Sent with the run, from a clone:
#   research run --on NODE --cwd clone --source TREE --send card_smoke.sh --send smoke.units --env UNITS=smoke.units \
#     --env DEV=gpu --env CARD=5 --env CORES=48-59 -- bash -c 'set -euo pipefail; bash "$RESEARCH_RUN_DIR/inputs/card_smoke.sh"'
# out/: tree.txt, the two stage_prove.sh jobs' steps (stage/, prove/), lean/ (the build, lean-verify.json and stdout for
# each Program handed: program-word.json, the word Definition's, and program.json where the staging wrote it), verdict.txt.
# The first failing step is the result: its line is the last of out/verdict.txt, and nothing is retried.
set -euo pipefail
IN=$RESEARCH_RUN_DIR/inputs; OUT=$RESEARCH_RUN_DIR/out; mkdir -p "$OUT"
export PATH=$HOME/.elan/bin:$HOME/.local/bin:$HOME/.cargo/bin:$PATH
V=$OUT/verdict.txt; say() { echo "$(date -u +%H:%M:%SZ) $*" | tee -a "$V"; }
DEV=${DEV:?gpu or cpu}; UNITS=${UNITS:?the units file sent with the run}; [ -f "$UNITS" ] || UNITS=$IN/$UNITS
git rev-parse HEAD > "$OUT/tree.txt"; say "tree $(cat "$OUT/tree.txt"), host $(hostname), device $DEV${CARD:+, card $CARD, cores $CORES}"
uv sync -q --all-packages --extra torch-cpu > "$OUT/uv.log" 2>&1 || { say "FAILED: uv sync: $(tail -1 "$OUT/uv.log")"; exit 3; }
FW=/workspace/tmp/proofs-smoke-${RESEARCH_RUN_ID:-$$}; S=$(mktemp -d); trap 'rm -rf "$FW" "$S"' EXIT
pin=(); [ "$DEV" = gpu ] && pin=(taskset -c "${CORES:?the card's NUMA node's cores}")
common=(DEV="$DEV" UNITS="$UNITS" FLOCK_WORK="$FW" KILL_AT=$(( $(date +%s) + 5400 )) MIN_LEFT_S=600 WARM=0 RUNS=1)
[ "$DEV" = gpu ] && common+=(CARDS="${CARD:?the card}" SM="${SM:-90}")

RESEARCH_RUN_DIR=$OUT/stage env "${common[@]}" STEPS=stage "${pin[@]}" bash benchmarks/served_units/stage_prove.sh \
  || { say "FAILED: stage_prove.sh STEPS=stage exit $?: $(tail -1 "$OUT/stage/out/steps.txt" 2>/dev/null)"; exit 4; }
d=$(ls -d "$FW"/data/*/ | head -1); name=$(basename "$d"); name=${name%-*}
cp -a "$d" "$S/st"; say "staged $name: $(stat -c %s "$S/st/circuit.txt") bytes of circuit"
RESEARCH_RUN_DIR=$OUT/prove env "${common[@]}" STEPS=prove "${pin[@]}" bash benchmarks/served_units/stage_prove.sh \
  || { say "FAILED: stage_prove.sh STEPS=prove exit $?: $(tail -1 "$OUT/prove/out/steps.txt" 2>/dev/null)"; exit 5; }
k=$OUT/prove/out/kept/$name; sd=$(ls -d "$k"/sessions/*/ 2>/dev/null | sort | tail -1)
[ -n "$sd" ] && [ "$(cat "$k/exit" 2>/dev/null)" = 0 ] || { say "FAILED: no kept session with prover exit 0 under $k"; exit 6; }
[ "$DEV" = cpu ] || [ -s "$OUT/prove/out/gpu0/gpu-lease-usage.jsonl" ] || { say "FAILED: no lease record: the proof did not run on a card"; exit 6; }
say "proved $name on $DEV: $(basename "$sd") kept"

L=verity/protocols/proofs/flock/verifier; mkdir -p "$OUT/lean"
exe=$(uv run --no-sync python -c 'import sys, tomllib
print({e.get("root"): e["name"] for e in tomllib.load(open(sys.argv[1], "rb"))["lean_exe"]}["FlockVerify"])' "$L/lakefile.toml")
(cd "$L" && lake build "$exe") > "$OUT/lean/build.log" 2>&1 || { say "FAILED: lake build $exe: $(grep -m1 -i error "$OUT/lean/build.log")"; exit 7; }
EXE=$PWD/$L/.lake/build/bin/$exe; say "the Lean verifier $exe built, sha256 $(sha256sum "$EXE" | cut -c1-16)"
uv run --no-sync python - "$S/st" > "$OUT/lean/program-word.txt" 2>&1 <<'EOF' || { say "FAILED: program-word.json: $(tail -1 "$OUT/lean/program-word.txt")"; exit 7; }
import json, sys
from pathlib import Path
from verity.circuits.codec import descriptor_chunks
from verity.circuits.program import Program
from verity_vllm.pipeline.serving_rows import template
d = Path(sys.argv[1]); word = json.loads((d / "stage.json").read_text())["word"]
with open(d / "program-word.json", "wb") as f:
    for c in descriptor_chunks(Program(template(word).fn)):
        f.write(c)
print(word)
EOF
st=$(python3 -c 'import json, sys; f = open(sys.argv[1], "rb"); f.readline(); print(json.loads(f.readline()[5:])["statement"])' "$S/st/circuit.txt")
ok=0
for p in program-word.json program.json; do
  [ -e "$S/st/$p" ] || continue
  t=$S/t-$p; o=$OUT/lean/$p; mkdir -p "$t" "$o"; cp -r "$sd" "$t/session"
  for f in circuit.txt stage.json "$p"; do ln -s "$S/st/$f" "$t/$f"; done
  for f in pub.bin partition.json; do [ -e "$k/$f" ] && ln -s "$k/$f" "$t/$f"; done
  [ -d "$k/tables" ] && cp -r "$k/tables" "$t/tables"
  uv run --no-sync verity bench prove lean-verify --lean "$EXE" --proof "$t" --statement "$st" --repeat 1 --out "$o" > "$o/stdout.txt" 2>&1 \
    || { say "lean-verify with $p: exit $?: $(grep -m1 -E 'Error|error|refus' "$o/stdout.txt" | cut -c1-300)"; continue; }
  a=$(python3 -c 'import json, sys; j = json.load(open(sys.argv[1])); print(j["accepted"], j.get("label", ""))' "$o/lean-verify.json")
  say "lean-verify with $p: accepted=$a"; [ "${a%% *}" = True ] && ok=1
done
[ "$ok" = 1 ] || { say "FAILED: the Lean verifier accepted the proof under no Program handed"; exit 8; }
say "PASSED: staged, proved on $DEV, and accepted by the tree's Lean verifier"
