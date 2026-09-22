#!/usr/bin/env bash
# Red-team VU battery (lane red-team-vu) for `research run --on vy-cpu3 --stage redteam.vu`:
#   A: forged run-vu directories (redteam.vu_forge_a) through the real verity-gkr prover + verifier (PCS and clear);
#   B: adversarial ChainAir traces through the real Plonky3 prover + verifier over both fields (redteam-vu-b battery);
#   challenger audit (10^5 samples of both challengers, the upstream bug reproduced in release).
# Writes everything into $RESEARCH_RUN_DIR (a_results.jsonl, a_*.log/json, vu_battery_b.json, challenger_audit.json).
# Env: VUS (A batch size, default 4), VERITY_GKR_THREADS (default 12), QUERIES (FRI queries for B, default = lane's 100).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
export PATH=$HOME/.cargo/bin:$HOME/.local/bin:$PATH
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-/workspace/target}
export VERITY_GKR_THREADS=${VERITY_GKR_THREADS:-12}
RUN_DIR=${RESEARCH_RUN_DIR:-/tmp/redteam-vu}
VUS=${VUS:-4}
mkdir -p "$RUN_DIR"
cd "$ROOT"
echo "commit=$(git rev-parse --short HEAD 2>/dev/null || echo snapshot) rustc=$(rustc --version) vus=$VUS threads=$VERITY_GKR_THREADS"

# ---- Candidate A -------------------------------------------------------------------------------------------------
(cd backends/gkr && cargo build --release -j12 2>&1 | tail -1)
uv run --package verity-numerical python -m verity_numerical.redteam.vu_forge_a --out "$RUN_DIR/forge_a" --vus "$VUS" --root fixtures/bench-instances/v1
: > "$RUN_DIR/a_results.jsonl"
for d in "$RUN_DIR"/forge_a/*/; do
    id=$(basename "$d")
    for mode in pcs clear; do
        flag=""; [ "$mode" = clear ] && flag="--clear"
        set +e
        "$CARGO_TARGET_DIR/release/verity-gkr" run-vu --dir "$d" $flag --out "$RUN_DIR/a_${id}_${mode}.json" 2> "$RUN_DIR/a_${id}_${mode}.log"
        rc=$?
        set -e
        verdict=$(grep -o 'verify=[^ ]*\(([^)]*)\)\?' "$RUN_DIR/a_${id}_${mode}.log" | head -1 | sed 's/^verify=//')
        printf '{"id": "%s", "mode": "%s", "exit": %d, "verify": %s}\n' "$id" "$mode" "$rc" "$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$verdict")" >> "$RUN_DIR/a_results.jsonl"
        echo "A $id $mode exit=$rc $verdict"
    done
done
rm -rf "$RUN_DIR"/forge_a/*/units.bin   # the witnesses are reproducible from the manifest; keep the run small

# ---- Candidate B + challenger ------------------------------------------------------------------------------------
(cd backends/redteam/vu-battery-b && cargo build --release -j12 2>&1 | tail -1)
"$CARGO_TARGET_DIR/release/redteam-vu-b" challenger --out "$RUN_DIR" --samples 100000 > "$RUN_DIR/challenger_audit.txt"
QARG=""; [ -n "${QUERIES:-}" ] && QARG="--queries $QUERIES"
"$CARGO_TARGET_DIR/release/redteam-vu-b" battery --root fixtures/bench-instances/v1 --out "$RUN_DIR" $QARG 2>&1 | tee "$RUN_DIR/vu_battery_b.log"
echo done
