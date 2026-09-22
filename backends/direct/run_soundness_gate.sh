#!/usr/bin/env bash
# Lane soundness-fixes (2026-09-22): the B gate after the statement-binding fix (PROTOCOL.md 12.5, red team F1-F3).
#   research run --on vy-cpu3 -- bash backends/direct/run_soundness_gate.sh [gate|battery|timing|all]
# gate    : cargo test --release (unit negatives, chain + statement tests), vu-neg (52) and vu-bench 64 VUs, both fields
# battery : fixtures/redteam/vu-battery/v1/run_battery.sh must exit 0 (the two former forgeries rejected)
# timing  : 4096-VU BabyBear at 2^-128 (166 queries, PoW 16), bound vs the pre-fix unbound path, same binary, same box
# Everything lands in $RESEARCH_RUN_DIR.  Instances: /workspace/bench-instances/v1 (positives copied from vy-cpu2;
# sha256 checked against the committed manifest before use).
set -euo pipefail
WHAT=${1:-all}
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:/workspace/uvbin:/usr/local/bin:$PATH"
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-/workspace/target}
export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-$(nproc)}
export VERITY_RUSTC="$(rustc --version)"
INST=${INSTANCES_ROOT:-/workspace/bench-instances/v1}
RUN_DIR=${RESEARCH_RUN_DIR:-/tmp/soundness-gate}
mkdir -p "$RUN_DIR"
cd "$ROOT"
echo "== commit=${VERITY_COMMIT:-snapshot} rustc=$VERITY_RUSTC threads=$RAYON_NUM_THREADS what=$WHAT"

# the positives are not in git: check their digests against the committed manifest
python3 - "$ROOT/fixtures/bench-instances/v1/manifest.json" "$INST" <<'EOF'
import hashlib, json, sys, pathlib
man = json.load(open(sys.argv[1])); root = pathlib.Path(sys.argv[2])
bad = 0
for tier in ("vu-k1536", "vu-k1536-neg"):
    for name, meta in man["tiers"][tier]["files"].items():
        p = root / name
        if not p.is_file():
            print(f"MISSING {p}"); bad += 1; continue
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        ok = h == meta["sha256"]
        print(f"{'ok ' if ok else 'BAD'} {name} {h[:16]}")
        bad += 0 if ok else 1
sys.exit(1 if bad else 0)
EOF

BIN="$CARGO_TARGET_DIR/release/direct-unit"
(cd backends/direct && cargo build --release -j"$RAYON_NUM_THREADS" 2>&1 | tail -1)

if [ "$WHAT" = gate ] || [ "$WHAT" = all ]; then
    # --tests: the doctest target does not build in release (p3_uni_stark::DebugConstraintBuilder is cfg(debug_assertions);
    # a bound inherited from main, not this lane) -- unit/integration tests are what the gate needs
    echo "== cargo test --release --tests (unit negatives + chain + statement binding)"
    (cd backends/direct && cargo test --release --tests -j"$RAYON_NUM_THREADS" 2>&1 | grep -E "^test |test result|FAILED|panicked" | tee "$RUN_DIR/cargo_test.txt" | grep -E "test result|FAILED|panicked")
    for FIELD in goldilocks babybear; do
        echo "== vu-neg $FIELD (52 vu-k1536-neg)"
        mkdir -p "$RUN_DIR/neg_$FIELD"
        "$BIN" vu-neg --root "$INST" --field "$FIELD" --out "$RUN_DIR/neg_$FIELD" | tail -2
        echo "== vu-bench $FIELD 64 VUs (bound; wrong-word / wrong-B statements must be rejected inside)"
        mkdir -p "$RUN_DIR/vu64_$FIELD"
        "$BIN" vu-bench --root "$INST" --field "$FIELD" --vus 64 --queries 100 --out "$RUN_DIR/vu64_$FIELD" | tail -1
    done
    echo "== vu-bench babybear 64 VUs with --bind-operands (33 public columns)"
    mkdir -p "$RUN_DIR/vu64_babybear_operands"
    "$BIN" vu-bench --root "$INST" --field babybear --vus 64 --queries 100 --bind-operands --out "$RUN_DIR/vu64_babybear_operands" | tail -1
fi

if [ "$WHAT" = battery ] || [ "$WHAT" = all ]; then
    echo "== red-team vu-battery (must exit 0)"
    mkdir -p "$RUN_DIR/battery"
    set +e
    RESEARCH_RUN_DIR="$RUN_DIR/battery" VERITY_GKR_THREADS="$RAYON_NUM_THREADS" bash fixtures/redteam/vu-battery/v1/run_battery.sh "$RUN_DIR/battery" 2>&1 | tee "$RUN_DIR/battery/run_battery.log" | tail -45
    rc=${PIPESTATUS[0]}
    set -e
    echo "run_battery.sh exit=$rc" | tee "$RUN_DIR/battery/exit.txt"
    [ "$rc" = 0 ] || exit "$rc"
fi

if [ "$WHAT" = timing ] || [ "$WHAT" = all ]; then
    Q=${QUERIES:-166}
    # ORDER swaps the two runs (first run on a cold box pays page-cache / allocator warm-up: r20260922-073904-7a6c
    # saw the identical witness phase 26% slower in the first run) -- always report both orders
    for MODE in ${ORDER:-bound unbound}; do
        FLAG=""; [ "$MODE" = unbound ] && FLAG="--unbound"
        echo "== vu-bench babybear 4096 VUs, $Q queries ($MODE)"
        mkdir -p "$RUN_DIR/vu4096_babybear_$MODE"
        "$BIN" vu-bench --root "$INST" --field babybear --vus 4096 --queries "$Q" $FLAG --out "$RUN_DIR/vu4096_babybear_$MODE" | tail -1
    done
fi
echo done
