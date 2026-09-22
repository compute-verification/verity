#!/usr/bin/env bash
# Lane b-bind (2026-09-22): the merged tree (soundness-fixes x b-lookup x latency) and the LogUp statement binding.
#   research run --on vy-cpu3 --project verity --source . --stage <name> -- bash backends/direct/run_bind_gate.sh [check|test|gate|battery|timing|all]
# check   : cargo check --release --all-targets (direct + redteam battery crate)
# test    : cargo test --release --tests (unit negatives, chain, statement suite, lookup, lookup statement suite)
# gate    : vu-neg (52) both fields, limb and --lookup; vu-bench 64 VUs bound (limb + lookup), --bind-operands
# battery : fixtures/redteam/vu-battery/v1/run_battery.sh exit 0 (limb) and the battery adapter in --lookup mode
# timing  : 4096-VU BabyBear^4 and ^5 LogUp at 2^-128 (166 queries), bound vs --unbound, back to back, twice
# Everything lands in $RESEARCH_RUN_DIR.  Instances: /workspace/bench-instances/v1 (sha256 checked against the manifest).
set -euo pipefail
WHAT=${1:-all}
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:/workspace/uvbin:/usr/local/bin:$PATH"
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-/workspace/cargo-target-b-bind}
export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-12}
export VERITY_RUSTC="$(rustc --version)"
INST=${INSTANCES_ROOT:-/workspace/bench-instances/v1}
RUN_DIR=${RESEARCH_RUN_DIR:-/tmp/bind-gate}
mkdir -p "$RUN_DIR"
cd "$ROOT"
echo "== commit=${VERITY_COMMIT:-snapshot} rustc=$VERITY_RUSTC threads=$RAYON_NUM_THREADS what=$WHAT target=$CARGO_TARGET_DIR"

# seed the target dir from the merge-check build once (deps only differ by the lane's own crates)
if [ ! -d "$CARGO_TARGET_DIR" ] && [ -d /workspace/cargo-target-merge-check ]; then
    cp -r /workspace/cargo-target-merge-check "$CARGO_TARGET_DIR" || true
fi

if [ "$WHAT" = check ] || [ "$WHAT" = all ]; then
    echo "== cargo check --release --all-targets"
    (cd backends/direct && cargo check --release --all-targets -j"$RAYON_NUM_THREADS" 2>&1 | grep -E "^(warning: unused|error|warning: `|\s+-->|\s+= )|Finished|could not compile" | tee "$RUN_DIR/cargo_check.txt" | tail -40)
    (cd backends/redteam/vu-battery-b && cargo check --release -j"$RAYON_NUM_THREADS" 2>&1 | grep -E "^error|Finished|could not compile|\s+-->" | tee "$RUN_DIR/cargo_check_redteam.txt" | tail -20)
fi

if [ "$WHAT" = test ] || [ "$WHAT" = all ]; then
    echo "== cargo test --release --tests"
    (cd backends/direct && cargo test --release --tests -j"$RAYON_NUM_THREADS" 2>&1 | grep -E "^test |test result|FAILED|panicked|error" | tee "$RUN_DIR/cargo_test.txt" | grep -E "test result|FAILED|panicked|error")
fi

if [ "$WHAT" = gate ] || [ "$WHAT" = battery ] || [ "$WHAT" = timing ] || [ "$WHAT" = all ]; then
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
        bad += 0 if ok else 1
print("instances ok" if not bad else f"instances BAD ({bad})")
sys.exit(1 if bad else 0)
EOF
    BIN="$CARGO_TARGET_DIR/release/direct-unit"
    (cd backends/direct && cargo build --release -j"$RAYON_NUM_THREADS" 2>&1 | tail -1)
fi

if [ "$WHAT" = gate ] || [ "$WHAT" = all ]; then
    for FIELD in goldilocks babybear; do
        for MODE in "" "--lookup"; do
            TAG="${FIELD}${MODE:+_lookup}"
            echo "== vu-neg $FIELD $MODE (52 vu-k1536-neg)"
            mkdir -p "$RUN_DIR/neg_$TAG"
            "$BIN" vu-neg --root "$INST" --field "$FIELD" $MODE --out "$RUN_DIR/neg_$TAG" | tail -2
            echo "== vu-bench $FIELD $MODE 64 VUs (bound; wrong-word / wrong-B / wrong-table statements must be rejected inside)"
            mkdir -p "$RUN_DIR/vu64_$TAG"
            "$BIN" vu-bench --root "$INST" --field "$FIELD" $MODE --vus 64 --queries 100 --out "$RUN_DIR/vu64_$TAG" | tail -1
        done
    done
    echo "== vu-bench babybear --lookup --ext 5 64 VUs"
    mkdir -p "$RUN_DIR/vu64_babybear_lookup_ext5"
    "$BIN" vu-bench --root "$INST" --field babybear --lookup --ext 5 --vus 64 --queries 100 --out "$RUN_DIR/vu64_babybear_lookup_ext5" | tail -1
    echo "== vu-bench babybear 64 VUs with --bind-operands (limb + lookup)"
    mkdir -p "$RUN_DIR/vu64_babybear_operands" "$RUN_DIR/vu64_babybear_lookup_operands"
    "$BIN" vu-bench --root "$INST" --field babybear --vus 64 --queries 100 --bind-operands --out "$RUN_DIR/vu64_babybear_operands" | tail -1
    "$BIN" vu-bench --root "$INST" --field babybear --lookup --vus 64 --queries 100 --bind-operands --out "$RUN_DIR/vu64_babybear_lookup_operands" | tail -1
fi

if [ "$WHAT" = battery ] || [ "$WHAT" = all ]; then
    echo "== red-team battery (limb realisation; exit 0 required)"
    mkdir -p "$RUN_DIR/battery"
    INSTANCES_ROOT="$INST" bash fixtures/redteam/vu-battery/v1/run_battery.sh "$RUN_DIR/battery" 2>&1 | tail -5
    echo "== babybear rules battery, --lookup (AIR constraints, check-rows)"
    python3 backends/direct/tools/battery_adapter.py --bin "$BIN" --lookup --out "$RUN_DIR/battery_babybear_lookup.json" 2>&1 | tail -3
fi

if [ "$WHAT" = timing ] || [ "$WHAT" = all ]; then
    # bound / unbound alternated, twice, per extension degree: the pair on the same box is the fix's cost
    for EXT in "" "--ext 5"; do
        for REP in 1 2; do
            for B in bound unbound; do
                TAG="lookup4096${EXT:+_ext5}_${B}_$REP"
                FLAG=""; [ "$B" = unbound ] && FLAG="--unbound"
                echo "== timing $TAG"
                mkdir -p "$RUN_DIR/$TAG"
                "$BIN" vu-bench --root "$INST" --field babybear --lookup $EXT --vus 4096 --queries 166 $FLAG --out "$RUN_DIR/$TAG" | tail -2
            done
        done
    done
fi
echo "== done"
