#!/usr/bin/env bash
# The drawn-unit replay bundle's fixture proved (prove_bundle.py; PRF-171's M1): the torch loop's window that
# benchmarks/e2e_demo's fixture was built from, written again under the fixture's test key and held to the checked-in
# bundle byte for byte. Then each drawn member is proved by C-Flock `prove --zk` (CPU) behind the Lean firewall and decided
# by Lean `verify --zk`. The negatives are the first session's one-bit proof edits, refused by Lean, and the tampered loop's
# member, stopped by the firewall and refused by Lean without it.
#   research run --on vy-nebius-cpu-3 --pool proofs --as proofs --project verity --source . --cwd clone \
#     -- bash benchmarks/one_stage/prove_bundle.sh [PB_BUNDLE=benchmarks/e2e_demo/tests/fixtures/drawn-unit-bundle.json]
# Kept in $OUT: the summary, the logs and the session records; never a window's private file or the bundle's payload.
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uo pipefail
REPO=$(pwd); O=${OUT:-${RESEARCH_RUN_DIR:?}/out}; mkdir -p "$O"
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
log() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$O/job.log"; }
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$HOME/.elan/bin:$PATH PYTHONDONTWRITEBYTECODE=1
{ lscpu | grep -E 'Model name|^CPU\(s\)'; nproc; free -g; git rev-parse HEAD; } > "$O/host.txt" 2>&1
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
command -v lake >/dev/null || curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none >/dev/null
uv sync -q --all-packages --extra torch-cpu > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 5; }
PY=$REPO/.venv/bin/python
BUNDLE=${PB_BUNDLE:-benchmarks/e2e_demo/tests/fixtures/drawn-unit-bundle.json}

log "prove-bundle: source $(git rev-parse HEAD), bundle $BUNDLE, builds in $W"
(cd verity/core/service/flock && lake build flock-verify flock-firewall) > "$O/build-lean.log" 2>&1 &
LP=$!
mkdir -p "$O/m0-build"
RESEARCH_RUN_DIR=$O/m0-build FLOCK_WORK=$W MODE=build GPU=0 bash verity/ml/flock/pod/60-circuit.sh > "$O/m0-build/build.log" 2>&1 \
  || { tail -40 "$O/m0-build/build.log"; exit 6; }
BIN=$W/flock-gpu-link/flock/target/release/flock-circuit
wait $LP || { tail -40 "$O/build-lean.log"; exit 7; }
LEAN=$REPO/verity/core/service/flock/.lake/build/bin/flock-verify
FWL=${LEAN%/*}/flock-firewall
sha256sum "$BIN" "$LEAN" "$FWL" | tee "$O/binaries.sha256" | tee -a "$O/job.log"
export PYTHONPATH=$REPO:$REPO/verity/ml/flock/python:$REPO/integrations/torch_loop:$REPO/benchmarks/e2e_demo:$REPO/benchmarks/numerical/python
export RAYON_NUM_THREADS=$(nproc)
# flock-circuit's rayon pool keeps Rust's default 2 MiB thread stack, which a --zk prover can overflow on a many-core node
# (prove_path/run.sh)
export RUST_MIN_STACK=$((64 << 20))

log "windows: the fixture's (honest) and the tampered loop's, under the fixture's test key"
"$PY" - "$W" "$BUNDLE" <<'EOF' 2>&1 | tee -a "$O/job.log"
import json, sys
from verity_e2e_demo import replay_bundle as RB
w, bundle = sys.argv[1], sys.argv[2]
_, weights = RB.torch_loop_window(f"{w}/window")
RB.torch_loop_window(f"{w}/tampered", tamper=True)
doc, pay = RB.build(f"{w}/window", RB.FIXTURE_DRAW, reads=RB.torch_loop_reads, weights=weights,
                    draw={"source": "fixture", "law": {"law": "subset", "k": len(RB.FIXTURE_DRAW)}})
ref, ref_pay = RB.read(bundle)
same = json.loads(json.dumps(doc)) == ref and pay == ref_pay
print("WINDOW " + json.dumps({"bundle_is_the_windows": same, "payload_sha512": ref["payload"]["sha512"]}))
sys.exit(0 if same else 3)
EOF
[ "${PIPESTATUS[0]}" = 0 ] || { log "the regenerated window is not the bundle's"; exit 8; }

log "prove: the bundle's drawn members, then the negatives"
"$PY" benchmarks/one_stage/prove_bundle.py --bundle "$BUNDLE" --window "$W/window" --tampered "$W/tampered" \
  --bin "$BIN" --lean "$LEAN" --firewall "$FWL" --out "$W/pb" ${PB_MEMBERS:+--members $PB_MEMBERS} \
  > "$O/prove_bundle.log" 2> "$O/prove_bundle.err"
rc=$?
grep '^PROVEBUNDLE' "$O/prove_bundle.log" | tee -a "$O/job.log"
[ $rc = 0 ] || tail -30 "$O/prove_bundle.err"
cp "$W/pb/summary.json" "$O/" 2>/dev/null
(cd "$W/pb" && { find honest tampered -type f \( -name 'session.json' -o -name '*.log' -o -name 'draw.json' \) -size -8M -print0
  find flipped -maxdepth 2 -name zk_mutants.json -print0; } | xargs -0 -r -I{} cp --parents {} "$O/")
log "prove-bundle: done, rc $rc"
exit $rc
