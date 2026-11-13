#!/usr/bin/env bash
# stream_case.py on a stand-in window of MEMBERS members (default 300; members_files.audited's), or on MF_CASE: builds
# Flock's verifier and prover, v2's executable, and a probe executable in this clone whose main is the stream's
# protocol's (`Spot.OneStageZk.Streamed.protocol.main`), which the verifier's mains do not call. The case is scratch; $OUT
# keeps each command's output. Run from a clone:
#   research run --on auto --cwd clone --mem-gb 32 -- bash benchmarks/one_stage/stream_case.sh
set -euo pipefail
O=${OUT:?run under research run, which sets OUT}
export PATH="$HOME/.elan/bin:$HOME/.local/bin:$PATH"
L=$PWD/verity/protocols/proofs/spot/verifier/lean
F=$PWD/verity/protocols/proofs/flock/verifier
S=$(mktemp -d)
keep() { mkdir -p "$O/kept"; cp "$S"/case/stream/*.out "$S"/case/stream/*.err "$S"/case/check/std* "$O/kept/" 2>/dev/null || true; rm -rf "$S"; }
trap keep EXIT
log() { echo "$(date -u +%H:%M:%SZ) $*" | tee -a "$O/steps.txt"; }
{ hostname; git rev-parse HEAD; nproc; free -g | head -2; } > "$O/machine.txt" 2>&1
log "uv sync"
uv sync -q --all-packages --extra torch-cpu > "$O/uv.log" 2>&1
cat > "$L/Spot/StreamProbeMain.lean" <<'EOT'
module

public import Spot.OneStageZkStream

@[expose] public section

def main (args : List String) : IO UInt32 := Spot.OneStageZk.Streamed.protocol.main args
EOT
printf '\n[[lean_exe]]\nname = "stream-probe"\nroot = "Spot.StreamProbeMain"\n' >> "$L/lakefile.toml"
log "lake build: the probe and v2's executable"
(cd "$L" && lake build stream-probe sampled-proofs-one-stage-zk) > "$O/spot-build.log" 2>&1 \
  || { log "spot build failed"; tail -60 "$O/spot-build.log"; exit 6; }
from=(--case "${MF_CASE:-}")
if [ -z "${MF_CASE:-}" ]; then
  log "lake build: the Lean verifier and the ZK server"
  (cd "$F" && lake build flock-verify flock-firewall) > "$O/verifier-build.log" 2>&1 \
    || { log "verifier build failed"; tail -60 "$O/verifier-build.log"; exit 7; }
  log "check_build.sh bin"
  bash verity/protocols/proofs/flock/prover/check_build.sh bin "$S/flock-circuit" > "$O/prover-build.log" 2>&1 \
    || { log "prover build failed"; tail -20 "$O/prover-build.log"; exit 8; }
  from=(--members "${MEMBERS:-300}" --bin "$S/flock-circuit" --lean-bin "$F/.lake/build/bin")
fi
log "stream_case.py"
env -u PYTHONPATH .venv/bin/python benchmarks/one_stage/stream_case.py "$S/case" "${from[@]}" \
  --stream-exe "$L/.lake/build/bin/stream-probe" | tee "$O/stream_case.jsonl"
log "done"
