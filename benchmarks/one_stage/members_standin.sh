#!/usr/bin/env bash
# Builds what the audit runs, then members_standin.py on stand-in windows of MEMBERS members (default: 300 100), each
# over TEMPLATES templates (default: members_standin.py's two).
# Run from the tree's root (`research run --cwd source`); what it keeps is under $OUT.
set -uo pipefail
export PATH="$HOME/.elan/bin:$HOME/.local/bin:$PATH"
OUT=${OUT:-$PWD/out}
SCR=$(mktemp -d)
mkdir -p "$OUT"
log() { echo "$(date -u +%H:%M:%SZ) $*" | tee -a "$OUT/steps.txt"; }
{ hostname; git rev-parse HEAD; git status --porcelain | head; nproc; free -g | head -2; } > "$OUT/machine.txt" 2>&1
log "uv sync"
uv sync --all-packages --extra torch-cpu > "$OUT/uv.log" 2>&1 || { log "uv sync failed"; exit 5; }
log "lake build flock-verify flock-firewall"
(cd verity/protocols/proofs/flock/verifier && mkdir -p .lake && flock .lake/verity-build.lock lake build flock-verify flock-firewall) \
  > "$OUT/lake-flock.log" 2>&1 || { log "lake build failed"; tail -20 "$OUT/lake-flock.log"; exit 6; }
log "check_build.sh bin"
bash verity/protocols/proofs/flock/prover/check_build.sh bin "$SCR/flock-circuit" > "$OUT/check-build.log" 2>&1 \
  || { log "check_build failed"; tail -20 "$OUT/check-build.log"; exit 7; }
log "v2 executable"
.venv/bin/python -c 'from verity.protocols.proofs.spot.verifier.one_stage import proofs as PF; print(PF.v2_executable())' \
  > "$OUT/v2-build.log" 2>&1 || { log "v2 build failed"; tail -20 "$OUT/v2-build.log"; exit 8; }
log "stand-in windows: ${MEMBERS:-300 100}${TEMPLATES:+ over $TEMPLATES templates}"
# shellcheck disable=SC2086
env -u PYTHONPATH .venv/bin/python benchmarks/one_stage/members_standin.py "$OUT/standin" "$SCR/flock-circuit" \
  "$PWD/verity/protocols/proofs/flock/verifier/.lake/build/bin" ${TEMPLATES:+--templates $TEMPLATES} ${MEMBERS:-300 100} 2>&1 \
  | tee "$OUT/standin.log" | cut -c1-3000
log "done"
rm -rf "$SCR"
