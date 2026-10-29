#!/usr/bin/env bash
# The drawn-unit replay bundle proved (prove_bundle.py; M1): the bundle PB_BUNDLE (its .bin beside it) of the
# window PB_WINDOW; without PB_WINDOW, the store's art:d0b6e81f (BUNDLE_WINDOW: a W8A8 MLP's window under a test key, its
# tampered twin and the H200 interfaces' bundle of four drawn units under the window's registered subset:4, a fixture's draw).
# Each drawn member is proved by C-Flock `prove --zk` (CPU) behind the Lean firewall, one outer proof at a time, and
# decided by Lean `verify --zk`. The negatives are the first session's one-bit proof edits, refused by Lean, and the
# member of PB_TAMPERED (with the fixture, its tampered twin unless PB_TAMPERED= is given empty), stopped by the firewall
# and refused by Lean without it. PB_LIVE=1 requires the bundle's draw to be live, with its receipt.
#   research run --on vy-nebius-cpu-4 --pool proofs --as proofs --project verity --source . --cwd clone \
#     -- bash benchmarks/one_stage/prove_bundle.sh [PB_BUNDLE=F PB_WINDOW=DIR [PB_TAMPERED=DIR]] [PB_MEMBERS=N] [PB_FLIPS=1] [PB_BINARIES=DIR] [PB_BUILD_ONLY=1]
# PB_BUILD_ONLY=1 builds the three binaries, keeps them in $OUT/bin with the stamp of the trees they're built from
# (verity/ml/flock and verity/core/service/flock), and stops. PB_BINARIES=DIR (on the node, e.g. that run's out/bin) uses them
# when DIR's stamp is this source's, and otherwise builds here: M3 builds before its window, not in it.
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

STAMP=$(git rev-parse HEAD:verity/ml/flock HEAD:verity/core/service/flock 2>/dev/null | tr '\n' ' ')
[ -n "${STAMP// }" ] || { echo "prove-bundle: no git trees to stamp binaries with"; exit 4; }
BUNDLE_WINDOW=art:d0b6e81fa20a0e2c9c6701ca4495e8092434ad8c9e7c08038d114ad42210980c
FX=$W/bundle-window
if [ -n "${PB_WINDOW:-}" ]; then
  BUNDLE=${PB_BUNDLE:?PB_WINDOW needs its PB_BUNDLE} WINDOW=$PB_WINDOW TAMPERED=${PB_TAMPERED:-}
else
  BUNDLE=$FX/bundle.json WINDOW=$FX/window TAMPERED=${PB_TAMPERED-$FX/tampered}
fi
log "prove-bundle: source $(git rev-parse HEAD), bundle $BUNDLE, window $WINDOW, builds in $W"
if [ -n "${PB_BINARIES:-}" ] && [ "$(cat "$PB_BINARIES/stamp" 2>/dev/null)" = "$STAMP" ]; then
  BIN=$PB_BINARIES/flock-circuit LEAN=$PB_BINARIES/flock-verify FWL=$PB_BINARIES/flock-firewall
  log "binaries: prebuilt in $PB_BINARIES, stamp $STAMP"
else
  [ -n "${PB_BINARIES:-}" ] && log "binaries: $PB_BINARIES's stamp is not this source's ($STAMP), so they're built here"
  (cd verity/core/service/flock && lake build flock-verify flock-firewall) > "$O/build-lean.log" 2>&1 &
  LP=$!
  mkdir -p "$O/m0-build"
  RESEARCH_RUN_DIR=$O/m0-build FLOCK_WORK=$W MODE=build GPU=0 bash verity/ml/flock/pod/60-circuit.sh > "$O/m0-build/build.log" 2>&1 \
    || { tail -40 "$O/m0-build/build.log"; exit 6; }
  BIN=$W/flock-gpu-link/flock/target/release/flock-circuit
  wait $LP || { tail -40 "$O/build-lean.log"; exit 7; }
  LEAN=$REPO/verity/core/service/flock/.lake/build/bin/flock-verify
  FWL=${LEAN%/*}/flock-firewall
fi
sha256sum "$BIN" "$LEAN" "$FWL" | tee "$O/binaries.sha256" | tee -a "$O/job.log"
if [ "${PB_BUILD_ONLY:-0}" = 1 ]; then
  mkdir -p "$O/bin" && cp "$BIN" "$LEAN" "$FWL" "$O/bin/" && echo "$STAMP" > "$O/bin/stamp" || exit 9
  log "build only: $O/bin, stamp $STAMP"
  exit 0
fi
export PYTHONPATH=$REPO:$REPO/verity/ml/flock/python:$REPO/demos/h200/interfaces:$REPO/benchmarks/numerical/python
export RAYON_NUM_THREADS=$(nproc)
# flock-circuit's rayon pool keeps Rust's default 2 MiB thread stack, which a --zk prover can overflow on a many-core node
# (prove_path/run.sh)
export RUST_MIN_STACK=$((64 << 20))

if [ -z "${PB_WINDOW:-}" ]; then   # the store CLI with the run's custody key in its environment only, as prove_path/run.sh's R
  PYTHONPATH=$REPO/benchmarks/proof_throughput "$PY" -c 'import os, sys
from verity_proof_throughput.store import store_env
os.execvpe(sys.executable, [sys.executable, "-m", "research", *sys.argv[1:]], store_env())' data fetch $BUNDLE_WINDOW --to "$FX" > "$O/fetch.log" 2>&1 \
    || { tail -20 "$O/fetch.log"; exit 8; }
fi

log "prove: the bundle's drawn members, then the negatives"
"$PY" benchmarks/one_stage/prove_bundle.py --bundle "$BUNDLE" --window "$WINDOW" ${TAMPERED:+--tampered "$TAMPERED"} \
  --bin "$BIN" --lean "$LEAN" --firewall "$FWL" --out "$W/pb" ${PB_MEMBERS:+--members $PB_MEMBERS} ${PB_LIVE:+--live} ${PB_FLIPS:+--flips $PB_FLIPS} \
  > "$O/prove_bundle.log" 2> "$O/prove_bundle.err"
rc=$?
grep '^PROVEBUNDLE' "$O/prove_bundle.log" | tee -a "$O/job.log"
[ $rc = 0 ] || tail -30 "$O/prove_bundle.err"
cp "$W/pb/summary.json" "$O/" 2>/dev/null
(cd "$W/pb" && { find honest tampered -type f \( -name 'session.json' -o -name '*.log' -o -name 'draw.json' \) -size -8M -print0
  find flipped -maxdepth 2 -name zk_mutants.json -print0; } | xargs -0 -r -I{} cp --parents {} "$O/")
log "prove-bundle: done, rc $rc"
exit $rc
