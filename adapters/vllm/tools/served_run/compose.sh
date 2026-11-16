#!/usr/bin/env bash
# The compose run off the served node, which `HANDOFF=DIR run.sh compose` starts (placement.toml's [run.compose]): the
# window run's compose hand-off (its out/handoff/compose.tar, the serve dir's verifier-side files and no witness, held to
# its .sha256) laid out in a scratch serve dir, then `served_verdicts.sh compose` at the served node's commit: the copy
# held to the served node's sums, the cases' blobs rebased, and `record --off-node --sampled-lean`, which builds remote's
# SPOT Lean verifier here and runs it on the record's audit (the drawn units' proofs), its answer tied to the served
# record. The record and its lines go to $OUT (out/record, out/verdicts.txt). This node is no auditor device.
set -uo pipefail
IN=${RESEARCH_RUN_DIR:?run under research run}/inputs OUT=$RESEARCH_RUN_DIR/out
mkdir -p "$OUT"
log() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$OUT/compose-job.log"; }
[ "$(sha256sum "$IN/compose.tar" | cut -d' ' -f1)" = "$(cut -d' ' -f1 "$IN/compose.tar.sha256")" ] \
  || { log "compose.tar is not the one the window run packed (its sha256 differs)"; exit 2; }
S=$(mktemp -d /tmp/served-compose.XXXXXX) && trap 'rm -rf "$S"' EXIT
D=$S/serve
mkdir -p "$D" && tar -xf "$IN/compose.tar" -C "$D" --no-same-owner || { log "compose.tar does not unpack"; exit 2; }
log "compose on $(hostname) ($(nproc) cores) at $(git rev-parse HEAD): $(tar -tf "$IN/compose.tar" | wc -l) files, sha256 $(cut -c1-16 "$IN/compose.tar.sha256")…"
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$HOME/.elan/bin:$PATH
t0=$(date +%s)
D=$D OUT=$OUT bash benchmarks/one_stage/served_verdicts.sh compose > "$OUT/compose.log" 2>&1
rc=$?
log "served_verdicts.sh compose exit $rc in $(( ($(date +%s) - t0) / 60 )) min"
grep -E '^(PREFLIGHT|COMPOSE|RECORD|SAMPLED|LEAN|INFERENCE-ONLY)' "$OUT/compose.log" | cut -c1-900 | tee -a "$OUT/compose-job.log"
[ "$rc" = 0 ] || tail -n 25 "$OUT/compose.log" | cut -c1-400 > "$OUT/compose_tail.txt"
exit "$rc"
