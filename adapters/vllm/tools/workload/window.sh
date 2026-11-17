#!/usr/bin/env bash
# The workload on its node (`run.sh window`): served_verdicts.sh under the card's lease, then its hand-off packed.
set -uo pipefail
for v in CARD CORES END COMPOSE_OFF_NODE COMPOSE_NODE ROW REPO REV VENV CUDA_HOME HF_HOME SERVE_ROOT FLOCK_WORK HIDDEN_SO NORM_TAP_SO; do
  [ -n "${!v:-}" ] || { echo "window.sh: no $v: start it with \`adapters/vllm/tools/workload/run.sh window\`"; exit 2; }
done
IN=${RESEARCH_RUN_DIR:?run under research run}/inputs OUT=$RESEARCH_RUN_DIR/out
mkdir -p "$OUT"
log() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$OUT/window-job.log"; }
TREE=$PWD SHA=$(git rev-parse HEAD)
row_id=${ROW##*/}
end=$(date -u -d "$END" +%s) || { log "END is not a time"; exit 2; }
for f in "$ROW/manifest.json" "$HIDDEN_SO" "$NORM_TAP_SO" "$FLOCK_WORK/flock-gpu-link/flock/target/release/flock-circuit"; do
  [ -e "$f" ] || { log "the node holds no $f (\`run.sh setup\` makes it): nothing run"; exit 2; }
done
cp "$IN/$row_id.json" adapters/vllm/workloads/ || exit 2

real=$(command -v gpu-lease) || { log "no gpu-lease on this node: nothing run"; exit 2; }
mkdir -p "$RESEARCH_RUN_DIR/bin"
printf '#!/usr/bin/env bash\n# the window run leases card %s alone: --on %s after the count\n[ "$1" = status ] && exec %q status\nn=$1; shift\nexec %q "$n" --on %s "$@"\n' \
  "$CARD" "$CARD" "$real" "$real" "$CARD" > "$RESEARCH_RUN_DIR/bin/gpu-lease" && chmod +x "$RESEARCH_RUN_DIR/bin/gpu-lease"
held=$(nvidia-smi -i "$CARD" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
used=$(nvidia-smi -i "$CARD" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -dc 0-9)
lease=$("$real" status 2>/dev/null | awk -v c="$CARD" '$1 == c {$1 = ""; print substr($0, 2)}')
if [ "$held" != 0 ] || [ "${used:-99999}" -ge 1024 ] || [ "${lease%% *}" != free ]; then
  log "card $CARD is in use ($held compute process(es), ${used:-?} MiB, lease: ${lease:-none}): nothing run"; exit 75
fi
lease_min=$(( (end - $(date -u +%s)) / 60 - 15 ))
[ "$lease_min" -ge "${LEASE_MIN:-60}" ] || { log "only $lease_min min of card before $END, under ${LEASE_MIN:-60}: nothing run"; exit 75; }
LEASE_MIN=${LEASE_MIN:-$(( lease_min < 150 ? lease_min : 150 ))}

[ ! -f "$IN/main.txt" ] || export MAIN_FILE=$IN/main.txt
log "coins: ${PP_CHALLENGER:-local} (peer ${PP_PEER:-none}${DEVELOPMENT:+, a development run})"
log "window run of $row_id from $SHA on $(hostname), card $CARD (free: 0 processes, ${used} MiB, lease: $lease), cores $CORES, lease at most $LEASE_MIN min, COMPOSE_NODE $COMPOSE_NODE${PEARLPLUS_TIE:+, tied to Pearl+}"
export PATH=$RESEARCH_RUN_DIR/bin:$PATH LEASE_MIN OUT
mkdir -p "$SERVE_ROOT" /dev/shm/vllm-workload && chmod 700 "$SERVE_ROOT" /dev/shm/vllm-workload
D=/dev/shm/vllm-workload/${RESEARCH_RUN_ID:?run under research run}; export D
echo "$D" > "$OUT/serve_dir.txt"
t0=$(date +%s)
bash adapters/vllm/tools/workload/run.sh settings "$REPO" -- bash benchmarks/one_stage/served_verdicts.sh > "$OUT/served_verdicts.log" 2>&1
rc=$?
log "served_verdicts.sh exit $rc after $(( ($(date +%s) - t0) / 60 )) min; its lines in out/verdicts.txt"
if [ "${PREFLIGHT_ONLY:-}" = 1 ] && [ "$rc" = 0 ]; then
  log "the preflight passed, nothing served: $(grep -h -m1 'PREFLIGHT passed' "$OUT/served_verdicts.log" | cut -c1-300)"; exit 0
fi
. adapters/vllm/tools/workload/handoff.sh
window_exit "$rc"
