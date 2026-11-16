#!/usr/bin/env bash
# The served window on its node (`run.sh window`): served_verdicts.sh under the card's lease, then its hand-off packed.
set -uo pipefail
for v in CARD CORES END COMPOSE_NODE ROW REPO REV VENV CUDA_HOME HF_HOME SERVE_ROOT FLOCK_WORK HIDDEN_SO NORM_TAP_SO; do
  [ -n "${!v:-}" ] || { echo "window.sh: no $v: start it with \`adapters/vllm/tools/served_run/run.sh window\`"; exit 2; }
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
. adapters/vllm/tools/served_run/pin_link.sh || exit 2

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
[ "$lease_min" -ge 60 ] || { log "only $lease_min min of card before $END: nothing run"; exit 75; }
LEASE_MIN=$(( lease_min < 150 ? lease_min : 150 ))

case ${PP_CHALLENGER:-local} in
  gate)
    [ -n "${CHALLENGER_ENV:-}" ] && [ "$(stat -c %a "$CHALLENGER_ENV" 2>/dev/null)" = 600 ] \
      || { log "the gate's coins read the gate's address from CHALLENGER_ENV (${CHALLENGER_ENV:-unset in placement.toml}), a mode-600 env file on this node: nothing run"; exit 2; }
    set -a; . "$CHALLENGER_ENV"; set +a
    [ -n "${VY_CHALLENGER_1_HOST:-}" ] || { log "$CHALLENGER_ENV sets no VY_CHALLENGER_1_HOST: nothing run"; exit 2; }
    PP_CHALLENGER=gate:research@$VY_CHALLENGER_1_HOST
    PP_CHALLENGER_KEY=$TREE/$PP_CHALLENGER_KEY PP_GATE_KEY=${PP_GATE_KEY/#\~/$HOME} MAIN_FILE=$IN/main.txt
    for f in "$PP_CHALLENGER_KEY" "$PP_GATE_KEY" "$MAIN_FILE"; do [ -r "$f" ] || { log "the gate's coins need $f: nothing run"; exit 2; }; done
    export PP_CHALLENGER PP_CHALLENGER_KEY PP_GATE_KEY MAIN_FILE
    log "coins: vy-challenger-1's gate (peer ${PP_PEER:-}), key record $(sha256sum "$PP_CHALLENGER_KEY" | cut -c1-16)…, main's line of $(wc -l < "$MAIN_FILE") commits" ;;
  *) log "coins: development, a local challenger session on this node (PP_CHALLENGER ${PP_CHALLENGER:-local}, DEVELOPMENT ${DEVELOPMENT:-})" ;;
esac
log "window run of $row_id from $SHA on $(hostname), card $CARD (free: 0 processes, ${used} MiB, lease: $lease), cores $CORES, lease at most $LEASE_MIN min, COMPOSE_NODE $COMPOSE_NODE${PEARLPLUS_TIE:+, tied to Pearl+}"
export PATH=$RESEARCH_RUN_DIR/bin:$PATH LEASE_MIN OUT
if [ ! -f "$CUDA_HOME/include/cusparse.h" ]; then  # the venv's CUDA headers where the node's CUDA lacks them
  for lib in cusparse cusolver cufft curand; do
    for d in "$VENV"/lib/python*/site-packages/nvidia/"$lib"/include; do [ -d "$d" ] && CPATH=${CPATH:+$CPATH:}$d; done
  done
  export CPATH
fi
mkdir -p "$SERVE_ROOT" /dev/shm/vllm-served-run && chmod 700 "$SERVE_ROOT" /dev/shm/vllm-served-run
D=/dev/shm/vllm-served-run/${RESEARCH_RUN_ID:?run under research run}; export D
echo "$D" > "$OUT/serve_dir.txt"
t0=$(date +%s)
bash benchmarks/one_stage/served_verdicts.sh > "$OUT/served_verdicts.log" 2>&1
rc=$?
log "served_verdicts.sh exit $rc after $(( ($(date +%s) - t0) / 60 )) min; its lines in out/verdicts.txt"
C=$D/private/compose pending=
if [ -f "$C/inputs.txt" ] && [ -f "$C/compose.env" ]; then
  why=$(python3 benchmarks/one_stage/served_verdicts.py witness < "$C/inputs.txt") \
    || { log "HANDOFF NOT PACKED: the list names the witness: $(echo "$why" | head -1 | cut -c1-300)"; exit 4; }
  mkdir -p "$OUT/handoff"
  (cd "$D" && { cat private/compose/inputs.txt; echo private/compose/compose.env; } | sort -u \
     | tar --create --dereference --no-recursion --file="$OUT/handoff/compose.tar" --files-from=-) \
    || { log "HANDOFF NOT PACKED: tar failed"; exit 4; }
  (cd "$OUT/handoff" && sha256sum compose.tar > compose.tar.sha256)
  log "HANDOFF packed: $(wc -l < "$C/inputs.txt") files, $(du -h "$OUT/handoff/compose.tar" | cut -f1); the serve dir $D stays on this node"
  pending="composed off this node: \`run.sh compose\` with this run's out/handoff/"
elif [ -f "$D/private/deferred/deferred.env" ]; then
  pending="audited and composed off this node: V3c's audit deferred to a prover node from $D/private/deferred (\`served_verdicts.sh copy deferred HOST\`, then \`served_verdicts.sh v3c\` there)"
else
  log "no compose hand-off in $C"
  [ "$rc" != 0 ] || rc=4
fi
[ "$rc" = 2 ] && [ -n "$pending" ] && { log "pending: $pending"; exit 0; }
exit "$rc"
