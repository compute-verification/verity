#!/usr/bin/env bash
# The window run on the served node, which `run.sh window` starts (placement.toml's [run.window]): benchmarks/one_stage/
# served_verdicts.sh on card CARD alone (a gpu-lease shim in the run dir adds `--on CARD`), its lease ending 15 min before
# END, with the node's paths and the run's settings as its environment. What it reads from the node, which the node's
# setup leaves there: the INT8 pin under HF_HOME, the taps (HIDDEN_SO, NORM_TAP_SO), the traced Build (ROW) and Flock's
# build (FLOCK_WORK). The engine reads the pin at the checkpoint manifest's local_path, so that model directory is one
# link to the pin under HF_HOME, made where nothing is at that name and removed at exit. The serve dir D is in memory
# (/dev/shm, mode 700), its caches under SERVE_ROOT, as /dev/shm is mounted noexec; D stays on the node. After the
# run, the compose hand-off (private/compose/inputs.txt's files, verifier-side, `served_verdicts.py witness` passing on
# the list again) is packed into $OUT/handoff/compose.tar for `run.sh compose`.
# Exit: served_verdicts.sh's (2 = pending, the expected end: the record is composed off this node); 75 when the card
# is in use or the row leaves under an hour of lease, with nothing run; 2 for a missing input; 4 when the hand-off is
# not packed.
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
  [ -e "$f" ] || { log "the node holds no $f: nothing run"; exit 2; }
done
want=$(awk -v f="$row_id.json" '$2 == f {print $1}' adapters/vllm/workloads/SHA256SUMS)
[ "$(sha256sum "$IN/$row_id.json" | cut -d' ' -f1)" = "$want" ] || { log "the workload file is not the one SHA256SUMS lists"; exit 2; }
cp "$IN/$row_id.json" adapters/vllm/workloads/ || exit 2

local_path=$(python3 -c 'import json, sys; print(next(c["local_path"] for c in json.load(open(sys.argv[1]))["checkpoints"] if c["repo"] == sys.argv[2] and c["revision"] == sys.argv[3]))' \
  adapters/vllm/manifests/checkpoints-int8.json "$REPO" "$REV") || { log "the checkpoint manifest names no local_path for $REPO@$REV"; exit 2; }
LINK=${local_path%/snapshots/*} SRC=$HF_HOME/hub/models--${REPO//\//--}
if ! { [ -L "$LINK" ] && [ "$(readlink "$LINK")" = "$SRC" ]; }; then
  { [ ! -e "$LINK" ] && [ ! -L "$LINK" ]; } || { log "$LINK exists and is not a link to $SRC: nothing touched, nothing run"; exit 2; }
  ln -s "$SRC" "$LINK" || { log "can't make the link $LINK: nothing run"; exit 2; }
fi
trap '[ -L "$LINK" ] && rm -f "$LINK"' EXIT
[ -s "$local_path/model.safetensors" ] || { log "no weights through $LINK: nothing run"; exit 2; }

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
  gate:*)
    PP_CHALLENGER_KEY=$TREE/$PP_CHALLENGER_KEY PP_GATE_KEY=${PP_GATE_KEY/#\~/$HOME} MAIN_FILE=$IN/main.txt
    for f in "$PP_CHALLENGER_KEY" "$PP_GATE_KEY" "$MAIN_FILE"; do [ -r "$f" ] || { log "the gate's coins need $f: nothing run"; exit 2; }; done
    export PP_CHALLENGER_KEY PP_GATE_KEY MAIN_FILE
    log "coins: the gate $PP_CHALLENGER (peer ${PP_PEER:-}), key record $(sha256sum "$PP_CHALLENGER_KEY" | cut -c1-16)…, main's line of $(wc -l < "$MAIN_FILE") commits" ;;
  *) log "coins: development, a local challenger session on this node (PP_CHALLENGER ${PP_CHALLENGER:-local}, DEVELOPMENT ${DEVELOPMENT:-})" ;;
esac
log "window run of $row_id from $SHA on $(hostname), card $CARD (free: 0 processes, ${used} MiB, lease: $lease), cores $CORES, lease at most $LEASE_MIN min, COMPOSE_NODE $COMPOSE_NODE"
log "disk: $(df -h --output=avail,target /workspace /dev/shm | tail -2 | tr -s ' ' | tr '\n' ';') memory available $(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo) GiB"
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
log "served_verdicts.sh exit $rc after $(( ($(date +%s) - t0) / 60 )) min (2 = pending: composed off this node); its lines in out/verdicts.txt"
C=$D/private/compose
if [ -f "$C/inputs.txt" ] && [ -f "$C/compose.env" ]; then
  why=$(python3 benchmarks/one_stage/served_verdicts.py witness < "$C/inputs.txt") \
    || { log "HANDOFF NOT PACKED: the list names the witness: $(echo "$why" | head -1 | cut -c1-300)"; exit 4; }
  mkdir -p "$OUT/handoff"
  (cd "$D" && { cat private/compose/inputs.txt; echo private/compose/compose.env; } | sort -u \
     | tar --create --dereference --no-recursion --file="$OUT/handoff/compose.tar" --files-from=-) \
    || { log "HANDOFF NOT PACKED: tar failed"; exit 4; }
  (cd "$OUT/handoff" && sha256sum compose.tar > compose.tar.sha256)
  log "HANDOFF packed: $(wc -l < "$C/inputs.txt") files, $(du -h "$OUT/handoff/compose.tar" | cut -f1); the serve dir $D stays on this node"
else
  log "no compose hand-off in $C"
  [ "$rc" != 0 ] || rc=4
fi
exit "$rc"
