#!/usr/bin/env bash
# Flock's zero-knowledge session cost on the run's GPU (zk_cost.py): A6's W8A8 coordinates in statements of 1-512 units, sessions
# by `flock-circuit prove --zk --gpu` with and without the ZK server's send check, the Lean verifier on them, and the fit. Run
# from the verity tree on a GPU node, as a GPU host job (the GPU taken by gpu-lease around the sessions only, where the node
# runs gpu-lease: its installed one, else the tree's gpu_lease.sh where /run/gpu-lease holds the node's locks; a one-GPU pod
# without either lends the run its GPU for the whole run):
#   bash benchmarks/one_stage/zk_cost.sh
#   (a job run from a clone of the tree, on a GPU host, 32 CPUs, 96 GB, up to 240 min)
# (a guest job on a shared GPU node is preemptible; one arm a run, ZK_ARMS=fw, takes about 3.5 h with its verification, the builds cached)
# env: ZK_KS=4096,14336  ZK_SIZES=1,8,64,512  ZK_DRAWS=1,8,64,c  ZK_ARMS=fw,nofw  ZK_COLD=2  ZK_RUNS=3  ZK_WARM=1
#      ZK_UNITS="fp8-coord:K=6144 router" (served types instead of A6's coordinates, zk_cost.py's --units: ZK_SIZES then
#      defaults to half,full, ZK_DRAWS to c:dev+nofw,8:dev+nofw+fw,64:dev+nofw+fw, ZK_ARMS to dev,nofw,fw; overhead instead of fit)
#      ZK_STAGE_ONLY=1 (with ZK_UNITS: stage on this node's CPUs, keep the records, no CUDA, build or GPU)
#      ZK_SM (the prover's target, from nvidia-smi unless given)  ZK_CARD=rtx-pro-6000-server (the card whose datasheet
#      peak divides; the overhead step refuses one whose arch isn't the run's GPU's; an H200: ZK_CARD=h200-sxm-141gb)
#      LEASE_MAX_MIN=30 (one GPU lease a point)
#      LEASE_MEM_GB (the lease's memory cap, gpu-lease's --mem-gb; else gpu-lease's own per-GPU default)
#      ZK_NICE=0 (the staging's and the builds' nice; the sessions run at the job's own)
#      ZK_SCRATCH=$FLOCK_WORK (the run's scratch, a mktemp dir under it: the staged statements, each deleted once its last
#      point is proved; data only, so a noexec tmpfs such as /dev/shm keeps them, about 24 GB of circuit.txt for
#      FlashInfer K=2048 a size, off a shared disk)
#      ZK_NEED_GB, ZK_WORK_GB (this run's peaks under ZK_SCRATCH, with the stagers' memory where it is a tmpfs, and under
#      FLOCK_WORK, the builds; given, nothing is built or staged until each filesystem, and memory for a tmpfs, has its
#      peak + ZK_MARGIN_GB=30 free, plus the GB of each ZK_HOLD="RUN_ID:GB ..." run not yet done, failed or cancelled;
#      ZK_GATE_UNTIL, a UTC time, gives up the wait)
#      ZK_LAST_START (a UTC time no point starts at or after: one granted later gives its GPU back unproved)
#      ZK_VERIFY=0 (no Lean verify: the sessions' prices alone, each accepted by its ZK server; with ZK_UNITS and no fw
#      arm, no Lean build either: nothing then reads flock-verify or the ZK server's Lean frame)
#      FLOCK_WORK=/workspace/jobs/proof-throughput (the CUDA prover's build and venv, as proof_throughput's job.sh keeps them)
#      PT_WORK=/workspace/cache/proof-throughput (the Lean verifier's build, as its verify.sh keeps it)
# Kept in $OUT: the records (stage, prove and verify .jsonl), the fit (fit.json, fit.md, observations.json) and the logs;
# never a circuit, a prover's file or a proof.
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uo pipefail
REPO=$(pwd); O=${OUT:-${RESEARCH_RUN_DIR:?}/out}; mkdir -p "$O"
export FLOCK_WORK=${FLOCK_WORK:-/workspace/jobs/proof-throughput}; mkdir -p "$FLOCK_WORK" || { echo "cannot make $FLOCK_WORK" >&2; exit 1; }
PT_WORK=${PT_WORK:-/workspace/cache/proof-throughput}
mkdir -p "$PT_WORK" 2>/dev/null && [ -w "$PT_WORK" ] || PT_WORK=$FLOCK_WORK/verifier
ZK_SCRATCH=${ZK_SCRATCH:-$FLOCK_WORK}; mkdir -p "$ZK_SCRATCH" || { echo "cannot make $ZK_SCRATCH" >&2; exit 1; }
S=$(mktemp -d "$ZK_SCRATCH/zk-cost.XXXXXX"); trap 'rm -rf "$S"' EXIT; trap 'exit 143' TERM INT HUP
log() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$O/job.log"; }
free_gb() {
  local f m; f=$(df -B1G --output=avail "$1" | tail -1 | tr -d ' ')
  if [ "$(stat -f -c %T "$1")" = tmpfs ]; then
    m=$(awk '/^MemAvailable:/ {print int($2 / 1048576)}' /proc/meminfo); [ "$m" -lt "$f" ] && f=$m
  fi
  echo "$f"
}
room() {
  local h st held d f n ok msg said= runs=${RESEARCH_RUN_DIR:-/workspace/research/runs/-}
  local -A need at
  while :; do
    held=0
    for h in ${ZK_HOLD:-}; do
      st=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1])).get("state", ""))' \
        "${runs%/*}/${h%%:*}/status.json" 2> /dev/null)
      case $st in done | failed | cancelled) ;; *) held=$((held + ${h##*:})) ;; esac
    done
    need=() at=() ok=1 msg=
    for h in "$ZK_SCRATCH:${ZK_NEED_GB:-0}" "$FLOCK_WORK:${ZK_WORK_GB:-0}"; do
      d=$(stat -c %d "${h%:*}"); at[$d]=${h%:*}; need[$d]=$((${need[$d]:-0} + ${h##*:}))
    done
    for d in "${!need[@]}"; do
      f=$(free_gb "${at[$d]}"); n=$((need[$d] + ${ZK_MARGIN_GB:-30} + held))
      msg="$msg ${at[$d]}: $f GB free, $n needed;"; [ "$f" -ge "$n" ] || ok=0
    done
    [ "$ok" = 1 ] && { log "zk-cost: room ($held GB held):$msg"; return 0; }
    if [ -n "${ZK_GATE_UNTIL:-}" ] && [ "$(date -u +%s)" -ge "$(date -u -d "$ZK_GATE_UNTIL" +%s)" ]; then
      log "zk-cost: no room by $ZK_GATE_UNTIL ($held GB held):$msg"; return 1
    fi
    [ "$said" = "$msg" ] || log "zk-cost: waiting for room ($held GB held):$msg"
    said=$msg; sleep 60
  done
}
[ -z "${ZK_NEED_GB:-}${ZK_WORK_GB:-}" ] || room || exit 75
export PYTHONDONTWRITEBYTECODE=1
unset CARGO_TARGET_DIR
{ lscpu | grep -E 'Model name|^CPU\(s\)'; nproc; free -g; } > "$O/host.txt" 2>&1
PY=$FLOCK_WORK/flock-circuit/py/bin/python3
if [ ! -x "$PY" ]; then
  UV=$(command -v uv || echo "$HOME/.local/bin/uv"); mkdir -p "$FLOCK_WORK/flock-circuit"
  (exec 8>"$FLOCK_WORK/flock-circuit/venv.lock"; flock 8; [ -x "$PY" ] || { "$UV" venv -q --python 3.12 "$FLOCK_WORK/flock-circuit/py" \
    && "$UV" pip install -q --python "$PY" numpy blake3; })
fi
export PYTHONPATH=$REPO:$REPO/verity/ml/flock/python:$REPO/integrations/vllm:$REPO/tools/verity/circuit_check/src:$REPO/benchmarks/numerical/python
Z=$S/zk
ZK() { "$PY" benchmarks/one_stage/zk_cost.py "$@" --out "$Z"; }
keep() {
  (cd "$Z" && find . -type f -size -8M ! -name circuit.txt ! -name 'inst*.bin' ! -name 'pub*.bin' ! -path '*/archive/*' \
    \( ! -path '*/sessions/*/*' -o -name session.json \) -print0 | xargs -0 -r -I{} cp --parents {} "$O/")
}
# a served type stages on CPUs alone (stage.py, no Lean), before the builds
if [ -n "${ZK_UNITS:-}" ]; then
  : "${ZK_SIZES:=half,full}" "${ZK_DRAWS:=c:dev+nofw,8:dev+nofw+fw,64:dev+nofw+fw}" "${ZK_ARMS:=dev,nofw,fw}"
  log "zk-cost: source ${RESEARCH_SOURCE_SHA:-$(git rev-parse HEAD 2>/dev/null)}; stage: units $ZK_UNITS, sizes $ZK_SIZES, draws $ZK_DRAWS"
  nice -n "${ZK_NICE:-0}" "$PY" benchmarks/one_stage/zk_cost.py stage --units "$ZK_UNITS" --sizes "$ZK_SIZES" \
    --draws "$ZK_DRAWS" --out "$Z" 2>&1 | tee -a "$O/zk.log" | grep ZKCOST
  rc=${PIPESTATUS[0]}
  if [ "${ZK_STAGE_ONLY:-0}" = 1 ] || [ "$rc" != 0 ]; then keep; log "zk-cost: staged only, rc $rc"; exit "$rc"; fi
fi
for c in "$FLOCK_WORK/cuda-root" /workspace/cache/flock/cuda-root; do
  [ ! -x /usr/local/cuda-13.3/bin/nvcc ] && [ -z "${CUDA_TK:-}" ] && [ -x "$c/usr/local/cuda-13.3/bin/nvcc" ] && export CUDA_TK=$c/usr/local/cuda-13.3
done
# else its packages unpacked under FLOCK_WORK without root, as 71-gemm-slowdown.sh does (an R580 driver runs 13.3)
if [ ! -x /usr/local/cuda-13.3/bin/nvcc ] && [ -z "${CUDA_TK:-}" ]; then
  CR=$FLOCK_WORK/cuda-root
  todo="cuda-nvcc-13-3 cuda-cudart-dev-13-3"; seen=
  while [ -n "$todo" ]; do
    set -- $todo; q=$1; shift; todo="$*"
    case " $seen " in *" $q "*) continue ;; esac
    seen="$seen $q"
    todo="$todo $(apt-cache show "$q" 2>/dev/null | awk '/^Depends:/ {sub(/^Depends: /, ""); print; exit}' | tr ',' '\n' \
      | sed 's/(.*//; s/|.*//; s/ //g' | grep -E -- '-13-3$' | tr '\n' ' ')"
  done
  (exec 8>"$FLOCK_WORK/cuda-root.lock"; flock 8; [ -x "$CR/usr/local/cuda-13.3/bin/nvcc" ] || {
    mkdir -p "$CR/debs" && (cd "$CR/debs" && apt-get download $(for q in $seen; do
      echo "$q=$(apt-cache show "$q" | awk '/^Version:/ {print $2; exit}')"; done)) \
      && for d in "$CR"/debs/*.deb; do dpkg-deb -x "$d" "$CR"; done; }) > "$O/cuda-unpack.log" 2>&1
  [ -x "$CR/usr/local/cuda-13.3/bin/nvcc" ] && export CUDA_TK=$CR/usr/local/cuda-13.3
fi
[ -x /usr/local/cuda-13.3/bin/nvcc ] || [ -n "${CUDA_TK:-}" ] || { log "no CUDA 13.3 on this node"; tail -5 "$O/cuda-unpack.log"; exit 1; }
C13=${CUDA_TK:-/usr/local/cuda-13.3}; export LD_LIBRARY_PATH=$C13/compat:$C13/lib64:${LD_LIBRARY_PATH:-}

# Lean only where a step reads it: A6's staging draws, the verify step, and the ZK server's frame for an fw arm
NEED_LEAN=1
if [ -n "${ZK_UNITS:-}" ] && [ "${ZK_VERIFY:-1}" = 0 ] \
  && ! echo ",${ZK_ARMS:-fw,nofw},${ZK_DRAWS:-}," | tr ':+' ',,' | grep -q ',fw,'; then NEED_LEAN=0; fi
log "zk-cost: source ${RESEARCH_SOURCE_SHA:-$(git rev-parse HEAD 2>/dev/null)}, builds under $FLOCK_WORK$([ "$NEED_LEAN" = 1 ] \
  && echo " and $PT_WORK" || echo " (no Lean: nothing reads it)")"
OUT=$O/build nice -n "${ZK_NICE:-0}" bash verity/ml/flock/pod/70-class-sweep.sh BUILD_ONLY=1 GPU=1 SM="${ZK_SM:-}" \
  > "$O/build.txt" 2>&1 &
[ "$NEED_LEAN" = 1 ] && OUT=$O/lean nice -n "${ZK_NICE:-0}" bash benchmarks/proof_throughput/verify.sh BUILD_ONLY=1 \
  PT_WORK="$PT_WORK" > "$O/lean-build.txt" 2>&1 &
wait
BIN=$(sed -n 's/^BINARY=//p' "$O/build.txt" | tail -1)
BINS=("$BIN")
if [ "$NEED_LEAN" = 1 ]; then
  LEAN=$(sed -n 's/.*BUILD_ONLY: \(\S*\) .*/\1/p' "$O/lean/job.log" 2>/dev/null | tail -1)
  # the ZK server's Lean frame (FC_FIREWALL=1 spawns it per rep), built in the same package as flock-verify
  export FC_FIREWALL_LEAN=${LEAN%/*}/flock-firewall
  if [ -x "$LEAN" ] && [ ! -x "$FC_FIREWALL_LEAN" ]; then
    command -v lake > /dev/null || export PATH="$HOME/.elan/bin:$PATH"
    (exec 8>"$PT_WORK/lean.lock"; flock 8; [ -x "$FC_FIREWALL_LEAN" ] || (cd "${LEAN%/.lake/*}" && lake build flock-firewall)) \
      > "$O/lake-firewall.log" 2>&1
  fi
  BINS+=("$LEAN" "$FC_FIREWALL_LEAN")
fi
for p in "${BINS[@]}" "$PY"; do
  [ -x "$p" ] || { log "missing: '$p'"; tail -20 "$O/build.txt" "$O/lean-build.txt" "$O/lake-firewall.log" 2>/dev/null; exit 1; }
done
sha256sum "${BINS[@]}" | tee "$O/binaries.sha256" | tee -a "$O/job.log"

if [ -z "${ZK_UNITS:-}" ]; then
  log "stage: K ${ZK_KS:-4096,14336}, sizes ${ZK_SIZES:-1,8,64,512}, draws ${ZK_DRAWS:-1,8,64,c}"
  ZK stage --lean "$LEAN" --ks "${ZK_KS:-4096,14336}" --sizes "${ZK_SIZES:-1,8,64,512}" --draws "${ZK_DRAWS:-1,8,64,c}" 2>&1 \
    | tee -a "$O/zk.log" | grep ZKCOST
  rc=${PIPESTATUS[0]}
fi
if [ "$rc" = 0 ]; then
  if [ -x /workspace/verity-guest/bin/gpu-lease ] && [ -e /run/gpu-lease/pool.gpus ]; then
    LEASE=(env GPU_LEASE_DIR=/run/gpu-lease GPU_LEASE_ALLOWED_FILE=/run/gpu-lease/pool.gpus /workspace/verity-guest/bin/gpu-lease)
  elif command -v gpu-lease > /dev/null; then
    LEASE=(gpu-lease)
  elif [ -d /run/gpu-lease ]; then
    LEASE=(bash "$REPO/tools/research/src/research/pods/sh/gpu_lease.sh")
  else
    LEASE=()
  fi
  [ ${#LEASE[@]} -gt 0 ] && LEASE+=(1 --wait --max-min "${LEASE_MAX_MIN:-30}" ${LEASE_MEM_GB:+--mem-gb "$LEASE_MEM_GB"} --)
  export GPU_LEASE_WHO=${GPU_LEASE_WHO:-proofs/zk-cost/${RESEARCH_RUN_ID:-local}}
  log "prove: arms ${ZK_ARMS:-fw,nofw}, ${ZK_COLD:-2} cold, ${ZK_WARM:-1}+${ZK_RUNS:-3} warm ($([ ${#LEASE[@]} -gt 0 ] \
    && echo "waiting for a GPU" || echo "no gpu-lease on this node: the run's own GPU"))"
  # one point's sessions at a time on this node, as proof_throughput's: two would share its host CPUs and memory bandwidth;
  # a lease a point keeps each lease short
  n=$("$PY" -c 'import json, sys; print(len(json.load(open(sys.argv[1]))))' "$Z/plan.json")
  last=${ZK_LAST_START:+$(date -u -d "$ZK_LAST_START" +%s)}
  for i in $(seq 0 $((n - 1))); do
    if [ -n "$last" ] && [ "$(date -u +%s)" -ge "$last" ]; then log "zk-cost: points $i on not started: past $ZK_LAST_START"; rc=1; break; fi
    flock "$FLOCK_WORK/sessions.lock" "${LEASE[@]}" bash -c '[ -z "$1" ] || [ "$(date -u +%s)" -lt "$1" ] \
      || { echo "ZKCOST point granted past ZK_LAST_START: not proved"; exit 75; }; shift; exec "$@"' _ "$last" \
      "$PY" benchmarks/one_stage/zk_cost.py prove --bin "$BIN" --out "$Z" --points "$i" --arms "${ZK_ARMS:-fw,nofw}" \
      --cold "${ZK_COLD:-2}" --warm "${ZK_WARM:-1}" --runs "${ZK_RUNS:-3}" 2>&1 | tee -a "$O/zk.log" | grep ZKCOST
    [ "${PIPESTATUS[0]}" = 0 ] || rc=1
    # a staged statement goes once its last point is proved: its records kept, its circuit and instances deleted
    keep
    for d in $("$PY" -c 'import json, sys; p, i = json.load(open(sys.argv[1])), int(sys.argv[2]); d = p[i].get("stage_dir")
print(d if d and all(q.get("stage_dir") != d for q in p[i + 1:]) else "")' "$Z/plan.json" "$i"); do
      rm -rf "$d" && log "zk-cost: point $i proved; deleted its statement ${d#"$S"/}"
    done
  done
  if [ "${ZK_VERIFY:-1}" = 1 ]; then
    log "prove: exit $rc; verify"
    ZK verify --lean "$LEAN" 2>&1 | tee -a "$O/zk.log" | grep ZKCOST
    [ "${PIPESTATUS[0]}" = 0 ] || rc=1
  fi
  if [ -n "${ZK_UNITS:-}" ]; then ZK overhead --card "${ZK_CARD:-rtx-pro-6000-server}" 2>&1 | tee -a "$O/zk.log"
  else ZK fit 2>&1 | tee -a "$O/zk.log"; fi
fi
keep
log "zk-cost: done, rc $rc"
exit "$rc"
