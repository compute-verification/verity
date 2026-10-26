#!/usr/bin/env bash
# The band-fill wipe (benchmarks/pous/band_wipe/) as research runs: the prover on the GPU node, the memory challenger on another
# machine, which reaches the prover's 127.0.0.1 through ssh (its line of /etc/vy/peers). A node with /etc/vy/runs-via-slot
# (vy-nebius-1 and -2) takes it only through a GPU slot, with the GPU from gpu-lease:
#
#   research run --on vy-nebius-1 --cwd source --source . -- env CHECK_SLOTS=/workspace/research/locks/gpu/slots \
#     python3 tools/verity/check/slot.py --by LANE -- gpu-lease 1 --wait --max-min 45 --mem-gb 24 --preemptible -- \
#     bash benchmarks/pous/band_wipe/run.sh ROLE=prover [SEGMENTS=N]
#   research run --on vy-nebius-cpu-3 --cwd source --source . -- \
#     bash benchmarks/pous/band_wipe/run.sh ROLE=challenger PEER=vy-nebius-1 [MODES=honest,cheat] [SESSIONS=1] [SIGNER=lms]
#   research run --on vy-dc-h200-2 --gpus 1 --cwd source --source . -- \
#     bash benchmarks/pous/band_wipe/run.sh ROLE=loopback [SEGMENTS=N] [VERIFY=0]
#   research run --on vy-nebius-cpu-2 --cwd source --source . --send <run's out/> -- \
#     bash benchmarks/pous/band_wipe/run.sh ROLE=verify RECORD=<its name under inputs/>
#
# ROLE=loopback runs both on the GPU node, the memory challenger over loopback with the test signer (SIGNER=lms for S1's key):
# a rehearsal and the fill time, never a run of record, since the memory challenger must run on a machine the prover doesn't
# control. VERIFY=0 leaves the verdict to a ROLE=verify run on a CPU node (at the full device, the honest session's 2152
# blocks take the reference encoder hours of CPU). CTRL_PORT and RESP_PORT (47610, 47611) are the prover's control and
# responder ports on its 127.0.0.1; CCPU, SSHCPU and RCPU the cores of the memory challenger, its ssh and the responder.
# WORK=<dir> holds the venvs (default /workspace as pod.sh, else $HOME where /workspace isn't writable: $WORK/pous-bench) and
# POUS_BUILD the pous_native build (default $WORK/pous-bench/band_wipe_build). S1's key and its session log are in
# ~/.local/state/verity-band-wipe on the challenger's machine (challenger.STATE), made there by the first run.
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uxo pipefail
REPO=$(pwd); I=$RESEARCH_RUN_DIR/inputs; O=${OUT:-$RESEARCH_RUN_DIR/out}
W=${WORK:-$([ -w /workspace ] && echo /workspace || echo "$HOME")}/pous-bench
CTRL_PORT=${CTRL_PORT:-47610}; RESP_PORT=${RESP_PORT:-47611}; ROLE=${ROLE:?ROLE=prover, challenger, loopback or verify}
export PYTHONPATH=$REPO:$REPO/benchmarks/pous
mkdir -p "$O" "$W"
command -v uv >/dev/null || { python3 -m pip install -q uv || exit 3; }
hostname; nproc; cat /sys/fs/cgroup/cpu.max 2>/dev/null || true

cpu_env() {  # the memory challenger's Python: verity's core needs only the standard library
  PY=$W/cpu312/bin/python
  [ -x "$PY" ] || uv venv -q "$W/cpu312" --python 3.12 || return 3
}
nvcc_release() { "$1/bin/nvcc" --version 2>/dev/null | sed -n 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1 \2/p'; }
gpu_env() {  # pod.sh's: the installed CUDA 13 or 12 toolkit (12.8+ on Blackwell), torch from the index for its major
  local need=0 cc home major minor
  cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
  [ "${cc%%.*}" -ge 10 ] 2>/dev/null && need=8
  for home in ${CUDA_HOME:-} /usr/local/cuda /usr/local/cuda-13.0 /usr/local/cuda-12.9 /usr/local/cuda-12.8 ""; do
    [ -n "$home" ] || { echo "no CUDA 13 or CUDA 12 toolkit for this GPU"; return 3; }
    read -r major minor <<< "$(nvcc_release "$home")"
    [ "${major:-0}" = 13 ] && break
    [ "${major:-0}" = 12 ] && [ "${minor:-0}" -ge "$need" ] && break
  done
  export CUDA_HOME=$home PATH=$home/bin:$PATH
  if [ "$major" = 13 ]; then TORCH_CUDA=${TORCH_CUDA:-cu130}; TORCH_VERSION=${TORCH_VERSION:-2.14.0}
  else TORCH_CUDA=${TORCH_CUDA:-$([ "${minor:-8}" -ge 8 ] && echo cu128 || echo cu126)}; TORCH_VERSION=${TORCH_VERSION:-2.7.1}; fi
  local venv=$W/gpu312-torch$TORCH_VERSION-$TORCH_CUDA
  PY=$venv/bin/python
  [ -x "$PY" ] || { uv venv -q "$venv" --python 3.12 && uv pip install -q --python "$PY" numpy ninja \
    && uv pip install -q --python "$PY" "torch==$TORCH_VERSION" --index-url "https://download.pytorch.org/whl/$TORCH_CUDA"; } || return 3
  "$PY" -c 'import torch; assert torch.cuda.is_available(), "torch sees no CUDA device"; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))' || return 3
  export PATH=$(dirname "$PY"):$PATH
  nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.total,clocks.max.sm,compute_cap --format=csv > "$O/gpu.txt" 2>&1
  nvcc --version > "$O/nvcc.txt" 2>&1
}
prover() {
  "$PY" -m band_wipe.prover --out "$O" --build-dir "${POUS_BUILD:-$W/band_wipe_build}" --listen "127.0.0.1:$CTRL_PORT" \
    --resp-port "$RESP_PORT" ${SEGMENTS:+--segments "$SEGMENTS"} ${RCPU:+--responder-cpu "$RCPU"} ${THREADS:+--threads "$THREADS"} \
    ${RESERVE_MIB:+--reserve-mib "$RESERVE_MIB"}
}
challenger() {
  local args=(--out "$O" --ctrl-port "$CTRL_PORT" --resp-port "$RESP_PORT" --modes "${MODES:-honest,cheat}"
              --sessions "${SESSIONS:-1}" --signer "$SIGNER" ${CCPU:+--cpu "$CCPU"} ${SSHCPU:+--ssh-cpu "$SSHCPU"}
              ${WORKERS:+--workers "$WORKERS"} ${WAIT_S:+--wait-s "$WAIT_S"} ${PINGS:+--pings "$PINGS"})
  [ "${VERIFY:-1}" = 0 ] && args+=(--no-verify)
  "$PY" -m band_wipe.challenger run "$@" "${args[@]}"
}
case $ROLE in
  prover) gpu_env || exit 3; prover; exit $? ;;
  challenger) cpu_env || exit 3; SIGNER=${SIGNER:-lms}; lscpu -e > "$O/lscpu.txt" 2>&1
    challenger --peer "${PEER:?PEER=<the prover machine, a line of /etc/vy/peers>}"; exit $? ;;
  loopback) gpu_env || exit 3; SIGNER=${SIGNER:-test}; lscpu -e > "$O/lscpu.txt" 2>&1
    prover > "$O/prover.log" 2>&1 &
    PP=$!
    challenger --direct 127.0.0.1; RC=$?
    wait $PP; PRC=$?
    tail -5 "$O/prover.log"
    exit $(( RC ? RC : PRC )) ;;
  verify) cpu_env || exit 3; "$PY" -m band_wipe.challenger verify --record "$I/${RECORD:-.}" --out "$O" ${WORKERS:+--workers "$WORKERS"}
    exit $? ;;
  *) echo "ROLE=$ROLE: prover, challenger, loopback or verify"; exit 2 ;;
esac
