#!/usr/bin/env bash
# The POUS benchmark on a GPU pod, as a research Attempt:
#
#   git -C <tree> archive -o impl.tar <commit>        # the implementation's tree: #188's cursor/dense-decode-opt-6667
#   research run --on <pod> --project verity --source . --cwd source --send impl.tar --campaign pous \
#     -- bash benchmarks/pous/pod.sh IMPL=vllm-gpu IMPL_COMMIT=<commit> [WS=qwen2.5-0.5b/linears+head] [AUDITS=20]
#   research run --on <pod> --project verity --source . --cwd source --campaign pous \
#     -- bash benchmarks/pous/pod.sh IMPL=reference [WS=...] [MAX_SEGMENTS=N]
#   ... -- bash benchmarks/pous/pod.sh IMPL=p2-gpu [WS=...]      # built for the device's arch (ARCH=sm_NN overrides)
#   ... -- bash benchmarks/pous/pod.sh IMPL=band-gpu [WS=...] [NVME_DIR=/workspace/nvme] [SCHEME=dense-chain/v1]
#   ... -- bash benchmarks/pous/pod.sh IMPL=p2-v1-gpu [WS=...] [NVME_DIR=...]       # P2 v1 (builds p2_v1/libp2v1.so with CGBN)
#   ... -- bash benchmarks/pous/pod.sh IMPL=p2-v1-gpu P2FAST='-DP2_FAST -DP2_KEYS4 -DP2SQ_KARA' TOKENS=1,8,32,2048,4096  # v4 keys
#   ... -- bash benchmarks/pous/pod.sh IMPL=p2-v3-gpu [WS=...] [NVME_DIR=...]       # p2-16448/v4 on the same library, audited by protocol.TimedAudit
#   ... -- bash benchmarks/pous/pod.sh IMPL=p2-v3-mc-gpu [WS=...] [NVME_DIR=...]    # the same, audited by the memory challenger,
#                                                                               signed with S1's key in $W/pous-mc-lms (MC_SIGNER=test: the test signer)
#   ... -- bash benchmarks/pous/pod.sh IMPL=side-by-side [WS=...] [NVME_DIR=...]    # band-gpu then P2_ARM (p2-v3-gpu), and the table
#   ... -- bash benchmarks/pous/pod.sh IMPL=side-by-side GATES_ONLY=1 MAX_SEGMENTS=2  # build check: kernels built, gates run
#   ... -- bash benchmarks/pous/pod.sh IMPL=band-gpu BAND_D=6 [WS=...]   # the band at in-degree d (>= 5; side-by-side's band arm too)
#
# On a shared node (band-gpu, p2-v1-gpu, side-by-side): WORK=<dir> holds the venvs and builds, VCPU and RCPU the verifier's and
# the responder's cores (default: the 3rd and 4th physical cores), WORKERS the reference gate's processes (default nproc).
# POUS_BUILD=<dir> is the pous_native JIT build directory (default $W/pous_build). A timed run needs a directory no other run
# builds into: a rebuild inside a timed lease is a compile the node's ops daemon stops.
# A timed run there goes inside the node's quiet slice, `gpu-lease 1 --on N --max-min M -- vy-pous-quiet CMD`, with VCPU and
# RCPU among its cores: outside it, neighbours' load makes honest answers late (r20261005-231145-6d0f accepted 18 of 20).
# AUDITS=0 is an untimed placement (a preemptible fill slot): the gates, encode, the synthetic loop and the step floor run, no
# limit, audit or control does, and the result is labelled timed: false, audits: skipped (untimed placement).
# LOAD_TOKENS=T (band-gpu and the P2 arms, bench.py --load-tokens) is T of the POUS forwards run during the honest audit (0: none,
# for a run whose load is another process on the same GPU, as benchmarks/resource_accounting's chain).
# BENCH_HOST='SCRIPT ARGS' (the P2 arms) runs bench.py inside SCRIPT's process: `python SCRIPT ARGS benchmarks/pous/bench.py ...`,
# where SCRIPT calls `bench.main(argv, load=...)` (benchmarks/resource_accounting/chain.py --host-bench: its responder in the
# chain's own process and CUDA context).
# P2_ENCODE_BLOCKS=N (the P2 arms) encodes the first N blocks of C on the device (0: all of C, so C is W's encoding; default
# bench.py's 2048, the rest random canonical blocks); with P2_ENCODE_BLOCKS=0, P2_C_CACHE=1 keeps C under $W/p2-c-cache and
# reuses it (an untimed run encodes, a timed one of the same setup loads it). P2_ARM picks side-by-side's P2 arm (default p2-v3-gpu).
# BANDWIDTH_SALT=<gpu> (p2-v3-gpu, p2-v3-mc-gpu; joint runs only, benchmarks/resource_accounting's run.sh JOINT=) sets C up under
# the bandwidth audit's device-bound salt for that GPU in place of PoUS's own (bench.py --bandwidth-salt).
# KERNELS='<json>' (band-gpu) sets kernel switches over harness.KERNELS (bench.py --kernels): '{"chain_pairs": false}' runs the
# run of record's kernels.
#
# GATE_ROLE=<role> runs the run as that party of the proof service (memory-challenger for the auditor's timed audit on a node):
# before anything is built or timed, bench/placement.py's gate checks where it runs and records placement-gate.json, and a
# refusal exits 4. GATE_CORES are the role's dedicated CPUs (vy-pous-quiet's 124-127 on node 2), GATE_PEERS its peers' probe
# records, space-separated ROLE[:NAME]=FILE (a relative FILE is one of the run's inputs, --send), and GATE_ABSENT the peer
# roles not deployed yet:
#   gpu-lease 1 --on 7 --max-min M --timed -- vy-pous-quiet bash benchmarks/pous/pod.sh IMPL=band-gpu GATE_ROLE=memory-challenger \
#     GATE_CORES=124-127 GATE_PEERS=worker:vy-nebius-2=/etc/vy/placement/vy-nebius-2.json GATE_ABSENT='firewall challenger'
#
# GATES_ONLY=1 (band-gpu, p2-v1-gpu, side-by-side) builds the kernels for the device, runs the kernel gates (the pinned vector
# on the device, the reference segments or blocks, the round trip) and stops: nothing is timed and nothing is published.
#
# HW, the census id recorded with the result, is the pod's GPU: the H100 SXM, the L40S, the RTX 4090 and the RTX PRO 6000
# Blackwell Server Edition are known, and any other GPU, or none, needs HW=<benchmarks/census/hardware.json id>.
#
# Every nvidia-smi query names the run's own GPU (--id): the lease's (GPU_LEASE_UUID), else CUDA_VISIBLE_DEVICES' first, else
# the node's only GPU, never GPU 0 by default; a GPU run on a node with more GPUs and neither is refused. The clocks-*.txt
# files start with the id queried and the index, UUID and bus id that answered.
#
# vllm-gpu bootstraps the implementation tree's Python 3.12 environment (its ops/pod_bootstrap.sh --cpu: the venv, torch cu129,
# vLLM, the QWEN05 checkpoint; no native taps, which this benchmark does not use), then runs bench.py, which drives that tree's
# frozen harness (pous-dense-bench) and gates and profiles its result with this tree's verity.core.protocols.pous. reference runs
# bench.py's CPU reference on the pod's cores in a plain Python 3.12 venv. band-gpu needs no implementation tree: a Python 3.12
# venv with torch built for the pod's CUDA toolkit (TORCH_CUDA=cuNNN overrides), numpy and ninja, and this tree's band_gpu/.
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uxo pipefail
REPO=$(pwd); I=$RESEARCH_RUN_DIR/inputs; O=$RESEARCH_RUN_DIR; W=${WORK:-/workspace}/pous-bench; POUS_BUILD=${POUS_BUILD:-$W/pous_build}
IMPL=${IMPL:-vllm-gpu}; WS=${WS:-qwen2.5-0.5b/linears+head}
# lake: the pod image's /opt/elan, else the user's elan (node 2's research user); verity_difftest serves the memory
# challenger's Lean verdict (its Server and build need only the stdlib)
export PATH=$HOME/.local/bin:/opt/elan/bin:$HOME/.elan/bin:$PATH
# Every entry must exist: importlib.metadata opens a missing one as a zip, and the suite guard counts that open as a read.
export PYTHONPATH=$REPO:$REPO/tools/research/src:$REPO/verity/core/difftest/python
# PREPARE=1 builds the Lean verdict a timed run serves, which it must not compile under its lease, and exits: for
# p2-v3-mc-gpu the memory challenger's (memory-challenger-verdict, live.py's judge), for p2-v3-gpu pous_check's
# (pous-timed-audit, band_gpu's judge) where its lakefile declares one. Run it from a clone (--cwd clone: the build writes
# the package's .lake) before the lease, in the same job:
#   research run --on vy-nebius-2 --cwd clone ... -- bash -c 'bash benchmarks/pous/pod.sh PREPARE=1 IMPL=p2-v3-mc-gpu && \
#     gpu-lease 1 --on N --max-min M -- vy-pous-quiet env ... bash benchmarks/pous/pod.sh IMPL=p2-v3-mc-gpu ...'
if [ "${PREPARE:-0}" = 1 ]; then
  case $IMPL in
    p2-v3-mc-gpu) set -- verity/core/devices/memory_challenger/lean memory-challenger-verdict ;;
    p2-v3-gpu) set -- verity/core/protocols/pous/check pous-timed-audit ;;
    *) exit 0 ;;
  esac
  grep -q "^name = \"$2\"" "$1/lakefile.toml" || exit 0
  (cd "$1" && lake build "$2") > "$O/lean-build.log" 2>&1 || { tail -30 "$O/lean-build.log"; exit 3; }
  exit 0
fi
# the run's GPU for nvidia-smi --id: a UUID, or an index as nvidia-smi counts them (gpu-lease's CUDA_DEVICE_ORDER=PCI_BUS_ID)
gpu_uuid() {
  local v=${GPU_LEASE_UUID:-${CUDA_VISIBLE_DEVICES:-}} l
  v=${v%%,*}
  case $v in
    GPU-*|MIG-*) echo "$v" ;;
    "") l=$(nvidia-smi -L 2>/dev/null | grep '^GPU ')
        [ "$(printf '%s\n' "$l" | grep -c '^GPU ')" = 1 ] && echo "$l" | sed -n 's/.*(UUID: \([^)]*\)).*/\1/p' ;;
    *) nvidia-smi --id="$v" --query-gpu=uuid --format=csv,noheader 2>/dev/null | head -1 ;;
  esac
}
GPU_UUID=$(gpu_uuid)
smi() { [ -n "$GPU_UUID" ] && nvidia-smi --id="$GPU_UUID" "$@"; }
clocks() {  # clocks NAME: $O/clocks-NAME.txt, headed by the id queried and the GPU that answered (untraced: no -x lines in it)
  local -; set +x
  { echo "nvidia-smi --id=$GPU_UUID: $(smi --query-gpu=index,uuid,pci.bus_id --format=csv,noheader)"
    smi -q -d CLOCK,PERFORMANCE,POWER; } > "$O/clocks-$1.txt" 2>&1 || true
}
if [ -z "$GPU_UUID" ] && [ "$IMPL" != reference ] && [ "$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ')" -gt 1 ]; then
  echo "several GPUs and neither GPU_LEASE_UUID nor CUDA_VISIBLE_DEVICES: name the run's GPU (gpu-lease, or" \
       "CUDA_VISIBLE_DEVICES=<index|UUID>) so its clocks and facts are read from it"; exit 2
fi
GPU=$(smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
case "$GPU" in                                            # the census id is the device's, never a default another GPU inherits
  *"H100 80GB HBM3"*) HW=${HW:-h100-sxm5-80gb/bf16} ;;
  *L40S*) HW=${HW:-l40s-48gb/bf16} ;;
  *"RTX 4090"*) HW=${HW:-rtx-4090/e4m3} ;;                 # the census's RTX 4090 row (it has no bf16 row yet)
  *"RTX PRO 6000 Blackwell Server Edition"*) HW=${HW:-rtx-pro-6000-server/bf16} ;;  # sm_120; other RTX PRO 6000s need HW=
  *) [ -n "${HW:-}" ] || { echo "no census id for GPU '$GPU': pass HW=<benchmarks/census/hardware.json id>"; exit 2; } ;;
esac
SCHEME=${SCHEME:-$([ "$IMPL" = p2-gpu ] && echo p2-16448/v2 || echo band-chain/d12/v1)}
AUDITS=${AUDITS:-20}; WORKERS=${WORKERS:-$(nproc)}; P2_ARM=${P2_ARM:-p2-v3-gpu}
GO=$([ "${GATES_ONLY:-0}" = 1 ] && echo --gates-only)
mkdir -p "$W" "$O/bench"
nvcc_release() { "$1/bin/nvcc" --version 2>/dev/null | sed -n 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1 \2/p'; }
# The installed CUDA toolkit for torch's JIT build and P2 v1's library; nothing is installed or changed on the system (a
# shared node's toolkit is the node's): CUDA_HOME, /usr/local/cuda, or a /usr/local/cuda-13.0, -12.9 or -12.8, the first that
# is CUDA 13 or CUDA 12 (at least 12.8 on a Blackwell GPU: sm_100, sm_120)
cuda_home() {
  local need=0 cc home major minor
  cc=$(smi --query-gpu=compute_cap --format=csv,noheader | head -1)
  [ "${cc%%.*}" -ge 10 ] 2>/dev/null && need=8
  for home in ${CUDA_HOME:-} /usr/local/cuda /usr/local/cuda-13.0 /usr/local/cuda-12.9 /usr/local/cuda-12.8; do
    read -r major minor <<< "$(nvcc_release "$home")"
    [ "${major:-0}" = 13 ] && { echo "$home"; return 0; }
    [ "${major:-0}" = 12 ] && [ "${minor:-0}" -ge "$need" ] && { echo "$home"; return 0; }
  done
  return 3
}
# band-gpu's Python: torch pinned (TORCH_VERSION) from the PyTorch index for the toolkit's CUDA (torch's JIT build needs the
# wheel's CUDA major to be nvcc's): 2.14.0 cu130 for CUDA 13, 2.7.1 cu128 for 12.8+, 2.7.1 cu126 below
gpu_env() {
  local home major minor
  home=$(cuda_home) || { echo "no CUDA 13 or CUDA 12 toolkit for this GPU (at least 12.8 on Blackwell) is installed"; return 3; }
  export CUDA_HOME=$home PATH=$home/bin:$PATH
  read -r major minor <<< "$(nvcc_release "$CUDA_HOME")"
  if [ "$major" = 13 ]; then
    TORCH_CUDA=${TORCH_CUDA:-cu130}; TORCH_VERSION=${TORCH_VERSION:-2.14.0}
  else
    TORCH_CUDA=${TORCH_CUDA:-$([ "${minor:-8}" -ge 8 ] && echo cu128 || echo cu126)}; TORCH_VERSION=${TORCH_VERSION:-2.7.1}
  fi
  local venv=$W/gpu312-torch$TORCH_VERSION-$TORCH_CUDA
  PY=$venv/bin/python
  [ -x "$PY" ] || { uv venv -q "$venv" --python 3.12 && uv pip install -q --python "$PY" numpy ninja \
    && uv pip install -q --python "$PY" "torch==$TORCH_VERSION" --index-url "https://download.pytorch.org/whl/$TORCH_CUDA"; } || return 3
  "$PY" -c 'import torch; assert torch.cuda.is_available(), "torch sees no CUDA device"; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))' || return 3
  export PATH=$(dirname "$PY"):$PATH
}
# the host facts a result is re-derived with, and two distinct physical cores (not SMT siblings) for the verifier and responder
record_host() {
  clocks before
  nvcc --version > "$O/nvcc.txt" 2>&1 || true
  lscpu -e > "$O/lscpu.txt" 2>&1 || true
  uv pip freeze --python "$PY" > "$O/pip-freeze.txt" 2>/dev/null || true
  if [ -n "${NVME_DIR:-}" ]; then
    mkdir -p "$NVME_DIR"; findmnt -T "$NVME_DIR" > "$O/nvme-mount.txt" 2>&1; lsblk -o NAME,MODEL,TRAN,ROTA,SIZE > "$O/lsblk.txt" 2>&1
  fi
  local v r
  read -r v r <<< "$(lscpu -p=CPU,CORE 2>/dev/null | grep -v '^#' | awk -F, '!seen[$2]++ {print $1}' | sed -n '3p;4p' | tr '\n' ' ')"
  VCPU=${VCPU:-${v:-3}}; RCPU=${RCPU:-${r:-2}}
}
smi --query-gpu=index,uuid,name,driver_version,clocks.max.sm --format=csv,noheader || true
nproc; cat /sys/fs/cgroup/cpu.max 2>/dev/null || true
command -v uv >/dev/null || { python3 -m pip install -q uv || exit 3; }
if [ -n "${GATE_ROLE:-}" ]; then
  gate=(--role "$GATE_ROLE" --out "$O/placement-gate.json" ${GATE_CORES:+--cores "$GATE_CORES"})
  for p in ${GATE_PEERS:-}; do f=${p#*=}; [ "${f#/}" = "$f" ] && f=$I/$f; gate+=(--peer "${p%%=*}=$f"); done
  for r in ${GATE_ABSENT:-}; do gate+=(--absent "$r"); done
  python3 benchmarks/numerical/python/verity_numerical/bench/placement.py gate "${gate[@]}" > /dev/null \
    || { echo "the placement gate refused this run as $GATE_ROLE (placement-gate.json)"; exit 4; }
fi
if [ "$IMPL" = vllm-gpu ]; then
  [ -f "$I/impl.tar" ] || { echo "vllm-gpu needs --send impl.tar"; exit 2; }
  SHA=$(sha256sum "$I/impl.tar" | cut -c1-16)
  if [ "$(cat "$W/impl.sha" 2>/dev/null)" != "$SHA" ]; then
    rm -rf "$W/impl" && mkdir -p "$W/impl" && tar -xf "$I/impl.tar" -C "$W/impl" && echo "$SHA" > "$W/impl.sha" || exit 3
  fi
  (cd "$W/impl/integrations/vllm" && bash verity_vllm/ops/pod_bootstrap.sh --cpu --cases QWEN05 --out "$O/bootstrap") \
    > "$O/bootstrap.log" 2>&1 || { tail -30 "$O/bootstrap.log"; exit 3; }
  PY=${VENV:-/workspace/venv312}/bin/python
  export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} PATH=/usr/local/cuda/bin:$PATH
  # torch cu129 needs a CUDA 12.9 driver; below R575 a datacenter GPU runs it through NVIDIA's forward-compatibility libraries
  DRV=$(smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
  if [ "$DRV" -lt 575 ]; then
    if [ ! -e /usr/local/cuda-12.9/compat/libcuda.so.1 ]; then
      . /etc/os-release; CUDA_REPO=ubuntu${VERSION_ID/./}
      dpkg -s cuda-keyring >/dev/null 2>&1 || { wget -q https://developer.download.nvidia.com/compute/cuda/repos/$CUDA_REPO/x86_64/cuda-keyring_1.1-1_all.deb && dpkg -i cuda-keyring_1.1-1_all.deb; } || exit 3
      export DEBIAN_FRONTEND=noninteractive
      apt-get update -qq && apt-get install -y -qq --no-install-recommends cuda-compat-12-9 >/dev/null || exit 3
    fi
    export LD_LIBRARY_PATH=/usr/local/cuda-12.9/compat:${LD_LIBRARY_PATH:-}
  fi
  "$PY" -c 'import torch; assert torch.cuda.is_available(), "torch sees no CUDA device"; print(torch.__version__, torch.cuda.get_device_name(0))' || exit 3
  "$PY" -c 'import ninja' 2>/dev/null || uv pip install -q --python "$PY" ninja || exit 3   # torch's JIT build of pous_native
  export PATH=$(dirname "$PY"):$PATH
  "$PY" benchmarks/pous/bench.py --scheme "$SCHEME" --impl vllm-gpu --impl-tree "$W/impl" --impl-commit "${IMPL_COMMIT:-unknown}" \
    --build-dir "$POUS_BUILD" --python "$PY" --weight-set "$WS" --hardware "$HW" --out "$O/bench" --audits "$AUDITS"
elif [ "$IMPL" = band-gpu ]; then
  gpu_env || exit 3
  record_host
  "$PY" benchmarks/pous/bench.py --scheme "$SCHEME" --impl band-gpu --build-dir "$POUS_BUILD" --impl-commit "${RESEARCH_SOURCE_COMMIT:-}" \
    --weight-set "$WS" --hardware "$HW" --out "$O/bench" --audits "$AUDITS" --workers "$WORKERS" --verifier-cpu "$VCPU" \
    --responder-cpu "$RCPU" ${NVME_DIR:+--nvme-dir "$NVME_DIR"} ${MAX_SEGMENTS:+--max-segments "$MAX_SEGMENTS"} $GO \
    ${BAND_D:+--band-d "$BAND_D"} ${KERNELS:+--kernels "$KERNELS"}
  RC=$?
  clocks after
  exit $RC
elif [ "$IMPL" = p2-v1-gpu ] || [ "$IMPL" = p2-v3-gpu ] || [ "$IMPL" = p2-v3-mc-gpu ] || [ "$IMPL" = side-by-side ]; then
  gpu_env || exit 3
  record_host
  # P2 v1's library (benchmarks/pous/p2_v1): CGBN at p2_gpu's pinned commit, gmp.h for CGBN's host pass (the system's, else
  # gmpy2's wheel's in band-gpu's venv: nothing is installed on the system), the device's arch
  ARCH=${ARCH:-sm_$(smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)}
  GMPINC=
  if ! echo '#include <gmp.h>' | cpp >/dev/null 2>&1; then
    uv pip install -q --python "$PY" gmpy2 || exit 3
    GMPINC=$("$PY" -c 'import gmpy2, os; print(os.path.dirname(gmpy2.__file__))') || exit 3
  fi
  CG=$W/CGBN
  [ -d "$CG/.git" ] || git clone -q https://github.com/NVlabs/CGBN "$CG" || exit 3
  git -C "$CG" checkout -q "$(cat benchmarks/pous/p2_gpu/CGBN.commit)" || exit 3
  NVCC_REL=$(nvcc_release "$CUDA_HOME" | tr ' ' .)
  [ -n "$NVCC_REL" ] || { echo "no nvcc under CUDA_HOME=$CUDA_HOME"; exit 3; }
  # keyed by the nvcc and the sources' bytes: a .so another toolkit or source built is never adopted, and every checkout of the
  # same sources adopts it rather than rebuilding (make goes by mtimes, and each run's source tree is a fresh checkout);
  # and by P2FAST, the Makefile's FAST when it is set: '-DP2_FAST -DP2_KEYS4 -DP2SQ_KARA' expands the v4 keys
  P2KEY=$({ cat benchmarks/pous/p2_v1/Makefile benchmarks/pous/p2_v1/*.cu benchmarks/pous/p2_v1/*.cuh benchmarks/pous/p2_gpu/{p2dec.cu,CGBN.commit}
            [ -z "${P2FAST+x}" ] || echo "FAST=$P2FAST"; } | sha256sum | cut -c1-12)
  P2LIB=$W/p2v1bin-$ARCH-cuda$NVCC_REL-$P2KEY
  if [ -f "$P2LIB/libp2v1.so" ]; then echo "adopted $P2LIB/libp2v1.so" > "$O/p2v1-make.log"
  else make -C benchmarks/pous/p2_v1 NVCC="$CUDA_HOME/bin/nvcc" CGBN="$CG" BIN="$P2LIB" ARCH="$ARCH" ${GMPINC:+"INC=-I $GMPINC"} \
    ${P2FAST+"FAST=$P2FAST"} > "$O/p2v1-make.log" 2>&1 || { tail -20 "$O/p2v1-make.log"; exit 3; }
  fi
  cp "$P2LIB"/*.ptxas "$O/" 2>/dev/null
  arm() {  # arm IMPL OUT [sub]: one bench.py run of that implementation into OUT; a sub-run writes no result.json of its own
    ${3:+env -u RESEARCH_RUN_DIR} "$PY" ${BENCH_HOST:-} benchmarks/pous/bench.py --scheme "$SCHEME" --impl "$1" --build-dir "$POUS_BUILD" --p2-lib "$P2LIB" \
      --impl-commit "${RESEARCH_SOURCE_COMMIT:-}" --weight-set "$WS" --hardware "$HW" --out "$2" --audits "$AUDITS" \
      --workers "$WORKERS" --verifier-cpu "$VCPU" --responder-cpu "$RCPU" ${NVME_DIR:+--nvme-dir "$NVME_DIR"} \
      ${MAX_SEGMENTS:+--max-segments "$MAX_SEGMENTS"} ${TOKENS:+--tokens "$TOKENS"} ${LOAD_TOKENS:+--load-tokens "$LOAD_TOKENS"} $GO \
      $([ "$1" = band-gpu ] && [ -n "${BAND_D:-}" ] && echo --band-d "$BAND_D") \
      $([ "$1" != band-gpu ] && [ -n "${P2_ENCODE_BLOCKS:-}" ] && echo --p2-encode-blocks "$P2_ENCODE_BLOCKS") \
      $([ "$1" != band-gpu ] && [ -n "${P2_C_CACHE:-}" ] && echo --p2-c-cache "$W/p2-c-cache") \
      $([ "$1" = p2-v3-mc-gpu ] && [ "${MC_SIGNER:-lms}" = lms ] && echo --mc-lms-key "$W/pous-mc-lms-n64") \
      $(case $1 in (p2-v3-gpu|p2-v3-mc-gpu) [ -n "${BANDWIDTH_SALT:-}" ] && echo --bandwidth-salt "$BANDWIDTH_SALT" ;; esac)
  }
  if [ "$IMPL" = side-by-side ]; then
    mkdir -p "$O/bench/band" "$O/bench/p2"
    arm band-gpu "$O/bench/band" sub; RB=$?
    clocks between
    arm "$P2_ARM" "$O/bench/p2" sub; RP=$?
    if [ -n "$GO" ]; then   # gates only: no measurements to table
      RC=0
    else
      "$PY" benchmarks/pous/side_by_side.py "$O/bench/band/bench-result.json" "$O/bench/p2/bench-result.json" \
        --out "$O/bench/side-by-side.json" --markdown "$O/bench/side-by-side.md"
      RC=$?
    fi
    [ $RB -ne 0 ] || [ $RP -ne 0 ] && RC=1
  else
    arm "$IMPL" "$O/bench"; RC=$?
  fi
  clocks after
  exit $RC
elif [ "$IMPL" = p2-gpu ]; then
  # P2's kernels (benchmarks/pous/p2_gpu): CGBN at its pinned commit, libgmp for CGBN's host pass, nvcc for the device's arch
  export PATH=/usr/local/cuda/bin:$PATH
  ARCH=${ARCH:-sm_$(smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)}
  echo '#include <gmp.h>' | cpp >/dev/null 2>&1 || { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libgmp-dev >/dev/null; } || exit 3
  CG=$W/CGBN
  [ -d "$CG/.git" ] || git clone -q https://github.com/NVlabs/CGBN "$CG" || exit 3
  git -C "$CG" checkout -q "$(cat benchmarks/pous/p2_gpu/CGBN.commit)" || exit 3
  BIN=$W/p2bin-$ARCH
  make -C benchmarks/pous/p2_gpu CGBN="$CG" BIN="$BIN" ARCH="$ARCH" -j2 > "$O/p2-make.log" 2>&1 || { tail -20 "$O/p2-make.log"; exit 3; }
  cp "$BIN"/*.ptxas "$O/" 2>/dev/null
  PY=$W/ref312/bin/python
  [ -x "$PY" ] || { uv venv -q "$W/ref312" --python 3.12 && uv pip install -q --python "$PY" numpy; } || exit 3
  "$PY" benchmarks/pous/bench.py --scheme "$SCHEME" --impl p2-gpu --p2-bin "$BIN" --scratch "$W/scratch" \
    --impl-commit "${RESEARCH_SOURCE_COMMIT:-}" --weight-set "$WS" --hardware "$HW" --out "$O/bench" --audits "$AUDITS" \
    ${MAX_SEGMENTS:+--max-segments "$MAX_SEGMENTS"}
else
  PY=$W/ref312/bin/python
  [ -x "$PY" ] || { uv venv -q "$W/ref312" --python 3.12 && uv pip install -q --python "$PY" numpy; } || exit 3
  "$PY" benchmarks/pous/bench.py --scheme "$SCHEME" --impl reference --impl-commit "${RESEARCH_SOURCE_COMMIT:-}" \
    --weight-set "$WS" --hardware "$HW" --out "$O/bench" --audits "$AUDITS" ${MAX_SEGMENTS:+--max-segments "$MAX_SEGMENTS"}
fi
