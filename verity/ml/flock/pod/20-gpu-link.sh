#!/usr/bin/env bash
# flock-gpu-link: build flock-gpu-link (Flock b684b12 + flock-link + flock-gpu-link patches, the live crate, the CUDA chunk
# prover) and run it. Run from the verity tree (research run --source . --cwd source).
# env: MODE=all|build|selftest|serve|prove  VUS="8 64" (selftest) SESS_VUS=4096 RUNS=3 VERIFIER=host:port (prove; empty =
#      a loopback verifier process) LISTEN=0.0.0.0:7300 SESSIONS=N (serve) OPERATOR=lane  GPU=1 (0: CPU-only build)  SM=90
set -uxo pipefail
SRC=$(pwd); SM=${SM:-90}; MODE=${MODE:-all}; GPU=${GPU:-1}
W=${FLOCK_WORK:-/workspace}/flock-gpu-link; F=$W/flock; O=${RESEARCH_RUN_DIR:-/tmp/fgl}/out; mkdir -p $W $O
nproc; cat /sys/fs/cgroup/cpu.max 2>/dev/null; free -g | head -2
read -r Q PER < /sys/fs/cgroup/cpu.max 2>/dev/null || { read -r Q < /sys/fs/cgroup/cpu/cpu.cfs_quota_us && read -r PER < /sys/fs/cgroup/cpu/cpu.cfs_period_us; } 2>/dev/null || { Q=max; PER=100000; }; [ "$Q" = -1 ] && Q=max
TH=$([ "$Q" = max ] && nproc || echo $(( Q / PER ))); export RAYON_NUM_THREADS=$TH
export DEBIAN_FRONTEND=noninteractive
C13=${CUDA_TK:-/usr/local/cuda-13.3}   # CUDA_TK: a toolkit the pod's driver runs natively (no compat), e.g. an R550 host's 12.4
if [ "$GPU" = 1 ]; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv | tee $O/gpu.txt
  if [ ! -x $C13/bin/nvcc ]; then
    . /etc/os-release; REPO=ubuntu${VERSION_ID/./}
    dpkg -s cuda-keyring >/dev/null 2>&1 || { wget -q https://developer.download.nvidia.com/compute/cuda/repos/$REPO/x86_64/cuda-keyring_1.1-1_all.deb && dpkg -i cuda-keyring_1.1-1_all.deb; }
    apt-get update -qq
    apt-get install -y -qq --no-install-recommends cuda-nvcc-13-3 cuda-cudart-dev-13-3 >/dev/null || exit 1
  fi
  DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
  [ "$DRV" -lt 580 ] && [ -z "${CUDA_TK:-}" ] && { apt-get install -y -qq cuda-compat-13-3 >/dev/null || exit 1; }
  export PATH=$C13/bin:$PATH NVCC=$C13/bin/nvcc LD_LIBRARY_PATH=$C13/compat:$C13/lib64:${LD_LIBRARY_PATH:-}
fi
command -v cargo >/dev/null || [ -x $HOME/.cargo/bin/cargo ] || curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable >/dev/null
source $HOME/.cargo/env; rustc --version
cd $W; [ -d flock ] || git clone -q https://github.com/succinctlabs/flock
cd $F && git checkout -q b684b12 && git checkout -q -- . && rm -rf crates/flock-live cuda-ghash/prove_chunk.cuh
git apply $SRC/verity/ml/flock/flock-link-b684b12.patch && git apply $SRC/verity/ml/flock/flock-gpu-link-b684b12.patch || exit 1
cp -r $SRC/verity/ml/flock/live crates/flock-live
grep -q '"crates/flock-live"' Cargo.toml || sed -i 's|    "crates/flock-transcript",|    "crates/flock-transcript",\n    "crates/flock-live",|' Cargo.toml
FEAT=; if [ "$GPU" = 1 ]; then SM=$SM CUDA=$C13 python3 $SRC/verity/ml/flock/cuda_chunk_patch.py || exit 1; FEAT="--features gpu"; fi
cargo build --release -p flock-live $FEAT --bin flock-gpu-link --bin flock-pure-gpu -j $TH > $O/build.txt 2>&1 || { tail -40 $O/build.txt; exit 1; }
B=$F/target/release/flock-gpu-link; sha256sum $B $F/target/release/flock-pure-gpu | tee $O/binary.sha256
[ "$MODE" = build ] && exit 0
G=; [ "$GPU" = 1 ] && G=--gpu
if [ "$MODE" = all ] || [ "$MODE" = selftest ]; then
  for v in ${VUS:-8 64}; do
    $B selftest --vus $v > $O/selftest-cpu-$v.txt 2>&1; grep -h SELFTEST $O/selftest-cpu-$v.txt
    [ -n "$G" ] && { $B selftest --vus $v --gpu > $O/selftest-gpu-$v.txt 2>&1; grep -h SELFTEST $O/selftest-gpu-$v.txt; grep -h '"pass":false' $O/selftest-gpu-$v.txt | cut -c1-400; }
  done
fi
if [ "$MODE" = serve ]; then
  $B serve --listen ${LISTEN:-0.0.0.0:7300} --out $O/sessions --vus ${SESS_VUS:-4096} --seed ${SEED:-10} --operator ${OPERATOR:-unknown} \
    ${SESSIONS:+--sessions $SESSIONS} 2>&1 | tee $O/serve.txt
fi
if [ "$MODE" = all ] || [ "$MODE" = prove ]; then
  for v in ${SESS_VUS:-4096}; do
    ADDR=${VERIFIER:-}
    if [ -z "$ADDR" ]; then
      $B serve --listen 127.0.0.1:7301 --out $O/sessions-loop-$v --vus $v --seed ${SEED:-10} --operator loopback-selftest > $O/serve-loop-$v.log 2>&1 &
      SP=$!; ADDR=127.0.0.1:7301
      for i in $(seq 600); do grep -q SERVING $O/serve-loop-$v.log && break; sleep 1; done
    fi
    $B prove --verifier $ADDR --vus $v --seed ${SEED:-10} $G --runs ${RUNS:-3} > $O/prove-$v.txt 2>&1
    grep -h "LIVE\|GPUPHASE\|panicked" $O/prove-$v.txt | cut -c1-900
    [ -n "${SP:-}" ] && { kill $SP; wait $SP 2>/dev/null; unset SP; }
  done
fi
true
