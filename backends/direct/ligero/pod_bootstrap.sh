#!/usr/bin/env bash
# Set up a GPU pod for B-Ligero work (RTX 4090 / 5090, L40S, A100, H100).  Idempotent: re-running after a failure is the fix loop.
#
#   research pods sync vy-foo                     # the worktree -> /workspace/src (+ .research-source.json)
#   research pods ssh vy-foo -- 'RELS=fp8-ada bash /workspace/src/backends/direct/ligero/pod_bootstrap.sh 2>&1 | tee /workspace/bootstrap.log'
#
# Leaves on the pod:
#   /workspace/venv312        Python 3.12 + torch (picked by compute capability: >= 10.0 needs 2.8.0+cu128, else 2.6.0+cu124),
#                             numpy, cupy-cuda12x, blake3, pytest, pytest-timeout
#   /workspace/bin/ligero-verify   cargo --release build of $SRC/backends/ligero-verify (target dir /workspace/cargo-target)
#   /workspace/env.sh         `source` it in every later shell: PY, PATH, PYTHONPATH, CARGO_TARGET_DIR, LIGERO_VERIFY,
#                             LIGERO_INSTANCES_CACHE (run.py's --instances-cache default), and the thread pools sized to the
#                             cgroup CPU quota: OMP_NUM_THREADS = MKL_NUM_THREADS = OPENBLAS_NUM_THREADS = VY_CPU_THREADS =
#                             floor(quota) (torch's intra-op pool follows OMP_NUM_THREADS).  A 4090 pod has ~10 cores of quota
#                             while nproc says 96; pools sized to nproc made jobs crawl (kb/pods-4090.md)
#   /workspace/instances-cache     relchain instance sets for $RELS at each of $NS VUs (+ 64x64 tiles for $TILE64)
#
# Knobs (environment):
#   SRC=/workspace/src   RELS=fp8-ada,bf16-hopper   NS=4096   TILE64=fp8-ada   EXTRA_PIP="galois"   TORCH=2.8.0+cu128 (override)
#   BENCH_INSTANCES=1    the frozen bench-instances/v1 arrays (bf16-ampere / vu.py read them): built from the committed seeds into
#                        /workspace/bench-instances/v1, refused unless the rebuilt manifest's sha256 equals the committed one, then
#                        linked into $SRC/fixtures/bench-instances/v1 (gitignored names) so the default --root works
#   VERIFIER=tcp://HOST:PORT   probe the live verifier's path (RTT / throughput) at the end
#   SKIP_RUST=1          no cargo build (a pod that only proves)
#   HEALTH=0             skip the closing pod health check (pod_health.py, as `research pods health`; DEGRADED fails the bootstrap)
set -uo pipefail
export DEBIAN_FRONTEND=noninteractive
SRC=${SRC:-/workspace/src}
RELS=${RELS:-}
NS=${NS:-4096}
TILE64=${TILE64:-}
W=/workspace
PY=$W/venv312/bin/python
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:/usr/local/cuda/bin:$PATH"
export CARGO_TARGET_DIR=$W/cargo-target
export PYTHONPATH="$SRC/packages/verity/src:$SRC/backends/numerical/python:$SRC/tools/research/src:$SRC"
mkdir -p "$W/bin" "$W/instances-cache"
cpu_quota() {  # floor(cgroup CPU quota) in cores: v2 cpu.max, else v1 cfs_quota_us / cfs_period_us; nproc when unlimited
  local cg=${CGROUP:-/sys/fs/cgroup} q="" p=""
  if [ -r "$cg/cpu.max" ]; then read -r q p < "$cg/cpu.max"
  elif [ -r "$cg/cpu/cpu.cfs_quota_us" ]; then q=$(cat "$cg/cpu/cpu.cfs_quota_us"); p=$(cat "$cg/cpu/cpu.cfs_period_us"); fi
  case "$q" in ''|max|-1) nproc ;; *) q=$((q / p)); echo $((q > 0 ? q : 1)) ;; esac
}
THREADS=$(cpu_quota)
NP=$(nproc); [ "$NP" -gt 24 ] && NP=24; [ "$NP" -gt "$THREADS" ] && NP=$THREADS
fail=0
stage() { echo; echo "=== [$(date -u +%H:%M:%S)] $*"; }
check() { [ "$1" -eq 0 ] || { echo "FAILED: $2"; fail=1; }; }

stage "machine"
[ -d "$SRC/backends/direct/ligero" ] || { echo "no source tree at $SRC (research pods sync TARGET first)"; exit 2; }
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version,compute_cap,power.limit,clocks.max.sm --format=csv,noheader
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
nproc; free -g | head -2; df -h $W | tail -1; grep -m1 'model name' /proc/cpuinfo; cat /proc/loadavg
echo "cgroup CPU quota: $THREADS cores of $(nproc) -> OMP/MKL/OpenBLAS/torch threads = $THREADS, instance procs NP = $NP"
tr '\0' '\n' < /proc/1/environ 2>/dev/null | grep -E '^RUNPOD_(POD_ID|DC_ID|GPU_COUNT)=' || true
echo "source: $SRC"; cat "$SRC/.research-source.json" 2>/dev/null || echo "(no .research-source.json: not shipped by research pods sync)"

stage "rsync (the next research pods sync ships only what changed)"
command -v rsync >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq --no-install-recommends rsync >/dev/null; }
command -v rsync >/dev/null 2>&1; check $? "rsync"

if [ "${SKIP_RUST:-0}" != 1 ]; then
  stage "rustup stable + ligero-verify (background; joined at the end)"
  ( command -v cargo >/dev/null 2>&1 || curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable >/tmp/rustup.log 2>&1
    "$HOME/.cargo/bin/rustc" --version
    cd "$SRC/backends/ligero-verify" && cargo build --release 2>&1 | tail -3 && cp "$CARGO_TARGET_DIR/release/ligero-verify" $W/bin/ ) > /tmp/rust_build.log 2>&1 &
  RUST_PID=$!
fi

stage "uv + python 3.12 (/workspace/venv312)"
command -v uv >/dev/null 2>&1 || curl -fsSL https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
uv --version && uv python install 3.12 2>&1 | tail -1
[ -x "$PY" ] || uv venv $W/venv312 --python 3.12 2>&1 | tail -1
"$PY" --version; check $? "python 3.12 venv"

cat > $W/env.sh <<EOF
# written by backends/direct/ligero/pod_bootstrap.sh; source it in every shell on this pod
export PY=$PY
export PATH="$W/venv312/bin:\$HOME/.cargo/bin:\$HOME/.local/bin:/usr/local/cuda/bin:\$PATH"
export PYTHONPATH="$PYTHONPATH"
export CARGO_TARGET_DIR=$CARGO_TARGET_DIR
export LIGERO_VERIFY=$W/bin/ligero-verify
export LIGERO_INSTANCES_CACHE=$W/instances-cache
# the cgroup CPU quota ($THREADS cores; nproc says $(nproc)): size every pool to it
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS VY_CPU_THREADS=$THREADS
cd $SRC
EOF

MAJOR=${CC%%.*}
if [ -n "${TORCH:-}" ]; then T_VER=${TORCH%%+*}; T_CU=${TORCH##*+}
elif [ "${MAJOR:-0}" -ge 10 ]; then T_VER=2.8.0; T_CU=cu128
else T_VER=2.6.0; T_CU=cu124; fi
stage "torch $T_VER+$T_CU (compute capability $CC), numpy, cupy-cuda12x, blake3, pytest ${EXTRA_PIP:-}"
if ! "$PY" -c "import torch, sys; sys.exit(torch.__version__ != '$T_VER+$T_CU')" 2>/dev/null; then
  uv pip install --python "$PY" --index-url "https://download.pytorch.org/whl/$T_CU" "torch==$T_VER" 2>&1 | tail -2
fi
# shellcheck disable=SC2086
uv pip install --python "$PY" numpy cupy-cuda12x blake3 pytest pytest-timeout ${EXTRA_PIP:-} 2>&1 | tail -2
"$PY" - <<'PYEOF'
import torch, cupy
print("torch", torch.__version__, "cuda", torch.version.cuda, "|", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0),
      f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB")
x = torch.randn(1024, 1024, device="cuda"); assert float((x @ x).abs().sum()) > 0
assert int(cupy.arange(16, dtype=cupy.uint32).sum()) == 120
k = cupy.RawKernel(r'extern "C" __global__ void f(unsigned* x){ x[threadIdx.x] += 1; }', "f"); z = cupy.zeros(8, dtype=cupy.uint32); k((1,), (8,), (z,))
assert int(z.sum()) == 8
print("torch matmul, cupy, nvrtc: ok")
PYEOF
check $? "GPU stack"

stage "import smoke (the shipped tree)"
cd "$SRC" && LIGERO_GPU_STRICT=1 "$PY" - <<'PYEOF'
import backends.direct.ligero.run, backends.direct.ligero.live, backends.direct.ligero.relchain
from backends.direct.ligero import merkle
print("imports ok; GPU Merkle:", merkle.gpu_available(), getattr(merkle, "GPU_HASH_NAME", merkle.HASH_NAME))
assert merkle.gpu_available()
PYEOF
check $? "import smoke / GPU Merkle"

if [ "${BENCH_INSTANCES:-0}" = 1 ]; then
  stage "bench-instances/v1 (frozen set, from the committed seeds)"
  BI=$W/bench-instances/v1; FZ=$SRC/fixtures/bench-instances/v1
  [ -f "$BI/vu-k1536.x.u16" ] || "$PY" -m verity_numerical.bench.instances build --out "$BI" --seeds "$FZ/seeds" --procs "$NP" 2>&1 | tail -3
  "$PY" - "$BI" "$FZ" <<'PYEOF'
import json, sys
from pathlib import Path
from verity_numerical.bench.instances import sha256_file
bi, fz = Path(sys.argv[1]), Path(sys.argv[2])
man = json.loads((fz / "manifest.json").read_text())
built = {n: f for t in man["tiers"].values() for n, f in t.get("files", {}).items() if not f.get("committed")}
bad = [n for n, f in built.items() if not (bi / n).is_file() or sha256_file(bi / n) != f["sha256"]]
assert not bad, f"rebuilt arrays differ from the committed manifest's sha256 (do not measure on them): {bad}"
for n in built:
    dst = fz / n
    if dst.is_symlink() or not dst.exists():
        dst.unlink(missing_ok=True)
        dst.symlink_to(bi / n)
print(f"{len(built)} built arrays match the committed manifest; linked into {fz}")
PYEOF
  check $? "bench-instances/v1 arrays vs committed manifest"
fi

if [ -n "$RELS" ] || [ -n "$TILE64" ]; then
  stage "instance caches: ${RELS:-none} at n = $NS${TILE64:+; 64x64 tiles: $TILE64}"
  cd "$SRC" && RELS="$RELS" NS="$NS" TILE64="$TILE64" NP="$NP" OMP_NUM_THREADS=1 "$PY" - <<'PYEOF'
import os, time
from backends.direct.ligero.relchain import instances, tile_instances
from backends.direct.ligero.relations import relation
cache, procs = "/workspace/instances-cache", int(os.environ["NP"])
for name in filter(None, os.environ["RELS"].split(",")):
    for n in map(int, os.environ["NS"].split(",")):
        t0 = time.perf_counter(); d = instances(relation(name), n, procs=procs, cache=cache)
        print(f"{name} n={n}: {len(d)} VUs, {time.perf_counter() - t0:.1f}s", flush=True)
for name in filter(None, os.environ["TILE64"].split(",")):
    t0 = time.perf_counter(); d, _, _ = tile_instances(relation(name), 64, 64, procs=procs, cache=cache)
    print(f"{name} tile 64x64: {len(d)} VUs, {time.perf_counter() - t0:.1f}s", flush=True)
PYEOF
  check $? "instance caches"
fi

if [ -n "${VERIFIER:-}" ]; then
  stage "live verifier path: $VERIFIER"
  cd "$SRC" && "$PY" -m backends.direct.ligero.live probe --verifier "$VERIFIER" --mb 0.5 --repeat 6 2>&1 | tail -8
  check "${PIPESTATUS[0]}" "live probe"
fi

if [ "${SKIP_RUST:-0}" != 1 ]; then
  stage "rust build (join)"
  wait "$RUST_PID"; cat /tmp/rust_build.log
  [ -x $W/bin/ligero-verify ]; check $? "ligero-verify build"
  sha256sum $W/bin/ligero-verify 2>/dev/null
fi

stage "versions"
uv pip list --python "$PY" 2>/dev/null | grep -Ei "^(torch|triton|numpy|cupy-cuda12x|blake3|pytest|pytest-timeout) "
"$HOME/.cargo/bin/rustc" --version 2>/dev/null
df -h $W | tail -1

if [ "${HEALTH:-1}" = 1 ]; then
  stage "pod health (quota, GPU state, encode + matmul vs pod_health_ref.json; = research pods health)"
  cd "$SRC" && OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS "$PY" -m backends.direct.ligero.pod_health
  check $? "pod health (exit 3 = DEGRADED: replace this pod)"
fi
echo "next shells: source $W/env.sh"
if [ $fail -eq 0 ]; then echo BOOTSTRAP_OK; else echo BOOTSTRAP_FAILED; exit 1; fi
