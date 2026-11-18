#!/bin/bash
# vcf_job.sh: `verity bench memory --local` on one card of a VCF Slurm node that has a driver and no CUDA toolkit, in two
# jobs of at most 30 minutes, each submitted as the job script of infra/vcf/vcf-job-run.sh (sbatch, one GPU).
#
#   STAGE=setup    builds and keeps under KEEP (default $HOME/verity-h100), each step once, the step's mark
#                  KEEP/done/STEP saying it holds: the tree at the commit, uv and the tree's environment, nvcc and ninja from
#                  NVIDIA's pip packages (KEEP/cuda), the session's kernels built with run.sh's flags (session.so, kv.so),
#                  the Lean verifier (elan under KEEP/home, `lake build poub-verifier`), a vLLM venv (KEEP/vllm) and the
#                  pinned checkpoint (KEEP/work/hf). A second setup job takes up where the first stopped.
#   STAGE=session  `verity bench memory --local` from what setup kept, as a trial (VERITY_BENCH_TRIAL=1: no metric's
#                  line), its records copied to result/session/.
#
# The tree is the payload's (SEND= infra/vcf/payload.sh's archive, in the job's directory) or else VERITY_COMMIT's tarball
# from GitHub; the commit is handed to the bench as RESEARCH_SOURCE_SHA, since the tree has no .git. The last line printed
# says how far the job got and what refused it. Logs go to result/ in the job's directory, which vcf-job-run.sh copies back.
# Variables: STAGE, KEEP, VERITY_COMMIT, VLLM_VERSION (0.34.0), BUDGET_MIN (27, the job's own deadline inside Slurm's 30),
# CXX_FROM_CONDA (set: conda-forge's g++ even where the node has one), NODE_ARGS and ENGINE_ARGS (passed to --local).
set -u -o pipefail
STAGE=${STAGE:-}
case $STAGE in setup | session) ;; *) echo "vcf_job: STAGE is setup or session, not '${STAGE}'"; exit 2 ;; esac
J=${SLURM_SUBMIT_DIR:-$PWD}
KEEP=${KEEP:-$HOME/verity-h100}
mkdir -p "$J/result" "$KEEP/done" "$KEEP/bin" || { echo "vcf_job $STAGE: got nowhere: $KEEP or $J/result can't be made"; exit 3; }
DEADLINE=$(( $(date +%s) + ${BUDGET_MIN:-27} * 60 ))
REACHED="nothing" WHY=""
LEAN_PKG=verity/protocols/resource_accounting/memory/poub/verifier/lean
CU_PKGS=("nvidia-cuda-nvcc==13.0.88" "nvidia-cuda-runtime==13.0.96" "nvidia-cuda-crt==13.0.88" "nvidia-nvvm==13.0.88"
         "nvidia-cuda-cccl==13.0.85" ninja)
export UV_CACHE_DIR=$KEEP/uv-cache UV_PYTHON_INSTALL_DIR=$KEEP/python PATH=$KEEP/bin:$PATH
last() { echo "vcf_job $STAGE: reached ${REACHED}${WHY:+; refused: $WHY}"; }
trap last EXIT
left() { echo $(( DEADLINE - $(date +%s) > 30 ? DEADLINE - $(date +%s) : 30 )); }
say() { echo "$(date -u +%H:%M:%SZ) $*"; }
fail() { WHY=$1; exit "${2:-1}"; }
got() { REACHED=$([ "$REACHED" = nothing ] && echo "$1" || echo "$REACHED, $1"); say "reached $1"; }
have() { [ -f "$KEEP/done/$1" ]; }

{
  echo "node ${SLURMD_NODENAME:-$(hostname -s)}, job ${SLURM_JOB_ID:-none}, gpus ${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-none}}, $(nproc) CPUs"
  nvidia-smi --query-gpu=index,name,compute_cap,memory.total,driver_version --format=csv,noheader 2>&1
  for c in g++ gcc cc python3 git curl; do printf '%s: %s\n' "$c" "$(command -v "$c" || echo absent)"; done
} > "$J/result/site.txt" 2>&1
cat "$J/result/site.txt"

# the tree
if [ -f "$J/payload.tar.gz" ]; then
  tar -xzf "$J/payload.tar.gz" -C "$J" request.sh || fail "the payload holds no request.sh"
  # shellcheck source=/dev/null
  . "$J/request.sh"
  COMMIT=$VCF_COMMIT
else
  COMMIT=${VERITY_COMMIT:-}
  [[ $COMMIT =~ ^[0-9a-f]{40}$ ]] || fail "no payload in $J and VERITY_COMMIT is not a full commit id" 2
fi
T=$KEEP/tree
if [ "$(cat "$KEEP/tree.commit" 2> /dev/null)" != "$COMMIT" ]; then
  rm -rf "$T.new" && mkdir -p "$T.new" || fail "$T.new can't be made"
  if [ -f "$J/payload.tar.gz" ]; then
    tar -xzf "$J/payload.tar.gz" -C "$T.new" --strip-components=1 tree || fail "the payload's tree doesn't unpack"
  else
    curl -fsSL -m 300 "https://codeload.github.com/compute-verification/verity/tar.gz/$COMMIT" |
      tar -xz -C "$T.new" --strip-components=1 || fail "GitHub's tarball of $COMMIT can't be fetched from here"
  fi
  rm -rf "$T" && mv "$T.new" "$T" && echo "$COMMIT" > "$KEEP/tree.commit" && rm -f "$KEEP/done/tree-env" "$KEEP/done/lean" ||
    fail "$T can't be replaced"
fi
got "the tree at $COMMIT"
cd "$T" || fail "no tree at $T"

# the allocated card: nvidia-smi's index of it, which the bench's GPU and the lease's allowed list take
if [ "$(nvidia-smi -L 2> /dev/null | wc -l)" -gt 1 ] && [[ ${CUDA_VISIBLE_DEVICES:-} =~ ^[0-9]+$ ]]; then
  CARD=$CUDA_VISIBLE_DEVICES
else
  CARD=0
fi
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i "$CARD" 2> /dev/null | head -1)
CC=${CC:-9.0}
SM=${CC/./}; [ "${CC%%.*}" -ge 9 ] && SM+=a

if [ "$STAGE" = setup ]; then
  command -v uv > /dev/null || { curl -LsSf -m 120 https://astral.sh/uv/install.sh |
    env UV_UNMANAGED_INSTALL="$KEEP/bin" sh > /dev/null || fail "uv can't be installed from astral.sh"; }
  say "uv $(uv --version)"

  # the long steps run beside the short ones: the Lean verifier, the vLLM venv and the tree's environment
  (
    have lean || {
      HOME=$KEEP/home timeout "$(left)" bash tools/verity/lean/setup.sh "$LEAN_PKG" &&
        (cd "$LEAN_PKG" && PATH=$KEEP/home/.elan/bin:$PATH timeout "$(left)" lake build poub-verifier) &&
        [ -x "$LEAN_PKG/.lake/build/bin/poub-verifier" ] && touch "$KEEP/done/lean"
    }
  ) > "$J/result/lean.log" 2>&1 &
  LEAN=$!
  (
    have vllm || {
      { [ -x "$KEEP/vllm/bin/python" ] || uv venv -q --python 3.12 "$KEEP/vllm"; } &&
        timeout "$(left)" uv pip install --python "$KEEP/vllm/bin/python" "vllm==${VLLM_VERSION:-0.34.0}" &&
        touch "$KEEP/done/vllm"
    }
  ) > "$J/result/vllm.log" 2>&1 &
  VLLM=$!
  (
    have tree-env || { timeout "$(left)" uv sync -q --all-packages --extra torch-cpu && touch "$KEEP/done/tree-env"; }
  ) > "$J/result/tree-env.log" 2>&1 &
  TREE=$!

  if ! have nvcc; then
    { [ -x "$KEEP/cuda/bin/python" ] || uv venv -q --python 3.12 "$KEEP/cuda"; } &&
      uv pip install -q --python "$KEEP/cuda/bin/python" "${CU_PKGS[@]}" > "$J/result/cuda.log" 2>&1 ||
      fail "NVIDIA's nvcc packages can't be installed ($(tail -1 "$J/result/cuda.log"))"
  fi
  CUDA_HOME=$KEEP/cuda/lib/python3.12/site-packages/nvidia/cu13
  [ -x "$CUDA_HOME/bin/nvcc" ] || fail "the nvcc package put no bin/nvcc under $CUDA_HOME"
  [ -e "$CUDA_HOME/lib64" ] || ln -s lib "$CUDA_HOME/lib64"
  if [ -n "${CXX_FROM_CONDA:-}" ] || ! command -v g++ > /dev/null; then
    # no host compiler (or CXX_FROM_CONDA asks): conda-forge's g++ through micromamba, both without root
    if [ ! -x "$KEEP/gcc/bin/x86_64-conda-linux-gnu-g++" ]; then
      curl -fsSL -m 120 https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$KEEP" bin/micromamba &&
        MAMBA_ROOT_PREFIX=$KEEP/mamba timeout "$(left)" "$KEEP/bin/micromamba" create -y -q -p "$KEEP/gcc" -c conda-forge \
          "gxx_linux-64=13" > "$J/result/gcc.log" 2>&1 || fail "no g++ here, and conda-forge's can't be installed ($(tail -1 "$J/result/gcc.log"))"
    fi
    mkdir -p "$KEEP/cc-bin"
    for n in cc gcc; do ln -sf "$KEEP/gcc/bin/x86_64-conda-linux-gnu-gcc" "$KEEP/cc-bin/$n"; done
    for n in c++ g++; do ln -sf "$KEEP/gcc/bin/x86_64-conda-linux-gnu-g++" "$KEEP/cc-bin/$n"; done
    echo 'export PATH=$KEEP/cc-bin:$PATH' > "$KEEP/cc.sh"
    say "g++ from conda-forge: $("$KEEP/cc-bin/g++" --version | head -1)"
  fi
  # shellcheck source=/dev/null
  [ -f "$KEEP/cc.sh" ] && . "$KEEP/cc.sh"
  export PATH=$CUDA_HOME/bin:$KEEP/cuda/bin:$PATH
  "$CUDA_HOME/bin/nvcc" --version | tail -2 || fail "nvcc doesn't run"
  touch "$KEEP/done/nvcc"; got "nvcc $("$CUDA_HOME/bin/nvcc" --version | sed -n 's/.*release \([0-9.]*\),.*/\1/p') and ninja $(ninja --version)"

  # run.sh's own flags and sources (its node-run build), for this card's sm
  FLAGS=(-O3 -gencode "arch=compute_$SM,code=sm_$SM" -std=c++17 -Xcompiler -fPIC -shared -Xptxas -v)
  K=$KEEP/kernels/$COMMIT; mkdir -p "$K"
  nvcc "${FLAGS[@]}" -o "$K/session.so" verity/protocols/resource_accounting/memory/poub/prover/poub_session/session.cu > "$J/result/session.ptxas" 2>&1 ||
    fail "nvcc failed on session.cu for sm_$SM ($(grep -m1 -i error "$J/result/session.ptxas"))"
  nvcc "${FLAGS[@]}" -DKV_FIELD=1024 -DKV_MASK=1 -o "$K/kv.so" verity/protocols/resource_accounting/memory/poub/prover/word_codec/kv.cu > "$J/result/kv.ptxas" 2>&1 ||
    fail "nvcc failed on kv.cu for sm_$SM ($(grep -m1 -i error "$J/result/kv.ptxas"))"
  (cd "$K" && sha256sum session.so kv.so | tee "$J/result/kernels.sha256")
  got "session.so and kv.so built for sm_$SM"

  if ! have checkpoint; then
    wait "$VLLM"
    have vllm || fail "the vLLM venv didn't build ($(tail -1 "$J/result/vllm.log"))"
    PIN=verity/protocols/resource_accounting/compute/pouw/verifier/lean/check/models.json
    HF_HOME=$KEEP/work/hf HF_HUB_OFFLINE=0 timeout "$(left)" "$KEEP/vllm/bin/python" -c '
import json, sys; from huggingface_hub import snapshot_download as d
p = json.load(open(sys.argv[1]))["LLAMA32_1B_BF16"]
print(d(p["repo"], revision=p["revision"], allow_patterns=sorted(p["files"]), max_workers=8))' "$PIN" > "$J/result/checkpoint.log" 2>&1 ||
      fail "the checkpoint can't be fetched from Hugging Face ($(tail -1 "$J/result/checkpoint.log"))"
    touch "$KEEP/done/checkpoint"
  fi
  got "the checkpoint"
  wait "$TREE"; have tree-env || fail "the tree's environment didn't sync ($(tail -1 "$J/result/tree-env.log"))"
  wait "$VLLM"; have vllm || fail "the vLLM venv didn't build ($(tail -1 "$J/result/vllm.log"))"
  got "the tree's environment" &&
    got "$("$KEEP/vllm/bin/python" -c 'import vllm, torch; print("vllm", vllm.__version__, "torch", torch.__version__, "for CUDA", torch.version.cuda)')"
  wait "$LEAN"; have lean || fail "the Lean verifier didn't build ($(tail -1 "$J/result/lean.log"))"
  got "the Lean verifier ($LEAN_PKG/.lake/build/bin/poub-verifier)" && got "setup whole: STAGE=session runs next"
  exit 0
fi

# STAGE=session
for s in tree-env nvcc lean vllm checkpoint; do have "$s" || fail "setup hasn't kept $s at $COMMIT: run STAGE=setup again" 3; done
got "what setup kept at $COMMIT"
CUDA_HOME=$KEEP/cuda/lib/python3.12/site-packages/nvidia/cu13
# shellcheck source=/dev/null
[ -f "$KEEP/cc.sh" ] && . "$KEEP/cc.sh"
mkdir -p "$KEEP/work"
echo "$CARD" > "$KEEP/work/allowed-gpus"
export CUDA_HOME PATH=$CUDA_HOME/bin:$KEEP/cuda/bin:$PATH VLLM_PY=$KEEP/vllm/bin/python LOCAL_WORK=$KEEP/work \
  LEAN_EXE=$T/$LEAN_PKG/.lake/build/bin/poub-verifier RESEARCH_SOURCE_SHA=$COMMIT VERITY_BENCH_TRIAL=1 GPU=$CARD \
  GPU_LEASE_ALLOWED_FILE=$KEEP/work/allowed-gpus
say "verity bench memory --local on card $CARD (sm_$SM), $(( $(left) / 60 )) min left"
timeout --signal=INT --kill-after=60 "$(left)" .venv/bin/verity bench memory --local > "$J/result/bench.out" 2> "$J/result/bench.err"
rc=$?
cat "$J/result/bench.out"; tail -40 "$J/result/bench.err"
held=$(grep -A1 "its records go to the directory on the next line" "$J/result/bench.err" | tail -1 | sed 's/^ *//')
[ -n "$held" ] && [ -d "$held" ] && cp -r "$held" "$J/result/session" 2> /dev/null
said=$(grep -m1 '^memory: ' "$J/result/bench.out" "$J/result/bench.err" | head -1 | cut -d: -f2- | cut -c1-400)
case $rc in
  0) got "the session, accepted, with a share:$said" ;;
  124 | 130 | 137) fail "the session ran past the job's ${BUDGET_MIN:-27} min (rc $rc): its records so far are in result/session" ;;
  *) got "the session (verity bench memory exited $rc)"; WHY=${said# } ;;
esac
exit "$rc"
