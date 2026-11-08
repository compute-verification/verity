#!/usr/bin/env bash
# What a GLM-5.3 replay's bits can depend on besides its inputs and the all-reduce tree (which the all-reduce probe reads), written
# so that two nodes' records compare with `diff -r A B`: no serials, clocks or host names.
#   VENV=DIR [CK=DIR PINS=FILE] bash node_record.sh OUTDIR
# gpu.txt        each GPU's name, compute capability and memory, the driver, VBIOS, ECC and MIG modes; SMs, torch, its CUDA, cuDNN
#                and NCCL
# cuda.txt       nvcc's version (FlashInfer's and DeepGEMM's JIT build with it)
# packages.txt   the venv's distributions and versions
# libs.sha256    every shared object under the venv's site-packages
# caches/*.sha256  each JIT, autotune and compile cache the engine reads, file by file: TensorRT-LLM's kernels, vLLM's FlashInfer
#                autotune choices and torch.compile cache, FlashInfer's JIT, DeepGEMM's, Triton's, and the driver's PTX JIT cache
#                (empty when no kernel was compiled from PTX, so the driver's own compiler reached no bits)
# checkpoint.txt with CK and PINS (benchmarks/ppl/pins/glm-5.3.json): each pinned file's sha256 against the pin, OK or its digest
# topo.txt       the NVLink topology, which reaches the bits only through the tree
set -uo pipefail
O=${1:?usage: VENV=DIR bash node_record.sh OUTDIR}; mkdir -p "$O/caches"
PY=${VENV:?VENV: the venv the engine runs}/bin/python
nvidia-smi --query-gpu=index,name,compute_cap,memory.total,driver_version,vbios_version,ecc.mode.current,mig.mode.current \
  --format=csv > "$O/gpu.txt"
"$PY" -c 'import torch; p = torch.cuda.get_device_properties(0)
print(f"sms={p.multi_processor_count} torch={torch.__version__} cuda={torch.version.cuda} cudnn={torch.backends.cudnn.version()} nccl={torch.cuda.nccl.version()}")' >> "$O/gpu.txt"
NVCC=${CUDA_HOME:-/usr/local/cuda-12.9}/bin/nvcc
{ echo "$NVCC"; "$NVCC" --version 2>&1 | tail -2; } > "$O/cuda.txt"
"$PY" -c 'import importlib.metadata as m
print("\n".join(sorted({d.metadata["Name"] + "==" + d.version for d in m.distributions()})))' > "$O/packages.txt"
SP=$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
(cd "$SP" && find . -name '*.so*' -type f -print0 | xargs -0 -r -P 16 -n 32 sha256sum | sort -k2) > "$O/libs.sha256"
VC=${VLLM_CACHE_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/vllm}
for kv in "tensorrt_llm:$HOME/.tensorrt_llm/cache" "flashinfer_autotune:$VC/flashinfer_autotune_cache" \
          "torch_compile:$VC/torch_compile_cache" "flashinfer_jit:${FLASHINFER_WORKSPACE_BASE:-$HOME}/.cache/flashinfer" \
          "deep_gemm:${DG_JIT_CACHE_DIR:-$HOME/.deep_gemm}" "triton:${TRITON_CACHE_DIR:-$HOME/.triton/cache}" \
          "driver_ptx_jit:${CUDA_CACHE_PATH:-$HOME/.nv/ComputeCache}"; do
  d=${kv#*:}
  if [ -d "$d" ]; then (cd "$d" && find . -type f ! -name '*.lock' -print0 | xargs -0 -r -P 16 -n 64 sha256sum | sort -k2)
  else echo "absent $d"; fi > "$O/caches/${kv%%:*}.sha256"
done
if [ -n "${CK:-}" ] && [ -n "${PINS:-}" ]; then
  "$PY" - "$CK" "$PINS" > "$O/checkpoint.txt" <<'EOF'
import hashlib, json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ck, pins = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text())
def sha(name):
    h = hashlib.sha256()
    with open(ck / name, "rb") as f:
        while b := f.read(1 << 24):
            h.update(b)
    return h.hexdigest()
names = sorted(pins["files"])
with ThreadPoolExecutor(16) as ex:
    got = dict(zip(names, ex.map(lambda n: sha(n) if (ck / n).is_file() else "absent", names)))
print(f"{pins['repo']}@{pins['revision']}")
for n in names:
    print(("OK " if got[n] == pins["files"][n] else f"DIFF {got[n]} ") + n)
EOF
fi
nvidia-smi topo -m 2>&1 | sed 's/[[:space:]]*$//' > "$O/topo.txt"
echo "node_record: $O: $(grep -c . "$O/libs.sha256") libs, caches $(for f in "$O"/caches/*.sha256; do printf '%s=%s ' "$(basename "$f" .sha256)" "$(grep -vc '^absent' "$f")"; done)"
