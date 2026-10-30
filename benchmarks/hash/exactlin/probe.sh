#!/usr/bin/env bash
# What stock vLLM int8 runs on an RTX PRO 6000 host, off the GPU: the stack's versions, vLLM's int8 linear sources
# (scaled_int8_quant's wrapper, triton_scaled_mm, the int8 kernel classes) copied into $OUT/src, and the SASS of the
# per-token quantizer in vLLM's _C (dynamic_scaled_int8_quant*), so its division and rounding can be read off the binary.
#   RESEARCH_RUN_DIR=DIR bash benchmarks/hash/exactlin/probe.sh      (on that host, no GPU lease)
set -uo pipefail
OUT=${RESEARCH_RUN_DIR:?}/out; mkdir -p "$OUT/src"
PY=${PY:-/workspace/jobs/venv312/bin/python3}
CUDA=${CUDA:-/usr/local/cuda-13.0}
git rev-parse HEAD > "$OUT/commit.txt" 2>/dev/null
"$PY" - "$OUT" <<'EOF'
import importlib, inspect, json, sys, pathlib
out = pathlib.Path(sys.argv[1])
import torch, triton, vllm
doc = {"torch": torch.__version__, "torch_cuda": torch.version.cuda, "triton": triton.__version__, "vllm": vllm.__version__,
       "vllm_dir": str(pathlib.Path(vllm.__file__).parent)}
for mod in ["vllm._custom_ops",
            "vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm",
            "vllm.model_executor.layers.quantization.kernels.scaled_mm.triton",
            "vllm.model_executor.layers.quantization.kernels.scaled_mm.cutlass",
            "vllm.model_executor.layers.quantization.kernels.scaled_mm.ScaledMMLinearKernel",
            "vllm.model_executor.layers.quantization.kernels.scaled_mm",
            "vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8"]:
    try:
        m = importlib.import_module(mod)
        f = pathlib.Path(inspect.getsourcefile(m))
        (out / "src" / (mod + ".py")).write_text(f.read_text())
        doc[mod] = str(f)
    except Exception as e:  # noqa: BLE001
        doc[mod] = f"{type(e).__name__}: {e}"
so = sorted(pathlib.Path(vllm.__file__).parent.glob("_C*.so"))
doc["vllm_C"] = [str(p) for p in so]
print(json.dumps(doc, indent=1))
(out / "versions.json").write_text(json.dumps(doc, indent=1) + "\n")
EOF
SO=$("$PY" -c "import vllm, pathlib; print(sorted(pathlib.Path(vllm.__file__).parent.glob('_C*.so'))[0])")
"$CUDA/bin/cuobjdump" -lelf "$SO" > "$OUT/elf.txt" 2>&1
"$CUDA/bin/cuobjdump" -lelf "$SO" 2>/dev/null | head -40
for arch in sm_120 sm_120a sm_100 sm_90 sm_80; do
  "$CUDA/bin/cuobjdump" -sass -arch "$arch" "$SO" 2>/dev/null \
    | awk '/Function : .*dynamic_scaled_int8_quant/{p=1} /Function : /&&!/dynamic_scaled_int8_quant/{p=0} p' > "$OUT/sass-quant-$arch.txt"
  n=$(grep -c "Function :" "$OUT/sass-quant-$arch.txt")
  echo "probe: $arch dynamic_scaled_int8_quant functions $n, MUFU.RCP $(grep -c MUFU.RCP "$OUT/sass-quant-$arch.txt"), F2I $(grep -c F2I "$OUT/sass-quant-$arch.txt")"
done
nvidia-smi --query-gpu=index,name,clocks.sm,clocks.max.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used --format=csv | tee "$OUT/smi.csv"
command -v gpu-lease && gpu-lease status 2>&1 | head -30 | tee "$OUT/lease.txt"
"$CUDA/bin/nsys" profile --help 2>&1 | grep -A12 "capture-range-end" | head -20 | tee "$OUT/nsys-help.txt"
