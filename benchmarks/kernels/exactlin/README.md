# exactlin: the int8-exact linear as two Triton kernels

The vLLM adapter serves an int8 linear layer exactly, as two kernels a call, stock vLLM int8's count: a quantizer bit
for bit `int8_exact.quantize_rows`, and a GEMM whose epilogue is `int8_exact.dequantize`. This directory checks those
kernels and times them. Each is gated bit for bit against `int8_exact` and the int8 Definitions' replay evaluators,
then timed against stock vLLM int8, at kernel level and served.

The kernels are the adapter's own, `verity_vllm.engine.int8_fused`, and this directory holds no copy of them. Every
gate and bench here imports that module, so a gate's pass and a timing describe the kernels vLLM serves. They run on an
RTX PRO 6000 (sm_120), an H200 (sm_90) and a B200 (sm_100). Nothing a verifier runs reads this directory.

## Try it

On one leased GPU, the gate and then the kernel-level timing:

~~~text
RESEARCH_RUN_DIR=DIR GPU_LEASE_WHO=WHO bash benchmarks/kernels/exactlin/kbench.sh
~~~

It times the exact, stock and fused arms interleaved rep by rep, and writes under `$RESEARCH_RUN_DIR/out`. A failed
gate times nothing and exits 4.

The other launchers take the same lease and pass extra arguments to their script:

| Launcher | What it runs |
| --- | --- |
| `KSCRIPT=kgate.py` before `kbench.sh` | the gate alone |
| `served.sh` | the stock and exact arms served, through `benchmarks/hash/dither/sbench.py` |
| `share.sh` | the share of a served step that PoUW doesn't credit |

The kernels' interface is `int8_fused`'s docstring. [`DESIGN.md`](DESIGN.md) holds the interface as this directory
uses it, and what each script checks or times.
