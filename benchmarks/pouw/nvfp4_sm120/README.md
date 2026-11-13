# nvfp4_sm120: the plain NVFP4 and FP8 GEMMs on the RTX PRO 6000

A plain GEMM on sm_120a over operands that carry one scale a scale group, with no proof of work in it, written so that a
Pearl-C4 or Pearl-C kernel is timed against a baseline on the same mainloop. `mainloop_nvf4.cuh` computes one tile of
C = A Bᵀ as the chain of tensor-core atoms the silicon model states (`BLACKWELL_SM120_NVF4`; in its FP8 mode
`BLACKWELL_SM120_E4M3_M16N8K32`), so its words are the model's whatever the schedule. Operands and outputs are in the
layout CUTLASS's and cuBLASLt's GEMMs take.

| File | What it is |
| --- | --- |
| `mainloop_nvf4.cuh` | the TMA-fed, warp-specialized mainloop, NVFP4 and FP8 |
| `nvf4_plain.cu` | the GEMMs on it as entries of the GPU harness's baseline registry; `../harness/native/build.sh` builds it into the harness's library |
| `nvf4_bench.cu`, `nvf4_bench.py` | the mainloop's own bench on one card: every kernel gated word for word against an independent reference and the silicon model, then timed beside CUTLASS's and cuBLASLt's |
| `fp8_bench.cu`, `fp8_bench.py` | the same for the mainloop's FP8 mode, with the model of another card as the gate's negative control |

## Running it

The baselines run inside the harness (`../harness/`), which gates and times them with every other arm; a number of
record is the harness's. The two benches are for work on the mainloop itself. Each is the command of a `research run`
on one leased RTX PRO 6000, from a cubin built off the card with the `nvcc` line in the script's docstring:

~~~text
python3 nvf4_bench.py CUBIN
python3 fp8_bench.py CUBIN
python3 fp8_bench.py --model-cache DIR    # no GPU: the model's words for every gated shape
~~~

A bench refuses any GPU but the expected one (exit 3), gates every kernel at every shape before it times any (exit 4
when a gate fails), and prints one JSON object a line with a summary on stderr. Its knobs are environment variables,
listed in its docstring. Its timing is diagnostic.

## Who reads it

The harness reads `nvf4_plain.cu`: its build compiles it, with the mainloop, into the harness's library, and
`../tests/test_harness.py` reads its text for its registry entries. `fp8_bench.py` imports `nvf4_bench.py` and builds
on its bench; nothing else imports either, and no test runs them, since each needs the card. Nothing here commits,
hashes or checks a proof.
