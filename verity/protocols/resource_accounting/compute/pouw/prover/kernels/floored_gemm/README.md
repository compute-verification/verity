# The floored GEMM

`fgemm.cu` is the served GEMM on the exact int8 grid: the operands are noised a chunk of 32 columns at a time, the
tensor cores multiply the noised operands, and an exact peel returns the product of the codes. Later kernels build on
it: [`masked_gemm/`](../masked_gemm/README.md)'s `mgemm.cu` includes `fgemm.cu`, and
[`edited_gemm/`](../edited_gemm/README.md)'s `cgemm_ref.py` derives its receipts with `derive.py`, under `ref.py`'s keys. They are gated and
timed through `benchmarks/pouw/edited_gemm/crun.sh`.
