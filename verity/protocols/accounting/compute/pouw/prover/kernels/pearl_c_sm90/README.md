# Pearl-C on sm_90a

The build of `pearl-c-h100-v1-h3` (G = 4, hashing format -h3) for the H100 and H200: `pearl_c_sm90.cu`'s main loop and
atoms on wgmma, a second binary for each word's inert-work counts, and `run.py` and `grouped.py`, the vLLM adapter's
served kernel on these cards. `build.sh` builds only with the nvcc sm_90a's pins name, and refuses a binary its
instruction gate or its numeric gate refuses. The subsystem is [`../README.md`](../README.md).

## Try it

Without a GPU, see which of GLM-5.3's linears the kernels take:

~~~sh
uv run python benchmarks/pouw/pearl_c_sm90/glm53.py
~~~

It prints one line for each linear one rank serves: its shape, the kernel vLLM serves it with, and either the
advantage ε at the H100 record, at both readings of the FP32 add's price, or why the kernels refuse it.

On the card: `bash benchmarks/pouw/pearl_c_sm90/ship.sh [OUT]` off it, then `python3 run.py [check.json]` in the built
tree (`benchmarks/pouw/pearl_c_sm90/h200.sh` does both under `research run`).
