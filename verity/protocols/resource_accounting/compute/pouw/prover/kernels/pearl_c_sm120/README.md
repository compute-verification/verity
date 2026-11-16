# Pearl-C on the RTX PRO 6000

These are Pearl-C's device kernels for sm_120a: the served variant `sm120` (G = 4) and `sm120-unpromoted`, with the
hashing format -h3. `run.py` checks every buffer the kernels write against the CPU reference, byte for byte, before it
times anything, and the vLLM adapter loads that same `run.py` by path as the served kernel and names the variant by
its sha256. A change to it is a new variant, which serves only once it has passed its own gate. Its timing is
diagnostic: a number of record is [the GPU harness's](../../../../../../../../benchmarks/pouw/harness/README.md), and
the served one is [`pearl_c_vllm/`](../../../../../../../../benchmarks/pouw/pearl_c_vllm/README.md)'s.

## Try it

Without a GPU, the tests run `run.py` on a stand-in for the CUDA driver:

~~~sh
uv run pytest tests/benchmarks/pouw/test_pearl_c_sm120.py -q
~~~

It prints `10 passed, 1 skipped`; the skipped test builds the cubin, and runs where `VERITY_POUW_NVCC` names a CUDA
12.8 or later toolkit.

On the card, the kernels are built off it and checked and timed on one RTX PRO 6000:

~~~text
bash benchmarks/pouw/pearl_c_sm120/ship.sh [OUT]    # off the card, nvcc 12.8 or later; default /tmp/pearl-c-sm120-ship
bash benchmarks/pouw/pearl_c_sm120/card.sh          # under research run, with OUT sent as ship.tar
python3 run.py [check.json]                         # in the built tree, on the card
~~~

`run.py` exits 3 on a GPU other than the expected one and 4 on a failed check. [`DESIGN.md`](DESIGN.md) says what the
build's gate refuses, what a run writes, what each other file is, and who reads this directory.
