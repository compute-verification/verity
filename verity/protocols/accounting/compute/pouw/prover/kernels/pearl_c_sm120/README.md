# Pearl-C on sm_120a

The build of `pearl-c-sm120-v1-h3` for the RTX PRO 6000: `pearl_c_sm120.cu` with `mainloop_sm120.cuh`'s main loop and
atoms, the variants `sm120` (G = 4) and `sm120-unpromoted`, hashing format -h3, and `run.py`, which the vLLM adapter
loads as the served kernel on this card. `fake_tma.c` is the stand-in driver library's tensor maps, which the sm_90a
build's tests share. The subsystem is [`../README.md`](../README.md).

## Try it

~~~sh
uv run pytest tests/benchmarks/pouw/test_pearl_c_sm120.py -q
~~~

It runs `run.py` on the stand-in driver library, with no GPU; the test that builds the cubin runs only where
`VERITY_POUW_NVCC` names a toolkit.

On the card: `bash benchmarks/pouw/pearl_c_sm120/ship.sh [OUT]` off it, then `python3 run.py [check.json]` in the built
tree (`benchmarks/pouw/pearl_c_sm120/card.sh` does both under `research run`). `run.py` exits 3 on a GPU other than the
expected one and 4 on a failed check.
