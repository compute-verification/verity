# attn_bf16: attention's BF16 GEMMs on sm_120, bit-exact to their Definition

QK^T and PV for one RTX PRO 6000 (sm_120), each output word exactly
`GemmCoordinate_v2{K, GEMM_SM120_BF16}` (`verity.ml.circuits.ops.gemm`): K/16 `mma.sync.m16n8k16` BF16 steps in ascending
k from +0, then `cvt.rn.bf16.f32`. QK^T's K is the head dim; PV's is the query's KV length, zero-padded to a multiple of 16
in both operands. Softmax is its own unit, so QK^T's words come out and PV's P goes in. Decode (one query a sequence),
causal prefill and prefill chunks, head dims 64 and 128, a BF16 KV cache.

- `attn_bf16.cu`: the kernels, and how each word's chain is fixed (one warp per 16 x 8 tile, no split-K, no atomics, and
  per-query zeroing of V in the block where the tile's queries stop at different keys).
- `build.sh`: the cubin (`-fmad=false`, no FTZ or fast-math), `sass_gate.py` on it, and `fixture.py`'s check.
- `fixture.py`: the cases and the buffers the kernels must leave, from the Definition's twin, held to its reference
  evaluator on a sample.
- `run.py`: on the card, the identity check, every case under 0xA5 poison at three launch shapes, a no-launch control
  that must fail, and diagnostic timing against SDPA and torch's matmuls at the model's decode step.
- `screen.sh`: all of it as one run on a node, the card's part under `gpu-lease 1`:

```text
research run --on vy-nebius-1 --cwd source -- env CHECK_SLOTS=/workspace/research/locks/gpu/slots \
  python3 tools/verity/check/slot.py --by LANE --wait-max 3600 --run-max 2700 -- \
  bash benchmarks/torch_gpu/attn_bf16/screen.sh
```

Results go to the evidence store with the run, never here.
