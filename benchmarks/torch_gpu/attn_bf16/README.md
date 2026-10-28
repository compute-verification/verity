# attn_bf16: attention's BF16 GEMMs, bit-exact to their Definition

QK^T and PV for one H100/H200 (sm_90a) or RTX PRO 6000 (sm_120a), each output word exactly
`GemmCoordinate_v2{K, GEMM_HOPPER}` (`verity.ml.circuits.ops.gemm`): K/16 `mma.sync.m16n8k16` BF16 steps in ascending k
from +0, then `cvt.rn.bf16.f32`. QK^T's K is the head dim; PV's is the larger of `kpv` and the query's KV length padded to
a multiple of 16, both operands zero-padded to it. Softmax is its own Definition, so QK^T's words come out and PV's P goes in.
Decode (one query a sequence), causal prefill and prefill chunks, head dims 64 and 128, MHA, GQA and MQA over a BF16 KV
cache. Two paths compute the same words: the fast one, and a slow reference path that is tuned for nothing.

```python
import attn                                      # beside attn_bf16.<arch>.cubin, or $ATTN_BF16_CUBIN
s = attn.qk(q, k, lens)   # q [B, H, D], k [B, L, D] (MQA) or [B, Hkv, L, D] -> [B, H, L] bf16; 0xFF80 past lens[b]
o = attn.pv(p, v, lens)   # p [B, H, L], v [B, L, D] or [B, Hkv, L, D] -> [B, H, D] bf16, K = max(kpv, pad16(lens[b]))
o = attn.pv(p, v, lens, path="ref")              # the same words through the reference kernels
```

- `attn_bf16.cu`: the kernels, and how each word's chain is fixed (one warp per 16 x 8 tile, no split-K, no atomics, and
  per-query zeroing of V in the 16-key segment where the tile's queries stop at different keys). The `_h8` kernels put 8 query
  heads that share a KV head in a tile's columns, so a decode step reads each key and value once for 8 heads, and a warp
  runs every tile of its head groups (Q or P loaded once a group). The `_h16` kernels put 16 heads in a tile's rows
  instead, and load and store word pairs 32 bits at a time where a buffer's strides allow; their PV transposes O's words
  within each quad of lanes, so that a lane stores 8 contiguous words as one 16-byte store where O is 16-byte aligned. The
  `run.py` check also runs every tiling with S and O 1 and 2 words into their buffers, to reach the narrower stores. The `_s16` kernels are
  `_h16`'s where no query sees more than 32 keys (the H200 model's decode step): a warp loads K's or V's fragments once
  for all of its head groups. `attn.LAUNCH` picks the tiling and launch shape for each op, and `attn.serves` says where
  each applies. The `_ref` kernels are the reference path: a CTA takes 8 columns of one KV head and sequence, its warps
  take the (16 query heads, query) pairs in turn, and every operand word is gathered on its own.
- `attn.py`: the kernels as torch functions (the H200 example's interface), `qk(q, k, lens=None, *, out, fill, warps,
  check_lens, path)` and `pv(p, v, lens=None, *, out, kpv, warps, check_lens, path)`: S past `lens` is `fill` (-inf),
  PV's K is `kpv` = 32 for every KV length 1..32 (the H200 circuit's `GemmCoordinatePad_v1{T, K=32}`), and `path` is
  `"fast"` or `"ref"`. Without `lens` each call is one launch on torch's current stream, from the shipped cubin.
- `build.sh`: on a CPU node, both cubins (`-fmad=false`, no FTZ or fast-math), `sass_gate.py` on each (every kernel of
  both paths), and `fixture.py`'s check.
- `fixture.py`: the cases and the buffers the kernels must leave, from the Definition's twin, held to its reference
  evaluator on a sample; among them `kinds_d64`, ten generators of hard inputs at every KV length 1..32 (cancellations,
  subnormal products and accumulators, uniform words); the H200 model's whole decode step as per-sequence digests over
  `gen.py`'s inputs.
- `run.py`: on the card, the identity check, every case under 0xA5 poison in each tiling of both paths at several launch
  shapes, a no-launch control that must fail, the whole decode step also over its first 4,096 and 16,384 sequences,
  `attn.py`'s functions through both paths on every decode case, and diagnostic timing against SDPA and torch's matmuls
  at the model's decode step for M = 4,096, 16,384 and 20,480.
- `run.sh`: the card's part as one run from the build's ship (`IN/ship.tar`), `run.py` under `gpu-lease 1`. The
  card's canary is `benchmarks/tc_probe`'s own run.
- `tests/`: on the CPU, the SASS gate's verdicts, `attn.py`'s kernels and the reference grid, and `fixture.py`'s words on
  `kinds_d64` against the Definition's reference evaluator (`GemmCoordinatePad` for PV).

```text
# from the tree's root, on a CPU node with CUDA's nvcc (12.8 or later), 16 CPUs and 48 GB: the cubins, their SASS gates
# and the fixture
uv sync --all-packages --extra torch-cpu
PROCS=16 bash benchmarks/torch_gpu/attn_bf16/build.sh /tmp/attn-bf16-ship
mkdir -p IN && tar -C /tmp/attn-bf16-ship -cf IN/ship.tar .
# on an H200 with CUDA torch and `gpu-lease`: run.py's checks and timing
OUT=DIR IN=IN bash benchmarks/torch_gpu/attn_bf16/run.sh
uv run tools/verity/check/suites.py benchmarks/torch_gpu/attn_bf16      # the CPU suite
```

`examples/h200`'s attention and `benchmarks/torch_gpu/commit_slots/overlap.sh` run the reference path (`path="ref"`) from
`build.sh`'s sm_90a cubin, named by `$ATTN_BF16_CUBIN` (`attn.cubin_path`).

Results go to the evidence store with the run, never here.
