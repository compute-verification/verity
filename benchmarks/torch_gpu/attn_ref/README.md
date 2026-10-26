# attn_ref: attention's BF16 GEMMs, slow and exact

The fallback for the e2e demo's decode attention on one H200 (sm_90a), so nothing waits on the fast kernel (the kernels
lane's attn_bf16). QK^T and PV, each output word exactly `GemmCoordinate_v2{K, GEMM_HOPPER}`
(`verity.ml.circuits.ops.gemm`): K/16 `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` steps (`sm90.mma.m16n8k16.bf16`,
pinned to `HOPPER_BF16_M16N8K16`) in ascending k from +0, then `cvt.rn.bf16.f32`. QK^T's K is the head dim. PV is the H200
circuit's `GemmCoordinatePad_v1{T=n, K=32}` at KV length n: the n live keys, both operands padded with +0 to K = 32, so two
steps at every length 1..32 (keyword `kpv`, default 32; past kpv keys, K is n padded to a multiple of 16). Softmax is its
own unit, so QK^T's words come out and PV's P goes in.

```python
import attn_ref                                  # beside attn_ref.sm_90a.cubin, or $ATTN_REF_CUBIN
s = attn_ref.qk(q, k, lens)   # q [B, H, D], k [B, L, D] (MQA) or [B, Hkv, L, D] -> [B, H, L] bf16; 0xFF80 past lens[b]
o = attn_ref.pv(p, v, lens)   # p [B, H, L], v [B, L, D] or [B, Hkv, L, D] -> [B, H, D] bf16, K = max(kpv, pad16(lens[b]))
```

- `attn_ref.cu`: the two kernels. One warp owns a 16 x 8 tile of output words (16 query heads by 8 keys, or by 8 of d) and
  carries its accumulators through the whole k range: no split-K, no atomics, no shared memory. Operands are gathered a
  word at a time, so any length and any head count works; it is not tuned.
- `attn_ref.py`: the torch wrapper. It loads the cubin through the CUDA driver API into torch's context and launches on
  torch's current stream (`lens` int32 on the device; `check_lens=False` under a CUDA graph).
- `build.sh`: on a CPU node, the cubin (`-fmad=false`, no FTZ or fast-math), `sass_gate.py` on it, and `fixture.py`'s check
  data.
- `fixture.py`: 40 random sequences at the frozen shape (256 query heads of head dim 64 on one KV head, 32 cached keys) in
  ten kinds (normal, softmax, clustered exponents, the unit binade, subnormal products, 5% specials, cancellations inside
  a step and across steps, uniform words, and PV accumulators that are FP32 subnormals after the first step), with every
  word the kernels must write at each KV length 1..32, from the twin held to the Definition's reference evaluator
  (`GemmCoordinatePad` for PV) on a sample; and a GQA case off the shape (head dim 128, 37 keys, lengths 0..37).
- `check.py`: on the card, every batch built from copies of the pool, so every word is compared: batch sizes 1, 7, 128 and
  20,480 at every KV length (the cache and P past it junk), mixed lengths, other warp counts, the odd case, a no-launch
  control, then diagnostic timing beside torch's cuBLAS matmuls.
- `session.sh`: the card's part as one run from the build's ship, under `gpu-lease 1` of at most 30 minutes.

```text
research run --on auto --cwd source --cpus 8 --mem-gb 16 -- \
  bash -lc 'uv sync -q --all-packages --extra torch-cpu && PROCS=8 bash benchmarks/torch_gpu/attn_ref/build.sh "$OUT/ship"'
research fetch BUILD_RUN --all && tar -C ~/.research/runs/BUILD_RUN/out/ship -cf /tmp/attn-ref-ship.tar .
research run --on vy-dc-h200 --project verity --source . --cwd source --env GPU_LEASE_WHO=LANE \
  --send /tmp/attn-ref-ship.tar -- bash benchmarks/torch_gpu/attn_ref/session.sh
uv run tools/verity/check/suites.py benchmarks/torch_gpu/attn_ref      # the CPU suite
```

Results go to the evidence store with the run, never here.
