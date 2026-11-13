[2026-11-13T00:37Z] 8d23e7a70945
TOPCODE, Thu 12 Nov, 4:40 PM PST.

**Design gate: the code is an autograd function that `step` applies to the grown top's output, the bytes are made by a new `codecs` function, and the trace sees a uint8 plane at d·bits + 16 bits a token (3,088 at 81M, the record's own figure). One change outside your list is needed, in `of_record.py`'s `norm` rule.**

Branch: `cursor/training-topcode-9478`, cut from `cursor/training-grow-depth-48ba` at b16502a23d. Nothing is pushed yet, and I have edited no file of the tree.

## 1. Where the code sits, and what the trace sees

- **In `step`.** Each micro-batch runs `model.hidden(x)`, which is every block, the grown top included. Then `rms_unit`, the record's own ops. Then `Code.apply(h, model.rotation, forward_bits, backward_bits)`, an autograd function in `routes/topcode.py`. Then the final norm and the tied head: `F.linear(F.rms_norm(h, …, model.norm), model.embed)`, which is `LM.top`'s last line. The record's forward hook on its block 17 computes the same (READ: train.py `install_cuts`, `rms_unit`, `Coded`).
- **Forward.** Autocast is off and everything runs in fp32. First z = x @ Rᵀ. Then `codecs` rounds each token on one scale: step = absmax · 2 / 2^bits at bf16, and j = floor(z / step) clamped to [−8, 7]. The codes j + 8 go through `codecs.through_bytes`: eight codes go into `bits` bytes of one uint8 tensor and come back out. The value is (j + ½) · step, then @ R. The op sequence copies the record's `quantize` and `through_bytes` op for op, so the count logs the record's rows: `mul [[8]]` ×2 and `mul [[4]]` ×2 a forward, and the `[[65536,1,768],[65536,1,1]]` quotients and products (READ: the record's modal structure).
- **Backward.** Straight through the rounding. The rotation is g @ Rᵀ, then stochastic rounding, then @ R.
- **R.** The record's orthonormal DCT-II times seeded ±1 signs, with seed 1000 + the boundary, which here is the grown depth. I build R once, in `build`, as a non-persistent buffer of the model (`model.rotation`). Each row's cos is computed alone, as `model.py` builds its rope tables, and the signs come from a CPU `torch.Generator`.
  - The products with R count as θ's (a constant buffer, README), as the record's in-step products do: `fwd/theta mm [[65536,768],[768,768]]` ×6 (READ).
  - Why not rebuild R every step, as the record does: one 768×768 `cos` call is the case `model.py`'s comment names. Torch splits it over threads, and the first such call a process makes can come back inaccurate in one thread's share, so a resumed job could compute another R (READ the comment; the risk to R is INFERRED).
  - Cost of the buffer: the record's step has 7 small lines that build R, and the route doesn't compute them each step. They are `other/param` `mul [[1,768],[768,1]]`, `mul [[1,768]]`, one of the 1,621 `mul [[768,768]]`, `div [[768,768]]`, `div [[768]]`, `mul [[768]]` and `mul [[768,768],[768]]`, together about 2.4M MACs a step. They go in `record_only` (READ: I matched each against `rotation()`'s ops).
  - The other way, building R in-step as the record does, needs no named lines and carries the resume risk. Your call.
- **What the trace sees at the cut.**
  - The uint8 plane: d·bits/8 entries a token, at 8 bits each. With the token's bf16 step (16 bits) that is d·bits + 16 bits a token: 3,088 at 81M and 4,112 at 238M.
  - RAN, as a local scratch probe outside the tree (no run id): at d = 16 and 4 bits, the capture's `a_bits` is 80 = 16·4 + 16 with the codes through bytes. It is 256 with the same code unpacked, and 256 with no code.
  - The record holds no trace (READ: `run.json` has no `trace` key). The 3,088 is its program's own `a_fwd_bits` (READ: `train.log` line 2). The 81M comparison's trace will give the figure at the real shape.

## 2. The backward's stochastic rounding

- **Its seed.** `torch.rand_like` on torch's global generator, as the record uses. The job's process seeds it (`torch.manual_seed(seed)`, run.py), and a kept state carries torch's host generator and every GPU's (`run.generators`). So a resumed job draws what the unbroken one draws (READ: run.py and the route contract). The resume test gets a tiny topcode job that continues bit for bit (to be RAN).
- **The trace reads it.** It is a random op, and τ's `tau_other_bits` counts 64 bits a seed for each random op a token op reads, here one a micro-batch. In the probe that was 0.031 bits a token (RAN, same probe). No dial or state of the route holds a generator.
- **The backward's codes don't pass through bytes.** In the record's program only `quantize`'s nearest branch calls `through_bytes`. The stochastic branch returns floats on its own grid: end levels at ±absmax·(1 + 2⁻⁷) at bf16, 2^bits − 1 steps, unbiased, nothing clipped (READ). I keep that. The trace measures no backward crossing, and the record's `a_bwd_bits` of 3,088 is its own arithmetic.

## 3. Changes outside `routes/topcode.py`, `codecs`, the two configs, `of_record.toml` and the tests

- **`of_record.py`'s `norm` rule must change.**
  - `rms_unit` writes `fwd/act mul [x, s]` and its backward's products over the norm's own shapes, on both sides: the record's CUDA program runs it as plain ops, 3 a step (READ: its modal log).
  - The rule returns None when the record holds any key it covers. It also requires the config's normalisations to equal its gains' multiplies, and here they are 114 against 111. So it wouldn't fire, and the whole CPU norm would print as differences (INFERRED from the code).
  - The change: the rule covers the config's excess over the record's count of each such key, with the same equalities on the excess. A test goes in `test_of_record.py`.
- **README:** the route's row and the codecs row (docs).
- **Nothing else.**
  - `run.py`, the capture and the evaluator don't change. The origin stacks the copies above θ's blocks, which GROW's test covers.
  - R is a non-persistent buffer, so θ′'s map, E and the evaluator's state don't see it. The model scored is the uncoded grown one.
  - `VERSION`: I'd leave it at 4.24.0, since no existing record's measurement changes. Say if you want a bump.

## 4. Where I read the ten lines differently

- **Line 1, the new blocks.** Every new block has its `wo` and `w_down` zeroed, not only a repeat. New block i copies θ's block ⌊iL/n⌋ (READ: `grow()`). So topcode's `origin` is not grow's ⌊jL/N⌋ over the whole depth.
- **Line 2, the code.**
  - The backward's grid is not the forward's, and the backward is not through bytes (above).
  - The record's R is rebuilt every step, with signs from the device's generator, CUDA's in the record (READ: `rotation()`). Mine is the same on every device, so it isn't numerically the record's R. The count is unaffected.
- **Line 3, the step.** The final gain and the blocks' gains train under fused AdamW: betas 0.9 and 0.95, decay 0 (the `--wd` default), gradient clip 1.0. The embedding is frozen (`--set` doesn't match it). Newton–Schulz divides by the scalar norm (the record logs `div [[768,2048],[]]`), so this is the default Muon with no `stack`, and nothing is logged at another shape. The WSD schedule is `subset.schedule` at `decay = 0.25`, `warmup = 40`. It runs over the config's 5,389 steps, where the program's ran over its 5,387 (READ: `train_iterative`, `main`).
- **Line 6, the `copy` dial.**
  - With `copy` false, the program starts the new blocks in model.py's own seeded init, and an origin can't state that: it maps each tensor to θ's tensor or to zeros. I'd drop the dial, so the route always copies. Say if you want it kept and refused at false.
  - The forward bits run from 4 to 8, the group is the whole token, and the backward bits run from 1 to 16. At 1 to 3 bits the program rounds the forward on another grid, Max's Gaussian step times the RMS, which the route doesn't carry.
  - The dials: `grow`, `forward_bits`, `backward_bits`, `lr`, `norm_lr`, `weight_decay` (Muon's, decoupled), `warmup`, `decay` and `micro_batches`.
- **Lines 3 and 9, 238M's micro-batches.** The 81M record runs 3 forwards of 64. For 36 blocks at 238M I'd take 6 forwards of 32, as GROW's 238M config does, unless the plan says otherwise.
- **Line 8, the counted step.** I expect step 3, as for grow (INFERRED):
  - At step 1, Muon's decoupled decay moves every matrix, while each new block's gains get no gradient behind its zero `wo` and `w_down`, and undecayed AdamW leaves them as they were.
  - At step 2 the gains leave θ′.
  - The record has 4 steps off its modal structure: its two probe steps and its first two training steps (READ: `structures`).
- **Line 8, `record_only`.** The probe windows run in steps of their own, and the snapshots' `encode` is the program's own uncounted work, so neither is in the modal step and neither needs a `record_only` line. The record's 5,389 steps include the two probe steps (READ: `train.log`'s `"steps": 5387`, and `cut_probes` under `step_of`). The `record_only` lines are the 7 that build R.

Until you answer I'll write `plan` and `origin`, the two configs and the tests that need no code path.

