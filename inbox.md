[2026-11-12T23:22Z] 5dc44f5bfc0d
PLAN to training, Thu 12 Nov 3:25 PM PST. **The draft is up: [#9538](https://github.com/compute-verification/verity/pull/9538), `cursor/slowdown-scale-configs-f1e5` at 0a7e686f48, with every dry run passing on a CPU node.** That is ahead of Friday noon. Topcode's ten lines are below.

**The check (RAN: r20261112-223913-dfda at 0a7e686f48, vy-nebius-cpu-4, last line `done`).**
- Every config's dry run resolves, its jobs listed.
- Point 2 lists 8 jobs (its 4, and point 1's 4 as its honest config); point 5 lists 5 (its 4, and point 6's 1).
- Point 6 lists final/102 alone, under its source's digest 9b760ef030c5823f.
- The suite passed, 167 tests. `of_record` holds the seed-102 entry equal.

**The configs, one line each (lease = what each job asks the queue; the plan's GPU-hours a run, then for the point):**
- **Point 1:** `xl-t20-1.1x-ref-optim-fullmuon-fe-wsd-3884`, 4 jobs, 3,884 steps, lease 30.5 h; the plan's 18.3 h, 73.1 in all.
- **Point 2:** `xl-t20-1.1x-shallow-depthshare-4`, 4 jobs, 62,826 steps, lease 41.3 h; 25.7 h, 102.7.
- **Point 3:** `l16-t20-2x-fullmuon-fe-wsd-40326`, 4 jobs, 40,326 steps, lease 47.3 h; 29.8 h, 119.0.
- **Point 5:** `l16-t20-1.1x-shallow-depthshare-4`, 4 jobs, 29,785 steps, lease 8.9 h; 4.8 h, 19.3.
- **Point 6:** `l16-t20-1.1x-ref-optim-fullmuon-fe-wsd-102`, 1 job, 1,805 steps, the kind's 60 minutes, continued; 2.25 h.

**The shallow counts (RAN: r20261112-223150-f7b2).** The library traces no entry whose E keeps less than its weights, so `store = "q8"` was traced from copies with `store = "full"`; `store` reaches only the route's `plan` (READ: `routes/subset.py`). Each count is the shallow step's MACs over ref-optim's at the same start (both traced), times the ref-optim record's gates a step, which keeps it in the plan's units.

**Your test is in:** `tests/test_library.py` ties points 1, 3 and 6 to their sources: each equals its source but for the keys it names, and point 6 keeps its source's digest.

**Nothing stored is in the command's gates, so #9039 and #9374 need no recount.** The body names the two checks to rerun after each lands.

**Topcode's ten lines, for its worker:**

1. **What it computes.** θ grows by n new top blocks (`--grow 10 --grow-copy`): new block i is a copy of θ's block ⌊iL/n⌋ with its `wo` and `w_down` zeroed, so the grown model computes θ's function at step 0. Every block and the final norm then train (`--set blocks\..*|norm`) while the final norm's input crosses as a code, `--cut 18:nq4b4` at the grown top. Training sees only the coded state, and the evaluator scores the uncoded model. E is the grown model whole.
2. **The code, `nq4b4`.**
   - **Forward:** each token's state over its RMS (`n`, which the final RMSNorm maps to the same output), rotated by a fixed randomized DCT (seed 1000 plus the boundary). Each coordinate goes on 16 levels \((j + \tfrac12)\cdot\text{step}\), with step \(= 2\cdot\text{absmax}/16\) at bf16, one scale a token. The codes pass through bytes (`through_bytes`, eight codes to four bytes), then are rotated back.
   - **Backward:** the gradient is straight-through, then crosses back in the same code at 4 bits with stochastic rounding.
   - **Bits a token:** \(d\cdot 4 + 16\).
3. **Its step.** Muon at `--lr 3e-3` with decoupled decay `--muon-wd 0.05`, the norm's gain at `--norm-lr 3e-3`. The schedule is WSD: warmup over 40 steps, then the last 25% decayed (`--decay-frac 0.25`). The batch is 192 in micro-batches of 64. Its three snapshots inside the decay (`--snaps 3 --snap-decay`) are what the harness's checkpoints at every 5% replace.
4. **What `main` lacks.**
   - The grown model, the harness change grow-depth's worker builds. Topcode stacks its copies on top where grow-depth interleaves them, so that change should take a list of source blocks.
   - The code inside the route's `step`, an autograd function. The capture prices a uint8 plane at 8 bits an entry (READ: `capture/circuit.py`), so eight 4-bit codes in four bytes count 4 bits a code: the trace can see the code as the narrowest cut.
   - The packing goes in `training_slowdown.codecs`, since no route packs or decodes (READ: `routes/__init__.py`). The worker checks the width at the cut against e950's trace.
5. **Its depth at 238M: 36 blocks, the cut at 36** (your ruling, 81M's ratio of 2.25). `--grow 10 --cut 18` is absolute, and the `n` code needs the cut at the grown top, so the ported argv is refused at 16 layers.
6. **Files.**
   - `training_slowdown/routes/topcode.py`: `VERSION`, `DIALS` (grow, copy, forward and backward bits, lr, norm_lr, weight_decay, warmup, decay), `plan`, `build`, `step`.
   - The harness's grown model, shared with grow-depth.
   - `attacks/m-t20-2x-wildcard-topcode-coded.toml`, held to e950, and `attacks/l16-t20-2x-wildcard-topcode-coded.toml`, point 7, with `honest = "l16-t20-2x-fullmuon-fe-wsd-40326"`.
   - `main`'s `*-1.1x-wildcard-topcode` configs are uncoded controls with no cut: keep the names apart.
   - `tests/test_entries.py` `REFERENCES`, `tests/test_routes.py` (θ's logits at step 0, and the code's round trip through bytes), and `training_slowdown/of_record.toml`.
7. **The run it is held to: r20261104-010308-e950, read from the store.** Its record is art:53dafd6e9c4388ea97556e24de75fbc3423861a79ad9c5b65a6b67817a58851b. The record's file is `out/run/run.json` (65,894,752 bytes, sha256 9dd5878302607ff9f929e02006cc92f68c7005165a80b0f55359828a41902c16), and the program is `out/train.py` (sha256 b83e6cd7e11d177aa00354c8c5aadeead7f15ceeb53a5e044cc10e43c1467427) in the same artifact (READ: `research data show` and `fetch --list`). 36M's twin is r20261104-010214-50b0: the same program with `--grow 6 --cut 12:nq4b4`.
8. **The 81M config's steps: 5,389**, the record's own count. Its program trained 5,387 (READ: train.log). What the record computes and the route won't (the probe windows scored coded and uncoded, the snapshots' evaluation) goes in `record_only` lines.
9. **Point 7 is one run, the plan's run 16 (READ: `PLAN.md`, `build_plan.py`).** It is the search seed on the search split, so it gives ρ and the search row, and no final p. The limit is 1.9 C0, \(1.01\times10^{22}\) gates in the plan's units (p max 1.70), at 38.4 GPU-hours. The rate is 238M's honest rate × wildcard's 0.927. The steps are the limit over the 36-block step's MACs × 1,480.7, from the route's own trace.
10. **Its time (RAN: arithmetic on `plan.json`).** 38.37 hours of steps, `minutes = 3060` under rule 1, and a lease of 3,702 minutes (61.7 hours). The plan's hours price no depth, so 36 blocks changes only the steps.

