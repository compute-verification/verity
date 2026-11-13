[2026-11-13T03:01Z] 06fd05e9776d
THREADS, Thu 12 Nov, 7:00 PM PST — TRN-153 point 3, the options

**I recommend option 2 with a guard: on a CPU the job pins attention to the math kernel in both of its processes, and the counter makes a run invalid if a fused CPU attention kernel runs at more than one thread. It makes all three statements true, keeps 8 threads (about 1.6 times slower per step than the fused kernel, still about 5 times faster than one thread), and touches no record of record. `VERSION` moves (a minor bump). I am stopping here for your ruling.**

## What the measurements say

Step timing (RAN `r20261113-022554-1c91` at 1 thread on cpu-1, `r20261113-022608-0cb3` at 8 threads on cpu-5; both AMD EPYC-Genoa, 192 CPUs):
- The script is scratch `steps.py`, outside the tree. It runs the s-t20 model (d_model 512, 6 layers, 8 heads, ctx 1024) the way a route's step does: bf16 autocast, cross-entropy, backward, AdamW, 8 rows a step on generated tokens. It times 3 steps after one warm-up, then runs the same steps again.
- No CPU node holds the starts or the data, so it is not a registered job (RAN `r20261113-021849-402a`).

Seconds per step (the mean of 3; the second run within 15%):

| Threads | Fused kernel | Math kernel |
|---|---|---|
| 1 | 21.3 | 27.1 |
| 8 | 2.81 | 4.43 |

So at 8 threads one thread is about 7.6 times slower than fused, and the math kernel about 1.6 times slower (INFERRED from the table). At the configs' batch of 64 rows that is roughly 22 s a step fused, 35 s math and 170 s at one thread (INFERRED, linear in rows). Both node runs ended with the same weights twice (`same_weights` true), as expected on AMD.

Whole training on the Intel AMX VM, which races (RAN locally, `steps.py`, s-t20, 4 rows, 4 steps, run twice and the weights' sha256 compared):

| Threads | Kernel | Pairs with the same weights |
|---|---|---|
| 8 | fused | 0 of 3 |
| 8 | math | 5 of 5 |
| 2 | fused | 1 of 1 |
| 2 | math | 1 of 1 |

At the kernel level at 8 threads on the AMX VM (RAN locally, `repro.py`, 1,000 calls each): math and plain forward and the plain backward differ in 0 calls, at topcode and at s-t20. Fused forward differs in 10 calls at topcode and 26 at s-t20.

**There is no dial for the kernel.** No route dial and nothing in `config.py`, `routes/` or `run.py` picks it (READ; `grep sdpa_kernel` finds only `capture/trace.py` and the tests' `job_counted`). The capture already traces under `sdpa_kernel([SDPBackend.MATH])` (READ `capture/trace.py:354-356`). The counter prices a fused attention kernel as the math kernel's ops (READ `flops.py`, `FUSED_ATTENTION`'s comment, and `Counter._attention`). So pinning math on a CPU changes no count, and the trace already describes the math path.

## The options

Across all four (READ, 2a), no record of record is touched: all 32 `of_record.toml` entries are CUDA, and none of the options changes anything on CUDA.

**1. A CPU job holds one thread.** `THREADS` becomes 1 on a CPU, and `--threads` above 1 on a CPU is refused.
- Makes true: all three statements. A one-thread fused call never differed (RAN, 0 of 2,000, forward and backward, Intel and AMD).
- Slows: a CPU job by about 7.6 times at s-t20 (RAN).
- Tests:
  - `test_resume`'s slow continuation test loses its `threads=2`. It then no longer checks that a continued job holds the kept count; a refusal test that computes nothing can keep that.
  - `test_run`'s test near `run.THREADS, []` changes its expected default.
- `VERSION`: moves. What a CPU job computes changes, and a kept state at 8 threads can no longer be continued.

**2. A CPU job's attention runs under the math kernel (recommended).** No dial does it today, so it has to be built.
- What to build:
  - `main` enters `sdpa_kernel([SDPBackend.MATH])` on a CPU before either process computes anything, next to `limit_cpu`. Both processes need it: the job's own process scores checkpoints with the model's forward (READ `evaluate.py:39`), and the grade reads those scores.
  - The guard: a route is screened, but `sdpa_kernel` is not on `screen.py`'s `FORBIDDEN`, so a route could turn the fused kernel back on. So the counter makes a run invalid when `_scaled_dot_product_flash_attention_for_cpu` or its backward runs with more than one thread held. The counter already sees that op (READ `flops.py:145`, `:791`).
  - The record names the kernel next to `cpu_isa`.
- Makes true:
  - "a job's products are to be the same every time": the math path never differed at the kernel level, and whole training repeated its weights (RAN, AMX, 8 threads);
  - the README's "follows that count": the count still changes the reductions and the cores do not;
  - `CPU_ISA`'s comment, rewritten so that it claims the same-every-time result for oneDNN's products only and names the fused attention kernel as the reason for the pin.
- Slows: a CPU job by about 1.6 times at 8 threads at s-t20 (RAN). At 1 thread math is also slower (27 against 21 s), so the pin holds at every count to keep one rule; the guard refuses the fused kernel only at more than one thread, where it races.
- Tests:
  - `test_resume`'s 2-thread test becomes safe; it runs through `main`.
  - `test_run`'s fused-against-math count test and `test_as_math` call `run.train` directly at 1 thread, so they keep comparing the two kernels.
  - New tests: a CPU job's record names the math kernel, and a route that turns the fused kernel back on at 2 threads is invalid. Both are cheap and compare no bits.
- `VERSION`: moves. A CPU job's losses change bits, and a kept state from before the change can't be continued (`Kept.load` refuses another VERSION). A minor bump, since counts don't change.
- Risk: the evidence that the math path is the same every time is empirical, from one Intel VM and the AMD nodes, and not a property of torch. A torch upgrade could change it. INFERRED.

**3. A job refuses bf16 attention on a CPU at several threads.** The counter's guard from option 2, without the pin.
- Makes true: the statements, by refusal.
- Slows: every route uses bf16 autocast and SDPA, so every CPU job at the default 8 threads would be invalid. In practice it is option 1, enforced late: the job runs to the end and is then invalid.
- Tests: `test_resume`'s `threads=2` job would be refused, so as in option 1. `VERSION` moves.

**4. Only the statements change.** For example: "on a CPU at more than one thread, the fused bf16 attention kernel can return a wrong tile, so a job's products are the same every time only at one thread or on CUDA".
- Makes true: only the docs. A CPU job at 8 threads can still train on a wrong tile.
- Slows: nothing.
- Tests: `test_resume`'s 2-thread test can still flake, so it is deleted or moved to 1 thread.
- `VERSION`: does not move.

**5. (Set aside) f32 attention on a CPU.** f32 fused SDPA never differed (RAN, 0 of 2,000 at 4 threads, Intel). But it changes the model's attention (`model.py`, which every route runs), makes CPU and CUDA compute different precisions, and rests on the same kernel family that races. Option 2 does the same job without touching the model.

## Why option 2

It is the only option that keeps a CPU job's thread count and makes the statements true. The trace and the counter already describe the math path, so the pin aligns the job with what is counted. The guard makes it enforced rather than opt-in.

What it costs: about 1.6 times per CPU step, and a minor `VERSION` bump. CPU jobs are tests and local runs; no record of record is a CPU job.

The draft (point 4) would carry the pin, the guard, the record's key, the three statements, the `VERSION` bump and the two tests, with `test_resume`'s 2-thread test kept. If you rule option 1 or 4 instead, that test loses `threads=2` or is deleted.

