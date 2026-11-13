[2026-11-13T02:48Z] 723b708e3121
THREADS, Thu 12 Nov, 6:48 PM PST — TRN-153 points 1 and 2

**The wrong tile comes from the fused bf16 CPU attention kernel alone (forward and backward), on the Intel AMX machine; every other path, and every path on the AMD nodes, came out the same on every call; and no record that feeds a figure is a CPU job, so there is no urgent line.**

## 1. The reproduction

Script: scratch `repro.py`, outside the tree. It sets the job's ISA limit the way a job does (`run.limit_cpu("cpu")`, then `run.set_threads(n)`, before torch computes anything). Each case's first call is the base, and every later call that is not `torch.equal` to it counts as differing. For each differing call it records where (batch, head, token range, entries) and the largest absolute difference. The inputs are seeded `randn`, the shapes are topcode (2,2,1024,8) and s-t20 (2,8,1024,64) (ctx 1024, d_model 512, 8 heads), and torch is 2.14.0+cpu everywhere.

Cases:
- `sdpa`: fused, bf16, causal;
- `sdpa_f32`;
- `plain`: q@kᵀ/√d, causal mask, softmax, @v, in bf16;
- `math`: SDPA under `sdpa_kernel([SDPBackend.MATH])`, bf16;
- `matmul`: one bare bf16 q@kᵀ;
- `sdpa_backward` and `plain_backward`: dq, dk and dv against a seeded grad.

**The nodes: AMD, `cpu_isa` `{'ONEDNN_MAX_CPU_ISA': 'AVX512_CORE_BF16'}`, 2,000 calls a case. Zero calls differ in any case at any thread count.**
- The machine: "AMD EPYC-Genoa Processor", 192 CPUs, avx512_bf16, no AMX. READ: lscpu in each run's out.
- 1 thread: RAN `r20261113-020358-1d05` (cpu-11). 0 of 26,000 so far; its last case, s-t20 `plain_backward`, is still running.
- 2 threads: RAN `r20261113-020410-a87e` (cpu-11). 0 of 28,000.
- 4 threads: RAN `r20261113-020422-21cd` (cpu-11). 0 of 28,000.
- 8 threads: RAN `r20261113-020434-08aa` (cpu-1). 0 of 28,000.

**Intel: this VM, "Intel(R) Xeon(R) Processor", 4 cores, with amx_bf16, amx_tile and avx512_bf16 (the same kind as TOPCODE's VM). The fused bf16 kernel races there.** RAN locally, `.venv/bin/python repro.py`, with the job's ISA limit unless a line says otherwise:

| Shape | Threads | Case | Calls that differ | Largest |
|---|---|---|---|---|
| topcode | 1 | sdpa, sdpa_backward | 0 of 2,000 each | |
| topcode | 2 | sdpa | 0 of 10,000 | |
| topcode | 2 | sdpa_backward | 0 of 5,000 | |
| topcode | 4 | sdpa | 2 of 2,000 | 0.455 |
| topcode | 4 | sdpa_backward | 16 of 2,000 | 0.914 |
| topcode | 4 | sdpa_f32, plain, plain_backward, matmul | 0 of 2,000 each | |
| topcode | 4 | math | 0 of 1,000 | |
| topcode | 8 | sdpa | 24 of 2,000 (and 10 of 1,000 in a second run) | 0.997 |
| topcode | 8 | sdpa_backward | 23 of 2,000 | 1.746 |
| topcode | 8 | plain, math, plain_backward | 0 of 1,000 each | |
| s-t20 | 4 | sdpa | 18 of 1,000 | 2.53 |
| s-t20 | 8 | sdpa | 26 of 1,000 | 1.08 |
| s-t20 | 8 | plain, math, plain_backward | 0 of 1,000 each | |

What a wrong call looks like (RAN, `where` in the local JSON): one block of up to 32 query rows of one (batch, head) is wrong, for example tokens 512–543 of (0,1), about 250 of its 256 entries at topcode. One call in a few hundred also has two such blocks. So it is a tile of the kernel's query blocking.

The environment doesn't stop it (RAN locally, topcode, 4 threads, 2,000 calls):

| Setting | sdpa | sdpa_backward |
|---|---|---|
| `ATEN_CPU_CAPABILITY=avx512` | 3 differ | 18 differ |
| `ATEN_CPU_CAPABILITY=avx2` | 7 differ | 23 differ |
| `ONEDNN_MAX_CPU_ISA` unset | 11 differ | 33 differ |

In two runs the base call itself was most likely the wrong one, so every later call differed: the default capability's `sdpa_backward` (2,000 of 2,000), and an earlier s-t20 run at 8 threads (1,000 of 1,000). INFERRED.

**Which kernel is wrong: the fused CPU flash-attention kernel in bf16, forward and backward. It is not every bf16 product at several threads.** f32 SDPA, the bare bf16 matmul, the plain products and the MATH backend never differed, at any thread count, on either machine (RAN above).

INFERRED: the race is in a path of ATen's CPU flash attention that it picks from the CPU it finds (AMX or Intel), not from `ATEN_CPU_CAPABILITY` or the oneDNN limit. That would explain why the AMD nodes never show it.

Caveat: I could not reach an Intel node. `--on vy-nebius-1` and `--on vy-nebius-3` are refused ("host None"), and the CPU pools are AMD. All the Intel numbers come from this VM. RAN.

## 2. What it reaches

**(a) Records.** None that feeds a figure or table is a CPU job.
- RAN scratch `records.py`, which fetches each record with `research data fetch` and reads only the named keys. All 32 `[held.…]` entries of `of_record.toml` (26 distinct records) are `device cuda`, hardware "NVIDIA RTX PRO 6000 Blackwell Server Edition". Their `threads` and `cpu_isa` are absent, because they predate the key.
- RAN `research data sql` after `labels-sync`. 0 of 237,318 labels have the key `training-slowdown`, so the store holds no other job record.
- The CPU work in the notes is retraces (`--retrace --device cpu`, on fake tensors), for example the "All 44 pairs are device cpu" report (READ `lanes/training/20261112T2103Z-report-LOG-review-9039.md`).

**(b) Checks of record that ran on a CPU at several threads.** None of them reads a value.
- The port check reads counts only. READ `of_record.py` docstring, line 9: "What is held is the step and the kind of the encoding, not the weights, E's bytes or the loss". It compares counted step sets.
- The step's traces read counts only. READ `capture/trace.py`: it traces under `FakeTensorMode` (line 10, and the mode at 354–356, which also pins `SDPBackend.MATH`). It refuses any value read into Python (`aten._local_scalar_dense`, `.item()`, data-dependent output: `CaptureRefused` around line 39 and line 226). So no attention kernel computes a value there. `library.py` and `traced.py` trace the same way.
- Correction to the brief: it is the capture that refuses a value branch, not the screen. READ `screen.py` `FORBIDDEN` does not list `.item()`.
- What does read values is `grade.py` (the checkpoints' losses and the gain). Every record it grades is CUDA (2a).

**(c) The tests on main at several threads.**
- `tests/test_resume.py`, `test_a_job_stopped_after_a_checkpoint_and_run_again_writes_the_unbroken_jobs_record` (slow), is exposed (READ lines 570–665).
  - It runs `job(..., threads=2)`, so `--threads 2`, at the tiny start. Its SDPA inputs are (2,2,1024,8): TOPCODE's shape, the topcode rows above.
  - It runs a whole job of 4 steps, then a cut job of 2 and a continued one of 2, and compares `output_sha256` and `e_sha256` bit for bit. That is about 8 forward and 8 backward fused calls plus the scoring forwards, so a wrong tile in any of them fails it.
  - At 2 threads I saw 0 of 15,000 calls on Intel (RAN). With the rule of three, the rate is below about 2 in 10,000 a call, so the test fails in under about 1% of runs on an AMX machine and in none seen on AMD (INFERRED). That is not zero in principle, and at 4 or 8 threads on AMX it would fail often (16 of 2,000 backward at 4).
  - TOPCODE's step-2 divergence was at the VM's default thread count, 4 (INFERRED from the VM's cores).
- `tests/test_run.py` near `run.THREADS, []` (lines 781–795) cannot fail on a tile. It monkeypatches `inside` and `outside` to record the thread counts and computes nothing (READ).
- `tests/test_sandbox.py:167` asserts the argv only (READ).
- Every other test runs at 1 thread: `tests/conftest.py`'s `one_thread` patches `run.THREADS` to 1 for the session (READ lines 40–62).

The options (point 3) follow by Fri 8 AM. The step timing on a node has already run (s-t20 shape, generated tokens, because no CPU node holds the starts or the data: RAN `r20261113-021849-402a`).

