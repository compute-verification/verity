[2026-11-12T21:03Z] 5cf349f25f32
PLAN to training, Thu 12 Nov 1:20 PM PST. First word on the config job: items 2, 3 and 5 as I read them, and one finding that changes item 5's premise.

**Item 5 changes the plan: under 60-minute leases, points 1, 2 and 3 never finish.** A job keeps a state only at its checkpoints, 20 of them, one every 5% of its steps (READ: `training_slowdown/run.py`, `CHECKPOINTS` and the module's docstring). A stop leaves the state of the last checkpoint and nothing newer. At the plan's limits and rates, 5% of the steps take 54.5 minutes at point 1, 76.7 at point 2 and 89.0 at point 3 (RAN: arithmetic on the plan's `cap_gates` and rates). A lease also builds the environment, loads the start and fetches the state. So no lease reaches its next checkpoint, and every lease starts again from the same state. Points 5 and 6 take 14.2 and 6.5 minutes a checkpoint and do fit. I am writing the configs anyway, as you asked. Your call is whether these three run under longer leases (`budget.minutes`, or a machine named with `--on`), or whether a job should keep a state more often.

**Item 5 (a), a lease that ends inside the trace: the job is stopped, not failed.** It continues from its 95% checkpoint (READ: `training_slowdown/run.py`, `outside`). The job's TERM handler stays installed until the route's process exits, and the trace runs inside that process. So `stop()` writes the stop record, ends the trace and leaves the state kept at checkpoint 19. The next lease runs the last 5% of the steps again, the last checkpoint and the trace. At 656M that stretch is about 55 minutes of steps plus a 24-minute trace, more than a lease. So a 656M job would never finish, even if every earlier stretch fit. The 24 minutes is the library's CPU trace; a job's trace on a GPU node is not measured.

**Item 5 (b), an eviction before the job is up: decided in `store_state` (READ: `benchmarks/training_slowdown/run.py`).** A latest run that ended non-zero is stopped only if its record's `run.json` says `stopped: "told to stop"`. Otherwise it is failed, and `main` never runs a failed job again. Only the job's own TERM handler writes that record, and it is installed after the job's imports. The shell's `trap : TERM` writes nothing. So a stop during `uv sync`, during the state's fetch or during Python's start leaves no record, and the job fails.

The smallest change is one line in `research_argv`: the shell's TERM trap writes `stop_record(seed, split)` to `$OUT/run.json` when no `run.json` is there yet. `store_state` then reads the run as stopped and continues it, and a test in `tests/test_job_shell.py` holds it.

There is one caveat. `store_state` reads only the latest run, and a run stopped during the fetch has no whole state in its record. Such a job continues from θ, and its progress is lost. Keeping that progress takes two more small changes: fetch into a scratch directory and rename it into `$OUT/state` once whole, and have `store_state` take the previous run's state when the latest kept none.

Two smaller facts. The kind's `restart = "requeue"` is parsed and read by nothing (READ: grep of `infra/cluster/src` and `tools/research/src`). On the queue's planner, an ordinary job evicts a preemptible allocation only after its grace, 40 minutes for a 60-minute job; a quiet job or a node stop evicts at once (READ: `infra/cluster/src/cluster/plan.py`, `in_grace`).

**Item 2, point 6: the two finished final seeds ran 1,805 steps, not the plan's limit of 2,256.** I am holding point 6 until you answer. The plan doesn't name the finals' runs. It gives the search run r20261103-132106-82a6 and "2 of 3 final seeds, sd 0", and the overlay row says "p over seeds [1, 1]" (READ: `plan.json` point 6, `overlay.json`). A p of 1 means each seed first reached 1.1x at exactly H, the search run's count at its step 1804. Under the warmup-and-decay schedule, a longer job would put its checkpoints elsewhere, so both seeds ran the 1,805-step schedule of the config on `main`. The plan's limit, 1.25 H, is 2,256 steps. A seed 102 at 2,256 steps would be a different job from the other two and would not make H three-seeded. My default, unless you say otherwise by 3 PM PST: 1,805 steps, final seed 102 alone. Its label is then the existing config's final/102 job.

**Item 3, the shallow points: the plan states a rule, and I am applying it.** The plan says each run is "the ported record's argv at the size" (READ: `PLAN.md`, the run list), and every shallow run names r20261104-061136-1a68 (READ: `plan.json` runs). That record names its blocks from the top, `--full t3.*,t2.*,t1.*,t0.*,final`, where `t0` is the top block (READ: the record's job.json in the plan's `records.tgz`, its block flag alone). So both sizes train the top four blocks and the final gain:
- 656M (21 layers): `blocks.17` to `blocks.20` and `norm`.
- 238M (16 layers): `blocks.12` to `blocks.15` and `norm`.
The other dials are those of `m-t20-1.1x-shallow-depthshare-4`, which `of_record.toml` holds to that run. The second port, 1b6b, trains every other block from the top; no run of the plan names it.

**One more thing about item 1: the command counts a step's gates in other units than the plan.** I derive every step count in the plan's own units. The plan's gates are its records' MACs at the bf16 GEMM price, 1,480.7 gates a MAC: 656M's record, \(2.0622\times10^{18}\) MACs over 2,428 steps, gives \(1.2576\times10^{18}\) gates a step against the plan's \(1.2578\times10^{18}\). The command's trace prices attention's products in float32, at 12,637 gates a MAC: 12.6% of the MACs at 238M and 9.2% at 656M. That makes its step 1.93 times the plan's at 238M (\(4.540\times10^{17}\) against \(2.354\times10^{17}\)) and 1.69 times at 656M (RAN by RESUME, r20261112-174639-330a's `library.json`).

Dividing the plan's limit by the command's count would roughly halve the plan's steps. So each step count is the limit over the ported record's own count:
- Point 1: 1.60002 × 2,428 = 3,884 steps.
- Point 3: 7.49142 × 5,383 = 40,326 steps.
- Points 2 and 5 have no record at their size. Their step is the command's traced MACs at the bf16 price, which I am tracing now on a CPU node.

**A correction to my report.** The 111.8 GPU-hours I called the command's estimate was my own arithmetic at the records' rates. The command prints no estimate. The table's GPU-hour column will be mine, with its basis stated.

**A slip.** Reading one run's job.json, I printed its `store.params` once, which holds argv and part of env. No secret was in the printed part. Since then I read named keys only.

**Next.** I am writing the configs for points 1, 2, 3 and 5 and their honest configs on a new branch off `main`, then running the shallow traces and every dry run on a CPU node with `--on auto`. The draft follows by Fri 13 Nov noon PST.

