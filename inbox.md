[2026-11-13T21:36Z] 8abe12b98c8c
GRADES, Fri 13 Nov 1:35 PM PST: the 6 PM message, early.

**The command grades its own jobs on both paths and every test the brief asked for is in and passing; one search job of 838 steps is scaled to about 16.5 to 17.5 hours on a CPU node at 53 GiB, and no GPU node's disk holds a root of any of the five configs that name an honest config.**

The change
- Branch `cursor/training-command-grades-a68c`, head `698da49f86e`, pushed: the change (`9d7a6d0ffbd`), `main` merged by hand (`36400e79728`, `main` at `fa3e619ee146`; no restack), and the README lines (`698da49f86e`). RAN.
- All six tests are in `tests/test_command.py`, the unlisted-seed test among them. They are tests 1 and 2, the unlisted seed, store against `--local` with pending, stopped, stalled and failed, a refusal with a dry run, and no honest config. RAN on the VM after the merge: `tests/test_command.py` (27, slow included) and `tests/test_grade.py`, `66 passed`.
- The README is +287 bytes net (46,316 to 46,603). "Running an attack" says the command grades its own jobs at the end and exits 5 where the grade refuses. "Grading a config" says the grade's own command, over whole directories, is for `--also` and `--as-if-assumed`. `run.py`'s docstring says the same, with `REFUSED` = 5. READ.

Checks
- RAN `r20261113-181452-a641` (vy-nebius-cpu-6, fresh, head `698da49f86e`): `4 suites: 4 passed (0 cached), 0 failed; 28.9 min of test time`. The suites: `benchmarks/training_slowdown` 705 passed, its `grader` 28 passed and 3 skipped, `repository` 362 passed, `verity-graph` 246 passed. Then `ruff check benchmarks/training_slowdown`: `All checks passed!`.
- RAN `r20261113-172812-1bae` (vy-nebius-cpu-8, fresh, head `9d7a6d0ffbd`): the same four suites passed. Its exit 1 is my `ruff check .` over the whole tree: 1,252 findings, none under `benchmarks/training_slowdown`.
- RAN on the VM, `ratchet_gate.py origin/main HEAD`: `ratchet gate 07ada1c65dee..698da49f86e4: 1 change(s), 0 failed`. On a node it can't run from a shipped clone: `r20261113-172833-2cab`, `fatal: couldn't find remote ref main`.
- RAN on the VM: a `--local --dry-run` of every config of `attacks/`. All 62 exit 3 with every job pending, and none grades. RAN: `verity bench training-slowdown s-t20-1.1x-shallow-strideshare --local --dry-run` gives `8 of 8 jobs pending`, rc 3.
- READ `attacks/*.toml`: 5 of 62 configs name an honest config.

The measurement
- The setup is a made copy of `attacks/s-t20-1.1x-adamw.toml`, at 20 and 40 steps of batch 64, with search seed 0, `--local --root "$OUT/jobs" --device cpu` and 8 threads, on stand-in data made in each run's scratch. RAN.
  - `r20261113-172313-ecba`, vy-nebius-cpu-1, AMD EPYC-Genoa, 20 steps: 64.8 s a step (mean; median 63.8), 433 s a checkpoint, 9,951 s of job clock at the last checkpoint, peak 52.9 GiB.
  - `r20261113-172342-3e29`, vy-nebius-cpu-8, AMD EPYC-Genoa, 40 steps: 61.6 s a step (mean; median 59.4), 435 s a checkpoint, 11,162 s, peak 53.4 GiB.
  - Both used about 5.5 CPU-seconds per wall second at 8 threads. Peak memory is the run's processes' summed RSS, from the run's `resources.jsonl`.
- The difference between the two gives (11,162 − 9,951) / 20 = 60.6 s a step. Each run's record also states every step's `wall_s`, which gives the same. RAN.
- Scaled from these two runs (INFERRED, not run):
  - One 838-step search job is 838 × 60.6 to 64.8 s plus 20 checkpoints, about 16.5 to 17.5 h.
  - Eight jobs are 132 to 140 h one after another (as `--local` runs them), or about 17.5 h on eight nodes side by side. That assumes a final checkpoint costs what a search one does.
  - A final checkpoint scores the whole final set (READ `evaluate.py`), whose size the tree doesn't state (sealed). At the stand-in's 2.1M tokens, twice a search slice's 2^20, a final job adds about 2.4 h, and the eight are 147 to 155 h.
- `s-t20-1.1x-adamw` names no honest config (READ), so its "honest config" in the eight is assumed to cost what it does. INFERRED.
- The stand-in's token distribution: 32,000 ids (bpe32k's count, `uint16`), each token drawn uniformly from 1..31999, with EOT (id 0) at every 997th position. It has a search pool of 8.4M tokens (v3's), a final set of 2.1M, and four training parts of about 1M. Its tokenizer is a random byte-level one with 2- to 7-byte merges. It's fine for seconds and memory, and its losses mean nothing. Making it took about 45 minutes of each run, which is outside the job's clock and not counted. RAN.
- For a CPU run, INFERRED: `training-attack` asks `phase = "gpu1"` and `max_wall_min = 60` (READ `infra/cluster/kinds/training.toml`). At 838 steps a CPU job's checkpoints are about 50 minutes apart, so a 60-minute run keeps at most one, and a CPU job there would need `budget.minutes` or `--local`. A run needs at least 53 GiB; I'd ask 64.

The store and the GPU disks
- RAN on the VM: after a successful `research data labels-sync --pull-only`, the store holds 0 `training-slowdown` labels, so no labelled record of any whole config.
- Item 5, RAN `r20261113-174355-2bf8` on vy-nebius-2 (host `computeinstance-e05v9sztsjcjsw4hm2`). It went through its `research` check pool, with no GPU, no card and idle priority.
  - It listed every disk (`/` 247 GB, `/workspace` 4.9 TB) except system and container trees. No directory there is named by any of the eight digests today's code gives: the five configs and their three honest configs.
  - None of the node's 20 `run.json` files names one.
  - So no root holds any job of those configs: 0 jobs found of those listed, for each of the five configs and for each of their honest configs.
  - With no records found there is no version to report; the 4.26.0 retrace rule meets nothing. An earlier listing with `-xdev` under four directories agrees (`r20261113-172625-8e1b`).
- vy-nebius-1, the other GPU node in `infra/cluster/descriptions/nebius.toml`, can't be reached: `research run --on vy-nebius-1` is refused with "host None is not a host name or address" (RAN, refused as `r20261113-173029-4dc6`). READ `nebius-after-n1.toml`: node 1 stops.

The PR
- No PR tool is in my session, only read-only GitHub tools, and no PR exists for the branch (RAN `gh pr list --head`).
- The body is ready, starting `Lane training` / `Goal training/attack-search-one-command` and ending `Refs TRN-157`.
- Should I wait for the environment to open the draft from the pushed branch, or will you open it with that body?

Left, in order
1. The draft PR, as above.
2. Rerun the suites if `main` moves under the branch before Sat 6 PM.
3. The handoff note.

Nothing is in flight.

