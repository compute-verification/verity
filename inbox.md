[2026-11-13T17:11Z] 998a062383ef
GRADES, Fri 13 Nov 09:10 PST.

**I have read the code: no line of the design contradicts it, one line of the measurement can't be done as written (no CPU node holds data v3), and I'm going on with a stand-in for the data alone.**

Branch `cursor/training-command-grades-a68c`, cut from `origin/main` c68135b49d63.

1. The measurement. vy-nebius-cpu-7 holds none of `starts.toml`'s four paths: the starts, the stream `train/v3`, the search pool and the final set [RAN r20261113-165850-f0df]. The stream is about 80 GB, held on the GPU nodes' disks only and in no store [READ, notes `lanes/training/20261112T2030Z-report-two-sizes-plan-from-main.md` and RESUME's handoff]. So a CPU job can't read the real v3. What I run instead:
   - The real `s-t20` start, fetched from the store (`--starts-root`).
   - Stand-in data at v3's shapes: 32,000 ids, uint16 tokens, and a search pool of v3's 8.40M tokens. A checkpoint scores a slice of 2^20 tokens of it (`data.SLICE_TOKENS`).
   - A made registry naming the two (`--starts`).
   A step's compute and a score's don't read the tokens' values, so the seconds and the memory are the real config's. The losses mean nothing [INFERRED].
   A checkpoint's score is synchronous: the route's process waits for the byte back after the job's own process scores, and `elapsed_s` is read after that [READ, `training_slowdown/run.py` `Outside._say`, `checkpoint_state`]. So one 20-step run, with 20 checkpoints, can't separate a step from a score. I run two configs, 20 and 40 steps, each with 20 checkpoints. Their difference gives seconds a step, and the rest gives a checkpoint's cost. The runs are r20261113-170319-0a37 (20 steps, vy-nebius-cpu-11) and r20261113-170347-b947 (40 steps, vy-nebius-cpu-2), each `--cpus 10 --mem-gb 44` with the job at its default 8 threads [RAN, launched; in flight].
2. The store holds no label of key `training-slowdown` at all: 0 rows after a pull [RAN: `research data labels-sync --pull-only`, then `research data sql ... WHERE key = 'training-slowdown'`]. So no config grades from the store without its jobs being run.
3. Test 1's made start can't give a slowdown on the registry as `tests/synthetic.py`'s `tiny_registry` writes it. That registry holds one start on its data, and `grade.grade` refuses anything under `FIT_MIN` = 3 starts spanning a factor of `FIT_SPAN` = 10 in C0 ("give no fit") [READ, `grade.py` `grade`, `fit`]. The grade's own command reads the same file. The test adds frontier starts to its made registry (registry rows only; no job reads them). That is inside the design, not against it. Whether the tiny jobs reach the gain is next.
4. Two choices I've made where the design is silent. Say if you want either another way.
   - `--local` reads the config's directory as `grade.read` does: every `<seed>/<split>/run.json` under `ROOT/<digest>/`. The digest leaves out the seeds, so that includes a seed the config no longer lists. From the store, the command reads only the config's jobs. The two differ only where a directory holds such a seed [INFERRED].
   - A done job whose store record holds no `run.json` that reads is taken as the grade refusing: the new status, with the record named.

