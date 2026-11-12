[2026-11-12T23:20Z] 4b99c0e84d82
GROW, Thu 12 Nov 3:19 PM PST

**The draft is open as [#9531](https://github.com/compute-verification/verity/pull/9531), head 9822461a75. It holds the route, the harness change as ruled, the 81M config (which of_record now calls "equal"), and point 4's config with its trace and dry run.**

**Configs**

- **81M config: equal to its record.** RAN r20261112-220601-7dad. `of_record m-t20-2x-hybrid-grow-depth` prints "equal, 1 expected difference, 1 named line of the record's program alone". The step it read is the config's step 3, as `of_record.toml` states. `STEPS` stays 2.
  - The named line is the running loss's `mul`, counted before naming it. READ the record's `train.py`: `total += ce.detach() * part` is one scalar `mul` per forward, 4 per step, beside the 4 of `(loss * part)` that the route has too. That accounts for the record's 8 against the config's 4.
  - So it is `record_only` with count 4, macs 4, and nothing else differs.
- **Point 4 steps.** RAN r20261112-220636-c204. The route's own trace at 238M, depth 24, batch 192, has 234,904,646,320,128 MACs a step, so ⌊8.49e21 / (MACs × 1480.7)⌋ = 24,408 steps.
  - The same trace's circuit counts 6.82e17 gates a step, about 2,900 gates a MAC. If the limit is meant against circuit gates, steps would be about 12,456. I followed your rule (MACs × 1480.7).
  - The trace took 244 s on a CPU node.
- **Point 4 minutes.** `budget.minutes = 2340`: 29.2 h × 1.3 = 37.96 h, plus the start and the trace, rounded up to 39 h.
- **Point 4 dials** are the 81M config's, at 192 rows in 6 forwards of 32, as the other l16 entries use (INFERRED choice; say if you want other dials).
- **Dry run.** RAN r20261112-221307-23e7, `verity bench training-slowdown CONFIG --dry-run` on a CPU node:
  - The 81M config lists 4 jobs, all missing, digest 160c3577f54ebd06.
  - The 238M config refuses: "config .../attacks/l16-t20-2x-fullmuon-fe-wsd-40326.toml: no such file", which is PLAN's honest config.
  - With the honest line dropped in the run's clone, it lists 4 jobs at `--disk-gb 29 --max-min 2838`, digest 727a49d568fd62b2. The digest doesn't read `honest`.

**Your conditions**

- **Condition 3: I took the Muon option** (`stack`, off by default, its key only when on), because of_record has no rule for one line at another shape. The RAN proof that no other entry's count moves is in flight as r20261112-224717-24bd. It counts all 44 other entries at a tiny shape under base 5f0c008aa0 and under this head and compares them. An earlier attempt's head half finished clean, and its base half failed on a path outside the package, now fixed.

**Suites**

- **`suites.py benchmarks/training_slowdown`**, slow tests included, at 9822461a75. RAN r20261112-221738-8a4a: "passed benchmarks/training_slowdown 593 passed in 552 s"; "1 suites: 1 passed (0 cached), 0 failed".
- **Every changed suite** at 8d1412f7cf is in flight as r20261112-215110-fb6f, and passing so far.

**Other**

- **Goal warning.** `research queue open` warned that `Goal training/slowdown-across-scale` names a goal no label of TRN-42 has; TRN-42's live goals are training/second-swarm-results and training/stopped-job-resumes. I kept the line you gave. Say if it should change.
- **Merging main.** `main` hasn't moved since 5f0c008aa0, so there is nothing to merge yet.

**Left, in order**

1. The comparison's result into the body.
2. The full suites' last lines.
3. Merge main again before ready, taking the higher VERSION against #9039.
4. The handoff note.

**Runs in flight:** r20261112-224717-24bd (the comparison) and r20261112-215110-fb6f (all changed suites).

