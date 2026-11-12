[2026-11-12T21:37Z] 85f23212aaa7
PLAN to training, Thu 12 Nov 1:40 PM PST. **The longest lease is point 7's, about 61.7 hours, not 47.3.** This matters for your INF-762 ask today.

**Point 7 (topcode, one search run at 238M 2x).** At the plan's rate its steps take 38.37 hours (READ: `plan.json`, 38.45 GPU-hours less 283 s of start). Under your rule 1: × 1.3, plus the start and a trace of about 11 minutes, is 50.1 hours, so 51 hours, `minutes = 3060`. The lease is `ceil(1.2 × 3060) + 30` = 3,702 minutes, 61.7 hours (READ: `run.py`, `run_minutes`). Its worker writes that config, but the window infra must admit is this one.

**The others, by the same rule:** point 1, 30.5 hours; point 3, 47.3; point 4, about 47.3; point 5, 8.9. Point 2's comes when its trace runs (queued on vy-nebius-cpu-10 behind memory).

**One change on my branch:** I hold only the seed-102 config in `of_record.toml`. `of_record` lists a step count other than the record's as a difference (READ: `of_record.py`), so the 3,884- and 40,326-step configs can't be held to records of 2,428 and 5,383 steps. Their source configs stay held to the same records, and differ from them only in `budget`.

**Topcode at 238M needs a depth too:** `--grow 10 --cut 18:nq4b4` is absolute, and the `n` code needs the cut at the grown top. 81M grew 8 to 18 (× 2.25) and 36M 6 to 12 (× 2). My default is 81M's ratio, as for grow-depth: 36 blocks, the cut at 36. It goes in topcode's ten lines with the draft unless you rule otherwise.

