[2026-11-12T19:09Z] 97c776bccfa8
PLAN, Thu 12 Nov 11:10 AM PST.

**Run D failed, and the branch's message had a bug; fixed, rerunning.** D's copy, r20261112-184612-107b (vy-nebius-cpu-7, tree b098faf2f87b, AMD EPYC-Genoa, no AMX), ran first and exited 0, but both of its parts failed [RAN, its `stdout.log`]:
- Part A, the suite's runner by the command the new failure message names: `suites.py training_slowdown` refused itself, "matches 2 suites: benchmarks/training_slowdown/grader, benchmarks/training_slowdown". So the message told a reader to run a command that doesn't run. `suites.py` takes an exact path first, so `benchmarks/training_slowdown` names the suite alone [READ, `select` in `tools/verity/check/suites.py`].
- Part B, the scratch check: `uv run pytest` in `benchmarks/training_slowdown`, which is not a workspace member, so uv made a separate environment with no pytest. The scratch script was at fault, not the branch. It also left a `.venv` in vy-nebius-cpu-7's shipped tree of b098faf2f87b, which no other run uses.

**Fix:** a second commit on the branch, 177b3db24a, pushed. It changes the message to `suites.py benchmarks/training_slowdown`, one line, and ruff passes. The draft's head will be 177b3db24a, under TRN-75.

**Rerun:** r20261112-190853-3b46 (vy-nebius-cpu-7, expected to start about 11:36 AM PST). It runs the message's command on `tests/test_resume.py` with the slow tests, and runs the scratch check with the root environment's Python. I open the draft when it passes.

**Which ids ran:** D's original, r20261112-182200-eeb0, never started. I cancelled it. A's two copies are both still waiting: r20261112-174531-7468 (vy-nebius-cpu-5) and r20261112-184648-ce12 (vy-nebius-cpu-1).

[2026-11-12T19:09Z] 154046b13186
CREDIT to training: both totals are right arithmetic for different words, and 96,256 is the one the bound reads. compute-accounting's 106,496 is the 19 rows' A (input) words, rows x k (its note: "19 rows × their k": 4x2,048 + 2x4x8,192 + 4x2,048 + 3x8,192); TIE's 96,256 is their output words, rows x n (4x16,384 + 2x4x2,048 + 4x2,048 + 3x2,048), and `tied.py` divides uncredited MACs by the fan-in, so it counts output words (MACs/k = rows x n). The difference: down rows have k = 8,192 > n = 2,048 (11 rows x 6,144 = 67,584 more A words) and gate_up rows n = 16,384 > k = 2,048 (4 x 14,336 = 57,344 fewer): net 10,240. READ (the two notes, tied.py on cursor/trn86-period-print-20af); RAN confirmation from 614b's document in run r20261112-184728-1d4a (queued) follows with the draft.

