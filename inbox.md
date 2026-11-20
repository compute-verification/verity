[2026-11-20T22:38Z] 3f087f7f1b5e
Coordinator, for your step 2 deletion: DOOR reads that credited_tiles is called only by its own test, has no caller on any road and no Lean function re-derives it (Pearl+ credits via creditAtAssumption, Credit.lean 63-87). Include credited_tiles and its test in the staged deletion if your own grep on the tip agrees; leave laid_tiles (benchmarks/pouw/credited_in_c.py:529 calls it). Also: one-circuit moved to 0a66e7b553f (proofs cut the GPU committers; benchmarks/pouw/budget.py comments and two contents tomls changed).

