[2026-11-16T23:57Z] f985e448e531
From ci, one more part for your SECOND change (vllm's row 14, TOP-402 3:43 PM): unmet() lists the demo file's [program] commit as unmet because it isn't the running tree's (run.py:247-248), so every demo run is 'simulated' for that alone. For verity demo NAME, the commit that runs is the running tree's: run the file with the tree's HEAD as its program commit and have the record name both the file's hash and the commit that ran. verity run on a deployment's own file keeps naming a mismatch as unmet. Test both. Same PR, same day (pushed by Tue 10 AM PST).

