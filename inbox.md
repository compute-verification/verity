[2026-11-18T08:44Z] de74701386eb
From coordinator, two more of the same ci run (r20261118-074045-b857) are ours, add to your task: (C) `verity/protocols/experimental/pearl_c4`: 2 failing in test_pouw_pearl_c4.py, "the volunteer rows" (likely from the salts change: say what the volunteer rows should be now that salts are data handed in, mend test or code accordingly); (D) `verity/protocols/experimental/dither` fails only the read guard: its tests read `verity/hooks.py` undeclared: declare the smallest input. Same return, 1:50 AM PST.

