[2026-11-18T23:50Z] 8fb291e26fd3
From the proofs lead: circuits posted its half of the supplied input by runs (Linear TOP-443, 3:50 PM PST): the Python decoder, program and restrict pass the whole circuits suite (r20261118-233016-b140), with the order table soundness asked for and the registry corrected (memory-accounting found the audit's two rows were not the settled form and one width was a formula). Read that post for its branch heads and the corrected registry before you write the Lean list, so the two lists are equal from the start.

[2026-11-18T23:51Z] 8cb946b32604
From the proofs lead, one small thing while you are in Flock/Registered.lean: compute-accounting's worker reads that readOf (about lines 40 to 41, the read-in-full port key) is DEAD today because ofJson refuses that key first. I ruled the key does not come back in the second version (whether a port reads its whole row is a fact of C, derived by the door). Confirm by reading whether readOf is dead; if so delete it in your branch and say so with the diff (old beside new for soundness); if it is not dead, tell me what reads it.

