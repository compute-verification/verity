[2026-11-13T09:03Z] 3becf8e8ea76
One more thing for task 2 (13:00Z), from memory-accounting on TOP-415 at 12:27 AM PST, addressed to proofs:

- **Their finding:** no accepted memory answer accounts for a registered part today. Soundness's default for a part no protocol accounts for is to charge it at its registered capacity, never at 0. For memory held that is 8 times the leaf's registered bytes, and for memory read at HBM the device sheet's rate times the period.
- **The gap they hand us:** memory read at DRAM or disk has no number. They say the Lean `Site` holds no rate for a leaf under a host, while the Python node files do (`verity/hardware/nodes/*.toml`). "Registration needs a read rate for a host's memory leaves, or those rows are charged at nothing." Host CPU compute is the same gap.
- **For your plan of the split:** check their two statements against the tree (which fields `Site` has for a leaf under a host; which rates the node files hold), and say what the registration would have to carry so that every registered leaf has a capacity the account can charge: the field, its unit, where it is read from, and whether adding it changes a listed guarantee's statement or a lock record. If it does, that part is under the STOP RULE heading, not in the plan as decided.

Mark each as [read] with file and line. No code for this tonight.

