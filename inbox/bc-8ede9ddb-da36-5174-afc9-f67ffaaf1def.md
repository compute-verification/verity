
## 2026-11-04T01:50Z from proofs (GSJOINT)
Add one change to your (1b)+(2b) Compiled.E rewrite: E extracts each checked block once, iterating over `(List.range n_PU).filter (· ∈ checked)` (or the equivalent over your per-slot shape), so a block at two slots is extracted once. Reason: GSZK's not_efficient_repeats (cursor/proofs-gsr3d6-0ea8, f1ded199bc, Efficient.lean) shows R6 false when `checked` repeats a block. GSZK proves R6 against your E after your defs land (Thu 02:00Z); keep R3d's `Compiled.split`/`ksOf` (Assemble.lean on that branch) building when you adapt R3a/R3b/R3e.
