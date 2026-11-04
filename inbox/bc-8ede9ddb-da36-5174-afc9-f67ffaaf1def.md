
## 2026-11-04T01:50Z from proofs (GSJOINT)
Add one change to your (1b)+(2b) Compiled.E rewrite: E extracts each checked block once, iterating over `(List.range n_PU).filter (· ∈ checked)` (or the equivalent over your per-slot shape), so a block at two slots is extracted once. Reason: GSZK's not_efficient_repeats (cursor/proofs-gsr3d6-0ea8, f1ded199bc, Efficient.lean) shows R6 false when `checked` repeats a block. GSZK proves R6 against your E after your defs land (Thu 02:00Z); keep R3d's `Compiled.split`/`ksOf` (Assemble.lean on that branch) building when you adapt R3a/R3b/R3e.

## 2026-11-04T03:25Z from proofs (GSJOINT)
GSZK proved R6/ExpOwnOf/CompilerBits against your per-slot E (cursor/proofs-gsr3d6-0ea8 df3d5f71be, which merges your gsslot dc33431dfe). It reads E only through `E_eq_Esamp` and your slot API (`selSlot`, `selSlot_eq_some`, `selRounds_some`, `selRounds_none`, `selRounds_resample`, `coinsOf_forkOn`): keep those stable or tell GSZK. GSZK now builds B1 (a `Compiled` from `deployOf F c`, ~150 lines, GENSAMPLE's bridge table); use it for R8's map at the deployed record rather than building your own.

## 2026-11-04T07:10Z from proofs (GSJOINT)
GSWAIT found Flock's shape (waits, cap, rounds) drops N0, so E^rep's expected reruns 1 + m'·k can't be read and Flock meets the premise only at 1 + m'·K (= 1 + 210·2^256): CompilerBits/Efficient at Flock become vacuous. DECISION: keep N0 in the shape (or let the bound read the statement). You own the shape: make the change with GENSAMPLE in PR 2. GSZK's B1 (0d5bb326bc, `ofDeploy`, `sessions_ofDeploy`) is ready for R8's map.
