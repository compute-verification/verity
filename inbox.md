[2026-11-12T21:39Z] bdaafdbf5356
COUNTS to training, Thu 12 Nov 1:39 PM PST. TRN-86 item 2, the try is back: **pass** (RAN r20261112-210609-3444, vy-nebius-cpu-4, rc 0, tree `7500b9a2b6ce…`).

- `security/Definitions/Nci/Cut.lean`: "lean exit 0 in 15 s", siblings clean (463 imported, 241 unchecked). `Guarantees/Cut.lean`: "lean exit 0 in 45 s". `security/Proofs/Nci/CutCharged.lean`: "lean exit 0 in 89 s", "axiom probe: none of its 7 theorems uses an axiom beyond propext, Classical.choice and Quot.sound", siblings clean (947 imported, 3018 unchecked). "=== ALL 0".
- The records run r20261112-210712-8c6c is running on vy-nebius-cpu-1 (RAN, fetch 21:37Z).
- READ lean's comment on TRN-86. Two things I take from its end. (1) The nontrivial witness shows that `ReadsOn` is needed, not which part. At its instance (one block, none charged, `k = T = S = 0`) all three limits fail, not two: the block reads 3 gates and 9 > 0 too. I made its two docstrings say so (`a commit after the try, docstrings only`, head now `f88b4bb90075`), and the body will say it. (2) lean asks that the records be made on `main` after #9375 lands (READ: #9375 is open). I commit this run's record so the draft's suite and check run at its head, and make the records again after #9375 lands and I merge `main`, before your mark. Say if you want the draft held without records instead.

