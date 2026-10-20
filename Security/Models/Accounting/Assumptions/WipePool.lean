import Definitions.Accounting.Actual.WipePool

/-!
# Timed-round soundness, the lifetime pool's premise (trusted)

The one physical premise of the memory cap's lifetime pool (`Accounting.Guarantees.Actual.WipePool`), owned by
memory-accounting: a `Prop` about one run tree of wipes (`Accounting.WipePlay`), taken as a hypothesis, never an
`axiom`. It has no claim id in `verity.core.claims` yet.
-/

namespace Accounting.Assumptions

/-- **Timed-round soundness** (memory-accounting's premise of the lifetime pool): a timed round that challenges a
block the prover does not hold misses its deadline, so the wipe fails. On the run tree `P` it reads: at every wipe,
each draw that passes challenges no block of the wipe's un-held set `U` (`WipePlay.UnheldFails`).

`U` is what the prover cannot serve within a round's deadline, fixed by the history before the wipe's coins, and it
must stay un-held through all `k` of the wipe's rounds: a block fetched or rebuilt within one round's deadline is held
for that round. With the draws uniform on the store, a round then catches the wipe with probability at least its
un-held fraction `x = |U| / |B|`, independently of the other rounds, so the wipe passes with probability at most
`(1 − x)^k ≤ exp(−k x)`. -/
def TimedRoundSound {k : ℕ} (P : WipePlay k) : Prop := P.UnheldFails

end Accounting.Assumptions
