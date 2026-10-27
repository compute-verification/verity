import Properties.Pous.Period
import Definitions.Pous.P2.Mutable

/-!
# PoUS as an obligation over one audit window: the deployed check's acceptance certifies the snapshot's space

Under PoUS as an obligation the device's memory holds the weights' P2 encoding while the prover serves, each block under
the key pair of its tag, its coin drawn by the memory challenger, and a block re-encoded later under a fresh tag
(`Security/Definitions/Pous/P2/Mutable.lean`). One audit window runs one session against the store's snapshot at
the window's start, so the window is `PeriodHeld`'s period with that snapshot as `C*`. The store's oracle is the ring's
labeling and a key pair per tag. The window's capacity coins are the snapshot's primitive (`Mutable.snap`) and the
memory challenger's coins. The key pairs of every other tag (`Mutable.rest`: earlier versions, pages outside the
snapshot) were drawn before the window, like the other audits' coins, and setup's inputs and outputs, the payload and
the server may depend on them.

* `ObligationHeld`, **about the deployed code**: for every injective snapshot of the store's tags and every value of the
  other audits' coins, over the store's whole oracle and the memory challenger's coins, the verifier's whole decision
  accepts the window with no break in its own data while the capacity account held fewer than `S · len` bit-ticks, with
  probability at most `δ + ε_max` (`1% + 2^-128`).

It rests on `PeriodHeld`'s premises at the snapshot, `RingDevice` among them, with the other tags' keys read as part of
the other coins. The weights are content fixed before the ring is drawn; content that depends on the ring itself (the
labeling, or keys of blocks committed earlier) is outside it. `PeriodCapacity` gives the joint's capacity guarantee at
each value of the other tags' keys as it stands.
-/

namespace Pous.Period

open Pous.EndToEnd PousCheck PousCheck.Properties
open scoped ENNReal

set_option exponentiation.threshold 20000

/-- **PoUS's per-window obligation guarantee, about the deployed check.** Given `PeriodHeld`'s premises
(`CoinCommitHides`, `P2RingAuditSecure` and `RingDevice` with the other tags' keys among the other audits' coins): for
every injective snapshot `tag` of `2^19` blocks into the store's tags `T`, every joint run and audit `P` of the window,
every certificate `S` with `S · len + S_chip + ∑ cap_i · round ≤ S₀` and every value of the other audits' coins, the
verifier's whole decision accepts the window with no break in its own data while the capacity account held fewer than
`S · len` bit-ticks with probability at most `δ + ε_max` over the store's oracle (the labeling and every tag's key pair)
and the memory challenger's coins. -/
def ObligationHeld : Prop :=
  Pous.P2.Assumptions.CoinCommitHides coinView ((2⁻¹ : ℝ≥0∞) ^ 131) →
    ∀ D Q L : ℕ, Pous.P2.Assumptions.P2RingAuditSecure D Q L →
      ∀ (ι α ΩW ΩQ T : Type) [Fintype ι] [Fintype α] [Fintype T] [DecidableEq T]
        (tag : Fin (2 ^ 19) → T), Function.Injective tag →
        ∀ (d : Accounting.Device ι)
          (r : Accounting.Run d α ΩW (ΩQ × Mutable.RestKeys tag 16448 (2 ^ 16448 - 21065))
            ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams))
          (P : Audits ΩW (ΩQ × Mutable.RestKeys tag 16448 (2 ^ 16448 - 21065))) (round : ℕ) (S : ℝ),
          RingDevice r P D Q L round →
            S * (r.win .capacity).len + d.chip + ((∑ i, d.cap i * round : ℕ) : ℝ) ≤ Pous.EndToEnd.S →
              ∀ o : ΩW × ΩQ,
                Pouw.pr (fun y : Mutable.Oracle 16448 (2 ^ 16448 - 21065) T × HmCoins × Streams =>
                    P.Accepts (o.1, (o.2, Mutable.rest tag y.1), (Mutable.snap tag y.1, y.2)) ∧
                      ¬ P.Break (o.1, (o.2, Mutable.rest tag y.1), (Mutable.snap tag y.1, y.2)) ∧
                        (r.held (o.1, (o.2, Mutable.rest tag y.1), (Mutable.snap tag y.1, y.2)) : ℝ) <
                          S * (r.win .capacity).len) ≤
                  (Params.δ : ℝ) + Params.εMax.toReal

end Pous.Period
