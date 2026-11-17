module

public import Properties.Pous.Period
public import Definitions.Pous.P2.Mutable
import Mathlib.Basic.ENNReal.Real

/-! PoUS as an obligation over one audit window (`ObligationHeld`); content that depends on the ring itself is
outside it. -/

@[expose] public section

namespace Pous.Period

open Pous.EndToEnd PousCheck PousCheck.Properties
open scoped ENNReal

set_option exponentiation.threshold 20000

/-- **PoUS's per-window obligation guarantee, about the deployed check.** Given `PeriodHeld`'s premises
(`P2RingAuditSecure`, and `DeviceModel` and `Instantiates` with the other tags' keys among the other audits' coins):
for every injective snapshot `tag` of `2^19` blocks into the store's tags `T`, every joint run and audit `P` of the
window, every prover strategy `τ` of its device, every certificate `S` with `S · len + S_chip + ∑ cap_i · round ≤ S₀`
and every value of the other audits' coins, the verifier's whole decision accepts the window with no break in its own
data while the capacity account held fewer than `S · len` bit-ticks with probability at most `δ + ε_max` over the
store's oracle (the labeling and every tag's key pair) and the memory challenger's coins. -/
def ObligationHeld : Prop :=
  ∀ D Q L : ℕ, Pous.P2.Assumptions.P2RingAuditSecure D Q L →
    ∀ (ι α ΩW ΩQ T : Type) [Fintype ι] [Fintype α] [Fintype T] [DecidableEq T]
      (tag : Fin (2 ^ 19) → T), Function.Injective tag →
      ∀ (d : ResourceAccounting.Device ι)
        (r : ResourceAccounting.Run d α ΩW (ΩQ × Mutable.RestKeys tag 16448 (2 ^ 16448 - 21065))
          ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams))
        (P : Audits ΩW (ΩQ × Mutable.RestKeys tag 16448 (2 ^ 16448 - 21065)))
        (τ : Strategy ΩW (ΩQ × Mutable.RestKeys tag 16448 (2 ^ 16448 - 21065))) (round : ℕ) (S : ℝ),
        DeviceModel r P τ D Q L round → Instantiates P τ →
          S * (r.win .capacity).len + d.chip + ((∑ i, d.cap i * round : ℕ) : ℝ) ≤ Pous.EndToEnd.S →
            ∀ o : ΩW × ΩQ,
              Pouw.pr (fun y : Mutable.Oracle 16448 (2 ^ 16448 - 21065) T × HmCoins × Streams =>
                  P.Accepts (o.1, (o.2, Mutable.rest tag y.1), (Mutable.snap tag y.1, y.2)) ∧
                    ¬ P.Break (o.1, (o.2, Mutable.rest tag y.1), (Mutable.snap tag y.1, y.2)) ∧
                      (r.held (o.1, (o.2, Mutable.rest tag y.1), (Mutable.snap tag y.1, y.2)) : ℝ) <
                        S * (r.win .capacity).len) ≤
                (Params.δ : ℝ) + Params.εMax.toReal

end Pous.Period
