import Properties.Pous.Period
import Definitions.Accounting.JointBridge.SpaceOfPeriodHeld

/-!
# PoUS's per-period guarantee as the coins model's `.pous` input

`SpaceOfPeriodHeld` states that a period realizing its PoUS audit as a joint run's capacity audit
(`Accounting.RealizesSpaceOfPeriodHeld`, under the verifier's whole decision outside a break) has the coins model's per-period
guarantee for `.pous` (`Accounting.PeriodRun`), `PeriodRun.Sound c p .pous δ`, at PoUS's own error
`δ + ε_max` (`1% + 2^-128`), under `Pous.Period.PeriodHeld`'s premises. It holds at the challenge times PoUS grades
until `Pous.Guarantees.ContinuousAudit` is tied to the deployed check (`Definitions.Accounting.JointBridge.SpaceOfPeriodHeld`'s
header). `Accounting.SecurityProofs.Bridge.SpaceOfPeriodHeld` proves it from `Pous.Period.SecurityProofs.periodHeld`.
-/

namespace Accounting.Bridge

open Pous Pous.EndToEnd PousCheck PousCheck.Properties Pous.Period
open scoped ENNReal

set_option exponentiation.threshold 20000

/-- **PoUS's per-period guarantee at a period that realizes it.** Given `PeriodHeld`'s premises (the setup check's
`AcceptsDecodes` and `PublishesRoot`, `CoinCommitHides` and `P2RingAuditSecure` at `(D, Q, L)`): for every period `r`
that realizes its PoUS audit as the capacity audit of a joint run `run` whose capacity coins are the ring game's, under
the verifier's whole decision outside a break in the period's data (`RealizesSpaceOfPeriodHeld`), every prover strategy
`τ` of its device under the device model (`DeviceModel`) and the instantiation (`Instantiates`), and every certificate
whose `S · len + S_chip + ∑ cap_i · round` fits the ring game's state bound `S₀`, PoUS's audit of the period has
`PeriodRun.Sound` at error `δ + ε_max`. -/
def SpaceOfPeriodHeld : Prop :=
  AcceptsDecodes → PublishesRoot →
    Pous.P2.Assumptions.CoinCommitHides coinView ((2⁻¹ : ℝ≥0∞) ^ 131) →
      ∀ D Q L : ℕ, Pous.P2.Assumptions.P2RingAuditSecure D Q L →
        ∀ (ι α : Type) [Fintype ι] [Fintype α] (d : Device ι) (C : Verity.Circuit) (I : Type) [DecidableEq I]
          (Ω : I → Type) [∀ i, Fintype (Ω i)] (r : PeriodRun d α C Ω) (ΩW ΩQ : Type)
          (run : Run d α ΩW ΩQ ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams))
          (P : Audits ΩW ΩQ) (τ : Strategy ΩW ΩQ) (round : ℕ) (c : Certificates) (p : Price),
          RealizesSpaceOfPeriodHeld r run (fun ω => P.Accepts ω ∧ ¬ P.Break ω) →
            DeviceModel run P τ D Q L round → Instantiates P τ →
              c.S * (run.win .capacity).len + d.chip + ((∑ i, d.cap i * round : ℕ) : ℝ) ≤ Pous.EndToEnd.S →
                r.Sound c p .pous ((Params.δ : ℝ) + Params.εMax.toReal)

end Accounting.Bridge
