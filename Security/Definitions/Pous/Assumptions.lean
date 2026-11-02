import Definitions.Pous.Accounting.LemmaA
import Definitions.Pous.Model.Inc
import Definitions.Pous.Assumptions.P2

/-!
# Named assumptions

Each is a `Prop` taken as a hypothesis by the statements that use it, never an `axiom`.
-/

namespace Pous.Assumptions

open scoped ENNReal

/-- **Conjecture B1′** (tradeoff §3), as a property of a scheme: its sequential audit (an `S`-bit state at each
challenge, `(D, Q)` per answer) obeys the M1 completion bound with a domain of `Mdom` points. For every `g` and `λ'`
with `|Hist| · Γ_M1(g) ≤ 2^-λ'`, `Pr[pass] ≤ ((g − 1)/B)^k + 2^-λ'`.

For the M1 scheme with `Mdom = 2^ℓ` it follows from A2 and A3′ (`Guarantees.A2LemmaAM1`, `Guarantees.A3PrimeLam` with
`G = True`). For a scheme whose blocks share a primitive (models M2/M3, RW–SMS) it is the conjecture that the sharing
costs nothing beyond M1's bound. -/
def B1Prime {M : IdealModel} (sch : Scheme M) (Mdom S D Q k : ℕ) : Prop :=
  ∀ g lam : ℕ, 1 ≤ g → g ≤ sch.B →
    (histCount sch.B k : ℝ≥0∞) * gammaM1 sch.B S Q g Mdom ≤ 2⁻¹ ^ lam →
    AuditSecure sch .sequential S D Q k ((((g - 1 : ℕ) : ℝ≥0∞) / sch.B) ^ k) (2⁻¹ ^ lam)

/-- **Lemma A as a hypothesis** (the hypothesis of `A3Prime` and `A3PrimeLam`,
`Security/Proofs/Pous/TrackA/A3Prime.lean`) (sms5 `tradeoff.md` Lemma A in its per-run form, review F4). On the
good setups `G W` (the review's `Dist` for M2/M3; `True` for M1), every `S`-bit state and every per-block responder that
spends at most `(D, Q)` per run answers at least `g` blocks correctly with probability at most `Γ`. -/
def LemmaA {M : IdealModel} (sch : Scheme M) (S D Q g : ℕ) (Γ : ℝ≥0∞)
    (G : Bits sch.n → M.Ω × sch.Coins → Prop) : Prop :=
  ∀ (W : Bits sch.n) (A₁ : Preprocessor sch S) (A₂ : TimedProdResponder sch S D Q),
    pr (fun x : M.Ω × sch.Coins =>
      let pp := (sch.enc W x.1 x.2).1
      let C := (sch.enc W x.1 x.2).2
      G W x ∧ g ≤ (Finset.univ.filter fun i => (A₂.prog W pp (A₁ W pp C x.1) i).run (M.answer x.1) = C i).card)
      ≤ Γ

end Pous.Assumptions
