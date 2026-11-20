module

public import Proofs.Spot.Sampling.RegScheme
public import Proofs.Spot.Sampling.AmountRate
public import Proofs.Core.Scheme

/-!
# What a scheme gets back at the rate of a design the prover registers

`getsBack_auditReg_rate` chains the audit bound at the registered design (`givesAuditBound_auditReg`), read along the
map from the scheme's runs to the audit's outcomes (`AuditBound.comap`), the accounting layer's theorem
(`Scheme.accountingLayer`) and the missable cost at a rate (`Scheme.getsBack_rate`).
-/

@[expose] public section

namespace Spot.Sampling

open Verity Verity.Game
open scoped ENNReal

/-- **What a scheme gets back at the rate of the design its prover registers.** A prover `σ` of `auditReg L session`
registers its design `L σ.1` before the coin; every design it can register meets the rate `NUM/DEN` at the scheme's
costs. The scheme's runs `out` to the audit's outcomes, with `pr` the audit's law at `σ`; an accepted run is an
accepted outcome, and its incorrect blocks are among those of `σ`'s committed transcript under the analysis at
`L σ.1`. Then at `δ > 0`, an accepted run whose log meets the conditions has `over` above `under` by more than
`⌊ln(1/δ) · DEN/NUM⌋` plus the charges with probability at most `δ`, plus the audit's error at `σ`, plus the scheme's
term. -/
theorem getsBack_auditReg_rate (S : Scheme) [DecidableEq S.Block] {Reg : Type}
    (L : Reg → Verity.SamplingDesign S.Block) (session : Finset S.Block → Reg → Game Bool) (h : GivesAuditBound)
    (A : (R₀ : Reg) → ExtractionAnalysis (L R₀) Reg session)
    (εks εbind : (R₀ : Reg) → (R : Reg) → Cont (L R₀) Reg session R → ℝ≥0∞)
    (hks : ∀ R₀, (A R₀).KnowledgeSound (εks R₀)) (hbind : ∀ R₀, (A R₀).BindSound (εbind R₀))
    (num den : ℕ) (hnum : 0 < num) (hden : 0 < den)
    (hrate : ∀ (R : Reg) (T : Finset S.Block),
      (L R).missProb T ≤ ENNReal.ofReal (Real.exp (-∑ b ∈ T, (S.cost b : ℝ) * num / den)))
    (σ : Strategy (Pouw.Window.auditReg L session)) (pr : (S.Run → Prop) → ℝ≥0∞) (out : S.Run → Bool × Finset S.Block)
    (hpr : ∀ p, pr (fun r => p (out r)) = prob p (Pouw.Window.auditReg L session) σ)
    (hmono : ∀ p q : S.Run → Prop, (∀ r, p r → q r) → pr p ≤ pr q)
    (hunion : ∀ p q : S.Run → Prop, pr (fun r => p r ∨ q r) ≤ pr p + pr q)
    (hacc : ∀ r, S.accepts r → (out r).1 = true)
    (hinc : ∀ r, S.incorrect r ⊆ (A σ.1).incorrect ((A σ.1).committedOf (atRegOf σ)))
    (hbr : pr S.breaks ≤ S.δscheme) {δ : ℝ≥0∞} (hδ : 0 < δ) :
    pr (fun r => S.accepts r ∧ S.logOK r ∧ S.under r + ⌊Real.log δ.toReal⁻¹ * den / num⌋₊ + S.charges < S.over r) ≤
      δ + (εks σ.1 (reg (atRegOf σ)) (cont (atRegOf σ)) + εbind σ.1 (reg (atRegOf σ)) (cont (atRegOf σ))) +
        S.δscheme := by
  have hg := givesAuditBound_auditReg h A εks εbind hks hbind σ
  have hlaw : (fun p => prob p (Pouw.Window.auditReg L session) σ) = fun p => pr fun r => p (out r) :=
    funext fun p => (hpr p).symm
  rw [hlaw, AuditBound.comap] at hg
  have haud : AuditBound (fun _ => L σ.1) pr S.accepts S.incorrect
      (εks σ.1 (reg (atRegOf σ)) (cont (atRegOf σ)) + εbind σ.1 (reg (atRegOf σ)) (cont (atRegOf σ))) := by
    intro δ'
    refine (hmono _ _ fun r ⟨ha, hm⟩ => ⟨hacc r ha, le_trans ?_ hm⟩).trans (hg δ')
    unfold Verity.SamplingDesign.missProb
    exact prCoin_mono fun ω hω => Finset.disjoint_of_subset_right (hinc r) hω
  exact Scheme.getsBack_rate S (fun _ => L σ.1) pr hmono num den hnum hden (fun _ T => hrate σ.1 T) hδ
    (Scheme.accountingLayer S (fun _ => L σ.1) pr _ hmono hunion haud hbr δ)

end Spot.Sampling
