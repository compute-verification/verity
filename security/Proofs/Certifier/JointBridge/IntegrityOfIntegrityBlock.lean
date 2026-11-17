module

public import Proofs.Certifier.JointBridge.Guarantees.IntegrityOfIntegrityBlock
public import Proofs.Flock.EndToEnd.Carry
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Proofs.Flock.Soundness.Game
import Proofs.Pouw.Prob

/-! The integrity audit of a server-time block, proved from Flock's integrity audit of that block. -/

@[expose] public section

namespace ResourceAccounting.SecurityProofs.Bridge

open scoped ENNReal

/-- **The integrity audit of a server-time block** (`ResourceAccounting.Guarantees.Bridge.IntegrityOfIntegrityBlock`). -/
theorem IntegrityOfIntegrityBlock : ResourceAccounting.Guarantees.Bridge.IntegrityOfIntegrityBlock := by
  intro ι d α _ C I Ω _ _ r S b q δ hq hδ hqt hA3 R hT hL hA2 c p hcert hfin ξ
  have hsub : ∀ x : Ω (r.own .sampled),
      (r.accepts .sampled (Function.update ξ (r.own .sampled) x) ∧
        ¬ r.Holds c p .sampled (Function.update ξ (r.own .sampled) x)) →
      (∀ w ∈ R.record (Function.update ξ (r.own .sampled) x), w.accepted) ∧
        Real.log (1 / δ) / q ≤
          (((R.record (Function.update ξ (r.own .sampled) x)).map Flock.Guarantees.SamplingBlockRec.incorrect).sum : ℝ) := by
    rintro x ⟨hacc, hfail⟩
    refine ⟨(R.accepts_iff _).1 hacc, ?_⟩
    have hfail' : ¬ (((r.parts.incorrect (r.trace (Function.update ξ (r.own .sampled) x))).card : ℝ) ≤
        c.ε * r.units) := hfail
    have hlt := not_le.mp hfail'
    rw [R.wrong_eq] at hlt
    exact hcert.trans hlt.le
  have hlaw := R.samplingDesign ξ fun l => (∀ w ∈ l, w.accepted) ∧
    Real.log (1 / δ) / q ≤ ((l.map Flock.Guarantees.SamplingBlockRec.incorrect).sum : ℝ)
  have hev : FlockSoundness.Game.value (fun l => (∀ w ∈ l.filter (fun w => w.idx ∈ S.sessions b), w.accepted) ∧
        Real.log (1 / δ) / q ≤
          (((l.filter fun w => w.idx ∈ S.sessions b).map Flock.Guarantees.SamplingBlockRec.incorrect).sum : ℝ))
        (S.run (R.strat ξ)) =
      FlockSoundness.Game.value (S.BlockMissProb q δ b) (S.run (R.strat ξ)) :=
    FlockSoundness.Game.value_congr (fun l => by
      simp only [Flock.Guarantees.ServerBlocks.BlockMissProb, Flock.Guarantees.ServerBlocks.blockIncorrect,
        List.mem_filter, decide_eq_true_eq, and_imp]) _
  have hsrc := Flock.SecurityProofs.IntegrityBlockCarried S b q δ hq hδ hqt hA3 (R.strat ξ) (hT ξ) (hL ξ) (hA2 ξ)
  have hfin' : ENNReal.ofReal δ + S.err b + S.carry b ≠ ⊤ :=
    ENNReal.add_ne_top.2 ⟨hfin, ENNReal.sum_ne_top.2 fun _ _ => ENNReal.ofReal_ne_top⟩
  refine (Pouw.pr_mono hsub).trans ?_
  rw [← ENNReal.toReal_ofReal (Pouw.pr_nonneg _), hlaw, hev]
  exact ENNReal.toReal_mono hfin' hsrc

end ResourceAccounting.SecurityProofs.Bridge
