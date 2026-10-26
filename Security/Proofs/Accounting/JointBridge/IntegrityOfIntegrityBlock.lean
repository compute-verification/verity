import Proofs.Accounting.JointBridge.Guarantees.IntegrityOfIntegrityBlock
import Proofs.Accounting.JointBlocks
import Proofs.Flock.EndToEnd.Blocks
import Proofs.Pouw.Prob

/-!
# Proof of the integrity audit of a server-time block from C-Flock's integrity audit of that block

* `IntegrityOfIntegrityBlock`: at each value of the other coins, a draw of block `b`'s sampled coins on which its audit
  accepts while more than `ε` of its units are wrong is one on which every session of the block is accepted and the
  block's wrong units total more than `ε·units ≥ ln(1/δ)/q`. Read on the horizon's records, that is the block escaping
  (`ServerBlocks.BlockEscapes`), the event `Flock.SecurityProofs.IntegrityBlock` bounds; the realization's `law` carries
  the bound from the horizon's game to the uniform coins.
* `JointHorizonBlocksOfIntegrityBlock`: `Accounting.SecurityProofs.JointHorizonBlocks` with each block's sampled error
  set to its `IntegrityBlock` error, each block's sampled leg by `IntegrityOfIntegrityBlock`.
-/

namespace Accounting.SecurityProofs.Bridge

open scoped ENNReal

/-- **The integrity audit of a server-time block** (`Accounting.Guarantees.Bridge.IntegrityOfIntegrityBlock`). -/
theorem IntegrityOfIntegrityBlock : Accounting.Guarantees.Bridge.IntegrityOfIntegrityBlock := by
  intro ι d α _ C I Ω _ _ B s S b q δ hq hδ hqt hA3 R hCR c p hcert hfin ξ
  have hsub : ∀ x : Ω ((s.block b).own .sampled),
      ((s.block b).accepts .sampled (Function.update ξ ((s.block b).own .sampled) x) ∧
        ¬ (s.block b).Holds (c b) p .sampled (Function.update ξ ((s.block b).own .sampled) x)) →
      (∀ w ∈ R.record (Function.update ξ ((s.block b).own .sampled) x), w.accepted) ∧
        Real.log (1 / δ) / q ≤
          (((R.record (Function.update ξ ((s.block b).own .sampled) x)).map
            Flock.Guarantees.WindowRec.wrong).sum : ℝ) := by
    rintro x ⟨hacc, hfail⟩
    refine ⟨(R.accepts_iff _).1 hacc, ?_⟩
    have hfail' : ¬ ((((s.block b).parts.wrong
        ((s.block b).transcript (Function.update ξ ((s.block b).own .sampled) x))).card : ℝ) ≤
        (c b).ε * (s.block b).units) := hfail
    have hlt := not_le.mp hfail'
    rw [R.wrong_eq] at hlt
    exact hcert.trans hlt.le
  have hlaw := R.law ξ fun l => (∀ w ∈ l, w.accepted) ∧
    Real.log (1 / δ) / q ≤ ((l.map Flock.Guarantees.WindowRec.wrong).sum : ℝ)
  have hev : FlockSoundness.Game.value (fun l => (∀ w ∈ l.filter (fun w => w.idx ∈ S.sessions b), w.accepted) ∧
        Real.log (1 / δ) / q ≤
          (((l.filter fun w => w.idx ∈ S.sessions b).map Flock.Guarantees.WindowRec.wrong).sum : ℝ))
        (S.run (R.strat ξ)) =
      FlockSoundness.Game.value (S.BlockEscapes q δ b) (S.run (R.strat ξ)) :=
    FlockSoundness.Game.value_congr (fun l => by
      simp only [Flock.Guarantees.ServerBlocks.BlockEscapes, Flock.Guarantees.ServerBlocks.blockWrong,
        List.mem_filter, decide_eq_true_eq, and_imp]
      exact and_congr (forall_congr' fun w => forall_comm) Iff.rfl) _
  have hsrc := Flock.SecurityProofs.IntegrityBlock S b q δ hq hδ hqt hA3 (R.strat ξ) (hCR ξ)
  refine (Pouw.pr_mono hsub).trans ?_
  rw [← ENNReal.toReal_ofReal (Pouw.pr_nonneg _), hlaw, hev]
  exact ENNReal.toReal_mono hfin hsrc

/-- **The joint guarantee over a horizon of server-time blocks, with every block's integrity leg from C-Flock's
integrity audit of that block** (`Accounting.Guarantees.Bridge.JointHorizonBlocksOfIntegrityBlock`). -/
theorem JointHorizonBlocksOfIntegrityBlock : Accounting.Guarantees.Bridge.JointHorizonBlocksOfIntegrityBlock := by
  intro ι α I _ _ _ _ Ω _ _ d C B s S q δI hq hδ hqt hA3 R hCR c p f δ εWipe δPous hcert hfin hS hW hM
  let δ' : Fin B → Audit → ℝ := fun b a =>
    if a = .sampled then (ENNReal.ofReal δI + S.err b).toReal else δ b a
  have hS' : ∀ b a, (s.block b).Sound (c b) p a (δ' b a) := by
    intro b a
    by_cases ha : a = .sampled
    · subst ha
      simp only [δ', if_pos rfl]
      exact IntegrityOfIntegrityBlock ι d α C I Ω B s S b q δI hq hδ (hqt b) (hA3 b) (R b) (hCR b) c p
        (hcert b) (hfin b)
    · simp only [δ', if_neg ha]
      exact hS b a ha
  refine (Accounting.SecurityProofs.JointHorizonBlocks ι α I Ω d C B s c p f δ' εWipe δPous hS' hW hM).trans_eq ?_
  congr 2
  refine Finset.sum_congr rfl fun b _ => ?_
  rw [← Finset.add_sum_erase _ _ (Finset.mem_univ Audit.sampled)]
  simp only [δ', if_pos rfl]
  congr 1
  exact Finset.sum_congr rfl fun a ha => if_neg (Finset.ne_of_mem_erase ha)

end Accounting.SecurityProofs.Bridge
