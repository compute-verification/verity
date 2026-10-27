import Proofs.Accounting.JointBridge.Guarantees.IntegrityOfIntegrityBlock
import Proofs.Accounting.JointBlocks
import Proofs.Flock.EndToEnd.Blocks
import Proofs.Pouw.Prob

/-!
# Proof of the integrity audit of a server-time block from C-Flock's integrity audit of that block

* `IntegrityOfIntegrityBlock`: at each value of the other coins, a draw of the period's sampled coins on which its audit
  accepts while more than `ε` of its units are wrong is one on which every sampling block of the server-time block is
  accepted and their wrong units total more than `ε·units ≥ ln(1/δ)/q`. Read on the horizon's records, that is the
  block escaping (`ServerBlocks.BlockEscapes`), the event `Flock.SecurityProofs.IntegrityBlock` bounds; the
  realization's `law` carries the bound from the horizon's game to the uniform coins.
* `JointHorizonBlocksAOfIntegrityBlock`: `Accounting.SecurityProofs.JointHorizonBlocksA` with each block's sampled error
  set to its `IntegrityBlock` error, each block's sampled leg by `IntegrityOfIntegrityBlock`.
* `JointHorizonBlocksOfIntegrityBlock`: the same over `Accounting.SecurityProofs.JointHorizonBlocks`.
-/

namespace Accounting.SecurityProofs.Bridge

open scoped ENNReal

/-- **The integrity audit of a server-time block** (`Accounting.Guarantees.Bridge.IntegrityOfIntegrityBlock`). -/
theorem IntegrityOfIntegrityBlock : Accounting.Guarantees.Bridge.IntegrityOfIntegrityBlock := by
  intro ι d α _ C I Ω _ _ r S b q δ hq hδ hqt hA3 R hT hL c p hcert hfin ξ
  have hsub : ∀ x : Ω (r.own .sampled),
      (r.accepts .sampled (Function.update ξ (r.own .sampled) x) ∧
        ¬ r.Holds c p .sampled (Function.update ξ (r.own .sampled) x)) →
      (∀ w ∈ R.record (Function.update ξ (r.own .sampled) x), w.accepted) ∧
        Real.log (1 / δ) / q ≤
          (((R.record (Function.update ξ (r.own .sampled) x)).map Flock.Guarantees.WindowRec.wrong).sum : ℝ) := by
    rintro x ⟨hacc, hfail⟩
    refine ⟨(R.accepts_iff _).1 hacc, ?_⟩
    have hfail' : ¬ (((r.parts.wrong (r.transcript (Function.update ξ (r.own .sampled) x))).card : ℝ) ≤
        c.ε * r.units) := hfail
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
        List.mem_filter, decide_eq_true_eq, and_imp]) _
  have hsrc := Flock.SecurityProofs.IntegrityBlock S b q δ hq hδ hqt hA3 (R.strat ξ) (hT ξ) (hL ξ)
  refine (Pouw.pr_mono hsub).trans ?_
  rw [← ENNReal.toReal_ofReal (Pouw.pr_nonneg _), hlaw, hev]
  exact ENNReal.toReal_mono hfin hsrc

/-- **Demo A's composed theorem, with every server-time block's integrity leg from C-Flock's integrity audit of that
block** (`Accounting.Guarantees.Bridge.JointHorizonBlocksAOfIntegrityBlock`). -/
theorem JointHorizonBlocksAOfIntegrityBlock : Accounting.Guarantees.Bridge.JointHorizonBlocksAOfIntegrityBlock := by
  intro ι α I _ _ _ Ω _ _ d C B r S q δI hq hδ hqt hA3 R hT hL c p δ hcert hfin hW hN
  let δ' : Fin B → Audit → ℝ := fun b a =>
    if a = .sampled then (ENNReal.ofReal δI + S.err b).toReal else δ b a
  have hS' : ∀ b, ∀ a ∈ ({.sampled, .pouw, .network} : Finset Audit), (r b).Sound (c b) p a (δ' b a) := by
    intro b a ha
    simp only [Finset.mem_insert, Finset.mem_singleton] at ha
    rcases ha with rfl | rfl | rfl
    · simp only [δ', ite_true]
      exact IntegrityOfIntegrityBlock ι d α C I Ω (r b) S b q δI hq hδ (hqt b) (hA3 b) (R b) (hT b) (hL b) (c b) p
        (hcert b) (hfin b)
    · simp only [δ', reduceCtorEq, ite_false]
      exact hW b
    · simp only [δ', reduceCtorEq, ite_false]
      exact hN b
  refine (Accounting.SecurityProofs.JointHorizonBlocksA ι α I Ω d C B r c p δ' hS').trans_eq
    (Finset.sum_congr rfl fun b _ => ?_)
  rw [Finset.sum_insert (by simp), Finset.sum_insert (by simp), Finset.sum_singleton]
  simp only [δ', ite_true, reduceCtorEq, ite_false, add_assoc]

/-- **The joint guarantee over a horizon of server-time blocks, with every block's integrity leg from C-Flock's
integrity audit of that block** (`Accounting.Guarantees.Bridge.JointHorizonBlocksOfIntegrityBlock`). -/
theorem JointHorizonBlocksOfIntegrityBlock : Accounting.Guarantees.Bridge.JointHorizonBlocksOfIntegrityBlock := by
  intro ι α I _ _ _ _ Ω _ _ d C B s S q δI hq hδ hqt hA3 R hT hL c p f δ εWipe δPous hcert hfin hS hW hM
  let δ' : Fin B → Audit → ℝ := fun b a =>
    if a = .sampled then (ENNReal.ofReal δI + S.err b).toReal else δ b a
  have hS' : ∀ b a, (s.block b).Sound (c b) p a (δ' b a) := by
    intro b a
    by_cases ha : a = .sampled
    · subst ha
      simp only [δ', ite_true]
      exact IntegrityOfIntegrityBlock ι d α C I Ω (s.block b) S b q δI hq hδ (hqt b) (hA3 b) (R b) (hT b) (hL b)
        (c b) p (hcert b) (hfin b)
    · simp only [δ', ha, ite_false]
      exact hS b a ha
  have hsum : ∀ b, ∑ a, δ' b a =
      (ENNReal.ofReal δI + S.err b).toReal + ∑ a ∈ Finset.univ.erase .sampled, δ b a := by
    intro b
    rw [← Finset.add_sum_erase _ _ (Finset.mem_univ Audit.sampled)]
    refine congrArg₂ (· + ·) (by simp only [δ', ite_true])
      (Finset.sum_congr rfl fun a ha => ?_)
    simp only [δ', Finset.ne_of_mem_erase ha, ite_false]
  refine (Accounting.SecurityProofs.JointHorizonBlocks ι α I Ω d C B s c p f δ' εWipe δPous hS' hW hM).trans_eq ?_
  rw [Finset.sum_congr rfl fun b _ => hsum b]

end Accounting.SecurityProofs.Bridge
