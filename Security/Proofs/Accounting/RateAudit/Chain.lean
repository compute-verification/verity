import Mathlib
import Models.Accounting.Guarantees.RateAuditRemote
import Proofs.Accounting.RateAudit.Interactive
import Proofs.Accounting.RateAudit.Windows

/-!
# Chained passes: the proof

**Sequencing.** Fix every pass's coins but pass `j`'s. Pass `j`'s device and the root its seed is drawn at are then
fixed functions of the store (`Chain.device`, `Chain.root`), and its coins are a stream, an oracle family over roots
and a sample. Split the family at the root the store picks: the family is the oracle at that root together with the
family elsewhere (`rootSplit`), a relabelling of the coins for each store, so the oracle at the root is uniform and
independent of the store, and the event depends on nothing else (`pr_root_le`). Pass `j` is then the rate audit's game
for one fixed interactive device, and `RateAuditInteractiveRO` or `RateAuditInteractive` bounds it at each value of
the other passes' coins (`Chain.pr_short_le`).

**The window.** As `RateAuditWindows`': footprints summed short of `P + 1 − c` floors in passes all accepted within the
budget leave at least `c` passes short (`Chain.le_card_short`), and Markov's inequality over the count of short passes
(`mul_pr_le_sum`) gives `P` times the per-pass error.
-/

namespace Accounting.Rate

open Pous (Bits pr)
open scoped ENNReal

variable {N b h ρ r s : ℕ}

/-- An oracle family as its oracle at the root `rt` and the family off a base point `x₀`: the family is relabelled
by the swap of `rt` and `x₀`, then split at `x₀`. -/
def atRoot (x₀ rt : Bits r) : (Bits r → Oracle N b h ρ) ≃ Oracle N b h ρ × ({x : Bits r // x ≠ x₀} → Oracle N b h ρ) :=
  (Equiv.arrowCongr (Equiv.swap rt x₀) (Equiv.refl _)).trans (Equiv.funSplitAt x₀ _)

theorem atRoot_apply_fst (x₀ rt : Bits r) (F : Bits r → Oracle N b h ρ) : (atRoot x₀ rt F).1 = F rt := by
  simp [atRoot, Equiv.funSplitAt, Equiv.arrowCongr]

theorem atRoot_symm_apply (x₀ rt : Bits r) (O : Oracle N b h ρ) (g : {x : Bits r // x ≠ x₀} → Oracle N b h ρ) :
    (atRoot x₀ rt).symm (O, g) rt = O := by
  rw [← atRoot_apply_fst x₀ rt ((atRoot x₀ rt).symm (O, g)), Equiv.apply_symm_apply]

/-- A pass's coins, with its oracle family split at the root the store picks: the game's coins and the family
elsewhere. -/
def rootSplit (rt : Store N b → Bits r) (x₀ : Bits r) :
    Store N b × PassCoins N b h ρ r s ≃ (Store N b × Coins N b h ρ s) × ({x : Bits r // x ≠ x₀} → Oracle N b h ρ) where
  toFun x := ((x.1, x.2.1, (atRoot x₀ (rt x.1) x.2.2.1).1, x.2.2.2), (atRoot x₀ (rt x.1) x.2.2.1).2)
  invFun y := (y.1.1, y.1.2.1, (atRoot x₀ (rt y.1.1)).symm (y.1.2.2.1, y.2), y.1.2.2.2)
  left_inv x := by simp
  right_inv y := by simp

/-- The oracle family at a root the store picks is a uniform oracle: an event of the game's coins has the same bound
at the family's value there. -/
theorem pr_root_le (rt : Store N b → Bits r) (Q : Store N b × Coins N b h ρ s → Prop) (ε : ℝ≥0∞) (hQ : pr Q ≤ ε) :
    pr (fun x : Store N b × PassCoins N b h ρ r s => Q (x.1, x.2.1, x.2.2.1 (rt x.1), x.2.2.2)) ≤ ε := by
  classical
  let x₀ : Bits r := fun _ => false
  have : Fintype ({x : Bits r // x ≠ x₀} → Oracle N b h ρ) := inferInstance
  rw [pr_equiv (rootSplit (s := s) rt x₀)]
  refine pr_le_of_fiber _ _ (fun g => le_of_eq_of_le ?_ hQ)
  congr 1
  funext y
  change Q (y.1, y.2.1, (atRoot _ (rt y.1)).symm (y.2.2.1, g) (rt y.1), y.2.2.2) = Q y
  rw [atRoot_symm_apply]

namespace Chain

variable {P S L q : ℕ}

/-- Pass `j` was accepted within the compute budget with its footprint short of the floor `fl`. -/
def Short (A : Chain P N b h ρ S L q r s) (cTag cEnc F fl : ℕ) (j : Fin P)
    (x : Store N b × (Fin P → PassCoins N b h ρ r s)) : Prop :=
  (A.dev j x.2).Accepts x.1 (x.2 j).1 (A.oracle j x.1 x.2) (x.2 j).2.2 ∧
    (A.dev j x.2).compute cTag cEnc x.1 (x.2 j).1 (A.oracle j x.1 x.2) ≤ F ∧
    (A.dev j x.2).footprint x.1 (x.2 j).1 (A.oracle j x.1 x.2) * cEnc < fl

/-- **Sequencing**: pass `j` is accepted short with at most the game's error for the devices the earlier passes' coins
pick. -/
theorem pr_short_le (A : Chain P N b h ρ S L q r s) (cTag cEnc F fl : ℕ) (ε : ℝ≥0∞) (j : Fin P)
    (hε : ∀ η, pr (fun x : Store N b × Coins N b h ρ s =>
        (A.device j η).Accepts x.1 x.2.1 x.2.2.1 x.2.2.2 ∧ (A.device j η).compute cTag cEnc x.1 x.2.1 x.2.2.1 ≤ F ∧
          (A.device j η).footprint x.1 x.2.1 x.2.2.1 * cEnc < fl) ≤ ε) :
    pr (Short A cTag cEnc F fl j) ≤ ε := by
  classical
  let e : Store N b × (Fin P → PassCoins N b h ρ r s) ≃
      (Store N b × PassCoins N b h ρ r s) × ({i : Fin P // i ≠ j} → PassCoins N b h ρ r s) :=
    ((Equiv.refl _).prodCongr (Equiv.funSplitAt j _)).trans (Equiv.prodAssoc _ _ _).symm
  rw [pr_equiv e]
  refine pr_le_of_fiber _ _ (fun ω' => ?_)
  let η : {i : Fin P // i < j} → PassCoins N b h ρ r s := fun i => ω' ⟨i.1, i.2.ne⟩
  refine (Bandwidth.pr_mono (fun x hx => ?_)).trans (pr_root_le (A.root j η) _ ε (hε η))
  have hω : ((Equiv.funSplitAt j (PassCoins N b h ρ r s)).symm (x.2, ω')) j = x.2 := by simp
  have hη : (fun i : {i : Fin P // i < j} => ((Equiv.funSplitAt j (PassCoins N b h ρ r s)).symm (x.2, ω')) i.1) =
      η := by
    funext i
    simp [η, i.2.ne]
  have hd : A.dev j ((Equiv.funSplitAt j (PassCoins N b h ρ r s)).symm (x.2, ω')) = A.device j η := by
    simp only [Chain.dev]
    rw [hη]
  have ho : A.oracle j x.1 ((Equiv.funSplitAt j (PassCoins N b h ρ r s)).symm (x.2, ω')) =
      x.2.2.1 (A.root j η x.1) := by
    simp only [Chain.oracle]
    rw [hη, hω]
  simpa [e, Short, hω, hd, ho] using hx

open Classical in
/-- Footprints summing short of `P + 1 − c` floors, in passes all accepted within the budget, leave at least `c`
passes short. -/
theorem le_card_short (A : Chain P N b h ρ S L q r s) (cTag cEnc F fl c : ℕ)
    (x : Store N b × (Fin P → PassCoins N b h ρ r s))
    (hall : ∀ j, (A.dev j x.2).Accepts x.1 (x.2 j).1 (A.oracle j x.1 x.2) (x.2 j).2.2 ∧
      (A.dev j x.2).compute cTag cEnc x.1 (x.2 j).1 (A.oracle j x.1 x.2) ≤ F)
    (hsum : ∑ j, (A.dev j x.2).footprint x.1 (x.2 j).1 (A.oracle j x.1 x.2) * cEnc < (P + 1 - c) * fl) :
    c ≤ (Finset.univ.filter (fun j => Short A cTag cEnc F fl j x)).card := by
  classical
  by_contra hlt
  push Not at hlt
  set T := Finset.univ.filter (fun j => ¬ Short A cTag cEnc F fl j x)
  have hT : (Finset.univ.filter (fun j => Short A cTag cEnc F fl j x)).card + T.card = P := by
    rw [Finset.card_filter_add_card_filter_not, Finset.card_univ, Fintype.card_fin]
  have hle : T.card * fl ≤ ∑ j, (A.dev j x.2).footprint x.1 (x.2 j).1 (A.oracle j x.1 x.2) * cEnc :=
    calc T.card * fl = ∑ _j ∈ T, fl := by rw [Finset.sum_const, smul_eq_mul]
      _ ≤ ∑ j ∈ T, (A.dev j x.2).footprint x.1 (x.2 j).1 (A.oracle j x.1 x.2) * cEnc := by
          refine Finset.sum_le_sum (fun j hj => ?_)
          have hj := (Finset.mem_filter.1 hj).2
          simp only [Short, not_and, not_lt] at hj
          exact hj (hall j).1 (hall j).2
      _ ≤ ∑ j, (A.dev j x.2).footprint x.1 (x.2 j).1 (A.oracle j x.1 x.2) * cEnc :=
          Finset.sum_le_sum_of_subset_of_nonneg (Finset.filter_subset _ _) (fun _ _ _ => Nat.zero_le _)
  have : (P + 1 - c) * fl ≤ T.card * fl := Nat.mul_le_mul_right _ (by omega)
  omega

/-- The window's bound from a per-pass bound: `c` times the probability that every pass is accepted within the budget
with the footprints summed short of `P + 1 − c` floors is at most `P·ε`. -/
theorem mul_pr_sum_le (A : Chain P N b h ρ S L q r s) (cTag cEnc F fl c : ℕ) (ε : ℝ≥0∞)
    (hε : ∀ j, pr (Short A cTag cEnc F fl j) ≤ ε) :
    (c : ℝ≥0∞) * pr (fun x : Store N b × (Fin P → PassCoins N b h ρ r s) =>
        (∀ j, (A.dev j x.2).Accepts x.1 (x.2 j).1 (A.oracle j x.1 x.2) (x.2 j).2.2 ∧
          (A.dev j x.2).compute cTag cEnc x.1 (x.2 j).1 (A.oracle j x.1 x.2) ≤ F) ∧
        ∑ j, (A.dev j x.2).footprint x.1 (x.2 j).1 (A.oracle j x.1 x.2) * cEnc < (P + 1 - c) * fl)
      ≤ (P : ℝ≥0∞) * ε := by
  classical
  calc (c : ℝ≥0∞) * pr (fun x : Store N b × (Fin P → PassCoins N b h ρ r s) =>
          (∀ j, (A.dev j x.2).Accepts x.1 (x.2 j).1 (A.oracle j x.1 x.2) (x.2 j).2.2 ∧
            (A.dev j x.2).compute cTag cEnc x.1 (x.2 j).1 (A.oracle j x.1 x.2) ≤ F) ∧
          ∑ j, (A.dev j x.2).footprint x.1 (x.2 j).1 (A.oracle j x.1 x.2) * cEnc < (P + 1 - c) * fl)
      ≤ (c : ℝ≥0∞) * pr (fun x : Store N b × (Fin P → PassCoins N b h ρ r s) =>
          c ≤ (Finset.univ.filter (fun j => Short A cTag cEnc F fl j x)).card) := by
        gcongr
        exact Bandwidth.pr_mono (fun x hx => le_card_short A cTag cEnc F fl c x hx.1 hx.2)
    _ ≤ ∑ j, pr (Short A cTag cEnc F fl j) := mul_pr_le_sum _ c
    _ ≤ ∑ _j : Fin P, ε := Finset.sum_le_sum (fun j _ => hε j)
    _ = (P : ℝ≥0∞) * ε := by rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]

end Chain

end Accounting.Rate

namespace Accounting.SecurityProofs

open Accounting.Rate
open Pous (pr)
open scoped ENNReal

/-- **A chained pass is a fresh audit, with side sources that ignore the oracle**
(`Accounting.Guarantees.RateChainPassRO`). -/
theorem RateChainPassRO : Accounting.Guarantees.RateChainPassRO := by
  intro P N b h ρ S L q r s k F cTag cEnc lam A hA j
  exact Chain.pr_short_le A cTag cEnc F _ _ j (fun η =>
    RateAuditInteractiveRO N b h ρ S L q s k F cTag cEnc lam (A.device j η) (hA j η))

/-- **A chained pass is a fresh audit, with any side sources** (`Accounting.Guarantees.RateChainPass`). -/
theorem RateChainPass : Accounting.Guarantees.RateChainPass := by
  intro P N b h ρ S L q r s k m F cTag cEnc lam A j
  exact Chain.pr_short_le A cTag cEnc F _ _ j (fun η =>
    RateAuditInteractive N b h ρ S L q s k m F cTag cEnc lam (A.device j η))

/-- **A window of chained passes certifies the sum over its passes, with side sources that ignore the oracle**
(`Accounting.Guarantees.RateChainRO`). -/
theorem RateChainRO : Accounting.Guarantees.RateChainRO := by
  intro P N b h ρ S L q r s k F cTag cEnc lam c A hA
  exact Chain.mul_pr_sum_le A cTag cEnc F _ c _ (fun j =>
    RateChainPassRO P N b h ρ S L q r s k F cTag cEnc lam A hA j)

/-- **A window of chained passes certifies the sum over its passes, with any side sources**
(`Accounting.Guarantees.RateChain`). -/
theorem RateChain : Accounting.Guarantees.RateChain := by
  intro P N b h ρ S L q r s k m F cTag cEnc lam c A
  exact Chain.mul_pr_sum_le A cTag cEnc F _ c _ (fun j =>
    RateChainPass P N b h ρ S L q r s k m F cTag cEnc lam A j)

end Accounting.SecurityProofs
