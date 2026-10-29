import Definitions.Nci.Guarantees.Capability.Incidence
import Proofs.Nci.Capability.Items
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Tactic.Linarith

/-!
# Local bounds: proofs of the caps from incidence

* `IncidenceCap`: the completed products lie in the union, over the slot-operand pairs they use, of each pair's products
  (at most `r` each), and in the union, over the operands they use, of each operand's (at most `aout` each);
* `OperandCapSeparate`, `OperandCapJoint`: the slots and the operands a proof block's completed products read are
  selections it holds (`exists_selection`), so their prices bound how many there are, and the incidence cap at those
  numbers is at most `Q`.
-/

namespace Nci.SecurityProofs

open Verity Finset Nci.Capability

theorem IncidenceCap : Guarantees.IncidenceCap := by
  intro d B κ _ W O I r aout hr ha S
  set Q := completedIn B S
  have hQ : Q ⊆ B.P := filter_subset _ _
  refine le_min ?_ ?_
  · have hsub : Q ⊆ ((Q.image Prod.fst) ×ˢ (Q.image I.occ)).biUnion
        (fun p => B.P.filter fun q => q.1 = p.1 ∧ I.occ q = p.2) := by
      intro q hq
      simp only [mem_biUnion, mem_product, mem_image, mem_filter]
      exact ⟨(q.1, I.occ q), ⟨⟨q, hq, rfl⟩, ⟨q, hq, rfl⟩⟩, hQ hq, rfl, rfl⟩
    calc (Q.card : ℝ)
        ≤ ∑ p ∈ (Q.image Prod.fst) ×ˢ (Q.image I.occ),
            ((B.P.filter fun q => q.1 = p.1 ∧ I.occ q = p.2).card : ℝ) := by
          rw [← Nat.cast_sum]
          exact Nat.cast_le.2 ((card_le_card hsub).trans card_biUnion_le)
      _ ≤ ∑ _p ∈ (Q.image Prod.fst) ×ˢ (Q.image I.occ), r := sum_le_sum fun p _ => hr p.1 p.2
      _ = r * (Q.image Prod.fst).card * (Q.image I.occ).card := by
          rw [sum_const, card_product, nsmul_eq_mul]
          push_cast
          ring
  · have hsub : Q ⊆ (Q.image I.occ).biUnion (fun o => B.P.filter fun q => I.occ q = o) := by
      intro q hq
      simp only [mem_biUnion, mem_image, mem_filter]
      exact ⟨I.occ q, ⟨q, hq, rfl⟩, hQ hq, rfl⟩
    calc (Q.card : ℝ)
        ≤ ∑ o ∈ Q.image I.occ, ((B.P.filter fun q => I.occ q = o).card : ℝ) := by
          rw [← Nat.cast_sum]
          exact Nat.cast_le.2 ((card_le_card hsub).trans card_biUnion_le)
      _ ≤ ∑ _o ∈ Q.image I.occ, aout := sum_le_sum fun o _ => ha o
      _ = aout * (Q.image I.occ).card := by
          rw [sum_const, nsmul_eq_mul, mul_comm]

/-- The slots and the operands a proof block's completed products read are selections it holds: with uniform prices,
`βw |U| ≤ hw` and `βx |V| ≤ hx` for separate budgets, `βw |U| + βx |V| ≤ h` for a joint one. -/
theorem held_slots_operands (R : Committed) (B : Nci.Covering.MulBlocks R.d) {κ : Type} [DecidableEq κ]
    (W : Items R.d.D (Fin B.nw)) (O : Items R.d.D κ) (I : Incidence B W O) (u : Fin R.d.n) :
    (∃ σ, Selects W ((completedIn B (R.d.𝓘.gates u)).image Prod.fst) σ ∧
      ∀ i ∈ (completedIn B (R.d.𝓘.gates u)).image Prod.fst, σ i ⊆ holds R.d.D (R.d.𝓘.gates u)) ∧
    ∃ σ, Selects O ((completedIn B (R.d.𝓘.gates u)).image I.occ) σ ∧
      ∀ o ∈ (completedIn B (R.d.𝓘.gates u)).image I.occ, σ o ⊆ holds R.d.D (R.d.𝓘.gates u) := by
  have hP : ∀ q ∈ completedIn B (R.d.𝓘.gates u), q ∈ B.P ∧ B.m q ⊆ R.d.𝓘.gates u := fun q hq => mem_filter.1 hq
  exact ⟨exists_selection W _ Prod.fst B.m _ (fun q hq => I.slots q (hP q hq).1) (fun q hq => (hP q hq).2),
    exists_selection O _ I.occ B.m _ (fun q hq => I.operands q (hP q hq).1) (fun q hq => (hP q hq).2)⟩

theorem OperandCapSeparate : Guarantees.OperandCapSeparate := by
  intro R B κ _ W O I r aout βw βx hw hx Q _ _ hpw hpx hr ha hW hO hQ u
  obtain ⟨⟨σw, hσw, hσw'⟩, ⟨σx, hσx, hσx'⟩⟩ := held_slots_operands R B W O I u
  have h1 := hW u _ σw hσw hσw'
  have h2 := hO u _ σx hσx hσx'
  simp only [hpw, hpx, sum_const, nsmul_eq_mul] at h1 h2
  exact (IncidenceCap R.d B W O I r aout hr ha (R.d.𝓘.gates u)).trans
    (hQ _ _ (Nat.cast_nonneg _) (Nat.cast_nonneg _) (by linarith) (by linarith))

theorem OperandCapJoint : Guarantees.OperandCapJoint := by
  intro R B κ _ W O I r aout βw βx h Q _ _ hpw hpx hr ha hWO hQ u
  obtain ⟨⟨σw, hσw, hσw'⟩, ⟨σx, hσx, hσx'⟩⟩ := held_slots_operands R B W O I u
  set U := (completedIn B (R.d.𝓘.gates u)).image Prod.fst
  set V := (completedIn B (R.d.𝓘.gates u)).image I.occ
  have hsel : Selects (W.sum O) (U.disjSum V) (Sum.elim σw σx) := by
    intro i hi
    cases i with
    | inl i => exact hσw i (by simpa using hi)
    | inr o => exact hσx o (by simpa using hi)
  have hheld : ∀ i ∈ U.disjSum V, Sum.elim σw σx i ⊆ holds R.d.D (R.d.𝓘.gates u) := by
    intro i hi
    cases i with
    | inl i => exact hσw' i (by simpa using hi)
    | inr o => exact hσx' o (by simpa using hi)
  have h1 := hWO u _ _ hsel hheld
  rw [sum_disjSum] at h1
  simp only [Items.sum, Sum.elim_inl, Sum.elim_inr, hpw, hpx, sum_const, nsmul_eq_mul] at h1
  exact (IncidenceCap R.d B W O I r aout hr ha (R.d.𝓘.gates u)).trans
    (hQ _ _ (Nat.cast_nonneg _) (Nat.cast_nonneg _) (by linarith))

end Nci.SecurityProofs
