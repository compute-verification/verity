import Definitions.Accounting.Guarantees.GatesAdditive
import Proofs.Accounting.Additive

/-!
# Compute additivity in gates: proofs

`Accounting.Guarantees.GatesAdditive` and the bridges to the joint ledgers' named assumptions. A device's work is at
most its supply over the window and its backlog (`work_le_supply`, from `PaidBy` and the period's rate), works that
share no gate total at most the region's work (`sum_le_work`), and the node's region is its devices' and its other
compute's (`work_union_le`).
-/

open scoped NNRat

namespace Accounting.SecurityProofs.Additivity

open Finset Verity.Accel Accounting.Assumptions

section Machine

variable {ι : Type} {m : Machine ι}

/-- A device's work over `w` is at most its supply over the window and its period's backlog. -/
theorem work_le_supply {L : Type*} (A : Placed L) (X : Finset L) (π : Fin A.N → ℕ) (iss : List (List m.Instr))
    (β : ℕ) (R : Finset m.Res) (κ : m.Res → ℚ≥0) (w : Window) (hrate : PeriodRate iss β)
    (hpaid : PaidBy A X π R κ iss w) :
    (A.work π (A.region X ↑w.ticks) : ℚ≥0) ≤ supply m R κ * ((w.len + β : ℕ) : ℚ≥0) := by
  refine hpaid.trans ?_
  rw [Finset.sum_comm, supply, Finset.sum_mul]
  refine Finset.sum_le_sum fun r _ => ?_
  rw [← Finset.mul_sum, mul_assoc]
  gcongr
  exact hrate w r

end Machine

section Gates

variable {L : Type*} (A : Placed L)

/-- Works that share no gate of a region total at most its work. -/
theorem sum_le_work {X : Finset L} {I : Set ℕ} {π : Fin A.N → ℕ} {k : ℕ} {W : Fin k → ℕ}
    (h : ShareNoGate A X I π W) : ∑ j, W j ≤ A.work π (A.region X I) := by
  obtain ⟨S, hS, hdisj, hW⟩ := h
  calc ∑ j, W j ≤ ∑ j, A.work π (S j) := Finset.sum_le_sum fun j _ => hW j
    _ = A.work π (univ.biUnion S) := by
        simp only [Placed.work]
        rw [Finset.sum_biUnion fun i _ j _ hij => hdisj i j hij]
    _ ≤ A.work π (A.region X I) := Finset.sum_le_sum_of_subset (Finset.biUnion_subset.2 fun j _ => hS j)

theorem sum_biUnion_le {α β : Type*} [DecidableEq β] (s : Finset α) (t : α → Finset β) (f : β → ℕ) :
    ∑ x ∈ s.biUnion t, f x ≤ ∑ i ∈ s, ∑ x ∈ t i, f x := by
  classical
  induction s using Finset.induction_on with
  | empty => simp
  | insert i s hi ih =>
    rw [Finset.biUnion_insert, Finset.sum_insert hi]
    calc ∑ x ∈ t i ∪ s.biUnion t, f x ≤ ∑ x ∈ t i, f x + ∑ x ∈ s.biUnion t, f x := by
          rw [← Finset.sum_union_inter]
          exact Nat.le_add_right _ _
      _ ≤ _ := Nat.add_le_add_left ih _

/-- The node's work over `I` is at most its devices' and its other compute's. -/
theorem work_union_le [DecidableEq L] {V : Type} [Fintype V] (π : Fin A.N → ℕ) (X : V → Finset L) (Y : Finset L)
    (I : Set ℕ) :
    A.work π (A.region (univ.biUnion X ∪ Y) I) ≤ ∑ v, A.work π (A.region (X v) I) + A.work π (A.region Y I) := by
  have hsub : A.region (univ.biUnion X ∪ Y) I ⊆ (univ.biUnion fun v => A.region (X v) I) ∪ A.region Y I := by
    intro g hg
    simp only [Placed.region, Finset.mem_filter, Finset.mem_union, Finset.mem_biUnion, Finset.mem_univ,
      true_and] at hg ⊢
    obtain ⟨hc, hx | hx, ht⟩ := hg
    · obtain ⟨v, hv⟩ := hx
      exact Or.inl ⟨v, hc, hv, ht⟩
    · exact Or.inr ⟨hc, hx, ht⟩
  calc A.work π (A.region (univ.biUnion X ∪ Y) I)
      ≤ A.work π ((univ.biUnion fun v => A.region (X v) I) ∪ A.region Y I) :=
        Finset.sum_le_sum_of_subset hsub
    _ ≤ A.work π (univ.biUnion fun v => A.region (X v) I) + A.work π (A.region Y I) := by
        simp only [Placed.work]
        rw [← Finset.sum_union_inter]
        exact Nat.le_add_right _ _
    _ ≤ _ := Nat.add_le_add_right (sum_biUnion_le _ _ _) _

end Gates

end Accounting.SecurityProofs.Additivity

namespace Accounting.SecurityProofs

open Finset Verity.Accel Accounting.Assumptions Additivity

theorem GatesAdditive : Guarantees.GatesAdditive := by
  intro L _ A π V ι _ m iss β R κ X Y ΦY w k W hrate hpaid hY hshare
  have hW := sum_le_work A hshare
  have hU := work_union_le A π X Y ↑w.ticks
  have hYw : A.work π (A.region Y ↑w.ticks) ≤ ΦY * w.len := by
    have := hY Y (Set.mem_singleton Y) w.start (w.start + w.len)
    rw [Window.ticks, Finset.coe_Ico]
    simpa using this
  calc ∑ j, (W j : ℚ≥0) = ((∑ j, W j : ℕ) : ℚ≥0) := by push_cast; rfl
    _ ≤ ((∑ v, A.work π (A.region (X v) ↑w.ticks) + A.work π (A.region Y ↑w.ticks) : ℕ) : ℚ≥0) := by
        exact_mod_cast hW.trans hU
    _ = ∑ v, (A.work π (A.region (X v) ↑w.ticks) : ℚ≥0) + (A.work π (A.region Y ↑w.ticks) : ℚ≥0) := by
        push_cast; rfl
    _ ≤ _ := add_le_add (Finset.sum_le_sum fun v _ =>
        work_le_supply A (X v) π (iss v) (β v) (R v) (κ v) w (hrate v) (hpaid v)) (by exact_mod_cast hYw)

/-- `GatesAdditive` for two works. -/
theorem Additivity.gates_two {L : Type*} [DecidableEq L] (A : Placed L) (π : Fin A.N → ℕ) {V ι : Type} [Fintype V]
    (m : V → Machine ι) (iss : ∀ v, List (List (m v).Instr)) (β : V → ℕ) (R : ∀ v, Finset (m v).Res)
    (κ : ∀ v, (m v).Res → ℚ≥0) (X : V → Finset L) (Y : Finset L) (ΦY : ℕ) (w : Window) (W₁ W₂ : ℕ)
    (hrate : ∀ v, PeriodRate (iss v) (β v)) (hpaid : ∀ v, PaidBy A (X v) π (R v) (κ v) (iss v) w)
    (hY : A.ComputeRate {Y} π fun _ => ΦY) (hshare : ShareNoGate A (univ.biUnion X ∪ Y) ↑w.ticks π ![W₁, W₂]) :
    ((W₁ + W₂ : ℕ) : ℚ≥0) ≤
      ∑ v, supply (m v) (R v) (κ v) * ((w.len + β v : ℕ) : ℚ≥0) + ((ΦY * w.len : ℕ) : ℚ≥0) := by
  have h := GatesAdditive A π m iss β R κ X Y ΦY w ![W₁, W₂] hrate hpaid hY hshare
  simp only [Fin.sum_univ_two, Matrix.cons_val_zero, Matrix.cons_val_one] at h
  rw [Nat.cast_add]
  exact h

theorem RemoteComputeAdditiveOfGates : Guarantees.RemoteComputeAdditiveOfGates := by
  intro τ N a Ω r L _ A π V ι _ m iss β R κ X Y ΦY w hw hrate hpaid hY hshare hcap ω
  have h := gates_two (A ω) (π ω) m (iss ω) β R κ X Y ΦY w _ _ (hrate ω) (hpaid ω) (hY ω) (hshare ω)
  rw [hw] at h
  exact_mod_cast h.trans hcap

theorem RateComputeAdditiveOfGates : Guarantees.RateComputeAdditiveOfGates := by
  intro p cTag cost N b h ρ s ι _ d α ΩW ΩS _ r real L _ A π V ι' _ m iss β R κ X Y ΦY hrate hpaid hY hshare hcap
    x y s'
  have h := gates_two (A (x, y, s')) (π (x, y, s')) m (iss (x, y, s')) β R κ X Y ΦY (r.win .compute) _ _
    (hrate _) (hpaid _) (hY _) (hshare x y s')
  exact_mod_cast h.trans hcap

theorem AdditiveOfShareNoGate : Guarantees.AdditiveOfShareNoGate := by
  intro L 𝓒 μ cert r impl X I π hμ hshare R hR himpl
  obtain ⟨n, hn, hs⟩ := hshare R hR himpl
  have h := sum_le_work R.A hs
  simp only [Fin.sum_univ_two, Matrix.cons_val_zero, Matrix.cons_val_one] at h
  calc (cert : ℕ∞) + r ≤ cert + n := by gcongr
    _ = ((cert + n : ℕ) : ℕ∞) := by push_cast; rfl
    _ ≤ μ R := by exact_mod_cast h.trans (hμ R hR)

end Accounting.SecurityProofs
