import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.BigOperators.Pi
import Mathlib.Algebra.Order.Pi
import Mathlib.Tactic.Ring
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Order.Interval.Finset.Nat
import Definitions.Accounting.Ledger.Physical

/-!
# The ledger's arithmetic

A meter is linear and monotone in the ledger (`Meter.eval_add`, `Meter.eval_mono`). A region's ledger adds over
disjoint sets of ticks, consecutive windows among them (`ledger_union_ticks`, `ledger_append`), and its gates and bits
held add over disjoint parts (`ledger_union_part_gates`, `ledger_union_part_held`), so a meter that reads no crossing
does (`eval_union_part`). Crossings only subadd over parts (`ledger_union_part_crossIn_le`,
`ledger_union_part_crossOut_le`): a bit moving between the two parts crosses neither boundary of their union. The
ledger of a circuit made of accounts' shares is the sum of the shares' ledgers (`ledger_sum`), and a capacity per tick
bounds a region of `n` ticks by `n` times it (`ledger_le_card_smul`, `eval_le_capacity`).
-/

namespace Accounting.Ledger

variable {G : Type}

section Meter

variable [Fintype G] (m : Meter G)

theorem Meter.eval_add (L L' : Ledger G) : m.eval (L + L') = m.eval L + m.eval L' := by
  simp only [Meter.eval, Pi.add_apply, mul_add, Finset.sum_add_distrib]
  ring

theorem Meter.eval_zero : m.eval 0 = 0 := by simp [Meter.eval]

theorem Meter.eval_sum {α : Type} (s : Finset α) (L : α → Ledger G) :
    m.eval (∑ a ∈ s, L a) = ∑ a ∈ s, m.eval (L a) := by
  classical
  induction s using Finset.induction_on with
  | empty => simp [Meter.eval_zero]
  | insert a s ha ih => rw [Finset.sum_insert ha, Finset.sum_insert ha, Meter.eval_add, ih]

theorem Meter.eval_smul (n : ℕ) (L : Ledger G) : m.eval (n • L) = n * m.eval L := by
  induction n with
  | zero => simp [Meter.eval_zero]
  | succ n ih => rw [succ_nsmul, Meter.eval_add, ih]; ring

theorem Meter.eval_mono {L L' : Ledger G} (h : L ≤ L') : m.eval L ≤ m.eval L' := by
  unfold Meter.eval
  gcongr with g
  · exact h _
  · exact h _
  · exact h _
  · exact h _

theorem Meter.eval_mono_meter {m m' : Meter G} (h : m ≤ m') (L : Ledger G) : m.eval L ≤ m'.eval L := by
  unfold Meter.eval
  gcongr with g
  · exact h _
  · exact h _
  · exact h _
  · exact h _

theorem Meter.eval_entry [DecidableEq G] (e : Entry G) (L : Ledger G) : (Meter.entry e).eval L = L e := by
  cases e <;> simp [Meter.eval, Meter.entry, Finset.sum_ite_eq']

theorem Meter.eval_priced (p : G → ℕ) (L : Ledger G) :
    (Meter.priced p).eval L = ∑ g, p g * L (.gates g) := by
  simp [Meter.eval, Meter.priced]

omit [Fintype G] in
theorem Meter.noCrossing_priced (p : G → ℕ) : (Meter.priced p).NoCrossing := ⟨rfl, rfl⟩

omit [Fintype G] in
theorem Meter.noCrossing_held [DecidableEq G] : (Meter.entry (.held : Entry G)).NoCrossing := by
  simp [Meter.NoCrossing, Meter.entry]

end Meter

variable {Λ : Type} [Fintype Λ] [DecidableEq Λ] (A : Annotation Λ G)

/-- A region's ledger is the sum of its ticks' ledgers. -/
theorem ledger_eq_sum_ticks (p : Finset Λ) (T : Finset ℕ) :
    A.ledger ⟨p, T⟩ = ∑ t ∈ T, A.ledger ⟨p, {t}⟩ := by
  funext e
  cases e <;> simp [Annotation.ledger, Finset.sum_apply]

/-- **Composition over time.** A part's ledger over two disjoint sets of ticks is the sum of its ledgers over each. -/
theorem ledger_union_ticks (p : Finset Λ) {T T' : Finset ℕ} (h : Disjoint T T') :
    A.ledger ⟨p, T ∪ T'⟩ = A.ledger ⟨p, T⟩ + A.ledger ⟨p, T'⟩ := by
  funext e
  cases e <;> simp [Annotation.ledger, Finset.sum_union h]

/-- A part's ledger over two consecutive windows is the sum of its ledgers over each. -/
theorem ledger_append (p : Finset Λ) (s a b : ℕ) :
    A.ledger (Region.ofWindow p ⟨s, a + b⟩) =
      A.ledger (Region.ofWindow p ⟨s, a⟩) + A.ledger (Region.ofWindow p ⟨s + a, b⟩) := by
  have hT : (⟨s, a + b⟩ : Window).ticks = (⟨s, a⟩ : Window).ticks ∪ (⟨s + a, b⟩ : Window).ticks := by
    simp only [Window.ticks]
    rw [Finset.Ico_union_Ico_eq_Ico (by omega) (by omega), Nat.add_assoc]
  have hd : Disjoint (⟨s, a⟩ : Window).ticks (⟨s + a, b⟩ : Window).ticks := by
    simp only [Window.ticks]
    exact Finset.Ico_disjoint_Ico_consecutive _ _ _
  simp only [Region.ofWindow]
  rw [hT, ledger_union_ticks A p hd]

/-- Every entry of a part's ledger grows with its ticks. -/
theorem ledger_mono_ticks (p : Finset Λ) {T T' : Finset ℕ} (h : T ⊆ T') : A.ledger ⟨p, T⟩ ≤ A.ledger ⟨p, T'⟩ := by
  intro e
  cases e <;> exact Finset.sum_le_sum_of_subset h

/-- A part's gates grow with the part. -/
theorem ledger_mono_part_gates {p q : Finset Λ} (h : p ⊆ q) (T : Finset ℕ) (g : G) :
    A.ledger ⟨p, T⟩ (.gates g) ≤ A.ledger ⟨q, T⟩ (.gates g) :=
  Finset.sum_le_sum fun _ _ => Finset.sum_le_sum_of_subset h

/-- A part's bits held grow with the part. -/
theorem ledger_mono_part_held {p q : Finset Λ} (h : p ⊆ q) (T : Finset ℕ) :
    A.ledger ⟨p, T⟩ .held ≤ A.ledger ⟨q, T⟩ .held :=
  Finset.sum_le_sum fun _ _ => Finset.sum_le_sum_of_subset h

/-- **Composition over space, gates.** Disjoint parts' gates add. -/
theorem ledger_union_part_gates {p q : Finset Λ} (h : Disjoint p q) (T : Finset ℕ) (g : G) :
    A.ledger ⟨p ∪ q, T⟩ (.gates g) = A.ledger ⟨p, T⟩ (.gates g) + A.ledger ⟨q, T⟩ (.gates g) := by
  simp [Annotation.ledger, Finset.sum_union h, Finset.sum_add_distrib]

/-- **Composition over space, bits held.** Disjoint parts' bits held add. -/
theorem ledger_union_part_held {p q : Finset Λ} (h : Disjoint p q) (T : Finset ℕ) :
    A.ledger ⟨p ∪ q, T⟩ .held = A.ledger ⟨p, T⟩ .held + A.ledger ⟨q, T⟩ .held := by
  simp [Annotation.ledger, Finset.sum_union h, Finset.sum_add_distrib]

/-- **Crossings only subadd over space.** The bits crossing into the union of disjoint parts are at most those crossing
into each: a bit moving from one part to the other crosses into one of them, and into their union not at all. -/
theorem ledger_union_part_crossIn_le {p q : Finset Λ} (h : Disjoint p q) (T : Finset ℕ) :
    A.ledger ⟨p ∪ q, T⟩ .crossIn ≤ A.ledger ⟨p, T⟩ .crossIn + A.ledger ⟨q, T⟩ .crossIn := by
  have hp : (p ∪ q)ᶜ ⊆ pᶜ := Finset.compl_subset_compl.mpr Finset.subset_union_left
  have hq : (p ∪ q)ᶜ ⊆ qᶜ := Finset.compl_subset_compl.mpr Finset.subset_union_right
  simp only [Annotation.ledger]
  rw [← Finset.sum_add_distrib]
  refine Finset.sum_le_sum fun t _ => ?_
  simp only [Finset.sum_union h, Finset.sum_add_distrib]
  exact add_le_add (Finset.sum_le_sum_of_subset hp) (Finset.sum_le_sum_of_subset hq)

/-- The bits crossing out of the union of disjoint parts are at most those crossing out of each. -/
theorem ledger_union_part_crossOut_le {p q : Finset Λ} (h : Disjoint p q) (T : Finset ℕ) :
    A.ledger ⟨p ∪ q, T⟩ .crossOut ≤ A.ledger ⟨p, T⟩ .crossOut + A.ledger ⟨q, T⟩ .crossOut := by
  have hp : (p ∪ q)ᶜ ⊆ pᶜ := Finset.compl_subset_compl.mpr Finset.subset_union_left
  have hq : (p ∪ q)ᶜ ⊆ qᶜ := Finset.compl_subset_compl.mpr Finset.subset_union_right
  simp only [Annotation.ledger]
  rw [← Finset.sum_add_distrib]
  refine Finset.sum_le_sum fun t _ => ?_
  rw [Finset.sum_union h]
  exact add_le_add (Finset.sum_le_sum fun _ _ => Finset.sum_le_sum_of_subset hp)
    (Finset.sum_le_sum fun _ _ => Finset.sum_le_sum_of_subset hq)

/-- A meter that reads no crossing adds over disjoint parts. -/
theorem eval_union_part [Fintype G] {m : Meter G} (hm : m.NoCrossing) {p q : Finset Λ} (h : Disjoint p q)
    (T : Finset ℕ) : m.eval (A.ledger ⟨p ∪ q, T⟩) = m.eval (A.ledger ⟨p, T⟩) + m.eval (A.ledger ⟨q, T⟩) := by
  simp only [Meter.eval, hm.1, hm.2, zero_mul, add_zero, ledger_union_part_gates A h, ledger_union_part_held A h,
    mul_add, Finset.sum_add_distrib]
  ring

/-- The ledger of a circuit made of accounts' shares is the sum of the shares' ledgers. -/
theorem ledger_sum {α : Type} [Fintype α] (A : α → Annotation Λ G) (ρ : Region Λ) :
    (Annotation.sum A).ledger ρ = ∑ a, (A a).ledger ρ := by
  funext e
  cases e <;> simp only [Annotation.ledger, Annotation.sum, Finset.sum_apply] <;>
    simp only [Finset.sum_comm (γ := α)]

/-- **A capacity over a region.** A part with capacity `c` per tick has ledger at most `n • c` over a region of `n`
ticks. -/
theorem ledger_le_card_smul {X : Type} {A : X → Annotation Λ G} {p : Finset Λ} {c : Ledger G}
    (hc : Capacity A p c) (x : X) (T : Finset ℕ) : (A x).ledger ⟨p, T⟩ ≤ T.card • c := by
  rw [ledger_eq_sum_ticks, ← Finset.sum_const]
  exact Finset.sum_le_sum fun t _ => hc x t

/-- A meter of a region of a part with capacity `c` per tick is at most the number of ticks times the meter of `c`. -/
theorem eval_le_capacity [Fintype G] {X : Type} {A : X → Annotation Λ G} {p : Finset Λ} {c : Ledger G}
    (hc : Capacity A p c) (m : Meter G) (x : X) (T : Finset ℕ) :
    m.eval ((A x).ledger ⟨p, T⟩) ≤ T.card * m.eval c := by
  rw [← Meter.eval_smul]
  exact Meter.eval_mono m (ledger_le_card_smul hc x T)

end Accounting.Ledger
