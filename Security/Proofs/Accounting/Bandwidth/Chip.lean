import Mathlib
import Definitions.Accounting.Bandwidth

/-!
# Chip programs: congruence, replay from a trace, and counting

`run_congr`: changing the oracle off the points a run queries changes neither the run nor its queries.
`replay_trace`: given the oracle, the run's queries are a function of the chip program and its trace, the bits device
memory, side sources and regeneration answered with. The counting lemmas turn per-fiber bounds into bounds on `pr`.
-/

namespace Bandwidth

open Pous (Bits pr)
open scoped ENNReal

namespace Chip

variable {N b h : ℕ} {α : Type}

/-- The bits a run was answered with, other than the oracle's: device memory, side sources and regenerated blocks. -/
def trace (w : World N b h) : Chip N b h α → List Bool
  | done _ => []
  | tag x k => (k (w.oracle x)).trace w
  | ingress a k => w.mem a :: (k (w.mem a)).trace w
  | side a k => w.side a :: (k (w.side a)).trace w
  | regen j k => List.ofFn (w.block j) ++ (k (w.block j)).trace w

/-- The queries a run makes, recomputed from the oracle and a stream of the bits it was answered with otherwise. -/
def replay (O : Oracle N b h) : Chip N b h α → (ℕ → Bool) → List (Fin N × Bits b)
  | done _, _ => []
  | tag x k, f => x :: (k (O x)).replay O f
  | ingress _ k, f => (k (f 0)).replay O (fun n => f (n + 1))
  | side _ k, f => (k (f 0)).replay O (fun n => f (n + 1))
  | regen _ k, f => (k (fun i => f i)).replay O (fun n => f (n + b))

theorem trace_length (w : World N b h) (c : Chip N b h α) :
    (c.trace w).length = c.ingressBits w + c.sideBits w + c.regens w * b := by
  induction c with
  | done a => simp [trace, ingressBits, sideBits, regens]
  | tag x k ih => simp [trace, ingressBits, sideBits, regens, ih]
  | ingress a k ih => simp [trace, ingressBits, sideBits, regens, ih]; omega
  | side a k ih => simp [trace, ingressBits, sideBits, regens, ih]; omega
  | regen j k ih => simp [trace, ingressBits, sideBits, regens, ih, Nat.succ_mul]; omega

theorem replay_trace (w : World N b h) (c : Chip N b h α) (l : List Bool) :
    c.replay w.oracle (fun n => (c.trace w ++ l).getD n false) = c.queries w := by
  induction c generalizing l with
  | done a => simp [replay, queries]
  | tag x k ih => simp only [replay, queries, trace, ih]
  | ingress a k ih => simpa [replay, queries, trace] using ih _ l
  | side a k ih => simpa [replay, queries, trace] using ih _ l
  | regen j k ih =>
    simp only [replay, queries, trace, List.append_assoc]
    have hb : (fun i : Fin b => (List.ofFn (w.block j) ++ ((k (w.block j)).trace w ++ l)).getD i false) =
        w.block j := by
      funext i
      rw [List.getD_eq_getElem?_getD, List.getElem?_append_left (by simp)]
      simp
    have hs : (fun n => (List.ofFn (w.block j) ++ ((k (w.block j)).trace w ++ l)).getD (n + b) false) =
        fun n => ((k (w.block j)).trace w ++ l).getD n false := by
      funext n
      rw [List.getD_eq_getElem?_getD, List.getElem?_append_right (by simp), List.getD_eq_getElem?_getD]
      simp
    rw [hb, hs, ih]

/-- The side-source bits a run read, in order. -/
def sideTrace (w : World N b h) : Chip N b h α → List Bool
  | done _ => []
  | tag x k => (k (w.oracle x)).sideTrace w
  | ingress a k => (k (w.mem a)).sideTrace w
  | side a k => w.side a :: (k (w.side a)).sideTrace w
  | regen j k => (k (w.block j)).sideTrace w

/-- A run's value, recomputed from an oracle, device memory, the store and a stream of the side-source bits it read. -/
def runSide (O : Oracle N b h) (mem : ℕ → Bool) (blk : Store N b) : Chip N b h α → (ℕ → Bool) → α
  | done a, _ => a
  | tag x k, f => (k (O x)).runSide O mem blk f
  | ingress a k, f => (k (mem a)).runSide O mem blk f
  | side _ k, f => (k (f 0)).runSide O mem blk (fun n => f (n + 1))
  | regen j k, f => (k (blk j)).runSide O mem blk f

theorem sideTrace_length (w : World N b h) (c : Chip N b h α) : (c.sideTrace w).length = c.sideBits w := by
  induction c with
  | done a => simp [sideTrace, sideBits]
  | tag x k ih => simp [sideTrace, sideBits, ih]
  | ingress a k ih => simp [sideTrace, sideBits, ih]
  | side a k ih => simp [sideTrace, sideBits, ih]
  | regen j k ih => simp [sideTrace, sideBits, ih]

/-- Given device memory and the store, the run is a function of the oracle at its queries and the side bits it read:
any oracle that agrees with the world's at the run's queries recomputes it from those bits. -/
theorem runSide_sideTrace (w : World N b h) (O' : Oracle N b h) (c : Chip N b h α)
    (ho : ∀ x ∈ c.queries w, O' x = w.oracle x) :
    c.runSide O' w.mem w.block (fun n => (c.sideTrace w).getD n false) = c.run w := by
  induction c with
  | done a => simp [runSide, run]
  | tag x k ih =>
    simp only [queries, List.mem_cons, forall_eq_or_imp] at ho
    obtain ⟨hx, ho⟩ := ho
    simp only [runSide, run, sideTrace, hx]
    exact ih _ ho
  | ingress a k ih => exact ih _ ho
  | side a k ih =>
    simp only [runSide, run, sideTrace, List.getD_cons_zero, List.getD_cons_succ]
    exact ih _ ho
  | regen j k ih => exact ih _ ho

/-- Changing the oracle off the points a run queries changes neither the run nor its queries. -/
theorem run_congr (w : World N b h) (O' : Oracle N b h) (c : Chip N b h α)
    (ho : ∀ x ∈ c.queries w, O' x = w.oracle x) :
    c.run { w with oracle := O' } = c.run w ∧ c.queries { w with oracle := O' } = c.queries w := by
  induction c with
  | done a => simp [run, queries]
  | tag x k ih =>
    simp only [queries, List.mem_cons, forall_eq_or_imp] at ho
    obtain ⟨hx, ho⟩ := ho
    simp only [run, queries, hx]
    obtain ⟨h1, h2⟩ := ih _ ho
    exact ⟨h1, by rw [h2]⟩
  | ingress a k ih => exact ih _ ho
  | side a k ih => exact ih _ ho
  | regen j k ih => exact ih _ ho

end Chip

/-! ## Counting -/

theorem card_filter_prod_le {α β : Type} [Fintype α] [Fintype β] (P : α × β → Prop) [DecidablePred P]
    (d m : ℕ) (h : ∀ a, (Finset.univ.filter (fun y => P (a, y))).card * d ≤ m) :
    (Finset.univ.filter P).card * d ≤ Fintype.card α * m := by
  have e : (Finset.univ.filter P).card = ∑ a, (Finset.univ.filter (fun y => P (a, y))).card := by
    rw [Finset.card_filter, Fintype.sum_prod_type]
    simp only [Finset.card_filter]
  rw [e, Finset.sum_mul]
  calc ∑ a, (Finset.univ.filter (fun y => P (a, y))).card * d ≤ ∑ _a : α, m := Finset.sum_le_sum (fun a _ => h a)
    _ = Fintype.card α * m := by simp

theorem card_filter_prod_le' {α β : Type} [Fintype α] [Fintype β] (P : α × β → Prop) [DecidablePred P]
    (d m : ℕ) (h : ∀ y, (Finset.univ.filter (fun a => P (a, y))).card * d ≤ m) :
    (Finset.univ.filter P).card * d ≤ m * Fintype.card β := by
  have e : (Finset.univ.filter P).card = ∑ y, (Finset.univ.filter (fun a => P (a, y))).card := by
    rw [Finset.card_filter, Fintype.sum_prod_type_right]
    simp only [Finset.card_filter]
  rw [e, Finset.sum_mul]
  calc ∑ y, (Finset.univ.filter (fun a => P (a, y))).card * d ≤ ∑ _y : β, m := Finset.sum_le_sum (fun y _ => h y)
    _ = m * Fintype.card β := by simp [mul_comm]

/-- A set of functions that is determined by its values off one point has at most one member per value there. -/
theorem card_line_le {X Y : Type} [Fintype X] [DecidableEq X] [Fintype Y] (x₀ : X) (s : Finset (X → Y))
    (hs : ∀ f ∈ s, ∀ g ∈ s, (∀ x, x ≠ x₀ → f x = g x) → f = g) :
    s.card * Fintype.card Y ≤ Fintype.card (X → Y) := by
  rw [← Finset.card_univ (α := Y), ← Finset.card_product, ← Finset.card_univ (α := X → Y)]
  refine Finset.card_le_card_of_injOn (fun p => Function.update p.1 x₀ p.2) (fun _ _ => Finset.mem_univ _) ?_
  rintro ⟨f, y⟩ hf ⟨g, y'⟩ hg he
  simp only [Finset.coe_product, Set.mem_prod, Finset.mem_coe] at hf hg
  have hy : y = y' := by simpa using congrFun he x₀
  subst hy
  refine Prod.ext (hs f hf.1 g hg.1 (fun x hx => ?_)) rfl
  simpa [Function.update_of_ne hx] using congrFun he x

theorem pr_le_of_card {Ω : Type} [Fintype Ω] (P : Ω → Prop) [DecidablePred P] (d m : ℕ) (hd : d ≠ 0)
    (h : (Finset.univ.filter P).card * d ≤ m * Fintype.card Ω) : pr P ≤ (m : ℝ≥0∞) / d := by
  classical
  unfold pr
  rw [Finset.filter_congr_decidable]
  refine ENNReal.div_le_of_le_mul ?_
  have hd' : (d : ℝ≥0∞) ≠ 0 := by simp [hd]
  calc ((Finset.univ.filter P).card : ℝ≥0∞) = (Finset.univ.filter P).card * d * (d : ℝ≥0∞)⁻¹ := by
        rw [mul_assoc, ENNReal.mul_inv_cancel hd' (by simp), mul_one]
    _ ≤ m * Fintype.card Ω * (d : ℝ≥0∞)⁻¹ := by gcongr ?_ * _; exact_mod_cast h
    _ = m / d * Fintype.card Ω := by rw [div_eq_mul_inv, mul_right_comm]

theorem pr_mono {Ω : Type} [Fintype Ω] {P Q : Ω → Prop} (h : ∀ x, P x → Q x) : pr P ≤ pr Q := by
  classical
  unfold pr
  gcongr
  exact h _

open Classical in
/-- A probability over a store and live coins whose public part is a pair is the same over the store, the public
part's two halves and the draw, in that order. -/
theorem pr_coins_flat {S A B D : Type} [Fintype S] [Fintype A] [Fintype B] [Fintype D]
    (P : S × ((A × B) × D) → Prop) :
    pr P = pr (fun x : S × A × B × D => P (x.1, (x.2.1, x.2.2.1), x.2.2.2)) := by
  let e : S × A × B × D ≃ S × ((A × B) × D) := (Equiv.refl S).prodCongr (Equiv.prodAssoc A B D).symm
  unfold pr
  rw [Fintype.card_congr e, Finset.card_equiv e (t := Finset.univ.filter P) fun x => by
    simp only [Finset.mem_filter, Finset.mem_univ, true_and]; exact Iff.rfl]

theorem pr_or_le {Ω : Type} [Fintype Ω] (P Q : Ω → Prop) : pr (fun x => P x ∨ Q x) ≤ pr P + pr Q := by
  classical
  unfold pr
  rw [Finset.filter_congr_decidable, ← ENNReal.add_div]
  gcongr
  calc (((Finset.univ.filter (fun x => P x ∨ Q x)).card : ℕ) : ℝ≥0∞)
      ≤ ((Finset.univ.filter P ∪ Finset.univ.filter Q).card : ℕ) := by
        exact_mod_cast Finset.card_le_card (fun x hx => by
          simp only [Finset.mem_filter, Finset.mem_union, Finset.mem_univ, true_and] at hx ⊢; exact hx)
    _ ≤ _ := by exact_mod_cast Finset.card_union_le _ _

theorem pr_exists_le {Ω ι : Type} [Fintype Ω] [Fintype ι] (P : ι → Ω → Prop) :
    pr (fun x => ∃ i, P i x) ≤ ∑ i, pr (P i) := by
  classical
  unfold pr
  rw [Finset.filter_congr_decidable, ← ENNReal.sum_div]
  gcongr
  calc (((Finset.univ.filter (fun x => ∃ i, P i x)).card : ℕ) : ℝ≥0∞)
      ≤ ((Finset.univ.biUnion (fun i => Finset.univ.filter (P i))).card : ℕ) := by
        exact_mod_cast Finset.card_le_card (fun x hx => by
          simp only [Finset.mem_filter, Finset.mem_biUnion, Finset.mem_univ, true_and] at hx ⊢; exact hx)
    _ ≤ ∑ i, (((Finset.univ.filter (P i)).card : ℕ) : ℝ≥0∞) := by exact_mod_cast Finset.card_biUnion_le

end Bandwidth
