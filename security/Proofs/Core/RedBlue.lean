module

public import Definitions.Core.RedBlue
import Mathlib.Algebra.Order.BigOperators.Group.Finset

/-!
# A window of a device's run is a block within the device's limits

For an ordered run within the limits `H`, the gates `K` of any `Δ` consecutive cycles (`Run.window`) have
`|K| ≤ rate · Δ` (`window_size`), paths of at most `depth · Δ` gates (`window_depth`) and
`width(in(K)) ≤ chip + min(mem, bw · Δ) + cap · Δ` (`window_into`): a gate of an earlier cycle that the window reads
was red at the window's start, or was blue then and loaded inside it, and any other gate it reads is an input arriving
inside it. So a circuit in `feasible T H` has at most `rate · T` computed gates (`feasible_size`).
-/

@[expose] public section

namespace Verity.RedBlue

open Finset

variable {C : Circuit} {T : ℕ} (R : Run C T) {H : Limits}

namespace Run

/-- A value on the die or in device memory at the end of cycle `c` is of cycle `c` or earlier. -/
theorem cycle_le_of_mem (hO : R.Order) : ∀ c, ∀ x ∈ R.red c ∪ R.blue c, R.cycle x ≤ c := by
  intro c
  induction c with
  | zero => exact fun x hx => (hO.start x hx).le
  | succ c ih =>
    intro x hx
    rcases hO.kept c x hx with h | h
    · exact h.le
    · exact (ih x h).trans (Nat.le_succ c)

/-- A gate reads gates of its own cycle or earlier. -/
theorem cycle_reads_le (hO : R.Order) {g x : Fin C.N} (hx : x ∈ C.reads g) : R.cycle x ≤ R.cycle g := by
  rcases hO.reads g x hx with h | h
  · exact h.le
  · exact (R.cycle_le_of_mem hO _ x (mem_union_left _ h)).trans (Nat.sub_le _ _)

/-- A value of cycle `a` or earlier that the device holds later, it held at the end of cycle `a`. -/
theorem mem_start (hO : R.Order) {a : ℕ} {x : Fin C.N} (hx : R.cycle x ≤ a) :
    ∀ j, x ∈ R.red (a + j) ∪ R.blue (a + j) → x ∈ R.red a ∪ R.blue a := by
  intro j
  induction j with
  | zero => exact id
  | succ j ih =>
    intro hj
    rcases hO.kept (a + j) x hj with h | h
    · omega
    · exact ih h

/-- A value of cycle `a` or earlier on the die later was on the die at the end of cycle `a`, or was loaded since. -/
theorem red_start_or_loaded (hO : R.Order) {a : ℕ} {x : Fin C.N} (hx : R.cycle x ≤ a) :
    ∀ j, x ∈ R.red (a + j) → x ∈ R.red a ∨ ∃ i < j, x ∈ R.loaded (a + i) := by
  intro j
  induction j with
  | zero => exact fun h => Or.inl h
  | succ j ih =>
    intro hj
    by_cases hr : x ∈ R.red (a + j)
    · rcases ih hr with h | ⟨i, hi, h⟩
      · exact Or.inl h
      · exact Or.inr ⟨i, Nat.lt_succ_of_lt hi, h⟩
    · rcases hO.kept (a + j) x (mem_union_left _ hj) with h | h
      · omega
      · exact Or.inr ⟨j, Nat.lt_succ_self j,
          mem_filter.mpr ⟨hj, (mem_union.mp h).resolve_left hr, hr⟩⟩

theorem width_union_le (s t : Finset (Fin C.N)) : C.width (s ∪ t) ≤ C.width s + C.width t := by
  have h := sum_union_inter (s₁ := s) (s₂ := t) (f := C.w)
  simp only [Circuit.width]
  omega

theorem width_biUnion_range_le (n : ℕ) (f : ℕ → Finset (Fin C.N)) (b : ℕ) (hf : ∀ i, C.width (f i) ≤ b) :
    C.width ((range n).biUnion f) ≤ b * n := by
  induction n with
  | zero => simp [Circuit.width]
  | succ n ih =>
    rw [range_add_one, biUnion_insert, Nat.mul_succ]
    have h1 := width_union_le (f n) ((range n).biUnion f)
    have h2 := hf n
    omega

/-- **A window's work**: `Δ` cycles compute at most `rate · Δ` gates. -/
theorem window_size (hW : R.Within H) (a Δ : ℕ) : C.size (R.window a Δ) ≤ H.rate * Δ := by
  have hsub : R.window a Δ ⊆ (range Δ).biUnion fun i => R.gatesAt (a + 1 + i) := by
    intro g hg
    simp only [window, mem_filter] at hg
    simp only [mem_biUnion, mem_range, gatesAt, mem_filter]
    exact ⟨R.cycle g - (a + 1), by omega, hg.1, by omega⟩
  calc C.size (R.window a Δ) ≤ ((range Δ).biUnion fun i => R.gatesAt (a + 1 + i)).card := card_le_card hsub
    _ ≤ ∑ i ∈ range Δ, (R.gatesAt (a + 1 + i)).card := card_biUnion_le
    _ ≤ ∑ _i ∈ range Δ, H.rate := sum_le_sum fun i _ => hW.rate _
    _ = H.rate * Δ := by rw [sum_const, card_range, smul_eq_mul, mul_comm]

/-- **A window's inputs**: what `Δ` cycles read from outside themselves is at most the die's state, what is loaded
from device memory (never more than it holds) and what arrives. -/
theorem window_into (hO : R.Order) (hW : R.Within H) (a Δ : ℕ) :
    C.width (C.into (R.window a Δ)) ≤ H.chip + min H.mem (H.bw * Δ) + H.cap * Δ := by
  classical
  have hsub : C.into (R.window a Δ) ⊆
      R.red a ∪ (((range Δ).biUnion fun i => R.loaded (a + i)) ∩ R.blue a) ∪
        (range Δ).biUnion fun i => R.inputsAt (a + 1 + i) := by
    intro x hx
    simp only [Circuit.into, mem_filter, mem_univ, true_and] at hx
    obtain ⟨hxK, g, hg, hxg⟩ := hx
    simp only [window, mem_filter] at hg
    have hle := R.cycle_reads_le hO hxg
    by_cases hc : R.cycle x ≤ a
    · have hred : x ∈ R.red (R.cycle g - 1) := (hO.reads g x hxg).resolve_left (by omega)
      obtain ⟨j, hj⟩ : ∃ j, R.cycle g - 1 = a + j := ⟨R.cycle g - 1 - a, by omega⟩
      rw [hj] at hred
      rcases R.red_start_or_loaded hO hc j hred with h | ⟨i, hi, h⟩
      · exact mem_union_left _ (mem_union_left _ h)
      · rcases mem_union.mp (R.mem_start hO hc i (mem_union_right _ (mem_filter.mp h).2.1)) with h' | h'
        · exact mem_union_left _ (mem_union_left _ h')
        · exact mem_union_left _ (mem_union_right _
            (mem_inter.mpr ⟨mem_biUnion.mpr ⟨i, mem_range.mpr (by omega), h⟩, h'⟩))
    · refine mem_union_right _ (mem_biUnion.mpr ⟨R.cycle x - (a + 1), mem_range.mpr (by omega), ?_⟩)
      simp only [inputsAt, mem_filter, mem_univ, true_and]
      refine ⟨?_, by omega⟩
      by_contra hin
      refine hxK ?_
      simp only [window, Circuit.computed, mem_filter, mem_univ, true_and]
      exact ⟨by simpa using hin, by omega, by omega⟩
  have h1 : C.width (((range Δ).biUnion fun i => R.loaded (a + i)) ∩ R.blue a) ≤ min H.mem (H.bw * Δ) :=
    le_min ((sum_le_sum_of_subset inter_subset_right).trans (hW.mem a))
      ((sum_le_sum_of_subset inter_subset_left).trans
        (width_biUnion_range_le Δ _ H.bw fun i => hW.bw _))
  have h2 : C.width ((range Δ).biUnion fun i => R.inputsAt (a + 1 + i)) ≤ H.cap * Δ :=
    width_biUnion_range_le Δ _ H.cap fun i => hW.cap _ (by omega)
  have h3 : C.width (C.into (R.window a Δ)) ≤ C.width (R.red a ∪
      (((range Δ).biUnion fun i => R.loaded (a + i)) ∩ R.blue a) ∪
        (range Δ).biUnion fun i => R.inputsAt (a + 1 + i)) := sum_le_sum_of_subset hsub
  have h4 := width_union_le (R.red a ∪ (((range Δ).biUnion fun i => R.loaded (a + i)) ∩ R.blue a))
    ((range Δ).biUnion fun i => R.inputsAt (a + 1 + i))
  have h5 := width_union_le (R.red a) (((range Δ).biUnion fun i => R.loaded (a + i)) ∩ R.blue a)
  have h6 := hW.chip a
  omega

theorem depthTo_mem {B : Finset (Fin C.N)} {g : Fin C.N} (hg : g ∈ B) :
    Structure.depthTo C B g = (C.reads g).sup (Structure.depthTo C B) + 1 := by
  rw [Structure.depthTo]
  simp only [hg, ↓reduceIte]
  congr 1
  exact sup_congr rfl fun a ha => by simp only [C.reads_lt g a ha, ↓reduceDIte]

theorem depthTo_not_mem {B : Finset (Fin C.N)} {g : Fin C.N} (hg : g ∉ B) : Structure.depthTo C B g = 0 := by
  rw [Structure.depthTo]
  simp only [hg, ↓reduceIte]

theorem depthTo_gatesAt_le (hW : R.Within H) {g : Fin C.N} (hg : g ∈ C.computed) :
    Structure.depthTo C (R.gatesAt (R.cycle g)) g ≤ H.depth := by
  have hm : g ∈ R.gatesAt (R.cycle g) := mem_filter.mpr ⟨hg, rfl⟩
  exact (le_sup (f := Structure.depthTo C (R.gatesAt (R.cycle g))) hm).trans (hW.depth (R.cycle g))

/-- A path inside a window ending at `g` has at most `depth` gates in each of the window's cycles before `g`'s, and
then those of its last stretch inside `g`'s cycle. -/
theorem depthTo_window (hO : R.Order) (hW : R.Within H) (a Δ : ℕ) (g : Fin C.N) :
    g ∈ R.window a Δ → Structure.depthTo C (R.window a Δ) g ≤
      H.depth * (R.cycle g - a - 1) + Structure.depthTo C (R.gatesAt (R.cycle g)) g := by
  refine WellFoundedLT.induction (motive := fun g => g ∈ R.window a Δ → Structure.depthTo C (R.window a Δ) g ≤
      H.depth * (R.cycle g - a - 1) + Structure.depthTo C (R.gatesAt (R.cycle g)) g) g ?_
  intro g ih hg
  have hgK := hg
  simp only [window, mem_filter] at hgK
  have hgG : g ∈ R.gatesAt (R.cycle g) := mem_filter.mpr ⟨hgK.1, rfl⟩
  rw [depthTo_mem hg, depthTo_mem hgG, ← Nat.add_assoc]
  refine Nat.add_le_add_right (Finset.sup_le fun x hx => ?_) 1
  by_cases hxK : x ∈ R.window a Δ
  · have hx' := hxK
    simp only [window, mem_filter] at hx'
    have ihx := ih x (C.reads_lt g x hx) hxK
    rcases (R.cycle_reads_le hO hx).eq_or_lt with heq | hlt
    · rw [heq] at ihx
      exact ihx.trans (Nat.add_le_add_left (le_sup (f := Structure.depthTo C (R.gatesAt (R.cycle g))) hx) _)
    · have hm : H.depth * (R.cycle x - a - 1) + H.depth ≤ H.depth * (R.cycle g - a - 1) := by
        rw [← Nat.mul_succ]
        exact Nat.mul_le_mul_left _ (by omega)
      exact (ihx.trans (Nat.add_le_add_left (R.depthTo_gatesAt_le hW hx'.1) _)).trans
        (hm.trans (Nat.le_add_right _ _))
  · rw [depthTo_not_mem hxK]
    exact Nat.zero_le _

/-- **A window's depth**: a path inside `Δ` cycles has at most `depth · Δ` gates. -/
theorem window_depth (hO : R.Order) (hW : R.Within H) (a Δ : ℕ) :
    Structure.depth C (R.window a Δ) ≤ H.depth * Δ := by
  unfold Structure.depth
  refine Finset.sup_le fun g hg => ?_
  have hgK := hg
  simp only [window, mem_filter] at hgK
  have hm : H.depth * (R.cycle g - a - 1) + H.depth ≤ H.depth * Δ := by
    rw [← Nat.mul_succ]
    exact Nat.mul_le_mul_left _ (by omega)
  exact ((R.depthTo_window hO hW a Δ g hg).trans (Nat.add_le_add_left (R.depthTo_gatesAt_le hW hgK.1) _)).trans hm

end Run

/-- **A device's work over `T` cycles**: a circuit it can run has at most `rate · T` computed gates. -/
theorem feasible_size {T : ℕ} {H : Limits} {C : Circuit} (hC : C ∈ feasible T H) :
    C.size C.computed ≤ H.rate * T := by
  obtain ⟨R, -, hW⟩ := hC
  have h : C.computed = R.window 0 T := by
    ext g
    simp only [Run.window, mem_filter, Nat.zero_add]
    exact ⟨fun hg => ⟨hg, R.computed_pos g (mem_filter.mp hg).2, R.cycle_le g⟩, fun hg => hg.1⟩
  rw [h]
  exact R.window_size hW 0 T

end Verity.RedBlue
