import Proofs.Core.NaiveStrassen.Sound

/-!
# Dominance and the size form (NaiveStrassen)

The other three instances reach the RTX PRO 6000 sheet's table through `Dom`: each of their atoms is covered by one of
the sheet's, no heavier and holding wherever it holds (`covers`). The size form follows from `bound`: Theorem 1 to
`thmL = 12`, the least cost above `8^13` at depth 13, and `cost ≥ 7^l` from depth 14 on, where `7^14 > 2^39`.
-/

namespace Verity.Accel.NaiveStrassen.Packed

open Verity.Accel.NaiveStrassen

/-! ## Dominance -/

/-- `f'` covers `f`: no heavier, a cap at least as large, and for a float format a `k` no larger and a `w` no
smaller; an integer `f'` holds on its whole cap. -/
def covers (f' f : Atom) : Bool :=
  f'.weight8 ≤ f.weight8 && f.cap ≤ f'.cap &&
    match f'.coupling, f.coupling with
    | none, _ => true
    | some (k', w'), some (k, w) => k' ≤ k && w ≤ w'
    | some _, none => false

theorem covers_holds {f' f : Atom} (hc : covers f' f = true) {a b : Nat} (h : f.holds a b = true) :
    f'.holds a b = true := by
  rcases f with ⟨x, c, _ | ⟨k, w⟩⟩ <;> rcases f' with ⟨x', c', _ | ⟨k', w'⟩⟩ <;>
    simp only [covers, Atom.holds, Bool.and_eq_true, decide_eq_true_eq, Bool.and_true, Bool.and_false,
      Bool.false_eq_true] at hc h ⊢
  · omega
  · omega
  · obtain ⟨⟨-, hcap⟩, hk, hw⟩ := hc
    obtain ⟨⟨ha, hb⟩, hkw⟩ := h
    refine ⟨⟨by omega, by omega⟩, Nat.le_trans (Nat.mul_le_mul_right _ (Nat.mul_le_mul_right _ hk))
      (Nat.le_trans hkw (Nat.pow_le_pow_right (by decide) hw))⟩

theorem dom_of_covers {fs : List Atom} (h : (fs.all fun f => rtx6000Sheet.any fun f' => covers f' f) = true) :
    Dom fs := by
  intro f hf a b hh
  obtain ⟨f', hf', hc⟩ := List.any_eq_true.mp (List.all_eq_true.mp h f hf)
  refine Nat.le_trans (atomMin_le hf' (covers_holds hc hh)) ?_
  simp only [covers, Bool.and_eq_true, decide_eq_true_eq] at hc
  exact hc.1.1

theorem dom_rtx6000Sheet : Dom rtx6000Sheet := fun _ hf _ _ h => atomMin_le hf h

theorem dom_h200Sheet : Dom h200Sheet := dom_of_covers (by decide)

theorem dom_h200Measured : Dom h200Measured := dom_of_covers (by decide)

theorem dom_rtx6000Measured : Dom rtx6000Measured := dom_of_covers (by decide)

/-! ## Every schedule costs at least `7^l` -/

/-- With every atom at least an eighth of an s8 MAC, a schedule at depth `l` costs at least `7^l`: an atom costs
`8^l` or more, and every scheme has at least seven products. -/
theorem cost_ge_pow7 {fs : List Atom} (hw : ∀ f ∈ fs, 1 ≤ f.weight8) :
    ∀ (S : Sched) (l a b : Nat), S.valid fs schemes 0 l a b = true → 7 ^ l ≤ S.cost l := by
  intro S
  induction S with
  | atom f =>
    intro l a b hv
    simp only [Sched.valid, Bool.and_eq_true] at hv
    simp only [Sched.cost]
    exact Nat.le_trans (Nat.pow_le_pow_left (by decide) l)
      (Nat.le_mul_of_pos_right _ (hw f (mem_of_contains hv.1)))
  | splitA d x y ihx _ =>
    intro l a b hv
    simp only [Sched.valid, Bool.and_eq_true] at hv
    simp only [Sched.cost]
    exact Nat.le_trans (ihx l _ _ hv.1.2) (Nat.le_add_right _ _)
  | splitB d x y ihx _ =>
    intro l a b hv
    simp only [Sched.valid, Bool.and_eq_true] at hv
    simp only [Sched.cost]
    exact Nat.le_trans (ihx l _ _ hv.1.2) (Nat.le_add_right _ _)
  | scheme s kids ih =>
    intro l a b hv
    cases l with
    | zero => simp [Sched.valid] at hv
    | succ l =>
      simp only [Sched.valid, Bool.and_eq_true, List.all_eq_true, List.mem_finRange, true_implies] at hv
      obtain ⟨hs, hk⟩ := hv
      have hcost : (Sched.scheme s kids).cost (l + 1) = ((List.finRange s.R).map fun t => (kids t).cost l).sum := by
        simp only [Sched.cost, Nat.add_sub_cancel, foldl_add, Nat.zero_add]
      rw [hcost]
      have hall : ∀ x ∈ (List.finRange s.R).map (fun t => (kids t).cost l), 7 ^ l ≤ x := by
        intro x hx
        obtain ⟨t, -, rfl⟩ := List.mem_map.mp hx
        exact ih t l _ _ (hk t)
      have hR : 7 ≤ s.R := by rcases scheme_cases hs with rfl | rfl | rfl <;> decide
      have hsum := mul_le_sum hall
      rw [List.length_map, List.length_finRange] at hsum
      rw [Nat.pow_succ']
      exact Nat.le_trans (Nat.mul_le_mul_right _ hR) hsum

/-! ## The size form -/

theorem cube_bound {n m b C : Nat} (hb : 0 < b) (hn : n ≤ m * b) (hC : 8 * m ^ 3 < C) : 8 * n ^ 3 < b ^ 3 * C := by
  have h1 : n ^ 3 ≤ m ^ 3 * b ^ 3 := Nat.mul_pow m b 3 ▸ Nat.pow_le_pow_left hn 3
  have hb3 : 0 < b ^ 3 := Nat.pow_pos hb
  calc 8 * n ^ 3 ≤ 8 * m ^ 3 * b ^ 3 := by rw [Nat.mul_assoc]; exact Nat.mul_le_mul_left 8 h1
    _ < C * b ^ 3 := Nat.mul_lt_mul_of_pos_right hC hb3
    _ = b ^ 3 * C := Nat.mul_comm _ _

theorem eight_mul_cube (l : Nat) : 8 * (2 ^ l) ^ 3 = 8 ^ (l + 1) := by
  rw [← Nat.pow_mul, Nat.mul_comm l 3, Nat.pow_mul]
  show 8 * 8 ^ l = 8 ^ (l + 1)
  exact (Nat.pow_succ' (m := 8) (n := l)).symm

/-- **The size form** at `c = 4096 = 2^thmL`, from the tables: `n ≤ 2^l b` gives `8 n³ ≤ 8^(l+1) b³`, which Theorem 1
beats to depth 12; `n ≤ 4096 b` gives `8 n³ ≤ 2^39 b³ = 8^13 b³`, which the least cost at depth 13 beats, and
`7^l ≥ 7^14 > 2^39` beyond. -/
theorem upTo_of_check (h : checkAll = true) {fs : List Atom} (hD : Dom fs) (hw : ∀ f ∈ fs, 1 ≤ f.weight8) :
    NaiveBeatsStrassenUpTo fs 4096 := by
  intro l b n S hv hf hn1 hnl hnc
  have hb : 0 < b := by
    rcases Nat.eq_zero_or_pos b with rfl | hb
    · rw [Nat.mul_zero] at hnl; omega
    · exact hb
  rcases Nat.lt_or_ge l 13 with hl | hl
  · have h1 := (bound h hD (l := l) (by unfold topL; omega) hv).1 (by unfold thmL; omega) hf
    exact cube_bound hb hnl (by rw [eight_mul_cube]; omega)
  rcases Nat.lt_or_ge l 14 with hl' | hl'
  · obtain rfl : l = 13 := by omega
    have h13 := (bound h hD (l := 13) (by decide) hv).2 (by decide)
    have e : 8 * 4096 ^ 3 = 8 ^ 13 := by decide
    exact cube_bound hb hnc (by omega)
  · have h7 := cost_ge_pow7 hw S l 256 256 hv
    have h14 : 7 ^ 14 ≤ 7 ^ l := Nat.pow_le_pow_right (by decide) hl'
    have e : 8 * 4096 ^ 3 < 7 ^ 14 := by decide
    exact cube_bound hb hnc (by omega)

/-- Theorem 1 to `thmL`, from the tables. -/
theorem beats_of_check (h : checkAll = true) {fs : List Atom} (hD : Dom fs) : NaiveBeatsStrassen fs thmL :=
  fun _ hl _ hv hf => (bound h hD (Nat.le_trans hl (by decide)) hv).1 hl hf

end Verity.Accel.NaiveStrassen.Packed
