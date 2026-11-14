module

public import NciCheck.Record
public import Mathlib.Tactic
public import Proofs.Properties.Nci.RecordGrid

/-!
# What the record check's `below`, `inSlot` and `tally` count

* `below_count`: `below S q x s` is the number of indices below `x` whose slot `(i / q) % S` is `s`, for a row
  length `q` above 0 and a slot below `S`. It grows by one exactly at an index of slot `s` (`below_succ`), so it
  is monotone in `x` (`below_mono`).
* `inSlot_count`: `inSlot S gs lo hi s` is the number of units of `[lo, hi)` whose slot by the grids `gs` is `s`
  (`slotOf`), where no unit lies in two grids and every grid's row length is above 0. One more unit adds one
  exactly where its slot is `s` (`inSlot_succ`), and an empty range has none (`inSlot_empty`).
* `tally_eq`: a tally's entry for a kind is the sum, over the declarations of that kind, of each one's reader units
  in the slot, times the rows it covers of each, times a row's bits.
-/

@[expose] public section

namespace NciCheck.Record.SecurityProofs

open RecordText

/-- A number written `P · k + r` with `r < P` has quotient `k` and rest `r` by `P`. -/
private theorem div_mod_of_eq (P k r x : ℕ) (hr : r < P) (hx : x = P * k + r) : x / P = k ∧ x % P = r := by
  subst hx
  have hP : 0 < P := by omega
  exact ⟨by rw [Nat.mul_add_div hP, Nat.div_eq_of_lt hr, Nat.add_zero],
    by rw [Nat.mul_add_mod, Nat.mod_eq_of_lt hr]⟩

/-- One more index adds one to `below` exactly where the index's slot is `s`. -/
theorem below_succ (S q x s : ℕ) (hq : 0 < q) (hs : s < S) :
    below S q (x + 1) s = below S q x s + if x / q % S = s then 1 else 0 := by
  have hP : 0 < q * S := Nat.mul_pos hq (by omega)
  have hslot : x / q % S = x % (q * S) / q := (Nat.mod_mul_right_div_self x q S).symm
  have hdm := Nat.div_add_mod x (q * S)
  have hlt := Nat.mod_lt x hP
  have hsq : s * q + q ≤ q * S := by
    have e := Nat.mul_le_mul_right q (show s + 1 ≤ S by omega)
    rw [Nat.add_mul, Nat.one_mul, Nat.mul_comm S q] at e
    exact e
  unfold below
  rw [hslot]
  generalize x / (q * S) = k at hdm ⊢
  generalize x % (q * S) = r at hdm hlt ⊢
  have hkq : (k + 1) * q = k * q + q := by rw [Nat.add_mul, Nat.one_mul]
  by_cases hc : r + 1 < q * S
  · obtain ⟨e1, e2⟩ := div_mod_of_eq (q * S) k (r + 1) (x + 1) hc (by omega)
    rw [e1, e2]
    by_cases h : r / q = s
    · have := (Nat.div_eq_iff hq).1 h
      rw [ite_eq_left h]
      omega
    · have := mt (Nat.div_eq_iff hq).2 h
      rw [ite_eq_right h]
      omega
  · obtain ⟨e1, e2⟩ := div_mod_of_eq (q * S) (k + 1) 0 (x + 1) hP (by rw [Nat.mul_add, Nat.mul_one]; omega)
    rw [e1, e2, hkq]
    by_cases hlast : s + 1 = S
    · have hsq' : s * q + q = q * S := by rw [← hlast, Nat.mul_add, Nat.mul_one, Nat.mul_comm q s]
      have h : r / q = s := (Nat.div_eq_iff hq).2 (by omega)
      rw [ite_eq_left h]
      omega
    · have hsq' : s * q + q + q ≤ q * S := by
        have e := Nat.mul_le_mul_right q (show s + 1 + 1 ≤ S by omega)
        rw [Nat.add_mul, Nat.add_mul, Nat.one_mul, Nat.mul_comm S q] at e
        exact e
      have h : ¬ r / q = s := fun h => by
        have := (Nat.div_eq_iff hq).1 h
        omega
      rw [ite_eq_right h]
      omega

/-- `below` is monotone in the number of indices. -/
theorem below_mono (S q s : ℕ) (hq : 0 < q) (hs : s < S) {x x' : ℕ} (h : x ≤ x') :
    below S q x s ≤ below S q x' s := by
  obtain ⟨d, rfl⟩ := Nat.exists_eq_add_of_le h
  induction d with
  | zero => exact Nat.le_refl _
  | succ d ih =>
    have e := below_succ S q (x + d) s hq hs
    rw [← Nat.add_assoc, e]
    exact Nat.le_trans (ih (Nat.le_add_right _ _)) (Nat.le_add_right _ _)

/-- `below` counts the indices below `x` whose slot is `s`. -/
theorem below_count (S q x s : ℕ) (hq : 0 < q) (hs : s < S) :
    below S q x s = ((List.range x).filter fun i => i / q % S == s).length := by
  induction x with
  | zero =>
    unfold below
    rw [Nat.zero_div, Nat.zero_mod, Nat.zero_mul, Nat.zero_sub, Nat.min_zero]
    rfl
  | succ x ih =>
    rw [below_succ S q x s hq hs, ih, List.range_succ, List.filter_append, List.length_append]
    congr 1
    by_cases h : x / q % S = s
    · rw [ite_eq_left h, List.filter_cons, ite_eq_left (beq_iff_eq.2 h)]
      rfl
    · rw [ite_eq_right h, List.filter_cons, ite_eq_right (fun hb => h (beq_iff_eq.1 hb))]
      rfl

/-- One more unit adds one to a grid's term of `inSlot` exactly where the grid holds the unit in slot `s`. -/
private theorem term_succ (S : ℕ) (g : Grid) (lo hi s : ℕ) (hq : 0 < g.q) (hs : s < S) (hlo : lo ≤ hi) :
    below S g.q (min (hi + 1) (g.start + g.units) - g.start) s - below S g.q (max lo g.start - g.start) s =
      below S g.q (min hi (g.start + g.units) - g.start) s - below S g.q (max lo g.start - g.start) s +
        if (g.start ≤ hi ∧ hi < g.start + g.units) ∧ (hi - g.start) / g.q % S = s then 1 else 0 := by
  by_cases hin : g.start ≤ hi ∧ hi < g.start + g.units
  · have e1 : min (hi + 1) (g.start + g.units) - g.start = hi - g.start + 1 := by omega
    have e2 : min hi (g.start + g.units) - g.start = hi - g.start := by omega
    have hm := below_mono S g.q s hq hs (show max lo g.start - g.start ≤ hi - g.start by omega)
    rw [e1, e2, below_succ S g.q (hi - g.start) s hq hs]
    by_cases h : (hi - g.start) / g.q % S = s
    · rw [ite_eq_left h, ite_eq_left ⟨hin, h⟩]
      omega
    · rw [ite_eq_right h, ite_eq_right (fun hc => h hc.2)]
      omega
  · have e1 : min (hi + 1) (g.start + g.units) - g.start = min hi (g.start + g.units) - g.start := by omega
    rw [e1, ite_eq_right (fun hc => hin hc.1)]
    rfl

/-- A range with no unit has none in any slot. -/
theorem inSlot_empty (S : ℕ) (gs : List Grid) (lo hi s : ℕ) (hq : ∀ g ∈ gs, 0 < g.q) (hs : s < S)
    (h : hi ≤ lo) : inSlot S gs lo hi s = 0 := by
  unfold inSlot
  induction gs with
  | nil => rfl
  | cons g gs ih =>
    have hm := below_mono S g.q s (hq g (List.mem_cons_self ..)) hs
      (show min hi (g.start + g.units) - g.start ≤ max lo g.start - g.start by omega)
    rw [List.map_cons, List.sum_cons, ih fun g' hg' => hq g' (List.mem_cons_of_mem _ hg')]
    omega

/-- The grids that hold the unit `u` in slot `s`: one where its slot by the grids is `s`, else none. -/
private theorem holders (S : ℕ) (gs : List Grid) (u s : ℕ) (hg : OneGrid gs) :
    (gs.map fun g => if (g.start ≤ u ∧ u < g.start + g.units) ∧ (u - g.start) / g.q % S = s then 1 else 0).sum =
      if slotOf S gs u = some s then 1 else 0 := by
  induction gs with
  | nil => rfl
  | cons a gs ih =>
    have hp := List.pairwise_cons.1 hg
    rw [List.map_cons, List.sum_cons, ih hp.2, slotOf_cons]
    by_cases ha : a.start ≤ u ∧ u < a.start + a.units
    · have hnone : slotOf S gs u = none := slotOf_none S gs u fun g hmem hc => by
        have := hp.1 g hmem
        omega
      rw [hnone, ite_eq_left ha]
      by_cases h : (u - a.start) / a.q % S = s
      · rw [ite_eq_left ⟨ha, h⟩, ite_eq_left (congrArg some h)]
        rfl
      · rw [ite_eq_right (fun hc => h hc.2), ite_eq_right (fun hc => h (Option.some.inj hc))]
        rfl
    · rw [ite_eq_right ha, ite_eq_right (fun hc => ha hc.1), Nat.zero_add]

/-- One more unit adds one to `inSlot` exactly where the unit's slot by the grids is `s`. -/
theorem inSlot_succ (S : ℕ) (gs : List Grid) (lo hi s : ℕ) (hg : OneGrid gs) (hq : ∀ g ∈ gs, 0 < g.q)
    (hs : s < S) (hlo : lo ≤ hi) :
    inSlot S gs lo (hi + 1) s = inSlot S gs lo hi s + if slotOf S gs hi = some s then 1 else 0 := by
  rw [← holders S gs hi s hg]
  unfold inSlot
  clear hg
  induction gs with
  | nil => rfl
  | cons g gs ih =>
    rw [List.map_cons, List.sum_cons, List.map_cons, List.sum_cons, List.map_cons, List.sum_cons,
      ih fun g' hg' => hq g' (List.mem_cons_of_mem _ hg'),
      term_succ S g lo hi s (hq g (List.mem_cons_self ..)) hs hlo]
    omega

/-- `inSlot` counts the units of `[lo, hi)` whose slot by `gs` is `s`. -/
theorem inSlot_count (S : ℕ) (gs : List Grid) (lo hi s : ℕ) (hg : OneGrid gs) (hq : ∀ g ∈ gs, 0 < g.q)
    (hs : s < S) :
    inSlot S gs lo hi s = ((List.range' lo (hi - lo)).filter fun u => slotOf S gs u == some s).length := by
  have key : ∀ n : ℕ, inSlot S gs lo (lo + n) s =
      ((List.range' lo n).filter fun u => slotOf S gs u == some s).length := by
    intro n
    induction n with
    | zero => exact inSlot_empty S gs lo lo s hq hs (Nat.le_refl _)
    | succ n ih =>
      rw [← Nat.add_assoc, inSlot_succ S gs lo (lo + n) s hg hq hs (Nat.le_add_right _ _), ih,
        List.range'_concat, List.filter_append, List.length_append, Nat.one_mul]
      congr 1
      by_cases h : slotOf S gs (lo + n) = some s
      · rw [ite_eq_left h, List.filter_cons, ite_eq_left (beq_iff_eq.2 h)]
        rfl
      · rw [ite_eq_right h, List.filter_cons, ite_eq_right (fun hb => h (beq_iff_eq.1 hb))]
        rfl
  by_cases h : lo ≤ hi
  · have e := key (hi - lo)
    rw [show lo + (hi - lo) = hi by omega] at e
    exact e
  · rw [inSlot_empty S gs lo hi s hq hs (by omega), show hi - lo = 0 by omega]
    rfl

/-- A kind's place in a tally is below 7. -/
theorem kind_index_lt (k : Kind) : k.index < 7 := by
  cases k <;> decide

/-- The tally's fold from any seven counts: place `j` gains the sum over the declarations of the kind at `j`. -/
private theorem tally_fold (S : ℕ) (s j : ℕ) (ds : List Counted) : ∀ acc : Array ℕ, j < acc.size →
    (ds.foldl (fun acc (k, gs, d) =>
        acc.modify k.index (· + inSlot S gs d.w.start (d.w.start + d.w.readerUnits) s * d.w.cuts * d.bits))
      acc)[j]! =
      acc[j]! + ((ds.filter fun x => x.1.index == j).map fun x =>
        inSlot S x.2.1 x.2.2.w.start (x.2.2.w.start + x.2.2.w.readerUnits) s * x.2.2.w.cuts * x.2.2.bits).sum := by
  induction ds with
  | nil => intro acc _; rfl
  | cons x ds ih =>
    intro acc hj
    obtain ⟨k, gs, d⟩ := x
    have hj' : j < (acc.modify k.index
        (· + inSlot S gs d.w.start (d.w.start + d.w.readerUnits) s * d.w.cuts * d.bits)).size := by
      rw [Array.size_modify]; exact hj
    rw [List.foldl_cons, ih _ hj', getElem!_pos _ j hj', getElem!_pos acc j hj, Array.getElem_modify,
      List.filter_cons]
    by_cases h : k.index = j
    · rw [ite_eq_left h, ite_eq_left (beq_iff_eq.2 h), List.map_cons, List.sum_cons]
      exact Nat.add_assoc _ _ _
    · rw [ite_eq_right h, ite_eq_right (fun hb => h (beq_iff_eq.1 hb))]

/-- A tally's entry: over the declarations of that kind, each one's reader units in the slot, times the rows it
covers of each, times a row's bits. -/
theorem tally_eq (S : ℕ) (ds : List Counted) (s j : ℕ) (hj : j < 7) :
    (tally S ds s)[j]! =
      ((ds.filter fun x => x.1.index == j).map fun x =>
        inSlot S x.2.1 x.2.2.w.start (x.2.2.w.start + x.2.2.w.readerUnits) s * x.2.2.w.cuts * x.2.2.bits).sum := by
  unfold tally
  have hsize : j < (Array.replicate 7 (0 : ℕ)).size := by rw [Array.size_replicate]; exact hj
  rw [tally_fold S s j ds _ hsize, getElem!_pos _ j hsize, Array.getElem_replicate, Nat.zero_add]

end NciCheck.Record.SecurityProofs
