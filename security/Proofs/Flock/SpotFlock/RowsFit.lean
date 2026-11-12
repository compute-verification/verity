module

public import Mathlib.Data.Nat.Notation
public import Spot.SecondStage

/-!
# The row a design's runs give a unit

Round 2 reads the row a template's unit reads at a linked or a wired port off the design's runs
(`Spot.SecondStage.rowAt`): the run that holds the unit, found by bisection over the runs' first units, in a loop.
Round 2's design check holds the runs to the template's units (`Spot.SecondStage.fits`). This file reads the loop as a
recursion and proves what the check gives of it.

* `bis`, `rowAt_eq`: the bisection as a recursion on its steps, and `rowAt` as the row of the run it ends at.
* `endAt`, `fits_spec`: runs that `fits` accepts for `n` units each start where the runs before them end, hold a
  unit at least, and end at `n` together.
* `rowAt_of_fits`: such runs give every unit under `n` a row, that of a run that holds it.
* `rowAt_lt_of_fits`: where each run's last row is under `per`, so is the row of every unit under `n`.
-/

@[expose] public section

namespace SpotFlock

open Spot.SecondStage (Run rowAt fits)

/-! ## The bisection as a recursion -/

/-- **One step of the bisection** for unit `j` over runs `rs`, at bounds `s`: the middle run's first unit at or
before `j` moves the lower bound there, else the upper. -/
def bisStep (rs : Array Run) (j : ℕ) (s : ℕ × ℕ) : ℕ × ℕ :=
  if rs[(s.1 + s.2) / 2]!.unit ≤ j then ((s.1 + s.2) / 2, s.2) else (s.1, (s.1 + s.2) / 2)

/-- **The bisection on `k` steps**: it stops where the bounds are one apart. -/
def bis (rs : Array Run) (j : ℕ) : ℕ → ℕ × ℕ → ℕ × ℕ
  | 0, s => s
  | k + 1, s => if s.2 - s.1 ≤ 1 then s else bis rs j k (bisStep rs j s)

/-- A loop whose body is the bisection's step, whatever it iterates over, computes `bis` on as many steps. -/
theorem forIn_bis (rs : Array Run) (j : ℕ) (F : ℕ → ℕ × ℕ → Id (ForInStep (ℕ × ℕ)))
    (hF : ∀ i s, F i s =
      if s.2 - s.1 ≤ 1 then pure (ForInStep.done s) else pure (ForInStep.yield (bisStep rs j s))) :
    ∀ (l : List ℕ) (s : ℕ × ℕ), forIn (m := Id) l s F = pure (bis rs j l.length s) := by
  intro l
  induction l with
  | nil =>
    intro s
    rw [List.forIn_nil]
    rfl
  | cons i is ih =>
    intro s
    rw [List.forIn_cons, hF]
    by_cases h : s.2 - s.1 ≤ 1
    · rw [ite_eq_left h]
      simp only [List.length_cons, bis, ite_eq_left h]
      rfl
    · rw [ite_eq_right h]
      simp only [List.length_cons, bis, ite_eq_right h]
      exact ih _

/-- **`rowAt` is the row of the run the bisection ends at**, where that run holds the unit. -/
theorem rowAt_eq (rs : Array Run) (j : ℕ) :
    rowAt rs j = match rs[(bis rs j rs.size (0, rs.size)).1]? with
      | some r => if r.unit ≤ j && j < r.unit + r.count then some (r.row + (j - r.unit) * r.step) else none
      | none => none := by
  unfold Spot.SecondStage.rowAt
  dsimp only
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  have hsize : ([:rs.size] : Std.Legacy.Range).size = rs.size := by
    show (rs.size - 0 + 1 - 1) / 1 = rs.size
    omega
  rw [hsize, forIn_bis rs j _ (fun i s => by
    by_cases h : s.2 - s.1 ≤ 1
    · simp only [ite_eq_left h]
    · simp only [ite_eq_right h, bisStep]
      split <;> rfl)]
  simp only [List.length_range']
  rfl

/-! ## What the placement check gives -/

/-- **Where the first `i` runs end**, each starting at the last's end: their units together. -/
def endAt (rs : Array Run) : ℕ → ℕ
  | 0 => 0
  | i + 1 => endAt rs i + rs[i]!.count

/-- **Runs the placement check accepts for `n` units**: each starts where the runs before it end and holds a unit at
least, and together they end at `n`. -/
theorem fits_spec {rs : Array Run} {n : ℕ} (h : fits rs n = true) :
    (∀ i, i < rs.size → rs[i]!.unit = endAt rs i ∧ 0 < rs[i]!.count) ∧ endAt rs rs.size = n := by
  unfold Spot.SecondStage.fits at h
  have key := Array.foldl_induction (as := rs)
    (motive := fun i (b : Option ℕ) =>
      b = none ∨ (b = some (endAt rs i) ∧ ∀ k, k < i → rs[k]!.unit = endAt rs k ∧ 0 < rs[k]!.count))
    (init := some 0)
    (f := fun (e : Option ℕ) r => e.bind fun e => if r.unit == e && 0 < r.count then some (e + r.count) else none)
    (Or.inr ⟨rfl, fun k hk => absurd hk (Nat.not_lt_zero k)⟩)
    (fun i b hb => by
      rcases hb with rfl | ⟨rfl, hk⟩
      · exact Or.inl rfl
      · have hi : rs[i.1]! = rs[i] := getElem!_pos rs i.1 i.2
        by_cases hc : (rs[i].unit == endAt rs i.1 && decide (0 < rs[i].count)) = true
        · refine Or.inr ⟨?_, fun k hk' => ?_⟩
          · simp only [Option.bind_some, ite_eq_left hc, endAt, hi]
          · rw [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at hc
            rcases Nat.lt_succ_iff_lt_or_eq.1 hk' with hlt | rfl
            · exact hk k hlt
            · rw [hi]
              exact hc
        · refine Or.inl ?_
          simp only [Option.bind_some, ite_eq_right hc])
  rcases key with hnone | ⟨hsome, hk⟩
  · rw [hnone] at h
    cases h
  · rw [hsome] at h
    exact ⟨hk, by simpa using h⟩

/-- **The bisection ends at the run that holds the unit**, where each run starts at the last's end: from bounds that
hold unit `j` between their runs' ends, on steps enough for their distance. -/
theorem bis_inv {rs : Array Run} {j : ℕ} (hu : ∀ i, i < rs.size → rs[i]!.unit = endAt rs i) :
    ∀ (k : ℕ) (s : ℕ × ℕ), s.1 < s.2 → s.2 ≤ rs.size → endAt rs s.1 ≤ j → j < endAt rs s.2 → s.2 - s.1 ≤ k + 1 →
      (bis rs j k s).1 < rs.size ∧ endAt rs (bis rs j k s).1 ≤ j ∧ j < endAt rs ((bis rs j k s).1 + 1) := by
  intro k
  induction k with
  | zero =>
    intro s h1 h2 h3 h4 h5
    have e : s.2 = s.1 + 1 := by omega
    rw [e] at h4
    exact ⟨by show s.1 < rs.size; omega, h3, h4⟩
  | succ k ih =>
    intro s h1 h2 h3 h4 h5
    by_cases hd : s.2 - s.1 ≤ 1
    · have e : s.2 = s.1 + 1 := by omega
      rw [e] at h4
      have hb : bis rs j (k + 1) s = s := by simp only [bis, ite_eq_left hd]
      rw [hb]
      exact ⟨by omega, h3, h4⟩
    · have hb : bis rs j (k + 1) s = bis rs j k (bisStep rs j s) := by simp only [bis, ite_eq_right hd]
      rw [hb]
      have hm := hu ((s.1 + s.2) / 2) (by omega)
      unfold bisStep
      by_cases hc : rs[(s.1 + s.2) / 2]!.unit ≤ j
      · rw [ite_eq_left hc]
        exact ih _ (by show (s.1 + s.2) / 2 < s.2; omega) h2 (by rw [← hm]; exact hc) h4
          (by show s.2 - (s.1 + s.2) / 2 ≤ k + 1; omega)
      · rw [ite_eq_right hc]
        exact ih _ (by show s.1 < (s.1 + s.2) / 2; omega) (by show (s.1 + s.2) / 2 ≤ rs.size; omega) h3
          (by rw [← hm]; omega) (by show (s.1 + s.2) / 2 - s.1 ≤ k + 1; omega)

/-- **Runs that hold units `0` to `n - 1` give each of them a row**: unit `j < n`'s is the row of a run that holds
it, `row + (j - unit)·step`. -/
theorem rowAt_of_fits {rs : Array Run} {n j : ℕ} (h : fits rs n = true) (hj : j < n) :
    ∃ r, r ∈ rs ∧ r.unit ≤ j ∧ j < r.unit + r.count ∧ rowAt rs j = some (r.row + (j - r.unit) * r.step) := by
  obtain ⟨hk, hn⟩ := fits_spec h
  have hpos : 0 < rs.size := by
    rcases Nat.eq_zero_or_pos rs.size with h0 | h0
    · rw [h0] at hn
      have h00 : endAt rs 0 = 0 := rfl
      omega
    · exact h0
  obtain ⟨hlt, h1, h2⟩ := bis_inv (j := j) (fun i hi => (hk i hi).1) rs.size (0, rs.size) hpos (Nat.le_refl _)
    (Nat.zero_le _) (by rw [hn]; exact hj) (by show rs.size - 0 ≤ rs.size + 1; omega)
  have hu := (hk _ hlt).1
  rw [getElem!_pos rs _ hlt] at hu
  have he : endAt rs ((bis rs j rs.size (0, rs.size)).1 + 1) =
      endAt rs (bis rs j rs.size (0, rs.size)).1 + rs[(bis rs j rs.size (0, rs.size)).1]!.count := rfl
  rw [he, getElem!_pos rs _ hlt] at h2
  refine ⟨rs[(bis rs j rs.size (0, rs.size)).1], Array.getElem_mem hlt, ?_, ?_, ?_⟩
  · rw [hu]
    exact h1
  · rw [hu]
    exact h2
  · rw [rowAt_eq, Array.getElem?_eq_getElem hlt]
    have hc : (decide (rs[(bis rs j rs.size (0, rs.size)).1].unit ≤ j) &&
        decide (j < rs[(bis rs j rs.size (0, rs.size)).1].unit +
          rs[(bis rs j rs.size (0, rs.size)).1].count)) = true := by
      rw [Bool.and_eq_true, decide_eq_true_eq, decide_eq_true_eq, hu]
      exact ⟨h1, h2⟩
    simp only [ite_eq_left hc]

/-- **Where each run's last row is under `per`, so is every unit's**: runs that hold units `0` to `n - 1`, each with
its last unit's row `row + (count - 1)·step` under `per`, give each unit under `n` a row under `per`. -/
theorem rowAt_lt_of_fits {rs : Array Run} {n j per : ℕ} (h : fits rs n = true) (hj : j < n)
    (hu : ∀ r ∈ rs, r.row + (r.count - 1) * r.step < per) : ∃ x, rowAt rs j = some x ∧ x < per := by
  obtain ⟨r, hr, h1, h2, hx⟩ := rowAt_of_fits h hj
  refine ⟨_, hx, ?_⟩
  have hle : (j - r.unit) * r.step ≤ (r.count - 1) * r.step := Nat.mul_le_mul_right _ (by omega)
  have := hu r hr
  omega

end SpotFlock
