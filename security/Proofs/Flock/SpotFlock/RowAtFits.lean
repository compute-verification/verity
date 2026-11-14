module

public import Spot.SecondStage

/-!
# The bisection finds a unit's run

A row's runs (`Spot.SecondStage.Run`) say which row each unit of a cell reads: run `r` holds units `r.unit` to
`r.unit + r.count - 1`, unit `j` of them at row `r.row + (j - r.unit) * r.step`. The verifier checks that the runs hold
units `0` to `n - 1`, each once, in order (`fits`), and reads unit `j`'s row by bisection over the runs' first units
(`rowAt`).

`rowAt_of_fits`: where the runs fit `n` units, the bisection finds the run that holds unit `j < n` and gives its row
there. The loop keeps `lo < hi ≤ size`, run `lo` starting at or before `j` and every run from `hi` on starting after
`j` (`Inv`), and `size` passes bring `hi - lo` to one (`loop_spec`), since the runs' first units increase
(`mono_of_facts`).
-/

@[expose] public section

namespace SpotFlock.RowAtFits

open Spot.SecondStage (Run rowAt fits)

/-- The loop's step. -/
def stepF (rs : Array Run) (j : Nat) : Nat → Nat × Nat → Id (ForInStep (Nat × Nat)) := fun _ s =>
  if s.snd - s.fst ≤ 1 then pure (ForInStep.done (s.fst, s.snd))
  else if rs[(s.fst + s.snd) / 2]!.unit ≤ j then pure (ForInStep.yield ((s.fst + s.snd) / 2, s.snd))
  else pure (ForInStep.yield (s.fst, (s.fst + s.snd) / 2))

def Inv (rs : Array Run) (j : Nat) (s : Nat × Nat) : Prop :=
  s.fst < s.snd ∧ s.snd ≤ rs.size ∧ rs[s.fst]!.unit ≤ j ∧ ∀ i, s.snd ≤ i → i < rs.size → j < rs[i]!.unit

theorem loop_spec (rs : Array Run) (j : Nat)
    (mono : ∀ a b, a ≤ b → b < rs.size → rs[a]!.unit ≤ rs[b]!.unit) :
    ∀ (l : List Nat) (s : Nat × Nat), Inv rs j s → s.snd - s.fst ≤ l.length + 1 →
      Inv rs j (forIn l s (stepF rs j) : Id (Nat × Nat)).run ∧
        (forIn l s (stepF rs j) : Id (Nat × Nat)).run.snd - (forIn l s (stepF rs j) : Id (Nat × Nat)).run.fst ≤ 1 := by
  intro l
  induction l with
  | nil =>
    intro s hI hd
    simp only [List.forIn_nil, Id.run_pure]
    exact ⟨hI, by
      have hd' : s.snd - s.fst ≤ 0 + 1 := hd
      omega⟩
  | cons a l ih =>
    intro s hI hd
    simp only [List.forIn_cons, stepF]
    by_cases h1 : s.snd - s.fst ≤ 1
    · simp only [h1, ↓reduceIte, pure_bind, Id.run_pure]
      first | exact ⟨hI, trivial⟩ | exact ⟨hI, h1⟩
    · obtain ⟨hlt, hsz, hlo, hhi⟩ := hI
      have hlen : (a :: l).length = l.length + 1 := rfl
      rw [hlen] at hd
      by_cases h2 : rs[(s.fst + s.snd) / 2]!.unit ≤ j
      · simp only [h1, h2, ↓reduceIte, pure_bind]
        exact ih _ ⟨by show (s.fst + s.snd) / 2 < s.snd; omega, hsz, h2, hhi⟩
          (by show s.snd - (s.fst + s.snd) / 2 ≤ l.length + 1; omega)
      · simp only [h1, h2, ↓reduceIte, pure_bind]
        refine ih _ ⟨by show s.fst < (s.fst + s.snd) / 2; omega, by show (s.fst + s.snd) / 2 ≤ rs.size; omega, hlo,
          fun i hi hir => ?_⟩ (by show (s.fst + s.snd) / 2 - s.fst ≤ l.length + 1; omega)
        have hm := mono _ _ (show (s.fst + s.snd) / 2 ≤ i from hi) hir
        omega

/-- The fold's step (`fits`). -/
def fitF : Option Nat → Run → Option Nat := fun e r =>
  e.bind fun e => if r.unit == e && 0 < r.count then some (e + r.count) else none

theorem foldl_none (l : List Run) : l.foldl fitF none = none := by
  induction l with
  | nil => rfl
  | cons a l ih =>
    rw [List.foldl_cons]
    exact ih

theorem nil_get {α : Type} {i : Nat} {r : α} (h : ([] : List α)[i]? = some r) : False := by
  rw [List.getElem?_nil] at h
  cases h

theorem fits_list : ∀ (l : List Run) (e n : Nat), l.foldl fitF (some e) = some n →
    (∀ (i : Nat) (r : Run), l[i]? = some r → 0 < r.count) ∧ (∀ r : Run, l[0]? = some r → r.unit = e) ∧
      (∀ (i : Nat) (r r' : Run), l[i]? = some r → l[i + 1]? = some r' → r'.unit = r.unit + r.count) ∧
      (∀ r : Run, l[l.length - 1]? = some r → r.unit + r.count = n) ∧ (l = [] → n = e) := by
  intro l
  induction l with
  | nil =>
    intro e n h
    have hn : e = n := Option.some.inj h
    exact ⟨fun i r hr => (nil_get hr).elim, fun r hr => (nil_get hr).elim, fun i r r' hr => (nil_get hr).elim,
      fun r hr => (nil_get hr).elim, fun _ => hn.symm⟩
  | cons a l ih =>
    intro e n h
    rw [List.foldl_cons] at h
    by_cases hc : (a.unit == e && decide (0 < a.count)) = true
    · have hstep : fitF (some e) a = some (e + a.count) := by
        show (if (a.unit == e && decide (0 < a.count)) = true then some (e + a.count) else none) = _
        exact ite_eq_left_of_eq_true _ _ (eq_true hc)
      rw [hstep] at h
      obtain ⟨h1, h2, h3, h4, h5⟩ := ih _ _ h
      rw [Bool.and_eq_true] at hc
      have hu : a.unit = e := eq_of_beq hc.1
      have hcnt : 0 < a.count := of_decide_eq_true hc.2
      refine ⟨?_, ?_, ?_, ?_, ?_⟩
      · intro i r hr
        cases i with
        | zero =>
          rw [List.getElem?_cons_zero] at hr
          cases hr
          exact hcnt
        | succ i =>
          rw [List.getElem?_cons_succ] at hr
          exact h1 i r hr
      · intro r hr
        rw [List.getElem?_cons_zero] at hr
        cases hr
        exact hu
      · intro i r r' hr hr'
        cases i with
        | zero =>
          rw [List.getElem?_cons_zero] at hr
          cases hr
          rw [List.getElem?_cons_succ] at hr'
          have h0 := h2 r' hr'
          omega
        | succ i =>
          rw [List.getElem?_cons_succ] at hr hr'
          exact h3 i r r' hr hr'
      · intro r hr
        cases l with
        | nil =>
          have hr0 : [a][0]? = some r := hr
          rw [List.getElem?_cons_zero] at hr0
          cases hr0
          have h0 := h5 rfl
          omega
        | cons b l' =>
          have hr1 : (b :: l')[(b :: l').length - 1]? = some r := by
            have e : (a :: b :: l').length - 1 = (b :: l').length - 1 + 1 := by
              show l'.length + 1 + 1 - 1 = l'.length + 1 - 1 + 1
              omega
            rw [e, List.getElem?_cons_succ] at hr
            exact hr
          exact h4 r hr1
      · intro hnil
        cases hnil
    · have hstep : fitF (some e) a = none := by
        show (if (a.unit == e && decide (0 < a.count)) = true then some (e + a.count) else none) = _
        exact ite_eq_right_of_eq_false _ _ (eq_false hc)
      rw [hstep, foldl_none] at h
      cases h

theorem fits_facts {rs : Array Run} {n : Nat} (hf : fits rs n = true) :
    (∀ i, i < rs.size → 0 < rs[i]!.count) ∧ (0 < rs.size → rs[0]!.unit = 0) ∧
      (∀ i, i + 1 < rs.size → rs[i + 1]!.unit = rs[i]!.unit + rs[i]!.count) ∧
      (0 < rs.size → rs[rs.size - 1]!.unit + rs[rs.size - 1]!.count = n) ∧ (rs.size = 0 → n = 0) := by
  have h : rs.toList.foldl fitF (some 0) = some n := by
    rw [Array.foldl_toList]
    exact eq_of_beq hf
  obtain ⟨h1, h2, h3, h4, h5⟩ := fits_list _ _ _ h
  have get : ∀ i, i < rs.size → rs.toList[i]? = some rs[i]! := by
    intro i hi
    rw [Array.getElem?_toList, getElem!_pos rs i hi, Array.getElem?_eq_getElem hi]
  refine ⟨fun i hi => h1 i _ (get i hi), fun hp => h2 _ (get 0 hp), fun i hi => h3 i _ _ (get i (by omega)) (get _ hi),
    fun hp => h4 _ (by rw [Array.length_toList]; exact get _ (by omega)),
    fun h0 => h5 (List.eq_nil_of_length_eq_zero (by rw [Array.length_toList]; exact h0))⟩

theorem mono_of_facts {rs : Array Run}
    (h3 : ∀ i, i + 1 < rs.size → rs[i + 1]!.unit = rs[i]!.unit + rs[i]!.count) :
    ∀ a b, a ≤ b → b < rs.size → rs[a]!.unit ≤ rs[b]!.unit := by
  intro a b
  induction b with
  | zero =>
    intro hab _
    have h0 : a = 0 := by omega
    subst h0
    exact Nat.le_refl _
  | succ b ih =>
    intro hab hb
    by_cases hab' : a = b + 1
    · subst hab'
      exact Nat.le_refl _
    · have h1 := ih (by omega) (by omega)
      have h2 := h3 b hb
      omega

/-- **Where runs hold units `0` to `n - 1`, each once, in order (`fits`), the bisection finds unit `j < n`'s run.** -/
theorem rowAt_of_fits {rs : Array Run} {n j : Nat} (hf : fits rs n = true) (hj : j < n) :
    ∃ i, i < rs.size ∧ rs[i]!.unit ≤ j ∧ j < rs[i]!.unit + rs[i]!.count ∧
      rowAt rs j = some (rs[i]!.row + (j - rs[i]!.unit) * rs[i]!.step) := by
  obtain ⟨h1, h2, h3, h4, h5⟩ := fits_facts hf
  have hsz : 0 < rs.size := by
    rcases Nat.eq_zero_or_pos rs.size with h0 | hp
    · have := h5 h0
      omega
    · exact hp
  have hinit : Inv rs j (0, rs.size) :=
    ⟨hsz, Nat.le_refl _, by show rs[0]!.unit ≤ j; rw [h2 hsz]; exact Nat.zero_le _,
      fun i hi hir => absurd hir (Nat.not_lt.2 hi)⟩
  have hloop := loop_spec rs j (mono_of_facts h3) (List.range' 0 rs.size 1) (0, rs.size) hinit
    (by show rs.size - 0 ≤ (List.range' 0 rs.size 1).length + 1; rw [List.length_range']; omega)
  have hrow : rowAt rs j =
      match rs[(forIn (List.range' 0 rs.size 1) (0, rs.size) (stepF rs j) : Id (Nat × Nat)).run.fst]? with
      | some r => if r.unit ≤ j && j < r.unit + r.count then some (r.row + (j - r.unit) * r.step) else none
      | none => none := by
    unfold rowAt
    simp only [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size, Nat.sub_zero, Nat.add_sub_cancel,
      Nat.div_one]
    rfl
  generalize (forIn (List.range' 0 rs.size 1) (0, rs.size) (stepF rs j) : Id (Nat × Nat)).run = s at hloop hrow
  obtain ⟨⟨hlt, hle, hlo, hhi⟩, hd⟩ := hloop
  have hi : s.fst < rs.size := by omega
  have hcount : j < rs[s.fst]!.unit + rs[s.fst]!.count := by
    by_cases hlast : s.fst + 1 < rs.size
    · have e1 := h3 s.fst hlast
      have e2 := hhi (s.fst + 1) (by omega) hlast
      omega
    · have e1 : s.fst = rs.size - 1 := by omega
      have e2 := h4 hsz
      rw [← e1] at e2
      omega
  refine ⟨s.fst, hi, hlo, hcount, ?_⟩
  rw [hrow, Array.getElem?_eq_getElem hi, ← getElem!_pos rs s.fst hi]
  have hc : (decide (rs[s.fst]!.unit ≤ j) && decide (j < rs[s.fst]!.unit + rs[s.fst]!.count)) = true := by
    rw [Bool.and_eq_true]
    exact ⟨decide_eq_true hlo, decide_eq_true hcount⟩
  exact ite_eq_left_of_eq_true _ _ (eq_true hc)

end SpotFlock.RowAtFits
