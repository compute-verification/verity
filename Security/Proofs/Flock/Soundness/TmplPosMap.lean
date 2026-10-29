import Flock.TmplLayout

/-!
# `posMap` puts every copy at its place

`posMap` is `fillMap`'s writes over the call tree (`fillMap_writes`). When every Definition's layout is in order
(`DefPos`: runs and calls ascending and apart, each column in one, copies side by side, callees earlier), every column
`y` of copy `(d, o)` is written exactly once (`writes_ok`), and with `A + B · posIn y` (`writes_mem`). So every copy sits
at its place (`posMap_tree`, `TreeAt`), which is what `fold_eq_foldRef` reads.

- `firstGt_spec`, `locate_call`, `locate_own`: `locate`'s halving lands on the call or run holding the column.
- `posIn_call`, `posIn_own`, `posIn_stable`: what `posIn` is in each, at any fuel past the depth.
-/

open Flock Flock.Tmpl

namespace FlockSoundness.TmplFold

/-- Copy `(d, o)` of the unrolled unit sits at `(A, B)` under `π`: its own column `y` at `A + B · posIn y`, and every copy
of its calls at its place. -/
def TreeAt (t : Tmpl) (lays : Array Lay) (π : Array Nat) : Nat → Nat → Nat → Nat → Nat → Prop
  | 0, _, _, _, _ => True
  | fuel + 1, d, o, A, B =>
    (∀ y < (t.defs.getD d default).cols,
      π.getD (o + y) 0 = A + B * Flock.Tmpl.posIn t lays (t.defs.size + 1) d y) ∧
    ∀ i < (t.defs.getD d default).calls.size, ∀ j < ((t.defs.getD d default).calls.getD i default).count,
      TreeAt t lays π fuel ((t.defs.getD d default).calls.getD i default).def_
        (o + ((t.defs.getD d default).calls.getD i default).base + j * ((t.defs.getD d default).calls.getD i default).stride)
        (A + B * (((lays.getD d default).places.getD i default).nb + j * ((lays.getD d default).places.getD i default).cs))
        (B * ((lays.getD d default).places.getD i default).ms)

/-- Every write `fillMap` makes, in order: each run's columns, then each call's copies'. -/
def writes (t : Tmpl) (lays : Array Lay) : Nat → Nat → Nat → Nat → Nat → List (Nat × Nat)
  | 0, _, _, _, _ => []
  | fuel + 1, d, o, A, B =>
    ((lays.getD d default).runs.toList.flatMap fun x =>
      (List.range (x.2.1 - x.1)).map fun i => (o + x.1 + i, A + B * (x.2.2 + i))) ++
    (List.range (t.defs.getD d default).calls.size).flatMap fun i =>
      (List.range ((t.defs.getD d default).calls.getD i default).count).flatMap fun j =>
        writes t lays fuel ((t.defs.getD d default).calls.getD i default).def_
          (o + ((t.defs.getD d default).calls.getD i default).base + j * ((t.defs.getD d default).calls.getD i default).stride)
          (A + B * (((lays.getD d default).places.getD i default).nb + j * ((lays.getD d default).places.getD i default).cs))
          (B * ((lays.getD d default).places.getD i default).ms)

/-- `ws`'s writes into `m`, in order. -/
def setAll (ws : List (Nat × Nat)) (m : Array Nat) : Array Nat := ws.foldl (fun m w => m.set! w.1 w.2) m

theorem fillMap_writes (t : Tmpl) (lays : Array Lay) :
    ∀ fuel d o A B m, fillMap t lays fuel d o A B m = setAll (writes t lays fuel d o A B) m
  | 0, _, _, _, _, _ => rfl
  | fuel + 1, d, o, A, B, m => by
    unfold fillMap writes setAll
    dsimp only
    rw [List.foldl_append, List.foldl_flatMap, List.foldl_flatMap, ← Array.foldl_toList]
    congr 1
    · funext acc i
      rw [List.foldl_flatMap]
      congr 1
      funext acc' j
      exact fillMap_writes t lays fuel _ _ _ _ acc'
    · congr 1
      funext acc x
      obtain ⟨a, b, p⟩ := x
      rw [List.foldl_map]

theorem setAll_size : ∀ (ws : List (Nat × Nat)) (m : Array Nat), (setAll ws m).size = m.size
  | [], _ => rfl
  | w :: ws, m => by
    unfold setAll
    rw [List.foldl_cons]
    exact (setAll_size ws _).trans (by simp)

theorem setAll_frame : ∀ (ws : List (Nat × Nat)) (m : Array Nat) (z : Nat), (∀ w ∈ ws, w.1 ≠ z) →
    (setAll ws m).getD z 0 = m.getD z 0
  | [], _, _, _ => rfl
  | w :: ws, m, z, h => by
    unfold setAll
    rw [List.foldl_cons]
    refine (setAll_frame ws _ z fun w' hw' => h w' (List.mem_cons_of_mem _ hw')).trans ?_
    have hz : w.1 ≠ z := h w (List.mem_cons_self ..)
    simp [Array.getD_eq_getD_getElem?, hz]

theorem setAll_hit : ∀ (ws : List (Nat × Nat)) (m : Array Nat) (z v : Nat), (ws.map Prod.fst).Nodup → (z, v) ∈ ws →
    z < m.size → (setAll ws m).getD z 0 = v
  | [], _, _, _, _, h, _ => absurd h List.not_mem_nil
  | w :: ws, m, z, v, hn, h, hz => by
    rw [List.map_cons, List.nodup_cons] at hn
    unfold setAll
    rw [List.foldl_cons]
    rcases List.mem_cons.mp h with rfl | h
    · refine (setAll_frame ws _ z fun w' hw' he => hn.1 (show z ∈ ws.map Prod.fst by rw [← he]; exact List.mem_map_of_mem hw')).trans ?_
      simp [Array.getD_eq_getD_getElem?, hz]
    · exact setAll_hit ws _ z v hn.2 h (by simpa using hz)

/-- `firstGt` halves to the first `i ∈ [lo, hi)` with `key i > x`, for `key` ascending there and `hi − lo < 2^fuel`. -/
theorem firstGt_spec (key : Nat → Nat) (x H : Nat) (hmono : ∀ i j, i ≤ j → j < H → key i ≤ key j) :
    ∀ fuel lo hi, lo ≤ hi → hi ≤ H → hi - lo < 2 ^ fuel →
      lo ≤ firstGt key x fuel lo hi ∧ firstGt key x fuel lo hi ≤ hi ∧
        (∀ i, lo ≤ i → i < firstGt key x fuel lo hi → key i ≤ x) ∧
        (∀ i, firstGt key x fuel lo hi ≤ i → i < hi → x < key i)
  | 0, lo, hi, hlh, hH, hlt => by
    have : hi = lo := by simp at hlt; omega
    subst this
    simp only [firstGt]
    exact ⟨Nat.le_refl _, Nat.le_refl _, fun i h1 h2 => absurd h2 (by omega), fun i h1 h2 => absurd h2 (by omega)⟩
  | fuel + 1, lo, hi, hlh, hH, hlt => by
    simp only [firstGt]
    by_cases h0 : hi ≤ lo
    · rw [if_pos h0]
      exact ⟨Nat.le_refl _, hlh, fun i h1 h2 => absurd h2 (by omega), fun i h1 h2 => absurd h2 (by omega)⟩
    rw [if_neg h0]
    have hp : 2 ^ (fuel + 1) = 2 * 2 ^ fuel := by rw [Nat.pow_succ, Nat.mul_comm]
    by_cases hk : key ((lo + hi) / 2) ≤ x
    · rw [if_pos hk]
      obtain ⟨h1, h2, h3, h4⟩ := firstGt_spec key x H hmono fuel ((lo + hi) / 2 + 1) hi (by omega) hH (by omega)
      refine ⟨by omega, h2, fun i hi1 hi2 => ?_, h4⟩
      by_cases hm : i ≤ (lo + hi) / 2
      · exact Nat.le_trans (hmono _ _ hm (by omega)) hk
      · exact h3 i (by omega) hi2
    · rw [if_neg hk]
      obtain ⟨h1, h2, h3, h4⟩ := firstGt_spec key x H hmono fuel lo ((lo + hi) / 2) (by omega) (by omega) (by omega)
      refine ⟨h1, by omega, h3, fun i hi1 hi2 => ?_⟩
      by_cases hm : i < (lo + hi) / 2
      · exact h4 i hi1 hm
      · exact Nat.lt_of_lt_of_le (Nat.lt_of_not_le hk) (hmono _ _ (by omega) (by omega))

/-- What `posIn` and `fillMap` read of Definition `d`'s layout: its runs ascending and apart, its calls ascending and
apart with their copies side by side, every column in one of them, and its callees earlier Definitions. -/
structure DefPos (t : Tmpl) (lays : Array Lay) (d : Nat) : Prop where
  runs_lt : ∀ r < (lays.getD d default).runs.size,
    ((lays.getD d default).runs.getD r default).1 < ((lays.getD d default).runs.getD r default).2.1
  runs_asc : ∀ r r', r < r' → r' < (lays.getD d default).runs.size →
    ((lays.getD d default).runs.getD r default).2.1 ≤ ((lays.getD d default).runs.getD r' default).1
  runs_in : ∀ r < (lays.getD d default).runs.size, ((lays.getD d default).runs.getD r default).2.1 ≤ (t.defs.getD d default).cols
  calls_asc : ∀ i i', i < i' → i' < (t.defs.getD d default).calls.size →
    ((t.defs.getD d default).calls.getD i default).base + ((t.defs.getD d default).calls.getD i default).count *
      (t.defs.getD ((t.defs.getD d default).calls.getD i default).def_ default).cols ≤ ((t.defs.getD d default).calls.getD i' default).base
  calls_in : ∀ i < (t.defs.getD d default).calls.size,
    ((t.defs.getD d default).calls.getD i default).base + ((t.defs.getD d default).calls.getD i default).count *
      (t.defs.getD ((t.defs.getD d default).calls.getD i default).def_ default).cols ≤ (t.defs.getD d default).cols
  stride : ∀ i < (t.defs.getD d default).calls.size, 1 < ((t.defs.getD d default).calls.getD i default).count →
    ((t.defs.getD d default).calls.getD i default).stride = (t.defs.getD ((t.defs.getD d default).calls.getD i default).def_ default).cols
  cover : ∀ y < (t.defs.getD d default).cols,
    (∃ r < (lays.getD d default).runs.size, ((lays.getD d default).runs.getD r default).1 ≤ y ∧ y < ((lays.getD d default).runs.getD r default).2.1) ∨
    (∃ i < (t.defs.getD d default).calls.size, ((t.defs.getD d default).calls.getD i default).base ≤ y ∧
      y < ((t.defs.getD d default).calls.getD i default).base + ((t.defs.getD d default).calls.getD i default).count *
        (t.defs.getD ((t.defs.getD d default).calls.getD i default).def_ default).cols)
  apart : ∀ r < (lays.getD d default).runs.size, ∀ i < (t.defs.getD d default).calls.size,
    ((lays.getD d default).runs.getD r default).2.1 ≤ ((t.defs.getD d default).calls.getD i default).base ∨
      ((t.defs.getD d default).calls.getD i default).base + ((t.defs.getD d default).calls.getD i default).count *
        (t.defs.getD ((t.defs.getD d default).calls.getD i default).def_ default).cols ≤ ((lays.getD d default).runs.getD r default).1
  runs_small : (lays.getD d default).runs.size < 2 ^ 64
  calls_small : (t.defs.getD d default).calls.size < 2 ^ 64
  callee : ∀ i < (t.defs.getD d default).calls.size, ((t.defs.getD d default).calls.getD i default).def_ < d

theorem calls_mono (t : Tmpl) (lays : Array Lay) {d : Nat} (hp : DefPos t lays d) :
    ∀ i j, i ≤ j → j < (t.defs.getD d default).calls.size →
      ((t.defs.getD d default).calls.getD i default).base ≤ ((t.defs.getD d default).calls.getD j default).base := by
  intro i j hij hj
  rcases Nat.lt_or_eq_of_le hij with h | rfl
  · exact Nat.le_trans (Nat.le_add_right _ _) (hp.calls_asc i j h hj)
  · exact Nat.le_refl _

theorem firstGt_le (key : Nat → Nat) (x : Nat) : ∀ fuel lo hi, lo ≤ hi → firstGt key x fuel lo hi ≤ hi
  | 0, _, _, h => h
  | fuel + 1, lo, hi, h => by
    simp only [firstGt]
    split
    · exact h
    split
    · exact firstGt_le key x fuel _ _ (by omega)
    · exact Nat.le_trans (firstGt_le key x fuel _ _ (by omega)) (by omega)

theorem locate_call (t : Tmpl) (lays : Array Lay) {d : Nat} (hp : DefPos t lays d) {y i : Nat}
    (hi : i < (t.defs.getD d default).calls.size) (h1 : ((t.defs.getD d default).calls.getD i default).base ≤ y)
    (h2 : y < ((t.defs.getD d default).calls.getD i default).base + ((t.defs.getD d default).calls.getD i default).count *
        (t.defs.getD ((t.defs.getD d default).calls.getD i default).def_ default).cols) :
    locate t lays d y = .call i ((y - ((t.defs.getD d default).calls.getD i default).base) /
        (t.defs.getD ((t.defs.getD d default).calls.getD i default).def_ default).cols)
      ((y - ((t.defs.getD d default).calls.getD i default).base) % (t.defs.getD ((t.defs.getD d default).calls.getD i default).def_ default).cols) := by
  obtain ⟨-, hI2, hI3, hI4⟩ := firstGt_spec (fun i => ((t.defs.getD d default).calls.getD i default).base) y _ (calls_mono t lays hp) 64 0 _
    (Nat.zero_le _) (Nat.le_refl _) (by have := hp.calls_small; omega)
  unfold locate
  dsimp only
  generalize firstGt (fun i => ((t.defs.getD d default).calls.getD i default).base) y 64 0 (t.defs.getD d default).calls.size = I at hI2 hI3 hI4 ⊢
  have hlt : i < I := Nat.lt_of_not_le fun h => Nat.lt_irrefl _ (Nat.lt_of_lt_of_le (hI4 i h hi) h1)
  have hle : I - 1 ≤ i := Nat.le_of_not_lt fun h => by
    have := hp.calls_asc i (I - 1) h (by omega)
    have := hI3 (I - 1) (Nat.zero_le _) (by omega)
    omega
  have hIi : I - 1 = i := by omega
  rw [hIi, if_pos (by simp only [Bool.and_eq_true, decide_eq_true_eq]; exact ⟨by omega, h2⟩)]


theorem runs_mono (t : Tmpl) (lays : Array Lay) {d : Nat} (hp : DefPos t lays d) :
    ∀ i j, i ≤ j → j < (lays.getD d default).runs.size →
      ((lays.getD d default).runs.getD i default).1 ≤ ((lays.getD d default).runs.getD j default).1 := by
  intro i j hij hj
  rcases Nat.lt_or_eq_of_le hij with h | rfl
  · exact Nat.le_of_lt (Nat.lt_of_lt_of_le (hp.runs_lt i (by omega)) (hp.runs_asc i j h hj))
  · exact Nat.le_refl _

theorem locate_own (t : Tmpl) (lays : Array Lay) {d : Nat} (hp : DefPos t lays d) {y r a b p : Nat}
    (hr : r < (lays.getD d default).runs.size) (hx : (lays.getD d default).runs.getD r default = (a, b, p))
    (h1 : a ≤ y) (h2 : y < b) : locate t lays d y = .own (p + (y - a)) := by
  have hra : ((lays.getD d default).runs.getD r default).1 = a := by rw [hx]
  have hrb : ((lays.getD d default).runs.getD r default).2.1 = b := by rw [hx]
  obtain ⟨-, hI2, hI3, -⟩ := firstGt_spec (fun i => ((t.defs.getD d default).calls.getD i default).base) y _
    (calls_mono t lays hp) 64 0 _ (Nat.zero_le _) (Nat.le_refl _) (by have := hp.calls_small; omega)
  obtain ⟨-, hR2, hR3, hR4⟩ := firstGt_spec (fun r => ((lays.getD d default).runs.getD r default).1) y _
    (runs_mono t lays hp) 64 0 _ (Nat.zero_le _) (Nat.le_refl _) (by have := hp.runs_small; omega)
  unfold locate
  simp only [Bool.and_eq_true, decide_eq_true_eq]
  generalize firstGt (fun i => ((t.defs.getD d default).calls.getD i default).base) y 64 0 (t.defs.getD d default).calls.size = I at hI2 hI3 ⊢
  generalize firstGt (fun r => ((lays.getD d default).runs.getD r default).1) y 64 0 (lays.getD d default).runs.size = R at hR2 hR3 hR4 ⊢
  have hnc : ¬(0 < I ∧ y < ((t.defs.getD d default).calls.getD (I - 1) default).base +
      ((t.defs.getD d default).calls.getD (I - 1) default).count *
        (t.defs.getD ((t.defs.getD d default).calls.getD (I - 1) default).def_ default).cols) := by
    intro h
    have := hI3 (I - 1) (Nat.zero_le _) (by omega)
    have := hp.apart r hr (I - 1) (by omega)
    omega
  have hlt : r < R := Nat.lt_of_not_le fun h => Nat.lt_irrefl _ (Nat.lt_of_lt_of_le (hR4 r h hr) (hra ▸ h1))
  have hle : R - 1 ≤ r := Nat.le_of_not_lt fun h => by
    have := hp.runs_asc r (R - 1) h (by omega)
    have := hR3 (R - 1) (Nat.zero_le _) (by omega)
    omega
  have hRr : R - 1 = r := by omega
  simp only [hRr, hx]
  rw [if_neg hnc, if_pos ⟨by omega, h2⟩]

theorem locate_call_lt (t : Tmpl) (lays : Array Lay) {d y i j x : Nat} (h : locate t lays d y = .call i j x) :
    i < (t.defs.getD d default).calls.size := by
  have hI := firstGt_le (fun i => ((t.defs.getD d default).calls.getD i default).base) y 64 0
    (t.defs.getD d default).calls.size (Nat.zero_le _)
  unfold locate at h
  simp only [Bool.and_eq_true, decide_eq_true_eq] at h
  generalize firstGt (fun i => ((t.defs.getD d default).calls.getD i default).base) y 64 0 (t.defs.getD d default).calls.size = I at hI h
  split at h
  · rename_i hc
    cases h
    omega
  · split at h <;> (try split at h) <;> simp at h

theorem posIn_call (t : Tmpl) (lays : Array Lay) {d : Nat} (hp : DefPos t lays d) {y i : Nat}
    (hi : i < (t.defs.getD d default).calls.size) (h1 : ((t.defs.getD d default).calls.getD i default).base ≤ y)
    (h2 : y < ((t.defs.getD d default).calls.getD i default).base + ((t.defs.getD d default).calls.getD i default).count *
        (t.defs.getD ((t.defs.getD d default).calls.getD i default).def_ default).cols) (f : Nat) :
    posIn t lays (f + 1) d y = ((lays.getD d default).places.getD i default).nb +
      (y - ((t.defs.getD d default).calls.getD i default).base) /
        (t.defs.getD ((t.defs.getD d default).calls.getD i default).def_ default).cols *
        ((lays.getD d default).places.getD i default).cs +
      ((lays.getD d default).places.getD i default).ms * posIn t lays f ((t.defs.getD d default).calls.getD i default).def_
        ((y - ((t.defs.getD d default).calls.getD i default).base) %
          (t.defs.getD ((t.defs.getD d default).calls.getD i default).def_ default).cols) := by
  simp only [posIn, locate_call t lays hp hi h1 h2]

theorem posIn_own (t : Tmpl) (lays : Array Lay) {d : Nat} (hp : DefPos t lays d) {y r a b p : Nat}
    (hr : r < (lays.getD d default).runs.size) (hx : (lays.getD d default).runs.getD r default = (a, b, p))
    (h1 : a ≤ y) (h2 : y < b) (f : Nat) : posIn t lays (f + 1) d y = p + (y - a) := by
  simp only [posIn, locate_own t lays hp hr hx h1 h2]

/-- `posIn` reads the same with any fuel past the depth. -/
theorem posIn_stable (t : Tmpl) (lays : Array Lay) (hpos : ∀ d < t.defs.size, DefPos t lays d) :
    ∀ f1 f2 d y, d < t.defs.size → d < f1 → d < f2 → posIn t lays f1 d y = posIn t lays f2 d y
  | 0, _, _, _, _, h, _ => absurd h (Nat.not_lt_zero _)
  | _, 0, _, _, _, _, h => absurd h (Nat.not_lt_zero _)
  | f1 + 1, f2 + 1, d, y, hd, h1, h2 => by
    simp only [posIn]
    cases hl : locate t lays d y with
    | call i j x =>
      have hi := locate_call_lt t lays hl
      have hc := (hpos d hd).callee i hi
      dsimp only
      rw [posIn_stable t lays hpos f1 f2 _ x (by omega) (by omega) (by omega)]
    | own p => rfl
    | none => rfl

theorem toList_getD {α : Type} [Inhabited α] (a : Array α) :
    a.toList = (List.range a.size).map fun r => a.getD r default := by
  apply List.ext_getElem (by simp)
  intro n h1 h2
  rw [Array.length_toList] at h1
  simp [Array.getD, h1]

theorem nodup_flatMap_range (g : Nat → List Nat) (lo hi : Nat → Nat) : ∀ n, (∀ i < n, (g i).Nodup) →
    (∀ i < n, ∀ x ∈ g i, lo i ≤ x ∧ x < hi i) → (∀ i j, i < j → j < n → hi i ≤ lo j) → ((List.range n).flatMap g).Nodup
  | 0, _, _, _ => by simp
  | n + 1, hnd, hb, hs => by
    rw [List.range_succ, List.flatMap_append, List.nodup_append]
    refine ⟨nodup_flatMap_range g lo hi n (fun i h => hnd i (by omega)) (fun i h => hb i (by omega))
      (fun i j h h' => hs i j h (by omega)), by simpa using hnd n (by omega), ?_⟩
    intro a ha b hb'
    obtain ⟨i, hi, ha⟩ := List.mem_flatMap.mp ha
    rw [List.mem_range] at hi
    simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil] at hb'
    have := hb i (by omega) a ha
    have := hb n (by omega) b hb'
    have := hs i n hi (by omega)
    omega

theorem writes_fst (t : Tmpl) (lays : Array Lay) (f d o A B : Nat) :
    (writes t lays (f + 1) d o A B).map Prod.fst =
      ((List.range (lays.getD d default).runs.size).flatMap fun r =>
        (List.range (((lays.getD d default).runs.getD r default).2.1 - ((lays.getD d default).runs.getD r default).1)).map
          fun i => o + ((lays.getD d default).runs.getD r default).1 + i) ++
      (List.range (t.defs.getD d default).calls.size).flatMap fun i =>
        (List.range ((t.defs.getD d default).calls.getD i default).count).flatMap fun j =>
          (writes t lays f ((t.defs.getD d default).calls.getD i default).def_
            (o + ((t.defs.getD d default).calls.getD i default).base + j * ((t.defs.getD d default).calls.getD i default).stride)
            (A + B * (((lays.getD d default).places.getD i default).nb + j * ((lays.getD d default).places.getD i default).cs))
            (B * ((lays.getD d default).places.getD i default).ms)).map Prod.fst := by
  rw [writes, List.map_append, List.map_flatMap, List.map_flatMap, toList_getD, List.flatMap_map]
  simp only [List.map_map, List.map_flatMap]
  rfl

theorem copy_le {base count stride Le j : Nat} (hs : 1 < count → stride = Le) (hj : j < count) :
    base + j * stride + Le ≤ base + count * Le := by
  by_cases h : 1 < count
  · rw [hs h]
    have : (j + 1) * Le ≤ count * Le := Nat.mul_le_mul_right _ hj
    rw [Nat.succ_mul] at this
    omega
  · have h0 : j = 0 := by omega
    have h1 : count = 1 := by omega
    subst h0 h1
    simp

/-- `writes` puts copy `(d, o)` in `[o, o + cols)`, each column once. -/
theorem writes_ok (t : Tmpl) (lays : Array Lay) (hpos : ∀ d < t.defs.size, DefPos t lays d) :
    ∀ f d o A B, d < t.defs.size →
      (∀ x ∈ (writes t lays f d o A B).map Prod.fst, o ≤ x ∧ x < o + (t.defs.getD d default).cols) ∧
        ((writes t lays f d o A B).map Prod.fst).Nodup
  | 0, _, _, _, _, _ => by simp [writes]
  | f + 1, d, o, A, B, hd => by
    have hp := hpos d hd
    have IH : ∀ i < (t.defs.getD d default).calls.size, ∀ j,
        (∀ x ∈ (writes t lays f ((t.defs.getD d default).calls.getD i default).def_ (o + ((t.defs.getD d default).calls.getD i default).base + j * ((t.defs.getD d default).calls.getD i default).stride) (A + B * (((lays.getD d default).places.getD i default).nb + j * ((lays.getD d default).places.getD i default).cs)) (B * ((lays.getD d default).places.getD i default).ms)).map Prod.fst, o + ((t.defs.getD d default).calls.getD i default).base + j * ((t.defs.getD d default).calls.getD i default).stride ≤ x ∧ x < o + ((t.defs.getD d default).calls.getD i default).base + j * ((t.defs.getD d default).calls.getD i default).stride + (t.defs.getD ((t.defs.getD d default).calls.getD i default).def_ default).cols) ∧
          ((writes t lays f ((t.defs.getD d default).calls.getD i default).def_ (o + ((t.defs.getD d default).calls.getD i default).base + j * ((t.defs.getD d default).calls.getD i default).stride) (A + B * (((lays.getD d default).places.getD i default).nb + j * ((lays.getD d default).places.getD i default).cs)) (B * ((lays.getD d default).places.getD i default).ms)).map Prod.fst).Nodup := fun i hi j =>
      writes_ok t lays hpos f _ _ _ _ (by have := hp.callee i hi; omega)
    have hR : ∀ x ∈ ((List.range (lays.getD d default).runs.size).flatMap fun r => (List.range (((lays.getD d default).runs.getD r default).2.1 - ((lays.getD d default).runs.getD r default).1)).map fun i => o + ((lays.getD d default).runs.getD r default).1 + i), ∃ r < (lays.getD d default).runs.size, o + ((lays.getD d default).runs.getD r default).1 ≤ x ∧ x < o + ((lays.getD d default).runs.getD r default).2.1 := by
      intro x hx
      obtain ⟨r, hr, hx⟩ := List.mem_flatMap.mp hx
      rw [List.mem_range] at hr
      obtain ⟨k, hk, rfl⟩ := List.mem_map.mp hx
      rw [List.mem_range] at hk
      exact ⟨r, hr, by omega, by omega⟩
    have hC : ∀ x ∈ ((List.range (t.defs.getD d default).calls.size).flatMap fun i => (List.range ((t.defs.getD d default).calls.getD i default).count).flatMap fun j => (writes t lays f ((t.defs.getD d default).calls.getD i default).def_ (o + ((t.defs.getD d default).calls.getD i default).base + j * ((t.defs.getD d default).calls.getD i default).stride) (A + B * (((lays.getD d default).places.getD i default).nb + j * ((lays.getD d default).places.getD i default).cs)) (B * ((lays.getD d default).places.getD i default).ms)).map Prod.fst), ∃ i < (t.defs.getD d default).calls.size, o + ((t.defs.getD d default).calls.getD i default).base ≤ x ∧ x < o + ((t.defs.getD d default).calls.getD i default).base + ((t.defs.getD d default).calls.getD i default).count * (t.defs.getD ((t.defs.getD d default).calls.getD i default).def_ default).cols := by
      intro x hx
      obtain ⟨i, hi, hx⟩ := List.mem_flatMap.mp hx
      rw [List.mem_range] at hi
      obtain ⟨j, hj, hx⟩ := List.mem_flatMap.mp hx
      rw [List.mem_range] at hj
      have := (IH i hi j).1 x hx
      have := copy_le (base := o + ((t.defs.getD d default).calls.getD i default).base) (hp.stride i hi) hj
      exact ⟨i, hi, by omega, by omega⟩
    rw [writes_fst]
    refine ⟨fun x hx => ?_, ?_⟩
    · rcases List.mem_append.mp hx with h | h
      · obtain ⟨r, hr, h1, h2⟩ := hR x h
        have := hp.runs_in r hr
        omega
      · obtain ⟨i, hi, h1, h2⟩ := hC x h
        have := hp.calls_in i hi
        omega
    rw [List.nodup_append]
    refine ⟨?_, ?_, fun a ha b hb => ?_⟩
    · refine nodup_flatMap_range _ (fun r => o + ((lays.getD d default).runs.getD r default).1) (fun r => o + ((lays.getD d default).runs.getD r default).2.1) _
        (fun r _ => List.Pairwise.map _ (fun a b h e => h (Nat.add_left_cancel e)) List.nodup_range) (fun r _ x hx => ?_)
        (fun r r' h h' => by have := hp.runs_asc r r' h h'; omega)
      obtain ⟨k, hk, rfl⟩ := List.mem_map.mp hx
      rw [List.mem_range] at hk
      omega
    · refine nodup_flatMap_range _ (fun i => o + ((t.defs.getD d default).calls.getD i default).base) (fun i => o + ((t.defs.getD d default).calls.getD i default).base + ((t.defs.getD d default).calls.getD i default).count * (t.defs.getD ((t.defs.getD d default).calls.getD i default).def_ default).cols) _
        (fun i hi => ?_) (fun i hi x hx => ?_) (fun i i' h h' => by have := hp.calls_asc i i' h h'; omega)
      · refine nodup_flatMap_range _ (fun j => o + ((t.defs.getD d default).calls.getD i default).base + j * ((t.defs.getD d default).calls.getD i default).stride)
          (fun j => o + ((t.defs.getD d default).calls.getD i default).base + j * ((t.defs.getD d default).calls.getD i default).stride + (t.defs.getD ((t.defs.getD d default).calls.getD i default).def_ default).cols) _ (fun j _ => (IH i hi j).2) (fun j _ x hx => (IH i hi j).1 x hx)
          (fun j j' h h' => ?_)
        rw [hp.stride i hi (by omega)]
        have : (j + 1) * (t.defs.getD ((t.defs.getD d default).calls.getD i default).def_ default).cols ≤ j' * (t.defs.getD ((t.defs.getD d default).calls.getD i default).def_ default).cols := Nat.mul_le_mul_right _ h
        rw [Nat.succ_mul] at this
        omega
      · obtain ⟨j, hj, hx⟩ := List.mem_flatMap.mp hx
        rw [List.mem_range] at hj
        have := (IH i hi j).1 x hx
        have := copy_le (base := o + ((t.defs.getD d default).calls.getD i default).base) (hp.stride i hi) hj
        omega
    · obtain ⟨r, hr, h1, h2⟩ := hR a ha
      obtain ⟨i, hi, h3, h4⟩ := hC b hb
      have := hp.apart r hr i hi
      omega

/-- Every column `y` of copy `(d, o)` is written, at `A + B · posIn y`. -/
theorem writes_mem (t : Tmpl) (lays : Array Lay) (hpos : ∀ d < t.defs.size, DefPos t lays d) :
    ∀ f d o A B, d < t.defs.size → d < f → ∀ y < (t.defs.getD d default).cols, (o + y, A + B * posIn t lays f d y) ∈ writes t lays f d o A B
  | 0, _, _, _, _, _, h, _, _ => absurd h (Nat.not_lt_zero _)
  | f + 1, d, o, A, B, hd, hf, y, hy => by
    have hp := hpos d hd
    rcases hp.cover y hy with ⟨r, hr, h1, h2⟩ | ⟨i, hi, h1, h2⟩
    · rw [posIn_own t lays hp hr (rfl : ((lays.getD d default).runs.getD r default) = (((lays.getD d default).runs.getD r default).1, ((lays.getD d default).runs.getD r default).2.1, ((lays.getD d default).runs.getD r default).2.2)) h1 h2 f, writes]
      apply List.mem_append_left
      refine List.mem_flatMap.mpr ⟨((lays.getD d default).runs.getD r default), ?_, List.mem_map.mpr ⟨y - ((lays.getD d default).runs.getD r default).1, List.mem_range.mpr (by omega), ?_⟩⟩
      · rw [toList_getD]
        exact List.mem_map.mpr ⟨r, List.mem_range.mpr hr, rfl⟩
      · exact Prod.ext (by dsimp only; omega) rfl
    · have hLe : 0 < (t.defs.getD ((t.defs.getD d default).calls.getD i default).def_ default).cols := Nat.pos_of_ne_zero fun h => by rw [h, Nat.mul_zero] at h2; omega
      have hj : ((y - ((t.defs.getD d default).calls.getD i default).base) / (t.defs.getD ((t.defs.getD d default).calls.getD i default).def_ default).cols) < ((t.defs.getD d default).calls.getD i default).count := Nat.div_lt_of_lt_mul (by rw [Nat.mul_comm]; omega)
      have hc := hp.callee i hi
      have IH := writes_mem t lays hpos f ((t.defs.getD d default).calls.getD i default).def_ (o + ((t.defs.getD d default).calls.getD i default).base + ((y - ((t.defs.getD d default).calls.getD i default).base) / (t.defs.getD ((t.defs.getD d default).calls.getD i default).def_ default).cols) * ((t.defs.getD d default).calls.getD i default).stride)
        (A + B * (((lays.getD d default).places.getD i default).nb + ((y - ((t.defs.getD d default).calls.getD i default).base) / (t.defs.getD ((t.defs.getD d default).calls.getD i default).def_ default).cols) * ((lays.getD d default).places.getD i default).cs)) (B * ((lays.getD d default).places.getD i default).ms) (by omega) (by omega) ((y - ((t.defs.getD d default).calls.getD i default).base) % (t.defs.getD ((t.defs.getD d default).calls.getD i default).def_ default).cols) (Nat.mod_lt _ hLe)
      have ho : o + ((t.defs.getD d default).calls.getD i default).base + ((y - ((t.defs.getD d default).calls.getD i default).base) / (t.defs.getD ((t.defs.getD d default).calls.getD i default).def_ default).cols) * ((t.defs.getD d default).calls.getD i default).stride + ((y - ((t.defs.getD d default).calls.getD i default).base) % (t.defs.getD ((t.defs.getD d default).calls.getD i default).def_ default).cols) = o + y := by
        by_cases hcnt : 1 < ((t.defs.getD d default).calls.getD i default).count
        · rw [hp.stride i hi hcnt]
          have := Nat.div_add_mod' (y - ((t.defs.getD d default).calls.getD i default).base) (t.defs.getD ((t.defs.getD d default).calls.getD i default).def_ default).cols
          omega
        · have hc1 : ((t.defs.getD d default).calls.getD i default).count = 1 := by
            refine Nat.le_antisymm (by omega) (Nat.pos_of_ne_zero fun h0 => ?_)
            rw [h0, Nat.zero_mul] at h2
            omega
          rw [hc1, Nat.one_mul] at h2
          rw [Nat.div_eq_of_lt (by omega), Nat.zero_mul, Nat.mod_eq_of_lt (by omega)]
          omega
      rw [ho] at IH
      rw [posIn_call t lays hp hi h1 h2 f, writes]
      apply List.mem_append_right
      refine List.mem_flatMap.mpr ⟨i, List.mem_range.mpr hi, List.mem_flatMap.mpr ⟨((y - ((t.defs.getD d default).calls.getD i default).base) / (t.defs.getD ((t.defs.getD d default).calls.getD i default).def_ default).cols), List.mem_range.mpr hj, ?_⟩⟩
      simp only [Nat.mul_add, Nat.mul_assoc, Nat.add_assoc] at IH ⊢
      exact IH

/-- A copy whose writes are among `W`'s, each column written once, sits at its place under `setAll W`. -/
theorem treeAt_writes (t : Tmpl) (lays : Array Lay) (hpos : ∀ d < t.defs.size, DefPos t lays d) (W : List (Nat × Nat))
    (m : Array Nat) (hW : (W.map Prod.fst).Nodup) (hsz : ∀ w ∈ W, w.1 < m.size) :
    ∀ f d o A B, d < t.defs.size → d < f → (∀ w ∈ writes t lays f d o A B, w ∈ W) → TreeAt t lays (setAll W m) f d o A B
  | 0, _, _, _, _, _, h, _ => absurd h (Nat.not_lt_zero _)
  | f + 1, d, o, A, B, hd, hf, hsub => by
    refine ⟨fun y hy => ?_, fun i hi j hj => ?_⟩
    · have hmem := hsub _ (writes_mem t lays hpos (f + 1) d o A B hd hf y hy)
      rw [setAll_hit W m _ _ hW hmem (hsz _ hmem), posIn_stable t lays hpos (f + 1) (t.defs.size + 1) d y hd hf (by omega)]
    · have hc := (hpos d hd).callee i hi
      exact treeAt_writes t lays hpos W m hW hsz f _ _ _ _ (by omega) (by omega) fun w hw => hsub w (by
        rw [writes]
        exact List.mem_append_right _ (List.mem_flatMap.mpr ⟨i, List.mem_range.mpr hi,
          List.mem_flatMap.mpr ⟨j, List.mem_range.mpr hj, hw⟩⟩))

/-- `posMap` puts every copy of the unrolled unit at its place. -/
theorem posMap_tree (t : Tmpl) (lays : Array Lay) (hpos : ∀ d < t.defs.size, DefPos t lays d)
    (hroot : t.root < t.defs.size) : TreeAt t lays (posMap t lays) (t.defs.size + 1) t.root 0 0 1 := by
  have hok := writes_ok t lays hpos (t.defs.size + 1) t.root 0 0 1 hroot
  unfold posMap
  rw [fillMap_writes]
  refine treeAt_writes t lays hpos _ _ hok.2 (fun w hw => ?_) _ _ _ _ _ hroot (by omega) fun w hw => hw
  have := (hok.1 w.1 (List.mem_map_of_mem hw)).2
  rw [Array.size_replicate]
  omega

end FlockSoundness.TmplFold
