import Properties.Proofs.Pouw.Grid

/-!
# Proofs of the window's properties (`Properties.Pouw`, "The window")

`WindowAccepts`, `WindowCreditsTheCalls` and `TileAtCovers`, on Lean core. Each of `windowProblem`'s passes is restated
declaratively: `ordered` finds the calls exactly when the kinds read `kindAt` (`ordered_iff`), and `rangesFrom`
accepts exactly when there are cut points `cut 0 = at0 < cut 1 < …` that bound each entry's range (`rangesFrom_ok_iff`).
The calls then tile `[0, cut calls)` (`cut_sum`), so `tileAt` finds the one entry whose range holds `t` (`locate`,
`go_spec`) and its strictly increasing `keep` maps that range one to one (`keep_inj`).
-/

namespace Properties.Pouw.Window

open PouwWindow

/-! ### The entries' shape -/

/-- The kind entry `i` of a window with `calls` calls has. -/
def kindAt (calls i : Nat) : Kind := if i < calls then .call else if i = calls then .filler else .none

theorem ordered_iff (ks : List (Option Kind)) (c : Nat) :
    ordered ks = some c ↔ 1 ≤ c ∧ c < ks.length ∧ ∀ i x, ks[i]? = some x → x = some (kindAt c i) := by
  constructor
  · intro h
    unfold ordered at h
    split at h
    · simp at h
    · rename_i calls hf
      split at h
      · rename_i hc
        obtain ⟨h1, htake, hdrop⟩ := hc
        simp only [Option.some.injEq] at h
        subst h
        obtain ⟨hlt, hfil, hbefore⟩ := List.findIdx?_eq_some_iff_getElem.1 hf
        refine ⟨h1, hlt, fun i x hx => ?_⟩
        simp only [List.all_eq_true, beq_iff_eq] at htake hdrop
        unfold kindAt
        by_cases hi : i < calls
        · simp only [hi, ↓reduceIte]
          refine htake x (List.mem_of_getElem? (i := i) ?_)
          rw [List.getElem?_take, ite_eq_left hi, hx]
        · by_cases he : i = calls
          · subst he
            simp only [Nat.lt_irrefl, ↓reduceIte]
            rw [List.getElem?_eq_getElem hlt] at hx
            simp only [Option.some.injEq] at hx
            subst hx; simpa using hfil
          · simp only [hi, he, ↓reduceIte]
            refine hdrop x (List.mem_of_getElem? (i := i - (calls + 1)) ?_)
            rw [List.getElem?_drop, show calls + 1 + (i - (calls + 1)) = i by omega, hx]
      · simp at h
  · rintro ⟨h1, hlt, hk⟩
    have hget : ∀ i (h : i < ks.length), ks[i] = some (kindAt c i) := fun i h =>
      hk i _ (List.getElem?_eq_getElem h)
    have hf : ks.findIdx? (· == some .filler) = some c := by
      rw [List.findIdx?_eq_some_iff_getElem]
      refine ⟨hlt, by simp [hget, kindAt], fun j hj => ?_⟩
      simp [hget, kindAt, hj]
    have htk : ((ks.take c).all fun x => x == some Kind.call) = true := by
      simp only [List.all_eq_true, beq_iff_eq]
      intro x hx
      obtain ⟨i, hi⟩ := List.mem_iff_getElem?.1 hx
      rw [List.getElem?_take] at hi
      split at hi
      · rename_i h; rw [hk i x hi]; simp [kindAt, h]
      · simp at hi
    have hdp : ((ks.drop (c + 1)).all fun x => x == some Kind.none) = true := by
      simp only [List.all_eq_true, beq_iff_eq]
      intro x hx
      obtain ⟨i, hi⟩ := List.mem_iff_getElem?.1 hx
      rw [List.getElem?_drop] at hi
      rw [hk _ x hi]
      simp only [kindAt, ite_eq_right (show ¬ c + 1 + i < c by omega), ite_eq_right (show c + 1 + i ≠ c by omega)]
    unfold ordered
    rw [hf]
    exact ite_eq_left ⟨h1, htk, hdp⟩

/-! ### The ranges -/

/-- What `rangesFrom` checks of one entry with range `[lo, hi)`. -/
def RangeOK (x : Raw) (lo hi : Nat) : Prop :=
  (∃ a b, x.a = some a ∧ x.b = some b ∧ a.ok = true ∧ b.ok = true) ∧ x.lo = some lo ∧ x.hi = some hi ∧ lo < hi ∧
    (x.kind = some .call → hi - lo = x.tiles) ∧ (x.kind = some .filler → hi - lo ≤ x.tiles)

theorem rangesFrom_ok_iff (xs : List Raw) : ∀ e at0 r, rangesFrom e at0 xs = .ok r ↔
    ∃ cut : Nat → Nat, cut 0 = at0 ∧ cut xs.length = r ∧
      ∀ i x, xs[i]? = some x → RangeOK x (cut i) (cut (i + 1)) := by
  induction xs with
  | nil =>
    intro e at0 r
    simp only [rangesFrom, Except.ok.injEq, List.length_nil, List.getElem?_nil, reduceCtorEq, false_implies,
      implies_true, and_true]
    exact ⟨fun h => ⟨fun _ => at0, rfl, h⟩, fun ⟨cut, h0, h1⟩ => h0 ▸ h1⟩
  | cons x xs ih =>
    intro e at0 r
    constructor
    · intro h
      unfold rangesFrom at h
      split at h
      · rename_i a b ha hb
        split at h
        · simp at h
        · rename_i hab
          split at h
          · rename_i lo hi hlo hhi
            split at h
            · simp at h
            · rename_i hr
              split at h
              · simp at h
              · rename_i hcall
                split at h
                · simp at h
                · rename_i hfill
                  obtain ⟨cut, h0, h1, h2⟩ := (ih _ _ _).1 h
                  refine ⟨fun i => if i = 0 then at0 else cut (i - 1), by simp, by simpa using h1, ?_⟩
                  intro i y hy
                  cases i with
                  | zero =>
                    simp only [List.getElem?_cons_zero, Option.some.injEq] at hy
                    subst hy
                    simp only [↓reduceIte, Nat.zero_add, Nat.add_one_ne_zero, h0]
                    simp only [Bool.not_eq_true', Bool.and_eq_false_iff, not_or, Bool.not_eq_false] at hab
                    simp only [not_or, Decidable.not_not, Nat.not_le] at hr
                    obtain ⟨rfl, hlt⟩ := hr
                    simp only [not_and, Decidable.not_not] at hcall
                    simp only [not_and, Nat.not_lt] at hfill
                    exact ⟨⟨a, b, ha, hb, hab.1, hab.2⟩, hlo, hhi, hlt, hcall, hfill⟩
                  | succ i =>
                    simp only [List.getElem?_cons_succ] at hy
                    simpa using h2 i y hy
          · simp at h
      · simp at h
    · rintro ⟨cut, h0, h1, h2⟩
      obtain ⟨⟨a, b, ha, hb, haok, hbok⟩, hlo, hhi, hlt, hcall, hfill⟩ := h2 0 x rfl
      have hrest : rangesFrom (e + 1) (cut 1) xs = .ok r :=
        (ih _ _ _).2 ⟨fun i => cut (i + 1), rfl, by simpa using h1, fun i y hy => by simpa using h2 (i + 1) y hy⟩
      unfold rangesFrom
      simp only [ha, hb, haok, hbok, Bool.and_self, Bool.not_true, Bool.false_eq_true, ↓reduceIte, hlo, hhi, h0]
      rw [h0] at hlt hcall hfill
      rw [ite_eq_right (by omega), ite_eq_right (by rintro ⟨hk, hne⟩; exact hne (hcall hk)),
        ite_eq_right (by rintro ⟨hk, hgt⟩; have := hfill hk; omega)]
      exact hrest

/-! ### The window statement -/

theorem entryOK_iff {x : Raw} {lo hi calls i : Nat} (hk : x.kind = some (kindAt calls i)) (hi' : i ≤ calls) :
    EntryOK x lo hi (decide (i < calls)) ↔ RangeOK x lo hi := by
  unfold EntryOK RangeOK
  by_cases h : i < calls
  · simp only [kindAt, h, ↓reduceIte] at hk
    simp [hk, h]
  · have he : i = calls := by omega
    subst he
    simp only [kindAt, Nat.lt_irrefl, ↓reduceIte] at hk
    simp [hk]

theorem windowAccepts : WindowAccepts := by
  intro es work bound bits
  have hmap : ∀ i, (es.map (·.kind))[i]? = es[i]?.map (fun x : Raw => x.kind) := fun i => List.getElem?_map
  constructor
  · intro h
    unfold windowProblem at h
    split at h
    · simp at h
    · rename_i hlen
      split at h
      · simp at h
      · rename_i hany
        split at h
        · simp at h
        · rename_i calls hord
          split at h
          · simp at h
          · rename_i at_ hr
            dsimp only at h
            split at h
            · simp at h
            · rename_i hat
              split at h
              · simp at h
              · rename_i hbk
                obtain ⟨h1, hlt, hk⟩ := (ordered_iff _ _).1 hord
                rw [List.length_map] at hlt
                have hkind : ∀ i x, es[i]? = some x → x.kind = some (kindAt calls i) := fun i x hx =>
                  hk i x.kind (by rw [hmap, hx]; rfl)
                obtain ⟨cut, h0, hend, hcut⟩ := (rangesFrom_ok_iff _ 0 0 at_).1 hr
                rw [List.length_take, Nat.min_eq_left (by omega)] at hend
                have hgood : ∀ i x, i ≤ calls → es[i]? = some x → RangeOK x (cut i) (cut (i + 1)) :=
                  fun i x hi hx => hcut i x (by rw [List.getElem?_take, ite_eq_left (by omega)]; exact hx)
                obtain ⟨xc, hxc⟩ : ∃ xc, es[calls]? = some xc := ⟨_, List.getElem?_eq_getElem hlt⟩
                have hreal : ((es.getD calls ⟨none, none, none, none, none⟩).lo).getD 0 = cut calls := by
                  rw [List.getD_eq_getElem?_getD, hxc]; simp [(hgood calls xc (Nat.le_refl _) hxc).2.1]
                rw [hreal] at hbk
                refine ⟨by omega, calls, h1, hlt, fun i x hx => hkind i x hx, cut, h0, by omega, by
                  simpa using hbk, fun i x hi hx => (entryOK_iff (hkind i x hx) hi).2 (hgood i x hi hx)⟩
  · rintro ⟨hlen, calls, h1, hlt, hkind, cut, h0, hend, hbk, hok⟩
    have hk : ∀ i x, es[i]? = some x → x.kind = some (kindAt calls i) := hkind
    have hord : ordered (es.map (·.kind)) = some calls := by
      refine (ordered_iff _ _).2 ⟨h1, by simpa using hlt, fun i k hx => ?_⟩
      rw [hmap] at hx
      obtain ⟨x, hx', rfl⟩ := Option.map_eq_some_iff.1 hx
      exact hk i x hx'
    have hgood : ∀ i x, i ≤ calls → es[i]? = some x → RangeOK x (cut i) (cut (i + 1)) :=
      fun i x hi hx => (entryOK_iff (hk i x hx) hi).1 (hok i x hi hx)
    have hr : rangesFrom 0 0 (es.take (calls + 1)) = .ok (cut (calls + 1)) := by
      refine (rangesFrom_ok_iff _ 0 0 _).2 ⟨cut, h0, by rw [List.length_take, Nat.min_eq_left (by omega)], ?_⟩
      intro i x hx
      rw [List.getElem?_take] at hx
      split at hx
      · exact hgood i x (by omega) hx
      · simp at hx
    obtain ⟨xc, hxc⟩ : ∃ xc, es[calls]? = some xc := ⟨_, List.getElem?_eq_getElem hlt⟩
    have hreal : ((es.getD calls ⟨none, none, none, none, none⟩).lo).getD 0 = cut calls := by
      rw [List.getD_eq_getElem?_getD, hxc]; simp [(hgood calls xc (Nat.le_refl _) hxc).2.1]
    have hany : es.any (·.kind.isNone) = false := by
      rw [List.any_eq_false]
      intro x hx
      obtain ⟨i, hi⟩ := List.mem_iff_getElem?.1 hx
      simp [hk i x hi]
    unfold windowProblem
    simp only [hlen, ne_eq, not_true_eq_false, ↓reduceIte, hany, Bool.false_eq_true, hord, hr, hreal, hend, hbk]

theorem cut_sum {es : List Raw} {calls : Nat} {cut : Nat → Nat} (hlt : calls < es.length) (h0 : cut 0 = 0)
    (hok : ∀ i x, i ≤ calls → es[i]? = some x → EntryOK x (cut i) (cut (i + 1)) (decide (i < calls))) :
    ∀ k, k ≤ calls → cut k = ((es.take k).map Raw.tiles).sum := by
  intro k
  induction k with
  | zero => intro _; simp [h0]
  | succ k ih =>
    intro hk
    have hx : es[k]? = some es[k] := List.getElem?_eq_getElem (by omega)
    have := hok k es[k] (by omega) hx
    unfold EntryOK at this
    simp only [show k < calls by omega, decide_true, ↓reduceIte] at this
    rw [List.take_add_one, hx, List.map_append, List.sum_append, ← ih (by omega)]
    simp only [Option.toList_some, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, Nat.add_zero]
    omega

theorem windowCreditsTheCalls : WindowCreditsTheCalls := by
  intro es work bound bits h
  obtain ⟨-, calls, h1, hlt, hkind, cut, h0, hend, hbk, hok⟩ := (windowAccepts es work bound bits).1 h
  refine ⟨calls, ?_, ?_, ?_⟩
  · rw [List.all_eq_true]
    intro x hx
    obtain ⟨i, hi⟩ := List.mem_iff_getElem?.1 hx
    rw [List.getElem?_take] at hi
    split at hi
    · rename_i h; simp [hkind i x hi, h]
    · simp at hi
  · rw [List.getElem?_eq_getElem hlt]
    simp [hkind calls _ (List.getElem?_eq_getElem hlt)]
  · rw [← cut_sum hlt h0 hok calls (Nat.le_refl _), hbk]

/-! ### The tile map -/

theorem getD_mem {l : List Nat} {n d : Nat} (h : n < l.length) : l.getD n d ∈ l := by
  rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem h]; exact List.getElem_mem h

theorem getD_eq {l : List Nat} {n d : Nat} (h : n < l.length) : l.getD n d = l[n] := by
  rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem h]; rfl

/-- Kept rows strictly ascend, so a kept row has one index. -/
theorem keep_strict {l : List Nat} (h : ((l.zip l.tail).all fun (x, y) => decide (x < y)) = true) :
    ∀ p q (hp : p < l.length) (hq : q < l.length), p < q → l[p] < l[q] := by
  have adj : ∀ k (hk : k + 1 < l.length), l[k] < l[k + 1] := by
    intro k hk
    have hz : k < (l.zip l.tail).length := by simp [List.length_zip]; omega
    have := (List.all_eq_true.1 h) _ (List.getElem_mem hz)
    simpa [List.getElem_zip, List.getElem_tail] using this
  intro p q hp hq hpq
  induction q with
  | zero => omega
  | succ q ih =>
    rcases Nat.lt_succ_iff_lt_or_eq.1 hpq with h' | rfl
    · exact Nat.lt_trans (ih (by omega) h') (adj q hq)
    · exact adj p hq

theorem keep_inj {s : Side} (h : s.ok = true) {p q : Nat} (hp : p < s.keep.length) (hq : q < s.keep.length)
    (he : s.keep[p] = s.keep[q]) : p = q := by
  have hs : ((s.keep.zip s.keep.tail).all fun (x, y) => decide (x < y)) = true := by
    simp only [Side.ok, Bool.and_eq_true] at h; exact h.2
  rcases Nat.lt_trichotomy p q with h' | h' | h'
  · have := keep_strict hs p q hp hq h'; omega
  · exact h'
  · have := keep_strict hs q p hq hp h'; omega

theorem keep_pos {s : Side} (h : s.ok = true) : 0 < s.keep.length := by
  simp only [Side.ok, Bool.and_eq_true, Bool.not_eq_true', List.isEmpty_eq_false_iff] at h
  exact List.length_pos_iff.2 h.1.1.2

theorem tileIn_kind_none {x : Raw} (h : x.kind = some .none) (t : Nat) : tileIn x t = none := by
  unfold tileIn; split <;> simp_all

theorem tileIn_in {x : Raw} {lo hi t : Nat} {a b : Side} (hk : x.kind ≠ some .none) (hk' : x.kind.isSome)
    (hlo : x.lo = some lo) (hhi : x.hi = some hi) (ha : x.a = some a) (hb : x.b = some b) (h1 : lo ≤ t) (h2 : t < hi) :
    tileIn x t = some (a.keep.getD ((t - lo) / b.keep.length) 0, b.keep.getD ((t - lo) % b.keep.length) 0) := by
  obtain ⟨k, hkk⟩ := Option.isSome_iff_exists.1 hk'
  unfold tileIn
  rw [hkk, hlo, hhi]
  simp only
  rw [ite_eq_right (fun h => h.elim (fun h' => hk (by rw [hkk, h'])) (fun h' => h' ⟨h1, h2⟩))]
  simp [ha, hb]

theorem tileIn_out {x : Raw} {lo hi t : Nat} (hlo : x.lo = some lo) (hhi : x.hi = some hi) (h : ¬ (lo ≤ t ∧ t < hi)) :
    tileIn x t = none := by
  unfold tileIn
  rw [hlo, hhi]
  split
  · rename_i k _ _ hkk hl hh
    simp only [Option.some.injEq] at hl hh
    subst hl hh
    rw [ite_eq_left (Or.inr h)]
  · rfl

theorem go_some {t : Nat} {v : Nat × Nat} : ∀ (es : List Raw) (e0 i : Nat) (x : Raw), es[i]? = some x →
    tileIn x t = some v → (∀ k y, es[k]? = some y → k ≠ i → tileIn y t = none) →
    tileAt.go t e0 es = some (e0 + i, v.1, v.2) := by
  intro es
  induction es with
  | nil => intro _ _ _ h; simp at h
  | cons y ys ih =>
    intro e0 i x hx hv hother
    cases i with
    | zero =>
      simp only [List.getElem?_cons_zero, Option.some.injEq] at hx
      subst hx
      simp [tileAt.go, hv]
    | succ i =>
      have hy : tileIn y t = none := hother 0 y rfl (by omega)
      simp only [tileAt.go, hy]
      rw [ih (e0 + 1) i x (by simpa using hx) hv (fun k z hz hne => hother (k + 1) z (by simpa using hz) (by omega))]
      congr 2; omega

theorem go_none {t : Nat} : ∀ (es : List Raw) (e0 : Nat), (∀ (k : Nat) (y : Raw), es[k]? = some y → tileIn y t = none) →
    tileAt.go t e0 es = none := by
  intro es
  induction es with
  | nil => intro _ _; rfl
  | cons y ys ih =>
    intro e0 h
    simp only [tileAt.go, h 0 y rfl]
    exact ih _ fun k z hz => h (k + 1) z (by simpa using hz)

theorem go_spec {t : Nat} : ∀ (es : List Raw) (e0 e i j : Nat), tileAt.go t e0 es = some (e, i, j) →
    ∃ k x, e = e0 + k ∧ es[k]? = some x ∧ tileIn x t = some (i, j) := by
  intro es
  induction es with
  | nil => intro _ _ _ _ h; simp [tileAt.go] at h
  | cons y ys ih =>
    intro e0 e i j h
    simp only [tileAt.go] at h
    split at h
    · rename_i i' j' hy
      simp only [Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl, rfl⟩ := h
      exact ⟨0, y, by omega, rfl, hy⟩
    · obtain ⟨k, x, he, hx, hv⟩ := ih _ _ _ _ h
      exact ⟨k + 1, x, by omega, by simpa using hx, hv⟩

theorem cut_lt {cut : Nat → Nat} {calls : Nat} (hstep : ∀ i, i ≤ calls → cut i < cut (i + 1)) :
    ∀ a b, a < b → b ≤ calls + 1 → cut a < cut b := by
  intro a b hab hb
  induction b with
  | zero => omega
  | succ b ih =>
    rcases Nat.lt_succ_iff_lt_or_eq.1 hab with h | rfl
    · exact Nat.lt_trans (ih h (by omega)) (hstep b (by omega))
    · exact hstep a (by omega)

theorem locate {cut : Nat → Nat} (h0 : cut 0 = 0) : ∀ m t, t < cut m → ∃ i, i < m ∧ cut i ≤ t ∧ t < cut (i + 1) := by
  intro m
  induction m with
  | zero => intro t h; omega
  | succ m ih =>
    intro t h
    by_cases hm : t < cut m
    · obtain ⟨i, h1, h2, h3⟩ := ih t hm; exact ⟨i, by omega, h2, h3⟩
    · exact ⟨m, by omega, by omega, h⟩

theorem tileAtCovers : TileAtCovers := by
  intro es work bound bits h
  obtain ⟨-, calls, h1, hlt, hkind, cut, h0, hend, -, hok⟩ := (windowAccepts es work bound bits).1 h
  have hgood : ∀ i x, i ≤ calls → es[i]? = some x → RangeOK x (cut i) (cut (i + 1)) :=
    fun i x hi hx => (entryOK_iff (hkind i x hx) hi).1 (hok i x hi hx)
  have hstep : ∀ i, i ≤ calls → cut i < cut (i + 1) :=
    fun i hi => (hgood i _ hi (List.getElem?_eq_getElem (by omega))).2.2.2.1
  have hlt' := cut_lt hstep
  have hle : ∀ a b, a ≤ b → b ≤ calls + 1 → cut a ≤ cut b := fun a b hab hb => by
    rcases Nat.lt_or_eq_of_le hab with h | rfl
    · exact Nat.le_of_lt (hlt' a b h hb)
    · exact Nat.le_refl _
  have hlen : ∀ k y, es[k]? = some y → k < es.length := fun k y hy => (List.getElem?_eq_some_iff.1 hy).1
  have hnotnone : ∀ k y, k ≤ calls → es[k]? = some y → y.kind ≠ some .none ∧ y.kind.isSome := by
    intro k y hk hy
    rw [hkind k y hy]
    split <;> (try split) <;> simp <;> omega
  -- the one entry whose range holds t, and every other entry's `tileIn` is none
  have hother : ∀ t i, i ≤ calls → cut i ≤ t → t < cut (i + 1) → ∀ k y, es[k]? = some y → k ≠ i →
      tileIn y t = none := by
    intro t i hi h1 h2 k y hy hne
    by_cases hk : k ≤ calls
    · obtain ⟨-, hlo, hhi, -⟩ := hgood k y hk hy
      refine tileIn_out hlo hhi fun ⟨h3, h4⟩ => ?_
      rcases Nat.lt_or_gt_of_ne hne with h' | h'
      · have := hle (k + 1) i (by omega) (by omega); omega
      · have := hle (i + 1) k (by omega) (by omega); omega
    · refine tileIn_kind_none ?_ t
      rw [hkind k y hy]; simp [show ¬ k < calls by omega, show k ≠ calls by omega]
  have hside : ∀ i x, i ≤ calls → es[i]? = some x → ∃ a b, x.a = some a ∧ x.b = some b ∧ a.ok = true ∧
      b.ok = true ∧ cut (i + 1) - cut i ≤ a.keep.length * b.keep.length := by
    intro i x hi hx
    obtain ⟨⟨a, b, ha, hb, haok, hbok⟩, -, -, -, hcall, hfill⟩ := hgood i x hi hx
    have htiles : x.tiles = a.keep.length * b.keep.length := by
      unfold Raw.tiles; rw [ite_eq_right (hnotnone i x hi hx).1, ha, hb]; rfl
    refine ⟨a, b, ha, hb, haok, hbok, ?_⟩
    rw [← htiles]
    rcases Nat.lt_or_ge i calls with h' | h'
    · rw [hcall (by rw [hkind i x hx]; simp [h'])]; exact Nat.le_refl _
    · exact hfill (by rw [hkind i x hx]; simp [show i = calls by omega])
  -- tileAt on an index in entry i's range
  have hat : ∀ t i x a b, i ≤ calls → es[i]? = some x → x.a = some a → x.b = some b → cut i ≤ t →
      t < cut (i + 1) → tileAt es t = some (i, a.keep.getD ((t - cut i) / b.keep.length) 0,
        b.keep.getD ((t - cut i) % b.keep.length) 0) := by
    intro t i x a b hi hx ha hb h1 h2
    obtain ⟨-, hlo, hhi, -⟩ := hgood i x hi hx
    have hv := tileIn_in (hnotnone i x hi hx).1 (hnotnone i x hi hx).2 hlo hhi ha hb h1 h2
    have := go_some es 0 i x hx hv (hother t i hi h1 h2)
    simpa [tileAt] using this
  have hsome : ∀ t, t < work.n → ∃ i, i ≤ calls ∧ cut i ≤ t ∧ t < cut (i + 1) := fun t ht => by
    obtain ⟨i, h1, h2, h3⟩ := locate h0 (calls + 1) t (by omega)
    exact ⟨i, by omega, h2, h3⟩
  have hfind : ∀ t e i j, tileAt es t = some (e, i, j) → ∃ x, es[e]? = some x ∧ tileIn x t = some (i, j) := by
    intro t e i j h
    obtain ⟨k, x, he, hx, hv⟩ := go_spec es 0 e i j h
    exact ⟨x, by rw [he, Nat.zero_add]; exact hx, hv⟩
  have hin : ∀ t e i j, tileAt es t = some (e, i, j) → e ≤ calls ∧ cut e ≤ t ∧ t < cut (e + 1) := by
    intro t e i j h
    obtain ⟨x, hx, hv⟩ := hfind t e i j h
    by_cases he : e ≤ calls
    · obtain ⟨-, hlo, hhi, -⟩ := hgood e x he hx
      by_cases hr : cut e ≤ t ∧ t < cut (e + 1)
      · exact ⟨he, hr⟩
      · rw [tileIn_out hlo hhi hr] at hv; simp at hv
    · rw [tileIn_kind_none (by rw [hkind e x hx]; simp [show ¬ e < calls by omega,
        show e ≠ calls by omega]) t] at hv
      simp at hv
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro t
    by_cases ht : t < work.n
    · obtain ⟨i, hi, h1, h2⟩ := hsome t ht
      obtain ⟨a, b, ha, hb, -⟩ := hside i _ hi (List.getElem?_eq_getElem (by omega))
      rw [hat t i _ a b hi (List.getElem?_eq_getElem (by omega)) ha hb h1 h2]
      simp [ht]
    · have : tileAt es t = none := by
        unfold tileAt
        refine go_none es 0 fun k y hy => ?_
        by_cases hk : k ≤ calls
        · obtain ⟨-, hlo, hhi, -⟩ := hgood k y hk hy
          refine tileIn_out hlo hhi fun ⟨_, h4⟩ => ?_
          have := hle (k + 1) (calls + 1) (by omega) (Nat.le_refl _); omega
        · refine tileIn_kind_none ?_ t
          rw [hkind k y hy]; simp [show ¬ k < calls by omega, show k ≠ calls by omega]
      rw [this]; simp [ht]
  · intro t t' hs heq
    obtain ⟨⟨e, i, j⟩, hr⟩ := Option.isSome_iff_exists.1 hs
    obtain ⟨he, h1, h2⟩ := hin t e i j hr
    obtain ⟨he', h1', h2'⟩ := hin t' e i j (heq ▸ hr)
    obtain ⟨x, hx⟩ : ∃ x, es[e]? = some x := ⟨_, List.getElem?_eq_getElem (by omega)⟩
    obtain ⟨a, b, ha, hb, haok, hbok, htl⟩ := hside e x he hx
    have hL := keep_pos hbok
    have r1 := hat t e x a b he hx ha hb h1 h2
    have r2 := hat t' e x a b he hx ha hb h1' h2'
    rw [r1] at heq; rw [r2] at heq
    simp only [Option.some.injEq, Prod.mk.injEq, true_and] at heq
    obtain ⟨hA, hB⟩ := heq
    have hu : t - cut e < a.keep.length * b.keep.length := by omega
    have hu' : t' - cut e < a.keep.length * b.keep.length := by omega
    have hd := (Nat.div_lt_iff_lt_mul hL).2 hu
    have hd' := (Nat.div_lt_iff_lt_mul hL).2 hu'
    rw [getD_eq hd, getD_eq hd'] at hA
    rw [getD_eq (Nat.mod_lt _ hL), getD_eq (Nat.mod_lt _ hL)] at hB
    have q1 := keep_inj haok hd hd' hA
    have q2 := keep_inj hbok (Nat.mod_lt _ hL) (Nat.mod_lt _ hL) hB
    have := Nat.div_add_mod (t - cut e) b.keep.length
    have := Nat.div_add_mod (t' - cut e) b.keep.length
    rw [q1, q2] at *
    omega
  · intro t e i j hr
    obtain ⟨he, h1, h2⟩ := hin t e i j hr
    obtain ⟨x, hx⟩ : ∃ x, es[e]? = some x := ⟨_, List.getElem?_eq_getElem (by omega)⟩
    obtain ⟨a, b, ha, hb, haok, hbok, htl⟩ := hside e x he hx
    have hL := keep_pos hbok
    rw [hat t e x a b he hx ha hb h1 h2] at hr
    simp only [Option.some.injEq, Prod.mk.injEq, true_and] at hr
    obtain ⟨rfl, rfl⟩ := hr
    have hu : t - cut e < a.keep.length * b.keep.length := by omega
    exact ⟨x, a, b, hx, (hnotnone e x he hx).1, ha, hb, getD_mem ((Nat.div_lt_iff_lt_mul hL).2 hu),
      getD_mem (Nat.mod_lt _ hL)⟩
  · intro e x a b hx hcall ha hb i hi j hj
    have he : e < calls := by
      rw [hkind e x hx] at hcall
      split at hcall
      · assumption
      · split at hcall <;> simp at hcall
    obtain ⟨-, hlo, hhi, hlt, htiles, -⟩ := hgood e x (by omega) hx
    have htl : cut (e + 1) - cut e = a.keep.length * b.keep.length := by
      rw [htiles hcall]; unfold Raw.tiles; rw [ite_eq_right (by rw [hcall]; simp), ha, hb]; rfl
    obtain ⟨p, hp, rfl⟩ := List.mem_iff_getElem.1 hi
    obtain ⟨q, hq, rfl⟩ := List.mem_iff_getElem.1 hj
    have hm : b.keep.length * p + q < a.keep.length * b.keep.length := by
      have : b.keep.length * (p + 1) ≤ b.keep.length * a.keep.length := Nat.mul_le_mul_left _ hp
      rw [Nat.mul_succ, Nat.mul_comm b.keep.length a.keep.length] at this; omega
    refine ⟨cut e + (b.keep.length * p + q), ?_⟩
    rw [hat _ e x a b (by omega) hx ha hb (by omega) (by omega), Nat.add_sub_cancel_left]
    obtain ⟨d1, m1⟩ := divmod (q := p) hq
    rw [d1, m1, getD_eq hp, getD_eq hq]

end Properties.Pouw.Window
