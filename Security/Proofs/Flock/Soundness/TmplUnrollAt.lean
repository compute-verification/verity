import Proofs.Flock.Soundness.TmplLayOk
import Proofs.Flock.Soundness.TmplWf
import Flock.TmplAt

/-!
# A templated unit's committed net exists, checked on its description

`Tmpl.unrollAt` (the rows, `Net.ofRows`' checks, then `Net.ofRowsAt`'s at `posMap`'s positions) succeeds on a template
that passes `Tmpl.unrollOk` at `layoutM`'s layouts (`unrollAt_ok`), so a verifier that checks the description knows the
committed net exists without building it.

- `invOf_spec`: the inverse `ofRowsAt` builds of a one-to-one map is its inverse.
- `place_inj`, `layApart_apart`: a layout's ranges are apart and each place's copies don't meet, so `posIn` is one-to-one
  below the layout's size (`posIn_ok`).
- `posIn_ident`: a layout `identOk` passes has every column at its own index, and so do the root's input rows
  (`posOf_input`).
- `layoutM_facts`: what `layoutM` checked: the input rows' runs and calls at their columns, each output group at a
  multiple of its width with its columns in order, the constant last.
- `unitNet_eq`: the net the verifier carries (`Tmpl.unitNet`, its header from the description and its rows built only
  when read) is `unrollAt`'s, so a read templated unit's net is its template's committed net (`TmplUnit.unrolled`).
-/

open Flock Flock.Tmpl Flock.TmplApart

namespace FlockSoundness.TmplFold

/-! ## The inverse map -/

theorem getD_setIfInBounds {α : Type} (xs : Array α) (i j : Nat) (v dflt : α) :
    (xs.setIfInBounds i v).getD j dflt = if i = j ∧ i < xs.size then v else xs.getD j dflt := by
  simp only [Array.getD_eq_getD_getElem?, Array.getElem?_setIfInBounds]
  by_cases e : i = j
  · subst e
    by_cases h : i < xs.size
    · simp [h]
    · simp [h]
  · simp [e]

theorem invOf_fold (pos : Array Nat) (size : Nat)
    (hinj : ∀ x y, x < pos.size → y < pos.size → pos.getD x 0 = pos.getD y 0 → x = y) :
    ∀ n, n ≤ pos.size →
      let inv := (List.range n).foldl (fun inv x => inv.setIfInBounds (pos.getD x 0) (some x))
        (Array.replicate size none)
      inv.size = size ∧ (∀ x < n, pos.getD x 0 < size → inv.getD (pos.getD x 0) none = some x) ∧
        ∀ p x, inv.getD p none = some x → x < n ∧ pos.getD x 0 = p
  | 0, _ => by
    refine ⟨by simp, fun x hx => absurd hx (Nat.not_lt_zero _), fun p x h => ?_⟩
    simp only [List.range_zero, List.foldl_nil, Array.getD_eq_getD_getElem?, Array.getElem?_replicate] at h
    split at h <;> simp at h
  | n + 1, hn => by
    obtain ⟨hs, h1, h2⟩ := invOf_fold pos size hinj n (by omega)
    dsimp only at hs h1 h2 ⊢
    rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil]
    generalize (List.range n).foldl (fun inv x => inv.setIfInBounds (pos.getD x 0) (some x)) (Array.replicate size none) = inv at hs h1 h2
    refine ⟨by rw [Array.size_setIfInBounds, hs], fun x hx hlt => ?_, fun p x h => ?_⟩
    · rw [getD_setIfInBounds]
      by_cases e : x = n
      · subst e; rw [if_pos ⟨rfl, by omega⟩]
      · have hne : pos.getD n 0 ≠ pos.getD x 0 := fun h => e (hinj x n (by omega) (by omega) h.symm)
        rw [if_neg fun h => hne h.1]
        exact h1 x (by omega) hlt
    · rw [getD_setIfInBounds] at h
      split at h
      · rename_i e
        cases h; exact ⟨by omega, e.1⟩
      · have := h2 p x h; exact ⟨by omega, this.2⟩

/-- **`invOf` inverts a one-to-one map**: it has `size` entries, `pos[x]`'s is `x` wherever `pos[x] < size`, and each
`some x` it holds is at `pos[x]`, `x` a column of `pos`. -/
theorem invOf_spec {pos : Array Nat} {size : Nat}
    (hinj : ∀ x y, x < pos.size → y < pos.size → pos.getD x 0 = pos.getD y 0 → x = y) :
    (Net.invOf pos size).size = size ∧
      (∀ x < pos.size, pos.getD x 0 < size → (Net.invOf pos size).getD (pos.getD x 0) none = some x) ∧
      ∀ p x, (Net.invOf pos size).getD p none = some x → x < pos.size ∧ pos.getD x 0 = p :=
  invOf_fold pos size hinj pos.size (Nat.le_refl _)

/-! ## One place's copies -/

/-- **A place's copies don't meet**: copy `j`'s position `q` at `j · cs + ms · q` is one-to-one in `(j, q)`. -/
theorem place_inj {count cs ms S j1 j2 q1 q2 : Nat}
    (hp : 0 < ms ∧ (count ≤ 1 ∨ (ms = 1 ∧ S ≤ cs) ∨ (cs = 1 ∧ count ≤ ms)))
    (hj1 : j1 < count) (hj2 : j2 < count) (hq1 : q1 < S) (hq2 : q2 < S)
    (he : j1 * cs + ms * q1 = j2 * cs + ms * q2) : j1 = j2 ∧ q1 = q2 := by
  obtain ⟨hms, h | ⟨e, hS⟩ | ⟨e, hc⟩⟩ := hp
  · have : j1 = 0 := by omega
    have : j2 = 0 := by omega
    subst j1; subst j2
    simp only [Nat.zero_mul, Nat.zero_add] at he
    exact ⟨rfl, Nat.eq_of_mul_eq_mul_left hms he⟩
  · subst e
    have hcs : 0 < cs := by omega
    simp only [Nat.one_mul] at he
    have hd := congrArg (· / cs) he
    have hm := congrArg (· % cs) he
    simp only [Nat.mul_comm _ cs, Nat.mul_add_div hcs, Nat.mul_add_mod, Nat.div_eq_of_lt (Nat.lt_of_lt_of_le hq1 hS),
      Nat.div_eq_of_lt (Nat.lt_of_lt_of_le hq2 hS), Nat.mod_eq_of_lt (Nat.lt_of_lt_of_le hq1 hS),
      Nat.mod_eq_of_lt (Nat.lt_of_lt_of_le hq2 hS), Nat.add_zero] at hd hm
    exact ⟨hd, hm⟩
  · subst e
    simp only [Nat.mul_one] at he
    have hd := congrArg (· / ms) he
    have hm := congrArg (· % ms) he
    simp only [Nat.add_comm _ (ms * _), Nat.mul_add_div hms, Nat.mul_add_mod, Nat.div_eq_of_lt (by omega : j1 < ms),
      Nat.div_eq_of_lt (by omega : j2 < ms), Nat.mod_eq_of_lt (by omega : j1 < ms), Nat.mod_eq_of_lt (by omega : j2 < ms),
      Nat.add_zero] at hd hm
    exact ⟨hm, hd⟩

/-! ## A layout's ranges -/

section
variable {t : Tmpl} {lays : Array Lay} {d : Nat}

/-- **`layApart`'s ranges are apart**: two of a layout's nonempty ranges share no position. -/
theorem layApart_apart (h : layApart t lays d = true) {k1 k2 : Nat}
    (hk1 : k1 < (lays.getD d default).runs.size + (t.defs.getD d default).calls.size)
    (hk2 : k2 < (lays.getD d default).runs.size + (t.defs.getD d default).calls.size) (hne : k1 ≠ k2)
    (h1 : (region t lays d k1).1 < (region t lays d k1).2) (h2 : (region t lays d k2).1 < (region t lays d k2).2) :
    Apart (region t lays d k1) (region t lays d k2) := by
  unfold layApart at h
  simp only [Bool.and_eq_true] at h
  generalize hR : ((List.range ((lays.getD d default).runs.size + (t.defs.getD d default).calls.size)).map
    (region t lays d)).filter (fun s => decide (s.1 < s.2)) = R at h
  have hlt : ∀ s ∈ R.mergeSort (fun a b => decide (a.1 ≤ b.1)), s.1 < s.2 := fun s hs => by
    have := List.mem_mergeSort.1 hs
    rw [← hR] at this
    simpa using (List.mem_filter.1 this).2
  have hP : R.Pairwise Apart := (ascend_pairwise hlt h.1.1).perm (List.mergeSort_perm _ _) apart_symm
  rw [← hR, List.pairwise_filter, List.pairwise_map, List.pairwise_iff_getElem] at hP
  rcases Nat.lt_or_gt_of_ne hne with hl | hl
  · have := hP k1 k2 (by simpa using hk1) (by simpa using hk2) hl
    simp only [List.getElem_range] at this
    exact this (by simpa using h1) (by simpa using h2)
  · have := hP k2 k1 (by simpa using hk2) (by simpa using hk1) hl
    simp only [List.getElem_range] at this
    exact apart_symm (this (by simpa using h2) (by simpa using h1))

/-- Each of a layout's nonempty ranges ends inside its size. -/
theorem layApart_size (h : layApart t lays d = true) {k : Nat}
    (hk : k < (lays.getD d default).runs.size + (t.defs.getD d default).calls.size)
    (h1 : (region t lays d k).1 < (region t lays d k).2) : (region t lays d k).2 ≤ (lays.getD d default).size := by
  unfold layApart at h
  simp only [Bool.and_eq_true, List.all_eq_true, decide_eq_true_eq] at h
  exact h.1.2 _ (List.mem_filter.2 ⟨List.mem_map.2 ⟨k, List.mem_range.2 hk, rfl⟩, by simpa using h1⟩)

theorem layApart_place (h : layApart t lays d = true) {i : Nat} (hi : i < (t.defs.getD d default).calls.size) :
    0 < ((lays.getD d default).places.getD i default).ms ∧
      (((t.defs.getD d default).calls.getD i default).count ≤ 1 ∨
        (((lays.getD d default).places.getD i default).ms = 1 ∧
          (lays.getD ((t.defs.getD d default).calls.getD i default).def_ default).size ≤
            ((lays.getD d default).places.getD i default).cs) ∨
        (((lays.getD d default).places.getD i default).cs = 1 ∧
          ((t.defs.getD d default).calls.getD i default).count ≤ ((lays.getD d default).places.getD i default).ms)) := by
  unfold layApart at h
  simp only [Bool.and_eq_true, List.all_eq_true, List.mem_range] at h
  have := h.2 i hi
  simp only [placeOk, Bool.and_eq_true, Bool.or_eq_true, decide_eq_true_eq, beq_iff_eq] at this
  exact ⟨this.1, by rcases this.2 with (h | h) | h <;> simp_all⟩

end

/-! ## `posIn` is one-to-one below the layout's size -/

section
variable {t : Tmpl} {lays : Array Lay} {d : Nat}

/-- A column in run `r` sits in range `r`. -/
theorem run_at (hp : DefPos t lays d) {r y : Nat} (hr : r < (lays.getD d default).runs.size)
    (h1 : ((lays.getD d default).runs.getD r default).1 ≤ y) (h2 : y < ((lays.getD d default).runs.getD r default).2.1)
    (f : Nat) :
    posIn t lays (f + 1) d y = ((lays.getD d default).runs.getD r default).2.2 + (y - ((lays.getD d default).runs.getD r default).1) ∧
      (region t lays d r).1 ≤ posIn t lays (f + 1) d y ∧ posIn t lays (f + 1) d y < (region t lays d r).2 := by
  have e := posIn_own t lays hp hr (a := ((lays.getD d default).runs.getD r default).1)
    (b := ((lays.getD d default).runs.getD r default).2.1) (p := ((lays.getD d default).runs.getD r default).2.2) rfl h1 h2 f
  refine ⟨e, ?_⟩
  simp only [region, hr, ↓reduceIte, e]
  omega

/-- A column in call `i`'s copies sits in range `runs.size + i`, at copy `j`'s callee position `q`. -/
theorem call_at (hp : DefPos t lays d) {i y : Nat} (hi : i < (t.defs.getD d default).calls.size)
    (h1 : ((t.defs.getD d default).calls.getD i default).base ≤ y)
    (h2 : y < ((t.defs.getD d default).calls.getD i default).base + ((t.defs.getD d default).calls.getD i default).count *
        (t.defs.getD ((t.defs.getD d default).calls.getD i default).def_ default).cols) (f : Nat)
    (hq : posIn t lays f ((t.defs.getD d default).calls.getD i default).def_
        ((y - ((t.defs.getD d default).calls.getD i default).base) %
          (t.defs.getD ((t.defs.getD d default).calls.getD i default).def_ default).cols) <
      (lays.getD ((t.defs.getD d default).calls.getD i default).def_ default).size) :
    (region t lays d ((lays.getD d default).runs.size + i)).1 ≤ posIn t lays (f + 1) d y ∧
      posIn t lays (f + 1) d y < (region t lays d ((lays.getD d default).runs.size + i)).2 := by
  rw [posIn_call t lays hp hi h1 h2 f]
  unfold region
  simp only [Nat.not_lt.2 (Nat.le_add_right _ i), ↓reduceIte, Nat.add_sub_cancel_left]
  generalize (t.defs.getD d default).calls.getD i default = c at h1 h2 hq ⊢
  generalize (lays.getD d default).places.getD i default = pl
  generalize hLe : (t.defs.getD c.def_ default).cols = Le at h2 hq ⊢
  generalize posIn t lays f c.def_ ((y - c.base) % Le) = q at hq ⊢
  generalize (lays.getD c.def_ default).size = S at hq ⊢
  have hcnt : 0 < c.count * Le := by omega
  have hc0 : 0 < c.count := Nat.pos_of_ne_zero fun h => by simp [h] at hcnt
  have hL0 : 0 < Le := Nat.pos_of_ne_zero fun h => by simp [h] at hcnt
  have hj : (y - c.base) / Le < c.count := by
    rw [Nat.div_lt_iff_lt_mul hL0]; omega
  have e1 : (c.count == 0 || Le == 0) = false := by simp; omega
  rw [e1]
  simp only [Bool.false_eq_true, ↓reduceIte]
  have m1 : (y - c.base) / Le * pl.cs ≤ (c.count - 1) * pl.cs := Nat.mul_le_mul_right _ (by omega)
  have m2 : pl.ms * q ≤ pl.ms * (S - 1) := Nat.mul_le_mul_left _ (by omega)
  omega

/-- **`posIn` is one-to-one below the layout's size**, for layouts in order (`DefPos`) whose ranges are apart. -/
theorem posIn_ok (hpos : ∀ d < t.defs.size, DefPos t lays d) (hla : ∀ d < t.defs.size, layApart t lays d = true) :
    ∀ f d, d < f → d < t.defs.size →
      (∀ y < (t.defs.getD d default).cols, posIn t lays f d y < (lays.getD d default).size) ∧
      ∀ y1 y2, y1 < (t.defs.getD d default).cols → y2 < (t.defs.getD d default).cols →
        posIn t lays f d y1 = posIn t lays f d y2 → y1 = y2
  | 0, _, h, _ => absurd h (Nat.not_lt_zero _)
  | f + 1, d, hf, hd => by
    have hp := hpos d hd
    have ha := hla d hd
    have ih : ∀ i < (t.defs.getD d default).calls.size, _ := fun i hi =>
      posIn_ok hpos hla f ((t.defs.getD d default).calls.getD i default).def_ (by have := hp.callee i hi; omega)
        (by have := hp.callee i hi; omega)
    -- every column sits in a range of the layout: a run's, or a call's
    have hin : ∀ y < (t.defs.getD d default).cols, ∃ k, k < (lays.getD d default).runs.size + (t.defs.getD d default).calls.size ∧
        (region t lays d k).1 ≤ posIn t lays (f + 1) d y ∧ posIn t lays (f + 1) d y < (region t lays d k).2 := by
      intro y hy
      rcases hp.cover y hy with ⟨r, hr, h1, h2⟩ | ⟨i, hi, h1, h2⟩
      · exact ⟨r, by omega, (run_at hp hr h1 h2 f).2⟩
      · refine ⟨_, by omega, call_at hp hi h1 h2 f ((ih i hi).1 _ (Nat.mod_lt _ ?_))⟩
        have : 0 < ((t.defs.getD d default).calls.getD i default).count *
            (t.defs.getD ((t.defs.getD d default).calls.getD i default).def_ default).cols := by omega
        exact Nat.pos_of_ne_zero fun h => by rw [h, Nat.mul_zero] at this; exact Nat.lt_irrefl 0 this
    refine ⟨fun y hy => ?_, fun y1 y2 hy1 hy2 he => ?_⟩
    · obtain ⟨k, hk, h1, h2⟩ := hin y hy
      have := layApart_size ha hk (by omega)
      omega
    · have hsep : ∀ k1 k2, k1 < (lays.getD d default).runs.size + (t.defs.getD d default).calls.size →
          k2 < (lays.getD d default).runs.size + (t.defs.getD d default).calls.size → k1 ≠ k2 →
          (region t lays d k1).1 ≤ posIn t lays (f + 1) d y1 → posIn t lays (f + 1) d y1 < (region t lays d k1).2 →
          (region t lays d k2).1 ≤ posIn t lays (f + 1) d y1 → posIn t lays (f + 1) d y1 < (region t lays d k2).2 → False := by
        intro k1 k2 hk1 hk2 hne a1 a2 b1 b2
        have := layApart_apart ha hk1 hk2 hne (by omega) (by omega)
        unfold Apart at this
        omega
      have hpos' : ∀ i < (t.defs.getD d default).calls.size, ∀ y,
          ((t.defs.getD d default).calls.getD i default).base ≤ y →
          y < ((t.defs.getD d default).calls.getD i default).base + ((t.defs.getD d default).calls.getD i default).count *
            (t.defs.getD ((t.defs.getD d default).calls.getD i default).def_ default).cols →
          0 < (t.defs.getD ((t.defs.getD d default).calls.getD i default).def_ default).cols := by
        intro i hi y h1 h2
        have : 0 < ((t.defs.getD d default).calls.getD i default).count *
            (t.defs.getD ((t.defs.getD d default).calls.getD i default).def_ default).cols := by omega
        exact Nat.pos_of_ne_zero fun h => by rw [h, Nat.mul_zero] at this; exact Nat.lt_irrefl 0 this
      rcases hp.cover y1 hy1 with ⟨r1, hr1, a1, a2⟩ | ⟨i1, hi1, a1, a2⟩ <;>
        rcases hp.cover y2 hy2 with ⟨r2, hr2, b1, b2⟩ | ⟨i2, hi2, b1, b2⟩
      · obtain ⟨e1, m1, m2⟩ := run_at hp hr1 a1 a2 f
        obtain ⟨e2, n1, n2⟩ := run_at hp hr2 b1 b2 f
        by_cases hr : r1 = r2
        · subst hr; omega
        · exact (hsep r1 r2 (by omega) (by omega) hr m1 m2 (he ▸ n1) (he ▸ n2)).elim
      · obtain ⟨-, m1, m2⟩ := run_at hp hr1 a1 a2 f
        obtain ⟨n1, n2⟩ := call_at hp hi2 b1 b2 f ((ih i2 hi2).1 _ (Nat.mod_lt _ (hpos' i2 hi2 y2 b1 b2)))
        exact (hsep r1 _ (by omega) (by omega) (by omega) m1 m2 (he ▸ n1) (he ▸ n2)).elim
      · obtain ⟨m1, m2⟩ := call_at hp hi1 a1 a2 f ((ih i1 hi1).1 _ (Nat.mod_lt _ (hpos' i1 hi1 y1 a1 a2)))
        obtain ⟨-, n1, n2⟩ := run_at hp hr2 b1 b2 f
        exact (hsep _ r2 (by omega) (by omega) (by omega) m1 m2 (he ▸ n1) (he ▸ n2)).elim
      · have q1 := (ih i1 hi1).1 ((y1 - ((t.defs.getD d default).calls.getD i1 default).base) %
          (t.defs.getD ((t.defs.getD d default).calls.getD i1 default).def_ default).cols) (Nat.mod_lt _ (hpos' i1 hi1 y1 a1 a2))
        have q2 := (ih i2 hi2).1 ((y2 - ((t.defs.getD d default).calls.getD i2 default).base) %
          (t.defs.getD ((t.defs.getD d default).calls.getD i2 default).def_ default).cols) (Nat.mod_lt _ (hpos' i2 hi2 y2 b1 b2))
        by_cases hi : i1 = i2
        · subst hi
          have hL := hpos' i1 hi1 y1 a1 a2
          have hP := layApart_place ha hi1
          rw [posIn_call t lays hp hi1 a1 a2 f, posIn_call t lays hp hi1 b1 b2 f] at he
          rw [Nat.add_assoc, Nat.add_assoc] at he
          have he' := Nat.add_left_cancel he
          have hj : ∀ y, ((t.defs.getD d default).calls.getD i1 default).base ≤ y →
              y < ((t.defs.getD d default).calls.getD i1 default).base + ((t.defs.getD d default).calls.getD i1 default).count *
                (t.defs.getD ((t.defs.getD d default).calls.getD i1 default).def_ default).cols →
              (y - ((t.defs.getD d default).calls.getD i1 default).base) /
                (t.defs.getD ((t.defs.getD d default).calls.getD i1 default).def_ default).cols <
                ((t.defs.getD d default).calls.getD i1 default).count := by
            intro y h1 h2
            rw [Nat.div_lt_iff_lt_mul hL]; omega
          obtain ⟨ej, eq⟩ := place_inj hP (hj y1 a1 a2) (hj y2 b1 b2) q1 q2 he'
          have ex := (ih i1 hi1).2 _ _ (Nat.mod_lt _ hL) (Nat.mod_lt _ hL) eq
          have k1 := Nat.div_add_mod (y1 - ((t.defs.getD d default).calls.getD i1 default).base)
            (t.defs.getD ((t.defs.getD d default).calls.getD i1 default).def_ default).cols
          have k2 := Nat.div_add_mod (y2 - ((t.defs.getD d default).calls.getD i1 default).base)
            (t.defs.getD ((t.defs.getD d default).calls.getD i1 default).def_ default).cols
          rw [ej, ex] at k1
          omega
        · obtain ⟨m1, m2⟩ := call_at hp hi1 a1 a2 f q1
          obtain ⟨n1, n2⟩ := call_at hp hi2 b1 b2 f q2
          exact (hsep _ _ (by omega) (by omega) (by omega) m1 m2 (he ▸ n1) (he ▸ n2)).elim

end

/-! ## What `layoutM` checked -/

theorem forIn_all {α ε : Type} (P : α → Prop) {f : α → Unit → Except ε (ForInStep Unit)}
    (hf : ∀ a s t, f a s = .ok t → P a ∧ t = .yield ()) :
    ∀ (l : List α) {v : Unit}, forIn l () f = .ok v → ∀ a ∈ l, P a
  | [], _, _, a, ha => by cases ha
  | b :: l, v, h, a, ha => by
    rw [List.forIn_cons] at h
    obtain ⟨t, ht, h⟩ := ebo h
    obtain ⟨hb, rfl⟩ := hf b _ t ht
    rcases List.mem_cons.1 ha with rfl | ha
    · exact hb
    · exact forIn_all P hf l h a ha

theorem layoutM_facts {t : Tmpl} {mode : Mode} {lays : Array Lay} (h : layoutM t mode = .ok lays) :
    (∀ r < (lays.getD t.root default).runs.size, ((lays.getD t.root default).runs.getD r default).1 < t.inCols →
      ((lays.getD t.root default).runs.getD r default).1 = ((lays.getD t.root default).runs.getD r default).2.2) ∧
    (∀ i < (lays.getD t.root default).places.size, ((t.defs.getD t.root default).calls.getD i default).base < t.inCols →
      ((lays.getD t.root default).places.getD i default).nb = ((t.defs.getD t.root default).calls.getD i default).base ∧
        (lays.getD ((t.defs.getD t.root default).calls.getD i default).def_ default).ident = true) ∧
    (∀ g ∈ t.outGroups.toList, posOf t lays (g.col * WORD) % (g.words * WORD) = 0 ∧
      ∀ i < g.words * WORD, posOf t lays (g.col * WORD + i) = posOf t lays (g.col * WORD) + i) ∧
    posOf t lays ((t.defs.getD t.root default).cols - 1) + 1 = (lays.getD t.root default).size := by
  unfold layoutM at h
  obtain ⟨lays', h1, h⟩ := ebo h
  dsimp only at h
  split at h
  · cases h
  rename_i hin
  obtain ⟨u, hfor, h⟩ := ebo h
  split at h
  · cases h
  rename_i hk
  cases h
  simp only [Bool.not_eq_true', Bool.not_eq_false, Bool.and_eq_true, Array.all_eq_true, List.all_eq_true,
    List.mem_range] at hin
  generalize lays.getD t.root default = L at hin hk ⊢
  generalize t.defs.getD t.root default = D at hin hk ⊢
  refine ⟨fun r hr ha => ?_, fun i hi hb => ?_, fun g hg => ?_, by simpa using hk⟩
  · have := hin.1 r hr
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hr, Option.getD_some] at ha ⊢
    simp only [Bool.or_eq_true, decide_eq_true_eq, beq_iff_eq] at this
    rcases this with h | h
    · exact absurd h (by omega)
    · exact h
  · have := hin.2 i hi
    simp only [Bool.or_eq_true, decide_eq_true_eq, Bool.and_eq_true, beq_iff_eq] at this
    rcases this with h | h
    · exact absurd h (by omega)
    · exact h
  · rw [← Array.forIn_toList] at hfor
    have key := forIn_all (fun g : Group => posOf t lays (g.col * WORD) % (g.words * WORD) = 0 ∧
      ∀ i < g.words * WORD, posOf t lays (g.col * WORD + i) = posOf t lays (g.col * WORD) + i) (fun a s r hr => by
        split at hr
        · cases hr
        rename_i hc
        cases hr
        simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, List.any_eq_true, List.mem_range, not_or, not_exists, not_and,
          Decidable.not_not] at hc
        exact ⟨⟨hc.1, hc.2⟩, rfl⟩) _ hfor
    exact key g hg

/-! ## Layouts at their own columns -/

section
variable {t : Tmpl} {lays : Array Lay}

theorem identOk_parts (h : identOk t lays = true) :
    (∀ d < t.defs.size, (lays.getD d default).ident = true →
      (∀ r < (lays.getD d default).runs.size,
        ((lays.getD d default).runs.getD r default).1 = ((lays.getD d default).runs.getD r default).2.2) ∧
      ∀ i < (t.defs.getD d default).calls.size,
        ((lays.getD d default).places.getD i default).nb = ((t.defs.getD d default).calls.getD i default).base ∧
        ((lays.getD d default).places.getD i default).ms = 1 ∧
        ((lays.getD d default).places.getD i default).cs =
          (t.defs.getD ((t.defs.getD d default).calls.getD i default).def_ default).cols ∧
        (lays.getD ((t.defs.getD d default).calls.getD i default).def_ default).ident = true) ∧
    ∀ i < (t.defs.getD t.root default).calls.size, ((t.defs.getD t.root default).calls.getD i default).base < t.inCols →
      ((lays.getD t.root default).places.getD i default).ms = 1 ∧
      ((lays.getD t.root default).places.getD i default).cs =
        (t.defs.getD ((t.defs.getD t.root default).calls.getD i default).def_ default).cols := by
  unfold identOk at h
  simp only [Bool.and_eq_true, List.all_eq_true, List.mem_range] at h
  refine ⟨fun d hd hident => ?_, fun i hi hb => ?_⟩
  · have := h.1 d hd
    simp only [hident, Bool.not_true, Bool.false_or, Bool.and_eq_true, Array.all_eq_true, List.all_eq_true,
      List.mem_range, beq_iff_eq] at this
    refine ⟨fun r hr => ?_, fun i hi => by obtain ⟨⟨⟨a, b⟩, c⟩, e⟩ := this.2 i hi; exact ⟨a, b, c, e⟩⟩
    have := this.1 r hr
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hr, Option.getD_some]
    exact this
  · have := h.2 i hi
    simp only [Bool.or_eq_true, decide_eq_true_eq, Bool.and_eq_true, beq_iff_eq] at this
    rcases this with h | h
    · exact absurd h (by omega)
    · exact h

/-- **A layout `identOk` passes has every column at its own index.** -/
theorem posIn_ident (hpos : ∀ d < t.defs.size, DefPos t lays d) (hid : identOk t lays = true) :
    ∀ f d, d < f → d < t.defs.size → (lays.getD d default).ident = true →
      ∀ y < (t.defs.getD d default).cols, posIn t lays f d y = y
  | 0, _, h, _, _, _, _ => absurd h (Nat.not_lt_zero _)
  | f + 1, d, hf, hd, hident, y, hy => by
    have hp := hpos d hd
    obtain ⟨hruns, hcalls⟩ := (identOk_parts hid).1 d hd hident
    rcases hp.cover y hy with ⟨r, hr, h1, h2⟩ | ⟨i, hi, h1, h2⟩
    · rw [(run_at hp hr h1 h2 f).1, ← hruns r hr]
      omega
    · obtain ⟨hnb, hms, hcs, hcid⟩ := hcalls i hi
      have hc := hp.callee i hi
      have hL : 0 < (t.defs.getD ((t.defs.getD d default).calls.getD i default).def_ default).cols := by
        have : 0 < ((t.defs.getD d default).calls.getD i default).count *
            (t.defs.getD ((t.defs.getD d default).calls.getD i default).def_ default).cols := by omega
        exact Nat.pos_of_ne_zero fun h => by rw [h, Nat.mul_zero] at this; exact Nat.lt_irrefl 0 this
      rw [posIn_call t lays hp hi h1 h2 f, hnb, hms, hcs, Nat.one_mul,
        posIn_ident hpos hid f _ (by omega) (by omega) hcid _ (Nat.mod_lt _ hL), Nat.mul_comm, Nat.add_assoc,
        Nat.div_add_mod]
      omega

/-- **The input rows keep their columns** at `layoutM`'s layouts that `identOk` passes. -/
theorem posOf_input {mode : Mode} (hL : layoutM t mode = .ok lays) (hpos : ∀ d < t.defs.size, DefPos t lays d)
    (hid : identOk t lays = true) (hla : layApart t lays t.root = true) (hroot : t.root < t.defs.size) {i : Nat}
    (hi : i < t.inCols) (hc : i < (t.defs.getD t.root default).cols) : posOf t lays i = i := by
  have hp := hpos t.root hroot
  obtain ⟨hfr, hfc, -, -⟩ := layoutM_facts hL
  unfold posOf
  rcases hp.cover i hc with ⟨r, hr, h1, h2⟩ | ⟨k, hk, h1, h2⟩
  · rw [(run_at hp hr h1 h2 t.defs.size).1, ← hfr r hr (by omega)]
    omega
  · have hms0 := (layApart_place hla hk).1
    have hkp : k < (lays.getD t.root default).places.size := Nat.lt_of_not_le fun h => by
      rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_none h] at hms0
      exact Nat.lt_irrefl 0 hms0
    obtain ⟨hnb, hcid⟩ := hfc k hkp (by omega)
    obtain ⟨hms, hcs⟩ := (identOk_parts hid).2 k hk (by omega)
    have hcl := hp.callee k hk
    have hL : 0 < (t.defs.getD ((t.defs.getD t.root default).calls.getD k default).def_ default).cols := by
      have : 0 < ((t.defs.getD t.root default).calls.getD k default).count *
          (t.defs.getD ((t.defs.getD t.root default).calls.getD k default).def_ default).cols := by omega
      exact Nat.pos_of_ne_zero fun h => by rw [h, Nat.mul_zero] at this; exact Nat.lt_irrefl 0 this
    rw [posIn_call t lays hp hk h1 h2 t.defs.size, hnb, hms, hcs, Nat.one_mul,
      posIn_ident hpos hid t.defs.size _ (by omega) (by omega) hcid _ (Nat.mod_lt _ hL), Nat.mul_comm, Nat.add_assoc,
      Nat.div_add_mod]
    omega

end

/-! ## `posMap` is `posOf` -/

theorem posMap_size (t : Tmpl) (lays : Array Lay) : (posMap t lays).size = (t.defs.getD t.root default).cols := by
  unfold posMap
  rw [fillMap_writes, setAll_size, Array.size_replicate]

theorem posMap_getD {t : Tmpl} {lays : Array Lay} (hpos : ∀ d < t.defs.size, DefPos t lays d)
    (hroot : t.root < t.defs.size) {x : Nat} (hx : x < (t.defs.getD t.root default).cols) :
    (posMap t lays).getD x 0 = posOf t lays x := by
  have := (posMap_tree t lays hpos hroot).1 x hx
  rw [Nat.zero_add, Nat.zero_add, Nat.one_mul] at this
  exact this

/-! ## The committed net -/

/-- **A template that passes `unrollOk` at `layoutM`'s layouts has its committed net**: `Tmpl.unrollAt`, every check
`Net.ofRows` and `Net.ofRowsAt` run, succeeds. -/
theorem unrollAt_ok {t : Tmpl} {mode : Mode} {lays : Array Lay} (hL : layoutM t mode = .ok lays)
    (h : unrollOk t lays = true) : ∃ n, Tmpl.unrollAt t lays = .ok n := by
  unfold unrollOk at h
  simp only [Bool.and_eq_true, List.all_eq_true, List.mem_range] at h
  obtain ⟨⟨⟨⟨⟨⟨⟨hwf, ha⟩, ho⟩, hi⟩, hg⟩, hla⟩, hid⟩, hout⟩ := h
  have hpos := layouts_defPos (layoutM_layouts hL) (wf_call hwf) (wf_small hwf)
  have hroot := (wf_tmpl hwf).1
  have hk := Flock.TmplUnroll.wf_cols hwf
  have hic := Flock.TmplUnroll.groupsOk_inCols hg
  obtain ⟨acc, hr⟩ := rows_ok hwf ha
  obtain ⟨n, hn⟩ := Flock.TmplUnroll.ofRows_of_rows hwf ho hi hg hr
  have hinW : n.inWords * WORD = t.inCols := by
    rw [(Net.ofRows_ok hn).2.2.2.2.2.2.2.2.2.2.2.2.2.1]; rfl
  have hreads := Flock.TmplUnroll.rows_reads hwf ho hi hg hr
  obtain ⟨-, -, houts, hlast⟩ := layoutM_facts hL
  have hok := posIn_ok hpos hla (t.defs.size + 1) t.root (by omega) hroot
  have hpsize := posMap_size t lays
  have hpeq : ∀ x < (t.defs.getD t.root default).cols, (posMap t lays).getD x 0 = posOf t lays x :=
    fun x hx => posMap_getD hpos hroot hx
  have hlt : ∀ x < (t.defs.getD t.root default).cols, posOf t lays x < (lays.getD t.root default).size :=
    fun x hx => hok.1 x hx
  have hsize : (posMap t lays).getD t.constPos 0 + 1 = (lays.getD t.root default).size := by
    rw [hpeq _ (by omega)]
    rw [show t.constPos = (t.defs.getD t.root default).cols - 1 by omega]
    exact hlast
  have hinj : ∀ x y, x < (posMap t lays).size → y < (posMap t lays).size →
      (posMap t lays).getD x 0 = (posMap t lays).getD y 0 → x = y := by
    intro x y hx hy he
    rw [hpsize] at hx hy
    rw [hpeq x hx, hpeq y hy] at he
    exact hok.2 x y hx hy he
  obtain ⟨hinvs, hinv1, hinv2⟩ := invOf_spec hinj (size := (lays.getD t.root default).size)
  have hposOk : Net.posOk (t.defs.getD t.root default).cols (n.inWords * WORD) (lays.getD t.root default).size
      (posMap t lays) (Net.invOf (posMap t lays) (lays.getD t.root default).size) = true := by
    unfold Net.posOk
    simp only [Bool.and_eq_true, beq_iff_eq, List.all_eq_true, List.mem_range, decide_eq_true_eq]
    refine ⟨⟨⟨⟨hpsize, hinvs⟩, fun x hx => ?_⟩, fun p hp => ?_⟩, fun i hi' => ?_⟩
    · have h1 : (posMap t lays).getD x 0 < (lays.getD t.root default).size := by rw [hpeq x hx]; exact hlt x hx
      exact ⟨h1, by rw [hinv1 x (by omega) h1]⟩
    · split
      · rename_i x hx
        obtain ⟨h1, h2⟩ := hinv2 p x hx
        simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq]
        exact ⟨by omega, h2⟩
      · rfl
    · rw [hinW] at hi'
      rw [hpeq i (by omega), posOf_input hL hpos hid (hla t.root hroot) hroot hi' (by omega)]
  -- each output group inside the columns, of at least one word, at its committed positions in order
  have hgw : ∀ g ∈ t.outGroups.toList, 0 < g.words ∧
      g.col * WORD + g.words * WORD ≤ (t.defs.getD t.root default).cols ∧
      posOf t lays (g.col * WORD) / WORD + g.words ≤ 2 ^ max 7 (log2ceil (lays.getD t.root default).size) / WORD := by
    intro g hg'
    have hm := Array.mem_toList_iff.1 hg'
    unfold Tmpl.groupsOk at hg
    simp only [Bool.and_eq_true, Bool.not_eq_true', Array.any_eq_false', Array.mem_append] at hg
    have h1 := hg.1.1.1.1 g (Or.inr hm)
    simp only [Bool.not_eq_false, Bool.and_eq_true, isPow2, decide_eq_true_eq] at h1
    unfold outsAt at hout
    simp only [Array.all_eq_true', Bool.and_eq_true, decide_eq_true_eq] at hout
    exact ⟨h1.1.1, (hout g hm).1, (hout g hm).2⟩
  have houtsOk : Net.outsOk (posMap t lays) t.outGroups = true := by
    unfold Net.outsOk
    simp only [Array.all_eq_true', Bool.and_eq_true, beq_iff_eq, List.all_eq_true, List.mem_range]
    intro g hm
    obtain ⟨hw, hc, -⟩ := hgw g (Array.mem_toList_iff.2 hm)
    have hW : 0 < g.words * WORD := Nat.mul_pos hw (by decide)
    obtain ⟨h1, h2⟩ := houts g (Array.mem_toList_iff.2 hm)
    refine ⟨by rw [hpeq _ (by omega)]; exact h1, fun i hi' => ?_⟩
    rw [hpeq _ (by omega), hpeq _ (by omega)]
    exact h2 i hi'
  have hpow : (lays.getD t.root default).size ≤ 2 ^ max 7 (log2ceil (lays.getD t.root default).size) :=
    Nat.le_trans (Flock.TmplUnroll.log2ceil_le _) (Nat.pow_le_pow_right (by omega) (Nat.le_max_right _ _))
  have hplaced : ∀ rows : Array (Array Nat), (∀ i, ∀ c ∈ rows.getD i #[], c ≤ t.constPos) →
      ∀ r ∈ Net.placed (posMap t lays) (Net.invOf (posMap t lays) (lays.getD t.root default).size) rows, ∀ c ∈ r,
        c < 2 ^ max 7 (log2ceil (lays.getD t.root default).size) := by
    intro rows hrows r hr c hc
    unfold Net.placed at hr
    obtain ⟨o, -, rfl⟩ := Array.mem_map.1 hr
    cases o with
    | none => simp at hc
    | some x =>
      obtain ⟨v, hv, rfl⟩ := Array.mem_map.1 hc
      have := hrows x v hv
      rw [hpeq v (by omega)]
      exact Nat.lt_of_lt_of_le (hlt v (by omega)) hpow
  obtain ⟨sa, hsa⟩ := Flock.TmplUnroll.sparse_ofRows_ok (hplaced acc.ra fun i => (hreads i).1)
  obtain ⟨sb, hsb⟩ := Flock.TmplUnroll.sparse_ofRows_ok (hplaced acc.rb fun i => (hreads i).2)
  have hd11 : ((t.outGroups.map (Net.Group.at (posMap t lays))).any fun g =>
      decide (g.col + g.words > 2 ^ max 7 (log2ceil (lays.getD t.root default).size) / WORD)) = false := by
    rw [Array.any_eq_false']
    intro g' hg'
    obtain ⟨g, hm, rfl⟩ := Array.mem_map.1 hg'
    obtain ⟨hw, hc, h3⟩ := hgw g (Array.mem_toList_iff.2 hm)
    have hW : 0 < g.words * WORD := Nat.mul_pos hw (by decide)
    have e := hpeq (g.col * WORD) (by omega)
    simp only [Net.Group.at, e, decide_eq_true_eq, gt_iff_lt, Nat.not_lt]
    exact h3
  unfold Tmpl.unrollAt Net.ofRowsAt
  simp only [hr, hn, bind, Except.bind, pure, Except.pure, hsize, hposOk, houtsOk, hsa, hsb, hd11, Bool.not_true,
    Bool.false_eq_true, ↓reduceIte]
  exact ⟨_, rfl⟩

/-! ## The net the verifier carries -/

/-- `ofRows` names its net as asked. -/
theorem ofRows_name {name : String} {useful constPos : Nat} {ig og : Array Group} {ra rb : Array (Array Nat)} {n : Net}
    (h : Net.ofRows name useful constPos ig og ra rb = .ok n) : n.name = name := by
  unfold Net.ofRows at h
  simp only [bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw] at h
  repeat' (first | split at h | (injection h with h; subst h))
  all_goals first
    | exact absurd h (by simp)
    | rfl

/-- Two nets with the same fields, their A and B read, are one. -/
theorem net_ext {n m : Net} (h1 : n.name = m.name) (h2 : n.useful = m.useful) (h3 : n.constPos = m.constPos)
    (h4 : n.inGroups = m.inGroups) (h5 : n.outGroups = m.outGroups) (h6 : n.inWords = m.inWords) (h7 : n.a = m.a)
    (h8 : n.b = m.b) (h9 : n.unitLog = m.unitLog) (h10 : n.lookup = m.lookup) : n = m := by
  have e7 : n.aT = m.aT := Thunk.ext h7
  have e8 : n.bT = m.bT := Thunk.ext h8
  cases n; cases m
  simp only at h1 h2 h3 h4 h5 h6 h9 h10 e7 e8
  subst h1 h2 h3 h4 h5 h6 h9 h10 e7 e8
  rfl

/-- **The net the verifier carries is the committed net**: wherever `unrollOk` passes at `layoutM`'s layouts,
`Tmpl.unitNet`, its header from the description and its rows built only when read, is `unrollAt`'s net. -/
theorem unitNet_eq {t : Tmpl} {mode : Mode} {lays : Array Lay} (hL : layoutM t mode = .ok lays)
    (h : unrollOk t lays = true) : Tmpl.unrollAt t lays = .ok (unitNet t lays) := by
  obtain ⟨n, hn⟩ := unrollAt_ok hL h
  rw [hn]
  congr 1
  have hu := hn
  unfold Tmpl.unrollAt at hu
  obtain ⟨acc, -, hat⟩ := ebo hu
  obtain ⟨n0, hn0, huse, hcp, hlog, -, -, hname, hig, hiw, hlk, hog, -⟩ := Net.ofRowsAt_ok hat
  have h0 := Net.ofRows_ok hn0
  unfold unrollOk at h
  simp only [Bool.and_eq_true, List.all_eq_true, List.mem_range] at h
  obtain ⟨⟨⟨⟨⟨⟨⟨hwf, -⟩, -⟩, -⟩, hg⟩, -⟩, -⟩, hout⟩ := h
  have hpos := layouts_defPos (layoutM_layouts hL) (wf_call hwf) (wf_small hwf)
  have hroot := (wf_tmpl hwf).1
  have hk := Flock.TmplUnroll.wf_cols hwf
  obtain ⟨-, -, -, hlast⟩ := layoutM_facts hL
  have hpeq : ∀ x < (t.defs.getD t.root default).cols, (posMap t lays).getD x 0 = posOf t lays x :=
    fun x hx => posMap_getD hpos hroot hx
  have hsize : (posMap t lays).getD t.constPos 0 + 1 = (lays.getD t.root default).size := by
    rw [hpeq _ (by omega)]
    rw [show t.constPos = (t.defs.getD t.root default).cols - 1 by omega]
    exact hlast
  have hfull : unitFull t lays = n := by unfold unitFull; rw [hn]
  apply net_ext
  · rw [hname, ofRows_name hn0]; rfl
  · rw [huse, hsize]; rfl
  · show n.constPos = (lays.getD t.root default).size - 1
    omega
  · rw [hig, h0.2.2.2.2.2.1]; rfl
  · rw [hog]
    show _ = t.outGroups.map fun g => { g with col := posOf t lays (g.col * WORD) / WORD }
    apply Array.map_congr_left
    intro g hm
    unfold Tmpl.groupsOk at hg
    simp only [Bool.and_eq_true, Bool.not_eq_true', Array.any_eq_false', Array.mem_append] at hg
    have h1 := hg.1.1.1.1 g (Or.inr hm)
    simp only [Bool.not_eq_false, Bool.and_eq_true, isPow2, decide_eq_true_eq] at h1
    unfold outsAt at hout
    simp only [Array.all_eq_true', Bool.and_eq_true, decide_eq_true_eq] at hout
    have hW : 0 < g.words * WORD := Nat.mul_pos h1.1.1 (by decide)
    simp only [Net.Group.at, hpeq (g.col * WORD) (by have := (hout g hm).1; omega)]
  · rw [hiw, h0.2.2.2.2.2.2.2.2.2.2.2.2.2.1]; rfl
  · show n.a = (unitFull t lays).a
    rw [hfull]
  · show n.b = (unitFull t lays).b
    rw [hfull]
  · rw [hlog, hsize]; rfl
  · rw [hlk, h0.2.2.2.2.2.2.2.2.2.2.2.2.1]; rfl

end FlockSoundness.TmplFold

namespace Flock

/-- **A read templated unit's net is its template's committed net** (`FlockSoundness.TmplFold.unitNet_eq`). -/
theorem TmplUnit.unrolled (tu : TmplUnit) : Tmpl.unrollAt tu.t tu.lays = .ok tu.net := by
  rw [tu.isNet]
  exact FlockSoundness.TmplFold.unitNet_eq tu.laid tu.ok

end Flock
