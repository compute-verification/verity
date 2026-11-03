/-
A templated unit's input rows, checked on its description. `Net.ofRows` walks the unrolled input rows twice:
each is a self row `A = B = [col]` or empty, and each input port bit has its self row (`Net.inputsOk`). `Tmpl.inputsT`
checks both once per Definition and once over the root's copies (O(description), no copy unrolled): a Definition that
writes only self rows wherever it is copied (`Def.selfOnly`), one whose self rows and runs of such copies side by side
cover every column it has (`Def.full`), the root's input rows written only by self rows and copies of the first kind,
and each input group's port bits covered by the root's self rows and runs of copies of the second. `rows_inputRows` and
`rows_inputsOk` prove each gives its unrolled check, and `rows_order` the order `Net.checkOrder` checks with
`Tmpl.ordered` alone (`Flock.TmplOrder.rows_ordered` without its `inputsOk` hypothesis).

A column once written keeps its rows: `write` refuses a written column. So a self row, once written, is the final one
(`Self`, `place_keep`), and a cover of a Definition's columns by self rows and full copies (`reach_mem`) leaves every one
of them a self row (`place_full`).
-/
import Flock.TmplOrder

namespace Flock.Tmpl

/-- Row `r` of a Definition of `cols` columns is a self row: `A = B = [col]`, inside its columns. -/
def selfRow (cols : Nat) (r : Nat × Array Nat × Array Nat) : Bool :=
  decide (r.1 < cols) && r.2.1 == #[r.1] && r.2.2 == #[r.1]

/-- Definition `df` writes only self rows, wherever it is copied (`S`: the Definitions that do). -/
def Def.selfOnly (S : Array Bool) (df : Def) : Bool :=
  df.rows.toList.all (selfRow df.cols) && df.calls.toList.all fun c => S.getD c.def_ false

/-- Which Definitions write only self rows, each from the earlier ones. -/
def selfFlags (defs : Array Def) : Array Bool := defs.foldl (fun S df => S.push (df.selfOnly S)) #[]

/-- The column ranges `[lo, hi)` of `df`'s self rows and of its runs of copies, side by side, of Definitions in `Fl`. -/
def Def.selfSpans (defs : Array Def) (Fl : Array Bool) (df : Def) : List (Nat × Nat) :=
  (df.rows.toList.filter (selfRow df.cols)).map (fun r => (r.1, r.1 + 1)) ++
    (df.calls.toList.filter fun c => Fl.getD c.def_ false &&
        (decide (c.count ≤ 1) || c.stride == (defs.getD c.def_ default).cols)).map
      fun c => (c.base, c.base + c.count * (defs.getD c.def_ default).cols)

/-- How far from `lo` the ranges reach, each taken in turn when it holds the point reached so far. -/
def reach (spans : List (Nat × Nat)) (lo : Nat) : Nat :=
  spans.foldl (fun cur s => if s.1 ≤ cur && cur < s.2 then s.2 else cur) lo

/-- The ranges cover `[lo, hi)`, taken in order of their first column. -/
def covers (spans : List (Nat × Nat)) (lo hi : Nat) : Bool :=
  decide (hi ≤ reach (spans.mergeSort fun a b => decide (a.1 ≤ b.1)) lo)

/-- Definition `df`'s self rows and runs of copies of Definitions in `Fl` cover its columns. -/
def Def.full (defs : Array Def) (Fl : Array Bool) (df : Def) : Bool := covers (df.selfSpans defs Fl) 0 df.cols

/-- Which Definitions are full, each from the earlier ones. -/
def fullFlags (defs : Array Def) : Array Bool := defs.foldl (fun Fl df => Fl.push (df.full defs Fl)) #[]

/-- **The unrolled input rows are self rows or empty, and each input port bit has its self row**, checked on the
description (`Flock.Tmpl.rows_inputRows`, `rows_inputsOk`): the flags are each Definition's, the root's rows in the input
rows are self rows and its copies there write only self rows, the input groups ascend, and each one's port bits are
covered by the root's self rows and runs of full copies. -/
def Tmpl.inputsT (t : Tmpl) : Bool :=
  let S := selfFlags t.defs
  let Fl := fullFlags t.defs
  let df := t.defs.getD t.root default
  let ic := t.inCols
  let spans := (df.selfSpans t.defs Fl).filter (·.1 < ic)
  (List.range t.defs.size).all (fun d => (!S.getD d false || (t.defs.getD d default).selfOnly S) &&
      (!Fl.getD d false || (t.defs.getD d default).full t.defs Fl)) &&
    df.rows.toList.all (fun r => decide (ic ≤ r.1) || selfRow df.cols r) &&
    df.calls.toList.all (fun c => decide (ic ≤ c.base) || S.getD c.def_ false) &&
    (List.range (t.inGroups.size - 1)).all (fun i => (t.inGroups.getD i default).col + (t.inGroups.getD i default).words ≤
      (t.inGroups.getD (i + 1) default).col) &&
    t.inGroups.all fun g => covers spans (g.col * WORD) (g.col * WORD + g.bits.foldl (· + ·) 0)

end Flock.Tmpl

namespace Flock.TmplRows

open Flock.Tmpl Flock.TmplOrder

/-! ## Arrays, `write`, folds -/

theorem getD_set!_ne {α : Type} (xs : Array α) {c x : Nat} (v d : α) (h : x ≠ c) :
    (xs.set! c v).getD x d = xs.getD x d := by
  simp [Array.getD_eq_getD_getElem?, Array.set!_eq_setIfInBounds, Ne.symm h]

theorem getD_set!_self {α : Type} (xs : Array α) {c : Nat} (v d : α) (h : c < xs.size) : (xs.set! c v).getD c d = v := by
  simp [Array.getD_eq_getD_getElem?, Array.set!_eq_setIfInBounds, h]

theorem lt_of_getD_false {xs : Array Bool} {c : Nat} (h : xs.getD c true = false) : c < xs.size := by
  rcases Nat.lt_or_ge c xs.size with h' | h'
  · exact h'
  · simp [Array.getD_eq_getD_getElem?, Array.getElem?_eq_none h'] at h

theorem write_eq {acc acc' : Acc} {c : Nat} {a b : Array Nat} (h : write acc c a b = .ok acc') :
    acc.set.getD c true = false ∧ acc' = { ra := acc.ra.set! c a, rb := acc.rb.set! c b, set := acc.set.set! c true } := by
  unfold write at h
  split at h
  · cases h
  · rename_i hn; cases h; exact ⟨Bool.eq_false_iff.2 hn, rfl⟩

/-- Every column has its rows and its flag. -/
abbrev Sized (acc : Acc) : Prop := acc.ra.size = acc.set.size ∧ acc.rb.size = acc.set.size

/-- Column `x` is written, with its self row. -/
abbrev Self (x : Nat) (acc : Acc) : Prop :=
  acc.set.getD x true = true ∧ acc.ra.getD x #[] = #[x] ∧ acc.rb.getD x #[] = #[x]

/-- Column `x` holds a self row, or none. -/
abbrev SelfOrEmpty (x : Nat) (acc : Acc) : Prop :=
  (acc.ra.getD x #[] = #[x] ∧ acc.rb.getD x #[] = #[x]) ∨ (acc.ra.getD x #[] = #[] ∧ acc.rb.getD x #[] = #[])

theorem write_sized {acc acc' : Acc} {c : Nat} {a b : Array Nat} (h : write acc c a b = .ok acc') (hs : Sized acc) :
    Sized acc' := by
  obtain ⟨-, rfl⟩ := write_eq h
  simp only [Sized, Array.set!_eq_setIfInBounds, Array.size_setIfInBounds] at hs ⊢
  exact hs

theorem write_keep {acc acc' : Acc} {c x : Nat} {a b : Array Nat} (h : write acc c a b = .ok acc') (hx : Self x acc) :
    Self x acc' := by
  obtain ⟨hc, rfl⟩ := write_eq h
  have hne : x ≠ c := fun e => by subst e; rw [hx.1] at hc; cases hc
  exact ⟨by rw [getD_set!_ne _ _ _ hne]; exact hx.1, by rw [getD_set!_ne _ _ _ hne]; exact hx.2.1,
    by rw [getD_set!_ne _ _ _ hne]; exact hx.2.2⟩

theorem write_self {acc acc' : Acc} {c : Nat} (hs : Sized acc) (h : write acc c #[c] #[c] = .ok acc') : Self c acc' := by
  obtain ⟨hc, rfl⟩ := write_eq h
  have hlt := lt_of_getD_false hc
  exact ⟨getD_set!_self _ _ _ hlt, getD_set!_self _ _ _ (by rw [hs.1]; exact hlt),
    getD_set!_self _ _ _ (by rw [hs.2]; exact hlt)⟩

/-- A write of a self row keeps every column a self row or empty. -/
theorem write_selfOrEmpty {acc acc' : Acc} {c x : Nat} (hs : Sized acc) (h : write acc c #[c] #[c] = .ok acc')
    (hx : SelfOrEmpty x acc) : SelfOrEmpty x acc' := by
  by_cases e : x = c
  · subst e; exact Or.inl (write_self hs h).2
  · obtain ⟨-, rfl⟩ := write_eq h
    simp only [SelfOrEmpty, getD_set!_ne _ _ _ e]
    exact hx

/-- A write elsewhere keeps column `x`'s rows. -/
theorem write_same {acc acc' : Acc} {c x : Nat} {a b : Array Nat} (h : write acc c a b = .ok acc') (e : x ≠ c) :
    Same x acc' acc := by
  obtain ⟨-, rfl⟩ := write_eq h
  exact ⟨getD_set!_ne _ _ _ e, getD_set!_ne _ _ _ e⟩

theorem same_selfOrEmpty {acc acc' : Acc} {x : Nat} (hs : Same x acc' acc) (h : SelfOrEmpty x acc) :
    SelfOrEmpty x acc' := by
  unfold SelfOrEmpty; rw [hs.1, hs.2]; exact h

/-- What each step makes for its own element, and every later step keeps, holds at the fold's end for every element; `I`
holds throughout. -/
theorem foldlM_mem {α σ : Type} (I : σ → Prop) (Q : α → σ → Prop) {f : σ → α → Except String σ}
    (hI : ∀ a u u', I u → f u a = .ok u' → I u')
    (hkeep : ∀ a b u u', I u → f u b = .ok u' → Q a u → Q a u')
    (hmake : ∀ a u u', I u → f u a = .ok u' → Q a u') :
    ∀ {l : List α} {s s' : σ}, I s → l.foldlM f s = .ok s' → I s' ∧ ∀ a ∈ l, Q a s'
  | [], _, _, hs, h => by cases h; exact ⟨hs, fun _ h => by cases h⟩
  | a :: l, s, s', hs, h => by
    rw [List.foldlM_cons] at h
    obtain ⟨s1, h1, h2⟩ := bind_ok h
    have hs1 := hI a s s1 hs h1
    obtain ⟨hs', hl⟩ := foldlM_mem I Q hI hkeep hmake hs1 h2
    refine ⟨hs', fun b hb => ?_⟩
    rcases List.mem_cons.1 hb with e | hb
    · subst e
      have hq := hmake b s s1 hs h1
      exact (foldlM_inv (fun u => I u ∧ Q b u) (fun c _ u u' hu hw => ⟨hI c u u' hu.1 hw, hkeep b c u u' hu.1 hw hu.2⟩)
        ⟨hs1, hq⟩ h2).2
    · exact hl b hb

theorem self_canon {df : Def} {o k c : Nat} {inp : List Nat} (hc : c < df.cols) :
    Flock.Tmpl.canon (#[c].map (df.ref o inp k)) = #[o + c] := by
  simp [Flock.Tmpl.canon, dropPairs, Def.ref, hc]

/-! ## Copies -/

/-- **A copy keeps every column's flag and its size.** -/
theorem place_sized {defs : Array Def} {k : Nat} (fuel : Nat) :
    ∀ {d o inp acc acc'}, place defs k fuel d o inp acc = .ok acc' → Sized acc → Sized acc' := by
  induction fuel with
  | zero => intro d o inp acc acc' h; cases h
  | succ fuel ih =>
    intro d o inp acc acc' h hs
    simp only [place] at h
    cases hdf : defs[d]? with
    | none => simp only [hdf] at h; cases h
    | some df =>
      simp only [hdf] at h
      obtain ⟨acc1, h1, h2⟩ := bind_ok h
      have s1 := foldlM_inv Sized (fun _ _ u u' hu hw => write_sized hw hu) hs h1
      exact foldlM_inv Sized (fun c _ u u' hu hw =>
        foldlM_inv Sized (fun _ _ w w' hw0 hp => ih hp hw0) hu hw) s1 h2

/-- **A copy writes nothing before its first column.** -/
theorem place_above {defs : Array Def} {k : Nat} (fuel : Nat) :
    ∀ {d o inp acc acc'}, place defs k fuel d o inp acc = .ok acc' → ∀ x < o, Same x acc' acc := by
  induction fuel with
  | zero => intro d o inp acc acc' h; cases h
  | succ fuel ih =>
    intro d o inp acc acc' h x hx
    simp only [place] at h
    cases hdf : defs[d]? with
    | none => simp only [hdf] at h; cases h
    | some df =>
      simp only [hdf] at h
      obtain ⟨acc1, h1, h2⟩ := bind_ok h
      have s1 : Same x acc1 acc := foldlM_inv (fun a => Same x a acc) (fun _ _ u u' hu hw =>
        let e := write_same hw (by omega); ⟨e.1.trans hu.1, e.2.trans hu.2⟩) ⟨rfl, rfl⟩ h1
      have s2 : Same x acc' acc1 := foldlM_inv (fun a => Same x a acc1) (fun _ _ u u' hu hw =>
        foldlM_inv (fun a => Same x a acc1) (fun _ _ w w' hw0 hp =>
          let e := ih hp x (by omega); ⟨e.1.trans hw0.1, e.2.trans hw0.2⟩) hu hw) ⟨rfl, rfl⟩ h2
      exact ⟨s2.1.trans s1.1, s2.2.trans s1.2⟩

/-- **A copy keeps every self row written before it.** -/
theorem place_keep {defs : Array Def} {k x : Nat} (fuel : Nat) :
    ∀ {d o inp acc acc'}, place defs k fuel d o inp acc = .ok acc' → Self x acc → Self x acc' := by
  induction fuel with
  | zero => intro d o inp acc acc' h; cases h
  | succ fuel ih =>
    intro d o inp acc acc' h hs
    simp only [place] at h
    cases hdf : defs[d]? with
    | none => simp only [hdf] at h; cases h
    | some df =>
      simp only [hdf] at h
      obtain ⟨acc1, h1, h2⟩ := bind_ok h
      have s1 := foldlM_inv (Self x) (fun _ _ u u' hu hw => write_keep hw hu) hs h1
      exact foldlM_inv (Self x) (fun c _ u u' hu hw =>
        foldlM_inv (Self x) (fun _ _ w w' hw0 hp => ih hp hw0) hu hw) s1 h2

/-- **A copy of a Definition that writes only self rows keeps every column a self row or empty.** -/
theorem place_selfOnly {defs : Array Def} {k : Nat} {S : Array Bool}
    (hS : ∀ (d : Nat) (df : Def), defs[d]? = some df → S.getD d false = true → df.selfOnly S = true) (fuel : Nat) :
    ∀ {d o inp acc acc'}, place defs k fuel d o inp acc = .ok acc' → S.getD d false = true → Sized acc →
      ∀ x, SelfOrEmpty x acc → SelfOrEmpty x acc' := by
  induction fuel with
  | zero => intro d o inp acc acc' h; cases h
  | succ fuel ih =>
    intro d o inp acc acc' h hd hs x hx
    simp only [place] at h
    cases hdf : defs[d]? with
    | none => simp only [hdf] at h; cases h
    | some df =>
      simp only [hdf] at h
      have ho := hS d df hdf hd
      unfold Def.selfOnly at ho
      simp only [Bool.and_eq_true, List.all_eq_true] at ho
      obtain ⟨hrows, hcalls⟩ := ho
      obtain ⟨acc1, h1, h2⟩ := bind_ok h
      have s1 : Sized acc1 ∧ SelfOrEmpty x acc1 :=
        foldlM_inv (fun u => Sized u ∧ SelfOrEmpty x u) (fun r hr u u' hu hw => by
          have hr' := hrows r hr
          unfold selfRow at hr'
          simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hr'
          obtain ⟨⟨hc, ha⟩, hb⟩ := hr'
          rw [ha, hb, self_canon hc] at hw
          exact ⟨write_sized hw hu.1, write_selfOrEmpty hu.1 hw hu.2⟩) ⟨hs, hx⟩ h1
      exact (foldlM_inv (fun u => Sized u ∧ SelfOrEmpty x u) (fun c hc u u' hu hw =>
        foldlM_inv (fun u => Sized u ∧ SelfOrEmpty x u) (fun _ _ w w' hw0 hp =>
          ⟨place_sized _ hp hw0.1, ih hp (hcalls c hc) hw0.1 x hw0.2⟩) hu hw) s1 h2).2

/-- A run of copies side by side: copy `j < count` at `base + j · stride`, `stride` its width or a single copy. Column
`y` of the run is in copy `y / width`. -/
theorem run_copy {count stride w base y : Nat} (hrun : count ≤ 1 ∨ stride = w) (h1 : base ≤ y)
    (h2 : y < base + count * w) :
    ∃ j < count, base + j * stride ≤ y ∧ y < base + j * stride + w := by
  have hw : 0 < w := by
    rcases Nat.eq_zero_or_pos w with e | e
    · subst e; omega
    · exact e
  refine ⟨(y - base) / w, ?_, ?_⟩
  · exact (Nat.div_lt_iff_lt_mul hw).2 (by omega)
  · have hd := Nat.div_add_mod (y - base) w
    have hm := Nat.mod_lt (y - base) hw
    rcases hrun with hc | e
    · obtain rfl | rfl : count = 0 ∨ count = 1 := by omega
      · omega
      · have hj : (y - base) / w = 0 := Nat.div_eq_of_lt (by omega)
        rw [hj]; omega
    · subst e; rw [Nat.mul_comm ((y - base) / stride) stride]; omega

/-- Every column the ranges reach from `lo` is in one of them. -/
theorem reach_mem : ∀ {spans : List (Nat × Nat)} {lo x : Nat}, lo ≤ x → x < reach spans lo →
    ∃ s ∈ spans, s.1 ≤ x ∧ x < s.2
  | [], lo, x, h1, h2 => by simp only [reach, List.foldl_nil] at h2; omega
  | s :: rest, lo, x, h1, h2 => by
    simp only [reach, List.foldl_cons] at h2
    have step : ∀ lo', lo' = (if s.1 ≤ lo && lo < s.2 then s.2 else lo) → ∃ s' ∈ s :: rest, s'.1 ≤ x ∧ x < s'.2 := by
      intro lo' e
      by_cases hx : lo' ≤ x
      · obtain ⟨s', hs', h⟩ := reach_mem (spans := rest) hx (by rw [e]; exact h2)
        exact ⟨s', List.mem_cons.2 (Or.inr hs'), h⟩
      · split at e
        · rename_i hs
          simp only [Bool.and_eq_true, decide_eq_true_eq] at hs
          exact ⟨s, List.mem_cons.2 (Or.inl rfl), by omega, by omega⟩
        · omega
    exact step _ rfl

theorem covers_mem {spans : List (Nat × Nat)} {lo hi x : Nat} (h : covers spans lo hi = true) (h1 : lo ≤ x)
    (h2 : x < hi) : ∃ s ∈ spans, s.1 ≤ x ∧ x < s.2 := by
  unfold covers at h
  obtain ⟨s, hs, h⟩ := reach_mem h1 (Nat.lt_of_lt_of_le h2 (of_decide_eq_true h))
  exact ⟨s, List.mem_mergeSort.1 hs, h⟩

/-- **A copy writes a self row at every column of its self rows and of its runs of full copies**, given that a full copy
writes one at each of its columns. -/
theorem place_spans {defs : Array Def} {k : Nat} {Fl : Array Bool} {fuel : Nat}
    (ih : ∀ {d o inp acc acc'}, place defs k fuel d o inp acc = .ok acc' → Fl.getD d false = true → Sized acc →
      ∀ x, o ≤ x → x < o + (defs.getD d default).cols → Self x acc')
    {d o : Nat} {inp : List Nat} {acc acc' : Acc} {df : Def} (hdf : defs[d]? = some df)
    (h : place defs k (fuel + 1) d o inp acc = .ok acc') (hs : Sized acc) :
    ∀ s ∈ df.selfSpans defs Fl, ∀ x, o + s.1 ≤ x → x < o + s.2 → Self x acc' := by
  simp only [place, hdf] at h
  obtain ⟨acc1, h1, h2⟩ := bind_ok h
  let fr := fun (acc : Acc) (r : Nat × Array Nat × Array Nat) =>
    write acc (o + r.1) (Flock.Tmpl.canon (r.2.1.map (df.ref o inp k))) (Flock.Tmpl.canon (r.2.2.map (df.ref o inp k)))
  have r1 := foldlM_mem (f := fr) Sized (fun r acc => selfRow df.cols r = true → Self (o + r.1) acc)
    (fun _ _ _ hu hw => write_sized hw hu) (fun _ _ _ _ _ hw hq hr => write_keep hw (hq hr))
    (fun r u u' hu hw hr => by
      unfold selfRow at hr
      simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hr
      obtain ⟨⟨hc, ha⟩, hb⟩ := hr
      simp only [fr] at hw
      rw [ha, hb, self_canon hc] at hw
      exact write_self hu hw) hs h1
  let fc := fun (acc : Acc) (c : Call) => (List.range c.count).foldlM (fun acc j =>
    place defs k fuel c.def_ (o + c.base + j * c.stride) (c.bound (df.ref o inp k) j) acc) acc
  have keepC : ∀ c u u', fc u c = .ok u' → ∀ x, Self x u → Self x u' := fun c u u' hw x hx =>
    foldlM_inv (Self x) (fun _ _ w w' hw0 hp => place_keep _ hp hw0) hx hw
  have sizedC : ∀ c u u', Sized u → fc u c = .ok u' → Sized u' := fun c u u' hu hw =>
    foldlM_inv Sized (fun _ _ w w' hw0 hp => place_sized _ hp hw0) hu hw
  have r2 := foldlM_mem (f := fc) Sized
    (fun c acc => Fl.getD c.def_ false = true → (c.count ≤ 1 ∨ c.stride = (defs.getD c.def_ default).cols) →
      ∀ x, o + c.base ≤ x → x < o + c.base + c.count * (defs.getD c.def_ default).cols → Self x acc)
    (fun c u u' hu hw => sizedC c u u' hu hw)
    (fun _ b u u' _ hw hq hF hrun x h1 h2 => keepC b u u' hw x (hq hF hrun x h1 h2))
    (fun c u u' hu hw hF hrun x hx1 hx2 => by
      obtain ⟨j, hj, hj1, hj2⟩ := run_copy hrun hx1 hx2
      have := foldlM_mem (f := fun acc j =>
          place defs k fuel c.def_ (o + c.base + j * c.stride) (c.bound (df.ref o inp k) j) acc) Sized
        (fun j acc => ∀ x, o + c.base + j * c.stride ≤ x → x < o + c.base + j * c.stride + (defs.getD c.def_ default).cols →
          Self x acc)
        (fun _ _ _ hu hp => place_sized _ hp hu) (fun _ _ _ _ _ hp hq x h1 h2 => place_keep _ hp (hq x h1 h2))
        (fun j _ _ hu hp x h1 h2 => ih hp hF hu x h1 h2) hu hw
      exact this.2 j (List.mem_range.2 hj) x hj1 hj2) r1.1 h2
  intro s hs x hx1 hx2
  unfold Def.selfSpans at hs
  rcases List.mem_append.1 hs with hs | hs
  · obtain ⟨r, hr, rfl⟩ := List.mem_map.1 hs
    obtain ⟨hr, hsr⟩ := List.mem_filter.1 hr
    have e : x = o + r.1 := by simp only at hx1 hx2; omega
    subst e
    exact foldlM_inv (Self (o + r.1)) (fun c _ u u' hu hw => keepC c u u' hw _ hu) (r1.2 r hr hsr) h2
  · obtain ⟨c, hc, rfl⟩ := List.mem_map.1 hs
    obtain ⟨hc, hcf⟩ := List.mem_filter.1 hc
    simp only [Bool.and_eq_true, Bool.or_eq_true, decide_eq_true_eq, beq_iff_eq] at hcf
    simp only at hx1 hx2
    exact r2.2 c hc hcf.1 hcf.2 x hx1 (by omega)

/-- **A full copy writes a self row at every column it has.** -/
theorem place_full {defs : Array Def} {k : Nat} {Fl : Array Bool}
    (hFl : ∀ (d : Nat) (df : Def), defs[d]? = some df → Fl.getD d false = true → df.full defs Fl = true) (fuel : Nat) :
    ∀ {d o inp acc acc'}, place defs k fuel d o inp acc = .ok acc' → Fl.getD d false = true → Sized acc →
      ∀ x, o ≤ x → x < o + (defs.getD d default).cols → Self x acc' := by
  induction fuel with
  | zero => intro d o inp acc acc' h; cases h
  | succ fuel ih =>
    intro d o inp acc acc' h hd hs x hx1 hx2
    cases hdf : defs[d]? with
    | none => simp only [place, hdf] at h; cases h
    | some df =>
      obtain ⟨-, hdd⟩ := getD_of_some hdf
      rw [hdd] at hx2
      obtain ⟨s, hs', h1, h2⟩ := covers_mem (hFl d df hdf hd) (Nat.zero_le (x - o)) (by omega)
      exact place_spans ih hdf h hs s hs' x (by omega) (by omega)

/-! ## The unit -/

theorem sized_empty (n : Nat) : Sized (Acc.empty n) := by simp [Sized, Acc.empty]

theorem selfOrEmpty_empty (n x : Nat) : SelfOrEmpty x (Acc.empty n) := by
  right
  simp only [Acc.empty, Array.getD_eq_getD_getElem?, Array.getElem?_replicate]
  constructor <;> split <;> rfl

/-- The template's flags, each Definition's as `inputsT` rechecks it. -/
theorem inputsT_flags {t : Tmpl} (h : t.inputsT = true) :
    (∀ (d : Nat) (df : Def), t.defs[d]? = some df → (selfFlags t.defs).getD d false = true →
      df.selfOnly (selfFlags t.defs) = true) ∧
    (∀ (d : Nat) (df : Def), t.defs[d]? = some df → (fullFlags t.defs).getD d false = true →
      df.full t.defs (fullFlags t.defs) = true) := by
  unfold Tmpl.inputsT at h
  simp only [Bool.and_eq_true, List.all_eq_true, List.mem_range] at h
  have hd := h.1.1.1.1
  refine ⟨fun d df hdf hf => ?_, fun d df hdf hf => ?_⟩
  all_goals obtain ⟨hlt, e⟩ := getD_of_some hdf; rw [← e]
  · exact imp_of_or (hd d hlt).1 hf
  · exact imp_of_or (hd d hlt).2 hf

/-- **The unrolled input rows are self rows or empty** (`Net.ofRows`' check on the rows before the input words' end). -/
theorem rows_inputRows {t : Tmpl} {acc : Acc} (hi : t.inputsT = true) (hr : t.rows = .ok acc) :
    ∀ i < t.inCols, (acc.ra.getD i #[] = #[i] ∧ acc.rb.getD i #[] = #[i]) ∨
      (acc.ra.getD i #[] = #[] ∧ acc.rb.getD i #[] = #[]) := by
  have hS := (inputsT_flags hi).1
  unfold Tmpl.inputsT at hi
  simp only [Bool.and_eq_true, List.all_eq_true, decide_eq_true_eq, Bool.or_eq_true] at hi
  obtain ⟨⟨⟨⟨-, hrows⟩, hcalls⟩, -⟩, -⟩ := hi
  unfold Tmpl.rows at hr
  obtain ⟨acc1, h1, h2⟩ := bind_ok hr
  let I := fun (u : Acc) => Sized u ∧ ∀ x < t.inCols, SelfOrEmpty x u
  suffices hI : I acc1 from fun i hi => write_selfOrEmpty hI.1 h2 (hI.2 i hi)
  simp only [place] at h1
  cases hdf : t.defs[t.root]? with
  | none => simp only [hdf] at h1; cases h1
  | some df =>
    simp only [hdf] at h1
    obtain ⟨-, hdd⟩ := getD_of_some hdf
    rw [hdd] at hrows hcalls
    obtain ⟨acc2, h3, h4⟩ := bind_ok h1
    have i2 : I acc2 := foldlM_inv I (fun r hr u u' hu hw => by
        refine ⟨write_sized hw hu.1, fun x hx => ?_⟩
        rcases hrows r hr with hc | hs
        · exact same_selfOrEmpty (write_same hw (by omega)) (hu.2 x hx)
        · unfold selfRow at hs
          simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hs
          obtain ⟨⟨hc, ha⟩, hb⟩ := hs
          rw [ha, hb, self_canon hc] at hw
          exact write_selfOrEmpty hu.1 hw (hu.2 x hx))
      ⟨sized_empty _, fun x _ => selfOrEmpty_empty _ x⟩ h3
    exact foldlM_inv I (fun c hc u u' hu hw =>
      foldlM_inv I (fun j hj w w' hw0 hp => by
        refine ⟨place_sized _ hp hw0.1, fun x hx => ?_⟩
        rcases hcalls c hc with hb | hsc
        · exact same_selfOrEmpty (place_above _ hp x (by omega)) (hw0.2 x hx)
        · exact place_selfOnly hS _ hp hsc hw0.1 x (hw0.2 x hx)) hu hw) i2 h4

/-- **Each input port bit has its unrolled self row, and the input groups ascend** (`Net.inputsOk`). -/
theorem rows_inputsOk {t : Tmpl} {acc : Acc} (hi : t.inputsT = true) (hr : t.rows = .ok acc) :
    Net.inputsOk t.inGroups acc.ra acc.rb = true := by
  have hFl := (inputsT_flags hi).2
  unfold Tmpl.inputsT at hi
  simp only [Bool.and_eq_true, List.all_eq_true, Array.all_eq_true'] at hi
  obtain ⟨⟨-, hord⟩, hcov⟩ := hi
  unfold Tmpl.rows at hr
  obtain ⟨acc1, h1, h2⟩ := bind_ok hr
  cases hdf : t.defs[t.root]? with
  | none => simp only [place, hdf] at h1; cases h1
  | some df =>
    obtain ⟨-, hdd⟩ := getD_of_some hdf
    have hself := place_spans (place_full hFl t.defs.size) hdf h1 (sized_empty _)
    unfold Net.inputsOk
    simp only [Bool.and_eq_true, List.all_eq_true, Array.all_eq_true', List.mem_range, beq_iff_eq]
    refine ⟨fun i hi => hord i (List.mem_range.2 hi), fun g hg j hj => ?_⟩
    have hc := hcov g hg
    rw [hdd] at hc
    obtain ⟨s, hs, h3, h4⟩ := covers_mem hc (Nat.le_add_right _ j) (by omega)
    have := write_keep h2 (hself s (List.mem_filter.1 hs).1 (g.col * WORD + j) (by omega) (by omega))
    exact ⟨this.2.1, this.2.2⟩

/-- **An ordered template whose input rows pass `inputsT` unrolls to rows in the order `Net.checkOrder` checks**
(`Flock.TmplOrder.rows_ordered`, its `Net.inputsOk` from the description). -/
theorem rows_order {t : Tmpl} {acc : Acc} (ho : t.ordered = true) (hi : t.inputsT = true) (hr : t.rows = .ok acc) :
    ∀ r, t.inCols ≤ r → r < t.constPos → ∀ c ∈ (acc.ra.getD r #[]).toList ++ (acc.rb.getD r #[]).toList,
      c = t.constPos ∨
        (c < r ∧ ¬ (c < t.inCols ∧ (acc.ra.getD c #[]).isEmpty = true ∧ (acc.rb.getD c #[]).isEmpty = true)) :=
  rows_ordered ho hr (rows_inputsOk hi hr)

end Flock.TmplRows
