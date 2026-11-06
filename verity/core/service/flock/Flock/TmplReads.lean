/-
A templated unit's input columns that a computed row surely reads, from its description. `RecResiduals.readIns`
marks the input columns the unit's computed rows read by walking every row of the unrolled net; `Tmpl.readsSure` marks a
subset of them per Definition, so `KnownRead.readsOk` can check a templated unit's known bits without building its rows.

A row's side is a sum over GF(2), so a reference can cancel against another bound to the same column. A side that names
exactly one of its Definition's inputs, once (`soleIn`), keeps that input's column wherever the copy's inputs are bound:
its other references are the copy's own columns, at or past the copy's offset, and the constant, while an input that
reaches an input row's column is bound below the copy. An input of a Definition is then surely read where one of its rows
has it as its side's sole input, or where one of its calls binds it to a callee's input that is (`Def.sure`, Definitions
in order, each callee before its callers). The root marks its input rows' columns below the constant the same way, from
its rows and calls past the input rows (`Tmpl.readsSure`): those are computed rows. `rows_sure` proves each column it
marks is read by a computed row of `Tmpl.rows`. Its cost is each Definition's rows and each call's bound inputs, never
the unit's gates.
-/
import Flock.TmplOrder

namespace Flock.Tmpl

/-- The one reference `v - lo` among `[lo, hi)` that side `xs` names, where it names exactly one such reference, once. -/
def soleIn (lo hi : Nat) (xs : Array Nat) : Option Nat :=
  match xs.toList.filter (fun v => decide (lo ≤ v ∧ v < hi)) with
  | [v] => some (v - lo)
  | _ => none

/-- Mark `v - lo` in `m` where `v` is in `[lo, hi)`. -/
def markIn (lo hi v : Nat) (m : Array Bool) : Array Bool :=
  if lo ≤ v ∧ v < hi then m.setIfInBounds (v - lo) true else m

/-- Mark, in `m`, each reference in `[lo, hi)` that a copy `j < copies` binds, in run `r` of segment `s` (its callee's
inputs from `off`), to an input its callee surely reads (`se`). -/
def segMarks (se : Array Bool) (lo hi copies off : Nat) (s : Seg) (m : Array Bool) : Array Bool :=
  (List.range s.reps).foldl (fun m r => (List.range s.len).foldl (fun m t =>
      if se.getD (off + r * s.len + t) false then
        (List.range copies).foldl (fun m j => markIn lo hi (s.at_ + r * s.rstride + j * s.step + t) m) m
      else m) m) m

/-- `segMarks` over a call's segments, from its callee's input `off`. A segment of step 0 binds every copy alike, so only
its first copy is read. -/
def segsMarks (se : Array Bool) (lo hi count : Nat) : List Seg → Nat → Array Bool → Array Bool
  | [], _, m => m
  | s :: ss, off, m =>
    segsMarks se lo hi count ss (off + s.reps * s.len)
      (segMarks se lo hi (if s.step == 0 then min 1 count else count) off s m)

/-- Mark, in `m`, each reference in `[lo, hi)` that some copy of call `c` binds to an input its callee surely reads
(`se`). -/
def Call.marks (c : Call) (se : Array Bool) (lo hi : Nat) (m : Array Bool) : Array Bool :=
  segsMarks se lo hi c.count c.segs.toList 0 m

/-- Mark, in `m`, each reference in `[lo, hi)` that a row of `rows` names as its side's sole one. -/
def rowMarks (rows : List (Nat × Array Nat × Array Nat)) (lo hi : Nat) (m : Array Bool) : Array Bool :=
  rows.foldl (fun m r =>
    let m := match soleIn lo hi r.2.1 with | some i => m.setIfInBounds i true | none => m
    match soleIn lo hi r.2.2 with | some i => m.setIfInBounds i true | none => m) m

/-- Definition `df`'s inputs that each of its copies surely reads (`S`: the earlier Definitions'). -/
def Def.sure (S : Array (Array Bool)) (df : Def) : Array Bool :=
  df.calls.toList.foldl (fun m c => c.marks (S.getD c.def_ #[]) df.cols (df.cols + df.ins) m)
    (rowMarks df.rows.toList df.cols (df.cols + df.ins) (Array.replicate df.ins false))

/-- Each Definition's surely read inputs, in order. -/
def sures (defs : Array Def) : Array (Array Bool) := defs.foldl (fun S df => S.push (df.sure S)) #[]

/-- **The input rows' columns some computed row surely reads**: the root's rows past the input rows, and its calls past
them, each read as `Def.sure` reads a Definition's, over the root's columns below the input rows' end and the constant. -/
def Tmpl.readsSure (t : Tmpl) : Array Bool :=
  let S := sures t.defs
  let df := t.defs.getD t.root default
  let hi := min t.inCols t.constPos
  ((df.calls.toList.filter fun c => decide (t.inCols ≤ c.base)).foldl (fun m c => c.marks (S.getD c.def_ #[]) 0 hi m)
    (rowMarks (df.rows.toList.filter fun r => decide (t.inCols ≤ r.1)) 0 hi (Array.replicate hi false)))

end Flock.Tmpl

namespace Flock.TmplReads

open Flock.Tmpl Flock.TmplOrder

/-! ## Marks -/

theorem getD_mark {m : Array Bool} {i j : Nat} (h : (m.setIfInBounds j true).getD i false = true) :
    m.getD i false = true ∨ i = j := by
  simp only [Array.getD_eq_getD_getElem?, Array.getElem?_setIfInBounds] at h ⊢
  by_cases e : j = i
  · exact Or.inr e.symm
  · simp only [e, ite_false] at h
    exact Or.inl h

theorem getD_replicate_false {n i : Nat} : (Array.replicate n false).getD i false = false := by
  simp only [Array.getD_eq_getD_getElem?, Array.getElem?_replicate]
  split <;> rfl

/-- A mark a fold sets is set before it or by one of its steps. -/
theorem foldl_mark {α : Type} {Q : α → Prop} {g : Array Bool → α → Array Bool} {i : Nat}
    (hg : ∀ m a, (g m a).getD i false = true → m.getD i false = true ∨ Q a) :
    ∀ {l : List α} {m : Array Bool}, (l.foldl g m).getD i false = true → m.getD i false = true ∨ ∃ a ∈ l, Q a
  | [], _, h => Or.inl h
  | a :: l, m, h => by
    rw [List.foldl_cons] at h
    rcases foldl_mark hg (l := l) h with h | ⟨b, hb, hq⟩
    · rcases hg m a h with h | hq
      · exact Or.inl h
      · exact Or.inr ⟨a, List.mem_cons_self, hq⟩
    · exact Or.inr ⟨b, List.mem_cons_of_mem _ hb, hq⟩

theorem markIn_mark {lo hi v i : Nat} {m : Array Bool} (h : (markIn lo hi v m).getD i false = true) :
    m.getD i false = true ∨ (lo ≤ v ∧ v < hi ∧ i = v - lo) := by
  unfold markIn at h
  split at h
  · rename_i hv
    rcases getD_mark h with h | h
    · exact Or.inl h
    · exact Or.inr ⟨hv.1, hv.2, h⟩
  · exact Or.inl h

theorem soleIn_some {lo hi i : Nat} {xs : Array Nat} (h : soleIn lo hi xs = some i) :
    ∃ v, xs.toList.filter (fun v => decide (lo ≤ v ∧ v < hi)) = [v] ∧ lo ≤ v ∧ v < hi ∧ i = v - lo := by
  unfold soleIn at h
  split at h
  · rename_i v hv
    cases h
    have hm : v ∈ xs.toList.filter (fun v => decide (lo ≤ v ∧ v < hi)) := by rw [hv]; exact List.mem_cons_self
    have := (List.mem_filter.1 hm).2
    simp only [decide_eq_true_eq] at this
    exact ⟨v, hv, this.1, this.2, rfl⟩
  · cases h

theorem rowMarks_mark {rows : List (Nat × Array Nat × Array Nat)} {lo hi i : Nat} {m : Array Bool}
    (h : (rowMarks rows lo hi m).getD i false = true) :
    m.getD i false = true ∨ ∃ r ∈ rows, soleIn lo hi r.2.1 = some i ∨ soleIn lo hi r.2.2 = some i := by
  unfold rowMarks at h
  refine foldl_mark (fun m r h => ?_) h
  dsimp only at h
  split at h
  · rename_i j hj
    rcases getD_mark h with h | rfl
    · split at h
      · rename_i j' hj'
        rcases getD_mark h with h | rfl
        · exact Or.inl h
        · exact Or.inr (Or.inl hj')
      · exact Or.inl h
    · exact Or.inr (Or.inr hj)
  · split at h
    · rename_i j' hj'
      rcases getD_mark h with h | rfl
      · exact Or.inl h
      · exact Or.inr (Or.inl hj')
    · exact Or.inl h

theorem segMarks_mark {se : Array Bool} {lo hi copies off i : Nat} {s : Seg} {m : Array Bool}
    (h : (segMarks se lo hi copies off s m).getD i false = true) :
    m.getD i false = true ∨ ∃ r < s.reps, ∃ t < s.len, ∃ j < copies, se.getD (off + r * s.len + t) false = true ∧
      lo ≤ s.at_ + r * s.rstride + j * s.step + t ∧ s.at_ + r * s.rstride + j * s.step + t < hi ∧
      i = s.at_ + r * s.rstride + j * s.step + t - lo := by
  unfold segMarks at h
  have step : ∀ (m : Array Bool) (r : Nat),
      ((List.range s.len).foldl (fun m t =>
        if se.getD (off + r * s.len + t) false = true then
          (List.range copies).foldl (fun m j => markIn lo hi (s.at_ + r * s.rstride + j * s.step + t) m) m
        else m) m).getD i false = true →
      m.getD i false = true ∨ ∃ t < s.len, ∃ j < copies, se.getD (off + r * s.len + t) false = true ∧
        lo ≤ s.at_ + r * s.rstride + j * s.step + t ∧ s.at_ + r * s.rstride + j * s.step + t < hi ∧
        i = s.at_ + r * s.rstride + j * s.step + t - lo := by
    intro m r h
    rcases foldl_mark (Q := fun t => ∃ j < copies, se.getD (off + r * s.len + t) false = true ∧
        lo ≤ s.at_ + r * s.rstride + j * s.step + t ∧ s.at_ + r * s.rstride + j * s.step + t < hi ∧
        i = s.at_ + r * s.rstride + j * s.step + t - lo) (fun m t h => by
      split at h
      · rename_i hse
        rcases foldl_mark (Q := fun j => lo ≤ s.at_ + r * s.rstride + j * s.step + t ∧
            s.at_ + r * s.rstride + j * s.step + t < hi ∧ i = s.at_ + r * s.rstride + j * s.step + t - lo)
          (fun m j h => markIn_mark h) h with h | ⟨j, hj, hq⟩
        · exact Or.inl h
        · exact Or.inr ⟨j, List.mem_range.1 hj, hse, hq⟩
      · exact Or.inl h) h with h | ⟨t, ht, hq⟩
    · exact Or.inl h
    · exact Or.inr ⟨t, List.mem_range.1 ht, hq⟩
  rcases foldl_mark step h with h | ⟨r, hr, hq⟩
  · exact Or.inl h
  · exact Or.inr ⟨r, List.mem_range.1 hr, hq⟩

/-! ## A call's bound inputs -/

/-- Copy `j`'s inputs bound by segments `ss` through the caller's references `g` (`Call.bound`). -/
def segsBound (ss : List Seg) (g : Nat → Nat) (j : Nat) : List Nat :=
  ss.flatMap fun s => (List.range s.reps).flatMap fun r =>
    (List.range s.len).map fun t => g (s.at_ + r * s.rstride + j * s.step + t)

theorem grid_length (n len : Nat) (f : Nat → Nat → Nat) :
    ((List.range n).flatMap fun r => (List.range len).map (f r)).length = n * len := by
  induction n with
  | zero => simp
  | succ n ih => rw [List.range_succ, List.flatMap_append, List.length_append, ih]; simp [Nat.succ_mul]

theorem grid_getD {len t d : Nat} {f : Nat → Nat → Nat} (ht : t < len) :
    ∀ {n r : Nat}, r < n → ((List.range n).flatMap fun r => (List.range len).map (f r)).getD (r * len + t) d = f r t
  | 0, _, hr => absurd hr (Nat.not_lt_zero _)
  | n + 1, r, hr => by
    rw [List.range_succ, List.flatMap_append, List.getD_eq_getElem?_getD]
    rcases Nat.lt_succ_iff_lt_or_eq.1 hr with hr | rfl
    · have h1 : (r + 1) * len ≤ n * len := Nat.mul_le_mul_right len hr
      rw [List.getElem?_append_left (by rw [grid_length, Nat.succ_mul] at *; omega), ← List.getD_eq_getElem?_getD,
        grid_getD ht hr]
    · rw [List.getElem?_append_right (by rw [grid_length]; omega), grid_length]
      simp [ht]

theorem segsMarks_mark {se : Array Bool} {lo hi count i : Nat} :
    ∀ {ss : List Seg} {off : Nat} {m : Array Bool}, (segsMarks se lo hi count ss off m).getD i false = true →
      m.getD i false = true ∨ ∃ q j v, j < count ∧ se.getD (off + q) false = true ∧ lo ≤ v ∧ v < hi ∧ i = v - lo ∧
        ∀ (g : Nat → Nat) (d : Nat), (segsBound ss g j).getD q d = g v
  | [], _, _, h => Or.inl h
  | s :: ss, off, m, h => by
    unfold segsMarks at h
    have hlen : ∀ (g : Nat → Nat) (j : Nat), ((List.range s.reps).flatMap fun r =>
        (List.range s.len).map fun t => g (s.at_ + r * s.rstride + j * s.step + t)).length = s.reps * s.len :=
      fun g j => grid_length _ _ _
    rcases segsMarks_mark h with h | ⟨q, j, v, hj, hse, hv⟩
    · rcases segMarks_mark h with h | ⟨r, hr, t, ht, j, hj, hse, hv1, hv2, hv3⟩
      · exact Or.inl h
      · refine Or.inr ⟨r * s.len + t, j, _, ?_, by rw [← Nat.add_assoc]; exact hse, hv1, hv2, hv3, fun g d => ?_⟩
        · split at hj
          · exact Nat.lt_of_lt_of_le hj (Nat.min_le_right _ _)
          · exact hj
        · unfold segsBound
          rw [List.flatMap_cons, List.getD_eq_getElem?_getD, List.getElem?_append_left (by
            rw [hlen]
            have : (r + 1) * s.len ≤ s.reps * s.len := Nat.mul_le_mul_right _ hr
            rw [Nat.succ_mul] at this
            omega), ← List.getD_eq_getElem?_getD]
          exact grid_getD (f := fun r t => g (s.at_ + r * s.rstride + j * s.step + t)) ht hr
    · refine Or.inr ⟨s.reps * s.len + q, j, v, hj, by rw [← Nat.add_assoc]; exact hse, hv.1, hv.2.1, hv.2.2.1,
        fun g d => ?_⟩
      have := hv.2.2.2 g d
      unfold segsBound at this ⊢
      rw [List.flatMap_cons, List.getD_eq_getElem?_getD, List.getElem?_append_right (by rw [hlen]; omega), hlen,
        Nat.add_sub_cancel_left, ← List.getD_eq_getElem?_getD]
      exact this

/-! ## `canon` keeps a column named once -/

theorem count_dropPairs : ∀ {l : List Nat}, l.Pairwise (· ≤ ·) → ∀ y, (dropPairs l).count y = l.count y % 2
  | [], _, y => by simp [dropPairs]
  | [a], _, y => by
    simp only [dropPairs, List.count_singleton]
    split <;> simp
  | a :: b :: rest, hs, y => by
    have hs' := List.pairwise_cons.1 hs
    simp only [dropPairs]
    split
    · rename_i hab
      have e : a = b := by simpa using hab
      subst e
      rw [count_dropPairs (List.pairwise_cons.1 hs'.2).2 y]
      simp only [List.count_cons]
      split <;> omega
    · rename_i hab
      have hab' : a ≠ b := by simpa using hab
      rw [List.count_cons, count_dropPairs hs'.2 y]
      by_cases hy : a = y
      · subst hy
        have h0 : (b :: rest).count a = 0 := List.count_eq_zero.2 fun hm => by
          rcases List.mem_cons.1 hm with e | hm
          · exact hab' e
          · have h1 := hs'.1 b (List.mem_cons_self)
            have h2 := (List.pairwise_cons.1 hs'.2).1 a hm
            have h3 := hs'.1 a (List.mem_cons_of_mem _ hm)
            exact hab' (Nat.le_antisymm h1 (by omega))
        rw [h0, List.count_cons, h0]
        simp
      · have hy' : (a == y) = false := by simpa using hy
        simp [List.count_cons, hy]

/-- A column a side names exactly once is in its canonical row. -/
theorem mem_canon_of_count {xs : Array Nat} {y : Nat} (h : xs.toList.count y = 1) :
    y ∈ (Flock.Tmpl.canon xs).toList := by
  have hp := List.mergeSort_perm xs.toList (fun a b => decide (a ≤ b))
  have hs : (xs.toList.mergeSort fun a b => decide (a ≤ b)).Pairwise (· ≤ ·) :=
    (List.pairwise_mergeSort (le := fun a b => decide (a ≤ b))
      (fun a b c h1 h2 => by simp only [decide_eq_true_eq] at *; omega)
      (fun a b => by simp only [Bool.or_eq_true, decide_eq_true_eq]; omega) xs.toList).imp fun h => by
        simpa using h
  have hc := count_dropPairs hs y
  rw [hp.count_eq, h] at hc
  show y ∈ dropPairs (xs.toList.mergeSort fun a b => decide (a ≤ b))
  exact List.count_pos_iff.1 (by omega)

/-! ## Written rows stay -/

/-- The accumulator's arrays, each of `n` columns. -/
def SizeOk (n : Nat) (acc : Acc) : Prop := acc.set.size = n ∧ acc.ra.size = n ∧ acc.rb.size = n

/-- Column `y` is read, in A or B, by a written row at or past `ic`. -/
def Wit (ic y : Nat) (acc : Acc) : Prop :=
  ∃ r, ic ≤ r ∧ acc.set.getD r false = true ∧ (y ∈ (acc.ra.getD r #[]).toList ∨ y ∈ (acc.rb.getD r #[]).toList)

theorem getD_true_of {m : Array Bool} {r : Nat} (h : m.getD r false = true) : m.getD r true = true ∧ r < m.size := by
  simp only [Array.getD_eq_getD_getElem?] at h ⊢
  cases e : m[r]? with
  | none => rw [e] at h; cases h
  | some v =>
    rw [e] at h
    exact ⟨h, (Array.getElem?_eq_some_iff.1 e).1⟩

theorem write_size {n : Nat} {acc acc' : Acc} {c : Nat} {a b : Array Nat} (h : write acc c a b = .ok acc')
    (hs : SizeOk n acc) : SizeOk n acc' := by
  unfold write at h
  split at h
  · cases h
  · cases h
    simp only [SizeOk, Array.set!_eq_setIfInBounds, Array.size_setIfInBounds] at hs ⊢
    exact hs

theorem write_wit {ic y : Nat} {acc acc' : Acc} {c : Nat} {a b : Array Nat} (h : write acc c a b = .ok acc')
    (hw : Wit ic y acc) : Wit ic y acc' := by
  obtain ⟨r, hr, hset, hy⟩ := hw
  have hrc : r ≠ c := fun e => by
    subst e
    unfold write at h
    rw [ite_eq_left (getD_true_of hset).1] at h
    cases h
  unfold write at h
  split at h
  · cases h
  · cases h
    refine ⟨r, hr, ?_, ?_⟩
    · simp only [Array.getD_eq_getD_getElem?, Array.set!_eq_setIfInBounds, Array.getElem?_setIfInBounds,
        Ne.symm hrc, ite_false] at hset ⊢
      exact hset
    · rcases getD_set! acc.ra c r a with e | ⟨e, -⟩
      · rcases getD_set! acc.rb c r b with e' | ⟨e', -⟩
        · dsimp only
          rw [e, e']
          exact hy
        · exact absurd e' hrc
      · exact absurd e hrc

theorem write_new {n ic y : Nat} {acc acc' : Acc} {c : Nat} {a b : Array Nat} (h : write acc c a b = .ok acc')
    (hs : SizeOk n acc) (hic : ic ≤ c) (hy : y ∈ a.toList ∨ y ∈ b.toList) : Wit ic y acc' := by
  unfold write at h
  split at h
  · cases h
  · rename_i hc
    cases h
    have hcs : c < acc.set.size := Nat.lt_of_not_le fun hle => hc (by
      simp only [Array.getD_eq_getD_getElem?, Array.getElem?_eq_none hle, Option.getD_none])
    obtain ⟨h1, h2, h3⟩ := hs
    refine ⟨c, hic, ?_, ?_⟩
    · simp only [Array.getD_eq_getD_getElem?, Array.set!_eq_setIfInBounds, Array.getElem?_setIfInBounds, hcs,
        ite_true, Option.getD_some]
    · simp only [Array.getD_eq_getD_getElem?, Array.set!_eq_setIfInBounds, Array.getElem?_setIfInBounds,
        ite_true, Option.getD_some, show c < acc.ra.size by omega, show c < acc.rb.size by omega]
      exact hy

/-- A property every write keeps, every copy keeps. -/
theorem place_inv (P : Acc → Prop)
    (hw : ∀ {acc acc' : Acc} {c : Nat} {a b : Array Nat}, write acc c a b = .ok acc' → P acc → P acc')
    {defs : Array Def} {k : Nat} :
    ∀ (fuel : Nat) {d o : Nat} {inp : List Nat} {acc acc' : Acc}, place defs k fuel d o inp acc = .ok acc' → P acc → P acc'
  | 0, _, _, _, _, _, h, _ => by cases h
  | fuel + 1, d, o, inp, acc, acc', h, hp => by
    simp only [place] at h
    cases hdf : defs[d]? with
    | none => simp only [hdf] at h; cases h
    | some df =>
      simp only [hdf] at h
      obtain ⟨acc1, h1, h2⟩ := bind_ok h
      have p1 := foldlM_inv P (fun _ _ _ _ hu hw' => hw hw' hu) hp h1
      exact foldlM_inv P (fun _ _ _ _ hu hw' =>
        foldlM_inv P (fun _ _ _ _ hw0 hpl => place_inv P hw fuel hpl hw0) hu hw') p1 h2

/-- A fold one of whose steps makes `Q` hold, every step keeping `P` and `Q`, ends with `Q`. -/
theorem foldlM_wit {α σ : Type} (P Q : σ → Prop) {f : σ → α → Except String σ} :
    ∀ {l : List α} {s s' : σ}, (∀ a ∈ l, ∀ u u', P u → f u a = .ok u' → P u') →
      (∀ a ∈ l, ∀ u u', Q u → f u a = .ok u' → Q u') →
      (∃ a ∈ l, ∀ u u', P u → f u a = .ok u' → Q u') → P s → l.foldlM f s = .ok s' → Q s'
  | [], _, _, _, _, hmk, _, _ => by
    obtain ⟨a, ha, -⟩ := hmk
    cases ha
  | a :: l, s, s', hP, hQ, hmk, hs, h => by
    rw [List.foldlM_cons] at h
    obtain ⟨s1, h1, h2⟩ := bind_ok h
    obtain ⟨b, hb, hb'⟩ := hmk
    rcases List.mem_cons.1 hb with rfl | hb
    · exact foldlM_inv Q (fun c hc => hQ c (List.mem_cons_of_mem _ hc)) (hb' s s1 hs h1) h2
    · exact foldlM_wit P Q (fun c hc => hP c (List.mem_cons_of_mem _ hc))
        (fun c hc => hQ c (List.mem_cons_of_mem _ hc)) ⟨b, hb, hb'⟩ (hP a List.mem_cons_self s s1 hs h1) h2

/-! ## Surely read inputs are read -/

/-- Input `i` of Definition `d` is read, in each copy placed past `ic` whose input `i` is bound below the copy and not to
the constant `k`, by a row the copy writes at or past `ic`. -/
def Sure (defs : Array Def) (k ic d i : Nat) : Prop :=
  ∀ (fuel o : Nat) (inp : List Nat) (acc acc' : Acc) (n : Nat), place defs k fuel d o inp acc = .ok acc' → ic ≤ o →
    inp.getD i k < o → inp.getD i k ≠ k → SizeOk n acc → Wit ic (inp.getD i k) acc'

/-- Every input `S` marks is surely read. -/
def Sound (defs : Array Def) (k ic : Nat) (S : Array (Array Bool)) : Prop :=
  ∀ e i, (S.getD e #[]).getD i false = true → Sure defs k ic e i

/-- A side whose sole reference in `[lo, hi)` is `lo + i`, mapped by `g` to `y` there and away from `y` elsewhere, names
`y` once. -/
theorem sole_count {lo hi i y : Nat} {xs : Array Nat} {g : Nat → Nat} (h : soleIn lo hi xs = some i)
    (hout : ∀ v, ¬(lo ≤ v ∧ v < hi) → g v ≠ y) (hin : ∀ v, lo ≤ v → v < hi → v - lo = i → g v = y) :
    (xs.map g).toList.count y = 1 := by
  obtain ⟨v, hf, hv1, hv2, hv3⟩ := soleIn_some h
  rw [Array.toList_map, List.count_eq_countP, List.countP_map,
    List.countP_congr (q := fun v => decide (lo ≤ v ∧ v < hi)) (fun x hx => ?_), List.countP_eq_length_filter, hf]
  · rfl
  · simp only [Function.comp, beq_iff_eq, decide_eq_true_eq]
    constructor
    · intro e
      exact Classical.byContradiction fun hn => hout x hn e
    · intro hr
      have hm : x ∈ xs.toList.filter (fun v => decide (lo ≤ v ∧ v < hi)) :=
        List.mem_filter.2 ⟨hx, by simpa using hr⟩
      rw [hf] at hm
      have e := List.mem_singleton.1 hm
      subst e
      exact hin x hv1 hv2 hv3.symm

theorem ref_in {df : Def} {o k v : Nat} {inp : List Nat} (h1 : df.cols ≤ v) (h2 : v < df.cols + df.ins) :
    df.ref o inp k v = inp.getD (v - df.cols) k := by
  unfold Def.ref
  rw [ite_eq_right (by omega), ite_eq_left h2]

theorem ref_out {df : Def} {o k v y : Nat} {inp : List Nat} (hy : y < o) (hk : y ≠ k)
    (h : ¬(df.cols ≤ v ∧ v < df.cols + df.ins)) : df.ref o inp k v ≠ y := by
  unfold Def.ref
  split
  · omega
  · rw [ite_eq_right (by omega)]
    exact Ne.symm hk

theorem keep_wit {ic y : Nat} {defs : Array Def} {k fuel : Nat} {df : Def} {o : Nat} {inp : List Nat} {u u' : Acc}
    (h : df.calls.toList.foldlM (fun acc c => (List.range c.count).foldlM (fun acc j =>
      place defs k fuel c.def_ (o + c.base + j * c.stride) (c.bound (df.ref o inp k) j) acc) acc) u = .ok u')
    (hw : Wit ic y u) : Wit ic y u' :=
  foldlM_inv (Wit ic y) (fun _ _ _ _ hu hw' => foldlM_inv (Wit ic y) (fun _ _ _ _ hw0 hp =>
    place_inv (Wit ic y) (fun hw h => write_wit hw h) fuel hp hw0) hu hw') hw h

theorem keep_size {n : Nat} {defs : Array Def} {k fuel : Nat} {df : Def} {o : Nat} {inp : List Nat} {u u' : Acc}
    (h : df.calls.toList.foldlM (fun acc c => (List.range c.count).foldlM (fun acc j =>
      place defs k fuel c.def_ (o + c.base + j * c.stride) (c.bound (df.ref o inp k) j) acc) acc) u = .ok u')
    (hs : SizeOk n u) : SizeOk n u' :=
  foldlM_inv (SizeOk n) (fun _ _ _ _ hu hw' => foldlM_inv (SizeOk n) (fun _ _ _ _ hw0 hp =>
    place_inv (SizeOk n) (fun hw h => write_size hw h) fuel hp hw0) hu hw') hs h

/-- **A copy's call that binds a surely read input of its callee to `y` reads `y`.** -/
theorem calls_wit {defs : Array Def} {k ic fuel n y : Nat} {S : Array (Array Bool)} (hS : Sound defs k ic S) {df : Def}
    {o : Nat} {inp : List Nat} {u u' : Acc}
    (h : df.calls.toList.foldlM (fun acc c => (List.range c.count).foldlM (fun acc j =>
      place defs k fuel c.def_ (o + c.base + j * c.stride) (c.bound (df.ref o inp k) j) acc) acc) u = .ok u')
    (hs : SizeOk n u) (hk : y ≠ k)
    (hc : ∃ c ∈ df.calls.toList, ic ≤ o + c.base ∧ y < o + c.base ∧ ∃ q j, j < c.count ∧
      (S.getD c.def_ #[]).getD q false = true ∧ (c.bound (df.ref o inp k) j).getD q k = y) : Wit ic y u' := by
  obtain ⟨c, hc, hic, hy, q, j, hj, hse, hb⟩ := hc
  refine foldlM_wit (SizeOk n) (Wit ic y)
    (fun _ _ _ _ hu hw' => foldlM_inv (SizeOk n) (fun _ _ _ _ hw0 hp =>
      place_inv (SizeOk n) (fun hw h => write_size hw h) fuel hp hw0) hu hw')
    (fun _ _ _ _ hu hw' => foldlM_inv (Wit ic y) (fun _ _ _ _ hw0 hp =>
      place_inv (Wit ic y) (fun hw h => write_wit hw h) fuel hp hw0) hu hw')
    ⟨c, hc, fun w w' hw hp => ?_⟩ hs h
  refine foldlM_wit (SizeOk n) (Wit ic y)
    (fun _ _ _ _ hu hp => place_inv (SizeOk n) (fun hw h => write_size hw h) fuel hp hu)
    (fun _ _ _ _ hu hp => place_inv (Wit ic y) (fun hw h => write_wit hw h) fuel hp hu)
    ⟨j, List.mem_range.2 hj, fun w w' hw hp => ?_⟩ hw hp
  have := hS c.def_ q hse fuel _ _ w w' n hp (by omega) (by rw [hb]; omega) (by rw [hb]; exact hk) hw
  rwa [hb] at this

/-- **Each input `Def.sure` marks is surely read**, its callees' marks (`S`) sound. -/
theorem def_sure {defs : Array Def} {k ic : Nat} {S : Array (Array Bool)} (hS : Sound defs k ic S) {d : Nat} {df : Def}
    (hdf : defs[d]? = some df) {i : Nat} (hi : (df.sure S).getD i false = true) : Sure defs k ic d i := by
  intro fuel o inp acc acc' n h hic hlt hne hsz
  cases fuel with
  | zero => cases h
  | succ fuel =>
    simp only [place, hdf] at h
    obtain ⟨acc1, h1, h2⟩ := bind_ok h
    have hsz1 : SizeOk n acc1 := foldlM_inv (SizeOk n) (fun _ _ _ _ hu hw => write_size hw hu) hsz h1
    unfold Def.sure at hi
    rcases foldl_mark (Q := fun c => ∃ q j v, j < c.count ∧ (S.getD c.def_ #[]).getD (0 + q) false = true ∧
        df.cols ≤ v ∧ v < df.cols + df.ins ∧ i = v - df.cols ∧
        ∀ (g : Nat → Nat) (d : Nat), (segsBound c.segs.toList g j).getD q d = g v)
      (fun m c h => segsMarks_mark h) hi with h0 | ⟨c, hc, q, j, v, hj, hse, hv1, hv2, hv3, hb⟩
    · rcases rowMarks_mark h0 with h0 | ⟨r, hr, hs⟩
      · rw [getD_replicate_false] at h0
        cases h0
      · have hrow : ∀ (xs : Array Nat), soleIn df.cols (df.cols + df.ins) xs = some i →
            (xs.map (df.ref o inp k)).toList.count (inp.getD i k) = 1 := fun xs hx =>
          sole_count hx (fun v hv => ref_out hlt hne hv) (fun v h1 h2 h3 => by rw [ref_in h1 h2, h3])
        refine keep_wit h2 (foldlM_wit (SizeOk n) (Wit ic (inp.getD i k))
          (fun _ _ _ _ hu hw => write_size hw hu) (fun _ _ _ _ hu hw => write_wit hw hu)
          ⟨r, hr, fun u u' hu hw => write_new hw hu (by omega) ?_⟩ hsz h1)
        rcases hs with hs | hs
        · exact Or.inl (mem_canon_of_count (hrow _ hs))
        · exact Or.inr (mem_canon_of_count (hrow _ hs))
    · refine calls_wit hS h2 hsz1 hne ⟨c, hc, by omega, by omega, q, j, hj, by simpa using hse, ?_⟩
      show (segsBound c.segs.toList (df.ref o inp k) j).getD q k = _
      rw [hb, ref_in hv1 hv2, ← hv3]

/-- **Every Definition's marks are sound.** -/
theorem sures_sound (defs : Array Def) (k ic : Nat) : Sound defs k ic (sures defs) := by
  unfold sures
  refine (Array.foldl_induction (as := defs) (motive := fun i S => S.size = i ∧ Sound defs k ic S)
    ⟨rfl, fun e i h => by simp at h⟩ (fun i S hS => ⟨by rw [Array.size_push, hS.1], fun e j h => ?_⟩)).2
  by_cases he : e = S.size
  · have e1 : (S.push (defs[i].sure S)).getD e #[] = defs[i].sure S := by
      simp [Array.getD_eq_getD_getElem?, he]
    rw [e1] at h
    obtain rfl : e = i.1 := he.trans hS.1
    exact def_sure hS.2 (d := i.1) (by simp) h
  · have e1 : (S.push (defs[i].sure S)).getD e #[] = S.getD e #[] := by
      simp only [Array.getD_eq_getD_getElem?, Array.getElem?_push, he, ite_false]
    rw [e1] at h
    exact hS.2 e j h

theorem root_ref_out {df : Def} {k hi v y : Nat} (hk : k = df.cols - 1) (hy : y < hi) (hhi : hi ≤ k) (hv : hi ≤ v) :
    df.ref 0 [] k v ≠ y := by
  rcases root_ref (df := df) (k := k) (v := v) with ⟨-, e⟩ | e
  · rw [e]; omega
  · rw [e]; omega

theorem root_ref_in {df : Def} {k v : Nat} (hk : k = df.cols - 1) (hv : v < k) : df.ref 0 [] k v = v := by
  unfold Def.ref
  rw [ite_eq_left (by omega)]
  omega

/-- **Every column `Tmpl.readsSure` marks is read by a computed row**: it is below the input rows' end and the constant,
and a row of `Tmpl.rows` from the input rows to the constant reads it, in A or in B. -/
theorem rows_sure {t : Tmpl} {acc : Acc} (hr : t.rows = .ok acc) {y : Nat} (hy : t.readsSure.getD y false = true) :
    y < t.inCols ∧ y < t.constPos ∧ ∃ r, t.inCols ≤ r ∧ r < t.constPos ∧
      (y ∈ (acc.ra.getD r #[]).toList ∨ y ∈ (acc.rb.getD r #[]).toList) := by
  unfold Tmpl.rows at hr
  obtain ⟨acc1, h1, h2⟩ := bind_ok hr
  have hS := sures_sound t.defs t.constPos t.inCols
  simp only [place] at h1
  cases hdf : t.defs[t.root]? with
  | none => simp only [hdf] at h1; cases h1
  | some df =>
    simp only [hdf] at h1
    obtain ⟨-, hdd⟩ := getD_of_some hdf
    have hk : t.constPos = df.cols - 1 := by unfold Tmpl.constPos; rw [hdd]
    obtain ⟨acc2, h3, h4⟩ := bind_ok h1
    rw [hdd] at h3
    have hsz0 : SizeOk df.cols (Acc.empty df.cols) := by simp [SizeOk, Acc.empty]
    have hsz2 : SizeOk df.cols acc2 := foldlM_inv (SizeOk df.cols) (fun _ _ _ _ hu hw => write_size hw hu) hsz0 h3
    have hsz1 : SizeOk df.cols acc1 := keep_size h4 hsz2
    have hw : y < min t.inCols t.constPos ∧ Wit t.inCols y acc1 := by
      unfold Tmpl.readsSure at hy
      dsimp only at hy
      rw [hdd] at hy
      rcases foldl_mark (Q := fun c => ∃ q j v, j < c.count ∧
          ((sures t.defs).getD c.def_ #[]).getD (0 + q) false = true ∧ 0 ≤ v ∧ v < min t.inCols t.constPos ∧
          y = v - 0 ∧ ∀ (g : Nat → Nat) (d : Nat), (segsBound c.segs.toList g j).getD q d = g v)
        (fun m c h => segsMarks_mark h) hy with h0 | ⟨c, hc, q, j, v, hj, hse, -, hv2, hv3, hb⟩
      · rcases rowMarks_mark h0 with h0 | ⟨r, hr, hs⟩
        · rw [getD_replicate_false] at h0
          cases h0
        · obtain ⟨hr, hrc⟩ := List.mem_filter.1 hr
          simp only [decide_eq_true_eq] at hrc
          have hlt : y < min t.inCols t.constPos := by
            rcases hs with hs | hs <;> (obtain ⟨v, -, -, hv, e⟩ := soleIn_some hs; omega)
          have hrow : ∀ (xs : Array Nat), soleIn 0 (min t.inCols t.constPos) xs = some y →
              (xs.map (df.ref 0 [] t.constPos)).toList.count y = 1 := fun xs hx =>
            sole_count hx (fun v hv => root_ref_out hk hlt (Nat.min_le_right _ _) (by omega))
              (fun v _ h2 h3 => by rw [root_ref_in hk (by omega)]; omega)
          refine ⟨hlt, keep_wit h4 (foldlM_wit (SizeOk df.cols) (Wit t.inCols y)
            (fun _ _ _ _ hu hw => write_size hw hu) (fun _ _ _ _ hu hw => write_wit hw hu)
            ⟨r, hr, fun u u' hu hw => write_new hw hu (by omega) ?_⟩ hsz0 h3)⟩
          rcases hs with hs | hs
          · exact Or.inl (mem_canon_of_count (hrow _ hs))
          · exact Or.inr (mem_canon_of_count (hrow _ hs))
      · obtain ⟨hc, hcb⟩ := List.mem_filter.1 hc
        simp only [decide_eq_true_eq] at hcb
        refine ⟨by omega, calls_wit hS h4 hsz2 (by omega) ⟨c, hc, by omega, by omega, q, j, hj,
          by simpa using hse, ?_⟩⟩
        show (segsBound c.segs.toList (df.ref 0 [] t.constPos) j).getD q t.constPos = y
        rw [hb, root_ref_in hk (by omega)]
        omega
    obtain ⟨hlt, r, hr1, hset, hyr⟩ := hw
    have hwr : acc1.set.getD t.constPos true = false := by
      unfold write at h2
      split at h2
      · cases h2
      · rename_i h
        simpa using h
    have hrK : r ≠ t.constPos := fun e => by
      subst e
      rw [(getD_true_of hset).1] at hwr
      cases hwr
    have hrc : r < df.cols := hsz1.1 ▸ (getD_true_of hset).2
    refine ⟨by omega, by omega, r, hr1, by omega, ?_⟩
    obtain ⟨ea, eb⟩ := write_ok h2 r
    rcases ea with ea | ⟨e, -⟩
    · rcases eb with eb | ⟨e, -⟩
      · rw [ea, eb]
        exact hyr
      · exact absurd e hrK
    · exact absurd e hrK

end Flock.TmplReads
