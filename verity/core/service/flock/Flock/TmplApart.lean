/-
A templated unit unrolls without a collision, checked on its description. `Tmpl.rows` refuses a column written
twice or past the unit, so a reader that does not unroll must know that no two writes meet. `Tmpl.apart` checks it once
per Definition: its rows and its calls' runs of copies take column ranges that, in order of their first column, each end
before the next begins, inside its columns; a call's copies sit side by side (or there is one); and the root writes
nothing at its last column, the constant's. `rows_ok` proves that a well-formed template that passes it unrolls
(`Tmpl.rows` succeeds). The proof is an induction over the copies `place` writes: a copy all of whose columns are unwritten
succeeds and writes only inside its rows' and calls' ranges (`place_ok`), so copies whose ranges are apart each find their
columns unwritten (`foldlM_apart`).
-/
import Flock.TmplRows

namespace Flock.Tmpl

/-- The column ranges `[lo, hi)` a Definition's rows and calls write, the empty ones left out: a row its column, a call
from its first copy's first column to its last copy's last. -/
def Def.spans (defs : Array Def) (df : Def) : List (Nat × Nat) :=
  ((df.rows.toList.map fun (r : Nat × Array Nat × Array Nat) => (r.1, r.1 + 1)) ++
    df.calls.toList.map fun (c : Call) => (c.base, c.base + (c.count - 1) * c.stride + (defs.getD c.def_ default).cols)).filter
    fun s => decide (s.1 < s.2)

/-- Each range ends before the next begins. -/
def ascend : List (Nat × Nat) → Bool
  | a :: b :: rest => decide (a.2 ≤ b.1) && ascend (b :: rest)
  | _ => true

/-- Definition `df`'s rows and calls write no column twice: their ranges, in order of their first column, end before the
next begins, and a call's copies sit side by side, or there is one. -/
def Def.apart (defs : Array Def) (df : Def) : Bool :=
  ascend ((df.spans defs).mergeSort fun a b => decide (a.1 ≤ b.1)) &&
    df.calls.toList.all fun c => decide (c.count ≤ 1) || decide ((defs.getD c.def_ default).cols ≤ c.stride)

/-- **The template unrolls without a collision** (`Flock.TmplApart.rows_ok`): every Definition's rows and copies inside
its columns and apart, and the root's before its constant. -/
def Tmpl.apart (t : Tmpl) : Bool :=
  (List.range t.defs.size).all (fun d => (t.defs.getD d default).fits t.defs && (t.defs.getD d default).apart t.defs) &&
    ((t.defs.getD t.root default).spans t.defs).all fun s => decide (s.2 ≤ t.constPos)

end Flock.Tmpl

namespace Flock.TmplApart

open Flock.Tmpl Flock.TmplOrder Flock.TmplRows

/-! ## Ranges -/

/-- Column `x` is in range `s` at offset `o`. -/
abbrev In (o : Nat) (s : Nat × Nat) (x : Nat) : Prop := o + s.1 ≤ x ∧ x < o + s.2

/-- Two ranges share no column. -/
abbrev Apart (a b : Nat × Nat) : Prop := a.2 ≤ b.1 ∨ b.2 ≤ a.1

theorem ascend_head : ∀ {a : Nat × Nat} {l : List (Nat × Nat)}, (∀ s ∈ a :: l, s.1 < s.2) → ascend (a :: l) = true →
    ∀ b ∈ l, a.2 ≤ b.1
  | _, [], _, _, _, hb => by cases hb
  | a, b :: rest, hlt, h, c, hc => by
    simp only [ascend, Bool.and_eq_true, decide_eq_true_eq] at h
    rcases List.mem_cons.1 hc with e | hc
    · subst e; exact h.1
    · have := ascend_head (a := b) (l := rest) (fun s hs => hlt s (List.mem_cons.2 (Or.inr hs))) h.2 c hc
      have := hlt b (List.mem_cons.2 (Or.inr (List.mem_cons.2 (Or.inl rfl))))
      omega

theorem ascend_pairwise : ∀ {l : List (Nat × Nat)}, (∀ s ∈ l, s.1 < s.2) → ascend l = true → l.Pairwise Apart
  | [], _, _ => List.Pairwise.nil
  | a :: l, hlt, h => by
    refine List.Pairwise.cons (fun b hb => Or.inl (ascend_head hlt h b hb)) (ascend_pairwise (fun s hs => hlt s
      (List.mem_cons.2 (Or.inr hs))) ?_)
    cases l with
    | nil => rfl
    | cons b rest => simp only [ascend, Bool.and_eq_true] at h; exact h.2

theorem apart_symm {a b : Nat × Nat} (h : Apart a b) : Apart b a := h.symm

/-- A Definition's ranges, apart. -/
theorem spans_pairwise {defs : Array Def} {df : Def} (h : df.apart defs = true) : (df.spans defs).Pairwise Apart := by
  unfold Def.apart at h
  simp only [Bool.and_eq_true] at h
  have hlt : ∀ s ∈ (df.spans defs).mergeSort (fun a b => decide (a.1 ≤ b.1)), s.1 < s.2 := fun s hs => by
    have := (List.mem_filter.1 (List.mem_mergeSort.1 hs)).2
    simpa using this
  exact (ascend_pairwise hlt h.1).perm (List.mergeSort_perm _ _) apart_symm

/-! ## Unwritten columns, and folds over writes apart -/

/-- Column `x` is unwritten and inside the unit. -/
abbrev Free (x : Nat) (acc : Acc) : Prop := acc.set.getD x true = false

theorem write_free {acc : Acc} {c : Nat} (a b : Array Nat) (h : Free c acc) :
    ∃ acc', write acc c a b = .ok acc' ∧ ∀ x, x ≠ c → Free x acc → Free x acc' := by
  refine ⟨{ ra := acc.ra.set! c a, rb := acc.rb.set! c b, set := acc.set.set! c true }, ?_, fun x hx hf => ?_⟩
  · unfold write; rw [ite_eq_right_iff.2 (by simp only [Free] at h; simp [h])]; rfl
  · show (acc.set.set! c true).getD x true = false
    rw [getD_set!_ne _ _ _ hx]; exact hf

/-- **A fold whose steps write apart succeeds**: each step succeeds when its own columns are unwritten and writes only
there, so steps whose columns no two share each find theirs unwritten. -/
theorem foldlM_apart {α σ : Type} {f : σ → α → Except String σ} (M : α → Prop) (R : α → Nat → Prop)
    (F : σ → Nat → Prop)
    (step : ∀ a s, M a → (∀ x, R a x → F s x) → ∃ s', f s a = .ok s' ∧ ∀ x, ¬ R a x → F s x → F s' x) :
    ∀ {l : List α}, (∀ a ∈ l, M a) → l.Pairwise (fun a b => ∀ x, R a x → ¬ R b x) → ∀ s,
      (∀ a ∈ l, ∀ x, R a x → F s x) → ∃ s', l.foldlM f s = .ok s' ∧ ∀ x, (∀ a ∈ l, ¬ R a x) → F s x → F s' x
  | [], _, _, s, _ => ⟨s, rfl, fun _ _ h => h⟩
  | a :: l, hM, hp, s, hf => by
    obtain ⟨s1, h1, k1⟩ := step a s (hM a (List.mem_cons.2 (Or.inl rfl))) (hf a (List.mem_cons.2 (Or.inl rfl)))
    obtain ⟨hpa, hpl⟩ := List.pairwise_cons.1 hp
    obtain ⟨s2, h2, k2⟩ := foldlM_apart M R F step (fun b hb => hM b (List.mem_cons.2 (Or.inr hb))) hpl s1
      (fun b hb x hx => k1 x (hpa b hb x · hx |>.elim) (hf b (List.mem_cons.2 (Or.inr hb)) x hx))
    refine ⟨s2, by rw [List.foldlM_cons, h1]; exact h2, fun x hx h => ?_⟩
    exact k2 x (fun b hb => hx b (List.mem_cons.2 (Or.inr hb))) (k1 x (hx a (List.mem_cons.2 (Or.inl rfl))) h)

/-! ## Copies -/

/-- A column outside a Definition's columns is in none of its ranges. -/
theorem outside_spans {defs : Array Def} {df : Def} {o x : Nat} (hf : df.fits defs = true) (hx : x < o ∨ o + df.cols ≤ x) :
    ∀ s ∈ df.spans defs, ¬ In o s x := by
  unfold Def.fits at hf
  simp only [Bool.and_eq_true, List.all_eq_true, decide_eq_true_eq] at hf
  intro s hs hin
  unfold Def.spans at hs
  obtain ⟨hs, -⟩ := List.mem_filter.1 hs
  rcases List.mem_append.1 hs with hs | hs
  · obtain ⟨r, hr, rfl⟩ := List.mem_map.1 hs
    have := hf.1 r hr
    simp only [In] at hin; omega
  · obtain ⟨c, hc, rfl⟩ := List.mem_map.1 hs
    have := hf.2 c hc
    simp only [In] at hin; omega

/-- **A copy whose columns are all unwritten succeeds, and writes only inside its rows' and calls' ranges.** -/
theorem place_ok {defs : Array Def} {k : Nat}
    (hwf : ∀ d, d < defs.size → (defs.getD d default).wellFormed defs d = true)
    (hfits : ∀ d, d < defs.size → (defs.getD d default).fits defs = true)
    (hapart : ∀ d, d < defs.size → (defs.getD d default).apart defs = true) (fuel : Nat) :
    ∀ {d}, d < fuel → d < defs.size → ∀ o inp acc, (∀ x, o ≤ x → x < o + (defs.getD d default).cols → Free x acc) →
      ∃ acc', place defs k fuel d o inp acc = .ok acc' ∧
        ∀ x, (∀ s ∈ (defs.getD d default).spans defs, ¬ In o s x) → Free x acc → Free x acc' := by
  induction fuel with
  | zero => intro d h; omega
  | succ fuel ih =>
    intro d hdf hd o inp acc hfree
    have hdf' : defs[d]? = some (defs.getD d default) := by simp [Array.getD_eq_getD_getElem?, hd]
    have hw := hwf d hd
    have hf := hfits d hd
    have hp := spans_pairwise (hapart d hd)
    have hap := hapart d hd
    generalize hdfe : defs.getD d default = df at hw hf hp hap hfree hdf' ⊢
    unfold Def.fits at hf
    simp only [Bool.and_eq_true, List.all_eq_true, decide_eq_true_eq] at hf
    unfold Def.wellFormed at hw
    simp only [Bool.and_eq_true, List.all_eq_true, decide_eq_true_eq, beq_iff_eq] at hw
    unfold Def.apart at hap
    simp only [Bool.and_eq_true, List.all_eq_true, Bool.or_eq_true, decide_eq_true_eq] at hap
    have hcopies := hap.2
    unfold Def.spans at hp
    rw [List.pairwise_filter, List.pairwise_append, List.pairwise_map, List.pairwise_map] at hp
    obtain ⟨hpr, hpc, hprc⟩ := hp
    -- a range of `df`'s, nonempty or not, and a column in it
    let rowR := fun (r : Nat × Array Nat × Array Nat) (x : Nat) => x = o + r.1
    let span := fun (c : Call) => (c.base, c.base + (c.count - 1) * c.stride + (defs.getD c.def_ default).cols)
    let callR := fun (c : Call) (x : Nat) => In o (span c) x
    -- the rows
    obtain ⟨acc1, h1, k1⟩ := foldlM_apart (f := fun acc r =>
        write acc (o + r.1) (Flock.Tmpl.canon (r.2.1.map (df.ref o inp k))) (Flock.Tmpl.canon (r.2.2.map (df.ref o inp k))))
      (fun r => r ∈ df.rows.toList) rowR (fun s x => Free x s)
      (fun r s _ hs => by
        obtain ⟨s', e, kk⟩ := write_free (Flock.Tmpl.canon (r.2.1.map (df.ref o inp k)))
          (Flock.Tmpl.canon (r.2.2.map (df.ref o inp k))) (hs _ rfl)
        exact ⟨s', e, fun x hx => kk x hx⟩)
      (fun r hr => hr)
      (hpr.imp fun {a b} h x ha hb => by
        simp only [rowR] at ha hb
        have := h (by simp) (by simp)
        simp only [Apart] at this; omega)
      acc (fun r hr x hx => hfree x (by simp only [rowR] at hx; omega) (by simp only [rowR] at hx; have := hf.1 r hr; omega))
    -- the calls, each a run of copies
    have hcall : ∀ c s, c ∈ df.calls.toList → (∀ x, callR c x → Free x s) →
        ∃ s', (List.range c.count).foldlM (fun acc j =>
          place defs k fuel c.def_ (o + c.base + j * c.stride) (c.bound (df.ref o inp k) j) acc) s = .ok s' ∧
          ∀ x, ¬ callR c x → Free x s → Free x s' := by
      intro c s hc hs
      have hcd := (hw.2 c hc).1.1
      have hce : c.def_ < defs.size := by omega
      let copyR := fun (j x : Nat) => j < c.count ∧ o + c.base + j * c.stride ≤ x ∧
        x < o + c.base + j * c.stride + (defs.getD c.def_ default).cols
      have inSpan : ∀ j, j < c.count → ∀ x, copyR j x → callR c x := fun j hj x hx => by
        have hm : j * c.stride ≤ (c.count - 1) * c.stride := Nat.mul_le_mul (by omega) (Nat.le_refl _)
        simp only [copyR] at hx; simp only [callR, In, span]; omega
      obtain ⟨s', e, kk⟩ := foldlM_apart (f := fun acc j =>
          place defs k fuel c.def_ (o + c.base + j * c.stride) (c.bound (df.ref o inp k) j) acc)
        (fun j => j < c.count) copyR (fun s x => Free x s)
        (fun j s hj hs => by
          obtain ⟨s', e, kk⟩ := ih (d := c.def_) (by omega) hce _ _ s fun x h1 h2 => hs x ⟨hj, h1, h2⟩
          exact ⟨s', e, fun x hx => kk x (outside_spans (hfits _ hce) (by simp only [copyR] at hx; omega))⟩)
        (fun j hj => List.mem_range.1 hj)
        ((List.pairwise_lt_range (n := c.count)).imp fun {a b} hab x ha hb => by
          simp only [copyR] at ha hb
          rcases hcopies c hc with h1 | h1
          · omega
          · have : a * c.stride + c.stride ≤ b * c.stride := by
              rw [← Nat.succ_mul]; exact Nat.mul_le_mul (by omega) (Nat.le_refl _)
            omega)
        s (fun j hj x hx => hs x (inSpan j (List.mem_range.1 hj) x hx))
      exact ⟨s', e, fun x hx => kk x fun j hj hc' => hx (inSpan j (List.mem_range.1 hj) x hc')⟩
    obtain ⟨acc2, h2, k2⟩ := foldlM_apart (f := fun acc c => (List.range c.count).foldlM (fun acc j =>
          place defs k fuel c.def_ (o + c.base + j * c.stride) (c.bound (df.ref o inp k) j) acc) acc)
      (fun c => c ∈ df.calls.toList) callR (fun s x => Free x s)
      (fun c s hc hs => hcall c s hc hs)
      (fun c hc => hc)
      (hpc.imp fun {a b} h x ha hb => by
        simp only [callR, In, span] at ha hb
        have := h (by simp only [decide_eq_true_eq]; omega) (by simp only [decide_eq_true_eq]; omega)
        simp only [Apart] at this; omega)
      acc1 (fun c hc x hx => by
        have := hf.2 c hc
        simp only [callR, In, span] at hx
        refine k1 x (fun r hr e => ?_) (hfree x (by omega) (by omega))
        simp only [rowR] at e
        have := hprc _ (List.mem_map.2 ⟨r, hr, rfl⟩) _ (List.mem_map.2 ⟨c, hc, rfl⟩) (by simp)
          (by simp only [decide_eq_true_eq]; omega)
        simp only [Apart] at this; omega)
    refine ⟨acc2, ?_, fun x hx h => ?_⟩
    · simp only [place, hdf']
      rw [h1]
      exact h2
    · have notIn : ∀ s, s ∈ df.spans defs → ¬ In o s x := hx
      refine k2 x (fun c hc hin => ?_) (k1 x (fun r hr e => ?_) h)
      · refine notIn (span c) ?_ hin
        unfold Def.spans
        refine List.mem_filter.2 ⟨List.mem_append.2 (Or.inr (List.mem_map.2 ⟨c, hc, rfl⟩)), ?_⟩
        simp only [callR, In, span] at hin
        simp only [decide_eq_true_eq, span]; omega
      · refine notIn (r.1, r.1 + 1) ?_ ⟨by simp only [rowR] at e; omega, by simp only [rowR] at e; omega⟩
        unfold Def.spans
        exact List.mem_filter.2 ⟨List.mem_append.2 (Or.inl (List.mem_map.2 ⟨r, hr, rfl⟩)), by simp⟩

/-! ## The unit -/

/-- **A well-formed template whose rows and copies are apart unrolls** (`Tmpl.rows` succeeds). -/
theorem rows_ok {t : Tmpl} (hwf : t.wellFormed = true) (ha : t.apart = true) : ∃ acc, t.rows = .ok acc := by
  unfold Tmpl.wellFormed at hwf
  simp only [Bool.and_eq_true, List.all_eq_true, List.mem_range, decide_eq_true_eq, beq_iff_eq] at hwf
  obtain ⟨⟨⟨hroot, -⟩, hcols⟩, hdefs⟩ := hwf
  unfold Tmpl.apart at ha
  simp only [Bool.and_eq_true, List.all_eq_true, List.mem_range, decide_eq_true_eq] at ha
  obtain ⟨hd, hk⟩ := ha
  have hempty : ∀ x, x < (t.defs.getD t.root default).cols →
      Free x (Acc.empty (t.defs.getD t.root default).cols) := fun x hx => by
    simp [Free, Acc.empty, Array.getD_eq_getD_getElem?, Array.getElem?_replicate] at hx ⊢; simp [hx]
  obtain ⟨acc1, h1, k1⟩ := place_ok (k := t.constPos) hdefs (fun d h => (hd d h).1) (fun d h => (hd d h).2)
    (t.defs.size + 1) (d := t.root) (by omega) hroot 0 [] _ (fun x _ hx => hempty x (by omega))
  have hc : Free t.constPos acc1 := k1 _ (fun s hs hin => by have := hk s hs; simp only [In] at hin; omega)
    (hempty _ (by unfold Tmpl.constPos; omega))
  obtain ⟨acc, e, -⟩ := write_free #[t.constPos] #[t.constPos] hc
  exact ⟨acc, by unfold Tmpl.rows; rw [h1]; exact e⟩

end Flock.TmplApart
