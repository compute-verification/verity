import Proofs.Flock.Soundness.TmplFoldTree

/-!
# A class's fold, valued (PRF-133, part 3, stage 2)

`foldDef` builds a class's `Out` by `Out.ref` and `Out.add` steps. Each step adds a known term to one of two linear
valuations of `Out`, at slot `s`:

- `expVal` (explicit classes): `p[s]`, and each copy `k`'s open input `t` weighted by `iv t k` (the slot weight of the
  column its caller binds);
- `sumVal` (summed classes): `s` by `gl`, `c` by `gc` and each open input `t` by `iv t`, the factors of the copies'
  common low part.

`ref_exp`, `ref_sum`, `add_exp` and `add_sum` are those steps.
-/

open Flock Flock.Tmpl FlockLevel3 GF128

namespace FlockSoundness.TmplFold

/-- Slot `s`'s weight at position `q`: `[q mod 64 = s] · g(q)`. -/
noncomputable def Gs (pt : FoldPt) (s q : ℕ) : GF128 := if q % 64 = s then of (pt.g q) else 0

/-- An `Out`'s arrays: `p` of 64, and per input (`ins`) one entry per class copy (`nk`). -/
def Shape (ins nk : ℕ) (o : Out) : Prop :=
  o.p.size = 64 ∧ o.u.size = ins ∧ ∀ t < ins, (o.u.getD t #[]).size = nk

/-- An explicit class's value at slot `s`: `p[s]`, and copy `k`'s open input `t` weighted by `iv t k`. -/
noncomputable def expVal (s ins nk : ℕ) (iv : ℕ → ℕ → GF128) (o : Out) : GF128 :=
  of (o.p.getD s 0) + ∑ t ∈ Finset.range ins, ∑ k ∈ Finset.range nk, of ((o.u.getD t #[]).getD k 0) * iv t k

/-- A summed class's value: its own entries' sum `s` weighted by `gl`, the constant's `c` by `gc`, input `t` by `iv t`. -/
noncomputable def sumVal (ins : ℕ) (gl gc : GF128) (iv : ℕ → GF128) (o : Out) : GF128 :=
  gl * of o.s + gc * of o.c + ∑ t ∈ Finset.range ins, of ((o.u.getD t #[]).getD 0 0) * iv t

theorem getD_modify {α : Type} (xs : Array α) (f : α → α) {i : ℕ} (hi : i < xs.size) (j : ℕ) (d : α) :
    (xs.modify i f).getD j d = if i = j then f (xs.getD j d) else xs.getD j d := by
  by_cases h : i = j
  · subst h
    simp [Array.getD_eq_getD_getElem?, hi, Array.getElem_modify_self]
  · simp [Array.getD_eq_getD_getElem?, Array.getElem?_modify, h]

theorem getD_zipAdd (a b : Array F128) (ha : a.size = 64) (hb : b.size = 64) {s : ℕ} (hs : s < 64) :
    of (((a.zip b).map fun (x, y) => x + y).getD s 0) = of (a.getD s 0) + of (b.getD s 0) := by
  simp [Array.getD_eq_getD_getElem?, ha, hb, hs, Array.getElem_zip, of_add]

theorem size_zipAdd (a b : Array F128) (ha : a.size = 64) (hb : b.size = 64) :
    ((a.zip b).map fun (x, y) => x + y).size = 64 := by
  simp [ha, hb]

/-- Copy `k0`'s input `t0` gains `w`. -/
theorem modU_get (u : Array (Array F128)) {t0 k0 : ℕ} (ht : t0 < u.size) (hk : k0 < (u.getD t0 #[]).size) (w : F128)
    (t k : ℕ) :
    of (((u.modify t0 (·.modify k0 (· + w))).getD t #[]).getD k 0) =
      of ((u.getD t #[]).getD k 0) + if t = t0 then (if k = k0 then of w else 0) else 0 := by
  rw [getD_modify _ _ ht]
  by_cases h1 : t0 = t
  · subst h1
    rw [if_pos rfl, if_pos rfl, getD_modify _ _ hk]
    by_cases h2 : k0 = k
    · subst h2
      rw [if_pos rfl, if_pos rfl, of_add]
    · rw [if_neg h2, if_neg (Ne.symm h2), add_zero]
  · rw [if_neg h1, if_neg (Ne.symm h1), add_zero]

theorem modU_shape (u : Array (Array F128)) {ins nk t0 k0 : ℕ} (hu : u.size = ins)
    (huk : ∀ t < ins, (u.getD t #[]).size = nk) (ht : t0 < ins) (w : F128) :
    (u.modify t0 (·.modify k0 (· + w))).size = ins ∧
      ∀ t < ins, ((u.modify t0 (·.modify k0 (· + w))).getD t #[]).size = nk := by
  refine ⟨by simp [hu], fun t htl => ?_⟩
  rw [getD_modify _ _ (hu ▸ ht)]
  split_ifs with h
  · subst h
    simp [huk _ htl]
  · exact huk t htl

/-! ## Explicit classes -/

theorem exp_u {s ins nk t0 k0 : ℕ} (ht : t0 < ins) (hk : k0 < nk) (iv : ℕ → ℕ → GF128) (w : F128) {o : Out}
    (hsh : Shape ins nk o) :
    Shape ins nk { o with u := o.u.modify t0 (·.modify k0 (· + w)) } ∧
      expVal s ins nk iv { o with u := o.u.modify t0 (·.modify k0 (· + w)) } = expVal s ins nk iv o + of w * iv t0 k0 := by
  obtain ⟨hp, hu, huk⟩ := hsh
  have ht' : t0 < o.u.size := hu ▸ ht
  have hk' : k0 < (o.u.getD t0 #[]).size := (huk t0 ht).symm ▸ hk
  obtain ⟨hu', huk'⟩ := modU_shape o.u hu huk ht (k0 := k0) w
  refine ⟨⟨hp, hu', huk'⟩, ?_⟩
  have hz : ∑ t ∈ Finset.range ins, ∑ k ∈ Finset.range nk,
      (if t = t0 then (if k = k0 then of w else 0) else 0) * iv t k = of w * iv t0 k0 := by
    rw [Finset.sum_eq_single t0 (fun t _ h => Finset.sum_eq_zero fun k _ => by rw [if_neg h, zero_mul])
        (fun h => absurd (Finset.mem_range.mpr ht) h),
      Finset.sum_eq_single k0 (fun k _ h => by rw [if_pos rfl, if_neg h, zero_mul])
        (fun h => absurd (Finset.mem_range.mpr hk) h), if_pos rfl, if_pos rfl]
  show of (o.p.getD s 0) + ∑ t ∈ Finset.range ins, ∑ k ∈ Finset.range nk,
      of (((o.u.modify t0 (·.modify k0 (· + w))).getD t #[]).getD k 0) * iv t k = _
  simp only [modU_get o.u ht' hk' w, add_mul, Finset.sum_add_distrib, hz]
  unfold expVal
  rw [add_assoc]

theorem exp_addAt {s ins nk : ℕ} (hs : s < 64) (iv : ℕ → ℕ → GF128) (q : ℕ) (x : F128) {o : Out}
    (hsh : Shape ins nk o) :
    Shape ins nk { o with p := addAt o.p q x } ∧
      expVal s ins nk iv { o with p := addAt o.p q x } = expVal s ins nk iv o + if q % 64 = s then of x else 0 := by
  refine ⟨⟨by rw [addAt_size]; exact hsh.1, hsh.2.1, hsh.2.2⟩, ?_⟩
  show of ((addAt o.p q x).getD s 0) + ∑ t ∈ Finset.range ins, ∑ k ∈ Finset.range nk,
      of ((o.u.getD t #[]).getD k 0) * iv t k = _
  rw [addAt_get _ _ _ (hsh.1 ▸ hs)]
  unfold expVal
  ring

/-- **An explicit class's reference**: `v` read with weight `w` by copy `k` adds `w` times the slot weight of where it
reads, an own column's as `gOwn` adds it (`X`), an input's as its caller binds it, the constant's at `qc`. -/
theorem ref_exp (pt : FoldPt) {s ins nk cols v k : ℕ} (hs : s < 64) (hk : k < nk) (iv : ℕ → ℕ → GF128) (w : F128)
    (gOwn : ℕ → Out → Out) (X : GF128) {o : Out} (hsh : Shape ins nk o)
    (hown : v < cols → Shape ins nk (gOwn v o) ∧ expVal s ins nk iv (gOwn v o) = expVal s ins nk iv o + X) :
    Shape ins nk (o.ref pt false cols ins v k w gOwn) ∧
      expVal s ins nk iv (o.ref pt false cols ins v k w gOwn) = expVal s ins nk iv o +
        (if v < cols then X else if v < cols + ins then of w * iv (v - cols) k else of w * Gs pt s pt.qc) := by
  unfold Out.ref
  by_cases h1 : v < cols
  · simp only [h1, if_true]
    exact hown h1
  · simp only [h1, if_false]
    by_cases h2 : v < cols + ins
    · simp only [h2, if_true]
      exact exp_u (by omega) hk iv w hsh
    · simp only [h2, if_false, Bool.false_eq_true]
      obtain ⟨h3, h4⟩ := exp_addAt hs iv pt.qc (w * pt.g pt.qc) hsh
      refine ⟨h3, h4.trans ?_⟩
      unfold Gs
      split_ifs <;> simp [of_mul]

/-- The own column's step of an explicit class: `w · g(q)` at `q`. -/
theorem own_exp (pt : FoldPt) {s ins nk : ℕ} (hs : s < 64) (iv : ℕ → ℕ → GF128) (w : F128) (q : ℕ) {o : Out}
    (hsh : Shape ins nk o) :
    Shape ins nk { o with p := addAt o.p q (w * pt.g q) } ∧
      expVal s ins nk iv { o with p := addAt o.p q (w * pt.g q) } = expVal s ins nk iv o + of w * Gs pt s q := by
  obtain ⟨h3, h4⟩ := exp_addAt hs iv q (w * pt.g q) hsh
  refine ⟨h3, h4.trans ?_⟩
  unfold Gs
  split_ifs <;> simp [of_mul]

/-- A child class's `Out.add` into an explicit class adds its `p[s]`. -/
theorem add_exp {s ins nk : ℕ} (hs : s < 64) (iv : ℕ → ℕ → GF128) {o ch : Out} (ls lc : F128) (hsh : Shape ins nk o)
    (hch : ch.p.size = 64) :
    Shape ins nk (o.add ch ls lc) ∧ expVal s ins nk iv (o.add ch ls lc) = expVal s ins nk iv o + of (ch.p.getD s 0) := by
  refine ⟨⟨size_zipAdd _ _ hsh.1 hch, hsh.2.1, hsh.2.2⟩, ?_⟩
  show of (((o.p.zip ch.p).map fun (x, y) => x + y).getD s 0) + ∑ t ∈ Finset.range ins, ∑ k ∈ Finset.range nk,
      of ((o.u.getD t #[]).getD k 0) * iv t k = _
  rw [getD_zipAdd _ _ hsh.1 hch hs]
  unfold expVal
  ring

/-! ## Summed classes -/

/-- No entry of `p`: a summed class adds only to `s`, `c` and `u`. -/
def PZero (o : Out) : Prop := ∀ i < 64, of (o.p.getD i 0) = 0

theorem sum_u {ins t0 : ℕ} (ht : t0 < ins) (gl gc : GF128) (iv : ℕ → GF128) (w : F128) {o : Out}
    (hsh : Shape ins 1 o) :
    Shape ins 1 { o with u := o.u.modify t0 (·.modify 0 (· + w)) } ∧
      sumVal ins gl gc iv { o with u := o.u.modify t0 (·.modify 0 (· + w)) } = sumVal ins gl gc iv o + of w * iv t0 := by
  obtain ⟨hp, hu, huk⟩ := hsh
  have ht' : t0 < o.u.size := hu ▸ ht
  have hk' : 0 < (o.u.getD t0 #[]).size := by rw [huk t0 ht]; exact Nat.one_pos
  obtain ⟨hu', huk'⟩ := modU_shape o.u hu huk ht (k0 := 0) w
  refine ⟨⟨hp, hu', huk'⟩, ?_⟩
  have hz : ∑ t ∈ Finset.range ins, (if t = t0 then (if 0 = 0 then of w else 0) else 0) * iv t = of w * iv t0 := by
    rw [Finset.sum_eq_single t0 (fun t _ h => by rw [if_neg h, zero_mul])
        (fun h => absurd (Finset.mem_range.mpr ht) h), if_pos rfl, if_pos rfl]
  show gl * of o.s + gc * of o.c + ∑ t ∈ Finset.range ins,
      of (((o.u.modify t0 (·.modify 0 (· + w))).getD t #[]).getD 0 0) * iv t = _
  simp only [modU_get o.u ht' hk' w, add_mul, Finset.sum_add_distrib, hz]
  unfold sumVal
  rw [add_assoc]

/-- **A summed class's reference**: `v` read with weight `w` adds `w` times its factor, an own column's as `gOwn` adds
it (`X`), an input's `iv`, the constant's `gc`; `p` is untouched. -/
theorem ref_sum (pt : FoldPt) {ins cols v : ℕ} (gl gc : GF128) (iv : ℕ → GF128) (w : F128) (gOwn : ℕ → Out → Out)
    (X : GF128) {o : Out} (hsh : Shape ins 1 o)
    (hown : v < cols → Shape ins 1 (gOwn v o) ∧ (gOwn v o).p = o.p ∧
      sumVal ins gl gc iv (gOwn v o) = sumVal ins gl gc iv o + X) :
    Shape ins 1 (o.ref pt true cols ins v 0 w gOwn) ∧ (o.ref pt true cols ins v 0 w gOwn).p = o.p ∧
      sumVal ins gl gc iv (o.ref pt true cols ins v 0 w gOwn) = sumVal ins gl gc iv o +
        (if v < cols then X else if v < cols + ins then of w * iv (v - cols) else gc * of w) := by
  unfold Out.ref
  by_cases h1 : v < cols
  · simp only [h1, if_true]
    exact hown h1
  · simp only [h1, if_false]
    by_cases h2 : v < cols + ins
    · simp only [h2, if_true]
      obtain ⟨h3, h4⟩ := sum_u (by omega) gl gc iv w hsh
      exact ⟨h3, rfl, h4⟩
    · simp only [h2, if_false, ↓reduceIte]
      refine ⟨hsh, rfl, ?_⟩
      show gl * of o.s + gc * of (o.c + w) + ∑ t ∈ Finset.range ins, of ((o.u.getD t #[]).getD 0 0) * iv t = _
      rw [of_add]
      unfold sumVal
      ring

/-- The own column's step of a summed class: `w · gHi` into `s`. -/
theorem own_sum {ins : ℕ} (gl gc : GF128) (iv : ℕ → GF128) (x : F128) {o : Out} (hsh : Shape ins 1 o) :
    Shape ins 1 { o with s := o.s + x } ∧ ({ o with s := o.s + x } : Out).p = o.p ∧
      sumVal ins gl gc iv { o with s := o.s + x } = sumVal ins gl gc iv o + gl * of x := by
  refine ⟨hsh, rfl, ?_⟩
  show gl * of (o.s + x) + gc * of o.c + ∑ t ∈ Finset.range ins, of ((o.u.getD t #[]).getD 0 0) * iv t = _
  rw [of_add]
  unfold sumVal
  ring

/-- A child class's `Out.add` into a summed class adds its `s` and `c`, scaled by `ls` and `lc`. -/
theorem add_sum {ins : ℕ} (gl gc : GF128) (iv : ℕ → GF128) {o ch : Out} (ls lc : F128) (hsh : Shape ins 1 o)
    (hch : ch.p.size = 64) :
    Shape ins 1 (o.add ch ls lc) ∧ (o.add ch ls lc).p = (o.p.zip ch.p).map (fun (x, y) => x + y) ∧
      sumVal ins gl gc iv (o.add ch ls lc) =
        sumVal ins gl gc iv o + gl * (of ls * of ch.s) + gc * (of lc * of ch.c) := by
  refine ⟨⟨size_zipAdd _ _ hsh.1 hch, hsh.2.1, hsh.2.2⟩, rfl, ?_⟩
  show gl * of (o.s + ls * ch.s) + gc * of (o.c + lc * ch.c) +
      ∑ t ∈ Finset.range ins, of ((o.u.getD t #[]).getD 0 0) * iv t = _
  rw [of_add, of_add, of_mul, of_mul]
  unfold sumVal
  ring

theorem pzero_zipAdd {a b : Array F128} (ha : a.size = 64) (hb : b.size = 64) (h1 : ∀ i < 64, of (a.getD i 0) = 0)
    (h2 : ∀ i < 64, of (b.getD i 0) = 0) : ∀ i < 64, of (((a.zip b).map fun (x, y) => x + y).getD i 0) = 0 :=
  fun i hi => by rw [getD_zipAdd _ _ ha hb hi, h1 i hi, h2 i hi, add_zero]

end FlockSoundness.TmplFold
