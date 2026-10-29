import Proofs.Flock.Soundness.TmplFoldTree

/-!
# A class's fold, valued

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
    rw [Array.size_modify]
    exact huk _ htl
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
  have hterm : ∀ t, of (((o.u.modify t0 (·.modify 0 (· + w))).getD t #[]).getD 0 0) * iv t =
      of ((o.u.getD t #[]).getD 0 0) * iv t + if t = t0 then of w * iv t0 else 0 := by
    intro t
    rw [modU_get o.u ht' hk' w, if_pos (rfl : (0 : ℕ) = 0), add_mul]
    split_ifs with h
    · subst h; rfl
    · rw [zero_mul]
  have hz : ∑ t ∈ Finset.range ins, (if t = t0 then of w * iv t0 else 0) = of w * iv t0 := by
    rw [Finset.sum_eq_single t0 (fun t _ h => if_neg h) (fun h => absurd (Finset.mem_range.mpr ht) h), if_pos rfl]
  show gl * of o.s + gc * of o.c + ∑ t ∈ Finset.range ins,
      of (((o.u.modify t0 (·.modify 0 (· + w))).getD t #[]).getD 0 0) * iv t = _
  rw [Finset.sum_congr rfl fun t _ => hterm t, Finset.sum_add_distrib, hz]
  unfold sumVal
  ring

/-- **A summed class's reference**: `v` read with weight `w` adds `w` times its factor, an own column's as `gOwn` adds
it (`X`), an input's `iv`, the constant's `gc`; `p` is untouched. -/
theorem ref_sum (pt : FoldPt) {ins cols v : ℕ} (gl gc : GF128) (iv : ℕ → GF128) (w : F128) (gOwn : ℕ → Out → Out)
    (X : GF128) {o : Out} (hsh : Shape ins 1 o)
    (hown : v < cols → Shape ins 1 (gOwn v o) ∧ (gOwn v o).p = o.p ∧
      sumVal ins gl gc iv (gOwn v o) = sumVal ins gl gc iv o + X) :
    Shape ins 1 (o.ref pt true cols ins v 0 w gOwn) ∧ (o.ref pt true cols ins v 0 w gOwn).p = o.p ∧
      sumVal ins gl gc iv (o.ref pt true cols ins v 0 w gOwn) = sumVal ins gl gc iv o +
        (if v < cols then X else if v < cols + ins then of w * iv (v - cols) else gc * of w) := by
  by_cases h1 : v < cols
  · have e : o.ref pt true cols ins v 0 w gOwn = gOwn v o := by
      unfold Out.ref; rw [if_pos h1]
    rw [e, if_pos h1]
    exact hown h1
  · by_cases h2 : v < cols + ins
    · have e : o.ref pt true cols ins v 0 w gOwn = { o with u := o.u.modify (v - cols) (·.modify 0 (· + w)) } := by
        unfold Out.ref; rw [if_neg h1, if_pos h2]
      rw [e, if_neg h1, if_pos h2]
      obtain ⟨h3, h4⟩ := sum_u (t0 := v - cols) (by omega) gl gc iv w hsh
      exact ⟨h3, rfl, h4⟩
    · have e : o.ref pt true cols ins v 0 w gOwn = { o with c := o.c + w } := by
        unfold Out.ref; rw [if_neg h1, if_neg h2, if_pos rfl]
      rw [e, if_neg h1, if_neg h2]
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

/-! ## A summed class's position split: `q = a + 2^β · y`, `a < 2^β`, `β ≥ 6` -/

theorem halves_getD (r : Array F128) {lo : ℕ} (hlo : lo ≤ r.size) :
    (Tmpl.halves r).getD lo (0, #[], #[]) =
      ((r.size - lo) / 2, (Array.range (2 ^ ((r.size - lo) / 2))).map (Tmpl.eqAt r lo ((r.size - lo) / 2)),
        (Array.range (2 ^ (r.size - lo - (r.size - lo) / 2))).map
          (Tmpl.eqAt r (lo + (r.size - lo) / 2) (r.size - lo - (r.size - lo) / 2))) := by
  simp [Tmpl.halves, Nat.lt_succ_of_le hlo]

/-- Past its coordinates, `eq(r[lo ..], u)` read from the halves is 0. -/
theorem eqTail_out (r : Array F128) (lo u : ℕ) (hlo : lo ≤ r.size) (hu : 2 ^ (r.size - lo) ≤ u) :
    of (Tmpl.eqTail (Tmpl.halves r) lo u) = 0 := by
  have hh : (r.size - lo) / 2 ≤ r.size - lo := Nat.div_le_self _ _
  have h2 : ¬ u / 2 ^ ((r.size - lo) / 2) < 2 ^ (r.size - lo - (r.size - lo) / 2) := by
    rw [Nat.div_lt_iff_lt_mul (Nat.two_pow_pos _), ← Nat.pow_add, Nat.sub_add_cancel hh]
    omega
  unfold Tmpl.eqTail
  rw [halves_getD r hlo]
  dsimp only
  simp only [Array.getD_eq_getD_getElem?, Array.getElem?_map, Array.getElem?_range, h2, ↓reduceIte,
    Option.map_none, Option.getD_none]
  rw [of_mul, of_zero', mul_zero]

/-- `eqTail_split` for every high part `v`. -/
theorem eqTail_split' (r : Array F128) (k u v : ℕ) (hk : k ≤ r.size) (hu : u < 2 ^ k) :
    of (Tmpl.eqTail (Tmpl.halves r) 0 (u + 2 ^ k * v)) =
      of (Tmpl.eqAt r 0 k u) * of (Tmpl.eqTail (Tmpl.halves r) k v) := by
  by_cases hv : v < 2 ^ (r.size - k)
  · exact eqTail_split r k u v hk hu hv
  · rw [eqTail_out r k v hk (by omega), mul_zero, eqTail_out r 0 _ (Nat.zero_le _)]
    calc 2 ^ (r.size - 0) = 2 ^ k * 2 ^ (r.size - k) := by rw [← pow_add]; congr 1; omega
      _ ≤ 2 ^ k * v := Nat.mul_le_mul_left _ (by omega)
      _ ≤ u + 2 ^ k * v := Nat.le_add_left _ _

theorem pow_six (β : ℕ) (hβ : 6 ≤ β) : 2 ^ β = 64 * 2 ^ (β - 6) := by
  rw [show (64 : ℕ) = 2 ^ 6 from rfl, ← pow_add]; congr 1; omega

theorem split_mod (β a y : ℕ) (hβ : 6 ≤ β) : (a + 2 ^ β * y) % 64 = a % 64 := by
  rw [pow_six β hβ, mul_assoc, Nat.add_mul_mod_self_left]

theorem split_div (β a y : ℕ) (hβ : 6 ≤ β) : (a + 2 ^ β * y) / 64 = a / 64 + 2 ^ (β - 6) * y := by
  rw [pow_six β hβ, mul_assoc, Nat.add_mul_div_left _ _ (by norm_num : 0 < 64)]

theorem split_lo (β a : ℕ) (hβ : 6 ≤ β) (ha : a < 2 ^ β) : a / 64 < 2 ^ (β - 6) := by
  rw [Nat.div_lt_iff_lt_mul (by norm_num : 0 < 64), mul_comm, ← pow_six β hβ]
  exact ha

/-- A summed class's low part's `E` factor, `lw[a mod 64] · eq(ρ[.. β−6], ⌊a/64⌋)`. -/
def eLow (pt : FoldPt) (β a : ℕ) : F128 := pt.lw.getD (a % 64) 0 * Tmpl.eqAt pt.rho 0 (β - 6) (a / 64)

/-- A summed class's low part's slot-`s` factor, `[a mod 64 = s] · eq(r[.. β−6], ⌊a/64⌋)`. -/
noncomputable def gLow (pt : FoldPt) (s β a : ℕ) : GF128 :=
  if a % 64 = s then of (Tmpl.eqAt pt.r 0 (β - 6) (a / 64)) else 0

theorem e_split (pt : FoldPt) (hT : pt.rhoT = Tmpl.halves pt.rho) {β a : ℕ} (hβ : 6 ≤ β) (hk : β - 6 ≤ pt.rho.size)
    (ha : a < 2 ^ β) (y : ℕ) :
    of (pt.e (a + 2 ^ β * y)) = of (eLow pt β a) * of (Tmpl.eqTail pt.rhoT (β - 6) y) := by
  unfold FoldPt.e eLow
  rw [split_mod β a y hβ, split_div β a y hβ, of_mul, hT, eqTail_split' pt.rho _ _ _ hk (split_lo β a hβ ha), of_mul,
    mul_assoc]

theorem g_split (pt : FoldPt) (hT : pt.rT = Tmpl.halves pt.r) {s β a : ℕ} (hβ : 6 ≤ β) (hk : β - 6 ≤ pt.r.size)
    (ha : a < 2 ^ β) (y : ℕ) :
    Gs pt s (a + 2 ^ β * y) = gLow pt s β a * of (Tmpl.eqTail pt.rT (β - 6) y) := by
  unfold Gs gLow FoldPt.g
  rw [split_mod β a y hβ, split_div β a y hβ]
  split_ifs
  · rw [hT, eqTail_split' pt.r _ _ _ hk (split_lo β a hβ ha)]
  · rw [zero_mul]

/-! ## Loops whose every step adds a known term -/

theorem forIn_list_sum {α β : Type} (S : β → GF128) (Q : β → Prop) (g : α → GF128) :
    ∀ (l : List α) {f : α → β → V (ForInStep β)} {b b' : β}, forIn l b f = .ok b' → Q b →
      (∀ a ∈ l, ∀ b r, f a b = .ok r → Q b → ∃ b', r = .yield b' ∧ Q b' ∧ S b' = S b + g a) →
      Q b' ∧ S b' = S b + (l.map g).sum
  | [], f, b, b', h, hq, _ => by
    rw [List.forIn_nil] at h
    cases h
    exact ⟨hq, by rw [List.map_nil, List.sum_nil, add_zero]⟩
  | a :: l, f, b, b', h, hq, hf => by
    rw [List.forIn_cons] at h
    obtain ⟨r, hr, h⟩ := except_bind_ok h
    obtain ⟨b1, rfl, hq1, hs1⟩ := hf a (List.mem_cons_self ..) b r hr hq
    obtain ⟨hq2, hs2⟩ := forIn_list_sum S Q g l h hq1 (fun a ha => hf a (List.mem_cons_of_mem _ ha))
    exact ⟨hq2, by rw [hs2, hs1, List.map_cons, List.sum_cons, add_assoc]⟩

theorem forIn_array_sum {α β : Type} (S : β → GF128) (Q : β → Prop) (g : α → GF128) {l : Array α}
    {f : α → β → V (ForInStep β)} {b b' : β} (h : forIn l b f = .ok b') (hq : Q b)
    (hf : ∀ a ∈ l, ∀ b r, f a b = .ok r → Q b → ∃ b', r = .yield b' ∧ Q b' ∧ S b' = S b + g a) :
    Q b' ∧ S b' = S b + (l.toList.map g).sum := by
  rw [← Array.forIn_toList] at h
  exact forIn_list_sum S Q g l.toList h hq fun a ha => hf a (Array.mem_toList_iff.mp ha)

theorem forIn_range_sum {β : Type} (S : β → GF128) (Q : β → Prop) (g : ℕ → GF128) {n : ℕ}
    {f : ℕ → β → V (ForInStep β)} {b b' : β} (h : forIn [:n] b f = .ok b') (hq : Q b)
    (hf : ∀ k < n, ∀ b r, f k b = .ok r → Q b → ∃ b', r = .yield b' ∧ Q b' ∧ S b' = S b + g k) :
    Q b' ∧ S b' = S b + ∑ k ∈ Finset.range n, g k := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range'] at h
  have e : List.range' 0 [:n].size 1 = List.range n := by
    rw [List.range_eq_range']; simp [Std.Legacy.Range.size]
  rw [e] at h
  rw [← list_range_sum]
  exact forIn_list_sum S Q g _ h hq fun k hk => hf k (List.mem_range.mp hk)

theorem foldl_sum {α β : Type} (S : β → GF128) (Q : β → Prop) (g : α → GF128) {f : β → α → β} :
    ∀ (l : List α) (b : β), Q b → (∀ a ∈ l, ∀ b, Q b → Q (f b a) ∧ S (f b a) = S b + g a) →
      Q (l.foldl f b) ∧ S (l.foldl f b) = S b + (l.map g).sum
  | [], b, hq, _ => ⟨hq, by rw [List.foldl_nil, List.map_nil, List.sum_nil, add_zero]⟩
  | a :: l, b, hq, hf => by
    obtain ⟨hq1, hs1⟩ := hf a (List.mem_cons_self ..) b hq
    obtain ⟨hq2, hs2⟩ := foldl_sum S Q g l (f b a) hq1 (fun a ha => hf a (List.mem_cons_of_mem _ ha))
    exact ⟨hq2, by rw [List.foldl_cons, hs2, hs1, List.map_cons, List.sum_cons, add_assoc]⟩

theorem pzero_zipAdd {a b : Array F128} (ha : a.size = 64) (hb : b.size = 64) (h1 : ∀ i < 64, of (a.getD i 0) = 0)
    (h2 : ∀ i < 64, of (b.getD i 0) = 0) : ∀ i < 64, of (((a.zip b).map fun (x, y) => x + y).getD i 0) = 0 :=
  fun i hi => by rw [getD_zipAdd _ _ ha hb hi, h1 i hi, h2 i hi, add_zero]

end FlockSoundness.TmplFold
