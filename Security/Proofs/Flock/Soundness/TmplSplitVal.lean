import Proofs.Flock.Soundness.TmplSplitTab

/-!
# A class's table, valued at a chunk origin (PRF-133, part 3, stage 3)

`walk` folds a class at local offset `off` in chunks of `2^al`; placed at chunk origin `H` (so at `h = H · 2^al + off`
in `foldDef`'s terms), an entry at row chunk `cx` and column chunk `cy` has high factors `eq(ρ[L ..], H + cx)` and
`eq(r[L ..], H + cy)`, `L = β + al − 6` (`hiE`, `hiG`). `tabExp` and `tabSum` value a table as `expVal` and `sumVal`
value an `Out`.

- `e_chunk`, `g_chunk` (explicit) and `e_chunk_sum`, `g_chunk_sum` (summed): a position's `E` and slot weights are the
  low factors `walk` reads from its `eq` halves times the high factors.
- `ref_tabExp`, `ref_tabSum`: `Tab.ref` adds its reference's term.
-/

open Flock Flock.Tmpl FlockLevel3 GF128 Std

namespace FlockSoundness.TmplFold

/-- Chunk `H + c`'s high `E` factor, `eq(ρ[L ..], H + c)`. -/
noncomputable def hiE (pt : FoldPt) (L H c : ℕ) : GF128 := of (Tmpl.eqTail pt.rhoT L (H + c))

/-- Chunk `H + c`'s high slot factor, `eq(r[L ..], H + c)`. -/
noncomputable def hiG (pt : FoldPt) (L H c : ℕ) : GF128 := of (Tmpl.eqTail pt.rT L (H + c))

/-- An explicit class's table at chunk origin `H`, slot `s`: its own entries at `s` by their chunks' high factors, the
constant's by its row chunk's, and copy `k`'s open input `t` by `iv t k`. -/
noncomputable def tabExp (pt : FoldPt) (s L H : ℕ) (iv : ℕ → ℕ → GF128) (T : Tab) : GF128 :=
  hsum T.P (fun e => if e.2.2 = s then hiE pt L H e.1 * hiG pt L H e.2.1 else 0) +
    hsum T.PC (fun e => if e.2 = s then hiE pt L H e.1 else 0) +
    hsum T.U (fun e => hiE pt L H e.2.2 * iv e.1 e.2.1)

/-- A summed class's table at chunk origin `H`: its own entries by `gl`, the constant's by `gc`, open input `t` by
`iv t`, each by its chunks' high factors. -/
noncomputable def tabSum (pt : FoldPt) (L H : ℕ) (gl gc : GF128) (iv : ℕ → GF128) (T : Tab) : GF128 :=
  gl * hsum T.S (fun e => hiE pt L H e.1 * hiG pt L H e.2) + gc * hsum T.C (fun c => hiE pt L H c) +
    hsum T.U (fun e => hiE pt L H e.2.2 * iv e.1)

/-! ## A position's factors, split at its chunk -/

theorem chunk_pos (β al a H P : ℕ) :
    a + 2 ^ β * (H * 2 ^ al + P) = (a + 2 ^ β * (P % 2 ^ al)) + 2 ^ (β + al) * (H + P / 2 ^ al) := by
  have h := Nat.mod_add_div P (2 ^ al)
  rw [Nat.pow_add]
  calc a + 2 ^ β * (H * 2 ^ al + P) = a + 2 ^ β * (H * 2 ^ al + (P % 2 ^ al + 2 ^ al * (P / 2 ^ al))) := by rw [h]
    _ = _ := by ring

theorem chunk_low {β al a P : ℕ} (ha : a < 2 ^ β) : a + 2 ^ β * (P % 2 ^ al) < 2 ^ (β + al) :=
  lane_lt ha (Nat.mod_lt _ (Nat.two_pow_pos _))

theorem e_chunk (pt : FoldPt) (hT : pt.rhoT = Tmpl.halves pt.rho) {β al a : ℕ} (h6 : 6 ≤ β + al)
    (hL : β + al - 6 ≤ pt.rho.size) (ha : a < 2 ^ β) (H P : ℕ) :
    of (pt.e (a + 2 ^ β * (H * 2 ^ al + P))) =
      of (pt.lw.getD ((a + 2 ^ β * (P % 2 ^ al)) % 64) 0 *
        (eqHalf pt.rho 0 (β + al - 6)).at ((a + 2 ^ β * (P % 2 ^ al)) / 64)) * hiE pt (β + al - 6) H (P / 2 ^ al) := by
  have hq := chunk_low (al := al) (P := P) ha
  unfold FoldPt.e hiE
  rw [chunk_pos, split_mod _ _ _ h6, split_div _ _ _ h6, of_mul, hT,
    eqTail_split' pt.rho _ _ _ hL (split_lo _ _ h6 hq), of_mul, of_mul,
    eqHalf_at _ _ _ _ (split_lo _ _ h6 hq), mul_assoc]

theorem g_chunk (pt : FoldPt) (hR : pt.rT = Tmpl.halves pt.r) {s β al a : ℕ} (h6 : 6 ≤ β + al)
    (hL : β + al - 6 ≤ pt.r.size) (ha : a < 2 ^ β) (H P : ℕ) :
    Gs pt s (a + 2 ^ β * (H * 2 ^ al + P)) =
      (if (a + 2 ^ β * (P % 2 ^ al)) % 64 = s then
        of ((eqHalf pt.r 0 (β + al - 6)).at ((a + 2 ^ β * (P % 2 ^ al)) / 64)) else 0) *
        hiG pt (β + al - 6) H (P / 2 ^ al) := by
  have hq := chunk_low (al := al) (P := P) ha
  unfold Gs FoldPt.g hiG
  rw [chunk_pos, split_mod _ _ _ h6, split_div _ _ _ h6]
  split_ifs
  · rw [hR, eqTail_split' pt.r _ _ _ hL (split_lo _ _ h6 hq), eqHalf_at _ _ _ _ (split_lo _ _ h6 hq)]
  · rw [zero_mul]

theorem e_chunk_sum (pt : FoldPt) (hT : pt.rhoT = Tmpl.halves pt.rho) {β al a : ℕ} (h6 : 6 ≤ β)
    (hL : β + al - 6 ≤ pt.rho.size) (ha : a < 2 ^ β) (H P : ℕ) :
    of (pt.e (a + 2 ^ β * (H * 2 ^ al + P))) =
      of (eLow pt β a) * of ((eqHalf pt.rho (β - 6) al).at (P % 2 ^ al)) * hiE pt (β + al - 6) H (P / 2 ^ al) := by
  have hP := Nat.mod_add_div P (2 ^ al)
  rw [e_split pt hT h6 (by omega) ha, hT,
    show H * 2 ^ al + P = P % 2 ^ al + 2 ^ al * (H + P / 2 ^ al) by
      calc H * 2 ^ al + P = H * 2 ^ al + (P % 2 ^ al + 2 ^ al * (P / 2 ^ al)) := by rw [hP]
        _ = _ := by ring,
    eqTail_split_at pt.rho _ _ _ _ (by omega) (Nat.mod_lt _ (Nat.two_pow_pos _)),
    eqHalf_at _ _ _ _ (Nat.mod_lt _ (Nat.two_pow_pos _)), show β - 6 + al = β + al - 6 by omega, mul_assoc]
  rfl

theorem g_chunk_sum (pt : FoldPt) (hR : pt.rT = Tmpl.halves pt.r) {s β al a : ℕ} (h6 : 6 ≤ β)
    (hL : β + al - 6 ≤ pt.r.size) (ha : a < 2 ^ β) (H P : ℕ) :
    Gs pt s (a + 2 ^ β * (H * 2 ^ al + P)) =
      gLow pt s β a * of ((eqHalf pt.r (β - 6) al).at (P % 2 ^ al)) * hiG pt (β + al - 6) H (P / 2 ^ al) := by
  have hP := Nat.mod_add_div P (2 ^ al)
  rw [g_split pt hR h6 (by omega) ha, hR,
    show H * 2 ^ al + P = P % 2 ^ al + 2 ^ al * (H + P / 2 ^ al) by
      calc H * 2 ^ al + P = H * 2 ^ al + (P % 2 ^ al + 2 ^ al * (P / 2 ^ al)) := by rw [hP]
        _ = _ := by ring,
    eqTail_split_at pt.r _ _ _ _ (by omega) (Nat.mod_lt _ (Nat.two_pow_pos _)),
    eqHalf_at _ _ _ _ (Nat.mod_lt _ (Nat.two_pow_pos _)), show β - 6 + al = β + al - 6 by omega, mul_assoc]
  rfl

/-! ## A reference's step -/

/-- **An explicit reference's step**: an own column adds at its slot, an input opens, the constant adds at `qc`'s slot. -/
theorem ref_tabExp (D : Def) (pt : FoldPt) (s L H A B : ℕ) (as : Array ℕ) (gL : Half) (pos : ℕ → ℕ)
    (iv : ℕ → ℕ → GF128) (o : Tab) (k cx : ℕ) (w : F128) (v : ℕ) :
    tabExp pt s L H iv (Tab.ref D pt false A B as gL pos o k cx w v) = tabExp pt s L H iv o +
      of w * hiE pt L H cx *
        (if v < D.cols then
          (if (as.getD k 0 + B * (pos v % A)) % 64 = s then
            of (gL.at ((as.getD k 0 + B * (pos v % A)) / 64)) * hiG pt L H (pos v / A) else 0)
        else if v < D.cols + D.ins then iv (v - D.cols) k
        else if pt.qc % 64 = s then of (pt.g pt.qc) else 0) := by
  unfold Tab.ref tabExp
  by_cases h1 : v < D.cols
  · simp only [h1, ↓reduceIte, Bool.false_eq_true, hsum_bump]
    split_ifs <;> simp only [of_mul] <;> ring
  · by_cases h2 : v < D.cols + D.ins
    · simp only [h1, h2, ↓reduceIte, Bool.false_eq_true, hsum_bump]
      ring
    · simp only [h1, h2, ↓reduceIte, Bool.false_eq_true, hsum_bump]
      split_ifs <;> simp only [of_mul] <;> ring

/-- **A summed reference's step**: an own column adds to the own entries, an input opens, the constant adds. -/
theorem ref_tabSum (D : Def) (pt : FoldPt) (L H A B : ℕ) (as : Array ℕ) (gL : Half) (pos : ℕ → ℕ)
    (gl gc : GF128) (iv : ℕ → GF128) (o : Tab) (k cx : ℕ) (w : F128) (v : ℕ) :
    tabSum pt L H gl gc iv (Tab.ref D pt true A B as gL pos o k cx w v) = tabSum pt L H gl gc iv o +
      of w * hiE pt L H cx *
        (if v < D.cols then gl * of (gL.at (pos v % A)) * hiG pt L H (pos v / A)
        else if v < D.cols + D.ins then iv (v - D.cols) else gc) := by
  unfold Tab.ref tabSum
  by_cases h1 : v < D.cols
  · simp only [h1, ↓reduceIte, hsum_bump, of_mul]
    ring
  · by_cases h2 : v < D.cols + D.ins
    · simp only [h1, h2, ↓reduceIte, hsum_bump]
      ring
    · simp only [h1, h2, ↓reduceIte, hsum_bump]
      ring

end FlockSoundness.TmplFold
