import Properties.Proofs.Nci.Novel

/-!
# Toward `Novel.ReadsExact`: what `Desc.analyze`'s pieces and segments name

`Novel.ReadsExact` says that the segments `Desc.analyze` gives a root node name exactly the gates outside the node that
its gates read. Its proof reads each piece, segment and sequence of segments as the list of positions it names in order
(`pos`, `posAll`), a position being a target (a parameter's or a node's leaves, or the activation's gates of one width)
and an index, and holds each of `Desc`'s operations on them to the positions it means: a slice of a piece is its
positions `[a, b)` (`slice_pos`).
-/

namespace NciCheck.Properties.SecurityProofs.ReadsExact

open NciCheck.Desc (Piece Seg PS Tgt)

theorem flatMap_range_length {α : Type} (f : Nat → List α) (L : Nat) (hf : ∀ r, (f r).length = L) :
    ∀ n, ((List.range n).flatMap f).length = n * L
  | 0 => by simp
  | n + 1 => by
    rw [List.range_succ, List.flatMap_append, List.flatMap_singleton, List.length_append,
      flatMap_range_length f L hf n, hf, Nat.succ_mul]

/-- A list of blocks of one length `L`: element `t` is element `t % L` of block `t / L`. -/
theorem flatMap_range_getElem? {α : Type} (f : Nat → List α) (L : Nat) (hf : ∀ r, (f r).length = L) :
    ∀ (n t : Nat), ((List.range n).flatMap f)[t]? = if t < n * L then (f (t / L))[t % L]? else none
  | 0, t => by simp
  | n + 1, t => by
    rw [List.range_succ, List.flatMap_append, List.flatMap_singleton]
    have hlen := flatMap_range_length f L hf n
    have hs : (n + 1) * L = n * L + L := Nat.succ_mul n L
    by_cases ht : t < n * L
    · rw [List.getElem?_append_left (by omega), flatMap_range_getElem? f L hf n t, if_pos ht,
        if_pos (by omega)]
    · rw [List.getElem?_append_right (by omega), hlen]
      have hL : 0 < L ∨ L = 0 := by omega
      rcases hL with hL | hL
      · by_cases ht' : t < (n + 1) * L
        · rw [if_pos ht']
          have hd := Nat.div_add_mod t L
          have hm := Nat.mod_lt t hL
          have h1 : t / L = n := by
            have hlo : n * L ≤ t := by omega
            have hhi : t < (n + 1) * L := ht'
            have e1 : n ≤ t / L := (Nat.le_div_iff_mul_le hL).2 hlo
            have e2 : t / L < n + 1 := (Nat.div_lt_iff_lt_mul hL).2 hhi
            omega
          rw [h1]
          rw [h1] at hd
          have : t - n * L = t % L := by
            have : L * n = n * L := Nat.mul_comm L n
            omega
          rw [this]
        · rw [if_neg ht']
          apply List.getElem?_eq_none
          rw [hf]; omega
      · subst hL
        simp only [Nat.mul_zero]
        apply List.getElem?_eq_none
        rw [hf]; omega

/-- A piece's positions in order: row `r`, column `t` is `(tgt, base + r·rowStride + t·stride)`. -/
def pos (q : Piece) : List (Tgt × Int) :=
  (List.range q.rows).flatMap fun (r : Nat) => (List.range q.len).map fun (t : Nat) =>
    (q.tgt, q.base + (r : Int) * q.rowStride + (t : Int) * q.stride)

/-- Position `t` of a piece: row `t / len`, column `t % len`. -/
def ppt (q : Piece) (t : Nat) : Int :=
  q.base + ((t / q.len : Nat) : Int) * q.rowStride + ((t % q.len : Nat) : Int) * q.stride

theorem pos_length (q : Piece) : (pos q).length = q.count := by
  unfold pos Piece.count
  rw [flatMap_range_length _ q.len (by simp)]

theorem pos_getElem? (q : Piece) (t : Nat) :
    (pos q)[t]? = if t < q.count then some (q.tgt, ppt q t) else none := by
  unfold pos Piece.count
  rw [flatMap_range_getElem? _ q.len (by simp)]
  split
  · rename_i h
    have hL : 0 < q.len := by
      rcases Nat.eq_zero_or_pos q.len with h0 | h0
      · rw [h0, Nat.mul_zero] at h; omega
      · exact h0
    rw [List.getElem?_map, List.getElem?_range (Nat.mod_lt t hL)]
    rfl
  · rfl


theorem pos_eq_map (q : Piece) : pos q = (List.range q.count).map fun t => (q.tgt, ppt q t) := by
  apply List.ext_getElem?
  intro t
  rw [pos_getElem?, List.getElem?_map]
  by_cases h : t < q.count
  · rw [if_pos h, List.getElem?_range h]; rfl
  · rw [if_neg h, List.getElem?_eq_none (by simp; omega)]; rfl

def posAll (ps : Array Piece) : List (Tgt × Int) := ps.toList.flatMap pos

/-- A one-row piece of `q`'s target, `len'` positions of `q` from `s` (all in row `s / q.len`): `q`'s positions
`[s, s + len')`. -/
theorem range'_getElem? (s n t : Nat) : (List.range' s n)[t]? = if t < n then some (s + t) else none := by
  by_cases h : t < n
  · rw [if_pos h, List.getElem?_eq_getElem (by simpa using h), List.getElem_range']; simp
  · rw [if_neg h, List.getElem?_eq_none (by simp; omega)]

theorem toNat_div_cast (a b : Nat) : (((a / b : Nat)) : Int) = ((a / b : Nat) : Int) := rfl

/-- A one-row piece of `q`'s target, `len'` positions of `q` from `s` (all in row `s / q.len`): `q`'s positions
`[s, s + len')`. -/
theorem pos_run (q : Piece) (r s e : Nat) (B : Int) (hB : B = q.base + (r : Int) * q.rowStride + (s : Int) * q.stride)
    (hse : s ≤ e) (he : e ≤ q.len) :
    pos ⟨q.tgt, B, q.stride, e - s, 1, 0⟩ =
      (List.range' (r * q.len + s) (e - s)).map fun t => (q.tgt, ppt q t) := by
  rw [pos_eq_map]
  apply List.ext_getElem?
  intro t
  simp only [Desc.Piece.count, List.getElem?_map, Nat.one_mul]
  rw [range'_getElem?]
  by_cases h : t < e - s
  · rw [List.getElem?_range h, if_pos h]
    simp only [Option.map_some, Option.some.injEq, Prod.mk.injEq, true_and]
    have hL : 0 < q.len := by omega
    have h1 : (r * q.len + s + t) / q.len = r := by
      rw [Nat.add_assoc, Nat.mul_comm, Nat.mul_add_div hL, Nat.div_eq_of_lt (by omega), Nat.add_zero]
    have h2 : (r * q.len + s + t) % q.len = s + t := by
      rw [Nat.add_assoc, Nat.mul_comm, Nat.mul_add_mod, Nat.mod_eq_of_lt (by omega)]
    unfold ppt
    simp only [Nat.mod_eq_of_lt (show t < e - s by omega), Nat.div_eq_of_lt (show t < e - s by omega)]
    rw [h1, h2, hB]
    push_cast
    rw [Int.add_mul]; omega
  · rw [List.getElem?_eq_none (by simp; omega), if_neg h]; rfl


/-- Whole rows `[lo, lo + n)` of `q`: its positions `[lo · len, (lo + n) · len)`. -/
theorem pos_rows (q : Piece) (lo n : Nat) :
    pos ⟨q.tgt, q.base + (lo : Int) * q.rowStride, q.stride, q.len, n, q.rowStride⟩ =
      (List.range' (lo * q.len) (n * q.len)).map fun t => (q.tgt, ppt q t) := by
  rw [pos_eq_map]
  apply List.ext_getElem?
  intro t
  simp only [Desc.Piece.count, List.getElem?_map]
  rw [range'_getElem?]
  by_cases h : t < n * q.len
  · rw [List.getElem?_range h, if_pos h]
    simp only [Option.map_some, Option.some.injEq, Prod.mk.injEq, true_and]
    have hL : 0 < q.len := by
      rcases Nat.eq_zero_or_pos q.len with h0 | h0
      · rw [h0, Nat.mul_zero] at h; omega
      · exact h0
    have h1 : (lo * q.len + t) / q.len = lo + t / q.len := by
      rw [Nat.mul_comm, Nat.mul_add_div hL]
    have h2 : (lo * q.len + t) % q.len = t % q.len := by
      rw [Nat.mul_comm, Nat.mul_add_mod]
    unfold ppt
    simp only
    rw [h1, h2]
    push_cast
    rw [Int.add_mul]; omega
  · rw [List.getElem?_eq_none (by simp; omega), if_neg h]; rfl

theorem posAll_append (a b : Array Piece) : posAll (a ++ b) = posAll a ++ posAll b := by
  simp [posAll, List.flatMap_append]

theorem posAll_singleton (q : Piece) : posAll #[q] = pos q := by simp [posAll]

theorem posAll_empty : posAll #[] = [] := rfl

theorem range'_split (a m b : Nat) (h1 : a ≤ m) (h2 : m ≤ b) :
    List.range' a (b - a) = List.range' a (m - a) ++ List.range' m (b - m) := by
  have := @List.range'_append a (m - a) (b - m) 1
  rw [show a + 1 * (m - a) = m by omega] at this
  rw [show b - a = (m - a) + (b - m) by omega, ← this]

theorem range'_join (x m n : Nat) (h : x ≤ m) :
    List.range' x (m - x) ++ List.range' m n = List.range' x (m + n - x) := by
  rw [range'_split x m (m + n) h (by omega), show m + n - m = n by omega]

/-- **A slice of a piece** is its positions `[a, b)`. -/
theorem slice_pos (q : Piece) (a b : Nat) (hb : b ≤ q.count) :
    posAll (q.slice a b) = (List.range' a (b - a)).map fun t => (q.tgt, ppt q t) := by
  unfold Desc.Piece.slice
  by_cases hab : b ≤ a
  · rw [if_pos hab, posAll_empty, show b - a = 0 by omega]; rfl
  rw [if_neg hab]
  have hcnt : q.count = q.rows * q.len := rfl
  have hL : 0 < q.len := by
    rcases Nat.eq_zero_or_pos q.len with h0 | h0
    · rw [hcnt, h0, Nat.mul_zero] at hb; omega
    · exact h0
  simp only [Desc.Piece.row, Desc.Piece.at]
  by_cases hr : q.rows ≤ 1
  · rw [if_pos hr, posAll_singleton]
    have hb' : b ≤ q.len := by
      have : q.rows * q.len ≤ 1 * q.len := Nat.mul_le_mul_right _ hr
      omega
    have := pos_run q 0 a b _ rfl (by omega) hb'
    simp only [Int.natCast_zero, Int.zero_mul, Int.add_zero, Nat.zero_mul, Nat.zero_add] at this
    exact this
  rw [if_neg hr]
  have hpart : ∀ r s e : Nat, s ≤ e → e ≤ q.len →
      posAll (if s < e then #[(⟨q.tgt, q.base + (r : Int) * q.rowStride + (s : Int) * q.stride, q.stride, e - s, 1, 0⟩ :
        Piece)] else #[]) = (List.range' (r * q.len + s) (e - s)).map fun t => (q.tgt, ppt q t) := by
    intro r s e h1 h2
    split
    · rw [posAll_singleton]
      exact pos_run q r s e _ rfl h1 h2
    · rw [posAll_empty, show e - s = 0 by omega]; rfl
  have ha := Nat.div_add_mod a q.len
  have hb' := Nat.div_add_mod b q.len
  have ha2 := Nat.mod_lt a hL
  have hb2 := Nat.mod_lt b hL
  have hdiv : a / q.len ≤ b / q.len := Nat.div_le_div_right (by omega)
  rw [Nat.mul_comm] at ha hb'
  generalize a / q.len = r0 at *
  generalize a % q.len = t0 at *
  generalize b / q.len = r1 at *
  generalize b % q.len = t1 at *
  split
  · rename_i heq
    have heq' : r0 = r1 := by simpa using heq
    subst heq'
    rw [hpart _ _ _ (by omega) (by omega)]
    congr 2; omega
  · rename_i hne
    have hne' : r0 < r1 := by
      have : r0 ≠ r1 := by simpa using hne
      omega
    have hlast := hpart r1 0 t1 (by omega) (by omega)
    rw [posAll_append, posAll_append, hlast, Nat.add_zero, Nat.sub_zero]
    have hmid : ∀ lo, lo ≤ r1 → posAll (if lo < r1 then #[(⟨q.tgt, q.base + (lo : Int) * q.rowStride, q.stride, q.len,
        r1 - lo, q.rowStride⟩ : Piece)] else #[]) =
        (List.range' (lo * q.len) (r1 * q.len - lo * q.len)).map fun t => (q.tgt, ppt q t) := by
      intro lo hlo2
      split
      · rw [posAll_singleton, pos_rows, Nat.sub_mul]
      · rw [posAll_empty, show r1 * q.len - lo * q.len = 0 by
          have : r1 = lo := by omega
          subst this; omega]
        rfl
    have hend : r1 * q.len + (b - r1 * q.len) = b := by omega
    by_cases h0 : t0 = 0
    · subst h0
      simp only [show ((0 : Nat) == 0) = true from rfl, if_true, posAll_empty, List.nil_append]
      rw [hmid r0 (by omega), ← List.map_append, Nat.add_zero,
        range'_join _ _ _ (Nat.mul_le_mul_right _ (by omega))]
      congr 2; omega
    · have h0' : (t0 == 0) = false := by simpa using h0
      simp only [h0', Bool.false_eq_true, if_false]
      rw [posAll_singleton, pos_run q r0 t0 q.len _ rfl (by omega) (Nat.le_refl _), hmid (r0 + 1) (by omega),
        ← List.map_append, ← List.map_append, Nat.add_zero]
      have e1 : r0 * q.len + t0 + (q.len - t0) = (r0 + 1) * q.len := by rw [Nat.succ_mul]; omega
      have e2 : (r0 + 1) * q.len ≤ r1 * q.len := Nat.mul_le_mul_right _ (by omega)
      rw [show q.len - t0 = (r0 + 1) * q.len - (r0 * q.len + t0) by omega,
        range'_join _ _ _ (by omega), show (r0 + 1) * q.len + (r1 * q.len - (r0 + 1) * q.len) - (r0 * q.len + t0) =
          r1 * q.len - (r0 * q.len + t0) by omega]
      simp only [Nat.add_zero]
      rw [range'_join _ _ _ (by omega)]
      congr 2; omega

/-! ## Slices of pieces laid end to end -/

/-- Positions `[a, b)` of a list. -/
def sl {α : Type} (l : List α) (a b : Nat) : List α := (l.drop a).take (b - a)

theorem sl_append {α : Type} (l₁ l₂ : List α) (a b : Nat) :
    sl (l₁ ++ l₂) a b = sl l₁ a (min b l₁.length) ++ sl l₂ (a - l₁.length) (b - l₁.length) := by
  unfold sl
  rw [List.drop_append, List.take_append, List.length_drop]
  congr 1
  · rw [List.take_eq_take_iff, List.length_drop]; omega
  · congr 1; omega

theorem sl_map_range {α : Type} (f : Nat → α) (n a b : Nat) (hb : b ≤ n) :
    sl ((List.range n).map f) a b = (List.range' a (b - a)).map f := by
  unfold sl
  rw [← List.map_drop, ← List.map_take, List.range_eq_range', List.drop_range',
    List.take_range'_of_length_ge (by omega)]
  congr 2; omega

theorem slice_sl (q : Piece) (a b : Nat) (hb : b ≤ q.count) : posAll (q.slice a b) = sl (pos q) a b := by
  rw [slice_pos q a b hb, pos_eq_map, sl_map_range _ _ _ _ hb]

def slGo (a b : Nat) : List Piece → Nat → Array Piece → Array Piece
  | [], _, out => out
  | q :: qs, st, out =>
    if st ≥ b then out
    else slGo a b qs (st + q.count)
      (if st + q.count > a then out ++ q.slice (max a st - st) (min b (st + q.count) - st) else out)

theorem slicePieces_go (ps : Array Piece) (a b : Nat) : Desc.slicePieces ps a b = slGo a b ps.toList 0 #[] := by
  unfold Desc.slicePieces
  simp only [Id.run]
  rw [← Array.forIn_toList]
  generalize ps.toList = l
  generalize (#[] : Array Piece) = out
  generalize (0 : Nat) = st
  induction l generalizing st out with
  | nil => rfl
  | cons q qs ih =>
    simp only [List.forIn_cons, slGo]
    split
    · rfl
    · rw [← ih]
      split <;> rfl

theorem sl_nil {α : Type} (a b : Nat) : sl ([] : List α) a b = [] := by simp [sl]

theorem sl_empty {α : Type} (l : List α) (a b : Nat) (h : b ≤ a ∨ l.length ≤ a) : sl l a b = [] := by
  unfold sl
  rcases h with h | h
  · rw [show b - a = 0 by omega, List.take_zero]
  · rw [List.drop_eq_nil_of_le h, List.take_nil]

theorem slGo_pos (a b : Nat) : ∀ (l : List Piece) (st : Nat) (out : Array Piece),
    posAll (slGo a b l st out) = posAll out ++ sl (l.flatMap pos) (a - st) (b - st)
  | [], st, out => by simp only [slGo, List.flatMap_nil, sl_nil, List.append_nil]
  | q :: qs, st, out => by
    unfold slGo
    split
    · rw [sl_empty _ _ _ (Or.inl (by omega)), List.append_nil]
    · rw [slGo_pos a b qs, List.flatMap_cons, sl_append, pos_length, ← List.append_assoc,
        show a - st - q.count = a - (st + q.count) by omega, show b - st - q.count = b - (st + q.count) by omega]
      congr 1
      split
      · rw [posAll_append, slice_sl _ _ _ (by omega)]
        congr 2 <;> omega
      · rw [sl_empty (pos q) _ _ (Or.inr (by rw [pos_length]; omega)), List.append_nil]

/-- **Positions `[a, b)` of pieces laid end to end.** -/
theorem slicePieces_pos (ps : Array Piece) (a b : Nat) :
    posAll (Desc.slicePieces ps a b) = sl (posAll ps) a b := by
  rw [slicePieces_go, slGo_pos, posAll_empty, List.nil_append, Nat.sub_zero, Nat.sub_zero]; rfl

/-! ## Copies and segments -/

/-- A position moved by `δ` when it is a gate's. -/
def shiftPos (δ : Int) (x : Tgt × Int) : Tgt × Int :=
  match x.1 with
  | .gate _ => (x.1, x.2 + δ)
  | .ref _ _ => x

theorem shiftPos_zero : shiftPos 0 = id := by
  funext x
  unfold shiftPos
  split <;> simp

theorem shiftPos_shiftPos (δ ε : Int) : shiftPos ε ∘ shiftPos δ = shiftPos (δ + ε) := by
  funext x
  rcases x with ⟨t, v⟩
  cases t <;> simp [shiftPos, Int.add_assoc]

theorem pos_shiftGate (δ : Int) (q : Piece) :
    pos (match q.tgt with
      | .gate _ => { q with base := q.base + δ }
      | _ => q) = (pos q).map (shiftPos δ) := by
  rcases q with ⟨t, base, stride, len, rows, rowStride⟩
  cases t with
  | gate w =>
    simp only [pos, List.map_flatMap, List.map_map]
    congr 1; funext r; congr 1; funext t
    simp only [Function.comp, shiftPos]
    congr 1; omega
  | ref p i =>
    simp only [pos, List.map_flatMap, List.map_map]
    congr 1

/-- **Moving the gates by `δ`** moves each gate's position by `δ`. -/
theorem shiftGates_pos (δ : Int) (ps : Array Piece) :
    posAll (Desc.shiftGates δ ps) = (posAll ps).map (shiftPos δ) := by
  unfold Desc.shiftGates posAll
  rw [Array.toList_map, List.flatMap_map, List.map_flatMap]
  congr 1; funext q
  exact pos_shiftGate δ q

theorem shiftGates_length (δ : Int) (ps : Array Piece) :
    (posAll (Desc.shiftGates δ ps)).length = (posAll ps).length := by
  rw [shiftGates_pos, List.length_map]

/-- A segment's positions: its copies' in order. -/
def segPos (s : Seg) : List (Tgt × Int) := (List.range s.reps).flatMap fun m => posAll (s.copy m)

def segsPos (ss : Array Seg) : List (Tgt × Int) := ss.toList.flatMap segPos

theorem segsPos_append (a b : Array Seg) : segsPos (a ++ b) = segsPos a ++ segsPos b := by
  simp [segsPos, List.flatMap_append]

theorem segsPos_empty : segsPos #[] = [] := rfl

theorem segsPos_singleton (s : Seg) : segsPos #[s] = segPos s := by simp [segsPos]

theorem copy_pos (s : Seg) (m : Nat) : posAll (s.copy m) = (posAll s.ps).map (shiftPos ((m : Int) * s.shift)) :=
  shiftGates_pos _ _

theorem segPos_one (q : Piece) : segPos (Seg.one q) = pos q := by
  simp only [segPos, Seg.one, Desc.Seg.copy, List.range_one, List.flatMap_singleton, shiftGates_pos,
    Int.mul_zero, shiftPos_zero, List.map_id, posAll_singleton]

theorem segsPos_one (ps : Array Piece) : segsPos (ps.map Seg.one) = posAll ps := by
  simp only [segsPos, posAll, Array.toList_map, List.flatMap_map, segPos_one]

/-- A list of blocks of one length `L`, as the map of its positions. -/
theorem blocks_eq_map {α : Type} (F : Nat → List α) (L : Nat) (hF : ∀ m, (F m).length = L) (d : α) (n : Nat) :
    (List.range n).flatMap F = (List.range (n * L)).map fun t => (F (t / L)).getD (t % L) d := by
  apply List.ext_getElem?
  intro t
  rw [flatMap_range_getElem? F L hF, List.getElem?_map]
  by_cases h : t < n * L
  · have hL : 0 < L := by
      rcases Nat.eq_zero_or_pos L with h0 | h0
      · rw [h0, Nat.mul_zero] at h; omega
      · exact h0
    have hm : t % L < (F (t / L)).length := by rw [hF]; exact Nat.mod_lt t hL
    rw [if_pos h, List.getElem?_range h]
    simp only [Option.map_some, List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hm, Option.getD_some]
  · rw [if_neg h, List.getElem?_eq_none (by simp; omega)]; rfl

theorem list_eq_map {α : Type} (l : List α) (d : α) : l = (List.range l.length).map fun t => l.getD t d := by
  apply List.ext_getElem?
  intro t
  rw [List.getElem?_map]
  by_cases h : t < l.length
  · rw [List.getElem?_range h, List.getElem?_eq_getElem h]
    simp [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem h]
  · rw [List.getElem?_eq_none (by omega), List.getElem?_eq_none (by simp; omega)]; rfl

theorem copy_length (s : Seg) (m : Nat) : (posAll (s.copy m)).length = (posAll s.ps).length := by
  rw [copy_pos, List.length_map]

theorem copy_zero (s : Seg) : posAll (s.copy 0) = posAll s.ps := by
  rw [copy_pos, show ((0 : Nat) : Int) * s.shift = 0 by simp, shiftPos_zero, List.map_id]

theorem copy_copy (s : Seg) (lo m : Nat) :
    posAll (Desc.shiftGates ((m : Int) * s.shift) (s.copy lo)) = posAll (s.copy (lo + m)) := by
  rw [shiftGates_pos, copy_pos, copy_pos, List.map_map, shiftPos_shiftPos, Int.natCast_add, Int.add_mul]

/-- Position `t` of a segment whose pieces name `s.len` positions: position `t % len` of copy `t / len`. -/
def spt (s : Seg) (t : Nat) : Tgt × Int := (posAll (s.copy (t / s.len))).getD (t % s.len) default

theorem segPos_eq_map (s : Seg) (hlen : (posAll s.ps).length = s.len) :
    segPos s = (List.range s.count).map (spt s) :=
  blocks_eq_map (fun m => posAll (s.copy m)) s.len (fun m => by rw [copy_length, hlen]) default s.reps

theorem spt_add (s : Seg) (m t : Nat) (ht : t < s.len) :
    spt s (m * s.len + t) = (posAll (s.copy m)).getD t default := by
  have hL : 0 < s.len := by omega
  unfold spt
  rw [show m * s.len + t = t + s.len * m by rw [Nat.mul_comm]; omega, Nat.add_mul_div_left _ _ hL,
    Nat.add_mul_mod_self_left, Nat.div_eq_of_lt ht, Nat.mod_eq_of_lt ht, Nat.zero_add]

/-- **A slice of a segment** is its positions `[a, b)`. -/
theorem segSlice_pos (s : Seg) (hlen : (posAll s.ps).length = s.len) (a b : Nat) (hb : b ≤ s.count) :
    segsPos (s.slice a b) = sl (segPos s) a b := by
  rw [segPos_eq_map s hlen, sl_map_range _ _ _ _ hb]
  unfold Desc.Seg.slice
  by_cases hab : b ≤ a
  · rw [if_pos hab, segsPos_empty, show b - a = 0 by omega]; rfl
  rw [if_neg hab]
  have hcnt : s.count = s.reps * s.len := rfl
  have hL : 0 < s.len := by
    rcases Nat.eq_zero_or_pos s.len with h0 | h0
    · rw [hcnt, h0, Nat.mul_zero] at hb; omega
    · exact h0
  by_cases hr : s.reps ≤ 1
  · rw [if_pos hr, segsPos_one, slicePieces_pos]
    have hr1 : s.reps = 1 := by
      rcases Nat.lt_or_ge s.reps 1 with h | h
      · rw [hcnt, show s.reps = 0 by omega, Nat.zero_mul] at hb; omega
      · omega
    have e : segPos s = posAll s.ps := by
      simp only [segPos, hr1, List.range_one, List.flatMap_singleton, copy_zero]
    rw [← e, segPos_eq_map s hlen, sl_map_range _ _ _ _ hb]
  rw [if_neg hr]
  dsimp only
  have hpart : ∀ m x y : Nat, x ≤ y → y ≤ s.len →
      segsPos ((Desc.slicePieces (s.copy m) x y).map Seg.one) =
        (List.range' (m * s.len + x) (y - x)).map (spt s) := by
    intro m x y hxy hy
    have hl : (posAll (s.copy m)).length = s.len := by rw [copy_length, hlen]
    rw [segsPos_one, slicePieces_pos]
    conv => lhs; rw [list_eq_map (posAll (s.copy m)) default]
    rw [hl, sl_map_range _ _ _ _ hy, ← List.map_add_range', List.map_map]
    apply List.map_congr_left
    intro t ht
    obtain ⟨i, hi, rfl⟩ := List.mem_range'.1 ht
    simp only [Function.comp, Nat.one_mul]
    rw [spt_add s m _ (by omega)]
  have hmid : ∀ lo m1 : Nat, lo ≤ m1 →
      segsPos (if lo < m1 then #[{ s with ps := s.copy lo, reps := m1 - lo }] else #[]) =
        (List.range' (lo * s.len) (m1 * s.len - lo * s.len)).map (spt s) := by
    intro lo m1 hlo
    split
    · rw [segsPos_singleton]
      have e : segPos { s with ps := s.copy lo, reps := m1 - lo } =
          (List.range (m1 - lo)).flatMap fun m => posAll (s.copy (lo + m)) := by
        unfold segPos
        congr 1; funext m
        exact copy_copy s lo m
      rw [e, blocks_eq_map (fun m => posAll (s.copy (lo + m))) s.len (fun m => by rw [copy_length, hlen]) default,
        List.range'_eq_map_range, List.map_map, Nat.sub_mul]
      apply List.map_congr_left
      intro t _
      simp only [Function.comp, spt]
      rw [show lo * s.len + t = t + s.len * lo by rw [Nat.mul_comm]; omega, Nat.add_mul_div_left _ _ hL,
        Nat.add_mul_mod_self_left, Nat.add_comm (t / s.len) lo]
    · rw [segsPos_empty, show m1 * s.len - lo * s.len = 0 by
        have : m1 = lo := by omega
        subst this; omega]
      rfl
  have ha := Nat.div_add_mod a s.len
  have hb' := Nat.div_add_mod b s.len
  have ha2 := Nat.mod_lt a hL
  have hb2 := Nat.mod_lt b hL
  have hdiv : a / s.len ≤ b / s.len := Nat.div_le_div_right (by omega)
  rw [Nat.mul_comm] at ha hb'
  generalize a / s.len = r0 at *
  generalize a % s.len = t0 at *
  generalize b / s.len = r1 at *
  generalize b % s.len = t1 at *
  split
  · rename_i heq
    have heq' : r0 = r1 := by simpa using heq
    subst heq'
    rw [hpart _ _ _ (by omega) (by omega)]
    congr 2; omega
  · rename_i hne
    have hne' : r0 < r1 := by
      have : r0 ≠ r1 := by simpa using hne
      omega
    have hlast := hpart r1 0 t1 (by omega) (by omega)
    rw [segsPos_append, segsPos_append, hlast, Nat.add_zero, Nat.sub_zero]
    have hend : r1 * s.len + (b - r1 * s.len) = b := by omega
    by_cases h0 : t0 = 0
    · subst h0
      simp only [show ((0 : Nat) == 0) = true from rfl, if_true, segsPos_empty, List.nil_append]
      rw [hmid r0 r1 (by omega), ← List.map_append, Nat.add_zero,
        range'_join _ _ _ (Nat.mul_le_mul_right _ (by omega))]
      congr 2; omega
    · have h0' : (t0 == 0) = false := by simpa using h0
      simp only [h0', Bool.false_eq_true, if_false]
      rw [hpart r0 t0 s.len (by omega) (Nat.le_refl _), hmid (r0 + 1) r1 (by omega),
        ← List.map_append, ← List.map_append]
      have e1 : r0 * s.len + t0 + (s.len - t0) = (r0 + 1) * s.len := by rw [Nat.succ_mul]; omega
      have e2 : (r0 + 1) * s.len ≤ r1 * s.len := Nat.mul_le_mul_right _ (by omega)
      rw [show s.len - t0 = (r0 + 1) * s.len - (r0 * s.len + t0) by omega,
        range'_join _ _ _ (by omega), show (r0 + 1) * s.len + (r1 * s.len - (r0 + 1) * s.len) - (r0 * s.len + t0) =
          r1 * s.len - (r0 * s.len + t0) by omega]
      simp only [Nat.add_zero]
      rw [range'_join _ _ _ (by omega)]
      congr 2; omega

/-! ## Sequences of segments -/

/-- A segment whose pieces name `len` positions. -/
def SegOk (s : Seg) : Prop := (posAll s.ps).length = s.len

theorem segPos_length (s : Seg) (h : SegOk s) : (segPos s).length = s.count := by
  unfold segPos Seg.count
  exact flatMap_range_length _ s.len (fun m => by rw [copy_length]; exact h) s.reps

theorem segsPos_length (ss : List Seg) (h : ∀ g ∈ ss, SegOk g) :
    (ss.flatMap segPos).length = (ss.map Seg.count).sum := by
  induction ss with
  | nil => rfl
  | cons g gs ih =>
    rw [List.flatMap_cons, List.length_append, segPos_length g (h g (by simp)),
      ih (fun g' hg' => h g' (by simp [hg'])), List.map_cons, List.sum_cons]

/-- A sequence's positions: its segments' in order. -/
def psPos (s : PS) : List (Tgt × Int) := segsPos s.segs

/-- What `PS.of` makes: segments that name their `len` positions, of positive count, each starting where the ones before
it end. -/
def PSOk (s : PS) : Prop :=
  (∀ i < s.segs.size, SegOk s.segs[i]! ∧ 0 < s.segs[i]!.count) ∧ s.starts.size = s.segs.size ∧
    (∀ i < s.segs.size, s.starts[i]! = ((s.segs.toList.take i).map Seg.count).sum) ∧
    s.count = (s.segs.toList.map Seg.count).sum

theorem forIn_inv {α σ : Type} (P : σ → Prop) (f : α → σ → Id (ForInStep σ))
    (hstep : ∀ x st, P st → P (f x st).value) : ∀ (l : List α) (init : σ), P init → P (forIn l init f)
  | [], init, h => h
  | x :: l, init, h => by
    rw [List.forIn_cons]
    have hx := hstep x init h
    revert hx
    cases f x init with
    | done b => intro hx; exact hx
    | yield b => intro hx; exact forIn_inv P f hstep l b hx

/-- `PS.slice`'s loop over the segments from the one the bisection found. -/
def psGo (s : PS) (lo hi : Nat) : List Nat → Array Seg → Array Seg
  | [], out => out
  | i :: is, out =>
    if s.starts[i]! ≥ hi then out
    else psGo s lo hi is (if s.starts[i]! + s.segs[i]!.count > lo then out ++ s.segs[i]!.slice
      (max lo s.starts[i]! - s.starts[i]!) (min hi (s.starts[i]! + s.segs[i]!.count) - s.starts[i]!) else out)

theorem psGo_eq (s : PS) (lo hi : Nat) : ∀ (l : List Nat) (out : Array Seg),
    (forIn l out fun i (r : Array Seg) =>
      if s.starts[i]! ≥ hi then (pure (ForInStep.done r) : Id _)
      else if s.starts[i]! + s.segs[i]!.count > lo then
        pure (ForInStep.yield (r ++ s.segs[i]!.slice (max lo s.starts[i]! - s.starts[i]!)
          (min hi (s.starts[i]! + s.segs[i]!.count) - s.starts[i]!)))
      else pure (ForInStep.yield r)) = psGo s lo hi l out
  | [], _ => rfl
  | i :: is, out => by
    rw [List.forIn_cons]
    simp only [psGo]
    split
    · rfl
    · split
      · exact psGo_eq s lo hi is _
      · exact psGo_eq s lo hi is _

theorem psGo_pos (s : PS) (hs : PSOk s) (lo hi : Nat) : ∀ (n a : Nat) (out : Array Seg), a + n = s.segs.size →
    segsPos (psGo s lo hi (List.range' a n) out) = segsPos out ++
      sl ((s.segs.toList.drop a).flatMap segPos) (lo - ((s.segs.toList.take a).map Seg.count).sum)
        (hi - ((s.segs.toList.take a).map Seg.count).sum)
  | 0, a, out, h => by
    rw [List.drop_eq_nil_of_le (by simp; omega)]
    simp [psGo, sl]
  | n + 1, a, out, h => by
    have hseg := hs.1
    have hst := hs.2.2.1
    have ha : a < s.segs.size := by omega
    have hga : s.segs[a]! = s.segs.toList[a]'(by simpa using ha) := by
      rw [getElem!_pos s.segs a ha, Array.getElem_toList]
    have hpre : ((s.segs.toList.take (a + 1)).map Seg.count).sum =
        ((s.segs.toList.take a).map Seg.count).sum + s.segs[a]!.count := by
      rw [List.take_add_one, List.getElem?_eq_getElem (by simpa using ha), hga]
      simp
    rw [List.range'_succ]
    simp only [psGo]
    rw [hst a ha]
    generalize ((s.segs.toList.take a).map Seg.count).sum = st at hpre
    split
    · rw [sl_empty _ _ _ (Or.inl (by omega)), List.append_nil]
    · rw [psGo_pos s hs lo hi n (a + 1) _ (by omega), hpre,
        List.drop_eq_getElem_cons (i := a) (by simpa using ha), ← hga, List.flatMap_cons, sl_append,
        segPos_length _ (hseg a ha).1, ← List.append_assoc,
        show lo - st - s.segs[a]!.count = lo - (st + s.segs[a]!.count) by omega,
        show hi - st - s.segs[a]!.count = hi - (st + s.segs[a]!.count) by omega]
      congr 1
      split
      · rw [segsPos_append, segSlice_pos _ (hseg a ha).1 _ _ (by omega)]
        congr 2 <;> omega
      · rw [sl_empty (segPos s.segs[a]!) _ _ (Or.inr (by rw [segPos_length _ (hseg a ha).1]; omega)),
          List.append_nil]

theorem segOk_of_mem (s : PS) (hs : PSOk s) (g : Seg) (hg : g ∈ s.segs.toList) : SegOk g := by
  obtain ⟨i, hi, rfl⟩ := List.mem_iff_getElem.1 hg
  have hi' : i < s.segs.size := by simpa using hi
  have := (hs.1 i hi').1
  rwa [getElem!_pos s.segs i hi', ← Array.getElem_toList] at this

theorem bind_id {α β : Type} (x : Id α) (f : α → Id β) : (x >>= f) = f x := rfl

/-- **A slice of a sequence** is its positions `[lo, hi)`. -/
theorem psSlice_pos (s : PS) (hs : PSOk s) (lo hi : Nat) : segsPos (s.slice lo hi) = sl (psPos s) lo hi := by
  unfold Desc.PS.slice
  simp only [Id.run, Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size]
  split
  · rename_i h
    simp only [Bool.or_eq_true, decide_eq_true_eq] at h
    rcases h with h | h
    · exact (sl_empty _ _ _ (Or.inl h)).symm
    · have h0 : s.segs.toList = [] := by simpa using h
      show [] = (((s.segs.toList.flatMap segPos).drop lo).take (hi - lo))
      rw [h0]; simp
  · rename_i h
    simp only [Bool.or_eq_true, decide_eq_true_eq, not_or] at h
    have hne : 0 < s.segs.size := by
      rcases Nat.eq_zero_or_pos s.segs.size with h0 | h0
      · exfalso; apply h.2; simp [Array.isEmpty, h0]
      · exact h0
    generalize hB : (forIn (m := Id) (List.range' 0 _) (0, s.segs.size) _ : Id (Nat × Nat)) = B
    have hinv : B.1 < s.segs.size ∧ B.2 ≤ s.segs.size ∧ (B.1 = 0 ∨ s.starts[B.1]! ≤ lo) := by
      rw [← hB]
      refine forIn_inv (fun x : Nat × Nat => x.1 < s.segs.size ∧ x.2 ≤ s.segs.size ∧ (x.1 = 0 ∨ s.starts[x.1]! ≤ lo))
        _ ?_ _ _ ⟨hne, Nat.le_refl _, Or.inl rfl⟩
      intro x st ⟨h1, h2, h3⟩
      split
      · exact ⟨h1, h2, h3⟩
      · rename_i hgt
        split
        · rename_i hle
          exact ⟨by show (st.1 + st.2) / 2 < s.segs.size; omega, h2, Or.inr hle⟩
        · exact ⟨h1, by show (st.1 + st.2) / 2 ≤ s.segs.size; omega, h3⟩
    rw [bind_id, psGo_eq, show (s.segs.size - B.1 + 1 - 1) / 1 = s.segs.size - B.1 by omega, bind_id]
    change segsPos (psGo s lo hi (List.range' B.1 (s.segs.size - B.1)) #[]) = _
    rw [psGo_pos s hs lo hi (s.segs.size - B.1) B.1 #[] (by omega), segsPos_empty, List.nil_append]
    obtain ⟨h1, -, h3⟩ := hinv
    generalize B.1 = a at h1 h3
    have hpre : ((s.segs.toList.take a).map Seg.count).sum ≤ lo := by
      rcases h3 with h3 | h3
      · subst h3; simp
      · rw [← hs.2.2.1 a h1]; exact h3
    show _ = sl (s.segs.toList.flatMap segPos) lo hi
    conv => rhs; rw [← List.take_append_drop a s.segs.toList]
    have hlen : ((s.segs.toList.take a).flatMap segPos).length = ((s.segs.toList.take a).map Seg.count).sum :=
      segsPos_length _ fun g hg => segOk_of_mem s hs g (List.mem_of_mem_take hg)
    rw [List.flatMap_append, sl_append, hlen, sl_empty ((s.segs.toList.take a).flatMap segPos) lo _
      (Or.inr (by rw [hlen]; exact hpre)), List.nil_append]

theorem posAll_flatMap {β : Type} (xs : Array β) (f : β → Array Piece) :
    posAll (xs.flatMap f) = xs.toList.flatMap fun x => posAll (f x) := by
  simp [posAll, Array.toList_flatMap, List.flatMap_assoc]

/-- **A sequence's pieces `[lo, hi)`** are its positions `[lo, hi)`. -/
theorem pieces_pos (s : PS) (hs : PSOk s) (lo hi : Nat) : posAll (s.pieces lo hi) = sl (psPos s) lo hi := by
  rw [← psSlice_pos s hs lo hi]
  unfold Desc.PS.pieces
  rw [posAll_flatMap]
  unfold segsPos
  congr 1; funext g
  rw [posAll_flatMap, Array.toList_range]
  rfl

theorem of_fold (l : List Seg) : ∀ (A : Array Nat) (c : Nat),
    l.foldl (fun (acc : Array Nat × Nat) s => (acc.1.push acc.2, acc.2 + s.count)) (A, c) =
      (A ++ ((List.range l.length).map fun i => c + ((l.take i).map Seg.count).sum).toArray,
        c + (l.map Seg.count).sum) := by
  induction l with
  | nil => intro A c; simp
  | cons g gs ih =>
    intro A c
    rw [List.foldl_cons, ih]
    simp only [List.length_cons, List.map_cons, List.sum_cons, Prod.mk.injEq]
    refine ⟨?_, by omega⟩
    apply Array.ext'
    simp only [Array.toList_append, Array.toList_push, List.append_assoc]
    congr 1
    rw [List.range_succ_eq_map, List.map_cons, List.map_map]
    simp only [List.take_zero, List.map_nil, List.sum_nil, Nat.add_zero, List.singleton_append]
    congr 1
    apply List.map_congr_left
    intro i _
    simp only [Function.comp, List.take_succ_cons, List.map_cons, List.sum_cons]
    omega

theorem flatMap_filter_count (l : List Seg) (h : ∀ g ∈ l, SegOk g) :
    (l.filter (fun g => decide (g.count > 0))).flatMap segPos = l.flatMap segPos := by
  induction l with
  | nil => rfl
  | cons g gs ih =>
    have ih' := ih (fun g' hg' => h g' (by simp [hg']))
    rw [List.filter_cons]
    split
    · rw [List.flatMap_cons, List.flatMap_cons, ih']
    · rename_i hc
      have h0 : g.count = 0 := by simpa using hc
      have : segPos g = [] := List.eq_nil_of_length_eq_zero (by rw [segPos_length g (h g (by simp)), h0])
      rw [List.flatMap_cons, this, List.nil_append, ih']

theorem of_segs (segs : Array Seg) : (PS.of segs).segs = segs.filter fun g => decide (g.count > 0) := rfl

theorem of_starts_count (segs : Array Seg) :
    ((PS.of segs).starts, (PS.of segs).count) = (segs.filter fun g => decide (g.count > 0)).foldl
      (fun (acc : Array Nat × Nat) s => (acc.1.push acc.2, acc.2 + s.count)) (#[], 0) := rfl

/-- **`PS.of`** keeps its segments' positions, and makes a sequence `PS.slice` reads. -/
theorem of_ok (segs : Array Seg) (h : ∀ g ∈ segs.toList, SegOk g) :
    PSOk (PS.of segs) ∧ psPos (PS.of segs) = segsPos segs ∧ (PS.of segs).count = (segsPos segs).length := by
  have hf : ∀ g ∈ (PS.of segs).segs.toList, SegOk g ∧ 0 < g.count := by
    intro g hg
    rw [of_segs, Array.toList_filter, List.mem_filter] at hg
    exact ⟨h g hg.1, by simpa using hg.2⟩
  have hsc := of_starts_count segs
  rw [← Array.foldl_toList, of_fold, ← of_segs] at hsc
  simp only [Prod.mk.injEq] at hsc
  obtain ⟨hst, hcnt⟩ := hsc
  have hpos : psPos (PS.of segs) = segsPos segs := by
    unfold psPos segsPos
    rw [of_segs, Array.toList_filter, flatMap_filter_count _ h]
  have hsz : (PS.of segs).segs.size = (PS.of segs).segs.toList.length := by simp
  refine ⟨⟨?_, ?_, ?_, ?_⟩, hpos, ?_⟩
  · intro i hi
    have hi' : i < (PS.of segs).segs.toList.length := by rw [← hsz]; exact hi
    have := hf _ (List.getElem_mem hi')
    rwa [getElem!_pos _ i hi, ← Array.getElem_toList]
  · rw [hst, hsz]; simp
  · intro i hi
    have hi' : i < (PS.of segs).segs.toList.length := by rw [← hsz]; exact hi
    rw [hst, getElem!_pos _ i (by simpa using hi')]
    simp
  · rw [hcnt, Nat.zero_add]
  · rw [hcnt, Nat.zero_add, ← hpos]
    unfold psPos segsPos
    rw [segsPos_length _ (fun g hg => (hf g hg).1)]

end NciCheck.Properties.SecurityProofs.ReadsExact
