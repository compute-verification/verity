import Properties.Proofs.Nci.Novel

/-!
# Toward `Novel.ReadsExact`: what `Desc.analyze`'s pieces and segments name

`Novel.ReadsExact` says that the segments `Desc.analyze` gives a root node name exactly the gates outside the node that
its gates read. Its proof reads each piece, segment and sequence of segments as the list of positions it names in order
(`pos`, `posAll`, `segPos`, `psPos`), a position being a target (a parameter's or a node's leaves, or the activation's
gates of one width) and an index, and holds each of `Desc`'s operations on them to the positions it means:

* a slice of pieces, of a segment or of a sequence of segments is its positions `[a, b)` (`slice_pos`,
  `slicePieces_pos`, `segSlice_pos`, `psSlice_pos`, `pieces_pos`), and `shiftGates` moves gate positions
  (`shiftGates_pos`);
* `PS.of` lays its segments end to end (`of_ok`), `ofPart` and `ofSeq` name a decoded sequence's references in order
  (`ofSeq_pos`, `partsPos_get`), where no strided part names a negative position, which the decoder ensures
  (`parsePartsIn_nonneg`), and `memberSpans` are a batch member's indices into its argument (`memberSpans_idx`);
* a segment's gates (`Width.segGates`) are its gate positions (`segGates_pos`).

So `ReadsExact` holds wherever one statement about the positions `Desc.nodeReads` gives each root node does, a named
`Prop` here, open (`readsExact_of_rootReads`): `RootReads`.
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

/-! ## Reference sequences as pieces -/

open Flock.Program (Ref Part Seq)

/-- A reference as a position: its collection's leaf. -/
def refPos (r : Ref) : Tgt × Int := (.ref r.p r.i, (r.j : Int))

/-- A part's references never step below leaf 0: a strided part's pattern (`parsePartsIn` checks its corners). -/
def PartNonneg : Part → Prop
  | .strided _ _ b inner os is c => ∀ k < c, 0 ≤ b + ((k / inner : Nat) : Int) * os + ((k % inner : Nat) : Int) * is
  | _ => True

theorem pos_one (t : Tgt) (base stride : Int) (len : Nat) (rs : Int) :
    pos ⟨t, base, stride, len, 1, rs⟩ = (List.range len).map fun (k : Nat) => (t, base + (k : Int) * stride) := by
  simp [pos]

theorem posAll_filter_count (ps : Array Piece) :
    posAll (ps.filter fun q => decide (q.count > 0)) = posAll ps := by
  unfold posAll
  rw [Array.toList_filter]
  generalize ps.toList = l
  induction l with
  | nil => rfl
  | cons q qs ih =>
    rw [List.filter_cons]
    split
    · rw [List.flatMap_cons, List.flatMap_cons, ih]
    · rename_i hc
      have h0 : q.count = 0 := by simpa using hc
      have : pos q = [] := List.eq_nil_of_length_eq_zero (by rw [pos_length, h0])
      rw [List.flatMap_cons, this, List.nil_append, ih]

theorem posAll_pair (x y : Piece) : posAll #[x, y] = pos x ++ pos y := by simp [posAll]

theorem range_split (a n : Nat) : List.range (a + n) = List.range a ++ List.range' a n := by
  rw [List.range_eq_range', List.range_eq_range', show List.range' a n = List.range' (0 + a) n by rw [Nat.zero_add],
    List.range'_append_1]

theorem tgt_eq_of_beq {t : Tgt} {p : Bool} {i : Nat} (h : (t == Tgt.ref p i) = true) : t = Tgt.ref p i := by
  cases t with
  | gate w =>
    have e : (Tgt.gate w == Tgt.ref p i) = false := rfl
    rw [e] at h; cases h
  | ref p' i' =>
    have e : (Tgt.ref p' i' == Tgt.ref p i) = (p' == p && i' == i) := rfl
    rw [e, Bool.and_eq_true, beq_iff_eq, beq_iff_eq] at h
    rw [h.1, h.2]

theorem ofPart_affine (p : Bool) (i b c : Nat) :
    posAll (Desc.ofPart (.affine p i b c)) = (List.range c).map fun k => refPos ((Part.affine p i b c).get k) := by
  simp only [Desc.ofPart, posAll_singleton, pos_one, Flock.Program.Part.get, refPos]
  apply List.map_congr_left
  intro k _
  simp

theorem ofPart_strided (p : Bool) (i : Nat) (b : Int) (inner : Nat) (os is : Int) (c : Nat)
    (hq : PartNonneg (.strided p i b inner os is c)) :
    posAll (Desc.ofPart (.strided p i b inner os is c)) =
      (List.range c).map fun k => refPos ((Part.strided p i b inner os is c).get k) := by
  have hget : ∀ k < c, refPos ((Part.strided p i b inner os is c).get k) =
      (.ref p i, b + ((k / inner : Nat) : Int) * os + ((k % inner : Nat) : Int) * is) := by
    intro k hk
    simp only [Flock.Program.Part.get, refPos]
    rw [Int.toNat_of_nonneg (hq k hk)]
  have hmap : ∀ (f : Nat → Tgt × Int), (∀ k < c, f k = (.ref p i,
      b + ((k / inner : Nat) : Int) * os + ((k % inner : Nat) : Int) * is)) →
      (List.range c).map f = (List.range c).map fun k => refPos ((Part.strided p i b inner os is c).get k) := by
    intro f hf
    apply List.map_congr_left
    intro k hk
    have hk' := List.mem_range.1 hk
    rw [hf k hk', hget k hk']
  simp only [Desc.ofPart]
  split
  · rename_i h0
    have : c = 0 := by simpa using h0
    subst this; rfl
  split
  · rename_i h1
    have : inner = 1 := by simpa using h1
    subst this
    rw [posAll_singleton, pos_one]
    apply hmap
    intro k _
    simp
  split
  · rename_i h2
    have h2' : os = (inner : Int) * is := by simpa using h2
    rw [posAll_singleton, pos_one]
    apply hmap
    intro k _
    have hk : (k : Int) = ((k / inner : Nat) : Int) * inner + ((k % inner : Nat) : Int) := by
      have := Nat.div_add_mod k inner
      rw [Nat.mul_comm] at this
      exact_mod_cast this.symm
    rw [h2', hk, Int.add_mul, ← Int.mul_assoc, Int.add_assoc]
  · rw [posAll_filter_count, posAll_pair, pos_one, pos_eq_map]
    simp only [Piece.count]
    have hc : c = c / inner * inner + c % inner := by
      have := Nat.div_add_mod c inner
      rw [Nat.mul_comm] at this; omega
    conv => rhs; rw [hc]
    rw [range_split, List.map_append]
    rw [← hc]
    congr 1
    · apply List.map_congr_left
      intro k hk
      have hk' := List.mem_range.1 hk
      have hkc : k < c := by
        have : c / inner * inner ≤ c := Nat.div_mul_le_self c inner
        omega
      rw [hget k hkc]
      simp [ppt]
    · rw [List.range'_eq_map_range, List.map_map]
      apply List.map_congr_left
      intro t ht
      have ht' := List.mem_range.1 ht
      have hkc : c / inner * inner + t < c := by omega
      simp only [Function.comp]
      rw [hget _ hkc]
      by_cases hin : inner = 0
      · subst hin
        simp
      · have hin' : 0 < inner := by omega
        have hlt : t < inner := by
          have := Nat.mod_lt c hin'
          omega
        rw [show c / inner * inner + t = t + inner * (c / inner) by rw [Nat.mul_comm]; omega,
          Nat.add_mul_div_left _ _ hin', Nat.add_mul_mod_self_left, Nat.div_eq_of_lt hlt, Nat.mod_eq_of_lt hlt,
          Nat.zero_add]

/-- `ofPart`'s step over an explicit part's reference: extend the last piece when the reference is its next element. -/
def exStep (acc : Array Piece) (r : Ref) : Array Piece :=
  let one : Piece := { tgt := .ref r.p r.i, base := r.j, stride := 1, len := 1 }
  match acc.back? with
  | some q => if q.tgt == .ref r.p r.i && q.stride == 1 && q.at q.len == r.j then acc.pop.push { q with len := q.len + 1 }
    else acc.push one
  | none => acc.push one

theorem posAll_push (acc : Array Piece) (q : Piece) : posAll (acc.push q) = posAll acc ++ pos q := by
  simp [posAll]

theorem exStep_spec (acc : Array Piece) (r : Ref) (h : ∀ q ∈ acc.toList, q.rows = 1) :
    posAll (exStep acc r) = posAll acc ++ [refPos r] ∧ ∀ q ∈ (exStep acc r).toList, q.rows = 1 := by
  have hone : pos { tgt := .ref r.p r.i, base := r.j, stride := 1, len := 1 } = [refPos r] := by
    rw [pos_one]; simp [refPos]
  unfold exStep
  rcases hb : acc.back? with _ | q
  · refine ⟨by rw [posAll_push, hone], ?_⟩
    intro x hx
    rw [Array.toList_push, List.mem_append] at hx
    rcases hx with hx | hx
    · exact h x hx
    · simp at hx; rw [hx]
  · obtain ⟨ys, rfl⟩ := Array.back?_eq_some_iff.1 hb
    have hq : q.rows = 1 := h q (by simp)
    have hys : ∀ x ∈ ys.toList, x.rows = 1 := fun x hx => h x (by simp [hx])
    dsimp only
    split
    · rename_i hc
      simp only [Bool.and_eq_true, beq_iff_eq] at hc
      obtain ⟨⟨htgt, hst⟩, hat⟩ := hc
      rw [Array.pop_push]
      refine ⟨?_, ?_⟩
      · rw [posAll_push, posAll_push, List.append_assoc]
        congr 1
        rcases q with ⟨t, base, stride, len, rows, rs⟩
        simp only at hq hst hat htgt
        subst hq hst
        rw [pos_one, pos_one, List.range_succ, List.map_append]
        simp only [Desc.Piece.at, Int.mul_one] at hat
        rw [tgt_eq_of_beq htgt]
        simp [refPos, hat]
      · intro x hx
        rw [Array.toList_push, List.mem_append] at hx
        rcases hx with hx | hx
        · exact hys x hx
        · simp at hx; rw [hx]; exact hq
    · refine ⟨by rw [posAll_push, hone], ?_⟩
      intro x hx
      rw [Array.toList_push, List.mem_append] at hx
      rcases hx with hx | hx
      · exact h x hx
      · simp at hx; rw [hx]

theorem exFold_spec : ∀ (l : List Ref) (acc : Array Piece), (∀ q ∈ acc.toList, q.rows = 1) →
    posAll (l.foldl exStep acc) = posAll acc ++ l.map refPos
  | [], acc, _ => by simp
  | r :: l, acc, h => by
    obtain ⟨h1, h2⟩ := exStep_spec acc r h
    rw [List.foldl_cons, exFold_spec l _ h2, h1, List.map_cons, List.append_assoc, List.singleton_append]

theorem ofPart_explicit (it : Array Ref) :
    posAll (Desc.ofPart (.explicit it)) = (List.range it.size).map fun k => refPos ((Part.explicit it).get k) := by
  have e : Desc.ofPart (.explicit it) = it.toList.foldl exStep #[] := by
    simp only [Desc.ofPart]
    rw [← Array.foldl_toList]
    rfl
  rw [e, exFold_spec _ _ (by simp), posAll_empty, List.nil_append]
  apply List.ext_getElem?
  intro k
  rw [List.getElem?_map, List.getElem?_map]
  by_cases hk : k < it.size
  · rw [List.getElem?_range hk, Array.getElem?_toList, Array.getElem?_eq_getElem hk]
    simp [Flock.Program.Part.get, getElem!_pos it k hk]
  · rw [List.getElem?_eq_none (by simp; omega), List.getElem?_eq_none (by simp; omega)]; rfl

/-- **A part as pieces** names its references, in order. -/
theorem ofPart_pos (q : Part) (hq : PartNonneg q) :
    posAll (Desc.ofPart q) = (List.range q.count).map fun k => refPos (q.get k) := by
  cases q with
  | affine p i b c => exact ofPart_affine p i b c
  | strided p i b inner os is c => exact ofPart_strided p i b inner os is c hq
  | explicit it => exact ofPart_explicit it

/-- A sequence's references as positions, part by part. -/
def partsPos (ps : Array Part) : List (Tgt × Int) :=
  ps.toList.flatMap fun q => (List.range q.count).map fun k => refPos (q.get k)

theorem partsFold_pos : ∀ (l : List Part) (acc : Array Piece), (∀ q ∈ l, PartNonneg q) →
    posAll (l.foldl (fun acc q => acc ++ Desc.ofPart q) acc) =
      posAll acc ++ l.flatMap fun q => (List.range q.count).map fun k => refPos (q.get k)
  | [], acc, _ => by simp
  | q :: l, acc, h => by
    rw [List.foldl_cons, partsFold_pos l _ (fun q' hq' => h q' (by simp [hq'])), posAll_append,
      ofPart_pos q (h q (by simp)), List.flatMap_cons, List.append_assoc]

theorem segOk_one (q : Piece) : SegOk (Seg.one q) := by
  show (posAll #[q]).length = q.count
  rw [posAll_singleton, pos_length]

/-- **A sequence as segments** (`Desc.ofSeq`) names its references, in order. -/
theorem ofSeq_pos (s : Seq) (hq : ∀ q ∈ s.parts.toList, PartNonneg q) :
    PSOk (Desc.ofSeq s) ∧ psPos (Desc.ofSeq s) = partsPos s.parts ∧
      (Desc.ofSeq s).count = (partsPos s.parts).length := by
  have hok : ∀ g ∈ ((s.parts.foldl (fun acc q => acc ++ Desc.ofPart q) #[]).map Seg.one).toList, SegOk g := by
    intro g hg
    rw [Array.toList_map, List.mem_map] at hg
    obtain ⟨q, -, rfl⟩ := hg
    exact segOk_one q
  have hpos : segsPos ((s.parts.foldl (fun acc q => acc ++ Desc.ofPart q) #[]).map Seg.one) = partsPos s.parts := by
    rw [segsPos_one, ← Array.foldl_toList, partsFold_pos _ _ hq, posAll_empty, List.nil_append]; rfl
  obtain ⟨h1, h2, h3⟩ := of_ok _ hok
  exact ⟨h1, h2.trans hpos, h3.trans (by rw [hpos])⟩

/-- Block `i` of a list of blocks starts after the blocks before it. -/
theorem flatMap_getElem?_block {α β : Type} (F : α → List β) :
    ∀ (l : List α) (i t : Nat) (hi : i < l.length), t < (F l[i]).length →
      (l.flatMap F)[((l.take i).map fun x => (F x).length).sum + t]? = (F l[i])[t]?
  | [], _, _, hi, _ => absurd hi (by simp)
  | x :: l, 0, t, _, ht => by
    simp only [List.take_zero, List.map_nil, List.sum_nil, Nat.zero_add, List.flatMap_cons,
      List.getElem_cons_zero] at ht ⊢
    rw [List.getElem?_append_left ht]
  | x :: l, i + 1, t, hi, ht => by
    simp only [List.take_succ_cons, List.map_cons, List.sum_cons, List.flatMap_cons, List.getElem_cons_succ]
    rw [List.getElem?_append_right (by omega), Nat.add_assoc, Nat.add_sub_cancel_left]
    exact flatMap_getElem?_block F l i t (by simpa using hi) ht

/-! ## A member's runs of a mapped argument -/

/-- Runs `[x, y)` laid end to end. -/
def spansPos (xs : Array (Nat × Nat)) : List Nat := xs.toList.flatMap fun p => List.range' p.1 (p.2 - p.1)

theorem range'_shift (X s n : Nat) : List.range' X n = (List.range' s n).map fun t => X + (t - s) := by
  rw [List.range'_eq_map_range, List.range'_eq_map_range, List.map_map]
  apply List.map_congr_left
  intro i _
  simp only [Function.comp]
  omega

/-- The runs of `[lo, hi)` in blocks of `L`, from block `q0`, for `n` blocks. -/
theorem blocks_cover (L lo hi q0 : Nat) (h0 : q0 * L ≤ lo) : ∀ n : Nat,
    (List.range n).flatMap (fun q' => List.range' (max lo ((q0 + q') * L))
      (min hi ((q0 + q' + 1) * L) - max lo ((q0 + q') * L))) =
      List.range' lo (min hi ((q0 + n) * L) - lo)
  | 0 => by
    rw [List.range_zero, List.flatMap_nil, Nat.add_zero, show min hi (q0 * L) - lo = 0 by omega]; rfl
  | n + 1 => by
    rw [List.range_succ, List.flatMap_append, blocks_cover L lo hi q0 h0 n, List.flatMap_singleton]
    have hY : (q0 + n + 1) * L = (q0 + n) * L + L := Nat.succ_mul _ _
    have hX : q0 * L ≤ (q0 + n) * L := Nat.mul_le_mul_right _ (by omega)
    rw [show q0 + (n + 1) = q0 + n + 1 by omega]
    generalize (q0 + n) * L = X at hY hX
    rw [hY]
    by_cases hx : hi ≤ X
    · rw [show min hi (X + L) - max lo X = 0 by omega, show min hi X = hi by omega,
        show min hi (X + L) = hi by omega]
      simp
    · by_cases hl : X ≤ lo
      · rw [show min hi X - lo = 0 by omega, show max lo X = lo by omega]
        simp
      · rw [show max lo X = X by omega, show min hi X = X by omega, range'_join _ _ _ (by omega)]
        congr 1; omega

theorem flatMap_filterMap_spans {β : Type} (g : β → Option (Nat × Nat)) (l : List β) :
    (l.filterMap g).flatMap (fun p => List.range' p.1 (p.2 - p.1)) =
      l.flatMap fun b => match g b with
        | some p => List.range' p.1 (p.2 - p.1)
        | none => [] := by
  induction l with
  | nil => rfl
  | cons b l ih =>
    rw [List.filterMap_cons, List.flatMap_cons]
    cases g b with
    | none => simp [ih]
    | some p => simp [ih]

/-- **A member's runs** (`Desc.memberSpans`) are the argument's indices of the member's parameter leaves `[lo, hi)`,
in order: leaf `t` of member `m` of an argument mapped with dims `(d, L)` is index `(t / L · d + m) · L + t % L`. -/
theorem memberSpans_pos (a : Flock.Program.Arg) (m lo hi : Nat) (d L : Nat) (hd : a.dims = some (d, L))
    (hL : L = 0 → hi ≤ lo) :
    spansPos (Desc.memberSpans a m lo hi) =
      (List.range' lo (hi - lo)).map fun t => (t / L * d + m) * L + t % L := by
  unfold Desc.memberSpans
  rw [hd]
  dsimp only
  split
  · rename_i h
    have : L = 0 := by simpa using h
    rw [show hi - lo = 0 by have := hL this; omega]; rfl
  rename_i h
  have hL0 : 0 < L := by
    have : L ≠ 0 := by simpa using h
    omega
  unfold spansPos
  rw [Array.toList_filterMap, Array.toList_range, flatMap_filterMap_spans]
  have hq0 : lo / L * L ≤ lo := Nat.div_mul_le_self lo L
  have hblock : ∀ q' : Nat,
      (match (if max lo ((lo / L + q') * L) < min hi ((lo / L + q' + 1) * L) then
          some (((lo / L + q') * d + m) * L + (max lo ((lo / L + q') * L) - (lo / L + q') * L),
            ((lo / L + q') * d + m) * L + (min hi ((lo / L + q' + 1) * L) - (lo / L + q') * L)) else none) with
        | some p => List.range' p.1 (p.2 - p.1)
        | none => []) =
      (List.range' (max lo ((lo / L + q') * L)) (min hi ((lo / L + q' + 1) * L) - max lo ((lo / L + q') * L))).map
        fun t => (t / L * d + m) * L + t % L := by
    intro q'
    generalize lo / L + q' = q
    have hY : (q + 1) * L = q * L + L := Nat.succ_mul _ _
    by_cases hse : max lo (q * L) < min hi ((q + 1) * L)
    · rw [if_pos hse]
      dsimp only
      rw [range'_shift _ (max lo (q * L)), show ((q * d + m) * L + (min hi ((q + 1) * L) - q * L) -
          ((q * d + m) * L + (max lo (q * L) - q * L))) = min hi ((q + 1) * L) - max lo (q * L) by omega]
      apply List.map_congr_left
      intro t ht
      obtain ⟨i, hi', rfl⟩ := List.mem_range'.1 ht
      have hM1 : q * L ≤ max lo (q * L) := Nat.le_max_right _ _
      have hlt : max lo (q * L) + 1 * i - q * L < L := by omega
      have hsplit : max lo (q * L) + 1 * i = (max lo (q * L) + 1 * i - q * L) + L * q := by
        rw [Nat.mul_comm L q]; omega
      have e1 : (max lo (q * L) + 1 * i) / L = q := by
        conv => lhs; rw [hsplit]
        rw [Nat.add_mul_div_left _ _ hL0, Nat.div_eq_of_lt hlt, Nat.zero_add]
      have e2 : (max lo (q * L) + 1 * i) % L = max lo (q * L) + 1 * i - q * L := by
        conv => lhs; rw [hsplit]
        rw [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hlt]
      rw [e1, e2]
      generalize max lo (q * L) = M at *
      generalize q * L = Q at *
      omega
    · rw [if_neg hse]
      dsimp only
      rw [show min hi ((q + 1) * L) - max lo (q * L) = 0 by omega]; rfl
  simp only [hblock]
  rw [← List.map_flatMap, blocks_cover L lo hi (lo / L) hq0]
  congr 1
  by_cases hlh : hi ≤ lo
  · rw [show hi - lo = 0 by omega]
    rw [show min hi ((lo / L + ((hi + L - 1) / L - lo / L)) * L) - lo = 0 by omega]
  · have hce : hi ≤ (hi + L - 1) / L * L := by
      have h1 := Nat.div_add_mod (hi + L - 1) L
      have h2 := Nat.mod_lt (hi + L - 1) hL0
      have h3 : (hi + L - 1) / L * L = L * ((hi + L - 1) / L) := Nat.mul_comm _ _
      omega
    have hle : lo / L ≤ (hi + L - 1) / L := Nat.div_le_div_right (by omega)
    rw [show lo / L + ((hi + L - 1) / L - lo / L) = (hi + L - 1) / L by omega, show min hi ((hi + L - 1) / L * L) = hi
      by omega]

theorem memberSpans_none (a : Flock.Program.Arg) (m lo hi : Nat) (hd : a.dims = none) :
    spansPos (Desc.memberSpans a m lo hi) = List.range' lo (hi - lo) := by
  unfold Desc.memberSpans
  rw [hd]
  simp [spansPos]

/-- **Reference `k` of a decoded sequence** is position `k` of its parts' (`Walk.seq_get`). -/
theorem partsPos_get (ps : Array Part) :
    (List.range (Seq.ofParts ps).count).map (fun k => refPos ((Seq.ofParts ps).get k)) = partsPos ps := by
  obtain ⟨-, -, hst, hcnt⟩ := Walk.ofParts_spec ps
  have hlen : (partsPos ps).length = (Seq.ofParts ps).count := by
    unfold partsPos
    rw [hcnt, List.length_flatMap]
    simp
  apply List.ext_getElem?
  intro k
  rw [List.getElem?_map]
  by_cases hk : k < (Seq.ofParts ps).count
  · obtain ⟨i, hi, h1, h2, hget⟩ := Walk.seq_get ps hk
    have hi' : i < ps.toList.length := by simpa using hi
    have hpi : ps.toList[i] = ps[i]! := by rw [Array.getElem_toList, getElem!_pos ps i hi]
    have hblock := flatMap_getElem?_block (fun q : Part => (List.range q.count).map fun k => refPos (q.get k))
      ps.toList i (k - (Seq.ofParts ps).starts[i]!) hi' (by simp only [List.length_map, List.length_range, hpi]; omega)
    have hsum : ((ps.toList.take i).map fun x => ((List.range x.count).map fun k => refPos (x.get k)).length).sum =
        (Seq.ofParts ps).starts[i]! := by
      rw [hst i hi]; simp
    rw [hsum, Nat.add_sub_cancel' h1] at hblock
    unfold partsPos
    rw [hblock, List.getElem?_range hk, Option.map_some, hget, hpi, List.getElem?_map,
      List.getElem?_range (by omega)]
    rfl
  · rw [List.getElem?_eq_none (by simp; omega)]
    symm
    apply List.getElem?_eq_none
    omega

/-- **A member's runs** are its parameter leaves' indices into the argument (`Resolve.memberIdx`). -/
theorem memberSpans_idx (a : Flock.Program.Arg) (m lo hi : Nat)
    (hL : ∀ d L, a.dims = some (d, L) → L = 0 → hi ≤ lo) :
    spansPos (Desc.memberSpans a m lo hi) = (List.range' lo (hi - lo)).map (Resolve.memberIdx a m) := by
  rcases hd : a.dims with _ | ⟨d, L⟩
  · rw [memberSpans_none a m lo hi hd]
    conv => lhs; rw [← List.map_id (List.range' lo (hi - lo))]
    apply List.map_congr_left
    intro t _
    simp [Resolve.memberIdx, hd]
  · rw [memberSpans_pos a m lo hi d L hd (hL d L hd)]
    apply List.map_congr_left
    intro t _
    simp [Resolve.memberIdx, hd]


/-! ## Gates, as `Width.segGates` names them -/

/-- A position as `Width.segGates` names it: a gate with its width; a parameter's or a node's leaf names none. -/
def gatePt (x : Tgt × Int) : Option (Int × Nat) :=
  match x.1 with
  | .gate w => some (x.2, w)
  | .ref _ _ => none

theorem flatMap_congr {α β : Type} {l : List α} {f g : α → List β} (h : ∀ a ∈ l, f a = g a) :
    l.flatMap f = l.flatMap g := by
  simp only [List.flatMap, List.map_congr_left h]

/-- **A piece's gates are its gate positions**, where it has a row or no columns (`pieceGates` reads a piece of no rows
as one row). -/
theorem pieceGates_pos (q : Piece) (h : q.rows ≠ 0 ∨ q.len = 0) :
    Width.pieceGates q = (pos q).filterMap gatePt := by
  unfold Width.pieceGates pos
  rw [List.filterMap_flatMap]
  rcases ht : q.tgt with ⟨p, i⟩ | w
  · dsimp only
    symm
    apply List.flatMap_eq_nil_iff.2
    intro r _
    rw [List.filterMap_map]
    simp [gatePt, Function.comp_def]
  · dsimp only
    rcases h with h | h
    · rw [show max q.rows 1 = q.rows by omega]
      apply flatMap_congr
      intro r _
      rw [List.filterMap_map]
      simp [gatePt, Function.comp_def]
    · rw [List.flatMap_eq_nil_iff.2 (fun _ _ => by rw [h]; rfl), List.flatMap_eq_nil_iff.2 (fun _ _ => by rw [h]; rfl)]

theorem piecesGates_pos (ps : Array Piece) (h : ∀ q ∈ ps.toList, q.rows ≠ 0 ∨ q.len = 0) :
    Width.piecesGates ps = (posAll ps).filterMap gatePt := by
  unfold Width.piecesGates posAll
  rw [List.filterMap_flatMap]
  exact flatMap_congr fun q hq => pieceGates_pos q (h q hq)

theorem shiftGates_rows (δ : Int) (ps : Array Piece) :
    ∀ q ∈ (Desc.shiftGates δ ps).toList, ∃ q' ∈ ps.toList, q.rows = q'.rows ∧ q.len = q'.len := by
  intro q hq
  simp only [Desc.shiftGates, Array.toList_map, List.mem_map] at hq
  obtain ⟨q', hq', rfl⟩ := hq
  refine ⟨q', hq', ?_⟩
  split <;> exact ⟨rfl, rfl⟩

/-- **A segment's gates are its gate positions**, where it has a copy and each piece a row or no columns. -/
theorem segGates_pos (s : Seg) (hr : s.reps ≠ 0) (h : ∀ q ∈ s.ps.toList, q.rows ≠ 0 ∨ q.len = 0) :
    Width.segGates s = (segPos s).filterMap gatePt := by
  unfold Width.segGates segPos
  rw [List.filterMap_flatMap]
  by_cases h1 : s.reps ≤ 1
  · rw [if_pos h1, show s.reps = 1 by omega, List.range_one, List.flatMap_singleton, copy_zero]
    exact piecesGates_pos s.ps h
  · rw [if_neg h1]
    apply flatMap_congr
    intro m _
    apply piecesGates_pos
    intro q hq
    obtain ⟨q', hq', e1, e2⟩ := shiftGates_rows _ _ q hq
    rw [e1, e2]; exact h q' hq'


/-! ## Parsed strided parts name no negative leaf -/

/-- One unfolding of `parsePartsIn`, as `Walk.parsePartsIn_step` takes it, for `PartNonneg`: a strided part's corners
are within its collection, so every position it names is (`Walk.strided_corners`). -/
theorem parsePartsIn_nonneg_step (leaves : Bool → Nat → Option Nat) (args : Nat → Nat → Option Seq)
    (hargs : ∀ k x s, args k x = some s → ∀ q ∈ s.parts, PartNonneg q) (n : Nat)
    (b : Nat.below (motive := fun _ => Lean.Json → Flock.V (Array Part)) (n + 1))
    (hb : ∀ x ps, b.1 x = .ok ps → ∀ q ∈ ps, PartNonneg q) (j : Lean.Json) (ps : Array Part)
    (h : Flock.Program.parsePartsIn._f leaves args (n + 1) b j = .ok ps) : ∀ q ∈ ps, PartNonneg q := by
  delta Flock.Program.parsePartsIn._f at h
  dsimp only at h
  cases j with
  | null => exact nomatch h
  | bool _ => exact nomatch h
  | num _ => exact nomatch h
  | str _ => exact nomatch h
  | arr _ => exact nomatch h
  | obj kvs =>
    dsimp only at h
    split at h
    · rename_i a _
      rcases a with _ | _ | _ | _ | ⟨⟨l⟩⟩ | _
      all_goals try (injection h; done)
      rcases l with _ | ⟨k, _ | ⟨x, _ | ⟨y, l⟩⟩⟩
      all_goals try (injection h; done)
      conv at h => lhs; whnf
      split at h
      all_goals try (injection h; done)
      split at h
      all_goals try (injection h; done)
      rename_i s hs
      injection h with h
      subst h
      exact hargs _ _ s hs
    · split at h
      · rename_i c _
        rcases c with _ | _ | _ | _ | cs | _
        all_goals try (injection h; done)
        dsimp only at h
        rcases hm : cs.mapM (fun x => b.1 x) with e | r <;> simp only [hm, bind, Except.bind] at h
        · exact absurd h (by intro h; injection h)
        injection h with h; subst h
        intro q hq
        rw [← Array.foldl_toList] at hq
        rcases Walk.mem_foldl_append r.toList #[] q hq with hq | ⟨y, hy, hq⟩
        · simp at hq
        · obtain ⟨x, -, hx⟩ := Walk.array_mapM_ok _ cs r hm y (by simpa using hy)
          exact hb x y hx q hq
      · split at h
        · rename_i a _
          rcases a with _ | _ | _ | _ | ⟨⟨l⟩⟩ | _
          all_goals try (injection h; done)
          rcases l with _ | ⟨s0, _ | ⟨i0, _ | ⟨b0, _ | ⟨c0, _ | ⟨y, l⟩⟩⟩⟩⟩
          all_goals try (injection h; done)
          conv at h => lhs; whnf
          split at h
          all_goals try (injection h; done)
          split at h
          · injection h with h; subst h
            intro q hq
            simp only [Array.mem_singleton] at hq
            subst hq
            trivial
          · split at h
            all_goals try (injection h; done)
            injection h with h; subst h
            intro q hq
            simp only [Array.mem_singleton] at hq
            subst hq
            trivial
        · split at h
          · rename_i a _
            rcases a with _ | _ | _ | _ | ⟨⟨l⟩⟩ | _
            all_goals try (injection h; done)
            rcases l with _ | ⟨s0, _ | ⟨i0, _ | ⟨b0, _ | ⟨n0, _ | ⟨o0, _ | ⟨j0, _ | ⟨c0, _ | ⟨y, l⟩⟩⟩⟩⟩⟩⟩⟩
            all_goals try (injection h; done)
            conv at h => lhs; whnf
            split at h
            all_goals try (injection h; done)
            rename_i _ cc inn ost ist c _
            split at h
            all_goals try (injection h; done)
            rename_i hinn
            try dsimp only at h
            split at h
            · injection h with h; subst h
              intro q hq
              simp only [Array.mem_singleton] at hq
              subst hq
              intro k hk
              exact absurd hk (Nat.not_lt_zero k)
            try dsimp only at h
            split at h
            all_goals try (injection h; done)
            rename_i p i hw
            injection h with h; subst h
            intro q hq
            simp only [Array.mem_singleton] at hq
            subst hq
            intro k hk
            have hin : 0 < inn := by simp at hinn; omega
            obtain ⟨⟨⟨x1, y1⟩, hp1, h1⟩, -⟩ := Walk.strided_corners cc ost ist hin hk
            split at hw
            all_goals try (injection hw; done)
            split at hw
            all_goals try (injection hw; done)
            split at hw
            all_goals rename_i hq0
            · rw [if_pos hq0] at hp1
              split at hw
              all_goals try (injection hw; done)
              rename_i hc
              simp only [Bool.and_eq_true, decide_eq_true_eq] at hc
              have e1 := le_trans hc.1 (Walk.foldl_min_le _ _ _ (Or.inl (List.mem_map_of_mem hp1)))
              dsimp only at h1 e1
              exact le_trans e1 h1
            · rw [if_neg hq0] at hp1
              split at hw
              all_goals try (injection hw; done)
              rename_i hc
              simp only [Bool.and_eq_true, decide_eq_true_eq] at hc
              have e1 := le_trans hc.1 (Walk.foldl_min_le _ _ _ (Or.inl (List.mem_map_of_mem hp1)))
              dsimp only at h1 e1
              exact le_trans e1 h1
          · split at h
            · rename_i e _
              rcases e with _ | _ | _ | _ | items | _
              all_goals try (injection h; done)
              dsimp only at h
              obtain ⟨rs, -, h⟩ := Walk.except_bind_ok h
              injection h with h; subst h
              intro q hq
              simp only [Array.mem_singleton] at hq
              subst hq
              trivial
            · exact absurd h (by intro h; injection h)

/-- **No position of a parsed strided part is negative**, given none of an argument an alias may name is: its corners'
least is within its collection (`parsePartsIn`'s `within`), so `Desc.ofPart`'s piece names its references
(`ofPart_pos`). -/
theorem parsePartsIn_nonneg (leaves : Bool → Nat → Option Nat) (args : Nat → Nat → Option Seq)
    (hargs : ∀ k x s, args k x = some s → ∀ q ∈ s.parts, PartNonneg q) :
    ∀ fuel j ps, Flock.Program.parsePartsIn fuel j leaves args = .ok ps → ∀ q ∈ ps, PartNonneg q := by
  intro fuel
  induction fuel with
  | zero =>
    intro j ps h
    delta Flock.Program.parsePartsIn at h
    exact nomatch h
  | succ n ih =>
    intro j ps h
    delta Flock.Program.parsePartsIn at h
    exact parsePartsIn_nonneg_step leaves args hargs n _
      (fun x ps hx => ih x ps (by delta Flock.Program.parsePartsIn; exact hx)) j ps h


/-! ## `ReadsExact` from what `nodeReads` gives each root node

`Desc.analyze` summarizes the reached Definitions (`memoOf`), then reads each root node with those summaries
(`readsOf`, `analyze_eq`), refusing a read that names no gate. So `ReadsExact` holds wherever `RootReads` does, which
speaks of `Desc.nodeReads`' segments as the positions they name (`segPos`): their gates are `Width.segGates`'
(`segGates_pos`), and the root node of a segment's first piece is `Desc.nodeOf`'s (`nodeOf_root`). -/

open Flock.Program (Prog)
open DescAgree (nodeCount rootOffs)

/-- `Desc.analyze`'s summaries of the reached Definitions, callees first. -/
def memoOf (P : Prog) : Except String (Array Desc.Summary) := do
  let seen := Desc.reached P
  let mut memo : Array Desc.Summary := #[]
  for d in [0:P.defs.size] do
    memo := memo.push (← if seen[d]! && d != P.root then Desc.summarize P memo d else pure default)
  return memo

/-- The context `Desc.analyze` reads the root's nodes in. -/
def rootCtx (P : Prog) (memo : Array Desc.Summary) : Desc.Ctx :=
  { P, memo, d := P.root, args := (P.defs[P.root]!).body.nodes.map fun n => n.args.map (Desc.ofSeq ·.seq) }

/-- `Desc.analyze`'s reads of the root's nodes, given the summaries. -/
def readsOf (P : Prog) (memo : Array Desc.Summary) : Except String (Array (Array Seg)) := do
  let body := (P.defs[P.root]!).body
  let c := rootCtx P memo
  let mut reads := #[]
  for k in [0:body.nodes.size] do
    let rs ← Desc.nodeReads c k
    if rs.any (·.ps.any fun q => match q.tgt with | .gate _ => false | _ => true) then throw "program"
    reads := reads.push rs
  return reads

theorem analyze_eq (P : Prog) : Desc.analyze P = (memoOf P >>= readsOf P) := by
  unfold Desc.analyze memoOf readsOf
  simp only [bind_assoc, pure_bind]
  rfl

/-- A loop that pushes one value per element, each one `G` holds of. -/
theorem forIn_push {ε α : Type} [Inhabited α] (f : Nat → Array α → Except ε (ForInStep (Array α)))
    (G : Nat → α → Prop) (hf : ∀ k r y, f k r = .ok y → ∃ x, G k x ∧ y = .yield (r.push x)) :
    ∀ (l : List Nat) (acc out : Array α), forIn l acc f = .ok out →
      out.size = acc.size + l.length ∧ (∀ i < acc.size, out[i]! = acc[i]!) ∧
        ∀ i (hi : i < l.length), G l[i] out[acc.size + i]!
  | [], acc, out, h => by
    simp only [List.forIn_nil, pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact ⟨by simp, fun _ _ => rfl, fun i hi => absurd hi (by simp)⟩
  | k :: l, acc, out, h => by
    rw [List.forIn_cons] at h
    rcases hk : f k acc with e | y <;> rw [hk] at h
    · exact absurd h (by intro h; cases h)
    obtain ⟨x, hx, rfl⟩ := hf k acc _ hk
    obtain ⟨h1, h2, h3⟩ := forIn_push f G hf l (acc.push x) out h
    simp only [Array.size_push] at h1 h2 h3
    refine ⟨by simp only [List.length_cons]; omega, fun i hi => ?_, fun i hi => ?_⟩
    · rw [h2 i (by omega), getElem!_pos (acc.push x) i (by simp; omega), getElem!_pos acc i hi,
        Array.getElem_push_lt]
    · rcases i with _ | i
      · rw [Nat.add_zero, h2 acc.size (by omega), getElem!_pos (acc.push x) acc.size (by simp),
          Array.getElem_push_eq]
        exact hx
      · have := h3 i (by simp at hi; omega)
        simp only [List.getElem_cons_succ]
        rw [show acc.size + (i + 1) = acc.size + 1 + i by omega]
        exact this


theorem pieces_gates (rs : Array Seg)
    (h : (rs.any fun x => x.ps.any fun q => match q.tgt with | .gate _ => false | _ => true) = false) :
    ∀ s ∈ rs, ∀ q ∈ s.ps, ∃ w, q.tgt = .gate w := by
  intro s hs q hq
  rw [Array.any_eq_false] at h
  obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.1 hs
  have h2 := h i hi
  rw [Bool.not_eq_true, Array.any_eq_false] at h2
  obtain ⟨j, hj, rfl⟩ := Array.mem_iff_getElem.1 hq
  have h3 := h2 j hj
  revert h3
  rcases (rs[i].ps[j]).tgt with ⟨p, i'⟩ | w
  · simp
  · exact fun _ => ⟨w, rfl⟩

/-- What `readsOf` gives: per root node, `nodeReads`' segments, every piece of which names gates. -/
theorem readsOf_spec (P : Prog) (memo : Array Desc.Summary) (reads : Array (Array Seg))
    (h : readsOf P memo = .ok reads) :
    reads.size = nodeCount P ∧ ∀ k < nodeCount P, Desc.nodeReads (rootCtx P memo) k = .ok reads[k]! ∧
      ∀ s ∈ reads[k]!, ∀ q ∈ s.ps, ∃ w, q.tgt = .gate w := by
  unfold readsOf at h
  simp only [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size] at h
  obtain ⟨r, hr, h⟩ := Walk.except_bind_ok h
  simp only [pure, Except.pure, Except.ok.injEq] at h
  subst h
  obtain ⟨h1, -, h3⟩ := forIn_push _ (fun k rs => Desc.nodeReads (rootCtx P memo) k = .ok rs ∧
      (rs.any fun x => x.ps.any fun q => match q.tgt with | .gate _ => false | _ => true) = false)
    (fun k acc y hy => by
      obtain ⟨rs, hrs, hy⟩ := Walk.except_bind_ok hy
      by_cases hb : (rs.any fun x => x.ps.any fun q => match q.tgt with | .gate _ => false | _ => true) = true
      · rw [if_pos hb] at hy; exact absurd hy (by intro h; cases h)
      · rw [if_neg hb] at hy
        simp only [pure, Except.pure, Except.ok.injEq] at hy
        exact ⟨rs, ⟨hrs, by simpa using hb⟩, hy.symm⟩) _ #[] r hr
  have hn : ((P.defs[P.root]!).body.nodes.size - 0 + 1 - 1) / 1 = nodeCount P := by
    simp [nodeCount, DescAgree.rootBody]
  rw [List.length_range', hn] at h1
  refine ⟨by simpa using h1, fun k hk => ?_⟩
  have := h3 k (by rw [List.length_range', hn]; exact hk)
  simp only [List.getElem_range', Array.size_empty, Nat.zero_add, Nat.one_mul] at this
  exact ⟨this.1, pieces_gates _ this.2⟩


/-- **`Desc.nodeOf` finds the root node holding a gate**, over offsets laid out as `rootLaidOut` checks
(`Walk.bisect_spec`). -/
theorem nodeOf_eq (offs : Array Nat) (K g k : Nat) (hsz : offs.size = K + 1) (h0 : offs[0]! = 0)
    (hmono : ∀ i < K, offs[i]! ≤ offs[i + 1]!) (hk : k < K) (hg : offs[k]! ≤ g ∧ g < offs[k + 1]!) :
    Desc.nodeOf offs g = k := by
  let f : Nat → Nat × Nat → Id (ForInStep (Nat × Nat)) := fun _ (st : Nat × Nat) =>
    have lo := st.1; have hi := st.2
    if hi - lo ≤ 1 then pure (ForInStep.done (lo, hi))
    else
      have mid := (lo + hi) / 2
      if offs[mid]! ≤ g then pure (ForInStep.yield (mid, hi)) else pure (ForInStep.yield (lo, mid))
  obtain ⟨b1, b2, b3, b4, b5⟩ := Walk.bisect_spec offs g (offs.size - 1) f (fun _ _ _ => rfl)
    (List.range' 0 offs.size 1) 0 (offs.size - 1) (by omega) (le_refl _) (by rw [h0]; omega)
    (fun h => absurd h (lt_irrefl _)) (by simp; omega)
  have hr : Desc.nodeOf offs g = (forIn (List.range' 0 offs.size 1) (0, offs.size - 1) f : Id (Nat × Nat)).1 := by
    unfold Desc.nodeOf
    have hfor : ∀ (g : Nat → Nat × Nat → Id (ForInStep (Nat × Nat))) (init : Nat × Nat),
        (forIn [:offs.size] init g : Id (Nat × Nat)) = forIn (List.range' 0 offs.size) init g := by
      intro g init
      rw [Std.Legacy.Range.forIn_eq_forIn_range']
      simp [Std.Legacy.Range.size]
    simp only [hfor]
    rfl
  generalize (forIn (List.range' 0 offs.size 1) (0, offs.size - 1) f : Id (Nat × Nat)) = r at hr b1 b2 b3 b4 b5
  rw [hsz, Nat.add_sub_cancel] at b2 b4
  have hr2 : r.2 = r.1 + 1 := by omega
  have hlt : g < offs[r.1 + 1]! := by
    rcases Nat.lt_or_ge r.2 K with h | h
    · rw [← hr2]; exact b4 h
    · have : r.1 + 1 = K := by omega
      rw [this]; exact Nat.lt_of_lt_of_le hg.2 (DescAgree.offs_mono offs K hmono (k + 1) K (by omega) (le_refl _))
  rw [hr]
  exact DescAgree.node_unique offs K hmono (by omega) hk ⟨b3, hlt⟩ hg

/-- The root node `Desc.nodeOf` finds for a gate of root node `k` is `k`. -/
theorem nodeOf_root (P : Prog) (hl : Desc.rootLaidOut P = true) {k a : Nat} (hk : k < nodeCount P)
    (ha : a ∈ nodeGates P k) : Desc.nodeOf (rootOffs P) a = k := by
  obtain ⟨hsz, h0, -, hmono⟩ := DescAgree.rootLaidOut_spec P hl
  exact nodeOf_eq (rootOffs P) (nodeCount P) a k hsz h0 hmono hk ((DescAgree.mem_nodeGates P k a).1 ha)


/-- **Open: `nodeReads`' segments at the root, as the positions they name.** On a Program `Desc.analyze` reads, with
its summaries `memo` and `circuitOf`'s circuit `c`: for root node `k`, whose segments `rs` (`Desc.nodeReads`) have
pieces naming gates only, every segment of positive count has pieces of positive count and names gates `a` of width
`c.w[a]` of one root node; and the gates the segments of positive count name are exactly the gates outside node `k`
that its gates read. `readsExact_of_rootReads` gives `Novel.ReadsExact` from it. -/
def RootReads : Prop :=
  ∀ (bytes : ByteArray) (P : Prog) (memo : Array Desc.Summary) (c : Circuit),
    Flock.Program.decode bytes = .ok P → P.orderViolation = none → Desc.rootLaidOut P = true →
    Desc.inputsAtRoot P = true → memoOf P = .ok memo → circuitOf P = .ok c →
    ∀ k < nodeCount P, ∀ rs, Desc.nodeReads (rootCtx P memo) k = .ok rs →
      (∀ s ∈ rs, ∀ q ∈ s.ps, ∃ w, q.tgt = .gate w) →
      (∀ s ∈ rs, s.count ≠ 0 → (∀ q ∈ s.ps.toList, 0 < q.count) ∧
        ∃ k' < nodeCount P, ∀ x ∈ segPos s, ∃ a : Nat, x = (.gate c.w[a]!, (a : Int)) ∧ a ∈ nodeGates P k') ∧
      ∀ a : Nat, (∃ s ∈ rs, s.count ≠ 0 ∧ ∃ w, (Tgt.gate w, (a : Int)) ∈ segPos s) ↔
        ∃ g ∈ nodeGates P k, a ∈ c.reads[g]! ∧ a ∉ nodeGates P k

theorem reps_ne_zero {s : Seg} (h : s.count ≠ 0) : s.reps ≠ 0 := by
  intro h0; apply h; simp [Desc.Seg.count, h0]

theorem rows_ok {s : Seg} (h : ∀ q ∈ s.ps.toList, 0 < q.count) : ∀ q ∈ s.ps.toList, q.rows ≠ 0 ∨ q.len = 0 := by
  intro q hq
  have := h q hq
  left; intro h0; simp [Desc.Piece.count, h0] at this

/-- A segment of a copy names its first piece's first position. -/
theorem first_mem_segPos {s : Seg} {q : Piece} (hr : s.reps ≠ 0) (hq : s.ps[0]? = some q) (hc : 0 < q.count) :
    (q.tgt, q.base) ∈ segPos s := by
  unfold segPos
  refine List.mem_flatMap.2 ⟨0, List.mem_range.2 (by omega), ?_⟩
  rw [copy_zero]
  refine List.mem_flatMap.2 ⟨q, ?_, ?_⟩
  · obtain ⟨h0, rfl⟩ := Array.getElem?_eq_some_iff.1 hq
    exact Array.getElem_mem_toList h0
  · rw [pos_eq_map]
    refine List.mem_map.2 ⟨0, List.mem_range.2 hc, ?_⟩
    simp [ppt]

theorem gatePt_eq {y : Tgt × Int} {a : Int} {w : Nat} (h : gatePt y = some (a, w)) : y = (Tgt.gate w, a) := by
  rcases y with ⟨t, i⟩
  rcases t with ⟨p, i'⟩ | w'
  · simp [gatePt] at h
  · simp only [gatePt, Option.some.injEq, Prod.mk.injEq] at h
    rw [h.1, h.2]

/-- **`ReadsExact` from `RootReads`**: `analyze`'s reads are `nodeReads`' (`readsOf_spec`), their gates the gate
positions they name (`segGates_pos`), and a segment's first piece names a gate of the one root node its gates are in,
which `Desc.nodeOf` finds (`nodeOf_root`). -/
theorem readsExact_of_rootReads (h : RootReads) : Novel.ReadsExact := by
  intro bytes P reads c hd ho hl hi ha hc k hk
  rw [analyze_eq] at ha
  obtain ⟨memo, hm, hr⟩ := Walk.except_bind_ok ha
  obtain ⟨-, hks⟩ := readsOf_spec P memo reads hr
  obtain ⟨hnr, hg⟩ := hks k hk
  obtain ⟨h1, h2⟩ := h bytes P memo c hd ho hl hi hm hc k hk _ hnr hg
  refine ⟨fun s hs hcnt q hq => ?_, fun a => ?_⟩
  · obtain ⟨hpos, k', hk', hx⟩ := h1 s hs hcnt
    have hq0 := first_mem_segPos (reps_ne_zero hcnt) hq
      (hpos q (by obtain ⟨h0, rfl⟩ := Array.getElem?_eq_some_iff.1 hq; exact Array.getElem_mem_toList h0))
    obtain ⟨a, ha, hak⟩ := hx _ hq0
    have hb : q.base.toNat = a := by
      simp only [Prod.mk.injEq] at ha
      rw [ha.2]; simp
    rw [hb, nodeOf_root P hl hk' hak]
    refine ⟨hk', fun x hxs => ?_⟩
    rw [segGates_pos s (reps_ne_zero hcnt) (rows_ok hpos)] at hxs
    obtain ⟨y, hy, hgy⟩ := List.mem_filterMap.1 hxs
    obtain ⟨b, rfl, hb'⟩ := hx y hy
    simp only [gatePt, Option.some.injEq] at hgy
    exact ⟨b, hgy.symm, hb'⟩
  · rw [← h2 a]
    constructor
    · rintro ⟨s, hs, hcnt, w, hw⟩
      rw [segGates_pos s (reps_ne_zero hcnt) (rows_ok (h1 s hs hcnt).1)] at hw
      obtain ⟨y, hy, hgy⟩ := List.mem_filterMap.1 hw
      exact ⟨s, hs, hcnt, w, by rw [← gatePt_eq hgy]; exact hy⟩
    · rintro ⟨s, hs, hcnt, w, hw⟩
      refine ⟨s, hs, hcnt, w, ?_⟩
      rw [segGates_pos s (reps_ne_zero hcnt) (rows_ok (h1 s hs hcnt).1)]
      exact List.mem_filterMap.2 ⟨_, hw, rfl⟩

end NciCheck.Properties.SecurityProofs.ReadsExact
