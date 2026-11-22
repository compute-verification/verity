module

public import Proofs.Properties.InferenceOnly.Fresh
import all Init.Data.Array.QSort.Basic
import Batteries.Tactic.OpenPrivate
import Mathlib.Analysis.SpecialFunctions.Pow.NNReal
import Mathlib.Data.NNRat.Floor
import Mathlib.Tactic.ENatToNat
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.Measurability.Init
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Tactic.Polynomial.Basic
import Mathlib.Tactic.ReduceModChar
import Proofs.Core.QSort
import Proofs.Properties.InferenceOnly.Lemmas
import Proofs.Properties.Poub.Remote.Sort
import Std.Data.HashSet.Lemmas

/-!
# What `Desc`'s pieces and segments name, and the sequences `ofPart` and `ofSeq` make

The positions a piece, a segment and a sequence of segments name in order (`pos`, `posAll`, `segPos`, `psPos`), and
`Desc`'s operations on them: slices, copies, `PS.of`, `coalesce`, and a decoded sequence's parts (`ofSeq_pos`).
`ReadsExact` reads them toward `Fresh.ReadsExact`.
-/

@[expose] public section

namespace InferenceOnlyVerifier.Properties.SecurityProofs.ReadsExact

open InferenceOnlyVerifier.Desc (Piece Seg PS Tgt)


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

private theorem flatMap_congr_mem {α β : Type} {f g : α → List β} : ∀ {l : List α}, (∀ x ∈ l, f x = g x) →
    l.flatMap f = l.flatMap g
  | [], _ => rfl
  | a :: l, h => by
    rw [List.flatMap_cons, List.flatMap_cons, h a (by simp), flatMap_congr_mem (fun x hx => h x (by simp [hx]))]

theorem flatMap_range_congr {β : Type} {n : Nat} {f g : Nat → List β} (h : ∀ m < n, f m = g m) :
    (List.range n).flatMap f = (List.range n).flatMap g :=
  flatMap_congr_mem fun m hm => h m (List.mem_range.1 hm)

/-- A piece's positions in order: row `r`, column `t` is `(tgt, base + rowOff r + t·stride)`. -/
def pos (q : Piece) : List (Tgt × Int) :=
  (List.range q.rows).flatMap fun (r : Nat) => (List.range q.len).map fun (t : Nat) =>
    (q.tgt, q.base + q.rowOff r + (t : Int) * q.stride)

/-- Position `t` of a piece: row `t / len`, column `t % len`. -/
def ppt (q : Piece) (t : Nat) : Int :=
  q.base + q.rowOff (t / q.len) + ((t % q.len : Nat) : Int) * q.stride

theorem rowOff_flat (q : Piece) (h : q.lv = []) (r : Nat) : q.rowOff r = (r : Int) * q.rowStride := by
  simp [Desc.Piece.rowOff, h]

theorem rowOff_with (q : Piece) (t : Tgt) (b : Int) (r : Nat) : ({ q with tgt := t, base := b } : Piece).rowOff r = q.rowOff r := by
  simp [Desc.Piece.rowOff]

/-- What `Piece.rowsFrom` keeps: its rows are `q`'s from `lo`. -/
theorem rowsFrom_spec (q : Piece) (lo n : Nat) :
    (q.rowsFrom lo n).tgt = q.tgt ∧ (q.rowsFrom lo n).stride = q.stride ∧ (q.rowsFrom lo n).len = q.len ∧
      (q.rowsFrom lo n).rows = n ∧
      ∀ r < n, (q.rowsFrom lo n).base + (q.rowsFrom lo n).rowOff r = q.base + q.rowOff (lo + r) := by
  unfold Desc.Piece.rowsFrom
  by_cases h1 : q.lv.isEmpty = true
  · have hl : q.lv = [] := by simpa using h1
    simp only [h1, ↓reduceIte]
    refine ⟨by simp, by simp, by simp, by simp, fun r _ => ?_⟩
    simp only [Desc.Piece.rowOff, hl, List.isEmpty_nil, ↓reduceIte]
    push_cast; ring
  · have hl : q.lv ≠ [] := by simpa using h1
    simp only [h1, Bool.false_eq_true, ↓reduceIte]
    by_cases h2 : (n == 1) = true
    · have hn : n = 1 := by simpa using h2
      simp only [h2, ↓reduceIte, Desc.Piece.row]
      refine ⟨by simp, by simp, by simp, by simp [hn], fun r hr => ?_⟩
      obtain rfl : r = 0 := by omega
      simp [Desc.Piece.rowOff]
    · simp only [h2, Bool.false_eq_true, ↓reduceIte]
      refine ⟨by simp, by simp, by simp, by simp, fun r _ => ?_⟩
      simp [Desc.Piece.rowOff, hl, Nat.add_assoc]

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
theorem pos_run (q : Piece) (r s e : Nat) (B : Int) (hB : B = q.base + q.rowOff r + (s : Int) * q.stride)
    (hse : s ≤ e) (he : e ≤ q.len) :
    pos ⟨q.tgt, B, q.stride, e - s, 1, 0, [], 0⟩ =
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
    rw [h1, h2, rowOff_flat ⟨q.tgt, B, q.stride, e - s, 1, 0, [], 0⟩ rfl, hB]
    push_cast; ring
  · rw [List.getElem?_eq_none (by simp; omega), if_neg h]; rfl


/-- Whole rows `[lo, lo + n)` of `q` (`Piece.rowsFrom`): its positions `[lo · len, (lo + n) · len)`. -/
theorem pos_rows (q : Piece) (lo n : Nat) :
    pos (q.rowsFrom lo n) = (List.range' (lo * q.len) (n * q.len)).map fun t => (q.tgt, ppt q t) := by
  obtain ⟨ht, hs, hl, hr, hoff⟩ := rowsFrom_spec q lo n
  rw [pos_eq_map]
  apply List.ext_getElem?
  intro t
  simp only [Desc.Piece.count, List.getElem?_map, hr, hl]
  rw [range'_getElem?]
  by_cases h : t < n * q.len
  · rw [List.getElem?_range h, if_pos h]
    simp only [Option.map_some, Option.some.injEq, Prod.mk.injEq, ht, true_and]
    have hL : 0 < q.len := by
      rcases Nat.eq_zero_or_pos q.len with h0 | h0
      · rw [h0, Nat.mul_zero] at h; omega
      · exact h0
    have h1 : (lo * q.len + t) / q.len = lo + t / q.len := by
      rw [Nat.mul_comm, Nat.mul_add_div hL]
    have h2 : (lo * q.len + t) % q.len = t % q.len := by
      rw [Nat.mul_comm, Nat.mul_add_mod]
    unfold ppt
    rw [h1, h2, hl, hs, ← hoff _ ((Nat.div_lt_iff_lt_mul hL).2 h)]
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
    simp only [Nat.zero_mul, Nat.zero_add] at this
    exact this
  rw [if_neg hr]
  have hpart : ∀ r s e : Nat, s ≤ e → e ≤ q.len →
      posAll (if s < e then #[(⟨q.tgt, q.base + q.rowOff r + (s : Int) * q.stride, q.stride, e - s, 1, 0, [], 0⟩ :
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
    have hmid : ∀ lo, lo ≤ r1 → posAll (if lo < r1 then #[q.rowsFrom lo (r1 - lo)] else #[]) =
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
  rcases q with ⟨t, base, stride, len, rows, rowStride, lv, start⟩
  cases t with
  | gate w =>
    simp only [pos, List.map_flatMap, List.map_map, Desc.Piece.rowOff]
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

theorem copy_flat (s : Seg) (h : s.lanes = false) (m : Nat) : s.copy m = Desc.shiftGates (m * s.shift) s.ps := by
  simp [Desc.Seg.copy, h]

theorem copy_rowed (s : Seg) (h : s.lanes = true) (m : Nat) : s.copy m = s.ps.map (·.row m) := by
  simp [Desc.Seg.copy, h]

theorem lanes_of_reps {s : Seg} (h : s.reps ≤ 1) : s.lanes = false := by
  unfold Desc.Seg.lanes; cases s.rowed <;> simp <;> omega

theorem posAll_map (ps : Array Piece) (f : Piece → Piece) : posAll (ps.map f) = ps.toList.flatMap fun p => pos (f p) := by
  simp [posAll, List.flatMap_map]

/-- Row `r` of `p`: its `len` positions from `base + rowOff r`. -/
theorem pos_row (p : Piece) (r : Nat) :
    pos (p.row r) = (List.range p.len).map fun (t : Nat) => (p.tgt, p.base + p.rowOff r + (t : Int) * p.stride) := by
  unfold pos
  simp only [Desc.Piece.row, List.range_one, List.flatMap_singleton]
  apply List.map_congr_left
  intro t _
  simp [Desc.Piece.rowOff]

theorem pos_row_rowsFrom (p : Piece) (lo n m : Nat) (hm : m < n) : pos ((p.rowsFrom lo n).row m) = pos (p.row (lo + m)) := by
  obtain ⟨ht, hs, hl, -, hoff⟩ := rowsFrom_spec p lo n
  rw [pos_row, pos_row, ht, hs, hl, hoff m hm]

theorem copy_pos (s : Seg) (h : s.lanes = false) (m : Nat) :
    posAll (s.copy m) = (posAll s.ps).map (shiftPos ((m : Int) * s.shift)) := by
  rw [copy_flat s h]; exact shiftGates_pos _ _

theorem segPos_one (q : Piece) : segPos (Seg.one q) = pos q := by
  simp only [segPos, Seg.one, Desc.Seg.copy, Desc.Seg.lanes, Bool.false_and, Bool.false_eq_true, ↓reduceIte,
    List.range_one, List.flatMap_singleton,
    shiftGates_pos, Int.mul_zero, shiftPos_zero, List.map_id, posAll_singleton]

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

/-- Every copy of a segment names as many positions as copy 0. -/
theorem copy_length (s : Seg) (m : Nat) : (posAll (s.copy m)).length = (posAll (s.copy 0)).length := by
  cases h : s.lanes
  · rw [copy_flat s h, copy_flat s h, shiftGates_length, shiftGates_length]
  · rw [copy_rowed s h, copy_rowed s h, posAll_map, posAll_map]
    simp only [List.length_flatMap, pos_row, List.length_map, List.length_range]

theorem copy_zero (s : Seg) (h : s.lanes = false) : posAll (s.copy 0) = posAll s.ps := by
  rw [copy_pos s h, show ((0 : Nat) : Int) * s.shift = 0 by simp, shiftPos_zero, List.map_id]

theorem segPos_one_rep {s : Seg} (h : s.reps = 1) : segPos s = posAll s.ps := by
  unfold segPos
  rw [h, List.range_one, List.flatMap_singleton, copy_pos s (lanes_of_reps h.le)]
  simp [shiftPos_zero]

theorem copy_copy (s : Seg) (h : s.lanes = false) (lo m : Nat) :
    posAll (Desc.shiftGates ((m : Int) * s.shift) (s.copy lo)) = posAll (s.copy (lo + m)) := by
  rw [shiftGates_pos, copy_pos s h, copy_pos s h, List.map_map, shiftPos_shiftPos, Int.natCast_add, Int.add_mul]

/-- Position `t` of a segment whose pieces name `s.len` positions: position `t % len` of copy `t / len`. -/
def spt (s : Seg) (t : Nat) : Tgt × Int := (posAll (s.copy (t / s.len))).getD (t % s.len) default

theorem segPos_eq_map (s : Seg) (hlen : (posAll (s.copy 0)).length = s.len) :
    segPos s = (List.range s.count).map (spt s) :=
  blocks_eq_map (fun m => posAll (s.copy m)) s.len (fun m => by rw [copy_length, hlen]) default s.reps

theorem spt_add (s : Seg) (m t : Nat) (ht : t < s.len) :
    spt s (m * s.len + t) = (posAll (s.copy m)).getD t default := by
  have hL : 0 < s.len := by omega
  unfold spt
  rw [show m * s.len + t = t + s.len * m by rw [Nat.mul_comm]; omega, Nat.add_mul_div_left _ _ hL,
    Nat.add_mul_mod_self_left, Nat.div_eq_of_lt ht, Nat.mod_eq_of_lt ht, Nat.zero_add]

/-- **A slice of a segment** is its positions `[a, b)`. -/
theorem segSlice_pos (s : Seg) (hlen : (posAll (s.copy 0)).length = s.len) (a b : Nat) (hb : b ≤ s.count) :
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
    have hlen' := hlen
    rw [copy_zero s (lanes_of_reps hr)] at hlen'
    have hr1 : s.reps = 1 := by
      rcases Nat.lt_or_ge s.reps 1 with h | h
      · rw [hcnt, show s.reps = 0 by omega, Nat.zero_mul] at hb; omega
      · omega
    have e : segPos s = posAll s.ps := segPos_one_rep hr1
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
      segsPos (if lo < m1 then
          (if s.lanes && lo + 1 < m1 then #[{ s with ps := s.ps.map (·.rowsFrom lo (m1 - lo)), reps := m1 - lo }]
          else #[{ s with ps := s.copy lo, reps := m1 - lo, rowed := false }]) else #[]) =
        (List.range' (lo * s.len) (m1 * s.len - lo * s.len)).map (spt s) := by
    intro lo m1 hlo
    have hgen : ∀ X : Seg, X.reps = m1 - lo → (∀ m < m1 - lo, posAll (X.copy m) = posAll (s.copy (lo + m))) →
        segsPos #[X] = (List.range' (lo * s.len) (m1 * s.len - lo * s.len)).map (spt s) := by
      intro X hX hc
      rw [segsPos_singleton]
      have e : segPos X = (List.range (m1 - lo)).flatMap fun m => posAll (s.copy (lo + m)) := by
        unfold segPos; rw [hX]; exact flatMap_range_congr hc
      rw [e, blocks_eq_map (fun m => posAll (s.copy (lo + m))) s.len (fun m => by rw [copy_length, hlen]) default,
        List.range'_eq_map_range, List.map_map, Nat.sub_mul]
      apply List.map_congr_left
      intro t _
      simp only [Function.comp, spt]
      rw [show lo * s.len + t = t + s.len * lo by rw [Nat.mul_comm]; omega, Nat.add_mul_div_left _ _ hL,
        Nat.add_mul_mod_self_left, Nat.add_comm (t / s.len) lo]
    split
    · split
      · rename_i hrw
        simp only [Bool.and_eq_true, decide_eq_true_eq] at hrw
        refine hgen _ rfl fun m hm => ?_
        have hl : ({ s with ps := s.ps.map (·.rowsFrom lo (m1 - lo)), reps := m1 - lo } : Seg).lanes = true := by
          have h1 := hrw.1
          simp only [Desc.Seg.lanes, Bool.and_eq_true, decide_eq_true_eq] at h1 ⊢
          exact ⟨h1.1, by omega⟩
        rw [copy_rowed _ hl, copy_rowed s hrw.1,
          posAll_map, posAll_map, Array.toList_map, List.flatMap_map]
        exact flatMap_congr_mem fun p _ => pos_row_rowsFrom p lo _ m hm
      · rename_i hrw
        refine hgen _ rfl fun m hm => ?_
        rw [copy_flat { s with ps := s.copy lo, reps := m1 - lo, rowed := false } rfl]
        cases hs : s.lanes
        · exact copy_copy s hs lo m
        · simp only [hs, Bool.true_and, decide_eq_true_eq] at hrw
          obtain rfl : m = 0 := by omega
          simp [shiftGates_pos, shiftPos_zero]
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

/-- A segment whose copies name `len` positions each. -/
def SegOk (s : Seg) : Prop := (posAll (s.copy 0)).length = s.len

theorem segOk_flat {s : Seg} (h : s.lanes = false) : SegOk s ↔ (posAll s.ps).length = s.len := by
  unfold SegOk; rw [copy_flat s h, shiftGates_length]

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

/-- A part's references never step below leaf 0: a strided part's pattern (`parsePartsIn` checks its corners), and a
grid's lanes at every period and offset in a run (`parsePartsIn` checks each lane's least leaf). -/
def PartNonneg : Part → Prop
  | .strided _ _ b inner os is c => ∀ k < c, 0 ≤ b + ((k / inner : Nat) : Int) * os + ((k % inner : Nat) : Int) * is
  | .grid run lanes => ∀ l ∈ lanes, ∀ t < (lanes[0]?.map Flock.Program.Lane.periods).getD 0, ∀ o < run, 0 ≤ l.leafAt t o
  | _ => True

theorem pos_one (t : Tgt) (base stride : Int) (len : Nat) (rs : Int) (st : Nat) :
    pos ⟨t, base, stride, len, 1, rs, [], st⟩ = (List.range len).map fun (k : Nat) => (t, base + (k : Int) * stride) := by
  simp [pos, Desc.Piece.rowOff]

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

theorem form_eq_scan {f : Flock.Program.Form} (h : (f == .scan) = true) : f = .scan := by
  cases f <;> first | rfl | cases h

/-! ## A summary's return coalesced

`Desc.coalesce` joins a return's one-piece segments where they continue one progression (`Piece.join`) and stacks equal
runs as rows (`Piece.stack`): it names the same positions in the same order (`coalesce_pos`), and each segment it gives
is one it was given or one piece of positive count (`coalesce_mem`). -/

theorem segsPos_push (X : Array Seg) (C : Seg) : segsPos (X.push C) = segsPos X ++ segPos C := by
  rw [Array.push_eq_append, segsPos_append, segsPos_singleton]

theorem gate_beq_eq {w : Nat} {t : Tgt} (h : (Tgt.gate w == t) = true) : t = .gate w := by
  cases t with
  | ref p i => cases h
  | gate w' =>
    have e : (Tgt.gate w == Tgt.gate w') = (w == w') := rfl
    rw [e, beq_iff_eq] at h
    rw [h]

/-- **Joining two runs of a progression** names `p`'s positions then `q`'s. -/
theorem join_pos (p q r : Piece) (h : Desc.Piece.join p q = some r) : pos r = pos p ++ pos q ∧ 0 < r.count := by
  rcases p with ⟨tp, bp, sp, lp, rp, rsp, lvp, stp⟩
  rcases q with ⟨tq, bq, sq, lq, rq, rsq, lvq, stq⟩
  unfold Desc.Piece.join at h
  cases tp with
  | ref _ _ => cases h
  | gate w =>
    dsimp only at h
    generalize hst0 : (if lp == 1 then bq - bp else sp) = st at h
    split at h
    · rename_i hc
      simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq, Bool.or_eq_true] at hc
      obtain ⟨⟨⟨⟨⟨⟨⟨⟨⟨htq, hrp⟩, hrq⟩, hlp⟩, hlq⟩, -⟩, hst⟩, hb⟩, hlvp⟩, hlvq⟩ := hc
      have hlvp' : lvp = [] := by simpa using hlvp
      have hlvq' : lvq = [] := by simpa using hlvq
      cases h
      cases gate_beq_eq htq
      subst hrp hrq hlvp' hlvq'
      refine ⟨?_, by simp [Desc.Piece.count]; omega⟩
      rw [pos_one, pos_one, pos_one, List.range_add, List.map_append, List.map_map]
      have hfirst : ((List.range lp).map fun (k : Nat) => (Tgt.gate w, bp + (k : Int) * st)) =
          (List.range lp).map fun (k : Nat) => (Tgt.gate w, bp + (k : Int) * sp) := by
        by_cases h1 : lp = 1
        · subst h1; simp
        · have h1' : (lp == 1) = false := by simpa using h1
          simp only [h1', Bool.false_eq_true, ite_false] at hst0
          rw [hst0]
      rw [hfirst]
      congr 1
      apply List.map_congr_left
      intro k hk
      simp only [Function.comp, Prod.mk.injEq, true_and]
      rcases hst with hst | hst
      · subst hst
        have : k = 0 := by simpa using hk
        subst this
        simp only [Nat.add_zero, Int.natCast_zero, Int.zero_mul, Int.add_zero]
        omega
      · subst hst
        rw [Int.natCast_add, Int.add_mul]
        omega
    · cases h

/-- **Stacking a row under rows** names `p`'s positions then `q`'s. -/
theorem stack_pos (p q r : Piece) (h : Desc.Piece.stack p q = some r) : pos r = pos p ++ pos q ∧ 0 < r.count := by
  rcases p with ⟨tp, bp, sp, lp, rp, rsp, lvp, stp⟩
  rcases q with ⟨tq, bq, sq, lq, rq, rsq, lvq, stq⟩
  unfold Desc.Piece.stack at h
  cases tp with
  | ref _ _ => cases h
  | gate w =>
    dsimp only at h
    generalize hrs0 : (if rp == 1 then bq - bp else rsp) = rs at h
    split at h
    · rename_i hc
      simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq, Bool.or_eq_true] at hc
      obtain ⟨⟨⟨⟨⟨⟨⟨⟨⟨htq, hrp⟩, hrq⟩, hlp⟩, hlq⟩, hst⟩, -⟩, hb⟩, hlvp⟩, hlvq⟩ := hc
      have hlvp' : lvp = [] := by simpa using hlvp
      have hlvq' : lvq = [] := by simpa using hlvq
      cases h
      cases gate_beq_eq htq
      subst hrq hlq hlvp' hlvq'
      refine ⟨?_, by simp only [Desc.Piece.count]; exact Nat.mul_pos (Nat.succ_pos _) hlp⟩
      rw [pos_one]
      unfold pos
      simp only [Desc.Piece.rowOff, List.isEmpty_nil, ↓reduceIte]
      rw [List.range_succ, List.flatMap_append, List.flatMap_singleton]
      have hfirst : ((List.range rp).flatMap fun (r' : Nat) => (List.range lq).map fun (t : Nat) =>
            (Tgt.gate w, bp + (r' : Int) * rs + (t : Int) * sp)) =
          (List.range rp).flatMap fun (r' : Nat) => (List.range lq).map fun (t : Nat) =>
            (Tgt.gate w, bp + (r' : Int) * rsp + (t : Int) * sp) := by
        by_cases h1 : rp = 1
        · subst h1; simp
        · have h1' : (rp == 1) = false := by simpa using h1
          simp only [h1', Bool.false_eq_true, ite_false] at hrs0
          rw [hrs0]
      rw [hfirst]
      congr 1
      apply List.map_congr_left
      intro k hk
      simp only [Prod.mk.injEq, true_and]
      rcases hst with hst | hst
      · subst hst
        have : k = 0 := by simpa using hk
        subst this
        simp only [Int.natCast_zero, Int.zero_mul, Int.add_zero]
        omega
      · subst hst
        omega
    · cases h

/-- The piece of a segment of one copy of one piece names the segment's positions. -/
theorem single_pos (s : Seg) (p : Piece) (h : s.single = some p) : segPos s = pos p := by
  rcases s with ⟨⟨l⟩, len, reps, shift, rowed⟩
  unfold Desc.Seg.single at h
  split at h
  · rename_i hc
    simp only [Bool.and_eq_true, beq_iff_eq] at hc
    obtain ⟨hr, hsz⟩ := hc
    subst hr
    obtain ⟨a, rfl⟩ := List.length_eq_one_iff.1 (by simpa using hsz)
    have : a = p := by simpa using h
    subst this
    simp [segPos, Desc.Seg.copy, Desc.Seg.lanes, shiftGates_pos, shiftPos_zero, posAll_singleton]
  · cases h

theorem single_mem {s : Seg} {p : Piece} (h : s.single = some p) : p ∈ s.ps.toList := by
  unfold Desc.Seg.single at h
  split at h
  · exact Array.mem_toList_iff.2 (Array.mem_of_getElem? h)
  · cases h

/-- **`joinWith`** names `out`'s positions then `s`'s, wherever `f`'s joins name their two pieces'. -/
theorem joinWith_pos (f : Piece → Piece → Option Piece) (hf : ∀ p q r, f p q = some r → pos r = pos p ++ pos q)
    (out : Array Seg) (s : Seg) : segsPos (Desc.Seg.joinWith f out s) = segsPos out ++ segPos s := by
  unfold Desc.Seg.joinWith
  split
  · rename_i p q hp hq
    split
    · rename_i r hr
      obtain ⟨b, hb, hbp⟩ := Option.bind_eq_some_iff.1 hp
      obtain ⟨ys, rfl⟩ := Array.back?_eq_some_iff.1 hb
      rw [Array.pop_push, segsPos_push, segsPos_push, segPos_one, hf p q r hr, single_pos b p hbp,
        single_pos s q hq, List.append_assoc]
    · exact segsPos_push out s
  · exact segsPos_push out s

/-- A property each of `out`'s segments and `s` have, and every one-piece segment of positive count, each segment
`joinWith` gives has, wherever `f`'s joins are of positive count. -/
theorem joinWith_mem (f : Piece → Piece → Option Piece) (hf : ∀ p q r, f p q = some r → 0 < r.count)
    (Q : Seg → Prop) (hQ : ∀ r : Piece, 0 < r.count → Q (Seg.one r)) (out : Array Seg) (s : Seg)
    (hout : ∀ g ∈ out.toList, Q g) (hs : Q s) : ∀ g ∈ (Desc.Seg.joinWith f out s).toList, Q g := by
  have hpush : ∀ g ∈ (out.push s).toList, Q g := by
    intro g hg
    rw [Array.toList_push, List.mem_append, List.mem_singleton] at hg
    rcases hg with hg | rfl
    · exact hout g hg
    · exact hs
  unfold Desc.Seg.joinWith
  split
  · rename_i p q hp hq
    split
    · rename_i r hr
      obtain ⟨b, hb, -⟩ := Option.bind_eq_some_iff.1 hp
      obtain ⟨ys, rfl⟩ := Array.back?_eq_some_iff.1 hb
      intro g hg
      rw [Array.pop_push, Array.toList_push, List.mem_append, List.mem_singleton] at hg
      rcases hg with hg | rfl
      · exact hout g (by rw [Array.toList_push]; exact List.mem_append_left _ hg)
      · exact hQ r (hf p q r hr)
    · exact hpush
  · exact hpush

theorem foldl_joinWith_pos (f : Piece → Piece → Option Piece)
    (hf : ∀ p q r, f p q = some r → pos r = pos p ++ pos q) :
    ∀ (l : List Seg) (out : Array Seg), segsPos (l.foldl (Desc.Seg.joinWith f) out) = segsPos out ++ l.flatMap segPos
  | [], out => by simp
  | s :: l, out => by
    rw [List.foldl_cons, foldl_joinWith_pos f hf l, joinWith_pos f hf, List.flatMap_cons, List.append_assoc]

theorem foldl_joinWith_mem (f : Piece → Piece → Option Piece) (hf : ∀ p q r, f p q = some r → 0 < r.count)
    (Q : Seg → Prop) (hQ : ∀ r : Piece, 0 < r.count → Q (Seg.one r)) :
    ∀ (l : List Seg) (out : Array Seg), (∀ g ∈ out.toList, Q g) → (∀ g ∈ l, Q g) →
      ∀ g ∈ (l.foldl (Desc.Seg.joinWith f) out).toList, Q g
  | [], out, ho, _ => by simpa using ho
  | s :: l, out, ho, hl => by
    rw [List.foldl_cons]
    exact foldl_joinWith_mem f hf Q hQ l _ (joinWith_mem f hf Q hQ out s ho (hl s (by simp)))
      (fun g hg => hl g (by simp [hg]))

/-- **`coalesce`** names its segments' positions in order. -/
theorem coalesce_pos (segs : Array Seg) : segsPos (Desc.coalesce segs) = segsPos segs := by
  have e : ∀ f : Piece → Piece → Option Piece, (∀ p q r, f p q = some r → pos r = pos p ++ pos q) →
      ∀ x : Array Seg, segsPos (x.foldl (Desc.Seg.joinWith f) #[]) = segsPos x := by
    intro f hf x
    rw [← Array.foldl_toList, foldl_joinWith_pos f hf]
    rfl
  unfold Desc.coalesce
  rw [e _ (fun p q r h => (stack_pos p q r h).1), e _ (fun p q r h => (join_pos p q r h).1)]

/-- **Each segment `coalesce` gives is one of its segments, or one piece of positive count**: a property both have,
it has. -/
theorem coalesce_mem (Q : Seg → Prop) (hQ : ∀ r : Piece, 0 < r.count → Q (Seg.one r)) (segs : Array Seg)
    (h : ∀ g ∈ segs.toList, Q g) : ∀ g ∈ (Desc.coalesce segs).toList, Q g := by
  have e : ∀ f : Piece → Piece → Option Piece, (∀ p q r, f p q = some r → 0 < r.count) →
      ∀ x : Array Seg, (∀ g ∈ x.toList, Q g) → ∀ g ∈ (x.foldl (Desc.Seg.joinWith f) #[]).toList, Q g := by
    intro f hf x hx
    rw [← Array.foldl_toList]
    exact foldl_joinWith_mem f hf Q hQ _ _ (by simp) hx
  exact e _ (fun p q r h => (stack_pos p q r h).2) _ (e _ (fun p q r h => (join_pos p q r h).2) _ h)

theorem ofPart_affine (p : Bool) (i b c : Nat) :
    segsPos (Desc.ofPart (.affine p i b c)) = (List.range c).map fun k => refPos ((Part.affine p i b c).get k) := by
  simp only [Desc.ofPart, segsPos_singleton, segPos_one, pos_one, Flock.Program.Part.get, refPos]
  apply List.map_congr_left
  intro k _
  simp

theorem ofPart_strided (p : Bool) (i : Nat) (b : Int) (inner : Nat) (os is : Int) (c : Nat)
    (hq : PartNonneg (.strided p i b inner os is c)) :
    segsPos (Desc.ofPart (.strided p i b inner os is c)) =
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
    rw [segsPos_singleton, segPos_one, pos_one]
    apply hmap
    intro k _
    simp
  split
  · rename_i h2
    have h2' : os = (inner : Int) * is := by simpa using h2
    rw [segsPos_singleton, segPos_one, pos_one]
    apply hmap
    intro k _
    have hk : (k : Int) = ((k / inner : Nat) : Int) * inner + ((k % inner : Nat) : Int) := by
      have := Nat.div_add_mod k inner
      rw [Nat.mul_comm] at this
      exact_mod_cast this.symm
    rw [h2', hk, Int.add_mul, ← Int.mul_assoc, Int.add_assoc]
  · rw [segsPos_one, posAll_filter_count, posAll_pair, pos_one, pos_eq_map]
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
      simp [ppt, Desc.Piece.rowOff]
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

theorem exStep_spec (acc : Array Piece) (r : Ref) (h : ∀ q ∈ acc.toList, q.rows = 1 ∧ q.lv = []) :
    posAll (exStep acc r) = posAll acc ++ [refPos r] ∧ ∀ q ∈ (exStep acc r).toList, q.rows = 1 ∧ q.lv = [] := by
  have hone : pos { tgt := .ref r.p r.i, base := r.j, stride := 1, len := 1 } = [refPos r] := by
    rw [pos_one]; simp [refPos]
  unfold exStep
  rcases hb : acc.back? with _ | q
  · refine ⟨by rw [posAll_push, hone], ?_⟩
    intro x hx
    rw [Array.toList_push, List.mem_append] at hx
    rcases hx with hx | hx
    · exact h x hx
    · simp at hx; subst hx; exact ⟨rfl, rfl⟩
  · obtain ⟨ys, rfl⟩ := Array.back?_eq_some_iff.1 hb
    have hq : q.rows = 1 ∧ q.lv = [] := h q (by simp)
    have hys : ∀ x ∈ ys.toList, x.rows = 1 ∧ x.lv = [] := fun x hx => h x (by simp [hx])
    dsimp only
    split
    · rename_i hc
      simp only [Bool.and_eq_true, beq_iff_eq] at hc
      obtain ⟨⟨htgt, hst⟩, hat⟩ := hc
      rw [Array.pop_push]
      refine ⟨?_, ?_⟩
      · rw [posAll_push, posAll_push, List.append_assoc]
        congr 1
        rcases q with ⟨t, base, stride, len, rows, rs, lv, st⟩
        obtain ⟨hq1, hq2⟩ := hq
        simp only at hq1 hq2 hst hat htgt
        subst hq1 hq2 hst
        dsimp only
        rw [pos_one, pos_one, List.range_succ, List.map_append]
        simp only [Desc.Piece.at, Int.mul_one] at hat
        rw [tgt_eq_of_beq htgt]
        simp [refPos, hat]
      · intro x hx
        rw [Array.toList_push, List.mem_append] at hx
        rcases hx with hx | hx
        · exact hys x hx
        · simp at hx; subst hx; exact hq
    · refine ⟨by rw [posAll_push, hone], ?_⟩
      intro x hx
      rw [Array.toList_push, List.mem_append] at hx
      rcases hx with hx | hx
      · exact h x hx
      · simp at hx; subst hx; exact ⟨rfl, rfl⟩

theorem exFold_spec : ∀ (l : List Ref) (acc : Array Piece), (∀ q ∈ acc.toList, q.rows = 1 ∧ q.lv = []) →
    posAll (l.foldl exStep acc) = posAll acc ++ l.map refPos
  | [], acc, _ => by simp
  | r :: l, acc, h => by
    obtain ⟨h1, h2⟩ := exStep_spec acc r h
    rw [List.foldl_cons, exFold_spec l _ h2, h1, List.map_cons, List.append_assoc, List.singleton_append]

theorem ofPart_explicit (it : Array Ref) :
    segsPos (Desc.ofPart (.explicit it)) = (List.range it.size).map fun k => refPos ((Part.explicit it).get k) := by
  have e : Desc.ofPart (.explicit it) = (it.toList.foldl exStep #[]).map Seg.one := by
    simp only [Desc.ofPart]
    rw [← Array.foldl_toList]
    rfl
  rw [e, segsPos_one, exFold_spec _ _ (by simp), posAll_empty, List.nil_append]
  apply List.ext_getElem?
  intro k
  rw [List.getElem?_map, List.getElem?_map]
  by_cases hk : k < it.size
  · rw [List.getElem?_range hk, Array.getElem?_toList, Array.getElem?_eq_getElem hk]
    simp [Flock.Program.Part.get, getElem!_pos it k hk]
  · rw [List.getElem?_eq_none (by simp; omega), List.getElem?_eq_none (by simp; omega)]; rfl

private theorem range_mul_map {α : Type} (f : Nat → α) (m : Nat) : ∀ n,
    (List.range (n * m)).map f = (List.range n).flatMap fun i => (List.range m).map fun j => f (i * m + j)
  | 0 => by simp
  | n + 1 => by
    rw [Nat.succ_mul, List.range_add, List.map_append, range_mul_map f m n, List.range_succ, List.flatMap_append,
      List.flatMap_singleton, List.map_map]
    rfl

/-- Reference `t·(g·run) + m·run + o` of a grid of `g` lanes: lane `m`, period `t`, offset `o` in its run. -/
theorem grid_index (g run t m o : Nat) (hm : m < g) (ho : o < run) :
    (t * (g * run) + (m * run + o)) % (g * run) / run = m ∧ (t * (g * run) + (m * run + o)) / (g * run) = t ∧
      (t * (g * run) + (m * run + o)) % run = o := by
  have hr : 0 < run := by omega
  have hlt : m * run + o < g * run := by
    have := Nat.mul_le_mul_right run (show m + 1 ≤ g by omega)
    rw [Nat.succ_mul] at this; omega
  have hgr : 0 < g * run := by omega
  have e : t * (g * run) + (m * run + o) = (o + m * run) + t * (g * run) := by omega
  rw [e, Nat.add_mul_mod_self_right, Nat.add_mul_div_right _ _ hgr, show o + m * run = m * run + o by omega,
    Nat.mod_eq_of_lt hlt, Nat.div_eq_of_lt hlt, show m * run + o = o + m * run by omega, Nat.add_mul_div_right _ _ hr,
    Nat.div_eq_of_lt ho, show t * (g * run) = (t * g) * run by rw [Nat.mul_assoc], Nat.add_mul_mod_self_right,
    Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt ho]
  omega

theorem pos_eq_rows (q : Piece) : pos q = (List.range q.rows).flatMap fun r => pos (q.row r) := by
  simp only [pos_row]; rfl

theorem lanePiece_spec (run P : Nat) (l : Flock.Program.Lane) :
    (Desc.lanePiece run P l).tgt = .ref l.p l.i ∧ (Desc.lanePiece run P l).base = l.b ∧
      (Desc.lanePiece run P l).stride = l.e ∧ (Desc.lanePiece run P l).len = run ∧ (Desc.lanePiece run P l).rows = P ∧
      ∀ r, (Desc.lanePiece run P l).rowOff r = Flock.Program.Lane.leafAt.go l.lv r := by
  rcases h : l.lv with _ | ⟨⟨r, d⟩, _ | ⟨x, xs⟩⟩ <;>
    simp [Desc.lanePiece, h, Desc.Piece.rowOff, Flock.Program.Lane.leafAt.go]

/-- Row `t` of a lane's piece: the lane's run at period `t`. -/
theorem pos_lane_row (run P : Nat) (l : Flock.Program.Lane) (t : Nat) :
    pos ((Desc.lanePiece run P l).row t) = (List.range run).map fun o => (.ref l.p l.i, l.leafAt t o) := by
  obtain ⟨h1, h2, h3, h4, -, h6⟩ := lanePiece_spec run P l
  rw [pos_row, h1, h2, h3, h4, h6]
  apply List.map_congr_left
  intro o _
  simp only [Flock.Program.Lane.leafAt, Prod.mk.injEq, true_and]
  ring

/-- A grid's lanes' runs, period by period, are its references in order. -/
theorem grid_rows (run : Nat) (lanes : Array Flock.Program.Lane) (hq : PartNonneg (.grid run lanes)) :
    ((List.range ((lanes[0]?.map Flock.Program.Lane.periods).getD 0)).flatMap fun t => lanes.toList.flatMap fun l =>
      pos ((Desc.lanePiece run ((lanes[0]?.map Flock.Program.Lane.periods).getD 0) l).row t)) =
      (List.range (Part.grid run lanes).count).map fun k => refPos ((Part.grid run lanes).get k) := by
  have hL : lanes.toList = (List.range lanes.size).map fun m => lanes[m]! := by
    apply List.ext_getElem (by simp)
    intro i h1 h2
    simp [getElem!_pos lanes i (by simpa using h1)]
  change ∀ l ∈ lanes, ∀ t < (lanes[0]?.map Flock.Program.Lane.periods).getD 0, ∀ o < run, 0 ≤ l.leafAt t o at hq
  simp only [Flock.Program.Part.count]
  generalize (lanes[0]?.map Flock.Program.Lane.periods).getD 0 = P at hq ⊢
  rw [Nat.mul_comm (lanes.size * run) P, range_mul_map]
  refine flatMap_congr_mem fun t ht => ?_
  rw [range_mul_map, hL, List.flatMap_map]
  refine flatMap_congr_mem fun m hm => ?_
  rw [List.mem_range] at ht hm
  rw [pos_lane_row]
  refine List.map_congr_left fun o ho => ?_
  rw [List.mem_range] at ho
  obtain ⟨h1, h2, h3⟩ := grid_index lanes.size run t m o hm ho
  have hmem : lanes[m]! ∈ lanes := by rw [getElem!_pos lanes m hm]; exact Array.getElem_mem hm
  simp only [Flock.Program.Part.get, refPos, h1, h2, h3]
  rw [Int.toNat_of_nonneg (hq _ hmem t ht o ho)]

/-- A grid part as segments names its references in order: lane by lane when one lane or one period, else period by
period (`Seg.rowed`). -/
theorem ofPart_grid (run : Nat) (lanes : Array Flock.Program.Lane) (hq : PartNonneg (.grid run lanes)) :
    segsPos (Desc.ofPart (.grid run lanes)) = (List.range (Part.grid run lanes).count).map fun k =>
      refPos ((Part.grid run lanes).get k) := by
  rw [← grid_rows run lanes hq]
  simp only [Desc.ofPart, Desc.ofGrid]
  generalize (lanes[0]?.map Flock.Program.Lane.periods).getD 0 = P
  split
  · rename_i h
    simp only [Bool.or_eq_true, beq_iff_eq, decide_eq_true_eq] at h
    simp only [segsPos, Array.toList_map, List.flatMap_map, segPos_one]
    rw [flatMap_congr_mem fun l _ => by rw [pos_eq_rows, (lanePiece_spec run P l).2.2.2.2.1]]
    rcases h with h | h
    · rcases hl : lanes.toList with _ | ⟨x, _ | ⟨y, l⟩⟩
      · simp
      · simp
      · exfalso
        have := congrArg List.length hl
        simp only [Array.length_toList, List.length_cons] at this
        omega
    · rcases Nat.lt_or_ge P 1 with h0 | h0
      · obtain rfl : P = 0 := by omega
        simp
      · obtain rfl : P = 1 := by omega
        simp
  · rename_i h
    have hP : 1 < P := by simp only [Bool.or_eq_true, beq_iff_eq, decide_eq_true_eq, not_or] at h; omega
    simp only [segsPos_singleton, segPos, Desc.Seg.copy, Desc.Seg.lanes, Bool.true_and, decide_eq_true_eq, hP,
      ↓reduceIte, posAll_map, Array.toList_map, List.flatMap_map]

/-- **A part as pieces** names its references, in order. -/
theorem ofPart_pos (q : Part) (hq : PartNonneg q) :
    segsPos (Desc.ofPart q) = (List.range q.count).map fun k => refPos (q.get k) := by
  cases q with
  | affine p i b c => exact ofPart_affine p i b c
  | strided p i b inner os is c => exact ofPart_strided p i b inner os is c hq
  | explicit it => exact ofPart_explicit it
  | grid run lanes => exact ofPart_grid run lanes hq

/-- A sequence's references as positions, part by part. -/
def partsPos (ps : Array Part) : List (Tgt × Int) :=
  ps.toList.flatMap fun q => (List.range q.count).map fun k => refPos (q.get k)

theorem partsFold_pos : ∀ (l : List Part) (acc : Array Seg), (∀ q ∈ l, PartNonneg q) →
    segsPos (l.foldl (fun acc q => acc ++ Desc.ofPart q) acc) =
      segsPos acc ++ l.flatMap fun q => (List.range q.count).map fun k => refPos (q.get k)
  | [], acc, _ => by simp
  | q :: l, acc, h => by
    rw [List.foldl_cons, partsFold_pos l _ (fun q' hq' => h q' (by simp [hq'])), segsPos_append,
      ofPart_pos q (h q (by simp)), List.flatMap_cons, List.append_assoc]

theorem segOk_one (q : Piece) : SegOk (Seg.one q) := by
  show (posAll (Desc.Seg.copy (Seg.one q) 0)).length = q.count
  rw [copy_zero _ rfl]
  show (posAll #[q]).length = q.count
  rw [posAll_singleton, pos_length]

theorem ofPart_ok (q : Part) : ∀ g ∈ (Desc.ofPart q).toList, SegOk g := by
  have hone : ∀ (X : Array Piece) g, g ∈ (X.map Seg.one).toList → SegOk g := fun X g hg => by
    rw [Array.toList_map, List.mem_map] at hg
    obtain ⟨x, -, rfl⟩ := hg
    exact segOk_one x
  have hone1 : ∀ (x : Piece) g, g ∈ #[Seg.one x].toList → SegOk g := fun x g hg => by
    simp at hg; subst hg; exact segOk_one x
  cases q with
  | affine p i b c => exact hone1 _
  | strided p i b inner os is c =>
    simp only [Desc.ofPart]
    split_ifs
    · intro g hg; simp at hg
    · exact hone1 _
    · exact hone1 _
    · exact hone _
  | explicit it => exact hone _
  | grid run lanes =>
    simp only [Desc.ofPart, Desc.ofGrid]
    split
    · intro g hg
      rw [Array.toList_map, List.mem_map] at hg
      obtain ⟨l, -, rfl⟩ := hg
      exact segOk_one _
    · rename_i h
      have hP : 1 < (lanes[0]?.map Flock.Program.Lane.periods).getD 0 := by
        simp only [Bool.or_eq_true, beq_iff_eq, decide_eq_true_eq, not_or] at h; omega
      intro g hg
      simp at hg
      subst hg
      show (posAll (Desc.Seg.copy _ 0)).length = _
      simp only [Desc.Seg.copy, Desc.Seg.lanes, Bool.true_and, decide_eq_true_eq, hP, ↓reduceIte]
      rw [posAll_map, List.length_flatMap]
      simp only [Array.toList_map, List.map_map, Function.comp_def, pos_lane_row, List.length_map,
        List.length_range]
      simp [Nat.mul_comm]

theorem partsFold_ok : ∀ (l : List Part) (acc : Array Seg), (∀ g ∈ acc.toList, SegOk g) →
    ∀ g ∈ (l.foldl (fun acc q => acc ++ Desc.ofPart q) acc).toList, SegOk g
  | [], _, h => h
  | q :: l, acc, h => by
    rw [List.foldl_cons]
    refine partsFold_ok l _ fun g hg => ?_
    rw [Array.toList_append, List.mem_append] at hg
    rcases hg with hg | hg
    · exact h g hg
    · exact ofPart_ok q g hg

/-- **A sequence as segments** (`Desc.ofSeq`) names its references, in order. -/
theorem ofSeq_pos (s : Seq) (hq : ∀ q ∈ s.parts.toList, PartNonneg q) :
    PSOk (Desc.ofSeq s) ∧ psPos (Desc.ofSeq s) = partsPos s.parts ∧
      (Desc.ofSeq s).count = (partsPos s.parts).length := by
  have hok : ∀ g ∈ (s.parts.foldl (fun acc q => acc ++ Desc.ofPart q) #[]).toList, SegOk g := by
    rw [← Array.foldl_toList]
    exact partsFold_ok _ _ (by simp)
  have hpos : segsPos (s.parts.foldl (fun acc q => acc ++ Desc.ofPart q) #[]) = partsPos s.parts := by
    rw [← Array.foldl_toList, partsFold_pos _ _ hq, segsPos_empty, List.nil_append]; rfl
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

/-- A read of one copy that `Seg.split` makes is one piece of a row at least: a piece of the segment, or a lane's rows
of its copies. -/
theorem split_one {s g : Seg} (hr : s.reps ≠ 0) (hc : ∀ q ∈ s.ps.toList, 0 < q.count) (hg : g ∈ (Seg.split s).toList)
    (h1 : g.reps ≤ 1) : ∃ q, g = Seg.one q ∧ 1 ≤ q.rows := by
  have har : s.alike.reps ≠ 0 := by
    unfold Desc.Seg.alike
    split
    · simp
    · exact hr
  have hap : s.alike.ps = s.ps := by
    unfold Desc.Seg.alike
    split <;> rfl
  revert hg
  unfold Desc.Seg.split
  dsimp only
  split
  · intro hg
    simp only [Array.toList_map, List.mem_map] at hg
    obtain ⟨q, -, rfl⟩ := hg
    refine ⟨_, rfl, ?_⟩
    rw [(rowsFrom_spec q 0 _).2.2.2.1]
    omega
  split
  · intro hg
    simp only [Array.toList_map, List.mem_map] at hg
    obtain ⟨q, hq, rfl⟩ := hg
    refine ⟨q, rfl, ?_⟩
    have := hc q (by rwa [hap] at hq)
    unfold Desc.Piece.count at this
    rcases Nat.eq_zero_or_pos q.rows with h0 | h0
    · rw [h0, Nat.zero_mul] at this; omega
    · exact h0
  · rename_i hne
    intro hg
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hg
    subst hg
    exfalso
    have hal : Seg.alike s = s := by
      unfold Desc.Seg.alike
      split
      · rename_i hc
        exfalso
        apply hne
        unfold Desc.Seg.alike
        rw [if_pos hc]
        rfl
      · rfl
    rw [hal] at h1 hne
    simp only [beq_iff_eq] at hne
    omega

end InferenceOnlyVerifier.Properties.SecurityProofs.ReadsExact
