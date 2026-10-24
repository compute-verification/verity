import Flock.Net
import Mathlib.Algebra.BigOperators.Group.List.Basic
import Mathlib.Algebra.Group.Action.Defs
import Mathlib.Data.Nat.Bitwise
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring
-- as in `ExecRows`: level3 already realizes the lemmas this file's `simp` calls need (`Std.Legacy.Range.mk.congr_simp`)
import Proofs.Flock.Level3.LookupRows

/-!
# `Sparse`: the bytes `ofRows` writes, and `row` reads them back

`Sparse.ofRows` packs each row as little-endian `u32`s (`le32`, `encRow`), and `Sparse.row` reads row `i` back
(`ofRows_spec`, `ofRows_row`). In a namespace of its own, which no file opens: `ExecRows` exports these names into
`FlockSoundness`, and the template fold (`TmplFoldRef`) reads them without `ExecRows`' imports.
-/

namespace FlockSoundness.SparseRows

open Flock

theorem except_bind_ok {α β ε : Type} {x : Except ε α} {f : α → Except ε β} {v : β}
    (h : (x >>= f) = .ok v) : ∃ a, x = .ok a ∧ f a = .ok v := by
  cases x with
  | error e => cases h
  | ok a => exact ⟨a, rfl, h⟩

/-! ## `Sparse`: the bytes `ofRows` writes, and `row` reads them back -/

/-- The four little-endian bytes `pushLe32` appends. -/
def le32 (x : ℕ) : List UInt8 :=
  [(x % 256).toUInt8, ((x >>> 8) % 256).toUInt8, ((x >>> 16) % 256).toUInt8, ((x >>> 24) % 256).toUInt8]

theorem pushLe32_data (b : ByteArray) (x : ℕ) : (pushLe32 b x).data.toList = b.data.toList ++ le32 x := by
  simp [pushLe32, Std.Legacy.Range.forIn_eq_forIn_range', List.range', le32, ByteArray.push, Id.run]
  rw [show ∀ v : ByteArray, (pure v : Id ByteArray) = v from fun _ => rfl]
  simp

theorem getElem!_of_some {a : Array UInt8} {i : ℕ} {v : UInt8} (h : a[i]? = some v) : a[i]! = v := by
  simp [getElem!_def, h]

theorem testBit_mod256 (y k : ℕ) : (y % 256).testBit k = (decide (k < 8) && y.testBit k) :=
  Nat.testBit_mod_two_pow y 8 k

/-- The four bytes of `x < 2^32`, reassembled. -/
theorem le32_or (x : ℕ) (hx : x < 2 ^ 32) :
    (x % 256 ||| (x >>> 8 % 256) <<< 8 ||| (x >>> 16 % 256) <<< 16 ||| (x >>> 24 % 256) <<< 24) = x := by
  apply Nat.eq_of_testBit_eq
  intro n
  simp only [Nat.testBit_or, Nat.testBit_shiftLeft, testBit_mod256, Nat.testBit_shiftRight]
  have hx' : 32 ≤ n → x.testBit n = false := fun hn =>
    Nat.testBit_eq_false_of_lt (lt_of_lt_of_le hx (Nat.pow_le_pow_right (by norm_num) hn))
  rcases Nat.lt_or_ge n 8 with h8 | h8
  · simp [h8, show ¬ 8 ≤ n by omega, show ¬ 16 ≤ n by omega, show ¬ 24 ≤ n by omega]
  rcases Nat.lt_or_ge n 16 with h16 | h16
  · simp [h8, show ¬ n < 8 by omega, show n - 8 < 8 by omega, show 8 + (n - 8) = n by omega,
      show ¬ 16 ≤ n by omega, show ¬ 24 ≤ n by omega]
  rcases Nat.lt_or_ge n 24 with h24 | h24
  · simp [h16, show ¬ n < 8 by omega, show ¬ n - 8 < 8 by omega, show n - 16 < 8 by omega,
      show 16 + (n - 16) = n by omega, show ¬ 24 ≤ n by omega]
  rcases Nat.lt_or_ge n 32 with h32 | h32
  · simp [h24, show ¬ n < 8 by omega, show ¬ n - 8 < 8 by omega, show ¬ n - 16 < 8 by omega,
      show n - 24 < 8 by omega, show 24 + (n - 24) = n by omega]
  · simp [hx' h32, show ¬ n < 8 by omega, show ¬ n - 8 < 8 by omega, show ¬ n - 16 < 8 by omega,
      show ¬ n - 24 < 8 by omega]

/-- `rd32` reads back what `pushLe32` wrote, below `2^32`. -/
theorem rd32_le32 (b : ByteArray) (o x : ℕ) (hx : x < 2 ^ 32)
    (hb : ∀ k (hk : k < 4), b.data[o + k]? = some ((le32 x).get ⟨k, by simp [le32]; omega⟩)) :
    rd32 b o = x := by
  have h0 := hb 0 (by omega)
  have h1 := hb 1 (by omega)
  have h2 := hb 2 (by omega)
  have h3 := hb 3 (by omega)
  simp only [Nat.add_zero] at h0
  simp only [rd32, ByteArray.get!, getElem!_of_some h0, getElem!_of_some h1, getElem!_of_some h2,
    getElem!_of_some h3]
  simp only [le32, List.get_eq_getElem, List.getElem_cons_zero, List.getElem_cons_succ]
  simpa using le32_or x hx

/-! ## `Sparse.ofRows`, read back -/

/-- The bytes `ofRows` writes for one row. -/
def encRow (r : Array ℕ) : List UInt8 := r.toList.flatMap le32

theorem le32_length (x : ℕ) : (le32 x).length = 4 := rfl

theorem encRow_length (r : Array ℕ) : (encRow r).length = 4 * r.size := by
  simp only [encRow, List.length_flatMap, le32_length, List.map_const', List.sum_replicate, smul_eq_mul,
    Array.length_toList]
  ring

/-- Where each row starts in the data, from byte `base`. -/
def rowStarts : ℕ → List (Array ℕ) → List ℕ
  | _, [] => []
  | base, r :: L => base :: rowStarts (base + 4 * r.size) L

theorem forIn_cols_ok {width : ℕ} {f : ℕ → ByteArray → Except String (ForInStep ByteArray)}
    (hf : ∀ c s v, f c s = .ok v → c < width ∧ v = .yield (pushLe32 s c)) :
    ∀ (l : List ℕ) (d d' : ByteArray), forIn l d f = .ok d' →
      d'.data.toList = d.data.toList ++ l.flatMap le32 ∧ ∀ c ∈ l, c < width := by
  intro l
  induction l with
  | nil => intro d d' h; cases h; simp
  | cons c l ih =>
    intro d d' h
    rw [List.forIn_cons] at h
    rcases hc : f c d with e | v
    · rw [hc] at h; cases h
    · obtain ⟨hlt, rfl⟩ := hf c d v hc
      rw [hc] at h
      obtain ⟨h1, h2⟩ := ih _ _ h
      refine ⟨by rw [h1, pushLe32_data]; simp, ?_⟩
      intro x hx
      rcases List.mem_cons.1 hx with rfl | hx
      · exact hlt
      · exact h2 x hx

theorem forIn_rows_ok {width : ℕ}
    {F : Array ℕ → ByteArray × Array ℕ × Array ℕ → Except String (ForInStep (ByteArray × Array ℕ × Array ℕ))}
    (hF : ∀ r st v, F r st = .ok v → ∃ d', d'.data.toList = st.1.data.toList ++ encRow r ∧
      (∀ c ∈ r.toList, c < width) ∧ v = .yield (d', st.2.1.push st.1.size, st.2.2.push r.size)) :
    ∀ (L : List (Array ℕ)) (st st' : ByteArray × Array ℕ × Array ℕ), forIn L st F = .ok st' →
      st'.1.data.toList = st.1.data.toList ++ L.flatMap encRow ∧
      st'.2.1.toList = st.2.1.toList ++ rowStarts st.1.size L ∧
      st'.2.2.toList = st.2.2.toList ++ L.map Array.size ∧
      ∀ r ∈ L, ∀ c ∈ r.toList, c < width := by
  intro L
  induction L with
  | nil => intro st st' h; cases h; simp [rowStarts]
  | cons r L ih =>
    intro st st' h
    rw [List.forIn_cons] at h
    rcases hr : F r st with e | v
    · rw [hr] at h; cases h
    · obtain ⟨d', hd, hw, rfl⟩ := hF r st v hr
      rw [hr] at h
      obtain ⟨h1, h2, h3, h4⟩ := ih _ _ h
      have hsize : d'.size = st.1.size + 4 * r.size := by
        have := congrArg List.length hd
        simp only [List.length_append, Array.length_toList, encRow_length] at this
        exact this
      refine ⟨by rw [h1, hd]; simp, ?_, by rw [h3]; simp, ?_⟩
      · rw [h2]; simp [rowStarts, hsize]
      · intro x hx c hc
        rcases List.mem_cons.1 hx with rfl | hx
        · exact hw c hc
        · exact h4 x hx c hc

/-- **What `Sparse.ofRows` built**: the rows' `le32` bytes in order, each row's start and length, and
every column below the width. -/
theorem ofRows_spec {rows : Array (Array ℕ)} {width : ℕ} {m : Sparse} (h : Sparse.ofRows rows width = .ok m) :
    m.data.data.toList = rows.toList.flatMap encRow ∧ m.start.toList = rowStarts 0 rows.toList ∧
      m.len.toList = rows.toList.map Array.size ∧ ∀ r ∈ rows.toList, ∀ c ∈ r.toList, c < width := by
  unfold Sparse.ofRows at h
  simp only at h
  obtain ⟨st, hst, h⟩ := except_bind_ok h
  cases h
  rw [← Array.forIn_toList] at hst
  have := forIn_rows_ok (width := width) (fun r st v hv => by
    obtain ⟨d, hd, hv⟩ := except_bind_ok hv
    cases hv
    rw [← Array.forIn_toList] at hd
    obtain ⟨h1, h2⟩ := forIn_cols_ok (width := width) (fun c s v hv => by
      by_cases hc : c ≥ width
      · rw [ite_eq_left_of_eq_true _ _ (eq_true hc)] at hv; cases hv
      · rw [ite_eq_right_of_eq_false _ _ (eq_false hc)] at hv
        cases hv
        exact ⟨by omega, rfl⟩) _ _ _ hd
    exact ⟨d, by rw [h1]; rfl, h2, rfl⟩) _ _ _ hst
  obtain ⟨h1, h2, h3, h4⟩ := this
  exact ⟨by simpa using h1, by simpa using h2, by simpa using h3, h4⟩

theorem rowStarts_getElem? : ∀ (L : List (Array ℕ)) (base i : ℕ), i < L.length →
    (rowStarts base L)[i]? = some (base + 4 * ((L.take i).map Array.size).sum)
  | [], _, _, h => absurd h (by simp)
  | r :: L, base, 0, _ => by simp [rowStarts]
  | r :: L, base, i + 1, h => by
      simp only [rowStarts, List.getElem?_cons_succ]
      rw [rowStarts_getElem? L _ i (by simpa using h)]
      simp only [List.take_succ_cons, List.map_cons, List.sum_cons]
      congr 1
      ring

theorem rowStarts_length : ∀ (L : List (Array ℕ)) (base : ℕ), (rowStarts base L).length = L.length
  | [], _ => rfl
  | r :: L, base => by simp [rowStarts, rowStarts_length L]

theorem flatMap_le32_getElem? : ∀ (l : List ℕ) (t k : ℕ), t < l.length → k < 4 →
    (l.flatMap le32)[4 * t + k]? = (le32 (l.getD t 0))[k]?
  | [], _, _, h, _ => absurd h (by simp)
  | x :: l, 0, k, _, hk => by
      simp only [List.flatMap_cons, Nat.mul_zero, Nat.zero_add, List.getD_cons_zero]
      rw [List.getElem?_append_left (by rw [le32_length]; exact hk)]
  | x :: l, t + 1, k, ht, hk => by
      simp only [List.flatMap_cons, List.getD_cons_succ]
      rw [List.getElem?_append_right (by rw [le32_length]; omega), le32_length,
        show 4 * (t + 1) + k - 4 = 4 * t + k by omega]
      exact flatMap_le32_getElem? l t k (by simpa using ht) hk

theorem flatMap_encRow_length (L : List (Array ℕ)) :
    (L.flatMap encRow).length = 4 * (L.map Array.size).sum := by
  induction L with
  | nil => rfl
  | cons r L ih => simp [List.flatMap_cons, encRow_length, ih]; ring

/-- **`Sparse.ofRows` reads back its rows**: row `i` of what it built is `rows[i]` (empty past the
last row), when every column is below `2^32`. -/
theorem ofRows_row {rows : Array (Array ℕ)} {width : ℕ} {m : Sparse} (h : Sparse.ofRows rows width = .ok m)
    (hw : width ≤ 2 ^ 32) (i : ℕ) : m.row i = rows.getD i #[] := by
  obtain ⟨hd, hs, hl, hc⟩ := ofRows_spec h
  have hlen : m.len.getD i 0 = (rows.getD i #[]).size := by
    simp only [Array.getD_eq_getD_getElem?, ← Array.getElem?_toList, hl, List.getElem?_map]
    cases rows.toList[i]? <;> rfl
  by_cases hi : i < rows.size
  · have hri : rows.getD i #[] = rows[i] := by simp [Array.getD_eq_getD_getElem?, hi]
    have hst : m.start.getD i 0 = 4 * ((rows.toList.take i).map Array.size).sum := by
      simp only [Array.getD_eq_getD_getElem?, ← Array.getElem?_toList, hs]
      rw [rowStarts_getElem? _ 0 i (by simpa using hi)]
      simp
    apply Array.ext
    · simp [Sparse.row, hlen]
    · intro t h1 h2
      simp only [Sparse.row, Array.getElem_map, Array.getElem_range]
      have ht : t < rows[i].size := by rw [hri] at h2; exact h2
      simp only [hri]
      have hlt : rows[i][t] < 2 ^ 32 :=
        lt_of_lt_of_le (hc rows[i] (Array.getElem_mem_toList hi) _ (Array.getElem_mem_toList ht)) hw
      refine rd32_le32 _ _ _ hlt fun k hk => ?_
      rw [hst]
      have hsplit : rows.toList.flatMap encRow = (rows.toList.take i).flatMap encRow ++
          (encRow rows[i] ++ (rows.toList.drop (i + 1)).flatMap encRow) := by
        conv_lhs => rw [← List.take_append_drop i rows.toList]
        rw [List.flatMap_append, List.drop_eq_getElem_cons (by simpa using hi), List.flatMap_cons]
        simp
      rw [← Array.getElem?_toList, hd, hsplit, List.getElem?_append_right (by rw [flatMap_encRow_length]; omega),
        flatMap_encRow_length, show 4 * ((rows.toList.take i).map Array.size).sum + 4 * t + k -
          4 * ((rows.toList.take i).map Array.size).sum = 4 * t + k by omega,
        List.getElem?_append_left (by rw [encRow_length]; omega), encRow,
        flatMap_le32_getElem? _ t k (by simpa using ht) hk]
      simp [ht]
  · have hri : rows.getD i #[] = #[] := by simp [Array.getD_eq_getD_getElem?, hi]
    rw [hri] at hlen ⊢
    simp [Sparse.row, hlen]

end FlockSoundness.SparseRows
