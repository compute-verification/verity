/-
The lincheck fold of two coin vectors in one pass over a sparse matrix (`Sparse.foldT2`), equal to `Sparse.foldT` of each
(`foldT2_eq`): a table's two reps fold the same matrices at two points, and the pass reads each row's entries once.
-/
module

public import Flock.Comp

@[expose] public section

namespace Flock

private theorem eq_zero_of_isZero {a : F128} (h : a.isZero = true) : a = 0 := by
  cases a with
  | mk lo hi =>
    simp only [F128.isZero, Bool.and_eq_true, beq_iff_eq] at h
    obtain ⟨h1, h2⟩ := h
    subst h1; subst h2; rfl

/-- `g` of a modified array is the array of `g`s modified, where `g ∘ f = f' ∘ g`. -/
private theorem map_modify {α β : Type} (g : α → β) (f : α → α) (f' : β → β) (hf : ∀ v, g (f v) = f' (g v))
    (xs : Array α) (i : Nat) : (xs.modify i f).map g = (xs.map g).modify i f' := by
  apply Array.ext
  · simp
  · intro j h1 h2
    simp only [Array.getElem_map, Array.getElem_modify]
    split
    · rw [hf]
    · rfl

private theorem modify_self {α : Type} (f : α → α) (hf : ∀ v, f v = v) (xs : Array α) (i : Nat) : xs.modify i f = xs := by
  apply Array.ext
  · simp
  · intro j h1 h2
    simp only [Array.getElem_modify]
    split
    · rw [hf]
    · rfl

namespace Sparse

/-- Row `i`'s entries, each adding `ei` to its column: what `foldRows` does at a row. -/
def rowAdd (m : Sparse) (i : Nat) (ei : F128) (acc : Array F128) : Array F128 :=
  (m.row i).foldl (fun a c => a.modify c (· + ei)) acc

private theorem rowAdd_zero (m : Sparse) (i : Nat) (acc : Array F128) : m.rowAdd i 0 acc = acc := by
  unfold rowAdd
  rw [← Array.foldl_toList]
  induction (m.row i).toList generalizing acc with
  | nil => rfl
  | cons c cs ih => rw [List.foldl_cons, modify_self _ F128.add_zero, ih]

theorem foldRows_step (m : Sparse) (e : Array F128) (fuel i : Nat) (acc : Array F128) :
    m.foldRows e (fuel + 1) i acc = m.foldRows e fuel (i + 1) (m.rowAdd i e[i]! acc) := by
  rw [foldRows]
  by_cases h : (e[i]!).isZero = true
  · have hz := eq_zero_of_isZero h
    simp only [h, ↓reduceIte]
    rw [hz, rowAdd_zero]
  · simp only [h, Bool.false_eq_true, ↓reduceIte, foldRow_eq_row, rowAdd]

/-- A column's sums for two coin vectors, the first's words then the second's. -/
structure Cell2 where
  lo0 : UInt64
  hi0 : UInt64
  lo1 : UInt64
  hi1 : UInt64
  deriving Inhabited

namespace Cell2

/-- The first vector's sum. -/
@[inline] def fst (v : Cell2) : F128 := ⟨v.lo0, v.hi0⟩

/-- The second vector's sum. -/
@[inline] def snd (v : Cell2) : F128 := ⟨v.lo1, v.hi1⟩

/-- `v` with `⟨a, b⟩` added to its first sum and `⟨c, d⟩` to its second, as a call that owns `v`: compiled, the sums are
written into `v`'s own cell when nothing else holds it, and `Array.modify` hands the call a cell it has taken out of the
array (`addCell`). -/
@[noinline] def add (a b c d : UInt64) (v : Cell2) : Cell2 := ⟨v.lo0 ^^^ a, v.hi0 ^^^ b, v.lo1 ^^^ c, v.hi1 ^^^ d⟩

theorem fst_add (x y : F128) (v : Cell2) : (add x.lo x.hi y.lo y.hi v).fst = v.fst + x := rfl

theorem snd_add (x y : F128) (v : Cell2) : (add x.lo x.hi y.lo y.hi v).snd = v.snd + y := rfl

end Cell2

/-- Row `i`'s entries, each adding `x` to its column's first sum and `y` to its second. -/
@[inline] def rowAdd2 (m : Sparse) (i : Nat) (x y : F128) (acc : Array Cell2) : Array Cell2 :=
  let a := x.lo
  let b := x.hi
  let c := y.lo
  let d := y.hi
  m.rowFold i (fun acc col => acc.modify col (Cell2.add a b c d)) acc

theorem rowAdd2_fst (m : Sparse) (i : Nat) (x y : F128) (acc : Array Cell2) :
    (m.rowAdd2 i x y acc).map Cell2.fst = m.rowAdd i x (acc.map Cell2.fst) := by
  unfold rowAdd2 rowAdd
  rw [rowFold_eq, ← Array.foldl_toList, ← Array.foldl_toList]
  induction (m.row i).toList generalizing acc with
  | nil => rfl
  | cons c cs ih => rw [List.foldl_cons, List.foldl_cons, ih, map_modify Cell2.fst _ (· + x) (Cell2.fst_add x y)]

theorem rowAdd2_snd (m : Sparse) (i : Nat) (x y : F128) (acc : Array Cell2) :
    (m.rowAdd2 i x y acc).map Cell2.snd = m.rowAdd i y (acc.map Cell2.snd) := by
  unfold rowAdd2 rowAdd
  rw [rowFold_eq, ← Array.foldl_toList, ← Array.foldl_toList]
  induction (m.row i).toList generalizing acc with
  | nil => rfl
  | cons c cs ih => rw [List.foldl_cons, List.foldl_cons, ih, map_modify Cell2.snd _ (· + y) (Cell2.snd_add x y)]

/-- Rows `i, …, i + fuel − 1` for two coin vectors in one pass over the rows. -/
def foldRows2 (m : Sparse) (e0 e1 : Array F128) : Nat → Nat → Array Cell2 → Array Cell2
  | 0, _, acc => acc
  | fuel + 1, i, acc => foldRows2 m e0 e1 fuel (i + 1) (m.rowAdd2 i e0[i]! e1[i]! acc)

theorem foldRows2_fst (m : Sparse) (e0 e1 : Array F128) (fuel i : Nat) (acc : Array Cell2) :
    (m.foldRows2 e0 e1 fuel i acc).map Cell2.fst = m.foldRows e0 fuel i (acc.map Cell2.fst) := by
  induction fuel generalizing i acc with
  | zero => rfl
  | succ k ih => rw [foldRows2, ih, foldRows_step, rowAdd2_fst]

theorem foldRows2_snd (m : Sparse) (e0 e1 : Array F128) (fuel i : Nat) (acc : Array Cell2) :
    (m.foldRows2 e0 e1 fuel i acc).map Cell2.snd = m.foldRows e1 fuel i (acc.map Cell2.snd) := by
  induction fuel generalizing i acc with
  | zero => rfl
  | succ k ih => rw [foldRows2, ih, foldRows_step, rowAdd2_snd]

/-- One pass over the matrix for two coin vectors: column `c`'s cell holds `(Xᵀ e₀)[c]` (`foldC2_fst`) and, where the
vectors have one length, `(Xᵀ e₁)[c]` (`foldC2_snd`). -/
def foldC2 (m : Sparse) (e0 e1 : Array F128) : Array Cell2 :=
  m.foldRows2 e0 e1 (min m.rows e0.size) 0 (Array.replicate m.width ⟨0, 0, 0, 0⟩)

theorem foldC2_fst (m : Sparse) (e0 e1 : Array F128) : (m.foldC2 e0 e1).map Cell2.fst = m.foldT e0 := by
  rw [foldC2, foldRows2_fst, Array.map_replicate, foldT]
  rfl

theorem foldC2_snd (m : Sparse) {e0 e1 : Array F128} (h : e0.size = e1.size) :
    (m.foldC2 e0 e1).map Cell2.snd = m.foldT e1 := by
  rw [foldC2, foldRows2_snd, Array.map_replicate, foldT, ← h]
  rfl

/-- `(Xᵀ e₀, Xᵀ e₁)` (`foldT2_eq`): one pass over the matrix where the vectors have one length (`foldC2`), its cells
split into the two vectors' arrays; two walks otherwise. -/
def foldT2 (m : Sparse) (e0 e1 : Array F128) : Array F128 × Array F128 :=
  if e0.size = e1.size then
    let acc := m.foldC2 e0 e1
    (acc.map Cell2.fst, acc.map Cell2.snd)
  else (m.foldT e0, m.foldT e1)

theorem foldT2_eq (m : Sparse) (e0 e1 : Array F128) : m.foldT2 e0 e1 = (m.foldT e0, m.foldT e1) := by
  unfold foldT2
  split
  · rename_i h
    simp only [foldC2_fst, foldC2_snd m h]
  · rfl

end Sparse
end Flock

