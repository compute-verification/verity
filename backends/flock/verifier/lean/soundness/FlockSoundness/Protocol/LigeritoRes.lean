import FlockSoundness.Protocol.LigeritoFolds

/-!
# How Ligerito's weights fold (PROTOCOL.md §13.3, §13.6)

Every term of the final check's weight is a constant times a multilinear extension read at the
point `fixed ++ bits(y)`: the fixed challenges so far, then the bits of the still-open variables.
Folding the open variables' lowest bits by challenges `ch` replaces those bits by `ch`
(`foldLoL_mle_bits`), and the split coordinate's weight `(b ? u : 1)` folds to `cff(c)`
(`foldLoL_split`). These two facts carry the claim's weight from one level to the next.
-/

namespace FlockSoundness

open Game Model Accounting
open scoped ENNReal

section Fold

variable {L : Type} [Field L]

theorem foldLo_eq' (t : L) (u : ℕ → L) (y : ℕ) :
    foldLo t u y = (1 + t) * u (2 * y) + t * u (2 * y + 1) := by
  unfold foldLo; ring

/-- Folding by the challenges `ch` is the `eq(ch, ·)`-weighted sum over the folded bits. -/
theorem foldLoL_eq_sum : ∀ (ch : List L) (u : ℕ → L) (y : ℕ),
    foldLoL ch u y = ∑ x ∈ Finset.range (2 ^ ch.length), eqList ch x * u (x + 2 ^ ch.length * y)
  | [], u, y => by simp [foldLoL, eqList]
  | r :: rs, u, y => by
      rw [foldLoL, foldLoL_eq_sum rs, List.length_cons, sum_range_pairs]
      refine Finset.sum_congr rfl fun x _ => ?_
      rw [foldLo_eq', eqList_cons, eqList_cons]
      have h0 : (2 * x).testBit 0 = false := by simp [Nat.testBit_zero]
      have h1 : (2 * x + 1).testBit 0 = true := by simp [Nat.testBit_zero]
      rw [h0, h1, show 2 * x / 2 = x by omega, show (2 * x + 1) / 2 = x by omega]
      simp only [Bool.false_eq_true, if_false, if_true]
      rw [show 2 * (x + 2 ^ rs.length * y) + 1 = 2 * x + 1 + 2 ^ (rs.length + 1) * y by ring,
        show 2 * (x + 2 ^ rs.length * y) = 2 * x + 2 ^ (rs.length + 1) * y by ring]
      ring

end Fold

/-! ## Points with open bits -/

section Points

variable {K : Type} [Field K]

end Points

/-! ## List products as range products -/

section Lists

variable {K : Type} [Field K]

end Lists

/-! ## The final check's weight at any stage -/

section Res

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]

/-- The challenges on the original variables so far. -/
def origOf (initial : List K) (levels : List (List K)) : List K :=
  initial ++ (levels.map List.tail).flatten

/-- An OOD claim's fixed challenges: those on its table's variables so far. -/
def oodFixed (initial : List K) (levels : List (List K)) (o : OodCtx F) : List K :=
  match o.split with
  | none => origOf initial levels
  | some a => (levels.getD a []) ++ ((levels.drop (a + 1)).map List.tail).flatten

def oodScale (A : Arith F K) (levels : List (List K)) (o : OodCtx F) : K :=
  match o.split with
  | none => cffFrom A levels 0
  | some a => cffFrom A levels (a + 1)

/-- An OOD claim's term of the final check's weight. -/
def oodTerm (A : Arith F K) (initial : List K) (levels : List (List K)) (ext : ℕ) (o : OodCtx F)
    (y : ℕ) : K :=
  A.embed o.β * oodScale A levels o
    * ((List.zip (o.z.map A.embed) (oodFixed initial levels o)).map fun (zt, ft) => 1 + zt + ft).prod
    * ((List.range ext).map fun j =>
        let zj := A.embed (o.z.getD ((oodFixed initial levels o).length + j) 0)
        if y.testBit j then zj else 1 + zj).prod

/-- A consistency claim's fixed challenges: those on its folded columns so far. -/
def consFixed (levels : List (List K)) (c : ConsCtx F) : List K :=
  ((levels.drop c.start).map List.tail).flatten

/-- A consistency claim's term of the final check's weight. -/
def consTerm (A : Arith F K) (levels : List (List K)) (ext : ℕ) (c : ConsCtx F) (y : ℕ) : K :=
  A.embed c.β * cffFrom A levels c.start
    * ((c.positions.zipIdx).map fun (p, j) =>
        A.embed (eqList c.α j)
          * (((consFixed levels c).zipIdx).map fun (fi, i) =>
              1 + fi * (1 + A.embed (A.What i (A.omega p)))).prod
          * ((List.range ext).map fun i =>
              if y.testBit i then A.embed (A.What ((consFixed levels c).length + i) (A.omega p))
              else 1).prod).sum

/-! ### Each term as a multilinear extension read at `fixed ++ bits(y)` -/

/-! ### Appending a level's challenges -/

/-! ### Each term across one level -/

/-! ### One level -/

end Res

end FlockSoundness
