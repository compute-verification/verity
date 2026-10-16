import Mathlib.FieldTheory.Finite.Basic

/-!
# Counted field arithmetic on coefficient lists

A step count is the number of field operations a computation performs. `Counted α` is a value with that number: `op a`
is the result of one field operation, `pure` costs none and `>>=` adds the counts, so a function written in `Counted`
counts each operation it performs, and its `.val` is the plain computation's result. Every field operation below is one
`op` of its own (`+`, `-`, `*`, a negation, an inverse, a test for zero); building lists and comparing their lengths
are not field operations.

A polynomial is its coefficient list, lowest degree first (`toPoly`), and a list is normalized (`Norm`) when its last
coefficient isn't zero. The operations are the schoolbook ones: `trim` (normalize), `add`, `smul`, `mul`, `divmod`
(long division by a normalized divisor), `gcd` (Euclid) and `monic`.
-/

namespace Claims.RootFind

open Polynomial

/-- A value and the number of field operations that computed it. -/
structure Counted (α : Type) where
  val : α
  ops : ℕ

instance : Monad Counted where
  pure a := ⟨a, 0⟩
  bind x f := ⟨(f x.val).val, x.ops + (f x.val).ops⟩

namespace Counted

@[simp] theorem val_pure {α : Type} (a : α) : (pure a : Counted α).val = a := rfl
@[simp] theorem ops_pure {α : Type} (a : α) : (pure a : Counted α).ops = 0 := rfl
@[simp] theorem val_bind {α β : Type} (x : Counted α) (f : α → Counted β) : (x >>= f).val = (f x.val).val := rfl
@[simp] theorem ops_bind {α β : Type} (x : Counted α) (f : α → Counted β) :
    (x >>= f).ops = x.ops + (f x.val).ops := rfl

end Counted

/-- The result of one field operation. -/
def op {α : Type} (a : α) : Counted α := ⟨a, 1⟩

@[simp] theorem val_op {α : Type} (a : α) : (op a).val = a := rfl
@[simp] theorem ops_op {α : Type} (a : α) : (op a).ops = 1 := rfl

section Ops

variable {F : Type}

/-- The polynomial with coefficient list `l`, lowest degree first. -/
noncomputable def toPoly [Semiring F] : List F → F[X]
  | [] => 0
  | a :: l => C a + X * toPoly l

/-- A polynomial's coefficient list, up to its degree. -/
noncomputable def ofPoly [Semiring F] (p : F[X]) : List F := (List.range (p.natDegree + 1)).map p.coeff

/-- A coefficient list is normalized when its last coefficient isn't zero (the empty list is). -/
def Norm [Zero F] (l : List F) : Prop := ∀ h : l ≠ [], l.getLast h ≠ 0

/-- `l` without its zero coefficients at the top. -/
def trim [Zero F] [DecidableEq F] : List F → Counted (List F)
  | [] => pure []
  | a :: l => do
    let t ← trim l
    if t = [] then do
      let z ← op (decide (a = 0))
      pure (if z then [] else [a])
    else pure (a :: t)

/-- The sum, coefficient by coefficient. -/
def add [Add F] : List F → List F → Counted (List F)
  | [], q => pure q
  | a :: p, [] => pure (a :: p)
  | a :: p, b :: q => do
    let c ← op (a + b)
    let r ← add p q
    pure (c :: r)

/-- `c` times each coefficient. -/
def smul [Mul F] (c : F) : List F → Counted (List F)
  | [] => pure []
  | a :: p => do
    let b ← op (c * a)
    let r ← smul c p
    pure (b :: r)

/-- The schoolbook product: `(a + X·p)·q = a·q + X·(p·q)`. -/
def mul [Zero F] [Add F] [Mul F] : List F → List F → Counted (List F)
  | [], _ => pure []
  | a :: p, q => do
    let s ← smul a q
    let r ← mul p q
    add s (0 :: r)

/-- `s − c·g`, coefficient by coefficient. -/
def subMul [Sub F] [Mul F] (c : F) : List F → List F → Counted (List F)
  | [], _ => pure []
  | a :: s, [] => pure (a :: s)
  | a :: s, b :: g => do
    let d ← op (c * b)
    let e ← op (a - d)
    let r ← subMul c s g
    pure (e :: r)

/-- Long division of `p` by `gl ++ [t]`, given `ginv = t⁻¹`, from `p`'s top coefficient down: the quotient (one
coefficient for each of `p`'s) and the remainder (`gl.length` coefficients). Bringing down `a` makes `a :: r`, whose
top coefficient times `ginv` is the next quotient coefficient `c`; `a :: r − c·g` loses its top. -/
def divLoop [Field F] (ginv : F) (gl : List F) : List F → Counted (List F × List F)
  | [] => pure ([], List.replicate gl.length 0)
  | a :: p => do
    let qr ← divLoop ginv gl p
    let s := a :: qr.2
    let c ← op (s.getLast (List.cons_ne_nil _ _) * ginv)
    let r ← subMul c s.dropLast gl
    pure (c :: qr.1, r)

/-- Quotient and remainder of `p` by `g`, both normalized; `g` is normalized and nonempty for the identity
`p = q·g + r` (`toPoly_divmod`). -/
def divmod [Field F] [DecidableEq F] (p g : List F) : Counted (List F × List F) :=
  if h : g = [] then pure ([], p) else do
    let ginv ← op (g.getLast h)⁻¹
    let qr ← divLoop ginv g.dropLast p
    let q ← trim qr.1
    let r ← trim qr.2
    pure (q, r)

/-- Euclid's algorithm with `fuel` steps. -/
def gcdLoop [Field F] [DecidableEq F] : ℕ → List F → List F → Counted (List F)
  | 0, p, _ => pure p
  | fuel + 1, p, g =>
    if g = [] then pure p else do
      let qr ← divmod p g
      gcdLoop fuel g qr.2

/-- A greatest common divisor of `p` (normalized) and `g`. -/
def gcd [Field F] [DecidableEq F] (p g : List F) : Counted (List F) := do
  let g ← trim g
  gcdLoop (g.length + 1) p g

/-- `p` divided by its top coefficient. -/
def monic [Field F] (p : List F) : Counted (List F) :=
  if h : p = [] then pure [] else do
    let c ← op (p.getLast h)⁻¹
    smul c p

end Ops

/-! ## Coefficient lists -/

section Lists

variable {F : Type}

@[simp] theorem toPoly_nil [Semiring F] : toPoly ([] : List F) = 0 := rfl
@[simp] theorem toPoly_cons [Semiring F] (a : F) (l : List F) : toPoly (a :: l) = C a + X * toPoly l := rfl

theorem coeff_toPoly [Semiring F] : ∀ (l : List F) (i : ℕ), (toPoly l).coeff i = l.getD i 0
  | [], i => by simp
  | a :: l, 0 => by simp
  | a :: l, i + 1 => by simp [coeff_toPoly l i]

@[simp] theorem length_ofPoly [Semiring F] (p : F[X]) : (ofPoly p).length = p.natDegree + 1 := by
  simp [ofPoly]

@[simp] theorem toPoly_ofPoly [Semiring F] (p : F[X]) : toPoly (ofPoly p) = p := by
  ext i
  rw [coeff_toPoly, ofPoly, List.getD_eq_getElem?_getD, List.getElem?_map]
  by_cases hi : i < p.natDegree + 1
  · simp [List.getElem?_range hi]
  · rw [List.getElem?_eq_none (by simpa using hi)]
    exact (coeff_eq_zero_of_natDegree_lt (by omega)).symm

end Lists

end Claims.RootFind
