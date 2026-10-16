import Mathlib.FieldTheory.Finite.Basic
import Mathlib.Data.List.GetD

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

theorem toPoly_append [CommSemiring F] (l₁ l₂ : List F) :
    toPoly (l₁ ++ l₂) = toPoly l₁ + X ^ l₁.length * toPoly l₂ := by
  induction l₁ with
  | nil => simp
  | cons a l ih => simp only [List.cons_append, toPoly_cons, ih, List.length_cons]; ring

@[simp] theorem toPoly_singleton [Semiring F] (a : F) : toPoly [a] = C a := by simp

@[simp] theorem toPoly_replicate_zero [Semiring F] (n : ℕ) : toPoly (List.replicate n (0 : F)) = 0 := by
  induction n with
  | zero => rfl
  | succ n ih => simp [List.replicate_succ, ih]

theorem degree_toPoly_lt [Semiring F] (l : List F) : (toPoly l).degree < l.length := by
  rw [degree_lt_iff_coeff_zero]
  intro m hm
  rw [coeff_toPoly, List.getD_eq_default _ _ hm]

theorem getD_pred_length [Zero F] {l : List F} (h : l ≠ []) : l.getD (l.length - 1) 0 = l.getLast h := by
  rw [List.getLast_eq_getElem, List.getD_eq_getElem]

theorem Norm.nil [Zero F] : Norm ([] : List F) := fun h => absurd rfl h

theorem Norm.toPoly_ne_zero [Semiring F] {l : List F} (hl : Norm l) (h : l ≠ []) : toPoly l ≠ 0 := by
  intro h0
  apply hl h
  rw [← getD_pred_length h, ← coeff_toPoly, h0, coeff_zero]

theorem Norm.natDegree [Semiring F] {l : List F} (hl : Norm l) (h : l ≠ []) :
    (toPoly l).natDegree = l.length - 1 := by
  have : 0 < l.length := List.length_pos_iff.2 h
  refine natDegree_eq_of_le_of_coeff_ne_zero ?_ ?_
  · rw [natDegree_le_iff_coeff_eq_zero]
    intro N hN
    rw [coeff_toPoly, List.getD_eq_default _ _ (by omega)]
  · rw [coeff_toPoly, getD_pred_length h]; exact hl h

theorem Norm.leadingCoeff [Semiring F] {l : List F} (hl : Norm l) (h : l ≠ []) :
    (toPoly l).leadingCoeff = l.getLast h := by
  rw [Polynomial.leadingCoeff, hl.natDegree h, coeff_toPoly, getD_pred_length h]

theorem Norm.eq_nil_iff [Semiring F] {l : List F} (hl : Norm l) : toPoly l = 0 ↔ l = [] :=
  ⟨fun h0 => by_contra fun h => hl.toPoly_ne_zero h h0, fun h => by rw [h, toPoly_nil]⟩

theorem Norm.length_eq [Semiring F] {l : List F} (hl : Norm l) (h : toPoly l ≠ 0) :
    l.length = (toPoly l).natDegree + 1 := by
  have hne : l ≠ [] := fun h' => h (by rw [h', toPoly_nil])
  rw [hl.natDegree hne]
  have : 0 < l.length := List.length_pos_iff.2 hne
  omega

theorem Norm.cons [Zero F] {a : F} {l : List F} (hl : Norm l) (h : l ≠ []) : Norm (a :: l) := by
  intro _
  rw [List.getLast_cons h]
  exact hl h

theorem Norm.singleton [Zero F] {a : F} (ha : a ≠ 0) : Norm [a] := fun _ => ha

end Lists

/-! ## The operations: values, lengths and counts -/

section Specs

variable {F : Type}

theorem trim_spec [CommSemiring F] [DecidableEq F] : ∀ l : List F,
    toPoly (trim l).val = toPoly l ∧ Norm (trim l).val ∧ (trim l).val.length ≤ l.length ∧
      (trim l).ops ≤ l.length
  | [] => by simp [trim, Norm.nil]
  | a :: l => by
    obtain ⟨h1, h2, h3, h4⟩ := trim_spec l
    by_cases ht : (trim l).val = []
    · have hl : toPoly l = 0 := by rw [← h1, ht, toPoly_nil]
      by_cases ha : a = 0
      · simp [trim, ht, ha, hl, Norm.nil, h4]
      · simp [trim, ht, ha, hl, Norm.singleton ha]
        omega
    · simp only [trim, Counted.val_bind, Counted.ops_bind, ht, ite_false, Counted.val_pure, Counted.ops_pure,
        toPoly_cons, h1, List.length_cons, add_zero]
      exact ⟨trivial, h2.cons ht, by omega, by omega⟩

theorem toPoly_trim [CommSemiring F] [DecidableEq F] (l : List F) : toPoly (trim l).val = toPoly l := (trim_spec l).1
theorem norm_trim [CommSemiring F] [DecidableEq F] (l : List F) : Norm (trim l).val := (trim_spec l).2.1
theorem length_trim_le [CommSemiring F] [DecidableEq F] (l : List F) : (trim l).val.length ≤ l.length :=
  (trim_spec l).2.2.1
theorem ops_trim_le [CommSemiring F] [DecidableEq F] (l : List F) : (trim l).ops ≤ l.length := (trim_spec l).2.2.2

theorem add_spec [CommSemiring F] : ∀ p q : List F,
    toPoly (add p q).val = toPoly p + toPoly q ∧ (add p q).val.length = max p.length q.length ∧
      (add p q).ops = min p.length q.length
  | [], q => by simp [add]
  | a :: p, [] => by simp [add]
  | a :: p, b :: q => by
    obtain ⟨h1, h2, h3⟩ := add_spec p q
    simp only [add, Counted.val_bind, Counted.ops_bind, val_op, ops_op, Counted.val_pure, Counted.ops_pure,
      toPoly_cons, h1, List.length_cons, h2, h3, C_add]
    refine ⟨by ring, by omega, by omega⟩

theorem toPoly_add [CommSemiring F] (p q : List F) : toPoly (add p q).val = toPoly p + toPoly q := (add_spec p q).1
theorem length_add [CommSemiring F] (p q : List F) : (add p q).val.length = max p.length q.length := (add_spec p q).2.1
theorem ops_add [CommSemiring F] (p q : List F) : (add p q).ops = min p.length q.length := (add_spec p q).2.2

theorem smul_spec [CommSemiring F] (c : F) : ∀ p : List F,
    (smul c p).val = p.map (c * ·) ∧ toPoly (smul c p).val = C c * toPoly p ∧ (smul c p).ops = p.length
  | [] => by simp [smul]
  | a :: p => by
    obtain ⟨h1, h2, h3⟩ := smul_spec c p
    refine ⟨by simp [smul, h1], ?_, by simp [smul, h3]; omega⟩
    simp only [smul, Counted.val_bind, val_op, Counted.val_pure, toPoly_cons, h2, C_mul]
    ring

theorem mul_spec [CommSemiring F] : ∀ p q : List F,
    toPoly (mul p q).val = toPoly p * toPoly q ∧ (mul p q).val.length ≤ p.length + q.length ∧
      (mul p q).ops ≤ 2 * p.length * q.length
  | [], q => by simp [mul]
  | a :: p, q => by
    obtain ⟨h1, h2, h3⟩ := mul_spec p q
    obtain ⟨s1, s2, s3⟩ := smul_spec a q
    simp only [mul, Counted.val_bind, Counted.ops_bind]
    rw [toPoly_add, length_add, ops_add, s2, toPoly_cons, h1, s3, List.length_cons, List.length_cons]
    have hs : (smul a q).val.length = q.length := by rw [s1, List.length_map]
    rw [hs]
    refine ⟨by simp; ring, by omega, ?_⟩
    have : min q.length ((mul p q).val.length + 1) ≤ q.length := min_le_left _ _
    nlinarith

theorem toPoly_mul [CommSemiring F] (p q : List F) : toPoly (mul p q).val = toPoly p * toPoly q := (mul_spec p q).1
theorem length_mul_le [CommSemiring F] (p q : List F) : (mul p q).val.length ≤ p.length + q.length := (mul_spec p q).2.1
theorem ops_mul_le [CommSemiring F] (p q : List F) : (mul p q).ops ≤ 2 * p.length * q.length := (mul_spec p q).2.2

theorem subMul_spec [CommRing F] (c : F) : ∀ s g : List F, g.length ≤ s.length →
    toPoly (subMul c s g).val = toPoly s - C c * toPoly g ∧ (subMul c s g).val.length = s.length ∧
      (subMul c s g).ops = 2 * g.length
  | [], g, h => by
    have : g = [] := List.eq_nil_of_length_eq_zero (by simpa using h)
    subst this; simp [subMul]
  | a :: s, [], _ => by simp [subMul]
  | a :: s, b :: g, h => by
    obtain ⟨h1, h2, h3⟩ := subMul_spec c s g (by simpa using h)
    simp only [subMul, Counted.val_bind, Counted.ops_bind, val_op, ops_op, Counted.val_pure, Counted.ops_pure,
      toPoly_cons, h1, h2, h3, List.length_cons, C_sub, C_mul]
    refine ⟨by ring, trivial, by omega⟩

theorem divLoop_shape [Field F] (ginv : F) (gl : List F) : ∀ p : List F,
    (divLoop ginv gl p).val.1.length = p.length ∧ (divLoop ginv gl p).val.2.length = gl.length ∧
      (divLoop ginv gl p).ops = p.length * (2 * gl.length + 1)
  | [] => by simp [divLoop]
  | a :: p => by
    obtain ⟨h2, h3, h4⟩ := divLoop_shape ginv gl p
    have hsl : (a :: (divLoop ginv gl p).val.2).dropLast.length = gl.length := by simp [h3]
    obtain ⟨-, m2, m3⟩ := subMul_spec ((a :: (divLoop ginv gl p).val.2).getLast (List.cons_ne_nil _ _) * ginv)
      (a :: (divLoop ginv gl p).val.2).dropLast gl hsl.ge
    simp only [divLoop, Counted.val_bind, Counted.ops_bind, val_op, ops_op, Counted.val_pure, Counted.ops_pure,
      List.length_cons]
    refine ⟨by omega, by omega, by rw [m3, h4]; ring⟩

theorem divLoop_eq [Field F] {ginv t : F} (hginv : t * ginv = 1) (gl : List F) : ∀ p : List F,
    toPoly p = toPoly (divLoop ginv gl p).val.1 * toPoly (gl ++ [t]) + toPoly (divLoop ginv gl p).val.2
  | [] => by simp [divLoop]
  | a :: p => by
    have h1 := divLoop_eq hginv gl p
    have h3 := (divLoop_shape ginv gl p).2.1
    have hsne : (a :: (divLoop ginv gl p).val.2) ≠ [] := List.cons_ne_nil _ _
    have hsl : (a :: (divLoop ginv gl p).val.2).dropLast.length = gl.length := by simp [h3]
    have m1 := (subMul_spec ((a :: (divLoop ginv gl p).val.2).getLast hsne * ginv)
      (a :: (divLoop ginv gl p).val.2).dropLast gl hsl.ge).1
    have eS : toPoly (a :: (divLoop ginv gl p).val.2) = toPoly (a :: (divLoop ginv gl p).val.2).dropLast +
        X ^ gl.length * C ((a :: (divLoop ginv gl p).val.2).getLast hsne) := by
      conv_lhs => rw [← List.dropLast_append_getLast hsne]
      rw [toPoly_append, hsl, toPoly_singleton]
    have ec : C ((a :: (divLoop ginv gl p).val.2).getLast hsne * ginv) * C t =
        C ((a :: (divLoop ginv gl p).val.2).getLast hsne) := by
      rw [← C_mul, mul_assoc, mul_comm ginv, hginv, mul_one]
    have eG : toPoly (gl ++ [t]) = toPoly gl + X ^ gl.length * C t := by rw [toPoly_append, toPoly_singleton]
    simp only [divLoop, Counted.val_bind, val_op, Counted.val_pure]
    rw [toPoly_cons, toPoly_cons, m1, h1, eG]
    rw [toPoly_cons] at eS
    linear_combination eS - X ^ gl.length * ec

theorem divmod_nil [Field F] [DecidableEq F] (p : List F) : divmod p [] = pure ([], p) := by
  simp [divmod]

theorem divmod_spec [Field F] [DecidableEq F] (p g : List F) (hne : g ≠ []) :
    Norm (divmod p g).val.1 ∧ Norm (divmod p g).val.2 ∧
      (divmod p g).val.1.length ≤ p.length ∧ (divmod p g).val.2.length < g.length ∧
      (divmod p g).ops ≤ 2 * p.length * g.length + g.length := by
  have hlen : 0 < g.length := List.length_pos_iff.2 hne
  have hdl : g.dropLast.length = g.length - 1 := List.length_dropLast
  obtain ⟨h2, h3, h4⟩ := divLoop_shape (g.getLast hne)⁻¹ g.dropLast p
  simp only [divmod, hne, dite_false, Counted.val_bind, Counted.ops_bind, val_op, ops_op, Counted.val_pure,
    Counted.ops_pure]
  have t1 := trim_spec (divLoop (g.getLast hne)⁻¹ g.dropLast p).val.1
  have t2 := trim_spec (divLoop (g.getLast hne)⁻¹ g.dropLast p).val.2
  refine ⟨t1.2.1, t2.2.1, by omega, by omega, ?_⟩
  obtain ⟨m, hm⟩ : ∃ m, g.length = m + 1 := ⟨g.length - 1, by omega⟩
  have o1 := t1.2.2.2
  have o2 := t2.2.2.2
  rw [h2] at o1
  rw [h3, hdl] at o2
  rw [h4, hdl, hm, Nat.add_sub_cancel] at *
  have key : 2 * p.length * (m + 1) = p.length * (2 * m + 1) + p.length := by ring
  omega

theorem divmod_eq [Field F] [DecidableEq F] (p g : List F) (hg : Norm g) (hne : g ≠ []) :
    toPoly p = toPoly (divmod p g).val.1 * toPoly g + toPoly (divmod p g).val.2 := by
  have hl := divLoop_eq (ginv := (g.getLast hne)⁻¹) (mul_inv_cancel₀ (hg hne)) g.dropLast p
  rw [List.dropLast_append_getLast hne] at hl
  simp only [divmod, hne, dite_false, Counted.val_bind, val_op, Counted.val_pure, toPoly_trim]
  exact hl

theorem dvd_rem_iff [CommRing F] {p q g r d : F[X]} (e : p = q * g + r) : d ∣ p ∧ d ∣ g ↔ d ∣ g ∧ d ∣ r := by
  have er : r = p - q * g := by rw [e]; ring
  constructor
  · rintro ⟨h1, h2⟩; exact ⟨h2, er ▸ dvd_sub h1 (dvd_mul_of_dvd_right h2 _)⟩
  · rintro ⟨h1, h2⟩; exact ⟨e ▸ dvd_add (dvd_mul_of_dvd_right h1 _) h2, h1⟩

theorem gcdLoop_spec [Field F] [DecidableEq F] : ∀ (fuel : ℕ) (p g : List F), Norm p → Norm g → g.length < fuel →
    Norm (gcdLoop fuel p g).val ∧ ∀ d : F[X], d ∣ toPoly (gcdLoop fuel p g).val ↔ d ∣ toPoly p ∧ d ∣ toPoly g
  | 0, _, _, _, _, h => absurd h (Nat.not_lt_zero _)
  | fuel + 1, p, g, hp, hg, h => by
    by_cases hne : g = []
    · subst hne; simp [gcdLoop, hp]
    · obtain ⟨-, n2, -, l2, -⟩ := divmod_spec p g hne
      obtain ⟨ih1, ih2⟩ := gcdLoop_spec fuel g (divmod p g).val.2 hg n2 (by omega)
      simp only [gcdLoop, hne, ite_false, Counted.val_bind]
      exact ⟨ih1, fun d => by rw [ih2, dvd_rem_iff (divmod_eq p g hg hne)]⟩

theorem length_gcdLoop_le [Field F] [DecidableEq F] : ∀ (fuel : ℕ) (p g : List F),
    (gcdLoop fuel p g).val.length ≤ max p.length g.length
  | 0, p, g => by simp [gcdLoop]
  | fuel + 1, p, g => by
    by_cases hne : g = []
    · simp [gcdLoop, hne]
    · have l2 := (divmod_spec p g hne).2.2.2.1
      have ih := length_gcdLoop_le fuel g (divmod p g).val.2
      simp only [gcdLoop, hne, ite_false, Counted.val_bind]
      omega

theorem ops_gcdLoop_le [Field F] [DecidableEq F] (L : ℕ) : ∀ (fuel : ℕ) (p g : List F), p.length ≤ L →
    g.length ≤ L → (gcdLoop fuel p g).ops ≤ fuel * (2 * L * L + L)
  | 0, p, g, _, _ => by simp [gcdLoop]
  | fuel + 1, p, g, hp, hg => by
    by_cases hne : g = []
    · simp [gcdLoop, hne]
    · obtain ⟨-, -, -, l2, o⟩ := divmod_spec p g hne
      have ih := ops_gcdLoop_le L fuel g (divmod p g).val.2 hg (by omega)
      simp only [gcdLoop, hne, ite_false, Counted.ops_bind]
      have : 2 * p.length * g.length ≤ 2 * L * L := by
        have := Nat.mul_le_mul hp hg
        nlinarith
      nlinarith

theorem gcd_spec [Field F] [DecidableEq F] {p : List F} (hp : Norm p) (g : List F) :
    Norm (gcd p g).val ∧ ∀ d : F[X], d ∣ toPoly (gcd p g).val ↔ d ∣ toPoly p ∧ d ∣ toPoly g := by
  have h := gcdLoop_spec ((trim g).val.length + 1) p (trim g).val hp (norm_trim g) (Nat.lt_succ_self _)
  simp only [gcd, Counted.val_bind]
  rw [toPoly_trim] at h
  exact h

theorem length_gcd_le [Field F] [DecidableEq F] (p g : List F) : (gcd p g).val.length ≤ max p.length g.length := by
  have := length_gcdLoop_le ((trim g).val.length + 1) p (trim g).val
  have := length_trim_le g
  simp only [gcd, Counted.val_bind]
  omega

theorem ops_gcd_le [Field F] [DecidableEq F] {p g : List F} {L : ℕ} (hp : p.length ≤ L) (hg : g.length ≤ L) :
    (gcd p g).ops ≤ L + (L + 1) * (2 * L * L + L) := by
  have h1 := ops_trim_le g
  have h2 := length_trim_le g
  have h3 := ops_gcdLoop_le L ((trim g).val.length + 1) p (trim g).val hp (by omega)
  simp only [gcd, Counted.ops_bind]
  have : ((trim g).val.length + 1) * (2 * L * L + L) ≤ (L + 1) * (2 * L * L + L) :=
    Nat.mul_le_mul_right _ (by omega)
  omega

@[simp] theorem monic_nil [Field F] : monic ([] : List F) = pure [] := by simp [monic]

theorem monic_val [Field F] {p : List F} (hne : p ≠ []) : (monic p).val = p.map ((p.getLast hne)⁻¹ * ·) := by
  simp only [monic, hne, dite_false, Counted.val_bind, val_op]
  exact (smul_spec _ p).1

theorem length_monic [Field F] (p : List F) : (monic p).val.length = p.length := by
  by_cases hne : p = []
  · subst hne; simp
  · rw [monic_val hne, List.length_map]

theorem ops_monic_le [Field F] (p : List F) : (monic p).ops ≤ p.length + 1 := by
  by_cases hne : p = []
  · subst hne; simp
  · simp only [monic, hne, dite_false, Counted.ops_bind, ops_op, (smul_spec _ p).2.2]
    omega

theorem monic_spec [Field F] {p : List F} (hp : Norm p) (hne : p ≠ []) :
    Norm (monic p).val ∧ (toPoly (monic p).val).Monic ∧ ∀ d : F[X], d ∣ toPoly (monic p).val ↔ d ∣ toPoly p := by
  have hc : (p.getLast hne)⁻¹ ≠ 0 := inv_ne_zero (hp hne)
  have hm : (monic p).val ≠ [] := by rw [monic_val hne]; simpa using hne
  have hlast : (monic p).val.getLast hm = 1 := by
    simp only [monic_val hne, List.getLast_map]
    exact inv_mul_cancel₀ (hp hne)
  have hn : Norm (monic p).val := fun h => by rw [hlast]; exact one_ne_zero
  refine ⟨hn, by rw [Monic, hn.leadingCoeff hm, hlast], fun d => ?_⟩
  have e : toPoly (monic p).val = C (p.getLast hne)⁻¹ * toPoly p := by
    simp only [monic, hne, dite_false, Counted.val_bind, val_op]
    exact (smul_spec _ p).2.1
  rw [e]
  exact (isUnit_C.2 (Ne.isUnit hc)).dvd_mul_left

end Specs

end Claims.RootFind
