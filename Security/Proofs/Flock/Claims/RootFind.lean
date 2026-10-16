import Proofs.Flock.Claims.RootFindPoly

/-!
# A deterministic root finder over a finite field of characteristic 2: Berlekamp's trace method

Over `F` with `2^n` elements, `roots n basis f` lists the distinct roots of `f` (a coefficient list, `toPoly`), and
`rootsC` is the same computation with its field-operation count (`roots` is `(rootsC …).val`).

1. `rootProduct`: `g = gcd(f, X^(2^n) − X)`, monic, by `n` squarings modulo `f`. `X^(2^n) − X` is the product of
   `X − a` over every `a : F`, so `g` is the product of `f`'s distinct linear factors: it has `f`'s roots and divides
   `X^(2^n) − X`.
2. `split`: such a `g` splits by `gcd(g, Tr(β·X) mod g)`, where `Tr(z) = Σ_{i<n} z^(2^i)` (`trace`) takes values in
   `{0, 1}` on `F`: the factor's roots are `g`'s roots `a` with `Tr(β·a) = 0`. `trySplit` computes it (`traceLoop`,
   `n` squarings modulo `g`) and keeps it when it is a proper factor, `firstSplit` tries `basis` in order, and `split`
   recurses on the factor and its cofactor until each piece is linear. When `basis` separates `F` (`Separates`: for
   `c ≠ 0` some `β ∈ basis` has `Tr(β·c) ≠ 0`), two distinct roots differ in some `Tr(β·−)`, so some `β` splits every
   piece of degree at least 2.

`roots_spec`: for `toPoly f ≠ 0` the list has no duplicates, and its members are exactly `f`'s roots.
`rootsC_ops_le`: at most `costBound n basis.length f.length` field operations, for every `f`.
-/

namespace Claims.RootFind

open Polynomial

section Algorithm

variable {F : Type} [Field F] [DecidableEq F]

/-- `u^(2^j) mod g`, by `j` squarings modulo `g`. -/
def sqLoop (g : List F) : ℕ → List F → Counted (List F)
  | 0, u => pure u
  | j + 1, u => do
    let s ← mul u u
    let qr ← divmod s g
    sqLoop g j qr.2

/-- `acc + Σ_{i<j} u^(2^i)`, modulo `g`. -/
def traceLoop (g : List F) : ℕ → List F → List F → Counted (List F)
  | 0, acc, _ => pure acc
  | j + 1, acc, u => do
    let acc ← add acc u
    let s ← mul u u
    let qr ← divmod s g
    traceLoop g j acc qr.2

/-- **The root product**: `gcd(f, X^(2^n) − X)`, monic, with `X^(2^n)` reduced modulo `f` (`[]` for `f = 0`). -/
def rootProduct (n : ℕ) (f : List F) : Counted (List F) := do
  let f ← trim f
  if f = [] then pure [] else do
    let x ← divmod [0, 1] f
    let u ← sqLoop f n x.2
    let w ← add u [0, -1]
    let g ← gcd f w
    monic g

/-- `g`'s factor `gcd(g, Tr(β·X) mod g)`, monic, and its cofactor, when the factor is proper. -/
def trySplit (n : ℕ) (g : List F) (β : F) : Counted (Option (List F × List F)) := do
  let u ← divmod [0, β] g
  let s ← traceLoop g n [] u.2
  let h ← gcd g s
  if 1 < h.length ∧ h.length < g.length then do
    let h ← monic h
    let k ← divmod g h
    pure (some (h, k.1))
  else pure none

/-- The first `β` of `basis` that splits `g`, as `trySplit` splits it. -/
def firstSplit (n : ℕ) (g : List F) : List F → Counted (Option (List F × List F))
  | [] => pure none
  | β :: basis => do
    let o ← trySplit n g β
    match o with
    | some hk => pure (some hk)
    | none => firstSplit n g basis

/-- The roots of a monic product `g` of distinct linear factors: none for a constant, `−a` for `a + X`, and otherwise
the roots of the two pieces of the first split, with `fuel` levels of recursion. -/
def split (n : ℕ) (basis : List F) : ℕ → List F → Counted (List F)
  | 0, _ => pure []
  | fuel + 1, g =>
    if g.length ≤ 1 then pure []
    else if g.length = 2 then do
      let r ← op (-g.headD 0)
      pure [r]
    else do
      let o ← firstSplit n g basis
      match o with
      | none => pure []
      | some hk => do
        let a ← split n basis fuel hk.1
        let b ← split n basis fuel hk.2
        pure (a ++ b)

/-- **The counted root finder**: the root product's roots, by splitting. -/
def rootsC (n : ℕ) (basis : List F) (f : List F) : Counted (List F) := do
  let g ← rootProduct n f
  split n basis g.length g

/-- **The root finder**: `rootsC`'s value. -/
def roots (n : ℕ) (basis : List F) (f : List F) : List F := (rootsC n basis f).val

end Algorithm

/-- The trace `Σ_{i<n} z^(2^i)`, from `F` with `2^n` elements to its prime field `{0, 1}`. -/
def trace {F : Type} [CommRing F] (n : ℕ) (z : F) : F := ∑ i ∈ Finset.range n, z ^ 2 ^ i

/-- `basis` separates `F`: for each `c ≠ 0` some `β ∈ basis` has `Tr(β·c) ≠ 0`. -/
def Separates {F : Type} [CommRing F] (n : ℕ) (basis : List F) : Prop :=
  ∀ c : F, c ≠ 0 → ∃ β ∈ basis, trace n (β * c) ≠ 0

/-- The root finder's operation bound, at `n` squarings, `b` basis elements and an input of `d` coefficients. -/
def costBound (n b d : ℕ) : ℕ := 64 * (n + 1) * (b + 1) * (d + 3) ^ 4

/-- **The root finder is correct**: over `F` of characteristic 2 with `2^n` elements and a separating `basis`, for
`f ≠ 0` it lists `f`'s roots, each once. -/
theorem roots_spec {F : Type} [Field F] [Fintype F] [DecidableEq F] [CharP F 2] {n : ℕ}
    (hcard : Fintype.card F = 2 ^ n) {basis : List F} (hsep : Separates n basis) {f : List F}
    (hf : toPoly f ≠ 0) :
    (roots n basis f).Nodup ∧ ∀ a, a ∈ roots n basis f ↔ (toPoly f).IsRoot a := by
  sorry

/-- **The root finder's step count**: at most `costBound n basis.length f.length` field operations, on every input. -/
theorem rootsC_ops_le {F : Type} [Field F] [DecidableEq F] (n : ℕ) (basis f : List F) :
    (rootsC n basis f).ops ≤ costBound n basis.length f.length := by
  sorry

end Claims.RootFind
