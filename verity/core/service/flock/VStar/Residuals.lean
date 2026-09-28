import Flock.HmNets

/-!
# Sums of GF(2^128) products over GF(2) forms

`residualForms S W V` is `verity_flock.rec_residuals.residual_forms`, the gadget of `InnerRepCheck`'s unit and of
`RecOpen`'s running sums. `S` is `GfResiduals`' S (`gf2k`'s module doc): each residual is a sum of added terms and of
products `ops[a]·ops[b]`, each operand a sum of terms, a term an element of the port `w` or `v` (`["w", i]`, `["v", i]`) or
the transposed block `["wt", i, u]`, whose bit `j` is bit `u` of `w[i + j]`. Element `i` of a port is its forms `128 i ..`.

The products are not multiplied one by one: for each of Karatsuba's 3^7 = 2,187 leaves (`gf2k.karatsuba`'s recursion: the
low halves', the high halves', then the halves' sums'), the leaf's AND of every product is XORed into one sum, each sum is
committed (`sha512.commit`), and the sums are recombined (`recomb`) and reduced modulo the GHASH polynomial (`reduceG`)
once. Forms are canonical (ascending supports), so a sum's form does not depend on the order of its terms: `recomb` and
`reduceG` give the forms `rec_residuals._recombine`'s leaf sets and `gf2k.reduce_forms` give.
-/

namespace VStar

open Flock.HmNets

/-! ## Polynomials of forms -/

/-- Coefficient-wise XOR of two lists of forms, the shorter padded with zero. -/
def addL : List Lin → List Lin → List Lin
  | [], g => g
  | f :: fs, [] => f :: fs
  | f :: fs, g :: gs => (f ^^^ g) :: addL fs gs

/-- Multiplication by `X^k`. -/
def shiftL (k : Nat) (f : List Lin) : List Lin := List.replicate k Lin.zero ++ f

/-- Multiplication by `X^7 + X^2 + X + 1`, which is `X^128` modulo the GHASH polynomial. -/
def mulS (g : List Lin) : List Lin := addL (addL (addL g (shiftL 1 g)) (shiftL 2 g)) (shiftL 7 g)

/-- One folding step of the reduction: `f mod X^128 + (f div X^128)·(X^7 + X^2 + X + 1)`. -/
def foldG (f : List Lin) : List Lin := addL (f.take 128) (mulS (f.drop 128))

/-- The reduction of a product of at most 255 coefficients to 128: two folds. -/
def reduceG (f : List Lin) : List Lin := foldG (foldG f)

/-! ## `GfResiduals`' S -/

/-- A term: `["w", i]`, `["v", i]` or `["wt", i, u]`. -/
inductive Term where
  | w (i : Nat)
  | v (i : Nat)
  | wt (i u : Nat)
  deriving Inhabited, Repr, BEq

/-- A residual: its products `ops[a]·ops[b]` and its added terms. -/
structure Res where
  mul : List (Nat × Nat)
  add : List Term
  deriving Inhabited, Repr

/-- `GfResiduals`' S: the ports' elements `nw` and `nv`, the operands (each a sum of terms), the residuals. -/
structure GfS where
  nw : Nat
  nv : Nat
  ops : List (List Term)
  res : List Res
  deriving Inhabited, Repr

/-- `gf2k.check_structure`: every `w`/`v` term's element inside its port, a `wt` term's 128 elements inside `w` and its
bit below 128, every product's operands among `ops`. Past a port's end the Python slice comes up short, so the builders
agree only on an in-range S. -/
def checkTerm (S : GfS) : Term → Except String Unit
  | .w i => if i < S.nw then pure () else throw s!"GfResiduals' S: term [\"w\", {i}] is out of range for ports of {S.nw} and {S.nv} elements"
  | .v i => if i < S.nv then pure () else throw s!"GfResiduals' S: term [\"v\", {i}] is out of range for ports of {S.nw} and {S.nv} elements"
  | .wt i u => if i + 128 ≤ S.nw && u < 128 then pure ()
    else throw s!"GfResiduals' S: term [\"wt\", {i}, {u}] is out of range for ports of {S.nw} and {S.nv} elements"

def checkS (S : GfS) : Except String Unit := do
  for o in S.ops do
    for t in o do checkTerm S t
  for r in S.res do
    for t in r.add do checkTerm S t
    for (a, b) in r.mul do
      unless a < S.ops.length && b < S.ops.length do
        throw s!"GfResiduals' S: product [{a}, {b}] of {S.ops.length} operands"

/-! ## The gadget -/

/-- A term's 128 forms. -/
def termForms (W V : Array Lin) : Term → List Lin
  | .w i => (List.range 128).map fun j => W.getD (128 * i + j) Lin.zero
  | .v i => (List.range 128).map fun j => V.getD (128 * i + j) Lin.zero
  | .wt i u => (List.range 128).map fun j => W.getD (128 * (i + j) + u) Lin.zero

def xorL (a b : List Lin) : List Lin := List.zipWith (· ^^^ ·) a b

def xorAll (xs : List (List Lin)) : List Lin := xs.foldl xorL (List.replicate 128 Lin.zero)

/-- Every operand's 128 forms, the sum of its terms' (`residual_forms.op`). -/
def opForms (S : GfS) (W V : Array Lin) : Array (List Lin) :=
  S.ops.toArray.map fun o => xorAll (o.map (termForms W V))

/-- Karatsuba's leaves of two polynomials of `2^n` coefficients, in `gf2k.karatsuba`'s order. -/
def leaves : Nat → List Lin → List Lin → List (Lin × Lin)
  | 0, a, b => [(a.headD Lin.zero, b.headD Lin.zero)]
  | n + 1, a, b =>
    leaves n (a.take (2 ^ n)) (b.take (2 ^ n)) ++ leaves n (a.drop (2 ^ n)) (b.drop (2 ^ n)) ++
      leaves n (addL (a.take (2 ^ n)) (a.drop (2 ^ n))) (addL (b.take (2 ^ n)) (b.drop (2 ^ n)))

/-- Karatsuba's recombination of `3^n` leaf products into the product's coefficients. -/
def recomb : Nat → List Lin → List Lin
  | 0, l => [l.headD Lin.zero]
  | n + 1, l =>
    let p0 := recomb n (l.take (3 ^ n))
    let p2 := recomb n ((l.drop (3 ^ n)).take (3 ^ n))
    let p1 := recomb n (l.drop (2 * 3 ^ n))
    addL (addL p0 (shiftL (2 ^ n) (addL (addL p1 p0) p2))) (shiftL (2 ^ n * 2) p2)

/-- Each product's leaf ANDs, XORed leaf by leaf into `sums`. -/
def leafSums : List (List Lin × List Lin) → List Lin → M (List Lin)
  | [], sums => pure sums
  | p :: ps, sums => do
    let zs ← (leaves 7 p.1 p.2).mapM fun xy => Flock.HmNets.and xy.1 xy.2
    leafSums ps (xorL sums zs)

/-- One residual's 128 forms: its added terms' sum, plus, if it has products, the recombined and reduced committed leaf
sums. -/
def residual (ops : Array (List Lin)) (W V : Array Lin) (r : Res) : M (List Lin) := do
  let acc := xorAll (r.add.map (termForms W V))
  if r.mul.isEmpty then pure acc
  else do
    let sums ← leafSums (r.mul.map fun ab => (ops.getD ab.1 [], ops.getD ab.2 [])) (List.replicate (3 ^ 7) Lin.zero)
    let leaf ← commitW sums
    pure (xorL acc (reduceG (recomb 7 leaf)))

def residualsGo (ops : Array (List Lin)) (W V : Array Lin) : List Res → M (List Lin)
  | [] => pure []
  | r :: rs => do
    let acc ← residual ops W V r
    let bits ← residualsGo ops W V rs
    pure (acc ++ bits)

/-- The residuals' forms, 128 each, in order. -/
def residualForms (S : GfS) (W V : Array Lin) : M (List Lin) := residualsGo (opForms S W V) W V S.res

end VStar
