import Definitions.Pouw.PearlC.Fp4

/-! The proof-only part of `Definitions.Pouw.PearlC.Fp4`: the definitions and lemmas that no guarantee
statement or assumption reads. -/

namespace Pouw.PearlC.Fp4

open Pouw.Fp8Atom

/-- A scale byte's exact value; `none` for an invalid byte. -/
def scaleValue (f : ScaleFmt) (s : Fin 256) : Option ℚ := (scaleDyadic f s).map fun d => (d.1 : ℚ) * (2 : ℚ) ^ d.2

/-- `BLACKWELL_SM120_MXF4`: the same adder with UE8M0 scales per 32. -/
def sm120Mxf4 : Fp4Atom := ⟨.ue8m0, 64, 16, 27, 36, -174⟩

/-- The offset from a group's unit weight to the exponent of its `groupSum` term: `accWindow − 1 − fracBits` (8 at
sm_120), so that the `accWindow`-bit adder's grid below that exponent is `2^(unit − fracBits)`. -/
def Fp4Atom.termShift (p : Fp4Atom) : Nat := p.accWindow - 1 - p.fracBits

/-- A group's scaled dot as one `Fp8Atom` term, `|mant|·2^(23 − termShift)` at exponent `unit + termShift`; the zero
term if the dot is zero. -/
def Fp4Atom.asTerm (p : Fp4Atom) (t : GroupTerm) : Term :=
  if t.mant = 0 then Term.zero
  else ⟨decide (t.mant < 0), t.unit + p.termShift, t.mant.natAbs <<< (23 - p.termShift)⟩

/-- The step as one `Fp8Atom.groupStepTotal`: the accumulator and the `k / group` scaled dots in a single group, at width
`accWindow` and exponent floor `gridFloor + accWindow − 1` (36 and −139 at sm_120). It is `step` except where a
participating group's dot cancels, which anchors `step`'s grid but not `maxExp` (`Fp4Checks`); and it is total where
`step` is not. -/
def Fp4Atom.groupSumStep (p : Fp4Atom) (acc : Nat) (a b : List Nib) (sa sb : List (Fin 256)) : Option Nat := do
  let ts ← p.groupTerms a b sa sb
  return packValue (groupStepTotal p.accWindow (p.gridFloor + ((p.accWindow - 1 : Nat) : Int)) (fp32Value acc)
    (ts.map fun t => .fin (p.asTerm t)))

/-- The first `n` nibbles packed little-endian in `A` (code `i` is bits `4i .. 4i + 3`). -/
def nibs (n A : Nat) : List Nib := List.ofFn fun i : Fin n => ⟨(A >>> (4 * i.val)) % 16, Nat.mod_lt _ (by decide)⟩

/-- The first `n` bytes packed little-endian in `S`. -/
def bytes (n S : Nat) : List (Fin 256) := List.ofFn fun i : Fin n => ⟨(S >>> (8 * i.val)) % 256, Nat.mod_lt _ (by decide)⟩

/-- A conformance vector: the accumulator word, the packed codes `a` and `b`, the packed scale bytes `sa` and `sb`, and
the word `verity.ml.tc` computes (`none` where it raises). -/
structure Vec where
  acc : Nat
  a : Nat
  b : Nat
  sa : Nat
  sb : Nat
  d : Option Nat

/-- A step on a vector's operands, read at the atom `p`'s widths. -/
def Vec.run (p : Fp4Atom) (f : Nat → List Nib → List Nib → List (Fin 256) → List (Fin 256) → Option Nat) (v : Vec) :
    Option Nat :=
  f v.acc (nibs p.k v.a) (nibs p.k v.b) (bytes p.scales v.sa) (bytes p.scales v.sb)

/-- A vector holds for the atom `p`. -/
def Vec.holds (p : Fp4Atom) (v : Vec) : Bool := v.run p p.step == v.d

/-- A participating group of the vector's firing has a dot that cancels to zero. -/
def Vec.cancels (p : Fp4Atom) (v : Vec) : Bool :=
  match p.groupTerms (nibs p.k v.a) (nibs p.k v.b) (bytes p.scales v.sa) (bytes p.scales v.sb) with
  | some ts => ts.any fun t => t.live && t.mant == 0
  | none => false

/-- `groupSumStep` writes the vector's word, unless the vector is outside the model or a participating group cancels. -/
def Vec.groupSumAgrees (p : Fp4Atom) (v : Vec) : Bool := v.d.isNone || v.cancels p || v.run p p.groupSumStep == v.d

/-- The vector is inside the model, a participating group cancels, and `groupSumStep`'s word differs. -/
def Vec.groupSumDiffers (p : Fp4Atom) (v : Vec) : Bool := v.d.isSome && v.cancels p && v.run p p.groupSumStep != v.d

end Pouw.PearlC.Fp4
