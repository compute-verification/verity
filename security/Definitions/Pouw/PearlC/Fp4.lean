module

public import Definitions.Pouw.Fp8Atom.Atom
import Mathlib.Tactic.Bound.Init
import Mathlib.Tactic.Finiteness.Attr
import Mathlib.Tactic.SetLike

/-! sm_120's block-scaled FP4 step, transcribed from `verity.ml.tc.models`; it is no `Pipeline`, as a participating group whose dot cancels still anchors the grid, which `groupStepTotal`'s `maxExp` skips. -/

@[expose] public section

namespace Pouw.PearlC.Fp4

open Pouw.Fp8Atom

/-- An E2M1 code: a nibble. -/
abbrev Nib := Fin 16

/-- The sign bit (bit 3). -/
def Nib.sign (c : Nib) : Bool := c.val >>> 3 == 1

/-- The exponent field (bits 2..1). -/
def Nib.expField (c : Nib) : Nat := (c.val >>> 1) % 4

/-- The mantissa field (bit 0). -/
def Nib.manField (c : Nib) : Nat := c.val % 2

/-- The significand, hidden bit included for normals: `m` if the exponent field is 0, else `2 + m`. -/
def Nib.sig (c : Nib) : Nat := if c.expField = 0 then c.manField else 2 + c.manField

/-- The exponent field with subnormals read at 1. -/
def Nib.expOr1 (c : Nib) : Nat := max c.expField 1

/-- The weight of the significand's least significant bit: `2^(max e 1 − 2)`. -/
def Nib.lsbExp (c : Nib) : Int := (c.expOr1 : Int) - 2

/-- The exact value of a code (`decode_e2m1`): every nibble is finite. -/
def decode (c : Nib) : ℚ := (if c.sign then -1 else 1) * (c.sig : ℚ) * (2 : ℚ) ^ c.lsbExp

/-- `_e2m1_numerator`: twice the code's value, `sig · 2^(max e 1 − 1)` with the sign, one of
`±{0, 1, 2, 3, 4, 6, 8, 12}` (`_E2M1_NUMERATOR`). -/
def numerator (c : Nib) : Int :=
  let n : Int := (c.sig <<< (c.expOr1 - 1) : Nat)
  if c.sign then -n else n

/-- Carried at bits 21..24 and unscaled: block scales are not applied. -/
def e2m1Product (a b : Nib) : Term :=
  let mag := (a.sig * b.sig) <<< 21
  if mag = 0 then Term.zero else ⟨a.sign ^^ b.sign, (a.expOr1 : Int) + b.expOr1 - 2, mag⟩

/-- A block-scale format: UE4M3 (NVFP4) or UE8M0 (MXFP4). -/
inductive ScaleFmt where
  | ue4m3
  | ue8m0
  deriving DecidableEq, Repr

/-- Elements per scale byte: 16 for UE4M3, 32 for UE8M0. -/
def ScaleFmt.block : ScaleFmt → Nat
  | .ue4m3 => 16
  | .ue8m0 => 32

/-- A scale byte as sm_120 reads it, `(mant, exp)` with value `mant·2^exp`; `none` for NaN. UE4M3 reads the low 7 bits
(bit 7 is ignored: `0x7F` and `0xFF` are NaN); UE8M0's NaN is `0xFF`. -/
def scaleDyadic : ScaleFmt → Fin 256 → Option (Nat × Int)
  | .ue8m0, s => if s.val = 255 then none else some (8, (s.val : Int) - 130)
  | .ue4m3, s => if s.val % 128 = 127 then none else
      some (Code.sig ⟨s.val % 128, by omega⟩, Code.lsbExp ⟨s.val % 128, by omega⟩)

def scaleValue (f : ScaleFmt) (s : Fin 256) : Option ℚ := (scaleDyadic f s).map fun d => (d.1 : ℚ) * (2 : ℚ) ^ d.2

def blockProduct (f : ScaleFmt) (a b : Nib) (sa sb : Fin 256) : Option Term := do
  let (ma, ea) ← scaleDyadic f sa
  let (mb, eb) ← scaleDyadic f sb
  let t := e2m1Product a b
  let mag := t.mag * ma * mb
  if mag = 0 then some Term.zero else some ⟨t.neg, t.exp + ea + eb, mag⟩

/-- A block-scaled align-add (`models.BlockScaledAlignAdd`, E2M1 elements, `underflow_zero = "positive"`): the scale
format, `k` elements per firing, groups of `group`, `fracBits` bits below the leading participating group's unit weight,
an accumulator window of `accWindow` bits, and the adder's absolute grid floor. -/
structure Fp4Atom where
  fmt : ScaleFmt
  k : Nat
  group : Nat
  fracBits : Nat
  accWindow : Nat
  gridFloor : Int

/-- `BLACKWELL_SM120_NVF4`: UE4M3 scales per 16, `k = 64` in groups of 16, 27 fraction bits, a 36-bit accumulator window,
grid floor −174. -/
def sm120Nvf4 : Fp4Atom := ⟨.ue4m3, 64, 16, 27, 36, -174⟩

def sm120Mxf4 : Fp4Atom := ⟨.ue8m0, 64, 16, 27, 36, -174⟩

/-- Elements per scale byte. -/
def Fp4Atom.block (p : Fp4Atom) : Nat := p.fmt.block

/-- Scale bytes per operand per firing: 4 for NVFP4, 2 for MXFP4. -/
def Fp4Atom.scales (p : Fp4Atom) : Nat := p.k / p.block

/-- One group's scaled dot (`group_terms`'s triple): the value `mant·2^unit`, and whether the group participates in the
grid's anchor. -/
structure GroupTerm where
  mant : Int
  unit : Int
  live : Bool
  deriving Repr

/-- Group `j`'s scaled dot: the exact dot of its numerators (quarters, at unit `−2`) times its blocks' scale mantissas;
it participates if an elementwise product is nonzero (both codes nonzero) and both scale mantissas are nonzero. -/
def Fp4Atom.groupTerm (p : Fp4Atom) (a b : List Nib) (sa sb : List (Fin 256)) (j : Nat) : Option GroupTerm := do
  let ps := (((a.drop (j * p.group)).take p.group).zip ((b.drop (j * p.group)).take p.group)).map fun x =>
    numerator x.1 * numerator x.2
  let gd : Int := ps.sum
  let anyP := ps.any (· != 0)
  let (ma, ea) ← scaleDyadic p.fmt (← sa[j * p.group / p.block]?)
  let (mb, eb) ← scaleDyadic p.fmt (← sb[j * p.group / p.block]?)
  return ⟨gd * ma * mb, -2 + ea + eb, anyP && ma != 0 && mb != 0⟩

/-- `group_terms`: the `k / group` scaled dots of one firing; `none` for operands of the wrong length or an invalid scale
byte. -/
def Fp4Atom.groupTerms (p : Fp4Atom) (a b : List Nib) (sa sb : List (Fin 256)) : Option (List GroupTerm) :=
  if a.length = p.k ∧ b.length = p.k ∧ sa.length = p.scales ∧ sb.length = p.scales then
    (List.range (p.k / p.group)).mapM (p.groupTerm a b sa sb)
  else none

/-- `_trunc_rz_to_grid`: `m·2^e` truncated toward zero onto the grid `2^lsb`, as an integer multiple of `2^lsb`. -/
def truncRZ (m e lsb : Int) : Int :=
  if lsb ≤ e then m * (2 ^ (e - lsb).toNat : Nat)
  else
    let q : Int := (m.natAbs >>> (lsb - e).toNat : Nat)
    if m < 0 then -q else q

/-- `_to_f32_rz`: the nonzero `n·2^lsb` rounded toward zero to a binary32 word, with gradual underflow; `+0` if the
whole magnitude is below the subnormal grid, and `none` beyond binary32. -/
def toF32RZ (n lsb : Int) : Option Nat :=
  let mag := n.natAbs
  let eVal : Int := (bitLength mag : Int) - 1 + lsb
  let grid : Int := if -126 ≤ eVal then eVal - 23 else -149
  let shift : Int := grid - lsb
  let q := if 0 < shift then mag >>> shift.toNat else mag <<< (-shift).toNat
  if q = 0 then some 0
  else
    let e : Int := (bitLength q : Int) - 1 + grid
    if 127 < e then none
    else
      let bits := if -126 ≤ e then ((e + 127).toNat <<< 23) ||| (q &&& 0x7FFFFF) else q
      some (bits ||| (if n < 0 then 0x80000000 else 0))

/-- **One firing** (`step_scaled`, whose `step` takes each operand's codes and scale bytes concatenated): the
accumulator word, `k` code pairs and each operand's `k / block` scale bytes to the FP32 word written, or `none` outside
the model. -/
def Fp4Atom.step (p : Fp4Atom) (acc : Nat) (a b : List Nib) (sa sb : List (Fin 256)) : Option Nat := do
  let ts ← p.groupTerms a b sa sb
  let c ← (fp32Value acc).finite?
  let anchors := ts.filterMap (fun t => if t.live then some (t.unit - p.fracBits) else none) ++
    (if c.mag = 0 then [] else [c.exp - ((p.accWindow - 1 : Nat) : Int)])
  match anchors with
  | [] => some 0
  | x :: xs =>
    let lsb := max (xs.foldl max x) p.gridFloor
    let cm : Int := if c.neg then -(c.mag : Int) else c.mag
    let total := (ts.map fun t => truncRZ t.mant t.unit lsb).sum + truncRZ cm (c.exp - 23) lsb
    if total = 0 then some 0 else toF32RZ total lsb

def Fp4Atom.termShift (p : Fp4Atom) : Nat := p.accWindow - 1 - p.fracBits

def Fp4Atom.asTerm (p : Fp4Atom) (t : GroupTerm) : Term :=
  if t.mant = 0 then Term.zero
  else ⟨decide (t.mant < 0), t.unit + p.termShift, t.mant.natAbs <<< (23 - p.termShift)⟩

/-- It is `step` except where a participating group's dot cancels (`Fp4Checks`), and total where `step` is not. -/
def Fp4Atom.groupSumStep (p : Fp4Atom) (acc : Nat) (a b : List Nib) (sa sb : List (Fin 256)) : Option Nat := do
  let ts ← p.groupTerms a b sa sb
  return packValue (groupStepTotal p.accWindow (p.gridFloor + ((p.accWindow - 1 : Nat) : Int)) (fp32Value acc)
    (ts.map fun t => .fin (p.asTerm t)))

/-- The first `n` nibbles packed little-endian in `A` (code `i` is bits `4i .. 4i + 3`). -/
def nibs (n A : Nat) : List Nib := List.ofFn fun i : Fin n => ⟨(A >>> (4 * i.val)) % 16, Nat.mod_lt _ (by decide)⟩

/-- The first `n` bytes packed little-endian in `S`. -/
def bytes (n S : Nat) : List (Fin 256) := List.ofFn fun i : Fin n => ⟨(S >>> (8 * i.val)) % 256, Nat.mod_lt _ (by decide)⟩

structure Vec where
  acc : Nat
  a : Nat
  b : Nat
  sa : Nat
  sb : Nat
  d : Option Nat

def Vec.run (p : Fp4Atom) (f : Nat → List Nib → List Nib → List (Fin 256) → List (Fin 256) → Option Nat) (v : Vec) :
    Option Nat :=
  f v.acc (nibs p.k v.a) (nibs p.k v.b) (bytes p.scales v.sa) (bytes p.scales v.sb)

def Vec.holds (p : Fp4Atom) (v : Vec) : Bool := v.run p p.step == v.d

def Vec.cancels (p : Fp4Atom) (v : Vec) : Bool :=
  match p.groupTerms (nibs p.k v.a) (nibs p.k v.b) (bytes p.scales v.sa) (bytes p.scales v.sb) with
  | some ts => ts.any fun t => t.live && t.mant == 0
  | none => false

def Vec.groupSumAgrees (p : Fp4Atom) (v : Vec) : Bool := v.d.isNone || v.cancels p || v.run p p.groupSumStep == v.d

def Vec.groupSumDiffers (p : Fp4Atom) (v : Vec) : Bool := v.d.isSome && v.cancels p && v.run p p.groupSumStep != v.d

end Pouw.PearlC.Fp4
