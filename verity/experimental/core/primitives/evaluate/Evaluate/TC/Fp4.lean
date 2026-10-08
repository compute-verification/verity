import Evaluate.TC.Fp8Atom

/-!
# The block-scaled FP4 tensor-core steps: `verity.ml.circuits.silicon.models.BlockScaledAlignAdd`

A copy of `Security/Definitions/Pouw/PearlC/Fp4.lean`'s step, definition for definition (its `decode`, `scaleValue`,
`blockProduct`, the `groupSum` form and the vectors left out): E2M1 nibbles, the UE4M3 and UE8M0 block scales, the group
terms and the five-way align-add with its truncations, on sm_120's NVFP4 and MXFP4 atoms. `Fp4Atom.step` is `none` exactly
where the Python model raises; `Fp4Atom.sumNegative`, the sign the MXFP4 primitive reads where the sum leaves binary32, is
transcribed from `prims._block_scaled_sum_negative` (Security: none yet). Each definition names the Security definition it
must equal, to be proved where the equalities live, since code never requires `Security/`.
-/

namespace Evaluate.TC.Fp4

open Evaluate.TC.Fp8Atom

/-- An E2M1 code. Must equal `Pouw.PearlC.Fp4.Nib`. -/
abbrev Nib := Fin 16

/-- A word's low nibble as a code; the identity on words below 16. -/
def nib (w : Nat) : Nib := ⟨w % 16, Nat.mod_lt _ (by decide)⟩

/-- Must equal `Pouw.PearlC.Fp4.Nib.sign`. -/
def Nib.sign (c : Nib) : Bool := c.val >>> 3 == 1

/-- Must equal `Pouw.PearlC.Fp4.Nib.expField`. -/
def Nib.expField (c : Nib) : Nat := (c.val >>> 1) % 4

/-- Must equal `Pouw.PearlC.Fp4.Nib.manField`. -/
def Nib.manField (c : Nib) : Nat := c.val % 2

/-- Must equal `Pouw.PearlC.Fp4.Nib.sig`. -/
def Nib.sig (c : Nib) : Nat := if c.expField = 0 then c.manField else 2 + c.manField

/-- Must equal `Pouw.PearlC.Fp4.Nib.expOr1`. -/
def Nib.expOr1 (c : Nib) : Nat := max c.expField 1

/-- `models._e2m1_numerator`. Must equal `Pouw.PearlC.Fp4.numerator`. -/
def numerator (c : Nib) : Int :=
  let n : Int := (c.sig <<< (c.expOr1 - 1) : Nat)
  if c.sign then -n else n

/-- Must equal `Pouw.PearlC.Fp4.ScaleFmt`. -/
inductive ScaleFmt where
  | ue4m3
  | ue8m0
  deriving DecidableEq

/-- Must equal `Pouw.PearlC.Fp4.ScaleFmt.block`. -/
def ScaleFmt.block : ScaleFmt → Nat
  | .ue4m3 => 16
  | .ue8m0 => 32

/-- A word's low byte as a scale byte. -/
def byte (w : Nat) : Fin 256 := ⟨w % 256, Nat.mod_lt _ (by decide)⟩

/-- `models._scale_dyadic`. Must equal `Pouw.PearlC.Fp4.scaleDyadic`. -/
def scaleDyadic : ScaleFmt → Fin 256 → Option (Nat × Int)
  | .ue8m0, s => if s.val = 255 then none else some (8, (s.val : Int) - 130)
  | .ue4m3, s => if s.val % 128 = 127 then none else
      some (Code.sig ⟨s.val % 128, by omega⟩, Code.lsbExp ⟨s.val % 128, by omega⟩)

/-- `models.BlockScaledAlignAdd` with `underflow_zero = "positive"`. Must equal `Pouw.PearlC.Fp4.Fp4Atom`. -/
structure Fp4Atom where
  fmt : ScaleFmt
  k : Nat
  group : Nat
  fracBits : Nat
  accWindow : Nat
  gridFloor : Int

/-- `models.BLACKWELL_SM120_NVF4`. Must equal `Pouw.PearlC.Fp4.sm120Nvf4`. -/
def sm120Nvf4 : Fp4Atom := ⟨.ue4m3, 64, 16, 27, 36, -174⟩

/-- `models.BLACKWELL_SM120_MXF4`. Must equal `Pouw.PearlC.Fp4.sm120Mxf4`. -/
def sm120Mxf4 : Fp4Atom := ⟨.ue8m0, 64, 16, 27, 36, -174⟩

/-- Must equal `Pouw.PearlC.Fp4.Fp4Atom.block`. -/
def Fp4Atom.block (p : Fp4Atom) : Nat := p.fmt.block

/-- Must equal `Pouw.PearlC.Fp4.Fp4Atom.scales`. -/
def Fp4Atom.scales (p : Fp4Atom) : Nat := p.k / p.block

/-- Must equal `Pouw.PearlC.Fp4.GroupTerm`. -/
structure GroupTerm where
  mant : Int
  unit : Int
  live : Bool

/-- Must equal `Pouw.PearlC.Fp4.Fp4Atom.groupTerm`. -/
def Fp4Atom.groupTerm (p : Fp4Atom) (a b : List Nib) (sa sb : List (Fin 256)) (j : Nat) : Option GroupTerm := do
  let ps := (((a.drop (j * p.group)).take p.group).zip ((b.drop (j * p.group)).take p.group)).map fun x =>
    numerator x.1 * numerator x.2
  let gd : Int := ps.sum
  let anyP := ps.any (· != 0)
  let (ma, ea) ← scaleDyadic p.fmt (← sa[j * p.group / p.block]?)
  let (mb, eb) ← scaleDyadic p.fmt (← sb[j * p.group / p.block]?)
  return ⟨gd * ma * mb, -2 + ea + eb, anyP && ma != 0 && mb != 0⟩

/-- `BlockScaledAlignAdd.group_terms`. Must equal `Pouw.PearlC.Fp4.Fp4Atom.groupTerms`. -/
def Fp4Atom.groupTerms (p : Fp4Atom) (a b : List Nib) (sa sb : List (Fin 256)) : Option (List GroupTerm) :=
  if a.length = p.k ∧ b.length = p.k ∧ sa.length = p.scales ∧ sb.length = p.scales then
    (List.range (p.k / p.group)).mapM (p.groupTerm a b sa sb)
  else none

/-- `models._trunc_rz_to_grid`. Must equal `Pouw.PearlC.Fp4.truncRZ`. -/
def truncRZ (m e lsb : Int) : Int :=
  if lsb ≤ e then m * (2 ^ (e - lsb).toNat : Nat)
  else
    let q : Int := (m.natAbs >>> (lsb - e).toNat : Nat)
    if m < 0 then -q else q

/-- `BlockScaledAlignAdd._to_f32_rz`. Must equal `Pouw.PearlC.Fp4.toF32RZ`. -/
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

/-- `BlockScaledAlignAdd.step_scaled`, `none` where it raises. Must equal `Pouw.PearlC.Fp4.Fp4Atom.step`. -/
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

/-- `prims._block_scaled_sum_negative`: whether the align-add's sum of `step` is negative, read where it leaves binary32
(a finite accumulator, valid scales, a participating group or a nonzero accumulator). Security: none yet. -/
def Fp4Atom.sumNegative (p : Fp4Atom) (acc : Nat) (a b : List Nib) (sa sb : List (Fin 256)) : Bool :=
  match p.groupTerms a b sa sb, (fp32Value acc).finite? with
  | some ts, some c =>
    let anchors := ts.filterMap (fun t => if t.live then some (t.unit - p.fracBits) else none) ++
      (if c.mag = 0 then [] else [c.exp - ((p.accWindow - 1 : Nat) : Int)])
    let lsb := max (anchors.foldl max p.gridFloor) p.gridFloor
    let cm : Int := if c.neg then -(c.mag : Int) else c.mag
    decide ((ts.map fun t => truncRZ t.mant t.unit lsb).sum + truncRZ cm (c.exp - 23) lsb < 0)
  | _, _ => false

end Evaluate.TC.Fp4
