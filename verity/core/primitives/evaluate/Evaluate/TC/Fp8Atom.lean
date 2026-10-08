/-!
# The FP8 tensor-core steps: `verity.ml.circuits.silicon.total_fp8`

A copy of Security's FP8 step, definition for definition: the E4M3 codec of `Security/Definitions/Pouw/Fp8Atom/E4M3.lean`
(its `decode` to `ℚ` left out), the grouped-sum step of `Fp8Atom/Atom.lean` (`tc_dot_total_e4m3`, its `Term.value` left out)
and sm_120's pipeline of `PearlC/Sm120.lean`. Each definition names the Security definition it must equal, to be proved where
the equalities live, since code never requires `Security/`.

The E5M2 step (`total_fp8.tc_dot_total_e5m2`) is transcribed from the Python on the same terms, values and groups: its
operand classes and product are `total_fp8.e5m2_class`, `e5m2_product_total` and `term.E5M2.product`. Security: none yet.

An operand is a byte `Code`; the accumulator a word below `2^32`.
-/

namespace Evaluate.TC.Fp8Atom

/-- An E4M3 code. Must equal `Pouw.Fp8Atom.Code`. -/
abbrev Code := Fin 256

/-- A word's low byte as a code; the identity on words below 256. -/
def code (w : Nat) : Code := ⟨w % 256, Nat.mod_lt _ (by decide)⟩

/-- Must equal `Pouw.Fp8Atom.Code.sign`. -/
def Code.sign (c : Code) : Bool := c.val >>> 7 == 1

/-- Must equal `Pouw.Fp8Atom.Code.expField`. -/
def Code.expField (c : Code) : Nat := (c.val >>> 3) % 16

/-- Must equal `Pouw.Fp8Atom.Code.manField`. -/
def Code.manField (c : Code) : Nat := c.val % 8

/-- `0x7F` or `0xFF`. Must equal `Pouw.Fp8Atom.Code.isNaN`. -/
def Code.isNaN (c : Code) : Bool := c.val % 128 == 127

/-- Must equal `Pouw.Fp8Atom.Code.sig`. -/
def Code.sig (c : Code) : Nat := if c.expField = 0 then c.manField else 8 + c.manField

/-- Must equal `Pouw.Fp8Atom.Code.expOr1`. -/
def Code.expOr1 (c : Code) : Nat := max c.expField 1

/-- Must equal `Pouw.Fp8Atom.Code.lsbExp`. -/
def Code.lsbExp (c : Code) : Int := (c.expOr1 : Int) - 10

/-- `term.Term`. Must equal `Pouw.Fp8Atom.Term`. -/
structure Term where
  neg : Bool
  exp : Int
  mag : Nat

/-- Must equal `Pouw.Fp8Atom.Term.zero`. -/
def Term.zero : Term := ⟨false, 0, 0⟩

/-- `term.e4m3_product` on non-NaN codes. Must equal `Pouw.Fp8Atom.product`. -/
def product (a b : Code) : Term :=
  let mag := (a.sig * b.sig) <<< 17
  if mag = 0 then Term.zero else ⟨a.sign ^^ b.sign, (a.expOr1 : Int) + b.expOr1 - 14, mag⟩

/-- `total.Value`. Must equal `Pouw.Fp8Atom.Value`. -/
inductive Value where
  | nan
  | inf (neg : Bool)
  | fin (t : Term)

/-- Must equal `Pouw.Fp8Atom.Value.isNaN`. -/
def Value.isNaN : Value → Bool
  | .nan => true
  | _ => false

/-- Must equal `Pouw.Fp8Atom.Value.isInf`. -/
def Value.isInf (s : Bool) : Value → Bool
  | .inf n => n == s
  | _ => false

/-- Must equal `Pouw.Fp8Atom.Value.finite?`. -/
def Value.finite? : Value → Option Term
  | .fin t => some t
  | _ => none

/-- `total_fp8.e4m3_product_total`. Must equal `Pouw.Fp8Atom.productTotal`. -/
def productTotal (a b : Code) : Value := if a.isNaN || b.isNaN then .nan else .fin (product a b)

/-- `total.fp32_value`. Must equal `Pouw.Fp8Atom.fp32Value`. -/
def fp32Value (w : Nat) : Value :=
  let e := (w >>> 23) % 256
  let m := w % 2 ^ 23
  let s := (w >>> 31) % 2 == 1
  if e = 255 then (if m = 0 then .inf s else .nan)
  else if e = 0 then (if m = 0 then .fin Term.zero else .fin ⟨s, -126, m⟩)
  else .fin ⟨s, (e : Int) - 127, (1 <<< 23) ||| m⟩

/-- Must equal `Pouw.Fp8Atom.toAdder`. -/
def toAdder (width m : Nat) : Nat := if 24 ≤ width then m <<< (width - 24) else m >>> (24 - width)

/-- Must equal `Pouw.Fp8Atom.fromAdder`. -/
def fromAdder (width m : Nat) : Nat := if 24 ≤ width then m >>> (width - 24) else m <<< (24 - width)

/-- Must equal `Pouw.Fp8Atom.maxExp`. -/
def maxExp (zeroExp : Int) (ts : List Term) : Int :=
  ts.foldl (fun m t => if t.mag ≠ 0 ∧ m < t.exp then t.exp else m) zeroExp

/-- Must equal `Pouw.Fp8Atom.aligned`. -/
def aligned (width : Nat) (E : Int) (t : Term) : Int :=
  if t.mag = 0 then 0
  else
    let a := toAdder width t.mag >>> (E - t.exp).toNat
    if t.neg then -(a : Int) else (a : Int)

/-- Must equal `Pouw.Fp8Atom.alignedSum`. -/
def alignedSum (width : Nat) (E : Int) (ts : List Term) : Int := (ts.map (aligned width E)).sum

/-- Must equal `Pouw.Fp8Atom.bitLength`. -/
def bitLength (n : Nat) : Nat := if n = 0 then 0 else Nat.log2 n + 1

/-- Must equal `Pouw.Fp8Atom.normalize`. -/
def normalize (width : Nat) (E : Int) (total : Int) : Term :=
  if total = 0 then Term.zero
  else
    let neg := decide (total < 0)
    let bl := bitLength total.natAbs
    let e : Int := E + (bl : Int) - (width : Int)
    let mag := if width < bl then total.natAbs >>> (bl - width) else total.natAbs <<< (width - bl)
    let mag := if e < -126 then mag >>> (-126 - e).toNat else mag
    let e := if e < -126 then -126 else e
    let mag := fromAdder width mag
    if mag = 0 then Term.zero else ⟨neg, e, mag⟩

/-- `models.group_sum`. Must equal `Pouw.Fp8Atom.groupSum`. -/
def groupSum (width : Nat) (zeroExp : Int) (ts : List Term) : Term :=
  normalize width (maxExp zeroExp ts) (alignedSum width (maxExp zeroExp ts) ts)

/-- `total.group_step_total`. Must equal `Pouw.Fp8Atom.groupStepTotal`. -/
def groupStepTotal (width : Nat) (zeroExp : Int) (acc : Value) (ps : List Value) : Value :=
  let vs := acc :: ps
  if vs.any Value.isNaN then .nan
  else
    let pos := vs.any (Value.isInf false)
    let neg := vs.any (Value.isInf true)
    if pos && neg then .nan
    else if pos then .inf false
    else if neg then .inf true
    else
      let t := groupSum width zeroExp (vs.filterMap Value.finite?)
      if t.mag ≠ 0 ∧ 127 < t.exp then .inf t.neg else .fin t

/-- `term.pack_fp32`. Must equal `Pouw.Fp8Atom.packTerm`. -/
def packTerm (t : Term) : Nat :=
  if t.mag = 0 then 0
  else if 127 < t.exp then ((if t.neg then 1 else 0) <<< 31) ||| 0x7F800000
  else
    let ef : Int := t.exp + 127 - (if t.mag &&& (1 <<< 23) = 0 then 1 else 0)
    ((if t.neg then 1 else 0) <<< 31) ||| (ef.toNat <<< 23) ||| (t.mag &&& 0x7FFFFF)

/-- `total.pack_value`. Must equal `Pouw.Fp8Atom.packValue`. -/
def packValue : Value → Nat
  | .nan => 0x7FFFFFFF
  | .inf s => ((if s then 1 else 0) <<< 31) ||| 0x7F800000
  | .fin t => packTerm t

/-- `models.GroupSum`. Must equal `Pouw.Fp8Atom.Pipeline`. -/
structure Pipeline where
  groups : List Nat
  width : Nat
  zeroExp : Int

/-- `total_fp8.HOPPER_E4M3_WGMMA_K32`. Must equal `Pouw.Fp8Atom.hopperE4m3K32`. -/
def hopperE4m3K32 : Pipeline := ⟨[32], 14, -139⟩

/-- `models.BLACKWELL_SM120_E4M3_M16N8K32`. Must equal `Pouw.PearlC.sm120E4m3`. -/
def sm120E4m3 : Pipeline := ⟨[32], 26, -133⟩

/-- `models.BLACKWELL_SM120_E5M2_M16N8K32`: `sm120E4m3`'s adder; its products are E5M2's. -/
def sm120E5m2 : Pipeline := ⟨[32], 26, -133⟩

/-- Must equal `Pouw.Fp8Atom.Pipeline.fold`. -/
def Pipeline.fold (p : Pipeline) : List Nat → Value → List (Code × Code) → Value
  | [], v, _ => v
  | size :: gs, v, ab =>
      p.fold gs (groupStepTotal p.width p.zeroExp v ((ab.take size).map fun x => productTotal x.1 x.2)) (ab.drop size)

/-- `total_fp8.tc_dot_total_e4m3`. Must equal `Pouw.Fp8Atom.Pipeline.step`, and on `hopperE4m3K32` and `sm120E4m3`, with
`List.ofFn` of the codes, `Pouw.Fp8Atom.hopperStep` and `Pouw.PearlC.sm120Step`. -/
def Pipeline.step (p : Pipeline) (acc : Nat) (a b : List Code) : Nat :=
  packValue (p.fold p.groups (fp32Value acc) (a.zip b))

/-! ## E5M2 (Security: none yet) -/

/-- `total_fp8.e5m2_class` at NaN: exponent field 31, nonzero mantissa. Security: none yet. -/
def Code.e5m2IsNaN (c : Code) : Bool := (c.val >>> 2) % 32 == 31 && c.val % 4 != 0

/-- `total_fp8.e5m2_class` at the infinities: exponent field 31, mantissa 0. Security: none yet. -/
def Code.e5m2IsInf (c : Code) : Bool := (c.val >>> 2) % 32 == 31 && c.val % 4 == 0

/-- `total_fp8.e5m2_class` at the zeros. Security: none yet. -/
def Code.e5m2IsZero (c : Code) : Bool := c.val % 128 == 0

/-- `term.E5M2.product` on finite codes: the 3-bit significands' product at bits 19..24, exponent
`(e_a or 1) + (e_b or 1) - 30`. Security: none yet. -/
def productE5m2 (a b : Code) : Term :=
  let ea := (a.val >>> 2) % 32
  let eb := (b.val >>> 2) % 32
  let sa := a.val % 4 ||| (if ea ≠ 0 then 4 else 0)
  let sb := b.val % 4 ||| (if eb ≠ 0 then 4 else 0)
  let mag := (sa * sb) <<< 19
  if mag = 0 then Term.zero
  else ⟨a.sign ^^ b.sign, ((max ea 1 : Nat) : Int) + ((max eb 1 : Nat) : Int) - 30, mag⟩

/-- `total_fp8.e5m2_product_total`: NaN for a NaN operand or an infinity times a zero, an infinity of the signs' XOR for an
infinity, else the exact product. Security: none yet. -/
def productTotalE5m2 (a b : Code) : Value :=
  if a.e5m2IsNaN || b.e5m2IsNaN then .nan
  else if a.e5m2IsInf || b.e5m2IsInf then
    if a.e5m2IsZero || b.e5m2IsZero then .nan else .inf (a.sign ^^ b.sign)
  else .fin (productE5m2 a b)

/-- `Pipeline.fold` with E5M2 products. Security: none yet. -/
def Pipeline.foldE5m2 (p : Pipeline) : List Nat → Value → List (Code × Code) → Value
  | [], v, _ => v
  | size :: gs, v, ab =>
      p.foldE5m2 gs (groupStepTotal p.width p.zeroExp v ((ab.take size).map fun x => productTotalE5m2 x.1 x.2))
        (ab.drop size)

/-- `total_fp8.tc_dot_total_e5m2`. Security: none yet. -/
def Pipeline.stepE5m2 (p : Pipeline) (acc : Nat) (a b : List Code) : Nat :=
  packValue (p.foldE5m2 p.groups (fp32Value acc) (a.zip b))

end Evaluate.TC.Fp8Atom
