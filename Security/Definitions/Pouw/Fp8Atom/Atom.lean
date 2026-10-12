import Definitions.Pouw.Fp8Atom.E4M3

/-!
# `HOPPER_E4M3_K32`: one Hopper FP8 tensor-core accumulator step, bit-exact (trusted definitions)

The total step of `verity.ml.tc.total_fp8.tc_dot_total_e4m3` on `verity.ml.tc.models.HOPPER_E4M3_K32` (one accumulator
coordinate of `wgmma.mma_async.sync.aligned.m64nNk32.f32.e4m3.e4m3`, SASS `QGMMA.64xNx32.F32.E4M3.E4M3`), transcribed
function by function from `verity.ml.tc` (`term.py`, `models.py`, `total.py`, `total_fp8.py`):

* **Terms** (`Term`, `term.Term`): `(−1)^neg · mag · 2^(exp − 23)`, `exp` the weight of bit 23.
* **Products** (`product`, `term.e4m3_product`): exact, `mag = (sig a · sig b) <<< 17` at
  `exp = (e_a or 1) + (e_b or 1) − 14`; a zero significand gives the unsigned zero. A NaN code gives NaN
  (`productTotal`, `total_fp8.e4m3_product_total`). A product never overflows (`448² < 2^18`).
* **The accumulator** (`fp32Value`, `total.fp32_value` and `term.fp32_term`): NaN, `±inf`, or a finite term; `±0`
  is the unsigned zero.
* **One group** (`groupSum`, `models.group_sum`): every term goes to the `width`-bit adder (`toAdder`: `>>> 10` at
  width 14), is aligned to the group's largest exponent (`maxExp`, at least `zeroExp`) by a truncating right shift
  (`aligned`), the aligned terms are summed exactly (`alignedSum`), and the sum is normalised to `width` bits by
  truncation, denormalised below `2^−126`, and returned to the 24-bit grid (`normalize`).
* **The total group step** (`groupStepTotal`, `total.group_step_total`): NaN if any term is NaN, NaN for infinities of
  both signs, the infinity if one sign, otherwise `groupSum` of the finite terms, which saturates to an infinity of its
  sign above the FP32 range (exponent above 127).
* **The word** (`packValue`, `total.pack_value` and `term.pack_fp32`): `0x7FFFFFFF` for NaN, `±0x7F800000` for the
  infinities, the FP32 encoding otherwise; zero is always `+0`.

`HOPPER_E4M3_K32` is `groups = (32,)`, `width = 14`, `zero_exponent = −139` (`hopperE4m3K32`): the 32 products and the
accumulator are summed in one group. `ADA_E4M3_M16N8K32` (`adaE4m3K32`) is the same adder with two groups of 16.
Words are natural numbers; the step reads the low 32 bits of the accumulator word and writes a word below `2^32`.
-/

namespace Pouw.Fp8Atom

/-- A finite term on the FP32 significand grid: the value `(−1)^neg · mag · 2^(exp − 23)` (`verity.ml.tc.term.Term`;
`exp` is the weight of bit 23). A term with `mag = 0` is zero, whatever `neg` and `exp`. -/
structure Term where
  neg : Bool
  exp : Int
  mag : Nat
  deriving Repr

/-- The unsigned zero (`term.ZERO`). -/
def Term.zero : Term := ⟨false, 0, 0⟩

/-- A term's exact value. -/
def Term.value (t : Term) : ℚ := (if t.neg then -1 else 1) * (t.mag : ℚ) * (2 : ℚ) ^ (t.exp - 23)

/-- `e4m3_product`: the exact product of two non-NaN codes, carried with 25 significand bits, at exponent
`(e_a or 1) + (e_b or 1) − 14`. -/
def product (a b : Code) : Term :=
  let mag := (a.sig * b.sig) <<< 17
  if mag = 0 then Term.zero else ⟨a.sign ^^ b.sign, (a.expOr1 : Int) + b.expOr1 - 14, mag⟩

/-- A value of the total semantics (`total.Value`): NaN, a signed infinity, or a finite term. -/
inductive Value where
  | nan
  | inf (neg : Bool)
  | fin (t : Term)
  deriving Repr

/-- NaN. -/
def Value.isNaN : Value → Bool
  | .nan => true
  | _ => false

/-- An infinity of sign `s`. -/
def Value.isInf (s : Bool) : Value → Bool
  | .inf n => n == s
  | _ => false

/-- The finite term, if any. -/
def Value.finite? : Value → Option Term
  | .fin t => some t
  | _ => none

/-- `e4m3_product_total`: NaN if either code is NaN, else the exact product. -/
def productTotal (a b : Code) : Value := if a.isNaN || b.isNaN then .nan else .fin (product a b)

/-- `fp32_value`: an FP32 word as a value. Exponent field 255 is `±inf` (mantissa 0) or NaN; field 0 is zero or a
subnormal at exponent −126; otherwise the 24-bit significand at exponent `field − 127`. -/
def fp32Value (w : Nat) : Value :=
  let e := (w >>> 23) % 256
  let m := w % 2 ^ 23
  let s := (w >>> 31) % 2 == 1
  if e = 255 then (if m = 0 then .inf s else .nan)
  else if e = 0 then (if m = 0 then .fin Term.zero else .fin ⟨s, -126, m⟩)
  else .fin ⟨s, (e : Int) - 127, (1 <<< 23) ||| m⟩

/-! ## One group -/

/-- A 24-bit FP32-grid magnitude on the `width`-bit adder (`rescale = width − 24`; a truncating shift when negative). -/
def toAdder (width m : Nat) : Nat := if 24 ≤ width then m <<< (width - 24) else m >>> (24 - width)

/-- A `width`-bit adder magnitude back on the 24-bit grid. -/
def fromAdder (width m : Nat) : Nat := if 24 ≤ width then m >>> (width - 24) else m <<< (24 - width)

/-- The group's alignment exponent: the largest exponent of a nonzero term, and at least `zeroExp` (the floor). -/
def maxExp (zeroExp : Int) (ts : List Term) : Int :=
  ts.foldl (fun m t => if t.mag ≠ 0 ∧ m < t.exp then t.exp else m) zeroExp

/-- **The first truncation**: a term on the adder, aligned to the exponent `E` by a truncating right shift, signed.
A zero term contributes `0`. -/
def aligned (width : Nat) (E : Int) (t : Term) : Int :=
  if t.mag = 0 then 0
  else
    let a := toAdder width t.mag >>> (E - t.exp).toNat
    if t.neg then -(a : Int) else (a : Int)

/-- The exact sum of the group's aligned terms. -/
def alignedSum (width : Nat) (E : Int) (ts : List Term) : Int := (ts.map (aligned width E)).sum

/-- `bit_length`. -/
def bitLength (n : Nat) : Nat := if n = 0 then 0 else Nat.log2 n + 1

/-- **The second truncation**: the sum `total` of a group aligned to `E`, normalised to `width` bits by truncation
(round toward zero), denormalised if its exponent is below −126, and returned to the 24-bit grid. -/
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

/-- `group_sum`: one grouped, truncating accumulation. -/
def groupSum (width : Nat) (zeroExp : Int) (ts : List Term) : Term :=
  normalize width (maxExp zeroExp ts) (alignedSum width (maxExp zeroExp ts) ts)

/-- `group_step_total`: the incoming value and the group's products to the outgoing value. -/
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

/-! ## The word -/

/-- `pack_fp32`: a term as an FP32 word. Zero is `+0`; above exponent 127, `±inf`; a magnitude without bit 23 (at
exponent −126) is a subnormal. -/
def packTerm (t : Term) : Nat :=
  if t.mag = 0 then 0
  else if 127 < t.exp then ((if t.neg then 1 else 0) <<< 31) ||| 0x7F800000
  else
    let ef : Int := t.exp + 127 - (if t.mag &&& (1 <<< 23) = 0 then 1 else 0)
    ((if t.neg then 1 else 0) <<< 31) ||| (ef.toNat <<< 23) ||| (t.mag &&& 0x7FFFFF)

/-- `pack_value`: `0x7FFFFFFF` for NaN, `±0x7F800000` for the infinities, `packTerm` for a finite value. -/
def packValue : Value → Nat
  | .nan => 0x7FFFFFFF
  | .inf s => ((if s then 1 else 0) <<< 31) ||| 0x7F800000
  | .fin t => packTerm t

/-! ## The pipeline and the step -/

/-- A grouped-sum pipeline (`models.GroupSum`): the group sizes in `k` order, the adder width and the exponent floor. -/
structure Pipeline where
  groups : List Nat
  width : Nat
  zeroExp : Int

/-- `HOPPER_E4M3_K32`: one group of 32, a 14-bit adder, exponent floor −139. -/
def hopperE4m3K32 : Pipeline := ⟨[32], 14, -139⟩

/-- `ADA_E4M3_M16N8K32`: the same adder, two groups of 16. -/
def adaE4m3K32 : Pipeline := ⟨[16, 16], 14, -139⟩

/-- The groups folded over the operand pairs: each group's products join the value carried from the last. -/
def Pipeline.fold (p : Pipeline) : List Nat → Value → List (Code × Code) → Value
  | [], v, _ => v
  | size :: gs, v, ab =>
      p.fold gs (groupStepTotal p.width p.zeroExp v ((ab.take size).map fun x => productTotal x.1 x.2)) (ab.drop size)

/-- `tc_dot_total_e4m3`: `D = acc + Σ_k a[k]·b[k]` on every encoding, for `a` and `b` of length `Σ groups`. -/
def Pipeline.step (p : Pipeline) (acc : Nat) (a b : List Code) : Nat :=
  packValue (p.fold p.groups (fp32Value acc) (a.zip b))

/-- **One `HOPPER_E4M3_K32` step**: the accumulator word and 32 code pairs to the FP32 word written. -/
def hopperStep (acc : Nat) (a b : Fin 32 → Code) : Nat := hopperE4m3K32.step acc (List.ofFn a) (List.ofFn b)

/-- One `ADA_E4M3_M16N8K32` step. -/
def adaStep (acc : Nat) (a b : Fin 32 → Code) : Nat := adaE4m3K32.step acc (List.ofFn a) (List.ofFn b)

/-- The 32 codes packed little-endian in `A` (code `i` is byte `i`). -/
def codes (A : Nat) : Fin 32 → Code := fun i => ⟨(A >>> (8 * i.val)) % 256, Nat.mod_lt _ (by decide)⟩

/-- A conformance vector: the accumulator word, the packed codes `a` and `b`, and the word `verity.ml.tc` computes. -/
structure Vec where
  acc : Nat
  a : Nat
  b : Nat
  d : Nat

/-- A vector holds for `step`. -/
def Vec.holds (step : Nat → (Fin 32 → Code) → (Fin 32 → Code) → Nat) (v : Vec) : Bool :=
  step v.acc (codes v.a) (codes v.b) == v.d

end Pouw.Fp8Atom
