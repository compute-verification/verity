import Evaluate.MufuFits
import Flock.Bytes
import Flock.Hash

/-!
# The MUFU primitives

`verity.ml.circuits.ops.mufu` (and the vLLM integration's `hopper.Fa3InvSum_v1`): the special-function unit's `ex2`, `rcp`,
`sqrt`, `rsqrt` and `tanh` as sm_89 computes them, FA2's and FA3's inverse sums, and `div.full.f32`'s reciprocal. Each is an
exact bit rule around a table read, transcribed from the Python. Security has no MUFU definition yet.

The Python reads the measured tables (`mufu.table`, the library's words; `mufu_tanh_shards` for MUFU.TANH). Here a table is
its interpolant (`Evaluate.MufuFits`, from `verity.ml.circuits.ops.boolean.mufu`, where the circuit build checks it against
the measured words), and `tableSha512` hashes each table as the library stores it, so the suite holds the words to the
library's SHA-512 (`Flock.LIBRARY_TABLES`).
-/

namespace Evaluate.Mufu

def NAN : Nat := 0x7FFFFFFF
def INF : Nat := 0x7F800000
def ONE : Nat := 0x3F800000
def SIGN : Nat := 0x80000000

/-- `boolean.mufu._square`: of `u²`'s folded partial products (bit `a` at column `2a`, bits `a < b` at `a + b + 1`) of an
`nb`-bit `u`, the sum of those in columns `≥ p`, shifted right by `p`. -/
def square (nb p u : Nat) : Nat := Id.run do
  let mut q := 0
  for a in [0:nb] do
    let ua := (u >>> a) &&& 1
    if 2 * a ≥ p then q := q + (ua <<< (2 * a - p))
    for b in [max (a + 1) (p - a - 1):nb] do
      q := q + ((ua &&& ((u >>> b) &&& 1)) <<< (a + b + 1 - p))
  return q

def SQUARE_16_17 : Array Nat := Array.ofFn (n := 2 ^ 16) fun i => square 16 17 i.val
def SQUARE_17_19 : Array Nat := Array.ofFn (n := 2 ^ 17) fun i => square 17 19 i.val

/-- `square nb p u`, read from the precomputed squares of the fits' two shapes. -/
def squareOf (nb p u : Nat) : Nat :=
  if nb == 16 && p == 17 then SQUARE_16_17[u]! else if nb == 17 && p == 19 then SQUARE_17_19[u]! else square nb p u

/-- `Fit.interpolant` with `overrides`: the table's word at index `i`, `base + ((C0 + C1 u + (C2 Q(u)) << ss) >> sh)`. -/
def Fit.word (f : Fit) (i : Nat) : Nat :=
  let u := i % 2 ^ f.nb
  let g := i >>> f.nb
  let v : Int := (f.c0[g]! + f.c1[g]! * u + f.c2[g]! * squareOf f.nb f.p u * 2 ^ f.ss) >>> f.sh
  (f.base + v + (f.overrides.lookup i).getD 0).toNat

/-- `mufu_ex2_bits` on the ex2 table: x as the signed fixed-point number `floor(x 2^23)`, whose integer part scales the
table's word at its fraction. -/
def ex2Bits (x : Nat) : Nat :=
  let x := x % 2 ^ 32
  let ax := x &&& 0x7FFFFFFF
  let neg := x >>> 31 == 1
  let e := ax >>> 23
  if e == 0xFF then (if ax &&& 0x7FFFFF != 0 then NAN else if neg then 0 else INF)
  else if e == 0 then ONE
  else if e ≥ 127 + 30 then (if neg then 0 else INF)
  else
    let mant := (ax &&& 0x7FFFFF) ||| 0x800000
    let k := if e ≥ 127 then mant <<< (e - 127) else mant >>> (127 - e)
    let fixed : Int := if !neg then k else if k &&& 0x7FFFFF == 0 then -(k : Int) else -(k : Int) - 1
    let w := EX2_FIT.word (fixed % 2 ^ 23).toNat
    let re : Int := (((w >>> 23) &&& 0xFF : Nat) : Int) + (fixed >>> 23)
    if re ≥ 0xFF then INF else if re ≤ 0 then 0 else (re.toNat <<< 23) ||| (w &&& 0x7FFFFF)

/-- `mufu_rcp_bits` on the rcp table: the table's word at the mantissa, the exponent negated exactly. -/
def rcpBits (x : Nat) : Nat :=
  let x := x % 2 ^ 32
  let s := x &&& SIGN
  let ax := x &&& 0x7FFFFFFF
  let e := ax >>> 23
  if e == 0xFF then (if ax &&& 0x7FFFFF != 0 then NAN else s)
  else if e == 0 then s ||| INF
  else
    let w := RCP_FIT.word (ax &&& 0x7FFFFF)
    let re : Int := (((w >>> 23) &&& 0xFF : Nat) : Int) - ((e : Int) - 127)
    if re ≤ 0 then s else if re ≥ 0xFF then s ||| INF else s ||| (re.toNat <<< 23) ||| (w &&& 0x7FFFFF)

/-- `mufu._sqrt_like`: the rsq or sqrt table's word at the exponent's parity and the mantissa, the halved exponent added
(sqrt) or subtracted (rsq). -/
def sqrtLike (x : Nat) (rsq : Bool) : Nat :=
  let x := x % 2 ^ 32
  let s := x &&& SIGN
  let ax := x &&& 0x7FFFFFFF
  let e := ax >>> 23
  if e == 0xFF then (if ax &&& 0x7FFFFF != 0 || s != 0 then NAN else if rsq then 0 else INF)
  else if e == 0 then (if rsq then s ||| INF else s)
  else if s != 0 then NAN
  else
    let k : Int := (e : Int) - 127
    let par := (k % 2).toNat
    let w := (if rsq then RSQ_FIT else SQRT_FIT).word ((par <<< 23) ||| (ax &&& 0x7FFFFF))
    let half : Int := (k - par) / 2
    let re : Int := (((w >>> 23) &&& 0xFF : Nat) : Int) + (if rsq then -half else half)
    (re * 2 ^ 23).toNat ||| (w &&& 0x7FFFFF)

/-- `mufu._times_2_24`: `u · 2^24` for `|u| < 2^-126`, exact (the mantissa normalized by its bit length). -/
def times2p24 (u : Nat) : Nat :=
  let m := u &&& 0x7FFFFF
  if m == 0 then u else
  let k := Nat.log2 m + 1
  (u &&& SIGN) ||| ((k + 1) <<< 23) ||| ((m <<< (24 - k)) &&& 0x7FFFFF)

/-- `mufu._div_full_divisor`: `b · 0.25` when `2^126 < |b| < ∞`, `b · 2^24` when `|b| < 2^-126`, else `b`. -/
def divFullDivisor (b : Nat) : Nat :=
  let b := b % 2 ^ 32
  let ab := b &&& 0x7FFFFFFF
  if ab < 0x00800000 then times2p24 b else if 0x7E800000 < ab && ab < INF then b - (2 <<< 23) else b

private def isNan (x : Nat) : Bool := x &&& 0x7FFFFFFF > INF

/-- `MufuEx2Ftz_v1`: `ex2.approx.ftz.f32`. Security: none yet. -/
def mufuEx2Ftz (x : Nat) : Nat := ex2Bits x

/-- `MufuRcpFtz_v1`: `rcp.approx.ftz.f32`. Security: none yet. -/
def mufuRcpFtz (x : Nat) : Nat := rcpBits x

/-- `Fa2InvSum_v1`: FA2's `(sum == 0 || sum != sum) ? 1 : MUFU.RCP(sum)`. Security: none yet. -/
def fa2InvSum (s : Nat) : Nat := if s &&& 0x7FFFFFFF == 0 || isNan s then ONE else rcpBits s

/-- `Fa3InvSum_v1` (`hopper`): FA3's `(sum == 0 || sum != sum) ? 0 : MUFU.RCP(sum)`. Security: none yet. -/
def fa3InvSum (s : Nat) : Nat := if s &&& 0x7FFFFFFF == 0 || isNan s then 0 else rcpBits s

/-- `MufuSqrtFtz_v1`: `sqrt.approx.ftz.f32`. Security: none yet. -/
def mufuSqrtFtz (x : Nat) : Nat := sqrtLike x false

/-- `DivFullRcp_v1`: `div.full.f32`'s reciprocal, MUFU.RCP of the scaled divisor. Security: none yet. -/
def divFullRcp (b : Nat) : Nat := rcpBits (divFullDivisor b)

/-- `RsqrtApprox_v1`: `rsqrt.approx.f32`, MUFU.RSQ, a subnormal scaled by 2^24 first and its result by 2^12 unless it is
infinite or NaN. Security: none yet. -/
def rsqrtApprox (x : Nat) : Nat :=
  let x := x % 2 ^ 32
  if x &&& 0x7FFFFFFF ≥ 0x00800000 then sqrtLike x true else
  let r := sqrtLike (times2p24 x) true
  if r &&& INF == INF then r else r + (12 <<< 23)

/-- The first segment and the segment bits of MUFU.TANH's binade of exponent `e`, if `e` is one of the fit's. -/
def tanhBinade? (e : Nat) : Option (Nat × Nat) :=
  (TANH_FIT.binades.foldl (fun (acc : Option (Nat × Nat) × Nat) (be, sb) =>
    (if be == e then some (acc.2, sb) else acc.1, acc.2 + 2 ^ sb)) (none, 0)).1

/-- `TanhFit.words` at the positive word `a` of one of the fit's binades: the segment's `E` and `V`, then `((E - 1) << 23) + V`
below `V = 2^24`, else `(E << 23) + (V >> 1)`. Off the binades, 0 (no table word). -/
def tanhWord (a : Nat) : Nat :=
  match tanhBinade? (a >>> 23) with
  | none => 0
  | some (first, sb) =>
    let f := TANH_FIT
    let m := a &&& 0x7FFFFF
    let g := first + (m >>> (23 - sb))
    let u := (m >>> (23 - sb - f.nb)) % 2 ^ f.nb
    let v : Int := (f.c0[g]! + f.c1[g]! * u + f.c2[g]! * squareOf f.nb f.p u * 2 ^ f.ss) >>> f.sh
    let ee : Int := f.e[g]!
    (if v < 2 ^ 24 then (ee - 1) * 2 ^ 23 + v else ee * 2 ^ 23 + (v >>> 1)).toNat

/-- `MUFU_TANH_RULES`: `|x|` below `IDENTITY_BELOW` is x, from `SATURATE_FROM` ±1. -/
def IDENTITY_BELOW : Nat := 0x3B800001
def SATURATE_FROM : Nat := 0x41000000

/-- `MufuTanh_v1`: `tanh.approx.f32` on sm_89: NaN gives 0x7FFFFFFF, `|x| ≥ 8` gives ±1, `|x| ≤ 2^-8` gives x, every other
word the table at `|x|` with x's sign. Security: none yet. -/
def mufuTanh (x : Nat) : Nat :=
  let u := x % 2 ^ 32
  let sign := u &&& SIGN
  let a := u &&& 0x7FFFFFFF
  if a > INF then NAN
  else if a ≥ SATURATE_FROM then ONE ||| sign
  else if a < IDENTITY_BELOW then u
  else tanhWord a ^^^ sign

/-- `tail_pieces._mufu_tanh_table`: the library's `tanh_mufu`, MUFU.TANH's word at every positive word of
`[IDENTITY_BELOW, SATURATE_FROM)` by its low 27 bits, 0 elsewhere. -/
def tanhTableWord (i : Nat) : Nat :=
  let w := (IDENTITY_BELOW / 2 ^ 27) * 2 ^ 27 + i
  let w := if w < IDENTITY_BELOW then w + 2 ^ 27 else w
  if w < SATURATE_FROM then tanhWord w else 0

/-- The library tables these primitives read (`verity.ml.circuits.library.TABLES`): each one's word count and words. -/
def libraryTable? (name : String) : Option (Nat × (Nat → Nat)) :=
  match name with
  | "rsq" => some (2 ^ 24, RSQ_FIT.word)
  | "sqrt" => some (2 ^ 24, SQRT_FIT.word)
  | "rcp" => some (2 ^ 23, RCP_FIT.word)
  | "ex2" => some (2 ^ 23, EX2_FIT.word)
  | "tanh_mufu" => some (2 ^ 27, tanhTableWord)
  | _ => none

/-- The SHA-512 (hex) of a library table's words as the library stores them, raw little-endian `u32`. -/
def tableSha512 (name : String) : Option String := do
  let (n, word) ← libraryTable? name
  let mut bytes := ByteArray.emptyWithCapacity (4 * n + 256)
  for i in [0:n] do
    let w := word i
    bytes := ((((bytes.push (w % 256).toUInt8).push ((w >>> 8) % 256).toUInt8).push ((w >>> 16) % 256).toUInt8).push
      ((w >>> 24) % 256).toUInt8)
  return Flock.hexOf (Flock.Sha512.hash bytes)

end Evaluate.Mufu
