import Evaluate.Fp32

/-!
# The sampler's primitives: libdevice's logs, Philox and Triton's rand, and top-p's keep count

Executable transcriptions of the vLLM integration's `sampling` (`GumbelStreamKey_v1`, `GumbelNoiseLane_v1` / `_v2`) and
`topp_word_gates` (`NvLogf_v1` / `_v2`, `TopPNumkeep_v1`). Security defines none of them: the Python function is the
definition.

The Python computes libdevice's FTZ steps in Python `float`, a double, on the host, so its words are what those doubles give:

- a product of two FP32 values is exact in a double, so `mul_ftz` is `RN32` of the exact product;
- `add_ftz` and `fma_ftz` round the exact sum to a double first (`p + c`, `a + b`), and `_round_exact` then gives that
  double's `RN32` (its `math.nextafter` steps a double's ulp, not a float's, so it never moves the word): both are
  `RN32(RN64(exact))`, a double rounding;
- `add_rz_ftz` is the exact sum rounded toward zero, saturating at the largest finite word;
- a NaN is the host's: an FP32 NaN read into a double and back is quieted (bit 22 set), CPython's double product returns
  its first NaN operand and its double sum its second, and an invalid operation (`inf · 0`, `inf - inf`) the x86 default
  NaN, `0xFFC00000` (`HostDouble`).

Words as in `Evaluate.Scalar`; `Int` holds the Python's signed integers, and its `x & -2^23` is `x - x mod 2^23`.
-/

namespace Evaluate.Libdevice

open Evaluate.Scalar (M32 lowBits)
open Evaluate.Fp32 (decode roundRn infWord ftzBits f32Gt i32ToF32)

/-- A Python `float` the logs compute with: a NaN as the FP32 word it packs to, an infinity, or a finite value in units of
`2^-298` (every product of two FP32 values, and every sum of one with an FP32 value, is one). -/
inductive HostDouble where
  | nan (word : Nat)
  | inf (negative : Bool)
  | fin (negative : Bool) (n : Nat)

/-- `_f(u)` widened to units of `2^-298`: a NaN quieted. -/
def ofF32 (u : Nat) : HostDouble :=
  match decode u with
  | .nan => .nan ((u &&& M32) ||| 0x400000)
  | .inf s => .inf s
  | .fin s m => .fin s (m <<< 149)

/-- `_f(a) * _f(b)`, exact: the first NaN operand quieted, else `0xFFC00000` for `inf · 0`. -/
def hostMul (a b : Nat) : HostDouble :=
  match decode a, decode b with
  | .nan, _ => .nan ((a &&& M32) ||| 0x400000)
  | _, .nan => .nan ((b &&& M32) ||| 0x400000)
  | .inf sa, .inf sb => .inf (sa ^^ sb)
  | .inf sa, .fin sb m | .fin sb m, .inf sa => if m = 0 then .nan 0xFFC00000 else .inf (sa ^^ sb)
  | .fin sa ma, .fin sb mb => .fin (sa ^^ sb) (ma * mb)

/-- `x + y`, the exact sum (not yet rounded to a double): the second NaN operand, else the first, else `0xFFC00000` for
`inf - inf`; an exact zero is `-0` only when both are zeros of negative sign. -/
def hostAdd : HostDouble → HostDouble → HostDouble
  | _, .nan w => .nan w
  | .nan w, _ => .nan w
  | .inf sx, .inf sy => if sx = sy then .inf sx else .nan 0xFFC00000
  | .inf s, .fin _ _ | .fin _ _, .inf s => .inf s
  | .fin sx nx, .fin sy ny =>
    if sx = sy then .fin sx (nx + ny)
    else if nx > ny then .fin sx (nx - ny) else if ny > nx then .fin sy (ny - nx) else .fin false 0

/-- `n` rounded to 53 significant bits, ties to even: a double's rounding (the values here are far from a double's
subnormals and its overflow). -/
def rn53 (n : Nat) : Nat :=
  if n < 2 ^ 53 then n
  else
    let k := n.log2 - 52
    let q := n >>> k
    let r := n % 2 ^ k
    let q := if 2 * r > 2 ^ k ∨ (2 * r = 2 ^ k ∧ q % 2 = 1) then q + 1 else q
    q <<< k

/-- A double's rounding of an exact value. -/
def rn64 : HostDouble → HostDouble
  | .fin s n => .fin s (rn53 n)
  | x => x

/-- `_u(x)`: the FP32 word of the double, to nearest even, `±inf` past the range. -/
def toF32 : HostDouble → Nat
  | .nan w => w
  | .inf s => infWord s
  | .fin s n => roundRn s n (2 ^ 149)

/-- `sampling.fma_ftz`, `fma.rn.ftz.f32` as the Python computes it: `ftz(RN32(RN64(ftz(a) · ftz(b) + ftz(c))))`. -/
def fmaFtz (a b c : Nat) : Nat := ftzBits (toF32 (rn64 (hostAdd (hostMul (ftzBits a) (ftzBits b)) (ofF32 (ftzBits c)))))

/-- `sampling.mul_ftz`: `ftz(RN32(ftz(a) · ftz(b)))`. -/
def mulFtz (a b : Nat) : Nat := ftzBits (toF32 (hostMul (ftzBits a) (ftzBits b)))

/-- `sampling.add_ftz`: `ftz(RN32(RN64(ftz(a) + ftz(b))))`. -/
def addFtz (a b : Nat) : Nat := ftzBits (toF32 (rn64 (hostAdd (ofF32 (ftzBits a)) (ofF32 (ftzBits b)))))

/-- `sampling.add_rz_ftz`, `add.rz.ftz.f32`: the exact sum rounded toward zero (its significand truncated, the largest
finite word past the range), a NaN, an infinity and an exact zero as `add_ftz` gives them. -/
def addRzFtz (a b : Nat) : Nat :=
  match hostAdd (ofF32 (ftzBits a)) (ofF32 (ftzBits b)) with
  | .fin s n =>
    if n = 0 then ftzBits (toF32 (.fin s 0))
    else
      let j := (n / 2 ^ 149).log2 - 23
      let word := (j <<< 23) + n / (2 ^ 149 <<< j)
      ftzBits (Fp32.zeroWord s ||| (if word ≥ 0x7F800000 then 0x7F7FFFFF else word))
  | x => ftzBits (toF32 x)

/-- `sampling._i32`: the signed value of a 32-bit word. -/
def i32 (u : Nat) : Int := let u := u &&& M32; if u &&& 0x80000000 ≠ 0 then (u : Int) - 2 ^ 32 else u

/-- The Python's `x & -8388608`: `x` with its low 23 bits cleared. -/
def clearLow23 (x : Int) : Int := x - x % 2 ^ 23

/-- `sampling._neg_not_nan`. -/
def negNotNan (a : Nat) : Bool := a ≥ 0x80000000 && (a &&& 0x7FFFFFFF) ≤ 0x7F800000

/-- `sampling.LOGF_POLY`. -/
def LOGF_POLY : List Nat := [0xBE055027, 0x3E1039F6, 0xBDF8CDCC, 0x3E0F2955, 0xBE2AD8B9, 0x3E4CED0B, 0xBE7FFF22, 0x3EAAAA78]
/-- `sampling.LOG1PF_POLY`. -/
def LOG1PF_POLY : List Nat := [0xBD39BF78, 0x3DD80012, 0xBE0778E0, 0x3E146475, 0xBE2A68DD, 0x3E4CAF9E, 0xBE800042, 0x3EAAAAE6]
/-- `sampling.LN2_BITS`. -/
def LN2_BITS : Nat := 0x3F317218
/-- `sampling.LOG_INVALID_NAN`. -/
def LOG_INVALID_NAN : Nat := 0xFFC00000
/-- `sampling.LOG_NAN`. -/
def LOG_NAN : Nat := 0x7FFFFFFF

/-- The Horner tail both logs share: `fma` down the coefficients from the first two, then `-1/2`, `· x`, `+ x`, and the
exponent's `e · ln 2`. -/
def logPoly (poly : List Nat) (x e : Nat) : Nat :=
  let r := (poly.drop 2).foldl (fun r k => fmaFtz r x k) (fmaFtz poly[0]! x poly[1]!)
  let r := fmaFtz (mulFtz (fmaFtz r x 0xBF000000) x) x x
  fmaFtz e LN2_BITS r

/-- `NvLogf_v1`, `sampling.nv_logf`: libdevice `__nv_logf` on an FP32 word (FTZ). Security: none yet. -/
def nvLogf (a : Nat) : Nat :=
  let a := a &&& M32
  let small := f32Gt 0x00800000 a
  let a2 := if small then mulFtz a 0x4B000000 else a
  let ai := i32 a2
  let e := i32 (lowBits (clearLow23 (ai - 1059760811)) 32)
  let m := lowBits (ai - e) 32
  let i2 := fmaFtz (i32ToF32 (lowBits e 32)) 0x34000000 (if small then 0xC1B80000 else 0)
  let r := logPoly LOGF_POLY (addFtz m 0xBF800000) i2
  let r := if a2 ≥ 0x7F800000 then (if negNotNan a2 then LOG_INVALID_NAN else fmaFtz a2 0x7F800000 0x7F800000) else r
  if a2 &&& 0x7FFFFFFF = 0 then 0xFF800000 else r

/-- `sampling._log1pf_from_sum`: `__nv_log1pf` after its `__fadd_rz(a, 1)`, whose word is `s`. -/
def log1pfFromSum (a s : Nat) : Nat :=
  let e := i32 (lowBits (clearLow23 (i32 s - 1061158912)) 32)
  let m1 := lowBits (i32 a - e) 32
  let t := lowBits (1082130432 - e) 32
  let u := addFtz m1 (fmaFtz 0x3E800000 t 0xBF800000)
  let r := logPoly LOG1PF_POLY u (mulFtz (i32ToF32 (lowBits e 32)) 0x34000000)
  if a ≥ 0x7F800000 then
    let r := if i32 a > -1082130432 then (if negNotNan a then LOG_INVALID_NAN else fmaFtz a 0x7F800000 0x7F800000) else r
    if a &&& 0x7FFFFFFF = 0 then 0x80000000 else r
  else r

/-- `sampling.nv_log1pf`: libdevice `__nv_log1pf` (FTZ), its `__fadd_rz` a true toward-zero add. -/
def nvLog1pf (a : Nat) : Nat := log1pfFromSum (a &&& M32) (addRzFtz (a &&& M32) 0x3F800000)

/-- `sampling._gpu_nan`: every NaN `LOG_NAN`. -/
def gpuNan (u : Nat) : Nat := if Scalar.f32IsNan u then LOG_NAN else u

/-- `NvLogf_v2`, `sampling.nv_logf_gpu`: `nv_logf` with every NaN result `0x7FFFFFFF`. Security: none yet. -/
def nvLogfGpu (a : Nat) : Nat := gpuNan (nvLogf a)

/-- `sampling.nv_log1pf_gpu`. -/
def nvLog1pfGpu (a : Nat) : Nat := gpuNan (nvLog1pf a)

/-- `sampling.philox4x32_10_word0`: the first output word of Random123's Philox4x32-10 (Triton's `philox_impl`). -/
def philoxWord0 (c0 c1 c2 c3 k0 k1 : Nat) : Nat := Id.run do
  let mut c0 := c0
  let mut c1 := c1
  let mut c2 := c2
  let mut c3 := c3
  let mut k0 := k0
  let mut k1 := k1
  for _ in [0:10] do
    let pb := c2 * 0xCD9E8D57
    let pa := c0 * 0xD2511F53
    (c0, c1, c2, c3) := (((pb >>> 32) &&& M32) ^^^ c1 ^^^ k0, pb &&& M32, ((pa >>> 32) &&& M32) ^^^ c3 ^^^ k1, pa &&& M32)
    k0 := (k0 + 0x9E3779B9) &&& M32
    k1 := (k1 + 0xBB67AE85) &&& M32
  return c0

/-- `GumbelStreamKey_v1`, `sampling.stream_key`: `tl.randint(seed, pos)`, key `(seed_lo, seed_hi)`, counter
`(pos_lo, pos_hi, 0, 0)`. Security: none yet. -/
def streamKey (seed pos : Nat) : Nat :=
  let seed := seed &&& (2 ^ 64 - 1)
  let pos := pos &&& (2 ^ 64 - 1)
  philoxWord0 (pos &&& M32) ((pos >>> 32) &&& M32) 0 0 (seed &&& M32) ((seed >>> 32) &&& M32)

/-- `sampling._rand_lane`: `max(tl.rand(key, v), 2^-31')`, Philox's word 0 through `uint_to_uniform_float`. -/
def randLane (key v : Nat) : Nat :=
  let xi := i32 (philoxWord0 (v &&& M32) 0 0 0 (key &&& M32) 0)
  let xi := if xi < 0 then -xi - 1 else xi
  let u := mulFtz (i32ToF32 xi.toNat) 0x2FFFFFFF
  if f32Gt 0x2FFFFFFF u then 0x2FFFFFFF else u

/-- `GumbelNoiseLane_v1`, `sampling.noise_lane`: `-logf(-log1pf(-u))` of the lane's uniform. Security: none yet. -/
def noiseLane (key v : Nat) : Nat :=
  nvLogf (nvLog1pf (randLane key v ^^^ 0x80000000) ^^^ 0x80000000) ^^^ 0x80000000

/-- `GumbelNoiseLane_v2`, `sampling.noise_lane_gpu`: `noise_lane` through the GPU-NaN logs. Security: none yet. -/
def noiseLaneGpu (key v : Nat) : Nat :=
  nvLogfGpu (nvLog1pfGpu (randLane key v ^^^ 0x80000000) ^^^ 0x80000000) ^^^ 0x80000000

/-- `TopPNumkeep_v1`, `topp_word_gates._numkeep`: `min(max(numdup - qi, 1), numdup)`, `qi` the FP32 `q` truncated (0
unless `q` is finite and positive, `2^32 - 1` from `2^32` on); 0 when `numdup` is 0. Security: none yet. -/
def toppNumkeep (nd q : Nat) : Nat :=
  let nd := nd &&& M32
  let qi := match decode q with
    | .fin false n => min (n / 2 ^ 149) M32
    | _ => 0
  if nd = 0 then 0 else if qi < nd then nd - qi else 1

end Evaluate.Libdevice
