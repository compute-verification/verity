import Evaluate.Fp32

/-!
# The correctly rounded transcendentals of the reference kinds

`F32ExpRn_v1`, `F32ErfRn_v1` and `F32TanhRn_v1` (the vLLM integration's `ref_prims`): RN32 of the value the stated
`decimal` algorithm computes at 100 digits (`_CTX`: precision 100, `ROUND_HALF_EVEN`). Each `decimal` operation the
Python performs is its exact result rounded to 100 significant digits, half to even: `add`, `subtract`, `multiply` and
`divide` (`Dec.add`, `Dec.sub`, `Dec.mul`, `Dec.div`), and `sqrt` and `exp`, which libmpdec rounds correctly too
(`Dec.sqrt`, and `Dec.exp` by Ziv's loop on directed-rounding bounds of the Taylor series). Only values matter: a
decimal's exponent and trailing zeros change no result here. Security: none yet.
-/

namespace Evaluate.Decimal

/-- `_DEC_PREC`. -/
def PREC : Nat := 100

/-- The decimal `(-1)^neg · m · 10^e`. -/
structure Dec where
  neg : Bool
  m : Nat
  e : Int
  deriving BEq, Inhabited

def Dec.zero : Dec := ⟨false, 0, 0⟩
def Dec.one : Dec := ⟨false, 1, 0⟩
def Dec.ofNat (n : Nat) : Dec := ⟨false, n, 0⟩

/-- The decimal digits of `n` (1 for 0). -/
def numDigits (n : Nat) : Nat := Id.run do
  let mut k := n.log2 * 30103 / 100000
  while 10 ^ (k + 1) ≤ n do k := k + 1
  return k + 1

/-- `(-1)^neg · a / b`, `b > 0`, rounded to `p` significant digits (`PREC` unless given), ties to even. -/
def ofQ (neg : Bool) (a b : Nat) (p : Nat := PREC) : Dec :=
  if a = 0 then .zero else
  let scaled (k : Int) : Nat × Nat := if k ≥ 0 then (a, b * 10 ^ k.toNat) else (a * 10 ^ (-k).toNat, b)
  let k0 : Int := (numDigits a : Int) - numDigits b - p
  let k := if (scaled k0).1 / (scaled k0).2 ≥ 10 ^ p then k0 + 1 else k0
  let (n, d) := scaled k
  let q := n / d
  let r := n % d
  let q := if 2 * r > d ∨ (2 * r = d ∧ q % 2 = 1) then q + 1 else q
  if q = 10 ^ p then ⟨neg, 10 ^ (p - 1), k + 1⟩ else ⟨neg, q, k⟩

/-- The decimal's value `(-1)^neg · a / b`. -/
def Dec.q (x : Dec) : Bool × Nat × Nat :=
  if x.e ≥ 0 then (x.neg, x.m * 10 ^ x.e.toNat, 1) else (x.neg, x.m, 10 ^ (-x.e).toNat)

def Dec.negate (x : Dec) : Dec := { x with neg := !x.neg }

/-- `_CTX.multiply`. -/
def Dec.mul (x y : Dec) : Dec :=
  let (sx, ax, bx) := x.q
  let (sy, ay, bY) := y.q
  ofQ (sx != sy) (ax * ay) (bx * bY)

/-- `_CTX.divide`, `y ≠ 0`. -/
def Dec.div (x y : Dec) : Dec :=
  let (sx, ax, bx) := x.q
  let (sy, ay, bY) := y.q
  ofQ (sx != sy) (ax * bY) (bx * ay)

/-- `_CTX.add`. -/
def Dec.add (x y : Dec) : Dec :=
  let (sx, ax, bx) := x.q
  let (sy, ay, bY) := y.q
  let s : Int := (if sx then -1 else 1) * ((ax * bY : Nat) : Int) + (if sy then -1 else 1) * ((ay * bx : Nat) : Int)
  ofQ (s < 0) s.natAbs (bx * bY)

/-- `_CTX.subtract`. -/
def Dec.sub (x y : Dec) : Dec := x.add y.negate

/-- `|x| < |y|`. -/
def Dec.absLt (x y : Dec) : Bool :=
  let (_, ax, bx) := x.q
  let (_, ay, bY) := y.q
  ax * bY < ay * bx

/-- `_CTX.sqrt` of a positive decimal: the root rounded to `PREC` digits, ties to even, from `⌊2 √Y⌋` for the scaling
`Y = x · 10^(-2k)` whose root has `PREC` digits. -/
def Dec.sqrt (x : Dec) : Dec := Id.run do
  let (_, a, b) := x.q
  if a = 0 then return .zero
  let scaled (k : Int) : Nat × Nat := if k ≥ 0 then (a, b * 10 ^ (2 * k).toNat) else (a * 10 ^ (2 * (-k)).toNat, b)
  let mut k : Int := ((numDigits a : Int) - numDigits b) / 2 - PREC
  for _ in [0:8] do
    let r := Fp32.isqrt ((scaled k).1 / (scaled k).2)
    if r ≥ 10 ^ PREC then k := k + 1 else if r < 10 ^ (PREC - 1) then k := k - 1 else break
  let (n, d) := scaled k
  let t := Fp32.isqrt (4 * n / d)
  let m := if t % 2 = 1 ∧ t * t * d = 4 * n then (if (t / 2) % 2 = 0 then t / 2 else t / 2 + 1) else (t + 1) / 2
  return ofQ false (m * (if k ≥ 0 then 10 ^ k.toNat else 1)) (if k ≥ 0 then 1 else 10 ^ (-k).toNat)

/-- Directed-rounding bounds `[lo, hi] / S`, `S = 10^g`, on `exp(a / b)` for `0 ≤ a / b ≤ 2^k / 2`: the Taylor series of
`exp(a / (b 2^k))` (floored terms below, ceiled terms and a tail of at most the last term above, the ratio of terms being
at most 1/2), squared `k` times. -/
def expBounds (a b k g : Nat) : Nat × Nat := Id.run do
  let s := 10 ^ g
  let rLo := a * s / (b * 2 ^ k)
  let rHi := (a * s + b * 2 ^ k - 1) / (b * 2 ^ k)
  let mut lo := s
  let mut t := s
  let mut n := 1
  while t > 0 do
    t := t * rLo / (n * s)
    lo := lo + t
    n := n + 1
  let mut hi := s
  t := s
  n := 1
  while t > 1 do
    t := (t * rHi + n * s - 1) / (n * s)
    hi := hi + t
    n := n + 1
  hi := hi + t
  for _ in [0:k] do
    lo := lo * lo / s
    hi := (hi * hi + s - 1) / s
  return (lo, hi)

/-- `_CTX.exp`: `e^x` rounded to `PREC` digits. `e^x` of a nonzero rational is no rounding boundary, so once both bounds
round alike that is the result (Ziv's loop, the guard digits growing until they do). -/
def Dec.exp (x : Dec) : Dec := Id.run do
  let (neg, a, b) := x.q
  if a = 0 then return .one
  let k := (2 * a / b + 1).log2 + 1
  let mut g := PREC + 30
  for _ in [0:64] do
    let (lo, hi) := expBounds a b k g
    let s := 10 ^ g
    let (rLo, rHi) := if neg then (ofQ false s hi, ofQ false s lo) else (ofQ false lo s, ofQ false hi s)
    if rLo == rHi then return rLo
    g := g + 60
  return .zero

/-- `_PI_100`. -/
def PI_100 : Dec := ⟨false, 314159265358979323846264338327950288419716939937510582097494459230781640628620899862803482534211706798214808651, -110⟩

/-- `2 / sqrt(pi)` as `_dec_erf` computes it. -/
def TWO_OVER_SQRT_PI : Dec := (Dec.ofNat 2).div PI_100.sqrt

/-- `10^-(_DEC_PREC + 5)`. -/
def TINY : Dec := ⟨false, 1, -(PREC + 5 : Nat)⟩

/-- `abs(x)` in Python's default context (28 digits, half to even), which rounds as it takes the magnitude. -/
def Dec.abs28 (x : Dec) : Dec := let (_, a, b) := x.q; ofQ false a b 28

/-- `_dec_erf`: `(2 / sqrt(pi)) Σ (-1)^n x^(2n+1) / (n! (2n+1))`, summed until a term, its magnitude taken by `abs`, is
below `TINY`. -/
def erf (x : Dec) : Dec := Id.run do
  let x2 := x.mul x
  let mut term := x
  let mut total := x
  let mut n := 0
  for _ in [0:100000] do
    n := n + 1
    term := (term.mul x2).div (.ofNat n)
    let contrib := term.div (.ofNat (2 * n + 1))
    total := if n % 2 == 0 then total.add contrib else total.sub contrib
    if contrib.abs28.absLt TINY then break
  return total.mul TWO_OVER_SQRT_PI

/-- `_dec_of_frac` of a finite word's value, `(-1)^neg · u · 2^-149`. -/
def ofUnits (neg : Bool) (u : Nat) : Dec := ofQ neg u (2 ^ 149)

/-- `_f32_of_dec`: RN-even of the decimal to a word (`fp32._round_f32`); zero is `+0`. -/
def Dec.toF32 (x : Dec) : Nat :=
  let (neg, a, b) := x.q
  if a = 0 then 0 else Fp32.roundRn neg (a * 2 ^ 149) b

open Fp32 (Decoded decode F32_NAN) in
/-- `F32ExpRn_v1`, `ref_prims.f32_exp_rn_bits`: NaN gives `0x7FC00000`, `+inf` and `x > 89` give `+inf`, `-inf` and
`x < -105` give `+0`, any other x the 100-digit `exp` rounded. Security: none yet. -/
def f32ExpRn (u : Nat) : Nat :=
  match decode u with
  | .nan => F32_NAN
  | .inf neg => if neg then 0 else 0x7F800000
  | .fin neg n =>
    if !neg && n > 89 * 2 ^ 149 then 0x7F800000
    else if neg && n > 105 * 2 ^ 149 then 0
    else (ofUnits neg n).exp.toF32

open Fp32 (Decoded decode F32_NAN) in
/-- `F32ErfRn_v1`, `ref_prims.f32_erf_rn_bits`: NaN gives `0x7FC00000`, `±0` itself, `±inf` and `|x| ≥ 4.2` give `±1`,
any other x the 100-digit series rounded. Security: none yet. -/
def f32ErfRn (u : Nat) : Nat :=
  match decode u with
  | .nan => F32_NAN
  | .inf neg => if neg then 0xBF800000 else 0x3F800000
  | .fin neg n =>
    if n = 0 then u &&& 0x80000000
    else if 5 * n ≥ 21 * 2 ^ 149 then (if neg then 0xBF800000 else 0x3F800000)
    else (erf (ofUnits neg n)).toF32

open Fp32 (Decoded decode F32_NAN) in
/-- `F32TanhRn_v1`, `ref_prims.f32_tanh_rn_bits`: NaN gives `0x7FC00000`, `±0` itself, `±inf` and `|x| ≥ 9.2` give `±1`,
any other x `(e^{2x} - 1) / (e^{2x} + 1)` at 100 digits, rounded. Security: none yet. -/
def f32TanhRn (u : Nat) : Nat :=
  match decode u with
  | .nan => F32_NAN
  | .inf neg => if neg then 0xBF800000 else 0x3F800000
  | .fin neg n =>
    if n = 0 then u &&& 0x80000000
    else if 5 * n ≥ 46 * 2 ^ 149 then (if neg then 0xBF800000 else 0x3F800000)
    else
      let e2 := ((Dec.ofNat 2).mul (ofUnits neg n)).exp
      (e2.sub .one).div (e2.add .one) |>.toF32

end Evaluate.Decimal
