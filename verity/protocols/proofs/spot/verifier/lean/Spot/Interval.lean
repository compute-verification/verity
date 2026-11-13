module

/-!
# Interval bounds on `e^x`, `ln y` and `c^θ` (`Spot.Interval`)

Executable bounds on rationals, each a multiple of `2^-p` rounded outward, for checks whose bounds read `e^x`, `ln y` or
`c^θ` (the planner's two-stage designs):

* `expLower x p ≤ e^x ≤ expUpper x p` for every rational `x` (`Properties.Spot.ExpLowerSound`,
  `ExpUpperSound`);
* `lnLower y p ≤ ln y ≤ lnUpper y p` for every rational `y` (`LnLowerSound`, `LnUpperSound`);
* `powLower c θ p ≤ c^θ ≤ powUpper c θ p` for every rational `c` and `θ` (`PowLowerSound`, `PowUpperSound`).

`ln` and `c^θ` are `Real.log` and `Real.rpow`, so at `y ≤ 0` and `c ≤ 0` they take Mathlib's values (`ln 0 = 0`,
`ln y = ln (−y)`, `0^θ = 0` for `θ ≠ 0`, `c^θ = |c|^θ cos(πθ)` for `c < 0`), and the bounds hold there too.

`e^x` for `x ≥ 0` is `(e^r)^(2^s)` at `r = x / 2^s ≤ 1`: the exponential series of `r` to `n` terms, each term rounded
toward the bound's side at `2^-w`, plus the remainder `r^n (n + 1) / (n! n)` for the upper bound, then squared `s`
times, each square rounded the same way. `e^x` for `x < 0` is `1 / e^{-x}`. `ln y` is checked rather than summed: an
approximation (`lnApprox`, on which no bound rests) proposes a multiple `u` of `2^-p`, which `lnUpper` returns once
`y ≤ expLower u`, trying the next few multiples before falling back on `ln y ≤ y − 1` (`lnLower`: `expUpper l ≤ y`, and
`ln y ≥ 1 − 1/y`). `c^θ` is `e^{θ ln c}`, with `ln c` bounded on the side `θ`'s sign picks, and `±|c|^θ` for
`c < 0`.

The working precisions (`expPrec`, `lnPrec`, `powPrec`) and the counts (`expShift`, `expTerms`, `lnTries`) only decide
how tight a bound is and how long it takes: the bounds hold at every choice.
-/

@[expose] public section

namespace Spot.Interval

/-- `⌊q · 2^p⌋ / 2^p`: the largest multiple of `2^-p` at most `q`. -/
def floorTo (q : Rat) (p : Nat) : Rat := ((q * 2 ^ p).floor : Rat) / 2 ^ p

/-- `⌈q · 2^p⌉ / 2^p`: the least multiple of `2^-p` at least `q`. -/
def ceilTo (q : Rat) (p : Nat) : Rat := ((q * 2 ^ p).ceil : Rat) / 2 ^ p

/-- `q` rounded to a multiple of `2^-p`: up when `up`, else down. -/
def roundTo (up : Bool) (q : Rat) (p : Nat) : Rat := if up then ceilTo q p else floorTo q p

/-! ## The exponential -/

/-- The exponential series of `r` to `n` terms, each term rounded toward `up` at `2^-w`, and its next term: from `1`,
each term is the last times `r / (i + 1)`, rounded. For `r ≥ 0` the sum is at most `∑ i < n, r^i / i!` and the next term
at most `r^n / n!`, or at least them when `up`. -/
def series (up : Bool) (r : Rat) (w : Nat) : Nat → Rat × Rat
  | 0 => (0, 1)
  | n + 1 =>
    let st := series up r w n
    (st.1 + st.2, roundTo up (st.2 * r / ((n + 1 : Nat) : Rat)) w)

/-- `v` squared `k` times, each square rounded toward `up` at `2^-w`: for `v ≥ 0`, at most `v^(2^k)`, or at least it
when `up`. -/
def squares (up : Bool) (w : Nat) : Nat → Rat → Rat
  | 0, v => v
  | k + 1, v => squares up w k (roundTo up (v * v) w)

/-- How far past `x ≤ 2^s` `halvings` halves, so the series runs at `r ≤ 2^-expShift`. -/
def expShift : Nat := 8

/-- How many times `expNonneg` halves `x ≥ 0`: `s` with `x ≤ 2^(s - expShift)`. -/
def halvings (x : Rat) : Nat := x.ceil.toNat.log2 + 1 + expShift

/-- How many terms of the series `expNonneg` sums at precision `w` (at least one). -/
def expTerms (w : Nat) : Nat := w / 10 + 4

/-- `e^x` for `x ≥ 0` at precision `w`: below it, or above it when `up`. -/
def expNonneg (up : Bool) (x : Rat) (w : Nat) : Rat :=
  let s := halvings x
  let n := expTerms w
  let st := series up (x / 2 ^ s) w n
  squares up w s (if up then st.1 + st.2 * ((n + 1 : Nat) : Rat) / (n : Rat) else st.1)

/-- The precision `expUpper` and `expLower` work at: `p`, `e^x`'s integer bits, and room for the squarings' error. -/
def expPrec (x : Rat) (p : Nat) : Nat := p + halvings (if 0 ≤ x then x else -x) + 2 * x.ceil.toNat + 8

/-- A multiple of `2^-p` at least `e^x`. -/
def expUpper (x : Rat) (p : Nat) : Rat :=
  let w := expPrec x p
  if 0 ≤ x then ceilTo (expNonneg true x w) p else ceilTo (1 / max 1 (expNonneg false (-x) w)) p

/-- A multiple of `2^-p` at most `e^x`. -/
def expLower (x : Rat) (p : Nat) : Rat :=
  let w := expPrec x p
  if 0 ≤ x then floorTo (expNonneg false x w) p else floorTo (1 / expNonneg true (-x) w) p

/-! ## The logarithm -/

/-- `2 ∑ i < n, t^(2i+1) / (2i + 1)` from the `i`-th term on, the powers rounded down at `2^-w`, added to `acc`. -/
def atanhTerms (t2 : Rat) (w : Nat) : Nat → Nat → Rat → Rat → Rat
  | 0, _, _, acc => acc
  | n + 1, i, pw, acc => atanhTerms t2 w n (i + 1) (floorTo (pw * t2) w) (acc + 2 * pw / ((2 * i + 1 : Nat) : Rat))

/-- About `ln((1 + t)/(1 − t)) = 2 atanh t` for `|t| ≤ 1/3`, within about `2^-(w - 4)`: its series to `w/3 + 2`
terms. -/
def twoAtanh (t : Rat) (w : Nat) : Rat := atanhTerms (floorTo (t * t) w) w (w / 3 + 2) 0 (floorTo t w) 0

/-- About `ln y` for `y > 0`, within about `2^-(w - 8)`; no bound rests on it. With `y = 2^j z`, `1/2 < z < 2`, it is
`j ln 2 + 2 atanh((z − 1)/(z + 1))`, `ln 2 = 2 atanh(1/3)`. -/
def lnApprox (y : Rat) (w : Nat) : Rat :=
  let j : Int := (y.num.natAbs.log2 : Int) - (y.den.log2 : Int)
  let z := if 0 ≤ j then y / 2 ^ j.toNat else y * 2 ^ (-j).toNat
  let w' := w + j.natAbs.log2 + 2
  (j : Rat) * twoAtanh (1 / 3) w' + twoAtanh ((z - 1) / (z + 1)) w'

/-- How many multiples of `2^-p` past `lnApprox`'s `lnUpper` and `lnLower` try before falling back. -/
def lnTries : Nat := 4

/-- The precision `lnUpper` and `lnLower` check a candidate at: `p`, and room for `y`'s scale below `1`, since `e^u ≈ y`
is bounded to a multiple of `2^-w`. -/
def lnPrec (y : Rat) (p : Nat) : Nat := p + 16 + (y.den.log2 - y.num.natAbs.log2)

/-- The first of `u, u + 2^-p, …` (`k` of them) with `y ≤ expLower u w`, so `ln y ≤ u`. -/
def firstAbove (y : Rat) (p w : Nat) : Nat → Rat → Option Rat
  | 0, _ => none
  | k + 1, u => if y ≤ expLower u w then some u else firstAbove y p w k (u + 1 / 2 ^ p)

/-- The first of `l, l − 2^-p, …` (`k` of them) with `expUpper l w ≤ y`, so `l ≤ ln y`. -/
def firstBelow (y : Rat) (p w : Nat) : Nat → Rat → Option Rat
  | 0, _ => none
  | k + 1, l => if expUpper l w ≤ y then some l else firstBelow y p w k (l - 1 / 2 ^ p)

/-- A multiple of `2^-p` at least `ln y`. At `y ≤ 0` it takes `Real.log`'s values: `ln 0 = 0` and `ln y = ln (−y)`. -/
def lnUpper (y : Rat) (p : Nat) : Rat :=
  if y = 0 then 0
  else
    let a := if 0 < y then y else -y
    let w := lnPrec a p
    (firstAbove a p w lnTries (ceilTo (lnApprox a w) p)).getD (ceilTo (a - 1) p)

/-- A multiple of `2^-p` at most `ln y`. At `y ≤ 0` it takes `Real.log`'s values: `ln 0 = 0` and `ln y = ln (−y)`. -/
def lnLower (y : Rat) (p : Nat) : Rat :=
  if y = 0 then 0
  else
    let a := if 0 < y then y else -y
    let w := lnPrec a p
    (firstBelow a p w lnTries (floorTo (lnApprox a w) p)).getD (floorTo (1 - 1 / a) p)

/-! ## Powers -/

/-- The precision `powUpper` and `powLower` bound `ln c` at: `p`, `|θ|`'s bits (its error is `θ` times `ln c`'s), and
`c^θ`'s integer bits. -/
def powPrec (c θ : Rat) (p : Nat) : Nat :=
  let a := if 0 ≤ θ then θ else -θ
  p + 8 + a.ceil.toNat.log2 + 2 * (θ * lnApprox c (p + 8)).ceil.toNat

/-- A multiple of `2^-p` at least `c^θ` (`Real.rpow`): `1` at `θ = 0`, `0` at `c = 0` (`0^θ` for every `θ ≠ 0`), else
`e^{θ ln |c|}` with `ln |c|` bounded above when `θ ≥ 0` and below when `θ < 0`. That is `|c|^θ`, at least
`c^θ = |c|^θ cos(πθ)` for `c < 0` too. -/
def powUpper (c θ : Rat) (p : Nat) : Rat :=
  if θ = 0 then 1
  else if c = 0 then 0
  else
    let a := if 0 < c then c else -c
    let q := powPrec a θ p
    expUpper (θ * if 0 ≤ θ then lnUpper a q else lnLower a q) p

/-- A multiple of `2^-p` at most `c^θ` (`Real.rpow`): `1` at `θ = 0`, `0` at `c = 0` (`0^θ` for every `θ ≠ 0`),
`−powUpper c θ p` for `c < 0`, where `c^θ = |c|^θ cos(πθ)`, else `e^{θ ln c}` with `ln c` bounded below when `θ ≥ 0`
and above when `θ < 0`. -/
def powLower (c θ : Rat) (p : Nat) : Rat :=
  if θ = 0 then 1
  else if c = 0 then 0
  else if c < 0 then -powUpper c θ p
  else
    let q := powPrec c θ p
    expLower (θ * if 0 ≤ θ then lnLower c q else lnUpper c q) p

#guard expLower 0 64 == 1 && expUpper 0 64 == 1
#guard expLower 1 64 < 2.718281828459045235360287 && 2.718281828459045235360287 < expUpper 1 64
#guard expUpper 1 64 - expLower 1 64 ≤ 1 / 2 ^ 63
#guard expLower (-1) 64 < 0.367879441171442321595523 && 0.367879441171442321595523 < expUpper (-1) 64
#guard expUpper (-1) 64 - expLower (-1) 64 ≤ 1 / 2 ^ 63
#guard expUpper 100 64 - expLower 100 64 ≤ 1 / 2 ^ 63
#guard expLower (-100) 64 == 0 && expUpper (-100) 64 == 1 / 2 ^ 64
#guard expUpper (1 / 3) 8 * 2 ^ 8 == (expUpper (1 / 3) 8 * 2 ^ 8).floor
#guard lnLower 1 64 == 0 && lnUpper 1 64 == 0
#guard lnLower 2 64 < 0.693147180559945309417232 && 0.693147180559945309417232 < lnUpper 2 64
#guard lnUpper 2 64 - lnLower 2 64 ≤ 1 / 2 ^ 62
#guard lnUpper (1 / 10 ^ 30) 64 - lnLower (1 / 10 ^ 30) 64 ≤ 1 / 2 ^ 62
#guard lnLower (10 ^ 30) 64 < 69.07755278982137052053974364 && 69.07755278982137052053974364 < lnUpper (10 ^ 30) 64
#guard lnUpper (10 ^ 30) 64 - lnLower (10 ^ 30) 64 ≤ 1 / 2 ^ 62
#guard powLower (1 / 2) 2 64 ≤ 1 / 4 && 1 / 4 ≤ powUpper (1 / 2) 2 64 &&
  powUpper (1 / 2) 2 64 - powLower (1 / 2) 2 64 ≤ 1 / 2 ^ 62
#guard powLower (1 / 2) (1 / 2) 64 < 0.707106781186547524400844 &&
  0.707106781186547524400844 < powUpper (1 / 2) (1 / 2) 64
#guard powLower 2 (-1) 64 ≤ 1 / 2 && 1 / 2 ≤ powUpper 2 (-1) 64 && powUpper 2 (-1) 64 - powLower 2 (-1) 64 ≤ 1 / 2 ^ 62
#guard powLower 0 3 64 == 0 && powUpper 0 3 64 == 0 && powLower 0 0 64 == 1 && powUpper (1 / 7) 0 64 == 1
#guard lnLower 0 64 == 0 && lnUpper 0 64 == 0
#guard lnLower (-2) 64 == lnLower 2 64 && lnUpper (-2) 64 == lnUpper 2 64
#guard powUpper (-2) (1 / 2) 64 == powUpper 2 (1 / 2) 64 && powLower (-2) (1 / 2) 64 == -powUpper 2 (1 / 2) 64

end Spot.Interval
