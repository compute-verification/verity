module

/-!
# The planner's one-stage designs, checked (`Spot.PlanCheck`)

The sampling planner (`verity/core/protocols/spot/optimize.py`) picks designs in floating point. This module
checks a one-stage design in exact arithmetic and returns what it certifies at δ, or refuses it:

* `leakBound`: `units` units, each exporting at most `width` bits and sampled independently with probability
  `q = num / den`, and the count `K` it certifies. When a set of `K + 1` units is shown to escape with probability
  below δ (`Leak.rateCovers`: `q = 1`, or `e^{q (K + 1)} ≥ 1/δ`), it returns `width · K + locBitsUpper units K`, at least
  the exfiltration bound `width · K + log₂ ∑_{j ≤ K} C(units, j)` that a Bernoulli design certifies at δ
  (`Properties.Spot.LeakCheckSound`).
* `checkWork`: strata of units doing `work` each, sampled independently with probability `num / den`, meet the
  free-compute target `C` at δ when every stratum not sampled whole has `q · C ≥ λ · work` for one `λ` with `e^λ ≥ 1/δ`
  (`Properties.Spot.WorkCheckSound`).

`e^x ≥ 1/δ` is shown by a partial sum of the exponential series, which `e^x` exceeds for `x ≥ 0` (`expSum`), at `x`
capped where it is already enough (`lnCap`). `log₂(n/K)` is bounded by an exponent `a` with `n^(2^s) ≤ K^(2^s) · 2^a`
(`log2RatioUp`), and `log₂ e` by `ln 2 > 0.6931471803` (`log2eUp`).
-/

@[expose] public section

namespace Spot.PlanCheck

/-- `n!`. -/
def fact : Nat → Nat
  | 0 => 1
  | n + 1 => (n + 1) * fact n

/-- `∑ i < N, x^i / i!`, at most `e^x` for `x ≥ 0`. -/
def expSum (x : Rat) : Nat → Rat
  | 0 => 0
  | N + 1 => expSum x N + x ^ N / (fact N : Rat)

/-- An integer above `ln(1/δ)` for `0 < δ < 1`, `log₂ ⌊1/δ⌋ + 2`: `e^x` at `x` capped there still covers `1/δ`. -/
def lnCap (δ : Rat) : Nat := (δ.den / δ.num.toNat).log2 + 2

/-- `e^x ≥ 1/δ`, shown by `expSum` at `x` capped at `lnCap δ`, to `4 · lnCap δ + 128` terms. -/
def expCovers (x δ : Rat) : Bool :=
  let c := lnCap δ
  let y := min x (c : Rat)
  decide (0 < δ) && decide (0 ≤ y) && decide (1 ≤ expSum y (4 * c + 128) * δ)

/-- The precision `s` of `log2RatioUp`'s bound, `a / 2^s`. -/
def log2Prec : Nat := 12

/-- An `a` with `n^(2^s) ≤ K^(2^s) · 2^a` (`s = log2Prec`), so `log₂(n/K) ≤ a / 2^s`; `none` when neither guess holds. -/
def log2RatioUp (n K : Nat) : Option Nat :=
  let N := n ^ 2 ^ log2Prec
  let D := K ^ 2 ^ log2Prec
  let a := N.log2 - D.log2
  if N ≤ D * 2 ^ a then some a else if N ≤ D * 2 ^ (a + 1) then some (a + 1) else none

/-- At least `log₂ e = 1 / ln 2`, since `ln 2 > 0.6931471803`. -/
def log2eUp : Rat := 10000000000 / 6931471803

/-- At least `log₂ ∑_{j ≤ K} C(n, j)`, the location term: `0` at `K = 0`, `n` when `K ≥ n`, else the smaller of `n` and
`K · (log₂(n/K) + log₂ e)`, since `∑_{j ≤ K} C(n, j) ≤ (e · n / K)^K`. -/
def locBitsUpper (n K : Nat) : Rat :=
  if K = 0 then 0
  else if n ≤ K then n
  else match log2RatioUp n K with
    | some a => min (n : Rat) (K * ((a : Rat) / (2 ^ log2Prec : Nat) + log2eUp))
    | none => n

/-- A one-stage design against exfiltration: `units` units, each exporting at most `width` bits and sampled
independently with probability `num / den`, certifying at most `count` incorrect ones at `delta`. -/
structure Leak where
  units : Nat
  width : Nat
  num : Nat
  den : Nat
  count : Nat
  delta : Rat
  deriving Repr, Inhabited

/-- A set of `count + 1` units escapes with probability below `delta`: every unit is sampled (`num = den`), or
`delta < 1` and `e^{q (count + 1)} ≥ 1/delta`. -/
def Leak.rateCovers (d : Leak) : Bool :=
  if d.num = d.den then decide (0 < d.delta)
  else decide (d.delta < 1) && expCovers ((d.num : Rat) / (d.den : Rat) * ((d.count : Rat) + 1)) d.delta

/-- What the design certifies at `delta`, in bits: `width · count + locBitsUpper units count`; `none` when `den = 0`,
`num > den`, or a set of `count + 1` units isn't shown to escape below `delta`. -/
def leakBound (d : Leak) : Option Rat :=
  if d.den ≠ 0 ∧ d.num ≤ d.den ∧ d.rateCovers = true then
    some (((d.width * d.count : Nat) : Rat) + locBitsUpper d.units d.count)
  else none

/-- A stratum under a work target: each of its units does `work` and is sampled independently with probability
`num / den`. -/
structure Rate where
  work : Rat
  num : Nat
  deriving Repr, Inhabited

/-- A one-stage design against skipped work: its strata over one `den`, the target `target` and `delta`. -/
structure Work where
  strata : List Rate
  den : Nat
  target : Rat
  delta : Rat
  deriving Repr, Inhabited

/-- The least `q · C / work` over the strata not sampled whole that do work, `none` when there are none: each such
stratum's units have `q ≥ λ · work / C`. -/
def lamOf (den : Nat) (C : Rat) : List Rate → Option Rat
  | [] => none
  | r :: rs =>
    if r.num < den ∧ 0 < r.work then
      let v := (r.num : Rat) / (den : Rat) * C / r.work
      some (match lamOf den C rs with
        | some a => min a v
        | none => v)
    else lamOf den C rs

/-- The design meets the target at `delta`: `den > 0`, `target > 0`, `0 < delta < 1`, each stratum has `num ≤ den` and
`work ≥ 0`, and `e^λ ≥ 1/delta` at `λ = lamOf` (nothing to show when every stratum doing work is sampled whole). -/
def checkWork (d : Work) : Bool :=
  decide (d.den ≠ 0) && decide (0 < d.target) && decide (0 < d.delta) && decide (d.delta < 1) &&
    d.strata.all (fun r => decide (r.num ≤ d.den) && decide (0 ≤ r.work)) &&
    match lamOf d.den d.target d.strata with
    | some l => expCovers l d.delta
    | none => true

#guard fact 5 == 120
#guard expSum 1 20 < 2.7182818286 && 2.718281828 < expSum 1 20
#guard expCovers 5 (1 / 100) && !expCovers (46 / 10) (1 / 100) && expCovers (4606 / 1000) (1 / 100)
#guard log2RatioUp 1024 1 == some (10 * 2 ^ 12)
#guard locBitsUpper 10 0 == 0 && locBitsUpper 10 12 == 10 && locBitsUpper 3 1 == 3
#guard locBitsUpper 1000000 1000 < 11410 && 11408 < locBitsUpper 1000000 1000
#guard leakBound { units := 1000, width := 8, num := 1, den := 2, count := 9, delta := 1 / 100 } |>.isSome
#guard leakBound { units := 1000, width := 8, num := 1, den := 2, count := 8, delta := 1 / 100 } |>.isNone
#guard leakBound { units := 10, width := 8, num := 3, den := 3, count := 0, delta := 1 / 100 } == some 0
#guard checkWork { strata := [⟨1, 1⟩, ⟨2, 2⟩], den := 1000, target := 4606, delta := 1 / 100 }
#guard !checkWork { strata := [⟨1, 1⟩, ⟨2, 1⟩], den := 1000, target := 4606, delta := 1 / 100 }
#guard checkWork { strata := [⟨5, 7⟩, ⟨0, 0⟩], den := 7, target := 1, delta := 1 / 100 }

end Spot.PlanCheck
