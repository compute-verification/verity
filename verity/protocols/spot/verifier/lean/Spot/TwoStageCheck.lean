module

public import Spot.FactorUpper

/-!
# A two-stage point, checked (`Spot.TwoStageCheck`)

A two-stage design's exfiltration count (`Claims.Sampling.CompiledTwo.Exfil`'s sum over the admissible sets, in its
sets' or patterns' form) is at most `δ^{−θ} ∏_ρ G_ρ(θ)` at every `θ ≥ 0` (`Claims.Sampling.AdmissibleTwoRankin` and its
patterns' form), and each replay block's factor is at most `e^v` for `v` from its cells (`Spot.FactorUpper`).
A point states its replay blocks as classes (`ReplayClass`): `count` replay blocks, each covered by the class's cells,
so the count is at most `e^L`, `L = θ ln(1/δ) + Σ_c count_c max(v_c, 0)`, and `checkTwoStage` returns `L/ln 2` from
above, the bits the point certifies (`Properties.Spot.TwoStageCheckSound`, `TwoStageCheckPatternsSound`). A
class that holds fewer replay blocks than its `count` only raises the bound, since each term is at least `0`.

Every number is exact: `ln` and `ln 2` are `Spot.Interval`'s bounds at precision `p`, and `θ` is the caller's.
`searchTheta` picks `θ` for the bound itself, by a Fibonacci search over a grid around a hint; the bound holds at every
`θ`, so the search is untrusted.
-/

@[expose] public section

namespace Spot.TwoStageCheck

open Spot.Interval
open Spot.FactorUpper

/-- Which of the counts a point bounds: the admissible sets' (`sets`), their patterns' over requests (`patterns`), or
the admissible sets' with each set's bits capped at what the requests reading it serve (`capped`). -/
inductive Form where
  | sets
  | patterns
  | capped
  deriving Repr, Inhabited, DecidableEq

/-- **A class of replay blocks**: `count` of them, each covered by `cells` (one per stratum); in the patterns' form each
read by at most `N` requests, each reading at most `reads[t]`'s `(count, bits)` in stratum `t`, and in the capped form
by requests serving at most `N` times the point's `S` bits together. -/
structure ReplayClass where
  count : Nat
  cells : List Cell
  reads : List (List (Nat × Nat))
  N : Nat
  deriving Repr, Inhabited

/-- **A two-stage point**: its form, the first stage's rate `q`, Rankin's `θ`, the confidence `δ`, the bits `S` each
request serves (the patterns' and capped forms), its classes of replay blocks and the precision `p`. -/
structure Point where
  form : Form
  q : Rat
  θ : Rat
  δ : Rat
  S : Nat
  classes : List ReplayClass
  p : Nat
  deriving Repr, Inhabited

/-- The factor bound's parameters: `q`, `θ`, `optimize.CHORDS`' breakpoints and `p`. -/
def Point.params (D : Point) : Params := ⟨D.q, D.θ, chords, D.p⟩

/-- At least `ln` of a replay block's factor in class `c`. -/
def classFactor (D : Point) (c : ReplayClass) : Option Rat :=
  match D.form with
  | .sets => lnFactorUpper c.cells D.params
  | .patterns => lnFactorUpperPatterns c.cells c.reads c.N D.S D.params
  | .capped => lnFactorUpperCapped c.cells (c.N * D.S) D.params

/-- `Σ_c count_c max(v_c, 0)` over the classes, `v_c = f c`; `none` when some `f c` is. -/
def classSum (f : ReplayClass → Option Rat) : List ReplayClass → Option Rat
  | [] => some 0
  | c :: cs =>
    match f c, classSum f cs with
    | some v, some s => some (c.count * max v 0 + s)
    | _, _ => none

/-- The point's numbers are in range: `0 ≤ q ≤ 1`, `θ ≥ 0`, `δ > 0`, and `ln 2`'s lower bound at `p` positive. -/
def Point.valid (D : Point) : Bool :=
  decide (0 ≤ D.q) && decide (D.q ≤ 1) && decide (0 ≤ D.θ) && decide (0 < D.δ) && decide (0 < lnLower 2 D.p)

/-- **At least `ln` of the count**, `θ ln(1/δ) + Σ_c count_c max(v_c, 0)`; `none` when the point is out of range or a
class's factor bound doesn't return. -/
def lnCount (D : Point) : Option Rat :=
  if D.valid then (classSum (classFactor D) D.classes).map (D.θ * lnUpper (1 / D.δ) D.p + ·) else none

/-- **The bits a two-stage point certifies**: `lnCount`'s `L` over `ln 2`, from above (`ln 2`'s lower bound at
`L ≥ 0`, its upper one below). -/
def checkTwoStage (D : Point) : Option Rat :=
  (lnCount D).map fun L => if 0 ≤ L then L / lnLower 2 D.p else L / lnUpper 2 D.p

/-! ## Searching `θ` -/

/-- The Fibonacci numbers, `fib 0 = 0`, `fib 1 = 1`. -/
def fib : Nat → Nat
  | 0 => 0
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

/-- `a ≤ b` with `none` above every value. -/
def le : Option Rat → Option Rat → Bool
  | some a, some b => decide (a ≤ b)
  | some _, none => true
  | none, _ => false

/-- **A Fibonacci search** for the least of `g` over the grid `[lo, lo + fib m]`, at `m ≥ 3`: its probes
`lo + fib (m − 2)` and `lo + fib (m − 1)` have values `g1` and `g2`, and the half beyond the greater probe goes, the
lesser probe becoming a probe of the rest. -/
def fibSearch (g : Nat → Option Rat) : Nat → Nat → Option Rat → Option Rat → Nat
  | n + 4, lo, g1, g2 =>
    if le g1 g2 then fibSearch g (n + 3) lo (g (lo + fib (n + 1))) g1
    else
      let lo' := lo + fib (n + 2)
      fibSearch g (n + 3) lo' g2 (g (lo' + fib (n + 2)))
  | _, lo, _, _ => lo + 1

/-- **`θ` for the bound itself**, `lnCount` evaluated at precision `ps`: the least of `0` and `θ₀ 2^j` for `|j| ≤ span`;
then, unless that is `0`, the least on a grid of `fib (n + 4)` steps from half of it to twice it. -/
def searchTheta (D : Point) (θ₀ : Rat) (span n ps : Nat) : Rat :=
  let at_ (θ : Rat) := lnCount { D with θ := θ, p := ps }
  let coarse := 0 :: if θ₀ ≤ 0 then [] else (List.range (2 * span + 1)).map fun i => θ₀ * 2 ^ i / 2 ^ span
  let best := ((coarse.map fun θ => (θ, at_ θ)).foldl (fun b x => if le x.2 b.2 then x else b) (0, none)).1
  if best = 0 then 0
  else
    let step := 3 * best / 2 / (fib (n + 4) : Rat)
    let g (i : Nat) := at_ (best / 2 + step * i)
    best / 2 + step * fibSearch g (n + 4) 0 (g (fib (n + 2))) (g (fib (n + 3)))

end Spot.TwoStageCheck
