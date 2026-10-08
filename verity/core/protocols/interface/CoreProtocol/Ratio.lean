/-
Exact non-negative rationals, for probabilities and δ. Canonical JSON carries no non-integer number (`Flock.canon`
refuses one), so a ratio is written as a string in Python's `str(Fraction)` form: `"n/d"` in lowest terms, or `"n"` when
the denominator is 1.
-/
import Flock.Bytes

namespace Core

open Flock (V)

/-- `num / den` in lowest terms with `den > 0`; `Ratio.mk'` is the only way to build one from arbitrary integers. -/
structure Ratio where
  num : Nat
  den : Nat
  deriving BEq, DecidableEq, Repr, Inhabited

/-- `n / d` in lowest terms; `none` when `d = 0`. -/
def Ratio.mk' (n d : Nat) : Option Ratio :=
  if d = 0 then none else
    let g := Nat.gcd n d
    some ⟨n / g, d / g⟩

def Ratio.ofNat (n : Nat) : Ratio := ⟨n, 1⟩

def Ratio.toString (r : Ratio) : String :=
  if r.den = 1 then s!"{r.num}" else s!"{r.num}/{r.den}"

private def natOf (s : String) (ctx : String) : V Nat :=
  if !s.isEmpty && s.all Char.isDigit && (s.length = 1 || s.front != '0') then .ok s.toNat!
  else .error s!"{ctx}: {s.quote} is not a natural number in canonical form"

/-- The inverse of `toString`: refuses a zero denominator, a denominator written as 1, and a ratio not in lowest terms,
so every ratio has exactly one spelling. -/
def Ratio.ofString (s : String) (ctx : String) : V Ratio := do
  match s.splitOn "/" with
  | [n] => pure ⟨← natOf n ctx, 1⟩
  | [n, d] =>
    let n ← natOf n ctx
    let d ← natOf d ctx
    if d ≤ 1 then throw s!"{ctx}: {s.quote} has denominator {d}"
    if Nat.gcd n d != 1 then throw s!"{ctx}: {s.quote} is not in lowest terms"
    pure ⟨n, d⟩
  | _ => throw s!"{ctx}: {s.quote} is not a ratio n/d"

/-- `a ≤ b`, compared exactly. -/
def Ratio.le (a b : Ratio) : Bool := a.num * b.den ≤ b.num * a.den

end Core
