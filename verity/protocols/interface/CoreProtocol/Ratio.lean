/-
Exact non-negative rationals, for probabilities and δ. Canonical JSON carries no non-integer number (`Flock.canon`
refuses one), so a ratio is written as a string in Python's `str(Fraction)` form: `"n/d"` in lowest terms, or `"n"` when
the denominator is 1.
-/
module

public import Flock.Bytes

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

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

/-- `s` as a natural number in canonical form: ASCII digits, with no leading zero. Read over the string's characters,
which the kernel evaluates (`String.all` and `String.toNat!` it doesn't). -/
private def natOf (s : String) (ctx : String) : V Nat :=
  let cs := s.toList
  if !cs.isEmpty && cs.all Char.isDigit && (cs.length = 1 || cs.head? != some '0') then
    .ok (cs.foldl (fun n c => n * 10 + (c.toNat - '0'.toNat)) 0)
  else .error s!"{ctx}: {s.quote} is not a natural number in canonical form"

/-- A string's characters cut at each `/`, as `String.splitOn "/"` cuts the string: the pieces between its `/`s in
order, the empty ones too, so one more piece than it has `/`s. `done` holds the pieces already cut and `piece` the
characters of the one being read, both in reverse. A walk over the characters, which the kernel evaluates
(`String.splitOn` it doesn't). -/
private def pieces (done : List String) (piece : List Char) : List Char → List String
  | [] => (String.ofList piece.reverse :: done).reverse
  | c :: cs => if c = '/' then pieces (String.ofList piece.reverse :: done) [] cs else pieces done (c :: piece) cs

/-- The inverse of `toString`: refuses a zero denominator, a denominator written as 1, and a ratio not in lowest terms,
so every ratio has exactly one spelling. -/
def Ratio.ofString (s : String) (ctx : String) : V Ratio := do
  match pieces [] [] s.toList with
  | [n] => pure ⟨← natOf n ctx, 1⟩
  | [n, d] =>
    let n ← natOf n ctx
    let d ← natOf d ctx
    if d ≤ 1 then throw s!"{ctx}: {s.quote} has denominator {d}"
    if Nat.gcd n d != 1 then throw s!"{ctx}: {s.quote} is not in lowest terms"
    pure ⟨n, d⟩
  | _ => throw s!"{ctx}: {s.quote} is not a ratio n/d"

/-! What `Ratio.ofString` gives, whole, on each of its branches and at each edge of a string's pieces, digits and
quoting: the ratio, or the refusal's message. -/

-- accepts a natural number, and a ratio in lowest terms whose denominator is above 1, at any number of digits
#guard Ratio.ofString "0" "x" matches .ok ⟨0, 1⟩
#guard Ratio.ofString "7" "x" matches .ok ⟨7, 1⟩
#guard Ratio.ofString "1/100" "x" matches .ok ⟨1, 100⟩
#guard Ratio.ofString "3/2" "x" matches .ok ⟨3, 2⟩
#guard Ratio.ofString "10/3" "x" matches .ok ⟨10, 3⟩
#guard Ratio.ofString "123456789012345678901234567890" "x" matches .ok ⟨123456789012345678901234567890, 1⟩
#guard Ratio.ofString "1000000000000000000001/1000000000000000000000" "x" matches
  .ok ⟨1000000000000000000001, 1000000000000000000000⟩
-- refuses a numerator that is not a natural number in canonical form, and quotes it
#guard Ratio.ofString "" "x" matches .error "x: \"\" is not a natural number in canonical form"
#guard Ratio.ofString "a" "x" matches .error "x: \"a\" is not a natural number in canonical form"
#guard Ratio.ofString "00" "x" matches .error "x: \"00\" is not a natural number in canonical form"
#guard Ratio.ofString "01/100" "x" matches .error "x: \"01\" is not a natural number in canonical form"
#guard Ratio.ofString "+1/2" "x" matches .error "x: \"+1\" is not a natural number in canonical form"
#guard Ratio.ofString "-1/2" "x" matches .error "x: \"-1\" is not a natural number in canonical form"
#guard Ratio.ofString " 1/2" "x" matches .error "x: \" 1\" is not a natural number in canonical form"
#guard Ratio.ofString "1 /2" "x" matches .error "x: \"1 \" is not a natural number in canonical form"
#guard Ratio.ofString "1_0/3" "x" matches .error "x: \"1_0\" is not a natural number in canonical form"
#guard Ratio.ofString "1.5" "x" matches .error "x: \"1.5\" is not a natural number in canonical form"
#guard Ratio.ofString "/" "x" matches .error "x: \"\" is not a natural number in canonical form"
#guard Ratio.ofString "/1" "x" matches .error "x: \"\" is not a natural number in canonical form"
#guard Ratio.ofString "/7" "x" matches .error "x: \"\" is not a natural number in canonical form"
#guard Ratio.ofString "a/b" "x" matches .error "x: \"a\" is not a natural number in canonical form"
#guard Ratio.ofString "01/0" "x" matches .error "x: \"01\" is not a natural number in canonical form"
-- refuses such a denominator
#guard Ratio.ofString "1/" "x" matches .error "x: \"\" is not a natural number in canonical form"
#guard Ratio.ofString "7/" "x" matches .error "x: \"\" is not a natural number in canonical form"
#guard Ratio.ofString "1/a" "x" matches .error "x: \"a\" is not a natural number in canonical form"
#guard Ratio.ofString "1/00" "x" matches .error "x: \"00\" is not a natural number in canonical form"
#guard Ratio.ofString "1/01" "x" matches .error "x: \"01\" is not a natural number in canonical form"
#guard Ratio.ofString "1/02" "x" matches .error "x: \"02\" is not a natural number in canonical form"
#guard Ratio.ofString "1/+2" "x" matches .error "x: \"+2\" is not a natural number in canonical form"
#guard Ratio.ofString "1/ 2" "x" matches .error "x: \" 2\" is not a natural number in canonical form"
#guard Ratio.ofString "1/2 " "x" matches .error "x: \"2 \" is not a natural number in canonical form"
#guard Ratio.ofString "1/1_0" "x" matches .error "x: \"1_0\" is not a natural number in canonical form"
-- refuses a denominator of 0 or 1
#guard Ratio.ofString "1/0" "x" matches .error "x: \"1/0\" has denominator 0"
#guard Ratio.ofString "0/0" "x" matches .error "x: \"0/0\" has denominator 0"
#guard Ratio.ofString "1/1" "x" matches .error "x: \"1/1\" has denominator 1"
#guard Ratio.ofString "0/1" "x" matches .error "x: \"0/1\" has denominator 1"
#guard Ratio.ofString "7/1" "x" matches .error "x: \"7/1\" has denominator 1"
-- refuses a ratio not in lowest terms
#guard Ratio.ofString "2/4" "x" matches .error "x: \"2/4\" is not in lowest terms"
#guard Ratio.ofString "0/5" "x" matches .error "x: \"0/5\" is not in lowest terms"
#guard Ratio.ofString "6/3" "x" matches .error "x: \"6/3\" is not in lowest terms"
#guard Ratio.ofString "10/100" "x" matches .error "x: \"10/100\" is not in lowest terms"
-- refuses more than one `/` before it reads a number
#guard Ratio.ofString "1/2/3" "x" matches .error "x: \"1/2/3\" is not a ratio n/d"
#guard Ratio.ofString "1//2" "x" matches .error "x: \"1//2\" is not a ratio n/d"
#guard Ratio.ofString "//" "x" matches .error "x: \"//\" is not a ratio n/d"
#guard Ratio.ofString "1/2/" "x" matches .error "x: \"1/2/\" is not a ratio n/d"
#guard Ratio.ofString "/1/2" "x" matches .error "x: \"/1/2\" is not a ratio n/d"
#guard Ratio.ofString "a/b/c" "x" matches .error "x: \"a/b/c\" is not a ratio n/d"
-- only ASCII's digits are digits, and a character of several bytes is one character: U+0661 and U+0662
-- (Arabic-Indic one and two), U+FF11 (fullwidth one), U+1D7D0 (mathematical bold two, four bytes), U+00BD (one half)
#guard Ratio.ofString "\u0661/2" "x" matches .error "x: \"\u0661\" is not a natural number in canonical form"
#guard Ratio.ofString "1/\u0662" "x" matches .error "x: \"\u0662\" is not a natural number in canonical form"
#guard Ratio.ofString "\uff11/2" "x" matches .error "x: \"\uff11\" is not a natural number in canonical form"
#guard Ratio.ofString "1/𝟐" "x" matches .error "x: \"𝟐\" is not a natural number in canonical form"
#guard Ratio.ofString "\u00bd" "x" matches .error "x: \"\u00bd\" is not a natural number in canonical form"
-- a refusal quotes the piece, or the whole string, as `String.quote` writes it
#guard Ratio.ofString "1/\"2\"" "x" matches .error "x: \"\\\"2\\\"\" is not a natural number in canonical form"
#guard Ratio.ofString "1\n/2" "x" matches .error "x: \"1\\n\" is not a natural number in canonical form"
#guard Ratio.ofString "1/2\t" "x" matches .error "x: \"2\\t\" is not a natural number in canonical form"
#guard Ratio.ofString "1\\2" "x" matches .error "x: \"1\\\\2\" is not a natural number in canonical form"
#guard Ratio.ofString "\x01/2" "x" matches .error "x: \"\\x01\" is not a natural number in canonical form"
#guard Ratio.ofString "1/2/\"" "x" matches .error "x: \"1/2/\\\"\" is not a ratio n/d"
-- and opens with `ctx` as it is given
#guard Ratio.ofString "2/4" "profile.delta" matches .error "profile.delta: \"2/4\" is not in lowest terms"
#guard Ratio.ofString "1/100" "profile.delta" matches .ok ⟨1, 100⟩
#guard Ratio.ofString "01" "levels[0].bernoulli" matches
  .error "levels[0].bernoulli: \"01\" is not a natural number in canonical form"
#guard Ratio.ofString "1/2/3" "" matches .error ": \"1/2/3\" is not a ratio n/d"

/-! The kernel evaluates `Ratio.ofString`: on a string it accepts, to the ratio; on one it refuses, as far as the
refusal and no further, since `String.quote` writes the message through functions no proof unfolds. `Except` has no
decidable equality, so `decide` reads it through `Except.toOption`. -/

example : Ratio.ofString "1/100" "x" = .ok ⟨1, 100⟩ := rfl
example : (Ratio.ofString "1/100" "x").toOption = some ⟨1, 100⟩ := by decide
example : (Ratio.ofString "2/4" "x").toOption = none := by decide

/-- `a ≤ b`, compared exactly. -/
def Ratio.le (a b : Ratio) : Bool := a.num * b.den ≤ b.num * a.den

end Core
