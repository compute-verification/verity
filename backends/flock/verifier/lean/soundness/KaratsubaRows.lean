import FlockSoundness.Discharge.Recursion.Karatsuba

/-!
# The Karatsuba gadget's rows, against the Definition's

A test the audit runs (`runs`), not a proof. `sound_mul128` is about the rows `mul128` builds, and C-Flock's
`Gf128Mul_v1` has to lower to the same rows for the lemma to be about V*'s gadget. Both sides pin one digest: the SHA-512
of the rows in build order, each `[out, a's variables, a's constant, b's variables, b's constant]`, with `a`'s inputs
numbered 0–127, `b`'s 128–255 and the rows from 256, as compact JSON (`KARATSUBA_ROWS` in
`backends/flock/tests/test_rec_algebra.py`, on the Definition's recursion). This prints the digest of `mul128`'s rows
and fails unless it is that one.
-/

namespace KaratsubaRows

open Flock Flock.HmNets FlockSoundness.Discharge.Recursion.Karatsuba

def natsJson (l : List Nat) : String := "[" ++ ",".intercalate (l.map toString) ++ "]"

def rowJson (r : Row) : String :=
  s!"[{r.out},{natsJson r.a.s},{if r.a.c then 1 else 0},{natsJson r.b.s},{if r.b.c then 1 else 0}]"

def expected : String :=
  "e6b8a513843b112aefe5b265144cd01747a443ffb42eba34e37e640bd604c816" ++
  "f82db680ff40c5d48e76304d5a7a5df406bb94309540a2eec5cb07f435333634"

end KaratsubaRows

open Flock Flock.HmNets FlockSoundness.Discharge.Recursion.Karatsuba KaratsubaRows in
def main : IO UInt32 := do
  let a := (List.range 128).map Lin.var
  let b := (List.range 128).map fun i => Lin.var (128 + i)
  let (out, st) := (mul128 a b).run ⟨256, []⟩
  let rows := st.rows.reverse
  let got := hexOf (Sha512.hash ("[" ++ ",".intercalate (rows.map rowJson) ++ "]").toUTF8)
  IO.println s!"mul128: {rows.length} rows, {out.length} outputs, SHA-512 {got}"
  if rows.length == 2187 && out.length == 128 && got == expected then return 0
  IO.eprintln s!"KaratsubaRows: mul128's rows are not the Definition's (expected {expected})"
  return 1
