import Mathlib.Algebra.BigOperators.Fin
import Definitions.Pous.Model.M1

/-!
# Model M1 on P2's exact domain `[0, p)`

P2 permutes `Z_p` with `p = 2^w − c₀`, not `Bits w`. `permsp B w p` draws `B` independent uniform permutations of
`Fin p`. They act on `w`-bit strings through the numeric value (bit `i` has weight `2^i`): values `< p` move, and
values `≥ p` (the tail `[p, 2^w)`) are fixed. Queries and answers are `w`-bit strings, so `Scheme`, the audit game and
`Meets` apply unchanged.

`m1p B w p b` carries `b` payload bits per block. They are zero-extended to `w` bits, so the value is `< 2^b`, and it
lies in `[0, p)` whenever `2^b ≤ p`. Block `j` of the code is `Π_j⁻¹` of the extended payload; the decoder makes one
forward query per block and reduces the answer mod `2^b`. The definitions are total. The side conditions
(`2^b ≤ p ≤ 2^w`, `b ≤ w`, `20·w ≤ 21·b`) are hypotheses of the theorems that need them.
-/

namespace Pous

/-- `w` bits as a number in `[0, 2^w)`, bit `i` with weight `2^i`. -/
def bitsEquiv (w : ℕ) : Bits w ≃ Fin (2 ^ w) :=
  (Equiv.arrowCongr (Equiv.refl (Fin w)) finTwoEquiv.symm).trans finFunctionFinEquiv

/-- The numeric value of a `w`-bit string. -/
def bitsVal {w : ℕ} (x : Bits w) : ℕ := (bitsEquiv w x : ℕ)

/-- The `w`-bit string of `n mod 2^w`. -/
def bitsOfNat (w n : ℕ) : Bits w := (bitsEquiv w).symm ⟨n % 2 ^ w, Nat.mod_lt _ (Nat.two_pow_pos w)⟩

/-- A permutation of `[0, p)` acting on `w`-bit strings: values `< p` move, and values `≥ p` are fixed. -/
def permAct {w p : ℕ} (π : Equiv.Perm (Fin p)) (x : Bits w) : Bits w :=
  if h : bitsVal x < p then bitsOfNat w (π ⟨bitsVal x, h⟩) else x

/-- Model M1 on `[0, p)`: `B` independent uniform permutations of `Fin p`, forward queries only, acting on `Bits w`. -/
def permsp (B w p : ℕ) : IdealModel where
  Ω := Fin B → Equiv.Perm (Fin p)
  Qry := Fin B × Bits w
  Ans := Bits w
  answer ω q := permAct (ω q.1) q.2

/-- The M1 encoding on `[0, p)` with `b` payload bits per block. `W` has `B·b` bits; block `j` of `C` is `Π_j⁻¹` of
`W_j` zero-extended to `w` bits, and decoding makes one forward query per block and takes the answer mod `2^b`. No
`pp` and no coins. -/
def m1p (B w p b : ℕ) [NeZero B] : Scheme (permsp B w p) where
  n := B * b
  t := 0
  B := B
  ℓ := w
  B_pos := Nat.pos_of_ne_zero (NeZero.ne B)
  Coins := Unit
  enc W ω _ := (Fin.elim0, fun j => permAct (ω j).symm (bitsOfNat w (bitsVal ((codeEquiv B b).symm W j))))
  dec _ C f := codeEquiv B b fun j => bitsOfNat b (bitsVal (f (j, C j)))

end Pous
