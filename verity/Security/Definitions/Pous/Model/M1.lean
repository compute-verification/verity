import Mathlib.Data.Fintype.Perm
import Mathlib.Logic.Equiv.Fin.Basic
import Definitions.Pous.Model.Scheme

/-!
# Model M1: independent ideal permutations, and its encoding

The pinned A2, A3′, A7′ and `M1Meets` statements are about this model (tradeoff §1.1). Moved verbatim from the targets
file, with `codeEquiv` from `Sanity.lean`.
-/

namespace Pous

/-- `B` blocks of `ℓ` bits as one string of `B·ℓ` bits. -/
def codeEquiv (B ℓ : ℕ) : (Fin B → Bits ℓ) ≃ Bits (B * ℓ) :=
  (Equiv.curry (Fin B) (Fin ℓ) Bool).symm.trans (Equiv.arrowCongr finProdFinEquiv (Equiv.refl Bool))

/-- Model M1 (tradeoff §1.1): `B` independent uniform permutations of `Bits ℓ`, forward queries only. -/
def perms (B ℓ : ℕ) : IdealModel where
  Ω := Fin B → Equiv.Perm (Bits ℓ)
  Qry := Fin B × Bits ℓ
  Ans := Bits ℓ
  answer ω q := ω q.1 q.2

/-- The M1 encoding: block `j` of `C` is `Π_j⁻¹(W_j)`, and decoding makes one forward query per block. No `pp` and
no coins; the encoder inverts the permutations, which the public cannot do cheaply. -/
def m1 (B ℓ : ℕ) [NeZero B] : Scheme (perms B ℓ) where
  n := B * ℓ
  t := 0
  B := B
  ℓ := ℓ
  B_pos := Nat.pos_of_ne_zero (NeZero.ne B)
  Coins := Unit
  enc W ω _ := (Fin.elim0, fun j => (ω j).symm ((codeEquiv B ℓ).symm W j))
  dec _ C f := codeEquiv B ℓ fun j => f (j, C j)

end Pous
