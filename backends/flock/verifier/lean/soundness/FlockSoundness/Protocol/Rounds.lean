import FlockSoundness.Model.Piop
import FlockSoundness.Model.Ligerito
import Mathlib.Algebra.CharP.Defs
import Mathlib.Algebra.MvPolynomial.SchwartzZippel
import Mathlib.Algebra.Polynomial.Roots

/-!
# One sumcheck round

The spec encodes a round's quadratic in three ways (PROTOCOL.md §9.2 Z6, §10 L3, §13.2). In each,
the verifier derives a quadratic from the prover's message and the running claim, and the new
claim is its value at the fresh coin. If the running claim is false, the derived quadratic
differs from the true one whatever the prover sends, so the new claim is false except with
probability `2/|F|` (`Game.prCoin_eval_eq_eval_le`). These are the round-by-round lemmas the
zerocheck, lincheck and Ligerito phases of `rep_sound` chain; a Fiat–Shamir proof would reuse
them unchanged (`DESIGN.md` §8).

All three identities use characteristic 2 (`1 + 1 = 0`).
-/

namespace FlockSoundness.Rounds

open Polynomial Game Model

open scoped ENNReal

variable {F : Type} [Field F] [Fintype F] [DecidableEq F] [CharP F 2]

omit [Fintype F] [DecidableEq F] in
theorem two_eq_zero : (2 : F) = 0 := by exact_mod_cast CharP.cast_eq_zero F 2

omit [Fintype F] [DecidableEq F] in
theorem add_self (a : F) : a + a = 0 := by
  rw [← two_mul, two_eq_zero, zero_mul]

/-! ## The residual round of the zerocheck (Z6) -/

/-! ## Rounds that impose `q(0) + q(1) = claim` (L3 and Ligerito) -/

end FlockSoundness.Rounds
