import Mathlib.LinearAlgebra.Lagrange
import Mathlib.Algebra.MvPolynomial.SchwartzZippel
import Mathlib.Algebra.Polynomial.Roots
import FlockSoundness.Protocol.Defs

/-!
# The lincheck phase (PROTOCOL.md §10)

For one codeword near the committed table, whose witness `z` makes the `c` claim and the region
claims true: the doomed zerocheck says the `a` claim, the `b` claim or the pin fails. Then the
lincheck ends with a true `ab` claim for `z` with probability at most `(2 + 2(k_log − 6) + 63)/|F|`
(`value_lincheck_le`):

* **α, β (L1, L2).** The lincheck starts from `α·v_a + v_b + β`; the truth is
  `α·a(z) + b(z) + β·pin(z)`, which is `Σ_c comb[c]·h[c]` for the witness's block-folded columns
  `h` (`combSum_eq`). The difference is a nonzero polynomial of degree 1 in `(α, β)`: one coin each.
* **The rounds (L3).** Each keeps a false claim false except with probability `2/|F|`
  (`Rounds.lincheckRound_sound`), about the vectors folded so far (`value_lincheckRounds_le`).
* **The final check and `r'` (L4, L5).** A passing check forces `z_partial` off the folded columns;
  the `ab` claim's value `Σ_s L_s(r')·z_partial[s]` then misses the truth `Σ_s L_s(r')·h'[s]`
  (`abEval_eq`) except with probability `63/|F|`, a nonzero polynomial of degree below 64 through
  the skip nodes (`prCoin_lagS_le`).

`lincheck_phase_of_card` takes the union over a list of at most `L` codewords.
-/

namespace FlockSoundness

open Game Model Accounting Polynomial
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]

/-! ## Equality weights and sums over split indices -/

/-! ## Two coin bounds -/

/-! ## The rounds -/

/-! ## The witness through the lincheck's eyes -/

/-- The witness's block-folded columns: `h[c] = Σ_o eq(ρ_out, o)·ω[o·2^k_log + c]`. -/
def hcol (S : Statement) (ρOut : Fin S.nbl → F) (ω : Witness S.m) : ℕ → F := fun c =>
  if hc : c < 2 ^ S.kLog then
    ∑ o : Fin (2 ^ S.nbl), eqAt ρOut o.val * ((ω (S.addr o ⟨c, hc⟩)).val : F)
  else 0

/-! ## The phase -/

end FlockSoundness
