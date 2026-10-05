import FlockSoundness.Protocol.Defs
import Mathlib.Algebra.MvPolynomial.SchwartzZippel
import Mathlib.Algebra.Polynomial.Roots

/-!
# The opening phase: ring switching and batching (PROTOCOL.md §12)

From a doomed start (for every codeword near the committed table, some claim handed to the
opening is false), the opening ends doomed (no codeword near the table has the batched value)
except with probability `L · (7·n + 1)/|F|`, for `n` claims and `L` codewords (`opening_phase_of_card`).

The argument, per codeword `M` with witness `z` (paper Appendix B):

* **The honest message.** For a claim with point `x = (x₀, x')`, the honest ring-switching
  message is `ŝ[v] = Σ_y eq(x', y)·z[128y + v]`, and the claim's value is
  `Σ_v w[v]·ŝ[v]` (`claim_eval_eq`). So the prover passes the check on a false claim only with a
  message `s ≠ ŝ`.
* **Ring switching.** The verifier's recombination of the honest message is the inner product
  of the packed table with `Φ_{r''}(eq(x', ·))` (`recombine_shat`), the quantity the batched
  claim `Σ_y f(y)·b(y) = T` checks.
* **One claim.** For `s ≠ ŝ`, `recombine(s, r'') − recombine(ŝ, r'')` is a nonzero multilinear
  polynomial in the seven coins `r''` (`recombine_poly`), so it vanishes with probability at most
  `7/|F|` (`prCoin_recombine_le`).
* **The batch.** A nonzero linear form in the batching coins vanishes with probability at most
  `1/|F|` (`prCoin_batch_le`).
-/

namespace FlockSoundness

open Game Model Accounting
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]

/-! ## Equality weights at Boolean points -/

/-! ## Multilinear polynomials in the equality basis -/


/-! ## The honest ring-switching message -/

/-- The point beyond the packed seven address bits: `x[1 … m−7]`. -/
def Model.Claim.tail {m : ℕ} (c : Claim F m) (hm : 7 ≤ m) : Fin (m - 7) → F :=
  fun j => c.x ⟨j.val + 1, by omega⟩


/-! ## Packing is a bijection of additive groups -/

section Packing

variable (A : Arith F K) (hA : A.Correct)
include hA

end Packing

/-! ## Ring switching -/


/-! ## The opening as a loop over claims, then a batch -/

/-- One claim's ring switch (O3, O5): the prover's message, then seven coins. -/
def opStep {m : ℕ} (c : Claim F m) : Game (Claim F m × (Fin 128 → F) × (Fin 7 → F)) := do
  let s ← recv (Fin 128 → F)
  let r'' ← draw (Fin 7 → F)
  pure (c, s, r'')

/-- The opening's verdict from the switched claims and the batching coins (O6). -/
def opResult (A : Arith F K) {m : ℕ} (hm : 7 ≤ m) (claims : List (Claim F m))
    (switched : List (Claim F m × (Fin 128 → F) × (Fin 7 → F))) (γ : Fin claims.length → F) :
    Option (OpenOut F) :=
  let ok := switched.all fun (c, s, _) => decide ((∑ v, packedWeight c hm v * s v) = c.value)
  let T := ∑ k : Fin claims.length, γ k *
    recombine A (switched.getD k.val (claims.getD k.val ⟨0, 0, 0⟩, 0, 0)).2.1
      (switched.getD k.val (claims.getD k.val ⟨0, 0, 0⟩, 0, 0)).2.2
  let b : ℕ → F := fun y => ∑ k : Fin claims.length, γ k *
    let e := switched.getD k.val (claims.getD k.val ⟨0, 0, 0⟩, 0, 0)
    phi A e.2.2 (eqAt (fun j : Fin (m - 7) => e.1.x ⟨j.val + 1, by omega⟩) y)
  if ok then some ⟨T, b⟩ else none

/-! ## One codeword -/

/-! ## The phase -/

end FlockSoundness
