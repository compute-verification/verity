import Definitions.Pous.Model.SecureErasure
import Definitions.Pous.Accounting.Params

/-!
# The secure-erasure requirement

`EraseMeets`: the fill covers the prover's whole declared volatile memory, `|C| = B·ℓ` bits, and at most a fraction `f`
of it can keep anything but `C` through all the spot reads. Added on 4 Oct 2026, with `Pous.Guarantees.SecureErasureMeetsFamily`
(`Security/Definitions/Pous/Model/SecureErasure.lean` is the game).
-/

namespace Pous

open scoped ENNReal

variable {M : IdealModel}

/-- **The secure-erasure requirement** at fraction `f`, read bound `(D, Q)`, `s` spot reads and error `ε ≤ 2^-λ`, for a
fill `C` that covers the prover's whole declared volatile memory of `|C| = B·ℓ` bits:

- **full use:** every prover whose memory at each spot read's arrival is at most `⌊(1 − f)·|C|⌋` bits is accepted with
  probability at most `ε` (the sequential audit's bound, with the fill unbounded);
- **erasure:** every prover with `|C|` bits of memory that keeps foreign content of `S_φ ≥ f·|C| + 2λ` bits through the
  spot reads wins with probability at most `ε`, whatever the digest of the fill check. So the memory challenger accepts
  a prover that keeps `f·|C| + 2λ` bits of uniform content, independent of the fill, the primitive and the challenges,
  through all the spot reads with probability at most `ε`. -/
def EraseMeets (sch : Scheme M) (f : ℚ) (D Q s : ℕ) (ε : ℝ≥0∞) : Prop :=
  ε ≤ Params.εMax ∧
    AuditSecure sch .sequential ⌊(1 - f) * ((sch.B * sch.ℓ : ℕ) : ℚ)⌋₊ D Q s 0 ε ∧
    ∀ S_φ : ℕ, f * ((sch.B * sch.ℓ : ℕ) : ℚ) + 2 * (Params.secParam : ℚ) ≤ (S_φ : ℚ) →
      ∀ (K V : Type) [Fintype K] [Nonempty K] (dig : K → sch.Code → V),
        EraseSecure sch dig (sch.B * sch.ℓ) S_φ D Q s ε

end Pous
