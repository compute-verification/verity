import Pous.Protocol.Model.ReadDraw
import Pous.Protocol.Accounting.Params

/-!
# The secure-erasure requirement with distinct reads

`EraseMeetsDistinct`: `EraseMeets` (`Pous/Protocol/Accounting/EraseMeets.lean`) with the `s` spot reads at distinct
indices (`Pous.distinctReads`, `Pous/Protocol/Model/ReadDraw.lean`), as `verity_pous.erasure.ErasureVerifier` draws them.
-/

namespace Pous

open scoped ENNReal

variable {M : IdealModel}

/-- **The secure-erasure requirement with distinct reads** at fraction `f`, read bound `(D, Q)`, `s` spot reads at
distinct indices and error `ε ≤ 2^-λ`, for a fill `C` that covers the prover's whole declared volatile memory of
`|C| = B·ℓ` bits:

- **full use:** every prover whose memory at each spot read's arrival is at most `⌊(1 − f)·|C|⌋` bits is accepted with
  probability at most `ε` (the sequential audit's bound over distinct reads, with the fill unbounded);
- **erasure:** every prover with `|C|` bits of memory that keeps foreign content of `S_φ ≥ f·|C| + 2λ` bits at every
  spot read's arrival wins with probability at most `ε`, whatever the digest of the fill check. So the verifier
  accepts a prover that keeps `f·|C| + 2λ` bits of uniform content, independent of the fill, the primitive and the
  challenges, through all the spot reads with probability at most `ε`. -/
def EraseMeetsDistinct (sch : Scheme M) (f : ℚ) (D Q s : ℕ) (ε : ℝ≥0∞) : Prop :=
  ε ≤ Params.εMax ∧
    AuditSecureDraw sch (distinctReads sch s) ⌊(1 - f) * ((sch.B * sch.ℓ : ℕ) : ℚ)⌋₊ D Q 0 ε ∧
    ∀ S_φ : ℕ, f * ((sch.B * sch.ℓ : ℕ) : ℚ) + 2 * (Params.secParam : ℚ) ≤ (S_φ : ℚ) →
      ∀ (K V : Type) [Fintype K] [Nonempty K] (dig : K → sch.Code → V),
        EraseSecureDraw sch (distinctReads sch s) dig (sch.B * sch.ℓ) S_φ D Q ε

end Pous
