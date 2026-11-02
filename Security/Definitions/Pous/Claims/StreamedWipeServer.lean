import Definitions.Pous.Model.StreamedWipeServer

/-!
# The streamed-label wipe per server, as claims of an acceptance rule

The statements of `Security/Definitions/Pous/Guarantees/StreamedWipeServer.lean` with the memory challenger's acceptance
rule a parameter `V` (`Pous.StreamedWipe.Acceptance`). Each guarantee there is its claim at the wipe's own acceptance
rule, `StreamedWipe.Accepts`. The lock lists, for each claim, a `fails` theorem: the claim is false at
`StreamedWipe.acceptAll`, the acceptance rule that accepts everything, since a prover inside the physical model that
keeps all its prior content is then accepted with probability 1. It also lists a `meets` theorem: the claim's numeric
antecedents hold at the deployed, measured `StreamedWipe.h200`. The PRF term in each is the advantage of
`StreamedWipe.wipeDistinguisherBy V P`, a distinguisher built from the prover `P` (at `V = Accepts`,
`StreamedWipe.wipeDistinguisher P`).
-/

namespace Pous.Claims

open Pous.StreamedWipe
open scoped ENNReal

/-- `Pous.Guarantees.StreamedWipeMeetsReadsTight` under the acceptance rule `V`: for every `r ≥ 1` and `k` with
`(r − 1)^k · 2^129 ≤ r^k`, every fill of `n ≥ 1` labels on a server of `M = n · 512 + R` bits, and every prover with
`E` bits of egress, a prover that keeps at least `fillShare n r k + R + E` bits of uniform prior content is accepted by
`V` with probability at most `2^-128` plus the PRF's advantage against the distinguisher built from it. -/
def StreamedWipeMeetsReadsTight (V : Acceptance) : Prop :=
  ∀ r k : ℕ, 0 < r → (r - 1) ^ k * 2 ^ 129 ≤ r ^ k →
    ∀ n : ℕ, 0 < n →
      ∀ (R q E κ : ℕ) (prf : Bits κ → Fin n → Bits 512) (P : Prover n k 512 q (n * 512 + R) E),
        fillShare n r k + R + E ≤ q →
        winRealBy V prf P ≤ (2⁻¹ : ℝ≥0∞) ^ 128 + prfAdvantage prf (wipeDistinguisherBy V P)

/-- `Pous.Guarantees.StreamedWipeMeetsServer` under the acceptance rule `V`: on a server of assumed memory capacity
`M_s = 512 · n + R` over every tier, at every `r ≥ 1` and `k` with `(r − 1)^k · 2^129 ≤ r^k`, a prover that keeps at
least `B(n, r, k, R) + E` bits of uniform prior content (`serverBound`) is accepted by `V` with probability at most
`2^-128` plus the PRF's advantage against the distinguisher built from it. -/
def StreamedWipeMeetsServer (V : Acceptance) : Prop :=
  ∀ Ms n R : ℕ, Ms = 512 * n + R → 0 < n →
    ∀ r k : ℕ, 0 < r → (r - 1) ^ k * 2 ^ 129 ≤ r ^ k →
      ∀ (q E κ : ℕ) (prf : Bits κ → Fin n → Bits 512) (P : Prover n k 512 q Ms E),
        serverBound n r k R + E ≤ q →
        winRealBy V prf P ≤ (2⁻¹ : ℝ≥0∞) ^ 128 + prfAdvantage prf (wipeDistinguisherBy V P)

/-- `Pous.Guarantees.StreamedWipeMeetsServer5GB` under the acceptance rule `V`: `StreamedWipeMeetsServer V` at a ratio
`r` and read count `k` at the 5 GB point (`OperatingPoint`, `B(n, r, k, R) ≤ 4·10^10`), for a prover that keeps at least
`4·10^10 + E` bits. -/
def StreamedWipeMeetsServer5GB (V : Acceptance) : Prop :=
  ∀ Ms n R : ℕ, Ms = 512 * n + R → 0 < n →
    ∀ r k : ℕ, 0 < r → (r - 1) ^ k * 2 ^ 129 ≤ r ^ k → OperatingPoint n r k R →
      ∀ (q E κ : ℕ) (prf : Bits κ → Fin n → Bits 512) (P : Prover n k 512 q Ms E),
        40000000000 + E ≤ q →
        winRealBy V prf P ≤ (2⁻¹ : ℝ≥0∞) ^ 128 + prfAdvantage prf (wipeDistinguisherBy V P)

/-- `Pous.Guarantees.StreamedWipeMeetsServerH200` under the acceptance rule `V`: at the deployed H200 (`h200`, as
measured on the deployed H200: `n = 2,319,974,400` labels, `M_s = 1,206,038,560,768` bits, `k = 5500` reads), a prover
with `E` bits of egress that keeps at least `4·10^10 + E` bits of uniform prior content is accepted by `V` with
probability at most `2^-128` plus the PRF's advantage against the distinguisher built from it. -/
def StreamedWipeMeetsServerH200 (V : Acceptance) : Prop :=
  ∀ (q E κ : ℕ) (prf : Bits κ → Fin h200.n → Bits 512) (P : Prover h200.n h200.k 512 q h200.Ms E),
    40000000000 + E ≤ q →
    winRealBy V prf P ≤ (2⁻¹ : ℝ≥0∞) ^ 128 + prfAdvantage prf (wipeDistinguisherBy V P)

end Pous.Claims
