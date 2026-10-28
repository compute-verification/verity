import Definitions.Pous.Model.StreamedWipeDisk

/-!
# The streamed-label wipe with a server's disks in the fill, as claims of a verdict

The statements of `Security/Definitions/Pous/Guarantees/StreamedWipeDisk.lean` with the memory challenger's verdict a
parameter `V` (`Pous.StreamedWipe.Verdict`), as `Security/Definitions/Pous/Claims/StreamedWipeServer.lean` states the
wipe per server. Each guarantee there is its claim at the wipe's own verdict, `StreamedWipe.Accepts`. The lock lists,
for each claim, a `fails` theorem (the claim is false at `StreamedWipe.acceptAll`, where a prover inside the physical
model that keeps all its prior content is accepted with probability 1) and a `meets` theorem (the claim's numeric
antecedents hold at the deployed session as measured: vy-dc-h200-2's card, `StreamedWipe.h200`, with its disks,
`StreamedWipe.h200Disks`).
-/

namespace Pous.Claims

open Pous.StreamedWipe
open scoped ENNReal

/-- `Pous.Guarantees.StreamedWipeMeetsDisk` under the verdict `V`: a server whose tiers besides its disks hold `n`
labels of the fill in `M` bits (`512·n ≤ M`), and whose disks hold `D` bytes of which the fill covers `f` labels
(`64·f ≤ D`), wiped in one session of `n + f` labels on `M_s = M + 8·D` bits. At every `r ≥ 1` and `k` with
`(r − 1)^k · 2^129 ≤ r^k`, a prover with `E` bits of egress that keeps at least
`serverBound (n + f) r k ((M − 512·n) + (8·D − 512·f)) + E` bits of uniform prior content is accepted by `V` with
probability at most `2^-128` plus the PRF's advantage against the distinguisher built from it. -/
def StreamedWipeMeetsDisk (V : Verdict) : Prop :=
  ∀ M n D f : ℕ, 512 * n ≤ M → 64 * f ≤ D → 0 < n + f →
    ∀ r k : ℕ, 0 < r → (r - 1) ^ k * 2 ^ 129 ≤ r ^ k →
      ∀ (q E κ : ℕ) (prf : Bits κ → Fin (n + f) → Bits 512) (P : Prover (n + f) k 512 q (M + 8 * D) E),
        serverBound (n + f) r k (M - 512 * n + (8 * D - 512 * f)) + E ≤ q →
        winRealBy V prf P ≤ (2⁻¹ : ℝ≥0∞) ^ 128 + prfAdvantage prf (wipeDistinguisherBy V P)

end Pous.Claims
