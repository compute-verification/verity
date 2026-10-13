import Definitions.Accounting.RateAuditDeployed
import Models.Accounting.Assumptions.RateAudit
import Models.Accounting.Assumptions.RateAuditDeployed

/-!
# The remote bandwidth audit as deployed: conservation over a session (trusted)

One guarantee from "the verifier accepts a session of `n` chained windows" to "the session conserves its words, at
`γ` no more than the sum of its named terms", over the deployed game (`Definitions/Accounting/RateAuditDeployed.lean`):
partial completion, interactive side sources, memory tiers, chained passes whose roots the prover grinds, and one
pooled draw a window. Proofs are in `Security/Proofs/Accounting/RateAudit/Deployed.lean`.

* `RemoteSessionConservation`: at every `d̄`, `d_w` and every parameter.
* `RemoteSessionConservationDeployed`: at the deployed parameters (masked `p = 2^1024 − 105` at `k = 3` rounds, 4 KiB
  blocks, SHA-512 receipts, 2,773 openings a window, `d̄ = 0.15%`, `d_w = 1%`, at most `2^55` compressions), where
  the draw's two terms are each at most `2^-40` from seven windows on.

It rests on the random oracle (the game's), and on two named assumptions: `Accounting.Assumptions.WorkHardness`
(completing `x` of a word's `b` bits costs `x/b` of `rounds · c_root`) and `Accounting.Assumptions.SquaringCeiling`
(a pass's completions cost at most the node's measured squaring capacity `F`). The other measured capacities (the
on-chip state `S`, the link `L`, the tiers) are the game's bounds, as in every guarantee it composes.

The receipt here is the oracle's tag of the whole block; the deployed receipt hashes a 16-row sketch of it, and the
step from one to the other is not proved (the PR's gap list).
-/

namespace Accounting.Guarantees

open Accounting Accounting.Rate
open Pous (Bits pr)
open scoped ENNReal

/-- **The remote bandwidth audit as deployed: an accepted session conserves its words, over the horizon and in every
window.** For every deployed session of `n` windows on one store (`Rate.DeployedSession`: in each window `P` chained
passes whose `G` candidate roots the prover grinds, pass devices with partial completion, interactive side sources that
ignore the oracle, at most `S` bits on chip, `L` over the link and `q` oracle queries a pass, and one pooled draw of `d`
of the window's `P·N` receipts), every split of device memory into `τ` tiers, and every `d̄` and `d_w ≤ 1`: if
completions are as hard as `WorkHardness b rounds c_root cost` says and every pass's completions fit the squaring
ceiling `F` (`SquaringCeiling cost F`), then the probability that every opening of every window passes while either the
session's tier reads fall short of `(1 − γ̄)` of its `n·P·N·b` bits,

  `∑_w reads_w < (1 − γ̄)·n·P·N·b`,   `γ̄ = d̄ + μ_chip + μ_link + μ_rebuild + μ_λ`,

or some window's fall short of `(1 − γ_w)` of its `P·N·b`,

  `reads_w < (1 − γ_w)·P·N·b`,   `γ_w = d_w + μ_chip + μ_link + μ_rebuild + μ_λ`,

is at most

  `(1 − d̄)^(d·n) + (1 − d_w/n)^(d·n) + n·P·G·(N/2^h + C(q, 2)/2^h + 2^N/2^λ)`.

The terms of `γ` are fractions of a pass's `N·b` bits (`Rate.Deployed`): `μ_chip = S/(N·b)` the on-chip state,
`μ_link = (L − ρ)/(N·b)` the link beyond the certified stream, `μ_rebuild = (b·F/(rounds·c_root))/(N·b)` the bits a
pass can complete within its ceiling, and `μ_λ = λ/(N·b)` the compression slack.

**The reading.** Every way to a receipt's block other than reading it from a tier in the pass is one term: holding it
on chip, taking it over the link, completing it (whole or in part), or the slack the compression leaves; the draw's
shortfall is `d̄` over the horizon and `d_w` in each window. The prover is one resource: nothing here says which
device, machine or party reads, completes or holds a word. Reads are counted over the tiers, so `R·T` is their sum's
capacity; the term `G` is the grinding the Fiat–Shamir seed allows.

**The proof** (`Accounting.SecurityProofs.RemoteSessionConservation`): each pass device is simulated by an interactive
device of `RatePaidNamedRO`'s class whose side sources also answer the completed bits, so its link carries
`L + b·F/(rounds·c_root)` (`WorkHardness` and `SquaringCeiling`), with the same tier reads, receipts and oracle-free
side sources and nothing rebuilt; `RatePaidNamedRO` bounds each pass at each candidate root (a union over the `G`
candidates, `GrindChain`), the pooled draws are `RateSessionDraw`'s causal horizon, and the two readings are its
`K = ⌊(1 − d̄)·n·P·N⌋` and `K_w = ⌊(1 − d_w)·P·N⌋`.

Frozen (no-function, hash): a bound in the random-oracle model over a session of abstract chip programs, which no Lean
function of the rate audit's code computes. -/
def RemoteSessionConservation : Prop :=
  ∀ (n P N b h ρ S L q r s d G lam τ rounds cRoot F : ℕ) (cost : ℕ → ℕ) (tier : ℕ → Fin τ)
    (A : Rate.DeployedSession n P N b h ρ S L q r s d G) (dbar dw : ℝ),
    0 < n → 0 < P → 0 < N → 0 < b → 0 < rounds * cRoot → dw ≤ 1 →
    Assumptions.WorkHardness b rounds cRoot cost →
    (∀ w η j ι c, Assumptions.SquaringCeiling cost F ((A.window w η).device j ι c)) →
    (∀ w η j ι c E R O O', ((A.window w η).device j ι c).side E R O = ((A.window w η).device j ι c).side E R O') →
    pr (fun x : Store N b × (Fin n → WindowCoins P N b h ρ r s d) =>
        (∀ w t, (x.2 w).2 t ∈ (A.win w x.2).correctAll x.1 (x.2 w).1) ∧
          ((((∑ w, (A.win w x.2).reads tier x.1 (x.2 w).1 : ℕ)) : ℝ) <
              (1 - (dbar + Deployed.gammaPass N b ρ S L lam rounds cRoot F)) * (n * P * N * b) ∨
            ∃ w, (((A.win w x.2).reads tier x.1 (x.2 w).1 : ℕ) : ℝ) <
              (1 - (dw + Deployed.gammaPass N b ρ S L lam rounds cRoot F)) * (P * N * b)))
      ≤ ENNReal.ofReal (1 - dbar) ^ (d * n) + ENNReal.ofReal (1 - dw / n) ^ (d * n) +
        (n * P * G : ℝ≥0∞) * ((N : ℝ≥0∞) / 2 ^ h + (q.choose 2 : ℝ≥0∞) / 2 ^ h + (2 ^ N : ℝ≥0∞) / 2 ^ lam)

/-- **The remote bandwidth audit at its deployed parameters.** `RemoteSessionConservation` at the stated codec, masked
Sloth over `p = 2^1024 − 105` at `k = 3` rounds (`WorkHardness 32768 3 c_root cost`, a block's price `3·c_root`), on
4 KiB blocks (`b = 32,768` bits), SHA-512 receipts (`h = 512`), 2,773 openings a window, `d̄ = 0.15%` over the horizon
and `d_w = 1%` in every window, and at most `C = 2^55` SHA-512 compressions a pass and `2^55` candidate roots: for every
session of at least seven windows, the probability that every opening passes while the session's tier reads fall
short of `(1 − γ̄)` of its bits or some window's fall short of `(1 − γ_w)` of its own is at most

  `2·2^-40 + n·P·2^55·(N/2^512 + C(2^55, 2)/2^512 + 2^N/2^λ)`.

**The reading.** `(1 − 0.0015)^(2773·n) ≤ e^(−4.16·n) ≤ 2^-40` from `n = 7`, and `(1 − 0.01/n)^(2773·n) ≤ e^(−27.73)
≤ 2^-40` at every `n`. At `λ = N + 40 + log₂(n·P) + 55` the last term is about `2^-40` too, a slack of `μ_λ ≈ 1/b`.
The capacities `S`, `L`, `F` and the tier split are the node's, measured; `γ̄ − d̄` and `γ_w − d_w` are the same sum
of named terms.

Proof: `Accounting.SecurityProofs.RemoteSessionConservationDeployed`, from `RemoteSessionConservation`, with
`1 + x ≤ e^x` and `log 2 < 0.6931471808`.

Frozen (no-function, hash): as `RemoteSessionConservation`. -/
def RemoteSessionConservationDeployed : Prop :=
  ∀ (n P N ρ S L q r s G lam τ cRoot F : ℕ) (cost : ℕ → ℕ) (tier : ℕ → Fin τ)
    (A : Rate.DeployedSession n P N 32768 512 ρ S L q r s 2773 G),
    7 ≤ n → 0 < P → 0 < N → 0 < cRoot → q ≤ 2 ^ 55 → G ≤ 2 ^ 55 →
    Assumptions.WorkHardness 32768 3 cRoot cost →
    (∀ w η j ι c, Assumptions.SquaringCeiling cost F ((A.window w η).device j ι c)) →
    (∀ w η j ι c E R O O', ((A.window w η).device j ι c).side E R O = ((A.window w η).device j ι c).side E R O') →
    pr (fun x : Store N 32768 × (Fin n → WindowCoins P N 32768 512 ρ r s 2773) =>
        (∀ w t, (x.2 w).2 t ∈ (A.win w x.2).correctAll x.1 (x.2 w).1) ∧
          ((((∑ w, (A.win w x.2).reads tier x.1 (x.2 w).1 : ℕ)) : ℝ) <
              (1 - (0.0015 + Deployed.gammaPass N 32768 ρ S L lam 3 cRoot F)) * (n * P * N * 32768) ∨
            ∃ w, (((A.win w x.2).reads tier x.1 (x.2 w).1 : ℕ) : ℝ) <
              (1 - (0.01 + Deployed.gammaPass N 32768 ρ S L lam 3 cRoot F)) * (P * N * 32768)))
      ≤ 2 / 2 ^ 40 + (n * P * 2 ^ 55 : ℝ≥0∞) *
        ((N : ℝ≥0∞) / 2 ^ 512 + (((2 ^ 55 : ℕ).choose 2 : ℕ) : ℝ≥0∞) / 2 ^ 512 + (2 ^ N : ℝ≥0∞) / 2 ^ lam)

end Accounting.Guarantees
