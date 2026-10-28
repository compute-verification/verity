import Definitions.Accounting.RateAuditDeployed
import Definitions.Accounting.Remote.Deployed
import Definitions.Accounting.Assumptions.RateAudit
import Definitions.Accounting.Assumptions.RateAuditDeployed

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
* `RemoteSessionRoutes`: the session as a run of the remote audit (`Accounting.Remote.Run.ofDeployed`) meets its routes
  `Sampled d̄` and `Paid`, at the errors above.
* `RemoteSessionRunConservation`: `Accounting.SecurityProofs.Remote.Conservation` at those routes, read at the node: the
  run's tier reads are at least `(1 − (d̄ + μ))·C`, the same `γ̄` as `RemoteSessionConservation`'s pooled reading.
* `RemoteWindowRunConservation`: the same for each window (`Accounting.Remote.Run.ofDeployedWindow`) at `γ_w`.
* `RemoteSessionConservationRelay`, `RemoteSessionRunConservationRelay`, `RemoteWindowRunConservationRelay`: the three
  readings with side sources that may depend on the oracle (relayed receipts), at `m` relayed words a pass.

It rests on the random oracle (the game's; its pass seeds are Fiat–Shamir, `fiat-shamir`), and on two named assumptions:
`Accounting.Assumptions.WorkHardness` (completing `x` of a word's `b` bits costs `x/b` of `rounds · c_root`) and
`Accounting.Assumptions.SquaringCeiling` (a pass's completions cost at most the node's measured squaring capacity `F`).
The other measured capacities (the on-chip state `S`, the link `L`, the tiers) are the game's bounds, as in every
guarantee it composes.

The receipt here is the oracle's tag of the whole block; the deployed receipt hashes a 16-row sketch of it (the driver
also supports 64 rows). For the verbatim-sector class the sketch receipt is proved directly:
`RemoteSessionConservationSketch` (the product form, each grinding candidate at its own query cap) and
`RemoteSessionConservationSketchDeployed` (at the deployed 16-row parameters, under the per-pass hash budget
`Accounting.Assumptions.SketchHashBudget`: the sum form, whose query term stays non-vacuous at the measured caps).
-/

namespace Accounting.Guarantees

open Accounting Accounting.Rate
open Pous (Bits pr)
open scoped ENNReal

/-- **The remote bandwidth audit as deployed: an accepted session conserves its words, over the monitoring period and
in every window.** For every deployed session of `n` windows on one store (`Rate.DeployedSession`: in each window `P`
chained passes whose `G` candidate roots the prover grinds, pass devices with partial completion, interactive side
sources that ignore the oracle, at most `S` bits on chip, `L` over the link and `q` oracle queries a pass, and one
pooled draw of `d` of the window's `P·N` receipts), every split of device memory into `τ` tiers, and every `d̄` and
`d_w ≤ 1`: if
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
shortfall is `d̄` over the monitoring period and `d_w` in each window. The prover is one resource: nothing here says
which device, machine or party reads, completes or holds a word. Reads are counted over the tiers, so `R·T` is their
sum's capacity; the term `G` is the grinding the Fiat–Shamir seed allows (`fiat-shamir`).

**The proof** (`Accounting.SecurityProofs.RemoteSessionConservation`): each pass device is simulated by an interactive
device of `RatePaidNamedRO`'s class whose side sources also answer the completed bits, so its link carries
`L + b·F/(rounds·c_root)` (`WorkHardness` and `SquaringCeiling`), with the same tier reads, receipts and oracle-free
side sources and nothing rebuilt; `RatePaidNamedRO` bounds each pass at each candidate root (a union over the `G`
candidates, `GrindChain`), the pooled draws are `RateSessionDraw`'s causal rounds, and the two readings are its
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
    pr (fun x : Payloads N b × (Fin n → WindowCoins P N b h ρ r s d) =>
        (∀ w t, (x.2 w).2 t ∈ (A.win w x.2).correctAll x.1 (x.2 w).1) ∧
          ((((∑ w, (A.win w x.2).reads tier x.1 (x.2 w).1 : ℕ)) : ℝ) <
              (1 - (dbar + Deployed.gammaPass N b ρ S L lam rounds cRoot F)) * (n * P * N * b) ∨
            ∃ w, (((A.win w x.2).reads tier x.1 (x.2 w).1 : ℕ) : ℝ) <
              (1 - (dw + Deployed.gammaPass N b ρ S L lam rounds cRoot F)) * (P * N * b)))
      ≤ ENNReal.ofReal (1 - dbar) ^ (d * n) + ENNReal.ofReal (1 - dw / n) ^ (d * n) +
        (n * P * G : ℝ≥0∞) * ((N : ℝ≥0∞) / 2 ^ h + (q.choose 2 : ℝ≥0∞) / 2 ^ h + (2 ^ N : ℝ≥0∞) / 2 ^ lam)

/-- **The remote bandwidth audit at its deployed parameters.** `RemoteSessionConservation` at the stated codec, masked
Sloth over `p = 2^1024 − 105` at `k = 3` rounds (`WorkHardness 32768 3 c_root cost`, a block's price `3·c_root`), on
4 KiB blocks (`b = 32,768` bits), SHA-512 receipts (`h = 512`), 2,773 openings a window, `d̄ = 0.15%` over the
monitoring period and `d_w = 1%` in every window, and at most `C = 2^55` SHA-512 compressions a pass and `2^55`
candidate roots: for every session of at least seven windows, the probability that every opening passes while the
session's tier reads fall short of `(1 − γ̄)` of its bits or some window's fall short of `(1 − γ_w)` of its own is at
most

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
    pr (fun x : Payloads N 32768 × (Fin n → WindowCoins P N 32768 512 ρ r s 2773) =>
        (∀ w t, (x.2 w).2 t ∈ (A.win w x.2).correctAll x.1 (x.2 w).1) ∧
          ((((∑ w, (A.win w x.2).reads tier x.1 (x.2 w).1 : ℕ)) : ℝ) <
              (1 - (0.0015 + Deployed.gammaPass N 32768 ρ S L lam 3 cRoot F)) * (n * P * N * 32768) ∨
            ∃ w, (((A.win w x.2).reads tier x.1 (x.2 w).1 : ℕ) : ℝ) <
              (1 - (0.01 + Deployed.gammaPass N 32768 ρ S L lam 3 cRoot F)) * (P * N * 32768)))
      ≤ 2 / 2 ^ 40 + (n * P * 2 ^ 55 : ℝ≥0∞) *
        ((N : ℝ≥0∞) / 2 ^ 512 + (((2 ^ 55 : ℕ).choose 2 : ℕ) : ℝ≥0∞) / 2 ^ 512 + (2 ^ N : ℝ≥0∞) / 2 ^ lam)

/-- **The deployed session meets the remote audit's routes.** Read as a run of the remote audit on a node `Nd` whose
declared capacity over `a`'s interval is the session's `n·P·N·b` bits, with `a`'s passes the session's `n·P`, its word
`b` and its slack `λ` (`Accounting.Remote.Run.ofDeployed`, at `b·F/(rounds·c_root)` bits completed a pass), every
deployed session under `RemoteSessionConservation`'s hypotheses meets

* `Sampled d̄` (μ_sample) except with probability `(1 − d̄)^(d·n)`: the pooled draws pass while the correct receipts
  cover less than `(1 − d̄)·C`; and
* `Paid` except with probability `n·P·G·(N/2^h + C(q, 2)/2^h + 2^N/2^λ)`: the correct receipts' words exceed what the
  run paid for them (its tier reads, its on-chip state, the side link beyond the stream, its completed bits) and its
  `n·P` slacks `λ`.

**The reading.** These are the routes `Accounting.SecurityProofs.Remote.Conservation` composes, proved for the runs the
deployed game realizes; the run's fields are the game's (`Run.ofDeployed`), so its `paid` is the session's tier reads
and the `n·P` passes' `S + (L − ρ) + b·F/(rounds·c_root)` bits.

Frozen (no-function, hash): as `RemoteSessionConservation`. -/
def RemoteSessionRoutes : Prop :=
  ∀ (n P N b h ρ S L q r s d G lam τ rounds cRoot F : ℕ) (cost : ℕ → ℕ) (tier : ℕ → Fin τ)
    (A : Rate.DeployedSession n P N b h ρ S L q r s d G) (Nd : Remote.Node (Fin τ)) (a : Remote.Audit) (dbar : ℝ),
    0 < n → 0 < P → 0 < N → 0 < b → 0 < rounds * cRoot → dbar ≤ 1 →
    a.P = n * P → a.b = b → a.lam = lam → Nd.declared a.T = n * P * N * b →
    Assumptions.WorkHardness b rounds cRoot cost →
    (∀ w η j ι c, Assumptions.SquaringCeiling cost F ((A.window w η).device j ι c)) →
    (∀ w η j ι c E R O O', ((A.window w η).device j ι c).side E R O = ((A.window w η).device j ι c).side E R O') →
    (Remote.Run.ofDeployed A tier (Deployed.rebuildBits b rounds cRoot F) Nd a).Sampled dbar ((1 - dbar) ^ (d * n)) ∧
      (Remote.Run.ofDeployed A tier (Deployed.rebuildBits b rounds cRoot F) Nd a).Paid
        ((n * P * G : ℝ) * ((N : ℝ) / 2 ^ h + (q.choose 2 : ℝ) / 2 ^ h + (2 ^ N : ℝ) / 2 ^ lam))

/-- **The deployed session conserves its words as a run of the remote audit, at the same `γ`.** Under
`RemoteSessionRoutes`' hypotheses, the probability that the run of the session (`Accounting.Remote.Run.ofDeployed`)
accepts while its correct receipts cover less than `(1 − d̄)·C`, or its tier reads are less than `(1 − γ̄)·C`, is at
most the two routes' errors summed:

  `γ̄ = d̄ + μ_chip + μ_link + μ_rebuild + μ_λ`,   the error `(1 − d̄)^(d·n) + n·P·G·(N/2^h + C(q, 2)/2^h + 2^N/2^λ)`.

**The reading.** `Accounting.SecurityProofs.Remote.Conservation` gives `(1 − d̄ − λ)·C` of payments at every
location; the run's payments other than its tier reads are its state, its side link beyond the stream and its
completed bits, which are `μ_chip`, `μ_link` and `μ_rebuild` of `C`, and the remote model's `λ` share
(`Accounting.Remote.muLambda`) is `μ_λ`. So this is `RemoteSessionConservation`'s pooled reading, the same `γ̄`, with
the window's term dropped.

Frozen (no-function, hash): as `RemoteSessionConservation`. -/
def RemoteSessionRunConservation : Prop :=
  ∀ (n P N b h ρ S L q r s d G lam τ rounds cRoot F : ℕ) (cost : ℕ → ℕ) (tier : ℕ → Fin τ)
    (A : Rate.DeployedSession n P N b h ρ S L q r s d G) (Nd : Remote.Node (Fin τ)) (a : Remote.Audit) (dbar : ℝ),
    0 < n → 0 < P → 0 < N → 0 < b → 0 < rounds * cRoot → dbar ≤ 1 →
    a.P = n * P → a.b = b → a.lam = lam → Nd.declared a.T = n * P * N * b →
    Assumptions.WorkHardness b rounds cRoot cost →
    (∀ w η j ι c, Assumptions.SquaringCeiling cost F ((A.window w η).device j ι c)) →
    (∀ w η j ι c E R O O', ((A.window w η).device j ι c).side E R O = ((A.window w η).device j ι c).side E R O') →
    (Remote.Run.ofDeployed A tier (Deployed.rebuildBits b rounds cRoot F) Nd a).Holds
      ((1 - dbar) ^ (d * n) + (n * P * G : ℝ) * ((N : ℝ) / 2 ^ h + (q.choose 2 : ℝ) / 2 ^ h + (2 ^ N : ℝ) / 2 ^ lam))
      fun x =>
        (1 - dbar) * Nd.declared a.T ≤
            ((Remote.Run.ofDeployed A tier (Deployed.rebuildBits b rounds cRoot F) Nd a).correct x * a.b : ℝ) ∧
          (1 - (dbar + Deployed.gammaPass N b ρ S L lam rounds cRoot F)) * Nd.declared a.T ≤
            ((Remote.Run.ofDeployed A tier (Deployed.rebuildBits b rounds cRoot F) Nd a).reads x : ℝ)

/-- **Each window of the deployed session conserves its words as a run of the remote audit, at the same `γ_w`.**
Window `w` read as a run of the remote audit (`Accounting.Remote.Run.ofDeployedWindow`) on a node `Nd` whose declared
capacity over `a`'s interval is the window's `P·N·b` bits, with `a`'s passes the window's `P`: under
`RemoteSessionConservation`'s hypotheses and `d_w ≤ 1`, the run meets `Sampled d_w` except with probability
`(1 − d_w/n)^(d·n)` and `Paid` except with `P·G·(N/2^h + C(q, 2)/2^h + 2^N/2^λ)`, and
`Accounting.SecurityProofs.Remote.Conservation` at them gives: the probability that every window's draw passes while
window `w`'s correct receipts cover less than `(1 − d_w)·C`, or its tier reads are less than `(1 − γ_w)·C`,
`γ_w = d_w + μ_chip + μ_link + μ_rebuild + μ_λ`, is at most

  `(1 − d_w/n)^(d·n) + P·G·(N/2^h + C(q, 2)/2^h + 2^N/2^λ)`.

**The reading.** `RemoteSessionConservation`'s per-window reading, one window at a time: the draw's term is the same
(it bounds the event that some window holds at most `⌊(1 − d_w)·P·N⌋`), and the unpaid term is one window's.

Frozen (no-function, hash): as `RemoteSessionConservation`. -/
def RemoteWindowRunConservation : Prop :=
  ∀ (n P N b h ρ S L q r s d G lam τ rounds cRoot F : ℕ) (cost : ℕ → ℕ) (tier : ℕ → Fin τ)
    (A : Rate.DeployedSession n P N b h ρ S L q r s d G) (w : Fin n) (Nd : Remote.Node (Fin τ)) (a : Remote.Audit)
    (dw : ℝ),
    0 < P → 0 < N → 0 < b → 0 < rounds * cRoot → dw ≤ 1 →
    a.P = P → a.b = b → a.lam = lam → Nd.declared a.T = P * N * b →
    Assumptions.WorkHardness b rounds cRoot cost →
    (∀ w η j ι c, Assumptions.SquaringCeiling cost F ((A.window w η).device j ι c)) →
    (∀ w η j ι c E R O O', ((A.window w η).device j ι c).side E R O = ((A.window w η).device j ι c).side E R O') →
    (Remote.Run.ofDeployedWindow A w tier (Deployed.rebuildBits b rounds cRoot F) Nd a).Holds
      ((1 - dw / n) ^ (d * n) + (P * G : ℝ) * ((N : ℝ) / 2 ^ h + (q.choose 2 : ℝ) / 2 ^ h + (2 ^ N : ℝ) / 2 ^ lam))
      fun x =>
        (1 - dw) * Nd.declared a.T ≤
            ((Remote.Run.ofDeployedWindow A w tier (Deployed.rebuildBits b rounds cRoot F) Nd a).correct x * a.b : ℝ) ∧
          (1 - (dw + Deployed.gammaPass N b ρ S L lam rounds cRoot F)) * Nd.declared a.T ≤
            ((Remote.Run.ofDeployedWindow A w tier (Deployed.rebuildBits b rounds cRoot F) Nd a).reads x : ℝ)

/-- **The remote bandwidth audit as deployed, with side sources that may relay receipts.** `RemoteSessionConservation`
without its hypothesis that the side sources ignore the oracle: the rest of the prover may answer the chip's side reads
with anything, receipts it computed elsewhere included. For every `m`, the probability that every opening of every
window passes while the session's tier reads fall short of `(1 − γ̄)` of its bits or some window's fall short of
`(1 − γ_w)` of its own, with `γ` read at the slack `λ + m·b`,

  `γ̄ = d̄ + μ_chip + μ_link + μ_rebuild + (λ + m·b)/(N·b)`,   `γ_w = d_w + μ_chip + μ_link + μ_rebuild + (λ + m·b)/(N·b)`,

is at most

  `(1 − d̄)^(d·n) + (1 − d_w/n)^(d·n) + n·P·G·(C(N, m)·2^L/2^(h·m) + 2^q/2^λ)`.

**The reading.** Each pass may take `m` receipts over the link instead of reading their blocks: a share `m/N` of its
words, beyond `μ_λ`. That costs it `h` bits of the oracle per receipt from at most `L` link bits, so `m` a little over
`L/h` makes the term small; the relay share is then about `L/(h·N)`, which is `μ_link·b/h`. The link carries the
completed bits no more than in `RemoteSessionConservation`: completions are counted in `μ_rebuild` alone. The
compression's term is `2^q/2^λ` in place of `2^N/2^λ`, since naming the words needs collision-free queries, which
relayed answers can break.

**The proof** (`Accounting.SecurityProofs.RemoteSessionConservationRelay`): each pass device is simulated by an
interactive device whose device memory also answers the completed bits from the store (`PIDevice.toInteractiveMem`), on
the same side sources and link, with its tier reads the device's plus its completed bits; `RatePaid` bounds each pass
at each candidate root, and the session is as in `RemoteSessionConservation`.

Frozen (no-function, hash): as `RemoteSessionConservation`. -/
def RemoteSessionConservationRelay : Prop :=
  ∀ (n P N b h ρ S L q r s d G lam m τ rounds cRoot F : ℕ) (cost : ℕ → ℕ) (tier : ℕ → Fin τ)
    (A : Rate.DeployedSession n P N b h ρ S L q r s d G) (dbar dw : ℝ),
    0 < n → 0 < P → 0 < N → 0 < b → 0 < rounds * cRoot → dw ≤ 1 →
    Assumptions.WorkHardness b rounds cRoot cost →
    (∀ w η j ι c, Assumptions.SquaringCeiling cost F ((A.window w η).device j ι c)) →
    pr (fun x : Payloads N b × (Fin n → WindowCoins P N b h ρ r s d) =>
        (∀ w t, (x.2 w).2 t ∈ (A.win w x.2).correctAll x.1 (x.2 w).1) ∧
          ((((∑ w, (A.win w x.2).reads tier x.1 (x.2 w).1 : ℕ)) : ℝ) <
              (1 - (dbar + Deployed.gammaPass N b ρ S L (lam + m * b) rounds cRoot F)) * (n * P * N * b) ∨
            ∃ w, (((A.win w x.2).reads tier x.1 (x.2 w).1 : ℕ) : ℝ) <
              (1 - (dw + Deployed.gammaPass N b ρ S L (lam + m * b) rounds cRoot F)) * (P * N * b)))
      ≤ ENNReal.ofReal (1 - dbar) ^ (d * n) + ENNReal.ofReal (1 - dw / n) ^ (d * n) +
        (n * P * G : ℝ≥0∞) * ((N.choose m : ℝ≥0∞) * 2 ^ L / 2 ^ (h * m) + (2 ^ q : ℝ≥0∞) / 2 ^ lam)

/-- **The deployed session with relaying side sources conserves its words as a run of the remote audit.**
`RemoteSessionRunConservation` without the hypothesis that the side sources ignore the oracle, for every `m`, with the
run's slack `a.λ = λ + m·b` a pass: the probability that the run (`Accounting.Remote.Run.ofDeployed`) accepts while its
correct receipts cover less than `(1 − d̄)·C`, or its tier reads are less than `(1 − γ̄)·C`, with
`γ̄ = d̄ + μ_chip + μ_link + μ_rebuild + (λ + m·b)/(N·b)`, is at most

  `(1 − d̄)^(d·n) + n·P·G·(C(N, m)·2^L/2^(h·m) + 2^q/2^λ)`.

**The reading.** `RemoteSessionConservationRelay`'s pooled reading through
`Accounting.SecurityProofs.Remote.Conservation`: the relayed receipts are the run's `λ` share beyond the compression's,
`m` words a pass.

Frozen (no-function, hash): as `RemoteSessionConservation`. -/
def RemoteSessionRunConservationRelay : Prop :=
  ∀ (n P N b h ρ S L q r s d G lam m τ rounds cRoot F : ℕ) (cost : ℕ → ℕ) (tier : ℕ → Fin τ)
    (A : Rate.DeployedSession n P N b h ρ S L q r s d G) (Nd : Remote.Node (Fin τ)) (a : Remote.Audit) (dbar : ℝ),
    0 < n → 0 < P → 0 < N → 0 < b → 0 < rounds * cRoot → dbar ≤ 1 →
    a.P = n * P → a.b = b → a.lam = lam + m * b → Nd.declared a.T = n * P * N * b →
    Assumptions.WorkHardness b rounds cRoot cost →
    (∀ w η j ι c, Assumptions.SquaringCeiling cost F ((A.window w η).device j ι c)) →
    (Remote.Run.ofDeployed A tier (Deployed.rebuildBits b rounds cRoot F) Nd a).Holds
      ((1 - dbar) ^ (d * n) +
        (n * P * G : ℝ) * ((N.choose m : ℝ) * 2 ^ L / 2 ^ (h * m) + (2 ^ q : ℝ) / 2 ^ lam))
      fun x =>
        (1 - dbar) * Nd.declared a.T ≤
            ((Remote.Run.ofDeployed A tier (Deployed.rebuildBits b rounds cRoot F) Nd a).correct x * a.b : ℝ) ∧
          (1 - (dbar + Deployed.gammaPass N b ρ S L (lam + m * b) rounds cRoot F)) * Nd.declared a.T ≤
            ((Remote.Run.ofDeployed A tier (Deployed.rebuildBits b rounds cRoot F) Nd a).reads x : ℝ)

/-- **Each window of the deployed session with relaying side sources conserves its words as a run of the remote
audit.** `RemoteWindowRunConservation` without the hypothesis that the side sources ignore the oracle, for every `m`,
with the run's slack `a.λ = λ + m·b` a pass: the probability that every window's draw passes while window `w`'s correct
receipts cover less than `(1 − d_w)·C`, or its tier reads are less than `(1 − γ_w)·C`, with
`γ_w = d_w + μ_chip + μ_link + μ_rebuild + (λ + m·b)/(N·b)`, is at most

  `(1 − d_w/n)^(d·n) + P·G·(C(N, m)·2^L/2^(h·m) + 2^q/2^λ)`.

Frozen (no-function, hash): as `RemoteSessionConservation`. -/
def RemoteWindowRunConservationRelay : Prop :=
  ∀ (n P N b h ρ S L q r s d G lam m τ rounds cRoot F : ℕ) (cost : ℕ → ℕ) (tier : ℕ → Fin τ)
    (A : Rate.DeployedSession n P N b h ρ S L q r s d G) (w : Fin n) (Nd : Remote.Node (Fin τ)) (a : Remote.Audit)
    (dw : ℝ),
    0 < P → 0 < N → 0 < b → 0 < rounds * cRoot → dw ≤ 1 →
    a.P = P → a.b = b → a.lam = lam + m * b → Nd.declared a.T = P * N * b →
    Assumptions.WorkHardness b rounds cRoot cost →
    (∀ w η j ι c, Assumptions.SquaringCeiling cost F ((A.window w η).device j ι c)) →
    (Remote.Run.ofDeployedWindow A w tier (Deployed.rebuildBits b rounds cRoot F) Nd a).Holds
      ((1 - dw / n) ^ (d * n) +
        (P * G : ℝ) * ((N.choose m : ℝ) * 2 ^ L / 2 ^ (h * m) + (2 ^ q : ℝ) / 2 ^ lam))
      fun x =>
        (1 - dw) * Nd.declared a.T ≤
            ((Remote.Run.ofDeployedWindow A w tier (Deployed.rebuildBits b rounds cRoot F) Nd a).correct x * a.b : ℝ) ∧
          (1 - (dw + Deployed.gammaPass N b ρ S L (lam + m * b) rounds cRoot F)) * Nd.declared a.T ≤
            ((Remote.Run.ofDeployedWindow A w tier (Deployed.rebuildBits b rounds cRoot F) Nd a).reads x : ℝ)

/-- **The remote bandwidth audit with deployed tensor-sketch receipts, for the verbatim class.** A session of `n`
windows draws one `ks × (32·m)` signs matrix per window, runs `P` causal passes with `G` candidate oracles per pass (the
grinding its Fiat–Shamir pass seed allows, `fiat-shamir`), then draws `d` of the window's `P·N` receipts. Every pass is
a `RateSketch.Device`: it obtains 32-byte sectors only whole and verbatim. If each pass rebuilds at most `X` bits, and
the signs' rank tail is `Assumptions.SketchRank ks m r ε`, then the probability that every opening passes while the
session or some window falls below its reading is at most

  `(1 − d̄)^(d·n) + (1 − d_w/n)^(d·n) + n·ε + n·P·G·(N/2^h + q/2^(8·r))`.

The reading's non-sampling share is

  `γ_sketch = (S + (L − ρ·256) + X)/(N·m·256)`.

The two added receipt terms are exactly the rank-priced reduction behind `RateAuditTensorSketchRank`: one rank-tail
event per window, and one `q/2^(8r)` unknown-sector query term per candidate pass. `N/2^h` covers a correct receipt
whose tag was never queried. There is no compression slack `λ`: in the verbatim class an unobtained sector is uniform.
`RateTensorSketchTable` proves why this statement cannot be extended to arbitrary layouts. Here each of a pass's `G`
candidates gets the whole cap `q`; the deployed guarantee (`RemoteSessionConservationSketchDeployed`) prices the
candidates' queries by one shared per-pass budget instead.

Frozen (no-function, hash): a bound in the random-oracle model over a causal session of abstract verbatim-sector
devices, not the rate audit's code. -/
def RemoteSessionConservationSketch : Prop :=
  ∀ (n P N m ρ h S L q ks G d r X : ℕ) (ε : ℝ≥0∞)
    (A : RateSketch.SketchDeployedSession n P N m ρ h S L q ks G d) (dbar dw : ℝ),
    0 < n → 0 < P → 0 < N → 0 < m → r ≤ 32 → dw ≤ 1 →
    Assumptions.SketchRank ks m r ε →
    (∀ w η j ι c E σ R O,
      let B := (A.window w η).device j ι c
      let D := RateSketch.Data.ofParts E R
      (B.prog σ D).rebuilds O D * 256 ≤ X) →
    pr (fun x : RateSketch.SketchStore N m ×
        (Fin n → RateSketch.SketchWindowCoins P N m ρ h ks G d) =>
      (∀ w t, (x.2 w).2.2 t ∈
        (A.win w x.2).correctAll x.1 (x.2 w).1 (x.2 w).2.1) ∧
        ((((∑ w, (A.win w x.2).reads x.1 (x.2 w).1 (x.2 w).2.1 : ℕ)) : ℝ) <
            (1 - (dbar + RateSketch.gammaSketch N m ρ S L X)) * (n * P * N * m * 256) ∨
          ∃ w, (((A.win w x.2).reads x.1 (x.2 w).1 (x.2 w).2.1 : ℕ) : ℝ) <
            (1 - (dw + RateSketch.gammaSketch N m ρ S L X)) * (P * N * m * 256)))
      ≤ ENNReal.ofReal (1 - dbar) ^ (d * n) + ENNReal.ofReal (1 - dw / n) ^ (d * n) +
        (n : ℝ≥0∞) * ε + (n * P * G : ℝ≥0∞) * ((N : ℝ≥0∞) / 2 ^ h + (q : ℝ≥0∞) / 2 ^ (8 * r))

/-- **The verbatim sketch session at the deployed receipt parameters, under the per-pass hash budget.**
`RemoteSessionConservationSketch`'s game at 4 KiB tiles (`m = 128` 32-byte sectors), 16 sketch rows, SHA-512 tags,
rank `r = 12`, 2,773 pooled openings per window, `d̄ = 0.15%` and `d_w = 1%`, with one added named assumption: across
each pass's `G` grinding candidates the oracle queries total at most `qtot`
(`Accounting.Assumptions.SketchHashBudget`). The receipt term is the sum form — queries add across a pass's
candidates, and grinding multiplies only the tag-guess term. From seven windows on, the two draw terms are at most
`2^-40` each:

  `2·2^-40 + n·ε + (n·P·G)·N/2^512 + (n·P)·qtot/2^96`.

**The reading.** `qtot` is the node's hash budget for one pass, so the deployed cap of at most `2^55` compressions a
pass reads `qtot = 2^55`, where the query term is `n·P·2^-41` (`sketchQueryTermCapsSum`) and the guess term
`n·P·G·N/2^512` stays negligible at `G ≤ 2^55`. The multiplicative bound `n·P·G·(N/2^512 + q/2^96)` is the special
case `qtot = G·q` that the per-candidate caps alone give (`Device.budget_of_bounded`;
`Accounting.SecurityProofs.RemoteSessionConservationSketchDeployedProduct` keeps it stated), which at those caps was
vacuous: its query term alone is `2^14` before `n·P` (`sketchQueryTermCaps`).

Proof: `Accounting.SecurityProofs.RemoteSessionConservationSketchDeployed`, from
`RemoteSessionConservationSketchSum`'s budgeted union over the queries each pass's candidates actually make.

Frozen (no-function, hash): `RemoteSessionConservationSketch`'s game at the deployed parameters, under the per-pass
hash budget. -/
def RemoteSessionConservationSketchDeployed : Prop :=
  ∀ (n P N ρ S L q G X qtot : ℕ) (ε : ℝ≥0∞)
    (A : RateSketch.SketchDeployedSession n P N 128 ρ 512 S L q 16 G 2773),
    7 ≤ n → 0 < P → 0 < N →
    Assumptions.SketchRank 16 128 12 ε →
    (∀ w η j ι c E σ R O,
      let B := (A.window w η).device j ι c
      let D := RateSketch.Data.ofParts E R
      (B.prog σ D).rebuilds O D * 256 ≤ X) →
    (∀ w η j ι, Assumptions.SketchHashBudget ((A.window w η).device j ι) qtot) →
    pr (fun x : RateSketch.SketchStore N 128 ×
        (Fin n → RateSketch.SketchWindowCoins P N 128 ρ 512 16 G 2773) =>
      (∀ w t, (x.2 w).2.2 t ∈
        (A.win w x.2).correctAll x.1 (x.2 w).1 (x.2 w).2.1) ∧
        ((((∑ w, (A.win w x.2).reads x.1 (x.2 w).1 (x.2 w).2.1 : ℕ)) : ℝ) <
            (1 - (0.0015 + RateSketch.gammaSketch N 128 ρ S L X)) * (n * P * N * 32768) ∨
          ∃ w, (((A.win w x.2).reads x.1 (x.2 w).1 (x.2 w).2.1 : ℕ) : ℝ) <
            (1 - (0.01 + RateSketch.gammaSketch N 128 ρ S L X)) * (P * N * 32768)))
      ≤ 2 / 2 ^ 40 + (n : ℝ≥0∞) * ε +
        (n * P * G : ℝ≥0∞) * ((N : ℝ≥0∞) / 2 ^ 512) +
          (n * P : ℝ≥0∞) * ((qtot : ℝ≥0∞) / 2 ^ 96)

end Accounting.Guarantees
