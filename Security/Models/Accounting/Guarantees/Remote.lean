import Models.Accounting.Assumptions.Remote

/-!
# The remote bandwidth audit's guarantees (trusted)

Statements about a run of the remote audit (`Accounting.Remote.Run`), each at the coins drawn (a per-run reading): the
routes it composes are bounds at each outcome, and the guarantee bounds the probability, over the audit's coins, that
the audit accepts and the bound fails, by the routes' errors summed.

* **Conservation**, at no location: the correct receipts cover `(1 − μ_sample) · C` of the declared capacity over HBM
  and host DRAM, and every one was paid for, wherever it was computed, so the reads, held state and rebuilt bits at
  every location together are at least `(1 − μ_sample − λ) · C`. The state and compute of locations outside the node
  are unbounded without a location assumption, so conservation reports them as terms (`Run.paid`).
* **NodeLevel**, under Identity (statement C, no undeclared device of any kind): the node's tiers carried at least
  `(1 − γ) · C_true` bits to its compute, `γ = μ_sample + μ_chip + μ_rebuild + μ_host + λ` (`Accounting.Remote.gamma`).
  μ_link is reported beside it, not charged. **NodeLevelJoint** is its counting form (Mbw at the node): the bits of
  data independent of the store, seeds and oracle that the node determines from its tiers are at most `γ · C_true`.
* **NodeLevelLinks**, the weaker statement without Identity: memory-only helpers outside the node and outside work at
  most `Φ_out` per tick, with the links charged (inbound `1:1`, outbound at `b / z_b`):
  `γ + μ_link + μ_out` (`Accounting.Remote.gammaLinks`).

Each takes the rebuild's route as a fit at π_r (`Run.Rebuild`, `Run.Fit`). A fleet audited at once, or an NVLink
domain, is a node (`Accounting.Remote.Node.fleet`), whose μ_link counts only the whole's boundary. Each `def G : Prop`
is proved from its routes as `Accounting.SecurityProofs.Remote.G : Accounting.Guarantees.Remote.G`; the routes are
proved for the runs the actual circuit realizes by the lanes each route names (`Accounting.Remote.Run`).
-/

namespace Accounting.Guarantees.Remote

open Accounting.Remote

/-- **Conservation, at no location.** On any node and any run whose receipts are sampled at error `ε` (`Sampled`) and
whose correct receipts were paid for (`Paid`), the probability that the audit accepts and either the correct receipts
cover less than `(1 − ε) · C` or the payments at every location together are less than `(1 − ε − λ) · C` is at most
the two routes' errors summed. -/
def Conservation : Prop :=
  ∀ (τ Ω : Type) [Fintype τ] [Fintype Ω] (N : Node τ) (a : Audit) (r : Run N a Ω) (ε δs δp : ℝ),
    0 < N.declared a.T → r.Sampled ε δs → r.Paid δp →
    r.Holds (δs + δp) fun ω =>
      (1 - ε) * N.declared a.T ≤ (r.correct ω * a.b : ℝ) ∧ (1 - ε - muLambda N a) * N.declared a.T ≤ r.paid ω

/-- **The node-level guarantee, under Identity** (statement C). On any node and any run meeting Identity and the routes
`Sampled`, `Paid`, `ChipState`, `Rebuild` and `Fit W_r`, the probability that the audit accepts and the node's tiers
carried fewer than `(1 − γ) · C_true` bits to its compute is at most the errors summed. -/
def NodeLevel : Prop :=
  ∀ (τ Ω : Type) [Fintype τ] [Fintype Ω] (N : Node τ) (a : Audit) (r : Run N a Ω) (ε Wr δs δp δm δr δf : ℝ),
    0 < N.declared a.T → 0 ≤ ε → 0 < a.cWord → Assumptions.Identity r →
    r.Sampled ε δs → r.Paid δp → r.ChipState δm → r.Rebuild δr → r.Fit Wr δf →
    r.Holds (δs + δp + δm + δr + δf) fun ω => (1 - gamma N a ε Wr) * N.capacity a.T ≤ r.reads ω

/-- **The node-level guarantee's counting form (Mbw at the node), under Identity.** Under `NodeLevel`'s hypotheses, with
`PaidJoint` for `Paid` and the tiers' ports within their ceilings (`Ports`), the probability that the audit accepts and
the node determined more than `γ · C_true` bits of data independent of the store, seeds and oracle from its tiers is at
most the errors summed. -/
def NodeLevelJoint : Prop :=
  ∀ (τ Ω : Type) [Fintype τ] [Fintype Ω] (N : Node τ) (a : Audit) (r : Run N a Ω) (ε Wr δs δp δm δr δf δq : ℝ),
    0 < N.declared a.T → 0 ≤ ε → 0 < a.cWord → Assumptions.Identity r →
    r.Sampled ε δs → r.PaidJoint δp → r.ChipState δm → r.Rebuild δr → r.Fit Wr δf → r.Ports δq →
    r.Holds (δs + δp + δm + δr + δf + δq) fun ω => (r.other ω : ℝ) ≤ gamma N a ε Wr * N.capacity a.T

/-- **The node-level guarantee without Identity.** On any node and any run whose outside work is at most `Φ_out` per
tick, meeting PoUW's conjecture for the bound `z` at tile cost `W_tile` and the routes `Sampled`, `CoLocated`,
`ChipState`, `Rebuild`, `Fit W_r` and `Links`, the probability that the audit accepts and the node's tiers carried fewer
than `(1 − γ − μ_link − μ_out) · C_true` bits to its compute is at most the errors summed. -/
def NodeLevelLinks : Prop :=
  ∀ (τ Ω : Type) [Fintype τ] [Fintype Ω] (N : Node τ) (a : Audit) (r : Run N a Ω)
    (ε Wr Φout Wtile δs δc δm δr δf δl δz : ℝ),
    0 < N.declared a.T → 0 ≤ ε → 0 < a.cWord → 0 < Wtile →
    Assumptions.OutsideWork r Φout → Assumptions.ZCost r Wtile δz →
    r.Sampled ε δs → r.CoLocated δc → r.ChipState δm → r.Rebuild δr → r.Fit Wr δf → r.Links δl →
    r.Holds (δs + δc + δm + δr + δf + δl + δz) fun ω =>
      (1 - gammaLinks N a ε Wr Φout Wtile) * N.capacity a.T ≤ r.reads ω

end Accounting.Guarantees.Remote
