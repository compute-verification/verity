import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Data.Fintype.Sigma
import Mathlib.Basic.Real.Basic

/-!
# The remote bandwidth audit's node, its parameters and the γ terms (trusted)

The remote bandwidth audit (`docs/remote-bandwidth-exhaustion.md`) certifies that a node's memory read bandwidth went
to registered data. The node is a set of compute (every GPU die, host CPU and root complex: the console's actual-circuit
model's location `X_N`), its memory tiers and its boundary links (the NIC; NVLink to any GPU outside the node). A
memory tier (each GPU's HBM, the host's DRAM) is a location whose links all go to the node's compute (model v2.9);
with the tag's oracle a location of its own, causality and the Links row give that it hears only the node's compute
(`HearsChip`), so its port is counted at `β_m` bits a tick. A tier the audit rates at `R = 0` is unrated. A fleet
audited at once, or an NVLink domain, is a node too (`Node.fleet`): its boundary is the fleet's, and the links between
its members are inside it.

Over an audited interval `I` of `T` ticks, in `P` chained passes of `T + RTT` each, the audit's receipts cover the
declared capacity `C = Σ_m R_m · T` (`Node.declared`), against the node's calibrated capacity `C_true = Σ_m β_m · T`
(`Node.capacity`). Each γ term is a quantity of the model, named for the row of the model it comes from:

* `muChip` (μ_chip, the Memory row and the content's length bit): `P · (M_N + 1) / C`.
* `muRebuild` (μ_rebuild, the rebuild as a fit at π_r: the Compute row, WorkHardness in partial-completion form, and W
  when `W_r > 0`): `b · (Φ_r · T − W_r) / (c_word · C)`.
* `muHost` (μ_host, the tiers' calibrated ceilings): `(C_true − C) / C_true`, host DRAM's ceiling above its rate, any
  HBM rated below its ceiling, and the whole ceiling of an unrated tier.
* `muLambda` (λ, the guessing slack of each pass): `P · λ / C`.
* μ_sample is the sampler's `ε` itself (`Run.Sampled`).
* `muLink` (μ_link, the Links row, counted by signals, and the bound receipt): `(β_in + β_out · b / z_b) · T / C`.
* `muOut` (outside work at PoUW's pricing): `b · Φ_out · T / (W_tile · C)`.

`gamma` is the node-level guarantee's γ under Identity (statement C): μ_sample + μ_chip + μ_rebuild + μ_host + λ. Under
Identity there is no undeclared device of any kind, so μ_link is reported beside it, not charged. `gammaLinks` adds
μ_link and μ_out, for the weaker statement that allows memory-only helpers and outside work. Every term but `muHost` is
relative to `C`; `muHost` converts `C` into `C_true`.
-/

namespace Accounting.Remote

/-- An audited node over memory tiers `τ`. -/
structure Node (τ : Type) where
  /-- β_m: each tier's calibrated read ceiling, in bits per tick, the capacity of tier `m`'s port to the node's compute
  (the Links row) -/
  ceil : τ → ℕ
  /-- R_m: the rate the audit rates each tier at, in bits per tick, which its receipts exhaust: R_HBM for an HBM tier,
  R_host for host DRAM, `0` for an unrated tier -/
  rate : τ → ℕ
  /-- M_N: the on-chip state of the node's compute (GPU dies, host CPUs, root complex, NIC buffers), in bits -/
  chip : ℕ
  /-- β_in: the bits per tick the node's boundary links carry inward -/
  linkIn : ℕ
  /-- β_out: the bits per tick the node's boundary links carry outward -/
  linkOut : ℕ
  /-- Φ_r: the node compute's priced work per tick at the rebuild pricing π_r (the codec's squarings, on every GPU and
  CPU of the node), the Compute row's fit at π_r -/
  rebuild : ℕ
  /-- a tier's rate is within its calibrated ceiling -/
  rate_le : ∀ m, rate m ≤ ceil m

/-- The remote audit's parameters over an audited interval. -/
structure Audit where
  /-- |I|: the audited interval's ticks, `P` passes of `T + RTT` each -/
  T : ℕ
  /-- the chained passes; pass `j`'s seed is `H(σ_t, root_{j−1})` -/
  P : ℕ
  /-- b: a word's bits -/
  b : ℕ
  /-- z_b: the bits of the prior PoUW tile's noised output `z` that a receipt binds (`sketch(E_j ⊙ expand(z_{j−1}))`) -/
  zb : ℕ
  /-- c_word: a whole word's rebuild work at π_r, which WorkHardness charges `x · c_word / b` for `x` of its bits -/
  cWord : ℕ
  /-- λ: each pass's guessing slack, in bits -/
  lam : ℕ

variable {τ : Type} [Fintype τ]

/-- C: the declared capacity over `T` ticks, `Σ_m R_m · T` bits, which the audit's receipts cover. -/
def Node.declared (N : Node τ) (T : ℕ) : ℕ := ∑ m, N.rate m * T

/-- C_true: the node's calibrated capacity over `T` ticks, `Σ_m β_m · T` bits, all its memory tiers could carry. -/
def Node.capacity (N : Node τ) (T : ℕ) : ℕ := ∑ m, N.ceil m * T

/-- A fleet audited at once, or an NVLink domain, as one node: the members' tiers, chips and rebuild work together, with
the boundary links of the whole (`linkIn`, `linkOut`); links between members are inside it. -/
def Node.fleet {ι : Type} {τ : ι → Type} [Fintype ι] (Ns : ∀ i, Node (τ i)) (linkIn linkOut : ℕ) :
    Node (Σ i, τ i) where
  ceil m := (Ns m.1).ceil m.2
  rate m := (Ns m.1).rate m.2
  chip := ∑ i, (Ns i).chip
  linkIn := linkIn
  linkOut := linkOut
  rebuild := ∑ i, (Ns i).rebuild
  rate_le m := (Ns m.1).rate_le m.2

/-- μ_chip: `P · (M_N + 1) / C`, the node's on-chip state, with its length bit, at each pass boundary. -/
noncomputable def muChip (N : Node τ) (a : Audit) : ℝ := a.P * (N.chip + 1) / N.declared a.T

/-- μ_rebuild: `b · (Φ_r · T − W_r) / (c_word · C)`, the bits the node's spare work at the rebuild pricing can rebuild,
where `W_r` is the work at π_r that PoUW certifies. At `W_r = 0` it rests on the Compute row alone; subtracting `W_r`
needs W (`Run.Fit`). -/
noncomputable def muRebuild (N : Node τ) (a : Audit) (Wr : ℝ) : ℝ :=
  a.b * (N.rebuild * a.T - Wr) / (a.cWord * N.declared a.T)

/-- μ_host: `(C_true − C) / C_true`, the tiers' calibrated capacity the rates leave unexhausted (host DRAM's ceiling
above R_host, any HBM rated below its ceiling, an unrated tier's whole ceiling). -/
noncomputable def muHost (N : Node τ) (a : Audit) : ℝ :=
  ((N.capacity a.T : ℝ) - N.declared a.T) / N.capacity a.T

/-- μ_link: `(β_in + β_out · b / z_b) · T / C`: words carried in over the boundary pay one bit each, and a bound `z`
carried out unlocks `b / z_b` bits of reads outside the node. -/
noncomputable def muLink (N : Node τ) (a : Audit) : ℝ :=
  (N.linkIn + N.linkOut * a.b / a.zb) * a.T / N.declared a.T

/-- λ's share: `P · λ / C`. -/
noncomputable def muLambda (N : Node τ) (a : Audit) : ℝ := a.P * a.lam / N.declared a.T

/-- μ_out: `b · Φ_out · T / (W_tile · C)`, the bound `z` that outside work at PoUW's pricing (at most `Φ_out` per tick,
each `z` costing `W_tile`) can compute, amplified by `b / z_b`. -/
noncomputable def muOut (N : Node τ) (a : Audit) (Φout Wtile : ℝ) : ℝ := a.b * Φout * a.T / (Wtile * N.declared a.T)

/-- γ under Identity (statement C) = μ_sample + μ_chip + μ_rebuild + μ_host + λ, at sampler error `ε` (μ_sample) and
certified rebuild work `W_r`. -/
noncomputable def gamma (N : Node τ) (a : Audit) (ε Wr : ℝ) : ℝ :=
  ε + muChip N a + muRebuild N a Wr + muHost N a + muLambda N a

/-- γ without Identity, for memory-only helpers and outside work at most `Φ_out` per tick: `gamma` + μ_link + μ_out,
at tile cost `W_tile`. -/
noncomputable def gammaLinks (N : Node τ) (a : Audit) (ε Wr Φout Wtile : ℝ) : ℝ :=
  gamma N a ε Wr + muLink N a + muOut N a Φout Wtile

end Accounting.Remote
