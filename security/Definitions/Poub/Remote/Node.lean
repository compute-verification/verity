module

public import Mathlib.Basic.Real.Basic
public import Mathlib.Algebra.BigOperators.Group.Finset.Defs

/-! PoUB's node, its audit parameters and the γ terms. -/

@[expose] public section

namespace ResourceAccounting.Remote

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
  rate_le : ∀ m, rate m ≤ ceil m

/-- PoUB's parameters over an audited interval. -/
structure Audit where
  /-- |I|: the audited interval's ticks, `P` passes of `T + RTT` each -/
  T : ℕ
  /-- the chained passes; pass `j`'s seed is `H(σ_t, root_{j−1})` -/
  P : ℕ
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

noncomputable def muChip (N : Node τ) (a : Audit) : ℝ := a.P * (N.chip + 1) / N.declared a.T

/-- μ_rebuild: `b · (Φ_r · T − W_r) / (c_word · C)`, `W_r` the work at π_r that PoUW certifies. Subtracting `W_r`
needs W (`Run.Fit`). -/
noncomputable def muRebuild (N : Node τ) (a : Audit) (Wr : ℝ) : ℝ :=
  a.b * (N.rebuild * a.T - Wr) / (a.cWord * N.declared a.T)

noncomputable def muHost (N : Node τ) (a : Audit) : ℝ :=
  ((N.capacity a.T : ℝ) - N.declared a.T) / N.capacity a.T

/-- μ_link: `(β_in + β_out · b / z_b) · T / C`: words carried in over the boundary pay one bit each, and a bound `z`
carried out unlocks `b / z_b` bits of reads outside the node. -/
noncomputable def muLink (N : Node τ) (a : Audit) : ℝ :=
  (N.linkIn + N.linkOut * a.b / a.zb) * a.T / N.declared a.T

noncomputable def muLambda (N : Node τ) (a : Audit) : ℝ := a.P * a.lam / N.declared a.T

noncomputable def muOut (N : Node τ) (a : Audit) (Φout Wtile : ℝ) : ℝ := a.b * Φout * a.T / (Wtile * N.declared a.T)

/-- γ under Identity (statement C) = μ_sample + μ_chip + μ_rebuild + μ_host + λ, at sampler error `ε` (μ_sample) and
certified rebuild work `W_r`. -/
noncomputable def gamma (N : Node τ) (a : Audit) (ε Wr : ℝ) : ℝ :=
  ε + muChip N a + muRebuild N a Wr + muHost N a + muLambda N a

/-- γ without Identity, for memory-only helpers and outside work at most `Φ_out` per tick: `gamma` + μ_link + μ_out,
at tile cost `W_tile`. -/
noncomputable def gammaLinks (N : Node τ) (a : Audit) (ε Wr Φout Wtile : ℝ) : ℝ :=
  gamma N a ε Wr + muLink N a + muOut N a Φout Wtile

end ResourceAccounting.Remote
