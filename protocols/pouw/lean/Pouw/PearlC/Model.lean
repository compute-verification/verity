import Pouw.PearlC.Game
import Mathlib.Data.Rat.Defs

/-!
# Pearl-C's accounting and domain (trusted)

Pearl-C is Pearl's output tickets on the noisy FP8 product plus #311's clean-up, pinned bit for bit and checked
(`docs/pouw/cheap-binding.md` in the Project store, §1). Costs are in H100 units, one unit being one dense FP8
multiply-accumulate on the tensor cores, at the measured H100 prices.

Per unit of shape `(m, k, n)`, the work the named assumption credits is
- the ticket chain, `m·k·n`, plus the promotion FADDs, `m·k·n / G` when the chain is promoted every `G` atoms;
- the A-side peel factor `A′·F_B`, fused as `r` extra columns: `m·k·r`;
- the peel combine, a BF16 product of depth `2r` at 2 units per MAC: `4·r·m·n`;
- the forming that depends on the salt (the noise product, the mix with the activation and the cast): `fs·m·k`.

The honest reference also pays the forming that depends on the activations alone, `qa·m·k`, which is not credited:
an adversary that chose its activations can skip it. The per-job share of `B̃·F_A` would enter both costs equally,
and the unscale is the model's own glue outside `W_ref`; leaving the first out can only raise the γ computed here.
-/

namespace Pouw.PearlC

/-- Pearl-C's accounting parameters. -/
structure Params where
  /-- the noise rank (32) -/
  r : ℕ
  /-- the promotion depth in atoms: the chain restarts from `+0` every `G` atoms into an FP32 total; `0` is none -/
  G : ℕ
  /-- forming that depends on the salt, H100 units per activation element (credited) -/
  fs : ℕ
  /-- forming that depends on the activations alone, per activation element (not credited) -/
  qa : ℕ

/-- The headline parameters: `r = 32`, no promotion, `fs = 112`, `qa = 16`. -/
def Params.headline : Params := ⟨32, 0, 112, 16⟩

/-- The headline parameters with promotion every 4 atoms. -/
def Params.headlineG4 : Params := ⟨32, 4, 112, 16⟩

/-- The credited work of one unit. -/
def creditOf (p : Params) (sh : Shape) : ℚ :=
  let mkn : ℚ := (sh.m : ℚ) * sh.k * sh.n
  mkn + (if p.G = 0 then 0 else mkn / p.G) + (sh.m : ℚ) * sh.k * p.r + 4 * (p.r : ℚ) * sh.m * sh.n +
    (p.fs : ℚ) * sh.m * sh.k

/-- The honest reference's work on one unit: the credited work plus the forming that depends on the activations alone. -/
def wrefOf (p : Params) (sh : Shape) : ℚ := creditOf p sh + (p.qa : ℚ) * sh.m * sh.k

/-- The credited work when only the ticket chain and the clean-up are credited (forming not credited at all). -/
def creditChainOnly (p : Params) (sh : Shape) : ℚ := creditOf p sh - (p.fs : ℚ) * sh.m * sh.k

/-- The reference overhead over the credited work at one shape. -/
def omegaOf (p : Params) (sh : Shape) : ℚ := wrefOf p sh / creditOf p sh

/-- The gap the reduction gives: `γ = 1 − (1 − γ₀)/Ω`. -/
def gammaOf (γ₀ Ω : ℚ) : ℚ := 1 - (1 - γ₀) / Ω

/-- `Π`'s costs are Pearl-C's accounting at parameters `p`. -/
def PearlCAccounting {Q R S : Type} (P : Protocol Q R S) (p : Params) : Prop :=
  ∀ (L : Layout) (u : ℕ), P.credit L u = (creditOf p (L.shape u) : ℝ) ∧ P.Wref L u = (wrefOf p (L.shape u) : ℝ)

/-- Pearl-C's shape domain. `k` is a positive multiple of 32 up to `2^16`. Without promotion the chain must have
`k ≤ 2^15`: at δ = 1 a single aligned spike freezes the truncating accumulator's tail once `k/δ² ≥ 2^16`
(`internal/pouw/cheap-binding/red-team-and-census.md` §3); promotion every 4 atoms confines it. The noise law
(δ = 1, the seeds, the promotion the chain uses) lives in the protocol's reference functions. -/
def pearlCDomain (p : Params) (L : Layout) : Prop :=
  ∀ u < L.N, 0 < (L.shape u).m ∧ 0 < (L.shape u).n ∧ 0 < (L.shape u).k ∧ 32 ∣ (L.shape u).k ∧
    (L.shape u).k ≤ 2 ^ 16 ∧ (p.G = 4 ∨ (p.G = 0 ∧ (L.shape u).k ≤ 2 ^ 15))

/-- Pearl-C's domain restricted to one shape `sh`. -/
def pearlCDomainAt (p : Params) (sh : Shape) (L : Layout) : Prop :=
  pearlCDomain p L ∧ ∀ u < L.N, L.shape u = sh

end Pouw.PearlC
