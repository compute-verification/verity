import Definitions.Nci.Sources
import Mathlib.Analysis.SpecialFunctions.Pow.Real

/-!
# NCI over threads, each value charged once over its copies: the named assumption

`WeightsSupportFmt` is false at the deployed point: on weights a few
AdamW steps past the checkpoint, on a weight's fprop and dgrad copies, and at llama-3-405b's fan-out.
`WeightsSupportFmt2` restates it with weights charged per source, a fan-out bound in place of the margin, `e` per model,
and the counted multiplications off weights fewer than `k₀` steps from the baseline. It is not signed off.
-/

namespace Nci.Assumptions

open Verity

/-- **Weights are incompressible in support form, each value charged once over its copies**: if no value's fan-out over
its copies exceeds, at `b_a` per `e` counted multiplications, the `log₂ n` bits a value of the activations' format
carries (`Sources.FanOutBound`), and every counted multiplication reads a weight at least `k₀` optimizer steps from the
baseline (`Sources.OffFresh`), then every subcircuit `R` of the reference circuit `c` small by information has novel
reads that take at least `2^(b_w · #(sources of the weights R uses) + b_a · Σ_t min(d, m_t / e))` distinct values over
the runs with the baseline at `θv`, where `m_t` is the number of thread `t`'s counted multiplications in `R`.

It is `WeightsSupportFmt` with four changes, at the deployed point (`d = 2048`, E4M3 at `b_a = 4`, `S = 2·10⁷`,
`b_w = 2.5`) and under `benchmarks/nci/kappa.py`'s minimal reference circuits:

* Weights are charged per source, so a weight's fprop and dgrad copies are charged once. An NVFP4 16 × 16 2D-block
  weight block is then charged 640 bits against at most 1,269, false only for `b_w > 4.305`, the format's own
  `(1024 + 7 + 71) / 256` bits a weight with the token's activations; a cached FP8 transpose only for `b_w > 8.11`.
* The margin gives way to the fan-out bound, `F(s)` the counted multiplications that read any copy of `s`: a value read
  only by counted multiplications is charged at most `log₂ n` over all its threads. For a block-scaled format `n` is
  taken below the format's count by what the block's scale costs an entry: at most 0.008 bits for E4M3 in 1 × 128
  tiles or 32-blocks, so `n = 251`, and 0.0096 for NVFP4's E2M1, so `n = 14`.
* `e` is the model's: the bound needs `e ≥ b_a · F_max / log₂ n`, at E4M3 (`n = 253`, and at `n = 251`) 18,664
  (18,691) for kimi-k2, 18,600 (18,627) for deepseek-v3, 12,379 (12,397) for qwen3-235b-a22b and 53,362 (53,439) for
  llama-3-405b, whose gate and up projections read each activation 106,496 times. At that `e` llama's FP8
  gate-and-up subcircuit over 300,000 tokens is charged 1.597·10⁷ bits against at most 1.947·10⁷; three copies of
  each of deepseek-v3's dense inputs would need 55,414 (55,494).
* It claims nothing of counted multiplications on weights fewer than `k₀` AdamW steps past `θ`, which `θ` pins to an
  interval. Under the interval model one token's fprop on a DeepSeek-V3 128 × 128 E4M3 block refutes
  it below `k₀` = 2, 4 and 7 steps at lr 2.2·10⁻⁴ and σ = 0.006, 0.01 and 0.02 (4, 5 and 9 when `|m₀|/√v₀` is
  gradient noise's 0.23), below 18 to 61 at 2.2·10⁻⁵, 53 to 175 at 7.3·10⁻⁶, and 83 to 151 in warmup from the
  initialization. A subcircuit that reads the block's copies at several consecutive steps, one token a step, needs
  more, since each step's copy is its own source charged `b_w` and a step moves an FP8 weight by fewer bits: over
  chains of 1 to 64 steps the least `k₀` no chain refutes is 3, 6 and 13 to 16 at lr 2.2·10⁻⁴ (17 to 20 at noise's
  0.23 and σ = 0.02), 61 to 80 and 201 to 300 at 2.2·10⁻⁵ (σ = 0.006 and 0.02), 151 to 250 and 801 to 1,200 at
  7.3·10⁻⁶, and 151 to 200 and 301 to 400 in warmup. `k₀` is a model's and its schedule's, and the first `k₀` steps
  after each checkpoint leave the count.

It fails at a `k₀` below those. The chain values bound the support from above by a worst-case reachable interval per
step, so they refute soundly and their not refuting proves nothing; chains longer than 64 steps are untried, and at
7.3·10⁻⁶ and σ = 0.02 the `b_w` above which a chain refutes it still falls slowly at 64 steps. It fails where counted
multiplications read activations that take fewer than `n` values (a one-hot embedding matmul), and the fan-out bound
pays only for values counted multiplications read directly: a subcircuit that computes its activations from fewer novel
reads pays only by those. -/
def WeightsSupportFmt2 (c : Registered) (θv : c.D.Transcript) (T : Threading c) (Src : T.Sources) (Γ : ℕ)
    (S b_w b_a : ℝ) (d e n k₀ : ℕ) : Prop :=
  Src.FanOutBound b_a e n → Src.OffFresh k₀ →
    ∀ R, SmallInfo c θv Γ S R →
      (2 : ℝ) ^ (b_w * (Src.sourcesUsed R).card + b_a * ∑ t, T.work d e R t) ≤
        support c θv (c.D.into R \ c.θ)

end Nci.Assumptions
