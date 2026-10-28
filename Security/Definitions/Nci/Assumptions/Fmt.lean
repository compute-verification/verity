import Definitions.Nci.Work
import Mathlib.Analysis.SpecialFunctions.Pow.Real

/-!
# NCI over threads, each thread charged by the work done on it at a format's rate: the named assumption

`WeightsSupport` charges each novel thread value a subcircuit reads, and fails at every format where reads are
redundant. `WeightsSupportFmt` charges each thread by the work done on it, in thread values (`Threading.work`), at the
rate of the activations' format less a margin.
-/

namespace Nci.Assumptions

open Verity

/-- **Weights are incompressible in support form, each thread charged by the work done on it, at a format's rate**:
every subcircuit `R` of the reference circuit `c` small by information (at most `Γ` computed gates, at most `S` bits of
information read) has novel reads that take at least
`2^(b_w · #(weights R uses) + (log₂ n − μ) · Σ_t min(d, m_t / e))` distinct values over the runs with the baseline at
`θv`, where `m_t` is the number of thread `t`'s counted multiplications in `R`. `n` is the number of values of the
activations' format (E4M3 253, BF16 65281, E2M1 15) and `μ` a margin; only `log₂ n − μ` enters, the bits charged per
thread value's worth of work.

It is `WeightsSupport` with each thread's reads replaced by the fewest that `(d, e)`-threading asks of `R`'s work on
the thread, so `WeightsSupport` at rate `log₂ n − μ` and `ThreadedInfo` at `(d, e)` give it
(`Nci.SecurityProofs.weightsSupportFmt_of_weightsSupport`), and the threading premise is part of it rather than a
hypothesis beside it. Reads are charged only through work, so redundant reads cost nothing: a value and a copy of it,
values another determines, products of weights and an activation read by gates that hold none, a weight-free product.
Those are where `WeightsSupport` fails, at every format: under the standard threading the gates reading a 128 × 128 × 2
tile of fprop products are charged `b_a · 32768` bits against at most 4096 bits of values at E4M3.

Where `R`'s counted multiplications read its novel reads directly, a value at its format's full support pays for its
work when `e` is at least the most counted multiplications one activation feeds: under `benchmarks/nci/kappa.py` that
is a dense layer's `2 · 18432` against `e = 4·10⁴`, with the LM head's forward products, `163840` per value, left out
of the counted multiplications. It fails where `WeightsSupport`'s width form does (weights committed in `θ`); where
counted multiplications read activations that take fewer than `2^(log₂ n − μ)` values (a one-hot embedding matmul); and
where one activation, through its copies, feeds more than `e · log₂ n / (log₂ n − μ)` counted multiplications in a
subcircuit with many threads (the LM head's forward products, were they counted; an MoE that runs every expert on
masked copies of its input). At the deployed point (`d = 2048`, `e = 4·10⁴`, E4M3, `log₂ n − μ = 4`) such a value is
charged at most `4 · 36864 / 4·10⁴ = 3.69` bits against the 7.98 it carries; a reference circuit that recomputed a dense
layer's activations, which `kappa.py`'s minimal one does not, would be charged 7.37 bits for two copies of each.

**Falsified** at the deployed point (`d = 2048`, `e = 4·10⁴`,
E4M3 at rate 4, `S = 2·10⁷`, `b_w = 2.5`), on its weight term and its margin. One step after a checkpoint, an AdamW
weight lies in an interval `θ` fixes: one token's 16,384 fprop products on a 128 × 128 E4M3 block are charged 40,961.6
bits against at most 29,113, false for `b_w > 1.78` at σ = 0.006 and over the first 1, 3 and 7 steps at σ = 0.006, 0.01
and 0.02. A weight quantized for fprop and again for dgrad is two weight gates charged `b_w` each for one weight's
values: an NVFP4 16 × 16 2D-block weight is charged 1,280 bits against at most 1,269, false for every `b_w > 2.153`.
llama-3-405b's gate and up projections read each activation 106,496 times, above `e · log₂ n / 4`: its FP8 run is
charged 2.077·10⁷ bits against at most 1.947·10⁷. `WeightsSupportFmt2` restates it per source, with a fan-out bound in
place of `μ` and the counted multiplications off weights fewer than `k₀` steps from `θ`. -/
def WeightsSupportFmt (c : RegisteredCircuit) (θv : c.D.Transcript) (T : Threading c) (Γ : ℕ) (S b_w : ℝ) (d e n : ℕ)
    (μ : ℝ) : Prop :=
  ∀ R, SmallInfo c θv Γ S R →
    (2 : ℝ) ^ (b_w * (T.weightsUsed R).card + (Real.logb 2 n - μ) * ∑ t, T.work d e R t) ≤
      support c θv (c.D.into R \ c.θ)

end Nci.Assumptions
