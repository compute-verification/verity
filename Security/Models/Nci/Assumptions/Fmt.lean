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

It fails where `WeightsSupport`'s width form does (weights committed in `θ`); where counted multiplications read
activations that take fewer than `2^(log₂ n − μ)` values (a one-hot embedding matmul); and where one activation, through
its copies, feeds more than `e · log₂ n / (log₂ n − μ)` counted multiplications in a subcircuit with many threads (an MoE
that runs every expert on masked copies of its input). The tightest case found at the deployed point (`d = 2048`,
`e = 4·10⁴`, E4M3, `log₂ n − μ = 4`) is activation recompute at a dense layer: two copies of each input, each a GEMM
operand of `2 · 18432` multiplications, charged 7.37 bits against the 7.98 one value carries. -/
def WeightsSupportFmt (c : Declaration) (θv : c.D.Transcript) (T : Threading c) (Γ : ℕ) (S b_w : ℝ) (d e n : ℕ)
    (μ : ℝ) : Prop :=
  ∀ R, SmallInfo c θv Γ S R →
    (2 : ℝ) ^ (b_w * (T.weightsUsed R).card + (Real.logb 2 n - μ) * ∑ t, T.work d e R t) ≤
      support c θv (c.D.into R \ c.θ)

end Nci.Assumptions
