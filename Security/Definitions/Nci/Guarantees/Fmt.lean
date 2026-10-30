import Definitions.Nci.Assumptions.Fmt
import Definitions.Nci.Assumptions.Info

/-!
# The inference-only policy over threads, each thread charged by its work at a format's rate: the guarantees

`ThreadedReadsInfo`, `ThreadedContainedInfo` and `ThreadedOfState` restated under `WeightsSupportFmt`, which takes the
place of both `WeightsSupport` and the threading premise `ThreadedInfo`: Theorem 1 with reads counted by information
(`ThreadedReadsInfoFmt`), through a containment (`ThreadedContainedInfoFmt`), and for a set read by a state of `k` bits
and an ingress (`ThreadedOfStateFmt`). The rate `b_a` is at most the format's `log₂ n` less the margin `μ`.
-/

namespace Nci.Guarantees

open Verity Finset

/-- **Theorem 1 with reads counted by information, each thread charged by its work**: if the reference circuit `r`'s
weights are incompressible with each thread charged by the work done on it, at `log₂ n − μ ≥ b_a` bits per thread
value's worth, every subcircuit `R` small by information computes at most `Λ = max(e, c · S / (4 d b_w))` counted
multiplications per `b_a` bits of information its fresh input reads carry. It is `ThreadedReadsInfo` with
`WeightsSupportFmt` in place of `WeightsSupport` and the threading premise. -/
def ThreadedReadsInfoFmt : Prop :=
  ∀ (r : Declaration) (θv : r.D.Transcript) (T : Threading r) (c : ℕ) (S b_w b_a : ℝ) (d e n : ℕ) (μ : ℝ),
    0 < b_w → 0 < b_a → 0 < d → 0 < e → b_a ≤ Real.logb 2 n - μ →
    Assumptions.WeightsSupportFmt r θv T c S b_w d e n μ →
    ∀ R, SmallInfo r θv c S R → b_a * (T.mults R).card ≤ novelInfo r θv R * T.bound S b_w d e

/-- **Theorem 1 through a containment, each thread charged by its work**: a declaration `d` contains the reference
circuit `r` through `M`, with `r`'s registered values `θc` the ones `θv` gives their images, and `r` is as in
`ThreadedReadsInfoFmt` at cap `K`. Every set `W` of `d`'s computed gates, at most `c` of them, whose fresh input reads
carry at most `K` bits of information, holds at most `Λ` of `r`'s counted multiplications per `b_a` bits of information
`W`'s fresh input reads carry. -/
def ThreadedContainedInfoFmt : Prop :=
  ∀ (r d : Declaration) (M : Containment r.D d.D) (θc : r.D.Transcript) (θv : d.D.Transcript) (T : Threading r)
    (c : ℕ) (K b_w b_a : ℝ) (dd e n : ℕ) (μ : ℝ),
    Assumptions.RegisteredThrough M θc θv →
    0 < b_w → 0 < b_a → 0 < dd → 0 < e → b_a ≤ Real.logb 2 n - μ →
    Assumptions.WeightsSupportFmt r θc T c K b_w dd e n μ →
    ∀ W, SmallInfo d θv c K W →
      b_a * (T.mults (univ.filter fun g => M.emb g ∈ W)).card ≤ novelInfo d θv W * T.bound K b_w dd e

/-- **The count a memory cap needs, each thread charged by its work**: with `d`, `r` and `M` as in
`ThreadedContainedInfoFmt`, a set `W` of `d`'s computed gates, at most `c` of them, whose fresh input reads are a state
that is a function of `k` bits and an ingress `Sin`, holds at most `K · Λ(K) / b_a` of `r`'s counted multiplications, at
`K = k + width(Sin)`. -/
def ThreadedOfStateFmt : Prop :=
  ∀ (r d : Declaration) (M : Containment r.D d.D) (θc : r.D.Transcript) (θv : d.D.Transcript) (T : Threading r)
    (c : ℕ) (b_w b_a : ℝ) (dd e n : ℕ) (μ : ℝ) (W Sst Sin : Finset (Fin d.D.N)) (k : ℕ)
    (f : BitVec k → (g : Sst) → BitVec (d.D.w g)),
    Assumptions.RegisteredThrough M θc θv →
    0 < b_w → 0 < b_a → 0 < dd → 0 < e → b_a ≤ Real.logb 2 n - μ →
    Assumptions.WeightsSupportFmt r θc T c (k + d.D.width Sin) b_w dd e n μ →
    Assumptions.WithinBlock d c W → d.D.into W \ d.θ ⊆ Sst ∪ Sin →
    (∀ a ∈ admissible d θv, ∃ s, values Sst a = f s) →
      b_a * (T.mults (univ.filter fun g => M.emb g ∈ W)).card ≤
        (k + d.D.width Sin) * T.bound (k + d.D.width Sin) b_w dd e

end Nci.Guarantees
