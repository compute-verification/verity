import Definitions.Nci.Assumptions.Fmt2
import Definitions.Nci.Assumptions.Info

/-!
# The inference-only policy over threads, each value charged once over its copies: the guarantees

`ThreadedReadsInfoFmt`, `ThreadedContainedInfoFmt` and `ThreadedOfStateFmt` restated under `WeightsSupportFmt2`: Theorem
1 with reads counted by information (`ThreadedReadsInfoFmt2`), through a containment (`ThreadedContainedInfoFmt2`), and
for a set read by a state of `k` bits and an ingress (`ThreadedOfStateFmt2`). Each holds for a reference circuit whose
sources meet the fan-out bound at `(b_a, e, n)` and keep its counted multiplications `k₀` steps off the registered
information.
-/

namespace Nci.Guarantees

open Verity Finset

/-- **Theorem 1 with reads counted by information, each value charged once over its copies**: if the reference circuit
`r`'s values meet the fan-out bound, its counted multiplications read weights at least `k₀` steps from the registered
information, and its weights are incompressible with each source charged once and each thread charged by its work at
`b_a`, every subcircuit `R` small by information computes at most `Λ = max(e, c · S / (4 d b_w))` counted
multiplications per `b_a` bits of information its fresh input reads carry. -/
def ThreadedReadsInfoFmt2 : Prop :=
  ∀ (r : RegisteredCircuit) (θv : r.D.Transcript) (T : Threading r) (Src : T.Sources) (c : ℕ) (S b_w b_a : ℝ)
    (d e n k₀ : ℕ),
    0 < b_w → 0 < b_a → 0 < d → 0 < e → Src.FanOutBound b_a e n → Src.OffFresh k₀ →
    Assumptions.WeightsSupportFmt2 r θv T Src c S b_w b_a d e n k₀ →
    ∀ R, SmallInfo r θv c S R → b_a * (T.mults R).card ≤ novelInfo r θv R * T.bound S b_w d e

/-- **Theorem 1 through a containment, each value charged once over its copies**: a declaration `d` contains the
reference circuit `r` through `M`, with `r`'s registered values `θc` the ones `θv` gives their images, and `r` is as in
`ThreadedReadsInfoFmt2` at cap `K`. Every set `W` of `d`'s computed gates, at most `c` of them, whose fresh input reads
carry at most `K` bits of information, holds at most `Λ` of `r`'s counted multiplications per `b_a` bits of information
`W`'s fresh input reads carry. -/
def ThreadedContainedInfoFmt2 : Prop :=
  ∀ (r d : RegisteredCircuit) (M : Containment r.D d.D) (θc : r.D.Transcript) (θv : d.D.Transcript) (T : Threading r)
    (Src : T.Sources) (c : ℕ) (K b_w b_a : ℝ) (dd e n k₀ : ℕ),
    Assumptions.RegisteredThrough M θc θv →
    0 < b_w → 0 < b_a → 0 < dd → 0 < e → Src.FanOutBound b_a e n → Src.OffFresh k₀ →
    Assumptions.WeightsSupportFmt2 r θc T Src c K b_w b_a dd e n k₀ →
    ∀ W, SmallInfo d θv c K W →
      b_a * (T.mults (univ.filter fun g => M.emb g ∈ W)).card ≤ novelInfo d θv W * T.bound K b_w dd e

/-- **The count a memory cap needs, each value charged once over its copies**: with `d`, `r` and `M` as in
`ThreadedContainedInfoFmt2`, a set `W` of `d`'s computed gates, at most `c` of them, whose fresh input reads are a state
that is a function of `k` bits and an ingress `Sin`, holds at most `K · Λ(K) / b_a` of `r`'s counted multiplications, at
`K = k + width(Sin)`. -/
def ThreadedOfStateFmt2 : Prop :=
  ∀ (r d : RegisteredCircuit) (M : Containment r.D d.D) (θc : r.D.Transcript) (θv : d.D.Transcript) (T : Threading r)
    (Src : T.Sources) (c : ℕ) (b_w b_a : ℝ) (dd e n k₀ : ℕ) (W Sst Sin : Finset (Fin d.D.N)) (k : ℕ)
    (f : BitVec k → (g : Sst) → BitVec (d.D.w g)),
    Assumptions.RegisteredThrough M θc θv →
    0 < b_w → 0 < b_a → 0 < dd → 0 < e → Src.FanOutBound b_a e n → Src.OffFresh k₀ →
    Assumptions.WeightsSupportFmt2 r θc T Src c (k + d.D.width Sin) b_w b_a dd e n k₀ →
    Assumptions.WithinBlock d c W → d.D.into W \ d.θ ⊆ Sst ∪ Sin →
    (∀ a ∈ admissible d θv, ∃ s, values Sst a = f s) →
      b_a * (T.mults (univ.filter fun g => M.emb g ∈ W)).card ≤
        (k + d.D.width Sin) * T.bound (k + d.D.width Sin) b_w dd e

end Nci.Guarantees
