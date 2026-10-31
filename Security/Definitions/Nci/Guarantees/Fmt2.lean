import Definitions.Nci.Assumptions.Fmt2

/-!
# NCI over threads, each value charged once over its copies: the guarantees

`ThreadedReadsInfoFmt`, `ThreadedContainedInfoFmt` and `ThreadedOfStateFmt` restated under `WeightsSupportFmt2`: Theorem
1 with reads counted by information (`ThreadedReadsInfoFmt2`), through a containment (`ThreadedContainedInfoFmt2`), and
for a set read by a state of `k` bits and an ingress (`ThreadedOfStateFmt2`). Each holds for a reference circuit whose
sources meet the fan-out bound at `(b_a, e, n)` and keep its counted multiplications `k₀` steps off the baseline.
-/

namespace Nci.Guarantees

open Verity Finset

/-- **Theorem 1 with reads counted by information, each value charged once over its copies**: if the reference circuit
`c`'s values meet the fan-out bound, its counted multiplications read weights at least `k₀` steps from the baseline,
and its weights are incompressible with each source charged once and each thread charged by its work at `b_a`, every
subcircuit `R` small by information computes at most `Λ = max(e, c · S / (4 d b_w))` counted multiplications per `b_a`
bits of information its novel reads carry. -/
def ThreadedReadsInfoFmt2 : Prop :=
  ∀ (c : RegisteredCircuit) (θv : c.D.Transcript) (T : Threading c) (Src : T.Sources) (Γ : ℕ) (S b_w b_a : ℝ)
    (d e n k₀ : ℕ),
    0 < b_w → 0 < b_a → 0 < d → 0 < e → Src.FanOutBound b_a e n → Src.OffFresh k₀ →
    Assumptions.WeightsSupportFmt2 c θv T Src Γ S b_w b_a d e n k₀ →
    ∀ R, SmallInfo c θv Γ S R → b_a * (T.mults R).card ≤ novelInfo c θv R * T.bound S b_w d e

/-- **Theorem 1 through a containment, each value charged once over its copies**: a declaration `d` contains the
reference circuit `c` through `M`, with `c`'s baseline values `θc` the ones `θv` gives their images, and `c` is as in
`ThreadedReadsInfoFmt2` at cap `K`. Every set `W` of `d`'s computed gates, at most `Γ` of them, whose novel reads carry
at most `K` bits of information, holds at most `Λ` of `c`'s counted multiplications per `b_a` bits of information `W`'s
novel reads carry. -/
def ThreadedContainedInfoFmt2 : Prop :=
  ∀ (c d : RegisteredCircuit) (M : Containment c.D d.D) (θc : c.D.Transcript) (θv : d.D.Transcript) (T : Threading c)
    (Src : T.Sources) (Γ : ℕ) (K b_w b_a : ℝ) (dd e n k₀ : ℕ),
    (∀ g, c.D.isInput g = true → M.emb g ∈ d.θ → g ∈ c.θ) →
    (∀ g ∈ c.θ, M.emb g ∈ d.θ → (θv (M.emb g)).cast (M.emb_w g) = θc g) →
    0 < b_w → 0 < b_a → 0 < dd → 0 < e → Src.FanOutBound b_a e n → Src.OffFresh k₀ →
    Assumptions.WeightsSupportFmt2 c θc T Src Γ K b_w b_a dd e n k₀ →
    ∀ W, W ⊆ d.D.computed → d.D.size W ≤ Γ → novelInfo d θv W ≤ K →
      b_a * (T.mults (univ.filter fun g => M.emb g ∈ W)).card ≤ novelInfo d θv W * T.bound K b_w dd e

/-- **The count a memory cap needs, each value charged once over its copies**: with `d`, `c` and `M` as in
`ThreadedContainedInfoFmt2`, a set `W` of `d`'s computed gates, at most `Γ` of them, whose novel reads are a state that
is a function of `k` bits and an ingress `Sin`, holds at most `K · Λ(K) / b_a` of `c`'s counted multiplications, at
`K = k + width(Sin)`. -/
def ThreadedOfStateFmt2 : Prop :=
  ∀ (c d : RegisteredCircuit) (M : Containment c.D d.D) (θc : c.D.Transcript) (θv : d.D.Transcript) (T : Threading c)
    (Src : T.Sources) (Γ : ℕ) (b_w b_a : ℝ) (dd e n k₀ : ℕ) (W Sst Sin : Finset (Fin d.D.N)) (k : ℕ)
    (f : BitVec k → (g : Sst) → BitVec (d.D.w g)),
    (∀ g, c.D.isInput g = true → M.emb g ∈ d.θ → g ∈ c.θ) →
    (∀ g ∈ c.θ, M.emb g ∈ d.θ → (θv (M.emb g)).cast (M.emb_w g) = θc g) →
    0 < b_w → 0 < b_a → 0 < dd → 0 < e → Src.FanOutBound b_a e n → Src.OffFresh k₀ →
    Assumptions.WeightsSupportFmt2 c θc T Src Γ (k + d.D.width Sin) b_w b_a dd e n k₀ →
    W ⊆ d.D.computed → d.D.size W ≤ Γ → d.D.into W \ d.θ ⊆ Sst ∪ Sin →
    (∀ a ∈ admissible d θv, ∃ s, values Sst a = f s) →
      b_a * (T.mults (univ.filter fun g => M.emb g ∈ W)).card ≤
        (k + d.D.width Sin) * T.bound (k + d.D.width Sin) b_w dd e

end Nci.Guarantees
