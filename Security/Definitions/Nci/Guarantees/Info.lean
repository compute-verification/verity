import Definitions.Nci.Assumptions.Info

/-!
# NCI over threads, reads counted by information: the guarantees

Theorem 1 of the thread draft with reads counted by information (`ThreadedReadsInfo`), carried through a containment to
any set of gates of a declaration that contains the reference circuit (`ThreadedContainedInfo`). A set whose novel reads
are a state that is a function of `k` bits and an ingress counted by width reads at most `k + width(ingress)` bits of
information (`InfoOfState`), so it holds at most `K · Λ(K) / b_a` counted multiplications at
`K = k + width(ingress)` (`ThreadedOfState`).
-/

namespace Nci.Guarantees

open Verity Finset

/-- **Theorem 1 with reads counted by information**: if the reference circuit `c` is `(d, e)`-threaded over the
subcircuits small by information and its weights are incompressible in support form, every such subcircuit `R`
computes at most `Λ = max(e, c · S / (4 d b_w))` counted multiplications per `b_a` bits of information its novel reads
carry. It is `ThreadedReads` with `novelInfo`, which is at most `novelReads`, in place of `novelReads`. -/
def ThreadedReadsInfo : Prop :=
  ∀ (c : RegisteredCircuit) (θv : c.D.Transcript) (T : Threading c) (Γ : ℕ) (S b_w b_a : ℝ) (d e : ℕ),
    0 < b_w → 0 < b_a → 0 < d → T.ThreadedInfo θv Γ S d e → Assumptions.WeightsSupport c θv T Γ S b_w b_a →
    ∀ R, SmallInfo c θv Γ S R → b_a * (T.mults R).card ≤ novelInfo c θv R * T.bound S b_w d e

/-- **Reads bounded by a state**: if a set `R` of computed gates reads, outside the baseline, only gates of a state `Sst`
and an ingress `Sin`, and the state's values over the runs with the baseline at `θv` are a function `f` of `k` bits,
then `R`'s gates take at most `2^(k + width(Sin))` values and its novel reads carry at most `k + width(Sin)` bits of
information. The baseline is what is registered: after a wipe, a state that is registered data entangled with at most
`k` bits of slack reads as `k` bits, however wide it is. -/
def InfoOfState : Prop :=
  ∀ (D : RegisteredCircuit) (θv : D.D.Transcript) (R Sst Sin : Finset (Fin D.D.N)) (k : ℕ)
    (f : BitVec k → (g : Sst) → BitVec (D.D.w g)),
    R ⊆ D.D.computed → D.D.into R \ D.θ ⊆ Sst ∪ Sin → (∀ a ∈ admissible D θv, ∃ s, values Sst a = f s) →
      support D θv R ≤ 2 ^ (k + D.D.width Sin) ∧ novelInfo D θv R ≤ k + D.D.width Sin

/-- **Theorem 1 through a containment**: a declaration `d` contains the reference circuit `c` through `M`, with `c`'s
baseline values `θc` the ones `θv` gives their images, and `c` is as in `ThreadedReadsInfo` at cap `K`. Every set `W`
of `d`'s computed gates, at most `Γ` of them, whose novel reads carry at most `K` bits of information, holds at most `Λ`
of `c`'s counted multiplications per `b_a` bits of information `W`'s novel reads carry. -/
def ThreadedContainedInfo : Prop :=
  ∀ (c d : RegisteredCircuit) (M : Containment c.D d.D) (θc : c.D.Transcript) (θv : d.D.Transcript) (T : Threading c)
    (Γ : ℕ) (K b_w b_a : ℝ) (dd e : ℕ),
    (∀ g, c.D.isInput g = true → M.emb g ∈ d.θ → g ∈ c.θ) →
    (∀ g ∈ c.θ, M.emb g ∈ d.θ → (θv (M.emb g)).cast (M.emb_w g) = θc g) →
    0 < b_w → 0 < b_a → 0 < dd → T.ThreadedInfo θc Γ K dd e → Assumptions.WeightsSupport c θc T Γ K b_w b_a →
    ∀ W, W ⊆ d.D.computed → d.D.size W ≤ Γ → novelInfo d θv W ≤ K →
      b_a * (T.mults (univ.filter fun g => M.emb g ∈ W)).card ≤ novelInfo d θv W * T.bound K b_w dd e

/-- **The count a memory cap needs**: with `d`, `c` and `M` as in `ThreadedContainedInfo`, a set `W` of `d`'s computed
gates, at most `Γ` of them, whose novel reads are a state that is a function of `k` bits and an ingress `Sin`, holds at
most `K · Λ(K) / b_a` of `c`'s counted multiplications, at `K = k + width(Sin)`. -/
def ThreadedOfState : Prop :=
  ∀ (c d : RegisteredCircuit) (M : Containment c.D d.D) (θc : c.D.Transcript) (θv : d.D.Transcript) (T : Threading c)
    (Γ : ℕ) (b_w b_a : ℝ) (dd e : ℕ) (W Sst Sin : Finset (Fin d.D.N)) (k : ℕ)
    (f : BitVec k → (g : Sst) → BitVec (d.D.w g)),
    (∀ g, c.D.isInput g = true → M.emb g ∈ d.θ → g ∈ c.θ) →
    (∀ g ∈ c.θ, M.emb g ∈ d.θ → (θv (M.emb g)).cast (M.emb_w g) = θc g) →
    0 < b_w → 0 < b_a → 0 < dd → T.ThreadedInfo θc Γ (k + d.D.width Sin) dd e →
    Assumptions.WeightsSupport c θc T Γ (k + d.D.width Sin) b_w b_a →
    W ⊆ d.D.computed → d.D.size W ≤ Γ → d.D.into W \ d.θ ⊆ Sst ∪ Sin →
    (∀ a ∈ admissible d θv, ∃ s, values Sst a = f s) →
      b_a * (T.mults (univ.filter fun g => M.emb g ∈ W)).card ≤
        (k + d.D.width Sin) * T.bound (k + d.D.width Sin) b_w dd e

end Nci.Guarantees
