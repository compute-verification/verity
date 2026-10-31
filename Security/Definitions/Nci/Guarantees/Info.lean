import Definitions.Nci.Assumptions.Info

/-!
# The inference-only policy over threads, reads counted by information: the guarantees

Theorem 1 of the thread draft with reads counted by information (`ThreadedReadsInfo`), carried through a containment to
any set of gates of a declaration that contains the reference circuit (`ThreadedContainedInfo`). A set whose fresh input
reads are a state that is a function of `k` bits and an ingress counted by width reads at most `k + width(ingress)` bits
of information (`InfoOfState`), so it holds at most `K · Λ(K) / b_a` counted multiplications at
`K = k + width(ingress)` (`ThreadedOfState`).
-/

namespace Nci.Guarantees

open Verity Finset

/-- **Theorem 1 with reads counted by information**: if the reference circuit `r` is `(d, e)`-threaded over the
subcircuits small by information and its weights are incompressible in support form, every such subcircuit `R`
computes at most `Λ = max(e, c · S / (4 d b_w))` counted multiplications per `b_a` bits of information its fresh input
reads carry. It is `ThreadedReads` with `novelInfo`, which is at most `novelReads`, in place of `novelReads`. -/
def ThreadedReadsInfo : Prop :=
  ∀ (r : RegisteredCircuit) (θv : r.D.Transcript) (T : Threading r) (c : ℕ) (S b_w b_a : ℝ) (d e : ℕ),
    0 < b_w → 0 < b_a → 0 < d → T.ThreadedInfo θv c S d e → Assumptions.WeightsSupport r θv T c S b_w b_a →
    ∀ R, SmallInfo r θv c S R → b_a * (T.mults R).card ≤ novelInfo r θv R * T.bound S b_w d e

/-- **Reads bounded by a state**: if a set `R` of computed gates reads, outside the registered information, only gates
of a state `Sst` and an ingress `Sin`, and the state's values over the runs with the registered information at `θv` are
a function `f` of `k` bits, then `R`'s gates take at most `2^(k + width(Sin))` values and its fresh input reads carry at
most `k + width(Sin)` bits of information. The baseline is what is registered: after a wipe, a state that is registered
data entangled with at most `k` bits of slack reads as `k` bits, however wide it is. -/
def InfoOfState : Prop :=
  ∀ (D : RegisteredCircuit) (θv : D.D.Transcript) (R Sst Sin : Finset (Fin D.D.N)) (k : ℕ)
    (f : BitVec k → (g : Sst) → BitVec (D.D.w g)),
    R ⊆ D.D.computed → D.D.into R \ D.θ ⊆ Sst ∪ Sin → (∀ a ∈ admissible D θv, ∃ s, values Sst a = f s) →
      support D θv R ≤ 2 ^ (k + D.D.width Sin) ∧ novelInfo D θv R ≤ k + D.D.width Sin

/-- **Theorem 1 through a containment**: a declaration `d` contains the reference circuit `r` through `M`, with `r`'s
registered values `θc` the ones `θv` gives their images, and `r` is as in `ThreadedReadsInfo` at cap `K`. Every set `W`
of `d`'s computed gates, at most `c` of them, whose fresh input reads carry at most `K` bits of information, holds at
most `Λ` of `r`'s counted multiplications per `b_a` bits of information `W`'s fresh input reads carry. -/
def ThreadedContainedInfo : Prop :=
  ∀ (r d : RegisteredCircuit) (M : Containment r.D d.D) (θc : r.D.Transcript) (θv : d.D.Transcript) (T : Threading r)
    (c : ℕ) (K b_w b_a : ℝ) (dd e : ℕ),
    Assumptions.RegisteredThrough M θc θv →
    0 < b_w → 0 < b_a → 0 < dd → T.ThreadedInfo θc c K dd e → Assumptions.WeightsSupport r θc T c K b_w b_a →
    ∀ W, SmallInfo d θv c K W →
      b_a * (T.mults (univ.filter fun g => M.emb g ∈ W)).card ≤ novelInfo d θv W * T.bound K b_w dd e

/-- **The count a memory cap needs**: with `d`, `r` and `M` as in `ThreadedContainedInfo`, a set `W` of `d`'s computed
gates, at most `c` of them, whose fresh input reads are a state that is a function of `k` bits and an ingress `Sin`,
holds at most `K · Λ(K) / b_a` of `r`'s counted multiplications, at `K = k + width(Sin)`. -/
def ThreadedOfState : Prop :=
  ∀ (r d : RegisteredCircuit) (M : Containment r.D d.D) (θc : r.D.Transcript) (θv : d.D.Transcript) (T : Threading r)
    (c : ℕ) (b_w b_a : ℝ) (dd e : ℕ) (W Sst Sin : Finset (Fin d.D.N)) (k : ℕ)
    (f : BitVec k → (g : Sst) → BitVec (d.D.w g)),
    Assumptions.RegisteredThrough M θc θv →
    0 < b_w → 0 < b_a → 0 < dd → T.ThreadedInfo θc c (k + d.D.width Sin) dd e →
    Assumptions.WeightsSupport r θc T c (k + d.D.width Sin) b_w b_a →
    Assumptions.WithinBlock d c W → d.D.into W \ d.θ ⊆ Sst ∪ Sin →
    (∀ a ∈ admissible d θv, ∃ s, values Sst a = f s) →
      b_a * (T.mults (univ.filter fun g => M.emb g ∈ W)).card ≤
        (k + d.D.width Sin) * T.bound (k + d.D.width Sin) b_w dd e

end Nci.Guarantees
