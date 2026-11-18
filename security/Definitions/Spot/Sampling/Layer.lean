module

public import Definitions.Spot.Sampling.Analysis

/-!
# The sampling layer's theorem, stated

**Takes** a sampling design `L` over any units with decidable equality (one partition's proof blocks, or a
`Verity.Levels` design's gate sets), the session the argument system runs on each sample after the registration
`Reg`, and that session's analysis: transcripts, the incorrect set of each, the committed transcript, and per sampled
unit the failure probabilities `ks` (accepts, extraction fails) and `link` (extraction disagrees with the committed
transcript), covering acceptance at each sampled incorrect unit. **Gives** `Pr[accept ∧ E(X, S)] ≤
Pr_S[E(X, S) ∧ S misses incorrect(X)] + ε_ks + ε_bind`, `X` the committed transcript: `ε_ks` the argument system's
knowledge error at a unit picked before the session's coins, averaged over the sample, `ε_bind` the commitment's binding
error; the first term is at most `L.miss K` at `E = (K ≤ |incorrect X|)`.

`Sound` is `Spot.Sampling.extraction_audit_le`'s statement, over any units, so `extraction_audit_le A hks hlink E σ`
proves it.

**Instances.** `Flock.Guarantees.SampledKSLiveV5` and `SampledKSLiveV3`, deployed and proved, apply
`extraction_audit_le` (V3 also `extraction_audit_count`, which `Sound` does not state). `Claims.SamplingGeneric.CompilerKS`
and `FlockSoundness.SampledGeneric` state the layer apart and prove it; nothing deployed instantiates either.
-/

@[expose] public section

namespace Spot.Sampling

open Verity Verity.Game
open scoped ENNReal

/-- **The sampling layer's theorem** (stated). For every design `L` over any units, session, and analysis whose `ks` and
`link` cover acceptance at every sampled incorrect unit, with `ks` and `link` bounded at the unit any rule `tgt` picks
before the session's first coin by `εks` and `εbind` on average over the sample: every audit strategy accepts with an
event `E` of the committed transcript and the sample with probability at most the sample's probability of `E` while
missing every incorrect unit, plus `εks` and `εbind` at the strategy's registration and state. Proved by
`Spot.Sampling.extraction_audit_le`. -/
def Sound : Prop :=
  ∀ {U : Type} [DecidableEq U] (L : Verity.SamplingDesign U) (Reg : Type) (session : Finset U → Reg → Game Bool)
    (A : ExtractionAnalysis L Reg session) (εks εbind : (R : Reg) → Cont L Reg session R → ℝ≥0∞),
    A.KnowledgeSound εks → A.BindSound εbind → ∀ (E : A.Tr → Finset U → Prop) (σ : Strategy (audit L Reg session)),
    prob (fun o => o.1 = true ∧ E (A.committedOf σ) o.2) (audit L Reg session) σ ≤
      prCoin (fun ω => E (A.committedOf σ) (L.draw ω) ∧ Disjoint (L.draw ω) (A.incorrect (A.committedOf σ))) +
        εks (reg σ) (cont σ) + εbind (reg σ) (cont σ)

/-- **The sampling layer's count curve** (stated): with `Sound`'s takes, every audit strategy accepts with at least `K`
incorrect units in its committed transcript with probability at most `L.miss K` plus `εks` and `εbind`. Proved by
`Spot.Sampling.extraction_audit_count`. -/
def SoundCount : Prop :=
  ∀ {U : Type} [DecidableEq U] (L : Verity.SamplingDesign U) (Reg : Type) (session : Finset U → Reg → Game Bool)
    (A : ExtractionAnalysis L Reg session) (εks εbind : (R : Reg) → Cont L Reg session R → ℝ≥0∞),
    A.KnowledgeSound εks → A.BindSound εbind → ∀ (K : ℕ) (σ : Strategy (audit L Reg session)),
    prob (fun o => o.1 = true ∧ K ≤ (A.incorrect (A.committedOf σ)).card) (audit L Reg session) σ ≤
      L.miss K + εks (reg σ) (cont σ) + εbind (reg σ) (cont σ)

end Spot.Sampling
