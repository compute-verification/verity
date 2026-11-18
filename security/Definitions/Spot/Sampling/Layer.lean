module

public import Definitions.Core.Draw

/-!
# The sampling layer's theorem, stated

**Takes** a sampling design `L` over the units `Fin n` (a partition's proof blocks; one level's design, or the design a
levels sample induces on the units), the session the argument system runs on each sample after the registration
`Reg`, and that session's analysis: transcripts, the incorrect set of each, the committed transcript, and per sampled
unit the failure probabilities `ks` (accepts, extraction fails) and `link` (extraction disagrees with the committed
transcript), covering acceptance at each sampled incorrect unit. **Gives** `Pr[accept ∧ E(X, S)] ≤
Pr_S[E(X, S) ∧ S misses incorrect(X)] + ε_ks + ε_bind`, `X` the committed transcript: `ε_ks` the argument system's
knowledge error at a unit picked before the session's coins, averaged over the sample, `ε_bind` the commitment's binding
error; the first term is at most `L.miss K` at `E = (K ≤ |incorrect X|)`.

`Sound` is `FlockSoundness.Audit.extraction_audit_le`'s statement with `ExtractionAnalysis`'s fields as binders and
`audit` unfolded, so `extraction_audit_le ⟨Tr, inc, com, ks, link, cover⟩ hks hlink E σ` proves it.

**Instances.** `Flock.Guarantees.SampledKSLiveV5` and `SampledKSLiveV3`, deployed and proved, apply
`extraction_audit_le` (V3 also `extraction_audit_count`, which `Sound` does not state). `Claims.SamplingGeneric.CompilerKS`
and `FlockSoundness.SampledGeneric` state the layer apart and prove it; nothing deployed instantiates either.
-/

@[expose] public section

namespace Spot.Sampling

open Verity Verity.Game
open scoped ENNReal

/-- **The sampling layer's theorem** (stated). For every design `L` over `n` units, session, and analysis whose `ks` and
`link` cover acceptance at every sampled incorrect unit, with `ks` and `link` bounded at the unit any rule `tgt` picks
before the session's first coin by `εks` and `εbind` on average over the sample: every audit strategy accepts with an
event `E` of the committed transcript and the sample with probability at most the sample's probability of `E` while
missing every incorrect unit, plus `εks` and `εbind` at the strategy's registration and state. Proved by
`FlockSoundness.Audit.extraction_audit_le`. -/
def Sound : Prop :=
  ∀ (n : ℕ) (L : SamplingDesign (Fin n)) (Reg : Type) (session : Finset (Fin n) → Reg → Game Bool)
    (Tr : Type) (incorrect : Tr → Finset (Fin n))
    (committed : (R : Reg) → ((ω : L.Ω) → Strategy (session (L.draw ω) R)) → Tr)
    (ks : (S : Finset (Fin n)) → (R : Reg) → Strategy (session S R) → Fin n → ℝ≥0∞)
    (link : (S : Finset (Fin n)) → (R : Reg) → Strategy (session S R) → Fin n → Tr → ℝ≥0∞),
    (∀ S R (τ : Strategy (session S R)) u X, u ∈ S → u ∈ incorrect X →
      prob (· = true) (session S R) τ ≤ ks S R τ u + link S R τ u X) →
    ∀ (εks εbind : (R : Reg) → ((ω : L.Ω) → Strategy (session (L.draw ω) R)) → ℝ≥0∞),
    (∀ R (τ : (ω : L.Ω) → Strategy (session (L.draw ω) R)) (tgt : L.Ω → Option (Fin n)),
      avg (fun ω => (tgt ω).elim 0 fun u => if u ∈ L.draw ω then ks (L.draw ω) R (τ ω) u else 0) ≤ εks R τ) →
    (∀ R (τ : (ω : L.Ω) → Strategy (session (L.draw ω) R)) (tgt : L.Ω → Option (Fin n)),
      avg (fun ω => (tgt ω).elim 0 fun u =>
        if u ∈ L.draw ω then link (L.draw ω) R (τ ω) u (committed R τ) else 0) ≤ εbind R τ) →
    ∀ (E : Tr → Finset (Fin n) → Prop)
      (σ : Strategy (Game.send Reg fun R => Game.coin L.Ω fun ω =>
        (session (L.draw ω) R).map fun ok => (ok, L.draw ω))),
      let cont : (ω : L.Ω) → Strategy (session (L.draw ω) σ.1) := fun ω =>
        Strategy.ofMap (fun ok => (ok, L.draw ω)) (session (L.draw ω) σ.1) (σ.2 ω)
      let X := committed σ.1 cont
      prob (fun o => o.1 = true ∧ E X o.2)
          (Game.send Reg fun R => Game.coin L.Ω fun ω => (session (L.draw ω) R).map fun ok => (ok, L.draw ω)) σ ≤
        prCoin (fun ω => E X (L.draw ω) ∧ Disjoint (L.draw ω) (incorrect X)) + εks σ.1 cont + εbind σ.1 cont

end Spot.Sampling
