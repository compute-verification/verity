module

public import Definitions.Core.Draw

/-!
# The sampling layer: its interface and its theorem's shape

The sampling layer is SPOT's, namespace `Spot.Sampling`. It sits on the argument system (`Definitions.Core.Layer.Argument`)
and under the recursion layer and compose.

**What it takes.** Units `Fin n`, a circuit's proof blocks under its partition; a sampling design `L` over them, the
verifier's live coins; the registration type `Reg`, sent before the coin; and the argument system's session on each
sample against each registration. Of the argument system it takes its analysis at each sampled unit: transcripts `Tr`
with the incorrect set of each; the committed transcript, a function of the prover's state at registration; and two
failure probabilities, `ks` (the session accepts while extraction of `u` fails) and `link` (extraction succeeds but
disagrees with the committed transcript on `u`'s wires), which cover acceptance at every sampled incorrect unit.

**What it gives** (`Sound`). For every event `E` on the committed transcript `X` and the sample `S`,
`Pr[accept ∧ E(X, S)] ≤ Pr_S[E(X, S) ∧ S ∩ incorrect(X) = ∅] + ε_ks(σ) + ε_bind(σ)`.

**Its error, in the layer below's.** `ε_ks` is the argument system's knowledge error at the sampled unit any rule picks
before the session's first coin, averaged over the sample (`κ` and the collision terms of `Verity.ProofSystem.KS` at the
instance's extractor); `ε_bind` is the commitment's binding error at the committed transcript. The sampling term is the
design's own: at `E = (K ≤ |incorrect X|)` it is `L.miss K`, at `E = (W ⊆ incorrect X)` it is `L.missProb W`.

**Levels.** `L` is any design over the units. Lean's levels (`Verity.Levels`) enter as the design their sample induces on
the units, and one level is `L` itself, so `Sound` at a levels design is `Sound` at that `L` and needs no second form.

**What proves it.** `Spot.Sampling.sound`, today `FlockSoundness.Audit.extraction_audit_le`
(`Proofs/Flock/Soundness/Audit/Extraction.lean`), once lean's move places it. `Sound` is that theorem's statement with
`ExtractionAnalysis`'s fields as binders and `audit`, `reg`, `cont`, `committedOf`, `atTgt`, `KnowledgeSound` and
`BindSound` unfolded, so one term proves it:
`fun _ _ _ _ Tr inc com ks link cover _ _ hks hlink E σ => sound ⟨Tr, inc, com, ks, link, cover⟩ hks hlink E σ`.
The game and the analysis are written out here because the definitions package can't read the proofs tree; when the
move lifts `audit` and `ExtractionAnalysis` into `Definitions/Spot/Sampling/`, `Sound` reads them by name in that change.

**Instances.**
* SPOT[Flock] (`SpotFlock`), the ZK compiled analysis (`Flock.Bridge.analysisZCRegC`): `Flock.Guarantees.SampledKSLiveV5`
  and `Flock.Guarantees.SampledKSLiveV3`, deployed and proved. The recursive guarantees reach it through
  `extraction_audit_sampled` (`Proofs/Flock/Recursive/Transfer.lean`).
* The compiler generic in the argument system, `FlockSoundness.SampledGeneric` (`Flock.Guarantees.GenericSampledKS`):
  proved, not deployed.
* `Claims.SamplingGeneric.CompilerKS` and PoUW's sampled window (`Proofs/Pouw/TS/Sampled.lean`) state the same bound
  and prove it apart; they are to be its instances.

**Duplicates, for next week.** `FlockSoundness.Audit.audit_le` proves the deterministic-extractor case again rather
than through `ExtractionAnalysis`; `FlockSoundness.Discharge.Exec.auditLive` is the game again with its transcript kept;
`FlockSoundness.SampledGeneric.ProofSystem` is `Claims.SamplingGeneric.Arg` again, and both are a second
`Verity.ProofSystem`.
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
`Spot.Sampling.sound` (today `FlockSoundness.Audit.extraction_audit_le`) once the move places it. -/
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
