module

public import Properties.Verity.Soundness
public import Definitions.Core.CircuitLevels

/-!
# Soundness at the verifier, stated over levels

`Verity.Soundness` and `Verity.RemoteSoundness` state SPOT's claim on one fixed partition, the committed files' units
(`Vocab.partsOf`). Here the partition is the last level of a chain of partitions of the committed circuit, nested below
the whole circuit, coarsest first, each with its sampling rule (`Partition.Chain`, whose `Verity.Levels` is
`Levels.ofPartitions`), so one statement covers a sample drawn in one level, two, or more.

* `Vocab.LevelsAt`: a chain at each parameters and committed files.
* `Vocab.SpotHoldsLevels`: SPOT's claim over levels, at most the published share `ε` of the last level's units unbound
  in the extractor's transcript. `Vocab.SpotSoundLevels`: its target.
* `Vocab.HoldsLevels`, `Vocab.SoundnessAtLevels`: `Vocab.Holds` and `Vocab.SoundnessAt` with SPOT's claim the one over
  levels and every other subprotocol's unchanged.
* `OnsiteSoundnessLevels`, `RemoteSoundnessLevels`: the two composites, stated; nothing proves them yet.

At a chain whose last partition is the files' units, each is today's statement: one level is the one-stage form, two
levels with the files' units last the two-stage form. The statements read the levels' partitions and not their
sampling rules: the rules enter where `Vocab.SpotSoundLevels` is proved. The verifier accepts the one- and two-stage
forms only, so the statements cover more chains than any accepted session has.
-/

@[expose] public section

namespace Verity

open ResourceAccounting
open scoped ENNReal

namespace Vocab

variable {ι : Type} {d : Device ι} {α : Type} [Fintype α] (V : Vocab d α)

/-- **Levels at the committed files**: at each parameters and committed files, partitions of the committed circuit
nested below the whole circuit, coarsest first, each with its sampling rule. -/
abbrev LevelsAt : Type 1 :=
  (θ : ByteArray) → (F : Array ByteArray) → Partition.Chain (V.circuitOf θ F) (Partition.whole (V.circuitOf θ F))

/-- **SPOT's claim over levels**: at most the published share `ε` of the last level's units unbound in the extractor's
transcript. -/
def SpotHoldsLevels (ch : V.LevelsAt) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Game.Strategy (V.game θ F c)) (_ : Exchange) : Prop :=
  IntegrityHolds (ch θ F).last.2 (V.Opens θ F c) (V.extract θ F c σ) (V.certOf θ).ε

/-- **The claims over levels**: SPOT's on the last level, every other subprotocol's as `Vocab.Holds`. -/
def HoldsLevels (k : Verity.Registration.Composite) (ch : V.LevelsAt) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Game.Strategy (V.game θ F c)) : Subprotocol → Exchange → Prop
  | .sampled, o => V.SpotHoldsLevels ch θ F c σ o
  | a, o => V.Holds k θ F c σ a o

/-- **SPOT's target over levels**: SPOT accepts while its claim over levels fails with probability at most its error. -/
def SpotSoundLevels (k : Verity.Registration.Composite) (ch : V.LevelsAt) : Prop :=
  ∀ (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (σ : Game.Strategy (V.game θ F c)),
    Game.prob (fun o => V.Verdict k .sampled o ∧ ¬ V.SpotHoldsLevels ch θ F c σ o) (V.game θ F c) σ ≤
      V.error .sampled θ F c σ

/-- **Soundness at composite `k`'s compiled verifier, over levels.** -/
def SoundnessAtLevels (k : Verity.Registration.Composite) (ch : V.LevelsAt) : Prop :=
  ∀ (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (σ : Game.Strategy (V.game θ F c)),
    Game.prob (fun o => Accepts k o.sent ∧ V.Registered o.params o.files ∧
        ∃ a ∈ subprotocols k, ¬ V.HoldsLevels k ch θ F c σ a o) (V.game θ F c) σ ≤
      ∑ a ∈ subprotocols k, V.error a θ F c σ

end Vocab

/-- **Soundness at the onsite verifier, over levels** (stated). `Verity.Soundness` with SPOT's target and claim over a
chain of levels. -/
def OnsiteSoundnessLevels : Prop :=
  ∀ (ι α : Type) [Fintype α] (d : Device ι) (V : Vocab d α) (ch : V.LevelsAt),
    V.SpotSoundLevels .onsite ch → V.PouwSound .onsite → V.InferenceOnlySound .onsite → V.PousSound .onsite →
      V.BandwidthAuditSound .onsite → V.NetworkAttesterSound .onsite → V.WipeSound .onsite →
      V.SoundnessAtLevels .onsite ch

/-- **Soundness at remote's verifier, over levels** (stated). `Verity.RemoteSoundness` with SPOT's target and claim
over a chain of levels. -/
def RemoteSoundnessLevels : Prop :=
  ∀ (ι α : Type) [Fintype α] (d : Device ι) (V : Vocab d α) (ch : V.LevelsAt),
    V.SpotSoundLevels .remote ch → V.PouwSound .remote → V.SoundnessAtLevels .remote ch

end Verity
