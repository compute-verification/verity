import Mathlib.Probability.ProbabilityMassFunction.Constructions
import Mathlib.Probability.ProbabilityMassFunction.Monad
import Mathlib.Probability.Distributions.Uniform
import Verity.Protocol.Game
import Run.World

/-!
# The run as a distribution (trusted)

`Run.World` is deterministic given the events and each trusted step's fresh bytes. Here the bytes are uniform and
independent, and each side of the adversary is a function of what it sees:

* soundness (`soundRun`): one adversary, the developer with the network, sees every frame sent and picks every event;
  the auditor's four parties are the only trusted ones;
* zero knowledge (`zkRun`): the boundary's four parties are the trusted ones; each tick the GPU rack takes one event of
  its own from what reaches it (`World.gpuView`), then the network, which here includes the auditor, one of its own
  from what crosses the boundary (`World.netView`). The split is the egress premise: the rack reaches the network only
  through the boundary, so nothing it sees or does reaches the network except as the boundary's frames and their times.

That the operating system's bytes are uniform (A3, `uniform/io-getrandombytes`) is not a `Prop` here: the run draws
them itself (`fresh`), so the model embodies it, as C-Flock's session game does for its coins.

An adversary is a function of its view rather than a strategy of core's `Game`: a trusted party's state holds secrets
the adversary never sees (the coin tree's unrevealed coins in soundness; V*'s witness and the masks in zero knowledge),
and a `Game` strategy sees every coin. `soundGame` is the soundness run as a `Game`, the form core's and C-Flock's
composition theorems take; it is sound only with the coin tree's coins drawn when revealed, a hybrid the proofs take
at the hiding term.
-/

namespace RunModel

open Run
open scoped ENNReal

/-- `n` fresh random bytes. -/
abbrev Bytes (n : ℕ) := Fin n → Fin 256

def Bytes.toByteArray {n : ℕ} (r : Bytes n) : ByteArray := ⟨Array.ofFn fun i => (r i).val.toUInt8⟩

/-- A trusted step's fresh bytes: uniform, independent of everything before. -/
noncomputable def fresh (n : ℕ) : PMF ByteArray := (PMF.uniformOfFintype (Bytes n)).map Bytes.toByteArray

/-- An adversary: its next event as a function of what it has seen. -/
abbrev Adversary := List (ℕ × Frame) → Event

/-- The soundness run of `fuel` events from `w`. -/
noncomputable def soundRun (A : Agreement) (dev : Adversary) :
    ℕ → World (Setting.sound A) → PMF (World (Setting.sound A))
  | 0, w => PMF.pure w
  | n + 1, w =>
    let ev := dev w.devView
    (fresh (w.need ev)).bind fun r => soundRun A dev n (w.step ev r)

/-- The zero-knowledge run of `fuel` ticks from `w`. -/
noncomputable def zkRun (A : Agreement) (net gpu : Adversary) :
    ℕ → World (Setting.zk A) → PMF (World (Setting.zk A))
  | 0, w => PMF.pure w
  | n + 1, w =>
    let e₁ : Event := if w.byGpu (gpu w.gpuView) then gpu w.gpuView else .idle
    (fresh (w.need e₁)).bind fun r₁ =>
      let w₁ := w.step e₁ r₁
      let e₂ : Event := if w₁.byGpu (net w₁.netView) then .idle else net w₁.netView
      (fresh (w₁.need e₂)).bind fun r₂ => zkRun A net gpu n (w₁.step e₂ r₂)

/-- The run in setting `S` as one of core's games: the adversary sends each event, and a trusted step's fresh bytes are
a live coin. -/
def soundGame (S : Setting) : ℕ → World S → Verity.Game (World S)
  | 0, w => .ret w
  | n + 1, w => .send Event fun ev => .coin (Bytes (w.need ev)) fun r => soundGame S n (w.step ev r.toByteArray)

end RunModel
