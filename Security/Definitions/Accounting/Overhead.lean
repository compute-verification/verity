import Definitions.Core.Circuit
import Definitions.Core.Game

/-!
# Overhead: a task's threshold over a class of computations (trusted)

The objects of the end-to-end overhead guarantee (`note:training-20261027T0134Z-draft-e2e-overhead-proof` §0, the
ontology Daniel fixed on 26 Oct), on core's circuit (`Verity.Circuit`) and its games (`Verity.Game`):

* a **class** `𝒜` (`Class X`): the computations allowed, a set of pairs `(C, x)` of a circuit and an input. A deployment's
  class is physics intersected with each subprotocol's envelope, `𝒜' = 𝒜_phys ∩ ⋂_a 𝒜_a`;
* a **static task** `𝒢` (`Task X Y`): a draw of a pair `(x, 𝒴)`, an input and the set of its acceptable outputs, made up
  front by one uniform coin. It is a `Verity.Game` with one coin and no message (`Task.game`), so its probabilities
  (`Task.prob`) are the games' (`Verity.Game.prob`), the probabilities the protocols' guarantees bound;
* a circuit's **size** `|C|`: its number of operation gates, `C.size C.computed`;
* its **output** `C(x)`: a map `out : Verity.Circuit → X → Y`, a parameter of what reads it, which the canonical format of
  circuit inputs and outputs fixes;
* the **success probability** of `C` in `𝒜` (`success`): `Pr_{(x, 𝒴) ∼ 𝒢}[(C, x) ∈ 𝒜 ∧ C(x) ∈ 𝒴]`;
* the **threshold** `T_𝒢(η; 𝒜) = inf { |C| : Pr_{(x, 𝒴) ∼ 𝒢}[(C, x) ∈ 𝒜 ∧ C(x) ∈ 𝒴] > η }` (`threshold`), `⊤` when no
  circuit's success probability exceeds `η`;
* the **overhead** of `𝒜` over `𝒜_phys` (`overhead`): `T_𝒢(η; 𝒜) / T_𝒢(η; 𝒜_phys)`, read where the denominator is in
  `(0, ∞)`.
-/

namespace Accounting.Overhead

open Verity
open scoped ENNReal

/-- **A class of computations** over inputs `X`: the pairs `(C, x)` of a circuit and an input that are allowed. -/
abbrev Class (X : Type) := Set (Circuit × X)

/-- **A static task** over inputs `X` and outputs `Y`: one uniform coin `ω`, drawn up front, and the input `x` and the
set `𝒴` of its acceptable outputs that it gives (`draw ω = (x, 𝒴)`). -/
structure Task (X Y : Type) where
  /-- the coin -/
  Ω : Type
  [fintype : Fintype Ω]
  [nonempty : Nonempty Ω]
  /-- the input and its acceptable outputs, at each value of the coin -/
  draw : Ω → X × Set Y

attribute [instance] Task.fintype Task.nonempty

namespace Task

variable {X Y : Type} (G : Task X Y)

/-- **The task as a game**: its one coin, then the pair it draws. -/
def game : Game (X × Set Y) := .coin G.Ω fun ω => .ret (G.draw ω)

/-- The task's one strategy: its game sends nothing. -/
def play : Game.Strategy G.game := fun _ => PUnit.unit

/-- **The probability over the task** `Pr_{(x, 𝒴) ∼ 𝒢}[E x 𝒴]`. -/
noncomputable def prob (E : X → Set Y → Prop) : ℝ≥0∞ :=
  Game.prob (fun p => E p.1 p.2) G.game G.play

end Task

/-- **The success probability** of the circuit `C` in the class `𝒜` on the task `𝒢`, with outputs `out`:
`Pr_{(x, 𝒴) ∼ 𝒢}[(C, x) ∈ 𝒜 ∧ C(x) ∈ 𝒴]`. -/
noncomputable def success {X Y : Type} (G : Task X Y) (A : Class X) (out : Circuit → X → Y) (C : Circuit) : ℝ≥0∞ :=
  G.prob fun x 𝒴 => (C, x) ∈ A ∧ out C x ∈ 𝒴

/-- **The threshold** `T_𝒢(η; 𝒜)`: the least size of a circuit whose success probability in `𝒜` exceeds `η`, and `⊤`
when no circuit's does. -/
noncomputable def threshold {X Y : Type} (G : Task X Y) (A : Class X) (out : Circuit → X → Y) (η : ℝ≥0∞) : ℝ≥0∞ :=
  ⨅ (C : Circuit) (_ : η < success G A out C), ((C.size C.computed : ℕ) : ℝ≥0∞)

/-- **The overhead** of the class `𝒜` over the physical class `𝒜_phys` on the task `𝒢` at `η`:
`T_𝒢(η; 𝒜) / T_𝒢(η; 𝒜_phys)`. It is read where the denominator is in `(0, ∞)`; outside, `ℝ≥0∞`'s division gives `⊤`
over `0` (of a nonzero numerator) and `0` over `⊤`. -/
noncomputable def overhead {X Y : Type} (G : Task X Y) (A Aphys : Class X) (out : Circuit → X → Y) (η : ℝ≥0∞) :
    ℝ≥0∞ :=
  threshold G A out η / threshold G Aphys out η

end Accounting.Overhead
