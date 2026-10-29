import Definitions.Nci.Capability.Pointwise
import Definitions.Nci.Capability.Toy
import Definitions.Core.Game

/-!
# The pointwise slowdown's toy instance

The counted computations, decompositions and executions at which the pointwise slowdown's witnesses
(`Definitions.Nci.Guarantees.Capability.PointwiseWitnesses`) are stated, on the local bounds' toy circuit
`Nci.Capability.Toy.C`, whose operation gates are `2` and `3`:

* `X`: three occurrences of weight `1`, occurrence `0` computed by gate `2`, `1` by gates `2` and `3`, and `2` by
  gate `3`. Its decomposition `D` at `c = 1` has one region, `{2}`, and the remainder `{3}`; it represents `0` and `1`
  at gate `2` and `2` at gate `3`. So the region computes `0` whole, `1` is split there, and `2` is in the remainder;
* `heavy`: one occurrence of weight `10`, at gate `2`, more than two gates carry under `D`'s caps;
* `game`: one fair coin, the execution `exec true = X` or `exec false = heavy`;
* `Xp`: the toy's one counted product (`Toy.B`) as one occurrence at gate `2`, weighted by its slot's price, one bit,
  times its use's weight `1`; its decomposition `Dp` at `c = 2` has one region, `R1`'s proof block `{2, 3}`.
-/

namespace Nci.Capability.PointwiseToy

open Verity Finset Nci.Capability

/-- The occurrences' gates: `0` at gate `2`, `1` at gates `2` and `3`, `2` at gate `3`. -/
def G (o : Fin 3) : Finset (Fin 4) := if o.val = 0 then {2} else if o.val = 1 then {2, 3} else {3}

/-- Three occurrences of weight `1` on the toy circuit. -/
@[reducible] def X : Counted where
  C := Toy.C
  κ := Fin 3
  G := G
  ω := fun _ => 1

/-- `X`'s decomposition at `c = 1`: the region `{2}`, the remainder `{3}`; occurrences `0` and `1` represented at gate
`2`, occurrence `2` at gate `3`. -/
def D : X.Decomposition 1 where
  B := 1
  region := fun _ => {2}
  region_computed := by decide
  region_card := fun _ => rfl
  region_disjoint := fun b b' h => absurd (Subsingleton.elim b b') h
  rep := fun o => if o.val = 2 then 3 else 2
  rep_mem := by decide
  rep_computed := by decide

/-- One occurrence of weight `10`, at gate `2`. -/
@[reducible] def heavy : Counted where
  C := Toy.C
  κ := Fin 1
  G := fun _ => {2}
  ω := fun _ => 10

/-- The executions: one fair coin. -/
def game : Game Bool := Game.draw Bool

/-- The game's one strategy: it has no messages. -/
def strat : Game.Strategy game := fun _ => PUnit.unit

/-- The execution's counted computation: `X` on `true`, `heavy` on `false`. -/
def exec (b : Bool) : Counted := if b then X else heavy

/-- The toy's counted product as one occurrence at gate `2`, its slot's price times its use's weight `1`. -/
@[reducible] noncomputable def Xp : Counted :=
  Counted.ofProducts Toy.B Toy.W (fun _ => 1) (fun _ : Fin 1 => (0, 0)) fun _ => {2}

/-- `Xp`'s decomposition at `c = 2`: one region, `R1`'s proof block `{2, 3}`; the occurrence represented at gate `2`. -/
noncomputable def Dp : Xp.Decomposition 2 where
  B := 1
  region := fun _ => {2, 3}
  region_computed := by decide
  region_card := fun _ => rfl
  region_disjoint := fun b b' h => absurd (Subsingleton.elim b b') h
  rep := fun _ => 2
  rep_mem := by decide
  rep_computed := by decide

end Nci.Capability.PointwiseToy
