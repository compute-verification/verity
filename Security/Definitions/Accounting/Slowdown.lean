import Definitions.Core.Game
import Definitions.Accounting.Physical

/-!
# Tasks, classes, thresholds and slowdown (trusted)

The symbol table's task and slowdown rows, and the classes they are read in:

* a **task** `𝒢` (`Task`): one uniform coin, drawn up front, giving the input `x` and the set `𝒴` of its acceptable
  outputs. Tasks are static;
* **success** (`Succeeds`): the output a run holds at output positions fixed before `x` lies in `𝒴`;
* **cost** is fleet-time (`fleetTime`), not size: the elapsed time, from tick `0` to one past the run's last computed
  gate, multiplied by the whole fleet, every server, registered and unregistered. A computation's cost is its
  fleet-time on its worst input (`cost`). The run's own server-time would not do: a run can sit on unregistered
  servers alone, so its server-time under every protocol is about its server-time without one;
* a **class** `𝒜` (`Class`): the pairs of a computation (a run on each input) and an input that are allowed. The
  physical model's class is `𝒜_phys` (`physClass`), a subprotocol application's envelope `𝒜_a` (`envelope`), and
  `𝒜′ = 𝒜_phys ∩ ⋂_a 𝒜_a` (`restricted`) the class every accepted run lies in except with probability `Σ_a δ_a` and
  the cryptographic terms;
* the **threshold** `T_𝒢(η; 𝒜)` (`threshold`): the least cost of a computation whose probability of succeeding inside
  the class `𝒜` exceeds `η`, and the **slowdown** (`slowdown`) the ratio of two thresholds,
  `T_𝒢(η; 𝒜′) / T_𝒢(η; 𝒜_phys)`.
-/

namespace Accounting.Slowdown

open Verity
open scoped ENNReal

/-- **A static task** `𝒢` over inputs `In` and outputs `Y`: one uniform coin, drawn up front, and the input `x` and the
set `𝒴` of its acceptable outputs that it gives (`draw ω = (x, 𝒴)`). -/
structure Task (In Y : Type) where
  /-- the coin -/
  Ω : Type
  [fintype : Fintype Ω]
  [nonempty : Nonempty Ω]
  /-- the input and its acceptable outputs, at each value of the coin -/
  draw : Ω → In × Set Y

attribute [instance] Task.fintype Task.nonempty

/-- The task as a game: its one coin, then the pair it draws. -/
def Task.game {In Y : Type} (G : Task In Y) : Game (In × Set Y) := .coin G.Ω fun ω => .ret (G.draw ω)

/-- **The probability over the task** `Pr_{(x, 𝒴) ∼ 𝒢}[E x 𝒴]`. -/
noncomputable def Task.prob {In Y : Type} (G : Task In Y) (E : In → Set Y → Prop) : ℝ≥0∞ :=
  Game.prob (fun p => E p.1 p.2) G.game fun _ => PUnit.unit

/-- **The output at positions** `pos` (each a leaf, a cell and a tick) in the run `R` is `y`: each position holds a gate
whose value's bits are `y i`. -/
def Reads {L : Type} {m : ℕ} (R : Actual.Run L) (pos : Fin m → L × ℕ × ℕ) (y : Fin m → List Bool) : Prop :=
  ∀ i, ∃ g, (R.A.x g, R.A.cell g, R.A.t g) = pos i ∧ Actual.bits (R.A.w g) (R.val g).toNat = y i

/-- **Success**: the output the run `R` holds at the positions `pos`, fixed before the input is drawn, lies in `𝒴`. -/
def Succeeds {L : Type} {m : ℕ} (R : Actual.Run L) (pos : Fin m → L × ℕ × ℕ) (𝒴 : Set (Fin m → List Bool)) :
    Prop :=
  ∃ y ∈ 𝒴, Reads R pos y

/-- **The elapsed time** of the actual circuit `A`: one past the last tick at which any leaf computes a gate, and `0`
when none does. -/
noncomputable def elapsed {L : Type} (A : Placed L) : ℕ :=
  A.computed.sup fun g => A.t g + 1

/-- **Fleet-time**: the elapsed time multiplied by the whole fleet, the servers `Srv`, registered and unregistered. -/
noncomputable def fleetTime {L : Type} (Srv : Type) [Fintype Srv] (A : Placed L) : ℕ :=
  Fintype.card Srv * elapsed A

/-- **A class** `𝒜` of computations over inputs `In`: the pairs of a computation (a run on each input) and an input
that are allowed. -/
abbrev Class (In L : Type) := Set ((In → Actual.Run L) × In)

/-- **The success probability** of the computation `C`, with outputs at `pos`, in the class `𝒜` on the task `𝒢`:
`Pr_{(x, 𝒴) ∼ 𝒢}[(C, x) ∈ 𝒜 ∧ C(x) ∈ 𝒴]`. -/
noncomputable def success {In L : Type} {m : ℕ} (G : Task In (Fin m → List Bool)) (𝒜 : Class In L)
    (C : In → Actual.Run L) (pos : Fin m → L × ℕ × ℕ) : ℝ≥0∞ :=
  G.prob fun x 𝒴 => (C, x) ∈ 𝒜 ∧ Succeeds (C x) pos 𝒴

/-- **The cost** of a computation on the fleet `Srv`: its fleet-time on its worst input. -/
noncomputable def cost {In L : Type} (Srv : Type) [Fintype Srv] (C : In → Actual.Run L) : ℝ≥0∞ :=
  ⨆ x, (fleetTime Srv (C x).A : ℝ≥0∞)

/-- **The threshold** `T_𝒢(η; 𝒜)`: the least cost of a computation, with its output positions, whose success
probability in `𝒜` exceeds `η`; `⊤` when none's does. -/
noncomputable def threshold {In L : Type} {m : ℕ} (G : Task In (Fin m → List Bool)) (𝒜 : Class In L)
    (Srv : Type) [Fintype Srv] (η : ℝ≥0∞) : ℝ≥0∞ :=
  ⨅ (C : In → Actual.Run L) (pos : Fin m → L × ℕ × ℕ) (_ : η < success G 𝒜 C pos), cost Srv C

/-- **The slowdown** of the class `𝒜′` over the physical class `𝒜_phys` on `𝒢` at `η`: `T_𝒢(η; 𝒜′) / T_𝒢(η; 𝒜_phys)`,
both in fleet-time, read where the denominator is in `(0, ∞)`. -/
noncomputable def slowdown {In L : Type} {m : ℕ} (G : Task In (Fin m → List Bool)) (𝒜 𝒜phys : Class In L)
    (Srv : Type) [Fintype Srv] (η : ℝ≥0∞) : ℝ≥0∞ :=
  threshold G 𝒜 Srv η / threshold G 𝒜phys Srv η

/-- **The physical model's class** `𝒜_phys` on the hardware `H`: the computations whose runs are feasible under the
physical model on every input (`Feasible`, each run read with an annotation of its own). The fleet's fixed-machine
dynamics and its chosen-state budget (`FixedMachine`, `InitialBudget`) join it when stated. -/
def physClass {In L E : Type} (H : Hardware L E) : Class In L :=
  {p | ∀ x, ∃ ann, Feasible H (p.1 x).A ann}

/-- **The envelope** `𝒜_a` of a subprotocol application `a` whose claim on a run is `claim`: the computations on whose
run at the input the claim holds. Protocol soundness bounds `Pr[accept ∧ (C, x) ∉ 𝒜_a]` by its failure bound `δ_a`
(`Verity.Sound`). -/
def envelope {In L : Type} (claim : Actual.Run L → Prop) : Class In L :=
  {p | claim (p.1 p.2)}

/-- **The class `𝒜′ = 𝒜_phys ∩ ⋂_a 𝒜_a`** on the hardware `H`, over a finite family of subprotocol applications `a`,
each with its claim `claim a`: the physical model's class cut down to every envelope. Every accepted run lies in it
except with probability `Σ_a δ_a` and the cryptographic terms. -/
def restricted {In L E ι : Type} [Fintype ι] (H : Hardware L E) (claim : ι → Actual.Run L → Prop) : Class In L :=
  physClass H ∩ ⋂ a, envelope (claim a)

end Accounting.Slowdown
