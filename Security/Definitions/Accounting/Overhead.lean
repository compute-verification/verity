import Definitions.Core.Game
import Definitions.Accounting.Physical

/-!
# Tasks, classes, thresholds and overhead (trusted)

The symbol table's task and overhead rows, and the classes they are read in:

* a **task** `𝒢` (`Task`): one uniform coin, drawn up front, giving the input `x` and the set `𝒴` of its acceptable
  outputs. Tasks are static;
* **success** (`Succeeds`): the output a run holds at output positions fixed before `x` lies in `𝒴`;
* **cost** is server-time (`serverTime`), not size or wall-clock: each server is occupied from tick `0` to one past its
  last computed gate, and the costs of the servers add. A computation's cost is its server-time on its worst input
  (`cost`);
* a **class** `𝒜` (`Class`): the pairs of a computation (a run on each input) and an input that are allowed. The
  physical model's class is `𝒜_phys` (`physClass`), a subprotocol application's envelope `𝒜_a` (`envelope`), and
  `𝒜′ = 𝒜_phys ∩ ⋂_a 𝒜_a` (`restricted`) the class every accepted run lies in except with probability `Σ_a δ_a` and
  the cryptographic terms;
* the **threshold** `T_𝒢(η; 𝒜)` (`threshold`): the least cost of a computation whose probability of succeeding inside
  the class `𝒜` exceeds `η`, and the **overhead** (`overhead`) the ratio of two thresholds,
  `T_𝒢(η; 𝒜′) / T_𝒢(η; 𝒜_phys)`.
-/

namespace Accounting.Overhead

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

open Classical in
/-- **A location's busy time** in the actual circuit `A`: one past the last tick at which a leaf of `Y` computes a gate,
and `0` when none does. -/
noncomputable def busy {L : Type} (A : Placed L) (Y : Finset L) : ℕ :=
  (A.computed.filter fun g => A.x g ∈ Y).sup fun g => A.t g + 1

/-- **Server-time**: the servers' busy times summed, each server `X s` occupied from tick `0` to one past its last
computed gate. -/
noncomputable def serverTime {L Srv : Type} [Fintype Srv] (X : Srv → Finset L) (A : Placed L) : ℕ :=
  ∑ s, busy A (X s)

/-- **A class** `𝒜` of computations over inputs `In`: the pairs of a computation (a run on each input) and an input
that are allowed. -/
abbrev Class (In L : Type) := Set ((In → Actual.Run L) × In)

/-- **The success probability** of the computation `C`, with outputs at `pos`, in the class `𝒜` on the task `𝒢`:
`Pr_{(x, 𝒴) ∼ 𝒢}[(C, x) ∈ 𝒜 ∧ C(x) ∈ 𝒴]`. -/
noncomputable def success {In L : Type} {m : ℕ} (G : Task In (Fin m → List Bool)) (𝒜 : Class In L)
    (C : In → Actual.Run L) (pos : Fin m → L × ℕ × ℕ) : ℝ≥0∞ :=
  G.prob fun x 𝒴 => (C, x) ∈ 𝒜 ∧ Succeeds (C x) pos 𝒴

/-- **The cost** of a computation: its server-time on its worst input. -/
noncomputable def cost {In L Srv : Type} [Fintype Srv] (X : Srv → Finset L) (C : In → Actual.Run L) : ℝ≥0∞ :=
  ⨆ x, (serverTime X (C x).A : ℝ≥0∞)

/-- **The threshold** `T_𝒢(η; 𝒜)`: the least cost of a computation, with its output positions, whose success
probability in `𝒜` exceeds `η`; `⊤` when none's does. -/
noncomputable def threshold {In L Srv : Type} [Fintype Srv] {m : ℕ} (G : Task In (Fin m → List Bool))
    (𝒜 : Class In L) (X : Srv → Finset L) (η : ℝ≥0∞) : ℝ≥0∞ :=
  ⨅ (C : In → Actual.Run L) (pos : Fin m → L × ℕ × ℕ) (_ : η < success G 𝒜 C pos), cost X C

/-- **The overhead** of the class `𝒜′` over the physical class `𝒜_phys` on `𝒢` at `η`: `T_𝒢(η; 𝒜′) / T_𝒢(η; 𝒜_phys)`,
read where the denominator is in `(0, ∞)`. -/
noncomputable def overhead {In L Srv : Type} [Fintype Srv] {m : ℕ} (G : Task In (Fin m → List Bool))
    (𝒜 𝒜phys : Class In L) (X : Srv → Finset L) (η : ℝ≥0∞) : ℝ≥0∞ :=
  threshold G 𝒜 X η / threshold G 𝒜phys X η

/-- **The physical model's class** `𝒜_phys` on the hardware `H`: the computations whose runs are feasible under the
physical model on every input (`Feasible`, each run read with a usage of its own). The fleet's fixed-machine
dynamics and its chosen-state budget (`FixedMachine`, `InitialBudget`) join it when stated. -/
def physClass {In L E : Type} (H : Hardware L E) : Class In L :=
  {p | ∀ x, ∃ usage, Feasible H (p.1 x).A usage}

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

end Accounting.Overhead
