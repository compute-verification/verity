import Definitions.Population.Dataflow

/-!
# Obliviousness: when a program's trace is a population

A program's trace is a `DynPopulation`: its write sets are public, and a step's read set may depend on values, as a
step reads memory at an address it holds. This file names the properties of those read sets that decide whether the
trace is a population with public wiring, and the construction that makes it one.

* `Oblivious`: every unit reads the same slots in every assignment, honest or not. Control flow and addresses depend
  on no value at all.
* `AddressedBy P`: the read sets depend only on the values of the slots `P`, where the addresses live. With `P` the
  verifier-bound public inputs, this is obliviousness relative to the public inputs: a circuit family indexed by them.
* `HonestlyOblivious`: the classical notion. Every run in which every unit is correct reads the same slots, whatever
  the inputs. It says nothing about runs with a wrong unit.
* `AddressClosed`: a unit's read set is determined by the values of the slots it reads, so an address it uses is a
  slot it reads. A machine step that loads through a pointer reads the pointer.
* `padOver C`, **the padding over the assignments `C`**: the static population in which unit `u` reads every slot it
  reads in some assignment of `C`, with `padFns C` its functions. `padOver Set.univ` is the static superset of a padded
  static circuit (every candidate wired, the choice a value inside the unit); `padOver {h}` freezes the read sets at
  `h`.
* `downAt σ B`: the units downstream of `B` in the dataflow of the assignment `σ`, the cone the run `σ` itself draws.
* `SkipsLucky`: work accounting's per-unit premise for a trace.
* `downOver C B`, `coneEgressOver C O B`: the cone of `B` in the dataflow of some assignment of `C`, and the egress
  written in it. Over the honest runs, the cones the damage theorems need.
-/

namespace Population

open Finset

namespace DynPopulation

variable {U S V : Type} (D : DynPopulation U S V)

/-- **Oblivious**: every unit reads the same slots in every assignment. -/
def Oblivious : Prop := ∀ u σ τ, D.reads u σ = D.reads u τ

/-- **Addressed by `P`**: two assignments that agree on `P` give every unit the same read set. With `P` the public
inputs, obliviousness relative to them. -/
def AddressedBy (P : Finset S) : Prop := ∀ u σ τ, (∀ s ∈ P, σ s = τ s) → D.reads u σ = D.reads u τ

/-- **Honestly oblivious**, the classical notion: every two runs in which every unit is correct, on any inputs, give
every unit the same read set. -/
def HonestlyOblivious : Prop :=
  ∀ u σ τ, (∀ v, D.Correct σ v) → (∀ v, D.Correct τ v) → D.reads u σ = D.reads u τ

/-- **Address closure**: two assignments that agree on what a unit reads in the first give it the same read set. -/
def AddressClosed : Prop := ∀ u σ τ, (∀ s ∈ D.reads u σ, σ s = τ s) → D.reads u σ = D.reads u τ

open Classical in
/-- **The padding over `C`**: unit `u` reads every slot it reads in some assignment of `C`. The inputs and write sets
are the program's. -/
noncomputable def padOver [Fintype S] (C : Set (S → V)) : Dataflow U S where
  input := D.input
  reads u := univ.filter fun s => ∃ σ ∈ C, s ∈ D.reads u σ
  writes := D.writes

open Classical in
/-- The padding's unit functions: unit `u` runs the program's step on the values of its padded read set, every other
slot at `default`. -/
noncomputable def padFns [Fintype S] [Inhabited V] (C : Set (S → V)) : (D.padOver C).Fns V :=
  fun u r s => D.f u (fun t => if h : t ∈ (D.padOver C).reads u then r ⟨t, h⟩ else default) s.1

/-- The units downstream of `B`, `B` included, in the dataflow of the assignment `σ`. -/
def downAt (σ : S → V) (B : Finset U) : Set U := {v | ∃ u ∈ B, Relation.ReflTransGen (D.Dep σ) u v}

/-- **Work accounting's per-unit premise for a trace**, as `Dataflow.SkipsLucky`: a unit the run skipped (outside
`K`), if it reads in `σ` the honest values at its honest read set and is correct, is in the lucky set `L`. -/
def SkipsLucky (honest σ : S → V) (K L : Finset U) : Prop :=
  ∀ u, u ∉ K → (∀ r ∈ D.reads u honest, σ r = honest r) → D.Correct σ u → u ∈ L

open Classical in
/-- **The cone over the assignments `C`**: the units downstream of `B` in the dataflow of some assignment of `C`. With
`C` the honest runs it is a property of the program alone, and it can be smaller than the padding's cone, whose paths
may mix the read sets of different runs. -/
noncomputable def downOver [Fintype U] (C : Set (S → V)) (B : Finset U) : Finset U :=
  univ.filter fun v => ∃ h ∈ C, v ∈ D.downAt h B

open Classical in
/-- The egress slots `O` written in the cone over `C` of `B`. -/
noncomputable def coneEgressOver [Fintype U] (C : Set (S → V)) (O : Finset S) (B : Finset U) : Finset S :=
  O ∩ D.writesOf (D.downOver C B)

end DynPopulation

end Population
