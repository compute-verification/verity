module

public import Mathlib.Data.Fintype.Pi
public import Mathlib.Data.Finset.Prod

/-! Populations, units over committed value slots with public dataflow, the objects SPOT's statement of at most `m` incorrect units is asked to be about. -/

@[expose] public section

namespace Population

open Finset

/-- A population's public dataflow: units `U` over value slots `S`, the input slots, and what each unit reads and
writes. None of it depends on a value. -/
structure Dataflow (U S : Type) where
  input : Finset S
  reads : U → Finset S
  writes : U → Finset S

namespace Dataflow

variable {U S : Type} (P : Dataflow U S)

/-- Unit functions over the dataflow: unit `u` maps the values of the slots it reads to the values of the slots it
writes. It sees nothing else, and may be any program. -/
def Fns (V : Type) : Type := (u : U) → (P.reads u → V) → (P.writes u → V)

/-- Unit `u` is correct in the assignment `σ`: its written slots hold its function of its read slots. -/
def Correct {V : Type} (f : P.Fns V) (σ : S → V) (u : U) : Prop :=
  ∀ s : P.writes u, σ s = f u (fun r => σ r) s

open Classical in
/-- The incorrect proof blocks of an assignment. -/
noncomputable def incorrect [Fintype U] {V : Type} (f : P.Fns V) (σ : S → V) : Finset U :=
  univ.filter fun u => ¬ P.Correct f σ u

/-- `σ` holds the inputs `a` on every input slot outside `X` (the secret slots, left free). -/
def Holds {V : Type} (a : S → V) (X : Finset S) (σ : S → V) : Prop :=
  ∀ s ∈ P.input, s ∉ X → σ s = a s

def Covered : Prop := ∀ s, s ∉ P.input → ∃ u, s ∈ P.writes u

/-- **Single assignment**: no unit writes an input slot, and every other slot is written by exactly one unit. -/
def SingleAssignment : Prop :=
  ∀ s, (s ∈ P.input → ∀ u, s ∉ P.writes u) ∧ (s ∉ P.input → ∃! u, s ∈ P.writes u)

/-- `v` depends on `u`: `v` reads a slot `u` writes. -/
def Dep (u v : U) : Prop := ∃ s ∈ P.writes u, s ∈ P.reads v

/-- **Acyclicity**: no unit depends on itself through a chain of units. -/
def Acyclic : Prop := ∀ u, ¬ Relation.TransGen P.Dep u u

/-- What the checker confirms from the dataflow alone: single assignment and acyclicity. -/
def Checked : Prop := P.SingleAssignment ∧ P.Acyclic

def Determinate : Prop :=
  ∀ (V : Type) (f : P.Fns V) (a : S → V), ∃! σ : S → V, (∀ s ∈ P.input, σ s = a s) ∧ ∀ u, P.Correct f σ u

open Classical in
/-- The units downstream of `B`, `B` included: those a chain of dependencies leads to from a unit of `B`. -/
noncomputable def down [Fintype U] (B : Finset U) : Finset U :=
  univ.filter fun v => ∃ u ∈ B, Relation.ReflTransGen P.Dep u v

def Feeds (s t : S) : Prop := ∃ u, s ∈ P.reads u ∧ t ∈ P.writes u

/-- The slots upstream of `O`, `O` included: those from which a chain of units leads into `O`. -/
def up (O : Finset S) : Set S := {s | ∃ o ∈ O, Relation.ReflTransGen P.Feeds s o}

open Classical in
/-- The slots the units of `B` write. -/
noncomputable def writesOf (B : Finset U) : Finset S := B.biUnion P.writes

open Classical in
/-- **The egress in `B`'s cone**: the egress slots `O` that a unit downstream of `B` writes. At `B = {u}` its size is
`E(u)`, the egress an incorrect `u` reaches, which the wiring alone determines. -/
noncomputable def coneEgress [Fintype U] (O : Finset S) (B : Finset U) : Finset S := O ∩ P.writesOf (P.down B)

open Classical in
noncomputable def egress [Fintype U] [Fintype S] {V : Type} [Fintype V] (f : P.Fns V) (a : S → V) (X O : Finset S)
    (B : Finset U) : Finset (O → V) :=
  (univ.filter fun σ : S → V => P.Holds a X σ ∧ P.incorrect f σ ⊆ B).image fun σ o => σ o

open Classical in
noncomputable def egressAt [Fintype U] [Fintype S] {V : Type} [Fintype V] (f : P.Fns V) (a : S → V) (X O : Finset S)
    (m : ℕ) : Finset (O → V) :=
  (univ.filter fun σ : S → V => P.Holds a X σ ∧ (P.incorrect f σ).card ≤ m).image fun σ o => σ o

open Classical in
/-- **The amplification set** at the incorrect set `B` and its writes `b`, over the secrets `Cl`: what the egress slots
`O` show in the assignments that hold the inputs `a` with a secret `s ∈ Cl` on the secret slots `X`, are incorrect only in
`B`, and write `b` on `B`'s slots. Its size is how many egress values the rest of the population makes of the secrets
once the incorrect proof blocks' writes are fixed. -/
noncomputable def amp [Fintype U] [Fintype S] [DecidableEq S] {V : Type} [Fintype V] [DecidableEq V] (f : P.Fns V) (a : S → V) (X O : Finset S)
    (Cl : Finset (S → V)) (B : Finset U) (b : P.writesOf B → V) : Finset (O → V) :=
  ((Cl ×ˢ (univ : Finset (S → V))).filter fun p =>
      (∀ t ∈ P.input, p.2 t = if t ∈ X then p.1 t else a t) ∧ P.incorrect f p.2 ⊆ B ∧
        ∀ t : P.writesOf B, p.2 t = b t).image fun p o => p.2 o

/-- **The per-unit hardness premise of work accounting**, for a run that committed `σ` while computing only the units
`K`: a unit it skipped, if it reads the honest values `honest` and is correct, is in the lucky set `L`. Whether a
skipped unit can be right without its work is a fact about the unit (its floor and its luck), not about the
population. -/
def SkipsLucky {V : Type} (f : P.Fns V) (honest σ : S → V) (K L : Finset U) : Prop :=
  ∀ u, u ∉ K → (∀ r ∈ P.reads u, σ r = honest r) → P.Correct f σ u → u ∈ L

end Dataflow

/-- A population whose read sets depend on values: the object a machine trace is, a step reading memory at an address
it holds. Unit `u` reads `reads u σ` in the assignment `σ`, and its function `f u` depends only on those slots. The
write sets stay public. -/
structure DynPopulation (U S V : Type) where
  input : Finset S
  writes : U → Finset S
  reads : U → (S → V) → Finset S
  f : U → (S → V) → S → V
  reads_only : ∀ u σ τ, (∀ s ∈ reads u σ, σ s = τ s) → ∀ s ∈ writes u, f u σ s = f u τ s

namespace DynPopulation

variable {U S V : Type} (D : DynPopulation U S V)

def Correct (σ : S → V) (u : U) : Prop := ∀ s ∈ D.writes u, σ s = D.f u σ s

open Classical in
noncomputable def incorrect [Fintype U] (σ : S → V) : Finset U := univ.filter fun u => ¬ D.Correct σ u

def Holds (a : S → V) (X : Finset S) (σ : S → V) : Prop := ∀ s ∈ D.input, s ∉ X → σ s = a s

def Covered : Prop := ∀ s, s ∉ D.input → ∃ u, s ∈ D.writes u

/-- Single assignment, as for a static population: the write sets are public. -/
def SingleAssignment : Prop :=
  ∀ s, (s ∈ D.input → ∀ u, s ∉ D.writes u) ∧ (s ∉ D.input → ∃! u, s ∈ D.writes u)

def Dep (σ : S → V) (u v : U) : Prop := ∃ s ∈ D.writes u, s ∈ D.reads v σ

def AcyclicAt (σ : S → V) : Prop := ∀ u, ¬ Relation.TransGen (D.Dep σ) u u

def TimeOrdered (step : U → ℕ) : Prop :=
  ∀ u σ, ∀ s ∈ D.reads u σ, s ∈ D.input ∨ ∃ v, s ∈ D.writes v ∧ step v < step u

def Feeds (σ : S → V) (s t : S) : Prop := ∃ u, s ∈ D.reads u σ ∧ t ∈ D.writes u

def up (σ : S → V) (O : Finset S) : Set S := {s | ∃ o ∈ O, Relation.ReflTransGen (D.Feeds σ) s o}

open Classical in
noncomputable def writesOf (B : Finset U) : Finset S := B.biUnion D.writes

open Classical in
noncomputable def egress [Fintype U] [Fintype S] [Fintype V] (a : S → V) (X O : Finset S) (B : Finset U) :
    Finset (O → V) :=
  (univ.filter fun σ : S → V => D.Holds a X σ ∧ D.incorrect σ ⊆ B).image fun σ o => σ o

end DynPopulation

end Population
