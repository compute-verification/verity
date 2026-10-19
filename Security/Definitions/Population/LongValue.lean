import Definitions.Population.Dataflow
import Mathlib.Data.Fintype.Option

/-!
# Long values: slots that may hold no value of their width

A prover may commit a value of any length: an hm96 leaf's inner hash takes a string of any length, so a leaf can commit
to a value longer than its position's width. The verifier derives each position's width from the public structure and
refuses an opening of another length before it reads it, so what it can learn of a slot is a value of the slot's width,
or that the slot has none. This file is the vocabulary for that view.

* `OptCorrect`, `optWrong`: a slot holds `Option V`, `none` (⊥) when it has no opening of its width. Unit `u` is correct
  when every slot it reads holds a value and its written slots hold its function of those values, so a unit that reads
  or writes ⊥ is wrong. `OptHolds`, `optEgress`, `optEgressAt` and `OptSkipsLucky` are `Dataflow`'s over `Option V`.
* `fill d σ`: the assignment of short values that puts `d s` wherever `σ` has ⊥.
* `Openings`: what the verifier sees of one run's commitments: each slot's width, how it decodes a string of that width,
  the strings that climb to the published root at the slot's position, and the strings the run hashes. `Accepts` is the
  verifier's test (the declared width first, then the climb), and `view` its value per slot. `Opened` is what a unit's
  proof establishes: accepted openings of every slot it reads and writes, which satisfy its function.
* `Collides` and `CollisionStep`: two different strings that climb to one position give a collision of the hash among
  the run's hashed strings. `CollisionStep` is a named assumption, taken as a hypothesis.
-/

namespace Population

open Finset

/-- The assignment of short values that puts `d s` on every slot where `σ` has ⊥. -/
def fill {S V : Type} (d : S → V) (σ : S → Option V) : S → V := fun s => (σ s).getD (d s)

namespace Dataflow

variable {U S : Type} (P : Dataflow U S)

/-- Unit `u` is correct in `σ : S → Option V`: every slot it reads holds a value, and its written slots hold its
function of those values. A unit that reads or writes ⊥ is wrong. -/
def OptCorrect {V : Type} (f : P.Fns V) (σ : S → Option V) (u : U) : Prop :=
  ∃ x : P.reads u → V, (∀ r : P.reads u, σ r = some (x r)) ∧ ∀ s : P.writes u, σ s = some (f u x s)

open Classical in
/-- The wrong units of an assignment over `Option V`. -/
noncomputable def optWrong [Fintype U] {V : Type} (f : P.Fns V) (σ : S → Option V) : Finset U :=
  univ.filter fun u => ¬ P.OptCorrect f σ u

/-- `σ` holds the inputs `a`, as values, on every input slot outside `X`. -/
def OptHolds {V : Type} (a : S → V) (X : Finset S) (σ : S → Option V) : Prop :=
  ∀ s ∈ P.input, s ∉ X → σ s = some (a s)

open Classical in
/-- **The egress over `Option V`** at the wrong set `B`: what the assignments over `Option V` that hold the inputs `a`
off the secret slots `X`, and whose wrong units lie in `B`, show on the egress slots `O`, ⊥ included. -/
noncomputable def optEgress [Fintype U] [Fintype S] {V : Type} [Fintype V] (f : P.Fns V) (a : S → V)
    (X O : Finset S) (B : Finset U) : Finset (O → Option V) :=
  (univ.filter fun σ : S → Option V => P.OptHolds a X σ ∧ P.optWrong f σ ⊆ B).image fun σ o => σ o

open Classical in
/-- **The egress over `Option V` with at most `m` wrong units**. -/
noncomputable def optEgressAt [Fintype U] [Fintype S] {V : Type} [Fintype V] (f : P.Fns V) (a : S → V)
    (X O : Finset S) (m : ℕ) : Finset (O → Option V) :=
  (univ.filter fun σ : S → Option V => P.OptHolds a X σ ∧ (P.optWrong f σ).card ≤ m).image fun σ o => σ o

/-- **Work accounting's per-unit premise over `Option V`**: a unit the run skipped, if it reads the honest values and is
correct, is in the lucky set `L`. -/
def OptSkipsLucky {V : Type} (f : P.Fns V) (honest : S → V) (σ : S → Option V) (K L : Finset U) : Prop :=
  ∀ u, u ∉ K → (∀ r ∈ P.reads u, σ r = some (honest r)) → P.OptCorrect f σ u → u ∈ L

end Dataflow

/-- **What the verifier sees of one run's commitments**, slot by slot. `width s` is the length in bytes that the public
structure gives slot `s`'s value, `read s` how the verifier decodes a string of that length, and `climbs s` the strings
that climb to the published root at `s`'s position, with whatever salt and path the prover supplies; a string of any
length may climb, since the inner hash takes any length. `hashed` is the strings the run hashes on the way to its roots,
the prover's long values among them. -/
structure Openings (S V : Type) where
  width : S → ℕ
  read : S → List UInt8 → V
  climbs : S → Set (List UInt8)
  hashed : Set (List UInt8)

namespace Openings

variable {S V : Type} (O : Openings S V)

/-- The verifier accepts the opening `x` at slot `s`: it has the declared width, which the verifier tests before it
reads `x`, and it climbs to the root. -/
def Accepts (s : S) (x : List UInt8) : Prop := x.length = O.width s ∧ x ∈ O.climbs s

open Classical in
/-- **The verifier's view of slot `s`**: the value of an opening it accepts, or ⊥ when it accepts none. -/
noncomputable def view (s : S) : Option V :=
  if h : ∃ x, O.Accepts s x then some (O.read s h.choose) else none

end Openings

/-- **A collision of `H` among the strings `T`**: two different strings of `T` with one digest. -/
def Collides {D : Type} (H : List UInt8 → D) (T : Set (List UInt8)) : Prop :=
  ∃ a ∈ T, ∃ b ∈ T, a ≠ b ∧ H a = H b

/-- **The collision step**, a named assumption: two different strings that climb to one slot's position, whatever their
lengths, give a collision of `H` among the run's hashed strings. It is hm96's binding (`PROTOCOL.md` §4: two openings
of one leaf give a collision of the hash) with frame-v3's length-prefixed nodes on the path; the deployment's `H` is
SHA-512 with its whole output. -/
def CollisionStep {S V D : Type} (H : List UInt8 → D) (O : Openings S V) : Prop :=
  ∀ s, ∀ x ∈ O.climbs s, ∀ y ∈ O.climbs s, x ≠ y → Collides H O.hashed

/-- **What a unit's proof establishes**: the verifier accepts an opening of every slot unit `u` reads and writes (in the
clear or inside the zero-knowledge proof that opens it; either way the width test comes first), and the written values
are `u`'s function of the read ones. -/
def Dataflow.Opened {U S V : Type} (P : Dataflow U S) (O : Openings S V) (f : P.Fns V) (u : U) : Prop :=
  ∃ x : P.reads u → V, (∀ r : P.reads u, ∃ y, O.Accepts r y ∧ O.read r y = x r) ∧
    ∀ s : P.writes u, ∃ y, O.Accepts s y ∧ O.read s y = f u x s

end Population
