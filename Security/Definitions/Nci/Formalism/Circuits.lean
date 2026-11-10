module

public import Definitions.Core.Formalism

/-!
# The training formalism v2, §1: circuits

Section 1 of the training formalism v2, on core's one circuit type `Verity.Circuit` and its one declaration
`Verity.RegisteredCircuit`, which say it as written:

* a **circuit** of §1 is a `Verity.Circuit` that is `Wellformed`: every value is a bit (`Circuit.Boolean`) and every gate
  reads one or two values (`Circuit.Boolean`'s positive fan-in and `Circuit.FanIn C 2`). A value is a position
  `Fin C.N`, identified by its source: a gate (`C.computed`) or an input (`C.isInput`);
* a gate **depends on** a value (`DependsOn`) if it reads it, or reads a value computed by a gate that depends on it.
  No gate depends on the value it computes: a gate reads only earlier positions (`Circuit.reads_lt`);
* for a set of gates `S` (`GateSet`), `in(S)` is `C.into S`, the values outside `S` that gates of `S` read, which are
  the values read by gates of `S` that no gate of `S` computes; `|S|` is `S.card`. `in(C)` is `C`'s inputs (`inputs`),
  so the computed values `S` reads are `in(S) \ in(C)`;
* the **reconstruction** of `S` from `x` (`reconstruct`) evaluates `S`'s gates once each, in increasing order, reading
  every other value from `x`; the reconstruction lemma (`Nci.Formalism.reconstruction`, in
  `Security/Proofs/Nci/Formalism/Reconstruction.lean`) says it recovers every value `S` computes from the values of
  `in(S)`, so `S` itself is the witness.

Core's circuits are more general than §1's: values of any width, output gates, gates in topological order. A circuit of
§1 is the `Wellformed` case, and nothing in this formalism reads the rest.
-/

@[expose] public section

namespace Nci.Formalism

open Finset
open Verity (Circuit)

/-- **A circuit of §1**: every value is a bit, and every gate reads one or two values. -/
structure Wellformed (C : Circuit) : Prop where
  boolean : C.Boolean
  fanIn : C.FanIn 2

/-- **`S` is a set of gates**: it holds no input. -/
def GateSet (C : Circuit) (S : Finset (Fin C.N)) : Prop := S ⊆ C.computed

/-- `in(C)`: `C`'s inputs. -/
def inputs (C : Circuit) : Finset (Fin C.N) := univ.filter fun v => C.isInput v = true

/-- `x` agrees with `y` on the values `T`. -/
def AgreesOn (C : Circuit) (x y : C.Trace) (T : Finset (Fin C.N)) : Prop := ∀ v ∈ T, x v = y v

/-- **Gate `g` depends on the value `v`**: some gate reads `v` and feeds `g`, so `g` reads `v` or reads a value computed
by a gate that depends on `v`. -/
def DependsOn (C : Circuit) (g v : Fin C.N) : Prop := ∃ h, v ∈ C.reads h ∧ C.Feeds h g

/-- **`h` feeds `g` within `S`**: a path from `h` to `g`, each value on it read by the next, all of them in `S`. -/
inductive FeedsWithin (C : Circuit) (S : Finset (Fin C.N)) : Fin C.N → Fin C.N → Prop
  | refl {g : Fin C.N} : g ∈ S → FeedsWithin C S g g
  | step {h h' g : Fin C.N} : h ∈ C.reads h' → h ∈ S → FeedsWithin C S h' g → FeedsWithin C S h g

/-- **The reconstruction of `S` from `x`**: `S`'s gates evaluated once each in increasing order, each on the values its
predecessors left, starting from `x`; every value outside `S` keeps `x`'s. -/
def reconstruct (C : Circuit) (S : Finset (Fin C.N)) (x : C.Trace) : C.Trace :=
  (S.sort (· ≤ ·)).foldl (fun y g => Function.update y g (C.op g y)) x

end Nci.Formalism
