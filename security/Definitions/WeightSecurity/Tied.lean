module

public import Definitions.Spot.Dataflow
public import Definitions.WeightSecurity.Served

/-!
# Weight security at a population, each window with its extracted trace

The served circuit is a population (`Population.Dataflow`) with unit functions `f`, inputs `inp θ x` at the weights `θ`
and a query served with an advice `x`, and egress slots `eg (o, i)`, by charge unit `o` and position `i` in its output.
Its served function (`popServed`) is the population's function (`Population.Dataflow.outOf`) read unit by unit. A run
records, of each window, what `Served` records and the trace the audit extracts from the window's commitments
(`Served.Traced`): an assignment of the population, whose incorrect proof blocks (`Population.Dataflow.incorrect`) are
the window's.

* `Population.Dataflow.FanOut P eg F`: each proof block's reach (`coneEgress`) holds the egress of at most `F` charge
  units (`unitsInReach`), so an incorrect proof block changes at most `F` of them;
* `Served.Traced.counted P f w`: the window's record with its incorrect proof blocks counted at its trace, and
  `Served.Traced.countRun P f run` the run of such records, the run the audit's undetected bound is about;
* `Served.Traced.Tied eg L l`: each accepted window that releases released its trace's values on the egress slots
  (the network certifier's egress tie: the released bytes are the trace's outputs);
* `Served.Traced.Covered P inp Ou L θ r l`: each accepted window that releases holds, on the population's input slots,
  the inputs of the weights and of the outsider's query served with the window's advice (advice coverage: whatever
  else the prover chose is in the advice, which is charged).

On a run where both hold, an accepted release differs from the served function's answer only on the charge units in
the reaches of its trace's incorrect proof blocks, so the run counted at its traces is `Served.Isolated` at `F`.
-/

@[expose] public section

namespace Population.Dataflow

variable {U S : Type} (P : Dataflow U S)

open Classical in
/-- **The population's function**: its values on the slots `eg` at the inputs `a`, those of the assignment that holds
`a` on the input slots and makes every unit correct, and `a`'s own where there is none. A checked population has
exactly one. -/
noncomputable def outOf {V : Type} (f : P.Fns V) {O : Type} (eg : O → S) (a : S → V) : O → V :=
  fun o => if h : ∃ σ : S → V, (∀ s ∈ P.input, σ s = a s) ∧ ∀ u, P.Correct f σ u then Classical.choose h (eg o)
    else a (eg o)

open Classical in
/-- **The charge units in `u`'s reach**: those with an egress slot `eg (o, i)` in it (`coneEgress`, over the slots `eg`
names). A wrong `u` changes no other's output. -/
noncomputable def unitsInReach [Fintype U] {O I : Type} [Fintype O] [Fintype I] (eg : O × I → S) (u : U) : Finset O :=
  Finset.univ.filter fun o => ∃ i, eg (o, i) ∈ P.coneEgress (Finset.univ.image eg) {u}

/-- **Each unit's reach holds at most `F` charge units.** With each request's units writing only its own rows and no
request reading another's, each unit's holds one: its own request. A request whose cached prefix is another's rows is in
the reach of the units that wrote them, so with requests as the charge units, `F` is then the most requests one
request's rows reach: itself and every request that reads them, directly or through another's. -/
def FanOut [Fintype U] {O I : Type} [Fintype O] [Fintype I] (eg : O × I → S) (F : ℕ) : Prop :=
  ∀ u, (P.unitsInReach eg u).card ≤ F

end Population.Dataflow

namespace WeightSecurity

/-- **A population's served function**: at the weights `θ` and a query served with an advice, `x`, charge unit `o`'s
output at position `i` is the population `P`'s function at the inputs `inp θ x` on the egress slot `eg (o, i)`. -/
noncomputable def popServed {U S Θ X O I V : Type} (P : Population.Dataflow U S) (f : P.Fns V) (eg : O × I → S)
    (inp : Θ → X → S → V) : Θ → X → O → I → V :=
  fun θ x o i => P.outOf f eg (inp θ x) (o, i)

/-- **A served window with its extracted trace**: what `Served` records of it, with `W` a charge unit's output, and the
assignment of the served circuit's population that the audit extracts from the window's commitments. -/
structure Served.Traced (O W Tick Adv S V : Type) extends Served O W Tick Adv where
  trace : S → V

namespace Served.Traced

variable {O I V Tick Adv S U R Q Θ Θ' : Type}

/-- **The window counted at its trace**: its record, with its incorrect proof blocks those of its trace under the
population `P`'s unit functions `f`. -/
noncomputable def counted [Fintype U] (P : Population.Dataflow U S) (f : P.Fns V) (w : Traced O (I → V) Tick Adv S V) :
    Served O (I → V) Tick Adv :=
  { w.toServed with incorrect := (P.incorrect f w.trace).card }

/-- **The run counted at its traces**: each window's record counted at its trace. -/
noncomputable def countRun [Fintype U] (P : Population.Dataflow U S) (f : P.Fns V)
    (run : Θ → R → Verity.Game (List (Traced O (I → V) Tick Adv S V))) :
    Θ → R → Verity.Game (List (Served O (I → V) Tick Adv)) :=
  fun θ r => (run θ r).map (List.map (counted P f))

/-- **The releases are their traces' outputs**: each accepted window that releases released, as each charge unit `o`'s
output, its trace's values on the egress slots `eg (o, ·)`. -/
def Tied (eg : O × I → S) (L : ℕ) (l : List (Traced O (I → V) Tick Adv S V)) : Prop :=
  ∀ n < count L (l.map toServed), ∀ w, l[n]? = some w → w.accepted → w.egress = fun o i => w.trace (eg (o, i))

/-- **The traces hold the windows' inputs**: each accepted window that releases holds, on the input slots of the
population `P`, the inputs `inp θ x` of the weights `θ` and of the outsider's query served with the window's advice,
`x`. -/
def Covered (P : Population.Dataflow U S) (inp : Θ → Q × Adv → S → V)
    (Ou : Outsider R (Q × Adv) (Option ((O → I → V) × Tick)) Θ') (L : ℕ) (θ : Θ) (r : R)
    (l : List (Traced O (I → V) Tick Adv S V)) : Prop :=
  ∀ n < count L (l.map toServed), ∀ w, l[n]? = some w → w.accepted →
    ∀ s ∈ P.input, w.trace s = inp θ ((Ou.query r (view Ou L r (l.map toServed) n)).1, w.adv) s

end Served.Traced

end WeightSecurity
