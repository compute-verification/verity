module

public import Definitions.Spot.Faults
public import Definitions.WeightSecurity.Faulted

/-!
# Weight security under SPOT's fault model, each window with its extracted trace

The served circuit is a population (`Population.Dataflow`) with unit functions `f`, egress slots `eg`, and inputs
`inp θ x` at the weights `θ` and a query served with an advice `x`. Its faulted function (`faultedServed`) is the
population's (`Population.Dataflow.faultedOut`), and a fault pattern's size is how many units it names. A run records,
of each window, what `Served` records and the trace the audit extracts from the window's commitments
(`Served.Traced`): an assignment of the population, whose wrong units are the window's incorrect proof blocks.

* `Served.Traced.counted P f w`: the window's record with its trace's wrong units as its incorrect proof blocks, and
  `Served.Traced.countRun P f run` the run of such records, the run the audit's undetected bound is about;
* `Served.Traced.Tied eg L l`: each accepted window that releases released its trace's values on the egress slots
  (the network certifier's egress tie: the released bytes are the trace's outputs);
* `Served.Traced.Covered P inp Ou L θ r l`: each accepted window that releases holds, on the population's input slots,
  the inputs of the weights and of the outsider's query served with the window's advice (advice coverage: whatever
  else the prover chose is in the advice, which is charged).

On a run where both hold, each accepted release is the faulted function's answer at its trace's own fault pattern, so
the run counted at its traces is `Served.Faithful`.
-/

@[expose] public section

namespace WeightSecurity

/-- **The served circuit's faulted function** when the circuit is the population `P` with unit functions `f`: at the
weights `θ` and a query served with an advice, `x.1`, the values on the egress slots `eg` of the run with the inputs
`inp θ x.1` and the fault pattern `x.2`. -/
noncomputable def faultedServed {U S O V Θ Q Adv : Type} (P : Population.Dataflow U S) (f : P.Fns V) (eg : O → S)
    (inp : Θ → Q × Adv → S → V) : Θ → (Q × Adv) × P.Faults V → O → V :=
  fun θ x => P.faultedOut f eg (inp θ x.1) x.2

/-- **A served window with its extracted trace**: what `Served` records of it, and the assignment of the served
circuit's population that the audit extracts from the window's commitments. -/
structure Served.Traced (O V Tick Adv S : Type) extends Served O V Tick Adv where
  trace : S → V

namespace Served.Traced

variable {O V Tick Adv S U R Q Θ Θ' : Type}

/-- **The window counted at its trace**: its record, with its trace's wrong units under the population `P`'s unit
functions `f` as its incorrect proof blocks. -/
noncomputable def counted [Fintype U] (P : Population.Dataflow U S) (f : P.Fns V) (w : Traced O V Tick Adv S) :
    Served O V Tick Adv :=
  { w.toServed with incorrect := (P.wrong f w.trace).card }

/-- **The run counted at its traces**: each window's record counted at its trace. -/
noncomputable def countRun [Fintype U] (P : Population.Dataflow U S) (f : P.Fns V)
    (run : Θ → R → Verity.Game (List (Traced O V Tick Adv S))) : Θ → R → Verity.Game (List (Served O V Tick Adv)) :=
  fun θ r => (run θ r).map (List.map (counted P f))

/-- **The releases are their traces' outputs**: each accepted window that releases released its trace's values on the
egress slots `eg`. -/
def Tied (eg : O → S) (L : ℕ) (l : List (Traced O V Tick Adv S)) : Prop :=
  ∀ n < count L (l.map toServed), ∀ w, l[n]? = some w → w.accepted → w.egress = fun o => w.trace (eg o)

/-- **The traces hold the windows' inputs**: each accepted window that releases holds, on the input slots of the
population `P`, the inputs `inp θ x` of the weights `θ` and of the outsider's query served with the window's advice,
`x`. -/
def Covered (P : Population.Dataflow U S) (inp : Θ → Q × Adv → S → V)
    (Ou : Outsider R (Q × Adv) (Option ((O → V) × Tick)) Θ') (L : ℕ) (θ : Θ) (r : R)
    (l : List (Traced O V Tick Adv S)) : Prop :=
  ∀ n < count L (l.map toServed), ∀ w, l[n]? = some w → w.accepted →
    ∀ s ∈ P.input, w.trace s = inp θ ((Ou.query r (view Ou L r (l.map toServed) n)).1, w.adv) s

end Served.Traced

end WeightSecurity
