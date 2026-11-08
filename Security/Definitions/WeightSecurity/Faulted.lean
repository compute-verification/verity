module

public import Definitions.WeightSecurity.Served

/-!
# Weight security under SPOT's fault model: faults as inputs to the served circuit

SPOT's guarantee is a fault model: an accepted server's outputs are the honest circuit's with at most `M` incorrect proof
blocks, each writing what the server chose. Here a fault is an input to the circuit, not an overwrite of what it
releases. A fault pattern `φ : F` names some proof blocks and the values they write instead, and `size φ` counts the
blocks. The faulted served function `Cf θ (x, φ)` is the circuit's run at the weights `θ` and the input `x` with the
blocks `φ` names writing `φ`'s values; a checked population's run is unique (`Population.Dataflow.RunOf`), so its
egress is a function of the inputs and the pattern alone.

* `Outsider.withFaults O φ₀`: the outsider asking the faulted function, each query with the pattern `φ₀` beside it;
* `faultChannel … size T M K`: the covert channel of a deployment whose releases are faulted runs. A covert string is
  each window's fault pattern, the sizes summing to at most `M`, and then the served channel's string: each window's
  advice and timing bucket, an overwrite of at most `K` released values, and the halt;
* `Served.Faithful Cf size Ou L θ r l`: each accepted window that releases releases the faulted function's answer to the
  outsider's query at the window's advice and a pattern of size at most the window's incorrect proof blocks.

`Faithful` replaces `Served.Local`'s reach: a faulted run's release may differ from the honest answer anywhere
downstream of its faults, and the channel charges the faults, not what they reach.
-/

@[expose] public section

namespace WeightSecurity

variable {R X M Θ' F : Type}

/-- **The outsider asking a faulted function**: `O`'s queries, each with the fault pattern `φ₀` beside it, and `O`'s
output on what it saw with the patterns dropped. -/
def Outsider.withFaults (O : Outsider R X M Θ') (φ₀ : F) : Outsider R (X × F) M Θ' where
  query r h := (O.query r (h.map fun e => (e.1.1, e.2)), φ₀)
  out r h := O.out r (h.map fun e => (e.1.1, e.2))

/-- **The fault channel** over `T` windows, at fault patterns `F` of sizes `size`. A covert string is each window's
fault pattern, the patterns' sizes summing to at most `M`; each window's advice and timing bucket; an overwrite of at
most `K` of the `T · |O|` released values; and how many windows release before the halt. With it the datacenter
evaluates the faulted function at the outsider's `n`-th query with the `n`-th advice and the `n`-th pattern, and, for the
windows before the halt, tells it that answer with the overwrite applied, at the `n`-th bucket; after the halt it tells
it nothing. Past the `T`-th query it evaluates the query as asked. -/
noncomputable def faultChannel (R Q Adv O V Tick F : Type) [Fintype Adv] [Fintype O] [DecidableEq O] [Fintype V]
    [Fintype Tick] [Fintype F] (size : F → ℕ) (T M K : ℕ) :
    Channel R ((Q × Adv) × F) (O → V) (Option ((O → V) × Tick)) where
  S := {φ : Fin T → F // ∑ t, size (φ t) ≤ M} × (Fin T → Adv) × (Fin T → Tick) ×
    {d : Fin T × O → Option V // (Finset.univ.filter fun p => d p ≠ none).card ≤ K} × Fin (T + 1)
  ask s _ h x := if hn : h.length < T then ((x.1.1, s.2.1 ⟨h.length, hn⟩), s.1.val ⟨h.length, hn⟩) else x
  tell s _ h _ y := if hn : h.length < s.2.2.2.2.val ∧ h.length < T then
      some (fun o => (s.2.2.2.1.val (⟨h.length, hn.2⟩, o)).getD (y o), s.2.2.1 ⟨h.length, hn.2⟩)
    else none

namespace Served

variable {O V Tick Adv Q Θ : Type}

/-- **The releases are faulted runs**: each accepted window that releases released the faulted function `Cf`'s answer
at the weights `θ` to the outsider's query, served with the window's advice, at some fault pattern of size at most the
window's incorrect proof blocks. It is what the layers below the composition give: the network certifier ties the
release to the trace's outputs, and a checked population's trace is the run with its incorrect proof blocks' writes
fixed (`Population.Dataflow.RunOf`). -/
def Faithful (Cf : Θ → (Q × Adv) × F → O → V) (size : F → ℕ)
    (Ou : Outsider R (Q × Adv) (Option ((O → V) × Tick)) Θ') (L : ℕ) (θ : Θ) (r : R)
    (l : List (Served O V Tick Adv)) : Prop :=
  ∀ n < count L l, ∀ w, l[n]? = some w → w.accepted →
    ∃ φ, size φ ≤ w.incorrect ∧ w.egress = Cf θ (((Ou.query r (view Ou L r l n)).1, w.adv), φ)

end Served

end WeightSecurity
