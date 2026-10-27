import Definitions.Accounting.Served
import Properties.Accounting.Compose
import SampledProofs.Draw

/-!
# The served window's units and draw are the service's, per call (`SampledProofs.Slots.layout`, `Draw.drawUnits`)

The served replacement of `Properties.Compose.UnitsOfLayout` for a window of per-call Programs
(`Accounting.Served.Shape`, `Accounting.Served.Tie`). `UnitsOfLayout` reads one Program's partition and puts every
value the prover writes at the draw (its commit-coin conjunct, PoUW's seeded salt-and-draw schedule), so it can't sit
beside `LayerCommitted` on a circuit with a run input any gate reads (`SeededScheduleNotLive`): the live statements need
the served schedule, which `Tie.commit` and `Tie.receipt` state. What stays of it here, per call:

* **units** (`CallUnits`): call site `c` is a run of linear `linearOf c`'s Program (`VllmInt8LinearH1{K, N, M = T}`, in
  the form's variant: with each unit's `z` words for `zTree`, its `y` words and the `dequant(z) = y` check for `yOpen`,
  neither variant written yet) on the gates `[start c, start (c + 1))` of the declared circuit; each non-input gate lies
  in a proof-unit span of a replay unit of that Program's layout, and its declared unit is that replay unit's number
  shifted by the call's first position (`Shape.base`); each input slot's gate is an input; and every replay unit's
  number is below the call's unit count (`Shape.unitsOf`), so the calls' units tile the window's positions without
  overlap (W-a);
* **population** (`ServedUnits`): each linear's Program has the shape's unit count, and the declared circuit's units are
  the window's `population`;
* **the draw** (`DrawMissesSubset`): the verifier's `subset:K_d` draw over the window's population from `L` uniform bytes
  (`SampledProofs.Draw.drawUnits`, what `flock-verify draw --population N --k K` runs) misses a set `U` no more often
  than `Pouw.Window.subsetDraw K_d` of a uniform permutation does. It is over the code alone, like
  `Properties.SampledProofs.DrawMisses`.

The seed-preimage conjunct goes: the served live path's rounds are all live (`Tie.live`), and the live statements read no
seed.

These are stated, not proved. The units and population conjuncts are facts of four compiled Programs a form, checkable
once by evaluation; the forms' Programs differ in each unit's spans, never in the unit count, so `population` is one for
all three. The draw conjunct is a property of `drawUnits`, with no premise.
-/

open Lean (Json)
open Pouw.RelativeSecurity (prob)

namespace Properties.Served

open Accounting.Compose Accounting.Served

/-- **Call site `c`'s gates are its Program's**, on layout `l` of linear `linearOf c`'s Program: each of the call's
non-input gates `i` (its gates are `[start c, start (c + 1))`) lies in a proof-unit span of a replay unit of `l`, read as
the Program's root gate `i − start c`, and its declared unit is `base (stepOf c) (linearOf c)` plus that replay unit's
number; each input slot's gate is an input of the call; and every replay unit's number is below the call's units. -/
def CallUnits {ι : Type} (L : Shape) (Cd : Declared ι) (start : ℕ → ℕ) (c : ℕ) (l : SampledProofs.Slots.Layout) :
    Prop :=
  (∀ i, start c ≤ i → i < start (c + 1) → i < Cd.dag.length → Cd.binding i = none →
    ∃ r ∈ l.rus.toList, ∃ u, r.unit = some u ∧ Cd.unit i = L.base (L.stepOf c) (L.linearOf c) + u ∧
      ∃ g ∈ r.pus.toList, g.1 ≤ i - start c ∧ i - start c < g.2) ∧
    (∀ r ∈ l.rus.toList, r.unit = none → ∀ g ∈ r.pus.toList,
      start c + g.1 < start (c + 1) ∧ (Cd.binding (start c + g.1)).isSome) ∧
    (∀ r ∈ l.rus.toList, ∀ u, r.unit = some u → u < L.unitsOf (L.linearOf c))

/-- **The declared circuit's units are the served window's, call by call**: linear `w`'s Program `prog w` under
`query w` has `unitsOf w` units; the declared circuit has the window's `population` of units; each gate lies in its
call site's gate range `[start c, start (c + 1))`; and every call site's gates are its Program's (`CallUnits`), wherever
that Program's layout is defined. -/
def ServedUnits {ι : Type} (L : Shape) (Cd : Declared ι) (prog : ℕ → SampledProofs.Program) (query : ℕ → Json)
    (start : ℕ → ℕ) : Prop :=
  (∀ w, w < L.calls → SampledProofs.Partition.units (prog w) (query w) = .ok (L.unitsOf w)) ∧
    Cd.units = L.population ∧
    (∀ i, i < Cd.dag.length → start (Cd.call i) ≤ i ∧ i < start (Cd.call i + 1)) ∧
    ∀ c, c < Cd.calls → ∀ l, SampledProofs.Slots.layout (prog (L.linearOf c)) (query (L.linearOf c)) = .ok l →
      CallUnits L Cd start c l

/-- **The verifier's subset draw over a population misses no more often than a uniform `K`-subset**: for every `K ≤ N`,
number `L` of uniform bytes and set `U`, `drawUnits (subset K) N` is accepted on the bytes and draws no unit of `U` with
probability at most the probability that `Pouw.Window.subsetDraw K` of a uniform permutation of `Fin N` misses `U`. -/
def DrawMissesSubset : Prop :=
  ∀ (K N L : ℕ) (U : Finset ℕ), K ≤ N →
    prob (fun src : Fin L → Fin 256 => ∃ r,
        SampledProofs.Draw.drawUnits (.subset K) N (Properties.SampledProofs.bytesOf src) = .ok r ∧
          ∀ u ∈ U, u ∉ r.1) ≤
      prob (fun β : Equiv.Perm (Fin N) => ∀ t ∈ Pouw.Window.subsetDraw K β, t.val ∉ U)

/-- **`UnitsOfLayout`'s seeded schedule and `LayerCommitted` exclude each other** on a well-formed declared circuit with
a run input that some gate reads: `UnitsOfLayout` commits the run input at the draw, and `LayerCommitted` needs it by
its first reader's round, which is a round of the window. So no circuit with an activation meets both, and the live
statements can't take `UnitsOfLayout` as written. -/
def SeededScheduleNotLive : Prop :=
  ∀ (ι : Type) (Cd : Declared ι) (Np Kd : ℕ) (p : SampledProofs.Program) (query : Json)
    (l : SampledProofs.Slots.Layout),
    SampledProofs.Slots.layout p query = .ok l → Properties.Compose.UnitsOfLayout Cd Np Kd p query → Cd.Wf →
      (∃ ℓ j, Cd.binding ℓ = some .run ∧ j < Cd.dag.length ∧ Cd.Reads j ℓ) → ¬ LayerCommitted Cd

/-- **The served hypothesis**: declared circuit `Cd`, classed by `cls`, is served window `L` in form `F` (`Tie`), its
units are the per-call Programs' (`ServedUnits`), and the verifier's draw over its population misses as a uniform
subset does (`DrawMissesSubset`). With `TieCommits` and `TieLive` it gives `Scope`'s two protocol conjuncts
(`LayerCommitted`, `Declared.Live`; `TieScope`), and it carries `UnitsOfLayout`'s units and draw, per call. -/
def IsServed {ι : Type} (L : Shape) (F : Form) (Cd : Declared ι) (cls : ℕ → Cls) (prog : ℕ → SampledProofs.Program)
    (query : ℕ → Json) (start : ℕ → ℕ) : Prop :=
  Tie L F Cd cls ∧ ServedUnits L Cd prog query start ∧ DrawMissesSubset

/-- **A tied circuit is in Compose's scope given the three conjuncts the tie doesn't read**: `Tie` gives `Scope`'s
`Declared.Live` (`TieLive`) and `LayerCommitted` (`TieCommits`), so a tied circuit that is well formed, with a positive
`δs` and its scheme's luck proved, meets `Scope`. -/
def TieScope : Prop :=
  ∀ (ι K : Type) (Sc : Scheme K) (L : Shape) (F : Form) (Cd : Declared ι) (cls : ℕ → Cls) (δs : ℝ),
    Tie L F Cd cls → Cd.Wf → 0 < δs → Sc.LuckProved → Scope Sc Cd δs

end Properties.Served
