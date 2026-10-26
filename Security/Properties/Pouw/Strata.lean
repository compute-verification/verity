import PouwCheck.Credit

/-!
# A served window's work strata tile its units (`PouwCheck.Credit.strata`)

The property of `strata` (`credit.strata`) that makes `established`'s two numbers the draw's: the strata it builds from a
window's layout hold the window's units call by call, in `h1window.program`'s order, each unit at its call's work, and no
other unit. `callWork` (`credit.call_work`) lists each call's units and the work of each of them; call `i`'s units are
`[callStart i, callStart (i + 1))`. So the credited calls' work `established` passes `epsAt` as `C` is the work of their
units in the strata the draw runs over (`unitWork`), and the window's work it passes as `W` is the strata's
(`Flock.Draw.totalWork`). Each is a `Prop`, and `Proofs.Properties.Pouw.Strata` proves every one, on Lean core.
-/

namespace Properties.Pouw.Strata

open Lean (Json)
open PouwCheck.Credit (callWork)

/-- Unit `u` is in one of stratum `s`'s ranges `[lo, hi)`. -/
def holds (s : Flock.Draw.Stratum) (u : Nat) : Bool := s.units.any fun (lo, hi) => lo ≤ u && u < hi

/-- The work of unit `u` in the strata `ss`: the first stratum's that holds it, and `0` for a unit none holds. -/
def unitWork (ss : Array Flock.Draw.WorkStratum) (u : Nat) : Nat :=
  ((ss.find? fun s => holds s.toStratum u).map (·.work)).getD 0

/-- The first unit of call `i` of `layout`: the units of the calls before it (`callWork`'s sizes). -/
def callStart (layout : Json) (i : Nat) : Nat := (((callWork layout).take i).map (·.1)).sum

/-- **`strata` tiles the window's units, call by call**: wherever it accepts a layout, each unit of call `i` (in
`[callStart i, callStart (i + 1))`) is held by some stratum and does its call's work in every stratum that holds it; no
unit from the window's end on is held; the strata's sizes add up to the window's units, so no unit is held twice; the
strata's work (`Flock.Draw.totalWork`) is the calls' (`callWork`'s units times their work); and the strata's templates,
which label their draw streams, are distinct. -/
def StrataTile : Prop :=
  ∀ (layout : Json) (ss : Array Flock.Draw.WorkStratum), PouwCheck.Credit.strata layout = .ok ss →
    (∀ i (hi : i < (callWork layout).length) u, callStart layout i ≤ u → u < callStart layout (i + 1) →
      (∃ s ∈ ss, holds s.toStratum u = true) ∧
        ∀ s ∈ ss, holds s.toStratum u = true → s.work = ((callWork layout)[i]).2) ∧
    (∀ u, callStart layout (callWork layout).length ≤ u → ∀ s ∈ ss, holds s.toStratum u = false) ∧
    (ss.toList.map (·.size)).sum = callStart layout (callWork layout).length ∧
    Flock.Draw.totalWork ss = ((callWork layout).map fun p => p.1 * p.2).sum ∧
    (ss.toList.map (·.template)).Nodup

end Properties.Pouw.Strata
