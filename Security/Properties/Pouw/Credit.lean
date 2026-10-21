import Definitions.Pouw.RelativeSecurity
import Properties.Pouw.Strata
import Properties.SampledProofs.Draw

/-!
# The ε a served window's credit reports (`PouwCheck.Credit.established`, `PouwWindow.epsAt`)

The claim behind every ε a credit reports. `periodCredit` reports, for each credited window, `established layout got K`:
`PouwWindow.epsAt K δ C W 1` at `δ = DELTA = 2⁻⁴⁰`, with `C` the credited calls' work and `W` the window's
(`callWork`), where `K` is the K the period's open committed and the draw is `work:K` over the window's strata
(`strata`, then `law`; `drawnOf` runs it). These say what that ε means: the draw misses a set of the window's units doing
at least ε of the credited work with probability at most δ.

* `WorkDrawMisses`: the work law at `K ≥ K′ = kPrime ε δ C W 1` over strata of work `W` misses a set of units of work
  at least `ε·C` with probability at most `δ`, for every `C` (`kPrime` refuses `C` outside `(0, W]`).
* `EpsAtSound`: the same at the ε `epsAt K δ C W 1` answers.
* `EstablishedSound`: the same at the ε `established` answers, with `C` the work of the credited calls' units in the
  draw's strata (`creditedWork`) and `δ = DELTA`: `strata` tiles the window's units call by call at their calls' work
  (`Properties.Pouw.Strata.StrataTile`, proved), so its `C` and `W` are the strata's.

The draw is the one `PouwCheck.workUnits` runs: `Flock.Draw.stratified` over the strata `law` gives, stratum `i` drawing
`Flock.Draw.subset` of its `k_i` units from its own stream of `need` bytes (`drawnOn`). The streams are modelled as
independent and uniform, as `Properties.Served.DrawMissesSubset` and `Claims.Sampling.live_work_miss_le` model them:
the service derives stratum `i`'s stream as `stream key ["draw", template] need`, its label distinct for each stratum
(the templates are distinct, `StrataTile`), and these are that derivation in the random-oracle model. They add no named
assumption: the probability is over the streams' bytes.
-/

open Lean (Json toJson)
open Pouw.RelativeSecurity (prob)

namespace Properties.Pouw.Credit

open Properties.Pouw.Strata (unitWork callStart)

/-- The units the work draw over `strata` draws from the streams `g`, one per stratum (`PouwCheck.workUnits`'s
`Flock.Draw.stratified`, stratum `i`'s stream `g i`'s bytes, as `Properties.SampledProofs.bytesOf` reads them); none
where a stream runs out. -/
def drawnOn (strata : Array Flock.Draw.Stratum) (g : (i : Fin strata.size) → Fin strata[i].need → Fin 256) :
    Option (Array ℕ) :=
  Flock.Draw.stratified strata (Array.ofFn fun i => Properties.SampledProofs.bytesOf (g i))

/-- The credited calls' work in the strata `ss`: the work of each unit of each call of `layout` in `got`
(`unitWork`, call `i`'s units `[callStart i, callStart (i + 1))`). -/
def creditedWork (ss : Array Flock.Draw.WorkStratum) (layout : Json) (got : List ℕ) : ℕ :=
  ∑ i ∈ (Finset.range (PouwCheck.Credit.callWork layout).length).filter (· ∈ got),
    ∑ u ∈ Finset.Ico (callStart layout i) (callStart layout (i + 1)), unitWork ss u

/-- **The work law at `K ≥ K′` misses a set of work `ε·C` with probability at most `δ`**, the claim behind every ε a
credit reports: wherever `strata` accepts a layout with strata `ss`, `law` puts them under `work:K`, and
`kPrime ε δ C W 1` (`W` the strata's work) answers `K′ ≤ K`, the draw over uniform streams (`drawnOn`) draws no unit of a
set `B` whose units do at least `ε·C` of the work (`unitWork`) with probability at most `δ`. The streams are independent
and uniform, as `Properties.Served.DrawMissesSubset` and `Claims.Sampling.live_work_miss_le` model them: the service
derives stratum `i`'s as `stream key ["draw", template] need`, its label distinct for each stratum, and this is that
derivation in the random-oracle model. -/
def WorkDrawMisses : Prop :=
  ∀ (layout : Json) (ss : Array Flock.Draw.WorkStratum) (K : ℕ) (strata : Array Flock.Draw.Stratum) (eps δ : ℚ)
    (C k' : ℕ),
    PouwCheck.Credit.strata layout = .ok ss → PouwCheck.Credit.law ss (toJson K) = .ok strata →
      PouwWindow.kPrime eps δ C (Flock.Draw.totalWork ss) 1 = .ok k' →
      ∀ B : Finset ℕ, k' ≤ K → eps * C ≤ ((∑ u ∈ B, unitWork ss u : ℕ) : ℚ) →
        prob (fun g => ∃ d, drawnOn strata g = some d ∧ ∀ u ∈ B, u ∉ d) ≤ (δ : ℝ)

/-- **The ε `epsAt` answers is established**: wherever `strata` accepts a layout with strata `ss`, `law` puts them under
`work:K`, and `epsAt K δ C W 1` (`W` the strata's work) answers `ε`, the draw over uniform streams draws no unit of a set
`B` whose units do at least `ε·C` of the work with probability at most `δ`. -/
def EpsAtSound : Prop :=
  ∀ (layout : Json) (ss : Array Flock.Draw.WorkStratum) (K : ℕ) (strata : Array Flock.Draw.Stratum) (δ e : ℚ) (C : ℕ),
    PouwCheck.Credit.strata layout = .ok ss → PouwCheck.Credit.law ss (toJson K) = .ok strata →
      PouwWindow.epsAt K δ C (Flock.Draw.totalWork ss) 1 = .ok e →
      ∀ B : Finset ℕ, e * C ≤ ((∑ u ∈ B, unitWork ss u : ℕ) : ℚ) →
        prob (fun g => ∃ d, drawnOn strata g = some d ∧ ∀ u ∈ B, u ∉ d) ≤ (δ : ℝ)

/-- **The ε a credit reports is established**, the claim behind every ε `periodCredit` reports: wherever `strata` accepts
a layout with strata `ss`, `law` puts them under `work:K`, and `established layout got K` answers `ε` for the credited
calls `got`, the draw over uniform streams draws no unit of a set `B` whose units do at least `ε` of the credited calls'
work (`creditedWork`) with probability at most `DELTA`. -/
def EstablishedSound : Prop :=
  ∀ (layout : Json) (ss : Array Flock.Draw.WorkStratum) (K : ℕ) (strata : Array Flock.Draw.Stratum) (got : List ℕ)
    (e : ℚ),
    PouwCheck.Credit.strata layout = .ok ss → PouwCheck.Credit.law ss (toJson K) = .ok strata →
      PouwCheck.Credit.established layout got K = some e →
      ∀ B : Finset ℕ, e * creditedWork ss layout got ≤ ((∑ u ∈ B, unitWork ss u : ℕ) : ℚ) →
        prob (fun g => ∃ d, drawnOn strata g = some d ∧ ∀ u ∈ B, u ∉ d) ≤ (PouwCheck.Credit.DELTA : ℝ)

end Properties.Pouw.Credit
