import Definitions.Pouw.RelativeSecurity
import Properties.Pouw.Strata
import Properties.SampledProofs.Draw

/-!
# The ε a served window's credit reports (`PouwCheck.Credit.established`, `PouwWindow.epsAt`)

The claim behind every ε a credit reports. `periodCredit` reports, for each credited window,
`established layout got served K`: `PouwWindow.epsAt K δ C W 1` at `δ = DELTA = 2⁻⁴⁰`, with `C` the credited calls'
served work (`servedWork`, m_s·n·K a call at its m_s served rows) and `W` the window's units' (`callWork`, M slot rows a
call, the draw's population), where `K` is the K the period's open committed and the draw is `work:K` over the window's strata
(`strata`, then `law`; `drawnOf` runs it). These say what that ε means: the draw misses a set of the window's units doing
at least ε of the credited work with probability at most δ.

* `WorkDrawMisses`: the work law at `K ≥ K′ = kPrime ε δ C W 1` over strata of work `W` misses a set of units of work
  at least `ε·C` with probability at most `δ`, for every `C` (`kPrime` refuses `C` outside `(0, W]`).
* `EpsAtSound`: the same at the ε `epsAt K δ C W 1` answers.
* `EstablishedSound`: the same at the ε `established` answers, with `C` the credited calls' served work
  (`creditedWork`) and `δ = DELTA`: `strata` tiles the window's units call by call at their calls' work
  (`Properties.Pouw.Strata.StrataTile`, proved), so its `W` is the strata's.
* `GroupedDrawMisses`: the same draw, where each drawn unit (a leaf) is proved as its group of blocks (a leaf past the
  block cap as its lanes and their combine, `pouw.leaf_lanes`): a set of blocks doing at least `ε·C` of the work, no
  leaf's blocks doing more than the leaf, has no block in a drawn group with probability at most `δ`. δ counts leaves.

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

open Properties.Pouw.Strata (unitWork)

/-- The units the work draw over `strata` draws from the streams `g`, one per stratum (`PouwCheck.workUnits`'s
`Flock.Draw.stratified`, stratum `i`'s stream `g i`'s bytes, as `Properties.SampledProofs.bytesOf` reads them); none
where a stream runs out. -/
def drawnOn (strata : Array Flock.Draw.Stratum) (g : (i : Fin strata.size) → Fin strata[i].need → Fin 256) :
    Option (Array ℕ) :=
  Flock.Draw.stratified strata (Array.ofFn fun i => Properties.SampledProofs.bytesOf (g i))

/-- The credited calls' served work: each call of `layout` in `got` at its served rows `served` is m_s·(n/C) live
positions of C·K MACs each (`PouwCheck.Credit.servedWork`), never its M slot rows' units (`callWork`). -/
def creditedWork (layout : Json) (got served : List ℕ) : ℕ :=
  ∑ i ∈ (Finset.range (PouwCheck.Credit.servedWork layout served).length).filter (· ∈ got),
    ((PouwCheck.Credit.servedWork layout served).getD i (0, 0)).1 *
      ((PouwCheck.Credit.servedWork layout served).getD i (0, 0)).2

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
a layout with strata `ss`, `law` puts them under `work:K`, and `established layout got served K` answers `ε` for the
credited calls `got` at their served rows `served`, the draw over uniform streams draws no unit of a set `B` whose units
do at least `ε` of the credited calls' served work (`creditedWork`) with probability at most `DELTA`. -/
def EstablishedSound : Prop :=
  ∀ (layout : Json) (ss : Array Flock.Draw.WorkStratum) (K : ℕ) (strata : Array Flock.Draw.Stratum)
    (got served : List ℕ) (e : ℚ),
    PouwCheck.Credit.strata layout = .ok ss → PouwCheck.Credit.law ss (toJson K) = .ok strata →
      PouwCheck.Credit.established layout got served K = some e →
      ∀ B : Finset ℕ, e * creditedWork layout got served ≤ ((∑ u ∈ B, unitWork ss u : ℕ) : ℚ) →
        prob (fun g => ∃ d, drawnOn strata g = some d ∧ ∀ u ∈ B, u ∉ d) ≤ (PouwCheck.Credit.DELTA : ℝ)

/-- **The grouped draw at `K ≥ K′` misses a set of blocks of work `ε·C` with probability at most `δ`**: the work law
draws leaves (`drawnOn`, as in `WorkDrawMisses`), and each drawn leaf `u` is proved as its group, every block `b` with
`leafOf b = u` (a leaf past the block cap: its lanes and their combine, each a one-unit statement). Wherever `strata`
accepts a layout with strata `ss`, `law` puts them under `work:K`, and `kPrime ε δ C W 1` (`W` the strata's work)
answers `K′ ≤ K`, a set `Bb` of blocks whose work `w` sums to at least `ε·C`, the blocks of `Bb` in any one leaf doing at
most that leaf's work (`unitWork`), has no block in a drawn leaf's group with probability at most `δ`. The streams are
uniform and independent, as in `WorkDrawMisses`. -/
def GroupedDrawMisses : Prop :=
  ∀ (layout : Json) (ss : Array Flock.Draw.WorkStratum) (K : ℕ) (strata : Array Flock.Draw.Stratum) (eps δ : ℚ)
    (C k' : ℕ),
    PouwCheck.Credit.strata layout = .ok ss → PouwCheck.Credit.law ss (toJson K) = .ok strata →
      PouwWindow.kPrime eps δ C (Flock.Draw.totalWork ss) 1 = .ok k' →
      ∀ {β : Type} (leafOf w : β → ℕ) (Bb : Finset β), k' ≤ K →
        (∀ u, ∑ b ∈ Bb.filter (fun b => leafOf b = u), w b ≤ unitWork ss u) →
        eps * C ≤ ((∑ b ∈ Bb, w b : ℕ) : ℚ) →
          prob (fun g => ∃ d, drawnOn strata g = some d ∧ ∀ b ∈ Bb, leafOf b ∉ d) ≤ (δ : ℝ)

end Properties.Pouw.Credit
