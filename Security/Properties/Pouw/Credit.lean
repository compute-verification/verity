module

public import Definitions.Pouw.RelativeSecurity
public import Properties.Pouw.Strata
public import Properties.Spot.Draw
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Topology.Sheaves.Init

/-!
# The ε a served window's credit reports (`PouwCheck.Credit.established`, `PouwWindow.epsAt`)

The claim behind every ε a credit reports. `periodCredit` reports, for each credited window,
`established layout got served K`: `PouwWindow.epsAt K δ C W 1` at `δ = DELTA = 2⁻⁴⁰`, with `C` the credited calls'
served work (`servedWork`, m_s·n·K a call at its m_s served rows) and `W` the window's units' (`callWork`, M slot rows a
call, the sample's population), where `K` is the K the period's open committed and the sample is `work:K` over the window's strata
(`strata`, then `law`; `drawnOf` runs it). These say what that ε means: the sample misses a set of the window's units doing
at least ε of the credited work with probability at most δ.

* `WorkDrawMisses`: the work sampling design at `K ≥ K′ = kPrime ε δ C W 1` over strata of work `W` misses a set of units of work
  at least `ε·C` with probability at most `δ`, for every `C` (`kPrime` refuses `C` outside `(0, W]`).
* `EpsAtSound`: the same at the ε `epsAt K δ C W 1` answers.
* `EstablishedSound`: the same at the ε `established` answers, with `C` the credited calls' served work
  (`creditedWork`) and `δ = DELTA`: `strata` tiles the window's units call by call at their calls' work
  (`Properties.Pouw.Strata.StrataTile`, proved), so its `W` is the strata's.
* `DrawWithheld`: a credited window's sampling coin comes from a session the auditor's ledger opens withheld on the window's
  sample root, so the challenger released its coin commitment only after the prover's commitment was in, and no
  statement here takes a hiding premise on that commitment.
* `GroupedDrawMisses`: the same sample, where each sampled proof block (a leaf) is proved as its group of blocks (a leaf past the
  block cap as its lanes and their combine, `pouw.leaf_lanes`): a set of blocks doing at least `ε·C` of the work, no
  leaf's blocks doing more than the leaf, has no block in a sampled group with probability at most `δ`. δ counts leaves.

The sample is the one `PouwCheck.workUnits` runs: `Flock.Draw.stratified` over the strata `law` gives, stratum `i` sampling
`Flock.Draw.subset` of its `k_i` units from its own stream of `need` bytes (`drawnOn`). The streams are modelled as
independent and uniform, as `Properties.Served.SampleMissesSubset` models them: the service derives stratum `i`'s stream
as `stream key ["draw", template] need`, its label distinct for each stratum (the templates are distinct,
`StrataTile`), and these are that derivation in the random-oracle model. They add no named assumption: the probability
is over the streams' bytes.
-/

@[expose] public section

open Lean (Json toJson)
open Pouw.RelativeSecurity (prob)

namespace Properties.Pouw.Credit

open Properties.Pouw.Strata (unitWork)

/-- The units the work sample over `strata` samples from the streams `g`, one per stratum (`PouwCheck.workUnits`'s
`Flock.Draw.stratified`, stratum `i`'s stream `g i`'s bytes, as `Properties.Spot.bytesOf` reads them); none
where a stream runs out. -/
def drawnOn (strata : Array Flock.Draw.Stratum) (g : (i : Fin strata.size) → Fin strata[i].need → Fin 256) :
    Option (Array ℕ) :=
  Flock.Draw.stratified strata (Array.ofFn fun i => Properties.Spot.bytesOf (g i))

/-- The credited calls' served work: each call of `layout` in `got` at its served rows `served` is m_s·(n/C) live
positions of C·K MACs each (`PouwCheck.Credit.servedWork`), never its M slot rows' units (`callWork`). -/
def creditedWork (layout : Json) (got served : List ℕ) : ℕ :=
  ∑ i ∈ (Finset.range (PouwCheck.Credit.servedWork layout served).length).filter (· ∈ got),
    ((PouwCheck.Credit.servedWork layout served).getD i (0, 0)).1 *
      ((PouwCheck.Credit.servedWork layout served).getD i (0, 0)).2

/-- **The work sampling design at `K ≥ K′` misses a set of work `ε·C` with probability at most `δ`**, the claim behind every ε a
credit reports: wherever `strata` accepts a layout with strata `ss`, `law` puts them under `work:K`, and
`kPrime ε δ C W 1` (`W` the strata's work) answers `K′ ≤ K`, the sample over uniform streams (`drawnOn`) samples no unit of a
set `B` whose units do at least `ε·C` of the work (`unitWork`) with probability at most `δ`. The streams are independent
and uniform, as `Properties.Served.SampleMissesSubset` models them: the service derives stratum `i`'s as
`stream key ["draw", template] need`, its label distinct for each stratum, and this is that derivation in the
random-oracle model. -/
def WorkDrawMisses : Prop :=
  ∀ (layout : Json) (ss : Array Flock.Draw.WorkStratum) (K : ℕ) (strata : Array Flock.Draw.Stratum) (eps δ : ℚ)
    (C k' : ℕ),
    PouwCheck.Credit.strata layout = .ok ss → PouwCheck.Credit.law ss (toJson K) = .ok strata →
      PouwWindow.kPrime eps δ C (Flock.Draw.totalWork ss) 1 = .ok k' →
      ∀ B : Finset ℕ, k' ≤ K → eps * C ≤ ((∑ u ∈ B, unitWork ss u : ℕ) : ℚ) →
        prob (fun g => ∃ d, drawnOn strata g = some d ∧ ∀ u ∈ B, u ∉ d) ≤ (δ : ℝ)

/-- **The ε `epsAt` answers is established**: wherever `strata` accepts a layout with strata `ss`, `law` puts them under
`work:K`, and `epsAt K δ C W 1` (`W` the strata's work) answers `ε`, the sample over uniform streams samples no unit of a set
`B` whose units do at least `ε·C` of the work with probability at most `δ`. -/
def EpsAtSound : Prop :=
  ∀ (layout : Json) (ss : Array Flock.Draw.WorkStratum) (K : ℕ) (strata : Array Flock.Draw.Stratum) (δ e : ℚ) (C : ℕ),
    PouwCheck.Credit.strata layout = .ok ss → PouwCheck.Credit.law ss (toJson K) = .ok strata →
      PouwWindow.epsAt K δ C (Flock.Draw.totalWork ss) 1 = .ok e →
      ∀ B : Finset ℕ, e * C ≤ ((∑ u ∈ B, unitWork ss u : ℕ) : ℚ) →
        prob (fun g => ∃ d, drawnOn strata g = some d ∧ ∀ u ∈ B, u ∉ d) ≤ (δ : ℝ)

/-- **The ε a credit reports is established**, the claim behind every ε `periodCredit` reports: wherever `strata` accepts
a layout with strata `ss`, `law` puts them under `work:K`, and `established layout got served K` answers `ε` for the
credited calls `got` at their served rows `served`, the sample over uniform streams samples no unit of a set `B` whose units
do at least `ε` of the credited calls' served work (`creditedWork`) with probability at most `DELTA`. -/
def EstablishedSound : Prop :=
  ∀ (layout : Json) (ss : Array Flock.Draw.WorkStratum) (K : ℕ) (strata : Array Flock.Draw.Stratum)
    (got served : List ℕ) (e : ℚ),
    PouwCheck.Credit.strata layout = .ok ss → PouwCheck.Credit.law ss (toJson K) = .ok strata →
      PouwCheck.Credit.established layout got served K = some e →
      ∀ B : Finset ℕ, e * creditedWork layout got served ≤ ((∑ u ∈ B, unitWork ss u : ℕ) : ℚ) →
        prob (fun g => ∃ d, drawnOn strata g = some d ∧ ∀ u ∈ B, u ∉ d) ≤ (PouwCheck.Credit.DELTA : ℝ)

/-- **A credited window's sampling coin comes from a withheld session on its sample root** (a challenger releases its coin
commitment only after the prover's commitment is in). Wherever `credited` accepts a window,
the sample's `root` is a string `r` (the root `draw_root` gives over the window's manifest, Commit record, receipts and
round log, which `credited` recomputes), and the auditor's ledger opens a session on `r` and marks every open of it
withheld (`withheldOn`, `remote.open_session(..., withhold=True)`): that session released only its coins' raw bytes
until it closed, so its commitment came after the prover's sample root. -/
def DrawWithheld : Prop :=
  ∀ key ledger pins main salt calls served commit rounds draw got,
    PouwCheck.Credit.Internal.credited key ledger pins main salt calls served commit rounds draw = .ok got →
      ∃ r, PouwCheck.get draw "root" = .str r ∧ Spot.Challenger.withheldOn ledger r = true

/-- **The grouped sample at `K ≥ K′` misses a set of blocks of work `ε·C` with probability at most `δ`**: the work sampling design
samples leaves (`drawnOn`, as in `WorkDrawMisses`), and each sampled leaf `u` is proved as its group, every block `b` with
`leafOf b = u` (a leaf past the block cap: its lanes and their combine, each a one-unit statement). Wherever `strata`
accepts a layout with strata `ss`, `law` puts them under `work:K`, and `kPrime ε δ C W 1` (`W` the strata's work)
answers `K′ ≤ K`, a set `Bb` of blocks whose work `w` sums to at least `ε·C`, the blocks of `Bb` in any one leaf doing at
most that leaf's work (`unitWork`), has no block in a sampled leaf's group with probability at most `δ`. The streams are
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

/-! ## A window of leaves proved as their lanes and combines

The instances `GroupedDrawMisses`'s two lemmas read (`Proofs.Pouw.Credit`). Its honest lemma: a served H1 window of
sixteen 32 × 32 leaves at K = 2048, past the block cap, each proved as its 32 lanes (its 4 × 8 tiles, each doing
4·8·2048 of its MACs) and their combine (doing none), sampled under `work:8` at ε = 1/2 and δ = 1/256, where `K′ = 8`.
Its nontrivial lemma: a sampling design that accepts everything may hand the sample no strata, and then every block escapes. -/

/-- The window: one call of one linear `w`, K = 2048 and n = 512 columns, at 32 × 32 tiles and a 32-row A slot, so
sixteen leaves of 32·32·2048 MACs each. -/
def lanesLayout : Json :=
  Json.mkObj [("calls", Json.arr #[Json.arr #[Json.str "w", toJson (0 : ℕ)]]),
    ("linears", Json.arr #[Json.arr #[Json.str "w", toJson (2048 : ℕ), toJson (512 : ℕ)]]),
    ("m", toJson (32 : ℕ)), ("tile", Json.arr #[toJson (32 : ℕ), toJson (32 : ℕ)])]

/-- Its strata (`PouwCheck.Credit.strata`): K = 2048's, its sixteen leaves at 32·32·2048 each. -/
def lanesStrata : Array Flock.Draw.WorkStratum :=
  #[{ template := "k2048", units := #[(0, 16)], k := 0, work := 32 * 32 * 2048, floor := 1 }]

/-- The K its period committed. -/
def lanesK : ℕ := 8

/-- Its strata under `work:8` (`PouwCheck.Credit.law`): eight samples from K = 2048's sixteen leaves. -/
def lanesLaw : Array Flock.Draw.Stratum := #[{ template := "k2048", units := #[(0, 16)], k := 8 }]

/-- The blocks: leaf `u`'s are `33u, …, 33u + 32`, its 32 lanes and then their combine. -/
def lanesBlocks : Finset ℕ := Finset.range (16 * 33)

/-- A block's leaf. -/
def lanesLeaf (b : ℕ) : ℕ := b / 33

/-- A block's work: a lane's 4·8·2048 MACs, and none for a combine. -/
def lanesWork (b : ℕ) : ℕ := if b % 33 < 32 then 4 * 8 * 2048 else 0

/-- The strata a sampling design that accepts everything may give: none. -/
def idleLaw : Array Flock.Draw.Stratum := #[]

end Properties.Pouw.Credit
