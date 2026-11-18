module

public import Definitions.Core.Scheme
public import Definitions.Pouw.PearlC.TileSample

/-!
# Pearl+ as a scheme (`Verity.Scheme`)

Pearl+'s instance of `Verity.Scheme`, a family over its game's setting (`Setting`): the cost model, Π and its audit
tiles, the sample's weight, the workload, the adversary and its declarations, the number of samples, the budget, the
TT_OUT target `g₀`, the gap `γ = γn/γd`, the resolution `D`, the oracle bound `q` and the error `η(q, N)`. Its game is
Pearl-C's sampled audit sampled as the run samples (`Pouw.PearlC.GGapSampledDraw`, `Definitions.Pouw.PearlC.TileSample`),
whose python is `verity.protocols.resource_accounting.compute.pouw.scheme` (`pearl-plus`).

* **Blocks** are the layout's audit tiles, `Fin NT` of `TR.tiling`.
* **The unit.** The sides are naturals in one unit: `1/(D·γd)` of the cost model's unit (one dense FP8 MAC at the H100's
  record). `over` is the claimed work (`claimedDraw`, the run's ticket weight) times `D·(γd − γn)`, rounded down;
  `under` is the budget `T` times `D·γd`, rounded up; so `over ≤ under` is `(1 − γ)·claimed ≤ T` up to the rounding,
  which only weakens `over`. The sample's share `ε_s` is not in the sides: `Scheme.missable` carries it.
* **A tile's cost** is its sample weight `Wd` at its unit's full declaration (every row and column credited, the unit's
  `k` kept, `fullDecl`) times `D·(γd − γn)`, rounded up: at least what any declaration of it adds to `over`, as
  `pearlCDrawRev1` is monotone in the declared cells and kept width. The python's cost is the tile's MACs at the
  record's kept width (`TM·TN·K_KEPT`); this one is in the sample's `W_ref` units, at least `K` a cell.
* **Accepted** is the audit that accepts only checked tiles (`AuditAccepts` over the good tiles whose declaration is
  consistent and the blank units' tiles), the verifier's `over_cap = reject`. The run of record's tolerant audit
  (`uncredited`, `AuditAcceptsTol`) accepts a run whose sample lands on a tile over its inert-work bound and deducts
  `u(X_O)` of the claimed work; a scheme has no run-dependent deduction, so this instance does not cover it.
* **Incorrect** are the tiles neither good with a consistent declaration nor of a blank unit.
* **The log.** The game has none: its order is its quantifiers. The declarations are a function of the oracle and the
  salt, fixed with the trace, and the sample is drawn over the tickets they give. `logOK` holds of every run; with a
  challenger's log it is `Verity.Challenger.Log.Cut` at the run commitment and the coin `pouw/draw`, which the code
  checks (`PouwVerifier.PearlPlus.drawReason`).
* **Breaks** is TT_OUT per tile's event at the budget: the good tiles' credit exceeds `T/(1 − g₀)`
  (`Pouw.PearlC.Assumptions.TTOutTile`). Its bound, the scheme's own term, is `η(q, N)`, which TT_OUT per tile gives for
  an adversary within `(T, q)`. Pearl-C's commitments are ideal, so no hash break is here; the code's
  (`Properties.Pouw.PearlPlus.OpeningBinds`) is the link's.
* **Charges** are none: the inert-work share is in the tiles' credit and in `γ`, the sample's share is the missable
  cost, a tile over its bound is incorrect and an excluded row is credited nothing.

**The one design is not met.** `Scheme.GetsBack` reads one sampling design over the blocks. Pearl+'s tickets
(`drawTickets`) are the declared tiles' weights, and the declarations (each unit's real rows, kept width, excluded and
voluntary rows) are the prover's, fixed after the salt and recorded before the sample's coin. The design is a function
of the public layout and of what is recorded before the coin, so it is one design per run commitment, not one for the
scheme.

**The charge is an open premise** (`ChargeHolds`): the scheme is a function of its proof, which nothing proves.
-/

@[expose] public section

namespace Pouw.PearlPlus

open Finset
open scoped ENNReal

/-- **Pearl+'s setting**: the game the scheme is an instance at, with its own sides' parameters. `dec` gives each unit's
declaration on the oracle and salt; `T` is the adversary's budget; `g₀` is TT_OUT's target, `γn/γd` the gap, `D` the
resolution and `η(q, N)` TT_OUT's error. -/
structure Setting where
  Q : Type
  R : Type
  S : Type
  CM : PearlC.CostModel Q R S
  P : PearlC.Protocol Q R S
  TR : PearlC.TileRules Q R S
  Wd : PearlC.Layout → ℕ → PearlC.Decl → ℚ
  U : PearlC.Workload
  A : PearlC.Adversary CM
  dec : (Q → R) → S → ℕ → PearlC.Decl
  t : ℕ
  T : ℝ
  g₀ : ℝ
  γn : ℕ
  γd : ℕ
  D : ℕ
  q : ℕ
  η : ℕ → ℕ → ℝ

namespace Setting

variable (σ : Setting)

/-- The audit tiles of the workload's layout. -/
abbrev Tile : Type := Fin (σ.TR.tiling σ.U.layout).NT

/-- A run: the oracle and the salt, then `t` samples over the tickets of the trace and declarations they give. -/
abbrev Run : Type :=
  Σ x : (σ.Q → σ.R) × σ.S,
    Fin σ.t → Ticket (PearlC.drawTickets σ.TR σ.Wd σ.U (σ.A.trace σ.U x.1 x.2) (σ.dec x.1 x.2)) (σ.TR.tiling σ.U.layout).NT

/-- The run's trace. -/
def trace (r : σ.Run) : PearlC.Trace := σ.A.trace σ.U r.1.1 r.1.2

/-- The tiles a sampled check passes: good with a consistent declaration, or of a blank unit. -/
noncomputable def passed (r : σ.Run) : Finset ℕ :=
  PearlC.goodDeclTiles σ.P σ.TR σ.U r.1.1 r.1.2 (σ.trace r) (σ.dec r.1.1 r.1.2) ∪ PearlC.blankTiles σ.TR σ.U (σ.trace r)

/-- Unit `u`'s full declaration: its `k` kept, every row and column credited. -/
def fullDecl (u : ℕ) : PearlC.Decl :=
  ⟨(σ.U.layout.shape u).k, range (σ.U.layout.shape u).m, range (σ.U.layout.shape u).n⟩

/-- A tile's cost: its weight at its unit's full declaration, times `D·(γd − γn)`, rounded up. -/
noncomputable def cost (g : σ.Tile) : ℕ :=
  ⌈((σ.D * (σ.γd - σ.γn) : ℕ) : ℝ) * (σ.Wd σ.U.layout g (σ.fullDecl ((σ.TR.tiling σ.U.layout).unit g)) : ℝ)⌉₊

/-- The charges: none. -/
def charges : ℕ := 0

/-- The audit that accepts only checked tiles accepts. -/
def accepts (r : σ.Run) : Prop := AuditAccepts (σ.passed r) r.2

/-- `over`: the claimed work times `D·(γd − γn)`, rounded down. -/
noncomputable def «over» (r : σ.Run) : ℕ :=
  ⌊((σ.D * (σ.γd - σ.γn) : ℕ) : ℝ) * PearlC.claimedDraw σ.TR σ.Wd σ.U (σ.trace r) (σ.dec r.1.1 r.1.2)⌋₊

/-- `under`: the budget times `D·γd`, rounded up. -/
noncomputable def under (_ : σ.Run) : ℕ := ⌈((σ.D * σ.γd : ℕ) : ℝ) * σ.T⌉₊

/-- The game's order is its quantifiers, so every run meets it. -/
def logOK (_ : σ.Run) : Prop := True

open Classical in
/-- The incorrect tiles: those a sampled check doesn't pass. -/
noncomputable def incorrect (r : σ.Run) : Finset σ.Tile := univ.filter fun g => (g : ℕ) ∉ σ.passed r

/-- TT_OUT per tile's event at the budget: the good tiles' credit exceeds `T/(1 − g₀)`. -/
def breaks (r : σ.Run) : Prop :=
  σ.T / (1 - σ.g₀) < ∑ g ∈ PearlC.goodTiles σ.P σ.TR σ.U r.1.1 r.1.2 (σ.trace r),
    σ.TR.credit r.1.1 r.1.2 σ.U g (PearlC.actOf (σ.trace r) ((σ.TR.tiling σ.U.layout).unit g))

/-- The scheme's own term: `η(q, N)`. -/
noncomputable def δscheme : ℝ≥0∞ := ENNReal.ofReal (σ.η σ.q σ.U.layout.N)

/-- **Pearl+'s charge inequality**, an open premise: in an accepted run that doesn't break TT_OUT per tile's event,
`(1 − γ)` of the claimed work is within the budget plus the incorrect tiles' cost. It rests on TT_OUT per tile, through
`breaks`, and on each good tile's weight being within `1/(1 − γ)` of `(1 − g₀)` times its credit (`KernelWithin` at
`γ = 1 − (1 − g₀)/ω`); it is proved nowhere. -/
def ChargeHolds : Prop :=
  ∀ r : σ.Run, σ.accepts r → σ.logOK r → ¬ σ.breaks r →
    σ.over r ≤ σ.under r + ∑ b ∈ σ.incorrect r, σ.cost b + charges

/-- **Pearl+ as a scheme**, given its charge inequality. -/
noncomputable def scheme (h : σ.ChargeHolds) : Verity.Scheme where
  Block := σ.Tile
  cost := σ.cost
  charges := Setting.charges
  Run := σ.Run
  accepts := σ.accepts
  «over» := σ.over
  under := σ.under
  logOK := σ.logOK
  incorrect := σ.incorrect
  breaks := σ.breaks
  δscheme := σ.δscheme
  charge := h

end Setting

end Pouw.PearlPlus
