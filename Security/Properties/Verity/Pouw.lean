import Properties.Verity.Game
import Mathlib.Analysis.SpecialFunctions.Log.Basic

/-!
# PoUW's slot in the causal game: its component bound from PoUW's named premises

`Properties.Verity.Game` (#4613) leaves each resource subprotocol a target, `Vocab.PouwSound := V.Target .pouw`. lean's
ONE-18 (LEAN-62) proves `V.PouwSound` from the component-level bound `V.ComponentBound .pouw` by monotonicity
(`pouwSound_of`), so that bound is what PoUW owes. `Pouw.ComponentBound V` is lean's `V.ComponentBound .pouw`
(`Properties.Verity.Subprotocols` on ONE-18's branch), unfolded here until lean's module lands: the probability that
PoUW's component accepts its own case while PoUW's claim fails is at most PoUW's error at the strategy (`Vocab.error`).
`PouwComponentBound` is PoUW's statement: on every device and every vocabulary `V`, PoUW's named premises give it.

PoUW's claim is in gates (`Holds .pouw`: at least `W` gates of the committed circuit the play computed after their
window's coin, `Vocab.computed`). PoUW's credit floor is `(1 − ε)₊` of the registered circuit's live gates, with
`ε = Λ·ln(1/δ_s)/n` over the period's `n` windows (`creditGates`): `H1J10RCheckedCreditPeriod`'s (#4541) conclusion,
counted in gates.

The event splits as ONE-18's coupling splits it (`Vocab.Coupling.verdict`, `BreakBound`). A release PoUW's check read
that is missing from the device's log is a forgery of its LMS signature, and PoUW's break (`Reading.brk`, two openings of
one position) a SHA-512 collision; both are PoUW's finder's (`FinderAnswers`). The rest is the verdict at the code
(`Reading.AtCode`), where the service's soundness and the period model's credit bound apply.

The premises:

* `SaltIsLiveCoin`, `RegisteredBeforeDraw`, `TileDrawIsCoin`: what PoUW's compiled check reads from the device's log,
  stated on what the prover sent (`Sent`) and the releases the check read (`coinsRead .pouw`).
* `FinderAnswers`: where PoUW's finder meets `FinderCR`, an unlogged release or a break is within PoUW's collision term.
* `ServiceSound`: the proof service's soundness at the drawn tiles (R4′): a drawn tile unbound in the fixed extractor's
  transcript under the commitment's openings.
* `CreditedPeriodRefines`: the period model's credit bound (`H1J10RCheckedCreditPeriod`) carried into this game along
  the refinement of PoUW's compiled acceptance into the model's (`Properties.Pouw.PeriodJ10R.Check.acceptedWindows`,
  #4857). The lucks (`SharedNoiseLucks` at `p = 7/8`, `p0 = 1/2`), `Scope`, `DeadConeDead`, `UnitFixed` and
  `MovedIn ≤ δ_bind` are that guarantee's premises, stated over the model's play. `Properties.Pouw.PeriodJ10R` isn't in
  #4613's tree, so they enter here through this one premise.
* `PublishedCredit`, `PublishedError`: what the product publishes (ONE-19) at PoUW's floor and error.
-/

namespace Verity

open Accounting Core
open scoped ENNReal

namespace Pouw

/-- **PoUW's component bound** (lean's `V.ComponentBound .pouw`, ONE-18): at all `θ`, `F` and `c`, and for every
strategy, the probability that PoUW's component accepts its own case in what the prover sent while PoUW's claim fails
is at most PoUW's error at that strategy. -/
def ComponentBound {ι α : Type} [Fintype α] {d : Device ι} (V : Vocab d α) : Prop :=
  ∀ (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)),
    Game.prob (fun o => ComponentAccepts .pouw o.sent ∧ ¬ V.Holds θ F c σ .pouw o) (V.game θ F c) σ ≤
      V.error .pouw θ F c σ

/-- **What PoUW's compiled check reads, and what it credits.** The windows a period's parameters register; for each
window the release whose coins are its salt and the release whose coins draw its tiles, as `PouwCheck.Period.protocol`
decodes them from what the prover sent; each window's registration digest; the tiles published as drawn and the
service's subset draw (`SampledProofs.Select.select`) at the file's population; the period's live gates in the
registered file's circuit; and PoUW's break in a play (two openings of one position), which its finder answers for. -/
structure Reading {ι α : Type} [Fintype α] {d : Device ι} (V : Vocab d α) where
  windows : ByteArray → ℕ
  salt : Sent → ℕ → Option Release
  draw : Sent → ℕ → Option Release
  registration : Sent → ℕ → ByteArray
  drawn : (F : ByteArray) → Sent → ℕ → Finset (Fin (V.units F))
  select : (F : ByteArray) → List Bool → Finset (Fin (V.units F))
  live : (θ : ByteArray) → Finset (Fin (V.circuitOf (V.registered θ)).N)
  brk : (θ F c : ByteArray) → Game.Strategy (V.game θ F c) → Outcome → Prop

/-- **PoUW's credit floor in gates**: `(1 − ε)₊` of `live` gates, `ε = Λ·ln(1/δ_s)/n` over `n` windows. -/
noncomputable def creditGates (Λ δs : ℝ) (n live : ℕ) : ℝ :=
  max 0 (1 - Λ * Real.log (1 / δs) / n) * live

variable {ι α : Type} [Fintype α] {d : Device ι} {V : Vocab d α} (R : Reading V)

/-- The size of the period's live gates in the registered file's circuit. -/
noncomputable def Reading.liveSize (θ : ByteArray) : ℕ := (V.circuitOf (V.registered θ)).size (R.live θ)

/-- **The verdict at the code** (ONE-18's `Coupling.verdict`): PoUW's component accepts its own case, every release its
check read is in the device's log, and there is no break. -/
def Reading.AtCode (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)) (o : Outcome) : Prop :=
  ComponentAccepts .pouw o.sent ∧ V.Logged .pouw o ∧ ¬ R.brk θ F c σ o

/-- **Each window's salt is a live coin**: when PoUW's component accepts, each window's salt is the coins of a release
its check read. With `Logged`, the salt is a coin the device drew after the bytes it answers. Deployment gap until
CMP-90 (ALLP-SALT): the loop is given its salt before it serves. -/
def SaltIsLiveCoin : Prop :=
  ∀ s, ComponentAccepts .pouw s → ∀ w < R.windows s.params,
    ∃ r, R.salt s w = some r ∧ r ∈ V.coinsRead .pouw s

/-- **Each registration is fixed before its window's draw**: when PoUW's component accepts, each window's draw is a
release its check read that answers the window's registration digest, so the device released the draw's coins after
the registration (`releases`). Deployment gap until CMP-91 (ALLP-WIRE): the registration commits no row. -/
def RegisteredBeforeDraw : Prop :=
  ∀ s, ComponentAccepts .pouw s → ∀ w < R.windows s.params,
    ∃ r, R.draw s w = some r ∧ r ∈ V.coinsRead .pouw s ∧ r.answers = R.registration s w

/-- **The published draw is the draw's coins**: each window's drawn tiles are the service's subset draw at its draw
release's coins. A code fact once the verifier draws the tiles inside the stratified draw (CMP-126). -/
def TileDrawIsCoin : Prop :=
  ∀ F s w r, R.draw s w = some r → R.drawn F s w = R.select F r.coins

/-- **PoUW's finder answers for the rest of the event** (ONE-18's `Coupling.BreakBound`): where it meets `FinderCR`, the
probability that PoUW's component accepts while a release its check read is missing from the log (a forged LMS
signature) or there is a break (two openings of one position) is at most PoUW's collision term at the strategy. -/
def FinderAnswers : Prop :=
  ∀ (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)), V.FinderCR .pouw θ F c σ →
    Game.prob (fun o => ComponentAccepts .pouw o.sent ∧ (¬ V.Logged .pouw o ∨ R.brk θ F c σ o)) (V.game θ F c) σ ≤
      V.loss .pouw θ ((V.live θ F c).collides sha512Msg .pouw σ)

/-- **The service's soundness at the drawn tiles** (R4′): at the verdict at the code, some drawn tile of some window is
unbound in the fixed extractor's transcript under the commitment's openings with probability at most `n·δ_ks`. C-Flock's
soundness at each drawn unit under live coins with one-stage's acceptance (`TileServiceSound` in #4541). -/
def ServiceSound (δks : ℝ≥0∞) : Prop :=
  ∀ (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)),
    Game.prob (fun o => R.AtCode θ F c σ o ∧ ∃ w < R.windows o.params, ∃ u ∈ R.drawn F o.sent w,
        u ∈ unbound (V.partsOf F) (V.Opens θ F c) (V.extract θ F c σ)) (V.game θ F c) σ ≤
      R.windows θ * δks

/-- **The period model's credit bound, in this game** (`H1J10RCheckedCreditPeriod`, #4541, along the refinement of
PoUW's compiled acceptance, #4857): at the verdict at the code, with each window's salt and draw releases its check
read (the draw answering the registration) and its drawn tiles the draw's coins and bound in the extractor's
transcript, the play computed fewer gates of the committed circuit after their window's coin than the credit floor of
the live gates with probability at most `δ_s + n·δ_bind`. Discharged by that guarantee at the model run, under its
premises: the J10-R lucks (`SharedNoiseLucks`: `OperandHMu`, `ReceiptLuckAt`, `GuessLuckAt` at `p = 7/8`,
`p0 = 1/2`), `Scope`, `DeadConeDead`, `UnitFixed` and `MovedIn ≤ δ_bind`. -/
def CreditedPeriodRefines (Λ δs : ℝ) (δbind : ℝ≥0∞) : Prop :=
  ∀ (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)),
    Game.prob (fun o => R.AtCode θ F c σ o ∧
        (∀ w < R.windows o.params, (∃ r, R.salt o.sent w = some r ∧ r ∈ V.coinsRead .pouw o.sent) ∧
          ∃ r, R.draw o.sent w = some r ∧ r ∈ V.coinsRead .pouw o.sent ∧ r.answers = R.registration o.sent w ∧
            R.drawn F o.sent w = R.select F r.coins ∧
            ∀ u ∈ R.drawn F o.sent w, u ∉ unbound (V.partsOf F) (V.Opens θ F c) (V.extract θ F c σ)) ∧
        ((V.circuitOf F).size (V.computed θ F c σ o) : ℝ) < creditGates Λ δs (R.windows θ) (R.liveSize θ))
      (V.game θ F c) σ ≤
      ENNReal.ofReal δs + R.windows θ * δbind

/-- **PoUW's published certificate is its credit floor** (ONE-19): `W` is at most the credit floor of the registered
file's live gates. -/
def PublishedCredit (Λ δs : ℝ) : Prop :=
  ∀ θ, (V.certOf θ).W ≤ creditGates Λ δs (R.windows θ) (R.liveSize θ)

/-- **PoUW's published error covers its bound** (ONE-19): at every count of SHA-512 calls, `δ` is at least
`δ_s + n·(δ_bind + δ_ks)`. -/
def PublishedError (δs : ℝ) (δbind δks : ℝ≥0∞) : Prop :=
  ∀ θ q, ENNReal.ofReal δs + R.windows θ * (δbind + δks) ≤ V.delta θ .pouw q

end Pouw

/-- **PoUW's component bound** (ONE-16, CMP-145): on any device, for any names `V` the subprotocols define and any
reading `R` of PoUW's compiled check, PoUW's premises give `Pouw.ComponentBound V` (lean's `V.ComponentBound .pouw`):
where PoUW's finder meets `FinderCR`, PoUW's component accepts while the play computed fewer than `W` gates after their
window's coin with probability at most `δ_s + n·(δ_bind + δ_ks)` plus PoUW's collision term. -/
def PouwComponentBound : Prop :=
  ∀ (ι α : Type) [Fintype α] (d : Device ι) (V : Vocab d α) (R : Pouw.Reading V) (Λ δs : ℝ) (δbind δks : ℝ≥0∞),
    Pouw.SaltIsLiveCoin R → Pouw.RegisteredBeforeDraw R → Pouw.TileDrawIsCoin R → Pouw.FinderAnswers R →
      Pouw.ServiceSound R δks → Pouw.CreditedPeriodRefines R Λ δs δbind → Pouw.PublishedCredit R Λ δs →
        Pouw.PublishedError R δs δbind δks → Pouw.ComponentBound V

end Verity
