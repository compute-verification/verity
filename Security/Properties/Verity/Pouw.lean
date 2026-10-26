import Properties.Verity.Game
import Mathlib.Analysis.SpecialFunctions.Log.Basic

/-!
# PoUW's slot in the causal game: `ComponentBound .pouw` from PoUW's named premises

`Properties.Verity.Game` (#4613) leaves each resource subprotocol a target, `Vocab.PouwSound := V.Target .pouw`. lean's
ONE-18 (LEAN-62) proves `V.PouwSound` from the component-level bound `V.ComponentBound .pouw` by monotonicity
(`pouwSound_of`), so `ComponentBound .pouw` is what PoUW owes. `ComponentBound` is lean's text on LEAN-62, restated here
until lean's module lands. `PouwComponentBound` is PoUW's statement of it: on every device and every vocabulary `V`,
PoUW's named premises give `V.ComponentBound .pouw`.

PoUW's claim is in circuit size. PoUW credits gates of the registered file's circuit: the period's live gates
(`Reading.live`, the gates its windows' coins feed, the tiles' words) and of them the gates the play computed only
after their window's coin (`Reading.fixed`). The credit floor is `(1 − ε)₊` of the live gates' size, where
`ε = Λ·ln(1/δ_s)/n` over the period's `n` windows (`creditGates`). That is `H1J10RCheckedCreditPeriod`'s (#4541)
conclusion, counted in gates.

`Holds .pouw` in #4613 is `certOf.W ≤ Schedule.work … (priceOf θ)`, the price-weighted ops of the compute account on
the device schedule, not a gate count. PoUW reaches it only through `WorkOfFixedGates`, a premise of the machine model
(ACCEL's `θ`), which is the one place price appears.

The premises:

* `SaltIsLiveCoin`, `RegisteredBeforeDraw`, `TileDrawIsCoin`: what PoUW's compiled check reads from the device's log,
  stated on what the prover sent (`Sent`) and the releases the check read (`coinsRead .pouw`).
* `ServiceSound`: the proof service's soundness at the drawn tiles, read off the commitment (R4): a drawn tile unbound
  in a transcript the commitment opens to.
* `CreditedPeriodRefines`: the period model's credit bound (`H1J10RCheckedCreditPeriod`) carried into this game along
  the refinement of PoUW's compiled acceptance into the model's (lean's `PouwRefines`). The lucks (`SharedNoiseLucks` at
  `p = 7/8`, `p0 = 1/2`), `Scope`, `DeadConeDead`, `UnitFixed` and `MovedIn ≤ δ_bind` are that guarantee's premises,
  stated over the model's play. Neither `Properties.Pouw.Period` (main) nor `Properties.Pouw.PeriodJ10R` (#4541) is in
  #4613's tree, so they enter here through this one premise.
* `WorkOfFixedGates`, `PublishedCredit`, `PublishedError`: the machine model's tie of a computed gate to priced work,
  and what the product publishes (ONE-19) at PoUW's floor and error.
-/

namespace Verity

open Accounting Core
open scoped ENNReal

namespace Vocab

variable {ι α : Type} [Fintype α] {d : Device ι} (V : Vocab d α)

/-- **The component-level bound** (lean, LEAN-62, 26 Oct 20:36Z): on the causal game, subprotocol `a`'s component
accepts, every release it read is in the device's log, its claim fails and there is no break, with probability at most
its published error. -/
def ComponentBound (a : Audit) : Prop :=
  ∀ (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)),
    Game.prob (fun o => ComponentAccepts a o.sent ∧ (∀ r ∈ V.coinsRead a o.sent, r ∈ o.log) ∧
        ¬ V.Holds θ F c σ a o ∧ ¬ V.Break θ F c o) (V.game θ F c) σ ≤ V.delta θ a

end Vocab

namespace Pouw

/-- **What PoUW's compiled check reads, and what it credits.** The windows a period's parameters register; for each
window the release whose coins are its salt and the release whose coins draw its tiles, as `PouwCheck.protocol`
decodes them from what the prover sent; each window's registration digest; the tiles published as drawn and the
service's subset draw (`SampledProofs.Select.select`) at the file's population; the period's live gates in the
registered file's circuit; and the number of them the play computed only after their window's coin (the model's
`UnitFixed` read on the play). -/
structure Reading {ι α : Type} [Fintype α] {d : Device ι} (V : Vocab d α) where
  windows : ByteArray → ℕ
  salt : Sent → ℕ → Option Release
  draw : Sent → ℕ → Option Release
  registration : Sent → ℕ → ByteArray
  drawn : (F : ByteArray) → Sent → ℕ → Finset (Fin (V.units F))
  select : (F : ByteArray) → List Bool → Finset (Fin (V.units F))
  live : (θ : ByteArray) → Finset (Fin (V.circuitOf (V.registered θ)).N)
  fixed : (θ F c : ByteArray) → Game.Strategy (V.game θ F c) → Outcome → ℕ

/-- **PoUW's credit floor in gates**: `(1 − ε)₊` of `live` gates, `ε = Λ·ln(1/δ_s)/n` over `n` windows. -/
noncomputable def creditGates (Λ δs : ℝ) (n live : ℕ) : ℝ :=
  max 0 (1 - Λ * Real.log (1 / δs) / n) * live

variable {ι α : Type} [Fintype α] {d : Device ι} {V : Vocab d α} (R : Reading V)

/-- The size of the period's live gates in the registered file's circuit. -/
def Reading.liveSize (θ : ByteArray) : ℕ := (V.circuitOf (V.registered θ)).size (R.live θ)

/-- **Each window's salt is a live coin**: when PoUW's component accepts, each window's salt is the coins of a release
its check read. With `Verdict`'s logged releases, the salt is a coin the device drew after the bytes it answers.
Deployment gap until CMP-90 (ALLP-SALT): the loop is given its salt before it serves. -/
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

/-- **The service's soundness at the drawn tiles** (R4): PoUW's component accepts, there is no break, and some drawn
tile of some window is unbound in a transcript the commitment opens to, with probability at most `n·δ_ks`. C-Flock's
soundness at each drawn unit under live coins with one-stage's acceptance (`TileServiceSound` in #4541). -/
def ServiceSound (δks : ℝ≥0∞) : Prop :=
  ∀ (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)),
    Game.prob (fun o => ComponentAccepts .pouw o.sent ∧ ¬ V.Break θ F c o ∧
        ∃ w < R.windows θ, ∃ u ∈ R.drawn F o.sent w, ∃ X : (V.circuitOf F).Transcript,
          (∀ g v, V.Opens θ F c g v → X g = v) ∧ u ∈ V.unbound θ F c X) (V.game θ F c) σ ≤
      R.windows θ * δks

/-- **The period model's credit bound, in this game** (`H1J10RCheckedCreditPeriod`, #4541, along lean's
`PouwRefines`): PoUW's component accepts, every release it read is logged, each window's salt and draw are such
releases (the draw answering the registration), the drawn tiles are the draw's coins and bound, there is no break, and
the play computed fewer gates after their window's coin than the credit floor of the live gates, with probability at
most `δ_s + n·δ_bind`. Discharged by that guarantee at the model run, under its premises: the J10-R lucks
(`SharedNoiseLucks`: `OperandHMu`, `ReceiptLuckAt`, `GuessLuckAt` at `p = 7/8`, `p0 = 1/2`), `Scope`, `DeadConeDead`,
`UnitFixed` and `MovedIn ≤ δ_bind`. -/
def CreditedPeriodRefines (Λ δs : ℝ) (δbind : ℝ≥0∞) : Prop :=
  ∀ (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)),
    Game.prob (fun o => ComponentAccepts .pouw o.sent ∧ (∀ r ∈ V.coinsRead .pouw o.sent, r ∈ o.log) ∧
        (∀ w < R.windows θ, (∃ r, R.salt o.sent w = some r ∧ r ∈ V.coinsRead .pouw o.sent) ∧
          ∃ r, R.draw o.sent w = some r ∧ r ∈ V.coinsRead .pouw o.sent ∧ r.answers = R.registration o.sent w ∧
            R.drawn F o.sent w = R.select F r.coins ∧
            ∀ u ∈ R.drawn F o.sent w, ∀ X : (V.circuitOf F).Transcript,
              (∀ g v, V.Opens θ F c g v → X g = v) → u ∉ V.unbound θ F c X) ∧
        ¬ V.Break θ F c o ∧
        (R.fixed θ F c σ o : ℝ) < creditGates Λ δs (R.windows θ) (R.liveSize θ)) (V.game θ F c) σ ≤
      ENNReal.ofReal δs + R.windows θ * δbind

/-- **A computed gate is priced work** (the machine model, ACCEL's `θ`): the gates the play computed after their
window's coin are at most the compute account's priced work over the world's window. -/
def WorkOfFixedGates : Prop :=
  ∀ (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)) (o : Outcome),
    (R.fixed θ F c σ o : ℝ) ≤
      ((V.world θ F c σ o).schedule.work (V.world θ F c σ o).win ((V.world θ F c σ o).acct .compute)
        (V.priceOf θ) : ℝ)

/-- **PoUW's published certificate is its credit floor** (ONE-19): `W` is at most the credit floor of the registered
file's live gates. -/
def PublishedCredit (Λ δs : ℝ) : Prop :=
  ∀ θ, (V.certOf θ).W ≤ creditGates Λ δs (R.windows θ) (R.liveSize θ)

/-- **PoUW's published error** (ONE-19): `δ_s + n·(δ_bind + δ_ks)`. -/
def PublishedError (δs : ℝ) (δbind δks : ℝ≥0∞) : Prop :=
  ∀ θ, V.delta θ .pouw = ENNReal.ofReal δs + R.windows θ * (δbind + δks)

end Pouw

/-- **PoUW's component bound** (ONE-16, CMP-145): on any device, for any names `V` the subprotocols define and any
reading `R` of PoUW's compiled check, PoUW's premises give `V.ComponentBound .pouw`: except with probability its
published error `δ_s + n·(δ_bind + δ_ks)`, when PoUW's component accepts with every release it read logged and no break,
the compute account's priced work is at least the certificate, because the play computed at least `(1 − ε)₊` of the
period's live gates after their window's coin. -/
def PouwComponentBound : Prop :=
  ∀ (ι α : Type) [Fintype α] (d : Device ι) (V : Vocab d α) (R : Pouw.Reading V) (Λ δs : ℝ) (δbind δks : ℝ≥0∞),
    Pouw.SaltIsLiveCoin R → Pouw.RegisteredBeforeDraw R → Pouw.TileDrawIsCoin R → Pouw.ServiceSound R δks →
      Pouw.CreditedPeriodRefines R Λ δs δbind → Pouw.WorkOfFixedGates R → Pouw.PublishedCredit R Λ δs →
        Pouw.PublishedError R δs δbind δks → V.ComponentBound .pouw

end Verity
