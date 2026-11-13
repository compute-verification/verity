module

public import Proofs.Pouw.PearlC.S8H100Rev1Gap
public import Proofs.Pouw.Prob

/-!
# A reviewer's probes of the split of TT_OUT on the H100 (not for landing)

These lemmas belong to a review of `Definitions.Assumptions.PearlC`. They state no guarantee and no assumption, and no
lock lists them.

* `w1Priced_commits`: a W1 program commits every declared unit, so no W1 run leaves a unit blank.
* `w1Priced_run_oracle`: a W1 run's transcript is the same on every oracle.
* `blankCM`: the cost model with one program, which commits nothing, at no cost, with no oracle call.
  `blankCM_ttOutTilePearlCH100Rev1` is the abstract form of TT_OUT per tile at it, and `blankCM_not_w1Complete` says the
  device model `W1CompleteH100` fails at it. So the device model and the conjecture over W1 together say strictly more
  than the abstract form at one cost model: a cost model with a program that leaves a declared unit blank meets no
  `W1CompleteH100`.
-/

@[expose] public section

namespace Pouw.PearlC.RatingProbe

open Pouw.PearlC Pouw.PearlC.W1 Pouw.PearlC.Assumptions

variable {Q R S : Type}

/-- A W1 program commits every declared unit. -/
theorem w1Priced_commits (dv : W1Device) (salt : S → List ℕ) (p : CodeImage) (H : Q → R) (U : Workload)
    (act : ℕ → Codes) (σ : State) (s : S) {u : ℕ} (hu : u < U.layout.N) :
    (w1Priced dv salt).run p H U act σ s u ≠ none := by
  simp [w1Priced, readout, hu]

/-- A W1 run's transcript does not read the oracle. -/
theorem w1Priced_run_oracle (dv : W1Device) (salt : S → List ℕ) (p : CodeImage) (H H' : Q → R) (U : Workload)
    (act : ℕ → Codes) (σ : State) (s : S) :
    (w1Priced dv salt).run p H U act σ s = (w1Priced dv salt).run p H' U act σ s := rfl

/-- The cost model with one program, which commits nothing, at no cost, with no oracle call. -/
def blankCM : CostModel Q R S where
  Prog := Unit
  run := fun _ _ _ _ _ _ _ => none
  cost := fun _ _ _ _ _ _ => 0
  calls := fun _ _ _ _ _ _ => 0

theorem blankCM_queries : (blankCM : CostModel Q R S).Queries :=
  fun _ _ _ _ _ => ⟨.done fun _ => none, fun _ => ⟨rfl, rfl⟩⟩

/-- TT_OUT per tile holds at `blankCM`, for any protocol, tiles and domain: no tile of a blank transcript is good. -/
theorem blankCM_ttOutTile [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (P : Protocol Q R S)
    (TR : TileRules Q R S) (D : Layout → Prop) {g₀ : ℝ} (hg : g₀ < 1) {δ : ℕ → ℕ → ℝ} (hδ : ∀ q N, 0 ≤ δ q N) :
    TTOutTile (blankCM : CostModel Q R S) P TR D g₀ δ := by
  refine ⟨blankCM_queries, fun U _ T hT q _ A _ => ?_⟩
  have h0 : pr (fun x : (Q → R) × S =>
      T / (1 - g₀) < ∑ g ∈ goodTiles P TR U x.1 x.2 (A.transcript U x.1 x.2),
        TR.credit x.1 x.2 U g (actOf (A.transcript U x.1 x.2) ((TR.tiling U.layout).unit g))) = 0 := by
    apply pr_eq_zero
    intro x hx
    have hempty : goodTiles P TR U x.1 x.2 (A.transcript U x.1 x.2) = ∅ := by
      unfold goodTiles
      refine Finset.filter_eq_empty_iff.2 fun g _ hg' => ?_
      obtain ⟨c, hc, -⟩ := hg'
      exact Option.noConfusion hc
    rw [hempty, Finset.sum_empty] at hx
    exact absurd hx (not_lt.2 (div_nonneg hT (by linarith)))
  rw [h0]
  exact hδ _ _

/-- The abstract form of TT_OUT per tile, rev1, at the H100 record holds at `blankCM`. -/
theorem blankCM_ttOutTilePearlCH100Rev1 [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sem : PearlCSem Q R S)
    (ρ : ℚ) : TTOutTilePearlCH100Rev1 (blankCM : CostModel Q R S) sem ρ :=
  blankCM_ttOutTile _ _ _ (by norm_num) fun q N => by unfold δPearlC; positivity

/-- The device model fails at `blankCM`: W1 has no program that leaves a declared unit blank. -/
theorem blankCM_not_w1Complete [Nonempty (Q → R)] [Nonempty S] (salt : S → List ℕ) :
    ¬ W1CompleteH100 salt (blankCM : CostModel Q R S) := by
  rintro ⟨-, h⟩
  obtain ⟨p', hp'⟩ := h ()
  obtain ⟨H⟩ := ‹Nonempty (Q → R)›
  obtain ⟨s⟩ := ‹Nonempty S›
  let U : Workload := ⟨⟨1, fun _ => ⟨1, 64, 128⟩, fun _ => 0⟩, fun _ _ _ => 0⟩
  have h1 := congrFun (hp' H U (fun _ _ _ => 0) [] s).1 0
  exact w1Priced_commits W1Device.s8H100Best salt p' H U (fun _ _ _ => 0) [] s (u := 0) Nat.zero_lt_one h1

end Pouw.PearlC.RatingProbe
