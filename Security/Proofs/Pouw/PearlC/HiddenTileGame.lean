import Definitions.Pouw.PearlC.HiddenTileGame
import Proofs.Pouw.PearlC.HiddenGamma
import Proofs.Core.Game

/-!
# The hidden audit's open obligation from what a proof system gives per tile

`tileProofSoundAll_of_tileGame`: a hidden audit whose class is `BoundedByInduced Cls`, with a `TileGame` for every tile
of every in-domain workload, meets `TileProofSoundAll` at `ηzk`. Per draw `i`, a tile that is neither good nor blank has
its verdict accept only on a run in `Bad` (`TileGame.prob_le`); and the batch prover, read at position `i`, does no
better than some strategy it induces on that tile's game, which is in `Cls` (core's `ProbBatchInduced`).
-/

namespace Pouw.PearlC

open scoped ENNReal
open Pouw.PearlC.Assumptions

variable {Q R S : Type}

theorem prob_cast_game {α : Type} (p : α → Prop) {g₁ g₂ : Verity.Game α} (e : g₁ = g₂)
    (s : Verity.Game.Strategy g₁) :
    Verity.Game.prob p g₂ (cast (congrArg Verity.Game.Strategy e) s) = Verity.Game.prob p g₁ s := by
  subst e
  rfl

/-- One position of a batch prover in the class `BoundedByInduced Cls`: a bound for every strategy in `Cls` holds
there. -/
theorem prob_getD_le_of_boundedByInduced {O : Type}
    (Cls : CommitLog → O → (g : Verity.Game Bool) → Verity.Game.Strategy g → ℕ → Prop) (τ : CommitLog) (o : O)
    (G : Verity.Game (List Bool)) (σ : Verity.Game.Strategy G) (q : ℕ) (hσ : BoundedByInduced Cls τ o G σ q)
    (pre : List (Verity.Game Bool)) (g : Verity.Game Bool) (post : List (Verity.Game Bool))
    (e : Verity.Game.batch (pre ++ g :: post) = G) (n : ℕ) (hn : pre.length = n) (p : Bool → Prop) (ε : ℝ≥0∞)
    (hg : ∀ s, Cls τ o g s q → Verity.Game.prob p g s ≤ ε) :
    Verity.Game.prob (fun v => p (v.getD n false)) G σ ≤ ε := by
  subst hn
  subst e
  exact Verity.SecurityProofs.ProbBatchInduced pre post g false p (fun s => Cls τ o g s q) ε hg σ
    (fun s hs => hσ pre g post rfl s hs)

/-- On a tile neither good nor blank, the verdict is accept with probability at most `ηzk q` for every strategy in
`Cls` at `q`. -/
theorem TileGame.prob_le [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] {P : Protocol Q R S}
    {TR : TileRules Q R S} {HA : HiddenAudit Q R S}
    {Cls : CommitLog → HA.Opening → (g : Verity.Game Bool) → Verity.Game.Strategy g → ℕ → Prop} {ηzk : ℕ → ℝ}
    {U : Workload} {τ : CommitLog} {o : HA.Opening} {H : Q → R} {s : S} {g : ℕ}
    (T : TileGame P TR HA Cls ηzk U τ o H s g) (h0 : ∀ q, 0 ≤ ηzk q)
    (hbad : g ∉ goodTiles P TR U H s τ ∪ blankTiles TR U τ) (q : ℕ)
    (s' : Verity.Game.Strategy (HA.game H s U (HA.root U τ o) g)) (hs : Cls τ o _ s' q) :
    Verity.Game.prob (fun b => b = true) (HA.game H s U (HA.root U τ o) g) s' ≤ ENNReal.ofReal (ηzk q) := by
  let σ' := Verity.Game.Strategy.ofMap T.verdict T.G (cast (congrArg Verity.Game.Strategy T.game_eq) s')
  have h1 : Verity.Game.prob (fun b => b = true) (HA.game H s U (HA.root U τ o) g) s' =
      Verity.Game.prob (fun x => T.verdict x = true) T.G σ' := by
    rw [← prob_cast_game _ T.game_eq s', Verity.Game.prob_map]
  have hcls : Cls τ o _ (cast (congrArg Verity.Game.Strategy T.game_eq.symm)
      (Verity.Game.Strategy.toMap T.verdict T.G σ')) q := by
    simp only [σ', Verity.Game.Strategy.toMap_ofMap, cast_cast, cast_eq]
    exact hs
  have h3 : Verity.Game.prob (fun x => T.verdict x = true) T.G σ' ≤
      Verity.Game.prob (fun x => T.verdict x = true ∧ T.Bad σ' x) T.G σ' :=
    Verity.Game.prob_mono
      (fun x hx => ⟨hx, Classical.byContradiction fun hb => hbad (T.good_of_accept σ' x hx hb)⟩) _ _
  rw [h1]
  refine h3.trans ?_
  rw [ENNReal.le_ofReal_iff_toReal_le (ne_top_of_le_ne_top ENNReal.one_ne_top (Verity.Game.prob_le_one _ _ _)) (h0 q)]
  exact T.bad_le q σ' hcls

theorem ex_const_le {α : Type} [Fintype α] {c : ℝ} (hc : 0 ≤ c) : ex (fun _ : α => c) ≤ c := by
  unfold ex
  rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
  rcases Nat.eq_zero_or_pos (Fintype.card α) with h | h
  · simp [h, hc]
  · rw [mul_div_cancel_left₀ c (ne_of_gt (by exact_mod_cast h))]

/-- **The hidden audit's open obligation from what a proof system gives per tile.** A hidden audit whose class at `q`
is within `BoundedByInduced Cls`, with a `TileGame` for every tile of every in-domain workload, transcript and opening,
meets `TileProofSoundAll` at `ηzk`, given `0 ≤ ηzk`. -/
theorem tileProofSoundAll_of_tileGame [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (P : Protocol Q R S)
    (TR : TileRules Q R S) (D : UnitLayout → Prop) (HA : HiddenAudit Q R S)
    (Cls : CommitLog → HA.Opening → (g : Verity.Game Bool) → Verity.Game.Strategy g → ℕ → Prop) (ηzk : ℕ → ℝ)
    (h0 : ∀ q, 0 ≤ ηzk q) (hB : ∀ τ o G σ q, HA.Bounded τ o G σ q → BoundedByInduced Cls τ o G σ q)
    (tg : ∀ U : Workload, U.InDomain P D → ∀ (τ : CommitLog) (o : HA.Opening) (H : Q → R) (s : S) (g : ℕ),
      TileGame P TR HA Cls ηzk U τ o H s g) :
    TileProofSoundAll P TR D HA ηzk := by
  intro U hU τ op w t q σ hσ i
  refine (ex_mono fun x => ?_).trans (ex_const_le (h0 q))
  split_ifs with hgood
  · exact h0 q
  · let f : Fin t → Verity.Game Bool := fun j =>
      HA.game x.1 x.2.1 U (HA.root U (τ x.1 x.2.1) (op x.1 x.2.1)) (x.2.2 j).1
    have hi : (i : ℕ) < (List.ofFn f).length := by simp
    have hl : (List.ofFn f).drop i = f i :: (List.ofFn f).drop (i + 1) := by
      rw [List.drop_eq_getElem_cons hi, List.getElem_ofFn]
    have e : Verity.Game.batch ((List.ofFn f).take i ++ f i :: (List.ofFn f).drop (i + 1)) =
        HA.batch x.1 x.2.1 U (HA.root U (τ x.1 x.2.1) (op x.1 x.2.1)) x.2.2 := by
      show _ = Verity.Game.batch (List.ofFn f)
      rw [← hl, List.take_append_drop]
    have hle := prob_getD_le_of_boundedByInduced Cls _ _ _ (σ x) q (hB _ _ _ _ q (hσ x)) _ (f i) _ e i
      (by simp) (fun b => b = true) (ENNReal.ofReal (ηzk q))
      (fun s hs => (tg U hU (τ x.1 x.2.1) (op x.1 x.2.1) x.1 x.2.1 (x.2.2 i).1).prob_le h0 hgood q s hs)
    calc _ ≤ (ENNReal.ofReal (ηzk q)).toReal := ENNReal.toReal_mono ENNReal.ofReal_ne_top hle
      _ = ηzk q := ENNReal.toReal_ofReal (h0 q)

end Pouw.PearlC
