import Definitions.Pouw.PearlC.HiddenSystem
import Proofs.Pouw.PearlC.HiddenGamma
import Proofs.Pouw.PearlC.HiddenTileGame
import Proofs.Pouw.PearlC.TileCheck
import Proofs.Pouw.Draft.BatchSeq
import Proofs.Core.Circuit

/-!
# The hidden audit's open obligation from registered soundness

`tileProofSoundAll_of_registeredSound`: the hidden audit of tile checks that are sound against registered values
(`RegisteredSound`, provisional), whose accepted proofs draw the cone of the checked gates (`Covers`), and that compute
the good-tile predicate (`ChecksGood`, L3), meets `TileProofSoundAll` at any `ηzk` above every tile check's error. Per
tile (`TileGame.ofRegisteredSound`): outside the event that every transcript has a drawn wrong unit or is off the
tile's registered values, an accepted proof has a transcript holding them and correct on the cone of the checked
gates, so the tile is good or blank by L3. The class reads the game at the commitment's name, so a tile that the
commitment does not name for the verifier's oracle, salt and workload is rejected outright.

`HiddenSystem.honestProver_bounded`: the honest prover is in the class (`ProverBounded`), from `HonestIn`, when no two
of its own tiles' proof games are equal on different values and its batch reads as no other list of games.
`gammaHidden_of_registeredSound` is both: `GγHidden` from `GγSampled`, and the honest prover in its class.

`HiddenSystem.checksGood_ofTileCheck`: a tile check's L3 (`TileCheck.Computes`) is L3 on its hidden system, when its
registered values bind the circuit's words (no `Collides`), its public and registered wires are apart, its tiles lie in
their units and its proof systems register both. `gammaHidden_of_tileCheck` is `gammaHidden_of_registeredSound` on it.
-/

namespace Pouw.PearlC

open scoped ENNReal
open Pouw.PearlC.Assumptions Pouw.Draft.SampledProofs

variable {Q R S : Type}

/-- `tileProofSoundAll_of_tileGame` with a `TileGame` for the layout's tiles only. -/
theorem tileProofSoundAll_of_tileGame_lt [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (P : Protocol Q R S)
    (TR : TileRules Q R S) (D : Layout → Prop) (HA : HiddenAudit Q R S)
    (Cls : Transcript → HA.Opening → (g : Verity.Game Bool) → Verity.Game.Strategy g → ℕ → Prop) (ηzk : ℕ → ℝ)
    (h0 : ∀ q, 0 ≤ ηzk q) (hB : ∀ τ o G σ q, HA.Bounded τ o G σ q → BoundedByInduced Cls τ o G σ q)
    (tg : ∀ U : Workload, U.InDomain P D → ∀ (τ : Transcript) (o : HA.Opening) (H : Q → R) (s : S) (g : ℕ),
      g < (TR.tiling U.layout).NT → TileGame P TR HA Cls ηzk U τ o H s g) :
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
      (fun s hs => (tg U hU (τ x.1 x.2.1) (op x.1 x.2.1) x.1 x.2.1 (x.2.2 i).1 (x.2.2 i).1.isLt).prob_le h0
        hgood q s hs)
    calc _ ≤ (ENNReal.ofReal (ηzk q)).toReal := ENNReal.toReal_mono ENNReal.ofReal_ne_top hle
      _ = ηzk q := ENNReal.toReal_ofReal (h0 q)

/-- **What a tile check sound against registered values gives for one tile.** On a commitment that names the
verifier's oracle, salt and workload: the tile check's game, with `Bad` the event that every transcript has a drawn
wrong unit or is off the tile's registered values; its probability is at most `η` by `RegisteredSound`, and outside it
an accepted proof means a good or blank tile by `Covers` and L3. On any other commitment the game rejects. -/
noncomputable def TileGame.ofRegisteredSound [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (P : Protocol Q R S) (TR : TileRules Q R S) (D : Layout → Prop) (HS : HiddenSystem Q R S)
    (η : (k : HS.K) → ℕ → (HS.sys k).Reg → ℝ≥0∞) (ηzk : ℕ → ℝ) (h0 : ∀ q, 0 ≤ ηzk q)
    (hRS : ∀ k, RegisteredSound (HS.sys k) (η k)) (hη : ∀ k q r, η k q r ≤ ENNReal.ofReal (ηzk q))
    (hcov : HS.Covers) (hL3 : HS.ChecksGood P TR D) (U : Workload) (hU : U.InDomain P D) (τ : Transcript)
    (o : HS.Opening) (H : Q → R) (s : S) (g : ℕ) (hg : g < (TR.tiling U.layout).NT) :
    TileGame P TR (HiddenAudit.ofSystem HS) HS.Cls ηzk U τ o H s g := by
  classical
  obtain ⟨⟨H', s', U'⟩, salts⟩ := o
  by_cases hc : ((H', s', U') : (Q → R) × S × Workload) = (H, s, U)
  · obtain ⟨rfl, rfl, rfl⟩ : H = H' ∧ s = s' ∧ U = U' := by simpa [eq_comm] using hc
    refine
      { Out := (HS.sys (HS.kind U g)).Out
        G := (HS.sys (HS.kind U g)).game ((HS.sys (HS.kind U g)).commit (HS.valueAt H s U τ g) (salts g))
        verdict := (HS.sys (HS.kind U g)).accept
        game_eq := ite_eq_left rfl
        Bad := fun _ x => ∀ X : (HS.C (HS.kind U g)).Transcript,
          ¬ Disjoint ((HS.part (HS.kind U g)).wrong X) ((HS.sys (HS.kind U g)).drawn x) ∨
            ∃ v ∈ (HS.sys (HS.kind U g)).registered, X v ≠ HS.valueAt H s U τ g v
        bad_le := fun q σ hσ => ?_
        good_of_accept := fun σ x hx hnb => ?_ }
    · have h1 := hσ g (ite_eq_left rfl)
      simp only [cast_cast, cast_eq, Pouw.Draft.Game.ofMap_toMap] at h1
      obtain ⟨X, hp⟩ := hRS _ q _ _ σ h1
      refine ENNReal.toReal_le_of_le_ofReal (h0 q)
        (le_trans (le_trans (Verity.Game.prob_mono ?_ _ σ) hp) (hη _ q _))
      intro x hx
      exact ⟨hx.1, hx.2 X⟩
    · push Not at hnb
      obtain ⟨X, hD, hreg⟩ := hnb
      refine hL3 U hU H s τ g hg X (fun v hv => ?_) hreg
      by_contra hw
      exact Finset.disjoint_left.mp hD ((Verity.Partition.mem_wrong _).mpr hw) (hcov _ x hx hv)
  · refine
      { Out := Unit
        G := .ret ()
        verdict := fun _ => false
        game_eq := ite_eq_right hc
        Bad := fun _ _ => False
        bad_le := fun q σ _ => ?_
        good_of_accept := fun _ _ h _ => absurd h (by simp) }
    simp [Verity.Game.prob, h0 q]

/-- **The hidden audit's open obligation from registered soundness.** If every tile check is sound against registered
values up to an error at most `ηzk`, its accepted proofs draw the cone of its checked gates, and it computes the
good-tile predicate (L3), the hidden audit of the tile checks meets `TileProofSoundAll` at `ηzk`. -/
theorem tileProofSoundAll_of_registeredSound [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (P : Protocol Q R S) (TR : TileRules Q R S) (D : Layout → Prop) (HS : HiddenSystem Q R S)
    (η : (k : HS.K) → ℕ → (HS.sys k).Reg → ℝ≥0∞) (ηzk : ℕ → ℝ) (h0 : ∀ q, 0 ≤ ηzk q)
    (hRS : ∀ k, RegisteredSound (HS.sys k) (η k)) (hη : ∀ k q r, η k q r ≤ ENNReal.ofReal (ηzk q))
    (hcov : HS.Covers) (hL3 : HS.ChecksGood P TR D) :
    TileProofSoundAll P TR D (HiddenAudit.ofSystem HS) ηzk :=
  tileProofSoundAll_of_tileGame_lt P TR D (HiddenAudit.ofSystem HS) HS.Cls ηzk h0 (fun _ _ _ _ _ h => h)
    fun U hU τ o H s g hg => TileGame.ofRegisteredSound P TR D HS η ηzk h0 hRS hη hcov hL3 U hU τ o H s g hg

/-- The honest prover of a tile check, carried to the game of another reading with the same kind, registered values
and opening, is in that reading's class. -/
theorem cls_honest_of_sigma {K O Rt : Type} {C : K → Verity.Circuit} {n : K → ℕ}
    {part : (k : K) → Verity.Partition (C k) (n k)} (sys : (k : K) → System (C k) (n k) (part k) O Rt)
    (hHon : ∀ k, HonestIn (sys k)) {k k' : K} {a : (C k).Transcript} {a' : (C k').Transcript} {op op' : O}
    (h : (⟨k, a, op⟩ : (k : K) × (C k).Transcript × O) = ⟨k', a', op'⟩)
    (σ : Verity.Game.Strategy (((sys k').game ((sys k').commit a' op')).map (sys k').accept))
    (hσ : HEq σ (Verity.Game.Strategy.toMap (sys k).accept _ ((sys k).honest a op))) (q : ℕ) :
    (sys k').Cls a' op' (Verity.Game.Strategy.ofMap (sys k').accept _ σ) q := by
  obtain ⟨rfl, h2⟩ := Sigma.mk.inj h
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj (eq_of_heq h2)
  rw [eq_of_heq hσ, Pouw.Draft.Game.ofMap_toMap]
  exact hHon k q a op

/-- **The honest prover is in the class.** If every tile check's honest prover is in its class (`HonestIn`), the honest
prover of the proof phase, under the honest opening, is in the hidden audit's class at every budget, on every
transcripts whose tiles' proof games are equal only on equal values (`Distinct`) and whose batches read as no other
list of games (`Unambiguous`). -/
theorem HiddenSystem.honestProver_bounded (HS : HiddenSystem Q R S) (hHon : ∀ k, HonestIn (HS.sys k))
    (TR : TileRules Q R S) (U : Workload) (τ : (Q → R) → S → Transcript) (salts : (Q → R) → S → ℕ → HS.O)
    (w : ℕ → ℕ) (t q : ℕ) (hD : ∀ H s, HS.Distinct U H s (τ H s) (HS.honestOpening U salts H s))
    (hU : ∀ x : (Q → R) × S × (Fin t → Ticket w (TR.tiling U.layout).NT),
      Pouw.Draft.Game.Unambiguous (List.ofFn fun i =>
        HS.game x.1 x.2.1 U (HS.root U (τ x.1 x.2.1) (HS.honestOpening U salts x.1 x.2.1)) (x.2.2 i).1)) :
    (HiddenAudit.ofSystem HS).ProverBounded (HS.honestProver TR U τ salts w t) q := by
  classical
  intro x pre g post e σ hσ
  let Gm := HS.game x.1 x.2.1 U (HS.root U (τ x.1 x.2.1) (HS.honestOpening U salts x.1 x.2.1))
  let hon := HS.honestTile U (τ x.1 x.2.1) (HS.honestOpening U salts x.1 x.2.1) x.1 x.2.1
  let ts : List ℕ := List.ofFn fun i => ((x.2.2 i).1 : ℕ)
  have hL : pre ++ g :: post = ts.map Gm :=
    (hU x pre g post e).trans (List.map_ofFn (f := fun i => ((x.2.2 i).1 : ℕ)) (g := Gm)).symm
  have hheq : HEq (HS.honestProver TR U τ salts w t x)
      (Pouw.Draft.Game.Strategy.batchSeq (ts.map Gm) (Pouw.Draft.Game.Strategies.ofMap Gm hon ts)) :=
    cast_heq _ _
  obtain ⟨ss', hss', rfl⟩ := Pouw.Draft.Game.inducedAt_of_heq _ _ hheq pre g post e hL σ hσ
  obtain ⟨t₀, rfl, hget⟩ := Pouw.Draft.Game.get_ofMap Gm hon pre ts ss' hL hss'
  intro t' e'
  have hon_eq : hon t₀ = cast (congrArg Verity.Game.Strategy (ite_eq_left rfl).symm)
      (Verity.Game.Strategy.toMap (HS.sys (HS.kind U t₀)).accept _
        ((HS.sys (HS.kind U t₀)).honest (HS.valueAt x.1 x.2.1 U (τ x.1 x.2.1) t₀) (salts x.1 x.2.1 t₀))) := by
    show HS.honestTile _ _ _ _ _ _ = _
    unfold HiddenSystem.honestTile
    exact dite_eq_left rfl
  have h1 : HEq (hon t₀) (Verity.Game.Strategy.toMap (HS.sys (HS.kind U t₀)).accept _
      ((HS.sys (HS.kind U t₀)).honest (HS.valueAt x.1 x.2.1 U (τ x.1 x.2.1) t₀) (salts x.1 x.2.1 t₀))) := by
    rw [hon_eq]
    exact cast_heq _ _
  have h2 : HEq (Pouw.Draft.Game.Strategies.get pre ss') (hon t₀) := by
    rw [hget]
    exact cast_heq _ _
  exact cls_honest_of_sigma HS.sys hHon (hD x.1 x.2.1 t₀ t' ((ite_eq_left rfl).symm.trans e')) _
    ((cast_heq _ _).trans (h2.trans h1)) q

/-- **`GγHidden` from `GγSampled` and registered soundness, with the honest prover in the class.** For tile checks
sound against registered values up to an error at most `ηzk`, whose accepted proofs draw the cone of their checked
gates, that compute the good-tile predicate (L3) and whose honest provers are in their classes: the hidden audit of the
tile checks has `GγSampled`'s γ at `ηzk` more error per draw, and its honest prover is in its class on every
transcripts whose tiles' proof games are equal only on equal values and whose batches read as no other list of games. -/
theorem gammaHidden_of_registeredSound [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
    (P : Protocol Q R S) (TR : TileRules Q R S) (D : Layout → Prop) (HS : HiddenSystem Q R S) (γ : ℝ)
    (η : ℕ → ℕ → ℝ) (ηrs : (k : HS.K) → ℕ → (HS.sys k).Reg → ℝ≥0∞) (ηzk : ℕ → ℝ) (h0 : ∀ q, 0 ≤ ηzk q)
    (hS : GγSampled CM P TR D γ η) (hRS : ∀ k, RegisteredSound (HS.sys k) (ηrs k))
    (hη : ∀ k q r, ηrs k q r ≤ ENNReal.ofReal (ηzk q)) (hHon : ∀ k, HonestIn (HS.sys k)) (hcov : HS.Covers)
    (hL3 : HS.ChecksGood P TR D) :
    GγHidden CM P TR D (HiddenAudit.ofSystem HS) γ η ηzk ∧
      ∀ (U : Workload) (τ : (Q → R) → S → Transcript) (salts : (Q → R) → S → ℕ → HS.O) (w : ℕ → ℕ) (t q : ℕ),
        (∀ H s, HS.Distinct U H s (τ H s) (HS.honestOpening U salts H s)) →
        (∀ x : (Q → R) × S × (Fin t → Ticket w (TR.tiling U.layout).NT),
          Pouw.Draft.Game.Unambiguous (List.ofFn fun i =>
            HS.game x.1 x.2.1 U (HS.root U (τ x.1 x.2.1) (HS.honestOpening U salts x.1 x.2.1)) (x.2.2 i).1)) →
        (HiddenAudit.ofSystem HS).ProverBounded (HS.honestProver TR U τ salts w t) q :=
  ⟨gammaHidden_of_sampled_all CM P TR D (HiddenAudit.ofSystem HS) γ η ηzk hS
      (tileProofSoundAll_of_registeredSound P TR D HS ηrs ηzk h0 hRS hη hcov hL3),
    fun U τ salts w t q hD hU => HS.honestProver_bounded hHon TR U τ salts w t q hD hU⟩

/-- **A tile check's L3 is L3 on its hidden system**, when its registered values bind the circuit's words (no
`Collides`: the plain Program), its public and registered wires are apart, every proof system registers both, and each
tile lies in its unit (`hunit`, `hcols`), so that the domain's weight clause gives the check's weight rows. -/
theorem HiddenSystem.checksGood_ofTileCheck [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (P : Protocol Q R S)
    (TR : TileRules Q R S) (D : Layout → Prop) (TC : TileCheck Q R S) (O Rt : Type)
    (sys : (k : Shape × Finset ℕ × Finset ℕ) → System (TC.C k.1) (TC.n k.1) (TC.part k.1) O Rt)
    (hL3 : TC.Computes P TR D) (hcol : ∀ U u rows cols c X, ¬ TC.Collides U u rows cols c X)
    (hdis : ∀ sh, Disjoint (TC.pub sh) (TC.reg sh)) (hreg : ∀ k, TC.pub k.1 ∪ TC.reg k.1 ⊆ (sys k).registered)
    (hunit : ∀ L, ∀ g < (TR.tiling L).NT, (TR.tiling L).unit g < L.N)
    (hcols : ∀ L g, ∀ j ∈ (TR.tiling L).cols g, j < (L.shape ((TR.tiling L).unit g)).n) :
    (HiddenSystem.ofTileCheck TC TR O Rt sys).ChecksGood P TR D := by
  intro U hU H s τ t ht X hcone hX
  cases hc : τ ((TR.tiling U.layout).unit t) with
  | none => exact Finset.mem_union_right _ (TileCheck.mem_blank TR U τ t ht hc)
  | some c =>
    have hval : ∀ w, TC.registeredAt H s U τ ((TR.tiling U.layout).unit t) w =
        if w ∈ TC.pub _ then TC.anchorsOf H s U ((TR.tiling U.layout).unit t) c.act w
        else TC.rowsOf U ((TR.tiling U.layout).unit t) c w := fun w => by
      simp only [TileCheck.registeredAt, hc]
    have hacc : TC.Accepts P TR H s U t c X := by
      refine ⟨hcone, fun w hw => ?_, fun w hw => ?_, fun j hj => hU.2.2 _ (hunit _ t ht) j (hcols _ t j hj)⟩
      · exact (hX w (hreg _ (Finset.mem_union_left _ hw))).trans
          ((hval w).trans (ite_eq_left_of_eq_true _ _ (eq_true hw)))
      · have hnp : w ∉ TC.pub _ := Finset.disjoint_right.mp (hdis _) hw
        exact (hX w (hreg _ (Finset.mem_union_right _ hw))).trans
          ((hval w).trans (ite_eq_right_of_eq_false _ _ (eq_false hnp)))
    rcases TC.mem_of_computes P TR D hL3 U hU H s τ t ht c hc X hacc with h | h
    · exact h
    · exact absurd h (hcol _ _ _ _ _ _)

/-- **`GγHidden` for a tile check.** `gammaHidden_of_registeredSound` on the hidden system of a tile check meeting L3
(`TileCheck.Computes`), whose registered values bind the circuit's words, whose public and registered wires are apart,
whose tiles lie in their units, and whose proof systems register both, are sound against registered values up to an
error at most `ηzk`, draw the cone of a tile's words on an accepted proof, and have their honest provers in their
classes. -/
theorem gammaHidden_of_tileCheck [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
    (P : Protocol Q R S) (TR : TileRules Q R S) (D : Layout → Prop) (TC : TileCheck Q R S) (O Rt : Type)
    (sys : (k : Shape × Finset ℕ × Finset ℕ) → System (TC.C k.1) (TC.n k.1) (TC.part k.1) O Rt) (γ : ℝ)
    (η : ℕ → ℕ → ℝ) (ηrs : (k : Shape × Finset ℕ × Finset ℕ) → ℕ → (sys k).Reg → ℝ≥0∞) (ηzk : ℕ → ℝ)
    (h0 : ∀ q, 0 ≤ ηzk q) (hS : GγSampled CM P TR D γ η) (hRS : ∀ k, RegisteredSound (sys k) (ηrs k))
    (hη : ∀ k q r, ηrs k q r ≤ ENNReal.ofReal (ηzk q)) (hHon : ∀ k, HonestIn (sys k))
    (hcov : ∀ k (x : (sys k).Out), (sys k).accept x = true →
      (TC.part k.1).cone (TC.out k.1 k.2.1 k.2.2) ⊆ (sys k).drawn x)
    (hL3 : TC.Computes P TR D) (hcol : ∀ U u rows cols c X, ¬ TC.Collides U u rows cols c X)
    (hdis : ∀ sh, Disjoint (TC.pub sh) (TC.reg sh)) (hreg : ∀ k, TC.pub k.1 ∪ TC.reg k.1 ⊆ (sys k).registered)
    (hunit : ∀ L, ∀ g < (TR.tiling L).NT, (TR.tiling L).unit g < L.N)
    (hcols : ∀ L g, ∀ j ∈ (TR.tiling L).cols g, j < (L.shape ((TR.tiling L).unit g)).n) :
    GγHidden CM P TR D (HiddenAudit.ofSystem (HiddenSystem.ofTileCheck TC TR O Rt sys)) γ η ηzk ∧
      ∀ (U : Workload) (τ : (Q → R) → S → Transcript) (salts : (Q → R) → S → ℕ → O) (w : ℕ → ℕ) (t q : ℕ),
        (∀ H s, (HiddenSystem.ofTileCheck TC TR O Rt sys).Distinct U H s (τ H s)
          ((HiddenSystem.ofTileCheck TC TR O Rt sys).honestOpening U salts H s)) →
        (∀ x : (Q → R) × S × (Fin t → Ticket w (TR.tiling U.layout).NT),
          Pouw.Draft.Game.Unambiguous (List.ofFn fun i =>
            (HiddenSystem.ofTileCheck TC TR O Rt sys).game x.1 x.2.1 U
              ((HiddenSystem.ofTileCheck TC TR O Rt sys).root U (τ x.1 x.2.1)
                ((HiddenSystem.ofTileCheck TC TR O Rt sys).honestOpening U salts x.1 x.2.1)) (x.2.2 i).1)) →
        (HiddenAudit.ofSystem (HiddenSystem.ofTileCheck TC TR O Rt sys)).ProverBounded
          ((HiddenSystem.ofTileCheck TC TR O Rt sys).honestProver TR U τ salts w t) q :=
  gammaHidden_of_registeredSound CM P TR D (HiddenSystem.ofTileCheck TC TR O Rt sys) γ η ηrs ηzk h0 hS hRS hη hHon
    hcov (HiddenSystem.checksGood_ofTileCheck P TR D TC O Rt sys hL3 hcol hdis hreg hunit hcols)

end Pouw.PearlC
