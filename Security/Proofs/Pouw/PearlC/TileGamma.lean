import Proofs.Pouw.PearlC.Gamma
import Proofs.Pouw.Sampling

/-!
# γ under the sampled audit, from TT_OUT per tile

Staged; nothing pinned. The pins proposed are `pearlCSampled8192`, `pearlCSampled16384`, `pearlCTiles_total` and
`ttOutTilePearlCWitness`.

* `gammaSampled`: if every good tile's `W_ref` is at most `Ω` times its credit, `TTOutTile(γ₀)` gives `GγSampled` at
  `γ = 1 − (1 − γ₀)/Ω` with error `(1 − ε_s)^t + ε(q, N)`. Pointwise, an accepted run either has more than `ε_s` of the
  layout's tickets on bad tiles of committed units, which `t` work-weighted draws all miss with probability at most
  `(1 − ε_s)^t` (`workWeightedSampling`, pinned), or its good tiles carry the claimed work less `ε_s`, and then TT_OUT's
  event holds.
* **The headlines are the unit-level worst case under the cap.** A good tile's credit is `(1 − ρ)·creditOf` times its
  share and its `W_ref` is `wrefOf` times the same share, so `Ω = wrefOf/((1 − ρ)·creditOf)` per shape and γ is
  `pearlCGamma8192`'s and `pearlCGamma16384`'s: `281273/52340000` (0.5374%) at 8192³ and `536953/103540000` (0.5186%) at
  16384³. The sampled audit certifies `(1 − γ)·(W_claimed − ε_s·W)`.
* `pearlCTiles_total`: the audit grid covers each unit's words once, so the `W` that `GγSampled` certifies against is
  the layout's `∑_u wrefOf`.
* `ttOutTilePearlCWitness`: `TTOutTilePearlC` is satisfiable, beside an honest reference whose cost is at most the
  layout's `W_ref` tile by tile, and an in-domain workload whose one tile is within the cap and has positive credit.
-/

namespace Pouw.PearlC

open Finset Assumptions Pouw.Fp8Atom

theorem goodTiles_subset {Q R S : Type} (P : Protocol Q R S) (TR : TileRules Q R S) (U : Workload) (H : Q → R)
    (s : S) (τ : CommitLog) : goodTiles P TR U H s τ ⊆ range (TR.tiling U.layout).NT \ blankTiles TR U τ := by
  classical
  intro g hg
  unfold goodTiles at hg
  rw [mem_filter] at hg
  obtain ⟨hr, c, hc, _⟩ := hg
  rw [mem_sdiff]
  refine ⟨hr, fun hb => ?_⟩
  unfold blankTiles at hb
  rw [mem_filter, hc] at hb
  exact Option.some_ne_none _ hb.2

/-- **The sampled audit's bound over any set of accepted tiles.** Fix a workload, the transcripts `τ` and, for each
oracle and salt, a set `good` of committed tiles of `τ` on which `Wcred ≤ Ω·credit`. If their credit exceeds `T/(1 − γ₀)`
with probability at most `e`, then the audit that accepts `good ∪ blank` does so while `T` is below `1 − γ` of the
claimed work less `ε_s` of the layout's, `γ = 1 − (1 − γ₀)/Ω`, with probability at most `(1 − ε_s)^t + e`. -/
theorem sampledOfGood {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (TR : TileRules Q R S)
    (U : Workload) (τ : (Q → R) → S → CommitLog) (good : (Q → R) → S → Finset ℕ) (γ₀ Ω T e : ℝ) (hγ₀ : γ₀ < 1)
    (hΩ : 0 < Ω) (hsub : ∀ H s, good H s ⊆ range (TR.tiling U.layout).NT \ blankTiles TR U (τ H s))
    (hW : ∀ H s, ∀ g ∈ good H s, TR.Wcred U.layout g (actOf (τ H s) ((TR.tiling U.layout).unit g)) ≤
      Ω * TR.credit H s U g (actOf (τ H s) ((TR.tiling U.layout).unit g)))
    (hWc : ∀ g < (TR.tiling U.layout).NT, ∀ act : Codes, TR.Wcred U.layout g act ≤ TR.Wref U.layout g)
    (hTT : pr (fun x : (Q → R) × S => T / (1 - γ₀) < ∑ g ∈ good x.1 x.2,
        TR.credit x.1 x.2 U g (actOf (τ x.1 x.2) ((TR.tiling U.layout).unit g))) ≤ e)
    (w : ℕ → ℕ) (κ : ℝ) (hκ : 0 < κ) (hw : ∀ g < (TR.tiling U.layout).NT, (w g : ℝ) = κ * TR.Wref U.layout g)
    (t : ℕ) (εs : ℝ) (hεs : εs ≤ 1) :
    pr (fun x : (Q → R) × S × (Fin t → Ticket w (TR.tiling U.layout).NT) =>
        AuditAccepts (good x.1 x.2.1 ∪ blankTiles TR U (τ x.1 x.2.1)) x.2.2 ∧
        T < (1 - (1 - (1 - γ₀) / Ω)) * (claimedWref TR U (τ x.1 x.2.1) - εs * tileWrefTotal TR U.layout))
      ≤ (1 - εs) ^ t + e := by
  classical
  have h1γ₀ : 0 < 1 - γ₀ := by linarith
  let K : (Q → R) × S → Finset ℕ := fun y => good y.1 y.2 ∪ blankTiles TR U (τ y.1 y.2)
  let E1 : ((Q → R) × S) × (Fin t → Ticket w (TR.tiling U.layout).NT) → Prop := fun y =>
    AuditAccepts (K y.1) y.2 ∧
      εs * ∑ g ∈ range (TR.tiling U.layout).NT, (w g : ℝ) < ∑ g ∈ range (TR.tiling U.layout).NT \ K y.1, (w g : ℝ)
  let E2 : (Q → R) × S → Prop := fun y =>
    T / (1 - γ₀) < ∑ g ∈ good y.1 y.2, TR.credit y.1 y.2 U g (actOf (τ y.1 y.2) ((TR.tiling U.layout).unit g))
  have hsumw : ∀ F : Finset ℕ, F ⊆ range (TR.tiling U.layout).NT →
      ∑ g ∈ F, (w g : ℝ) = κ * ∑ g ∈ F, TR.Wref U.layout g := by
    intro F hF
    rw [mul_sum]
    exact sum_congr rfl fun g hg => hw g (mem_range.mp (hF hg))
  have hstep : ∀ x : (Q → R) × S × (Fin t → Ticket w (TR.tiling U.layout).NT),
      (AuditAccepts (K (x.1, x.2.1)) x.2.2 ∧
        T < (1 - (1 - (1 - γ₀) / Ω)) * (claimedWref TR U (τ x.1 x.2.1) - εs * tileWrefTotal TR U.layout)) →
      E1 ((x.1, x.2.1), x.2.2) ∨ E2 (x.1, x.2.1) := by
    intro x ⟨hacc, hTlt⟩
    by_cases h1 : E1 ((x.1, x.2.1), x.2.2)
    · exact Or.inl h1
    right
    set τ' := τ x.1 x.2.1 with hτ
    set gd := good x.1 x.2.1 with hgd
    set blank := blankTiles TR U τ' with hblank
    have hK : K (x.1, x.2.1) = gd ∪ blank := rfl
    have hbadw : ∑ g ∈ range (TR.tiling U.layout).NT \ K (x.1, x.2.1), (w g : ℝ) ≤
        εs * ∑ g ∈ range (TR.tiling U.layout).NT, (w g : ℝ) := by
      by_contra h
      exact h1 ⟨hacc, lt_of_not_ge h⟩
    rw [hsumw _ sdiff_subset, hsumw _ (Subset.refl _)] at hbadw
    have hbad : ∑ g ∈ range (TR.tiling U.layout).NT \ K (x.1, x.2.1), TR.Wref U.layout g ≤
        εs * tileWrefTotal TR U.layout := by
      unfold tileWrefTotal
      have h' : κ * ∑ g ∈ range (TR.tiling U.layout).NT \ K (x.1, x.2.1), TR.Wref U.layout g ≤
          κ * (εs * ∑ g ∈ range (TR.tiling U.layout).NT, TR.Wref U.layout g) := by linarith
      exact le_of_mul_le_mul_left h' hκ
    have hset : (range (TR.tiling U.layout).NT \ blank) \ gd = range (TR.tiling U.layout).NT \ K (x.1, x.2.1) := by
      rw [hK]
      ext g
      simp only [mem_sdiff, mem_union]
      tauto
    let f : ℕ → ℝ := fun g => TR.Wcred U.layout g (actOf τ' ((TR.tiling U.layout).unit g))
    have hsplit := sum_sdiff (f := f) (hsub x.1 x.2.1)
    rw [hset] at hsplit
    have hbadc : ∑ g ∈ range (TR.tiling U.layout).NT \ K (x.1, x.2.1), f g ≤ εs * tileWrefTotal TR U.layout :=
      le_trans (sum_le_sum fun g hg => hWc g (mem_range.mp (mem_sdiff.mp hg).1) _) hbad
    have hclaim : claimedWref TR U τ' - εs * tileWrefTotal TR U.layout ≤ ∑ g ∈ gd, f g := by
      show ∑ g ∈ range (TR.tiling U.layout).NT \ blank, f g - εs * tileWrefTotal TR U.layout ≤ ∑ g ∈ gd, f g
      rw [← hsplit]
      linarith
    have hcred : ∑ g ∈ gd, f g ≤
        Ω * ∑ g ∈ gd, TR.credit x.1 x.2.1 U g (actOf τ' ((TR.tiling U.layout).unit g)) := by
      rw [mul_sum]
      exact sum_le_sum fun g hg => hW x.1 x.2.1 g hg
    have hden : 1 - (1 - (1 - γ₀) / Ω) = (1 - γ₀) / Ω := by ring
    rw [hden] at hTlt
    show T / (1 - γ₀) < ∑ g ∈ gd, TR.credit x.1 x.2.1 U g (actOf τ' ((TR.tiling U.layout).unit g))
    rw [div_lt_iff₀ h1γ₀]
    have hpos : 0 < (1 - γ₀) / Ω := div_pos h1γ₀ hΩ
    calc T < (1 - γ₀) / Ω * (claimedWref TR U τ' - εs * tileWrefTotal TR U.layout) := hTlt
      _ ≤ (1 - γ₀) / Ω * ∑ g ∈ gd, f g := mul_le_mul_of_nonneg_left hclaim hpos.le
      _ ≤ (1 - γ₀) / Ω * (Ω * ∑ g ∈ gd, TR.credit x.1 x.2.1 U g (actOf τ' ((TR.tiling U.layout).unit g))) :=
          mul_le_mul_of_nonneg_left hcred hpos.le
      _ = (∑ g ∈ gd, TR.credit x.1 x.2.1 U g (actOf τ' ((TR.tiling U.layout).unit g))) * (1 - γ₀) := by
          field_simp
  calc _ ≤ pr (fun x : (Q → R) × S × (Fin t → Ticket w (TR.tiling U.layout).NT) =>
          E1 ((x.1, x.2.1), x.2.2) ∨ E2 (x.1, x.2.1)) := pr_mono hstep
    _ ≤ pr (fun x : (Q → R) × S × (Fin t → Ticket w (TR.tiling U.layout).NT) => E1 ((x.1, x.2.1), x.2.2)) +
        pr (fun x : (Q → R) × S × (Fin t → Ticket w (TR.tiling U.layout).NT) => E2 (x.1, x.2.1)) := pr_or_le _ _
    _ ≤ (1 - εs) ^ t + e := by
        refine add_le_add ?_ ((pr_fst2_le E2).trans hTT)
        calc _ = pr E1 := pr_equiv (Equiv.prodAssoc _ _ _).symm E1
          _ ≤ _ := Pouw.Proofs.workWeightedSampling ((Q → R) × S) w (TR.tiling U.layout).NT t K εs hεs

theorem gammaSampled {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
    (P : Protocol Q R S) (TR : TileRules Q R S) (D : UnitLayout → Prop) (γ₀ Ω : ℝ) (ε : ℕ → ℕ → ℝ) (hγ₀ : γ₀ < 1)
    (hΩ : 0 < Ω)
    (hW : ∀ U : Workload, U.InDomain P D → ∀ (H : Q → R) (s : S), ∀ g < (TR.tiling U.layout).NT, ∀ act : Codes,
      TR.capOK H s U g act → TR.Wcred U.layout g act ≤ Ω * TR.credit H s U g act)
    (hWc : ∀ U : Workload, U.InDomain P D → ∀ g < (TR.tiling U.layout).NT, ∀ act : Codes,
      TR.Wcred U.layout g act ≤ TR.Wref U.layout g)
    (hTT : TTOutTile CM P TR D γ₀ ε) : GγSampled CM P TR D (1 - (1 - γ₀) / Ω) ε := by
  intro U hU T hT q hq A hA w κ hκ hw t εs hεs
  classical
  refine sampledOfGood TR U (fun H s => A.transcript U H s) (fun H s => goodTiles P TR U H s (A.transcript U H s))
    γ₀ Ω T (ε q U.layout.N) hγ₀ hΩ (fun H s => goodTiles_subset P TR U H s _) ?_ (hWc U hU)
    (hTT.2 U hU T hT q hq A hA) w κ hκ hw t εs hεs
  intro H s g hg
  unfold goodTiles at hg
  rw [mem_filter] at hg
  obtain ⟨hgr, c, hc, hgc⟩ := hg
  have hact : actOf (A.transcript U H s) ((TR.tiling U.layout).unit g) = c.act := by simp [actOf, hc]
  rw [hact]
  exact hW U hU H s g (mem_range.mp hgr) c.act hgc.1

/-- TT_OUT per tile on a domain gives it on any smaller domain. -/
theorem ttOutTile_mono_domain {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    {CM : CostModel Q R S} {P : Protocol Q R S} {TR : TileRules Q R S} {D D' : UnitLayout → Prop} {γ₀ : ℝ}
    {ε : ℕ → ℕ → ℝ} (hD : ∀ L, D' L → D L) (h : TTOutTile CM P TR D γ₀ ε) : TTOutTile CM P TR D' γ₀ ε :=
  ⟨h.1, fun U hU => h.2 U ⟨hU.1, hD _ hU.2⟩⟩

theorem gridTiles_unit_lt {tm tn : ℕ} {L : UnitLayout} {b : ℕ × ℕ × ℕ} (hb : b ∈ gridTiles tm tn L) : b.1 < L.N := by
  unfold gridTiles at hb
  simp only [List.mem_flatMap, List.mem_range, List.mem_map] at hb
  obtain ⟨u, hu, bi, _, bj, _, rfl⟩ := hb
  exact hu

theorem auditTiling_unit_lt (tm tn : ℕ) (L : UnitLayout) {g : ℕ} (hg : g < (auditTiling tm tn L).NT) :
    (auditTiling tm tn L).unit g < L.N := by
  simp only [auditTiling] at hg ⊢
  rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hg, Option.getD_some]
  exact gridTiles_unit_lt (List.getElem_mem hg)

/-- On a one-shape domain where `W_ref = ω·(1 − ρ)·creditOf`, every tile's credited `W_ref` is `ω` times its credit. -/
theorem pearlCTiles_wref {Q R S : Type} (sem : PearlCSem Q R S) (c : Costs) (ρ : ℚ) (sh : Shape) (ω : ℚ)
    (hω : wrefOf Params.pi c sh = ω * ((1 - ρ) * creditOf Params.pi c sh)) :
    ∀ U : Workload, U.InDomain (pearlCProtocol sem c ρ) (pearlCDomainAt Params.pi sh) → ∀ (H : Q → R) (s : S),
      ∀ g < ((pearlCTiles sem c ρ).tiling U.layout).NT, ∀ act : Codes, (pearlCTiles sem c ρ).capOK H s U g act →
        (pearlCTiles sem c ρ).Wcred U.layout g act ≤ (ω : ℝ) * (pearlCTiles sem c ρ).credit H s U g act := by
  intro U hU H s g hg act _
  have hu := auditTiling_unit_lt 64 64 U.layout hg
  have hsh : U.layout.shape ((auditTiling 64 64 U.layout).unit g) = sh := hU.2.2 _ hu
  simp only [pearlCTiles]
  rw [hsh, hω]
  push_cast
  exact le_of_eq (by ring)

theorem tileShare_mono (sh : Shape) {r r' : Finset ℕ} (h : r ⊆ r') (cols : Finset ℕ) :
    tileShare sh r cols ≤ tileShare sh r' cols := by
  unfold tileShare
  have hc : (r.card : ℚ) ≤ r'.card := by exact_mod_cast card_le_card h
  gcongr

theorem tileShare_nonneg' (sh : Shape) (rows cols : Finset ℕ) : 0 ≤ tileShare sh rows cols := by
  unfold tileShare
  positivity

theorem passRows_subset {Q R S : Type} (sem : PearlCSem Q R S) (sh : Shape) (act : Codes) (rows : Finset ℕ) :
    sem.passRows sh act rows ⊆ rows := by
  classical
  unfold PearlCSem.passRows
  exact filter_subset _ _

theorem wrefOf_nonneg (c : Costs) (s : Shape) : 0 ≤ wrefOf Params.pi c s := by
  unfold wrefOf creditOf Params.pi
  positivity

/-- A tile's credited `W_ref` is at most its `W_ref`. -/
theorem pearlCTiles_wcred_le {Q R S : Type} (sem : PearlCSem Q R S) (c : Costs) (ρ : ℚ) (L : UnitLayout) (g : ℕ)
    (act : Codes) : (pearlCTiles sem c ρ).Wcred L g act ≤ (pearlCTiles sem c ρ).Wref L g := by
  simp only [pearlCTiles]
  exact_mod_cast mul_le_mul_of_nonneg_left (tileShare_mono _ (passRows_subset sem _ act _) _) (wrefOf_nonneg c _)

/-- The headline pattern: `TTOutTilePearlC` at costs `c` and cap `ρ` gives `GγSampled` on the one-shape domain at
`γ = 1 − (399/400)/ω`, where `W_ref = ω·(1 − ρ)·creditOf`. -/
theorem pearlCSampledAt {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
    (sem : PearlCSem Q R S) (c : Costs) (ρ : ℚ) (s : Shape) (ω γ : ℚ) (hω0 : 0 < ω)
    (hω : wrefOf Params.pi c s = ω * ((1 - ρ) * creditOf Params.pi c s)) (hγ : γ = 1 - (1 - 1 / 400) / ω)
    (hTT : TTOutTilePearlC CM sem c ρ) :
    GγSampled CM (pearlCProtocol sem c ρ) (pearlCTiles sem c ρ) (pearlCDomainAt Params.pi s) (γ : ℝ) εPearlC := by
  have key := gammaSampled CM (pearlCProtocol sem c ρ) (pearlCTiles sem c ρ) (pearlCDomainAt Params.pi s) (1 / 400)
    (ω : ℝ) εPearlC (by norm_num) (by exact_mod_cast hω0) (pearlCTiles_wref sem c ρ s ω hω)
    (fun U _ g _ act => pearlCTiles_wcred_le sem c ρ U.layout g act) (ttOutTile_mono_domain (fun _ h => h.1) hTT)
  have e : (γ : ℝ) = 1 - (1 - 1 / 400) / (ω : ℝ) := by rw [hγ]; push_cast; ring
  rw [e]
  exact key

/-- The sampled audit at 8192³ under the deployment cap `ρ = 1/400`, credit `(1 − ρ)·creditOf` per accepted tile. -/
theorem pearlCSampled8192 {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
    (sem : PearlCSem Q R S) (hTT : TTOutTilePearlC CM sem Costs.adopted (1 / 400)) :
    GγSampled CM (pearlCProtocol sem Costs.adopted (1 / 400)) (pearlCTiles sem Costs.adopted (1 / 400))
      (pearlCDomainAt Params.pi sh8192) ((281273 / 52340000 : ℚ) : ℝ) εPearlC :=
  pearlCSampledAt CM sem _ _ _ (130850 / 130473) _ (by norm_num)
    (by norm_num [wrefOf, creditOf, Params.pi, Costs.adopted, sh8192]) (by norm_num) hTT

/-- The sampled audit at 16384³ under the deployment cap `ρ = 1/400`. -/
theorem pearlCSampled16384 {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
    (sem : PearlCSem Q R S) (hTT : TTOutTilePearlC CM sem Costs.adopted (1 / 400)) :
    GγSampled CM (pearlCProtocol sem Costs.adopted (1 / 400)) (pearlCTiles sem Costs.adopted (1 / 400))
      (pearlCDomainAt Params.pi sh16384) ((536953 / 103540000 : ℚ) : ℝ) εPearlC :=
  pearlCSampledAt CM sem _ _ _ (258850 / 258153) _ (by norm_num)
    (by norm_num [wrefOf, creditOf, Params.pi, Costs.adopted, sh16384]) (by norm_num) hTT

/-! ## The audit grid partitions each unit's words, so `W` is the layout's `W_ref` -/

theorem sum_range_getD {α : Type} (F : α → ℝ) (d : α) : ∀ l : List α,
    ∑ g ∈ range l.length, F (l.getD g d) = (l.map F).sum
  | [] => by simp
  | a :: l => by
    rw [List.length_cons, sum_range_succ', List.map_cons, List.sum_cons]
    simp only [List.getD_cons_succ, List.getD_cons_zero]
    rw [sum_range_getD F d l, add_comm]

theorem sum_map_flatMap {α β : Type} (F : β → ℝ) (f : α → List β) : ∀ l : List α,
    ((l.flatMap f).map F).sum = (l.map fun a => ((f a).map F).sum).sum
  | [] => by simp
  | a :: l => by simp [List.flatMap_cons, List.sum_append, sum_map_flatMap F f l]

theorem sum_map_range (f : ℕ → ℝ) : ∀ n, ((List.range n).map f).sum = ∑ i ∈ range n, f i
  | 0 => by simp
  | n + 1 => by
    rw [List.range_succ, List.map_append, List.sum_append, sum_map_range f n, sum_range_succ]
    simp

theorem card_blocks (t m : ℕ) : ∀ B, ∑ b ∈ range B, (Ico (b * t) (min ((b + 1) * t) m)).card = min (B * t) m
  | 0 => by simp
  | B + 1 => by
    rw [sum_range_succ, card_blocks t m B, Nat.card_Ico]
    have h : (B + 1) * t = B * t + t := by ring
    rw [h]
    omega

theorem card_blocks_all (t m : ℕ) (ht : 0 < t) :
    ∑ b ∈ range ((m + t - 1) / t), (Ico (b * t) (min ((b + 1) * t) m)).card = m := by
  rw [card_blocks]
  have h1 := Nat.div_add_mod (m + t - 1) t
  have h2 := Nat.mod_lt (m + t - 1) ht
  rw [Nat.mul_comm] at h1
  omega

/-- One unit's tiles: their shares of `w` add up to `w`. -/
theorem unit_blocks (w : ℚ) (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
    ∑ bi ∈ range ((m + 64 - 1) / 64), ∑ bj ∈ range ((n + 64 - 1) / 64),
      w * (((Ico (bi * 64) (min ((bi + 1) * 64) m)).card * (Ico (bj * 64) (min ((bj + 1) * 64) n)).card : ℚ) /
        (m * n)) = w := by
  have e : ∀ a b : ℚ, w * (a * b / (m * n)) = w / (m * n) * (a * b) := fun a b => by ring
  simp_rw [e, ← mul_sum]
  rw [← sum_mul]
  have hR : (∑ bi ∈ range ((m + 64 - 1) / 64), ((Ico (bi * 64) (min ((bi + 1) * 64) m)).card : ℚ)) = m := by
    exact_mod_cast card_blocks_all 64 m (by norm_num)
  have hC : (∑ bj ∈ range ((n + 64 - 1) / 64), ((Ico (bj * 64) (min ((bj + 1) * 64) n)).card : ℚ)) = n := by
    exact_mod_cast card_blocks_all 64 n (by norm_num)
  rw [hR, hC]
  have hm' : (m : ℚ) ≠ 0 := by exact_mod_cast hm.ne'
  have hn' : (n : ℚ) ≠ 0 := by exact_mod_cast hn.ne'
  field_simp

/-- On a layout of nonempty units, the audit tiles' `W_ref` adds up to the layout's: each unit's words are covered
once, so the `W` of `GγSampled` is `∑_u wrefOf`. -/
theorem pearlCTiles_total {Q R S : Type} (sem : PearlCSem Q R S) (c : Costs) (ρ : ℚ) (L : UnitLayout)
    (hpos : ∀ u < L.N, 0 < (L.shape u).m ∧ 0 < (L.shape u).n) :
    tileWrefTotal (pearlCTiles sem c ρ) L = ∑ u ∈ range L.N, (wrefOf Params.pi c (L.shape u) : ℝ) := by
  let F : ℕ × ℕ × ℕ → ℝ := fun b => ((wrefOf Params.pi c (L.shape b.1) * tileShare (L.shape b.1)
    (Ico (b.2.1 * 64) (min ((b.2.1 + 1) * 64) (L.shape b.1).m))
    (Ico (b.2.2 * 64) (min ((b.2.2 + 1) * 64) (L.shape b.1).n)) : ℚ) : ℝ)
  have hdef : tileWrefTotal (pearlCTiles sem c ρ) L =
      ∑ g ∈ range (gridTiles 64 64 L).length, F ((gridTiles 64 64 L).getD g (0, 0, 0)) := rfl
  rw [hdef, sum_range_getD, gridTiles, sum_map_flatMap, sum_map_range]
  refine sum_congr rfl fun u hu => ?_
  obtain ⟨hm, hn⟩ := hpos u (mem_range.mp hu)
  rw [sum_map_flatMap, sum_map_range]
  have inner : ∀ bi, (((List.range (((L.shape u).n + 64 - 1) / 64)).map fun bj => (u, bi, bj)).map F).sum
      = ∑ bj ∈ range (((L.shape u).n + 64 - 1) / 64), F (u, bi, bj) := fun bi => by
    rw [List.map_map]
    exact sum_map_range _ _
  rw [sum_congr rfl fun bi _ => inner bi]
  rw [← unit_blocks (wrefOf Params.pi c (L.shape u)) _ _ hm hn, Rat.cast_sum]
  refine sum_congr rfl fun bi _ => ?_
  rw [Rat.cast_sum]
  rfl

/-! ## TT_OUT per tile for Pearl-C is satisfiable -/

/-- The witness cost model for a tiled protocol: `true` runs `P`'s honest reference at the tiles' credit of the
activations it is given (each at least 0), `false` commits nothing. -/
noncomputable def witnessTileCM (P : Protocol Unit Unit Unit) (TR : TileRules Unit Unit Unit) :
    CostModel Unit Unit Unit where
  Prog := Bool
  run := fun b H U act _ s => if b then honestTranscript P H s U act else fun _ => none
  cost := fun b H U act _ s => if b then
    ∑ g ∈ range (TR.tiling U.layout).NT, max (TR.credit H s U g (act ((TR.tiling U.layout).unit g))) 0 else 0
  calls := fun _ _ _ _ _ _ => 0

theorem witnessTile_sum_le (P : Protocol Unit Unit Unit) (TR : TileRules Unit Unit Unit) (U : Workload)
    (H : Unit → Unit) (s : Unit) (b : Bool) (act : ℕ → Codes) (σ : State) :
    ∑ g ∈ goodTiles P TR U H s ((witnessTileCM P TR).run b H U act σ s),
        TR.credit H s U g (actOf ((witnessTileCM P TR).run b H U act σ s) ((TR.tiling U.layout).unit g))
      ≤ (witnessTileCM P TR).cost b H U act σ s := by
  classical
  cases b with
  | false =>
    have hempty : goodTiles P TR U H s ((witnessTileCM P TR).run false H U act σ s) = ∅ := by
      unfold goodTiles
      apply filter_false_of_mem
      intro g _ h
      obtain ⟨c, hc, _⟩ := h
      simp [witnessTileCM] at hc
    rw [hempty, sum_empty]
    simp [witnessTileCM]
  | true =>
    have hsub : goodTiles P TR U H s ((witnessTileCM P TR).run true H U act σ s) ⊆ range (TR.tiling U.layout).NT := by
      unfold goodTiles
      exact filter_subset _ _
    have heq : ∀ g ∈ goodTiles P TR U H s ((witnessTileCM P TR).run true H U act σ s),
        TR.credit H s U g (actOf ((witnessTileCM P TR).run true H U act σ s) ((TR.tiling U.layout).unit g)) =
          TR.credit H s U g (act ((TR.tiling U.layout).unit g)) := by
      intro g hg
      unfold goodTiles at hg
      rw [mem_filter] at hg
      obtain ⟨_, c, hc, _⟩ := hg
      have hu : (TR.tiling U.layout).unit g < U.layout.N := by
        by_contra h
        simp [witnessTileCM, honestTranscript, h] at hc
      simp only [witnessTileCM, ↓reduceIte]
      rw [actOf_honest P H s U act hu]
    rw [sum_congr rfl heq]
    calc ∑ g ∈ goodTiles P TR U H s ((witnessTileCM P TR).run true H U act σ s),
          TR.credit H s U g (act ((TR.tiling U.layout).unit g))
        ≤ ∑ g ∈ goodTiles P TR U H s ((witnessTileCM P TR).run true H U act σ s),
            max (TR.credit H s U g (act ((TR.tiling U.layout).unit g))) 0 :=
          sum_le_sum fun g _ => le_max_left _ _
      _ ≤ ∑ g ∈ range (TR.tiling U.layout).NT, max (TR.credit H s U g (act ((TR.tiling U.layout).unit g))) 0 :=
          sum_le_sum_of_subset_of_nonneg hsub (fun g _ _ => le_max_right _ _)
      _ = (witnessTileCM P TR).cost true H U act σ s := by simp [witnessTileCM]

theorem tileShare_nonneg (sh : Shape) (rows cols : Finset ℕ) : 0 ≤ tileShare sh rows cols := by
  unfold tileShare
  positivity

/-- The witness layout's one tile: unit 0, row 0, column 0. -/
theorem witness_gridTiles : gridTiles 64 64 witnessWorkload.layout = [(0, 0, 0)] := by decide

open Classical in
/-- The witness tile's debit is zero: no atom of its word is flagged, no element is salt-dead, and its word needs every
element. -/
theorem witness_tileDebit_zero (H : Unit → Unit) (s : Unit) :
    witnessSem.tileDebit Costs.adopted H s witnessWorkload 0 (Ico 0 1) (Ico 0 1) (witnessAct 0) = 0 := by
  unfold PearlCSem.tileDebit
  dsimp only
  have ha : ∑ e ∈ (univ : Finset (Fin (witnessWorkload.layout.shape 0).m × Fin (witnessWorkload.layout.shape 0).n)).filter
      (fun e => (e.1 : ℕ) ∈ Ico 0 1 ∧ (e.2 : ℕ) ∈ Ico 0 1),
      ((range ((witnessWorkload.layout.shape 0).k / 32)).filter fun t =>
        Skip.unitFlags (witnessSem.unitChain H s witnessWorkload 0 (witnessAct 0)).1
          (witnessSem.unitChain H s witnessWorkload 0 (witnessAct 0)).2 e.1 e.2 t = true).card = 0 := by
    apply sum_eq_zero
    intro e _
    rw [card_eq_zero, filter_eq_empty_iff]
    intro t ht
    have ht' : t < 4 := by simpa [witnessWorkload] using mem_range.mp ht
    rw [witness_unitFlags H s e.1 e.2 t ht']
    simp
  have he : ((univ : Finset (Fin (witnessWorkload.layout.shape 0).m × Fin (witnessWorkload.layout.shape 0).k)).filter
      fun e => (e.1 : ℕ) ∈ Ico 0 1 ∧
        (SaltDead.saltDead witnessSem.alpha witnessSem.beta (witnessWorkload.layout.shape 0).k
            (wordsVal (witnessWorkload.layout.shape 0).m (witnessWorkload.layout.shape 0).k (witnessAct 0) e.1) e.2 ∨
          ∀ j : Fin (witnessWorkload.layout.shape 0).n, (j : ℕ) ∈ Ico 0 1 →
            Skip.unitFlags (witnessSem.unitChain H s witnessWorkload 0 (witnessAct 0)).1
              (witnessSem.unitChain H s witnessWorkload 0 (witnessAct 0)).2 e.1 j ((e.2 : ℕ) / 32) = true)).card = 0 := by
    rw [card_eq_zero, filter_eq_empty_iff]
    rintro e _ ⟨_, hd | hall⟩
    · exact witness_saltLive _ _ _ (by simp [wordsVal, witnessAct, wordQ_one]) hd
    have hl : (e.2 : ℕ) / 32 < 4 := by
      have := e.2.isLt
      simp only [witnessWorkload] at this
      omega
    have h0 := hall ⟨0, by simp [witnessWorkload]⟩ (by simp)
    rw [witness_unitFlags H s e.1 _ _ hl] at h0
    exact Bool.false_ne_true h0
  rw [ha, he]
  simp [debitOf]

theorem PearlCSem.rowOK_of_operandOK {Q R S : Type} (sem : PearlCSem Q R S) {p q : ℕ} {W : Codes}
    (h : sem.OperandOK p q W) {i : ℕ} (hi : i < p) : sem.RowOK p q W i :=
  ⟨fun l hl => h.1 i hi l hl, fun hi' => h.2 ⟨i, hi'⟩⟩

/-- The witness tile's one row passes Π's rules, so it is credited. -/
theorem witness_passRows :
    witnessSem.passRows (witnessWorkload.layout.shape 0) (witnessAct 0) (Ico 0 1) = Ico 0 1 := by
  classical
  unfold PearlCSem.passRows
  apply filter_true_of_mem
  intro i hi
  have hi' : i < (witnessWorkload.layout.shape 0).m := by simpa [witnessWorkload] using hi
  exact witnessSem.rowOK_of_operandOK (witness_operandOK 1) hi'

theorem ttOutTilePearlCWitness :
    AdmitsRefTile (witnessTileCM (pearlCProtocol witnessSem Costs.adopted (1 / 400))
        (pearlCTiles witnessSem Costs.adopted (1 / 400)))
      (pearlCProtocol witnessSem Costs.adopted (1 / 400)) (pearlCTiles witnessSem Costs.adopted (1 / 400))
      (pearlCDomain Params.pi) (fun _ => 0) ∧
    TTOutTilePearlC (witnessTileCM (pearlCProtocol witnessSem Costs.adopted (1 / 400))
        (pearlCTiles witnessSem Costs.adopted (1 / 400))) witnessSem Costs.adopted (1 / 400) ∧
    witnessWorkload.InDomain (pearlCProtocol witnessSem Costs.adopted (1 / 400)) (pearlCDomain Params.pi) ∧
    witnessWorkload.WeightsOK (pearlCProtocol witnessSem Costs.adopted (1 / 400)) ∧
    (pearlCTiles witnessSem Costs.adopted (1 / 400)).capOK (fun _ => ()) () witnessWorkload 0 (witnessAct 0) ∧
    0 < ∑ g ∈ range ((pearlCTiles witnessSem Costs.adopted (1 / 400)).tiling witnessWorkload.layout).NT,
      (pearlCTiles witnessSem Costs.adopted (1 / 400)).credit (fun _ => ()) () witnessWorkload g (witnessAct 0) := by
  set P := pearlCProtocol witnessSem Costs.adopted (1 / 400)
  set TR := pearlCTiles witnessSem Costs.adopted (1 / 400)
  have hNT : (TR.tiling witnessWorkload.layout).NT = 1 := by
    show (gridTiles 64 64 witnessWorkload.layout).length = 1
    rw [witness_gridTiles]; rfl
  refine ⟨?_, ?_, ttOutPearlCWitness.2.2.1, ttOutPearlCWitness.2.2.2.1, ?_, ?_⟩
  · refine ⟨true, fun U _ _ act _ H s => ⟨by simp [witnessTileCM], ?_, by simp [witnessTileCM]⟩⟩
    show (if true = true then ∑ g ∈ range (TR.tiling U.layout).NT,
        max (TR.credit H s U g (act ((TR.tiling U.layout).unit g))) 0 else 0) ≤ tileWrefTotal TR U.layout
    simp only [↓reduceIte, tileWrefTotal]
    apply sum_le_sum
    intro g _
    simp only [TR, pearlCTiles]
    set sh := U.layout.shape ((auditTiling 64 64 U.layout).unit g)
    set rows := (auditTiling 64 64 U.layout).rows g
    set cols := (auditTiling 64 64 U.layout).cols g
    set pr := witnessSem.passRows sh (act ((auditTiling 64 64 U.layout).unit g)) rows
    have h1 := creditOf_le_wref Costs.adopted sh
    have h2 := creditOf_nonneg Costs.adopted sh
    have h3 := tileShare_nonneg sh pr cols
    have h4 := tileShare_mono sh (passRows_subset witnessSem sh (act ((auditTiling 64 64 U.layout).unit g)) rows) cols
    have h5 := wrefOf_nonneg Costs.adopted sh
    have key : (1 - 1 / 400) * creditOf Params.pi Costs.adopted sh * tileShare sh pr cols ≤
        wrefOf Params.pi Costs.adopted sh * tileShare sh rows cols :=
      calc (1 - 1 / 400) * creditOf Params.pi Costs.adopted sh * tileShare sh pr cols
          ≤ wrefOf Params.pi Costs.adopted sh * tileShare sh pr cols :=
            mul_le_mul_of_nonneg_right (by linarith) h3
        _ ≤ wrefOf Params.pi Costs.adopted sh * tileShare sh rows cols := mul_le_mul_of_nonneg_left h4 h5
    apply max_le
    · exact_mod_cast key
    · exact_mod_cast mul_nonneg h5 (tileShare_nonneg sh rows cols)
  · refine ⟨fun prog U act σ s => ⟨QTree.done ((witnessTileCM P TR).run prog (fun _ => ()) U act σ s),
      fun H => ?_⟩, ?_⟩
    · have hH : H = fun _ => () := funext fun _ => Subsingleton.elim _ _
      subst hH
      exact ⟨rfl, rfl⟩
    · intro U _ T hT q _ A hA
      rw [Pouw.pr_eq_zero]
      · unfold εPearlC
        positivity
      intro x hx
      have hsum := witnessTile_sum_le P TR U x.1 x.2 A.online A.act (A.pre.eval x.1)
      have hcost := (hA x.1 x.2).1
      have hT' : T ≤ T / (1 - 1 / 400) := by
        rw [le_div_iff₀ (by norm_num)]
        nlinarith
      unfold Adversary.transcript at hx
      linarith
  · have hu : (TR.tiling witnessWorkload.layout).unit 0 = 0 := by
      show ((gridTiles 64 64 witnessWorkload.layout).getD 0 (0, 0, 0)).1 = 0
      rw [witness_gridTiles]; rfl
    have hr : (TR.tiling witnessWorkload.layout).rows 0 = Ico 0 1 := by
      show Ico (((gridTiles 64 64 witnessWorkload.layout).getD 0 (0, 0, 0)).2.1 * 64)
        (min ((((gridTiles 64 64 witnessWorkload.layout).getD 0 (0, 0, 0)).2.1 + 1) * 64)
          (witnessWorkload.layout.shape ((gridTiles 64 64 witnessWorkload.layout).getD 0 (0, 0, 0)).1).m) = Ico 0 1
      rw [witness_gridTiles]; rfl
    have hc : (TR.tiling witnessWorkload.layout).cols 0 = Ico 0 1 := by
      show Ico (((gridTiles 64 64 witnessWorkload.layout).getD 0 (0, 0, 0)).2.2 * 64)
        (min ((((gridTiles 64 64 witnessWorkload.layout).getD 0 (0, 0, 0)).2.2 + 1) * 64)
          (witnessWorkload.layout.shape ((gridTiles 64 64 witnessWorkload.layout).getD 0 (0, 0, 0)).1).n) = Ico 0 1
      rw [witness_gridTiles]; rfl
    show witnessSem.tileDebit Costs.adopted (fun _ => ()) () witnessWorkload ((TR.tiling witnessWorkload.layout).unit 0)
        (witnessSem.passRows (witnessWorkload.layout.shape ((TR.tiling witnessWorkload.layout).unit 0)) (witnessAct 0)
          ((TR.tiling witnessWorkload.layout).rows 0)) ((TR.tiling witnessWorkload.layout).cols 0) (witnessAct 0) ≤
      1 / 400 * creditOf Params.pi Costs.adopted (witnessWorkload.layout.shape ((TR.tiling witnessWorkload.layout).unit 0)) *
        tileShare (witnessWorkload.layout.shape ((TR.tiling witnessWorkload.layout).unit 0))
          (witnessSem.passRows (witnessWorkload.layout.shape ((TR.tiling witnessWorkload.layout).unit 0)) (witnessAct 0)
            ((TR.tiling witnessWorkload.layout).rows 0)) ((TR.tiling witnessWorkload.layout).cols 0)
    rw [hu, hr, hc, witness_passRows, witness_tileDebit_zero]
    have h2 := creditOf_nonneg Costs.adopted (witnessWorkload.layout.shape 0)
    have h3 := tileShare_nonneg (witnessWorkload.layout.shape 0) (Ico 0 1) (Ico 0 1)
    positivity
  · rw [hNT, sum_range_one]
    have hu : (TR.tiling witnessWorkload.layout).unit 0 = 0 := by
      show ((gridTiles 64 64 witnessWorkload.layout).getD 0 (0, 0, 0)).1 = 0
      rw [witness_gridTiles]; rfl
    simp only [TR, pearlCTiles]
    show (0 : ℝ) < (((1 - 1 / 400) * creditOf Params.pi Costs.adopted
        (witnessWorkload.layout.shape ((auditTiling 64 64 witnessWorkload.layout).unit 0)) *
        tileShare (witnessWorkload.layout.shape ((auditTiling 64 64 witnessWorkload.layout).unit 0))
          (witnessSem.passRows (witnessWorkload.layout.shape ((auditTiling 64 64 witnessWorkload.layout).unit 0))
            (witnessAct 0) ((auditTiling 64 64 witnessWorkload.layout).rows 0))
          ((auditTiling 64 64 witnessWorkload.layout).cols 0) : ℚ) : ℝ)
    have hr : (auditTiling 64 64 witnessWorkload.layout).rows 0 = Ico 0 1 := by
      show Ico (((gridTiles 64 64 witnessWorkload.layout).getD 0 (0, 0, 0)).2.1 * 64)
        (min ((((gridTiles 64 64 witnessWorkload.layout).getD 0 (0, 0, 0)).2.1 + 1) * 64)
          (witnessWorkload.layout.shape ((gridTiles 64 64 witnessWorkload.layout).getD 0 (0, 0, 0)).1).m) = Ico 0 1
      rw [witness_gridTiles]; rfl
    have hc : (auditTiling 64 64 witnessWorkload.layout).cols 0 = Ico 0 1 := by
      show Ico (((gridTiles 64 64 witnessWorkload.layout).getD 0 (0, 0, 0)).2.2 * 64)
        (min ((((gridTiles 64 64 witnessWorkload.layout).getD 0 (0, 0, 0)).2.2 + 1) * 64)
          (witnessWorkload.layout.shape ((gridTiles 64 64 witnessWorkload.layout).getD 0 (0, 0, 0)).1).n) = Ico 0 1
      rw [witness_gridTiles]; rfl
    rw [show (auditTiling 64 64 witnessWorkload.layout).unit 0 = 0 from hu, hr, hc, witness_passRows]
    norm_num [creditOf, tileShare, Params.pi, Costs.adopted, witnessWorkload]

end Pouw.PearlC
