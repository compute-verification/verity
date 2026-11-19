module

public import Proofs.Flock.SamplingGeneric.LevelsKS
public import Proofs.Flock.SamplingGeneric.Two
public import Proofs.Core.Levels

/-!
# The compiler at levels is knowledge sound: two levels

`CompilerKSLev` on the statements whose levels are two (`CompiledLev.TwoLevel`), on `CompiledTwo`'s waits at the
compiled verifier's two-stage view (`twoView`): the outer wait unions over the sets of proof blocks the levels miss
with probability at most `δ` (`cands`), each standing for the sets the last level misses no more often in any
first-level block (`aboveLev`); each kept outer run's inner wait unions over those sets on the proof blocks inside the
blocks the first level sampled.

* `Ex_toReal_avg`, `Ex_prod_parts_le_avg`, `Ex_prod_parts_le`: **the first level's average**. Over coins whose
  first-level sample misses each set of first-level blocks no more often than the first level's design, a product of
  factors `≤ 1` over the sampled blocks averages at most its average over the design's draw; at the last level's
  misses, the levels' miss (`Verity.Levels.missProb_two`).
* `esc_proj_le`: a set below `Y` in every first-level block, projected on the sampled blocks' proof blocks, is missed
  by the last level no more often than `Y` is, factor by factor.
* `FailsTwoLev`, `pr_failsTwoLev_le`: **the failure event at two levels** and its probability, `κ(δ) + √(γ·Pr[B])`.
* `splitTwoLev`: **the deterministic half**. Each first-level block reads one top block's interior, so a best
  interior per top block (`exists_best`) gives a candidate `Y` above every kept outer run's incorrect set.
* `ks_two`, `ks_two_lev`: **`CompiledLev.KS` at two levels**, at `Compiled.E` and at `Elev`.
-/

@[expose] public section

open scoped ENNReal

namespace Claims.SamplingGeneric

open Claims Finset
open Claims.Sampling (AuditStatement Partition InScopeLev RLev slot topOf incorrectLev levM forkOn forkOn_forkOn'
  accList AllBad wait_bad_le slack slack_pos Pr_acc_forkOn OwnColl agree_of_passes val_assemble ιOf ιOf_eq
  acc_of_accList accList_length_eq assemble KSLinOn Bext card_image_inter_le replayOf TopIs)

namespace CompiledLev

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-! ## The first level's average -/

theorem Ex_toReal_avg {α : Type} [Fintype α] [Nonempty α] (f : α → ℝ≥0∞) (hf : ∀ a, f a ≠ ⊤) :
    Ex (fun a => (f a).toReal) = (Verity.Game.avg f).toReal := by
  have hc : (Fintype.card α : ℝ≥0∞)⁻¹ ≠ ⊤ := ENNReal.inv_ne_top.2 (by exact_mod_cast Fintype.card_ne_zero)
  unfold Ex Verity.Game.avg
  rw [ENNReal.toReal_sum fun a _ => ENNReal.mul_ne_top hc (hf a), Finset.sum_div]
  refine Finset.sum_congr rfl fun a _ => ?_
  rw [ENNReal.toReal_mul, ENNReal.toReal_inv, ENNReal.toReal_natCast]
  ring

/-- **The first level's average, at any factors**: over coins `r` whose sample `S r` of first-level blocks misses
each set of them no more often than the first level's design, the product over the sampled blocks of factors `e ≤ 1`
averages at most its average over the design's draw. -/
theorem Ex_prod_parts_le_avg {R G : Type} [Fintype R] [Nonempty R] [Fintype G] [DecidableEq G] (l₁ : Verity.Level G)
    (hd : ∀ ω, (l₁.design Finset.univ).draw ω ⊆ l₁.blocks.parts) (S : R → Finset (Finset G))
    (hS : ∀ T, Pr (fun r => Disjoint (S r) T) ≤ ((l₁.design Finset.univ).missProb T).toReal)
    (e : Finset G → ℝ≥0∞) (he1 : ∀ Z, e Z ≤ 1) :
    Ex (fun r => ∏ Z ∈ (S r).filter (· ∈ l₁.blocks.parts), (e Z).toReal) ≤
      (Verity.Game.avg fun ω => ∏ Z ∈ (l₁.design Finset.univ).draw ω, e Z).toReal := by
  classical
  set Ap := l₁.blocks.parts
  have hfin : ∀ Z, e Z ≠ ⊤ := fun Z => ne_top_of_le_ne_top ENNReal.one_ne_top (he1 Z)
  let D' : Verity.SamplingDesign (Finset G) := { Ω := R, draw := fun r => (S r).filter (· ∈ Ap) }
  have hw : ∀ R' ∈ Ap.powerset, ((∏ u ∈ R', (1 - e u)) * ∏ u ∈ Ap \ R', e u) ≠ ⊤ := fun R' _ =>
    ENNReal.mul_ne_top (ENNReal.prod_ne_top fun _ _ => ENNReal.sub_ne_top ENNReal.one_ne_top)
      (ENNReal.prod_ne_top fun u _ => hfin u)
  have hmiss : ∀ R' ∈ Ap.powerset, D'.missProb R' ≤ (l₁.design Finset.univ).missProb R' := by
    intro R' hR'
    have hR := Finset.mem_powerset.1 hR'
    have hdisj : ∀ r, Disjoint ((S r).filter (· ∈ Ap)) R' ↔ Disjoint (S r) R' := fun r => by
      constructor
      · intro h
        rw [Finset.disjoint_left] at h ⊢
        intro Z hZ hZR
        exact h (Finset.mem_filter.2 ⟨hZ, hR hZR⟩) hZR
      · exact fun h => Finset.disjoint_of_subset_left (Finset.filter_subset _ _) h
    have hp : D'.missProb R' = Verity.Game.prCoin (fun r : R => Disjoint (S r) R') := by
      unfold Verity.SamplingDesign.missProb
      exact congrArg Verity.Game.prCoin (funext fun r => propext (hdisj r))
    have hfin' : Verity.Game.prCoin (fun r : R => Disjoint (S r) R') ≠ ⊤ := by
      rw [← hp]
      exact ne_top_of_le_ne_top ENNReal.one_ne_top (Verity.SamplingDesign.missProb_le_one _ _)
    rw [hp, ← ENNReal.ofReal_toReal hfin', ← Claims.Sampling.Pr_eq_toReal_prCoin,
      ← ENNReal.ofReal_toReal (ne_top_of_le_ne_top ENNReal.one_ne_top
        (Verity.SamplingDesign.missProb_le_one (l₁.design Finset.univ) R'))]
    exact ENNReal.ofReal_le_ofReal (hS R')
  calc Ex (fun r => ∏ Z ∈ (S r).filter (· ∈ Ap), (e Z).toReal)
      = Ex (fun r : R => (∏ Z ∈ D'.draw r, e Z).toReal) := by
        congr 1
        funext r
        rw [ENNReal.toReal_prod]
    _ = (Verity.Game.avg fun r : R => ∏ Z ∈ D'.draw r, e Z).toReal :=
        Ex_toReal_avg _ fun r => ENNReal.prod_ne_top fun Z _ => hfin Z
    _ = (∑ R' ∈ Ap.powerset, D'.missProb R' * ((∏ u ∈ R', (1 - e u)) * ∏ u ∈ Ap \ R', e u)).toReal := by
        rw [Verity.SamplingDesign.avg_prod_draw D' Ap
          (fun r x hx => (Finset.mem_filter.1 hx).2) e fun u _ => he1 u]
    _ ≤ (∑ R' ∈ Ap.powerset,
          (l₁.design Finset.univ).missProb R' * ((∏ u ∈ R', (1 - e u)) * ∏ u ∈ Ap \ R', e u)).toReal := by
        refine ENNReal.toReal_mono (ENNReal.sum_ne_top.2 fun R' hR' => ENNReal.mul_ne_top
          (ne_top_of_le_ne_top ENNReal.one_ne_top (Verity.SamplingDesign.missProb_le_one _ _)) (hw R' hR')) ?_
        exact Finset.sum_le_sum fun R' hR' => mul_le_mul_left (hmiss R' hR') _
    _ = (Verity.Game.avg fun ω => ∏ Z ∈ (l₁.design Finset.univ).draw ω, e Z).toReal := by
        rw [Verity.SamplingDesign.avg_prod_draw (l₁.design Finset.univ) Ap hd e fun u _ => he1 u]

/-- **The first level's average**: over coins `r` whose sample `S r` of first-level blocks misses each set of them no
more often than the first level's design, the product over the sampled blocks of the last level's miss of `T`
averages at most the levels' miss of `T`. -/
theorem Ex_prod_parts_le {R G : Type} [Fintype R] [Nonempty R] [Fintype G] [DecidableEq G] (Lv : Verity.Levels G)
    {l₁ l₂ : Verity.Level G} (hL : Lv.levels = [l₁, l₂]) (S : R → Finset (Finset G))
    (hS : ∀ T, Pr (fun r => Disjoint (S r) T) ≤ ((l₁.design Finset.univ).missProb T).toReal)
    (T : Finset (Finset G)) :
    Ex (fun r => ∏ Z ∈ (S r).filter (· ∈ l₁.blocks.parts), ((l₂.design Z).missProb T).toReal) ≤
      (Lv.design.missProb T).toReal := by
  classical
  have hi := Lv.inside
  rw [hL] at hi
  simp only [Verity.Level.InsideBelow, Finset.mem_singleton, forall_eq, and_true] at hi
  obtain ⟨h₁, h₂⟩ := hi
  let e : Finset G → ℝ≥0∞ := fun Z => (l₂.design Z).missProb (T.filter (· ⊆ Z))
  have heq : ∀ Z ∈ l₁.blocks.parts, (l₂.design Z).missProb T = e Z := fun Z hZ =>
    (Verity.SamplingDesign.missProb_filter (l₂.design Z) (· ⊆ Z) (fun ω b hb => (h₂ Z hZ ω b hb).2) T).symm
  calc Ex (fun r => ∏ Z ∈ (S r).filter (· ∈ l₁.blocks.parts), ((l₂.design Z).missProb T).toReal)
      = Ex (fun r => ∏ Z ∈ (S r).filter (· ∈ l₁.blocks.parts), (e Z).toReal) := by
        congr 1
        funext r
        exact Finset.prod_congr rfl fun Z hZ => by rw [heq Z (Finset.mem_filter.1 hZ).2]
    _ ≤ (Verity.Game.avg fun ω => ∏ Z ∈ (l₁.design Finset.univ).draw ω, e Z).toReal :=
        Ex_prod_parts_le_avg l₁ (fun ω b hb => (h₁ ω b hb).1) S hS e fun _ =>
          Verity.SamplingDesign.missProb_le_one _ _
    _ = (Lv.design.missProb T).toReal := by rw [Lv.missProb_two hL]

/-! ## The failure event -/

/-- The proof blocks inside a set of first-level blocks. -/
def blocksOf (Pt : Partition) (U : Finset (Finset (Fin Pt.blocks.length))) : Finset ℕ :=
  (U.biUnion id).image Fin.val

/-- **The candidate sets at levels**: sets of proof blocks the levels miss with probability at most `δ`. -/
noncomputable def cands (Pt : Partition) (Lv : Verity.Levels (Fin Pt.blocks.length)) (δ : ℝ) : Finset (Finset ℕ) :=
  (range Pt.blocks.length).powerset.filter fun B => missLev Pt Lv B ≤ δ

open Classical in
/-- The sets of proof blocks the last level misses no more often than `Y` in every first-level block. -/
noncomputable def aboveLev (Pt : Partition) (l₁ l₂ : Verity.Level (Fin Pt.blocks.length)) (Y : Finset ℕ) :
    Finset (Finset ℕ) :=
  (range Pt.blocks.length).powerset.filter fun X => ∀ Z ∈ l₁.blocks.parts,
    (l₂.design Z).missProb (X.image (slot Pt)) ≤ (l₂.design Z).missProb (Y.image (slot Pt))

theorem missProb_toReal_le_one {U : Type} (D : Verity.SamplingDesign U) (T : Finset U) : (D.missProb T).toReal ≤ 1 :=
  ENNReal.toReal_le_of_le_ofReal zero_le_one (by rw [ENNReal.ofReal_one]; exact Verity.SamplingDesign.missProb_le_one _ _)

/-- **The inner wait's escape at a sample `U` of first-level blocks**: a set `X` below `Y` in every first-level block,
projected on `U`'s proof blocks, is missed by the last level at each sampled block no more often than `Y`; the factors
at sets that are no first-level block are at most `1`. -/
theorem esc_proj_le {Pt : Partition} {l₁ l₂ : Verity.Level (Fin Pt.blocks.length)}
    (hin : ∀ Z ∈ l₁.blocks.parts, ∀ ω, ∀ b ∈ (l₂.design Z).draw ω, b ∈ l₂.blocks.parts ∧ b ⊆ Z)
    (U : Finset (Finset (Fin Pt.blocks.length))) {X Y : Finset ℕ} (hX : X ⊆ range Pt.blocks.length)
    (hXY : ∀ Z ∈ l₁.blocks.parts,
      (l₂.design Z).missProb (X.image (slot Pt)) ≤ (l₂.design Z).missProb (Y.image (slot Pt))) :
    ∏ Z ∈ U, ((l₂.design Z).missProb ((X ∩ blocksOf Pt U).image (slot Pt))).toReal ≤
      ∏ Z ∈ U.filter (· ∈ l₁.blocks.parts), ((l₂.design Z).missProb (Y.image (slot Pt))).toReal := by
  classical
  rw [← prod_filter_mul_prod_filter_not U (· ∈ l₁.blocks.parts)]
  have h1 : ∏ Z ∈ U.filter (fun Z => ¬ Z ∈ l₁.blocks.parts),
      ((l₂.design Z).missProb ((X ∩ blocksOf Pt U).image (slot Pt))).toReal ≤ 1 :=
    prod_le_one₀ (fun _ _ => ENNReal.toReal_nonneg) fun Z _ => missProb_toReal_le_one _ _
  refine (mul_le_mul_of_nonneg_left h1 (prod_nonneg fun _ _ => ENNReal.toReal_nonneg)).trans ?_
  rw [mul_one]
  refine Finset.prod_le_prod₀ (fun _ _ => ENNReal.toReal_nonneg) fun Z hZ => ?_
  obtain ⟨hZU, hZp⟩ := mem_filter.1 hZ
  have hfin : ∀ T, (l₂.design Z).missProb T ≠ ⊤ := fun T =>
    ne_top_of_le_ne_top ENNReal.one_ne_top (Verity.SamplingDesign.missProb_le_one _ _)
  refine ENNReal.toReal_mono (hfin _) ?_
  have hsame : (l₂.design Z).missProb ((X ∩ blocksOf Pt U).image (slot Pt)) =
      (l₂.design Z).missProb (X.image (slot Pt)) := by
    rw [← Verity.SamplingDesign.missProb_filter (l₂.design Z) (· ⊆ Z) (fun ω b hb => (hin Z hZp ω b hb).2)
        ((X ∩ blocksOf Pt U).image (slot Pt)),
      ← Verity.SamplingDesign.missProb_filter (l₂.design Z) (· ⊆ Z) (fun ω b hb => (hin Z hZp ω b hb).2)
        (X.image (slot Pt))]
    congr 1
    ext b
    simp only [mem_filter, mem_image, mem_inter]
    constructor
    · rintro ⟨⟨v, ⟨hvX, -⟩, rfl⟩, hsub⟩
      exact ⟨⟨v, hvX, rfl⟩, hsub⟩
    · rintro ⟨⟨v, hvX, rfl⟩, hsub⟩
      have hv : v < Pt.blocks.length := mem_range.1 (hX hvX)
      have hmem : (⟨v, hv⟩ : Fin Pt.blocks.length) ∈ slot Pt v := by simp [slot]
      exact ⟨⟨v, ⟨hvX, mem_image.2 ⟨⟨v, hv⟩, mem_biUnion.2 ⟨Z, hZU, hsub hmem⟩, rfl⟩⟩, rfl⟩, hsub⟩
  rw [hsame]
  exact hXY Z hZp

/-- **Two levels at `x`**: the levels of every statement `x` parses to are two. -/
def TwoLevel (L : CompiledLev Coin A) (x : Msg) : Prop :=
  ∀ s Pt Lv, L.parse x = some s → L.partition s = some Pt → L.levels s Pt = some Lv →
    ∃ l₁ l₂, Lv.levels = [l₁, l₂]

/-- **The extraction's failure event at two levels**: the statement is in scope with two levels, and the outer wait
comes up short or, for some candidate `Y`, every kept outer run is bad for the sets below `Y`
(`CompiledTwo.BadOuterTwo` at `twoView`). -/
def FailsTwoLev (L : CompiledLev Coin A) (δ : ℝ) (x : Msg) (run' : (Fin (L.V.k x) → Coin) → Transcript)
    (found : (u : Fin (L.nPU x)) → (Fin (L.V.k x) → Coin) → A.EC (L.sh x u) → Prop) (c : L.ECoins x) : Prop :=
  ∃ s Pt Lv l₁ l₂, L.parse x = some s ∧ L.partition s = some Pt ∧ L.levels s Pt = some Lv ∧
    InScopeLev Pt Lv (L.top s) s ∧ Lv.levels = [l₁, l₂] ∧
    ((accList (fun r => (run' r).out = .acc) (L.kOut x) ((List.finRange (L.capOut x)).map c.1)).length < L.kOut x ∨
      ∃ Y ∈ cands Pt Lv δ, AllBad (fun r => (run' r).out = .acc) (L.kOut x)
        (fun r e => L.twoView.BadOuterTwo x run' found (aboveLev Pt l₁ l₂ Y) (forkOn (L.secondStage x) r) e)
        ((List.finRange (L.capOut x)).map c.1) (L.toCompiled.toE1 x c))

/-! ## The failure event's probability -/

set_option synthInstance.maxSize 1024 in
omit [Inhabited Coin] in
/-- **One tape's outer wait at two levels**: `CompiledTwo.outer_badTwo_le`'s bound, its first stage the first level's
sample (`Samples`' second clause, averaged by `Ex_prod_parts_le`) and its second the last level's (`Samples`' last
clause, projected by `esc_proj_le`); the outer wait unions over at most `2^{n_PU}` sets. -/
theorem outer_badTwoLev_le (L : CompiledLev Coin A) {δ : ℝ} (hδ : 0 ≤ δ) (x : Msg)
    (run' : (Fin (L.V.k x) → Coin) → Transcript)
    (found : (u : Fin (L.nPU x)) → (Fin (L.V.k x) → Coin) → A.EC (L.sh x u) → Prop)
    (Pt : Partition) (Lv : Verity.Levels (Fin Pt.blocks.length)) {l₁ l₂ : Verity.Level (Fin Pt.blocks.length)}
    (hL : Lv.levels = [l₁, l₂]) (hPU : L.nPU x = Pt.blocks.length)
    (hun : L.nPU x ≤ L.nOut x ∧ ∀ r e, (run' (forkOn (L.secondStage x) r e)).out = .acc →
      (((L.upper x Pt r).getLastD ∅).biUnion id).card ≤ L.nIn x)
    (κr : (Fin (L.V.k x) → Coin) → ℝ) (hκr0 : ∀ r, 0 ≤ κr r)
    (hfirst : ∀ T, Pr (fun r : Fin (L.V.k x) → Coin => Disjoint ((L.upper x Pt r).headD ∅) T) ≤
      ((l₁.design Finset.univ).missProb T).toReal)
    (hlast : ∀ (r : Fin (L.V.k x) → Coin) (X : Finset ℕ),
      Pr (fun e : Fin (L.V.k x) → Coin => ∀ u ∈ L.checked x (forkOn (L.secondStage x) r e), u ∉ X) ≤
        ∏ Y ∈ (L.upper x Pt r).getLastD ∅, ((l₂.design Y).missProb (X.image (slot Pt))).toReal)
    (hhead : ∀ r, (L.upper x Pt r).getLastD ∅ = (L.upper x Pt r).headD ∅)
    (hslots : ∀ r', (L.checked x r').length ≤ L.nPU x)
    (hsess : ∀ r (sel : List ℕ → Option (Fin (L.nPU x))),
      Pr (fun p : (Fin (L.V.k x) → Coin) × L.SlotEC x =>
        (run' (forkOn (L.secondStage x) r p.1)).out = .acc ∧
          ∃ i, sel (L.checked x (forkOn (L.secondStage x) r p.1)) = some i ∧
            ∃ u : Fin (L.nPU x), (L.checked x (forkOn (L.secondStage x) r p.1))[i.val]? = some u.val ∧
              ¬ found u (forkOn (L.secondStage x) r p.1) (p.2 i u)) ≤
        κr r) :
    Pr (fun r : Fin (L.V.k x) → Coin => (run' r).out = .acc) *
        Pr (fun c : (Fin (L.capOut x) → Fin (L.V.k x) → Coin) × (Fin (L.kOut x) → L.E1 x) =>
          (accList (fun r => (run' r).out = .acc) (L.kOut x) ((List.finRange (L.capOut x)).map c.1)).length <
              L.kOut x ∨
            ∃ Y ∈ cands Pt Lv δ, AllBad (fun r => (run' r).out = .acc) (L.kOut x)
              (fun r e => L.twoView.BadOuterTwo x run' found (aboveLev Pt l₁ l₂ Y) (forkOn (L.secondStage x) r) e)
              ((List.finRange (L.capOut x)).map c.1) c.2) ≤
      2 * (L.kOut x : ℝ) / L.capOut x + Real.exp (-(L.kOut x : ℝ) / 8) +
        slack * (2 * (L.kIn x : ℝ) / L.capIn x + Real.exp (-(L.kIn x : ℝ) / 8) + slack * (δ + Ex κr)) := by
  classical
  have hc := slack_pos
  have hEκ : 0 ≤ Ex κr := Ex_nonneg hκr0
  have hin : ∀ Z ∈ l₁.blocks.parts, ∀ ω, ∀ b ∈ (l₂.design Z).draw ω, b ∈ l₂.blocks.parts ∧ b ⊆ Z := by
    have hi := Lv.inside
    rw [hL] at hi
    simp only [Verity.Level.InsideBelow, Finset.mem_singleton, forall_eq, and_true] at hi
    exact hi.2
  have hy : ∀ Y ∈ cands Pt Lv δ,
      Pr (fun p : (Fin (L.V.k x) → Coin) × L.E1 x =>
        (run' p.1).out = .acc ∧
          L.twoView.BadOuterTwo x run' found (aboveLev Pt l₁ l₂ Y) (forkOn (L.secondStage x) p.1) p.2) ≤
      2 * (L.kIn x : ℝ) / L.capIn x + Real.exp (-(L.kIn x : ℝ) / 8) + slack * (δ + Ex κr) := by
    intro Y hY
    obtain ⟨_, hYδ⟩ := mem_filter.1 hY
    let esc : (Fin (L.V.k x) → Coin) → ℝ := fun r =>
      ∏ Z ∈ ((L.upper x Pt r).headD ∅).filter (· ∈ l₁.blocks.parts),
        ((l₂.design Z).missProb (Y.image (slot Pt))).toReal
    have hesc0 : ∀ r, 0 ≤ esc r := fun r => prod_nonneg fun _ _ => ENNReal.toReal_nonneg
    have hesc : ∀ r, ∀ X ∈ (aboveLev Pt l₁ l₂ Y).image (· ∩ blocksOf Pt ((L.upper x Pt r).getLastD ∅)),
        Pr (fun e : Fin (L.V.k x) → Coin => ∀ u ∈ L.checked x (forkOn (L.secondStage x) r e), u ∉ X) ≤ esc r := by
      intro r X' hX'
      obtain ⟨X, hX, rfl⟩ := mem_image.1 hX'
      obtain ⟨hXr, hXY⟩ := mem_filter.1 hX
      refine (hlast r _).trans ?_
      have h := esc_proj_le hin ((L.upper x Pt r).getLastD ∅) (mem_powerset.1 hXr) hXY
      rw [hhead r] at h ⊢
      exact h
    have hXs : ∀ r, ∀ X ∈ (aboveLev Pt l₁ l₂ Y).image (· ∩ blocksOf Pt ((L.upper x Pt r).getLastD ∅)),
        ∀ v ∈ X, v < L.nPU x := by
      intro r X' hX' v hv
      obtain ⟨X, hX, rfl⟩ := mem_image.1 hX'
      rw [hPU]
      exact mem_range.1 (mem_powerset.1 (mem_filter.1 hX).1 (mem_inter.1 hv).1)
    have hEesc : Ex esc ≤ δ :=
      (Ex_prod_parts_le Lv hL (fun r => (L.upper x Pt r).headD ∅) hfirst (Y.image (slot Pt))).trans hYδ
    refine (Pr_acc_forkOn (L.secondStage x) (fun r => (run' r).out = .acc)
      (fun r e => L.twoView.BadOuterTwo x run' found (aboveLev Pt l₁ l₂ Y) (forkOn (L.secondStage x) r) e)
      fun r e' e => Iff.of_eq (congrArg (fun g => L.twoView.BadOuterTwo x run' found (aboveLev Pt l₁ l₂ Y) g e)
        (forkOn_forkOn' (L.secondStage x) r e'))).trans_le ?_
    calc Ex (fun r => Pr (fun e' => (run' (forkOn (L.secondStage x) r e')).out = .acc) *
          Pr (fun e : L.E1 x =>
            L.twoView.BadOuterTwo x run' found (aboveLev Pt l₁ l₂ Y) (forkOn (L.secondStage x) r) e))
        ≤ Ex (fun r => Pr (fun e' => (run' (forkOn (L.secondStage x) r e')).out = .acc) *
          Pr (fun e : L.E1 x => L.twoView.BadOuterTwo x run' found
            ((aboveLev Pt l₁ l₂ Y).image (· ∩ blocksOf Pt ((L.upper x Pt r).getLastD ∅)))
            (forkOn (L.secondStage x) r) e)) :=
          Ex_mono fun r => mul_le_mul_of_nonneg_left (Pr_mono fun e h => L.twoView.badOuterTwo_inter _ h)
            (Pr_nonneg _)
      _ ≤ Ex (fun r => 2 * (L.kIn x : ℝ) / L.capIn x + Real.exp (-(L.kIn x : ℝ) / 8) + slack * (esc r + κr r)) :=
          Ex_mono fun r => by
            by_cases hacc : ∃ e, (run' (forkOn (L.secondStage x) r e)).out = .acc
            · obtain ⟨e, he⟩ := hacc
              exact L.twoView.inner_badTwo_le x run' found (hXs r) (card_image_inter_le _ _)
                ((card_image_le).trans (hun.2 r e he)) hslots r (hesc0 r) (hκr0 r) (hesc r) (hsess r)
            · have h0 : Pr (fun e : Fin (L.V.k x) → Coin => (run' (forkOn (L.secondStage x) r e)).out = .acc) = 0 :=
                (Pr_congr fun e => ⟨fun h => hacc ⟨e, h⟩, False.elim⟩).trans Pr_false
              rw [h0, zero_mul]
              exact add_nonneg (add_nonneg (by positivity) (Real.exp_pos _).le)
                (mul_nonneg hc.le (add_nonneg (hesc0 r) (hκr0 r)))
      _ = 2 * (L.kIn x : ℝ) / L.capIn x + Real.exp (-(L.kIn x : ℝ) / 8) + slack * (Ex esc + Ex κr) := by
          rw [Ex_add, Ex_const, Ex_mul_left, Ex_add]
      _ ≤ 2 * (L.kIn x : ℝ) / L.capIn x + Real.exp (-(L.kIn x : ℝ) / 8) + slack * (δ + Ex κr) := by
          gcongr
  have hy0 : 0 ≤ 2 * (L.kIn x : ℝ) / L.capIn x + Real.exp (-(L.kIn x : ℝ) / 8) + slack * (δ + Ex κr) := by
    positivity
  have hy' : ∀ Xs ∈ (cands Pt Lv δ).image (aboveLev Pt l₁ l₂),
      Pr (fun p : (Fin (L.V.k x) → Coin) × L.E1 x =>
        (run' p.1).out = .acc ∧ L.twoView.BadOuterTwo x run' found Xs (forkOn (L.secondStage x) p.1) p.2) ≤
      2 * (L.kIn x : ℝ) / L.capIn x + Real.exp (-(L.kIn x : ℝ) / 8) + slack * (δ + Ex κr) := by
    intro Xs hXs
    obtain ⟨Y, hY, rfl⟩ := mem_image.1 hXs
    exact hy Y hY
  have hw := wait_bad_le (fun r => (run' r).out = .acc) (L.kOut x) (L.capOut x) (L.capOut_pos x) (L.kOut_pos x)
    ((cands Pt Lv δ).image (aboveLev Pt l₁ l₂))
    (fun Xs r e => L.twoView.BadOuterTwo x run' found Xs (forkOn (L.secondStage x) r) e) hy0 hy'
  have hcard0 : ((cands Pt Lv δ).image (aboveLev Pt l₁ l₂)).card ≤ 2 ^ Pt.blocks.length :=
    card_image_le.trans ((card_le_card (filter_subset _ _)).trans (by rw [card_powerset, card_range]))
  have hcard : ((((cands Pt Lv δ).image (aboveLev Pt l₁ l₂)).card : ℕ) : ℝ) ^ (1 / (L.kOut x : ℝ)) ≤ slack :=
    L.rpow_kOut_le_slack x hcard0 (by have := hun.1; omega)
  refine (mul_le_mul_of_nonneg_left (Pr_mono fun c hev => ?_) (Pr_nonneg _)).trans
    (hw.trans (add_le_add le_rfl (mul_le_mul_of_nonneg_right hcard hy0)))
  rcases hev with h | ⟨Y, hY, h⟩
  · exact Or.inl h
  · exact Or.inr ⟨aboveLev Pt l₁ l₂ Y, mem_image_of_mem _ hY, h⟩

theorem getLastD_eq_headD {α : Type} {l : List α} {d : α} (h : l.length = 1) : l.getLastD d = l.headD d := by
  match l, h with
  | [_], _ => rfl

set_option synthInstance.maxSize 1024 in
/-- **The failure event's probability at two levels**: `Compiled.pr_le_of_tape` with, at each tape, the outer wait's
bound at two levels (`outer_badTwoLev_le`), its samples from `Samples` (one entry of `upper` by its length clause) and
its unions from `Unions`. -/
theorem pr_failsTwoLev_le (H : Msg → Msg) (L : CompiledLev Coin A) {δ : ℝ} (hδ : 0 ≤ δ) (hS : L.Sessions H)
    (x : Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω)
    (found : Ω → (u : Fin (L.nPU x)) → (Fin (L.V.k x) → Coin) → A.EC (L.sh x u) → Prop)
    (β : Ω → (Fin (L.V.k x) → Coin) → ℝ) (hβ0 : ∀ ω r, 0 ≤ β ω r)
    (hβ : Ex (fun p : Ω × (Fin (L.V.k x) → Coin) => β p.1 p.2) ≤ PrB H L.V P x)
    (hsess : ∀ ω r (sel : List ℕ → Option (Fin (L.nPU x))),
      Pr (fun p : (Fin (L.V.k x) → Coin) × L.SlotEC x =>
        (run H P L.V x ω (forkOn (L.secondStage x) r p.1)).out = .acc ∧
          ∃ i, sel (L.checked x (forkOn (L.secondStage x) r p.1)) = some i ∧
            ∃ u : Fin (L.nPU x), (L.checked x (forkOn (L.secondStage x) r p.1))[i.val]? = some u.val ∧
              ¬ found ω u (forkOn (L.secondStage x) r p.1) (p.2 i u)) ≤
        L.κs x + Real.sqrt (L.V.k x * L.γs x * β ω r)) :
    Pr (fun s : Ω × (Fin (L.V.k x) → Coin) × L.ECoins x =>
      (run H P L.V x s.1 s.2.1).out = .acc ∧ L.FailsTwoLev δ x (run H P L.V x s.1) (found s.1) s.2.2) ≤
      L.κ δ x + Real.sqrt (L.γ x * PrB H L.V P x) := by
  refine L.toCompiled.pr_le_of_tape H x Ω P (fun ω c => L.FailsTwoLev δ x (run H P L.V x ω) (found ω) c) β hβ0 hβ
    fun ω => ?_
  have hκ0 := L.toCompiled.κ_nonneg hδ x
  by_cases hsc : ∃ s Pt Lv l₁ l₂, L.parse x = some s ∧ L.partition s = some Pt ∧ L.levels s Pt = some Lv ∧
      InScopeLev Pt Lv (L.top s) s ∧ Lv.levels = [l₁, l₂]
  swap
  · have h0 : Pr (fun c : L.ECoins x => L.FailsTwoLev δ x (run H P L.V x ω) (found ω) c) = 0 :=
      (Pr_congr fun c => ⟨fun ⟨s, Pt, Lv, l₁, l₂, h1, h2, h3, h4, h5, _⟩ =>
        hsc ⟨s, Pt, Lv, l₁, l₂, h1, h2, h3, h4, h5⟩, False.elim⟩).trans Pr_false
    rw [h0, mul_zero]
    exact add_nonneg hκ0 (mul_nonneg (by positivity) (Ex_nonneg fun _ => Real.sqrt_nonneg _))
  obtain ⟨s, Pt, Lv, l₁, l₂, hs, hPt, hLv, hin, hL⟩ := hsc
  have hF : Pr (fun c : L.ECoins x => L.FailsTwoLev δ x (run H P L.V x ω) (found ω) c) =
      Pr (fun c : (Fin (L.capOut x) → Fin (L.V.k x) → Coin) × (Fin (L.kOut x) → L.toCompiled.E1 x) =>
        (accList (fun r => (run H P L.V x ω r).out = .acc) (L.kOut x)
          ((List.finRange (L.capOut x)).map c.1)).length < L.kOut x ∨
          ∃ Y ∈ cands Pt Lv δ, AllBad (fun r => (run H P L.V x ω r).out = .acc) (L.kOut x)
            (fun r e => L.twoView.BadOuterTwo x (run H P L.V x ω) (found ω) (aboveLev Pt l₁ l₂ Y)
              (forkOn (L.secondStage x) r) e)
            ((List.finRange (L.capOut x)).map c.1) c.2) := by
    refine Eq.trans ?_ (Claims.Sampling.Pr_equiv (L.toCompiled.ecoinsEquiv x) _)
    refine Pr_congr fun c => ⟨fun ⟨s', Pt', Lv', l₁', l₂', hs', hPt', hLv', _, hL', h⟩ => ?_,
      fun h => ⟨s, Pt, Lv, l₁, l₂, hs, hPt, hLv, hin, hL, h⟩⟩
    rw [hs, Option.some.injEq] at hs'
    subst hs'
    rw [hPt, Option.some.injEq] at hPt'
    subst hPt'
    rw [hLv, Option.some.injEq] at hLv'
    subst hLv'
    rw [hL, List.cons.injEq, List.cons.injEq] at hL'
    obtain ⟨rfl, rfl, -⟩ := hL'
    exact h
  rw [hF]
  have hsam := hS.samples x s Pt Lv hs hPt hLv hin
  have hun := hS.unions x s Pt Lv hs hPt hLv hin
  obtain ⟨Q, hQ⟩ : ∃ Q, L.top s = some Q := by
    cases hq : L.top s with
    | none =>
      have h := hin.2.2.2
      rw [hq] at h
      have h' : Lv.levels.length = 1 := h
      rw [hL] at h'
      simp at h'
    | some Q => exact ⟨Q, rfl⟩
  rw [hQ] at hun
  have hlen : ∀ r, (L.upper x Pt r).length = 1 := fun r => by
    have h := hsam.2.2.2.2 r
    rw [hL] at h
    simpa using h
  refine (L.outer_badTwoLev_le hδ x (run H P L.V x ω) (found ω) Pt Lv hL (Compiled.nPU_eq hs hPt)
    ⟨hun.1, hun.2 Ω P ω⟩ (fun r => L.κs x + Real.sqrt (L.V.k x * L.γs x * β ω r))
    (fun r => add_nonneg (L.toCompiled.κs_nonneg x) (Real.sqrt_nonneg _))
    (hsam.2.1 l₁ l₂ [] hL) (hsam.2.2.2.1 l₂ (by rw [hL]; rfl) (by rw [hL]; simp))
    (fun r => getLastD_eq_headD (hlen r)) (hS.slots x) (hsess ω)).trans (le_of_eq ?_)
  rw [Ex_add, Ex_const]
  unfold Compiled.κ
  ring

/-! ## The split -/

theorem mem_incorrectLev_congr {s : AuditStatement} {Pt : Partition} {top : Option Partition}
    {val : ℕ → Option Bool} {ι ι' : ℕ → ℕ → Bool} {v ρ : ℕ} (hv : topOf Pt top v = some ρ) (h : ι ρ = ι' ρ) :
    v ∈ incorrectLev s Pt top val ι ↔ v ∈ incorrectLev s Pt top val ι' := by
  classical
  unfold incorrectLev
  simp only [mem_filter]
  refine and_congr_right fun _ => not_congr ⟨fun ⟨u, ρ', r, h1, h2, h3, h4⟩ => ⟨u, ρ', r, h1, h2, h3, ?_⟩,
    fun ⟨u, ρ', r, h1, h2, h3, h4⟩ => ⟨u, ρ', r, h1, h2, h3, ?_⟩⟩
  · rw [hv, Option.some.injEq] at h2
    subst h2
    rwa [← h]
  · rw [hv, Option.some.injEq] at h2
    subst h2
    rwa [h]

theorem incorrectLev_subset (s : AuditStatement) (Pt : Partition) (top : Option Partition) (val : ℕ → Option Bool)
    (ι : ℕ → ℕ → Bool) : incorrectLev s Pt top val ι ⊆ range Pt.blocks.length := by
  classical
  unfold incorrectLev
  exact filter_subset _ _

/-- **A first-level block reads one top block's interior**: in a first-level block `Z` whose proof blocks all have top
block `ρ`, a design drawing inside `Z` misses the incorrect set alike at any two interiors that agree at `ρ`. -/
theorem missProb_incorrectLev_congr {s : AuditStatement} {Pt Q : Partition} {val : ℕ → Option Bool}
    {D : Verity.SamplingDesign (Finset (Fin Pt.blocks.length))} {Z : Finset (Fin Pt.blocks.length)}
    (hZ : ∀ ω, ∀ b ∈ D.draw ω, b ⊆ Z) {ρ : ℕ} (hZρ : ∀ i ∈ Z, replayOf Pt Q i.val = some ρ)
    (ι ι' : ℕ → ℕ → Bool) (h : ι ρ = ι' ρ) :
    D.missProb ((incorrectLev s Pt (some Q) val ι).image (slot Pt)) =
      D.missProb ((incorrectLev s Pt (some Q) val ι').image (slot Pt)) := by
  classical
  rw [← Verity.SamplingDesign.missProb_filter D (· ⊆ Z) hZ
      ((incorrectLev s Pt (some Q) val ι).image (slot Pt)),
    ← Verity.SamplingDesign.missProb_filter D (· ⊆ Z) hZ
      ((incorrectLev s Pt (some Q) val ι').image (slot Pt))]
  congr 1
  ext b
  simp only [mem_filter, mem_image]
  have key : ∀ v, slot Pt v ⊆ Z → (v ∈ incorrectLev s Pt (some Q) val ι ∨ v ∈ incorrectLev s Pt (some Q) val ι') →
      (v ∈ incorrectLev s Pt (some Q) val ι ↔ v ∈ incorrectLev s Pt (some Q) val ι') := by
    intro v hsub hv
    have hvlt : v < Pt.blocks.length := by
      rcases hv with hv | hv <;> exact mem_range.1 (incorrectLev_subset _ _ _ _ _ hv)
    have hmem : (⟨v, hvlt⟩ : Fin Pt.blocks.length) ∈ slot Pt v := by simp [slot]
    exact mem_incorrectLev_congr (hZρ _ (hsub hmem)) h
  constructor
  · rintro ⟨⟨v, hv, rfl⟩, hsub⟩
    exact ⟨⟨v, (key v hsub (Or.inl hv)).1 hv, rfl⟩, hsub⟩
  · rintro ⟨⟨v, hv, rfl⟩, hsub⟩
    exact ⟨⟨v, (key v hsub (Or.inr hv)).2 hv, rfl⟩, hsub⟩

theorem exists_best {α : Type} [Nonempty α] (f : α → Finset ℕ) (N : Finset ℕ) (hf : ∀ a, f a ⊆ N)
    (g : Finset ℕ → ℝ≥0∞) : ∃ a, ∀ a', g (f a') ≤ g (f a) := by
  classical
  obtain ⟨Z, hZ, hmax⟩ := (N.powerset.filter fun Z => ∃ a, f a = Z).exists_max_image g
    ⟨f (Classical.arbitrary α), mem_filter.2 ⟨mem_powerset.2 (hf _), _, rfl⟩⟩
  obtain ⟨a, rfl⟩ := (mem_filter.1 hZ).2
  exact ⟨a, fun a' => hmax _ (mem_filter.2 ⟨mem_powerset.2 (hf a'), a', rfl⟩)⟩

/-- **The split at two levels**: at an accepted run whose extraction fails, the failure event holds or the extractor's
own calls collide. Otherwise the kept openings assemble a witness; each first-level block's proof blocks share one top
block, so the incorrect set at a best interior per top block (`exists_best`) is a candidate `Y` (its miss is at most
`δ`), and each kept outer run's incorrect set, at the interiors the kept openings under its interior roots give, is
below `Y` in every first-level block (`missProb_incorrectLev_congr`); a kept inner run at which one of its proof
blocks is good would make it correct (`Sessions.opens` at its top block). -/
theorem splitTwoLev {H : Msg → Msg} {L : CompiledLev Coin A} (hbind : L.commit.Binds H) (hS : L.Sessions H) (δ : ℝ)
    (x : Msg) (htwo : L.TwoLevel x) {Ω : Type} (P : Prover Ω) (ω : Ω) (r₀ : Fin (L.V.k x) → Coin)
    (c : L.ECoins x) (hacc : (run H P L.V x ω r₀).out = .acc)
    (hfail : ¬ (RLev H L.commit L.parse L.partition L.levels L.top δ).ext x (L.E.eval H P x ω r₀ c).1) :
    L.FailsTwoLev δ x (run H P L.V x ω) (L.toCompiled.Found H x (run H P L.V x ω)) c ∨ OwnColl H L.E P x ω r₀ c := by
  classical
  obtain ⟨s, Pt, Lv, hs, hPt, hLv, hin⟩ := hS.refuses x Ω P ω r₀ hacc
  obtain ⟨l₁, l₂, hL⟩ := htwo s Pt Lv hs hPt hLv
  obtain ⟨Q, hQ⟩ : ∃ Q, L.top s = some Q := by
    cases hq : L.top s with
    | none =>
      have h := hin.2.2.2
      rw [hq] at h
      have h' : Lv.levels.length = 1 := h
      rw [hL] at h'
      simp at h'
    | some Q => exact ⟨Q, rfl⟩
  have hT := hin.2.2.2
  rw [hQ] at hT
  obtain ⟨-, hsome, l, rest, hl, -, hiff⟩ := hT
  rw [hL, List.cons.injEq] at hl
  obtain ⟨rfl, -⟩ := hl
  have hins : ∀ Z ∈ l₁.blocks.parts, ∀ ω, ∀ b ∈ (l₂.design Z).draw ω, b ⊆ Z := by
    have hi := Lv.inside
    rw [hL] at hi
    simp only [Verity.Level.InsideBelow, Finset.mem_singleton, forall_eq, and_true] at hi
    exact fun Z hZ ω b hb => (hi.2 Z hZ ω b hb).2
  have hρ : ∀ Z ∈ l₁.blocks.parts, ∃ ρ, ∀ i ∈ Z, replayOf Pt Q i.val = some ρ := by
    intro Z hZ
    obtain ⟨i, hi⟩ := l₁.blocks.nonempty_of_mem_parts hZ
    obtain ⟨ρ, hρ⟩ := Option.isSome_iff_exists.1 (hsome i.val i.isLt)
    exact ⟨ρ, fun j hj => ((hiff j i).2 ⟨Z, hZ, hj, hi⟩).trans hρ⟩
  by_cases hcoll : OwnColl H L.E P x ω r₀ c
  · exact Or.inr hcoll
  refine Or.inl ⟨s, Pt, Lv, l₁, l₂, hs, hPt, hLv, hin, hL, ?_⟩
  set run' := run H P L.V x ω with hrun'
  set fd := L.toCompiled.found H x s run' c (L.toCompiled.outerList H x run' c) with hfd
  have hpass : ∀ q ∈ fd, L.commit.Opens H q.1 q.2 = true ∧
      ∀ t ∈ ((L.commit.opens q.1 q.2).eval H).2, t ∈ Claims.Sampling.ownLog H run' (L.E.ext x r₀ c) := by
    intro q hq
    rw [hfd, Compiled.found_eq] at hq
    obtain ⟨h1, h2⟩ := L.toCompiled.passes_found H x run' s c _ q hq
    exact ⟨h1, fun t ht => L.toCompiled.ownLog_E H P x ω r₀ c hs hacc t (h2 t ht)⟩
  have hagree : ∀ q ∈ fd, ∀ q' ∈ fd, q.1 = q'.1 → q.2.pos = q'.2.pos → q.2.val = q'.2.val :=
    fun q hq q' hq' h1 h2 => agree_of_passes hbind hpass hcoll hq hq' h1 h2
  rw [L.toCompiled.E_eval H P x ω r₀ c hs hacc] at hfail
  set w := assemble s fd with hw
  set I : (ℕ → ℕ → Bool) → Finset ℕ := fun ι => incorrectLev s Pt (some Q) (w.val H L.commit s) ι with hI
  let m : Finset (Fin Pt.blocks.length) → Finset ℕ → ℝ≥0∞ := fun Z X => (l₂.design Z).missProb (X.image (slot Pt))
  have hbest : ∀ ρ : ℕ, ∃ ιρ : ℕ → Bool, ∀ Z ∈ l₁.blocks.parts, (∀ i ∈ Z, replayOf Pt Q i.val = some ρ) →
      ∀ ι' : ℕ → Bool, m Z (I fun _ => ι') ≤ m Z (I fun _ => ιρ) := by
    intro ρ
    by_cases hex : ∃ Z₀ ∈ l₁.blocks.parts, ∀ i ∈ Z₀, replayOf Pt Q i.val = some ρ
    · obtain ⟨Z₀, hZ₀, hZ₀ρ⟩ := hex
      obtain ⟨ιρ, hιρ⟩ := exists_best (fun ι' : ℕ → Bool => I fun _ => ι') (range Pt.blocks.length)
        (fun _ => incorrectLev_subset _ _ _ _ _) (m Z₀)
      refine ⟨ιρ, fun Z hZ hZρ ι' => ?_⟩
      obtain ⟨i, hi⟩ := l₁.blocks.nonempty_of_mem_parts hZ
      obtain ⟨j, hj⟩ := l₁.blocks.nonempty_of_mem_parts hZ₀
      obtain ⟨X, hX, hiX, hjX⟩ := (hiff i j).1 ((hZρ i hi).trans (hZ₀ρ j hj).symm)
      have h1 : Z = X := l₁.blocks.eq_of_mem_parts hZ hX hi hiX
      have h2 : Z₀ = X := l₁.blocks.eq_of_mem_parts hZ₀ hX hj hjX
      rw [h1, ← h2]
      exact hιρ ι'
    · refine ⟨fun _ => false, fun Z hZ hZρ _ => absurd ⟨Z, hZ, hZρ⟩ hex⟩
  choose ιs hιs using hbest
  have hYc : I ιs ∈ cands Pt Lv δ := by
    refine mem_filter.2 ⟨mem_powerset.2 (incorrectLev_subset _ _ _ _ _), ?_⟩
    by_contra hlt
    refine hfail ⟨s, Pt, Lv, hs, hPt, hLv, ιs, ?_⟩
    rw [hQ]
    exact lt_of_not_ge hlt
  by_cases hshort : (accList (fun r => (run' r).out = .acc) (L.kOut x)
    ((List.finRange (L.capOut x)).map c.1)).length < L.kOut x
  · exact Or.inl hshort
  refine Or.inr ⟨_, hYc, accList_length_eq hshort, fun j r hj => ?_⟩
  have hjZ := L.toCompiled.mem_outerList H x run' c hj
  by_cases hshort2 : (accList (fun d => (run' (forkOn (L.secondStage x) r d)).out = .acc) (L.kIn x)
    ((List.finRange (L.capIn x)).map (c.2.1 j))).length < L.kIn x
  · exact Or.inl hshort2
  let ιr : ℕ → ℕ → Bool := fun ρ => ιOf fd (L.iroots x (run' r).seen ρ)
  have hXr : I ιr ∈ aboveLev Pt l₁ l₂ (I ιs) := by
    refine mem_filter.2 ⟨mem_powerset.2 (incorrectLev_subset _ _ _ _ _), fun Z hZ => ?_⟩
    obtain ⟨ρ, hZρ⟩ := hρ Z hZ
    have e1 := missProb_incorrectLev_congr (s := s) (val := w.val H L.commit s) (hins Z hZ) hZρ ιr
      (fun _ => ιr ρ) rfl
    have e2 := missProb_incorrectLev_congr (s := s) (val := w.val H L.commit s) (hins Z hZ) hZρ ιs
      (fun _ => ιs ρ) rfl
    show m Z (I ιr) ≤ m Z (I ιs)
    simp only [m, hI] at e1 e2 ⊢
    rw [e1, e2]
    exact hιs ρ Z hZ hZρ (ιr ρ)
  refine Or.inr ⟨_, hXr, accList_length_eq hshort2, fun l' e hl v hv hgood => ?_⟩
  obtain ⟨hvc, xS, vS, wS, hsess, hRw, hkept⟩ :=
    L.toCompiled.good_found H P ω x s c (hS.shared x Ω P ω) hjZ hl hgood
  have hacc' : (run' (forkOn (L.secondStage x) r e)).out = .acc := by
    have h := acc_of_accList hl
    exact h
  obtain ⟨b, ρ, rb, hb, hρv, hrb, hcor⟩ :=
    hS.opens x s Pt Lv Ω P ω (forkOn (L.secondStage x) r e) v xS vS wS hs hPt hLv hin hacc' hvc hsess hRw
  rw [hQ] at hρv hrb
  have hcorr := hcor (w.val H L.commit s) (ιr ρ) fun q hq hok => by
    have hqf : q ∈ fd := hkept q hq hok
    refine ⟨fun hr => ?_, fun hi => ?_⟩
    · rw [hr] at hok
      exact val_assemble hagree hqf hr hok
    · rw [(hS.interior x Ω P ω r e ρ).1] at hi
      show ιOf fd (L.iroots x (run' r).seen ρ) q.2.pos = q.2.val
      rw [← hi]
      exact ιOf_eq hagree hqf
  have hvI : v ∈ incorrectLev s Pt (some Q) (w.val H L.commit s) ιr := hv
  unfold incorrectLev at hvI
  exact (mem_filter.1 hvI).2 ⟨b, ρ, rb, hb, hρv, hrb, hcorr⟩

/-! ## KS at two levels -/

set_option synthInstance.maxSize 1024 in
open Classical in
/-- **`CompiledLev.KS` at two levels**, at `Compiled`'s extractor and errors: `KSLinOn` on the statements in scope
with two levels (`Sampling.KSLin_of_ownColl` at `FailsTwoLev`, `pr_failsTwoLev_le` and `splitTwoLev`, with the bound
`1` off them). -/
theorem ks_two {H : Msg → Msg} {L : CompiledLev Coin A} (hA : A.KS H) (hbind : L.commit.Binds H) (hS : L.Sessions H)
    (δ : ℝ) (hδ : 0 ≤ δ) :
    KSLinOn H L.V (fun x => L.Scope x ∧ L.TwoLevel x) (RLev H L.commit L.parse L.partition L.levels L.top δ) L.E
      (L.κ δ) L.γ (fun _ => 1) (Bext H L.V L.E) := by
  have hSK : L.toCompiled.SessionKS H := Compiled.sessionKS_of hA ⟨hS.fixed, hS.runs⟩
  have h := Sampling.KSLin_of_ownColl H L.V (RLev H L.commit L.parse L.partition L.levels L.top δ) L.E
    (fun x => if L.TwoLevel x then L.κ δ x else 1) L.γ
    (fun x Ω P => if L.TwoLevel x then
        fun s => (run H P L.V x s.1 s.2.1).out = .acc ∧
          L.FailsTwoLev δ x (run H P L.V x s.1) (L.toCompiled.Found H x (run H P L.V x s.1)) s.2.2
      else fun s => (run H P L.V x s.1 s.2.1).out = .acc ∧
          ¬ (RLev H L.commit L.parse L.partition L.levels L.top δ).ext x (L.E.eval H P x s.1 s.2.1 s.2.2).1)
    (fun x Ω P s hacc hfail => by
      by_cases h1 : L.TwoLevel x
      · rw [if_pos h1]
        exact (splitTwoLev hbind hS δ x h1 P s.1 s.2.1 s.2.2 hacc hfail).imp_left fun h => ⟨hacc, h⟩
      · rw [if_neg h1]
        exact Or.inl ⟨hacc, hfail⟩)
    (fun x Ω _ _ P => by
      by_cases h1 : L.TwoLevel x
      · rw [if_pos h1, if_pos h1]
        obtain ⟨β, hβ0, hβ, hκ⟩ := hSK x Ω P
        exact L.pr_failsTwoLev_le H hδ hS x Ω P (fun ω => L.toCompiled.Found H x (run H P L.V x ω)) β hβ0 hβ hκ
      · rw [if_neg h1, if_neg h1]
        exact (Pr_le_one' _).trans (le_add_of_nonneg_right (Real.sqrt_nonneg _)))
  intro x hx Ω _ _ P
  have hx' := h x Ω P
  dsimp only at hx'
  rwa [if_pos hx.2] at hx'

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem mid_eq_zero_of_twoLevel {L : CompiledLev Coin A} {x : Msg} (hx : L.Scope x) (h2 : L.TwoLevel x) :
    L.mid x = 0 := by
  obtain ⟨s, Pt, Lv, hs, hPt, hLv, -⟩ := hx
  obtain ⟨l₁, l₂, hL⟩ := h2 s Pt Lv hs hPt hLv
  rw [mid_eq hs hPt hLv, hL]
  rfl

/-- **`CompiledLev.KS` at two levels**, at the extractor with one wait per level: `ks_two` along `coinsLev`. -/
theorem ks_two_lev {H : Msg → Msg} {L : CompiledLev Coin A} (hA : A.KS H) (hbind : L.commit.Binds H)
    (hS : L.Sessions H) (δ : ℝ) (hδ : 0 ≤ δ) :
    KSLinOn H L.V (fun x => L.Scope x ∧ L.TwoLevel x) (RLev H L.commit L.parse L.partition L.levels L.top δ) L.Elev
      (L.κLev δ) L.γLev (fun _ => 1) (Bext H L.V L.Elev) := by
  have h := KSLinOn_of_equiv (E := L.E) (E' := L.Elev)
    (fun x hx => L.coinsLev x (mid_eq_zero_of_twoLevel hx.1 hx.2))
    (fun x hx r₀ c => L.Elev_ext_coinsLev x _ r₀ c) (ks_two hA hbind hS δ hδ)
  intro x hx Ω _ _ P
  have h0 := mid_eq_zero_of_twoLevel hx.1 hx.2
  rw [L.κLev_eq δ x h0, L.γLev_eq x h0]
  exact h x hx Ω P

/-- **`Samples` needs its length clause.** At two levels whose first samples its one block, a compiler that checks no
proof block and shows the upper samples `[{univ}, ∅]` meets every other clause of `Samples`, though what it checks
misses every set: the last clause reads the empty entry, a bound of `1`. -/
theorem samples_iff_length_of_opens_nothing {L : CompiledLev Coin A} {x : Msg} {Pt : Partition}
    {Lv : Verity.Levels (Fin Pt.blocks.length)} {l₁ l₂ : Verity.Level (Fin Pt.blocks.length)}
    (hL : Lv.levels = [l₁, l₂]) (hd : l₁.design Finset.univ = Verity.SamplingDesign.const {Finset.univ})
    (hc : ∀ r, L.checked x r = []) (hu : ∀ r, L.upper x Pt r = [{Finset.univ}, ∅]) :
    (L.Samples x Pt Lv ↔ ∀ r, (L.upper x Pt r).length + 1 = Lv.levels.length) ∧ ¬ L.Samples x Pt Lv ∧
      ∀ r (X : Finset ℕ),
        Pr (fun e : Fin (L.V.k x) → Coin => ∀ u ∈ L.checked x (forkOn (L.secondStage x) r e), u ∉ X) = 1 := by
  have hlen : ¬ ∀ r, (L.upper x Pt r).length + 1 = Lv.levels.length := fun h => by
    have h' := h (fun _ => default)
    rw [hu, hL] at h'
    simp only [List.length_cons, List.length_nil] at h'
    omega
  have hiff : L.Samples x Pt Lv ↔ ∀ r, (L.upper x Pt r).length + 1 = Lv.levels.length := by
    refine ⟨fun h => h.2.2.2.2, fun h5 => ⟨?_, ?_, ?_, ?_, h5⟩⟩
    · intro h1
      rw [hL] at h1
      simp only [List.length_cons, List.length_nil] at h1
      omega
    · intro l l' rest hl T
      rw [hL] at hl
      rw [← (List.cons.inj hl).1, hd]
      have hup : ∀ r, (L.upper x Pt r).headD ∅ = {Finset.univ} := fun r => by rw [hu]; rfl
      simp only [hup]
      show _ ≤ (Verity.Game.prCoin fun _ : Unit =>
        Disjoint ({Finset.univ} : Finset (Finset (Fin Pt.blocks.length))) T).toReal
      rw [← Claims.Sampling.Pr_eq_toReal_prCoin]
      by_cases hT : Disjoint ({Finset.univ} : Finset (Finset (Fin Pt.blocks.length))) T
      · rw [Claims.Pr_congr (q := fun _ => True) fun _ => iff_true_intro hT,
          Claims.Pr_congr (q := fun _ => True) fun _ => iff_true_intro hT, Claims.Sampling.Pr_true,
          Claims.Sampling.Pr_true]
      · rw [Claims.Pr_congr (q := fun _ => False) fun _ => iff_false_intro hT,
          Claims.Pr_congr (q := fun _ => False) fun _ => iff_false_intro hT, Claims.Pr_false, Claims.Pr_false]
    · intro j l' _ hlt
      rw [hL] at hlt
      simp only [List.length_cons, List.length_nil] at hlt
      omega
    · intro lm _ _ r X
      have hlast : (L.upper x Pt r).getLastD ∅ = ∅ := by rw [hu]; rfl
      rw [hlast, Finset.prod_empty]
      exact Pr_le_one' _
  refine ⟨hiff, fun h => hlen (hiff.1 h), fun r X => ?_⟩
  rw [Claims.Pr_congr (q := fun _ => True) fun e => iff_true_intro (by rw [hc]; simp), Claims.Sampling.Pr_true]

/-- **`Samples` ties each middle entry of `upper` to its level's design**: by its third clause at `e ≡ 0`, below an
entry `U_j` whose blocks the next level's design never leaves unsampled, every fork of level `j + 1`'s rounds names a
nonempty entry `U_{j+1}`, so `[U₀, ∅, …]` meets no `Samples` at three levels or more. -/
theorem samples_middle_nonempty {L : CompiledLev Coin A} {x : Msg} {Pt : Partition}
    {Lv : Verity.Levels (Fin Pt.blocks.length)} (h : L.Samples x Pt Lv) {j : ℕ}
    {l' : Verity.Level (Fin Pt.blocks.length)} (hl : Lv.levels[j + 1]? = some l') (hj : j + 2 < Lv.levels.length)
    (r : Fin (L.V.k x) → Coin) (hU : ((L.upper x Pt r).getD j ∅).Nonempty)
    (hdraw : ∀ X ∈ (L.upper x Pt r).getD j ∅, ∀ ω, ((l'.design X).draw ω).Nonempty) (d : Fin (L.V.k x) → Coin) :
    ((L.upper x Pt (forkOn (L.stages x j) r d)).getD (j + 1) ∅).Nonempty := by
  have h3 := (h.2.2.1 j l' hl hj r (fun _ => 0) (fun _ => zero_le_one)).1
  have hR : (∏ X ∈ (L.upper x Pt r).getD j ∅,
      Verity.Game.avg fun ω => ∏ Y ∈ (l'.design X).draw ω, (fun _ => (0 : ℝ≥0∞)) Y) = 0 := by
    obtain ⟨X₀, hX₀⟩ := hU
    refine Finset.prod_eq_zero hX₀ ?_
    have h0 : (fun ω => ∏ Y ∈ (l'.design X₀).draw ω, (fun _ => (0 : ℝ≥0∞)) Y) = fun _ => 0 :=
      funext fun ω => Finset.prod_eq_zero (hdraw X₀ hX₀ ω).choose_spec rfl
    rw [h0]
    unfold Verity.Game.avg
    simp
  rw [hR, ENNReal.toReal_zero] at h3
  by_contra hne
  rw [Finset.not_nonempty_iff_eq_empty] at hne
  set f : (Fin (L.V.k x) → Coin) → ℝ := fun d =>
    (∏ Y ∈ (L.upper x Pt (forkOn (L.stages x j) r d)).getD (j + 1) ∅, (fun _ => (0 : ℝ≥0∞)) Y).toReal with hf
  have hf0 : ∀ d, 0 ≤ f d := fun _ => ENNReal.toReal_nonneg
  have hfd : f d = 1 := by
    rw [hf]
    dsimp only
    rw [hne, Finset.prod_empty, ENNReal.toReal_one]
  have hsum : (1 : ℝ) ≤ ∑ a, f a := hfd.symm.le.trans (Finset.single_le_sum (fun a _ => hf0 a) (Finset.mem_univ d))
  have hpos : 0 < Ex f := div_pos (lt_of_lt_of_le one_pos hsum) (by exact_mod_cast Fintype.card_pos)
  exact absurd h3 (not_le.2 hpos)

end CompiledLev

end Claims.SamplingGeneric
