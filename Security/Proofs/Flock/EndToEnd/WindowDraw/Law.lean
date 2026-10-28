import Proofs.Flock.EndToEnd.WindowDraw.Member
import Proofs.Flock.Soundness.ExecDraw

/-!
# The stratified law at a member

A member whose units `R` are all in stratum `s₀` (`T`) sees only `s₀`'s coordinate of a stratified draw. Splitting the
coordinates at `s₀` (`avg_splitAt`), the draws of `T`'s `k`-subsets meeting `R` in `κ` units are a `(k − κ)`-subset of
`T \ R` beside a `κ`-subset of `R` (`sum_inter_split`), so the share given its size is the member's own uniform
`κ`-subset (`stratified_share`), of hypergeometric size (`stratified_share_card`), and at most `k s₀` units
(`share_card_le`). Two draws with the same units are the same draw (`draw_injective`), and a draw's share can be
replaced by any as many of the member's units (`exists_resample`).

`Flock.SecurityProofs.WindowDraw : Flock.Guarantees.WindowDraw`.
-/

namespace FlockSoundness.Audit.Window

open Finset Game
open scoped ENNReal

/-! ## Resampling the part of a subset in `R` -/

section Split

variable {α : Type} [DecidableEq α]

theorem sum_inter_split (T R : Finset α) (hRT : R ⊆ T) (k κ : ℕ) (hκ : κ ≤ k) (F : Finset α → ℝ≥0∞) :
    ∑ D ∈ (T.powersetCard k).filter (fun D => (D ∩ R).card = κ), F D =
      ∑ A ∈ (T \ R).powersetCard (k - κ), ∑ B ∈ R.powersetCard κ, F (A ∪ B) := by
  rw [← Finset.sum_product' (f := fun A B => F (A ∪ B))]
  refine Finset.sum_nbij' (fun D => (D \ R, D ∩ R)) (fun p => p.1 ∪ p.2) ?_ ?_ ?_ ?_ ?_
  · intro D hD
    simp only [mem_filter, mem_powersetCard] at hD
    obtain ⟨⟨hDT, hDk⟩, hDR⟩ := hD
    simp only [mem_product, mem_powersetCard]
    refine ⟨⟨sdiff_subset_sdiff hDT le_rfl, ?_⟩, inter_subset_right, hDR⟩
    have := card_sdiff_add_card_inter D R
    omega
  · rintro ⟨A, B⟩ hp
    simp only [mem_product, mem_powersetCard] at hp
    obtain ⟨⟨hAT, hAk⟩, hBR, hBk⟩ := hp
    have hdis : Disjoint A R := disjoint_of_subset_left hAT sdiff_disjoint
    simp only [mem_filter, mem_powersetCard]
    refine ⟨⟨union_subset (hAT.trans sdiff_subset) (hBR.trans hRT), ?_⟩, ?_⟩
    · rw [card_union_of_disjoint (disjoint_of_subset_right hBR hdis)]
      omega
    · rw [union_inter_distrib_right, disjoint_iff_inter_eq_empty.1 hdis, empty_union,
        inter_eq_left.2 hBR, hBk]
  · intro D _
    exact sdiff_union_inter D R
  · rintro ⟨A, B⟩ hp
    simp only [mem_product, mem_powersetCard] at hp
    obtain ⟨⟨hAT, _⟩, hBR, _⟩ := hp
    have hdis : Disjoint A R := disjoint_of_subset_left hAT sdiff_disjoint
    simp only [Prod.mk.injEq]
    refine ⟨?_, ?_⟩
    · rw [union_sdiff_distrib, sdiff_eq_self_of_disjoint hdis, sdiff_eq_empty_iff_subset.2 hBR, union_empty]
    · rw [union_inter_distrib_right, disjoint_iff_inter_eq_empty.1 hdis, empty_union, inter_eq_left.2 hBR]
  · intro D _
    simp only [sdiff_union_inter]

theorem sum_inter_resample (T R : Finset α) (hRT : R ⊆ T) (k κ : ℕ) (F : Finset α → ℝ≥0∞) :
    ∑ D ∈ T.powersetCard k, (if (D ∩ R).card = κ then F D else 0) =
      ∑ D ∈ T.powersetCard k, (if (D ∩ R).card = κ then
        ((R.powersetCard κ).card : ℝ≥0∞)⁻¹ * ∑ B ∈ R.powersetCard κ, F ((D \ R) ∪ B) else 0) := by
  rw [← sum_filter, ← sum_filter]
  by_cases hκ : κ ≤ k
  · rw [sum_inter_split T R hRT k κ hκ, sum_inter_split T R hRT k κ hκ]
    refine sum_congr rfl fun A hA => ?_
    rw [mem_powersetCard] at hA
    have hdis : Disjoint A R := disjoint_of_subset_left hA.1 sdiff_disjoint
    have hAB : ∀ B ∈ R.powersetCard κ, (A ∪ B) \ R = A := fun B hB => by
      rw [union_sdiff_distrib, sdiff_eq_self_of_disjoint hdis,
        sdiff_eq_empty_iff_subset.2 (mem_powersetCard.1 hB).1, union_empty]
    have hR : ∑ B ∈ R.powersetCard κ, ((R.powersetCard κ).card : ℝ≥0∞)⁻¹ *
        ∑ B' ∈ R.powersetCard κ, F ((A ∪ B) \ R ∪ B') =
        ∑ B ∈ R.powersetCard κ, ((R.powersetCard κ).card : ℝ≥0∞)⁻¹ * ∑ B' ∈ R.powersetCard κ, F (A ∪ B') :=
      sum_congr rfl fun B hB => by rw [hAB B hB]
    rw [hR, sum_const, nsmul_eq_mul, ← mul_assoc]
    rcases (R.powersetCard κ).eq_empty_or_nonempty with h | h
    · simp [h]
    · rw [ENNReal.mul_inv_cancel (by exact_mod_cast h.card_pos.ne') (ENNReal.natCast_ne_top _), one_mul]
  · have he : (T.powersetCard k).filter (fun D => (D ∩ R).card = κ) = ∅ := by
      refine filter_false_of_mem fun D hD h => hκ ?_
      rw [← (mem_powersetCard.1 hD).2, ← h]
      exact card_le_card inter_subset_left
    rw [he, sum_empty, sum_empty]

/-- The `k`-subsets of `T` meeting `R ⊆ T` in `κ` units: `C(|R|, κ) · C(|T| − |R|, k − κ)` of them. -/
theorem card_filter_inter (T R : Finset α) (hRT : R ⊆ T) (k κ : ℕ) (hκ : κ ≤ k) :
    ((T.powersetCard k).filter fun D => (D ∩ R).card = κ).card =
      R.card.choose κ * (T.card - R.card).choose (k - κ) := by
  have h := sum_inter_split T R hRT k κ hκ (fun _ => 1)
  simp only [sum_const, nsmul_eq_mul, mul_one, card_powersetCard, card_sdiff_of_subset hRT] at h
  exact_mod_cast h.trans (mul_comm _ _)

end Split

/-! ## The stratified law at a member -/

section Stratified

variable {N m : ℕ} {σ : Fin N → Fin m} {k : Fin m → ℕ}

/-- A coordinate of the stratified law: stratum `s`'s `k s`-subset. -/
abbrev Coord (σ : Fin N → Fin m) (k : Fin m → ℕ) (s : Fin m) : Type := ↥((Law.stratum σ s).powersetCard (k s))

theorem coord_subset {s : Fin m} (x : Coord σ k s) : x.1 ⊆ Law.stratum σ s := (mem_powersetCard.1 x.2).1

/-- The draw outside stratum `s₀`: the other strata's subsets. -/
def rest (s₀ : Fin m) (r : ∀ j : {j // j ≠ s₀}, Coord σ k j) : Finset (Fin N) := univ.biUnion fun j => (r j).1

theorem disjoint_rest (s₀ : Fin m) (r : ∀ j : {j // j ≠ s₀}, Coord σ k j) :
    Disjoint (rest s₀ r) (Law.stratum σ s₀) := by
  rw [disjoint_left]
  intro u hu hs
  simp only [rest, mem_biUnion, mem_univ, true_and] at hu
  obtain ⟨j, hj⟩ := hu
  exact j.2 ((Law.mem_stratum.1 (coord_subset (r j) hj)).symm.trans (Law.mem_stratum.1 hs))

variable (hk : ∀ s, k s ≤ (Law.stratum σ s).card)

theorem draw_splitAt (s₀ : Fin m) (x : Coord σ k s₀) (r : ∀ j : {j // j ≠ s₀}, Coord σ k j) :
    (Law.stratified σ k hk).draw ((Equiv.piSplitAt s₀ (Coord σ k)).symm (x, r)) = x.1 ∪ rest s₀ r := by
  ext u
  show u ∈ univ.biUnion (fun s => ((Equiv.piSplitAt s₀ (Coord σ k)).symm (x, r) s).1) ↔ _
  simp only [mem_biUnion, mem_univ, true_and, mem_union, rest, Equiv.piSplitAt_symm_apply]
  constructor
  · rintro ⟨s, hs⟩
    by_cases h : s = s₀
    · subst h
      left
      simpa using hs
    · right
      exact ⟨⟨s, h⟩, by simpa [h] using hs⟩
  · rintro (hu | ⟨j, hu⟩)
    · exact ⟨s₀, by simpa using hu⟩
    · exact ⟨j.1, by simpa [j.2] using hu⟩

theorem avg_splitAt (s₀ : Fin m) (g : (Law.stratified σ k hk).Ω → ℝ≥0∞) :
    avg g = avg fun r : (∀ j : {j // j ≠ s₀}, Coord σ k j) => avg fun x : Coord σ k s₀ =>
      g ((Equiv.piSplitAt s₀ (Coord σ k)).symm (x, r)) := by
  have h := DrawExec.avg_equiv (Equiv.piSplitAt s₀ (Coord σ k)).symm (g : (∀ s, Coord σ k s) → ℝ≥0∞)
  rw [DrawExec.avg_prod, Law.avg_comm] at h
  exact h.symm

variable {hk}

theorem range_subset_stratum (mb : Member) (s₀ : Fin m)
    (hin : ∀ u : Fin N, mb.base ≤ u.val → u.val < mb.base + mb.size → σ u = s₀) :
    (mb.range : Finset (Fin N)) ⊆ Law.stratum σ s₀ := by
  intro u hu
  simp only [Member.range, mem_filter, mem_univ, true_and] at hu
  exact Law.mem_stratum.2 (hin u hu.1 hu.2)

theorem draw_inter_range (mb : Member) (s₀ : Fin m) (hRT : (mb.range : Finset (Fin N)) ⊆ Law.stratum σ s₀)
    (ω : (Law.stratified σ k hk).Ω) :
    (Law.stratified σ k hk).draw ω ∩ mb.range = (ω s₀).1 ∩ mb.range := by
  ext u
  show u ∈ univ.biUnion (fun s => (ω s).1) ∩ _ ↔ _
  simp only [mem_inter, mem_biUnion, mem_univ, true_and]
  constructor
  · rintro ⟨⟨s, hs⟩, hu⟩
    have h1 : σ u = s := Law.mem_stratum.1 (coord_subset (ω s) hs)
    have h2 : σ u = s₀ := Law.mem_stratum.1 (hRT hu)
    exact ⟨h1.symm.trans h2 ▸ hs, hu⟩
  · rintro ⟨hs, hu⟩
    exact ⟨⟨s₀, hs⟩, hu⟩

theorem avg_coord_resample (T R O : Finset (Fin N)) (hRT : R ⊆ T) (hO : Disjoint O T) (k₀ κ : ℕ)
    (f : Finset (Fin N) → ℝ≥0∞) :
    avg (fun x : ↥(T.powersetCard k₀) => if ((x.1 ∪ O) ∩ R).card = κ then f (x.1 ∪ O) else 0) =
    avg (fun x : ↥(T.powersetCard k₀) => if ((x.1 ∪ O) ∩ R).card = κ then
      ((R.powersetCard κ).card : ℝ≥0∞)⁻¹ * ∑ B ∈ R.powersetCard κ, f (((x.1 ∪ O) \ R) ∪ B) else 0) := by
  have hOR : Disjoint O R := disjoint_of_subset_right hRT hO
  have hint : ∀ D : Finset (Fin N), (D ∪ O) ∩ R = D ∩ R := fun D => by
    rw [union_inter_distrib_right, disjoint_iff_inter_eq_empty.1 hOR, union_empty]
  have hsd : ∀ D B : Finset (Fin N), ((D ∪ O) \ R) ∪ B = ((D \ R) ∪ B) ∪ O := fun D B => by
    rw [union_sdiff_distrib, sdiff_eq_self_of_disjoint hOR, union_right_comm]
  simp only [avg, hint, hsd]
  rw [← Finset.mul_sum, ← Finset.mul_sum]
  congr 1
  rw [Finset.sum_coe_sort (T.powersetCard k₀) (fun D => if (D ∩ R).card = κ then f (D ∪ O) else 0),
    Finset.sum_coe_sort (T.powersetCard k₀) (fun D => if (D ∩ R).card = κ then
      ((R.powersetCard κ).card : ℝ≥0∞)⁻¹ * ∑ B ∈ R.powersetCard κ, f (((D \ R) ∪ B) ∪ O) else 0)]
  exact sum_inter_resample T R hRT k₀ κ (fun D => f (D ∪ O))

theorem avg_subset_unshift (mb : Member) (h : mb.base + mb.size ≤ N) (κ : ℕ) (hκ : κ ≤ mb.size)
    (g : Finset (Fin N) → ℝ≥0∞) :
    avg (fun S : (Law.subset mb.size κ hκ).Ω => g (mb.unshift ((Law.subset mb.size κ hκ).draw S))) =
      (((mb.range : Finset (Fin N)).powersetCard κ).card : ℝ≥0∞)⁻¹ *
        ∑ B ∈ (mb.range : Finset (Fin N)).powersetCard κ, g B := by
  have hS : ∀ S : Finset (Fin mb.size), S ∈ (univ : Finset (Fin mb.size)).powersetCard κ ↔ S.card = κ :=
    fun S => by simp [mem_powersetCard]
  have hsum : ∑ S : (Law.subset mb.size κ hκ).Ω, g (mb.unshift ((Law.subset mb.size κ hκ).draw S)) =
      ∑ B ∈ (mb.range : Finset (Fin N)).powersetCard κ, g B := by
    show ∑ S : {S : Finset (Fin mb.size) // S.card = κ}, g (mb.unshift S.1) = _
    rw [← Finset.sum_subtype _ hS (fun S => g (mb.unshift S)), Member.range_eq_map h, powersetCard_map, sum_map]
    refine sum_congr rfl fun S _ => ?_
    rw [Member.unshift_eq_map h]
    rfl
  have hcard : Fintype.card (Law.subset mb.size κ hκ).Ω = ((mb.range : Finset (Fin N)).powersetCard κ).card := by
    rw [Member.range_eq_map h, powersetCard_map, card_map]
    show Fintype.card {S : Finset (Fin mb.size) // S.card = κ} = _
    rw [Fintype.card_subtype]
    congr 1
    ext S
    simp [mem_powersetCard]
  simp only [avg]
  rw [← Finset.mul_sum, hsum, hcard]

/-- **Decomposition, the law.** Given its size `κ`, the member's share of the stratified draw is its uniform
`κ`-subset, independent of the draw outside the member. -/
theorem stratified_share (mb : Member) (s₀ : Fin m) (hfit : mb.base + mb.size ≤ N)
    (hin : ∀ u : Fin N, mb.base ≤ u.val → u.val < mb.base + mb.size → σ u = s₀) (κ : ℕ) (hκ : κ ≤ mb.size)
    (f : Finset (Fin N) → ℝ≥0∞) :
    avg (fun ω => if (mb.share ((Law.stratified σ k hk).draw ω)).card = κ then
      f ((Law.stratified σ k hk).draw ω) else 0) =
    avg (fun ω => if (mb.share ((Law.stratified σ k hk).draw ω)).card = κ then
      avg (fun S : (Law.subset mb.size κ hκ).Ω =>
        f (((Law.stratified σ k hk).draw ω \ mb.range) ∪ mb.unshift ((Law.subset mb.size κ hκ).draw S))) else 0) := by
  have hRT := range_subset_stratum mb s₀ hin
  simp only [Member.card_share hfit]
  rw [avg_splitAt hk s₀, avg_splitAt hk s₀]
  refine congrArg avg (funext fun r => ?_)
  simp only [draw_splitAt]
  rw [avg_coord_resample (Law.stratum σ s₀) mb.range (rest s₀ r) hRT (disjoint_rest s₀ r) (k s₀) κ f]
  refine congrArg avg (funext fun x => ?_)
  split_ifs with hc
  · exact (avg_subset_unshift mb hfit κ hκ (fun B => f (((x.1 ∪ rest s₀ r) \ mb.range) ∪ B))).symm
  · rfl

theorem share_card_le (mb : Member) (s₀ : Fin m) (hfit : mb.base + mb.size ≤ N)
    (hin : ∀ u : Fin N, mb.base ≤ u.val → u.val < mb.base + mb.size → σ u = s₀) (ω : (Law.stratified σ k hk).Ω) :
    (mb.share ((Law.stratified σ k hk).draw ω)).card ≤ k s₀ := by
  rw [Member.card_share hfit, draw_inter_range mb s₀ (range_subset_stratum mb s₀ hin)]
  exact (card_le_card inter_subset_left).trans (mem_powersetCard.1 (ω s₀).2).2.le

/-- **Decomposition, the share's size**: hypergeometric in the member's stratum. -/
theorem stratified_share_card (mb : Member) (s₀ : Fin m) (hfit : mb.base + mb.size ≤ N)
    (hin : ∀ u : Fin N, mb.base ≤ u.val → u.val < mb.base + mb.size → σ u = s₀) (κ : ℕ) (hκ : κ ≤ k s₀) :
    prCoin (fun ω => (mb.share ((Law.stratified σ k hk).draw ω)).card = κ) =
      ((mb.size.choose κ * ((Law.stratum σ s₀).card - mb.size).choose (k s₀ - κ) : ℕ) : ℝ≥0∞) /
        ((Law.stratum σ s₀).card.choose (k s₀) : ℝ≥0∞) := by
  classical
  have hRT := range_subset_stratum mb s₀ hin
  have hP : ∀ ω : (Law.stratified σ k hk).Ω, (mb.share ((Law.stratified σ k hk).draw ω)).card = κ ↔
      ∀ i ∈ ({s₀} : Finset (Fin m)), ((ω i).1 ∩ mb.range).card = κ := fun ω => by
    rw [Member.card_share hfit, draw_inter_range mb s₀ hRT]
    simp only [Finset.mem_singleton, forall_eq]
  have hNe : Nonempty (Coord σ k s₀) := ⟨⟨_, (Finset.powersetCard_nonempty.2 (hk s₀)).choose_spec⟩⟩
  have hNe' : ∀ s, Nonempty (Coord σ k s) := fun s =>
    ⟨⟨_, (Finset.powersetCard_nonempty.2 (hk s)).choose_spec⟩⟩
  have h1 : prCoin (fun ω : (Law.stratified σ k hk).Ω => (mb.share ((Law.stratified σ k hk).draw ω)).card = κ) =
      prCoin (fun ω : (∀ s, Coord σ k s) => ∀ i ∈ ({s₀} : Finset (Fin m)), ((ω i).1 ∩ mb.range).card = κ) :=
    congrArg prCoin (funext fun ω => propext (hP ω))
  rw [h1, prCoin_pi_forall (Ω := Coord σ k) {s₀} (fun i x => (x.1 ∩ mb.range).card = κ), prod_singleton,
    Law.prCoin_coe ((Law.stratum σ s₀).powersetCard (k s₀)) (fun D => (D ∩ mb.range).card = κ),
    card_filter_inter _ _ hRT _ _ hκ, card_powersetCard, Member.card_range hfit]

/-- At `k = 1`, the member's share is one unit with probability `size / n_{s₀}`. -/
theorem prCoin_share_one (mb : Member) (s₀ : Fin m) (hfit : mb.base + mb.size ≤ N)
    (hin : ∀ u : Fin N, mb.base ≤ u.val → u.val < mb.base + mb.size → σ u = s₀) (h1 : k s₀ = 1) :
    prCoin (fun ω => (mb.share ((Law.stratified σ k hk).draw ω)).card = 1) =
      (mb.size : ℝ≥0∞) / ((Law.stratum σ s₀).card : ℝ≥0∞) := by
  rw [stratified_share_card mb s₀ hfit hin 1 h1.ge, h1]
  simp

theorem draw_inter_stratum (ω : (Law.stratified σ k hk).Ω) (s : Fin m) :
    (Law.stratified σ k hk).draw ω ∩ Law.stratum σ s = (ω s).1 := by
  ext u
  show u ∈ univ.biUnion (fun s => (ω s).1) ∩ _ ↔ _
  simp only [mem_inter, mem_biUnion, mem_univ, true_and]
  constructor
  · rintro ⟨⟨s', hs'⟩, hu⟩
    have h1 : σ u = s' := Law.mem_stratum.1 (coord_subset (ω s') hs')
    exact (h1.symm.trans (Law.mem_stratum.1 hu)) ▸ hs'
  · intro hu
    exact ⟨⟨s, hu⟩, coord_subset (ω s) hu⟩

theorem draw_injective : Function.Injective (Law.stratified σ k hk).draw := by
  intro ω ω' h
  funext s
  exact Subtype.ext (by rw [← draw_inter_stratum ω s, ← draw_inter_stratum ω' s, h])

/-- **Resampling the share**: a draw whose share has `κ` units, with the share replaced by any `κ` of the member's
units, is a draw. -/
theorem exists_resample (mb : Member) (s₀ : Fin m) (hfit : mb.base + mb.size ≤ N)
    (hRT : (mb.range : Finset (Fin N)) ⊆ Law.stratum σ s₀) (ω : (Law.stratified σ k hk).Ω)
    (S : Finset (Fin mb.size)) (hc : ((ω s₀).1 ∩ mb.range).card = S.card) :
    ∃ ω', (Law.stratified σ k hk).draw ω' = ((Law.stratified σ k hk).draw ω \ mb.range) ∪ mb.unshift S := by
  set e := Equiv.piSplitAt s₀ (Coord σ k)
  have hω : (Law.stratified σ k hk).draw ω = (ω s₀).1 ∪ rest s₀ (e ω).2 :=
    (congrArg (Law.stratified σ k hk).draw (e.symm_apply_apply ω).symm).trans
      (draw_splitAt hk s₀ (e ω).1 (e ω).2)
  have hsub : ((ω s₀).1 \ mb.range) ∪ mb.unshift S ⊆ Law.stratum σ s₀ :=
    union_subset (sdiff_subset.trans (coord_subset _)) ((Member.unshift_subset_range hfit S).trans hRT)
  have hcard : (((ω s₀).1 \ mb.range) ∪ mb.unshift S).card = k s₀ := by
    rw [card_union_of_disjoint (disjoint_of_subset_right (Member.unshift_subset_range hfit S) sdiff_disjoint),
      Member.card_unshift hfit, card_sdiff, inter_comm, hc, (mem_powersetCard.1 (ω s₀).2).2]
    have : S.card ≤ k s₀ := hc ▸ (card_le_card inter_subset_left).trans (mem_powersetCard.1 (ω s₀).2).2.le
    omega
  refine ⟨e.symm (⟨_, mem_powersetCard.2 ⟨hsub, hcard⟩⟩, (e ω).2), ?_⟩
  have hOR : Disjoint (rest s₀ (e ω).2) mb.range := disjoint_of_subset_right hRT (disjoint_rest s₀ _)
  rw [draw_splitAt, hω, union_sdiff_distrib, sdiff_eq_self_of_disjoint hOR, union_right_comm]

end Stratified

end FlockSoundness.Audit.Window

namespace Flock.SecurityProofs

open FlockSoundness FlockSoundness.Audit FlockSoundness.Audit.Window

/-- **A window's stratified draw, restricted to a member, is the member's `subset` draw.** -/
theorem WindowDraw : Flock.Guarantees.WindowDraw := by
  refine ⟨fun N mb hfit => ⟨fun units v => Member.mem_share_filter hfit units v,
    fun W hnd => Member.sessionFile_eq hfit W hnd⟩, ?_⟩
  intro N m σ k hk mb s₀ hfit hin
  refine ⟨share_card_le mb s₀ hfit hin, fun κ hκ => stratified_share_card mb s₀ hfit hin κ hκ,
    fun κ hκ f => stratified_share mb s₀ hfit hin κ hκ f, fun h1 => ⟨prCoin_share_one mb s₀ hfit hin h1, ?_⟩⟩
  intro ω W hnd hW F hF
  obtain ⟨hnone, hsome⟩ := Member.sessionFile_eq hfit W hnd
  obtain ⟨-, -, hk', -⟩ := hsome F hF
  rw [hW] at hnone hk'
  have hle := share_card_le (hk := hk) mb s₀ hfit hin ω
  have hne : (mb.share ((Law.stratified σ k hk).draw ω)) ≠ ∅ := fun he => by
    rw [← hnone] at he
    rw [he] at hF
    cases hF
  have hpos := Finset.card_pos.2 (Finset.nonempty_iff_ne_empty.2 hne)
  omega

end Flock.SecurityProofs
