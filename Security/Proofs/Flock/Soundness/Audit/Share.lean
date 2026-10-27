import Proofs.Flock.Soundness.Audit.Law

/-!
# A block's share of the uniform `kd`-subset

A window draws one uniform `kd`-subset `S` of its `N` units (`Law.subset N kd`), and a member's units are a block `B`.
Conditioned on its share size `|S ∩ B| = j`, the member's share `S ∩ B` is a uniform `j`-subset of `B`, so the member's
draw is its own `Law.subset |B| j` and its escape is the hypergeometric one, not a union bound.

* `card_share_eq`: the draws whose share is a given `T ⊆ B` number `C(N - |B|, kd - |T|)`, through the bijection
  `S ↦ S \ B` onto the `(kd - |T|)`-subsets of `Bᶜ`. The count depends on `T` only through `|T|`.
* `share_uniform`: two shares of one size are equally likely; `share_cond`: each is `1 / C(|B|, j)` of its size's mass.
* `share_card`: the share size is hypergeometric, `Pr[|S ∩ B| = j] = C(|B|, j) · C(N - |B|, kd - j) / C(N, kd)`.
* `share_escape`: a set `W ⊆ B` escapes the draw at share size `j` with that size's probability times
  `C(|B| - |W|, j) / C(|B|, j)`, the member's own `subset_escape` at `|B|` units and `j` draws.
* `share_escape_total`: so `W` escapes the window's draw with the average of that escape over the share size.
-/

namespace FlockSoundness.Audit

open Game
open scoped ENNReal

namespace Law

variable {N kd : ℕ}

/-- **The draws with a given share.** Among the `kd`-subsets of `Fin N`, those whose share of `B` is `T ⊆ B`, with
`|T| ≤ kd`, number `C(N - |B|, kd - |T|)`: `S ↦ S \ B` is a bijection onto the `(kd - |T|)`-subsets of `Bᶜ`, with
inverse `U ↦ T ∪ U`. -/
theorem card_share_eq {B T : Finset (Fin N)} (hT : T ⊆ B) (hj : T.card ≤ kd) :
    (Finset.univ.filter fun S : {S : Finset (Fin N) // S.card = kd} => S.1 ∩ B = T).card =
      (N - B.card).choose (kd - T.card) := by
  classical
  have hc : N - B.card = Bᶜ.card := by rw [Finset.card_compl, Fintype.card_fin]
  have hTU : ∀ U ∈ Bᶜ.powersetCard (kd - T.card), (T ∪ U).card = kd := fun U hU => by
    obtain ⟨hU, hUc⟩ := Finset.mem_powersetCard.1 hU
    rw [Finset.card_union_of_disjoint (Finset.disjoint_of_subset_left hT
      (Finset.subset_compl_iff_disjoint_left.1 hU)), hUc]
    omega
  rw [hc, ← Finset.card_powersetCard]
  refine Finset.card_bij' (fun S _ => S.1 \ B) (fun U hU => ⟨T ∪ U, hTU U hU⟩) ?_ ?_ ?_ ?_
  · intro S hS
    obtain hST := (Finset.mem_filter.1 hS).2
    refine Finset.mem_powersetCard.2 ⟨fun x hx => Finset.mem_compl.2 (Finset.mem_sdiff.1 hx).2, ?_⟩
    have := Finset.card_sdiff_add_card_inter S.1 B
    rw [hST, S.2] at this
    omega
  · intro U hU
    have hU' := (Finset.mem_powersetCard.1 hU).1
    refine Finset.mem_filter.2 ⟨Finset.mem_univ _, Finset.ext fun x => ?_⟩
    have h1 := @hT x
    have h2 : x ∈ U → x ∉ B := fun h => Finset.mem_compl.1 (hU' h)
    simp only [Finset.mem_inter, Finset.mem_union]
    tauto
  · intro S hS
    obtain hST := (Finset.mem_filter.1 hS).2
    refine Subtype.ext (Finset.ext fun x => ?_)
    have h1 : x ∈ T ↔ x ∈ S.1 ∧ x ∈ B := by rw [← hST, Finset.mem_inter]
    simp only [Finset.mem_union, Finset.mem_sdiff, h1]
    tauto
  · intro U hU
    have hU' := (Finset.mem_powersetCard.1 hU).1
    refine Finset.ext fun x => ?_
    have h1 := @hT x
    have h2 : x ∈ U → x ∉ B := fun h => Finset.mem_compl.1 (hU' h)
    simp only [Finset.mem_sdiff, Finset.mem_union]
    tauto

/-- No draw has a share larger than `kd`. -/
theorem card_share_eq_zero {B T : Finset (Fin N)} (hj : kd < T.card) :
    (Finset.univ.filter fun S : {S : Finset (Fin N) // S.card = kd} => S.1 ∩ B = T).card = 0 := by
  refine Finset.card_eq_zero.2 (Finset.filter_eq_empty_iff.2 fun S _ h => ?_)
  have := Finset.card_le_card (Finset.inter_subset_left (s₁ := S.1) (s₂ := B))
  rw [h, S.2] at this
  omega

/-- **The draws whose share lies in a family of `j`-subsets of `B`**: `C(N - |B|, kd - j)` for each member. -/
theorem card_share_mem {B : Finset (Fin N)} {j : ℕ} (𝒯 : Finset (Finset (Fin N)))
    (h𝒯 : ∀ T ∈ 𝒯, T ⊆ B ∧ T.card = j) (hj : j ≤ kd) :
    (Finset.univ.filter fun S : {S : Finset (Fin N) // S.card = kd} => S.1 ∩ B ∈ 𝒯).card =
      𝒯.card * (N - B.card).choose (kd - j) := by
  classical
  rw [Finset.card_eq_sum_card_fiberwise (f := fun S : {S : Finset (Fin N) // S.card = kd} => S.1 ∩ B) (t := 𝒯)
    (s := Finset.univ.filter fun S : {S : Finset (Fin N) // S.card = kd} => S.1 ∩ B ∈ 𝒯)
    fun S hS => Finset.mem_coe.2 (Finset.mem_filter.1 hS).2]
  refine Finset.sum_const_nat fun T hT => ?_
  rw [Finset.filter_filter, ← (h𝒯 T hT).2, ← card_share_eq (h𝒯 T hT).1 ((h𝒯 T hT).2 ▸ hj)]
  refine congrArg Finset.card (Finset.filter_congr fun S _ => ⟨fun h => h.2, fun h => ⟨h ▸ hT, h⟩⟩)

/-- **The share size is hypergeometric**, in counts: `C(|B|, j) · C(N - |B|, kd - j)` draws have `|S ∩ B| = j`. -/
theorem card_share_card (B : Finset (Fin N)) {j : ℕ} (hj : j ≤ kd) :
    (Finset.univ.filter fun S : {S : Finset (Fin N) // S.card = kd} => (S.1 ∩ B).card = j).card =
      B.card.choose j * (N - B.card).choose (kd - j) := by
  classical
  rw [← Finset.card_powersetCard j B, ← card_share_mem (B.powersetCard j)
    (fun T hT => Finset.mem_powersetCard.1 hT) hj]
  refine congrArg Finset.card (Finset.filter_congr fun S _ => ?_)
  rw [Finset.mem_powersetCard]
  exact ⟨fun h => ⟨Finset.inter_subset_right, h⟩, fun h => h.2⟩

/-- **The escape at a share size**, in counts: the draws with `|S ∩ B| = j` that miss `W ⊆ B` number
`C(|B| - |W|, j) · C(N - |B|, kd - j)`. -/
theorem card_share_escape {B W : Finset (Fin N)} (hW : W ⊆ B) {j : ℕ} (hj : j ≤ kd) :
    (Finset.univ.filter fun S : {S : Finset (Fin N) // S.card = kd} =>
        (S.1 ∩ B).card = j ∧ Disjoint S.1 W).card =
      (B.card - W.card).choose j * (N - B.card).choose (kd - j) := by
  classical
  have hBW : (B \ W).card = B.card - W.card := by
    have := Finset.card_sdiff_add_card_inter B W
    rw [Finset.inter_eq_right.2 hW] at this
    omega
  rw [← hBW, ← Finset.card_powersetCard j (B \ W), ← card_share_mem ((B \ W).powersetCard j)
    (fun T hT => ⟨(Finset.mem_powersetCard.1 hT).1.trans Finset.sdiff_subset, (Finset.mem_powersetCard.1 hT).2⟩) hj]
  refine congrArg Finset.card (Finset.filter_congr fun S _ => ?_)
  rw [Finset.mem_powersetCard]
  constructor
  · rintro ⟨hc, hd⟩
    refine ⟨fun x hx => Finset.mem_sdiff.2 ⟨(Finset.mem_inter.1 hx).2, fun hxW => ?_⟩, hc⟩
    exact Finset.disjoint_left.1 hd (Finset.mem_inter.1 hx).1 hxW
  · rintro ⟨hs, hc⟩
    refine ⟨hc, Finset.disjoint_left.2 fun x hxS hxW => ?_⟩
    exact (Finset.mem_sdiff.1 (hs (Finset.mem_inter.2 ⟨hxS, hW hxW⟩))).2 hxW

theorem card_subset_Ω (hk : kd ≤ N) : Fintype.card (subset N kd hk).Ω = N.choose kd :=
  (Fintype.card_finset_len kd).trans (by rw [Fintype.card_fin])

/-- **A given share's probability**: `C(N - |B|, kd - |T|) / C(N, kd)` for `T ⊆ B` with `|T| ≤ kd`, and `0` past
`kd`. -/
theorem share_eq (hk : kd ≤ N) {B T : Finset (Fin N)} (hT : T ⊆ B) :
    prCoin (fun S : (subset N kd hk).Ω => S.1 ∩ B = T) =
      ((if T.card ≤ kd then (N - B.card).choose (kd - T.card) else 0 : ℕ) : ℝ≥0∞) / (N.choose kd : ℝ≥0∞) := by
  classical
  rw [prCoin_eq_card, card_subset_Ω hk]
  congr 2
  split_ifs with hj
  · exact card_share_eq hT hj
  · exact card_share_eq_zero (by omega)

/-- **The share lemma (A5).** Conditioned on its size, a block's share of the uniform `kd`-subset is uniform: two
shares `T, T' ⊆ B` of one size are equally likely. -/
theorem share_uniform (hk : kd ≤ N) {B T T' : Finset (Fin N)} (hT : T ⊆ B) (hT' : T' ⊆ B)
    (hc : T.card = T'.card) :
    prCoin (fun S : (subset N kd hk).Ω => S.1 ∩ B = T) = prCoin (fun S : (subset N kd hk).Ω => S.1 ∩ B = T') := by
  rw [share_eq hk hT, share_eq hk hT', hc]

/-- **The share size is hypergeometric**: `Pr[|S ∩ B| = j] = C(|B|, j) · C(N - |B|, kd - j) / C(N, kd)`. -/
theorem share_card (hk : kd ≤ N) (B : Finset (Fin N)) {j : ℕ} (hj : j ≤ kd) :
    prCoin (fun S : (subset N kd hk).Ω => (S.1 ∩ B).card = j) =
      ((B.card.choose j * (N - B.card).choose (kd - j) : ℕ) : ℝ≥0∞) / (N.choose kd : ℝ≥0∞) := by
  classical
  rw [prCoin_eq_card, card_subset_Ω hk]
  congr 2
  exact card_share_card B hj

/-- **The conditional law of the share (A5)**: a share `T ⊆ B` has `1 / C(|B|, |T|)` of its size's probability, the
uniform `|T|`-subset of `B`'s. -/
theorem share_cond (hk : kd ≤ N) {B T : Finset (Fin N)} (hT : T ⊆ B) (hj : T.card ≤ kd) :
    prCoin (fun S : (subset N kd hk).Ω => S.1 ∩ B = T) * (B.card.choose T.card : ℝ≥0∞) =
      prCoin (fun S : (subset N kd hk).Ω => (S.1 ∩ B).card = T.card) := by
  rw [share_eq hk hT, share_card hk B hj, ENNReal.div_eq_inv_mul, ENNReal.div_eq_inv_mul]
  simp only [hj, ↓reduceIte]
  push_cast
  ring

/-- **The escape at a share size (A5, the form a window of members consumes).** A set `W ⊆ B` escapes the window's
uniform `kd`-subset at share size `j` with that size's probability times `C(|B| - |W|, j) / C(|B|, j)`, the escape of
the member's own uniform `j`-subset of its `|B|` units (`subset_escape`). -/
theorem share_escape (hk : kd ≤ N) {B W : Finset (Fin N)} (hW : W ⊆ B) {j : ℕ} (hj : j ≤ kd) :
    prCoin (fun S : (subset N kd hk).Ω => (S.1 ∩ B).card = j ∧ Disjoint S.1 W) =
      prCoin (fun S : (subset N kd hk).Ω => (S.1 ∩ B).card = j) *
        (((B.card - W.card).choose j : ℝ≥0∞) / (B.card.choose j : ℝ≥0∞)) := by
  classical
  have e : prCoin (fun S : (subset N kd hk).Ω => (S.1 ∩ B).card = j ∧ Disjoint S.1 W) =
      (((B.card - W.card).choose j * (N - B.card).choose (kd - j) : ℕ) : ℝ≥0∞) / (N.choose kd : ℝ≥0∞) := by
    rw [prCoin_eq_card, card_subset_Ω hk]
    congr 2
    exact card_share_escape hW hj
  rw [e, share_card hk B hj]
  rcases Nat.eq_zero_or_pos (B.card.choose j) with hb | hb
  · have ha : (B.card - W.card).choose j = 0 :=
      Nat.choose_eq_zero_of_lt ((Nat.sub_le _ _).trans_lt (Nat.choose_eq_zero_iff.1 hb))
    simp [ha, hb]
  · have hb0 : (B.card.choose j : ℝ≥0∞) ≠ 0 := by exact_mod_cast hb.ne'
    have hbt : (B.card.choose j : ℝ≥0∞) ≠ ⊤ := ENNReal.natCast_ne_top _
    push_cast
    rw [ENNReal.div_eq_inv_mul, ENNReal.div_eq_inv_mul, ENNReal.div_eq_inv_mul,
      show ((N.choose kd : ℝ≥0∞))⁻¹ * ((B.card.choose j : ℝ≥0∞) * ((N - B.card).choose (kd - j) : ℝ≥0∞)) *
          ((B.card.choose j : ℝ≥0∞)⁻¹ * ((B.card - W.card).choose j : ℝ≥0∞)) =
        ((N.choose kd : ℝ≥0∞))⁻¹ * (((B.card - W.card).choose j : ℝ≥0∞) * ((N - B.card).choose (kd - j) : ℝ≥0∞)) *
          ((B.card.choose j : ℝ≥0∞) * (B.card.choose j : ℝ≥0∞)⁻¹) by ring,
      ENNReal.mul_inv_cancel hb0 hbt, mul_one]

/-- **The window's escape of a block's set is hypergeometric**: `W ⊆ B` escapes the uniform `kd`-subset with the
average, over the share size `j` (`share_card`), of the member's own escape at `j` draws, `C(|B| - |W|, j) / C(|B|, j)`
(`share_escape`). -/
theorem share_escape_total (hk : kd ≤ N) {B W : Finset (Fin N)} (hW : W ⊆ B) :
    prCoin (fun S : (subset N kd hk).Ω => Disjoint S.1 W) =
      ∑ j ∈ Finset.range (kd + 1), prCoin (fun S : (subset N kd hk).Ω => (S.1 ∩ B).card = j) *
        (((B.card - W.card).choose j : ℝ≥0∞) / (B.card.choose j : ℝ≥0∞)) := by
  classical
  rw [show (∑ j ∈ Finset.range (kd + 1), prCoin (fun S : (subset N kd hk).Ω => (S.1 ∩ B).card = j) *
      (((B.card - W.card).choose j : ℝ≥0∞) / (B.card.choose j : ℝ≥0∞))) =
      ∑ j ∈ Finset.range (kd + 1), prCoin (fun S : (subset N kd hk).Ω => (S.1 ∩ B).card = j ∧ Disjoint S.1 W) from
    Finset.sum_congr rfl fun j hj => (share_escape hk hW (Nat.lt_succ_iff.1 (Finset.mem_range.1 hj))).symm]
  simp only [prCoin_eq_card, ENNReal.div_eq_inv_mul, ← Finset.mul_sum]
  congr 1
  rw [← Nat.cast_sum]
  congr 1
  rw [Finset.card_eq_sum_card_fiberwise (f := fun S : (subset N kd hk).Ω => (S.1 ∩ B).card)
    (t := Finset.range (kd + 1)) fun S _ => Finset.mem_coe.2 (Finset.mem_range.2 (Nat.lt_succ_of_le
      ((Finset.card_le_card Finset.inter_subset_left).trans_eq S.2)))]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [Finset.filter_filter]
  exact congrArg Finset.card (Finset.filter_congr fun S _ => And.comm)

end Law

end FlockSoundness.Audit
