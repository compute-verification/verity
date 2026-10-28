import Mathlib.Data.Finset.Sort
import Proofs.Flock.EndToEnd.Window
import Proofs.Flock.Soundness.Audit.Share

/-!
# A window of members sharing one draw: `window_union`'s `hmarg`

`FlockSoundness.Audit.window` is the window game: the prover registers `R`, the verifier draws one live uniform
`kd`-subset `S` of the window's `N` units (`Law.subset N kd`), and the members' sessions run on it. Member `m`'s units are
a block `B m`, read in its own indices `Fin |B m|` by `(B m).orderEmbOfFin rfl`.

Conditioned on its share size `j = |S ∩ B|` and on the rest `U = S \ B` of the draw, a member's share is its own uniform
`j`-subset of `B` (A5, `Audit/Share.lean`). So a member bound at a fixed share size, in a fixed-`kd` member game, becomes
the member's marginal in the window, mixed by the hypergeometric share size (`share_card`):

* `Law.sum_share_le`: the draws with `|S ∩ B| = j` split as `S = T ∪ U`, `T` a `j`-subset of `B` and `U` a
  `(kd - j)`-subset of `Bᶜ`; `Law.sum_powersetCard_eq` reads the `T` as the member's own `Law.subset |B| j`.
* `Law.avg_le_share_mix`: a window average is at most the share-size mixture of the member-law averages' bounds.
* `window_marg`: member `B`'s event in the window has probability at most `∑ j, Pr[|S ∩ B| = j] · bd j`, where `bd j`
  bounds it in its own game at `j` draws, for every registration `R`, rest `U` and member strategy (`hmem`). `hmem` is
  where a member strategy runs the rest of the window around the member's session, now at a fixed share size.
* `Law.share_mix_choose`: mixing the member's own hypergeometric escape `C(|B| - K₀, j) / C(|B|, j)` by the share size
  gives the window's, `C(N - K₀, kd) / C(N, kd)` (`share_escape_total`, `subset_escape`).
* `Flock.SecurityProofs.window_union_shares`: the union bound over the members, each term its share-size mixture.
  `Flock.SecurityProofs.window_union_choose`: `window_union` (EndToEnd/Window) with its `hmarg` and `hE2E` discharged
  at `n m = N` and `kd m = kd`, when each member's bound at `j` draws is its `EndToEnd` bound there.

Nothing here is an assumption: `hmem` is a hypothesis about how the members' sessions sit in the window.
-/

namespace FlockSoundness.Audit

open Game
open scoped ENNReal

namespace Law

variable {N kd : ℕ}

theorem sum_eq_card_mul_avg {C : Type} [Fintype C] [Nonempty C] (f : C → ℝ≥0∞) :
    ∑ c, f c = (Fintype.card C : ℝ≥0∞) * avg f := by
  unfold avg
  rw [← Finset.mul_sum, ← mul_assoc, ENNReal.mul_inv_cancel (by exact_mod_cast Fintype.card_ne_zero)
    (ENNReal.natCast_ne_top _), one_mul]

/-- The `j`-subsets of `B`, read in `B`'s own indices. -/
theorem sum_powersetCard_eq (B : Finset (Fin N)) {j : ℕ} (hjB : j ≤ B.card) (f : Finset (Fin N) → ℝ≥0∞) :
    ∑ T ∈ B.powersetCard j, f T =
      ∑ T : (subset B.card j hjB).Ω, f (T.1.map (B.orderEmbOfFin rfl).toEmbedding) := by
  classical
  set e := (B.orderEmbOfFin rfl).toEmbedding
  have hmem : ∀ a, e a ∈ B := fun a => Finset.orderEmbOfFin_mem B rfl a
  symm
  refine Finset.sum_bij (fun T _ => T.1.map e) (fun T _ => ?_) (fun T₁ _ T₂ _ h => Subtype.ext
    ((Finset.map_inj).1 h)) (fun T hT => ?_) (fun _ _ => rfl)
  · refine Finset.mem_powersetCard.2 ⟨fun x hx => ?_, by rw [Finset.card_map, T.2]⟩
    obtain ⟨a, -, rfl⟩ := Finset.mem_map.1 hx
    exact hmem a
  · obtain ⟨hTB, hTc⟩ := Finset.mem_powersetCard.1 hT
    have hmap : (Finset.univ.filter fun a => e a ∈ T).map e = T := by
      ext x
      simp only [Finset.mem_map, Finset.mem_filter, Finset.mem_univ, true_and]
      refine ⟨fun ⟨a, ha, he⟩ => he ▸ ha, fun hx => ?_⟩
      have : x ∈ Set.range (B.orderEmbOfFin rfl) := by
        rw [Finset.range_orderEmbOfFin]
        exact hTB hx
      obtain ⟨a, rfl⟩ := this
      exact ⟨a, hx, rfl⟩
    refine ⟨⟨Finset.univ.filter fun a => e a ∈ T, ?_⟩, Finset.mem_univ _, hmap⟩
    rw [← Finset.card_map e, hmap, hTc]

/-- **The window's sum at one share size**, split at the share `T = S ∩ B` and the rest `U = S \ B`: when every rest
`U` averages at most `x` over the member's own uniform `j`-subset, the draws with `|S ∩ B| = j` sum to at most
`C(|B|, j) · C(N - |B|, kd - j) · x`. -/
theorem sum_share_le (hk : kd ≤ N) (B : Finset (Fin N)) (F : Finset (Fin N) → ℝ≥0∞) {j : ℕ} (hj : j ≤ kd)
    (x : ℝ≥0∞)
    (hF : ∀ (hjB : j ≤ B.card) U, Disjoint U B → U.card = kd - j →
      avg (fun T : (subset B.card j hjB).Ω => F (T.1.map (B.orderEmbOfFin rfl).toEmbedding ∪ U)) ≤ x) :
    ∑ S ∈ Finset.univ.filter (fun S : (subset N kd hk).Ω => (S.1 ∩ B).card = j), F S.1 ≤
      ((B.card.choose j * (N - B.card).choose (kd - j) : ℕ) : ℝ≥0∞) * x := by
  classical
  have hsplit : ∑ S ∈ Finset.univ.filter (fun S : (subset N kd hk).Ω => (S.1 ∩ B).card = j), F S.1 =
      ∑ p ∈ Bᶜ.powersetCard (kd - j) ×ˢ B.powersetCard j, F (p.2 ∪ p.1) := by
    have hTU : ∀ p ∈ Bᶜ.powersetCard (kd - j) ×ˢ B.powersetCard j, (p.2 ∪ p.1).card = kd := fun p hp => by
      obtain ⟨hU, hT⟩ := Finset.mem_product.1 hp
      obtain ⟨hU, hUc⟩ := Finset.mem_powersetCard.1 hU
      obtain ⟨hT, hTc⟩ := Finset.mem_powersetCard.1 hT
      rw [Finset.card_union_of_disjoint (Finset.disjoint_of_subset_left hT
        (Finset.subset_compl_iff_disjoint_left.1 hU)), hUc, hTc]
      omega
    refine Finset.sum_bij' (fun S _ => (S.1 \ B, S.1 ∩ B)) (fun p hp => ⟨p.2 ∪ p.1, hTU p hp⟩) ?_ ?_ ?_ ?_ ?_
    · intro S hS
      have hSj := (Finset.mem_filter.1 hS).2
      refine Finset.mem_product.2 ⟨Finset.mem_powersetCard.2 ⟨fun x hx => Finset.mem_compl.2
        (Finset.mem_sdiff.1 hx).2, ?_⟩, Finset.mem_powersetCard.2 ⟨Finset.inter_subset_right, hSj⟩⟩
      have := Finset.card_sdiff_add_card_inter S.1 B
      rw [hSj, S.2] at this
      exact (by omega : (S.1 \ B).card = kd - j)
    · intro p hp
      obtain ⟨hU, hT⟩ := Finset.mem_product.1 hp
      have hU' := (Finset.mem_powersetCard.1 hU).1
      obtain ⟨hT', hTc⟩ := Finset.mem_powersetCard.1 hT
      refine Finset.mem_filter.2 ⟨Finset.mem_univ _, ?_⟩
      rw [← hTc]
      refine congrArg Finset.card (Finset.ext fun x => ?_)
      have h1 := @hT' x
      have h2 : x ∈ p.1 → x ∉ B := fun h => Finset.mem_compl.1 (hU' h)
      simp only [Finset.mem_inter, Finset.mem_union]
      tauto
    · intro S _
      refine Subtype.ext (Finset.ext fun x => ?_)
      simp only [Finset.mem_union, Finset.mem_sdiff, Finset.mem_inter]
      tauto
    · intro p hp
      obtain ⟨hU, hT⟩ := Finset.mem_product.1 hp
      have hU' := (Finset.mem_powersetCard.1 hU).1
      have hT' := (Finset.mem_powersetCard.1 hT).1
      refine Prod.ext (Finset.ext fun x => ?_) (Finset.ext fun x => ?_)
      · have h1 := @hT' x
        have h2 : x ∈ p.1 → x ∉ B := fun h => Finset.mem_compl.1 (hU' h)
        simp only [Finset.mem_sdiff, Finset.mem_union]
        tauto
      · have h1 := @hT' x
        have h2 : x ∈ p.1 → x ∉ B := fun h => Finset.mem_compl.1 (hU' h)
        simp only [Finset.mem_inter, Finset.mem_union]
        tauto
    · intro S _
      refine congrArg F (Finset.ext fun x => ?_)
      simp only [Finset.mem_union, Finset.mem_sdiff, Finset.mem_inter]
      tauto
  rw [hsplit, Finset.sum_product]
  have hU : ∀ U ∈ Bᶜ.powersetCard (kd - j), ∑ T ∈ B.powersetCard j, F (T ∪ U) ≤
      (B.card.choose j : ℝ≥0∞) * x := by
    intro U hU
    obtain ⟨hU, hUc⟩ := Finset.mem_powersetCard.1 hU
    rcases le_or_gt j B.card with hjB | hjB
    · rw [sum_powersetCard_eq B hjB (fun T => F (T ∪ U)), sum_eq_card_mul_avg, card_subset_Ω]
      exact mul_le_mul_right (hF hjB U (Finset.subset_compl_iff_disjoint_right.1 hU) hUc) _
    · rw [Finset.powersetCard_eq_empty.2 hjB, Finset.sum_empty]
      exact zero_le
  refine (Finset.sum_le_sum hU).trans_eq ?_
  rw [Finset.sum_const, Finset.card_powersetCard, Finset.card_compl, Fintype.card_fin, nsmul_eq_mul]
  push_cast
  ring

/-- **A window average is the share-size mixture of member-law averages.** For a member's block `B` of the window's
`N` units: if, at each share size `j` and each rest `U` of the draw, the window's `F` averages at most `bd j` over the
member's own uniform `j`-subset of its `|B|` units, then over the window's uniform `kd`-subset `F` averages at most the
mixture of the `bd j` by the hypergeometric share size (`share_card`). -/
theorem avg_le_share_mix (hk : kd ≤ N) (B : Finset (Fin N)) (F : Finset (Fin N) → ℝ≥0∞) (bd : ℕ → ℝ≥0∞)
    (hF : ∀ j (hjB : j ≤ B.card), j ≤ kd → ∀ U, Disjoint U B → U.card = kd - j →
      avg (fun T : (subset B.card j hjB).Ω => F (T.1.map (B.orderEmbOfFin rfl).toEmbedding ∪ U)) ≤ bd j) :
    avg (fun S : (subset N kd hk).Ω => F S.1) ≤
      ∑ j ∈ Finset.range (kd + 1), prCoin (fun S : (subset N kd hk).Ω => (S.1 ∩ B).card = j) * bd j := by
  classical
  unfold avg
  rw [← Finset.mul_sum, ← Finset.sum_fiberwise_of_maps_to (g := fun S : (subset N kd hk).Ω => (S.1 ∩ B).card)
    (t := Finset.range (kd + 1)) fun S _ => Finset.mem_range.2 (Nat.lt_succ_of_le
      ((Finset.card_le_card Finset.inter_subset_left).trans_eq S.2)), Finset.mul_sum]
  refine Finset.sum_le_sum fun j hj => ?_
  have hjk := Nat.lt_succ_iff.1 (Finset.mem_range.1 hj)
  rw [share_card hk B hjk, card_subset_Ω hk, ENNReal.div_eq_inv_mul, mul_assoc]
  exact mul_le_mul_right (sum_share_le hk B F hjk (bd j) fun hjB U hUB hUc => hF j hjB hjk U hUB hUc) _

/-- The share sizes' probabilities sum to one. -/
theorem sum_share_card (hk : kd ≤ N) (B : Finset (Fin N)) :
    ∑ j ∈ Finset.range (kd + 1), prCoin (fun S : (subset N kd hk).Ω => (S.1 ∩ B).card = j) = 1 := by
  classical
  have h := Finset.card_eq_sum_card_fiberwise (s := (Finset.univ : Finset (subset N kd hk).Ω))
    (t := Finset.range (kd + 1)) (f := fun S => (S.1 ∩ B).card) fun S _ => Finset.mem_range.2
      (Nat.lt_succ_of_le ((Finset.card_le_card Finset.inter_subset_left).trans_eq S.2))
  simp only [prCoin_eq_card, ENNReal.div_eq_inv_mul, ← Finset.mul_sum]
  rw [← Nat.cast_sum, ← h, Finset.card_univ]
  exact ENNReal.inv_mul_cancel (by exact_mod_cast Fintype.card_ne_zero) (ENNReal.natCast_ne_top _)

/-- **The mixture of the members' hypergeometric bounds is the window's**: averaging a member's own escape at share
size `j`, `C(|B| - K₀, j) / C(|B|, j)`, plus a constant `c`, over the share size gives the window's escape
`C(N - K₀, kd) / C(N, kd)` plus `c` (`share_escape_total`, `subset_escape`). -/
theorem share_mix_choose (hk : kd ≤ N) (B : Finset (Fin N)) {K₀ : ℕ} (hK : K₀ ≤ B.card) (c : ℝ≥0∞) :
    ∑ j ∈ Finset.range (kd + 1), prCoin (fun S : (subset N kd hk).Ω => (S.1 ∩ B).card = j) *
        (((B.card - K₀).choose j : ℝ≥0∞) / (B.card.choose j : ℝ≥0∞) + c) =
      ((N - K₀).choose kd : ℝ≥0∞) / (N.choose kd : ℝ≥0∞) + c := by
  obtain ⟨W, hWB, hW⟩ := Finset.exists_subset_card_eq hK
  simp only [mul_add, Finset.sum_add_distrib, ← Finset.sum_mul, sum_share_card hk B, one_mul]
  rw [← hW, ← share_escape_total hk hWB]
  exact congrArg (· + c) (subset_escape hk W)

end Law

/-- **The window**: the prover registers `R`, the verifier draws one live uniform `kd`-subset `S` of the window's `N`
units, and the members' sessions run on it. Its output is the registration, the draw and the sessions' outcome. -/
def window (N kd : ℕ) (hk : kd ≤ N) (Reg β : Type) (live : Finset (Fin N) → Reg → Game β) :
    Game (Reg × Finset (Fin N) × β) :=
  .send Reg fun R => .coin (Law.subset N kd hk).Ω fun S => (live S.1 R).map fun t => (R, S.1, t)

/-- **A member's marginal in the window.** Member `B`'s event `E` in the window has probability at most the mixture,
by the hypergeometric share size, of its bound `bd j` in its own game at `j` draws: the member's share drawn as its own
uniform `j`-subset of its `|B|` units, the registration `R` and the rest `U` of the draw fixed (`hmem`). -/
theorem window_marg {N kd : ℕ} {Reg β : Type} (hk : kd ≤ N) (live : Finset (Fin N) → Reg → Game β)
    (B : Finset (Fin N)) (E : Reg × Finset (Fin N) × β → Prop) (bd : ℕ → ℝ≥0∞)
    (hmem : ∀ R j (hjB : j ≤ B.card), j ≤ kd → ∀ U, Disjoint U B → U.card = kd - j →
      ∀ τ : (T : (Law.subset B.card j hjB).Ω) → Strategy (live (T.1.map (B.orderEmbOfFin rfl).toEmbedding ∪ U) R),
        avg (fun T => prob (fun t => E (R, T.1.map (B.orderEmbOfFin rfl).toEmbedding ∪ U, t))
          (live (T.1.map (B.orderEmbOfFin rfl).toEmbedding ∪ U) R) (τ T)) ≤ bd j)
    (σ : Strategy (window N kd hk Reg β live)) :
    prob E (window N kd hk Reg β live) σ ≤
      ∑ j ∈ Finset.range (kd + 1), prCoin (fun S : (Law.subset N kd hk).Ω => (S.1 ∩ B).card = j) * bd j := by
  classical
  obtain ⟨R, τ'⟩ := σ
  let F : Finset (Fin N) → ℝ≥0∞ := fun S' => if h : S'.card = kd then
    prob (fun t => E (R, S', t)) (live S' R) (Strategy.ofMap (fun t => (R, S', t)) (live S' R) (τ' ⟨S', h⟩)) else 0
  have hwin : prob E (window N kd hk Reg β live) ⟨R, τ'⟩ = avg fun S : (Law.subset N kd hk).Ω => F S.1 := by
    change avg (fun S : (Law.subset N kd hk).Ω => prob E ((live S.1 R).map fun t => (R, S.1, t)) (τ' S)) = _
    refine congrArg avg (funext fun S => ?_)
    rw [prob_map]
    simp only [F, dite_eq_left S.2]
    rfl
  rw [hwin]
  refine Law.avg_le_share_mix hk B F bd fun j hjB hjk U hUB hUc => ?_
  have hc : ∀ T : (Law.subset B.card j hjB).Ω, (T.1.map (B.orderEmbOfFin rfl).toEmbedding ∪ U).card = kd := by
    intro T
    rw [Finset.card_union_of_disjoint (Finset.disjoint_of_subset_left (fun x hx => by
      obtain ⟨a, -, rfl⟩ := Finset.mem_map.1 hx
      exact Finset.orderEmbOfFin_mem B rfl a) hUB.symm), Finset.card_map, T.2, hUc]
    omega
  refine le_trans (le_of_eq (congrArg avg (funext fun T => ?_))) (hmem R j hjB hjk U hUB hUc fun T =>
    Strategy.ofMap _ _ (τ' ⟨T.1.map (B.orderEmbOfFin rfl).toEmbedding ∪ U, hc T⟩))
  simp only [F, dite_eq_left (hc T)]

end FlockSoundness.Audit

namespace Flock.SecurityProofs

open FlockSoundness FlockSoundness.Game
open scoped ENNReal

/-- **A window of members sharing one draw, end to end.** The window draws one uniform `kd`-subset of its `N` units
and member `m`'s units are the block `B m`. If, at each share size `j`, member `m`'s event (accepted and at least
`K₀ m` units wrong) has probability at most `bd m j` in its own game, its share its own uniform `j`-subset of
`B m` with the registration and the rest of the draw fixed (`hmem`), then the window accepts every member while some
member has at least `K₀ m` units wrong with probability at most the sum over members of their share-size mixtures
(`FlockSoundness.Audit.window_marg`, the union bound `window_prob_exists_le`). -/
theorem window_union_shares {ι Reg β : Type} {N kd : ℕ} (hk : kd ≤ N) (live : Finset (Fin N) → Reg → Game β)
    (σ : Strategy (FlockSoundness.Audit.window N kd hk Reg β live)) (ms : Finset ι) (B : ι → Finset (Fin N))
    (acc : ι → Reg × Finset (Fin N) × β → Prop) (wrong : ι → Reg × Finset (Fin N) × β → ℕ) (K₀ : ι → ℕ)
    (bd : ι → ℕ → ℝ≥0∞)
    (hmem : ∀ m ∈ ms, ∀ R j (hjB : j ≤ (B m).card), j ≤ kd → ∀ U, Disjoint U (B m) → U.card = kd - j →
      ∀ τ : (T : (FlockSoundness.Audit.Law.subset (B m).card j hjB).Ω) →
          Strategy (live (T.1.map ((B m).orderEmbOfFin rfl).toEmbedding ∪ U) R),
        avg (fun T => prob (fun t => acc m (R, T.1.map ((B m).orderEmbOfFin rfl).toEmbedding ∪ U, t) ∧
            K₀ m ≤ wrong m (R, T.1.map ((B m).orderEmbOfFin rfl).toEmbedding ∪ U, t))
          (live (T.1.map ((B m).orderEmbOfFin rfl).toEmbedding ∪ U) R) (τ T)) ≤ bd m j) :
    prob (fun a => (∀ m ∈ ms, acc m a) ∧ ∃ m ∈ ms, K₀ m ≤ wrong m a)
        (FlockSoundness.Audit.window N kd hk Reg β live) σ ≤
      ∑ m ∈ ms, ∑ j ∈ Finset.range (kd + 1),
        prCoin (fun S : (FlockSoundness.Audit.Law.subset N kd hk).Ω => (S.1 ∩ B m).card = j) * bd m j :=
  (prob_mono (q := fun a => ∃ m ∈ ms, acc m a ∧ K₀ m ≤ wrong m a)
      (fun _ ⟨hA, m, hm, hw⟩ => ⟨m, hm, hA m hm, hw⟩) _ σ).trans
    ((window_prob_exists_le ms _ _ σ).trans (Finset.sum_le_sum fun m hm =>
      FlockSoundness.Audit.window_marg hk live (B m) (fun a => acc m a ∧ K₀ m ≤ wrong m a) (bd m) (hmem m hm) σ))

/-- **`window_union`'s `hmarg` and `hE2E`, discharged for members sharing one draw.** When member `m`'s bound at each
share size `j` is its `EndToEnd` bound at its own `|B m|` units and `j` draws,
`C(|B m| - K₀ m, j) / C(|B m|, j) + ε_ks m + δ_link m` (`hmem`), and `K₀ m ≤ |B m|`, the window is `window_union` at
`n m = N` and `kd m = kd`: the members' share-size mixtures are the window's own hypergeometric escapes
(`FlockSoundness.Audit.Law.share_mix_choose`, from `share_escape_total`), not a union bound inside a member. -/
theorem window_union_choose {ι Reg β : Type} {N kd : ℕ} (hk : kd ≤ N) (live : Finset (Fin N) → Reg → Game β)
    (σ : Strategy (FlockSoundness.Audit.window N kd hk Reg β live)) (ms : Finset ι) (B : ι → Finset (Fin N))
    (acc : ι → Reg × Finset (Fin N) × β → Prop) (wrong : ι → Reg × Finset (Fin N) × β → ℕ) (K₀ : ι → ℕ)
    (εks δlink : ι → ℝ≥0∞) (hK : ∀ m ∈ ms, K₀ m ≤ (B m).card)
    (hmem : ∀ m ∈ ms, ∀ R j (hjB : j ≤ (B m).card), j ≤ kd → ∀ U, Disjoint U (B m) → U.card = kd - j →
      ∀ τ : (T : (FlockSoundness.Audit.Law.subset (B m).card j hjB).Ω) →
          Strategy (live (T.1.map ((B m).orderEmbOfFin rfl).toEmbedding ∪ U) R),
        avg (fun T => prob (fun t => acc m (R, T.1.map ((B m).orderEmbOfFin rfl).toEmbedding ∪ U, t) ∧
            K₀ m ≤ wrong m (R, T.1.map ((B m).orderEmbOfFin rfl).toEmbedding ∪ U, t))
          (live (T.1.map ((B m).orderEmbOfFin rfl).toEmbedding ∪ U) R) (τ T)) ≤
          (((B m).card - K₀ m).choose j : ℝ≥0∞) / ((B m).card.choose j : ℝ≥0∞) + εks m + δlink m) :
    prob (fun a => (∀ m ∈ ms, acc m a) ∧ ∃ m ∈ ms, K₀ m ≤ wrong m a)
        (FlockSoundness.Audit.window N kd hk Reg β live) σ ≤
      ∑ m ∈ ms, (((N - K₀ m).choose kd : ℝ≥0∞) / (N.choose kd : ℝ≥0∞) + εks m + δlink m) :=
  window_union _ σ ms acc wrong (fun _ => N) (fun _ => kd) K₀
    (fun m => ∑ j ∈ Finset.range (kd + 1),
      prCoin (fun S : (FlockSoundness.Audit.Law.subset N kd hk).Ω => (S.1 ∩ B m).card = j) *
        ((((B m).card - K₀ m).choose j : ℝ≥0∞) / ((B m).card.choose j : ℝ≥0∞) + εks m + δlink m)) εks δlink
    (fun m hm => FlockSoundness.Audit.window_marg hk live (B m) (fun a => acc m a ∧ K₀ m ≤ wrong m a) _
      (hmem m hm) σ)
    (fun m hm => le_of_eq (by
      simp only [add_assoc]
      exact FlockSoundness.Audit.Law.share_mix_choose hk (B m) (hK m hm) _))

end Flock.SecurityProofs
