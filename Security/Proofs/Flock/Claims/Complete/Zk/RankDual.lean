import Proofs.Flock.Claims.Complete.Zk.MaskRank

/-!
# The rank bound's dual side

The mask slot's points are rank-deficient exactly when a nonzero `GF(2)`-functional kills their weights' span. Over
`A = F^ι`, every such functional is `⟨μ, ·⟩ = Σ_i τ(μ_i ·)` for one `μ` (`τ` a nonzero additive map `F → GF(2)`,
`card_ann_mul`, `exists_ann`), and it kills the span exactly when `μ` restricted to its support `S` kills the span's
projection `π_S` (`pairing_zS`). So the deficient coins are at most the sum over the supports `S ≠ ∅` of the coins
where some `μ` of support `S` survives (`card_deficient_le`).

* `NS_mul_le`: at most `|F|^|S| / |π_S U|` functionals of support `S` kill `U`.
* `NS_step`, `dual_col`, `dual_iter`: a coordinate that draws freshly takes `U` to `U + c·U`, and on average over its
  coin divides the surviving functionals of support `S` by `|π_S U|`.
* `sum_ns_le`: so with the coordinates `Ys` fresh, a functional of support `S` survives with probability at most
  `E[min(1, |F|^|S| / |π_S U_Xs|^(|Ys| + 1))]`, the span `U_Xs` over the other coordinates.
* `card_deficient_le`: the bound summed over the supports, one choice of `Xs` per support.
-/

namespace FlockComplete.MaskRank

open Finset


section Pairing

theorem zmod2_eq_one {x : ZMod 2} (h : x ≠ 0) : x = 1 := by
  fin_cases x
  · exact absurd rfl h
  · rfl

/-- **An additive map to `GF(2)` taking the value `1` on a subgroup vanishes on exactly half of it.** -/
theorem card_ker_two {M : Type} [AddCommGroup M] [Fintype M] [DecidableEq M] (H : AddSubgroup M)
    [DecidablePred (· ∈ H)] (φ : M →+ ZMod 2) {g₀ : M} (hg₀ : g₀ ∈ H) (h₀ : φ g₀ = 1) :
    2 * ((univ.filter (· ∈ H)).filter fun g => φ g = 0).card = (univ.filter (· ∈ H)).card := by
  have hsplit := Finset.card_filter_add_card_filter_not (s := univ.filter (· ∈ H)) (fun g => φ g = 0)
  have hbij : ((univ.filter (· ∈ H)).filter fun g => φ g = 0).card =
      ((univ.filter (· ∈ H)).filter fun g => ¬ φ g = 0).card := by
    refine Finset.card_nbij' (fun g => g + g₀) (fun g => g - g₀) ?_ ?_ ?_ ?_
    · intro g hg
      simp only [Finset.mem_coe, Finset.mem_filter, Finset.mem_univ, true_and] at hg ⊢
      refine ⟨show g + g₀ ∈ H from H.add_mem hg.1 hg₀, ?_⟩
      rw [map_add, hg.2, h₀]
      decide
    · intro g hg
      simp only [Finset.mem_coe, Finset.mem_filter, Finset.mem_univ, true_and] at hg ⊢
      refine ⟨show g - g₀ ∈ H from H.sub_mem hg.1 hg₀, ?_⟩
      rw [map_sub, zmod2_eq_one hg.2, h₀, sub_self]
    · intro g _
      simp
    · intro g _
      simp
  omega

variable {ι F : Type} [Fintype ι] [DecidableEq ι] [Field F] [Fintype F] [DecidableEq F]

/-- The `GF(2)`-pairing `⟨μ, a⟩ = Σ_i τ(μ_i·a_i)`. -/
def pairing (τ : F →+ ZMod 2) (μ a : ι → F) : ZMod 2 := ∑ i, τ (μ i * a i)

/-- `⟨μ, ·⟩` as an additive map. -/
def pairingR (τ : F →+ ZMod 2) (μ : ι → F) : (ι → F) →+ ZMod 2 where
  toFun a := pairing τ μ a
  map_zero' := by simp [pairing]
  map_add' a b := by simp [pairing, mul_add, Finset.sum_add_distrib]

/-- `⟨·, a⟩` as an additive map. -/
def pairingL (τ : F →+ ZMod 2) (a : ι → F) : (ι → F) →+ ZMod 2 where
  toFun μ := pairing τ μ a
  map_zero' := by simp [pairing]
  map_add' μ ν := by simp [pairing, add_mul, Finset.sum_add_distrib]

omit [DecidableEq ι] [Fintype F] [DecidableEq F] in
theorem pairing_mul (τ : F →+ ZMod 2) (μ c u : ι → F) : pairing τ μ (c * u) = pairing τ (μ * c) u := by
  simp [pairing, mul_assoc]

omit [DecidableEq ι] [Fintype F] [DecidableEq F] in
theorem pairing_zero_right (τ : F →+ ZMod 2) (μ : ι → F) : pairing τ μ 0 = 0 := by simp [pairing]

/-- **The annihilator count**: in a subgroup `G` on which the pairing is nondegenerate, the annihilator of `V ≤ G` has
`|G|/|V|` elements. Double counting the pairs `(μ, v)` with `⟨μ, v⟩ = 0`. -/
theorem card_ann_mul (τ : F →+ ZMod 2) (G V : AddSubgroup (ι → F)) (hVG : V ≤ G)
    (hnd : ∀ v ∈ G, v ≠ 0 → ∃ μ ∈ G, pairing τ μ v = 1) [DecidablePred (· ∈ G)] [DecidablePred (· ∈ V)] :
    (univ.filter fun μ => μ ∈ G ∧ ∀ v ∈ V, pairing τ μ v = 0).card * (univ.filter (· ∈ V)).card =
      (univ.filter (· ∈ G)).card := by
  set Gf := univ.filter (· ∈ G)
  set Vf := univ.filter (· ∈ V)
  set Af := univ.filter fun μ => μ ∈ G ∧ ∀ v ∈ V, pairing τ μ v = 0
  have hrow : ∀ v ∈ Vf, 2 * (∑ μ ∈ Gf, if pairing τ μ v = 0 then 1 else 0) =
      Gf.card + if v = 0 then Gf.card else 0 := by
    intro v hv
    rw [Finset.sum_boole, Nat.cast_id]
    by_cases h0 : v = 0
    · subst h0
      simp only [↓reduceIte, pairing_zero_right, Finset.filter_true]
      ring
    · simp only [h0, ↓reduceIte, add_zero]
      obtain ⟨μ₀, hμ₀G, hμ₀⟩ := hnd v (hVG (Finset.mem_filter.1 hv).2) h0
      exact card_ker_two G (pairingL τ v) hμ₀G hμ₀
  have hcol : ∀ μ ∈ Gf, 2 * (∑ v ∈ Vf, if pairing τ μ v = 0 then 1 else 0) =
      Vf.card + if μ ∈ Af then Vf.card else 0 := by
    intro μ hμ
    rw [Finset.sum_boole, Nat.cast_id]
    by_cases hA : μ ∈ Af
    · simp only [hA, ↓reduceIte]
      have : Vf.filter (fun v => pairing τ μ v = 0) = Vf :=
        Finset.filter_true_of_mem fun v hv => (Finset.mem_filter.1 hA).2.2 v (Finset.mem_filter.1 hv).2
      rw [this]
      ring
    · simp only [hA, ↓reduceIte, add_zero]
      have : ∃ v ∈ V, pairing τ μ v ≠ 0 := by
        by_contra hc
        push Not at hc
        exact hA (Finset.mem_filter.2 ⟨mem_univ _, (Finset.mem_filter.1 hμ).2, hc⟩)
      obtain ⟨v₀, hv₀, hne⟩ := this
      exact card_ker_two V (pairingR τ μ) hv₀ (zmod2_eq_one hne)
  have h1 : 2 * (∑ μ ∈ Gf, ∑ v ∈ Vf, if pairing τ μ v = 0 then 1 else 0) = Vf.card * Gf.card + Gf.card := by
    rw [Finset.sum_comm, Finset.mul_sum, Finset.sum_congr rfl hrow, Finset.sum_add_distrib, Finset.sum_const,
      smul_eq_mul, Finset.sum_ite_eq' Vf 0 (fun _ => Gf.card)]
    simp [Vf, V.zero_mem]
  have h2 : 2 * (∑ μ ∈ Gf, ∑ v ∈ Vf, if pairing τ μ v = 0 then 1 else 0) = Gf.card * Vf.card + Af.card * Vf.card := by
    rw [Finset.mul_sum, Finset.sum_congr rfl hcol, Finset.sum_add_distrib, Finset.sum_const, smul_eq_mul,
      ← Finset.sum_filter]
    have : Gf.filter (· ∈ Af) = Af := by
      ext μ
      simp only [Finset.mem_filter, Gf, Af, mem_univ, true_and]
      exact ⟨fun h => h.2, fun h => ⟨h.1, h⟩⟩
    rw [this, Finset.sum_const, smul_eq_mul]
  have := h1.symm.trans h2
  nlinarith

end Pairing


section Support

open Classical

variable {ι F : Type} [Fintype ι] [DecidableEq ι] [Field F] [Fintype F] [DecidableEq F]

/-- The vectors supported in `S`. -/
def coordSub (S : Finset ι) : AddSubgroup (ι → F) where
  carrier := {a | ∀ i ∉ S, a i = 0}
  add_mem' ha hb i hi := by simp [ha i hi, hb i hi]
  zero_mem' i _ := rfl
  neg_mem' ha i hi := by simp [ha i hi]

/-- Zeroing the coordinates off `S`. -/
def zS (S : Finset ι) : (ι → F) →+ (ι → F) where
  toFun a i := if i ∈ S then a i else 0
  map_zero' := by funext i; simp
  map_add' a b := by funext i; by_cases h : i ∈ S <;> simp [h]

/-- The projection of `U` to the coordinates `S` (zero elsewhere). -/
def projS (S : Finset ι) (U : AddSubgroup (ι → F)) : AddSubgroup (ι → F) := U.map (zS S)

/-- The size of a subgroup. -/
noncomputable def csz (V : AddSubgroup (ι → F)) : ℕ := (univ.filter (· ∈ V)).card

/-- A vector's support. -/
def supp (μ : ι → F) : Finset ι := univ.filter (μ · ≠ 0)

/-- **The functionals of support exactly `S` killing `U`**, as `μ` with `⟨μ, ·⟩`. -/
noncomputable def NS (τ : F →+ ZMod 2) (S : Finset ι) (U : AddSubgroup (ι → F)) : ℕ :=
  (univ.filter fun μ => supp μ = S ∧ ∀ u ∈ U, pairing τ μ u = 0).card

theorem projS_le (S : Finset ι) (U : AddSubgroup (ι → F)) : projS S U ≤ coordSub S := by
  rintro _ ⟨u, -, rfl⟩ i hi
  simp [zS, hi]

theorem nondeg_coordSub (τ : F →+ ZMod 2) {z : F} (hz : τ z = 1) (S : Finset ι) :
    ∀ v ∈ coordSub S, v ≠ 0 → ∃ μ ∈ coordSub S, pairing τ μ v = 1 := by
  intro v hv h0
  obtain ⟨i, hi⟩ : ∃ i, v i ≠ 0 := by
    by_contra h
    push Not at h
    exact h0 (funext h)
  have hiS : i ∈ S := by
    by_contra h
    exact hi (hv i h)
  refine ⟨Pi.single i (z * (v i)⁻¹), fun j hj => ?_, ?_⟩
  · have : j ≠ i := fun h => hj (h ▸ hiS)
    simp [Pi.single_apply, this]
  · unfold pairing
    rw [Finset.sum_eq_single i (fun j _ hj => by simp [Pi.single_apply, hj]) (by simp)]
    simp [mul_assoc, inv_mul_cancel₀ hi, hz]

theorem card_coordSub (S : Finset ι) :
    (univ.filter (· ∈ (coordSub S : AddSubgroup (ι → F)))).card = Fintype.card F ^ S.card := by
  rw [← Fintype.card_subtype]
  have e : {a : ι → F // a ∈ (coordSub S : AddSubgroup (ι → F))} ≃ ({i // i ∈ S} → F) :=
    { toFun := fun a i => a.1 i
      invFun := fun ν => ⟨fun i => if h : i ∈ S then ν ⟨i, h⟩ else 0, fun i hi => by simp [hi]⟩
      left_inv := fun a => by
        apply Subtype.ext
        funext i
        by_cases h : i ∈ S
        · simp [h]
        · simp [h, a.2 i h]
      right_inv := fun ν => by
        funext i
        simp [i.2] }
  rw [Fintype.card_congr e, Fintype.card_fun, Fintype.card_coe]

theorem pairing_zS (τ : F →+ ZMod 2) {S : Finset ι} {μ : ι → F} (hμ : μ ∈ coordSub S) (u : ι → F) :
    pairing τ μ (zS S u) = pairing τ μ u := by
  unfold pairing
  refine Finset.sum_congr rfl fun i _ => ?_
  by_cases h : i ∈ S
  · simp [zS, h]
  · simp [zS, h, hμ i h]

theorem mem_coordSub_supp {μ : ι → F} {S : Finset ι} (h : supp μ = S) : μ ∈ coordSub S := by
  intro i hi
  by_contra hne
  exact hi (h ▸ Finset.mem_filter.2 ⟨mem_univ _, hne⟩)

/-- **At most `|F|^|S|/|π_S U|` functionals of support `S` kill `U`**: each is one of `π_S U`'s annihilator in the
vectors supported in `S`. -/
theorem NS_mul_le (τ : F →+ ZMod 2) {z : F} (hz : τ z = 1) (S : Finset ι) (U : AddSubgroup (ι → F)) :
    NS τ S U * csz (projS S U) ≤ Fintype.card F ^ S.card := by
  have h := card_ann_mul τ (coordSub S) (projS S U) (projS_le S U) (nondeg_coordSub τ hz S)
  rw [card_coordSub] at h
  rw [← h, NS, csz]
  refine Nat.mul_le_mul_right _ (Finset.card_le_card fun μ hμ => ?_)
  obtain ⟨hs, hU⟩ := (Finset.mem_filter.1 hμ).2
  refine Finset.mem_filter.2 ⟨mem_univ _, mem_coordSub_supp hs, ?_⟩
  rintro _ ⟨u, hu, rfl⟩
  rw [pairing_zS τ (mem_coordSub_supp hs)]
  exact hU u hu

/-- **A rank-deficient span has a nonzero functional killing it.** -/
theorem exists_ann (τ : F →+ ZMod 2) {z : F} (hz : τ z = 1) {U : AddSubgroup (ι → F)} (hU : U ≠ ⊤) :
    ∃ μ : ι → F, μ ≠ 0 ∧ ∀ u ∈ U, pairing τ μ u = 0 := by
  have h := card_ann_mul τ (coordSub univ) U (fun u _ i hi => absurd (mem_univ i) hi) (nondeg_coordSub τ hz univ)
  rw [card_coordSub, Finset.card_univ] at h
  by_contra hc
  push Not at hc
  have hA : (univ.filter fun μ => μ ∈ (coordSub univ : AddSubgroup (ι → F)) ∧ ∀ v ∈ U, pairing τ μ v = 0) ⊆
      {0} := by
    intro μ hμ
    obtain ⟨-, h2⟩ := (Finset.mem_filter.1 hμ).2
    by_contra hne
    obtain ⟨u, hu, hne'⟩ := hc μ (fun h => hne (Finset.mem_singleton.2 h))
    exact hne' (h2 u hu)
  have h1 := Finset.card_le_card hA
  rw [Finset.card_singleton] at h1
  have hle : (univ.filter (· ∈ U)).card ≤ Fintype.card (ι → F) := by
    rw [← Finset.card_univ]; exact Finset.card_le_card (Finset.filter_subset _ _)
  have hcard : Fintype.card (ι → F) = Fintype.card F ^ Fintype.card ι := Fintype.card_fun
  have hge : (univ.filter (· ∈ U)).card = (univ : Finset (ι → F)).card := by
    rw [Finset.card_univ]; nlinarith
  apply hU
  rw [eq_top_iff]
  intro x _
  have := Finset.filter_eq_self.1 (Finset.eq_of_subset_of_card_le (Finset.filter_subset _ _) hge.ge)
  exact this x (mem_univ x)

theorem supp_ne {μ : ι → F} {S : Finset ι} (hμ : supp μ = S) {i : ι} (h : i ∈ S) : μ i ≠ 0 := by
  rw [← hμ] at h
  exact (Finset.mem_filter.1 h).2

theorem mul_div_mem {μ : ι → F} {S : Finset ι} (hμ : supp μ = S) {ν : ι → F} (hν : ν ∈ coordSub S) (k : ι → F) :
    μ * (fun i => if i ∈ S then ν i / μ i else k i) = ν := by
  funext i
  by_cases h : i ∈ S
  · simp [h, mul_div_cancel₀ _ (supp_ne hμ h)]
  · simp [h, mem_coordSub_supp hμ i h, hν i h]

/-- **The coins that keep one functional off `c·U`**: for `μ` of support `S`, the `c` with `⟨μ, c·u⟩ = 0` on `U` are
`|F|^|ι|/|π_S U|`: `c ↦ (μ·c, c off S)` is a bijection onto `S`-vectors times the rest, and `μ·c` must annihilate
`π_S U`. -/
theorem card_step (τ : F →+ ZMod 2) {z : F} (hz : τ z = 1) {μ : ι → F} {S : Finset ι} (hμ : supp μ = S)
    (U : AddSubgroup (ι → F)) :
    (univ.filter fun c => ∀ u ∈ U, pairing τ μ (c * u) = 0).card * csz (projS S U) = Fintype.card (ι → F) := by
  have hW := card_ann_mul τ (coordSub S) (projS S U) (projS_le S U) (nondeg_coordSub τ hz S)
  rw [card_coordSub] at hW
  set W := univ.filter fun ν => ν ∈ (coordSub S : AddSubgroup (ι → F)) ∧ ∀ v ∈ projS S U, pairing τ ν v = 0
  have hbij : (univ.filter fun c => ∀ u ∈ U, pairing τ μ (c * u) = 0).card =
      (W ×ˢ univ.filter (· ∈ (coordSub Sᶜ : AddSubgroup (ι → F)))).card := by
    refine Finset.card_nbij' (fun c => (μ * c, zS Sᶜ c)) (fun p i => if i ∈ S then p.1 i / μ i else p.2 i)
      ?_ ?_ ?_ ?_
    · intro c hc
      simp only [Finset.mem_coe, Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_product, W] at hc ⊢
      have hmc : μ * c ∈ (coordSub S : AddSubgroup (ι → F)) := fun i hi => by
        simp [mem_coordSub_supp hμ i hi]
      refine ⟨⟨hmc, ?_⟩, fun i hi => by simp only [zS, AddMonoidHom.coe_mk, ZeroHom.coe_mk, hi, ↓reduceIte]⟩
      rintro _ ⟨u, hu, rfl⟩
      rw [pairing_zS τ hmc, ← pairing_mul]
      exact hc u hu
    · rintro ⟨ν, k⟩ hp
      simp only [Finset.mem_coe, Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_product, W] at hp ⊢
      intro u hu
      rw [pairing_mul, mul_div_mem hμ hp.1.1, ← pairing_zS τ hp.1.1]
      exact hp.1.2 _ ⟨u, hu, rfl⟩
    · intro c _
      funext i
      by_cases h : i ∈ S
      · simp [h, mul_div_cancel_left₀ _ (supp_ne hμ h)]
      · simp [h, zS]
    · rintro ⟨ν, k⟩ hp
      simp only [Finset.mem_coe, Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_product, W] at hp
      refine Prod.ext (mul_div_mem hμ hp.1.1 k) ?_
      funext i
      by_cases h : i ∈ S
      · simp [zS, h, hp.2 i (by simpa using h)]
      · simp [zS, h]
  rw [hbij, Finset.card_product, card_coordSub, mul_right_comm]
  unfold csz
  rw [hW, ← pow_add, Finset.card_add_card_compl, Fintype.card_fun]

theorem ann_sup_iff (τ : F →+ ZMod 2) (μ : ι → F) (U V : AddSubgroup (ι → F)) :
    (∀ x ∈ U ⊔ V, pairing τ μ x = 0) ↔ (∀ x ∈ U, pairing τ μ x = 0) ∧ ∀ x ∈ V, pairing τ μ x = 0 := by
  have key : ∀ W : AddSubgroup (ι → F), (∀ x ∈ W, pairing τ μ x = 0) ↔ W ≤ (pairingR τ μ).ker :=
    fun W => ⟨fun h x hx => h x hx, fun h x hx => h hx⟩
  rw [key, key, key, sup_le_iff]

/-- **One dual step**: summed over a uniform coordinate `c`, the functionals of support `S` killing `U + c·U` are those
killing `U`, each kept by a `1/|π_S U|` fraction of the coins. -/
theorem NS_step (τ : F →+ ZMod 2) {z : F} (hz : τ z = 1) (S : Finset ι) (U : AddSubgroup (ι → F)) :
    (∑ c : ι → F, NS τ S (U ⊔ U.map (AddMonoidHom.mulLeft c))) * csz (projS S U) =
      NS τ S U * Fintype.card (ι → F) := by
  have hrw : ∀ c : ι → F, NS τ S (U ⊔ U.map (AddMonoidHom.mulLeft c)) =
      ∑ μ ∈ univ.filter (fun μ => supp μ = S ∧ ∀ u ∈ U, pairing τ μ u = 0),
        if ∀ u ∈ U, pairing τ μ (c * u) = 0 then 1 else 0 := by
    intro c
    rw [NS, Finset.sum_boole, Nat.cast_id, Finset.filter_filter]
    congr 1
    refine Finset.filter_congr fun μ _ => ?_
    rw [ann_sup_iff]
    constructor
    · rintro ⟨hs, hU, hc⟩
      exact ⟨⟨hs, hU⟩, fun u hu => hc _ ⟨u, hu, rfl⟩⟩
    · rintro ⟨⟨hs, hU⟩, hc⟩
      exact ⟨hs, hU, by rintro _ ⟨u, hu, rfl⟩; exact hc u hu⟩
  simp only [hrw]
  rw [Finset.sum_comm, Finset.sum_mul, NS, Finset.card_eq_sum_ones, Finset.sum_mul, one_mul]
  refine Finset.sum_congr rfl fun μ hμ => ?_
  rw [Finset.sum_boole, Nat.cast_id]
  exact card_step τ hz (Finset.mem_filter.1 hμ).2.1 U

end Support

section Columns

open Classical

variable {ι F : Type} [Fintype ι] [DecidableEq ι] [Field F] [Fintype F] [DecidableEq F] {D : ℕ}

/-- **The span over the coordinates `Xs`**: the eq-vectors' products over `Xs` only. -/
def spanX (c : Fin D → ι → F) (Xs : Finset (Fin D)) : AddSubgroup (ι → F) :=
  AddSubgroup.closure (Set.range fun x : Fin D → Bool => ∏ t ∈ Xs, if x t then c t else 1 + c t)

theorem spanX_congr {c c' : Fin D → ι → F} {Xs : Finset (Fin D)} (h : ∀ t ∈ Xs, c t = c' t) :
    spanX c Xs = spanX c' Xs := by
  unfold spanX
  congr 2
  funext x
  exact Finset.prod_congr rfl fun t ht => by rw [h t ht]

theorem range_spanX_insert (c : Fin D → ι → F) {Xs : Finset (Fin D)} {t : Fin D} (ht : t ∉ Xs) :
    Set.range (fun x : Fin D → Bool => ∏ s ∈ insert t Xs, if x s then c s else 1 + c s) =
      (fun v => (1 + c t) * v) '' Set.range (fun x : Fin D → Bool => ∏ s ∈ Xs, if x s then c s else 1 + c s) ∪
        (fun v => c t * v) '' Set.range (fun x : Fin D → Bool => ∏ s ∈ Xs, if x s then c s else 1 + c s) := by
  ext y
  constructor
  · rintro ⟨x, rfl⟩
    dsimp only
    rw [Finset.prod_insert ht]
    cases hx : x t
    · exact Or.inl ⟨_, ⟨x, rfl⟩, by simp⟩
    · exact Or.inr ⟨_, ⟨x, rfl⟩, by simp⟩
  · have hupd : ∀ (x : Fin D → Bool) (b : Bool), (∏ s ∈ Xs, if Function.update x t b s then c s else 1 + c s) =
        ∏ s ∈ Xs, if x s then c s else 1 + c s := fun x b =>
      Finset.prod_congr rfl fun s hs => by rw [Function.update_of_ne (ne_of_mem_of_not_mem hs ht)]
    rintro (⟨v, ⟨x, rfl⟩, rfl⟩ | ⟨v, ⟨x, rfl⟩, rfl⟩)
    · exact ⟨Function.update x t false, by dsimp only; rw [Finset.prod_insert ht, hupd]; simp⟩
    · exact ⟨Function.update x t true, by dsimp only; rw [Finset.prod_insert ht, hupd]; simp⟩

/-- **The span recursion over a coordinate set** (`span_cons` at any coordinate): adding the coordinate `t` takes the
span `U` to `U + c_t·U`. -/
theorem spanX_insert [CharP F 2] (c : Fin D → ι → F) {Xs : Finset (Fin D)} {t : Fin D} (ht : t ∉ Xs) :
    spanX c (insert t Xs) = spanX c Xs ⊔ (spanX c Xs).map (AddMonoidHom.mulLeft (c t)) := by
  have h2 : ∀ a : ι → F, a + a = 0 := fun a => funext fun i => CharTwo.add_self_eq_zero (a i)
  apply le_antisymm
  · rw [spanX, AddSubgroup.closure_le, range_spanX_insert c ht]
    rintro y (⟨v, hv, rfl⟩ | ⟨v, hv, rfl⟩)
    · show (1 + c t) * v ∈ _
      rw [add_mul, one_mul]
      exact AddSubgroup.add_mem _ (AddSubgroup.mem_sup_left (AddSubgroup.subset_closure hv))
        (AddSubgroup.mem_sup_right ⟨v, AddSubgroup.subset_closure hv, rfl⟩)
    · exact AddSubgroup.mem_sup_right ⟨v, AddSubgroup.subset_closure hv, rfl⟩
  · refine sup_le ?_ ?_
    · rw [spanX, AddSubgroup.closure_le]
      intro v hv
      have e : v = (1 + c t) * v + c t * v := by rw [add_mul, one_mul, add_assoc, h2, add_zero]
      rw [SetLike.mem_coe, e]
      refine AddSubgroup.add_mem _ (AddSubgroup.subset_closure ?_) (AddSubgroup.subset_closure ?_)
      · rw [range_spanX_insert c ht]; exact Or.inl ⟨v, hv, rfl⟩
      · rw [range_spanX_insert c ht]; exact Or.inr ⟨v, hv, rfl⟩
    · rw [spanX, AddMonoidHom.map_closure, AddSubgroup.closure_le]
      rintro _ ⟨v, hv, rfl⟩
      apply AddSubgroup.subset_closure
      rw [range_spanX_insert c ht]
      exact Or.inr ⟨v, hv, rfl⟩

theorem spanX_le_union [CharP F 2] (c : Fin D → ι → F) (Xs Zs : Finset (Fin D)) : spanX c Xs ≤ spanX c (Xs ∪ Zs) := by
  induction Zs using Finset.induction_on with
  | empty => simp
  | insert t Zs ht ih =>
    rw [Finset.union_insert]
    by_cases h : t ∈ Xs ∪ Zs
    · rwa [Finset.insert_eq_of_mem h]
    · rw [spanX_insert c h]
      exact ih.trans le_sup_left

theorem spanX_mono [CharP F 2] (c : Fin D → ι → F) {Xs Ys : Finset (Fin D)} (h : Xs ⊆ Ys) :
    spanX c Xs ≤ spanX c Ys := by
  have := spanX_le_union c Xs (Ys \ Xs)
  rwa [Finset.union_sdiff_of_subset h] at this

theorem csz_mono {V W : AddSubgroup (ι → F)} (h : V ≤ W) : csz V ≤ csz W :=
  Finset.card_le_card fun v hv => Finset.mem_filter.2 ⟨mem_univ _, h (Finset.mem_filter.1 hv).2⟩

theorem csz_pos (V : AddSubgroup (ι → F)) : 0 < csz V :=
  Finset.card_pos.2 ⟨0, Finset.mem_filter.2 ⟨mem_univ _, V.zero_mem⟩⟩

/-- The coins' coordinate `t`, the fixed entries at their constants. -/
def col (fix : ι → Fin D → Option F) (r : ι → Fin D → F) (t : Fin D) : ι → F := fun i => (fix i t).getD (r i t)

/-- The coins with coordinate `t` replaced by `a`. -/
def setCol (r : ι → Fin D → F) (t : Fin D) (a : ι → F) : ι → Fin D → F := fun i s => if s = t then a i else r i s

theorem col_setCol_self {fix : ι → Fin D → Option F} {t : Fin D} (hfree : ∀ i, fix i t = none)
    (r : ι → Fin D → F) (a : ι → F) : col fix (setCol r t a) t = a := by
  funext i
  simp [col, setCol, hfree i]

theorem col_setCol_ne (fix : ι → Fin D → Option F) {s t : Fin D} (h : s ≠ t) (r : ι → Fin D → F) (a : ι → F) :
    col fix (setCol r t a) s = col fix r s := by
  funext i
  simp [col, setCol, h]

/-- **Resampling one coordinate**: summing `f` at the coins with coordinate `t` replaced, over the replacement, is
`|F|^|ι|` times summing `f`. -/
theorem sum_setCol {M : Type} [AddCommMonoid M] (t : Fin D) (f : (ι → Fin D → F) → M) :
    ∑ r, ∑ a : ι → F, f (setCol r t a) = Fintype.card (ι → F) • ∑ r, f r := by
  let e : (ι → Fin D → F) × (ι → F) ≃ (ι → Fin D → F) × (ι → F) :=
    { toFun := fun p => (setCol p.1 t p.2, fun i => p.1 i t)
      invFun := fun p => (setCol p.1 t p.2, fun i => p.1 i t)
      left_inv := fun p => by
        refine Prod.ext ?_ ?_
        · funext i s; by_cases h : s = t <;> simp [setCol, h]
        · funext i; simp [setCol]
      right_inv := fun p => by
        refine Prod.ext ?_ ?_
        · funext i s; by_cases h : s = t <;> simp [setCol, h]
        · funext i; simp [setCol] }
  rw [← Fintype.sum_prod_type', Fintype.sum_equiv e (fun p => f (setCol p.1 t p.2)) (fun p => f p.1) (fun _ => rfl),
    Fintype.sum_prod_type]
  simp only [Finset.sum_const, Finset.card_univ, Finset.smul_sum]

end Columns

section DualIter

open Classical

variable {ι F : Type} [Fintype ι] [DecidableEq ι] [Field F] [Fintype F] [DecidableEq F] [CharP F 2] {D : ℕ}

/-- **One dual step at the coins**: at a coordinate `t` every point draws freshly, the functionals of support `S`
killing the span with `t` are, on average over `t`'s coins, those killing it without `t` over `|π_S U|`. -/
theorem dual_col (τ : F →+ ZMod 2) {z : F} (hz : τ z = 1) (S : Finset ι) (fix : ι → Fin D → Option F)
    {Xs : Finset (Fin D)} {t : Fin D} (ht : t ∉ Xs) (hfree : ∀ i, fix i t = none)
    (h : (ι → Fin D → F) → ℝ) (hh : ∀ r a, h (setCol r t a) = h r) :
    ∑ r, (NS τ S (spanX (col fix r) (insert t Xs)) : ℝ) * h r =
      ∑ r, (NS τ S (spanX (col fix r) Xs) : ℝ) * h r / csz (projS S (spanX (col fix r) Xs)) := by
  have hA : (0 : ℝ) < Fintype.card (ι → F) := by exact_mod_cast Fintype.card_pos
  have key : ∀ r a, spanX (col fix (setCol r t a)) Xs = spanX (col fix r) Xs := fun r a =>
    spanX_congr fun s hs => col_setCol_ne fix (ne_of_mem_of_not_mem hs ht) r a
  have key2 : ∀ r a, spanX (col fix (setCol r t a)) (insert t Xs) =
      spanX (col fix r) Xs ⊔ (spanX (col fix r) Xs).map (AddMonoidHom.mulLeft a) := fun r a => by
    rw [spanX_insert _ ht, key, col_setCol_self hfree]
  have hstep : ∀ r, ∑ a : ι → F,
      (NS τ S (spanX (col fix r) Xs ⊔ (spanX (col fix r) Xs).map (AddMonoidHom.mulLeft a)) : ℝ) =
        (NS τ S (spanX (col fix r) Xs) : ℝ) * Fintype.card (ι → F) / csz (projS S (spanX (col fix r) Xs)) := by
    intro r
    have hc : (0 : ℝ) < csz (projS S (spanX (col fix r) Xs)) := by exact_mod_cast csz_pos _
    rw [eq_div_iff hc.ne']
    exact_mod_cast NS_step τ hz S (spanX (col fix r) Xs)
  apply mul_left_cancel₀ hA.ne'
  calc (Fintype.card (ι → F) : ℝ) * ∑ r, (NS τ S (spanX (col fix r) (insert t Xs)) : ℝ) * h r
      = ∑ r, ∑ a : ι → F, (NS τ S (spanX (col fix (setCol r t a)) (insert t Xs)) : ℝ) * h (setCol r t a) := by
        rw [sum_setCol t (fun r => (NS τ S (spanX (col fix r) (insert t Xs)) : ℝ) * h r), nsmul_eq_mul]
    _ = ∑ r, (∑ a : ι → F,
          (NS τ S (spanX (col fix r) Xs ⊔ (spanX (col fix r) Xs).map (AddMonoidHom.mulLeft a)) : ℝ)) * h r := by
        refine Finset.sum_congr rfl fun r _ => ?_
        rw [Finset.sum_mul]
        exact Finset.sum_congr rfl fun a _ => by rw [key2, hh]
    _ = (Fintype.card (ι → F) : ℝ) *
          ∑ r, (NS τ S (spanX (col fix r) Xs) : ℝ) * h r / csz (projS S (spanX (col fix r) Xs)) := by
        rw [Finset.mul_sum]
        refine Finset.sum_congr rfl fun r _ => ?_
        rw [hstep]
        ring

/-- **The dual steps at the fresh coordinates `Ys`**: each divides the expected count by at least `|π_S U_Xs|`. -/
theorem dual_iter (τ : F →+ ZMod 2) {z : F} (hz : τ z = 1) (S : Finset ι) (fix : ι → Fin D → Option F)
    (Xs : Finset (Fin D)) (Ys : Finset (Fin D)) (hd : Disjoint Xs Ys) (hfree : ∀ t ∈ Ys, ∀ i, fix i t = none)
    (h : (ι → Fin D → F) → ℝ) (h0 : ∀ r, 0 ≤ h r) (hh : ∀ t ∈ Ys, ∀ r a, h (setCol r t a) = h r) :
    ∑ r, (NS τ S (spanX (col fix r) (Xs ∪ Ys)) : ℝ) * h r ≤
      ∑ r, (NS τ S (spanX (col fix r) Xs) : ℝ) * h r / (csz (projS S (spanX (col fix r) Xs)) : ℝ) ^ Ys.card := by
  induction Ys using Finset.induction_on generalizing h with
  | empty => simp
  | insert t Ys ht ih =>
    have htX : t ∉ Xs := fun h' => Finset.disjoint_left.1 hd h' (Finset.mem_insert_self t Ys)
    have hd' : Disjoint Xs Ys := Finset.disjoint_of_subset_right (Finset.subset_insert t Ys) hd
    have htU : t ∉ Xs ∪ Ys := by simp [htX, ht]
    have hinv : ∀ t' ∈ Ys, ∀ r a, spanX (col fix (setCol r t' a)) Xs = spanX (col fix r) Xs :=
      fun t' ht' r a => spanX_congr fun s hs =>
        col_setCol_ne fix (ne_of_mem_of_not_mem hs fun h' => Finset.disjoint_left.1 hd' h' ht') r a
    rw [Finset.union_insert, dual_col τ hz S fix htU (hfree t (Finset.mem_insert_self t Ys)) h
      (hh t (Finset.mem_insert_self t Ys))]
    calc ∑ r, (NS τ S (spanX (col fix r) (Xs ∪ Ys)) : ℝ) * h r / csz (projS S (spanX (col fix r) (Xs ∪ Ys)))
        ≤ ∑ r, (NS τ S (spanX (col fix r) (Xs ∪ Ys)) : ℝ) * (h r / csz (projS S (spanX (col fix r) Xs))) := by
          refine Finset.sum_le_sum fun r _ => ?_
          rw [mul_div_assoc]
          refine mul_le_mul_of_nonneg_left (div_le_div_of_nonneg_left (h0 r) (by exact_mod_cast csz_pos _)
            (by exact_mod_cast csz_mono (AddSubgroup.map_mono (spanX_mono _ Finset.subset_union_left))))
            (Nat.cast_nonneg _)
      _ ≤ ∑ r, (NS τ S (spanX (col fix r) Xs) : ℝ) * (h r / csz (projS S (spanX (col fix r) Xs))) /
            (csz (projS S (spanX (col fix r) Xs)) : ℝ) ^ Ys.card :=
          ih hd' (fun t' ht' => hfree t' (Finset.mem_insert_of_mem ht'))
            (fun r => h r / csz (projS S (spanX (col fix r) Xs)))
            (fun r => div_nonneg (h0 r) (Nat.cast_nonneg _))
            (fun t' ht' r a => by rw [hh t' (Finset.mem_insert_of_mem ht'), hinv t' ht'])
      _ = ∑ r, (NS τ S (spanX (col fix r) Xs) : ℝ) * h r /
            (csz (projS S (spanX (col fix r) Xs)) : ℝ) ^ (insert t Ys).card := by
          rw [Finset.card_insert_of_notMem ht]
          refine Finset.sum_congr rfl fun r _ => ?_
          rw [pow_succ]
          field_simp

/-- **Some functional of support `S` survives** with probability at most `E[min(1, |F|^|S|/|π_S U_Xs|^(|Ys|+1))]`,
the coordinates `Ys` fresh: on the coins of `Xs` where that is below `1`, the expected count bounds the probability. -/
theorem sum_ns_le (τ : F →+ ZMod 2) {z : F} (hz : τ z = 1) (S : Finset ι) (fix : ι → Fin D → Option F)
    (Xs Ys : Finset (Fin D)) (hU : Xs ∪ Ys = univ) (hd : Disjoint Xs Ys) (hfree : ∀ t ∈ Ys, ∀ i, fix i t = none) :
    ∑ r, (if 1 ≤ NS τ S (spanX (col fix r) univ) then (1 : ℝ) else 0) ≤
      ∑ r, min 1 ((Fintype.card F : ℝ) ^ S.card / (csz (projS S (spanX (col fix r) Xs)) : ℝ) ^ (Ys.card + 1)) := by
  set g : (ι → Fin D → F) → ℝ := fun r =>
    (NS τ S (spanX (col fix r) Xs) : ℝ) / (csz (projS S (spanX (col fix r) Xs)) : ℝ) ^ Ys.card
  set h : (ι → Fin D → F) → ℝ := fun r => if 1 ≤ g r then 0 else 1
  have hinv : ∀ t ∈ Ys, ∀ r a, spanX (col fix (setCol r t a)) Xs = spanX (col fix r) Xs :=
    fun t ht r a => spanX_congr fun s hs =>
      col_setCol_ne fix (ne_of_mem_of_not_mem hs fun h' => Finset.disjoint_left.1 hd h' ht) r a
  have hiter := dual_iter τ hz S fix Xs Ys hd hfree h (fun r => by simp only [h]; split <;> norm_num)
    (fun t ht r a => by simp only [h, g, hinv t ht r a])
  rw [hU] at hiter
  calc ∑ r, (if 1 ≤ NS τ S (spanX (col fix r) univ) then (1 : ℝ) else 0)
      ≤ ∑ r, ((if 1 ≤ g r then (1 : ℝ) else 0) + (NS τ S (spanX (col fix r) univ) : ℝ) * h r) := by
        refine Finset.sum_le_sum fun r _ => ?_
        by_cases hg : 1 ≤ g r
        · simp only [h, hg, ↓reduceIte, mul_zero, add_zero]
          split <;> norm_num
        · simp only [h, hg, ↓reduceIte, mul_one, zero_add]
          split
          · exact_mod_cast ‹1 ≤ _›
          · exact Nat.cast_nonneg _
    _ = ∑ r, (if 1 ≤ g r then (1 : ℝ) else 0) + ∑ r, (NS τ S (spanX (col fix r) univ) : ℝ) * h r :=
        Finset.sum_add_distrib
    _ ≤ ∑ r, (if 1 ≤ g r then (1 : ℝ) else 0) +
          ∑ r, (NS τ S (spanX (col fix r) Xs) : ℝ) * h r / (csz (projS S (spanX (col fix r) Xs)) : ℝ) ^ Ys.card :=
        add_le_add (le_refl _) hiter
    _ = ∑ r, min 1 (g r) := by
        rw [← Finset.sum_add_distrib]
        refine Finset.sum_congr rfl fun r _ => ?_
        by_cases hg : 1 ≤ g r
        · simp only [h, hg, ↓reduceIte, mul_zero, zero_div, add_zero, min_eq_left hg]
        · simp only [h, hg, ↓reduceIte, mul_one, zero_add]
          rw [min_eq_right (le_of_not_ge hg)]
    _ ≤ _ := by
        refine Finset.sum_le_sum fun r _ => min_le_min_left _ ?_
        have hc : (0 : ℝ) < csz (projS S (spanX (col fix r) Xs)) := by exact_mod_cast csz_pos _
        have hle : (NS τ S (spanX (col fix r) Xs) : ℝ) * csz (projS S (spanX (col fix r) Xs)) ≤
            (Fintype.card F : ℝ) ^ S.card := by exact_mod_cast NS_mul_le τ hz S (spanX (col fix r) Xs)
        simp only [g]
        rw [pow_succ, div_le_div_iff₀ (by positivity) (by positivity)]
        calc (NS τ S (spanX (col fix r) Xs) : ℝ) *
              ((csz (projS S (spanX (col fix r) Xs)) : ℝ) ^ Ys.card * csz (projS S (spanX (col fix r) Xs)))
            = (NS τ S (spanX (col fix r) Xs) : ℝ) * csz (projS S (spanX (col fix r) Xs)) *
              (csz (projS S (spanX (col fix r) Xs)) : ℝ) ^ Ys.card := by ring
          _ ≤ (Fintype.card F : ℝ) ^ S.card * (csz (projS S (spanX (col fix r) Xs)) : ℝ) ^ Ys.card :=
            mul_le_mul_of_nonneg_right hle (by positivity)

/-- **The rank bound's reduction to the spans' sizes**: the coins at which the points are rank-deficient number at
most `Σ_{S ≠ ∅} Σ_r min(1, |F|^|S|/|π_S U_{X_S}|^(|X_Sᶜ|+1))`, for any choice of coordinates `X_S` per support off
which every point draws freshly. -/
theorem card_deficient_le (τ : F →+ ZMod 2) {z : F} (hz : τ z = 1) (fix : ι → Fin D → Option F)
    (Xs : Finset ι → Finset (Fin D)) (hfree : ∀ S, ∀ t ∉ Xs S, ∀ i, fix i t = none) :
    ((univ.filter fun r : ι → Fin D → F => span (col fix r) ≠ ⊤).card : ℝ) ≤
      ∑ S ∈ univ.filter (fun S : Finset ι => S.Nonempty), ∑ r,
        min 1 ((Fintype.card F : ℝ) ^ S.card /
          (csz (projS S (spanX (col fix r) (Xs S))) : ℝ) ^ ((Xs S)ᶜ.card + 1)) := by
  have hpt : ∀ r : ι → Fin D → F, (if span (col fix r) ≠ ⊤ then (1 : ℝ) else 0) ≤
      ∑ S ∈ univ.filter (fun S : Finset ι => S.Nonempty),
        (if 1 ≤ NS τ S (spanX (col fix r) univ) then (1 : ℝ) else 0) := by
    intro r
    split
    · rename_i hne
      obtain ⟨μ, hμ0, hμ⟩ := exists_ann τ hz hne
      have hS : (supp μ).Nonempty := by
        by_contra hc
        rw [Finset.not_nonempty_iff_eq_empty] at hc
        exact hμ0 (funext fun i => by
          by_contra hi
          have : i ∈ supp μ := Finset.mem_filter.2 ⟨mem_univ _, hi⟩
          rw [hc] at this
          exact absurd this (Finset.notMem_empty i))
      have h1 : 1 ≤ NS τ (supp μ) (spanX (col fix r) univ) :=
        Finset.card_pos.2 ⟨μ, Finset.mem_filter.2 ⟨mem_univ _, rfl, hμ⟩⟩
      refine le_trans ?_ (Finset.single_le_sum (f := fun S =>
        if 1 ≤ NS τ S (spanX (col fix r) univ) then (1 : ℝ) else 0)
        (fun S _ => by split <;> norm_num) (Finset.mem_filter.2 ⟨mem_univ _, hS⟩))
      simp [h1]
    · exact Finset.sum_nonneg fun S _ => by split <;> norm_num
  calc ((univ.filter fun r : ι → Fin D → F => span (col fix r) ≠ ⊤).card : ℝ)
      = ∑ r, (if span (col fix r) ≠ ⊤ then (1 : ℝ) else 0) := by
        rw [Finset.sum_boole]
    _ ≤ ∑ r, ∑ S ∈ univ.filter (fun S : Finset ι => S.Nonempty),
          (if 1 ≤ NS τ S (spanX (col fix r) univ) then (1 : ℝ) else 0) := Finset.sum_le_sum fun r _ => hpt r
    _ = ∑ S ∈ univ.filter (fun S : Finset ι => S.Nonempty), ∑ r,
          (if 1 ≤ NS τ S (spanX (col fix r) univ) then (1 : ℝ) else 0) := Finset.sum_comm
    _ ≤ _ := Finset.sum_le_sum fun S _ =>
        sum_ns_le τ hz S fix (Xs S) (Xs S)ᶜ (Finset.union_compl _) disjoint_compl_right
          (fun t ht => hfree S t (Finset.mem_compl.1 ht))

end DualIter

end FlockComplete.MaskRank
