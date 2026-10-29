import Proofs.Flock.Claims.Complete.Zk.RankDual

/-!
# The rank bound's primal side

`RankDual` bounds the deficient coins by `E[min(1, |F|^|S| / |π_S U_X|^(|Xᶜ| + 1))]` per support `S`. This file
bounds the spans' sizes. One coordinate takes a span `V` to `V + a·V`, of size `|V|²/K(V, a)`, where `K(V, a)` counts
the points of `V` that `a` keeps in `V` (`csz_sup_mul_Kc`).

* `sum_Kc_le`, `sum_supp_le`, `sum_Kc_sub_le`: the expected excess `E_a[K − 1]`, by the points' supports.
* `min_env`, `sum_env_le`: one step's `E_a[min(1, |F|^|S|/|V + a·V|^p)]`, enveloped by that excess.
* `sum_step`: a step as a sum over one column of coins, the fixed entries at their constants (`colA`).
* `sum_row_le`, `sum_mul_rows`, `chain`: over two free rows, `E[1/(|π_i U_T|^α |π_j U_T|^β)] ≤ Bc(T, α)·Bc(T, β)`.
* `csz_spanX_pre_le`: `|U_T| ≤ 2^(2^T)`.
-/

namespace FlockComplete.MaskRank

open Finset

section Primal

open Classical

variable {ι F : Type} [Fintype ι] [DecidableEq ι] [Field F] [Fintype F] [DecidableEq F]

/-- **The points `a` keeps**: the `w ∈ V` with `a·w ∈ V`. -/
noncomputable def Kc (V : AddSubgroup (ι → F)) (a : ι → F) : ℕ :=
  (univ.filter fun w => w ∈ V ∧ a * w ∈ V).card

theorem mem_sup_mul {V : AddSubgroup (ι → F)} {a z : ι → F} :
    z ∈ V ⊔ V.map (AddMonoidHom.mulLeft a) ↔ ∃ v ∈ V, ∃ w ∈ V, v + a * w = z := by
  rw [AddSubgroup.mem_sup]
  constructor
  · rintro ⟨v, hv, _, ⟨w, hw, rfl⟩, rfl⟩
    exact ⟨v, hv, w, hw, rfl⟩
  · rintro ⟨v, hv, w, hw, rfl⟩
    exact ⟨v, hv, a * w, ⟨w, hw, rfl⟩, rfl⟩

/-- **One coordinate's step in size**: `|V + a·V| · K(V, a) = |V|²`, the pairs `(v, w)` of `V` with `v + a·w = z` a
coset of those with `v = a·w`. -/
theorem csz_sup_mul_Kc (V : AddSubgroup (ι → F)) (a : ι → F) :
    csz (V ⊔ V.map (AddMonoidHom.mulLeft a)) * Kc V a = csz V ^ 2 := by
  set PV := univ.filter (· ∈ V) with hPV
  let φ : (ι → F) × (ι → F) → ι → F := fun p => p.1 + a * p.2
  have hcard : (PV ×ˢ PV).card = csz V ^ 2 := by rw [Finset.card_product, sq]; rfl
  have himg : (PV ×ˢ PV).image φ = univ.filter (· ∈ V ⊔ V.map (AddMonoidHom.mulLeft a)) := by
    ext z
    simp only [Finset.mem_image, Finset.mem_product, Finset.mem_filter, Finset.mem_univ, true_and, hPV, mem_sup_mul,
      φ, Prod.exists]
    constructor
    · rintro ⟨v, w, ⟨hv, hw⟩, rfl⟩
      exact ⟨v, hv, w, hw, rfl⟩
    · rintro ⟨v, hv, w, hw, rfl⟩
      exact ⟨v, w, ⟨hv, hw⟩, rfl⟩
  have hfib : ∀ z ∈ (PV ×ˢ PV).image φ, ((PV ×ˢ PV).filter fun p => φ p = z).card = Kc V a := by
    intro z hz
    obtain ⟨⟨v₀, w₀⟩, hp₀, rfl⟩ := Finset.mem_image.1 hz
    simp only [Finset.mem_product, hPV, Finset.mem_filter, Finset.mem_univ, true_and] at hp₀
    refine Finset.card_nbij' (fun p => p.2 - w₀) (fun w => (v₀ - a * w, w + w₀)) ?_ ?_ ?_ ?_
    · rintro ⟨v, w⟩ hp
      simp only [Finset.mem_coe, Finset.mem_filter, Finset.mem_product, hPV, Finset.mem_univ, true_and, φ] at hp ⊢
      obtain ⟨⟨hv, hw⟩, he⟩ := hp
      refine ⟨V.sub_mem hw hp₀.2, ?_⟩
      have : a * (w - w₀) = v₀ - v := by linear_combination he
      rw [this]
      exact V.sub_mem hp₀.1 hv
    · intro w hw
      simp only [Finset.mem_coe, Finset.mem_filter, Finset.mem_product, hPV, Finset.mem_univ, true_and, φ] at hw ⊢
      exact ⟨⟨V.sub_mem hp₀.1 hw.2, V.add_mem hw.1 hp₀.2⟩, by ring⟩
    · rintro ⟨v, w⟩ hp
      simp only [Finset.mem_coe, Finset.mem_filter, Finset.mem_product, hPV, Finset.mem_univ, true_and, φ] at hp
      simp only [sub_add_cancel, Prod.mk.injEq, and_true]
      linear_combination -hp.2
    · intro w _
      simp
  rw [← hcard, Finset.card_eq_sum_card_image φ (PV ×ˢ PV), Finset.sum_congr rfl hfib, Finset.sum_const,
    smul_eq_mul, himg]
  rfl

theorem Kc_le (V : AddSubgroup (ι → F)) (a : ι → F) : Kc V a ≤ csz V :=
  Finset.card_le_card fun w hw => by
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hw ⊢
    exact hw.1

theorem one_le_Kc (V : AddSubgroup (ι → F)) (a : ι → F) : 1 ≤ Kc V a :=
  Finset.card_pos.2 ⟨0, by simp [V.zero_mem]⟩

theorem Kc_pos (V : AddSubgroup (ι → F)) (a : ι → F) : 0 < Kc V a := one_le_Kc V a

/-- The step's size from the kept points. -/
theorem csz_sup_mul_eq (V : AddSubgroup (ι → F)) (a : ι → F) :
    (csz (V ⊔ V.map (AddMonoidHom.mulLeft a)) : ℝ) = (csz V : ℝ) ^ 2 / Kc V a := by
  have h := csz_sup_mul_Kc V a
  have hk : (0 : ℝ) < Kc V a := by exact_mod_cast Kc_pos V a
  rw [eq_div_iff hk.ne']
  exact_mod_cast h

/-- **The kernel bound**: the points of `V` vanishing on `S'` and `V`'s projection to `S'` multiply to at most
`|V|`. -/
theorem csz_ker_mul_le (V : AddSubgroup (ι → F)) (S' : Finset ι) :
    csz (V ⊓ coordSub S'ᶜ) * csz (projS S' V) ≤ csz V := by
  set PV := univ.filter (· ∈ V) with hPV
  have himg : PV.image (zS S') = univ.filter (· ∈ projS S' V) := by
    ext z
    simp only [Finset.mem_image, Finset.mem_filter, Finset.mem_univ, true_and, hPV, projS, AddSubgroup.mem_map]
  have hfib : ∀ z ∈ PV.image (zS S'), csz (V ⊓ coordSub S'ᶜ) ≤ (PV.filter fun v => zS S' v = z).card := by
    intro z hz
    obtain ⟨v₀, hv₀, rfl⟩ := Finset.mem_image.1 hz
    simp only [hPV, Finset.mem_filter, Finset.mem_univ, true_and] at hv₀
    refine Finset.card_le_card_of_injOn (fun w => w + v₀) ?_ ?_
    · intro w hw
      simp only [Finset.mem_coe, Finset.mem_filter, Finset.mem_univ, true_and, AddSubgroup.mem_inf, hPV] at hw ⊢
      refine ⟨V.add_mem hw.1 hv₀, ?_⟩
      rw [map_add]
      have : zS S' w = 0 := by
        funext i
        by_cases hi : i ∈ S'
        · simp only [zS, AddMonoidHom.coe_mk, ZeroHom.coe_mk, hi, ↓reduceIte, Pi.zero_apply]
          exact hw.2 i (by simpa using hi)
        · simp [zS, hi]
      rw [this, zero_add]
    · intro w _ w' _ h
      simpa using h
  calc csz (V ⊓ coordSub S'ᶜ) * csz (projS S' V)
      = csz (V ⊓ coordSub S'ᶜ) * (PV.image (zS S')).card := by rw [himg]; rfl
    _ = ∑ _z ∈ PV.image (zS S'), csz (V ⊓ coordSub S'ᶜ) := by rw [Finset.sum_const, smul_eq_mul, mul_comm]
    _ ≤ ∑ z ∈ PV.image (zS S'), (PV.filter fun v => zS S' v = z).card := Finset.sum_le_sum hfib
    _ = csz V := (Finset.card_eq_sum_card_image (zS S') PV).symm

/-- **The coins keeping one point**: `a·w ∈ V` for at most `|V ∩ F^{supp w}| · |F|^{|ι| − |supp w|}` of them,
the map `a ↦ a·w` onto `F^{supp w}` with fibers of `|F|^{|ι| − |supp w|}`. -/
theorem card_mul_mem_le (V : AddSubgroup (ι → F)) (w : ι → F) :
    (univ.filter fun a : ι → F => a * w ∈ V).card ≤
      csz (V ⊓ coordSub (supp w)) * Fintype.card F ^ (supp w)ᶜ.card := by
  set P := univ.filter fun a : ι → F => a * w ∈ V with hP
  have hfib : ∀ z ∈ P.image (· * w), (P.filter fun a => a * w = z).card ≤ Fintype.card F ^ (supp w)ᶜ.card := by
    intro z hz
    obtain ⟨a₀, -, rfl⟩ := Finset.mem_image.1 hz
    rw [← card_coordSub (F := F) (supp w)ᶜ]
    refine Finset.card_le_card_of_injOn (fun a => a - a₀) ?_ ?_
    · intro a ha
      simp only [Finset.mem_coe, Finset.mem_filter, Finset.mem_univ, true_and, hP] at ha ⊢
      intro i hi
      have hwi : w i ≠ 0 := by simpa [supp, Finset.mem_compl] using hi
      have := congrFun ha.2 i
      simp only [Pi.mul_apply] at this
      simp only [Pi.sub_apply]
      exact sub_eq_zero.2 (mul_right_cancel₀ hwi this)
    · intro a _ b _ h
      simpa using h
  calc P.card ≤ Fintype.card F ^ (supp w)ᶜ.card * (P.image (· * w)).card := Finset.card_le_mul_card_image P _ hfib
    _ ≤ Fintype.card F ^ (supp w)ᶜ.card * csz (V ⊓ coordSub (supp w)) := by
        gcongr
        refine Finset.card_le_card fun z hz => ?_
        obtain ⟨a, ha, rfl⟩ := Finset.mem_image.1 hz
        simp only [hP, Finset.mem_filter, Finset.mem_univ, true_and, AddSubgroup.mem_inf] at ha ⊢
        refine ⟨ha, fun i hi => ?_⟩
        have : w i = 0 := by simpa [supp] using hi
        simp [this]
    _ = _ := mul_comm _ _

/-- The kept points summed over the coins, as each point's keeping coins. -/
theorem sum_Kc (V : AddSubgroup (ι → F)) :
    ∑ a, Kc V a = ∑ w ∈ univ.filter (· ∈ V), (univ.filter fun a : ι → F => a * w ∈ V).card := by
  simp only [Kc, Finset.card_filter]
  rw [Finset.sum_comm, Finset.sum_filter]
  refine Finset.sum_congr rfl fun w _ => ?_
  by_cases hw : w ∈ V <;> simp [hw]

/-- **The expected count of kept points**: `E_a[K(V, a)] ≤ Σ_{w ∈ V} |V ∩ F^{supp w}| / |F|^{|supp w|}`. -/
theorem sum_Kc_le (V : AddSubgroup (ι → F)) :
    ∑ a, (Kc V a : ℝ) ≤ Fintype.card (ι → F) *
      ∑ w ∈ univ.filter (· ∈ V), (csz (V ⊓ coordSub (supp w)) : ℝ) / Fintype.card F ^ (supp w).card := by
  have hq : (0 : ℝ) < Fintype.card F := by exact_mod_cast Fintype.card_pos
  rw [Finset.mul_sum]
  have h : (∑ a, (Kc V a : ℝ)) = ((∑ a, Kc V a : ℕ) : ℝ) := by push_cast; rfl
  rw [h, sum_Kc V]
  push_cast
  refine Finset.sum_le_sum fun w _ => ?_
  have hA : (Fintype.card (ι → F) : ℝ) =
      (Fintype.card F : ℝ) ^ (supp w).card * Fintype.card F ^ (supp w)ᶜ.card := by
    rw [← pow_add, Finset.card_add_card_compl, Fintype.card_fun]
    push_cast
    ring
  have h' : ((univ.filter fun a : ι → F => a * w ∈ V).card : ℝ) ≤
      csz (V ⊓ coordSub (supp w)) * (Fintype.card F : ℝ) ^ (supp w)ᶜ.card := by
    exact_mod_cast card_mul_mem_le V w
  refine h'.trans (le_of_eq ?_)
  rw [hA]
  field_simp

end Primal

section Steps

open Classical

variable {ι F : Type} [Fintype ι] [DecidableEq ι] [Field F] [Fintype F] [DecidableEq F]

/-- **A power of the kept count, linearly**: `K^p ≤ 1 + 2·u^(p−1)·(K − 1)` for `1 ≤ K ≤ u`. -/
theorem pow_le_lin {K u : ℕ} (h1 : 1 ≤ K) (hu : K ≤ u) {p : ℕ} (hp : 1 ≤ p) :
    (K : ℝ) ^ p ≤ 1 + 2 * (u : ℝ) ^ (p - 1) * ((K : ℝ) - 1) := by
  rcases Nat.lt_or_ge K 2 with hK | hK
  · have : K = 1 := by omega
    subst this
    simp
  · have hK' : (2 : ℝ) ≤ K := by exact_mod_cast hK
    have hKu : (K : ℝ) ≤ u := by exact_mod_cast hu
    have hK0 : (0 : ℝ) ≤ K := by positivity
    have hu0 : (0 : ℝ) ≤ (u : ℝ) ^ (p - 1) := by positivity
    calc (K : ℝ) ^ p = K * K ^ (p - 1) := by rw [← pow_succ', Nat.sub_add_cancel hp]
      _ ≤ K * u ^ (p - 1) := by gcongr
      _ ≤ 2 * ((K : ℝ) - 1) * u ^ (p - 1) := by gcongr; linarith
      _ = 2 * (u : ℝ) ^ (p - 1) * ((K : ℝ) - 1) := by ring
      _ ≤ _ := by linarith

/-- **The envelope**: `min(1, α·K^p) ≤ min(1, α) + 2·(K − 1)/M` for a natural `K ≥ 1`, when `α·M^p ≤ 1`. -/
theorem min_env {α M : ℝ} {p K : ℕ} (hα : 0 ≤ α) (hM : 0 < M) (hαM : α * M ^ p ≤ 1) (hp : 1 ≤ p)
    (hK : 1 ≤ K) :
    min 1 (α * (K : ℝ) ^ p) ≤ min 1 α + 2 * ((K : ℝ) - 1) / M := by
  rcases Nat.lt_or_ge K 2 with h | h
  · have : K = 1 := by omega
    subst this
    simp
  · have hK2 : (2 : ℝ) ≤ K := by exact_mod_cast h
    have hm : 0 ≤ min 1 α := le_min zero_le_one hα
    have key : min 1 (α * (K : ℝ) ^ p) ≤ K / M := by
      rcases le_or_gt M K with hMK | hMK
      · exact (min_le_left _ _).trans ((one_le_div hM).2 hMK)
      · refine (min_le_right _ _).trans ?_
        rw [le_div_iff₀ hM]
        have hp' : M ^ p = M ^ (p - 1) * M := by rw [← pow_succ, Nat.sub_add_cancel hp]
        have hK0 : (0 : ℝ) ≤ K := by positivity
        calc α * (K : ℝ) ^ p * M = α * (K * K ^ (p - 1)) * M := by rw [← pow_succ', Nat.sub_add_cancel hp]
          _ ≤ α * (K * M ^ (p - 1)) * M := by gcongr
          _ = K * (α * M ^ p) := by rw [hp']; ring
          _ ≤ K * 1 := by gcongr
          _ = K := mul_one _
    calc _ ≤ (K : ℝ) / M := key
      _ ≤ 2 * ((K : ℝ) - 1) / M := by gcongr; linarith
      _ ≤ _ := le_add_of_nonneg_left hm

theorem zS_mul (S : Finset ι) (c u : ι → F) : zS S (c * u) = c * zS S u := by
  funext i
  by_cases h : i ∈ S <;> simp [zS, h]

/-- **The projection commutes with the step**: `π_S(U + c·U) = π_S U + c·π_S U`. -/
theorem projS_sup_mul (S : Finset ι) (U : AddSubgroup (ι → F)) (c : ι → F) :
    projS S (U ⊔ U.map (AddMonoidHom.mulLeft c)) = projS S U ⊔ (projS S U).map (AddMonoidHom.mulLeft c) := by
  unfold projS
  rw [AddSubgroup.map_sup, AddSubgroup.map_map, AddSubgroup.map_map]
  congr 2
  exact AddMonoidHom.ext fun u => zS_mul S c u

theorem mul_coordSub {S : Finset ι} {c c' : ι → F} (h : ∀ i ∈ S, c i = c' i) {v : ι → F}
    (hv : v ∈ coordSub S) : c * v = c' * v := by
  funext i
  by_cases hi : i ∈ S
  · simp [h i hi]
  · simp [hv i hi]

theorem map_mul_congr {S : Finset ι} {V : AddSubgroup (ι → F)} (hV : V ≤ coordSub S) {c c' : ι → F}
    (h : ∀ i ∈ S, c i = c' i) : V.map (AddMonoidHom.mulLeft c) = V.map (AddMonoidHom.mulLeft c') := by
  ext z
  simp only [AddSubgroup.mem_map, AddMonoidHom.coe_mulLeft]
  constructor
  · rintro ⟨v, hv, rfl⟩
    exact ⟨v, hv, (mul_coordSub h (hV hv)).symm⟩
  · rintro ⟨v, hv, rfl⟩
    exact ⟨v, hv, mul_coordSub h (hV hv)⟩

theorem csz_map_le (V : AddSubgroup (ι → F)) (f : (ι → F) →+ (ι → F)) : csz (V.map f) ≤ csz V := by
  unfold csz
  calc (univ.filter (· ∈ V.map f)).card = ((univ.filter (· ∈ V)).image f).card := by
        congr 1
        ext z
        simp [AddSubgroup.mem_map]
    _ ≤ _ := Finset.card_image_le

theorem zS_zS {S' S : Finset ι} (h : S' ⊆ S) (u : ι → F) : zS S' (zS S u) = zS S' u := by
  funext i
  by_cases hi : i ∈ S'
  · simp [zS, hi, h hi]
  · simp [zS, hi]

theorem projS_projS {S' S : Finset ι} (h : S' ⊆ S) (U : AddSubgroup (ι → F)) :
    projS S' (projS S U) = projS S' U := by
  unfold projS
  rw [AddSubgroup.map_map]
  congr 1
  exact AddMonoidHom.ext fun u => zS_zS h u

theorem csz_projS_mono {S' S : Finset ι} (h : S' ⊆ S) (U : AddSubgroup (ι → F)) :
    csz (projS S' U) ≤ csz (projS S U) := by
  rw [← projS_projS h U]
  exact csz_map_le _ _

theorem csz_projS_le (S : Finset ι) (U : AddSubgroup (ι → F)) : csz (projS S U) ≤ csz U := csz_map_le _ _

/-- The points of `V` vanishing at `e`. -/
noncomputable def kerE (V : AddSubgroup (ι → F)) (e : ι) : ℕ := csz (V ⊓ coordSub ({e} : Finset ι)ᶜ)

theorem kerE_mul_le (V : AddSubgroup (ι → F)) (e : ι) : kerE V e * csz (projS {e} V) ≤ csz V := by
  unfold kerE
  exact csz_ker_mul_le V {e}

theorem one_le_kerE (V : AddSubgroup (ι → F)) (e : ι) : 1 ≤ kerE V e := csz_pos _

/-- **The kept count by supports**: with every nonzero point of `V ≤ F^S` of support at least `m ≥ 1`, the nonzero
points off full support each vanish at some `e ∈ S` and so lie among `kerE V e`'s, each of
`|V ∩ F^{supp}| ≤ kerE V e` over `|F|^m`; the full-support ones number at most `|V|`, each `|V|/|F|^|S|`. -/
theorem sum_supp_le {S : Finset ι} {V : AddSubgroup (ι → F)} (hV : V ≤ coordSub S) {m : ℕ}
    (hsupp : ∀ w ∈ V, w ≠ 0 → m ≤ (supp w).card) :
    ∑ w ∈ univ.filter (· ∈ V), (csz (V ⊓ coordSub (supp w)) : ℝ) / Fintype.card F ^ (supp w).card ≤
      1 + ∑ e ∈ S, ((kerE V e : ℝ) - 1) * kerE V e / Fintype.card F ^ m +
        (csz V : ℝ) ^ 2 / Fintype.card F ^ S.card := by
  set q : ℝ := (Fintype.card F : ℝ) with hq
  have hq1 : (1 : ℝ) ≤ q := by rw [hq]; exact_mod_cast (Fintype.card_pos : 0 < Fintype.card F)
  set PV := univ.filter (· ∈ V) with hPV
  have h0 : (0 : ι → F) ∈ PV := by simp [hPV, V.zero_mem]
  have hsupp0 : supp (0 : ι → F) = ∅ := by ext i; simp [supp]
  have hcsz0 : csz (V ⊓ coordSub (∅ : Finset ι)) = 1 := by
    unfold csz
    rw [Finset.card_eq_one]
    refine ⟨0, ?_⟩
    ext w
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, AddSubgroup.mem_inf, Finset.mem_singleton]
    constructor
    · rintro ⟨-, hw⟩
      funext i
      exact hw i (Finset.notMem_empty i)
    · rintro rfl
      exact ⟨V.zero_mem, (coordSub ∅).zero_mem⟩
  rw [← Finset.add_sum_erase PV _ h0, hsupp0, hcsz0]
  simp only [Finset.card_empty, pow_zero, Nat.cast_one, div_one]
  rw [add_assoc]
  refine add_le_add (le_refl _) ?_
  have hpt : ∀ w ∈ PV.erase 0, (csz (V ⊓ coordSub (supp w)) : ℝ) / q ^ (supp w).card ≤
      ∑ e ∈ S, (if w e = 0 then (kerE V e : ℝ) / q ^ m else 0) + (csz V : ℝ) / q ^ S.card := by
    intro w hw
    obtain ⟨hw0, hwV⟩ := Finset.mem_erase.1 hw
    simp only [hPV, Finset.mem_filter, Finset.mem_univ, true_and] at hwV
    have hsub : supp w ⊆ S := fun i hi => by
      by_contra h
      exact (Finset.mem_filter.1 hi).2 (hV hwV i h)
    have hs : 0 ≤ ∑ e ∈ S, (if w e = 0 then (kerE V e : ℝ) / q ^ m else 0) :=
      Finset.sum_nonneg fun e _ => by split <;> positivity
    by_cases hfull : supp w = S
    · rw [hfull]
      have : (csz (V ⊓ coordSub S) : ℝ) ≤ csz V := by exact_mod_cast csz_mono inf_le_left
      calc _ ≤ (csz V : ℝ) / q ^ S.card := by gcongr
        _ ≤ _ := le_add_of_nonneg_left hs
    · obtain ⟨e, heS, he⟩ : ∃ e ∈ S, e ∉ supp w := by
        by_contra h
        push Not at h
        exact hfull (Finset.Subset.antisymm hsub h)
      have hwe : w e = 0 := by simpa [supp] using he
      have hmono : coordSub (supp w) ≤ (coordSub ({e} : Finset ι)ᶜ : AddSubgroup (ι → F)) := by
        intro a ha i hi
        refine ha i fun h => ?_
        simp only [Finset.mem_compl, Finset.mem_singleton, not_not] at hi
        exact he (hi ▸ h)
      have hle : (csz (V ⊓ coordSub (supp w)) : ℝ) ≤ kerE V e := by
        unfold kerE
        exact_mod_cast csz_mono (inf_le_inf_left _ hmono)
      have hpow : q ^ m ≤ q ^ (supp w).card := pow_le_pow_right₀ hq1 (hsupp w hwV hw0)
      have hqm : 0 < q ^ m := by positivity
      have hB : 0 ≤ (csz V : ℝ) / q ^ S.card := by positivity
      calc (csz (V ⊓ coordSub (supp w)) : ℝ) / q ^ (supp w).card ≤ (kerE V e : ℝ) / q ^ m := by
            gcongr
        _ = if w e = 0 then (kerE V e : ℝ) / q ^ m else 0 := by rw [if_pos hwe]
        _ ≤ ∑ e ∈ S, (if w e = 0 then (kerE V e : ℝ) / q ^ m else 0) :=
            Finset.single_le_sum (f := fun e => if w e = 0 then (kerE V e : ℝ) / q ^ m else 0)
              (fun e _ => by split <;> positivity) heS
        _ ≤ _ := le_add_of_nonneg_right hB
  have hcnt : ∀ e, (((PV.erase 0).filter fun w => w e = 0).card : ℝ) ≤ (kerE V e : ℝ) - 1 := by
    intro e
    have hsub : (PV.erase 0).filter (fun w => w e = 0) ⊆
        (univ.filter (· ∈ V ⊓ (coordSub ({e} : Finset ι)ᶜ : AddSubgroup (ι → F)))).erase 0 := by
      intro w hw
      simp only [Finset.mem_filter, Finset.mem_erase, hPV, Finset.mem_univ, true_and] at hw
      simp only [Finset.mem_erase, Finset.mem_filter, Finset.mem_univ, true_and, AddSubgroup.mem_inf]
      refine ⟨hw.1.1, hw.1.2, fun i hi => ?_⟩
      simp only [Finset.mem_compl, Finset.mem_singleton, not_not] at hi
      rw [hi]
      exact hw.2
    have h1 := Finset.card_le_card hsub
    have h0' : (0 : ι → F) ∈
        univ.filter (· ∈ V ⊓ (coordSub ({e} : Finset ι)ᶜ : AddSubgroup (ι → F))) := by
      simp [(V ⊓ coordSub ({e} : Finset ι)ᶜ).zero_mem]
    rw [Finset.card_erase_of_mem h0'] at h1
    have h2 : 1 ≤ kerE V e := one_le_kerE V e
    have : ((((PV.erase 0).filter fun w => w e = 0).card : ℕ) : ℝ) ≤ ((kerE V e - 1 : ℕ) : ℝ) := by
      exact_mod_cast h1
    rw [Nat.cast_sub h2, Nat.cast_one] at this
    exact this
  have hPV' : ((PV.erase 0).card : ℝ) ≤ csz V := by
    exact_mod_cast (Finset.card_erase_le).trans (le_of_eq rfl)
  calc ∑ w ∈ PV.erase 0, (csz (V ⊓ coordSub (supp w)) : ℝ) / q ^ (supp w).card
      ≤ ∑ w ∈ PV.erase 0, (∑ e ∈ S, (if w e = 0 then (kerE V e : ℝ) / q ^ m else 0) +
          (csz V : ℝ) / q ^ S.card) := Finset.sum_le_sum hpt
    _ = ∑ e ∈ S, (((PV.erase 0).filter fun w => w e = 0).card : ℝ) * ((kerE V e : ℝ) / q ^ m) +
          ((PV.erase 0).card : ℝ) * ((csz V : ℝ) / q ^ S.card) := by
        rw [Finset.sum_add_distrib, Finset.sum_comm, Finset.sum_const, nsmul_eq_mul]
        congr 1
        refine Finset.sum_congr rfl fun e _ => ?_
        rw [Finset.sum_ite, Finset.sum_const_zero, add_zero, Finset.sum_const, nsmul_eq_mul]
    _ ≤ ∑ e ∈ S, ((kerE V e : ℝ) - 1) * ((kerE V e : ℝ) / q ^ m) +
          (csz V : ℝ) * ((csz V : ℝ) / q ^ S.card) := by
        gcongr with e
        exact hcnt e
    _ = _ := by
        congr 1
        · refine Finset.sum_congr rfl fun e _ => ?_
          ring
        · ring

end Steps

section Chain

open Classical

variable {ι F : Type} [Fintype ι] [DecidableEq ι] [Field F] [Fintype F] [DecidableEq F]

/-- **The expected excess of kept points**: `E_a[K(V, a) − 1]` by supports (`sum_Kc_le`, `sum_supp_le`). -/
theorem sum_Kc_sub_le {S : Finset ι} {V : AddSubgroup (ι → F)} (hV : V ≤ coordSub S) {m : ℕ}
    (hsupp : ∀ w ∈ V, w ≠ 0 → m ≤ (supp w).card) :
    ∑ a, ((Kc V a : ℝ) - 1) ≤ Fintype.card (ι → F) *
      (∑ e ∈ S, ((kerE V e : ℝ) - 1) * kerE V e / Fintype.card F ^ m +
        (csz V : ℝ) ^ 2 / Fintype.card F ^ S.card) := by
  have h1 := sum_Kc_le V
  have h2 := sum_supp_le hV hsupp
  have hA : (0 : ℝ) ≤ Fintype.card (ι → F) := Nat.cast_nonneg _
  have h3 := mul_le_mul_of_nonneg_left h2 hA
  rw [Finset.sum_sub_distrib, Finset.sum_const, Finset.card_univ, nsmul_eq_mul, mul_one]
  rw [mul_add, mul_add, mul_one] at h3
  linarith

/-- **One step's survival, enveloped**: with `α = |F|^|S|/|V|^(2p)` and `α·M^p ≤ 1`, the step's
`E_a[min(1, |F|^|S|/|V + a·V|^p)]` is at most `min(1, α)` plus `2/M` times the expected excess. -/
theorem sum_env_le {S : Finset ι} {V : AddSubgroup (ι → F)} (hV : V ≤ coordSub S) {m : ℕ}
    (hsupp : ∀ w ∈ V, w ≠ 0 → m ≤ (supp w).card) {p : ℕ} (hp : 1 ≤ p) {M : ℝ} (hM : 0 < M)
    (hαM : (Fintype.card F : ℝ) ^ S.card / (csz V : ℝ) ^ (2 * p) * M ^ p ≤ 1) :
    ∑ a, min 1 ((Fintype.card F : ℝ) ^ S.card / (csz (V ⊔ V.map (AddMonoidHom.mulLeft a)) : ℝ) ^ p) ≤
      Fintype.card (ι → F) * (min 1 ((Fintype.card F : ℝ) ^ S.card / (csz V : ℝ) ^ (2 * p)) +
        2 / M * (∑ e ∈ S, ((kerE V e : ℝ) - 1) * kerE V e / Fintype.card F ^ m +
          (csz V : ℝ) ^ 2 / Fintype.card F ^ S.card)) := by
  set α := (Fintype.card F : ℝ) ^ S.card / (csz V : ℝ) ^ (2 * p) with hα
  have hα0 : 0 ≤ α := by positivity
  have hv : (0 : ℝ) < csz V := by exact_mod_cast csz_pos V
  have hpt : ∀ a, min 1 ((Fintype.card F : ℝ) ^ S.card / (csz (V ⊔ V.map (AddMonoidHom.mulLeft a)) : ℝ) ^ p) ≤
      min 1 α + 2 * ((Kc V a : ℝ) - 1) / M := by
    intro a
    have hK : (0 : ℝ) < Kc V a := by exact_mod_cast Kc_pos V a
    have e : (Fintype.card F : ℝ) ^ S.card / (csz (V ⊔ V.map (AddMonoidHom.mulLeft a)) : ℝ) ^ p =
        α * (Kc V a : ℝ) ^ p := by
      rw [csz_sup_mul_eq, hα, div_pow, ← pow_mul]
      field_simp
    rw [e]
    exact min_env hα0 hM hαM hp (one_le_Kc V a)
  have hE := sum_Kc_sub_le hV hsupp
  have hM2 : 0 ≤ 2 / M := by positivity
  calc _ ≤ ∑ a : ι → F, (min 1 α + 2 * ((Kc V a : ℝ) - 1) / M) := Finset.sum_le_sum fun a _ => hpt a
    _ = Fintype.card (ι → F) * min 1 α + 2 / M * ∑ a, ((Kc V a : ℝ) - 1) := by
        rw [Finset.sum_add_distrib, Finset.sum_const, Finset.card_univ, nsmul_eq_mul, Finset.mul_sum]
        congr 1
        exact Finset.sum_congr rfl fun a _ => by ring
    _ ≤ Fintype.card (ι → F) * min 1 α + 2 / M * (Fintype.card (ι → F) *
          (∑ e ∈ S, ((kerE V e : ℝ) - 1) * kerE V e / Fintype.card F ^ m +
            (csz V : ℝ) ^ 2 / Fintype.card F ^ S.card)) := by
        gcongr
    _ = _ := by ring

/-- **One row's kept points**: at a constant coin `y`, the nonzero `w` keep at most `|V|` coins each, so the
expected excess is at most `|V|²/|F|`. -/
theorem sum_Kc_row (V : AddSubgroup (ι → F)) :
    ∑ y : F, ((Kc V (fun _ => y) : ℝ) - 1) ≤ (csz V : ℝ) ^ 2 := by
  set PV := univ.filter (· ∈ V) with hPV
  have hsum : ∑ y : F, Kc V (fun _ => y) =
      ∑ w ∈ PV, (univ.filter fun y : F => (fun _ => y) * w ∈ V).card := by
    simp only [Kc, Finset.card_filter]
    rw [Finset.sum_comm, Finset.sum_filter]
    refine Finset.sum_congr rfl fun w _ => ?_
    by_cases hw : w ∈ V <;> simp [hw]
  have h0 : (0 : ι → F) ∈ PV := by simp [hPV, V.zero_mem]
  have hw0 : (univ.filter fun y : F => (fun _ => y) * (0 : ι → F) ∈ V).card = Fintype.card F := by
    rw [Finset.filter_true_of_mem (fun y _ => by simp [V.zero_mem]), Finset.card_univ]
  have hle : ∀ w ∈ PV.erase 0, (univ.filter fun y : F => (fun _ => y) * w ∈ V).card ≤ csz V := by
    intro w hw
    obtain ⟨hw0, -⟩ := Finset.mem_erase.1 hw
    obtain ⟨k, hk⟩ : ∃ k, w k ≠ 0 := by
      by_contra h
      push Not at h
      exact hw0 (funext h)
    refine Finset.card_le_card_of_injOn (fun y => (fun _ => y) * w) ?_ ?_
    · intro y hy
      simp only [Finset.coe_filter, Finset.mem_univ, true_and, Set.mem_setOf_eq] at hy
      simp only [csz, Finset.coe_filter, Finset.mem_univ, true_and, Set.mem_setOf_eq]
      exact hy
    · intro y _ y' _ h
      have := congrFun h k
      simp only [Pi.mul_apply] at this
      exact mul_right_cancel₀ hk this
  have hPV' : (PV.erase 0).card ≤ csz V := Finset.card_erase_le
  have hN : ∑ y : F, Kc V (fun _ => y) ≤ Fintype.card F + csz V * csz V := by
    rw [hsum, ← Finset.add_sum_erase PV _ h0, hw0]
    refine Nat.add_le_add_left ?_ _
    calc ∑ w ∈ PV.erase 0, (univ.filter fun y : F => (fun _ => y) * w ∈ V).card
        ≤ ∑ _w ∈ PV.erase 0, csz V := Finset.sum_le_sum hle
      _ = (PV.erase 0).card * csz V := by rw [Finset.sum_const, smul_eq_mul]
      _ ≤ csz V * csz V := Nat.mul_le_mul_right _ hPV'
  have hR : (∑ y : F, (Kc V (fun _ => y) : ℝ)) ≤ Fintype.card F + (csz V : ℝ) ^ 2 := by
    rw [sq]
    exact_mod_cast hN
  rw [Finset.sum_sub_distrib, Finset.sum_const, Finset.card_univ, nsmul_eq_mul, mul_one]
  linarith

/-- The step's coefficient: `0` at `p = 0`, else `2/q`. -/
noncomputable def cB (q : ℝ) (p : ℕ) : ℝ := if p = 0 then 0 else 2 / q

theorem cB_nonneg {q : ℝ} (hq : 0 ≤ q) (p : ℕ) : 0 ≤ cB q p := by
  unfold cB
  split <;> positivity

/-- One row's bound on `E_y[1/|V + y·V|^p]`: `1/|V|^(2p) + cB(p)/|V|^(p−1)`. -/
noncomputable def rowB (q u : ℝ) (p : ℕ) : ℝ := 1 / u ^ (2 * p) + cB q p * (1 / u ^ (p - 1))

theorem rowB_nonneg {q u : ℝ} (hq : 0 ≤ q) (hu : 0 ≤ u) (p : ℕ) : 0 ≤ rowB q u p := by
  unfold rowB
  have := cB_nonneg hq p
  positivity

/-- **One row's step**: `Σ_y 1/|V + y·V|^p ≤ |F|·rowB`, from `|V + y·V| = |V|²/K` and
`K^p ≤ 1 + 2|V|^(p−1)(K − 1)`. -/
theorem sum_row_le (V : AddSubgroup (ι → F)) (p : ℕ) :
    ∑ y : F, 1 / (csz (V ⊔ V.map (AddMonoidHom.mulLeft (fun _ => y))) : ℝ) ^ p ≤
      Fintype.card F * rowB (Fintype.card F) (csz V) p := by
  have hq : (0 : ℝ) < Fintype.card F := by exact_mod_cast Fintype.card_pos
  rcases Nat.eq_zero_or_pos p with rfl | hp
  · simp [rowB, cB]
  have hu : (0 : ℝ) < csz V := by exact_mod_cast csz_pos V
  set u : ℝ := (csz V : ℝ) with hu_def
  have hpt : ∀ y : F, 1 / (csz (V ⊔ V.map (AddMonoidHom.mulLeft (fun _ => y))) : ℝ) ^ p ≤
      (1 + 2 * u ^ (p - 1) * ((Kc V (fun _ => y) : ℝ) - 1)) / u ^ (2 * p) := by
    intro y
    rw [csz_sup_mul_eq, div_pow, ← pow_mul, one_div_div]
    gcongr
    exact pow_le_lin (one_le_Kc _ _) (Kc_le _ _) hp
  have hrow := sum_Kc_row V
  calc _ ≤ ∑ y : F, (1 + 2 * u ^ (p - 1) * ((Kc V (fun _ => y) : ℝ) - 1)) / u ^ (2 * p) :=
        Finset.sum_le_sum fun y _ => hpt y
    _ = (Fintype.card F + 2 * u ^ (p - 1) * ∑ y : F, ((Kc V (fun _ => y) : ℝ) - 1)) / u ^ (2 * p) := by
        rw [← Finset.sum_div, Finset.sum_add_distrib, ← Finset.mul_sum, Finset.sum_const, Finset.card_univ,
          nsmul_eq_mul, mul_one]
    _ ≤ (Fintype.card F + 2 * u ^ (p - 1) * u ^ 2) / u ^ (2 * p) := by gcongr
    _ = _ := by
        have hp' : p ≠ 0 := by omega
        simp only [rowB, cB, hp', ↓reduceIte]
        have e : u ^ (2 * p) = u ^ (p - 1) * u ^ 2 * u ^ (p - 1) := by
          rw [← pow_add, ← pow_add]
          congr 1
          omega
        rw [e]
        field_simp

/-- **Two rows at once**: a function of the coin's row `i` times one of its row `j ≠ i` sums to the product of
their sums, times `|F|` for each other row. -/
theorem sum_mul_rows {i j : ι} (hij : i ≠ j) (g h : F → ℝ) :
    ∑ a : ι → F, g (a i) * h (a j) =
      (Fintype.card F : ℝ) ^ (Fintype.card ι - 2) * ((∑ y, g y) * ∑ y, h y) := by
  let f : ι → F → ℝ := fun k y => (if k = i then g y else 1) * (if k = j then h y else 1)
  have hL : ∀ a : ι → F, ∏ k, f k (a k) = g (a i) * h (a j) := by
    intro a
    simp only [f, Finset.prod_mul_distrib, Finset.prod_ite_eq', Finset.mem_univ, ite_true]
  have hfi : ∑ y, f i y = ∑ y, g y := by simp [f, hij]
  have hfj : ∑ y, f j y = ∑ y, h y := by simp [f, Ne.symm hij]
  have hj : j ∈ univ.erase i := Finset.mem_erase.2 ⟨Ne.symm hij, mem_univ j⟩
  have hfk : ∀ k ∈ (univ.erase i).erase j, ∑ y, f k y = Fintype.card F := by
    intro k hk
    simp only [Finset.mem_erase, Finset.mem_univ, and_true] at hk
    simp [f, hk.1, hk.2]
  have hcard : ((univ.erase i).erase j).card = Fintype.card ι - 2 := by
    rw [Finset.card_erase_of_mem hj, Finset.card_erase_of_mem (mem_univ i), Finset.card_univ]
    omega
  rw [← Finset.sum_congr rfl (fun a _ => hL a), ← Fintype.prod_sum, ← Finset.mul_prod_erase univ _ (mem_univ i),
    ← Finset.mul_prod_erase _ _ hj, Finset.prod_congr rfl hfk, Finset.prod_const, hcard, hfi, hfj]
  ring

variable {D : ℕ}

/-- The coordinate `t`'s column at the coin `a`, the fixed entries at their constants. -/
def colA (fix : ι → Fin D → Option F) (t : Fin D) (a : ι → F) : ι → F := fun i => (fix i t).getD (a i)

theorem col_setCol (fix : ι → Fin D → Option F) (r : ι → Fin D → F) (t : Fin D) (a : ι → F) :
    col fix (setCol r t a) t = colA fix t a := by
  funext i
  simp [col, setCol, colA]

theorem colA_free {fix : ι → Fin D → Option F} {t : Fin D} (hfree : ∀ i, fix i t = none) (a : ι → F) :
    colA fix t a = a := by
  funext i
  simp [colA, hfree i]

/-- **The column step**: summing over the coins a function of the span with the coordinate `t` is, `|F|^|ι|`
times, summing over the coins and `t`'s column `a` the function at `U + colA(a)·U`. -/
theorem sum_step [CharP F 2] (fix : ι → Fin D → Option F) {Xs : Finset (Fin D)} {t : Fin D} (ht : t ∉ Xs)
    (f : AddSubgroup (ι → F) → ℝ) :
    (Fintype.card (ι → F) : ℝ) * ∑ r, f (spanX (col fix r) (insert t Xs)) =
      ∑ r, ∑ a : ι → F,
        f (spanX (col fix r) Xs ⊔ (spanX (col fix r) Xs).map (AddMonoidHom.mulLeft (colA fix t a))) := by
  have key : ∀ r a, spanX (col fix (setCol r t a)) Xs = spanX (col fix r) Xs := fun r a =>
    spanX_congr fun s hs => col_setCol_ne fix (ne_of_mem_of_not_mem hs ht) r a
  rw [← nsmul_eq_mul, ← sum_setCol t (fun r => f (spanX (col fix r) (insert t Xs)))]
  refine Finset.sum_congr rfl fun r _ => Finset.sum_congr rfl fun a _ => ?_
  rw [spanX_insert _ ht, key, col_setCol]

/-- The first `T` coordinates. -/
def pre (D T : ℕ) : Finset (Fin D) := univ.filter fun t => t.val < T

theorem pre_zero : pre D 0 = ∅ := by
  ext t
  simp [pre]

theorem pre_succ {T : ℕ} (h : T < D) : pre D (T + 1) = insert ⟨T, h⟩ (pre D T) := by
  ext t
  simp only [pre, Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_insert, Fin.ext_iff]
  omega

theorem not_mem_pre {T : ℕ} (h : T < D) : (⟨T, h⟩ : Fin D) ∉ pre D T := by
  simp [pre]

theorem pre_mono {T T' : ℕ} (h : T ≤ T') : pre D T ⊆ pre D T' := fun t ht => by
  simp only [pre, Finset.mem_filter, Finset.mem_univ, true_and] at ht ⊢
  omega

/-- **The chain's bound** on `E[1/(|π_i U_T|^α |π_j U_T|^β)]` over two free rows. -/
noncomputable def Bc (q : ℝ) : ℕ → ℕ → ℝ
  | 0, p => (1 / 2) ^ p
  | t + 1, p => Bc q t (2 * p) + cB q p * Bc q t (p - 1)

theorem two_le_csz_base (c : Fin D → ι → F) (i : ι) : 2 ≤ csz (projS {i} (spanX c ∅)) := by
  have h1 : (1 : ι → F) ∈ spanX c ∅ := AddSubgroup.subset_closure ⟨fun _ => true, by simp⟩
  have hz : zS {i} (1 : ι → F) ∈ projS {i} (spanX c ∅) := ⟨1, h1, rfl⟩
  have hne : (0 : ι → F) ≠ zS {i} 1 := fun h => by
    have := congrFun h i
    simp [zS] at this
  have hsub : ({0, zS {i} 1} : Finset (ι → F)) ⊆ univ.filter (· ∈ projS {i} (spanX c ∅)) := by
    intro x hx
    simp only [Finset.mem_insert, Finset.mem_singleton] at hx
    rcases hx with rfl | rfl
    · simp [(projS {i} (spanX c ∅)).zero_mem]
    · simp [hz]
  rw [← Finset.card_pair hne]
  exact Finset.card_le_card hsub

theorem csz_spanX_empty_le [CharP F 2] (c : Fin D → ι → F) : csz (spanX c ∅) ≤ 2 := by
  have h2 : ∀ x : ι → F, x + x = 0 := fun x => funext fun i => CharTwo.add_self_eq_zero (x i)
  have hsub : univ.filter (· ∈ spanX c ∅) ⊆ ({0, 1} : Finset (ι → F)) := by
    intro x hx
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hx
    simp only [spanX, Finset.prod_empty, Set.range_const] at hx
    obtain ⟨n, rfl⟩ := AddSubgroup.mem_closure_singleton.1 hx
    obtain ⟨k, rfl | rfl⟩ := Int.even_or_odd' n
    · rw [mul_zsmul, two_zsmul, h2]
      simp
    · rw [add_zsmul, mul_zsmul, two_zsmul, h2, one_zsmul, zero_add]
      simp
  exact (Finset.card_le_card hsub).trans Finset.card_le_two

/-- **The spans' sizes**: `|U_T| ≤ 2^(2^T)`, each coordinate at most squaring. -/
theorem csz_spanX_pre_le [CharP F 2] (c : Fin D → ι → F) :
    ∀ T, T ≤ D → csz (spanX c (pre D T)) ≤ 2 ^ 2 ^ T
  | 0, _ => by
    rw [pre_zero]
    simpa using csz_spanX_empty_le c
  | T + 1, h => by
    have hT : T < D := by omega
    rw [pre_succ hT, spanX_insert _ (not_mem_pre hT)]
    have h1 := csz_sup_mul_Kc (spanX c (pre D T)) (c ⟨T, hT⟩)
    have h2 := one_le_Kc (spanX c (pre D T)) (c ⟨T, hT⟩)
    have ih := csz_spanX_pre_le c T (by omega)
    calc _ ≤ csz (spanX c (pre D T) ⊔ (spanX c (pre D T)).map (AddMonoidHom.mulLeft (c ⟨T, hT⟩))) *
          Kc (spanX c (pre D T)) (c ⟨T, hT⟩) := Nat.le_mul_of_pos_right _ h2
      _ = csz (spanX c (pre D T)) ^ 2 := h1
      _ ≤ (2 ^ 2 ^ T) ^ 2 := Nat.pow_le_pow_left ih 2
      _ = 2 ^ 2 ^ (T + 1) := by rw [← pow_mul, pow_succ]

/-- **The joint chain over two free rows**: `Σ_r 1/(|π_i U_T|^α |π_j U_T|^β) ≤ |R|·Bc(T, α)·Bc(T, β)`, rows `i`
and `j` drawing independently at every coordinate (`sum_mul_rows`) and each step bounded by `sum_row_le`. -/
theorem chain [CharP F 2] (fix : ι → Fin D → Option F) {i j : ι} (hij : i ≠ j)
    (hfi : ∀ t, fix i t = none) (hfj : ∀ t, fix j t = none) :
    ∀ T, T ≤ D → ∀ α β : ℕ,
      ∑ r, 1 / ((csz (projS {i} (spanX (col fix r) (pre D T))) : ℝ) ^ α *
        (csz (projS {j} (spanX (col fix r) (pre D T))) : ℝ) ^ β) ≤
      Fintype.card (ι → Fin D → F) * (Bc (Fintype.card F) T α * Bc (Fintype.card F) T β)
  | 0, _, α, β => by
    rw [pre_zero]
    have hpt : ∀ r : ι → Fin D → F, 1 / ((csz (projS {i} (spanX (col fix r) ∅)) : ℝ) ^ α *
        (csz (projS {j} (spanX (col fix r) ∅)) : ℝ) ^ β) ≤
          Bc (Fintype.card F) 0 α * Bc (Fintype.card F) 0 β := by
      intro r
      have hi : (2 : ℝ) ≤ csz (projS {i} (spanX (col fix r) ∅)) := by exact_mod_cast two_le_csz_base _ i
      have hj : (2 : ℝ) ≤ csz (projS {j} (spanX (col fix r) ∅)) := by exact_mod_cast two_le_csz_base _ j
      simp only [Bc]
      rw [div_pow, div_pow, one_pow, one_pow, div_mul_div_comm, one_mul]
      gcongr
    calc _ ≤ ∑ _r : ι → Fin D → F, Bc (Fintype.card F) 0 α * Bc (Fintype.card F) 0 β :=
          Finset.sum_le_sum fun r _ => hpt r
      _ = _ := by rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
  | T + 1, hT, α, β => by
    have hTD : T < D := by omega
    have ih := chain fix hij hfi hfj T (by omega)
    set q : ℝ := (Fintype.card F : ℝ) with hq
    have hq0 : 0 < q := by rw [hq]; exact_mod_cast Fintype.card_pos
    have hA : (0 : ℝ) < Fintype.card (ι → F) := by exact_mod_cast Fintype.card_pos
    have hn : 2 ≤ Fintype.card ι := by
      have := Finset.card_le_univ ({i, j} : Finset ι)
      rwa [Finset.card_pair hij] at this
    have hrow : ∀ (U : AddSubgroup (ι → F)) (k : ι), (∀ t, fix k t = none) → ∀ a : ι → F,
        projS {k} (U ⊔ U.map (AddMonoidHom.mulLeft (colA fix ⟨T, hTD⟩ a))) =
          projS {k} U ⊔ (projS {k} U).map (AddMonoidHom.mulLeft (fun _ => a k)) := by
      intro U k hk a
      rw [projS_sup_mul]
      congr 1
      exact map_mul_congr (projS_le {k} U) fun k' hk' => by
        rw [Finset.mem_singleton] at hk'
        subst hk'
        simp [colA, hk]
    have hstep : ∀ U : AddSubgroup (ι → F), ∑ a : ι → F,
        1 / ((csz (projS {i} (U ⊔ U.map (AddMonoidHom.mulLeft (colA fix ⟨T, hTD⟩ a)))) : ℝ) ^ α *
          (csz (projS {j} (U ⊔ U.map (AddMonoidHom.mulLeft (colA fix ⟨T, hTD⟩ a)))) : ℝ) ^ β) ≤
        Fintype.card (ι → F) * (rowB q (csz (projS {i} U)) α * rowB q (csz (projS {j} U)) β) := by
      intro U
      set g : F → ℝ := fun y =>
        1 / (csz (projS {i} U ⊔ (projS {i} U).map (AddMonoidHom.mulLeft (fun _ => y))) : ℝ) ^ α
      set h : F → ℝ := fun y =>
        1 / (csz (projS {j} U ⊔ (projS {j} U).map (AddMonoidHom.mulLeft (fun _ => y))) : ℝ) ^ β
      have hg := sum_row_le (projS {i} U) α
      have hh := sum_row_le (projS {j} U) β
      have hg0 : 0 ≤ ∑ y, g y := Finset.sum_nonneg fun y _ => by positivity
      have hri := rowB_nonneg hq0.le (Nat.cast_nonneg (csz (projS {i} U))) α
      calc _ = ∑ a : ι → F, g (a i) * h (a j) := by
            refine Finset.sum_congr rfl fun a _ => ?_
            rw [hrow U i hfi a, hrow U j hfj a, one_div_mul_one_div]
        _ = q ^ (Fintype.card ι - 2) * ((∑ y, g y) * ∑ y, h y) := sum_mul_rows hij g h
        _ ≤ q ^ (Fintype.card ι - 2) *
              ((q * rowB q (csz (projS {i} U)) α) * (q * rowB q (csz (projS {j} U)) β)) := by
            gcongr
        _ = _ := by
            rw [Fintype.card_fun]
            push_cast
            have e : q ^ Fintype.card ι = q ^ (Fintype.card ι - 2) * q ^ 2 := by
              rw [← pow_add]
              congr 1
              omega
            rw [← hq, e]
            ring
    have hsum := sum_step fix (not_mem_pre hTD)
      (fun W => 1 / ((csz (projS {i} W) : ℝ) ^ α * (csz (projS {j} W) : ℝ) ^ β))
    rw [pre_succ hTD]
    set ui : (ι → Fin D → F) → ℝ := fun r => (csz (projS {i} (spanX (col fix r) (pre D T))) : ℝ) with hui
    set uj : (ι → Fin D → F) → ℝ := fun r => (csz (projS {j} (spanX (col fix r) (pre D T))) : ℝ) with huj
    have hle : ∑ r, 1 / ((csz (projS {i} (spanX (col fix r) (insert ⟨T, hTD⟩ (pre D T)))) : ℝ) ^ α *
        (csz (projS {j} (spanX (col fix r) (insert ⟨T, hTD⟩ (pre D T)))) : ℝ) ^ β) ≤
        ∑ r, rowB q (ui r) α * rowB q (uj r) β := by
      refine le_of_mul_le_mul_left ?_ hA
      rw [hsum, Finset.mul_sum]
      exact Finset.sum_le_sum fun r _ => hstep _
    refine hle.trans ?_
    have hexp : ∀ r, rowB q (ui r) α * rowB q (uj r) β =
        1 / (ui r ^ (2 * α) * uj r ^ (2 * β)) + cB q β * (1 / (ui r ^ (2 * α) * uj r ^ (β - 1))) +
          cB q α * (1 / (ui r ^ (α - 1) * uj r ^ (2 * β))) +
            cB q α * cB q β * (1 / (ui r ^ (α - 1) * uj r ^ (β - 1))) := by
      intro r
      simp only [rowB]
      ring
    have hcα := cB_nonneg hq0.le α
    have hcβ := cB_nonneg hq0.le β
    calc ∑ r, rowB q (ui r) α * rowB q (uj r) β
        = ∑ r, 1 / (ui r ^ (2 * α) * uj r ^ (2 * β)) + cB q β * ∑ r, 1 / (ui r ^ (2 * α) * uj r ^ (β - 1)) +
            cB q α * ∑ r, 1 / (ui r ^ (α - 1) * uj r ^ (2 * β)) +
              cB q α * cB q β * ∑ r, 1 / (ui r ^ (α - 1) * uj r ^ (β - 1)) := by
          simp only [hexp, Finset.sum_add_distrib, Finset.mul_sum]
      _ ≤ Fintype.card (ι → Fin D → F) * (Bc q T (2 * α) * Bc q T (2 * β)) +
            cB q β * (Fintype.card (ι → Fin D → F) * (Bc q T (2 * α) * Bc q T (β - 1))) +
            cB q α * (Fintype.card (ι → Fin D → F) * (Bc q T (α - 1) * Bc q T (2 * β))) +
            cB q α * cB q β * (Fintype.card (ι → Fin D → F) * (Bc q T (α - 1) * Bc q T (β - 1))) :=
          add_le_add (add_le_add (add_le_add (ih _ _) (mul_le_mul_of_nonneg_left (ih _ _) hcβ))
            (mul_le_mul_of_nonneg_left (ih _ _) hcα)) (mul_le_mul_of_nonneg_left (ih _ _) (mul_nonneg hcα hcβ))
      _ = _ := by
          simp only [Bc]
          ring

end Chain

end FlockComplete.MaskRank
