module

public import Proofs.Flock.Soundness.Audit.Closure
import Mathlib.Analysis.MeanInequalities
import Proofs.Flock.Soundness.Game

/-! Windows of calls, modelled with each call sampled on its own coins under a distinct context and the window's audit as one `Analysis`, which the statements neither state nor check. -/

@[expose] public section

namespace FlockSoundness.Audit

open Game Finset
open scoped ENNReal NNReal

namespace SamplingDesign

variable {n m C : ℕ}

def Covers (σ : Fin n → Fin m) (k w : Fin m → ℕ) (K : ℕ) : Prop :=
  ∀ s, K * w s * (stratum σ s).card ≤ k s * totalWork σ w ∨ k s = (stratum σ s).card

theorem stratified_escape_le_of_covers (σ : Fin n → Fin m) (k : Fin m → ℕ) (hk : ∀ s, k s ≤ (stratum σ s).card)
    (w : Fin m → ℕ) (K : ℕ) (hW : 0 < totalWork σ w) (hc : Covers σ k w K) (B : Finset (Fin n)) :
    (stratified σ k hk).missProb B ≤
      (((totalWork σ w - workOf σ w B : ℕ) : ℝ≥0∞) / (totalWork σ w : ℝ≥0∞)) ^ K := by
  classical
  set W := totalWork σ w with hWdef
  set N : Fin m → ℕ := fun s => (stratum σ s).card with hNdef
  set M : Fin m → ℕ := fun s => (B.filter (σ · = s)).card with hMdef
  have hMN : ∀ s, M s ≤ N s := fun s => card_le_card (filter_subset_filter _ (subset_univ B))
  have hkN : ∀ s, k s ≤ N s := hk
  have hW0 : (W : ℝ≥0) ≠ 0 := by exact_mod_cast hW.ne'
  set x : Fin m → ℝ≥0 := fun s => ((N s - M s : ℕ) : ℝ≥0) / (N s : ℝ≥0) with hxdef
  set a : Fin m → ℝ≥0 := fun s => ((w s * N s : ℕ) : ℝ≥0) / (W : ℝ≥0) with hadef
  have hsumN : ∑ s, w s * N s = W := (totalWork_eq σ w).symm
  have ha1 : ∑ s, a s = 1 := by
    simp only [hadef]
    rw [← Finset.sum_div, ← Nat.cast_sum, hsumN, div_self hW0]
  have hterm : ∀ s, a s * x s = ((w s * (N s - M s) : ℕ) : ℝ≥0) / (W : ℝ≥0) := by
    intro s
    simp only [hadef, hxdef]
    rcases Nat.eq_zero_or_pos (N s) with h0 | hpos
    · have hM0 : M s = 0 := by have := hMN s; omega
      simp [h0, hM0]
    · have hN0 : (N s : ℝ≥0) ≠ 0 := by exact_mod_cast hpos.ne'
      rw [div_mul_div_comm, ← Nat.cast_mul, show w s * N s * (N s - M s) = w s * (N s - M s) * N s by ring,
        Nat.cast_mul, mul_div_mul_right _ _ hN0]
  have hax : ∑ s, a s * x s = ((W - workOf σ w B : ℕ) : ℝ≥0) / (W : ℝ≥0) := by
    rw [Finset.sum_congr rfl fun s _ => hterm s, ← Finset.sum_div, ← Nat.cast_sum]
    congr 2
    have hB : workOf σ w B = ∑ s, w s * M s := workOf_eq σ w B
    have hsplit : ∑ s, w s * (N s - M s) + ∑ s, w s * M s = ∑ s, w s * N s := by
      rw [← Finset.sum_add_distrib]
      exact Finset.sum_congr rfl fun s _ => by rw [← Nat.mul_add, Nat.sub_add_cancel (hMN s)]
    omega
  have hfac : ∀ s, ((N s - M s).choose (k s) : ℝ≥0) / ((N s).choose (k s) : ℝ≥0) ≤ x s ^ ((a s : ℝ) * K) := by
    intro s
    rcases Nat.eq_zero_or_pos (N s) with h0 | hpos
    · have hk0 : k s = 0 := by have := hkN s; omega
      have hM0 : M s = 0 := by have := hMN s; omega
      simp [hadef, h0, hk0, hM0]
    have hC0 : ((N s).choose (k s) : ℝ≥0) ≠ 0 := by exact_mod_cast (Nat.choose_pos (hkN s)).ne'
    rcases Nat.eq_zero_or_pos (M s) with hM0 | hMpos
    · have hx1 : x s = 1 := by
        simp only [hxdef, hM0, Nat.sub_zero]
        exact div_self (by exact_mod_cast hpos.ne')
      rw [hx1, NNReal.one_rpow, hM0, Nat.sub_zero, div_self hC0]
    rcases Nat.eq_zero_or_pos (k s) with hk0 | hk1
    · have hKw : K * w s * N s = 0 := by
        have h' : K * w s * N s ≤ k s * W ∨ k s = N s := hc s
        rcases h' with h | h
        · rw [hk0, zero_mul] at h; omega
        · omega
      have haK : (a s : ℝ) * K = 0 := by
        rcases Nat.mul_eq_zero.1 hKw with h | h
        · rcases Nat.mul_eq_zero.1 h with h | h
          · simp [h]
          · simp [hadef, h]
        · simp [hadef, h]
      rw [haK, NNReal.rpow_zero, hk0, Nat.choose_zero_right, Nat.choose_zero_right, Nat.cast_one, div_one]
    have hMNs := hMN s
    by_cases hzero : k s = N s ∨ M s = N s
    · have : (N s - M s).choose (k s) = 0 := Nat.choose_eq_zero_of_lt (by omega)
      rw [this, Nat.cast_zero, zero_div]
      exact zero_le
    push Not at hzero
    have hge : K * w s * N s ≤ k s * W := (hc s).resolve_right hzero.1
    have hx0 : 0 < x s := div_pos (by exact_mod_cast (show 0 < N s - M s by omega)) (by exact_mod_cast hpos)
    have hx1 : x s ≤ 1 := div_le_one_of_le₀ (by exact_mod_cast Nat.sub_le _ _) zero_le
    have hexp : (a s : ℝ) * K ≤ (k s : ℝ) := by
      simp only [hadef, NNReal.coe_div, NNReal.coe_natCast]
      rw [div_mul_eq_mul_div, div_le_iff₀ (by exact_mod_cast hW)]
      exact_mod_cast (show w s * N s * K ≤ k s * W from
        calc w s * N s * K = K * w s * N s := by ring
          _ ≤ k s * W := hge)
    calc ((N s - M s).choose (k s) : ℝ≥0) / ((N s).choose (k s) : ℝ≥0) ≤ x s ^ (k s) :=
          choose_div_le_pow (hkN s) hpos
      _ = x s ^ ((k s : ℕ) : ℝ) := (NNReal.rpow_natCast _ _).symm
      _ ≤ x s ^ ((a s : ℝ) * K) := NNReal.rpow_le_rpow_of_exponent_ge hx0 hx1 hexp
  have hprod : ∏ s, ((N s - M s).choose (k s) : ℝ≥0) / ((N s).choose (k s) : ℝ≥0) ≤
      (((W - workOf σ w B : ℕ) : ℝ≥0) / (W : ℝ≥0)) ^ K := by
    calc _ ≤ ∏ s, x s ^ ((a s : ℝ) * K) := Finset.prod_le_prod fun s _ => hfac s
      _ = (∏ s, x s ^ (a s : ℝ)) ^ K := by
          rw [← Finset.prod_pow]
          exact Finset.prod_congr rfl fun s _ => by rw [NNReal.rpow_mul, NNReal.rpow_natCast]
      _ ≤ (∑ s, a s * x s) ^ K :=
          pow_le_pow_left₀ zero_le (NNReal.geom_mean_le_arith_mean_weighted univ a x ha1) K
      _ = _ := by rw [hax]
  have hC : ∀ s, ((N s).choose (k s) : ℝ≥0) ≠ 0 := fun s => by exact_mod_cast (Nat.choose_pos (hkN s)).ne'
  rw [stratified_escape]
  calc ∏ s, (((stratum σ s).card - (B.filter (σ · = s)).card).choose (k s) : ℝ≥0∞) /
        ((stratum σ s).card.choose (k s) : ℝ≥0∞)
      = ((∏ s, ((N s - M s).choose (k s) : ℝ≥0) / ((N s).choose (k s) : ℝ≥0) : ℝ≥0) : ℝ≥0∞) := by
        rw [ENNReal.ofNNReal_finsetProd]
        refine Finset.prod_congr rfl fun s _ => ?_
        rw [ENNReal.coe_div (hC s)]
        simp [hNdef, hMdef]
    _ ≤ (((((W - workOf σ w B : ℕ) : ℝ≥0) / (W : ℝ≥0)) ^ K : ℝ≥0) : ℝ≥0∞) := ENNReal.coe_le_coe.2 hprod
    _ = _ := by rw [ENNReal.coe_pow, ENNReal.coe_div hW0, ENNReal.coe_natCast, ENNReal.coe_natCast]

theorem covers_work (σ : Fin n → Fin m) (w f : Fin m → ℕ) (K : ℕ) (hW : 0 < totalWork σ w) :
    Covers σ (workK σ w f K) w K := fun s =>
  (workK_ge s hW).symm.imp (fun h => h.trans_eq (mul_comm _ _)) id

/-- Call `c`'s work `W_c`, its strata's. -/
def callWork (σ : Fin n → Fin m) (call : Fin m → Fin C) (w : Fin m → ℕ) (c : Fin C) : ℕ :=
  ∑ s ∈ univ.filter (call · = c), w s * (stratum σ s).card

/-- Call `c`'s units `N_c`. -/
def callUnits (σ : Fin n → Fin m) (call : Fin m → Fin C) (c : Fin C) : ℕ :=
  ∑ s ∈ univ.filter (call · = c), (stratum σ s).card

/-- Call `c`'s budget, its share of the window's `K`: `K_c = min(N_c, max(1, ⌈K · W_c / W⌉))`. -/
def callK (σ : Fin n → Fin m) (call : Fin m → Fin C) (w : Fin m → ℕ) (K : ℕ) (c : Fin C) : ℕ :=
  min (callUnits σ call c) (max 1 ((K * callWork σ call w c + totalWork σ w - 1) / totalWork σ w))

/-- The window's sizes: each call's work sampling design at its own budget, `k_s = min(n_s, max(f_s, ⌈K_c · w_s · n_s / W_c⌉))`. -/
def executionPhaseK (σ : Fin n → Fin m) (call : Fin m → Fin C) (w f : Fin m → ℕ) (K : ℕ) (s : Fin m) : ℕ :=
  workRule (callK σ call w K (call s)) (w s) (stratum σ s).card (callWork σ call w (call s)) (f s)

theorem executionPhaseK_le (σ : Fin n → Fin m) (call : Fin m → Fin C) (w f : Fin m → ℕ) (K : ℕ) (s : Fin m) :
    executionPhaseK σ call w f K s ≤ (stratum σ s).card := min_le_left _ _

theorem card_le_callUnits (σ : Fin n → Fin m) (call : Fin m → Fin C) (s : Fin m) :
    (stratum σ s).card ≤ callUnits σ call (call s) :=
  Finset.single_le_sum (f := fun s => (stratum σ s).card) (fun _ _ => Nat.zero_le _)
    (mem_filter.2 ⟨mem_univ s, rfl⟩)

/-- Needs each call's work in one stratum (`hone`): when a call's share is capped at its units, a second work
stratum in the call can sample well below its share, and the window bound fails. -/
theorem covers_executionPhase (σ : Fin n → Fin m) (call : Fin m → Fin C) (w f : Fin m → ℕ) (K : ℕ)
    (hW : 0 < totalWork σ w) (hone : ∀ s, 0 < w s → callWork σ call w (call s) = w s * (stratum σ s).card) :
    Covers σ (executionPhaseK σ call w f K) w K := by
  intro s
  rcases Nat.eq_zero_or_pos (w s) with hw | hw
  · exact Or.inl (by simp [hw])
  rcases Nat.eq_zero_or_pos (stratum σ s).card with hn | hn
  · exact Or.inr (by have := executionPhaseK_le σ call w f K s; omega)
  set Kc := callK σ call w K (call s) with hKc
  have hWc := hone s hw
  have hpos : 0 < w s * (stratum σ s).card := Nat.mul_pos hw hn
  have hceil : (Kc * w s * (stratum σ s).card + callWork σ call w (call s) - 1) / callWork σ call w (call s) = Kc := by
    rw [hWc, show Kc * w s * (stratum σ s).card + w s * (stratum σ s).card - 1 =
        (w s * (stratum σ s).card - 1) + Kc * (w s * (stratum σ s).card) by rw [mul_assoc]; omega,
      Nat.add_mul_div_right _ _ hpos, Nat.div_eq_of_lt (by omega), Nat.zero_add]
  have hk : executionPhaseK σ call w f K s = min (stratum σ s).card (max (f s) Kc) := by
    unfold executionPhaseK workRule
    rw [hceil]
  rcases le_or_gt (stratum σ s).card (max (f s) Kc) with h | h
  · exact Or.inr (by rw [hk, min_eq_left h])
  refine Or.inl ?_
  rw [hk, min_eq_right h.le]
  have hlt : Kc < callUnits σ call (call s) :=
    lt_of_lt_of_le (lt_of_le_of_lt (le_max_right _ _) h) (card_le_callUnits σ call s)
  have hKc' : Kc = max 1 ((K * callWork σ call w (call s) + totalWork σ w - 1) / totalWork σ w) := by
    have e : Kc = min (callUnits σ call (call s))
        (max 1 ((K * callWork σ call w (call s) + totalWork σ w - 1) / totalWork σ w)) := by rw [hKc, callK]
    rcases min_choice (callUnits σ call (call s))
        (max 1 ((K * callWork σ call w (call s) + totalWork σ w - 1) / totalWork σ w)) with h' | h'
    · rw [e, h'] at hlt; exact absurd hlt (lt_irrefl _)
    · rw [e, h']
  calc K * w s * (stratum σ s).card = K * callWork σ call w (call s) := by rw [hWc, mul_assoc]
    _ ≤ totalWork σ w * ((K * callWork σ call w (call s) + totalWork σ w - 1) / totalWork σ w) :=
        le_mul_ceil_div _ _ hW
    _ ≤ totalWork σ w * Kc := Nat.mul_le_mul_left _ (by rw [hKc']; exact le_max_right _ _)
    _ ≤ totalWork σ w * max (f s) Kc := Nat.mul_le_mul_left _ (le_max_right _ _)
    _ = max (f s) Kc * totalWork σ w := mul_comm _ _

theorem covers_of_floor (σ : Fin n → Fin m) (k f v : Fin m → ℕ) (Ky : ℕ) (hk : ∀ s, k s ≤ (stratum σ s).card)
    (hkf : ∀ s, min (stratum σ s).card (f s) ≤ k s)
    (hfy : ∀ s, Ky * v s * (stratum σ s).card ≤ f s * totalWork σ v) :
    Covers σ k v Ky := by
  intro s
  rcases le_total (stratum σ s).card (f s) with h | h
  · have := hkf s
    rw [min_eq_left h] at this
    exact Or.inr (le_antisymm (hk s) this)
  · have hfk := hkf s
    rw [min_eq_right h] at hfk
    exact Or.inl ((hfy s).trans (Nat.mul_le_mul_right _ hfk))

theorem closure_escape_le_executionPhase {L : SamplingDesign n} (σ : Fin n → Fin m) (k : Fin m → ℕ) (hk : ∀ s, k s ≤ (stratum σ s).card)
    (hL : ∀ B, L.missProb B ≤ (stratified σ k hk).missProb B) (cl : Fin n → Finset (Fin n)) {w v : Fin m → ℕ} {K Ky : ℕ}
    (hW : 0 < totalWork σ w) (hc : Covers σ k w K) (hN : 0 < totalWork σ v) (hcy : Covers σ k v Ky) {T Ty : ℕ}
    {B : Finset (Fin n)} (hB : T ≤ unsoundWork σ w cl B ∨ Ty ≤ unsoundWork σ v cl B) :
    (L.closure cl).missProb B ≤
      max ((((totalWork σ w - T : ℕ) : ℝ≥0∞) / (totalWork σ w : ℝ≥0∞)) ^ K)
        ((((totalWork σ v - Ty : ℕ) : ℝ≥0∞) / (totalWork σ v : ℝ≥0∞)) ^ Ky) := by
  rw [closure_escape]
  refine (hL _).trans ?_
  rcases hB with hB | hB
  · refine le_max_of_le_left ((stratified_escape_le_of_covers σ k hk w K hW hc _).trans ?_)
    gcongr
  · refine le_max_of_le_right ((stratified_escape_le_of_covers σ k hk v Ky hN hcy _).trans ?_)
    gcongr

theorem closure_escape_le_executionPhase_slack {L : SamplingDesign n} (σ : Fin n → Fin m) (k : Fin m → ℕ)
    (hk : ∀ s, k s ≤ (stratum σ s).card) {η : ℝ≥0∞} (hL : ∀ B, L.missProb B ≤ (stratified σ k hk).missProb B + η)
    (cl : Fin n → Finset (Fin n)) {w v : Fin m → ℕ} {K Ky : ℕ} (hW : 0 < totalWork σ w) (hc : Covers σ k w K)
    (hN : 0 < totalWork σ v) (hcy : Covers σ k v Ky) {T Ty : ℕ} {B : Finset (Fin n)}
    (hB : T ≤ unsoundWork σ w cl B ∨ Ty ≤ unsoundWork σ v cl B) :
    (L.closure cl).missProb B ≤
      max ((((totalWork σ w - T : ℕ) : ℝ≥0∞) / (totalWork σ w : ℝ≥0∞)) ^ K)
        ((((totalWork σ v - Ty : ℕ) : ℝ≥0∞) / (totalWork σ v : ℝ≥0∞)) ^ Ky) + η := by
  have h := closure_escape_le_executionPhase σ k hk (fun _ => le_rfl) cl hW hc hN hcy hB
  rw [closure_escape] at h ⊢
  exact (hL _).trans (add_le_add h le_rfl)

theorem pow_record_le {W U : ℕ} (hW : 0 < W) (hU : W ≤ 1000 * U) :
    (((W - U : ℕ) : ℝ≥0∞) / (W : ℝ≥0∞)) ^ 27713 ≤ ((2 : ℝ≥0∞) ^ 40)⁻¹ := by
  refine le_trans (pow_le_pow_left₀ zero_le ?_ 27713) record_sizing
  have hWt : (W : ℝ≥0∞) ≠ ⊤ := ENNReal.natCast_ne_top _
  have hW0 : (W : ℝ≥0∞) ≠ 0 := by exact_mod_cast hW.ne'
  rw [ENNReal.div_le_iff hW0 hWt, ← ENNReal.mul_div_right_comm, ENNReal.le_div_iff_mul_le (Or.inl (by norm_num))
    (Or.inl (by norm_num))]
  exact_mod_cast (show (W - U) * 1000 ≤ 999 * W by omega)

end SamplingDesign

section Audits

variable {n m : ℕ} {σ : Fin n → Fin m} {k : Fin m → ℕ} {hk : ∀ s, k s ≤ (SamplingDesign.stratum σ s).card}
  {cl : Fin n → Finset (Fin n)} {Reg : Type} {session : Finset (Fin n) → Reg → Game Bool}

theorem audit_executionPhase (A : Analysis ((SamplingDesign.stratified σ k hk).closure cl) Reg session) {εks εbind : ℝ≥0∞}
    (hks : A.KnowledgeSound εks) (hlink : A.BindSound εbind) {w v : Fin m → ℕ} {K Ky : ℕ}
    (hW : 0 < SamplingDesign.totalWork σ w) (hc : SamplingDesign.Covers σ k w K) (hN : 0 < SamplingDesign.totalWork σ v) (hcy : SamplingDesign.Covers σ k v Ky)
    (T Ty : ℕ) (σ' : Strategy (audit ((SamplingDesign.stratified σ k hk).closure cl) Reg session)) :
    prob (fun o => o.1 = true ∧ (T ≤ SamplingDesign.unsoundWork σ w cl (A.incorrect (A.committedOf σ')) ∨
        Ty ≤ SamplingDesign.unsoundWork σ v cl (A.incorrect (A.committedOf σ'))))
        (audit ((SamplingDesign.stratified σ k hk).closure cl) Reg session) σ' ≤
      max ((((SamplingDesign.totalWork σ w - T : ℕ) : ℝ≥0∞) / (SamplingDesign.totalWork σ w : ℝ≥0∞)) ^ K)
          ((((SamplingDesign.totalWork σ v - Ty : ℕ) : ℝ≥0∞) / (SamplingDesign.totalWork σ v : ℝ≥0∞)) ^ Ky) + εks + εbind := by
  refine (audit_profile A hks hlink (fun B => T ≤ SamplingDesign.unsoundWork σ w cl B ∨ Ty ≤ SamplingDesign.unsoundWork σ v cl B) σ').trans ?_
  gcongr
  exact iSup₂_le fun B hB => SamplingDesign.closure_escape_le_executionPhase σ k hk (fun _ => le_rfl) cl hW hc hN hcy hB

theorem audit_executionPhase_of_le {L : SamplingDesign n} (hL : ∀ B, L.missProb B ≤ (SamplingDesign.stratified σ k hk).missProb B)
    (A : Analysis (L.closure cl) Reg session) {εks εbind : ℝ≥0∞} (hks : A.KnowledgeSound εks)
    (hlink : A.BindSound εbind) {w v : Fin m → ℕ} {K Ky : ℕ} (hW : 0 < SamplingDesign.totalWork σ w) (hc : SamplingDesign.Covers σ k w K)
    (hN : 0 < SamplingDesign.totalWork σ v) (hcy : SamplingDesign.Covers σ k v Ky) (T Ty : ℕ)
    (σ' : Strategy (audit (L.closure cl) Reg session)) :
    prob (fun o => o.1 = true ∧ (T ≤ SamplingDesign.unsoundWork σ w cl (A.incorrect (A.committedOf σ')) ∨
        Ty ≤ SamplingDesign.unsoundWork σ v cl (A.incorrect (A.committedOf σ'))))
        (audit (L.closure cl) Reg session) σ' ≤
      max ((((SamplingDesign.totalWork σ w - T : ℕ) : ℝ≥0∞) / (SamplingDesign.totalWork σ w : ℝ≥0∞)) ^ K)
          ((((SamplingDesign.totalWork σ v - Ty : ℕ) : ℝ≥0∞) / (SamplingDesign.totalWork σ v : ℝ≥0∞)) ^ Ky) + εks + εbind := by
  refine (audit_profile A hks hlink (fun B => T ≤ SamplingDesign.unsoundWork σ w cl B ∨ Ty ≤ SamplingDesign.unsoundWork σ v cl B) σ').trans ?_
  gcongr
  exact iSup₂_le fun B hB => SamplingDesign.closure_escape_le_executionPhase σ k hk hL cl hW hc hN hcy hB

theorem extraction_audit_executionPhase (A : ExtractionAnalysis ((SamplingDesign.stratified σ k hk).closure cl) Reg session)
    {εks εbind : (R : Reg) → Cont ((SamplingDesign.stratified σ k hk).closure cl) Reg session R → ℝ≥0∞}
    (hks : A.KnowledgeSound εks) (hlink : A.BindSound εbind) {w v : Fin m → ℕ} {K Ky : ℕ}
    (hW : 0 < SamplingDesign.totalWork σ w) (hc : SamplingDesign.Covers σ k w K) (hN : 0 < SamplingDesign.totalWork σ v) (hcy : SamplingDesign.Covers σ k v Ky)
    (T Ty : ℕ) (σ' : Strategy (audit ((SamplingDesign.stratified σ k hk).closure cl) Reg session)) :
    prob (fun o => o.1 = true ∧ (T ≤ SamplingDesign.unsoundWork σ w cl (A.incorrect (A.committedOf σ')) ∨
        Ty ≤ SamplingDesign.unsoundWork σ v cl (A.incorrect (A.committedOf σ'))))
        (audit ((SamplingDesign.stratified σ k hk).closure cl) Reg session) σ' ≤
      max ((((SamplingDesign.totalWork σ w - T : ℕ) : ℝ≥0∞) / (SamplingDesign.totalWork σ w : ℝ≥0∞)) ^ K)
          ((((SamplingDesign.totalWork σ v - Ty : ℕ) : ℝ≥0∞) / (SamplingDesign.totalWork σ v : ℝ≥0∞)) ^ Ky) +
        εks (reg σ') (cont σ') + εbind (reg σ') (cont σ') := by
  refine (extraction_audit_le A hks hlink (fun X _ => T ≤ SamplingDesign.unsoundWork σ w cl (A.incorrect X) ∨
    Ty ≤ SamplingDesign.unsoundWork σ v cl (A.incorrect X)) σ').trans ?_
  gcongr
  by_cases hB : T ≤ SamplingDesign.unsoundWork σ w cl (A.incorrect (A.committedOf σ')) ∨
      Ty ≤ SamplingDesign.unsoundWork σ v cl (A.incorrect (A.committedOf σ'))
  · exact (prCoin_mono fun ω h => h.2).trans (SamplingDesign.closure_escape_le_executionPhase σ k hk (fun _ => le_rfl) cl hW hc hN hcy hB)
  · exact (prCoin_mono (q := fun _ => False) fun ω h => hB h.1).trans (prCoin_false.le.trans zero_le)

theorem audit_executionPhase_of_record (A : Analysis ((SamplingDesign.stratified σ k hk).closure cl) Reg session) {εks εbind : ℝ≥0∞}
    (hks : A.KnowledgeSound εks) (hlink : A.BindSound εbind) {w v : Fin m → ℕ}
    (hW : 0 < SamplingDesign.totalWork σ w) (hc : SamplingDesign.Covers σ k w 27713) (hN : 0 < SamplingDesign.totalWork σ v)
    (hcy : SamplingDesign.Covers σ k v 27713) (σ' : Strategy (audit ((SamplingDesign.stratified σ k hk).closure cl) Reg session)) :
    prob (fun o => o.1 = true ∧ (SamplingDesign.totalWork σ w ≤ 1000 * SamplingDesign.unsoundWork σ w cl (A.incorrect (A.committedOf σ')) ∨
        SamplingDesign.totalWork σ v ≤ 1000 * SamplingDesign.unsoundWork σ v cl (A.incorrect (A.committedOf σ'))))
        (audit ((SamplingDesign.stratified σ k hk).closure cl) Reg session) σ' ≤ ((2 : ℝ≥0∞) ^ 40)⁻¹ + εks + εbind := by
  refine (audit_profile A hks hlink (fun B => SamplingDesign.totalWork σ w ≤ 1000 * SamplingDesign.unsoundWork σ w cl B ∨
    SamplingDesign.totalWork σ v ≤ 1000 * SamplingDesign.unsoundWork σ v cl B) σ').trans ?_
  gcongr
  refine iSup₂_le fun B hB => ?_
  rw [SamplingDesign.closure_escape]
  rcases hB with hB | hB
  · exact (SamplingDesign.stratified_escape_le_of_covers σ k hk w 27713 hW hc _).trans (SamplingDesign.pow_record_le hW hB)
  · exact (SamplingDesign.stratified_escape_le_of_covers σ k hk v 27713 hN hcy _).trans (SamplingDesign.pow_record_le hN hB)

/-- **The per-call split with y's floors at the sizing of record**: every call sampled by
its own work sampling design at its share of `K = 27,713`, each stratum's floor covering `v` at `K_y = 27,713`, with closures. -/
theorem audit_executionPhase_split_of_record {C : ℕ} {call : Fin m → Fin C} {w v f : Fin m → ℕ}
    (A : Analysis ((SamplingDesign.stratified σ (SamplingDesign.executionPhaseK σ call w f 27713) (SamplingDesign.executionPhaseK_le σ call w f 27713)).closure cl)
      Reg session) {εks εbind : ℝ≥0∞} (hks : A.KnowledgeSound εks) (hlink : A.BindSound εbind)
    (hW : 0 < SamplingDesign.totalWork σ w)
    (hone : ∀ s, 0 < w s → SamplingDesign.callWork σ call w (call s) = w s * (SamplingDesign.stratum σ s).card)
    (hN : 0 < SamplingDesign.totalWork σ v) (hfy : ∀ s, 27713 * v s * (SamplingDesign.stratum σ s).card ≤ f s * SamplingDesign.totalWork σ v)
    (σ' : Strategy (audit ((SamplingDesign.stratified σ (SamplingDesign.executionPhaseK σ call w f 27713)
      (SamplingDesign.executionPhaseK_le σ call w f 27713)).closure cl) Reg session)) :
    prob (fun o => o.1 = true ∧ (SamplingDesign.totalWork σ w ≤ 1000 * SamplingDesign.unsoundWork σ w cl (A.incorrect (A.committedOf σ')) ∨
        SamplingDesign.totalWork σ v ≤ 1000 * SamplingDesign.unsoundWork σ v cl (A.incorrect (A.committedOf σ'))))
        (audit ((SamplingDesign.stratified σ (SamplingDesign.executionPhaseK σ call w f 27713)
          (SamplingDesign.executionPhaseK_le σ call w f 27713)).closure cl) Reg session) σ' ≤ ((2 : ℝ≥0∞) ^ 40)⁻¹ + εks + εbind :=
  audit_executionPhase_of_record A hks hlink hW (SamplingDesign.covers_executionPhase σ call w f 27713 hW hone) hN
    (SamplingDesign.covers_of_floor σ _ f v 27713 (SamplingDesign.executionPhaseK_le σ call w f 27713)
      (fun _ => min_le_min le_rfl (le_max_left _ _)) hfy) σ'

theorem audit_executionPhase_of_le_slack {L : SamplingDesign n} {η : ℝ≥0∞} (hL : ∀ B, L.missProb B ≤ (SamplingDesign.stratified σ k hk).missProb B + η)
    (A : Analysis (L.closure cl) Reg session) {εks εbind : ℝ≥0∞} (hks : A.KnowledgeSound εks)
    (hlink : A.BindSound εbind) {w v : Fin m → ℕ} {K Ky : ℕ} (hW : 0 < SamplingDesign.totalWork σ w) (hc : SamplingDesign.Covers σ k w K)
    (hN : 0 < SamplingDesign.totalWork σ v) (hcy : SamplingDesign.Covers σ k v Ky) (T Ty : ℕ)
    (σ' : Strategy (audit (L.closure cl) Reg session)) :
    prob (fun o => o.1 = true ∧ (T ≤ SamplingDesign.unsoundWork σ w cl (A.incorrect (A.committedOf σ')) ∨
        Ty ≤ SamplingDesign.unsoundWork σ v cl (A.incorrect (A.committedOf σ'))))
        (audit (L.closure cl) Reg session) σ' ≤
      max ((((SamplingDesign.totalWork σ w - T : ℕ) : ℝ≥0∞) / (SamplingDesign.totalWork σ w : ℝ≥0∞)) ^ K)
          ((((SamplingDesign.totalWork σ v - Ty : ℕ) : ℝ≥0∞) / (SamplingDesign.totalWork σ v : ℝ≥0∞)) ^ Ky) + η + εks + εbind := by
  refine (audit_profile A hks hlink (fun B => T ≤ SamplingDesign.unsoundWork σ w cl B ∨ Ty ≤ SamplingDesign.unsoundWork σ v cl B) σ').trans ?_
  gcongr
  exact iSup₂_le fun B hB => SamplingDesign.closure_escape_le_executionPhase_slack σ k hk hL cl hW hc hN hcy hB

example {L : SamplingDesign n} (hL : ∀ B, L.missProb B ≤ (SamplingDesign.stratified σ k hk).missProb B)
    (A : Analysis (L.closure cl) Reg session) {εks εbind : ℝ≥0∞} (hks : A.KnowledgeSound εks)
    (hlink : A.BindSound εbind) {w v : Fin m → ℕ} {K Ky : ℕ} (hW : 0 < SamplingDesign.totalWork σ w) (hc : SamplingDesign.Covers σ k w K)
    (hN : 0 < SamplingDesign.totalWork σ v) (hcy : SamplingDesign.Covers σ k v Ky) (T Ty : ℕ)
    (σ' : Strategy (audit (L.closure cl) Reg session)) :
    prob (fun o => o.1 = true ∧ (T ≤ SamplingDesign.unsoundWork σ w cl (A.incorrect (A.committedOf σ')) ∨
        Ty ≤ SamplingDesign.unsoundWork σ v cl (A.incorrect (A.committedOf σ'))))
        (audit (L.closure cl) Reg session) σ' ≤
      max ((((SamplingDesign.totalWork σ w - T : ℕ) : ℝ≥0∞) / (SamplingDesign.totalWork σ w : ℝ≥0∞)) ^ K)
          ((((SamplingDesign.totalWork σ v - Ty : ℕ) : ℝ≥0∞) / (SamplingDesign.totalWork σ v : ℝ≥0∞)) ^ Ky) + εks + εbind := by
  simpa only [add_zero] using
    audit_executionPhase_of_le_slack (η := 0) (fun B => by rw [add_zero]; exact hL B) A hks hlink hW hc hN hcy T Ty σ'

theorem extraction_audit_executionPhase_of_le_slack {L : SamplingDesign n} {η : ℝ≥0∞}
    (hL : ∀ B, L.missProb B ≤ (SamplingDesign.stratified σ k hk).missProb B + η) (A : ExtractionAnalysis (L.closure cl) Reg session)
    {εks εbind : (R : Reg) → Cont (L.closure cl) Reg session R → ℝ≥0∞}
    (hks : A.KnowledgeSound εks) (hlink : A.BindSound εbind) {w v : Fin m → ℕ} {K Ky : ℕ}
    (hW : 0 < SamplingDesign.totalWork σ w) (hc : SamplingDesign.Covers σ k w K) (hN : 0 < SamplingDesign.totalWork σ v) (hcy : SamplingDesign.Covers σ k v Ky)
    (T Ty : ℕ) (σ' : Strategy (audit (L.closure cl) Reg session)) :
    prob (fun o => o.1 = true ∧ (T ≤ SamplingDesign.unsoundWork σ w cl (A.incorrect (A.committedOf σ')) ∨
        Ty ≤ SamplingDesign.unsoundWork σ v cl (A.incorrect (A.committedOf σ'))))
        (audit (L.closure cl) Reg session) σ' ≤
      max ((((SamplingDesign.totalWork σ w - T : ℕ) : ℝ≥0∞) / (SamplingDesign.totalWork σ w : ℝ≥0∞)) ^ K)
          ((((SamplingDesign.totalWork σ v - Ty : ℕ) : ℝ≥0∞) / (SamplingDesign.totalWork σ v : ℝ≥0∞)) ^ Ky) + η +
        εks (reg σ') (cont σ') + εbind (reg σ') (cont σ') := by
  refine (extraction_audit_le A hks hlink (fun X _ => T ≤ SamplingDesign.unsoundWork σ w cl (A.incorrect X) ∨
    Ty ≤ SamplingDesign.unsoundWork σ v cl (A.incorrect X)) σ').trans ?_
  gcongr
  by_cases hB : T ≤ SamplingDesign.unsoundWork σ w cl (A.incorrect (A.committedOf σ')) ∨
      Ty ≤ SamplingDesign.unsoundWork σ v cl (A.incorrect (A.committedOf σ'))
  · exact (prCoin_mono fun ω h => h.2).trans (SamplingDesign.closure_escape_le_executionPhase_slack σ k hk hL cl hW hc hN hcy hB)
  · exact (prCoin_mono (q := fun _ => False) fun ω h => hB h.1).trans (prCoin_false.le.trans zero_le)

theorem extraction_audit_executionPhase_split_of_record_of_le_slack {C : ℕ} {call : Fin m → Fin C} {w v f : Fin m → ℕ}
    {L : SamplingDesign n} {η : ℝ≥0∞}
    (hL : ∀ B, L.missProb B ≤
      (SamplingDesign.stratified σ (SamplingDesign.executionPhaseK σ call w f 27713) (SamplingDesign.executionPhaseK_le σ call w f 27713)).missProb B + η)
    (A : ExtractionAnalysis (L.closure cl) Reg session)
    {εks εbind : (R : Reg) → Cont (L.closure cl) Reg session R → ℝ≥0∞}
    (hks : A.KnowledgeSound εks) (hlink : A.BindSound εbind) (hW : 0 < SamplingDesign.totalWork σ w)
    (hone : ∀ s, 0 < w s → SamplingDesign.callWork σ call w (call s) = w s * (SamplingDesign.stratum σ s).card)
    (hN : 0 < SamplingDesign.totalWork σ v) (hfy : ∀ s, 27713 * v s * (SamplingDesign.stratum σ s).card ≤ f s * SamplingDesign.totalWork σ v)
    (σ' : Strategy (audit (L.closure cl) Reg session)) :
    prob (fun o => o.1 = true ∧ (SamplingDesign.totalWork σ w ≤ 1000 * SamplingDesign.unsoundWork σ w cl (A.incorrect (A.committedOf σ')) ∨
        SamplingDesign.totalWork σ v ≤ 1000 * SamplingDesign.unsoundWork σ v cl (A.incorrect (A.committedOf σ'))))
        (audit (L.closure cl) Reg session) σ' ≤
      ((2 : ℝ≥0∞) ^ 40)⁻¹ + η + εks (reg σ') (cont σ') + εbind (reg σ') (cont σ') := by
  have hc := SamplingDesign.covers_executionPhase σ call w f 27713 hW hone
  have hcy := SamplingDesign.covers_of_floor σ _ f v 27713 (SamplingDesign.executionPhaseK_le σ call w f 27713)
    (fun _ => min_le_min le_rfl (le_max_left _ _)) hfy
  refine (extraction_audit_le A hks hlink (fun X _ => SamplingDesign.totalWork σ w ≤ 1000 * SamplingDesign.unsoundWork σ w cl (A.incorrect X) ∨
    SamplingDesign.totalWork σ v ≤ 1000 * SamplingDesign.unsoundWork σ v cl (A.incorrect X)) σ').trans ?_
  gcongr
  by_cases hB : SamplingDesign.totalWork σ w ≤ 1000 * SamplingDesign.unsoundWork σ w cl (A.incorrect (A.committedOf σ')) ∨
      SamplingDesign.totalWork σ v ≤ 1000 * SamplingDesign.unsoundWork σ v cl (A.incorrect (A.committedOf σ'))
  · refine (prCoin_mono fun ω h => h.2).trans ?_
    change (L.closure cl).missProb _ ≤ _
    rw [SamplingDesign.closure_escape]
    refine (hL _).trans (add_le_add ?_ le_rfl)
    rcases hB with hB | hB
    · exact (SamplingDesign.stratified_escape_le_of_covers σ _ _ w 27713 hW hc _).trans (SamplingDesign.pow_record_le hW hB)
    · exact (SamplingDesign.stratified_escape_le_of_covers σ _ _ v 27713 hN hcy _).trans (SamplingDesign.pow_record_le hN hB)
  · exact (prCoin_mono (q := fun _ => False) fun ω h => hB h.1).trans (prCoin_false.le.trans zero_le)

theorem audit_executionPhase_split_of_record_of_le_slack {C : ℕ} {call : Fin m → Fin C} {w v f : Fin m → ℕ} {L : SamplingDesign n}
    {η : ℝ≥0∞}
    (hL : ∀ B, L.missProb B ≤
      (SamplingDesign.stratified σ (SamplingDesign.executionPhaseK σ call w f 27713) (SamplingDesign.executionPhaseK_le σ call w f 27713)).missProb B + η)
    (A : Analysis (L.closure cl) Reg session) {εks εbind : ℝ≥0∞} (hks : A.KnowledgeSound εks)
    (hlink : A.BindSound εbind) (hW : 0 < SamplingDesign.totalWork σ w)
    (hone : ∀ s, 0 < w s → SamplingDesign.callWork σ call w (call s) = w s * (SamplingDesign.stratum σ s).card)
    (hN : 0 < SamplingDesign.totalWork σ v) (hfy : ∀ s, 27713 * v s * (SamplingDesign.stratum σ s).card ≤ f s * SamplingDesign.totalWork σ v)
    (σ' : Strategy (audit (L.closure cl) Reg session)) :
    prob (fun o => o.1 = true ∧ (SamplingDesign.totalWork σ w ≤ 1000 * SamplingDesign.unsoundWork σ w cl (A.incorrect (A.committedOf σ')) ∨
        SamplingDesign.totalWork σ v ≤ 1000 * SamplingDesign.unsoundWork σ v cl (A.incorrect (A.committedOf σ'))))
        (audit (L.closure cl) Reg session) σ' ≤ ((2 : ℝ≥0∞) ^ 40)⁻¹ + η + εks + εbind := by
  have hc := SamplingDesign.covers_executionPhase σ call w f 27713 hW hone
  have hcy := SamplingDesign.covers_of_floor σ _ f v 27713 (SamplingDesign.executionPhaseK_le σ call w f 27713)
    (fun _ => min_le_min le_rfl (le_max_left _ _)) hfy
  refine (audit_profile A hks hlink (fun B => SamplingDesign.totalWork σ w ≤ 1000 * SamplingDesign.unsoundWork σ w cl B ∨
    SamplingDesign.totalWork σ v ≤ 1000 * SamplingDesign.unsoundWork σ v cl B) σ').trans ?_
  gcongr
  refine iSup₂_le fun B hB => ?_
  rw [SamplingDesign.closure_escape]
  refine (hL _).trans (add_le_add ?_ le_rfl)
  rcases hB with hB | hB
  · exact (SamplingDesign.stratified_escape_le_of_covers σ _ _ w 27713 hW hc _).trans (SamplingDesign.pow_record_le hW hB)
  · exact (SamplingDesign.stratified_escape_le_of_covers σ _ _ v 27713 hN hcy _).trans (SamplingDesign.pow_record_le hN hB)

end Audits

end FlockSoundness.Audit
