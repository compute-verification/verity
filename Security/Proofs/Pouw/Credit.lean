import Properties.Pouw.Credit
import Proofs.Properties.Pouw
import Proofs.Properties.Pouw.Strata
import Proofs.Flock.Soundness.ExecStratified
import Proofs.Flock.Soundness.Audit.Work
import Proofs.Flock.VBridge.Full
import Proofs.Flock.Claims.Sampling.Draws
import Proofs.Flock.Claims.Sampling.Select.Misses
import Proofs.Accounting.Served

/-!
# Proofs: the ε a served window's credit reports (`WorkDrawMisses`, `EpsAtSound`, `EstablishedSound`)

The draw `drawnOn` runs over the strata `law` gives misses a set `B` only if every stratum's own `subset` draw misses
`B`'s units in it, and the streams are independent (`miss_le_prod`, the first steps of
`Law.stratified_exec_escape_le`): the miss probability is at most `∏_s C(n_s − m_s, k_s)/C(n_s, k_s)`, `m_s` the units
of stratum `s` in `B` (`localOf`). The work law's sizes make that product at most `((W − Σ_s w_s m_s)/W)^K`
(`prod_le_pow`, `Law.work_escape_le`'s weighted AM-GM over the strata, here with strata that may overlap and floors
that may be zero), and `B`'s work (`unitWork`) is at most `Σ_s w_s m_s` (`unitWork_le`). So a set of work `ε·C`
escapes with at most `(1 − εC/W)^K ≤ (1 − εC/W)^K′ ≤ δ` (`kPrimeLeast`) when `K′ < W`; at `K′ = W ≤ K` every stratum
doing work draws all its units, and a set of positive work escapes never.
-/

namespace Properties.Pouw.Credit.SecurityProofs

open Lean (Json toJson)
open FlockSoundness FlockSoundness.Game FlockSoundness.Audit Finset
open Properties.Pouw.Strata (unitWork)
open scoped ENNReal NNReal

/-- Stratum `s`'s local indices whose units are in `B`. -/
def localOf (S : Array Flock.Draw.Stratum) (B : Finset ℕ) (s : Fin S.size) : Finset (Fin S[s].size) :=
  univ.filter fun j => S[s].unit j.val ∈ B

theorem card_localOf_le (S : Array Flock.Draw.Stratum) (B : Finset ℕ) (s : Fin S.size) :
    (localOf S B s).card ≤ S[s].size :=
  (Finset.card_filter_le _ _).trans (by rw [Finset.card_univ, Fintype.card_fin])

/-- **The draw misses `B` no more often than each stratum's `subset` draw misses `B`'s units in it, independently.** -/
theorem miss_le_prod (S : Array Flock.Draw.Stratum) (hk : ∀ s : Fin S.size, S[s].k ≤ S[s].size) (B : Finset ℕ) :
    prCoin (fun g : (i : Fin S.size) → Fin S[i].need → Fin 256 => ∃ d, drawnOn S g = some d ∧ ∀ u ∈ B, u ∉ d) ≤
      ∏ s : Fin S.size, ((S[s].size - (localOf S B s).card).choose S[s].k : ℝ≥0∞) /
        (S[s].size.choose S[s].k : ℝ≥0∞) := by
  classical
  calc prCoin (fun g : (i : Fin S.size) → Fin S[i].need → Fin 256 => ∃ d, drawnOn S g = some d ∧ ∀ u ∈ B, u ∉ d)
      ≤ prCoin (fun g : (i : Fin S.size) → Fin S[i].need → Fin 256 => ∀ s ∈ (univ : Finset (Fin S.size)),
          (fun (s : Fin S.size) (f : Fin S[s].need → Fin 256) =>
            ∃ D, Flock.Draw.subset S[s].k S[s].size (Law.streamOf f) = some D ∧
            ∀ j ∈ localOf S B s, j.val ∉ D) s (g s)) := by
        refine prCoin_mono fun g ⟨D, hD, hmiss⟩ s _ => ?_
        have hD' : Flock.Draw.stratified S (Array.ofFn fun i => Properties.SampledProofs.bytesOf (g i)) = some D := hD
        obtain ⟨li, hli, hmem⟩ := Law.stratified_units _ _ D hD' s.val s.isLt
        have hSs : S[s.val]! = S[s] := getElem!_pos S s.val s.isLt
        have hfs : (Array.ofFn fun i => Properties.SampledProofs.bytesOf (g i)).getD s.val ByteArray.empty =
            Law.streamOf (g s) := by
          simp [Array.getD_eq_getD_getElem?, Properties.SampledProofs.bytesOf, Law.streamOf]
        rw [hSs, hfs] at hli
        rw [hSs] at hmem
        exact ⟨li, hli, fun j hj hjli => hmiss _ (Finset.mem_filter.1 hj).2 (hmem _ hjli)⟩
    _ = ∏ s : Fin S.size, prCoin (fun f : Fin S[s].need → Fin 256 =>
          ∃ D, Flock.Draw.subset S[s].k S[s].size (Law.streamOf f) = some D ∧ ∀ j ∈ localOf S B s, j.val ∉ D) :=
        prCoin_pi_forall (ι := Fin S.size) (Ω := fun s : Fin S.size => Fin S[s].need → Fin 256) univ
          (fun (s : Fin S.size) (f : Fin S[s].need → Fin 256) =>
          ∃ D, Flock.Draw.subset S[s].k S[s].size (Law.streamOf f) = some D ∧ ∀ j ∈ localOf S B s, j.val ∉ D)
    _ ≤ ∏ s : Fin S.size, (Law.subset S[s].size S[s].k (hk s)).escape (localOf S B s) :=
        Finset.prod_le_prod fun s _ => Law.subset_exec_escape_le _ _ _ (hk s) _
    _ = _ := Finset.prod_congr rfl fun s _ => Law.subset_escape (hk s) _

/-- `miss_le_prod` as `prob` reads it, the product in `ℝ≥0`. -/
theorem prob_le (S : Array Flock.Draw.Stratum) (hk : ∀ s : Fin S.size, S[s].k ≤ S[s].size) (B : Finset ℕ) :
    Pouw.RelativeSecurity.prob
        (fun g : (i : Fin S.size) → Fin S[i].need → Fin 256 => ∃ d, drawnOn S g = some d ∧ ∀ u ∈ B, u ∉ d) ≤
      ((∏ s : Fin S.size, ((S[s].size - (localOf S B s).card).choose S[s].k : ℝ≥0) /
        (S[s].size.choose S[s].k : ℝ≥0) : ℝ≥0) : ℝ) := by
  rw [Properties.Served.SecurityProofs.prob_eq_pr, Claims.Sampling.Pr_eq_toReal_prCoin]
  have hC : ∀ s : Fin S.size, ((S[s].size.choose S[s].k : ℕ) : ℝ≥0) ≠ 0 := fun s => by
    exact_mod_cast (Nat.choose_pos (hk s)).ne'
  have h := miss_le_prod S hk B
  have heq : (∏ s : Fin S.size, ((S[s].size - (localOf S B s).card).choose S[s].k : ℝ≥0∞) /
      (S[s].size.choose S[s].k : ℝ≥0∞)) =
      ((∏ s : Fin S.size, ((S[s].size - (localOf S B s).card).choose S[s].k : ℝ≥0) /
        (S[s].size.choose S[s].k : ℝ≥0) : ℝ≥0) : ℝ≥0∞) := by
    rw [ENNReal.ofNNReal_finsetProd]
    refine Finset.prod_congr rfl fun s _ => ?_
    rw [ENNReal.coe_div (hC s)]
    simp
  rw [heq] at h
  exact (ENNReal.toReal_mono ENNReal.coe_ne_top h).trans_eq (ENNReal.coe_toReal _)

/-- **The work law's weighted AM-GM**: strata of `N s` units, `k s` drawn and `M s` in the set, each unit doing `w s`,
draw sizes at least `K w_s N_s / W` (or all), escape with at most `((W − Σ_s w_s M_s)/W)^K`. -/
theorem prod_le_pow {ι : Type} [Fintype ι] (N M k w : ι → ℕ) (K W : ℕ) (hMN : ∀ s, M s ≤ N s) (hkN : ∀ s, k s ≤ N s)
    (hW : 0 < W) (hWsum : ∑ s, w s * N s = W) (hge : ∀ s, k s = N s ∨ K * w s * N s ≤ W * k s)
    (hk1 : ∀ s, 0 < w s → 0 < N s → 1 ≤ k s) :
    ∏ s, ((N s - M s).choose (k s) : ℝ≥0) / ((N s).choose (k s) : ℝ≥0) ≤
      (((W - ∑ s, w s * M s : ℕ) : ℝ≥0) / (W : ℝ≥0)) ^ K := by
  classical
  have hW0 : (W : ℝ≥0) ≠ 0 := by exact_mod_cast hW.ne'
  set x : ι → ℝ≥0 := fun s => ((N s - M s : ℕ) : ℝ≥0) / (N s : ℝ≥0) with hxdef
  set a : ι → ℝ≥0 := fun s => ((w s * N s : ℕ) : ℝ≥0) / (W : ℝ≥0) with hadef
  have ha1 : ∑ s, a s = 1 := by
    simp only [hadef]
    rw [← Finset.sum_div, ← Nat.cast_sum, hWsum, div_self hW0]
  have hterm : ∀ s, a s * x s = ((w s * (N s - M s) : ℕ) : ℝ≥0) / (W : ℝ≥0) := by
    intro s
    simp only [hadef, hxdef]
    rcases Nat.eq_zero_or_pos (N s) with h0 | hpos
    · have hM0 : M s = 0 := by have := hMN s; omega
      simp [h0, hM0]
    · have hN0 : (N s : ℝ≥0) ≠ 0 := by exact_mod_cast hpos.ne'
      rw [div_mul_div_comm, ← Nat.cast_mul, show w s * N s * (N s - M s) = w s * (N s - M s) * N s by ring,
        Nat.cast_mul, mul_div_mul_right _ _ hN0]
  have hax : ∑ s, a s * x s = ((W - ∑ s, w s * M s : ℕ) : ℝ≥0) / (W : ℝ≥0) := by
    rw [Finset.sum_congr rfl fun s _ => hterm s, ← Finset.sum_div, ← Nat.cast_sum]
    congr 2
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
    rcases Nat.eq_zero_or_pos (w s) with hw0 | hwpos
    · have ha0 : a s = 0 := by simp [hadef, hw0]
      rw [ha0, NNReal.coe_zero, zero_mul, NNReal.rpow_zero]
      exact div_le_one_of_le₀ (by exact_mod_cast Nat.choose_le_choose (k s) (Nat.sub_le (N s) (M s))) zero_le
    have hk1s : 1 ≤ k s := hk1 s hwpos hpos
    have hMNs := hMN s
    by_cases hzero : k s = N s ∨ M s = N s
    · have : (N s - M s).choose (k s) = 0 := Nat.choose_eq_zero_of_lt (by omega)
      rw [this, Nat.cast_zero, zero_div]
      exact zero_le
    push Not at hzero
    have hges : K * w s * N s ≤ W * k s := (hge s).resolve_left hzero.1
    have hx0 : 0 < x s := div_pos (by exact_mod_cast (show 0 < N s - M s by omega)) (by exact_mod_cast hpos)
    have hx1 : x s ≤ 1 := div_le_one_of_le₀ (by exact_mod_cast Nat.sub_le _ _) zero_le
    have hexp : (a s : ℝ) * K ≤ (k s : ℝ) := by
      simp only [hadef, NNReal.coe_div, NNReal.coe_natCast]
      rw [div_mul_eq_mul_div, div_le_iff₀ (by exact_mod_cast hW)]
      exact_mod_cast (show w s * N s * K ≤ k s * W by linarith [hges, mul_comm (K * w s) (N s)])
    calc ((N s - M s).choose (k s) : ℝ≥0) / ((N s).choose (k s) : ℝ≥0) ≤ x s ^ (k s) :=
          Law.choose_div_le_pow (hkN s) hpos
      _ = x s ^ ((k s : ℕ) : ℝ) := (NNReal.rpow_natCast _ _).symm
      _ ≤ x s ^ ((a s : ℝ) * K) := NNReal.rpow_le_rpow_of_exponent_ge hx0 hx1 hexp
  calc _ ≤ ∏ s, x s ^ ((a s : ℝ) * K) := Finset.prod_le_prod fun s _ => hfac s
    _ = (∏ s, x s ^ (a s : ℝ)) ^ K := by
        rw [← Finset.prod_pow]
        exact Finset.prod_congr rfl fun s _ => by rw [NNReal.rpow_mul, NNReal.rpow_natCast]
    _ ≤ (∑ s, a s * x s) ^ K :=
        pow_le_pow_left₀ zero_le (NNReal.geom_mean_le_arith_mean_weighted univ a x ha1) K
    _ = _ := by rw [hax]

/-- **The work bound over executable strata**: the draw over strata `S` sized by the work law at `K` (`hge`, `hk1`),
units of stratum `s` doing `w s` and all of them `W`, misses a set `B` with at most `((W − T)/W)^K` for any `T` at most
`Σ_s w_s m_s`. -/
theorem prob_le_pow (S : Array Flock.Draw.Stratum) (w : Fin S.size → ℕ) (K W T : ℕ) (hW : 0 < W)
    (hWsum : ∑ s, w s * S[s].size = W) (hkN : ∀ s : Fin S.size, S[s].k ≤ S[s].size)
    (hge : ∀ s : Fin S.size, S[s].k = S[s].size ∨ K * w s * S[s].size ≤ W * S[s].k)
    (hk1 : ∀ s : Fin S.size, 0 < w s → 0 < S[s].size → 1 ≤ S[s].k) (B : Finset ℕ)
    (hT : T ≤ ∑ s, w s * (localOf S B s).card) :
    Pouw.RelativeSecurity.prob
        (fun g : (i : Fin S.size) → Fin S[i].need → Fin 256 => ∃ d, drawnOn S g = some d ∧ ∀ u ∈ B, u ∉ d) ≤
      (((W - T : ℕ) : ℝ) / W) ^ K := by
  refine (prob_le S hkN B).trans ?_
  have h := prod_le_pow (fun s : Fin S.size => S[s].size) (fun s => (localOf S B s).card) (fun s => S[s].k) w K W
    (fun s => card_localOf_le S B s) hkN hW hWsum hge hk1
  have h2 : (((W - ∑ s, w s * (localOf S B s).card : ℕ) : ℝ≥0) / (W : ℝ≥0)) ^ K ≤
      (((W - T : ℕ) : ℝ≥0) / (W : ℝ≥0)) ^ K := by
    gcongr
  calc _ ≤ ((((((W - T : ℕ) : ℝ≥0) / (W : ℝ≥0)) ^ K : ℝ≥0)) : ℝ) := NNReal.coe_le_coe.2 (h.trans h2)
    _ = _ := by rw [NNReal.coe_pow, NNReal.coe_div, NNReal.coe_natCast, NNReal.coe_natCast]

/-- A unit doing work is one of a stratum's units, in a stratum doing that work. -/
theorem holder (ss : Array Flock.Draw.WorkStratum) (S : Array Flock.Draw.Stratum) (hsz : S.size = ss.size)
    (hunits : ∀ s : Fin S.size, S[s].units = ss[s.val]!.units) {u : ℕ} (hu : 0 < unitWork ss u) :
    ∃ s : Fin S.size, ss[s.val]!.work = unitWork ss u ∧ ∃ j : Fin S[s].size, S[s].unit j.val = u := by
  obtain ⟨i, hi, hhold, hwork⟩ := Properties.Pouw.Strata.unitWork_pos hu
  have hi' : i < S.size := by omega
  refine ⟨⟨i, hi'⟩, by rw [getElem!_pos ss i hi]; exact hwork, ?_⟩
  have hany : S[(⟨i, hi'⟩ : Fin S.size)].units.any (fun (lo, hi) => lo ≤ u && u < hi) = true := by
    rw [hunits, getElem!_pos ss i hi]
    exact hhold
  obtain ⟨j, hj, hju⟩ := FlockVBridge.Full.mem_unit _ u hany
  exact ⟨⟨j, hj⟩, hju⟩

/-- The units of stratum `s`'s local indices in `B`. -/
def unitsIn (S : Array Flock.Draw.Stratum) (B : Finset ℕ) (s : Fin S.size) : Finset ℕ :=
  (localOf S B s).image fun j => S[s].unit j.val

/-- **`B`'s work is at most `Σ_s w_s m_s`**: each unit of `B` doing work is a unit of a stratum doing that work. -/
theorem unitWork_le (ss : Array Flock.Draw.WorkStratum) (S : Array Flock.Draw.Stratum) (hsz : S.size = ss.size)
    (hunits : ∀ s : Fin S.size, S[s].units = ss[s.val]!.units) (B : Finset ℕ) :
    ∑ u ∈ B, unitWork ss u ≤ ∑ s : Fin S.size, ss[s.val]!.work * (localOf S B s).card := by
  classical
  have hone : ∀ u ∈ B, unitWork ss u ≤
      ∑ s : Fin S.size, if u ∈ unitsIn S B s then ss[s.val]!.work else 0 := by
    intro u hu
    rcases Nat.eq_zero_or_pos (unitWork ss u) with h0 | hpos
    · rw [h0]
      exact Nat.zero_le _
    obtain ⟨s, hws, j, hju⟩ := holder ss S hsz hunits hpos
    have hmem : u ∈ unitsIn S B s :=
      Finset.mem_image.2 ⟨j, Finset.mem_filter.2 ⟨Finset.mem_univ _, by rw [hju]; exact hu⟩, hju⟩
    calc unitWork ss u = if u ∈ unitsIn S B s then ss[s.val]!.work else 0 := by rw [ite_eq_left hmem, hws]
      _ ≤ _ := Finset.single_le_sum (f := fun s => if u ∈ unitsIn S B s then ss[s.val]!.work else 0)
          (fun _ _ => Nat.zero_le _) (Finset.mem_univ s)
  calc ∑ u ∈ B, unitWork ss u
      ≤ ∑ u ∈ B, ∑ s : Fin S.size, (if u ∈ unitsIn S B s then ss[s.val]!.work else 0) := Finset.sum_le_sum hone
    _ = ∑ s : Fin S.size, ∑ u ∈ B, (if u ∈ unitsIn S B s then ss[s.val]!.work else 0) := Finset.sum_comm
    _ = ∑ s : Fin S.size, ss[s.val]!.work * (B.filter (· ∈ unitsIn S B s)).card := by
        refine Finset.sum_congr rfl fun s _ => ?_
        rw [← Finset.sum_filter, Finset.sum_const, smul_eq_mul, mul_comm]
    _ ≤ ∑ s : Fin S.size, ss[s.val]!.work * (localOf S B s).card := by
        refine Finset.sum_le_sum fun s _ => Nat.mul_le_mul_left _ ?_
        calc (B.filter (· ∈ unitsIn S B s)).card ≤ (unitsIn S B s).card :=
              Finset.card_le_card fun x hx => (Finset.mem_filter.1 hx).2
          _ ≤ (localOf S B s).card := Finset.card_image_le

/-- The strata's work, read stratum by stratum over the law's strata `S` (as many as `ss`). -/
theorem sum_work (ss : Array Flock.Draw.WorkStratum) (S : Array Flock.Draw.Stratum) (hsz : S.size = ss.size) :
    ∑ s : Fin S.size, ss[s.val]!.work * ss[s.val]!.size = Flock.Draw.totalWork ss := by
  rw [Properties.SampledProofs.SecurityProofs.totalWork_eq, ← Fin.sum_univ_fun_getElem]
  refine Fintype.sum_equiv (finCongr (hsz.trans Array.length_toList.symm)) _ _ fun s => ?_
  have hs : s.val < ss.size := by omega
  simp only [finCongr_apply_coe, Array.getElem_toList, getElem!_pos ss s.val hs]
  exact Nat.mul_comm _ _

theorem workDrawMisses : WorkDrawMisses := by
  intro layout ss K strata eps δ C k' _ hlaw hk' B hk'K hB
  classical
  obtain ⟨hK1, -, hstrata⟩ := Properties.Pouw.Strata.law_ok hlaw
  have hsz : strata.size = ss.size := by rw [hstrata, Array.size_map]
  have hS : ∀ s : Fin strata.size,
      strata[s] = Properties.Pouw.Strata.lawOf K (Flock.Draw.totalWork ss) ss[s.val]! := by
    subst hstrata
    exact Properties.Pouw.Strata.getElem_lawOf ss K _
  have hWsum' := sum_work ss strata hsz
  generalize hWdef : Flock.Draw.totalWork ss = W at hS hk' hWsum'
  obtain ⟨he0, he1, hd0, -, hW1, -, hC0, hCW⟩ := Properties.Pouw.Window.kPrimeRefuses _ _ _ _ _ _ hk'
  have hq := Properties.Pouw.Window.kPrimeLeast _ _ _ _ _ _ hk'
  dsimp only at hq
  obtain ⟨-, hk'W, hlt, -⟩ := hq
  simp only [Int.toNat_natCast] at hk'W hlt
  have hW : 0 < W := by omega
  have hCW' : C ≤ W := by omega
  have hN : ∀ s : Fin strata.size, strata[s].size = ss[s.val]!.size := fun s => by rw [hS s]; rfl
  have hunits : ∀ s : Fin strata.size, strata[s].units = ss[s.val]!.units := fun s => by rw [hS s]; rfl
  have hkN : ∀ s : Fin strata.size, strata[s].k ≤ strata[s].size := fun s => by
    rw [hS s]
    exact Properties.Pouw.Strata.workK_le _ _ _ _ _
  rcases Nat.lt_or_ge k' W with hlt' | hge'
  · have hWsum : ∑ s : Fin strata.size, ss[s.val]!.work * strata[s].size = W := by
      rw [← hWsum']
      exact Finset.sum_congr rfl fun s _ => by rw [hN s]
    have hge : ∀ s : Fin strata.size, strata[s].k = strata[s].size ∨
        K * ss[s.val]!.work * strata[s].size ≤ W * strata[s].k := fun s => by
      rw [hS s]
      exact Properties.Pouw.Strata.workK_ge K _ _ W _ hW
    have hk1 : ∀ s : Fin strata.size, 0 < ss[s.val]!.work → 0 < strata[s].size → 1 ≤ strata[s].k := by
      intro s hw hn
      rw [hS s] at hn ⊢
      exact Properties.Pouw.Strata.workK_pos K _ _ W _ (Nat.mul_pos (Nat.mul_pos hK1 hw) hn) hW
    have hT := unitWork_le ss strata hsz hunits B
    have hP := prob_le_pow strata (fun s => ss[s.val]!.work) K W (∑ u ∈ B, unitWork ss u) hW hWsum hkN hge hk1 B hT
    have hTW : ∑ u ∈ B, unitWork ss u ≤ W := by
      refine hT.trans ?_
      rw [← hWsum]
      exact Finset.sum_le_sum fun s _ => Nat.mul_le_mul_left _ (card_localOf_le _ _ _)
    have hWr : (0 : ℝ) < W := by exact_mod_cast hW
    have he0r : (0 : ℝ) < eps := by exact_mod_cast he0
    have he1r : (eps : ℝ) ≤ 1 := by exact_mod_cast he1
    have hBr : (eps : ℝ) * C ≤ ((∑ u ∈ B, unitWork ss u : ℕ) : ℝ) := by exact_mod_cast hB
    have hx : ((W - ∑ u ∈ B, unitWork ss u : ℕ) : ℝ) / W ≤ 1 - (eps : ℝ) * C / W := by
      rw [Nat.cast_sub hTW, sub_div, div_self hWr.ne']
      have := (div_le_div_iff_of_pos_right hWr).2 hBr
      linarith
    have hx0 : 0 ≤ ((W - ∑ u ∈ B, unitWork ss u : ℕ) : ℝ) / W := div_nonneg (Nat.cast_nonneg _) hWr.le
    have hCWr : (eps : ℝ) * C ≤ W :=
      calc (eps : ℝ) * C ≤ 1 * C := mul_le_mul_of_nonneg_right he1r (Nat.cast_nonneg _)
        _ = C := one_mul _
        _ ≤ W := by exact_mod_cast hCW'
    have hq0 : 0 ≤ 1 - (eps : ℝ) * C / W := by
      rw [sub_nonneg, div_le_one₀ hWr]
      exact hCWr
    have hq1 : 1 - (eps : ℝ) * C / W ≤ 1 := by
      have : 0 ≤ (eps : ℝ) * C / W := div_nonneg (mul_nonneg he0r.le (Nat.cast_nonneg _)) hWr.le
      linarith
    have hδq : (((1 - PouwWindow.missShare eps C W 1) ^ k' : ℚ) : ℝ) ≤ (δ : ℝ) := Rat.cast_le.2 (hlt hlt')
    have hδ : (1 - (eps : ℝ) * C / W) ^ k' ≤ (δ : ℝ) := by
      convert hδq using 1
      rw [PouwWindow.missShare]
      push_cast
      ring
    calc _ ≤ (((W - ∑ u ∈ B, unitWork ss u : ℕ) : ℝ) / W) ^ K := hP
      _ ≤ (1 - (eps : ℝ) * C / W) ^ K := pow_le_pow_left₀ hx0 hx K
      _ ≤ (1 - (eps : ℝ) * C / W) ^ k' := pow_le_pow_of_le_one hq0 hq1 hk'K
      _ ≤ δ := hδ
  · have hWK : W ≤ K := by omega
    have hTpos : 0 < ∑ u ∈ B, unitWork ss u := by
      have h1 : (0 : ℚ) < eps * C := mul_pos he0 (by exact_mod_cast (show 0 < C by omega))
      exact_mod_cast h1.trans_le hB
    obtain ⟨u, hu, hu0⟩ := Finset.exists_ne_zero_of_sum_ne_zero hTpos.ne'
    obtain ⟨s, hws, j, hju⟩ := holder ss strata hsz hunits (Nat.pos_of_ne_zero hu0)
    have hjl : j ∈ localOf strata B s := Finset.mem_filter.2 ⟨Finset.mem_univ _, by rw [hju]; exact hu⟩
    have hM : 1 ≤ (localOf strata B s).card := Finset.card_pos.2 ⟨j, hjl⟩
    have hMN := card_localOf_le strata B s
    have hkall : strata[s].k = strata[s].size := by
      rw [hS s]
      exact Properties.Pouw.Strata.workK_cap K _ _ W _ hWK (by rw [hws]; omega) hW
    have hzero : ((strata[s].size - (localOf strata B s).card).choose strata[s].k : ℝ≥0) /
        (strata[s].size.choose strata[s].k : ℝ≥0) = 0 := by
      rw [Nat.choose_eq_zero_of_lt (by omega), Nat.cast_zero, zero_div]
    have h0 : (∏ s : Fin strata.size, ((strata[s].size - (localOf strata B s).card).choose strata[s].k : ℝ≥0) /
        (strata[s].size.choose strata[s].k : ℝ≥0)) = 0 :=
      Finset.prod_eq_zero (Finset.mem_univ s) hzero
    calc _ ≤ _ := prob_le strata hkN B
      _ = 0 := by rw [h0, NNReal.coe_zero]
      _ ≤ δ := by exact_mod_cast hd0.le

theorem epsAtSound : EpsAtSound := by
  intro layout ss K strata δ e C hss hlaw he B hB
  obtain ⟨k', hk', hk'K⟩ := Properties.Pouw.Strata.epsAt_ok he
  exact workDrawMisses layout ss K strata e δ C k' hss hlaw hk' B (by exact_mod_cast hk'K) hB

theorem sum_range_list (g : ℕ → ℕ) : ∀ n, ∑ i ∈ Finset.range n, g i = ((List.range n).map g).sum
  | 0 => by simp
  | n + 1 => by
    rw [Finset.sum_range_succ, List.range_succ, List.map_append, List.sum_append, sum_range_list g n]
    simp

/-- `EstablishedSound` from the tiling: the total work `established` passes `epsAt` is the strata's, and its credited
work is `creditedWork`. -/
theorem establishedSound_of (htile : Properties.Pouw.Strata.StrataTile) : EstablishedSound := by
  intro layout ss K strata got served e hss hlaw hest B hB
  obtain ⟨-, -, -, htotal, -⟩ := htile layout ss hss
  have hok := Properties.Pouw.Strata.established_ok hest
  have key : ((List.range (PouwCheck.Credit.servedWork layout served).length).map fun i =>
      if i ∈ got then ((PouwCheck.Credit.servedWork layout served).getD i (0, 0)).1 *
        ((PouwCheck.Credit.servedWork layout served).getD i (0, 0)).2 else 0).sum =
      creditedWork layout got served := by
    rw [← sum_range_list, creditedWork, Finset.sum_filter]
  rw [key, ← htotal] at hok
  exact epsAtSound layout ss K strata PouwCheck.Credit.DELTA e (creditedWork layout got served) hss hlaw hok B hB

/-- **The ε a credit reports is established** (`EstablishedSound`): `strata` tiles the window's units
(`Properties.Pouw.Strata.strataTile`), so the total work `established` passes `epsAt` is the strata's. -/
theorem establishedSound : EstablishedSound := establishedSound_of Properties.Pouw.Strata.strataTile

/-- **The grouped draw misses a set of blocks of work `ε·C` with probability at most `δ`** (`GroupedDrawMisses`): the
blocks of `Bb` miss every drawn group exactly when their leaves `Bb.image leafOf` are all missed, and those leaves do at
least the blocks' work, so `workDrawMisses` at `B = Bb.image leafOf`. -/
theorem groupedDrawMisses : GroupedDrawMisses := by
  intro layout ss K strata eps δ C k' hss hlaw hk' β leafOf w Bb hk'K hw hB
  classical
  have hle : ∑ b ∈ Bb, w b ≤ ∑ u ∈ Bb.image leafOf, unitWork ss u := by
    rw [← Finset.sum_fiberwise_of_maps_to (fun b hb => Finset.mem_image_of_mem leafOf hb)]
    exact Finset.sum_le_sum fun u _ => hw u
  have hev : (fun g : (i : Fin strata.size) → Fin strata[i].need → Fin 256 =>
      ∃ d, drawnOn strata g = some d ∧ ∀ b ∈ Bb, leafOf b ∉ d) =
      fun g => ∃ d, drawnOn strata g = some d ∧ ∀ u ∈ Bb.image leafOf, u ∉ d := by
    funext g
    simp only [Finset.forall_mem_image]
  rw [hev]
  exact workDrawMisses layout ss K strata eps δ C k' hss hlaw hk' (Bb.image leafOf) hk'K
    (hB.trans (Nat.cast_le.mpr hle))

end Properties.Pouw.Credit.SecurityProofs
