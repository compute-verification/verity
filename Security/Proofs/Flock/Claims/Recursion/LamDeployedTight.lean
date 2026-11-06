import Proofs.Flock.Claims.Recursion.DeployedBringsLam

/-!
# Route 2C's bound at the deployed shape: `λ₂ ≥ 135.7`

`lamLin2_deployedB_ge` bounds `λ₂` below by `100`, each of its six terms by `103`. At the same hypotheses and caps,
the terms' tighter bounds give `λ₂ ≥ 135.7` (`lamLin2_deployedB_ge_tight`; `lamLin2_deployed_ge_dep_tight` at every
`v ≤ 2^55`): each term is at least `138.3`, and `log₂ 6 ≤ 2.6` (`6^5 ≤ 2^13`). The bounds are exact rational and real
inequalities:

* `k_I`, `k_O ≤ 2468800000` (`k_le_200`: `⌈4·N₀/e⌉` at `N₀ ≤ 200·2^23`, and `4·200·2^23/e < 2468797139`), so
  `N_I ≤ 1 + 210·2468800000` and `c_S ≤ 4 + 420·2^16·2468800000 + 3·2^56 ≤ 2^58` at `|S(x_I)| ≤ 2^56`, `n ≤ 2^55`
  and `v ≤ 2^55`;
* the inner verifier's rounds `≤ 880002` (`2 + 4400·200`);
* the deployed caps' positions `W ≤ 5591040` (`CpDep_width_le_tight`): two reps of table 0's `fast100` levels after
  level 0, widest at `m = 35`, `2·(2^21 + 2^19 + 2^17 + 2^15 + 2^13 + 2^11)`.

Fdev's term, `255 − log₂(β₂·c_S)`, binds (`lamLin2_core_tight`): at the hypotheses' caps (`200` inner tables, `2^16`
V\* parts of `200` tables each, `|S(x_I)| = 2^56`, `n = v = 2^55`) these bounds put it at about `138.33`, `κ`'s at about
`140.4` and the rest above `146`, so a larger `L` needs smaller caps on `n`, `v`, `|S(x_I)|`, `nS x` or the tables.
-/

namespace RecDeploy

open Claims (Msg Out)
open FlockDeploy (Coin)

/-! ## The arithmetic -/

theorem logb_le_div_of_pow_le {y : ℝ} {q N : ℕ} (h0 : 0 ≤ y) (hq : 0 < q) (h : y ^ q ≤ 2 ^ N) :
    Real.logb 2 y ≤ N / q := by
  have hq' : (0 : ℝ) < q := by exact_mod_cast hq
  rcases h0.eq_or_lt with rfl | hy
  · rw [Real.logb_zero]; positivity
  · rw [le_div_iff₀ hq']
    calc Real.logb 2 y * q = Real.logb 2 (y ^ q) := by rw [Real.logb_pow]; ring
      _ ≤ Real.logb 2 ((2 : ℝ) ^ N) := Real.logb_le_logb_of_le (by norm_num) (by positivity) h
      _ = N := by rw [Real.logb_pow, Real.logb_self_eq_one (by norm_num)]; ring

theorem le_sub_logb {a b y : ℝ} {n : ℕ} (h0 : 0 ≤ y) (h : y ≤ 2 ^ n) (ha : (n : ℝ) + b ≤ a) :
    b ≤ a - Real.logb 2 y := by
  have := logb_le_of_le_pow h0 h
  linarith

theorem logb_six_le_tight : Real.logb 2 6 ≤ 2.6 := by
  have := logb_le_div_of_pow_le (y := 6) (q := 5) (N := 13) (by norm_num) (by norm_num) (by norm_num)
  have e : ((13 : ℕ) : ℝ) / ((5 : ℕ) : ℝ) = 2.6 := by norm_num
  linarith

/-- **`⌈4·N₀/e⌉ ≤ 2468800000`** at `N₀ ≤ 200·2^23`: `4·200·2^23/e < 2468797139`. -/
theorem k_le_200 {N : ℝ} (hN : N ≤ 200 * 2 ^ 23) : ⌈4 * N / Real.exp 1⌉₊ ≤ 2468800000 := by
  have he0 : 0 < Real.exp 1 := Real.exp_pos 1
  have hlo := Real.exp_one_gt_d9
  have h2 : (200 : ℝ) * 2 ^ 23 = 1677721600 := by norm_num
  refine Nat.ceil_le.2 ?_
  rw [div_le_iff₀ he0]
  push_cast
  linarith

/-- **`λ₂ ≥ 135.7` from its numbers**: `lamLin2`'s six terms, over the numbers they read as in `lamLin2_core`, each at
least `138.3`, at the inner verifier's rounds `VI ≤ 880002`, the inner and each part's `E^rep` reruns per wait at most
`2468800000`, the inner caps' positions `W ≤ 5591040` and `|S(x_I)| + 2(n + v) ≤ 3·2^56`. Fdev's term binds. -/
theorem lamLin2_core_tight {nS : ℕ} {VI kI wI W kx NI cBe nSet nU : ℕ} {κI : ℝ} {κs γs : Fin nS → ℝ}
    {ws ks : Fin nS → ℕ} (hNI : NI = 1 + wI * kI) (hcBe : cBe = 4 + 2 * ∑ s, ws s * ks s)
    (hSet : nSet + 2 * nU ≤ 3 * 2 ^ 56) (hVI : VI ≤ 880002) (hkI : kI ≤ 2468800000) (hwI : wI = 210)
    (hW : W ≤ 5591040) (hnS : nS ≤ 2 ^ 16) (hkx : kx ≤ 2 ^ 37) (hκI0 : 0 < κI)
    (hκI : κI ≤ 200 * 2 ^ (-(202.98 : ℝ))) (hκs0 : ∀ s, 0 ≤ κs s) (hκs : ∀ s, κs s ≤ 200 * 2 ^ (-(202.98 : ℝ)))
    (hγs : ∀ s, γs s ≤ 2 ^ 68) (hws : ∀ s, ws s = 210) (hks : ∀ s, ks s ≤ 2468800000) :
    135.7 ≤ min (min (min (min (min (Real.logb 2 (1 / (κI + NI * ∑ s, κs s)))
      (255.5 - Real.logb 2 ((NI * ∑ s, Real.sqrt (γs s * kx)) ^ 2) / 2))
      (256 - Real.logb 2 ((4 * VI + 1) * ((cBe : ℝ) + nSet + 2 * nU))))
      (255 - Real.logb 2 (4 * W * ((cBe : ℝ) + nSet + 2 * nU))))
      (256 - Real.logb 2 (4 * ((kI : ℝ) + 1) * ((cBe : ℝ) + nSet + 2 * nU))))
      (255 - Real.logb 2 ((NI + 8 * W + 4 * ((kI : ℝ) + 1)) * VI * ((cBe : ℝ) + nSet + 2 * nU))) -
      Real.logb 2 6 := by
  have hNI' : NI ≤ 518448000001 := by
    rw [hNI, hwI]
    calc 1 + 210 * kI ≤ 1 + 210 * 2468800000 := by gcongr
      _ = 518448000001 := by norm_num
  have hcBe' : cBe ≤ 4 + 2 * (2 ^ 16 * (210 * 2468800000)) := by
    have hsum : ∑ s, ws s * ks s ≤ nS * (210 * 2468800000) :=
      sum_fin_le_nat _ fun s => by rw [hws s]; exact Nat.mul_le_mul_left 210 (hks s)
    rw [hcBe]
    gcongr
    exact hsum.trans (Nat.mul_le_mul_right _ hnS)
  have hcS : cBe + nSet + 2 * nU ≤ 4 + 2 * (2 ^ 16 * (210 * 2468800000)) + 3 * 2 ^ 56 := by omega
  have hcS' : cBe + nSet + 2 * nU ≤ 2 ^ 58 := hcS.trans (by norm_num)
  have hNIR : (NI : ℝ) ≤ 518448000001 := by exact_mod_cast hNI'
  have hNI40 : (NI : ℝ) ≤ 2 ^ 40 := hNIR.trans (by norm_num)
  have hkxR : (kx : ℝ) ≤ 2 ^ 37 := by exact_mod_cast hkx
  rw [le_sub_iff_add_le]
  refine le_trans (b := (138.3 : ℝ)) (by linarith [logb_six_le_tight]) ?_
  refine le_min (le_min (le_min (le_min (le_min ?_ ?_) ?_) ?_) ?_) ?_
  · have hS0 : 0 ≤ ∑ s, κs s := Finset.sum_nonneg fun s _ => hκs0 s
    have hS : ∑ s, κs s ≤ 2 ^ 16 * (200 * 2 ^ (-(202.98 : ℝ))) := by
      refine (sum_fin_le_real _ hκs).trans ?_
      have : (nS : ℝ) ≤ 2 ^ 16 := by exact_mod_cast hnS
      gcongr
    have hκ0 : 0 < κI + NI * ∑ s, κs s := by
      have := mul_nonneg (Nat.cast_nonneg NI : (0 : ℝ) ≤ NI) hS0
      linarith
    have hκ : κI + NI * ∑ s, κs s ≤
        200 * 2 ^ (-(202.98 : ℝ)) + 518448000001 * (2 ^ 16 * (200 * 2 ^ (-(202.98 : ℝ)))) := by
      have := mul_le_mul hNIR hS hS0 (by norm_num)
      linarith
    have he := two_rpow_neg_le
    have key : (2 : ℝ) ^ 139 * (κI + NI * ∑ s, κs s) ≤ 1 :=
      calc (2 : ℝ) ^ 139 * (κI + NI * ∑ s, κs s)
          ≤ 2 ^ 139 * (200 * 2 ^ (-(202.98 : ℝ)) + 518448000001 * (2 ^ 16 * (200 * 2 ^ (-(202.98 : ℝ))))) := by
            gcongr
        _ = (2 ^ 139 * 200 * (1 + 518448000001 * 2 ^ 16)) * (2 : ℝ) ^ (-(202.98 : ℝ)) := by ring
        _ ≤ (2 ^ 139 * 200 * (1 + 518448000001 * 2 ^ 16)) * ((2 : ℝ) ^ 202)⁻¹ := by gcongr
        _ ≤ 1 := by norm_num
    have := le_logb_inv (n := 139) hκ0 key
    push_cast at this
    linarith
  · have hsq : ∀ s, Real.sqrt (γs s * kx) ≤ 2 ^ 53 := fun s => by
      rw [Real.sqrt_le_left (by norm_num)]
      calc γs s * kx ≤ 2 ^ 68 * kx := mul_le_mul_of_nonneg_right (hγs s) (Nat.cast_nonneg _)
        _ ≤ 2 ^ 68 * 2 ^ 37 := by gcongr
        _ ≤ (2 ^ 53) ^ 2 := by norm_num
    have hT0 : 0 ≤ ∑ s, Real.sqrt (γs s * kx) := Finset.sum_nonneg fun s _ => Real.sqrt_nonneg _
    have hT : ∑ s, Real.sqrt (γs s * kx) ≤ 2 ^ 16 * 2 ^ 53 := by
      refine (sum_fin_le_real _ hsq).trans ?_
      have : (nS : ℝ) ≤ 2 ^ 16 := by exact_mod_cast hnS
      gcongr
    have hγ : ((NI : ℝ) * ∑ s, Real.sqrt (γs s * kx)) ^ 2 ≤ 2 ^ 218 :=
      calc ((NI : ℝ) * ∑ s, Real.sqrt (γs s * kx)) ^ 2 ≤ (2 ^ 40 * (2 ^ 16 * 2 ^ 53)) ^ 2 :=
            pow_le_pow_left₀ (mul_nonneg (Nat.cast_nonneg _) hT0) (mul_le_mul hNI40 hT hT0 (by norm_num)) 2
        _ = 2 ^ 218 := by ring
    have := logb_le_of_le_pow (n := 218) (by positivity) hγ
    push_cast at this
    linarith
  · refine le_sub_logb (n := 80) (by positivity) ?_ (by norm_num)
    have : (4 * VI + 1) * (cBe + nSet + 2 * nU) ≤ 2 ^ 80 :=
      calc (4 * VI + 1) * (cBe + nSet + 2 * nU) ≤ (4 * 880002 + 1) * 2 ^ 58 := by gcongr
        _ ≤ 2 ^ 80 := by norm_num
    exact_mod_cast this
  · refine le_sub_logb (n := 83) (by positivity) ?_ (by norm_num)
    have : 4 * W * (cBe + nSet + 2 * nU) ≤ 2 ^ 83 :=
      calc 4 * W * (cBe + nSet + 2 * nU) ≤ 4 * 5591040 * 2 ^ 58 := by gcongr
        _ ≤ 2 ^ 83 := by norm_num
    exact_mod_cast this
  · refine le_sub_logb (n := 92) (by positivity) ?_ (by norm_num)
    have : 4 * (kI + 1) * (cBe + nSet + 2 * nU) ≤ 2 ^ 92 :=
      calc 4 * (kI + 1) * (cBe + nSet + 2 * nU) ≤ 4 * (2468800000 + 1) * 2 ^ 58 := by gcongr
        _ ≤ 2 ^ 92 := by norm_num
    exact_mod_cast this
  · have hy : (NI + 8 * W + 4 * (kI + 1)) * VI * (cBe + nSet + 2 * nU) ≤
        (518448000001 + 8 * 5591040 + 4 * (2468800000 + 1)) * 880002 *
          (4 + 2 * (2 ^ 16 * (210 * 2468800000)) + 3 * 2 ^ 56) :=
      Nat.mul_le_mul (Nat.mul_le_mul (by gcongr) hVI) hcS
    have hy0 : (0 : ℝ) ≤ ((NI : ℝ) + 8 * W + 4 * ((kI : ℝ) + 1)) * VI * ((cBe : ℝ) + nSet + 2 * nU) := by
      positivity
    -- `2^1167` is past `exponentiation.threshold`, so the comparison is the kernel's, on `ℕ`, and no tactic that
    -- reads the context (`positivity`, `linarith`) runs while it is there.
    have hY : ((518448000001 + 8 * 5591040 + 4 * (2468800000 + 1)) * 880002 *
        (4 + 2 * (2 ^ 16 * (210 * 2468800000)) + 3 * 2 ^ 56)) ^ 10 ≤ 2 ^ 1167 := by decide +kernel
    have hp : ((NI + 8 * W + 4 * (kI + 1)) * VI * (cBe + nSet + 2 * nU)) ^ 10 ≤ 2 ^ 1167 :=
      (Nat.pow_le_pow_left hy 10).trans hY
    have hpR : (((NI : ℝ) + 8 * W + 4 * ((kI : ℝ) + 1)) * VI * ((cBe : ℝ) + nSet + 2 * nU)) ^ 10 ≤ 2 ^ 1167 := by
      have := (Nat.cast_le (α := ℝ)).2 hp
      simpa only [Nat.cast_pow, Nat.cast_mul, Nat.cast_add, Nat.cast_one, Nat.cast_ofNat] using this
    have := logb_le_div_of_pow_le (q := 10) (N := 1167) hy0 (by norm_num) hpR
    have e : ((1167 : ℕ) : ℝ) / ((10 : ℕ) : ℝ) = 116.7 := by norm_num
    clear hY hp hpR hy
    linarith

/-- **`λ₂ ≥ 135.7` at a compiled record and an inner protocol**, from bounds on the numbers `lamLin2` reads. -/
theorem lamLin2_ge_of_tight {C W' : Type} [Fintype C] [Nonempty C] [Inhabited C] (D : Claims.Recursion.Deploy C)
    (I : Claims.Recursion.Inner C W') (Cp : Claims.Recursion.Caps C) (x : Msg) (n v : ℕ)
    (hVI : I.V.k (Claims.Recursion.xI D x) ≤ 880002) (hkI : I.k (Claims.Recursion.xI D x) ≤ 2468800000)
    (hwI : I.waits (Claims.Recursion.xI D x) = 210) (hW : Cp.width (Claims.Recursion.xI D x) ≤ 5591040)
    (hnS : D.nS x ≤ 2 ^ 16) (hSet : Claims.Recursion.setupLen D x ≤ 2 ^ 56) (hU : n ≤ 2 ^ 55) (hV : v ≤ 2 ^ 55)
    (hkx : D.V.k x ≤ 2 ^ 37)
    (hκI0 : 0 < I.κ (Claims.Recursion.xI D x))
    (hκI : I.κ (Claims.Recursion.xI D x) ≤ 200 * 2 ^ (-(202.98 : ℝ)))
    (hκs0 : ∀ s : Fin (D.nS x), 0 ≤ (D.nums x s).κ)
    (hκs : ∀ s : Fin (D.nS x), (D.nums x s).κ ≤ 200 * 2 ^ (-(202.98 : ℝ)))
    (hγs : ∀ s : Fin (D.nS x), (D.nums x s).γ ≤ 2 ^ 68) (hws : ∀ s : Fin (D.nS x), (D.nums x s).waits = 210)
    (hks : ∀ s : Fin (D.nS x), (D.nums x s).k ≤ 2468800000) :
    135.7 ≤ Claims.Recursion.lamLin2 D I Cp x n v :=
  lamLin2_core_tight (nS := D.nS x) (VI := I.V.k (Claims.Recursion.xI D x)) (kI := I.k (Claims.Recursion.xI D x))
    (wI := I.waits (Claims.Recursion.xI D x)) (W := Cp.width (Claims.Recursion.xI D x)) (kx := D.V.k x)
    (NI := Claims.Recursion.NI D I x) (cBe := Claims.Recursion.cBe D x) (nSet := Claims.Recursion.setupLen D x)
    (nU := n + v) (κI := I.κ (Claims.Recursion.xI D x)) (κs := fun s => (D.nums x s).κ)
    (γs := fun s => (D.nums x s).γ) (ws := fun s => (D.nums x s).waits) (ks := fun s => (D.nums x s).k) rfl rfl
    (by omega) hVI hkI hwI hW hnS hkx hκI0 hκI hκs0 hκs hγs hws hks

/-! ## At the deployed caps -/

/-- At every `m` `fast100` serves, the levels after level 0 hold at most `2795520` positions together, the most at
`m = 35`: `2^21 + 2^19 + 2^17 + 2^15 + 2^13 + 2^11`. -/
theorem fast100_capSum_le {m : ℕ} {sch : FlockSoundness.Accounting.Schedule}
    (h : FlockSoundness.Accounting.fast100 m = some sch) :
    ∑ ℓ ∈ Finset.range (sch.levels.length - 1), 2 ^ (FlockSoundness.lvOf sch (ℓ + 1)).logLen ≤ 2795520 := by
  have hm : 22 ≤ m ∧ m ≤ 35 := by
    by_contra hc
    have : FlockSoundness.Accounting.fast100 m = none := by unfold FlockSoundness.Accounting.fast100; split <;> simp_all
    simp [this] at h
  obtain ⟨h1, h2⟩ := hm
  interval_cases m <;> (simp only [FlockSoundness.Accounting.fast100, Option.some.injEq] at h; subst h; decide +kernel)

/-- **The deployed caps have at most `5591040` positions**: two reps of table 0's `fast100` levels after level 0
(`fast100_capSum_le`). -/
theorem CpDep_width_le_tight (x : Msg) : FlockDeploy.KS.CpDep.width x ≤ 5591040 := by
  have key : ∑ ℓ ∈ Finset.range (FlockDeploy.KS.depL x), 2 ^ FlockDeploy.KS.depLogLen x (ℓ + 1) ≤ 2795520 := by
    unfold FlockDeploy.KS.depL FlockDeploy.KS.depLogLen
    rcases hd : FlockDeploy.KS.depSetup x with _ | ⟨spec, st, sch⟩
    · simp
    · obtain ⟨-, -, -, -, hF⟩ := FlockDeploy.KS.depSetup_some hd
      have hA : FlockSoundness.Accounting.fast100 st.m = some (FlockSoundness.Refine.schedOf sch) := by
        have e := FlockSoundness.Refine.fast100_schedOf st.m
        rw [hF] at e
        exact e.symm
      have hlen : sch.levels.size = (FlockSoundness.Refine.schedOf sch).levels.length := by
        simp [FlockSoundness.Refine.schedOf]
      dsimp only
      rw [hlen]
      exact fast100_capSum_le hA
  have hw : FlockDeploy.KS.CpDep.width x =
      2 * ∑ ℓ ∈ Finset.range (FlockDeploy.KS.depL x), 2 ^ FlockDeploy.KS.depLogLen x (ℓ + 1) := by
    show Fintype.card (FlockDeploy.KS.DepIdx x) = _
    rw [Fintype.card_sigma]
    simp only [Fintype.card_sigma, Fintype.card_fin, Finset.sum_const, Finset.card_univ, smul_eq_mul]
    exact congrArg (2 * ·) (Fin.sum_univ_eq_sum_range (fun ℓ => 2 ^ FlockDeploy.KS.depLogLen x (ℓ + 1)) _)
  omega

/-! ## At the deployment -/

/-- **Route 2C's bound `λ₂` evaluates to at least `135.7` at the deployed caps**: `lamLin2_deployed_ge_dep`'s
statement, at the same hypotheses, with `135.7` for `100` (`lamLin2_core_tight`). -/
theorem lamLin2_deployed_ge_dep_tight (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (b : Brings) (L : Claims.Recursion.VStarB.LayoutB) (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (decI : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace)
    {x : Msg} {s : FlockDeploy.Ins} {spec : _root_.Flock.SessionSpec} {sts : Array _root_.Flock.Stmt}
    (hin : FlockDeploy.PlainIn (xIOf x) s spec sts)
    (hO : ∀ j < nS x, ∃ i spec' sts', PartIn (staged b.st x (pick b.st x j) j) i spec' sts')
    (hn : nS x ≤ 2 ^ 16) (hS : (innerSetupIns (xIOf x)).length ≤ 2 ^ 56) {n : ℕ} (hU : n ≤ 2 ^ 55) {v : ℕ}
    (hV : v ≤ 2 ^ 55) :
    135.7 ≤ Claims.Recursion.lamLin2
      ((recBoundAt rel b L).record (bdDeployed rel nextE colsE wE join rowIn final w).IVStar)
      (Claims.Recursion.Inner.ofFlock (bdDeployed rel nextE colsE wE join rowIn final w).deploy decI)
      FlockDeploy.KS.CpDep x n v := by
  have hsz : (sts.size : ℝ) ≤ 200 := by exact_mod_cast hin.size_le
  have hVI : (FlockDeploy.deployPlain rel).V.k (xIOf x) ≤ 880002 :=
    (FlockDeploy.Vk_plain_le rel hin).trans <|
      calc 2 + 4400 * sts.size ≤ 2 + 4400 * 200 := by gcongr; exact hin.size_le
        _ = 880002 := by norm_num
  have hkI : (FlockDeploy.deployPlain rel).k (xIOf x) ≤ 2468800000 := by
    obtain ⟨-, hhi⟩ := FlockDeploy.N₀_le hin.setup hin.size_pos fun st hst =>
      have := hin.tab st hst
      FlockDeploy.tableLen_plain_bounds st this.1 this.2.1
    exact k_le_200 (N := FlockDeploy.N₀ false (xIOf x)) (hhi.trans (by gcongr))
  have hsum : ((List.range (nS x)).map fun j => (nums b.st x j).rounds).sum ≤ nS x * 880002 := by
    refine (List.sum_le_length_nsmul _ 880002 fun a ha => ?_).trans (le_of_eq (by simp))
    obtain ⟨j, hj, rfl⟩ := List.mem_map.1 ha
    obtain ⟨i, spec', sts', hp⟩ := hO j (List.mem_range.1 hj)
    exact numsOf_rounds_le hp
  have hkx : (FlockDeploy.deployPlain rel).V.k (xIOf x) + 1 +
      ((List.range (nS x)).map fun j => (nums b.st x j).rounds).sum ≤ 2 ^ 37 :=
    (Nat.add_le_add (Nat.add_le_add_right hVI 1) (hsum.trans (Nat.mul_le_mul_right _ hn))).trans (by norm_num)
  have hκI0 : 0 < Claims.Flock.κ (FlockDeploy.deployPlain rel) (xIOf x) :=
    Claims.Flock.κ_pos _ _ (FlockDeploy.εtable_nonneg false _)
  have hκI : Claims.Flock.κ (FlockDeploy.deployPlain rel) (xIOf x) ≤ 200 * 2 ^ (-(202.98 : ℝ)) :=
    (FlockDeploy.κ_plain_le rel hin).trans (mul_le_mul_of_nonneg_right hsz (by positivity))
  refine lamLin2_ge_of_tight ((recBoundAt rel b L).record (bdDeployed rel nextE colsE wE join rowIn final w).IVStar)
    (Claims.Recursion.Inner.ofFlock (bdDeployed rel nextE colsE wE join rowIn final w).deploy decI)
    FlockDeploy.KS.CpDep x n v hVI hkI rfl (CpDep_width_le_tight (xIOf x)) hn hS hU hV hkx hκI0 hκI
    (fun j => numsκ_nonneg _ (εtableO_nonneg (staged b.st x (pick b.st x j) j))) (fun j => ?_) (fun j => ?_)
    (fun _ => rfl) (fun j => ?_)
  · obtain ⟨i, spec', sts', hp⟩ := hO j j.isLt
    have hsz' : (sts'.size : ℝ) ≤ 200 := by exact_mod_cast hp.size_le
    exact (numsOf_κ_le hp).trans (mul_le_mul_of_nonneg_right hsz' (by positivity))
  · obtain ⟨i, spec', sts', hp⟩ := hO j j.isLt
    exact numsOf_γ_le hp
  · obtain ⟨i, spec', sts', hp⟩ := hO j j.isLt
    obtain ⟨-, hhi⟩ := N₀O_le hp
    have hsz' : (sts'.size : ℝ) ≤ 200 := by exact_mod_cast hp.size_le
    exact k_le_200 (N := N₀O (staged b.st x (pick b.st x j) j)) (hhi.trans (by gcongr))

/-- **Route 2C's `λ₂ ≥ 135.7` at the deployed brings**: `lamLin2_deployedB_ge`'s statement, at the same hypotheses and
the call bound `v = 2^55`, with `135.7` for `100`. -/
theorem lamLin2_deployedB_ge_tight (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings)
    (L : Claims.Recursion.VStarB.LayoutB) (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (decI : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace)
    {x : Msg} {s : FlockDeploy.Ins} {spec : _root_.Flock.SessionSpec} {sts : Array _root_.Flock.Stmt}
    (hin : FlockDeploy.PlainIn (xIOf x) s spec sts)
    (hO : ∀ j < nS x, ∃ i spec' sts',
      PartIn (staged (deployedB b).st x (pick (deployedB b).st x j) j) i spec' sts')
    (hn : nS x ≤ 2 ^ 16) (hS : (innerSetupIns (xIOf x)).length ≤ 2 ^ 56) {n : ℕ} (hU : n ≤ 2 ^ 55) :
    135.7 ≤ Claims.Recursion.lamLin2
      ((recBoundAt rel (deployedB b) L).record (bdDeployed rel nextE colsE wE join rowIn final w).IVStar)
      (Claims.Recursion.Inner.ofFlock (bdDeployed rel nextE colsE wE join rowIn final w).deploy decI)
      FlockDeploy.KS.CpDep x n (2 ^ 55) :=
  lamLin2_deployed_ge_dep_tight rel (deployedB b) L nextE colsE wE join rowIn final w decI hin hO hn hS hU le_rfl

end RecDeploy
