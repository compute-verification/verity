module

public import Proofs.Flock.Claims.Recursion.KappaEpsDeployed

/-!
# Route 2C's split bound at the deployed caps: `κ ≤ 2^{−140.4}` and `ε ≤ 2^{−138.3}`

`bitsLin2KOn_of_ksLin2On` bounds `Pr[acc ∧ ¬R]` by `κ + t·ε` (`Claims.Recursion.epsLin2`). At
`lamLin2_deployedB_ge_tight`'s caps (at most `200` inner tables, `2^16` V\* parts of at most `200` tables each,
`|S(x_I)| ≤ 2^56` and `n, v ≤ 2^55`) `κ ≤ 2^{−140.4}` and `ε ≤ 2^{−138.3}` (`kappa_eps_deployed_dep_tight`), over the
numbers `lamLin2_core_tight` bounds:

* `κ ≤ κ_I + N_I·Σ_s κ_s ≤ 200·(1 + 2^16·N_I)·2^{−202.98}`, at `N_I ≤ 518448000001`, and
  `200·(1 + 2^16·518448000001) ≤ 2^{62.58}`;
* each of `ε`'s five terms `2^{−(a − log₂ y)}` is at most `u·2^{−255}` at `a ≥ 255` and `y ≤ u`
  (`two_rpow_neg_sub_le`; `γ`'s at `y ≤ u²`), at `c_S ≤ 284126798369783812`, and the five `u` sum to at most
  `2^{116.7}`. Fdev's, `β₂·c_S`, is all but about `2^{−8.8}` of the sum.

`λ₂ ≥ 135.7` is the least of the six terms' bits less `log₂ 6`; summed term by term, `ε` is about `2^{−138.33}`.
-/

@[expose] public section

namespace RecDeploy

open Claims (Msg Out)
open FlockDeploy (Coin)

/-- **`κ ≤ 2^{−140.4}` and `ε ≤ 2^{−138.3}` from their numbers**: `κ` and `epsLin2`'s five terms, over the numbers they
read as in `lamLin2_core_tight` and at its bounds on them. -/
theorem kappa_eps_core_tight {nS : ℕ} {VI kI wI W kx NI cBe nSet nU : ℕ} {κI : ℝ} {κs γs : Fin nS → ℝ}
    {ws ks : Fin nS → ℕ} (hNI : NI = 1 + wI * kI) (hcBe : cBe = 4 + 2 * ∑ s, ws s * ks s)
    (hSet : nSet + 2 * nU ≤ 3 * 2 ^ 56) (hVI : VI ≤ 880002) (hkI : kI ≤ 2468800000) (hwI : wI = 210)
    (hW : W ≤ 5591040) (hnS : nS ≤ 2 ^ 16) (hkx : kx ≤ 2 ^ 37) (hκI : κI ≤ 200 * 2 ^ (-(202.98 : ℝ)))
    (hκs : ∀ s, κs s ≤ 200 * 2 ^ (-(202.98 : ℝ))) (hγs : ∀ s, γs s ≤ 2 ^ 68) (hws : ∀ s, ws s = 210)
    (hks : ∀ s, ks s ≤ 2468800000) :
    κI + NI * ∑ s, κs s ≤ 2 ^ (-(140.4 : ℝ)) ∧
      (2 : ℝ) ^ (-(255.5 - Real.logb 2 ((NI * ∑ s, Real.sqrt (γs s * kx)) ^ 2) / 2)) +
        (2 : ℝ) ^ (-(256 - Real.logb 2 ((4 * VI + 1) * ((cBe : ℝ) + nSet + 2 * nU)))) +
        (2 : ℝ) ^ (-(255 - Real.logb 2 (4 * W * ((cBe : ℝ) + nSet + 2 * nU)))) +
        (2 : ℝ) ^ (-(256 - Real.logb 2 (4 * ((kI : ℝ) + 1) * ((cBe : ℝ) + nSet + 2 * nU)))) +
        (2 : ℝ) ^ (-(255 - Real.logb 2 ((NI + 8 * W + 4 * ((kI : ℝ) + 1)) * VI * ((cBe : ℝ) + nSet + 2 * nU)))) ≤
        2 ^ (-(138.3 : ℝ)) := by
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
  have hcS' : cBe + nSet + 2 * nU ≤ 284126798369783812 := hcS.trans (by norm_num)
  have hNIR : (NI : ℝ) ≤ 518448000001 := by exact_mod_cast hNI'
  have hcSR : (cBe : ℝ) + nSet + 2 * nU ≤ 284126798369783812 := by exact_mod_cast hcS'
  have hc0 : (0 : ℝ) ≤ (cBe : ℝ) + nSet + 2 * nU := by positivity
  have hVIR : (VI : ℝ) ≤ 880002 := by exact_mod_cast hVI
  have hkIR : (kI : ℝ) ≤ 2468800000 := by exact_mod_cast hkI
  have hWR : (W : ℝ) ≤ 5591040 := by exact_mod_cast hW
  have hnSR : (nS : ℝ) ≤ 2 ^ 16 := by exact_mod_cast hnS
  have hkxR : (kx : ℝ) ≤ 2 ^ 37 := by exact_mod_cast hkx
  constructor
  · have hS : ∑ s, κs s ≤ 2 ^ 16 * (200 * 2 ^ (-(202.98 : ℝ))) := by
      refine (sum_fin_le_real _ hκs).trans ?_
      gcongr
    have h1 : (NI : ℝ) * ∑ s, κs s ≤ 518448000001 * (2 ^ 16 * (200 * 2 ^ (-(202.98 : ℝ)))) :=
      (mul_le_mul_of_nonneg_left hS (Nat.cast_nonneg _)).trans
        (mul_le_mul_of_nonneg_right hNIR (by positivity))
    have hp0 : (0 : ℝ) ≤ 2 ^ (-(202.98 : ℝ)) := by positivity
    have e : (2 : ℝ) ^ (((3129 : ℕ) : ℝ) / ((50 : ℕ) : ℝ)) * 2 ^ (-(202.98 : ℝ)) = 2 ^ (-(140.4 : ℝ)) := by
      rw [← Real.rpow_add (by norm_num)]
      congr 1
      norm_num
    have hK : ((6795401625613107400 : ℕ) : ℝ) ≤ 2 ^ (((3129 : ℕ) : ℝ) / ((50 : ℕ) : ℝ)) :=
      le_two_rpow_of_pow_le (Nat.cast_nonneg _) (by norm_num) (cast_pow_le_two_pow (by decide +kernel))
    calc κI + NI * ∑ s, κs s
        ≤ 200 * 2 ^ (-(202.98 : ℝ)) + 518448000001 * (2 ^ 16 * (200 * 2 ^ (-(202.98 : ℝ)))) := add_le_add hκI h1
      _ = ((6795401625613107400 : ℕ) : ℝ) * 2 ^ (-(202.98 : ℝ)) := by push_cast; ring
      _ ≤ 2 ^ (((3129 : ℕ) : ℝ) / ((50 : ℕ) : ℝ)) * 2 ^ (-(202.98 : ℝ)) := mul_le_mul_of_nonneg_right hK hp0
      _ = 2 ^ (-(140.4 : ℝ)) := e
  · have hsq : ∀ s, Real.sqrt (γs s * kx) ≤ 2 ^ 53 := fun s => by
      rw [Real.sqrt_le_left (by norm_num)]
      calc γs s * kx ≤ 2 ^ 68 * kx := mul_le_mul_of_nonneg_right (hγs s) (Nat.cast_nonneg _)
        _ ≤ 2 ^ 68 * 2 ^ 37 := by gcongr
        _ ≤ (2 ^ 53) ^ 2 := by norm_num
    have hT0 : 0 ≤ ∑ s, Real.sqrt (γs s * kx) := Finset.sum_nonneg fun s _ => Real.sqrt_nonneg _
    have hT : ∑ s, Real.sqrt (γs s * kx) ≤ 2 ^ 16 * 2 ^ 53 := by
      refine (sum_fin_le_real _ hsq).trans ?_
      gcongr
    have hY : (NI : ℝ) * ∑ s, Real.sqrt (γs s * kx) ≤ 518448000001 * (2 ^ 16 * 2 ^ 53) :=
      mul_le_mul hNIR hT hT0 (by norm_num)
    have hY0 : (0 : ℝ) ≤ (NI : ℝ) * ∑ s, Real.sqrt (γs s * kx) := mul_nonneg (Nat.cast_nonneg _) hT0
    have hp2 : (0 : ℝ) ≤ 2 ^ (-(255 : ℝ)) := by positivity
    have e : (2 : ℝ) ^ (((1167 : ℕ) : ℝ) / ((10 : ℕ) : ℝ)) * 2 ^ (-(255 : ℝ)) = 2 ^ (-(138.3 : ℝ)) := by
      rw [← Real.rpow_add (by norm_num)]
      congr 1
      norm_num
    refine le_trans (add_le_add (add_le_add (add_le_add (add_le_add
      (two_rpow_neg_sub_half_le (u := 518448000001 * (2 ^ 16 * 2 ^ 53)) (by positivity)
        (pow_le_pow_left₀ hY0 hY 2) (by norm_num))
      (two_rpow_neg_sub_le (u := (4 * 880002 + 1) * 284126798369783812) (by norm_num) (by positivity)
        (mul_le_mul (by linarith) hcSR hc0 (by norm_num)) (by norm_num)))
      (two_rpow_neg_sub_le (u := 4 * 5591040 * 284126798369783812) (by norm_num) (by positivity)
        (mul_le_mul (by linarith) hcSR hc0 (by norm_num)) (by norm_num)))
      (two_rpow_neg_sub_le (u := 4 * (2468800000 + 1) * 284126798369783812) (by norm_num) (by positivity)
        (mul_le_mul (by linarith) hcSR hc0 (by norm_num)) (by norm_num)))
      (two_rpow_neg_sub_le
        (u := (518448000001 + 8 * 5591040 + 4 * (2468800000 + 1)) * 880002 * 284126798369783812)
        (by norm_num) (by positivity)
        (mul_le_mul (mul_le_mul (by linarith) hVIR (Nat.cast_nonneg _) (by norm_num)) hcSR hc0 (by norm_num))
        (by norm_num))) ?_
    -- `2^1167` is past `exponentiation.threshold`, so the comparison is the kernel's, on `ℕ`.
    have hK : ((132415010038335200324691815141008988 : ℕ) : ℝ) ≤ 2 ^ (((1167 : ℕ) : ℝ) / ((10 : ℕ) : ℝ)) :=
      le_two_rpow_of_pow_le (Nat.cast_nonneg _) (by norm_num) (cast_pow_le_two_pow (by decide +kernel))
    calc _ = ((132415010038335200324691815141008988 : ℕ) : ℝ) * 2 ^ (-(255 : ℝ)) := by push_cast; ring
      _ ≤ 2 ^ (((1167 : ℕ) : ℝ) / ((10 : ℕ) : ℝ)) * 2 ^ (-(255 : ℝ)) := mul_le_mul_of_nonneg_right hK hp2
      _ = 2 ^ (-(138.3 : ℝ)) := e

/-- **`κ ≤ 2^{−140.4}` and `ε ≤ 2^{−138.3}` at a compiled record and an inner protocol**, from bounds on the numbers
they read: `lamLin2_ge_of_tight`'s. -/
theorem kappa_eps_of_tight {C W' : Type} [Fintype C] [Nonempty C] [Inhabited C] (D : Claims.Recursion.Deploy C)
    (I : Claims.Recursion.Inner C W') (Cp : Claims.Recursion.Caps C) (x : Msg) (n v : ℕ)
    (hVI : I.V.k (Claims.Recursion.xI D x) ≤ 880002) (hkI : I.k (Claims.Recursion.xI D x) ≤ 2468800000)
    (hwI : I.waits (Claims.Recursion.xI D x) = 210) (hW : Cp.width (Claims.Recursion.xI D x) ≤ 5591040)
    (hnS : D.nS x ≤ 2 ^ 16) (hSet : Claims.Recursion.setupLen D x ≤ 2 ^ 56) (hU : n ≤ 2 ^ 55) (hV : v ≤ 2 ^ 55)
    (hkx : D.V.k x ≤ 2 ^ 37) (hκI : I.κ (Claims.Recursion.xI D x) ≤ 200 * 2 ^ (-(202.98 : ℝ)))
    (hκs : ∀ s : Fin (D.nS x), (D.nums x s).κ ≤ 200 * 2 ^ (-(202.98 : ℝ)))
    (hγs : ∀ s : Fin (D.nS x), (D.nums x s).γ ≤ 2 ^ 68) (hws : ∀ s : Fin (D.nS x), (D.nums x s).waits = 210)
    (hks : ∀ s : Fin (D.nS x), (D.nums x s).k ≤ 2468800000) :
    Claims.Recursion.κ D I x ≤ 2 ^ (-(140.4 : ℝ)) ∧
      Claims.Recursion.epsLin2 D I Cp x n v ≤ 2 ^ (-(138.3 : ℝ)) :=
  kappa_eps_core_tight (nS := D.nS x) (VI := I.V.k (Claims.Recursion.xI D x))
    (kI := I.k (Claims.Recursion.xI D x)) (wI := I.waits (Claims.Recursion.xI D x))
    (W := Cp.width (Claims.Recursion.xI D x)) (kx := D.V.k x) (NI := Claims.Recursion.NI D I x)
    (cBe := Claims.Recursion.cBe D x) (nSet := Claims.Recursion.setupLen D x) (nU := n + v)
    (κI := I.κ (Claims.Recursion.xI D x)) (κs := fun s => (D.nums x s).κ) (γs := fun s => (D.nums x s).γ)
    (ws := fun s => (D.nums x s).waits) (ks := fun s => (D.nums x s).k) rfl rfl (by omega) hVI hkI hwI hW hnS hkx
    hκI hκs hγs hws hks

/-- **Route 2C's split bound at the deployed caps: `κ ≤ 2^{−140.4}` and `ε ≤ 2^{−138.3}`**, at
`lamLin2_deployed_ge_dep_tight`'s hypotheses: the inner statement's plain setup in `PlainIn` (`hin`), each staged V\*
part's in `PartIn` (`hO`), at most `2^16` parts (`hn`), at most `2^56` inner setup inputs (`hS`) and call bounds
`n, v ≤ 2^55` (`hU`, `hV`). With `bitsLin2KOn_of_ksLin2On` it makes `Pr[acc ∧ ¬R] ≤ 2^{−140.4} + t·2^{−138.3}`. -/
theorem kappa_eps_deployed_dep_tight (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (b : Brings) (L : Claims.Recursion.VStarB.LayoutB) (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (decI : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace)
    {x : Msg} {s : FlockDeploy.Ins} {spec : _root_.Flock.SessionSpec} {sts : Array _root_.Flock.Stmt}
    (hin : FlockDeploy.PlainIn (xIOf x) s spec sts)
    (hO : ∀ j < nS x, ∃ i spec' sts', PartIn (staged b.st x (pick b.st x j) j) i spec' sts')
    (hn : nS x ≤ 2 ^ 16) (hS : (innerSetupIns (xIOf x)).length ≤ 2 ^ 56) {n : ℕ} (hU : n ≤ 2 ^ 55) {v : ℕ}
    (hV : v ≤ 2 ^ 55) :
    Claims.Recursion.κ ((recBoundAt rel b L).record (bdDeployed rel nextE colsE wE join rowIn final w).IVStar)
        (Claims.Recursion.Inner.ofFlock (bdDeployed rel nextE colsE wE join rowIn final w).deploy decI) x ≤
        2 ^ (-(140.4 : ℝ)) ∧
      Claims.Recursion.epsLin2
        ((recBoundAt rel b L).record (bdDeployed rel nextE colsE wE join rowIn final w).IVStar)
        (Claims.Recursion.Inner.ofFlock (bdDeployed rel nextE colsE wE join rowIn final w).deploy decI)
        FlockDeploy.KS.CpDep x n v ≤ 2 ^ (-(138.3 : ℝ)) := by
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
  have hκI : Claims.Flock.κ (FlockDeploy.deployPlain rel) (xIOf x) ≤ 200 * 2 ^ (-(202.98 : ℝ)) :=
    (FlockDeploy.κ_plain_le rel hin).trans (mul_le_mul_of_nonneg_right hsz (by positivity))
  refine kappa_eps_of_tight ((recBoundAt rel b L).record (bdDeployed rel nextE colsE wE join rowIn final w).IVStar)
    (Claims.Recursion.Inner.ofFlock (bdDeployed rel nextE colsE wE join rowIn final w).deploy decI)
    FlockDeploy.KS.CpDep x n v hVI hkI rfl (CpDep_width_le_tight (xIOf x)) hn hS hU hV hkx hκI (fun j => ?_)
    (fun j => ?_) (fun _ => rfl) (fun j => ?_)
  · obtain ⟨i, spec', sts', hp⟩ := hO j j.isLt
    have hsz' : (sts'.size : ℝ) ≤ 200 := by exact_mod_cast hp.size_le
    exact (numsOf_κ_le hp).trans (mul_le_mul_of_nonneg_right hsz' (by positivity))
  · obtain ⟨i, spec', sts', hp⟩ := hO j j.isLt
    exact numsOf_γ_le hp
  · obtain ⟨i, spec', sts', hp⟩ := hO j j.isLt
    obtain ⟨-, hhi⟩ := N₀O_le hp
    have hsz' : (sts'.size : ℝ) ≤ 200 := by exact_mod_cast hp.size_le
    exact k_le_200 (N := N₀O (staged b.st x (pick b.st x j) j)) (hhi.trans (by gcongr))

end RecDeploy
