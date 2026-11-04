import Proofs.Flock.Claims.Recursion.KappaEpsDeployed

/-!
# Route 2C's split bound at the audit's call caps: `κ ≤ 2^{−167.7}` and `ε ≤ 2^{−166.95}`

`kappa_eps_deployed` evaluates `κ + t·ε` where the inner setup `|S(x_I)|` and the call bounds `n` and `v` are each at
most `2^32`. The audit reads it at the bounds every accepted statement meets: the inner setup's calls at most
`innerB ≤ 2^43` (`innerSetupCalls_le_acc`), the inner verifier's `2^28 + (innerB + 2^24) ≤ 2^43`
(`innerCalls_R_le_acc'`), and V\*'s setup calls `nS x · 17716871168` (`setupCalls_deployedB_le_cap`), at most
`16 · 17716871168 ≈ 2^{38.04}` at the verifier's cap of `16` parts (`RecBudget.MAX_STATEMENTS`, `nS_le_sixteen`). At the
deployment's tables (`OneTableAt`: `SizesAt` but for its inner-setup bound) and at `|S(x_I)|, n ≤ 2^43` and
`v ≤ nS x · 17716871168`, `κ ≤ 2^{−167.7}` and `ε ≤ 2^{−166.95}` (`kappa_eps_oneTable_cap`).

`κ` reads neither the setup nor `n` nor `v`, so it is `kappa_eps_core`'s. `ε`'s five terms read
`c_S = c_Be + |S(x_I)| + 2(n + v)`, now at most `27038170529924 ≈ 2^{44.62}`, so each `u` but `γ`'s grows by `c_S`'s
ratio, about `2^{8.02}`, and the five sum to at most `2^{88.05}`; `Fdev`'s, `β₂·c_S`, is all but about `2^{−17}` of the
sum.
-/

namespace RecDeploy

open Claims (Msg Out)
open FlockDeploy (Coin)

/-- **`κ ≤ 2^{−167.7}` and `ε ≤ 2^{−166.95}` from their numbers**: `kappa_eps_core`'s statement at
`|S(x_I)| + 2(n + v) ≤ 2^43 + 2(2^43 + 16 · 17716871168)`. -/
theorem kappa_eps_core_cap {nS : ℕ} {VI kI wI W kx NI cBe nSet nU : ℕ} {κI : ℝ} {κs γs : Fin nS → ℝ}
    {ws ks : Fin nS → ℕ} (hNI : NI = 1 + wI * kI) (hcBe : cBe = 4 + 2 * ∑ s, ws s * ks s)
    (hSet : nSet + 2 * nU ≤ 2 ^ 43 + 2 * (2 ^ 43 + 16 * 17716871168)) (hVI : VI ≤ 4402) (hkI : kI ≤ 12343986)
    (hwI : wI = 210) (hW : W ≤ 5591040) (hnS : nS ≤ 16) (hkx : kx ≤ 74835) (hκI : κI ≤ 2 ^ (-(202.98 : ℝ)))
    (hκs : ∀ s, κs s ≤ 2 ^ (-(202.98 : ℝ))) (hγs : ∀ s, γs s ≤ 2 ^ 51) (hws : ∀ s, ws s = 210)
    (hks : ∀ s, ks s ≤ 12343986) :
    κI + NI * ∑ s, κs s ≤ 2 ^ (-(167.7 : ℝ)) ∧
      (2 : ℝ) ^ (-(255.5 - Real.logb 2 ((NI * ∑ s, Real.sqrt (γs s * kx)) ^ 2) / 2)) +
        (2 : ℝ) ^ (-(256 - Real.logb 2 ((4 * VI + 1) * ((cBe : ℝ) + nSet + 2 * nU)))) +
        (2 : ℝ) ^ (-(255 - Real.logb 2 (4 * W * ((cBe : ℝ) + nSet + 2 * nU)))) +
        (2 : ℝ) ^ (-(256 - Real.logb 2 (4 * ((kI : ℝ) + 1) * ((cBe : ℝ) + nSet + 2 * nU)))) +
        (2 : ℝ) ^ (-(255 - Real.logb 2 ((NI + 8 * W + 4 * ((kI : ℝ) + 1)) * VI * ((cBe : ℝ) + nSet + 2 * nU)))) ≤
        2 ^ (-(166.95 : ℝ)) := by
  refine ⟨(kappa_eps_core (nSet := 0) (nU := 0) hNI hcBe (by norm_num) hVI hkI hwI hW hnS hkx hκI hκs hγs hws
    hks).1, ?_⟩
  have hNI' : NI ≤ 2592237061 := by
    rw [hNI, hwI]
    calc 1 + 210 * kI ≤ 1 + 210 * 12343986 := by gcongr
      _ = 2592237061 := by norm_num
  have hcBe' : cBe ≤ 4 + 2 * (16 * (210 * 12343986)) := by
    have hsum : ∑ s, ws s * ks s ≤ nS * (210 * 12343986) :=
      sum_fin_le_nat _ fun s => by rw [hws s]; exact Nat.mul_le_mul_left 210 (hks s)
    rw [hcBe]
    gcongr
    exact hsum.trans (Nat.mul_le_mul_right _ hnS)
  have hcS : cBe + nSet + 2 * nU ≤ 27038170529924 := by omega
  have hNIR : (NI : ℝ) ≤ 2592237061 := by exact_mod_cast hNI'
  have hcSR : (cBe : ℝ) + nSet + 2 * nU ≤ 27038170529924 := by exact_mod_cast hcS
  have hc0 : (0 : ℝ) ≤ (cBe : ℝ) + nSet + 2 * nU := by positivity
  have hVIR : (VI : ℝ) ≤ 4402 := by exact_mod_cast hVI
  have hkIR : (kI : ℝ) ≤ 12343986 := by exact_mod_cast hkI
  have hWR : (W : ℝ) ≤ 5591040 := by exact_mod_cast hW
  have hkxR : (kx : ℝ) ≤ 74835 := by exact_mod_cast hkx
  have hsq : ∀ s, Real.sqrt (γs s * kx) ≤ 13000000000 := fun s => by
    rw [Real.sqrt_le_left (by norm_num)]
    calc γs s * kx ≤ 2 ^ 51 * kx := mul_le_mul_of_nonneg_right (hγs s) (Nat.cast_nonneg _)
      _ ≤ 2 ^ 51 * 74835 := by gcongr
      _ ≤ 13000000000 ^ 2 := by norm_num
  have hT0 : 0 ≤ ∑ s, Real.sqrt (γs s * kx) := Finset.sum_nonneg fun s _ => Real.sqrt_nonneg _
  have hT : ∑ s, Real.sqrt (γs s * kx) ≤ 16 * 13000000000 := by
    refine (sum_fin_le_real _ hsq).trans ?_
    have hnSR : (nS : ℝ) ≤ 16 := by exact_mod_cast hnS
    gcongr
  have hY : (NI : ℝ) * ∑ s, Real.sqrt (γs s * kx) ≤ 2592237061 * (16 * 13000000000) :=
    mul_le_mul hNIR hT hT0 (by norm_num)
  have hY0 : (0 : ℝ) ≤ (NI : ℝ) * ∑ s, Real.sqrt (γs s * kx) := mul_nonneg (Nat.cast_nonneg _) hT0
  have hp2 : (0 : ℝ) ≤ 2 ^ (-(255 : ℝ)) := by positivity
  have e : (2 : ℝ) ^ (((1761 : ℕ) : ℝ) / ((20 : ℕ) : ℝ)) * 2 ^ (-(255 : ℝ)) = 2 ^ (-(166.95 : ℝ)) := by
    rw [← Real.rpow_add (by norm_num)]
    congr 1
    norm_num
  refine le_trans (add_le_add (add_le_add (add_le_add (add_le_add
    (two_rpow_neg_sub_half_le (u := 2592237061 * (16 * 13000000000)) (by positivity)
      (pow_le_pow_left₀ hY0 hY 2) (by norm_num))
    (two_rpow_neg_sub_le (u := (4 * 4402 + 1) * 27038170529924) (by norm_num) (by positivity)
      (mul_le_mul (by linarith) hcSR hc0 (by norm_num)) (by norm_num)))
    (two_rpow_neg_sub_le (u := 4 * 5591040 * 27038170529924) (by norm_num) (by positivity)
      (mul_le_mul (by linarith) hcSR hc0 (by norm_num)) (by norm_num)))
    (two_rpow_neg_sub_le (u := 4 * (12343986 + 1) * 27038170529924) (by norm_num) (by positivity)
      (mul_le_mul (by linarith) hcSR hc0 (by norm_num)) (by norm_num)))
    (two_rpow_neg_sub_le (u := (2592237061 + 8 * 5591040 + 4 * (12343986 + 1)) * 4402 * 27038170529924)
      (by norm_num) (by positivity)
      (mul_le_mul (mul_le_mul (by linarith) hVIR (Nat.cast_nonneg _) (by norm_num)) hcSR hc0 (by norm_num))
      (by norm_num))) ?_
  -- `2^1761` is past `exponentiation.threshold`, so the comparison is the kernel's, on `ℕ`.
  have hK : ((319736268694980500058863900 : ℕ) : ℝ) ≤ 2 ^ (((1761 : ℕ) : ℝ) / ((20 : ℕ) : ℝ)) :=
    le_two_rpow_of_pow_le (Nat.cast_nonneg _) (by norm_num) (cast_pow_le_two_pow (by decide +kernel))
  calc _ = ((319736268694980500058863900 : ℕ) : ℝ) * 2 ^ (-(255 : ℝ)) := by push_cast; ring
    _ ≤ 2 ^ (((1761 : ℕ) : ℝ) / ((20 : ℕ) : ℝ)) * 2 ^ (-(255 : ℝ)) := mul_le_mul_of_nonneg_right hK hp2
    _ = 2 ^ (-(166.95 : ℝ)) := e

/-- **`κ ≤ 2^{−167.7}` and `ε ≤ 2^{−166.95}` at a compiled record and an inner protocol**, from bounds on the numbers
they read: `kappa_eps_of`'s statement at the inner setup `|S(x_I)| ≤ 2^43` and the call bounds `n ≤ 2^43` and
`v ≤ 16 · 17716871168`. -/
theorem kappa_eps_of_cap {C W' : Type} [Fintype C] [Nonempty C] [Inhabited C] (D : Claims.Recursion.Deploy C)
    (I : Claims.Recursion.Inner C W') (Cp : Claims.Recursion.Caps C) (x : Msg) (n v : ℕ)
    (hVI : I.V.k (Claims.Recursion.xI D x) ≤ 4402) (hkI : I.k (Claims.Recursion.xI D x) ≤ 12343986)
    (hwI : I.waits (Claims.Recursion.xI D x) = 210) (hW : Cp.width (Claims.Recursion.xI D x) ≤ 5591040)
    (hnS : D.nS x ≤ 16) (hSet : Claims.Recursion.setupLen D x ≤ 2 ^ 43) (hU : n ≤ 2 ^ 43)
    (hV : v ≤ 16 * 17716871168) (hkx : D.V.k x ≤ 74835) (hκI : I.κ (Claims.Recursion.xI D x) ≤ 2 ^ (-(202.98 : ℝ)))
    (hκs : ∀ s : Fin (D.nS x), (D.nums x s).κ ≤ 2 ^ (-(202.98 : ℝ)))
    (hγs : ∀ s : Fin (D.nS x), (D.nums x s).γ ≤ 2 ^ 51) (hws : ∀ s : Fin (D.nS x), (D.nums x s).waits = 210)
    (hks : ∀ s : Fin (D.nS x), (D.nums x s).k ≤ 12343986) :
    Claims.Recursion.κ D I x ≤ 2 ^ (-(167.7 : ℝ)) ∧
      Claims.Recursion.epsLin2 D I Cp x n v ≤ 2 ^ (-(166.95 : ℝ)) :=
  kappa_eps_core_cap (nS := D.nS x) (VI := I.V.k (Claims.Recursion.xI D x)) (kI := I.k (Claims.Recursion.xI D x))
    (wI := I.waits (Claims.Recursion.xI D x)) (W := Cp.width (Claims.Recursion.xI D x)) (kx := D.V.k x)
    (NI := Claims.Recursion.NI D I x) (cBe := Claims.Recursion.cBe D x) (nSet := Claims.Recursion.setupLen D x)
    (nU := n + v) (κI := I.κ (Claims.Recursion.xI D x)) (κs := fun s => (D.nums x s).κ)
    (γs := fun s => (D.nums x s).γ) (ws := fun s => (D.nums x s).waits) (ks := fun s => (D.nums x s).k) rfl rfl
    (by omega) hVI hkI hwI hW hnS hkx hκI hκs hγs hws hks

/-- **Route 2C's split bound at the audit's call caps: `κ ≤ 2^{−167.7}` and `ε ≤ 2^{−166.95}`**, at
`kappa_eps_deployed`'s records and tables (`hin`, `h1`, `hO`), at an inner setup of at most `2^43` calls (`hS`) and at
call bounds `n ≤ 2^43` and `v ≤ nS x · 17716871168` (`hU`, `hV`): what `innerSetupCalls_le_acc` proves of the inner
setup and `innerCalls_R_le_acc'` of the inner verifier's calls at every accepted statement, and what
`setupCalls_deployedB_le_cap` proves of V\*'s setup calls. The parts are at most `16`, the verifier's cap
(`nS_le_sixteen`, from `hO`), so `v ≤ 16 · 17716871168`. With `bitsLin2KOn_of_ksLin2On` it makes
`Pr[acc ∧ ¬R] ≤ 2^{−167.7} + t·2^{−166.95}`. -/
theorem kappa_eps_deployed_cap (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings)
    (L : Claims.Recursion.VStarB.LayoutB) (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (decI : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace)
    {x : Msg} {s : FlockDeploy.Ins} {spec : _root_.Flock.SessionSpec} {sts : Array _root_.Flock.Stmt}
    (hin : FlockDeploy.PlainIn (xIOf x) s spec sts) (h1 : sts.size = 1)
    (hO : ∀ j < nS x, ∃ i spec' sts', PartIn (staged b.st x (pick b.st x j) j) i spec' sts' ∧ sts'.size = 1)
    (hS : (innerSetupIns (xIOf x)).length ≤ 2 ^ 43) {n : ℕ} (hU : n ≤ 2 ^ 43) {v : ℕ}
    (hV : v ≤ nS x * 17716871168) :
    Claims.Recursion.κ ((recBoundAt rel b L).record (bdDeployed rel nextE colsE wE join rowIn final w).IVStar)
        (Claims.Recursion.Inner.ofFlock (bdDeployed rel nextE colsE wE join rowIn final w).deploy decI) x ≤
        2 ^ (-(167.7 : ℝ)) ∧
      Claims.Recursion.epsLin2
        ((recBoundAt rel b L).record (bdDeployed rel nextE colsE wE join rowIn final w).IVStar)
        (Claims.Recursion.Inner.ofFlock (bdDeployed rel nextE colsE wE join rowIn final w).deploy decI)
        FlockDeploy.KS.CpDep x n v ≤ 2 ^ (-(166.95 : ℝ)) := by
  have hn := nS_le_sixteen hO
  have hV16 : v ≤ 16 * 17716871168 := hV.trans (Nat.mul_le_mul_right _ hn)
  have hVI : (FlockDeploy.deployPlain rel).V.k (xIOf x) ≤ 4402 := by
    have := FlockDeploy.Vk_plain_le rel hin
    omega
  have hkI : (FlockDeploy.deployPlain rel).k (xIOf x) ≤ 12343986 := by
    obtain ⟨-, hhi⟩ := FlockDeploy.N₀_le hin.setup hin.size_pos fun st hst =>
      have := hin.tab st hst
      FlockDeploy.tableLen_plain_bounds st this.1 this.2.1
    rw [h1, Nat.cast_one, one_mul] at hhi
    exact k_le_one (N := FlockDeploy.N₀ false (xIOf x)) hhi
  have hsum : ((List.range (nS x)).map fun j => (nums b.st x j).rounds).sum ≤ nS x * 4402 := by
    refine (List.sum_le_length_nsmul _ 4402 fun a ha => ?_).trans (le_of_eq (by simp))
    obtain ⟨j, hj, rfl⟩ := List.mem_map.1 ha
    obtain ⟨i, spec', sts', hp, hp1⟩ := hO j (List.mem_range.1 hj)
    exact numsOf_rounds_le_one hp hp1
  have hkx : (FlockDeploy.deployPlain rel).V.k (xIOf x) + 1 +
      ((List.range (nS x)).map fun j => (nums b.st x j).rounds).sum ≤ 74835 := by
    have := Nat.mul_le_mul_right 4402 hn
    omega
  have hκI : Claims.Flock.κ (FlockDeploy.deployPlain rel) (xIOf x) ≤ 2 ^ (-(202.98 : ℝ)) := by
    have := FlockDeploy.κ_plain_le rel hin
    rwa [h1, Nat.cast_one, one_mul] at this
  refine kappa_eps_of_cap ((recBoundAt rel b L).record (bdDeployed rel nextE colsE wE join rowIn final w).IVStar)
    (Claims.Recursion.Inner.ofFlock (bdDeployed rel nextE colsE wE join rowIn final w).deploy decI)
    FlockDeploy.KS.CpDep x n v hVI hkI rfl (CpDep_width_le_tight (xIOf x)) hn hS hU hV16 hkx hκI (fun j => ?_)
    (fun j => ?_) (fun _ => rfl) (fun j => ?_)
  · obtain ⟨i, spec', sts', hp, hp1⟩ := hO j j.isLt
    have := numsOf_κ_le hp
    rwa [hp1, Nat.cast_one, one_mul] at this
  · obtain ⟨i, spec', sts', hp, hp1⟩ := hO j j.isLt
    exact numsOf_γ_le_one hp hp1
  · obtain ⟨i, spec', sts', hp, hp1⟩ := hO j j.isLt
    obtain ⟨-, hhi⟩ := N₀O_le hp
    rw [hp1, Nat.cast_one, one_mul] at hhi
    exact k_le_one (N := N₀O (staged b.st x (pick b.st x j) j)) hhi

/-- **The deployment's tables at a statement `x`**, `SizesAt` but for its inner-setup bound (which every accepted
statement meets at `2^43`, `innerSetupCalls_le_acc`): its inner statement is one plain table, and each V\* part it
stages is one `--zk` table. The parts are at most `16`, the verifier's cap (`nS_le_sixteen`). -/
def OneTableAt (b : Brings) (x : Msg) : Prop :=
  (∃ s spec sts, FlockDeploy.PlainIn (xIOf x) s spec sts ∧ sts.size = 1) ∧
    ∀ j < nS x, ∃ i spec' sts', PartIn (staged b.st x (pick b.st x j) j) i spec' sts' ∧ sts'.size = 1

theorem SizesAt.oneTableAt {b : Brings} {x : Msg} (h : SizesAt b x) : OneTableAt b x :=
  ⟨h.1, h.2.1⟩

/-- **`kappa_eps_deployed_cap` at a statement at the deployment's tables** (`OneTableAt b x`). -/
theorem kappa_eps_oneTable_cap (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings)
    (L : Claims.Recursion.VStarB.LayoutB) (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (decI : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace) {x : Msg} (h : OneTableAt b x)
    (hS : (innerSetupIns (xIOf x)).length ≤ 2 ^ 43) {n : ℕ} (hU : n ≤ 2 ^ 43) {v : ℕ} (hV : v ≤ nS x * 17716871168) :
    Claims.Recursion.κ ((recBoundAt rel b L).record (bdDeployed rel nextE colsE wE join rowIn final w).IVStar)
        (Claims.Recursion.Inner.ofFlock (bdDeployed rel nextE colsE wE join rowIn final w).deploy decI) x ≤
        2 ^ (-(167.7 : ℝ)) ∧
      Claims.Recursion.epsLin2
        ((recBoundAt rel b L).record (bdDeployed rel nextE colsE wE join rowIn final w).IVStar)
        (Claims.Recursion.Inner.ofFlock (bdDeployed rel nextE colsE wE join rowIn final w).deploy decI)
        FlockDeploy.KS.CpDep x n v ≤ 2 ^ (-(166.95 : ℝ)) := by
  obtain ⟨⟨s, spec, sts, hin, h1⟩, hO⟩ := h
  exact kappa_eps_deployed_cap rel b L nextE colsE wE join rowIn final w decI hin h1 hO hS hU hV

end RecDeploy
