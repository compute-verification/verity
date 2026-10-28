import Proofs.Flock.Claims.Recursion.BoundDeploy
import Proofs.Flock.Claims.KS.PlainCapsDep
import Proofs.Flock.Soundness.Refine.Seam
import Definitions.Claims.RecursionLin

/-!
# Route 2C's bound at the deployed shape: `λ₂ ≥ 100`

`Claims.Recursion.lamLin2` is the number `BitsLin2` states route 2C's bits at, with the inner Flock's roots linear
(`KSLin2`). Here it is evaluated at the deployed records `KSLin2On` at the dense statements is stated at from its
premises (`ksLin2_deployed`, `ksLin2_deployed_plain`): the compiled record `recBoundAt rel b L` whose inner view runs
`IVStar` of the deployed bound record `bdDeployed`, over the bound record's inner Flock (`Bound.deploy`, the deployed
plain record's numbers).

**`lamLin2_deployed_ge`**: `λ₂(x) ≥ 100` at every statement `x` whose inner statement is in the plain scope
(`FlockDeploy.PlainIn`: up to `200` tables, each at an `m` the plain verifier serves and of a planned shape) and each of
whose `nS x ≤ 2^16` V* parts sets up, at the inner rounds `nums` reads it at, within the masked verifier's served shapes
(`PartIn`, `ZkIn`'s fields at the recursive setup `setupO`), at inner caps of at most `2^64` positions and at most
`2^56` calls in the inner statement's plain setup (`|S(x_I)|`, `innerSetupIns`; 390 at the honest rec-coins
statement), at a count `n ≤ 2^55` of the inner verifier's calls on one draw's inner messages at its coins
(`Deploy.innerCalls`, which `BitsLin2` bounds `n` by) and a count `v ≤ 2^55` of the compiled verifier's own evaluations at a draw's V\*
statements (`setupCalls`, which `BitsLin2` bounds `v` by: at the deployed shapes at most 15,829 a table, check 5's row
commitments among them); the deployed caps `CpDep` are (`CpDep_width_le`), so `lamLin2_deployed_ge_dep`
states it at them with no width premise. Each of `λ₂`'s six terms is at least `103 > 100 + log₂ 6`, from bounds that
are exact rational and real inequalities:

* the inner Flock: `κ_I ≤ 200·2^{−202.98}` (`FlockDeploy.κ_plain_le`), `k_I ≤ 2^32`, `m′ = 210`, rounds `≤ 2^20`;
* each part: `κ_O ≤ 200·2^{−202.98}` (`numsOf_κ_le`, `Flock.Numbers.κ_le_mul` at `Nums`), `k_O ≤ 2^32`, rounds
  `≤ 880002` and `γ_O ≤ 2^68` (`numsOf_γ_le`);
* so `N_I ≤ 2^40`, `c_Be ≤ 2^57`, `c_S = c_Be + |S(x_I)| + 2(n + v) ≤ 2^59`,
  `k(x) ≤ 2^37`, `κ ≤ 2^{−137}`, `γ ≤ 2^218` and `β₂ ≤ 2^88` (`lamLin2_core`).

The bound is crude (`λ₂` there is about 165.8, `κ` binding: the compiled verifier's evaluations lower each collision
term by `log₂(1 + 2v/c_S)`, at most about 0.006 bits at the deployed shapes, and the nearest, `Fdev`'s, sits near 178.9);
it is what the parameters give at every such statement.
It bounds a number and establishes no security level: `KSLin2` at the deployment's dense statements (`KSLin2On` at
`DenseAt`) only follows from premises still open (`PartChecksBV`, `PartOpensB` and `DecompB` at the
coin server's record, which reads back; `ksLin2_deployed_full`), Holo statements being
candidate 3's own model, and `BitsLin2` there, that `Pr[acc ∧ ¬R] ≤ t·2^{−λ₂}`, follows from it
(`bitsLin2On_of_ksLin2On`). The proved level at the deployment is still `Claims.Recursion.lam`'s, about 65 bits, below
the 100-bit floor.
-/

namespace RecDeploy

open Claims (Msg Out)
open FlockDeploy (Coin)

/-! ## The arithmetic -/

theorem logb_le_of_le_pow {y : ℝ} {n : ℕ} (h0 : 0 ≤ y) (h : y ≤ 2 ^ n) : Real.logb 2 y ≤ n := by
  rcases h0.eq_or_lt with rfl | hy
  · simp
  · calc Real.logb 2 y ≤ Real.logb 2 ((2 : ℝ) ^ n) := Real.logb_le_logb_of_le (by norm_num) hy h
      _ = n := by rw [Real.logb_pow, Real.logb_self_eq_one (by norm_num)]; norm_num

theorem le_logb_inv {κ : ℝ} {n : ℕ} (h0 : 0 < κ) (h : 2 ^ n * κ ≤ 1) : (n : ℝ) ≤ Real.logb 2 (1 / κ) := by
  have h2 : (2 : ℝ) ^ n ≤ 1 / κ := by rw [le_div_iff₀ h0]; exact h
  calc (n : ℝ) = Real.logb 2 ((2 : ℝ) ^ n) := by
        rw [Real.logb_pow, Real.logb_self_eq_one (by norm_num)]; norm_num
    _ ≤ Real.logb 2 (1 / κ) := Real.logb_le_logb_of_le (by norm_num) (by positivity) h2

theorem sub_logb_ge {a y : ℝ} {n : ℕ} (h0 : 0 ≤ y) (h : y ≤ 2 ^ n) (ha : (n : ℝ) + 103 ≤ a) :
    103 ≤ a - Real.logb 2 y := by
  have := logb_le_of_le_pow h0 h
  linarith

theorem logb_six_le : Real.logb 2 6 ≤ 3 := by
  have := logb_le_of_le_pow (y := 6) (n := 3) (by norm_num) (by norm_num)
  exact_mod_cast this

theorem sum_fin_le_nat {n c : ℕ} (f : Fin n → ℕ) (h : ∀ i, f i ≤ c) : ∑ i, f i ≤ n * c :=
  calc ∑ i, f i ≤ ∑ _i : Fin n, c := Finset.sum_le_sum fun i _ => h i
    _ = n * c := by simp

theorem sum_fin_le_real {n : ℕ} {c : ℝ} (f : Fin n → ℝ) (h : ∀ i, f i ≤ c) : ∑ i, f i ≤ n * c :=
  calc ∑ i, f i ≤ ∑ _i : Fin n, c := Finset.sum_le_sum fun i _ => h i
    _ = n * c := by simp

theorem two_rpow_neg_le : (2 : ℝ) ^ (-(202.98 : ℝ)) ≤ ((2 : ℝ) ^ 202)⁻¹ := by
  have h1 : (2 : ℝ) ^ (-(202.98 : ℝ)) ≤ (2 : ℝ) ^ (-(202 : ℝ)) :=
    Real.rpow_le_rpow_of_exponent_le (by norm_num) (by norm_num)
  have h2 : (2 : ℝ) ^ (-(202 : ℝ)) = ((2 : ℝ) ^ 202)⁻¹ := by
    rw [Real.rpow_neg (by norm_num), show (202 : ℝ) = ((202 : ℕ) : ℝ) by norm_num, Real.rpow_natCast]
  exact h1.trans h2.le

/-- **`λ₂ ≥ 100` from its numbers**: `lamLin2`'s six terms, written over the numbers they read (the inner verifier's
rounds `VI`, the inner `E^rep`'s accepted reruns `kI` per wait and its waits `wI`, the inner caps' positions `W`, the
compiled verifier's rounds `kx`, each part's `κ`, `γ`, waits and reruns), each at least `103`. -/
theorem lamLin2_core {nS : ℕ} {VI kI wI W kx NI cBe nSet nU : ℕ} {κI : ℝ} {κs γs : Fin nS → ℝ} {ws ks : Fin nS → ℕ}
    (hNI : NI = 1 + wI * kI) (hcBe : cBe = 4 + 2 * ∑ s, ws s * ks s) (hSet : nSet + 2 * nU ≤ 2 ^ 58)
    (hVI : VI ≤ 2 ^ 20) (hkI : kI ≤ 2 ^ 32) (hwI : wI = 210) (hW : W ≤ 2 ^ 64) (hnS : nS ≤ 2 ^ 16)
    (hkx : kx ≤ 2 ^ 37) (hκI0 : 0 < κI) (hκI : κI ≤ 200 * 2 ^ (-(202.98 : ℝ)))
    (hκs0 : ∀ s, 0 ≤ κs s) (hκs : ∀ s, κs s ≤ 200 * 2 ^ (-(202.98 : ℝ))) (hγs : ∀ s, γs s ≤ 2 ^ 68)
    (hws : ∀ s, ws s = 210) (hks : ∀ s, ks s ≤ 2 ^ 32) :
    100 ≤ min (min (min (min (min (Real.logb 2 (1 / (κI + NI * ∑ s, κs s)))
      (255.5 - Real.logb 2 ((NI * ∑ s, Real.sqrt (γs s * kx)) ^ 2) / 2))
      (256 - Real.logb 2 ((4 * VI + 1) * ((cBe : ℝ) + nSet + 2 * nU))))
      (255 - Real.logb 2 (4 * W * ((cBe : ℝ) + nSet + 2 * nU))))
      (256 - Real.logb 2 (4 * ((kI : ℝ) + 1) * ((cBe : ℝ) + nSet + 2 * nU))))
      (255 - Real.logb 2 ((NI + 8 * W + 4 * ((kI : ℝ) + 1)) * VI * ((cBe : ℝ) + nSet + 2 * nU))) - Real.logb 2 6 := by
  have hNI' : NI ≤ 2 ^ 40 := by
    rw [hNI, hwI]
    calc 1 + 210 * kI ≤ 1 + 210 * 2 ^ 32 := by gcongr
      _ ≤ 2 ^ 40 := by norm_num
  have hcBe' : cBe ≤ 2 ^ 57 := by
    have hsum : ∑ s, ws s * ks s ≤ nS * (210 * 2 ^ 32) :=
      sum_fin_le_nat _ fun s => by rw [hws s]; exact Nat.mul_le_mul_left 210 (hks s)
    rw [hcBe]
    calc 4 + 2 * ∑ s, ws s * ks s ≤ 4 + 2 * (2 ^ 16 * (210 * 2 ^ 32)) := by
          gcongr
          exact hsum.trans (Nat.mul_le_mul_right _ hnS)
      _ ≤ 2 ^ 57 := by norm_num
  have hcS : cBe + nSet + 2 * nU ≤ 2 ^ 59 := by omega
  have hNIR : (NI : ℝ) ≤ 2 ^ 40 := by exact_mod_cast hNI'
  have hnR : (nS : ℝ) ≤ 2 ^ 16 := by exact_mod_cast hnS
  have hkxR : (kx : ℝ) ≤ 2 ^ 37 := by exact_mod_cast hkx
  rw [le_sub_iff_add_le]
  refine le_trans (b := (103 : ℝ)) (by linarith [logb_six_le]) ?_
  refine le_min (le_min (le_min (le_min (le_min ?_ ?_) ?_) ?_) ?_) ?_
  · have hS0 : 0 ≤ ∑ s, κs s := Finset.sum_nonneg fun s _ => hκs0 s
    have hS : ∑ s, κs s ≤ 2 ^ 16 * (200 * 2 ^ (-(202.98 : ℝ))) :=
      (sum_fin_le_real _ hκs).trans (by gcongr)
    have hκ0 : 0 < κI + NI * ∑ s, κs s := by
      have := mul_nonneg (Nat.cast_nonneg NI : (0 : ℝ) ≤ NI) hS0
      linarith
    have hκ : κI + NI * ∑ s, κs s ≤
        200 * 2 ^ (-(202.98 : ℝ)) + 2 ^ 40 * (2 ^ 16 * (200 * 2 ^ (-(202.98 : ℝ)))) := by
      have := mul_le_mul hNIR hS hS0 (by norm_num)
      linarith
    have he := two_rpow_neg_le
    have key : (2 : ℝ) ^ 103 * (κI + NI * ∑ s, κs s) ≤ 1 :=
      calc (2 : ℝ) ^ 103 * (κI + NI * ∑ s, κs s)
          ≤ 2 ^ 103 * (200 * 2 ^ (-(202.98 : ℝ)) + 2 ^ 40 * (2 ^ 16 * (200 * 2 ^ (-(202.98 : ℝ))))) := by
            gcongr
        _ = (2 ^ 103 * 200 * (1 + 2 ^ 56)) * (2 : ℝ) ^ (-(202.98 : ℝ)) := by ring
        _ ≤ (2 ^ 103 * 200 * (1 + 2 ^ 56)) * ((2 : ℝ) ^ 202)⁻¹ := by gcongr
        _ ≤ 1 := by norm_num
    exact_mod_cast le_logb_inv (n := 103) hκ0 key
  · have hsq : ∀ s, Real.sqrt (γs s * kx) ≤ 2 ^ 53 := fun s => by
      rw [Real.sqrt_le_left (by norm_num)]
      calc γs s * kx ≤ 2 ^ 68 * kx := mul_le_mul_of_nonneg_right (hγs s) (Nat.cast_nonneg _)
        _ ≤ 2 ^ 68 * 2 ^ 37 := by gcongr
        _ ≤ (2 ^ 53) ^ 2 := by norm_num
    have hT0 : 0 ≤ ∑ s, Real.sqrt (γs s * kx) := Finset.sum_nonneg fun s _ => Real.sqrt_nonneg _
    have hT : ∑ s, Real.sqrt (γs s * kx) ≤ 2 ^ 16 * 2 ^ 53 := (sum_fin_le_real _ hsq).trans (by gcongr)
    have hγ : ((NI : ℝ) * ∑ s, Real.sqrt (γs s * kx)) ^ 2 ≤ 2 ^ 218 :=
      calc ((NI : ℝ) * ∑ s, Real.sqrt (γs s * kx)) ^ 2 ≤ (2 ^ 40 * (2 ^ 16 * 2 ^ 53)) ^ 2 :=
            pow_le_pow_left₀ (mul_nonneg (Nat.cast_nonneg _) hT0) (mul_le_mul hNIR hT hT0 (by norm_num)) 2
        _ = 2 ^ 218 := by ring
    have := logb_le_of_le_pow (n := 218) (by positivity) hγ
    push_cast at this
    linarith
  · refine sub_logb_ge (n := 153) (by positivity) ?_ (by norm_num)
    have : (4 * VI + 1) * (cBe + nSet + 2 * nU) ≤ 2 ^ 153 :=
      calc (4 * VI + 1) * (cBe + nSet + 2 * nU) ≤ (4 * 2 ^ 20 + 1) * 2 ^ 59 := by gcongr
        _ ≤ 2 ^ 153 := by norm_num
    exact_mod_cast this
  · refine sub_logb_ge (n := 152) (by positivity) ?_ (by norm_num)
    have : 4 * W * (cBe + nSet + 2 * nU) ≤ 2 ^ 152 :=
      calc 4 * W * (cBe + nSet + 2 * nU) ≤ 4 * 2 ^ 64 * 2 ^ 59 := by gcongr
        _ ≤ 2 ^ 152 := by norm_num
    exact_mod_cast this
  · refine sub_logb_ge (n := 153) (by positivity) ?_ (by norm_num)
    have : 4 * (kI + 1) * (cBe + nSet + 2 * nU) ≤ 2 ^ 153 :=
      calc 4 * (kI + 1) * (cBe + nSet + 2 * nU) ≤ 4 * (2 ^ 32 + 1) * 2 ^ 59 := by gcongr
        _ ≤ 2 ^ 153 := by norm_num
    exact_mod_cast this
  · refine sub_logb_ge (n := 152) (by positivity) ?_ (by norm_num)
    have : (NI + 8 * W + 4 * (kI + 1)) * VI * (cBe + nSet + 2 * nU) ≤ 2 ^ 152 :=
      calc (NI + 8 * W + 4 * (kI + 1)) * VI * (cBe + nSet + 2 * nU) ≤
          (2 ^ 40 + 8 * 2 ^ 64 + 4 * (2 ^ 32 + 1)) * 2 ^ 20 * 2 ^ 59 := by
            gcongr
        _ ≤ 2 ^ 152 := by norm_num
    exact_mod_cast this

/-- **`λ₂ ≥ 100` at a compiled record and an inner protocol**, from bounds on the numbers `lamLin2` reads. -/
theorem lamLin2_ge_of {C W' : Type} [Fintype C] [Nonempty C] [Inhabited C] (D : Claims.Recursion.Deploy C)
    (I : Claims.Recursion.Inner C W') (Cp : Claims.Recursion.Caps C) (x : Msg) (n v : ℕ)
    (hVI : I.V.k (Claims.Recursion.xI D x) ≤ 2 ^ 20) (hkI : I.k (Claims.Recursion.xI D x) ≤ 2 ^ 32)
    (hwI : I.waits (Claims.Recursion.xI D x) = 210) (hW : Cp.width (Claims.Recursion.xI D x) ≤ 2 ^ 64)
    (hnS : D.nS x ≤ 2 ^ 16) (hSet : Claims.Recursion.setupLen D x ≤ 2 ^ 56) (hU : n ≤ 2 ^ 55) (hV : v ≤ 2 ^ 55)
    (hkx : D.V.k x ≤ 2 ^ 37)
    (hκI0 : 0 < I.κ (Claims.Recursion.xI D x))
    (hκI : I.κ (Claims.Recursion.xI D x) ≤ 200 * 2 ^ (-(202.98 : ℝ)))
    (hκs0 : ∀ s : Fin (D.nS x), 0 ≤ (D.nums x s).κ)
    (hκs : ∀ s : Fin (D.nS x), (D.nums x s).κ ≤ 200 * 2 ^ (-(202.98 : ℝ)))
    (hγs : ∀ s : Fin (D.nS x), (D.nums x s).γ ≤ 2 ^ 68) (hws : ∀ s : Fin (D.nS x), (D.nums x s).waits = 210)
    (hks : ∀ s : Fin (D.nS x), (D.nums x s).k ≤ 2 ^ 32) :
    100 ≤ Claims.Recursion.lamLin2 D I Cp x n v :=
  lamLin2_core (nS := D.nS x) (VI := I.V.k (Claims.Recursion.xI D x)) (kI := I.k (Claims.Recursion.xI D x))
    (wI := I.waits (Claims.Recursion.xI D x)) (W := Cp.width (Claims.Recursion.xI D x)) (kx := D.V.k x)
    (NI := Claims.Recursion.NI D I x) (cBe := Claims.Recursion.cBe D x) (nSet := Claims.Recursion.setupLen D x)
    (nU := n + v) (κI := I.κ (Claims.Recursion.xI D x)) (κs := fun s => (D.nums x s).κ)
    (γs := fun s => (D.nums x s).γ) (ws := fun s => (D.nums x s).waits) (ks := fun s => (D.nums x s).k) rfl rfl
    (by omega) hVI hkI hwI hW hnS hkx hκI0 hκI hκs0 hκs hγs hws hks

/-! ## An outer session's numbers -/

/-- **`κ ≤ n·2^{−202.98}`** at an outer session's numbers, as `Flock.Numbers.κ_le_mul` at a `Flock.Deploy`'s. -/
theorem numsκ_le_mul (m : Claims.Recursion.Nums) {n : ℝ} (hn : 1 ≤ n) (hε : m.εtable ≤ n * 2 ^ (-(205 : ℝ)))
    (hw : 210 ≤ m.waits) (hK : m.cap = Claims.Flock.Numbers.cap) (hN : 2 ^ 10 ≤ m.N₀) (hN' : m.N₀ ≤ n * 2 ^ 23) :
    m.κ ≤ n * 2 ^ (-(202.98 : ℝ)) := by
  have hn0 : (0 : ℝ) ≤ n := by linarith
  have hk1 : 1507 ≤ m.k := Claims.Flock.Numbers.k_ge hN
  have hk2 : (m.k : ℝ) ≤ n * 2 ^ 24 := Claims.Flock.Numbers.k_le_mul hn (le_trans (by norm_num) hN) hN'
  have h1 : 4 * m.εtable ≤ n * 2 ^ (-(203 : ℝ)) := by
    have e : (2 : ℝ) ^ (-(203 : ℝ)) = 4 * 2 ^ (-(205 : ℝ)) := by
      rw [show (-(203 : ℝ)) = 2 + -(205 : ℝ) by norm_num, Real.rpow_add (by norm_num)]; norm_num
    calc 4 * m.εtable ≤ 4 * (n * 2 ^ (-(205 : ℝ))) := by linarith
      _ = n * 2 ^ (-(203 : ℝ)) := by rw [e]; ring
  have h2 : (2 : ℝ) ^ (-(m.waits : ℝ)) ≤ n * 2 ^ (-(210 : ℝ)) :=
    (Real.rpow_le_rpow_of_exponent_le (by norm_num) (neg_le_neg (by exact_mod_cast hw))).trans
      (le_mul_of_one_le_left (by positivity) hn)
  have h3 : 2 * (m.k : ℝ) / m.cap ≤ n * 2 ^ (-(231 : ℝ)) := by
    have hc : (m.cap : ℝ) = 2 ^ 256 := by rw [hK, Claims.Flock.Numbers.cap]; push_cast; ring
    have e : (2 : ℝ) ^ (-(231 : ℝ)) = (2 ^ 231)⁻¹ := by
      rw [Real.rpow_neg (by norm_num), show (231 : ℝ) = ((231 : ℕ) : ℝ) by norm_num, Real.rpow_natCast]
    rw [hc, e, div_le_iff₀ (by positivity)]
    calc 2 * (m.k : ℝ) ≤ 2 * (n * 2 ^ 24) := by linarith
      _ = n * (2 ^ 231)⁻¹ * 2 ^ 256 := by ring
  have h4 : Real.exp (-(m.k : ℝ) / 8) ≤ n * 2 ^ (-(250 : ℝ)) :=
    (Claims.Flock.Numbers.exp_neg_k_le hk1).trans (le_mul_of_one_le_left (by positivity) hn)
  unfold Claims.Recursion.Nums.κ
  calc 4 * m.εtable + 2 ^ (-(m.waits : ℝ)) + 2 * m.k / m.cap + Real.exp (-(m.k : ℝ) / 8)
      ≤ n * 2 ^ (-(203 : ℝ)) + n * 2 ^ (-(210 : ℝ)) + n * 2 ^ (-(231 : ℝ)) + n * 2 ^ (-(250 : ℝ)) :=
        add_le_add (add_le_add (add_le_add h1 h2) h3) h4
    _ = n * (2 ^ (-(203 : ℝ)) + 2 ^ (-(210 : ℝ)) + 2 ^ (-(231 : ℝ)) + 2 ^ (-(250 : ℝ))) := by ring
    _ ≤ n * 2 ^ (-(202.98 : ℝ)) := mul_le_mul_of_nonneg_left Claims.Flock.Numbers.terms_le hn0

theorem numsκ_nonneg (m : Claims.Recursion.Nums) (h : 0 ≤ m.εtable) : 0 ≤ m.κ := by
  unfold Claims.Recursion.Nums.κ
  have h2 : (0 : ℝ) ≤ 2 ^ (-(m.waits : ℝ)) := by positivity
  have h3 : (0 : ℝ) ≤ 2 * (m.k : ℝ) / m.cap := by positivity
  have h4 := Real.exp_pos (-(m.k : ℝ) / 8)
  linarith

/-- **A V\* part in scope**: its statement sets up at the recursive setup (`setupO`) with `1 ≤ n ≤ 200` tables at two
link points, each at an `m` the masked verifier serves and of a shape in `InRange`, as `FlockDeploy.ZkIn` at Flock's
setup.

Assumption record. No claim: a scope condition on the statement, the parts' counterpart of `FlockDeploy.PlainIn`.
`lamLin2_deployed_ge` and `lamLin2_deployed_ge_dep` take it at each staged V\* part (`hO`), at the inner rounds `pick`
reads the part at. It assumes that each part's statement, run through the verifier's own setup (`setupO`), has 1 to 200
tables at at most two link points, each at `25 ≤ m ≤ 35` and of a shape in `InRange`. That is a deterministic
computation on public data and says nothing about the prover or the world. It is reasonable because these are the
shapes the masked verifier is parametrized for, the scope `ZkIn` states for a masked session: `λ₂`'s numbers are
stated there and not claimed outside it. It is no hash premise. It reduces to evaluating `setupO` on the staged
statements, and a refusal of out-of-scope parts in the compiled verifier's scope check (`scopeCheck`) would discharge
it, as an input condition should be. -/
structure PartIn (xO : Msg) (s : RecIns) (spec : _root_.Flock.SessionSpec) (sts : Array _root_.Flock.Stmt) : Prop where
  setup : setupO xO = some (s, spec, sts)
  size_pos : 0 < sts.size
  size_le : sts.size ≤ 200
  points : spec.points ≤ 2
  tab : ∀ st ∈ sts.toList, 25 ≤ st.m ∧ st.m ≤ 35 ∧ (FlockDeploy.acct spec st).InRange

theorem tablesO_eq {xO : Msg} {s : RecIns} {spec : _root_.Flock.SessionSpec} {sts : Array _root_.Flock.Stmt}
    (hs : setupO xO = some (s, spec, sts)) : tablesO xO = sts.toList.map fun st => (spec, st) := by
  simp only [tablesO, hs]

section Part

variable {xO : Msg} {s : RecIns} {spec : _root_.Flock.SessionSpec} {sts : Array _root_.Flock.Stmt}

theorem εtableO_le (h : PartIn xO s spec sts) : εtableO xO ≤ sts.size * 2 ^ (-(205 : ℝ)) := by
  unfold εtableO
  rw [tablesO_eq h.setup, List.map_map]
  refine (List.sum_le_card_nsmul _ ((2 : ℝ) ^ (-(205 : ℝ))) fun a ha => ?_).trans (le_of_eq (by simp))
  obtain ⟨st, hst, rfl⟩ := List.mem_map.1 ha
  have := h.tab st hst
  exact FlockDeploy.tableErr_zk_le spec st this.1 this.2.1 this.2.2

theorem N₀O_le (h : PartIn xO s spec sts) : 2 ^ 10 ≤ N₀O xO ∧ N₀O xO ≤ sts.size * 2 ^ 23 := by
  have hb : ∀ st ∈ sts.toList, 2 ^ 10 ≤ FlockDeploy.tableLen true st ∧ FlockDeploy.tableLen true st ≤ 2 ^ 23 :=
    fun st hst =>
      have := h.tab st hst
      FlockDeploy.tableLen_zk_bounds st this.1 this.2.1
  have h0 := h.size_pos
  unfold N₀O
  rw [tablesO_eq h.setup, List.map_map]
  constructor
  · have hm : FlockDeploy.tableLen true sts[0] ∈
        sts.toList.map ((fun p => FlockDeploy.tableLen true p.2) ∘ fun st => (spec, st)) :=
      List.mem_map.2 ⟨sts[0], Array.getElem_mem_toList h0, rfl⟩
    refine (hb _ (Array.getElem_mem_toList h0)).1.trans (List.single_le_sum (fun a ha => ?_) _ hm)
    obtain ⟨st, hst, rfl⟩ := List.mem_map.1 ha
    exact le_trans (by norm_num) (hb st hst).1
  · refine (List.sum_le_card_nsmul _ ((2 : ℝ) ^ 23) fun a ha => ?_).trans (le_of_eq (by simp))
    obtain ⟨st, hst, rfl⟩ := List.mem_map.1 ha
    exact (hb st hst).2

theorem VO_k_le (h : PartIn xO s spec sts) : VO.k xO ≤ 2 + 4400 * sts.size := by
  have := FlockDeploy.rounds_le (zk := true) (spec := spec) (sts := sts) (b := 2200) fun st hst =>
    have := h.tab st hst
    FlockDeploy.repRounds_zk_le spec st this.1 this.2.1 this.2.2.1 (this.2.2.2.1.trans (by norm_num)) h.points
  have hk : VO.k xO = 2 + FlockDeploy.rounds true spec sts := by
    show (match setupO xO with | some (_, spec, sts) => 2 + FlockDeploy.rounds true spec sts | none => 1) = _
    rw [h.setup]
  omega

theorem weightO_le (h : PartIn xO s spec sts) : weightO xO ≤ sts.size * 2 ^ 33 := by
  unfold weightO
  rw [tablesO_eq h.setup, List.map_map]
  refine (List.sum_le_card_nsmul _ ((2 : ℝ) ^ 33) fun a ha => ?_).trans (le_of_eq (by simp))
  obtain ⟨st, hst, rfl⟩ := List.mem_map.1 ha
  have := h.tab st hst
  exact FlockDeploy.tableWeight_zk_le st this.1 this.2.1 this.2.2.1

/-- **A part's `κ_O ≤ n·2^{−202.98}`**, `n` its tables. -/
theorem numsOf_κ_le (h : PartIn xO s spec sts) : (numsOf xO).κ ≤ sts.size * 2 ^ (-(202.98 : ℝ)) := by
  obtain ⟨hlo, hhi⟩ := N₀O_le h
  exact numsκ_le_mul (numsOf xO) (FlockDeploy.size_one_le h.size_pos) (εtableO_le h) (le_of_eq rfl) rfl hlo hhi

theorem numsOf_k_le (h : PartIn xO s spec sts) : (numsOf xO).k ≤ 2 ^ 32 := by
  obtain ⟨hlo, hhi⟩ := N₀O_le h
  have h1 : ((numsOf xO).k : ℝ) ≤ sts.size * 2 ^ 24 :=
    Claims.Flock.Numbers.k_le_mul (N := N₀O xO) (FlockDeploy.size_one_le h.size_pos) (le_trans (by norm_num) hlo) hhi
  have h2 : (sts.size : ℝ) ≤ 200 := by exact_mod_cast h.size_le
  have h3 : ((numsOf xO).k : ℝ) ≤ 2 ^ 32 := by nlinarith
  exact_mod_cast h3

theorem numsOf_rounds_le (h : PartIn xO s spec sts) : (numsOf xO).rounds ≤ 880002 := by
  have h1 := VO_k_le h
  have h2 := h.size_le
  show VO.k xO ≤ 880002
  omega

/-- **A part's `γ_O ≤ 2^68`**: at most `880002` rounds and a weight at most `200·2^33`. -/
theorem numsOf_γ_le (h : PartIn xO s spec sts) : (numsOf xO).γ ≤ 2 ^ 68 := by
  have hr : ((numsOf xO).rounds : ℝ) ≤ 2 ^ 20 := by
    have : ((numsOf xO).rounds : ℝ) ≤ 880002 := by exact_mod_cast numsOf_rounds_le h
    linarith
  have hr0 : (0 : ℝ) ≤ (numsOf xO).rounds := Nat.cast_nonneg _
  have hC : (numsOf xO).weight ≤ 2 ^ 41 := by
    have h1 := weightO_le h
    have h2 : (sts.size : ℝ) ≤ 200 := by exact_mod_cast h.size_le
    show weightO xO ≤ 2 ^ 41
    nlinarith
  have hs : Real.sqrt (2 * (numsOf xO).rounds * (numsOf xO).weight) ≤ 2 ^ 31 := by
    rw [Real.sqrt_le_left (by positivity)]
    nlinarith [mul_le_mul_of_nonneg_left hC hr0]
  have hsr : Real.sqrt (2 * (numsOf xO).rounds * (numsOf xO).weight) + (numsOf xO).rounds ≤ 2 ^ 32 := by linarith
  unfold Claims.Recursion.Nums.γ
  calc 16 * (Real.sqrt (2 * (numsOf xO).rounds * (numsOf xO).weight) + (numsOf xO).rounds) ^ 2
      ≤ 16 * (2 ^ 32) ^ 2 := by gcongr
    _ = 2 ^ 68 := by norm_num

end Part

/-! ## At the deployment -/

/-- **Route 2C's bound `λ₂` evaluates to at least `100` at the deployed shape**: `λ₂(x) = lamLin2` at the compiled
record whose inner view runs the deployed bound record's `IVStar` (`recBoundAt rel b L`, `bdDeployed`) and at the bound
record's inner Flock (`Bound.deploy`), at every statement whose inner statement is in the plain scope (`hin`), each of
whose `nS x ≤ 2^16` V* parts sets up within the masked verifier's served shapes at the inner rounds its numbers are
read at (`hO`), at inner caps `Cp` of at most `2^64` positions (`hW`). The element-level steps, columns and weight of `DE`
(`nextE`, `colsE`, `wE`), the code side's `join`, `rowIn`, `final` and `γ`'s weight at `IVStar` (`w`) are any: `λ₂`
reads none of them. This bounds a number: `KSLin2` at the deployment's dense statements only follows from premises still
open (`ksLin2_deployed_open`), and `BitsLin2` there follows from it (`bitsLin2On_of_ksLin2On`). -/
theorem lamLin2_deployed_ge (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings)
    (L : Claims.Recursion.VStarB.LayoutB) (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (decI : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace) (Cp : Claims.Recursion.Caps Coin)
    {x : Msg} {s : FlockDeploy.Ins} {spec : _root_.Flock.SessionSpec} {sts : Array _root_.Flock.Stmt}
    (hin : FlockDeploy.PlainIn (xIOf x) s spec sts)
    (hO : ∀ j < nS x, ∃ i spec' sts', PartIn (staged b.st x (pick b.st x j) j) i spec' sts')
    (hn : nS x ≤ 2 ^ 16) (hS : (innerSetupIns (xIOf x)).length ≤ 2 ^ 56) {n : ℕ} (hU : n ≤ 2 ^ 55) {v : ℕ}
    (hV : v ≤ 2 ^ 55) (hW : Cp.width (xIOf x) ≤ 2 ^ 64) :
    100 ≤ Claims.Recursion.lamLin2
      ((recBoundAt rel b L).record (bdDeployed rel nextE colsE wE join rowIn final w).IVStar)
      (Claims.Recursion.Inner.ofFlock (bdDeployed rel nextE colsE wE join rowIn final w).deploy decI) Cp x n v := by
  have hsz : (sts.size : ℝ) ≤ 200 := by exact_mod_cast hin.size_le
  have hVI : (FlockDeploy.deployPlain rel).V.k (xIOf x) ≤ 2 ^ 20 :=
    (FlockDeploy.Vk_plain_le rel hin).trans <|
      calc 2 + 4400 * sts.size ≤ 2 + 4400 * 200 := by gcongr; exact hin.size_le
        _ ≤ 2 ^ 20 := by norm_num
  have hkI : (FlockDeploy.deployPlain rel).k (xIOf x) ≤ 2 ^ 32 := by
    obtain ⟨hlo, hhi⟩ := FlockDeploy.N₀_le hin.setup hin.size_pos fun st hst =>
      have := hin.tab st hst
      FlockDeploy.tableLen_plain_bounds st this.1 this.2.1
    have h1 : ((FlockDeploy.deployPlain rel).k (xIOf x) : ℝ) ≤ sts.size * 2 ^ 24 :=
      Claims.Flock.Numbers.k_le_mul (N := FlockDeploy.N₀ false (xIOf x)) (FlockDeploy.size_one_le hin.size_pos)
        (le_trans (by norm_num) hlo) hhi
    have h3 : ((FlockDeploy.deployPlain rel).k (xIOf x) : ℝ) ≤ 2 ^ 32 := by nlinarith
    exact_mod_cast h3
  have hsum : ((List.range (nS x)).map fun j => (nums b.st x j).rounds).sum ≤ nS x * 880002 := by
    refine (List.sum_le_card_nsmul _ 880002 fun a ha => ?_).trans (le_of_eq (by simp))
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
  refine lamLin2_ge_of ((recBoundAt rel b L).record (bdDeployed rel nextE colsE wE join rowIn final w).IVStar)
    (Claims.Recursion.Inner.ofFlock (bdDeployed rel nextE colsE wE join rowIn final w).deploy decI) Cp x n v
    hVI hkI rfl hW hn hS hU hV hkx hκI0 hκI
    (fun j => numsκ_nonneg _ (εtableO_nonneg (staged b.st x (pick b.st x j) j))) (fun j => ?_) (fun j => ?_)
    (fun _ => rfl) (fun j => ?_)
  · obtain ⟨i, spec', sts', hp⟩ := hO j j.isLt
    have hsz' : (sts'.size : ℝ) ≤ 200 := by exact_mod_cast hp.size_le
    exact (numsOf_κ_le hp).trans (mul_le_mul_of_nonneg_right hsz' (by positivity))
  · obtain ⟨i, spec', sts', hp⟩ := hO j j.isLt
    exact numsOf_γ_le hp
  · obtain ⟨i, spec', sts', hp⟩ := hO j j.isLt
    exact numsOf_k_le hp

/-! ## At the deployed caps -/

/-- Every `fast100` schedule has at most `7` levels, each of codeword length at most `2^56`. -/
theorem fast100_small {m : ℕ} {sch : FlockSoundness.Accounting.Schedule}
    (h : FlockSoundness.Accounting.fast100 m = some sch) :
    sch.levels.length ≤ 7 ∧ ∀ l ∈ sch.levels, l.logLen ≤ 56 := by
  have hm : 22 ≤ m ∧ m ≤ 35 := by
    by_contra hc
    have : FlockSoundness.Accounting.fast100 m = none := by unfold FlockSoundness.Accounting.fast100; split <;> simp_all
    simp [this] at h
  obtain ⟨h1, h2⟩ := hm
  have key : (decide (sch.levels.length ≤ 7) && sch.levels.all fun l => decide (l.logLen ≤ 56)) = true := by
    interval_cases m <;> (simp only [FlockSoundness.Accounting.fast100, Option.some.injEq] at h; subst h; decide +kernel)
  simp only [Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true] at key
  exact key

/-- The deployed plain record's schedule, where it has one, is a small `fast100` schedule. -/
theorem depSetup_small {x : Msg} {spec : _root_.Flock.SessionSpec} {st : _root_.Flock.Stmt}
    {sch : _root_.Flock.Schedule} (hd : FlockDeploy.KS.depSetup x = some (spec, st, sch)) :
    sch.levels.size ≤ 7 ∧ ∀ l ∈ (FlockSoundness.Refine.schedOf sch).levels, l.logLen ≤ 56 := by
  obtain ⟨-, -, -, -, hF⟩ := FlockDeploy.KS.depSetup_some hd
  have hA : FlockSoundness.Accounting.fast100 st.m = some (FlockSoundness.Refine.schedOf sch) := by
    have e := FlockSoundness.Refine.fast100_schedOf st.m
    rw [hF] at e
    exact e.symm
  obtain ⟨h7, hall⟩ := fast100_small hA
  refine ⟨?_, hall⟩
  simpa only [FlockSoundness.Refine.schedOf, List.length_map, Array.length_toList] using h7

/-- **The deployed caps have at most `2^64` positions**: two reps, at most `7` levels, each of at most `2^56`
positions. -/
theorem CpDep_width_le (x : Msg) : FlockDeploy.KS.CpDep.width x ≤ 2 ^ 64 := by
  have hL : FlockDeploy.KS.depL x ≤ 7 := by
    unfold FlockDeploy.KS.depL
    rcases hd : FlockDeploy.KS.depSetup x with _ | ⟨spec, st, sch⟩
    · exact Nat.zero_le _
    · show sch.levels.size - 1 ≤ 7
      exact (Nat.sub_le _ _).trans (depSetup_small hd).1
  have hg : ∀ ℓ, FlockDeploy.KS.depLogLen x ℓ ≤ 56 := by
    intro ℓ
    unfold FlockDeploy.KS.depLogLen
    rcases hd : FlockDeploy.KS.depSetup x with _ | ⟨spec, st, sch⟩
    · exact Nat.zero_le _
    · show ((FlockSoundness.Refine.schedOf sch).levels.getD ℓ default).logLen ≤ 56
      rw [List.getD_eq_getElem?_getD]
      cases hl : (FlockSoundness.Refine.schedOf sch).levels[ℓ]? with
      | none => exact Nat.zero_le _
      | some l => exact (depSetup_small hd).2 l (List.mem_of_getElem? hl)
  show Fintype.card (FlockDeploy.KS.DepIdx x) ≤ 2 ^ 64
  rw [Fintype.card_sigma]
  simp only [Fintype.card_sigma, Fintype.card_fin]
  calc ∑ _r : Fin 2, ∑ ℓ : Fin (FlockDeploy.KS.depL x), 2 ^ FlockDeploy.KS.depLogLen x (ℓ.val + 1)
      ≤ ∑ _r : Fin 2, ∑ _ℓ : Fin (FlockDeploy.KS.depL x), 2 ^ 56 :=
        Finset.sum_le_sum fun _ _ => Finset.sum_le_sum fun ℓ _ => Nat.pow_le_pow_right (by norm_num) (hg _)
    _ = 2 * (FlockDeploy.KS.depL x * 2 ^ 56) := by
        rw [Finset.sum_const, Finset.sum_const, Finset.card_univ, Finset.card_univ, Fintype.card_fin, Fintype.card_fin,
          smul_eq_mul, smul_eq_mul]
    _ ≤ 2 * (7 * 2 ^ 56) := by gcongr
    _ ≤ 2 ^ 64 := by norm_num

/-- **Route 2C's bound `λ₂` evaluates to at least `100` at the deployed caps**: `lamLin2_deployed_ge` at `CpDep`, the
caps `ksLin2_deployed_plain` states `KSLin2` at from its premises, whose width bound is `CpDep_width_le`. This bounds a
number: `KSLin2` at the deployment's dense statements only follows from premises still open (`ksLin2_deployed_open`),
and `BitsLin2` there follows from it (`bitsLin2On_of_ksLin2On`).

`hn`, `hS` and `hO` are deployment-size conditions on public data, and none is a hash premise or says anything about
the prover's behaviour: `hn` says `x` stages at most `2^16` V\* parts and `hS` that the inner setup `x` names has at
most `2^56` hm96 inputs, both numbers read off `x`; `hO` is the scope condition `PartIn` at each staged V\* part (1–200
tables, at most two link points, `25 ≤ m ≤ 35`), which the verifier's own setup `setupO` evaluates on the staged
public statement. -/
theorem lamLin2_deployed_ge_dep (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings)
    (L : Claims.Recursion.VStarB.LayoutB) (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (decI : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace)
    {x : Msg} {s : FlockDeploy.Ins} {spec : _root_.Flock.SessionSpec} {sts : Array _root_.Flock.Stmt}
    (hin : FlockDeploy.PlainIn (xIOf x) s spec sts)
    (hO : ∀ j < nS x, ∃ i spec' sts', PartIn (staged b.st x (pick b.st x j) j) i spec' sts')
    (hn : nS x ≤ 2 ^ 16) (hS : (innerSetupIns (xIOf x)).length ≤ 2 ^ 56) {n : ℕ} (hU : n ≤ 2 ^ 55) {v : ℕ}
    (hV : v ≤ 2 ^ 55) :
    100 ≤ Claims.Recursion.lamLin2
      ((recBoundAt rel b L).record (bdDeployed rel nextE colsE wE join rowIn final w).IVStar)
      (Claims.Recursion.Inner.ofFlock (bdDeployed rel nextE colsE wE join rowIn final w).deploy decI)
      FlockDeploy.KS.CpDep x n v :=
  lamLin2_deployed_ge rel b L nextE colsE wE join rowIn final w decI FlockDeploy.KS.CpDep hin hO hn hS hU hV
    (CpDep_width_le (xIOf x))

end RecDeploy
