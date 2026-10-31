import Proofs.Flock.Claims.Recursion.LamDeployedTight
import Proofs.Flock.Claims.Recursion.BitsLin2

/-!
# Route 2C's split bound at the deployment's sizes: `κ ≤ 2^{−167.7}` and `ε ≤ 2^{−174.95}`

`Claims.Recursion.bitsLin2KOn_of_ksLin2On` bounds `Pr[acc ∧ ¬R]` by `κ + t·ε` (`Claims.Recursion.epsLin2`): `κ` once a
session, `ε` once per SHA-512 call of the prover's run. At the deployment's sizes `κ ≤ 2^{−167.7}` and
`ε ≤ 2^{−174.95}` (`kappa_eps_deployed`). The sizes are side conditions on the public statement `x` and on the call
bounds `n` and `v` `BitsLin2KOn` is read at:

* the inner statement is one plain table (`FlockDeploy.PlainIn` at `|sts| = 1`), so the inner verifier's rounds are at
  most `4402`, `k_I ≤ 12343986` (`k_le_one`) and `κ_I ≤ 2^{−202.98}`;
* `x` stages at most `16` V\* parts, each one `--zk` table (`PartIn` at `|sts'| = 1`), so each part's rounds are at most
  `4402`, `k_O ≤ 12343986`, `κ_O ≤ 2^{−202.98}` and `γ_O ≤ 2^{51}` (`numsOf_γ_le_one`), `c_Be ≤ 82951585924` and the
  compiled verifier's rounds `k(x) ≤ 74835`;
* `|S(x_I)|`, `n` and `v` are each at most `2^{32}`, so `c_S = c_Be + |S(x_I)| + 2(n + v) ≤ 104426422404`;
* the deployed caps have at most `5591040` positions (`CpDep_width_le_tight`).

`κ ≤ (1 + 16·N_I)·2^{−202.98}`, and `1 + 16·N_I ≤ 2^{35.28}`. Each of `ε`'s five terms `2^{−(a − log₂ y)}` is at most
`u·2^{−255}` at `a ≥ 255` and `y ≤ u` (`two_rpow_neg_sub_le`; `γ`'s at `y ≤ u²`), and the five `u` sum to at most
`2^{80.05}`. `Fdev`'s, `β₂·c_S`, is all but about `2^{−11}` of the sum: it is the term that sets `ε`.
-/

namespace RecDeploy

open Claims (Msg Out)
open FlockDeploy (Coin)

/-! ## The arithmetic -/

/-- `y ≤ 2^{N/q}` wherever `y^q ≤ 2^N`. -/
theorem le_two_rpow_of_pow_le {y : ℝ} {q N : ℕ} (h0 : 0 ≤ y) (hq : 0 < q) (h : y ^ q ≤ 2 ^ N) :
    y ≤ (2 : ℝ) ^ ((N : ℝ) / q) := by
  rcases h0.eq_or_lt with rfl | hy
  · positivity
  · calc y = 2 ^ Real.logb 2 y := (Real.rpow_logb (by norm_num) (by norm_num) hy).symm
      _ ≤ 2 ^ ((N : ℝ) / q) :=
        Real.rpow_le_rpow_of_exponent_le (by norm_num) (logb_le_div_of_pow_le h0 hq h)

theorem logb_le_logb_of_one_le {y u : ℝ} (hy : 0 ≤ y) (hyu : y ≤ u) (hu : 1 ≤ u) :
    Real.logb 2 y ≤ Real.logb 2 u := by
  rcases hy.eq_or_lt with rfl | hy
  · rw [Real.logb_zero]
    exact Real.logb_nonneg (by norm_num) hu
  · exact Real.logb_le_logb_of_le (by norm_num) hy hyu

theorem two_rpow_le_mul {e u : ℝ} (hu : 0 < u) (h : e ≤ Real.logb 2 u - 255) :
    (2 : ℝ) ^ e ≤ u * 2 ^ (-(255 : ℝ)) :=
  calc (2 : ℝ) ^ e ≤ 2 ^ (Real.logb 2 u + -(255 : ℝ)) :=
        Real.rpow_le_rpow_of_exponent_le (by norm_num) (by linarith)
    _ = u * 2 ^ (-(255 : ℝ)) := by
        rw [Real.rpow_add (by norm_num), Real.rpow_logb (by norm_num) (by norm_num) hu]

/-- **A term `2^{−(a − log₂ y)}` at `a ≥ 255` is at most `u·2^{−255}`**, at `0 ≤ y ≤ u` and `1 ≤ u`. -/
theorem two_rpow_neg_sub_le {a y u : ℝ} (ha : 255 ≤ a) (hy : 0 ≤ y) (hyu : y ≤ u) (hu : 1 ≤ u) :
    (2 : ℝ) ^ (-(a - Real.logb 2 y)) ≤ u * 2 ^ (-(255 : ℝ)) := by
  have := logb_le_logb_of_one_le hy hyu hu
  exact two_rpow_le_mul (by linarith) (by linarith)

/-- **`γ`'s term `2^{−(255.5 − ½·log₂ y)}` is at most `u·2^{−255}`**, at `0 ≤ y ≤ u²` and `1 ≤ u`. -/
theorem two_rpow_neg_sub_half_le {y u : ℝ} (hy : 0 ≤ y) (hyu : y ≤ u ^ 2) (hu : 1 ≤ u) :
    (2 : ℝ) ^ (-(255.5 - Real.logb 2 y / 2)) ≤ u * 2 ^ (-(255 : ℝ)) := by
  have h := logb_le_logb_of_one_le hy hyu (by nlinarith)
  rw [Real.logb_pow] at h
  push_cast at h
  exact two_rpow_le_mul (by linarith) (by linarith)

/-- `a^q ≤ 2^N` on `ℕ` is `a^q ≤ 2^N` on `ℝ`; `a` stays a variable here, so no simproc evaluates the literal power it
is applied at. -/
theorem cast_pow_le_two_pow {a q N : ℕ} (h : a ^ q ≤ 2 ^ N) : (a : ℝ) ^ q ≤ (2 : ℝ) ^ N := by
  exact_mod_cast h

/-- **`⌈4·N₀/e⌉ ≤ 12343986`** at `N₀ ≤ 2^23`, one table's: `4·2^23/e < 12343985.7`. -/
theorem k_le_one {N : ℝ} (hN : N ≤ 2 ^ 23) : ⌈4 * N / Real.exp 1⌉₊ ≤ 12343986 := by
  have he0 : 0 < Real.exp 1 := Real.exp_pos 1
  have hlo := Real.exp_one_gt_d9
  have h2 : (2 : ℝ) ^ 23 = 8388608 := by norm_num
  refine Nat.ceil_le.2 ?_
  rw [div_le_iff₀ he0]
  push_cast
  linarith

/-- **`κ ≤ 2^{−167.7}` and `ε ≤ 2^{−174.95}` from their numbers**: `κ` and `epsLin2`'s five terms, over the numbers
they read as in `lamLin2_core_tight`, at the inner verifier's rounds `VI ≤ 4402`, the inner and each part's `E^rep`
reruns per wait at most `12343986`, at most `16` parts, the inner caps' positions `W ≤ 5591040`, the compiled
verifier's rounds `kx ≤ 74835` and `|S(x_I)| + 2(n + v) ≤ 5·2^32`. -/
theorem kappa_eps_core {nS : ℕ} {VI kI wI W kx NI cBe nSet nU : ℕ} {κI : ℝ} {κs γs : Fin nS → ℝ}
    {ws ks : Fin nS → ℕ} (hNI : NI = 1 + wI * kI) (hcBe : cBe = 4 + 2 * ∑ s, ws s * ks s)
    (hSet : nSet + 2 * nU ≤ 5 * 2 ^ 32) (hVI : VI ≤ 4402) (hkI : kI ≤ 12343986) (hwI : wI = 210)
    (hW : W ≤ 5591040) (hnS : nS ≤ 16) (hkx : kx ≤ 74835) (hκI : κI ≤ 2 ^ (-(202.98 : ℝ)))
    (hκs : ∀ s, κs s ≤ 2 ^ (-(202.98 : ℝ))) (hγs : ∀ s, γs s ≤ 2 ^ 51) (hws : ∀ s, ws s = 210)
    (hks : ∀ s, ks s ≤ 12343986) :
    κI + NI * ∑ s, κs s ≤ 2 ^ (-(167.7 : ℝ)) ∧
      (2 : ℝ) ^ (-(255.5 - Real.logb 2 ((NI * ∑ s, Real.sqrt (γs s * kx)) ^ 2) / 2)) +
        (2 : ℝ) ^ (-(256 - Real.logb 2 ((4 * VI + 1) * ((cBe : ℝ) + nSet + 2 * nU)))) +
        (2 : ℝ) ^ (-(255 - Real.logb 2 (4 * W * ((cBe : ℝ) + nSet + 2 * nU)))) +
        (2 : ℝ) ^ (-(256 - Real.logb 2 (4 * ((kI : ℝ) + 1) * ((cBe : ℝ) + nSet + 2 * nU)))) +
        (2 : ℝ) ^ (-(255 - Real.logb 2 ((NI + 8 * W + 4 * ((kI : ℝ) + 1)) * VI * ((cBe : ℝ) + nSet + 2 * nU)))) ≤
        2 ^ (-(174.95 : ℝ)) := by
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
  have hcS : cBe + nSet + 2 * nU ≤ 104426422404 := by omega
  have hNIR : (NI : ℝ) ≤ 2592237061 := by exact_mod_cast hNI'
  have hcSR : (cBe : ℝ) + nSet + 2 * nU ≤ 104426422404 := by exact_mod_cast hcS
  have hc0 : (0 : ℝ) ≤ (cBe : ℝ) + nSet + 2 * nU := by positivity
  have hVIR : (VI : ℝ) ≤ 4402 := by exact_mod_cast hVI
  have hkIR : (kI : ℝ) ≤ 12343986 := by exact_mod_cast hkI
  have hWR : (W : ℝ) ≤ 5591040 := by exact_mod_cast hW
  have hnSR : (nS : ℝ) ≤ 16 := by exact_mod_cast hnS
  have hkxR : (kx : ℝ) ≤ 74835 := by exact_mod_cast hkx
  constructor
  · have hS : ∑ s, κs s ≤ 16 * 2 ^ (-(202.98 : ℝ)) := by
      refine (sum_fin_le_real _ hκs).trans ?_
      gcongr
    have h1 : (NI : ℝ) * ∑ s, κs s ≤ 2592237061 * (16 * 2 ^ (-(202.98 : ℝ))) :=
      (mul_le_mul_of_nonneg_left hS (Nat.cast_nonneg _)).trans
        (mul_le_mul_of_nonneg_right hNIR (by positivity))
    have hp0 : (0 : ℝ) ≤ 2 ^ (-(202.98 : ℝ)) := by positivity
    have e : (2 : ℝ) ^ (((882 : ℕ) : ℝ) / ((25 : ℕ) : ℝ)) * 2 ^ (-(202.98 : ℝ)) = 2 ^ (-(167.7 : ℝ)) := by
      rw [← Real.rpow_add (by norm_num)]
      congr 1
      norm_num
    have hK : ((41475792977 : ℕ) : ℝ) ≤ 2 ^ (((882 : ℕ) : ℝ) / ((25 : ℕ) : ℝ)) :=
      le_two_rpow_of_pow_le (Nat.cast_nonneg _) (by norm_num) (cast_pow_le_two_pow (by decide +kernel))
    calc κI + NI * ∑ s, κs s ≤ 2 ^ (-(202.98 : ℝ)) + 2592237061 * (16 * 2 ^ (-(202.98 : ℝ))) := add_le_add hκI h1
      _ = ((41475792977 : ℕ) : ℝ) * 2 ^ (-(202.98 : ℝ)) := by push_cast; ring
      _ ≤ 2 ^ (((882 : ℕ) : ℝ) / ((25 : ℕ) : ℝ)) * 2 ^ (-(202.98 : ℝ)) := mul_le_mul_of_nonneg_right hK hp0
      _ = 2 ^ (-(167.7 : ℝ)) := e
  · have hsq : ∀ s, Real.sqrt (γs s * kx) ≤ 13000000000 := fun s => by
      rw [Real.sqrt_le_left (by norm_num)]
      calc γs s * kx ≤ 2 ^ 51 * kx := mul_le_mul_of_nonneg_right (hγs s) (Nat.cast_nonneg _)
        _ ≤ 2 ^ 51 * 74835 := by gcongr
        _ ≤ 13000000000 ^ 2 := by norm_num
    have hT0 : 0 ≤ ∑ s, Real.sqrt (γs s * kx) := Finset.sum_nonneg fun s _ => Real.sqrt_nonneg _
    have hT : ∑ s, Real.sqrt (γs s * kx) ≤ 16 * 13000000000 := by
      refine (sum_fin_le_real _ hsq).trans ?_
      gcongr
    have hY : (NI : ℝ) * ∑ s, Real.sqrt (γs s * kx) ≤ 2592237061 * (16 * 13000000000) :=
      mul_le_mul hNIR hT hT0 (by norm_num)
    have hY0 : (0 : ℝ) ≤ (NI : ℝ) * ∑ s, Real.sqrt (γs s * kx) := mul_nonneg (Nat.cast_nonneg _) hT0
    have hp2 : (0 : ℝ) ≤ 2 ^ (-(255 : ℝ)) := by positivity
    have e : (2 : ℝ) ^ (((1601 : ℕ) : ℝ) / ((20 : ℕ) : ℝ)) * 2 ^ (-(255 : ℝ)) = 2 ^ (-(174.95 : ℝ)) := by
      rw [← Real.rpow_add (by norm_num)]
      congr 1
      norm_num
    refine le_trans (add_le_add (add_le_add (add_le_add (add_le_add
      (two_rpow_neg_sub_half_le (u := 2592237061 * (16 * 13000000000)) (by positivity)
        (pow_le_pow_left₀ hY0 hY 2) (by norm_num))
      (two_rpow_neg_sub_le (u := (4 * 4402 + 1) * 104426422404) (by norm_num) (by positivity)
        (mul_le_mul (by linarith) hcSR hc0 (by norm_num)) (by norm_num)))
      (two_rpow_neg_sub_le (u := 4 * 5591040 * 104426422404) (by norm_num) (by positivity)
        (mul_le_mul (by linarith) hcSR hc0 (by norm_num)) (by norm_num)))
      (two_rpow_neg_sub_le (u := 4 * (12343986 + 1) * 104426422404) (by norm_num) (by positivity)
        (mul_le_mul (by linarith) hcSR hc0 (by norm_num)) (by norm_num)))
      (two_rpow_neg_sub_le (u := (2592237061 + 8 * 5591040 + 4 * (12343986 + 1)) * 4402 * 104426422404)
        (by norm_num) (by positivity)
        (mul_le_mul (mul_le_mul (by linarith) hVIR (Nat.cast_nonneg _) (by norm_num)) hcSR hc0 (by norm_num))
        (by norm_num))) ?_
    -- `2^1601` is past `exponentiation.threshold`, so the comparison is the kernel's, on `ℕ`.
    have hK : ((1235417791850338775791900 : ℕ) : ℝ) ≤ 2 ^ (((1601 : ℕ) : ℝ) / ((20 : ℕ) : ℝ)) :=
      le_two_rpow_of_pow_le (Nat.cast_nonneg _) (by norm_num) (cast_pow_le_two_pow (by decide +kernel))
    calc _ = ((1235417791850338775791900 : ℕ) : ℝ) * 2 ^ (-(255 : ℝ)) := by push_cast; ring
      _ ≤ 2 ^ (((1601 : ℕ) : ℝ) / ((20 : ℕ) : ℝ)) * 2 ^ (-(255 : ℝ)) := mul_le_mul_of_nonneg_right hK hp2
      _ = 2 ^ (-(174.95 : ℝ)) := e

/-- **`κ ≤ 2^{−167.7}` and `ε ≤ 2^{−174.95}` at a compiled record and an inner protocol**, from bounds on the numbers
they read. -/
theorem kappa_eps_of {C W' : Type} [Fintype C] [Nonempty C] [Inhabited C] (D : Claims.Recursion.Deploy C)
    (I : Claims.Recursion.Inner C W') (Cp : Claims.Recursion.Caps C) (x : Msg) (n v : ℕ)
    (hVI : I.V.k (Claims.Recursion.xI D x) ≤ 4402) (hkI : I.k (Claims.Recursion.xI D x) ≤ 12343986)
    (hwI : I.waits (Claims.Recursion.xI D x) = 210) (hW : Cp.width (Claims.Recursion.xI D x) ≤ 5591040)
    (hnS : D.nS x ≤ 16) (hSet : Claims.Recursion.setupLen D x ≤ 2 ^ 32) (hU : n ≤ 2 ^ 32) (hV : v ≤ 2 ^ 32)
    (hkx : D.V.k x ≤ 74835) (hκI : I.κ (Claims.Recursion.xI D x) ≤ 2 ^ (-(202.98 : ℝ)))
    (hκs : ∀ s : Fin (D.nS x), (D.nums x s).κ ≤ 2 ^ (-(202.98 : ℝ)))
    (hγs : ∀ s : Fin (D.nS x), (D.nums x s).γ ≤ 2 ^ 51) (hws : ∀ s : Fin (D.nS x), (D.nums x s).waits = 210)
    (hks : ∀ s : Fin (D.nS x), (D.nums x s).k ≤ 12343986) :
    Claims.Recursion.κ D I x ≤ 2 ^ (-(167.7 : ℝ)) ∧
      Claims.Recursion.epsLin2 D I Cp x n v ≤ 2 ^ (-(174.95 : ℝ)) :=
  kappa_eps_core (nS := D.nS x) (VI := I.V.k (Claims.Recursion.xI D x)) (kI := I.k (Claims.Recursion.xI D x))
    (wI := I.waits (Claims.Recursion.xI D x)) (W := Cp.width (Claims.Recursion.xI D x)) (kx := D.V.k x)
    (NI := Claims.Recursion.NI D I x) (cBe := Claims.Recursion.cBe D x) (nSet := Claims.Recursion.setupLen D x)
    (nU := n + v) (κI := I.κ (Claims.Recursion.xI D x)) (κs := fun s => (D.nums x s).κ)
    (γs := fun s => (D.nums x s).γ) (ws := fun s => (D.nums x s).waits) (ks := fun s => (D.nums x s).k) rfl rfl
    (by omega) hVI hkI hwI hW hnS hkx hκI hκs hγs hws hks

/-! ## At the deployment's sizes -/

section Part

variable {xO : Msg} {s : RecIns} {spec : _root_.Flock.SessionSpec} {sts : Array _root_.Flock.Stmt}

theorem numsOf_rounds_le_one (h : PartIn xO s spec sts) (h1 : sts.size = 1) : (numsOf xO).rounds ≤ 4402 := by
  have := VO_k_le h
  show VO.k xO ≤ 4402
  omega

/-- **A one-table part's `γ_O ≤ 2^51`**: at most `4402` rounds and a weight at most `2^33`. -/
theorem numsOf_γ_le_one (h : PartIn xO s spec sts) (h1 : sts.size = 1) : (numsOf xO).γ ≤ 2 ^ 51 := by
  have hr : ((numsOf xO).rounds : ℝ) ≤ 4402 := by exact_mod_cast numsOf_rounds_le_one h h1
  have hr0 : (0 : ℝ) ≤ (numsOf xO).rounds := Nat.cast_nonneg _
  have hC : (numsOf xO).weight ≤ 2 ^ 33 := by
    have h2 := weightO_le h
    rw [h1, Nat.cast_one, one_mul] at h2
    exact h2
  have hs : Real.sqrt (2 * (numsOf xO).rounds * (numsOf xO).weight) ≤ 9000000 := by
    rw [Real.sqrt_le_left (by norm_num)]
    nlinarith [mul_le_mul_of_nonneg_left hC hr0]
  have hsr : Real.sqrt (2 * (numsOf xO).rounds * (numsOf xO).weight) + (numsOf xO).rounds ≤ 9004402 := by linarith
  unfold Claims.Recursion.Nums.γ
  calc 16 * (Real.sqrt (2 * (numsOf xO).rounds * (numsOf xO).weight) + (numsOf xO).rounds) ^ 2
      ≤ 16 * 9004402 ^ 2 := by gcongr
    _ ≤ 2 ^ 51 := by norm_num

end Part

/-- **Route 2C's split bound at the deployment's sizes: `κ ≤ 2^{−167.7}` and `ε ≤ 2^{−174.95}`**, at the compiled
record whose inner view runs the deployed bound record's `IVStar` (`recBoundAt rel b L`, `bdDeployed`), the bound
record's inner Flock and the deployed caps `CpDep`, at every statement `x` whose inner statement is one plain table
(`hin`, `h1`), which stages at most `16` V\* parts (`hn`), each one `--zk` table (`hO`), whose inner setup has at most
`2^32` calls (`hS`), and at call bounds `n ≤ 2^32` and `v ≤ 2^32` (`hU`, `hV`). With `bitsLin2KOn_of_ksLin2On` it
makes `Pr[acc ∧ ¬R] ≤ 2^{−167.7} + t·2^{−174.95}`.

The size conditions are on public data and none is a hash premise or says anything about the prover's behaviour:
`h1`, `hO`, `hn` and `hS` are read off `x` by the verifier's own setups (`PlainIn`'s and `PartIn`'s `setupO`), and `n`
and `v` are the call bounds `BitsLin2KOn` quantifies over, which `hU` and `hV` cap (`v` at the deployed brings is at
most `nS x` times one statement's `deployedSetup` inputs, `setupCalls_deployedB_le`). -/
theorem kappa_eps_deployed (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings)
    (L : Claims.Recursion.VStarB.LayoutB) (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (decI : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace)
    {x : Msg} {s : FlockDeploy.Ins} {spec : _root_.Flock.SessionSpec} {sts : Array _root_.Flock.Stmt}
    (hin : FlockDeploy.PlainIn (xIOf x) s spec sts) (h1 : sts.size = 1)
    (hO : ∀ j < nS x, ∃ i spec' sts', PartIn (staged b.st x (pick b.st x j) j) i spec' sts' ∧ sts'.size = 1)
    (hn : nS x ≤ 16) (hS : (innerSetupIns (xIOf x)).length ≤ 2 ^ 32) {n : ℕ} (hU : n ≤ 2 ^ 32) {v : ℕ}
    (hV : v ≤ 2 ^ 32) :
    Claims.Recursion.κ ((recBoundAt rel b L).record (bdDeployed rel nextE colsE wE join rowIn final w).IVStar)
        (Claims.Recursion.Inner.ofFlock (bdDeployed rel nextE colsE wE join rowIn final w).deploy decI) x ≤
        2 ^ (-(167.7 : ℝ)) ∧
      Claims.Recursion.epsLin2
        ((recBoundAt rel b L).record (bdDeployed rel nextE colsE wE join rowIn final w).IVStar)
        (Claims.Recursion.Inner.ofFlock (bdDeployed rel nextE colsE wE join rowIn final w).deploy decI)
        FlockDeploy.KS.CpDep x n v ≤ 2 ^ (-(174.95 : ℝ)) := by
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
  refine kappa_eps_of ((recBoundAt rel b L).record (bdDeployed rel nextE colsE wE join rowIn final w).IVStar)
    (Claims.Recursion.Inner.ofFlock (bdDeployed rel nextE colsE wE join rowIn final w).deploy decI)
    FlockDeploy.KS.CpDep x n v hVI hkI rfl (CpDep_width_le_tight (xIOf x)) hn hS hU hV hkx hκI (fun j => ?_)
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

/-- **The deployment's sizes at a statement `x`**, `kappa_eps_deployed`'s side conditions on `x`: its inner statement
is one plain table, it stages at most `16` V\* parts, each one `--zk` table, and its inner setup has at most `2^32`
calls. -/
def SizesAt (b : Brings) (x : Msg) : Prop :=
  (∃ s spec sts, FlockDeploy.PlainIn (xIOf x) s spec sts ∧ sts.size = 1) ∧
    (∀ j < nS x, ∃ i spec' sts', PartIn (staged b.st x (pick b.st x j) j) i spec' sts' ∧ sts'.size = 1) ∧
    nS x ≤ 16 ∧ (innerSetupIns (xIOf x)).length ≤ 2 ^ 32

end RecDeploy
