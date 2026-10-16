import Proofs.Flock.Claims.GS.Radius
import Proofs.Flock.Claims.ZkDecodeFlock

/-!
# The masked level-0 code in the radius form

The masked (`--zk`) verifier's level-0 code at each `m` it serves (`Flock.flockZkLevelZero i`, `m = 25 + i`): the
`s = 2^k0` witness lanes of degree below `k = 2^cols + t` (`t = 2·q₀`) on `n = 2^d` positions, at rate
`ρ_zk = (2^cols + t)/2^d`.

* `zkReader`: a masked message from its lanes' coefficient lists. Per lane, its `k` novel-basis coefficients by the
  inverse of `encPoly`'s coefficient matrix (`padInv`, the code's public data, fixed by `d` and `k`), `2k²` operations,
  then its padding `μ`, the last `t`, and its message `M = a − c₀·μ` on `[0, t)`, `a` above, `2t` more: `s·(2k² + 2t)`
  per message, at most `2·(s·k)² + 2·s·k`.
* `zkDecodeR i`: fill the word (`fillPts`), then the counted Guruswami–Sudan decoder (`gsDecode`, with `rf128` and
  `flockZkReader i`) on its `n` points at agreement `√ρ_zk + 1/200`.
* `zkJohnson_le_len`: the Johnson count fits the codeword at every `m` served, since `n·k > 10^4` there (`zkShape_nk`,
  smallest at `m = 25`: `n·k = 2^13·4650`).
* `listWithin_zkDecodeR`: it meets `ListWithin` at `θ = 1 − √ρ_zk − 1/200`.
* `zkDecodeR_complexity`: its count is at most `zkPolyR = zkPoly + X^2 + X` at the size `(os.length + n)·s·128`, with
  `zkPoly = gsBound roots128Cost (2·X^2 + 2·X) 1` (`gsBound_zk`).
-/

namespace Claims.GS

open Polynomial FlockSoundness FlockSoundness.Model FlockLevel3

/-! ## Reading a masked message from its lanes' coefficients -/

/-- A lane polynomial's first `k` coefficients, linear in its `k` novel-basis coefficients. -/
noncomputable def padCoeffs {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K]
    [DecidableEq K] (A : Arith F K) (hA : A.Correct) (d k : ℕ) (hd : d ≤ 40) (hk : k ≤ 2 ^ d) :
    (Fin k → F) →ₗ[F] (Fin k → F) where
  toFun a i := (encPoly A hA d k hd hk a).coeff i
  map_add' a b := by
    funext i
    simp [encPoly, Finset.sum_add_distrib, add_mul, finsetSum_coeff]
  map_smul' a b := by
    funext i
    simp [encPoly, finsetSum_coeff, coeff_C_mul, Finset.mul_sum, mul_assoc]

theorem padCoeffs_injective {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K]
    [DecidableEq K] (A : Arith F K) (hA : A.Correct) (d k : ℕ) (hd : d ≤ 40) (hk : k ≤ 2 ^ d) (hk0 : 0 < k) :
    Function.Injective (padCoeffs A hA d k hd hk) := by
  intro a b h
  apply Flock.encPoly_injective A hA d k hd hk
  ext i
  by_cases hi : i < k
  · exact congrFun h ⟨i, hi⟩
  · rw [coeff_eq_zero_of_natDegree_lt ((encPoly_natDegree A hA d k hd hk hk0 a).trans_le (not_lt.1 hi)),
      coeff_eq_zero_of_natDegree_lt ((encPoly_natDegree A hA d k hd hk hk0 b).trans_le (not_lt.1 hi))]

/-- The inverse of `padCoeffs`: a lane's novel-basis coefficients from its polynomial's. -/
noncomputable def padInvEquiv {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K]
    [DecidableEq K] (A : Arith F K) (hA : A.Correct) (d k : ℕ) (hd : d ≤ 40) (hk : k ≤ 2 ^ d) (hk0 : 0 < k) :
    (Fin k → F) ≃ₗ[F] (Fin k → F) :=
  (LinearEquiv.ofInjectiveEndo (padCoeffs A hA d k hd hk) (padCoeffs_injective A hA d k hd hk hk0)).symm

/-- **The inverse coefficient matrix of `encPoly`**: a lane's novel-basis coefficient `c` is `Σ_i padInv c i · a_i` on
its polynomial's coefficients `a_i`. -/
noncomputable def padInv {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K]
    [DecidableEq K] (A : Arith F K) (hA : A.Correct) (d k : ℕ) (hd : d ≤ 40) (hk : k ≤ 2 ^ d) (hk0 : 0 < k)
    (c i : Fin k) : F :=
  padInvEquiv A hA d k hd hk hk0 (Pi.single i 1) c

/-- A lane's novel-basis coefficients from its polynomial's coefficient list: per coefficient, `k` products and their
sum. -/
noncomputable def padLane {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K]
    [DecidableEq K] (A : Arith F K) (hA : A.Correct) (d k : ℕ) (hd : d ≤ 40) (hk : k ≤ 2 ^ d) (hk0 : 0 < k)
    (a : List F) : Fin k → F :=
  fun c => ∑ i : Fin k, padInv A hA d k hd hk hk0 c i * a.getD i 0

theorem padLane_eq {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K]
    [DecidableEq K] (A : Arith F K) (hA : A.Correct) (d k : ℕ) (hd : d ≤ 40) (hk : k ≤ 2 ^ d) (hk0 : 0 < k)
    (a : List F) : padLane A hA d k hd hk hk0 a = padInvEquiv A hA d k hd hk hk0 fun i => a.getD i 0 := by
  set v : Fin k → F := fun i => a.getD i 0
  have hv : v = ∑ i, v i • Pi.single i (1 : F) := by
    conv_lhs => rw [← Finset.univ_sum_single v]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [← Pi.single_smul, smul_eq_mul, mul_one]
  funext c
  rw [hv, map_sum, Finset.sum_apply]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [map_smul, Pi.smul_apply, smul_eq_mul, mul_comm]
  rfl

/-- On a coefficient list of length `k`, the lane read back has that list's polynomial. -/
theorem encPoly_padLane {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K]
    [DecidableEq K] (A : Arith F K) (hA : A.Correct) (d k : ℕ) (hd : d ≤ 40) (hk : k ≤ 2 ^ d) (hk0 : 0 < k)
    (a : List F) (hlen : a.length = k) : encPoly A hA d k hd hk (padLane A hA d k hd hk hk0 a) = toPolyL a := by
  rw [padLane_eq]
  ext i
  rw [coeff_toPolyL]
  by_cases hi : i < k
  · have := congrFun (LinearEquiv.apply_symm_apply
      (LinearEquiv.ofInjectiveEndo (padCoeffs A hA d k hd hk) (padCoeffs_injective A hA d k hd hk hk0))
      fun i => a.getD i 0) ⟨i, hi⟩
    rw [LinearEquiv.coe_ofInjectiveEndo] at this
    exact this
  · rw [coeff_eq_zero_of_natDegree_lt ((encPoly_natDegree A hA d k hd hk hk0 _).trans_le (not_lt.1 hi)),
      List.getD_eq_default _ _ (by omega)]

/-- **A masked lane's message and padding** from its `2^cols + t` novel-basis coefficients `a`: the padding `μ` is the
last `t`, the message `a − c₀·μ` on `[0, t)` and `a` above. -/
def zkUnpad {F : Type} [Field F] (c₀ : F) {cols t : ℕ} (a : Fin (2 ^ cols + t) → F) :
    (Fin (2 ^ cols) → F) × (Fin t → F) :=
  (fun c => a (Fin.castAdd t c) - if h : c.val < t then c₀ * a (Fin.natAdd (2 ^ cols) ⟨c.val, h⟩) else 0,
    fun j => a (Fin.natAdd (2 ^ cols) j))

theorem zkCoeffs_zkUnpad {F : Type} [Field F] (c₀ : F) {cols t : ℕ} (a : Fin (2 ^ cols + t) → F) :
    Flock.zkCoeffs c₀ (zkUnpad c₀ a).1 (zkUnpad c₀ a).2 = a := by
  funext x
  induction x using Fin.addCases with
  | left c =>
    rw [Flock.zkCoeffs_left]
    exact sub_add_cancel _ _
  | right j =>
    rw [Flock.zkCoeffs_right]
    rfl

/-- **Read a masked message** from its lanes' coefficient lists: each lane's novel-basis coefficients (`padLane`),
then its message and padding (`zkUnpad`). -/
noncomputable def readZk {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K]
    [DecidableEq K] (A : Arith F K) (hA : A.Correct) (s cols t d : ℕ) (c₀ : F) (hd : d ≤ 40)
    (hk : 2 ^ cols + t ≤ 2 ^ d) (ps : List (List F)) : Flock.ZkMessage F s cols t :=
  (fun ℓ => (zkUnpad c₀ (padLane A hA d _ hd hk (by positivity) (ps.getD ℓ []))).1,
    fun ℓ => (zkUnpad c₀ (padLane A hA d _ hd hk (by positivity) (ps.getD ℓ []))).2)

/-- **The masked reader**: `s·(2k² + 2t)` operations, at most `2·(s·k)² + 2·s·k`. -/
noncomputable def zkReader {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K]
    [DecidableEq K] (A : Arith F K) (hA : A.Correct) (s cols t d : ℕ) (c₀ : F) (hd : d ≤ 40)
    (hk : 2 ^ cols + t ≤ 2 ^ d) : Reader (Flock.zkLevelZero A hA s cols t d c₀ hd hk) where
  read := readZk A hA s cols t d c₀ hd hk
  read_eq ps hl hk' := by
    have hrows : (fun r : Fin s => toPolyL (ps.getD r [])) =
        (Flock.zkLevelZero A hA s cols t d c₀ hd hk).rows (readZk A hA s cols t d c₀ hd hk ps) := by
      funext r
      change toPolyL (ps.getD r []) = encPoly A hA d (2 ^ cols + t) hd hk
        (Flock.zkCoeffs c₀ (zkUnpad c₀ (padLane A hA d _ hd hk (by positivity) (ps.getD r []))).1
          (zkUnpad c₀ (padLane A hA d _ hd hk (by positivity) (ps.getD r []))).2)
      rw [zkCoeffs_zkUnpad]
      refine (encPoly_padLane A hA d _ hd hk _ _ ?_).symm
      rw [List.getD_eq_getElem _ _ (by rw [hl]; exact r.isLt)]
      exact hk' _ (List.getElem_mem _)
    rw [hrows, (Flock.zkLevelZero A hA s cols t d c₀ hd hk).ofRows_rows]
  cost := s * (2 * (2 ^ cols + t) ^ 2 + 2 * t)
  bound := C 2 * X ^ 2 + C 2 * X
  cost_le := by
    change s * (2 * (2 ^ cols + t) ^ 2 + 2 * t) ≤
      (C 2 * X ^ 2 + C 2 * X : Polynomial ℕ).eval (s * (2 ^ cols + t))
    simp only [eval_add, eval_mul, eval_C, eval_pow, eval_X]
    have hs : s ≤ s ^ 2 := Nat.le_self_pow (by norm_num) _
    have ht : t ≤ 2 ^ cols + t := Nat.le_add_left _ _
    calc s * (2 * (2 ^ cols + t) ^ 2 + 2 * t) = 2 * s * (2 ^ cols + t) ^ 2 + 2 * (s * t) := by ring
      _ ≤ 2 * s ^ 2 * (2 ^ cols + t) ^ 2 + 2 * (s * (2 ^ cols + t)) := by gcongr
      _ = 2 * (s * (2 ^ cols + t)) ^ 2 + 2 * (s * (2 ^ cols + t)) := by ring

/-! ## The deployed masked code in the radius form -/

/-- The masked reader at index `i` (`m = 25 + i`). -/
noncomputable def flockZkReader (i : Flock.ZkIdx) : Reader (Flock.flockZkLevelZero i) :=
  zkReader execArith execArith_correct (Flock.zkWidth i) (Flock.zkCols i) (Flock.zkPad i) (Flock.zkLen i)
    (GF128.of Flock.Zk.PAD_C) (Flock.zkShape_fits i).1 (Flock.zkShape_fits i).2

/-- **The masked radius decoder's agreement** `√ρ_zk + 1/200`, at the masked level-0 rate
`ρ_zk = (2^cols + t)/2^d`. -/
noncomputable def zkAgreeR (i : Flock.ZkIdx) : ℝ :=
  Real.sqrt (((2 : ℝ) ^ Flock.zkCols i + Flock.zkPad i) / 2 ^ Flock.zkLen i) + 1 / 200

theorem zkAgreeR_pos (i : Flock.ZkIdx) : 0 < zkAgreeR i := by
  unfold zkAgreeR
  positivity

/-- Every masked level 0 the verifier serves has `n·k > 10^4`, `n = 2^d` and `k = 2^cols + t`: the least is `m = 25`'s,
`2^13·4650`. -/
theorem zkShape_nk : ∀ i : Flock.ZkIdx, 10000 < 2 ^ Flock.zkLen i * (2 ^ Flock.zkCols i + Flock.zkPad i) := by
  decide +kernel

/-- **The Johnson count fits the masked codeword**: at agreement `√ρ_zk + 1/200` the count of `2^cols + t` is at most
`n = 2^d`, at every `m` the masked verifier serves. -/
theorem zkJohnson_le_len (i : Flock.ZkIdx) :
    johnsonCount (2 ^ Flock.zkCols i + Flock.zkPad i) (zkAgreeR i) ≤ 2 ^ Flock.zkLen i := by
  have := johnsonCount_le_len (k := 2 ^ Flock.zkCols i + Flock.zkPad i) (n := 2 ^ Flock.zkLen i) (by positivity)
    (zkShape_nk i)
  push_cast at this
  exact this

/-- **The masked radius decoder**: the zero-filled word of length `n = 2^d`, then the counted Guruswami–Sudan decoder
on its `n` points at agreement `√ρ_zk + 1/200`. -/
noncomputable def zkDecodeR (i : Flock.ZkIdx) (os : List (ℕ × (Fin (Flock.zkWidth i) → GF128))) :
    List (Flock.ZkMessage GF128 (Flock.zkWidth i) (Flock.zkCols i) (Flock.zkPad i)) :=
  gsDecode (Flock.flockZkLevelZero i) rf128 (flockZkReader i) (zkAgreeR i) (fillPts (2 ^ Flock.zkLen i) os)

/-- `zkDecodeR` with its count: the fill step's (`fillPtsC`), then the decoder's on the `n` points, one step per
point. -/
noncomputable def zkCountedR (i : Flock.ZkIdx) (os : List (ℕ × (Fin (Flock.zkWidth i) → GF128))) :
    List (Flock.ZkMessage GF128 (Flock.zkWidth i) (Flock.zkCols i) (Flock.zkPad i)) × ℕ :=
  ((gsCounted (Flock.flockZkLevelZero i) rf128 (flockZkReader i) (zkAgreeR i) 1
      (fillPtsC (2 ^ Flock.zkLen i) os).1).1,
    (fillPtsC (2 ^ Flock.zkLen i) os).2 +
      (gsCounted (Flock.flockZkLevelZero i) rf128 (flockZkReader i) (zkAgreeR i) 1
        (fillPtsC (2 ^ Flock.zkLen i) os).1).2)

/-- **The masked decoder's complexity polynomial** on its points, `gsBound` at `rf128`'s bound, the masked reader's
`2·X^2 + 2·X` and one step per point:
`20·X^9 + 171·X^7 + 9·X^6 + 12·X^4 + X^2 + X + 1 + 3·64·129²·X^2·(3X + 3)^4 + 2·(X + 1)·(X^2 + X)`. -/
noncomputable def zkPoly : Polynomial ℕ :=
  C 20 * X ^ 9 + C 171 * X ^ 7 + C 9 * X ^ 6 + C 12 * X ^ 4 + X ^ 2 + X + 1 +
    C (3 * 64 * 129 * 129) * X ^ 2 * (C 3 * X + C 3) ^ 4 + C 2 * (X + 1) * (X ^ 2 + X)

theorem gsBound_zk : gsBound Flock.roots128Cost (C 2 * X ^ 2 + C 2 * X) 1 = zkPoly := by
  simp only [gsBound, zkPoly, Flock.roots128Cost, mul_comp, pow_comp, add_comp, C_comp, X_comp, map_one,
    one_mul, C_mul]
  ring

/-- **The masked radius decoder's complexity polynomial**: `zkPoly + X^2 + X`, the fill step's `X^2 + X` added. -/
noncomputable def zkPolyR : Polynomial ℕ := zkPoly + X ^ 2 + X

/-- **The masked radius decoder meets `ListWithin`** at the masked level-0 code, at `θ = 1 − √ρ_zk − 1/200`, at every
`m` the masked verifier serves. -/
theorem listWithin_zkDecodeR (i : Flock.ZkIdx) :
    Flock.ListWithin (fun _ => 2 ^ Flock.zkLen i)
      (Flock.zkEnc execArith (Flock.zkLen i) (GF128.of Flock.Zk.PAD_C) (s := Flock.zkWidth i)
        (cols := Flock.zkCols i) (t := Flock.zkPad i))
      (fun _ => 1 - Real.sqrt (((2 : ℝ) ^ Flock.zkCols i + Flock.zkPad i) / 2 ^ Flock.zkLen i) - 1 / 200)
      (zkDecodeR i) := by
  intro os M hM
  have hlt : ∀ p ∈ fillPts (2 ^ Flock.zkLen i) os, p.1 < 2 ^ 64 := fun p hp =>
    lt_of_lt_of_le (lt_of_mem_fillPts hp)
      (Nat.pow_le_pow_right (by norm_num) ((Flock.zkShape_fits i).1.trans (by norm_num)))
  refine mem_gsDecode (Flock.flockZkLevelZero i) rf128 (flockZkReader i) (zkAgreeR_pos i) _
    (Flock.zkLevelZero_points_nodup _ _ _ _ _ _ _ _ _ _ (nodup_fillPts _ os) hlt)
    (by rw [length_fillPts]; exact zkJohnson_le_len i) M ?_
  unfold IRS.agreeCount
  rw [Flock.flockZkLevelZero, Flock.zkLevelZero_enc, length_fillPts, length_filter_fillPts]
  have e : zkAgreeR i =
      1 - (1 - Real.sqrt (((2 : ℝ) ^ Flock.zkCols i + Flock.zkPad i) / 2 ^ Flock.zkLen i) - 1 / 200) := by
    unfold zkAgreeR
    ring
  rw [e]
  exact hM

/-- **The masked radius decoder's complexity**: at most `zkPolyR` at the size `(os.length + n)·s·128`. -/
theorem zkDecodeR_complexity (i : Flock.ZkIdx) :
    DecoderComplexity (zkDecodeR i) (zkCountedR i)
      (fun os => (os.length + 2 ^ Flock.zkLen i) * Flock.zkWidth i * 128) zkPolyR := by
  intro os
  have hc := gsDecode_complexity (Flock.flockZkLevelZero i) rf128 (flockZkReader i) (zkAgreeR i) 1 128
    (by norm_num) (fillPts (2 ^ Flock.zkLen i) os)
  have hb : gsBound rf128.bound (flockZkReader i).bound 1 = zkPoly := gsBound_zk
  rw [hb] at hc
  simp only [zkCountedR, zkDecodeR, fillPtsC_fst]
  refine ⟨hc.1, ?_⟩
  have hf := fillPtsC_snd_le (2 ^ Flock.zkLen i) os
  have hg := hc.2
  simp only [length_fillPts] at hg
  simp only [zkPolyR, eval_add, eval_pow, eval_X]
  generalize hQ : (os.length + 2 ^ Flock.zkLen i) * Flock.zkWidth i * 128 = Q
  have hw : 1 ≤ Flock.zkWidth i * 128 := Nat.one_le_iff_ne_zero.2 (by positivity)
  have hnQ : 2 ^ Flock.zkLen i * Flock.zkWidth i * 128 ≤ Q := by
    rw [← hQ, mul_assoc, mul_assoc]
    exact Nat.mul_le_mul_right _ (Nat.le_add_left _ _)
  have hn : 2 ^ Flock.zkLen i ≤ Q := le_trans (by rw [mul_assoc]; exact Nat.le_mul_of_pos_right _ hw) hnQ
  have hL : os.length ≤ Q := by
    rw [← hQ, mul_assoc]
    exact le_trans (Nat.le_add_right _ _) (Nat.le_mul_of_pos_right _ hw)
  have hg' := hg.trans (eval_mono_nat zkPoly hnQ)
  have hf' : 2 ^ Flock.zkLen i * (os.length + 1) ≤ Q ^ 2 + Q := by
    calc 2 ^ Flock.zkLen i * (os.length + 1) ≤ Q * (Q + 1) := Nat.mul_le_mul hn (by omega)
      _ = Q ^ 2 + Q := by ring
  omega

end Claims.GS
