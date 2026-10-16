import Proofs.Flock.Claims.GS.Decode
import Proofs.Flock.Claims.DecodeFlock
import Proofs.Flock.Claims.RootFindGF128

/-!
# The counted decoder at Flock's level-0 code

`flockDecode h agree` is `gsDecode` at Flock's level-0 code in the executable's arithmetic (`flockLevelZero h`,
GF(2^128), `s = 2^k0` lanes of `k = 2^level₀.logCols` columns, for a `fast100` schedule), with

* the root finder `rf128`: Berlekamp's trace method over GF(2^128) (`Flock.roots128`, at most
  `Flock.roots128Cost = 64·129²·(d + 3)^4` operations on a list of length `d`, `Flock.roots128_complexity`);
* the reader `levelZeroReader`: a lane's message columns from its row polynomial's `k` coefficients, by the inverse of
  the novel basis's coefficient matrix (`novelInv`, the code's public data, fixed by `logCols`), `2k` operations per
  column, `2·s·k²` per message;
* one step per opened position for its point (`FlockLevel3.omega q` packs the low 64 bits of `q`).

`flockDecode_complexity`: at the input size `N·2^k0·128` (the opened columns' bits), the count is at most `flockPoly`
there:

`20·X^9 + 171·X^7 + 9·X^6 + 12·X^4 + X^2 + X + 1 + 3·64·129²·X^2·(3X + 3)^4 + 2·(X + 1)·X^2`.

`mem_flockDecode`: on opened columns at distinct positions of the codeword (below `2^level₀.logLen`, where the points
`ω_q` are distinct), from the Johnson count on, at `agree > 0`, every message agreeing with at least `agree·N` of them is
listed. `ListComplete` asks the same at every distinct position, and positions `q` and `q + 2^64` share their point: on
`nMin` columns at one point a decoder that meets it lists every message with those values there, and if its count bounds
its output's length its polynomial is at least their number, `2^(128·s·(k−1))`
(`Flock.ListComplete.card_le_eval_of_agree`).
-/

namespace Claims.GS

open Polynomial FlockSoundness FlockSoundness.Model FlockLevel3

theorem toPolyL_eq_toPoly {F : Type} [Field F] : ∀ cs : List F, toPolyL cs = RootFind.toPoly cs
  | [] => rfl
  | c :: cs => by rw [toPolyL, RootFind.toPoly_cons, toPolyL_eq_toPoly cs]

/-- **The root finder over GF(2^128)**, Berlekamp's trace method, as Roth–Ruckenstein's. -/
noncomputable def rf128 : RootFinder GF128 where
  run := Flock.roots128
  cost f := (Flock.roots128Counted f).2
  nodup cs h := (Flock.roots128_spec (by rwa [← toPolyL_eq_toPoly])).1
  mem_iff cs h a := by
    rw [toPolyL_eq_toPoly]
    exact (Flock.roots128_spec (by rwa [← toPolyL_eq_toPoly])).2 a
  bound := Flock.roots128Cost
  cost_le f := (Flock.roots128_complexity f).2

/-- A lane's row polynomial's first `2^cols` coefficients, linear in the lane. -/
noncomputable def novelCoeffs {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K]
    [DecidableEq K] (A : Arith F K) (hA : A.Correct) {cols : ℕ} (hcols : cols ≤ 40) :
    (Fin (2 ^ cols) → F) →ₗ[F] (Fin (2 ^ cols) → F) where
  toFun row i := (rowPoly A hA hcols row).coeff i
  map_add' a b := by
    funext i
    simp [rowPoly, Finset.sum_add_distrib, add_mul, finsetSum_coeff]
  map_smul' a b := by
    funext i
    simp [rowPoly, finsetSum_coeff, coeff_C_mul, Finset.mul_sum, mul_assoc]

theorem novelCoeffs_injective {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K]
    [DecidableEq K] (A : Arith F K) (hA : A.Correct) {cols : ℕ} (hcols : cols ≤ 40) :
    Function.Injective (novelCoeffs A hA hcols) := by
  intro a b h
  apply Flock.rowPoly_injective A hA hcols
  ext i
  by_cases hi : i < 2 ^ cols
  · exact congrFun h ⟨i, hi⟩
  · rw [coeff_eq_zero_of_natDegree_lt ((rowPoly_natDegree A hA hcols a).trans_le (not_lt.1 hi)),
      coeff_eq_zero_of_natDegree_lt ((rowPoly_natDegree A hA hcols b).trans_le (not_lt.1 hi))]

/-- The inverse of `novelCoeffs`: a lane from its row polynomial's coefficients. -/
noncomputable def novelInvEquiv {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K]
    [DecidableEq K] (A : Arith F K) (hA : A.Correct) {cols : ℕ} (hcols : cols ≤ 40) :
    (Fin (2 ^ cols) → F) ≃ₗ[F] (Fin (2 ^ cols) → F) :=
  (LinearEquiv.ofInjectiveEndo (novelCoeffs A hA hcols) (novelCoeffs_injective A hA hcols)).symm

/-- **The novel basis's inverse coefficient matrix**: column `c` of a lane is `Σ_i novelInv c i · a_i` on its row
polynomial's coefficients `a_i`. -/
noncomputable def novelInv {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K]
    [DecidableEq K] (A : Arith F K) (hA : A.Correct) {cols : ℕ} (hcols : cols ≤ 40) (c i : Fin (2 ^ cols)) : F :=
  novelInvEquiv A hA hcols (Pi.single i 1) c

/-- Read a message from its lanes' coefficient lists: per lane and column, `2^cols` products and their sum. -/
noncomputable def readLevelZero {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K]
    [DecidableEq K] (A : Arith F K) (hA : A.Correct) (lanes : ℕ) {cols : ℕ} (hcols : cols ≤ 40)
    (ps : List (List F)) : Message F lanes cols :=
  fun ℓ c => ∑ i : Fin (2 ^ cols), novelInv A hA hcols c i * (ps.getD ℓ []).getD i 0

theorem readLevelZero_eq {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K]
    [DecidableEq K] (A : Arith F K) (hA : A.Correct) (lanes : ℕ) {cols : ℕ} (hcols : cols ≤ 40)
    (ps : List (List F)) (ℓ : Fin (2 ^ lanes)) :
    readLevelZero A hA lanes hcols ps ℓ = novelInvEquiv A hA hcols fun i => (ps.getD ℓ []).getD i 0 := by
  set v : Fin (2 ^ cols) → F := fun i => (ps.getD ℓ []).getD i 0
  have hv : v = ∑ i, v i • Pi.single i (1 : F) := by
    conv_lhs => rw [← Finset.univ_sum_single v]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [← Pi.single_smul, smul_eq_mul, mul_one]
  funext c
  rw [hv, map_sum, Finset.sum_apply]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [map_smul, Pi.smul_apply, smul_eq_mul, mul_comm]
  rfl

theorem rowPoly_readLevelZero {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K]
    [DecidableEq K] (A : Arith F K) (hA : A.Correct) (lanes : ℕ) {cols : ℕ} (hcols : cols ≤ 40)
    (ps : List (List F)) (ℓ : Fin (2 ^ lanes)) (hlen : (ps.getD ℓ []).length = 2 ^ cols) :
    rowPoly A hA hcols (readLevelZero A hA lanes hcols ps ℓ) = toPolyL (ps.getD ℓ []) := by
  rw [readLevelZero_eq]
  ext i
  rw [coeff_toPolyL]
  by_cases hi : i < 2 ^ cols
  · have := congrFun (LinearEquiv.apply_symm_apply
      (LinearEquiv.ofInjectiveEndo (novelCoeffs A hA hcols) (novelCoeffs_injective A hA hcols))
      fun i => (ps.getD ℓ []).getD i 0) ⟨i, hi⟩
    rw [LinearEquiv.coe_ofInjectiveEndo] at this
    exact this
  · rw [coeff_eq_zero_of_natDegree_lt ((rowPoly_natDegree A hA hcols _).trans_le (not_lt.1 hi)),
      List.getD_eq_default _ _ (by omega)]

/-- **The level-0 reader**: `2·s·k²` operations, at most `2·(s·k)²`. -/
noncomputable def levelZeroReader {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K]
    [DecidableEq K] (A : Arith F K) (hA : A.Correct) (lanes cols : ℕ) (hcols : cols ≤ 40) :
    Reader (Flock.levelZero A hA lanes cols hcols) where
  read := readLevelZero A hA lanes hcols
  read_eq ps hl hk := by
    have hrows : (fun r : Fin (2 ^ lanes) => toPolyL (ps.getD r [])) =
        (Flock.levelZero A hA lanes cols hcols).rows (readLevelZero A hA lanes hcols ps) := by
      funext r
      refine (rowPoly_readLevelZero A hA lanes hcols ps r ?_).symm
      rw [List.getD_eq_getElem _ _ (by rw [hl]; exact r.isLt)]
      exact hk _ (List.getElem_mem _)
    rw [hrows, (Flock.levelZero A hA lanes cols hcols).ofRows_rows]
  cost := 2 * 2 ^ lanes * (2 ^ cols) ^ 2
  bound := C 2 * X ^ 2
  cost_le := by
    change 2 * 2 ^ lanes * (2 ^ cols) ^ 2 ≤ (C 2 * X ^ 2 : Polynomial ℕ).eval (2 ^ lanes * 2 ^ cols)
    simp only [eval_mul, eval_C, eval_pow, eval_X]
    have : 2 ^ lanes ≤ (2 ^ lanes) ^ 2 := Nat.le_self_pow (by norm_num) _
    calc 2 * 2 ^ lanes * (2 ^ cols) ^ 2 ≤ 2 * (2 ^ lanes) ^ 2 * (2 ^ cols) ^ 2 := by gcongr
      _ = 2 * (2 ^ lanes * 2 ^ cols) ^ 2 := by ring

/-- The level-0 reader at a `fast100` schedule. -/
noncomputable def flockReader {m : ℕ} {sch : Accounting.Schedule} (h : Accounting.fast100 m = some sch) :
    Reader (Flock.flockLevelZero h) :=
  levelZeroReader execArith execArith_correct sch.k0 sch.level₀.logCols (Flock.fast100_level₀_fits h).1

/-- **The counted decoder at Flock's level-0 code.** -/
noncomputable def flockDecode {m : ℕ} {sch : Accounting.Schedule} (h : Accounting.fast100 m = some sch)
    (agree : ℝ) : List (ℕ × (Fin (2 ^ sch.k0) → GF128)) → List (Message GF128 sch.k0 sch.level₀.logCols) :=
  gsDecode (Flock.flockLevelZero h) rf128 (flockReader h) agree

/-- `flockDecode` with its count, one step per point. -/
noncomputable def flockCounted {m : ℕ} {sch : Accounting.Schedule} (h : Accounting.fast100 m = some sch)
    (agree : ℝ) (os : List (ℕ × (Fin (2 ^ sch.k0) → GF128))) :
    List (Message GF128 sch.k0 sch.level₀.logCols) × ℕ :=
  gsCounted (Flock.flockLevelZero h) rf128 (flockReader h) agree 1 os

/-- **Flock's complexity polynomial**:
`20·X^9 + 171·X^7 + 9·X^6 + 12·X^4 + X^2 + X + 1 + 3·64·129²·X^2·(3X + 3)^4 + 2·(X + 1)·X^2`. -/
noncomputable def flockPoly : Polynomial ℕ :=
  C 20 * X ^ 9 + C 171 * X ^ 7 + C 9 * X ^ 6 + C 12 * X ^ 4 + X ^ 2 + X + 1 +
    C (3 * 64 * 129 * 129) * X ^ 2 * (C 3 * X + C 3) ^ 4 + C 2 * (X + 1) * X ^ 2

theorem gsBound_flock : gsBound Flock.roots128Cost (C 2 * X ^ 2) 1 = flockPoly := by
  simp only [gsBound, flockPoly, Flock.roots128Cost, mul_comp, pow_comp, add_comp, C_comp, X_comp, map_one,
    one_mul, C_mul]
  ring

/-- **Flock's level-0 decoder's complexity**: at most `flockPoly` at the opened columns' bits `N·2^k0·128`. -/
theorem flockDecode_complexity {m : ℕ} {sch : Accounting.Schedule} (h : Accounting.fast100 m = some sch)
    (agree : ℝ) :
    DecoderComplexity (flockDecode h agree) (flockCounted h agree) (fun os => os.length * 2 ^ sch.k0 * 128)
      flockPoly := by
  rw [← gsBound_flock]
  exact gsDecode_complexity (Flock.flockLevelZero h) rf128 (flockReader h) agree 1 128 (by norm_num)

/-- **Flock's level-0 decoder is complete on the codeword's positions**: on opened columns at distinct positions below
`2^level₀.logLen`, from the Johnson count on, at `agree > 0`, it lists every message whose level-0 encoding agrees with
at least `agree·N` of them. -/
theorem mem_flockDecode {m : ℕ} {sch : Accounting.Schedule} (h : Accounting.fast100 m = some sch) {agree : ℝ}
    (ha : 0 < agree) (os : List (ℕ × (Fin (2 ^ sch.k0) → GF128))) (hnd : (os.map Prod.fst).Nodup)
    (hlt : ∀ p ∈ os, p.1 < 2 ^ sch.level₀.logLen) (hN : johnsonCount (2 ^ sch.level₀.logCols) agree ≤ os.length)
    (M : Message GF128 sch.k0 sch.level₀.logCols)
    (hM : agree * os.length ≤
      (os.filter fun p => decide (Flock.levelZeroEnc execArith sch.k0 sch.level₀.logCols M p.1 = p.2)).length) :
    M ∈ flockDecode h agree os := by
  have hlt' : ∀ p ∈ os, p.1 < 2 ^ 64 := fun p hp =>
    lt_of_lt_of_le (hlt p hp) (Nat.pow_le_pow_right (by norm_num) (Flock.fast100_level₀_fits h).2)
  refine mem_gsDecode (Flock.flockLevelZero h) rf128 (flockReader h) ha os
    (Flock.levelZero_points_nodup execArith execArith_correct sch.k0 sch.level₀.logCols _ os hnd hlt') hN M ?_
  unfold IRS.agreeCount
  rw [Flock.flockLevelZero, Flock.levelZero_enc]
  exact hM

end Claims.GS
