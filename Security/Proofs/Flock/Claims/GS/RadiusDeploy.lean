import Proofs.Flock.Claims.GS.RadiusFamily
import Proofs.Flock.Claims.DeployCode
import Proofs.Flock.Claims.ERepCost

/-!
# The radius decoders at Flock's deployed instances

`FlockDeploy.deployPlain rel` and `FlockDeploy.deployZk rel` fix the column type `List GF128`, the traces
(`Flock.PlainTrace`, `m = 22..35`, and `Flock.ZkTrace`, `m = 25..35`) and their code (`enc`: `Flock.plainEncD`,
`Flock.zkEncD`), and each trace's codeword length and radius (`len`, `θ`), read off the trace's `m`
(`Flock.plainLenD`, `Flock.plainThetaD`, `Flock.zkLenD`, `Flock.zkThetaD`).

* `plainDecodeR i`: at the plain level-0 code at `m = 22 + i` (`Flock.flockPlainLevelZero i`, rate `ρ = 2^cols/2^d`),
  fill the word, then `gsDecode` with `rf128` and `levelZeroReader` on its `n = 2^d` points at agreement
  `√ρ + 1/200`. It meets `ListWithin` at `θ = 1 − √ρ − 1/200` (`listWithin_plainDecodeR`; the Johnson count fits,
  `plainShape_nk`, least `n·k = 2^10·2^9` at `m = 22`), with `DecoderComplexity` `flockPolyR` at
  `(os.length + n)·2^k0·128` (`plainDecodeR_complexity`).
* `plainDecodeRD`, the plain family (`famDecodeR`): `ListWithin` at `deployPlain rel`'s `len`, `enc` and `θ`
  (`listWithin_deployPlain`), with `DecoderComplexity` `14·(flockPolyR + X)` at the schedules' sizes summed
  (`plainSizeRD`, `plainDecodeRD_complexity`), its lists at most that long (`plainDecodeRD_length_le`).
* `zkDecodeRD`, the masked family: `ListWithin` at `deployZk rel`'s `len`, `enc` and `θ` (`listWithin_deployZk`), with
  `DecoderComplexity` `11·(zkPolyR + X)` at `zkSizeRD` (`zkDecodeRD_complexity`), its lists at most that long
  (`zkDecodeRD_length_le`).

`radius_deployPlain` and `radius_deployZk` state each record's pair, `ListWithin` and `DecoderComplexity`;
`efficient_deployPlain` and `efficient_deployZk` are `Claims.Flock.Efficient` at each record and its decoder.
-/

namespace Claims.GS

open Polynomial FlockSoundness FlockSoundness.Model FlockLevel3

/-! ## The plain level-0 code at each `m` in the radius form -/

/-- The plain reader at index `i` (`m = 22 + i`). -/
noncomputable def plainReader (i : Flock.PlainIdx) : Reader (Flock.flockPlainLevelZero i) :=
  levelZeroReader execArith execArith_correct (Flock.plainLanes i) (Flock.plainCols i) (Flock.plainCols_le i)

/-- **The plain radius decoder's agreement** `√ρ + 1/200`, at the plain level-0 rate `ρ = 2^cols/2^d`. -/
noncomputable def plainAgreeR (i : Flock.PlainIdx) : ℝ :=
  Real.sqrt ((2 : ℝ) ^ Flock.plainCols i / 2 ^ Flock.plainLen i) + 1 / 200

theorem plainAgreeR_pos (i : Flock.PlainIdx) : 0 < plainAgreeR i := by
  unfold plainAgreeR
  positivity

/-- Every plain level 0 the verifier serves has `n·k > 10^4`, `n = 2^d` and `k = 2^cols`: the least is `m = 22`'s,
`2^10·2^9`. -/
theorem plainShape_nk : ∀ i : Flock.PlainIdx, 10000 < 2 ^ Flock.plainLen i * 2 ^ Flock.plainCols i := by
  decide +kernel

/-- **The Johnson count fits the plain codeword**: at agreement `√ρ + 1/200` the count of `2^cols` is at most
`n = 2^d`, at every `m` the plain verifier serves. -/
theorem plainJohnson_le_len (i : Flock.PlainIdx) :
    johnsonCount (2 ^ Flock.plainCols i) (plainAgreeR i) ≤ 2 ^ Flock.plainLen i := by
  have := johnsonCount_le_len (k := 2 ^ Flock.plainCols i) (n := 2 ^ Flock.plainLen i) (by positivity)
    (plainShape_nk i)
  push_cast at this
  exact this

/-- **The plain radius decoder** at `m = 22 + i`: the zero-filled word of length `n = 2^d`, then the counted
Guruswami–Sudan decoder on its `n` points at agreement `√ρ + 1/200`. -/
noncomputable def plainDecodeR (i : Flock.PlainIdx) (os : List (ℕ × (Fin (2 ^ Flock.plainLanes i) → GF128))) :
    List (Message GF128 (Flock.plainLanes i) (Flock.plainCols i)) :=
  gsDecode (Flock.flockPlainLevelZero i) rf128 (plainReader i) (plainAgreeR i) (fillPts (2 ^ Flock.plainLen i) os)

/-- `plainDecodeR` with its count: the fill step's (`fillPtsC`), then the decoder's on the `n` points, one step per
point. -/
noncomputable def plainCountedR (i : Flock.PlainIdx) (os : List (ℕ × (Fin (2 ^ Flock.plainLanes i) → GF128))) :
    List (Message GF128 (Flock.plainLanes i) (Flock.plainCols i)) × ℕ :=
  ((gsCounted (Flock.flockPlainLevelZero i) rf128 (plainReader i) (plainAgreeR i) 1
      (fillPtsC (2 ^ Flock.plainLen i) os).1).1,
    (fillPtsC (2 ^ Flock.plainLen i) os).2 +
      (gsCounted (Flock.flockPlainLevelZero i) rf128 (plainReader i) (plainAgreeR i) 1
        (fillPtsC (2 ^ Flock.plainLen i) os).1).2)

/-- **The plain radius decoder meets `ListWithin`** at the plain level-0 code, at `θ = 1 − √ρ − 1/200`, at every `m`
the plain verifier serves. -/
theorem listWithin_plainDecodeR (i : Flock.PlainIdx) :
    Flock.ListWithin (fun _ => 2 ^ Flock.plainLen i)
      (Flock.levelZeroEnc execArith (Flock.plainLanes i) (Flock.plainCols i))
      (fun _ => 1 - Real.sqrt ((2 : ℝ) ^ Flock.plainCols i / 2 ^ Flock.plainLen i) - 1 / 200)
      (plainDecodeR i) := by
  intro os M hM
  have hlt : ∀ p ∈ fillPts (2 ^ Flock.plainLen i) os, p.1 < 2 ^ 64 := fun p hp =>
    lt_of_lt_of_le (lt_of_mem_fillPts hp)
      (Nat.pow_le_pow_right (by norm_num) ((Flock.plainShape_table i).2.2.2.2.1.trans (by norm_num)))
  refine mem_gsDecode (Flock.flockPlainLevelZero i) rf128 (plainReader i) (plainAgreeR_pos i) _
    (Flock.levelZero_points_nodup execArith execArith_correct _ _ _ _ (nodup_fillPts _ os) hlt)
    (by rw [length_fillPts]; exact plainJohnson_le_len i) M ?_
  unfold IRS.agreeCount
  rw [Flock.flockPlainLevelZero, Flock.levelZero_enc, length_fillPts, length_filter_fillPts]
  have e : plainAgreeR i =
      1 - (1 - Real.sqrt ((2 : ℝ) ^ Flock.plainCols i / 2 ^ Flock.plainLen i) - 1 / 200) := by
    unfold plainAgreeR
    ring
  rw [e]
  exact hM

/-- **The plain radius decoder's complexity**: at most `flockPolyR` at the size `(os.length + n)·2^k0·128`. -/
theorem plainDecodeR_complexity (i : Flock.PlainIdx) :
    DecoderComplexity (plainDecodeR i) (plainCountedR i)
      (fun os => (os.length + 2 ^ Flock.plainLen i) * 2 ^ Flock.plainLanes i * 128) flockPolyR := by
  intro os
  have hc := gsDecode_complexity (Flock.flockPlainLevelZero i) rf128 (plainReader i) (plainAgreeR i) 1 128
    (by norm_num) (fillPts (2 ^ Flock.plainLen i) os)
  have hb : gsBound rf128.bound (plainReader i).bound 1 = flockPoly := gsBound_flock
  rw [hb] at hc
  simp only [plainCountedR, plainDecodeR, fillPtsC_fst]
  refine ⟨hc.1, ?_⟩
  have hf := fillPtsC_snd_le (2 ^ Flock.plainLen i) os
  have hg := hc.2
  simp only [length_fillPts] at hg
  simp only [flockPolyR, eval_add, eval_pow, eval_X]
  generalize hQ : (os.length + 2 ^ Flock.plainLen i) * 2 ^ Flock.plainLanes i * 128 = Q
  have hw : 1 ≤ 2 ^ Flock.plainLanes i * 128 := Nat.one_le_iff_ne_zero.2 (by positivity)
  have hnQ : 2 ^ Flock.plainLen i * 2 ^ Flock.plainLanes i * 128 ≤ Q := by
    rw [← hQ, mul_assoc, mul_assoc]
    exact Nat.mul_le_mul_right _ (Nat.le_add_left _ _)
  have hn : 2 ^ Flock.plainLen i ≤ Q := le_trans (by rw [mul_assoc]; exact Nat.le_mul_of_pos_right _ hw) hnQ
  have hL : os.length ≤ Q := by
    rw [← hQ, mul_assoc]
    exact le_trans (Nat.le_add_right _ _) (Nat.le_mul_of_pos_right _ hw)
  have hg' := hg.trans (eval_mono_nat flockPoly hnQ)
  have hf' : 2 ^ Flock.plainLen i * (os.length + 1) ≤ Q ^ 2 + Q := by
    calc 2 ^ Flock.plainLen i * (os.length + 1) ≤ Q * (Q + 1) := Nat.mul_le_mul hn (by omega)
      _ = Q ^ 2 + Q := by ring
  omega

/-! ## The plain verifier's radius decoder -/

/-- **The plain verifier's radius decoder**: each `m`'s plain radius decoder (`plainDecodeR`) on the opened columns
read at its width. -/
noncomputable def plainDecodeRD (os : List (ℕ × List GF128)) : List Flock.PlainTrace :=
  famDecodeR plainDecodeR os

/-- `plainDecodeRD` with its count. -/
noncomputable def plainCountedRD (os : List (ℕ × List GF128)) : List Flock.PlainTrace × ℕ :=
  famCountedR plainCountedR os

/-- **The plain family's size**: its schedules' sizes summed, `Σ_i (os.length + 2^{d_i})·2^{k0_i}·128`. -/
def plainSizeRD (os : List (ℕ × List GF128)) : ℕ :=
  ∑ i : Flock.PlainIdx, (os.length + 2 ^ Flock.plainLen i) * 2 ^ Flock.plainLanes i * 128

/-- **The plain verifier's radius decoder meets `ListWithin`** at the deployed plain code, with the length `2^d` and
the radius `1 − √ρ − 1/200` read off the trace's `m`. -/
theorem listWithin_plainDecodeRD :
    Flock.ListWithin Flock.plainLenD Flock.plainEncD Flock.plainThetaD plainDecodeRD :=
  listWithin_famDecodeR (s := fun i => 2 ^ Flock.plainLanes i)
    (T := fun i => Message GF128 (Flock.plainLanes i) (Flock.plainCols i))
    (fun i _ => 2 ^ Flock.plainLen i) (fun i => (Flock.flockPlainLevelZero i).enc)
    (fun i _ => 1 - Real.sqrt ((2 : ℝ) ^ Flock.plainCols i / 2 ^ Flock.plainLen i) - 1 / 200) plainDecodeR
    fun i => by
      rw [Flock.flockPlainLevelZero, Flock.levelZero_enc]
      exact listWithin_plainDecodeR i

/-- **The plain verifier's radius decoder's complexity**: at most `14·(flockPolyR + X)` at the schedules' sizes
summed. -/
theorem plainDecodeRD_complexity :
    DecoderComplexity plainDecodeRD plainCountedRD plainSizeRD (C 14 * (flockPolyR + Polynomial.X)) := by
  have hle : ∀ (i : Flock.PlainIdx) (os : List (ℕ × List GF128)),
      (os.length + 2 ^ Flock.plainLen i) * 2 ^ Flock.plainLanes i * 128 ≤ plainSizeRD os := fun i os =>
    Finset.single_le_sum
      (f := fun i : Flock.PlainIdx => (os.length + 2 ^ Flock.plainLen i) * 2 ^ Flock.plainLanes i * 128)
      (fun _ _ => Nat.zero_le _) (Finset.mem_univ i)
  have := famDecodeR_complexity plainDecodeR plainCountedR
    (fun i os => (os.length + 2 ^ Flock.plainLen i) * 2 ^ Flock.plainLanes i * 128) flockPolyR
    plainDecodeR_complexity plainSizeRD
    (fun i os => by
      rw [IRS.length_colsAt]
      exact hle i os)
    (fun i os => le_trans (calc os.length * 2 ^ Flock.plainLanes i
        _ ≤ (os.length + 2 ^ Flock.plainLen i) * 2 ^ Flock.plainLanes i :=
          Nat.mul_le_mul_right _ (Nat.le_add_right _ _)
        _ ≤ (os.length + 2 ^ Flock.plainLen i) * 2 ^ Flock.plainLanes i * 128 :=
          Nat.le_mul_of_pos_right _ (by norm_num))
      (hle i os))
  rw [Fintype.card_fin] at this
  exact this

/-- **The plain verifier's radius decoder lists at most `14·(flockPolyR + X)` at `plainSizeRD`**: each `m`'s decoder
lists at most `2^d`, one per point of the filled word (`length_gsDecode_le`), and `2^d` is within the size. -/
theorem plainDecodeRD_length_le (os : List (ℕ × List GF128)) :
    (plainDecodeRD os).length ≤ (C 14 * (flockPolyR + Polynomial.X)).eval (plainSizeRD os) := by
  have hi : ∀ i : Flock.PlainIdx,
      (plainDecodeR i (IRS.colsAt (2 ^ Flock.plainLanes i) os)).length ≤ plainSizeRD os := by
    intro i
    unfold plainDecodeR
    refine (length_gsDecode_le _ _ _ _ _).trans ?_
    rw [length_fillPts, max_eq_right Nat.one_le_two_pow]
    refine le_trans ?_ (Finset.single_le_sum
      (f := fun i : Flock.PlainIdx => (os.length + 2 ^ Flock.plainLen i) * 2 ^ Flock.plainLanes i * 128)
      (fun _ _ => Nat.zero_le _) (Finset.mem_univ i))
    have hw : 1 ≤ 2 ^ Flock.plainLanes i * 128 := Nat.one_le_iff_ne_zero.2 (by positivity)
    calc 2 ^ Flock.plainLen i ≤ os.length + 2 ^ Flock.plainLen i := Nat.le_add_left _ _
      _ ≤ (os.length + 2 ^ Flock.plainLen i) * (2 ^ Flock.plainLanes i * 128) := Nat.le_mul_of_pos_right _ hw
      _ = (os.length + 2 ^ Flock.plainLen i) * 2 ^ Flock.plainLanes i * 128 := by ring
  refine (length_famDecodeR_le plainDecodeR os _ hi).trans ?_
  simp only [Fintype.card_fin, eval_mul, eval_C, eval_add, eval_X]
  exact Nat.mul_le_mul_left 14 (Nat.le_add_left _ _)

/-! ## At the deployed instances -/

/-- **The plain verifier's radius decoder meets `ListWithin`** at the deployed plain instance's code, length and
radius: the spec `Flock.KS` and `Flock.Bits` ask of a decoder at that instance. -/
theorem listWithin_deployPlain (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) :
    Flock.ListWithin (FlockDeploy.deployPlain rel).len (FlockDeploy.deployPlain rel).enc
      (FlockDeploy.deployPlain rel).θ plainDecodeRD :=
  listWithin_plainDecodeRD

/-- **The masked verifier's radius decoder meets `ListWithin`** at the deployed masked instance's code, length and
radius. -/
theorem listWithin_deployZk (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) :
    Flock.ListWithin (FlockDeploy.deployZk rel).len (FlockDeploy.deployZk rel).enc (FlockDeploy.deployZk rel).θ
      zkDecodeRD :=
  listWithin_zkDecodeRD

/-- **The plain instance's radius decoder**: `ListWithin` at its code, and `DecoderComplexity`. -/
theorem radius_deployPlain (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) :
    Flock.ListWithin (FlockDeploy.deployPlain rel).len (FlockDeploy.deployPlain rel).enc
        (FlockDeploy.deployPlain rel).θ plainDecodeRD ∧
      DecoderComplexity plainDecodeRD plainCountedRD plainSizeRD (C 14 * (flockPolyR + Polynomial.X)) :=
  ⟨listWithin_deployPlain rel, plainDecodeRD_complexity⟩

/-- **The masked instance's radius decoder**: `ListWithin` at its code, and `DecoderComplexity`. -/
theorem radius_deployZk (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) :
    Flock.ListWithin (FlockDeploy.deployZk rel).len (FlockDeploy.deployZk rel).enc (FlockDeploy.deployZk rel).θ
        zkDecodeRD ∧
      DecoderComplexity zkDecodeRD zkCountedRD zkSizeRD (C 11 * (zkPolyR + Polynomial.X)) :=
  ⟨listWithin_deployZk rel, zkDecodeRD_complexity⟩

/-- **Flock's `Efficient` at the plain instance**: `E^rep`'s reruns, the radius decoder's `ListWithin` and
`DecoderComplexity`, and its lists, which bound `assemble`'s checks, at most `14·(flockPolyR + X)` at `plainSizeRD`. -/
theorem efficient_deployPlain (H : Msg → Msg) (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) :
    Flock.Efficient H (FlockDeploy.deployPlain rel) plainDecodeRD plainCountedRD plainSizeRD
      (C 14 * (flockPolyR + Polynomial.X)) :=
  Flock.Efficient_of_decoder H _ _ _ _ _ (listWithin_deployPlain rel) plainDecodeRD_complexity plainDecodeRD_length_le

/-- **Flock's `Efficient` at the masked instance**: `E^rep`'s reruns, the radius decoder's `ListWithin` and
`DecoderComplexity`, and its lists, which bound `assemble`'s checks, at most `11·(zkPolyR + X)` at `zkSizeRD`. -/
theorem efficient_deployZk (H : Msg → Msg) (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) :
    Flock.Efficient H (FlockDeploy.deployZk rel) zkDecodeRD zkCountedRD zkSizeRD (C 11 * (zkPolyR + Polynomial.X)) :=
  Flock.Efficient_of_decoder H _ _ _ _ _ (listWithin_deployZk rel) zkDecodeRD_complexity zkDecodeRD_length_le

end Claims.GS
