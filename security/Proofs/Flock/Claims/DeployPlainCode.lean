module

public import Proofs.Flock.Claims.DecodeFlock
public import Proofs.Flock.Claims.ZkDecodeFlock
import all Init.Data.Array.Basic
import all Init.Data.Range.Basic

/-! The deployed plain verifier's level-0 code at each `m` it serves, as one code. -/

@[expose] public section

namespace Claims.Flock

open FlockSoundness FlockSoundness.Model

local notation "𝔽" => FlockLevel3.GF128

/-- **The plain verifier's schedule at `m`**: `fast100`'s; the default where the verifier refuses `m`. -/
def plainSched (m : ℕ) : _root_.Flock.Schedule := (_root_.Flock.fast100 m).getD default

/-- The plain verifier's level 0 at `m`. -/
def plainLevel0 (m : ℕ) : _root_.Flock.Level := (plainSched m).levels.getD 0 default

/-- The plain verifier's `m`, `22 + i`. -/
abbrev PlainIdx := Fin 14

/-- The statement's `m` at index `i`. -/
def PlainIdx.m (i : PlainIdx) : ℕ := 22 + i.val

/-- Level 0's lanes (log), columns (log) and codeword length (log) at index `i`. -/
def plainLanes (i : PlainIdx) : ℕ := (plainLevel0 i.m).lanes
def plainCols (i : PlainIdx) : ℕ := (plainLevel0 i.m).cols
def plainLen (i : PlainIdx) : ℕ := (plainLevel0 i.m).d

theorem plainShape_table : ∀ i : PlainIdx,
    (_root_.Flock.fast100 i.m).isSome = true ∧ plainLanes i = (plainSched i.m).k0 ∧ plainCols i + plainLanes i + 7 = i.m ∧
      plainLen i = plainCols i + 1 ∧ plainLen i ≤ 40 ∧ 9 ≤ plainCols i ∧ (plainLevel0 i.m).q = 218 := by
  decide +kernel

theorem plainCols_le (i : PlainIdx) : plainCols i ≤ 40 := by
  have := plainShape_table i
  omega

/-- **The deployed plain level-0 code at index `i`**: `levelZero` in the executable's arithmetic, `2^k0` lanes
of `2^cols` columns. -/
noncomputable def flockPlainLevelZero (i : PlainIdx) :
    IRS 𝔽 (Message 𝔽 (plainLanes i) (plainCols i)) (2 ^ plainLanes i) :=
  levelZero execArith execArith_correct (plainLanes i) (plainCols i) (plainCols_le i)

/-- **A plain trace candidate**: the statement's `m` (as `PlainIdx`) and a message of its level-0 code. -/
abbrev PlainTrace := Σ i : PlainIdx, Message 𝔽 (plainLanes i) (plainCols i)

/-- **The deployed plain code**: a candidate's column at `q`, its code's lanes as a list. -/
noncomputable def plainEncD (M : PlainTrace) (q : ℕ) : List 𝔽 :=
  IRS.famEnc flockPlainLevelZero M q

theorem plainEncD_eq (M : PlainTrace) (q : ℕ) :
    plainEncD M q = List.ofFn (levelZeroEnc execArith (plainLanes M.1) (plainCols M.1) M.2 q) := by
  unfold plainEncD IRS.famEnc flockPlainLevelZero
  rw [levelZero_enc]

/-- **The plain code's length** at a candidate: `2^d` at its `m`. -/
def plainLenD (M : PlainTrace) : ℕ := 2 ^ plainLen M.1

/-- **The plain code's radius** at a candidate: `1 − √ρ − 1/200` at its `m`, `ρ = 2^cols/2^d`. -/
noncomputable def plainThetaD (M : PlainTrace) : ℝ :=
  1 - Real.sqrt ((2 : ℝ) ^ plainCols M.1 / 2 ^ plainLen M.1) - 1 / 200

noncomputable def plainNMinD (agree : ℝ) (M : PlainTrace) : ℕ :=
  IRS.famNMin flockPlainLevelZero agree M

theorem plainNMinD_eq (agree : ℝ) (M : PlainTrace) :
    plainNMinD agree M = johnsonCount (2 ^ plainCols M.1) agree := rfl

noncomputable def plainDecodeD (agree : ℝ) (os : List (ℕ × List 𝔽)) : List PlainTrace :=
  IRS.famDecode flockPlainLevelZero agree os

theorem listComplete_plainDecodeD (agree : ℝ) (ha : 0 < agree) :
    ListComplete plainEncD agree (plainNMinD agree) (plainDecodeD agree) :=
  IRS.listComplete_famDecode flockPlainLevelZero agree ha

/-- **The deployed plain read step**: the level-0 message's trace at the candidate's `m`, `2^m` hidden values. -/
noncomputable def plainReadD (M : PlainTrace) : Wit :=
  readTrace execArith M.1.m (plainLanes M.1) (plainCols M.1) M.2

end Claims.Flock
