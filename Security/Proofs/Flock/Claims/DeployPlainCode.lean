import Proofs.Flock.Claims.DecodeFlock
import Proofs.Flock.Claims.ZkDecodeFlock

/-!
# The deployed plain verifier's level-0 code at each `m`

The plain verifier serves `22 ≤ m ≤ 35` (`PlainIdx`), each at `fast100`'s schedule (`plainSched`), whose level 0 is a
code of `2^k0` lanes of `2^cols` columns at rate `1/2` (`plainShape_table`, `flockPlainLevelZero`). Read as one code
over every `m` (`IRS.famEnc`), it is the plain record's (`FlockDeploy.deployPlain`): its code (`plainEncD`), each
trace's codeword length `2^d` and radius `1 − √ρ − 1/200` at its `m` (`plainLenD`, `plainThetaD`), and its read step
(`plainReadD`). The radius decoder at it (`GS.plainDecodeRD`) reads only this file, so the record can name it.
-/

namespace Claims.Flock

open FlockSoundness FlockSoundness.Model

local notation "𝔽" => FlockLevel3.GF128

/-! ## The plain level-0 code at each `m` -/

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

/-- **The plain shapes**: `fast100` accepts every `m = 22 + i`; level 0 has `k0` lanes (log), `m − 7 − k0` columns
(log), at least `2^9`, rate `1/2` (`d = cols + 1`, `d ≤ 40`) and `218` queries. -/
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

/-- **The deployed plain code's Johnson count** at `agree`: `⌊(2^cols + 1)/agree²⌋ + 1` at the candidate's `m`. -/
noncomputable def plainNMinD (agree : ℝ) (M : PlainTrace) : ℕ :=
  IRS.famNMin flockPlainLevelZero agree M

theorem plainNMinD_eq (agree : ℝ) (M : PlainTrace) :
    plainNMinD agree M = johnsonCount (2 ^ plainCols M.1) agree := rfl

/-- **The deployed plain decoder**: each `m`'s level-0 decoder on the opened columns read at its width. -/
noncomputable def plainDecodeD (agree : ℝ) (os : List (ℕ × List 𝔽)) : List PlainTrace :=
  IRS.famDecode flockPlainLevelZero agree os

/-- **The deployed plain decoder meets `ListComplete`** at its Johnson count, at every `agree > 0`. -/
theorem listComplete_plainDecodeD (agree : ℝ) (ha : 0 < agree) :
    ListComplete plainEncD agree (plainNMinD agree) (plainDecodeD agree) :=
  IRS.listComplete_famDecode flockPlainLevelZero agree ha

/-- **The deployed plain read step**: the level-0 message's trace at the candidate's `m`, `2^m` hidden values. -/
noncomputable def plainReadD (M : PlainTrace) : Wit :=
  readTrace execArith M.1.m (plainLanes M.1) (plainCols M.1) M.2

theorem plainReadD_hid_length (M : PlainTrace) : (plainReadD M).hid.length = 2 ^ M.1.m :=
  readTrace_hid_length _ _ _ _ _

end Claims.Flock
