import Proofs.Flock.Claims.NumbersDeploy

/-!
# The radius `ε_table` is computed at, and the decoder's

A table's soundness theorem bounds the chance that the verifier accepts while no codeword within a proximity radius `γ`
of the committed level-0 table packs a satisfying witness, and `ε_table` (`FlockDeploy.εtable`) sums that bound over the
session's tables. The radius decoder (`Flock.ListWithin`) lists every trace within `θ = 1 − √ρ − 1/200` of the filled
word. Both count the positions, out of `2^d`, at which every lane they read agrees.

* Plain (`table_sound`, `Committed`): `γ` is `Level.radius` of `fast100`'s level 0 over its `2^k0` lanes, and that
  level has the plain code's lanes, columns and length (`plain_level₀`). So `γ = 1 − √(2^cols/2^d) − 1/200`, the
  decoder's `θ` (`plain_radius_eq`): `1 − √(1/2) − 1/200`, between `0.2878` and `0.2879`, at every `m`
  (`plain_radius_half`, `radius_half_bounds`).
* Masked (`table_sound_pad`, `CommittedPad`): `γ` is `Level.padRadius` of `m1Schedule`'s level 0 at padding `m1Pad`,
  and that level and padding are the masked code's columns, length and padding (`zk_level₀`). So
  `γ = 1 − √((2^cols + t)/2^d) − 1/200`, the decoder's `θ` (`zk_radius_eq`): at least `0.241` (least at `m = 25`) and
  below the plain radius at every `m` (`zk_radius_ge`, `zk_radius_lt_plain`). `CommittedPad` reads the `2^k0` witness
  lanes (`padWits`, `padList … 0 0`); the masked code's columns have `2^k0 + 4` (`zkWidth_eq`).
-/

namespace Claims.Flock.Numbers

open FlockSoundness FlockSoundness.Accounting

/-- A level's rate is its message length over its codeword length. -/
theorem level_rate_eq (l : Level) : l.rate = (2 : ℝ) ^ l.logCols / 2 ^ l.logLen := by
  rw [Level.rate, Level.logLen, pow_add]
  field_simp
  rw [← mul_pow]
  norm_num

theorem level_radius_eq (l : Level) :
    l.radius = 1 - Real.sqrt ((2 : ℝ) ^ l.logCols / 2 ^ l.logLen) - 1 / 200 := by
  rw [Level.radius, level_rate_eq, eta]

theorem level_padRadius_eq (l : Level) (t : ℕ) :
    l.padRadius t = 1 - Real.sqrt (((2 : ℝ) ^ l.logCols + t) / 2 ^ l.logLen) - 1 / 200 := by
  rw [Level.padRadius, Level.padRate, eta]
  push_cast
  rfl

/-! ## The plain code -/

/-- **The plain table's level 0 is the plain code's**: at every `m` the plain verifier serves, `fast100`'s level 0
has the plain code's lanes, columns and length, at rate `1/2`. -/
theorem plain_level₀ : ∀ i : PlainIdx,
    ((Accounting.fast100 i.m).getD default).k0 = plainLanes i ∧
      ((Accounting.fast100 i.m).getD default).level₀.logCols = plainCols i ∧
      ((Accounting.fast100 i.m).getD default).level₀.logLen = plainLen i ∧
      ((Accounting.fast100 i.m).getD default).level₀.logInvRate = 1 := by
  decide +kernel

/-- **The plain table's radius is the decoder's `θ`**: `table_sound`'s radius at the plain code's columns and length. -/
theorem plain_radius_eq (i : PlainIdx) :
    ((Accounting.fast100 i.m).getD default).level₀.radius =
      1 - Real.sqrt ((2 : ℝ) ^ plainCols i / 2 ^ plainLen i) - 1 / 200 := by
  obtain ⟨-, hc, hl, -⟩ := plain_level₀ i
  rw [level_radius_eq, hc, hl]

theorem plain_rate_half (i : PlainIdx) : (2 : ℝ) ^ plainCols i / 2 ^ plainLen i = 1 / 2 := by
  rw [(plainShape_table i).2.2.2.1, pow_succ]
  field_simp

theorem plain_radius_half (i : PlainIdx) :
    ((Accounting.fast100 i.m).getD default).level₀.radius = 1 - Real.sqrt (1 / 2) - 1 / 200 := by
  rw [plain_radius_eq, plain_rate_half]

theorem radius_half_bounds :
    (0.2878 : ℝ) < 1 - Real.sqrt (1 / 2) - 1 / 200 ∧ 1 - Real.sqrt (1 / 2) - 1 / 200 < 0.2879 := by
  have h1 : (0.7071 : ℝ) < Real.sqrt (1 / 2) := by
    rw [Real.lt_sqrt (by norm_num)]
    norm_num
  have h2 : Real.sqrt (1 / 2) < 0.7072 := by
    rw [Real.sqrt_lt' (by norm_num)]
    norm_num
  constructor <;> linarith

/-! ## The masked code -/

/-- **The masked table's level 0 is the masked code's**: at every `m` the masked verifier serves, `m1Schedule`'s level 0
and `m1Pad` have the masked code's lanes, columns, length, padding and level-0 query count. -/
theorem zk_level₀ : ∀ i : ZkIdx,
    (m1Schedule i.m).k0 = zkLanes i ∧ (m1Schedule i.m).level₀.logCols = zkCols i ∧
      (m1Schedule i.m).level₀.logLen = zkLen i ∧ m1Pad i.m = zkPad i ∧
      (m1Schedule i.m).level₀.queries = (zkLevel0 i.m).q := by
  decide +kernel

/-- **The masked table's radius is the decoder's `θ`**: `table_sound_pad`'s padded radius at the masked code's columns,
length and padding. -/
theorem zk_radius_eq (i : ZkIdx) :
    (m1Schedule i.m).level₀.padRadius (m1Pad i.m) =
      1 - Real.sqrt (((2 : ℝ) ^ zkCols i + zkPad i) / 2 ^ zkLen i) - 1 / 200 := by
  obtain ⟨-, hc, hl, ht, -⟩ := zk_level₀ i
  rw [level_padRadius_eq, hc, hl, ht]

/-- The masked code's columns have the table's `2^k0` witness lanes and four more. -/
theorem zkWidth_eq (i : ZkIdx) : zkWidth i = 2 ^ (m1Schedule i.m).k0 + 4 := by
  rw [(zk_level₀ i).1]
  rfl

theorem m1SqrtUp_le : ∀ m, 25 ≤ m → m ≤ 35 → m1SqrtUp m ≤ 377 / 500 := by
  intro m h1 h2
  interval_cases m <;> decide +kernel

theorem zk_radius_ge (i : ZkIdx) : (241 : ℝ) / 1000 ≤ (m1Schedule i.m).level₀.padRadius (m1Pad i.m) := by
  have h1 : 25 ≤ i.m := by unfold ZkIdx.m; omega
  have h2 : i.m ≤ 35 := by have := i.isLt; unfold ZkIdx.m; omega
  have hp := m1_padNumOK i.m h1 h2
  have hs : Real.sqrt ((m1Schedule i.m).level₀.padRate (m1Pad i.m)) ≤ (m1SqrtUp i.m : ℝ) :=
    Real.sqrt_le_iff.2 ⟨by exact_mod_cast hp.h0, hp.hs⟩
  have hu := (Rat.cast_le (K := ℝ)).2 (m1SqrtUp_le i.m h1 h2)
  push_cast at hu
  rw [Level.padRadius, eta]
  linarith

theorem zkPad_pos : ∀ i : ZkIdx, 1 ≤ zkPad i := by
  decide +kernel

theorem zk_radius_lt_plain (i : ZkIdx) :
    (m1Schedule i.m).level₀.padRadius (m1Pad i.m) < 1 - Real.sqrt (1 / 2) - 1 / 200 := by
  rw [zk_radius_eq]
  have ht : (1 : ℝ) ≤ zkPad i := by exact_mod_cast zkPad_pos i
  have hr : (1 : ℝ) / 2 < ((2 : ℝ) ^ zkCols i + zkPad i) / 2 ^ zkLen i := by
    rw [(zkShape_table i).2.2.2.1, pow_succ, lt_div_iff₀ (by positivity)]
    linarith
  have := Real.sqrt_lt_sqrt (by norm_num) hr
  linarith

end Claims.Flock.Numbers
