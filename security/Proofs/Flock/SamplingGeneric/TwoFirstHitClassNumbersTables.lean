module

public import Proofs.Flock.SamplingGeneric.TwoFirstHitClassNumbers

/-! Instances of `hashBound` from `2^39` classes at sessions of one table and of several: arithmetic only, none a premise of a guarantee. -/

@[expose] public section

namespace Claims.Flock.Numbers

theorem k_le_tables2 {N : ℝ} (hN : N ≤ 2 * 2 ^ 23) : ⌈4 * N / Real.exp 1⌉₊ ≤ 24687972 := by
  have he0 : 0 < Real.exp 1 := Real.exp_pos 1
  have hlo := Real.exp_one_gt_d9
  rw [Nat.ceil_le, div_le_iff₀ he0]
  push_cast
  linarith

theorem k_le_tables16 {N : ℝ} (hN : N ≤ 16 * 2 ^ 23) : ⌈4 * N / Real.exp 1⌉₊ ≤ 197503772 := by
  have he0 : 0 < Real.exp 1 := Real.exp_pos 1
  have hlo := Real.exp_one_gt_d9
  rw [Nat.ceil_le, div_le_iff₀ he0]
  push_cast
  linarith

theorem k_le_tables200 {N : ℝ} (hN : N ≤ 200 * 2 ^ 23) : ⌈4 * N / Real.exp 1⌉₊ ≤ 2468797139 := by
  have he0 : 0 < Real.exp 1 := Real.exp_pos 1
  have hlo := Real.exp_one_gt_d9
  rw [Nat.ceil_le, div_le_iff₀ he0]
  push_cast
  linarith

theorem k_tables_eq :
    ⌈4 * (2 * 2 ^ 23 : ℝ) / Real.exp 1⌉₊ = 24687972 ∧ ⌈4 * (16 * 2 ^ 23 : ℝ) / Real.exp 1⌉₊ = 197503772 ∧
      ⌈4 * (200 * 2 ^ 23 : ℝ) / Real.exp 1⌉₊ = 2468797139 :=
  ⟨ceil_div_exp_one_eq (n := 24687972) (by norm_num) (by norm_num) (by norm_num),
    ceil_div_exp_one_eq (n := 197503772) (by norm_num) (by norm_num) (by norm_num),
    ceil_div_exp_one_eq (n := 2468797139) (by norm_num) (by norm_num) (by norm_num)⟩

theorem reruns_tables_eq :
    1 + waits * 24687972 = 5184474121 ∧ 1 + waits * 197503772 = 41475792121 ∧
      1 + waits * 2468797139 = 518447399191 := by
  norm_num [waits]

theorem hashBound_table1_513_k39 :
    hashBound (513 * 2 ^ 80) 2592237061 (2 ^ 30) (2 ^ 39) (2 ^ 39) (513 * 2 ^ 80) ≤ 1 / 2 ^ 94 := by
  norm_num [hashBound, firstHitCalls]

theorem hashBound_tables2_513_k39 :
    hashBound (513 * 2 ^ 80) 5184474121 (2 ^ 30) (2 ^ 39) (2 ^ 39) (513 * 2 ^ 80) ≤ 1 / 2 ^ 93 := by
  norm_num [hashBound, firstHitCalls]

theorem hashBound_tables16_513_k39 :
    hashBound (513 * 2 ^ 80) 41475792121 (2 ^ 30) (2 ^ 39) (2 ^ 39) (513 * 2 ^ 80) ≤ 1 / 2 ^ 90 := by
  norm_num [hashBound, firstHitCalls]

theorem hashBound_tables200_513_k39 :
    hashBound (513 * 2 ^ 80) 518447399191 (2 ^ 30) (2 ^ 39) (2 ^ 39) (513 * 2 ^ 80) ≤ 1 / 2 ^ 87 := by
  norm_num [hashBound, firstHitCalls]

theorem hashBound_513_k39_gt :
    1 / 2 ^ 95 < hashBound (513 * 2 ^ 80) 2592237061 (2 ^ 30) (2 ^ 39) (2 ^ 39) (513 * 2 ^ 80) ∧
      1 / 2 ^ 94 < hashBound (513 * 2 ^ 80) 5184474121 (2 ^ 30) (2 ^ 39) (2 ^ 39) (513 * 2 ^ 80) ∧
      1 / 2 ^ 91 < hashBound (513 * 2 ^ 80) 41475792121 (2 ^ 30) (2 ^ 39) (2 ^ 39) (513 * 2 ^ 80) ∧
      1 / 2 ^ 88 < hashBound (513 * 2 ^ 80) 518447399191 (2 ^ 30) (2 ^ 39) (2 ^ 39) (513 * 2 ^ 80) := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;> norm_num [hashBound, firstHitCalls]

theorem hashBound_table1_513_k41 :
    hashBound (513 * 2 ^ 80) 2592237061 (2 ^ 30) (2 ^ 41) (2 ^ 41) (513 * 2 ^ 80) ≤ 1 / 2 ^ 92 := by
  norm_num [hashBound, firstHitCalls]

theorem hashBound_table1_513_k45 :
    hashBound (513 * 2 ^ 80) 2592237061 (2 ^ 30) (2 ^ 45) (2 ^ 45) (513 * 2 ^ 80) ≤ 1 / 2 ^ 88 := by
  norm_num [hashBound, firstHitCalls]

theorem hashBound_table1_513_k41_k45_gt :
    1 / 2 ^ 93 < hashBound (513 * 2 ^ 80) 2592237061 (2 ^ 30) (2 ^ 41) (2 ^ 41) (513 * 2 ^ 80) ∧
      1 / 2 ^ 89 < hashBound (513 * 2 ^ 80) 2592237061 (2 ^ 30) (2 ^ 45) (2 ^ 45) (513 * 2 ^ 80) := by
  refine ⟨?_, ?_⟩ <;> norm_num [hashBound, firstHitCalls]

end Claims.Flock.Numbers
