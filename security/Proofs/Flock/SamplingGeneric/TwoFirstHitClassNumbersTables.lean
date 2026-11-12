module

public import Proofs.Flock.SamplingGeneric.TwoFirstHitClassNumbers

/-!
# The by-class first-hit bound's sum from `2^39` classes, at sessions of one table and of several

`hashBound` (`TwoFirstHitClassNumbers`) is the part over `2^256` of `CompiledTwo.εclsAt`, and
`CompiledTwo.eventTwoX_le_cls_of_hashBound` takes any bound of it at rational numbers. This file holds instances at
`2^39`, `2^41` and `2^45` in place of `K_s` and of `K_m`, past that file's largest (`2^37`). Each is at `N_s ≤ 2^30`
sessions with every call charged as `513` (`513·2^80` in place of a run's `2^80` calls and of its own opening
checks'), and each is the smallest power of two the sum is within (`hashBound_513_k39_gt`,
`hashBound_table1_513_k41_k45_gt`: the sum is above half of it):

* `hashBound_table1_513_k39`, at sessions of one table (`e_s ≤ 2592237061 = 1 + 210·12343986`): at most `2^-94`;
* `hashBound_tables2_513_k39`, `hashBound_tables16_513_k39` and `hashBound_tables200_513_k39`, at sessions of `2`, `16`
  and `200` tables: at most `2^-93`, `2^-90` and `2^-87`;
* `hashBound_table1_513_k41` and `hashBound_table1_513_k45`, at sessions of one table and `2^41` and `2^45`: at most
  `2^-92` and `2^-88`.

**The sessions' expected reruns at `n` tables.** `Compiled.es` is the largest of `E^rep`'s `1 + m′·k` at the proof
blocks' shapes, with `m′ = 210` (`waits`) and `k = ⌈4N₀/e⌉` at the level-0 symbols `N₀` of the block's session, summed
over the session's tables. A table has at most `2^23` of them, so a session of `n` tables has `N₀ ≤ n·2^23`
(`SpotFlock.ref_numbers`, at each template's reference statement, which Flock's masked scope holds to `1` to `200`
tables), and its `k` is at most `⌈4·n·2^23/e⌉`: `24687972`, `197503772` and `2468797139` at `2`, `16` and `200` tables
(`k_le_tables2`, `k_le_tables16`, `k_le_tables200`; each is that ceiling exactly, `k_tables_eq`). So `e_s` is at most
`5184474121`, `41475792121` and `518447399191` there (`reruns_tables_eq`). At one table `k` is
`SpotFlock.ref_one_table`'s `12343986`; `SpotFlock.κF_le` reads the same `k` at up to `200` tables through the coarser
`200·2^24` (`k_le_mul`).

Every statement here is arithmetic on the bound's expression. The two-stage verifier's code holds each template's
reference statement to one table (`SpotFlock.templateOk`, `SpotFlock.ref_one_table`), so the sessions of a run it
accepts have one table: no instance here says a verifier accepts a session of more, and none is a premise of a
guarantee.
-/

@[expose] public section

namespace Claims.Flock.Numbers

/-! ## A session's `k` at several tables -/

/-- `k` at `N₀ ≤ 2·2^23`, a session of two tables, is at most `24687972`. -/
theorem k_le_tables2 {N : ℝ} (hN : N ≤ 2 * 2 ^ 23) : ⌈4 * N / Real.exp 1⌉₊ ≤ 24687972 := by
  have he0 : 0 < Real.exp 1 := Real.exp_pos 1
  have hlo := Real.exp_one_gt_d9
  rw [Nat.ceil_le, div_le_iff₀ he0]
  push_cast
  linarith

/-- `k` at `N₀ ≤ 16·2^23`, a session of `16` tables, is at most `197503772`. -/
theorem k_le_tables16 {N : ℝ} (hN : N ≤ 16 * 2 ^ 23) : ⌈4 * N / Real.exp 1⌉₊ ≤ 197503772 := by
  have he0 : 0 < Real.exp 1 := Real.exp_pos 1
  have hlo := Real.exp_one_gt_d9
  rw [Nat.ceil_le, div_le_iff₀ he0]
  push_cast
  linarith

/-- `k` at `N₀ ≤ 200·2^23`, a session of `200` tables, is at most `2468797139`. -/
theorem k_le_tables200 {N : ℝ} (hN : N ≤ 200 * 2 ^ 23) : ⌈4 * N / Real.exp 1⌉₊ ≤ 2468797139 := by
  have he0 : 0 < Real.exp 1 := Real.exp_pos 1
  have hlo := Real.exp_one_gt_d9
  rw [Nat.ceil_le, div_le_iff₀ he0]
  push_cast
  linarith

/-- **Each is the ceiling exactly**: `⌈4·n·2^23/e⌉` at `n = 2`, `16` and `200`, from `e`'s nine-digit bounds. -/
theorem k_tables_eq :
    ⌈4 * (2 * 2 ^ 23 : ℝ) / Real.exp 1⌉₊ = 24687972 ∧ ⌈4 * (16 * 2 ^ 23 : ℝ) / Real.exp 1⌉₊ = 197503772 ∧
      ⌈4 * (200 * 2 ^ 23 : ℝ) / Real.exp 1⌉₊ = 2468797139 :=
  ⟨ceil_div_exp_one_eq (n := 24687972) (by norm_num) (by norm_num) (by norm_num),
    ceil_div_exp_one_eq (n := 197503772) (by norm_num) (by norm_num) (by norm_num),
    ceil_div_exp_one_eq (n := 2468797139) (by norm_num) (by norm_num) (by norm_num)⟩

/-- **The sessions' expected reruns at those `k`**: `E^rep`'s `1 + m′·k` at `m′ = 210` (`waits`). -/
theorem reruns_tables_eq :
    1 + waits * 24687972 = 5184474121 ∧ 1 + waits * 197503772 = 41475792121 ∧
      1 + waits * 2468797139 = 518447399191 := by
  norm_num [waits]

/-! ## The sum at `2^39` classes, and at `2^41` and `2^45` at sessions of one table -/

/-- **At most `2^-94` with every call charged as `513`, at `2^39`** in place of `K_s` and of `K_m`, at sessions of one
table: `e = 1 + 210·12343986` and `2^30` sessions. -/
theorem hashBound_table1_513_k39 :
    hashBound (513 * 2 ^ 80) 2592237061 (2 ^ 30) (2 ^ 39) (2 ^ 39) (513 * 2 ^ 80) ≤ 1 / 2 ^ 94 := by
  norm_num [hashBound, firstHitCalls]

/-- **At most `2^-93` at sessions of two tables**: the same sum at `e = 5184474121 = 1 + 210·24687972`. -/
theorem hashBound_tables2_513_k39 :
    hashBound (513 * 2 ^ 80) 5184474121 (2 ^ 30) (2 ^ 39) (2 ^ 39) (513 * 2 ^ 80) ≤ 1 / 2 ^ 93 := by
  norm_num [hashBound, firstHitCalls]

/-- **At most `2^-90` at sessions of `16` tables**: the same sum at `e = 41475792121 = 1 + 210·197503772`. -/
theorem hashBound_tables16_513_k39 :
    hashBound (513 * 2 ^ 80) 41475792121 (2 ^ 30) (2 ^ 39) (2 ^ 39) (513 * 2 ^ 80) ≤ 1 / 2 ^ 90 := by
  norm_num [hashBound, firstHitCalls]

/-- **At most `2^-87` at sessions of `200` tables**: the same sum at `e = 518447399191 = 1 + 210·2468797139`. -/
theorem hashBound_tables200_513_k39 :
    hashBound (513 * 2 ^ 80) 518447399191 (2 ^ 30) (2 ^ 39) (2 ^ 39) (513 * 2 ^ 80) ≤ 1 / 2 ^ 87 := by
  norm_num [hashBound, firstHitCalls]

/-- **Each of the four is the smallest power of two the sum is within**: the sum is above `2^-95` at one table, and
above `2^-94`, `2^-91` and `2^-88` at `2`, `16` and `200` tables. -/
theorem hashBound_513_k39_gt :
    1 / 2 ^ 95 < hashBound (513 * 2 ^ 80) 2592237061 (2 ^ 30) (2 ^ 39) (2 ^ 39) (513 * 2 ^ 80) ∧
      1 / 2 ^ 94 < hashBound (513 * 2 ^ 80) 5184474121 (2 ^ 30) (2 ^ 39) (2 ^ 39) (513 * 2 ^ 80) ∧
      1 / 2 ^ 91 < hashBound (513 * 2 ^ 80) 41475792121 (2 ^ 30) (2 ^ 39) (2 ^ 39) (513 * 2 ^ 80) ∧
      1 / 2 ^ 88 < hashBound (513 * 2 ^ 80) 518447399191 (2 ^ 30) (2 ^ 39) (2 ^ 39) (513 * 2 ^ 80) := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;> norm_num [hashBound, firstHitCalls]

/-- **At most `2^-92` at `2^41`** in place of `K_s` and of `K_m`, at sessions of one table: `hashBound_table1_513_k39`'s
sum with four times the classes. -/
theorem hashBound_table1_513_k41 :
    hashBound (513 * 2 ^ 80) 2592237061 (2 ^ 30) (2 ^ 41) (2 ^ 41) (513 * 2 ^ 80) ≤ 1 / 2 ^ 92 := by
  norm_num [hashBound, firstHitCalls]

/-- **At most `2^-88` at `2^45`** in place of `K_s` and of `K_m`, at sessions of one table. -/
theorem hashBound_table1_513_k45 :
    hashBound (513 * 2 ^ 80) 2592237061 (2 ^ 30) (2 ^ 45) (2 ^ 45) (513 * 2 ^ 80) ≤ 1 / 2 ^ 88 := by
  norm_num [hashBound, firstHitCalls]

/-- **Those two are the smallest powers of two as well**: the sum is above `2^-93` at `2^41` and above `2^-89` at
`2^45`. -/
theorem hashBound_table1_513_k41_k45_gt :
    1 / 2 ^ 93 < hashBound (513 * 2 ^ 80) 2592237061 (2 ^ 30) (2 ^ 41) (2 ^ 41) (513 * 2 ^ 80) ∧
      1 / 2 ^ 89 < hashBound (513 * 2 ^ 80) 2592237061 (2 ^ 30) (2 ^ 45) (2 ^ 45) (513 * 2 ^ 80) := by
  refine ⟨?_, ?_⟩ <;> norm_num [hashBound, firstHitCalls]

end Claims.Flock.Numbers
