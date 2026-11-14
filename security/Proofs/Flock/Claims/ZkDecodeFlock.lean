module

public import Proofs.Flock.Claims.ZkDecode
public import Proofs.Flock.Claims.DecodeFamily
public import Proofs.Flock.Soundness.Instance
public import Flock.Zk
import all Init.Data.Array.Basic
import all Init.Data.Range.Basic

/-!
# The deployed masked level-0 code, its decoder and read step

The masked verifier's level 0 at a statement version and `m` is `Zk.schedule`'s (`zkSched`, `zkLevel0`): the version's
schedule's level 0 (`fast100`'s, or the wide-leaf schedule's), `2^k0` witness lanes of `2^cols` columns on
`2^d = 2^{cols+1}` positions, with level 0's query count `q₀` and padding `t = 2·q₀`. The masked verifier serves
`25 ≤ m ≤ 35` at either version, and the two versions' schedules differ only from `m = 32` (`schedule_norm`), so its
tables are 15 (`ZkIdx`): today's version at `m = 25 + i`, then the wide-leaf version at `m = 32 … 35`; a statement's
is `ZkIdx.of` (`ZkIdx.of_at`). At each, its masked code is `zkLevelZero` in the executable's arithmetic (`execArith`,
GF(2^128)) on the `2^k0` witness lanes, with `c₀ = Zk.PAD_C` (`flockZkLevelZero`). The committed codeword also carries `Zk.EXTRA_LANES` lanes after the witness's, two per rep, each
rep's own; extraction needs only the witness lanes, so a masked column is an opened row's first `2^k0` symbols
(`zkRepColsOf`), the lanes the old `--zk` theorems bound closeness on (`padList` with no extra lanes):

| `m`      | 25   | 26   | 27   | 28   | 29   | 30   | 31   | 32..35     | 32..35, wide leaf |
|----------|------|------|------|------|------|------|------|------------|-------------------|
| lanes    | 64   | 64   | 64   | 16   | 32   | 64   | 64   | 64         | 2^7..2^10         |
| degree < | 4650 | 8676 | 16842| 131510 | 131510 | 131510 | 262582 | 2^cols + 436 | 262582     |
| `2^d`    | 2^13 | 2^14 | 2^15 | 2^18 | 2^18 | 2^18 | 2^19 | 2^20..2^23 | 2^19              |

(`zkShape_table`). The masked deployment's code is that family read as one (`IRS.famEnc`): a candidate is the
statement's table (its `ZkIdx`) and a masked message (`ZkTrace`), a column a list of GF(2^128) symbols (`zkEncD`). Its decoder
(`zkDecodeD`, `IRS.famDecode`) meets `ListComplete` at the candidate's Johnson count (`zkNMinD`,
`⌊(2^cols + t + 1)/agree²⌋ + 1`), at every `agree > 0` (`listComplete_zkDecodeD`); one run's two reps open fewer
level-0 columns than that (`zkQueries_lt_zkNMinD`). Its read step (`zkReadD`) is `zkRead` at the candidate's `m`: the
witness lanes' trace, `2^m` bits (`zkReadD_hid_length`).
-/

@[expose] public section

namespace Claims.Flock

open FlockSoundness FlockSoundness.Model

local notation "𝔽" => FlockLevel3.GF128

/-- **The masked verifier's schedule at `m`**: `Zk.schedule`'s schedule and padding `t = 2·q₀`; the default where the
verifier refuses `m`. -/
def zkSched (wideLeaf : Bool) (m : ℕ) : Flock.Schedule × ℕ :=
  match Flock.Zk.schedule wideLeaf m with
  | .ok p => p
  | .error _ => (default, 0)

/-- The masked verifier's level 0 at `m`. -/
def zkLevel0 (wideLeaf : Bool) (m : ℕ) : Flock.Level := (zkSched wideLeaf m).1.levels.getD 0 default

/-- **The two statement versions' schedules differ only from `m = 32`**: below it the wide-leaf schedule is
`fast100`. -/
theorem fast100At_of_le {m : ℕ} (h : m ≤ 31) : _root_.Flock.fast100At true m = _root_.Flock.fast100At false m := by
  show _root_.Flock.fast100Wide m = _root_.Flock.fast100 m
  unfold _root_.Flock.fast100Wide
  exact if_pos h

/-- Level 0's query count is one below `m = 32` too. -/
theorem queries0_of_le {m : ℕ} (h : m ≤ 31) : Flock.Zk.queries0 true m = Flock.Zk.queries0 false m := by
  unfold Flock.Zk.queries0
  split_ifs <;> first | rfl | omega

/-- **`Zk.schedule` reads the version only from `m = 32`**: a wide-leaf statement's table below it has today's
schedule and padding. -/
theorem schedule_norm (w : Bool) (m : ℕ) :
    Flock.Zk.schedule (w && decide (32 ≤ m)) m = Flock.Zk.schedule w m := by
  cases w
  · rfl
  · by_cases h : 32 ≤ m
    · rw [show (true && decide (32 ≤ m)) = true by simp [h]]
    · have hm : m ≤ 31 := by omega
      rw [show (true && decide (32 ≤ m)) = false by simp [h]]
      unfold Flock.Zk.schedule
      rw [queries0_of_le hm, fast100At_of_le hm]

theorem zkSched_norm (w : Bool) (m : ℕ) : zkSched (w && decide (32 ≤ m)) m = zkSched w m := by
  unfold zkSched
  rw [schedule_norm]

theorem zkLevel0_norm (w : Bool) (m : ℕ) : zkLevel0 (w && decide (32 ≤ m)) m = zkLevel0 w m := by
  unfold zkLevel0
  rw [zkSched_norm]

/-- The masked verifier's tables: today's version at `m = 25 + i` for `i < 11` (`Zk.queries0` refuses `m ≤ 24`, the
schedules `m > 35`), then the wide-leaf version at `m = 32 … 35` (`i = 11 … 14`), the sizes where its schedule is its
own. -/
abbrev ZkIdx := Fin 15

/-- The statement version at index `i`. -/
def ZkIdx.wideLeaf (i : ZkIdx) : Bool := decide (11 ≤ i.val)

/-- The statement's `m` at index `i`. -/
def ZkIdx.m (i : ZkIdx) : ℕ := if i.val < 11 then 25 + i.val else 21 + i.val

theorem ZkIdx.m_range : ∀ i : ZkIdx, 25 ≤ i.m ∧ i.m ≤ 35 := by
  decide +kernel

/-- The table of a statement of version `wideLeaf` at `m`: the wide-leaf version's from `m = 32`, where the two
versions' schedules differ; today's otherwise. -/
def ZkIdx.of (wideLeaf : Bool) (m : ℕ) : ZkIdx :=
  if wideLeaf && decide (32 ≤ m) then ⟨11 + (m - 32) % 4, by omega⟩ else ⟨(m - 25) % 11, by omega⟩

/-- **Index `i` is the table of a statement of version `wideLeaf` at `m`**: its `m`, and its version where the two
versions' schedules differ. -/
structure ZkIdx.At (i : ZkIdx) (wideLeaf : Bool) (m : ℕ) : Prop where
  eq_m : i.m = m
  eq_wideLeaf : i.wideLeaf = (wideLeaf && decide (32 ≤ m))

/-- **Every statement the masked verifier serves has its table**, at either version. -/
theorem ZkIdx.of_at (wideLeaf : Bool) {m : ℕ} (h1 : 25 ≤ m) (h2 : m ≤ 35) : (ZkIdx.of wideLeaf m).At wideLeaf m := by
  have key : ∀ (w : Bool) (k : Fin 11), (ZkIdx.of w (25 + k.val)).m = 25 + k.val ∧
      (ZkIdx.of w (25 + k.val)).wideLeaf = (w && decide (32 ≤ 25 + k.val)) := by
    decide +kernel
  obtain ⟨k, rfl⟩ : ∃ k : Fin 11, m = 25 + k.val := ⟨⟨m - 25, by omega⟩, by show m = 25 + (m - 25); omega⟩
  exact ⟨(key wideLeaf k).1, (key wideLeaf k).2⟩

namespace ZkIdx.At

variable {i : ZkIdx} {w : Bool} {m : ℕ}

/-- The index's schedule is the statement's. -/
theorem schedule (h : i.At w m) : Flock.Zk.schedule i.wideLeaf i.m = Flock.Zk.schedule w m := by
  rw [h.eq_wideLeaf, h.eq_m, schedule_norm]

theorem sched (h : i.At w m) : zkSched i.wideLeaf i.m = zkSched w m := by
  rw [h.eq_wideLeaf, h.eq_m, zkSched_norm]

theorem level0 (h : i.At w m) : zkLevel0 i.wideLeaf i.m = zkLevel0 w m := by
  rw [h.eq_wideLeaf, h.eq_m, zkLevel0_norm]

end ZkIdx.At

/-- Level 0's lanes (log), columns (log), padding and codeword length (log) at index `i`. -/
def zkLanes (i : ZkIdx) : ℕ := (zkLevel0 i.wideLeaf i.m).lanes
def zkCols (i : ZkIdx) : ℕ := (zkLevel0 i.wideLeaf i.m).cols
def zkPad (i : ZkIdx) : ℕ := (zkSched i.wideLeaf i.m).2
def zkLen (i : ZkIdx) : ℕ := (zkLevel0 i.wideLeaf i.m).d

/-- The masked code's lanes at index `i`: the witness's `2^k0`. -/
abbrev zkWidth (i : ZkIdx) : ℕ := 2 ^ zkLanes i

theorem zkShape_fits : ∀ i : ZkIdx, zkLen i ≤ 40 ∧ 2 ^ zkCols i + zkPad i ≤ 2 ^ zkLen i := by
  decide +kernel

/-- **The masked shapes**: `Zk.schedule` accepts every table; level 0 has `k0` lanes (log), `m − 7 − k0`
columns (log), rate `1/2` of the unpadded message (`d = cols + 1`), `q₀` queries, and padding `t = 2·q₀`. -/
theorem zkShape_table : ∀ i : ZkIdx,
    (Flock.Zk.schedule i.wideLeaf i.m).toBool = true ∧ zkLanes i = (zkSched i.wideLeaf i.m).1.k0 ∧ zkCols i + zkLanes i + 7 = i.m ∧
      zkLen i = zkCols i + 1 ∧ zkPad i = 2 * (zkLevel0 i.wideLeaf i.m).q ∧
      (Flock.Zk.queries0 i.wideLeaf i.m).toOption = some (zkLevel0 i.wideLeaf i.m).q := by
  decide +kernel

/-- **The deployed masked level-0 code at index `i`**: `zkLevelZero` in the executable's arithmetic, the `2^k0`
witness lanes of degree below `2^cols + t` at `2^d` positions, padded with `c₀ = Zk.PAD_C`. -/
noncomputable def flockZkLevelZero (i : ZkIdx) : IRS 𝔽 (ZkMessage 𝔽 (zkWidth i) (zkCols i) (zkPad i)) (zkWidth i) :=
  zkLevelZero execArith execArith_correct (zkWidth i) (zkCols i) (zkPad i) (zkLen i)
    (FlockLevel3.GF128.of Flock.Zk.PAD_C) (zkShape_fits i).1 (zkShape_fits i).2

/-- **A masked trace candidate**: the statement's table (its `ZkIdx`) and a message of its masked level-0 code. -/
abbrev ZkTrace := Σ i : ZkIdx, ZkMessage 𝔽 (zkWidth i) (zkCols i) (zkPad i)

/-- **The deployed masked code**: a candidate's column at `q`, its code's lanes as a list. -/
noncomputable def zkEncD (M : ZkTrace) (q : ℕ) : List 𝔽 :=
  IRS.famEnc flockZkLevelZero M q

theorem zkEncD_eq (M : ZkTrace) (q : ℕ) :
    zkEncD M q = List.ofFn (zkEnc execArith (zkLen M.1) (FlockLevel3.GF128.of Flock.Zk.PAD_C) M.2 q) := by
  unfold zkEncD IRS.famEnc flockZkLevelZero
  rw [zkLevelZero_enc]

/-- **The masked code's length** at a candidate: `2^d` at its `m`. -/
def zkLenD (M : ZkTrace) : ℕ := 2 ^ zkLen M.1

/-- **The masked code's radius** at a candidate: `1 − √ρ_zk − 1/200` at its `m`, `ρ_zk = (2^cols + t)/2^d`. -/
noncomputable def zkThetaD (M : ZkTrace) : ℝ :=
  1 - Real.sqrt (((2 : ℝ) ^ zkCols M.1 + zkPad M.1) / 2 ^ zkLen M.1) - 1 / 200

/-- **The deployed masked code's Johnson count** at `agree`: the candidate's code's,
`⌊(2^cols + t + 1)/agree²⌋ + 1` at its `m` (`zkNMinD_eq`). -/
noncomputable def zkNMinD (agree : ℝ) (M : ZkTrace) : ℕ :=
  IRS.famNMin flockZkLevelZero agree M

theorem zkNMinD_eq (agree : ℝ) (M : ZkTrace) :
    zkNMinD agree M = johnsonCount (2 ^ zkCols M.1 + zkPad M.1) agree := rfl

/-- **The deployed masked decoder**: each `m`'s masked decoder on the opened columns read at its width. -/
noncomputable def zkDecodeD (agree : ℝ) (os : List (ℕ × List 𝔽)) : List ZkTrace :=
  IRS.famDecode flockZkLevelZero agree os

/-- **The deployed masked decoder meets `ListComplete`** for the deployed masked code, at its Johnson count, at every
`agree > 0`. -/
theorem listComplete_zkDecodeD (agree : ℝ) (ha : 0 < agree) :
    ListComplete zkEncD agree (zkNMinD agree) (zkDecodeD agree) :=
  IRS.listComplete_famDecode flockZkLevelZero agree ha

/-- **One masked run opens fewer level-0 columns than the Johnson count**: at `0 < agree ≤ 1`, both reps' `2·q₀`
level-0 queries are below the candidate's count. -/
theorem zkQueries_lt_zkNMinD {agree : ℝ} (ha0 : 0 < agree) (ha1 : agree ≤ 1) (M : ZkTrace) :
    2 * (zkLevel0 M.1.wideLeaf M.1.m).q < zkNMinD agree M := by
  rw [zkNMinD_eq, ← (zkShape_table M.1).2.2.2.2.1]
  exact pad_lt_zkJohnsonCount ha0 ha1

/-- **The deployed masked read step**: `zkRead` at the candidate's `m`, every lane a witness lane. -/
noncomputable def zkReadD (M : ZkTrace) : Wit :=
  zkRead execArith M.1.m (zkLanes M.1) 0 (zkCols M.1) (zkPad M.1) M.2

/-- The read covers the statement's trace: `2^m` hidden values. -/
theorem zkReadD_hid_length (M : ZkTrace) : (zkReadD M).hid.length = 2 ^ M.1.m :=
  readTrace_hid_length _ _ _ _ _

end Claims.Flock
