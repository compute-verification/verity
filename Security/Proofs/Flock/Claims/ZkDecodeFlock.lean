import Proofs.Flock.Claims.ZkDecode
import Proofs.Flock.Claims.DecodeFamily
import Proofs.Flock.Soundness.Instance
import Flock.Zk

/-!
# The deployed masked level-0 code, its decoder and read step

The masked verifier's level 0 at `m` is `Zk.schedule m`'s (`zkSched`, `zkLevel0`): `fast100`'s level 0, `2^k0` witness
lanes of `2^cols` columns on `2^d = 2^{cols+1}` positions, with level 0's query count `q₀` and padding `t = 2·q₀`. At
the `m` the masked verifier serves, `25 ≤ m ≤ 35` (`ZkIdx`, `m = 25 + i`), its masked code is `zkLevelZero` in the
executable's arithmetic (`execArith`, GF(2^128)) with `2^k0 + Zk.EXTRA_LANES` lanes and `c₀ = Zk.PAD_C`
(`flockZkLevelZero`):

| `m`      | 25   | 26   | 27   | 28   | 29   | 30   | 31   | 32..35     |
|----------|------|------|------|------|------|------|------|------------|
| lanes    | 68   | 68   | 68   | 20   | 36   | 68   | 68   | 68         |
| degree < | 4650 | 8676 | 16842| 131510 | 131510 | 131510 | 262582 | 2^cols + 436 |
| `2^d`    | 2^13 | 2^14 | 2^15 | 2^18 | 2^18 | 2^18 | 2^19 | 2^20..2^23 |

(`zkShape_table`). The masked deployment's code is that family read as one (`IRS.famEnc`): a candidate is the
statement's `m` and a masked message (`ZkTrace`), a column a list of GF(2^128) symbols (`zkEncD`). Its decoder
(`zkDecodeD`, `IRS.famDecode`) meets `ListComplete` at every `agree` (`listComplete_zkDecodeD`), and its read step
(`zkReadD`) is `zkRead` at the candidate's `m`: the witness lanes' trace, `2^m` bits (`zkReadD_hid_length`).
-/

namespace Claims.Flock

open FlockSoundness FlockSoundness.Model

local notation "𝔽" => FlockLevel3.GF128

/-- **The masked verifier's schedule at `m`**: `Zk.schedule`'s schedule and padding `t = 2·q₀`; the default where the
verifier refuses `m`. -/
def zkSched (m : ℕ) : Flock.Schedule × ℕ :=
  match Flock.Zk.schedule m with
  | .ok p => p
  | .error _ => (default, 0)

/-- The masked verifier's level 0 at `m`. -/
def zkLevel0 (m : ℕ) : Flock.Level := (zkSched m).1.levels.getD 0 default

/-- The masked verifier's `m`, `25 + i`: `Zk.queries0` refuses `m ≤ 24`, `fast100` `m > 35`. -/
abbrev ZkIdx := Fin 11

/-- The statement's `m` at index `i`. -/
def ZkIdx.m (i : ZkIdx) : ℕ := 25 + i.val

/-- Level 0's lanes (log), columns (log), padding and codeword length (log) at index `i`. -/
def zkLanes (i : ZkIdx) : ℕ := (zkLevel0 i.m).lanes
def zkCols (i : ZkIdx) : ℕ := (zkLevel0 i.m).cols
def zkPad (i : ZkIdx) : ℕ := (zkSched i.m).2
def zkLen (i : ZkIdx) : ℕ := (zkLevel0 i.m).d

/-- The masked code's lanes at index `i`: the witness's `2^k0`, then the extra lanes. -/
abbrev zkWidth (i : ZkIdx) : ℕ := 2 ^ zkLanes i + Flock.Zk.EXTRA_LANES

theorem zkShape_fits : ∀ i : ZkIdx, zkLen i ≤ 40 ∧ 2 ^ zkCols i + zkPad i ≤ 2 ^ zkLen i := by
  decide +kernel

/-- **The masked shapes**: `Zk.schedule` accepts every `m = 25 + i`; level 0 has `k0` lanes (log), `m − 7 − k0`
columns (log), rate `1/2` of the unpadded message (`d = cols + 1`), `q₀` queries, and padding `t = 2·q₀`. -/
theorem zkShape_table : ∀ i : ZkIdx,
    (Flock.Zk.schedule i.m).toBool = true ∧ zkLanes i = (zkSched i.m).1.k0 ∧ zkCols i + zkLanes i + 7 = i.m ∧
      zkLen i = zkCols i + 1 ∧ zkPad i = 2 * (zkLevel0 i.m).q ∧
      (Flock.Zk.queries0 i.m).toOption = some (zkLevel0 i.m).q := by
  decide +kernel

/-- **The deployed masked level-0 code at index `i`**: `zkLevelZero` in the executable's arithmetic, `2^k0 + 4` lanes
of degree below `2^cols + t` at `2^d` positions, padded with `c₀ = Zk.PAD_C`. -/
noncomputable def flockZkLevelZero (i : ZkIdx) : IRS 𝔽 (ZkMessage 𝔽 (zkWidth i) (zkCols i) (zkPad i)) (zkWidth i) :=
  zkLevelZero execArith execArith_correct (zkWidth i) (zkCols i) (zkPad i) (zkLen i)
    (FlockLevel3.GF128.of Flock.Zk.PAD_C) (zkShape_fits i).1 (zkShape_fits i).2

/-- **A masked trace candidate**: the statement's `m` (as `ZkIdx`) and a message of its masked level-0 code. -/
abbrev ZkTrace := Σ i : ZkIdx, ZkMessage 𝔽 (zkWidth i) (zkCols i) (zkPad i)

/-- **The deployed masked code**: a candidate's column at `q`, its code's lanes as a list. -/
noncomputable def zkEncD (M : ZkTrace) (q : ℕ) : List 𝔽 :=
  IRS.famEnc flockZkLevelZero M q

theorem zkEncD_eq (M : ZkTrace) (q : ℕ) :
    zkEncD M q = List.ofFn (zkEnc execArith (zkLen M.1) (FlockLevel3.GF128.of Flock.Zk.PAD_C) M.2 q) := by
  unfold zkEncD IRS.famEnc flockZkLevelZero
  rw [zkLevelZero_enc]

/-- **The deployed masked decoder**: each `m`'s masked decoder on the opened columns of its width. -/
noncomputable def zkDecodeD (agree : ℝ) (os : List (ℕ × List 𝔽)) : List ZkTrace :=
  IRS.famDecode flockZkLevelZero agree os

/-- **The deployed masked decoder meets `ListComplete`** for the deployed masked code, at every `agree`. -/
theorem listComplete_zkDecodeD (agree : ℝ) : ListComplete zkEncD agree (zkDecodeD agree) :=
  IRS.listComplete_famDecode flockZkLevelZero agree

/-- **The deployed masked read step**: `zkRead` at the candidate's `m`. -/
noncomputable def zkReadD (M : ZkTrace) : Wit :=
  zkRead execArith M.1.m (zkLanes M.1) Flock.Zk.EXTRA_LANES (zkCols M.1) (zkPad M.1) M.2

/-- The read covers the statement's trace: `2^m` hidden values. -/
theorem zkReadD_hid_length (M : ZkTrace) : (zkReadD M).hid.length = 2 ^ M.1.m :=
  readTrace_hid_length _ _ _ _ _

/-- The witness lanes pack exactly the `2^m` trace bits: `128 · 2^k0 · 2^cols = 2^m`. -/
theorem zkPack_covers (i : ZkIdx) : 128 * 2 ^ zkLanes i * 2 ^ zkCols i = 2 ^ i.m := by
  have h := (zkShape_table i).2.2.1
  rw [← h, show (128 : ℕ) = 2 ^ 7 from rfl, ← pow_add, ← pow_add]
  congr 1
  omega

end Claims.Flock
