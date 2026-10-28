import Proofs.Flock.Claims.KS.ZkClashRaw
import Proofs.Flock.Claims.KS.ZkAbortJR
import Proofs.Flock.Claims.KS.ZkSlackFit
import Proofs.Flock.Claims.KS.ForkCoupling
import Proofs.Flock.Claims.KS.PlainWire
import Proofs.Flock.Claims.KS.Rounds
import Proofs.Flock.Claims.KS.RoundsNZk
import Proofs.Flock.Claims.KS.ZkBoundJ
import Proofs.Flock.VBridge.Family.Lig

/-!
# The masked simulator

`zkSim P x spec sts` is a live prover at the masked wire (`zkWire`) that runs `P` round by round, at every table. Its
tape is a uniform function per deployed round of the live coins' revealed history (`zkTape`), and the deployed coins it
feeds `P` are `fnC` of that tape and the live coins' revealed entries (`zkSimCoins`), which are `zkΦ`'s coins (`fnAt`
at the history key `FnFork.histKey`) wherever the history is the live session's. Deployed round `i`'s stream `k` is
table `k / 2`'s rep `k % 2` (`zkStream`), its count the session's (`zkCnt`, from `zkNs`), and its message `P`'s with
the stream and count after tag `2` (`zkRound`, read back by `zkSub`).

* **The fork rounds** (`zkRd`): table `j`'s rep `r` starts `B_r` rounds in (`zkBase`, the reps in `spec.repStreams`
  order), its pads at its first round, its level-`ℓ + 1` cap after the masked replay's and the ring switch's rounds
  (`zkPre`) and two lane-phase rounds at Ligerito's `capRound`, its `τ` at its third inner round, and level 0 at the
  link coin: no two of a table's at one round (`zkRd_inj`), only level 0's at round `0` (`zkRd_zero`), each before
  the last (`zkRd_succ_lt`).
* **The slack** (`zk_simFork_le`): the simulated fork at `zkΦ` is at most the deployed fork and the slack at the chances
  `δN GF128 65536 (zkNOf …)` (`forkCoinCoupling_zk`, `slack_of_coupling`), which fit the masked weight's room at
  every table (`zk_slackFitW_δN`).
* **What is left**, four named Props of the deployed verifier and the simulator: the lockstep (`ZkSimRunAt`), the
  abort (`ZkSimAbortAt`, at `ZkSimAbort`), the model's rewinding advantage at the simulator against the simulated fork
  (`ZkSimForkLe`, at `zkAdvSim`), and a self-clash at an accepting run's view opening two columns of a table at one
  position or colliding in the run's log (`ZkViewClash`).

From those, `ZkSimWJ` (`zkSimWJ_of_parts`), so `Flock.KS` at the trimmed masked record (`zk_KS_of_parts`), every
table's bound with escapes on accepted runs counted per round (`zk_tableBoundJOnAccR_of_parts`), and the recursion's KS
over the masked record (`zk_KSFlock_of_parts`).
-/

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkSession
open FlockSoundness.Discharge
open FlockSoundness.Discharge.Tables (LiveOutJ liveSession DecodesTable SubmitJ)
open FlockSoundness.Discharge.Composed (outJ repJ)
open Claims.Bridge.Live

local notation "𝔽" => FlockLevel3.GF128

/-! ## The setup's numbers -/

namespace Gen

variable [S : ZkSetup]

/-- `V true`'s rounds at a statement that sets up: the link message, the session's rounds and the last message. -/
theorem zkK {x : Claims.Msg} {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf true x = some (ins, spec, sts)) : (V true).k x = 2 + rounds true spec sts := by
  simp only [V, hset]

end Gen

/-- `V true`'s rounds at a statement that sets up: the link message, the session's rounds and the last message. -/
theorem zkK {x : Claims.Msg} {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf true x = some (ins, spec, sts)) : (V true).k x = 2 + rounds true spec sts :=
  Gen.zkK (S := S₀) (x := x) (ins := ins) (spec := spec) (sts := sts) (hset := hset)

namespace Gen

variable [S : ZkSetup]

theorem zk_tables_length {x : Claims.Msg} {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf true x = some (ins, spec, sts)) : (tables true x).length = sts.size := by
  unfold tables
  rw [hset]
  simp

end Gen

theorem zk_tables_length {x : Claims.Msg} {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf true x = some (ins, spec, sts)) : (tables true x).length = sts.size :=
  Gen.zk_tables_length (S := S₀) (x := x) (ins := ins) (spec := spec) (sts := sts) (hset := hset)

/-- **A table's setup passed the range check, with well-formed regions**: `setupH`'s and `setupHidden`'s
`checkInRange` at its `k_log`, regions and link points, and `RegionsWF`. -/
theorem zk_setupOne_range {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tbls : List (String × ByteArray)}
    {draw partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt}
    (h : Discharge.HiddenExec.setupOne tags cf pf coins tbls draw partition program = .ok st) :
    checkInRange st.c.kLog st.regions.size (PT_LOCAL + st.nbl) = .ok () ∧ Refine.RegionsWF st := by
  unfold Discharge.HiddenExec.setupOne at h
  split at h
  · exact ⟨(Discharge.Hidden.setupHidden_spec' h).inRange, (Discharge.Hidden.setupHidden_wf h).2⟩
  · exact ⟨FlockVBridge.Family.setupH_inRange h, (Refine.setupH_wf h).2⟩

namespace Gen

variable [S : ZkSetup]

/-- **A masked statement in scope has link points of between 1 and 64 coordinates**, so its two link points fit a
live coin: table 0's range check and regions, whose link points are the session's. -/
theorem zk_mPts {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse true Readable.relation x = some s)
    (hin : Claims.Flock.InScope true s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf true x = some (ins, spec, sts)) : 2 * spec.mPts ≤ 65536 ∧ 1 ≤ spec.mPts := by
  obtain ⟨p, hp, rfl, -, -, -⟩ := parsed_of_parse hs
  obtain ⟨hpst, hpr, hsr⟩ := parseBytes_setup hp hset
  subst hpst
  obtain ⟨-, -, -, -, -, -, -, -, -, hpi, -⟩ := hin
  simp only [Readable.toReadable, Readable.Read.form, Bool.not_eq_true] at hpi
  have hpi' : p.publicInputs = none := Option.not_isSome_iff_eq_none.1 (by simp [hpi])
  rw [← hpr] at hsr
  unfold Readable.Read.setup at hsr
  rw [hpi'] at hsr
  obtain ⟨h0, hall⟩ := Discharge.HiddenExec.setupTables_setupOne hsr
  obtain ⟨d, hd, -⟩ := hall 0 h0
  have hpts := (Discharge.HiddenExec.setupTables_pointsAny hsr).2 0 h0
  obtain ⟨hr, hwf⟩ := zk_setupOne_range hd
  have h64 := ((checkInRange_ok _ _ _).1 hr).2.2
  have hm := hwf.mPts
  unfold PT_LOCAL at h64 hm
  exact ⟨by omega, by omega⟩

end Gen

/-- **A masked statement in scope has link points of between 1 and 64 coordinates**, so its two link points fit a
live coin: table 0's range check and regions, whose link points are the session's. -/
theorem zk_mPts {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse true Readable.relation x = some s)
    (hin : Claims.Flock.InScope true s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf true x = some (ins, spec, sts)) : 2 * spec.mPts ≤ 65536 ∧ 1 ≤ spec.mPts :=
  Gen.zk_mPts (S := S₀) (x := x) (s := s) (hs := hs) (hin := hin) (ins := ins) (spec := spec) (sts := sts)
    (hset := hset)

namespace Gen

variable [S : ZkSetup]

/-- **A masked statement in scope has link points of at most 64 coordinates** (`zk_mPts`). -/
theorem zk_mPts_le {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse true Readable.relation x = some s)
    (hin : Claims.Flock.InScope true s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf true x = some (ins, spec, sts)) : 2 * spec.mPts ≤ 65536 :=
  (zk_mPts hs hin hset).1

end Gen

/-- **A masked statement in scope has link points of at most 64 coordinates** (`zk_mPts`). -/
theorem zk_mPts_le {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse true Readable.relation x = some s)
    (hin : Claims.Flock.InScope true s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf true x = some (ins, spec, sts)) : 2 * spec.mPts ≤ 65536 :=
  Gen.zk_mPts_le (S := S₀) (x := x) (s := s) (hs := hs) (hin := hin) (ins := ins) (spec := spec) (sts := sts)
    (hset := hset)

/-! ## The fork rounds -/

/-- Table `j`'s place among the setup's tables (past the last, table 0's). -/
def zkTbl (sts : Array Stmt) (j : ℕ) : ℕ := if j < sts.size then j else 0

theorem zkTbl_lt {sts : Array Stmt} (h0 : 0 < sts.size) (j : ℕ) : zkTbl sts j < sts.size := by
  unfold zkTbl
  split_ifs with h
  · exact h
  · exact h0

/-- **The round table `t`'s reps start at**: after the link message, the earlier tables' two reps each. -/
def zkBase (spec : SessionSpec) (sts : Array Stmt) (t : ℕ) : ℕ :=
  1 + ((List.range t).map fun i => 2 * repRounds true spec sts[i]!).sum

/-- A masked rep's rounds before Ligerito: the masked replay's and the ring switch's (`repRounds`). -/
def zkPre (spec : SessionSpec) (st : Stmt) : ℕ :=
  (st.m + (st.c.kLog - 6)) + (2 + spec.points * st.regions.size + 1)

/-- The masked Ligerito's rounds at `L` levels: two lane-phase coins and the plain one's `k0 + 5 + 11 (L − 1)`. -/
def zkLig (k0 L : ℕ) : ℕ := k0 + 7 + 11 * (L - 1)

/-- Rep `r`'s first round, of reps of `R` rounds from round `B`. -/
def zkB2 (B R : ℕ) (r : Fin 2) : ℕ := if r.val = 0 then B else B + R

theorem zkB2_cases (B R : ℕ) (r : Fin 2) :
    (r.val = 0 ∧ zkB2 B R r = B) ∨ (r.val = 1 ∧ zkB2 B R r = B + R) := by
  unfold zkB2
  have := r.isLt
  split_ifs with h
  · exact .inl ⟨h, rfl⟩
  · exact .inr ⟨by omega, rfl⟩

/-- **A table's fork rounds**, its reps from round `B`, of `R` rounds, `pre` before Ligerito's `lig`: rep `r`'s
level-`ℓ + 1` cap two lane-phase rounds past `capRound`, its pads at its first round, its `τ` at its third inner
round, and level 0 at the link coin. -/
def zkRdOf (B R pre k0 lig : ℕ) {Lv : ℕ} : Claims.Flock.IZW Lv → ℕ
  | .inl (.inl (r, ℓ)) => zkB2 B R r + pre + capRound k0 (ℓ.val + 1) + 2
  | .inl (.inr (.inl r)) => zkB2 B R r
  | .inl (.inr (.inr r)) => zkB2 B R r + pre + lig + 2
  | .inr () => 0

theorem capRound_ge (k0 n : ℕ) : 2 ≤ capRound k0 n := by
  unfold capRound
  split <;> omega

/-- **No two of a table's fork rounds are one.** -/
theorem zkRdOf_inj {B R pre k0 lig Lv : ℕ} (hB : 1 ≤ B) (hR : R = pre + lig + 4)
    (hcap : ∀ ℓ : Fin Lv, capRound k0 (ℓ.val + 1) + 2 < lig) :
    Function.Injective (zkRdOf (Lv := Lv) B R pre k0 lig) := by
  rintro ((⟨r, ℓ⟩ | r | r) | ⟨⟩) ((⟨r', ℓ'⟩ | r' | r') | ⟨⟩) h <;> dsimp only [zkRdOf] at h
  · have := hcap ℓ
    have := hcap ℓ'
    have := capRound_ge k0 (ℓ.val + 1)
    have := capRound_ge k0 (ℓ'.val + 1)
    have hrr : r = r' := Fin.ext (by
      rcases zkB2_cases B R r with ⟨a, e⟩ | ⟨a, e⟩ <;> rcases zkB2_cases B R r' with ⟨a', e'⟩ | ⟨a', e'⟩ <;> omega)
    subst hrr
    have := capRound_inj (k0 := k0) (by omega) (by omega)
      (show capRound k0 (ℓ.val + 1) = capRound k0 (ℓ'.val + 1) by omega)
    rw [show ℓ = ℓ' from Fin.ext (by omega)]
  · exfalso
    have := hcap ℓ
    rcases zkB2_cases B R r with ⟨a, e⟩ | ⟨a, e⟩ <;> rcases zkB2_cases B R r' with ⟨a', e'⟩ | ⟨a', e'⟩ <;> omega
  · exfalso
    have := hcap ℓ
    rcases zkB2_cases B R r with ⟨a, e⟩ | ⟨a, e⟩ <;> rcases zkB2_cases B R r' with ⟨a', e'⟩ | ⟨a', e'⟩ <;> omega
  · exfalso
    rcases zkB2_cases B R r with ⟨a, e⟩ | ⟨a, e⟩ <;> omega
  · exfalso
    have := hcap ℓ'
    rcases zkB2_cases B R r with ⟨a, e⟩ | ⟨a, e⟩ <;> rcases zkB2_cases B R r' with ⟨a', e'⟩ | ⟨a', e'⟩ <;> omega
  · rw [show r = r' from Fin.ext (by
      rcases zkB2_cases B R r with ⟨a, e⟩ | ⟨a, e⟩ <;> rcases zkB2_cases B R r' with ⟨a', e'⟩ | ⟨a', e'⟩ <;> omega)]
  · exfalso
    rcases zkB2_cases B R r with ⟨a, e⟩ | ⟨a, e⟩ <;> rcases zkB2_cases B R r' with ⟨a', e'⟩ | ⟨a', e'⟩ <;> omega
  · exfalso
    rcases zkB2_cases B R r with ⟨a, e⟩ | ⟨a, e⟩ <;> omega
  · exfalso
    have := hcap ℓ'
    rcases zkB2_cases B R r with ⟨a, e⟩ | ⟨a, e⟩ <;> rcases zkB2_cases B R r' with ⟨a', e'⟩ | ⟨a', e'⟩ <;> omega
  · exfalso
    rcases zkB2_cases B R r with ⟨a, e⟩ | ⟨a, e⟩ <;> rcases zkB2_cases B R r' with ⟨a', e'⟩ | ⟨a', e'⟩ <;> omega
  · rw [show r = r' from Fin.ext (by
      rcases zkB2_cases B R r with ⟨a, e⟩ | ⟨a, e⟩ <;> rcases zkB2_cases B R r' with ⟨a', e'⟩ | ⟨a', e'⟩ <;> omega)]
  · exfalso
    rcases zkB2_cases B R r with ⟨a, e⟩ | ⟨a, e⟩ <;> omega
  · exfalso
    rcases zkB2_cases B R r' with ⟨a', e'⟩ | ⟨a', e'⟩ <;> omega
  · exfalso
    rcases zkB2_cases B R r' with ⟨a', e'⟩ | ⟨a', e'⟩ <;> omega
  · exfalso
    rcases zkB2_cases B R r' with ⟨a', e'⟩ | ⟨a', e'⟩ <;> omega
  · rfl

/-- **Only level 0's fork round is round `0`.** -/
theorem zkRdOf_zero {B R pre k0 lig Lv : ℕ} (hB : 1 ≤ B) (i : Claims.Flock.IZW Lv)
    (h : zkRdOf B R pre k0 lig i = 0) : i = .inr () := by
  rcases i with ((⟨r, ℓ⟩ | r | r) | ⟨⟩) <;> dsimp only [zkRdOf] at h
  · exfalso
    rcases zkB2_cases B R r with ⟨a, e⟩ | ⟨a, e⟩ <;> omega
  · exfalso
    rcases zkB2_cases B R r with ⟨a, e⟩ | ⟨a, e⟩ <;> omega
  · exfalso
    rcases zkB2_cases B R r with ⟨a, e⟩ | ⟨a, e⟩ <;> omega
  · rfl

/-- **A table's fork rounds are among its reps' rounds**, two before their end at the latest. -/
theorem zkRdOf_le {B R pre k0 lig Lv : ℕ} (hB : 1 ≤ B) (hR : R = pre + lig + 4)
    (hcap : ∀ ℓ : Fin Lv, capRound k0 (ℓ.val + 1) + 2 < lig) (i : Claims.Flock.IZW Lv) :
    zkRdOf B R pre k0 lig i + 2 ≤ B + 2 * R := by
  rcases i with ((⟨r, ℓ⟩ | r | r) | ⟨⟩) <;> dsimp only [zkRdOf]
  · have := hcap ℓ
    rcases zkB2_cases B R r with ⟨a, e⟩ | ⟨a, e⟩ <;> omega
  · rcases zkB2_cases B R r with ⟨a, e⟩ | ⟨a, e⟩ <;> omega
  · rcases zkB2_cases B R r with ⟨a, e⟩ | ⟨a, e⟩ <;> omega
  · omega

theorem repRounds_zk {spec : SessionSpec} {st : Stmt} {sch : Flock.Schedule} {tPad : ℕ}
    (hz : Zk.schedule st.m = .ok (sch, tPad)) :
    repRounds true spec st = zkPre spec st + zkLig sch.k0 sch.levels.size + 4 := by
  unfold repRounds zkPre zkLig
  simp only [↓reduceIte, hz]

theorem sum_range_le_zk (f : ℕ → ℕ) {a b : ℕ} (h : a ≤ b) :
    ((List.range a).map f).sum ≤ ((List.range b).map f).sum := by
  induction b, h using Nat.le_induction with
  | base => exact le_rfl
  | succ b _ ih =>
    rw [List.range_succ, List.map_append, List.sum_append]
    simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil]
    omega

/-- The deployed schedule of table `j`'s place. -/
def zkSchT (sts : Array Stmt) (j : ℕ) : Flock.Schedule := (Claims.Flock.zkSched sts[zkTbl sts j]!.m).1

theorem zkTabs_len {sts : Array Stmt} {mPts : ℕ} (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL mPts) (j : ℕ) :
    (zkTabs h0 L hz hk hlay j).sch.levels.length = (zkSchT sts j).levels.size := by
  unfold zkTabs zkSchT zkTbl
  split_ifs with h
  · rw [getElem!_pos sts j h]
    simp only [zkTab, Refine.schedOf, List.length_map, Array.length_toList, hz j h]
  · rw [getElem!_pos sts 0 h0]
    simp only [zkTab, Refine.schedOf, List.length_map, Array.length_toList, hz 0 h0]

/-- **Table `j`'s fork rounds**: `zkRdOf` at its place's reps. -/
def zkRdAt (spec : SessionSpec) (sts : Array Stmt) (j : ℕ) {Lv : ℕ} (i : Claims.Flock.IZW Lv) : ℕ :=
  zkRdOf (zkBase spec sts (zkTbl sts j)) (repRounds true spec sts[zkTbl sts j]!) (zkPre spec sts[zkTbl sts j]!)
    (zkSchT sts j).k0 (zkLig (zkSchT sts j).k0 (zkSchT sts j).levels.size) i

/-- **Table `j`'s reps' numbers**: from round `1` on, of `zkPre`, `zkLig` and four inner rounds, each cap's round
inside Ligerito, and both within the session's rounds. -/
theorem zkRd_facts {spec : SessionSpec} {sts : Array Stmt} (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    (j : ℕ) :
    1 ≤ zkBase spec sts (zkTbl sts j) ∧
      repRounds true spec sts[zkTbl sts j]! =
        zkPre spec sts[zkTbl sts j]! + zkLig (zkSchT sts j).k0 (zkSchT sts j).levels.size + 4 ∧
      (∀ ℓ : Fin (zkTabs h0 L hz hk hlay j).sch.levels.length,
        capRound (zkSchT sts j).k0 (ℓ.val + 1) + 2 < zkLig (zkSchT sts j).k0 (zkSchT sts j).levels.size) ∧
      zkBase spec sts (zkTbl sts j) + 2 * repRounds true spec sts[zkTbl sts j]! ≤ 1 + rounds true spec sts := by
  have ht := zkTbl_lt h0 j
  refine ⟨by unfold zkBase; omega, ?_, fun ℓ => ?_, ?_⟩
  · unfold zkSchT
    rw [getElem!_pos sts _ ht]
    exact repRounds_zk (hz _ ht)
  · have hlen := zkTabs_len h0 L hz hk hlay j
    have h2 := (zkTabs h0 L hz hk hlay j).hshape.1
    have hℓ := ℓ.isLt
    have := capRound_lt (k0 := (zkSchT sts j).k0) (L := (zkSchT sts j).levels.size) (by omega)
      (show ℓ.val + 1 ≤ (zkSchT sts j).levels.size by omega)
    unfold zkLig
    omega
  · have hs := sum_range_le_zk (fun i => 2 * repRounds true spec sts[i]!) (Nat.succ_le_of_lt ht)
    rw [List.range_succ, List.map_append, List.sum_append] at hs
    simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil] at hs
    rw [rounds_sum]
    unfold zkBase
    omega

theorem zkRdAt_le {spec : SessionSpec} {sts : Array Stmt} (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    (j : ℕ) (i : Claims.Flock.IZW (zkTabs h0 L hz hk hlay j).sch.levels.length) :
    zkRdAt spec sts j i + 2 ≤ 1 + rounds true spec sts := by
  obtain ⟨hB, hR, hcap, hle⟩ := zkRd_facts h0 L hz hk hlay j
  have := zkRdOf_le hB hR hcap i
  unfold zkRdAt
  omega

/-- **The masked fork rounds** of table `j`'s events, rounds of `V true`. -/
noncomputable def zkRd {x : Claims.Msg} {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    (j : ℕ) (i : Claims.Flock.IZW (zkTabs h0 L hz hk hlay j).sch.levels.length) : Fin ((V true).k x) :=
  ⟨zkRdAt spec sts j i, by have := zkRdAt_le h0 L hz hk hlay j i; rw [zkK hset]; omega⟩

namespace Gen

variable [S : ZkSetup]

/-- **The masked fork rounds** of table `j`'s events, rounds of `V true`. -/
noncomputable def zkRd {x : Claims.Msg} {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    (j : ℕ) (i : Claims.Flock.IZW (zkTabs h0 L hz hk hlay j).sch.levels.length) : Fin ((V true).k x) :=
  ⟨zkRdAt spec sts j i, by have := zkRdAt_le h0 L hz hk hlay j i; rw [zkK hset]; omega⟩

theorem zkRd_succ_lt {x : Claims.Msg} {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    (j : ℕ) (i : Claims.Flock.IZW (zkTabs h0 L hz hk hlay j).sch.levels.length) :
    (zkRd hset h0 L hz hk hlay j i).val + 1 < (V true).k x := by
  have := zkRdAt_le h0 L hz hk hlay j i
  have := zkK hset
  exact show zkRdAt spec sts j i + 1 < (V true).k x by omega

end Gen

theorem zkRd_succ_lt {x : Claims.Msg} {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    (j : ℕ) (i : Claims.Flock.IZW (zkTabs h0 L hz hk hlay j).sch.levels.length) :
    (zkRd hset h0 L hz hk hlay j i).val + 1 < (V true).k x :=
  Gen.zkRd_succ_lt (S := S₀) (x := x) (ins := ins) (spec := spec) (sts := sts) (hset := hset) (h0 := h0) (L := L)
    (hz := hz) (hk := hk) (hlay := hlay) (j := j) (i := i)

namespace Gen

variable [S : ZkSetup]

theorem zkRd_inj {x : Claims.Msg} {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    (j : ℕ) : Function.Injective (zkRd hset h0 L hz hk hlay j) := fun a b h => by
  obtain ⟨hB, hR, hcap, -⟩ := zkRd_facts h0 L hz hk hlay j
  exact zkRdOf_inj hB hR hcap (congrArg Fin.val h)

end Gen

theorem zkRd_inj {x : Claims.Msg} {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    (j : ℕ) : Function.Injective (zkRd hset h0 L hz hk hlay j) :=
  Gen.zkRd_inj (S := S₀) (x := x) (ins := ins) (spec := spec) (sts := sts) (hset := hset) (h0 := h0) (L := L) (hz := hz)
    (hk := hk) (hlay := hlay) (j := j)

namespace Gen

variable [S : ZkSetup]

theorem zkRd_zero {x : Claims.Msg} {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    (j : ℕ) (i : Claims.Flock.IZW (zkTabs h0 L hz hk hlay j).sch.levels.length)
    (h : (zkRd hset h0 L hz hk hlay j i).val = 0) : i = .inr () :=
  zkRdOf_zero (zkRd_facts h0 L hz hk hlay j).1 i h

end Gen

theorem zkRd_zero {x : Claims.Msg} {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    (j : ℕ) (i : Claims.Flock.IZW (zkTabs h0 L hz hk hlay j).sch.levels.length)
    (h : (zkRd hset h0 L hz hk hlay j i).val = 0) : i = .inr () :=
  Gen.zkRd_zero (S := S₀) (x := x) (ins := ins) (spec := spec) (sts := sts) (hset := hset) (h0 := h0) (L := L)
    (hz := hz) (hk := hk) (hlay := hlay) (j := j) (i := i) (h := h)

/-! ## The simulator's coins -/

/-- **The live session's rounds at `x`**: the deployed verifier's, between the link message and the last. -/
def zkN (x : Claims.Msg) : ℕ := (V true).k x - 2

namespace Gen

variable [S : ZkSetup]

/-- **The live session's rounds at `x`**: the deployed verifier's, between the link message and the last. -/
def zkN (x : Claims.Msg) : ℕ := (V true).k x - 2

theorem zkN_add {x : Claims.Msg} {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf true x = some (ins, spec, sts)) : zkN x + 2 = (V true).k x := by
  have := zkK hset
  unfold zkN
  omega

end Gen

theorem zkN_add {x : Claims.Msg} {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf true x = some (ins, spec, sts)) : zkN x + 2 = (V true).k x :=
  Gen.zkN_add (S := S₀) (x := x) (ins := ins) (spec := spec) (sts := sts) (hset := hset)

/-- **The live coins of deployed round `i`**: the link's `2·mPts` link coordinates, a round's count in the session's
(`zkNs`, at most a live coin's `65536`), and at least one, so one at the last message, whose coin the session never
reads. -/
def zkCnt (x : Claims.Msg) (spec : SessionSpec) (sts : Array Stmt) (i : ℕ) : ℕ :=
  max 1 (if i = 0 then 2 * spec.mPts else if i ≤ zkN x then min ((zkNs spec sts).getD (i - 1) 0) 65536 else 0)

namespace Gen

variable [S : ZkSetup]

/-- **The live coins of deployed round `i`**: the link's `2·mPts` link coordinates, a round's count in the session's
(`zkNs`, at most a live coin's `65536`), and at least one, so one at the last message, whose coin the session never
reads. -/
def zkCnt (x : Claims.Msg) (spec : SessionSpec) (sts : Array Stmt) (i : ℕ) : ℕ :=
  max 1 (if i = 0 then 2 * spec.mPts else if i ≤ zkN x then min ((zkNs spec sts).getD (i - 1) 0) 65536 else 0)

end Gen

/-- `zkCnt` at the deployed rounds. -/
def zkNOf (x : Claims.Msg) (spec : SessionSpec) (sts : Array Stmt) (i : Fin ((V true).k x)) : ℕ :=
  zkCnt x spec sts i.val

namespace Gen

variable [S : ZkSetup]

/-- `zkCnt` at the deployed rounds. -/
def zkNOf (x : Claims.Msg) (spec : SessionSpec) (sts : Array Stmt) (i : Fin ((V true).k x)) : ℕ :=
  zkCnt x spec sts i.val

theorem zkNOf_pos (x : Claims.Msg) (spec : SessionSpec) (sts : Array Stmt) (i : Fin ((V true).k x)) :
    1 ≤ zkNOf x spec sts i :=
  le_max_left _ _

end Gen

theorem zkNOf_pos (x : Claims.Msg) (spec : SessionSpec) (sts : Array Stmt) (i : Fin ((V true).k x)) :
    1 ≤ zkNOf x spec sts i :=
  Gen.zkNOf_pos (S := S₀) (x := x) (spec := spec) (sts := sts) (i := i)

/-- The revealed history a deployed round's tape entry is read at (`FnFork.histKey`). -/
abbrev zkD (x : Claims.Msg) : Type := Fin ((V true).k x) → Fin 65536 → Option 𝔽

namespace Gen

variable [S : ZkSetup]

/-- The revealed history a deployed round's tape entry is read at (`FnFork.histKey`). -/
abbrev zkD (x : Claims.Msg) : Type := Fin ((V true).k x) → Fin 65536 → Option 𝔽

end Gen

/-- **The simulator's tape**: per deployed round, a coin at each revealed history. -/
abbrev zkTape (x : Claims.Msg) : Type := Fin ((V true).k x) → zkD x → LCoin 65536

namespace Gen

variable [S : ZkSetup]

/-- **The simulator's tape**: per deployed round, a coin at each revealed history. -/
abbrev zkTape (x : Claims.Msg) : Type := Fin ((V true).k x) → zkD x → LCoin 65536

end Gen

/-- The history key: the live coins' entries revealed by deployed round `j`. -/
def zkKey (x : Claims.Msg) (spec : SessionSpec) (sts : Array Stmt) :
    Fin ((V true).k x) → (Fin ((V true).k x) → LCoin 65536) → zkD x :=
  Claims.Flock.FnFork.histKey (zkNOf x spec sts)

namespace Gen

variable [S : ZkSetup]

/-- The history key: the live coins' entries revealed by deployed round `j`. -/
def zkKey (x : Claims.Msg) (spec : SessionSpec) (sts : Array Stmt) :
    Fin ((V true).k x) → (Fin ((V true).k x) → LCoin 65536) → zkD x :=
  Claims.Flock.FnFork.histKey (zkNOf x spec sts)

end Gen

/-- **The simulator's coins**: `fnAt` at the tape and the live coins, the live coin's revealed entries and the tape's at
the revealed history for the rest. -/
noncomputable def zkΦ (x : Claims.Msg) (spec : SessionSpec) (sts : Array Stmt) :
    zkTape x × (Fin (zkN x + 2) → LCoin 65536) → Fin ((V true).k x) → Coin :=
  if hL : zkN x + 2 = (V true).k x then fun q => fnAt hL (zkNOf x spec sts) (zkKey x spec sts) q.1 q.2
  else fun _ => default

namespace Gen

variable [S : ZkSetup]

/-- **The simulator's coins**: `fnAt` at the tape and the live coins, the live coin's revealed entries and the tape's at
the revealed history for the rest. -/
noncomputable def zkΦ (x : Claims.Msg) (spec : SessionSpec) (sts : Array Stmt) :
    zkTape x × (Fin (zkN x + 2) → LCoin 65536) → Fin ((V true).k x) → Coin :=
  if hL : zkN x + 2 = (V true).k x then fun q => fnAt hL (zkNOf x spec sts) (zkKey x spec sts) q.1 q.2
  else fun _ => default

theorem zkΦ_eq {x : Claims.Msg} {spec : SessionSpec} {sts : Array Stmt} (hL : zkN x + 2 = (V true).k x) :
    zkΦ x spec sts = fun q => fnAt hL (zkNOf x spec sts) (zkKey x spec sts) q.1 q.2 :=
  dite_eq_left hL

end Gen

theorem zkΦ_eq {x : Claims.Msg} {spec : SessionSpec} {sts : Array Stmt} (hL : zkN x + 2 = (V true).k x) :
    zkΦ x spec sts = fun q => fnAt hL (zkNOf x spec sts) (zkKey x spec sts) q.1 q.2 :=
  Gen.zkΦ_eq (S := S₀) (x := x) (spec := spec) (sts := sts) (hL := hL)

namespace Gen

variable [S : ZkSetup]

/-- **The simulator's coins are uniform** (`Pr_fnAt`). -/
theorem Pr_zkΦ {x : Claims.Msg} {spec : SessionSpec} {sts : Array Stmt} (hL : zkN x + 2 = (V true).k x)
    (E : (Fin ((V true).k x) → Coin) → Prop) :
    Claims.Pr (fun p => E (zkΦ x spec sts p)) = Claims.Pr E := by
  rw [zkΦ_eq hL]
  exact Pr_fnAt hL (zkNOf x spec sts) (zkKey x spec sts) E

end Gen

/-- **The simulator's coins are uniform** (`Pr_fnAt`). -/
theorem Pr_zkΦ {x : Claims.Msg} {spec : SessionSpec} {sts : Array Stmt} (hL : zkN x + 2 = (V true).k x)
    (E : (Fin ((V true).k x) → Coin) → Prop) :
    Claims.Pr (fun p => E (zkΦ x spec sts p)) = Claims.Pr E :=
  Gen.Pr_zkΦ (S := S₀) (x := x) (spec := spec) (sts := sts) (hL := hL) (E := E)

namespace Gen

variable [S : ZkSetup]

/-- **The coupling at the deployed masked verifier**: at `V true`, with a tape of uniform functions of the live coins
`LCoin 65536` read through a prefix key (`Claims.Bridge.Live.fnAt`), `ForkCoinCoupling` holds at the chances
`FnFork.δN GF128 65536 n` (`forkCoinCoupling_fnAt`). -/
theorem forkCoinCoupling_zk {D : Type} [Fintype D] [DecidableEq D] {Ω : Type} (P : Claims.Prover Ω)
    (x : Claims.Msg) (ω : Ω) {L : ℕ} (h : L = (V true).k x) (n : Fin ((V true).k x) → ℕ)
    {key : Fin ((V true).k x) → (Fin ((V true).k x) → LCoin 65536) → D}
    (hk : Claims.Flock.FnFork.PrefixKey n key) :
    Claims.Flock.ForkCoinCoupling sha512 (V true) P x (Claims.Flock.FnFork.δN 𝔽 65536 n)
      (Claims.Flock.simForkPr sha512 (V true) P x
        (fun q : (Fin ((V true).k x) → D → LCoin 65536) × (Fin L → LCoin 65536) => fnAt h n key q.1 q.2)
        (fun t => Claims.Flock.FnFork.forkN t) ω) ω :=
  Claims.Flock.forkCoinCoupling_fnAt (A := 𝔽) (M := 65536) (hI := instInhabitedCoin) sha512 (V true) P x ω h n hk

end Gen

/-- **The coupling at the deployed masked verifier**: at `V true`, with a tape of uniform functions of the live coins
`LCoin 65536` read through a prefix key (`Claims.Bridge.Live.fnAt`), `ForkCoinCoupling` holds at the chances
`FnFork.δN GF128 65536 n` (`forkCoinCoupling_fnAt`). -/
theorem forkCoinCoupling_zk {D : Type} [Fintype D] [DecidableEq D] {Ω : Type} (P : Claims.Prover Ω)
    (x : Claims.Msg) (ω : Ω) {L : ℕ} (h : L = (V true).k x) (n : Fin ((V true).k x) → ℕ)
    {key : Fin ((V true).k x) → (Fin ((V true).k x) → LCoin 65536) → D}
    (hk : Claims.Flock.FnFork.PrefixKey n key) :
    Claims.Flock.ForkCoinCoupling sha512 (V true) P x (Claims.Flock.FnFork.δN 𝔽 65536 n)
      (Claims.Flock.simForkPr sha512 (V true) P x
        (fun q : (Fin ((V true).k x) → D → LCoin 65536) × (Fin L → LCoin 65536) => fnAt h n key q.1 q.2)
        (fun t => Claims.Flock.FnFork.forkN t) ω) ω :=
  Gen.forkCoinCoupling_zk (S := S₀) (D := D) (Ω := Ω) (P := P) (x := x) (ω := ω) (L := L) (h := h) (n := n) (key := key)
    (hk := hk)

namespace Gen

variable [S : ZkSetup]

/-- **The simulated fork at `zkΦ` against the deployed one**: at most the deployed fork at round `t` with its collision
and the slack there at the chances `δN GF128 65536 (zkNOf …)` (`slack_of_coupling` at `forkCoinCoupling_zk`). -/
theorem zk_simFork_le {Ω : Type} (P : Claims.Prover Ω) {x : Claims.Msg} (ω : Ω) {ins : Ins} {spec : SessionSpec}
    {sts : Array Stmt} (hset : setupOf true x = some (ins, spec, sts)) (t : Fin ((V true).k x)) :
    Claims.Flock.simForkPr sha512 (V true) P x (zkΦ x spec sts) (fun t => Claims.Flock.FnFork.forkN t) ω t.val ≤
      (Claims.Pr fun s : (Fin ((V true).k x) → Coin) × (Fin ((V true).k x) → Coin) => zkForkEv P x t ω s.1 s.2) +
        Claims.Flock.forkSlack sha512 (V true) P x (Claims.Flock.FnFork.δN 𝔽 65536 (zkNOf x spec sts)) t.val ω := by
  rw [zkΦ_eq (zkN_add hset)]
  exact Claims.Flock.slack_of_coupling sha512 (V true) P x
    (fun t => Claims.Flock.FnFork.δN_nonneg («X» := 𝔽) (M := 65536) (zkNOf x spec sts) t)
    (forkCoinCoupling_zk P x ω (zkN_add hset) (zkNOf x spec sts) (Claims.Flock.FnFork.prefixKey_histKey _)) t

end Gen

/-- **The simulated fork at `zkΦ` against the deployed one**: at most the deployed fork at round `t` with its collision
and the slack there at the chances `δN GF128 65536 (zkNOf …)` (`slack_of_coupling` at `forkCoinCoupling_zk`). -/
theorem zk_simFork_le {Ω : Type} (P : Claims.Prover Ω) {x : Claims.Msg} (ω : Ω) {ins : Ins} {spec : SessionSpec}
    {sts : Array Stmt} (hset : setupOf true x = some (ins, spec, sts)) (t : Fin ((V true).k x)) :
    Claims.Flock.simForkPr sha512 (V true) P x (zkΦ x spec sts) (fun t => Claims.Flock.FnFork.forkN t) ω t.val ≤
      (Claims.Pr fun s : (Fin ((V true).k x) → Coin) × (Fin ((V true).k x) → Coin) => zkForkEv P x t ω s.1 s.2) +
        Claims.Flock.forkSlack sha512 (V true) P x (Claims.Flock.FnFork.δN 𝔽 65536 (zkNOf x spec sts)) t.val ω :=
  Gen.zk_simFork_le (S := S₀) (Ω := Ω) (P := P) (x := x) (ω := ω) (ins := ins) (spec := spec) (sts := sts)
    (hset := hset) (t := t)

/-! ## The simulator -/

/-- `n` field elements as a coin, `0` past them. -/
noncomputable def zkPadE (n : ℕ) (c : Fin n → 𝔽) : LCoin 65536 := fun e => if h : e.val < n then c ⟨e.val, h⟩ else 0

/-- **Live coin `i`'s revealed entries, read back from the verifier's reply `m`**: the link points at `i = 0`, a
round's coins after. -/
noncomputable def zkReveal (x : Claims.Msg) (spec : SessionSpec) (sts : Array Stmt) (i : ℕ) (m : Claims.Msg) :
    LCoin 65536 :=
  if i = 0 then zkPadE (2 * spec.mPts) ((ptsE spec.mPts).symm (finDecode m))
  else zkPadE (zkCnt x spec sts i) (finDecode m)

namespace Gen

variable [S : ZkSetup]

/-- **Live coin `i`'s revealed entries, read back from the verifier's reply `m`**: the link points at `i = 0`, a
round's coins after. -/
noncomputable def zkReveal (x : Claims.Msg) (spec : SessionSpec) (sts : Array Stmt) (i : ℕ) (m : Claims.Msg) :
    LCoin 65536 :=
  if i = 0 then zkPadE (2 * spec.mPts) ((ptsE spec.mPts).symm (finDecode m))
  else zkPadE (zkCnt x spec sts i) (finDecode m)

end Gen

/-- The live coins' revealed entries at the live history `hl`, `0` past it. -/
noncomputable def zkRecon (x : Claims.Msg) (spec : SessionSpec) (sts : Array Stmt)
    (hl : List (Claims.Msg × Claims.Msg)) : Fin ((V true).k x) → LCoin 65536 :=
  fun i => if i.val < hl.length then zkReveal x spec sts i.val (hl.getD i.val default).2 else fun _ => 0

namespace Gen

variable [S : ZkSetup]

/-- The live coins' revealed entries at the live history `hl`, `0` past it. -/
noncomputable def zkRecon (x : Claims.Msg) (spec : SessionSpec) (sts : Array Stmt)
    (hl : List (Claims.Msg × Claims.Msg)) : Fin ((V true).k x) → LCoin 65536 :=
  fun i => if i.val < hl.length then zkReveal x spec sts i.val (hl.getD i.val default).2 else fun _ => 0

end Gen

/-- **The deployed coins the simulator feeds `P` at the live history `hl`**: `fnC` at its tape `u` and the revealed
entries. -/
noncomputable def zkSimCoins (x : Claims.Msg) (spec : SessionSpec) (sts : Array Stmt) (u : zkTape x)
    (hl : List (Claims.Msg × Claims.Msg)) : Fin ((V true).k x) → Coin :=
  fnC (zkNOf x spec sts) (zkKey x spec sts) u (zkRecon x spec sts hl)

namespace Gen

variable [S : ZkSetup]

/-- **The deployed coins the simulator feeds `P` at the live history `hl`**: `fnC` at its tape `u` and the revealed
entries. -/
noncomputable def zkSimCoins (x : Claims.Msg) (spec : SessionSpec) (sts : Array Stmt) (u : zkTape x)
    (hl : List (Claims.Msg × Claims.Msg)) : Fin ((V true).k x) → Coin :=
  fnC (zkNOf x spec sts) (zkKey x spec sts) u (zkRecon x spec sts hl)

end Gen

/-- **`P`'s history after `j` rounds of a verifier**, at the coins `c`: each message answered by its coin's bytes. -/
def zkDepHist {Ω : Type} (H : Claims.Msg → Claims.Msg) (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω)
    (c : ℕ → Coin) : ℕ → List (Claims.Msg × Claims.Msg)
  | 0 => []
  | j + 1 => zkDepHist H P x ω c j ++ [(((P.next x (zkDepHist H P x ω c j) ω).eval H).1, (c j).bytes)]

/-- `P`'s message after `j` rounds, at the coins `c`. -/
def zkDepMsg {Ω : Type} (H : Claims.Msg → Claims.Msg) (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω)
    (c : ℕ → Coin) (j : ℕ) : Claims.Msg :=
  ((P.next x (zkDepHist H P x ω c j) ω).eval H).1

/-- **The rep streams of the deployed rounds**, in `spec.repStreams` order: stream `k`, table `k / 2`'s rep `k % 2`,
once per round of its rep. -/
def zkStreams (spec : SessionSpec) (sts : Array Stmt) : List ℕ :=
  (List.range (2 * sts.size)).flatMap fun k => List.replicate (zkRepNsOf spec sts[k / 2]!).length k

/-- Deployed round `i`'s rep stream (`i ≥ 1`). -/
def zkStream (spec : SessionSpec) (sts : Array Stmt) (i : ℕ) : ℕ := (zkStreams spec sts).getD (i - 1) 0

/-- **A round on the masked wire** (`zkSub`): tag `2`, the table in unary (`k / 2` ones, then `0`), the rep, the count
in three bytes (low first) and the retained bytes, the deployed round message's after its tag. -/
def zkRound (k n : ℕ) (a : Claims.Msg) : Claims.Msg :=
  2 :: (List.replicate (k / 2) 1 ++ 0 :: UInt8.ofNat (k % 2) :: UInt8.ofNat (n % 256) :: UInt8.ofNat (n / 256 % 256) ::
    UInt8.ofNat (n / 65536) :: a.tail)

/-- The leading ones of a message, counted, and the rest. -/
def zkOnes : List UInt8 → ℕ × List UInt8
  | [] => (0, [])
  | b :: rest => if b = 1 then ((zkOnes rest).1 + 1, (zkOnes rest).2) else (0, b :: rest)

/-- A round of table `t` read after its unary table: `0`, the rep, the count and the retained bytes; a table past
`J`, a rep past `1` or a count past `65536` is a stop, as is any other message. -/
def zkSubAt (J t : ℕ) : List UInt8 → Option (SubmitJ J)
  | 0 :: r :: n₀ :: n₁ :: n₂ :: rest =>
    if h : t < J ∧ r.toNat < 2 ∧ n₀.toNat + 256 * n₁.toNat + 65536 * n₂.toNat ≤ 65536 then
      some { tab := ⟨t, h.1⟩, rep := ⟨r.toNat, h.2.1⟩, bytes := ⟨rest.toArray⟩,
             n := n₀.toNat + 256 * n₁.toNat + 65536 * n₂.toNat }
    else none
  | _ => none

theorem zkSubAt_le (J t : ℕ) (a : List UInt8) (s : SubmitJ J) (h : zkSubAt J t a = some s) : s.n ≤ 65536 := by
  unfold zkSubAt at h
  split at h
  · split at h
    · next hc =>
      cases h
      exact hc.2.2
    · cases h
  · cases h

/-- **A round on the masked wire read back** (`zkRound`): tag `2`, the table in unary, then `zkSubAt`. -/
def zkSub (J : ℕ) : Claims.Msg → Option (SubmitJ J)
  | 2 :: rest => zkSubAt J (zkOnes rest).1 (zkOnes rest).2
  | _ => none

theorem zkSub_le (J : ℕ) (a : Claims.Msg) (s : SubmitJ J) (h : zkSub J a = some s) : s.n ≤ 65536 := by
  unfold zkSub at h
  split at h
  · exact zkSubAt_le J _ _ s h
  · cases h

/-- **The masked wire** at `J` tables and `mPts` link coordinates: the plain wire's roots, proof files and codes, its
rounds at their table (`zkSub`). -/
noncomputable def zkWire (J mPts : ℕ) (h : 2 * mPts ≤ 65536) : Wire 65536 J mPts where
  root a := ((linkOf a).getD default).1.toArray
  sub := zkSub J
  sub_le := zkSub_le J
  proofs a := ((finalOf a).map Final.proofs).getD #[]
  pts_le := h
  encPts := finCode
  decPts := finDecode
  decPts_enc := finDecode_code
  encRd _ := finCode
  decRd _ := finDecode
  decRd_enc _ := finDecode_code

/-- The simulator's wire: the masked wire at the session's tables and link coordinates. -/
noncomputable def zkW (J : ℕ) (spec : SessionSpec) (h : 2 * spec.mPts ≤ 65536) : Claims.Msg → Wire 65536 J spec.mPts :=
  fun _ => zkWire J spec.mPts h

/-- **The simulator's message at the live history `hl`**: `P`'s at the coins `zkSimCoins` it rebuilds, its first as
it is, each round's as `zkRound` at the round's stream and count, its last only where `V true` accepts `P` at those
coins, else nothing. -/
noncomputable def zkMsg {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (spec : SessionSpec) (sts : Array Stmt)
    (ω : Ω) (u : zkTape x) (hl : List (Claims.Msg × Claims.Msg)) : Claims.Msg :=
  if hl.length = 0 then zkDepMsg sha512 P x ω (Claims.coinSeq (zkSimCoins x spec sts u hl)) 0
  else if hl.length ≤ zkN x then
    zkRound (zkStream spec sts hl.length) (zkCnt x spec sts hl.length)
      (zkDepMsg sha512 P x ω (Claims.coinSeq (zkSimCoins x spec sts u hl)) hl.length)
  else if (Claims.run sha512 P (V true) x ω (zkSimCoins x spec sts u hl)).out = .acc then
    zkDepMsg sha512 P x ω (Claims.coinSeq (zkSimCoins x spec sts u hl)) hl.length
  else []

namespace Gen

variable [S : ZkSetup]

/-- **The simulator's message at the live history `hl`**: `P`'s at the coins `zkSimCoins` it rebuilds, its first as
it is, each round's as `zkRound` at the round's stream and count, its last only where `V true` accepts `P` at those
coins, else nothing. -/
noncomputable def zkMsg {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (spec : SessionSpec) (sts : Array Stmt)
    (ω : Ω) (u : zkTape x) (hl : List (Claims.Msg × Claims.Msg)) : Claims.Msg :=
  if hl.length = 0 then zkDepMsg sha512 P x ω (Claims.coinSeq (zkSimCoins x spec sts u hl)) 0
  else if hl.length ≤ zkN x then
    zkRound (zkStream spec sts hl.length) (zkCnt x spec sts hl.length)
      (zkDepMsg sha512 P x ω (Claims.coinSeq (zkSimCoins x spec sts u hl)) hl.length)
  else if (Claims.run sha512 P (V true) x ω (zkSimCoins x spec sts u hl)).out = .acc then
    zkDepMsg sha512 P x ω (Claims.coinSeq (zkSimCoins x spec sts u hl)) hl.length
  else []

end Gen

/-- **The masked simulator** of `P` at `x`, at the tape `(ω, u)`. -/
noncomputable def zkSim {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (spec : SessionSpec) (sts : Array Stmt) :
    Claims.Prover (Ω × zkTape x) :=
  ⟨fun _ hl p => .ret (zkMsg P x spec sts p.1 p.2 hl)⟩

namespace Gen

variable [S : ZkSetup]

/-- **The masked simulator** of `P` at `x`, at the tape `(ω, u)`. -/
noncomputable def zkSim {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (spec : SessionSpec) (sts : Array Stmt) :
    Claims.Prover (Ω × zkTape x) :=
  ⟨fun _ hl p => .ret (zkMsg P x spec sts p.1 p.2 hl)⟩

/-- **The simulator's first message is `P`'s.** -/
theorem zkSim_first {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (spec : SessionSpec) (sts : Array Stmt)
    (ω : Ω) (u : zkTape x) :
    (((zkSim P x spec sts).next x [] (ω, u)).eval sha512).1 = ((P.next x [] ω).eval sha512).1 := by
  simp only [zkSim, Claims.OC.eval, zkMsg]
  rfl

end Gen

/-- **The simulator's first message is `P`'s.** -/
theorem zkSim_first {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (spec : SessionSpec) (sts : Array Stmt)
    (ω : Ω) (u : zkTape x) :
    (((zkSim P x spec sts).next x [] (ω, u)).eval sha512).1 = ((P.next x [] ω).eval sha512).1 :=
  Gen.zkSim_first (S := S₀) (Ω := Ω) (P := P) (x := x) (spec := spec) (sts := sts) (ω := ω) (u := u)

namespace Gen

variable [S : ZkSetup]

/-- **The simulator sends `P`'s roots** (`zkRoot_of_first`). -/
theorem zkSim_root {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (spec : SessionSpec) (sts : Array Stmt)
    (ω : Ω) (J : ℕ) (h : 2 * spec.mPts ≤ 65536) (u : zkTape x) :
    (liveStrat (fun _ => J) (fun _ => spec.mPts) zkN spec.dg (zkW J spec h) sha512 (zkSim P x spec sts) x
      (ω, u)).1 = zkRoot P x ω :=
  zkRoot_of_first (J := fun _ => J) zkN spec.dg (fun _ => rfl) (zkSim_first P x spec sts ω) u

end Gen

/-- **The simulator sends `P`'s roots** (`zkRoot_of_first`). -/
theorem zkSim_root {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (spec : SessionSpec) (sts : Array Stmt)
    (ω : Ω) (J : ℕ) (h : 2 * spec.mPts ≤ 65536) (u : zkTape x) :
    (liveStrat (fun _ => J) (fun _ => spec.mPts) zkN spec.dg (zkW J spec h) sha512 (zkSim P x spec sts) x
      (ω, u)).1 = zkRoot P x ω :=
  Gen.zkSim_root (S := S₀) (Ω := Ω) (P := P) (x := x) (spec := spec) (sts := sts) (ω := ω) (J := J) (h := h) (u := u)

/-! ## What is left -/

/-- **`P'` aborts where `P`'s masked run is rejected** (`PlainSimAbort` at `V true`): at every tape and live coins at
which `V true` does not accept `P` at `Φ`, the live event `pL` fails at `P'`'s outcome. -/
def ZkSimAbort {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) (mPts : ℕ) (J N : Claims.Msg → ℕ)
    (dg : ByteArray → ByteArray) (w : (x : Claims.Msg) → Wire 65536 (J x) mPts) {τ : Type}
    (P' : Claims.Prover (Ω × τ)) (Φ : τ × (Fin (N x + 2) → LCoin 65536) → Fin ((V true).k x) → Coin)
    (pL : LiveOutJ (J x) mPts → Prop) : Prop :=
  ∀ u r', (Claims.run sha512 P (V true) x ω (Φ (u, r'))).out ≠ .acc →
    ¬ pL (liveOut J (fun _ => mPts) N dg w sha512 P' x (ω, u) r')

namespace Gen

variable [S : ZkSetup]

/-- **`P'` aborts where `P`'s masked run is rejected** (`PlainSimAbort` at `V true`): at every tape and live coins at
which `V true` does not accept `P` at `Φ`, the live event `pL` fails at `P'`'s outcome. -/
def ZkSimAbort {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) (mPts : ℕ) (J N : Claims.Msg → ℕ)
    (dg : ByteArray → ByteArray) (w : (x : Claims.Msg) → Wire 65536 (J x) mPts) {τ : Type}
    (P' : Claims.Prover (Ω × τ)) (Φ : τ × (Fin (N x + 2) → LCoin 65536) → Fin ((V true).k x) → Coin)
    (pL : LiveOutJ (J x) mPts → Prop) : Prop :=
  ∀ u r', (Claims.run sha512 P (V true) x ω (Φ (u, r'))).out ≠ .acc →
    ¬ pL (liveOut J (fun _ => mPts) N dg w sha512 P' x (ω, u) r')

end Gen

/-- **The lockstep**: at every masked statement in scope that sets up, at C-Flock ZK's padded tables, and every prover
`P` and its tape `ω`, the masked simulator simulates `P`'s accepting runs at `zkΦ`'s coins on the masked wire
(`ZkSimRun`): its live outcome is the view of each record `P`'s run decides on. -/
def ZkSimRunAt (x : Claims.Msg) : Prop :=
  ∀ (s : Claims.Flock.Stmt) (hs : parse true Readable.relation x = some s) (hin : Claims.Flock.InScope true s)
    (ins : Ins) (spec : SessionSpec) (sts : Array Stmt) (hset : setupOf true x = some (ins, spec, sts))
    (h0 : 0 < sts.size) (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    (Ω : Type) (P : Claims.Prover Ω) (ω : Ω),
    ZkSimRun P x ω (fun _ => (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length) spec.mPts zkN spec.dg
      (zkW (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length spec (zk_mPts_le hs hin hset))
      (zkSim P x spec sts) (zkΦ x spec sts)

namespace Gen

variable [S : ZkSetup]

/-- **The lockstep**: at every masked statement in scope that sets up, at C-Flock ZK's padded tables, and every prover
`P` and its tape `ω`, the masked simulator simulates `P`'s accepting runs at `zkΦ`'s coins on the masked wire
(`ZkSimRun`): its live outcome is the view of each record `P`'s run decides on. -/
def ZkSimRunAt (x : Claims.Msg) : Prop :=
  ∀ (s : Claims.Flock.Stmt) (hs : parse true Readable.relation x = some s) (hin : Claims.Flock.InScope true s)
    (ins : Ins) (spec : SessionSpec) (sts : Array Stmt) (hset : setupOf true x = some (ins, spec, sts))
    (h0 : 0 < sts.size) (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    (Ω : Type) (P : Claims.Prover Ω) (ω : Ω),
    ZkSimRun P x ω (fun _ => (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length) spec.mPts zkN spec.dg
      (zkW (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length spec (zk_mPts_le hs hin hset))
      (zkSim P x spec sts) (zkΦ x spec sts)

end Gen

/-- **The abort**: at every masked statement in scope that sets up at a draw `d`, every prover and tape, every table
`j` and table of full-width columns, where `P`'s masked run at `zkΦ`'s coins is rejected the live verdict at the draw
(`AccTxJ`) fails at the masked simulator's outcome (`ZkSimAbort`). -/
def ZkSimAbortAt (x : Claims.Msg) : Prop :=
  ∀ (s : Claims.Flock.Stmt) (hs : parse true Readable.relation x = some s) (hin : Claims.Flock.InScope true s)
    (ins : Ins) (spec : SessionSpec) (sts : Array Stmt) (hset : setupOf true x = some (ins, spec, sts))
    (h0 : 0 < sts.size) (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    (d : Lean.Json) (_ : ins.read.draw = some d) (Ω : Type) (P : Claims.Prover Ω) (ω : Ω) (j : ℕ)
    (_ : j < (tables true x).length) (C : Fin (wordLenJ true x j) → List 𝔽), (∀ p, wideJ true x j (C p)) →
    ZkSimAbort P x ω spec.mPts (fun _ => (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length) zkN spec.dg
      (zkW (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length spec (zk_mPts_le hs hin hset))
      (zkSim P x spec sts) (zkΦ x spec sts) (AccTxJ ins.read.I d j (zkRoot P x ω) C)

namespace Gen

variable [S : ZkSetup]

/-- **The abort**: at every masked statement in scope that sets up at a draw `d`, every prover and tape, every table
`j` and table of full-width columns, where `P`'s masked run at `zkΦ`'s coins is rejected the live verdict at the draw
(`AccTxJ`) fails at the masked simulator's outcome (`ZkSimAbort`). -/
def ZkSimAbortAt (x : Claims.Msg) : Prop :=
  ∀ (s : Claims.Flock.Stmt) (hs : parse true Readable.relation x = some s) (hin : Claims.Flock.InScope true s)
    (ins : Ins) (spec : SessionSpec) (sts : Array Stmt) (hset : setupOf true x = some (ins, spec, sts))
    (h0 : 0 < sts.size) (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    (d : Lean.Json) (_ : ins.read.draw = some d) (Ω : Type) (P : Claims.Prover Ω) (ω : Ω) (j : ℕ)
    (_ : j < (tables true x).length) (C : Fin (wordLenJ true x j) → List 𝔽), (∀ p, wideJ true x j (C p)) →
    ZkSimAbort P x ω spec.mPts (fun _ => (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length) zkN spec.dg
      (zkW (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length spec (zk_mPts_le hs hin hset))
      (zkSim P x spec sts) (zkΦ x spec sts) (AccTxJ ins.read.I d j (zkRoot P x ω) C)

end Gen

open Classical in
/-- **The masked simulator's fork advantages**, `ZkSimWJ`'s fork clauses' left sides verbatim at the masked simulator:
at event `i` of table `j`, the mean over the tape of `advRSZ` (a cap), `advPadsSZ` (the pads), `advTauSZ` (`τ`) or
`adv₀SZ` (level 0) after `Commit` at the deployed decoders right at `OkOffWJ`, against the simulated strategy at
`zkSim`'s live strategy. -/
noncomputable def zkAdvSim {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse true Readable.relation x = some s)
    (hin : Claims.Flock.InScope true s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    {d : Lean.Json} (hd : ins.read.draw = some d) {Ω : Type} (P : Claims.Prover Ω) (ω : Ω) (j : ℕ)
    (hj : j < (tables true x).length) (C : Fin (wordLenJ true x j) → List 𝔽) :
    Claims.Flock.IZW (zkTabs h0 L hz hk hlay j).sch.levels.length → ℝ
  | .inl (.inl (r, ℓ)) => (∑ u, advRSZ execArith ZkExec.hs512 (zkTabs h0 L hz hk hlay j) _ _
      (Exec.capOfRoot (zkRoot P x ω)[j]!) r.val (ℓ.val + 1)
      (simAtJZ execArith ZkExec.hs512 _ (OkOffWJ (tabsJ _ (zkTabs h0 L hz hk hlay)) j (zkRoot P x ω) C)
        (AccTxJ ins.read.I d j (zkRoot P x ω) C) (opnD sts _) spec.dg (decD _) (fun _ _ => true) (fun _ _ _ => [])
        (fun rs j => Exec.capOfRoot rs[j.val]!) (zkN x)
        (decodesTableW_deploy hs hin hset h0 L hz hk hlay hd j (zkRoot P x ω) C) (zkRoot P x ω)
        (liveStrat (fun _ => (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length)
          (fun _ => spec.mPts) zkN spec.dg
          (zkW (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length spec (zk_mPts_le hs hin hset)) sha512
          (zkSim P x spec sts) x (ω, u))
        (zkSim_root P x spec sts ω _ (zk_mPts_le hs hin hset) u) (finJ (zkTabs h0 L hz hk hlay) hj)
        (tabsJ_get _ _ _))) / Fintype.card (zkTape x)
  | .inl (.inr (.inl r)) => (∑ u, advPadsSZ execArith ZkExec.hs512 (zkTabs h0 L hz hk hlay j) _ _
      (Exec.capOfRoot (zkRoot P x ω)[j]!) r.val
      (simAtJZ execArith ZkExec.hs512 _ (OkOffWJ (tabsJ _ (zkTabs h0 L hz hk hlay)) j (zkRoot P x ω) C)
        (AccTxJ ins.read.I d j (zkRoot P x ω) C) (opnD sts _) spec.dg (decD _) (fun _ _ => true) (fun _ _ _ => [])
        (fun rs j => Exec.capOfRoot rs[j.val]!) (zkN x)
        (decodesTableW_deploy hs hin hset h0 L hz hk hlay hd j (zkRoot P x ω) C) (zkRoot P x ω)
        (liveStrat (fun _ => (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length)
          (fun _ => spec.mPts) zkN spec.dg
          (zkW (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length spec (zk_mPts_le hs hin hset)) sha512
          (zkSim P x spec sts) x (ω, u))
        (zkSim_root P x spec sts ω _ (zk_mPts_le hs hin hset) u) (finJ (zkTabs h0 L hz hk hlay) hj)
        (tabsJ_get _ _ _))) / Fintype.card (zkTape x)
  | .inl (.inr (.inr r)) => (∑ u, advTauSZ execArith ZkExec.hs512 (zkTabs h0 L hz hk hlay j) _ _
      (Exec.capOfRoot (zkRoot P x ω)[j]!) r.val
      (simAtJZ execArith ZkExec.hs512 _ (OkOffWJ (tabsJ _ (zkTabs h0 L hz hk hlay)) j (zkRoot P x ω) C)
        (AccTxJ ins.read.I d j (zkRoot P x ω) C) (opnD sts _) spec.dg (decD _) (fun _ _ => true) (fun _ _ _ => [])
        (fun rs j => Exec.capOfRoot rs[j.val]!) (zkN x)
        (decodesTableW_deploy hs hin hset h0 L hz hk hlay hd j (zkRoot P x ω) C) (zkRoot P x ω)
        (liveStrat (fun _ => (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length)
          (fun _ => spec.mPts) zkN spec.dg
          (zkW (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length spec (zk_mPts_le hs hin hset)) sha512
          (zkSim P x spec sts) x (ω, u))
        (zkSim_root P x spec sts ω _ (zk_mPts_le hs hin hset) u) (finJ (zkTabs h0 L hz hk hlay) hj)
        (tabsJ_get _ _ _))) / Fintype.card (zkTape x)
  | .inr () => (∑ u, adv₀SZ execArith ZkExec.hs512 (zkTabs h0 L hz hk hlay j) _ _
      (Exec.capOfRoot (zkRoot P x ω)[j]!)
      (simAtJZ execArith ZkExec.hs512 _ (OkOffWJ (tabsJ _ (zkTabs h0 L hz hk hlay)) j (zkRoot P x ω) C)
        (AccTxJ ins.read.I d j (zkRoot P x ω) C) (opnD sts _) spec.dg (decD _) (fun _ _ => true) (fun _ _ _ => [])
        (fun rs j => Exec.capOfRoot rs[j.val]!) (zkN x)
        (decodesTableW_deploy hs hin hset h0 L hz hk hlay hd j (zkRoot P x ω) C) (zkRoot P x ω)
        (liveStrat (fun _ => (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length)
          (fun _ => spec.mPts) zkN spec.dg
          (zkW (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length spec (zk_mPts_le hs hin hset)) sha512
          (zkSim P x spec sts) x (ω, u))
        (zkSim_root P x spec sts ω _ (zk_mPts_le hs hin hset) u) (finJ (zkTabs h0 L hz hk hlay) hj)
        (tabsJ_get _ _ _))) / Fintype.card (zkTape x)

namespace Gen

variable [S : ZkSetup]

open Classical in
/-- **The masked simulator's fork advantages**, `ZkSimWJ`'s fork clauses' left sides verbatim at the masked simulator:
at event `i` of table `j`, the mean over the tape of `advRSZ` (a cap), `advPadsSZ` (the pads), `advTauSZ` (`τ`) or
`adv₀SZ` (level 0) after `Commit` at the deployed decoders right at `OkOffWJ`, against the simulated strategy at
`zkSim`'s live strategy. -/
noncomputable def zkAdvSim {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse true Readable.relation x = some s)
    (hin : Claims.Flock.InScope true s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    {d : Lean.Json} (hd : ins.read.draw = some d) {Ω : Type} (P : Claims.Prover Ω) (ω : Ω) (j : ℕ)
    (hj : j < (tables true x).length) (C : Fin (wordLenJ true x j) → List 𝔽) :
    Claims.Flock.IZW (zkTabs h0 L hz hk hlay j).sch.levels.length → ℝ
  | .inl (.inl (r, ℓ)) => (∑ u, advRSZ execArith ZkExec.hs512 (zkTabs h0 L hz hk hlay j) _ _
      (Exec.capOfRoot (zkRoot P x ω)[j]!) r.val (ℓ.val + 1)
      (simAtJZ execArith ZkExec.hs512 _ (OkOffWJ (tabsJ _ (zkTabs h0 L hz hk hlay)) j (zkRoot P x ω) C)
        (AccTxJ ins.read.I d j (zkRoot P x ω) C) (opnD sts _) spec.dg (decD _) (fun _ _ => true) (fun _ _ _ => [])
        (fun rs j => Exec.capOfRoot rs[j.val]!) (zkN x)
        (decodesTableW_deploy hs hin hset h0 L hz hk hlay hd j (zkRoot P x ω) C) (zkRoot P x ω)
        (liveStrat (fun _ => (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length)
          (fun _ => spec.mPts) zkN spec.dg
          (zkW (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length spec (zk_mPts_le hs hin hset)) sha512
          (zkSim P x spec sts) x (ω, u))
        (zkSim_root P x spec sts ω _ (zk_mPts_le hs hin hset) u) (finJ (zkTabs h0 L hz hk hlay) hj)
        (tabsJ_get _ _ _))) / Fintype.card (zkTape x)
  | .inl (.inr (.inl r)) => (∑ u, advPadsSZ execArith ZkExec.hs512 (zkTabs h0 L hz hk hlay j) _ _
      (Exec.capOfRoot (zkRoot P x ω)[j]!) r.val
      (simAtJZ execArith ZkExec.hs512 _ (OkOffWJ (tabsJ _ (zkTabs h0 L hz hk hlay)) j (zkRoot P x ω) C)
        (AccTxJ ins.read.I d j (zkRoot P x ω) C) (opnD sts _) spec.dg (decD _) (fun _ _ => true) (fun _ _ _ => [])
        (fun rs j => Exec.capOfRoot rs[j.val]!) (zkN x)
        (decodesTableW_deploy hs hin hset h0 L hz hk hlay hd j (zkRoot P x ω) C) (zkRoot P x ω)
        (liveStrat (fun _ => (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length)
          (fun _ => spec.mPts) zkN spec.dg
          (zkW (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length spec (zk_mPts_le hs hin hset)) sha512
          (zkSim P x spec sts) x (ω, u))
        (zkSim_root P x spec sts ω _ (zk_mPts_le hs hin hset) u) (finJ (zkTabs h0 L hz hk hlay) hj)
        (tabsJ_get _ _ _))) / Fintype.card (zkTape x)
  | .inl (.inr (.inr r)) => (∑ u, advTauSZ execArith ZkExec.hs512 (zkTabs h0 L hz hk hlay j) _ _
      (Exec.capOfRoot (zkRoot P x ω)[j]!) r.val
      (simAtJZ execArith ZkExec.hs512 _ (OkOffWJ (tabsJ _ (zkTabs h0 L hz hk hlay)) j (zkRoot P x ω) C)
        (AccTxJ ins.read.I d j (zkRoot P x ω) C) (opnD sts _) spec.dg (decD _) (fun _ _ => true) (fun _ _ _ => [])
        (fun rs j => Exec.capOfRoot rs[j.val]!) (zkN x)
        (decodesTableW_deploy hs hin hset h0 L hz hk hlay hd j (zkRoot P x ω) C) (zkRoot P x ω)
        (liveStrat (fun _ => (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length)
          (fun _ => spec.mPts) zkN spec.dg
          (zkW (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length spec (zk_mPts_le hs hin hset)) sha512
          (zkSim P x spec sts) x (ω, u))
        (zkSim_root P x spec sts ω _ (zk_mPts_le hs hin hset) u) (finJ (zkTabs h0 L hz hk hlay) hj)
        (tabsJ_get _ _ _))) / Fintype.card (zkTape x)
  | .inr () => (∑ u, adv₀SZ execArith ZkExec.hs512 (zkTabs h0 L hz hk hlay j) _ _
      (Exec.capOfRoot (zkRoot P x ω)[j]!)
      (simAtJZ execArith ZkExec.hs512 _ (OkOffWJ (tabsJ _ (zkTabs h0 L hz hk hlay)) j (zkRoot P x ω) C)
        (AccTxJ ins.read.I d j (zkRoot P x ω) C) (opnD sts _) spec.dg (decD _) (fun _ _ => true) (fun _ _ _ => [])
        (fun rs j => Exec.capOfRoot rs[j.val]!) (zkN x)
        (decodesTableW_deploy hs hin hset h0 L hz hk hlay hd j (zkRoot P x ω) C) (zkRoot P x ω)
        (liveStrat (fun _ => (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length)
          (fun _ => spec.mPts) zkN spec.dg
          (zkW (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length spec (zk_mPts_le hs hin hset)) sha512
          (zkSim P x spec sts) x (ω, u))
        (zkSim_root P x spec sts ω _ (zk_mPts_le hs hin hset) u) (finJ (zkTabs h0 L hz hk hlay) hj)
        (tabsJ_get _ _ _))) / Fintype.card (zkTape x)

end Gen

open Classical in
/-- **The simulated fork bounds the model's rewinding advantage**: at every masked statement in scope that sets up at a
draw, every prover and tape, every table `j` and table of full-width columns, the masked simulator's fork advantage at
each event (`zkAdvSim`) is at most the simulated fork at `zkΦ` at the event's round (`simForkPr` at the live fork
`forkN`). -/
def ZkSimForkLe (x : Claims.Msg) : Prop :=
  ∀ (s : Claims.Flock.Stmt) (hs : parse true Readable.relation x = some s) (hin : Claims.Flock.InScope true s)
    (ins : Ins) (spec : SessionSpec) (sts : Array Stmt) (hset : setupOf true x = some (ins, spec, sts))
    (h0 : 0 < sts.size) (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    (d : Lean.Json) (hd : ins.read.draw = some d) (Ω : Type) (P : Claims.Prover Ω) (ω : Ω) (j : ℕ)
    (hj : j < (tables true x).length) (C : Fin (wordLenJ true x j) → List 𝔽), (∀ p, wideJ true x j (C p)) →
    ∀ i : Claims.Flock.IZW (zkTabs h0 L hz hk hlay j).sch.levels.length,
      zkAdvSim hs hin hset h0 L hz hk hlay hd P ω j hj C i ≤
        Claims.Flock.simForkPr sha512 (V true) P x (zkΦ x spec sts) (fun t => Claims.Flock.FnFork.forkN t) ω
          (zkRd hset h0 L hz hk hlay j i).val

namespace Gen

variable [S : ZkSetup]

open Classical in
/-- **The simulated fork bounds the model's rewinding advantage**: at every masked statement in scope that sets up at a
draw, every prover and tape, every table `j` and table of full-width columns, the masked simulator's fork advantage at
each event (`zkAdvSim`) is at most the simulated fork at `zkΦ` at the event's round (`simForkPr` at the live fork
`forkN`). -/
def ZkSimForkLe (x : Claims.Msg) : Prop :=
  ∀ (s : Claims.Flock.Stmt) (hs : parse true Readable.relation x = some s) (hin : Claims.Flock.InScope true s)
    (ins : Ins) (spec : SessionSpec) (sts : Array Stmt) (hset : setupOf true x = some (ins, spec, sts))
    (h0 : 0 < sts.size) (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    (d : Lean.Json) (hd : ins.read.draw = some d) (Ω : Type) (P : Claims.Prover Ω) (ω : Ω) (j : ℕ)
    (hj : j < (tables true x).length) (C : Fin (wordLenJ true x j) → List 𝔽), (∀ p, wideJ true x j (C p)) →
    ∀ i : Claims.Flock.IZW (zkTabs h0 L hz hk hlay j).sch.levels.length,
      zkAdvSim hs hin hset h0 L hz hk hlay hd P ω j hj C i ≤
        Claims.Flock.simForkPr sha512 (V true) P x (zkΦ x spec sts) (fun t => Claims.Flock.FnFork.forkN t) ω
          (zkRd hset h0 L hz hk hlay j i).val

end Gen

/-- **A self-clash at an accepting run's view is a column clash or a collision**: at every masked statement in scope
that sets up, at its draw `d`, every prover and tape, every table `j` and table of full-width columns, and every coins
at which `V true` accepts `P` deciding a record, a decoded self-clash of table `j` at the record's live outcome
(`LiveSelfClashZ` at `viewOf`) gives two different columns the run's reps open at one position of some table, or a
collision among the run's verifier calls. -/
def ZkViewClash (x : Claims.Msg) : Prop :=
  ∀ (s : Claims.Flock.Stmt) (_ : parse true Readable.relation x = some s) (_ : Claims.Flock.InScope true s)
    (ins : Ins) (spec : SessionSpec) (sts : Array Stmt) (_ : setupOf true x = some (ins, spec, sts))
    (h0 : 0 < sts.size) (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    (d : Lean.Json) (_ : ins.read.draw = some d) (Ω : Type) (P : Claims.Prover Ω) (ω : Ω) (j : ℕ)
    (_ : j < (tables true x).length) (C : Fin (wordLenJ true x j) → List 𝔽), (∀ p, wideJ true x j (C p)) →
    ∀ r : Fin ((V true).k x) → Coin, (Claims.run sha512 P (V true) x ω r).out = .acc →
    ∀ s' spec' sts' link c₁ mids f,
      act true x ((Claims.run sha512 P (V true) x ω r).msgs.map Prod.fst)
          ((List.range ((Claims.run sha512 P (V true) x ω r).msgs.map Prod.fst).length).map (Claims.coinSeq r)) =
        .decide s' spec' sts' link c₁ mids f →
      LiveSelfClashZ ZkExec.hs512 (zkTabs h0 L hz hk hlay j)
          (outJ ZkExec.hs512 (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)))
          (AccTxJ ins.read.I d j (zkRoot P x ω) C)
          (opnD sts _) (decD _) (fun _ _ => true) (fun _ _ _ => []) (fun rs j => Exec.capOfRoot rs[j.val]!)
          (repJ execArith ZkExec.hs512 (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay))) j
          (Claims.Bridge.Custody.viewOf (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length spec.mPts spec'
            link c₁ f) →
        (∃ (j' q : ℕ) (a b : List 𝔽), (q, a) ∈ zkRawColsJ (Claims.run sha512 P (V true) x ω r).seen j' ∧
          (q, b) ∈ zkRawColsJ (Claims.run sha512 P (V true) x ω r).seen j' ∧ a ≠ b) ∨
          Claims.TiedColl sha512 (Claims.run sha512 P (V true) x ω r).vLog

namespace Gen

variable [S : ZkSetup]

/-- **A self-clash at an accepting run's view is a column clash or a collision**: at every masked statement in scope
that sets up, at its draw `d`, every prover and tape, every table `j` and table of full-width columns, and every coins
at which `V true` accepts `P` deciding a record, a decoded self-clash of table `j` at the record's live outcome
(`LiveSelfClashZ` at `viewOf`) gives two different columns the run's reps open at one position of some table, or a
collision among the run's verifier calls. -/
def ZkViewClash (x : Claims.Msg) : Prop :=
  ∀ (s : Claims.Flock.Stmt) (_ : parse true Readable.relation x = some s) (_ : Claims.Flock.InScope true s)
    (ins : Ins) (spec : SessionSpec) (sts : Array Stmt) (_ : setupOf true x = some (ins, spec, sts))
    (h0 : 0 < sts.size) (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    (d : Lean.Json) (_ : ins.read.draw = some d) (Ω : Type) (P : Claims.Prover Ω) (ω : Ω) (j : ℕ)
    (_ : j < (tables true x).length) (C : Fin (wordLenJ true x j) → List 𝔽), (∀ p, wideJ true x j (C p)) →
    ∀ r : Fin ((V true).k x) → Coin, (Claims.run sha512 P (V true) x ω r).out = .acc →
    ∀ s' spec' sts' link c₁ mids f,
      act true x ((Claims.run sha512 P (V true) x ω r).msgs.map Prod.fst)
          ((List.range ((Claims.run sha512 P (V true) x ω r).msgs.map Prod.fst).length).map (Claims.coinSeq r)) =
        .decide s' spec' sts' link c₁ mids f →
      LiveSelfClashZ ZkExec.hs512 (zkTabs h0 L hz hk hlay j)
          (outJ ZkExec.hs512 (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)))
          (AccTxJ ins.read.I d j (zkRoot P x ω) C)
          (opnD sts _) (decD _) (fun _ _ => true) (fun _ _ _ => []) (fun rs j => Exec.capOfRoot rs[j.val]!)
          (repJ execArith ZkExec.hs512 (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay))) j
          (Claims.Bridge.Custody.viewOf (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length spec.mPts spec'
            link c₁ f) →
        (∃ (j' q : ℕ) (a b : List 𝔽), (q, a) ∈ zkRawColsJ (Claims.run sha512 P (V true) x ω r).seen j' ∧
          (q, b) ∈ zkRawColsJ (Claims.run sha512 P (V true) x ω r).seen j' ∧ a ≠ b) ∨
          Claims.TiedColl sha512 (Claims.run sha512 P (V true) x ω r).vLog

end Gen

/-! ## The masked chain from the four -/

namespace Gen

variable [S : ZkSetup]

theorem zk_m_range {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse true Readable.relation x = some s)
    (hin : Claims.Flock.InScope true s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size) (j : ℕ) (hj : j < (tables true x).length) :
    25 ≤ (stJ h0 j).m ∧ (stJ h0 j).m ≤ 35 := by
  have hj' : j < sts.size := by rw [← zk_tables_length hset]; exact hj
  simp only [stJ, dite_eq_left hj']
  exact ⟨(zk_scopeJ hs hin hset hj').1, (zk_scopeJ hs hin hset hj').2.1⟩

end Gen

theorem zk_m_range {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse true Readable.relation x = some s)
    (hin : Claims.Flock.InScope true s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size) (j : ℕ) (hj : j < (tables true x).length) :
    25 ≤ (stJ h0 j).m ∧ (stJ h0 j).m ≤ 35 :=
  Gen.zk_m_range (S := S₀) (x := x) (s := s) (hs := hs) (hin := hin) (ins := ins) (spec := spec) (sts := sts)
    (hset := hset) (h0 := h0) (j := j) (hj := hj)

namespace Gen

variable [S : ZkSetup]

theorem zk_sch_m1 {x : Claims.Msg} {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    (j : ℕ) (hj : j < (tables true x).length) :
    (zkTabs h0 L hz hk hlay j).sch = Accounting.m1Schedule (stJ h0 j).m := by
  have hj' : j < sts.size := by rw [← zk_tables_length hset]; exact hj
  rw [zkTabs_of_lt h0 L hz hk hlay hj']
  simp only [stJ, dite_eq_left hj']
  exact (zkTab_m1 (L j hj') (hz j hj') (hk j hj') spec (hlay j hj')).1

end Gen

theorem zk_sch_m1 {x : Claims.Msg} {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    (j : ℕ) (hj : j < (tables true x).length) :
    (zkTabs h0 L hz hk hlay j).sch = Accounting.m1Schedule (stJ h0 j).m :=
  Gen.zk_sch_m1 (S := S₀) (x := x) (ins := ins) (spec := spec) (sts := sts) (hset := hset) (h0 := h0) (L := L)
    (hz := hz) (hk := hk) (hlay := hlay) (j := j) (hj := hj)

namespace Gen

variable [S : ZkSetup]

open Classical in
/-- **The masked simulator meets `ZkSimWJ`** wherever the four hold: the fork rounds `zkRd`, the chances
`δN GF128 65536 (zkNOf …)` and their fit (`zk_slackFitW_δN`), and at each tape the masked simulator at `zkΦ`'s uniform
coins sending `P`'s roots; the simulation is `ZkSimRunAt`, the self-clash from the abort, the lockstep and the view's
clash (`zkClashRun_of_raw`), and each fork clause `ZkSimForkLe` then the slack (`zk_simFork_le`). -/
theorem zkSimWJ_of_parts {x : Claims.Msg} (hrun : ZkSimRunAt x) (habort : ZkSimAbortAt x) (hfork : ZkSimForkLe x)
    (hclash : ZkViewClash x) : ZkSimWJ x := by
  intro s hs hin ins spec sts hset h0 L hz hk hlay d hd Ω _ _ P
  refine ⟨zkRd hset h0 L hz hk hlay, zkRd_inj hset h0 L hz hk hlay, zkRd_zero hset h0 L hz hk hlay,
    Claims.Flock.FnFork.δN 𝔽 65536 (zkNOf x spec sts),
    fun t => Claims.Flock.FnFork.δN_nonneg («X» := 𝔽) (M := 65536) (zkNOf x spec sts) t,
    zk_slackFitW_δN (n := (tables true x).length) (stJ h0) (zk_m_range hs hin hset h0)
      (fun j => (zkTabs h0 L hz hk hlay j).sch) (zk_sch_m1 hset h0 L hz hk hlay) (fun j => (zkTabs h0 L hz hk hlay j).N)
      (zkRd hset h0 L hz hk hlay) (fun j _ i => zkRd_succ_lt hset h0 L hz hk hlay j i) (zkNOf x spec sts)
      (fun t _ => zkNOf_pos x spec sts t),
    fun ω => ⟨zkN, spec.dg,
      zkW (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length spec (zk_mPts_le hs hin hset), zkTape x,
      inferInstance, inferInstance, zkSim P x spec sts, zkΦ x spec sts, Pr_zkΦ (zkN_add hset),
      fun u => zkSim_root P x spec sts ω _ (zk_mPts_le hs hin hset) u,
      hrun s hs hin ins spec sts hset h0 L hz hk hlay Ω P ω, fun j hj C hC => ⟨?_,
      fun r ℓ => (hfork s hs hin ins spec sts hset h0 L hz hk hlay d hd Ω P ω j hj C hC
        (.inl (.inl (r, ℓ)))).trans (zk_simFork_le P ω hset _),
      fun r => (hfork s hs hin ins spec sts hset h0 L hz hk hlay d hd Ω P ω j hj C hC
        (.inl (.inr (.inl r)))).trans (zk_simFork_le P ω hset _),
      fun r => (hfork s hs hin ins spec sts hset h0 L hz hk hlay d hd Ω P ω j hj C hC
        (.inl (.inr (.inr r)))).trans (zk_simFork_le P ω hset _),
      (hfork s hs hin ins spec sts hset h0 L hz hk hlay d hd Ω P ω j hj C hC (.inr ())).trans
        (zk_simFork_le P ω hset _)⟩⟩⟩
  refine zkClashRun_of_raw fun u r' hp => ?_
  obtain ⟨a, ⟨hpL, -⟩, -⟩ := id hp
  have hacc : (Claims.run sha512 P (V true) x ω (zkΦ x spec sts (u, r'))).out = .acc := by
    by_contra hn
    exact habort s hs hin ins spec sts hset h0 L hz hk hlay d hd Ω P ω j hj C hC u r' hn hpL
  obtain ⟨s', spec', sts', link, c₁, mids, f, hact, -⟩ := acc_decide hacc
  have hv := hrun s hs hin ins spec sts hset h0 L hz hk hlay Ω P ω u r' hacc s' spec' sts' link c₁ mids f hact
  rw [hv] at hp
  exact ⟨hacc, hclash s hs hin ins spec sts hset h0 L hz hk hlay d hd Ω P ω j hj C hC _ hacc s' spec' sts' link c₁
    mids f hact hp⟩

end Gen

open Classical in
/-- **The masked simulator meets `ZkSimWJ`** wherever the four hold: the fork rounds `zkRd`, the chances
`δN GF128 65536 (zkNOf …)` and their fit (`zk_slackFitW_δN`), and at each tape the masked simulator at `zkΦ`'s uniform
coins sending `P`'s roots; the simulation is `ZkSimRunAt`, the self-clash from the abort, the lockstep and the view's
clash (`zkClashRun_of_raw`), and each fork clause `ZkSimForkLe` then the slack (`zk_simFork_le`). -/
theorem zkSimWJ_of_parts {x : Claims.Msg} (hrun : ZkSimRunAt x) (habort : ZkSimAbortAt x) (hfork : ZkSimForkLe x)
    (hclash : ZkViewClash x) : ZkSimWJ x :=
  Gen.zkSimWJ_of_parts (S := S₀) (x := x) (hrun := hrun) (habort := habort) (hfork := hfork) (hclash := hclash)

/-- **The masked record the claims read: KS**, at SHA-512 at the deployed relation, at every number of tables, from
the masked simulator's lockstep, abort, fork bounds and view clashes at the statements some run accepts
(`zk_KS_trim_of_simWJ`). -/
theorem zk_KS_of_parts (hrun : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkSimRunAt x)
    (habort : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkSimAbortAt x)
    (hfork : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkSimForkLe x)
    (hclash : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkViewClash x) :
    Claims.Flock.KS sha512 (deployZkTrim Readable.relation) :=
  zk_KS_trim_of_simWJ fun x hno => zkSimWJ_of_parts (hrun x hno) (habort x hno) (hfork x hno) (hclash x hno)

namespace Gen

variable [S : ZkSetup]

/-- **The masked record the claims read: KS**, at SHA-512 at the deployed relation, at every number of tables, from
the masked simulator's lockstep, abort, fork bounds and view clashes at the statements some run accepts
(`zk_KS_trim_of_simWJ`). -/
theorem zk_KS_of_parts (hrun : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkSimRunAt x)
    (habort : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkSimAbortAt x)
    (hfork : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkSimForkLe x)
    (hclash : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkViewClash x) :
    Claims.Flock.KS sha512 (deployZkTrim Readable.relation) :=
  zk_KS_trim_of_simWJ fun x hno => zkSimWJ_of_parts (hrun x hno) (habort x hno) (hfork x hno) (hclash x hno)

end Gen

end FlockDeploy.KS

namespace FlockDeploy.KS

open Flock
open Claims.Recursion

/-- **Every masked table's bound with escapes on accepted runs counted per round**, at full-width columns, from the
four at the statements some run accepts (`zk_tableBoundJOnAccR_of_simWJ`): `zk_KSFlockJR`'s premise. -/
theorem zk_tableBoundJOnAccR_of_parts (hrun : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkSimRunAt x)
    (habort : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkSimAbortAt x)
    (hfork : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkSimForkLe x)
    (hclash : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkViewClash x) :
    TableBoundJOnAccR sha512 (deployZk Readable.relation) (wordLenJ true) (wideJ true) :=
  zk_tableBoundJOnAccR_of_simWJ fun x hno => zkSimWJ_of_parts (hrun x hno) (habort x hno) (hfork x hno) (hclash x hno)

/-- **KS of the recursion over the masked record the claims read** at the deployed relation, at every number of
tables, from the four at the statements some run accepts (`zk_KSFlockJR` at `zk_tableBoundJOnAccR_of_parts`). -/
theorem zk_KSFlock_of_parts (D : Deploy Coin)
    (DO : Claims.Flock.Deploy Coin (List FlockLevel3.GF128) Claims.Flock.ZkTrace)
    (opening : Claims.Recursion.Opening) (hF : Claims.Flock.KS sha512 DO)
    (hrun : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkSimRunAt x)
    (habort : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkSimAbortAt x)
    (hfork : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkSimForkLe x)
    (hclash : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkViewClash x) (hRef : Refuses sha512 D)
    (hRounds : ∀ x, (deployZkTrim Readable.relation).V.k (xI D x) ≤ D.V.k x) (hN : SessionNums D DO)
    (hε : TableErrNonneg D) (hRd : ∀ decI, SessionRounds D (Inner.ofFlock (deployZkTrim Readable.relation) decI) DO)
    (hS : SessionAccepts D) (hR : ∀ decI, SessionRuns sha512 D (Inner.ofFlock (deployZkTrim Readable.relation) decI) DO)
    (hV : ∀ decI, InnerViewRun sha512 D (Inner.ofFlock (deployZkTrim Readable.relation) decI) DO)
    (hT : ∀ decI, CallsTrue sha512 D (Inner.ofFlock (deployZkTrim Readable.relation) decI) DO)
    (hL : ∀ decI, CallsLogColl sha512 D (Inner.ofFlock (deployZkTrim Readable.relation) decI) DO)
    (hO : ∀ decI, CallsOpenPlayed sha512 D (Inner.ofFlock (deployZkTrim Readable.relation) decI) DO opening)
    (hAcc : ∀ decI, InnerAccepts sha512 D (Inner.ofFlock (deployZkTrim Readable.relation) decI) DO)
    (hN0 : CallsNil sha512 D) :
    KSFlock sha512 D (deployZkTrim Readable.relation) DO :=
  zk_KSFlockJR D DO opening hF (zk_tableBoundJOnAccR_of_parts hrun habort hfork hclash) hRef hRounds hN hε hRd hS hR hV
    hT hL hO hAcc hN0

end FlockDeploy.KS
