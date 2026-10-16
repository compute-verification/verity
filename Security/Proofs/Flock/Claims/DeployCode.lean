import Proofs.Flock.Claims.Deploy
import Proofs.Flock.Claims.ZkCols
import Proofs.Flock.Claims.DecodeFlock
import Proofs.Flock.Claims.NumbersGamma
import Proofs.Flock.Claims.NumbersNMin
import Proofs.Flock.Claims.NumbersTable

/-!
# Flock's deployed instances of `Claims.Flock.Deploy`

`FlockDeploy.deployPlain rel` and `FlockDeploy.deployZk rel` are the shipped verifier, plain and masked (`--zk`), as
the record Flock's claims read, at the relation's fields `rel` (C's `Readable.Relation`):

* `V`, `parse`, `scopeCheck`: the verifier in public-coin form (`FlockDeploy.V`), the verifier's own parser and scope
  check on the statement's bytes (`FlockDeploy.parse`, `FlockDeploy.scopeCheck`).
* `cols`: the opened level-0 columns of table 0's two reps, read off the run's last message (its statement, record and
  proof files) with no hash: the record's rep streams give the level-0 query round's coins, the proof files the opened
  rows (`runOf`, `Claims.Flock.plainRunCols`, `Claims.Flock.zkRunCols`).
* `enc`, `nMin`, `read`: the level-0 code at each `m` the verifier serves, read as one (`IRS.famEnc`): the plain one
  (`plainEncD`, `22 ≤ m ≤ 35`) and the masked one (`zkEncD`, `25 ≤ m ≤ 35`), each with its Johnson count at `agree`
  and its read step. A decoder meeting `ListComplete` exists for each (`listComplete_plain`, `listComplete_zk`).
* `agree`: `√ρ + 1/200`, at rate `1/2` without `--zk` (`Numbers.agree₀`) and at the masked code's largest rate
  `(2^12 + 554)/2^13` (`m = 25`) with it (`zkAgree`, `zkRate_le`).
* `εtable`: the tables' errors summed: `tableError` at the table's `fast100` schedule, or under `--zk` the padded
  table's `padLinkError + (padRepError + ε_inner)²` at `m1Schedule` with `ε_inner`'s bound at the inner proof's
  parameters (`Numbers.innerUp`). `N₀`: table 0's level-0 codeword length `2^{d₀}`. `weight`: the tables' weights
  summed (`Numbers.weightOf`, at the pads' domain `Numbers.padDomain` under `--zk`). `waits`, `cap`: `Numbers.waits`
  (`210`) and `Numbers.cap` (`2^256`). A statement that does not set up has no tables: `εtable`, `N₀` and `weight` are
  `0` there, where the verifier refuses every run.

One run opens fewer level-0 columns than `nMin` of every trace (`cols_length_lt_nMin_plain`, `_zk`): ruling 2's count
is reached only by `E^rep`'s columns, gathered over its accepted reruns.
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

/-- **The deployed plain level-0 code at index `i`**: #2104's `levelZero` in the executable's arithmetic, `2^k0` lanes
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

/-! ## A plain run's opened level-0 columns -/

/-- **The round of level 0's query coins** in a plain rep's stream: after zerocheck's `m − 3` rounds, lincheck's
`kLog − 3`, the ring switch's `claims + 1` and Ligerito's `k0 + 6` before level 0's queries (the out-of-domain point and
its coin, the `k0` lane challenges, and level 1's two OOD intros of two rounds each). -/
def plainQueryRound (m kLog k0 claims : ℕ) : ℕ := (m - 3) + (kLog - 3) + (claims + 1) + (k0 + 6)

/-- The plain verifier's proof shape for a table (`verifyRep`'s `shape`). -/
def plainProofShape (m kLog claims hashLen : ℕ) (saltLen : Option ℕ) : _root_.Flock.Shape :=
  { m, kLog, claims, sched := plainSched m, hashLen, saltLen }

/-- **A plain rep's opened level-0 columns** from its recorded rounds and proof bytes: none if the proof does not
decode. -/
def plainRepColsOf (sh : _root_.Flock.Shape) (rounds : Array _root_.Flock.Round) (bytes : ByteArray) :
    List (ℕ × List 𝔽) :=
  match _root_.Flock.decodeProof sh bytes with
  | .ok p =>
    zkRepCols (sh.sched.levels.getD 0 default)
      ((rounds[plainQueryRound sh.m sh.kLog sh.sched.k0 sh.claims]?.map (·.coins)).getD #[])
      p.ligerito.initialProof.rows
  | .error _ => []

/-- A run's plain proof shape. -/
def ZkRun.plainShape (r : ZkRun) : _root_.Flock.Shape := plainProofShape r.m r.kLog r.claims r.hashLen r.saltLen

/-- **A plain run's opened level-0 columns** for one table: its reps' columns, one per position, below `2^d`. -/
def plainRunCols (r : ZkRun) : List (ℕ × List 𝔽) :=
  distinctBelow (plainLevel0 r.m).d (r.reps.flatMap fun x => plainRepColsOf r.plainShape x.1 x.2)

theorem length_plainRepColsOf_le (sh : _root_.Flock.Shape) (rounds : Array _root_.Flock.Round) (bytes : ByteArray) :
    (plainRepColsOf sh rounds bytes).length ≤ (sh.sched.levels.getD 0 default).q := by
  unfold plainRepColsOf
  split
  · exact length_zkRepCols_le _ _ _
  · exact Nat.zero_le _

/-- **A plain run's columns sit at distinct positions.** -/
theorem plainRunCols_nodup (r : ZkRun) : ((plainRunCols r).map Prod.fst).Nodup := distinctBelow_nodup _ _

/-- **A plain run's columns sit below `2^d`.** -/
theorem plainRunCols_lt (r : ZkRun) : ∀ p ∈ plainRunCols r, p.1 < 2 ^ (plainLevel0 r.m).d :=
  fun _ h => distinctBelow_lt h

/-- **A plain run opens at most `q₀` level-0 columns per rep.** -/
theorem plainRunCols_length_le (r : ZkRun) : (plainRunCols r).length ≤ r.reps.length * (plainLevel0 r.m).q := by
  refine (length_distinctBelow_le _ _).trans ?_
  rw [List.length_flatMap]
  refine (List.sum_le_length_nsmul _ (plainLevel0 r.m).q fun n hn => ?_).trans (by simp)
  obtain ⟨x, -, rfl⟩ := List.mem_map.1 hn
  exact length_plainRepColsOf_le r.plainShape x.1 x.2

/-- Level 0 makes at most `218` queries at every `m` (none where `fast100` refuses `m`). -/
theorem plainLevel0_q_le (m : ℕ) : (plainLevel0 m).q ≤ 218 := by
  by_cases hm : 22 ≤ m ∧ m ≤ 35
  · obtain ⟨i, rfl⟩ : ∃ i : PlainIdx, i.m = m := ⟨⟨m - 22, by omega⟩, by simp only [PlainIdx.m]; omega⟩
    exact (plainShape_table i).2.2.2.2.2.2.le
  · have h : _root_.Flock.fast100 m = none := by
      unfold _root_.Flock.fast100
      have : (decide (m < 22) || decide (m > 35)) = true := by
        simp only [Bool.or_eq_true, decide_eq_true_eq]
        omega
      simp [this]
    simp [plainLevel0, plainSched, h]
    decide

/-- **One plain run's two reps open fewer level-0 columns than the Johnson count of every trace**, at
`0 < agree ≤ 1`: at most `2·218`, against at least `2^9 + 2`. -/
theorem plainRunCols_length_lt_plainNMinD {agree : ℝ} (ha0 : 0 < agree) (ha1 : agree ≤ 1) (r : ZkRun)
    (h2 : r.reps.length ≤ 2) (M : PlainTrace) : (plainRunCols r).length < plainNMinD agree M := by
  have hl := (plainRunCols_length_le r).trans (Nat.mul_le_mul h2 (plainLevel0_q_le r.m))
  have hj := lt_johnsonCount (k := 2 ^ plainCols M.1) ha0 ha1
  have hc : 2 ^ 9 ≤ 2 ^ plainCols M.1 := Nat.pow_le_pow_right (by norm_num) (plainShape_table M.1).2.2.2.2.2.1
  rw [plainNMinD_eq]
  omega

/-! ## The masked code's threshold and query bound -/

/-- The masked level-0 rate is largest at `m = 25`: `(2^cols + t)/2^d ≤ (2^12 + 554)/2^13` at every `m` served. -/
theorem zkRate_le : ∀ i : ZkIdx, (2 ^ zkCols i + zkPad i) * 2 ^ zkLen 0 ≤ (2 ^ zkCols 0 + zkPad 0) * 2 ^ zkLen i := by
  decide +kernel

theorem zkShape_zero : zkCols 0 = 12 ∧ zkPad 0 = 554 ∧ zkLen 0 = 13 := by
  decide +kernel

/-- **The masked decoding threshold**: `√ρ' + 1/200` at the masked code's largest rate, `ρ' = (2^12 + 554)/2^13`
(`m = 25`, `zkRate_le`). -/
noncomputable def zkAgree : ℝ := Real.sqrt (((2 ^ zkCols 0 + zkPad 0 : ℕ) : ℝ) / 2 ^ zkLen 0) + 1 / 200

theorem zkAgree_eq : zkAgree = Real.sqrt (4650 / 8192) + 1 / 200 := by
  obtain ⟨h1, h2, h3⟩ := zkShape_zero
  simp only [zkAgree, h1, h2, h3]
  norm_num

theorem zkAgree_pos : 0 < zkAgree := by
  unfold zkAgree
  positivity

theorem zkAgree_le_one : zkAgree ≤ 1 := by
  rw [zkAgree_eq]
  have : Real.sqrt (4650 / 8192) ≤ 0.995 := by
    rw [Real.sqrt_le_left (by norm_num)]
    norm_num
  linarith

/-- The masked level-0 code's dimension is at least `4650` at every `m` served. -/
theorem zkDim_ge : ∀ i : ZkIdx, 4650 ≤ 2 ^ zkCols i + zkPad i := by
  decide +kernel

/-- The masked level 0 makes at most `277` queries at every `m` (none where `Zk.schedule` refuses `m`). -/
theorem zkLevel0_q_le (m : ℕ) : (zkLevel0 m).q ≤ 277 := by
  by_cases hm : 25 ≤ m ∧ m ≤ 35
  · obtain ⟨i, rfl⟩ : ∃ i : ZkIdx, i.m = m := ⟨⟨m - 25, by omega⟩, by simp only [ZkIdx.m]; omega⟩
    have h : ∀ i : ZkIdx, (zkLevel0 i.m).q ≤ 277 := by decide +kernel
    exact h i
  · have hq : ∃ e, _root_.Flock.Zk.queries0 m = .error e := by
      unfold _root_.Flock.Zk.queries0
      split_ifs <;> first | exact ⟨_, rfl⟩ | (exfalso; simp only [beq_iff_eq] at *; omega)
    obtain ⟨e, he⟩ := hq
    have h : _root_.Flock.Zk.schedule m = .error e := by
      unfold _root_.Flock.Zk.schedule
      rw [he]
      rfl
    simp only [zkLevel0, zkSched, h]
    decide

/-- **One masked run's two reps open fewer level-0 columns than the Johnson count of every trace**, at
`0 < agree ≤ 1`: at most `2·277`, against at least `4650 + 2`. -/
theorem zkRunCols_length_lt_zkNMinD' {agree : ℝ} (ha0 : 0 < agree) (ha1 : agree ≤ 1) (r : ZkRun)
    (h2 : r.reps.length ≤ 2) (M : ZkTrace) : (zkRunCols r).length < zkNMinD agree M := by
  have hl := (zkRunCols_length_le r).trans (Nat.mul_le_mul h2 (zkLevel0_q_le r.m))
  have hj := lt_johnsonCount (k := 2 ^ zkCols M.1 + zkPad M.1) ha0 ha1
  have hc := zkDim_ge M.1
  rw [zkNMinD_eq]
  omega

end Claims.Flock

namespace FlockDeploy

open Flock
open FlockSoundness

local notation "𝔽" => FlockLevel3.GF128

/-! ## The columns a run opens -/

/-- The run's last message, read as the prover's last message. -/
def lastFinal (s : Claims.Seen) : Option Final := s.msgs.getLast?.bind fun p => finalOf p.1

/-- The Merkle salt width a table's proofs carry (`verifyRep`'s `saltLen`; the masked verifier's opens no salts). -/
def saltLenOf (zk : Bool) (tab : Setup) : Option ℕ :=
  if zk then (if tab.merkle.saltBytes = 0 then none else some tab.merkle.saltBytes)
  else if tab.merkle.saltBytes = 0 && !tab.openedSalts then none else some tab.merkle.saltBytes

/-- **What the column reader takes of a run**, with no hash: from the last message's statement, record and proof files,
table 0's `m`, `kLog`, claim count and Merkle widths (its setup's), and its two reps' recorded rounds (the record's
streams `<t>/rep0`, `<t>/rep1`, the first two in `spec.repStreams` order) and proof files (the first two). -/
def runOf (zk : Bool) (s : Claims.Seen) : Option Claims.Flock.ZkRun := do
  let f ← lastFinal s
  let (_, spec, sts) ← setupOf zk f.stmt
  let st ← sts[0]?
  let tab := Setup.ofCircuit st
  let lv ← (liveOf spec f.record).toOption
  let extras ← (tab.extra lv.points).toOption
  pure { m := tab.m, kLog := tab.kLog, claims := 2 + extras.size, hashLen := tab.merkle.digestLen,
         saltLen := saltLenOf zk tab,
         reps := [(lv.streams.getD 0 #[], f.proofs.getD 0 ByteArray.empty),
                  (lv.streams.getD 1 #[], f.proofs.getD 1 ByteArray.empty)] }

/-- **The opened level-0 columns of a run**: table 0's two reps' (`plainRunCols`, `zkRunCols`); none where the run's
last message does not set up. -/
def cols (zk : Bool) (s : Claims.Seen) : List (ℕ × List 𝔽) :=
  match runOf zk s with
  | some r => if zk then Claims.Flock.zkRunCols r else Claims.Flock.plainRunCols r
  | none => []

theorem runOf_reps {zk : Bool} {s : Claims.Seen} {r : Claims.Flock.ZkRun} (h : runOf zk s = some r) :
    r.reps.length = 2 := by
  unfold runOf at h
  simp only [Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def, Option.some.injEq] at h
  obtain ⟨_, _, ⟨_, _, _⟩, _, _, _, _, _, _, _, rfl⟩ := h
  rfl

theorem cols_nodup (zk : Bool) (s : Claims.Seen) : ((cols zk s).map Prod.fst).Nodup := by
  unfold cols
  split
  · split
    · exact Claims.Flock.zkRunCols_nodup _
    · exact Claims.Flock.plainRunCols_nodup _
  · exact List.nodup_nil

/-! ## The numbers -/

/-- The session's tables with its spec; none where the statement does not set up. -/
def tables (zk : Bool) (x : Claims.Msg) : List (SessionSpec × Stmt) :=
  match setupOf zk x with
  | some (_, spec, sts) => sts.toList.map fun st => (spec, st)
  | none => []

/-- A table's shape as the soundness analysis reads it: `k_log`, its public regions and the link points' coordinates. -/
def acct (spec : SessionSpec) (st : Stmt) : Accounting.Stmt := ⟨st.c.kLog, st.regions.size, spec.mPts⟩

/-- **A table's error**: `tableError` at its `fast100` schedule; under `--zk`, the padded table's
`padLinkError + (padRepError + ε_inner)²` at `m1Schedule` and padding `m1Pad`, two reps, with the inner proof's error
at its parameters bounded by `Numbers.innerUp`. -/
noncomputable def tableErr (zk : Bool) (spec : SessionSpec) (st : Stmt) : ℝ :=
  if zk then
    Accounting.padLinkError (Accounting.m1Schedule st.m) (acct spec st) (Accounting.m1Pad st.m) +
      (Accounting.padRepError (Accounting.m1Schedule st.m) (acct spec st) (Accounting.m1Pad st.m) 2 +
        (Claims.Flock.Numbers.innerUp : ℝ)) ^ 2
  else Accounting.tableError ((Accounting.fast100 st.m).getD default) (acct spec st)

/-- **`ε_table`**: the session's tables' errors, summed. -/
noncomputable def εtable (zk : Bool) (x : Claims.Msg) : ℝ := ((tables zk x).map fun p => tableErr zk p.1 p.2).sum

/-- **`N₀`**: the length of the first committed codeword, table 0's level 0, `2^{d₀}`. -/
noncomputable def N₀ (zk : Bool) (x : Claims.Msg) : ℝ :=
  match (tables zk x).head? with
  | some (_, st) => 2 ^ (if zk then Claims.Flock.zkLevel0 st.m else Claims.Flock.plainLevel0 st.m).d
  | none => 0

/-- A table's weight: `Numbers.weightOf` at its schedule, and under `--zk` at its pads' domain. -/
noncomputable def tableWeight (zk : Bool) (st : Stmt) : ℝ :=
  if zk then
    Claims.Flock.Numbers.weightOf (Accounting.m1Schedule st.m)
      (Claims.Flock.Numbers.padDomain (Zk.PadLayout.new st.m st.c.kLog (Claims.Flock.zkSched st.m).1.k0))
  else Claims.Flock.Numbers.weightOf ((Accounting.fast100 st.m).getD default) 0

/-- **The weight `C` of `γ`**: the session's tables' weights, summed. -/
noncomputable def weight (zk : Bool) (x : Claims.Msg) : ℝ := ((tables zk x).map fun p => tableWeight zk p.2).sum

/-! ## The instances -/

/-- **Flock's deployed plain verifier**, as the record Flock's claims read. -/
noncomputable def deployPlain (rel : Readable.Parsed → Readable.Relation) :
    Claims.Flock.Deploy Coin (List 𝔽) Claims.Flock.PlainTrace where
  zk := false
  V := V false
  parse := parse false rel
  scopeCheck := scopeCheck false
  cols := cols false
  enc := Claims.Flock.plainEncD
  agree := Claims.Flock.Numbers.agree₀
  nMin := Claims.Flock.plainNMinD Claims.Flock.Numbers.agree₀
  read := Claims.Flock.plainReadD
  εtable := εtable false
  N₀ := N₀ false
  weight := weight false
  waits _ := Claims.Flock.Numbers.waits
  cap _ := Claims.Flock.Numbers.cap

/-- **Flock's deployed masked (`--zk`) verifier**, as the record Flock's claims read. -/
noncomputable def deployZk (rel : Readable.Parsed → Readable.Relation) :
    Claims.Flock.Deploy Coin (List 𝔽) Claims.Flock.ZkTrace where
  zk := true
  V := V true
  parse := parse true rel
  scopeCheck := scopeCheck true
  cols := cols true
  enc := Claims.Flock.zkEncD
  agree := Claims.Flock.zkAgree
  nMin := Claims.Flock.zkNMinD Claims.Flock.zkAgree
  read := Claims.Flock.zkReadD
  εtable := εtable true
  N₀ := N₀ true
  weight := weight true
  waits _ := Claims.Flock.Numbers.waits
  cap _ := Claims.Flock.Numbers.cap

/-! ## What the instances meet -/

/-- **A plain decoder meets the list-decoding spec** at the deployed code, threshold and count. -/
theorem listComplete_plain (rel : Readable.Parsed → Readable.Relation) :
    Claims.Flock.ListComplete (deployPlain rel).enc (deployPlain rel).agree (deployPlain rel).nMin
      (Claims.Flock.plainDecodeD Claims.Flock.Numbers.agree₀) :=
  Claims.Flock.listComplete_plainDecodeD _ Claims.Flock.Numbers.agree₀_pos

/-- **A masked decoder meets the list-decoding spec** at the deployed code, threshold and count. -/
theorem listComplete_zk (rel : Readable.Parsed → Readable.Relation) :
    Claims.Flock.ListComplete (deployZk rel).enc (deployZk rel).agree (deployZk rel).nMin
      (Claims.Flock.zkDecodeD Claims.Flock.zkAgree) :=
  Claims.Flock.listComplete_zkDecodeD _ Claims.Flock.zkAgree_pos

/-- **`ScopeMatches` for the masked verifier** (C's `scopeMatches_zk` at the verifier's parser and scope check). -/
theorem scopeMatches_zk (rel : Readable.Parsed → Readable.Relation) : Claims.Flock.ScopeMatches (deployZk rel) :=
  Readable.scopeMatches_zk (deployZk rel) rel (readOf true) rfl rfl (fun _ _ h => readOf_zk h) rfl

/-- **`ScopeMatches` for the plain verifier**, exactly when every statement it parses and checks has `22 ≤ m`. -/
theorem scopeMatches_plain_iff (rel : Readable.Parsed → Readable.Relation) :
    Claims.Flock.ScopeMatches (deployPlain rel) ↔
      ∀ x p, Readable.parseBytes (readOf false) x = some p → p.check = .ok () → 22 ≤ (Readable.toReadable rel p).m :=
  Readable.scopeMatches_iff (deployPlain rel) rel (readOf false) rfl rfl (fun _ _ h => readOf_zk h)

/-- The opened columns of a plain run sit at distinct positions. -/
theorem cols_nodup_plain (rel : Readable.Parsed → Readable.Relation) (s : Claims.Seen) :
    (((deployPlain rel).cols s).map Prod.fst).Nodup := cols_nodup false s

/-- The opened columns of a masked run sit at distinct positions. -/
theorem cols_nodup_zk (rel : Readable.Parsed → Readable.Relation) (s : Claims.Seen) :
    (((deployZk rel).cols s).map Prod.fst).Nodup := cols_nodup true s

/-- **A plain run opens fewer level-0 columns than `nMin`** of every trace: ruling 2's count is never reached by one
run. -/
theorem cols_length_lt_nMin_plain (rel : Readable.Parsed → Readable.Relation) (s : Claims.Seen)
    (M : Claims.Flock.PlainTrace) : ((deployPlain rel).cols s).length < (deployPlain rel).nMin M := by
  show (cols false s).length < Claims.Flock.plainNMinD Claims.Flock.Numbers.agree₀ M
  unfold cols
  split
  · rename_i r hr
    exact Claims.Flock.plainRunCols_length_lt_plainNMinD Claims.Flock.Numbers.agree₀_pos
      Claims.Flock.Numbers.agree₀_le_one r (runOf_reps hr).le M
  · exact Claims.one_le_johnsonCount _ _

/-- **A masked run opens fewer level-0 columns than `nMin`** of every trace: ruling 2's count is never reached by one
run. -/
theorem cols_length_lt_nMin_zk (rel : Readable.Parsed → Readable.Relation) (s : Claims.Seen)
    (M : Claims.Flock.ZkTrace) : ((deployZk rel).cols s).length < (deployZk rel).nMin M := by
  show (cols true s).length < Claims.Flock.zkNMinD Claims.Flock.zkAgree M
  unfold cols
  split
  · rename_i r hr
    exact Claims.Flock.zkRunCols_length_lt_zkNMinD' Claims.Flock.zkAgree_pos Claims.Flock.zkAgree_le_one r
      (runOf_reps hr).le M
  · exact Claims.one_le_johnsonCount _ _

end FlockDeploy
