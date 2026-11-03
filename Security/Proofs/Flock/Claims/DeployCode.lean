import Proofs.Flock.Claims.DeploySetup
import Proofs.Flock.Claims.Deploy
import Proofs.Flock.Claims.ZkCols
import Proofs.Flock.Claims.DecodeFlock
import Proofs.Flock.Claims.NumbersGamma
import Proofs.Flock.Claims.NumbersNMin
import Proofs.Flock.Claims.NumbersTable
import Proofs.Flock.Claims.GS.RadiusDeploy
import Proofs.Flock.Claims.Complete.Honest
import Proofs.Flock.Claims.Complete.Zk.Honest
import Proofs.Flock.Claims.Complete.Ready
import Proofs.Flock.Claims.Complete.ZkTables.Faithful

/-!
# Flock's deployed instances of `Claims.Flock.Deploy`

`FlockDeploy.deployPlain rel` and `FlockDeploy.deployZk rel` are the shipped verifier, plain and masked (`--zk`), as
records, at the relation's fields `rel` (C's `Readable.Relation`); Flock's claims read them with their columns trimmed
of trailing zeros (`FlockDeploy.deployPlainTrim`, `FlockDeploy.deployZkTrim`):

* `V`, `parse`, `scopeCheck`, `table`: the verifier in public-coin form (`FlockDeploy.V`), the verifier's own parser and
  scope check on the statement's bytes (`FlockDeploy.parse`, `FlockDeploy.scopeCheck`), and each table's own statement,
  table `j` of the parsed statement alone (`FlockDeploy.table`).
* `cols`: table `j`'s opened level-0 columns, its two reps', read off the run's last message (its statement, record
  and proof files) with no hash: the record's rep streams `2j` and `2j + 1` give the level-0 query round's coins, the
  proof files at the same places the opened rows (`runOf`, `Claims.Flock.plainRunCols`, `Claims.Flock.zkRunCols`).
  The masked record reads every table of the session; the plain one table `0` only (`plainTableCols`), the one table
  its scope admits.
* `enc`, `len`, `θ`, `read`: the level-0 code at each `m` the verifier serves, read as one (`IRS.famEnc`): the plain
  one (`plainEncD`, `22 ≤ m ≤ 35`, in `DeployPlainCode`) and the masked one (`zkEncD`, `25 ≤ m ≤ 35`), each trace's
  codeword length `2^d` and radius `1 − √ρ − 1/200` at its `m` (`plainLenD`, `plainThetaD`, `zkLenD`, `zkThetaD`), and
  its read step, the tables' traces read as one witness, interleaved as the relation reads them
  (`Claims.Flock.readTables`, table `c`'s hidden value `a` at `a·J + c`: `deployZk_read_getD`). At one table both
  records read as before (`deployPlain_one`, `deployZk_one`, `Claims.Flock.assemble_of_tables_eq_one`).
* `dec`: the counted radius decoder at that code (`GS.plainCountedRD`, `GS.zkCountedRD`: the counted Guruswami–Sudan
  decoder at each `m`, its count summed over the `m`s). It meets `ListWithin` at `len`, `enc` and `θ`
  (`FlockDeploy.listWithin_plain`, `FlockDeploy.listWithin_zk`), and at the trimmed records'
  (`FlockDeploy.listWithin_plainTrim`, `FlockDeploy.listWithin_zkTrim`).
* `assemble`, table by table: what it returns is one decoded trace per table, each in its own statement's relation
  read alone (`Claims.Flock.assemble_eq_some`), and it returns where every table has one
  (`Claims.Flock.assemble_isSome`).
* The list-decoding spec's earlier `nMin` form (`ListComplete`) is met at one agreement, `√ρ + 1/200`, at rate `1/2`
  without `--zk` (`Numbers.agree₀`) and at the masked code's largest rate `(2^12 + 554)/2^13` (`m = 25`) with it
  (`zkAgree`, `zkRate_le`): `listComplete_plain`, `listComplete_zk`.
* `εtable`: the tables' errors summed: `tableError` at the table's `fast100` schedule, or under `--zk` the padded
  table's `padLinkError + (padRepError + ε_inner)²` at `m1Schedule` with `ε_inner`'s bound at the inner proof's
  parameters (`Numbers.innerUp`). `N₀`: the tables' level-0 codeword lengths `2^{d₀}`, summed, since one wait's
  reruns open every table (`tableLen`). `weight`: the tables' weights summed (`Numbers.weightOf`, at the pads' domain
  `Numbers.padDomain` under `--zk`). `waits`, `cap`: `Numbers.waits` (`210`) and `Numbers.cap` (`2^256`). A statement
  that does not set up has no tables: `εtable`, `N₀` and `weight` are `0` there, where the verifier refuses every run.
* `Tape`, `P`: the reference prover, each verifier's honest prover on its tapes: `FlockComplete.honestP` on
  `FlockComplete.Tape` (`Complete.Honest`), and `FlockComplete.ZkTables.honestPR` on `FlockComplete.TapeZk`
  (`Complete.ZkTables.Faithful`), `flock-circuit --zk`'s: it runs each table on its share of the witness (the bits the
  relation reads, `Readable.tableWit`) with the tape's words in the mask slot (`ZkTables.maskWOf`), as
  `FlockComplete.ZkTables.honestP` does on that witness, and stops once a zerocheck outer coin is `1`.
* `δc`: the reference prover's completeness error at each hash `H`, `ε` (`εZk`), the bad coins' count over `2^128`, on
  the statements the verifier can accept against `H` (`FlockComplete.ready H`, `ZkTables.ready H`, at any number of
  tables), and `1` elsewhere (`FlockComplete.δc`, `ZkTables.δc`, in `Complete.Ready`, which states that set).

One run opens fewer level-0 columns than `nMin` of every trace (`cols_length_lt_nMin_plain`, `_zk`): that count
is reached only by `E^rep`'s columns, gathered over its accepted reruns.
-/

namespace Claims.Flock

open FlockSoundness FlockSoundness.Model

local notation "𝔽" => FlockLevel3.GF128

/-! ## A plain run's opened level-0 columns -/

/-- **The round of level 0's query coins** in a plain rep's stream: after zerocheck's `3 + (m − 6)` rounds, lincheck's
`3 + (kLog − 6)`, the ring switch's `claims + 1` and Ligerito's `k0 + 6` before level 0's queries (the out-of-domain
point and its coin, the `k0` lane challenges, and level 1's two OOD intros of two rounds each). -/
def plainQueryRound (m kLog k0 claims : ℕ) : ℕ := (3 + (m - 6)) + (3 + (kLog - 6)) + (claims + 1) + (k0 + 6)

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

/-! ## The tables read as one witness -/

/-- **The tables' traces read as one witness**, interleaved as the relation reads them: of `J` traces, hidden value
`a·J + c` is trace `c`'s hidden value `a`, so where a table's values sit depends on no other table's trace (each trace
reads `2^m` values at its own `m`). -/
def readTables {T : Type} (read : T → Wit) (Ms : List T) : Wit where
  hid := (List.range (Ms.length * (Ms.map fun M => (read M).hid.length).sum)).map fun k =>
    ((Ms[k % Ms.length]?).map fun M => (read M).hid.getD (k / Ms.length) false).getD false

/-- **One table's trace reads as before.** -/
theorem readTables_singleton {T : Type} (read : T → Wit) (M : T) : readTables read [M] = read M := by
  rcases hM : read M with ⟨hid⟩
  show (⟨_⟩ : Wit) = ⟨hid⟩
  congr 1
  apply List.ext_getElem
  · simp [hM]
  · intro k h1 h2
    simp only [List.getElem_map, List.getElem_range, List.length_singleton, Nat.mod_one, Nat.div_one,
      List.getElem?_cons_zero, Option.map_some, Option.getD_some, hM, List.getD_eq_getElem _ _ h2]

/-- **Table `c`'s hidden value `a` is the witness's `a·J + c`**, of `J` tables' traces. -/
theorem readTables_getD {T : Type} (read : T → Wit) (Ms : List T) {c : ℕ} (hc : c < Ms.length) (a : ℕ) :
    (readTables read Ms).hid.getD (a * Ms.length + c) false = (read Ms[c]).hid.getD a false := by
  have hJ : 0 < Ms.length := by omega
  have hmod : (a * Ms.length + c) % Ms.length = c := by
    rw [show a * Ms.length + c = c + a * Ms.length by omega, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hc]
  have hdiv : (a * Ms.length + c) / Ms.length = a := by
    rw [show a * Ms.length + c = c + a * Ms.length by omega, Nat.add_mul_div_right _ _ hJ, Nat.div_eq_of_lt hc,
      Nat.zero_add]
  by_cases hk : a * Ms.length + c < Ms.length * (Ms.map fun M => (read M).hid.length).sum
  · simp only [readTables, List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range hk, Option.map_some,
      Option.getD_some, hmod, hdiv, List.getElem?_eq_getElem hc]
  · have hle : (read Ms[c]).hid.length ≤ (Ms.map fun M => (read M).hid.length).sum :=
      List.le_sum_of_mem (List.mem_map.2 ⟨Ms[c], List.getElem_mem hc, rfl⟩)
    have ha : (read Ms[c]).hid.length ≤ a := by
      by_contra h
      have : (a + 1) * Ms.length ≤ Ms.length * (Ms.map fun M => (read M).hid.length).sum := by
        rw [Nat.mul_comm Ms.length]
        exact Nat.mul_le_mul_right _ (by omega)
      rw [Nat.succ_mul] at this
      omega
    rw [List.getD_eq_default _ _ (by simp only [readTables, List.length_map, List.length_range]; omega),
      List.getD_eq_default _ _ ha]

/-- **At one table, `assemble` is the single-table step**: when the statement's one table is its own statement, the
first decoded trace of table `0`'s columns whose witness, the trace read alone, is in the relation. -/
theorem assemble_of_tables_eq_one {Col T : Type} (parse : Msg → Option Claims.Flock.Stmt)
    (table : Msg → ℕ → Option Claims.Flock.Stmt) (decode : List (ℕ × Col) → List T) (read : List T → Wit) (x : Msg)
    (opens : List (ℕ → List (ℕ × Col))) (s : Claims.Flock.Stmt) (hp : parse x = some s) (h1 : s.tables = 1)
    (ht : table x 0 = some s) :
    assemble parse table decode read x opens =
      ((decode (opens.flatMap fun o => o 0)).map fun M => read [M]).find? fun w => decide (RelS s w) := by
  rw [List.find?_map]
  simp only [assemble, hp, h1, List.range_one, List.mapM_cons, List.mapM_nil, ht, Option.bind_some, Function.comp_def]
  cases (decode (opens.flatMap fun o => o 0)).find? (fun M => decide (RelS s (read [M]))) <;> rfl

/-- `Option`'s `mapM` returns one result per entry, each the entry's. -/
theorem mapM_option_some {α β : Type} {f : α → Option β} :
    ∀ {l : List α} {ys : List β}, l.mapM f = some ys →
      ys.length = l.length ∧ ∀ i (hi : i < ys.length) (hl : i < l.length), f l[i] = some ys[i]
  | [], ys, h => by
    simp only [List.mapM_nil, pure, Option.some.injEq] at h
    subst h
    exact ⟨rfl, fun i hi => absurd hi (Nat.not_lt_zero i)⟩
  | a :: l, ys, h => by
    simp only [List.mapM_cons, Option.bind_eq_bind, Option.bind_eq_some_iff, pure, Option.some.injEq] at h
    obtain ⟨b, hb, bs, hbs, rfl⟩ := h
    obtain ⟨hlen, hall⟩ := mapM_option_some hbs
    refine ⟨by simp [hlen], fun i hi hl => ?_⟩
    cases i with
    | zero => exact hb
    | succ i => exact hall i (by simpa using hi) (by simpa using hl)

/-- `Option`'s `mapM` returns where every entry does. -/
theorem mapM_option_isSome {α β : Type} {f : α → Option β} :
    ∀ {l : List α}, (∀ a ∈ l, (f a).isSome) → (l.mapM f).isSome
  | [], _ => rfl
  | a :: l, h => by
    obtain ⟨b, hb⟩ := Option.isSome_iff_exists.1 (h a List.mem_cons_self)
    obtain ⟨bs, hbs⟩ := Option.isSome_iff_exists.1
      (mapM_option_isSome (l := l) fun a' ha' => h a' (List.mem_cons_of_mem _ ha'))
    simp [List.mapM_cons, hb, hbs]

/-- **What `assemble` returns, table by table**: the tables' choices read as one witness, one per table of the
statement, each a decoded trace of its table's columns that, read alone, is in the relation of the table's own
statement. -/
theorem assemble_eq_some {Col T : Type} {parse : Msg → Option Claims.Flock.Stmt}
    {table : Msg → ℕ → Option Claims.Flock.Stmt} {decode : List (ℕ × Col) → List T} {read : List T → Wit} {x : Msg}
    {opens : List (ℕ → List (ℕ × Col))} {s : Claims.Flock.Stmt} {w : Wit} (hp : parse x = some s)
    (h : assemble parse table decode read x opens = some w) :
    ∃ Ms : List T, w = read Ms ∧ Ms.length = s.tables ∧ ∀ j (hj : j < Ms.length), ∃ t, table x j = some t ∧
      Ms[j] ∈ decode (opens.flatMap fun o => o j) ∧ RelS t (read [Ms[j]]) := by
  simp only [assemble, hp] at h
  obtain ⟨Ms, hMs, rfl⟩ := Option.map_eq_some_iff.1 h
  obtain ⟨hlen, hall⟩ := mapM_option_some hMs
  rw [List.length_range] at hlen
  refine ⟨Ms, rfl, hlen, fun j hj => ?_⟩
  have hj' := hall j hj (by rw [List.length_range]; omega)
  rw [List.getElem_range] at hj'
  obtain ⟨t, ht, hf⟩ := Option.bind_eq_some_iff.1 hj'
  have hq := List.find?_some hf
  simp only [decide_eq_true_eq] at hq
  exact ⟨t, ht, List.mem_of_find?_eq_some hf, hq⟩

/-- **`assemble` returns where every table has a candidate**: a decoded trace of its columns that, read alone, is in
the relation of its own statement. -/
theorem assemble_isSome {Col T : Type} {parse : Msg → Option Claims.Flock.Stmt}
    {table : Msg → ℕ → Option Claims.Flock.Stmt} {decode : List (ℕ × Col) → List T} {read : List T → Wit} {x : Msg}
    {opens : List (ℕ → List (ℕ × Col))} {s : Claims.Flock.Stmt} (hp : parse x = some s)
    (h : ∀ j < s.tables, ∃ t, table x j = some t ∧ ∃ M ∈ decode (opens.flatMap fun o => o j), RelS t (read [M])) :
    (assemble parse table decode read x opens).isSome := by
  simp only [assemble, hp, Option.isSome_map]
  refine mapM_option_isSome fun j hj => ?_
  obtain ⟨t, ht, M, hM, hR⟩ := h j (List.mem_range.1 hj)
  rw [ht, Option.bind_some, List.find?_isSome]
  exact ⟨M, hM, decide_eq_true hR⟩

/-- A statement has one table exactly when its count is at most one. -/
theorem Stmt.tables_eq_one_iff (s : Claims.Flock.Stmt) : s.tables = 1 ↔ s.form.tables ≤ 1 := by
  unfold Claims.Flock.Stmt.tables
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

/-- **What the column reader takes of a run's table `j`**, with no hash: from the last message's statement, record and
proof files, table `j`'s `m`, `kLog`, claim count and Merkle widths (its setup's, its claims at the session's link
points), and its two reps' recorded rounds (the record's streams `<t>/rep0`, `<t>/rep1`, at `2j` and `2j + 1` in
`spec.repStreams` order) and proof files (at the same two places, as `verify` reads them). -/
def runOf (zk : Bool) (s : Claims.Seen) (j : ℕ) : Option Claims.Flock.ZkRun := do
  let f ← lastFinal s
  let (_, spec, sts) ← setupOf zk f.stmt
  let st ← sts[j]?
  let tab := Setup.ofCircuit st
  let lv ← (liveOf spec f.record).toOption
  let extras ← (tab.extra lv.points).toOption
  pure { m := tab.m, kLog := tab.kLog, claims := 2 + extras.size, hashLen := tab.merkle.digestLen,
         saltLen := saltLenOf zk tab,
         reps := [(lv.streams.getD (2 * j) #[], f.proofs.getD (2 * j) ByteArray.empty),
                  (lv.streams.getD (2 * j + 1) #[], f.proofs.getD (2 * j + 1) ByteArray.empty)] }


namespace KS.Gen

variable [S : ZkSetup]

/-- **What the column reader takes of a run's table `j`**, with no hash: from the last message's statement, record and
proof files, table `j`'s `m`, `kLog`, claim count and Merkle widths (its setup's, its claims at the session's link
points), and its two reps' recorded rounds (the record's streams `<t>/rep0`, `<t>/rep1`, at `2j` and `2j + 1` in
`spec.repStreams` order) and proof files (at the same two places, as `verify` reads them). -/
def runOf (zk : Bool) (s : Claims.Seen) (j : ℕ) : Option Claims.Flock.ZkRun := do
  let f ← lastFinal s
  let (_, spec, sts) ← setupOf zk f.stmt
  let st ← sts[j]?
  let tab := Setup.ofCircuit st
  let lv ← (liveOf spec f.record).toOption
  let extras ← (tab.extra lv.points).toOption
  pure { m := tab.m, kLog := tab.kLog, claims := 2 + extras.size, hashLen := tab.merkle.digestLen,
         saltLen := saltLenOf zk tab,
         reps := [(lv.streams.getD (2 * j) #[], f.proofs.getD (2 * j) ByteArray.empty),
                  (lv.streams.getD (2 * j + 1) #[], f.proofs.getD (2 * j + 1) ByteArray.empty)] }

end KS.Gen

/-- **The opened level-0 columns of a run's table `j`**: its two reps' (`plainRunCols`, `zkRunCols`); none where the
run's last message does not set up table `j`. -/
def cols (zk : Bool) (s : Claims.Seen) (j : ℕ) : List (ℕ × List 𝔽) :=
  match runOf zk s j with
  | some r => if zk then Claims.Flock.zkRunCols r else Claims.Flock.plainRunCols r
  | none => []


namespace KS.Gen

variable [S : ZkSetup]

/-- **The opened level-0 columns of a run's table `j`**: its two reps' (`plainRunCols`, `zkRunCols`); none where the
run's last message does not set up table `j`. -/
def cols (zk : Bool) (s : Claims.Seen) (j : ℕ) : List (ℕ × List 𝔽) :=
  match runOf zk s j with
  | some r => if zk then Claims.Flock.zkRunCols r else Claims.Flock.plainRunCols r
  | none => []

end KS.Gen

/-- **The plain record's columns**: table `0`'s, none at another table. The plain verifier's scope admits one table. -/
def plainTableCols (s : Claims.Seen) (j : ℕ) : List (ℕ × List 𝔽) := if j = 0 then cols false s 0 else []

theorem runOf_reps {zk : Bool} {s : Claims.Seen} {j : ℕ} {r : Claims.Flock.ZkRun} (h : runOf zk s j = some r) :
    r.reps.length = 2 := by
  unfold runOf at h
  simp only [Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def, Option.some.injEq] at h
  obtain ⟨_, _, ⟨_, _, _⟩, _, _, _, _, _, _, _, rfl⟩ := h
  rfl

theorem cols_nodup (zk : Bool) (s : Claims.Seen) (j : ℕ) : ((cols zk s j).map Prod.fst).Nodup := by
  unfold cols
  split
  · split
    · exact Claims.Flock.zkRunCols_nodup _
    · exact Claims.Flock.plainRunCols_nodup _
  · exact List.nodup_nil

theorem plainTableCols_nodup (s : Claims.Seen) (j : ℕ) : ((plainTableCols s j).map Prod.fst).Nodup := by
  unfold plainTableCols
  split
  · exact cols_nodup false s 0
  · exact List.nodup_nil

/-! ## The numbers -/

/-- The session's tables with its spec; none where the statement does not set up. -/
def tables (zk : Bool) (x : Claims.Msg) : List (SessionSpec × Stmt) :=
  match setupOf zk x with
  | some (_, spec, sts) => sts.toList.map fun st => (spec, st)
  | none => []


namespace KS.Gen

variable [S : ZkSetup]

/-- The session's tables with its spec; none where the statement does not set up. -/
def tables (zk : Bool) (x : Claims.Msg) : List (SessionSpec × Stmt) :=
  match setupOf zk x with
  | some (_, spec, sts) => sts.toList.map fun st => (spec, st)
  | none => []

end KS.Gen

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


namespace KS.Gen

variable [S : ZkSetup]

/-- **`ε_table`**: the session's tables' errors, summed. -/
noncomputable def εtable (zk : Bool) (x : Claims.Msg) : ℝ := ((tables zk x).map fun p => tableErr zk p.1 p.2).sum

end KS.Gen

/-- A table's level-0 codeword length, `2^{d₀}`. -/
noncomputable def tableLen (zk : Bool) (st : Stmt) : ℝ :=
  2 ^ (if zk then Claims.Flock.zkLevel0 st.m else Claims.Flock.plainLevel0 st.m).d

/-- **`N₀`**: the session's tables' level-0 codeword lengths, summed: table 0's `2^{d₀}` at one table. -/
noncomputable def N₀ (zk : Bool) (x : Claims.Msg) : ℝ := ((tables zk x).map fun p => tableLen zk p.2).sum


namespace KS.Gen

variable [S : ZkSetup]

/-- **`N₀`**: the session's tables' level-0 codeword lengths, summed: table 0's `2^{d₀}` at one table. -/
noncomputable def N₀ (zk : Bool) (x : Claims.Msg) : ℝ := ((tables zk x).map fun p => tableLen zk p.2).sum

end KS.Gen

/-- A table's weight: `Numbers.weightOf` at its schedule, and under `--zk` at its pads' domain. -/
noncomputable def tableWeight (zk : Bool) (st : Stmt) : ℝ :=
  if zk then
    Claims.Flock.Numbers.weightOf (Accounting.m1Schedule st.m)
      (Claims.Flock.Numbers.padDomain (Zk.PadLayout.new st.m st.c.kLog (Claims.Flock.zkSched st.m).1.k0))
  else Claims.Flock.Numbers.weightOf ((Accounting.fast100 st.m).getD default) 0

/-- **The weight `C` of `γ`**: the session's tables' weights, summed. -/
noncomputable def weight (zk : Bool) (x : Claims.Msg) : ℝ := ((tables zk x).map fun p => tableWeight zk p.2).sum


namespace KS.Gen

variable [S : ZkSetup]

/-- **The weight `C` of `γ`**: the session's tables' weights, summed. -/
noncomputable def weight (zk : Bool) (x : Claims.Msg) : ℝ := ((tables zk x).map fun p => tableWeight zk p.2).sum

end KS.Gen

/-! ## The instances -/

/-- **Flock's deployed plain verifier**, as a record; Flock's claims read it trimmed (`deployPlainTrim`). -/
noncomputable def deployPlain (rel : Readable.Parsed → Readable.Relation) :
    Claims.Flock.Deploy Coin (List 𝔽) Claims.Flock.PlainTrace where
  zk := false
  V := V false
  parse := parse false rel
  scopeCheck := scopeCheck false
  table := table false rel
  cols := plainTableCols
  enc := Claims.Flock.plainEncD
  len := Claims.Flock.plainLenD
  θ := Claims.Flock.plainThetaD
  dec := Claims.GS.plainCountedRD
  read := Claims.Flock.readTables Claims.Flock.plainReadD
  εtable := εtable false
  N₀ := N₀ false
  weight := weight false
  waits _ := Claims.Flock.Numbers.waits
  cap _ := Claims.Flock.Numbers.cap
  Tape := FlockComplete.Tape
  P := FlockComplete.honestP
  δc := FlockComplete.δc

/-- **Flock's deployed masked (`--zk`) verifier**, as a record; Flock's claims read it trimmed (`deployZkTrim`). -/
noncomputable def deployZk (rel : Readable.Parsed → Readable.Relation) :
    Claims.Flock.Deploy Coin (List 𝔽) Claims.Flock.ZkTrace where
  zk := true
  V := V true
  parse := parse true rel
  scopeCheck := scopeCheck true
  table := table true rel
  cols := cols true
  enc := Claims.Flock.zkEncD
  len := Claims.Flock.zkLenD
  θ := Claims.Flock.zkThetaD
  dec := Claims.GS.zkCountedRD
  read := Claims.Flock.readTables Claims.Flock.zkReadD
  εtable := εtable true
  N₀ := N₀ true
  weight := weight true
  waits _ := Claims.Flock.Numbers.waits
  cap _ := Claims.Flock.Numbers.cap
  Tape := FlockComplete.TapeZk
  P := FlockComplete.ZkTables.honestPR
  δc := FlockComplete.ZkTables.δc


namespace KS.Gen

variable [S : ZkSetup]

/-- **Flock's deployed masked (`--zk`) verifier**, as a record; Flock's claims read it trimmed (`deployZkTrim`). -/
noncomputable def deployZk (rel : Readable.Parsed → Readable.Relation) :
    Claims.Flock.Deploy Coin (List 𝔽) Claims.Flock.ZkTrace where
  zk := true
  V := V true
  parse := parse true rel
  scopeCheck := scopeCheck true
  table := table true rel
  cols := cols true
  enc := Claims.Flock.zkEncD
  len := Claims.Flock.zkLenD
  θ := Claims.Flock.zkThetaD
  dec := Claims.GS.zkCountedRD
  read := Claims.Flock.readTables Claims.Flock.zkReadD
  εtable := εtable true
  N₀ := N₀ true
  weight := weight true
  waits _ := Claims.Flock.Numbers.waits
  cap _ := Claims.Flock.Numbers.cap
  Tape := FlockComplete.TapeZk
  P := FlockComplete.ZkTables.honestPR
  δc := FlockComplete.ZkTables.δc

end KS.Gen

/-! ## The records Flock's claims read: columns trimmed of trailing zeros

The verifier opens full-width columns, and `fill` reads a position no run opened as `[]` (`Flock.instZeroListCol`),
which is no full-width column. Read through `trimCol`, the zero column of every width is `[]`, and a column is still
its lanes (`trimCol_getD`), so the claims' records (`deployPlainTrim`, `deployZkTrim`) are the instances with their
opened columns and their code trimmed (`Deploy.mapCols`), everything else the instances'. -/

/-- **A column trimmed of its trailing zeros**: the zero column of every width is `[]`. -/
def trimCol {F : Type} [Zero F] [DecidableEq F] : List F → List F
  | [] => []
  | a :: u => if a = 0 ∧ trimCol u = [] then [] else a :: trimCol u

/-- **A trimmed column has the column's lanes**, `0` past both. -/
theorem trimCol_getD {F : Type} [Zero F] [DecidableEq F] (u : List F) (ℓ : ℕ) :
    (trimCol u).getD ℓ 0 = u.getD ℓ 0 := by
  induction u generalizing ℓ with
  | nil => rfl
  | cons a u ih =>
    rw [trimCol.eq_2]
    split_ifs with h
    · cases ℓ with
      | zero => rw [List.getD_nil, List.getD_cons_zero, h.1]
      | succ ℓ => rw [List.getD_nil, List.getD_cons_succ, ← ih, h.2, List.getD_nil]
    · cases ℓ with
      | zero => rfl
      | succ ℓ => rw [List.getD_cons_succ, List.getD_cons_succ, ih]

theorem eq_of_getD {F : Type} [Zero F] {u v : List F} (hl : u.length = v.length)
    (h : ∀ ℓ, u.getD ℓ 0 = v.getD ℓ 0) : u = v := by
  refine List.ext_getElem hl fun ℓ h1 h2 => ?_
  have := h ℓ
  rwa [List.getD_eq_getElem _ _ h1, List.getD_eq_getElem _ _ h2] at this

/-- **Two columns of one width with one trim are one.** -/
theorem trimCol_inj {F : Type} [Zero F] [DecidableEq F] {u v : List F} (hl : u.length = v.length)
    (h : trimCol u = trimCol v) : u = v :=
  eq_of_getD hl fun ℓ => by rw [← trimCol_getD u, h, trimCol_getD]

/-- **The zero column of every width trims to `[]`.** -/
theorem trimCol_replicate {F : Type} [Zero F] [DecidableEq F] (n : ℕ) :
    trimCol (List.replicate n (0 : F)) = [] := by
  induction n with
  | zero => rfl
  | succ n ih => rw [List.replicate_succ, trimCol.eq_2, ite_eq_left ⟨rfl, ih⟩]

/-- **A record with its columns read through `f`**: its opened columns and its code's, everything else its own. -/
def _root_.Claims.Flock.Deploy.mapCols {Coin Col T : Type} (D : Claims.Flock.Deploy Coin Col T) (f : Col → Col) :
    Claims.Flock.Deploy Coin Col T :=
  { D with cols := fun s j => (D.cols s j).map fun p => (p.1, f p.2), enc := fun M q => f (D.enc M q) }

/-- **Flock's deployed plain verifier, as the record Flock's claims read**: `deployPlain rel` with its opened columns
and its code trimmed of trailing zeros. -/
noncomputable def deployPlainTrim (rel : Readable.Parsed → Readable.Relation) :
    Claims.Flock.Deploy Coin (List 𝔽) Claims.Flock.PlainTrace :=
  (deployPlain rel).mapCols trimCol

/-- **Flock's deployed masked (`--zk`) verifier, as the record Flock's claims read**: `deployZk rel` with its opened
columns and its code trimmed of trailing zeros. -/
noncomputable def deployZkTrim (rel : Readable.Parsed → Readable.Relation) :
    Claims.Flock.Deploy Coin (List 𝔽) Claims.Flock.ZkTrace :=
  (deployZk rel).mapCols trimCol


namespace KS.Gen

variable [S : ZkSetup]

/-- **Flock's deployed masked (`--zk`) verifier, as the record Flock's claims read**: `deployZk rel` with its opened
columns and its code trimmed of trailing zeros. -/
noncomputable def deployZkTrim (rel : Readable.Parsed → Readable.Relation) :
    Claims.Flock.Deploy Coin (List 𝔽) Claims.Flock.ZkTrace :=
  (deployZk rel).mapCols trimCol

end KS.Gen

/-! ## What the instances meet -/

/-- **A plain decoder meets the list-decoding spec's earlier `nMin` form** at the deployed code, at `agree₀` and its
Johnson count. -/
theorem listComplete_plain (rel : Readable.Parsed → Readable.Relation) :
    Claims.Flock.ListComplete (deployPlain rel).enc Claims.Flock.Numbers.agree₀
      (Claims.Flock.plainNMinD Claims.Flock.Numbers.agree₀) (Claims.Flock.plainDecodeD Claims.Flock.Numbers.agree₀) :=
  Claims.Flock.listComplete_plainDecodeD _ Claims.Flock.Numbers.agree₀_pos

/-- **A masked decoder meets the list-decoding spec's earlier `nMin` form** at the deployed code, at `zkAgree` and its
Johnson count. -/
theorem listComplete_zk (rel : Readable.Parsed → Readable.Relation) :
    Claims.Flock.ListComplete (deployZk rel).enc Claims.Flock.zkAgree (Claims.Flock.zkNMinD Claims.Flock.zkAgree)
      (Claims.Flock.zkDecodeD Claims.Flock.zkAgree) :=
  Claims.Flock.listComplete_zkDecodeD _ Claims.Flock.zkAgree_pos

/-- **`ScopeMatches` for the masked verifier** (C's `scopeMatches` at the verifier's parser and scope check). -/
theorem scopeMatches_zk (rel : Readable.Parsed → Readable.Relation) : Claims.Flock.ScopeMatches (deployZk rel) :=
  Readable.scopeMatches (deployZk rel) rel (readOf true) rfl rfl (fun _ _ h => readOf_zk h)

/-- **`ScopeMatches` for the plain verifier** (C's `scopeMatches` at the verifier's parser and scope check). -/
theorem scopeMatches_plain (rel : Readable.Parsed → Readable.Relation) : Claims.Flock.ScopeMatches (deployPlain rel) :=
  Readable.scopeMatches (deployPlain rel) rel (readOf false) rfl rfl (fun _ _ h => readOf_zk h)

/-- The opened columns of a plain run's table sit at distinct positions. -/
theorem cols_nodup_plain (rel : Readable.Parsed → Readable.Relation) (s : Claims.Seen) (j : ℕ) :
    (((deployPlain rel).cols s j).map Prod.fst).Nodup := plainTableCols_nodup s j

/-- The opened columns of a masked run's table sit at distinct positions. -/
theorem cols_nodup_zk (rel : Readable.Parsed → Readable.Relation) (s : Claims.Seen) (j : ℕ) :
    (((deployZk rel).cols s j).map Prod.fst).Nodup := cols_nodup true s j

/-- **A plain run opens fewer level-0 columns than `nMin`** of every trace, at every table: one run never reaches
it. -/
theorem cols_length_lt_nMin_plain (rel : Readable.Parsed → Readable.Relation) (s : Claims.Seen) (j : ℕ)
    (M : Claims.Flock.PlainTrace) :
    ((deployPlain rel).cols s j).length < Claims.Flock.plainNMinD Claims.Flock.Numbers.agree₀ M := by
  show (plainTableCols s j).length < Claims.Flock.plainNMinD Claims.Flock.Numbers.agree₀ M
  unfold plainTableCols cols
  split
  · split
    · rename_i r hr
      exact Claims.Flock.plainRunCols_length_lt_plainNMinD Claims.Flock.Numbers.agree₀_pos
        Claims.Flock.Numbers.agree₀_le_one r (runOf_reps hr).le M
    · exact Claims.one_le_johnsonCount _ _
  · exact Claims.one_le_johnsonCount _ _

/-- **A masked run opens fewer level-0 columns than `nMin`** of every trace, at every table: one run never reaches
it. -/
theorem cols_length_lt_nMin_zk (rel : Readable.Parsed → Readable.Relation) (s : Claims.Seen) (j : ℕ)
    (M : Claims.Flock.ZkTrace) : ((deployZk rel).cols s j).length < Claims.Flock.zkNMinD Claims.Flock.zkAgree M := by
  show (cols true s j).length < Claims.Flock.zkNMinD Claims.Flock.zkAgree M
  unfold cols
  split
  · rename_i r hr
    exact Claims.Flock.zkRunCols_length_lt_zkNMinD' Claims.Flock.zkAgree_pos Claims.Flock.zkAgree_le_one r
      (runOf_reps hr).le M
  · exact Claims.one_le_johnsonCount _ _

/-- **The plain record at one table reads as before**: table `0`'s columns, one trace read alone. -/
theorem deployPlain_one (rel : Readable.Parsed → Readable.Relation) (s : Claims.Seen) (M : Claims.Flock.PlainTrace) :
    (deployPlain rel).cols s 0 = cols false s 0 ∧ (deployPlain rel).read [M] = Claims.Flock.plainReadD M :=
  ⟨rfl, Claims.Flock.readTables_singleton _ _⟩

/-- **The masked record at one table reads as before**: table `0`'s columns, one trace read alone. -/
theorem deployZk_one (rel : Readable.Parsed → Readable.Relation) (s : Claims.Seen) (M : Claims.Flock.ZkTrace) :
    (deployZk rel).cols s 0 = cols true s 0 ∧ (deployZk rel).read [M] = Claims.Flock.zkReadD M :=
  ⟨rfl, Claims.Flock.readTables_singleton _ _⟩

/-- **The masked record reads table `c`'s hidden value `a` at `a·J + c`**, of `J` tables' traces: the session's tables
in one witness, interleaved, at each table's own `m`. -/
theorem deployZk_read_getD (rel : Readable.Parsed → Readable.Relation) (Ms : List Claims.Flock.ZkTrace) {c : ℕ}
    (hc : c < Ms.length) (a : ℕ) :
    ((deployZk rel).read Ms).hid.getD (a * Ms.length + c) false = (Claims.Flock.zkReadD Ms[c]).hid.getD a false :=
  Claims.Flock.readTables_getD _ Ms hc a

end FlockDeploy
