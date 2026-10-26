import Proofs.Flock.Claims.Complete.Sim
import Proofs.Flock.Claims.Complete.ZkTables.Honest

/-!
# The deployed completeness error

`deployPlain`'s and `deployZk`'s `δc` (`Claims.Flock.Deploy.δc`), the error `Claims.Flock.Complete` reads: at the hash
`H`, `δc H x` is the bad coins' count over `2^128` (`ε`, `εZk`) on the statements the verifier can accept against `H`
(`ready H`, `ZkTables.ready H`), and `1` on every other statement.

* **The bad coins** (`Coins`: `pr_bad_le`): zerocheck's outer coins (`outerSlots`): rep `q`'s second round (`rOuter`,
  after `r_skip`), elements `0` to `m − 14`, at its index among the verifier's coins (`repStart`: the link's coin, then
  the earlier reps' recorded rounds, `spec.repStreams` order). At `r = 1` the verifier's `(1 + r)⁻¹` is `0⁻¹ = 0`. The
  masked verifier adds each rep's inner `β` (`betaSlotsZk`, refused at `0`) and level 0's pair coins (`rhoSlotsZk`,
  refused when dependent), counted twice, and each table's rank checks (`rankSlotsZk`: the prover refuses coins where
  the mask slot's weights at its claim points are rank-deficient), counted `4·m` times (`εZk`).
* **`δc H x = 1` exactly where `ready H x` fails**, which is on the statements that:
  * set up no plain session (`setupOf false x = none`), which the verifier refuses at every run;
  * set up a session that draws no unit: `checks` ends in `drawLaw zk tags false none`, which refuses;
  * are `2^64` bytes or more: the last message (`Final.encode`) writes the statement's length in a 64-bit word;
  * have a table whose regions do not fit (`RegionsFit`: some region's bytes at some block are not its free bits): the
    extra claims refuse (`Stmt.regionValue`), the honest prover's as the verifier's, and the relation, on the circuit's
    values, does not pin the public bytes the regions read;
  * set up otherwise against `H` than at SHA-512: the verifier sets the statement up against `H`
    (`setupWith (hOf H)`), recomputing with `H` the SHA-512 digests the statement pins, and the honest prover sets up
    at SHA-512. At `sha512` the two agree (`ready_sha512`);
  * have an honest record that does not fit (`RecordFitsAt`): at some witness, tape and coins, the honest session's
    printed record is `2^64` bytes or more, past the last message's 64-bit word for its length.
* **`ZkTables.δc H x = 1` exactly where `ZkTables.ready H x` fails** (`deployZk`'s): the same of the masked session
  (`setupOf true`, and `ZkTables.RecordFitsAt` at the run of the honest masked prover on each table's share,
  `ZkTables.traces`), at any number of tables, and one more: its `Hello` names the keyed coin tree (`spec.coinTree`),
  and the masked verifier refuses any other session. `εZk` counts every table's two reps, so it is the masked union
  bound at every number of tables.
* **`δcZk H x = 1` exactly where `readyZk H x` fails**, the error of `honestPZk` (the prover that reads table `i`'s
  bits at `i·2^m + a`, `tracesOf`, the relation's only at one table): `ZkTables.ready` at the masked session's run of
  `honestPZk` (`RecordFitsZkAt`), at one table.
* `CompleteWith H D P δc`: `Claims.Flock.Complete`'s statement at `D` with a prover and an error of one's own, which
  the masked proofs state `honestPZk`'s completeness at `δcZk` in, and `ZkTables.honestPR`'s at `ZkTables.δc` is
  `Claims.Flock.Complete H (deployZk rel)`.
-/

namespace FlockComplete

open Flock FlockDeploy

/-! ## The plain verifier -/

/-- **Rep `q`'s first coin's index** (table `q / 2`'s rep `q % 2`, `spec.repStreams` order): the link's coin, then every
earlier rep's recorded rounds. -/
def repStart (spec : SessionSpec) (sts : Array Flock.Stmt) (q : ℕ) : ℕ :=
  1 + ((List.range q).map fun q' => repRounds false spec sts[q' / 2]!).sum

/-- **The zerocheck outer coins** of the session the statement sets up: rep `q`'s second round (`rOuter`, after
`r_skip`), elements `0` to `m − 14`, as (coin index, element) pairs. -/
def outerSlots (x : Claims.Msg) : List (ℕ × ℕ) :=
  match setupOf false x with
  | some (_, spec, sts) => (List.range (2 * sts.size)).flatMap fun q =>
      (List.range (sts[q / 2]!.m - 13)).map fun i => (repStart spec sts q + 1, i)
  | none => []

/-- `ε`: the outer coins' count over `2^128`. -/
noncomputable def ε (x : Claims.Msg) : ℝ := (outerSlots x).length / 2 ^ 128

/-- **A statement's regions fit**: each region's bytes at each block are its free bits. -/
def RegionsFit (st : Flock.Stmt) : Prop :=
  ∀ reg ∈ st.regions, ∀ b < 2 ^ st.nbl, (reg.bytes b).size * 8 = 2 ^ reg.free.size

/-- **The honest record fits its length word** at the plain session `spec`, `sts` of `x`: at every witness, tape and
coins, a record the pure honest session returns (`sessionS`) is under `2^64` bytes. -/
def RecordFitsAt (H : Claims.Msg → Claims.Msg) (x : Claims.Msg) (spec : SessionSpec) (sts : Array Flock.Stmt) : Prop :=
  ∀ (w : Claims.Flock.Wit) (ω : Tape) (r : Fin ((V false).k x) → Coin) (record : String) (proofs : Array ByteArray)
    (bodies : Array Claims.Msg),
    sessionS H (fun j => (Claims.coinSeq r j).bytes) spec sts w ω = (.ok (record, proofs), bodies) →
      record.toUTF8.size < 2 ^ 64

open Classical in
/-- **The statements the deployed verifier can accept** against `H`: it sets up the session, the session draws a unit,
the statement is under `2^64` bytes, each table's regions fit, its setup against `H` is the one at SHA-512, and the
honest record fits. -/
noncomputable def ready (H : Claims.Msg → Claims.Msg) (x : Claims.Msg) : Bool :=
  match setupOf false x with
  | some (s, spec, sts) => s.read.draw.isSome && decide (x.length < 2 ^ 64) && decide (∀ st ∈ sts, RegionsFit st) &&
      decide (setupWith (Claims.hOf H) s.read = .ok (spec, sts)) && decide (RecordFitsAt H x spec sts)
  | none => false

open Classical in
/-- **At SHA-512, `ready` asks no setup agreement**: the setup against `sha512` is the statement's. -/
theorem ready_sha512 (x : Claims.Msg) :
    ready sha512 x = match setupOf false x with
      | some (s, spec, sts) => s.read.draw.isSome && decide (x.length < 2 ^ 64) &&
          decide (∀ st ∈ sts, RegionsFit st) && decide (RecordFitsAt sha512 x spec sts)
      | none => false := by
  unfold ready
  split
  · rename_i s spec sts hs
    rw [hOf_sha512, setupWith_sha512, Ins.setup_read (setupOf_some hs).2]
    simp
  · rfl

/-- **Flock's `δc` at the plain verifier** against `H`: `ε` where the verifier can accept, else `1`. -/
noncomputable def δc (H : Claims.Msg → Claims.Msg) (x : Claims.Msg) : ℝ := if ready H x then ε x else 1

/-- **A drawn session's `δc` at SHA-512 is below 1**: where the statement sets up a session that draws, is under `2^64`
bytes, has regions and an honest record that fit, and has fewer than `2^128` outer coins, `ready` holds
(`ready_sha512`), so `δc` is `ε`. -/
theorem δc_sha512_lt_one {x : Claims.Msg} {s spec sts} (hs : setupOf false x = some (s, spec, sts))
    (hd : s.read.draw.isSome = true) (hl : x.length < 2 ^ 64) (hreg : ∀ st ∈ sts, RegionsFit st)
    (hrec : RecordFitsAt sha512 x spec sts) (hε : (outerSlots x).length < 2 ^ 128) :
    δc sha512 x < 1 := by
  have hr : ready sha512 x = true := by
    rw [ready_sha512, hs]
    simp only [Bool.and_eq_true, decide_eq_true_eq]
    exact ⟨⟨⟨hd, hl⟩, hreg⟩, hrec⟩
  have h2 : ((outerSlots x).length : ℝ) < 2 ^ 128 := by exact_mod_cast hε
  rw [δc, if_pos hr, ε]
  exact (div_lt_one (by positivity)).2 h2

/-! ## The masked verifier -/

/-- **Rep `q`'s first coin's index** at the masked verifier (table `q / 2`'s rep `q % 2`): the link's coin, then every
earlier rep's recorded rounds. -/
def repStartZk (spec : SessionSpec) (sts : Array Flock.Stmt) (q : ℕ) : ℕ :=
  1 + ((List.range q).map fun q' => repRounds true spec sts[q' / 2]!).sum

/-- A masked rep's rounds before Ligerito: the masked replay's `m + k_log − 6`, the ring switch's `claims + 1`. -/
def piopRoundsZk (spec : SessionSpec) (st : Flock.Stmt) : ℕ :=
  (st.m + st.c.kLog - 6) + (2 + spec.points * st.regions.size + 1)

/-- **The zerocheck outer coins** of the masked session: rep `q`'s second round, elements `0` to `m − 14`. -/
def outerSlotsZk (x : Claims.Msg) : List (ℕ × ℕ) :=
  match setupOf true x with
  | some (_, spec, sts) => (List.range (2 * sts.size)).flatMap fun q =>
      (List.range (sts[q / 2]!.m - 13)).map fun i => (repStartZk spec sts q + 1, i)
  | none => []

/-- **Level 0's pair coins**: rep `q`'s `ρ_0` (`ρ_1` the next), Ligerito's round `k_0 + 2`. -/
def rhoSlotsZk (x : Claims.Msg) : List ℕ :=
  match setupOf true x with
  | some (_, spec, sts) => (List.range (2 * sts.size)).flatMap fun q =>
      match Zk.schedule sts[q / 2]!.m with
      | .ok (sch, _) => [repStartZk spec sts q + piopRoundsZk spec sts[q / 2]! + (sch.k0 + 2)]
      | .error _ => []
  | none => []

/-- **The inner proof's `β`**: rep `q`'s inner round `2`, after Ligerito's `k_0 + 7 + 11R` rounds. -/
def betaSlotsZk (x : Claims.Msg) : List ℕ :=
  match setupOf true x with
  | some (_, spec, sts) => (List.range (2 * sts.size)).flatMap fun q =>
      match Zk.schedule sts[q / 2]!.m with
      | .ok (sch, _) =>
        [repStartZk spec sts q + piopRoundsZk spec sts[q / 2]! + (sch.k0 + 7 + 11 * (sch.levels.size - 1)) + 2]
      | .error _ => []
  | none => []

/-- **The mask slots' rank checks**: table `i`'s statement, its rep `0`'s first coin and its rep `1`'s. -/
def rankSlotsZk (x : Claims.Msg) : List (Flock.Stmt × ℕ × ℕ) :=
  match setupOf true x with
  | some (_, spec, sts) => (List.range sts.size).flatMap fun i =>
      match Zk.schedule sts[i]!.m with
      | .ok _ => [(sts[i]!, repStartZk spec sts (2 * i), repStartZk spec sts (2 * i + 1))]
      | .error _ => []
  | none => []

/-- `εZk`: the outer coins, the `β`s, twice the pairs and `4·m` per table's rank checks, over `2^128`. -/
noncomputable def εZk (x : Claims.Msg) : ℝ :=
  ((outerSlotsZk x).length + (betaSlotsZk x).length + 2 * (rhoSlotsZk x).length +
    (((rankSlotsZk x).map fun s => 4 * s.1.m).sum : ℕ)) / 2 ^ 128

/-- **The honest masked record fits its length word** at the masked session `spec`, `sts` of `x`: at every witness,
tape and coins, a record the honest masked session returns, run against `H` with each send answered by its coin
(`PI.sim`), is under `2^64` bytes. -/
def RecordFitsZkAt (H : Claims.Msg → Claims.Msg) (x : Claims.Msg) (spec : SessionSpec) (sts : Array Flock.Stmt) :
    Prop :=
  ∀ (w : Claims.Flock.Wit) (ω : TapeZk) (r : Fin ((V true).k x) → Coin) (record : String) (proofs : Array ByteArray),
    ((sessionPZk asks Hashes.pi spec (tabsOf sts) (MaskRank.masksOf sts) (tracesOf sts w) ω).run.sim H
      (fun j => (Claims.coinSeq r j).bytes) 0).2 = .ok (record, proofs) →
      record.toUTF8.size < 2 ^ 64

open Classical in
/-- **The statements the masked verifier can accept** against `H`: as `ready H`, of the masked session, at one table,
with a `Hello` that names the keyed coin tree. -/
noncomputable def readyZk (H : Claims.Msg → Claims.Msg) (x : Claims.Msg) : Bool :=
  match setupOf true x with
  | some (s, spec, sts) => sts.size == 1 && s.read.draw.isSome && decide (x.length < 2 ^ 64) &&
      decide (∀ st ∈ sts, RegionsFit st) && decide (setupWith (Claims.hOf H) s.read = .ok (spec, sts)) &&
      spec.coinTree.isSome && decide (RecordFitsZkAt H x spec sts)
  | none => false

open Classical in
/-- **At SHA-512, `readyZk` asks no setup agreement** (`ready_sha512`). -/
theorem readyZk_sha512 (x : Claims.Msg) :
    readyZk sha512 x = match setupOf true x with
      | some (s, spec, sts) => sts.size == 1 && s.read.draw.isSome && decide (x.length < 2 ^ 64) &&
          decide (∀ st ∈ sts, RegionsFit st) && spec.coinTree.isSome && decide (RecordFitsZkAt sha512 x spec sts)
      | none => false := by
  unfold readyZk
  split
  · rename_i s spec sts hs
    rw [hOf_sha512, setupWith_sha512, Ins.setup_read (setupOf_some hs).2]
    simp
  · rfl

/-- **The error of `honestPZk`**, against `H`: `εZk` where `readyZk` holds, else `1`. -/
noncomputable def δcZk (H : Claims.Msg → Claims.Msg) (x : Claims.Msg) : ℝ := if readyZk H x then εZk x else 1

/-- **Flock's completeness at a prover and an error of one's own**: `Claims.Flock.Complete`'s statement at `D`, of `P`
at `δc`. -/
def CompleteWith {Coin Col T Ω : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Nonempty Ω]
    (H : Claims.Msg → Claims.Msg)
    (D : Claims.Flock.Deploy Coin Col T) (P : Claims.Flock.Wit → Claims.Prover Ω) (δc : Claims.Msg → ℝ) : Prop :=
  Claims.Complete H D.V P (Claims.Flock.R D.parse) (Claims.Flock.Scope D) δc

theorem complete_iff_with {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
    (H : Claims.Msg → Claims.Msg)
    (D : Claims.Flock.Deploy Coin Col T) : Claims.Flock.Complete H D ↔ CompleteWith H D D.P (D.δc H) :=
  Iff.rfl

namespace ZkTables

/-- **The honest masked record fits its length word** at the masked session `spec`, `sts` of `x`: at every witness,
tape and coins, a record the honest masked session on each table's share returns, run against `H` with each send
answered by its coin (`PI.sim`), is under `2^64` bytes. -/
def RecordFitsAt (H : Claims.Msg → Claims.Msg) (x : Claims.Msg) (spec : SessionSpec) (sts : Array Flock.Stmt) :
    Prop :=
  ∀ (w : Claims.Flock.Wit) (ω : TapeZk) (r : Fin ((V true).k x) → Coin) (record : String) (proofs : Array ByteArray),
    ((sessionPZk asks Hashes.pi spec (tabsOf sts) (MaskRank.masksOf sts) (traces sts w) ω).run.sim H
      (fun j => (Claims.coinSeq r j).bytes) 0).2 = .ok (record, proofs) →
      record.toUTF8.size < 2 ^ 64

open Classical in
/-- **The statements the masked verifier can accept** against `H`, at any number of tables: `readyZk`'s but its one
table, at the honest masked prover on each table's share. The session draws a unit, the statement is under `2^64`
bytes, every table's regions fit, its setup against `H` is the one at SHA-512, its `Hello` names the keyed coin tree,
and the honest record fits. -/
noncomputable def ready (H : Claims.Msg → Claims.Msg) (x : Claims.Msg) : Bool :=
  match setupOf true x with
  | some (s, spec, sts) => s.read.draw.isSome && decide (x.length < 2 ^ 64) &&
      decide (∀ st ∈ sts, RegionsFit st) && decide (setupWith (Claims.hOf H) s.read = .ok (spec, sts)) &&
      spec.coinTree.isSome && decide (RecordFitsAt H x spec sts)
  | none => false

open Classical in
/-- **At SHA-512, `ready` asks no setup agreement.** -/
theorem ready_sha512 (x : Claims.Msg) :
    ready sha512 x = match setupOf true x with
      | some (s, spec, sts) => s.read.draw.isSome && decide (x.length < 2 ^ 64) &&
          decide (∀ st ∈ sts, RegionsFit st) && spec.coinTree.isSome && decide (RecordFitsAt sha512 x spec sts)
      | none => false := by
  unfold ready
  split
  · rename_i s spec sts hs
    rw [hOf_sha512, setupWith_sha512, Ins.setup_read (setupOf_some hs).2]
    simp
  · rfl

/-- **Flock's `δc` at the masked verifier** (`deployZk`'s), at any number of tables: `εZk` where the verifier can
accept, else `1`. -/
noncomputable def δc (H : Claims.Msg → Claims.Msg) (x : Claims.Msg) : ℝ := if ready H x then εZk x else 1

/-- **At one table the honest record fits exactly when `honestPZk`'s does.** -/
theorem recordFitsAt_one {H : Claims.Msg → Claims.Msg} {x : Claims.Msg} {spec : SessionSpec} {sts : Array Flock.Stmt}
    (h1 : sts.size = 1) : RecordFitsAt H x spec sts ↔ RecordFitsZkAt H x spec sts := by
  unfold RecordFitsAt RecordFitsZkAt
  simp only [traces_one h1]

/-- **Where `readyZk` holds, `ready` holds.** -/
theorem ready_of_readyZk {H : Claims.Msg → Claims.Msg} {x : Claims.Msg} (h : readyZk H x = true) :
    ready H x = true := by
  unfold readyZk at h
  unfold ready
  split at h
  · simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at h
    simp only [Bool.and_eq_true, decide_eq_true_eq]
    exact ⟨⟨⟨⟨⟨h.1.1.1.1.1.2, h.1.1.1.1.2⟩, h.1.1.1.2⟩, h.1.1.2⟩, h.1.2⟩,
      (recordFitsAt_one h.1.1.1.1.1.1).2 h.2⟩
  · cases h

/-- **Where `readyZk` holds, `δc H` is `δcZk H`.** -/
theorem δc_eq_δcZk {H : Claims.Msg → Claims.Msg} {x : Claims.Msg} (h : readyZk H x = true) : δc H x = δcZk H x := by
  simp [δc, δcZk, h, ready_of_readyZk h]

end ZkTables

end FlockComplete
