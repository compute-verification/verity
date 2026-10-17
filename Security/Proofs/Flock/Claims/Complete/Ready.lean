import Proofs.Flock.Claims.Complete.Sim
import Proofs.Flock.Claims.Complete.Zk.Honest

/-!
# The deployed completeness error

`deployPlain`'s and `deployZk`'s `δc` (`Claims.Flock.Deploy.δc`), the error `Claims.Flock.Complete` reads: at the hash
`H`, `δc H x` is the bad coins' count over `2^128` (`ε`, `εZk`) on the statements the verifier can accept against `H`
(`ready H`, `readyZk H`), and `1` on every other statement.

* **The bad coins** (`Coins`: `pr_bad_le`): zerocheck's outer coins (`outerSlots`): rep `q`'s second round (`rOuter`,
  after `r_skip`), elements `0` to `m − 14`, at its index among the verifier's coins (`repStart`: the link's coin, then
  the earlier reps' recorded rounds, `spec.repStreams` order). At `r = 1` the verifier's `(1 + r)⁻¹` is `0⁻¹ = 0`. The
  masked verifier adds each rep's inner `β` (`betaSlotsZk`, refused at `0`) and level 0's pair coins (`rhoSlotsZk`,
  refused when dependent), counted twice (`εZk`).
* **`δc H x = 1` exactly where `ready H x` fails**, which is on the statements that:
  * set up no plain session (`setupOf false x = none`), which the verifier refuses at every run;
  * set up a session that draws no unit: `checks` ends in `drawLaw zk none`, which refuses;
  * are `2^64` bytes or more: the last message (`Final.encode`) writes the statement's length in a 64-bit word;
  * have a table whose regions do not fit (`RegionsFit`: some region's bytes at some block are not its free bits): the
    extra claims refuse (`Stmt.regionValue`), the honest prover's as the verifier's, and the relation, on the circuit's
    values, does not pin the public bytes the regions read;
  * set up otherwise against `H` than at SHA-512: the verifier sets the statement up against `H`
    (`setupWith (hOf H)`), recomputing with `H` the SHA-512 digests the statement pins, and the honest prover sets up
    at SHA-512. At `sha512` the two agree (`ready_sha512`);
  * have an honest record that does not fit (`RecordFitsAt`): at some witness, tape and coins, the honest session's
    printed record is `2^64` bytes or more, past the last message's 64-bit word for its length.
* **`δcZk H x = 1` exactly where `readyZk H x` fails**: the same of the masked session (`setupOf true`, and
  `RecordFitsZkAt` at the masked session's run), and two more. The session has one table: the masked completeness
  proof is per table, so a `--zk` statement of several tables gets `1`. Its `Hello` names the keyed coin tree
  (`spec.coinTree`): the masked verifier refuses any other session.
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

/-- `εZk`: the outer coins, the `β`s and twice the pairs, over `2^128`. -/
noncomputable def εZk (x : Claims.Msg) : ℝ :=
  ((outerSlotsZk x).length + (betaSlotsZk x).length + 2 * (rhoSlotsZk x).length) / 2 ^ 128

/-- **The honest masked record fits its length word** at the masked session `spec`, `sts` of `x`: at every witness,
tape and coins, a record the honest masked session returns, run against `H` with each send answered by its coin
(`PI.sim`), is under `2^64` bytes. -/
def RecordFitsZkAt (H : Claims.Msg → Claims.Msg) (x : Claims.Msg) (spec : SessionSpec) (sts : Array Flock.Stmt) :
    Prop :=
  ∀ (w : Claims.Flock.Wit) (ω : TapeZk) (r : Fin ((V true).k x) → Coin) (record : String) (proofs : Array ByteArray),
    ((sessionPZk asks Hashes.pi spec (tabsOf sts) (tracesOf sts w) ω).run.sim H
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

/-- **Flock's `δc` at the masked verifier** against `H`: `εZk` where the verifier can accept, else `1`. -/
noncomputable def δcZk (H : Claims.Msg → Claims.Msg) (x : Claims.Msg) : ℝ := if readyZk H x then εZk x else 1

end FlockComplete
