import Proofs.Flock.Claims.Complete.Coins
import Proofs.Flock.Claims.Complete.Sim
import Proofs.Flock.Claims.Complete.Verdict
import Proofs.Flock.Claims.Scope.Matches

/-!
# The completeness proof's skeleton

`complete_of_goals`: Flock's plain verifier is complete for `honestP`, the record's reference prover, at the record's
error `δc H` (`Claims.Flock.Complete H (deployPlain rel)`), given one goal per check the honest run must pass.

* **The setting.** A good case (`Case`): a statement in scope that the verifier can accept (`ready H`), at its setup, a
  witness in the relation, a tape, and coins outside `Bad`. The honest run at it (`Run`): the pure session's record,
  proof files and sends' bodies (`sessionS`, which is the honest run, `honestRun_eq`).
* **What the verifier computes** on that run (`Run.sess`, `Run.proof`, `Run.zc`, …): each step's value where it returns
  (`val`), so each goal states only that its step returns, on the earlier steps' values.
* **The goals** (`Goals`): the session runs, its sends fit `k(x)`, the first is the link message, the last message's
  lengths fit their words; B's checks (the record's draw, its live session, the five custody comparisons); the verdict
  (§3, S1–S14, the public words, the schedules, the proof count, S17 twice); and per rep, S11, the extra claims, the proof
  file's decoding, R7/F2, zerocheck, lincheck, the ring switch, Ligerito, S15 and S16.
* **Proved here**: the last message's statement (`Run.stmt_ok`), the scope check (`Case.scope_ok`, from the statement
  being in scope), a drawn session (`Case.drawn`, from `ready`), the printed record's length (`Run.recordFits`, from
  `ready`), the link and last messages read back given their lengths (`Run.link_ok`, `Run.final_ok`, with the
  statement's from `ready`), and S18 (`Run.verdict_ok`: both reps' caps are the committed `root_B`, S16).
* **The composition**: `Run.rep_ok` (`verifyRepWith_ok`), `Run.verdict_ok` (`verifyWith_ok`), `Run.checks_ok'`
  (`checks_ok`), then `Run.acc` (`run_acc_of`), and `complete_of_goals` (`complete_of`).
-/

open Flock FlockDeploy Transcript

namespace FlockComplete

/-! ## The honest run at a good case -/

/-- **A good case**: a statement in scope that the verifier can accept, at its setup, a witness in the relation, a
tape, and coins outside `Bad`. -/
structure Case (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) where
  x : Claims.Msg
  w : Claims.Flock.Wit
  ω : Tape
  r : Fin ((V false).k x) → Coin
  s : Ins
  spec : SessionSpec
  sts : Array Flock.Stmt
  scope : Claims.Flock.Scope (deployPlain rel) x
  rel_w : Claims.Flock.R (deployPlain rel).parse x w
  ready : ready H x = true
  good : ¬ Bad x r
  setup : setupOf false x = some (s, spec, sts)

/-- The verifier's replies at the case's coins: each coin's bytes. -/
def Case.resp {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation} (c : Case H rel) :
    ℕ → Claims.Msg :=
  fun j => (Claims.coinSeq c.r j).bytes

/-- **The honest session ran** at a good case: the record, the proof files, and the sends' bodies. -/
structure Run (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) extends Case H rel where
  record : String
  proofs : Array ByteArray
  bodies : Array Claims.Msg
  ran : sessionS H (fun j => (Claims.coinSeq r j).bytes) spec sts w ω = (.ok (record, proofs), bodies)

/-! ## What the verifier computes on it

Each step's value where it returns, `default` where it refuses: the goals below say it returns. -/

/-- A step's value, `default` on a refusal. -/
def val {α : Type} [Inhabited α] : Except String α → α
  | .ok a => a
  | .error _ => default

namespace Run

/-- The hashes the verdict runs with: SHA-512 the oracle `H`. -/
def hs {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}
    (_ : Run H rel) : Flock.Hashes V := Flock.Hashes.ofSha512 (Claims.hOf H)

/-- Each table's setup. -/
def tabs {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}
    (o : Run H rel) : Array Setup := tabsOf o.sts

/-- The last message. -/
def final {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}
    (o : Run H rel) : Final := ⟨o.x, o.record, o.proofs⟩

/-- The sends: each body with tag `0`. -/
def sends {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}
    (o : Run H rel) : List Claims.Msg := o.bodies.toList.map (List.cons 0)

/-- The link the verifier reads off the first send. -/
def link {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}
    (o : Run H rel) : List ByteArray × List ByteArray := (linkOf (o.sends.headD [])).getD ([], [])

/-- The round messages and their coins, as the decision reads them. -/
def mids {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}
    (o : Run H rel) : List (Claims.Msg × Coin) :=
  midsOf (o.sends ++ [o.final.encode]) ((List.range (o.bodies.size + 1)).map (Claims.coinSeq o.r))

/-- The record's live session. -/
def live {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}
    (o : Run H rel) : Live :=
  match liveOf o.spec o.record with
  | .ok lv => lv
  | .error _ => ⟨#[], #[], #[], #[]⟩

/-- The record, decoded. -/
def sess {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}
    (o : Run H rel) : RunRecord := val (Record.decodeWith o.hs o.spec o.record)

/-- Table `i`'s schedule. -/
def sch {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}
    (o : Run H rel) (i : ℕ) : Schedule := (schedsOf o.tabs)[i]!

/-- Table `i`'s rep `r`'s stream. -/
def stream {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}
    (o : Run H rel) (i r : ℕ) : StreamRec :=
  (o.sess.streams.find? (·.name == s!"{o.spec.tables[i]!}/rep{r}")).getD default

/-- Table `i`'s extra claims at the link's points. -/
def extras {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}
    (o : Run H rel) (i : ℕ) : Array Claim := val (o.tabs[i]!.extra o.sess.points)

/-- Table `i`'s rep `r`'s proof file. -/
def bytes {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}
    (o : Run H rel) (i r : ℕ) : ByteArray := o.proofs[2 * i + r]!

/-- Table `i`'s rep `r`'s proof, decoded. -/
def proof {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}
    (o : Run H rel) (i r : ℕ) : Proof :=
  val (decodeProof (shapeOf o.tabs[i]! (o.sch i) (o.extras i).size) (o.bytes i r))

/-- The rep's replay transcript at its start. -/
def t₀ {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}
    (o : Run H rel) (i r : ℕ) : Transcript :=
  Transcript.init (o.stream i r).rounds (o.spec.domain o.spec.tables[i]! r).toUTF8

/-- Zerocheck's replay: its output and the transcript it leaves. -/
def zc {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}
    (o : Run H rel) (i r : ℕ) : ZcOut × Transcript :=
  val ((bindAndZerocheckWith o.hs o.tabs[i]!.m o.tabs[i]!.digest (flat (o.proof i r).cap)
    (o.proof i r).zerocheck).run (o.t₀ i r))

/-- Lincheck's replay. -/
def lc {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}
    (o : Run H rel) (i r : ℕ) : LcOut × Transcript :=
  val ((lincheckWith o.hs o.tabs[i]!.fold (o.zc i r).1 (o.proof i r).lincheck).run (o.zc i r).2)

/-- The ring switch's replay: the batched claim `T`, the switched claims, and the transcript. -/
def sw {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}
    (o : Run H rel) (i r : ℕ) : (F128 × Array Switched) × Transcript :=
  val ((ringSwitchWith o.hs o.tabs[i]!.m
    (abcClaims o.tabs[i]!.m o.tabs[i]!.kLog (o.zc i r).1 (o.lc i r).1 ++ o.extras i)
    (o.proof i r).ringSwitches).run (o.lc i r).2)

/-- Ligerito's replay: the transcript it leaves. -/
def lg {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}
    (o : Run H rel) (i r : ℕ) : Unit × Transcript :=
  val ((ligeritoWith o.hs o.tabs[i]!.merkle (o.sch i) (o.sw i r).1.1 (o.proof i r).cap (o.proof i r).ligerito
    (basis (o.sw i r).1.2)).run (o.sw i r).2)

end Run


/-! ## The goals

One per check the honest run must pass, each at every honest run at a good case (`Run`), on the values the verifier
computes from the earlier checks. -/

/-- **The session runs**: at every good case the honest session returns its record and proof files. -/
def SessionRuns (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) : Prop :=
  ∀ c : Case H rel, ∃ record proofs bodies,
    sessionS H c.resp c.spec c.sts c.w c.ω = (.ok (record, proofs), bodies)

/-- **The sends fit the verifier's rounds** (`k(x)`): the link message and the recorded rounds. -/
def SendCount (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) : Prop := ∀ o : Run H rel, o.bodies.size ≤ 1 + rounds false o.spec o.sts

/-- **The first send is the link message**, each length in its word. -/
def LinkFirst (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) : Prop :=
  ∀ o : Run H rel, ∃ rootB publics : List ByteArray, o.bodies.toList.head? = some (linkBody rootB publics) ∧
    rootB.length < 2 ^ 64 ∧ publics.length < 2 ^ 64 ∧ (∀ b ∈ rootB, b.size < 2 ^ 64) ∧ ∀ b ∈ publics, b.size < 2 ^ 64

/-- **The last message's lengths fit their words** (the statement's is `ready`'s). -/
def FinalFits (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) : Prop :=
  ∀ o : Run H rel, o.record.toUTF8.size < 2 ^ 64 ∧ o.proofs.size < 2 ^ 64 ∧ ∀ b ∈ o.proofs, b.size < 2 ^ 64

/-! ### B's checks -/

/-- **The record's draw is the statement's**, as the verifier compares them (`drawEq`). -/
def DrawMatches (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) : Prop :=
  ∀ o : Run H rel, drawEq (FlockSoundness.Discharge.Exec.recordDraw o.record) o.s.read.draw = true

/-- **The record's live session reads** (`liveOf`: the record is JSON with its link, points and streams). -/
def LiveReads (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) : Prop := ∀ o : Run H rel, liveOf o.spec o.record = .ok o.live

/-- **The record is in its canonical text** (S0, `Flock.RecordText.check`). -/
def RecordCanonical (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) : Prop :=
  ∀ o : Run H rel, Flock.RecordText.check o.record = .ok ()

/-- **Custody: the record's `root_B` is the link message's.** -/
def CustodyRootB (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) : Prop := ∀ o : Run H rel, (o.live.rootB.toList == o.link.1) = true

/-- **Custody: the record's public words are the link message's.** -/
def CustodyPublics (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) : Prop := ∀ o : Run H rel, (o.live.publics.toList == o.link.2) = true

/-- **Custody: the record's link points are the first coin's.** -/
def CustodyPoints (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) : Prop :=
  ∀ o : Run H rel, (o.live.points == pointsOf o.spec (Claims.coinSeq o.r 0)) = true

/-- **Custody: the record has one round per round message.** -/
def CustodyCount (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) : Prop :=
  ∀ o : Run H rel, ((o.live.streams.toList.flatMap (·.toList)).length == o.mids.length) = true

/-- **Custody: each recorded round is its message and its coin's first `n` elements.** -/
def CustodyRounds (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) : Prop :=
  ∀ o : Run H rel, ((o.live.streams.toList.flatMap (·.toList)).zip o.mids).all
    (fun (r, a, c) => r.bytes == some ⟨a.toArray⟩ && r.coins == c.vals r.n) = true

/-! ### The verdict -/

/-- **One statement per table** (§3). -/
def Tables (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) : Prop := ∀ o : Run H rel, (o.tabs.size != o.spec.tables.size || o.tabs.size == 0) = false

/-- **The record decodes** (S1–S14). -/
def Decodes (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) : Prop := ∀ o : Run H rel, Record.decodeWith o.hs o.spec o.record = .ok o.sess

/-- **Each table's public words are its statement's.** -/
def Publics (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) : Prop := ∀ o : Run H rel, ∀ q ∈ (o.tabs.zip o.sess.publics).toList, q.1.publics q.2 = .ok ()

/-- **Each table has its fast100 schedule.** -/
def Schedules (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) : Prop := ∀ o : Run H rel, ∀ st ∈ o.tabs.toList, (fast100 st.m).isSome = true

/-- **One proof file per rep stream.** -/
def ProofCount (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) : Prop := ∀ o : Run H rel, (o.proofs.size != o.spec.repStreams.size) = false

/-- **The record's proof digests list exactly the rep streams** (S17). -/
def ProofList (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) : Prop :=
  ∀ o : Run H rel, (o.sess.proofSha.size != o.spec.repStreams.size ||
    !o.spec.repStreams.all fun a => o.sess.proofSha.any fun x => x.1 == a.2.2) = false

/-- **Each proof file is the one the record's digest names** (S17). -/
def ProofDigests (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) : Prop :=
  ∀ o : Run H rel, ∀ q ∈ (o.spec.repStreams.zip o.proofs).toList,
    ((o.sess.proofSha.find? (·.1 == q.1.2.2)).map (·.2) != some (hexOf (dgOf o.spec (Claims.hOf H) q.2))) = false

/-! ### Each rep: table `i < |tables|`, rep `r < 2` -/

/-- **The rep's stream is recorded** (S11). -/
def RepStream (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) : Prop :=
  ∀ o : Run H rel, ∀ i < o.tabs.size, ∀ r < 2,
    (o.sess.streams.find? (·.name == s!"{o.spec.tables[i]!}/rep{r}")).isSome = true

/-- **The table's extra claims evaluate** at the link's points. -/
def RepExtras (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) : Prop :=
  ∀ o : Run H rel, ∀ i < o.tabs.size, o.tabs[i]!.extra o.sess.points = .ok (o.extras i)

/-- **The rep's proof file decodes** at its shape. -/
def RepDecodes (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) : Prop :=
  ∀ o : Run H rel, ∀ i < o.tabs.size, ∀ r < 2,
    decodeProof (shapeOf o.tabs[i]! (o.sch i) (o.extras i).size) (o.bytes i r) = .ok (o.proof i r)

/-- **The rep's PCS parameters are fast100's** (R7/F2). -/
def RepParams (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) : Prop :=
  ∀ o : Run H rel, ∀ i < o.tabs.size, ∀ r < 2, ((o.proof i r).params != pcsParams o.tabs[i]!.merkle (o.sch i)) = false

/-- **Zerocheck replays** from the rep's recorded rounds (§9). -/
def Zerocheck (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) : Prop :=
  ∀ o : Run H rel, ∀ i < o.tabs.size, ∀ r < 2,
    (bindAndZerocheckWith o.hs o.tabs[i]!.m o.tabs[i]!.digest (flat (o.proof i r).cap) (o.proof i r).zerocheck).run
      (o.t₀ i r) = .ok (o.zc i r)

/-- **Lincheck replays** from where zerocheck left the transcript (§10). -/
def Lincheck (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) : Prop :=
  ∀ o : Run H rel, ∀ i < o.tabs.size, ∀ r < 2,
    (lincheckWith o.hs o.tabs[i]!.fold (o.zc i r).1 (o.proof i r).lincheck).run (o.zc i r).2 = .ok (o.lc i r)

/-- **The ring switch replays** on zerocheck's and lincheck's claims and the extra claims (§11). -/
def RingSwitch (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) : Prop :=
  ∀ o : Run H rel, ∀ i < o.tabs.size, ∀ r < 2,
    (ringSwitchWith o.hs o.tabs[i]!.m (abcClaims o.tabs[i]!.m o.tabs[i]!.kLog (o.zc i r).1 (o.lc i r).1 ++ o.extras i)
      (o.proof i r).ringSwitches).run (o.lc i r).2 = .ok (o.sw i r)

/-- **Ligerito replays** on the batched claim, with its Merkle openings against the committed rows (§12–§14). -/
def Ligerito (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) : Prop :=
  ∀ o : Run H rel, ∀ i < o.tabs.size, ∀ r < 2,
    (ligeritoWith o.hs o.tabs[i]!.merkle (o.sch i) (o.sw i r).1.1 (o.proof i r).cap (o.proof i r).ligerito
      (basis (o.sw i r).1.2)).run (o.sw i r).2 = .ok (o.lg i r)

/-- **Every recorded round is consumed** (S15). -/
def Consumed (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) : Prop :=
  ∀ o : Run H rel, ∀ i < o.tabs.size, ∀ r < 2, ((o.lg i r).2.k != (o.stream i r).rounds.size) = false

/-- **The rep's cap is the committed `root_B`** (S16). -/
def CapCommitted (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) : Prop :=
  ∀ o : Run H rel, ∀ i < o.tabs.size, ∀ r < 2, (flat (o.proof i r).cap != o.sess.rootB[i]!) = false

/-- **The goals**, every check but the five proved here (the statement, the scope, a drawn session, the link and last
messages read back given their lengths, and S18 from S16). -/
structure Goals (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) : Prop where
  session : SessionRuns H rel
  sends : SendCount H rel
  linkFirst : LinkFirst H rel
  finalFits : FinalFits H rel
  draw : DrawMatches H rel
  liveReads : LiveReads H rel
  rootB : CustodyRootB H rel
  publicsLink : CustodyPublics H rel
  points : CustodyPoints H rel
  count : CustodyCount H rel
  rounds : CustodyRounds H rel
  tables : Tables H rel
  decodes : Decodes H rel
  publics : Publics H rel
  schedules : Schedules H rel
  proofCount : ProofCount H rel
  proofList : ProofList H rel
  proofDigests : ProofDigests H rel
  repStream : RepStream H rel
  repExtras : RepExtras H rel
  repDecodes : RepDecodes H rel
  repParams : RepParams H rel
  zerocheck : Zerocheck H rel
  lincheck : Lincheck H rel
  ringSwitch : RingSwitch H rel
  ligerito : Ligerito H rel
  consumed : Consumed H rel
  capCommitted : CapCommitted H rel
  canonical : RecordCanonical H rel

/-! ## The checks proved here -/

/-- `ready` at the case's setup: the session draws a unit, the statement is under `2^64` bytes, each table's regions
fit, the setup against `H` is the case's, and the honest record fits. -/
theorem Case.ready_parts {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation} (c : Case H rel) :
    c.s.read.draw.isSome = true ∧ c.x.length < 2 ^ 64 ∧ (∀ st ∈ c.sts, RegionsFit st) ∧
      setupWith (Claims.hOf H) c.s.read = .ok (c.spec, c.sts) ∧ RecordFitsAt H c.x c.spec c.sts := by
  have h := c.ready
  unfold FlockComplete.ready at h
  rw [c.setup] at h
  simpa [and_assoc] using h

/-- **The printed honest record is under `2^64` bytes**: `ready` asks it of every honest session. -/
theorem Run.recordFits {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation} (o : Run H rel) :
    o.record.toUTF8.size < 2 ^ 64 :=
  o.ready_parts.2.2.2.2 o.w o.ω o.r o.record o.proofs o.bodies o.ran

/-- **A drawn session**: `checks` never reaches `drawLaw zk tags false none`. -/
theorem Case.drawn {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation} (c : Case H rel) : c.s.read.draw.isNone = false := by
  have := c.ready_parts.1
  cases h : c.s.read.draw
  · rw [h] at this; cases this
  · rfl

/-- **The scope check passes** (`ProvedScope.check`): the statement is in scope. -/
theorem Case.scope_ok {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation} (c : Case H rel) : c.s.read.check = .ok () := by
  obtain ⟨p, hp, hsc⟩ := (Readable.scope_iff (deployPlain rel) rel (readOf false) rfl
    (fun _ _ h => readOf_zk h) c.x).1 c.scope
  have hi := (setupOf_some c.setup).1
  unfold Readable.parseBytes readOf at hp
  rw [hi] at hp
  have ht := Readable.toRead_of_parse (r := c.s.read) hp
  rw [← ht]
  exact ((Readable.scope_ok_iff p).1 hsc).1

/-- **The last message's statement is the statement.** -/
theorem Run.stmt_ok {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation} (o : Run H rel) : (o.final.stmt == o.x) = true := beq_self_eq_true _

/-- **The link message reads back** as the verifier's link. -/
theorem Run.link_ok {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation} (hL : LinkFirst H rel) (o : Run H rel) :
    linkOf (o.sends.headD []) = some o.link := by
  obtain ⟨rootB, publics, hh, h1, h2, h3, h4⟩ := hL o
  have e : o.sends.headD [] = 0 :: linkBody rootB publics := by
    unfold Run.sends
    cases hb : o.bodies.toList with
    | nil => rw [hb] at hh; cases hh
    | cons b bs =>
      rw [hb] at hh
      simp only [List.head?_cons, Option.some.injEq] at hh
      simp [hh]
  have hl := linkOf_linkBody h1 h2 h3 h4
  unfold Run.link
  rw [e, hl]
  rfl

/-- **The last message reads back.** -/
theorem Run.final_ok {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation} (hF : FinalFits H rel) (o : Run H rel) :
    finalOf o.final.encode = some o.final := by
  obtain ⟨h1, h2, h3⟩ := hF o
  exact finalOf_encode o.ready_parts.2.1 h1 h2 h3

theorem bytes_eq_of_bne {a b : ByteArray} (h : (a != b) = false) : a = b := by
  have h' : (a.data == b.data) = true := by
    have : (!(a.data == b.data)) = false := h
    simpa using this
  exact ByteArray.ext (beq_iff_eq.1 h')

/-! ## The composition -/

/-- **A rep accepts** (`verifyRepWith_ok`), with its cap the committed `root_B`. -/
theorem Run.rep_ok {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation} (g : Goals H rel) (o : Run H rel) {i : ℕ} (hi : i < o.tabs.size) {r : ℕ}
    (hr : r < 2) : verifyRepWith o.hs o.spec o.tabs[i]! (o.sch i) o.sess i r (o.bytes i r) = .ok o.sess.rootB[i]! := by
  have hS := g.repStream o i hi r hr
  have hfind : o.sess.streams.find? (·.name == s!"{o.spec.tables[i]!}/rep{r}") = some (o.stream i r) := by
    unfold Run.stream
    cases hf : o.sess.streams.find? (·.name == s!"{o.spec.tables[i]!}/rep{r}") with
    | none => rw [hf] at hS; cases hS
    | some st => rfl
  rw [← bytes_eq_of_bne (g.capCommitted o i hi r hr)]
  exact verifyRepWith_ok o.hs o.spec o.tabs[i]! (o.sch i) o.sess i r (o.bytes i r) hfind (g.repExtras o i hi)
    (g.repDecodes o i hi r hr) (g.repParams o i hi r hr) (g.zerocheck o i hi r hr) (g.lincheck o i hi r hr)
    (g.ringSwitch o i hi r hr) (g.ligerito o i hi r hr) (g.consumed o i hi r hr) (g.capCommitted o i hi r hr)

/-- **The plain verdict accepts** (`verifyWith_ok`); S18 from S16, both reps' caps the committed `root_B`. -/
theorem Run.verdict_ok {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation} (g : Goals H rel) (o : Run H rel) :
    verdictWith false (Claims.hOf H) o.spec (tabsOf o.sts) o.record o.proofs = .ok () :=
  verifyWith_ok (Claims.hOf H) o.spec o.tabs o.record o.proofs (g.tables o) (g.decodes o) (g.publics o)
    (g.schedules o) (g.proofCount o) (g.proofList o) (g.proofDigests o)
    fun i hi => ⟨o.sess.rootB[i]!, o.rep_ok g hi (by decide), o.rep_ok g hi (by decide)⟩

/-- **B's checks pass** (`checks_ok`). -/
theorem Run.checks_ok' {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation} (g : Goals H rel) (o : Run H rel) :
    checks false o.s o.spec o.x o.link (Claims.coinSeq o.r 0) o.mids o.final = .ok () :=
  checks_ok o.stmt_ok (g.draw o) (g.liveReads o)
    (custody_ok (g.rootB o) (g.publicsLink o) (g.points o) (g.count o) (g.rounds o)) o.scope_ok o.drawn (g.canonical o)

/-- **The honest prover is accepted** at every honest run at a good case. -/
theorem Run.acc {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation} (g : Goals H rel) (o : Run H rel) :
    (Claims.run H (honestP o.w) (V false) o.x o.ω o.r).out = .acc := by
  have he := honestRun_eq H o.x o.w o.ω o.r o.setup o.ran
  have hk : (V false).k o.x = 2 + rounds false o.spec o.sts := by
    show (match setupOf false o.x with
      | some (_, spec, sts) => 2 + rounds false spec sts
      | none => 1) = _
    rw [o.setup]
  have hlen : (honestRun H o.x o.w o.ω o.r).1.length = o.bodies.size := by
    rw [he]; simp
  refine run_acc_of H o.w o.x o.ω o.r (link := o.link) (F := o.final) o.setup o.ready_parts.2.2.2.1 ?_ ?_ ?_ ?_ ?_
    (o.verdict_ok g) ?_
  · rw [hlen, hk]
    have := g.sends o
    omega
  · rw [he]
    intro m hm
    obtain ⟨b, -, rfl⟩ := List.mem_map.1 hm
    exact ⟨b, rfl⟩
  · rw [he]
    obtain ⟨rootB, publics, hh, -⟩ := g.linkFirst o
    intro h0
    simp only [List.map_eq_nil_iff, Array.toList_eq_nil_iff] at h0
    rw [Array.toList_eq_nil_iff.2 h0] at hh
    cases hh
  · rw [he]
    exact o.link_ok g.linkFirst
  · rw [he]
    exact o.final_ok g.finalFits
  · have hc := o.checks_ok' g
    rw [he]
    unfold Run.mids Run.sends at hc
    simp only [List.length_map, Array.length_toList]
    exact hc

/-- **Completeness from the goals**: Flock's plain verifier is complete for `honestP` at the record's error, `δc H`,
given each goal. -/
theorem complete_of_goals (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) (g : Goals H rel) :
    Claims.Flock.Complete H (deployPlain rel) := by
  refine complete_of H rel fun x w hsc hr hready ω r hgood => ?_
  have hset : ∃ s spec sts, setupOf false x = some (s, spec, sts) := by
    unfold ready at hready
    split at hready
    · rename_i s spec sts hs
      exact ⟨s, spec, sts, hs⟩
    · cases hready
  obtain ⟨s, spec, sts, hs⟩ := hset
  let c : Case H rel := ⟨x, w, ω, r, s, spec, sts, hsc, hr, hready, hgood, hs⟩
  obtain ⟨record, proofs, bodies, hran⟩ := g.session c
  exact Run.acc g ⟨c, record, proofs, bodies, hran⟩

end FlockComplete
