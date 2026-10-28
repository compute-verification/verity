import Proofs.Flock.Claims.Complete.Skeleton
import Proofs.Flock.Claims.Complete.ZkTables.Ready
import Proofs.Flock.Claims.Complete.ZkTables.Run
import Proofs.Flock.Claims.Complete.Zk.Verdict

/-!
# The masked completeness proof's skeleton, at any number of tables

`complete_of_goals`: Flock's masked verifier is complete for `honestP` with error `δc H`
(`Claims.Flock.Complete H (deployZk rel)`), given one goal per check the honest masked run must pass.
This is `Zk/Skeleton.lean`'s at a good case whose readiness is `ready` (any number of tables) in place of `readyZk`,
and whose run is the masked session on each table's share of the witness (`sessionS`): each goal is `GoalsZk`'s of the
same name, at `Run`, and already stated per table (`∀ i < |tables|, ∀ r < 2`).

* **Proved here**: the last message's statement, the scope check (`Case.scope_ok`), a drawn session, the link and last
  messages read back given their lengths, S18 from S16, and the keyed coin tree (`coinTree_goal`, from `ready`).
* **The composition**: `Run.rep_ok` (`zk_verifyRepWith_ok`), `Run.verdict_ok` (`zk_verifyWith_ok`), `Run.checks_ok'`
  (`checks_ok`), `Run.acc` (`run_acc_of`), and `complete_of_goals` (`complete_of`).
-/

open Flock FlockDeploy Transcript

namespace FlockComplete.ZkTables

/-! ## The honest masked run at a good case -/

/-- **A good masked case**: a statement in scope that the masked verifier can accept (`ready`), at its setup, a witness
in the relation, a masked tape, and coins outside `BadZk`. -/
structure Case (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) where
  x : Claims.Msg
  w : Claims.Flock.Wit
  ω : TapeZk
  r : Fin ((V true).k x) → Coin
  s : Ins
  spec : SessionSpec
  sts : Array Flock.Stmt
  scope : Claims.Flock.Scope (deployZk rel) x
  rel_w : Claims.Flock.R (deployZk rel).parse x w
  ready : ready H x = true
  good : ¬ BadZk x r
  setup : setupOf true x = some (s, spec, sts)

/-- The verifier's replies at the case's coins: each coin's bytes. -/
def Case.resp {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation} (c : Case H rel) :
    ℕ → Claims.Msg :=
  fun j => (Claims.coinSeq c.r j).bytes

/-- **The honest masked session ran** at a good case: the record, the proof files, and the sends' bodies. -/
structure Run (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) extends Case H rel where
  record : String
  proofs : Array ByteArray
  bodies : Array Claims.Msg
  ran : sessionS H (fun j => (Claims.coinSeq r j).bytes) spec sts w ω = (.ok (record, proofs), bodies)

/-! ## What the masked verifier computes on it

Each step's value where it returns, `default` where it refuses: the goals below say it returns. -/

namespace Run

variable {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}

/-- The hashes the verdict runs with: SHA-512 the oracle `H`. -/
def hs (_ : Run H rel) : Flock.Hashes V := Flock.Hashes.ofSha512 (Claims.hOf H)

/-- Each table's setup. -/
def tabs (o : Run H rel) : Array Setup := tabsOf o.sts

/-- The last message. -/
def final (o : Run H rel) : Final := ⟨o.x, o.record, o.proofs⟩

/-- The sends: each body with tag `0`. -/
def sends (o : Run H rel) : List Claims.Msg := o.bodies.toList.map (List.cons 0)

/-- The link the verifier reads off the first send. -/
def link (o : Run H rel) : List ByteArray × List ByteArray := (linkOf (o.sends.headD [])).getD ([], [])

/-- The round messages and their coins, as the decision reads them. -/
def mids (o : Run H rel) : List (Claims.Msg × Coin) :=
  midsOf (o.sends ++ [o.final.encode]) ((List.range (o.bodies.size + 1)).map (Claims.coinSeq o.r))

/-- The record's live session. -/
def live (o : Run H rel) : Live :=
  match liveOf o.spec o.record with
  | .ok lv => lv
  | .error _ => ⟨#[], #[], #[], #[]⟩

/-- The record, decoded. -/
def sess (o : Run H rel) : RunRecord := val (Record.decodeWith o.hs o.spec o.record)

/-- Table `i`'s `--zk` schedule. -/
def sch (o : Run H rel) (i : ℕ) : Schedule := (zkSchedsOf o.tabs)[i]!.1

/-- Table `i`'s level-0 padding `t_pad`. -/
def tPad (o : Run H rel) (i : ℕ) : ℕ := (zkSchedsOf o.tabs)[i]!.2

/-- Table `i`'s rep `r`'s stream. -/
def stream (o : Run H rel) (i r : ℕ) : StreamRec :=
  (o.sess.streams.find? (·.name == s!"{o.spec.tables[i]!}/rep{r}")).getD default

/-- Table `i`'s extra claims at the link's points. -/
def extras (o : Run H rel) (i : ℕ) : Array Claim := val (o.tabs[i]!.extra o.sess.points)

/-- Table `i`'s rep `r`'s proof file. -/
def bytes (o : Run H rel) (i r : ℕ) : ByteArray := o.proofs[2 * i + r]!

/-- Table `i`'s rep `r`'s masked proof, decoded. -/
def zp (o : Run H rel) (i r : ℕ) : Zk.ZkProof :=
  val (Zk.decode (shapeZk o.tabs[i]! (o.sch i) (o.extras i).size) (o.bytes i r))

/-- Table `i`'s pad layout. -/
def lay (o : Run H rel) (i : ℕ) : Zk.PadLayout := Zk.PadLayout.new o.tabs[i]!.m o.tabs[i]!.kLog (o.sch i).k0

/-- The rep's replay transcript at its start. -/
def t₀ (o : Run H rel) (i r : ℕ) : Transcript :=
  Transcript.init (o.stream i r).rounds (o.spec.domain o.spec.tables[i]! r).toUTF8

/-- The masked replay of zerocheck and lincheck: the claims and constraints, and the transcript it leaves. -/
def rp (o : Run H rel) (i r : ℕ) : Zk.Replayed × Transcript :=
  val ((Zk.replayWith o.hs o.tabs[i]! (o.lay i) (o.zp i r).proof (o.zp i r).inner.root).run (o.t₀ i r))

/-- The ring switch's replay on the replay's claims valued by their checks and the extra claims. -/
def sw (o : Run H rel) (i r : ℕ) : (F128 × Array Switched) × Transcript :=
  val ((ringSwitchWith o.hs o.tabs[i]!.m (ringClaimsZk (o.rp i r).1 (o.zp i r).proof ++ o.extras i)
    (o.zp i r).proof.ringSwitches).run (o.rp i r).2)

/-- The masked Ligerito's replay: level 0's record, and the transcript it leaves. -/
def lg (o : Run H rel) (i r : ℕ) : Zk.L0Rec × Transcript :=
  val ((Zk.ligeritoWith o.hs o.tabs[i]!.merkle (o.sch i) (o.sw i r).1.1 (o.zp i r).proof.cap
    (o.zp i r).proof.ligerito (basis (o.sw i r).1.2) ⟨o.tPad i, r, (o.zp i r).level0⟩).run (o.sw i r).2)

/-- Level 0's lane phase as constraints. -/
def lanes (o : Run H rel) (i r : ℕ) : Array (String × Zk.Lin) :=
  val (Zk.ligeritoConstraints (o.lay i) (o.lg i r).1 (o.zp i r).proof.ligerito (o.zp i r).level0)

/-- The inner proof's replay on every constraint: the transcript it leaves. -/
def inner (o : Run H rel) (i r : ℕ) : Unit × Transcript :=
  val ((Zk.verifyInnerWith o.hs (o.lay i) (o.zp i r).inner (o.zp i r).proof.zerocheck.finalC
    (repConsZk (o.rp i r).1 (o.zp i r).proof ++ o.lanes i r)).run (o.lg i r).2)

end Run

/-! ## The goals

`GoalsZk`'s, by name, at every honest masked run at a good case of any number of tables (`Run`). -/

section Goals

variable (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation)

/-- **The masked session runs**: at every good case the honest masked session returns its record and proof files. -/
def SessionRunsZk : Prop :=
  ∀ c : Case H rel, ∃ record proofs bodies, sessionS H c.resp c.spec c.sts c.w c.ω = (.ok (record, proofs), bodies)

/-- **The sends fit the masked verifier's rounds** (`k(x)`): the link message and the recorded rounds. -/
def SendCountZk : Prop := ∀ o : Run H rel, o.bodies.size ≤ 1 + rounds true o.spec o.sts

/-- **The first send is the link message**, each length in its word. -/
def LinkFirstZk : Prop :=
  ∀ o : Run H rel, ∃ rootB publics : List ByteArray, o.bodies.toList.head? = some (linkBody rootB publics) ∧
    rootB.length < 2 ^ 64 ∧ publics.length < 2 ^ 64 ∧ (∀ b ∈ rootB, b.size < 2 ^ 64) ∧ ∀ b ∈ publics, b.size < 2 ^ 64

/-- **The last message's lengths fit their words** (the statement's is `ready`'s). -/
def FinalFitsZk : Prop :=
  ∀ o : Run H rel, o.record.toUTF8.size < 2 ^ 64 ∧ o.proofs.size < 2 ^ 64 ∧ ∀ b ∈ o.proofs, b.size < 2 ^ 64

/-- **The record's draw is the statement's**, whatever its key order (`drawEq`). -/
def DrawMatchesZk : Prop :=
  ∀ o : Run H rel, drawEq (FlockSoundness.Discharge.Exec.recordDraw o.record) o.s.read.draw = true

/-- **The record's live session reads** (`liveOf`: the record is JSON with its link, points and streams). -/
def LiveReadsZk : Prop := ∀ o : Run H rel, liveOf o.spec o.record = .ok o.live

/-- **The record is in its canonical text** (S0, `Flock.RecordText.check`). -/
def RecordCanonicalZk : Prop := ∀ o : Run H rel, Flock.RecordText.check o.record = .ok ()

/-- **Custody: the record's `root_B` is the link message's.** -/
def CustodyRootBZk : Prop := ∀ o : Run H rel, (o.live.rootB.toList == o.link.1) = true

/-- **Custody: the record's public words are the link message's.** -/
def CustodyPublicsZk : Prop := ∀ o : Run H rel, (o.live.publics.toList == o.link.2) = true

/-- **Custody: the record's link points are the first coin's.** -/
def CustodyPointsZk : Prop := ∀ o : Run H rel, (o.live.points == pointsOf o.spec (Claims.coinSeq o.r 0)) = true

/-- **Custody: the record has one round per round message.** -/
def CustodyCountZk : Prop :=
  ∀ o : Run H rel, ((o.live.streams.toList.flatMap (·.toList)).length == o.mids.length) = true

/-- **Custody: each recorded round is its message and its coin's first `n` elements.** -/
def CustodyRoundsZk : Prop :=
  ∀ o : Run H rel, ((o.live.streams.toList.flatMap (·.toList)).zip o.mids).all
    (fun (r, a, c) => r.bytes == some ⟨a.toArray⟩ && r.coins == c.vals r.n) = true

/-- **One statement per table** (§3). -/
def TablesZk : Prop := ∀ o : Run H rel, (o.tabs.size != o.spec.tables.size || o.tabs.size == 0) = false

/-- **The session's coins are the keyed coin tree** (the spec's `coin_tree`, from its `hello`). -/
def CoinTree : Prop := ∀ o : Run H rel, o.spec.coinTree.isNone = false

/-- **The record decodes** (S1–S14). -/
def DecodesZk : Prop := ∀ o : Run H rel, Record.decodeWith o.hs o.spec o.record = .ok o.sess

/-- **Each table's public words are its statement's.** -/
def PublicsZk : Prop := ∀ o : Run H rel, ∀ q ∈ (o.tabs.zip o.sess.publics).toList, q.1.publics q.2 = .ok ()

/-- **Each table has its `--zk` schedule** (`Zk.schedule`: fast100's, level 0 at `q_0` queries). -/
def SchedulesZk : Prop := ∀ o : Run H rel, ∀ st ∈ o.tabs.toList, ∃ v, Zk.schedule st.m = .ok v

/-- **One proof file per rep stream.** -/
def ProofCountZk : Prop := ∀ o : Run H rel, (o.proofs.size != o.spec.repStreams.size) = false

/-- **The record's proof digests list exactly the rep streams** (S17). -/
def ProofListZk : Prop :=
  ∀ o : Run H rel, (o.sess.proofSha.size != o.spec.repStreams.size ||
    !o.spec.repStreams.all fun a => o.sess.proofSha.any fun x => x.1 == a.2.2) = false

/-- **Each proof file is the one the record's digest names** (S17). -/
def ProofDigestsZk : Prop :=
  ∀ o : Run H rel, ∀ q ∈ (o.spec.repStreams.zip o.proofs).toList,
    ((o.sess.proofSha.find? (·.1 == q.1.2.2)).map (·.2) != some (hexOf (dgOf o.spec (Claims.hOf H) q.2))) = false

/-- **The rep's stream is recorded** (S11). -/
def RepStreamZk : Prop :=
  ∀ o : Run H rel, ∀ i < o.tabs.size, ∀ r < 2,
    (o.sess.streams.find? (·.name == s!"{o.spec.tables[i]!}/rep{r}")).isSome = true

/-- **The table's extra claims evaluate** at the link's points. -/
def RepExtrasZk : Prop := ∀ o : Run H rel, ∀ i < o.tabs.size, o.tabs[i]!.extra o.sess.points = .ok (o.extras i)

/-- **The rep's masked proof file decodes** at its shape (`shapeZk`). -/
def RepDecodesZk : Prop :=
  ∀ o : Run H rel, ∀ i < o.tabs.size, ∀ r < 2,
    Zk.decode (shapeZk o.tabs[i]! (o.sch i) (o.extras i).size) (o.bytes i r) = .ok (o.zp i r)

/-- **The rep's PCS parameters are the `--zk` schedule's** (R7/F2). -/
def RepParamsZk : Prop :=
  ∀ o : Run H rel, ∀ i < o.tabs.size, ∀ r < 2,
    ((o.zp i r).proof.params != pcsParams o.tabs[i]!.merkle (o.sch i)) = false

/-- **The pads' root is 64 bytes.** -/
def PadsRoot : Prop := ∀ o : Run H rel, ∀ i < o.tabs.size, ∀ r < 2, ((o.zp i r).inner.root.size != 64) = false

/-- **The masked replay passes** from the rep's recorded rounds: zerocheck and lincheck on the masked values, the pads'
root bound (§9–§10). -/
def Replay : Prop :=
  ∀ o : Run H rel, ∀ i < o.tabs.size, ∀ r < 2,
    (Zk.replayWith o.hs o.tabs[i]! (o.lay i) (o.zp i r).proof (o.zp i r).inner.root).run (o.t₀ i r) = .ok (o.rp i r)

/-- **The ring switch replays** on the replay's claims, each valued by its check, and the extra claims (§11). -/
def RingSwitchZk : Prop :=
  ∀ o : Run H rel, ∀ i < o.tabs.size, ∀ r < 2,
    (ringSwitchWith o.hs o.tabs[i]!.m (ringClaimsZk (o.rp i r).1 (o.zp i r).proof ++ o.extras i)
      (o.zp i r).proof.ringSwitches).run (o.rp i r).2 = .ok (o.sw i r)

/-- **The masked Ligerito replays** on the batched claim, level 0's hiding messages and openings included (§12–§14). -/
def LigeritoZk : Prop :=
  ∀ o : Run H rel, ∀ i < o.tabs.size, ∀ r < 2,
    (Zk.ligeritoWith o.hs o.tabs[i]!.merkle (o.sch i) (o.sw i r).1.1 (o.zp i r).proof.cap (o.zp i r).proof.ligerito
      (basis (o.sw i r).1.2) ⟨o.tPad i, r, (o.zp i r).level0⟩).run (o.sw i r).2 = .ok (o.lg i r)

/-- **Level 0's lane phase gives its constraints** (`Zk.ligeritoConstraints`). -/
def LaneConstraints : Prop :=
  ∀ o : Run H rel, ∀ i < o.tabs.size, ∀ r < 2,
    Zk.ligeritoConstraints (o.lay i) (o.lg i r).1 (o.zp i r).proof.ligerito (o.zp i r).level0 = .ok (o.lanes i r)

/-- **The inner proof passes** on every constraint: the replay's, the ring switch's checks and level 0's. -/
def Inner : Prop :=
  ∀ o : Run H rel, ∀ i < o.tabs.size, ∀ r < 2,
    (Zk.verifyInnerWith o.hs (o.lay i) (o.zp i r).inner (o.zp i r).proof.zerocheck.finalC
      (repConsZk (o.rp i r).1 (o.zp i r).proof ++ o.lanes i r)).run (o.lg i r).2 = .ok (o.inner i r)

/-- **Every recorded round is consumed** (S15). -/
def ConsumedZk : Prop :=
  ∀ o : Run H rel, ∀ i < o.tabs.size, ∀ r < 2, ((o.inner i r).2.k != (o.stream i r).rounds.size) = false

/-- **The rep's cap is the committed `root_B`** (S16). -/
def CapCommittedZk : Prop :=
  ∀ o : Run H rel, ∀ i < o.tabs.size, ∀ r < 2, (flat (o.zp i r).proof.cap != o.sess.rootB[i]!) = false

/-- **The masked goals** at any number of tables, every check but the six proved here; `GoalsZk`'s fields. -/
structure Goals : Prop where
  session : SessionRunsZk H rel
  sends : SendCountZk H rel
  linkFirst : LinkFirstZk H rel
  finalFits : FinalFitsZk H rel
  draw : DrawMatchesZk H rel
  liveReads : LiveReadsZk H rel
  rootB : CustodyRootBZk H rel
  publicsLink : CustodyPublicsZk H rel
  points : CustodyPointsZk H rel
  count : CustodyCountZk H rel
  rounds : CustodyRoundsZk H rel
  tables : TablesZk H rel
  decodes : DecodesZk H rel
  publics : PublicsZk H rel
  schedules : SchedulesZk H rel
  proofCount : ProofCountZk H rel
  proofList : ProofListZk H rel
  proofDigests : ProofDigestsZk H rel
  repStream : RepStreamZk H rel
  repExtras : RepExtrasZk H rel
  repDecodes : RepDecodesZk H rel
  repParams : RepParamsZk H rel
  padsRoot : PadsRoot H rel
  replay : Replay H rel
  ringSwitch : RingSwitchZk H rel
  ligerito : LigeritoZk H rel
  lanes : LaneConstraints H rel
  inner : Inner H rel
  consumed : ConsumedZk H rel
  capCommitted : CapCommittedZk H rel
  canonical : RecordCanonicalZk H rel

end Goals

/-! ## The checks proved here -/

section Proved

variable {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}

/-- `ready` at the case's setup: the session draws a unit, the statement is under `2^64` bytes, every table's regions
fit, the setup against `H` is the case's, the session's coins are the keyed coin tree, and the honest record fits. -/
theorem Case.ready_all (c : Case H rel) :
    c.s.read.draw.isSome = true ∧ c.x.length < 2 ^ 64 ∧ (∀ st ∈ c.sts, RegionsFit st) ∧
      setupWith (Claims.hOf H) c.s.read = .ok (c.spec, c.sts) ∧ c.spec.coinTree.isSome = true ∧
      RecordFitsAt H c.x c.spec c.sts := by
  have h := c.ready
  unfold ZkTables.ready at h
  rw [c.setup] at h
  simp only [Bool.and_eq_true, decide_eq_true_eq] at h
  exact ⟨h.1.1.1.1.1.1, h.1.1.1.1.2, h.1.1.1.2, h.1.1.2, h.1.2, h.2⟩

/-- `ready` at the case's setup: at several tables, the session's draw is ASCII. -/
theorem Case.drawAscii (c : Case H rel) : DrawAscii c.s.read.draw c.sts := by
  have h := c.ready
  unfold ZkTables.ready at h
  rw [c.setup] at h
  simp only [Bool.and_eq_true, decide_eq_true_eq] at h
  exact h.1.1.1.1.1.2

/-- `ready`'s parts but the record's. -/
theorem Case.ready_parts (c : Case H rel) :
    c.s.read.draw.isSome = true ∧ c.x.length < 2 ^ 64 ∧ (∀ st ∈ c.sts, RegionsFit st) ∧
      setupWith (Claims.hOf H) c.s.read = .ok (c.spec, c.sts) ∧ c.spec.coinTree.isSome = true :=
  have h := c.ready_all
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1⟩

/-- **The printed honest masked record is under `2^64` bytes**, from `ready` (`RecordFitsAt`). -/
theorem Run.recordFits (o : Run H rel) : o.record.toUTF8.size < 2 ^ 64 := by
  obtain ⟨bs, h1, -⟩ := sessionPZk_sim H (fun j => (Claims.coinSeq o.r j).bytes) o.spec (tabsOf o.sts)
    (MaskRank.masksOf o.sts) (traces o.sts o.w) o.ω #[]
  have hran := o.ran
  unfold sessionS at hran
  rw [hran] at h1
  exact o.ready_all.2.2.2.2.2 o.w o.ω o.r o.record o.proofs (by simpa using congrArg Prod.snd h1)

/-- **The session's coins are the keyed coin tree**: the goal `CoinTree`, from `ready`. -/
theorem coinTree_goal : CoinTree H rel := fun o => by
  have := o.ready_parts.2.2.2.2
  cases h : o.spec.coinTree
  · rw [h] at this; cases this
  · rfl

/-- **A drawn session**: `checks` never reaches `drawLaw zk tags false none`. -/
theorem Case.drawn (c : Case H rel) : c.s.read.draw.isNone = false := by
  have := c.ready_parts.1
  cases h : c.s.read.draw
  · rw [h] at this; cases this
  · rfl

/-- **The scope check passes** (`ProvedScope.check`): the statement is in the masked scope. -/
theorem Case.scope_ok (c : Case H rel) : c.s.read.check = .ok () := by
  obtain ⟨p, hp, hsc⟩ := (Readable.scope_iff (deployZk rel) rel (readOf true) rfl
    (fun _ _ h => readOf_zk h) c.x).1 c.scope
  have hi := (setupOf_some c.setup).1
  unfold Readable.parseBytes readOf at hp
  rw [hi] at hp
  have ht := Readable.toRead_of_parse (r := c.s.read) hp
  rw [← ht]
  exact ((Readable.scope_ok_iff p).1 hsc).1

/-- **The last message's statement is the statement.** -/
theorem Run.stmt_ok (o : Run H rel) : (o.final.stmt == o.x) = true := beq_self_eq_true _

/-- **The link message reads back** as the verifier's link. -/
theorem Run.link_ok (hL : LinkFirstZk H rel) (o : Run H rel) : linkOf (o.sends.headD []) = some o.link := by
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
theorem Run.final_ok (hF : FinalFitsZk H rel) (o : Run H rel) : finalOf o.final.encode = some o.final := by
  obtain ⟨h1, h2, h3⟩ := hF o
  exact finalOf_encode o.ready_parts.2.1 h1 h2 h3

/-! ## The composition -/

/-- **A masked rep accepts** (`zk_verifyRepWith_ok`), with its cap the committed `root_B`. -/
theorem Run.rep_ok (g : Goals H rel) (o : Run H rel) {i : ℕ} (hi : i < o.tabs.size) {r : ℕ} (hr : r < 2) :
    Zk.verifyRepWith o.hs o.spec o.tabs[i]! (o.sch i) (o.tPad i) o.sess i r (o.bytes i r) = .ok o.sess.rootB[i]! := by
  have hS := g.repStream o i hi r hr
  have hfind : o.sess.streams.find? (·.name == s!"{o.spec.tables[i]!}/rep{r}") = some (o.stream i r) := by
    unfold Run.stream
    cases hf : o.sess.streams.find? (·.name == s!"{o.spec.tables[i]!}/rep{r}") with
    | none => rw [hf] at hS; cases hS
    | some st => rfl
  rw [← bytes_eq_of_bne (g.capCommitted o i hi r hr)]
  exact zk_verifyRepWith_ok o.hs o.spec o.tabs[i]! (o.sch i) (o.tPad i) o.sess i r (o.bytes i r) hfind
    (g.repExtras o i hi) (g.repDecodes o i hi r hr) (g.repParams o i hi r hr) (g.padsRoot o i hi r hr)
    (g.replay o i hi r hr) (g.ringSwitch o i hi r hr) (g.ligerito o i hi r hr) (g.lanes o i hi r hr)
    (g.inner o i hi r hr) (g.consumed o i hi r hr) (g.capCommitted o i hi r hr)

/-- **The masked verdict accepts** (`zk_verifyWith_ok`) at every table; S18 from S16, both reps' caps the committed
`root_B`. -/
theorem Run.verdict_ok (g : Goals H rel) (o : Run H rel) :
    verdictWith true (Claims.hOf H) o.spec (tabsOf o.sts) o.record o.proofs = .ok () :=
  zk_verifyWith_ok (Claims.hOf H) o.spec o.tabs o.record o.proofs (g.tables o) (coinTree_goal o) (g.decodes o)
    (g.publics o) (g.schedules o) (g.proofCount o) (g.proofList o) (g.proofDigests o)
    fun i hi => ⟨o.sess.rootB[i]!, o.rep_ok g hi (by decide), o.rep_ok g hi (by decide)⟩

/-- **B's checks pass** (`checks_ok`). -/
theorem Run.checks_ok' (g : Goals H rel) (o : Run H rel) :
    checks true o.s o.spec o.x o.link (Claims.coinSeq o.r 0) o.mids o.final = .ok () :=
  checks_ok o.stmt_ok (g.draw o) (g.liveReads o)
    (custody_ok (g.rootB o) (g.publicsLink o) (g.points o) (g.count o) (g.rounds o)) o.scope_ok o.drawn
    (g.canonical o)

/-- **The honest masked prover is accepted** at every honest masked run at a good case. -/
theorem Run.acc (g : Goals H rel) (o : Run H rel) :
    (Claims.run H (honestP o.w) (V true) o.x o.ω o.r).out = .acc := by
  have he := honestRun_eq H o.x o.w o.ω o.r o.setup o.ran
  have hk : (V true).k o.x = 2 + rounds true o.spec o.sts := by
    show (match setupOf true o.x with
      | some (_, spec, sts) => 2 + rounds true spec sts
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

end Proved

/-- **Masked completeness from the goals, at any number of tables**: Flock's masked verifier is complete for `honestP`
with error `δc H`, given each goal and the mask slots' rank bound. -/
theorem complete_of_goals (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation)
    (hMR : MaskRank.MaskRankBound) (hMF : MaskFree rel) (g : Goals H rel) : Claims.Flock.Complete H (deployZk rel) := by
  refine complete_of H rel hMR hMF fun x w hsc hr hready ω r hgood => ?_
  have hset : ∃ s spec sts, setupOf true x = some (s, spec, sts) := by
    unfold ZkTables.ready at hready
    split at hready
    · rename_i s spec sts hs
      exact ⟨s, spec, sts, hs⟩
    · cases hready
  obtain ⟨s, spec, sts, hs⟩ := hset
  let c : Case H rel := ⟨x, w, ω, r, s, spec, sts, hsc, hr, hready, hgood, hs⟩
  obtain ⟨record, proofs, bodies, hran⟩ := g.session c
  exact Run.acc g ⟨c, record, proofs, bodies, hran⟩

end FlockComplete.ZkTables
