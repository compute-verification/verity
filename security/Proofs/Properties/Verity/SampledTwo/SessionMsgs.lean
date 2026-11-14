module

public import Proofs.Properties.Verity.SampledTwo.Messages
public import Proofs.Flock.Claims.Deploy
public import Proofs.Flock.Claims.Complete.Encode

/-!
# The sessions' messages: what the two-stage form's check read of each inner block's session

`SampledTwo.formReading` reads a play round by round of `SpotFlock.V h`, and leaves the second stage's statements'
message (round `stage + 1`) and each session's messages to the sessions' reading (`SessionReading.stmts`, `msg`). This
module gives both, from the files the form's check verified each session on (`Spot.OneStageZk.stepMember`), as the
messages `SpotFlock.V h` reads: SpotFlock's tagged Flock messages, not the record's bytes the device's digests hash.

* **Template `t`'s session** (`filesAt`, `memberAt`, `recordAt`): the `t`-th of the statements' sessions, in the
  order the check reads their records, the member its statement names, found by name as
  `Spot.OneStageZk.heldStep` finds it, and the session's record on the blobs.
* **Its statement** (`stmtRaw`, `stmtAtCase`): the statement the form verifies the session at
  (`Member.statement`, then `step` at the record's sample): `flock-verify verify --zk` at `verity/flock-circuit`, the
  member's circuit and public file, the blobs' lookup tables, the record's sample (`drawBytes`), the member's partition
  object and program and its registered reads, each JSON value as `Flock.RecordText.printT` prints it. Its fields are
  `SpotFlock.HeldOf.memberRaw`'s, so `pinOk` compares it with template `t`'s reference statement like with like.
* **The statements' message** (`stmtsMsg`): one statement an entry of the evidence's `stmts`, which the run's
  second-stage statement check reads by position, the `t`-th template `t`'s session statement, in the encoding
  `SpotFlock.stmtsOf` reads back (`stmtsOf_encStmts`, `stmtAt_stmtsMsg`). A statement whose fields fit their length
  words decodes back to its fields (`stmtAtCase_some`, then `FlockDeploy.decode_encode_of_fits`).
* **A session's messages** (`sessionMsgs`, `msgAtCase`): as `FlockDeploy.act` reads them, the link message (tag `0`,
  the record's `root_B` and public words, `linkBody`), each recorded round's retained bytes after tag `0`, in
  `spec.repStreams` order, and the last message (tag `1`: the statement, the record's text and the proof files the
  session's reps name, `Final.encode`), at the session that the statement sets up. None where the record doesn't parse
  or a round retained no bytes, so no reading defaults. `SessionParts` names the parts `msgsAtCase` reads them from
  (`msgsAtCase_some`, `msgsAtCase_of_parts`).
* **The reading** (`sessionReading`): `stmts` and `msg` on the play's case; the coins and their agreement are the
  coins' piece's, parameters here. Each session's prover that sends those messages whatever the answers is
  `replayProver`, a prover that reads its messages off its tape (`Messages.proverOf`).
-/

@[expose] public section

namespace Verity.SampledTwo.SessionMsgs

open Claims
open Lean (Json)

/-- **The form's parameters and evidence** on a case; none where either doesn't decode. -/
def formOf (k : Case) : Option (VStar.Recursive.SampledTwo.Params × VStar.Recursive.SampledTwo.Evidence) :=
  match (do
      let p ← Core.canonicalJson k.params >>= VStar.Recursive.SampledTwo.paramsOfJson
      let e ← VStar.Recursive.SampledTwo.evidenceOf k.evidence
      pure (p, e) : Flock.V (VStar.Recursive.SampledTwo.Params × VStar.Recursive.SampledTwo.Evidence)) with
  | .ok pe => some pe
  | .error _ => none

/-- **Template `t`'s session's files**: the `t`-th of the statements' sessions, in the order the check reads their
records. -/
def filesAt (e : VStar.Recursive.SampledTwo.Evidence) (t : ℕ) : Option Spot.OneStageZk.RunFiles :=
  (e.statements.toList.flatMap (·.sessions.toList))[t]?

/-- **The statement of the `t`-th of the statements' sessions**: each statement once a session it holds, in the
sessions' order. -/
def stmtFilesAt (e : VStar.Recursive.SampledTwo.Evidence) (t : ℕ) : Option Spot.OneStageZk.StatementFiles :=
  (e.statements.toList.flatMap fun st => st.sessions.toList.map fun _ => st)[t]?

/-- **Template `t`'s member**: the member of the parameters that the statement of the `t`-th of the statements'
sessions names, found by name as `Spot.OneStageZk.heldStep` finds it. -/
def memberAt (p : VStar.Recursive.SampledTwo.Params) (e : VStar.Recursive.SampledTwo.Evidence) (t : ℕ) :
    Option Spot.OneStageZk.Member := do
  let st ← stmtFilesAt e t
  let name ← st.member
  p.members.find? (·.name == name)

/-- **A session's record**, its text on the blobs. -/
def recordAt (bs : Core.Blobs) (s : Spot.OneStageZk.RunFiles) : Option String :=
  (bs.get s.record "record").toOption.bind String.fromUTF8?

/-- **The record's sample, as a session's statement holds it**: its text as `Flock.RecordText.printT` prints it. -/
def drawBytes (record : String) : Option ByteArray :=
  (_root_.Flock.Witness.recordDraw record).map fun j => (_root_.Flock.RecordText.printT j).toUTF8

/-- **A member's session statement**, at the circuit and public files `circuit` and `pub`, the lookup tables
`tables`, the sample `draw` and the registered reads `registered`: `SpotFlock.HeldOf.memberRaw`'s fields. -/
def stmtRaw (m : Spot.OneStageZk.Member) (circuit pub : ByteArray) (tables : List (String × ByteArray))
    (draw registered : Option ByteArray) : FlockDeploy.Raw :=
  { statement := Spot.OneStageZk.STATEMENT, circuit, publicFile := pub, tables, zk := true, osCoins := false,
    count := 1, draw, partition := some (_root_.Flock.RecordText.printT m.own.partition).toUTF8,
    program := some m.own.program, work := none, workTable := none, stratified := none, closure := none, registered,
    publicInputs := none }

/-- **Template `t`'s session statement on the case**: its member's statement at its session's record's sample, with
the member's own circuit, public file and registered reads and the blobs' lookup tables. -/
def stmtAtCase (k : Case) (t : ℕ) : Option Msg := do
  let pe ← formOf k
  let m ← memberAt pe.1 pe.2 t
  let s ← filesAt pe.2 t
  let record ← recordAt k.blobs s
  let circuit ← (k.blobs.get m.own.circuit "circuit").toOption
  let pub ← (k.blobs.get m.own.pub "public file").toOption
  pure (stmtRaw m circuit pub (Spot.OneStageZk.tablesOf k.blobs) (drawBytes record)
    (m.own.registered.map fun j => (_root_.Flock.RecordText.printT j).toUTF8)).encode

/-- **The session statements' encoding**: the list of statements as `SpotFlock.stmtsOf` reads one back. -/
def encStmts (xs : List Msg) : Msg := FlockDeploy.encList (fun r => FlockDeploy.encBytes ⟨r.toArray⟩) xs

/-- **The session statements' message on the case**: one statement an entry of the evidence's `stmts` (the statements
the run's second-stage statement check reads, by position), the `t`-th template `t`'s session statement. -/
def stmtsMsg (k : Case) : Option Msg := do
  let pe ← formOf k
  let xs ← (List.range pe.2.stmts.size).mapM (stmtAtCase k)
  pure (encStmts xs)

/-- **A session's messages**, as `FlockDeploy.act` reads them at the session `spec` its statement `x` sets up: the link
message, each recorded round's retained bytes after tag `0`, and the last message with the record's text `record` and
the proof files `proofs`. None where the record doesn't parse or a round retained no bytes. -/
def sessionMsgs (spec : _root_.Flock.SessionSpec) (x : Msg) (record : String) (proofs : Array ByteArray) :
    Option (List Msg) := do
  let lv ← (FlockDeploy.liveOf spec record).toOption
  let mids ← (lv.streams.toList.flatMap Array.toList).mapM fun r => r.bytes.map fun b => 0 :: b.data.toList
  pure ((0 :: FlockDeploy.linkBody lv.rootB.toList lv.publics.toList) :: mids ++
    [FlockDeploy.Final.encode ⟨x, record, proofs⟩])

/-- **Template `t`'s session's messages on the case**: at its session statement's setup, its record and the proof
files its reps name (`Spot.OneStageZk.filesOf`). -/
def msgsAtCase (k : Case) (t : ℕ) : Option (List Msg) := do
  let pe ← formOf k
  let s ← filesAt pe.2 t
  let record ← recordAt k.blobs s
  let x ← stmtAtCase k t
  let su ← FlockDeploy.setupOf true x
  let proofs ← (Spot.OneStageZk.filesOf k.blobs s (.ok (su.2.1, #[], Json.null, Json.null))).toOption
  sessionMsgs su.2.1 x record proofs

/-- **Template `t`'s session on the case, part by part**, as `msgsAtCase` reads it: its statement `x`, the setup
`(s, spec, sts)` the statement gives, its record's text `record` and the proof files `proofs` its reps name. -/
def SessionParts (k : Case) (t : ℕ) (x : Msg) (s : FlockDeploy.Ins) (spec : _root_.Flock.SessionSpec)
    (sts : Array _root_.Flock.Stmt) (record : String) (proofs : Array ByteArray) : Prop :=
  ∃ p e fs, formOf k = some (p, e) ∧ filesAt e t = some fs ∧ recordAt k.blobs fs = some record ∧
    stmtAtCase k t = some x ∧ FlockDeploy.setupOf true x = some (s, spec, sts) ∧
    (Spot.OneStageZk.filesOf k.blobs fs (.ok (spec, #[], Json.null, Json.null))).toOption = some proofs

/-- **Template `t`'s session's `i`-th message on the case.** -/
def msgAtCase (k : Case) (t i : ℕ) : Option Msg := (msgsAtCase k t).bind fun ms => ms[i]?

/-- **The sessions' reading of the messages**, on the play's case `C`, with the coins' piece's coins and agreement. -/
def sessionReading (C : CaseOf) (coin : Verity.Exchange → ℕ → ℕ → Option FlockDeploy.Coin)
    (agree : Verity.Exchange → ℕ → ℕ → FlockDeploy.Coin → FlockDeploy.Coin → Prop) : SessionReading where
  stmts o := (C o).bind stmtsMsg
  msg o t i := (C o).bind fun k => msgAtCase k t i
  coin := coin
  agree := agree

/-- **Template `t`'s session's prover that replays what the check read**: at a history of `n` rounds it sends the
session's `n`-th message on the play's case, whatever the answers, and `[]` where there is none. -/
def replayProver (C : CaseOf) (o : Verity.Exchange) (t : ℕ) : Claims.Prover Unit :=
  Messages.proverOf fun _ hist => ((C o).bind fun k => msgAtCase k t hist.length).getD []

/-! ## Read back -/

theorem mapM_some {α β : Type} {f : α → Option β} :
    ∀ (l : List α) {l' : List β}, l.mapM f = some l' →
      l'.length = l.length ∧ ∀ i (hi : i < l.length) (hi' : i < l'.length), f l[i] = some l'[i]
  | [], l', h => by
    simp only [List.mapM_nil, Option.pure_def, Option.some.injEq] at h
    subst h
    exact ⟨rfl, fun i hi => absurd hi (Nat.not_lt_zero i)⟩
  | a :: l, l', h => by
    simp only [List.mapM_cons, Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def,
      Option.some.injEq] at h
    obtain ⟨b, hb, bs, hbs, rfl⟩ := h
    obtain ⟨hlen, hget⟩ := mapM_some l hbs
    refine ⟨by simp [hlen], fun i hi hi' => ?_⟩
    cases i with
    | zero => exact hb
    | succ i => exact hget i (by simpa using hi) (by simpa using hi')

/-- **The statements' encoding reads back**, where the list and each statement fit their length words. -/
theorem stmtsOf_encStmts {xs : List Msg} (hl : xs.length < 2 ^ 64) (hb : ∀ x ∈ xs, x.length < 2 ^ 64) :
    SpotFlock.stmtsOf (encStmts xs) = some xs := by
  have hmap : (xs.map fun r => (⟨r.toArray⟩ : ByteArray)).map (fun b => b.data.toList) = xs := by
    rw [List.map_map]
    exact List.map_id'' (fun r => rfl) xs
  have henc : encStmts xs =
      FlockDeploy.encList FlockDeploy.encBytes (xs.map fun r => (⟨r.toArray⟩ : ByteArray)) := by
    unfold encStmts FlockDeploy.encList
    rw [List.length_map, List.flatMap_map]
  have hrd : (FlockDeploy.rdList FlockDeploy.rdBytes).run (encStmts xs) =
      some (xs.map fun r => (⟨r.toArray⟩ : ByteArray), []) := by
    have h := FlockComplete.rdList_encList (r := FlockDeploy.rdBytes) (e := FlockDeploy.encBytes)
      (l := xs.map fun r => (⟨r.toArray⟩ : ByteArray))
      (fun b hm rest => by
        obtain ⟨r, hr, rfl⟩ := List.mem_map.1 hm
        exact FlockComplete.rdBytes_encBytes (b := ⟨r.toArray⟩)
          (by show r.toArray.size < 2 ^ 64; rw [List.size_toArray]; exact hb r hr) rest)
      (by simpa using hl) []
    rw [List.append_nil] at h
    rw [henc]
    exact h
  unfold SpotFlock.stmtsOf
  rw [hrd]
  simp only [hmap]
  exact ite_eq_left rfl

/-- **The statements' message states each template's session statement**, where they fit their length words. -/
theorem stmtsOf_stmtsMsg {k : Case} {m : Msg} (h : stmtsMsg k = some m) :
    ∃ p e xs, formOf k = some (p, e) ∧ (List.range e.stmts.size).mapM (stmtAtCase k) = some xs ∧
      m = encStmts xs := by
  unfold stmtsMsg at h
  simp only [Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def, Option.some.injEq] at h
  obtain ⟨⟨p, e⟩, hf, xs, hxs, rfl⟩ := h
  exact ⟨p, e, xs, hf, hxs, rfl⟩

/-- **Template `t`'s statement in the statements' message** is its session statement on the case, where the
statements fit their length words. -/
theorem stmtAt_stmtsMsg (h : SpotFlock.Held) {k : Case} {m : Msg} (hm : stmtsMsg k = some m) {p e xs}
    (hf : formOf k = some (p, e)) (hxs : (List.range e.stmts.size).mapM (stmtAtCase k) = some xs)
    (hl : xs.length < 2 ^ 64) (hb : ∀ x ∈ xs, x.length < 2 ^ 64) {t : ℕ} (ht : t < e.stmts.size) :
    stmtAtCase k t = some (SpotFlock.stmtAt h m t) := by
  obtain ⟨p', e', xs', hf', hxs', rfl⟩ := stmtsOf_stmtsMsg hm
  rw [hf] at hf'
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Option.some.inj hf')
  rw [hxs] at hxs'
  obtain rfl := Option.some.inj hxs'
  unfold SpotFlock.stmtAt
  rw [stmtsOf_encStmts hl hb]
  dsimp only
  obtain ⟨hlen, hget⟩ := mapM_some _ hxs
  rw [List.length_range] at hlen
  rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem (by omega : t < xs.length), Option.getD_some]
  have h2 := hget t (by rw [List.length_range]; exact ht) (by omega)
  simpa using h2

/-- **Template `t`'s session statement is its member's statement** at its session's record's sample. -/
theorem stmtAtCase_some {k : Case} {t : ℕ} {x : Msg} (h : stmtAtCase k t = some x) :
    ∃ p e m s record circuit pub, formOf k = some (p, e) ∧ memberAt p e t = some m ∧ filesAt e t = some s ∧
      recordAt k.blobs s = some record ∧ (k.blobs.get m.own.circuit "circuit").toOption = some circuit ∧
      (k.blobs.get m.own.pub "public file").toOption = some pub ∧
      x = (stmtRaw m circuit pub (Spot.OneStageZk.tablesOf k.blobs) (drawBytes record)
        (m.own.registered.map fun j => (_root_.Flock.RecordText.printT j).toUTF8)).encode := by
  unfold stmtAtCase at h
  simp only [Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def, Option.some.injEq] at h
  obtain ⟨⟨p, e⟩, hf, m, hm, s, hs, record, hr, circuit, hc, pub, hp, rfl⟩ := h
  exact ⟨p, e, m, s, record, circuit, pub, hf, hm, hs, hr, hc, hp, rfl⟩

theorem getLast?_cons_concat (a b : Msg) : ∀ l : List Msg, ((a :: l) ++ [b]).getLast? = some b
  | [] => rfl
  | c :: l => getLast?_cons_concat c b l

/-- **A session's messages are its link message, its rounds' and its last message.** -/
theorem sessionMsgs_some {spec : _root_.Flock.SessionSpec} {x : Msg} {record : String} {proofs : Array ByteArray}
    {ms : List Msg} (h : sessionMsgs spec x record proofs = some ms) :
    ∃ lv mids, (FlockDeploy.liveOf spec record).toOption = some lv ∧
      (lv.streams.toList.flatMap Array.toList).mapM (fun r => r.bytes.map fun b => 0 :: b.data.toList) = some mids ∧
      ms = (0 :: FlockDeploy.linkBody lv.rootB.toList lv.publics.toList) :: mids ++
        [FlockDeploy.Final.encode ⟨x, record, proofs⟩] := by
  unfold sessionMsgs at h
  simp only [Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def, Option.some.injEq] at h
  obtain ⟨lv, hlv, mids, hmids, rfl⟩ := h
  exact ⟨lv, mids, hlv, hmids, rfl⟩

/-- **A session's first message reads back as its link message**, where the record's roots and public words fit their
length words. -/
theorem linkOf_sessionMsgs {spec : _root_.Flock.SessionSpec} {x : Msg} {record : String} {proofs : Array ByteArray}
    {ms : List Msg} (h : sessionMsgs spec x record proofs = some ms) {lv : FlockDeploy.Live}
    (hlv : (FlockDeploy.liveOf spec record).toOption = some lv) (hr : lv.rootB.size < 2 ^ 64)
    (hp : lv.publics.size < 2 ^ 64) (hrb : ∀ b ∈ lv.rootB.toList, b.size < 2 ^ 64)
    (hpb : ∀ b ∈ lv.publics.toList, b.size < 2 ^ 64) :
    ∃ a ∈ ms.head?, FlockDeploy.linkOf a = some (lv.rootB.toList, lv.publics.toList) := by
  obtain ⟨lv', mids, hlv', -, rfl⟩ := sessionMsgs_some h
  rw [hlv] at hlv'
  obtain rfl := Option.some.inj hlv'
  exact ⟨_, rfl, FlockComplete.linkOf_linkBody (by simpa using hr) (by simpa using hp) hrb hpb⟩

/-- **A session's last message reads back as the statement, the record's text and the proof files**, where they fit
their length words. -/
theorem finalOf_sessionMsgs {spec : _root_.Flock.SessionSpec} {x : Msg} {record : String} {proofs : Array ByteArray}
    {ms : List Msg} (h : sessionMsgs spec x record proofs = some ms) (hs : x.length < 2 ^ 64)
    (hrec : record.toUTF8.size < 2 ^ 64) (hp : proofs.size < 2 ^ 64) (hb : ∀ b ∈ proofs, b.size < 2 ^ 64) :
    ∃ a ∈ ms.getLast?, FlockDeploy.finalOf a = some ⟨x, record, proofs⟩ := by
  obtain ⟨lv, mids, -, -, rfl⟩ := sessionMsgs_some h
  exact ⟨FlockDeploy.Final.encode ⟨x, record, proofs⟩, getLast?_cons_concat _ _ _,
    FlockComplete.finalOf_encode hs hrec hp hb⟩

/-! ## The case's session, part by part -/

/-- **Template `t`'s messages on the case are its session's parts' messages.** -/
theorem msgsAtCase_some {k : Case} {t : ℕ} {ms : List Msg} (h : msgsAtCase k t = some ms) :
    ∃ x s spec sts record proofs, SessionParts k t x s spec sts record proofs ∧
      sessionMsgs spec x record proofs = some ms := by
  unfold msgsAtCase at h
  simp only [Option.bind_eq_bind, Option.bind_eq_some_iff] at h
  obtain ⟨⟨p, e⟩, hf, fs, hs, record, hr, x, hx, ⟨s, spec, sts⟩, hsu, proofs, hp, hm⟩ := h
  exact ⟨x, s, spec, sts, record, proofs, ⟨p, e, fs, hf, hs, hr, hx, hsu, hp⟩, hm⟩

/-- **The messages of a session's parts are template `t`'s messages on the case.** -/
theorem msgsAtCase_of_parts {k : Case} {t : ℕ} {x : Msg} {s : FlockDeploy.Ins} {spec : _root_.Flock.SessionSpec}
    {sts : Array _root_.Flock.Stmt} {record : String} {proofs : Array ByteArray} {ms : List Msg}
    (h : SessionParts k t x s spec sts record proofs) (hm : sessionMsgs spec x record proofs = some ms) :
    msgsAtCase k t = some ms := by
  obtain ⟨p, e, fs, hf, hs, hr, hx, hsu, hp⟩ := h
  unfold msgsAtCase
  simp only [Option.bind_eq_bind, Option.bind_eq_some_iff]
  exact ⟨(p, e), hf, fs, hs, record, hr, x, hx, (s, spec, sts), hsu, proofs, hp, hm⟩

/-- **The form's evidence of a case is the one its evidence's bytes decode to.** -/
theorem formOf_evidence {k : Case} {p : VStar.Recursive.SampledTwo.Params}
    {e : VStar.Recursive.SampledTwo.Evidence} (h : formOf k = some (p, e)) :
    VStar.Recursive.SampledTwo.evidenceOf k.evidence = .ok e := by
  unfold formOf at h
  cases hp : (Core.canonicalJson k.params >>= VStar.Recursive.SampledTwo.paramsOfJson) with
  | error w =>
    rw [hp] at h
    cases h
  | ok p' =>
    cases he : VStar.Recursive.SampledTwo.evidenceOf k.evidence with
    | error w =>
      rw [hp, he] at h
      cases h
    | ok e' =>
      rw [hp, he] at h
      cases h
      rfl

/-- **Each template below the statements' count has a session statement**, where the case has a statements'
message. -/
theorem stmtAtCase_of_stmtsMsg {k : Case} {m : Msg} (hm : stmtsMsg k = some m)
    {p : VStar.Recursive.SampledTwo.Params} {e : VStar.Recursive.SampledTwo.Evidence}
    (hf : formOf k = some (p, e)) {t : ℕ} (ht : t < e.stmts.size) : ∃ x, stmtAtCase k t = some x := by
  obtain ⟨p', e', xs, hf', hxs, -⟩ := stmtsOf_stmtsMsg hm
  rw [hf] at hf'
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Option.some.inj hf')
  obtain ⟨hlen, hget⟩ := mapM_some _ hxs
  rw [List.length_range] at hlen
  have h := hget t (by rw [List.length_range]; exact ht) (by omega)
  rw [List.getElem_range] at h
  exact ⟨_, h⟩

/-- **Where template `t` has a session statement, the statements have a `t`-th session.** -/
theorem filesAt_of_stmt {k : Case} {t : ℕ} {x : Msg} (h : stmtAtCase k t = some x)
    {p : VStar.Recursive.SampledTwo.Params} {e : VStar.Recursive.SampledTwo.Evidence}
    (hf : formOf k = some (p, e)) : ∃ fs, filesAt e t = some fs := by
  obtain ⟨p', e', m, fs, record, circuit, pub, hf', -, hs, -⟩ := stmtAtCase_some h
  rw [hf] at hf'
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Option.some.inj hf')
  exact ⟨fs, hs⟩

end Verity.SampledTwo.SessionMsgs
