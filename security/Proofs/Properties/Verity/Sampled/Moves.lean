module

public import Proofs.Properties.Verity.SampledRead
public import Proofs.Properties.Verity.Sampled.Messages
public import Proofs.Flock.Claims.Bridge.Custody
public import Proofs.Flock.Claims.Bridge.CustodyZkP
public import Proofs.Flock.Soundness.Discharge.Exec.Retained
public import Proofs.Flock.Soundness.ExecStrataOf
import all Init.Data.Repr

/-!
# Sampled proofs' coupling: the live audit's typed moves, from what v2 reads

The logged coupling reads a play of the period's game as a play of Flock's live audit (`Live.game`): the prover sends
its registration, the tables' roots `rootB`, a submission (`SubmitJ`) for each round of a rep stream, and the proof
files. In the period's game those moves are only in the evidence, and the messages piece (`Reading`) fixes each one at
its cut. This module reads them back, from what v2 reads of the evidence and from the messages the evidence reveals,
and holds what it reads to v2's check and to the writer.

* **What v2 reads** (`sessionRecord`, `sessionProofs`, `sentRecord`, `sentProofs`, `registrationIn`, `proofsIn`): the
  session's record and its proof files on the blobs, as `Spot.OneStageZk.run` reads them, and the
  registration v2's case decodes. Where v2's check accepts, they are the record and the files `Flock.Zk.verify`
  accepted, at the session v2 set up (`checkBlobsAt_verified`, `sent_verified`, `V2Verified`).
* **The messages** (`linkMsg`, `roundMsg`, `linkIn`, `roundIn`): the link's message, `root_F` then the tables' `root_B`
  in lexicographic table order (S14's), and a rep round's retained bytes (S12's `msg`), each read off the record.
  Each parses the record with `Flock.JsonParse.parse`, as `Record.decode` does, and refuses (`none`) wherever a parse
  fails, so no reading defaults.
* **The typed moves** (`rootBOf`, `repOf`, `submitOf`, `repsOf`): `rootB` from the link's message, past `Σ`'s bytes;
  a round's submission from its stream's name (`<table>/rep<r>`) and its message; and the rounds a list of answered
  submissions leaves on each rep stream, as `Tables.liveRoundsJ` appends them.
* **Against the record v2 decodes** (`rootBOf_linkMsg`, `roundMsg_decode`, `repsOf_stream`): at one table, the roots
  read from the link's message are the decoded session's `root_B`; under retained rounds, a round's message is its
  decoded bytes; and answered submissions whose rounds are a decoded stream's, in order, leave that stream's rounds.
* **Against the writer** (`WritesView`, `writesView_wrZP`, `eq_viewZP`, `wr_accepts`): an outcome whose roots, link
  points, rounds and proof files are an accepted transcript's decoded ones is that transcript's view, and where the
  session's writer keeps an accepted transcript's view (`WritesView`, which `wrZP` does), what it writes for that
  outcome is accepted.

What the link's message binds is the record's: `Spot.Challenger.sessionTreeReason` opens the link points after
the SHA-512 of `root_F` and every `link.root_b` entry, while `Record.decode` reads only the tables' entries. The two are
one message where `link.root_b` names exactly the session's tables, which v2 doesn't check today.
-/

@[expose] public section

namespace Verity.Sampled

open Lean (Json)
open Flock (SessionSpec RunRecord StreamRec Round)
open FlockSoundness.Discharge.Tables (SubmitJ addRoundJ addRoundJ_self addRoundJ_ne LiveOutJ)
open FlockSoundness.Refine (roundOf ebind_ok check_ok_iff bytes_eq_of_beq ptsOf)
open Claims.Bridge.Custody (decode_ok RootField DecStream RoundFields forall₂_find? forall₂_getElem)
open Claims.Bridge.CustodyZkP (sessOf sessOf_eq repRounds viewA viewZP wrZP acceptsZKP_wrZP_planZHJP)
open FlockSoundness.Discharge.ZkHidden (AcceptsZKP)

local notation "𝔽" => FlockLevel3.GF128

/-! ## Lists, arrays and options -/

theorem forall₂_swap {α β : Type} {R : α → β → Prop} :
    ∀ {l : List α} {l' : List β}, List.Forall₂ R l l' → List.Forall₂ (fun b a => R a b) l' l
  | _, _, .nil => .nil
  | _, _, .cons h t => .cons h (forall₂_swap t)

theorem find?_unique {α : Type} {p : α → Bool} {a : α} (hp : ∀ x, p x = true ↔ x = a) :
    ∀ {l : List α}, a ∈ l → l.find? p = some a
  | [], ha => by cases ha
  | y :: l, ha => by
    by_cases hy : p y = true
    · rw [List.find?_cons_of_pos hy, (hp y).1 hy]
    · rw [List.find?_cons_of_neg hy]
      rcases List.mem_cons.1 ha with rfl | ha
      · exact absurd ((hp _).2 rfl) hy
      · exact find?_unique hp ha

theorem qsort_single {α : Type} (lt : α → α → Bool) (a : α) : #[a].qsort lt = #[a] :=
  Array.toList_inj.1 (List.perm_singleton.1 (FlockSoundness.Audit.StrataExec.qsort_perm lt #[a]))

theorem mapM_single {α β ε : Type} {f : α → Except ε β} {a : α} {b : β} (h : f a = .ok b) :
    #[a].mapM f = .ok #[b] := by
  rw [Array.mapM_eq_mapM_toList]
  simp only [List.mapM_cons, List.mapM_nil, h]
  rfl

theorem foldlM_single {α β ε : Type} {f : β → α → Except ε β} {b r : β} {a : α}
    (h : #[a].foldlM f b = .ok r) : f b a = .ok r := by
  rw [← Array.foldlM_toList] at h
  simpa only [List.foldlM_cons, List.foldlM_nil, bind_pure] using h

theorem ok_toOption {ε α : Type} (a : α) : (Except.ok a : Except ε α).toOption = some a := rfl

theorem eq_single {α : Type} {xs : Array α} (h : xs.size = 1) : xs = #[xs[0]] := by
  apply Array.ext
  · simp [h]
  · intro i h1 _
    have : i = 0 := by omega
    subst this
    rfl

/-! ## What v2 reads -/

/-- **The session's record as v2 reads it** on the blobs `bs`: the blob the evidence's first session names for it, as
UTF-8 (`Spot.OneStageZk.step`). -/
def sessionRecord (e : _root_.Spot.OneStageZk.Evidence) (bs : _root_.Core.Blobs) : Option String :=
  e.sessions[0]?.bind fun s => (bs.get s.record s!"session {s.name}'s record").toOption.bind String.fromUTF8?

/-- **The session's proof files as v2 reads them** on the blobs `bs`: at the session v2 sets up at its record's sample
(`Spot.OneStageZk.setupAt`) on the parameters' statement, circuit and public files and registered reads, the
blob the session names for each rep stream (`Spot.OneStageZk.filesOf`), as `Spot.OneStageZk.run`
hands them to `Flock.Zk.verify`. -/
def sessionProofs (p : _root_.Spot.OneStageZk.Params) (e : _root_.Spot.OneStageZk.Evidence)
    (bs : _root_.Core.Blobs) : Option (Array ByteArray) := do
  let s ← e.sessions[0]?
  let tags0 ← Flock.Tags.find? _root_.Spot.OneStageZk.STATEMENT
  let circuitB ← (bs.get p.own.circuit "circuit").toOption
  let pubB ← (bs.get p.own.pub "public file").toOption
  let rd ← (_root_.Spot.OneStageZk.readsOf p.own.registered).toOption
  let record ← sessionRecord e bs
  (_root_.Spot.OneStageZk.filesOf bs s
    (_root_.Spot.OneStageZk.setupAt (_root_.Spot.OneStageZk.inputsOf tags0 p.own circuitB pubB bs) rd
      (Flock.Witness.recordDraw record))).toOption

/-- The session's record v2 reads in what the prover sent, on the blobs sent. -/
def sentRecord (s : Verity.Sent) : Option String :=
  (caseOf s).toOption.bind fun x => sessionRecord x.2.2 (_root_.Core.Blobs.ofArray s.blobs)

/-- The session's proof files v2 reads in what the prover sent, on the blobs sent. -/
def sentProofs (s : Verity.Sent) : Option (Array ByteArray) :=
  (caseOf s).toOption.bind fun x => sessionProofs x.1 x.2.2 (_root_.Core.Blobs.ofArray s.blobs)

/-- **The registration**, as v2's case decodes it in what the prover sent. -/
def registrationIn (o : Verity.Exchange) : Option _root_.Spot.OneStage.Commitment :=
  (caseOf o.sent).toOption.map (·.2.1)

/-- **The proof files** of the play that ended with `o`, as v2 reads them. -/
def proofsIn (o : Verity.Exchange) : Option (Array ByteArray) := sentProofs o.sent

/-! ## The messages -/

/-- Table `t`'s `root_B` in the record's link `l` (S5's reading). -/
def rootBAt (l : Json) (t : String) : Flock.V ByteArray :=
  Flock.Record.fld l "root_b" >>= fun v => Flock.Record.str v t >>= Flock.unhex

/-- **The link's message** in the record `record` of a session of the tables `tables`: `root_F`, then each table's
`root_B` in lexicographic table order (S14's exchange before its points); none where the record does not parse or
lacks one. -/
def linkMsg (record : String) (tables : Array String) : Option Claims.Msg :=
  (Flock.JsonParse.parse record).toOption.bind fun j =>
    (Flock.Record.fld j "link").toOption.bind fun l =>
      (Flock.Record.str l "root_f" >>= Flock.unhex).toOption.bind fun rootF =>
        ((tables.qsort (· < ·)).mapM (rootBAt l)).toOption.map fun rbs => (rbs.foldl (· ++ ·) rootF).data.toList

/-- **Round `k`'s message on the stream `name`** in the record `record`: the round's retained bytes (S12's `msg`); none
where the record does not parse, has no such round, or the round retains no bytes. -/
def roundMsg (record name : String) (k : ℕ) : Option Claims.Msg :=
  (Flock.JsonParse.parse record).toOption.bind fun j =>
    (Flock.Record.arr j "streams").toOption.bind fun ss =>
      (ss.find? fun x => (Flock.Record.str x "stream").toOption == some name).bind fun x =>
        (Flock.Record.arr x "rounds").toOption.bind fun rs =>
          rs[k]?.bind fun r =>
            if Flock.Record.isNull r "msg" then none
            else ((Flock.Record.str r "msg" >>= Flock.unhex).toOption).map (·.data.toList)

/-- **The link's message** of the play that ended with `o`, in the record v2 reads, at the session's tables. -/
def linkIn (tables : Array String) (o : Verity.Exchange) : Option Claims.Msg :=
  (sentRecord o.sent).bind (linkMsg · tables)

/-- **Round `k`'s message on the stream `name`** of the play that ended with `o`, in the record v2 reads. -/
def roundIn (name : String) (k : ℕ) (o : Verity.Exchange) : Option Claims.Msg :=
  (sentRecord o.sent).bind (roundMsg · name k)

/-! ## The typed moves -/

/-- **The tables' roots in the link's message `m`**, past `Σ`'s bytes `σ`: none where `m` does not start with `σ`. -/
def rootBOf (σ : ByteArray) (m : Claims.Msg) : Option (Array ByteArray) :=
  if m.take σ.size = σ.data.toList then some #[⟨(m.drop σ.size).toArray⟩] else none

/-- Table `t`'s rep `r` stream, `<t>/rep<r>` (`SessionSpec.repStreams`). -/
def repStreamName (t : String) (r : ℕ) : String := s!"{t}/rep{r}"

/-- **The rep stream the name `name` is**, among the tables `tables`. -/
def repOf {J : ℕ} (tables : Fin J → String) (name : String) : Option (Fin J × Fin 2) :=
  ((List.finRange J).flatMap fun j => [(j, 0), (j, 1)]).find? fun x => repStreamName (tables x.1) x.2.val == name

/-- **The submission of a round** on the stream `name` with `n` coins and the message `m`. -/
def submitOf {J : ℕ} (tables : Fin J → String) (name : String) (n : ℕ) (m : Claims.Msg) : Option (SubmitJ J) :=
  (repOf tables name).map fun x => ⟨x.1, x.2, ⟨m.toArray⟩, n⟩

/-- A submission with the coins the live session answered it with. -/
abbrev Answered (J : ℕ) := (s : SubmitJ J) × (Fin s.n → 𝔽)

/-- **The rounds the answered submissions `as` leave on each rep stream**, as `Tables.liveRoundsJ` appends them. -/
def repsOf (dg : ByteArray → ByteArray) {J : ℕ} (as : List (Answered J)) : Fin J → Fin 2 → List Round :=
  as.foldl (fun rs a => addRoundJ rs a.1.tab a.1.rep (roundOf dg a.1.bytes a.1.n a.2)) fun _ _ => []

theorem take_drop (σ rb : ByteArray) :
    (σ ++ rb).data.toList.take σ.size = σ.data.toList ∧
      (⟨((σ ++ rb).data.toList.drop σ.size).toArray⟩ : ByteArray) = rb := by
  have hl : σ.data.toList.length = σ.size := Array.length_toList
  rw [ByteArray.data_append, Array.toList_append, List.take_left' hl, List.drop_left' hl, Array.toArray_toList]
  exact ⟨rfl, rfl⟩

/-- `rootBOf` reads `rb` back from `σ ‖ rb`. -/
theorem rootBOf_append (σ rb : ByteArray) : rootBOf σ (σ ++ rb).data.toList = some #[rb] := by
  obtain ⟨h1, h2⟩ := take_drop σ rb
  rw [rootBOf, ite_eq_left h1, h2]

theorem repStreamName_inj {t t' : String} {r r' : Fin 2} (h : repStreamName t r.val = repStreamName t' r'.val) :
    t = t' ∧ r = r' := by
  have h' := congrArg String.toList h
  simp only [repStreamName, String.toList_append, List.append_assoc] at h'
  have key : ∀ a b : Fin 2, ((toString "/rep").toList ++ (toString a.val).toList).length =
      ((toString "/rep").toList ++ (toString b.val).toList).length := by decide
  obtain ⟨ht, hr⟩ := List.append_inj' h' (key r r')
  have hr' := List.append_cancel_left hr
  have keyr : ∀ a b : Fin 2, (toString a.val).toList = (toString b.val).toList → a = b := by decide
  exact ⟨String.toList_inj.1 ht, keyr r r' hr'⟩

/-- **A rep stream's name reads back as that stream** among tables with distinct names. -/
theorem repOf_name {J : ℕ} {tables : Fin J → String} (hinj : Function.Injective tables) (j : Fin J) (r : Fin 2) :
    repOf tables (repStreamName (tables j) r.val) = some (j, r) := by
  refine find?_unique (fun x => ?_) (List.mem_flatMap.2 ⟨j, List.mem_finRange j, ?_⟩)
  · rw [beq_iff_eq]
    constructor
    · intro hx
      obtain ⟨ht, hr⟩ := repStreamName_inj hx
      exact Prod.ext (hinj ht) hr
    · rintro rfl
      rfl
  · fin_cases r <;> simp

/-- `repOf_name`'s submission. -/
theorem submitOf_name {J : ℕ} {tables : Fin J → String} (hinj : Function.Injective tables) (j : Fin J) (r : Fin 2)
    (n : ℕ) (m : Claims.Msg) : submitOf tables (repStreamName (tables j) r.val) n m = some ⟨j, r, ⟨m.toArray⟩, n⟩ := by
  rw [submitOf, repOf_name hinj]
  rfl

/-- At one table every name is distinct. -/
theorem injective_one (tables : Fin 1 → String) : Function.Injective tables := fun a b _ => Subsingleton.elim a b

theorem foldl_addRound (dg : ByteArray → ByteArray) {J : ℕ} :
    ∀ (as : List (Answered J)) (rs : Fin J → Fin 2 → List Round) (j : Fin J) (r : Fin 2),
      (as.foldl (fun rs a => addRoundJ rs a.1.tab a.1.rep (roundOf dg a.1.bytes a.1.n a.2)) rs) j r =
        rs j r ++ ((as.filter fun a => decide (a.1.tab = j ∧ a.1.rep = r)).map fun a =>
          roundOf dg a.1.bytes a.1.n a.2)
  | [], rs, j, r => by simp
  | a :: as, rs, j, r => by
    rw [List.foldl_cons, foldl_addRound dg as, List.filter_cons]
    by_cases h : a.1.tab = j ∧ a.1.rep = r
    · obtain ⟨h1, h2⟩ := h
      subst h1 h2
      simp [addRoundJ_self]
    · rw [addRoundJ_ne _ _ fun h' => h ⟨h'.1.symm, h'.2.symm⟩]
      simp [h]

/-- **The rounds on stream `(j, r)`** are the answered submissions to it, in order. -/
theorem repsOf_eq (dg : ByteArray → ByteArray) {J : ℕ} (as : List (Answered J)) (j : Fin J) (r : Fin 2) :
    repsOf dg as j r =
      ((as.filter fun a => decide (a.1.tab = j ∧ a.1.rep = r)).map fun a => roundOf dg a.1.bytes a.1.n a.2) := by
  rw [repsOf, foldl_addRound]
  rfl

/-! ## What v2's check accepted -/

theorem push_ok {I : Flock.Witness.Inputs} {rd : _root_.Spot.OneStageZk.Reads} {programSha : Option String}
    {want : Json} {r r' : _root_.Spot.OneStageZk.Run} {s : _root_.Spot.OneStageZk.RunFiles}
    {record : String} {st : _root_.Spot.OneStageZk.SetUp} {coinKey : Option String} {verdict : Flock.V Unit}
    (h : _root_.Spot.OneStageZk.push I rd programSha want r s record st coinKey verdict = .ok r')
    (hall : r'.all = true) : verdict = .ok () := by
  unfold _root_.Spot.OneStageZk.push at h
  cases verdict with
  | error why =>
    dsimp only at h
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    cases hall
  | ok u => rfl

theorem verdictOf_ok {I : Flock.Witness.Inputs} {scope ksFiles population : Flock.V Unit} {coinKeys : Array String}
    {record : String} {draw : Option Json} {st : _root_.Spot.OneStageZk.SetUp}
    {files : Flock.V (Array ByteArray)} {coinKey : Option String}
    (h : _root_.Spot.OneStageZk.verdictOf I scope ksFiles population coinKeys record draw st files coinKey
      = .ok ()) :
    ∃ proofs spec tabs rest, files = .ok proofs ∧ st = .ok (spec, tabs, rest) ∧
      Flock.Zk.verify spec tabs record proofs = .ok () := by
  unfold _root_.Spot.OneStageZk.verdictOf at h
  obtain ⟨proofs, hf, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨⟨spec, tabs, rest⟩, hst, h⟩ := ebind_ok h
  dsimp only at h
  obtain ⟨_, hv, -⟩ := ebind_ok h
  exact ⟨proofs, spec, tabs, rest, hf, hst, hv⟩

/-- **One session's `step` after a run that accepted so far, where every session accepted**: its record is a blob,
UTF-8, set up at its own sample, its proof files the blobs it names, and `Flock.Zk.verify` accepts them. -/
theorem step_verified {I : Flock.Witness.Inputs} {rd : _root_.Spot.OneStageZk.Reads}
    {bs : _root_.Core.Blobs} {scope ksFiles population : Flock.V Unit} {programSha : Option String} {want : Json}
    {s : _root_.Spot.OneStageZk.RunFiles} {r : _root_.Spot.OneStageZk.Run}
    (h : _root_.Spot.OneStageZk.step I rd bs scope ksFiles population programSha want ⟨#[], #[], #[], true⟩ s
      = .ok r)
    (hall : r.all = true) :
    ∃ recB record spec tabs rest proofs, bs.get s.record s!"session {s.name}'s record" = .ok recB ∧
      String.fromUTF8? recB = some record ∧
      _root_.Spot.OneStageZk.setupAt I rd (Flock.Witness.recordDraw record) = .ok (spec, tabs, rest) ∧
      _root_.Spot.OneStageZk.filesOf bs s (.ok (spec, tabs, rest)) = .ok proofs ∧
      Flock.Zk.verify spec tabs record proofs = .ok () := by
  unfold _root_.Spot.OneStageZk.step at h
  obtain ⟨recB, hb, h⟩ := ebind_ok h
  dsimp only at h
  split at h
  · rename_i record hrec
    obtain ⟨_, -, h⟩ := ebind_ok h
    obtain ⟨proofs, spec, tabs, rest, hf, hst, hv⟩ := verdictOf_ok (push_ok h hall)
    rw [hst] at hf
    exact ⟨recB, record, spec, tabs, rest, proofs, hb, hrec, hst, hf, hv⟩
  · cases h

/-- **v2's run on one session, where it accepted**: the statement's tags, the circuit and public files and the
registered reads, the session's `step` from the empty run at the population check of the public file
(`FilePart.check`), which ends with the run, and, at the units `want` v1 proved where the run accepted every session,
the run's proved units held to them (`provedHeld`). -/
theorem run_verified {p : _root_.Spot.OneStageZk.Own} {s : _root_.Spot.OneStageZk.RunFiles}
    {pu : Option (Array ℕ)} {bs : _root_.Core.Blobs} {r : _root_.Spot.OneStageZk.Run}
    (h : _root_.Spot.OneStageZk.run p #[s] pu bs = .ok r) :
    ∃ tags0 circuitB pubB rd, Flock.Tags.find? _root_.Spot.OneStageZk.STATEMENT = some tags0 ∧
      bs.get p.circuit "circuit" = .ok circuitB ∧ bs.get p.pub "public file" = .ok pubB ∧
      _root_.Spot.OneStageZk.readsOf p.registered = .ok rd ∧
      ∃ scope ksFiles programSha want,
        _root_.Spot.OneStageZk.step (_root_.Spot.OneStageZk.inputsOf tags0 p circuitB pubB bs) rd bs
          scope ksFiles
          (_root_.Spot.FilePart.check p.program (_root_.Spot.getD p.partition "query") pubB)
          programSha want ⟨#[], #[], #[], true⟩ s = .ok r ∧
        ∀ want', pu = some want' → r.all = true →
          _root_.Spot.OneStageZk.provedHeld r.proved want' = .ok () := by
  unfold _root_.Spot.OneStageZk.run at h
  dsimp only at h
  obtain ⟨_, -, h⟩ := ebind_ok h
  split at h
  · rename_i tags0 htags
    obtain ⟨_, -, h⟩ := ebind_ok h
    obtain ⟨_, -, h⟩ := ebind_ok h
    obtain ⟨circuitB, hc, h⟩ := ebind_ok h
    obtain ⟨pubB, hp, h⟩ := ebind_ok h
    obtain ⟨_, -, h⟩ := ebind_ok h
    obtain ⟨rd, hrd, h⟩ := ebind_ok h
    obtain ⟨r', hr', h⟩ := ebind_ok h
    have e : r' = r ∧ ∀ want', pu = some want' → r'.all = true →
        _root_.Spot.OneStageZk.provedHeld r'.proved want' = .ok () := by
      cases pu with
      | none =>
        refine ⟨?_, fun _ hw => by cases hw⟩
        simpa only [pure, Except.pure, Except.ok.injEq] using h
      | some want =>
        refine ⟨?_, fun want' hw hall => ?_⟩
        · repeat' (first | (obtain ⟨_, _, h⟩ := ebind_ok h) | (split at h))
          all_goals simpa only [pure, Except.pure, Except.ok.injEq] using h
        · cases hw
          simp only [hall, ↓reduceIte] at h
          obtain ⟨⟨⟩, hu, -⟩ := ebind_ok h
          exact hu
    obtain ⟨rfl, hw⟩ := e
    exact ⟨tags0, circuitB, pubB, rd, htags, hc, hp, hrd, _, _, _, _, foldlM_single hr', hw⟩
  · cases h

/-- **v2's check at the coins `ξ` accepts only where its run on the evidence's one session accepted it**, on the public
file v2 held to the registration (`publicHeld`). -/
theorem checkBlobsAt_run {p : _root_.Spot.OneStageZk.Params} {c : _root_.Spot.OneStage.Commitment}
    {ξ : ByteArray} {e : _root_.Spot.OneStageZk.Evidence} {bs : _root_.Core.Blobs}
    {pub : _root_.Spot.OneStage.Public} {prof : _root_.Core.Integrity}
    (h : _root_.Spot.OneStageZk.checkBlobsAt p c ξ e bs = .accept pub prof) :
    ∃ r pubB, e.sessions.size = 1 ∧ bs.get p.own.pub "public file" = .ok pubB ∧
      _root_.Spot.OneStageZk.publicHeld c.record pubB = .ok () ∧
      _root_.Spot.OneStageZk.run p.own e.sessions (some e.base.provedUnits) bs = .ok r ∧ r.all = true := by
  unfold _root_.Spot.OneStageZk.checkBlobsAt at h
  dsimp only at h
  split at h
  · cases h
  · rename_i r hr
    obtain ⟨_, h1, hr⟩ := ebind_ok hr
    obtain ⟨_, -, hr⟩ := ebind_ok hr
    obtain ⟨⟨⟩, -, hr⟩ := ebind_ok hr
    obtain ⟨_, -, hr⟩ := ebind_ok hr
    obtain ⟨_, -, hr⟩ := ebind_ok hr
    obtain ⟨pubB, hpub, hr⟩ := ebind_ok hr
    obtain ⟨⟨⟩, hph, hr⟩ := ebind_ok hr
    refine ⟨r, pubB, ?_, hpub, hph, hr, ?_⟩
    · unfold _root_.Spot.OneStageZk.oneSession at h1
      split at h1
      · rename_i hs
        exact beq_iff_eq.1 hs
      · cases h1
    · revert h
      cases _root_.Spot.OneStage.checkAt p.base c (some ξ) { e.base with verdicts := r.out } with
      | accept _ _ =>
        intro h
        dsimp only at h
        cases hall : r.all
        · rw [hall] at h
          cases h
        · rfl
      | reject _ => intro h; cases h
      | undecided _ => intro h; cases h

/-- **What v2's check verified of the evidence `e` on the blobs `bs`**: its one session `rs`, the statement's tags, the
parameters' circuit and public files and registered reads, the session's record `record` (`sessionRecord`), the session
`(spec, tabs)` v2 sets up at its sample, and its proof files `proofs`, which `Flock.Zk.verify` accepts. -/
def V2Verified (p : _root_.Spot.OneStageZk.Params) (e : _root_.Spot.OneStageZk.Evidence)
    (bs : _root_.Core.Blobs) (record : String) (spec : SessionSpec) (tabs : Array Flock.Setup) (rest : Json × Json)
    (proofs : Array ByteArray) : Prop :=
  ∃ rs tags0 circuitB pubB rd, e.sessions = #[rs] ∧
    Flock.Tags.find? _root_.Spot.OneStageZk.STATEMENT = some tags0 ∧
    bs.get p.own.circuit "circuit" = .ok circuitB ∧ bs.get p.own.pub "public file" = .ok pubB ∧
    _root_.Spot.OneStageZk.readsOf p.own.registered = .ok rd ∧ sessionRecord e bs = some record ∧
    _root_.Spot.OneStageZk.setupAt (_root_.Spot.OneStageZk.inputsOf tags0 p.own circuitB pubB bs) rd
      (Flock.Witness.recordDraw record) = .ok (spec, tabs, rest) ∧
    _root_.Spot.OneStageZk.filesOf bs rs (.ok (spec, tabs, rest)) = .ok proofs ∧
    Flock.Zk.verify spec tabs record proofs = .ok ()

theorem sessionRecord_eq {e : _root_.Spot.OneStageZk.Evidence} {bs : _root_.Core.Blobs}
    {rs : _root_.Spot.OneStageZk.RunFiles} {recB : ByteArray} {record : String} (hs : e.sessions = #[rs])
    (hb : bs.get rs.record s!"session {rs.name}'s record" = .ok recB) (hr : String.fromUTF8? recB = some record) :
    sessionRecord e bs = some record := by
  unfold sessionRecord
  rw [hs]
  show (bs.get rs.record s!"session {rs.name}'s record").toOption.bind String.fromUTF8? = some record
  rw [hb]
  exact hr

/-- **`sessionProofs` reads the files v2 verified.** -/
theorem sessionProofs_of {p : _root_.Spot.OneStageZk.Params} {e : _root_.Spot.OneStageZk.Evidence}
    {bs : _root_.Core.Blobs} {record : String} {spec : SessionSpec} {tabs : Array Flock.Setup} {rest : Json × Json}
    {proofs : Array ByteArray} (h : V2Verified p e bs record spec tabs rest proofs) :
    sessionProofs p e bs = some proofs := by
  obtain ⟨rs, tags0, circuitB, pubB, rd, hs, ht, hc, hp, hrd, hrec, hst, hf, -⟩ := h
  have h0 : (#[rs] : Array _root_.Spot.OneStageZk.RunFiles)[0]? = some rs := rfl
  simp only [sessionProofs, hs, h0, ht, hc, hp, hrd, hrec, hst, hf, ok_toOption, Option.bind_eq_bind,
    Option.bind_some]

/-- **Where v2's check accepts, it verified the session's record and proof files** (`V2Verified`). -/
theorem checkBlobsAt_verified {p : _root_.Spot.OneStageZk.Params}
    {c : _root_.Spot.OneStage.Commitment} {ξ : ByteArray} {e : _root_.Spot.OneStageZk.Evidence}
    {bs : _root_.Core.Blobs} {pub : _root_.Spot.OneStage.Public} {prof : _root_.Core.Integrity}
    (h : _root_.Spot.OneStageZk.checkBlobsAt p c ξ e bs = .accept pub prof) :
    ∃ record spec tabs rest proofs, V2Verified p e bs record spec tabs rest proofs := by
  obtain ⟨r, _, hsz, -, -, hrun, hall⟩ := checkBlobsAt_run h
  have hs := eq_single hsz
  rw [hs] at hrun
  obtain ⟨tags0, circuitB, pubB, rd, ht, hc, hp, hrd, _, _, _, _, hstep, -⟩ := run_verified hrun
  obtain ⟨recB, record, spec, tabs, rest, proofs, hb, hrec, hst, hf, hv⟩ := step_verified hstep hall
  exact ⟨record, spec, tabs, rest, proofs, e.sessions[0], tags0, circuitB, pubB, rd, hs, ht, hc, hp, hrd,
    sessionRecord_eq hs hb hrec, hst, hf, hv⟩

/-- **What v2 accepted in what the prover sent is what `sentRecord` and `sentProofs` read**: where v2's case accepts
in an accepted composition, its record and proof files are the ones `Flock.Zk.verify` accepted at the session v2 set
up. -/
theorem sent_verified {k : Verity.Registration.Composite} {s : Verity.Sent} {a b e : ByteArray}
    {p : _root_.Spot.OneStageZk.Params} {c : _root_.Spot.OneStage.Commitment}
    {ev : _root_.Spot.OneStageZk.Evidence} {ξ : ByteArray} {dr : _root_.Properties.Challenger.Draws}
    (hS : Verity.SameAccepts k s) (hcase : V2Case k s a b e)
    (hacc : V2Accepts (_root_.Core.Blobs.ofArray s.blobs) a b e p c ev ξ dr) :
    ∃ record spec tabs rest proofs, caseOf s = .ok (p, c, ev) ∧ sentRecord s = some record ∧
      sentProofs s = some proofs ∧ V2Verified p ev (_root_.Core.Blobs.ofArray s.blobs) record spec tabs rest proofs := by
  obtain ⟨hp, hc, hev, -, -, pub, prof, hck⟩ := hacc
  have hcs := caseOf_of hS hcase hp hc hev
  obtain ⟨record, spec, tabs, rest, proofs, hv⟩ := checkBlobsAt_verified hck
  have hrec : sessionRecord ev (_root_.Core.Blobs.ofArray s.blobs) = some record := by
    obtain ⟨_, _, _, _, _, _, _, _, _, _, h, -⟩ := hv
    exact h
  refine ⟨record, spec, tabs, rest, proofs, hcs, ?_, ?_, hv⟩
  · simp only [sentRecord, hcs, ok_toOption, Option.bind_some, hrec]
  · simp only [sentProofs, hcs, ok_toOption, Option.bind_some, sessionProofs_of hv]

/-! ## The messages against the record v2 decodes -/

set_option maxHeartbeats 1000000 in
/-- **`root_F` as `Record.decode` reads it** (S6): the record's link's `root_f`, which is `Σ`. -/
theorem decode_rootF {spec : SessionSpec} {text : String} {sess : RunRecord} (h : Flock.Record.decode spec text = .ok sess) :
    ∃ j l s, Flock.JsonParse.parse text = .ok j ∧ Flock.Record.fld j "link" = .ok l ∧
      Flock.Record.str l "root_f" = .ok s ∧ Flock.unhex s = .ok sess.rootF ∧ sess.rootF = spec.sigma := by
  revert h
  rw [Flock.Record.decode.eq_def]
  cases hp : Flock.JsonParse.parse text with
  | error e => intro h; cases h
  | ok j =>
    intro h
    dsimp only at h
    rw [pure_bind] at h
    repeat' (first | (obtain ⟨_, _, h⟩ := ebind_ok h) | (split at h))
    all_goals
      simp only [pure, Except.pure, Except.ok.injEq] at h
      subst h
      exact ⟨j, _, _, rfl, by assumption, by assumption, by assumption,
        bytes_eq_of_beq (check_ok_iff.1 ‹Flock.Record.check (_ == spec.sigma) _ = .ok _›)⟩

theorem rootBAt_of {l : Json} {t : String} {rb : ByteArray} (h : RootField l t rb) : rootBAt l t = .ok rb := by
  obtain ⟨v, s, h1, h2, h3⟩ := h
  unfold rootBAt
  rw [h1]
  show (Flock.Record.str v t >>= Flock.unhex) = _
  rw [h2]
  exact h3

/-- **At one table, the roots read from the link's message are the decoded session's `root_B`.** -/
theorem rootBOf_linkMsg {spec : SessionSpec} {record : String} {sess : RunRecord} {t : String}
    (hd : Flock.Record.decode spec record = .ok sess) (ht : spec.tables = #[t]) :
    (linkMsg record spec.tables).bind (rootBOf spec.sigma) = some sess.rootB := by
  obtain ⟨j, l, _, hp, hl, -, hroot, -⟩ := decode_ok hd
  obtain ⟨j', l', s, hp', hl', hs, hu, hσ⟩ := decode_rootF hd
  rw [hp] at hp'
  cases hp'
  rw [hl] at hl'
  cases hl'
  rw [ht] at hroot
  obtain ⟨rb, u, hrb, hnil, hu'⟩ := List.forall₂_cons_left_iff.1 hroot
  rw [List.forall₂_nil_left_iff.1 hnil] at hu'
  have hB : sess.rootB = #[rb] :=
    Array.toList_inj.1 (show sess.rootB.toList = (#[rb] : Array ByteArray).toList from hu'.trans rfl)
  have hm : linkMsg record spec.tables = some (sess.rootF ++ rb).data.toList := by
    rw [ht]
    simp only [linkMsg, hp, hl, hs, hu, ok_toOption, Option.bind_some, ok_bind, qsort_single,
      mapM_single (rootBAt_of hrb), Option.map_some]
    rfl
  rw [hm, Option.bind_some, hσ, rootBOf_append, hB]

/-- **Under retained rounds, a decoded round's message is its bytes**: round `k` of the decoded stream named `name`
has bytes `b`, and `roundMsg` reads them. -/
theorem roundMsg_decode {spec : SessionSpec} {record : String} {sess : RunRecord} {name : String} {y : StreamRec}
    {k : ℕ} (hd : Flock.Record.decode spec record = .ok sess) (hret : spec.retainRounds = true)
    (hy : sess.streams.find? (·.name == name) = some y) (hk : k < y.rounds.size) :
    ∃ b, y.rounds[k].bytes = some b ∧ roundMsg record name k = some b.data.toList := by
  obtain ⟨j, l, recs, hp, -, hrecs, -, -, hstr⟩ := decode_ok hd
  have hy' : sess.streams.toList.find? (·.name == name) = some y := by rw [Array.find?_toList]; exact hy
  obtain ⟨x, ⟨-, rs, hrs, hrds⟩, hx⟩ := forall₂_find?
    (R := fun y x => DecStream spec x y) (p := fun y => y.name == name)
    (q := fun x => (Flock.Record.str x "stream").toOption == some name)
    (fun a b hab => by rw [hab.1]; rfl) (forall₂_swap hstr) hy'
  have hlen : rs.size = y.rounds.size := by simpa using hrds.length_eq
  have hrk : k < rs.size := hlen ▸ hk
  have hfield := forall₂_getElem hrds hrk
  rw [getElem!_pos y.rounds k hk] at hfield
  obtain ⟨⟨-, -, hnone, hsome⟩, -⟩ := hfield
  have hmem : y ∈ sess.streams.toList := List.mem_of_find?_eq_some hy'
  obtain ⟨b, hb⟩ := FlockSoundness.Discharge.Exec.decode_retained hret hd y hmem _ (Array.getElem_mem_toList hk)
  have hnull : Flock.Record.isNull rs[k] "msg" = false := by
    cases hn : Flock.Record.isNull rs[k] "msg"
    · rfl
    · rw [hnone hn] at hb
      cases hb
  obtain ⟨s, b', hs, hu, hb'⟩ := hsome hnull
  rw [hb] at hb'
  cases hb'
  refine ⟨b, hb, ?_⟩
  have hx' : recs.find? (fun x => (Flock.Record.str x "stream").toOption == some name) = some x := by
    rw [← Array.find?_toList]; exact hx
  simp only [roundMsg, hp, hrecs, hx', hrs, Array.getElem?_eq_getElem hrk, hnull, hs, hu, ok_toOption,
    Option.bind_some, ok_bind, Option.map_some, Bool.false_eq_true, ↓reduceIte]

/-- A recorded round is `roundOf` of its bytes, its digest, its coin count and its coins. -/
theorem eq_roundOf {dg : ByteArray → ByteArray} {rd : Round} {b : ByteArray} {n : ℕ} {c : Fin n → 𝔽}
    (hb : rd.bytes = some b) (hs : rd.sha = dg b) (hn : rd.n = n) (hz : rd.coins.size = n)
    (hc : ∀ i : Fin n, (c i).val = rd.coins[i.val]!) : rd = roundOf dg b n c := by
  obtain ⟨n', sha, coins, bytes⟩ := rd
  dsimp only at hb hs hn hz hc
  subst hb hs hn
  unfold roundOf
  congr 1
  apply Array.ext
  · simp [hz]
  · intro i h1 h2
    rw [Array.getElem_ofFn, hc ⟨i, by simpa using h2⟩, getElem!_pos coins i h1]

/-- **Answered submissions whose rounds are a decoded stream's leave that stream's rounds**: where the submissions to
`(j, r)` are, in order, the rounds `rds` (each round's bytes the submission's, its digest theirs under `dg`, its coin
count and coins the submission's), `repsOf` leaves `rds` on `(j, r)`. -/
theorem repsOf_stream {dg : ByteArray → ByteArray} {J : ℕ} {as : List (Answered J)} {j : Fin J} {r : Fin 2}
    {rds : List Round}
    (h : List.Forall₂ (fun (a : Answered J) (rd : Round) => rd.bytes = some a.1.bytes ∧ rd.sha = dg a.1.bytes ∧
      rd.n = a.1.n ∧ rd.coins.size = a.1.n ∧ ∀ i : Fin a.1.n, (a.2 i).val = rd.coins[i.val]!)
      (as.filter fun a => decide (a.1.tab = j ∧ a.1.rep = r)) rds) :
    repsOf dg as j r = rds := by
  rw [repsOf_eq]
  generalize as.filter (fun a => decide (a.1.tab = j ∧ a.1.rep = r)) = l at h
  induction h with
  | nil => rfl
  | cons hab _ ih =>
    obtain ⟨hb, hs, hn, hz, hc⟩ := hab
    rw [List.map_cons, ih, ← eq_roundOf hb hs hn hz hc]

/-! ## The moves against the writer -/

/-- **The view of a transcript, field by field**: an outcome whose roots, link points, rounds and proof files are those
of the session an accepted record decodes to, and of the transcript's files, is the transcript's view. -/
theorem eq_viewZP {I : FlockSoundness.Discharge.Exec.Inputs} {own : Option ByteArray} {J mPts : ℕ}
    {t : FlockSoundness.Discharge.Exec.Transcript} {spec : SessionSpec} {sts : Array Flock.Stmt} {sess : RunRecord}
    (hs : FlockSoundness.Discharge.Exec.setupOfP (FlockSoundness.Discharge.ZkExec.zkInputs I) own t.record =
      .ok (spec, sts))
    (hd : Flock.Record.decode spec t.record = .ok sess) {o : LiveOutJ J mPts} (hB : o.rootB = sess.rootB)
    (hP : o.pts = ptsOf mPts sess.points) (hR : ∀ j r, o.reps j r = repRounds spec sess j.val r.val)
    (hF : o.proofs = t.proofs) : o = viewZP I own J mPts t := by
  have e := sessOf_eq hs hd
  obtain ⟨rootB, pts, reps, proofs⟩ := o
  dsimp only at hB hP hR hF
  subst hB hP hF
  simp only [viewZP, viewA, e, Option.map_some, Option.getD_some,
    FlockSoundness.Discharge.Tables.LiveOutJ.mk.injEq, true_and, and_true]
  funext j r
  exact hR j r

/-- **The session's writer keeps an accepted transcript's view**: what `A.wr` writes, at a sample `S` and a registration
`R`, for the view of a transcript `AcceptsZKP A.I A.own` accepts whose record carries the sample `A.dj S`, is accepted.
`wrZP` does (`writesView_wrZP`); the writer the period's record names is to. -/
def WritesView (A : Live) : Prop :=
  ∀ (S : Finset (Fin A.n)) (R : A.Reg) (t : FlockSoundness.Discharge.Exec.Transcript), AcceptsZKP A.I A.own t →
    FlockSoundness.Discharge.Exec.recordDraw t.record = some (A.dj S) →
      AcceptsZKP A.I A.own (A.wr S R (viewZP A.I A.own _ _ t))

/-- **`wrZP` keeps an accepted transcript's view** (`acceptsZKP_wrZP_planZHJP`, under the session's hidden outputs). -/
theorem writesView_wrZP (A : Live)
    (hw : A.wr = wrZP A.I A.own A.dj fun S R => (FlockSoundness.Discharge.ZkHidden.planZHJP A.dj A.y₀ S R).length) :
    WritesView A := by
  intro S R t ha hd
  rw [hw]
  exact acceptsZKP_wrZP_planZHJP A.hTags.1 A.dj A.y₀ ha hd

/-- **What the writer writes for the moves read off an accepted transcript is accepted**: where the session's writer
keeps an accepted transcript's view (`WritesView`), an outcome whose roots, link points, rounds and proof files are the
accepted transcript's decoded ones (`eq_viewZP`) is written to a transcript `AcceptsZKP A.I A.own` accepts. -/
theorem wr_accepts {A : Live} (hW : WritesView A) {S : Finset (Fin A.n)} {R : A.Reg}
    {t : FlockSoundness.Discharge.Exec.Transcript} (ha : AcceptsZKP A.I A.own t)
    (hdr : FlockSoundness.Discharge.Exec.recordDraw t.record = some (A.dj S)) {spec : SessionSpec}
    {sts : Array Flock.Stmt} {sess : RunRecord}
    (hs : FlockSoundness.Discharge.Exec.setupOfP (FlockSoundness.Discharge.ZkExec.zkInputs A.I) A.own t.record =
      .ok (spec, sts))
    (hd : Flock.Record.decode spec t.record = .ok sess)
    {o : LiveOutJ (FlockSoundness.Discharge.ZkHidden.planZHJP A.dj A.y₀ S R).length
      (FlockSoundness.Discharge.Exec.mPtsOf A.D.unroll.g (A.kd / FlockSoundness.Discharge.Composed.nTab A.I))}
    (hB : o.rootB = sess.rootB) (hP : o.pts = ptsOf _ sess.points)
    (hR : ∀ j r, o.reps j r = repRounds spec sess j.val r.val) (hF : o.proofs = t.proofs) :
    AcceptsZKP A.I A.own (A.wr S R o) := by
  rw [eq_viewZP hs hd hB hP hR hF]
  exact hW S R t ha hdr

end Verity.Sampled
