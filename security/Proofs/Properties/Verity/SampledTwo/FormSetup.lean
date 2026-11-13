module

public import Proofs.Properties.Verity.SampledTwo.FormVerdicts
public import Proofs.Properties.Verity.SampledTwo.LastSlot
public import Proofs.Properties.Verity.Sampled.Record.PartReadsBack
public import Proofs.Properties.Verity.Sampled.Held
public import Proofs.Flock.Claims.JsonNorm
public import Proofs.Flock.Claims.Complete.Json.Canon

/-!
# The session the form verified is the one template `t`'s statement sets up

`SessionCoins.verified_at` (`FormVerdicts.lean`) gives the session the form's check set up for template `t`
(`Spot.OneStageZk.setupAt`) and Flock's masked verdict on it. `SessionCoins.VerdictsAt` (`FormRecords.lean`) asks for
that verdict at the session the template's statement sets up as Flock's verifier reads it (`FlockDeploy.setupOf`).
This module shows the two sessions are one, and proves `VerdictsAt` from what is left.

* **The statement's fields** (`memberFields_stmtRaw`): `SessionMsgs.stmtRaw`'s fields are a member's
  (`MemberChecks.MemberFields`), by definition.
* **The statement's printed files read back** (`readsBack_stmtRaw`, `draw_readsBack`): where the member's partition
  object, its registered reads and the record's sample are printable (`Claims.Json.Good`), Flock's parser reads the
  statement's partition and registered reads back alike (`Flock.PartCong.partition_readsBack`,
  `registered_readsBack`), and its sample back as a value whose rebuilt form (`Flock.norm`) is the record's sample:
  the record's sample is itself rebuilt, and alike values rebuild to one.
* **One session** (`setupOf_of_fields`): a statement with a member's fields, whose printed files read back and whose
  bytes decode as it, sets up under `--zk` the session the form set up on the member's files at the record's sample
  (`MemberChecks.ins_of_fields`, `Verity.Sampled.Record.setup_of_setupAt`).
* **The session digests with SHA-512** (`setupAt_sha512`), as `setupAt_oneTable` gives its one table.
* **`VerdictsAt`** (`verdictsAt_of`), at the table `Flock.Tags.circuit.table`, and so `MsgsOfRecords`
  (`msgsOfRecords_form`).

What `verdictsAt_of` takes beside the join's fields (`HeldOfAt`, `StmtsRead`, `RoundsHeldAt`, `ScopeHeldAt`) and
`StmtsCount`, each a named hypothesis no check gives here:

* `StmtsPrint`: what each session's statement prints is printable. Nothing the setup checks holds a member's
  partition object to well-formed trees or its registered reads to integers.
* `StmtsFit`: each session's statement decodes from its bytes, and the statements' message holds fewer than `2^64`
  statements of fewer than `2^64` bytes each. The form refuses a field its statement can't hold
  (`Spot.OneStageZk.fitsWhy`, `drawFitsWhy`); that refusal is not carried to here, and it bounds no sum.
* `SessionsLive`: a session the masked verdict accepts has messages (`SessionMsgs.sessionMsgs`): its record reads as a
  live session and each recorded round kept its bytes.
-/

@[expose] public section

namespace Verity.SampledTwo.SessionCoins

open Claims
open Lean (Json)

/-! ## The statement, as Flock's parsers read it -/

/-- **`SessionMsgs.stmtRaw`'s fields are a member's** (`MemberChecks.MemberFields`), at the blobs' lookup tables,
whatever its sample and registered reads. -/
theorem memberFields_stmtRaw (m : Spot.OneStageZk.Member) (circuitB pubB : ByteArray) (bs : Core.Blobs)
    (draw registered : Option ByteArray) :
    MemberChecks.MemberFields
      (SessionMsgs.stmtRaw m circuitB pubB (Spot.OneStageZk.tablesOf bs) draw registered) m circuitB pubB bs :=
  ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

/-- **A record's printed sample reads back as the record's sample**: where the record's sample is printable, Flock's
parser reads the bytes `SessionMsgs.drawBytes` prints of it as a value whose rebuilt form (`Flock.norm`) is that
sample. The sample is `norm` of the record's `unit_draw` (`Verity.Sampled.recordDraw_raw`), which `norm` fixes
(`FlockDeploy.norm_norm`), and the parser reads a printable value's text back alike it
(`Flock.PartCong.printed_readsBack`), which rebuilds to the same value (`FlockDeploy.norm_congr`). -/
theorem draw_readsBack {record : String}
    (hd : ∀ dj, Flock.Witness.recordDraw record = some dj → Claims.Json.Good dj) :
    ∃ D, (SessionMsgs.drawBytes record).mapM (FlockDeploy.jsonOf "draw") = .ok D ∧
      D.map Flock.norm = Flock.Witness.recordDraw record := by
  unfold SessionMsgs.drawBytes
  cases hrd : Flock.Witness.recordDraw record with
  | none => exact ⟨none, rfl, rfl⟩
  | some dj =>
    obtain ⟨P, hP, hsame⟩ := Flock.PartCong.printed_readsBack "draw" (hd dj hrd)
    obtain ⟨_, d, -, -, -, hdj⟩ := Verity.Sampled.recordDraw_raw hrd
    have hn : Flock.norm dj = dj := by
      rw [hdj]
      exact FlockDeploy.norm_norm d
    refine ⟨P, hP, ?_⟩
    rcases hsame with rfl | ⟨a, b, rfl, hb, hs⟩
    · exact congrArg some hn
    · have hbd : b = dj := (Option.some.inj hb).symm
      rw [hbd] at hs
      exact congrArg some ((FlockDeploy.norm_congr a dj (FlockComplete.JsonRT.same_jsonEquiv hs)).trans hn)

/-- **A member's statement's printed files read back** (`MemberChecks.ReadsBack`), where the member's partition
object, its registered reads and the record's sample are printable. -/
theorem readsBack_stmtRaw {m : Spot.OneStageZk.Member} {circuitB pubB : ByteArray}
    {tables : List (String × ByteArray)} {record : String} (hp : Claims.Json.Good m.own.partition)
    (hr : ∀ j, m.own.registered = some j → Claims.Json.Good j)
    (hd : ∀ dj, Flock.Witness.recordDraw record = some dj → Claims.Json.Good dj) :
    MemberChecks.ReadsBack
      (SessionMsgs.stmtRaw m circuitB pubB tables (SessionMsgs.drawBytes record)
        (m.own.registered.map fun j => (Flock.RecordText.printT j).toUTF8)) m record :=
  ⟨Flock.PartCong.partition_readsBack hp, Flock.PartCong.registered_readsBack hr, draw_readsBack hd⟩

/-- **A member's statement sets up the session the form set up**: a statement `w` with the member's fields, whose
printed files read back and whose bytes decode as `w`, sets up under `--zk` (`FlockDeploy.setupOf true`) the session
`spec` the form set up on the member's files at the record's sample, with the tables' verifier-side setups the form
verified the record at. -/
theorem setupOf_of_fields {w : FlockDeploy.Raw} {m : Spot.OneStageZk.Member} {circuitB pubB : ByteArray}
    {bs : Core.Blobs} {record : String} {rd : Spot.OneStageZk.Reads}
    (hf : MemberChecks.MemberFields w m circuitB pubB bs) (hr : MemberChecks.ReadsBack w m record)
    (hdec : FlockDeploy.decode w.encode = some w) (hrd : Spot.OneStageZk.readsOf m.own.registered = .ok rd)
    {spec : Flock.SessionSpec} {tabs : Array Flock.Setup} {p o : Json}
    (hst : Spot.OneStageZk.setupAt (Spot.OneStageZk.inputsOf Flock.Tags.circuit m.own circuitB pubB bs) rd
      (Flock.Witness.recordDraw record) = .ok (spec, tabs, p, o)) :
    ∃ si sts, FlockDeploy.setupOf true w.encode = some (si, spec, sts) ∧ tabs = FlockDeploy.tabsOf sts := by
  have hpl : Flock.Tags.circuit.plainLeaves = false := rfl
  obtain ⟨si, hins, hm, hdraw⟩ := MemberChecks.ins_of_fields hf hr hrd hpl
  obtain ⟨sts, hsu, htabs⟩ := Verity.Sampled.Record.setup_of_setupAt hm hdraw hst
  have hinsOf : FlockDeploy.insOf true w.encode = some si := by
    unfold FlockDeploy.insOf
    rw [hdec]
    simp [hf.zk, hins, Except.toOption]
  refine ⟨si, sts, ?_, htabs⟩
  unfold FlockDeploy.setupOf
  rw [hinsOf]
  simp [hsu, Except.toOption]

/-- **The session `--zk` sets up at one table of hm96-sha512 rows digests its record with SHA-512.** -/
theorem setupAt_sha512 {W : Flock.Witness.Inputs} {rd : Spot.OneStageZk.Reads} {dr : Option Json}
    {spec : Flock.SessionSpec} {tabs : Array Flock.Setup} {x : Json × Json}
    (hh : (Flock.Witness.zkTags W).hiddenOutputs = true) (hr : (Flock.Witness.zkTags W).hm96Rows = true)
    (hc : W.count = 1) (h : Spot.OneStageZk.setupAt W rd dr = .ok (spec, tabs, x)) :
    spec.sha512Digests = true := by
  obtain ⟨sts, hs, -⟩ := Verity.Sampled.setupAt_ok h
  obtain ⟨hset, -⟩ := Verity.Sampled.sessionAt_ok hs
  rcases FlockSoundness.Discharge.HiddenExec.setupTables_okHP hh hset with ⟨-, st, hst, rfl, -⟩ | ⟨h1, -⟩
  · obtain ⟨_, hP⟩ := FlockSoundness.Discharge.Hidden.setupHidden_specP hst
    show st.tags.hm96Rows = true
    rw [hP.tagsEq]
    exact hr
  · exact absurd h1 (by omega)

/-! ## `VerdictsAt`, from what is left -/

/-- **What each session's statement prints is printable**: on a play the form accepts whose log the check read is
the device's, template `t`'s member's partition object and registered reads, and the sample of its session's record,
are printable (`Claims.Json.Good`: every object a well-formed tree, every number an integer). -/
def StmtsPrint (C : CaseOf) (Good : Verity.Exchange → Prop) : Prop :=
  ∀ o k p e, FormAccepts C o → Good o → C o = some k → SessionMsgs.formOf k = some (p, e) →
    ∀ t m fs record, SessionMsgs.memberAt p e t = some m → SessionMsgs.filesAt e t = some fs →
      SessionMsgs.recordAt k.blobs fs = some record →
        Claims.Json.Good m.own.partition ∧ (∀ j, m.own.registered = some j → Claims.Json.Good j) ∧
          ∀ dj, Flock.Witness.recordDraw record = some dj → Claims.Json.Good dj

/-- **The session statements fit their length words**: on a play the form accepts whose log the check read is the
device's, each template's session statement decodes from its bytes (`each`), and the statements' message holds
fewer than `2^64` statements, each of fewer than `2^64` bytes (`all`). -/
structure StmtsFit (C : CaseOf) (Good : Verity.Exchange → Prop) : Prop where
  each : ∀ o k p e, FormAccepts C o → Good o → C o = some k → SessionMsgs.formOf k = some (p, e) →
    ∀ t m fs record circuitB pubB, SessionMsgs.memberAt p e t = some m → SessionMsgs.filesAt e t = some fs →
      SessionMsgs.recordAt k.blobs fs = some record → k.blobs.get m.own.circuit "circuit" = .ok circuitB →
      k.blobs.get m.own.pub "public file" = .ok pubB →
        FlockDeploy.decode
          (SessionMsgs.stmtRaw m circuitB pubB (Spot.OneStageZk.tablesOf k.blobs) (SessionMsgs.drawBytes record)
            (m.own.registered.map fun j => (Flock.RecordText.printT j).toUTF8)).encode =
          some (SessionMsgs.stmtRaw m circuitB pubB (Spot.OneStageZk.tablesOf k.blobs)
            (SessionMsgs.drawBytes record) (m.own.registered.map fun j => (Flock.RecordText.printT j).toUTF8))
  all : ∀ o k p e xs, FormAccepts C o → Good o → C o = some k → SessionMsgs.formOf k = some (p, e) →
    (List.range e.stmts.size).mapM (SessionMsgs.stmtAtCase k) = some xs →
      xs.length < 2 ^ 64 ∧ ∀ x ∈ xs, x.length < 2 ^ 64

/-- **A session the masked verdict accepts has messages**: on a play the form accepts whose log the check read is
the device's, where the masked verdict accepts template `t`'s session's record and proof files at its statement's
setup (`SessionMsgs.SessionParts`), the session has messages (`SessionMsgs.sessionMsgs`). -/
def SessionsLive (C : CaseOf) (Good : Verity.Exchange → Prop) : Prop :=
  ∀ o k, FormAccepts C o → Good o → C o = some k → ∀ t x s spec sts record proofs,
    SessionMsgs.SessionParts k t x s spec sts record proofs →
      Flock.Zk.verify spec (FlockDeploy.tabsOf sts) record proofs = .ok () →
        ∃ ms, SessionMsgs.sessionMsgs spec x record proofs = some ms

/-- **`VerdictsAt`, at the table `Flock.Tags.circuit.table`**: on an accepted play, the statement the statements'
message states for template `t` is the template's session statement on the case (`SessionMsgs.stmtAt_stmtsMsg`),
which sets up one table with the template's count of rounds (`reps_of_scope`). The check verified the session at the
member's files (`verified_at`), which are that statement's fields (`stmtAtCase_of_member`), so the session the
statement sets up is the one the check verified (`setupOf_of_fields`), of one table, `Flock.Tags.circuit.table`
(`setupAt_oneTable`), under SHA-512 digests (`setupAt_sha512`). -/
theorem verdictsAt_of {A : Live₂} {C : CaseOf} {S : SessionReading} {Good : Verity.Exchange → Prop}
    (hS : ∀ o, S.stmts o = (C o).bind SessionMsgs.stmtsMsg) (hof : HeldOfAt A C) (hst : StmtsRead A C S)
    (hrounds : RoundsHeldAt A C S) (hscope : ScopeHeldAt A C S) (hcount : StmtsCount A C Good)
    (hprint : StmtsPrint C Good) (hfit : StmtsFit C Good) (hlive : SessionsLive C Good) :
    VerdictsAt A C Good fun _ => Flock.Tags.circuit.table := by
  intro o k hacc hgood hk t ht
  have hacc₀ := hacc
  obtain ⟨k', hk', hon⟩ := hacc₀
  obtain rfl : k = k' := Option.some.inj (hk.symm.trans hk')
  obtain ⟨_, inp, -, hin, -, it, _, hit, -⟩ := input_of_accepts hon
  have hi : inputOn C o = some inp := by
    show (C o).bind inputOfCase = some inp
    rw [hk]
    exact hin
  obtain ⟨m₂, hm₂, -⟩ := hst o inp it hacc hi hit
  have hmsg : SessionMsgs.stmtsMsg k = some m₂ := by
    have h := hS o
    rw [hk] at h
    exact h.symm.trans hm₂
  obtain ⟨p, e, xs, hf, hxs, -⟩ := SessionMsgs.stmtsOf_stmtsMsg hmsg
  have hte : t < e.stmts.size := by
    rw [hcount o k p e hacc hgood hk hf]
    exact ht
  obtain ⟨hl, hb⟩ := hfit.all o k p e xs hacc hgood hk hf hxs
  have hx := SessionMsgs.stmtAt_stmtsMsg A.h hmsg hf hxs hl hb hte
  obtain ⟨s', spec', st, hset', hk₀, -⟩ :=
    reps_of_scope (twoHolds_at hof hacc hi hit) (hrounds o m₂ hacc hm₂) (hscope o m₂ hacc hm₂) ht
  obtain ⟨fs, hfs⟩ := SessionMsgs.filesAt_of_stmt hx hf
  obtain ⟨m, circuitB, pubB, rd, record, spec, tabs, ⟨p0, o0⟩, proofs, hm, hc, hp, hrd, hrec, hsetup, hfiles,
    hzk⟩ := verified_at hacc hk hf hfs
  have hxw := stmtAtCase_of_member hf hm hfs hrec hc hp
  obtain ⟨hgp, hgr, hgd⟩ := hprint o k p e hacc hgood hk hf t m fs record hm hfs hrec
  have hdec := hfit.each o k p e hacc hgood hk hf t m fs record circuitB pubB hm hfs hrec hc hp
  obtain ⟨si, sts, hset, htabs⟩ :=
    setupOf_of_fields
      (memberFields_stmtRaw m circuitB pubB k.blobs (SessionMsgs.drawBytes record)
        (m.own.registered.map fun j => (Flock.RecordText.printT j).toUTF8))
      (readsBack_stmtRaw hgp hgr hgd) hdec hrd hsetup
  have hxe : SpotFlock.stmtAt A.h m₂ t =
      (SessionMsgs.stmtRaw m circuitB pubB (Spot.OneStageZk.tablesOf k.blobs) (SessionMsgs.drawBytes record)
        (m.own.registered.map fun j => (Flock.RecordText.printT j).toUTF8)).encode :=
    Option.some.inj (hx.symm.trans hxw)
  have hset₂ : FlockDeploy.setupOf true (SpotFlock.stmtAt A.h m₂ t) = some (si, spec, sts) := by
    rw [hxe]
    exact hset
  have hsame : (s', spec', #[st]) = (si, spec, sts) := Option.some.inj (hset'.symm.trans hset₂)
  have hrest : (spec', #[st]) = (spec, sts) := congrArg Prod.snd hsame
  have hspec : spec' = spec := congrArg Prod.fst hrest
  have hsts : #[st] = sts := congrArg Prod.snd hrest
  rw [hspec] at hset' hk₀
  rw [← hsts] at htabs
  rw [htabs] at hzk
  have htag : Flock.Tags.find? Spot.OneStageZk.STATEMENT = some Flock.Tags.circuit := rfl
  have hh : (Flock.Witness.zkTags
      (Spot.OneStageZk.inputsOf Flock.Tags.circuit m.own circuitB pubB k.blobs)).hiddenOutputs = true :=
    (Verity.Sampled.statement_zkTags htag _ _).1
  have hrows : (Flock.Witness.zkTags
      (Spot.OneStageZk.inputsOf Flock.Tags.circuit m.own circuitB pubB k.blobs)).hm96Rows = true := rfl
  have hparts : SessionMsgs.SessionParts k t (SpotFlock.stmtAt A.h m₂ t) s' spec #[st] record proofs :=
    ⟨p, e, fs, hf, hfs, hrec, hx, hset', hfiles⟩
  obtain ⟨ms, hms⟩ := hlive o k hacc hgood hk t _ _ _ _ _ _ hparts hzk
  exact ⟨_, s', spec, st, record, proofs, ms, hparts, hzk, setupAt_sha512 hh hrows rfl hsetup,
    setupAt_oneTable hh rfl hsetup, hk₀, hms⟩

/-- **`MsgsOfRecords` at the form's reading**, at the table `Flock.Tags.circuit.table`: `msgsOfRecords_of` at
`verdictsAt_of`, where the reading's statements and messages are the case's. -/
theorem msgsOfRecords_form {A : Live₂} {C : CaseOf} {S : SessionReading} {Good : Verity.Exchange → Prop}
    (hS : ∀ o, S.stmts o = (C o).bind SessionMsgs.stmtsMsg)
    (hM : ∀ o t j, S.msg o t j = (C o).bind fun k => SessionMsgs.msgAtCase k t j) (hof : HeldOfAt A C)
    (hst : StmtsRead A C S) (hrounds : RoundsHeldAt A C S) (hscope : ScopeHeldAt A C S)
    (hcount : StmtsCount A C Good) (hprint : StmtsPrint C Good) (hfit : StmtsFit C Good)
    (hlive : SessionsLive C Good) : MsgsOfRecords A C S Good fun _ => Flock.Tags.circuit.table :=
  msgsOfRecords_of hM (verdictsAt_of hS hof hst hrounds hscope hcount hprint hfit hlive)

end Verity.SampledTwo.SessionCoins
