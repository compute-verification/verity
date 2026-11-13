module

public import Proofs.Properties.Verity.SampledTwo.FormVerdicts
public import Proofs.Properties.Verity.SampledTwo.LastSlot
public import Proofs.Properties.Verity.Sampled.Record.PartReadsBack
public import Proofs.Properties.Verity.Sampled.Held
public import Proofs.Flock.Claims.JsonNorm
public import Proofs.Flock.Claims.Complete.Json.Canon
public import Proofs.Flock.Claims.Complete.Json.Stmt
public import Proofs.Flock.SessionLoop.Concrete
import all Init.Data.String.Defs

/-!
# The session the form verified is the one template `t`'s statement sets up

`SessionCoins.verified_at` (`FormVerdicts.lean`) gives the session the form's check set up for template `t`
(`Spot.OneStageZk.setupAt`) and Flock's masked verdict on it. `SessionCoins.VerdictsAt` (`FormRecords.lean`) asks for
that verdict at the session the template's statement sets up as Flock's verifier reads it (`FlockDeploy.setupOf`).
This module shows the two sessions are one, and proves `VerdictsAt` from what is left.

* **The statement's fields** (`memberFields_stmtRaw`): `SessionMsgs.stmtRaw`'s fields are a member's
  (`MemberChecks.MemberFields`), by definition.
* **The sample the form set a session up at is printable** (`draw_good`): it is the record's `unit_draw` rebuilt, so
  its objects are well-formed trees, and the setup decodes it to a subset or a stratified sample
  (`Verity.Sampled.setupAt_drawLaw`), each printable where it decodes.
* **The statement's printed files read back** (`readsBack_stmtRaw`, `draw_readsBack`): where the member's partition
  object's objects are well-formed trees (`Claims.Json.Wf`) and its registered reads are printable
  (`Claims.Json.Good`), at a session the form set up, Flock's parser reads the statement's partition and registered
  reads back alike (`Flock.PartCong.partition_readsBack_of_setup`, `registered_readsBack`), and its sample back as a
  value whose rebuilt form (`Flock.norm`) is the record's sample: the record's sample is itself rebuilt, and alike
  values rebuild to one.
* **One session** (`setupOf_of_fields`): a statement with a member's fields, whose printed files read back and whose
  bytes decode as it, sets up under `--zk` the session the form set up on the member's files at the record's sample
  (`MemberChecks.ins_of_fields`, `Verity.Sampled.Record.setup_of_setupAt`), where the circuit statement's leaves are
  not plain, which the member's statement passing gives (`verified_at`).
* **The statement decodes from its bytes** (`stmtRaw_decode`): the form refuses a member's file, lookup table,
  program, partition object or registered reads, or a record's sample, at `2^64` bytes or more
  (`Spot.OneStageZk.fitsWhy`, `drawFitsWhy`), so every length of the statement fits its 8-byte word
  (`stmtRaw_fits`) and `FlockDeploy.decode` reads it back (`FlockDeploy.decode_encode_of_fits`).
* **The session digests with SHA-512** (`setupAt_sha512`), as `setupAt_oneTable` gives its one table.
* **`VerdictsAt`** (`verdictsAt_of`), at the table `Flock.Tags.circuit.table`, and so `MsgsOfRecords`
  (`msgsOfRecords_form`).

What `verdictsAt_of` takes beside the join's fields (`HeldOfAt`, `StmtsRead`, `RoundsHeldAt`, `ScopeHeldAt`) and
`StmtsCount`, each a named hypothesis no check gives here:

* `StmtsPrint`: each member's partition object's objects are well-formed trees, and its registered reads are
  printable. Nothing the setup checks holds a member's partition object to well-formed trees or its registered
  reads to integers.
* `StmtsFit`: the statements' message holds fewer than `2^64` statements of fewer than `2^64` bytes each. The form
  refuses a field its statement can't hold, which bounds each field and no sum of them.
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

/-- **The sample the form set a session up at is printable**: the record's sample is its `unit_draw` rebuilt
(`Verity.Sampled.recordDraw_raw`), so each of its objects is a well-formed tree (`FlockDeploy.wf_norm`). The setup
decodes it to a subset or a stratified sample (`Verity.Sampled.setupAt_drawLaw`, `drawLaw_ok`, `lawWhy_zk`), and a
well-formed value that decodes to either is printable (`FlockComplete.JsonRT.good_subset`, `good_stratified`). -/
theorem draw_good {W : Flock.Witness.Inputs} {rd : Spot.OneStageZk.Reads} {record : String} {dj : Json}
    {x : Flock.SessionSpec × Array Flock.Setup × Json × Json} (hrd : Flock.Witness.recordDraw record = some dj)
    (h : Spot.OneStageZk.setupAt W rd (Flock.Witness.recordDraw record) = .ok x) : Claims.Json.Good dj := by
  obtain ⟨_, d, -, -, -, hdj⟩ := Verity.Sampled.recordDraw_raw hrd
  have hw : Claims.Json.Wf dj := by
    rw [hdj]
    exact FlockDeploy.wf_norm d
  rw [hrd] at h
  obtain ⟨D, hD, hlaw⟩ := Verity.Sampled.drawLaw_ok (Verity.Sampled.setupAt_drawLaw h)
  rcases Verity.Sampled.lawWhy_zk hlaw with ⟨kd, hl⟩ | ⟨K, St, hl⟩
  · exact FlockComplete.JsonRT.good_subset hw hD hl
  · exact FlockComplete.JsonRT.good_stratified hw hD hl

/-- **A member's statement's printed files read back** (`MemberChecks.ReadsBack`), at a session the form set up on
the member's files at the record's sample, where the member's partition object's objects are well-formed trees and
its registered reads are printable. The setup returned only where `Flock.canon` prints the partition object
(`Flock.PartCong.partition_readsBack_of_setup`), and its sample is printable (`draw_good`). -/
theorem readsBack_stmtRaw {m : Spot.OneStageZk.Member} {circuitB pubB : ByteArray} {bs : Core.Blobs}
    {tables : List (String × ByteArray)} {rd : Spot.OneStageZk.Reads} {record : String}
    {x : Flock.SessionSpec × Array Flock.Setup × Json × Json} (hp : Claims.Json.Wf m.own.partition)
    (hr : ∀ j, m.own.registered = some j → Claims.Json.Good j)
    (hst : Spot.OneStageZk.setupAt (Spot.OneStageZk.inputsOf Flock.Tags.circuit m.own circuitB pubB bs) rd
      (Flock.Witness.recordDraw record) = .ok x) :
    MemberChecks.ReadsBack
      (SessionMsgs.stmtRaw m circuitB pubB tables (SessionMsgs.drawBytes record)
        (m.own.registered.map fun j => (Flock.RecordText.printT j).toUTF8)) m record := by
  have hd : ∀ dj, Flock.Witness.recordDraw record = some dj → Claims.Json.Good dj := fun dj hdj => draw_good hdj hst
  obtain ⟨spec, tabs, rest⟩ := x
  obtain ⟨sts, hs, -⟩ := Verity.Sampled.setupAt_ok hst
  obtain ⟨hset, -⟩ := Verity.Sampled.sessionAt_ok hs
  have htag : Flock.Tags.find? Spot.OneStageZk.STATEMENT = some Flock.Tags.circuit := rfl
  have hh : (Flock.Witness.zkTags
      (Spot.OneStageZk.inputsOf Flock.Tags.circuit m.own circuitB pubB bs)).hiddenOutputs = true :=
    (Verity.Sampled.statement_zkTags htag _ _).1
  have hb : (Flock.Witness.zkTags
      (Spot.OneStageZk.inputsOf Flock.Tags.circuit m.own circuitB pubB bs)).bindings = true := rfl
  exact ⟨Flock.PartCong.partition_readsBack_of_setup (o := m.own.partition) hp hh hb hset,
    Flock.PartCong.registered_readsBack hr, draw_readsBack hd⟩

/-- **A member's statement sets up the session the form set up**: a statement `w` with the member's fields, whose
printed files read back and whose bytes decode as `w`, sets up under `--zk` (`FlockDeploy.setupOf true`) the session
`spec` the form set up on the member's files at the record's sample, with the tables' verifier-side setups the form
verified the record at. -/
theorem setupOf_of_fields {w : FlockDeploy.Raw} {m : Spot.OneStageZk.Member} {circuitB pubB : ByteArray}
    {bs : Core.Blobs} {record : String} {rd : Spot.OneStageZk.Reads}
    (hpl : Flock.Tags.circuit.plainLeaves = false) (hf : MemberChecks.MemberFields w m circuitB pubB bs)
    (hr : MemberChecks.ReadsBack w m record) (hdec : FlockDeploy.decode w.encode = some w)
    (hrd : Spot.OneStageZk.readsOf m.own.registered = .ok rd)
    {spec : Flock.SessionSpec} {tabs : Array Flock.Setup} {p o : Json}
    (hst : Spot.OneStageZk.setupAt (Spot.OneStageZk.inputsOf Flock.Tags.circuit m.own circuitB pubB bs) rd
      (Flock.Witness.recordDraw record) = .ok (spec, tabs, p, o)) :
    ∃ si sts, FlockDeploy.setupOf true w.encode = some (si, spec, sts) ∧ tabs = FlockDeploy.tabsOf sts := by
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

/-! ## The statement decodes from its bytes -/

private theorem ite_some_none {α : Type} {c : Prop} [Decidable c] {s : α} {o : Option α}
    (h : (if c then some s else o) = none) : ¬c ∧ o = none := by
  by_cases hc : c
  · rw [if_pos hc] at h
    cases h
  · rw [if_neg hc] at h
    exact ⟨hc, h⟩

private theorem statement_size : Spot.OneStageZk.STATEMENT.toUTF8.size < 2 ^ 64 := by decide +kernel

private theorem toUTF8_size (s : String) : s.toUTF8.size = s.utf8ByteSize := String.size_toByteArray

/-- A printed value the check holds under `2^64` bytes (`Spot.OneStageZk.printedSize`) is under `2^64` bytes as a
statement holds it. -/
private theorem printed_fits {o : Option Json}
    (h : ¬(o.any fun j => decide (Spot.OneStageZk.WORD ≤ Spot.OneStageZk.printedSize j)) = true) (b : ByteArray)
    (hb : (o.map fun j => (Flock.RecordText.printT j).toUTF8) = some b) : b.size < 2 ^ 64 := by
  cases o with
  | none => simp at hb
  | some j =>
    obtain rfl : (Flock.RecordText.printT j).toUTF8 = b := Option.some.inj hb
    rw [toUTF8_size]
    exact Nat.lt_of_not_le fun (hc : Spot.OneStageZk.WORD ≤ Spot.OneStageZk.printedSize j) => h (decide_eq_true hc)

/-- **Every length of a member's statement fits its 8-byte word** (`FlockDeploy.Raw.Fits`), where the form's check
refused none of them: the member's files, the lookup tables, its program, partition object and registered reads
(`Spot.OneStageZk.fitsWhy`), and the record's sample (`Spot.OneStageZk.drawFitsWhy`). -/
theorem stmtRaw_fits {m : Spot.OneStageZk.Member} {circuitB pubB : ByteArray} {bs : Core.Blobs} {record : String}
    (hfit : Spot.OneStageZk.fitsWhy Spot.OneStageZk.WORD circuitB pubB (Spot.OneStageZk.tablesOf bs)
      (some m.own.program) (some m.own.partition) m.own.registered = none)
    (hd : Spot.OneStageZk.drawFitsWhy Spot.OneStageZk.WORD (Flock.Witness.recordDraw record) = none) :
    (SessionMsgs.stmtRaw m circuitB pubB (Spot.OneStageZk.tablesOf bs) (SessionMsgs.drawBytes record)
      (m.own.registered.map fun j => (Flock.RecordText.printT j).toUTF8)).Fits := by
  unfold Spot.OneStageZk.fitsWhy at hfit
  obtain ⟨h1, hfit⟩ := ite_some_none hfit
  obtain ⟨h2, hfit⟩ := ite_some_none hfit
  obtain ⟨h3, hfit⟩ := ite_some_none hfit
  obtain ⟨h4, hfit⟩ := ite_some_none hfit
  obtain ⟨h5, hfit⟩ := ite_some_none hfit
  obtain ⟨h6, hfit⟩ := ite_some_none hfit
  obtain ⟨h7, -⟩ := ite_some_none hfit
  unfold Spot.OneStageZk.drawFitsWhy at hd
  obtain ⟨h8, -⟩ := ite_some_none hd
  refine ⟨statement_size, Nat.lt_of_not_le h1, Nat.lt_of_not_le h2, Nat.lt_of_not_le h3, fun q hq => ?_,
    (by show 1 < 2 ^ 64; decide), printed_fits h8, ?_, ?_, ?_, ?_, ?_, ?_, printed_fits h7, ?_⟩
  · have hq' : ¬(decide (Spot.OneStageZk.WORD ≤ q.1.utf8ByteSize) || decide (Spot.OneStageZk.WORD ≤ q.2.size)) =
        true := fun hc => h4 (List.any_eq_true.2 ⟨q, hq, hc⟩)
    simp only [Bool.or_eq_true, decide_eq_true_eq, not_or] at hq'
    exact ⟨by rw [toUTF8_size]; exact Nat.lt_of_not_le hq'.1, Nat.lt_of_not_le hq'.2⟩
  · intro b hb
    cases hb
    rw [toUTF8_size]
    exact Nat.lt_of_not_le
      fun (hc : Spot.OneStageZk.WORD ≤ Spot.OneStageZk.printedSize m.own.partition) => h6 (decide_eq_true hc)
  · intro b hb
    cases hb
    exact Nat.lt_of_not_le fun (hc : Spot.OneStageZk.WORD ≤ m.own.program.size) => h5 (decide_eq_true hc)
  · intro n hn; cases hn
  · intro b hb; cases hb
  · intro n hn; cases hn
  · intro b hb; cases hb
  · intro b hb; cases hb

/-- **A member's statement decodes from its bytes**, where the form's check refused none of its lengths
(`stmtRaw_fits`, `FlockDeploy.decode_encode_of_fits`). -/
theorem stmtRaw_decode {m : Spot.OneStageZk.Member} {circuitB pubB : ByteArray} {bs : Core.Blobs} {record : String}
    (hfit : Spot.OneStageZk.fitsWhy Spot.OneStageZk.WORD circuitB pubB (Spot.OneStageZk.tablesOf bs)
      (some m.own.program) (some m.own.partition) m.own.registered = none)
    (hd : Spot.OneStageZk.drawFitsWhy Spot.OneStageZk.WORD (Flock.Witness.recordDraw record) = none) :
    FlockDeploy.decode
        (SessionMsgs.stmtRaw m circuitB pubB (Spot.OneStageZk.tablesOf bs) (SessionMsgs.drawBytes record)
          (m.own.registered.map fun j => (Flock.RecordText.printT j).toUTF8)).encode =
      some (SessionMsgs.stmtRaw m circuitB pubB (Spot.OneStageZk.tablesOf bs) (SessionMsgs.drawBytes record)
        (m.own.registered.map fun j => (Flock.RecordText.printT j).toUTF8)) :=
  FlockDeploy.decode_encode_of_fits (stmtRaw_fits hfit hd)

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

/-- **What each member's statement prints of the member reads back**: on a play the form accepts whose log the
check read is the device's, each object of template `t`'s member's partition object is a well-formed tree
(`Claims.Json.Wf`), and the member's registered reads are printable (`Claims.Json.Good`: every object a well-formed
tree, every number an integer). -/
def StmtsPrint (C : CaseOf) (Good : Verity.Exchange → Prop) : Prop :=
  ∀ o k p e, FormAccepts C o → Good o → C o = some k → SessionMsgs.formOf k = some (p, e) →
    ∀ t m, SessionMsgs.memberAt p e t = some m →
      Claims.Json.Wf m.own.partition ∧ ∀ j, m.own.registered = some j → Claims.Json.Good j

/-- **The statements' message fits its length words**: on a play the form accepts whose log the check read is the
device's, the statements' message holds fewer than `2^64` statements, each of fewer than `2^64` bytes. -/
def StmtsFit (C : CaseOf) (Good : Verity.Exchange → Prop) : Prop :=
  ∀ o k p e xs, FormAccepts C o → Good o → C o = some k → SessionMsgs.formOf k = some (p, e) →
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
  obtain ⟨hl, hb⟩ := hfit o k p e xs hacc hgood hk hf hxs
  have hx := SessionMsgs.stmtAt_stmtsMsg A.h hmsg hf hxs hl hb hte
  obtain ⟨s', spec', st, hset', hk₀, -⟩ :=
    reps_of_scope (twoHolds_at hof hacc hi hit) (hrounds o m₂ hacc hm₂) (hscope o m₂ hacc hm₂) ht
  obtain ⟨fs, hfs⟩ := SessionMsgs.filesAt_of_stmt hx hf
  obtain ⟨m, circuitB, pubB, rd, record, spec, tabs, ⟨p0, o0⟩, proofs, hpl, hm, hc, hp, hrd, hrec, hsetup, hfiles,
    hzk, hfw, hfd⟩ := verified_at hacc hk hf hfs
  have hxw := stmtAtCase_of_member hf hm hfs hrec hc hp
  obtain ⟨hgp, hgr⟩ := hprint o k p e hacc hgood hk hf t m hm
  have hdec := stmtRaw_decode hfw hfd
  obtain ⟨si, sts, hset, htabs⟩ :=
    setupOf_of_fields hpl
      (memberFields_stmtRaw m circuitB pubB k.blobs (SessionMsgs.drawBytes record)
        (m.own.registered.map fun j => (Flock.RecordText.printT j).toUTF8))
      (readsBack_stmtRaw hgp hgr hsetup) hdec hrd hsetup
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
