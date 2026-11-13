module

public import Proofs.Properties.Verity.Sampled.Record.Session
public import Proofs.Properties.Verity.Sampled.Record.Inner
public import Proofs.Properties.Verity.Sampled.Event
public import Proofs.Flock.SpotFlock.Code

/-!
# A member's session, accepted by the per-member form, is in Flock's masked scope (`MemberChecks`)

The two-stage form checks each sampled member's sessions with `Spot.OneStageZk.stepMember`: the member's statement
(`Member.statement`: `statementOf` at the member's own inputs, then its population and block checks), then `step` on
each session, which sets the session up at its record's sample (`setupAt`) and pushes `verdictOf`'s verdict. Where the
statement's run accepts every session (`Run.all`), each session's statement, as `flock-verify verify --zk` reads it, is in
the scope the two-stage proof reads of it (`SpotFlock.sessionOk`): it sets up in Flock's masked scope (`zkOk`), passes
the deployed scope check (`FlockDeploy.scopeCheck true`) and has hidden outputs (`hiddenOf`).

The session's statement is any `FlockDeploy.Raw` whose fields are the member's (`MemberFields`: the statement
`verity/flock-circuit`, the member's circuit and public files, the blobs' lookup tables, `--zk`, one table, the member's
program, and no `--work`, `--stratified`, `--closure` or `--public-inputs`) and whose JSON files read back as the member's
(`ReadsBack`: its partition object, up to `PartSame`, its registered reads and the record's sample). The statement the
sessions' reading builds (the member's statement with each value printed by `Flock.RecordText.printT`) has these fields
by definition.

* `statement_ok` (the member's statement): `Member.statement` passing is `statementOf`'s statement at the member's
  files and registered reads (`StepAt.ofOwn`).
* `step_all`, `foldl_all`, `stepMember_sessions` (the accepted sessions): a run that accepts every session had each
  session's verdict `.ok`, at the record the blobs hold for it.
* `ins_of_fields` (the statement's reading): `FlockDeploy.Raw.ins` of such a statement reads v2's inputs at the
  member's files (`Record.SessionMatch`), with the record's sample.
* `memberSession_checks`, `memberSession_ok`: the three checks, and `sessionOk` where the statement has its template's
  reference statement's shape.

What the statements here take as hypotheses, besides the form's accept:
* `decode w.encode = some w`: the statement's bytes read back as its fields, which holds wherever each length fits its
  8-byte word (`Verity.Recursive.Part.decode_encode_fits`);
* `ReadsBack`: the verifier's JSON parser reads the statement's registered reads and sample as the values the form
  read, and its partition object as the form's or as an object alike it, which is what a printed object reads back as;
* in `memberSession_ok`, the shape: the statement's tables have the `m` of its template's reference statement's.
-/

@[expose] public section

namespace Verity.SampledTwo.MemberChecks

open Lean (Json)
open Flock (V)
open Verity.Recursive.BaseSetup (bind_ok)
open Verity.Sampled.Record (SessionMatch)
open Flock.PartCong (PartSame)

private theorem throw_ne {α : Type} (e : String) (a : α) : (throw e : V α) ≠ .ok a := fun h => by cases h

private theorem mapError_ok {ε α : Type} {x : Except ε α} {f : ε → ε} {a : α} (h : x.mapError f = .ok a) :
    x = .ok a := by
  cases x with
  | error e => cases h
  | ok a' => cases h; rfl

private theorem statement_circuit :
    Flock.Tags.find? _root_.Spot.OneStageZk.STATEMENT = some Flock.Tags.circuit := rfl

/-! ## The member's statement -/

/-- **A member's statement, where it passes, is `statementOf`'s at the member's files**: the circuit and public files
on the blobs, the member's registered reads, at the statement `verity/flock-circuit` (whose leaves are not plain), with
the member's population check at its slice. -/
theorem statement_ok {m : _root_.Spot.OneStageZk.Member} {slice : Json} {bs : Core.Blobs}
    {a : _root_.Spot.OneStageZk.StepAt} (h : m.statement slice bs = .ok a) :
    ∃ circuitB pubB rd, Flock.Tags.circuit.plainLeaves = false ∧
      bs.get m.own.circuit "circuit" = .ok circuitB ∧ bs.get m.own.pub "public file" = .ok pubB ∧
      _root_.Spot.OneStageZk.readsOf m.own.registered = .ok rd ∧
      a = _root_.Spot.OneStageZk.StepAt.ofOwn m.own Flock.Tags.circuit circuitB pubB bs rd
        (m.population slice pubB) := by
  unfold _root_.Spot.OneStageZk.Member.statement at h
  obtain ⟨a', ha', h⟩ := bind_ok h
  obtain ⟨⟨⟩, -, h⟩ := bind_ok h
  obtain rfl : a' = a := Except.ok.inj h
  unfold _root_.Spot.OneStageZk.statementOf at ha'
  obtain ⟨⟨⟩, -, h⟩ := bind_ok ha'
  split at h
  · exact absurd h (throw_ne _ _)
  rename_i tags0 htags
  obtain rfl : tags0 = Flock.Tags.circuit := Option.some.inj (htags.symm.trans statement_circuit)
  split at h
  · exact absurd h (throw_ne _ _)
  rename_i hpl
  split at h
  · exact absurd h (throw_ne _ _)
  obtain ⟨circuitB, hc, h⟩ := bind_ok h
  obtain ⟨pubB, hp, h⟩ := bind_ok h
  split at h
  · exact absurd h (throw_ne _ _)
  obtain ⟨rd, hrd, h⟩ := bind_ok h
  exact ⟨circuitB, pubB, rd, hpl, hc, hp, hrd, (Except.ok.inj h).symm⟩

/-! ## The accepted sessions -/

/-- **One session's step, where the run still accepts every session after it**: the run accepted every session before
it, and the session's record is UTF-8 text on the blobs, its sample fits its length word, and `verdictOf` at its record's
sample is `.ok`, after the coin-tree keys of the sessions before it. -/
theorem step_all {I : Flock.Witness.Inputs} {rd : _root_.Spot.OneStageZk.Reads} {bs : Core.Blobs}
    {scope ksFiles population : V Unit} {programSha : Option String} {want : Json}
    {r r' : _root_.Spot.OneStageZk.Run} {s : _root_.Spot.OneStageZk.RunFiles}
    (h : _root_.Spot.OneStageZk.step I rd bs scope ksFiles population programSha want r s = .ok r')
    (ha : r'.all = true) :
    r.all = true ∧ ∃ b record, bs.get s.record s!"session {s.name}'s record" = .ok b ∧
      String.fromUTF8? b = some record ∧
      _root_.Spot.OneStageZk.drawFitsWhy _root_.Spot.OneStageZk.WORD (Flock.Witness.recordDraw record) = none ∧
      _root_.Spot.OneStageZk.verdictOf I scope ksFiles population r.coinKeys record (Flock.Witness.recordDraw record)
        (_root_.Spot.OneStageZk.setupAt I rd (Flock.Witness.recordDraw record))
        (_root_.Spot.OneStageZk.filesOf bs s (_root_.Spot.OneStageZk.setupAt I rd (Flock.Witness.recordDraw record)))
        (_root_.Spot.OneStageZk.coinKeyOf record
          (_root_.Spot.OneStageZk.setupAt I rd (Flock.Witness.recordDraw record))) = .ok () := by
  unfold _root_.Spot.OneStageZk.step at h
  obtain ⟨b, hb, h⟩ := bind_ok h
  split at h
  · rename_i record hrec
    try dsimp only at h
    obtain ⟨⟨⟩, hfit, h⟩ := bind_ok h
    unfold _root_.Spot.OneStageZk.push at h
    try dsimp only at h
    split at h
    · obtain rfl := Except.ok.inj h
      simp at ha
    · rename_i hv
      split at h
      · exact absurd h (throw_ne _ _)
      obtain ⟨units, -, h⟩ := bind_ok h
      obtain rfl := Except.ok.inj h
      refine ⟨ha, b, record, hb, hrec, ?_, hv⟩
      cases hd : _root_.Spot.OneStageZk.drawFitsWhy _root_.Spot.OneStageZk.WORD (Flock.Witness.recordDraw record) with
      | none => rfl
      | some what =>
        rw [hd] at hfit
        cases hfit
  · exact absurd h (throw_ne _ _)

/-- **A session the statement's run accepted**: its record is UTF-8 text on the blobs, and `verdictOf` at the
statement `a` and the record's sample is `.ok`, after some sessions' coin-tree keys. -/
def SessionAccepted (a : _root_.Spot.OneStageZk.StepAt) (bs : Core.Blobs) (s : _root_.Spot.OneStageZk.RunFiles) :
    Prop :=
  ∃ coinKeys b record, bs.get s.record s!"session {s.name}'s record" = .ok b ∧ String.fromUTF8? b = some record ∧
    _root_.Spot.OneStageZk.verdictOf a.I a.scope a.ksFiles a.population coinKeys record
      (Flock.Witness.recordDraw record) (_root_.Spot.OneStageZk.setupAt a.I a.rd (Flock.Witness.recordDraw record))
      (_root_.Spot.OneStageZk.filesOf bs s (_root_.Spot.OneStageZk.setupAt a.I a.rd (Flock.Witness.recordDraw record)))
      (_root_.Spot.OneStageZk.coinKeyOf record
        (_root_.Spot.OneStageZk.setupAt a.I a.rd (Flock.Witness.recordDraw record))) = .ok ()

/-- **A statement's sessions, run in order, accepting every one**: the run before them accepted every session, and
so did each of them. -/
theorem foldl_all {a : _root_.Spot.OneStageZk.StepAt} {bs : Core.Blobs} :
    ∀ (l : List _root_.Spot.OneStageZk.RunFiles) (r r' : _root_.Spot.OneStageZk.Run),
      l.foldlM (fun r s => a.step bs r s) r = .ok r' → r'.all = true →
        r.all = true ∧ ∀ s ∈ l, SessionAccepted a bs s
  | [], r, r', h, ha => by
    obtain rfl : r = r' := Except.ok.inj h
    exact ⟨ha, fun _ hs => by simp at hs⟩
  | s :: l, r, r', h, ha => by
    rw [List.foldlM_cons] at h
    obtain ⟨r₁, h₁, h⟩ := bind_ok h
    obtain ⟨ha₁, hl⟩ := foldl_all l r₁ r' h ha
    have h₁' : _root_.Spot.OneStageZk.step a.I a.rd bs a.scope a.ksFiles a.population a.programSha a.want r s =
        .ok r₁ := h₁
    obtain ⟨har, b, record, hb, hrec, -, hv⟩ := step_all h₁' ha₁
    refine ⟨har, fun s' hs' => ?_⟩
    rcases List.mem_cons.1 hs' with rfl | hs'
    · exact ⟨r.coinKeys, b, record, hb, hrec, hv⟩
    · exact hl s' hs'

/-- **The per-member form's check of one member, accepting every session**: where `stepMember` on the member `m`'s
statement files `sf` returns a run that accepts every session, the member's statement passed at its files and
registered reads, and each of `sf`'s sessions was accepted there (`SessionAccepted`). -/
theorem stepMember_sessions {slice : _root_.Spot.OneStageZk.Member → Json} {bs : Core.Blobs}
    {r r' : _root_.Spot.OneStageZk.Run} {sf : _root_.Spot.OneStageZk.StatementFiles}
    {m : _root_.Spot.OneStageZk.Member}
    (h : _root_.Spot.OneStageZk.stepMember slice bs r (sf, m) = .ok r') (ha : r'.all = true) :
    ∃ circuitB pubB rd, Flock.Tags.circuit.plainLeaves = false ∧
      bs.get m.own.circuit "circuit" = .ok circuitB ∧ bs.get m.own.pub "public file" = .ok pubB ∧
      _root_.Spot.OneStageZk.readsOf m.own.registered = .ok rd ∧
      ∀ s ∈ sf.sessions.toList, SessionAccepted
        (_root_.Spot.OneStageZk.StepAt.ofOwn m.own Flock.Tags.circuit circuitB pubB bs rd
          (m.population (slice m) pubB)) bs s := by
  unfold _root_.Spot.OneStageZk.stepMember at h
  have h := mapError_ok h
  unfold _root_.Spot.OneStageZk.runFrom at h
  obtain ⟨a, hst, h⟩ := bind_ok h
  obtain ⟨circuitB, pubB, rd, hpl, hc, hp, hrd, rfl⟩ := statement_ok hst
  rw [← Array.foldlM_toList] at h
  exact ⟨circuitB, pubB, rd, hpl, hc, hp, hrd, (foldl_all _ r r' h ha).2⟩

/-! ## The session's statement as `flock-verify` reads it -/

/-- **A session statement's fields are a member's**: the statement `verity/flock-circuit` under `--zk` (no `--coins
os`), the member's circuit and public files `circuitB` and `pubB`, the blobs' lookup tables, one table, the member's
program, and no `--work`, `--work-table`, `--stratified`, `--closure` or `--public-inputs`. -/
structure MemberFields (w : FlockDeploy.Raw) (m : _root_.Spot.OneStageZk.Member) (circuitB pubB : ByteArray)
    (bs : Core.Blobs) : Prop where
  statement : w.statement = _root_.Spot.OneStageZk.STATEMENT
  circuit : w.circuit = circuitB
  publicFile : w.publicFile = pubB
  tables : w.tables = _root_.Spot.OneStageZk.tablesOf bs
  zk : w.zk = true
  osCoins : w.osCoins = false
  count : w.count = 1
  program : w.program = some m.own.program
  work : w.work = none
  workTable : w.workTable = none
  stratified : w.stratified = none
  closure : w.closure = none
  publicInputs : w.publicInputs = none

/-- **A session statement's JSON files read back as what the form read**: the verifier's parser reads its partition
object as the member's or as an object alike it (`PartSame`: what a printed object reads back as), its registered
reads as the member's (none where the member has none), and its sample, once normalized (`Flock.norm`), as the sample
of the session's record `record`. -/
structure ReadsBack (w : FlockDeploy.Raw) (m : _root_.Spot.OneStageZk.Member) (record : String) : Prop where
  partition : ∃ P, w.partition.mapM (FlockDeploy.jsonOf "partition") = .ok P ∧ PartSame P (some m.own.partition)
  registered : w.registered.mapM (FlockDeploy.jsonOf "registered reads") = .ok m.own.registered
  draw : ∃ D, w.draw.mapM (FlockDeploy.jsonOf "draw") = .ok D ∧ D.map Flock.norm = Flock.Witness.recordDraw record

/-- Registered reads that read back as the member's read as the verifier's registered reads `rd`. -/
theorem registered_of_reads {o : Option ByteArray} {j : Option Json}
    (hreg : o.mapM (FlockDeploy.jsonOf "registered reads") = .ok j) {rd : _root_.Spot.OneStageZk.Reads}
    (hrd : _root_.Spot.OneStageZk.readsOf j = .ok rd) :
    o.mapM (fun b => do Flock.Registered.ofJson (← FlockDeploy.jsonOf "registered reads" b)) =
      .ok rd.registered := by
  cases o with
  | none =>
    obtain rfl : none = j := Except.ok.inj hreg
    obtain rfl := Except.ok.inj hrd
    rfl
  | some b =>
    rw [Option.mapM_some] at hreg
    cases hj : FlockDeploy.jsonOf "registered reads" b with
    | error e => rw [hj] at hreg; cases hreg
    | ok j' =>
      rw [hj] at hreg
      obtain rfl : some j' = j := Except.ok.inj hreg
      unfold _root_.Spot.OneStageZk.readsOf at hrd
      obtain ⟨own, hown, hrd⟩ := bind_ok hrd
      obtain ⟨_, -, hrd⟩ := bind_ok hrd
      obtain ⟨_, -, hrd⟩ := bind_ok hrd
      obtain ⟨_, -, hrd⟩ := bind_ok hrd
      obtain ⟨_, -, hrd⟩ := bind_ok hrd
      obtain rfl := Except.ok.inj hrd
      rw [Option.mapM_some, hj]
      simp only [bind, Except.bind, hown]
      rfl

/-- **`flock-verify`'s reading of a member's session statement holds v2's inputs at the member's files**: where the
statement's fields are the member's and its JSON files read back, `FlockDeploy.Raw.ins` reads it as `si`, which matches
v2's inputs `inputsOf` at the member's files and its registered reads `rd` (`Record.SessionMatch`), with the record's
sample. -/
theorem ins_of_fields {w : FlockDeploy.Raw} {m : _root_.Spot.OneStageZk.Member} {circuitB pubB : ByteArray}
    {bs : Core.Blobs} {record : String} {rd : _root_.Spot.OneStageZk.Reads} (hf : MemberFields w m circuitB pubB bs)
    (hb : ReadsBack w m record) (hrd : _root_.Spot.OneStageZk.readsOf m.own.registered = .ok rd)
    (hpl : Flock.Tags.circuit.plainLeaves = false) :
    ∃ si, w.ins = .ok si ∧
      SessionMatch si (_root_.Spot.OneStageZk.inputsOf Flock.Tags.circuit m.own circuitB pubB bs) rd ∧
      si.read.draw = Flock.Witness.recordDraw record := by
  obtain ⟨D, hD, hDn⟩ := hb.draw
  obtain ⟨P, hP, hPs⟩ := hb.partition
  have hreg := registered_of_reads hb.registered hrd
  have hfind : Flock.Tags.byName? w.statement = some Flock.Tags.circuit := by
    rw [hf.statement]
    exact Flock.Tags.byName?_of_find? statement_circuit
  apply Exists.intro
  refine ⟨?hi, ?_, ?_⟩
  case hi =>
    unfold FlockDeploy.Raw.ins
    simp only [hfind, hf.zk, hf.osCoins, hpl, hD, hP, hf.workTable, hf.closure, hf.work, hreg,
      Option.mapM_none, Bool.true_and, Bool.false_eq_true, ↓reduceIte]
    rfl
  · exact
      { zk := by rfl
        stmtTags := rfl
        tags := rfl
        circuitFile := hf.circuit
        publicFile := hf.publicFile
        count := hf.count
        tables := hf.tables
        partition := hPs
        program := hf.program
        work := by rfl
        stratified := hf.stratified
        closure := by rfl
        publicInputs := hf.publicInputs
        registered := rfl
        readRegistered := rfl }
  · exact hDn

/-! ## The checks -/

variable {slice : _root_.Spot.OneStageZk.Member → Json} {bs : Core.Blobs} {r r' : _root_.Spot.OneStageZk.Run}
  {sf : _root_.Spot.OneStageZk.StatementFiles} {m : _root_.Spot.OneStageZk.Member}

/-- **A member's session statement, accepted by the per-member form, is in Flock's masked scope**: where `stepMember`
on the member `m`'s statement files `sf` returns a run that accepts every session, at a session `s` of `sf` whose record
on the blobs is `record`, a statement `w` whose fields are the member's (at the circuit and public files the blobs hold
for it), whose JSON files read back (`ReadsBack`) and whose bytes decode as `w` sets up in Flock's masked scope
(`SpotFlock.zkOk`), passes the deployed `--zk` scope check (`FlockDeploy.scopeCheck true`) and has hidden outputs
(`SpotFlock.hiddenOf`). -/
theorem memberSession_checks (h : _root_.Spot.OneStageZk.stepMember slice bs r (sf, m) = .ok r')
    (ha : r'.all = true) {s : _root_.Spot.OneStageZk.RunFiles} (hs : s ∈ sf.sessions.toList)
    {circuitB pubB b : ByteArray} {record : String}
    (hc : bs.get m.own.circuit "circuit" = .ok circuitB) (hp : bs.get m.own.pub "public file" = .ok pubB)
    (hb : bs.get s.record s!"session {s.name}'s record" = .ok b) (hu : String.fromUTF8? b = some record)
    {w : FlockDeploy.Raw} (hf : MemberFields w m circuitB pubB bs) (hr : ReadsBack w m record)
    (hdec : FlockDeploy.decode w.encode = some w) :
    _root_.SpotFlock.zkOk w.encode = true ∧ FlockDeploy.scopeCheck true w.encode = true ∧
      _root_.SpotFlock.hiddenOf w.encode = true := by
  obtain ⟨circuitB', pubB', rd, hpl, hc', hp', hrd, hall⟩ := stepMember_sessions h ha
  obtain rfl : circuitB' = circuitB := Except.ok.inj (hc'.symm.trans hc)
  obtain rfl : pubB' = pubB := Except.ok.inj (hp'.symm.trans hp)
  obtain ⟨coinKeys, b', record', hb', hu', hv⟩ := hall s hs
  obtain rfl : b' = b := Except.ok.inj (hb'.symm.trans hb)
  obtain rfl : record' = record := Option.some.inj (hu'.symm.trans hu)
  dsimp only [_root_.Spot.OneStageZk.StepAt.ofOwn] at hv
  unfold _root_.Spot.OneStageZk.verdictOf at hv
  obtain ⟨proofs, -, hv⟩ := bind_ok hv
  obtain ⟨_, -, hv⟩ := bind_ok hv
  obtain ⟨⟨spec, tabs, p0, o0⟩, hst, hv⟩ := bind_ok hv
  dsimp only at hv
  obtain ⟨_, -, hv⟩ := bind_ok hv
  obtain ⟨⟨⟩, hsc, hv⟩ := bind_ok hv
  obtain ⟨⟨⟩, hsz, -⟩ := bind_ok hv
  unfold _root_.Spot.OneStageZk.scopeOf at hsc
  try dsimp only at hsc
  obtain ⟨⟨⟩, hsc, -⟩ := bind_ok hsc
  obtain ⟨si, hins, hm, hdraw⟩ := ins_of_fields hf hr hrd hpl
  obtain ⟨sts, hsu, htabs⟩ := Verity.Sampled.Record.setup_of_setupAt hm hdraw hst
  have hz : si.read.zk = true := hm.zk
  have hinsOf : FlockDeploy.insOf true w.encode = some si := by
    unfold FlockDeploy.insOf
    rw [hdec]
    simp [hf.zk, hins, Except.toOption]
  have hset : FlockDeploy.setupOf true w.encode = some (si, spec, sts) := by
    unfold FlockDeploy.setupOf
    rw [hinsOf]
    simp [hsu, Except.toOption]
  have hread : FlockDeploy.readOf true w.encode = some si.read := by
    unfold FlockDeploy.readOf
    rw [hinsOf]
    rfl
  have hshape : sts.forM Flock.Zk.shapeOk = .ok () := Claims.Bridge.Custody.setup_shapeOk hsu hz
  obtain ⟨h0, h64, hp2, hrange⟩ := RecDeploy.read_setup_in (FlockDeploy.Ins.setup_read hsu) hm.publicInputs
  have hall' : ∀ st ∈ sts.toList, 25 ≤ st.m ∧ st.m ≤ 35 ∧ st.c.kLog ≤ 27 ∧ st.regions.size ≤ 1024 ∧
      spec.mPts ≤ 64 := by
    intro st hst
    obtain ⟨j, hj, rfl⟩ := Array.mem_iff_getElem.1 (Array.mem_toList_iff.1 hst)
    have h1 : Flock.Zk.shapeOk sts[j] = .ok () := Flock.Readable.array_forM_ok hshape j hj
    unfold Flock.Zk.shapeOk at h1
    obtain ⟨_, hq, -⟩ := bind_ok h1
    have hin : sts[j].c.kLog ≤ 27 ∧ sts[j].regions.size ≤ 1024 ∧ spec.mPts ≤ 64 := hrange _ hst
    exact ⟨(FlockDeploy.KS.queries0_range hq).1, (FlockDeploy.KS.queries0_range hq).2, hin⟩
  refine ⟨?_, ?_, ?_⟩
  · unfold _root_.SpotFlock.zkOk
    simp only [hset, Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true]
    exact ⟨⟨⟨h0, by omega⟩, by omega⟩, fun st hst => by
      obtain ⟨a, b, c, d, e⟩ := hall' st hst
      exact ⟨⟨⟨⟨a, b⟩, c⟩, d⟩, e⟩⟩
  · have htab : si.read.tablesOf = some sts := by
      unfold Flock.Readable.Read.tablesOf
      rw [FlockDeploy.Ins.setup_read hsu]
      simp [hz, hshape]
    have hpar : Flock.Readable.parseBytes (FlockDeploy.readOf true) w.encode =
        some ({ si.read with sts } : Flock.Readable.Parsed) := by
      unfold Flock.Readable.parseBytes Flock.Readable.Read.parse
      rw [hread, Option.bind_some, htab]
      rfl
    have hchk : si.read.check = .ok () := by
      rw [hm.read_check]
      exact hsc
    have hps : ({ si.read with sts } : Flock.Readable.Parsed).scope = .ok () := by
      unfold Flock.Readable.Parsed.scope
      change (si.read.check >>= fun _ =>
        Flock.ProvedScope.sizesOk si.read.zk ((sts.map Flock.Setup.ofCircuit).map Flock.Setup.m)) = .ok ()
      rw [hchk, hz]
      rw [htabs] at hsz
      exact hsz
    unfold FlockDeploy.scopeCheck Flock.Readable.scopeCheckOf
    rw [hpar]
    exact (Flock.Readable.isOk_iff _).2 hps
  · unfold _root_.SpotFlock.hiddenOf
    rw [hread]
    dsimp only
    rw [hm.read_tags]
    exact (Verity.Sampled.statement_zkTags statement_circuit _ _).1

/-- **A member's session statement, accepted by the per-member form, is in the scope the two-stage proof reads**
(`SpotFlock.sessionOk` at template `t`), where it has template `t`'s reference statement's shape (`hshape`). -/
theorem memberSession_ok (H : _root_.SpotFlock.Held) (t : ℕ)
    (h : _root_.Spot.OneStageZk.stepMember slice bs r (sf, m) = .ok r')
    (ha : r'.all = true) {s : _root_.Spot.OneStageZk.RunFiles} (hs : s ∈ sf.sessions.toList)
    {circuitB pubB b : ByteArray} {record : String}
    (hc : bs.get m.own.circuit "circuit" = .ok circuitB) (hp : bs.get m.own.pub "public file" = .ok pubB)
    (hb : bs.get s.record s!"session {s.name}'s record" = .ok b) (hu : String.fromUTF8? b = some record)
    {w : FlockDeploy.Raw} (hf : MemberFields w m circuitB pubB bs) (hr : ReadsBack w m record)
    (hdec : FlockDeploy.decode w.encode = some w)
    (hshape : _root_.SpotFlock.shapeOf w.encode = _root_.SpotFlock.shapeOf (H.xS₀ t)) :
    _root_.SpotFlock.sessionOk H t w.encode = true := by
  obtain ⟨h1, h2, h3⟩ := memberSession_checks h ha hs hc hp hb hu hf hr hdec
  unfold _root_.SpotFlock.sessionOk
  simp [h1, h2, h3, hshape]

end Verity.SampledTwo.MemberChecks
