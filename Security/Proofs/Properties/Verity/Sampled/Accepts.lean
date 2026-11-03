import Proofs.Properties.Verity.SampledRead
import Proofs.Flock.Recursive.Witness
import Proofs.Flock.Soundness.Refine.Walk

/-!
# Sampled proofs' acceptance: v2's verdict is flock-verify `--zk`'s on the session's files

`CouplingTarget` bounds the sampled verdict by `LiveAcceptsZKP` of the projected play, which judges the writer's
transcript with `AcceptsZKP`. v2 judges the session's files on the blobs sent with `flock-verify --zk` as
`SampledProofs.OneStageZk.run` runs it. They are one check on one pair of byte strings:

* **v2's session** (`buildSessionOf_ok`, `sessionAt_ok`, `setupAt_ok`, `setup_eq`): the session `sessionAt` sets up
  at a record's own draw is `Exec.setupOfP` at `--zk`'s inputs (`ZkExec.zkInputs`) of v2's (`toExec`), with no public
  inputs, and every table passes `Zk.shapeOk`.
* **v2's verdict** (`verdictOf_ok`, `push_ok`, `step_ok`, `step_accepts`, `foldlM_step`, `run_ok`, `checkBlobsAt_run`):
  where `checkBlobsAt` accepts, the evidence names one session and `run` accepted every one: for each, `Zk.verify`
  accepted its record and its proof files against that setup, and the record is canonical text.
* **The session's files** (`FilesAt`): the record's text, the UTF-8 of the blob the session's record names, and the
  proof files, the blobs its reps name in the `spec.repStreams` order of its setup, as v2's `step` and `filesOf` read
  them. They read the inputs only through `zkInputs` (`filesAt_congr`), and one session has one pair (`FilesAt.unique`).
* **v2's files are `A`'s** (`FilesOn`): v2's circuit and public file, which it finds on the blobs by its parameters'
  digests, and the lookup tables the blobs hold are `A`'s. `SetsUp` fixes the rest of v2's inputs (the statement, one
  table, the verifier's partition object and program, no work, stratified or closure input and no public inputs), so
  v2's `--zk` inputs are `A`'s (`zkInputs_toExec`).
* **The writer** (`WritesV2Files`): the transcript is v2's session's files on the blobs sent.

`session_of_verdict`: v2's verdict accepts its one session's files at `A`'s inputs. `acceptsZKP_of_verdict` and
`liveAccepts_of_verdict`: with `WritesV2Files`, `AcceptsZKP A.I A.own` of the writer's transcript, the projected play's
`LiveAcceptsZKP`.
-/

set_option Elab.async false

namespace Verity.Sampled

open Lean (Json)
open FlockSoundness.Discharge
open FlockSoundness.Discharge.ZkHidden (AcceptsZKP LiveAcceptsZKP)
open FlockSoundness.Discharge.Recursion.Witness (toExec setupOf_eq)
open FlockSoundness.Refine (ebind_ok throw_ne_ok)

/-! ## v2's session -/

theorem mapError_ok {ε ε' α : Type} {f : ε → ε'} {x : Except ε α} {a : α} (h : x.mapError f = .ok a) :
    x = .ok a := by
  cases x with
  | error e => cases h
  | ok b => cases h; rfl

/-- `buildSessionOf`'s setup is `Stmt.setupTables`'s, on either tag (`Stmt.setupTables_eq`). -/
theorem setup_if (tags : Flock.Tags) (cf pf : ByteArray) (coins : Flock.Coins) (count : ℕ)
    (tables : List (String × ByteArray)) (draw partition : Option Json) (program : Option ByteArray) :
    (if tags.hiddenOutputs then
        Flock.Stmt.setupTablesOf (Flock.ProvedScope.circuitOf tags cf tables) tags cf pf coins count tables draw
          partition program
      else Flock.Stmt.setupTables tags cf pf coins count tables draw partition program :
        Flock.V (Flock.SessionSpec × Array Flock.Stmt)) =
      Flock.Stmt.setupTables tags cf pf coins count tables draw partition program := by
  by_cases hh : tags.hiddenOutputs = true
  · rw [if_pos hh, Flock.Stmt.setupTables_eq, Flock.ProvedScope.circuitOf, if_pos hh]
  · rw [if_neg hh]

/-- **v2's `buildSessionOf` sets up what `Stmt.setupTables` sets up** on the circuit `ProvedScope.circuitOf` parses. -/
theorem buildSessionOf_ok {tags : Flock.Tags} {cf pf : ByteArray} {coins : Flock.Coins}
    {tables : List (String × ByteArray)} {count : ℕ} {draw partition : Option Json} {program : Option ByteArray}
    {registered : Option (Array Flock.Registered.Port)} {r : Flock.SessionSpec × Array Flock.Stmt}
    (h : _root_.SampledProofs.OneStageZk.buildSessionOf (Flock.ProvedScope.circuitOf tags cf tables) tags cf pf coins
      tables count draw partition program registered = .ok r) :
    Flock.Stmt.setupTables tags cf pf coins count tables draw partition program = .ok r := by
  unfold _root_.SampledProofs.OneStageZk.buildSessionOf at h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨r', hr, h⟩ := ebind_ok h
  rw [setup_if] at hr
  repeat' walk_step
  all_goals (cases h; exact hr)

/-- **The session `sessionAt` sets up** is `Stmt.setupTables`'s at `--zk`'s tags and coins, and every table passes
`Zk.shapeOk`. -/
theorem sessionAt_ok {W : Flock.Witness.Inputs} {rd : _root_.SampledProofs.OneStageZk.Reads} {draw : Option Json}
    {r : Flock.SessionSpec × Array Flock.Stmt} (h : _root_.SampledProofs.OneStageZk.sessionAt W rd draw = .ok r) :
    Flock.Stmt.setupTables (Flock.Witness.zkTags W) W.circuitFile W.publicFile (Flock.Witness.zkCoins W) W.count
        W.tables draw W.partition W.program = .ok r ∧
      r.2.forM Flock.Zk.shapeOk = .ok () := by
  unfold _root_.SampledProofs.OneStageZk.sessionAt at h
  try dsimp only at h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨r', hr, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨⟨⟩, hs, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  cases h
  exact ⟨buildSessionOf_ok hr, hs⟩

/-- **`setupAt`'s session is `sessionAt`'s**, its tables' verifier-side setups `Setup.ofCircuit` of its statements. -/
theorem setupAt_ok {W : Flock.Witness.Inputs} {rd : _root_.SampledProofs.OneStageZk.Reads} {draw : Option Json}
    {spec : Flock.SessionSpec} {tabs : Array Flock.Setup} {x : Json × Json}
    (h : _root_.SampledProofs.OneStageZk.setupAt W rd draw = .ok (spec, tabs, x)) :
    ∃ sts, _root_.SampledProofs.OneStageZk.sessionAt W rd draw = .ok (spec, sts) ∧
      tabs = sts.map Flock.Setup.ofCircuit := by
  unfold _root_.SampledProofs.OneStageZk.setupAt at h
  obtain ⟨⟨spec', sts⟩, h1, h⟩ := ebind_ok h
  try dsimp only at h
  obtain ⟨_, -, h⟩ := ebind_ok h
  cases h
  exact ⟨sts, mapError_ok h1, rfl⟩

/-- **v2's setup at a record's own draw is the event's** (`Recursion.Witness.setupOf_eq`): `Exec.setupOfP` at `--zk`'s
inputs of v2's, with no public inputs. -/
theorem setup_eq (W : Flock.Witness.Inputs) (record : String) :
    Flock.Stmt.setupTables (Flock.Witness.zkTags W) W.circuitFile W.publicFile (Flock.Witness.zkCoins W) W.count
        W.tables (Flock.Witness.recordDraw record) W.partition W.program =
      Exec.setupOfP (ZkExec.zkInputs (toExec W)) none record :=
  setupOf_eq W record

/-! ## v2's verdict -/

/-- **`verdictOf` accepts only** where the session's proof files and setup are some, `Zk.verify` accepts the record
and the proof files against the setup, and the record is canonical text. -/
theorem verdictOf_ok {W : Flock.Witness.Inputs} {scope ksFiles : Flock.V Unit} {coinKeys : Array String}
    {record : String} {draw : Option Json} {st : _root_.SampledProofs.OneStageZk.SetUp}
    {files : Flock.V (Array ByteArray)} {coinKey : Option String}
    (h : _root_.SampledProofs.OneStageZk.verdictOf W scope ksFiles coinKeys record draw st files coinKey = .ok ()) :
    ∃ proofs spec tabs x, files = .ok proofs ∧ st = .ok (spec, tabs, x) ∧
      Flock.Zk.verify spec tabs record proofs = .ok () ∧ Flock.RecordText.check record = .ok () := by
  unfold _root_.SampledProofs.OneStageZk.verdictOf at h
  obtain ⟨proofs, hf, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨⟨spec, tabs, x⟩, hst, h⟩ := ebind_ok h
  try dsimp only at h
  obtain ⟨⟨⟩, hz, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨⟨⟩, hrt, -⟩ := ebind_ok h
  exact ⟨proofs, spec, tabs, x, hf, hst, hz, hrt⟩

/-- **A run that still accepts all after `push` accepted all before it, and the verdict pushed accepted.** -/
theorem push_ok {W : Flock.Witness.Inputs} {rd : _root_.SampledProofs.OneStageZk.Reads}
    {programSha : Option String} {want : Json} {r r' : _root_.SampledProofs.OneStageZk.Run}
    {s : _root_.SampledProofs.OneStageZk.RunFiles} {record : String} {st : _root_.SampledProofs.OneStageZk.SetUp}
    {coinKey : Option String} {verdict : Flock.V Unit}
    (h : _root_.SampledProofs.OneStageZk.push W rd programSha want r s record st coinKey verdict = .ok r')
    (ha : r'.all = true) : r.all = true ∧ verdict = .ok () := by
  unfold _root_.SampledProofs.OneStageZk.push at h
  cases verdict with
  | error why =>
    dsimp only at h
    cases h
    first | cases ha | simp at ha
  | ok u =>
    cases u
    dsimp only at h
    split at h
    · cases h
    · obtain ⟨_, -, h⟩ := ebind_ok h
      cases h
      exact ⟨ha, rfl⟩

/-- **A run that still accepts all after `step` accepted all before it**, and the session's verdict accepted: its
record the UTF-8 of a blob it names, set up at its own draw. -/
theorem step_ok {W : Flock.Witness.Inputs} {rd : _root_.SampledProofs.OneStageZk.Reads} {bs : _root_.Core.Blobs}
    {scope ksFiles : Flock.V Unit} {programSha : Option String} {want : Json}
    {r r' : _root_.SampledProofs.OneStageZk.Run} {s : _root_.SampledProofs.OneStageZk.RunFiles}
    (h : _root_.SampledProofs.OneStageZk.step W rd bs scope ksFiles programSha want r s = .ok r')
    (ha : r'.all = true) :
    r.all = true ∧ ∃ b record, bs.get s.record s!"session {s.name}'s record" = .ok b ∧
      String.fromUTF8? b = some record ∧
      _root_.SampledProofs.OneStageZk.verdictOf W scope ksFiles r.coinKeys record (Flock.Witness.recordDraw record)
        (_root_.SampledProofs.OneStageZk.setupAt W rd (Flock.Witness.recordDraw record))
        (_root_.SampledProofs.OneStageZk.filesOf bs s
          (_root_.SampledProofs.OneStageZk.setupAt W rd (Flock.Witness.recordDraw record)))
        (_root_.SampledProofs.OneStageZk.coinKeyOf record
          (_root_.SampledProofs.OneStageZk.setupAt W rd (Flock.Witness.recordDraw record))) = .ok () := by
  unfold _root_.SampledProofs.OneStageZk.step at h
  obtain ⟨b, hb, h⟩ := ebind_ok h
  cases hu : String.fromUTF8? b with
  | none =>
    rw [hu] at h
    dsimp only at h
    cases h
  | some record =>
    rw [hu] at h
    dsimp only at h
    obtain ⟨hr, hv⟩ := push_ok h ha
    exact ⟨hr, b, record, hb, hu, hv⟩

/-! ## The session's files -/

/-- **`t` is the session `s`'s files on the blobs `bs`, as flock-verify `--zk` reads them at `I`**: the record's text
is the UTF-8 of the blob `s.record` names, and the proof files are the blobs `s.reps` names in the `spec.repStreams`
order of the session set up at the record's draw from `--zk`'s inputs (`ZkExec.zkInputs I`) with no public inputs, as
v2's `step` and `filesOf` read them. -/
def FilesAt (I : Exec.Inputs) (bs : _root_.Core.Blobs) (s : _root_.SampledProofs.OneStageZk.RunFiles)
    (t : Exec.Transcript) : Prop :=
  (∃ b, bs.get s.record s!"session {s.name}'s record" = .ok b ∧ String.fromUTF8? b = some t.record) ∧
    ∃ spec sts, Exec.setupOfP (ZkExec.zkInputs I) none t.record = .ok (spec, sts) ∧
      _root_.SampledProofs.OneStageZk.filesOf bs s (.ok (spec, #[], Json.null, Json.null)) = .ok t.proofs

/-- `filesOf` reads only the setup's spec. -/
theorem filesOf_ok (bs : _root_.Core.Blobs) (s : _root_.SampledProofs.OneStageZk.RunFiles)
    (spec : Flock.SessionSpec) (tabs tabs' : Array Flock.Setup) (x x' : Json × Json) :
    _root_.SampledProofs.OneStageZk.filesOf bs s (.ok (spec, tabs, x)) =
      _root_.SampledProofs.OneStageZk.filesOf bs s (.ok (spec, tabs', x')) := rfl

/-- **One session has one pair of files.** -/
theorem FilesAt.unique {I : Exec.Inputs} {bs : _root_.Core.Blobs} {s : _root_.SampledProofs.OneStageZk.RunFiles}
    {t t' : Exec.Transcript} (h : FilesAt I bs s t) (h' : FilesAt I bs s t') : t = t' := by
  obtain ⟨r, p⟩ := t
  obtain ⟨r', p'⟩ := t'
  obtain ⟨⟨b, hb, hu⟩, spec, sts, hs, hf⟩ := h
  obtain ⟨⟨b', hb', hu'⟩, spec', sts', hs', hf'⟩ := h'
  obtain rfl : b = b' := Except.ok.inj (hb.symm.trans hb')
  obtain rfl : r = r' := Option.some.inj (hu.symm.trans hu')
  obtain ⟨rfl, -⟩ : spec = spec' ∧ sts = sts' := Prod.mk.inj (Except.ok.inj (hs.symm.trans hs'))
  obtain rfl : p = p' := Except.ok.inj (hf.symm.trans hf')
  rfl

/-- The session's files read the inputs only through `--zk`'s. -/
theorem filesAt_congr {I I' : Exec.Inputs} (h : ZkExec.zkInputs I = ZkExec.zkInputs I') {bs : _root_.Core.Blobs}
    {s : _root_.SampledProofs.OneStageZk.RunFiles} {t : Exec.Transcript} : FilesAt I bs s t ↔ FilesAt I' bs s t := by
  unfold FilesAt
  rw [h]

/-- `AcceptsZKP` reads the inputs only through `--zk`'s. -/
theorem acceptsZKP_congr {I I' : Exec.Inputs} (h : ZkExec.zkInputs I = ZkExec.zkInputs I') {own : Option ByteArray}
    {t : Exec.Transcript} : AcceptsZKP I own t ↔ AcceptsZKP I' own t := by
  unfold AcceptsZKP
  rw [h]

/-- **A session `step` accepted is one flock-verify `--zk` accepts on its files**: where the run still accepts all
after it, the files (`FilesAt`) at v2's inputs are a transcript `AcceptsZKP` accepts with no public inputs, whose
record is canonical text. -/
theorem step_accepts {W : Flock.Witness.Inputs} {rd : _root_.SampledProofs.OneStageZk.Reads}
    {bs : _root_.Core.Blobs} {scope ksFiles : Flock.V Unit} {programSha : Option String} {want : Json}
    {r r' : _root_.SampledProofs.OneStageZk.Run} {s : _root_.SampledProofs.OneStageZk.RunFiles}
    (h : _root_.SampledProofs.OneStageZk.step W rd bs scope ksFiles programSha want r s = .ok r')
    (ha : r'.all = true) :
    r.all = true ∧ ∃ t, FilesAt (toExec W) bs s t ∧ AcceptsZKP (toExec W) none t ∧
      Flock.RecordText.check t.record = .ok () := by
  obtain ⟨hr, b, record, hb, hu, hv⟩ := step_ok h ha
  obtain ⟨proofs, spec, tabs, x, hf, hst, hz, hrt⟩ := verdictOf_ok hv
  obtain ⟨sts, hs, rfl⟩ := setupAt_ok hst
  obtain ⟨hset, hshape⟩ := sessionAt_ok hs
  rw [setup_eq] at hset
  rw [hst] at hf
  exact ⟨hr, ⟨record, proofs⟩, ⟨⟨b, hb, hu⟩, spec, sts, hset,
    (filesOf_ok bs s spec #[] _ (Json.null, Json.null) x).trans hf⟩, ⟨spec, sts, hset, hshape, hz⟩, hrt⟩

/-- **Every session of a fold of `step` that ends accepting all was accepted** (`step_accepts`). -/
theorem foldlM_step {W : Flock.Witness.Inputs} {rd : _root_.SampledProofs.OneStageZk.Reads}
    {bs : _root_.Core.Blobs} {scope ksFiles : Flock.V Unit} {programSha : Option String} {want : Json} :
    ∀ (l : List _root_.SampledProofs.OneStageZk.RunFiles) (r r' : _root_.SampledProofs.OneStageZk.Run),
      l.foldlM (_root_.SampledProofs.OneStageZk.step W rd bs scope ksFiles programSha want) r = .ok r' →
      r'.all = true → r.all = true ∧ ∀ s ∈ l, ∃ t, FilesAt (toExec W) bs s t ∧ AcceptsZKP (toExec W) none t ∧
        Flock.RecordText.check t.record = .ok () := by
  intro l
  induction l with
  | nil =>
    intro r r' h ha
    rw [List.foldlM_nil] at h
    obtain rfl : r = r' := Except.ok.inj h
    exact ⟨ha, fun _ hs => by simp at hs⟩
  | cons s l ih =>
    intro r r' h ha
    rw [List.foldlM_cons] at h
    obtain ⟨r₁, h₁, h⟩ := ebind_ok h
    obtain ⟨ha₁, hl⟩ := ih r₁ r' h ha
    obtain ⟨har, hs⟩ := step_accepts h₁ ha₁
    refine ⟨har, fun x hx => ?_⟩
    rcases List.mem_cons.1 hx with rfl | hx
    · exact hs
    · exact hl x hx

/-- **A run of v2's verifier that accepts all accepted every session**: the statement is known, the circuit and the
public file are blobs, and every session's files at v2's inputs (`inputsOf`) are a transcript flock-verify `--zk`
accepts, whose record is canonical text. -/
theorem run_ok {p : _root_.SampledProofs.OneStageZk.Own} {sessions : Array _root_.SampledProofs.OneStageZk.RunFiles}
    {pu : Option (Array ℕ)} {bs : _root_.Core.Blobs} {r : _root_.SampledProofs.OneStageZk.Run}
    (h : _root_.SampledProofs.OneStageZk.run p sessions pu bs = .ok r) (ha : r.all = true) :
    ∃ tags0 circuitB pubB, Flock.Tags.find? _root_.SampledProofs.OneStageZk.STATEMENT = some tags0 ∧
      bs.get p.circuit "circuit" = .ok circuitB ∧ bs.get p.pub "public file" = .ok pubB ∧
      ∀ s ∈ sessions, ∃ t,
        FilesAt (toExec (_root_.SampledProofs.OneStageZk.inputsOf tags0 p circuitB pubB bs)) bs s t ∧
          AcceptsZKP (toExec (_root_.SampledProofs.OneStageZk.inputsOf tags0 p circuitB pubB bs)) none t ∧
          Flock.RecordText.check t.record = .ok () := by
  unfold _root_.SampledProofs.OneStageZk.run at h
  cases ht : Flock.Tags.find? _root_.SampledProofs.OneStageZk.STATEMENT with
  | none =>
    rw [ht] at h
    dsimp only at h
    cases h
  | some tags0 =>
    rw [ht] at h
    dsimp only at h
    obtain ⟨_, -, h⟩ := ebind_ok h
    obtain ⟨_, -, h⟩ := ebind_ok h
    obtain ⟨circuitB, hc, h⟩ := ebind_ok h
    obtain ⟨pubB, hp, h⟩ := ebind_ok h
    try dsimp only at h
    obtain ⟨rd, -, h⟩ := ebind_ok h
    try dsimp only at h
    obtain ⟨r₁, hf, h⟩ := ebind_ok h
    have hr : r₁ = r := by
      try dsimp only at h
      repeat' walk_step
      all_goals (cases h; rfl)
    subst hr
    rw [← Array.foldlM_toList] at hf
    obtain ⟨-, hall⟩ := foldlM_step _ _ _ hf ha
    exact ⟨tags0, circuitB, pubB, ht, hc, hp, fun s hs => hall s (Array.mem_def.1 hs)⟩

/-- **Where v2's check at the coins accepts**, the evidence names one session, and the run of v2's verifier over it
on the blobs accepts all. -/
theorem checkBlobsAt_run {p : _root_.SampledProofs.OneStageZk.Params} {c : _root_.SampledProofs.OneStage.Commitment}
    {ξ : ByteArray} {e : _root_.SampledProofs.OneStageZk.Evidence} {bs : _root_.Core.Blobs}
    {pub : _root_.SampledProofs.OneStage.Public} {prof : _root_.Core.Integrity}
    (h : _root_.SampledProofs.OneStageZk.checkBlobsAt p c ξ e bs = .accept pub prof) :
    e.sessions.size = 1 ∧ ∃ r, _root_.SampledProofs.OneStageZk.run p.own e.sessions (some e.base.provedUnits) bs =
      .ok r ∧ r.all = true := by
  unfold _root_.SampledProofs.OneStageZk.checkBlobsAt at h
  try simp only [] at h
  split at h
  · cases h
  · rename_i r hr
    obtain ⟨_, h1, hr⟩ := ebind_ok hr
    obtain ⟨_, -, hr⟩ := ebind_ok hr
    obtain ⟨_, -, hr⟩ := ebind_ok hr
    obtain ⟨_, -, hr⟩ := ebind_ok hr
    obtain ⟨_, -, hr⟩ := ebind_ok hr
    obtain ⟨_, -, hr⟩ := ebind_ok hr
    refine ⟨?_, r, hr, ?_⟩
    · unfold _root_.SampledProofs.OneStageZk.oneSession at h1
      split at h1
      · exact beq_iff_eq.1 (by assumption)
      · cases h1
    · cases hca : _root_.SampledProofs.OneStage.checkAt p.base c (some ξ) { e.base with verdicts := r.out } <;>
        simp only [hca] at h
      all_goals first
        | cases h
        | exact Decidable.byContradiction fun hra => by rw [if_neg hra] at h; cases h

/-! ## v2's files and the writer -/

/-- **`--zk`'s inputs of v2's are `I`'s** where v2's inputs are `I`'s but the coins, and `I` has no work, stratified or
closure input. -/
theorem zkInputs_toExec {W : Flock.Witness.Inputs} {I : Exec.Inputs} (htags : W.tags = I.tags)
    (hcf : W.circuitFile = I.circuitFile) (hpf : W.publicFile = I.publicFile) (hcount : W.count = I.count)
    (htables : W.tables = I.tables) (hpart : W.partition = I.partition) (hprog : W.program = I.program)
    (hwork : I.work = none) (hstrat : I.stratified = none) (hclo : I.closure = none) :
    ZkExec.zkInputs (toExec W) = ZkExec.zkInputs I := by
  obtain ⟨tags, cf, pf, count, tables, partition, program⟩ := W
  obtain ⟨tags', cf', pf', coins', count', tables', partition', program', work', strat', clo'⟩ := I
  dsimp only at htags hcf hpf hcount htables hpart hprog hwork hstrat hclo
  subst htags hcf hpf hcount htables hpart hprog hwork hstrat hclo
  rfl

/-- **v2 reads `A`'s files on the blobs sent**: under v2's parameters in what the prover sent (`ParamsAt`), the circuit
and the public file v2 finds on the blobs by its parameters' digests are `A`'s, and so are the lookup tables the blobs
hold (`SampledProofs.OneStageZk.tablesOf`). -/
structure FilesOn (A : Live) (s : Verity.Sent) : Prop where
  circuit : ∀ p, ParamsAt s.params p → ∀ b,
    (_root_.Core.Blobs.ofArray s.blobs).get p.own.circuit "circuit" = .ok b → b = A.I.circuitFile
  publicFile : ∀ p, ParamsAt s.params p → ∀ b,
    (_root_.Core.Blobs.ofArray s.blobs).get p.own.pub "public file" = .ok b → b = A.I.publicFile
  tables : _root_.SampledProofs.OneStageZk.tablesOf (_root_.Core.Blobs.ofArray s.blobs) = A.I.tables

/-- **The writer writes v2's session files**: the transcript `t` is, at v2's case in what the prover sent, each of its
evidence's sessions' files on the blobs sent, as flock-verify `--zk` reads them at `A`'s inputs (`FilesAt`). -/
def WritesV2Files (A : Live) (s : Verity.Sent) (t : Exec.Transcript) : Prop :=
  ∀ p c ev, caseOf s = .ok (p, c, ev) → ∀ r ∈ ev.sessions, FilesAt A.I (_root_.Core.Blobs.ofArray s.blobs) r t

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- **v2's verdict accepts its one session's files at `A`'s inputs**: where composite `k` reads sampled proofs at v2,
`A` is the session v2 sets up on the parameters and v2 reads `A`'s files on the blobs sent, v2's case in what the
prover sent names one session, whose files on the blobs sent (`FilesAt` at `A`'s inputs) flock-verify `--zk` accepts at
`A`'s public inputs (`AcceptsZKP`), and whose record is canonical text. -/
theorem session_of_verdict {V : Vocab d α} {k : Verity.Registration.Composite} {A : Live} {θ : ByteArray}
    {F : Array ByteArray} {o : Verity.Exchange} (hk : ReadsV2 k) (hA : SetsUp V θ F A) (hθ : o.params = θ)
    (hv : V.Verdict k .sampled o) (hF : FilesOn A o.sent) :
    ∃ p c ev, caseOf o.sent = .ok (p, c, ev) ∧ ∃ s, ev.sessions = #[s] ∧ ∃ t,
      FilesAt A.I (_root_.Core.Blobs.ofArray o.sent.blobs) s t ∧ AcceptsZKP A.I A.own t ∧
        Flock.RecordText.check t.record = .ok () := by
  subst hθ
  obtain ⟨a, b, e, p, c, ev, ξ, dr, hcase, hacc⟩ := case_drew (verdict_case hk hv)
  have hcaseOf : caseOf o.sent = .ok (p, c, ev) := caseOf_of hv.2.1 hcase hacc.1 hacc.2.1 hacc.2.2.1
  obtain ⟨p', hp', hpart, hprog, -⟩ := hA.params
  have hpp : p' = p := paramsAt_case hv.2.1 hcase hacc hp'
  rw [hpp] at hp' hpart hprog
  obtain ⟨-, -, -, -, -, pub, prof, hck⟩ := hacc
  obtain ⟨hsize, r, hr, ha⟩ := checkBlobsAt_run hck
  obtain ⟨tags0, circuitB, pubB, ht0, hc, hpb, hall⟩ := run_ok hr ha
  obtain ⟨s, hs⟩ := Array.size_eq_one_iff.1 hsize
  have hsm : s ∈ ev.sessions := by rw [hs]; exact Array.mem_singleton_self s
  obtain ⟨t, hft, hat, hrt⟩ := hall s hsm
  have hz : ZkExec.zkInputs (toExec (_root_.SampledProofs.OneStageZk.inputsOf tags0 p.own circuitB pubB
      (_root_.Core.Blobs.ofArray o.sent.blobs))) = ZkExec.zkInputs A.I :=
    zkInputs_toExec (Option.some.inj (ht0.symm.trans hA.tags)) (hF.circuit p hp' circuitB hc)
      (hF.publicFile p hp' pubB hpb) hA.count.symm hF.tables hpart.symm hprog.symm hA.work hA.stratified
      hA.closure
  refine ⟨p, c, ev, hcaseOf, s, hs, t, (filesAt_congr hz).1 hft, ?_, hrt⟩
  rw [hA.own]
  exact (acceptsZKP_congr hz).1 hat

/-- **v2's verdict is flock-verify `--zk`'s acceptance of the writer's transcript**: where composite `k` reads sampled
proofs at v2, `A` is the session v2 sets up on the parameters, v2 reads `A`'s files on the blobs sent and the writer
writes v2's session files, the sampled verdict implies `AcceptsZKP A.I A.own` of the transcript. -/
theorem acceptsZKP_of_verdict {V : Vocab d α} {k : Verity.Registration.Composite} {A : Live} {θ : ByteArray}
    {F : Array ByteArray} {o : Verity.Exchange} {t : Exec.Transcript} (hk : ReadsV2 k) (hA : SetsUp V θ F A)
    (hθ : o.params = θ) (hv : V.Verdict k .sampled o) (hF : FilesOn A o.sent) (hW : WritesV2Files A o.sent t) :
    AcceptsZKP A.I A.own t := by
  obtain ⟨p, c, ev, hcaseOf, s, hs, t', hft, hat, -⟩ := session_of_verdict hk hA hθ hv hF
  obtain rfl := FilesAt.unique (hW p c ev hcaseOf s (by rw [hs]; exact Array.mem_singleton_self s)) hft
  exact hat

/-- **The projected play's acceptance**: at a play of the live audit whose transcript is v2's session files (the
projected play's `(R, S, A.wr S R out)` where the writer writes them), the sampled verdict implies `LiveAcceptsZKP` of
the play. -/
theorem liveAccepts_of_verdict {V : Vocab d α} {k : Verity.Registration.Composite} {A : Live} {θ : ByteArray}
    {F : Array ByteArray} {o : Verity.Exchange} {y : A.Reg × Finset (Fin A.n) × Exec.Transcript}
    (hk : ReadsV2 k) (hA : SetsUp V θ F A) (hθ : o.params = θ) (hv : V.Verdict k .sampled o)
    (hF : FilesOn A o.sent) (hW : WritesV2Files A o.sent y.2.2) : LiveAcceptsZKP A.I A.own y :=
  acceptsZKP_of_verdict hk hA hθ hv hF hW

end Verity.Sampled
