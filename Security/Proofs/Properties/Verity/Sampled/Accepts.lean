module

public import Proofs.Properties.Verity.Sampled.Moves
public import Proofs.Properties.Verity.Sampled.CoinMap
public import Proofs.Flock.Recursive.Witness
public import Proofs.Flock.Soundness.Refine.Walk
public import Proofs.Flock.Soundness.Refine.Lookups
public import Proofs.Flock.Soundness.Discharge.ZkExec.Verify
public import Proofs.Properties.NetworkCertifier.Check.Bytes

/-!
# Sampled proofs' acceptance: v2's verdict is flock-verify `--zk`'s on the session's files

The decoders' premise of the logged coupling (`Decodes`) asks that on the joint law's support, off the events at the
stops it is held off, an accepted play is beside an outcome flock-verify `--zk` accepts (`LiveAcceptsZKP`), which
judges the writer's transcript with `AcceptsZKP`.
v2 judges the session's files on the blobs sent with `flock-verify --zk` as `Spot.OneStageZk.run` runs it. They
are one check on one pair of byte strings, and the writer carries the acceptance to what it writes:

* **v2's session is `--zk`'s** (`buildSessionOf_ok`, `sessionAt_ok`, `setupAt_ok`, `setup_eq`, `setupAt_setupOfP`):
  the session `setupAt` sets up at a record's own sample is `Exec.setupOfP` at `--zk`'s inputs (`ZkExec.zkInputs`) of v2's
  (`toExec`), with no public inputs (`Recursion.Witness.setupOf_eq`), and every table passes `Zk.shapeOk`.
* **What v2 verified is accepted** (`v2Verified_accepts`): the record and the proof files v2's check verified
  (`V2Verified`) are a transcript `AcceptsZKP` accepts at v2's inputs, and the record decodes (`zkVerify_decode`).
* **v2's files are `A`'s** (`FilesOn`, `zkInputs_toExec`): v2's circuit and public file, which it finds on the blobs
  by its parameters' digests, are `A`'s, and the lookup tables the blobs hold extend `A`'s (`TablesExtend`). `SetsUp`
  fixes the rest of v2's inputs, so v2's `--zk` inputs are `A`'s but for the tables.
* **More lookup tables change nothing** (`setupOfP_back`, `acceptsZKP_back`): flock-verify reads a table only by
  `find?` at a name its circuit lists (`Refine.setupTables_back`), so where `A`'s circuit file parses with `A`'s tables,
  the session `--zk` sets up with tables that extend them is `A`'s, and so is what it accepts.
* **v2's files are the vocabulary's** (`DigestsAt`, `get_named`, `tablesOf_extends`, `filesOn_of`): the blobs sent are
  the committed files, then the evidence's, and `Blobs.get` returns the first whose SHA-512 is the digest, so where v2's
  parameters name the vocabulary's digests and the files are registered, v2 finds the committed files `namedFile`
  names, `A`'s by `SetsUp`; where the vocabulary reads the pinned MUFU tables off the committed files as v2 reads them
  off the blobs, v2 finds each table the committed files hold at the first blob with its pinned digest, a committed
  file, so v2's tables extend `A`'s. No SHA-512 collision enters: v2 and the vocabulary find a table at one blob.
* **v2's verdict at `A`** (`accepts_sent`): the record and the proof files v2 reads in what the prover sent
  (`sentRecord`, `sentProofs`) are a transcript `AcceptsZKP A.I A.own` accepts.
* **The writer** (`MovesOf`, `wr_accepts_of_verdict`): where the session's writer keeps an accepted transcript's view
  (`WritesView`, which `wrZP` does), what it writes for moves v2's files decode to, at a sample the record carries
  (`MovesOf`), is accepted.
* **The pointwise acceptance** (`play_ends`, `MovesReach`, `decodes_of`, `filesOn_play`, `decodes_of_digests`): a
  play of the period's game ends at its parameters and committed files, so `Decodes` holds wherever v2 reads `A`'s
  files on the accepted plays (at the vocabulary's digests and tables) and the outcomes on the joint law's support off
  the held-off events are the writer's at moves v2's files decode to (`MovesReach`, the decoders' and the coin map's).
-/

@[expose] public section

set_option Elab.async false

namespace Verity.Sampled

open Lean (Json)
open FlockSoundness.Discharge
open FlockSoundness.Discharge.ZkHidden (AcceptsZKP LiveAcceptsZKP)
open FlockSoundness.Discharge.Recursion.Witness (toExec setupOf_eq)
open FlockSoundness.Discharge.Tables (LiveOutJ)
open FlockSoundness.Refine (ebind_ok ptsOf OkLe TablesExtend setupTables_back extend_filterMap mufu_nodup)
open Claims.Bridge.CustodyZkP (repRounds)

/-! ## v2's session is `--zk`'s -/

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
  · rw [ite_eq_left hh, Flock.Stmt.setupTables_eq, Flock.ProvedScope.circuitOf, ite_eq_left hh]
  · rw [ite_eq_right hh]

/-- **v2's `buildSessionOf` sets up what `Stmt.setupTables` sets up** on the circuit `ProvedScope.circuitOf` parses. -/
theorem buildSessionOf_ok {tags : Flock.Tags} {cf pf : ByteArray} {coins : Flock.Coins}
    {tables : List (String × ByteArray)} {count : ℕ} {draw partition : Option Json} {program : Option ByteArray}
    {registered : Option (Array Flock.Registered.Port)} {r : Flock.SessionSpec × Array Flock.Stmt}
    (h : _root_.Spot.OneStageZk.buildSessionOf (Flock.ProvedScope.circuitOf tags cf tables) tags cf pf coins
      tables count draw partition program registered = .ok r) :
    Flock.Stmt.setupTables tags cf pf coins count tables draw partition program = .ok r := by
  unfold _root_.Spot.OneStageZk.buildSessionOf at h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨r', hr, h⟩ := ebind_ok h
  rw [setup_if] at hr
  repeat' walk_step
  all_goals (cases h; exact hr)

/-- **The session `sessionAt` sets up** is `Stmt.setupTables`'s at `--zk`'s tags and coins, and every table passes
`Zk.shapeOk`. -/
theorem sessionAt_ok {W : Flock.Witness.Inputs} {rd : _root_.Spot.OneStageZk.Reads} {draw : Option Json}
    {r : Flock.SessionSpec × Array Flock.Stmt} (h : _root_.Spot.OneStageZk.sessionAt W rd draw = .ok r) :
    Flock.Stmt.setupTables (Flock.Witness.zkTags W) W.circuitFile W.publicFile (Flock.Witness.zkCoins W) W.count
        W.tables draw W.partition W.program = .ok r ∧
      r.2.forM Flock.Zk.shapeOk = .ok () := by
  unfold _root_.Spot.OneStageZk.sessionAt at h
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
theorem setupAt_ok {W : Flock.Witness.Inputs} {rd : _root_.Spot.OneStageZk.Reads} {draw : Option Json}
    {spec : Flock.SessionSpec} {tabs : Array Flock.Setup} {x : Json × Json}
    (h : _root_.Spot.OneStageZk.setupAt W rd draw = .ok (spec, tabs, x)) :
    ∃ sts, _root_.Spot.OneStageZk.sessionAt W rd draw = .ok (spec, sts) ∧
      tabs = sts.map Flock.Setup.ofCircuit := by
  unfold _root_.Spot.OneStageZk.setupAt at h
  obtain ⟨⟨spec', sts⟩, h1, h⟩ := ebind_ok h
  try dsimp only at h
  obtain ⟨_, -, h⟩ := ebind_ok h
  cases h
  refine ⟨sts, ?_, rfl⟩
  cases hs : _root_.Spot.OneStageZk.sessionAt W rd draw with
  | error e => rw [hs] at h1; cases h1
  | ok b => rw [hs] at h1; cases h1; rfl

/-- **v2's setup at a record's own sample is the event's** (`Recursion.Witness.setupOf_eq`): `Exec.setupOfP` at `--zk`'s
inputs of v2's, with no public inputs. -/
theorem setup_eq (W : Flock.Witness.Inputs) (record : String) :
    Flock.Stmt.setupTables (Flock.Witness.zkTags W) W.circuitFile W.publicFile (Flock.Witness.zkCoins W) W.count
        W.tables (Flock.Witness.recordDraw record) W.partition W.program =
      Exec.setupOfP (ZkExec.zkInputs (toExec W)) none record :=
  setupOf_eq W record

/-- **v2's session at a record's own sample is `--zk`'s**: where `setupAt` sets up `(spec, tabs)`, `Exec.setupOfP` at
`--zk`'s inputs of v2's, with no public inputs, sets up `spec` with statements that each pass `Zk.shapeOk` and whose
verifier-side setups are `tabs`. -/
theorem setupAt_setupOfP {W : Flock.Witness.Inputs} {rd : _root_.Spot.OneStageZk.Reads} {record : String}
    {spec : Flock.SessionSpec} {tabs : Array Flock.Setup} {x : Json × Json}
    (h : _root_.Spot.OneStageZk.setupAt W rd (Flock.Witness.recordDraw record) = .ok (spec, tabs, x)) :
    ∃ sts, Exec.setupOfP (ZkExec.zkInputs (toExec W)) none record = .ok (spec, sts) ∧
      sts.forM Flock.Zk.shapeOk = .ok () ∧ tabs = sts.map Flock.Setup.ofCircuit := by
  obtain ⟨sts, hs, rfl⟩ := setupAt_ok h
  obtain ⟨hset, hshape⟩ := sessionAt_ok hs
  rw [setup_eq] at hset
  exact ⟨sts, hset, hshape, rfl⟩

/-! ## What v2 verified is accepted -/

/-- **`Zk.verify` accepts only a record that decodes** at the session's spec (`ZkExec.zk_verify_ok`). -/
theorem zkVerify_decode {spec : Flock.SessionSpec} {tabs : Array Flock.Setup} {record : String}
    {proofs : Array ByteArray} (h : Flock.Zk.verify spec tabs record proofs = .ok ()) :
    ∃ sess, Flock.Record.decode spec record = .ok sess := by
  obtain ⟨-, sess, -, hs, -⟩ := ZkExec.zk_verify_ok spec tabs record proofs h
  exact ⟨sess, hs⟩

/-- **What v2's check verified is a transcript flock-verify `--zk` accepts** at v2's inputs: the record and the proof
files of `V2Verified`, at the statement's tags and the circuit and public files v2 found on the blobs (`inputsOf`),
with no public inputs, at the session `Exec.setupOfP` sets up there. -/
theorem v2Verified_accepts {p : _root_.Spot.OneStageZk.Params} {e : _root_.Spot.OneStageZk.Evidence}
    {bs : _root_.Core.Blobs} {record : String} {spec : Flock.SessionSpec} {tabs : Array Flock.Setup}
    {rest : Json × Json} {proofs : Array ByteArray} (h : V2Verified p e bs record spec tabs rest proofs) :
    ∃ tags0 circuitB pubB sts, Flock.Tags.find? _root_.Spot.OneStageZk.STATEMENT = some tags0 ∧
      bs.get p.own.circuit "circuit" = .ok circuitB ∧ bs.get p.own.pub "public file" = .ok pubB ∧
      Exec.setupOfP (ZkExec.zkInputs (toExec (_root_.Spot.OneStageZk.inputsOf tags0 p.own circuitB pubB bs)))
        none record = .ok (spec, sts) ∧
      AcceptsZKP (toExec (_root_.Spot.OneStageZk.inputsOf tags0 p.own circuitB pubB bs)) none
        { record := record, proofs := proofs } := by
  obtain ⟨_, tags0, circuitB, pubB, _, -, ht, hc, hp, -, -, hst, -, hv⟩ := h
  obtain ⟨sts, hs, hshape, rfl⟩ := setupAt_setupOfP hst
  exact ⟨tags0, circuitB, pubB, sts, ht, hc, hp, hs, spec, sts, hs, hshape, hv⟩

/-! ## v2's files are `A`'s -/

/-- **`--zk`'s inputs of v2's are `I`'s at v2's lookup tables** where v2's inputs are `I`'s but the coins and the
tables, and `I` has no work, stratified or closure input. -/
theorem zkInputs_toExec {W : Flock.Witness.Inputs} {I : Exec.Inputs} (htags : W.tags = I.tags)
    (hcf : W.circuitFile = I.circuitFile) (hpf : W.publicFile = I.publicFile) (hcount : W.count = I.count)
    (hpart : W.partition = I.partition) (hprog : W.program = I.program)
    (hwork : I.work = none) (hstrat : I.stratified = none) (hclo : I.closure = none) :
    ZkExec.zkInputs (toExec W) = ZkExec.zkInputs { I with tables := W.tables } := by
  obtain ⟨tags, cf, pf, count, tables, partition, program⟩ := W
  obtain ⟨tags', cf', pf', coins', count', tables', partition', program', work', strat', clo'⟩ := I
  dsimp only at htags hcf hpf hcount hpart hprog hwork hstrat hclo
  subst htags hcf hpf hcount hpart hprog hwork hstrat hclo
  rfl

/-- `AcceptsZKP` reads the inputs only through `--zk`'s. -/
theorem acceptsZKP_congr {I I' : Exec.Inputs} (h : ZkExec.zkInputs I = ZkExec.zkInputs I') {own : Option ByteArray}
    {t : Exec.Transcript} : AcceptsZKP I own t ↔ AcceptsZKP I' own t := by
  unfold AcceptsZKP
  rw [h]

/-- **More lookup tables set up `I`'s session** (`Refine.setupTables_back`): where `T'` finds by name whatever `I`'s
tables find, `I`'s statement is untyped with at most one table and its circuit file parses at `--zk`'s tags, the
session `Exec.setupOfP` sets up at `--zk`'s inputs with the tables `T'` is the one it sets up with `I`'s. -/
theorem setupOfP_back {I : Exec.Inputs} {T' : List (String × ByteArray)} (hext : TablesExtend I.tables T')
    (hcount : I.count ≤ 1) (ht : I.tags.typed = false) {c : Flock.Circuit}
    (hp : Flock.HmOut.parse (ZkExec.zkTags I) I.circuitFile I.tables = .ok c) (own : Option ByteArray)
    (record : String) :
    OkLe (Exec.setupOfP (ZkExec.zkInputs { I with tables := T' }) own record)
      (Exec.setupOfP (ZkExec.zkInputs I) own record) :=
  setupTables_back hext hcount ((ZkExec.zkTags_typed I).trans ht) hp _ _ _ _ _ _ _ _ _

/-- **What flock-verify `--zk` accepts with more lookup tables it accepts with `I`'s**, under `setupOfP_back`'s
premises. -/
theorem acceptsZKP_back {I : Exec.Inputs} {T' : List (String × ByteArray)} (hext : TablesExtend I.tables T')
    (hcount : I.count ≤ 1) (ht : I.tags.typed = false) {c : Flock.Circuit}
    (hp : Flock.HmOut.parse (ZkExec.zkTags I) I.circuitFile I.tables = .ok c) {own : Option ByteArray}
    {t : Exec.Transcript} (h : AcceptsZKP { I with tables := T' } own t) : AcceptsZKP I own t := by
  obtain ⟨spec, sts, hs, hshape, hv⟩ := h
  exact ⟨spec, sts, setupOfP_back hext hcount ht hp own t.record _ hs, hshape, hv⟩

/-- **v2 reads `A`'s files on the blobs sent**: under v2's parameters in what the prover sent (`ParamsAt`), the circuit
and the public file v2 finds on the blobs by its parameters' digests are `A`'s, and the lookup tables the blobs hold
(`Spot.OneStageZk.tablesOf`) find by name whatever `A`'s find (`TablesExtend`). -/
structure FilesOn (A : Live) (s : Verity.Sent) : Prop where
  circuit : ∀ p, ParamsAt s.params p → ∀ b,
    (_root_.Core.Blobs.ofArray s.blobs).get p.own.circuit "circuit" = .ok b → b = A.I.circuitFile
  publicFile : ∀ p, ParamsAt s.params p → ∀ b,
    (_root_.Core.Blobs.ofArray s.blobs).get p.own.pub "public file" = .ok b → b = A.I.publicFile
  tables : TablesExtend A.I.tables (_root_.Spot.OneStageZk.tablesOf (_root_.Core.Blobs.ofArray s.blobs))

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-! ## v2's files are the vocabulary's -/

theorem getD_inj {a b : ByteArray} (ha : a.size = 64) (hb : b.size = 64)
    (h : (fun i : Fin 64 => a.data.getD i.val 0) = fun i => b.data.getD i.val 0) : a = b := by
  apply ByteArray.ext
  apply Array.ext (ha.trans hb.symm)
  intro i hi1 hi2
  have hi : i < 64 := by rw [← ha]; exact hi1
  have e := congrFun h ⟨i, hi⟩
  simp only [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hi1, Array.getElem?_eq_getElem hi2,
    Option.getD_some] at e
  exact e

theorem sha512_data (f : ByteArray) :
    Verity.sha512 f.data.toList = fun i => (Flock.Sha512.hash f).data.getD i.val 0 := by
  unfold Verity.sha512
  rw [Array.toArray_toList]

/-- **`Blobs.get` on the committed files, then the evidence's blobs, returns the first committed file `q` holds of**,
where `q` holds of a file exactly when its SHA-512 is the digest and one committed file has it. -/
theorem get_first {F B : Array ByteArray} {s what : String} {b : ByteArray} {q : ByteArray → Bool}
    (hq : ∀ dg, Flock.unhex s = .ok dg → dg.size = 64 → ∀ f, (Flock.Sha512.hash f == dg) = q f)
    (hreg : ∃ f ∈ F, q f = true) (hg : (_root_.Core.Blobs.ofArray (F ++ B)).get s what = .ok b) :
    b = (F.find? q).getD .empty := by
  unfold _root_.Core.Blobs.get at hg
  obtain ⟨dg, hu, hg⟩ := ebind_ok hg
  dsimp only at hg
  by_cases c0 : (dg.size != 64 || Flock.hexOf dg != s) = true
  · rw [ite_eq_left c0] at hg
    obtain ⟨_, h0, _⟩ := ebind_ok hg
    cases h0
  rw [ite_eq_right c0] at hg
  have hp := hq dg hu (by
    by_contra hne
    exact c0 (by simp [hne]))
  obtain ⟨f, hf, hfq⟩ := hreg
  obtain ⟨f0, hf0⟩ := Option.isSome_iff_exists.1 (Array.find?_isSome.2 ⟨f, hf, hfq⟩)
  have hfind : (_root_.Core.Blobs.ofArray (F ++ B)).entries.find? (fun x => x.1 == dg) =
      some (Flock.Sha512.hash f0, f0) := by
    show ((F ++ B).map fun b => (Flock.Sha512.hash b, b)).find? (fun x => x.1 == dg) = _
    simp only [Array.find?_map, Array.find?_append, Function.comp_def, hp, hf0, Option.some_or, Option.map_some]
  simp only [hfind] at hg
  have hb : f0 = b := by cases hg; rfl
  rw [← hb, hf0, Option.getD_some]

/-- **A blob v2 finds by a digest the committed files carry is the committed file the digest names**: the blobs are
the committed files `F`, then the evidence's `B`, and `Blobs.get` returns the first whose SHA-512 is the digest, as
`namedFile` does on `F`. -/
theorem get_named {F B : Array ByteArray} {s what : String} {h : Option Hash.Digest} {b : ByteArray}
    (hd : ∀ dg, Flock.unhex s = .ok dg → h = some fun i => dg.data.getD i.val 0)
    (hreg : ∃ f ∈ F, h = some (Verity.sha512 f.data.toList))
    (hg : (_root_.Core.Blobs.ofArray (F ++ B)).get s what = .ok b) : b = namedFile h F := by
  unfold namedFile
  refine get_first (fun dg hu h64 f => ?_) ?_ hg
  · try dsimp only
    rw [hd dg hu, sha512_data, Bool.eq_iff_iff, beq_bytes, decide_eq_true_iff, Option.some.injEq]
    constructor
    · rintro rfl
      rfl
    · intro e
      exact (getD_inj h64 (Properties.NetworkCertifier.Bytes.hash_size f) e).symm
  · obtain ⟨f, hf, hfh⟩ := hreg
    exact ⟨f, hf, decide_eq_true hfh⟩

/-- **The lookup tables on the committed files, then the evidence's blobs, extend those on the committed files**:
`tablesOf` finds each pinned MUFU table at the first blob whose SHA-512 is its pinned digest, which is a committed file
wherever one is. -/
theorem tablesOf_extends (F B : Array ByteArray) :
    TablesExtend (_root_.Spot.OneStageZk.tablesOf (_root_.Core.Blobs.ofArray F))
      (_root_.Spot.OneStageZk.tablesOf (_root_.Core.Blobs.ofArray (F ++ B))) := by
  refine extend_filterMap mufu_nodup (fun t x h => ?_) (fun t x h => ?_) (fun t x h => ?_)
  · obtain ⟨_, -, rfl⟩ := Option.map_eq_some_iff.1 h
    rfl
  · obtain ⟨_, -, rfl⟩ := Option.map_eq_some_iff.1 h
    rfl
  · obtain ⟨e, he, rfl⟩ := Option.map_eq_some_iff.1 h
    simp only [_root_.Core.Blobs.ofArray, Array.map_append, Array.find?_append] at he ⊢
    rw [he, Option.some_or, Option.map_some]

/-- **v2 reads `A`'s files** where its parameters name the vocabulary's digests, the vocabulary reads the lookup tables
as v2 does, the committed files are registered, and the blobs sent are the committed files, then the evidence's. -/
theorem filesOn_of {V : Vocab d α} {θ : ByteArray} {F B : Array ByteArray} {A : Live} {s : Verity.Sent}
    (hA : SetsUp V θ F A) (hD : DigestsAt V θ) (hreg : V.Registered θ F) (hθ : s.params = θ)
    (hs : s.blobs = F ++ B) : FilesOn A s where
  circuit p hp b hg := by
    rw [hs] at hg
    rw [hA.circuit]
    exact get_named (hD.circuit p (hθ ▸ hp)) hreg.1 hg
  publicFile p hp b hg := by
    rw [hs] at hg
    rw [hA.publicFile]
    exact get_named (hD.publicFile p (hθ ▸ hp)) hreg.2 hg
  tables := by
    have h : A.I.tables = _root_.Spot.OneStageZk.tablesOf (_root_.Core.Blobs.ofArray F) :=
      (congrArg Verity.FileRead.tables hA.read).symm.trans (hD.tables F)
    rw [h, hs]
    exact tablesOf_extends F B

/-! ## v2's verdict at `A` -/

/-- **v2's verdict is flock-verify `--zk`'s at `A` on the session's files**: where `A` is the session v2 sets up on
the parameters and v2 reads `A`'s files on the blobs sent, the record and the proof files v2 reads in what the prover
sent (`sentRecord`, `sentProofs`) are a transcript `AcceptsZKP A.I A.own` accepts, at the session `Exec.setupOfP` sets
up at `A`'s inputs. -/
theorem accepts_sent {V : Vocab d α} {k : Verity.Registration.Composite} {A : Live} {θ : ByteArray}
    {F : Array ByteArray} {o : Verity.Exchange} (hA : SetsUp V θ F A) (hθ : o.params = θ)
    (hv : V.Verdict k .sampled o) (hF : FilesOn A o.sent) :
    ∃ record proofs spec sts, sentRecord o.sent = some record ∧ sentProofs o.sent = some proofs ∧
      Exec.setupOfP (ZkExec.zkInputs A.I) A.own record = .ok (spec, sts) ∧
      AcceptsZKP A.I A.own { record := record, proofs := proofs } := by
  subst hθ
  obtain ⟨p', hp', hpart, hprog, -⟩ := hA.params
  obtain ⟨a, b, e, p, c, ev, ξ, dr, hcase, hacc⟩ := case_drew hv.2.1 hp' (verdict_case hv)
  have hpp : p' = p := paramsAt_case hv.2.1 hcase hacc hp'
  rw [hpp] at hp' hpart hprog
  obtain ⟨record, spec, tabs, rest, proofs, -, hrec, hpr, hver⟩ := sent_verified hv.2.1 hcase hacc
  obtain ⟨tags0, circuitB, pubB, sts, ht0, hc, hpb, hs, hat⟩ := v2Verified_accepts hver
  have hz := zkInputs_toExec (W := _root_.Spot.OneStageZk.inputsOf tags0 p.own circuitB pubB
      (_root_.Core.Blobs.ofArray o.sent.blobs)) (Option.some.inj (ht0.symm.trans hA.tags))
    (hF.circuit p hp' circuitB hc) (hF.publicFile p hp' pubB hpb) hA.count.symm hpart.symm hprog.symm hA.work
    hA.stratified hA.closure
  rw [hz] at hs
  rw [hA.own]
  have hp := Flock.Desc.parse_flat A.hD
  exact ⟨record, proofs, spec, sts, hrec, hpr, setupOfP_back hF.tables hA.count.le A.ht hp none record _ hs,
    acceptsZKP_back hF.tables hA.count.le A.ht hp ((acceptsZKP_congr hz).1 hat)⟩

/-! ## The writer -/

/-- **The moves of `out` are those v2's files decode to** in what the prover sent `s`, at the sample `S`: v2's record
carries the sample `A.dj S`, and `out`'s roots, link points and rounds are those of the session the record decodes to at
the session `--zk` sets up at `A`'s inputs, and its proof files are v2's. -/
def MovesOf (A : Live) (s : Verity.Sent) (S : Finset (Fin A.n)) (R : A.Reg) (out : LiveOutJ (A.J S R) A.mP) : Prop :=
  ∀ record proofs spec sts sess, sentRecord s = some record → sentProofs s = some proofs →
    Exec.setupOfP (ZkExec.zkInputs A.I) A.own record = .ok (spec, sts) →
    Flock.Record.decode spec record = .ok sess →
    Exec.recordDraw record = some (A.dj S) ∧ out.rootB = sess.rootB ∧ out.pts = ptsOf A.mP sess.points ∧
      (∀ j r, out.reps j r = repRounds spec sess j.val r.val) ∧ out.proofs = proofs

/-- **v2's verdict is flock-verify `--zk`'s acceptance of what the writer writes**: where `A` is the session v2 sets up
on the parameters, v2 reads `A`'s files on the blobs sent and the session's writer keeps an accepted transcript's view
(`WritesView`), the sampled verdict implies `AcceptsZKP A.I A.own` of what the writer writes for moves v2's files
decode to (`MovesOf`). -/
theorem wr_accepts_of_verdict {V : Vocab d α} {k : Verity.Registration.Composite} {A : Live} {θ : ByteArray}
    {F : Array ByteArray} {o : Verity.Exchange} (hA : SetsUp V θ F A) (hθ : o.params = θ)
    (hv : V.Verdict k .sampled o) (hF : FilesOn A o.sent) (hW : WritesView A) {S : Finset (Fin A.n)} {R : A.Reg}
    {out : LiveOutJ (A.J S R) A.mP} (hM : MovesOf A o.sent S R out) : AcceptsZKP A.I A.own (A.wr S R out) := by
  obtain ⟨record, proofs, spec, sts, hrec, hpr, hs, hat⟩ := accepts_sent hA hθ hv hF
  have hd : ∃ sess, Flock.Record.decode spec record = .ok sess := by
    obtain ⟨spec', sts', hs', -, hver⟩ := hat
    obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Except.ok.inj (hs.symm.trans hs'))
    exact zkVerify_decode hver
  obtain ⟨sess, hd⟩ := hd
  obtain ⟨hdr, hB, hP, hR, hPf⟩ := hM record proofs spec sts sess hrec hpr hs hd
  exact wr_accepts hW hat hdr hs hd hB hP hR hPf

/-! ## The pointwise acceptance -/

/-- A tape's play reaches its output. -/
theorem run_reaches {β : Type} : ∀ (g : Verity.Game β) (s : Verity.Game.Strategy g)
    (x : Verity.Game.CoinTape g s), Verity.Game.PlayReaches g s (Verity.Game.CoinTape.run g s x)
  | .ret _, _, _ => rfl
  | .send _ k, s, x => run_reaches (k s.1) s.2 x
  | @Verity.Game.coin _ _ _ _ k, s, x => ⟨x.1, run_reaches (k x.1) (s x.1) (x.2 x.1)⟩

/-- **A play of the period's game ends at its parameters and its committed files.** -/
theorem play_ends (V : Vocab d α) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) (x : Verity.Game.CoinTape (V.game θ F c) σ) :
    (Verity.Game.CoinTape.run (V.game θ F c) σ x).params = θ ∧
      (Verity.Game.CoinTape.run (V.game θ F c) σ x).files = F := by
  obtain ⟨log, s, h⟩ := Verity.Game.PlayReaches.of_bind (f := finish (V.signer θ) θ F c σ.1)
    (releases (frames (V.families θ)) (V.families θ) []) σ.2 (run_reaches (V.game θ F c) σ x)
  obtain ⟨ρ, ho⟩ : ∃ ρ : Fin (V.signer θ).draws → Bool, Verity.Game.CoinTape.run (V.game θ F c) σ x =
      ⟨θ, F, c, σ.1, log, s.1, List.ofFn ρ, (V.signer θ).signatures σ.1 log s.1 (List.ofFn ρ), (s.2 ρ).1.1,
        (s.2 ρ).1.2⟩ := h
  rw [ho]
  exact ⟨rfl, rfl⟩

/-- **The outcomes on the joint law's support are the writer's at moves v2's files decode to** (the decoders' and the
coin map's premise): on the joint law's support off the held-off events `BL` and `B`, beside a tape whose play's sampled
verdict at `k` accepts, that misses at no round and not at the record, and whose play `P` holds of, the outcome is the
writer's at a sample `S`, a registration `R` and an outcome `out` of the live session whose moves are those v2's files
decode to in what the prover sent (`MovesOf`). -/
def MovesReach (V : Vocab d α) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (k : Verity.Registration.Composite)
    (P : Verity.Exchange → Prop) (A : Live) (rd : Reading A.rounds) (σ : Verity.Game.Strategy (V.game θ F c))
    (D : Decoders A.rounds A.Reg) (M : CoinMap (V.game θ F c) σ A.Ω A.rounds A.mP) (BL : LinkHold A.mP)
    (B : RoundHold) : Prop :=
  ∀ o, ReachesOff V θ F c A rd σ D M BL B o →
    V.Verdict k .sampled (Verity.Game.CoinTape.run (V.game θ F c) σ o.2) ∧ NoMiss V θ F c rd σ o.2 ∧
      P (Verity.Game.CoinTape.run (V.game θ F c) σ o.2) →
    ∃ (S : Finset (Fin A.n)) (R : A.Reg) (out : LiveOutJ (A.J S R) A.mP), o.1 = (R, S, A.wr S R out) ∧
      MovesOf A (Verity.Game.CoinTape.run (V.game θ F c) σ o.2).sent S R out

/-- **The decoders' premise, pointwise** (`Decodes`): where `A` is the session v2 sets up on the parameters and the
committed files, v2 reads `A`'s files on the blobs of each accepted play,
the session's writer keeps an accepted transcript's view (`WritesView`) and the outcomes on the joint law's support off
`BL` and `B` are the writer's at moves v2's files decode to (`MovesReach`), every accepted play there that misses
nowhere and that `P` holds of is beside an outcome flock-verify `--zk` accepts at `A.own`. -/
theorem decodes_of {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray} {c : ByteArray}
    {k : Verity.Registration.Composite} {P : Verity.Exchange → Prop} {A : Live} {rd : Reading A.rounds}
    {σ : Verity.Game.Strategy (V.game θ F c)} {D : Decoders A.rounds A.Reg}
    {M : CoinMap (V.game θ F c) σ A.Ω A.rounds A.mP} {BL : LinkHold A.mP} {B : RoundHold} (hA : SetsUp V θ F A)
    (hF : ∀ x, V.Verdict k .sampled (Verity.Game.CoinTape.run (V.game θ F c) σ x) →
      FilesOn A (Verity.Game.CoinTape.run (V.game θ F c) σ x).sent)
    (hW : WritesView A) (hM : MovesReach V θ F c k P A rd σ D M BL B) : Decodes V θ F c k P A rd σ D M BL B := by
  intro o ho hacc
  obtain ⟨S, R, out, h1, hmo⟩ := hM o ho hacc
  rw [h1]
  exact wr_accepts_of_verdict hA (play_ends V θ F c σ o.2).1 hacc.1 (hF o.2 hacc.1) hW hmo

/-- **v2 reads `A`'s files on an accepted play**: the play ends at its parameters and committed files, the verdict
holds the files registered, and the blobs sent are the committed files, then the evidence's. -/
theorem filesOn_play {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray} {c : ByteArray}
    {k : Verity.Registration.Composite} {A : Live} {σ : Verity.Game.Strategy (V.game θ F c)}
    (hA : SetsUp V θ F A) (hD : DigestsAt V θ) {x : Verity.Game.CoinTape (V.game θ F c) σ}
    (hv : V.Verdict k .sampled (Verity.Game.CoinTape.run (V.game θ F c) σ x)) :
    FilesOn A (Verity.Game.CoinTape.run (V.game θ F c) σ x).sent := by
  obtain ⟨hp, hf⟩ := play_ends V θ F c σ x
  have hreg : V.Registered θ F := by
    have h := hv.1
    rwa [hp, hf] at h
  exact filesOn_of hA hD hreg hp (B := (Verity.Game.CoinTape.run (V.game θ F c) σ x).blobs)
    (by show (Verity.Game.CoinTape.run (V.game θ F c) σ x).files ++ _ = _; rw [hf])

/-- **The decoders' premise, pointwise, at the vocabulary's digests** (`decodes_of` with `filesOn_play`): where v2's
parameters name the vocabulary's digests and the vocabulary reads the lookup tables as v2 does (`DigestsAt`), `Decodes`
holds wherever the writer keeps an accepted transcript's view and the outcomes on the joint law's support off the
held-off events are the writer's at moves v2's files decode to. -/
theorem decodes_of_digests {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray} {c : ByteArray}
    {k : Verity.Registration.Composite} {P : Verity.Exchange → Prop} {A : Live} {rd : Reading A.rounds}
    {σ : Verity.Game.Strategy (V.game θ F c)} {D : Decoders A.rounds A.Reg}
    {M : CoinMap (V.game θ F c) σ A.Ω A.rounds A.mP} {BL : LinkHold A.mP} {B : RoundHold} (hA : SetsUp V θ F A)
    (hD : DigestsAt V θ) (hW : WritesView A) (hM : MovesReach V θ F c k P A rd σ D M BL B) :
    Decodes V θ F c k P A rd σ D M BL B :=
  decodes_of hA (fun _ hv => filesOn_play hA hD hv) hW hM

end Verity.Sampled
