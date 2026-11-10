module

public import Proofs.Properties.Verity.Sampled.Record.Accepts
public import Proofs.Properties.Verity.Sampled.Moves

/-!
# One-stage v2's acceptance over several sessions is their verdicts, in order

`Spot.OneStageZk.run` folds `Spot.OneStageZk.step` over the evidence's sessions, each a `flock-verify verify --zk` at
the session's own record's sample, carrying a `Run` from one to the next: the coin-tree keys the earlier sessions
used, the verdicts so far, the units the accepted ones proved and whether every one accepted. Where the last run
accepted every session, each session's verdict was `.ok`, in order:

* **What one session's `step` reads of its statement** (`StepAt`): the inputs, the verifier's registered reads, the
  scope, files and population checks, and the program's digest against the partition object's. A window whose members
  have their own statements has one per member; v2's `run` on one statement has the same one at every session.
* **One session's verdict after the run `r`** (`SessionVerdict`): its record is a blob, UTF-8; it is set up at its
  record's own sample; its proof files are the blobs it names; its coin-tree key is fresh among the keys the earlier
  sessions used (`r.coinKeys`); `Flock.Zk.verify` accepts them; and the whole of `verdictOf` is `.ok`, with the units
  its record proved.
* **The sessions' verdicts in order** (`Verdicts`): the run after each session is the run before the next, its coin
  keys and proved units grown by the session's.
* **`verdicts_of_steps`**: a fold of `step` over the sessions that ends accepting every session is their verdicts in
  order. **`run_verdicts`**: so is v2's `run` where it accepted, at the parameters' one statement, with the proved
  units held to the evidence's (`provedHeld`).
-/

@[expose] public section

namespace Verity.Sampled

open Lean (Json)

/-- **What `Spot.OneStageZk.step` reads of one session's statement**: its inputs, the verifier's registered reads, the
scope, files and population checks its verdict runs, and its program's digest against the partition object's
program. -/
structure StepAt where
  I : Flock.Witness.Inputs
  rd : _root_.Spot.OneStageZk.Reads
  scope : Flock.V Unit
  ksFiles : Flock.V Unit
  population : Flock.V Unit
  programSha : Option String
  want : Json

/-- One session's `step` at the statement `a`, on the blobs `bs`, after the run `r`. -/
def StepAt.step (a : StepAt) (bs : _root_.Core.Blobs) (r : _root_.Spot.OneStageZk.Run)
    (s : _root_.Spot.OneStageZk.RunFiles) : Flock.V _root_.Spot.OneStageZk.Run :=
  _root_.Spot.OneStageZk.step a.I a.rd bs a.scope a.ksFiles a.population a.programSha a.want r s

/-- **One session's verdict at the statement `a`, after the earlier sessions' coin-tree keys `keys`**: the session's
record is a blob, as UTF-8 `record`; v2 sets it up at its record's own sample as `(spec, tabs, rest)`; its proof files
are the blobs it names, `proofs`; its coin-tree key, where its setup flips coins from a coin tree, is none of `keys`;
`Flock.Zk.verify` accepts the record and the proof files; `verdictOf` is `.ok` on them; and its record proves the
units `units`. -/
structure SessionVerdict (a : StepAt) (bs : _root_.Core.Blobs) (keys : Array String)
    (s : _root_.Spot.OneStageZk.RunFiles) (record : String) (spec : Flock.SessionSpec) (tabs : Array Flock.Setup)
    (rest : Json × Json) (proofs : Array ByteArray) (units : Array ℕ) : Prop where
  blob : ∃ B, bs.get s.record s!"session {s.name}'s record" = .ok B ∧ String.fromUTF8? B = some record
  setup : _root_.Spot.OneStageZk.setupAt a.I a.rd (Flock.Witness.recordDraw record) = .ok (spec, tabs, rest)
  files : _root_.Spot.OneStageZk.filesOf bs s (.ok (spec, tabs, rest)) = .ok proofs
  fresh : ∀ key, _root_.Spot.OneStageZk.coinKeyOf record (.ok (spec, tabs, rest)) = some key →
    Flock.CoinTree.checkFresh keys key = .ok ()
  verify : Flock.Zk.verify spec tabs record proofs = .ok ()
  verdict : _root_.Spot.OneStageZk.verdictOf a.I a.scope a.ksFiles a.population keys record
    (Flock.Witness.recordDraw record) (.ok (spec, tabs, rest)) (.ok proofs)
    (_root_.Spot.OneStageZk.coinKeyOf record (.ok (spec, tabs, rest))) = .ok ()
  program : a.programSha.map Json.str = some a.want
  units : _root_.Spot.OneStageZk.provedUnits a.I.publicFile record = .ok units

/-- The run after an accepted session: the session's coin-tree key among the keys, its verdict of record pushed, and
the units its record proved appended (`Spot.OneStageZk.push` at an `.ok` verdict). -/
def ranOn (a : StepAt) (r : _root_.Spot.OneStageZk.Run) (s : _root_.Spot.OneStageZk.RunFiles) (record : String)
    (spec : Flock.SessionSpec) (tabs : Array Flock.Setup) (rest : Json × Json) (units : Array ℕ) :
    _root_.Spot.OneStageZk.Run where
  coinKeys := match _root_.Spot.OneStageZk.coinKeyOf record (.ok (spec, tabs, rest)) with
    | some key => r.coinKeys.push key
    | none => r.coinKeys
  out := r.out.push (_root_.Spot.OneStageZk.verdictJson a.rd a.programSha s (.ok (spec, tabs, rest)) true "" units)
  proved := r.proved ++ units
  all := r.all

/-- **The sessions' verdicts, in order**, from the run `r` to the run `r'` on the blobs `bs`: each session, at its
statement, has its verdict after the keys of the run before it (`SessionVerdict`), and the run after it (`ranOn`) is
the run before the next. -/
inductive Verdicts (bs : _root_.Core.Blobs) :
    _root_.Spot.OneStageZk.Run → List (StepAt × _root_.Spot.OneStageZk.RunFiles) → _root_.Spot.OneStageZk.Run → Prop
  | nil (r : _root_.Spot.OneStageZk.Run) : Verdicts bs r [] r
  | cons {r r' : _root_.Spot.OneStageZk.Run} {a : StepAt} {s : _root_.Spot.OneStageZk.RunFiles}
      {l : List (StepAt × _root_.Spot.OneStageZk.RunFiles)} (record : String) (spec : Flock.SessionSpec)
      (tabs : Array Flock.Setup) (rest : Json × Json) (proofs : Array ByteArray) (units : Array ℕ) :
      SessionVerdict a bs r.coinKeys s record spec tabs rest proofs units →
      Verdicts bs (ranOn a r s record spec tabs rest units) l r' → Verdicts bs r ((a, s) :: l) r'

/-- A session's `step` keeps a run that didn't accept every session so: a step's run accepts every session only where
the run before it did. -/
theorem step_all {a : StepAt} {bs : _root_.Core.Blobs} {r r' : _root_.Spot.OneStageZk.Run}
    {s : _root_.Spot.OneStageZk.RunFiles} (h : a.step bs r s = .ok r') (hall : r'.all = true) : r.all = true := by
  sorry

/-- **One session's `step` that ends accepting every session is the session's verdict**, and its run the run after
it. -/
theorem verdict_of_step {a : StepAt} {bs : _root_.Core.Blobs} {r r' : _root_.Spot.OneStageZk.Run}
    {s : _root_.Spot.OneStageZk.RunFiles} (h : a.step bs r s = .ok r') (hall : r'.all = true) :
    ∃ record spec tabs rest proofs units, SessionVerdict a bs r.coinKeys s record spec tabs rest proofs units ∧
      r' = ranOn a r s record spec tabs rest units := by
  sorry

/-- **v2's acceptance over several sessions is their verdicts in order**: a fold of `step` over the sessions, each at
its statement, from the run `r`, that ends at a run accepting every session, is the sessions' verdicts in order. -/
theorem verdicts_of_steps {bs : _root_.Core.Blobs} {l : List (StepAt × _root_.Spot.OneStageZk.RunFiles)}
    {r r' : _root_.Spot.OneStageZk.Run} (h : l.foldlM (fun r x => x.1.step bs r x.2) r = .ok r')
    (hall : r'.all = true) : Verdicts bs r l r' := by
  sorry

/-- **v2's `run` where it accepted every session is the sessions' verdicts in order, at the parameters' one
statement**: the statement's tags, the circuit and public files and the registered reads; the sessions' verdicts from
the empty run, each at the statement's inputs, scope and population check (`FilePart.check` on the public file); and
the proved units held to the evidence's (`provedHeld`). -/
theorem run_verdicts {p : _root_.Spot.OneStageZk.Own} {sessions : Array _root_.Spot.OneStageZk.RunFiles}
    {pu : Option (Array ℕ)} {bs : _root_.Core.Blobs} {r : _root_.Spot.OneStageZk.Run}
    (h : _root_.Spot.OneStageZk.run p sessions pu bs = .ok r) (hall : r.all = true) :
    ∃ (a : StepAt) (tags0 : Flock.Tags) (circuitB pubB : ByteArray),
      Flock.Tags.find? _root_.Spot.OneStageZk.STATEMENT = some tags0 ∧ tags0.plainLeaves = false ∧
      bs.get p.circuit "circuit" = .ok circuitB ∧ bs.get p.pub "public file" = .ok pubB ∧
      a.I = _root_.Spot.OneStageZk.inputsOf tags0 p circuitB pubB bs ∧
      _root_.Spot.OneStageZk.readsOf p.registered = .ok a.rd ∧ a.scope = Record.scopeOf a.I a.rd ∧
      a.population = _root_.Spot.FilePart.check p.program (_root_.Spot.getD p.partition "query") pubB ∧
      Verdicts bs ⟨#[], #[], #[], true⟩ (sessions.toList.map fun s => (a, s)) r ∧
      ∀ want, pu = some want → _root_.Spot.OneStageZk.provedHeld r.proved want = .ok () := by
  sorry

end Verity.Sampled
