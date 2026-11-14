module

public import Proofs.Properties.Verity.Sampled.Window.Logged
public import Proofs.Properties.Verity.Sampled.Window.Accepts

/-!
# A member's session's verdict is flock-verify `--zk`'s acceptance of what the member's writer writes

A member's step (`MemberSteps.step`) concludes that flock-verify `--zk` accepts, at the member's inputs, what the
member's writer writes for its session's outcome. One-stage v2 judges each session with `flock-verify --zk` at the
session's own statement (`SessionVerdict`), on the record and the proof files it reads on the blobs. This module
carries that verdict to the writer, at a member of a window, as `Sampled/Accepts.lean` does at one session:

* **What a session's verdict verified is accepted at its statement** (`SessionVerdict.accepts`): the record and the
  proof files are a transcript `AcceptsZKP` accepts at `--zk`'s inputs of the statement's, with no public inputs.
* **The statement is the member's** (`StatesMember`): the statement's tags, circuit file, public file, table count,
  partition and program are the member's, the member has no work, stratified or closure input and no recomputed
  public inputs, and the statement's lookup tables find by name whatever the member's find (`TablesExtend`).
* **A session's verdict at its member** (`SessionVerdict.accepts_member`): the record and the proof files are a
  transcript `AcceptsZKP m.I m.own` accepts, at the session `Exec.setupOfP` sets up at the member's inputs
  (`zkInputs_toExec`, `acceptsZKP_back`).
* **The member's writer** (`Member.wr_accepts`): `Member.toV5`'s writer is `wrZP`, which keeps an accepted
  transcript's view (`acceptsZKP_wrZP_planZHJP`), so what it writes for an outcome whose roots, link points, rounds
  and proof files are the accepted transcript's decoded ones (`eq_viewZP`) is accepted.
* **The moves of an outcome are a record's** (`MemberMovesOf`, `MovesOf` at a member): the record carries the
  member's part of the sample, and the outcome's roots, link points and rounds are those of the session the record
  decodes to at the session `--zk` sets up at the member's inputs, and its proof files are the session's.
* **The step's acceptance** (`member_step_accepts`): a session's verdict at the member's statement, with the
  outcome's moves the record's, gives `AcceptsZKP` of what the member's writer writes.
* **The members' steps, from each drawn member's session's verdict and moves** (`memberSteps_of_verdicts`):
  `MemberSteps` at an invariant wherever each drawn member's step gives the invariant at the next place, a session's
  verdict at the member's statement and the outcome's moves at its record.
-/

@[expose] public section

namespace Verity.Sampled

open Verity.Game
open Lean (Json)
open FlockSoundness.Discharge
open FlockSoundness.Discharge.ZkHidden (AcceptsZKP)
open FlockSoundness.Discharge.Recursion.Witness (toExec)
open FlockSoundness.Discharge.Tables (LiveOutJ)
open FlockSoundness.Refine (ptsOf TablesExtend)
open Claims.Bridge.CustodyZkP (repRounds acceptsZKP_wrZP_planZHJP)

/-! ## A session's verdict at its statement -/

/-- **What a session's verdict verified is a transcript flock-verify `--zk` accepts at its statement**: the record
and the proof files of `SessionVerdict`, at `--zk`'s inputs of the statement's with no public inputs, at the session
`Exec.setupOfP` sets up there (`setupAt_setupOfP`). -/
theorem SessionVerdict.accepts {a : StepAt} {bs : _root_.Core.Blobs} {keys : Array String}
    {s : _root_.Spot.OneStageZk.RunFiles} {record : String} {spec : Flock.SessionSpec} {tabs : Array Flock.Setup}
    {rest : Json × Json} {proofs : Array ByteArray} {units : Array ℕ}
    (h : SessionVerdict a bs keys s record spec tabs rest proofs units) :
    ∃ sts, Exec.setupOfP (ZkExec.zkInputs (toExec a.I)) none record = .ok (spec, sts) ∧
      AcceptsZKP (toExec a.I) none { record := record, proofs := proofs } := by
  have hv := h.verify
  obtain ⟨sts, hs, hshape, rfl⟩ := setupAt_setupOfP h.setup
  exact ⟨sts, hs, spec, sts, hs, hshape, hv⟩

/-- **The statement `a` is the member `m`'s**: its tags, circuit file, public file, table count, partition and
program are the member's inputs'; the member has no work, stratified or closure input and no recomputed public
inputs; and the statement's lookup tables find by name whatever the member's find. -/
structure StatesMember {nJ : ℕ} (a : StepAt) (m : Member nJ) : Prop where
  tags : a.I.tags = m.I.tags
  circuit : a.I.circuitFile = m.I.circuitFile
  publicFile : a.I.publicFile = m.I.publicFile
  count : a.I.count = m.I.count
  partition : a.I.partition = m.I.partition
  program : a.I.program = m.I.program
  work : m.I.work = none
  stratified : m.I.stratified = none
  closure : m.I.closure = none
  own : m.own = none
  tables : TablesExtend m.I.tables a.I.tables

/-- **A session's verdict is flock-verify `--zk`'s at its member**: where the session's statement is the member
`m`'s, the record and the proof files its verdict verified are a transcript `AcceptsZKP m.I m.own` accepts, at the
session `Exec.setupOfP` sets up at the member's inputs. -/
theorem SessionVerdict.accepts_member {nJ : ℕ} {a : StepAt} {m : Member nJ} (hm : StatesMember a m)
    {bs : _root_.Core.Blobs} {keys : Array String} {s : _root_.Spot.OneStageZk.RunFiles} {record : String}
    {spec : Flock.SessionSpec} {tabs : Array Flock.Setup} {rest : Json × Json} {proofs : Array ByteArray}
    {units : Array ℕ} (h : SessionVerdict a bs keys s record spec tabs rest proofs units) :
    ∃ sts, Exec.setupOfP (ZkExec.zkInputs m.I) m.own record = .ok (spec, sts) ∧
      AcceptsZKP m.I m.own { record := record, proofs := proofs } := by
  obtain ⟨sts, hs, hat⟩ := h.accepts
  have hz := zkInputs_toExec hm.tags hm.circuit hm.publicFile hm.count hm.partition hm.program hm.work
    hm.stratified hm.closure
  rw [hz] at hs
  rw [hm.own]
  have hp := Flock.Desc.parse_flat m.hD
  exact ⟨sts, setupOfP_back hm.tables m.hcount m.ht hp none record _ hs,
    acceptsZKP_back hm.tables m.hcount m.ht hp ((acceptsZKP_congr hz).1 hat)⟩

/-! ## The member's writer -/

/-- **What a member's writer writes for the moves read off an accepted transcript is accepted**: `Member.toV5`'s
writer is `wrZP`, which keeps an accepted transcript's view, so at a transcript `AcceptsZKP m.I m.own` accepts whose
record carries the member's part of the sample `S`, an outcome whose roots, link points, rounds and proof files are
the transcript's decoded ones (`eq_viewZP`) is written to a transcript `AcceptsZKP m.I m.own` accepts
(`wr_accepts`, at a member of a window). -/
theorem Member.wr_accepts {nJ : ℕ} (m : Member nJ) (t : ℕ) {S : Finset (Fin nJ)} {R : ByteArray}
    {tr : Exec.Transcript} (ha : AcceptsZKP m.I m.own tr)
    (hdr : Exec.recordDraw tr.record = some (printedPart (m.part S))) {spec : Flock.SessionSpec}
    {sts : Array Flock.Stmt} {sess : Flock.RunRecord}
    (hs : Exec.setupOfP (ZkExec.zkInputs m.I) m.own tr.record = .ok (spec, sts))
    (hd : Flock.Record.decode spec tr.record = .ok sess)
    {o : LiveOutJ (memberTabs S R (m.toV5 t)) (memberPts (m.toV5 t))} (hB : o.rootB = sess.rootB)
    (hP : o.pts = ptsOf _ sess.points) (hR : ∀ j r, o.reps j r = repRounds spec sess j.val r.val)
    (hF : o.proofs = tr.proofs) : AcceptsZKP m.I m.own ((m.toV5 t).wr (m.part S) R o) := by
  rw [eq_viewZP hs hd hB hP hR hF]
  exact acceptsZKP_wrZP_planZHJP m.hTags.1 printedPart m.y₀ ha hdr

/-- **The moves of `out` are those the record `record` decodes to** at the member `m`, with the proof files `proofs`,
at the sample `S` (`MovesOf`, at a member of a window): the record carries the member's part of `S` as its printed
sample file, and `out`'s roots, link points and rounds are those of the session the record decodes to at the session
`--zk` sets up at the member's inputs, and its proof files are `proofs`. -/
def MemberMovesOf {nJ : ℕ} (m : Member nJ) (t : ℕ) (S : Finset (Fin nJ)) (R : ByteArray) (record : String)
    (proofs : Array ByteArray) (out : LiveOutJ (memberTabs S R (m.toV5 t)) (memberPts (m.toV5 t))) : Prop :=
  ∀ spec sts sess, Exec.setupOfP (ZkExec.zkInputs m.I) m.own record = .ok (spec, sts) →
    Flock.Record.decode spec record = .ok sess →
    Exec.recordDraw record = some (printedPart (m.part S)) ∧ out.rootB = sess.rootB ∧
      out.pts = ptsOf _ sess.points ∧ (∀ j r, out.reps j r = repRounds spec sess j.val r.val) ∧ out.proofs = proofs

/-- **A member's step's acceptance**: where a session's statement is the member `m`'s, its verdict holds of the
record `record` and the proof files `proofs`, and the moves of `out` are those `record` decodes to with `proofs`
(`MemberMovesOf`), flock-verify `--zk` accepts at the member's inputs what the member's writer writes for `out`
(`wr_accepts_of_verdict`, at a member of a window). -/
theorem member_step_accepts {nJ : ℕ} (m : Member nJ) (t : ℕ) {a : StepAt} (hm : StatesMember a m)
    {bs : _root_.Core.Blobs} {keys : Array String} {s : _root_.Spot.OneStageZk.RunFiles} {record : String}
    {spec : Flock.SessionSpec} {tabs : Array Flock.Setup} {rest : Json × Json} {proofs : Array ByteArray}
    {units : Array ℕ} (h : SessionVerdict a bs keys s record spec tabs rest proofs units) {S : Finset (Fin nJ)}
    {R : ByteArray} {out : LiveOutJ (memberTabs S R (m.toV5 t)) (memberPts (m.toV5 t))}
    (hM : MemberMovesOf m t S R record proofs out) :
    AcceptsZKP (m.toV5 t).I (m.toV5 t).own ((m.toV5 t).wr ((m.toV5 t).part S) R out) := by
  obtain ⟨sts, hs, hat⟩ := h.accepts_member hm
  have hd : ∃ sess, Flock.Record.decode spec record = .ok sess := by
    obtain ⟨spec', sts', hs', -, hver⟩ := hat
    obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Except.ok.inj (hs.symm.trans hs'))
    exact zkVerify_decode hver
  obtain ⟨sess, hd⟩ := hd
  obtain ⟨hdr, hB, hP, hR, hPf⟩ := hM spec sts sess hs hd
  exact m.wr_accepts t hat hdr hs hd hB hP hR hPf

/-! ## The members' steps -/

/-- A member of Flock's window at `t` calls is a member of the window at that place, as Flock's (`Member.toV5`). -/
theorem Window.joint_getElem? (W : Window) (t i : ℕ) {m : Flock.Guarantees.MemberV5 ByteArray W.nJ}
    (h : (W.joint t).ms[i]? = some m) : ∃ m' : Member W.nJ, W.ms[i]? = some m' ∧ m = m'.toV5 t := by
  have h' : (W.ms.map fun m => m.toV5 t)[i]? = some m := h
  rw [List.getElem?_map] at h'
  obtain ⟨m', h1, h2⟩ := Option.map_eq_some_iff.1 h'
  exact ⟨m', h1, h2.symm⟩

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

variable (V : Vocab d α) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)

/-- **The members' steps, from each drawn member's session's verdict and moves**: `MemberSteps` at an invariant `I`
that holds at the play's start and is kept at a member with no proof block in the sample, wherever the session's trace
of each member `m` with a proof block in the sample, under `MemberSteps.step`'s premises, has `I` at the next place
and the position past the stop after its session's end, a session's verdict (`SessionVerdict`) at a statement that is
`m`'s (`StatesMember`), and its outcome's moves those the verdict's record decodes to with the verdict's proof files
(`MemberMovesOf`). The acceptance of what the member's writer writes is `member_step_accepts`. -/
theorem memberSteps_of_verdicts (k : Verity.Registration.Composite) (P : Exchange → Prop) (W : Window) (t : ℕ)
    (rd : Reading W.rounds) (σ : Strategy (V.game θ F c)) (D : WindowDecoders V θ F c W σ)
    (M : WindowCoinMap (V.game θ F c) σ W.Ω W.rounds) (BL : WindowLinkHold) (B : WindowRoundHold)
    (I : (W.extend M.widening).L.Ω → Pos (V.game θ F c) σ Claims.Msg (Fin W.rounds) → ℕ →
      Pos (V.game θ F c) σ Claims.Msg (Fin W.rounds) → Prop)
    (hstart : ∀ ω r, I ω r 0 (Pos.root (roundCuts V θ F c rd) (M.start ω)))
    (hstep : ∀ (ω : (W.extend M.widening).L.Ω) (i : ℕ) (m : Member W.nJ)
      (p q r : Pos (V.game θ F c) σ Claims.Msg (Fin W.rounds))
      (out : LiveOutJ (memberTabs ((W.extend M.widening).L.draw ω) (windowReg V θ F c W rd σ D) (m.toV5 t))
        (memberPts (m.toV5 t))),
      W.ms[i]? = some m → ((m.toV5 t).part ((W.extend M.widening).L.draw ω)).Nonempty →
      I ω r i p → p.Inv (roundCuts V θ F c rd) →
      StopsAt rd.round rd.reveal (V.signer θ) θ F c σ.1 p.toStop →
      MemberReach D M BL B ((W.extend M.widening).L.draw ω) (windowReg V θ F c W rd σ D) i (m.toV5 t) p q out →
      q.Past r → r.Inv (roundCuts V θ F c rd) →
      V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ r.tape) ∧ NoMiss V θ F c rd σ r.tape ∧
        P (CoinTape.run (V.game θ F c) σ r.tape) →
      I ω r (i + 1) q.toStop.consume ∧ ∃ (a : StepAt) (bs : _root_.Core.Blobs) (keys : Array String)
        (s : _root_.Spot.OneStageZk.RunFiles) (record : String) (spec : Flock.SessionSpec)
        (tabs : Array Flock.Setup) (rest : Json × Json) (proofs : Array ByteArray) (units : Array ℕ),
        StatesMember a m ∧ SessionVerdict a bs keys s record spec tabs rest proofs units ∧
          MemberMovesOf m t ((W.extend M.widening).L.draw ω) (windowReg V θ F c W rd σ D) record proofs out)
    (hskip : ∀ (ω : (W.extend M.widening).L.Ω) (i : ℕ) (m : Member W.nJ)
      (p r : Pos (V.game θ F c) σ Claims.Msg (Fin W.rounds)),
      W.ms[i]? = some m → ¬ ((m.toV5 t).part ((W.extend M.widening).L.draw ω)).Nonempty → I ω r i p →
      I ω r (i + 1) p) :
    MemberSteps V θ F c k P W t rd σ D M BL B I where
  start := hstart
  step ω i mv p q r out hm hd hI hpi hS hmr hqr hri hacc := by
    obtain ⟨m, hm', rfl⟩ := Window.joint_getElem? (W.extend M.widening) t i hm
    obtain ⟨hI', a, bs, keys, s, record, spec, tabs, rest, proofs, units, hst, hv, hmo⟩ :=
      hstep ω i m p q r out hm' hd hI hpi hS hmr hqr hri hacc
    exact ⟨hI', member_step_accepts m t hst hv hmo⟩
  skip ω i mv p r hm hd hI := by
    obtain ⟨m, hm', rfl⟩ := Window.joint_getElem? (W.extend M.widening) t i hm
    exact hskip ω i m p r hm' hd hI

end Verity.Sampled
