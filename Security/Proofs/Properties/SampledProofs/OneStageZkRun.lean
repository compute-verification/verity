import Properties.SampledProofs.OneStageZk
import Proofs.Flock.Claims.Sampling.Deployed
import Proofs.Flock.Claims.KS.ZkAccepts
import Proofs.Flock.Claims.Recursion.RecRecord
import Proofs.Flock.Claims.DeployCode
import Proofs.Flock.Claims.Recursion.Inner
import Proofs.Flock.Claims.Recursion.DeploySessions
import Proofs.Flock.Claims.Recursion.DeployFacts
import Proofs.Flock.Claims.Bridge.Committed
import Proofs.Flock.Claims.Complete.Zk.Run
import Proofs.Flock.Claims.Complete.Encode
import Proofs.Flock.Soundness.Discharge.Exec.Setup
import Proofs.Flock.Soundness.Discharge.ZkExec.Event
import Proofs.Flock.Claims.JsonEq
import Proofs.Flock.Claims.Recursion.NextShaDec

/-! # One-stage v2's acceptance is the sampled model's (ONE-3t, PRF-209)

Sampled proofs' one-stage audit on the proofs (`verity/sampled-proofs/one-stage/v2`, `checkBlobs`) runs
`flock-verify verify --zk` on each of the evidence's sessions, from the blobs. `acc_of_accepts` says what that
acceptance means for the run `Claims.Sampling`'s bounds count: the run of the deployed sampled verifier
(`SamplingDeploy.deploy rel cd`) at a checked unit whose session the blobs show accepts.

* `StmtNames`: the session's statement `xS`, as the deployed Flock verifier reads its bytes (`FlockDeploy.Raw`), is
  `verify --zk` at the component's own inputs: the `verity/flock-circuit` statement, the circuit and public files the
  digests name, the lookup tables the blobs hold, one table, the record's own draw, the verifier's partition, program
  and registered file, and no other input.
* `Shows`: the session's last message (`FlockDeploy.lastFinal`) is the `Final` whose statement is `xS`, whose record
  is the session's record blob, and whose proof files are the session's rep blobs, in the stream order of `xS`'s setup.
* `LiveCoins` (`live-verifier`, `Assumptions.Zk.RecordCustodyZKJ`, at the session): the session is the live one its
  record was written for. It opens with its link message, and wherever the deployed verifier decides on its messages
  and coins, the record's live session is that link, its coin and the round messages with theirs
  (`FlockDeploy.custody`). The model verifier checks this itself; the compiled component, which holds no live coin,
  can't. It is admissible until ONE-4 (PRF-200) lands, after which the component reads the coins through ONE-4's
  signed device log and this premise goes.
* `SessionRuns sha512 (deploy rel cd)`: the code fact that a checked unit's session is a Flock run deciding the
  sampled run, which `sampling_deployed` takes too. `SamplingDeploy.Code.V` is a field, so nothing else ties the
  sampled verifier's runs to Flock's.

The proof reads the session's verdict off `checkBlobs` (`session_verdict`): `verify --zk` set it up at its record's
draw (`setupAt`, `sessionAt`, `buildSessionOf`), with the checks `verifyCmd` makes there, and accepted its proof files
with the checks after `Zk.verify` (`verdictOf`). Those are the deployed verifier's setup of `xS` (`Ins.setup`, at the
inputs `StmtNames` decodes, `ins_eq`) and its decision's checks (`FlockDeploy.checks`, with `LiveCoins`' custody), so
at the run's last message the deployed verifier accepts (`flock_acc`), and `SessionRuns` carries that to the sampled
run.
-/

open Lean (Json)
open SampledProofs SampledProofs.OneStageZk

namespace Properties.SampledProofs.OneStageZk

open Claims (Msg Out)

/-- **The statement names the component's inputs.** `xS` decodes (`FlockDeploy.decode`) to `verify --zk` at the
`verity/flock-circuit` statement without `--coins`, on the circuit and public files the blobs hold under the digests
`p.circuit` and `pub`, the lookup tables the blobs hold (`tablesOf`), one table, the draw of `record` (`recordDraw`, as
the deployed verifier rebuilds a draw from its bytes), the verifier's own partition, program and registered file, and
no `--work`, `--work-table`, `--stratified`, `--closure` or `--public-inputs`. -/
def StmtNames (p : Own) (pub : String) (blobs : Array ByteArray) (record : String) (xS : Msg) : Prop :=
  ∃ w : FlockDeploy.Raw, FlockDeploy.decode xS = some w ∧ w.statement = STATEMENT ∧ w.zk = true ∧
    w.osCoins = false ∧
    (Core.Blobs.ofArray blobs).get p.circuit "circuit" = .ok w.circuit ∧
    (Core.Blobs.ofArray blobs).get pub "public file" = .ok w.publicFile ∧
    w.tables = tablesOf (Core.Blobs.ofArray blobs) ∧ w.count = 1 ∧
    (w.draw.mapM (FlockDeploy.jsonOf "draw")).map (Option.map Flock.norm) = .ok (Flock.Witness.recordDraw record) ∧
    w.partition.map (FlockDeploy.jsonOf "partition") = some (.ok p.partition) ∧ w.program = some p.program ∧
    w.work = none ∧ w.workTable = none ∧ w.stratified = none ∧ w.closure = none ∧
    w.registered.map (FlockDeploy.jsonOf "registered reads") = p.registered.map .ok ∧ w.publicInputs = none

/-- **The blobs show the session.** The session view `v`'s last message is a `Final` `f` whose statement is `xS`,
which names the component's inputs at `f`'s record (`StmtNames`); the blob under the session's `record` digest is
`f`'s record; and at `xS`'s setup, `f` carries one proof file per rep stream, each the blob the session's `reps` name
for that stream's file (`<stream with / as .>.bin`). -/
def Shows (p : Own) (pub : String) (blobs : Array ByteArray) (s : Session) (xS : Msg) (v : Claims.Seen) : Prop :=
  ∃ f, FlockDeploy.lastFinal v = some f ∧ f.stmt = xS ∧ StmtNames p pub blobs f.record xS ∧
    (Core.Blobs.ofArray blobs).get s.record s!"session {s.name}'s record" = .ok f.record.toUTF8 ∧
    ∀ i spec sts, FlockDeploy.setupOf true xS = some (i, spec, sts) →
      f.proofs.size = spec.repStreams.size ∧ ∀ k (hk : k < spec.repStreams.size), ∃ d,
        (s.reps.find? (·.1 == s!"{spec.repStreams[k].2.2.replace "/" "."}.bin")).map (·.2) = some d ∧
        ∃ hk' : k < f.proofs.size,
          (Core.Blobs.ofArray blobs).get d s!"session files: {spec.repStreams[k].2.2.replace "/" "."}.bin" =
            .ok f.proofs[k]

/-- **The record holds the session's live coins** (`live-verifier` at the session, `Assumptions.Zk.RecordCustodyZKJ`):
the session view `v` at `xS` opens with its link message (`FlockDeploy.linkOf`), and wherever the deployed verifier
decides on `v`'s messages and the coins its answers show (`RecDeploy.outCoin`), the live session the record shows is
that link, its coin and the round messages with theirs (`FlockDeploy.custody`). -/
noncomputable def LiveCoins (xS : Msg) (v : Claims.Seen) : Prop :=
  (∃ link, (v.msgs.head?).bind (fun m => FlockDeploy.linkOf m.1) = some link) ∧
    ∀ i spec sts link c₁ mids f,
      FlockDeploy.act true xS (v.msgs.map Prod.fst) (v.msgs.map fun m => RecDeploy.outCoin m.2) =
          .decide i spec sts link c₁ mids f →
        (do FlockDeploy.custody spec (← FlockDeploy.liveOf spec f.record) link c₁ mids : Flock.V Unit) = .ok ()

namespace Run

open Flock (V)

/-! ## `Except`'s evaluation -/

theorem bind_ok {ε α β : Type} {x : Except ε α} {f : α → Except ε β} {b : β} (h : x >>= f = .ok b) :
    ∃ a, x = .ok a ∧ f a = .ok b := by
  cases x with
  | error => cases h
  | ok a => exact ⟨a, rfl, h⟩

theorem map_ok {ε α β : Type} {x : Except ε α} {f : α → β} {b : β} (h : x.map f = .ok b) :
    ∃ a, x = .ok a ∧ f a = b := by
  cases x with
  | error => cases h
  | ok a => exact ⟨a, rfl, Except.ok.inj h⟩

theorem mapError_ok {ε ε' α : Type} {x : Except ε α} {f : ε → ε'} {a : α} (h : x.mapError f = .ok a) :
    x = .ok a := by
  cases x with
  | error e => cases h
  | ok b => cases h; rfl

theorem ok_bind {ε α β : Type} (a : α) (f : α → Except ε β) : (Except.ok a >>= f) = f a := rfl
theorem pure_bind' {ε α β : Type} (a : α) (f : α → Except ε β) : (pure a >>= f : Except ε β) = f a := rfl
theorem map_ok' {ε α β : Type} (a : α) (f : α → β) : (f <$> (Except.ok a : Except ε α)) = Except.ok (f a) := rfl

/-- A blob found under a digest is found whatever the error message. -/
theorem get_what {bs : Core.Blobs} {d w : String} {b : ByteArray} (h : bs.get d w = .ok b) (w' : String) :
    bs.get d w' = .ok b := by
  unfold Core.Blobs.get at h ⊢
  obtain ⟨d', hu, h⟩ := bind_ok h
  rw [hu]
  simp only [ok_bind]
  by_cases hc : (d'.size != 64 || Flock.hexOf d' != d) = true
  · rw [if_pos hc] at h
    cases h
  · rw [if_neg hc] at h ⊢
    split at h
    · rename_i fst snd hf
      first | (rw [hf]; exact h) | (simp only [hf]; exact h) | (rw [hf]) | (simp only [hf] at h ⊢; exact h)
    · cases h

/-! ## Loops -/

theorem forIn_arr_of {α : Type} {xs : Array α} {P : α → Prop}
    {b : α → PUnit → V (ForInStep PUnit)} {u : PUnit}
    (hb : ∀ a v r, b a v = .ok r → P a ∧ r = .yield PUnit.unit) (h : forIn xs PUnit.unit b = .ok u) :
    ∀ a ∈ xs, P a := by
  rcases xs with ⟨l⟩
  simp only [List.forIn_toArray] at h
  intro a ha
  simp only [List.mem_toArray] at ha
  induction l generalizing u with
  | nil => cases ha
  | cons c l ih =>
    simp only [List.forIn_cons, bind, Except.bind] at h
    cases hc : b c PUnit.unit with
    | error e => rw [hc] at h; cases h
    | ok r =>
      rw [hc] at h
      rcases List.mem_cons.1 ha with rfl | ha
      · exact (hb _ _ _ hc).1
      · rw [(hb _ _ _ hc).2] at h
        exact ih h ha

theorem forIn_arr_ok {α : Type} (xs : Array α) (g : α → Unit → V (ForInStep Unit))
    (h : ∀ a ∈ xs, g a () = .ok (.yield ())) : forIn xs () g = .ok () := by
  rcases xs with ⟨l⟩
  simp only [List.forIn_toArray]
  exact FlockDeploy.list_forIn_unit_ok l g fun a ha => h a (List.mem_toArray.2 ha)

/-! ## What the compiled verdict ran -/

theorem push_all {I : Flock.Witness.Inputs} {rd : Reads} {programSha : Option String} {want : Json} {r r' : Run}
    {s : Session} {record : String} {st : SetUp} {coinKey : Option String} {verdict : V Unit}
    (h : push I rd programSha want r s record st coinKey verdict = .ok r') (hall : r'.all = true) :
    r.all = true ∧ verdict = .ok () := by
  unfold push at h
  cases verdict with
  | error why =>
    cases h
    simp at hall
  | ok u =>
    cases u
    dsimp only at h
    split at h
    · cases h
    · obtain ⟨units, _, h⟩ := bind_ok h
      cases h
      exact ⟨hall, rfl⟩

theorem step_verdict {I : Flock.Witness.Inputs} {rd : Reads} {blobs : Array ByteArray} {scope : V Unit}
    {programSha : Option String} {want : Json} {r r' : Run} {s : Session}
    (h : step I rd (Core.Blobs.ofArray blobs) scope programSha want r s = .ok r') (hall : r'.all = true) :
    r.all = true ∧ ∃ b record,
      (Core.Blobs.ofArray blobs).get s.record s!"session {s.name}'s record" = .ok b ∧
      String.fromUTF8? b = some record ∧
      verdictOf I scope r.coinKeys record (Flock.Witness.recordDraw record)
        (setupAt I rd (Flock.Witness.recordDraw record))
        (filesOf (Core.Blobs.ofArray blobs) s (setupAt I rd (Flock.Witness.recordDraw record)))
        (coinKeyOf record (setupAt I rd (Flock.Witness.recordDraw record))) = .ok () := by
  unfold step at h
  obtain ⟨b, hb, h⟩ := bind_ok h
  cases hrec : String.fromUTF8? b with
  | none => (first | rw [hrec] at h | simp only [hrec] at h) <;> cases h
  | some record =>
    first | rw [hrec] at h | simp only [hrec] at h
    try dsimp only at h
    obtain ⟨hr, hv⟩ := push_all h hall
    exact ⟨hr, b, record, hb, hrec, hv⟩

theorem fold_verdict {I : Flock.Witness.Inputs} {rd : Reads} {blobs : Array ByteArray} {scope : V Unit}
    {programSha : Option String} {want : Json} :
    ∀ (l : List Session) (r0 r : Run),
      l.foldlM (step I rd (Core.Blobs.ofArray blobs) scope programSha want) r0 = .ok r → r.all = true →
      r0.all = true ∧ ∀ s ∈ l, ∃ ck b record,
        (Core.Blobs.ofArray blobs).get s.record s!"session {s.name}'s record" = .ok b ∧
        String.fromUTF8? b = some record ∧
        verdictOf I scope ck record (Flock.Witness.recordDraw record)
          (setupAt I rd (Flock.Witness.recordDraw record))
          (filesOf (Core.Blobs.ofArray blobs) s (setupAt I rd (Flock.Witness.recordDraw record)))
          (coinKeyOf record (setupAt I rd (Flock.Witness.recordDraw record))) = .ok ()
  | [], r0, r, h, hall => by
    cases h
    exact ⟨hall, fun s hs => by cases hs⟩
  | s :: l, r0, r, h, hall => by
    rw [List.foldlM_cons] at h
    obtain ⟨r1, h1, h2⟩ := bind_ok h
    obtain ⟨hr1, hrest⟩ := fold_verdict l r1 r h2 hall
    obtain ⟨hr0, b, record, hb, hu, hv⟩ := step_verdict h1 hr1
    refine ⟨hr0, fun s' hs' => ?_⟩
    rcases List.mem_cons.1 hs' with rfl | hs'
    · exact ⟨_, b, record, hb, hu, hv⟩
    · exact hrest s' hs'

/-- **What v2's acceptance ran on each session**: `run`'s inputs (the statement's tags, the circuit and public files
the digests name, the verifier's registered reads), and the session's record and verdict on them. -/
theorem session_verdict {p : Params} {c : OneStage.Commitment} {e : Evidence} {blobs : Array ByteArray}
    {pub : OneStage.Public} {prof : Core.Integrity}
    (hacc : checkBlobs p c e (Core.Blobs.ofArray blobs) = .accept pub prof) {s : Session} (hs : s ∈ e.sessions) :
    ∃ (tags0 : Flock.Tags) (circuitB pubB : ByteArray) (rd : Reads), Flock.Tags.find? STATEMENT = some tags0 ∧
      tags0.plainLeaves = false ∧
      (Core.Blobs.ofArray blobs).get p.own.circuit "circuit" = .ok circuitB ∧
      (Core.Blobs.ofArray blobs).get e.pub "public file" = .ok pubB ∧ readsOf p.own.registered = .ok rd ∧
      ∃ ck b record, (Core.Blobs.ofArray blobs).get s.record s!"session {s.name}'s record" = .ok b ∧
        String.fromUTF8? b = some record ∧
        verdictOf (inputsOf tags0 p.own circuitB pubB (Core.Blobs.ofArray blobs))
          (Flock.ProvedScope.checkOf
            (Flock.ProvedScope.circuitOf
              (Flock.Witness.zkTags (inputsOf tags0 p.own circuitB pubB (Core.Blobs.ofArray blobs))) circuitB
              (tablesOf (Core.Blobs.ofArray blobs)))
            (Flock.Witness.zkTags (inputsOf tags0 p.own circuitB pubB (Core.Blobs.ofArray blobs))) true pubB 1 false
            rd.registered.isSome)
          ck record (Flock.Witness.recordDraw record)
          (setupAt (inputsOf tags0 p.own circuitB pubB (Core.Blobs.ofArray blobs)) rd
            (Flock.Witness.recordDraw record))
          (filesOf (Core.Blobs.ofArray blobs) s
            (setupAt (inputsOf tags0 p.own circuitB pubB (Core.Blobs.ofArray blobs)) rd
              (Flock.Witness.recordDraw record)))
          (coinKeyOf record
            (setupAt (inputsOf tags0 p.own circuitB pubB (Core.Blobs.ofArray blobs)) rd
              (Flock.Witness.recordDraw record))) = .ok () := by
  unfold checkBlobs at hacc
  split at hacc
  · cases hacc
  · next r hrun =>
    split at hacc
    · next pub' prof' hchk =>
      by_cases hall : r.all = true
      · unfold run at hrun
        cases ht : Flock.Tags.find? STATEMENT with
        | none => simp only [ht] at hrun <;> cases hrun
        | some tags0 =>
          simp only [ht] at hrun
          obtain ⟨u0, hpl, hrun⟩ := bind_ok hrun
          obtain ⟨_, _, hrun⟩ := bind_ok hrun
          obtain ⟨circuitB, hc, hrun⟩ := bind_ok hrun
          obtain ⟨pubB, hpb, hrun⟩ := bind_ok hrun
          obtain ⟨rd, hrd, hrun⟩ := bind_ok hrun
          obtain ⟨r1, hf, hrun⟩ := bind_ok hrun
          rw [← Array.foldlM_toList] at hf
          try dsimp only at hrun
          split at hrun
          · cases hrun
          · cases hrun
            obtain ⟨_, hss⟩ := fold_verdict _ _ _ hf hall
            have hp : tags0.plainLeaves = false := by
              cases hp : tags0.plainLeaves
              · rfl
              · exfalso
                first
                | (rw [hp] at hpl; cases hpl)
                | (simp only [hp, ↓reduceIte] at hpl; cases hpl)
                | simp [hp] at hpl
            exact ⟨tags0, circuitB, pubB, rd, rfl, hp, hc, hpb, hrd, hss s (Array.mem_toList_iff.2 hs)⟩
      · rw [if_neg hall] at hacc
        cases hacc
    · next o hne hchk =>
      first
      | exact (hchk _ _ hacc).elim
      | exact (hchk _ _ (hne ▸ hacc)).elim
      | exact (hchk _ _ (hne.symm ▸ hacc)).elim
      | (subst hne; exact (hchk _ _ hacc).elim)

/-- An accepting verdict ran `Zk.verify` on the session's setup and proof files, then the scope check, the draw-less
session's refusal and the record's canonical text. -/
theorem verdictOf_parts {I : Flock.Witness.Inputs} {scope : V Unit} {ck : Array String} {record : String}
    {draw : Option Json} {st : SetUp} {files : V (Array ByteArray)} {coinKey : Option String}
    (h : verdictOf I scope ck record draw st files coinKey = .ok ()) :
    ∃ proofs spec tabs x, files = .ok proofs ∧ st = .ok (spec, tabs, x) ∧
      Flock.Zk.verify spec tabs record proofs = .ok () ∧ scope = .ok () ∧
      (draw.isNone = true → Flock.ProvedScope.drawLaw true (Flock.Witness.zkTags I) false draw = .ok ()) ∧
      Flock.RecordText.check record = .ok () := by
  unfold verdictOf at h
  obtain ⟨proofs, hf, h⟩ := bind_ok h
  obtain ⟨_, _, h⟩ := bind_ok h
  obtain ⟨⟨spec, tabs, x⟩, hst, h⟩ := bind_ok h
  obtain ⟨u, hv, h⟩ := bind_ok h
  obtain ⟨u2, hsc, h⟩ := bind_ok h
  obtain ⟨_, _, h⟩ := bind_ok h
  obtain ⟨u3, hdl, h⟩ := bind_ok h
  obtain ⟨_, _, h⟩ := bind_ok h
  obtain ⟨_, _, h⟩ := bind_ok h
  obtain ⟨_, _, h⟩ := bind_ok h
  cases u
  cases u2
  cases u3
  refine ⟨proofs, spec, tabs, x, hf, hst, hv, hsc, fun hn => ?_, h⟩
  rw [if_pos hn] at hdl
  exact hdl

theorem setupAt_parts {I : Flock.Witness.Inputs} {rd : Reads} {draw : Option Json} {spec : Flock.SessionSpec}
    {tabs : Array Flock.Setup} {x : Json × Json} (h : setupAt I rd draw = .ok (spec, tabs, x)) :
    ∃ sts, sessionAt I rd draw = .ok (spec, sts) ∧ tabs = sts.map Flock.Setup.ofCircuit ∧
      (draw.isSome = true → Flock.ProvedScope.drawLaw true (Flock.Witness.zkTags I) false draw = .ok ()) := by
  unfold setupAt at h
  obtain ⟨⟨spec', sts⟩, h1, h⟩ := bind_ok h
  obtain ⟨u, hdl, h⟩ := bind_ok h
  cases h
  cases u
  exact ⟨sts, mapError_ok h1, rfl, fun hs => by rw [if_pos hs] at hdl; exact hdl⟩

theorem sessionAt_parts {I : Flock.Witness.Inputs} {rd : Reads} {draw : Option Json} {spec : Flock.SessionSpec}
    {sts : Array Flock.Stmt} (h : sessionAt I rd draw = .ok (spec, sts)) :
    Flock.RecSetup.checkInner none none (pure (none, none)) = .ok () ∧
      buildSessionOf (Flock.ProvedScope.circuitOf (Flock.Witness.zkTags I) I.circuitFile I.tables)
        (Flock.Witness.zkTags I) I.circuitFile I.publicFile (Flock.Witness.zkCoins I) I.tables I.count draw
        I.partition I.program rd.registered = .ok (spec, sts) ∧
      setupChecks I rd sts = .ok () ∧ sts.forM Flock.Zk.shapeOk = .ok () ∧ recChecks rd sts = .ok () := by
  unfold sessionAt at h
  obtain ⟨_, _, h⟩ := bind_ok h
  obtain ⟨u1, hin, h⟩ := bind_ok h
  obtain ⟨r, hr, h⟩ := bind_ok h
  obtain ⟨u2, hsc, h⟩ := bind_ok h
  obtain ⟨u3, hsh, h⟩ := bind_ok h
  obtain ⟨u4, hrc, h⟩ := bind_ok h
  cases h
  cases u1
  cases u2
  cases u3
  cases u4
  exact ⟨hin, hr, hsc, hsh, hrc⟩

/-- `buildSessionOf` on `ProvedScope.circuitOf`'s circuit sets up what `Stmt.setupTables` sets up. -/
theorem bso_ok {tags : Flock.Tags} {cB pB : ByteArray} {coins : Flock.Coins}
    {tables : List (String × ByteArray)} {count : Nat} {draw partition : Option Json} {program : Option ByteArray}
    {registered : Option (Array Flock.Registered.Port)} {spec : Flock.SessionSpec} {sts : Array Flock.Stmt}
    (h : buildSessionOf (Flock.ProvedScope.circuitOf tags cB tables) tags cB pB coins tables count draw partition program
      registered = .ok (spec, sts)) :
    Flock.Stmt.setupTables tags cB pB coins count tables draw partition program = .ok (spec, sts) := by
  unfold buildSessionOf at h
  obtain ⟨_, _, h⟩ := bind_ok h
  obtain ⟨r, h1, h⟩ := bind_ok h
  obtain ⟨_, _, h⟩ := bind_ok h
  obtain ⟨_, _, h⟩ := bind_ok h
  cases h
  cases hh : tags.hiddenOutputs
  · simpa [hh] using h1
  · first
    | simpa [Flock.ProvedScope.circuitOf, hh, Flock.Stmt.setupTables_eq] using h1
    | (rw [Flock.Stmt.setupTables_eq]; simpa [Flock.ProvedScope.circuitOf, hh] using h1)

theorem buildSessionOf_parts {parsed : V Flock.Circuit} {tags : Flock.Tags} {cB pB : ByteArray}
    {coins : Flock.Coins} {tables : List (String × ByteArray)} {count : Nat} {draw partition : Option Json}
    {program : Option ByteArray} {registered : Option (Array Flock.Registered.Port)} {spec : Flock.SessionSpec}
    {sts : Array Flock.Stmt}
    (h : buildSessionOf parsed tags cB pB coins tables count draw partition program registered = .ok (spec, sts)) :
    tags.hm96Rows = true ∧
      (∀ st ∈ sts, Flock.checkRegistered st registered = .ok () ∧ checkIdentifier st = .ok ()) ∧
      Flock.Registered.checkFull registered draw = .ok () := by
  unfold buildSessionOf at h
  obtain ⟨_, h0, h⟩ := bind_ok h
  obtain ⟨r, _, h⟩ := bind_ok h
  obtain ⟨_, h2, h⟩ := bind_ok h
  obtain ⟨u3, h3, h⟩ := bind_ok h
  cases h
  cases u3
  refine ⟨?_, forIn_arr_of ?_ h2, h3⟩
  · cases hm : tags.hm96Rows
    · exfalso
      first
      | (rw [hm] at h0; cases h0)
      | (simp only [hm, Bool.not_false, ↓reduceIte] at h0; cases h0)
      | simp [hm] at h0
    · rfl
  · intro a v q hq
    simp only [bind, Except.bind, pure, Except.pure] at hq
    repeat' split at hq
    all_goals first | (cases hq; done) | (cases hq; exact ⟨⟨by assumption, by assumption⟩, rfl⟩)

theorem setupChecks_recSetup {I : Flock.Witness.Inputs} {rd : Reads} {sts : Array Flock.Stmt}
    (h : setupChecks I rd sts = .ok ()) :
    Flock.RecSetup.checks none none I.tags I.tables sts I.circuitFile I.partition I.program rd.registered none none
      none = .ok () := by
  unfold setupChecks at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals first | (cases h; done) | assumption

theorem recChecks_parts {rd : Reads} {sts : Array Flock.Stmt} (h : recChecks rd sts = .ok ()) :
    sts.forM (Flock.RecCoins.check · rd.registered (Flock.RecCoins.Source.atC3 none none none)
        (Flock.RecSetup.c3 none none)) = .ok () ∧
      sts.forM (Flock.RecVStar.checkPubRows · rd.registered none none) = .ok () := by
  unfold recChecks at h
  obtain ⟨u1, h1, h⟩ := bind_ok h
  obtain ⟨u2, h2, _⟩ := bind_ok h
  cases u1
  cases u2
  exact ⟨h1, h2⟩

theorem readsOf_some {j : Json} {rd : Reads} (h : readsOf (some j) = .ok rd) :
    ∃ own, Flock.Registered.ofJson j = .ok own ∧ rd.registered = some own := by
  simp only [readsOf, bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals first | (cases h; done) | (cases h; exact ⟨_, by assumption, rfl⟩)

theorem noIdentifier_of {st : Flock.Stmt} (h : checkIdentifier st = .ok ()) :
    FlockDeploy.noIdentifier st = .ok () := by
  cases hj : st.pub.header.getObjVal? "identifier" with
  | error e =>
    unfold FlockDeploy.noIdentifier
    first | (rw [hj]; done) | (rw [hj]; rfl) | simp [hj]
  | ok j =>
    exfalso
    unfold checkIdentifier at h
    first | rw [hj] at h | simp only [hj] at h
    try dsimp only at h
    cases j <;> (try dsimp only at h) <;>
      first
      | (cases h; done)
      | (split at h <;> cases h)
      | (simp only [bind, Except.bind, pure, Except.pure] at h; repeat' split at h; all_goals cases h)

/-! ## The deployed verifier's setup of the statement -/

/-- The deployed verifier's inputs at a `verify --zk` statement of the `verity/flock-circuit` kind on one table, with
no work, stratified, closure or public-inputs file. -/
def insE (tags0 : Flock.Tags) (circuit publicFile : ByteArray) (tables : List (String × ByteArray)) (part : Json)
    (program : ByteArray) (draw : Option Json) (reg : Option (Array Flock.Registered.Port)) : FlockDeploy.Ins :=
  { stmtTags := tags0, registered := reg, read := {
      I := { tags := tags0.withCoins (.tree (Flock.Zk.coinSpec (tags0.tableNames 1))), circuitFile := circuit,
             publicFile, coins := .tree (Flock.Zk.coinSpec (tags0.tableNames 1)), count := 1, tables,
             partition := some part, program := some program, work := none, stratified := none, closure := none }
      zk := true, draw, publicInputs := none, registered := reg.isSome } }

theorem ins_eq {w : FlockDeploy.Raw} {tags0 : Flock.Tags} {d : Option Json} {part : Json} {prog : ByteArray}
    {reg : Option (Array Flock.Registered.Port)}
    (ht : Flock.Tags.find? w.statement = some tags0) (hpl : tags0.plainLeaves = false) (hz : w.zk = true)
    (hos : w.osCoins = false) (hd : w.draw.mapM (FlockDeploy.jsonOf "draw") = .ok d)
    (hpart : w.partition.mapM (FlockDeploy.jsonOf "partition") = .ok (some part)) (hwt : w.workTable = none)
    (hcl : w.closure = none) (hwk : w.work = none) (hcount : w.count = 1) (hprog : w.program = some prog)
    (hstr : w.stratified = none) (hpi : w.publicInputs = none)
    (hreg : (w.registered = none ∧ reg = none) ∨ ∃ b j own, w.registered = some b ∧
      FlockDeploy.jsonOf "registered reads" b = .ok j ∧ Flock.Registered.ofJson j = .ok own ∧ reg = some own) :
    w.ins = .ok (insE tags0 w.circuit w.publicFile w.tables part prog (d.map Flock.norm) reg) := by
  unfold FlockDeploy.Raw.ins
  rcases hreg with ⟨hwr, rfl⟩ | ⟨b, j, own, hwr, hj, hown, rfl⟩
  · simp only [ht, hpl, hz, hos, hd, hpart, hwt, hcl, hwk, hcount, hprog, hstr, hpi, hwr, Bool.true_and,
      Bool.and_false, Bool.false_eq_true, ↓reduceIte, Option.mapM_none, Option.mapM_some, ok_bind, pure_bind',
      map_ok']
    try rfl
  · simp only [ht, hpl, hz, hos, hd, hpart, hwt, hcl, hwk, hcount, hprog, hstr, hpi, hwr, hj, hown, Bool.true_and,
      Bool.and_false, Bool.false_eq_true, ↓reduceIte, Option.mapM_none, Option.mapM_some, ok_bind, pure_bind',
      map_ok']
    try rfl

theorem insOf_of {x : Msg} {w : FlockDeploy.Raw} {s : FlockDeploy.Ins} (hdec : FlockDeploy.decode x = some w)
    (hz : w.zk = true) (hins : w.ins = .ok s) : FlockDeploy.insOf true x = some s := by
  unfold FlockDeploy.insOf
  rw [hdec]
  simp [hz, hins, Except.toOption]

theorem setupOf_of {x : Msg} {s : FlockDeploy.Ins} {spec : Flock.SessionSpec} {sts : Array Flock.Stmt}
    (hi : FlockDeploy.insOf true x = some s) (hs : s.setup = .ok (spec, sts)) :
    FlockDeploy.setupOf true x = some (s, spec, sts) := by
  unfold FlockDeploy.setupOf
  rw [hi]
  simp [hs, Except.toOption]

theorem insE_tags (tags0 : Flock.Tags) (circuit publicFile : ByteArray) (tables : List (String × ByteArray))
    (part : Json) (program : ByteArray) (draw : Option Json) (reg : Option (Array Flock.Registered.Port)) :
    (insE tags0 circuit publicFile tables part program draw reg).read.tags =
      Flock.Witness.zkTags { tags := tags0, circuitFile := circuit, publicFile, count := 1, tables,
        partition := some part, program := some program } := by
  simp only [insE, Flock.Readable.Read.tags, FlockSoundness.Discharge.ZkExec.zkTags,
    FlockSoundness.Discharge.ZkExec.zkCoins, Flock.Witness.zkTags, Flock.Witness.zkCoins,
    FlockSoundness.Discharge.ZkExec.withCoins_tree, ↓reduceIte]

theorem insE_coins (tags0 : Flock.Tags) (circuit publicFile : ByteArray) (tables : List (String × ByteArray))
    (part : Json) (program : ByteArray) (draw : Option Json) (reg : Option (Array Flock.Registered.Port)) :
    (insE tags0 circuit publicFile tables part program draw reg).read.coins =
      Flock.Witness.zkCoins { tags := tags0, circuitFile := circuit, publicFile, count := 1, tables,
        partition := some part, program := some program } := by
  simp only [insE, Flock.Readable.Read.coins, FlockSoundness.Discharge.ZkExec.zkCoins, Flock.Witness.zkCoins,
    FlockSoundness.Discharge.ZkExec.withCoins_tree, ↓reduceIte]

theorem ins_setup_ok {s : FlockDeploy.Ins} {spec : Flock.SessionSpec} {sts : Array Flock.Stmt}
    (h1 : s.read.tags.hm96Rows = true)
    (h2 : Flock.RecSetup.checkInner none none (pure (none, none)) = .ok ())
    (h3 : s.read.setup = .ok (spec, sts))
    (h4 : ∀ st ∈ sts, Flock.checkRegistered st s.registered = .ok () ∧ FlockDeploy.noIdentifier st = .ok ())
    (h5 : Flock.Registered.checkFull s.registered s.read.draw = .ok ())
    (h6 : Flock.RecSetup.checks none none s.stmtTags s.read.I.tables sts s.read.I.circuitFile s.read.I.partition
      s.read.I.program s.registered none none none = .ok ())
    (h7 : s.read.zk = true) (h8 : sts.forM Flock.Zk.shapeOk = .ok ())
    (h9 : sts.forM (Flock.RecCoins.check · s.registered (Flock.RecCoins.Source.atC3 none none none)
      (Flock.RecSetup.c3 none none)) = .ok ())
    (h10 : sts.forM (Flock.RecVStar.checkPubRows · s.registered none none) = .ok ())
    (h11 : s.read.draw.isSome = true →
      Flock.ProvedScope.drawLaw true s.read.tags false s.read.draw = .ok ()) :
    s.setup = .ok (spec, sts) := by
  unfold FlockDeploy.Ins.setup
  cases hd : s.read.draw.isSome
  · simp only [h1, h2, h3, h5, h6, h7, h8, h9, h10, hd, Bool.not_true, Bool.false_eq_true, ↓reduceIte, ok_bind,
      pure_bind']
    rw [forIn_arr_ok sts]
    · first | rfl | (simp only [ok_bind, pure_bind']; done) | simp only [ok_bind, pure_bind', h5, h6, h8, h9, h10]
    · intro st hst
      simp only [(h4 st hst).1, (h4 st hst).2, ok_bind, pure_bind']
      try rfl
  · simp only [h1, h2, h3, h5, h6, h7, h8, h9, h10, hd, h11 hd, Bool.not_true, Bool.false_eq_true, ↓reduceIte,
      ok_bind, pure_bind']
    rw [forIn_arr_ok sts]
    · first | rfl | (simp only [ok_bind, pure_bind']; done) |
        simp only [ok_bind, pure_bind', h5, h6, h8, h9, h10, h11 hd]
    · intro st hst
      simp only [(h4 st hst).1, (h4 st hst).2, ok_bind, pure_bind']
      try rfl

/-- The deployed verifier's checks after its verdict, from their parts. -/
theorem checks_ok {zk : Bool} {s : FlockDeploy.Ins} {spec : Flock.SessionSpec} {x : Msg}
    {link : List ByteArray × List ByteArray} {c₁ : FlockDeploy.Coin} {mids : List (Msg × FlockDeploy.Coin)}
    {f : FlockDeploy.Final} (h1 : f.stmt = x)
    (h2 : FlockDeploy.drawEq (FlockSoundness.Discharge.Exec.recordDraw f.record) s.read.draw = true)
    (h3 : (do FlockDeploy.custody spec (← FlockDeploy.liveOf spec f.record) link c₁ mids : V Unit) = .ok ())
    (h4 : s.read.check = .ok ())
    (h5 : s.read.draw.isNone = true → Flock.ProvedScope.drawLaw zk s.read.tags false none = .ok ())
    (h6 : Flock.RecordText.check f.record = .ok ()) :
    FlockDeploy.checks zk s spec x link c₁ mids f = .ok () := by
  obtain ⟨lv, hlv, hcu⟩ := bind_ok h3
  have hcu' : FlockDeploy.custody spec lv link c₁ mids = .ok () := hcu
  subst h1
  unfold FlockDeploy.checks
  cases hn : s.read.draw.isNone
  · simp only [Flock.Record.check, beq_self_eq_true, h2, hlv, hcu', h4, h6, hn, ↓reduceIte, Bool.false_eq_true,
      ok_bind, pure_bind']
  · simp only [Flock.Record.check, beq_self_eq_true, h2, hlv, hcu', h4, h6, hn, h5 hn, ↓reduceIte,
      Bool.false_eq_true, ok_bind, pure_bind']

theorem exec_recordDraw (r : String) :
    FlockSoundness.Discharge.Exec.recordDraw r = Flock.Witness.recordDraw r := by
  first
  | rfl
  | (unfold FlockSoundness.Discharge.Exec.recordDraw Flock.Witness.recordDraw; rfl)

theorem final_tag {a : Msg} {F : FlockDeploy.Final} (h : FlockDeploy.finalOf a = some F) : ∃ b, a = 1 :: b := by
  unfold FlockDeploy.finalOf at h
  split at h
  · exact ⟨_, rfl⟩
  · cases h

theorem link_tag {a : Msg} {l : List ByteArray × List ByteArray} (h : FlockDeploy.linkOf a = some l) :
    ∃ b, a = 0 :: b := by
  unfold FlockDeploy.linkOf at h
  split at h
  · exact ⟨_, rfl⟩
  · cases h

/-- The proof files `verify --zk` read are the ones the blobs show. -/
theorem files_eq {bs : Core.Blobs} {s : Session} {spec : Flock.SessionSpec} {tabs : Array Flock.Setup}
    {x0 : Json × Json} {proofs fp : Array ByteArray} (h : filesOf bs s (.ok (spec, tabs, x0)) = .ok proofs)
    (hsz : fp.size = spec.repStreams.size)
    (hk : ∀ k (hk : k < spec.repStreams.size), ∃ d,
      (s.reps.find? (·.1 == s!"{spec.repStreams[k].2.2.replace "/" "."}.bin")).map (·.2) = some d ∧
      ∃ hk' : k < fp.size, bs.get d s!"session files: {spec.repStreams[k].2.2.replace "/" "."}.bin" = .ok fp[k]) :
    proofs = fp := by
  unfold filesOf at h
  obtain ⟨⟨spec', tabs', x'⟩, he, h⟩ := bind_ok h
  cases he
  obtain ⟨hps, hall⟩ := FlockSoundness.Discharge.Exec.array_mapM_ok_getElem h
  refine Array.ext (by rw [hps, hsz]) fun k hk1 hk2 => ?_
  have hk0 : k < spec.repStreams.size := hps ▸ hk1
  have hg := hall k hk0 hk1
  obtain ⟨d, hd, hk', hget⟩ := hk k hk0
  generalize hq : spec.repStreams[k] = q at hg hd hget
  rcases q with ⟨a1, a2, w⟩
  dsimp only at hg hd hget
  obtain ⟨⟨n1, d1⟩, hfd, hd1⟩ := Option.map_eq_some_iff.1 hd
  dsimp only at hd1
  subst hd1
  first | rw [hfd] at hg | simp only [hfd] at hg
  try dsimp only at hg
  exact Except.ok.inj (hg.symm.trans (get_what hget _))

/-! ## The run -/

section Acc

variable {Ω : Type} (H : Msg → Msg) (P : Claims.Prover Ω) (x : Msg) (ω : Ω)

/-- **A run that accepts a round ends accepted.** -/
theorem stepWith_acc_out (vn : List Msg → Claims.OC Out) :
    ∀ (f i : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t : Claims.Trace) (a : Msg),
      (Claims.stepWith H P x ω vn f hist as t).msgs[t.msgs.length + i]? = some (a, .acc) →
      (Claims.stepWith H P x ω vn f hist as t).out = .acc
  | 0, i, _, _, t, a, h => by simp [Claims.stepWith] at h
  | f + 1, i, hist, as, t, a, h => by
    rcases hvb : (vn (as ++ [((P.next x hist ω).eval H).1])).eval H with ⟨o, lo⟩
    cases o with
    | msg mm =>
      rw [RecDeploy.stepWith_succ_msg H P x ω vn f hist as t hvb] at h ⊢
      cases i with
      | zero =>
        obtain ⟨rest, hrest⟩ := (RecDeploy.stepWith_mono H P x ω vn f _ _ _).1
        rw [← hrest] at h
        simp at h
      | succ i =>
        exact stepWith_acc_out vn f i _ _ _ a (by simpa [Nat.add_assoc, Nat.add_comm 1 i] using h)
    | acc =>
      rw [RecDeploy.stepWith_succ_acc H P x ω vn f hist as t hvb]
    | rej =>
      rw [RecDeploy.stepWith_succ_rej H P x ω vn f hist as t hvb] at h
      cases i <;> simp at h

end Acc

theorem run_acc_of_get {Ω : Type} (P : Claims.Prover Ω) {xS : Msg} (ω : Ω)
    (r : Fin ((FlockDeploy.V true).k xS) → FlockDeploy.Coin) {i : ℕ} {a : Msg}
    (h : (Claims.run FlockDeploy.sha512 P (FlockDeploy.V true) xS ω r).msgs[i]? = some (a, .acc)) :
    (Claims.run FlockDeploy.sha512 P (FlockDeploy.V true) xS ω r).out = .acc :=
  stepWith_acc_out FlockDeploy.sha512 P xS ω _ _ i [] [] ⟨[], [], [], .rej⟩ a
    (by first | simpa using h | (simp only [List.length_nil, Nat.zero_add]; exact h))

/-- The answer to a run's last message is the verifier's on all of the run's messages and coins. -/
theorem run_last {Ω : Type} (P : Claims.Prover Ω) {xS : Msg} (ω : Ω)
    (r : Fin ((FlockDeploy.V true).k xS) → FlockDeploy.Coin) {a : Msg} {o : Out}
    (h : (Claims.run FlockDeploy.sha512 P (FlockDeploy.V true) xS ω r).msgs.getLast? = some (a, o)) :
    o = (((FlockDeploy.V true).next xS
      ((Claims.run FlockDeploy.sha512 P (FlockDeploy.V true) xS ω r).msgs.map Prod.fst)
      ((List.range (Claims.run FlockDeploy.sha512 P (FlockDeploy.V true) xS ω r).msgs.length).map
        (Claims.coinSeq r))).eval FlockDeploy.sha512).1 := by
  obtain ⟨ys, hys⟩ := List.getLast?_eq_some_iff.1 h
  have hlt : ys.length < (Claims.run FlockDeploy.sha512 P (FlockDeploy.V true) xS ω r).msgs.length := by
    rw [hys]
    simp
  obtain ⟨new, hnew, hget⟩ := RecDeploy.run_get FlockDeploy.sha512 P (FlockDeploy.V true) xS ω r ys.length hlt
  rw [hys] at hnew hget ⊢
  rw [List.take_left] at hnew
  subst hnew
  have hgl : (ys ++ [(a, o)])[ys.length]? = some (a, o) := by simp
  rw [hgl] at hget
  obtain ⟨ha, ho⟩ := Prod.mk.inj (Option.some.inj hget)
  subst ha ho
  simp only [List.map_append, List.map_cons, List.map_nil, List.length_append, List.length_singleton]

/-- A round before a run's last is answered with its coin's bytes. -/
theorem run_coin {Ω : Type} (P : Claims.Prover Ω) {xS : Msg} (ω : Ω)
    (r : Fin ((FlockDeploy.V true).k xS) → FlockDeploy.Coin) (i : ℕ)
    (hi : i + 1 < (Claims.run FlockDeploy.sha512 P (FlockDeploy.V true) xS ω r).msgs.length) :
    ((Claims.run FlockDeploy.sha512 P (FlockDeploy.V true) xS ω r).msgs[i]?).map (fun m => RecDeploy.outCoin m.2) =
      some (Claims.coinSeq r i) := by
  obtain ⟨m, hm⟩ := Claims.Recursion.stepWith_msg_of_lt FlockDeploy.sha512 P xS ω
    (fun as => (FlockDeploy.V true).next xS as ((List.range as.length).map (Claims.coinSeq r)))
    ((FlockDeploy.V true).k xS) [] [] ⟨[], [], [], .rej⟩ i (Nat.zero_le _) hi
  change ((Claims.run FlockDeploy.sha512 P (FlockDeploy.V true) xS ω r).msgs[i]?).map Prod.snd = some (.msg m) at hm
  obtain ⟨new, -, hget⟩ := RecDeploy.run_get FlockDeploy.sha512 P (FlockDeploy.V true) xS ω r i (by omega)
  rw [hget] at hm ⊢
  simp only [Option.map_some, Option.some.injEq] at hm ⊢
  have hr : List.range (i + 1) = List.range i ++ [i] := by
    first | exact List.range_succ | exact List.range_succ (n := i) | simp [List.range_succ]
  have hm' := hm
  rw [hr, List.map_append, List.map_cons, List.map_nil] at hm'
  have hb := Claims.Bridge.answersCoin_deploy true FlockDeploy.sha512 xS _ _ _ m hm'
  rw [hm, hb]
  exact RecDeploy.coinOf_bytes _

/-- **The deployed verifier accepts the session's run** at SHA-512, where the statement sets up, the run's last
message is a `Final` after its link message, `Zk.verify` accepts its record and proof files, and the checks after the
verdict pass wherever the verifier decides on the run's messages and the coins its answers show. -/
theorem flock_acc {Ω : Type} (P : Claims.Prover Ω) {xS : Msg} (ω : Ω)
    (r : Fin ((FlockDeploy.V true).k xS) → FlockDeploy.Coin) {s : FlockDeploy.Ins} {spec : Flock.SessionSpec}
    {sts : Array Flock.Stmt} {f : FlockDeploy.Final}
    (hset : FlockDeploy.setupOf true xS = some (s, spec, sts))
    (hlast : FlockDeploy.lastFinal (Claims.run FlockDeploy.sha512 P (FlockDeploy.V true) xS ω r).seen = some f)
    (hlink : ∃ link, ((Claims.run FlockDeploy.sha512 P (FlockDeploy.V true) xS ω r).seen.msgs.head?).bind
      (fun m => FlockDeploy.linkOf m.1) = some link)
    (hv : Flock.Zk.verify spec (FlockDeploy.tabsOf sts) f.record f.proofs = .ok ())
    (hchk : ∀ link c₁ mids,
      FlockDeploy.act true xS ((Claims.run FlockDeploy.sha512 P (FlockDeploy.V true) xS ω r).msgs.map Prod.fst)
          ((Claims.run FlockDeploy.sha512 P (FlockDeploy.V true) xS ω r).msgs.map fun m => RecDeploy.outCoin m.2) =
        .decide s spec sts link c₁ mids f →
      FlockDeploy.checks true s spec xS link c₁ mids f = .ok ()) :
    (Claims.run FlockDeploy.sha512 P (FlockDeploy.V true) xS ω r).out = .acc := by
  obtain ⟨link, hl1⟩ := hlink
  have hlast' : ((Claims.run FlockDeploy.sha512 P (FlockDeploy.V true) xS ω r).msgs.getLast?).bind
      (fun p => FlockDeploy.finalOf p.1) = some f := hlast
  have hl1' : ((Claims.run FlockDeploy.sha512 P (FlockDeploy.V true) xS ω r).msgs.head?).bind
      (fun m => FlockDeploy.linkOf m.1) = some link := hl1
  obtain ⟨⟨pa, po⟩, hq, hpf⟩ := Option.bind_eq_some_iff.1 hlast'
  obtain ⟨⟨la, lo⟩, hh, hlk⟩ := Option.bind_eq_some_iff.1 hl1'
  have hpf' : FlockDeploy.finalOf pa = some f := hpf
  have hlk' : FlockDeploy.linkOf la = some link := hlk
  have ho := run_last P ω r hq
  obtain ⟨Lr, hM⟩ : ∃ Lr, (Claims.run FlockDeploy.sha512 P (FlockDeploy.V true) xS ω r).msgs = (la, lo) :: Lr := by
    cases hc : (Claims.run FlockDeploy.sha512 P (FlockDeploy.V true) xS ω r).msgs with
    | nil => rw [hc] at hh; cases hh
    | cons m Lr =>
      rw [hc] at hh
      cases hh
      first | exact ⟨Lr, rfl⟩ | exact ⟨Lr, hc⟩
  have hq' := hq
  rw [hM] at hq'
  have hLr : Lr ≠ [] := by
    rintro rfl
    have hpa : pa = la := by
      have e := congrArg (Option.map Prod.fst) hq'
      simpa using e.symm
    subst hpa
    obtain ⟨b1, hb1⟩ := final_tag hpf'
    obtain ⟨b0, hb0⟩ := link_tag hlk'
    exact absurd (List.cons.inj (hb1.symm.trans hb0)).1 (by decide)
  have hLpos : 0 < Lr.length := List.length_pos_iff.2 hLr
  have hc0 := run_coin P ω r 0 (by rw [hM, List.length_cons]; omega)
  rw [hM] at hc0
  have hc1 : RecDeploy.outCoin lo = Claims.coinSeq r 0 := by simpa using hc0
  have hdrop : (Lr.map fun m => RecDeploy.outCoin m.2).dropLast =
      ((List.range Lr.length).map fun j => Claims.coinSeq r (j + 1)).dropLast := by
    apply List.ext_getElem?
    intro j
    simp only [List.getElem?_dropLast, List.length_map, List.length_range]
    split_ifs with hj
    · have hcj := run_coin P ω r (j + 1) (by rw [hM, List.length_cons]; omega)
      rw [hM, List.getElem?_cons_succ] at hcj
      (rw [List.getElem?_map, List.getElem?_map, List.getElem?_range (by omega)]) <;>
        first | exact hcj | (rw [hcj]) | simpa using hcj
    · rfl
  have hl : (la :: Lr.map Prod.fst).getLast? = some pa := by
    have e : la :: Lr.map Prod.fst = ((la, lo) :: Lr).map Prod.fst := rfl
    (rw [e, List.getLast?_map, hq']) <;> rfl
  have hact' := FlockComplete.act_final_any (zk := true) (c₁ := RecDeploy.outCoin lo)
    (crest := Lr.map fun m => RecDeploy.outCoin m.2) hset hl hpf' hlk'
  have hchecks := hchk link _ _ (by rw [hM]; exact hact')
  rw [hc1, hdrop] at hchecks
  have hact := FlockComplete.act_final_any (zk := true) (c₁ := Claims.coinSeq r 0)
    (crest := (List.range Lr.length).map fun j => Claims.coinSeq r (j + 1)) hset hl hpf' hlk'
  have hnext := (FlockDeploy.next_decide_sha512 hact).2
    ⟨by first | exact hv | (rw [if_pos rfl]; exact hv) | simpa using hv, hchecks⟩
  have hcs : (List.range ((la, lo) :: Lr).length).map (Claims.coinSeq r) =
      Claims.coinSeq r 0 :: (List.range Lr.length).map (fun j => Claims.coinSeq r (j + 1)) := by
    (rw [List.length_cons, List.range_succ_eq_map, List.map_cons, List.map_map]) <;> rfl
  rw [hM, hcs] at ho
  have hpo : po = .acc := ho.trans hnext
  have hget : (Claims.run FlockDeploy.sha512 P (FlockDeploy.V true) xS ω r).msgs[Lr.length]? = some (pa, .acc) := by
    have e1 := hq
    rw [List.getLast?_eq_getElem?, hM, List.length_cons, Nat.add_sub_cancel] at e1
    rw [hM, e1, hpo]
  exact run_acc_of_get P ω r hget

/-- **The session's run accepts**: where v2 accepted on the blobs and they show a session of the evidence holding its
live coins, the deployed verifier's run on that session accepts. -/
theorem flock_of_shows {p : Params} {c : OneStage.Commitment} {e : Evidence} {blobs : Array ByteArray}
    {pub : OneStage.Public} {prof : Core.Integrity}
    (hacc : checkBlobs p c e (Core.Blobs.ofArray blobs) = .accept pub prof) {s : Session} (hs : s ∈ e.sessions)
    {Ω : Type} (P : Claims.Prover Ω) (xS : Msg) (ω : Ω) (r : Fin ((FlockDeploy.V true).k xS) → FlockDeploy.Coin)
    (hS : Shows p.own e.pub blobs s xS (Claims.run FlockDeploy.sha512 P (FlockDeploy.V true) xS ω r).seen)
    (hL : LiveCoins xS (Claims.run FlockDeploy.sha512 P (FlockDeploy.V true) xS ω r).seen) :
    (Claims.run FlockDeploy.sha512 P (FlockDeploy.V true) xS ω r).out = .acc := by
  obtain ⟨tags0, circuitB, pubB, rd, ht, hpl, hcB, hpB, hrd, ck, b, record, hb, hu, hver⟩ := session_verdict hacc hs
  obtain ⟨f, hlast, hstmt, hN, hrec, hfiles⟩ := hS
  have hbf : b = f.record.toUTF8 := Except.ok.inj (hb.symm.trans hrec)
  subst hbf
  rw [FlockComplete.fromUTF8?_toUTF8] at hu
  cases hu
  obtain ⟨proofs, spec, tabs, x0, hf, hst, hz, hsc, hdl, hrt⟩ := verdictOf_parts hver
  obtain ⟨sts, hsa, htabs, hdl2⟩ := setupAt_parts hst
  subst htabs
  obtain ⟨hin, hbso, hschk, hshape, hrc⟩ := sessionAt_parts hsa
  have hsetT := bso_ok hbso
  obtain ⟨hm96, hloop, hfull⟩ := buildSessionOf_parts hbso
  have hrecs := setupChecks_recSetup hschk
  obtain ⟨hcoins, hpubrows⟩ := recChecks_parts hrc
  obtain ⟨w, hdec, hstw, hzw, hosw, hcw, hpw, htw, hcnt, hdw, hpartw, hprogw, hwk, hwt, hstr, hcl, hregw, hpi⟩ := hN
  have hcB' : circuitB = w.circuit := Except.ok.inj (hcB.symm.trans hcw)
  have hpB' : pubB = w.publicFile := Except.ok.inj (hpB.symm.trans hpw)
  subst hcB' hpB'
  obtain ⟨d, hd, hdn⟩ := map_ok hdw
  obtain ⟨pb, hpw', hpbj⟩ := Option.map_eq_some_iff.1 hpartw
  have hpart : w.partition.mapM (FlockDeploy.jsonOf "partition") = .ok (some p.own.partition) := by
    rw [hpw']
    simp only [Option.mapM_some, hpbj, map_ok']
  have hregE : (w.registered = none ∧ rd.registered = none) ∨ ∃ b j own, w.registered = some b ∧
      FlockDeploy.jsonOf "registered reads" b = .ok j ∧ Flock.Registered.ofJson j = .ok own ∧
      rd.registered = some own := by
    cases hpr : p.own.registered with
    | none =>
      rw [hpr] at hregw hrd
      left
      refine ⟨?_, ?_⟩
      · cases hwr : w.registered with
        | none => rfl
        | some b0 => rw [hwr] at hregw; cases hregw
      · cases hrd
        rfl
    | some j =>
      rw [hpr] at hregw hrd
      right
      obtain ⟨bj, hwb, hjb⟩ := Option.map_eq_some_iff.1 hregw
      obtain ⟨own, hown, hrdreg⟩ := readsOf_some hrd
      exact ⟨bj, j, own, hwb, hjb, hown, hrdreg⟩
  have ht' : Flock.Tags.find? w.statement = some tags0 := by rw [hstw]; exact ht
  have hins := ins_eq ht' hpl hzw hosw hd hpart hwt hcl hwk hcnt hprogw hstr hpi hregE
  rw [hdn, htw] at hins
  have hsetup := ins_setup_ok
    (s := insE tags0 w.circuit w.publicFile (tablesOf (Core.Blobs.ofArray blobs)) p.own.partition p.own.program
      (Flock.Witness.recordDraw f.record) rd.registered)
    (by rw [insE_tags]; exact hm96) hin
    (by unfold Flock.Readable.Read.setup; rw [insE_tags, insE_coins]; exact hsetT)
    (fun st hst => ⟨(hloop st hst).1, noIdentifier_of (hloop st hst).2⟩) hfull hrecs rfl hshape hcoins hpubrows
    (fun hsm => by rw [insE_tags]; exact hdl2 hsm)
  have hset := setupOf_of (insOf_of hdec hzw hins) hsetup
  rw [hst] at hf
  obtain ⟨hfsz, hfk⟩ := hfiles _ _ _ hset
  have hpf := files_eq hf hfsz hfk
  subst hpf
  have hchkS : (insE tags0 w.circuit w.publicFile (tablesOf (Core.Blobs.ofArray blobs)) p.own.partition
      p.own.program (Flock.Witness.recordDraw f.record) rd.registered).read.check = .ok () := by
    unfold Flock.Readable.Read.check
    rw [Flock.ProvedScope.check_eq, insE_tags]
    exact hsc
  have h2 : FlockDeploy.drawEq (FlockSoundness.Discharge.Exec.recordDraw f.record)
      (insE tags0 w.circuit w.publicFile (tablesOf (Core.Blobs.ofArray blobs)) p.own.partition p.own.program
        (Flock.Witness.recordDraw f.record) rd.registered).read.draw = true := by
    show FlockDeploy.drawEq (FlockSoundness.Discharge.Exec.recordDraw f.record)
      (Flock.Witness.recordDraw f.record) = true
    rw [exec_recordDraw]
    exact FlockDeploy.drawEq_refl _
  have h5 : (insE tags0 w.circuit w.publicFile (tablesOf (Core.Blobs.ofArray blobs)) p.own.partition p.own.program
        (Flock.Witness.recordDraw f.record) rd.registered).read.draw.isNone = true →
      Flock.ProvedScope.drawLaw true (insE tags0 w.circuit w.publicFile (tablesOf (Core.Blobs.ofArray blobs))
        p.own.partition p.own.program (Flock.Witness.recordDraw f.record) rd.registered).read.tags false none =
        .ok () := by
    intro hn
    have hn' : Flock.Witness.recordDraw f.record = none := Option.isNone_iff_eq_none.1 hn
    have hlaw := hdl hn
    rw [hn'] at hlaw
    rw [insE_tags]
    exact hlaw
  exact flock_acc P ω r hset hlast hL.1 hz
    (fun link c₁ mids hact => checks_ok hstmt h2 (hL.2 _ _ _ _ _ _ _ hact) hchkS h5 hrt)

end Run

/-- **ONE-3t's property: one-stage v2's acceptance is the sampled model's.** Suppose one-stage v2 accepts
`(p, c, e, blobs)`, and the deployed sampled verifier (`SamplingDeploy.deploy rel cd`, whose checked units' sessions
are Flock runs, `SessionRuns`) checks the unit `u` at the coins `r`. If `u`'s session in the run's view shows one of
the evidence's sessions `s` (`Shows`) and holds its live coins (`LiveCoins`), the run accepts. -/
theorem acc_of_accepts (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (cd : SamplingDeploy.Code)
    (hruns : Claims.Sampling.SessionRuns FlockDeploy.sha512 (SamplingDeploy.deploy rel cd))
    (p : Params) (c : OneStage.Commitment) (e : Evidence) (blobs : Array ByteArray) {pub : OneStage.Public}
    {prof : Core.Integrity} (hacc : checkBlobs p c e (Core.Blobs.ofArray blobs) = .accept pub prof)
    (x : Msg) {Ω : Type} (P : Claims.Prover Ω) (ω : Ω)
    (r : Fin ((SamplingDeploy.deploy rel cd).V.k x) → FlockDeploy.Coin) (u : ℕ × ℕ)
    (hu : ∃ q : Fin (Claims.Sampling.nPU (SamplingDeploy.deploy rel cd) x),
      ((SamplingDeploy.deploy rel cd).checked x r)[q.val]? = some u)
    (s : Session) (hs : s ∈ e.sessions)
    (hshow : ∀ xS v, (SamplingDeploy.deploy rel cd).session x u
        (Claims.run FlockDeploy.sha512 P (SamplingDeploy.deploy rel cd).V x ω r).seen = some (xS, v) →
      Shows p.own e.pub blobs s xS v ∧ LiveCoins xS v) :
    (Claims.run FlockDeploy.sha512 P (SamplingDeploy.deploy rel cd).V x ω r).out = .acc := by
  obtain ⟨xS, emb, P', -, -, hiff, hsess⟩ := hruns x Ω P ω r u hu
  rw [hiff r (fun _ _ => rfl)]
  obtain ⟨hsome, -⟩ := hsess r (fun _ _ _ => rfl)
  obtain ⟨hS, hL⟩ := hshow _ _ hsome
  exact Run.flock_of_shows hacc hs P' xS () (fun i => r (emb i)) hS hL

/-- **`acc_of_accepts`' conclusion is not automatic**: the deployed Flock verifier, whose run `SessionRuns` reads the
sampled run's acceptance off, refuses a statement that does not decode, whatever the prover sends. -/
theorem acc_of_accepts_nontrivial {Ω : Type} (P : Claims.Prover Ω) (ω : Ω)
    (r : Fin ((FlockDeploy.V true).k []) → FlockDeploy.Coin) :
    (Claims.run FlockDeploy.sha512 P (FlockDeploy.V true) [] ω r).out ≠ .acc := by
  intro h
  obtain ⟨as', ha⟩ := RecDeploy.stepWith_acc FlockDeploy.sha512 P [] ω _ _ _ _ _ h
  have hdec : FlockDeploy.decode [] = none := rfl
  have hs : FlockDeploy.setupOf true [] = none := by
    first
    | rfl
    | simp [FlockDeploy.setupOf, FlockDeploy.insOf, hdec]
  have hact : FlockDeploy.act true [] as' ((List.range as'.length).map (Claims.coinSeq r)) = .rej := by
    unfold FlockDeploy.act
    rw [hs]
  change ((FlockDeploy.next true [] as' ((List.range as'.length).map (Claims.coinSeq r))).eval
    FlockDeploy.sha512).1 = .acc at ha
  rw [FlockDeploy.next_rej _ hact] at ha
  cases ha

end Properties.SampledProofs.OneStageZk
