module

public import Proofs.Properties.Verity.Sampled.Accepts
public import Proofs.Properties.Verity.Sampled.Held
public import Proofs.Flock.Bridge.ScopeGate
public import Proofs.Flock.Bridge.V5.Accepted

/-!
# Sampled proofs' acceptance is in the events V3 and V5 bound

`SampledKSLiveV3` bounds an event of the live audit, that `verify --zk` accepts the files and the transcript
(`Flock.Bridge.LiveAcceptsScoped`) and the run passes the registration's population (`Flock.Bridge.RunPopOk`), and
`SampledKSLiveV5` another (`Flock.Bridge.LiveAcceptsKSP`). The verifier of record is one-stage v2
(`Spot.OneStageZk.checkBlobs`), which runs `verify --zk`'s checks on the session's files itself. Where it
accepts, every outcome whose transcript is the session's record and proof files is in both events:

* **v2's verdict, whole** (`verdictOf_event`, `step_event`, `run_event`): a session's verdict accepts only where
  `Zk.verify` accepts, the files' ports and roots pass (`Flock.PortRoots.checkOf`), the files' form and the record's
  sample pass (`Flock.KsScope.checkOf`), and the population check passes.
* **v2's circuit is the one the events read** (`circuitOf_parse`, `statement_zkTags`): V5's event reads
  `ProvedScope.circuitOf`'s circuit, as v2's checks do, and at a statement with hidden outputs that is not templated,
  as v2's is, that circuit is `HmOut.parse`'s, which V3's (`PortRoots.check`, `KsScope.check`) reads.
* **What acceptance gives** (`Accepted`, `checkBlobsAt_event`, `checkBlobs_event`): at the session's inputs and
  transcript, `AcceptsZK`, `PortRoots.check`, `KsScope.check` at the record's sample, the program's proof blocks under
  the verifier's query the record's population `n` in order (`popHeld`), and every public file the session's file
  loads to of `n` proof blocks (`publicHeld`, `population_held`).
* **The events** (`Accepted.liveAcceptsScoped`, `Accepted.liveAcceptsKSP`, `Accepted.runPopOk`, `Accepted.v3`): at
  any registration and sample, and at V3's population, the public file's.

It is a statement about outcomes, not probabilities: that v2's acceptance in the period's game is no likelier than an
event of the live audit is the coupling (`CouplingTarget`), stated for V5's transcript event (`LiveAcceptsZKP`).
-/

@[expose] public section

namespace Verity.Sampled

open Lean (Json)
open FlockSoundness.Discharge
open FlockSoundness.Discharge.Recursion.Witness (toExec)
open FlockSoundness.Refine (ebind_ok)

/-! ## v2's verdict, whole -/

/-- **A session's verdict accepts only where every check `verify --zk` makes accepts**: `Zk.verify` on its proof files
at its session (`verdictOf_ok`), the files' ports and roots (`ksFiles`), the files' form and the sample `draw`
(`Flock.KsScope.checkOf`), and the population check last. -/
theorem verdictOf_event {I : Flock.Witness.Inputs} {scope ksFiles population : Flock.V Unit} {coinKeys : Array String}
    {record : String} {draw : Option Json} {st : _root_.Spot.OneStageZk.SetUp}
    {files : Flock.V (Array ByteArray)} {coinKey : Option String}
    (h : _root_.Spot.OneStageZk.verdictOf I scope ksFiles population coinKeys record draw st files coinKey
      = .ok ()) :
    (∃ proofs spec tabs rest, files = .ok proofs ∧ st = .ok (spec, tabs, rest) ∧
      Flock.Zk.verify spec tabs record proofs = .ok ()) ∧ ksFiles = .ok () ∧
      Flock.KsScope.checkOf (Flock.ProvedScope.circuitOf (Flock.Witness.zkTags I) I.circuitFile I.tables)
        (Flock.Witness.zkTags I) I.publicFile I.count draw = .ok () ∧ population = .ok () := by
  refine ⟨verdictOf_ok h, ?_⟩
  unfold _root_.Spot.OneStageZk.verdictOf at h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨⟨_, _, _⟩, -, h⟩ := ebind_ok h
  dsimp only at h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨⟨⟩, hk, h⟩ := ebind_ok h
  obtain ⟨⟨⟩, hs, h⟩ := ebind_ok h
  exact ⟨hk, hs, h⟩

/-- **One session's `step` after a run that accepted so far, where every session accepted**: `step_verified`'s record,
session and proof files, and the checks `verdictOf_event` reads at the record's own sample. -/
theorem step_event {I : Flock.Witness.Inputs} {rd : _root_.Spot.OneStageZk.Reads}
    {bs : _root_.Core.Blobs} {scope ksFiles population : Flock.V Unit} {programSha : Option String} {want : Json}
    {s : _root_.Spot.OneStageZk.RunFiles} {r : _root_.Spot.OneStageZk.Run}
    (h : _root_.Spot.OneStageZk.step I rd bs scope ksFiles population programSha want ⟨#[], #[], #[], true⟩ s
      = .ok r)
    (hall : r.all = true) :
    ∃ recB record spec tabs rest proofs, bs.get s.record s!"session {s.name}'s record" = .ok recB ∧
      String.fromUTF8? recB = some record ∧
      _root_.Spot.OneStageZk.setupAt I rd (Flock.Witness.recordDraw record) = .ok (spec, tabs, rest) ∧
      _root_.Spot.OneStageZk.filesOf bs s (.ok (spec, tabs, rest)) = .ok proofs ∧
      Flock.Zk.verify spec tabs record proofs = .ok () ∧ ksFiles = .ok () ∧
      Flock.KsScope.checkOf (Flock.ProvedScope.circuitOf (Flock.Witness.zkTags I) I.circuitFile I.tables)
        (Flock.Witness.zkTags I) I.publicFile I.count (Flock.Witness.recordDraw record) = .ok () ∧
      population = .ok () := by
  unfold _root_.Spot.OneStageZk.step at h
  obtain ⟨recB, hb, h⟩ := ebind_ok h
  dsimp only at h
  split at h
  · rename_i record hrec
    obtain ⟨_, -, h⟩ := ebind_ok h
    obtain ⟨⟨proofs, spec, tabs, rest, hf, hst, hv⟩, hk, hs, hp⟩ := verdictOf_event (push_ok h hall)
    rw [hst] at hf
    exact ⟨recB, record, spec, tabs, rest, proofs, hb, hrec, hst, hf, hv, hk, hs, hp⟩
  · cases h

/-- **v2's run on one session, where it accepted**, as `run_verified` with the files' ports and roots check it runs:
`Flock.PortRoots.checkOf` on the circuit v2's checks read and the public file. -/
theorem run_event {p : _root_.Spot.OneStageZk.Own} {s : _root_.Spot.OneStageZk.RunFiles}
    {pu : Option (Array ℕ)} {bs : _root_.Core.Blobs} {r : _root_.Spot.OneStageZk.Run}
    (h : _root_.Spot.OneStageZk.run p #[s] pu bs = .ok r) :
    ∃ tags0 circuitB pubB rd, Flock.Tags.find? _root_.Spot.OneStageZk.STATEMENT = some tags0 ∧
      bs.get p.circuit "circuit" = .ok circuitB ∧ bs.get p.pub "public file" = .ok pubB ∧
      _root_.Spot.OneStageZk.readsOf p.registered = .ok rd ∧
      ∃ scope programSha want,
        _root_.Spot.OneStageZk.step (_root_.Spot.OneStageZk.inputsOf tags0 p circuitB pubB bs) rd bs
          scope
          (Flock.PortRoots.checkOf
            (Flock.ProvedScope.circuitOf
              (Flock.Witness.zkTags (_root_.Spot.OneStageZk.inputsOf tags0 p circuitB pubB bs))
              (_root_.Spot.OneStageZk.inputsOf tags0 p circuitB pubB bs).circuitFile
              (_root_.Spot.OneStageZk.inputsOf tags0 p circuitB pubB bs).tables)
            (Flock.Witness.zkTags (_root_.Spot.OneStageZk.inputsOf tags0 p circuitB pubB bs)) pubB)
          (_root_.Spot.FilePart.check p.program (_root_.Spot.getD p.partition "query") pubB)
          programSha want ⟨#[], #[], #[], true⟩ s = .ok r := by
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
    have e : r' = r := by
      cases pu with
      | none => simpa only [pure, Except.pure, Except.ok.injEq] using h
      | some want =>
        repeat' (first | (obtain ⟨_, _, h⟩ := ebind_ok h) | (split at h))
        all_goals simpa only [pure, Except.pure, Except.ok.injEq] using h
    subst e
    exact ⟨tags0, circuitB, pubB, rd, htags, hc, hp, hrd, _, _, _, foldlM_single hr'⟩
  · cases h

/-- **v2's check at the coins `ξ` accepts only where the registration's population is held to the verifier's
program** (`popHeld`). -/
theorem checkBlobsAt_popHeld {p : _root_.Spot.OneStageZk.Params}
    {c : _root_.Spot.OneStage.Commitment} {ξ : ByteArray} {e : _root_.Spot.OneStageZk.Evidence}
    {bs : _root_.Core.Blobs} {pub : _root_.Spot.OneStage.Public} {prof : _root_.Core.Integrity}
    (h : _root_.Spot.OneStageZk.checkBlobsAt p c ξ e bs = .accept pub prof) :
    _root_.Spot.OneStageZk.popHeld p.base.verifier c.record = .ok () := by
  replace h := (_root_.Spot.OneStageZk.checkBlobsAt_accept h).1
  unfold _root_.Spot.OneStageZk.checkOneAt at h
  dsimp only at h
  split at h
  · cases h
  · rename_i r hr
    obtain ⟨_, -, hr⟩ := ebind_ok hr
    obtain ⟨_, -, hr⟩ := ebind_ok hr
    obtain ⟨⟨⟩, -, hr⟩ := ebind_ok hr
    obtain ⟨_, -, hr⟩ := ebind_ok hr
    obtain ⟨⟨⟩, hpop, -⟩ := ebind_ok hr
    exact hpop

/-- **`popHeld` passes only where the record names a population `n` that the program's proof blocks under the
query are, in order** (`Slots.popOk`). -/
theorem popHeld_ok {v : _root_.Spot.Registration.Verifier} {record : Json}
    (h : _root_.Spot.OneStageZk.popHeld v record = .ok ()) :
    ∃ n, _root_.Spot.nat? (_root_.Spot.getD record "population") = some n ∧
      _root_.Spot.Slots.popOk v.program v.query n = true := by
  unfold _root_.Spot.OneStageZk.popHeld at h
  split at h
  · cases h
  · rename_i n hn
    split at h
    · rename_i hp
      exact ⟨n, hn, hp⟩
    · cases h

/-! ## v2's circuit is the one the events read -/

/-- **At a statement with hidden outputs that is not templated, the circuit v2's checks read is `HmOut.parse`'s**,
which `Flock.PortRoots.check` and `Flock.KsScope.check` read. -/
theorem circuitOf_parse {t : Flock.Tags} (hh : t.hiddenOutputs = true) (ht : t.isTmpl = false) (f : ByteArray)
    (tb : List (String × ByteArray)) : Flock.ProvedScope.circuitOf t f tb = Flock.HmOut.parse t f tb := by
  simp only [Flock.ProvedScope.circuitOf, hh, ↓reduceIte, Flock.HmOut.parseCommit, ht, Bool.false_eq_true]

/-- **v2's statement under `--zk`'s tags has hidden outputs and is not templated**, at any mask and table count. -/
theorem statement_zkTags {t : Flock.Tags} (h : Flock.Tags.find? _root_.Spot.OneStageZk.STATEMENT = some t)
    (m k : ℕ) : (Flock.Zk.tags t m k).hiddenOutputs = true ∧ (Flock.Zk.tags t m k).isTmpl = false := by
  have htg : Flock.Tags.find? _root_.Spot.OneStageZk.STATEMENT = some Flock.Tags.circuit := rfl
  obtain rfl := Option.some.inj (htg.symm.trans h)
  exact ⟨rfl, rfl⟩

/-! ## What acceptance gives -/

/-- **What V3's and V5's events read of one session**, at its inputs `I` and transcript `t`, the verifier `v` and the
population `n`: `verify --zk` accepts the transcript (`AcceptsZK`) at a statement with hidden outputs that is not
templated; the files' ports and roots (`PortRoots.checkOf`) and their form at the record's sample (`KsScope.checkOf`)
pass on the circuit `verify` reads (`ProvedScope.circuitOf`); the program's proof blocks under the verifier's query are
`n` units in order (`Slots.popOk`); and every public file the session's file loads to has `n` proof blocks. -/
structure Accepted (I : Exec.Inputs) (t : Exec.Transcript) (v : _root_.Spot.Registration.Verifier) (n : ℕ) :
    Prop where
  accepts : ZkExec.AcceptsZK I t
  flat : (ZkExec.zkTags I).hiddenOutputs = true ∧ (ZkExec.zkTags I).isTmpl = false
  ports : Flock.PortRoots.checkOf (Flock.ProvedScope.circuitOf (ZkExec.zkTags I) I.circuitFile I.tables)
    (ZkExec.zkTags I) I.publicFile = .ok ()
  form : Flock.KsScope.checkOf (Flock.ProvedScope.circuitOf (ZkExec.zkTags I) I.circuitFile I.tables)
    (ZkExec.zkTags I) I.publicFile I.count (Exec.recordDraw t.record) = .ok ()
  pop : _root_.Spot.Slots.popOk v.program v.query n = true
  loads : ∀ {tags : Flock.Tags} {c : Flock.Circuit} {pub : Flock.Public},
    Flock.HmOut.loadPublic tags c I.publicFile = .ok pub → pub.n = n

/-- **Where v2's check accepts at the coins `ξ`, its one session is `Accepted`**: at the statement's tags, the circuit
and public files the parameters name on the blobs, the session's record and proof files (`sessionRecord`,
`sessionProofs`), the verifier of the parameters and the population the record names. -/
theorem checkBlobsAt_event {p : _root_.Spot.OneStageZk.Params} {c : _root_.Spot.OneStage.Commitment}
    {ξ : ByteArray} {e : _root_.Spot.OneStageZk.Evidence} {bs : _root_.Core.Blobs}
    {pub : _root_.Spot.OneStage.Public} {prof : _root_.Core.Integrity}
    (h : _root_.Spot.OneStageZk.checkBlobsAt p c ξ e bs = .accept pub prof) :
    ∃ tags0 circuitB pubB record proofs n,
      Flock.Tags.find? _root_.Spot.OneStageZk.STATEMENT = some tags0 ∧
      bs.get p.own.circuit "circuit" = .ok circuitB ∧ bs.get p.own.pub "public file" = .ok pubB ∧
      sessionRecord e bs = some record ∧ sessionProofs p e bs = some proofs ∧
      _root_.Spot.nat? (_root_.Spot.getD c.record "population") = some n ∧
      Accepted (toExec (_root_.Spot.OneStageZk.inputsOf tags0 p.own circuitB pubB bs))
        { record := record, proofs := proofs } p.base.verifier n := by
  obtain ⟨r, pubB', hsz, hpub', hph, hrun, hall⟩ := checkBlobsAt_run h
  obtain ⟨n, hn, hpop⟩ := popHeld_ok (checkBlobsAt_popHeld h)
  have hs := eq_single hsz
  rw [hs] at hrun
  obtain ⟨tags0, circuitB, pubB, rd, ht, hc, hp, hrd, scope, programSha, want, hstep⟩ := run_event hrun
  obtain rfl : pubB = pubB' := Except.ok.inj (hp.symm.trans hpub')
  obtain ⟨recB, record, spec, tabs, rest, proofs, hb, hrec, hst, hf, hv, hk, hks, -⟩ := step_event hstep hall
  have hrec' := sessionRecord_eq hs hb hrec
  have hv2 : V2Verified p e bs record spec tabs rest proofs :=
    ⟨e.sessions[0], tags0, circuitB, pubB, rd, hs, ht, hc, hp, hrd, hrec', hst, hf, hv⟩
  have hpf := sessionProofs_of hv2
  obtain ⟨sts, hset, hshape, rfl⟩ := setupAt_setupOfP hst
  let W := _root_.Spot.OneStageZk.inputsOf tags0 p.own circuitB pubB bs
  have hz : (Flock.Witness.zkTags W).hiddenOutputs = true ∧ (Flock.Witness.zkTags W).isTmpl = false :=
    statement_zkTags ht (Flock.Zk.maskLogOf circuitB) 1
  exact ⟨tags0, circuitB, pubB, record, proofs, n, ht, hc, hp, hrec', hpf, hn,
    ⟨(ZkHidden.acceptsZKP_none _ _).1 ⟨spec, sts, hset, hshape, hv⟩, hz, hk, hks, hpop,
      fun hl => population_held hph (Hidden.loadPublic_nil hl) hn⟩⟩

/-- **Where v2's check accepts, its one session is `Accepted`** (`checkBlobsAt_event` at the coins it read). -/
theorem checkBlobs_event {p : _root_.Spot.OneStageZk.Params} {c : _root_.Spot.OneStage.Commitment}
    {e : _root_.Spot.OneStageZk.Evidence} {bs : _root_.Core.Blobs}
    {pub : _root_.Spot.OneStage.Public} {prof : _root_.Core.Integrity}
    (h : _root_.Spot.OneStageZk.checkBlobs p c e bs = .accept pub prof) :
    ∃ tags0 circuitB pubB record proofs n,
      Flock.Tags.find? _root_.Spot.OneStageZk.STATEMENT = some tags0 ∧
      bs.get p.own.circuit "circuit" = .ok circuitB ∧ bs.get p.own.pub "public file" = .ok pubB ∧
      sessionRecord e bs = some record ∧ sessionProofs p e bs = some proofs ∧
      _root_.Spot.nat? (_root_.Spot.getD c.record "population") = some n ∧
      Accepted (toExec (_root_.Spot.OneStageZk.inputsOf tags0 p.own circuitB pubB bs))
        { record := record, proofs := proofs } p.base.verifier n := by
  unfold _root_.Spot.OneStageZk.checkBlobs _root_.Core.atCoins at h
  split at h
  · exact checkBlobsAt_event h
  · cases h

/-! ## The events -/

variable {I : Exec.Inputs} {t : Exec.Transcript} {v : _root_.Spot.Registration.Verifier} {n : ℕ}

/-- **`Accepted`'s ports and roots pass on `HmOut.parse`'s circuit** (`PortRoots.check`), the one `verify` reads at
its flat statement. -/
theorem Accepted.portsCheck (h : Accepted I t v n) :
    Flock.PortRoots.check (ZkExec.zkTags I) I.circuitFile I.publicFile I.tables = .ok () := by
  show Flock.PortRoots.checkOf (Flock.HmOut.parse (ZkExec.zkTags I) I.circuitFile I.tables) (ZkExec.zkTags I)
    I.publicFile = .ok ()
  rw [← circuitOf_parse h.flat.1 h.flat.2]
  exact h.ports

/-- **`Accepted`'s form passes on `HmOut.parse`'s circuit** (`KsScope.check`), the one `verify` reads at its flat
statement. -/
theorem Accepted.formCheck (h : Accepted I t v n) :
    Flock.KsScope.check (ZkExec.zkTags I) I.circuitFile I.publicFile I.tables I.count
      (Exec.recordDraw t.record) = .ok () := by
  show Flock.KsScope.checkOf (Flock.HmOut.parse (ZkExec.zkTags I) I.circuitFile I.tables) (ZkExec.zkTags I)
    I.publicFile I.count (Exec.recordDraw t.record) = .ok ()
  rw [← circuitOf_parse h.flat.1 h.flat.2]
  exact h.form

/-- **`Accepted` is in V3's acceptance event** at every outcome whose transcript is the session's. -/
theorem Accepted.liveAcceptsScoped (h : Accepted I t v n) {m : ℕ} {Reg : Type} (R : Reg) (S : Finset (Fin m)) :
    Flock.Bridge.LiveAcceptsScoped I (R, S, t) :=
  ⟨⟨h.accepts, h.portsCheck⟩, h.formCheck⟩

/-- **`Accepted` is in V5's acceptance event without public inputs** at every outcome whose transcript is the
session's. -/
theorem Accepted.liveAcceptsKSP (h : Accepted I t v n) {m : ℕ} {Reg : Type} (R : Reg) (S : Finset (Fin m)) :
    Flock.Bridge.LiveAcceptsKSP I none (R, S, t) :=
  ⟨h.accepts, h.ports, _, _, h.form⟩

/-- **`Accepted` passes the run's population check** at a descriptor that parses to the verifier's program. -/
theorem Accepted.runPopOk (h : Accepted I t v n) {desc : Claims.Msg}
    (hd : _root_.Spot.Program.parse ⟨desc.toArray⟩ = .ok v.program) : Flock.Bridge.RunPopOk desc v.query n :=
  ⟨v.program, hd, h.pop⟩

/-- **`Accepted` is in the event `SampledKSLiveV3` bounds**, at its population, the public file's (`pub.n = m`, at the
public file `pub` the session's file loads to under V3's circuit), and at a descriptor that parses to the verifier's
program. -/
theorem Accepted.v3 (h : Accepted I t v n) {D : Flock.Desc} {pub : Flock.Public}
    (hpub : Flock.HmOut.loadPublic (ZkExec.zkTags I) D.unroll I.publicFile = .ok pub) {m : ℕ} (hm : pub.n = m)
    {desc : Claims.Msg} (hd : _root_.Spot.Program.parse ⟨desc.toArray⟩ = .ok v.program) {Reg : Type}
    (R : Reg) (S : Finset (Fin m)) :
    Flock.Bridge.LiveAcceptsScoped I (R, S, t) ∧ Flock.Bridge.RunPopOk desc v.query m := by
  obtain rfl : n = m := (h.loads hpub).symm.trans hm
  exact ⟨h.liveAcceptsScoped R S, h.runPopOk hd⟩

end Verity.Sampled
