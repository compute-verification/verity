module

public import Proofs.Properties.Verity.Sampled.Accepts
public import Proofs.Properties.Verity.Sampled.Held
public import Proofs.Flock.Bridge.ScopeGate
public import Proofs.Flock.Bridge.V5.Accepted

/-! SPOT's acceptance is in the event V5 bounds. -/

@[expose] public section

namespace Verity.Sampled

open Lean (Json)
open FlockSoundness.Discharge
open FlockSoundness.Discharge.Recursion.Witness (toExec)
open FlockSoundness.Refine (ebind_ok)

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

theorem checkBlobsAt_popHeld {p : _root_.Spot.OneStageZk.Params}
    {c : _root_.Spot.OneStage.Commitment} {ξ : ByteArray} {e : _root_.Spot.OneStageZk.Evidence}
    {bs : _root_.Core.Blobs} {pub : _root_.Spot.OneStage.Public} {prof : _root_.Core.Integrity}
    (h : _root_.Spot.OneStageZk.checkBlobsAt p c ξ e bs = .accept pub prof) :
    _root_.Spot.OneStageZk.popHeld p.base.verifier c.record = .ok () := by
  rcases _root_.Spot.OneStageZk.checkBlobsAt_accept h with ⟨-, -, h⟩ | ⟨ms, sts, -, -, h⟩
  swap
  · obtain ⟨r, hr, -, -⟩ := _root_.Spot.OneStageZk.checkMembersAt_accept h
    exact (_root_.Spot.OneStageZk.membersRun_held hr).2
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

theorem popHeld_ok {v : _root_.Spot.Docket.Verifier} {record : Json}
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

theorem circuitOf_parse {t : Flock.Tags} (hh : t.hiddenOutputs = true) (ht : t.isTmpl = false) (f : ByteArray)
    (tb : List (String × ByteArray)) : Flock.ProvedScope.circuitOf t f tb = Flock.HmOut.parse t f tb := by
  simp only [Flock.ProvedScope.circuitOf, hh, ↓reduceIte, Flock.HmOut.parseCommit, ht, Bool.false_eq_true]

theorem statement_zkTags {t : Flock.Tags} (h : Flock.Tags.find? _root_.Spot.OneStageZk.STATEMENT = some t)
    (m k : ℕ) : (Flock.Zk.tags t m k).hiddenOutputs = true ∧ (Flock.Zk.tags t m k).isTmpl = false := by
  have htg : Flock.Tags.find? _root_.Spot.OneStageZk.STATEMENT = some Flock.Tags.circuit := rfl
  obtain rfl := Option.some.inj (htg.symm.trans h)
  exact ⟨rfl, rfl⟩

structure Accepted (I : Exec.Inputs) (t : Exec.Transcript) (v : _root_.Spot.Docket.Verifier) (n : ℕ) :
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

theorem checkBlobsAt_event {p : _root_.Spot.OneStageZk.Params} {c : _root_.Spot.OneStage.Commitment}
    {ξ : ByteArray} {e : _root_.Spot.OneStageZk.Evidence} {bs : _root_.Core.Blobs}
    {pub : _root_.Spot.OneStage.Public} {prof : _root_.Core.Integrity}
    (hm : p.members = none) (h : _root_.Spot.OneStageZk.checkBlobsAt p c ξ e bs = .accept pub prof) :
    ∃ tags0 circuitB pubB record proofs n,
      Flock.Tags.find? _root_.Spot.OneStageZk.STATEMENT = some tags0 ∧
      bs.get p.own.circuit "circuit" = .ok circuitB ∧ bs.get p.own.pub "public file" = .ok pubB ∧
      sessionRecord e bs = some record ∧ sessionProofs p e bs = some proofs ∧
      _root_.Spot.nat? (_root_.Spot.getD c.record "population") = some n ∧
      Accepted (toExec (_root_.Spot.OneStageZk.inputsOf tags0 p.own circuitB pubB bs))
        { record := record, proofs := proofs } p.base.verifier n := by
  obtain ⟨r, pubB', hsz, hpub', hph, hrun, hall⟩ := checkBlobsAt_run hm h
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

theorem checkBlobs_event {p : _root_.Spot.OneStageZk.Params} {c : _root_.Spot.OneStage.Commitment}
    {e : _root_.Spot.OneStageZk.Evidence} {bs : _root_.Core.Blobs}
    {pub : _root_.Spot.OneStage.Public} {prof : _root_.Core.Integrity}
    (hm : p.members = none) (h : _root_.Spot.OneStageZk.checkBlobs p c e bs = .accept pub prof) :
    ∃ tags0 circuitB pubB record proofs n,
      Flock.Tags.find? _root_.Spot.OneStageZk.STATEMENT = some tags0 ∧
      bs.get p.own.circuit "circuit" = .ok circuitB ∧ bs.get p.own.pub "public file" = .ok pubB ∧
      sessionRecord e bs = some record ∧ sessionProofs p e bs = some proofs ∧
      _root_.Spot.nat? (_root_.Spot.getD c.record "population") = some n ∧
      Accepted (toExec (_root_.Spot.OneStageZk.inputsOf tags0 p.own circuitB pubB bs))
        { record := record, proofs := proofs } p.base.verifier n := by
  unfold _root_.Spot.OneStageZk.checkBlobs _root_.Core.atCoins at h
  split at h
  · exact checkBlobsAt_event hm h
  · cases h

variable {I : Exec.Inputs} {t : Exec.Transcript} {v : _root_.Spot.Docket.Verifier} {n : ℕ}

theorem Accepted.liveAcceptsKSP (h : Accepted I t v n) {m : ℕ} {Reg : Type} (R : Reg) (S : Finset (Fin m)) :
    Flock.Bridge.LiveAcceptsKSP I none (R, S, t) :=
  ⟨h.accepts, h.ports, _, _, h.form⟩

end Verity.Sampled
