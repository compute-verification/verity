import Proofs.Properties.Verity.SampledDeploy
import Proofs.Properties.SampledProofs.OneStageZkRun

/-!
# `SampledDeploy.accept_le` from a coupling of the play to the live audit (ONE-8's field, PRF-205)

`SampledDeploy.accept_le` bounds the probability that sampled proofs' verdict accepts outside the break by the
probability that flock-verify `--zk` accepts the live audit's transcript at the session's inputs. It has two halves.

* **Pointwise, proved here.** Where sampled proofs' verdict accepts, onsite's sampled case decodes to one-stage v2's
  `(p, c, e)` and v2's `checkBlobs` accepts it on the blobs sent (`componentAccepts_sampled`). Then every session of
  the evidence is flock-verify `--zk`'s acceptance of its transcript (the record and proof files the blobs hold under
  the session's digests, `SessionTranscript`) at the session's inputs (`SessionInputs`): `acceptsZK_of_transcript`.
  `AcceptsZK` reads no coin, so this needs no run of the deployed Flock verifier, no `Shows` and no `LiveCoins`; it is
  ONE-3t's `session_verdict` read up to the decision.
* **The coupling, `PlayCoupling`, open.** A transcript of the play (`proj`) that, off the break and wherever v2
  accepts, is a session's of the evidence, at the coupling's inputs `I` (`covers`); and the transport of
  `AcceptsZK I ∘ proj` off the break from the play at `σ` to the live audit at `τ` (`prob_le`). `prob_le` is where the
  device's releases become the session's coins: the play's raw release bits against `auditLive`'s draw (A3's `src`)
  and the live session's points, with ONE-4's `DrawnByDevice` putting a record whose coins no device session released
  in the break.

`accept_le_of_coupling` is `accept_le` in the field's shape from a `PlayCoupling` at the session's setting `A`
(`I = A.I`, `G = A.G`); `accept_le_of_couplings` takes the coupling's break to be the collision or the forger's
success. In a play the blobs are the committed file alone (`Outcome.sent`), so the circuit and public files a session
reads are the file (`get_single`), and `SessionInputs` at `A.I` is a fact about `θ` and `F`. v2 accepts only evidence
with a session (`session_of_accept`), so `covers` has a session to name.

`prob_le` holds only where the session's coins are the device's: flock-verify `--zk` takes every coin of a session from
its record (`Flock.Transcript`: the record's rounds and points), and v2 reads the record from a blob the prover sends
and checks no device signature over it (`Flock.CoinTree.checkRecord` checks its form, `checkFresh` that no key
repeats). A strategy that writes a record whose coins it chose accepts a false statement with probability one, while
the live audit's coins are the verifier's.
-/

open Lean (Json)

namespace Properties.SampledProofs.OneStageZk

open _root_.SampledProofs _root_.SampledProofs.OneStageZk
open FlockSoundness.Discharge

/-- The inputs flock-verify `--zk` reads at a session of one-stage v2: the statement's tags `tags0`, the circuit and
public files, the lookup tables and the verifier's own partition and program (`Run.insE`'s, which the draw and the
registered reads don't change). -/
def sessionI (tags0 : Flock.Tags) (circuit publicFile : ByteArray) (tables : List (String × ByteArray)) (o : Own) :
    Exec.Inputs :=
  (Run.insE tags0 circuit publicFile tables o.partition o.program none none).read.I

/-- **A session's inputs**: the statement's tags, and the circuit and public files the blobs hold under the digests
the parameters and the evidence name. -/
def SessionInputs (p : Params) (e : Evidence) (blobs : Array ByteArray) (I : Exec.Inputs) : Prop :=
  ∃ tags0 circuitB pubB, Flock.Tags.find? STATEMENT = some tags0 ∧
    (Core.Blobs.ofArray blobs).get p.own.circuit "circuit" = .ok circuitB ∧
    (Core.Blobs.ofArray blobs).get e.pub "public file" = .ok pubB ∧
    I = sessionI tags0 circuitB pubB (tablesOf (Core.Blobs.ofArray blobs)) p.own

/-- **A session's transcript**: the record the blobs hold under the session's `record` digest, and the proof files
`verify --zk` reads for the session at its setup at that record's draw (`filesOf`). -/
def SessionTranscript (p : Params) (e : Evidence) (blobs : Array ByteArray) (s : Session) (t : Exec.Transcript) :
    Prop :=
  ∃ tags0 circuitB pubB rd b, Flock.Tags.find? STATEMENT = some tags0 ∧
    (Core.Blobs.ofArray blobs).get p.own.circuit "circuit" = .ok circuitB ∧
    (Core.Blobs.ofArray blobs).get e.pub "public file" = .ok pubB ∧ readsOf p.own.registered = .ok rd ∧
    (Core.Blobs.ofArray blobs).get s.record s!"session {s.name}'s record" = .ok b ∧
    String.fromUTF8? b = some t.record ∧
    filesOf (Core.Blobs.ofArray blobs) s
      (setupAt (inputsOf tags0 p.own circuitB pubB (Core.Blobs.ofArray blobs)) rd (Flock.Witness.recordDraw t.record)) =
      .ok t.proofs

/-- **v2's acceptance is flock-verify `--zk`'s at each session**: where `checkBlobs` accepts, each session of the
evidence set up at its record's draw from the session's inputs, passed `Zk.shapeOk` and `Zk.verify` on its record and
proof files. -/
theorem acceptsZK_of_session {p : Params} {c : OneStage.Commitment} {e : Evidence} {blobs : Array ByteArray}
    {pub : OneStage.Public} {prof : Core.Integrity}
    (hacc : checkBlobs p c e (Core.Blobs.ofArray blobs) = .accept pub prof) {s : Session} (hs : s ∈ e.sessions) :
    ∃ tags0 circuitB pubB rd b record proofs, Flock.Tags.find? STATEMENT = some tags0 ∧
      (Core.Blobs.ofArray blobs).get p.own.circuit "circuit" = .ok circuitB ∧
      (Core.Blobs.ofArray blobs).get e.pub "public file" = .ok pubB ∧ readsOf p.own.registered = .ok rd ∧
      (Core.Blobs.ofArray blobs).get s.record s!"session {s.name}'s record" = .ok b ∧
      String.fromUTF8? b = some record ∧
      filesOf (Core.Blobs.ofArray blobs) s
        (setupAt (inputsOf tags0 p.own circuitB pubB (Core.Blobs.ofArray blobs)) rd (Flock.Witness.recordDraw record)) =
        .ok proofs ∧
      ZkExec.AcceptsZK (sessionI tags0 circuitB pubB (tablesOf (Core.Blobs.ofArray blobs)) p.own) ⟨record, proofs⟩ := by
  obtain ⟨tags0, circuitB, pubB, rd, ht, -, hcB, hpB, hrd, ck, b, record, hb, hu, hver⟩ := Run.session_verdict hacc hs
  obtain ⟨proofs, spec, tabs, x0, hf, hst, hz, -, -, -⟩ := Run.verdictOf_parts hver
  obtain ⟨sts, hsa, htabs, -⟩ := Run.setupAt_parts hst
  subst htabs
  obtain ⟨-, hbso, -, hshape, -⟩ := Run.sessionAt_parts hsa
  have hsetT := Run.bso_ok hbso
  have h3 : (Run.insE tags0 circuitB pubB (tablesOf (Core.Blobs.ofArray blobs)) p.own.partition p.own.program
      (Flock.Witness.recordDraw record) rd.registered).read.setup = .ok (spec, sts) := by
    unfold Flock.Readable.Read.setup
    rw [Run.insE_tags, Run.insE_coins]
    exact hsetT
  have hset := Claims.Bridge.Custody.setupOf_zkInputs h3 rfl rfl (Run.exec_recordDraw record)
  exact ⟨tags0, circuitB, pubB, rd, b, record, proofs, ht, hcB, hpB, hrd, hb, hu, hf, spec, sts, hset, hshape, hz⟩

/-- **Each session's transcript is accepted at the session's inputs**, where v2 accepts. -/
theorem acceptsZK_of_transcript {p : Params} {c : OneStage.Commitment} {e : Evidence} {blobs : Array ByteArray}
    {pub : OneStage.Public} {prof : Core.Integrity}
    (hacc : checkBlobs p c e (Core.Blobs.ofArray blobs) = .accept pub prof) {s : Session} (hs : s ∈ e.sessions)
    {t : Exec.Transcript} (ht : SessionTranscript p e blobs s t) {I : Exec.Inputs} (hI : SessionInputs p e blobs I) :
    ZkExec.AcceptsZK I t := by
  obtain ⟨tags0, circuitB, pubB, rd, b, record, proofs, ht0, hcB, hpB, hrd, hb, hu, hf, hA⟩ :=
    acceptsZK_of_session hacc hs
  obtain ⟨tr, tp⟩ := t
  obtain ⟨tags1, c1, p1, rd1, b1, ht1, hc1, hp1, hrd1, hb1, hu1, hf1⟩ := ht
  obtain ⟨tags2, c2, p2, ht2, hc2, hp2, rfl⟩ := hI
  obtain rfl := Option.some.inj (ht1.symm.trans ht0)
  obtain rfl := Option.some.inj (ht2.symm.trans ht0)
  obtain rfl := Except.ok.inj (hc1.symm.trans hcB)
  obtain rfl := Except.ok.inj (hc2.symm.trans hcB)
  obtain rfl := Except.ok.inj (hp1.symm.trans hpB)
  obtain rfl := Except.ok.inj (hp2.symm.trans hpB)
  obtain rfl := Except.ok.inj (hrd1.symm.trans hrd)
  obtain rfl := Except.ok.inj (hb1.symm.trans hb)
  obtain rfl := Option.some.inj (hu1.symm.trans hu)
  obtain rfl := Except.ok.inj (hf1.symm.trans hf)
  exact hA

/-- A blob found among the one blob `F` is `F`. -/
theorem get_single {F b : ByteArray} {d w : String} (h : (Core.Blobs.ofArray #[F]).get d w = .ok b) : b = F := by
  unfold Core.Blobs.get at h
  obtain ⟨d', -, h⟩ := Run.bind_ok h
  by_cases hc : (d'.size != 64 || Flock.hexOf d' != d) = true
  · rw [if_pos hc] at h
    cases h
  · rw [if_neg hc] at h
    split at h
    · rename_i fst snd hf
      have hm : snd = F := by
        have hm := Array.mem_of_find?_eq_some hf
        simp [Core.Blobs.ofArray] at hm
        first | exact hm.2 | exact hm
      exact (Except.ok.inj h).symm.trans hm
    · cases h

/-- **v2 accepts only evidence with a session**: with none, `run` pushes no verdict, and v1's check on no verdicts is
undecided (`OneStage.verdictOf`: no verifier of record), never an acceptance. -/
theorem session_of_accept {p : Params} {c : OneStage.Commitment} {e : Evidence} {bs : Core.Blobs}
    {pub : OneStage.Public} {prof : Core.Integrity} (hacc : checkBlobs p c e bs = .accept pub prof) :
    ∃ s, s ∈ e.sessions := by
  unfold checkBlobs at hacc
  split at hacc
  · cases hacc
  · next r hrun =>
    split at hacc
    · next pub' prof' hchk =>
      obtain ⟨_, _, _, _, _, _, hva, _⟩ := (Properties.SampledProofs.acceptMeans_iff _ _ _ _ _).1 hchk
      obtain ⟨x, hx, -⟩ := hva.2.1
      rcases hl : e.sessions.toList with _ | ⟨s, l⟩
      · exfalso
        unfold run at hrun
        cases ht : Flock.Tags.find? STATEMENT with
        | none => simp only [ht] at hrun <;> cases hrun
        | some tags0 =>
          simp only [ht] at hrun
          obtain ⟨_, _, hrun⟩ := Run.bind_ok hrun
          obtain ⟨_, _, hrun⟩ := Run.bind_ok hrun
          obtain ⟨_, _, hrun⟩ := Run.bind_ok hrun
          obtain ⟨_, _, hrun⟩ := Run.bind_ok hrun
          obtain ⟨_, _, hrun⟩ := Run.bind_ok hrun
          obtain ⟨r1, hf, hrun⟩ := Run.bind_ok hrun
          rw [← Array.foldlM_toList, hl, List.foldlM_nil] at hf
          first | cases hf | obtain rfl := Except.ok.inj hf
          try dsimp only at hrun
          split at hrun
          · cases hrun
          · cases hrun
            simp at hx
      · exact ⟨s, Array.mem_toList_iff.1 (by rw [hl]; simp)⟩
    · next o hne hchk =>
      first
      | exact (hchk _ _ hacc).elim
      | exact (hchk _ _ (hne ▸ hacc)).elim
      | exact (hchk _ _ (hne.symm ▸ hacc)).elim
      | (subst hne; exact (hchk _ _ hacc).elim)

end Properties.SampledProofs.OneStageZk

namespace Verity

open Accounting Core
open scoped ENNReal
open Properties.SampledProofs.OneStageZk (SessionInputs SessionTranscript acceptsZK_of_transcript)
open FlockSoundness.Discharge

variable {ι α : Type} [Fintype α] {d : Device ι}

/-- **Onsite's sampled case** in what the prover sent: onsite's byte strings decode, and sampled proofs' one-stage v2
on the blobs sent has its triple among the components', whose parameters, commitment and evidence decode to
`(p, c, e)` under v2's decoders. -/
def SampledCase (s : Sent) (p : _root_.SampledProofs.OneStageZk.Params) (c : _root_.SampledProofs.OneStage.Commitment)
    (e : _root_.SampledProofs.OneStageZk.Evidence) : Prop :=
  ∃ n ps b dg k ev, Decoded s n ps b dg k ev ∧ ∃ nm cs,
    ((component .sampled).onBlobs (Core.Blobs.ofArray s.blobs), nm, cs) ∈
        Properties.Composition.triples s.components ps ev ∧
      (canonicalJson nm.params.toUTF8 >>= _root_.SampledProofs.OneStageZk.paramsOfJson) = .ok p ∧
      _root_.SampledProofs.OneStage.commitmentOf cs.commitment = .ok c ∧
      _root_.SampledProofs.OneStageZk.evidenceOf cs.evidence = .ok e

/-- **Sampled proofs' component accepts its case** only where its case decodes to v2's `(p, c, e)` and v2's
`checkBlobs` accepts them on the blobs sent. -/
theorem componentAccepts_sampled {s : Sent} (h : ComponentAccepts .sampled s) :
    ∃ p c e pub prof, SampledCase s p c e ∧
      _root_.SampledProofs.OneStageZk.checkBlobs p c e (Core.Blobs.ofArray s.blobs) = .accept pub prof := by
  obtain ⟨n, ps, b, dg, k, ev, hD, ⟨P, nm, cs⟩, hx, hx1, p, c, e, hp, hc, he, pub, prof, hacc⟩ := h
  try dsimp only at hx1
  subst hx1
  exact ⟨p, c, e, pub, prof, ⟨n, ps, b, dg, k, ev, hD, nm, cs, hx, hp, hc, he⟩, hacc⟩

/-- **The play coupled to a live audit**, at the session's inputs `I`, the live audit's game `G` and the strategy `τ`
the play at `σ` induces: a transcript of each outcome (`proj`) that, off the break `Brk` and wherever sampled proofs'
verdict accepts, is a session's transcript of v2's evidence, at inputs `I` (`covers`); and off the break, the play
gives `proj` a transcript flock-verify `--zk` accepts at most as often as the live audit gives its own (`prob_le`). -/
structure PlayCoupling (V : Vocab d α) (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)) (Brk : Outcome → Prop)
    {n : ℕ} {Reg : Type} (I : Exec.Inputs) (G : FlockSoundness.Game (Reg × Finset (Fin n) × Exec.Transcript))
    (τ : FlockSoundness.Game.Strategy G) where
  proj : Outcome → Exec.Transcript
  covers : ∀ o, V.Verdict .sampled o → ¬ Brk o → ∀ p c' e pub prof, SampledCase o.sent p c' e →
    _root_.SampledProofs.OneStageZk.checkBlobs p c' e (Core.Blobs.ofArray o.sent.blobs) = .accept pub prof →
    SessionInputs p e o.sent.blobs I ∧ ∃ s ∈ e.sessions, SessionTranscript p e o.sent.blobs s (proj o)
  prob_le : Game.prob (fun o => ZkExec.AcceptsZK I (proj o) ∧ ¬ Brk o) (V.game θ F c) σ ≤
    FlockSoundness.Game.prob (fun o => ZkExec.LiveAcceptsZK I o) G τ

/-- **The verdict off the break is a coupled acceptance**: off the break, sampled proofs' verdict accepts at most as
often as the live audit accepts at the coupling's inputs. -/
theorem accept_le_of_coupling {V : Vocab d α} {θ F c : ByteArray} {σ : Game.Strategy (V.game θ F c)}
    {Brk : Outcome → Prop} {n : ℕ} {Reg : Type} {I : Exec.Inputs}
    {G : FlockSoundness.Game (Reg × Finset (Fin n) × Exec.Transcript)} {τ : FlockSoundness.Game.Strategy G}
    (Cp : PlayCoupling V θ F c σ Brk I G τ) :
    Game.prob (fun o => V.Verdict .sampled o ∧ ¬ Brk o) (V.game θ F c) σ ≤
      FlockSoundness.Game.prob (fun o => ZkExec.LiveAcceptsZK I o) G τ := by
  refine (Game.prob_mono (fun o ho => ?_) _ σ).trans Cp.prob_le
  obtain ⟨hv, hb⟩ := ho
  obtain ⟨p, c', e, pub, prof, hcase, hacc⟩ := componentAccepts_sampled hv.2.2
  obtain ⟨hI, s, hs, ht⟩ := Cp.covers o hv hb p c' e pub prof hcase hacc
  exact ⟨acceptsZK_of_transcript hacc hs ht hI, hb⟩

/-- **`SampledDeploy.accept_le` from a coupling** at the session `A θ F c σ r hr` and its strategy `τOf`, for each
registered file that parses, off the collision `Brk` and the forger `Fg`'s success (the coupling's break is either). -/
theorem accept_le_of_couplings {V : Vocab d α} (Fg : LmsForger) (kd : ByteArray → ℕ) (desc : ByteArray → Claims.Msg)
    (A : ∀ (θ F c : ByteArray) (_σ : Game.Strategy (V.game θ F c)) (r : Flock.Bridge.Parsed), V.parsed F = some r →
      Flock.Guarantees.LiveAt r.n r.D r.hU (kd θ) (⌊(V.certOf θ).ε * r.n⌋₊ + 1) (desc θ))
    (τOf : ∀ (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)) (r : Flock.Bridge.Parsed)
      (hr : V.parsed F = some r), FlockSoundness.Game.Strategy (A θ F c σ r hr).G)
    (Brk : ByteArray → ByteArray → ByteArray → Outcome → Prop)
    (Cp : ∀ θ F c σ r (hr : V.parsed F = some r), V.Registered θ F →
      PlayCoupling V θ F c σ (fun o => Brk θ F c o ∨ Fg.Forges θ o) (A θ F c σ r hr).I (A θ F c σ r hr).G
        (τOf θ F c σ r hr)) :
    ∀ (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)) (r : Flock.Bridge.Parsed)
      (hr : V.parsed F = some r), V.Registered θ F →
      Game.prob (fun o => V.Verdict .sampled o ∧ ¬ Brk θ F c o ∧ ¬ Fg.Forges θ o) (V.game θ F c) σ ≤
        FlockSoundness.Game.prob (fun o => ZkExec.LiveAcceptsZK (A θ F c σ r hr).I o)
          (A θ F c σ r hr).G (τOf θ F c σ r hr) :=
  fun θ F c σ r hr hreg =>
    (Game.prob_mono (fun o ho => ⟨ho.1, fun h => h.elim ho.2.1 ho.2.2⟩) _ σ).trans
      (accept_le_of_coupling (Cp θ F c σ r hr hreg))

end Verity
