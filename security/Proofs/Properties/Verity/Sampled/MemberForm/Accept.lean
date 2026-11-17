module

public import Proofs.Properties.Verity.Sampled.Window.Accepts
public import Proofs.Properties.Verity.SampledDraw

/-! What an accept of the per-member form holds. -/

@[expose] public section

namespace Verity.Sampled.MemberForm

open Lean (Json)
open Properties.Challenger (Draws)

structure MembersAccept (p : _root_.Spot.OneStageZk.Params) (c : _root_.Spot.OneStage.Commitment) (ξ : ByteArray)
    (e : _root_.Spot.OneStageZk.Evidence) (bs : _root_.Core.Blobs) (ms : Array _root_.Spot.OneStageZk.Member)
    (sts : Array _root_.Spot.OneStageZk.StatementFiles) (r : _root_.Spot.OneStageZk.Run) : Prop where
  members : p.members = some ms
  statements : e.statements = some sts
  sessions : e.sessions = sts.flatMap (·.sessions)
  run : _root_.Spot.OneStageZk.membersRun p ms c e sts bs = .ok r
  all : r.all = true
  v1 : ∃ pub prof,
    _root_.Spot.OneStage.checkAt p.base c (some ξ) { e.base with verdicts := r.out } = .accept pub prof

def FormAccepts : Prop :=
  ∀ (bs : _root_.Core.Blobs) (a b e : ByteArray) (p : _root_.Spot.OneStageZk.Params)
    (c : _root_.Spot.OneStage.Commitment) (ev : _root_.Spot.OneStageZk.Evidence) (ξ : ByteArray) (dr : Draws)
    (ms : Array _root_.Spot.OneStageZk.Member),
    V2Accepts bs a b e p c ev ξ dr → p.members = some ms → ∃ sts r, MembersAccept p c ξ ev bs ms sts r

theorem formAccepts : FormAccepts := by
  intro bs a b e p c ev ξ dr ms h hm
  obtain ⟨-, -, hev, -, -, pub, prof, hacc⟩ := h
  obtain ⟨sts, hsts, hM⟩ := _root_.Spot.OneStageZk.checkBlobsAt_accept_members hm hacc
  obtain ⟨r, hrun, hall, hv1⟩ := _root_.Spot.OneStageZk.checkMembersAt_accept hM
  obtain ⟨b', hb'⟩ := dispatch_evidence hev
  exact ⟨sts, r, hm, hsts, _root_.Spot.OneStageZk.evidenceEitherOf_sessions hb' hsts, hrun, hall, pub, prof, hv1⟩

theorem accept_members {ι : Type} {d : ResourceAccounting.Device ι} {α : Type} [Fintype α] {V : Vocab d α}
    {k : Verity.Registration.Composite} {o : Verity.Exchange} (hF : FormAccepts)
    {p : _root_.Spot.OneStageZk.Params} {ms : Array _root_.Spot.OneStageZk.Member}
    (hp : ParamsAt o.params p) (hms : p.members = some ms) (hv : V.Verdict k .sampled o) :
    ∃ (a b e : ByteArray) (c : _root_.Spot.OneStage.Commitment) (ev : _root_.Spot.OneStageZk.Evidence)
      (ξ : ByteArray) (dr : Draws) (sts : Array _root_.Spot.OneStageZk.StatementFiles)
      (r : _root_.Spot.OneStageZk.Run),
      V2Case k o.sent a b e ∧ V2Accepts (_root_.Core.Blobs.ofArray o.sent.blobs) a b e p c ev ξ dr ∧
        MembersAccept p c ξ ev (_root_.Core.Blobs.ofArray o.sent.blobs) ms sts r := by
  obtain ⟨a, b, e, p', c, ev, ξ, dr, hcase, hacc⟩ := case_drew hv.2.1 hp (verdict_case hv)
  have hpp : p = p' := paramsAt_case hv.2.1 hcase hacc hp
  subst hpp
  obtain ⟨sts, r, hM⟩ := hF _ _ _ _ _ _ _ _ _ _ hacc hms
  exact ⟨a, b, e, c, ev, ξ, dr, sts, r, hcase, hacc, hM⟩

def stepAt (a : _root_.Spot.OneStageZk.StepAt) : StepAt :=
  ⟨a.I, a.rd, a.scope, a.ksFiles, a.population, a.programSha, a.want⟩

theorem stepAt_step (a : _root_.Spot.OneStageZk.StepAt) (bs : _root_.Core.Blobs) (r : _root_.Spot.OneStageZk.Run)
    (s : _root_.Spot.OneStageZk.RunFiles) : (stepAt a).step bs r s = a.step bs r s :=
  rfl

end Verity.Sampled.MemberForm
