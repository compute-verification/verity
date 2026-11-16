module

public import Proofs.Properties.Verity.Sampled.Window.Accepts
public import Proofs.Properties.Verity.SampledDraw

/-!
# What an accept of the per-member form holds

This module fixes what the proofs about an accept of the per-member form of one-stage v2
(`Spot.OneStageZk.checkMembersAt`) read of it, stated once, and proves it of the compiled check:

* **What an accept holds** (`MembersAccept`): the parameters name the members, the evidence its statements, whose
  sessions in order are the evidence's; the form's run (`Spot.OneStageZk.membersRun`) returned with every session
  accepted; and v1's check accepted on the run's verdicts at the coins.
* **The form's accepts hold it** (`FormAccepts`): of everything v2 accepts at the coins it read (`V2Accepts`) on
  parameters that name members. It holds of the check (`formAccepts`): what `checkBlobsAt` accepts on such
  parameters, `checkMembersAt` accepts, at evidence either form's parse returned.
* **An accepted play at parameters that name members is an accept of the form** (`accept_members`): v2's case in
  it accepted at the coins it read, on those parameters, and the form's accepts hold `MembersAccept`.
* **A statement of the form as the sessions' verdicts read it** (`stepAt`): what `Spot.OneStageZk.StepAt` holds
  besides the form of its circuit file is the statement `SessionVerdict` and `Verdicts` are stated at, with the same
  `step` (`stepAt_step`).
-/

@[expose] public section

namespace Verity.Sampled.MemberForm

open Lean (Json)
open Properties.Challenger (Draws)

/-- **What an accept of the per-member form holds**, of the parameters `p`, the registration `c`, the coins `ξ`, the
evidence `e` and the blobs `bs`, at the members `ms`, the statements `sts` and the run `r`: the parameters name the
members and the evidence the statements; the evidence's sessions are the statements', in order; the form's run
returned `r` with every session accepted; and v1's check accepts at the coins on the run's verdicts. -/
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

/-- **The per-member form's accepts hold what the proofs read of them**: wherever v2 accepts at the coins it read
(`V2Accepts`) on parameters that name members, the accept is one of the form (`MembersAccept`), at some statements
and some run. -/
def FormAccepts : Prop :=
  ∀ (bs : _root_.Core.Blobs) (a b e : ByteArray) (p : _root_.Spot.OneStageZk.Params)
    (c : _root_.Spot.OneStage.Commitment) (ev : _root_.Spot.OneStageZk.Evidence) (ξ : ByteArray) (dr : Draws)
    (ms : Array _root_.Spot.OneStageZk.Member),
    V2Accepts bs a b e p c ev ξ dr → p.members = some ms → ∃ sts r, MembersAccept p c ξ ev bs ms sts r

/-- **The check's accepts on parameters that name members are accepts of the form**: what `checkBlobsAt` accepts
there, `checkMembersAt` accepts at the evidence's statements (`Spot.OneStageZk.checkBlobsAt_accept_members`), so
the form's run returned with every session accepted and v1's check accepted on its verdicts
(`Spot.OneStageZk.checkMembersAt_accept`); and evidence either form's parse returned that names statements has
their sessions, in order (`Spot.OneStageZk.evidenceEitherOf_sessions`). -/
theorem formAccepts : FormAccepts := by
  intro bs a b e p c ev ξ dr ms h hm
  obtain ⟨-, -, hev, -, -, pub, prof, hacc⟩ := h
  obtain ⟨sts, hsts, hM⟩ := _root_.Spot.OneStageZk.checkBlobsAt_accept_members hm hacc
  obtain ⟨r, hrun, hall, hv1⟩ := _root_.Spot.OneStageZk.checkMembersAt_accept hM
  obtain ⟨b', hb'⟩ := dispatch_evidence hev
  exact ⟨sts, r, hm, hsts, _root_.Spot.OneStageZk.evidenceEitherOf_sessions hb' hsts, hrun, hall, pub, prof, hv1⟩

/-- **An accepted play at parameters that name members is an accept of the form**: v2's case in the play accepted
at the coins it read, on the play's parameters, and the form's accepts hold `MembersAccept`. -/
theorem accept_members {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α] {V : Vocab d α}
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

/-- **A statement of the form as the sessions' verdicts read it**: its inputs, the verifier's registered reads, the
scope, files and population checks and its program's digest against the partition object's; without the form of its
circuit file and its units a proof block, which no session's `step` reads. -/
def stepAt (a : _root_.Spot.OneStageZk.StepAt) : StepAt :=
  ⟨a.I, a.rd, a.scope, a.ksFiles, a.population, a.programSha, a.want⟩

/-- A session's `step` at a statement of the form is its `step` as the sessions' verdicts read it. -/
theorem stepAt_step (a : _root_.Spot.OneStageZk.StepAt) (bs : _root_.Core.Blobs) (r : _root_.Spot.OneStageZk.Run)
    (s : _root_.Spot.OneStageZk.RunFiles) : (stepAt a).step bs r s = a.step bs r s :=
  rfl

end Verity.Sampled.MemberForm
