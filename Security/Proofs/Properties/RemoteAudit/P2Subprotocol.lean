import Properties.RemoteAudit.P2Subprotocol

/-!
# PoUB-P2's subprotocol accepts only a session whose every window verified, proved (`Properties.RemoteAudit.P2Subprotocol`)

`check` matches on `replay`, which reads the request (`replay_request`) before it replays the log, and accepts only at
a state with no failure and a record, which ended accepted (`outcome_of_summary`), publishing `certify` of it
(`acceptIsVerified`).

Its witnesses:

* honest (`acceptIsVerified_honest`): the auditor's session requested for the honest store's node reads as its session
  parameters (`requestOf_honest`), and the check leaves the session undecided before its first request is answered: no
  window has verified yet;
* nontrivial (`acceptIsVerified_nontrivial`): a request that is no object is no node's, so no session replays from it
  and the conclusion fails at every record; a check that accepts everything accepts it.
-/

namespace Properties.RemoteAudit.P2Subprotocol

/-- A replay that ran read the request as the held registration's node's. -/
theorem replay_request {p : RemoteAudit.P2Protocol.Params} {c : RemoteAudit.P2Protocol.Commitment}
    {e : RemoteAudit.P2Protocol.Evidence} {s : RemoteAudit.Session.State}
    (h : RemoteAudit.P2Protocol.replay p c e = .ok s) :
    ∃ params, RemoteAudit.P2Session.requestOf c.held p.request = .ok params := by
  unfold RemoteAudit.P2Protocol.replay at h
  cases hq : RemoteAudit.P2Session.requestOf c.held p.request with
  | error why => simp [hq, bind, Except.bind, throw, throwThe, MonadExceptOf.throw] at h
  | ok params => exact ⟨params, rfl⟩

/-- A session with a record ended accepted. -/
theorem outcome_of_summary {s : RemoteAudit.Session.State} {sm : RemoteAudit.Session.Summary}
    (h : s.summary = some sm) : s.outcome = some "accepted" := by
  unfold RemoteAudit.Session.State.summary at h
  split at h
  · cases h
  · rename_i hc
    simpa using hc

theorem acceptIsVerified : AcceptIsVerified := by
  intro p c e pub prof h
  unfold RemoteAudit.P2Protocol.check at h
  cases hr : RemoteAudit.P2Protocol.replay p c e with
  | error why => rw [hr] at h; cases h
  | ok s =>
    rw [hr] at h
    dsimp only at h
    cases hf : s.failed with
    | some f => rw [hf] at h; cases h
    | none =>
      cases hsm : s.summary with
      | none => rw [hf, hsm] at h; cases h
      | some sm =>
        rw [hf, hsm] at h
        dsimp only at h
        simp only [Core.Outcome.accept.injEq] at h
        obtain ⟨rfl, -⟩ := h
        obtain ⟨params, hq⟩ := replay_request hr
        exact ⟨params, s, sm, hq, rfl, hf, outcome_of_summary hsm, hsm, rfl⟩

/-! ## Witnesses -/

theorem honest_request : (match RemoteAudit.P2Session.requestOf honestHeld honestRequest with
    | .ok params => params.deadlineNs == 1000000000 && params.windows == 2 && params.samples == 1024 &&
        params.minPasses == 1 && params.windowNs == 1000000000 && params.lifetime == 30 && params.pace.isNone
    | .error _ => false) = true := by decide +kernel

/-- The auditor's session requested for the honest store's node reads as its session parameters. -/
theorem requestOf_honest : RemoteAudit.P2Session.requestOf honestHeld honestRequest = .ok honestSessionParams := by
  have h := honest_request
  revert h
  cases RemoteAudit.P2Session.requestOf honestHeld honestRequest with
  | error _ => intro h; cases h
  | ok params =>
    intro h
    obtain ⟨d, w, sa, m, wn, l, pace⟩ := params
    simp only [Bool.and_eq_true, beq_iff_eq, Option.isNone_iff_eq_none] at h
    obtain ⟨⟨⟨⟨⟨⟨rfl, rfl⟩, rfl⟩, rfl⟩, rfl⟩, rfl⟩, rfl⟩ := h
    rfl

theorem honest_replay : (match RemoteAudit.P2Protocol.replay honestParams honestCommitment honestEvidence with
    | .ok s => s.failed.isNone && s.summary.isNone
    | .error _ => false) = true := by decide +kernel

/-- **The deployed check at the honest session before its first request is answered.** The auditor's session requested
for the honest store's node reads as its session parameters (two windows of 1 s, one pass each, 1,024 openings a
window), and the check leaves the session undecided: no window has verified yet. -/
theorem acceptIsVerified_honest :
    RemoteAudit.P2Session.requestOf honestHeld honestRequest = .ok honestSessionParams ∧
      RemoteAudit.P2Protocol.check honestParams honestCommitment honestEvidence =
        .undecided "the session has a window not yet verified" := by
  refine ⟨requestOf_honest, ?_⟩
  have h := honest_replay
  unfold RemoteAudit.P2Protocol.check
  revert h
  cases RemoteAudit.P2Protocol.replay honestParams honestCommitment honestEvidence with
  | error _ => intro h; cases h
  | ok s =>
    intro h
    simp only [Bool.and_eq_true, Option.isNone_iff_eq_none] at h
    obtain ⟨h1, h2⟩ := h
    simp only [h1, h2]

/-- **`AcceptIsVerified`'s conclusion fails where a check that accepts everything accepts.** A request that is no
object (`null`) is no node's, so at the honest store and log no session parameters, state and record meet the
conclusion, whatever the public outputs: a check that accepts every log accepts this one. -/
theorem acceptIsVerified_nontrivial :
    ∀ (params : RemoteAudit.Session.Params) (s : RemoteAudit.Session.State) (sm : RemoteAudit.Session.Summary),
      ¬ (RemoteAudit.P2Session.requestOf honestCommitment.held (⟨0, .null⟩ : RemoteAudit.P2Protocol.Params).request =
            .ok params ∧
          RemoteAudit.P2Protocol.replay ⟨0, .null⟩ honestCommitment honestEvidence = .ok s ∧ s.failed = none ∧
          s.outcome = some "accepted" ∧ s.summary = some sm ∧
          (⟨⟨0, 1⟩, 0, 0⟩ : RemoteAudit.P2Protocol.Public) = (RemoteAudit.P2Protocol.certify honestCommitment.held sm).1) := by
  intro params s sm h
  have hb : (match RemoteAudit.P2Session.requestOf honestHeld .null with
      | .ok _ => true
      | .error _ => false) = false := by decide +kernel
  have hq : RemoteAudit.P2Session.requestOf honestHeld .null = .ok params := h.1
  rw [hq] at hb
  simp at hb

end Properties.RemoteAudit.P2Subprotocol
