import Properties.RemoteAudit.Component

/-!
# The bandwidth component accepts only what `Session.run` accepts, proved (`Properties.RemoteAudit.Component`)

`check` matches on `replay`, which binds `sessionOf` and then `Session.run` (`replay_ok`), and accepts only at a state
that ended accepted with a record, publishing `certify` of it (`acceptIsVerified`). `plays_verified` reads `Plays` as
`Verified`: the session `sessionOf` decodes is the one it decodes, so it is the record's at the node.
-/

namespace Properties.RemoteAudit.Component

open Properties.RemoteAudit.Line

theorem replay_ok {p : RemoteAudit.Protocol.Params} {reg : RemoteAudit.Registration}
    {e : RemoteAudit.Protocol.Evidence} {s : RemoteAudit.Session.State}
    (h : RemoteAudit.Protocol.replay p reg e = .ok s) :
    ∃ (params : RemoteAudit.Session.Params) (coins appendCoins : Array ByteArray) (notBeforeNs : ℕ)
      (events : Array RemoteAudit.Session.Event) (rs : Array RemoteAudit.Session.Result),
      RemoteAudit.Protocol.sessionOf p e = .ok (params, coins, appendCoins, notBeforeNs, events) ∧
      RemoteAudit.Session.run (RemoteAudit.Session.State.init params reg coins appendCoins notBeforeNs) events =
        .ok (rs, s) := by
  unfold RemoteAudit.Protocol.replay at h
  cases hs : RemoteAudit.Protocol.sessionOf p e with
  | error why => simp [hs, bind, Except.bind] at h
  | ok x =>
    obtain ⟨params, coins, appendCoins, notBeforeNs, events⟩ := x
    simp only [hs, bind, Except.bind] at h
    cases hr : RemoteAudit.Session.run
        (RemoteAudit.Session.State.init params reg coins appendCoins notBeforeNs) events with
    | error why => simp [hr] at h
    | ok y =>
      obtain ⟨rs, s'⟩ := y
      simp only [hr, pure, Except.pure, Except.ok.injEq] at h
      subst h
      exact ⟨params, coins, appendCoins, notBeforeNs, events, rs, rfl, hr⟩

theorem acceptIsVerified : AcceptIsVerified := by
  intro p reg e pub prof h
  unfold RemoteAudit.Protocol.check at h
  split at h
  · cases h
  · rename_i s hr
    split at h
    · rename_i sm hout hsum
      obtain ⟨params, coins, appendCoins, notBeforeNs, events, rs, hs, hrun⟩ := replay_ok hr
      rcases hc : RemoteAudit.Protocol.certify sm 1 with ⟨pub', prof'⟩
      rw [hc] at h
      cases h
      exact ⟨params, coins, appendCoins, notBeforeNs, events, rs, s, sm, hs, hrun, hout, hsum, by rw [hc]⟩
    · cases h
    · cases h

/-- Bytes that play the record on the node, accepted by the component, are a session the Lean verifier accepts. -/
theorem plays_verified {node : Node} {rec : Record} {a b e : ByteArray} (h : Plays node rec a b e) :
    Verified node rec := by
  obtain ⟨p, ev, -, -, -, hs, pub, prof, hc⟩ := h
  obtain ⟨params, coins, appendCoins, notBeforeNs, events, rs, s, _, hs', hrun, hout, -, -⟩ :=
    acceptIsVerified p node.reg ev pub prof hc
  rw [hs] at hs'
  simp only [Except.ok.injEq, Prod.mk.injEq] at hs'
  obtain ⟨rfl, rfl, rfl, rfl, rfl⟩ := hs'
  exact ⟨rs, s, hrun, hout⟩

end Properties.RemoteAudit.Component
