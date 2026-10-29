import Properties.PoubCheck.Component

/-!
# The bandwidth component accepts only what `Shared.run g` accepts, proved (`Properties.PoubCheck.Component`)

`check` matches on `replay`, which binds `sessionOf` and then `Shared.run g` (`replay_ok`), and accepts only at a state
that ended accepted with a record, publishing `certify` of it at g (`acceptIsVerified`). `plays_verified` reads `Plays`
as `Verified`: the session `sessionOf` decodes is the one it decodes, so it is the record's at the node, at the node's
windows a draw.
-/

namespace Properties.PoubCheck.Component

open Properties.PoubCheck.Line

theorem replay_ok {p : PoubCheck.Protocol.Params} {reg : PoubCheck.Registration}
    {e : PoubCheck.Protocol.Evidence} {s : PoubCheck.Session.State} {g : ℕ}
    (h : PoubCheck.Protocol.replay p reg e = .ok (s, g)) :
    ∃ (params : PoubCheck.Session.Params) (coins appendCoins : Array ByteArray) (notBeforeNs : ℕ)
      (events : Array PoubCheck.Session.Event) (rs : Array PoubCheck.Session.Result),
      PoubCheck.Protocol.sessionOf p e = .ok (params, g, coins, appendCoins, notBeforeNs, events) ∧
      PoubCheck.Shared.run g (PoubCheck.Session.State.init params reg coins appendCoins notBeforeNs) events =
        .ok (rs, s) := by
  unfold PoubCheck.Protocol.replay at h
  cases hs : PoubCheck.Protocol.sessionOf p e with
  | error why => simp [hs, bind, Except.bind] at h
  | ok x =>
    obtain ⟨params, g', coins, appendCoins, notBeforeNs, events⟩ := x
    simp only [hs, bind, Except.bind] at h
    cases hr : PoubCheck.Shared.run g'
        (PoubCheck.Session.State.init params reg coins appendCoins notBeforeNs) events with
    | error why => simp [hr] at h
    | ok y =>
      obtain ⟨rs, s'⟩ := y
      simp only [hr, pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      exact ⟨params, coins, appendCoins, notBeforeNs, events, rs, rfl, hr⟩

theorem acceptIsVerified : AcceptIsVerified := by
  intro p reg e pub prof h
  unfold PoubCheck.Protocol.check at h
  split at h
  · cases h
  · rename_i s g hr
    split at h
    · rename_i sm hout hsum
      obtain ⟨params, coins, appendCoins, notBeforeNs, events, rs, hs, hrun⟩ := replay_ok hr
      rcases hc : PoubCheck.Protocol.certify sm g with ⟨pub', prof'⟩
      rw [hc] at h
      simp only [Core.Outcome.accept.injEq] at h
      obtain ⟨rfl, -⟩ := h
      exact ⟨params, g, coins, appendCoins, notBeforeNs, events, rs, s, sm, hs, hrun, hout, hsum, by rw [hc]⟩
    · cases h
    · cases h

/-- Bytes that play the record on the node, accepted by the component, are a session the Lean verifier accepts. -/
theorem plays_verified {node : Node} {rec : Record} {a b e : ByteArray} (h : Plays node rec a b e) :
    Verified node rec := by
  obtain ⟨p, ev, -, -, -, hs, pub, prof, hc⟩ := h
  obtain ⟨params, g, coins, appendCoins, notBeforeNs, events, rs, s, _, hs', hrun, hout, -, -⟩ :=
    acceptIsVerified p node.reg ev pub prof hc
  rw [hs] at hs'
  simp only [Except.ok.injEq, Prod.mk.injEq] at hs'
  obtain ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩ := hs'
  exact ⟨rs, s, hrun, hout⟩

end Properties.PoubCheck.Component
