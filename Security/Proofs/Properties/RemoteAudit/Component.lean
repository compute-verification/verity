import Properties.RemoteAudit.Component

/-!
# The bandwidth component accepts only what `Shared.run g` accepts, proved (`Properties.RemoteAudit.Component`)

`check` matches on `replay`, which binds `sessionOf` and then `Shared.run g` (`replay_ok`), and accepts only at a state
that ended accepted with a record, publishing `certify` of it at the session's parameters and g (`acceptIsVerified`).
`plays_verified` reads `Plays` as `Verified`: the session `sessionOf` decodes is the one it decodes, so it is the
record's at the node, at the node's windows a draw. `publishedEps` reads `certify`'s `eps` at the node's parameters
and g: `ratio` of a nonnegative rational is that rational (`ratio_val`), and `openingsCoins` is `Node.draws`.
-/

namespace Properties.RemoteAudit.Component

open Properties.RemoteAudit.Line

theorem replay_ok {p : RemoteAudit.Protocol.Params} {reg : RemoteAudit.Registration}
    {e : RemoteAudit.Protocol.Evidence} {s : RemoteAudit.Session.State} {params : RemoteAudit.Session.Params} {g : ℕ}
    (h : RemoteAudit.Protocol.replay p reg e = .ok (s, params, g)) :
    ∃ (coins appendCoins : Array ByteArray) (notBeforeNs : ℕ) (events : Array RemoteAudit.Session.Event)
      (rs : Array RemoteAudit.Session.Result),
      RemoteAudit.Protocol.sessionOf p e = .ok (params, g, coins, appendCoins, notBeforeNs, events) ∧
      RemoteAudit.Shared.run g (RemoteAudit.Session.State.init params reg coins appendCoins notBeforeNs) events =
        .ok (rs, s) := by
  unfold RemoteAudit.Protocol.replay at h
  cases hs : RemoteAudit.Protocol.sessionOf p e with
  | error why => simp [hs, bind, Except.bind] at h
  | ok x =>
    obtain ⟨params', g', coins, appendCoins, notBeforeNs, events⟩ := x
    simp only [hs, bind, Except.bind] at h
    cases hr : RemoteAudit.Shared.run g'
        (RemoteAudit.Session.State.init params' reg coins appendCoins notBeforeNs) events with
    | error why => simp [hr] at h
    | ok y =>
      obtain ⟨rs, s'⟩ := y
      simp only [hr, pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl, rfl⟩ := h
      exact ⟨coins, appendCoins, notBeforeNs, events, rs, rfl, hr⟩

theorem acceptIsVerified : AcceptIsVerified := by
  intro p reg e pub prof h
  unfold RemoteAudit.Protocol.check at h
  split at h
  · cases h
  · rename_i s params g hr
    split at h
    · rename_i sm hout hsum
      obtain ⟨coins, appendCoins, notBeforeNs, events, rs, hs, hrun⟩ := replay_ok hr
      rcases hc : RemoteAudit.Protocol.certify params g sm with ⟨pub', prof'⟩
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

/-- The ratio the component publishes of a nonnegative rational is that rational. -/
theorem ratio_val {q : ℚ} (hq : 0 ≤ q) :
    ((RemoteAudit.Protocol.ratio q).num : ℚ) / (RemoteAudit.Protocol.ratio q).den = q := by
  unfold RemoteAudit.Protocol.ratio
  dsimp only
  rw [show ((q.num.toNat : ℕ) : ℚ) = (q.num : ℚ) by exact_mod_cast Int.toNat_of_nonneg (Rat.num_nonneg.mpr hq)]
  exact Rat.num_div_den q

theorem eps_nonneg (params : RemoteAudit.Session.Params) (g : ℕ) : 0 ≤ RemoteAudit.Protocol.eps params g := by
  unfold RemoteAudit.Protocol.eps RemoteAudit.Protocol.DELTA
  positivity

theorem publishedEps : PublishedEps := by
  intro p reg e pub prof h
  obtain ⟨params, g, coins, appendCoins, notBeforeNs, events, rs, s, sm, hs, -, -, -, rfl⟩ :=
    acceptIsVerified p reg e pub prof h
  refine ⟨params, g, coins, appendCoins, notBeforeNs, events, hs, ?_⟩
  simp only [RemoteAudit.Protocol.certify]
  split
  · right
    intro node hp hg hδ
    subst hp hg
    dsimp only
    have hd : RemoteAudit.Protocol.openingsCoins node.params node.groupsPerCoin = node.draws := rfl
    rw [ratio_val (eps_nonneg _ _), Finset.sum_const, Finset.card_univ, Fintype.card_fin, smul_eq_mul, hδ]
    unfold RemoteAudit.Protocol.eps
    rw [hd]
    push_cast
    try ring
  · exact .inl ⟨rfl, rfl⟩

end Properties.RemoteAudit.Component
