module

public import Properties.Poub.Remote.SubprotocolDeployed

/-!
# The bandwidth component accepts only what `Shared.run g` accepts, proved (`Properties.Poub.Subprotocol`)

`check` matches on `replay`, which binds `sessionOf` and then `Shared.run g` (`replay_ok`), and accepts only at a state
that ended accepted with a record, publishing `certify` of it at the session's parameters and g (`acceptIsVerified`).
`plays_verified` reads `Plays` as `Verified`: the session `sessionOf` decodes is the one it decodes, so it is the
record's at the node, at the node's windows a sample; `plays_iff_publishes` reads it as `Publishes` at some public
outputs, and `publishes_publishedQBytes_iff` reads `PublishedQBytes`, which spells the acceptance out, as its statement
at `Publishes`. `publishedEps` reads `certify`'s `eps` at the node's parameters
and g: `ratio` of a nonnegative rational is that rational (`ratio_val`), and `openingsCoins` is `Node.samples`.
-/

@[expose] public section

namespace Properties.Poub.Subprotocol

open Properties.Poub.Line

theorem replay_ok {p : Poub.Protocol.Params} {reg : Poub.Registration}
    {e : Poub.Protocol.Evidence} {s : Poub.Session.State} {params : Poub.Session.Params} {g : ℕ}
    (h : Poub.Protocol.replay p reg e = .ok (s, params, g)) :
    ∃ (coins appendCoins : Array ByteArray) (notBeforeNs : ℕ) (events : Array Poub.Session.Event)
      (rs : Array Poub.Session.Result),
      Poub.Protocol.sessionOf p e = .ok (params, g, coins, appendCoins, notBeforeNs, events) ∧
      Poub.Shared.run g (Poub.Session.State.init params reg coins appendCoins notBeforeNs) events =
        .ok (rs, s) := by
  unfold Poub.Protocol.replay at h
  cases hs : Poub.Protocol.sessionOf p e with
  | error why => simp [hs, bind, Except.bind] at h
  | ok x =>
    obtain ⟨params', g', coins, appendCoins, notBeforeNs, events⟩ := x
    simp only [hs, bind, Except.bind] at h
    cases hr : Poub.Shared.run g'
        (Poub.Session.State.init params' reg coins appendCoins notBeforeNs) events with
    | error why => simp [hr] at h
    | ok y =>
      obtain ⟨rs, s'⟩ := y
      simp only [hr, pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl, rfl⟩ := h
      exact ⟨coins, appendCoins, notBeforeNs, events, rs, rfl, hr⟩

theorem acceptIsVerified : AcceptIsVerified := by
  intro p reg e pub prof h
  unfold Poub.Protocol.check at h
  split at h
  · cases h
  · rename_i s params g hr
    split at h
    · rename_i sm hout hsum
      obtain ⟨coins, appendCoins, notBeforeNs, events, rs, hs, hrun⟩ := replay_ok hr
      rcases hc : Poub.Protocol.certify params g sm with ⟨pub', prof'⟩
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

/-- `Plays` is `Publishes` at some public outputs. -/
theorem plays_iff_publishes {node : Node} {rec : Record} {a b e : ByteArray} :
    Plays node rec a b e ↔ ∃ pub, Publishes node rec a b e pub :=
  ⟨fun ⟨p, ev, ha, hb, he, hs, pub, prof, hc⟩ => ⟨pub, p, ev, ha, hb, he, hs, prof, hc⟩,
    fun ⟨pub, p, ev, ha, hb, he, hs, prof, hc⟩ => ⟨p, ev, ha, hb, he, hs, pub, prof, hc⟩⟩

/-- `PublishedQBytes`, which spells the acceptance out, is the same proposition as its statement at `Publishes`. -/
theorem publishes_publishedQBytes_iff : PublishedQBytes ↔
    ∀ (node : Node) (rec : Record) (a b e : ByteArray) (pub : Poub.Protocol.Public),
      Publishes node rec a b e pub →
      (pub.qBytes : ℝ) + (Poub.Codec.BLOCK_BYTES : ℝ) *
          Accounting.Remote.Sample.pooled node.samples (fun c => min (rec.sampleReceiptsAt node c) receiptMax)
            (fun c => min node.params.samples (min (rec.sampleReceiptsAt node c) receiptMax))
            (Poub.Protocol.DELTA : ℝ) ≤
        (Poub.Codec.BLOCK_BYTES : ℝ) * ((rec.receipts node).toList.sum : ℕ) := by
  unfold PublishedQBytes Publishes
  constructor
  · rintro h node rec a b e pub ⟨p, ev, ha, hb, he, hs, prof, hc⟩
    exact h node rec a b e pub p ev prof ha hb he hs hc
  · intro h node rec a b e pub p ev prof ha hb he hs hc
    exact h node rec a b e pub ⟨p, ev, ha, hb, he, hs, prof, hc⟩

/-- The ratio the component publishes of a nonnegative rational is that rational. -/
theorem ratio_val {q : ℚ} (hq : 0 ≤ q) :
    ((Poub.Protocol.ratio q).num : ℚ) / (Poub.Protocol.ratio q).den = q := by
  unfold Poub.Protocol.ratio
  dsimp only
  rw [show ((q.num.toNat : ℕ) : ℚ) = (q.num : ℚ) by exact_mod_cast Int.toNat_of_nonneg (Rat.num_nonneg.mpr hq)]
  exact Rat.num_div_den q

theorem eps_nonneg (params : Poub.Session.Params) (g : ℕ) : 0 ≤ Poub.Protocol.eps params g := by
  unfold Poub.Protocol.eps Poub.Protocol.DELTA
  positivity

theorem publishedEps : PublishedEps := by
  intro p reg e pub prof h
  obtain ⟨params, g, coins, appendCoins, notBeforeNs, events, rs, s, sm, hs, -, -, -, rfl⟩ :=
    acceptIsVerified p reg e pub prof h
  refine ⟨params, g, coins, appendCoins, notBeforeNs, events, hs, ?_⟩
  simp only [Poub.Protocol.certify]
  split
  · right
    intro node hp hg hδ
    subst hp hg
    dsimp only
    have hd : Poub.Protocol.openingsCoins node.params node.groupsPerCoin = node.samples := rfl
    rw [ratio_val (eps_nonneg _ _), Finset.sum_const, Finset.card_univ, Fintype.card_fin, smul_eq_mul, hδ]
    unfold Poub.Protocol.eps
    rw [hd]
    push_cast
    try ring
  · exact .inl ⟨rfl, rfl⟩

end Properties.Poub.Subprotocol
