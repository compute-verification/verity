module

public import Properties.Poub.Remote.SubprotocolDeployed

/-!
# The bandwidth component accepts only what `Shared.run g` accepts, proved (`Properties.Poub.Subprotocol`)

`check` matches on `replay`, which binds `sessionOf` and then `Shared.run g` (`replay_ok`), and accepts only at a state
that ended accepted with a record, publishing `certify` of it at the session's parameters, g, registration and append
coins (`acceptIsVerified`).
`plays_verified` reads `Plays` as `Verified`: the session `sessionOf` decodes is the one it decodes, so it is the
record's at the node, at the node's windows a sample; `plays_iff_publishes` reads it as `Publishes` at some public
outputs, and `publishes_publishedQBytes_iff` reads `PublishedQBytes`, which spells the acceptance out, as its statement
at `Publishes`. `publishedEps` reads `certify`'s `eps`: the constant where `refusal` passes the session, which bounds
`constantTerms` there (`constantTerms_le`), and 0 elsewhere.
-/

@[expose] public section

namespace Properties.Poub.Subprotocol

open Properties.Poub.Line

theorem replay_ok {p : Poub.Protocol.Params} {reg : Poub.Registration}
    {e : Poub.Protocol.Evidence} {s : Poub.Session.State} {params : Poub.Session.Params} {g : ℕ}
    {appendCoins : Array ByteArray}
    (h : Poub.Protocol.replay p reg e = .ok (s, params, g, appendCoins)) :
    ∃ (coins : Array ByteArray) (notBeforeNs : ℕ) (events : Array Poub.Session.Event)
      (rs : Array Poub.Session.Result),
      Poub.Protocol.sessionOf p e = .ok (params, g, coins, appendCoins, notBeforeNs, events) ∧
      Poub.Shared.run g (Poub.Session.State.init params reg coins appendCoins notBeforeNs) events =
        .ok (rs, s) := by
  unfold Poub.Protocol.replay at h
  cases hs : Poub.Protocol.sessionOf p e with
  | error why => simp [hs, bind, Except.bind] at h
  | ok x =>
    obtain ⟨params', g', coins, appendCoins', notBeforeNs, events⟩ := x
    simp only [hs, bind, Except.bind] at h
    cases hr : Poub.Shared.run g'
        (Poub.Session.State.init params' reg coins appendCoins' notBeforeNs) events with
    | error why => simp [hr] at h
    | ok y =>
      obtain ⟨rs, s'⟩ := y
      simp only [hr, pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl, rfl, rfl⟩ := h
      exact ⟨coins, notBeforeNs, events, rs, rfl, hr⟩

theorem acceptIsVerified : AcceptIsVerified := by
  intro p reg e pub prof h
  unfold Poub.Protocol.check at h
  split at h
  · cases h
  · rename_i s params g appendCoins hr
    split at h
    · rename_i sm hout hsum
      obtain ⟨coins, notBeforeNs, events, rs, hs, hrun⟩ := replay_ok hr
      rcases hc : Poub.Protocol.certify params g reg appendCoins.size sm with ⟨pub', prof'⟩
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
          ResourceAccounting.Remote.Sample.pooled node.samples (fun c => min (rec.sampleReceiptsAt node c) receiptMax)
            (fun c => min node.params.samples (min (rec.sampleReceiptsAt node c) receiptMax))
            (Poub.Protocol.DELTA : ℝ) ≤
        (Poub.Codec.BLOCK_BYTES : ℝ) * ((rec.receipts node).toList.sum : ℕ) := by
  unfold PublishedQBytes Publishes
  constructor
  · rintro h node rec a b e pub ⟨p, ev, ha, hb, he, hs, prof, hc⟩
    exact h node rec a b e pub p ev prof ha hb he hs hc
  · intro h node rec a b e pub p ev prof ha hb he hs hc
    exact h node rec a b e pub ⟨p, ev, ha, hb, he, hs, prof, hc⟩

theorem sqrt_two_ge_one : (1 : ℝ) ≤ Real.sqrt 2 := by
  nlinarith [Real.sq_sqrt (show (0 : ℝ) ≤ 2 by norm_num), Real.sqrt_nonneg 2]

/-- Inside `refusal`'s caps the constant bounds every term of `constantTerms`. -/
theorem constantTerms_le {params : Poub.Session.Params} {g : ℕ} {reg : Poub.Registration} {n : ℕ}
    (h : Poub.Protocol.refusal params g reg n = none) :
    constantTerms params g reg n (Poub.Protocol.DELTA : ℝ) ≤ (Poub.Protocol.EPS : ℝ) := by
  cases hp : params.pace with
  | none => simp [Poub.Protocol.refusal, Poub.Protocol.sampleReceipts, hp] at h
  | some pace =>
    have hu : Poub.Protocol.sampleReceipts params g reg n =
        some (max g 1 * pace.k * Poub.Protocol.mostBlocks reg n) := by
      simp [Poub.Protocol.sampleReceipts, hp]
    simp only [Poub.Protocol.refusal, hp, hu, Poub.Protocol.PASSES_MAX, Poub.Protocol.BINDING_SQ_MAX,
      Poub.Protocol.REJECTION_MAX] at h
    split_ifs at h with h1 h2 h3 h4 h5
    all_goals try exact absurd h (Option.some_ne_none _)
    simp only [constantTerms, hp, hu]
    set G := Poub.Protocol.openingsCoins params g
    set U := max g 1 * pace.k * Poub.Protocol.mostBlocks reg n
    have hP : (params.windows : ℝ) * pace.k ≤ 2 ^ 30 := by exact_mod_cast not_lt.mp h3
    have hN : (Poub.Protocol.mostBlocks reg n : ℝ) ≤ 2 ^ 64 := by exact_mod_cast not_lt.mp h2
    have hB : (G : ℝ) * G * U * params.samples ≤ 2 ^ 92 := by exact_mod_cast not_lt.mp h4
    have hGs : (G : ℝ) * params.samples ≤ 2 ^ 40 := by exact_mod_cast not_lt.mp h5
    have hx : 0 ≤ (G : ℝ) * Real.sqrt (U * params.samples) := by positivity
    have hsq : (G : ℝ) * Real.sqrt (U * params.samples) ≤ 2 ^ 46 := by
      have : ((G : ℝ) * Real.sqrt (U * params.samples)) ^ 2 ≤ (2 ^ 46) ^ 2 := by
        rw [mul_pow, Real.sq_sqrt (by positivity)]
        nlinarith [hB]
      nlinarith [this, hx]
    have hbind : (G : ℝ) * Real.sqrt (U * params.samples) * (2 * 2 ^ 80 + 2 * params.samples * 69) /
        (2 ^ 256 * Real.sqrt 2) ≤ 2 ^ 46 * 2 ^ 82 / 2 ^ 256 := by
      rcases Nat.eq_zero_or_pos G with hG | hG
      · rw [hG]
        simp only [Nat.cast_zero, zero_mul, zero_div]
        positivity
      · have hs : (params.samples : ℝ) ≤ 2 ^ 40 := by
          have : (1 : ℝ) ≤ G := by exact_mod_cast hG
          nlinarith [hGs, (Nat.cast_nonneg params.samples : (0 : ℝ) ≤ params.samples)]
        have hq : (2 : ℝ) * 2 ^ 80 + 2 * params.samples * 69 ≤ 2 ^ 82 := by nlinarith [hs]
        have hd : (2 : ℝ) ^ 256 ≤ 2 ^ 256 * Real.sqrt 2 := le_mul_of_one_le_right (by positivity) sqrt_two_ge_one
        calc (G : ℝ) * Real.sqrt (U * params.samples) * (2 * 2 ^ 80 + 2 * params.samples * 69) /
              (2 ^ 256 * Real.sqrt 2)
            ≤ (G : ℝ) * Real.sqrt (U * params.samples) * (2 * 2 ^ 80 + 2 * params.samples * 69) / 2 ^ 256 :=
              div_le_div_of_nonneg_left (by positivity) (by positivity) hd
          _ ≤ 2 ^ 46 * 2 ^ 82 / 2 ^ 256 := by gcongr
    have hrest : (Poub.Protocol.DELTA : ℝ) + G * params.samples / 2 ^ 256 +
        (2 ^ 80 / 2 ^ 216 + (2 ^ 80 + params.windows * pace.k) *
          (1 / 2 ^ 219 + Poub.Protocol.mostBlocks reg n / 2 ^ 512) +
          (2 ^ 80 + params.windows * pace.k) ^ 2 / 2 ^ 513) + params.windows * pace.k / 2 ^ 140 ≤
        (Poub.Protocol.DELTA : ℝ) + 2 ^ 40 / 2 ^ 256 +
        (2 ^ 80 / 2 ^ 216 + (2 ^ 80 + 2 ^ 30) * (1 / 2 ^ 219 + 2 ^ 64 / 2 ^ 512) +
          (2 ^ 80 + 2 ^ 30) ^ 2 / 2 ^ 513) + 2 ^ 30 / 2 ^ 140 := by
      gcongr
    have hnum : (Poub.Protocol.DELTA : ℝ) + 2 ^ 40 / 2 ^ 256 +
        (2 ^ 80 / 2 ^ 216 + (2 ^ 80 + 2 ^ 30) * (1 / 2 ^ 219 + 2 ^ 64 / 2 ^ 512) +
          (2 ^ 80 + 2 ^ 30) ^ 2 / 2 ^ 513) + 2 ^ 46 * 2 ^ 82 / 2 ^ 256 + 2 ^ 30 / 2 ^ 140 ≤
        (Poub.Protocol.EPS : ℝ) := by
      simp only [Poub.Protocol.DELTA, Poub.Protocol.EPS]
      norm_num
    linarith [hbind, hrest, hnum]

theorem publishedEps : PublishedEps := by
  intro p reg e pub prof h
  obtain ⟨params, g, coins, appendCoins, notBeforeNs, events, rs, s, sm, hs, -, -, -, rfl⟩ :=
    acceptIsVerified p reg e pub prof h
  refine ⟨params, g, coins, appendCoins, notBeforeNs, events, hs, ?_⟩
  cases hr : Poub.Protocol.refusal params g reg appendCoins.size with
  | some why => exact .inl (by simp [Poub.Protocol.certify, hr])
  | none =>
    simp only [Poub.Protocol.certify, hr]
    split
    · exact .inr ⟨rfl, constantTerms_le hr⟩
    · exact .inl ⟨rfl, rfl⟩

end Properties.Poub.Subprotocol
