import Properties.Challenger

/-!
# Proofs of `Properties.Challenger`

`liveAnswer`: `Live.factors` turns `answer`'s check into `checkAt` at the coins the evidence reads, and `answerAt` runs
the same decoders. `coinsFromLog`: the coins' target from `liveAnswer` and the reader's property `DrawnByDevice`.
`drawnByDevice_nontrivial`: a reader that accepts every record fails `DrawnByDevice`'s statement.
`oneStage_check_of_accept`: one-stage's live check accepts only what its coin-free `check` accepts, so the properties
proved about `check` (`acceptMeans_iff`) hold of what it accepts.
-/

namespace Properties.Challenger

/-- `answer` accepts exactly when its three byte strings decode and the check accepts what they decode to. -/
theorem answer_accept_iff (Q : Core.Protocol) {a b e : ByteArray} {p : Q.Params} {c : Q.Commitment} {ev : Q.Evidence}
    (hp : (Core.canonicalJson a >>= Q.params) = .ok p) (hc : Q.commitment b = .ok c) (hev : Q.evidence e = .ok ev) :
    (Q.answer a b e).verdict = .accept ↔ ∃ pub prof, Q.check p c ev = .accept pub prof := by
  unfold Core.Protocol.answer
  rw [hp]
  simp only []
  rw [hc]
  simp only []
  rw [hev]
  simp only []
  cases Q.check p c ev <;> simp

theorem liveAnswer : LiveAnswer := by
  intro P L a b e h
  unfold Core.Protocol.answer at h
  cases hp : (Core.canonicalJson a >>= P.params) with
  | error r => simp [hp] at h
  | ok p =>
    cases hc : P.commitment b with
    | error r => simp [hp, hc] at h
    | ok c =>
      cases hev : P.evidence e with
      | error r => simp [hp, hc, hev] at h
      | ok ev =>
        have hacc : ∃ pub prof, P.check p c ev = .accept pub prof := by
          cases ho : P.check p c ev <;> simp [hp, hc, hev, ho] at h ⊢
        cases hξ : L.coins p c ev with
        | error r =>
          obtain ⟨pub, prof, ho⟩ := hacc
          rw [L.factors, hξ] at ho
          simp [Core.atCoins] at ho
        | ok ξ =>
          refine ⟨p, c, ev, ξ, rfl, rfl, rfl, hξ, ?_⟩
          obtain ⟨pub, prof, ho⟩ := hacc
          rw [L.factors, hξ] at ho
          simp only [Core.atCoins] at ho
          have hp' : (Core.canonicalJson a >>= (L.at ξ).params) = .ok p := hp
          have hc' : (L.at ξ).commitment b = .ok c := hc
          have hev' : (L.at ξ).evidence e = .ok ev := hev
          exact (answer_accept_iff (L.at ξ) hp' hc' hev').2 ⟨pub, prof, ho⟩

theorem coinsFromLog (hd : DrawnByDevice) : CoinsFromLog := by
  intro P L R dev a b e h
  obtain ⟨p, c, ev, ξ, hp, hc, hev, hξ, hacc⟩ := liveAnswer P L a b e h
  refine ⟨p, c, ev, ξ, hp, hc, hev, hacc, fun d hd' => ?_⟩
  obtain ⟨raw, hraw, hdec⟩ := R.reads p c d hd' ev ξ hξ
  rcases hd dev d.key d.window d.label d.after (R.record ev) raw hraw with ⟨s, hs, hopen, hrel⟩ | hb
  · exact .inl ⟨s, hs, hopen, by rw [← hrel]; exact hdec⟩
  · exact .inr hb

/-- `DrawnByDevice`'s statement is not trivially true: a reader that accepts every record fails it, since the empty
device released nothing and the finder finds no break in an empty run. -/
theorem drawnByDevice_nontrivial : ¬ DrawnByDeviceAt (fun _ _ _ _ _ => .ok ByteArray.empty) := by
  intro h
  have hnone : findBreak ⟨fun _ => [], fun _ => []⟩ ⟨Lms.deployed, ByteArray.empty⟩ "" Lean.Json.null = none := rfl
  rcases h ⟨fun _ => [], fun _ => []⟩ ⟨Lms.deployed, ByteArray.empty⟩ "" "" ByteArray.empty Lean.Json.null
      ByteArray.empty rfl with ⟨s, hs, -⟩ | ⟨w, hw, -⟩
  · simp at hs
  · rw [hnone] at hw
    cases hw

/-- What one-stage's protocol accepts, its coin-free `check` accepts: `checkAt` only adds the coins' refusals. -/
theorem oneStage_check_of_accept {p : SampledProofs.OneStage.Params} {c : SampledProofs.OneStage.Commitment}
    {e : SampledProofs.OneStage.Evidence} {pub : SampledProofs.OneStage.Public} {prof : Core.Integrity}
    (h : SampledProofs.OneStage.protocol.check p c e = .accept pub prof) :
    SampledProofs.OneStage.check p c e = .accept pub prof := by
  change Core.atCoins (SampledProofs.OneStage.coinsOf p c e)
    (fun ξ => SampledProofs.OneStage.checkAt p c ξ e) = _ at h
  cases hξ : SampledProofs.OneStage.coinsOf p c e with
  | error r => rw [hξ] at h; simp [Core.atCoins] at h
  | ok ξ =>
    rw [hξ] at h
    simp only [Core.atCoins, SampledProofs.OneStage.checkAt] at h
    cases hw : ξ.bind (SampledProofs.OneStage.coinsWhy p c e) with
    | some why => rw [hw] at h; simp at h
    | none => rw [hw] at h; exact h

end Properties.Challenger
