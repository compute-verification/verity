import Properties.Challenger

/-!
# Proofs of `Properties.Challenger`

`liveAnswer`: `Live.factors` turns `answer`'s check into `checkAt` at the coins the evidence reads, and `answerAt` runs
the same decoders. `coinsFromLog`: ONE-4's target from `liveAnswer` and the reader's property `DrawnByDevice`.
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
          cases ho
        | ok ξ =>
          refine ⟨p, c, ev, ξ, rfl, rfl, rfl, hξ, ?_⟩
          obtain ⟨pub, prof, ho⟩ := hacc
          rw [L.factors, hξ] at ho
          have hp' : (Core.canonicalJson a >>= (L.at ξ).params) = .ok p := hp
          have hc' : (L.at ξ).commitment b = .ok c := hc
          have hev' : (L.at ξ).evidence e = .ok ev := hev
          exact (answer_accept_iff (L.at ξ) hp' hc' hev').2 ⟨pub, prof, ho⟩

theorem coinsFromLog (hd : DrawnByDevice) : CoinsFromLog := by
  intro P L R dev a b e h
  obtain ⟨p, c, ev, ξ, hp, hc, hev, hξ, hacc⟩ := liveAnswer P L a b e h
  refine ⟨p, c, ev, ξ, hp, hc, hev, hacc, fun hlive => ?_⟩
  rw [R.reads p hlive] at hξ
  cases hraw : SampledProofs.Challenger.drawCoins (R.key p) (Flock.hexOf (R.window p c)) R.label (R.after p c)
      (R.record ev) with
  | error r => rw [hraw] at hξ; cases hξ
  | ok raw =>
    rw [hraw] at hξ
    rcases hd dev (R.key p) (R.window p c) R.label (R.after p c) (R.record ev) raw hraw with
      ⟨s, hs, hopen, hrel⟩ | hb
    · exact .inl ⟨s, hs, hopen, by rw [← hrel]; exact hξ⟩
    · exact .inr hb

end Properties.Challenger
