import Properties.Challenger

/-!
# Proofs of `Properties.Challenger`

`liveAnswer`: `Live.factors` turns `answer`'s check into `checkAt` at the coins the evidence reads, and `answerAt` runs
the same decoders. `coinsFromLog`: ONE-4's target from `liveAnswer` and the reader's property `DrawnByDevice`.
-/

namespace Properties.Challenger

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
        cases hξ : L.coins p c ev with
        | error r => simp [hp, hc, hev, L.factors, hξ] at h
        | ok ξ =>
          refine ⟨p, c, ev, ξ, hp, hc, hev, hξ, ?_⟩
          simpa [Core.Protocol.answerAt, Core.Live.at, Core.Protocol.answer, hp, hc, hev, L.factors, hξ] using h

theorem coinsFromLog (hd : DrawnByDevice) : CoinsFromLog := by
  intro P L R dev a b e h
  obtain ⟨p, c, ev, ξ, hp, hc, hev, hξ, hacc⟩ := liveAnswer P L a b e h
  refine ⟨p, c, ev, ξ, hp, hc, hev, hacc, ?_⟩
  rw [R.reads] at hξ
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
