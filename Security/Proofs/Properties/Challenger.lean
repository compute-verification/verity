import Properties.Challenger

/-!
# Proofs of `Properties.Challenger`

`liveAnswer`: `Live.factors` turns `answer`'s check into `checkAt` at the coins the evidence reads, and `answerAt` runs
the same decoders.
-/

namespace Properties.Challenger

theorem liveAnswer : LiveAnswer := by
  intro P L a b e h
  unfold Core.Protocol.answer at h
  split at h
  · simp at h
  · rename_i p hp
    split at h
    · simp at h
    · rename_i c hc
      split at h
      · simp at h
      · rename_i ev hev
        rw [L.factors] at h
        split at h
        · rename_i ξ hξ
          refine ⟨p, c, ev, ξ, hp, hc, hev, hξ, ?_⟩
          have hp' : (Core.canonicalJson a >>= (L.at ξ).params) = .ok p := hp
          have hc' : (L.at ξ).commitment b = .ok c := hc
          have hev' : (L.at ξ).evidence e = .ok ev := hev
          unfold Core.Protocol.answerAt Core.Protocol.answer
          rw [hp', hc', hev']
          exact h
        · split at h <;> simp at h

/-- ONE-4's target from the reader's property: `liveAnswer` gives the coins read, `DrawnByDevice` what they are. -/
theorem coinsFromLog (hd : DrawnByDevice) : CoinsFromLog := by
  intro P L R dev a b e h
  obtain ⟨p, c, ev, ξ, hp, hc, hev, hξ, hacc⟩ := liveAnswer P L a b e h
  refine ⟨p, c, ev, ξ, hp, hc, hev, hacc, ?_⟩
  rw [R.reads] at hξ
  cases hraw : drawCoins (R.key p) (Flock.hexOf (R.window p c)) R.label (R.after p c) (R.record ev) with
  | error r => rw [hraw] at hξ; cases hξ
  | ok raw =>
    rw [hraw] at hξ
    rcases hd dev (R.key p) (R.window p c) R.label (R.after p c) (R.record ev) raw hraw with
      ⟨s, hs, hopen, hrel⟩ | hb
    · exact .inl ⟨s, hs, hopen, by rw [← hrel]; exact hξ⟩
    · exact .inr hb

end Properties.Challenger
