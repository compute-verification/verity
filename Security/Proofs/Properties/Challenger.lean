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

end Properties.Challenger
