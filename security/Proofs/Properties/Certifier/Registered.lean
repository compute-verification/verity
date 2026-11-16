module

public import Properties.Certifier.Registered

/-!
# The network attester's link states its rate: the proof

`registered` finds the named link, refuses it unless it joins a network attester and a host, and refuses it where it
states no rate; what it returns carries the rate it found (`registeredLinkRated`).
-/

@[expose] public section

open NetworkCheck (Registered)

namespace Properties.Certifier.Registered

/-- **The named link states a rate**, by unfolding `registered`. -/
theorem registeredLinkRated : RegisteredLinkRated := by
  intro S head link key r h
  unfold Registered.registered at h
  split at h
  · rename_i k hk
    refine ⟨k, List.mem_of_find?_eq_some hk, by simpa using List.find?_some hk, ?_⟩
    by_cases hj : S.Joins k .certifier .host
    · refine ⟨hj, ?_⟩
      cases hb : k.β with
      | none =>
        simp only [hj, hb] at h
        repeat' split at h
        all_goals simp_all [bind, Except.bind, throw, throwThe, MonadExceptOf.throw]
      | some b =>
        simp only [hj, hb] at h
        repeat' split at h
        all_goals simp_all [bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw]
        all_goals (cases h; rfl)
    · simp_all [bind, Except.bind, throw, throwThe, MonadExceptOf.throw]
  · simp_all [throw, throwThe, MonadExceptOf.throw]

end Properties.Certifier.Registered
