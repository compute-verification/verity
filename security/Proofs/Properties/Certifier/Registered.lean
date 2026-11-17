module

public import Properties.Certifier.Registered

/-! `registered` refuses a link that joins no network attester and host or states no rate, so what it returns carries a
rate. -/

@[expose] public section

open NetworkCheck (Registered)

namespace Properties.Certifier.Registered

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
