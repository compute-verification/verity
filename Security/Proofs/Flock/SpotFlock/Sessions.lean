module

public import Proofs.Flock.SpotFlock.Code
public import Proofs.Flock.Claims.Sampling.Deployed
public import Proofs.Flock.Claims.Sampling.SessionSteps

/-!
# SPOT[Flock]'s sessions are Flock runs

`SpotFlock.steps h rel`: SPOT[Flock]'s two-stage verifier (`SpotFlock.code h`) has the step structure
`Claims.Sampling.SessionSteps`. Rounds 0 to 2 answer with no `H` call: the two stages' coins, then the session
statement's check (`stmtOk`), which answers `[]` or refuses. Rounds 3 on are Flock's masked verifier at the round-2
statement, whose round count `stmtOk` holds to `k₀ h`. The session read off a transcript is that statement where it is in
scope (`sessionOk`), which `stmtOk` checks, so at every run that passed round 2 (`sessionOk_of_stmtOk`). A run refused
before its session shows `xS₀`'s refused run, at which Flock refuses the first message `[]` (`refuses_empty`); Flock's
verifier has a round at every statement (`k₀_pos`).

So `sessionRuns`: `SessionRuns` holds at that code, for every relation and every `H`.
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy Finset

variable (h : Held)

/-- **Flock's masked verifier has a round at every statement.** -/
theorem k₀_pos : 0 < k₀ h := by
  unfold k₀ FlockDeploy.V
  dsimp only
  split <;> omega

/-- **Flock refuses the first message `[]`**, with no `H` call, at every statement. -/
theorem refuses_empty (x : Msg) (c : FlockDeploy.Coin) : FlockDeploy.next true x [[]] [c] = .ret .rej := by
  have ha : FlockDeploy.act true x [[]] [c] = .rej := by
    unfold FlockDeploy.act
    split <;> rfl
  simp only [FlockDeploy.next, ha]

/-- **A session statement the run checks has `k₀` rounds.** -/
theorem k_of_stmtOk {x m₁ xS : Msg} {c₀ c₁ : FlockDeploy.Coin} (hok : stmtOk h x m₁ xS c₀ c₁ = true) :
    (FlockDeploy.V true).k xS = k₀ h := by
  unfold stmtOk at hok
  split at hok
  · simp only [Bool.and_eq_true, beq_iff_eq] at hok
    exact hok.1.1.2
  · exact absurd hok Bool.false_ne_true

/-- **A session statement the run checks is in scope** (`sessionOk`). -/
theorem sessionOk_of_stmtOk {x m₁ xS : Msg} {c₀ c₁ : FlockDeploy.Coin} (hok : stmtOk h x m₁ xS c₀ c₁ = true) :
    sessionOk h xS = true := by
  unfold stmtOk at hok
  split at hok
  · simp only [Bool.and_eq_true] at hok
    exact hok.1.2
  · exact absurd hok Bool.false_ne_true

/-- **The two-stage verifier's step structure**: a prefix of 3 rounds, then Flock's masked verifier at the round-2
statement for `k₀ h` rounds; a refused prefix shows `xS₀`'s refused run. -/
noncomputable def steps (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) :
    SessionSteps (deployOf (FlockDeploy.deployZk rel) (code h)) where
  pre _ := 3
  kS _ := k₀ h
  stmt _ as := as.getD 2 []
  xR _ := h.xS₀
  pre_pos _ := Nat.zero_lt_succ 2
  k_eq _ := rfl
  next_pre x as cs hne hle hcs := by
    change ∃ o, next h x as cs = .ret o ∧ o ≠ .acc
    rcases as with _ | ⟨a₀, _ | ⟨a₁, _ | ⟨a₂, _ | ⟨a₃, as⟩⟩⟩⟩
    · exact absurd rfl hne
    · rcases cs with _ | ⟨c₀, _ | ⟨c₁, cs⟩⟩ <;> simp at hcs
      exact ⟨_, rfl, nofun⟩
    · rcases cs with _ | ⟨c₀, _ | ⟨c₁, _ | ⟨c₂, cs⟩⟩⟩ <;> simp at hcs
      exact ⟨_, rfl, nofun⟩
    · rcases cs with _ | ⟨c₀, _ | ⟨c₁, _ | ⟨c₂, _ | ⟨c₃, cs⟩⟩⟩⟩ <;> simp at hcs
      exact ⟨_, rfl, by split <;> nofun⟩
    · simp only [List.length_cons] at hle
      omega
  next_post x as cs hlt hcs := by
    rcases as with _ | ⟨a₀, _ | ⟨a₁, _ | ⟨a₂, _ | ⟨a₃, as⟩⟩⟩⟩ <;> simp at hlt
    rcases cs with _ | ⟨c₀, _ | ⟨c₁, _ | ⟨c₂, _ | ⟨c₃, cs⟩⟩⟩⟩ <;> simp at hcs
    rfl
  k_stmt x as cs m hlen hcs hm := by
    rcases as with _ | ⟨a₀, _ | ⟨a₁, _ | ⟨a₂, _ | ⟨a₃, as⟩⟩⟩⟩ <;> simp at hlen
    rcases cs with _ | ⟨c₀, _ | ⟨c₁, _ | ⟨c₂, _ | ⟨c₃, cs⟩⟩⟩⟩ <;> simp at hcs
    change OC.ret (if stmtOk h x a₁ a₂ c₀ c₁ then Out.msg [] else Out.rej) = OC.ret (Out.msg m) at hm
    change (FlockDeploy.V true).k a₂ = k₀ h
    split at hm
    · exact k_of_stmtOk h ‹_›
    · cases hm
  rounds_iff x u t := by
    have e : ∀ t : Fin ((V h).k x), t ∈ ((univ.filter fun t : Fin ((V h).k x) => 2 < t.val).filter
        fun t => (order h x).second < t) ↔ 3 ≤ t.val := fun t => by
      rw [Finset.mem_filter, Finset.mem_filter]
      show (t ∈ univ ∧ 2 < t.val) ∧ 1 < t.val ↔ 3 ≤ t.val
      constructor
      · rintro ⟨⟨_, h2⟩, _⟩
        omega
      · intro h3
        exact ⟨⟨mem_univ t, by omega⟩, by omega⟩
    exact e t
  k_xR _ := rfl
  kS_pos _ := k₀_pos h
  refuses _ c := refuses_empty h.xS₀ c
  session_pass x u ms rest out cs m hlen hcs hms hm := by
    rcases ms with _ | ⟨⟨a₀, o₀⟩, _ | ⟨⟨a₁, o₁⟩, _ | ⟨⟨a₂, o₂⟩, _ | ⟨e, ms⟩⟩⟩⟩ <;> simp at hlen
    obtain ⟨m₀, h₀⟩ := hms (a₀, o₀) (by simp)
    obtain ⟨m₁, h₁⟩ := hms (a₁, o₁) (by simp)
    obtain ⟨m₂, h₂⟩ := hms (a₂, o₂) (by simp)
    dsimp only at h₀ h₁ h₂
    subst h₀ h₁ h₂
    rcases cs with _ | ⟨c₀, _ | ⟨c₁, _ | ⟨c₂, _ | ⟨c₃, cs⟩⟩⟩⟩ <;> simp at hcs
    change OC.ret (if stmtOk h x a₁ a₂ c₀ c₁ then Out.msg [] else Out.rej) = OC.ret (Out.msg m) at hm
    split at hm
    · rename_i hok
      show (if sessionOk h a₂ then some (a₂, (⟨rest, out⟩ : Seen)) else some (h.xS₀, refusedSeen)) =
        some (a₂, ⟨rest, out⟩)
      rw [ite_eq_left (sessionOk_of_stmtOk h hok)]
    · cases hm
  session_refused x u ms a hlen hms := by
    rcases ms with _ | ⟨⟨a₀, o₀⟩, _ | ⟨⟨a₁, o₁⟩, _ | ⟨e, ms⟩⟩⟩
    · rfl
    · obtain ⟨m₀, h₀⟩ := hms (a₀, o₀) (by simp)
      dsimp only at h₀
      subst h₀
      rfl
    · obtain ⟨m₀, h₀⟩ := hms (a₀, o₀) (by simp)
      obtain ⟨m₁, h₁⟩ := hms (a₁, o₁) (by simp)
      dsimp only at h₀ h₁
      subst h₀ h₁
      rfl
    · simp only [List.length_cons] at hlen
      omega

/-- **SPOT[Flock]'s sessions are Flock runs** (`SessionRuns`), at every relation and every `H`. -/
theorem sessionRuns (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (H : Msg → Msg) :
    SessionRuns H (SamplingDeploy.deploy rel (code h)) :=
  sessionRuns_of_steps (steps h rel) H

end SpotFlock
