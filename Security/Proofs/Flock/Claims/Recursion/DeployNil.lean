import Proofs.Flock.Claims.Recursion.Deploy
import Proofs.Flock.Claims.Bridge.CommittedNil

/-!
# The compiled verifier answers each round with its coin or with nothing

`answersCoinOrNil_V`: every message the compiled recursive verifier `RecDeploy.V` answers a round with is that round's
coin's bytes (an inner or staging round's, `ract_coin`, or a V\* session's round its outer verifier answers,
`nextO_msg`), or `[]`, where a V\* session before the last one accepts (`sessOut`). So
`Claims.Bridge.Pr_acc_runCM_le` bounds its run at Hello-committed coins.
-/

namespace RecDeploy

open FlockDeploy (Coin)

theorem actAt_coin {s spec sts} {as : List Claims.Msg} {cs : List Coin} {c : Coin}
    (h : actAt s spec sts as cs = .coin c) : cs.getLast? = some c := by
  unfold actAt at h
  repeat' split at h
  all_goals first | (cases h; done) | (cases h; assumption)

/-- The outer verifier answers a round with a coin only with the last of the coins it is given. -/
theorem actO_coin {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin} {c : Coin}
    (h : actO x as cs = .coin c) : cs.getLast? = some c := by
  unfold actO at h
  split at h
  · cases h
  · exact actAt_coin h

/-- **An outer session's message is its last coin's bytes**: a decision accepts or refuses. -/
theorem nextO_msg (H : Claims.Msg → Claims.Msg) {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    {m : Claims.Msg} (h : ((nextO x as cs).eval H).1 = .msg m) : ∃ c, cs.getLast? = some c ∧ m = c.bytes := by
  cases ha : actO x as cs with
  | rej =>
    rw [nextO_rej H ha] at h
    cases h
  | coin c =>
    rw [nextO_coin H ha] at h
    have h' : Claims.Out.msg c.bytes = .msg m := h
    exact ⟨c, actO_coin ha, (Claims.Out.msg.inj h').symm⟩
  | decide s spec sts link c₁ mids f =>
    rcases nextO_decide_out H ha with ho | ho <;> rw [ho] at h <;> cases h

section Steps

variable {IV : Claims.Verifier Coin} {st : Stage} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}

/-- An inner or staging round's coin is the last of the coins the verifier is given. -/
theorem ract_coin {c : Coin} (h : ract IV st x as cs = .coin c) : cs.getLast? = some c := by
  unfold ract at h
  split at h
  · cases h
  split_ifs at h with hk
  · split at h
    all_goals first | (cases h; done) | (cases h; assumption)
  dsimp only at h
  split at h
  · cases h
  split_ifs at h <;> cases h

/-- A session's coins are a suffix of the verifier's. -/
theorem ract_sess_drop {s j : ℕ} {xO : Claims.Msg} {as' : List Claims.Msg} {cs' : List Coin}
    (h : ract IV st x as cs = .sess s j xO as' cs') : ∃ off, cs' = cs.drop off := by
  unfold ract at h
  split at h
  · cases h
  split_ifs at h with hk
  · split at h <;> cases h
  dsimp only at h
  split at h
  · cases h
  split_ifs at h with hn hd
  obtain ⟨-, -, -, -, rfl⟩ := RAct.sess.inj h
  exact ⟨_, rfl⟩

end Steps

theorem getLast?_drop_concat {α : Type} : ∀ {l : List α} {a b : α} {n : ℕ},
    ((l ++ [a]).drop n).getLast? = some b → b = a
  | l, a, b, 0, h => by
    rw [List.drop_zero, List.getLast?_concat] at h
    exact (Option.some.inj h).symm
  | [], a, b, n + 1, h => by simp at h
  | _ :: l, a, b, n + 1, h => by
    have h' : ((l ++ [a]).drop n).getLast? = some b := by simpa using h
    exact getLast?_drop_concat h'

/-- **The compiled verifier answers each round with its coin's bytes or with nothing**: an inner or staging round, and
a V\* session's round its outer verifier answers, with the round's coin; a V\* session before the last one that accepts,
with `[]`. -/
theorem answersCoinOrNil_V (IV : Claims.Verifier Coin) (st : Stage) (H : Claims.Msg → Claims.Msg) (x : Claims.Msg) :
    Claims.Bridge.AnswersCoinOrNil H (V IV st) x FlockDeploy.Coin.bytes := by
  intro as cs c m h
  change ((next IV st x as (cs ++ [c])).eval H).1 = .msg m at h
  cases hr : ract IV st x as (cs ++ [c]) with
  | rej =>
    rw [next_rej H hr] at h
    cases h
  | coin c' =>
    rw [next_coin H hr] at h
    have h' : Claims.Out.msg c'.bytes = .msg m := h
    have hl := ract_coin hr
    rw [List.getLast?_concat] at hl
    refine .inl ?_
    rw [← Claims.Out.msg.inj h', Option.some.inj hl]
  | sess s j xO as' cs' =>
    rw [next_sess H hr] at h
    simp only at h
    cases ho : ((nextO xO as' cs').eval H).1 with
    | msg m' =>
      rw [ho] at h
      have hm : m' = m := by simpa [sessOut] using h
      obtain ⟨c', hc', rfl⟩ := nextO_msg H ho
      obtain ⟨off, hoff⟩ := ract_sess_drop hr
      rw [hoff] at hc'
      refine .inl ?_
      rw [← hm, getLast?_drop_concat hc']
    | acc =>
      rw [ho] at h
      refine .inr ?_
      simp only [sessOut] at h
      split_ifs at h <;> first | (cases h; done) | exact (Claims.Out.msg.inj h).symm
    | rej =>
      rw [ho] at h
      simp [sessOut] at h

end RecDeploy
