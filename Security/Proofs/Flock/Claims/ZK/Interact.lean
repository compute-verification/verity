import Proofs.Flock.Claims.ZK.GK

/-!
# Runs as transcripts

* `interact`: the transcript of `n` rounds of a prover's next message `pm` (evaluated) against a verifier's answers
  `ask`. `runStar_msgs`: `runStar`'s messages are `interact`'s; `spRun_eval`: so are the simulator's `spRun`.
* `interact_take_congr`, `interact_congr`: two provers that agree on a run's histories, up to a round, give the same
  transcript up to that round.
* The evaluation of composed programs (`ocBind`, `spBind`, `ocSP`): results, calls and logs.
-/

namespace Claims.ZKMal

open Claims

/-! ## Composed programs -/

section Eval

variable (H : Msg → Msg)

theorem ocBind_eval {α β : Type} (f : α → OC β) : ∀ o : OC α,
    (ocBind o f).eval H = (((f (o.eval H).1).eval H).1, (o.eval H).2 ++ ((f (o.eval H).1).eval H).2)
  | .ret a => by simp [ocBind, OC.eval]
  | .query q k => by
      have := ocBind_eval f (k (H q))
      simp only [ocBind, OC.eval] at this ⊢
      rw [this]
      simp

variable (ask : List Msg → Out)

theorem spBind_eval {α β : Type} (f : α → SP β) : ∀ p : SP α,
    SP.eval H ask (spBind p f) =
      ((SP.eval H ask (f (SP.eval H ask p).1)).1, (SP.eval H ask p).2 + (SP.eval H ask (f (SP.eval H ask p).1)).2)
  | .ret a => by simp [spBind, SP.eval]
  | .hash q k => by
      have := spBind_eval f (k (H q))
      simp only [spBind, SP.eval] at this ⊢
      rw [this]
      simp only [Prod.mk.injEq, true_and]
      omega
  | .ask as k => by
      have := spBind_eval f (k (ask as))
      simp only [spBind, SP.eval] at this ⊢
      rw [this]
      simp only [Prod.mk.injEq, true_and]
      omega

theorem spBind_log {α β : Type} (f : α → SP β) : ∀ p : SP α,
    spLog H ask (spBind p f) = spLog H ask p ++ spLog H ask (f (SP.eval H ask p).1)
  | .ret a => by simp [spBind, spLog, SP.eval]
  | .hash q k => by
      have := spBind_log f (k (H q))
      simp only [spBind, spLog, SP.eval] at this ⊢
      rw [this]
      simp
  | .ask as k => by
      have := spBind_log f (k (ask as))
      simp only [spBind, spLog, SP.eval] at this ⊢
      exact this

theorem ocSP_eval {α : Type} : ∀ o : OC α, SP.eval H ask (ocSP o) = ((o.eval H).1, (o.eval H).2.length)
  | .ret a => by simp [ocSP, SP.eval, OC.eval]
  | .query q k => by
      have := ocSP_eval (k (H q))
      simp only [ocSP, SP.eval, OC.eval] at this ⊢
      rw [this]
      simp

theorem ocSP_log {α : Type} : ∀ o : OC α, spLog H ask (ocSP o) = (o.eval H).2
  | .ret a => by simp [ocSP, spLog, OC.eval]
  | .query q k => by
      have := ocSP_log (k (H q))
      simp only [ocSP, spLog, OC.eval] at this ⊢
      rw [this]

end Eval

/-! ## Transcripts -/

/-- The transcript of `n` rounds of the prover's next message `pm` against the verifier's answers `ask`, from the
history `hist`. -/
def interact (ask : List Msg → Out) (pm : List (Msg × Msg) → Msg) : ℕ → List (Msg × Msg) → List (Msg × Out)
  | 0, _ => []
  | n + 1, hist =>
    match ask (hist.map Prod.fst ++ [pm hist]) with
    | .msg m => (pm hist, .msg m) :: interact ask pm n (hist ++ [(pm hist, m)])
    | .acc => [(pm hist, .acc)]
    | .rej => [(pm hist, .rej)]

/-- A verifier answer's message, `[]` for a decision. -/
def msgOf : Out → Msg
  | .msg m => m
  | _ => []

/-- The prover's history after the transcript `τ`. -/
def histOf (τ : List (Msg × Out)) : List (Msg × Msg) := τ.map fun p => (p.1, msgOf p.2)

theorem stepWith_msgs {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (x : Msg) (ω : Ω) (vnext : List Msg → OC Out) :
    ∀ (n : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t : Trace), as = hist.map Prod.fst →
      (stepWith H P x ω vnext n hist as t).msgs =
        t.msgs ++ interact (fun as => ((vnext as).eval H).1) (fun h => ((P.next x h ω).eval H).1) n hist
  | 0, _, _, _, _ => by simp [stepWith, interact]
  | n + 1, hist, as, t, has => by
      subst has
      simp only [stepWith, interact]
      cases hb : ((vnext (List.map Prod.fst hist ++ [((P.next x hist ω).eval H).1])).eval H).1 with
      | msg m =>
          simp only
          rw [stepWith_msgs H P x ω vnext n _ _ _ (by simp)]
          simp
      | acc => simp
      | rej => simp

/-- **`runStar`'s messages are `interact`'s**, against `V*`'s answers. -/
theorem runStar_msgs {Ω Θ : Type} (H : Msg → Msg) (P : Prover Ω) (Vs : VStar Θ) (x : Msg) (ω : Ω) (θ : Θ) :
    (runStar H P Vs x ω θ).msgs =
      interact (fun as => ((Vs.next x as θ).eval H).1) (fun h => ((P.next x h ω).eval H).1) (Vs.k x) [] := by
  unfold runStar
  rw [stepWith_msgs H P x ω _ (Vs.k x) [] [] _ rfl]
  rfl

/-- **The simulator's runs are `interact`'s.** -/
theorem spRun_eval (H : Msg → Msg) (ask : List Msg → Out) (nxt : List (Msg × Msg) → OC Msg) :
    ∀ (n : ℕ) (hist : List (Msg × Msg)),
      (SP.eval H ask (spRun nxt n hist)).1 = interact ask (fun h => ((nxt h).eval H).1) n hist
  | 0, _ => by simp [spRun, SP.eval, interact]
  | n + 1, hist => by
      simp only [spRun, spBind_eval, ocSP_eval, SP.eval, interact]
      cases ask (hist.map Prod.fst ++ [((nxt hist).eval H).1]) with
      | msg m =>
          simp only [spBind_eval, SP.eval]
          rw [spRun_eval H ask nxt n]
      | acc => simp [SP.eval]
      | rej => simp [SP.eval]

theorem interact_length_le (ask : List Msg → Out) (pm : List (Msg × Msg) → Msg) :
    ∀ (n : ℕ) (hist : List (Msg × Msg)), (interact ask pm n hist).length ≤ n
  | 0, _ => by simp [interact]
  | n + 1, hist => by
      simp only [interact]
      cases ask (hist.map Prod.fst ++ [pm hist]) with
      | msg m => simpa using interact_length_le ask pm n _
      | acc => simp
      | rej => simp

/-- The first round: the prover's message on `hist`, and the verifier's answer to it. -/
theorem interact_head (ask : List Msg → Out) (pm : List (Msg × Msg) → Msg) (n : ℕ) (hist : List (Msg × Msg)) :
    (interact ask pm (n + 1) hist)[0]? = some (pm hist, ask (hist.map Prod.fst ++ [pm hist])) := by
  simp only [interact]
  cases ask (hist.map Prod.fst ++ [pm hist]) <;> simp

/-- **Two provers that agree on a run's histories up to round `j` give its transcript up to round `j`.** -/
theorem interact_take_congr (ask : List Msg → Out) (pm₁ pm₂ : List (Msg × Msg) → Msg) :
    ∀ (n : ℕ) (hist : List (Msg × Msg)) (j : ℕ),
      (∀ i ≤ j, i < (interact ask pm₁ n hist).length →
        pm₂ (hist ++ histOf ((interact ask pm₁ n hist).take i)) =
          pm₁ (hist ++ histOf ((interact ask pm₁ n hist).take i))) →
      (interact ask pm₂ n hist).take (j + 1) = (interact ask pm₁ n hist).take (j + 1)
  | 0, _, _, _ => by simp [interact]
  | n + 1, hist, j, h => by
      have h0 := h 0 (Nat.zero_le _) (by
        simp only [interact]; cases ask (hist.map Prod.fst ++ [pm₁ hist]) <;> simp)
      simp only [List.take_zero, histOf, List.map_nil, List.append_nil] at h0
      simp only [interact] at h ⊢
      rw [h0]
      cases hb : ask (hist.map Prod.fst ++ [pm₁ hist]) with
      | msg m =>
          rw [hb] at h
          cases j with
          | zero => simp
          | succ j =>
              simp only [List.take_succ_cons]
              congr 1
              refine interact_take_congr ask pm₁ pm₂ n _ j fun i hi hlt => ?_
              have := h (i + 1) (by omega) (by simpa using hlt)
              simpa [histOf, msgOf] using this
      | acc => simp
      | rej => simp

/-- **Two provers that agree on a run's histories give its transcript.** -/
theorem interact_congr (ask : List Msg → Out) (pm₁ pm₂ : List (Msg × Msg) → Msg) (n : ℕ) (hist : List (Msg × Msg))
    (h : ∀ i < (interact ask pm₁ n hist).length,
      pm₂ (hist ++ histOf ((interact ask pm₁ n hist).take i)) =
        pm₁ (hist ++ histOf ((interact ask pm₁ n hist).take i))) :
    interact ask pm₂ n hist = interact ask pm₁ n hist := by
  have ht := interact_take_congr ask pm₁ pm₂ n hist (interact ask pm₁ n hist).length fun i _ hlt => h i hlt
  have e1 : (interact ask pm₁ n hist).take ((interact ask pm₁ n hist).length + 1) = interact ask pm₁ n hist :=
    List.take_of_length_le (by omega)
  rw [e1] at ht
  have hl := congrArg List.length ht
  rw [List.length_take] at hl
  have hle : (interact ask pm₂ n hist).length ≤ (interact ask pm₁ n hist).length + 1 := by omega
  rwa [List.take_of_length_le hle] at ht

end Claims.ZKMal
