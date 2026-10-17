import Proofs.Flock.Claims.Prob

/-!
# The collision finder `B`

Every `H` call a run logs is a true evaluation of `H`, so `B` outputs only true collisions, and each of its two inputs
is one the verifier hashed in one of `B`'s two runs: the break is tied to the runs. `B`'s cost is at most twice a run's.
-/

namespace Claims

/-- Every pair an oracle program logs is an evaluation of `H`. -/
theorem OC.eval_log {α : Type} (H : Msg → Msg) (c : OC α) : ∀ p ∈ (c.eval H).2, H p.1 = p.2 := by
  induction c with
  | ret a => simp [OC.eval]
  | query q k ih =>
    intro p hp
    simp only [OC.eval, List.mem_cons] at hp
    rcases hp with rfl | hp
    · rfl
    · exact ih _ p hp

/-- The rounds keep both logs evaluations of `H`. -/
theorem stepWith_logs {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (x : Msg) (ω : Ω) (vnext : List Msg → OC Out) :
    ∀ (n : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t : Trace),
      (∀ p ∈ t.pLog, H p.1 = p.2) → (∀ p ∈ t.vLog, H p.1 = p.2) →
        (∀ p ∈ (stepWith H P x ω vnext n hist as t).pLog, H p.1 = p.2) ∧
          (∀ p ∈ (stepWith H P x ω vnext n hist as t).vLog, H p.1 = p.2) := by
  intro n
  induction n with
  | zero => intro hist as t hp hv; exact ⟨hp, hv⟩
  | succ n ih =>
    intro hist as t hp hv
    have hp' : ∀ p ∈ t.pLog ++ ((P.next x hist ω).eval H).2, H p.1 = p.2 := by
      intro p h
      rcases List.mem_append.1 h with h | h
      · exact hp p h
      · exact OC.eval_log H _ p h
    have hv' : ∀ p ∈ t.vLog ++ ((vnext (as ++ [((P.next x hist ω).eval H).1])).eval H).2, H p.1 = p.2 := by
      intro p h
      rcases List.mem_append.1 h with h | h
      · exact hv p h
      · exact OC.eval_log H _ p h
    simp only [stepWith]
    split
    · exact ih _ _ _ hp' hv'
    · exact ⟨hp', hv'⟩
    · exact ⟨hp', hv'⟩

/-- Every pair a run logs, on either side, is an evaluation of `H`. -/
theorem run_logs {Ω Coin : Type} [Inhabited Coin] (H : Msg → Msg) (P : Prover Ω) (V : Verifier Coin) (x : Msg)
    (ω : Ω) (r : Fin (V.k x) → Coin) :
    (∀ p ∈ (run H P V x ω r).pLog, H p.1 = p.2) ∧ (∀ p ∈ (run H P V x ω r).vLog, H p.1 = p.2) :=
  stepWith_logs H P x ω _ _ [] [] _ (by simp) (by simp)

/-- `findColl` outputs only true collisions of a list of evaluations of `H`. -/
theorem findColl_spec (H : Msg → Msg) :
    ∀ (l : List (Msg × Msg)), (∀ p ∈ l, H p.1 = p.2) → ∀ u v, findColl l = some (u, v) → Collision H u v := by
  intro l
  induction l with
  | nil => intro _ u v h; simp [findColl] at h
  | cons hd tl ih =>
    obtain ⟨q, a⟩ := hd
    intro hl u v h
    simp only [findColl] at h
    split at h
    · rename_i p hp
      obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Option.some.inj h)
      have hmem := List.mem_of_find?_eq_some hp
      have hpr := List.find?_some hp
      simp only [Bool.and_eq_true, decide_eq_true_eq] at hpr
      refine ⟨fun e => hpr.1 e.symm, ?_⟩
      rw [hl (q, a) (by simp), hl p (by simp [hmem]), hpr.2]
    · exact ih (fun p hp => hl p (by simp [hp])) u v h

/-- Both inputs of `findColl`'s collision are inputs of the list. -/
theorem findColl_mem :
    ∀ (l : List (Msg × Msg)) u v, findColl l = some (u, v) → u ∈ l.map Prod.fst ∧ v ∈ l.map Prod.fst := by
  intro l
  induction l with
  | nil => intro u v h; simp [findColl] at h
  | cons hd tl ih =>
    obtain ⟨q, a⟩ := hd
    intro u v h
    simp only [findColl] at h
    split at h
    · rename_i p hp
      obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Option.some.inj h)
      exact ⟨by simp, List.mem_cons_of_mem _ (List.mem_map.2 ⟨p, List.mem_of_find?_eq_some hp, rfl⟩)⟩
    · obtain ⟨h1, h2⟩ := ih u v h
      exact ⟨List.mem_cons_of_mem _ h1, List.mem_cons_of_mem _ h2⟩

section B

variable {Ω Coin : Type} [Fintype Ω] [Nonempty Ω] [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- **`B`'s output is a collision tied to its runs**: two different inputs with one `H` value, each an input the
verifier hashed in one of `B`'s two runs. -/
theorem B_out_spec (H : Msg → Msg) (V : Verifier Coin) (P : Prover Ω) (x : Msg) (hk : 0 < V.k x)
    (s : (B H V P x hk).S) (u v : Msg) (h : (B H V P x hk).out s = some (u, v)) :
    Collision H u v ∧
      u ∈ ((run H P V x s.2.1 s.2.2.1).vLog ++ (run H P V x s.2.1 (forkCoins s.1 s.2.2.1 s.2.2.2)).vLog).map
        Prod.fst ∧
      v ∈ ((run H P V x s.2.1 s.2.2.1).vLog ++ (run H P V x s.2.1 (forkCoins s.1 s.2.2.1 s.2.2.2)).vLog).map
        Prod.fst := by
  refine ⟨findColl_spec H _ ?_ u v h, findColl_mem _ u v h⟩
  intro p hp
  rcases List.mem_append.1 hp with hp | hp
  · exact (run_logs H P V x _ _).2 p hp
  · exact (run_logs H P V x _ _).2 p hp

/-- `Pr[B outputs a collision]` is `Pr[B outputs anything]`. -/
theorem B_Pr_eq (H : Msg → Msg) (V : Verifier Coin) (P : Prover Ω) (x : Msg) (hk : 0 < V.k x) :
    (B H V P x hk).Pr H = Claims.Pr fun s : (B H V P x hk).S => ((B H V P x hk).out s).isSome := by
  unfold Finder.Pr
  refine Pr_congr fun s => ⟨fun ⟨u, v, h, _⟩ => by simp [h], fun h => ?_⟩
  obtain ⟨⟨u, v⟩, huv⟩ := Option.isSome_iff_exists.1 h
  exact ⟨u, v, huv, (B_out_spec H V P x hk s u v huv).1⟩

/-- **`B`'s cost is at most two runs'**: if every run makes at most `t` `H` calls (both parties'), `B` makes at most
`2t`. -/
theorem B_cost_le (H : Msg → Msg) (V : Verifier Coin) (P : Prover Ω) (x : Msg) (hk : 0 < V.k x) (t : ℕ)
    (ht : ∀ ω r, (run H P V x ω r).pLog.length + (run H P V x ω r).vLog.length ≤ t) (s : (B H V P x hk).S) :
    (B H V P x hk).cost s ≤ 2 * t := by
  have h1 := ht s.2.1 s.2.2.1
  have h2 := ht s.2.1 (forkCoins s.1 s.2.2.1 s.2.2.2)
  simp only [B]
  omega

end B

end Claims
