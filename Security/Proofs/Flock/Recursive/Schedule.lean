import Proofs.Flock.Recursive.Defs
import Proofs.Flock.Soundness.ExecDraw
import Proofs.Flock.Soundness.Refine.LiveSim

/-!
# Recursion: an inner protocol on a fixed schedule

`Inner` has `r` uniform rounds, each a message and then a live coin. A verifier written as a `Game`, whose shape may
depend on the prover's messages (it may stop early, or let the prover choose which of two interleaved runs moves next),
becomes an inner protocol once a fixed schedule says what each round carries: round `k < r` sends a list of messages,
each of one of the types `T i`, and then draws a coin of type `CT k`; round `r` sends the last messages.

* `schedInner`: that protocol, with `r + 1` rounds. Each round's coin is the whole product `∀ k, CT k`, of which round
  `k < r` reads coordinate `k` and round `r` reads none (a game that ends with a message, as C-Flock's table ends with
  the openings, needs a round after its last coin); the verifier plays the game with round `0`'s messages, round `0`'s
  coin, round `1`'s messages, and so on (`Game.PlaysT`), and accepts when the game ends in an accepted outcome. A round whose messages have the wrong types, or
  too many, or whose coin's type is not the one the game draws there, makes the run fail.
* `sched_sound`: the schedule's verifier is as sound as the game. Against every prover it accepts with probability at
  most the game's `value`, the best any prover achieves when the game draws its own coins.
-/

namespace FlockSoundness.Discharge.Recursion

open Game Audit.DrawExec
open scoped ENNReal

/-- Averaging a function of one coordinate over a product of uniform coordinates averages it over that coordinate. -/
theorem avg_coord {ι : Type} [Fintype ι] [DecidableEq ι] {Ω : ι → Type} [∀ i, Fintype (Ω i)] [∀ i, Nonempty (Ω i)]
    (i : ι) (G : Ω i → ℝ≥0∞) : avg (fun ω : ∀ j, Ω j => G (ω i)) = avg G := by
  calc avg (fun ω : ∀ j, Ω j => G (ω i))
      = avg (fun c => (fun ω : ∀ j, Ω j => G (ω i)) ((Equiv.piSplitAt i Ω).symm c)) := (avg_equiv _ _).symm
    _ = avg (fun c : Ω i × (∀ j : {j // j ≠ i}, Ω j) => G c.1) := by
        congr 1
        funext c
        simp [Equiv.piSplitAt_symm_apply]
    _ = avg fun x => avg fun _ : (∀ j : {j // j ≠ i}, Ω j) => G x := avg_prod _
    _ = avg G := by simp only [avg_const]

section Schedule

variable {nT r : ℕ} (T : Fin nT → Type) (CT : Fin r → Type) [fCT : ∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]

/-- A message of the schedule: one of the types `T i`, and a value of it. -/
abbrev SMsg := Σ i : Fin nT, T i

/-- The typed steps of the rounds from round `j` on: each round's messages in order, then its coin, coordinate `j` of
the round's coin. -/
def schedSteps : ℕ → List (List (SMsg T) × ((k : Fin r) → CT k)) → List Step
  | _, [] => []
  | j, (ms, ρ) :: t => ms.map (fun e => Sum.inl ⟨T e.1, e.2⟩) ++
      ((if h : j < r then [Sum.inr ⟨CT ⟨j, h⟩, ρ ⟨j, h⟩⟩] else []) ++ schedSteps (j + 1) t)

/-- **The inner protocol on the schedule `(T, CT)`**: statements `Stmt`, the verifier `g x` on the statement `x`, its
accepted outcomes `p x`, and the statement's truth `holds`. -/
def schedInner {α : Type} (Stmt : Type) (g : Stmt → Game α) (p : Stmt → α → Prop) (holds : Stmt → Prop) : Inner where
  Stmt := Stmt
  Msg := List (SMsg T)
  Coin := (k : Fin r) → CT k
  fintype := inferInstance
  nonempty := inferInstance
  r := r + 1
  accepts x t := ∃ a, PlaysT (g x) (schedSteps T CT 0 t) a ∧ p x a
  holds := holds

variable {α β : Type}

open Classical in
/-- One round, by induction on the game: averaged over the round's coin, the run that takes the round's messages `ms`,
then its coin, then the later rounds, ends accepted no more often than the game's value, given that bound for the later
rounds (`ih`). -/
theorem avg_round_le (p : α → Prop) {n j : ℕ} (hj : j < r)
    (ih : ∀ (f : List (List (SMsg T) × ((k : Fin r) → CT k)) → Game β)
      (s : Strategy (rounds (List (SMsg T)) ((k : Fin r) → CT k) n f)) (g : Game α),
      leafAvg n f (fun t _ => if ∃ a, PlaysT g (schedSteps T CT (j + 1) t) a ∧ p a then 1 else 0) s ≤ value p g)
    (f : ((k : Fin r) → CT k) → List (List (SMsg T) × ((k : Fin r) → CT k)) → Game β)
    (s : (ρ : (k : Fin r) → CT k) → Strategy (rounds (List (SMsg T)) ((k : Fin r) → CT k) n (f ρ))) :
    ∀ (g : Game α) (ms : List (SMsg T)),
      avg (fun ρ => leafAvg n (f ρ) (fun t _ => if ∃ a, PlaysT g (ms.map (fun e => Sum.inl ⟨T e.1, e.2⟩) ++
        (Sum.inr ⟨CT ⟨j, hj⟩, ρ ⟨j, hj⟩⟩ :: schedSteps T CT (j + 1) t)) a ∧ p a then 1 else 0) (s ρ)) ≤ value p g := by
  have hzero : ∀ (g : Game α) (P : ((k : Fin r) → CT k) → List (List (SMsg T) × ((k : Fin r) → CT k)) → Prop),
      (∀ ρ t, ¬ P ρ t) →
      avg (fun ρ => leafAvg n (f ρ) (fun t _ => if P ρ t then 1 else 0) (s ρ)) ≤ value p g := by
    intro g P hP
    calc avg (fun ρ => leafAvg n (f ρ) (fun t _ => if P ρ t then 1 else 0) (s ρ))
        ≤ avg (fun _ => (0 : ℝ≥0∞)) := avg_mono fun ρ =>
          (leafAvg_mono (G := fun _ _ => 0) (fun t _ => by simp [hP ρ t]) (s ρ)).trans (leafAvg_const 0 _).le
      _ = 0 := avg_const 0
      _ ≤ value p g := zero_le
  intro g
  induction g with
  | ret a =>
      intro ms
      refine hzero _ _ fun ρ t ⟨a', h, _⟩ => ?_
      have := h.ret_inv.1
      simp at this
  | send M k ihk =>
      intro ms
      cases ms with
      | nil =>
          refine hzero _ _ fun ρ t ⟨a', h, _⟩ => ?_
          obtain ⟨m, hm, _⟩ := h.send_inv
          simp at hm
      | cons e ms' =>
          by_cases hT : T e.1 = M
          · subst hT
            calc _ ≤ avg (fun ρ => leafAvg n (f ρ) (fun t _ => if ∃ a, PlaysT (k e.2)
                  (ms'.map (fun e => Sum.inl ⟨T e.1, e.2⟩) ++
                    (Sum.inr ⟨CT ⟨j, hj⟩, ρ ⟨j, hj⟩⟩ :: schedSteps T CT (j + 1) t)) a ∧ p a then 1 else 0) (s ρ)) := by
                  refine avg_mono fun ρ => leafAvg_mono (fun t _ => ?_) (s ρ)
                  split_ifs with h1 h2 h2
                  · exact le_rfl
                  · exfalso
                    obtain ⟨a, h, hp⟩ := h1
                    obtain ⟨m, hm, h⟩ := h.send_inv
                    have hm' := (Sigma.mk.inj_iff.1 (Sum.inl.inj hm)).2
                    exact h2 ⟨a, (eq_of_heq hm').symm ▸ h, hp⟩
                  · exact zero_le
                  · exact le_rfl
              _ ≤ value p (k e.2) := ihk e.2 ms'
              _ ≤ value p (.send (T e.1) k) := by
                  rw [value_send]
                  exact le_iSup (fun m => value p (k m)) e.2
          · refine hzero _ _ fun ρ t ⟨a', h, _⟩ => ?_
            obtain ⟨m, hm, _⟩ := h.send_inv
            exact hT (Sigma.mk.inj_iff.1 (Sum.inl.inj hm)).1
  | @coin C iF iN k _ =>
      intro ms
      cases ms with
      | cons e ms' =>
          refine hzero _ _ fun ρ t ⟨a', h, _⟩ => ?_
          obtain ⟨x, hx, _⟩ := h.coin_inv
          simp at hx
      | nil =>
          by_cases hC : CT ⟨j, hj⟩ = C
          · subst hC
            obtain rfl : iF = fCT ⟨j, hj⟩ := Subsingleton.elim _ _
            calc _ ≤ avg (fun ρ => leafAvg n (f ρ) (fun t _ => if ∃ a, PlaysT (k (ρ ⟨j, hj⟩))
                  (schedSteps T CT (j + 1) t) a ∧ p a then 1 else 0) (s ρ)) := by
                  refine avg_mono fun ρ => leafAvg_mono (fun t _ => ?_) (s ρ)
                  split_ifs with h1 h2 h2
                  · exact le_rfl
                  · exfalso
                    obtain ⟨a, h, hp⟩ := h1
                    obtain ⟨x, hx, h⟩ := h.coin_inv
                    have hx' := (Sigma.mk.inj_iff.1 (Sum.inr.inj hx)).2
                    exact h2 ⟨a, (eq_of_heq hx') ▸ h, hp⟩
                  · exact zero_le
                  · exact le_rfl
              _ ≤ avg (fun ρ : (k : Fin r) → CT k => value p (k (ρ ⟨j, hj⟩))) :=
                  avg_mono fun ρ => ih (f ρ) (s ρ) (k (ρ ⟨j, hj⟩))
              _ = avg (fun x => value p (k x)) := avg_coord (Ω := CT) ⟨j, hj⟩ (fun x => value p (k x))
              _ = value p (.coin (CT ⟨j, hj⟩) k) := by rw [value_coin]; rfl
          · refine hzero _ _ fun ρ t ⟨a', h, _⟩ => ?_
            obtain ⟨x, hx, _⟩ := h.coin_inv
            exact hC (Sigma.mk.inj_iff.1 (Sum.inr.inj hx)).1

open Classical in
/-- A run on messages alone ends accepted no more often than the game's value. -/
theorem plays_msgs_le (p : α → Prop) : ∀ (g : Game α) (ms : List (SMsg T)),
    (if ∃ a, PlaysT g (ms.map (fun e => Sum.inl ⟨T e.1, e.2⟩)) a ∧ p a then (1 : ℝ≥0∞) else 0) ≤ value p g
  | g, [] => by
      split_ifs with h
      · obtain ⟨a, ha, hp⟩ := h
        rw [ha.nil_inv]
        simp [value, hp]
      · exact zero_le
  | g, e :: ms => by
      split_ifs with h
      · obtain ⟨a, ha, hp⟩ := h
        cases g with
        | ret b =>
            have := ha.ret_inv.1
            simp at this
        | coin C k =>
            obtain ⟨x, hx, _⟩ := ha.coin_inv
            simp at hx
        | send M k =>
            obtain ⟨m, hm, hk⟩ := ha.send_inv
            have hT : T e.1 = M := (Sigma.mk.inj_iff.1 (Sum.inl.inj hm)).1
            subst hT
            have hm' := (Sigma.mk.inj_iff.1 (Sum.inl.inj hm)).2
            have ih := plays_msgs_le p (k e.2) ms
            rw [if_pos ⟨a, (eq_of_heq hm').symm ▸ hk, hp⟩] at ih
            refine ih.trans ?_
            rw [value_send]
            exact le_iSup (fun m => value p (k m)) e.2
      · exact zero_le

open Classical in
/-- **The schedule's rounds from round `j` on**, `n` of them with `j + n = r + 1`: the run on their steps ends accepted
no more often than the game's value. -/
theorem leafAvg_plays_le (p : α → Prop) : ∀ (n j : ℕ), j + n = r + 1 →
    ∀ (f : List (List (SMsg T) × ((k : Fin r) → CT k)) → Game β)
      (s : Strategy (rounds (List (SMsg T)) ((k : Fin r) → CT k) n f)) (g : Game α),
      leafAvg n f (fun t _ => if ∃ a, PlaysT g (schedSteps T CT j t) a ∧ p a then 1 else 0) s ≤ value p g
  | 0, j, _, f, s, g => by
      show (if ∃ a, PlaysT g [] a ∧ p a then (1 : ℝ≥0∞) else 0) ≤ value p g
      split_ifs with h
      · obtain ⟨a, ha, hp⟩ := h
        rw [ha.nil_inv]
        simp [value, hp]
      · exact zero_le
  | n + 1, j, hjn, f, s, g => by
      by_cases hj : j < r
      swap
      · obtain rfl : n = 0 := by omega
        show avg (fun ρ => if ∃ a, PlaysT g (schedSteps T CT j [(s.1, ρ)]) a ∧ p a then (1 : ℝ≥0∞) else 0) ≤ value p g
        simp only [schedSteps, dif_neg hj, List.nil_append, List.append_nil, avg_const]
        exact plays_msgs_le T p g s.1
      have := avg_round_le T CT p hj (leafAvg_plays_le p n (j + 1) (by omega)) (fun ρ t => f ((s.1, ρ) :: t))
        (fun ρ => s.2 ρ) g s.1
      refine le_of_eq_of_le ?_ this
      show avg _ = avg _
      congr 1
      funext ρ
      congr 1
      funext t _
      simp only [schedSteps, hj, ↓reduceDIte, List.singleton_append]

/-- **The schedule is as sound as the game.** If on every false statement the game's value is at most `ε`, the inner
protocol on the schedule is `ε`-sound. -/
theorem sched_sound {Stmt : Type} (g : Stmt → Game α) (p : Stmt → α → Prop) (holds : Stmt → Prop) {ε : ℝ≥0∞}
    (h : ∀ x, ¬ holds x → value (p x) (g x) ≤ ε) : InnerSound (schedInner T CT Stmt g p holds) ε := by
  intro x hx s
  refine le_trans (le_of_eq ?_) ((leafAvg_plays_le T CT (p x) (r + 1) 0 (by omega) .ret s (g x)).trans (h x hx))
  exact prob_rounds _ (r + 1) _ s

end Schedule

end FlockSoundness.Discharge.Recursion
