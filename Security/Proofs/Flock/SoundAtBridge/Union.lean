import Proofs.Flock.Soundness.Game.Prob
import Definitions.Accounting.TS.ProofLayer

/-!
# The bridge's probability steps, over any family of sessions

The proof layer's soundness (`Accounting.TS.ProofLayer.SoundAt`) is a probability over a window's paths (uniform on
`Paths Ω S`, `Pouw.RelativeSecurity.prob`), per round `w` and proof `i`; C-Flock's bounds are probabilities over one
session's audit game played by its prover (`FlockSoundness.Game.prob`). This file joins the two for any family of
sessions, with no C-Flock definition:

* `game_prob_exists_le`: the union bound for a fixed strategy over a finite family of events;
* `SessionLaw`: read through the session a path runs and its outcome there, the window's coins have the sessions' law;
* `soundAt_of_sessions`: `SoundAt` at `δo + δ` when each unsound proof is off (probability at most `δo`) or its
  session's outcome is in a bad event of probability at most `δ` in every session it runs.
-/

namespace Flock.SoundAtBridge

universe u

open FlockSoundness Accounting.TS
open Pouw.RelativeSecurity (Paths)
open scoped ENNReal

/-- **The union bound for a fixed strategy, over a finite family of events.** -/
theorem game_prob_exists_le {α : Type} {ι : Type*} (s : Finset ι) (E : ι → α → Prop) (g : Game α)
    (σ : Game.Strategy g) : Game.prob (fun a => ∃ i ∈ s, E i a) g σ ≤ ∑ i ∈ s, Game.prob (E i) g σ := by
  classical
  induction s using Finset.induction_on with
  | empty =>
    have h0 : (fun a : α => ∃ i ∈ (∅ : Finset ι), E i a) = fun _ => False :=
      funext fun a => propext ⟨fun ⟨_, hi, _⟩ => absurd hi (Finset.notMem_empty _), False.elim⟩
    rw [h0, Game.prob_false]
    exact zero_le
  | insert a s ha ih =>
    rw [Finset.sum_insert ha]
    refine (Game.prob_mono (fun o ho => ?_) g σ).trans
      ((Game.prob_or_le (E a) _ g σ).trans (add_le_add le_rfl ih))
    obtain ⟨i, hi, hE⟩ := ho
    rcases Finset.mem_insert.mp hi with rfl | hi
    · exact Or.inl hE
    · exact Or.inr ⟨i, hi, hE⟩

theorem wprob_mono {β : Type*} [Fintype β] {p q : β → Prop} (h : ∀ b, p b → q b) :
    Pouw.RelativeSecurity.prob p ≤ Pouw.RelativeSecurity.prob q := by
  classical
  unfold Pouw.RelativeSecurity.prob
  apply div_le_div_of_nonneg_right _ (by positivity)
  exact_mod_cast Finset.card_le_card fun b hb => by
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hb ⊢
    exact h b hb

theorem wprob_or_le {β : Type*} [Fintype β] (p q : β → Prop) :
    Pouw.RelativeSecurity.prob (fun b => p b ∨ q b) ≤
      Pouw.RelativeSecurity.prob p + Pouw.RelativeSecurity.prob q := by
  classical
  unfold Pouw.RelativeSecurity.prob
  rw [← add_div]
  apply div_le_div_of_nonneg_right _ (by positivity)
  have key : ∀ s t u : Finset β, (∀ b ∈ s, b ∈ t ∨ b ∈ u) → (s.card : ℝ) ≤ t.card + u.card := fun s t u h => by
    exact_mod_cast (Finset.card_le_card fun b hb => Finset.mem_union.mpr (h b hb)).trans (Finset.card_union_le t u)
  exact key _ _ _ fun b hb => by
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hb ⊢
    exact hb

theorem avg_le {β : Type*} [Fintype β] (f : β → ℝ) {δ : ℝ} (hδ : 0 ≤ δ) (h : ∀ b, f b ≤ δ) :
    (∑ b, f b) / (Fintype.card β : ℝ) ≤ δ := by
  rcases Nat.eq_zero_or_pos (Fintype.card β) with h0 | h0
  · simp [h0, hδ]
  · rw [div_le_iff₀ (by exact_mod_cast h0)]
    calc ∑ b, f b ≤ ∑ _b : β, δ := Finset.sum_le_sum fun b _ => h b
      _ = δ * Fintype.card β := by rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul, mul_comm]

/-- **Read through its sessions, the window has the sessions' law**: for every event `E` of a session's outcome, its
probability over the window's paths, at the session `sess x` the path `x` runs and its outcome `out x` there, is at
most the paths' average of its probability in that session's game `G (sess x)` played by its prover `τ (sess x)`. It
holds with equality when, given what the path fixes before the session (its setting and its prover), the session's
own coins are the path's next coins, fresh and uniform. -/
def SessionLaw {Ω : Type} [Fintype Ω] {S : ℕ} {Sess : Type u} {Out : Sess → Type}
    (G : ∀ s, Game (Out s)) (τ : ∀ s, Game.Strategy (G s)) (sess : Paths Ω S → Sess) (out : ∀ x, Out (sess x)) :
    Prop :=
  ∀ E : ∀ s, Out s → Prop,
    Pouw.RelativeSecurity.prob (fun x => E (sess x) (out x)) ≤
      (∑ x, (Game.prob (E (sess x)) (G (sess x)) (τ (sess x))).toReal) / (Fintype.card (Paths Ω S) : ℝ)

/-- **`SoundAt` from a family of sessions**: when proof `i` of round `w` runs the session `sess x w i` with outcome
`out x w i`, the window's law at each `(w, i)` is the sessions' (`SessionLaw`), every unsound proof is off (`Off`, of
probability at most `δo` at each `(w, i)`) or has its session's outcome in `Bad`, and `Bad` has probability at most
`δ ≥ 0` in every session a path runs, then each proof is unsound with probability at most `δo + δ`. -/
theorem soundAt_of_sessions {Ω : Type} [Fintype Ω] {S nπ : ℕ} {V : Type} (pf : ProofLayer Ω S nπ V)
    {Sess : Type u} {Out : Sess → Type} (G : ∀ s, Game (Out s)) (τ : ∀ s, Game.Strategy (G s))
    (Bad : ∀ s, Out s → Prop) (sess : Paths Ω S → Fin S → Fin nπ → Sess) (out : ∀ x w i, Out (sess x w i))
    (Off : Paths Ω S → Fin S → Fin nπ → Prop) (δo δ : ℝ)
    (hLaw : ∀ w i, SessionLaw G τ (fun x => sess x w i) (fun x => out x w i))
    (hOff : ∀ w i, Pouw.RelativeSecurity.prob (fun x => Off x w i) ≤ δo)
    (hB : ∀ x w i, Game.prob (Bad (sess x w i)) (G (sess x w i)) (τ (sess x w i)) ≤ ENNReal.ofReal δ)
    (hδ : 0 ≤ δ) (hIn : ∀ x w i, pf.Unsound x w i → Off x w i ∨ Bad (sess x w i) (out x w i)) :
    pf.SoundAt (δo + δ) := by
  intro w i
  refine (wprob_mono fun x hx => hIn x w i hx).trans ((wprob_or_le _ _).trans (add_le_add (hOff w i) ?_))
  exact (hLaw w i Bad).trans
    (avg_le _ hδ fun x => ENNReal.toReal_le_of_le_ofReal hδ (hB x w i))

end Flock.SoundAtBridge
