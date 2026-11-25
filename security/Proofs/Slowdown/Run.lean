module

public import Definitions.Slowdown.Guarantees.Run
public import Proofs.Slowdown.SecureWitness
public import Proofs.Core.Game.Union
public import Proofs.Core.Game.Prob

/-!
# An accepted run is not prohibited, and an honest developer's run is accepted

`RunSecure` proves `Guarantees.RunSecure` from `Secure`: with the probability of an event the run's value against an
optimal developer (`Game.value`), which is monotone and has the union bound, each audit's soundness at its own
acceptance is soundness at the verifier's, `accounted_of_windows` adds the errors over the windows, and `secure_pr`
takes `A` to `B`. `RunComplete` is the union bound for an honest strategy. Their witnesses are at the run with no
message and no coin, where soundness is the claim at the one outcome (`sound_still`, `not_sound_still`).
-/

@[expose] public section

namespace Slowdown.SecurityProofs

open Verity Finset Slowdown.Security Slowdown.Security.Witness SecureWitness

open scoped ENNReal

theorem RunSecure : Guarantees.RunSecure := by
  intro L E _ _ α g world F x_s x_b x_w y_b p Y acc_s acc_b acc_y acc_w δ_s δ_b δ_y δ_w hY hH hs hb hy hw
  have mono : ∀ P Q : α → Prop, (∀ r, P r → Q r) → Game.value P g ≤ Game.value Q g :=
    fun _ _ h => Game.value_mono h g
  have lift : ∀ {acc E : α → Prop} {δ : ℝ≥0∞}, (∀ a, acc_s a ∧ acc_b a ∧ acc_y a ∧ acc_w a → acc a) →
      Sound g acc E δ → Sound g (fun a => acc_s a ∧ acc_b a ∧ acc_y a ∧ acc_w a) E δ :=
    fun h hS => (Game.value_mono (fun a ha => ⟨h a ha.1, ha.2⟩) g).trans hS
  exact secure_pr (fun P => Game.value P g) mono _ world F x_s x_b x_w y_b p Y hY.1 hY.2.1 hY.2.2.1 hY.2.2.2 _ hH
    (accounted_of_windows (fun P => Game.value P g) mono _ world
      (fun s q => Game.value_exists_finset_le s q g) F x_s x_b x_w y_b δ_s δ_b δ_w δ_y
      (fun i j h => lift (fun _ ha => ha.1) (hs i j h))
      (fun i j h => lift (fun _ ha => ha.2.1) (hb i j h))
      (fun i j h => lift (fun _ ha => ha.2.2.1) (hy i j h))
      (fun i j h => lift (fun _ ha => ha.2.2.2) (hw i j h)))

/-- **An honest developer's run is accepted, outside the audits' errors**: where the verifier accepts whenever each
of the four audits does, a strategy `honest` that memory's audit rejects with probability at most `η_s`, bandwidth's
`η_b`, the count `η_y` and work's `η_w` is rejected with probability at most their sum. -/
theorem RunComplete {α : Type} (g : Game α) (honest : g.Strategy) (accept acc_s acc_b acc_y acc_w : α → Prop)
    (η_s η_b η_y η_w : ℝ≥0∞) (hacc : ∀ a, acc_s a → acc_b a → acc_y a → acc_w a → accept a)
    (hs : Game.prob (fun a => ¬ acc_s a) g honest ≤ η_s) (hb : Game.prob (fun a => ¬ acc_b a) g honest ≤ η_b)
    (hy : Game.prob (fun a => ¬ acc_y a) g honest ≤ η_y) (hw : Game.prob (fun a => ¬ acc_w a) g honest ≤ η_w) :
    Game.prob (fun a => ¬ accept a) g honest ≤ η_s + η_b + η_y + η_w := by
  have h1 : Game.prob (fun a => ¬ accept a) g honest ≤
      Game.prob (fun a => ((¬ acc_s a ∨ ¬ acc_b a) ∨ ¬ acc_y a) ∨ ¬ acc_w a) g honest :=
    Game.prob_mono (fun a ha => by
      by_contra hno
      obtain ⟨h123, h4⟩ := not_or.1 hno
      obtain ⟨h12, h3⟩ := not_or.1 h123
      obtain ⟨h1, h2⟩ := not_or.1 h12
      exact ha (hacc a (not_not.1 h1) (not_not.1 h2) (not_not.1 h3) (not_not.1 h4))) g honest
  exact h1.trans ((Game.prob_or_le _ _ g honest).trans (add_le_add ((Game.prob_or_le _ _ g honest).trans
    (add_le_add ((Game.prob_or_le _ _ g honest).trans (add_le_add hs hb)) hy)) hw))

/-- At the run with no message and no coin, a claim that holds at the one outcome is sound with any error. -/
theorem sound_still {acc E : Unit → Prop} (h : E ()) (δ : ℝ≥0∞) : Sound Guarantees.still acc E δ := by
  classical
  change Game.value _ (Game.ret ()) ≤ δ
  rw [Game.value_ret]
  simp [h]

/-- At the run with no message and no coin, a claim that fails at the one outcome, which the verifier accepts, is
sound with no error below 1. -/
theorem not_sound_still {acc E : Unit → Prop} (ha : acc ()) (h : ¬ E ()) {δ : ℝ≥0∞} (hδ : δ < 1) :
    ¬ Sound Guarantees.still acc E δ := by
  classical
  intro hS
  change Game.value _ (Game.ret ()) ≤ δ at hS
  rw [Game.value_ret] at hS
  simp only [ha, h, not_false_eq_true, and_self, ite_true] at hS
  exact absurd hS (not_le.2 hδ)

/-- The run with no gate the declared circuit's holds no bit of it. -/
theorem trained_not_held (I : Finset (Fin (1 * cuts.J))) (i : Fin (frame 1 1 1 I).n)
    (j : Fin (frame 1 1 1 I).cuts.J) : trained ∉ HeldAtLeast (frame 1 1 1 I) i j 1 := fun h => by
  have h1 : 1 ≤ run.width (actual.state ((frame 1 1 1 I).server i) ((frame 1 1 1 I).cuts.τ j.castSucc) ∩ ∅) := h
  simp [Circuit.width] at h1

theorem RunSecure_honest : Guarantees.RunSecure_honest :=
  ⟨actual.faithful, actual.holding, Set.mem_univ _, params ∅, fun _ => (served_mem ∅).1,
    fun i j h => sound_still (((served_mem ∅).2 i j).1 h).1 0,
    fun i j h => sound_still (((served_mem ∅).2 i j).1 h).2.1 0,
    fun i j h => sound_still (((served_mem ∅).2 i j).1 h).2.2 0⟩

theorem RunSecure_nontrivial : Guarantees.RunSecure_nontrivial :=
  ⟨actual.faithful, actual.holding, Set.mem_univ _, params ∅, fun _ => (trained_mem ∅).1,
    fun _ hδ i j => not_sound_still trivial (trained_not_held ∅ i j) hδ,
    fun _ hδ => not_sound_still trivial (fun h => h (trained_mem ∅).2) hδ⟩

end Slowdown.SecurityProofs
