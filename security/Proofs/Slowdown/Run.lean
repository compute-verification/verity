module

public import Definitions.Slowdown.Guarantees.Run
public import Proofs.Slowdown.SecureWitness

/-!
# `RunSecure`'s witnesses

`RunSecure`'s premises hold together and its conclusion fails without the audits, at the run with no message and no
coin, where soundness is the claim at the one outcome (`sound_still`, `not_sound_still`).
-/

@[expose] public section

namespace Slowdown.SecurityProofs

open Verity Finset Slowdown.Security Slowdown.Security.Witness SecureWitness

open scoped ENNReal

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
  ⟨actual.faithful, actual.holding, Set.mem_univ _, fit_holds recipe, fun _ => params ∅,
    fun _ => (served_mem ∅).1,
    fun i j h => sound_still (((served_mem ∅).2 i j).1 h).1 0,
    fun i j h => sound_still (((served_mem ∅).2 i j).1 h).2.1 0,
    fun i j h => sound_still (((served_mem ∅).2 i j).1 h).2.2 0⟩

theorem RunSecure_nontrivial : Guarantees.RunSecure_nontrivial :=
  ⟨actual.faithful, actual.holding, Set.mem_univ _, fit_holds recipe, fun _ => params ∅,
    fun _ => (trained_mem ∅).1, fun _ hδ i j => not_sound_still trivial (trained_not_held ∅ i j) hδ,
    fun _ hδ => not_sound_still trivial (fun h => h (trained_mem ∅).2) hδ⟩

end Slowdown.SecurityProofs
