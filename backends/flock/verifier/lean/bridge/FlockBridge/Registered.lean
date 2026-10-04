import FlockSoundness.Game.Prob
import FlockBridge.Game
import FlockBridge.Circuit

/-!
# Soundness against registered values, in core's terms

A system that verifies a circuit against registered values (the gates `Rg` whose values `a` a commitment binds) is
sound when, for some transcript `X`, an accepting run rarely draws a wrong unit of `X` or finds `X` off `a` on `Rg`
(PoUW's `RegisteredSound`). C-Flock's end-to-end theorem bounds the first half for the transcript it extracts. The
second half reduces to one more C-Flock bound, `RegReadSound`: an accepting run rarely draws a unit whose opening shows a
registered gate at which the extracted transcript is off `a`.

The reduction patches the extracted transcript to `a` on `Rg` (`patch`). The patched transcript is on `a` on `Rg` by
construction, and a unit wrong in it is wrong in the extracted one or shows a patched gate in its opening
(`wrong_patch_subset`, since a unit's correctness reads only its opening). `registered_core_joint` carries one bound on
the two events together to the core game of the same tree, for every core strategy; `registered_core` is it at the
union bound of the two, from C-Flock's end-to-end theorem and `RegReadSound` taken separately.
-/

namespace FlockBridge

namespace Registered

open scoped ENNReal

open FlockSoundness.Audit (Circuit Partition)

variable {C : Circuit} {n : ℕ} (P : Partition C n)

/-- `X` with each registered gate set to its registered value. -/
def patch (X a : Fin C.N → Bool) (Rg : Finset (Fin C.N)) : Fin C.N → Bool :=
  fun g => if g ∈ Rg then a g else X g

open Classical in
/-- The units whose opening shows a registered gate at which `X` is off the registered values `a`. -/
noncomputable def offAt (X a : Fin C.N → Bool) (Rg : Finset (Fin C.N)) : Finset (Fin n) :=
  Finset.univ.filter fun u => ∃ g ∈ P.io u, g ∈ Rg ∧ X g ≠ a g

theorem patch_of_mem {X a : Fin C.N → Bool} {Rg : Finset (Fin C.N)} {g : Fin C.N} (hg : g ∈ Rg) :
    patch X a Rg g = a g := ite_eq_left hg

/-- **A unit wrong in the patched transcript** is wrong in `X`, or its opening shows a registered gate at which `X` is off
the registered values. -/
theorem wrong_patch_subset (X a : Fin C.N → Bool) (Rg : Finset (Fin C.N)) :
    P.wrong (patch X a Rg) ⊆ P.wrong X ∪ offAt P X a Rg := by
  classical
  intro u hu
  rw [Finset.mem_union]
  by_contra h
  rw [not_or, P.mem_wrong, not_not] at h
  obtain ⟨hX, hoff⟩ := h
  refine (P.mem_wrong.1 hu) (P.correct_congr (fun g hg => ?_) hX)
  by_cases hR : g ∈ Rg
  · rw [patch_of_mem hR]
    by_contra hne
    exact hoff (by simpa [offAt] using ⟨g, hg, hR, Ne.symm hne⟩)
  · exact ite_eq_right hR

variable {Out : Type}

/-- **Registered reads are sound, up to `δ`**, at strategy `s`: an accepting run of `G` draws a unit whose opening
shows a registered gate at which `X` is off the registered values `a` with probability at most `δ`. This is the bound
the bridge needs of C-Flock beside its end-to-end theorem, for the transcript that theorem extracts. -/
def RegReadSound (G : FlockSoundness.Game Out) (s : FlockSoundness.Game.Strategy G) (acc : Out → Prop)
    (drawn : Out → Finset (Fin n)) (X a : Fin C.N → Bool) (Rg : Finset (Fin C.N)) (δ : ℝ≥0∞) : Prop :=
  FlockSoundness.Game.prob (fun o => acc o ∧ ¬ Disjoint (offAt P X a Rg) (drawn o)) G s ≤ δ

/-- **Soundness against registered values in the core game, from one joint bound.** Let `G` be a C-Flock game, `σ` any
strategy of its core game, and `X` a transcript for which C-Flock bounds, at `σ`'s C-Flock strategy, an accepting run
that draws a wrong unit of `X` or a unit whose opening shows `X` off the registered values `a` on `Rg`, by `η`. Then some
core transcript makes an accepting run draw a wrong unit or be off `a` on `Rg` with probability at most `η` at `σ`:
PoUW's `RegisteredSound` at `σ`. The patched transcript's wrong units are `X`'s or show it off `a`
(`wrong_patch_subset`), so the event is the joint one, with no union bound. -/
theorem registered_core_joint (G : FlockSoundness.Game Out) (acc : Out → Prop) (drawn : Out → Finset (Fin n))
    (Rg : Finset (Fin C.N)) (a : (Circuit.toCore C).Transcript) (σ : Verity.Game.Strategy (Game.toCore G))
    (X : Fin C.N → Bool) {η : ℝ≥0∞}
    (hjoint : FlockSoundness.Game.prob (fun o => acc o ∧ (¬ Disjoint (P.wrong X) (drawn o) ∨
        ¬ Disjoint (offAt P X (Circuit.Transcript.ofCore a) Rg) (drawn o))) G (Game.Strategy.ofCore G σ) ≤ η) :
    ∃ X' : (Circuit.toCore C).Transcript,
      Verity.Game.prob (fun o => acc o ∧
          (¬ Disjoint ((Partition.toCore P).wrong X') (drawn o) ∨ ∃ g ∈ Rg, X' g ≠ a g))
        (Game.toCore G) σ ≤ η := by
  refine ⟨Circuit.Transcript.toCore (patch X (Circuit.Transcript.ofCore a) Rg), ?_⟩
  rw [← Game.prob_ofCore]
  refine le_trans (FlockSoundness.Game.prob_mono (fun o ho => ?_) _ _) hjoint
  obtain ⟨hacc, h | ⟨g, hg, hne⟩⟩ := ho
  · rw [Partition.wrong_toCore] at h
    obtain ⟨u, hu, hdu⟩ := Finset.not_disjoint_iff.1 h
    rcases Finset.mem_union.1 (wrong_patch_subset P X _ Rg hu) with h' | h'
    · exact ⟨hacc, Or.inl (Finset.not_disjoint_iff.2 ⟨u, h', hdu⟩)⟩
    · exact ⟨hacc, Or.inr (Finset.not_disjoint_iff.2 ⟨u, h', hdu⟩)⟩
  · refine absurd ?_ hne
    show BitVec.ofBool (patch X (Circuit.Transcript.ofCore a) Rg g) = a g
    rw [patch_of_mem hg]
    exact congrFun (Circuit.Transcript.toCore_ofCore a) g

/-- **Soundness against registered values in the core game.** Let `G` be a C-Flock game, `σ` any strategy of its core
game, and `X` a transcript for which C-Flock bounds, at `σ`'s C-Flock strategy, an accepting run that draws a wrong
unit of `X` (or meets some other event `Off`, as C-Flock's end-to-end theorem states it) by `ε`, and registered reads
by `δ` (`RegReadSound`). Then some core transcript makes an accepting run draw a wrong unit or be off the registered
values `a` on `Rg` with probability at most `ε + δ` at `σ`: PoUW's `RegisteredSound` at `σ`, with `η = ε + δ`.
`registered_core_joint` at the union bound of the two. -/
theorem registered_core (G : FlockSoundness.Game Out) (acc : Out → Prop) (drawn : Out → Finset (Fin n))
    (Rg : Finset (Fin C.N)) (a : (Circuit.toCore C).Transcript) (σ : Verity.Game.Strategy (Game.toCore G))
    (X : Fin C.N → Bool) (Off : Prop) {ε δ : ℝ≥0∞}
    (hwrong : FlockSoundness.Game.prob (fun o => acc o ∧ (¬ Disjoint (P.wrong X) (drawn o) ∨ Off)) G
      (Game.Strategy.ofCore G σ) ≤ ε)
    (hreg : RegReadSound P G (Game.Strategy.ofCore G σ) acc drawn X (Circuit.Transcript.ofCore a) Rg δ) :
    ∃ X' : (Circuit.toCore C).Transcript,
      Verity.Game.prob (fun o => acc o ∧
          (¬ Disjoint ((Partition.toCore P).wrong X') (drawn o) ∨ ∃ g ∈ Rg, X' g ≠ a g))
        (Game.toCore G) σ ≤ ε + δ := by
  have hw : FlockSoundness.Game.prob (fun o => acc o ∧ ¬ Disjoint (P.wrong X) (drawn o)) G
      (Game.Strategy.ofCore G σ) ≤ ε :=
    (FlockSoundness.Game.prob_mono (fun o ho => ⟨ho.1, Or.inl ho.2⟩) _ _).trans hwrong
  refine registered_core_joint P G acc drawn Rg a σ X
    (le_trans (FlockSoundness.Game.prob_mono (fun o ho => ?_) _ _)
      ((FlockSoundness.Game.prob_or_le _ _ _ _).trans (add_le_add hw hreg)))
  obtain ⟨hacc, h | h⟩ := ho
  · exact Or.inl ⟨hacc, h⟩
  · exact Or.inr ⟨hacc, h⟩

end Registered

end FlockBridge
