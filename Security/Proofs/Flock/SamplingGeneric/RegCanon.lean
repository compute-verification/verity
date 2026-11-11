module

public import Proofs.Flock.SamplingGeneric.Reg
public import Proofs.Core.Game.RewindingCanon

/-!
# The rewinding lemma at the committed trace, counting each bad run at one gate

`pr_offReg_le` at `Verity.Rewinding.prBad_le_canon` in place of `prBad_le`: a run's extractions open a gate of the
program off the committed trace with probability at most `√(N·Pr[conflict])`, `N` the program's gates, where
`pr_offReg_le` gives `N·√Pr[conflict]`.
-/

@[expose] public section

namespace Claims.SamplingGeneric

open Claims Finset

namespace Compiled

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-- **The rewinding lemma at the committed trace, one gate a bad run**: at each tape, a run's extractions open a gate
of the program off the committed trace with probability at most `√(N·Pr[conflict])`. -/
theorem pr_offReg_le_canon (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) :
    Pr (fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => L.OffReg H P x ω rc.1 rc.2) ≤
      Real.sqrt (L.nG x * Pr fun rr : ((Fin (L.V.k x) → Coin) × L.SlotEC x) × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
        L.Conflict H P x ω rr) := by
  classical
  have hcard : (0 : ℝ) < Fintype.card ((Fin (L.V.k x) → Coin) × L.SlotEC x) := by
    exact_mod_cast Fintype.card_pos
  have hcnt : ∀ (g : Fin (L.nG x)) (b : Bool),
      (#(FlockSoundness.Rewinding.openedTo (L.opTable H P x ω) g b) : ℝ) =
        Pr (fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => L.opensAt H P x ω rc.1 rc.2 g = some b) *
          Fintype.card ((Fin (L.V.k x) → Coin) × L.SlotEC x) := by
    intro g b
    rw [Pr_eq_card_of_mem (fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => L.opensAt H P x ω rc.1 rc.2 g = some b)
      (FlockSoundness.Rewinding.openedTo (L.opTable H P x ω) g b)
      fun rc => mem_filter.trans (and_iff_right (mem_univ _)), div_mul_cancel₀ _ hcard.ne']
  have hpl : FlockSoundness.Rewinding.IsPlurality (L.opTable H P x ω) fun g => L.plur H P x ω g := by
    intro g v
    have h := le_plur (fun b => Pr fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x =>
      L.opensAt H P x ω rc.1 rc.2 g = some b) v
    have h' : (#(FlockSoundness.Rewinding.openedTo (L.opTable H P x ω) g v) : ℝ) ≤
        #(FlockSoundness.Rewinding.openedTo (L.opTable H P x ω) g (L.plur H P x ω g)) := by
      rw [hcnt, hcnt]
      exact mul_le_mul_of_nonneg_right h hcard.le
    exact_mod_cast h'
  have hbad := Verity.Rewinding.prBad_le_canon (op := L.opTable H P x ω) (fun g => L.plur H P x ω g) hpl
  have hoff : Pr (fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => L.OffReg H P x ω rc.1 rc.2) ≤
      (#(FlockSoundness.Rewinding.bad (L.opTable H P x ω) fun g => L.plur H P x ω g) : ℝ) /
        Fintype.card ((Fin (L.V.k x) → Coin) × L.SlotEC x) := by
    calc Pr (fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => L.OffReg H P x ω rc.1 rc.2)
        ≤ Pr (fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => ∃ g,
            FlockSoundness.Rewinding.Off (L.opTable H P x ω) (fun g => L.plur H P x ω g) rc g) :=
          Pr_mono fun rc hrc => by
            obtain ⟨g, hg, b, hb, hne⟩ := hrc
            refine ⟨⟨g, hg⟩, Option.isSome_iff_exists.2 ⟨b, hb⟩, fun h => hne ?_⟩
            rw [L.regVal_eq_plur H P x ω hb]
            exact (hb.symm.trans h).symm
      _ = _ := Pr_eq_card_of_mem _ (FlockSoundness.Rewinding.bad (L.opTable H P x ω) fun g => L.plur H P x ω g)
          fun rc => by simp only [FlockSoundness.Rewinding.bad, mem_filter, mem_univ, true_and]
  have hconf : (#(FlockSoundness.Rewinding.conflict (L.opTable H P x ω)) : ℝ) /
      (Fintype.card ((Fin (L.V.k x) → Coin) × L.SlotEC x) : ℝ) ^ 2 =
      Pr fun rr : ((Fin (L.V.k x) → Coin) × L.SlotEC x) × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
        L.Conflict H P x ω rr := by
    rw [Pr_eq_card_of_mem (fun rr : ((Fin (L.V.k x) → Coin) × L.SlotEC x) × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
        L.Conflict H P x ω rr) (FlockSoundness.Rewinding.conflict (L.opTable H P x ω))
        fun rr => by simp only [FlockSoundness.Rewinding.conflict, mem_filter, mem_univ, true_and]; exact Iff.rfl,
      Fintype.card_prod ((Fin (L.V.k x) → Coin) × L.SlotEC x) ((Fin (L.V.k x) → Coin) × L.SlotEC x),
      Nat.cast_mul, sq]
  calc Pr (fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => L.OffReg H P x ω rc.1 rc.2)
      ≤ _ := hoff
    _ ≤ _ := hbad
    _ = Real.sqrt (L.nG x * Pr fun rr : ((Fin (L.V.k x) → Coin) × L.SlotEC x) ×
          ((Fin (L.V.k x) → Coin) × L.SlotEC x) => L.Conflict H P x ω rr) := by
      rw [Fintype.card_fin, hconf]

end Compiled

end Claims.SamplingGeneric
