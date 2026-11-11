module

public import Proofs.Flock.SamplingGeneric.Reg
public import Proofs.Core.SamplingGeneric.FirstHit

/-!
# The first-hit finder at the committed trace

`pr_offReg_le` (`Proofs.Flock.SamplingGeneric.Reg`) bounds a run's openings off the committed trace by
`N·√Pr[conflict]`: the rewinding lemma at the runs' openings of the program's gates (`opTable`), whose plurality the
committed trace is (`plur`). This file bounds the same event at the same table and the same plurality linearly, by the
first-hit finder (`Proofs.Core.SamplingGeneric.FirstHit`), at each tape of the prover:

* `isPlurality_plur`: the committed trace's values are a plurality table of the runs' openings.
* `pr_offReg_le_firstHit`: `Pr[a run opens a gate off the committed trace] ≤ 2·Pr[Found] + N/(K + 1)`, where `Found`
  is the first-hit finder's event at `K` further runs' coins and `N` the program's gates.

It turns no found pair into a collision and builds no `Finder`: `Found` is two runs' extractions that open one gate two
ways, the event `Conflict` reads at a pair of runs.
-/

@[expose] public section

namespace Claims.SamplingGeneric

open Claims Finset

namespace Compiled

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-- The committed trace's values are a plurality table of the runs' openings of the program's gates. -/
theorem isPlurality_plur (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) :
    FlockSoundness.Rewinding.IsPlurality (L.opTable H P x ω) fun g => L.plur H P x ω g := by
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
  intro g v
  have h := le_plur (fun b => Pr fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x =>
    L.opensAt H P x ω rc.1 rc.2 g = some b) v
  have h' : (#(FlockSoundness.Rewinding.openedTo (L.opTable H P x ω) g v) : ℝ) ≤
      #(FlockSoundness.Rewinding.openedTo (L.opTable H P x ω) g (L.plur H P x ω g)) := by
    rw [hcnt, hcnt]
    exact mul_le_mul_of_nonneg_right h hcard.le
  exact_mod_cast h'

/-- **The first-hit bound at the committed trace**: at each tape, a run's extractions open a gate of the program off
the committed trace with probability at most twice the first-hit finder's probability of finding a pair at `K` further
runs' coins, plus `N/(K + 1)`, `N` the program's gates. -/
theorem pr_offReg_le_firstHit (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (K : ℕ) :
    Pr (fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => L.OffReg H P x ω rc.1 rc.2) ≤
      2 * Pr (fun s : ((Fin (L.V.k x) → Coin) × L.SlotEC x) × (Fin K → (Fin (L.V.k x) → Coin) × L.SlotEC x) =>
          Verity.FirstHit.Found (L.opTable H P x ω) K s.1 s.2) +
        (L.nG x : ℝ) / ((K : ℝ) + 1) := by
  have hoff : Pr (fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => L.OffReg H P x ω rc.1 rc.2) ≤
      Pr (fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => ∃ g,
        FlockSoundness.Rewinding.Off (L.opTable H P x ω) (fun g => L.plur H P x ω g) rc g) :=
    Pr_mono fun rc hrc => by
      obtain ⟨g, hg, b, hb, hne⟩ := hrc
      refine ⟨⟨g, hg⟩, Option.isSome_iff_exists.2 ⟨b, hb⟩, fun h => hne ?_⟩
      rw [L.regVal_eq_plur H P x ω hb]
      exact (hb.symm.trans h).symm
  have h1 := Verity.FirstHit.pr_off_le (L.opTable H P x ω) (L.isPlurality_plur H P x ω) K
  have h2 := Verity.FirstHit.pr_miss_le (L.opTable H P x ω) K
  have hK : (0 : ℝ) < (K : ℝ) + 1 := by positivity
  have hw : (#(univ.filter fun g : Fin (L.nG x) =>
      (FlockSoundness.Rewinding.opened (L.opTable H P x ω) g).Nonempty) : ℝ) ≤ (L.nG x : ℝ) := by
    have h := Verity.FirstHit.card_opened_nonempty_le (L.opTable H P x ω)
    rw [Fintype.card_fin] at h
    exact_mod_cast h
  have h3 := div_le_div_of_nonneg_right hw hK.le
  linarith

end Compiled

end Claims.SamplingGeneric
