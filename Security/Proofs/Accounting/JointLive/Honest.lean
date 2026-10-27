import Definitions.Accounting.Guarantees.JointLive.Honest
import Proofs.Core.Game.Prob

/-!
# The honest and nontrivial lemmas of `Bridge.BandwidthOfRemoteLineLive`

At the honest node's block as long as the honest remote bandwidth session (`Accounting.Honest.honestPeriod`), the
node's step is the same at every tick, so the bandwidth account's reads over the block are the block's length times the
step's (`read_const`), which is at least what the session credited; the session accepts, and its bad event fails on
every run, so its probability is 0 at every strategy (`prob_eq_zero_of_none`).

At the idle node's block (`Accounting.Honest.idlePeriod`) the bandwidth account reads nothing, so PoUB's claim at the
honest node's certificate `L · 38,400` fails on every run of every block of positive length (`idle_poub_fails`) while
its verdict accepts: the bad event holds on every run, so its probability is 1 at every strategy.
-/

namespace Accounting.Honest

open Verity Verity.Game

/-- An event that fails on every run has probability 0 at every strategy. -/
theorem prob_eq_zero_of_none {O : Type} {p : O → Prop} (h : ∀ o, ¬ p o) (G : Game O) (s : Strategy G) :
    prob p G s = 0 :=
  le_antisymm ((prob_mono (fun o ho => h o ho) G s).trans (prob_false G s).le) zero_le

theorem read_const (st : Step Unit (Fin 3)) (w : Window) (a : Fin 3) :
    Schedule.read (fun _ => st) w a = w.len * st.read a := by
  simp [Schedule.read, Window.ticks]

/-- The idle node reads nothing. -/
theorem idle_read (w : Window) (a : Fin 3) : Schedule.read (fun _ => idle) w a = 0 := by
  simp [Schedule.read, idle]

/-- PoUB's claim at the honest node's certificate fails on every run of an idle block of positive length. -/
theorem idle_poub_fails (rate : Op → ℕ) (cap : Unit → ℕ) {C : Circuit} {n : ℕ} (P : Partition C n)
    (v : C.Transcript) {O : Type} (G : Game O) (L E b : ℕ) (hL : 0 < L) (s : Strategy G) (o : O) :
    ¬ (idlePeriod rate cap P v G L E b).Holds (certs C L E) s .poub o := by
  intro h
  have h' : ((L * 38400 : ℕ) : ℝ) ≤ ((Schedule.read (fun _ => idle) ⟨b * L, L⟩ 1 : ℕ) : ℝ) := h
  rw [idle_read, Nat.cast_zero] at h'
  exact absurd h' (not_le.2 (Nat.cast_pos.2 (Nat.mul_pos hL (by norm_num))))

end Accounting.Honest

namespace Accounting.SecurityProofs

open Accounting Verity.Game
open scoped ENNReal

namespace Bridge

theorem BandwidthOfRemoteLineLive_honest : Accounting.Guarantees.Bridge.BandwidthOfRemoteLineLive_honest := by
  intro rate cap C n P v O G E b
  refine ⟨⟨fun _ => Iff.rfl, fun _ _ => le_rfl, fun s o => ?_⟩,
    fun s => (Honest.prob_eq_zero_of_none (fun _ h => lt_irrefl _ h.2) G s).le⟩
  show ((Honest.sessionBits : ℕ) : ℝ) ≤ ((Schedule.read (fun _ => Honest.step (Honest.h200 rate cap))
    ⟨b * Honest.sessionSpan, Honest.sessionSpan⟩ 1 : ℕ) : ℝ)
  rw [Honest.read_const]
  refine Nat.cast_le.2 ?_
  simp [Honest.step, Honest.h200, Honest.sessionBits, Honest.sessionSpan]

theorem BandwidthOfRemoteLineLive_nontrivial :
    Accounting.Guarantees.Bridge.BandwidthOfRemoteLineLive_nontrivial := by
  intro rate cap C n P v O G L E b s e hL he hle
  refine absurd hle (not_le.2 (lt_of_lt_of_le he ?_))
  calc (1 : ℝ≥0∞) = prob (fun _ => True) G s := (prob_true G s).symm
    _ ≤ _ := prob_mono (fun o _ => ⟨trivial, Honest.idle_poub_fails rate cap P v G L E b hL s o⟩) G s

end Bridge

end Accounting.SecurityProofs
