import Models.Accounting.Guarantees.Capacity
import Proofs.Accounting.Joint
import Mathlib.Tactic.Linarith

/-!
# Proof that a state-bounded game's soundness is a joint run's capacity guarantee, and that derived state shares the
residual

At a value of the other audits' coins, a bound `B` on the capacity account's bit-ticks over the capacity window on every
capacity coin bounds its memory at every tick of the window by `⌊B⌋` (`Run.mem_le_held`), so the device is a server with
state `⌊B⌋ + S_chip + ∑ cap_i · round`, at most `S₀` when `B` is below the certificate's bit-ticks, and the game bounds
its pass, which is the capacity audit's accept (`Run.capacitySound_of_state`). An account `w` beside the capacity account
is one of the other accounts, so what it holds and what the rest hold together are the other accounts' memory
(`Run.othersHeld_eq`), and the other accounts beyond the residual less `w`'s share on every capacity coin leave the
capacity account below the certificate on every one of them (`Run.capacityShort_of_covertBut`).
-/

namespace Accounting

open Pouw

namespace Run

variable {ι : Type} {d : Device ι} {α ΩW ΩQ ΩS : Type} [Fintype α] (r : Run d α ΩW ΩQ ΩS)

theorem mem_le_held {ω : ΩW × ΩQ × ΩS} {t : ℕ} (ht : t ∈ (r.win .capacity).ticks) :
    (r.trace ω t).mem (r.acct .capacity) ≤ r.held ω := by
  unfold Run.held Trace.held
  exact Finset.single_le_sum (f := fun t => (r.trace ω t).mem (r.acct .capacity)) (fun _ _ => Nat.zero_le _) ht

theorem capacitySound_of_state [Fintype ι] [Fintype ΩW] [Fintype ΩQ] [Fintype ΩS] {G : StateGame ΩS}
    {round S₀ : ℕ} {S ε : ℝ} (hε : 0 ≤ ε) (hG : ∀ n ≤ S₀, ∀ A : G.Server n, pr (G.Passes A) ≤ ε)
    (real : RealizesState G r round)
    (hS : S * (r.win .capacity).len + d.chip + ((∑ i, d.cap i * round : ℕ) : ℝ) ≤ S₀) : r.CapacitySound S ε := by
  intro x q B hB hheld
  by_cases hBS : B < S * (r.win .capacity).len
  · have hmem : ∀ s, ∀ t ∈ (r.win .capacity).ticks, (r.trace (x, q, s) t).mem (r.acct .capacity) ≤ ⌊B⌋₊ :=
      fun s t ht => Nat.le_floor ((Nat.cast_le.2 (r.mem_le_held ht)).trans (hheld s))
    have hn : ⌊B⌋₊ + d.chip + ∑ i, d.cap i * round ≤ S₀ := by
      have hfl := Nat.floor_le hB
      have : ((⌊B⌋₊ + d.chip + ∑ i, d.cap i * round : ℕ) : ℝ) ≤ S₀ := by
        rw [Nat.cast_add, Nat.cast_add]
        linarith
      exact_mod_cast this
    exact (pr_mono fun s hs => (real.accepts_iff (x, q) _ hmem s).1 hs.1).trans (hG _ hn _)
  · rw [pr_eq_zero fun s hs => hBS hs.2]
    exact hε

section Derived

variable [DecidableEq α]

theorem othersHeld_eq {w : α} (hw : w ≠ r.acct .capacity) (ω : ΩW × ΩQ × ΩS) :
    r.othersHeld ω = (r.trace ω).held (r.win .capacity) w + r.othersHeldBut w ω :=
  (Finset.add_sum_erase _ (fun a => (r.trace ω).held (r.win .capacity) a)
    (Finset.mem_erase.2 ⟨hw, Finset.mem_univ _⟩)).symm

theorem capacityShort_of_covertBut {w : α} (hw : w ≠ r.acct .capacity) {S : ℝ} {ω : ΩW × ΩQ × ΩS}
    (h : r.CapacityCovertBut w S ω) : r.CapacityShort S ω := fun s => by
  have hle : ((r.held (ω.1, ω.2.1, s) : ℕ) : ℝ) + r.othersHeld (ω.1, ω.2.1, s) ≤
      (d.mem : ℝ) * (r.win .capacity).len := by
    exact_mod_cast r.held_add_othersHeld (ω.1, ω.2.1, s)
  have heq : (r.othersHeld (ω.1, ω.2.1, s) : ℝ) =
      ((r.trace (ω.1, ω.2.1, s)).held (r.win .capacity) w : ℝ) + r.othersHeldBut w (ω.1, ω.2.1, s) := by
    exact_mod_cast r.othersHeld_eq hw _
  have hring : ((d.mem : ℝ) - S) * (r.win .capacity).len =
      d.mem * (r.win .capacity).len - S * (r.win .capacity).len := by
    ring
  linarith [h s]

theorem derivedStateResidual [Fintype ΩW] [Fintype ΩQ] [Fintype ΩS] [Nonempty ΩW] [Nonempty ΩQ] [Nonempty ΩS]
    {w : α} (hw : w ≠ r.acct .capacity) {S εS : ℝ} (hS : r.CapacitySound S εS) :
    pr (fun ω => r.accepts .capacity ω ∧ r.CapacityCovertBut w S ω) ≤ εS :=
  (pr_mono fun _ h => ⟨h.1, r.capacityShort_of_covertBut hw h.2⟩).trans (r.pr_capacityShort_le hS)

end Derived

end Run

namespace SecurityProofs

theorem CapacityOfStateBound : Accounting.Guarantees.CapacityOfStateBound := by
  intro ι α ΩW ΩQ ΩS _ _ _ _ _ d r G round S₀ S ε hε hG real hS
  exact r.capacitySound_of_state hε hG real hS

theorem DerivedStateResidual : Accounting.Guarantees.DerivedStateResidual := by
  intro ι α ΩW ΩQ ΩS _ _ _ _ _ _ _ _ d r w S εS hw hS
  exact r.derivedStateResidual hw hS

end SecurityProofs

end Accounting
