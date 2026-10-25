import Definitions.Core.Device
import Mathlib.Algebra.BigOperators.Group.Finset.Sigma
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.Order.BigOperators.Group.Finset

/-!
# The device's limits over a window

A schedule that fits the device charges, over a window, at most the device's limit per tick times the window's length
to all accounts together. So the accounts of any set of distinct accounts are, together, within that bound (no double
counting, `Schedule.sum_work_le_of_subset`).
-/

namespace Verity

theorem Window.card_ticks (w : Window) : w.ticks.card = w.len := by
  simp [Window.ticks]

namespace Schedule

variable {ι α : Type} [Fintype α] {d : Device ι} {sc : Schedule ι α}

theorem sum_ops_le (h : sc.Fits d) (w : Window) (o : OpClass) : ∑ a, sc.ops w a o ≤ d.rate o * w.len := by
  unfold Schedule.ops
  rw [Finset.sum_comm]
  calc ∑ t ∈ w.ticks, ∑ a, (sc t).ops a o ≤ ∑ _t ∈ w.ticks, d.rate o := Finset.sum_le_sum fun t _ => (h t).1 o
    _ = d.rate o * w.len := by rw [Finset.sum_const, Window.card_ticks, smul_eq_mul, mul_comm]

theorem ops_le (h : sc.Fits d) (w : Window) (a : α) (o : OpClass) : sc.ops w a o ≤ d.rate o * w.len :=
  (Finset.single_le_sum (fun _ _ => Nat.zero_le _) (Finset.mem_univ a)).trans (sum_ops_le h w o)

theorem sum_work_le (h : sc.Fits d) (w : Window) (p : Price) : ∑ a, sc.work w a p ≤ d.priced p * w.len := by
  unfold Schedule.work Device.priced
  rw [Finset.sum_comm, Finset.sum_mul]
  refine Finset.sum_le_sum fun o _ => ?_
  rw [← Finset.mul_sum, mul_assoc]
  exact Nat.mul_le_mul_left _ (sum_ops_le h w o)

/-- No double counting: the work of distinct accounts over a window is, together, at most the device's priced work
over it. -/
theorem sum_work_le_of_subset (h : sc.Fits d) (w : Window) (p : Price) (s : Finset α) :
    ∑ a ∈ s, sc.work w a p ≤ d.priced p * w.len :=
  (Finset.sum_le_sum_of_subset (Finset.subset_univ s)).trans (sum_work_le h w p)

theorem sum_read_le (h : sc.Fits d) (w : Window) : ∑ a, sc.read w a ≤ d.bw * w.len := by
  unfold Schedule.read
  rw [Finset.sum_comm]
  calc ∑ t ∈ w.ticks, ∑ a, (sc t).read a ≤ ∑ _t ∈ w.ticks, d.bw := Finset.sum_le_sum fun t _ => (h t).2.1
    _ = d.bw * w.len := by rw [Finset.sum_const, Window.card_ticks, smul_eq_mul, mul_comm]

theorem sum_held_le (h : sc.Fits d) (w : Window) : ∑ a, sc.held w a ≤ d.mem * w.len := by
  unfold Schedule.held
  rw [Finset.sum_comm]
  calc ∑ t ∈ w.ticks, ∑ a, (sc t).mem a ≤ ∑ _t ∈ w.ticks, d.mem := Finset.sum_le_sum fun t _ => (h t).2.2.2.2
    _ = d.mem * w.len := by rw [Finset.sum_const, Window.card_ticks, smul_eq_mul, mul_comm]

end Schedule

end Verity
