import Definitions.Accounting.Joint
import Mathlib.Algebra.BigOperators.Group.Finset.Sigma
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.Order.BigOperators.Group.Finset

/-!
# The device's limits over a window

A trace that fits the device charges, over a window, at most the device's limit per tick times the window's length to
all accounts together. So the accounts of any set of distinct accounts are, together, within that bound (no double
counting, `Trace.sum_work_le_of_subset`), and an account's charge plus every other account's is (`Run.work_add_othersWork`
and the two beside it).
-/

namespace Accounting

theorem Window.card_ticks (w : Window) : w.ticks.card = w.len := by
  simp [Window.ticks]

namespace Trace

variable {ι α : Type} [Fintype α] {d : Device ι} {tr : Trace ι α}

theorem sum_ops_le (h : tr.Fits d) (w : Window) (o : Op) : ∑ a, tr.ops w a o ≤ d.rate o * w.len := by
  unfold Trace.ops
  rw [Finset.sum_comm]
  calc ∑ t ∈ w.ticks, ∑ a, (tr t).ops a o ≤ ∑ _t ∈ w.ticks, d.rate o := Finset.sum_le_sum fun t _ => (h t).1 o
    _ = d.rate o * w.len := by rw [Finset.sum_const, Window.card_ticks, smul_eq_mul, mul_comm]

theorem ops_le (h : tr.Fits d) (w : Window) (a : α) (o : Op) : tr.ops w a o ≤ d.rate o * w.len :=
  (Finset.single_le_sum (fun _ _ => Nat.zero_le _) (Finset.mem_univ a)).trans (sum_ops_le h w o)

theorem sum_work_le (h : tr.Fits d) (w : Window) (p : Price) : ∑ a, tr.work w a p ≤ d.priced p * w.len := by
  unfold Trace.work Device.priced
  rw [Finset.sum_comm, Finset.sum_mul]
  refine Finset.sum_le_sum fun o _ => ?_
  rw [← Finset.mul_sum, mul_assoc]
  exact Nat.mul_le_mul_left _ (sum_ops_le h w o)

/-- No double counting: the work of distinct accounts over a window is, together, at most the device's priced work
over it. -/
theorem sum_work_le_of_subset (h : tr.Fits d) (w : Window) (p : Price) (s : Finset α) :
    ∑ a ∈ s, tr.work w a p ≤ d.priced p * w.len :=
  (Finset.sum_le_sum_of_subset (Finset.subset_univ s)).trans (sum_work_le h w p)

theorem sum_read_le (h : tr.Fits d) (w : Window) : ∑ a, tr.read w a ≤ d.bw * w.len := by
  unfold Trace.read
  rw [Finset.sum_comm]
  calc ∑ t ∈ w.ticks, ∑ a, (tr t).read a ≤ ∑ _t ∈ w.ticks, d.bw := Finset.sum_le_sum fun t _ => (h t).2.1
    _ = d.bw * w.len := by rw [Finset.sum_const, Window.card_ticks, smul_eq_mul, mul_comm]

theorem sum_held_le (h : tr.Fits d) (w : Window) : ∑ a, tr.held w a ≤ d.mem * w.len := by
  unfold Trace.held
  rw [Finset.sum_comm]
  calc ∑ t ∈ w.ticks, ∑ a, (tr t).mem a ≤ ∑ _t ∈ w.ticks, d.mem := Finset.sum_le_sum fun t _ => (h t).2.2.2.2
    _ = d.mem * w.len := by rw [Finset.sum_const, Window.card_ticks, smul_eq_mul, mul_comm]

end Trace

namespace Run

variable {ι : Type} {d : Device ι} {α ΩW ΩQ ΩS : Type} [Fintype α] [DecidableEq α] (r : Run d α ΩW ΩQ ΩS)

theorem work_add_othersWork (p : Price) (ω : ΩW × ΩQ × ΩS) :
    r.work p ω + r.othersWork p ω ≤ d.priced p * (r.win .compute).len := by
  calc _ = ∑ a, (r.trace ω).work (r.win .compute) a p := Finset.add_sum_erase _ (fun a => _) (Finset.mem_univ _)
    _ ≤ _ := Trace.sum_work_le (r.fits ω) _ p

theorem read_add_othersRead (ω : ΩW × ΩQ × ΩS) : r.read ω + r.othersRead ω ≤ d.bw * (r.win .bandwidth).len := by
  calc _ = ∑ a, (r.trace ω).read (r.win .bandwidth) a := Finset.add_sum_erase _ (fun a => _) (Finset.mem_univ _)
    _ ≤ _ := Trace.sum_read_le (r.fits ω) _

theorem held_add_othersHeld (ω : ΩW × ΩQ × ΩS) : r.held ω + r.othersHeld ω ≤ d.mem * (r.win .capacity).len := by
  calc _ = ∑ a, (r.trace ω).held (r.win .capacity) a := Finset.add_sum_erase _ (fun a => _) (Finset.mem_univ _)
    _ ≤ _ := Trace.sum_held_le (r.fits ω) _

end Run

end Accounting
