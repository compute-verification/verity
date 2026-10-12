import Mathlib.Data.Fintype.BigOperators
import Proofs.Accounting.Ledger.Basic
import Definitions.Accounting.Ledger.Device

/-!
# The device model is a ledger

A share's ledger on each leaf is the device model's charge (`Schedule.ledger_cores_gates` and its companions), so the
device's limits are capacities of the whole actual circuit (`Schedule.Fits.coresCap`, `Schedule.Fits.memCap`,
`Schedule.Fits.srcCap`; `Run.capacity_cores` and its companions), and each audit's guarantee is a budget-form pin of
its experiment: work at price `p` on the cores over the compute window (`Run.computeSound_iff`), device memory's
crossing outward over the bandwidth window (`Run.bandwidthSound_iff`) and its bits held over the capacity window
(`Run.capacitySound_iff`).
-/

namespace Accounting

open Ledger

theorem Leaf.sum_eq {ι M : Type} [Fintype ι] [AddCommMonoid M] (f : Leaf ι → M) :
    ∑ l, f l = f .cores + (f .mem + ∑ i, f (.src i)) := by
  rw [← (Leaf.equiv ι).sum_comp]
  simp [Fintype.sum_option, Leaf.equiv]

namespace Schedule

variable {ι α : Type} [Fintype ι] [DecidableEq ι] (sc : Schedule ι α) (a : α) (T : Finset ℕ)

omit [Fintype ι] [DecidableEq ι] in
theorem annotation_link_self (l : Leaf ι) (t : ℕ) : (sc.annotation a).link l l t = 0 := by
  cases l <;> rfl

theorem ledger_cores_gates (o : Op) :
    (sc.annotation a).ledger ⟨{.cores}, T⟩ (.gates o) = ∑ t ∈ T, (sc t).ops a o :=
  ledger_singleton_gates _ _ _ _

theorem ledger_cores_held : (sc.annotation a).ledger ⟨{.cores}, T⟩ .held = ∑ t ∈ T, (sc t).chip a :=
  ledger_singleton_held _ _ _

theorem ledger_cores_crossIn :
    (sc.annotation a).ledger ⟨{.cores}, T⟩ .crossIn = ∑ t ∈ T, ((sc t).read a + ∑ i, (sc t).side a i) := by
  rw [ledger_singleton_crossIn _ (sc.annotation_link_self a .cores)]
  refine Finset.sum_congr rfl fun t _ => ?_
  rw [Leaf.sum_eq]
  simp [annotation]

theorem ledger_cores_crossOut : (sc.annotation a).ledger ⟨{.cores}, T⟩ .crossOut = 0 := by
  rw [ledger_singleton_crossOut _ (sc.annotation_link_self a .cores)]
  refine Finset.sum_eq_zero fun t _ => ?_
  rw [Leaf.sum_eq]
  simp [annotation]

theorem ledger_mem_gates (o : Op) : (sc.annotation a).ledger ⟨{.mem}, T⟩ (.gates o) = 0 := by
  rw [ledger_singleton_gates]
  simp [annotation]

theorem ledger_mem_held : (sc.annotation a).ledger ⟨{.mem}, T⟩ .held = ∑ t ∈ T, (sc t).mem a :=
  ledger_singleton_held _ _ _

theorem ledger_mem_crossIn : (sc.annotation a).ledger ⟨{.mem}, T⟩ .crossIn = 0 := by
  rw [ledger_singleton_crossIn _ (sc.annotation_link_self a .mem)]
  refine Finset.sum_eq_zero fun t _ => ?_
  rw [Leaf.sum_eq]
  simp [annotation]

theorem ledger_mem_crossOut : (sc.annotation a).ledger ⟨{.mem}, T⟩ .crossOut = ∑ t ∈ T, (sc t).read a := by
  rw [ledger_singleton_crossOut _ (sc.annotation_link_self a .mem)]
  refine Finset.sum_congr rfl fun t _ => ?_
  rw [Leaf.sum_eq]
  simp [annotation]

theorem ledger_src_gates (i : ι) (o : Op) : (sc.annotation a).ledger ⟨{.src i}, T⟩ (.gates o) = 0 := by
  rw [ledger_singleton_gates]
  simp [annotation]

theorem ledger_src_held (i : ι) : (sc.annotation a).ledger ⟨{.src i}, T⟩ .held = 0 := by
  rw [ledger_singleton_held]
  simp [annotation]

theorem ledger_src_crossIn (i : ι) : (sc.annotation a).ledger ⟨{.src i}, T⟩ .crossIn = 0 := by
  rw [ledger_singleton_crossIn _ (sc.annotation_link_self a (.src i))]
  refine Finset.sum_eq_zero fun t _ => ?_
  rw [Leaf.sum_eq]
  simp [annotation]

theorem ledger_src_crossOut (i : ι) :
    (sc.annotation a).ledger ⟨{.src i}, T⟩ .crossOut = ∑ t ∈ T, (sc t).side a i := by
  rw [ledger_singleton_crossOut _ (sc.annotation_link_self a (.src i))]
  refine Finset.sum_congr rfl fun t _ => ?_
  rw [Leaf.sum_eq]
  simp [annotation]

variable [Fintype α]

/-- An account's share of a region's ledger is at most the whole actual circuit's. -/
theorem share_le_circuit (ρ : Region (Leaf ι)) : (sc.annotation a).ledger ρ ≤ sc.circuit.ledger ρ := by
  rw [circuit, ledger_sum]
  exact Finset.single_le_sum (f := fun b => (sc.annotation b).ledger ρ) (fun _ _ _ => Nat.zero_le _) (Finset.mem_univ a)

theorem circuit_ledger_apply (ρ : Region (Leaf ι)) (e : Entry Op) :
    sc.circuit.ledger ρ e = ∑ b, (sc.annotation b).ledger ρ e := by
  rw [circuit, ledger_sum, Finset.sum_apply]

variable {d : Device ι} {sc}

/-- **The cores' capacity**: a step that fits the device is within the cores' capacity. -/
theorem Fits.coresCap (h : sc.Fits d) (t : ℕ) : sc.circuit.ledger ⟨{.cores}, {t}⟩ ≤ d.coresCap := by
  intro e
  rw [circuit_ledger_apply]
  cases e with
  | gates o => simpa [ledger_cores_gates, Device.coresCap] using (h t).1 o
  | held => simpa [ledger_cores_held, Device.coresCap] using (h t).2.2.2.1
  | crossIn =>
    simp only [ledger_cores_crossIn, Finset.sum_singleton, Finset.sum_add_distrib, Device.coresCap]
    rw [Finset.sum_comm]
    exact add_le_add (h t).2.1 (Finset.sum_le_sum fun i _ => (h t).2.2.1 i)
  | crossOut => simp [ledger_cores_crossOut, Device.coresCap]

/-- **Device memory's capacity**: a step that fits the device is within device memory's capacity. -/
theorem Fits.memCap (h : sc.Fits d) (t : ℕ) : sc.circuit.ledger ⟨{.mem}, {t}⟩ ≤ d.memCap := by
  intro e
  rw [circuit_ledger_apply]
  cases e with
  | gates o => simp [ledger_mem_gates]
  | held => simpa [ledger_mem_held, Device.memCap] using (h t).2.2.2.2
  | crossIn => simp [ledger_mem_crossIn]
  | crossOut => simpa [ledger_mem_crossOut, Device.memCap] using (h t).2.1

/-- **A side channel's capacity**: a step that fits the device is within each source's capacity. -/
theorem Fits.srcCap (h : sc.Fits d) (i : ι) (t : ℕ) : sc.circuit.ledger ⟨{.src i}, {t}⟩ ≤ d.srcCap i := by
  intro e
  rw [circuit_ledger_apply]
  cases e with
  | gates o => simp [ledger_src_gates]
  | held => simp [ledger_src_held]
  | crossIn => simp [ledger_src_crossIn]
  | crossOut => simpa [ledger_src_crossOut, Device.srcCap] using (h t).2.2.1 i

end Schedule

namespace Run

variable {ι : Type} [Fintype ι] [DecidableEq ι] {d : Device ι} {α ΩW ΩQ ΩS : Type} [Fintype α]
  (r : Run d α ΩW ΩQ ΩS)

theorem capacity_cores : Capacity (fun ω => (r.schedule ω).circuit) {.cores} d.coresCap :=
  fun ω t => (r.fits ω).coresCap t

theorem capacity_mem : Capacity (fun ω => (r.schedule ω).circuit) {.mem} d.memCap :=
  fun ω t => (r.fits ω).memCap t

theorem capacity_src (i : ι) : Capacity (fun ω => (r.schedule ω).circuit) {.src i} (d.srcCap i) :=
  fun ω t => (r.fits ω).srcCap i t

theorem computeExperiment_read (p : Price) (o : ΩQ × ΩS) (x : ΩW) :
    r.computeExperiment.read ⟨{.cores}, (r.win .compute).ticks⟩ (Meter.priced p) o x = r.work p (x, o) := by
  simp only [Experiment.read, Meter.eval_priced, computeExperiment, Schedule.ledger_cores_gates, work,
    Schedule.work, Schedule.ops]

theorem bandwidthExperiment_read (o : ΩW × ΩS) (q : ΩQ) :
    r.bandwidthExperiment.read ⟨{.mem}, (r.win .bandwidth).ticks⟩ (Meter.entry .crossOut) o q =
      r.read (o.1, q, o.2) := by
  simp only [Experiment.read, Meter.eval_entry, bandwidthExperiment, Schedule.ledger_mem_crossOut, read,
    Schedule.read]

theorem capacityExperiment_read (o : ΩW × ΩQ) (s : ΩS) :
    r.capacityExperiment.read ⟨{.mem}, (r.win .capacity).ticks⟩ (Meter.entry .held) o s = r.held (o.1, o.2, s) := by
  simp only [Experiment.read, Meter.eval_entry, capacityExperiment, Schedule.ledger_mem_held, held, Schedule.held]

/-- **The compute audit's guarantee is a budget-form pin** of work at price `p` on the cores over the compute window. -/
theorem computeSound_iff [Fintype ΩW] (p : Price) (W ε : ℝ) :
    r.ComputeSound p W ε ↔
      r.computeExperiment.PinBudget ⟨{.cores}, (r.win .compute).ticks⟩ (Meter.priced p) W ε := by
  simp only [ComputeSound, Experiment.PinBudget, computeExperiment_read]
  rfl

/-- **The bandwidth audit's guarantee is a budget-form pin** of device memory's crossing outward over the bandwidth
window. -/
theorem bandwidthSound_iff [Fintype ΩQ] (Q ε : ℝ) :
    r.BandwidthSound Q ε ↔
      r.bandwidthExperiment.PinBudget ⟨{.mem}, (r.win .bandwidth).ticks⟩ (Meter.entry .crossOut) Q ε := by
  simp only [BandwidthSound, Experiment.PinBudget, bandwidthExperiment_read, Prod.forall]
  rfl

/-- **The capacity audit's guarantee is a budget-form pin** of device memory's bits held over the capacity window, at
`S` bits on average: `S·len` bit-ticks. -/
theorem capacitySound_iff [Fintype ΩS] (S ε : ℝ) :
    r.CapacitySound S ε ↔
      r.capacityExperiment.PinBudget ⟨{.mem}, (r.win .capacity).ticks⟩ (Meter.entry .held)
        (S * (r.win .capacity).len) ε := by
  simp only [CapacitySound, Experiment.PinBudget, capacityExperiment_read, Prod.forall]
  rfl

end Run

end Accounting
