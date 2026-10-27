import Definitions.Accounting.Guarantees.Joint
import Proofs.Accounting.Device
import Proofs.Accounting.Prob
import Mathlib.Tactic.Linarith

/-!
# Proof of the joint guarantee and its covert residual

Each audit's guarantee bounds, over all three audits' coins, the probability that it accepts and its account is below
the certificate on every one of the audit's own coins (`Run.pr_computeShort_le` and the two beside it: `pr_short_le`,
read through the equivalence that puts the audit's own coins last). A union bound sums the three (`Run.joint`). Other
accounts beyond the residual on every one of an audit's own coins leave its account below the certificate on every one
of them, because each account's charge and every other account's are, together, within the device's limit
(`Run.computeShort_of_covert` and the two beside it), so the residual follows from the joint guarantee.
-/

namespace Accounting

open Pouw

namespace Run

variable {ι : Type} {d : Device ι} {α ΩW ΩQ ΩS : Type} [Fintype α] (r : Run d α ΩW ΩQ ΩS)

section Short

variable [Fintype ΩW] [Fintype ΩQ] [Fintype ΩS] [Nonempty ΩW] [Nonempty ΩQ] [Nonempty ΩS]

theorem pr_computeShort_le {p : Price} {W ε : ℝ} (h : r.ComputeSound p W ε) :
    pr (fun ω => r.accepts .compute ω ∧ r.ComputeShort p W ω) ≤ ε := by
  have := pr_short_le (fun x o => r.accepts .compute (x, o)) (fun x o => r.work p (x, o)) W ε h
  rw [← pr_equiv (Equiv.prodComm ΩW (ΩQ × ΩS))] at this
  exact this

/-- The coins with the bandwidth audit's last. -/
def bandwidthLast : ΩW × ΩQ × ΩS ≃ (ΩW × ΩS) × ΩQ where
  toFun ω := ((ω.1, ω.2.2), ω.2.1)
  invFun y := (y.1.1, y.2, y.1.2)
  left_inv _ := rfl
  right_inv _ := rfl

theorem pr_bandwidthShort_le {Q ε : ℝ} (h : r.BandwidthSound Q ε) :
    pr (fun ω => r.accepts .bandwidth ω ∧ r.BandwidthShort Q ω) ≤ ε := by
  have := pr_short_le (fun q (o : ΩW × ΩS) => r.accepts .bandwidth (o.1, q, o.2))
    (fun q o => r.read (o.1, q, o.2)) Q ε fun o B hB hm => h o.1 o.2 B hB hm
  rw [← pr_equiv bandwidthLast] at this
  exact this

theorem pr_capacityShort_le {S ε : ℝ} (h : r.CapacitySound S ε) :
    pr (fun ω => r.accepts .capacity ω ∧ r.CapacityShort S ω) ≤ ε := by
  have := pr_short_le (fun s (o : ΩW × ΩQ) => r.accepts .capacity (o.1, o.2, s))
    (fun s o => r.held (o.1, o.2, s)) (S * (r.win .capacity).len) ε fun o B hB hm => h o.1 o.2 B hB hm
  rw [← pr_equiv (Equiv.prodAssoc ΩW ΩQ ΩS).symm] at this
  exact this

theorem joint {p : Price} {W Q S εW εQ εS : ℝ} (hW : r.ComputeSound p W εW) (hQ : r.BandwidthSound Q εQ)
    (hS : r.CapacitySound S εS) :
    pr (fun ω => r.Accepted ω ∧ (r.ComputeShort p W ω ∨ r.BandwidthShort Q ω ∨ r.CapacityShort S ω))
      ≤ εW + εQ + εS := by
  calc _ ≤ pr (fun ω => (r.accepts .compute ω ∧ r.ComputeShort p W ω) ∨
          (r.accepts .bandwidth ω ∧ r.BandwidthShort Q ω) ∨ (r.accepts .capacity ω ∧ r.CapacityShort S ω)) :=
        pr_mono fun ω h => h.2.imp (⟨h.1 .compute, ·⟩) (Or.imp (⟨h.1 .bandwidth, ·⟩) (⟨h.1 .capacity, ·⟩))
    _ ≤ εW + (εQ + εS) :=
        (pr_or_le _ _).trans (add_le_add (r.pr_computeShort_le hW)
          ((pr_or_le _ _).trans (add_le_add (r.pr_bandwidthShort_le hQ) (r.pr_capacityShort_le hS))))
    _ = εW + εQ + εS := (add_assoc _ _ _).symm

end Short

section Covert

variable [DecidableEq α]

theorem computeShort_of_covert {p : Price} {W : ℝ} {ω : ΩW × ΩQ × ΩS} (h : r.ComputeCovert p W ω) :
    r.ComputeShort p W ω := fun x => by
  have hle : ((r.work p (x, ω.2) : ℕ) : ℝ) + r.othersWork p (x, ω.2) ≤
      ((d.priced p * (r.win .compute).len : ℕ) : ℝ) := by
    exact_mod_cast r.work_add_othersWork p (x, ω.2)
  linarith [h x]

theorem bandwidthShort_of_covert {Q : ℝ} {ω : ΩW × ΩQ × ΩS} (h : r.BandwidthCovert Q ω) : r.BandwidthShort Q ω :=
  fun q => by
  have hle : ((r.read (ω.1, q, ω.2.2) : ℕ) : ℝ) + r.othersRead (ω.1, q, ω.2.2) ≤
      ((d.bw * (r.win .bandwidth).len : ℕ) : ℝ) := by
    exact_mod_cast r.read_add_othersRead (ω.1, q, ω.2.2)
  linarith [h q]

theorem capacityShort_of_covert {S : ℝ} {ω : ΩW × ΩQ × ΩS} (h : r.CapacityCovert S ω) : r.CapacityShort S ω :=
  fun s => by
  have hle : ((r.held (ω.1, ω.2.1, s) : ℕ) : ℝ) + r.othersHeld (ω.1, ω.2.1, s) ≤
      (d.mem : ℝ) * (r.win .capacity).len := by
    exact_mod_cast r.held_add_othersHeld (ω.1, ω.2.1, s)
  have hring : ((d.mem : ℝ) - S) * (r.win .capacity).len = d.mem * (r.win .capacity).len - S * (r.win .capacity).len := by
    ring
  linarith [h s]

variable [Fintype ΩW] [Fintype ΩQ] [Fintype ΩS] [Nonempty ΩW] [Nonempty ΩQ] [Nonempty ΩS]

theorem covertResidual {p : Price} {W Q S εW εQ εS : ℝ} (hW : r.ComputeSound p W εW)
    (hQ : r.BandwidthSound Q εQ) (hS : r.CapacitySound S εS) :
    pr (fun ω => r.Accepted ω ∧ (r.ComputeCovert p W ω ∨ r.BandwidthCovert Q ω ∨ r.CapacityCovert S ω))
      ≤ εW + εQ + εS :=
  (pr_mono fun _ h => ⟨h.1, h.2.imp r.computeShort_of_covert
    (Or.imp r.bandwidthShort_of_covert r.capacityShort_of_covert)⟩).trans (r.joint hW hQ hS)

end Covert

end Run

namespace SecurityProofs

theorem Joint : Accounting.Guarantees.Joint := by
  intro ι α ΩW ΩQ ΩS _ _ _ _ _ _ _ d r p W Q S εW εQ εS hW hQ hS
  exact r.joint hW hQ hS

theorem CovertResidual : Accounting.Guarantees.CovertResidual := by
  intro ι α ΩW ΩQ ΩS _ _ _ _ _ _ _ _ d r p W Q S εW εQ εS hW hQ hS
  exact r.covertResidual hW hQ hS

end SecurityProofs

end Accounting
