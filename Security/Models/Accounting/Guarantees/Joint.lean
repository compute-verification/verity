import Definitions.Accounting.Joint

/-!
# The joint guarantee (trusted)

A joint audit runs the compute, bandwidth and capacity audits on one device at once (`Accounting.Run`). Each audit's
guarantee, in the shape of PoUW's `Pouw.Guarantees.EndToEnd`, holds against the other two audits' coins as advice
(`Run.ComputeSound`, `Run.BandwidthSound`, `Run.CapacitySound`). Then an accepted joint audit gives all three
certificates at once, W_cert of work, Q_cert of reads and S_cert of capacity, except with the three errors summed
(`Joint`). Because every step fits the device (`Step.Fits`), each certificate leaves at most the device's capacity
over its window, less the certificate, to every other account: F·T − W_cert of work, BW·T − Q_cert of reads and
M − S_cert of capacity on average (`CovertResidual`).

Each `def G : Prop` is proved by `Accounting.SecurityProofs.G : Accounting.Guarantees.G` (`Proofs.Accounting.Joint`).
-/

namespace Accounting.Guarantees

open Accounting

/-- **The joint guarantee.** On any device and any run of a joint audit on it whose compute, bandwidth and capacity
audits have their guarantees at certificates `W`, `Q` and `S` with errors `εW`, `εQ` and `εS`, the probability over the
three audits' coins that every audit accepts and any one of the three certificates fails is at most
`εW + εQ + εS`. -/
def Joint : Prop :=
  ∀ (ι α ΩW ΩQ ΩS : Type) [Fintype α] [Fintype ΩW] [Fintype ΩQ] [Fintype ΩS] [Nonempty ΩW] [Nonempty ΩQ]
    [Nonempty ΩS] (d : Device ι) (r : Run d α ΩW ΩQ ΩS) (p : Price) (W Q S εW εQ εS : ℝ),
    r.ComputeSound p W εW → r.BandwidthSound Q εQ → r.CapacitySound S εS →
    Pouw.pr (fun ω => r.Accepted ω ∧ (r.ComputeShort p W ω ∨ r.BandwidthShort Q ω ∨ r.CapacityShort S ω))
      ≤ εW + εQ + εS

/-- **The covert residual.** Under the joint guarantee's hypotheses, the probability that every audit accepts and the
other accounts use more than one of the three residuals (work beyond `F·T − W` over the compute window, reads beyond
`BW·T − Q` over the bandwidth window, or more than `M − S` bits of device memory on average over the capacity window)
is at most `εW + εQ + εS`. -/
def CovertResidual : Prop :=
  ∀ (ι α ΩW ΩQ ΩS : Type) [Fintype α] [DecidableEq α] [Fintype ΩW] [Fintype ΩQ] [Fintype ΩS] [Nonempty ΩW]
    [Nonempty ΩQ] [Nonempty ΩS] (d : Device ι) (r : Run d α ΩW ΩQ ΩS) (p : Price) (W Q S εW εQ εS : ℝ),
    r.ComputeSound p W εW → r.BandwidthSound Q εQ → r.CapacitySound S εS →
    Pouw.pr (fun ω => r.Accepted ω ∧ (r.ComputeCovert p W ω ∨ r.BandwidthCovert Q ω ∨ r.CapacityCovert S ω))
      ≤ εW + εQ + εS

end Accounting.Guarantees
