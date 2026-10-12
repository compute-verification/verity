import Definitions.Accounting.Joint

/-!
# The joint guarantee (trusted)

A joint audit runs the compute, bandwidth and capacity audits on one device at once (`Accounting.Run`). Each audit's
guarantee, in the shape of PoUW's `Pouw.Guarantees.EndToEnd`, holds against the other two audits' coins as advice
(`Run.ComputeSound`, `Run.BandwidthSound`, `Run.CapacitySound`). Then the probability that every audit accepts and some
account is below its certificate (W_cert of work, Q_cert of reads, S_cert of capacity) on every one of its audit's own
coins is at most the three errors summed (`Joint`). Because every step fits the device (`Step.Fits`), the same bound
holds for the other accounts being beyond a residual on every one of its audit's own coins: F·T − W_cert of work,
BW·T − Q_cert of reads and M − S_cert of capacity on average (`CovertResidual`).

Both bound an account's worst case over its audit's own coins, not its charge at the coins drawn. Outside the error, in
an accepted run, `Joint` says each account reaches its certificate on some coin of its audit, and `CovertResidual` that
some coin of each audit leaves the other accounts within its residual. A device could do little charged work and a lot
of covert work on a few of the compute coins. A per-run reading, at the coins drawn, is not claimed yet; lane
`ra-per-run` is proving it by truncation.

Each `def G : Prop` is proved by `Accounting.SecurityProofs.G : Accounting.Guarantees.G` (`Proofs.Accounting.Joint`).
-/

namespace Accounting.Guarantees

open Accounting

/-- **The joint guarantee.** On any device and any run of a joint audit on it whose compute, bandwidth and capacity
audits have their guarantees at certificates `W`, `Q` and `S` with errors `εW`, `εQ` and `εS`, the probability over the
three audits' coins that every audit accepts and some audit's account is below its certificate on every one of the
audit's own coins (at the other audits' coins drawn) is at most `εW + εQ + εS`. This bounds the account's worst case
over its audit's own coins, not its charge at the coin drawn; that per-run reading is not claimed yet. -/
def Joint : Prop :=
  ∀ (ι α ΩW ΩQ ΩS : Type) [Fintype α] [Fintype ΩW] [Fintype ΩQ] [Fintype ΩS] [Nonempty ΩW] [Nonempty ΩQ]
    [Nonempty ΩS] (d : Device ι) (r : Run d α ΩW ΩQ ΩS) (p : Price) (W Q S εW εQ εS : ℝ),
    r.ComputeSound p W εW → r.BandwidthSound Q εQ → r.CapacitySound S εS →
    Pouw.pr (fun ω => r.Accepted ω ∧ (r.ComputeShort p W ω ∨ r.BandwidthShort Q ω ∨ r.CapacityShort S ω))
      ≤ εW + εQ + εS

/-- **The covert residual.** Under the joint guarantee's hypotheses, the probability that every audit accepts and, for
one of the three audits, the other accounts are beyond its residual on every one of the audit's own coins (work beyond
`F·T − W` over the compute window, reads beyond `BW·T − Q` over the bandwidth window, or more than `M − S` bits of
device memory on average over the capacity window) is at most `εW + εQ + εS`. Outside that probability, some coin of
each audit leaves the other accounts within its residual, not necessarily the coin drawn: a device could do little
charged work and a lot of covert work on a few of the compute coins. A per-run reading is not claimed yet. -/
def CovertResidual : Prop :=
  ∀ (ι α ΩW ΩQ ΩS : Type) [Fintype α] [DecidableEq α] [Fintype ΩW] [Fintype ΩQ] [Fintype ΩS] [Nonempty ΩW]
    [Nonempty ΩQ] [Nonempty ΩS] (d : Device ι) (r : Run d α ΩW ΩQ ΩS) (p : Price) (W Q S εW εQ εS : ℝ),
    r.ComputeSound p W εW → r.BandwidthSound Q εQ → r.CapacitySound S εS →
    Pouw.pr (fun ω => r.Accepted ω ∧ (r.ComputeCovert p W ω ∨ r.BandwidthCovert Q ω ∨ r.CapacityCovert S ω))
      ≤ εW + εQ + εS

end Accounting.Guarantees
