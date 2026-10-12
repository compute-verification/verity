import Mathlib.Data.Fintype.Option
import Definitions.Accounting.Ledger.Pin
import Definitions.Accounting.Joint

/-!
# The device model as a ledger (trusted)

The device model (`Definitions.Accounting.Device`) is a ledger over three kinds of leaf (`Leaf`): the chip's cores,
device memory, and the source of each side channel. Its parts are the cores, device memory and each source, and the
device (`Leaf.device`) is the cores with device memory.

A schedule's step charges every resource to an account. Account `a`'s share of the actual circuit
(`Schedule.annotation`): the cores run its ops and hold its on-chip state, device memory holds its device memory, its
reads from device memory move from device memory to the cores, and its side ingress moves from each channel's source
to the cores. The whole actual circuit (`Schedule.circuit`) is every account's share summed. The device's limits are
the ledger's capacities, one tick at a time (`Device.coresCap`, `Device.memCap`, `Device.srcCap`):

| Device limit | Ledger entry |
| --- | --- |
| `rate o`, ops of class `o` per tick | the cores' gates of type `o` |
| `chip`, on-chip state | the cores' bits held |
| `mem`, device memory | device memory's bits held |
| `bw`, reads per tick | device memory's crossing outward |
| `cap i`, side channel `i` per tick | source `i`'s crossing outward |

Each audit of a joint run is an experiment (`Run.computeExperiment`, `Run.bandwidthExperiment`,
`Run.capacityExperiment`): its own coins are the coins, the other two audits' coins are the adversary (the device may
know them), the actual circuit is the audit's account's share, and the verdict is the audit's. Each audit's guarantee
(`Run.ComputeSound`, `Run.BandwidthSound`, `Run.CapacitySound`) is a budget-form pin of that experiment: work at price
`p` on the cores over the compute window, device memory's crossing outward over the bandwidth window, and device
memory's bits held over the capacity window (`Accounting.Ledger.computeSound_iff` and its two companions).

So the device model is the ledger plus an attribution: each unit of every resource is charged to one account, and a pin
is about one account's share, not about the region. Additivity across accounts then holds by construction
(`Accounting.Ledger.pin_add_of_shares`); it is what the ledger names an assumption (`Experiment.Additive`).
-/

namespace Accounting

/-- A leaf of the device model with side channels `ι`. -/
inductive Leaf (ι : Type)
  /-- the chip's cores, with their on-chip state -/
  | cores
  /-- device memory -/
  | mem
  /-- the source of side channel `i` (NVMe, the network) -/
  | src (i : ι)
  deriving DecidableEq

/-- The device model's leaves, as `none` (the cores), `some none` (device memory) and `some (some i)` (source `i`). -/
def Leaf.equiv (ι : Type) : Option (Option ι) ≃ Leaf ι where
  toFun
    | none => .cores
    | some none => .mem
    | some (some i) => .src i
  invFun
    | .cores => none
    | .mem => some none
    | .src i => some (some i)
  left_inv x := by rcases x with _ | _ | _ <;> rfl
  right_inv l := by cases l <;> rfl

instance Leaf.instFintype {ι : Type} [Fintype ι] : Fintype (Leaf ι) := Fintype.ofEquiv _ (Leaf.equiv ι)

/-- The device: its cores and device memory. -/
def Leaf.device {ι : Type} [DecidableEq ι] : Finset (Leaf ι) := {.cores, .mem}

namespace Schedule

variable {ι α : Type} (sc : Schedule ι α)

/-- Account `a`'s share of the actual circuit: its ops and on-chip state on the cores, its device memory in device
memory, its reads from device memory to the cores, and its side ingress from each source to the cores. -/
def annotation (a : α) : Ledger.Annotation (Leaf ι) Op where
  gates l t o := match l with
    | .cores => (sc t).ops a o
    | _ => 0
  held l t := match l with
    | .cores => (sc t).chip a
    | .mem => (sc t).mem a
    | .src _ => 0
  link l l' t := match l, l' with
    | .mem, .cores => (sc t).read a
    | .src i, .cores => (sc t).side a i
    | _, _ => 0

/-- The whole actual circuit: every account's share. -/
def circuit [Fintype α] : Ledger.Annotation (Leaf ι) Op := Ledger.Annotation.sum sc.annotation

end Schedule

namespace Device

variable {ι : Type} [Fintype ι] (d : Device ι)

/-- The cores' capacity per tick: `rate o` ops of each class, `chip` bits held, and inward the reads and every side
channel; nothing outward, since the device model has no egress. -/
def coresCap : Ledger Op
  | .gates o => d.rate o
  | .held => d.chip
  | .crossIn => d.bw + ∑ i, d.cap i
  | .crossOut => 0

/-- Device memory's capacity per tick: `mem` bits held and `bw` bits read outward; no gates, and nothing inward, since
the device model has no writes into device memory. -/
def memCap : Ledger Op
  | .held => d.mem
  | .crossOut => d.bw
  | _ => 0

/-- Source `i`'s capacity per tick: `cap i` bits outward. The sources are the world's, so nothing else is bounded. -/
def srcCap (i : ι) : Ledger Op
  | .crossOut => d.cap i
  | _ => 0

end Device

namespace Run

open Ledger

variable {ι : Type} {d : Device ι} {α ΩW ΩQ ΩS : Type} [Fintype α] (r : Run d α ΩW ΩQ ΩS)

/-- The compute audit as an experiment: the other audits' coins are the adversary, the compute coins are the coins, the
actual circuit is the compute account's share, and the verdict is the compute audit's. -/
def computeExperiment : Experiment (Leaf ι) Op (ΩQ × ΩS) ΩW :=
  ⟨fun o x => (r.schedule (x, o)).annotation (r.acct .compute), fun o x => r.accepts .compute (x, o)⟩

/-- The bandwidth audit as an experiment, with the compute and capacity coins as the adversary. -/
def bandwidthExperiment : Experiment (Leaf ι) Op (ΩW × ΩS) ΩQ :=
  ⟨fun o q => (r.schedule (o.1, q, o.2)).annotation (r.acct .bandwidth), fun o q => r.accepts .bandwidth (o.1, q, o.2)⟩

/-- The capacity audit as an experiment, with the compute and bandwidth coins as the adversary. -/
def capacityExperiment : Experiment (Leaf ι) Op (ΩW × ΩQ) ΩS :=
  ⟨fun o s => (r.schedule (o.1, o.2, s)).annotation (r.acct .capacity), fun o s => r.accepts .capacity (o.1, o.2, s)⟩

end Run

end Accounting
