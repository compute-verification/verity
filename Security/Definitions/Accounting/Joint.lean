import Definitions.Accounting.Device
import Definitions.Pouw.Basic.Prob

/-!
# A joint audit's run, and what each audit's guarantee says about it (trusted)

A joint audit runs three audits on one device at once: the compute audit (PoUW), the bandwidth audit and the capacity
audit (PoUS). Each draws coins of its own (`ΩW`, `ΩQ`, `ΩS`), uniformly and independently (`Pouw.pr` over the product),
and has an account and a window of its own. The device sees every coin, so its trace is a function of all three.

Each audit's guarantee is stated in the shape of PoUW's `Pouw.Guarantees.EndToEnd`, on its own account and its own
coins: whatever the other two audits' coins are (the device may know them), a run whose account uses at most `B` of
the audit's resource on every one of its coins passes the audit with `B` below the certificate with probability at
most its error. So the other audits' coins are advice the bound holds against, and the guarantee is about the
account's charge alone: what the other accounts use is neither credited to it nor needed by it.

* `ComputeSound p W ε`: the compute account's work at price `p`, certificate W (W_cert).
* `BandwidthSound Q ε`: the bits the bandwidth account read from device memory, certificate Q (Q_cert).
* `CapacitySound S ε`: the device memory the capacity account held over its window, certificate S bits on average
  (S_cert; `S * len` bit-ticks).

The events these guarantees bound quantify over the audit's own coins; they are not about the coin drawn. At a value of
the other audits' coins, an account is short when its charge is below the certificate on every one of the audit's own
coins, so its worst case over them is below the certificate (`ComputeShort`, `BandwidthShort`, `CapacityShort`). The
other accounts are covert when their charges are beyond the residual (the device's capacity over the window, less the
certificate) on every one of the audit's own coins (`ComputeCovert`, `BandwidthCovert`, `CapacityCovert`). A device
that does little charged work and a lot of covert work on a few of the compute coins is in neither event. A per-run
reading, at the coins drawn, is not claimed yet; lane `ra-per-run` is proving it by truncation.
-/

namespace Accounting

/-- The audits of a joint audit. -/
inductive Audit
  /-- PoUW's compute audit -/
  | compute
  /-- the bandwidth audit -/
  | bandwidth
  /-- PoUS's capacity audit -/
  | capacity

/-- A joint audit's run on device `d` with accounts `α`: each audit's account and window, the trace at each value of
the compute, bandwidth and capacity audits' coins, which fits the device, and each audit's verdict. -/
structure Run {ι : Type} (d : Device ι) (α : Type) [Fintype α] (ΩW ΩQ ΩS : Type) where
  /-- the account an audit's resource is charged to -/
  acct : Audit → α
  /-- the window an audit measures -/
  win : Audit → Window
  /-- the execution at the audits' coins -/
  trace : ΩW × ΩQ × ΩS → Trace ι α
  /-- every execution fits the device -/
  fits : ∀ ω, (trace ω).Fits d
  /-- an audit's verdict at the audits' coins -/
  accepts : Audit → ΩW × ΩQ × ΩS → Prop

namespace Run

variable {ι : Type} {d : Device ι} {α ΩW ΩQ ΩS : Type} [Fintype α] (r : Run d α ΩW ΩQ ΩS)

/-- Every audit accepts. -/
def Accepted (ω : ΩW × ΩQ × ΩS) : Prop := ∀ a, r.accepts a ω

/-- The compute account's work over the compute window at price `p`. -/
def work (p : Price) (ω : ΩW × ΩQ × ΩS) : ℕ := (r.trace ω).work (r.win .compute) (r.acct .compute) p

/-- The bits the bandwidth account read from device memory over the bandwidth window. -/
def read (ω : ΩW × ΩQ × ΩS) : ℕ := (r.trace ω).read (r.win .bandwidth) (r.acct .bandwidth)

/-- The device memory the capacity account held over the capacity window, in bit-ticks. -/
def held (ω : ΩW × ΩQ × ΩS) : ℕ := (r.trace ω).held (r.win .capacity) (r.acct .capacity)

/-- The compute account is short of the certificate `W` at the other audits' coins `ω.2`: its work is below `W` on every
one of the compute audit's own coins. -/
def ComputeShort (p : Price) (W : ℝ) (ω : ΩW × ΩQ × ΩS) : Prop := ∀ x : ΩW, (r.work p (x, ω.2) : ℝ) < W

/-- The bandwidth account is short of the certificate `Q` at the other audits' coins: it reads fewer than `Q` bits on
every one of the bandwidth audit's own coins. -/
def BandwidthShort (Q : ℝ) (ω : ΩW × ΩQ × ΩS) : Prop := ∀ q : ΩQ, (r.read (ω.1, q, ω.2.2) : ℝ) < Q

/-- The capacity account is short of the certificate `S` at the other audits' coins: it holds fewer than `S` bits on
average over its window on every one of the capacity audit's own coins. -/
def CapacityShort (S : ℝ) (ω : ΩW × ΩQ × ΩS) : Prop :=
  ∀ s : ΩS, (r.held (ω.1, ω.2.1, s) : ℝ) < S * (r.win .capacity).len

section Residual

variable [DecidableEq α]

/-- The other accounts' work over the compute window at price `p`. -/
def othersWork (p : Price) (ω : ΩW × ΩQ × ΩS) : ℕ :=
  ∑ a ∈ Finset.univ.erase (r.acct .compute), (r.trace ω).work (r.win .compute) a p

/-- The bits the other accounts read from device memory over the bandwidth window. -/
def othersRead (ω : ΩW × ΩQ × ΩS) : ℕ :=
  ∑ a ∈ Finset.univ.erase (r.acct .bandwidth), (r.trace ω).read (r.win .bandwidth) a

/-- The device memory the other accounts held over the capacity window, in bit-ticks. -/
def othersHeld (ω : ΩW × ΩQ × ΩS) : ℕ :=
  ∑ a ∈ Finset.univ.erase (r.acct .capacity), (r.trace ω).held (r.win .capacity) a

/-- The other accounts are beyond the compute residual `F·T - W` at the other audits' coins: on every one of the compute
audit's own coins, their work over the compute window is more than the device's priced work over it less `W`. -/
def ComputeCovert (p : Price) (W : ℝ) (ω : ΩW × ΩQ × ΩS) : Prop :=
  ∀ x : ΩW, ((d.priced p * (r.win .compute).len : ℕ) : ℝ) - W < r.othersWork p (x, ω.2)

/-- The other accounts are beyond the bandwidth residual `BW·T - Q` at the other audits' coins: on every one of the
bandwidth audit's own coins, they read more than the device memory's bits over the bandwidth window less `Q`. -/
def BandwidthCovert (Q : ℝ) (ω : ΩW × ΩQ × ΩS) : Prop :=
  ∀ q : ΩQ, ((d.bw * (r.win .bandwidth).len : ℕ) : ℝ) - Q < r.othersRead (ω.1, q, ω.2.2)

/-- The other accounts are beyond the capacity residual `M - S` at the other audits' coins: on every one of the capacity
audit's own coins, they hold more than `M - S` bits of device memory on average over the capacity window. -/
def CapacityCovert (S : ℝ) (ω : ΩW × ΩQ × ΩS) : Prop :=
  ∀ s : ΩS, ((d.mem : ℝ) - S) * (r.win .capacity).len < r.othersHeld (ω.1, ω.2.1, s)

end Residual

section Sound

variable [Fintype ΩW] [Fintype ΩQ] [Fintype ΩS]

/-- **The compute audit's guarantee, at price `p`, certificate `W` and error `ε`**: at any value of the other audits'
coins, if the compute account's work is at most `T` on every compute coin, the compute audit accepts with `T < W` with
probability at most `ε` over the compute coins. -/
def ComputeSound (p : Price) (W ε : ℝ) : Prop :=
  ∀ (o : ΩQ × ΩS) (T : ℝ), 0 ≤ T → (∀ x : ΩW, (r.work p (x, o) : ℝ) ≤ T) →
    Pouw.pr (fun x : ΩW => r.accepts .compute (x, o) ∧ T < W) ≤ ε

/-- **The bandwidth audit's guarantee, at certificate `Q` and error `ε`**: at any value of the other audits' coins, if
the bandwidth account reads at most `B` bits from device memory on every bandwidth coin, the bandwidth audit accepts
with `B < Q` with probability at most `ε` over the bandwidth coins. -/
def BandwidthSound (Q ε : ℝ) : Prop :=
  ∀ (x : ΩW) (s : ΩS) (B : ℝ), 0 ≤ B → (∀ q : ΩQ, (r.read (x, q, s) : ℝ) ≤ B) →
    Pouw.pr (fun q : ΩQ => r.accepts .bandwidth (x, q, s) ∧ B < Q) ≤ ε

/-- **The capacity audit's guarantee, at certificate `S` bits on average and error `ε`**: at any value of the other
audits' coins, if the capacity account holds at most `B` bit-ticks of device memory over its window on every capacity
coin, the capacity audit accepts with `B < S * len` with probability at most `ε` over the capacity coins. -/
def CapacitySound (S ε : ℝ) : Prop :=
  ∀ (x : ΩW) (q : ΩQ) (B : ℝ), 0 ≤ B → (∀ s : ΩS, (r.held (x, q, s) : ℝ) ≤ B) →
    Pouw.pr (fun s : ΩS => r.accepts .capacity (x, q, s) ∧ B < S * (r.win .capacity).len) ≤ ε

end Sound

end Run

end Accounting
