import Definitions.Accounting.Joint
import Definitions.Core.Partition

/-!
# A period audited by every protocol, and what each audit's guarantee says about it (trusted)

A period is one window of one node's run, over one circuit (`Verity.Circuit`, core's) and one device
(`Accounting.Device`, the device model). Five audits decide it, each separately, on its own coins (`Audit`):

* sampled proofs, on the integrity of the circuit's committed transcript: at most a share `ε` of the units of its
  partition are wrong (`Verity.Partition.wrong`);
* PoUW, on work: the compute account did at least `W` of work at price `p`;
* PoUS, on space: the capacity account held at least `S` bits of device memory on average;
* PoUB, the bandwidth audit, on bandwidth: the bandwidth account read at least `Q` bits from device memory;
* the network certifier, on communication: what the network certifiers saw the node move is within `K` bits.

The period's coins are one space, `(i : I) → Ω i`, uniform (`Pouw.pr`, as `Accounting.Joint`'s), and each audit draws
its own coordinate of it (`PeriodRun.own`). The device sees every coin, so the schedule and the committed transcript are
functions of all of them, and so is each verdict.

Each audit's per-period guarantee (`PeriodRun.Sound`) is stated in the shape of `Accounting.Run.ComputeSoundPerRun`:
at any value of the other coins, the probability over the audit's own coins that it accepts while its claim fails on
that run is at most its error. That is the form each protocol's own per-period guarantee is to give for a run that
realizes its audit: `Accounting.Guarantees.ComposePeriod.CircuitWorkPeriod` for PoUW,
`Accounting.Guarantees.Remote.AuditRO` for PoUB, `Pous.Guarantees.ContinuousAudit` for PoUS, the network certifier's
leak over a period of sampled audits for communication, and the sampled audit's live knowledge soundness for integrity.

Two parts of the run are fields until their shared definitions exist: the verdicts (`accepts`), as in
`Accounting.Run`, and the network certifier's claim (`commWithin`), which its period statement is to define.
-/

namespace Accounting

/-- The audits that decide a period, one per protocol. -/
inductive Audit
  /-- sampled proofs: the integrity of the circuit's committed transcript -/
  | sampled
  /-- PoUW: work -/
  | pouw
  /-- PoUS: space -/
  | pous
  /-- PoUB, the bandwidth audit: bits read from device memory -/
  | poub
  /-- the network certifier: communication through the network certifiers -/
  | network
  deriving DecidableEq

instance instFintypeAudit : Fintype Audit where
  elems := {.sampled, .pouw, .pous, .poub, .network}
  complete a := by cases a <;> simp

/-- A period's certificates, one per audit. -/
structure Certificates where
  /-- integrity: at most this share of the circuit's units are wrong -/
  ε : ℝ
  /-- work: at least this much at the period's price (W_cert) -/
  W : ℝ
  /-- space: at least this many bits of device memory held on average (S_cert) -/
  S : ℝ
  /-- bandwidth: at least this many bits read from device memory (Q_cert) -/
  Q : ℝ
  /-- communication: at most this many bits through the network certifiers (K) -/
  K : ℝ

/-- A period of a node's run on device `d` with accounts `α`, over the circuit `C`, on the coins `(i : I) → Ω i`: the
partition whose units the integrity claim counts, each audit's coordinate of the coins, each resource's account, the
period's window, the schedule and the circuit's committed transcript at each value of the coins, the network
certifier's claim, and each audit's verdict. -/
structure PeriodRun {ι : Type} (d : Device ι) (α : Type) [Fintype α] (C : Verity.Circuit) {I : Type}
    (Ω : I → Type) where
  /-- the number of units of the partition -/
  units : ℕ
  /-- the partition of the circuit's computed gates into units -/
  parts : Verity.Partition C units
  /-- the coordinate of the coins an audit draws -/
  own : Audit → I
  /-- the account a resource is charged to -/
  acct : Resource → α
  /-- the period's window -/
  win : Window
  /-- the schedule at the coins -/
  schedule : ((i : I) → Ω i) → Schedule ι α
  /-- every schedule fits the device -/
  fits : ∀ ξ, (schedule ξ).Fits d
  /-- the circuit's transcript the node committed in the period, at the coins -/
  transcript : ((i : I) → Ω i) → C.Transcript
  /-- the network certifier's claim at `K`: what the network certifiers saw the node move in the period is within `K`
  bits -/
  commWithin : ℝ → ((i : I) → Ω i) → Prop
  /-- the verdict of an audit at the coins -/
  accepts : Audit → ((i : I) → Ω i) → Prop

namespace PeriodRun

variable {ι : Type} {d : Device ι} {α : Type} [Fintype α] {C : Verity.Circuit} {I : Type} {Ω : I → Type}
  (r : PeriodRun d α C Ω)

/-- The compute account's work over the period at price `p`. -/
def work (p : Price) (ξ : (i : I) → Ω i) : ℕ := (r.schedule ξ).work r.win (r.acct .compute) p

/-- The device memory the capacity account held over the period, in bit-ticks. -/
def held (ξ : (i : I) → Ω i) : ℕ := (r.schedule ξ).held r.win (r.acct .capacity)

/-- The bits the bandwidth account read from device memory over the period. -/
def read (ξ : (i : I) → Ω i) : ℕ := (r.schedule ξ).read r.win (r.acct .bandwidth)

/-- The claim an audit certifies at certificates `c` and price `p`, on the run at the coins `ξ`. -/
def Holds (c : Certificates) (p : Price) : Audit → ((i : I) → Ω i) → Prop
  | .sampled, ξ => ((r.parts.wrong (r.transcript ξ)).card : ℝ) ≤ c.ε * r.units
  | .pouw, ξ => c.W ≤ (r.work p ξ : ℝ)
  | .pous, ξ => c.S * r.win.len ≤ (r.held ξ : ℝ)
  | .poub, ξ => c.Q ≤ (r.read ξ : ℝ)
  | .network, ξ => r.commWithin c.K ξ

variable [DecidableEq I] [∀ i, Fintype (Ω i)]

/-- **An audit's per-period guarantee, at certificates `c`, price `p` and error `δ`**: at any value of the other
audits' coins, the probability over the audit's own coins that it accepts while its claim fails on that run is at
most `δ`. -/
def Sound (c : Certificates) (p : Price) (a : Audit) (δ : ℝ) : Prop :=
  ∀ ξ : (i : I) → Ω i,
    Pouw.pr (fun x : Ω (r.own a) =>
      r.accepts a (Function.update ξ (r.own a) x) ∧ ¬ r.Holds c p a (Function.update ξ (r.own a) x)) ≤ δ

end PeriodRun

end Accounting
