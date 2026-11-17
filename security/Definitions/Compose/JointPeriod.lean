module

public import Definitions.Compose.Joint
public import Definitions.Core.Partition

/-! A period audited by every protocol, and what each audit's guarantee says about it; the verdicts and the network
attester's claim are fields, since no shared definition of them exists. -/

@[expose] public section

namespace ResourceAccounting

/-- The audits that decide a period, one per protocol. -/
inductive Audit
  /-- SPOT: the integrity of the circuit's committed trace -/
  | sampled
  | pouw
  | pous
  | poub
  | network
  deriving DecidableEq

instance instFintypeAudit : Fintype Audit where
  elems := {.sampled, .pouw, .pous, .poub, .network}
  complete a := by cases a <;> simp

/-- A period's certificates, one per audit. -/
structure Certificates where
  /-- integrity: at most this share of the circuit's units are incorrect -/
  ε : ℝ
  /-- work: at least this much at the period's price (W_cert) -/
  W : ℝ
  /-- space: at least this many bits of device memory held on average (S_cert) -/
  S : ℝ
  /-- bandwidth: at least this many bits read from device memory (Q_cert) -/
  Q : ℝ
  /-- communication: at most this many bits through the network attesters (K) -/
  K : ℝ

/-- A period of a node's run on device `d` with accounts `α`, over the circuit `C`, on the coins `(i : I) → Ω i`: the
partition whose units the integrity claim counts, each audit's coordinate of the coins, each resource's account, the
period's window, the schedule and the circuit's committed trace at each value of the coins, the network
attester's claim, and each audit's verdict. -/
structure PeriodRun {ι : Type} (d : Device ι) (α : Type) [Fintype α] (C : Verity.Circuit) {I : Type}
    (Ω : I → Type) where
  units : ℕ
  parts : Verity.Partition C units
  own : Audit → I
  acct : Resource → α
  win : Window
  schedule : ((i : I) → Ω i) → Schedule ι α
  fits : ∀ ξ, (schedule ξ).Fits d
  trace : ((i : I) → Ω i) → C.Trace
  commWithin : ℝ → ((i : I) → Ω i) → Prop
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
  | .sampled, ξ => ((r.parts.incorrect (r.trace ξ)).card : ℝ) ≤ c.ε * r.units
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

end ResourceAccounting
