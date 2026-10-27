import Definitions.Accounting.JointBridge.BandwidthOfRemoteLinePeriod
import Definitions.Accounting.JointBlocks.Honest

/-!
# PoUB's per-period guarantee, from the remote bandwidth audit's (trusted)

The bridge from the remote bandwidth audit's per-session bound to PoUB's per-period guarantee
(`Accounting.PeriodRun.Sound` at audit `.poub`), the form `Accounting.Guarantees.JointPeriod` takes: in a period whose
PoUB audit is the session (`Accounting.RealizesBandwidthOfRemoteLinePeriod`), if, at every value of the other audits'
coins, the session accepts while its line credits more bits read than the node read with probability at most `δ` over
its own coins, PoUB has its per-period guarantee at `δ`. That bound is to be `Properties.RemoteAudit.Line.LineHeld`'s
bits-read form (MEM-97 item 4): `δs + δ + Σ_c s/2^64 + bind` at `LineHeldAtComponent`'s premises (#4786), the binding
term a SHA-512 collision on the session's receipts.

It replaces the bridge from the bandwidth audit v0's `Bandwidth.Guarantees.AuditRO`, which is in the random-oracle
model and which the joint guarantee no longer takes.

Proved by `Accounting.SecurityProofs.Bridge.BandwidthOfRemoteLinePeriod :
Accounting.Guarantees.Bridge.BandwidthOfRemoteLinePeriod` (`Proofs.Accounting.JointBridge.BandwidthOfRemoteLinePeriod`).
Its honest lemma (`BandwidthOfRemoteLinePeriodHonest`) and nontrivial lemma (`BandwidthOfRemoteLinePeriodNontrivial`)
are stated at `Accounting.Honest`'s instances (`Proofs.Accounting.JointBlocks.Honest`).
-/

namespace Accounting.Guarantees.Bridge

open Accounting

/-- **PoUB's per-period guarantee is the remote bandwidth audit's.** On any device and any period of a node's run over
a circuit on one coin space whose PoUB audit is the remote bandwidth audit's session
(`Accounting.RealizesBandwidthOfRemoteLinePeriod`: PoUB's coordinate of the coins is the session's, its verdict the
session's acceptance, the bandwidth certificate within the line's credit on an accepted period, and the bits the node
read from its registered store within the bandwidth account's reads), if at every value `ξ` of the coins the session
accepts while its line credits more bits read than the node read with probability at most `δ` over its own coins, then
PoUB's per-period guarantee holds at certificates `c`, price `p` and error `δ`: at any value of the other coins, the
probability over PoUB's own coins that it accepts while the bandwidth account read fewer than `Q` bits is at most
`δ`. -/
def BandwidthOfRemoteLinePeriod : Prop :=
  ∀ (ι α I : Type) [Fintype α] [DecidableEq I] (Ω : I → Type) [∀ i, Fintype (Ω i)] (d : Device ι)
    (C : Verity.Circuit) (r : PeriodRun d α C Ω) (c : Certificates) (p : Price) (X : Type) [Fintype X]
    (acc : ((i : I) → Ω i) → X → Prop) (credit readBits : ((i : I) → Ω i) → X → ℝ) (δ : ℝ),
    RealizesBandwidthOfRemoteLinePeriod r c X acc credit readBits →
    (∀ ξ, Pouw.pr (fun x => acc ξ x ∧ readBits ξ x < credit ξ x) ≤ δ) →
    r.Sound c p .poub δ

/-- **`BandwidthOfRemoteLinePeriod`'s honest lemma: its hypotheses hold at the honest deployed session, at error 0.**
At each block of the honest deployed horizon (`Accounting.Honest.honest`) as long as the honest remote bandwidth
session of `r20261026-180902-bc3d` (`Accounting.Honest.sessionSpan` ticks), with the bandwidth certificate the bits that
session credited as read (`Accounting.Honest.sessionCerts`), the period realizes a session that accepts and whose line
credits as read exactly the bits the node read from its registered store, and the session's bound holds at error 0.
Proof: `Accounting.SecurityProofs.Bridge.BandwidthOfRemoteLinePeriodHonest`. -/
def BandwidthOfRemoteLinePeriodHonest : Prop :=
  ∀ (rate : Op → ℕ) (cap : Unit → ℕ) (C : Verity.Circuit) (n : ℕ) (P : Verity.Partition C n) (v : C.Transcript)
    (B E : ℕ) (p : Price) (b : Fin B),
    Nonempty (RealizesBandwidthOfRemoteLinePeriod ((Honest.honest rate cap P v B Honest.sessionSpan E).block b)
      (Honest.sessionCerts rate E p) Bool (fun _ _ => True) (fun _ _ => (Honest.sessionBits : ℝ))
      (fun _ _ => (Honest.sessionBits : ℝ))) ∧
    Pouw.pr (fun _ : Bool => True ∧ (Honest.sessionBits : ℝ) < Honest.sessionBits) ≤ 0

/-- **`BandwidthOfRemoteLinePeriod`'s nontrivial lemma: its conclusion fails for a PoUB verdict that accepts
everything.** At each block of positive length of `Accounting.Honest.idleBlocks`, a node that reads nothing under
verdicts that accept everything, claimed at the honest node's certificates, PoUB's per-period guarantee fails at every
error below 1. Proof: `Accounting.SecurityProofs.Bridge.BandwidthOfRemoteLinePeriodNontrivial`. -/
def BandwidthOfRemoteLinePeriodNontrivial : Prop :=
  ∀ (rate : Op → ℕ) (cap : Unit → ℕ) (C : Verity.Circuit) (n : ℕ) (P : Verity.Partition C n) (v : C.Transcript)
    (B L E : ℕ) (p : Price) (b : Fin B) (δ : ℝ), 0 < L → δ < 1 →
    ¬ ((Honest.idleBlocks rate cap P v B L E).block b).Sound (Honest.certs rate L E p) p .poub δ

end Accounting.Guarantees.Bridge
