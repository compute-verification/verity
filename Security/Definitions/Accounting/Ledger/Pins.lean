import Definitions.Accounting.Ledger.Pin

/-!
# Every protocol's guarantee as a pin (trusted)

Each protocol's guarantee, restated as a pin over the ledger: its region, its meter, its bound and its error. These are
statement shapes, not proofs; each docstring names the guarantee it restates and the form that guarantee is proved in
today. The protocol's game enters only through the experiment: a guarantee gives its pin for an experiment whose
adversaries and verdict are the game's and whose actual circuit's meter covers the game's charge, as the device model's
realizations do (`Accounting.RealizesPouw`, `Accounting.RealizesBandwidth`, `Accounting.RealizesState`).

| Protocol | Pin | Meter | Region | Form proved today |
| --- | --- | --- | --- | --- |
| Proof of useful work | `PouwWork` | work at a price | the part that answers, after the seed | budget (`Pouw.Guarantees.EndToEnd`) |
| Proof of useful space | `PousHeldPeak` | bits held | the device at each challenge tick | budget, peak-graded (`Pous.Meets`) |
| Proof of useful space, graded | `PousHeldGraded` | bits held | the device over the challenge ticks | open (lane `ra-pous-graded`) |
| Bandwidth audit | `BandwidthRead` | crossing outward | device memory, one window | per run (`Bandwidth.Guarantees.AuditRO`) |
| Rate audit | `RateRead`, `RebuildGates` | crossing outward; work | device memory and the cores, one window | open (lane `ra-rate-audit-lean`) |
| Network certifier | `CertifierContent` | undeclared content | the unit's boundary, `m` windows | per run (`NetTiming.Guarantees.EncardDecodableLeConstantRatePerWindow`) |
-/

namespace Accounting.Ledger

open Experiment

variable {Λ G Adv Ω : Type} [Fintype Λ] [DecidableEq Λ] [Fintype G] [Fintype Ω] (E : Experiment Λ G Adv Ω)

/-- **PoUW's work pin** (`Pouw.Guarantees.EndToEnd`; on the device model, `Accounting.Guarantees.ComputeOfPouw`). Work
at price `p` (PoUW's cost unit: RTX 4090 multiply-adds and CUDA-core integer ops) in region `ρ` is at least
`(1 − γ)(1 − ε_s)·W`, except with error `err` (`(1 − ε_s)^t + ε(q, N)`), in budget form. `ρ` is the part whose gates
the transcript depends on, over the ticks after the seed: PoUW alone says nothing about where the work ran, so pinning
it to one device takes identity and the deadline (latency). -/
def PouwWork (ρ : Region Λ) (p : G → ℕ) (W γ εs err : ℝ) : Prop :=
  E.PinBudget ρ (Meter.priced p) ((1 - γ) * (1 - εs) * W) err

section Held

variable [DecidableEq G]

/-- **PoUS's bits-held pin, peak-graded** (`Pous.Meets`; on the device model, `Accounting.RealizesState` and
`Accounting.Guarantees.CapacityOfStateBound`). Against every adversary whose part `dev` holds at most `B` bits at
every challenge tick `c ∈ C` on every coin, the audit accepts with `B < S` with probability at most `err`: more than `S`
bits are held at some challenge. When the game bounds a server's state by `S₀` and counts the device's inward
crossing over a round as state, `S` is `S₀` less that crossing's capacity. -/
def PousHeldPeak (dev : Finset Λ) (C : Finset ℕ) (S err : ℝ) : Prop :=
  ∀ a (B : ℝ), 0 ≤ B → (∀ ω, ∀ c ∈ C, (E.read ⟨dev, {c}⟩ (Meter.entry .held) a ω : ℝ) ≤ B) →
    Pouw.pr (fun ω => E.accepts a ω ∧ B < S) ≤ err

/-- **PoUS's bits-held pin, graded at each challenge**: part `dev` holds at least `S` bits on average over the
challenge ticks `C`, `S·|C|` bit-ticks, except with error `err`. This is the sustained-width pin a continuous audit
gives; it implies the peak-graded pin (`Accounting.Ledger.pousHeldPeak_of_graded`). -/
def PousHeldGraded (dev : Finset Λ) (C : Finset ℕ) (S err : ℝ) : Prop :=
  E.PinBudget ⟨dev, C⟩ (Meter.entry .held) (S * C.card) err

/-- **The bandwidth audit's crossing pin** (`Bandwidth.Guarantees.AuditRO`; on the device model,
`Accounting.Guarantees.BandwidthOfAudit`). Device memory's crossing outward over window `w` (T + RTT) is at least
`k·b − S − c − G·b − λ` store bits, except with error `err` (`((k − 1)/N)^s + s/2^h + C(q, k)/2^λ`), per run: `S` is
the on-chip state at the seed, `c` the side channels' capacity over the window and `G` the regenerations. -/
def BandwidthRead (mem : Finset Λ) (w : Window) (k b S c G lam : ℕ) (err : ℝ) : Prop :=
  E.Pin (Region.ofWindow mem w) (Meter.entry .crossOut)
    ((k * b : ℕ) - (S : ℝ) - (c : ℝ) - ((G * b : ℕ) : ℝ) - (lam : ℝ)) err

/-- **The rate audit's crossing pin** (lane `ra-rate-audit-lean`). The audit tags every word of a store of `nE` bits,
`b` bits a word, under a fresh seed in window `w`. A word the device doesn't read from device memory it rebuilds
(`rebuilt a ω` words). Device memory's crossing outward over `w`, plus `b` bits for each word rebuilt, is at least
`(1 − μ)·nE`, except with error `err`, per run; `μ` is the share the chip's state, the side channels and the sample
leave open. What rebuilding costs is the requirement `RebuildGates`, and the claim that bounds the rebuilt words by the
gates left over is `Accounting.Ledger.rate_claim`. -/
def RateRead (mem : Finset Λ) (w : Window) (rebuilt : Adv → Ω → ℕ) (b nE : ℕ) (μ err : ℝ) : Prop :=
  ∀ a, Pouw.pr (fun ω => E.accepts a ω ∧
    (E.read (Region.ofWindow mem w) (Meter.entry .crossOut) a ω : ℝ) + b * rebuilt a ω < (1 - μ) * nE) ≤ err

end Held

/-- **Rebuilding is a gates requirement** (the codec's work-hardness: in the generic-ring model its inverse costs at
least its addition-chain length per word, with no amortization across words). An actual circuit that rebuilds at least
`k` words does at least `k·c` work at price `p` on part `cores` over window `w`. -/
def RebuildGates (cores : Finset Λ) (w : Window) (rebuilt : Adv → Ω → ℕ) (p : G → ℕ) (c : ℝ) : Prop :=
  ∀ k : ℕ, E.Requires (fun a ω => k ≤ rebuilt a ω) (Region.ofWindow cores w) (Meter.priced p) (k * c)

/-- **The network certifier's content pin** (`NetTiming.Guarantees.EncardDecodableLeConstantRatePerWindow`). Over a
run's first `m` windows of `T` buckets, a covert sender inside the unit moves at most
`A_c + m·(log₂ #Σ_sync + T·log₂ D)` bits through the unit's boundary outward: `nAc` content-advice values, `nSync`
clock-sync values per window and jitter below `D`. The console's table adds `b` bits; the Lean statement has no such
term. -/
def CertifierContent {Z W : Type} (accepted : Z → Prop) (wire : Z → W) (nAc nSync D T m : ℕ) : Prop :=
  ContentAtMost accepted wire ((nAc * (nSync * D ^ T) ^ m : ℕ) : ℕ∞)

end Accounting.Ledger
