import Definitions.Accounting.Bandwidth

/-!
# The bandwidth audit v0: what an accepted device read

The game is `Security/Definitions/Accounting/Bandwidth/Game.lean`; proofs are in `Security/Proofs/Accounting/Bandwidth/`.
No named assumption: the device's bounds (on-chip state `S`, side-source bits `c`, regenerations `G`, oracle queries
`q`) define the class of devices the guarantee quantifies over, and the claims registry names, for a deployed device,
the measurement that falsifies each (`bandwidth-s-chip`, `bandwidth-side-cap`, `bandwidth-regen-time`) and what the
model takes on trust (`random-oracle` for the tag, `bandwidth-instantiation` for the stored blocks).
-/

namespace Bandwidth.Guarantees

open Pous (pr)
open scoped ENNReal

/-- **The bandwidth audit v0 in the random-oracle model.** For every device with on-chip state of `S` bits that reads
at most `c` bits of side sources, regenerates at most `G` blocks and makes at most `q` oracle queries in the window,
over a uniform store of `N` blocks of `b` bits, a fresh oracle with `h`-bit tags and `s` sampled blocks: the
probability that the verifier accepts while the device's ingress, plus `S + c + G·b + λ`, falls short of `k` blocks
(`k·b` bits) is at most

  `((k − 1)/N)^s + s/2^h + C(q, k)/2^λ`.

Read with `k = ⌈(1 − ε_s)·N⌉`: an accepted device read at least `(1 − ε_s)·|E| − S − c − G·b − λ` bits of device memory,
except with probability `(1 − ε_s)^s + s·2^-h + C(q, k)·2^-λ`. The three terms are the three ways a sampled block
passes without its contents on chip at a query: the sample misses the blocks the device never queried (`((k − 1)/N)^s`),
a guessed tag (`2^-h` per sampled block), and a device that queried `k` blocks with less information than they hold
(`C(q, k)·2^-λ`, by compression: its state, side bits, regenerated blocks and ingress, plus which `k` of its queries
named them, would describe the store in fewer bits than it has). `λ` is the slack the last term charges; with
`q ≤ 2^40`, `λ = 40·k + 41` makes it at most `2^-41`, 40 bits of 16,448 per deployed block (0.24%).

Parameters, not constants: `c` is every side channel's capacity over the window (`cap · T`), and `G` is the window over
the regeneration time `R_regen` (about 2 ms sequential for a P2 root), times the devices that regenerate. Proof:
`Bandwidth.SecurityProofs.AuditRO`. Its completeness is `HonestPasses`: an honest device is accepted on every store,
oracle and sample with ingress exactly `N·b`.

Frozen (no-function, hash): a bound in the random-oracle model over an abstract chip program, which no Lean function of
the bandwidth audit's code computes. -/
def AuditRO : Prop :=
  ∀ (N b h S c G q s k lam : ℕ) (A : Device N b h S c G q),
    pr (fun x : Store N b × Oracle N b h × (Fin s → Fin N) =>
        Accepts A x.1 x.2.1 x.2.2 ∧ A.ingress x.1 x.2.1 + S + c + G * b + lam < k * b)
      ≤ (((k - 1 : ℕ) : ℝ≥0∞) / N) ^ s + (s : ℝ≥0∞) / 2 ^ h + (q.choose k : ℝ≥0∞) / 2 ^ lam

/-- **Completeness of `AuditRO`: an honest device is accepted, and its ingress is exactly the store.** For every `N`,
`b` and `h` there is a device with no on-chip state, no side sources and no regeneration that makes `N` oracle queries
(a `Device N b h 0 0 0 N`) whose tags are accepted on every store, oracle and sample, and whose ingress is exactly `N·b`
bits. The device reads every block once and queries the oracle at it. Against `AuditRO` at `S = c = G = 0` and `q = N`,
the honest `N·b` sits above the floor `k·b − λ` by the sample's slack `(N − k)·b`, about `ε_s·N·b`, plus `λ`. Proof:
`Bandwidth.SecurityProofs.HonestPasses`. -/
def HonestPasses : Prop :=
  ∀ N b h : ℕ, ∃ A : Device N b h 0 0 0 N,
    ∀ (E : Store N b) (O : Oracle N b h) (s : ℕ) (I : Fin s → Fin N), Accepts A E O I ∧ A.ingress E O = N * b

/-- **The bandwidth audit v0 with relayed tags, in the random-oracle model.** `AuditRO` for a `RelayDevice`, whose side
sources may depend on the oracle: a peer holding a copy of the store can compute tags after the seed and relay them,
and every bit the chip reads from it counts against `c`. For every `m`, the probability that the verifier accepts while
the device's ingress, plus `S + c + m·b + G·b + λ`, falls short of `k·b` is at most

  `((k − 1)/N)^s + C(N, m)·2^c/2^(h·m) + C(q, k − m)/2^λ`.

`m` is the number of blocks charged to the side sources' tags. The middle term bounds the chance that the chip
committed the right tag for `m` blocks it never queried: for a fixed store, the run depends on the oracle off its
queries only through the side bits it read, at most `2^c` strings, and those `m` tags are `h·m` uniform bits. Since
`C(N, m) ≤ (e·N/m)^m`, choosing `m·(h − log₂(e·N/m)) ≥ c + μ` makes it at most `2^-μ`. The side sources then cost
`c + m·b ≈ c·(1 + b/(h − log₂(e·N/m)))` bits of the deficit, `c` as information about the store and `m·b` as relayed
tags, so relay is covered inside the side budget rather than excluded. The other two terms are `AuditRO`'s, with the
compression at `k − m` blocks. Proof: `Bandwidth.SecurityProofs.AuditRelay`.

Frozen (no-function, hash): a bound in the random-oracle model over an abstract chip program, which no Lean function of
the bandwidth audit's code computes. -/
def AuditRelay : Prop :=
  ∀ (N b h S c G q s k m lam : ℕ) (A : RelayDevice N b h S c G q),
    pr (fun x : Store N b × Oracle N b h × (Fin s → Fin N) =>
        A.Accepts x.1 x.2.1 x.2.2 ∧ A.ingress x.1 x.2.1 + S + c + m * b + G * b + lam < k * b)
      ≤ (((k - 1 : ℕ) : ℝ≥0∞) / N) ^ s + (N.choose m : ℝ≥0∞) * 2 ^ c / 2 ^ (h * m)
        + ((q.choose (k - m) : ℕ) : ℝ≥0∞) / 2 ^ lam

/-- **The bandwidth audit v0 over a co-audited set, in the random-oracle model.** `d` devices run simultaneous windows
over pairwise-distinct stores of `N` blocks of `b` bits, and the verifier samples `s` blocks of each store
(`Security/Definitions/Accounting/Bandwidth/Set.lean`). Every device that could hold a copy of a store is in the set or
isolated, so the set's side sources are only its external links. The set is one `Device` over the `d·N` blocks: `S` is
its members' on-chip state summed, `c` the bits its external links deliver, `G` its regenerations summed and `q` its
oracle queries summed. The probability that the verifier accepts every member while the set's ingress, plus
`S + c + G·b + λ`, falls short of `d` stores' `k·b` is at most

  `d·(((k − 1)/N)^s + s/2^h) + C(q, d·k)/2^λ`.

Read with `k = ⌈(1 − ε_s)·N⌉` and `S`, `G`, `λ` the members' `S_i`, `G_i`, `λ_i` summed: the set's ingress is at least
the sum of the members' floors `k·b − S_i − G_i·b − λ_i` (`AuditRO`'s, with no side sources of their own) minus the
external side budget `c`, except with the members' sample and guess terms summed plus the set's compression term. Tags
relayed between members are internal: a member that computes another's tags reads those blocks from device memory, and
the set's ingress counts that read, so the members' ingress adds up although no member's own ingress is bounded alone.
Since `C(q, d·k) ≤ (e·q/(d·k))^(d·k)`, the slack `λ = d·k·log₂(e·q/(d·k)) + μ` makes the last term at most `2^-μ`: per
member, the `k·log₂(e·q/(d·k))` bits one device making the members' mean number of queries would be charged. Proof:
`Bandwidth.SecurityProofs.AuditSet`. Its completeness is `SetHonestPasses`: an honest set is accepted on every store,
oracle and sample with ingress exactly `d·N·b`.

Frozen (no-function, hash): a bound in the random-oracle model over an abstract chip program, which no Lean function of
the bandwidth audit's code computes. -/
def AuditSet : Prop :=
  ∀ (d N b h S c G q s k lam : ℕ) (A : Device (d * N) b h S c G q),
    pr (fun x : Store (d * N) b × Oracle (d * N) b h × (Fin d → Fin s → Fin N) =>
        SetAccepts A x.1 x.2.1 x.2.2 ∧ A.ingress x.1 x.2.1 + S + c + G * b + lam < d * (k * b))
      ≤ (d : ℝ≥0∞) * ((((k - 1 : ℕ) : ℝ≥0∞) / N) ^ s + (s : ℝ≥0∞) / 2 ^ h)
        + ((q.choose (d * k) : ℕ) : ℝ≥0∞) / 2 ^ lam

/-- **Completeness of `AuditSet`: an honest set is accepted, and its ingress is exactly its stores.** For every `d`,
`N`, `b` and `h` there is a set of `d` devices with no on-chip state, no side sources and no regeneration that makes
`d·N` oracle queries (a `Device (d * N) b h 0 0 0 (d * N)`), whose tags are accepted for every member on every store,
oracle and samples, and whose ingress is exactly `d·N·b` bits: every member reads its own store once, from its own
device memory, and queries the oracle at each block. Against `AuditSet` at `S = c = G = 0` and `q = d·N`, the honest
`d·N·b` sits above the floor `d·k·b − λ` by the samples' slack `d·(N − k)·b` plus `λ`. Proof:
`Bandwidth.SecurityProofs.SetHonestPasses`. -/
def SetHonestPasses : Prop :=
  ∀ d N b h : ℕ, ∃ A : Device (d * N) b h 0 0 0 (d * N),
    ∀ (E : Store (d * N) b) (O : Oracle (d * N) b h) (s : ℕ) (I : Fin d → Fin s → Fin N),
      SetAccepts A E O I ∧ A.ingress E O = d * (N * b)

end Bandwidth.Guarantees
