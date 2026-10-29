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

open Pous (Bits pr)
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

/-- **The bandwidth audit v0 with interactive side sources, in the random-oracle model.** `AuditRelay`'s bound for an
`InteractiveDevice`, whose chip may send bits and whose side sources answer each read as any function of the store,
the oracle and every bit sent so far. `c` bounds only the side bits the chip read, and sends are free: however many
bits the chip sends, and however they depend on the seed, the deficit is `S + c + m·b + G·b + λ`, as for a relay
device.

In the actual-circuit model, the rest of the prover answers the cores as such a source. So the bandwidth audit's help
term is the side bits the cores read in the window, `β_side·(T + RTT)`, not `2^e` times it for `e` seed-dependent bits
sent. Proof: `Bandwidth.SecurityProofs.AuditInteractive`, from `AuditRelay`: such a device is a relay device whose side
source is addressed by what was sent and where.

Frozen (no-function, hash): a bound in the random-oracle model over an abstract chip program, which no Lean function of
the bandwidth audit's code computes. -/
def AuditInteractive : Prop :=
  ∀ (N b h S c G q s k m lam : ℕ) (A : InteractiveDevice N b h S c G q),
    pr (fun x : Store N b × Oracle N b h × (Fin s → Fin N) =>
        A.Accepts x.1 x.2.1 x.2.2 ∧ A.ingress x.1 x.2.1 + S + c + m * b + G * b + lam < k * b)
      ≤ (((k - 1 : ℕ) : ℝ≥0∞) / N) ^ s + (N.choose m : ℝ≥0∞) * 2 ^ c / 2 ^ (h * m)
        + ((q.choose (k - m) : ℕ) : ℝ≥0∞) / 2 ^ lam

/-- **The bandwidth audit v0 with interactive side sources that ignore the oracle.** `AuditRO`'s bound for an
`InteractiveDevice` whose side sources' answers don't depend on the oracle. They may still depend on everything the
chip sent, including tags it computed. No relayed-tag term: the deficit is `S + c + G·b + λ` at `AuditRO`'s error,
with `c` the side bits read. Proof: `Bandwidth.SecurityProofs.AuditInteractiveRO`, from `AuditRO`.

Frozen (no-function, hash): as `AuditInteractive`. -/
def AuditInteractiveRO : Prop :=
  ∀ (N b h S c G q s k lam : ℕ) (A : InteractiveDevice N b h S c G q),
    (∀ E O O', A.side E O = A.side E O') →
    pr (fun x : Store N b × Oracle N b h × (Fin s → Fin N) =>
        A.Accepts x.1 x.2.1 x.2.2 ∧ A.ingress x.1 x.2.1 + S + c + G * b + lam < k * b)
      ≤ (((k - 1 : ℕ) : ℝ≥0∞) / N) ^ s + (s : ℝ≥0∞) / 2 ^ h + (q.choose k : ℝ≥0∞) / 2 ^ lam

/-- **Bandwidth adds, by counting (Mbw), in the random-oracle model.** For every `JointDevice` whose chip's view
determines `z` (`Recovers`), over a uniform store of `N` blocks of `b` bits, `z` uniform over `n` bits, a fresh oracle
and `s` sampled blocks, all independent (`z` independent of the oracle too: the oracle's answers at points that don't
depend on the store would be a `z` the chip's view determines with no reads): the probability that the verifier accepts
while the device's ingress, plus `S + c + G·b + λ`, falls short of `k·b + n` is at most `AuditRO`'s bound

  `((k − 1)/N)^s + s/2^h + C(q, k)/2^λ`.

Read with `cert_bw = k·b − S − c − G·b − λ`, the bits of ingress `AuditRO` certifies: an accept gives
`n ≤ ingress − cert_bw` except with `AuditRO`'s error and no extra slack, so whatever else the chip's view must
determine in the window, it reads on top of what the audit certifies, up to what it held or got another way. The
proof is `AuditRO`'s, with the compression run over the pair (store, `z`): the chip's state, the bits it read and which
`k` of its queries named blocks determine those `k` blocks and `z`, `n` bits more than the store's alone. This is the
actual-circuit model's Mbw for a device of this game; for the chip as a tick transducer whose view in the window
determines `z`, it is `Bandwidth.SecurityProofs.Ticked.audit_joint`, and for the chip of a strategy causal from every
tick, `Bandwidth.SecurityProofs.Causal.audit_joint`. Proof: `Bandwidth.SecurityProofs.AuditJoint`.

Frozen (no-function, hash): as `AuditRO`. -/
def AuditJoint : Prop :=
  ∀ (N b h S c G q n s k lam : ℕ) (A : JointDevice N b h S c G q n), A.Recovers →
    pr (fun x : (Store N b × Bits n) × Oracle N b h × (Fin s → Fin N) =>
        A.Accepts x.1.1 x.1.2 x.2.1 x.2.2 ∧ A.ingress x.1.1 x.1.2 x.2.1 + S + c + G * b + lam < k * b + n)
      ≤ (((k - 1 : ℕ) : ℝ≥0∞) / N) ^ s + (s : ℝ≥0∞) / 2 ^ h + (q.choose k : ℝ≥0∞) / 2 ^ lam

end Bandwidth.Guarantees
