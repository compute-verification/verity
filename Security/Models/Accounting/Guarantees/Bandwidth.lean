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

end Bandwidth.Guarantees
