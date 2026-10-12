import Definitions.Accounting.RateAudit
import Definitions.Accounting.RateAuditInteractive

/-!
# The remote rate audit: a store over tiers, and chained passes (trusted)

Two additions to the rate audit's game (`RateAudit.lean`, `RateAuditInteractive.lean`) for the remote protocol, where
no prover hardware is trusted, the store spans the node's memory tiers, and the verifier sends one challenge a window.

**Tiers.** The node's memory is one address space, device memory as `RateAudit.lean` lays it out before the seed, split
into tiers by a map `tier : ℕ → Fin τ` from addresses (each GPU's HBM, host DRAM: any partition). The chip is everything
on the node that computes (its GPUs and its host CPUs together), so a receipt the node computed is the chip's wherever
on the node it was computed, and a read of any tier is a read of device memory: host DRAM read over PCIe is not side
traffic. `IChip.tierBits` counts one tier's reads. A tier's read capacity over the window, `cap t = R_t·T` in bits,
bounds the chip's reads of it in every world (`InteractiveDevice.TierCapped`). The side link `L` is then what crosses
the node's boundary (its NIC) in the window and a round trip, and the side sources are everything beyond it.

**Correct receipts.** A receipt is `RateAudit.lean`'s tag. `InteractiveDevice.correct` is the set of words whose
committed receipt is the oracle's, `H(σ, R, j, E_j)`, however it was produced: queried by the chip, relayed in by a side
source that sees the oracle, or guessed. The verifier never sees this count; the guarantees bound it directly, with no
sample.

**Chained passes.** A window runs `P` passes, each one read of the store with a receipt per word. Pass `j`'s seed is
`H(σ_t, j, root_{j−1})`: its oracle is the family `H(σ_t, j, ·, ·, ·, ·)`, a uniform function of the root and the
receipt's inputs, evaluated at the root the prover committed for pass `j − 1` (the Merkle root of its receipts; any
root for pass 0). The pass index in the seed separates the passes' oracles, so equal roots in two passes give
independent oracles; the design's seed `H(σ_t, root_{j−1})` has no index. The pass boundary is where `root_{j−1}` is
fixed: from there on the chip runs pass `j` from what it holds then, at most `S` bits on chip (`bandwidth-s-chip` at
every boundary), and device memory as laid out. So pass `j`'s device (its state, device memory, side sources and chip
program) and the root it is seeded at are any function of the store and of the earlier passes' coins (their streams,
their oracle families at every root and their samples), and of nothing of its own pass or later. `Chain.oracle` is pass
`j`'s oracle at that root. Each pass draws its own sample of `s` words after its output.

**Live coins.** A challenger that streams a fresh coin for each pass on a fixed schedule, in place of the hash-chained
seed, gives pass `j` an oracle that is fresh whatever the prover committed: that is `Chain` at `r = 0`, whose one root
makes each pass's family a single uniform oracle. Its pass boundary is coin `j`'s release, and the challenger must
refuse a `root_{j−1}` that arrives later than that release plus the least round trip, so that pass `j − 1`'s receipts
were fixed before coin `j` could be seen, as `Chain` has them.
-/

namespace Accounting.Rate

open Pous (Bits)

namespace IChip

variable {N b h ρ : ℕ} {α : Type}

/-- The bits read from tier `t` of device memory in a world, having sent `v` so far: one per `ingress` step at an
address of tier `t`. -/
def tierBits {τ : ℕ} (tier : ℕ → Fin τ) (t : Fin τ) (w : IWorld N b h ρ) : List Bool → IChip N b h ρ α → ℕ
  | _, done _ => 0
  | v, tag x k => (k (w.oracle x)).tierBits tier t w v
  | v, ingress a k => (k (w.mem a)).tierBits tier t w v + if tier a = t then 1 else 0
  | v, side a k => (k (w.side v a)).tierBits tier t w v
  | v, stream i k => (k (w.stream i)).tierBits tier t w v
  | v, rebuild j k => (k (w.store j)).tierBits tier t w v
  | v, send β p => p.tierBits tier t w (β :: v)

end IChip

namespace InteractiveDevice

variable {N b h ρ S L q : ℕ}

/-- The device's reads of tier `t` in the window, in bits. -/
def tierReads {τ : ℕ} (A : InteractiveDevice N b h ρ S L q) (tier : ℕ → Fin τ) (t : Fin τ) (E : Store N b)
    (R : Bits ρ) (O : Oracle N b h ρ) : ℕ :=
  (A.chip (A.state E)).tierBits tier t (A.world E R O) []

/-- The words rebuilt in the window, each at `c_enc` of compute. -/
def rebuilt (A : InteractiveDevice N b h ρ S L q) (E : Store N b) (R : Bits ρ) (O : Oracle N b h ρ) : ℕ :=
  (A.chip (A.state E)).rebuilds (A.world E R O) []

/-- The words whose committed receipt is the oracle's under the stream, however the device came by it. -/
def correct (A : InteractiveDevice N b h ρ S L q) (E : Store N b) (R : Bits ρ) (O : Oracle N b h ρ) :
    Finset (Fin N) :=
  Finset.univ.filter (fun j => A.tags E R O j = O (R, j, E j))

/-- Each tier's read capacity over the window bounds the chip's reads of it, from every state and in every world. -/
def TierCapped {τ : ℕ} (A : InteractiveDevice N b h ρ S L q) (tier : ℕ → Fin τ) (cap : Fin τ → ℕ) : Prop :=
  ∀ (st : Bits S) (w : IWorld N b h ρ) (t : Fin τ), (A.chip st).tierBits tier t w [] ≤ cap t

end InteractiveDevice

/-- A pass's coins in a chained window: its certified stream, its oracle family `H(σ_t, j, ·, ·)` at every root of `r`
bits, and its sample. -/
abbrev PassCoins (N b h ρ r s : ℕ) : Type := Bits ρ × (Bits r → Oracle N b h ρ) × (Fin s → Fin N)

/-- The pass coins are finite, as a product: an instance of its own, as for `Coins`. -/
instance instFintypePassCoins (N b h ρ r s : ℕ) : Fintype (PassCoins N b h ρ r s) :=
  inferInstanceAs (Fintype (_ × _ × _))

/-- A chained prover over `P` passes with `r`-bit roots: pass `j`'s device, an interactive device of `RateAudit`'s
class, and the root its seed is drawn at, each any function of the earlier passes' coins (the root also of the
store). -/
structure Chain (P N b h ρ S L q r s : ℕ) where
  device : (j : Fin P) → ({i : Fin P // i < j} → PassCoins N b h ρ r s) → InteractiveDevice N b h ρ S L q
  root : (j : Fin P) → ({i : Fin P // i < j} → PassCoins N b h ρ r s) → Store N b → Bits r

namespace Chain

variable {P N b h ρ S L q r s : ℕ}

/-- Pass `j`'s device at the window's coins `ω`: it reads the earlier passes' coins. -/
def dev (A : Chain P N b h ρ S L q r s) (j : Fin P) (ω : Fin P → PassCoins N b h ρ r s) :
    InteractiveDevice N b h ρ S L q :=
  A.device j (fun i => ω i.1)

/-- Pass `j`'s oracle: its family at the root its seed is drawn at, `H(σ_t, j, root_{j−1}, ·)`. -/
def oracle (A : Chain P N b h ρ S L q r s) (j : Fin P) (E : Store N b) (ω : Fin P → PassCoins N b h ρ r s) :
    Oracle N b h ρ :=
  (ω j).2.1 (A.root j (fun i => ω i.1) E)

end Chain

end Accounting.Rate
