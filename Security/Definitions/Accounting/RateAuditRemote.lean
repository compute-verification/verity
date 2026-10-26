import Definitions.Accounting.RateAudit
import Definitions.Accounting.RateAuditInteractive

/-!
# The remote rate audit: a store over tiers, and chained passes (trusted)

Two additions to the rate audit's game (`RateAudit.lean`, `RateAuditInteractive.lean`) for the remote protocol, where
no prover hardware is trusted, the store spans the node's memory tiers, and the verifier sends one challenge a window
and one live coin a pass.

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

**Chained passes.** A window runs `P` passes, each one read of the store with a receipt per word. Under receipt format
`verity/remote-bandwidth/seed/v0`, pass `j`'s seed is `H(σ_t, j, root_{j−1})`: its oracle is the family `H(σ_t, j, ·, ·, ·, ·)`, a uniform function of the root and the
receipt's inputs, evaluated at the root the prover committed for pass `j − 1` (the Merkle root of its receipts; any
root for pass 0). The pass index in the seed separates the passes' oracles, so equal roots in two passes give
independent oracles; the design's seed `H(σ_t, root_{j−1})` has no index. The pass boundary is where `root_{j−1}` is
fixed: from there on the chip runs pass `j` from what it holds then, at most `S` bits on chip (`bandwidth-s-chip` at
every boundary), and device memory as laid out. So pass `j`'s device (its state, device memory, side sources and chip
program) and the root it is seeded at are any function of the store and of the earlier passes' coins (their streams,
their oracle families at every root and their samples), and of nothing of its own pass or later. `Chain.oracle` is pass
`j`'s oracle at that root. Each pass draws its own sample of `s` words after its output.

**Grinding.** A seed derived from the prover's root is Fiat–Shamir (`fiat-shamir`), so a prover may hash candidate roots
with pass `j`'s seed before it commits one, and keep the root whose oracle it likes. `GrindChain` gives it `G`
candidates, fixed by the earlier passes' coins and the store, and lets it pick one as any function of pass `j`'s whole
oracle family; pass `j`'s device may depend on the pick. Keeping one candidate at every pass is a `Chain`
(`GrindChain.atCand`), so grinding costs at most a factor `G` in each pass's error. The protocol's seeds were derived
so up to receipt format `verity/remote-bandwidth/seed/v0`, which the session now refuses.

**Live pass coins.** From `verity/remote-bandwidth/seed/v1` on, pass `j`'s seed is the challenger's live coin `c_j`
(512 bits, `remote_bandwidth.coins`' family `pass`), which it releases only after `root_{j−1}` arrives: the root fixes
nothing of pass `j`'s oracle, so there is no root to pick and no candidate to grind. `LiveChain` is that prover: pass
`j`'s device is any function of the earlier passes' coins, their live coins included, and pass `j`'s coins are its
live coin and `RateAudit`'s game's, a stream, an oracle and a sample, uniform and independent of it. At each value of
the live coins it is a `GrindChain` at `G = 1` over roots of no bits. The oracle at a coin is the hash keyed by it,
`H(c_j, ·)` (`atSeeds`); reading those as independent uniform oracles is the one random-oracle step left
(`Accounting.Assumptions.LiveSeedRandom`).
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

/-- A chained prover that grinds its roots: pass `j`'s root is one of `G` candidates, which the earlier passes' coins
and the store fix, picked as any function of pass `j`'s whole oracle family; pass `j`'s device may depend on the pick
too. -/
structure GrindChain (P N b h ρ S L q r s G : ℕ) where
  device : (j : Fin P) → ({i : Fin P // i < j} → PassCoins N b h ρ r s) → Fin G → InteractiveDevice N b h ρ S L q
  cand : (j : Fin P) → ({i : Fin P // i < j} → PassCoins N b h ρ r s) → Store N b → Fin G → Bits r
  pick : (j : Fin P) → ({i : Fin P // i < j} → PassCoins N b h ρ r s) → Store N b → (Bits r → Oracle N b h ρ) →
    Fin G

namespace GrindChain

variable {P N b h ρ S L q r s G : ℕ}

/-- The candidate pass `j` keeps at the window's coins `ω`. -/
def kept (A : GrindChain P N b h ρ S L q r s G) (j : Fin P) (E : Store N b) (ω : Fin P → PassCoins N b h ρ r s) :
    Fin G :=
  A.pick j (fun i => ω i.1) E (ω j).2.1

/-- Pass `j`'s device at the window's coins `ω`, at the candidate it kept. -/
def dev (A : GrindChain P N b h ρ S L q r s G) (j : Fin P) (E : Store N b) (ω : Fin P → PassCoins N b h ρ r s) :
    InteractiveDevice N b h ρ S L q :=
  A.device j (fun i => ω i.1) (A.kept j E ω)

/-- Pass `j`'s oracle: its family at the kept candidate root. -/
def oracle (A : GrindChain P N b h ρ S L q r s G) (j : Fin P) (E : Store N b) (ω : Fin P → PassCoins N b h ρ r s) :
    Oracle N b h ρ :=
  (ω j).2.1 (A.cand j (fun i => ω i.1) E (A.kept j E ω))

/-- The chained prover that keeps candidate `c` at every pass. -/
def atCand (A : GrindChain P N b h ρ S L q r s G) (c : Fin G) : Chain P N b h ρ S L q r s where
  device j η := A.device j η c
  root j η E := A.cand j η E c

end GrindChain

/-- A pass's coins in a window with live pass coins: the challenger's coin `c_j` of `κ` bits, which the prover sees,
and `RateAudit`'s game's coins, its certified stream, its oracle and its sample. -/
abbrev LiveCoins (N b h ρ κ s : ℕ) : Type := Bits κ × Coins N b h ρ s

/-- The live pass coins are finite, as a product: an instance of its own, as for `Coins`. -/
instance instFintypeLiveCoins (N b h ρ κ s : ℕ) : Fintype (LiveCoins N b h ρ κ s) :=
  inferInstanceAs (Fintype (_ × _))

/-- A chained prover over `P` passes with live pass coins: pass `j`'s device, an interactive device of `RateAudit`'s
class, any function of the earlier passes' coins (their live coins, streams, oracles and samples). Pass `j`'s own
coins are drawn after `root_{j−1}`, so nothing of its device depends on them. -/
structure LiveChain (P N b h ρ κ S L q s : ℕ) where
  device : (j : Fin P) → ({i : Fin P // i < j} → LiveCoins N b h ρ κ s) → InteractiveDevice N b h ρ S L q

namespace LiveChain

variable {P N b h ρ κ S L q s : ℕ}

/-- Pass `j`'s device at the window's coins `ω`: it reads the earlier passes' coins. -/
def dev (A : LiveChain P N b h ρ κ S L q s) (j : Fin P) (ω : Fin P → LiveCoins N b h ρ κ s) :
    InteractiveDevice N b h ρ S L q :=
  A.device j (fun i => ω i.1)

end LiveChain

/-- A pass's coins at the deployed seed: the challenger's live coin of `κ` bits, the certified stream, and the
sample. -/
abbrev SeedCoins (N ρ κ s : ℕ) : Type := Bits κ × Bits ρ × (Fin s → Fin N)

/-- The seed coins are finite, as a product: an instance of its own, as for `Coins`. -/
instance instFintypeSeedCoins (N ρ κ s : ℕ) : Fintype (SeedCoins N ρ κ s) :=
  inferInstanceAs (Fintype (_ × _ × _))

/-- The window's pass coins a hash gives at the live coins: pass `j`'s oracle is the hash keyed by its coin,
`H(c_j, ·)`. -/
def atSeeds {P N b h ρ κ s : ℕ} (H : Bits κ → Oracle N b h ρ) (ω : Fin P → SeedCoins N ρ κ s) :
    Fin P → LiveCoins N b h ρ κ s :=
  fun j => ((ω j).1, (ω j).2.1, H (ω j).1, (ω j).2.2)

end Accounting.Rate
