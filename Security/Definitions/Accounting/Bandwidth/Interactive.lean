import Definitions.Accounting.Bandwidth.Relay

/-!
# The bandwidth audit v0 with interactive side sources

`Relay.lean`'s game, except that the chip may also send bits to the side sources, and a side source answers each read
as a function of the store, the oracle and every bit the chip has sent so far. This is the rest of the prover as the
chip sees it in the actual-circuit model (`note:console/20261012T0645Z-draft-actual-circuit-model`, section 6): a peer,
the host or a network that sees what the chip sends (the seed, partial tags, which blocks it wants) and answers
adaptively. Sends are not bounded, so the chip may send any number `e` of bits that depend on the seed. Every bit the
chip reads from a side source still counts against `c`. Its on-chip state and device memory are still laid out from
the store before the seed, and the sample is still drawn after the output.
-/

namespace Bandwidth

open Pous (Bits)

/-- A chip program in the window that may also send: `Chip`'s steps, and `send β p`, which sends the bit `β` to the side
sources and continues with `p`. -/
inductive IChip (N b h : ℕ) (α : Type) : Type where
  | done : α → IChip N b h α
  | tag : Fin N × Bits b → (Bits h → IChip N b h α) → IChip N b h α
  | ingress : ℕ → (Bool → IChip N b h α) → IChip N b h α
  | side : ℕ → (Bool → IChip N b h α) → IChip N b h α
  | regen : Fin N → (Bits b → IChip N b h α) → IChip N b h α
  | send : Bool → IChip N b h α → IChip N b h α

/-- What answers an interactive chip's steps: `World`'s oracle, device memory and store, and side sources that answer a
read at address `a`, after the chip has sent `v` (newest bit first), with `side v a`. -/
structure IWorld (N b h : ℕ) where
  oracle : Oracle N b h
  mem : ℕ → Bool
  side : List Bool → ℕ → Bool
  block : Store N b

namespace IChip

variable {N b h : ℕ} {α : Type}

/-- The program's value in a world, having sent `v` so far. -/
def run (w : IWorld N b h) : List Bool → IChip N b h α → α
  | _, done a => a
  | v, tag x k => (k (w.oracle x)).run w v
  | v, ingress a k => (k (w.mem a)).run w v
  | v, side a k => (k (w.side v a)).run w v
  | v, regen j k => (k (w.block j)).run w v
  | v, send β p => p.run w (β :: v)

/-- The bits read from device memory in a world, having sent `v` so far: one per `ingress` step. -/
def ingressBits (w : IWorld N b h) : List Bool → IChip N b h α → ℕ
  | _, done _ => 0
  | v, tag x k => (k (w.oracle x)).ingressBits w v
  | v, ingress a k => (k (w.mem a)).ingressBits w v + 1
  | v, side a k => (k (w.side v a)).ingressBits w v
  | v, regen j k => (k (w.block j)).ingressBits w v
  | v, send β p => p.ingressBits w (β :: v)

/-- The bits read from side sources in a world, having sent `v` so far: one per `side` step. Sends are not counted. -/
def sideBits (w : IWorld N b h) : List Bool → IChip N b h α → ℕ
  | _, done _ => 0
  | v, tag x k => (k (w.oracle x)).sideBits w v
  | v, ingress a k => (k (w.mem a)).sideBits w v
  | v, side a k => (k (w.side v a)).sideBits w v + 1
  | v, regen j k => (k (w.block j)).sideBits w v
  | v, send β p => p.sideBits w (β :: v)

/-- The blocks regenerated in a world, having sent `v` so far: one per `regen` step. -/
def regens (w : IWorld N b h) : List Bool → IChip N b h α → ℕ
  | _, done _ => 0
  | v, tag x k => (k (w.oracle x)).regens w v
  | v, ingress a k => (k (w.mem a)).regens w v
  | v, side a k => (k (w.side v a)).regens w v
  | v, regen j k => (k (w.block j)).regens w v + 1
  | v, send β p => p.regens w (β :: v)

/-- The oracle queries in a world, in order, having sent `v` so far. -/
def queries (w : IWorld N b h) : List Bool → IChip N b h α → List (Fin N × Bits b)
  | _, done _ => []
  | v, tag x k => x :: (k (w.oracle x)).queries w v
  | v, ingress a k => (k (w.mem a)).queries w v
  | v, side a k => (k (w.side v a)).queries w v
  | v, regen j k => (k (w.block j)).queries w v
  | v, send β p => p.queries w (β :: v)

end IChip

/-- A device whose side sources answer what its chip sent: on-chip state and device memory laid out from the store
before the seed, side sources that answer as any function of the store, the oracle and the bits sent so far, and a chip
program from the state. In every world, starting from nothing sent, the program reads at most `c` bits of side sources,
regenerates at most `G` blocks and makes at most `q` oracle queries. -/
structure InteractiveDevice (N b h S c G q : ℕ) where
  state : Store N b → Bits S
  mem : Store N b → ℕ → Bool
  side : Store N b → Oracle N b h → List Bool → ℕ → Bool
  chip : Bits S → IChip N b h (Fin N → Bits h)
  bounded : ∀ (st : Bits S) (w : IWorld N b h),
    (chip st).sideBits w [] ≤ c ∧ (chip st).regens w [] ≤ G ∧ ((chip st).queries w []).length ≤ q

namespace InteractiveDevice

variable {N b h S c G q : ℕ}

/-- The world an interactive device's chip runs in, for store `E` and oracle `O`. -/
def world (A : InteractiveDevice N b h S c G q) (E : Store N b) (O : Oracle N b h) : IWorld N b h :=
  ⟨O, A.mem E, A.side E O, E⟩

/-- The tags the device commits to. -/
def tags (A : InteractiveDevice N b h S c G q) (E : Store N b) (O : Oracle N b h) : Fin N → Bits h :=
  (A.chip (A.state E)).run (A.world E O) []

/-- The device's ingress in the window, in bits. -/
def ingress (A : InteractiveDevice N b h S c G q) (E : Store N b) (O : Oracle N b h) : ℕ :=
  (A.chip (A.state E)).ingressBits (A.world E O) []

/-- The verifier accepts: every sampled block's committed tag is the oracle's tag of that block. -/
def Accepts {s : ℕ} (A : InteractiveDevice N b h S c G q) (E : Store N b) (O : Oracle N b h) (I : Fin s → Fin N) :
    Prop :=
  ∀ i, A.tags E O (I i) = O (I i, E (I i))

end InteractiveDevice

end Bandwidth
