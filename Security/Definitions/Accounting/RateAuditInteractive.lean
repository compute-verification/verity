import Definitions.Accounting.RateAudit
import Definitions.Accounting.RateAuditJoint

/-!
# The rate audit with interactive side sources (trusted)

`RateAudit.lean`'s game, except that the chip may also send bits to the side sources, and a side source answers each
read as a function of the store, the certified stream, the oracle and every bit the chip has sent so far.
`RateAudit.lean`'s side source already answers any function of the store and of what the chip sent, through the address
the chip picks; what this game adds is answers that also depend on the certified stream and the oracle. This is the
rest of the prover as the chip sees it in the actual-circuit model: a peer, the host or a network that sees what the chip sends (the seed, partial tags, which words it wants)
and answers adaptively, as the table localization lays out, read bit by bit. Sends are not bounded, so the chip may send
any number `e` of bits that depend on the seed. Every bit the chip reads from a side source still comes in over the
side link and counts against `L`, with the certified stream's `ρ`. Its on-chip state and device memory are still laid
out from the store before the seed and the stream, device memory hears only the chip's reads, rebuilding is still the
codec's inverse at `c_enc` a word, and the sample is still drawn after the output.

`RealizesRateInteractive` is `RealizesRate` (`RateAuditJoint.lean`) for a joint run whose bandwidth audit such a device
plays.
-/

namespace Accounting.Rate

open Pous (Bits)

/-- A chip program in the window that may also send: `Chip`'s steps, and `send β p`, which sends the bit `β` to the side
sources and continues with `p`. -/
inductive IChip (N b h ρ : ℕ) (α : Type) : Type where
  | done : α → IChip N b h ρ α
  | tag : Bits ρ × Fin N × Bits b → (Bits h → IChip N b h ρ α) → IChip N b h ρ α
  | ingress : ℕ → (Bool → IChip N b h ρ α) → IChip N b h ρ α
  | side : ℕ → (Bool → IChip N b h ρ α) → IChip N b h ρ α
  | stream : Fin ρ → (Bool → IChip N b h ρ α) → IChip N b h ρ α
  | rebuild : Fin N → (Bits b → IChip N b h ρ α) → IChip N b h ρ α
  | send : Bool → IChip N b h ρ α → IChip N b h ρ α

/-- What answers an interactive chip's steps: `World`'s oracle, device memory, certified stream and store, and side
sources that answer a read at address `a`, after the chip has sent `v` (newest bit first), with `side v a`. -/
structure IWorld (N b h ρ : ℕ) where
  oracle : Oracle N b h ρ
  mem : ℕ → Bool
  side : List Bool → ℕ → Bool
  stream : Bits ρ
  store : Store N b

namespace IChip

variable {N b h ρ : ℕ} {α : Type}

/-- The program's value in a world, having sent `v` so far. -/
def run (w : IWorld N b h ρ) : List Bool → IChip N b h ρ α → α
  | _, done a => a
  | v, tag x k => (k (w.oracle x)).run w v
  | v, ingress a k => (k (w.mem a)).run w v
  | v, side a k => (k (w.side v a)).run w v
  | v, stream i k => (k (w.stream i)).run w v
  | v, rebuild j k => (k (w.store j)).run w v
  | v, send β p => p.run w (β :: v)

/-- The bits read from device memory in a world, having sent `v` so far: one per `ingress` step. -/
def ingressBits (w : IWorld N b h ρ) : List Bool → IChip N b h ρ α → ℕ
  | _, done _ => 0
  | v, tag x k => (k (w.oracle x)).ingressBits w v
  | v, ingress a k => (k (w.mem a)).ingressBits w v + 1
  | v, side a k => (k (w.side v a)).ingressBits w v
  | v, stream i k => (k (w.stream i)).ingressBits w v
  | v, rebuild j k => (k (w.store j)).ingressBits w v
  | v, send β p => p.ingressBits w (β :: v)

/-- The device-memory addresses read in a world, having sent `v` so far. -/
def addrs (w : IWorld N b h ρ) : List Bool → IChip N b h ρ α → Finset ℕ
  | _, done _ => ∅
  | v, tag x k => (k (w.oracle x)).addrs w v
  | v, ingress a k => insert a ((k (w.mem a)).addrs w v)
  | v, side a k => (k (w.side v a)).addrs w v
  | v, stream i k => (k (w.stream i)).addrs w v
  | v, rebuild j k => (k (w.store j)).addrs w v
  | v, send β p => p.addrs w (β :: v)

/-- The bits that came in over the side link in a world, having sent `v` so far: one per `side` step and one per
`stream` step. Sends are not counted. -/
def linkBits (w : IWorld N b h ρ) : List Bool → IChip N b h ρ α → ℕ
  | _, done _ => 0
  | v, tag x k => (k (w.oracle x)).linkBits w v
  | v, ingress a k => (k (w.mem a)).linkBits w v
  | v, side a k => (k (w.side v a)).linkBits w v + 1
  | v, stream i k => (k (w.stream i)).linkBits w v + 1
  | v, rebuild j k => (k (w.store j)).linkBits w v
  | v, send β p => p.linkBits w (β :: v)

/-- The words rebuilt in a world, having sent `v` so far: one per `rebuild` step. -/
def rebuilds (w : IWorld N b h ρ) : List Bool → IChip N b h ρ α → ℕ
  | _, done _ => 0
  | v, tag x k => (k (w.oracle x)).rebuilds w v
  | v, ingress a k => (k (w.mem a)).rebuilds w v
  | v, side a k => (k (w.side v a)).rebuilds w v
  | v, stream i k => (k (w.stream i)).rebuilds w v
  | v, rebuild j k => (k (w.store j)).rebuilds w v + 1
  | v, send β p => p.rebuilds w (β :: v)

/-- The oracle queries in a world, in order, having sent `v` so far. -/
def queries (w : IWorld N b h ρ) : List Bool → IChip N b h ρ α → List (Bits ρ × Fin N × Bits b)
  | _, done _ => []
  | v, tag x k => x :: (k (w.oracle x)).queries w v
  | v, ingress a k => (k (w.mem a)).queries w v
  | v, side a k => (k (w.side v a)).queries w v
  | v, stream i k => (k (w.stream i)).queries w v
  | v, rebuild j k => (k (w.store j)).queries w v
  | v, send β p => p.queries w (β :: v)

end IChip

/-- A device whose side sources answer what its chip sent: on-chip state (`S` bits) and device memory laid out from the
store before the seed and the stream, side sources that answer as any function of the store, the certified stream, the
oracle and the bits sent so far, and a chip program from the state. In every world, starting from nothing sent, the
program takes in at most `L` bits over the side link (side sources and the certified stream together) and makes at most
`q` oracle queries. -/
structure InteractiveDevice (N b h ρ S L q : ℕ) where
  state : Store N b → Bits S
  mem : Store N b → ℕ → Bool
  side : Store N b → Bits ρ → Oracle N b h ρ → List Bool → ℕ → Bool
  chip : Bits S → IChip N b h ρ (Fin N → Bits h)
  bounded : ∀ (st : Bits S) (w : IWorld N b h ρ), (chip st).linkBits w [] ≤ L ∧ ((chip st).queries w []).length ≤ q

namespace InteractiveDevice

variable {N b h ρ S L q : ℕ}

/-- The world an interactive device's chip runs in, for store `E`, certified stream `R` and oracle `O`. -/
def world (A : InteractiveDevice N b h ρ S L q) (E : Store N b) (R : Bits ρ) (O : Oracle N b h ρ) : IWorld N b h ρ :=
  ⟨O, A.mem E, A.side E R O, R, E⟩

/-- The tags the device commits to. -/
def tags (A : InteractiveDevice N b h ρ S L q) (E : Store N b) (R : Bits ρ) (O : Oracle N b h ρ) : Fin N → Bits h :=
  (A.chip (A.state E)).run (A.world E R O) []

/-- The device's ingress in the window: every bit it read from device memory, in bits. -/
def ingress (A : InteractiveDevice N b h ρ S L q) (E : Store N b) (R : Bits ρ) (O : Oracle N b h ρ) : ℕ :=
  (A.chip (A.state E)).ingressBits (A.world E R O) []

/-- The device's footprint in the window: the distinct device-memory addresses it read, in bits. -/
def footprint (A : InteractiveDevice N b h ρ S L q) (E : Store N b) (R : Bits ρ) (O : Oracle N b h ρ) : ℕ :=
  ((A.chip (A.state E)).addrs (A.world E R O) []).card

/-- The device's compute in the window, at `c_tag` per oracle query and `c_enc` per rebuilt word. -/
def compute (A : InteractiveDevice N b h ρ S L q) (cTag cEnc : ℕ) (E : Store N b) (R : Bits ρ) (O : Oracle N b h ρ) :
    ℕ :=
  ((A.chip (A.state E)).queries (A.world E R O) []).length * cTag +
    (A.chip (A.state E)).rebuilds (A.world E R O) [] * cEnc

/-- The verifier accepts: every sampled word's committed tag is the oracle's tag of that word under the stream. -/
def Accepts {s : ℕ} (A : InteractiveDevice N b h ρ S L q) (E : Store N b) (R : Bits ρ) (O : Oracle N b h ρ)
    (I : Fin s → Fin N) : Prop :=
  ∀ i, A.tags E R O (I i) = O (R, I i, E (I i))

end InteractiveDevice

end Accounting.Rate

namespace Accounting

/-- Run `r`'s bandwidth audit is the rate audit's game with interactive side sources at price `p`, `c_tag` per tag and
`c_enc` per rebuilt word: `RealizesRate`, with an interactive device of the game, within the device model's limits over
the bandwidth window, at each value of the other audits' coins. -/
structure RealizesRateInteractive (p : Price) (cTag cEnc : ℕ) {N b h ρ s : ℕ} {ι : Type} [Fintype ι] {d : Device ι}
    {α ΩW ΩS : Type} [Fintype α] [DecidableEq α] (r : Run d α ΩW (RateCoins N b h ρ s) ΩS) where
  /-- the interactive device of the game at the other audits' coins -/
  dev : ΩW × ΩS → Rate.InteractiveDevice N b h ρ d.chip (∑ i, d.cap i * (r.win .bandwidth).len)
    (d.rate .oracle * (r.win .bandwidth).len)
  /-- the bandwidth audit's verdict is the game's -/
  accepts_iff : ∀ x y s', r.accepts .bandwidth (x, y, s') ↔ (dev (x, s')).Accepts y.1 y.2.1 y.2.2.1 y.2.2.2
  /-- the bandwidth account's reads from device memory cover the game's ingress -/
  ingress_le : ∀ x y s', (dev (x, s')).ingress y.1 y.2.1 y.2.2.1 ≤ r.read (x, y, s')
  /-- the work of the accounts other than the compute account covers the game's tags and rebuilds -/
  compute_le : ∀ x y s', (dev (x, s')).compute cTag cEnc y.1 y.2.1 y.2.2.1 ≤ r.othersWork p (x, y, s')

end Accounting
