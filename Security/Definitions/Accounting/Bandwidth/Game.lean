import Definitions.Pous.Game.Prob

/-!
# The bandwidth audit v0 in the random-oracle model

The game behind the bandwidth audit v0.
A device holds a store `E` of `N` blocks of `b` bits in device memory. The verifier reveals a fresh seed `σ`; the prover
must commit, before a deadline, to the tag `H(σ, j, E_j)` of every block `j`; the verifier then checks `s` blocks drawn
uniformly. Ingress is the number of bits the chip read from device memory in the window. The guarantee, PoUB's
per-pass lemma (`Definitions/Accounting/Guarantees/RemotePass.lean`), bounds the probability that a device is accepted
while its ingress falls short of the store by more than what it could have held or obtained another way.

**The random oracle.** `H(σ, ·, ·)` at the fresh seed is a uniform function `O : Fin N × Bits b → Bits h`, and the
device reaches it only through `tag` queries during the window. Nothing the device computed before `σ` depends on `O`.

**The device.** Before `σ` it lays out three things, each any function of `E`: its on-chip state (`S` bits), device
memory (read only through `ingress`, which counts each bit) and side sources (read only through `side`, at most `c`
bits per run; `c` stands for `cap · T`, every side channel's capacity over the window). During the window the chip runs
a program `Chip` from its state, querying the oracle, reading device memory and side sources, and regenerating blocks
(`regen j` returns `E_j`, at most `G` times per run: the window over the regeneration time `R_regen`, times the
devices that regenerate). It outputs a tag for every block. The sample `I` is drawn after the output and independently
of it, which is what the commitment before the deadline enforces. A program remembers every answer it has had, so the
chip's working memory in the window is unbounded: `S` bounds only what is on chip when the seed arrives, and writing to
device memory to read it back gains a device nothing.

**What is a parameter.** `S`, `c` and `G` are bounds the device class is drawn with; the claims registry names the
measurement that falsifies each for a deployed device (`bandwidth-s-chip`, `bandwidth-side-cap`,
`bandwidth-regen-time`). Treating the stored blocks as uniform, apart from regeneration, is the claim
`bandwidth-instantiation`; reading the deployed tag as the random oracle is `random-oracle`.
-/

namespace Bandwidth

open Pous (Bits)

/-- A store's contents: the payloads of its `N` registered blocks, `b` bits each. A draw `E` stands for the audit's store
`Accounting.Requirement.Store.ofPayloads E`, and is that store's payloads (`Store.payloads_ofPayloads`). -/
abbrev Payloads (N b : ℕ) : Type := Fin N → Bits b

/-- The random oracle at the fresh seed: `H(σ, j, x)` for block index `j` and block contents `x`, an `h`-bit tag. -/
abbrev Oracle (N b h : ℕ) : Type := Fin N × Bits b → Bits h

/-- A chip program in the window, ending in a value of type `α`. Each step waits on one answer: a tag from the oracle,
one bit of device memory at an address (ingress), one bit of a side source at an address, or a regenerated block. -/
inductive Chip (N b h : ℕ) (α : Type) : Type where
  | done : α → Chip N b h α
  | tag : Fin N × Bits b → (Bits h → Chip N b h α) → Chip N b h α
  | ingress : ℕ → (Bool → Chip N b h α) → Chip N b h α
  | side : ℕ → (Bool → Chip N b h α) → Chip N b h α
  | regen : Fin N → (Bits b → Chip N b h α) → Chip N b h α

/-- What answers the chip's steps: the oracle, device memory, side sources and the store regeneration rebuilds. -/
structure World (N b h : ℕ) where
  oracle : Oracle N b h
  mem : ℕ → Bool
  side : ℕ → Bool
  block : Payloads N b

namespace Chip

variable {N b h : ℕ} {α : Type}

/-- The program's value in a world. -/
def run (w : World N b h) : Chip N b h α → α
  | done a => a
  | tag x k => (k (w.oracle x)).run w
  | ingress a k => (k (w.mem a)).run w
  | side a k => (k (w.side a)).run w
  | regen j k => (k (w.block j)).run w

/-- The bits read from device memory in a world: one per `ingress` step. -/
def ingressBits (w : World N b h) : Chip N b h α → ℕ
  | done _ => 0
  | tag x k => (k (w.oracle x)).ingressBits w
  | ingress a k => (k (w.mem a)).ingressBits w + 1
  | side a k => (k (w.side a)).ingressBits w
  | regen j k => (k (w.block j)).ingressBits w

/-- The bits read from side sources in a world: one per `side` step. -/
def sideBits (w : World N b h) : Chip N b h α → ℕ
  | done _ => 0
  | tag x k => (k (w.oracle x)).sideBits w
  | ingress a k => (k (w.mem a)).sideBits w
  | side a k => (k (w.side a)).sideBits w + 1
  | regen j k => (k (w.block j)).sideBits w

/-- The blocks regenerated in a world: one per `regen` step. -/
def regens (w : World N b h) : Chip N b h α → ℕ
  | done _ => 0
  | tag x k => (k (w.oracle x)).regens w
  | ingress a k => (k (w.mem a)).regens w
  | side a k => (k (w.side a)).regens w
  | regen j k => (k (w.block j)).regens w + 1

/-- The oracle queries in a world, in order. -/
def queries (w : World N b h) : Chip N b h α → List (Fin N × Bits b)
  | done _ => []
  | tag x k => x :: (k (w.oracle x)).queries w
  | ingress a k => (k (w.mem a)).queries w
  | side a k => (k (w.side a)).queries w
  | regen j k => (k (w.block j)).queries w

end Chip

/-- A device: its on-chip state (`S` bits), device memory and side sources, each laid out from the store before the
seed, and its chip program from that state. In every world the program reads at most `c` bits of side sources,
regenerates at most `G` blocks and makes at most `q` oracle queries. -/
structure Device (N b h S c G q : ℕ) where
  state : Payloads N b → Bits S
  mem : Payloads N b → ℕ → Bool
  side : Payloads N b → ℕ → Bool
  chip : Bits S → Chip N b h (Fin N → Bits h)
  bounded : ∀ (st : Bits S) (w : World N b h),
    (chip st).sideBits w ≤ c ∧ (chip st).regens w ≤ G ∧ ((chip st).queries w).length ≤ q

namespace Device

variable {N b h S c G q : ℕ}

/-- The world a device's chip runs in, for store `E` and oracle `O`. -/
def world (A : Device N b h S c G q) (E : Payloads N b) (O : Oracle N b h) : World N b h :=
  ⟨O, A.mem E, A.side E, E⟩

/-- The tags the device commits to. -/
def tags (A : Device N b h S c G q) (E : Payloads N b) (O : Oracle N b h) : Fin N → Bits h :=
  (A.chip (A.state E)).run (A.world E O)

/-- The device's ingress in the window, in bits. -/
def ingress (A : Device N b h S c G q) (E : Payloads N b) (O : Oracle N b h) : ℕ :=
  (A.chip (A.state E)).ingressBits (A.world E O)

end Device

/-- The verifier accepts: every sampled block's committed tag is the oracle's tag of that block. -/
def Accepts {N b h S c G q s : ℕ} (A : Device N b h S c G q) (E : Payloads N b) (O : Oracle N b h)
    (I : Fin s → Fin N) : Prop :=
  ∀ i, A.tags E O (I i) = O (I i, E (I i))

end Bandwidth
