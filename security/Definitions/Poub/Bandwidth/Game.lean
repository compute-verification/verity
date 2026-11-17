module

public import Definitions.Pous.Game.Prob
import Mathlib.Basic.ENNReal.Real

/-! The bandwidth audit v0's game in the random-oracle model, `S`, `c` and `G` parameters of the device class. -/

@[expose] public section

namespace Bandwidth

open Pous (Bits)

/-- A store: `N` blocks of `b` bits. -/
abbrev Store (N b : ℕ) : Type := Fin N → Bits b

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
  block : Store N b

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
  state : Store N b → Bits S
  mem : Store N b → ℕ → Bool
  side : Store N b → ℕ → Bool
  chip : Bits S → Chip N b h (Fin N → Bits h)
  bounded : ∀ (st : Bits S) (w : World N b h),
    (chip st).sideBits w ≤ c ∧ (chip st).regens w ≤ G ∧ ((chip st).queries w).length ≤ q

namespace Device

variable {N b h S c G q : ℕ}

/-- The world a device's chip runs in, for store `E` and oracle `O`. -/
def world (A : Device N b h S c G q) (E : Store N b) (O : Oracle N b h) : World N b h :=
  ⟨O, A.mem E, A.side E, E⟩

/-- The tags the device commits to. -/
def tags (A : Device N b h S c G q) (E : Store N b) (O : Oracle N b h) : Fin N → Bits h :=
  (A.chip (A.state E)).run (A.world E O)

/-- The device's ingress in the window, in bits. -/
def ingress (A : Device N b h S c G q) (E : Store N b) (O : Oracle N b h) : ℕ :=
  (A.chip (A.state E)).ingressBits (A.world E O)

end Device

/-- The verifier accepts: every sampled block's committed tag is the oracle's tag of that block. -/
def Accepts {N b h S c G q s : ℕ} (A : Device N b h S c G q) (E : Store N b) (O : Oracle N b h)
    (I : Fin s → Fin N) : Prop :=
  ∀ i, A.tags E O (I i) = O (I i, E (I i))

end Bandwidth
