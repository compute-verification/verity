module

public import Definitions.Pous.Game.Prob
public import Definitions.Core.Draw

/-! The rate audit's game in the random-oracle model: the bandwidth audit's game for an encoded store, with a certified
stream and rebuilds paid for in compute. -/

@[expose] public section

namespace ResourceAccounting.Rate

open Pous (Bits)

/-- A store: `N` encoded words of `b` bits. -/
abbrev Store (N b : ℕ) : Type := Fin N → Bits b

/-- The random oracle at the window's fresh seed: `H(σ, R, j, x)` for the certified stream `R` of `ρ` bits, word index
`j` and word contents `x`, an `h`-bit tag. -/
abbrev Oracle (N b h ρ : ℕ) : Type := Bits ρ × Fin N × Bits b → Bits h

/-- What a window's checks read besides its sample: the certified stream and the oracle at the fresh seed. -/
abbrev Pub (N b h ρ : ℕ) : Type := Bits ρ × Oracle N b h ρ

/-- A window's coins: the live coins (`Verity.Coins`) with the certified stream and the oracle at the fresh seed as
their public part, and the `s` sampled words as their sample. -/
abbrev Coins (N b h ρ s : ℕ) : Type := Verity.Coins (Pub N b h ρ) s N

/-- The coins are finite, as a product: an instance of its own, because instance search with its default size does not
reach the product's under `n` windows' function type. -/
instance instFintypeCoins (N b h ρ s : ℕ) : Fintype (Coins N b h ρ s) :=
  inferInstanceAs (Fintype ((_ × _) × (Fin s → Fin N)))

/-- A chip program in the window, ending in a value of type `α`. Each step waits on one answer: a tag from the oracle,
one bit of device memory at an address (ingress), one bit of a free side source at an address, one bit of the certified
stream, or a rebuilt word. -/
inductive Chip (N b h ρ : ℕ) (α : Type) : Type where
  | done : α → Chip N b h ρ α
  | tag : Bits ρ × Fin N × Bits b → (Bits h → Chip N b h ρ α) → Chip N b h ρ α
  | ingress : ℕ → (Bool → Chip N b h ρ α) → Chip N b h ρ α
  | side : ℕ → (Bool → Chip N b h ρ α) → Chip N b h ρ α
  | stream : Fin ρ → (Bool → Chip N b h ρ α) → Chip N b h ρ α
  | rebuild : Fin N → (Bits b → Chip N b h ρ α) → Chip N b h ρ α

/-- What answers the chip's steps: the oracle, device memory, the free side sources, the certified stream and the store
the codec's inverse rebuilds. -/
structure World (N b h ρ : ℕ) where
  oracle : Oracle N b h ρ
  mem : ℕ → Bool
  side : ℕ → Bool
  stream : Bits ρ
  store : Store N b

namespace Chip

variable {N b h ρ : ℕ} {α : Type}

/-- The program's value in a world. -/
def run (w : World N b h ρ) : Chip N b h ρ α → α
  | done a => a
  | tag x k => (k (w.oracle x)).run w
  | ingress a k => (k (w.mem a)).run w
  | side a k => (k (w.side a)).run w
  | stream i k => (k (w.stream i)).run w
  | rebuild j k => (k (w.store j)).run w

/-- The bits read from device memory in a world: one per `ingress` step. -/
def ingressBits (w : World N b h ρ) : Chip N b h ρ α → ℕ
  | done _ => 0
  | tag x k => (k (w.oracle x)).ingressBits w
  | ingress a k => (k (w.mem a)).ingressBits w + 1
  | side a k => (k (w.side a)).ingressBits w
  | stream i k => (k (w.stream i)).ingressBits w
  | rebuild j k => (k (w.store j)).ingressBits w

/-- The device-memory addresses read in a world. -/
def addrs (w : World N b h ρ) : Chip N b h ρ α → Finset ℕ
  | done _ => ∅
  | tag x k => (k (w.oracle x)).addrs w
  | ingress a k => insert a ((k (w.mem a)).addrs w)
  | side a k => (k (w.side a)).addrs w
  | stream i k => (k (w.stream i)).addrs w
  | rebuild j k => (k (w.store j)).addrs w

/-- The bits that came in over the side link in a world: one per `side` step and one per `stream` step. -/
def linkBits (w : World N b h ρ) : Chip N b h ρ α → ℕ
  | done _ => 0
  | tag x k => (k (w.oracle x)).linkBits w
  | ingress a k => (k (w.mem a)).linkBits w
  | side a k => (k (w.side a)).linkBits w + 1
  | stream i k => (k (w.stream i)).linkBits w + 1
  | rebuild j k => (k (w.store j)).linkBits w

/-- The words rebuilt in a world: one per `rebuild` step. -/
def rebuilds (w : World N b h ρ) : Chip N b h ρ α → ℕ
  | done _ => 0
  | tag x k => (k (w.oracle x)).rebuilds w
  | ingress a k => (k (w.mem a)).rebuilds w
  | side a k => (k (w.side a)).rebuilds w
  | stream i k => (k (w.stream i)).rebuilds w
  | rebuild j k => (k (w.store j)).rebuilds w + 1

/-- The oracle queries in a world, in order. -/
def queries (w : World N b h ρ) : Chip N b h ρ α → List (Bits ρ × Fin N × Bits b)
  | done _ => []
  | tag x k => x :: (k (w.oracle x)).queries w
  | ingress a k => (k (w.mem a)).queries w
  | side a k => (k (w.side a)).queries w
  | stream i k => (k (w.stream i)).queries w
  | rebuild j k => (k (w.store j)).queries w

end Chip

/-- A prover: its strategy on a device of the device model (`ResourceAccounting.Device`), bounded by that device's figures. It
lays out on-chip state (`S` bits, the device's `chip`), device memory and free side sources, each from the store before
the seed and the stream, and runs its chip program from that state. In every world the program takes in at most `L`
bits over the side link (free side sources and the certified stream together: its `cap` over the window) and makes at
most `q` oracle queries. -/
structure Prover (N b h ρ S L q : ℕ) where
  state : Store N b → Bits S
  mem : Store N b → ℕ → Bool
  side : Store N b → ℕ → Bool
  chip : Bits S → Chip N b h ρ (Fin N → Bits h)
  bounded : ∀ (st : Bits S) (w : World N b h ρ), (chip st).linkBits w ≤ L ∧ ((chip st).queries w).length ≤ q

namespace Prover

variable {N b h ρ S L q : ℕ}

/-- The world a device's chip runs in, for store `E`, certified stream `R` and oracle `O`. -/
def world (A : Prover N b h ρ S L q) (E : Store N b) (R : Bits ρ) (O : Oracle N b h ρ) : World N b h ρ :=
  ⟨O, A.mem E, A.side E, R, E⟩

/-- The tags the device commits to. -/
def tags (A : Prover N b h ρ S L q) (E : Store N b) (R : Bits ρ) (O : Oracle N b h ρ) : Fin N → Bits h :=
  (A.chip (A.state E)).run (A.world E R O)

/-- The device's ingress in the window: every bit it read from device memory, in bits. -/
def ingress (A : Prover N b h ρ S L q) (E : Store N b) (R : Bits ρ) (O : Oracle N b h ρ) : ℕ :=
  (A.chip (A.state E)).ingressBits (A.world E R O)

/-- The device's footprint in the window: the distinct device-memory addresses it read, in bits. -/
def footprint (A : Prover N b h ρ S L q) (E : Store N b) (R : Bits ρ) (O : Oracle N b h ρ) : ℕ :=
  ((A.chip (A.state E)).addrs (A.world E R O)).card

/-- The device's compute in the window, at `c_tag` per oracle query and `c_enc` per rebuilt word. -/
def compute (A : Prover N b h ρ S L q) (cTag cEnc : ℕ) (E : Store N b) (R : Bits ρ) (O : Oracle N b h ρ) : ℕ :=
  ((A.chip (A.state E)).queries (A.world E R O)).length * cTag + (A.chip (A.state E)).rebuilds (A.world E R O) * cEnc

/-- The verifier accepts: every sampled word's committed tag is the oracle's tag of that word under the stream. -/
def Accepts {s : ℕ} (A : Prover N b h ρ S L q) (E : Store N b) (R : Bits ρ) (O : Oracle N b h ρ) (I : Fin s → Fin N) :
    Prop :=
  ∀ i, A.tags E R O (I i) = O (R, I i, E (I i))

end Prover

/-- **The rate audit's floor**, in bits times `c_enc`: `k·b·c_enc − (S + (L − ρ) + λ)·c_enc − (F − k·c_tag)·b`.
Divided by `c_enc` it is `k·b − S − c_free − λ − (F − k·c_tag)·b/c_enc` bits, where `c_free = L − ρ` is the link's
capacity the certified stream leaves free and `(F − k·c_tag)/c_enc` the words a compute budget `F` rebuilds after
tagging `k` words. -/
def floor (b ρ S L F cTag cEnc k lam : ℕ) : ℕ := k * b * cEnc - ((S + (L - ρ) + lam) * cEnc + (F - k * cTag) * b)

/-- **The devices of `n` windows on one store**: window `i`'s device, a function of every other window's coins. -/
abbrev Devices (n N b h ρ S L q s : ℕ) : Type :=
  (i : Fin n) → ({j : Fin n // j ≠ i} → Coins N b h ρ s) → Prover N b h ρ S L q

/-- Window `i`'s device at the windows' coins `ω`: it reads every coin but its own. -/
def Devices.dev {n N b h ρ S L q s : ℕ} (A : Devices n N b h ρ S L q s) (i : Fin n) (ω : Fin n → Coins N b h ρ s) :
    Prover N b h ρ S L q :=
  A i (fun j => ω j.1)

end ResourceAccounting.Rate
