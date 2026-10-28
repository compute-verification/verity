import Definitions.Pous.Game.Prob
import Definitions.Core.Draw

/-!
# The rate audit in the random-oracle model (trusted)

The game behind the rate audit: the bandwidth audit's game (`Bandwidth/Game.lean`) for a store that is encoded, on a
device whose side link carries a certified stream and whose regeneration is paid for in compute. A device holds a store
`E` of `N` encoded words of `b` bits in device memory, encoded under a seed drawn after registration. In each window
the trusted host streams `ρ` fresh bits `R` to the device over its side link (PCIe), and the verifier reveals a fresh
seed `σ`; the prover must commit, before a deadline, to the tag `H(σ, R, j, E_j)` of every word `j`, so every tag
absorbs the stream; the verifier then checks `s` words drawn uniformly. The guarantee
(`Definitions/Accounting/Guarantees/RateAudit.lean`) bounds the probability that a device is accepted while the device memory
its window read falls short of the store by more than what it could have held on chip, taken in over the link's free
capacity, guessed, or rebuilt with the compute it had.

**The random oracle.** `H(σ, ·, ·, ·)` at the fresh seed is a uniform function `O : Bits ρ × Fin N × Bits b → Bits h`,
and the device reaches it only through `tag` queries during the window. Nothing the device computed before `σ` depends
on `O` or on `R`.

**The device.** Before `σ` and `R` it lays out three things, each any function of `E`: its on-chip state (`S` bits),
device memory (read only through `ingress`) and free side sources (read only through `side`). During the window the
chip runs a program `Chip` from its state, querying the oracle, reading device memory, side sources and the certified
stream (`stream i` returns bit `i` of `R`), and rebuilding words (`rebuild j` returns `E_j`). A side source is read
at an address the chip picks, so it answers any function of the store and of what the chip sent, and sending is free;
it reaches the oracle only through what the chip sends it, so it relays no tag the chip did not query. Side-source and
stream bits both come in over the side link, which carries at most `L` bits per window: in the actual-circuit model
(B3 at `e = 0`), `L` is `c = β_side·(T + RTT)`, the
side links' signals over the deadline and a round trip, timing included, with no `2^e` factor for what the chip sent.
The certified stream's `ρ` bits (certified ingress) must come in too, because a tag absorbs all of `R`, so at most
`L − ρ` bits of the link are free for anything else. The audited location is the chip: device memory is what HBM held
when the seed arrived, and answers a read with the bit at its address (passive memory), so a bit DMA'd into HBM during
the window crossed a side link, and the chip's read of it is a side read, counted in `L`. The chip makes at most `q`
oracle queries. It
outputs a tag for every word. The sample is drawn after the output and independently of it, which is what the
commitment before the deadline enforces. A program remembers every answer it has had, so the chip's working memory in
the window is unbounded: `S` bounds only what is on chip when the seed arrives.

**Compute.** A tag costs `c_tag` and a rebuilt word `c_enc`, in one unit of work; a run's compute is its queries times
`c_tag` plus its rebuilt words times `c_enc` (`Prover.compute`). Rebuilding is the codec's inverse: the device knows
the weights the store encodes (the worst case, as with weights from a PRG), so it never needs to decode, and a word it
neither holds nor reads it can still recompute at `c_enc`. The model has no other way to produce a word, which is the
codec's work-hardness (claim `rate-codec-work`): the inverse costs `c_enc` per word, with no amortization across
words. The honest device decodes each word it reads, at `c_dec`, but decoding is the useful work, charged to the compute
account; the game does not charge it.

**Residency and bandwidth.** A run's footprint (`Prover.footprint`) is the number of distinct device-memory addresses
it read: device memory held at least that many bits, laid out before the seed, that the window depended on. Its
ingress (`Prover.ingress`) counts every read, so it is at least the footprint, and is the bits that crossed from device
memory to the chip.

**What is a parameter.** `S`, `L`, `q`, `c_tag`, `c_enc` and the compute budget are bounds the device class is drawn
with; the claims registry names the measurement that falsifies each for a deployed device (`bandwidth-s-chip`,
`bandwidth-side-cap`, and `rate-codec-work` for `c_enc`). Treating the stored words as uniform is
`bandwidth-instantiation`; reading the deployed tag as the random oracle is `random-oracle`.

**Windows.** `n` windows on one store each draw fresh coins (a stream, an oracle and a sample), each played by its own
device (`Devices`). Window `i`'s device, its layout included, may be any function of the other windows' coins, earlier
or later, and never of its own.
-/

namespace Accounting.Rate

open Pous (Bits)

/-- A store: `N` encoded words of `b` bits. -/
abbrev Store (N b : ℕ) : Type := Fin N → Bits b

/-- The random oracle at the window's fresh seed: `H(σ, R, j, x)` for the certified stream `R` of `ρ` bits, word index
`j` and word contents `x`, an `h`-bit tag. -/
abbrev Oracle (N b h ρ : ℕ) : Type := Bits ρ × Fin N × Bits b → Bits h

/-- What a window's checks read besides its sample: the certified stream and the oracle at the fresh seed. -/
abbrev Pub (N b h ρ : ℕ) : Type := Bits ρ × Oracle N b h ρ

/-- A window's coins: the live coins (`Verity.Coins`) with the certified stream and the oracle at the fresh seed as
their public part, and the `s` sampled words as their draw. -/
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

/-- A prover: its strategy on a device of the device model (`Accounting.Device`), bounded by that device's figures. It
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

end Accounting.Rate
