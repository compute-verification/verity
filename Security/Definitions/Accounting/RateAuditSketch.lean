import Mathlib.Data.ZMod.Basic
import Definitions.Pous.Game.Prob
import Definitions.Accounting.Tag.Sketch
import Definitions.Core.Draw

/-!
# The rate audit with a sketch-then-hash tag, for a device that reads sectors verbatim (trusted)

The rate audit's game (`RateAudit.lean`) with the tag the deployed operating point computes: in each window the verifier
reveals fresh signs `σ`, each `n`-byte tile `t` of the store gets the exact integer sketch `S·e_t` (`k_s` words, INT8 on
the tensor cores), and the prover commits to the tag `H(R, t, dig_σ(e_t))`, a random oracle over the certified stream
`R`, the tile's index and its sketch words; the verifier checks `s` tiles drawn uniformly. The guarantee
(`Definitions/Accounting/Guarantees/RateAuditSketch.lean`) bounds the probability that a device is accepted while the device
memory its window read falls short of the store by more than what it held on chip, took in over the link's free
capacity, or rebuilt with the compute it had.

**Sectors, verbatim.** The store is `N` tiles of `m` sectors (32 bytes each on sm_120, so `m = n/32`), and the stream
`ρ` sectors, all of one finite type `X`, with a sector worth `w` bits of the device's on-chip state, side link and
footprint. The device obtains a sector only whole and as it is (codec-theory §2.2's verbatim holder): it holds a fixed
set of the store's sectors on chip when the window opens (`held`, at most `S` bits), and in the window it reads a sector
from device memory (`mem`, ingress), takes one in over the side link (`link`, store or stream sectors: a host's copy of
the store, PCIe DMA into HBM among them), or rebuilds one with the codec's inverse at `c_enc` (`rebuild`). The link
carries at most `L` bits per window, `RateAudit`'s `c = β_side·(T + RTT)`. Nothing else the device has depends on the
store or the stream: unlike `RateAudit`'s, its on-chip state, device memory and link carry sectors, not arbitrary
functions of the store. That is the model's content: a device that holds a tile's sketch instead of the tile (256 bytes
per 4 KiB tile at `k_s = 64`), or a host that computes sketches and sends them over the link after `σ` is revealed, is
outside it, so the claims registry carries the model (`rate-sketch-verbatim`) and window isolation keeps such a host off
the link.

**The guess.** A tile the device queried the oracle at, without having obtained one of its sectors or one of the
stream's, needed the digest of contents it never saw: the unread sector is uniform and independent of the run, so the
digest is right with chance at most `fib/|X|`, where `fib` (`fib`) is the most contents of one sector that give one
digest with the rest of the tile fixed (at least `1`, a stream sector's, which the tag reads verbatim). The digest's
binding is that min-entropy: `fib = 1` for signs whose block on each sector is injective on INT8 sectors (`sketch`,
`SectorsBind`, which a `64 × 32` block of uniform signs is but for a small chance); at `k_s = 16` a `16 × 32` block
cannot be injective, and `fib` is the largest fiber of the sector's projection.

**The oracle and the windows** are `RateAudit`'s: `H` at the window's fresh seed is a uniform function the device
reaches only through `tag` queries, and its signs, stream, oracle and sample are fresh coins drawn after the layout.
-/

namespace Accounting.RateSketch

open Pous (Bits)

/-- A position of the window's data: sector `u` of tile `t` of the store (`inl (t, u)`), or sector `r` of the certified
stream (`inr r`). -/
abbrev Pos (N m ρ : ℕ) : Type := (Fin N × Fin m) ⊕ Fin ρ

/-- The window's data: the store's `N·m` sectors and the certified stream's `ρ`. -/
abbrev Data (N m ρ : ℕ) (X : Type) : Type := Pos N m ρ → X

/-- Tile `t` of the store: its `m` sectors. -/
def Data.tile {N m ρ : ℕ} {X : Type} (D : Data N m ρ X) (t : Fin N) : Fin m → X := fun u => D (.inl (t, u))

/-- The certified stream: `ρ` sectors. -/
def Data.stream {N m ρ : ℕ} {X : Type} (D : Data N m ρ X) : Fin ρ → X := fun r => D (.inr r)

/-- The random oracle at the window's fresh seed: `H(R, t, d)` for the certified stream `R`, tile index `t` and digest
`d`, an `h`-bit tag. -/
abbrev Oracle (N ρ h : ℕ) (X Dg : Type) : Type := (Fin ρ → X) × Fin N × Dg → Bits h

/-- What a window's checks read besides its stream and its sample: the signs and the oracle at the fresh seed. -/
abbrev Pub (N ρ h : ℕ) (X Sig Dg : Type) : Type := Sig × Oracle N ρ h X Dg

/-- A window's coins besides its stream: the live coins (`Verity.Coins`) with the signs and the oracle at the fresh seed
as their public part, and the `s` sampled tiles as their draw. -/
abbrev Coins (N ρ h s : ℕ) (X Sig Dg : Type) : Type := Verity.Coins (Pub N ρ h X Sig Dg) s N

instance instFintypeCoins (N ρ h s : ℕ) (X Sig Dg : Type) [Fintype X] [DecidableEq X] [Fintype Sig] [Fintype Dg]
    [DecidableEq Dg] : Fintype (Coins N ρ h s X Sig Dg) :=
  inferInstanceAs (Fintype ((_ × _) × (Fin s → Fin N)))

/-- A chip program in the window, ending in a value of type `α`. Each step waits on one answer: a tag from the oracle,
or one whole sector, read from device memory, taken in over the side link, or rebuilt. -/
inductive Chip (N m ρ h : ℕ) (X Dg : Type) (α : Type) : Type where
  | done : α → Chip N m ρ h X Dg α
  | tag : (Fin ρ → X) × Fin N × Dg → (Bits h → Chip N m ρ h X Dg α) → Chip N m ρ h X Dg α
  | mem : Fin N × Fin m → (X → Chip N m ρ h X Dg α) → Chip N m ρ h X Dg α
  | link : Pos N m ρ → (X → Chip N m ρ h X Dg α) → Chip N m ρ h X Dg α
  | rebuild : Fin N × Fin m → (X → Chip N m ρ h X Dg α) → Chip N m ρ h X Dg α

namespace Chip

variable {N m ρ h : ℕ} {X Dg α : Type}

/-- The program's value under oracle `O` and data `D`: every read returns the sector as it is. -/
def run (O : Oracle N ρ h X Dg) (D : Data N m ρ X) : Chip N m ρ h X Dg α → α
  | done a => a
  | tag x k => (k (O x)).run O D
  | mem p k => (k (D (.inl p))).run O D
  | link p k => (k (D p)).run O D
  | rebuild p k => (k (D (.inl p))).run O D

/-- The oracle queries, in order. -/
def queries (O : Oracle N ρ h X Dg) (D : Data N m ρ X) : Chip N m ρ h X Dg α → List ((Fin ρ → X) × Fin N × Dg)
  | done _ => []
  | tag x k => x :: (k (O x)).queries O D
  | mem p k => (k (D (.inl p))).queries O D
  | link p k => (k (D p)).queries O D
  | rebuild p k => (k (D (.inl p))).queries O D

/-- The distinct store sectors read from device memory. -/
def mems (O : Oracle N ρ h X Dg) (D : Data N m ρ X) : Chip N m ρ h X Dg α → Finset (Fin N × Fin m)
  | done _ => ∅
  | tag x k => (k (O x)).mems O D
  | mem p k => insert p ((k (D (.inl p))).mems O D)
  | link p k => (k (D p)).mems O D
  | rebuild p k => (k (D (.inl p))).mems O D

/-- The sectors taken in over the side link: one per `link` step. -/
def linkSteps (O : Oracle N ρ h X Dg) (D : Data N m ρ X) : Chip N m ρ h X Dg α → ℕ
  | done _ => 0
  | tag x k => (k (O x)).linkSteps O D
  | mem p k => (k (D (.inl p))).linkSteps O D
  | link p k => (k (D p)).linkSteps O D + 1
  | rebuild p k => (k (D (.inl p))).linkSteps O D

/-- The sectors rebuilt: one per `rebuild` step. -/
def rebuilds (O : Oracle N ρ h X Dg) (D : Data N m ρ X) : Chip N m ρ h X Dg α → ℕ
  | done _ => 0
  | tag x k => (k (O x)).rebuilds O D
  | mem p k => (k (D (.inl p))).rebuilds O D
  | link p k => (k (D p)).rebuilds O D
  | rebuild p k => (k (D (.inl p))).rebuilds O D + 1

/-- The positions a run's reads were answered at: device memory and rebuilding at store sectors, the link at any. -/
def reads (O : Oracle N ρ h X Dg) (D : Data N m ρ X) : Chip N m ρ h X Dg α → Finset (Pos N m ρ)
  | done _ => ∅
  | tag x k => (k (O x)).reads O D
  | mem p k => insert (.inl p) ((k (D (.inl p))).reads O D)
  | link p k => insert p ((k (D p)).reads O D)
  | rebuild p k => insert (.inl p) ((k (D (.inl p))).reads O D)

end Chip

/-- A device: the store sectors it holds on chip when the window opens (`w` bits each, at most `S` bits), and its
chip program from the signs and those sectors. In every run the program takes in at most `L` bits over the side link
and makes at most `q` oracle queries. -/
structure Device (N m ρ h S L q w : ℕ) (X Sig Dg : Type) where
  held : Finset (Fin N × Fin m)
  held_le : held.card * w ≤ S
  chip : Sig → (held → X) → Chip N m ρ h X Dg (Fin N → Bits h)
  bounded : ∀ (σ : Sig) (v : held → X) (O : Oracle N ρ h X Dg) (D : Data N m ρ X),
    (chip σ v).linkSteps O D * w ≤ L ∧ ((chip σ v).queries O D).length ≤ q

namespace Device

variable {N m ρ h S L q w : ℕ} {X Sig Dg : Type}

/-- The device's program in the window at signs `σ` and data `D`: its chip program from the sectors it holds. -/
def prog (A : Device N m ρ h S L q w X Sig Dg) (σ : Sig) (D : Data N m ρ X) : Chip N m ρ h X Dg (Fin N → Bits h) :=
  A.chip σ (fun p => D (.inl p.1))

/-- The tags the device commits to. -/
def tags (A : Device N m ρ h S L q w X Sig Dg) (σ : Sig) (O : Oracle N ρ h X Dg) (D : Data N m ρ X) :
    Fin N → Bits h :=
  (A.prog σ D).run O D

/-- The device's footprint in the window: the distinct store sectors it read from device memory, in bits. -/
def footprint (A : Device N m ρ h S L q w X Sig Dg) (σ : Sig) (O : Oracle N ρ h X Dg) (D : Data N m ρ X) : ℕ :=
  ((A.prog σ D).mems O D).card * w

/-- The device's compute in the window, at `c_tag` per oracle query and `c_enc` per rebuilt sector. -/
def compute (A : Device N m ρ h S L q w X Sig Dg) (cTag cEnc : ℕ) (σ : Sig) (O : Oracle N ρ h X Dg)
    (D : Data N m ρ X) : ℕ :=
  ((A.prog σ D).queries O D).length * cTag + (A.prog σ D).rebuilds O D * cEnc

/-- The verifier accepts: every sampled tile's committed tag is the oracle's tag of the stream, the tile's index and
the digest `dig σ` of the tile. -/
def Accepts {s : ℕ} (A : Device N m ρ h S L q w X Sig Dg) (dig : Sig → (Fin m → X) → Dg) (σ : Sig)
    (O : Oracle N ρ h X Dg) (D : Data N m ρ X) (I : Fin s → Fin N) : Prop :=
  ∀ i, A.tags σ O D (I i) = O (D.stream, I i, dig σ (D.tile (I i)))

/-- Every tile whose committed tag is correct, whether or not a verifier sampled it. -/
def correct (A : Device N m ρ h S L q w X Sig Dg) (dig : Sig → (Fin m → X) → Dg) (σ : Sig)
    (O : Oracle N ρ h X Dg) (D : Data N m ρ X) : Finset (Fin N) :=
  Finset.univ.filter fun t => A.tags σ O D t = O (D.stream, t, dig σ (D.tile t))

/-- The positions a run obtained: the sectors held on chip and every position it read. -/
def obtained (A : Device N m ρ h S L q w X Sig Dg) (σ : Sig) (O : Oracle N ρ h X Dg) (D : Data N m ρ X) :
    Finset (Pos N m ρ) :=
  A.held.map Function.Embedding.inl ∪ (A.prog σ D).reads O D

end Device

/-- **The digest's fiber** at signs `σ`: the most contents `a` of one sector `u` that give one digest `d` with the rest
of the tile `x` fixed, and at least `1`. The guess term is `fib/|X|` per query. -/
noncomputable def fib {m : ℕ} {X Sig Dg : Type} [Fintype X] [DecidableEq X] [Fintype Dg] [DecidableEq Dg]
    (dig : Sig → (Fin m → X) → Dg) (σ : Sig) : ℕ :=
  max 1 (Finset.univ.sup fun p : (Fin m → X) × Fin m × Dg =>
    (Finset.univ.filter fun a : X => dig σ (Function.update p.1 p.2.1 a) = p.2.2).card)

/-- **The sketch rate audit's floor**, in bits times `c_enc`: `k·m·w·c_enc − (S + (L − ρ·w))·c_enc − (F − k·c_tag)·w`.
Divided by `c_enc` it is `k·m·w − S − c_free − (F − k·c_tag)·w/c_enc` bits: `k` tiles of `m` sectors, less the on-chip
state, the link's capacity `c_free = L − ρ·w` the certified stream leaves free, and the sectors a compute budget `F`
rebuilds after tagging `k` tiles. -/
def floor (m w ρ S L F cTag cEnc k : ℕ) : ℕ := k * (m * w) * cEnc - ((S + (L - ρ * w)) * cEnc + (F - k * cTag) * w)

/-! ## The tensor-core sketch -/

/-- A 32-byte sector. -/
abbrev Sector : Type := Fin 32 → Fin 256

/-- A byte as an INT8, two's complement: `0 … 127` and `−128 … −1`. -/
def int8 (a : Fin 256) : ℤ := if a.val < 128 then a.val else (a.val : ℤ) - 256

/-- The window's signs over a tile of `m` sectors: row `i`'s sign at byte `l` of sector `u`, as the `k_s × n` matrix
`S` read sector by sector. -/
abbrev SketchSigns (k m : ℕ) : Type := Fin k → Fin m → Fin 32 → Bool

/-- The `k × 32` block of the signs on sector `u`. -/
def SketchSigns.block {k m : ℕ} (σ : SketchSigns k m) (u : Fin m) : Tag.Signs k 32 := fun i l => σ i u l

/-- **The sketch digest** of a tile: coordinate `i` is `Σ_u Σ_l sgn(σ i u l) · int8(x u l)`, the exact integer `S·e` the
INT8 tensor cores (`mma.sync` m16n8k32 `.s8.s8.s32`) accumulate, as its INT32 accumulator's bits. -/
def sketch {k m : ℕ} (σ : SketchSigns k m) (x : Fin m → Sector) : Fin k → ZMod (2 ^ 32) :=
  fun i => ((∑ u, Tag.project (σ.block u) (fun l => int8 (x u l)) i : ℤ) : ZMod (2 ^ 32))

/-- **The signs bind sectors**: each sector's block of signs is injective on INT8 sectors. A `k × 32` block of full
column rank is; a `16 × 32` block is not (Siegel's lemma gives it a kernel vector with entries at most 32). -/
def SectorsBind {k m : ℕ} (σ : SketchSigns k m) : Prop :=
  ∀ u, Function.Injective (fun a : Sector => Tag.project (σ.block u) (fun l => int8 (a l)))

end Accounting.RateSketch
