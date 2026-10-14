import Definitions.Accounting.RateAuditSketch

/-!
# The sketch rate audit with interactive side sources, verbatim (trusted)

`RateAuditSketch.lean`'s game, except that the chip may also send bits to the side sources, and a side source answers
each read with one whole sector of the store or of the certified stream, as it is, at an address it picks from every
bit the chip has sent so far. Sends are not bounded, so the chip may send any number `e` of bits that depend on the
signs and on what it has read. Every sector that comes in over the side link still counts `w` bits against `L`, with
the certified stream's `ρ` sectors, and the device class is still `rate-sketch-verbatim`'s: nothing on chip, read,
taken in or rebuilt is a function of several sectors. A source that picked its address from the store, the stream or
the oracle itself, rather than from what the chip sent, would tell the chip about sectors it never took in, or relay
tags, which this game leaves out.
-/

namespace Accounting.RateSketch

open Pous (Bits)

/-- A sketch chip program in the window that may also send: `Chip`'s steps, with `side`, which takes in one sector
over the side link at the address the side sources pick and returns it with that address, and `send β p`, which sends
the bit `β` to the side sources and continues with `p`. -/
inductive IChip (N m ρ h : ℕ) (X Dg : Type) (α : Type) : Type where
  | done : α → IChip N m ρ h X Dg α
  | tag : (Fin ρ → X) × Fin N × Dg → (Bits h → IChip N m ρ h X Dg α) → IChip N m ρ h X Dg α
  | mem : Fin N × Fin m → (X → IChip N m ρ h X Dg α) → IChip N m ρ h X Dg α
  | side : (Pos N m ρ × X → IChip N m ρ h X Dg α) → IChip N m ρ h X Dg α
  | rebuild : Fin N × Fin m → (X → IChip N m ρ h X Dg α) → IChip N m ρ h X Dg α
  | send : Bool → IChip N m ρ h X Dg α → IChip N m ρ h X Dg α

namespace IChip

variable {N m ρ h : ℕ} {X Dg α : Type}

/-- The program's value under side sources `addr` (the address of a read, after the chip has sent `v`, newest bit
first), oracle `O` and data `D`, having sent `v` so far: every read returns the sector as it is. -/
def run (addr : List Bool → Pos N m ρ) (O : Oracle N ρ h X Dg) (D : Data N m ρ X) :
    List Bool → IChip N m ρ h X Dg α → α
  | _, done a => a
  | v, tag x k => (k (O x)).run addr O D v
  | v, mem p k => (k (D (.inl p))).run addr O D v
  | v, side k => (k (addr v, D (addr v))).run addr O D v
  | v, rebuild p k => (k (D (.inl p))).run addr O D v
  | v, send β p => p.run addr O D (β :: v)

/-- The oracle queries, in order. -/
def queries (addr : List Bool → Pos N m ρ) (O : Oracle N ρ h X Dg) (D : Data N m ρ X) :
    List Bool → IChip N m ρ h X Dg α → List ((Fin ρ → X) × Fin N × Dg)
  | _, done _ => []
  | v, tag x k => x :: (k (O x)).queries addr O D v
  | v, mem p k => (k (D (.inl p))).queries addr O D v
  | v, side k => (k (addr v, D (addr v))).queries addr O D v
  | v, rebuild p k => (k (D (.inl p))).queries addr O D v
  | v, send β p => p.queries addr O D (β :: v)

/-- The distinct store sectors read from device memory. -/
def mems (addr : List Bool → Pos N m ρ) (O : Oracle N ρ h X Dg) (D : Data N m ρ X) :
    List Bool → IChip N m ρ h X Dg α → Finset (Fin N × Fin m)
  | _, done _ => ∅
  | v, tag x k => (k (O x)).mems addr O D v
  | v, mem p k => insert p ((k (D (.inl p))).mems addr O D v)
  | v, side k => (k (addr v, D (addr v))).mems addr O D v
  | v, rebuild p k => (k (D (.inl p))).mems addr O D v
  | v, send β p => p.mems addr O D (β :: v)

/-- The sectors taken in over the side link: one per `side` step. Sends are not counted. -/
def linkSteps (addr : List Bool → Pos N m ρ) (O : Oracle N ρ h X Dg) (D : Data N m ρ X) :
    List Bool → IChip N m ρ h X Dg α → ℕ
  | _, done _ => 0
  | v, tag x k => (k (O x)).linkSteps addr O D v
  | v, mem p k => (k (D (.inl p))).linkSteps addr O D v
  | v, side k => (k (addr v, D (addr v))).linkSteps addr O D v + 1
  | v, rebuild p k => (k (D (.inl p))).linkSteps addr O D v
  | v, send β p => p.linkSteps addr O D (β :: v)

/-- The sectors rebuilt: one per `rebuild` step. -/
def rebuilds (addr : List Bool → Pos N m ρ) (O : Oracle N ρ h X Dg) (D : Data N m ρ X) :
    List Bool → IChip N m ρ h X Dg α → ℕ
  | _, done _ => 0
  | v, tag x k => (k (O x)).rebuilds addr O D v
  | v, mem p k => (k (D (.inl p))).rebuilds addr O D v
  | v, side k => (k (addr v, D (addr v))).rebuilds addr O D v
  | v, rebuild p k => (k (D (.inl p))).rebuilds addr O D v + 1
  | v, send β p => p.rebuilds addr O D (β :: v)

end IChip

/-- A device whose side sources answer what its chip sent: the store sectors it holds on chip (`w` bits each, at most
`S` bits), side sources that pick each read's address from the bits sent so far, and a chip program from the signs and
the held sectors. In every run, starting from nothing sent, the program takes in at most `L` bits over the side link and
makes at most `q` oracle queries. -/
structure InteractiveDevice (N m ρ h S L q w : ℕ) (X Sig Dg : Type) where
  held : Finset (Fin N × Fin m)
  held_le : held.card * w ≤ S
  addr : List Bool → Pos N m ρ
  chip : Sig → (held → X) → IChip N m ρ h X Dg (Fin N → Bits h)
  bounded : ∀ (σ : Sig) (v : held → X) (O : Oracle N ρ h X Dg) (D : Data N m ρ X),
    (chip σ v).linkSteps addr O D [] * w ≤ L ∧ ((chip σ v).queries addr O D []).length ≤ q

namespace InteractiveDevice

variable {N m ρ h S L q w : ℕ} {X Sig Dg : Type}

/-- The device's program in the window at signs `σ` and data `D`: its chip program from the sectors it holds. -/
def prog (A : InteractiveDevice N m ρ h S L q w X Sig Dg) (σ : Sig) (D : Data N m ρ X) :
    IChip N m ρ h X Dg (Fin N → Bits h) :=
  A.chip σ (fun p => D (.inl p.1))

/-- The tags the device commits to. -/
def tags (A : InteractiveDevice N m ρ h S L q w X Sig Dg) (σ : Sig) (O : Oracle N ρ h X Dg) (D : Data N m ρ X) :
    Fin N → Bits h :=
  (A.prog σ D).run A.addr O D []

/-- The device's footprint in the window: the distinct store sectors it read from device memory, in bits. -/
def footprint (A : InteractiveDevice N m ρ h S L q w X Sig Dg) (σ : Sig) (O : Oracle N ρ h X Dg) (D : Data N m ρ X) :
    ℕ :=
  ((A.prog σ D).mems A.addr O D []).card * w

/-- The device's compute in the window, at `c_tag` per oracle query and `c_enc` per rebuilt sector. -/
def compute (A : InteractiveDevice N m ρ h S L q w X Sig Dg) (cTag cEnc : ℕ) (σ : Sig) (O : Oracle N ρ h X Dg)
    (D : Data N m ρ X) : ℕ :=
  ((A.prog σ D).queries A.addr O D []).length * cTag + (A.prog σ D).rebuilds A.addr O D [] * cEnc

/-- The verifier accepts: every sampled tile's committed tag is the oracle's tag of the stream, the tile's index and
the digest `dig σ` of the tile. -/
def Accepts {s : ℕ} (A : InteractiveDevice N m ρ h S L q w X Sig Dg) (dig : Sig → (Fin m → X) → Dg) (σ : Sig)
    (O : Oracle N ρ h X Dg) (D : Data N m ρ X) (I : Fin s → Fin N) : Prop :=
  ∀ i, A.tags σ O D (I i) = O (D.stream, I i, dig σ (D.tile (I i)))

end InteractiveDevice

end Accounting.RateSketch
