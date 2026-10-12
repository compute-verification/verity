import Definitions.Accounting.RateAudit

/-!
# Receipts bound to the matmul: the generic-mix game on two locations (trusted)

The remote protocol's receipt for block `j` binds the block to the noised output of the PoUW tile before it: it is
`H(σ, j, sketch(E_j ⊙ expand(z_{j−1})))`, so the tile output enters before the sketch. In the generic-mix model the
receipt is `Mix(j, E_j, z_j)` for a random oracle `Mix` at the window's fresh challenge on the block's index, its
encoded contents `E_j` (`b` bits) and the tile output `z_j` its receipt binds (`a` bits; `z_j` here is the design's
`z_{j−1}`), with `h`-bit receipts. The challenge is fresh, not secret, so every location reaches the mix; what it
lacks is an input.

**Two locations** (the actual-circuit model's localization, `note:console/20261012T0645Z-draft-actual-circuit-model`):
the GPU, the audited location, which holds the tile outputs natively and reads the store from HBM, and the rest of the
node, the memory side (host DRAM, a box outside), which holds all of the store and, before the window, nothing about
the tile outputs. The links between them carry at most `L` bits to the GPU and `Lz` bits from it per window.

- The GPU is a `Rate.Device` with no certified stream, for each value of the tile outputs: its layout, its chip and
  what the rest answers it may be any function of the store and the tile outputs. Its chip reaches the mix at the tile
  outputs it holds (`atZ`): a query names a block and contents. Its side reads are what came in from the rest, at most
  `L` bits; its rebuilds are the codec's inverse at `c_enc`, as in `RateAudit`.
- The rest is a chip of the same kind for each store (`Rest`): its memory, laid out from the store, it reads freely;
  it reaches the mix at the store it holds (`atE`): a query names a block and a tile output. Its side reads are what
  came in from the GPU, at most `Lz` bits, answering any function of the store and the tile outputs; its `rebuild j`
  returns `z_j` and is tile work, at `w` per tile (`Node.tileWork`).

The answers that cross between the locations ignore the mix: a location's receipts depend on the mix through its own
queries only. A receipt one location made and the other committed is the one it made, and the verifier takes each
sampled block's receipt from either location (`Node.Accepts`), so that choice is the node's, after the fact. A mix
answer relayed across the link and used inside a location's computation is outside this game, and inside
`InteractiveNode`'s: there the answers may depend on the mix, which reaches the node only through its queries
(`InteractiveNode.QueryDetermined`).

**The ideal tile.** The tile outputs are uniform, and the rest has none of them except over the link or by tile work:
that is the PoUW tile's unpredictability below its work (`Accounting.Guarantees.TileUnpredictable`, from PoUW's
named assumptions), read in the generic model.
-/

namespace Accounting.BoundReceipt

open Pous (Bits)
open Accounting.Rate (Store Oracle Chip World Device)

/-- The mix at the window's fresh challenge: block `j`'s receipt is `Mix(j, E_j, z_j)`, an `h`-bit receipt of the
block's index, its encoded contents (`b` bits) and the tile output its receipt binds (`a` bits). -/
abbrev Mix (N a b h : ℕ) : Type := Fin N × Bits b × Bits a → Bits h

/-- A window's coins: the tile outputs, the mix and the `s` sampled blocks. -/
abbrev Coins (N a b h s : ℕ) : Type := Store N a × Mix N a b h × (Fin s → Fin N)

instance instFintypeCoins (N a b h s : ℕ) : Fintype (Coins N a b h s) := inferInstanceAs (Fintype (_ × _ × _))

/-- No certified stream: the receipt absorbs the challenge through the mix. -/
def noStream : Bits 0 := fun i => i.elim0

variable {N a b h : ℕ}

/-- The mix as the GPU reaches it, at the tile outputs `z` it holds: block `j` at contents `x` is `Mix(j, x, z_j)`. -/
def atZ (G : Mix N a b h) (z : Store N a) : Oracle N b h 0 := fun x => G (x.2.1, x.2.2, z x.2.1)

/-- The mix as the rest reaches it, at the store `E` it holds: block `j` at tile output `y` is `Mix(j, E_j, y)`. -/
def atE (G : Mix N a b h) (E : Store N b) : Oracle N a h 0 := fun x => G (x.2.1, E x.2.1, x.2.2)

/-- The rest of the node at one store: its memory, what the GPU sends it as a function of the tile outputs, and its
chip, which takes in at most `Lz` bits from the GPU and makes at most `qz` mix queries in every world. -/
structure Rest (N a h Lz qz : ℕ) where
  mem : ℕ → Bool
  side : Store N a → ℕ → Bool
  chip : Chip N a h 0 (Fin N → Bits h)
  bounded : ∀ w : World N a h 0, chip.linkBits w ≤ Lz ∧ (chip.queries w).length ≤ qz

namespace Rest

variable {Lz qz : ℕ}

/-- The world the rest's chip runs in: the mix at the store, its memory, what the GPU sent, and the tile outputs, which
its `rebuild` steps recompute. -/
def world (B : Rest N a h Lz qz) (E : Store N b) (z : Store N a) (G : Mix N a b h) : World N a h 0 :=
  ⟨atE G E, B.mem, B.side z, noStream, z⟩

/-- The receipts the rest commits to. -/
def receipts (B : Rest N a h Lz qz) (E : Store N b) (z : Store N a) (G : Mix N a b h) : Fin N → Bits h :=
  B.chip.run (B.world E z G)

end Rest

/-- A node: the GPU at each value of the tile outputs, and the rest at each store. -/
structure Node (N a b h S L q Lz qz : ℕ) where
  gpu : Store N a → Device N b h 0 S L q
  rest : Store N b → Rest N a h Lz qz

namespace Node

variable {S L q Lz qz : ℕ}

/-- The receipts the GPU commits to. -/
def gpuReceipts (A : Node N a b h S L q Lz qz) (E : Store N b) (z : Store N a) (G : Mix N a b h) : Fin N → Bits h :=
  (A.gpu z).tags E noStream (atZ G z)

/-- The receipts the rest commits to. -/
def restReceipts (A : Node N a b h S L q Lz qz) (E : Store N b) (z : Store N a) (G : Mix N a b h) : Fin N → Bits h :=
  (A.rest E).receipts E z G

/-- The verifier accepts: every sampled block's receipt, from the GPU or from the rest, is the mix of its index, its
contents and its tile output. -/
def Accepts {s : ℕ} (A : Node N a b h S L q Lz qz) (E : Store N b) (z : Store N a) (G : Mix N a b h)
    (I : Fin s → Fin N) : Prop :=
  ∀ i, A.gpuReceipts E z G (I i) = G (I i, E (I i), z (I i)) ∨ A.restReceipts E z G (I i) = G (I i, E (I i), z (I i))

/-- The GPU made block `j`'s receipt: it queried the mix at the block's contents, under the tile outputs it holds. -/
def MadeAtGpu (A : Node N a b h S L q Lz qz) (E : Store N b) (z : Store N a) (G : Mix N a b h) (j : Fin N) : Prop :=
  (noStream, j, E j) ∈ ((A.gpu z).chip ((A.gpu z).state E)).queries ((A.gpu z).world E noStream (atZ G z))

/-- The rest made block `j`'s receipt: it queried the mix at the block's tile output, under the store it holds. -/
def MadeAtRest (A : Node N a b h S L q Lz qz) (E : Store N b) (z : Store N a) (G : Mix N a b h) (j : Fin N) : Prop :=
  (noStream, j, z j) ∈ (A.rest E).chip.queries ((A.rest E).world E z G)

/-- The GPU's footprint in the window: the distinct HBM addresses it read, in bits. -/
def footprint (A : Node N a b h S L q Lz qz) (E : Store N b) (z : Store N a) (G : Mix N a b h) : ℕ :=
  (A.gpu z).footprint E noStream (atZ G z)

/-- The GPU's compute in the window, at `c_tag` per mix query and `c_enc` per rebuilt block. -/
def compute (A : Node N a b h S L q Lz qz) (cTag cEnc : ℕ) (E : Store N b) (z : Store N a) (G : Mix N a b h) : ℕ :=
  (A.gpu z).compute cTag cEnc E noStream (atZ G z)

/-- The rest's tile work in the window, at `w` per tile output it recomputed. -/
def tileWork (A : Node N a b h S L q Lz qz) (w : ℕ) (E : Store N b) (z : Store N a) (G : Mix N a b h) : ℕ :=
  (A.rest E).chip.rebuilds ((A.rest E).world E z G) * w

end Node

namespace Node

variable {S L q Lz qz : ℕ}

/-- The GPU's mix queries in the window, each a block and contents, under the tile outputs it holds. -/
def gpuQueries (A : Node N a b h S L q Lz qz) (E : Store N b) (z : Store N a) (G : Mix N a b h) :
    List (Bits 0 × Fin N × Bits b) :=
  ((A.gpu z).chip ((A.gpu z).state E)).queries ((A.gpu z).world E noStream (atZ G z))

/-- The rest's mix queries in the window, each a block and a tile output, under the store it holds. -/
def restQueries (A : Node N a b h S L q Lz qz) (E : Store N b) (z : Store N a) (G : Mix N a b h) :
    List (Bits 0 × Fin N × Bits a) :=
  (A.rest E).chip.queries ((A.rest E).world E z G)

end Node

/-! ## Answers between the locations that depend on the mix (`e > 0`) -/

/-- A node whose locations answer each other as functions of the mix too (`e > 0`): `node`'s programs, state and
memory, fixed before the mix is drawn, with what the rest answers the GPU (`gpuSide`) and what the GPU sends the rest
(`restSide`) any functions of the mix, the store and the tile outputs. -/
structure InteractiveNode (N a b h S L q Lz qz : ℕ) where
  node : Node N a b h S L q Lz qz
  gpuSide : Store N a → Mix N a b h → Store N b → ℕ → Bool
  restSide : Store N b → Mix N a b h → Store N a → ℕ → Bool

namespace InteractiveNode

variable {S L q Lz qz : ℕ}

/-- The node under the mix `G`: its locations answer each other as they do under `G`. -/
def under (A : InteractiveNode N a b h S L q Lz qz) (G : Mix N a b h) : Node N a b h S L q Lz qz where
  gpu z := { A.node.gpu z with side := A.gpuSide z G }
  rest E := { A.node.rest E with side := A.restSide E G }

/-- **The mix reaches the node only through its queries** (the oracle as a location, as B3's `OracleLocation`): for
every store, tile outputs and two mixes that agree at every point either location queried under the first, both
locations commit the same receipts under both. An interleaved run of the two locations over one mix, each answering
the other from what it computed and was sent, has this property; the game takes it as a hypothesis on the node. -/
def QueryDetermined (A : InteractiveNode N a b h S L q Lz qz) : Prop :=
  ∀ (E : Store N b) (z : Store N a) (G G' : Mix N a b h),
    (∀ x ∈ (A.under G).gpuQueries E z G, G' (x.2.1, x.2.2, z x.2.1) = G (x.2.1, x.2.2, z x.2.1)) →
    (∀ y ∈ (A.under G).restQueries E z G, G' (y.2.1, E y.2.1, y.2.2) = G (y.2.1, E y.2.1, y.2.2)) →
    (A.under G').gpuReceipts E z G' = (A.under G).gpuReceipts E z G ∧
      (A.under G').restReceipts E z G' = (A.under G).restReceipts E z G

/-- A node whose locations answer each other as they do at every mix (`e = 0`). -/
def ofNode (A : Node N a b h S L q Lz qz) : InteractiveNode N a b h S L q Lz qz :=
  ⟨A, fun z _ => (A.gpu z).side, fun E _ => (A.rest E).side⟩

end InteractiveNode

/-- The blocks the rest's link and tile work cover: `⌊((L_z + λ)·w + F_z·a)/(a·w)⌋`, the tile outputs that `L_z` bits
from the GPU, `λ` bits of slack and `F_z/w` tiles' work give it, at `a` bits each. -/
def covered (a w Lz Fz lam : ℕ) : ℕ := ((Lz + lam) * w + Fz * a) / (a * w)

end Accounting.BoundReceipt
