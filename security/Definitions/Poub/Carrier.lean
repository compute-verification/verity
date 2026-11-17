module

public import Definitions.Poub.RateAuditSketchDeployed
public import Definitions.Poub.RateAuditDeployed
public import Definitions.ResourceAccounting.Gates.Codec
public import Definitions.ResourceAccounting.Requirement
import Mathlib.Basic.ENNReal.Real

/-! Carriers, device memory whose post-seed contents are a circuit's values, and the vocabulary stacking charges them
in. -/

@[expose] public section

namespace ResourceAccounting

/-- **The blocks a pass reads under the append-coin rule** (poub's PROTOCOL.md §3–§4). The store is made by
`T` appends in order, each of `M` block positions: the registration first, whose coin is the codec seed, then the
appends of blocks written after it. Append `a`'s choices `choose ω a κ_<a` (its payloads, their padding, the run inputs
behind them, how many blocks, the schedule up to it) read `ω` and the coins of the appends before it only, since its
own coin `κ a` is sampled once its root has committed every digest. The codec `enc` encodes each of its blocks from those
choices and the coins: a registered block under the seed's key, an appended one under `b_j(κ a)`. The pass reads the
blocks at the distinct positions `slot`, which no encoding chooses: block `u` of the pass is position `(slot u).2` of
append `(slot u).1`. -/
def appendCoinBlocks {Ω Ca Ch B : Type} {T M N : ℕ} (enc : Ω → Fin T → Ch → (Fin T → Ca) → Fin M → B)
    (choose : Ω → (a : Fin T) → ({b : Fin T // b < a} → Ca) → Ch) (slot : Fin N ↪ Fin T × Fin M)
    (y : Ω × (Fin T → Ca)) : Fin N → B :=
  fun u => enc y.1 (slot u).1 (choose y.1 (slot u).1 fun b => y.2 b.1) y.2 (slot u).2

end ResourceAccounting

namespace ResourceAccounting.RateSketch

open Pous (Bits)

/-- How a circuit's values over the store, `N` tiles of `m` sectors, came to depend on it, ending in the values (of
type `α`). Each step waits on `x` bits that depend on the store: an input the prover feeds, any function of the store
(`input`), or `x ≤ m·256` bits of a tile's encoding, any function of it, which the circuit computes (`encode`). -/
inductive Provenance (N m : ℕ) (α : Type) : Type where
  | done : α → Provenance N m α
  | input : (x : ℕ) → (SketchStore N m → Bits x) → (Bits x → Provenance N m α) → Provenance N m α
  | encode : Fin N → (x : ℕ) → x ≤ m * 256 → ((Fin m → Sector) → Bits x) → (Bits x → Provenance N m α) →
      Provenance N m α

namespace Provenance

variable {N m : ℕ} {α : Type}

/-- The circuit's values at store `E`. -/
def run (E : SketchStore N m) : Provenance N m α → α
  | done a => a
  | input _ f k => (k (f E)).run E
  | encode t _ _ f k => (k (f fun u => E (t, u))).run E

/-- The bits fed as inputs at store `E`: `x` per `input` step of `x` bits. -/
def inputs (E : SketchStore N m) : Provenance N m α → ℕ
  | done _ => 0
  | input x f k => (k (f E)).inputs E + x
  | encode t _ _ f k => (k (f fun u => E (t, u))).inputs E

/-- The bits encoded at store `E`: `x` per `encode` step of `x` bits. -/
def encoded (E : SketchStore N m) : Provenance N m α → ℕ
  | done _ => 0
  | input _ f k => (k (f E)).encoded E
  | encode t x _ f k => (k (f fun u => E (t, u))).encoded E + x

/-- The nonlinear units the encodes cost at store `E`, at `cost x` per `encode` step of `x` bits. The circuit's other
gates only add to its count, so a budget on its nonlinear units bounds this. -/
def gates (cost : ℕ → ℕ∞) (E : SketchStore N m) : Provenance N m α → ℕ∞
  | done _ => 0
  | input _ f k => (k (f E)).gates cost E
  | encode t x _ f k => (k (f fun u => E (t, u))).gates cost E + cost x

/-- **A provenance that is `W`'s over the windows `I`** of the audit `A`: at every store its inputs are at most the
bits of `W`'s run inputs in those windows, and its encodes, at `cost x` for `x` bits, cost at most `W`'s nonlinear
units there: its budgeted units (`Gates.census`, property/v1's units) and its data products (`Gates.productsAt`). -/
def OfCircuit {ι : Type} (K : Provenance N m α) (cost : ℕ → ℕ∞) (W : Gates.IntDag ι) (A : Gates.Audit)
    (I : Finset ℕ) : Prop :=
  ∀ E, K.inputs E ≤ ∑ v ∈ I, (Gates.census W fun t => A.win t = v).runBits ∧
    K.gates cost E ≤
      ((∑ v ∈ I, ((Gates.census W fun t => A.win t = v).budgeted + Gates.productsAt W fun t => A.win t = v) : ℕ) : ℕ∞)

end Provenance

/-- A carrier of the store sectors `carried`: a provenance whose values are those sectors, whole and as they are, at
every store. -/
structure Carrier (N m : ℕ) where
  carried : Finset (Fin N × Fin m)
  provenance : Provenance N m (carried → Sector)
  holds : ∀ E, provenance.run E = fun p => E p.1

/-- The coins a session has sampled before pass `j` of window `w`: every earlier window's, window `w`'s signs, and its
passes before `j`. -/
structure SketchPrefix (n P N m ρ h ks G d : ℕ) where
  w : Fin n
  windows : {v : Fin n // v < w} → SketchWindowCoins P N m ρ h ks G d
  signs : SketchSigns ks m
  j : Fin P
  passes : {i : Fin P // i < j} → SketchPassCoins N ρ h ks G

/-- The prefix before pass `j` of window `w`, at the session's coins `x`. -/
def SketchPrefix.of {n P N m ρ h ks G d : ℕ} (x : Fin n → SketchWindowCoins P N m ρ h ks G d) (w : Fin n)
    (j : Fin P) : SketchPrefix n P N m ρ h ks G d :=
  ⟨w, fun v => x v.1, (x w).1, j, fun i => (x w).2.1 i.1⟩

/-- **A pass's store under the append-coin rule**, at the prefix `c` of the session's coins before it: the blocks at
the layout `slot c` (`appendCoinBlocks`), the writers' choices `choose c` reading the prefix too, each tile's `m`
sectors its block's. `y` is the codec and every append's coin. The append coins are flipped during the session, but
independently of its coins, so they are sampled with the codec here; their order is `choose`'s, each append's choices
reading only the coins before it. -/
def coinStore {n P N m ρ h ks G d T M : ℕ} {Ω Ca Ch : Type}
    (enc : Ω → Fin T → Ch → (Fin T → Ca) → Fin M → (Fin m → Sector))
    (choose : SketchPrefix n P N m ρ h ks G d → Ω → (a : Fin T) → ({b : Fin T // b < a} → Ca) → Ch)
    (slot : SketchPrefix n P N m ρ h ks G d → Fin N ↪ Fin T × Fin M) (c : SketchPrefix n P N m ρ h ks G d)
    (y : Ω × (Fin T → Ca)) : SketchStore N m :=
  fun p => appendCoinBlocks enc (choose c) (slot c) y p.1 p.2

/-- **A pass's store under the append-coin rule on a steered layout**: `coinStore`, with the layout `layout c y` read
at the codec and the append coins `y` as well as the prefix `c`, so that it may follow inputs chosen after those coins.
-/
def steeredStore {n P N m ρ h ks G d T M : ℕ} {Ω Ca Ch : Type}
    (enc : Ω → Fin T → Ch → (Fin T → Ca) → Fin M → (Fin m → Sector))
    (choose : SketchPrefix n P N m ρ h ks G d → Ω → (a : Fin T) → ({b : Fin T // b < a} → Ca) → Ch)
    (layout : SketchPrefix n P N m ρ h ks G d → Ω × (Fin T → Ca) → Fin N ↪ Fin T × Fin M)
    (c : SketchPrefix n P N m ρ h ks G d) (y : Ω × (Fin T → Ca)) : SketchStore N m :=
  fun p => appendCoinBlocks enc (choose c) (layout c y) y p.1 p.2

/-- **A pass's appended tiles**: the tiles `i` that the layout `layout c y` puts at a position of an append after the
registration (append `0`). Their blocks are keyed under their append's coin; a registered block's key is fixed for the
registration. -/
def appendedTiles {Pfx Y : Type} {N T M : ℕ} (layout : Pfx → Y → Fin N ↪ Fin T × Fin M) (c : Pfx) (y : Y) :
    Finset (Fin N) :=
  Finset.univ.filter fun i => ((layout c y i).1 : ℕ) ≠ 0

/-- **Each append's blocks are read within `L` windows of the one it joined**: the game's form of PoUB's
`LifetimeCapped`. At the session's coins `x` (the codec and the append coins `x.1`, the windows' `x.2`), every pass
whose layout puts a block of an append `t` after the registration in its store is in a window `w` with
`joined x t ≤ w < joined x t + L`. `joined x t` is the window append `t` joined, the one being filled at the first pass
after it. -/
def LifetimeCapped {n P N m ρ h ks G d T M : ℕ} {Y : Type}
    (layout : SketchPrefix n P N m ρ h ks G d → Y → Fin N ↪ Fin T × Fin M)
    (joined : Y × (Fin n → SketchWindowCoins P N m ρ h ks G d) → Fin T → ℕ) (L : ℕ) : Prop :=
  ∀ (x : Y × (Fin n → SketchWindowCoins P N m ρ h ks G d)) (w : Fin n) (j : Fin P),
    ∀ i ∈ appendedTiles layout (SketchPrefix.of x.2 w j) x.1,
      joined x (layout (SketchPrefix.of x.2 w j) x.1 i).1 ≤ (w : ℕ) ∧
        (w : ℕ) < joined x (layout (SketchPrefix.of x.2 w j) x.1 i).1 + L

namespace Chip

variable {N m ρ h : ℕ} {X Dg α : Type}

/-- The sectors rebuilt in the tiles `U`: one per `rebuild` step at a sector of a tile in `U`. At every tile it is
`rebuilds`. -/
def rebuildsIn (U : Finset (Fin N)) (O : Oracle N ρ h X Dg) (D : Data N m ρ X) : Chip N m ρ h X Dg α → ℕ
  | done _ => 0
  | tag x k => (k (O x)).rebuildsIn U O D
  | mem p k => (k (D (.inl p))).rebuildsIn U O D
  | link p k => (k (D p)).rebuildsIn U O D
  | rebuild p k => (k (D (.inl p))).rebuildsIn U O D + if p.1 ∈ U then 1 else 0

end Chip

/-- A stacked session's coins at the deployed parameters (16 sketch rows of 128-sector tiles, 512-bit tags, 2,773
openings a window): the codec and every append's coin, then its `n` windows'. -/
abbrev StackedCoins (Ω Ca : Type) (T n P N ρ G : ℕ) : Type :=
  (Ω × (Fin T → Ca)) × (Fin n → SketchWindowCoins P N 128 ρ 512 16 G 2773)

/-- **The stores window `w`'s passes certify**: pass `j`'s is `view c ω` at the prefix `c` before it, `ω` sampled
independently of the session's coins (in the stacked guarantees, `coinStore` at the codec and the append coins). -/
def programmed {n P N m ρ h ks G d : ℕ} {Ω : Type} (view : SketchPrefix n P N m ρ h ks G d → Ω → SketchStore N m)
    (ω : Ω) (x : Fin n → SketchWindowCoins P N m ρ h ks G d) (w : Fin n) : Fin P → SketchStore N m :=
  fun j => view (.of x w j) ω

namespace SketchDeployedWindow

variable {P N m ρ h S L q ks G : ℕ}

/-- The pooled receipts whose sketch tags are correct, each pass on its own store `E j`. -/
def correctOn (A : SketchDeployedWindow P N m ρ h S L q ks G) (E : Fin P → SketchStore N m)
    (σ : SketchSigns ks m) (ω : Fin P → SketchPassCoins N ρ h ks G) : Finset (Fin P × Fin N) :=
  Finset.univ.filter fun p =>
    let D := Data.ofParts (E p.1) (ω p.1).1
    let B := A.dev p.1 (E p.1) σ ω
    let O := A.oracle p.1 (E p.1) σ ω
    p.2 ∈ B.correct sketch σ O D

/-- Device-memory sectors read in the window, in bits, summed over its passes, each on its own store `E j`. -/
def readsOn (A : SketchDeployedWindow P N m ρ h S L q ks G) (E : Fin P → SketchStore N m)
    (σ : SketchSigns ks m) (ω : Fin P → SketchPassCoins N ρ h ks G) : ℕ :=
  ∑ j,
    let D := Data.ofParts (E j) (ω j).1
    (A.dev j (E j) σ ω).footprint σ (A.oracle j (E j) σ ω) D

/-- Device-memory sectors read in the window from the tiles `I j` of each pass `j`, in bits, each pass on its own
store `E j`: `readsOn` restricted to those tiles. -/
def readsIn (A : SketchDeployedWindow P N m ρ h S L q ks G) (E : Fin P → SketchStore N m)
    (σ : SketchSigns ks m) (ω : Fin P → SketchPassCoins N ρ h ks G) (I : Fin P → Finset (Fin N)) : ℕ :=
  ∑ j,
    let D := Data.ofParts (E j) (ω j).1
    ((((A.dev j (E j) σ ω).prog σ D).mems (A.oracle j (E j) σ ω) D).filter fun p => p.1 ∈ I j).card * 256

end SketchDeployedWindow

end ResourceAccounting.RateSketch

namespace ResourceAccounting.Rate

open Pous (Bits)

/-- How a circuit's values in a pass came to depend on the pass's data, ending in the values (of type `α`):
`RateSketch.Provenance` for the general class. Each step waits on `x` bits: an input the prover feeds, any function of
the store, the certified stream and the oracle (`input`), or `x ≤ b` bits of word `j`'s encoding, any function of it,
which the circuit computes (`encode`). -/
inductive PassProvenance (N b h ρ : ℕ) (α : Type) : Type where
  | done : α → PassProvenance N b h ρ α
  | input : (x : ℕ) → (Store N b → Bits ρ → Oracle N b h ρ → Bits x) → (Bits x → PassProvenance N b h ρ α) →
      PassProvenance N b h ρ α
  | encode : Fin N → (x : ℕ) → x ≤ b → (Bits b → Bits x) → (Bits x → PassProvenance N b h ρ α) →
      PassProvenance N b h ρ α

namespace PassProvenance

variable {N b h ρ : ℕ} {α : Type}

/-- The circuit's values at store `E`, stream `R` and oracle `O`. -/
def run (E : Store N b) (R : Bits ρ) (O : Oracle N b h ρ) : PassProvenance N b h ρ α → α
  | done a => a
  | input _ f k => (k (f E R O)).run E R O
  | encode j _ _ f k => (k (f (E j))).run E R O

/-- The bits fed as inputs: `x` per `input` step of `x` bits. -/
def inputs (E : Store N b) (R : Bits ρ) (O : Oracle N b h ρ) : PassProvenance N b h ρ α → ℕ
  | done _ => 0
  | input x f k => (k (f E R O)).inputs E R O + x
  | encode j _ _ f k => (k (f (E j))).inputs E R O

def encoded (E : Store N b) (R : Bits ρ) (O : Oracle N b h ρ) : PassProvenance N b h ρ α → ℕ
  | done _ => 0
  | input _ f k => (k (f E R O)).encoded E R O
  | encode j x _ f k => (k (f (E j))).encoded E R O + x

/-- The budgeted gates the encodes cost, at `cost x` per `encode` step of `x` bits. -/
def gates (cost : ℕ → ℕ∞) (E : Store N b) (R : Bits ρ) (O : Oracle N b h ρ) : PassProvenance N b h ρ α → ℕ∞
  | done _ => 0
  | input _ f k => (k (f E R O)).gates cost E R O
  | encode j x _ f k => (k (f (E j))).gates cost E R O + cost x

end PassProvenance

namespace PIChip

variable {N b h ρ : ℕ} {α : Type}

/-- The bits that came in over the side link at even addresses and from the certified stream, having sent `v` so far:
`linkBits` without the side reads at odd addresses. -/
def linkBitsEven (w : IWorld N b h ρ) : List Bool → PIChip N b h ρ α → ℕ
  | _, done _ => 0
  | v, tag x k => (k (w.oracle x)).linkBitsEven w v
  | v, ingress a k => (k (w.mem a)).linkBitsEven w v
  | v, side a k => (k (w.side v a)).linkBitsEven w v + if a % 2 = 0 then 1 else 0
  | v, stream i k => (k (w.stream i)).linkBitsEven w v + 1
  | v, complete j _ _ g k => (k (g (w.store j))).linkBitsEven w v
  | v, send β p => p.linkBitsEven w (β :: v)

end PIChip

end ResourceAccounting.Rate

namespace ResourceAccounting

/-- **The encode cost of `x` bits of a block's encoding**, in nonlinear units: `Gates.encodeCost` (the fewest nonlinear
units, budgeted units and data products, of a restricted circuit with no run input, at most `B` operand bits a budgeted
gate, that computes that many codec words over `F` at `k` rounds on more than `εnum/εden` of the samples, `⊤` when there
is none) of the `⌈x / (8·bytes)⌉` words that `x` bits fill. A function of a block's encoding is charged as the whole
words its bits would fill; one a circuit gets without computing those words (a word's parity, the Legendre symbol of its
last round's input, which is why the words' targets leave it out) enters a provenance as an input. -/
noncomputable def encodeBitsCost (F : Gates.Field) (k B εnum εden x : ℕ) : ℕ∞ :=
  Gates.encodeCost F k B εnum εden ((x + 8 * F.bytes - 1) / (8 * F.bytes))

end ResourceAccounting

namespace ResourceAccounting.Requirement

variable {L G : Type} {n P : ℕ}

/-- **Pass `j` of window `w` on device `d`'s clock**: the `τ` ticks from the one at which `d` begins the engine step the
pass is bound to (`bind w j`), the pass's deadline. -/
def Placement.passTicks (p : Placement L G) (bind : Fin n → Fin P → ℕ) (τ : ℕ) (w : Fin n) (j : Fin P) (d : L) :
    Finset ℕ :=
  Finset.Ico (p.ticks d (bind w j)) (p.ticks d (bind w j) + τ)

/-- **Window `w` on device `d`'s clock**: the ticks of its passes. -/
def Placement.executionPhaseTicks (p : Placement L G) (bind : Fin n → Fin P → ℕ) (τ : ℕ) (w : Fin n) (d : L) : Finset ℕ :=
  Finset.univ.biUnion fun j => p.passTicks bind τ w j d

/-- **The passes keep their deadlines on every device's clock**: on each device, no tick is in two windows or in two
passes of one window, so each of a device's ticks is taken by at most one pass. -/
def Placement.Deadlines (p : Placement L G) (bind : Fin n → Fin P → ℕ) (τ : ℕ) : Prop :=
  ∀ d, (Pairwise fun w w' => Disjoint (p.executionPhaseTicks bind τ w d) (p.executionPhaseTicks bind τ w' d)) ∧
    ∀ w, Pairwise fun j j' => Disjoint (p.passTicks bind τ w j d) (p.passTicks bind τ w j' d)

end ResourceAccounting.Requirement
