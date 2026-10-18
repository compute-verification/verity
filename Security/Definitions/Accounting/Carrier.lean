import Definitions.Accounting.RateAuditSketchDeployed
import Definitions.Accounting.RateAuditDeployed
import Definitions.Accounting.Gates.Codec
import Mathlib.Data.ENat.Lattice

/-!
# Carriers: device memory whose post-seed contents are a circuit's values (trusted)

A verbatim device (`RateAuditSketch.lean`) obtains a store sector by holding it on chip, reading it from device memory,
taking it over the side link or rebuilding it. Stacking is a way to rebuild: a carrier, device memory whose contents
were chosen after the seed (a KV page the audit credits, say), holds other blocks' encodings, and the device decodes
them from it instead of reading them. In the verbatim class a sector obtained by computing rather than by reading it at
its position is a `rebuild`, so a carrier's sectors count in a pass's rebuilds. This file is a carrier's vocabulary;
`Models/Accounting/Guarantees/Stacking.lean` bounds what carriers hold and charges it in every pass.

**Provenance.** A carrier's contents are values of a circuit (the served model as an exact-integer Program). In the
ideal encoding model the store's sectors are uniform, they are the blocks' encodings, and the circuit's values depend
on them in two ways only:

* `input x f`: `x` bits the prover feeds the circuit, any function `f` of the store (the prover knows all of it);
* `encode t x _ f`: `x ≤ m·256` bits of tile `t`'s encoding, any `x`-bit function `f` of it, which the circuit
  computes itself (exact-integer ops on the tile's payload, the tile's other bits among what it may hold), at a cost in
  budgeted gates that depends only on `x`.

A `Provenance` is the circuit seen through those two channels: every other gate is a function of the values so far,
folded into the continuations, so the gates themselves are not here. It covers every value a carrier's contents
depend on, back to values that depend on no store sector (a request's start, under the property's isolation).

**A carrier** of the store sectors `carried` is a provenance whose values are those sectors, whole and as they are, at
every store (the device's decoding is its last step).

**Relayed inputs.** In the general class (`RateAuditDeployed.lean`) the prover may feed a circuit inputs that depend on
a pass's certified stream and oracle, receipts it computed elsewhere among them: `Rate.PassProvenance` is a provenance
whose inputs are any function of the store, the stream and the oracle. A device reads such a carrier through its side
sources at odd addresses, which cost it no link: `PIChip.linkBitsEven` counts the side link without them.

**A carrier of a declared circuit's.** A credited KV page's payload is a declared circuit `W`'s values. Its
carrier's provenance is `W`'s computation from the chain's first write to the pass. So its inputs are `W`'s run inputs
in those windows of the audit, and its encodes are `W`'s budgeted gates there (`Provenance.OfCircuit`), which `W`'s
census counts per window (`Gates.census`).

**The tiles a pass read.** `readsIn` is `readsOn` restricted to the tiles `I j` of each pass `j`: the distinct
device-memory sectors read from those tiles. With `I j` the tiles of a declared circuit's blocks, it is what a pass
read of that circuit's data.

**A steered layout.** `steeredStore` is `coinStore` on a layout that reads the codec and the append coins as well as
the prefix: the memory map's schedule when inputs chosen after those coins move it.

**The encode cost.** An `encode` step of `x` bits is charged `encodeBitsCost`: the gate model's `Gates.encodeCost`
(`Definitions/Accounting/Gates`) of the codec words `x` bits fill, `⊤` when no circuit of the model's restricted class
computes them, so that such a step exceeds every budget. The provenances' `gates` count them.

**The append-coin rule.** The audit appends blocks during a session (a KV page), and remote_bandwidth's PROTOCOL.md
(§3–§4) orders each append's choices before its encoding. The append's root commits its blocks' digests `d_j`, and only
then does the verifier draw the append's coin `c_a`, once, which each block's key reads: `b_j(c_a) = H(seed ‖ j ‖ d_j ‖
c_a)`. The registration is the first append, and its coin is the codec seed. So the writers' choices for an append
(its payloads, their padding, the run inputs behind them, how many blocks, the schedule up to it) are a function of the
codec and the coins of the appends before it, and the codec encodes its blocks from those choices and the coins
(`appendCoinBlocks`). Which blocks a pass reads, its plan, is a layout of positions that no encoding chooses: the
memory map's schedule, which may depend on the session's coins before the pass.

**A store per pass.** The store a pass certifies is the one resident going into the step the pass is bound to, and it
may depend on the session's coins drawn before it. A pass's `SketchPrefix` is those coins: every earlier window's, its
own window's signs, and the window's earlier passes. Pass `j`'s store is the append-coin rule's at its prefix
(`coinStore`, `programmed`), and a window's correct receipts and reads are counted pass by pass, each on its own store
(`correctOn`, `readsOn`). With one store for every pass they are `correctAll` and `reads`. The audit rule behind the
prefix: a block appended after a pass's seed is in no plan of that pass, so a pass's store doesn't depend on its own
coins.
-/

namespace Accounting

/-- **The blocks a pass reads under the append-coin rule** (remote_bandwidth's PROTOCOL.md §3–§4). The store is made by
`T` appends in order, each of `M` block positions: the registration first, whose coin is the codec seed, then the
appends of blocks written after it. Append `a`'s choices `choose ω a κ_<a` (its payloads, their padding, the run inputs
behind them, how many blocks, the schedule up to it) read `ω` and the coins of the appends before it only, since its
own coin `κ a` is drawn once its root has committed every digest. The codec `enc` encodes each of its blocks from those
choices and the coins: a registered block under the seed's key, an appended one under `b_j(κ a)`. The pass reads the
blocks at the distinct positions `slot`, which no encoding chooses: block `u` of the pass is position `(slot u).2` of
append `(slot u).1`. -/
def appendCoinBlocks {Ω Ca Ch B : Type} {T M N : ℕ} (enc : Ω → Fin T → Ch → (Fin T → Ca) → Fin M → B)
    (choose : Ω → (a : Fin T) → ({b : Fin T // b < a} → Ca) → Ch) (slot : Fin N ↪ Fin T × Fin M)
    (y : Ω × (Fin T → Ca)) : Fin N → B :=
  fun u => enc y.1 (slot u).1 (choose y.1 (slot u).1 fun b => y.2 b.1) y.2 (slot u).2

end Accounting

namespace Accounting.RateSketch

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

/-- The budgeted gates the encodes cost at store `E`, at `cost x` per `encode` step of `x` bits. The circuit's other
budgeted gates only add to its count, so a budget on its budgeted gates bounds this. -/
def gates (cost : ℕ → ℕ∞) (E : SketchStore N m) : Provenance N m α → ℕ∞
  | done _ => 0
  | input _ f k => (k (f E)).gates cost E
  | encode t x _ f k => (k (f fun u => E (t, u))).gates cost E + cost x

/-- **A provenance that is `W`'s over the windows `I`** of the audit `A`: at every store its inputs are at most the
bits of `W`'s run inputs in those windows, and its encodes, at `cost x` for `x` bits, cost at most `W`'s budgeted units
there (`Gates.census`, property/v1's units). -/
def OfCircuit {ι : Type} (K : Provenance N m α) (cost : ℕ → ℕ∞) (W : Gates.Dag ι) (A : Gates.Audit)
    (I : Finset ℕ) : Prop :=
  ∀ E, K.inputs E ≤ ∑ v ∈ I, (Gates.census W fun t => A.win t = v).runBits ∧
    K.gates cost E ≤ ((∑ v ∈ I, (Gates.census W fun t => A.win t = v).budgeted : ℕ) : ℕ∞)

end Provenance

/-- A carrier of the store sectors `carried`: a provenance whose values are those sectors, whole and as they are, at
every store. -/
structure Carrier (N m : ℕ) where
  carried : Finset (Fin N × Fin m)
  provenance : Provenance N m (carried → Sector)
  holds : ∀ E, provenance.run E = fun p => E p.1

/-- The coins a session has drawn before pass `j` of window `w`: every earlier window's, window `w`'s signs, and its
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
sectors its block's. `y` is the codec and every append's coin. The append coins are drawn during the session, but
independently of its coins, so they are drawn with the codec here; their order is `choose`'s, each append's choices
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

/-- A stacked session's coins at the deployed parameters (16 sketch rows of 128-sector tiles, 512-bit tags, 2,773
openings a window): the codec and every append's coin, then its `n` windows'. -/
abbrev StackedCoins (Ω Ca : Type) (T n P N ρ G : ℕ) : Type :=
  (Ω × (Fin T → Ca)) × (Fin n → SketchWindowCoins P N 128 ρ 512 16 G 2773)

/-- **The stores window `w`'s passes certify**: pass `j`'s is `view c ω` at the prefix `c` before it, `ω` drawn
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

end Accounting.RateSketch

namespace Accounting.Rate

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

/-- The bits encoded: `x` per `encode` step of `x` bits. -/
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

end Accounting.Rate

namespace Accounting

/-- **The encode cost of `x` bits of a block's encoding**, in budgeted units: `Gates.encodeCost` (the fewest budgeted
units of a restricted circuit with no run input, at most `B` operand bits a budgeted gate, that computes that many codec
words over `F` at `k` rounds on more than `εnum/εden` of the draws, `⊤` when there is none) of the `⌈x / (8·bytes)⌉`
words that `x` bits fill. A function of a block's encoding is charged as the whole words its bits would fill; one a
circuit gets without computing those words (a word's parity, the Legendre symbol of its last round's input, which is
why the words' targets leave it out) enters a provenance as an input. -/
noncomputable def encodeBitsCost (F : Gates.Field) (k B εnum εden x : ℕ) : ℕ∞ :=
  Gates.encodeCost F k B εnum εden ((x + 8 * F.bytes - 1) / (8 * F.bytes))

end Accounting
