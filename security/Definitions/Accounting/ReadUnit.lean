module

public import Definitions.Core.Interaction
public import Definitions.Core.Partition
public import Definitions.Hash.EmbeddingRead
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Topology.Sheaves.Init

/-!
# Read units: one row of a registered table, at an index another unit picks

A read unit is a unit of the circuit that returns one row of a registered table. The table is committed as a frame-v3
tree of hm96 row leaves (`Hash.RowLeaf`), one leaf per row, and the unit opens the leaf at the index its index gates
hold. In the wired flattening those gates are its picker's outputs: an expert router picks each expert weight table's
row, a sparse-attention indexer picks key and value rows, and a request's token picks its embedding row. A read whose
wired index row is not its picker's output row is incorrect in the wired flattening (`Spot.Wires.Summary.drawnOk`
holds the wires of the sampled proof blocks), so the index a correct read uses is its picker's. The read's reference is
`Hash.EmbeddingRead`: below the table's count it returns the witness row and accepts where the row's leaf opens at the
index; at or past the count it returns the fill row and accepts.

`Ports` names a read unit's gates and decodes their values, each gate's bit 0 (`bit`): the index (32 bits, least
significant first), the table's root and domain id (512 bits each, in leaf order: eight 64-bit words, each word's bits
least significant first and its bytes big-endian), the witness row (16-bit words, each least significant bit first), the
salt (`Hash.Hm96.SALT_BYTES` bytes, in a digest's layout) and the sibling path (one digest per level of the table's tree),
the row the read returns and its verdict bit. The read's template fixes its table's count and each rank's position
(`Ports.domainAt`); a dense table puts each row's leaf at its own rank.

What a table read's soundness takes of a read unit `u` at a transcript `X`:

* `GatesRefine`: `u`'s gates compute its reference, in `u`'s local evaluation (`Verity.Partition.localEval`), with the
  gates of its root, domain id and verdict committed where they lie in `u`. It is a fact about the circuit's gates,
  which the circuit alone decides, and not a cryptographic assumption.
* `RegisteredRows`: `u`'s registered rows hold in `X`: its root and domain id are its table's, its verdict bit is 1, and
  the registered constants among its inputs and outputs hold. A unit bound in `X` under the deployed openings (outside
  `Accounting.unbound`) has them wherever the openings pin those gates.

`Binds` is what follows, in `u`'s local evaluation: the row `u` returns is its table's row at its index, for every
opening of that leaf the table's root accepts, or SHA-512 collides on what the two checks hashed. The local evaluation
is the row `u`'s own gates consume, and where the row's gates are committed and `u` is correct it is the transcript's.
`Reads` collects a circuit's table reads, each in its unit with its table, and `Reads.finder` is the collision finder
their binding reads `Claims.ExpectedCR` at.
-/

@[expose] public section

namespace Accounting.ReadUnit

variable {C : Verity.Circuit}

/-- Bit 0 of gate `g`'s value in `X`: a read unit's ports are one-bit gates. -/
def bit (X : C.Trace) (g : Fin C.N) : Bool := (X g).getLsbD 0

/-- The bits the gates `gs` hold in `X`, in order. -/
def bits {k : ℕ} (X : C.Trace) (gs : Fin k → Fin C.N) : List Bool := List.ofFn fun i => bit X (gs i)

/-- A number from its bits, least significant first. -/
def num : List Bool → ℕ
  | [] => 0
  | b :: bs => b.toNat + 2 * num bs

/-- **Bytes from bits in leaf order**: each 64-bit word's bits least significant first, its eight bytes big-endian. -/
def bytesOf (bs : List Bool) : Hash.Msg :=
  (List.range (bs.length / 64)).flatMap fun w =>
    (List.range 8).map fun m => UInt8.ofNat (num ((bs.drop (64 * w + 8 * (7 - m))).take 8))

/-- **A digest from its 512 bits in leaf order**. -/
def digestOf (bs : List Bool) : Verity.Digest := fun i => (bytesOf bs).getD i.val 0

/-- **A read unit's gates**, for a table of `count` rows of `words` 16-bit words whose rank `r` row's leaf sits at
`position r`: the index, the table's root and domain id, the witness row, salt and sibling path, the row the unit returns
(`h`) and its verdict bit (`ok`). -/
structure Ports (C : Verity.Circuit) where
  count : ℕ
  words : ℕ
  position : ℕ → ℕ
  tok : Fin 32 → Fin C.N
  root : Fin 512 → Fin C.N
  domain : Fin 512 → Fin C.N
  row : Fin (16 * words) → Fin C.N
  salt : Fin (8 * Hash.Hm96.SALT_BYTES) → Fin C.N
  path : Fin (Hash.Tree.depth count) → Fin 512 → Fin C.N
  h : Fin (16 * words) → Fin C.N
  ok : Fin C.N

namespace Ports

variable (q : Ports C) (X : C.Trace)

/-- The index the read's index gates hold. -/
def tokOf : ℕ := num (bits X q.tok)

/-- The root the read's root gates hold. -/
def rootOf : Verity.Digest := digestOf (bits X q.root)

/-- The domain id the read's domain gates hold. -/
def domainOf : Verity.Digest := digestOf (bits X q.domain)

/-- The opening the read's witness gates hold: the row, the salt and one sibling digest per level. -/
def opening : Hash.RowLeaf.Opening :=
  ⟨bits X q.row, bytesOf (bits X q.salt), List.ofFn fun j => digestOf (bits X (q.path j))⟩

/-- The row the read returns. -/
def hOf : Hash.Row := bits X q.h

/-- The read's verdict bit. -/
def okOf : Bool := bit X q.ok

/-- The read's table domain under the domain id `id`: its count and positions, which the read's template fixes. -/
def domainAt (id : Verity.Digest) : Hash.Tree.Domain := ⟨id, q.count, q.position⟩

/-- The row-leaf verifier of the read's table under the domain id `id` and the hm96 key `key`. -/
def verifier (key : Hash.Msg) (id : Verity.Digest) : Hash.RowLeaf.Verifier :=
  Hash.EmbeddingRead.verifier (q.domainAt id) key

/-- **The read's ports hold its reference** in `Y`, under the hashes `H` and `G` and the hm96 key `key`: the row the
read returns is `Hash.EmbeddingRead.h`'s on its ports' values, and its verdict bit is set only where
`Hash.EmbeddingRead.ok` holds of them, in its table's domain under the domain id its port holds. -/
def Computes (H G : Hash.Msg → Verity.Digest) (key : Hash.Msg) (Y : C.Trace) : Prop :=
  q.hOf Y = Hash.EmbeddingRead.h q.count q.words (q.tokOf Y) (q.opening Y) ∧
    (q.okOf Y = true →
      Hash.EmbeddingRead.ok H G (q.domainAt (q.domainOf Y)) key (q.rootOf Y) q.count (q.tokOf Y) (q.opening Y))

end Ports

/-- **The read's gates compute its reference** (`Hash.EmbeddingRead`), under the hashes `H` and `G`, the hm96 key `key`
and the circuit's registered constants `consts`: at every transcript `X` where unit `u` of `P` is correct and the
registered constants among `u`'s inputs and outputs hold, the read's ports hold its reference (`Ports.Computes`) in
`u`'s local evaluation (`Verity.Partition.localEval`: `u`'s gates recomputed from what they read outside `u`), and the
gates of its root, domain id and verdict that lie in `u` are committed.

The row is bound where `u`'s gates compute it, whether or not its gates are committed: a block that reads a table row
and consumes it inside itself, as a read fused into the product that reads its row does, binds the row its product reads
as a block that commits the row does. The root, domain id and verdict are what the registered rows hold in the
transcript, so those of their gates in `u` are committed, where a correct unit's transcript holds its evaluation.

This is a fact about the circuit's gates, which the circuit alone decides, and not a cryptographic assumption: it names
no adversary and no probability, and holds or fails of the registered circuit whatever a prover does. What proves it is a
proof about the read's gates against the reference (the select, the row's segment digest, the hm96 leaf, the frame-v3
path and the SHA-512 compressions under them), or an instance of a lowering-correctness theorem for the template the unit
instantiates. -/
structure GatesRefine (H G : Hash.Msg → Verity.Digest) (key : Hash.Msg) (consts : Fin C.N → Option Bool) {n : ℕ}
    (P : Verity.Partition C n) (u : Fin n) (q : Ports C) : Prop where
  computes : ∀ X : C.Trace, P.Correct X u → (∀ g ∈ P.io u, ∀ b, consts g = some b → bit X g = b) →
    q.Computes H G key (P.localEval X u)
  root_committed : ∀ i, P.unit (q.root i) = some u → q.root i ∈ P.committed
  domain_committed : ∀ i, P.unit (q.domain i) = some u → q.domain i ∈ P.committed
  ok_committed : P.unit q.ok = some u → q.ok ∈ P.committed

/-- **The read's registered rows hold** in `X`, for the table under the root `r` and the domain id `d`: its root's and
domain id's gates hold them, its verdict bit is held to 1, and the registered constants `consts` among its inputs and
outputs hold. -/
structure RegisteredRows (consts : Fin C.N → Option Bool) {n : ℕ} (P : Verity.Partition C n) (u : Fin n) (q : Ports C)
    (r d : Verity.Digest) (X : C.Trace) : Prop where
  root_eq : q.rootOf X = r
  domain_eq : q.domainOf X = d
  ok_eq : q.okOf X = true
  consts_hold : ∀ g ∈ P.io u, ∀ b, consts g = some b → bit X g = b

/-- **The read returns its table's row at its index** in `X`, for the table under the root `r`, the domain id `d` and
the hm96 key `key`: for every opening `o₀` the root accepts at the index the read's index gates hold, the row the read
returns is `o₀`'s, or `H` or `G` collides on what the read's check and `o₀`'s hashed. -/
def Binds (H G : Hash.Msg → Verity.Digest) (key : Hash.Msg) (q : Ports C) (r d : Verity.Digest) (X : C.Trace) :
    Prop :=
  ∀ o₀ : Hash.RowLeaf.Opening, Hash.RowLeaf.accepts H G (q.verifier key d) r (q.tokOf X) o₀ →
    q.hOf X = o₀.row ∨
      Hash.CollisionIn H (Hash.RowLeaf.outerHashed H G (q.verifier key d) (q.tokOf X) (q.opening X) ++
        Hash.RowLeaf.outerHashed H G (q.verifier key d) (q.tokOf X) o₀) ∨
      Hash.CollisionIn G (Hash.RowLeaf.innerHashed G (q.verifier key d) (q.opening X) ++
        Hash.RowLeaf.innerHashed G (q.verifier key d) o₀)

/-- **A circuit's table reads**: the reads `reads`, each in the unit `unit i` of `n`, at its ports, with its table: the
root, the domain id and the opening of each row. A unit may hold several reads, as a proof block of stacked copies
holds one a copy. -/
structure Reads (C : Verity.Circuit) (n : ℕ) (ι : Type) where
  reads : Finset ι
  unit : ι → Fin n
  port : ι → Ports C
  root : ι → Verity.Digest
  domain : ι → Verity.Digest
  table : ι → ℕ → Hash.RowLeaf.Opening

namespace Reads

variable {n : ℕ} {ι : Type} (R : Reads C n ι)

/-- **Each read's table is committed under its root**: the root accepts the table's opening of each of its rows, under
the hashes `H` and `G` and the hm96 key `key`. -/
def Committed (H G : Hash.Msg → Verity.Digest) (key : Hash.Msg) : Prop :=
  ∀ i ∈ R.reads, ∀ k < (R.port i).count,
    Hash.RowLeaf.accepts H G ((R.port i).verifier key (R.domain i)) (R.root i) k (R.table i k)

/-- The index read `i`'s index gates hold in `X`. -/
def index (i : ι) (X : C.Trace) : ℕ := (R.port i).tokOf X

/-- **What read `i`'s check and its table's check hashed** in `X`, under one hash `H`: the read's opening and its
table's opening at the read's index, each checked there. -/
def hashedAt (H : Hash.Msg → Verity.Digest) (key : Hash.Msg) (i : ι) (X : C.Trace) : List Hash.Msg :=
  Hash.RowLeaf.outerHashed H H ((R.port i).verifier key (R.domain i)) (R.index i X) ((R.port i).opening X) ++
    Hash.RowLeaf.outerHashed H H ((R.port i).verifier key (R.domain i)) (R.index i X) (R.table i (R.index i X)) ++
    (Hash.RowLeaf.innerHashed H ((R.port i).verifier key (R.domain i)) ((R.port i).opening X) ++
      Hash.RowLeaf.innerHashed H ((R.port i).verifier key (R.domain i)) (R.table i (R.index i X)))

open Classical in
/-- **What the drawn reads hashed** at the outcome `ω`: `hashedAt` for each read whose unit is drawn, in its unit's
local evaluation of the transcript `X ω` under the partition `P`. -/
noncomputable def drawnHashed {Ω : Type} (P : Verity.Partition C n) (H : Hash.Msg → Verity.Digest) (key : Hash.Msg)
    (drawn : Ω → Finset (Fin n)) (X : Ω → C.Trace) (ω : Ω) : List Hash.Msg :=
  (R.reads.filter (R.unit · ∈ drawn ω)).toList.flatMap fun i => R.hashedAt H key i (P.localEval (X ω) (R.unit i))

open Classical in
/-- **The table reads' collision finder**: on the outcome `ω`, two messages the drawn reads hashed that `H` maps to one
digest, if there are two, at the cost `cost ω`: the SHA-512 calls of reaching `ω`'s transcript and hashing what its
drawn reads hashed. -/
noncomputable def finder {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Verity.Partition C n)
    (H : Hash.Msg → Verity.Digest) (key : Hash.Msg) (drawn : Ω → Finset (Fin n)) (X : Ω → C.Trace)
    (cost : Ω → ℕ) : Claims.Finder where
  S := Ω
  out ω :=
    if h : ∃ x ∈ R.drawnHashed P H key drawn X ω, ∃ y ∈ R.drawnHashed P H key drawn X ω,
        Hash.Collision (fun m => (H m).bytes) x y then
      some (h.choose, h.choose_spec.2.choose)
    else none
  cost := cost

end Reads

end Accounting.ReadUnit
