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
tree of hm96 row leaves (`Hash.RowLeaf`), one leaf per row, and the unit opens the leaf at an index its picker computes:
gates elsewhere in the circuit whose bits name the row. An expert router picks each expert weight table's row, a
sparse-attention indexer picks key and value rows, and a token picks its embedding row. The read's reference is
`Hash.EmbeddingRead`: below the table's count it returns the witness row and accepts where the row's leaf opens at the
index; at or past the count it returns the fill row and accepts.

`Ports` names a read unit's gates and decodes their values, each gate's bit 0 (`bit`): the index (32 bits, least
significant first), the table's root and domain id (512 bits each, in leaf order: eight 64-bit words, each word's bits
least significant first and its bytes big-endian), the witness row (16-bit words, each least significant bit first), the
salt (`Hash.Hm96.SALT_BYTES` bytes, in a digest's layout) and the sibling path (one digest per level of the table's tree),
the row the read returns and its verdict bit. The read's template fixes its table's count and each rank's position
(`Ports.domainAt`); a dense table puts each row's leaf at its own rank.

What a table read's soundness takes of a read unit `u` at a transcript `X`:

* `GatesRefine`: `u`'s gates compute its reference. It is a fact about the circuit's gates, which the circuit alone
  decides, and not a cryptographic assumption.
* `RegisteredRows`: `u`'s registered rows hold in `X`: its root and domain id are its table's, its verdict bit is 1, and
  the registered constants among its inputs and outputs hold. A unit bound in `X` under the deployed openings (outside
  `Accounting.unbound`) has them wherever the openings pin those gates.
* `Wired`: `u`'s index gates hold, in `X`, the bits its picker's gates hold.

`Binds` is what follows: the row `u` returns is its table's row at the picked index, for every opening of that leaf the
table's root accepts, or SHA-512 collides on what the two checks hashed. `Reads` collects a circuit's read units with
their pickers and tables, and `Reads.finder` is the collision finder their binding reads `Claims.ExpectedCR` at.
-/

@[expose] public section

namespace Accounting.ReadUnit

variable {C : Verity.Circuit}

/-- Bit 0 of gate `g`'s value in `X`: a read unit's ports are one-bit gates. -/
def bit (X : C.Transcript) (g : Fin C.N) : Bool := (X g).getLsbD 0

/-- The bits the gates `gs` hold in `X`, in order. -/
def bits {k : ℕ} (X : C.Transcript) (gs : Fin k → Fin C.N) : List Bool := List.ofFn fun i => bit X (gs i)

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

variable (q : Ports C) (X : C.Transcript)

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

end Ports

/-- **The read's gates compute its reference** (`Hash.EmbeddingRead`), under the hashes `H` and `G`, the hm96 key `key`
and the circuit's registered constants `consts`: at every transcript `X` where unit `u` of `P` is correct and the
registered constants among `u`'s inputs and outputs hold, the row `u` returns is `Hash.EmbeddingRead.h`'s on its ports'
values, and its verdict bit is set only where `Hash.EmbeddingRead.ok` holds of them, in its table's domain under the
domain id its port holds.

This is a fact about the circuit's gates, which the circuit alone decides, and not a cryptographic assumption: it names
no adversary and no probability, and holds or fails of the registered circuit whatever a prover does. What proves it is a
proof about the read's gates against the reference (the select, the row's segment digest, the hm96 leaf, the frame-v3
path and the SHA-512 compressions under them), or an instance of a lowering-correctness theorem for the template the unit
instantiates. -/
def GatesRefine (H G : Hash.Msg → Verity.Digest) (key : Hash.Msg) (consts : Fin C.N → Option Bool) {n : ℕ}
    (P : Verity.Partition C n) (u : Fin n) (q : Ports C) : Prop :=
  ∀ X : C.Transcript, P.Correct X u → (∀ g ∈ P.io u, ∀ b, consts g = some b → bit X g = b) →
    q.hOf X = Hash.EmbeddingRead.h q.count q.words (q.tokOf X) (q.opening X) ∧
      (q.okOf X = true →
        Hash.EmbeddingRead.ok H G (q.domainAt (q.domainOf X)) key (q.rootOf X) q.count (q.tokOf X) (q.opening X))

/-- **The read's registered rows hold** in `X`, for the table under the root `r` and the domain id `d`: its root's and
domain id's gates hold them, its verdict bit is held to 1, and the registered constants `consts` among its inputs and
outputs hold. -/
structure RegisteredRows (consts : Fin C.N → Option Bool) {n : ℕ} (P : Verity.Partition C n) (u : Fin n) (q : Ports C)
    (r d : Verity.Digest) (X : C.Transcript) : Prop where
  root_eq : q.rootOf X = r
  domain_eq : q.domainOf X = d
  ok_eq : q.okOf X = true
  consts_hold : ∀ g ∈ P.io u, ∀ b, consts g = some b → bit X g = b

/-- **The read's index is its picker's** in `X`: its index gates hold the bits the picker's gates `idx` hold (a
router's output, an indexer's, or a token's). -/
def Wired (q : Ports C) (idx : Fin 32 → Fin C.N) (X : C.Transcript) : Prop := ∀ i, bit X (q.tok i) = bit X (idx i)

/-- **The read returns its table's row at the picked index** in `X`, for the table under the root `r`, the domain id
`d` and the hm96 key `key`: for every opening `o₀` the root accepts at the index the picker's gates `idx` hold, the row
the read returns is `o₀`'s, or `H` or `G` collides on what the read's check and `o₀`'s hashed. -/
def Binds (H G : Hash.Msg → Verity.Digest) (key : Hash.Msg) (q : Ports C) (idx : Fin 32 → Fin C.N)
    (r d : Verity.Digest) (X : C.Transcript) : Prop :=
  ∀ o₀ : Hash.RowLeaf.Opening, Hash.RowLeaf.accepts H G (q.verifier key d) r (num (bits X idx)) o₀ →
    q.hOf X = o₀.row ∨
      Hash.CollisionIn H (Hash.RowLeaf.outerHashed H G (q.verifier key d) (num (bits X idx)) (q.opening X) ++
        Hash.RowLeaf.outerHashed H G (q.verifier key d) (num (bits X idx)) o₀) ∨
      Hash.CollisionIn G (Hash.RowLeaf.innerHashed G (q.verifier key d) (q.opening X) ++
        Hash.RowLeaf.innerHashed G (q.verifier key d) o₀)

/-- **A circuit's read units**: which units of `n` read a table, each one's ports, its picker's gates, and its table: the
root, the domain id and the opening of each row. -/
structure Reads (C : Verity.Circuit) (n : ℕ) where
  units : Finset (Fin n)
  port : Fin n → Ports C
  pick : Fin n → Fin 32 → Fin C.N
  root : Fin n → Verity.Digest
  domain : Fin n → Verity.Digest
  table : Fin n → ℕ → Hash.RowLeaf.Opening

namespace Reads

variable {n : ℕ} (R : Reads C n)

/-- **Each read's table is committed under its root**: the root accepts the table's opening of each of its rows, under
the hashes `H` and `G` and the hm96 key `key`. -/
def Committed (H G : Hash.Msg → Verity.Digest) (key : Hash.Msg) : Prop :=
  ∀ u ∈ R.units, ∀ i < (R.port u).count,
    Hash.RowLeaf.accepts H G ((R.port u).verifier key (R.domain u)) (R.root u) i (R.table u i)

/-- The index read `u`'s picker's gates hold in `X`. -/
def index (u : Fin n) (X : C.Transcript) : ℕ := num (bits X (R.pick u))

/-- **What read `u`'s check and its table's check hashed** in `X`, under one hash `H`: the read's opening and its
table's opening at the picked index, each checked there. -/
def hashedAt (H : Hash.Msg → Verity.Digest) (key : Hash.Msg) (u : Fin n) (X : C.Transcript) : List Hash.Msg :=
  Hash.RowLeaf.outerHashed H H ((R.port u).verifier key (R.domain u)) (R.index u X) ((R.port u).opening X) ++
    Hash.RowLeaf.outerHashed H H ((R.port u).verifier key (R.domain u)) (R.index u X) (R.table u (R.index u X)) ++
    (Hash.RowLeaf.innerHashed H ((R.port u).verifier key (R.domain u)) ((R.port u).opening X) ++
      Hash.RowLeaf.innerHashed H ((R.port u).verifier key (R.domain u)) (R.table u (R.index u X)))

open Classical in
/-- **What the drawn reads hashed** at the outcome `ω`: `hashedAt` for each drawn unit that is a read, in the
transcript `X ω`. -/
noncomputable def drawnHashed {Ω : Type} (H : Hash.Msg → Verity.Digest) (key : Hash.Msg) (drawn : Ω → Finset (Fin n))
    (X : Ω → C.Transcript) (ω : Ω) : List Hash.Msg :=
  ((drawn ω).filter (· ∈ R.units)).toList.flatMap fun u => R.hashedAt H key u (X ω)

open Classical in
/-- **The table reads' collision finder**: on the outcome `ω`, two messages the drawn reads hashed that `H` maps to one
digest, if there are two, at the cost `cost ω`: the SHA-512 calls of reaching `ω`'s transcript and hashing what its
drawn reads hashed. -/
noncomputable def finder {Ω : Type} [Fintype Ω] [Nonempty Ω] (H : Hash.Msg → Verity.Digest) (key : Hash.Msg)
    (drawn : Ω → Finset (Fin n)) (X : Ω → C.Transcript) (cost : Ω → ℕ) : Claims.Finder where
  S := Ω
  out ω :=
    if h : ∃ x ∈ R.drawnHashed H key drawn X ω, ∃ y ∈ R.drawnHashed H key drawn X ω,
        Hash.Collision (fun m => (H m).bytes) x y then
      some (h.choose, h.choose_spec.2.choose)
    else none
  cost := cost

end Reads

end Accounting.ReadUnit
