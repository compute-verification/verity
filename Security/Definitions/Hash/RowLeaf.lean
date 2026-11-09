module

public import Definitions.Hash.Tree
public import Definitions.Hash.RowV1
public import Definitions.Hash.RowV2
public import Definitions.Hash.Hm96
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
# The hm96 row leaf: `hm96-sha512/row-seg/v1`, `/row/v2` or `/row/v1` in a frame-v3-sha512 tree (trusted)

A row as the served path commits its bit rows (`frame_v3/PROTOCOL.md` §6, `hm96/PROTOCOL.md` §1): its inner digest `x`
under `G`, wrapped in the hm96-sha512 leaf under `H` with the row's salt and the verifier's key (`Hm96.leaf`), whose 64
bytes are the value of the frame leaf at the row's rank, under the format's schema, in the verifier's domain
(`Tree.leafMsg`), opened by a path to the root (`Tree.accepts`). The instance of record is `H = G = SHA-512`. `G` is
separate so that each break names the hash it came from.

The format (`Format`) is the verifier's, as it holds each port's schema from its registration: `seg`, the schema
`hm96-sha512/row-seg/v1` over `sha512/row-seg/v1` (`RowSeg.digest`; PoUW's A rows as `PouwCommit.RowsDomain.leaf`
commits them, the vLLM row tables), `v2`, the schema `hm96-sha512/row/v2` over `sha512/row/v2` (`RowV2.digest`;
H1's tables, the registered weights, serving's step rows), or `v1`, the schema `hm96-sha512/row/v1` over
`sha512/row/v1` (`RowV1.digest`; Flock's default ports, the registered values committed under it, M0's serving rows).
Nothing ties this text to the code yet: a `Properties` lemma or a difftest from `PouwCommit.RowsDomain.verify` to
`accepts` is a proof's first step.

What the verifier holds (`Verifier`): its domain, the hm96 key and the format, all from its own statement. An opening
(`Opening`): the row's bits, its salt and the path's siblings. The verifier accepts it at `rank` against `root`
(`accepts`) when the row's bit length is one its format takes (`Format.Fits`: a u64, `ℓ < 2^64`, as `u64be ℓ`
requires, and under `v1` whole 16-bit words, as its prefix counts words), the salt is 192 bytes, and the frame leaf's
digest opens at `rank`. What its check hashed: under `H` (`outerHashed`), the salt's and the hm96 leaf's
messages, the frame leaf's message, and the path's node messages; under `G` (`innerHashed`), the row digest's messages.
-/

@[expose] public section

namespace Hash.RowLeaf

/-- A leaf's format: its schema and the row digest under it. -/
inductive Format where
  /-- `hm96-sha512/row-seg/v1` over `sha512/row-seg/v1`. -/
  | seg
  /-- `hm96-sha512/row/v2` over `sha512/row/v2`. -/
  | v2
  /-- `hm96-sha512/row/v1` over `sha512/row/v1`. -/
  | v1

/-- The frame leaf's schema. -/
def Format.schema : Format → String
  | .seg => "hm96-sha512/row-seg/v1"
  | .v2 => "hm96-sha512/row/v2"
  | .v1 => "hm96-sha512/row/v1"

/-- The row's inner digest. -/
def Format.digest (G : Msg → Digest) : Format → Row → Digest
  | .seg => RowSeg.digest G
  | .v2 => RowV2.digest G
  | .v1 => RowV1.digest G

/-- The messages the inner digest hashed. -/
def Format.hashed (G : Msg → Digest) : Format → Row → List Msg
  | .seg => RowSeg.hashed G
  | .v2 => RowV2.hashed
  | .v1 => RowV1.hashed

/-- The row bit lengths the format takes: a u64, and under `v1` whole words. -/
def Format.Fits : Format → ℕ → Prop
  | .seg, ℓ => ℓ < 2 ^ 64
  | .v2, ℓ => ℓ < 2 ^ 64
  | .v1, ℓ => ℓ < 2 ^ 64 ∧ ℓ % RowV1.WORD_BITS = 0

/-- What the verifier holds: its domain, the hm96 key and the leaf's format. -/
structure Verifier where
  domain : Tree.Domain
  key : Msg
  format : Format

/-- An opening: the row's bits, its salt, and the path's siblings from the leaf level up. -/
structure Opening where
  row : Row
  salt : Msg
  sibs : List Digest

/-- The frame leaf's message: the hm96 leaf over the row's digest and salt, at `rank`. -/
def leafMsg (H G : Msg → Digest) (V : Verifier) (rank : ℕ) (o : Opening) : Msg :=
  Tree.leafMsg V.domain rank V.format.schema (Hm96.leaf H V.key (V.format.digest G o.row) o.salt).bytes

/-- The verifier accepts the opening `o` at `rank` against `root`. -/
def accepts (H G : Msg → Digest) (V : Verifier) (root : Digest) (rank : ℕ) (o : Opening) : Prop :=
  V.format.Fits o.row.length ∧ o.salt.length = Hm96.SALT_BYTES ∧
    Tree.accepts H V.domain root rank (H (leafMsg H G V rank o)) o.sibs

/-- What the check of `o` at `rank` hashed under `H`: the salt's and the hm96 leaf's messages, the frame leaf's, and the
path's nodes'. -/
def outerHashed (H G : Msg → Digest) (V : Verifier) (rank : ℕ) (o : Opening) : List Msg :=
  Hm96.hashed H V.key (V.format.digest G o.row) o.salt ++ [leafMsg H G V rank o] ++
    Tree.hashed H V.domain rank (H (leafMsg H G V rank o)) o.sibs

/-- What the check of `o` hashed under `G`: the row digest's messages. -/
def innerHashed (G : Msg → Digest) (V : Verifier) (o : Opening) : List Msg := V.format.hashed G o.row

end Hash.RowLeaf
