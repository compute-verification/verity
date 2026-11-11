module

public import Definitions.Hash
public import Definitions.Hash.EmbeddingRead
public import Definitions.Hash.HistoryTable
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
# The hash area's guarantee statements

Each is proved by a theorem `Hash.SecurityProofs.G : Hash.Guarantees.G` in `Security/Proofs/Hash/`. All are
deterministic and assume nothing about the hash: each conclusion is the property, or a collision among
the messages the two openings' checks themselves hashed, never an advantage bound. So no proof takes an assumption.

* `RowBinding`: two openings of one root at one rank, in the hm96 row leaf format the verifier holds (frame-v3-sha512
  tree, frame leaf, hm96-sha512 leaf under the verifier's key, and a `sha512/row-seg/v1`, `sha512/row/v2` or
  `sha512/row/v1` row digest), carry the same row, or the checks hashed a collision.
* `TreeBinding`: two openings of one frame-v3 root at one rank carry the same leaf digest, or the path checks hashed a
  collision of node messages.
* `EmbeddingReadBinding`: when `EmbeddingRead_v1` returns ok for a token in the vocabulary, the row it returns is the
  registered table's row at that token, or the checks hashed a collision. It is `RowBinding` at rank `tok`.
* `HistoryReadBinding`: when `EmbeddingRead_v1` returns ok at an index below the count, against the root the verifier
  computes from the rows' public commit strings (`HistoryTable.root`), the row it returns is the row behind the string
  at that index, under any salt that string commits it with, or the checks hashed a collision.

SHA-512 is the served hash, not BLAKE3. A probability read from
any of them is per prover: the collision comes from that prover's (extracted) openings, so it is a bound on that
prover's link finders under `ecr/sha-512` (`FlockSoundness.Assumptions.SHA512CRExpected`), never one `δ_bind` for every
prover, which is false at SHA-512 (a prover that hard-codes one collision opens one leaf two ways). In particular
none of them discharges a constant over all provers such as `Pouw.Assumptions.ServiceSoundStaged`'s `δb`.
-/

@[expose] public section

namespace Hash.Guarantees

/-- **Row binding, as a break tied to the run.** Two openings of one root at one rank, each accepted by the verifier of the
hm96 row leaf in its format (`RowLeaf.accepts`: a frame-v3 tree over the verifier's domain, its frame leaf binding the
domain id, rank, position and the format's schema, its nodes their level and index; the hm96-sha512 leaf under the
verifier's key, bound through `H(key)` in its prefix; the format's row digest binding the row's bit length and its bits,
each segment's under `sha512/row-seg/v1`, and under `sha512/row/v1` a row of whole words whose bytes give its length;
every part length-framed or fixed-width), carry the same row, or two different
messages that `H` hashed in the two checks have one digest, or two different messages that `G` hashed in the two row
digests have one digest.

The instance of record is `H = G = SHA-512`, where the last two disjuncts together are a SHA-512 collision among the
inputs the two openings hashed. It is stated for any 64-byte `H` and `G`.

Scope:
* **Leaves.** The verifier's format (`RowLeaf.Format`): `hm96-sha512/row-seg/v1` leaves (`seg`: PoUW's per-call A rows,
  the vLLM row tables, A6's `x` port, the clear sampled audit's rows) or `hm96-sha512/row/v2` leaves (`v2`: H1's tables
  and registered weights, serving's step rows, the chunked templates' `w`, `in<g>` and `o0` rows, the remote audit's
  receipts) or `hm96-sha512/row/v1` leaves (`v1`: Flock's default ports, the registered values, M0's serving rows,
  each a row of 16-bit words). A row cut into several leaves (H1's chunk rows past K = 4096, n leaves a row) is bound
  leaf by leaf, at each leaf's rank; which leaves form a row is the layout's.
* **Domain.** `V.domain` is free, and RowBinding holds for every domain. It does not say that a root is a rows tree. The
  schema doesn't separate them: PoUW's tile-digest trees write `hm96-sha512/row-seg/v1` too, with `x` the 64-byte digest
  itself, and there a row's leaf equals the leaf of its own row-seg digest with no collision. Only the domain's binding
  (`weight_binding`/`activation_binding` against `transcript_binding` in `verity/core/protocols/pouw/audit.py`)
  separates them, so a consumer establishes that `V.domain` is a rows domain from its binding.
* **Relation to Flock.** `FlockSoundness.Binding.ExplicitLeaf.rowSeg` already proves row-seg binding at the hm96 leaf,
  in explicit-pair form, over `SegRow l`, `List UInt8` digests and one hash. RowBinding's hm96 and row-seg part is the
  same property in this area's types, from a frame root. No lemma relates `RowSeg.digest` to `HmRow.segDigest` or
  `Hm96.leaf` to `Hm96.Default512` yet. A later bridge connects them, and states the break as a search over the listed
  messages (`Claims.findColl`'s shape) for a link finder. -/
def RowBinding : Prop :=
  ∀ (H G : Msg → Digest) (V : RowLeaf.Verifier) (root : Digest) (rank : ℕ) (o₁ o₂ : RowLeaf.Opening),
    RowLeaf.accepts H G V root rank o₁ → RowLeaf.accepts H G V root rank o₂ →
    o₁.row = o₂.row ∨
      CollisionIn H (RowLeaf.outerHashed H G V rank o₁ ++ RowLeaf.outerHashed H G V rank o₂) ∨
      CollisionIn G (RowLeaf.innerHashed G V o₁ ++ RowLeaf.innerHashed G V o₂)

/-- **Frame-v3 tree binding, as a break tied to the run.** Two leaf digests accepted at one rank against one root, with a
rank below the domain's count and paths of `merkle_depth count` siblings (`Tree.accepts`), are equal, or two different
node messages (each framing the domain id, its level and index, and its children) that the two path checks hashed have
one digest under `H`.

Its reader is the step from a serving root to the registered leaf. Flock takes the hm96 leaf per position as given
(`FlockSoundness.Binding.Layout.regLeaf`), and the frame-v3 tree above it is outside its proof. TreeBinding binds a root
that a record pins to that leaf, which is the content main doesn't have. It is not about Flock's own column trees
(`leafOfId`, nodes "SHA-512 of the pair"), so it feeds neither `TableCRL` nor `SessCRL`. -/
def TreeBinding : Prop :=
  ∀ (H : Msg → Digest) (D : Tree.Domain) (root : Digest) (rank : ℕ) (l₁ l₂ : Digest) (sibs₁ sibs₂ : List Digest),
    Tree.accepts H D root rank l₁ sibs₁ → Tree.accepts H D root rank l₂ sibs₂ →
    l₁ = l₂ ∨ CollisionIn H (Tree.hashed H D rank l₁ sibs₁ ++ Tree.hashed H D rank l₂ sibs₂)

/-- **The embedding's row read binds its row, as a break tied to the run.** When `EmbeddingRead_v1` (`EmbeddingRead.ok`)
returns ok on the witness opening `o` for a token `tok` in the vocabulary, the row it returns (`EmbeddingRead.h`) is the
row of any opening `o₀` accepted at rank `tok` against the same root under the read's verifier (the registered table's
row at `tok`), or two different messages that `H` hashed in the two checks have one digest, or two different messages
that `G` hashed in the two row digests have one digest.

Scope:
* **What it is about.** The reference's outputs (`Definitions/Hash/EmbeddingRead.lean`), not the gates; the gates'
  tests (`verity/ml/commitments/tests/test_boolean_embedding.py`) hold them to the reference.
* **What the unit's wiring supplies.** `root`, `domain` and `key` are the registered table's: the root and domain id are
  registered reads of the table's root in the unit that holds the read, the domain's count is `V`, and the key is hm96's
  pinned key, which the table's registration uses. `RowBinding`'s note on domains applies: the domain's binding says it
  is a rows domain.
* **A token outside the vocabulary** returns the fill and ok whatever the witness, and the statement says nothing there. -/
def EmbeddingReadBinding : Prop :=
  ∀ (H G : Msg → Digest) (domain : Tree.Domain) (key : Msg) (root : Digest) (vocab words tok : ℕ)
    (o : RowLeaf.Opening),
    EmbeddingRead.ok H G domain key root vocab tok o →
    ∀ o₀ : RowLeaf.Opening, tok < vocab → RowLeaf.accepts H G (EmbeddingRead.verifier domain key) root tok o₀ →
      EmbeddingRead.h vocab words tok o = o₀.row ∨
        CollisionIn H (RowLeaf.outerHashed H G (EmbeddingRead.verifier domain key) tok o ++
          RowLeaf.outerHashed H G (EmbeddingRead.verifier domain key) tok o₀) ∨
        CollisionIn G (RowLeaf.innerHashed G (EmbeddingRead.verifier domain key) o ++
          RowLeaf.innerHashed G (EmbeddingRead.verifier domain key) o₀)

/-- **A history table's read binds the row its writer committed, as a break tied to the run.** When `EmbeddingRead_v1`
returns ok on the witness opening `o` at an index `tok` below the domain's count, against the root of the history table
whose rank `r` holds the public commit string `s r` (`HistoryTable.root`, which the verifier computes and the prover
never supplies), the row it returns is `row` for every row-seg row `row` and 192-byte salt `salt` whose commit string is
`s tok`, or two different messages that `H` hashed in the read's check and in the check of that row's opening at `tok`
(`HistoryTable.opening`, the tree's own path) have one digest, or two different messages that `G` hashed in their row
digests have one digest.

Scope:
* **What the strings are.** `s` is the verifier's: a writer unit's committed `b ‖ c` at each position its program's wiring
  gives, below the table's length, and the placeholder's (`HistoryTable.placeholder`) past it. A read at an index past the
  length returns the zero row then, the placeholder's own opening, or the checks collided.
* **What it is about.** The reference's outputs, as for `EmbeddingReadBinding`; the domain, key and count are the
  read's template's and the table's, which the verifier holds. -/
def HistoryReadBinding : Prop :=
  ∀ (H G : Msg → Digest) (domain : Tree.Domain) (key : Msg) (s : ℕ → Msg) (words tok : ℕ) (o : RowLeaf.Opening),
    EmbeddingRead.ok H G domain key (HistoryTable.root H domain key s) domain.count tok o →
    ∀ (row : Row) (salt : Msg), tok < domain.count → row.length < 2 ^ 64 → salt.length = Hm96.SALT_BYTES →
      Hm96.commitString H key (RowLeaf.Format.seg.digest G row) salt = s tok →
      EmbeddingRead.h domain.count words tok o = row ∨
        CollisionIn H (RowLeaf.outerHashed H G (EmbeddingRead.verifier domain key) tok o ++
          RowLeaf.outerHashed H G (EmbeddingRead.verifier domain key) tok
            (HistoryTable.opening H domain key s row salt tok)) ∨
        CollisionIn G (RowLeaf.innerHashed G (EmbeddingRead.verifier domain key) o ++
          RowLeaf.innerHashed G (EmbeddingRead.verifier domain key) (HistoryTable.opening H domain key s row salt tok))

end Hash.Guarantees
