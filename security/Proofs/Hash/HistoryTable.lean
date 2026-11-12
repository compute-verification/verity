module

public import Definitions.Hash.Guarantees
public import Proofs.Hash.EmbeddingRead
public import Proofs.Hash.TreeRoot
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
# `Hash.SecurityProofs.HistoryReadBinding : Hash.Guarantees.HistoryReadBinding`

A row and salt behind the string at `tok` open at `tok` against the history table's root with the tree's own path
(`accepts_opening`): their frame leaf is the table's leaf at `tok`, and the tree over the table's leaves accepts every
rank below its count (`TreeRoot.accepts_rootOf`). That opening is the registered one `EmbeddingReadBinding` compares the
read's witness with. Past the table's length the string is the placeholder's, so the read returns the zero row
(`read_past_length`).
-/

@[expose] public section

namespace Hash.SecurityProofs.HistoryTable

open Hash

/-- The frame leaf of a row and salt whose commit string is the string at `r` is the table's leaf at `r`. -/
theorem leaf_eq (H G : Msg → Digest) (D : Tree.Domain) (key : Msg) (s : ℕ → Msg) (row : Row) (salt : Msg) (r : ℕ)
    (sibs : List Digest) (hs : Hm96.commitString H key (RowLeaf.Format.seg.digest G row) salt = s r) :
    H (RowLeaf.leafMsg H G (EmbeddingRead.verifier D key) r ⟨row, salt, sibs⟩) = Hash.HistoryTable.leaf H D key s r := by
  simp only [RowLeaf.leafMsg, Hash.HistoryTable.leaf, EmbeddingRead.verifier, Hm96.leaf, hs]

/-- **The table opens at each rank below its count**: a row and 192-byte salt whose commit string is the string at `tok`
open there against the table's root, with the tree's own path. -/
theorem accepts_opening (H G : Msg → Digest) (D : Tree.Domain) (key : Msg) (s : ℕ → Msg) (row : Row) (salt : Msg)
    (tok : ℕ) (hin : tok < D.count) (hrow : row.length < 2 ^ 64) (hsalt : salt.length = Hm96.SALT_BYTES)
    (hs : Hm96.commitString H key (RowLeaf.Format.seg.digest G row) salt = s tok) :
    RowLeaf.accepts H G (EmbeddingRead.verifier D key) (Hash.HistoryTable.root H D key s) tok
      (Hash.HistoryTable.opening H D key s row salt tok) := by
  unfold RowLeaf.accepts
  refine ⟨hrow, hsalt, ?_⟩
  rw [Hash.HistoryTable.opening, leaf_eq H G D key s row salt tok _ hs]
  exact TreeRoot.accepts_rootOf H D (Hash.HistoryTable.leaf H D key s) tok hin

end Hash.SecurityProofs.HistoryTable

open Hash in
/-- **A history table's read binds the row its writer committed** (`Hash.Guarantees.HistoryReadBinding`). -/
theorem Hash.SecurityProofs.HistoryReadBinding : Hash.Guarantees.HistoryReadBinding := by
  intro H G domain key s words tok o hok row salt hin hrow hsalt hs
  exact Hash.SecurityProofs.EmbeddingReadBinding H G domain key _ domain.count words tok o hok _ hin
    (Hash.SecurityProofs.HistoryTable.accepts_opening H G domain key s row salt tok hin hrow hsalt hs)

namespace Hash.SecurityProofs.HistoryTable

open Hash

/-- **A read past the table's length returns the zero row**: when the strings past the length `len` are the
placeholder's, a read at an index between `len` and the count that returns ok returns the zero row of `words` words, or
the checks hashed a collision against the placeholder's own opening. -/
theorem read_past_length (H G : Msg → Digest) (D : Tree.Domain) (key : Msg) (len : ℕ) (bc : ℕ → Msg) (words tok : ℕ)
    (o : RowLeaf.Opening) (hwords : 16 * words < 2 ^ 64)
    (hok : EmbeddingRead.ok H G D key
      (Hash.HistoryTable.root H D key (Hash.HistoryTable.string len bc (Hash.HistoryTable.placeholder H G key words)))
      D.count tok o)
    (hlen : len ≤ tok) (hin : tok < D.count) :
    let s := Hash.HistoryTable.string len bc (Hash.HistoryTable.placeholder H G key words)
    let o₀ := Hash.HistoryTable.opening H D key s (Hash.HistoryTable.zeroRow words) Hash.HistoryTable.zeroSalt tok
    EmbeddingRead.h D.count words tok o = Hash.HistoryTable.zeroRow words ∨
      CollisionIn H (RowLeaf.outerHashed H G (EmbeddingRead.verifier D key) tok o ++
        RowLeaf.outerHashed H G (EmbeddingRead.verifier D key) tok o₀) ∨
      CollisionIn G (RowLeaf.innerHashed G (EmbeddingRead.verifier D key) o ++
        RowLeaf.innerHashed G (EmbeddingRead.verifier D key) o₀) :=
  Hash.SecurityProofs.HistoryReadBinding H G D key _ words tok o hok _ _ hin (by simpa [Hash.HistoryTable.zeroRow] using hwords)
    (by simp [Hash.HistoryTable.zeroSalt]) (by simp [Hash.HistoryTable.string, Nat.not_lt.mpr hlen, Hash.HistoryTable.placeholder])

end Hash.SecurityProofs.HistoryTable
