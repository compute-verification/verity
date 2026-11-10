module

public import Definitions.Hash.RowSeg
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
# `sha512/row/v2` over a hash (trusted)

The unsegmented bit row's digest (`verity/ml/commitments/frame_v3/PROTOCOL.md` §6, "Bit rows"), with the hash a parameter
`G` (SHA-512 is the instance of record; the reference is `rowleaf.sha512_row_v2_digest`). A row is `ℓ` bits (`Row`),
its bytes as `sha512/row-seg/v1` packs them (`RowSeg.packBits`), and

    P(ℓ)   = ("verity/sha512-row/v2\0" ‖ u64be ℓ)   zero-padded to 128 bytes
    digest = G(P(ℓ) ‖ row bytes)

so the one message the digest hashes (`RowV2.hashed`) binds the row's bit length `ℓ` and its bytes.
-/

@[expose] public section

namespace Hash.RowV2

/-- `P(ℓ)`, the message's first block. -/
def rowPrefix (ℓ : ℕ) : Msg := padTo 128 (cstr "verity/sha512-row/v2" ++ be 8 ℓ)

/-- The message: `P(ℓ) ‖ row bytes`. -/
def msg (row : Row) : Msg := rowPrefix row.length ++ RowSeg.packBits row

/-- The row's digest. -/
def digest (G : Msg → Digest) (row : Row) : Digest := G (msg row)

/-- The message the digest hashed. -/
def hashed (row : Row) : List Msg := [msg row]

end Hash.RowV2
