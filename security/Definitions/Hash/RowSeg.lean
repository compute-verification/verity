module

public import Aesop.BuiltinRules
public import Definitions.Core.Hash
public import Mathlib.Tactic.ToAdditive
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
# `sha512/row-seg/v1` over a hash (trusted)

The row digest of PoUW's A rows (`verity/ml/commitments/frame_v3/PROTOCOL.md` §6, "Segmented bit rows"), with the hash a
parameter `G` (SHA-512 is the instance of record; the verifier's code is `Flock.HmRow.segDigest`). A row is a string of
`ℓ` bits (`Row`). Its bytes are eight bits a byte, row bit `i` being bit `i mod 8` of byte `⌊i/8⌋`, the last byte's fill
zero (`packBits`). It is cut into `s = ⌈ℓ/8192⌉` segments of 8192 bits, the last possibly shorter, and

    Q(ℓ)       = ("verity/sha512-row-seg/v1\0" ‖ u64be ℓ ‖ u64be s ‖ u32be 1024)             zero-padded to 128 bytes
    S(j, s, m) = ("verity/sha512-row-seg/v1/segment\0" ‖ u64be j ‖ u64be s ‖ u64be m)   zero-padded to 128 bytes
    d_j        = G(S(j, s, m_j) ‖ segment j's bytes)
    digest     = G(Q(ℓ) ‖ d_0 ‖ … ‖ d_(s−1))

so the digest's messages bind the row's bit length `ℓ` (in `Q(ℓ)`) and each segment's (in `S`). `RowSeg.hashed` lists
every message the digest hashed: the segments' messages, then the outer one.
-/

@[expose] public section

namespace Hash

/-- A row: its bits, in order. -/
abbrev Row := List Bool

namespace RowSeg

/-- Bits as a number, the first least significant. -/
def bitsVal : List Bool → ℕ
  | [] => 0
  | b :: bs => (if b then 1 else 0) + 2 * bitsVal bs

/-- A row's bytes: bit `i` is bit `i mod 8` of byte `⌊i/8⌋`, the last byte zero-filled. -/
def packBits (bs : Row) : Msg :=
  (List.range ((bs.length + 7) / 8)).map fun j => (bitsVal ((bs.drop (8 * j)).take 8)).toUInt8

/-- A segment's bits at most. -/
def SEG_BITS : ℕ := 8192

/-- The number of segments of a row of `ℓ` bits: `⌈ℓ / 8192⌉`. -/
def segCount (ℓ : ℕ) : ℕ := (ℓ + SEG_BITS - 1) / SEG_BITS

/-- Segment `j` of the row: its bits `[8192 j, 8192 j + m_j)`. -/
def segment (row : Row) (j : ℕ) : Row := (row.drop (SEG_BITS * j)).take SEG_BITS

/-- `Q(ℓ)`, the outer message's first block. -/
def outerPrefix (ℓ : ℕ) : Msg :=
  padTo 128 (cstr "verity/sha512-row-seg/v1" ++ be 8 ℓ ++ be 8 (segCount ℓ) ++ be 4 1024)

/-- `S(j, s, m)`, segment `j`'s first block. -/
def segmentPrefix (j s m : ℕ) : Msg :=
  padTo 128 (cstr "verity/sha512-row-seg/v1/segment" ++ be 8 j ++ be 8 s ++ be 8 m)

/-- Segment `j`'s message: `S(j, s, m_j) ‖ its bytes`. -/
def segmentMsg (row : Row) (j : ℕ) : Msg :=
  segmentPrefix j (segCount row.length) (segment row j).length ++ packBits (segment row j)

/-- The segments' messages, in order. -/
def segmentMsgs (row : Row) : List Msg := (List.range (segCount row.length)).map (segmentMsg row)

/-- The outer message: `Q(ℓ) ‖ d_0 ‖ … ‖ d_(s−1)`. -/
def outerMsg (G : Msg → Digest) (row : Row) : Msg :=
  outerPrefix row.length ++ (segmentMsgs row).flatMap fun m => (G m).bytes

/-- The row's digest. -/
def digest (G : Msg → Digest) (row : Row) : Digest := G (outerMsg G row)

/-- Every message the digest hashed: the segments', then the outer one. -/
def hashed (G : Msg → Digest) (row : Row) : List Msg := segmentMsgs row ++ [outerMsg G row]

end RowSeg

end Hash
