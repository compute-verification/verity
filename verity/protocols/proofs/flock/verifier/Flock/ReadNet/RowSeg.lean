/-
`sha512/row-seg/v1`'s row digest as builder programs (`gates.rowleaf`): the segment hash `Sha512RowSegSegment_v1{M}`
(`segment`), the outer hash `Sha512RowSegOuter_v1{N_BITS}` (`outer`), each a traced Definition, and the composite
`Sha512RowSeg_v1{N_BITS}` (`rowSeg`), whose body calls the constants 0 and 1, then the segment hash once a segment from its
prefix's chaining value over those two bits, then the outer hash.
-/
module

public import Flock.HmRow
public import Flock.ReadNet.Sha

@[expose] public section

namespace Flock
namespace ReadNet

open HmNets

/-- `rowleaf.segment_hash` (`Sha512RowSegSegment_v1{M}`): SHA-512 from the chaining value `cv` over the segment's `m`
bits as `⌈m / 8⌉` bytes with zero fill, padded as the end of a `128 + ⌈m / 8⌉`-byte message. -/
def segment (m : Nat) (cv : List Word) (bits : List Lin) : M (List Word) :=
  let n := (m + 7) / 8
  let msg := msgWords (bits ++ List.replicate (8 * n - m) Lin.zero ++ constBits (padTail (128 + n)))
  shaW (msg.length / 16) cv msg

/-- `rowleaf.outer_hash` (`Sha512RowSegOuter_v1{N_BITS}`): SHA-512 of `Q(l)` and the digests `d` (eight words each), from
`Q(l)`'s chaining value. -/
def outer (l : Nat) (d : List Word) : M (List Word) :=
  let msg := d ++ constWords (padTail (128 + 8 * d.length))
  shaW (msg.length / 16) (midstate (HmRow.rowPrefixSeg l)) msg

/-- A 128-byte prefix's chaining value as words over the bits `zero` and `one` (`rowleaf._body`'s `cv`): bit `k` of word
`i` is `one` where the chaining value's is set. -/
def cvWords (zero one : Lin) (pre : ByteArray) : List Word :=
  (List.range 8).map fun i => (List.range 64).map fun k =>
    if (Sha512.compress Sha512.iv pre 0)[i]!.toBitVec.getLsbD k then one else zero

/-- `rowleaf._body` (`Sha512RowSeg_v1{N_BITS}`), its calls `const` (`BL.constant(b)`), `seg`
(`Sha512RowSegSegment_v1{M}`) and `out` (`Sha512RowSegOuter_v1{N_BITS}`): the constants 0 and 1, each 8,192-bit segment
of the `l`-bit row from its prefix's chaining value over them, then the outer hash of the segments' digests. -/
def rowSeg (const : Bool → M Lin) (seg : Nat → List Word → List Lin → M (List Word))
    (out : Nat → List Word → M (List Word)) (l : Nat) (bits : List Lin) : M (List Word) := do
  let zero ← const false
  let one ← const true
  let s := (l + HmRow.ROW_SEG_BITS - 1) / HmRow.ROW_SEG_BITS
  let d ← (List.range s).mapM fun j =>
    let m := min (l - HmRow.ROW_SEG_BITS * j) HmRow.ROW_SEG_BITS
    seg m (cvWords zero one (HmRow.segmentPrefix j s m)) ((bits.drop (HmRow.ROW_SEG_BITS * j)).take m)
  out l d.flatten

/-- `BL.constant(b)` built in place: the constant form. -/
def constBit (b : Bool) : M Lin := pure (Lin.const b)

/-- `Sha512RowSeg_v1{N_BITS}` with its calls built in place. -/
def rowSegNet : Nat → List Lin → M (List Word) := rowSeg constBit segment outer

end ReadNet
end Flock
