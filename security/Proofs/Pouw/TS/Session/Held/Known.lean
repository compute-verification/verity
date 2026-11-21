module

public import Flock.RegKnown
public import Proofs.Flock.C3.RegLevels.ValueTie
public import Proofs.Flock.Recursive.ValueSrc
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
import Proofs.Flock.C3.RegLevels.OpensRegs
import Proofs.Flock.Soundness.Registered.Binding

/-!
# The known-content tie: a registrant's row at a port the verifier registers at known content is that content, or SHA-512 collides

`verity-verify-proof --registered F` refuses a file whose port named at known content (`Registered.KNOWN_KEY`: each row's
u16 words and its salt) has a registered root other than the frame-v3-sha512 root of those rows' `hm96-sha512/row/v2`
leaves (`Registered.checkKnownAll`, run in `FlockVerify.registeredInput`). What that refusal establishes, from the
verifier's code:

* `checkKnownAll_ok`, `checkKnown_ok`: each named port is `own`'s first port of that name, of `hm96-sha512/row/v2`
  rows, as many as its known rows, and its root is `FV3.root` of their leaves (`Registered.knownLeaves`);
* `knownPayload_eq`: a known row's hm96 leaf is `hm96Input`'s of its `sha512/row/v2` bytes (`rowBytesV2`);
* `knownColl_spec`: a registrant's row that opens such a root at position `i` (`opensLeaf`, `ReadsX.committed`) has
  the bytes of known row `i`, or `knownColl` outputs two different SHA-512 inputs with one digest. As `valueColl`
  (`C3/RegLevels/ValueTie.lean`), the pair is the hm96 leaf's (`hm96Pair`) or the opening's against the verifier's
  tree (`openPairAs`, the schema-generic `openPairV`);
* `bits_of_v2`: a value whose `sha512/row/v2` encoding is a known row's bytes has that row's bits.

`knownColl` evaluates SHA-512 at most `5n + 2^(D+1) + 2D + 7` times, `n` the rows and `D = log2ceil n`, as
`valueColl` does, with the known rows' digests in place of the verifier's.
-/

@[expose] public section

namespace FlockC3.RegLevels

open Flock
open FlockSoundness Spot.Sampling FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open FlockSoundness.Refine (H512)
open FlockSoundness.Binding (hm96Input hm96Pair hm96Pair_spec hm96_treeLeaf)

/-! ## What `checkKnownAll` checked -/

/-! ## A known row's leaf -/

/-! ## The extractor -/

/-- **The opening's extractor against the verifier's tree, at any leaf schema**: `openPairV` with the frame-v3 leaves
of `schema` (`Registered.leafInAs`) in place of `hm96-sha512/row/v1`'s. -/
def openPairAs (schema : String) (dom : ByteArray) (pos : ℕ) (x : ByteArray) (s : List ByteArray) (y : ByteArray)
    (t : List ByteArray) : ByteArray × ByteArray :=
  if Sha512.hash (Registered.leafInAs schema dom pos x) = Sha512.hash (Registered.leafInAs schema dom pos y) then
    (Registered.leafInAs schema dom pos x, Registered.leafInAs schema dom pos y)
  else Registered.climbPair dom (Sha512.hash (Registered.leafInAs schema dom pos x))
    (Sha512.hash (Registered.leafInAs schema dom pos y)) pos 0 s t

/-- **The known-content tie's extractor** at known row `i` of `ks`, registered under `reg`, against a registrant's
opening of the row bytes `r` under the salt `y` through `path`: nothing where `r` is the known row's
`sha512/row/v2` bytes; else hm96's pair where the two hm96 leaves are one; else the opening's pair (`openPairAs`) up the
registrant's path and the verifier's tree over the known rows. Each pair holds both preimages. -/
def knownColl (reg : Registered.Port) (ks : Array (Array ℕ × ByteArray)) (i : ℕ) (r y : List UInt8) (path : ByteArray) :
    Option (List UInt8 × List UInt8) :=
  if r = (Registered.rowBytesV2 ks[i]!.1).data.toList then none
  else if H512 (hm96Input r y) =
      H512 (hm96Input (Registered.rowBytesV2 ks[i]!.1).data.toList ks[i]!.2.data.toList) then
    some (hm96Pair r y (Registered.rowBytesV2 ks[i]!.1).data.toList ks[i]!.2.data.toList)
  else
    some ((openPairAs reg.schema reg.domain i (H512 (hm96Input r y)).1 (Registered.sibs path (log2ceil reg.leaves))
        (H512 (hm96Input (Registered.rowBytesV2 ks[i]!.1).data.toList ks[i]!.2.data.toList)).1
        (sibsV reg.domain (Registered.knownLeaves reg ks) i)).1.data.toList,
      (openPairAs reg.schema reg.domain i (H512 (hm96Input r y)).1 (Registered.sibs path (log2ceil reg.leaves))
        (H512 (hm96Input (Registered.rowBytesV2 ks[i]!.1).data.toList ks[i]!.2.data.toList)).1
        (sibsV reg.domain (Registered.knownLeaves reg ks) i)).2.data.toList)

/-! ## A known row's bits -/

end FlockC3.RegLevels
