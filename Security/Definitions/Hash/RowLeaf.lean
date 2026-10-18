import Definitions.Hash.Tree
import Definitions.Hash.RowSeg
import Definitions.Hash.Hm96

/-!
# The served row leaf: `hm96-sha512/row-seg/v1` in a frame-v3-sha512 tree (trusted)

A row as PoUW commits it (`PouwCommit.RowsDomain.leaf` under `hm96-sha512`; `frame_v3/PROTOCOL.md` §6, `hm96/PROTOCOL.md`
§1): its `sha512/row-seg/v1` digest `x` under `G` (`RowSeg.digest`), wrapped in the hm96-sha512 leaf under `H` with the
row's salt and the verifier's key (`Hm96.leaf`), whose 64 bytes are the value of the frame leaf at the row's rank, under
the schema `hm96-sha512/row-seg/v1`, in the verifier's domain (`Tree.leafMsg`), opened by a path to the root
(`Tree.accepts`). The instance of record is `H = G = SHA-512`; `G` is separate so that the same statement reads at
another inner digest.

What the verifier holds (`Verifier`): its domain and the hm96 key, both from its own statement. An opening (`Opening`):
the row's bits, its salt and the path's siblings. The verifier accepts it at `rank` against `root` (`accepts`) when the
row's bit length is a u64 (`ℓ < 2^64`, as `u64be ℓ` requires), the salt is 192 bytes, and the frame leaf's digest opens
at `rank`. What its check hashed: under `H` (`outerHashed`), the salt's and the hm96 leaf's messages, the frame leaf's
message, and the path's node messages; under `G` (`innerHashed`), the row digest's segment and outer messages.
-/

namespace Hash.RowLeaf

/-- The frame leaf's schema. -/
def SCHEMA : String := "hm96-sha512/row-seg/v1"

/-- What the verifier holds: its domain and the hm96 key. -/
structure Verifier where
  domain : Tree.Domain
  key : Msg

/-- An opening: the row's bits, its salt, and the path's siblings from the leaf level up. -/
structure Opening where
  row : Row
  salt : Msg
  sibs : List Digest

/-- The frame leaf's message: the hm96 leaf over the row's digest and salt, at `rank`. -/
def leafMsg (H G : Msg → Digest) (V : Verifier) (rank : ℕ) (o : Opening) : Msg :=
  Tree.leafMsg V.domain rank SCHEMA (Hm96.leaf H V.key (RowSeg.digest G o.row) o.salt).bytes

/-- The verifier accepts the opening `o` at `rank` against `root`. -/
def accepts (H G : Msg → Digest) (V : Verifier) (root : Digest) (rank : ℕ) (o : Opening) : Prop :=
  o.row.length < 2 ^ 64 ∧ o.salt.length = Hm96.SALT_BYTES ∧
    Tree.accepts H V.domain root rank (H (leafMsg H G V rank o)) o.sibs

/-- What the check of `o` at `rank` hashed under `H`: the salt's and the hm96 leaf's messages, the frame leaf's, and the
path's nodes'. -/
def outerHashed (H G : Msg → Digest) (V : Verifier) (rank : ℕ) (o : Opening) : List Msg :=
  Hm96.hashed H V.key (RowSeg.digest G o.row) o.salt ++ [leafMsg H G V rank o] ++
    Tree.hashed H V.domain rank (H (leafMsg H G V rank o)) o.sibs

/-- What the check of `o` hashed under `G`: the row digest's messages. -/
def innerHashed (G : Msg → Digest) (o : Opening) : List Msg := RowSeg.hashed G o.row

end Hash.RowLeaf
