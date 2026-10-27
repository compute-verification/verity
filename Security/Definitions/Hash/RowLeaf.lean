import Definitions.Hash.Tree
import Definitions.Hash.RowSeg
import Definitions.Hash.RowV2
import Definitions.Hash.RowV1
import Definitions.Hash.Hm96

/-!
# The hm96 row leaf: `hm96-sha512/row-seg/v1`, `hm96-sha512/row/v2` or `hm96-sha512/row/v1` in a frame-v3-sha512 tree
(trusted)

A row as the served path commits its bit rows (`frame_v3/PROTOCOL.md` §6, `hm96/PROTOCOL.md` §1): its inner digest `x`
under `G`, wrapped in the hm96-sha512 leaf under `H` with the row's salt and the verifier's key (`Hm96.leaf`), whose 64
bytes are the value of the frame leaf at the row's rank, under the format's schema, in the verifier's domain
(`Tree.leafMsg`), opened by a path to the root (`Tree.accepts`). The instance of record is `H = G = SHA-512`. `G` is
separate so that each break names the hash it came from.

The format (`Format`) is the verifier's, as it holds each port's schema from its registration: `seg`, the schema
`hm96-sha512/row-seg/v1` over `sha512/row-seg/v1` (`RowSeg.digest`; PoUW's A rows as `PouwCommit.RowsDomain.leaf`
commits them, the vLLM row tables), `v2`, the schema `hm96-sha512/row/v2` over `sha512/row/v2` (`RowV2.digest`;
H1's tables, the registered weights, serving's step rows), or `v1`, the schema `hm96-sha512/row/v1` over `sha512/row/v1`
(`RowV1.digest`; C-Flock's default ports, served vLLM's word rows). C-Flock's registered check
(`Flock.Registered.opensLeaf`) is this leaf at SHA-512 in each format (`Flock.Bridge.Openings.rowLeaf_accepts`).

What the verifier holds (`Verifier`): its domain, the hm96 key and the format, all from its own statement. An opening
(`Opening`): the row's bits, its salt and the path's siblings. The verifier accepts it at `rank` against `root`
(`accepts`) when the row's bit length is a u64 (`ℓ < 2^64`, as `u64be ℓ` requires) that the format's rows have
(`Format.fits`: whole u16 words under `v1`, whose prefix counts words), the salt is 192 bytes, and the frame leaf's
digest opens at `rank`. What its check hashed: under `H` (`outerHashed`), the salt's and the hm96 leaf's
messages, the frame leaf's message, and the path's node messages; under `G` (`innerHashed`), the row digest's messages.
-/

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

/-- The bit lengths of the format's rows: any under the bit-row formats, whole u16 words under `v1`. -/
def Format.fits : Format → ℕ → Prop
  | .seg, _ => True
  | .v2, _ => True
  | .v1, ℓ => ℓ % 16 = 0

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
  o.row.length < 2 ^ 64 ∧ V.format.fits o.row.length ∧ o.salt.length = Hm96.SALT_BYTES ∧
    Tree.accepts H V.domain root rank (H (leafMsg H G V rank o)) o.sibs

/-- What the check of `o` at `rank` hashed under `H`: the salt's and the hm96 leaf's messages, the frame leaf's, and the
path's nodes'. -/
def outerHashed (H G : Msg → Digest) (V : Verifier) (rank : ℕ) (o : Opening) : List Msg :=
  Hm96.hashed H V.key (V.format.digest G o.row) o.salt ++ [leafMsg H G V rank o] ++
    Tree.hashed H V.domain rank (H (leafMsg H G V rank o)) o.sibs

/-- What the check of `o` hashed under `G`: the row digest's messages. -/
def innerHashed (G : Msg → Digest) (V : Verifier) (o : Opening) : List Msg := V.format.hashed G o.row

end Hash.RowLeaf
