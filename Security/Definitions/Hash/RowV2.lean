import Definitions.Hash.RowSeg

/-!
# `sha512/row/v2` over a hash (trusted)

The unsegmented bit row's digest (`verity/ml/commitments/frame_v3/PROTOCOL.md` §6, "Bit rows"), with the hash a parameter
`G` (SHA-512 is the instance of record; the reference is `rowleaf.sha512_row_v2_digest`). A row is `ℓ` bits (`Row`),
its bytes as `sha512/row-seg/v1` packs them (`RowSeg.packBits`), and

    P(ℓ)   = ("verity/sha512-row/v2\0" ‖ u64be ℓ)   zero-padded to 128 bytes
    digest = G(P(ℓ) ‖ row bytes)

so the one message the digest hashes (`RowV2.hashed`) binds the row's bit length `ℓ` and its bytes.
-/

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
