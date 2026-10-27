import Definitions.Hash.RowSeg

/-!
# `sha512/row/v1` over a hash (trusted)

C-Flock's default row digest (`verity/ml/commitments/frame_v3/PROTOCOL.md` §6, "Row leaves"), with the hash a parameter
`G` (SHA-512 is the instance of record; the verifier's code is `Flock.HmRow.rowPrefix`). A row is `16 w` bits, `w` u16
words (`Row`), its bytes as `sha512/row-seg/v1` packs them (`RowSeg.packBits`, each word's low byte first), and

    P(w)   = ("verity/sha512-row/v1\0" ‖ u8 1 ‖ u8 16 ‖ u32be w)   zero-padded to 128 bytes
    digest = G(P(w) ‖ row bytes)

at role 1 and 16-bit words, C-Flock's. The one message the digest hashes (`RowV1.hashed`) fixes the row's bytes, and the
bytes fix the row only when it is whole words (`RowLeaf.Format.fits`): a 17-bit row and an 18-bit one ending in 0 share
them.
-/

namespace Hash.RowV1

/-- `P(w)`, the message's first block, for a row of `w` words. -/
def rowPrefix (w : ℕ) : Msg := padTo 128 (cstr "verity/sha512-row/v1" ++ [1, 16] ++ be 4 w)

/-- The message: `P(ℓ / 16) ‖ row bytes`. -/
def msg (row : Row) : Msg := rowPrefix (row.length / 16) ++ RowSeg.packBits row

/-- The row's digest. -/
def digest (G : Msg → Digest) (row : Row) : Digest := G (msg row)

/-- The message the digest hashed. -/
def hashed (row : Row) : List Msg := [msg row]

end Hash.RowV1
