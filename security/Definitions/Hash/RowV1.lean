module

public import Definitions.Hash.RowSeg

/-!
# `sha512/row/v1` over a hash (trusted)

The word row's digest (`verity/ml/commitments/frame_v3/PROTOCOL.md` §6, "Row leaves"), with the hash a parameter `G`
(SHA-512 is the instance of record; the reference is `rowleaf.sha512_row_digest`, the verifier's code
`HmRow.rowPrefix`). The verifier hashes every such row as a role 1 row of 16-bit words: a row is `16 · n` bits (`Row`),
its words in order, each least significant bit first, so its bytes as `sha512/row-seg/v1` packs them
(`RowSeg.packBits`) are the words little-endian, and

    prefix(n) = ("verity/sha512-row/v1\0" ‖ u8 1 ‖ u8 16 ‖ u32be n)   zero-padded to 128 bytes
    digest    = G(prefix(n) ‖ row bytes)

The prefix holds the word count, not the bit length, so the one message the digest hashes (`RowV1.hashed`) binds a row
of whole words (`RowLeaf.Format.Fits`): its bytes, two a word, give its length.
-/

@[expose] public section

namespace Hash.RowV1

/-- The role byte. -/
def ROLE : ℕ := 1

/-- The word width, in bits. -/
def WORD_BITS : ℕ := 16

/-- The message's first block, at `n` words. -/
def rowPrefix (n : ℕ) : Msg := padTo 128 (cstr "verity/sha512-row/v1" ++ be 1 ROLE ++ be 1 WORD_BITS ++ be 4 n)

/-- The message: `prefix(ℓ / 16) ‖ row bytes`. -/
def msg (row : Row) : Msg := rowPrefix (row.length / WORD_BITS) ++ RowSeg.packBits row

/-- The row's digest. -/
def digest (G : Msg → Digest) (row : Row) : Digest := G (msg row)

/-- The message the digest hashed. -/
def hashed (row : Row) : List Msg := [msg row]

end Hash.RowV1
