module

public import Definitions.Hash.RowSeg

/-! `sha512/row/v1`, the word row's digest over a hash `G`, which binds a row of whole 16-bit words. -/

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
