import Definitions.Hash.RowLeaf

/-!
# `EmbeddingRead_v1{V, H}`: the embedding's row read at a hidden position (trusted)

The reference of `verity/ml/commitments/gates/embedding.py`, on bits: what the read returns, not its gates. The table is
registered one token a row, each row a bit row (`H` bf16 words, each little-endian) under `hm96-sha512/row-seg/v1` in a
dense frame-v3-sha512 tree of `V` rows, so a row's rank is its token. The unit takes the token `tok` (its unsigned 32-bit
value), the registered root and domain, and the row, salt and path as witness (`RowLeaf.Opening`), and returns

    h  = row if tok < V, else 0x7FFF in every word                       (`h`)
    ok = tok ≥ V, or the row under its salt opens at rank tok to root    (`ok`)

The read's verifier (`verifier`) is the registered domain, the hm96 key and the format `seg`. The gates open at
`tok mod 2^D` with `D = merkle_depth V`, which is `tok` whenever `tok < V`, and with the domain's count `V` the frame
read's checks are `Tree.accepts`'s. Nothing ties this text to the gates yet: they are held to the Python reference by
`verity/ml/commitments/tests/test_boolean_embedding.py`, and the reference to this model by nothing but its reading.
-/

namespace Hash.EmbeddingRead

/-- The word a token outside the vocabulary reads, `0x7FFF`, as 16 bits least significant first. -/
def FILL_WORD : Row := (List.range 16).map fun j => Nat.testBit 0x7FFF j

/-- The row of `words` fill words. -/
def fill (words : ℕ) : Row := (List.replicate words FILL_WORD).flatten

/-- The read's verifier: the registered table's domain and hm96 key, under `hm96-sha512/row-seg/v1`. -/
def verifier (domain : Tree.Domain) (key : Msg) : RowLeaf.Verifier := ⟨domain, key, .seg⟩

/-- The row the read returns: the witness row for a token in the vocabulary, else `words` fill words. -/
def h (vocab words tok : ℕ) (o : RowLeaf.Opening) : Row := if tok < vocab then o.row else fill words

/-- The read's verdict: the token is outside the vocabulary, or the witness opens at rank `tok` to `root`. -/
def ok (H G : Msg → Digest) (domain : Tree.Domain) (key : Msg) (root : Digest) (vocab tok : ℕ) (o : RowLeaf.Opening) :
    Prop :=
  vocab ≤ tok ∨ RowLeaf.accepts H G (verifier domain key) root tok o

end Hash.EmbeddingRead
