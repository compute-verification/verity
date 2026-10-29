import Definitions.Pouw.Lead.Gap
import Definitions.Hash.Tree
import Definitions.Hash.Hm96

/-!
# The lead's deployed commitment and coin stream (trusted definitions)

The instance the honest lemma is about, as specs the code is tied to by properties:

* **The commitment** (`l1wCommit`, the layout `l1w`): a call's words and its A rows are two hm96 instances, each hiding
  one frame-v3 tree under its own domain (`Pouw.Assumptions.SubtreeLayout`), `Hash.Tree.root` under `Hash.Hm96.leaf`.
  The words tree's leaves are the words' bytes in chunks of `chunkWords` words, `wordChunks L` of them; A's are its
  rows, one a leaf. Each leaf's position is its rank, as the code's `PouwCommit.Domain.leaf` frames it. The registered
  root is the words instance's hm96 leaf, `msgs` every message the words instance hashes, and `aMsgs` every message A's
  instance hashes. A is committed before the coins, so `aMsgs` is fixed, at the hash `H₀` it was committed under.
* **The form classifier** (`l1wIsA`): a message is of A's form when A's instance hashes it.
* **The certifier's coin stream** (`certStream`): the call's coins where the certifier's feed puts them (`feedSlot`,
  the bit each starts at): the row words (row `i`'s `a₁[i, e]`, then its tag bits `b_t[i]` from step 1; step 0's tag
  bit enters no word), the G words (column `j`'s `G[e, j]`), then the lane-coin words (row tile `k`, step `t`, column
  `j`: `s_t[e, k, j]`, over the committed columns, which are the feed's computed ones at every served shape). The
  column sources the certifier sends beside the lane coins are a public function of them, not coins. The feed packs up
  to 32 coins into one u32 word (a row's bits, four G's, or three lane coins and a public byte); a `CoinStream` holds
  one coin a word, so `certStream`'s word `w` holds the coin starting at the feed's bit `w`, 128 a tile, within the
  node's memory.
-/

namespace Pouw.Lead

/-- **The deployed commitment's public parameters**, fixed before the coins: the verifier's hm96 key, and for the words
instance and A's its frame-v3 domain id (the verifier's, from its binding, owner, positions and count), its leaves'
schema and its hm96 salt. -/
structure L1w where
  key : Hash.Msg
  idw : Hash.Digest
  ida : Hash.Digest
  sw : String
  sa : String
  yw : Hash.Msg
  ya : Hash.Msg

/-- The words' domain: `wordChunks L` leaves, each at the position of its rank. -/
def L1w.wordsDom (P : L1w) (L : Layout) : Hash.Tree.Domain := ⟨P.idw, wordChunks L, id⟩

/-- A's domain: `L.rows` leaves, each at the position of its rank. -/
def L1w.aDom (P : L1w) (L : Layout) : Hash.Tree.Domain := ⟨P.ida, L.rows, id⟩

/-- A committed word's bytes: its 32-bit register's, two's complement, little-endian. -/
def wordBytes (x : ℤ) : Hash.Msg := (Hash.be 4 (x % 2 ^ 32).toNat).reverse

/-- The call's words' bytes, in `word`'s order: row, then step, then column. -/
def wordsBytes (L : Layout) (w : Fin L.m → Fin L.S → Fin L.n → ℤ) : Hash.Msg :=
  (List.ofFn fun i => (List.ofFn fun t => (List.ofFn fun j => wordBytes (w i t j)).flatten).flatten).flatten

/-- Words chunk `r`: the words' bytes from `4 · chunkWords · r`, `4 · chunkWords` of them (the last chunk short). -/
def wordsChunk (L : Layout) (w : Fin L.m → Fin L.S → Fin L.n → ℤ) (r : ℕ) : Hash.Msg :=
  ((wordsBytes L w).drop (4 * L.chunkWords * r)).take (4 * L.chunkWords)

/-- A's row `r`, `K` bytes (none past its rows). -/
def aRow (L : Layout) (A : Fin L.rows → Fin L.K → UInt8) (r : ℕ) : Hash.Msg :=
  if h : r < L.rows then List.ofFn (A ⟨r, h⟩) else []

/-- **The lead's deployed commitment** (`l1w`) under the parameters `P`, of A's rows `A`, committed under `H₀`: the
words tree's root under the words' hm96 leaf, and every message each instance hashes. -/
def l1wCommit (L : Layout) (P : L1w) (H₀ : Hash.Msg → Hash.Digest) (A : Fin L.rows → Fin L.K → UInt8) : Commit L where
  msgs H w := Hash.Tree.msgs H (P.wordsDom L) P.sw (wordsChunk L w) ++
    Hash.Hm96.hashed H P.key (Hash.Tree.root H (P.wordsDom L) P.sw (wordsChunk L w)) P.yw
  root H w := Hash.Hm96.leaf H P.key (Hash.Tree.root H (P.wordsDom L) P.sw (wordsChunk L w)) P.yw
  aMsgs := Hash.Tree.msgs H₀ (P.aDom L) P.sa (aRow L A) ++
    Hash.Hm96.hashed H₀ P.key (Hash.Tree.root H₀ (P.aDom L) P.sa (aRow L A)) P.ya
  chunks := wordChunks L

/-- **The form classifier** of `l1wCommit`: a message is of A's form when A's instance hashes it. -/
def l1wIsA (L : Layout) (P : L1w) (H₀ : Hash.Msg → Hash.Digest) (A : Fin L.rows → Fin L.K → UInt8) (x : Hash.Msg) :
    Bool :=
  decide (x ∈ (l1wCommit L P H₀ A).aMsgs)

/-- The feed's row words a row: `⌈(S + 3)/32⌉`. -/
def Layout.feedRowWords (L : Layout) : ℕ := (L.S + 3 + 31) / 32

/-- **A coin's bit in the certifier's feed**: the bit its value starts at, counting the feed's u32 words from its
start and each word's bits from the least significant. Row `i`'s `feedRowWords` words come first, `a₁[i, e]` at bit `e`
and `b_t[i]` at bit `t + 3` for `t ≥ 1` (step 0's tag bit enters no word); then column `j`'s G word, `G[e, j]` its
byte `e`; then lane-coin word `(k · S + t) · n + j`, `s_t[e, k, j]` its byte `e`. -/
def feedSlot (L : Layout) : CoinIdx L → Option ℕ
  | .inl (i, t) => if (t : ℕ) = 0 then none else some (32 * (L.feedRowWords * i) + t + 3)
  | .inr (.inl (i, e)) => some (32 * (L.feedRowWords * i) + e)
  | .inr (.inr (.inl (e, j))) => some (32 * (L.feedRowWords * L.m + j) + 8 * e)
  | .inr (.inr (.inr (t, k, e, j))) => some (32 * (L.feedRowWords * L.m + L.n + (k * L.S + t) * L.n + j) + 8 * e)

open Classical in
/-- **The certifier's coin stream** for a call: word `p` of tile `c` holds the coin that starts at the feed's bit
`tileWords · c + p` (zero where none does), for the node's memory's `L.mem` tiles (past them nothing is streamed). -/
noncomputable def certStream (L : Layout) : CoinStream L where
  ι c p := if h : c < L.mem ∧ ∃ x, feedSlot L x = some (tileWords * c + p) then some h.2.choose else none
  supp c p hc := by simp [not_lt.2 hc]
  once c p c' p' x h h' := by
    have slot : ∀ c (p : Fin tileWords),
        (if h : c < L.mem ∧ ∃ x, feedSlot L x = some (tileWords * c + p) then some h.2.choose else none) = some x →
          feedSlot L x = some (tileWords * c + p) := by
      intro c p h
      split_ifs at h with hc
      exact Option.some.inj h ▸ hc.2.choose_spec
    have e := Option.some.inj ((slot c p h).symm.trans (slot c' p' h'))
    have hp := p.isLt
    have hp' := p'.isLt
    unfold tileWords at e hp hp'
    exact ⟨by omega, Fin.ext (by omega)⟩

end Pouw.Lead
