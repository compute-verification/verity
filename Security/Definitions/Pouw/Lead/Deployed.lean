import Definitions.Pouw.Lead.Gap
import Definitions.Hash.Tree
import Definitions.Hash.Hm96

/-!
# The lead's deployed commitment and coin stream (trusted definitions)

The instance the honest lemma is about. These are specs of the served code, to be tied (difftest): no property or
difftest names them yet.

* **The commitment** (`l1wCommit`, the layout `l1w`, the served committer's): a call's A and its forward's words are
  hm96 instances, each hiding one frame-v3 tree under its own domain, `Hash.Tree.root` under `Hash.Hm96.leaf`. The
  served digest is the keyed leaf over the tree's top message; this model's hm96 leaf stands in for it until the
  instance roots are restated as keyed.
  - **Positions.** A window's instances are forward `f`'s call `j`'s A at `f · calls + j` (`aPos`), one a call, and
    forward `f`'s words at `2^63 + f` (`wordsPos`), one a forward, with `calls` the window's calls a forward.
  - **Domains.** The instance at position `p` with `count` leaves has the range-indexed domain of owner −1 over
    `RangeIndexedDomain(0, count)` bound to `SHA-512("pouw/lead/instance/v1" ‖ salt_w ‖ u64be(p))` (`instanceDom`),
    and each leaf is the inline frame-v3 leaf of its value at its own rank, position equal to rank
    (`dom.leaf(r, r, schema, v)`).
  - **The words.** A call's words are one stream (`wordsBytes`): the GEMM export's MMA tiles inside the call's
    `rowBlocks × colBlocks`, in the order (64-row tile, column tile, step, column block, row block), each tile's 128
    words in the MMA fragment's order, each word s32 little-endian. Each call's stream is cut into `chunkWords`-word
    chunks, the last one short (`callChunks`), and the forward's words instance holds its calls' chunks in call order
    under `pouw/lead/words/v1` (`CallForward.values`): a call's chunks sit at their own ranks, after its earlier calls'.
  - **A** is its served rows' row-major bytes cut into 1 KiB segments, the last one short, under `pouw/lead/a/v1`.
  - At a call, the forward's other calls' words are fixed (`CallForward`). The registered root is the forward's words
    instance's hm96 leaf and `msgs` every message that instance hashes; `aRoot` is A's hm96 leaf and `aMsgs` every
    message A's instance hashes. A is committed before the coins, so `aMsgs` is fixed, at the hash `H₀` it was
    committed under. A domain's id is the verifier's derivation from public parameters, not a message of the
    commitment.
* **The form classifier** (`l1wIsA`): a message is of A's form when A's instance hashes it.
* **The certifier's coin stream** (`certStream`): the call's coins where the certifier's feed puts them (`feedSlot`,
  the bit each starts at): the row words (row `i`'s `a₁[i, e]`, then its tag bits `b_t[i]` from step 1; step 0's tag
  bit enters no word), the G words (column `j`'s `G[e, j]`), then the lane-coin words (row tile `k`, step `t`, column
  `j`: `s_t[e, k, j]`, over the committed columns, which are the feed's computed ones at every served shape). The
  column sources the certifier sends beside the lane coins are a public function of them, not coins. The feed packs up
  to 32 coins into one u32 word (a row's bits, four G's, or three lane coins and a public byte); a `CoinStream` holds
  one coin a word, so `certStream`'s word `w` holds the coin starting at the feed's bit `w`, 128 a tile, within the
  node's storage.
-/

namespace Pouw.Lead

/-- **The deployed commitment's public parameters**, fixed before the coins: the verifier's hm96 key, the window's
live salt `salt_w` and its calls a forward, each instance's hm96 salt by its position, and the served GEMM's column tile
in column blocks (`BN / 8`), which orders the words' stream. -/
structure L1w where
  key : Hash.Msg
  salt : Hash.Msg
  calls : ℕ
  y : ℕ → Hash.Msg
  nb : ℕ

/-- The words leaves' schema. -/
def wordsSchema : String := "pouw/lead/words/v1"

/-- A's leaves' schema. -/
def aSchema : String := "pouw/lead/a/v1"

/-- A's segment: bytes a leaf of A (`ROW_SEG_BYTES`). -/
def aSegment : ℕ := 1024

/-- Word `(i, t, j)` of the words `w`, zero outside the call. -/
def wordAt (L : Layout) (w : Fin L.m → Fin L.S → Fin L.n → ℤ) (i t j : ℕ) : ℤ :=
  if h : i < L.m ∧ t < L.S ∧ j < L.n then w ⟨i, h.1⟩ ⟨t, h.2.1⟩ ⟨j, h.2.2⟩ else 0

/-- **The committed MMA tiles in stream order**, as (row block, column block, step), for a GEMM column tile of `nb`
column blocks: over the export's 64-row tiles `rt`, its column tiles `ct`, the steps `t`, then each tile's column block
`ni` and row block `w`, the MMA tile at row block `4 rt + w` and column block `nb ct + ni`, kept when it lies inside the
call's `rowBlocks × colBlocks`. -/
def wordsTiles (L : Layout) (nb : ℕ) : List (ℕ × ℕ × ℕ) :=
  (List.range ((L.rowBlocks + 3) / 4)).flatMap fun rt =>
    (List.range ((L.colBlocks + nb - 1) / nb)).flatMap fun ct =>
      (List.range L.S).flatMap fun t =>
        (List.range nb).flatMap fun ni =>
          ((List.range 4).filter fun w => 4 * rt + w < L.rowBlocks ∧ nb * ct + ni < L.colBlocks).map
            fun w => (4 * rt + w, nb * ct + ni, t)

/-- **An MMA tile's words in the fragment's order**: position `p = 16 g + 4 q + 2 eh + el` holds the word of row
`16 rb + 8 eh + g` and column `8 cb + 2 q + el`. -/
def tileFrag (L : Layout) (w : Fin L.m → Fin L.S → Fin L.n → ℤ) : ℕ × ℕ × ℕ → List ℤ
  | (rb, cb, t) => (List.range tileWords).map fun p =>
    wordAt L w (16 * rb + 8 * (p / 2 % 2) + p / 16) t (8 * cb + 2 * (p / 4 % 4) + p % 2)

/-- A committed word's bytes: its 32-bit register's, two's complement, little-endian. -/
def wordBytes (x : ℤ) : Hash.Msg := (Hash.be 4 (x % 2 ^ 32).toNat).reverse

/-- **The call's committed words' bytes**: the stream's tiles' words in order, each s32 little-endian. -/
def wordsBytes (L : Layout) (nb : ℕ) (w : Fin L.m → Fin L.S → Fin L.n → ℤ) : Hash.Msg :=
  ((wordsTiles L nb).flatMap (tileFrag L w)).flatMap wordBytes

/-- A call's words leaves: its stream's `chunkWords`-word chunks. -/
def wordsLeaves (L : Layout) (nb : ℕ) : ℕ := (tileWords * (wordsTiles L nb).length + L.chunkWords - 1) / L.chunkWords

/-- A call's words chunk `r`: the stream's bytes from `4 · chunkWords · r`, `4 · chunkWords` of them (the last chunk
short). -/
def wordsChunk (L : Layout) (nb : ℕ) (w : Fin L.m → Fin L.S → Fin L.n → ℤ) (r : ℕ) : Hash.Msg :=
  ((wordsBytes L nb w).drop (4 * L.chunkWords * r)).take (4 * L.chunkWords)

/-- A call's committed words: its layout and its words. -/
abbrev CallWords : Type := Σ L : Layout, (Fin L.m → Fin L.S → Fin L.n → ℤ)

/-- A call's words chunks, in order. -/
def callChunks (nb : ℕ) (c : CallWords) : List Hash.Msg :=
  (List.range (wordsLeaves c.1 nb)).map (wordsChunk c.1 nb c.2)

/-- **A call's forward**: the forward `f` and its other calls' committed words in call order, those before the call
(`pre`) and those after it (`post`). The call is the forward's call `pre.length`. -/
structure CallForward where
  f : ℕ
  pre : List CallWords
  post : List CallWords

/-- The forward's words leaves before the call: its earlier calls' chunks. -/
def CallForward.before (F : CallForward) (nb : ℕ) : ℕ := (F.pre.map fun c => wordsLeaves c.1 nb).sum

/-- The forward's words leaves, the call's at the layout `L` among them: every call's chunks. -/
def CallForward.leaves (F : CallForward) (nb : ℕ) (L : Layout) : ℕ :=
  F.before nb + wordsLeaves L nb + (F.post.map fun c => wordsLeaves c.1 nb).sum

/-- **The forward's words leaves' values**, the call's words `w` among them: each call's chunks, in call order. -/
def CallForward.values (F : CallForward) (nb : ℕ) (L : Layout) (w : Fin L.m → Fin L.S → Fin L.n → ℤ) : List Hash.Msg :=
  F.pre.flatMap (callChunks nb) ++ callChunks nb ⟨L, w⟩ ++ F.post.flatMap (callChunks nb)

/-- The forward's words leaf value at rank `r`. -/
def CallForward.value (F : CallForward) (nb : ℕ) (L : Layout) (w : Fin L.m → Fin L.S → Fin L.n → ℤ) (r : ℕ) : Hash.Msg :=
  (F.values nb L w).getD r []

/-- A's bytes: its served rows', row-major. -/
def aBytes (L : Layout) (A : Fin L.rows → Fin L.K → UInt8) : Hash.Msg := (List.ofFn fun i => List.ofFn (A i)).flatten

/-- A's leaves: its bytes' `aSegment`-byte segments. -/
def aLeaves (L : Layout) : ℕ := (L.rows * L.K + aSegment - 1) / aSegment

/-- A's segment `r`: its bytes from `aSegment · r`, `aSegment` of them (the last segment short). -/
def aSeg (L : Layout) (A : Fin L.rows → Fin L.K → UInt8) (r : ℕ) : Hash.Msg :=
  ((aBytes L A).drop (aSegment * r)).take aSegment

/-- Forward `f`'s words instance's position: `2^63 + f`. -/
def wordsPos (f : ℕ) : ℕ := 2 ^ 63 + f

/-- The A instance's position of forward `f`'s call `j`, at `calls` calls a forward: `f · calls + j`. -/
def aPos (calls f j : ℕ) : ℕ := f * calls + j

/-- **An instance's domain** at the hash `H`: `count` leaves in the range-indexed domain of owner −1 bound to
`SHA-512("pouw/lead/instance/v1" ‖ salt_w ‖ u64be(p))`, `p` the instance's position. -/
def instanceDom (H : Hash.Msg → Hash.Digest) (salt : Hash.Msg) (p count : ℕ) : Hash.Tree.Domain :=
  Hash.Tree.Domain.range H (H (Hash.utf8 "pouw/lead/instance/v1" ++ salt ++ Hash.be 8 p)) (-1) count

/-- The words' domain: the forward's words leaves, at its words instance's position. -/
def L1w.wordsDom (P : L1w) (H : Hash.Msg → Hash.Digest) (F : CallForward) (L : Layout) : Hash.Tree.Domain :=
  instanceDom H P.salt (wordsPos F.f) (F.leaves P.nb L)

/-- A's domain: its segments, at the call's A instance's position. -/
def L1w.aDom (P : L1w) (H : Hash.Msg → Hash.Digest) (F : CallForward) (L : Layout) : Hash.Tree.Domain :=
  instanceDom H P.salt (aPos P.calls F.f F.pre.length) (aLeaves L)

/-- **The lead's deployed commitment** (`l1w`) of a call of the forward `F`, under the parameters `P`, of its A,
committed under `H₀`: the forward's words tree's root under its hm96 leaf, A's tree's root under its own, and every
message each instance hashes. -/
def l1wCommit (L : Layout) (P : L1w) (F : CallForward) (H₀ : Hash.Msg → Hash.Digest) (A : Fin L.rows → Fin L.K → UInt8) :
    Commit L where
  msgs H w := Hash.Tree.msgs H (P.wordsDom H F L) wordsSchema (F.value P.nb L w) ++
    Hash.Hm96.hashed H P.key (Hash.Tree.root H (P.wordsDom H F L) wordsSchema (F.value P.nb L w)) (P.y (wordsPos F.f))
  root H w :=
    Hash.Hm96.leaf H P.key (Hash.Tree.root H (P.wordsDom H F L) wordsSchema (F.value P.nb L w)) (P.y (wordsPos F.f))
  aMsgs := Hash.Tree.msgs H₀ (P.aDom H₀ F L) aSchema (aSeg L A) ++
    Hash.Hm96.hashed H₀ P.key (Hash.Tree.root H₀ (P.aDom H₀ F L) aSchema (aSeg L A))
      (P.y (aPos P.calls F.f F.pre.length))
  aRoot H := Hash.Hm96.leaf H P.key (Hash.Tree.root H (P.aDom H F L) aSchema (aSeg L A))
    (P.y (aPos P.calls F.f F.pre.length))
  chunks := wordsLeaves L P.nb

/-- **The form classifier** of `l1wCommit`: a message is of A's form when A's instance hashes it. -/
def l1wIsA (L : Layout) (P : L1w) (F : CallForward) (H₀ : Hash.Msg → Hash.Digest) (A : Fin L.rows → Fin L.K → UInt8)
    (x : Hash.Msg) : Bool :=
  decide (x ∈ (l1wCommit L P F H₀ A).aMsgs)

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
`tileWords · c + p` (zero where none does), for the node's storage's `L.mem` tiles (past them nothing is streamed). -/
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
