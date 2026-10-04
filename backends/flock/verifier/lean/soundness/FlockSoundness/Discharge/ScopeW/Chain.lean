import FlockSoundness.Discharge.ScopeW.Delta

set_option Elab.async false

/-!
# A row's commit string at each widened shape, from its slots

`Hidden.port_commit_rowsP` is a row's `b ‖ c` from its slots for a row of whole blocks. This file gives it at the two
shapes whose last data are not a whole block before the `hm96` slot: a short `sha512/row/v2` row (`RowV2Short`), whose
bytes and padding are the `hm96` slot's one message block, and a whole-block segmented row (`RowSegWhole`), whose
segments are chains of their own and whose outer message holds the segments' digests.

* `rowMsgOf`: the bytes a row's digest hashes, from its bytes: the row's first block (`HmRow.prefixOf`), then the bytes,
  or at a segmented row each segment's digest of `segMsgOf` (`S(j, s, m)` and the segment's bytes);
* `chain_bits`: a chain of compressions on a padded message's blocks outputs its chaining values;
* `sha_stepW`, `hm_commitW`: a row's compression computes SHA-512's compression; its `hm96` slot the commit string of a
  message whose last chaining value and block it reads;
* `port_commit_shortW`: a short row's `hm96` slot, its `cv` the prefix's midstate, its block the row's bytes then its
  padding;
* `port_commit_segW`: a segmented row's segment chains (midstate, data blocks, padding block) and outer chain (the row
  prefix's midstate, the digests' blocks, the `hm96` slot with the last digest's 64 bytes when `s` is odd, then the
  padding).

The copies each lemma takes are lane B's (`row_copies_shortW`, `row_copies_segW`) with the block's bits `w`.
-/

namespace FlockSoundness.Discharge.ScopeW

open Flock FlockLevel3 Hm
open Flock.HmNets hiding and

/-! ## The bytes a row's digest hashes -/

/-- Segment `j`'s bytes of a segmented row's bytes: from byte `1024 j`, `m / 8` of them for the segment's `m` bits. -/
def segData (p : Port) (data : List UInt8) (j : ℕ) : List UInt8 :=
  (data.drop (1024 * j)).take (((segsOf p)[j]!).1 / 8)

/-- **Segment `j`'s message** (`HmRow.segDigest`'s inner hash's input): `S(j, s, m)`, then the segment's bytes. -/
def segMsgOf (p : Port) (data : List UInt8) (j : ℕ) : List UInt8 :=
  (HmRow.segmentPrefix j (segsOf p).size ((segsOf p)[j]!).1).data.toList ++ segData p data j

/-- **The bytes a row's digest hashes, from the row's bytes `data`**: the row's first block (`HmRow.prefixOf`), then at a
segmented row each segment's digest (`segMsgOf`), at any other row the bytes. -/
def rowMsgOf (p : Port) (data : List UInt8) : List UInt8 :=
  (HmRow.prefixOf p).data.toList ++
    if p.segs.isSome then
      (List.range (segsOf p).size).flatMap fun j => Binding.digBytes (Refine.H512 (segMsgOf p data j))
    else data

theorem rowMsgOf_unseg {p : Port} (hs : p.segs = none) (data : List UInt8) :
    rowMsgOf p data = (HmRow.prefixOf p).data.toList ++ data := by
  simp [rowMsgOf, hs]

/-! ## Chains of compressions -/

/-- **A chain of compressions outputs a padded message's chaining values**: when compression `i < n` (columns `cv i`,
`msg i`, `out i`) computes SHA-512's compression (`hstep`), the first `cv` holds `P`'s chaining value after block 0,
each next `cv` the previous output, and each `msg` the next block of `P`, compression `i`'s output is `P`'s chaining
value after block `i + 1`. -/
theorem chain_bits (w : ℕ → Bool) (cv msg out : ℕ → ℕ) (P : ByteArray) (n : ℕ)
    (hstep : ∀ i < n, ∀ A B : List (BitVec 64), A.length = 8 → B.length = 16 →
      (∀ q < 512, w (cv i + q) = wbit A q) → (∀ q < 1024, w (msg i + q) = wbit B q) →
      ∀ q < 512, w (out i + q) = wbit (compressV A B) q)
    (hcv0 : ∀ q < 512, w (cv 0 + q) = wbit (chainV P 1) q)
    (hchain : ∀ i, i + 1 < n → ∀ q < 512, w (cv (i + 1) + q) = w (out i + q))
    (hmsg : ∀ i < n, ∀ q < 1024, w (msg i + q) = wbit (blockV P (i + 1)) q) :
    ∀ i < n, ∀ q < 512, w (out i + q) = wbit (chainV P (i + 2)) q := by
  sorry

/-- **Compression `k` of row `p` computes SHA-512's compression** of what its `cv` and `msg` inputs hold (`cvAt`,
`msgAt`), into its output (`outAt`), when its `sha512x3` slot holds its net's rows. -/
theorem sha_stepW (c : Circuit) (w : ℕ → Bool) (g p k : ℕ) (hp : p < c.ports.size) (hk : k < c.ports[p]!.chunks)
    (hck : HmNets.check c = .ok ()) (hsha : LaidRows (shaLaid ()) 8192 fun j => w (shaBase c g p k + j))
    {A B : List (BitVec 64)} (hA : A.length = 8) (hB : B.length = 16)
    (hcv : ∀ q < 512, w (cvAt c g p k + q) = wbit A q) (hmsg : ∀ q < 1024, w (msgAt c g p k + q) = wbit B q) :
    ∀ q < 512, w (outAt c g p k + q) = wbit (compressV A B) q := by
  sorry

/-- **Row `p`'s `hm96` slot outputs the commit string of a message `r`** whose chaining value before its last block its
`cv` inputs hold and whose last block its `pad` inputs hold, under the salt its salt inputs hold, when the slot holds its
net's rows. -/
theorem hm_commitW (c : Circuit) (w : ℕ → Bool) (g p : ℕ) (hck : HmNets.check c = .ok ())
    (hhm : LaidRows (hmLaid ()) 4096 fun j => w (c.slotCol c.hmNet g p + j)) (r : List UInt8)
    (hcv : ∀ q < 512, w (c.slotCol c.hmNet g p + hmIn c 0 + q) = wbit (lastCv ⟨r.toArray⟩) q)
    (hpad : ∀ q < 1024, w (c.slotCol c.hmNet g p + hmIn c 1 + q) = wbit (lastBlock ⟨r.toArray⟩) q) :
    bytesOf w 128 (bcCol c g p) = Binding.commitString r (bytesOf w 192 (saltCol c g p)) := by
  sorry

/-! ## A short row -/

/-- **A short row's commit string** (`RowV2Short`): its `hm96` slot, which holds its net's rows, its `cv` the row
prefix's midstate and its block past the row's `2 · words` bytes the row's padding (`row_copies_shortW`), outputs the
commit string of the prefix and the bytes at the row's message columns. -/
theorem port_commit_shortW (c : Circuit) (w : ℕ → Bool) (g p : ℕ) (hp : p < c.ports.size)
    (h : RowV2Short c.ports[p]!) (hck : HmNets.check c = .ok ())
    (hhm : LaidRows (hmLaid ()) 4096 fun j => w (c.slotCol c.hmNet g p + j))
    (hcvh : ∀ t < 512, w (c.slotCol c.hmNet g p + hmIn c 0 + t) =
      wordsBit (Sha512.compress Sha512.iv (HmRow.prefixOf c.ports[p]!) 0) t)
    (hpad : ∀ t < 1024, 2 * c.ports[p]!.words ≤ (HmRow.blockBit t).1 →
      w (c.slotCol c.hmNet g p + hmIn c 1 + t) = tailBit (HmRow.tailOf c.ports[p]!) (2 * c.ports[p]!.words) t) :
    bytesOf w 128 (bcCol c g p) = Binding.commitString
      (rowMsgOf c.ports[p]! (bytesOf w (2 * c.ports[p]!.words) (Typed.msgCol c g p))) (bytesOf w 192 (saltCol c g p)) := by
  sorry

/-! ## A segmented row -/

/-- **A whole-block segmented row's commit string** (`RowSegWhole`): when its slots hold their nets' rows and its
compressions' inputs hold `row_copies_segW`'s copies (each segment's `cv` its midstate, its chain, its padding block;
the outer `cv` the row prefix's midstate, the outer chain, the digests' bytes from the segments' last outputs, the
outer padding), its `hm96` slot outputs the commit string of the row prefix and each segment's digest of its
message, `S(j, s, m)` and the segment's bytes at the row's message columns. -/
theorem port_commit_segW (c : Circuit) (w : ℕ → Bool) (g p : ℕ) (hp : p < c.ports.size)
    (h : RowSegWhole c.ports[p]!) (hck : HmNets.check c = .ok ())
    (hsha : ∀ k < c.ports[p]!.chunks, LaidRows (shaLaid ()) 8192 fun j => w (shaBase c g p k + j))
    (hhm : LaidRows (hmLaid ()) 4096 fun j => w (c.slotCol c.hmNet g p + j))
    (hscv : ∀ j < (segsOf c.ports[p]!).size, ∀ t < 512,
      w (cvAt c g p (segFirst c.ports[p]! j) + t) = wordsBit (HmRow.segMidstate c.ports[p]! j) t)
    (hschain : ∀ j < (segsOf c.ports[p]!).size, ∀ i < ((segsOf c.ports[p]!)[j]!).1 / 1024, ∀ t < 512,
      w (cvAt c g p (segFirst c.ports[p]! j + i + 1) + t) = w (outAt c g p (segFirst c.ports[p]! j + i) + t))
    (hspad : ∀ j < (segsOf c.ports[p]!).size, ∀ t < 1024,
      w (msgAt c g p (segFirst c.ports[p]! j + ((segsOf c.ports[p]!)[j]!).1 / 1024) + t) =
        wordsBit (HmRow.wordsBE (HmRow.padBlock (128 + ((segsOf c.ports[p]!)[j]!).1 / 8)) 16) t)
    (hocv : ∀ t < 512, w (cvAt c g p (segFirst c.ports[p]! (segsOf c.ports[p]!).size) + t) =
      wordsBit (Sha512.compress Sha512.iv (HmRow.prefixOf c.ports[p]!) 0) t)
    (hochain : ∀ k, segFirst c.ports[p]! (segsOf c.ports[p]!).size ≤ k → k < c.ports[p]!.chunks → ∀ t < 512,
      w (cvAt c g p (k + 1) + t) = w (outAt c g p k + t))
    (hdig : ∀ e < 64 * (segsOf c.ports[p]!).size, ∀ b < 8,
      w (msgAt c g p (segFirst c.ports[p]! (segsOf c.ports[p]!).size + e / 128) +
          (64 * (e % 128 / 8) + 56 - 8 * (e % 128 % 8) + b)) =
        w (outAt c g p (segFirst c.ports[p]! (e / 64 + 1) - 1) + (64 * (e % 64 / 8) + 56 - 8 * (e % 64 % 8) + b)))
    (htail : ∀ t < 1024, 64 * (segsOf c.ports[p]!).size % 128 ≤ (HmRow.blockBit t).1 →
      w (msgAt c g p c.ports[p]!.chunks + t) =
        tailBit (HmRow.tailOf c.ports[p]!) (64 * (segsOf c.ports[p]!).size % 128) t) :
    bytesOf w 128 (bcCol c g p) = Binding.commitString
      (rowMsgOf c.ports[p]! (bytesOf w (2 * c.ports[p]!.words) (Typed.msgCol c g p))) (bytesOf w 192 (saltCol c g p)) := by
  sorry

end FlockSoundness.Discharge.ScopeW
