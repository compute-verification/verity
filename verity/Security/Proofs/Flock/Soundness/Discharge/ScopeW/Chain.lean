import Proofs.Flock.Soundness.Discharge.ScopeW.Delta
import Proofs.Flock.Soundness.Discharge.ScopeW.RowBytes

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

namespace Chain

theorem padTail_data (total : ℕ) :
    (HmRow.padTail total).data.toList = [128] ++ List.replicate ((239 - total % 128) % 128) 0 ++ lenBytes total := by
  simp only [HmRow.padTail, Std.Legacy.Range.forIn_eq_forIn_range', List.forIn_pure_yield_eq_foldl, pure_bind,
    Id.run_pure, Std.Legacy.Range.size, Nat.reduceSub, Nat.reduceAdd, Nat.sub_zero, Nat.div_one]
  rw [foldl_push_data (fun i => ((8 * total) >>> (8 * (15 - i)) % 256).toUInt8), lenBytes]
  simp only [Nat.mul_comm 8 total]
  rfl

theorem length_padTail (total : ℕ) : (HmRow.padTail total).data.toList.length = (239 - total % 128) % 128 + 17 := by
  rw [padTail_data]
  simp only [List.length_append, List.length_cons, List.length_nil, List.length_replicate, lenBytes, List.length_map,
    List.length_range']
  omega

/-- **SHA-512's padded message is the message then `HmRow.padTail` of its length.** -/
theorem padData_list (M : List UInt8) :
    (padData ⟨M.toArray⟩).data.toList = M ++ (HmRow.padTail M.length).data.toList := by
  have hs : (⟨M.toArray⟩ : ByteArray).size = M.length := by simp [ByteArray.size]
  rw [padData_data, padTail_data, hs]
  have hz : zeroPad M.length = (239 - M.length % 128) % 128 := by unfold zeroPad; omega
  rw [hz]
  simp only [List.append_assoc, List.cons_append, List.nil_append]

theorem size_padData_list (M : List UInt8) :
    (padData ⟨M.toArray⟩).size = M.length + ((239 - M.length % 128) % 128 + 17) := by
  rw [← ByteArray.size_data, ← Array.length_toList, padData_list, List.length_append, length_padTail]

theorem bitL_append_left {A B : List UInt8} {x : ℕ} (h : x / 8 < A.length) : bitL (A ++ B) x = bitL A x := by
  rw [bitL, bitL, List.getD_append _ _ _ _ h]

theorem bitL_append_right {A B : List UInt8} {x : ℕ} (h : A.length ≤ x / 8) :
    bitL (A ++ B) x = bitL B (x - 8 * A.length) := by
  rw [bitL, bitL, List.getD_append_right _ _ _ _ h,
    show (x - 8 * A.length) / 8 = x / 8 - A.length by omega, show (x - 8 * A.length) % 8 = x % 8 by omega]

theorem tailBit_eq (tl : ByteArray) (r t : ℕ) :
    tailBit tl r t = bitL tl.data.toList (8 * ((HmRow.blockBit t).1 - r) + (HmRow.blockBit t).2) := by
  have hb : (HmRow.blockBit t).2 < 8 := by simp only [HmRow.blockBit]; omega
  rw [tailBit, bitL, get!_eq_getD, show (8 * ((HmRow.blockBit t).1 - r) + (HmRow.blockBit t).2) / 8 =
      (HmRow.blockBit t).1 - r by omega, show (8 * ((HmRow.blockBit t).1 - r) + (HmRow.blockBit t).2) % 8 =
      (HmRow.blockBit t).2 by omega, Nat.testBit_eq_decide_div_mod_eq, Nat.shiftRight_eq_div_pow]
  exact Bool.beq_eq_decide_eq _ _

theorem blockBit_lt {q : ℕ} (hq : q < 1024) : (HmRow.blockBit q).1 < 128 ∧ (HmRow.blockBit q).2 < 8 := by
  simp only [HmRow.blockBit]; omega

/-- Column `q` of sixteen big-endian words of a byte list at `off` is bit `(blockBit q).2` of byte
`off + (blockBit q).1`. -/
theorem wbit_wordL (l : List UInt8) (off : ℕ) {q : ℕ} (hq : q < 1024) :
    wbit ((List.range 16).map fun i => (wordL l off i).toBitVec) q =
      bitL l (8 * (off + (HmRow.blockBit q).1) + (HmRow.blockBit q).2) := by
  rw [wbit, List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range (show q / 64 < 16 by omega),
    Option.map_some, Option.getD_some, getLsbD_wordL _ _ _ (Nat.mod_lt _ (by norm_num)), bitL, HmRow.blockBit]
  dsimp only
  rw [show (8 * (off + (8 * (q / 64) + 7 - q % 64 / 8)) + q % 8) / 8 = off + 8 * (q / 64) + (7 - q % 64 / 8)
      by omega, show (8 * (off + (8 * (q / 64) + 7 - q % 64 / 8)) + q % 8) % 8 = q % 64 % 8 by omega]

/-- Column `q` of block `k` of `P` is bit `(blockBit q).2` of `P`'s byte `128 k + (blockBit q).1`. -/
theorem wbit_blockV (P : ByteArray) (k : ℕ) {q : ℕ} (hq : q < 1024) :
    wbit (blockV P k) q = bitL P.data.toList (8 * (128 * k + (HmRow.blockBit q).1) + (HmRow.blockBit q).2) := by
  rw [blockV]
  simp only [wordBE_eq_wordL]
  exact wbit_wordL _ _ hq

theorem wordsBit_wordsBE (b : ByteArray) {q : ℕ} (hq : q < 1024) :
    wordsBit (HmRow.wordsBE b 16) q = bitL b.data.toList (8 * (HmRow.blockBit q).1 + (HmRow.blockBit q).2) := by
  rw [wordsBit_eq _ (show q < 64 * 16 by omega)]
  rw [List.map_congr_left (g := fun i => (wordL b.data.toList 0 i).toBitVec) fun i hi => by
    rw [wordsBE_getElem! (List.mem_range.1 hi)]]
  rw [wbit_wordL _ _ hq, Nat.zero_add]

/-- After a 128-byte first block, the chain is the block's midstate. -/
theorem chainV_one_pre {pre : ByteArray} (hpre : pre.data.toList.length = 128) (rest : List UInt8) :
    chainV (padData ⟨(pre.data.toList ++ rest).toArray⟩) 1 =
      (List.range 8).map fun i => (Sha512.compress Sha512.iv pre 0)[i]!.toBitVec := by
  have h1 : Sha512.compress Sha512.iv (padData ⟨(pre.data.toList ++ rest).toArray⟩) 0 =
      Sha512.compress Sha512.iv pre 0 := by
    refine compress_congr _ fun k hk => ?_
    rw [get!_eq_getD, get!_eq_getD, padData_list, Nat.zero_add, List.append_assoc,
      List.getD_append _ _ _ _ (by rw [hpre]; exact hk)]
  rw [chainV, show chain (padData ⟨(pre.data.toList ++ rest).toArray⟩) 1 =
    Sha512.compress Sha512.iv (padData ⟨(pre.data.toList ++ rest).toArray⟩) 0 from rfl, h1]

theorem wordsBit_mid {pre : ByteArray} (hpre : pre.data.toList.length = 128) (rest : List UInt8) {q : ℕ}
    (hq : q < 512) :
    wordsBit (Sha512.compress Sha512.iv pre 0) q = wbit (chainV (padData ⟨(pre.data.toList ++ rest).toArray⟩) 1) q := by
  rw [wordsBit_eq _ (show q < 64 * 8 by omega), chainV_one_pre hpre]

/-- **A digest's bits are its message's last chaining value's**, in `wordBit` order. -/
theorem bitL_hash_chain (M : List UInt8) {t : ℕ} (ht : t < 512) :
    bitL (Sha512.hash ⟨M.toArray⟩).data.toList t =
      wbit (chainV (padData ⟨M.toArray⟩) ((padData ⟨M.toArray⟩).size / 128 - 1 + 1)) (wordBit t) := by
  rw [bitL_hash _ ht, lastCv_eq, lastBlock_eq, chainV_succ]

theorem getD_flatMap_range {f : ℕ → List UInt8} (hf : ∀ j, (f j).length = 64) :
    ∀ (s : ℕ) {e : ℕ}, e < 64 * s → ((List.range s).flatMap f).getD e 0 = (f (e / 64)).getD (e % 64) 0
  | 0, e, he => absurd he (by omega)
  | s + 1, e, he => by
    rw [List.range_succ, List.flatMap_append]
    by_cases h : e < 64 * s
    · rw [List.getD_append _ _ _ _ (by rw [RowBytes.length_flatMap_range hf]; omega), getD_flatMap_range hf s h]
    · rw [List.getD_append_right _ _ _ _ (by rw [RowBytes.length_flatMap_range hf]; omega),
        RowBytes.length_flatMap_range hf, List.flatMap_cons, List.flatMap_nil, List.append_nil,
        show e / 64 = s by omega, show e % 64 = e - 64 * s by omega, Nat.mul_comm]

end Chain

open Chain

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
  have hlen8 : ∀ k, (chainV P k).length = 8 := fun k => by rw [chainV, List.length_map, List.length_range]
  have hlen16 : ∀ k, (blockV P k).length = 16 := fun k => by rw [blockV, List.length_map, List.length_range]
  intro i
  induction i with
  | zero =>
    intro hi q hq
    rw [hstep 0 hi _ _ (hlen8 1) (hlen16 1) hcv0 (hmsg 0 hi) q hq, chainV_succ P 1]
  | succ i ih =>
    intro hi q hq
    rw [hstep (i + 1) hi _ _ (hlen8 (i + 2)) (hlen16 (i + 2))
      (fun q hq => by rw [hchain i hi q hq, ih (by omega) q hq]) (hmsg (i + 1) hi) q hq, chainV_succ P (i + 2)]

/-- **Compression `k` of row `p` computes SHA-512's compression** of what its `cv` and `msg` inputs hold (`cvAt`,
`msgAt`), into its output (`outAt`), when its `sha512x3` slot holds its net's rows. -/
theorem sha_stepW (c : Circuit) (w : ℕ → Bool) (g p k : ℕ) (hp : p < c.ports.size) (hk : k < c.ports[p]!.chunks)
    (hck : HmNets.check c = .ok ()) (hsha : LaidRows (shaLaid ()) 8192 fun j => w (shaBase c g p k + j))
    {A B : List (BitVec 64)} (hA : A.length = 8) (hB : B.length = 16)
    (hcv : ∀ q < 512, w (cvAt c g p k + q) = wbit A q) (hmsg : ∀ q < 1024, w (msgAt c g p k + q) = wbit B q) :
    ∀ q < 512, w (outAt c g p k + q) = wbit (compressV A B) q := by
  have hs : shaSub c p k < 3 := (rowSlot_lt hp hk).2
  have hshaL : LaidRows (layout (compression3.run St.empty).2 SHA_IN SHA_OUT
      (compression3.run St.empty).1.2.flatten.toArray) (inRowsOf SHA_IN) fun j => w (shaBase c g p k + j) := by
    have := hsha
    rw [shaLaid_eq] at this
    rw [inRows_SHA_IN]
    exact this
  have hgs : Good (SHA_IN.foldl (· + ·) 0) (compression3.run St.empty).2 := by
    rw [sum_SHA_IN]; exact good_compression3
  have hrows : stHolds (fun x => w (shaBase c g p k + col (SHA_IN.foldl (· + ·) 0) (inRowsOf SHA_IN) x))
      (compression3.run St.empty).2 := laid_rows _ _ _ _ _ hgs hshaL
  have hzs : ∀ v : ℕ, v < 4608 →
      w (shaBase c g p k + col (SHA_IN.foldl (· + ·) 0) (inRowsOf SHA_IN) v) = w (shaBase c g p k + v) := by
    intro v hv
    rw [col_input (by rw [sum_SHA_IN]; exact hv)]
  have hsin := shaIn_eq hck _ hs
  intro q hq
  obtain ⟨-, hl3, -⟩ := compression3_spec _ hrows
  have hlens := compression3_lens _ hrows
  have hflat : (compression3.run St.empty).1.2.flatten.length = 1536 := by
    rw [List.length_flatten, List.map_congr_left (g := fun _ => 512) hlens, List.map_const', List.sum_replicate, hl3]
    rfl
  have hj : 512 * shaSub c p k + q < (compression3.run St.empty).1.2.flatten.toArray.size := by
    rw [List.size_toArray, hflat]; omega
  have hlo := laid_outs (compression3.run St.empty).2 SHA_IN SHA_OUT _ (fun j => w (shaBase c g p k + j))
    (by rw [List.size_toArray, hflat]; decide) hshaL _ hj
  rw [outAt, shaOut_eq hck hs, Nat.add_assoc, Nat.add_assoc, hlo, List.getElem_toArray,
    ← bit_of_lt (by rw [hflat]; omega), bit_flatten_of_len _ hlens hq]
  exact compression3_bits _ hrows hs hA hB
    (fun (q : ℕ) (hq : q < 512) => by
      rw [hzs _ (by omega), ← hcv q hq, cvAt_lt hp hk, hsin.1]
      exact congrArg w (by omega))
    (fun (q : ℕ) (hq : q < 1024) => by
      rw [hzs _ (by omega), ← hmsg q hq, msgAt_lt hp hk, hsin.2]
      exact congrArg w (by omega)) q hq

/-- **Row `p`'s `hm96` slot outputs the commit string of a message `r`** whose chaining value before its last block its
`cv` inputs hold and whose last block its `pad` inputs hold, under the salt its salt inputs hold, when the slot holds its
net's rows. -/
theorem hm_commitW (c : Circuit) (w : ℕ → Bool) (g p : ℕ) (hck : HmNets.check c = .ok ())
    (hhm : LaidRows (hmLaid ()) 4096 fun j => w (c.slotCol c.hmNet g p + j)) (r : List UInt8)
    (hcv : ∀ q < 512, w (c.slotCol c.hmNet g p + hmIn c 0 + q) = wbit (lastCv ⟨r.toArray⟩) q)
    (hpad : ∀ q < 1024, w (c.slotCol c.hmNet g p + hmIn c 1 + q) = wbit (lastBlock ⟨r.toArray⟩) q) :
    bytesOf w 128 (bcCol c g p) = Binding.commitString r (bytesOf w 192 (saltCol c g p)) := by
  have hhin := hmIn_eq hck
  have hzh := Hidden.hm_zh_in c w g p
  beta_reduce at hzh
  have hrow := hm96_commit _ (Hidden.hm_zh c w g p hhm) r (length_bytesOf w 192 (saltCol c g p))
    (fun (q : ℕ) (hq : q < 512) => by rw [hzh q (by omega), ← hcv q hq, hhin.1, Nat.add_zero])
    (fun (q : ℕ) (hq : q < 1024) => by
      rw [hzh _ (by omega), ← hpad q hq, hhin.2.1]
      exact congrArg w (by omega))
    (fun (q : ℕ) (hq : q < 1536) => by
      rw [hzh _ (by omega), bitL_bytesOf _ _ (by omega), saltCol, hhin.2.2]
      exact congrArg w (by omega))
  refine eq_of_bitL (by rw [length_bytesOf, length_commitString]) fun t ht => ?_
  rw [length_bytesOf] at ht
  rw [bitL_bytesOf _ _ ht, ← hrow.2 t ht, ← Hidden.hm_bc c w g p hck hhm t ht, bcCol]
  exact congrArg w (by omega)

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
  obtain ⟨-, hs, -, -, hc, hw⟩ := id h
  have hd : (bytesOf w (2 * c.ports[p]!.words) (Typed.msgCol c g p)).length = 2 * c.ports[p]!.words :=
    length_bytesOf _ _ _
  have hpre : (HmRow.prefixOf c.ports[p]!).data.toList.length = 128 := by
    rw [Array.length_toList, ByteArray.size_data, Hidden.size_prefixOf]
  rw [rowMsgOf_unseg hs]
  have hrl : ((HmRow.prefixOf c.ports[p]!).data.toList ++
      bytesOf w (2 * c.ports[p]!.words) (Typed.msgCol c g p)).length = 128 + 2 * c.ports[p]!.words := by
    rw [List.length_append, hpre, hd]
  have hlast : (padData ⟨((HmRow.prefixOf c.ports[p]!).data.toList ++
      bytesOf w (2 * c.ports[p]!.words) (Typed.msgCol c g p)).toArray⟩).size / 128 - 1 = 1 := by
    rw [size_padData_list, hrl]; omega
  have hm0 : msgAt c g p 0 = c.slotCol c.hmNet g p + hmIn c 1 := by rw [← msgAt_hm hp, hc]
  refine hm_commitW c w g p hck hhm _ (fun q hq => ?_) (fun q hq => ?_)
  · rw [hcvh q hq, lastCv_eq, hlast, wordsBit_mid hpre _ hq]
  · obtain ⟨ho, hb⟩ := blockBit_lt hq
    rw [lastBlock_eq, hlast, wbit_blockV _ _ hq, padData_list]
    by_cases hr : (HmRow.blockBit q).1 < 2 * c.ports[p]!.words
    · rw [bitL_append_left (by rw [hrl]; omega), bitL_append_right (by rw [hpre]; omega), hpre,
        show 8 * (128 * 1 + (HmRow.blockBit q).1) + (HmRow.blockBit q).2 - 8 * 128 =
          8 * (HmRow.blockBit q).1 + (HmRow.blockBit q).2 by omega, bitL_bytesOf _ _ (by omega),
        show (8 * (HmRow.blockBit q).1 + (HmRow.blockBit q).2) / 8 = (HmRow.blockBit q).1 by omega,
        show (8 * (HmRow.blockBit q).1 + (HmRow.blockBit q).2) % 8 = (HmRow.blockBit q).2 by omega,
        msgCol_eqW, byteAt_unseg hs, Nat.div_eq_of_lt ho, Nat.mod_eq_of_lt ho, hm0, blockBit_col hq]
    · rw [hpad q hq (by omega), tailBit_eq, tailOf_short h,
        bitL_append_right (by rw [hrl]; omega), hrl]
      congr 1
      omega

/-! ## A segmented row -/

namespace Chain

theorem segShape_size (l : ℕ) : (HmRow.segShape l).1.size = (l + 8191) / 8192 := by
  simp [HmRow.segShape, HmRow.ROW_SEG_BITS]

theorem segShape_get (l : ℕ) {j : ℕ} (hj : j < (HmRow.segShape l).1.size) :
    ((HmRow.segShape l).1[j]!).1 = min (l - 8192 * j) 8192 := by
  rw [getElem!_pos _ j hj]
  simp [HmRow.segShape, HmRow.ROW_SEG_BITS]

theorem segsOf_getD (p : Port) {j : ℕ} (hj : j < (segsOf p).size) : (segsOf p).getD j default = (segsOf p)[j]! := by
  rw [getElem!_pos _ j hj, Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hj, Option.getD_some]

section Seg

variable {p : Port} (h : RowSegWhole p)
include h

/-- **Segment `j`'s place in a whole-block segmented row**: its bits a positive multiple of 1024, at most 8192, within the
row's bytes from byte `1024 j`; its compressions its data blocks and a padding block. -/
theorem seg_geom {j : ℕ} (hj : j < (segsOf p).size) :
    0 < ((segsOf p)[j]!).1 ∧ ((segsOf p)[j]!).1 % 1024 = 0 ∧ ((segsOf p)[j]!).1 ≤ 8192 ∧
      1024 * j + ((segsOf p)[j]!).1 / 8 ≤ 2 * p.words ∧
      segFirst p (j + 1) = segFirst p j + ((segsOf p)[j]!).1 / 1024 + 1 := by
  obtain ⟨-, hs, hb, -, -, -⟩ := id h
  obtain ⟨h0, h1, h2⟩ := segs_whole h hj
  have hS : segsOf p = (HmRow.segShape p.bits).1 := by simp [segsOf, hs]
  have hsz : j < (HmRow.segShape p.bits).1.size := by rw [← hS]; exact hj
  have hm := segShape_get p.bits hsz
  rw [← hS] at hm
  rw [segShape_size] at hsz
  refine ⟨h0, h1, by omega, by omega, ?_⟩
  rw [segFirst_succ p hj, h2, Nat.add_assoc]

theorem segFirst_mono : ∀ (d : ℕ) {j : ℕ}, j + d ≤ (segsOf p).size → segFirst p j ≤ segFirst p (j + d)
  | 0, _, _ => le_rfl
  | d + 1, j, hd => by
    have := segFirst_mono d (j := j) (by omega)
    rw [← Nat.add_assoc, (seg_geom h (j := j + d) (by omega)).2.2.2.2]
    omega

theorem seg_last_lt {j : ℕ} (hj : j < (segsOf p).size) :
    segFirst p j + ((segsOf p)[j]!).1 / 1024 < p.chunks := by
  have h1 := (seg_geom h hj).2.2.2.2
  have h2 := segFirst_mono h ((segsOf p).size - (j + 1)) (j := j + 1) (by omega)
  rw [show j + 1 + ((segsOf p).size - (j + 1)) = (segsOf p).size by omega] at h2
  have h3 := seg_chunks h
  omega

end Seg

theorem bitL_segData (p : Port) (data : List UInt8) (j : ℕ) {x b : ℕ} (hx : x < ((segsOf p)[j]!).1 / 8)
    (hb : b < 8) : bitL (segData p data j) (8 * x + b) = bitL data (8 * (1024 * j + x) + b) := by
  rw [bitL, bitL, show (8 * x + b) / 8 = x by omega, show (8 * x + b) % 8 = b by omega,
    show (8 * (1024 * j + x) + b) / 8 = 1024 * j + x by omega, show (8 * (1024 * j + x) + b) % 8 = b by omega,
    segData, List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD, List.getElem?_take_of_lt hx,
    List.getElem?_drop]

/-- **Segment `j`'s chain outputs its digest**: its compressions' inputs holding `port_commit_segW`'s copies, the output
of its last compression (`segFirst p j + m / 1024`) is the digest of `segMsgOf`, in `wordBit` order. -/
theorem seg_chain (c : Circuit) (w : ℕ → Bool) (g p : ℕ) (hp : p < c.ports.size)
    (h : RowSegWhole c.ports[p]!) (hck : HmNets.check c = .ok ())
    (hsha : ∀ k < c.ports[p]!.chunks, LaidRows (shaLaid ()) 8192 fun j => w (shaBase c g p k + j))
    (hscv : ∀ j < (segsOf c.ports[p]!).size, ∀ t < 512,
      w (cvAt c g p (segFirst c.ports[p]! j) + t) = wordsBit (HmRow.segMidstate c.ports[p]! j) t)
    (hschain : ∀ j < (segsOf c.ports[p]!).size, ∀ i < ((segsOf c.ports[p]!)[j]!).1 / 1024, ∀ t < 512,
      w (cvAt c g p (segFirst c.ports[p]! j + i + 1) + t) = w (outAt c g p (segFirst c.ports[p]! j + i) + t))
    (hspad : ∀ j < (segsOf c.ports[p]!).size, ∀ t < 1024,
      w (msgAt c g p (segFirst c.ports[p]! j + ((segsOf c.ports[p]!)[j]!).1 / 1024) + t) =
        wordsBit (HmRow.wordsBE (HmRow.padBlock (128 + ((segsOf c.ports[p]!)[j]!).1 / 8)) 16) t)
    {j : ℕ} (hj : j < (segsOf c.ports[p]!).size) {t : ℕ} (ht : t < 512) :
    w (outAt c g p (segFirst c.ports[p]! j + ((segsOf c.ports[p]!)[j]!).1 / 1024) + wordBit t) =
      bitL (Binding.digBytes (Refine.H512
        (segMsgOf c.ports[p]! (bytesOf w (2 * c.ports[p]!.words) (Typed.msgCol c g p)) j))) t := by
  obtain ⟨hm0, hm1, hm2, hm3, -⟩ := seg_geom h hj
  obtain ⟨-, hs, -, -, -, -⟩ := id h
  have hlast := seg_last_lt h hj
  have hpre := RowBytes.segmentPrefix_length j (segsOf c.ports[p]!).size ((segsOf c.ports[p]!)[j]!).1
  have hd : (bytesOf w (2 * c.ports[p]!.words) (Typed.msgCol c g p)).length = 2 * c.ports[p]!.words :=
    length_bytesOf _ _ _
  have hsd : (segData c.ports[p]! (bytesOf w (2 * c.ports[p]!.words) (Typed.msgCol c g p)) j).length =
      ((segsOf c.ports[p]!)[j]!).1 / 8 := by
    rw [segData, List.length_take, List.length_drop, hd]; omega
  have hml : (segMsgOf c.ports[p]! (bytesOf w (2 * c.ports[p]!.words) (Typed.msgCol c g p)) j).length =
      128 + ((segsOf c.ports[p]!)[j]!).1 / 8 := by
    rw [segMsgOf, List.length_append, hpre, hsd]
  have hsize : (padData ⟨(segMsgOf c.ports[p]! (bytesOf w (2 * c.ports[p]!.words) (Typed.msgCol c g p))
      j).toArray⟩).size / 128 - 1 + 1 = ((segsOf c.ports[p]!)[j]!).1 / 1024 + 2 := by
    rw [size_padData_list, hml]; omega
  have hout := chain_bits w (fun i => cvAt c g p (segFirst c.ports[p]! j + i))
    (fun i => msgAt c g p (segFirst c.ports[p]! j + i)) (fun i => outAt c g p (segFirst c.ports[p]! j + i))
    (padData ⟨(segMsgOf c.ports[p]! (bytesOf w (2 * c.ports[p]!.words) (Typed.msgCol c g p)) j).toArray⟩)
    (((segsOf c.ports[p]!)[j]!).1 / 1024 + 1)
    (fun i hi A B hA hB hcv hmsg => sha_stepW c w g p _ hp (by omega) hck (hsha _ (by omega)) hA hB hcv hmsg)
    (fun q hq => by
      rw [Nat.add_zero, hscv j hj q hq, HmRow.segMidstate, segsOf_getD _ hj, wordsBit_mid hpre _ hq]
      rfl)
    (fun i hi q hq => by
      rw [← Nat.add_assoc]
      exact hschain j hj i (by omega) q hq)
    (fun i hi q hq => by
      obtain ⟨ho, hb⟩ := blockBit_lt hq
      rw [wbit_blockV _ _ hq, padData_list, hml]
      by_cases hi' : i < ((segsOf c.ports[p]!)[j]!).1 / 1024
      · rw [bitL_append_left (by rw [hml]; omega), segMsgOf, bitL_append_right (by rw [hpre]; omega), hpre,
          show 8 * (128 * (i + 1) + (HmRow.blockBit q).1) + (HmRow.blockBit q).2 - 8 * 128 =
            8 * (128 * i + (HmRow.blockBit q).1) + (HmRow.blockBit q).2 by omega,
          bitL_segData _ _ _ (by omega) hb, bitL_bytesOf _ _ (by omega),
          show (8 * (1024 * j + (128 * i + (HmRow.blockBit q).1)) + (HmRow.blockBit q).2) / 8 =
            1024 * j + (128 * i + (HmRow.blockBit q).1) by omega,
          show (8 * (1024 * j + (128 * i + (HmRow.blockBit q).1)) + (HmRow.blockBit q).2) % 8 =
            (HmRow.blockBit q).2 by omega,
          msgCol_eqW, byteAt_seg hs,
          show (1024 * j + (128 * i + (HmRow.blockBit q).1)) / 1024 = j by omega,
          show (1024 * j + (128 * i + (HmRow.blockBit q).1)) % 1024 / 128 = i by omega,
          show (1024 * j + (128 * i + (HmRow.blockBit q).1)) % 128 = (HmRow.blockBit q).1 by omega, blockBit_col hq]
      · have hie : i = ((segsOf c.ports[p]!)[j]!).1 / 1024 := by omega
        subst hie
        rw [hspad j hj q hq, wordsBit_wordsBE _ hq, bitL_append_right (by rw [hml]; omega), hml, padTail_data,
          padBlock_data, show (239 - (128 + ((segsOf c.ports[p]!)[j]!).1 / 8) % 128) % 128 = 111 by omega]
        congr 1
        omega)
    (((segsOf c.ports[p]!)[j]!).1 / 1024) (by omega) (wordBit t) (by unfold wordBit; omega)
  rw [hout, Binding.digBytes, Refine.dig_eq, Refine.H512_val, bitL_hash_chain _ ht, hsize]

/-- **The outer message's digest bytes**: byte `e < 64 s`, bit `b`, of the segments' digests is bit `wordBit (8 (e % 64) + b)`
of segment `e / 64`'s last compression's output. -/
theorem seg_digests (c : Circuit) (w : ℕ → Bool) (g p : ℕ) (hp : p < c.ports.size)
    (h : RowSegWhole c.ports[p]!) (hck : HmNets.check c = .ok ())
    (hsha : ∀ k < c.ports[p]!.chunks, LaidRows (shaLaid ()) 8192 fun j => w (shaBase c g p k + j))
    (hscv : ∀ j < (segsOf c.ports[p]!).size, ∀ t < 512,
      w (cvAt c g p (segFirst c.ports[p]! j) + t) = wordsBit (HmRow.segMidstate c.ports[p]! j) t)
    (hschain : ∀ j < (segsOf c.ports[p]!).size, ∀ i < ((segsOf c.ports[p]!)[j]!).1 / 1024, ∀ t < 512,
      w (cvAt c g p (segFirst c.ports[p]! j + i + 1) + t) = w (outAt c g p (segFirst c.ports[p]! j + i) + t))
    (hspad : ∀ j < (segsOf c.ports[p]!).size, ∀ t < 1024,
      w (msgAt c g p (segFirst c.ports[p]! j + ((segsOf c.ports[p]!)[j]!).1 / 1024) + t) =
        wordsBit (HmRow.wordsBE (HmRow.padBlock (128 + ((segsOf c.ports[p]!)[j]!).1 / 8)) 16) t)
    {e b : ℕ} (he : e < 64 * (segsOf c.ports[p]!).size) (hb : b < 8) :
    bitL ((List.range (segsOf c.ports[p]!).size).flatMap fun j => Binding.digBytes (Refine.H512
        (segMsgOf c.ports[p]! (bytesOf w (2 * c.ports[p]!.words) (Typed.msgCol c g p)) j))) (8 * e + b) =
      w (outAt c g p (segFirst c.ports[p]! (e / 64 + 1) - 1) + (64 * (e % 64 / 8) + 56 - 8 * (e % 64 % 8) + b)) := by
  have hj : e / 64 < (segsOf c.ports[p]!).size := by omega
  have hc := seg_chain c w g p hp h hck hsha hscv hschain hspad hj (t := 8 * (e % 64) + b) (by omega)
  rw [show wordBit (8 * (e % 64) + b) = 64 * (e % 64 / 8) + 56 - 8 * (e % 64 % 8) + b by unfold wordBit; omega] at hc
  rw [(seg_geom h hj).2.2.2.2, Nat.add_sub_cancel, hc, bitL, bitL,
    show (8 * e + b) / 8 = e by omega, show (8 * e + b) % 8 = b by omega,
    show (8 * (e % 64) + b) / 8 = e % 64 by omega, show (8 * (e % 64) + b) % 8 = b by omega,
    getD_flatMap_range (f := fun j => Binding.digBytes (Refine.H512
      (segMsgOf c.ports[p]! (bytesOf w (2 * c.ports[p]!.words) (Typed.msgCol c g p)) j))) (fun j => Refine.hL512 _) _ he]

end Chain

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
  obtain ⟨-, hs, -, -, -, -⟩ := id h
  have hn := seg_chunks h
  have hpre : (HmRow.prefixOf c.ports[p]!).data.toList.length = 128 := by
    rw [Array.length_toList, ByteArray.size_data, Hidden.size_prefixOf]
  have hr : rowMsgOf c.ports[p]! (bytesOf w (2 * c.ports[p]!.words) (Typed.msgCol c g p)) =
      (HmRow.prefixOf c.ports[p]!).data.toList ++ (List.range (segsOf c.ports[p]!).size).flatMap fun j =>
        Binding.digBytes (Refine.H512 (segMsgOf c.ports[p]! (bytesOf w (2 * c.ports[p]!.words) (Typed.msgCol c g p)) j)) := by
    rw [rowMsgOf, hs]
    rfl
  have hrl : (rowMsgOf c.ports[p]! (bytesOf w (2 * c.ports[p]!.words) (Typed.msgCol c g p))).length =
      128 + 64 * (segsOf c.ports[p]!).size := by
    rw [hr, List.length_append, hpre, RowBytes.length_flatMap_range fun j => Refine.hL512 _]
    rfl
  have hlast : (padData ⟨(rowMsgOf c.ports[p]! (bytesOf w (2 * c.ports[p]!.words)
      (Typed.msgCol c g p))).toArray⟩).size / 128 - 1 = (segsOf c.ports[p]!).size / 2 + 1 := by
    rw [size_padData_list, hrl]; omega
  have hbyte : ∀ e < 64 * (segsOf c.ports[p]!).size, ∀ b < 8,
      bitL (rowMsgOf c.ports[p]! (bytesOf w (2 * c.ports[p]!.words) (Typed.msgCol c g p))) (8 * (128 + e) + b) =
        w (msgAt c g p (segFirst c.ports[p]! (segsOf c.ports[p]!).size + e / 128) +
          (64 * (e % 128 / 8) + 56 - 8 * (e % 128 % 8) + b)) := by
    intro e he b hb
    rw [hr, bitL_append_right (by rw [hpre]; omega), hpre, show 8 * (128 + e) + b - 8 * 128 = 8 * e + b by omega,
      seg_digests c w g p hp h hck hsha hscv hschain hspad he hb, hdig e he b hb]
  have hout := chain_bits w (fun i => cvAt c g p (segFirst c.ports[p]! (segsOf c.ports[p]!).size + i))
    (fun i => msgAt c g p (segFirst c.ports[p]! (segsOf c.ports[p]!).size + i))
    (fun i => outAt c g p (segFirst c.ports[p]! (segsOf c.ports[p]!).size + i))
    (padData ⟨(rowMsgOf c.ports[p]! (bytesOf w (2 * c.ports[p]!.words) (Typed.msgCol c g p))).toArray⟩)
    ((segsOf c.ports[p]!).size / 2)
    (fun i hi A B hA hB hcv hmsg => sha_stepW c w g p _ hp (by omega) hck (hsha _ (by omega)) hA hB hcv hmsg)
    (fun q hq => by rw [Nat.add_zero, hocv q hq, hr, wordsBit_mid hpre _ hq])
    (fun i hi q hq => by
      rw [← Nat.add_assoc]
      exact hochain _ (by omega) (by omega) q hq)
    (fun i hi q hq => by
      obtain ⟨ho, hb⟩ := blockBit_lt hq
      rw [wbit_blockV _ _ hq, padData_list, bitL_append_left (by rw [hrl]; omega),
        show 8 * (128 * (i + 1) + (HmRow.blockBit q).1) + (HmRow.blockBit q).2 =
          8 * (128 + (128 * i + (HmRow.blockBit q).1)) + (HmRow.blockBit q).2 by ring,
        hbyte _ (by omega) _ hb, show (128 * i + (HmRow.blockBit q).1) / 128 = i by omega,
        show (128 * i + (HmRow.blockBit q).1) % 128 = (HmRow.blockBit q).1 by omega, blockBit_col hq])
  refine hm_commitW c w g p hck hhm _ (fun q hq => ?_) (fun q hq => ?_)
  · rw [lastCv_eq, hlast, ← cvAt_hm hp]
    rcases Nat.eq_zero_or_pos ((segsOf c.ports[p]!).size / 2) with h0 | h0
    · rw [show c.ports[p]!.chunks = segFirst c.ports[p]! (segsOf c.ports[p]!).size by omega, hocv q hq, h0,
        Nat.zero_add, hr, wordsBit_mid hpre _ hq]
    · rw [show c.ports[p]!.chunks = segFirst c.ports[p]! (segsOf c.ports[p]!).size +
          ((segsOf c.ports[p]!).size / 2 - 1) + 1 by omega,
        hochain (segFirst c.ports[p]! (segsOf c.ports[p]!).size + ((segsOf c.ports[p]!).size / 2 - 1))
          (by omega) (by omega) q hq,
        hout _ (by omega) q hq, show (segsOf c.ports[p]!).size / 2 - 1 + 2 = (segsOf c.ports[p]!).size / 2 + 1 by omega]
  · obtain ⟨ho, hb⟩ := blockBit_lt hq
    rw [lastBlock_eq, hlast, wbit_blockV _ _ hq, padData_list, ← msgAt_hm hp]
    by_cases hd : (HmRow.blockBit q).1 < 64 * (segsOf c.ports[p]!).size % 128
    · rw [bitL_append_left (by rw [hrl]; omega),
        show 8 * (128 * ((segsOf c.ports[p]!).size / 2 + 1) + (HmRow.blockBit q).1) + (HmRow.blockBit q).2 =
          8 * (128 + (128 * ((segsOf c.ports[p]!).size / 2) + (HmRow.blockBit q).1)) + (HmRow.blockBit q).2 by ring,
        hbyte _ (by omega) _ hb,
        show (128 * ((segsOf c.ports[p]!).size / 2) + (HmRow.blockBit q).1) / 128 = (segsOf c.ports[p]!).size / 2 by
          omega,
        show (128 * ((segsOf c.ports[p]!).size / 2) + (HmRow.blockBit q).1) % 128 = (HmRow.blockBit q).1 by omega,
        blockBit_col hq, show segFirst c.ports[p]! (segsOf c.ports[p]!).size + (segsOf c.ports[p]!).size / 2 =
          c.ports[p]!.chunks by omega]
    · rw [htail q hq (by omega), tailBit_eq, tailOf_seg h, bitL_append_right (by rw [hrl]; omega), hrl]
      congr 1
      omega

end FlockSoundness.Discharge.ScopeW
