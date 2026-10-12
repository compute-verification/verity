import Proofs.Flock.Soundness.Discharge.Hm.Nets

/-!
# A row's commit string under any 128-byte prefix

`Hm.row_commit` and `Hm.port_commit_rows` read `HmRow.rowPrefix words` only through its 128 bytes and its midstate.
These are theirs for any 128-byte prefix `pre` (`row_commitP`, `port_commit_rowsP`), so for `HmRow.prefixOf` of a
`sha512/row/v1` or `sha512/row/v2` row (`size_prefixOf`). The row's bytes fill its `n` compressions' blocks and may end in
its `hm96` slot's block, where SHA-512's padding follows them (`rowMsgP_pad`, `128 n ≤ |data|`,
`|data| + 17 ≤ 128 (n + 1)`): that block's bits are the row bits `Typed.msgCol` reads there and the constants
`HmRow.constBit` gives elsewhere (`pad_bitsP`). A row of no compression (`port_commit_rows0`) is hashed by its `hm96` slot
alone, its `cv` the prefix's midstate.
-/

set_option Elab.async false

namespace FlockSoundness.Discharge.Hidden

open Flock.HmNets hiding and

open Flock Hm

/-! ## The prefix -/

theorem size_pad {b : ByteArray} {n : ℕ} (h : b.size ≤ n) : (Hm96.pad b n).size = n := by
  rw [Hm96.pad, ByteArray.size_append]
  simp only [ByteArray.size, Array.size_replicate]
  show b.size + (n - min n b.size) = n
  omega

theorem size_rowPrefixV2 (l : ℕ) : (HmRow.rowPrefixV2 l).size = 128 := by
  have h0 : ("verity/sha512-row/v2".toUTF8.push 0).size = 21 := by decide +kernel
  have h1 : (ByteArray.mk ((Array.range 8).map fun i => (l / 256 ^ (7 - i) % 256).toUInt8)).size = 8 := by
    simp [ByteArray.size]
  refine size_pad ?_
  rw [ByteArray.size_append, h0, h1]
  omega

theorem size_rowPrefixSeg (l : ℕ) : (HmRow.rowPrefixSeg l).size = 128 := by
  have h0 : ("verity/sha512-row-seg/v1".toUTF8.push 0).size = 25 := by decide +kernel
  have h8 : ∀ x, (HmRow.be64 x).size = 8 := fun x => by simp [HmRow.be64, ByteArray.size]
  have h4 : (be32 1024).size = 4 := by decide +kernel
  refine size_pad ?_
  rw [ByteArray.size_append, ByteArray.size_append, ByteArray.size_append, h0, h8, h8, h4]
  omega

theorem size_prefixOf (p : Port) : (HmRow.prefixOf p).size = 128 := by
  unfold HmRow.prefixOf
  split
  · exact size_rowPrefixSeg _
  · split
    · exact size_rowPrefixV2 _
    · exact size_rowPrefix _

/-! ## The row's bytes after a prefix -/

/-- The row's bytes after the 128-byte prefix `pre`. -/
def rowMsgP (pre : ByteArray) (data : List UInt8) : ByteArray := ⟨(pre.data.toList ++ data).toArray⟩

/-- SHA-512's padding after `total` bytes: `0x80`, `zeroPad total` zeros, the bit length. -/
theorem padTail_data (total : ℕ) :
    (HmRow.padTail total).data.toList = [128] ++ List.replicate (zeroPad total) 0 ++ lenBytes total := by
  have hz : (239 - total % 128) % 128 = zeroPad total := by unfold zeroPad; omega
  simp only [HmRow.padTail, Std.Legacy.Range.forIn_eq_forIn_range', List.forIn_pure_yield_eq_foldl, pure_bind,
    Id.run_pure, Std.Legacy.Range.size, Nat.reduceSub, Nat.reduceAdd, Nat.sub_zero, Nat.div_one, hz]
  rw [foldl_push_data (fun i => ((8 * total) >>> (8 * (15 - i)) % 256).toUInt8), lenBytes]
  simp only [Nat.mul_comm 8 total]
  rfl

/-- A row of `16 · words` bits has `2 · words` bytes, `sha512/row/v1` or `sha512/row/v2`. -/
theorem bytes_of_bits {p : Port} (hb : p.bits = 16 * p.words) : p.bytes = 2 * p.words := by
  unfold Port.bytes
  split <;> omega

section Pre

variable {pre : ByteArray} (hpre : pre.size = 128)
include hpre

theorem length_pre : pre.data.toList.length = 128 := by rw [Array.length_toList, ByteArray.size_data, hpre]

/-- The padded bytes of `pre ‖ data`: then SHA-512's padding after them (`HmRow.padTail`). -/
theorem rowMsgP_pad {data : List UInt8} :
    (padData (rowMsgP pre data)).data.toList =
      pre.data.toList ++ data ++ (HmRow.padTail (128 + data.length)).data.toList := by
  have hs : (rowMsgP pre data).size = 128 + data.length := by
    rw [← ByteArray.size_data, ← Array.length_toList]
    show (pre.data.toList ++ data).length = _
    rw [List.length_append, length_pre hpre]
  rw [padData_data, hs, padTail_data]
  show (pre.data.toList ++ data) ++ _ ++ _ ++ _ = _
  simp only [List.append_assoc]

/-- `pre ‖ data` pads to `n + 2` blocks when its data fill `n` blocks and the padding fits in one more
(`128 n ≤ |data|`, `|data| + 17 ≤ 128 (n + 1)`). -/
theorem size_padData_rowMsgP {n : ℕ} {data : List UInt8} (h1 : 128 * n ≤ data.length)
    (h2 : data.length + 17 ≤ 128 * (n + 1)) : (padData (rowMsgP pre data)).size = 128 * (n + 2) := by
  have hs : (rowMsgP pre data).size = 128 + data.length := by
    rw [← ByteArray.size_data, ← Array.length_toList]
    show (pre.data.toList ++ data).length = _
    rw [List.length_append, length_pre hpre]
  rw [size_padData, hs]
  unfold zeroPad
  omega

theorem getD_rowPadP_pre {data tl : List UInt8} {i : ℕ} (hi : i < 128) :
    (pre.data.toList ++ data ++ tl).getD i 0 = pre.data.toList.getD i 0 := by
  rw [List.getD_append _ _ _ _ (by simp [length_pre hpre]; omega),
    List.getD_append _ _ _ _ (by rw [length_pre hpre]; omega)]

theorem getD_rowPadP_data {data tl : List UInt8} {i : ℕ} (hi1 : 128 ≤ i) (hi2 : i < 128 + data.length) :
    (pre.data.toList ++ data ++ tl).getD i 0 = data.getD (i - 128) 0 := by
  rw [List.getD_append _ _ _ _ (by simp [length_pre hpre]; omega),
    List.getD_append_right _ _ _ _ (by rw [length_pre hpre]; omega), length_pre hpre]

theorem getD_rowPadP_tail {data tl : List UInt8} {i : ℕ} (hi : 128 + data.length ≤ i) :
    (pre.data.toList ++ data ++ tl).getD i 0 = tl.getD (i - (128 + data.length)) 0 := by
  rw [List.getD_append_right _ _ _ _ (by simp [length_pre hpre]; omega)]
  simp [length_pre hpre]

/-- After the prefix block, the chain is the prefix's midstate. -/
theorem chainV_one_rowP {data : List UInt8} :
    chainV (padData (rowMsgP pre data)) 1 =
      (List.range 8).map fun i => (Sha512.compress Sha512.iv pre 0)[i]!.toBitVec := by
  have h1 : Sha512.compress Sha512.iv (padData (rowMsgP pre data)) 0 = Sha512.compress Sha512.iv pre 0 := by
    refine compress_congr _ fun k hk => ?_
    rw [get!_eq_getD, get!_eq_getD, rowMsgP_pad hpre, Nat.zero_add, getD_rowPadP_pre hpre hk]
  rw [chainV, chain_succ, show chain (padData (rowMsgP pre data)) 0 = Sha512.iv from rfl, Nat.mul_zero, h1]

/-- Block `k + 1` (`k < n`, `128 n ≤ |data|`) is chunk `k` of the data. -/
theorem blockV_rowP {n : ℕ} {data : List UInt8} (h1 : 128 * n ≤ data.length) {k : ℕ} (hk : k < n) :
    blockV (padData (rowMsgP pre data)) (k + 1) = (List.range 16).map fun i => (wordL data (128 * k) i).toBitVec := by
  rw [blockV]
  refine List.map_congr_left fun i hi => ?_
  simp only [List.mem_range] at hi
  rw [wordBE_eq_wordL, wordL_congr (l' := data) (off' := 128 * k) (i' := i) fun j hj => ?_]
  rw [rowMsgP_pad hpre, getD_rowPadP_data hpre (by omega) (by nlinarith),
    show 128 * (k + 1) + 8 * i + j - 128 = 128 * k + 8 * i + j by omega]

theorem lastIdx_rowMsgP {n : ℕ} {data : List UInt8} (h1 : 128 * n ≤ data.length)
    (h2 : data.length + 17 ≤ 128 * (n + 1)) : (padData (rowMsgP pre data)).size / 128 - 1 = n + 1 := by
  rw [size_padData_rowMsgP hpre h1 h2, Nat.mul_div_cancel_left _ (by norm_num)]
  rfl

end Pre

/-- Message bit `q` of block `K` is bit `q % 8` of the block's byte `8 (q / 64) + 7 - q % 64 / 8` (its words
big-endian). -/
theorem wbit_blockV (P : ByteArray) (K : ℕ) {q : ℕ} (hq : q < 1024) :
    wbit (blockV P K) q = bitL P.data.toList (8 * (128 * K + (8 * (q / 64) + (7 - q % 64 / 8))) + q % 8) := by
  rw [blockV, wbit, List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range (show q / 64 < 16 by omega),
    Option.map_some, Option.getD_some, wordBE_eq_wordL, getLsbD_wordL _ _ _ (Nat.mod_lt _ (by norm_num)), bitL,
    show (8 * (128 * K + (8 * (q / 64) + (7 - q % 64 / 8))) + q % 8) / 8 = 128 * K + 8 * (q / 64) + (7 - q % 64 / 8)
      by omega,
    show (8 * (128 * K + (8 * (q / 64) + (7 - q % 64 / 8))) + q % 8) % 8 = q % 64 % 8 by omega]

section Pre

variable {pre : ByteArray} (hpre : pre.size = 128)
include hpre

/-- **`Hm.row_commit` under the prefix `pre`**: a row whose data fill `n > 0` blocks, the padding in one more, whose
slots hold Δ's copies, the first `cv` the prefix's midstate and the `hm96` slot's `pad` the last block, outputs the commit
string of `pre ‖ data`. -/
theorem row_commitP {n : ℕ} (hn : 0 < n) {data : List UInt8} (h1 : 128 * n ≤ data.length)
    (h2 : data.length + 17 ≤ 128 * (n + 1)) (zs : ℕ → Var → Bool) (sub : ℕ → ℕ) (hsub : ∀ k < n, sub k < 3)
    (hrows : ∀ k < n, stHolds (zs k) (compression3.run St.empty).2)
    (hcv0 : ∀ q : ℕ, q < 512 → zs 0 (1536 * sub 0 + q) = wordsBit (Sha512.compress Sha512.iv pre 0) q)
    (hchain : ∀ k, k + 1 < n → ∀ q : ℕ, q < 512 →
      zs (k + 1) (1536 * sub (k + 1) + q) = Hm.bit (zs k) ((compression3.run St.empty).1.2.getD (sub k) []) q)
    (hmsg : ∀ k < n, ∀ o < 128, ∀ t < 8,
      zs k (1536 * sub k + 512 + wordBit (8 * o + t)) = bitL data (8 * (128 * k + o) + t))
    (zh : Var → Bool) (hh : stHolds zh ((hm96Row Hm96.Default512.key).run St.empty).2)
    (hcvh : ∀ q : ℕ, q < 512 → zh q = Hm.bit (zs (n - 1)) ((compression3.run St.empty).1.2.getD (sub (n - 1)) []) q)
    (hpadh : ∀ q : ℕ, q < 1024 → zh (512 + q) = wbit (blockV (padData (rowMsgP pre data)) (n + 1)) q)
    {s : List UInt8} (hs : s.length = 192) (hy : ∀ q : ℕ, q < 1536 → zh (1536 + q) = bitL s q) :
    ∀ t < 1024, Hm.bit zh ((hm96Row Hm96.Default512.key).run St.empty).1.2.2.2 t =
      bitL (Binding.commitString (pre.data.toList ++ data) s) t := by
  set P := padData (rowMsgP pre data) with hP
  have hmsgW : ∀ k < n, ∀ q : ℕ, q < 1024 → zs k (1536 * sub k + 512 + q) = wbit (blockV P (k + 1)) q := by
    intro k hk q hq
    have ho : 8 * (q / 64) + (7 - q % 64 / 8) < 128 := by omega
    have hq' : wordBit (8 * (8 * (q / 64) + (7 - q % 64 / 8)) + q % 8) = q := by unfold wordBit; omega
    have e := hmsg k hk (8 * (q / 64) + (7 - q % 64 / 8)) ho (q % 8) (Nat.mod_lt _ (by norm_num))
    rw [hq'] at e
    rw [e, blockV_rowP hpre h1 hk, wbit, List.getD_eq_getElem?_getD, List.getElem?_map,
      List.getElem?_range (show q / 64 < 16 by omega), Option.map_some, Option.getD_some,
      getLsbD_wordL _ _ _ (Nat.mod_lt _ (by norm_num)), bitL,
      show (8 * (128 * k + (8 * (q / 64) + (7 - q % 64 / 8))) + q % 8) / 8 = 128 * k + 8 * (q / 64) + (7 - q % 64 / 8)
        by omega, show (8 * (128 * k + (8 * (q / 64) + (7 - q % 64 / 8))) + q % 8) % 8 = q % 64 % 8 by omega]
  have hlen8 : ∀ k, (chainV P k).length = 8 := fun k => by rw [chainV, List.length_map, List.length_range]
  have hlen16 : ∀ k, (blockV P k).length = 16 := fun k => by rw [blockV, List.length_map, List.length_range]
  have hout : ∀ k < n, ∀ q < 512,
      Hm.bit (zs k) ((compression3.run St.empty).1.2.getD (sub k) []) q = wbit (chainV P (k + 2)) q := by
    intro k
    induction k with
    | zero =>
      intro hk q hq
      refine (compression3_bits (zs 0) (hrows 0 hk) (hsub 0 hk) (hlen8 1) (hlen16 1) (fun q hq => ?_)
        (hmsgW 0 hk) q hq).trans (by rw [show 0 + 2 = 1 + 1 from rfl, chainV_succ P 1])
      rw [hcv0 q hq, wordsBit_eq _ (show q < 64 * 8 by omega), chainV_one_rowP hpre]
    | succ k ih =>
      intro hk q hq
      refine (compression3_bits (zs (k + 1)) (hrows _ hk) (hsub _ hk) (hlen8 (k + 2)) (hlen16 (k + 2))
        (fun q hq => ?_) (hmsgW _ hk) q hq).trans (by rw [chainV_succ P (k + 1 + 1)])
      rw [hchain k hk q hq, ih (by omega) q hq]
  have hlast := lastIdx_rowMsgP hpre h1 h2
  have hcv : ∀ q < 512, zh q = wbit (lastCv (rowMsgP pre data)) q := by
    intro q hq
    rw [hcvh q hq, hout (n - 1) (by omega) q hq, lastCv_eq, hlast, show n - 1 + 2 = n + 1 by omega]
  have hpad : ∀ q < 1024, zh (512 + q) = wbit (lastBlock (rowMsgP pre data)) q := by
    intro q hq
    rw [hpadh q hq, lastBlock_eq, hlast]
  exact (hm96_commit zh hh _ hs hcv hpad hy).2

/-- **A row of no compression**: its `hm96` slot, its `cv` the prefix's midstate and its `pad` the one block after the
prefix (the data and their padding, `|data| + 17 ≤ 128`), outputs the commit string of `pre ‖ data`. -/
theorem row_commit0P {data : List UInt8} (h2 : data.length + 17 ≤ 128) (zh : Var → Bool)
    (hh : stHolds zh ((hm96Row Hm96.Default512.key).run St.empty).2)
    (hcvh : ∀ q : ℕ, q < 512 → zh q = wordsBit (Sha512.compress Sha512.iv pre 0) q)
    (hpadh : ∀ q : ℕ, q < 1024 → zh (512 + q) = wbit (blockV (padData (rowMsgP pre data)) 1) q)
    {s : List UInt8} (hs : s.length = 192) (hy : ∀ q : ℕ, q < 1536 → zh (1536 + q) = bitL s q) :
    ∀ t < 1024, Hm.bit zh ((hm96Row Hm96.Default512.key).run St.empty).1.2.2.2 t =
      bitL (Binding.commitString (pre.data.toList ++ data) s) t := by
  have hlast := lastIdx_rowMsgP hpre (n := 0) (Nat.zero_le _) h2
  have hcv : ∀ q < 512, zh q = wbit (lastCv (rowMsgP pre data)) q := by
    intro q hq
    rw [hcvh q hq, wordsBit_eq _ (show q < 64 * 8 by omega), lastCv_eq, hlast, chainV_one_rowP hpre]
  have hpad : ∀ q < 1024, zh (512 + q) = wbit (lastBlock (rowMsgP pre data)) q := by
    intro q hq
    rw [hpadh q hq, lastBlock_eq, hlast]
  exact (hm96_commit zh hh _ hs hcv hpad hy).2

end Pre

/-! ## The port -/

/-- Row byte `128 n + o` past the row's `n` compressions is in its `hm96` slot's `pad`. -/
theorem msgCol_hm (c : Circuit) (g p : ℕ) {n o t : ℕ} (hs : c.ports[p]!.segs = none)
    (hn : ((HmRow.rowSlots c.ports).2[p]!).size = n) (ho : o < 128) :
    Typed.msgCol c g p (128 * n + o) t = c.slotCol c.hmNet g p + hmIn c 1 + (64 * (o / 8) + 56 - 8 * (o % 8) + t) := by
  have hk' : (128 * n + o) / 128 = n := by omega
  have ho' : (128 * n + o) % 128 = o := by omega
  have hg : ∀ xs : Array (ℕ × ℕ), xs[1]! = xs.getD 1 default := fun xs => by
    by_cases hi : 1 < xs.size
    · rw [getElem!_pos xs 1 hi, Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hi]; rfl
    · rw [getElem!_neg xs 1 hi, Array.getD_eq_getD_getElem?, Array.getElem?_eq_none (by omega)]; rfl
  simp only [Typed.msgCol, Typed.byteAt_none hs, hk', ho', hn, lt_irrefl, ↓reduceIte, hmIn, hg]
  omega

/-- **A row's `pad` bits are its last block's**, for a row of `16 · words` bits whose data fill its `n` compressions'
blocks and end, with their padding, in its `hm96` slot's block: at a row byte the bit `Typed.msgCol` reads there, at
every other bit the constant `HmRow.constBit` gives, which the block holds (`hconst`). -/
theorem pad_bitsP (c : Circuit) (w : ℕ → Bool) (g p : ℕ) {pre : ByteArray} (hpre : pre.size = 128) {n : ℕ}
    (hs : c.ports[p]!.segs = none) (hb : c.ports[p]!.bits = 16 * c.ports[p]!.words)
    (hn : ((HmRow.rowSlots c.ports).2[p]!).size = n) (h1 : 128 * n ≤ 2 * c.ports[p]!.words)
    (h2 : 2 * c.ports[p]!.words + 17 ≤ 128 * (n + 1))
    (hconst : ∀ t < 1024, ∀ v, HmRow.constBit c.ports[p]! (HmRow.tailOf c.ports[p]!) n t = some v →
      w (c.slotCol c.hmNet g p + hmIn c 1 + t) = v) :
    ∀ q < 1024, w (c.slotCol c.hmNet g p + hmIn c 1 + q) =
      wbit (blockV (padData (rowMsgP pre (bytesOf w (2 * c.ports[p]!.words) (Typed.msgCol c g p)))) (n + 1)) q := by
  intro q hq
  have hlen : (bytesOf w (2 * c.ports[p]!.words) (Typed.msgCol c g p)).length = 2 * c.ports[p]!.words :=
    length_bytesOf _ _ _
  have ho : 8 * (q / 64) + (7 - q % 64 / 8) < 128 := by omega
  rw [wbit_blockV _ _ hq, bitL, rowMsgP_pad hpre,
    show (8 * (128 * (n + 1) + (8 * (q / 64) + (7 - q % 64 / 8))) + q % 8) / 8 =
      128 * (n + 1) + (8 * (q / 64) + (7 - q % 64 / 8)) by omega,
    show (8 * (128 * (n + 1) + (8 * (q / 64) + (7 - q % 64 / 8))) + q % 8) % 8 = q % 8 by omega]
  by_cases hj : 128 * n + (8 * (q / 64) + (7 - q % 64 / 8)) < 2 * c.ports[p]!.words
  · rw [getD_rowPadP_data hpre (by omega) (by rw [hlen]; omega),
      show 128 * (n + 1) + (8 * (q / 64) + (7 - q % 64 / 8)) - 128 = 128 * n + (8 * (q / 64) + (7 - q % 64 / 8))
        by omega]
    have e := bitL_bytesOf w (N := 2 * c.ports[p]!.words) (Typed.msgCol c g p) (q := 8 * (128 * n + (8 * (q / 64) + (7 - q % 64 / 8))) + q % 8)
      (by omega)
    unfold bitL at e
    rw [show (8 * (128 * n + (8 * (q / 64) + (7 - q % 64 / 8))) + q % 8) / 8 = 128 * n + (8 * (q / 64) + (7 - q % 64 / 8))
        by omega, show (8 * (128 * n + (8 * (q / 64) + (7 - q % 64 / 8))) + q % 8) % 8 = q % 8 by omega] at e
    rw [e, msgCol_hm c g p hs hn ho]
    congr 1
    omega
  · have hcb : HmRow.constBit c.ports[p]! (HmRow.tailOf c.ports[p]!) n q =
        some (decide (2 * c.ports[p]!.words ≤ 128 * n + (8 * (q / 64) + (7 - q % 64 / 8))) &&
          (((HmRow.tailOf c.ports[p]!).get! (128 * n + (8 * (q / 64) + (7 - q % 64 / 8)) - 2 * c.ports[p]!.words)).toNat
            >>> (q % 8)) % 2 == 1) := by
      have hj1 : (HmRow.blockBit q).1 = 8 * (q / 64) + (7 - q % 64 / 8) := by unfold HmRow.blockBit; omega
      have hj2 : (HmRow.blockBit q).2 = q % 8 := rfl
      unfold HmRow.constBit
      simp only [bytes_of_bits hb, hj1, hj2]
      rw [ite_eq_right_iff.2 (fun h => absurd h (by omega))]
    rw [hconst q hq _ hcb, getD_rowPadP_tail hpre (by rw [hlen]; omega), hlen, decide_eq_true (by omega),
      Bool.true_and, get!_eq_getD, show 128 * (n + 1) + (8 * (q / 64) + (7 - q % 64 / 8)) - (128 + 2 * c.ports[p]!.words) =
        128 * n + (8 * (q / 64) + (7 - q % 64 / 8)) - 2 * c.ports[p]!.words by omega]
    simp only [HmRow.tailOf, hs, bytes_of_bits hb]
    generalize ((HmRow.padTail (128 + 2 * c.ports[p]!.words)).data.toList.getD
      (128 * n + (8 * (q / 64) + (7 - q % 64 / 8)) - 2 * c.ports[p]!.words) 0).toNat = x
    rw [Nat.testBit, Nat.one_and_eq_mod_two]
    rcases Nat.mod_two_eq_zero_or_one (x >>> (q % 8)) with h | h <;> simp [h]

/-- **`Hm.port_commit` under the prefix `pre`**, for a row of `16 · words` bits whose data fill its `n > 0`
compressions' blocks and end, with their padding, in its `hm96` slot's block. -/
theorem port_commitP (c : Circuit) (w : ℕ → Bool) (g p : ℕ) {pre : ByteArray} (hpre : pre.size = 128) {n : ℕ}
    (hs : c.ports[p]!.segs = none) (hb : c.ports[p]!.bits = 16 * c.ports[p]!.words)
    (hslots : ((HmRow.rowSlots c.ports).2[p]!).size = n) (hn : 0 < n) (h1 : 128 * n ≤ 2 * c.ports[p]!.words)
    (h2 : 2 * c.ports[p]!.words + 17 ≤ 128 * (n + 1)) (hsub : ∀ k < n, shaSub c p k < 3)
    (hsin : ∀ s < 3, shaIn c (2 * s) = 1536 * s ∧ shaIn c (2 * s + 1) = 1536 * s + 512)
    (hhin : hmIn c 0 = 0 ∧ hmIn c 1 = 512 ∧ hmIn c 2 = 1536)
    (zs : ℕ → Var → Bool) (hzs : ∀ k < n, ∀ v : ℕ, v < 4608 → zs k v = w (shaBase c g p k + v))
    (hrows : ∀ k < n, stHolds (zs k) (compression3.run St.empty).2)
    (hout : ∀ k < n, ∀ t < 512, w (shaBase c g p k + shaOut c (shaSub c p k) + t) =
      Hm.bit (zs k) ((compression3.run St.empty).1.2.getD (shaSub c p k) []) t)
    (zh : Var → Bool) (hzh : ∀ v : ℕ, v < 3072 → zh v = w (c.slotCol c.hmNet g p + v))
    (hh : stHolds zh ((hm96Row Hm96.Default512.key).run St.empty).2)
    (hbc : ∀ t < 1024, w (c.slotCol c.hmNet g p + hmOut c + t) = Hm.bit zh ((hm96Row Hm96.Default512.key).run St.empty).1.2.2.2 t)
    (hcv0 : ∀ t < 512, w (shaBase c g p 0 + shaIn c (2 * shaSub c p 0) + t) =
      wordsBit (Sha512.compress Sha512.iv pre 0) t)
    (hchain : ∀ k, k + 1 < n → ∀ t < 512, w (shaBase c g p (k + 1) + shaIn c (2 * shaSub c p (k + 1)) + t) =
      w (shaBase c g p k + shaOut c (shaSub c p k) + t))
    (hcvh : ∀ t < 512, w (c.slotCol c.hmNet g p + hmIn c 0 + t) =
      w (shaBase c g p (n - 1) + shaOut c (shaSub c p (n - 1)) + t))
    (hconst : ∀ t < 1024, ∀ v, HmRow.constBit c.ports[p]! (HmRow.tailOf c.ports[p]!) n t = some v →
      w (c.slotCol c.hmNet g p + hmIn c 1 + t) = v) :
    bytesOf w 128 (bcCol c g p) = Binding.commitString
      (pre.data.toList ++ bytesOf w (2 * c.ports[p]!.words) (Typed.msgCol c g p)) (bytesOf w 192 (saltCol c g p)) := by
  have hd : (bytesOf w (2 * c.ports[p]!.words) (Typed.msgCol c g p)).length = 2 * c.ports[p]!.words :=
    length_bytesOf _ _ _
  have hpb := pad_bitsP c w g p hpre hs hb hslots h1 h2 hconst
  have hrow := row_commitP hpre hn (by rw [hd]; exact h1) (by rw [hd]; exact h2) zs (shaSub c p) hsub hrows
    (fun q hq => by
      rw [hzs 0 hn _ (by have := hsub 0 hn; omega), ← hcv0 q hq, (hsin _ (hsub 0 hn)).1]
      exact congrArg w (by omega))
    (fun k hk q hq => by
      rw [hzs (k + 1) hk _ (by have := hsub (k + 1) hk; omega), ← hout k (by omega) q hq, ← hchain k hk q hq,
        (hsin _ (hsub (k + 1) hk)).1]
      exact congrArg w (by omega))
    (fun k hk o ho t ht => by
      have hwb : wordBit (8 * o + t) < 1024 := by unfold wordBit; omega
      rw [hzs k hk _ (by have := hsub k hk; omega), bitL_bytesOf _ _ (by omega),
        show (8 * (128 * k + o) + t) / 8 = 128 * k + o by omega, show (8 * (128 * k + o) + t) % 8 = t by omega,
        msgCol_eq c g p k hs (by omega) ho ht, (hsin _ (hsub k hk)).2]
      exact congrArg w (by omega))
    zh hh
    (fun q hq => by
      rw [hzh q (by omega), ← hout (n - 1) (by omega) q hq, ← hcvh q hq, hhin.1, Nat.add_zero])
    (fun q hq => by
      rw [hzh _ (by omega), ← hpb q hq, hhin.2.1, Nat.add_assoc])
    (length_bytesOf w 192 (saltCol c g p))
    (fun q hq => by
      rw [hzh _ (by omega), bitL_bytesOf _ _ (by omega), saltCol, hhin.2.2]
      exact congrArg w (by omega))
  refine eq_of_bitL (by rw [length_bytesOf, length_commitString]) fun t ht => ?_
  rw [length_bytesOf] at ht
  rw [bitL_bytesOf _ _ ht, ← hrow t ht, ← hbc t ht, bcCol]
  exact congrArg w (by omega)

section Nets

variable (c : Circuit) (w : ℕ → Bool) (g p : ℕ)

theorem hmLaidL (hhm : LaidRows (hmLaid ()) 4096 fun j => w (c.slotCol c.hmNet g p + j)) :
    LaidRows (layout ((hm96Row Hm96.Default512.key).run St.empty).2 HM_IN HM_OUT
      ((hm96Row Hm96.Default512.key).run St.empty).1.2.2.2.toArray) (inRowsOf HM_IN)
      fun j => w (c.slotCol c.hmNet g p + j) := by
  have := hhm
  rw [hmLaid_eq] at this
  rw [inRows_HM_IN]
  exact this

theorem hm_zh (hhm : LaidRows (hmLaid ()) 4096 fun j => w (c.slotCol c.hmNet g p + j)) :
    stHolds (fun x => w (c.slotCol c.hmNet g p + col (HM_IN.foldl (· + ·) 0) (inRowsOf HM_IN) x))
      ((hm96Row Hm96.Default512.key).run St.empty).2 := by
  have hgh : Good (HM_IN.foldl (· + ·) 0) ((hm96Row Hm96.Default512.key).run St.empty).2 := by
    rw [sum_HM_IN]; exact good_hm96Row _
  exact laid_rows _ _ _ _ _ hgh (hmLaidL c w g p hhm)

theorem hm_bc (hck : HmNets.check c = .ok ())
    (hhm : LaidRows (hmLaid ()) 4096 fun j => w (c.slotCol c.hmNet g p + j)) :
    ∀ t < 1024, w (c.slotCol c.hmNet g p + hmOut c + t) =
      Hm.bit (fun x => w (c.slotCol c.hmNet g p + col (HM_IN.foldl (· + ·) 0) (inRowsOf HM_IN) x))
        ((hm96Row Hm96.Default512.key).run St.empty).1.2.2.2 t := by
  intro t ht
  obtain ⟨-, -, -, hl, -⟩ := hm96Row_spec _ _ (hm_zh c w g p hhm)
  have hj : t < ((hm96Row Hm96.Default512.key).run St.empty).1.2.2.2.toArray.size := by
    rw [List.size_toArray, hl]; exact ht
  have := laid_outs ((hm96Row Hm96.Default512.key).run St.empty).2 HM_IN HM_OUT _ (fun j => w (c.slotCol c.hmNet g p + j))
    (by rw [List.size_toArray, hl]; decide) (hmLaidL c w g p hhm) _ hj
  rw [hmOut_eq hck, Nat.add_assoc, this, List.getElem_toArray, ← bit_of_lt (by rw [hl]; exact ht)]

theorem hm_zh_in : ∀ v : ℕ, v < 3072 →
    (fun x => w (c.slotCol c.hmNet g p + col (HM_IN.foldl (· + ·) 0) (inRowsOf HM_IN) x)) v =
      w (c.slotCol c.hmNet g p + v) := by
  intro v hv
  show w (c.slotCol c.hmNet g p + col _ _ v) = _
  rw [col_input (by rw [sum_HM_IN]; exact hv)]

end Nets

/-- **`Hm.port_commit_rows` under the prefix `pre`**: the slots hold their nets' rows and the block holds Δ's copies,
the first `cv` the prefix's midstate and the `pad` the constant bits of the row's last block. -/
theorem port_commit_rowsP (c : Circuit) (w : ℕ → Bool) (g p : ℕ) {pre : ByteArray} (hpre : pre.size = 128)
    {n : ℕ} (hs : c.ports[p]!.segs = none) (hb : c.ports[p]!.bits = 16 * c.ports[p]!.words)
    (hslots : ((HmRow.rowSlots c.ports).2[p]!).size = n) (hn : 0 < n) (h1 : 128 * n ≤ 2 * c.ports[p]!.words)
    (h2 : 2 * c.ports[p]!.words + 17 ≤ 128 * (n + 1)) (hck : HmNets.check c = .ok ())
    (hsub : ∀ k < n, shaSub c p k < 3)
    (hsha : ∀ k < n, LaidRows (shaLaid ()) 8192 fun j => w (shaBase c g p k + j))
    (hhm : LaidRows (hmLaid ()) 4096 fun j => w (c.slotCol c.hmNet g p + j))
    (hcv0 : ∀ t < 512, w (shaBase c g p 0 + shaIn c (2 * shaSub c p 0) + t) =
      wordsBit (Sha512.compress Sha512.iv pre 0) t)
    (hchain : ∀ k, k + 1 < n → ∀ t < 512, w (shaBase c g p (k + 1) + shaIn c (2 * shaSub c p (k + 1)) + t) =
      w (shaBase c g p k + shaOut c (shaSub c p k) + t))
    (hcvh : ∀ t < 512, w (c.slotCol c.hmNet g p + hmIn c 0 + t) =
      w (shaBase c g p (n - 1) + shaOut c (shaSub c p (n - 1)) + t))
    (hconst : ∀ t < 1024, ∀ v, HmRow.constBit c.ports[p]! (HmRow.tailOf c.ports[p]!) n t = some v →
      w (c.slotCol c.hmNet g p + hmIn c 1 + t) = v) :
    bytesOf w 128 (bcCol c g p) = Binding.commitString
      (pre.data.toList ++ bytesOf w (2 * c.ports[p]!.words) (Typed.msgCol c g p)) (bytesOf w 192 (saltCol c g p)) := by
  have hshaL : ∀ k < n, LaidRows (layout (compression3.run St.empty).2 SHA_IN SHA_OUT (compression3.run St.empty).1.2.flatten.toArray)
      (inRowsOf SHA_IN) fun j => w (shaBase c g p k + j) := by
    intro k hk
    have := hsha k hk
    rw [shaLaid_eq] at this
    rw [inRows_SHA_IN]
    exact this
  have hgs : Good (SHA_IN.foldl (· + ·) 0) (compression3.run St.empty).2 := by rw [sum_SHA_IN]; exact good_compression3
  let zs : ℕ → Var → Bool := fun k x => w (shaBase c g p k + col (SHA_IN.foldl (· + ·) 0) (inRowsOf SHA_IN) x)
  have hrows : ∀ k < n, stHolds (zs k) (compression3.run St.empty).2 := fun k hk => laid_rows _ _ _ _ _ hgs (hshaL k hk)
  have hzs : ∀ k < n, ∀ v : ℕ, v < 4608 → zs k v = w (shaBase c g p k + v) := by
    intro k _ v hv
    show w (shaBase c g p k + col _ _ v) = _
    rw [col_input (by rw [sum_SHA_IN]; exact hv)]
  have hout : ∀ k < n, ∀ t < 512, w (shaBase c g p k + shaOut c (shaSub c p k) + t) =
      Hm.bit (zs k) ((compression3.run St.empty).1.2.getD (shaSub c p k) []) t := by
    intro k hk t ht
    have hs := hsub k hk
    obtain ⟨-, hl3, -⟩ := compression3_spec (zs k) (hrows k hk)
    have hlens := compression3_lens (zs k) (hrows k hk)
    have hflat : (compression3.run St.empty).1.2.flatten.length = 1536 := by
      rw [List.length_flatten, List.map_congr_left (g := fun _ => 512) hlens, List.map_const', List.sum_replicate, hl3]
      rfl
    have hj : 512 * shaSub c p k + t < (compression3.run St.empty).1.2.flatten.toArray.size := by
      rw [List.size_toArray, hflat]; omega
    have := laid_outs (compression3.run St.empty).2 SHA_IN SHA_OUT _ (fun j => w (shaBase c g p k + j))
      (by rw [List.size_toArray, hflat]; decide) (hshaL k hk) _ hj
    rw [shaOut_eq hck hs, Nat.add_assoc, Nat.add_assoc, this, List.getElem_toArray,
      ← bit_of_lt (by rw [hflat]; omega), bit_flatten_of_len (zs k) hlens ht]
  exact port_commitP c w g p hpre hs hb hslots hn h1 h2 hsub (shaIn_eq hck) (hmIn_eq hck) zs hzs hrows hout _
    (hm_zh_in c w g p) (hm_zh c w g p hhm) (hm_bc c w g p hck hhm) hcv0 hchain hcvh hconst

/-- **A row of no compression at its columns**: its `hm96` slot holds its net's rows, its `cv` the prefix's midstate
and its `pad` the constant bits of its one block (its bytes and their padding, `2 · words + 17 ≤ 128`), so the bytes at
`bcCol` are the commit string of the prefix and the row's bytes. -/
theorem port_commit_rows0 (c : Circuit) (w : ℕ → Bool) (g p : ℕ) {pre : ByteArray} (hpre : pre.size = 128)
    (hs : c.ports[p]!.segs = none) (hb : c.ports[p]!.bits = 16 * c.ports[p]!.words)
    (hslots : ((HmRow.rowSlots c.ports).2[p]!).size = 0) (h2 : 2 * c.ports[p]!.words + 17 ≤ 128)
    (hck : HmNets.check c = .ok ()) (hhm : LaidRows (hmLaid ()) 4096 fun j => w (c.slotCol c.hmNet g p + j))
    (hcvh : ∀ t < 512, w (c.slotCol c.hmNet g p + hmIn c 0 + t) = wordsBit (Sha512.compress Sha512.iv pre 0) t)
    (hconst : ∀ t < 1024, ∀ v, HmRow.constBit c.ports[p]! (HmRow.tailOf c.ports[p]!) 0 t = some v →
      w (c.slotCol c.hmNet g p + hmIn c 1 + t) = v) :
    bytesOf w 128 (bcCol c g p) = Binding.commitString
      (pre.data.toList ++ bytesOf w (2 * c.ports[p]!.words) (Typed.msgCol c g p)) (bytesOf w 192 (saltCol c g p)) := by
  have hhin := hmIn_eq hck
  have hzh := hm_zh_in c w g p
  beta_reduce at hzh
  have hd : (bytesOf w (2 * c.ports[p]!.words) (Typed.msgCol c g p)).length = 2 * c.ports[p]!.words :=
    length_bytesOf _ _ _
  have hpb := pad_bitsP c w g p hpre hs hb hslots (by omega) (by omega) hconst
  have hrow := row_commit0P hpre (by rw [hd]; exact h2) _ (hm_zh c w g p hhm)
    (fun q hq => by rw [hzh q (by omega), ← hcvh q hq, hhin.1, Nat.add_zero])
    (fun q hq => by
      rw [hzh _ (by omega), ← hpb q hq, hhin.2.1, Nat.add_assoc])
    (length_bytesOf w 192 (saltCol c g p))
    (fun q hq => by
      rw [hzh _ (by omega), bitL_bytesOf _ _ (by omega), saltCol, hhin.2.2]
      exact congrArg w (by omega))
  refine eq_of_bitL (by rw [length_bytesOf, length_commitString]) fun t ht => ?_
  rw [length_bytesOf] at ht
  rw [bitL_bytesOf _ _ ht, ← hrow t ht, ← hm_bc c w g p hck hhm t ht, bcCol]
  exact congrArg w (by omega)

end FlockSoundness.Discharge.Hidden
