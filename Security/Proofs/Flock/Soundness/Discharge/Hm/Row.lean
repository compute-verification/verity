/-
A row's hashing across its slots: the row's bytes `rowPrefix words ‖ data` (`data` in 128-byte chunks), its in-circuit
compressions in `sha512x3` slots (the first from the prefix's midstate, each next one from the previous one's output, each
over one chunk of `data`), and its `hm96` slot (`cv` from the last compression, `pad` the padding block, the salt). When
every slot's rows hold and the slots' inputs carry what Δ copies into them (`HmRow.delta`: the midstate and the padding
block from the constants, the data bytes at `msgCol`, `HmRow.rowWires`' chain), the hm96 slot outputs the row's commit
string (`row_commit`).
-/
import Flock.HmRow
import Proofs.Flock.Soundness.Discharge.Hm.Commit

namespace FlockSoundness.Discharge.Hm

open Flock.HmNets hiding and

open Flock

/-! ## The sha512x3 slot on given inputs -/

/-- **One compression of the sha512x3 slot**: when compression `k`'s `cv` inputs hold `A`'s bits and its `msg` inputs `B`'s,
its output's bits are `compressV A B`'s. -/
theorem compression3_bits (z : Var → Bool) (h : stHolds z (compression3.run St.empty).2) {k : ℕ} (hk : k < 3)
    {A B : List (BitVec 64)} (hA : A.length = 8) (hB : B.length = 16)
    (hcv : ∀ q < 512, z (1536 * k + q) = wbit A q) (hmsg : ∀ q < 1024, z (1536 * k + 512 + q) = wbit B q) :
    ∀ q < 512, bit z ((compression3.run St.empty).1.2.getD k []) q = wbit (compressV A B) q := by
  intro q hq
  have l1 : (vars (1536 * k) 512).length = 64 * 8 := length_vars _ _
  have l2 : (vars (1536 * k + 512) 1024).length = 64 * 16 := length_vars _ _
  have e1 : (words (vars (1536 * k) 512) 8).map (bv z) = A := by
    refine map_bv_of_bits z (words64_words l1) (by rw [length_words, hA]) fun q hq => ?_
    rw [length_words] at hq
    rw [bit_words_flatten z l1 hq, bit_vars z _ 512 (show q < 512 by omega), hcv q (by omega)]
  have e2 : (words (vars (1536 * k + 512) 1024) 16).map (bv z) = B := by
    refine map_bv_of_bits z (words64_words l2) (by rw [length_words, hB]) fun q hq => ?_
    rw [length_words] at hq
    rw [bit_words_flatten z l2 hq, bit_vars z _ 1024 (show q < 1024 by omega), hmsg q (by omega)]
  rw [(compression3_spec z h).2.2 k hk q hq, e1, e2]

/-! ## The row's bytes -/

theorem size_be32 (x : UInt32) : (be32 x).size = 4 := by
  simp only [be32, Std.Legacy.Range.forIn_eq_forIn_range', List.forIn_pure_yield_eq_foldl, pure_bind, Id.run_pure,
    Std.Legacy.Range.size, Nat.reduceSub, Nat.reduceAdd, Nat.sub_zero, Nat.div_one]
  simp [List.range', ByteArray.size_push]
  rfl

theorem size_rowPrefix (words : ℕ) : (HmRow.rowPrefix words).size = 128 := by
  have h0 : ("verity/sha512-row/v1".toUTF8.push 0).size = 21 := by decide +kernel
  have h1 : (ByteArray.mk #[1, 16]).size = 2 := rfl
  have h2 : ("verity/sha512-row/v1".toUTF8.push 0 ++ ByteArray.mk #[1, 16] ++ be32 words.toUInt32).size = 27 := by
    rw [ByteArray.size_append, ByteArray.size_append, h0, h1, size_be32]
  rw [HmRow.rowPrefix, Hm96.pad, ByteArray.size_append, h2]
  rfl

theorem length_rowPrefix (words : ℕ) : (HmRow.rowPrefix words).data.toList.length = 128 := by
  rw [Array.length_toList, ByteArray.size_data, size_rowPrefix]

/-- The row's bytes as `sha512/row/v1` hashes them. -/
def rowMsg (words : ℕ) (data : List UInt8) : ByteArray := ⟨((HmRow.rowPrefix words).data.toList ++ data).toArray⟩

/-- The padding of a message of `128 (n + 1)` bytes. -/
def rowPadT (n : ℕ) : List UInt8 := [128] ++ List.replicate 111 0 ++ lenBytes (128 + 128 * n)

theorem rowMsg_pad {words n : ℕ} {data : List UInt8} (hd : data.length = 128 * n) :
    (padData (rowMsg words data)).data.toList = (HmRow.rowPrefix words).data.toList ++ data ++ rowPadT n := by
  have hs : (rowMsg words data).size = 128 + 128 * n := by
    rw [← ByteArray.size_data, ← Array.length_toList]
    show ((HmRow.rowPrefix words).data.toList ++ data).length = _
    rw [List.length_append, length_rowPrefix, hd]
  rw [padData_data, hs, rowPadT, show zeroPad (128 + 128 * n) = 111 by unfold zeroPad; omega]
  show ((HmRow.rowPrefix words).data.toList ++ data) ++ _ ++ _ ++ _ = _
  simp only [List.append_assoc]

theorem size_padData_rowMsg {words n : ℕ} {data : List UInt8} (hd : data.length = 128 * n) :
    (padData (rowMsg words data)).size = 128 * (n + 2) := by
  rw [← ByteArray.size_data, ← Array.length_toList, rowMsg_pad hd]
  simp only [List.length_append, length_rowPrefix, hd, rowPadT, lenBytes, List.length_cons, List.length_nil,
    List.length_replicate, List.length_map, List.length_range']
  ring

/-! ## The chain over the row's blocks -/

theorem getD_rowPad_pre {words n : ℕ} {data : List UInt8} (hd : data.length = 128 * n) {i : ℕ} (hi : i < 128) :
    ((HmRow.rowPrefix words).data.toList ++ data ++ rowPadT n).getD i 0 = (HmRow.rowPrefix words).data.toList.getD i 0 := by
  rw [List.getD_append _ _ _ _ (by simp [hd, size_rowPrefix]; omega),
    List.getD_append _ _ _ _ (by rw [length_rowPrefix]; omega)]

theorem getD_rowPad_data {words n : ℕ} {data : List UInt8} (hd : data.length = 128 * n) {i : ℕ} (hi1 : 128 ≤ i)
    (hi2 : i < 128 + 128 * n) :
    ((HmRow.rowPrefix words).data.toList ++ data ++ rowPadT n).getD i 0 = data.getD (i - 128) 0 := by
  rw [List.getD_append _ _ _ _ (by simp [hd, size_rowPrefix]; omega),
    List.getD_append_right _ _ _ _ (by rw [length_rowPrefix]; omega), length_rowPrefix]

theorem getD_rowPad_tail {words n : ℕ} {data : List UInt8} (hd : data.length = 128 * n) {i : ℕ}
    (hi : 128 + 128 * n ≤ i) :
    ((HmRow.rowPrefix words).data.toList ++ data ++ rowPadT n).getD i 0 = (rowPadT n).getD (i - (128 + 128 * n)) 0 := by
  rw [List.getD_append_right _ _ _ _ (by simp [hd, size_rowPrefix]; omega)]
  simp [hd, size_rowPrefix]

/-- After the prefix block, the chain is the prefix's midstate (`HmRow.midstate`). -/
theorem chainV_one_row {words n : ℕ} {data : List UInt8} (hd : data.length = 128 * n) :
    chainV (padData (rowMsg words data)) 1 = (List.range 8).map fun i => (HmRow.midstate words)[i]!.toBitVec := by
  have h1 : Sha512.compress Sha512.iv (padData (rowMsg words data)) 0 =
      Sha512.compress Sha512.iv (HmRow.rowPrefix words) 0 := by
    refine compress_congr _ fun k hk => ?_
    rw [get!_eq_getD, get!_eq_getD, rowMsg_pad hd, Nat.zero_add, getD_rowPad_pre hd hk]
  rw [chainV, chain_succ, show chain (padData (rowMsg words data)) 0 = Sha512.iv from rfl, Nat.mul_zero, h1]
  rfl

/-- Block `k + 1` (`k < n`) is chunk `k` of the data. -/
theorem blockV_row {words n : ℕ} {data : List UInt8} (hd : data.length = 128 * n) {k : ℕ} (hk : k < n) :
    blockV (padData (rowMsg words data)) (k + 1) = (List.range 16).map fun i => (wordL data (128 * k) i).toBitVec := by
  rw [blockV]
  refine List.map_congr_left fun i hi => ?_
  simp only [List.mem_range] at hi
  rw [wordBE_eq_wordL, wordL_congr (l' := data) (off' := 128 * k) (i' := i) fun j hj => ?_]
  rw [rowMsg_pad hd, getD_rowPad_data hd (by omega) (by nlinarith),
    show 128 * (k + 1) + 8 * i + j - 128 = 128 * k + 8 * i + j by omega]

theorem wordsBE_getElem! {b : ByteArray} {i : ℕ} (hi : i < 16) :
    (HmRow.wordsBE b 16)[i]! = wordL b.data.toList 0 i := by
  rw [HmRow.wordsBE, getElem!_pos _ i (by simpa using hi), Array.getElem_map, Array.getElem_range, wordL]
  refine List.foldl_ext _ _ _ fun x j _ => ?_
  rw [get!_eq_getD, Nat.zero_add]

theorem padBlock_data (total : ℕ) :
    (HmRow.padBlock total).data.toList = [128] ++ List.replicate 111 0 ++ lenBytes total := by
  simp only [HmRow.padBlock, Std.Legacy.Range.forIn_eq_forIn_range', List.forIn_pure_yield_eq_foldl, pure_bind,
    Id.run_pure, Std.Legacy.Range.size, Nat.reduceSub, Nat.reduceAdd, Nat.sub_zero, Nat.div_one]
  rw [foldl_push_data (fun i => ((8 * total) >>> (8 * (15 - i)) % 256).toUInt8), lenBytes]
  simp only [Nat.mul_comm 8 total]
  rfl

/-- The last block (`n + 1`) is the padding block's words (`HmRow.padWords`) when the row has `2 · words = 128 n` data
bytes. -/
theorem blockV_row_last {words n : ℕ} (hw : 2 * words = 128 * n) {data : List UInt8} (hd : data.length = 128 * n) :
    blockV (padData (rowMsg words data)) (n + 1) = (List.range 16).map fun i => (HmRow.padWords words)[i]!.toBitVec := by
  rw [blockV]
  refine List.map_congr_left fun i hi => ?_
  simp only [List.mem_range] at hi
  rw [HmRow.padWords, wordsBE_getElem! hi, wordBE_eq_wordL, padBlock_data, hw,
    wordL_congr (l' := [128] ++ List.replicate 111 0 ++ lenBytes (128 + 128 * n)) (off' := 0) (i' := i) fun j hj => ?_]
  rw [rowMsg_pad hd, getD_rowPad_tail hd (by nlinarith), rowPadT,
    show 128 * (n + 1) + 8 * i + j - (128 + 128 * n) = 0 + 8 * i + j by ring_nf; omega]

/-! ## Δ's bits -/

/-- Bit `q` of a word array as Δ reads a constant: bit `q % 64` of word `q / 64` (`HmRow.delta`). -/
def wordsBit (w : Array UInt64) (q : ℕ) : Bool := (w[q / 64]! >>> (q % 64).toUInt64) &&& 1 == 1

theorem wordsBit_eq (w : Array UInt64) {n q : ℕ} (hq : q < 64 * n) :
    wordsBit w q = wbit ((List.range n).map fun i => w[i]!.toBitVec) q := by
  rw [wordsBit, wbit, List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range (by omega), Option.map_some,
    Option.getD_some]
  have h1 : (q % 64).toUInt64.toNat % 64 = q % 64 := by
    rw [Nat.toUInt64_eq, UInt64.toNat_ofNat', Nat.mod_eq_of_lt (by omega), Nat.mod_eq_of_lt (by omega)]
  have : ∀ x : UInt64, (x &&& 1 == 1) = x.toNat.testBit 0 := by
    intro x
    have hx : (x &&& 1).toNat = x.toNat % 2 := by rw [UInt64.toNat_and]; exact Nat.and_one_is_mod _
    rw [Nat.testBit_zero]
    by_cases h : x.toNat % 2 = 1
    · have : x &&& 1 = 1 := UInt64.toNat_inj.mp (by rw [hx, h]; rfl)
      simp [this, h]
    · have : x &&& 1 ≠ 1 := fun e => h (by rw [← hx, e]; rfl)
      simp [this, h]
  rw [this, UInt64.toNat_shiftRight, Nat.testBit_shiftRight, h1, Nat.add_zero]
  rfl

/-! ## The row -/

/-- **A row's slots compute its commit string.** Let a row of `words` u16 words have `2 · words = 128 n` data bytes
`data` (`n > 0` chunks), its compression `k < n` in compression `sub k` of a `sha512x3` slot whose assignment is `zs k`,
and its `hm96` slot's assignment `zh`. Suppose every slot's rows hold, and the slots' inputs hold what Δ copies into
them: the first compression's `cv` the prefix's midstate, compression `k`'s `msg` the data's chunk `k` (data byte
`128 k + o`'s bit `t` at input bit `wordBit (8 o + t)`, `msgCol`), each next compression's `cv` the previous output, the
hm96 slot's `cv` the last output, its `pad` the padding block, and its salt inputs the bits of a 192-byte salt `s`. Then
output bit `t < 1024` of the hm96 slot is bit `t` of `commitString (rowPrefix words ‖ data) s`. -/
theorem row_commit {words n : ℕ} (hn : 0 < n) (hw : 2 * words = 128 * n) {data : List UInt8}
    (hd : data.length = 128 * n) (zs : ℕ → Var → Bool) (sub : ℕ → ℕ) (hsub : ∀ k < n, sub k < 3)
    (hrows : ∀ k < n, stHolds (zs k) (compression3.run St.empty).2)
    (hcv0 : ∀ q : ℕ, q < 512 → zs 0 (1536 * sub 0 + q) = wordsBit (HmRow.midstate words) q)
    (hchain : ∀ k, k + 1 < n → ∀ q : ℕ, q < 512 →
      zs (k + 1) (1536 * sub (k + 1) + q) = bit (zs k) ((compression3.run St.empty).1.2.getD (sub k) []) q)
    (hmsg : ∀ k < n, ∀ o < 128, ∀ t < 8,
      zs k (1536 * sub k + 512 + wordBit (8 * o + t)) = bitL data (8 * (128 * k + o) + t))
    (zh : Var → Bool) (hh : stHolds zh ((hm96Row Hm96.Default512.key).run St.empty).2)
    (hcvh : ∀ q : ℕ, q < 512 → zh q = bit (zs (n - 1)) ((compression3.run St.empty).1.2.getD (sub (n - 1)) []) q)
    (hpadh : ∀ q : ℕ, q < 1024 → zh (512 + q) = wordsBit (HmRow.padWords words) q)
    {s : List UInt8} (hs : s.length = 192) (hy : ∀ q : ℕ, q < 1536 → zh (1536 + q) = bitL s q) :
    ∀ t < 1024, bit zh ((hm96Row Hm96.Default512.key).run St.empty).1.2.2.2 t =
      bitL (Binding.commitString ((HmRow.rowPrefix words).data.toList ++ data) s) t := by
  set P := padData (rowMsg words data) with hP
  have hmsgW : ∀ k < n, ∀ q : ℕ, q < 1024 → zs k (1536 * sub k + 512 + q) = wbit (blockV P (k + 1)) q := by
    intro k hk q hq
    have ho : 8 * (q / 64) + (7 - q % 64 / 8) < 128 := by omega
    have hq' : wordBit (8 * (8 * (q / 64) + (7 - q % 64 / 8)) + q % 8) = q := by unfold wordBit; omega
    have e := hmsg k hk (8 * (q / 64) + (7 - q % 64 / 8)) ho (q % 8) (Nat.mod_lt _ (by norm_num))
    rw [hq'] at e
    rw [e, blockV_row hd hk, wbit, List.getD_eq_getElem?_getD, List.getElem?_map,
      List.getElem?_range (show q / 64 < 16 by omega), Option.map_some, Option.getD_some,
      getLsbD_wordL _ _ _ (Nat.mod_lt _ (by norm_num)), bitL,
      show (8 * (128 * k + (8 * (q / 64) + (7 - q % 64 / 8))) + q % 8) / 8 = 128 * k + 8 * (q / 64) + (7 - q % 64 / 8)
        by omega, show (8 * (128 * k + (8 * (q / 64) + (7 - q % 64 / 8))) + q % 8) % 8 = q % 64 % 8 by omega]
  have hlen8 : ∀ k, (chainV P k).length = 8 := fun k => by rw [chainV, List.length_map, List.length_range]
  have hlen16 : ∀ k, (blockV P k).length = 16 := fun k => by rw [blockV, List.length_map, List.length_range]
  have hout : ∀ k < n, ∀ q < 512,
      bit (zs k) ((compression3.run St.empty).1.2.getD (sub k) []) q = wbit (chainV P (k + 2)) q := by
    intro k
    induction k with
    | zero =>
      intro hk q hq
      refine (compression3_bits (zs 0) (hrows 0 hk) (hsub 0 hk) (hlen8 1) (hlen16 1) (fun q hq => ?_)
        (hmsgW 0 hk) q hq).trans (by rw [show 0 + 2 = 1 + 1 from rfl, chainV_succ P 1])
      rw [hcv0 q hq, wordsBit_eq _ (show q < 64 * 8 by omega), chainV_one_row hd]
    | succ k ih =>
      intro hk q hq
      refine (compression3_bits (zs (k + 1)) (hrows _ hk) (hsub _ hk) (hlen8 (k + 2)) (hlen16 (k + 2))
        (fun q hq => ?_) (hmsgW _ hk) q hq).trans (by rw [chainV_succ P (k + 1 + 1)])
      rw [hchain k hk q hq, ih (by omega) q hq]
  have hsize := size_padData_rowMsg (words := words) hd
  have hlast : (padData (rowMsg words data)).size / 128 - 1 = n + 1 := by
    rw [hsize, Nat.mul_div_cancel_left _ (by norm_num)]
    rfl
  have hcv : ∀ q < 512, zh q = wbit (lastCv (rowMsg words data)) q := by
    intro q hq
    rw [hcvh q hq, hout (n - 1) (by omega) q hq, lastCv_eq, hlast, show n - 1 + 2 = n + 1 by omega]
  have hpad : ∀ q < 1024, zh (512 + q) = wbit (lastBlock (rowMsg words data)) q := by
    intro q hq
    rw [hpadh q hq, wordsBit_eq _ (show q < 64 * 16 by omega), lastBlock_eq, hlast, blockV_row_last hw hd]
  exact (hm96_commit zh hh _ hs hcv hpad hy).2

end FlockSoundness.Discharge.Hm
