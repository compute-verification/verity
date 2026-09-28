/-
The salt digest `c = SHA-512(salt_prefix ‖ y)` of `hm96-sha512/v1` as the hm96 circuit computes it: from the salt
prefix's midstate (`saltMid`), over the salt's 24 big-endian words and the fixed padding of a 320-byte message
(`saltTail`).
-/
import Mathlib.Data.List.GetD
import Proofs.Flock.Soundness.Discharge.Hm.Hash
import Proofs.Flock.Soundness.Discharge.Hm.Hm96

namespace FlockSoundness.Discharge.Hm

open Flock.HmNets hiding and

open Flock

theorem size_saltPrefix : Hm96.Default512.saltPrefix.size = 128 := by decide +kernel

theorem length_saltPrefix : Hm96.Default512.saltPrefix.data.toList.length = 128 := by
  rw [Array.length_toList, ByteArray.size_data, size_saltPrefix]

/-- The salt's 24 message words, big-endian. -/
def saltWordsV (s : List UInt8) : List (BitVec 64) := (List.range 24).map fun i => (wordL s 0 i).toBitVec

/-- The circuit's salt words are the salt's when its salt bits are the salt's bytes' bits. -/
theorem map_bv_saltWords (z : Var → Bool) (y : List Lin) (s : List UInt8) (hy : ∀ q < 1536, bit z y q = bitL s q) :
    (saltWords y).map (bv z) = saltWordsV s := by
  simp only [saltWords, saltWordsV, List.map_map]
  refine List.map_congr_left fun i hi => ?_
  simp only [List.mem_range] at hi
  refine bv_ext (by simp) fun j hj => ?_
  rw [getLsbD_wordL _ _ _ hj]
  have h1 : bit z ((List.range 64).map fun j => y.getD (64 * i + 8 * (7 - j / 8) + j % 8) Lin.zero) j =
      bit z y (64 * i + 8 * (7 - j / 8) + j % 8) := by
    simp [bit, hj, List.getD_eq_getElem?_getD]
  rw [h1, hy _ (by omega), bitL, show (64 * i + 8 * (7 - j / 8) + j % 8) / 8 = 0 + 8 * i + (7 - j / 8) by omega,
    show (64 * i + 8 * (7 - j / 8) + j % 8) % 8 = j % 8 by omega]

/-- The salt digest's message `salt_prefix ‖ y`. -/
def saltMsg (s : List UInt8) : ByteArray := ⟨(Hm96.Default512.saltPrefix.data.toList ++ s).toArray⟩

theorem saltMsg_data (s : List UInt8) :
    (saltMsg s).data.toList = Hm96.Default512.saltPrefix.data.toList ++ s := rfl

theorem size_saltMsg {s : List UInt8} (hs : s.length = 192) : (saltMsg s).size = 320 := by
  rw [← ByteArray.size_data, ← Array.length_toList, saltMsg_data, List.length_append, length_saltPrefix, hs]

/-- The padding of a 320-byte message: `0x80`, 47 zeros, the bit count 2560 in 16 bytes. -/
def saltPadT : List UInt8 := [128] ++ List.replicate 47 0 ++ lenBytes 320

theorem saltMsg_pad {s : List UInt8} (hs : s.length = 192) :
    (padData (saltMsg s)).data.toList = Hm96.Default512.saltPrefix.data.toList ++ s ++ saltPadT := by
  rw [padData_data, size_saltMsg hs, saltMsg_data, saltPadT]
  simp only [List.append_assoc]
  rfl

theorem size_padData_saltMsg {s : List UInt8} (hs : s.length = 192) : (padData (saltMsg s)).size = 384 := by
  rw [size_padData, size_saltMsg hs]
  rfl

theorem saltPad_words : (List.range 8).map (fun k => (wordL saltPadT 0 k).toBitVec) = saltTail := by
  decide +kernel

theorem getD_saltPad_lo {s : List UInt8} (hs : s.length = 192) {k : ℕ} (hk : k < 128) :
    (Hm96.Default512.saltPrefix.data.toList ++ s ++ saltPadT).getD k 0 =
      Hm96.Default512.saltPrefix.data.toList.getD k 0 := by
  rw [List.getD_append _ _ _ _ (by simp [size_saltPrefix, hs]; omega),
    List.getD_append _ _ _ _ (by rw [length_saltPrefix]; omega)]

theorem getD_saltPad_mid {s : List UInt8} (hs : s.length = 192) {k : ℕ} (hk1 : 128 ≤ k) (hk2 : k < 320) :
    (Hm96.Default512.saltPrefix.data.toList ++ s ++ saltPadT).getD k 0 = s.getD (k - 128) 0 := by
  rw [List.getD_append _ _ _ _ (by simp [size_saltPrefix, hs]; omega),
    List.getD_append_right _ _ _ _ (by rw [length_saltPrefix]; omega), length_saltPrefix]

theorem getD_saltPad_hi {s : List UInt8} (hs : s.length = 192) {k : ℕ} (hk : 320 ≤ k) :
    (Hm96.Default512.saltPrefix.data.toList ++ s ++ saltPadT).getD k 0 = saltPadT.getD (k - 320) 0 := by
  rw [List.getD_append_right _ _ _ _ (by simp [size_saltPrefix, hs]; omega)]
  simp [size_saltPrefix, hs]

/-- The chaining value before the salt digest's last block: the prefix's midstate, compressed with the salt's first 16
words. -/
theorem lastCv_saltMsg {s : List UInt8} (hs : s.length = 192) :
    lastCv (saltMsg s) = compressV saltMid ((saltWordsV s).take 16) := by
  have hpd := saltMsg_pad hs
  have h1 : Sha512.compress Sha512.iv (padData (saltMsg s)) 0 =
      Sha512.compress Sha512.iv Hm96.Default512.saltPrefix 0 := by
    refine compress_congr _ fun k hk => ?_
    rw [get!_eq_getD, get!_eq_getD, hpd, Nat.zero_add, getD_saltPad_lo hs hk]
  have h2 : (List.range 16).map (fun i => (wordBE (padData (saltMsg s)) 128 i).toBitVec) =
      (saltWordsV s).take 16 := by
    rw [saltWordsV, ← List.map_take, List.take_range, show min 16 24 = 16 from rfl]
    refine List.map_congr_left fun i hi => ?_
    simp only [List.mem_range] at hi
    rw [wordBE_eq_wordL, wordL_congr (l' := s) (off' := 0) (i' := i) fun j hj => ?_]
    rw [hpd, getD_saltPad_mid hs (by omega) (by omega), show 128 + 8 * i + j - 128 = 0 + 8 * i + j by omega]
  have hc : chain (padData (saltMsg s)) 2 =
      Sha512.compress (Sha512.compress Sha512.iv Hm96.Default512.saltPrefix 0) (padData (saltMsg s)) 128 := by
    rw [chain_succ, chain_succ]
    exact congrArg (fun h => Sha512.compress h _ _) h1
  unfold lastCv
  rw [size_padData_saltMsg hs]
  rw [show 384 / 128 - 1 = 2 from rfl]
  rw [hc]
  rw [map_range_eq_toBV _ (size_compress _ _ _)]
  rw [compress_eq]
  rw [h2]
  rfl

/-- The salt digest's last block: the salt's last 8 words, then the padding's. -/
theorem lastBlock_saltMsg {s : List UInt8} (hs : s.length = 192) :
    lastBlock (saltMsg s) = (saltWordsV s).drop 16 ++ saltTail := by
  have hpd := saltMsg_pad hs
  unfold lastBlock
  rw [size_padData_saltMsg hs]
  rw [show 128 * (384 / 128 - 1) = 256 from rfl]
  apply List.ext_getElem
  · simp [saltWordsV, saltTail]
  · intro i h1 h2
    simp only [List.length_map, List.length_range] at h1
    rw [List.getElem_map, List.getElem_range, wordBE_eq_wordL]
    by_cases hi : i < 8
    · rw [List.getElem_append_left (by simp [saltWordsV]; omega), List.getElem_drop]
      simp only [saltWordsV, List.getElem_map, List.getElem_range]
      rw [wordL_congr (l' := s) (off' := 0) (i' := 16 + i) fun j hj => ?_]
      rw [hpd, getD_saltPad_mid hs (by omega) (by omega), show 256 + 8 * i + j - 128 = 0 + 8 * (16 + i) + j by omega]
    · rw [List.getElem_append_right (by simp [saltWordsV]; omega)]
      have e := congrArg (fun l => l[i - 8]?) saltPad_words
      simp only [List.getElem?_map, List.getElem?_range (show i - 8 < 8 by omega), Option.map_some] at e
      have hl : ((saltWordsV s).drop 16).length = 8 := by simp [saltWordsV]
      rw [eq_comm, List.getElem_eq_iff, hl, ← e]
      rw [wordL_congr (l := (padData (saltMsg s)).data.toList) (off := 256) (i := i) (l' := saltPadT) (off' := 0)
        (i' := i - 8) fun j hj => ?_]
      rw [hpd, getD_saltPad_hi hs (by omega), show 256 + 8 * i + j - 320 = 0 + 8 * (i - 8) + j by omega]

/-- **The salt digest's bits**: bit `t < 512` of `SHA-512(salt_prefix ‖ y)` (byte-string order) is bit `wordBit t` of
the prefix's midstate compressed with the salt's 24 words and the padding. -/
theorem bitL_saltHash {s : List UInt8} (hs : s.length = 192) {t : ℕ} (ht : t < 512) :
    bitL (Sha512.hash (saltMsg s)).data.toList t =
      wbit (compressV (compressV saltMid ((saltWordsV s).take 16)) ((saltWordsV s).drop 16 ++ saltTail))
        (wordBit t) := by
  rw [bitL_hash _ ht, lastCv_saltMsg hs, lastBlock_saltMsg hs]

end FlockSoundness.Discharge.Hm
