import Proofs.Flock.Soundness.Discharge.Hm.Slot

/-!
# V*'s SHA-512 after a known prefix

V*'s SHA-512 of a message that starts after a known prefix (`verity_flock.rec_open._sha`), as the verifier's builder
(`Flock.HmNets`) states it: the chaining value after the prefix's blocks is constant, the message's bits are forms, SHA-512's
padding is constant forms, and each block is one `Flock.HmNets.compress` of its bits in word order (`wordBit`).

`sound_shaFrom`: whenever the rows hold and the message forms carry the bits of a byte string `data` (bit `q % 8` of byte
`q / 8`), the output's bit `t` is bit `t` of `SHA-512(pre ‖ data)` (`Flock.Sha512.hash`), for a prefix `pre` of whole
blocks whose chaining value is the constant (`preCv`).
-/

namespace FlockVBridge

open Flock.HmNets hiding and
open Flock FlockSoundness.Discharge.Hm

/-! ## The builder -/

/-- A byte string's bits, bit `t % 8` of byte `t / 8` (`rec_open._bits`). -/
def bitsOf (l : List UInt8) : List Bool := (List.range (8 * l.length)).map (bitL l)

/-- SHA-512's padding after a message of `total` bytes (`sha512.pad_tail`): `0x80`, the zero bytes, the bit count as 16
bytes big-endian. -/
def padTail (total : ℕ) : List UInt8 := [128] ++ List.replicate (zeroPad total) 0 ++ lenBytes total

/-- The message's forms after a prefix of `pre` bytes: its bits, then its padding's as constants. -/
def shaMsg (bits : List Lin) (pre : ℕ) : List Lin := bits ++ (bitsOf (padTail (pre + bits.length / 8))).map Lin.const

/-- The block of `msg` at bit `off`, in word order: its bit `wordBit t` is `msg`'s bit `off + t` (`wordBit` is its own
inverse on a block). -/
def blockAt (msg : List Lin) (off : ℕ) : List Lin := (List.range 1024).map fun p => msg.getD (off + wordBit p) Lin.zero

/-- One compression of the chaining value `st` (512 forms, word order) and the block `blk` (1024), its words
concatenated. -/
def shaBlock (st blk : List Lin) : M (List Lin) := do
  let out ← compress (words st 8) (words blk 16)
  pure out.flatten

/-- `n` compressions over the blocks of `msg` from block `k`. -/
def shaBlocks (msg : List Lin) : ℕ → ℕ → List Lin → M (List Lin)
  | 0, _, st => pure st
  | n + 1, k, st => do
    let st ← shaBlock st (blockAt msg (1024 * k))
    shaBlocks msg n (k + 1) st

/-- `rec_open._sha`: SHA-512 of a message whose first `pre` bytes left the chaining value `cv` and whose rest has the bits
`bits` (byte-string order); the digest's 512 bits, byte-string order. -/
def shaFrom (cv : List (BitVec 64)) (bits : List Lin) (pre : ℕ) : M (List Lin) := do
  let st ← shaBlocks (shaMsg bits pre) ((shaMsg bits pre).length / 1024) 0 (cv.map constW).flatten
  pure ((List.range 512).map fun t => st.getD (wordBit t) Lin.zero)

/-- The chaining value after the first `np` blocks of the prefix `pre`. -/
def preCv (pre : List UInt8) (np : ℕ) : List (BitVec 64) := chainV ⟨pre.toArray⟩ np

/-! ## Bytes and bits -/

theorem wordBit_lt_1024 {t : ℕ} (ht : t < 1024) : wordBit t < 1024 := by
  unfold wordBit; omega

theorem length_padTail (total : ℕ) : (padTail total).length = zeroPad total + 17 := by
  simp only [padTail, lenBytes, List.length_append, List.length_cons, List.length_nil, List.length_replicate,
    List.length_map, List.length_range']
  omega

theorem length_shaMsg (bits : List Lin) (pre : ℕ) :
    (shaMsg bits pre).length = bits.length + 8 * (zeroPad (pre + bits.length / 8) + 17) := by
  simp [shaMsg, bitsOf, length_padTail]

theorem length_blockAt (msg : List Lin) (off : ℕ) : (blockAt msg off).length = 1024 := by
  simp [blockAt]

theorem bit_blockAt (z : Var → Bool) (msg : List Lin) (off : ℕ) {q : ℕ} (hq : q < 1024) :
    bit z (blockAt msg off) q = ev z (msg.getD (off + wordBit q) Lin.zero) := by
  simp only [bit, blockAt, List.getElem?_map, List.getElem?_range hq, Option.map_some, Option.getD_some]

theorem bit_wordOrder (z : Var → Bool) (st : List Lin) {t : ℕ} (ht : t < 512) :
    bit z ((List.range 512).map fun t => st.getD (wordBit t) Lin.zero) t = bit z st (wordBit t) := by
  simp only [bit, List.getElem?_map, List.getElem?_range ht, Option.map_some, Option.getD_some,
    List.getD_eq_getElem?_getD]

theorem length_flatten_constW : ∀ cv : List (BitVec 64), (cv.map constW).flatten.length = 64 * cv.length
  | [] => rfl
  | k :: cv => by
      rw [List.map_cons, List.flatten_cons, List.length_append, length_constW, length_flatten_constW cv,
        List.length_cons]
      ring

theorem bitL_append_left {a : List UInt8} (b : List UInt8) {t : ℕ} (ht : t < 8 * a.length) :
    bitL (a ++ b) t = bitL a t := by
  rw [bitL, bitL, List.getD_append _ _ _ _ (by omega)]

theorem bitL_append_right (a : List UInt8) {b : List UInt8} {t : ℕ} (ht : 8 * a.length ≤ t) :
    bitL (a ++ b) t = bitL b (t - 8 * a.length) := by
  rw [bitL, bitL, List.getD_append_right _ _ _ _ (by omega), show t / 8 - a.length = (t - 8 * a.length) / 8 by omega,
    show t % 8 = (t - 8 * a.length) % 8 by omega]

/-- **The message forms carry `data ‖ padding`**: bit `i` of the message is bit `i` of `data` followed by SHA-512's
padding after `pre + |data|` bytes. -/
theorem ev_shaMsg (z : Var → Bool) {bits : List Lin} {data : List UInt8} (hbits : bits.length = 8 * data.length)
    (hdata : ∀ q < bits.length, bit z bits q = bitL data q) (pre : ℕ) {i : ℕ} (hi : i < (shaMsg bits pre).length) :
    ev z ((shaMsg bits pre).getD i Lin.zero) = bitL (data ++ padTail (pre + data.length)) i := by
  have hl : bits.length / 8 = data.length := by omega
  unfold shaMsg
  rw [hl]
  by_cases h : i < bits.length
  · rw [List.getD_append _ _ _ _ h, bitL_append_left _ (by omega), ← hdata i h, bit, List.getD_eq_getElem?_getD]
  · rw [List.getD_append_right _ _ _ _ (by omega), bitL_append_right _ (by omega)]
    have hj : i - bits.length < 8 * (padTail (pre + data.length)).length := by
      rw [length_shaMsg, hl, ← length_padTail] at hi
      omega
    rw [List.getD_eq_getElem _ _ (by simpa [bitsOf] using hj), List.getElem_map, ev_const]
    simp only [bitsOf, List.getElem_map, List.getElem_range]
    rw [hbits]

/-- Bit `q` of block `b` in word order is the padded message's bit `1024 b + wordBit q`. -/
theorem wbit_blockV (P : ByteArray) (b : ℕ) {q : ℕ} (hq : q < 1024) :
    wbit (blockV P b) q = bitL P.data.toList (1024 * b + wordBit q) := by
  rw [wbit, blockV, List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range (by omega), Option.map_some,
    Option.getD_some, wordBE_eq_wordL, getLsbD_wordL _ _ _ (by omega), bitL,
    show (1024 * b + wordBit q) / 8 = 128 * b + 8 * (q / 64) + (7 - q % 64 / 8) by unfold wordBit; omega,
    show (1024 * b + wordBit q) % 8 = q % 64 % 8 by unfold wordBit; omega]

/-- The bytes of `pre ‖ data`. -/
def catBytes (pre data : List UInt8) : ByteArray := ⟨(pre ++ data).toArray⟩

theorem data_catBytes (pre data : List UInt8) : (catBytes pre data).data.toList = pre ++ data := by
  simp [catBytes]

theorem size_catBytes (pre data : List UInt8) : (catBytes pre data).size = pre.length + data.length := by
  simp [catBytes, ByteArray.size]

theorem padData_catBytes (pre data : List UInt8) :
    (padData (catBytes pre data)).data.toList = pre ++ (data ++ padTail (pre.length + data.length)) := by
  rw [padData_data, data_catBytes, size_catBytes, padTail]
  simp only [List.append_assoc]

/-! ## The chain -/

theorem chain_zero (d : ByteArray) : chain d 0 = Sha512.iv := rfl

/-- The chain over the first `n` blocks reads only those blocks. -/
theorem chain_congr {d d' : ByteArray} (n : ℕ) (h : ∀ i < 128 * n, d.get! i = d'.get! i) : chain d n = chain d' n := by
  induction n with
  | zero => rw [chain_zero, chain_zero]
  | succ n ih =>
      rw [chain_succ, chain_succ, ih fun i hi => h i (by omega)]
      exact compress_congr _ fun k hk => h _ (by omega)

/-- After the prefix's blocks, the padded message's chain is the prefix's. -/
theorem chainV_catBytes {pre : List UInt8} {np : ℕ} (hpre : pre.length = 128 * np) (data : List UInt8) :
    chainV (padData (catBytes pre data)) np = preCv pre np := by
  rw [chainV, preCv, chainV, chain_congr np fun i hi => ?_]
  rw [get!_eq_getD, get!_eq_getD, padData_catBytes, List.getD_append _ _ _ _ (by omega)]

theorem length_chainV (d : ByteArray) (k : ℕ) : (chainV d k).length = 8 := by simp [chainV]

theorem wbit_chainV_toBV (d : ByteArray) {k : ℕ} (hk : 0 < k) : chainV d k = toBV (chain d k) := by
  obtain ⟨k, rfl⟩ := Nat.exists_eq_add_of_lt hk
  rw [chainV, map_range_eq_toBV _ (by rw [Nat.zero_add, chain_succ]; exact size_compress _ _ _)]

/-- The constant chaining value's forms carry its bits. -/
theorem bit_constCv (z : Var → Bool) {cv : List (BitVec 64)} (hcv : cv.length = 8) {p : ℕ} (hp : p < 512) :
    bit z (cv.map constW).flatten p = wbit cv p := by
  rw [bit_flatten z (words64_constW cv), List.getD_eq_getElem _ _ (by simp; omega), List.getElem_map, bv_constW, wbit,
    List.getD_eq_getElem _ _ (by omega)]

/-! ## What the rows force -/

theorem sound_shaBlock (st blk : List Lin) : Sound (shaBlock st blk) fun z out =>
    st.length = 512 → blk.length = 1024 → out.length = 512 ∧ ∀ q < 512,
      bit z out q = wbit (compressV ((words st 8).map (bv z)) ((words blk 16).map (bv z))) q :=
  sound_compressFlat st blk

/-- **One compression a block**: when the chaining value's forms carry the chain after `b` blocks of `P` and each of the
next `n` blocks of `msg` carries `P`'s block, the result carries the chain after `b + n` blocks. -/
theorem sound_shaBlocks (msg : List Lin) : ∀ (n k : ℕ) (st : List Lin), Sound (shaBlocks msg n k st) fun z out =>
    ∀ (P : ByteArray) (b : ℕ), st.length = 512 → (∀ p < 512, bit z st p = wbit (chainV P b) p) →
      (∀ j < n, ∀ q < 1024, bit z (blockAt msg (1024 * (k + j))) q = wbit (blockV P (b + j)) q) →
      out.length = 512 ∧ ∀ p < 512, bit z out p = wbit (chainV P (b + n)) p
  | 0, _, _ => Sound.pure' fun _ _ _ hl hst _ => ⟨hl, hst⟩
  | n + 1, k, st => by
      unfold shaBlocks
      refine Sound.bind' (sound_shaBlock st (blockAt msg (1024 * k))) fun st' =>
        Sound.mono (sound_shaBlocks msg n (k + 1) st') fun z out hih hstep P b hl hst hblk => ?_
      obtain ⟨hl', hst'⟩ := hstep hl (length_blockAt _ _)
      have hst64 : st.length = 64 * 8 := by omega
      have hblk64 : (blockAt msg (1024 * k)).length = 64 * 16 := length_blockAt _ _
      have ecv : (words st 8).map (bv z) = chainV P b :=
        map_bv_of_bits z (words64_words hst64) (by rw [length_words, length_chainV]) fun q hq => by
          rw [length_words] at hq
          rw [bit_words_flatten z hst64 hq]
          exact hst q (by omega)
      have eblk : (words (blockAt msg (1024 * k)) 16).map (bv z) = blockV P b :=
        map_bv_of_bits z (words64_words hblk64) (by simp [length_words, blockV]) fun q hq => by
          rw [length_words] at hq
          rw [bit_words_flatten z hblk64 hq]
          simpa using hblk 0 (by omega) q (by omega)
      obtain ⟨h1, h2⟩ := hih P (b + 1) hl' (fun p hp => by rw [hst' p hp, ecv, eblk, ← chainV_succ])
        fun j hj q hq => by
          rw [show k + 1 + j = k + (j + 1) by omega, show b + 1 + j = b + (j + 1) by omega]
          exact hblk (j + 1) (by omega) q hq
      exact ⟨h1, fun p hp => by rw [h2 p hp, show b + 1 + n = b + (n + 1) by omega]⟩

/-- **V*'s SHA-512 after a known prefix computes SHA-512.** Whenever the rows of `shaFrom cv bits (128 np)` hold under
`z`: if `bits` carries the bits of `data` (bit `q % 8` of byte `q / 8`) and `cv` is the chaining value after the `np`
blocks of `pre`, the output's bit `t < 512` is bit `t` of `SHA-512(pre ‖ data)`, byte-string order. -/
theorem sound_shaFrom (cv : List (BitVec 64)) (bits : List Lin) (np : ℕ) :
    Sound (shaFrom cv bits (128 * np)) fun z out => ∀ pre data : List UInt8, pre.length = 128 * np →
      bits.length = 8 * data.length → (∀ q < bits.length, bit z bits q = bitL data q) → cv = preCv pre np →
      out.length = 512 ∧ ∀ t < 512, bit z out t = bitL (Sha512.hash (catBytes pre data)).data.toList t := by
  unfold shaFrom
  refine Sound.bind' (sound_shaBlocks _ _ 0 _) fun st => Sound.pure' fun z hst pre data hpre hbits hdata hcv => ?_
  set P := padData (catBytes pre data) with hP
  set msg := shaMsg bits (128 * np) with hmsg
  have hl8 : bits.length / 8 = data.length := by omega
  have hsize : P.size = 128 * np + data.length + 1 + zeroPad (128 * np + data.length) + 16 := by
    rw [hP, size_padData, size_catBytes, hpre]
  have hmod : P.size % 128 = 0 := size_padData_mod _
  have hlen : msg.length = bits.length + 8 * (zeroPad (128 * np + data.length) + 17) := by
    rw [hmsg, length_shaMsg, hl8]
  have hnb : np + msg.length / 1024 = P.size / 128 := by omega
  have hcvP : cv = chainV P np := by rw [hcv, hP, chainV_catBytes hpre]
  have hbytes : P.data.toList = pre ++ (data ++ padTail (128 * np + data.length)) := by
    rw [hP, padData_catBytes, hpre]
  obtain ⟨-, hout⟩ := hst P np (by rw [length_flatten_constW, hcvP, length_chainV])
    (fun p hp => by rw [bit_constCv z (by rw [hcvP, length_chainV]) hp, hcvP])
    fun j hj q hq => by
      have hw := wordBit_lt_1024 hq
      have hi : 1024 * (0 + j) + wordBit q < msg.length := by
        have : 1024 * (j + 1) ≤ msg.length := by
          have := (Nat.lt_div_iff_mul_lt (by norm_num : 0 < 1024)).1 hj
          omega
        omega
      rw [bit_blockAt z _ _ hq, ev_shaMsg z hbits hdata _ hi, wbit_blockV P _ hq, hbytes,
        bitL_append_right pre (by rw [hpre]; omega), hpre,
        show 1024 * (np + j) + wordBit q - 8 * (128 * np) = 1024 * (0 + j) + wordBit q by omega]
  refine ⟨by simp, fun t ht => ?_⟩
  rw [bit_wordOrder z st ht, hout _ (wordBit_lt ht), hnb, hash_eq, bitL_digestOf _ ht,
    wbit_chainV_toBV _ (blocks_pos _)]

end FlockVBridge
