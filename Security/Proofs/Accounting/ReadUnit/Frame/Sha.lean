/-
The bytes the frame-v3 read's words hold, and its SHA-512s: words holding a byte string (`HoldsBytes`) hold it in leaf
order (`wordsBytes_of_holds`) and, eight of them holding a digest's bytes, that digest (`wordsDigest_of_holds`); a byte
string's byte forms (`bytesW`) hold its words' bytes (`bytesV_bytesW`); a framed message's words hold the frame and
SHA-512's padding (`holds_message`); and the chain of compressions over a padded message's words returns the message's
SHA-512 (`sound_sha512W`).
-/
module

public import Proofs.Accounting.ReadUnit.Frame.Gadgets
public import Proofs.Flock.Soundness.Discharge.Hm.Hm96
public import Proofs.Flock.Soundness.Discharge.Hm.Hash

@[expose] public section

namespace Accounting.ReadUnit.FrameRead

open Flock.HmNets hiding and
open FlockSoundness.Discharge.Hm
open Flock.ReadNet Flock.ReadNet.Frame
open Accounting.ReadUnit.Gates

/-! ## Lists -/

theorem getD_take_drop {α : Type} (l : List α) (s k q : ℕ) (d : α) (hq : q < k) :
    ((l.drop s).take k).getD q d = l.getD (s + q) d := by
  rw [List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD, List.getElem?_take_of_lt hq, List.getElem?_drop]

theorem ext_getD {α : Type} {l l' : List α} (d : α) (hl : l.length = l'.length)
    (h : ∀ n < l.length, l.getD n d = l'.getD n d) : l = l' := by
  refine List.ext_getElem hl fun n h1 h2 => ?_
  have := h n h1
  rwa [List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD, List.getElem?_eq_getElem h1,
    List.getElem?_eq_getElem h2, Option.getD_some, Option.getD_some] at this

theorem getD_range {n i : ℕ} (hi : i < n) : (List.range n).getD i 0 = i := by
  rw [List.getD_eq_getElem?_getD, List.getElem?_range hi, Option.getD_some]

theorem getD_map_range {α : Type} (g : ℕ → α) {k m : ℕ} (hm : m < k) (d : α) : ((List.range k).map g).getD m d = g m := by
  rw [List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range hm]
  rfl

theorem length_flatMap_range {γ : Type} (N k : ℕ) (g : ℕ → ℕ → γ) :
    ((List.range N).flatMap fun w => (List.range k).map (g w)).length = k * N := by
  have := length_flatMap_const (fun w => (List.range k).map (g w)) k (List.range N) fun a _ => by simp
  rwa [List.length_range] at this

/-! ## Words as bytes -/

theorem length_flatten_words64 : ∀ {ws : List Word}, Words64 ws → ws.flatten.length = 64 * ws.length
  | [], _ => rfl
  | w :: ws, h => by
    rw [List.flatten_cons, List.length_append, h w List.mem_cons_self,
      length_flatten_words64 fun x hx => h x (List.mem_cons_of_mem _ hx), List.length_cons]
    omega

theorem ev_flatten (z : Var → Bool) {ws : List Word} (h : Words64 ws) {i p : ℕ} (hi : i < ws.length) (hp : p < 64) :
    ev z (ws.flatten.getD (64 * i + p) Lin.zero) = ev z ((ws.getD i []).getD p Lin.zero) := by
  have h1 := bit_flatten z h (64 * i + p)
  rw [show (64 * i + p) / 64 = i by omega, show (64 * i + p) % 64 = p by omega,
    ← bit_eq_getLsbD (w := ws.getD i []) (h.getD hi).le] at h1
  simpa only [bit, List.getD_eq_getElem?_getD] using h1

theorem length_wordsBytes' (z : Var → Bool) {ws : List Word} (hw : Words64 ws) :
    (wordsBytes z ws).length = 8 * ws.length := by
  rw [wordsBytes, ReadUnit.bytesOf, length_flatMap_range]
  simp only [evs, List.length_map, length_flatten_words64 hw]
  omega

/-- **Words holding bytes hold them in leaf order.** -/
theorem wordsBytes_of_holds {z : Var → Bool} {ws : List Word} {bs : List UInt8} (h : HoldsBytes z ws bs) :
    wordsBytes z ws = bs := by
  obtain ⟨hw, hlen, hb⟩ := h
  have hfl := length_flatten_words64 hw
  have hevs : (evs z ws.flatten).length = 64 * ws.length := by simp [evs, hfl]
  refine ext_getD 0 (by rw [length_wordsBytes' z hw, hlen]) fun n hn => ?_
  rw [length_wordsBytes' z hw] at hn
  obtain ⟨i, m, hm, rfl⟩ : ∃ i m, m < 8 ∧ n = 8 * i + m := ⟨n / 8, n % 8, Nat.mod_lt _ (by norm_num), by omega⟩
  have hi : i < ws.length := by omega
  rw [wordsBytes, ReadUnit.bytesOf,
    getD_flatMap (fun w => (List.range 8).map fun m =>
        UInt8.ofNat (ReadUnit.num (((evs z ws.flatten).drop (64 * w + 8 * (7 - m))).take 8))) 8 0 0
      (List.range ((evs z ws.flatten).length / 64)) (fun a _ => by simp) i m
      (by rw [List.length_range, hevs]; omega) hm,
    getD_range (n := (evs z ws.flatten).length / 64) (i := i) (by rw [hevs]; omega), getD_map_range _ hm]
  have hl8 : ((ws.flatten.drop (64 * i + 8 * (7 - m))).take 8).length = 8 := by
    simp only [List.length_take, List.length_drop, hfl]
    omega
  refine uint8_ext fun q hq => ?_
  have hc : ((evs z ws.flatten).drop (64 * i + 8 * (7 - m))).take 8 =
      evs z ((ws.flatten.drop (64 * i + 8 * (7 - m))).take 8) := by
    simp only [evs, List.map_drop, List.map_take]
  rw [hc]
  change (byteV z ((ws.flatten.drop (64 * i + 8 * (7 - m))).take 8)).toNat.testBit q = _
  rw [testBit_byteV z hl8 hq, getD_take_drop _ _ _ _ _ hq,
    show 64 * i + 8 * (7 - m) + q = 64 * i + (8 * (7 - m) + q) by omega,
    ev_flatten z hw hi (p := 8 * (7 - m) + q) (by omega), hb i hi (8 * (7 - m) + q) (by omega),
    show (8 * (7 - m) + q) / 8 = 7 - m by omega, show (8 * (7 - m) + q) % 8 = q by omega,
    show 8 * i + 7 - (7 - m) = 8 * i + m by omega]

/-- **Words holding a digest's bytes hold the digest.** -/
theorem wordsDigest_of_holds {z : Var → Bool} {ws : List Word} {d : Verity.Digest} (h : HoldsBytes z ws d.bytes) :
    wordsDigest z ws = d := by
  funext i
  have := congrArg (fun l => l.getD i.val 0) (wordsBytes_of_holds h)
  simp only [wordsBytes] at this
  show (ReadUnit.bytesOf (evs z ws.flatten)).getD i.val 0 = d i
  rw [this]
  simp [Verity.Digest.bytes, List.getD_eq_getElem?_getD, List.getElem?_ofFn]

theorem bytes_wordsDigest (z : Var → Bool) {ws : List Word} (h : Shaped 8 ws) :
    (wordsDigest z ws).bytes = wordsBytes z ws := by
  have hl : (wordsBytes z ws).length = 64 := by rw [length_wordsBytes' z h.2, h.1]
  refine ext_getD 0 (by rw [Verity.Digest.bytes, List.length_ofFn, hl]) fun n hn => ?_
  rw [Verity.Digest.bytes, List.length_ofFn] at hn
  simp only [Verity.Digest.bytes, List.getD_eq_getElem?_getD, List.getElem?_ofFn, hn, dite_true, Option.getD_some]
  rfl

theorem length_bytesW (ws : List Word) : (bytesW ws).length = 8 * ws.length :=
  length_flatMap_const (fun w : Word => (List.range 8).map fun b => (w.drop (8 * (7 - b))).take 8) 8 ws
    fun a _ => by simp

theorem bytes8_bytesW {ws : List Word} (hw : Words64 ws) : Bytes8 (bytesW ws) := by
  intro b hb
  obtain ⟨w, hw', hb⟩ := List.mem_flatMap.1 hb
  obtain ⟨j, hj, rfl⟩ := List.mem_map.1 hb
  rw [List.mem_range] at hj
  simp only [List.length_take, List.length_drop, hw w hw']
  omega

theorem holds_bytesW (z : Var → Bool) {ws : List Word} (hw : Words64 ws) : HoldsBytes z ws (bytesV z (bytesW ws)) := by
  refine ⟨hw, by rw [length_bytesV, length_bytesW], fun i hi p hp => ?_⟩
  have hwi := hw.getD hi
  rw [getD_bytesV, show 8 * i + 7 - p / 8 = 8 * i + (7 - p / 8) by omega, bytesW,
    getD_flatMap (fun w : Word => (List.range 8).map fun b => (w.drop (8 * (7 - b))).take 8) 8 [] [] ws
      (fun a _ => by simp) i (7 - p / 8) hi (by omega),
    getD_map_range _ (show 7 - p / 8 < 8 by omega), show 8 * (7 - (7 - p / 8)) = 8 * (p / 8) by omega]
  have hl8 : (((ws.getD i []).drop (8 * (p / 8))).take 8).length = 8 := by
    simp only [List.length_take, List.length_drop, hwi]
    omega
  rw [testBit_byteV z hl8 (Nat.mod_lt _ (by norm_num)), getD_take_drop _ _ _ _ _ (Nat.mod_lt _ (by norm_num)),
    Nat.div_add_mod]

/-- **A byte string's byte forms hold its words' bytes.** -/
theorem bytesV_bytesW (z : Var → Bool) {ws : List Word} (hw : Words64 ws) : bytesV z (bytesW ws) = wordsBytes z ws :=
  (wordsBytes_of_holds (holds_bytesW z hw)).symm

/-! ## Framed messages -/

/-- `frame_v3._message`'s bytes before SHA-512's padding: the frame of `parts` under `tag`. -/
def msgData (tag : List UInt8) (parts : List (List Byte)) : List Byte :=
  constBytes (FRAME ++ beBytes 4 tag.length ++ tag) ++ parts.flatMap fun p => constBytes (beBytes 8 p.length) ++ p

theorem message_eq (tag : List UInt8) (parts : List (List Byte)) :
    message tag parts = wordsB (msgData tag parts ++ constBytes (padTail (msgData tag parts).length)) := rfl

theorem bytesV_flatMap {α : Type} (z : Var → Bool) (f : α → List Byte) (l : List α) :
    bytesV z (l.flatMap f) = l.flatMap fun a => bytesV z (f a) := by
  rw [bytesV, List.map_flatMap]
  rfl

/-- **A framed message's bytes**: the frame of the parts' bytes. -/
theorem bytesV_msgData (z : Var → Bool) (tag : List UInt8) (parts : List (List Byte)) :
    bytesV z (msgData tag parts) =
      FRAME ++ beBytes 4 tag.length ++ tag ++ (parts.map (bytesV z)).flatMap fun p => beBytes 8 p.length ++ p := by
  rw [msgData, bytesV_append, bytesV_constBytes, bytesV_flatMap, List.flatMap_map]
  congr 1
  congr 1
  funext p
  rw [bytesV_append, bytesV_constBytes, length_bytesV]

theorem frame_eq (tag : String) (ps : List Hash.Msg) :
    FRAME ++ beBytes 4 (Hash.utf8 tag).length ++ Hash.utf8 tag ++ ps.flatMap (fun p => beBytes 8 p.length ++ p) =
      Hash.Tree.frame tag ps := rfl

theorem bytes8_flatMap {α : Type} {f : α → List Byte} {l : List α} (h : ∀ a ∈ l, Bytes8 (f a)) :
    Bytes8 (l.flatMap f) := by
  intro b hb
  obtain ⟨a, ha, hb⟩ := List.mem_flatMap.1 hb
  exact h a ha b hb

theorem bytes8_msgData (tag : List UInt8) {parts : List (List Byte)} (h : ∀ p ∈ parts, Bytes8 p) :
    Bytes8 (msgData tag parts) :=
  bytes8_append (bytes8_constBytes _) (bytes8_flatMap fun p hp => bytes8_append (bytes8_constBytes _) (h p hp))

theorem length_padTail (len : ℕ) : (padTail len).length = 17 + (128 - (len + 17) % 128) % 128 := by
  simp only [padTail, beBytes, List.length_cons, List.length_append, List.length_replicate, List.length_map,
    List.length_range] <;> omega

theorem length_msgData (tag : List UInt8) (parts : List (List Byte)) :
    (msgData tag parts).length = FRAME.length + 4 + tag.length + (parts.map fun p => 8 + p.length).sum := by
  simp only [msgData, List.length_append, length_constBytes, beBytes, List.length_map, List.length_range,
    List.length_flatMap] <;> omega

theorem length_message (tag : List UInt8) (parts : List (List Byte)) :
    (message tag parts).length =
      ((msgData tag parts).length + 17 + (128 - ((msgData tag parts).length + 17) % 128) % 128) / 8 := by
  rw [message_eq, wordsB, List.length_map, List.length_range, List.length_append, length_constBytes,
    length_padTail] <;> omega

/-- **A framed message's words** hold the frame's bytes and SHA-512's padding. -/
theorem holds_message (z : Var → Bool) (tag : List UInt8) {parts : List (List Byte)} (h : ∀ p ∈ parts, Bytes8 p) :
    HoldsBytes z (message tag parts) (bytesV z (msgData tag parts) ++ padTail (msgData tag parts).length) := by
  have := holds_wordsB z (data := msgData tag parts ++ constBytes (padTail (msgData tag parts).length))
    (bytes8_append (bytes8_msgData tag h) (bytes8_constBytes _))
    (by rw [List.length_append, length_constBytes, length_padTail] <;> omega)
  rwa [bytesV_append, bytesV_constBytes] at this

/-! ## SHA-512 -/

theorem lenBytes_eq (len : ℕ) : lenBytes len = beBytes 16 (8 * len) := by
  rw [lenBytes, beBytes, ← List.range_eq_range']
  refine List.map_congr_left fun a ha => ?_
  rw [List.mem_range] at ha
  rw [Nat.shiftRight_eq_div_pow, show 8 * (15 - a) = 8 * (16 - 1 - a) by omega, pow_mul, Nat.mul_comm len 8]
  norm_num

/-- SHA-512's padded message is the message and `padTail`. -/
theorem padData_toList (m : List UInt8) : (padData ⟨m.toArray⟩).data.toList = m ++ padTail m.length := by
  have hs : (⟨m.toArray⟩ : ByteArray).size = m.length := by simp [ByteArray.size]
  rw [padData_data, hs, lenBytes_eq, zeroPad, padTail,
    show (240 - (m.length + 1) % 128) % 128 = (128 - (m.length + 17) % 128) % 128 by omega]
  simp

theorem size_chain (data : ByteArray) : ∀ n, (chain data n).size = 8
  | 0 => by simp [chain, Flock.Sha512.iv]
  | n + 1 => by rw [chain_succ, size_compress]

theorem length_ivW : ivW.length = 8 := by simp [ivW, Flock.Sha512.iv]

theorem words64_ivW : Words64 ivW := by
  intro w hw
  simp only [ivW, List.mem_map] at hw
  obtain ⟨v, _, rfl⟩ := hw
  exact length_constW _

theorem map_bv_ivW (z : Var → Bool) (data : ByteArray) : ivW.map (bv z) = toBV (chain data 0) := by
  simp [ivW, List.map_map, Function.comp_def, bv_constW, toBV, chain]

/-- The words of block `b` of `ws` hold the 16 big-endian words of block `b` of the padded message `data`. -/
def BlockHolds (z : Var → Bool) (ws : List Word) (data : ByteArray) (b : ℕ) : Prop :=
  ((ws.drop (16 * b)).take 16).length = 16 ∧ Words64 ((ws.drop (16 * b)).take 16) ∧
    ((ws.drop (16 * b)).take 16).map (bv z) = (List.range 16).map fun j => (wordBE data (128 * b) j).toBitVec

/-- **The chain of compressions** over blocks `i … i + k - 1`, from the chaining value `st` after block `i`. -/
theorem sound_chain (ws : List Word) : ∀ (k i : ℕ) (st : List Word),
    Sound ((List.range' i k).foldlM (fun st i => compress st ((ws.drop (16 * i)).take 16)) st) fun z out =>
      ∀ data : ByteArray, (∀ b, i ≤ b → b < i + k → BlockHolds z ws data b) → Words64 st → st.length = 8 →
        st.map (bv z) = toBV (chain data i) →
          Words64 out ∧ out.length = 8 ∧ out.map (bv z) = toBV (chain data (i + k))
  | 0, i, st => Sound.pure' fun z data _ hw hl hv => ⟨hw, hl, by simpa using hv⟩
  | k + 1, i, st => by
    rw [List.range'_succ, List.foldlM_cons]
    refine Sound.bind' (sound_compress st _) fun st' =>
      Sound.mono (sound_chain k (i + 1) st') fun z out h hc data hb hw hl hv => ?_
    obtain ⟨hbl, hbw, hbv⟩ := hb i le_rfl (by omega)
    obtain ⟨hw', hv'⟩ := hc hl hw hbl hbw
    have hv'' : st'.map (bv z) = toBV (chain data (i + 1)) := by
      rw [hv', hv, hbv, chain_succ, compress_eq, map_range_eq_toBV _ (size_chain data i)]
    obtain ⟨h1, h2, h3⟩ := h data (fun b hb1 hb2 => hb b (by omega) (by omega)) hw'
      (by rw [length_of_map_bv hv'', length_toBV, size_chain]) hv''
    exact ⟨h1, h2, by rw [h3, show i + 1 + k = i + (k + 1) by omega]⟩

theorem blockHolds_of_holds {z : Var → Bool} {ws : List Word} {m : List UInt8} (h : HoldsBytes z ws (m ++ padTail m.length))
    {b : ℕ} (hb : 16 * b + 16 ≤ ws.length) : BlockHolds z ws (padData ⟨m.toArray⟩) b := by
  obtain ⟨hw, hlen, hbits⟩ := h
  have hl : ((ws.drop (16 * b)).take 16).length = 16 := by
    simp only [List.length_take, List.length_drop]
    omega
  refine ⟨hl, Words64.take (Words64.drop hw _) _,
    List.ext_getElem (by rw [List.length_map, List.length_map, hl, List.length_range]) fun j h1 h2 => ?_⟩
  rw [List.length_map, hl] at h1
  simp only [List.getElem_map, List.getElem_take, List.getElem_drop, List.getElem_range]
  have hj : 16 * b + j < ws.length := by omega
  have hw64 : ws[16 * b + j].length = 64 := hw _ (List.getElem_mem hj)
  refine bv_ext (by omega) fun p hp => ?_
  rw [wordBE_eq_wordL, getLsbD_wordL _ _ _ hp, padData_toList, bit, ← List.getD_eq_getElem?_getD]
  have := hbits (16 * b + j) hj p hp
  rw [List.getD_eq_getElem?_getD (l := ws), List.getElem?_eq_getElem hj, Option.getD_some] at this
  rw [this, show 8 * (16 * b + j) + 7 - p / 8 = 128 * b + 8 * j + (7 - p / 8) by omega]

theorem bitL_eq (l : List UInt8) (k q : ℕ) (hq : q < 8) : (l.getD k 0).toNat.testBit q = bitL l (8 * k + q) := by
  rw [bitL, show (8 * k + q) / 8 = k by omega, show (8 * k + q) % 8 = q by omega]

/-- **SHA-512 of a padded message**: on words holding `m` and SHA-512's padding, the chain of compressions from the IV
returns 8 words holding `m`'s SHA-512. -/
theorem sound_sha512W (ws : List Word) : Sound (sha512W ws) fun z out => ∀ m : List UInt8,
    HoldsBytes z ws (m ++ padTail m.length) → Shaped 8 out ∧ wordsDigest z out = Verity.sha512 m := by
  unfold sha512W
  rw [List.range_eq_range']
  refine Sound.mono (sound_chain ws _ 0 ivW) fun z out h m hm => ?_
  have hd := padData_toList m
  have hsz : (padData ⟨m.toArray⟩).size = 8 * ws.length := by
    rw [← ByteArray.size_data, ← Array.length_toList, hd, hm.2.1]
  have hmod := size_padData_mod ⟨m.toArray⟩
  have hnb : (ws.length + 15) / 16 = (padData ⟨m.toArray⟩).size / 128 := by omega
  obtain ⟨h1, h2, h3⟩ := h (padData ⟨m.toArray⟩) (fun b _ hb => blockHolds_of_holds hm (by omega)) words64_ivW
    length_ivW (map_bv_ivW z _)
  rw [Nat.zero_add, hnb] at h3
  refine ⟨⟨h2, h1⟩, wordsDigest_of_holds ⟨h1, by simp [h2, Verity.Digest.bytes], fun i hi p hp => ?_⟩⟩
  have hk : 8 * i + 7 - p / 8 < 64 := by omega
  have ht : 8 * (8 * i + 7 - p / 8) + p % 8 < 512 := by omega
  have hwb : wordBit (8 * (8 * i + 7 - p / 8) + p % 8) = 64 * i + p := by unfold wordBit; omega
  have eb : (Verity.sha512 m).bytes.getD (8 * i + 7 - p / 8) 0 =
      (Flock.Sha512.hash ⟨m.toArray⟩).data.toList.getD (8 * i + 7 - p / 8) 0 := by
    simp [Verity.Digest.bytes, Verity.sha512, List.getD_eq_getElem?_getD, List.getElem?_ofFn, hk]
  have e1 : bv z (out.getD i []) = ((chain (padData ⟨m.toArray⟩) ((padData ⟨m.toArray⟩).size / 128))[i]!).toBitVec := by
    rw [← getD_toBV, ← h3]
    simp [List.getD_eq_getElem?_getD, hi]
  change bit z (out.getD i []) p = _
  rw [eb, bitL_eq _ _ _ (Nat.mod_lt _ (by norm_num)), hash_eq, bitL_digestOf _ ht, hwb, wbit,
    show (64 * i + p) / 64 = i by omega, show (64 * i + p) % 64 = p by omega, getD_toBV, ← e1]
  exact bit_eq_getLsbD (h1.getD hi).le p

end Accounting.ReadUnit.FrameRead
