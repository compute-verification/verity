/-
The bytes the frame-v3 read's words hold, and its SHA-512s: words holding a byte string (`HoldsBytes`) hold it in leaf
order (`wordsBytes_of_holds`) and, eight of them holding a digest's bytes, that digest (`wordsDigest_of_holds`); a byte
string's byte forms (`bytesW`) hold its words' bytes (`bytesV_bytesW`); a framed message's words hold the frame and
SHA-512's padding (`holds_message`); and `shaW` from the IV over a padded message's words returns the message's SHA-512
(`sound_sha512W`, from `Gates.sound_shaW` and `Gates.hash_chainW`).
-/
module

public import Proofs.Accounting.ReadUnit.Frame.Gadgets
public import Proofs.Accounting.ReadUnit.Sha
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

theorem getD_of_lt {α : Type} (l : List α) (i : ℕ) (h : i < l.length) (d : α) : l.getD i d = l[i] := by
  rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem h, Option.getD_some]

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
  simpa only [FlockSoundness.Discharge.Hm.bit, List.getD_eq_getElem?_getD] using h1

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
  have hevs : (evs z ws.flatten).length = 64 * ws.length := by simp only [evs, List.length_map, hfl]
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
  rw [this, Verity.Digest.bytes, getD_of_lt (List.ofFn d) i.val (by rw [List.length_ofFn]; exact i.isLt) 0,
    List.getElem_ofFn]

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
  simp only [msgData, bytesV_append, bytesV_constBytes, bytesV_flatMap, List.flatMap_map, length_bytesV]

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

theorem length_msgData (tag : List UInt8) (parts : List (List Byte)) :
    (msgData tag parts).length = FRAME.length + 4 + tag.length + (parts.map fun p => 8 + p.length).sum := by
  simp only [msgData, List.length_append, length_constBytes, beBytes, List.length_map, List.length_range,
    List.length_flatMap]

theorem length_message (tag : List UInt8) (parts : List (List Byte)) :
    (message tag parts).length =
      ((msgData tag parts).length + 17 + (128 - ((msgData tag parts).length + 17) % 128) % 128) / 8 := by
  rw [message_eq, wordsB, List.length_map, List.length_range, List.length_append, length_constBytes,
    length_padTail]
  unfold zeroPad
  omega

/-- **A framed message's words** hold the frame's bytes and SHA-512's padding. -/
theorem holds_message (z : Var → Bool) (tag : List UInt8) {parts : List (List Byte)} (h : ∀ p ∈ parts, Bytes8 p) :
    HoldsBytes z (message tag parts) (bytesV z (msgData tag parts) ++ padTail (msgData tag parts).length) := by
  have := holds_wordsB z (data := msgData tag parts ++ constBytes (padTail (msgData tag parts).length))
    (bytes8_append (bytes8_msgData tag h) (bytes8_constBytes _))
    (by rw [List.length_append, length_constBytes, length_padTail]; unfold zeroPad; omega)
  rwa [bytesV_append, bytesV_constBytes] at this

/-! ## SHA-512 -/

theorem length_ivW : ivW.length = 8 := by simp [ivW, Flock.Sha512.iv]

theorem words64_ivW : Words64 ivW := by
  intro w hw
  simp only [ivW, List.mem_map] at hw
  obtain ⟨v, _, rfl⟩ := hw
  exact length_constW _

/-- The IV's words are the chaining value after no blocks. -/
theorem map_bv_ivW (z : Var → Bool) (data : ByteArray) : ivW.map (bv z) = chainV data 0 := by
  rw [chainV, map_range_eq_toBV _ (by simp [chain, Flock.Sha512.iv])]
  simp [ivW, List.map_map, Function.comp_def, bv_constW, toBV, chain]

/-- Words holding a byte string are its big-endian words. -/
theorem map_bv_of_holds {z : Var → Bool} {ws : List Word} {bs : List UInt8} (h : HoldsBytes z ws bs) :
    ws.map (bv z) = wordsBE bs := by
  obtain ⟨hw, hlen, hbits⟩ := h
  apply List.ext_getElem
  · rw [List.length_map, length_wordsBE]
    omega
  · intro i h1 _
    rw [List.length_map] at h1
    simp only [List.getElem_map, wordsBE, List.getElem_range]
    refine bv_ext (hw _ (List.getElem_mem h1)).le fun p hp => ?_
    rw [getLsbD_wordL _ _ _ hp, FlockSoundness.Discharge.Hm.bit, ← List.getD_eq_getElem?_getD]
    have := hbits i h1 p hp
    rw [List.getD_eq_getElem?_getD (l := ws), List.getElem?_eq_getElem h1, Option.getD_some] at this
    rw [this, show 8 * i + 7 - p / 8 = 0 + 8 * i + (7 - p / 8) by omega]

/-- **SHA-512 of a padded message**: on words holding `m` and SHA-512's padding, the chain of compressions from the IV
(`shaW` from `ivW`) returns 8 words holding `m`'s SHA-512. -/
theorem sound_sha512W (ws : List Word) : Sound (shaW ((ws.length + 15) / 16) ivW ws) fun z out => ∀ m : List UInt8,
    HoldsBytes z ws (m ++ padTail m.length) → Shaped 8 out ∧ wordsDigest z out = Verity.sha512 m := by
  refine Sound.mono (sound_shaW _ ivW ws) fun z out h m hm => ?_
  have hlen : 8 * ws.length = 128 * blocks m.length := by rw [hm.2.1, List.length_append, blocks_mul]
  obtain ⟨o1, o2, o3⟩ := h length_ivW words64_ivW (by omega) hm.1
  have hc := hash_chainW isSha512_sha512 (pre := []) (data := m) (np := 0) rfl
  simp only [List.nil_append, List.length_nil, Nat.zero_add, Nat.sub_zero] at hc
  rw [← map_bv_ivW z] at hc
  refine ⟨⟨o1, o2⟩, ?_⟩
  rw [← Hash.BindingProofs.bytes_inj, wordsDigest_bytes z ⟨o1, o2⟩, o3, hc, map_bv_of_holds hm,
    show (ws.length + 15) / 16 = blocks m.length by omega]

end Accounting.ReadUnit.FrameRead
