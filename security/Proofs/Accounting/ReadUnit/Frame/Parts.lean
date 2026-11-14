/-
The frame-v3 read's parts over `Sound`: a hidden `uint`'s bytes and the flags over its length, the selection of the
layout that length picks, a message over a hidden `uint` (SHA-512 of the frame at the `uint`'s own length), the leaf,
the swap, a node, and the climb up the path against `Hash.Tree.climb`.
-/
module

public import Proofs.Accounting.ReadUnit.Frame.Sha

@[expose] public section

namespace Accounting.ReadUnit.FrameRead

open Flock.HmNets hiding and
open FlockSoundness.Discharge.Hm
open Flock.ReadNet Flock.ReadNet.Frame
open Accounting.ReadUnit.Gates

/-! ## Lists -/

theorem getD_map' {α β : Type} (f : α → β) (l : List α) (i : ℕ) (hi : i < l.length) (d : α) (d' : β) :
    (l.map f).getD i d' = f (l.getD i d) := by
  rw [getD_of_lt (l.map f) i (by rw [List.length_map]; exact hi) d', List.getElem_map, getD_of_lt l i hi d]

theorem getD_zip {α β : Type} (a : List α) (b : List β) (i : ℕ) (ha : i < a.length) (hb : i < b.length) (da : α)
    (db : β) (d : α × β) : (a.zip b).getD i d = (a.getD i da, b.getD i db) := by
  rw [getD_of_lt (a.zip b) i (by rw [List.length_zip]; omega) d, List.getElem_zip, getD_of_lt a i ha da,
    getD_of_lt b i hb db]

theorem getD_zipWith' {α β γ : Type} (f : α → β → γ) (a : List α) (b : List β) (i : ℕ) (ha : i < a.length)
    (hb : i < b.length) (da : α) (db : β) (d : γ) :
    (List.zipWith f a b).getD i d = f (a.getD i da) (b.getD i db) := by
  rw [getD_of_lt (List.zipWith f a b) i (by rw [List.length_zipWith]; omega) d, List.getElem_zipWith,
    getD_of_lt a i ha da, getD_of_lt b i hb db]

theorem shaped_getD {x : List Word} (hx : Shaped 8 x) {i : ℕ} (hi : i < 8) : (x.getD i []).length = 64 := by
  rw [getD_of_lt x i (by rw [hx.1]; exact hi) []]
  exact hx.2 _ (List.getElem_mem _)

theorem shaped_of_getD {x : List Word} {k : ℕ} (hl : x.length = k) (h : ∀ i < k, (x.getD i []).length = 64) :
    Shaped k x := by
  refine ⟨hl, fun w hw => ?_⟩
  obtain ⟨i, hi, rfl⟩ := List.getElem_of_mem hw
  have := h i (hl ▸ hi)
  rwa [getD_of_lt x i hi []] at this

/-- Two shaped word lists hold the same bits when they agree bit by bit. -/
theorem map_evs_ext (z : Var → Bool) {x y : List Word} (hx : Shaped 8 x) (hy : Shaped 8 y)
    (h : ∀ i < 8, ∀ j < 64, ev z ((x.getD i []).getD j Lin.zero) = ev z ((y.getD i []).getD j Lin.zero)) :
    x.map (evs z) = y.map (evs z) := by
  refine List.ext_getElem (by rw [List.length_map, List.length_map, hx.1, hy.1]) fun i h1 _ => ?_
  have hi : i < 8 := by rw [List.length_map, hx.1] at h1; exact h1
  have hxl : i < x.length := by rw [hx.1]; exact hi
  have hyl : i < y.length := by rw [hy.1]; exact hi
  have hxi : x[i].length = 64 := hx.2 _ (List.getElem_mem _)
  have hyi : y[i].length = 64 := hy.2 _ (List.getElem_mem _)
  rw [List.getElem_map, List.getElem_map]
  refine List.ext_getElem (by simp only [evs, List.length_map, hxi, hyi]) fun j g1 _ => ?_
  have hj : j < 64 := by simpa only [evs, List.length_map, hxi] using g1
  have := h i hi j hj
  rw [getD_of_lt x i hxl [], getD_of_lt y i hyl [], getD_of_lt x[i] j (by rw [hxi]; exact hj) Lin.zero,
    getD_of_lt y[i] j (by rw [hyi]; exact hj) Lin.zero] at this
  simpa only [evs, List.getElem_map] using this

theorem wordsDigest_congr (z : Var → Bool) {x y : List Word} (h : x.map (evs z) = y.map (evs z)) :
    wordsDigest z x = wordsDigest z y := by
  simp only [wordsDigest, evs, List.map_flatten]
  exact congrArg _ (congrArg _ h)

theorem sound_mapM_map {β γ δ : Type} (d : γ) (f : β → M γ) (g : (Var → Bool) → γ → δ)
    (h : (Var → Bool) → β → δ) (hf : ∀ x, Sound (f x) fun z r => g z r = h z x) (l : List β) :
    Sound (l.mapM f) fun z rs => rs.map (g z) = l.map (h z) := by
  refine Sound.mono (sound_mapM_getD d f (fun z x r => g z r = h z x) hf l) fun z rs hr => ?_
  obtain ⟨hl, hall⟩ := hr
  refine List.ext_getElem (by rw [List.length_map, List.length_map, hl]) fun i h1 _ => ?_
  have hi : i < rs.length := by rw [List.length_map] at h1; exact h1
  rw [List.getElem_map, List.getElem_map, ← hall i (hl ▸ hi), getD_of_lt rs i hi d]

/-! ## Numbers and big-endian bytes -/

theorem testBit_ofNat8 (n : ℕ) {q : ℕ} (hq : q < 8) : (UInt8.ofNat n).toNat.testBit q = n.testBit q := by
  rw [UInt8.toNat_ofNat', Nat.testBit_mod_two_pow]
  simp [hq]

theorem testBit_beBytes (k v : ℕ) {i q : ℕ} (hi : i < k) (hq : q < 8) :
    ((beBytes k v).getD i 0).toNat.testBit q = v.testBit (8 * (k - 1 - i) + q) := by
  rw [beBytes, getD_map_range _ hi]
  show (UInt8.ofNat (v / 256 ^ (k - 1 - i) % 256)).toNat.testBit q = _
  rw [testBit_ofNat8 _ hq, show (256 : ℕ) = 2 ^ 8 by norm_num, ← pow_mul, Nat.testBit_mod_two_pow,
    Nat.testBit_div_two_pow, show q + 8 * (k - 1 - i) = 8 * (k - 1 - i) + q by omega]
  simp [hq]

theorem num_map_testBit (v w : ℕ) : ReadUnit.num ((List.range w).map v.testBit) = v % 2 ^ w := by
  refine Nat.eq_of_testBit_eq fun q => ?_
  rw [testBit_num, Nat.testBit_mod_two_pow, List.getD_eq_getElem?_getD, List.getElem?_map]
  by_cases hq : q < w
  · rw [List.getElem?_range hq]
    simp [hq]
  · have : (List.range w)[q]? = none := List.getElem?_eq_none (by simpa using hq)
    rw [this]
    simp [hq]

theorem evs_append (z : Var → Bool) (a b : List Lin) : evs z (a ++ b) = evs z a ++ evs z b := List.map_append

/-- Zero padding leaves a number's bits' value alone. -/
theorem num_evs_pad (z : Var → Bool) (a b : List Lin) (w : ℕ) (hb : evs z b = (List.range w).map (Nat.testBit 0)) :
    ReadUnit.num (evs z (a ++ b)) = ReadUnit.num (evs z a) := by
  rw [evs_append, num_append, hb, num_map_testBit, Nat.zero_mod, Nat.mul_zero, Nat.add_zero]

theorem num_drop (l : List Bool) (j : ℕ) : ReadUnit.num (l.drop j) = ReadUnit.num l / 2 ^ j := by
  refine Nat.eq_of_testBit_eq fun q => ?_
  rw [Nat.testBit_div_two_pow, testBit_num, testBit_num, List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD,
    List.getElem?_drop, Nat.add_comm]

theorem num_evs_drop (z : Var → Bool) (l : List Lin) (j : ℕ) :
    ReadUnit.num (evs z (l.drop j)) = ReadUnit.num (evs z l) / 2 ^ j := by
  rw [← num_drop]
  simp only [evs, List.map_drop]

/-! ## A hidden `uint`'s bytes -/

theorem length_uintBytes (x : List Lin) (k : ℕ) : (uintBytes x k).length = k := by
  simp [uintBytes]

theorem bytes8_uintBytes (x : List Lin) (k : ℕ) : Bytes8 (uintBytes x k) := by
  intro c hc
  simp only [uintBytes, List.mem_map, List.mem_range] at hc
  obtain ⟨b, hb, rfl⟩ := hc
  simp only [List.length_take, List.length_drop, List.length_append, List.length_replicate]
  omega

theorem getD_uintPad (x : List Lin) (k t : ℕ) (ht : t < 8 * k) :
    (x.take (8 * k) ++ List.replicate (8 * k - min x.length (8 * k)) Lin.zero).getD t Lin.zero =
      x.getD t Lin.zero := by
  by_cases h : t < x.length
  · rw [getD_append_left' (l := x.take (8 * k)) (n := t) (by simp; omega), List.getD_eq_getElem?_getD,
      List.getD_eq_getElem?_getD, List.getElem?_take_of_lt ht]
  · have hx : x[t]? = none := List.getElem?_eq_none (by omega)
    rw [getD_append_right' (l := x.take (8 * k)) (n := t) (by simp; omega), List.getD_eq_getElem?_getD,
      List.getD_eq_getElem?_getD, hx, List.getElem?_replicate]
    split_ifs <;> rfl

/-- **A hidden `uint`'s bytes** (`frame_v3._uint_bytes`): the `k`-byte big-endian encoding of the number its bits
hold. -/
theorem bytesV_uintBytes (z : Var → Bool) (x : List Lin) (k : ℕ) :
    bytesV z (uintBytes x k) = beBytes k (ReadUnit.num (evs z x)) := by
  refine ext_getD 0 (by simp [bytesV, uintBytes, beBytes]) fun b hb => ?_
  have hb' : b < k := by simpa [bytesV, uintBytes] using hb
  have e : (uintBytes x k).getD b [] = ((x.take (8 * k) ++ List.replicate (8 * k - min x.length (8 * k))
      Lin.zero).drop (8 * (k - 1 - b))).take 8 := getD_map_range _ hb' _
  have hl8 : (((x.take (8 * k) ++ List.replicate (8 * k - min x.length (8 * k)) Lin.zero).drop
      (8 * (k - 1 - b))).take 8).length = 8 := by
    simp only [List.length_take, List.length_drop, List.length_append, List.length_replicate]
    omega
  refine uint8_ext fun q hq => ?_
  rw [testBit_beBytes k _ hb' hq, testBit_num_evs, getD_bytesV, e, testBit_byteV z hl8 hq,
    getD_take_drop _ _ _ _ _ hq, getD_uintPad x k (8 * (k - 1 - b) + q) (by omega)]

/-! ## The flags -/

/-- `Hash.uint`'s length: the fewest bytes, one for 0. -/
def uLen (v : ℕ) : ℕ := max 1 ((Nat.size v + 7) / 8)

theorem uint_eq (v : ℕ) : Hash.uint v = beBytes (uLen v) v := rfl

theorem uLen_pos (v : ℕ) : 1 ≤ uLen v := le_max_left _ _

theorem uLen_le {v n : ℕ} (hn : 1 ≤ n) (hv : v < 2 ^ (8 * n)) : uLen v ≤ n := by
  have := Nat.size_le.2 hv
  unfold uLen
  omega

theorem two_pow_le_iff_uLen {v j : ℕ} : 2 ^ (8 * (j + 1)) ≤ v ↔ j + 1 < uLen v := by
  rw [← Nat.lt_size]
  unfold uLen
  constructor <;> intro h <;> omega

theorem uint_small {k : ℕ} (hk : k < 256) : Hash.uint k = [k.toUInt8] := by
  have hs : Nat.size k ≤ 8 := Nat.size_le.2 (by simpa using hk)
  rw [uint_eq, (show uLen k = 1 by unfold uLen; omega)]
  show [(k / 256 ^ (1 - 1 - 0) % 256).toUInt8] = [k.toUInt8]
  simp [Nat.mod_eq_of_lt hk]

theorem anyFrom_eq_decide (z : Var → Bool) (x : List Lin) (lo : ℕ) :
    anyFrom z x lo x.length = decide (2 ^ lo ≤ ReadUnit.num (evs z x)) := by
  have hv := num_evs_lt z x
  rw [Bool.eq_iff_iff, anyFrom_eq_true, decide_eq_true_iff]
  constructor
  · rintro ⟨i, h1, -, h3⟩
    rw [← testBit_num_evs] at h3
    exact le_trans (Nat.pow_le_pow_right (by norm_num) h1) (Nat.ge_two_pow_of_testBit h3)
  · intro h
    by_contra hne
    have hlt : ReadUnit.num (evs z x) < 2 ^ lo := Nat.lt_pow_two_of_testBit _ fun i hi => by
      by_cases hix : i < x.length
      · rw [testBit_num_evs]
        cases he : ev z (x.getD i Lin.zero)
        · rfl
        · exact absurd ⟨i, hi, hix, he⟩ hne
      · exact Nat.testBit_lt_two_pow (lt_of_lt_of_le hv (Nat.pow_le_pow_right (by norm_num) (by omega)))
    omega

theorem map_range_decide (m L : ℕ) (h1 : 1 ≤ L) (h2 : L ≤ m + 1) :
    (List.range m).map (fun j => decide (j + 1 < L)) =
      List.replicate (L - 1) true ++ List.replicate (m + 1 - L) false := by
  have hlen : ((List.range m).map fun j => decide (j + 1 < L)).length =
      (List.replicate (L - 1) true ++ List.replicate (m + 1 - L) false).length := by
    simp only [List.length_map, List.length_range, List.length_append, List.length_replicate]
    omega
  refine List.ext_getElem hlen fun j _ _ => ?_
  rw [List.getElem_map, List.getElem_range]
  by_cases h : j < L - 1
  · rw [List.getElem_append_left (as := List.replicate (L - 1) true) (i := j) (by simpa using h), List.getElem_replicate]
    simpa using (by omega : j + 1 < L)
  · rw [List.getElem_append_right (as := List.replicate (L - 1) true) (i := j) (by simpa using h), List.getElem_replicate]
    simpa using (by omega : L ≤ j + 1)

/-- **The flags** over a hidden number of `n` bytes: the first `uLen v - 1` set, the rest clear. -/
theorem flags_eq (z : Var → Bool) (x : List Lin) (n : ℕ) (fl : List Lin) (hn : 1 ≤ n) (hx : x.length = 8 * n)
    (hfl : evs z fl = (List.range (n - 1)).map fun j => anyFrom z x (8 * (j + 1)) (8 * n)) :
    evs z fl = List.replicate (uLen (ReadUnit.num (evs z x)) - 1) true ++
      List.replicate (n - uLen (ReadUnit.num (evs z x))) false := by
  have hv : ReadUnit.num (evs z x) < 2 ^ (8 * n) := by rw [← hx]; exact num_evs_lt z x
  have hU := uLen_le hn hv
  have e := map_range_decide (n - 1) (uLen (ReadUnit.num (evs z x))) (uLen_pos _) (by omega)
  rw [show n - 1 + 1 - uLen (ReadUnit.num (evs z x)) = n - uLen (ReadUnit.num (evs z x)) by omega] at e
  rw [hfl, ← e]
  refine List.map_congr_left fun j _ => ?_
  rw [← hx, anyFrom_eq_decide]
  exact decide_eq_decide.2 two_pow_le_iff_uLen

/-! ## The selection -/

theorem sound_select (fl : List Lin) (layouts : List (List Word)) : Sound (Frame.select fl layouts) fun z out =>
    out.length = (layouts.headD []).length ∧ ∀ i < out.length, (out.getD i []).length = 64 ∧
      ∀ j < 64, ∀ k s : ℕ, evs z fl = List.replicate k true ++ List.replicate s false → k < layouts.length →
        ev z ((out.getD i []).getD j Lin.zero) = ev z (((layouts.getD k []).getD i []).getD j Lin.zero) := by
  unfold Frame.select
  refine Sound.mono (sound_mapM_getD [] _ (fun z i w => w.length = 64 ∧ ∀ j < 64, ∀ k s : ℕ,
      evs z fl = List.replicate k true ++ List.replicate s false → k < layouts.length →
        ev z (w.getD j Lin.zero) = ev z (((layouts.getD k []).getD i []).getD j Lin.zero)) (fun i => ?_) _) ?_
  · refine Sound.mono (sound_mapM_getD Lin.zero _ (fun z j r => ∀ k s : ℕ,
        evs z fl = List.replicate k true ++ List.replicate s false → k < layouts.length →
          ev z r = ev z (((layouts.getD k []).getD i []).getD j Lin.zero)) (fun j => ?_) _) ?_
    · refine Sound.mono (sound_selectBit fl _) fun z r h k s hfl hk => ?_
      rw [h k s hfl (by simpa using hk), getD_map' _ layouts k hk []]
    · intro z rs h
      obtain ⟨hlen, hall⟩ := h
      refine ⟨by simpa using hlen, fun j hj k s hfl hk => ?_⟩
      have := hall j (by simpa using hj) k s hfl hk
      simpa using this
  · intro z out h
    obtain ⟨hlen, hall⟩ := h
    refine ⟨by simpa using hlen, fun i hi => ?_⟩
    have := hall i (by rw [← hlen]; exact hi)
    simpa using this

/-! ## A message over a hidden `uint` -/

/-- **A message over a hidden `uint`** (`frame_v3._hash_uint`): when every length's layout has the same number of
words, the net's digest is SHA-512 of the message at the `uint`'s own length. -/
theorem sound_hashUint (tag : List UInt8) (head : List (List Byte)) (x : List Lin) (n : ℕ)
    (tail : List Byte → List (List Byte)) : Sound (hashUint tag head x n tail) fun z out =>
      1 ≤ n → x.length = 8 * n → (∀ L, ∀ p ∈ head ++ tail (uintBytes x L), Bytes8 p) →
      (∀ L < n, (message tag (head ++ tail (uintBytes x (L + 1)))).length =
        (message tag (head ++ tail (uintBytes x 1))).length) →
      Shaped 8 out ∧ wordsDigest z out =
        Verity.sha512 (bytesV z (msgData tag (head ++ tail (uintBytes x (uLen (ReadUnit.num (evs z x))))))) := by
  simp only [hashUint]
  refine Sound.bind' (sound_flags x n) fun fl => Sound.bind' (sound_select fl _) fun ws =>
    Sound.mono (sound_sha512W ws) fun z out hsha hsel hfl hn hx h8 hML => ?_
  have hv : ReadUnit.num (evs z x) < 2 ^ (8 * n) := by rw [← hx]; exact num_evs_lt z x
  have hU1 := uLen_pos (ReadUnit.num (evs z x))
  have hUn := uLen_le hn hv
  have hfl' := flags_eq z x n fl hn hx (hfl hn)
  obtain ⟨hwl, hws⟩ := hsel
  have hk : uLen (ReadUnit.num (evs z x)) - 1 < n := by omega
  have hU : uLen (ReadUnit.num (evs z x)) - 1 + 1 = uLen (ReadUnit.num (evs z x)) := by omega
  have hhead : ((List.range n).map fun L => message tag (head ++ tail (uintBytes x (L + 1)))).headD [] =
      message tag (head ++ tail (uintBytes x 1)) := by
    obtain ⟨n', rfl⟩ : ∃ n', n = n' + 1 := ⟨n - 1, by omega⟩
    rw [List.range_succ_eq_map, List.map_cons, List.headD_cons, Nat.zero_add]
  have hlay : ((List.range n).map fun L => message tag (head ++ tail (uintBytes x (L + 1)))).getD
      (uLen (ReadUnit.num (evs z x)) - 1) [] =
        message tag (head ++ tail (uintBytes x (uLen (ReadUnit.num (evs z x))))) := by
    rw [getD_map_range _ hk, hU]
  have hML' := hML _ hk
  rw [hU] at hML'
  have hwl' : ws.length =
      (message tag (head ++ tail (uintBytes x (uLen (ReadUnit.num (evs z x)))))).length := by
    rw [hwl, hhead, hML']
  obtain ⟨-, hl64, hb64⟩ := holds_message z tag (h8 (uLen (ReadUnit.num (evs z x))))
  refine hsha _ ⟨fun w hw => ?_, ?_, fun i hi p hp => ?_⟩
  · obtain ⟨i, hi, rfl⟩ := List.getElem_of_mem hw
    have := (hws i hi).1
    rwa [getD_of_lt ws i hi] at this
  · rw [hwl', hl64, length_bytesV]
  · rw [(hws i hi).2 p hp _ _ hfl' (by simpa using hk), hlay, length_bytesV]
    exact hb64 i (hwl' ▸ hi) p hp

/-! ## The leaf -/

/-- A string's UTF-8 byte count from its characters: a literal's then reduces by `simp`, where `decide` can't see
`String.toUTF8`'s body from a module. -/
theorem utf8ByteSize_eq_sum (s : String) : s.utf8ByteSize = (s.toList.map Char.utf8Size).sum := by
  rw [← String.ofList_toList (s := s)]
  generalize s.toList = l
  induction l with
  | nil => simp
  | cons c l ih =>
    rw [String.ofList_cons, String.utf8ByteSize_append, String.utf8ByteSize_singleton, ih]
    simp

theorem length_utf8 (s : String) : s.toUTF8.data.toList.length = (s.toList.map Char.utf8Size).sum := by
  rw [Array.length_toList, ByteArray.size_data, String.toUTF8_eq_toByteArray, String.size_toByteArray,
    utf8ByteSize_eq_sum]

theorem FRAME_length : FRAME.length = 33 := by
  rw [FRAME, List.length_append, length_utf8]
  simp [Char.utf8Size]

theorem LEAF_eq : LEAF = Hash.utf8 "leaf" := rfl

theorem NODE_eq : NODE = Hash.utf8 "node" := rfl

theorem LEAF_length : LEAF.length = 4 := by
  rw [LEAF, length_utf8]
  simp [Char.utf8Size]

theorem NODE_length : NODE.length = 4 := by
  rw [NODE, length_utf8]
  simp [Char.utf8Size]

theorem bytesV_single (z : Var → Bool) (b : Byte) : bytesV z [b] = [byteV z b] := rfl

theorem length_message_leaf {domain value : List Word} (hd : domain.length = 8) (hv : value.length = 8)
    (x : List Lin) (L : ℕ) (S : List UInt8) (hL : S.length + 2 * L ≤ 30) :
    (message LEAF ([bytesW domain] ++ [uintBytes x L, uintBytes x L, constBytes S, bytesW value])).length = 32 := by
  rw [length_message, length_msgData]
  simp only [List.cons_append, List.nil_append, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil,
    length_uintBytes, length_bytesW, hd, hv, length_constBytes, LEAF_length, FRAME_length]
  omega

theorem bytes8_leafParts {domain value : List Word} (hd : Words64 domain) (hv : Words64 value) (x : List Lin)
    (S : List UInt8) (L : ℕ) :
    ∀ p ∈ [bytesW domain] ++ [uintBytes x L, uintBytes x L, constBytes S, bytesW value], Bytes8 p := by
  intro p hp
  simp only [List.cons_append, List.nil_append, List.mem_cons, List.not_mem_nil, or_false] at hp
  rcases hp with rfl | rfl | rfl | rfl | rfl
  · exact bytes8_bytesW hd
  · exact bytes8_uintBytes x L
  · exact bytes8_uintBytes x L
  · exact bytes8_constBytes S
  · exact bytes8_bytesW hv

/-- **The leaf** (`frame_v3.leaf_hash`): SHA-512 of the leaf's frame at rank = position = the number `x`'s bits
hold. -/
theorem sound_leafHash (schema : String) (n : ℕ) (domain value : List Word) (x : List Lin) :
    Sound (leafHash schema n domain value x) fun z out => 1 ≤ n → x.length = 8 * n →
      (Hash.utf8 schema).length + 2 * n ≤ 30 → Shaped 8 domain → Shaped 8 value →
      Shaped 8 out ∧ wordsDigest z out = Verity.sha512 (Hash.Tree.frame "leaf" [(wordsDigest z domain).bytes,
        Hash.uint (ReadUnit.num (evs z x)), Hash.uint (ReadUnit.num (evs z x)), Hash.utf8 schema,
        (wordsDigest z value).bytes]) := by
  unfold leafHash
  refine Sound.mono (sound_hashUint _ _ x n _) fun z out h hn hx hS hd hv => ?_
  have hS' : schema.toUTF8.data.toList.length + 2 * n ≤ 30 := hS
  obtain ⟨hs, he⟩ := h hn hx (fun L => bytes8_leafParts hd.2 hv.2 x schema.toUTF8.data.toList L)
    fun L hL => (length_message_leaf hd.1 hv.1 x (L + 1) schema.toUTF8.data.toList (by omega)).trans
      (length_message_leaf hd.1 hv.1 x 1 schema.toUTF8.data.toList (by omega)).symm
  refine ⟨hs, he.trans ?_⟩
  rw [bytesV_msgData, ← frame_eq]
  simp only [List.cons_append, List.nil_append, List.map_cons, List.map_nil, bytesV_bytesW z hd.2,
    bytesV_bytesW z hv.2, bytesV_uintBytes, bytesV_constBytes, ← bytes_wordsDigest z hd,
    ← bytes_wordsDigest z hv, uint_eq, LEAF_eq]
  rfl

/-! ## The swap and the node -/

theorem sound_swap (d : Lin) (cur sib : List Word) : Sound ((cur.zip sib).mapM fun (cs : Word × Word) =>
    (cs.1.zip cs.2).mapM fun (xy : Lin × Lin) => Flock.HmNets.and d (xy.1 ^^^ xy.2)) fun z t =>
      t.map (evs z) = (cur.zip sib).map fun (cs : Word × Word) =>
        (cs.1.zip cs.2).map fun (xy : Lin × Lin) => (ev z d && (ev z xy.1 ^^ ev z xy.2)) :=
  sound_mapM_map [] _ evs (fun z (cs : Word × Word) =>
      (cs.1.zip cs.2).map fun (xy : Lin × Lin) => (ev z d && (ev z xy.1 ^^ ev z xy.2)))
    (fun cs => sound_mapM_map Lin.zero _ ev (fun z (xy : Lin × Lin) => (ev z d && (ev z xy.1 ^^ ev z xy.2)))
      (fun xy => Sound.mono (sound_and d _) fun z r h => by rw [h, ev_xor]) _) _

theorem swap_bits (z : Var → Bool) (dv : Bool) {cur sib t : List Word} (hc : Shaped 8 cur) (hs : Shaped 8 sib)
    (ht : t.map (evs z) = (cur.zip sib).map fun cs =>
      (cs.1.zip cs.2).map fun xy => (dv && (ev z xy.1 ^^ ev z xy.2))) :
    t.length = 8 ∧ ∀ i < 8, (t.getD i []).length = 64 ∧ ∀ j < 64, ev z ((t.getD i []).getD j Lin.zero) =
      (dv && (ev z ((cur.getD i []).getD j Lin.zero) ^^ ev z ((sib.getD i []).getD j Lin.zero))) := by
  have hl : t.length = 8 := by
    have := congrArg List.length ht
    simp only [List.length_map, List.length_zip, hc.1, hs.1, Nat.min_self] at this
    exact this
  refine ⟨hl, fun i hi => ?_⟩
  have hci := shaped_getD hc hi
  have hsi := shaped_getD hs hi
  have e := congrArg (fun L => L.getD i []) ht
  try simp only at e
  rw [getD_map' (evs z) t i (by omega) [], getD_map' _ (cur.zip sib) i
      (by rw [List.length_zip, hc.1, hs.1]; omega) ([], []),
    getD_zip cur sib i (by rw [hc.1]; exact hi) (by rw [hs.1]; exact hi) [] [] ([], [])] at e
  try dsimp only at e
  have hti : (t.getD i []).length = 64 := by
    have := congrArg List.length e
    simp only [evs, List.length_map, List.length_zip, hci, hsi, Nat.min_self] at this
    exact this
  refine ⟨hti, fun j hj => ?_⟩
  have e' := congrArg (fun L => L.getD j false) e
  try simp only at e'
  unfold evs at e'
  rw [getD_map' (ev z) (t.getD i []) j (by omega) Lin.zero, getD_map' _ ((cur.getD i []).zip (sib.getD i [])) j
      (by rw [List.length_zip, hci, hsi]; omega) (Lin.zero, Lin.zero),
    getD_zip (cur.getD i []) (sib.getD i []) j (by omega) (by omega) Lin.zero Lin.zero (Lin.zero, Lin.zero)] at e'
  exact e'

theorem bits_xorW_zip (z : Var → Bool) {a b : List Word} (ha : Shaped 8 a) (hb : Shaped 8 b) :
    Shaped 8 ((a.zip b).map fun ab => xorW ab.1 ab.2) ∧ ∀ i < 8, ∀ j < 64,
      ev z ((((a.zip b).map fun ab => xorW ab.1 ab.2).getD i []).getD j Lin.zero) =
        (ev z ((a.getD i []).getD j Lin.zero) ^^ ev z ((b.getD i []).getD j Lin.zero)) := by
  have hl : ((a.zip b).map fun ab => xorW ab.1 ab.2).length = 8 := by
    simp only [List.length_map, List.length_zip, ha.1, hb.1, Nat.min_self]
  have key : ∀ i < 8, ((a.zip b).map fun ab => xorW ab.1 ab.2).getD i [] = xorW (a.getD i []) (b.getD i []) := by
    intro i hi
    rw [getD_map' _ (a.zip b) i (by rw [List.length_zip, ha.1, hb.1]; omega) ([], []),
      getD_zip a b i (by rw [ha.1]; exact hi) (by rw [hb.1]; exact hi) [] [] ([], [])]
  refine ⟨shaped_of_getD hl fun i hi => ?_, fun i hi j hj => ?_⟩
  · rw [key i hi, xorW, List.length_zipWith, shaped_getD ha hi, shaped_getD hb hi, Nat.min_self]
  · rw [key i hi, xorW, getD_zipWith' _ (a.getD i []) (b.getD i []) j (by rw [shaped_getD ha hi]; exact hj)
      (by rw [shaped_getD hb hi]; exact hj) Lin.zero Lin.zero Lin.zero, ev_xor]

theorem length_message_node {domain left right : List Word} (hd : domain.length = 8) (hl : left.length = 8)
    (hr : right.length = 8) (level : Byte) (x : List Lin) (L : ℕ) (hL : L ≤ 93) :
    (message NODE ([bytesW domain, [level]] ++ [uintBytes x L, bytesW left, bytesW right])).length = 48 := by
  rw [length_message, length_msgData]
  simp only [List.cons_append, List.nil_append, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil,
    length_uintBytes, length_bytesW, hd, hl, hr, NODE_length, FRAME_length, List.length_cons, List.length_nil]
  omega

theorem bytes8_nodeParts {domain left right : List Word} {level : Byte} (hd : Words64 domain)
    (hlv : level.length = 8) (hl : Words64 left) (hr : Words64 right) (x : List Lin) (L : ℕ) :
    ∀ p ∈ [bytesW domain, [level]] ++ [uintBytes x L, bytesW left, bytesW right], Bytes8 p := by
  intro p hp
  simp only [List.cons_append, List.nil_append, List.mem_cons, List.not_mem_nil, or_false] at hp
  rcases hp with rfl | rfl | rfl | rfl | rfl
  · exact bytes8_bytesW hd
  · intro b hb
    rw [List.mem_singleton] at hb
    rw [hb]
    exact hlv
  · exact bytes8_uintBytes x L
  · exact bytes8_bytesW hl
  · exact bytes8_bytesW hr

/-- The swap's two outputs: `cur ⊕ t` and `sib ⊕ t` hold `(sib, cur)` when the bit is set and `(cur, sib)` when it
is clear. -/
theorem swap_digest (z : Var → Bool) (dv : Bool) {a b t : List Word} (ha : Shaped 8 a) (hb : Shaped 8 b)
    (hts : Shaped 8 t) (htb : ∀ i < 8, ∀ j < 64, ev z ((t.getD i []).getD j Lin.zero) =
      (dv && (ev z ((a.getD i []).getD j Lin.zero) ^^ ev z ((b.getD i []).getD j Lin.zero)))) :
    Shaped 8 ((a.zip t).map fun ct => xorW ct.1 ct.2) ∧
      wordsDigest z ((a.zip t).map fun ct => xorW ct.1 ct.2) = if dv then wordsDigest z b else wordsDigest z a := by
  obtain ⟨hls, hlb⟩ := bits_xorW_zip z ha hts
  refine ⟨hls, ?_⟩
  cases dv
  · rw [ite_eq_right Bool.false_ne_true]
    refine wordsDigest_congr z (map_evs_ext z hls ha fun i hi j hj => ?_)
    rw [hlb i hi j hj, htb i hi j hj]
    generalize ev z ((a.getD i []).getD j Lin.zero) = u
    generalize ev z ((b.getD i []).getD j Lin.zero) = w
    cases u <;> cases w <;> rfl
  · rw [ite_eq_left rfl]
    refine wordsDigest_congr z (map_evs_ext z hls hb fun i hi j hj => ?_)
    rw [hlb i hi j hj, htb i hi j hj]
    generalize ev z ((a.getD i []).getD j Lin.zero) = u
    generalize ev z ((b.getD i []).getD j Lin.zero) = w
    cases u <;> cases w <;> rfl

/-- **A node** (`frame_v3.node_hash`): SHA-512 of the node's frame over the swapped children, at the level byte and
the index the bits of `index` hold. -/
theorem sound_nodeHash (n : ℕ) (domain : List Word) (level : Byte) (index : List Lin) (d : Lin)
    (cur sib : List Word) : Sound (nodeHash n domain level index d cur sib) fun z out => 1 ≤ n →
      index.length = 8 * n → level.length = 8 → n ≤ 93 → Shaped 8 domain → Shaped 8 cur → Shaped 8 sib →
      Shaped 8 out ∧ wordsDigest z out = Verity.sha512 (Hash.Tree.frame "node" [(wordsDigest z domain).bytes,
        [byteV z level], Hash.uint (ReadUnit.num (evs z index)),
        (if ev z d then wordsDigest z sib else wordsDigest z cur).bytes,
        (if ev z d then wordsDigest z cur else wordsDigest z sib).bytes]) := by
  simp only [nodeHash]
  refine Sound.bind' (sound_swap d cur sib) fun t =>
    Sound.mono (sound_hashUint _ _ index n _) fun z out h ht hn hx hlv hn93 hd hc hs => ?_
  obtain ⟨htl, htb⟩ := swap_bits z (ev z d) hc hs ht
  have hts : Shaped 8 t := shaped_of_getD htl fun i hi => (htb i hi).1
  have htb' := fun i hi => (htb i hi).2
  obtain ⟨hls, hL⟩ := swap_digest z (ev z d) hc hs hts htb'
  obtain ⟨hrs, hR⟩ := swap_digest z (ev z d) hs hc hts fun i hi j hj => by
    rw [htb' i hi j hj, Bool.xor_comm]
  obtain ⟨hs', he⟩ := h hn hx (fun L => bytes8_nodeParts hd.2 hlv hls.2 hrs.2 index L)
    fun L hL => (length_message_node hd.1 hls.1 hrs.1 level index (L + 1) (by omega)).trans
      (length_message_node hd.1 hls.1 hrs.1 level index 1 (by omega)).symm
  refine ⟨hs', he.trans ?_⟩
  rw [bytesV_msgData, ← frame_eq]
  simp only [List.cons_append, List.nil_append, List.map_cons, List.map_nil, bytesV_bytesW z hd.2,
    bytesV_bytesW z hls.2, bytesV_bytesW z hrs.2, bytesV_uintBytes, bytesV_single, ← bytes_wordsDigest z hd,
    ← bytes_wordsDigest z hls, ← bytes_wordsDigest z hrs, hL, hR, uint_eq, NODE_eq]

/-! ## The climb -/

theorem climb_cons_snd (H : Hash.Msg → Verity.Digest) (D : Hash.Tree.Domain) (lv cur : ℕ) (x s : Verity.Digest)
    (ss : List Verity.Digest) : (Hash.Tree.climb H D lv cur x (s :: ss)).2 =
      (Hash.Tree.climb H D (lv + 1) (cur / 2) (H (if cur % 2 = 0 then Hash.Tree.nodeMsg D lv (cur / 2) x s
        else Hash.Tree.nodeMsg D lv (cur / 2) s x)) ss).2 := rfl

/-- **The climb** (`frame_v3._body`'s levels from `k`): the net's digest is `Hash.Tree.climb`'s from level `k`, at the
index `pos`'s bits from `k` up hold. -/
theorem sound_climb (Dd : ℕ) (domain : List Word) (pos : List Lin) :
    ∀ (ss : List (List Word)) (k : ℕ) (c : Consts) (cur : List Word),
      Sound (climbWith constForm nodeHash domain pos k ss c cur) fun z out => ConstsOk z c →
        k + ss.length = pos.length → pos.length ≤ 32 → Shaped 8 domain → Shaped 8 cur →
        (∀ s ∈ ss, Shaped 8 s) → Shaped 8 out ∧ wordsDigest z out =
          (Hash.Tree.climb Verity.sha512 (denseDomain Dd (wordsDigest z domain)) k
            (ReadUnit.num (evs z pos) / 2 ^ k) (wordsDigest z cur) (ss.map (wordsDigest z))).2
  | [], k, c, cur => by
    simp only [climbWith]
    exact Sound.pure' fun _ _ _ _ _ hc _ => ⟨hc, rfl⟩
  | s :: ss, k, c, cur => by
    simp only [climbWith]
    refine Sound.bind' (sound_constBits c 0 _) fun pad => Sound.bind' (sound_constBits pad.1 k 8) fun lv =>
      Sound.bind' (sound_nodeHash _ domain lv.2 _ (pos.getD k Lin.zero) cur s) fun cur' =>
        Sound.mono (sound_climb Dd domain pos ss (k + 1) lv.1 cur')
          fun z out hup hnode hlv hpad hcOk hk h32 hd hcur hss => ?_
    simp only [List.length_cons] at hk
    obtain ⟨hc1, hpe, hpl⟩ := hpad hcOk
    obtain ⟨hc2, hle, hll⟩ := hlv hc1
    have hhil : (pos.drop (k + 1)).length = pos.length - (k + 1) := by simp only [List.length_drop]
    have hn1 : 1 ≤ indexBytes (pos.drop (k + 1)).length := le_max_left _ _
    have hn4 : indexBytes (pos.drop (k + 1)).length ≤ 4 := by
      unfold indexBytes
      rw [hhil]
      omega
    have hix : (pos.drop (k + 1) ++ pad.2).length = 8 * indexBytes (pos.drop (k + 1)).length := by
      rw [List.length_append, hpl]
      unfold indexBytes
      omega
    have hs : Shaped 8 s := hss s (List.mem_cons.2 (Or.inl rfl))
    have hss' : ∀ s' ∈ ss, Shaped 8 s' := fun s' h => hss s' (List.mem_cons.2 (Or.inr h))
    obtain ⟨hcs, hce⟩ := hnode hn1 hix hll (by omega) hd hcur hs
    obtain ⟨hos, hoe⟩ := hup hc2 (by omega) h32 hd hcs hss'
    refine ⟨hos, hoe.trans ?_⟩
    have hlvb : byteV z lv.2 = k.toUInt8 := by
      rw [byteV, hle, num_map_testBit, Nat.mod_eq_of_lt (show k < 2 ^ 8 by norm_num; omega)]
    have hlvu : [byteV z lv.2] = Hash.uint k := by rw [hlvb, uint_small (k := k) (by omega)]
    have hnum : ReadUnit.num (evs z (pos.drop (k + 1) ++ pad.2)) = ReadUnit.num (evs z pos) / 2 ^ k / 2 := by
      rw [num_evs_pad z _ _ _ hpe, num_evs_drop, Nat.div_div_eq_div_mul, ← Nat.pow_succ]
    have hidx : ReadUnit.num (evs z pos) / 2 ^ (k + 1) = ReadUnit.num (evs z pos) / 2 ^ k / 2 := by
      rw [Nat.div_div_eq_div_mul, ← Nat.pow_succ]
    have hdv : ev z (pos.getD k Lin.zero) = decide (ReadUnit.num (evs z pos) / 2 ^ k % 2 = 1) := by
      rw [← testBit_num_evs, Nat.testBit_eq_decide_div_mod_eq]
    rw [hce, List.map_cons, climb_cons_snd, hidx, hlvu, hnum, hdv]
    by_cases h2 : ReadUnit.num (evs z pos) / 2 ^ k % 2 = 0
    · rw [ite_eq_left h2, decide_eq_false (show ¬(ReadUnit.num (evs z pos) / 2 ^ k % 2 = 1) by omega),
        ite_eq_right Bool.false_ne_true, ite_eq_right Bool.false_ne_true]
      all_goals rfl
    · rw [ite_eq_right h2, decide_eq_true (show ReadUnit.num (evs z pos) / 2 ^ k % 2 = 1 by omega), ite_eq_left rfl,
        ite_eq_left rfl]
      all_goals rfl

end Accounting.ReadUnit.FrameRead
