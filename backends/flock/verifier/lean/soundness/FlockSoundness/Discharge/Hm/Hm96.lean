/-
What the hm96 gadget's rows force (`Flock.HmNets.hm96Row`): its `b ‖ c` is the padding compression's digest XOR `M(key)·y`,
then SHA-512's last two compressions of `salt_prefix ‖ y`, bit by bit in byte-string order.
-/
import FlockSoundness.Discharge.Hm.Compress
import FlockSoundness.Discharge.Hm.Exec

namespace FlockSoundness.Discharge.Hm

open Flock.HmNets hiding and round Regs

theorem length_compressV {cv : List (BitVec 64)} (h : cv.length = 8) (msg : List (BitVec 64)) :
    (compressV cv msg).length = 8 := by
  simp [compressV, h, Regs.toList]

theorem length_of_map_bv {z : ℕ → Bool} {ws : List Word} {vs : List (BitVec 64)} (h : ws.map (bv z) = vs) :
    ws.length = vs.length := by
  rw [← h, List.length_map]

theorem length_words (bits : List Lin) (n : ℕ) : (words bits n).length = n := by simp [words]

theorem words64_words {bits : List Lin} {n : ℕ} (h : 64 * n ≤ bits.length) : Words64 (words bits n) := by
  intro w hw
  simp only [words, List.mem_map, List.mem_range] at hw
  obtain ⟨i, hi, rfl⟩ := hw
  simp only [List.length_take, List.length_drop]
  omega

theorem Words64.append {a b : List Word} (ha : Words64 a) (hb : Words64 b) : Words64 (a ++ b) := fun x hx => by
  rcases List.mem_append.1 hx with h | h
  · exact ha x h
  · exact hb x h

theorem words64_constW (ks : List UInt64) : Words64 (ks.map constW) := by
  intro w hw
  simp only [List.mem_map] at hw
  obtain ⟨k, _, rfl⟩ := hw
  exact length_constW k

theorem map_bv_constW (z : ℕ → Bool) (ks : List UInt64) : (ks.map constW).map (bv z) = ks.map UInt64.toBitVec := by
  simp [List.map_map, Function.comp_def, bv_constW]

/-- A bit of concatenated 64-bit words is its word's bit. -/
theorem bit_flatten (z : ℕ → Bool) : ∀ {ws : List Word}, Words64 ws → ∀ p : ℕ,
    bit z ws.flatten p = (bv z (ws.getD (p / 64) [])).getLsbD (p % 64)
  | [], _, p => by simp [bit, bv_nil]
  | w :: ws, h, p => by
      have hw : w.length = 64 := h w List.mem_cons_self
      have hws : Words64 ws := fun x hx => h x (List.mem_cons_of_mem _ hx)
      rw [List.flatten_cons]
      by_cases hp : p < 64
      · rw [show p / 64 = 0 by omega, show p % 64 = p by omega, List.getD_cons_zero, ← bit_eq_getLsbD (by omega)]
        simp only [bit]
        rw [List.getElem?_append_left (by omega)]
      · have ih := bit_flatten z hws (p - 64)
        rw [show p / 64 = (p - 64) / 64 + 1 by omega, show p % 64 = (p - 64) % 64 by omega, List.getD_cons_succ, ← ih]
        simp only [bit]
        rw [List.getElem?_append_right (by omega), hw]

/-- A bit of words' concatenation, read through an array, is the words' bit `p`. -/
theorem ev_flatten_toArray (z : ℕ → Bool) {ws : List Word} (h : Words64 ws) (p : ℕ) :
    ev z ws.flatten.toArray[p]! = wbit (ws.map (bv z)) p := by
  rw [wbit, getD_map_bv, ← bit_flatten z h]
  simp only [bit, getElem!_def, List.getElem?_toArray]
  cases ws.flatten[p]? <;> rfl

theorem ev_toArray (z : ℕ → Bool) (l : List Lin) (j : ℕ) : ev z l.toArray[j]! = bit z l j := by
  simp only [bit, getElem!_def, List.getElem?_toArray]
  cases l[j]? <;> rfl

/-! ## The salt in word order -/

theorem wordBit_wordBit (t : ℕ) : wordBit (wordBit t) = t := by unfold wordBit; omega

theorem wordBit_lt {t n : ℕ} (h : t < 64 * n) : wordBit t < 64 * n := by unfold wordBit; omega

/-- The salt's bits in word order: position `p` holds the salt's byte-string bit `wordBit p`. -/
def ywL (y : Array Lin) : List Lin := (List.range 1536).map fun p => y[wordBit p]!

theorem yw_step (y : Array Lin) : ∀ n ≤ 1536,
    ((List.range n).foldl (fun (acc : Array Lin) t => acc.set! (wordBit t) y[t]!) (Array.replicate 1536 zero)).toList =
      (List.range 1536).map fun p => if wordBit p < n then y[wordBit p]! else zero
  | 0, _ => by
      simp only [List.range_zero, List.foldl_nil, Array.toList_replicate, Nat.not_lt_zero, ↓reduceIte]
      rw [List.map_const', List.length_range]
  | n + 1, hn => by
      rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil, Array.set!, Array.toList_setIfInBounds,
        yw_step y n (by omega)]
      apply List.ext_getElem (by simp)
      intro p h1 h2
      simp only [List.length_map, List.length_range] at h2
      rw [List.getElem_set, List.getElem_map, List.getElem_map, List.getElem_range]
      by_cases hp : wordBit n = p
      · subst hp
        simp only [↓reduceIte, wordBit_wordBit, show n < n + 1 by omega]
      · rw [ite_eq_right_iff.mpr fun h => absurd h hp]
        have : wordBit p ≠ n := fun h => hp (by rw [← h, wordBit_wordBit])
        by_cases hlt : wordBit p < n
        · simp only [hlt, ↓reduceIte, show wordBit p < n + 1 by omega]
        · simp only [hlt, ↓reduceIte, show ¬ wordBit p < n + 1 by omega]

theorem yw_toList (y : Array Lin) :
    ((List.range 1536).foldl (fun (acc : Array Lin) t => acc.set! (wordBit t) y[t]!) (Array.replicate 1536 zero)).toList =
      ywL y := by
  rw [yw_step y 1536 le_rfl, ywL]
  apply List.map_congr_left
  intro p hp
  simp only [wordBit_lt (n := 24) (by simpa using hp), ↓reduceIte]

theorem length_ywL (y : Array Lin) : (ywL y).length = 1536 := by simp [ywL]

/-! ## `M(key)·y` -/

theorem foldr_pairs (z : ℕ → Bool) (l : List Lin) :
    (pairs l).foldr (fun f b => ev z f ^^ b) false = l.foldr (fun f b => ev z f ^^ b) false := by
  fun_induction pairs l with
  | case1 a b rest ih => simp only [List.foldr_cons, ih, ev_xor, Bool.xor_assoc]
  | case2 rest h => rfl

theorem ev_xorAll (z : ℕ → Bool) (l : List Lin) : ev z (xorAll l) = l.foldr (fun f b => ev z f ^^ b) false := by
  fun_induction xorAll l with
  | case1 => rfl
  | case2 a => simp
  | case3 a b rest ih => rw [ih, foldr_pairs]

theorem foldl_xor_foldr (g : ℕ → Bool) : ∀ (l : List ℕ) (a : Bool),
    l.foldl (fun acc j => acc ^^ g j) a = (a ^^ l.foldr (fun j b => g j ^^ b) false)
  | [], a => by simp
  | j :: l, a => by
      simp only [List.foldl_cons, List.foldr_cons, foldl_xor_foldr g l, Bool.xor_assoc]

theorem ev_mask (z : ℕ → Bool) (key : ByteArray) (y : Array Lin) (i : ℕ) :
    ev z (xorAll ((List.range 1536).filterMap fun j => if Flock.Hm96.bit key (i + j) then some y[j]! else none)) =
      maskV key (fun j => ev z y[j]!) i := by
  rw [ev_xorAll, maskV, foldl_xor_foldr, Bool.false_xor]
  generalize List.range 1536 = l
  induction l with
  | nil => rfl
  | cons j l ih =>
      rw [List.filterMap_cons, List.foldr_cons]
      split <;> simp_all

/-! ## The row -/

/-- `Flock.HmNets.hm96Row` after its three `input`s (`hm96Row_eq`): the row's chaining value `cv`, its padding block
`pad` and its salt `y`. -/
def hm96Body (cv pad : List Lin) (y : Array Lin) : M (List Lin) := do
  let x := (← compress (words cv 8) (words pad 16)).flatten.toArray
  let yw := (List.range 1536).foldl (fun (acc : Array Lin) t => acc.set! (wordBit t) y[t]!) (Array.replicate 1536 zero)
  let mid := (Flock.Sha512.compress Flock.Sha512.iv Flock.Hm96.Default512.saltPrefix 0).toList
  let st ← compress (mid.map constW) (words yw.toList 16)
  let tail := (wordsBE (ByteArray.mk (Array.replicate 64 0) ++ (padTail (128 + 192)).extract 0 64)).drop 8
  let st ← compress st (words (yw.toList.drop 1024) 8 ++ tail.map constW)
  let cbits := st.flatten.toArray
  let key := Flock.Hm96.Default512.key
  let b := (List.range 512).map fun i =>
    x[wordBit i]! ^^^ xorAll ((List.range 1536).filterMap fun j => if Flock.Hm96.bit key (i + j) then some y[j]! else none)
  pure (b ++ (List.range 512).map fun t => cbits[wordBit t]!)

theorem hm96Row_eq : hm96Row = (do
    let cv ← input 512
    let pad ← input 1024
    let y ← input 1536
    hm96Body cv pad y.toArray) := rfl

/-- SHA-512's padding of a 320-byte message, the salt digest's last 8 words. -/
theorem tail_eq : (wordsBE (ByteArray.mk (Array.replicate 64 0) ++ (padTail (128 + 192)).extract 0 64)).drop 8 =
    [0x8000000000000000, 0, 0, 0, 0, 0, 0, 2560] := by decide +kernel

/-- The chaining value after `hm96-sha512/v1`'s salt prefix (one block). -/
def saltMid : List (BitVec 64) :=
  (Flock.Sha512.compress Flock.Sha512.iv Flock.Hm96.Default512.saltPrefix 0).toList.map UInt64.toBitVec

/-- The salt digest's last 8 words: SHA-512's padding of a 320-byte message. -/
def saltTail : List (BitVec 64) := [0x8000000000000000#64, 0#64, 0#64, 0#64, 0#64, 0#64, 0#64, 2560#64]

theorem length_saltMid : saltMid.length = 8 := by
  rw [show saltMid = toBV _ from rfl, compress_eq, length_compressV (by simp)]

/-- **The hm96 gadget computes `b ‖ c`**: whenever its rows hold, its output's bit `t < 512` is bit `t` (byte-string
order) of the padding compression `x = F(cv, pad)` XOR bit `t` of `M(key)·y`, and its bit `512 + t` is bit `t` of
`c = F(F(mid, y₀), y₁ ‖ tail)`, SHA-512's last two compressions of `salt_prefix ‖ y`, with `y` in word order. -/
theorem sound_hm96Body (cv pad : List Lin) (y : Array Lin) : Sound (hm96Body cv pad y) fun z out =>
    cv.length = 512 → pad.length = 1024 → out.length = 1024 ∧ ∀ t < 512,
      bit z out t = (wbit (compressV ((words cv 8).map (bv z)) ((words pad 16).map (bv z))) (wordBit t) ^^
        maskV Flock.Hm96.Default512.key (fun j => ev z y[j]!) t) ∧
      bit z out (512 + t) = wbit (compressV (compressV saltMid ((words (ywL y) 16).map (bv z)))
        ((words ((ywL y).drop 1024) 8).map (bv z) ++ saltTail)) (wordBit t) := by
  unfold hm96Body
  simp only [yw_toList, tail_eq]
  refine Sound.bind' (sound_compress _ _) fun x => Sound.bind' (sound_compress _ _) fun st₁ =>
    Sound.bind' (sound_compress _ _) fun st₂ => Sound.pure' fun z hst₂ hst₁ hx hcv hpad => ?_
  obtain ⟨x1, x2⟩ := hx (length_words _ _) (words64_words (by omega)) (length_words _ _) (words64_words (by omega))
  have hmid : (Flock.Sha512.compress Flock.Sha512.iv Flock.Hm96.Default512.saltPrefix 0).toList.length = 8 :=
    length_saltMid ▸ (List.length_map _).symm
  obtain ⟨s11, s12⟩ := hst₁ (by rw [List.length_map, hmid]) (words64_constW _) (length_words _ _)
    (words64_words (by simp [length_ywL]))
  have hs1l : st₁.length = 8 := by
    rw [length_of_map_bv s12, length_compressV (by rw [List.length_map, List.length_map, hmid])]
  obtain ⟨s21, s22⟩ := hst₂ hs1l s11 (by simp [length_words])
    ((words64_words (by simp [length_ywL])).append (words64_constW _))
  have ec : compressV (compressV saltMid ((words (ywL y) 16).map (bv z)))
      ((words ((ywL y).drop 1024) 8).map (bv z) ++ saltTail) = st₂.map (bv z) := by
    rw [s22, s12, map_bv_constW, List.map_append, map_bv_constW]
    rfl
  refine ⟨by simp, fun t ht => ⟨?_, ?_⟩⟩
  · rw [bit, List.getElem?_append_left (by simp; omega), List.getElem?_map, List.getElem?_range ht]
    simp only [Option.map_some, Option.getD_some, ev_xor, ev_mask]
    rw [ev_flatten_toArray z x1, x2]
  · rw [bit, List.getElem?_append_right (by simp), List.getElem?_map]
    simp only [List.length_map, List.length_range, Nat.add_sub_cancel_left, List.getElem?_range ht, Option.map_some,
      Option.getD_some]
    rw [ev_flatten_toArray z s21, ec]

end FlockSoundness.Discharge.Hm
