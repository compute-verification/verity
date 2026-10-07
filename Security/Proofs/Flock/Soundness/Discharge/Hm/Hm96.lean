/-
What the hm96 gadget's rows force (`Flock.HmNets.hm96Row`): its `b ‖ c` is the padding compression's digest XOR `M(key)·y`,
then SHA-512's last two compressions of `salt_prefix ‖ y`, bit by bit in byte-string order.
-/
import Proofs.Flock.Soundness.Discharge.Hm.Compress

namespace FlockSoundness.Discharge.Hm

open Flock.HmNets hiding and

theorem length_compressV {cv : List (BitVec 64)} (h : cv.length = 8) (msg : List (BitVec 64)) :
    (compressV cv msg).length = 8 := by
  simp [compressV, h, Regs.toList]

theorem length_of_map_bv {z : Var → Bool} {ws : List Word} {vs : List (BitVec 64)} (h : ws.map (bv z) = vs) :
    ws.length = vs.length := by
  rw [← h, List.length_map]

theorem length_words (bits : List Lin) (n : ℕ) : (words bits n).length = n := by simp [words]

theorem words64_words {bits : List Lin} {n : ℕ} (h : bits.length = 64 * n) : Words64 (words bits n) := by
  intro w hw
  simp only [words, List.mem_map, List.mem_range] at hw
  obtain ⟨i, hi, rfl⟩ := hw
  simp only [List.length_take, List.length_drop, h]
  omega

theorem length_saltWords (y : List Lin) : (saltWords y).length = 24 := by simp [saltWords]

theorem words64_saltWords (y : List Lin) : Words64 (saltWords y) := by
  intro w hw
  simp only [saltWords, List.mem_map, List.mem_range] at hw
  obtain ⟨i, _, rfl⟩ := hw
  simp

theorem Words64.take {w : List Word} (h : Words64 w) (n : ℕ) : Words64 (w.take n) :=
  fun x hx => h x (List.mem_of_mem_take hx)

theorem Words64.drop {w : List Word} (h : Words64 w) (n : ℕ) : Words64 (w.drop n) :=
  fun x hx => h x (List.mem_of_mem_drop hx)

theorem Words64.append {a b : List Word} (ha : Words64 a) (hb : Words64 b) : Words64 (a ++ b) := fun x hx => by
  rcases List.mem_append.1 hx with h | h
  · exact ha x h
  · exact hb x h

theorem words64_constW (ks : List (BitVec 64)) : Words64 (ks.map constW) := by
  intro w hw
  simp only [List.mem_map] at hw
  obtain ⟨k, _, rfl⟩ := hw
  exact length_constW k

theorem map_bv_constW (z : Var → Bool) (ks : List (BitVec 64)) : (ks.map constW).map (bv z) = ks := by
  simp [List.map_map, Function.comp_def, bv_constW]

theorem length_saltMid : saltMid.length = 8 := by simp [saltMid]

theorem length_saltTail : saltTail.length = 8 := rfl

/-- A bit of concatenated 64-bit words is its word's bit. -/
theorem bit_flatten (z : Var → Bool) : ∀ {ws : List Word}, Words64 ws → ∀ p : ℕ,
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

theorem eval_maskBit (z : Var → Bool) (key : ByteArray) (y : List Lin) (i : ℕ) (x : Lin) :
    ev z (maskBit key y i x) = (ev z x ^^ maskV key (bit z y) i) := by
  have hy : ∀ j, ev z (y.getD j Lin.zero) = bit z y j := fun j => by rw [bit, List.getD_eq_getElem?_getD]
  rw [maskBit_hom (ev z) (ev_xor z) key y i x, maskV]
  simp only [hy]

theorem wordBit_lt {t : ℕ} (ht : t < 512) : wordBit t < 512 := by
  unfold wordBit; omega

/-- `Flock.HmNets.hm96Row` after its three inputs. -/
def hmTail (key : ByteArray) (cv pad y : List Lin) : M (List Lin × List Lin × List Lin × List Lin) := do
  let x ← compress (words cv 8) (words pad 16)
  let yw := saltWords y
  let st ← compress (saltMid.map constW) (yw.take 16)
  let st ← compress st (yw.drop 16 ++ saltTail.map constW)
  let b := (List.range 512).map fun i => maskBit key y i (x.flatten.getD (wordBit i) Lin.zero)
  let c := (List.range 512).map fun t => st.flatten.getD (wordBit t) Lin.zero
  pure (cv, pad, y, b ++ c)

theorem hm96Row_eq (key : ByteArray) : hm96Row key =
    (input 512 >>= fun cv => input 1024 >>= fun pad => input 1536 >>= fun y => hmTail key cv pad y) := rfl

/-- **The hm96 gadget computes `b ‖ c`** on inputs `cv`, `pad`, `y`: whenever its rows hold, its output's bit `t < 512` is
bit `t` (byte-string order) of the padding compression `x = F(cv, pad)` XOR bit `t` of `M(key)·y`, and its bit `512 + t`
is bit `t` of `c = F(F(mid, y₀), y₁ ‖ tail)`, SHA-512's last two compressions of `salt_prefix ‖ y`. -/
theorem sound_hmTail (key : ByteArray) (cv pad y : List Lin) : Sound (hmTail key cv pad y) fun z io =>
    io.1 = cv ∧ io.2.1 = pad ∧ io.2.2.1 = y ∧ (cv.length = 512 → pad.length = 1024 → io.2.2.2.length = 1024 ∧
      ∀ t < 512,
        bit z io.2.2.2 t = (wbit (compressV ((words cv 8).map (bv z)) ((words pad 16).map (bv z))) (wordBit t) ^^
          maskV key (bit z y) t) ∧
        bit z io.2.2.2 (512 + t) = wbit (compressV (compressV saltMid (((saltWords y).map (bv z)).take 16))
          (((saltWords y).map (bv z)).drop 16 ++ saltTail)) (wordBit t)) := by
  unfold hmTail
  refine Sound.bind' (sound_compress _ _) fun x =>
    Sound.bind' (sound_compress _ _) fun st₁ => Sound.bind' (sound_compress _ _) fun st₂ =>
    Sound.pure' fun z hst₂ hst₁ hx => ⟨rfl, rfl, rfl, fun hcv hpad => ?_⟩
  obtain ⟨x1, x2⟩ := hx (length_words _ _) (words64_words (by omega)) (length_words _ _) (words64_words (by omega))
  obtain ⟨s11, s12⟩ := hst₁ (by simp [length_saltMid]) (words64_constW _) (by simp [length_saltWords])
    ((words64_saltWords y).take 16)
  have hs1l : st₁.length = 8 := by
    rw [length_of_map_bv s12, length_compressV (by simp [length_saltMid])]
  obtain ⟨s21, s22⟩ := hst₂ hs1l s11 (by simp [length_saltWords, length_saltTail])
    (((words64_saltWords y).drop 16).append (words64_constW _))
  have hxl : x.length = 8 := by rw [length_of_map_bv x2, length_compressV (by simp [words])]
  have ec : compressV (compressV saltMid (((saltWords y).map (bv z)).take 16))
      (((saltWords y).map (bv z)).drop 16 ++ saltTail) = st₂.map (bv z) := by
    rw [s22, s12, map_bv_constW, List.map_append, map_bv_constW, List.map_take, List.map_drop]
  refine ⟨by simp, fun t ht => ⟨?_, ?_⟩⟩
  · rw [bit, List.getElem?_append_left (by simp; omega), List.getElem?_map, List.getElem?_range ht]
    simp only [Option.map_some, Option.getD_some]
    rw [eval_maskBit, List.getD_eq_getElem?_getD]
    rw [show ev z (x.flatten[wordBit t]?.getD Lin.zero) = bit z x.flatten (wordBit t) from rfl, bit_flatten z x1,
      ← x2, wbit, getD_map_bv]
  · rw [bit, List.getElem?_append_right (by simp), List.getElem?_map]
    simp only [List.length_map, List.length_range, Nat.add_sub_cancel_left, List.getElem?_range ht, Option.map_some,
      Option.getD_some]
    rw [List.getD_eq_getElem?_getD,
      show ev z (st₂.flatten[wordBit t]?.getD Lin.zero) = bit z st₂.flatten (wordBit t) from rfl, bit_flatten z s21,
      ec, wbit, getD_map_bv]

/-- **The hm96 gadget computes `b ‖ c`** (`sound_hmTail`), with its inputs: `cv` (512 bits), `pad` (1024) and `y`
(1536). -/
theorem sound_hm96Row (key : ByteArray) : Sound (hm96Row key) fun z io =>
    io.1.length = 512 ∧ io.2.1.length = 1024 ∧ io.2.2.1.length = 1536 ∧ io.2.2.2.length = 1024 ∧
      ∀ t < 512,
        bit z io.2.2.2 t = (wbit (compressV ((words io.1 8).map (bv z)) ((words io.2.1 16).map (bv z))) (wordBit t) ^^
          maskV key (bit z io.2.2.1) t) ∧
        bit z io.2.2.2 (512 + t) = wbit (compressV (compressV saltMid (((saltWords io.2.2.1).map (bv z)).take 16))
          (((saltWords io.2.2.1).map (bv z)).drop 16 ++ saltTail)) (wordBit t) := by
  rw [hm96Row_eq]
  refine Sound.bind' (sound_input 512) fun cv => Sound.bind' (sound_input 1024) fun pad =>
    Sound.bind' (sound_input 1536) fun y => Sound.mono (sound_hmTail key cv pad y) fun z io h hy hpad hcv => ?_
  obtain ⟨h1, h2, h3, h4⟩ := h
  obtain ⟨h5, h6⟩ := h4 hcv hpad
  subst h1 h2 h3
  exact ⟨hcv, hpad, hy, h5, h6⟩

end FlockSoundness.Discharge.Hm
