/-
Words of the builder (`Flock.HmNets`) and what their gadgets force: a word's value as a `BitVec 64`, rotations, shifts and
XORs as `BitVec` operations, `Ch` and `Maj` bit by bit, commit, and the ripple-carry adder as addition mod `2^64`.
-/
import Proofs.Flock.Soundness.Discharge.Hm.Gadget
import Proofs.Flock.Soundness.Discharge.Hm.Spec

namespace FlockSoundness.Discharge.Hm

open Flock.HmNets hiding and

/-- Bit `i` of a word under `z` (bits past its end are 0). -/
def bit (z : Var → Bool) (w : Word) (i : ℕ) : Bool := ev z (w[i]?.getD Lin.zero)

/-- A word's value, least significant bit first. -/
def valN (z : Var → Bool) : Word → ℕ
  | [] => 0
  | f :: w => (ev z f).toNat + 2 * valN z w

/-- A word's value as a 64-bit vector. -/
def bv (z : Var → Bool) (w : Word) : BitVec 64 := BitVec.ofNat 64 (valN z w)

theorem testBit_valN (z : Var → Bool) : ∀ (w : Word) (i : ℕ), (valN z w).testBit i = bit z w i
  | [], i => by simp [valN, bit]
  | f :: w, 0 => by
      simp only [valN, bit, Nat.testBit_zero, List.getElem?_cons_zero, Option.getD_some]
      cases h : ev z f <;> simp [Nat.add_mod]
  | f :: w, i + 1 => by
      rw [Nat.testBit_succ, show bit z (f :: w) (i + 1) = bit z w i from by simp [bit], ← testBit_valN z w i]
      congr 1
      simp only [valN]
      cases ev z f <;> simp; omega

theorem getLsbD_bv (z : Var → Bool) {w : Word} (hw : w.length ≤ 64) (i : ℕ) :
    (bv z w).getLsbD i = bit z w i := by
  rw [bv, BitVec.getLsbD_ofNat, testBit_valN]
  by_cases hi : i < 64
  · simp [hi]
  · have : w.length ≤ i := by omega
    simp [hi, bit, List.getElem?_eq_none this]

theorem bv_ext {z : Var → Bool} {w : Word} (hw : w.length ≤ 64) {x : BitVec 64}
    (h : ∀ i < 64, bit z w i = x.getLsbD i) : bv z w = x := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  rw [getLsbD_bv z hw]
  exact h i hi

theorem bit_eq_getLsbD {z : Var → Bool} {w : Word} (hw : w.length ≤ 64) (i : ℕ) : bit z w i = (bv z w).getLsbD i :=
  (getLsbD_bv z hw i).symm

@[simp] theorem length_rot (w : Word) (n : ℕ) (hn : n ≤ w.length) : (rot w n).length = w.length := by
  simp [rot]; omega

@[simp] theorem length_shr (w : Word) (n : ℕ) (hn : n ≤ w.length) : (shr w n).length = w.length := by
  simp [shr]; omega

@[simp] theorem length_xorW (a b : Word) : (xorW a b).length = min a.length b.length := by
  simp [xorW]

@[simp] theorem length_xor3 (a b c : Word) : (xor3 a b c).length = min (min a.length b.length) c.length := by
  simp [xor3]

@[simp] theorem length_constW (k : BitVec 64) : (constW k).length = 64 := by simp [constW]

theorem bit_rot (z : Var → Bool) {w : Word} (hw : w.length = 64) {n : ℕ} (hn : n < 64) {i : ℕ} (hi : i < 64) :
    bit z (rot w n) i = bit z w ((n + i) % 64) := by
  simp only [bit, rot]
  by_cases h : i < 64 - n
  · rw [List.getElem?_append_left (by simp [hw]; omega), List.getElem?_drop, Nat.mod_eq_of_lt (by omega)]
  · rw [List.getElem?_append_right (by simp [hw]; omega), List.getElem?_take]
    simp only [List.length_drop, hw]
    rw [ite_eq_left (by omega)]
    congr 3
    omega

theorem bit_shr (z : Var → Bool) {w : Word} (hw : w.length = 64) {n : ℕ} (hn : n < 64) {i : ℕ} (hi : i < 64) :
    bit z (shr w n) i = (decide (n + i < 64) && bit z w (n + i)) := by
  simp only [bit, shr]
  by_cases h : i < 64 - n
  · rw [List.getElem?_append_left (by simp [hw]; omega), List.getElem?_drop]
    simp [show n + i < 64 by omega]
  · rw [List.getElem?_append_right (by simp [hw]; omega)]
    simp only [List.length_drop, hw, List.getElem?_replicate]
    split
    · simp only [Option.getD_some, ev_zero, show ¬ n + i < 64 by omega, decide_false, Bool.false_and]
    · omega

theorem bit_xorW (z : Var → Bool) {a b : Word} {i : ℕ} (ha : i < a.length) (hb : i < b.length) :
    bit z (xorW a b) i = (bit z a i ^^ bit z b i) := by
  simp only [bit, xorW]
  rw [List.getElem?_eq_getElem (by simp; omega), List.getElem?_eq_getElem ha, List.getElem?_eq_getElem hb]
  simp

theorem bit_xor3 (z : Var → Bool) {a b c : Word} (ha : a.length = 64) (hb : b.length = 64) (hc : c.length = 64)
    {i : ℕ} (hi : i < 64) : bit z (xor3 a b c) i = (bit z a i ^^ bit z b i ^^ bit z c i) := by
  rw [xor3, bit_xorW z (by simp; omega) (by omega), bit_xorW z (by omega) (by omega)]

theorem bit_constW (z : Var → Bool) (k : BitVec 64) {i : ℕ} (hi : i < 64) : bit z (constW k) i = k.getLsbD i := by
  simp [bit, constW, hi]

theorem bv_constW (z : Var → Bool) (k : BitVec 64) : bv z (constW k) = k :=
  bv_ext (by simp) fun i hi => bit_constW z k hi

theorem bv_rot (z : Var → Bool) {w : Word} (hw : w.length = 64) {n : ℕ} (hn : n < 64) :
    bv z (rot w n) = (bv z w).rotateRight n := by
  refine bv_ext (by rw [length_rot w n (by omega), hw]) fun i hi => ?_
  rw [bit_rot z hw hn hi, BitVec.getLsbD_rotateRight, Nat.mod_eq_of_lt hn, bit_eq_getLsbD (by omega)]
  by_cases h : i < 64 - n
  · simp only [h, ↓reduceIte, Nat.mod_eq_of_lt (show n + i < 64 by omega)]
  · simp only [h, ↓reduceIte, hi, decide_true, Bool.true_and]
    congr 1
    omega

theorem bv_shr (z : Var → Bool) {w : Word} (hw : w.length = 64) {n : ℕ} (hn : n < 64) :
    bv z (shr w n) = bv z w >>> n := by
  refine bv_ext (by rw [length_shr w n (by omega), hw]) fun i hi => ?_
  rw [bit_shr z hw hn hi, BitVec.getLsbD_ushiftRight, bit_eq_getLsbD (by omega)]
  by_cases h : n + i < 64
  · rw [decide_eq_true h, Bool.true_and]
  · rw [decide_eq_false h, Bool.false_and, BitVec.getLsbD_of_ge _ _ (by omega)]

theorem bv_xor3 (z : Var → Bool) {a b c : Word} (ha : a.length = 64) (hb : b.length = 64) (hc : c.length = 64) :
    bv z (xor3 a b c) = bv z a ^^^ bv z b ^^^ bv z c := by
  refine bv_ext (by simp [ha, hb, hc]) fun i hi => ?_
  rw [bit_xor3 z ha hb hc hi, BitVec.getLsbD_xor, BitVec.getLsbD_xor, bit_eq_getLsbD (by omega),
    bit_eq_getLsbD (z := z) (w := b) (by omega), bit_eq_getLsbD (z := z) (w := c) (by omega)]


theorem bit_of_lt {z : Var → Bool} {w : Word} {i : ℕ} (h : i < w.length) : bit z w i = ev z (w[i]) := by
  simp [bit, List.getElem?_eq_getElem h]

theorem bv_congr {z : Var → Bool} {a b : Word} (ha : a.length = 64) (hb : b.length = 64)
    (h : ∀ i < 64, bit z a i = bit z b i) : bv z a = bv z b :=
  bv_ext (by omega) fun i hi => by rw [h i hi, bit_eq_getLsbD (by omega)]

/-! ## Bitwise gadgets -/

/-- A gadget applied bit by bit, each bit's result its form's value. -/
theorem sound_mapM {β : Type} (f : β → M Lin) (g : (Var → Bool) → β → Bool)
    (hf : ∀ x, Sound (f x) fun z r => ev z r = g z x) :
    ∀ l : List β, Sound (l.mapM f) fun z rs => rs.length = l.length ∧ ∀ i (h : i < l.length), bit z rs i = g z l[i]
  | [] => Sound.pure' fun z => ⟨rfl, fun i h => absurd h (by simp)⟩
  | x :: l => by
      rw [List.mapM_cons]
      refine Sound.bind' (hf x) fun r => Sound.bind' (sound_mapM f g hf l) fun rs => Sound.pure' fun z hl hr => ?_
      obtain ⟨hlen, hbits⟩ := hl
      refine ⟨by simp [hlen], fun i h => ?_⟩
      cases i with
      | zero => simp [bit, hr]
      | succ i =>
          have := hbits i (by simpa using h)
          simpa [bit] using this

theorem sound_chW (e f g : Word) : Sound (chW e f g) fun z r =>
    e.length = 64 → f.length = 64 → g.length = 64 → r.length = 64 ∧ bv z r = chV (bv z e) (bv z f) (bv z g) := by
  refine Sound.mono (sound_mapM _ (fun z x => ev z (x.2.2) ^^ (ev z (x.1) && (ev z (x.2.1) ^^ ev z (x.2.2))))
    (fun x => Sound.bind' (sound_and _ _) fun r => Sound.pure' fun z hr => by simp [hr]) _) fun z r h he hf hg => ?_
  obtain ⟨hlen, hbits⟩ := h
  have hl : r.length = 64 := by simp [hlen, he, hf, hg]
  refine ⟨hl, bv_ext (by omega) fun i hi => ?_⟩
  rw [hbits i (by simp [he, hf, hg, hi])]
  simp only [chV, BitVec.getLsbD_xor, BitVec.getLsbD_and, BitVec.getLsbD_not, List.getElem_zip]
  rw [← bit_of_lt (by omega), ← bit_of_lt (by omega), ← bit_of_lt (by omega), bit_eq_getLsbD (z := z) (w := e) (by omega),
    bit_eq_getLsbD (z := z) (w := f) (by omega), bit_eq_getLsbD (z := z) (w := g) (by omega)]
  simp only [hi, decide_true, Bool.true_and]
  cases (bv z e).getLsbD i <;> cases (bv z f).getLsbD i <;> cases (bv z g).getLsbD i <;> rfl

theorem sound_majW (a b c : Word) : Sound (majW a b c) fun z r =>
    a.length = 64 → b.length = 64 → c.length = 64 → r.length = 64 ∧ bv z r = majV (bv z a) (bv z b) (bv z c) := by
  refine Sound.mono (sound_mapM _ (fun z x => maj3 (ev z (x.1)) (ev z (x.2.1)) (ev z (x.2.2)))
    (fun x => sound_maj _ _ _) _) fun z r h ha hb hc => ?_
  obtain ⟨hlen, hbits⟩ := h
  have hl : r.length = 64 := by simp [hlen, ha, hb, hc]
  refine ⟨hl, bv_ext (by omega) fun i hi => ?_⟩
  rw [hbits i (by simp [ha, hb, hc, hi])]
  simp only [majV, maj3, BitVec.getLsbD_xor, BitVec.getLsbD_and, List.getElem_zip]
  rw [← bit_of_lt (by omega), ← bit_of_lt (by omega), ← bit_of_lt (by omega), bit_eq_getLsbD (z := z) (w := a) (by omega),
    bit_eq_getLsbD (z := z) (w := b) (by omega), bit_eq_getLsbD (z := z) (w := c) (by omega)]

theorem sound_commitW (w : Word) : Sound (commitW w) fun z r => w.length = 64 → r.length = 64 ∧ bv z r = bv z w := by
  refine Sound.mono (sound_mapM _ (fun z f => ev z f) (fun f => sound_commit f) w) fun z r h hw => ?_
  obtain ⟨hlen, hbits⟩ := h
  refine ⟨by omega, bv_congr (by omega) hw fun i hi => ?_⟩
  rw [hbits i (by omega), bit_of_lt (by omega)]

/-! ## The ripple-carry adder -/

theorem sound_carryCommit (i : ℕ) (c : Lin) : Sound (carryCommit i c) fun z r => ev z r = ev z c :=
  Sound.ite' (fun _ => sound_commit c) fun _ => Sound.pure' fun _ => rfl

theorem mod_two_mul (s X n : ℕ) (hs : s < 2) : (s + 2 * X) % 2 ^ (n + 1) = s + 2 * (X % 2 ^ n) := by
  have h := Nat.div_add_mod X (2 ^ n)
  have hr := Nat.mod_lt X (Nat.two_pow_pos n)
  rw [pow_succ]
  conv_lhs => rw [← h]
  rw [show s + 2 * (2 ^ n * (X / 2 ^ n) + X % 2 ^ n) = (s + 2 * (X % 2 ^ n)) + (2 ^ n * 2) * (X / 2 ^ n) by ring,
    Nat.add_mul_mod_self_left]
  generalize 2 ^ n = P at *
  exact Nat.mod_eq_of_lt (by omega)

theorem full_add (x y c : Bool) : x.toNat + y.toNat + c.toNat = (x ^^ y ^^ c).toNat + 2 * (maj3 x y c).toNat := by
  cases x <;> cases y <;> cases c <;> rfl

theorem sound_add2Go : ∀ (a b : Word) (c : Lin) (i : ℕ), Sound (add2Go a b c i) fun z r =>
    a.length = b.length → r.length = a.length ∧ valN z r = (valN z a + valN z b + (ev z c).toNat) % 2 ^ a.length
  | [], b, c, i => by
      rw [add2Go]
      · exact Sound.pure' fun z _ => ⟨rfl, by simp [Nat.mod_one, valN]⟩
      · simp
  | x :: a, [], c, i => by
      rw [add2Go]
      · exact Sound.pure' fun z h => absurd h (by simp)
      · simp
  | x :: a, y :: b, c, i => by
      rw [add2Go]
      refine Sound.ite' (fun ha => Sound.pure' fun z hl => ?_) fun ha => ?_
      · simp only [List.isEmpty_iff] at ha
        subst ha
        have hb : b = [] := List.eq_nil_of_length_eq_zero (by simpa using hl.symm)
        subst hb
        refine ⟨rfl, ?_⟩
        simp only [valN, List.length_cons, List.length_nil, zero_add, pow_one, mul_zero, add_zero, ev_xor]
        rw [full_add]
        cases (ev z x ^^ ev z y ^^ ev z c) <;> cases maj3 (ev z x) (ev z y) (ev z c) <;> simp
      · refine Sound.bind' (sound_maj x y c) fun c₁ => Sound.bind' (sound_carryCommit i c₁)
          fun c₂ => Sound.bind' (sound_add2Go a b c₂ (i + 1)) fun r => Sound.pure' fun z ih hc₂ hc₁ hl => ?_
        · have hl' : a.length = b.length := by simpa using hl
          obtain ⟨hr1, hr2⟩ := ih hl'
          refine ⟨by simp [hr1], ?_⟩
          simp only [valN, ev_xor, List.length_cons, hr2, hc₂, hc₁]
          have hfa := full_add (ev z x) (ev z y) (ev z c)
          rw [show (ev z x).toNat + 2 * valN z a + ((ev z y).toNat + 2 * valN z b) + (ev z c).toNat =
              (ev z x ^^ ev z y ^^ ev z c).toNat +
                2 * (valN z a + valN z b + (maj3 (ev z x) (ev z y) (ev z c)).toNat) by omega,
            mod_two_mul _ _ _ (Bool.toNat_lt _)]

theorem bv_ofNat_mod (x : ℕ) : BitVec.ofNat 64 (x % 2 ^ 64) = BitVec.ofNat 64 x := by
  apply BitVec.eq_of_toNat_eq
  simp

theorem sound_add2 (a b : Word) : Sound (add2 a b) fun z r =>
    a.length = 64 → b.length = 64 → r.length = 64 ∧ bv z r = bv z a + bv z b :=
  Sound.mono (sound_add2Go a b Lin.zero 0) fun z r h ha hb => by
    obtain ⟨h1, h2⟩ := h (by omega)
    refine ⟨by omega, ?_⟩
    rw [bv, h2, ha, ev_zero, Bool.toNat_false, add_zero, bv_ofNat_mod, ← BitVec.ofNat_add_ofNat]
    rfl

theorem sound_addN : ∀ (acc : Word) (xs : List Word), Sound (addN acc xs) fun z r =>
    acc.length = 64 → (∀ x ∈ xs, x.length = 64) → r.length = 64 ∧ bv z r = (xs.map (bv z)).foldl (· + ·) (bv z acc)
  | acc, [] => Sound.pure' fun z h _ => ⟨h, rfl⟩
  | acc, x :: xs => Sound.bind' (sound_add2 acc x) fun r => Sound.mono (sound_addN r xs) fun z r' ih h ha hx => by
      obtain ⟨hr1, hr2⟩ := h ha (hx x List.mem_cons_self)
      obtain ⟨h1, h2⟩ := ih hr1 fun y hy => hx y (List.mem_cons_of_mem _ hy)
      refine ⟨h1, ?_⟩
      rw [h2, hr2]
      rfl

end FlockSoundness.Discharge.Hm
