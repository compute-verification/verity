/-!
# Hiding a public digest: the counter-mode pad, its keyed forms, and hm96's mask as forced work

A public digest `x` (a frame-v3 subtree's digest) is hidden three ways here, each a function of a window key `ks` that
is committed under its own hm96 leaf before the window's coins, so nothing in it is a free salt:

* the pad (P1): `b = x ⊕ prf ks pos`, `prf ks pos = f(f(IV, ks ‖ salt_w), pos ‖ padding)`, `pos` the digest's frame-v3
  domain, rank and position;
* the keyed digest (P2) and the keyed leaf (P3): `d = G (ks, pos, x)`, SHA-512 of `ks ‖ salt_w ‖ pos ‖ x` (P2) or the
  digest's own hash started from `f(IV, ks ‖ salt_w)`, its first block row-bound's, which carries the position (P3, `x`
  its content).

Binding, premise-free over the functions:

* `pad_binding`: two openings of one `(commitment of ks, b)` to distinct payloads give a collision in ks's commitment
  or in the inner hash. `pad_unchecked_unbound`: if the proof does not check the pad against ks, `b` opens to every `x'`.
* `keyed_binding`: P2 and P3 bind by collision resistance alone, for any keys, committed or not.

Hiding, in the ideal hybrid (the PRF replaced by a uniform function of the position):

* `pad_reduction`: the real view is the ideal view at `F = prf ks`, so a distinguisher of real from ideal views is a
  distinguisher of `prf ks` from a uniform function, queried at the uses' positions.
* `pad_ideal_hiding`: if no two uses share a position, an involution of `F` carries data `x`'s ideal view to `x'`'s,
  so under uniform `F` the two views have one distribution. `keyed_ideal_hiding` is the same for P2 and P3.
* `pad_repeat_leaks`, `keyed_repeat_leaks`: a repeated position reveals `x u ⊕ x v` (the pad) or whether `x u = x v`
  (the keyed forms). `pad_key_reveals`: whoever holds ks reads every `x` in its scope.
* `prefix_free_of_head`: P3's messages are prefix-free when the first block fixes the length, so the cascade from `S`
  reduces to the compression function's PRF property (Bellare-Canetti-Krawczyk) with no length extension.

The forced mask (FM) computes hm96-sha512/v1's mask `M_key · y` over GF(2) as an int8 tensor-core product:

* `lane_int8`, `weight_int8`: the formed lanes `2^k · y` (`-128 · y` at `k = 7`) and the public weights `M · s_k`
  are int8.
* `fm_mask`: every product is `±128 · m · y`, so the int32 sum, read modulo `2^32` as the accumulator holds it, has
  bits 0..6 zero and bit 7 equal to the XOR of the row's `m && y`. Every k32 partial is a prefix's sum, so `fm_mask`
  holds for each.

Probability is the reader's step: an involution is a bijection, so a uniform `F` maps to a uniform `φ F`.
-/

namespace Pouw.Salt

section Pad

variable {n : Nat} {K C M P U : Type}

/-- The pad's public digest: the inner digest `x` XOR the key's pad at position `p`. -/
def padDigest (prf : K → P → BitVec n) (ks : K) (p : P) (x : BitVec n) : BitVec n := x ^^^ prf ks p

theorem xor_cancel (x r : BitVec n) : x ^^^ r ^^^ r = x := by
  rw [BitVec.xor_assoc, BitVec.xor_self, BitVec.xor_zero]

theorem xor_cancel_left (r x : BitVec n) : r ^^^ (r ^^^ x) = x := by
  rw [← BitVec.xor_assoc, BitVec.xor_self, BitVec.zero_xor]

theorem xor_xor_cancel (a b r : BitVec n) : a ^^^ r ^^^ (b ^^^ r) = a ^^^ b := by
  rw [BitVec.xor_comm b r, ← BitVec.xor_assoc, xor_cancel]

/-- Two openings `(ks, m)`, `(ks', m')` of one public pair (ks's commitment, the pad's digest at `p`) to distinct
payloads give a collision in ks's commitment (`ks` is an opening, its leaf salt included) or in the inner hash `H`. -/
theorem pad_binding (ck : K → C) (H : M → BitVec n) (prf : K → P → BitVec n) (p : P) {ks ks' : K} {m m' : M}
    (hk : ck ks = ck ks') (hb : padDigest prf ks p (H m) = padDigest prf ks' p (H m')) (hm : m ≠ m') :
    (ks ≠ ks' ∧ ck ks = ck ks') ∨ (m ≠ m' ∧ H m = H m') := by
  by_cases h : ks = ks'
  · subst h
    exact Or.inr ⟨hm, (BitVec.xor_left_inj (prf ks p)).mp hb⟩
  · exact Or.inl ⟨h, hk⟩

/-- A pad the proof does not check binds nothing: `b` is `x' ⊕ r'` for every `x'`, with `r' = x' ⊕ b`. -/
theorem pad_unchecked_unbound (b x' : BitVec n) : ∃ r', x' ^^^ r' = b :=
  ⟨x' ^^^ b, by rw [← BitVec.xor_assoc, BitVec.xor_self, BitVec.zero_xor]⟩

/-- Whoever holds the key reads the inner digest. -/
theorem pad_key_reveals (prf : K → P → BitVec n) (ks : K) (p : P) (x : BitVec n) :
    padDigest prf ks p x ^^^ prf ks p = x :=
  xor_cancel x (prf ks p)

/-- The ideal view of the uses `U`: each use's digest XOR a function `F` of its position. -/
def view (pos : U → P) (x : U → BitVec n) (F : P → BitVec n) : U → BitVec n := fun u => x u ^^^ F (pos u)

/-- The real view is the ideal view at `F = prf ks`. -/
theorem pad_reduction (prf : K → P → BitVec n) (ks : K) (pos : U → P) (x : U → BitVec n) :
    (fun u => padDigest prf ks (pos u) (x u)) = view pos x (prf ks) :=
  rfl

/-- A repeated position reveals the XOR of the two digests, whatever `F` is. -/
theorem pad_repeat_leaks (pos : U → P) (x : U → BitVec n) (F : P → BitVec n) {u v : U} (h : pos u = pos v) :
    view pos x F u ^^^ view pos x F v = x u ^^^ x v := by
  simp only [view, h]
  exact xor_xor_cancel (x u) (x v) (F (pos v))

theorem exists_delta (pos : U → P) (hpos : ∀ u v, pos u = pos v → u = v) (x x' : U → BitVec n) :
    ∃ δ : P → BitVec n, ∀ u, δ (pos u) = x u ^^^ x' u := by
  classical
  refine ⟨fun p => if h : ∃ u, pos u = p then x (Classical.choose h) ^^^ x' (Classical.choose h) else 0, ?_⟩
  intro u
  have h : ∃ v, pos v = pos u := ⟨u, rfl⟩
  simp only [dite_eq_left h]
  rw [hpos _ _ (Classical.choose_spec h)]

/-- With every position used once, the shift `F ↦ F ⊕ δ` is an involution carrying `x`'s ideal view to `x'`'s. -/
theorem pad_ideal_hiding (pos : U → P) (hpos : ∀ u v, pos u = pos v → u = v) (x x' : U → BitVec n) :
    ∃ φ : (P → BitVec n) → (P → BitVec n), (∀ F, φ (φ F) = F) ∧ ∀ F, view pos x' (φ F) = view pos x F := by
  obtain ⟨δ, hδ⟩ := exists_delta pos hpos x x'
  refine ⟨fun F p => F p ^^^ δ p, fun F => ?_, fun F => ?_⟩
  · funext p
    exact xor_cancel (F p) (δ p)
  · funext u
    simp only [view, hδ]
    rw [BitVec.xor_comm (F (pos u)), ← BitVec.xor_assoc, BitVec.xor_comm (x u) (x' u), xor_cancel_left]

end Pad

section Keyed

variable {K C P X D U : Type}

/-- P2 and P3 bind by collision resistance alone: two openings to distinct data are a collision of the keyed hash on
distinct inputs, whatever keys they use, committed or not. -/
theorem keyed_binding (G : K × P × X → D) (p : P) {ks ks' : K} {x x' : X} (hd : G (ks, p, x) = G (ks', p, x'))
    (hx : x ≠ x') : (ks, p, x) ≠ (ks', p, x') ∧ G (ks, p, x) = G (ks', p, x') :=
  ⟨fun h => hx (congrArg (fun t => t.2.2) h), hd⟩

/-- The swap of `a` and `b`. -/
def swap [DecidableEq X] (a b z : X) : X := if z = a then b else if z = b then a else z

theorem swap_swap [DecidableEq X] (a b z : X) : swap a b (swap a b z) = z := by
  unfold swap
  by_cases ha : z = a
  · subst ha
    by_cases hb : b = z
    · simp [hb]
    · simp [hb]
  · by_cases hb : z = b
    · subst hb
      simp [ha]
    · simp [ha, hb]

theorem swap_right [DecidableEq X] (a b : X) : swap a b b = a := by
  unfold swap
  by_cases h : b = a
  · simp [h]
  · simp [h]

/-- The keyed forms' ideal view: a uniform function of (position, data) at each use. -/
def keyedView (pos : U → P) (x : U → X) (F : P × X → D) : U → D := fun u => F (pos u, x u)

theorem swap_left [DecidableEq X] (a b : X) : swap a b a = b := by
  simp [swap]

/-- At each used position, the swap of the two data that use it. -/
noncomputable def keyedSwap [DecidableEq X] (pos : U → P) (x x' : U → X) (p : P) (z : X) : X :=
  open Classical in if h : ∃ u, pos u = p then swap (x' (Classical.choose h)) (x (Classical.choose h)) z else z

theorem keyedSwap_swap [DecidableEq X] (pos : U → P) (x x' : U → X) (p : P) (z : X) :
    keyedSwap pos x x' p (keyedSwap pos x x' p z) = z := by
  by_cases h : ∃ u, pos u = p
  · simp only [keyedSwap, dite_eq_left h, swap_swap]
  · simp only [keyedSwap, dite_eq_right h]

theorem keyedSwap_use [DecidableEq X] (pos : U → P) (hpos : ∀ u v, pos u = pos v → u = v) (x x' : U → X) (u : U) :
    keyedSwap pos x x' (pos u) (x' u) = x u := by
  have h : ∃ v, pos v = pos u := ⟨u, rfl⟩
  simp only [keyedSwap, dite_eq_left h]
  rw [hpos _ _ (Classical.choose_spec h), swap_left]

/-- With every position used once, swapping `F`'s values at `(pos u, x u)` and `(pos u, x' u)` is an involution carrying
`x`'s ideal view to `x'`'s. -/
theorem keyed_ideal_hiding [DecidableEq X] (pos : U → P) (hpos : ∀ u v, pos u = pos v → u = v) (x x' : U → X) :
    ∃ φ : (P × X → D) → (P × X → D), (∀ F, φ (φ F) = F) ∧ ∀ F, keyedView pos x' (φ F) = keyedView pos x F := by
  refine ⟨fun F q => F (q.1, keyedSwap pos x x' q.1 q.2), fun F => ?_, fun F => ?_⟩
  · funext q
    simp only [keyedSwap_swap]
  · funext u
    simp only [keyedView, keyedSwap_use pos hpos]

/-- The keyed leaf's messages, as block lists hashed from `S`, are prefix-free if every message is nonempty and its first
block fixes its length (row-bound's first block carries the position, and a position's layout fixes the length): no
message extends another, which is the set Bellare-Canetti-Krawczyk's cascade reduction needs. -/
theorem prefix_free_of_head {B : Type} (S : List B → Prop) (hne : ∀ m, S m → m ≠ [])
    (hlen : ∀ m m', S m → S m' → m.head? = m'.head? → m.length = m'.length) :
    ∀ m t, S m → S (m ++ t) → t = [] := by
  intro m t hm hmt
  have hh : (m ++ t).head? = m.head? := by
    cases m with
    | nil => exact absurd rfl (hne _ hm)
    | cons a l => rfl
  have hl := hlen m (m ++ t) hm hmt hh.symm
  rw [List.length_append] at hl
  exact List.eq_nil_of_length_eq_zero (by omega)

/-- At a repeated position the keyed forms reveal whether the two data are equal. -/
theorem keyed_repeat_leaks (pos : U → P) (x : U → X) (F : P × X → D) {u v : U} (h : pos u = pos v) (he : x u = x v) :
    keyedView pos x F u = keyedView pos x F v := by
  simp only [keyedView, h, he]

end Keyed

section ForcedMask

/-- Int8 lane `e` of the formed register `R_{w,k} = h_w & (0x01010101 << k)`, read signed: `2^k · y`, and `-128 · y` at
`k = 7`. -/
def lane (k : Fin 8) (y : Bool) : Int := if y then (if k.val = 7 then -128 else 2 ^ k.val) else 0

/-- The public weight `M[i][j] · s_k`: `s_0 = -128`, `s_k = 2^(7 - k)` for `k = 1..6`, `s_7 = 1`. -/
def weight (k : Fin 8) (m : Bool) : Int :=
  if m then (if k.val = 0 then -128 else if k.val = 7 then 1 else 2 ^ (7 - k.val)) else 0

def sgn (k : Fin 8) : Int := if k.val = 0 ∨ k.val = 7 then -1 else 1

def bit : Bool → Int
  | true => 1
  | false => 0

theorem lane_int8 : ∀ k y, -128 ≤ lane k y ∧ lane k y ≤ 127 := by decide

theorem weight_int8 : ∀ k m, -128 ≤ weight k m ∧ weight k m ≤ 127 := by decide

/-- Every product is `±128 · m · y`. -/
theorem mac : ∀ k m y, lane k y * weight k m = 128 * (sgn k * bit (m && y)) := by decide

theorem mac_parity : ∀ k m y, (sgn k * bit (m && y)) % 2 = bit (m && y) := by decide

/-- A row's int32 accumulation over `(k, M[i][j], y_j)`, `k = j mod 8`. -/
def acc : List (Fin 8 × Bool × Bool) → Int
  | [] => 0
  | (k, m, y) :: l => lane k y * weight k m + acc l

/-- The row's mask bit over GF(2): the XOR of `M[i][j] && y_j`. -/
def gf2 : List (Fin 8 × Bool × Bool) → Bool
  | [] => false
  | (_, m, y) :: l => Bool.xor (m && y) (gf2 l)

theorem fm_acc (l : List (Fin 8 × Bool × Bool)) : ∃ t : Int, acc l = 128 * t ∧ t % 2 = bit (gf2 l) := by
  induction l with
  | nil => exact ⟨0, rfl, rfl⟩
  | cons a l ih =>
    obtain ⟨k, m, y⟩ := a
    obtain ⟨t, ht, hp⟩ := ih
    refine ⟨sgn k * bit (m && y) + t, ?_, ?_⟩
    · simp only [acc]
      rw [mac, ht, Int.mul_add]
    · have hm := mac_parity k m y
      simp only [gf2]
      revert hm hp
      generalize sgn k * bit (m && y) = e
      generalize (m && y) = c
      generalize gf2 l = g
      cases c <;> cases g <;> simp [bit] <;> omega

/-- bit 7 of the int32 accumulator, which holds the sum modulo `2^32`, is the mask bit, and bits 0..6 are zero. -/
theorem fm_mask (l : List (Fin 8 × Bool × Bool)) :
    (acc l % 4294967296 / 128) % 2 = bit (gf2 l) ∧ acc l % 128 = 0 := by
  obtain ⟨t, ht, hp⟩ := fm_acc l
  rw [ht]
  constructor <;> omega

end ForcedMask

end Pouw.Salt
