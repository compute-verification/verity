import Proofs.Flock.Soundness.Discharge.Layout.HmRegion

/-!
# The returned outputs that the statement pins at the unit slots' output columns

In an untyped circuit whose unit returns the outputs, `HmRow.regions` ends with region "out" (`Circuit.outRegion`,
`regions_out`): for each unit slot `k`, the 16-bit returned outputs of the slot's leaf, packed into the output group's
words (`outPack`, `pack_bit`: port `j`'s bit `t` is bit `16 (j % 8) + t` of word `j / 8`), every word's 16 bytes in
order (`outRegion_bytes`). Its free bits are a slot's `7 + log₂ words` column bits and the slot's `log₂ count` bits
above the unit range's slot bits, at the output group's first column (`inBlock_out`). So in a satisfying witness of the
    10|model's statement at the executable's regions, the bit at slot `k`'s output column `16 j + t` is bit `t` of output `j`
of the slot's leaf at the slot's instance (the pinned dummy past the population; `out_pinned`).
-/

set_option Elab.async false

namespace FlockSoundness.Discharge.Layout

open Flock FlockLevel3 Model

/-! ### Region "out" -/

set_option maxHeartbeats 2000000 in
/-- **`HmRow.regions` ends with region "out"** in an untyped circuit whose unit returns the outputs. -/
theorem regions_out {c : Circuit} {pub : Public} {regs : Array Region} (h : HmRow.regions c pub = .ok regs)
    (ho : c.outNet = c.unitNet) (ht : c.typed = none) : ∃ r ∈ regs.toList, c.outRegion pub = .ok r := by
  unfold HmRow.regions at h
  dsimp only at h
  try simp only [Refine.throw_bind', pure_bind] at h
  obtain ⟨out, -, h⟩ := Refine.ebind_ok h
  split at h
  · rename_i hc
    simp [ho, ht] at hc
  · obtain ⟨r, hr, h⟩ := Refine.ebind_ok h
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact ⟨r, by simp, hr⟩

/-! ### Its bytes -/

/-- Output `j` of unit slot `k`'s leaf at block `b`, as region "out" packs it: the slot's instance's, or the pinned
dummy past the population. -/
def outVal (c : Circuit) (pub : Public) (b k j : ℕ) : ℕ :=
  match c.instOf pub.n b (k / c.upv) with
  | some i => pub.outputs.getD (i * c.outWords + (c.leavesOut.getD (k % c.upv) #[])[j]!) 0
  | none => c.dummyOutputs.getD (c.leavesOut.getD (k % c.upv) #[])[j]! 0

/-- Port `j`'s width in the unit's returned-output group. -/
def outBits (c : Circuit) (j : ℕ) : ℕ := (c.unit.outGroups.getD 0 default).bits.getD j 0

/-- One step of `Circuit.outRegion`'s packing: value `vals j` ORed in at column `col` (word `col / 128`, bit
`col % 128`), and the column moved on by port `j`'s width `bits j`. -/
def packStep (vals bits : ℕ → ℕ) (s : Array F128 × ℕ) (j : ℕ) : Array F128 × ℕ :=
  (s.1.set! (s.2 / 128) (if s.2 % 128 < 64 then
      { lo := s.1[s.2 / 128]!.lo ||| (vals j).toUInt64 <<< (s.2 % 128).toUInt64, hi := s.1[s.2 / 128]!.hi }
    else { lo := s.1[s.2 / 128]!.lo,
           hi := s.1[s.2 / 128]!.hi ||| (vals j).toUInt64 <<< (s.2 % 128 - 64).toUInt64 }),
   s.2 + bits j)

/-- Unit slot `k`'s words at block `b`: its leaf's outputs packed into the output group's words. -/
def outPack (c : Circuit) (pub : Public) (b k : ℕ) : Array F128 :=
  ((List.range (c.leavesOut.getD (k % c.upv) #[]).size).foldl (packStep (outVal c pub b k) (outBits c))
    (Array.replicate (c.unit.outGroups.getD 0 default).words 0, 0)).1

/-- Unit slot `k`'s bytes at block `b`: its words' 16 bytes each, in order. -/
def slotBytes (c : Circuit) (pub : Public) (b k : ℕ) : List UInt8 :=
  ((outPack c pub b k).toList.map fun x => x.toBytes.data.toList).flatten

/-- Region "out"'s bytes at block `b`: every unit slot's, slot by slot. -/
def outBytes (c : Circuit) (pub : Public) (b : ℕ) : List UInt8 :=
  ((List.range c.unitRange.count).map (slotBytes c pub b)).flatten

theorem foldl_pushBytes (L : List F128) (A : ByteArray) :
    (L.foldl F128.pushBytes A).data.toList = A.data.toList ++ (L.map fun x => x.toBytes.data.toList).flatten := by
  induction L generalizing A with
  | nil => simp
  | cons x L ih =>
    rw [List.foldl_cons, ih]
    simp [F128.pushBytes, ByteArray.data_append]

theorem foldl_slots (f : ℕ → Array F128) (L : List ℕ) (A : ByteArray) :
    (L.foldl (fun A k => (f k).toList.foldl F128.pushBytes A) A).data.toList =
      A.data.toList ++ (L.map fun k => ((f k).toList.map fun x => x.toBytes.data.toList).flatten).flatten := by
  induction L generalizing A with
  | nil => simp
  | cons k L ih =>
    rw [List.foldl_cons, ih, foldl_pushBytes]
    simp

/-- **Region "out"'s bytes**, as `Circuit.outRegion` packs them: every unit slot's words (`outPack`), slot by slot. -/
theorem outRegion_bytes {c : Circuit} {pub : Public} {r : Region} (h : c.outRegion pub = .ok r) (b : ℕ) :
    (r.bytes b).data.toList = outBytes c pub b := by
  unfold Circuit.outRegion at h
  dsimp only at h
  obtain ⟨rfl, -⟩ := mkRegion_spec h
  clear h
  dsimp only
  simp only [Std.Legacy.Range.forIn_eq_forIn_range', List.forIn_pure_yield_eq_foldl, Array.forIn_pure_yield_eq_foldl,
    Id.run, bind_pure_comp, pure_bind, Std.Legacy.Range.size, Nat.sub_zero, Nat.add_sub_cancel, Nat.div_one,
    List.range'_eq_map_range, map_pure, Nat.zero_add, List.map_id']
  simp only [← Array.foldl_toList]
  change (List.foldl (fun A k => (outPack c pub b k).toList.foldl F128.pushBytes A) ByteArray.empty
    (List.range c.unitRange.count)).data.toList = _
  rw [foldl_slots]
  rfl

/-! ### The packing -/

/-- Bit `p < 128` of a 128-bit word: bit `p` of `lo` below 64, bit `p - 64` of `hi` above. -/
def f128Bit (x : F128) (p : ℕ) : Bool := if p < 64 then x.lo.toNat.testBit p else x.hi.toNat.testBit (p - 64)

theorem toNat_or_shift (x : UInt64) {v sh : ℕ} (hv : v < 2 ^ 16) (hsh : sh + 16 ≤ 64) :
    (x ||| v.toUInt64 <<< sh.toUInt64).toNat = x.toNat ||| v <<< sh := by
  have h1 : v.toUInt64.toNat = v := by
    rw [Nat.toUInt64, UInt64.toNat_ofNat']; exact Nat.mod_eq_of_lt (by omega)
  have h2 : sh.toUInt64.toNat % 64 = sh := by
    rw [Nat.toUInt64, UInt64.toNat_ofNat']; omega
  rw [UInt64.toNat_or, UInt64.toNat_shiftLeft, h1, h2, Nat.mod_eq_of_lt]
  rw [Nat.shiftLeft_eq]
  calc v * 2 ^ sh < 2 ^ 16 * 2 ^ sh := Nat.mul_lt_mul_of_pos_right hv (Nat.two_pow_pos _)
    _ = 2 ^ (16 + sh) := by rw [pow_add]
    _ ≤ 2 ^ 64 := Nat.pow_le_pow_right (by norm_num) (by omega)

theorem testBit_or_shift (x v sh p : ℕ) (hv : v < 2 ^ 16) :
    (x ||| v <<< sh).testBit p = (x.testBit p || (decide (sh ≤ p) && decide (p < sh + 16) && v.testBit (p - sh))) := by
  rw [Nat.testBit_or, Nat.testBit_shiftLeft]
  by_cases h1 : sh ≤ p
  · by_cases h2 : p < sh + 16
    · simp [h1, h2]
    · have : v.testBit (p - sh) = false :=
        Nat.testBit_lt_two_pow (lt_of_lt_of_le hv (Nat.pow_le_pow_right (by norm_num) (by omega)))
      simp [h1, h2, this]
  · simp [h1]

/-- **What the first `J` packing steps leave**: bit `p` of word `wi` is bit `(128 wi + p) % 16` of value
`(128 wi + p) / 16` below column `16 J`, and 0 from there. -/
def Packed (vals : ℕ → ℕ) (J : ℕ) (w : Array F128) : Prop :=
  ∀ wi < w.size, ∀ p < 128, f128Bit w[wi]! p =
    (decide (128 * wi + p < 16 * J) && (vals ((128 * wi + p) / 16)).testBit ((128 * wi + p) % 16))

theorem packed_zero (vals : ℕ → ℕ) (words : ℕ) : Packed vals 0 (Array.replicate words 0) := by
  intro wi hwi p hp
  have : (Array.replicate words (0 : F128))[wi]! = 0 := by
    rw [getElem!_pos _ _ (by simpa using hwi)]; simp
  rw [this]
  have h0 : (0 : F128) = ⟨0, 0⟩ := rfl
  simp [f128Bit, h0]

/-- The bits a step ORs in: below the column, the old bits; at the column's 16 bits, the value's; above, none. -/
theorem pack_bits (old x : Bool) (q c0 v : ℕ) (hold : old = (decide (q < c0) && x))
    (hX : c0 ≤ q → q < c0 + 16 → x = v.testBit (q - c0)) :
    (old || (decide (c0 ≤ q) && decide (q < c0 + 16) && v.testBit (q - c0))) = (decide (q < c0 + 16) && x) := by
  subst hold
  by_cases h1 : q < c0
  · have h2 : ¬ c0 ≤ q := by omega
    have h3 : q < c0 + 16 := by omega
    simp [h1, h2, h3]
  · by_cases h3 : q < c0 + 16
    · have h2 : c0 ≤ q := by omega
      simp [h1, h2, h3, hX h2 h3]
    · simp [h1, h3]

theorem packed_step {vals bits : ℕ → ℕ} {J : ℕ} {w : Array F128} (hP : Packed vals J w)
    (hJ : 16 * J < 128 * w.size) (hv : vals J < 2 ^ 16) (hb : bits J = 16) :
    (packStep vals bits (w, 16 * J) J).1.size = w.size ∧ (packStep vals bits (w, 16 * J) J).2 = 16 * (J + 1) ∧
      Packed vals (J + 1) (packStep vals bits (w, 16 * J) J).1 := by
  refine ⟨by simp [packStep], by simp [packStep, hb]; ring, ?_⟩
  intro wi hwi p hp
  have hsz : (packStep vals bits (w, 16 * J) J).1.size = w.size := by simp [packStep]
  rw [hsz] at hwi
  have hwi0 : 16 * J / 128 < w.size := by omega
  have hdm : 128 * (16 * J / 128) + 16 * J % 128 = 16 * J := Nat.div_add_mod _ _
  have hsh16 : 16 * J % 128 % 16 = 0 := by omega
  rw [getElem!_pos _ _ (by rw [hsz]; exact hwi)]
  simp only [packStep, Array.set!_eq_setIfInBounds]
  rw [Array.getElem_setIfInBounds hwi]
  have hold := hP wi hwi p hp
  rw [getElem!_pos w wi hwi] at hold
  split
  · rename_i hw
    subst hw
    have hX : 16 * J ≤ 128 * (16 * J / 128) + p → 128 * (16 * J / 128) + p < 16 * J + 16 →
        (vals ((128 * (16 * J / 128) + p) / 16)).testBit ((128 * (16 * J / 128) + p) % 16) =
          (vals J).testBit (128 * (16 * J / 128) + p - 16 * J) := by
      intro h1 h2
      rw [show (128 * (16 * J / 128) + p) / 16 = J by omega,
        show (128 * (16 * J / 128) + p) % 16 = 128 * (16 * J / 128) + p - 16 * J by omega]
    rw [getElem!_pos w _ hwi]
    split
    · rename_i hlo
      unfold f128Bit at hold ⊢
      split
      · rename_i hp64
        simp only [hp64, ↓reduceIte] at hold
        rw [toNat_or_shift _ hv (by omega), testBit_or_shift _ _ _ _ hv, hold,
          show 16 * (J + 1) = 16 * J + 16 by ring]
        have := pack_bits _ _ (128 * (16 * J / 128) + p) (16 * J) (vals J) rfl hX
        simp only [show (16 * J % 128 ≤ p) = (16 * J ≤ 128 * (16 * J / 128) + p) by
            apply propext; omega,
          show (p < 16 * J % 128 + 16) = (128 * (16 * J / 128) + p < 16 * J + 16) by apply propext; omega,
          show p - 16 * J % 128 = 128 * (16 * J / 128) + p - 16 * J by omega]
        exact this
      · rename_i hp64
        simp only [hp64, ↓reduceIte] at hold
        rw [hold]
        have h1 : ¬ 128 * (16 * J / 128) + p < 16 * J := by omega
        have h2 : ¬ 128 * (16 * J / 128) + p < 16 * (J + 1) := by omega
        simp [h1, h2]
    · rename_i hlo
      unfold f128Bit at hold ⊢
      split
      · rename_i hp64
        simp only [hp64, ↓reduceIte] at hold
        rw [hold]
        have h1 : 128 * (16 * J / 128) + p < 16 * J := by omega
        have h2 : 128 * (16 * J / 128) + p < 16 * (J + 1) := by omega
        simp [h1, h2]
      · rename_i hp64
        simp only [hp64, ↓reduceIte] at hold
        rw [toNat_or_shift _ hv (by omega), testBit_or_shift _ _ _ _ hv, hold,
          show 16 * (J + 1) = 16 * J + 16 by ring]
        have := pack_bits _ _ (128 * (16 * J / 128) + p) (16 * J) (vals J) rfl hX
        simp only [show (16 * J % 128 - 64 ≤ p - 64) = (16 * J ≤ 128 * (16 * J / 128) + p) by
            apply propext; omega,
          show (p - 64 < 16 * J % 128 - 64 + 16) = (128 * (16 * J / 128) + p < 16 * J + 16) by apply propext; omega,
          show p - 64 - (16 * J % 128 - 64) = 128 * (16 * J / 128) + p - 16 * J by omega]
        exact this
  · rename_i hw
    rw [hold]
    have : (128 * wi + p < 16 * J) ↔ (128 * wi + p < 16 * (J + 1)) := by omega
    simp only [this]

/-- **The packing's words**: after the first `J` of `N` ports, each 16 bits wide, with values below `2^16`, that fit in
`words` words, the words are `words` and hold the first `J` ports' values (`Packed`). -/
theorem pack_spec {vals bits : ℕ → ℕ} {N words : ℕ} (hb : ∀ j < N, bits j = 16) (hv : ∀ j < N, vals j < 2 ^ 16)
    (hfit : 16 * N ≤ 128 * words) (J : ℕ) (hJ : J ≤ N) :
    ((List.range J).foldl (packStep vals bits) (Array.replicate words 0, 0)).1.size = words ∧
      ((List.range J).foldl (packStep vals bits) (Array.replicate words 0, 0)).2 = 16 * J ∧
      Packed vals J ((List.range J).foldl (packStep vals bits) (Array.replicate words 0, 0)).1 := by
  induction J with
  | zero => exact ⟨by simp, rfl, packed_zero vals words⟩
  | succ J ih =>
    obtain ⟨h1, h2, h3⟩ := ih (by omega)
    rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil]
    generalize (List.range J).foldl (packStep vals bits) (Array.replicate words 0, 0) = s at h1 h2 h3
    obtain ⟨w, col⟩ := s
    simp only at h1 h2 h3
    subst h2
    obtain ⟨e1, e2, e3⟩ := packed_step h3 (by rw [h1]; omega) (hv J (by omega)) (hb J (by omega))
    exact ⟨e1.trans h1, e2, e3⟩

/-- **Port `j`'s bit `t` in the packed words**: bit `16 (j % 8) + t` of word `j / 8` is bit `t` of its value. -/
theorem pack_bit {vals bits : ℕ → ℕ} {N words : ℕ} (hb : ∀ j < N, bits j = 16) (hv : ∀ j < N, vals j < 2 ^ 16)
    (hfit : 16 * N ≤ 128 * words) {j t : ℕ} (hj : j < N) (ht : t < 16) :
    j / 8 < ((List.range N).foldl (packStep vals bits) (Array.replicate words 0, 0)).1.size ∧
      f128Bit ((List.range N).foldl (packStep vals bits) (Array.replicate words 0, 0)).1[j / 8]! (16 * (j % 8) + t) =
        (vals j).testBit t := by
  obtain ⟨h1, -, h3⟩ := pack_spec hb hv hfit N le_rfl
  have hw : j / 8 < ((List.range N).foldl (packStep vals bits) (Array.replicate words 0, 0)).1.size := by
    rw [h1]; omega
  refine ⟨hw, ?_⟩
  rw [h3 _ hw _ (by omega)]
  have e1 : 128 * (j / 8) + (16 * (j % 8) + t) < 16 * N := by omega
  have e2 : (128 * (j / 8) + (16 * (j % 8) + t)) / 16 = j := by omega
  have e3 : (128 * (j / 8) + (16 * (j % 8) + t)) % 16 = t := by omega
  simp [e1, e2, e3]

/-- **A word's bytes' bits**: bit `t < 8` of byte `bi < 16` of `F128.toBytes x` is bit `8 bi + t` of `x`. -/
theorem toBytes_bit (x : F128) {bi t : ℕ} (hbi : bi < 16) (ht : t < 8) :
    (x.toBytes.data.toList.getD bi default).toNat.testBit t = f128Bit x (8 * bi + t) := by
  have hl : bi < x.toBytes.data.toList.length := by simp [F128.toBytes, hbi]
  rw [List.getD_eq_getElem _ _ hl]
  simp only [F128.toBytes, Array.toList_ofFn, List.getElem_ofFn, f128Bit]
  split
  · rename_i h8
    have : 8 * bi + t < 64 := by omega
    simp only [this, ↓reduceIte, UInt64.toNat_toUInt8, UInt64.toNat_shiftRight, Nat.testBit_mod_two_pow,
      Nat.testBit_shiftRight]
    have hs : (8 * bi).toUInt64.toNat % 64 = 8 * bi := by rw [Nat.toUInt64, UInt64.toNat_ofNat']; omega
    rw [hs]
    simp [ht]
  · rename_i h8
    have : ¬ 8 * bi + t < 64 := by omega
    simp only [this, ↓reduceIte, UInt64.toNat_toUInt8, UInt64.toNat_shiftRight, Nat.testBit_mod_two_pow,
      Nat.testBit_shiftRight]
    have hs : (8 * (bi - 8)).toUInt64.toNat % 64 = 8 * (bi - 8) := by rw [Nat.toUInt64, UInt64.toNat_ofNat']; omega
    rw [hs, show 8 * (bi - 8) + t = 8 * bi + t - 64 by omega]
    simp [ht]

/-! ### The byte region "out" has there -/

theorem foldl_packStep_size (vals bits : ℕ → ℕ) : ∀ (L : List ℕ) (s : Array F128 × ℕ),
    (L.foldl (packStep vals bits) s).1.size = s.1.size
  | [], _ => rfl
  | j :: L, s => by
      rw [List.foldl_cons, foldl_packStep_size vals bits L]
      simp [packStep]

theorem outPack_size (c : Circuit) (pub : Public) (b k : ℕ) :
    (outPack c pub b k).size = (c.unit.outGroups.getD 0 default).words := by
  rw [outPack, foldl_packStep_size]
  simp

theorem length_flatten_toBytes : ∀ L : List F128,
    (L.map fun x => x.toBytes.data.toList).flatten.length = 16 * L.length
  | [] => rfl
  | x :: L => by
      rw [List.map_cons, List.flatten_cons, List.length_append, length_flatten_toBytes L, List.length_cons]
      simp [F128.toBytes]
      ring

theorem slotBytes_length (c : Circuit) (pub : Public) (b k : ℕ) :
    (slotBytes c pub b k).length = 16 * (c.unit.outGroups.getD 0 default).words := by
  rw [slotBytes, length_flatten_toBytes, Array.length_toList, outPack_size]

theorem getD_flatten_const {α : Type} {B : ℕ} (d : α) : ∀ {L : List (List α)} (g j : ℕ),
    (∀ l ∈ L, l.length = B) → g < L.length → j < B → L.flatten.getD (g * B + j) d = (L.getD g []).getD j d
  | [], g, j, _, hg, _ => by simp at hg
  | l :: L, g, j, hL, hg, hj => by
      have hl : l.length = B := hL l List.mem_cons_self
      rw [List.flatten_cons]
      cases g with
      | zero =>
          simp only [zero_mul, zero_add, List.getD_cons_zero]
          rw [List.getD_eq_getElem?_getD, List.getElem?_append_left (by omega), ← List.getD_eq_getElem?_getD]
      | succ g =>
          rw [List.getD_eq_getElem?_getD, List.getElem?_append_right (by rw [hl, Nat.succ_mul]; omega), hl,
            show (g + 1) * B + j - B = g * B + j by rw [Nat.succ_mul]; omega, ← List.getD_eq_getElem?_getD,
            List.getD_cons_succ]
          exact getD_flatten_const d g j (fun l' hl' => hL l' (List.mem_cons_of_mem _ hl')) (by simp at hg; omega)
            hj

/-- **Region "out"'s bit at slot `k`'s port `j`, bit `t`**: bit `t % 8` of byte `k·16·words + 2j + t / 8` of its bytes is
bit `t` of the slot's output `j`, when the slot's leaf's outputs are 16-bit ports (`hb`) of 16-bit values (`hv`) that
fit in the output group's words (`hfit`). -/
theorem outBytes_bit {c : Circuit} {pub : Public} {b k j t : ℕ} (hk : k < c.unitRange.count)
    (hb : ∀ j < (c.leavesOut.getD (k % c.upv) #[]).size, outBits c j = 16)
    (hv : ∀ j < (c.leavesOut.getD (k % c.upv) #[]).size, outVal c pub b k j < 2 ^ 16)
    (hfit : 16 * (c.leavesOut.getD (k % c.upv) #[]).size ≤ 128 * (c.unit.outGroups.getD 0 default).words)
    (hj : j < (c.leavesOut.getD (k % c.upv) #[]).size) (ht : t < 16) :
    ((outBytes c pub b).getD (k * (16 * (c.unit.outGroups.getD 0 default).words) + (2 * j + t / 8)) default).toNat.testBit
      (t % 8) = (outVal c pub b k j).testBit t := by
  have hW : 2 * j + t / 8 < 16 * (c.unit.outGroups.getD 0 default).words := by omega
  have hslot : ((List.range c.unitRange.count).map (slotBytes c pub b)).getD k [] = slotBytes c pub b k := by
    rw [List.getD_eq_getElem _ _ (by simpa using hk)]
    simp
  rw [outBytes, getD_flatten_const default k _ (fun l hl => by
      obtain ⟨k', -, rfl⟩ := List.mem_map.1 hl
      exact slotBytes_length c pub b k') (by simpa using hk) hW, hslot, slotBytes,
    show 2 * j + t / 8 = j / 8 * 16 + (2 * (j % 8) + t / 8) by omega]
  have hsz : j / 8 < (outPack c pub b k).size := by rw [outPack_size]; omega
  have hword : ((outPack c pub b k).toList.map fun x => x.toBytes.data.toList).getD (j / 8) [] =
      (outPack c pub b k)[j / 8]!.toBytes.data.toList := by
    rw [List.getD_eq_getElem _ _ (by simpa using hsz), getElem!_pos (outPack c pub b k) (j / 8) hsz]
    simp
  rw [getD_flatten_const default (j / 8) _ (fun l hl => by
      obtain ⟨x, -, rfl⟩ := List.mem_map.1 hl
      simp [F128.toBytes]) (by simpa using hsz) (by omega), hword,
    toBytes_bit _ (by omega) (Nat.mod_lt _ (by norm_num)),
    show 8 * (2 * (j % 8) + t / 8) + t % 8 = 16 * (j % 8) + t by omega]
  exact (pack_bit hb hv hfit hj ht).2

/-! ### Where region "out" pins them -/

/-- A unit slot's column bits in region "out": `7 + log₂ words` of the returned-output group. -/
def outLw (c : Circuit) : ℕ := 7 + (c.unit.outGroups.getD 0 default).words.log2

/-- The in-block address region "out" pins bit `t` of port `j` of unit slot `k` at: column `16 j + t` of the slot's
returned-output group. -/
def outAddr (c : Circuit) (k j t : ℕ) : ℕ :=
  c.unitRange.start + (c.unit.outGroups.getD 0 default).col * WORD + (16 * j + t) + k * 2 ^ c.unitRange.slotLog

/-- Region bit `k·2^outLw + m` (slot `k`'s column `m`) of region "out" sits at the slot's column `m`. -/
theorem inBlock_out {c : Circuit} {pub : Public} {r : Region} (nbl : ℕ) (h : c.outRegion pub = .ok r) {k m : ℕ}
    (hk : k < 2 ^ c.unitRange.count.log2) (hm : m < 2 ^ outLw c) :
    (Refine.regionOf c.kLog nbl r).inBlock (k * 2 ^ outLw c + m) =
      c.unitRange.start + (c.unit.outGroups.getD 0 default).col * WORD + m + k * 2 ^ c.unitRange.slotLog := by
  unfold Circuit.outRegion at h
  dsimp only at h
  rw [inBlock_mkRegion h]
  simp only [Array.toList_append, Array.toList_map, Array.toList_range, spread_append, spread_range,
    List.length_map, List.length_range, pow_zero, mul_one]
  have hP : 0 < 2 ^ outLw c := Nat.two_pow_pos _
  have h1 : (k * 2 ^ outLw c + m) % 2 ^ outLw c = m := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hm]
  have h2 : (k * 2 ^ outLw c + m) / 2 ^ outLw c = k := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hP, Nat.div_eq_of_lt hm, zero_add]
  show (k * 2 ^ outLw c + m) % 2 ^ outLw c + (k * 2 ^ outLw c + m) / 2 ^ outLw c % 2 ^ c.unitRange.count.log2 *
      2 ^ c.unitRange.slotLog + (c.unitRange.start + (c.unit.outGroups.getD 0 default).col * WORD) = _
  rw [h1, h2, Nat.mod_eq_of_lt hk]
  ring

/-- **The returned outputs a satisfying witness holds at the unit slots' output columns** are the ones the statement
pins: in an untyped circuit whose unit returns the outputs, at block `o`, unit slot `k` (below `2^⌊log₂ count⌋`, all of
them when the unit range's count is a power of two), the bit at column `16 j + t` of the slot's returned-output group
(`outAddr`) is bit `t` of output `j` of the slot's leaf at the slot's instance (the pinned dummy past the population),
when the group's words are a power of two (`hw`) and the leaf's outputs are 16-bit ports (`hb`) of 16-bit values (`hv`)
that fit in them (`hfit`). It holds in the model's statement at setupH's layout whose regions contain the executable's
(`hregs`). -/
theorem out_pinned {st : Stmt} (L : FlockSoundness.Layout st) (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog)))
    (hregs : ∀ r ∈ st.regions.toList, Refine.regionOf st.c.kLog (st.m - st.c.kLog) r ∈ regs)
    (hR : HmRow.regions st.c st.pub = .ok st.regions) (ho : st.c.outNet = st.c.unitNet) (hty : st.c.typed = none)
    {z : Witness (L.model regs).m} (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl))
    {k j t : ℕ} (hk : k < st.c.unitRange.count) (hk2 : k < 2 ^ st.c.unitRange.count.log2)
    (hw : (st.c.unit.outGroups.getD 0 default).words = 2 ^ (st.c.unit.outGroups.getD 0 default).words.log2)
    (hb : ∀ j < (st.c.leavesOut.getD (k % st.c.upv) #[]).size, outBits st.c j = 16)
    (hv : ∀ j < (st.c.leavesOut.getD (k % st.c.upv) #[]).size, outVal st.c st.pub o.val k j < 2 ^ 16)
    (hfit : 16 * (st.c.leavesOut.getD (k % st.c.upv) #[]).size ≤ 128 * (st.c.unit.outGroups.getD 0 default).words)
    (hj : j < (st.c.leavesOut.getD (k % st.c.upv) #[]).size) (ht : t < 16) :
    bit ((L.model regs).block z o (inBlockOr st.c.kLog (outAddr st.c k j t))) =
      (outVal st.c st.pub o.val k j).testBit t := by
  obtain ⟨r, hr, hmk⟩ := regions_out hR ho hty
  have hmem : Refine.regionOf (L.model regs).kLog ((L.model regs).m - (L.model regs).kLog) r ∈
      (L.model regs).regions := hregs r hr
  have hmk' := hmk
  unfold Circuit.outRegion at hmk'
  dsimp only at hmk'
  obtain ⟨hrf, -, -, hnd, -, hlt⟩ := mkRegion_spec hmk'
  have hfreeA := congrArg Flock.Region.free hrf
  simp only at hfreeA
  have hlen : (Refine.regionOf st.c.kLog (st.m - st.c.kLog) r).free.length =
      outLw st.c + st.c.unitRange.count.log2 := by
    rw [← List.length_map (f := Fin.val), regionOf_free_map _ _ _ (fun b hb => hlt b (by
      rw [hfreeA] at hb; exact Array.mem_toList_iff.1 hb)), hfreeA]
    simp [outLw]
  have hP : 2 ^ outLw st.c = 128 * (st.c.unit.outGroups.getD 0 default).words := by
    rw [outLw, pow_add, ← hw]
    norm_num
  have hm : 16 * j + t < 2 ^ outLw st.c := by rw [hP]; omega
  have hi : k * 2 ^ outLw st.c + (16 * j + t) <
      2 ^ (Refine.regionOf st.c.kLog (st.m - st.c.kLog) r).free.length := by
    rw [hlen, pow_add]
    calc k * 2 ^ outLw st.c + (16 * j + t) < k * 2 ^ outLw st.c + 2 ^ outLw st.c := by omega
      _ = (k + 1) * 2 ^ outLw st.c := by ring
      _ ≤ 2 ^ st.c.unitRange.count.log2 * 2 ^ outLw st.c := Nat.mul_le_mul_right _ hk2
      _ = 2 ^ outLw st.c * 2 ^ st.c.unitRange.count.log2 := by ring
  have hib : (Refine.regionOf st.c.kLog (st.m - st.c.kLog) r).inBlock (k * 2 ^ outLw st.c + (16 * j + t)) =
      outAddr st.c k j t :=
    inBlock_out (st.m - st.c.kLog) hmk hk2 hm
  have hnd' : (Refine.regionOf st.c.kLog (st.m - st.c.kLog) r).free.Nodup := by
    unfold Refine.regionOf
    rw [hfreeA]
    exact Refine.filterMap_fin_nodup _ _ hnd
  have ha : (Refine.regionOf st.c.kLog (st.m - st.c.kLog) r).inBlock (k * 2 ^ outLw st.c + (16 * j + t)) <
      2 ^ st.c.kLog :=
    inBlock_lt _ hnd' _
  have hread := region_read hz hmem o _ hi ha
  have hfin : inBlockOr st.c.kLog (outAddr st.c k j t) = ⟨_, ha⟩ := by
    simp only [inBlockOr, ← hib, ha, ↓reduceDIte]
  rw [hfin]
  refine hread.trans ?_
  have e1 : k * 2 ^ outLw st.c = 8 * (k * (16 * (st.c.unit.outGroups.getD 0 default).words)) := by rw [hP]; ring
  rw [get!_eq_getD, outRegion_bytes hmk, e1,
    show (8 * (k * (16 * (st.c.unit.outGroups.getD 0 default).words)) + (16 * j + t)) / 8 =
      k * (16 * (st.c.unit.outGroups.getD 0 default).words) + (2 * j + t / 8) by omega,
    show (8 * (k * (16 * (st.c.unit.outGroups.getD 0 default).words)) + (16 * j + t)) % 8 = t % 8 by omega]
  exact outBytes_bit hk hb hv hfit hj ht

end FlockSoundness.Discharge.Layout
