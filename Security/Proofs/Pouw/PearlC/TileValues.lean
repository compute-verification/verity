import Proofs.Pouw.PearlC.TileCircuit
import Proofs.Pouw.PearlC.PeelExactProofsP
import Definitions.Pouw.PearlC.Device

/-!
# The tile unit computes the ticket and the useful value (proofs)

L3's step 5 for the tile template: the words `TileCircuit.tile_correct` puts on the tile unit's outputs read back as
the values Pearl-C's protocol names at the record `devAt p 4` (`tile_values`, `tile_devAt`). Given row `i` of `A′`,
row `j` of `B̃` and the factor rows U's pinned order forms from them (`PeelExactProofsP.peelRowsP`), `C̃_ij`'s word has
the ticket's value `Accum.chainAt p 4` and `U_ij`'s the useful value `peelExactValP p 4`, at every entry of every tile.
`C̃_ij`'s needs only the chain to be finite (`tile_ticket`).

* **The chain** (`chainWordP_value`): `chainWordP? p G`'s word reads back as `Accum.chainP p G`, for every step `p` and
  period `G`. A window's word is its group's (`windowWord_group`), and a promotion on words is `rnAdd` on values
  (`promoteWord_wordQ`): FP32 values lie on the grid `2^−149`, where a finite `f32Word y` has the value `rn y`
  (`f32Word_value`), exactly below `2^−126` and rounded to 24 bits above it, while `rn` has no exponent range.
* **U** (`wordQ_uFold`): the template's four K16 steps run on any words. A step on a NaN or infinite accumulator writes
  NaN or an infinity (`k16Step_nonfin`), so their word reads back as `uWord?`'s, `none` once a step leaves the finite
  range, as the useful value does.
-/

namespace Pouw.PearlC.TileCircuit

open Pouw.Fp8Atom Pouw.Fp8Atom.Proofs Pouw.PearlC Pouw.PearlC.PeelExact Pouw.PearlC.PeelExactP

/-! ## FP32 words and their values -/

section Words

theorem wordQ_of_fin {w : ℕ} {t : Term} (h : fp32Value w = .fin t) : wordQ w = some t.value := by
  unfold wordQ; rw [h]; rfl

theorem fp32Value_nonfin {w : ℕ} (h : (w >>> 23) % 256 = 255) (t : Term) : fp32Value w ≠ .fin t := by
  intro ht
  unfold fp32Value at ht
  simp only [h, ↓reduceIte] at ht
  split_ifs at ht

theorem wordQ_nonfin {w : ℕ} (h : (w >>> 23) % 256 = 255) : wordQ w = none := by
  unfold wordQ
  rcases hv : fp32Value w with _ | s | t
  · rfl
  · rfl
  · exact absurd hv (fp32Value_nonfin h t)

theorem roundEven_intCast (z : ℤ) : roundEven (z : ℚ) = z := by
  simp [roundEven]

/-- **`f32Word` on FP32's grid**: a finite word of a value on the grid `2^−149` has the value `rn y`. Below `2^−126`
such a value is exact in both; above, `f32Word` rounds to 24 bits as `rn` does. -/
theorem f32Word_value {y : ℚ} (hg : ∃ m : ℤ, y = m * 2 ^ (-149 : ℤ)) (hf : (f32Word y >>> 23) % 256 ≠ 255) :
    wordQ (f32Word y) = some (rn y) := by
  rcases eq_or_ne y 0 with rfl | hy0
  · rw [show f32Word 0 = 0 by simp [f32Word], Freeze.wordQ_zero, rn_exact ⟨0, 0, by norm_num, by simp⟩]
  have ha : 0 < |y| := abs_pos.2 hy0
  obtain ⟨hb1, hb2⟩ := binade_spec ha
  set E₀ := max (binade |y|) (-126) with hE₀
  set u : ℚ := 2 ^ (E₀ - 23) with hu_def
  have hu : 0 < u := two_zpow_pos _
  set R := roundEven (|y| / u) with hR
  have hR0 : 0 ≤ R := le_trans (Int.floor_nonneg.2 (div_nonneg ha.le hu.le)) (floor_le_roundEven _)
  have hR24 : R.toNat ≤ 2 ^ 24 := f32Word_N ha
  have hR23 : -126 ≤ binade |y| → 2 ^ 23 ≤ R := by
    intro h
    have hE : E₀ = binade |y| := max_eq_left h
    have h1 : (2 : ℚ) ^ 23 ≤ |y| / u := by
      rw [le_div_iff₀ hu, hu_def, hE, two_pow_23]; exact hb1
    have h2 : (2 ^ 23 : ℤ) ≤ ⌊|y| / u⌋ := Int.le_floor.2 (by exact_mod_cast h1)
    exact le_trans h2 (floor_le_roundEven _)
  set sb : ℕ := if y < 0 then 1 else 0 with hsb
  have hs1 : sb ≤ 1 := by rw [hsb]; split <;> omega
  have hs2 : (if y < 0 then 2 ^ 31 else 0) = sb * 2 ^ 31 := by rw [hsb]; split <;> simp
  obtain ⟨N₀, hN₀⟩ : ∃ n : ℕ, (n : ℤ) = R := ⟨R.toNat, Int.toNat_of_nonneg hR0⟩
  have hNt : R.toNat = N₀ := by omega
  have hR' : (R : ℚ) = N₀ := by rw [← hN₀]; norm_cast
  have hw : f32Word y =
      let E := if N₀ = 2 ^ 24 then E₀ + 1 else E₀
      let N := if N₀ = 2 ^ 24 then 2 ^ 23 else N₀
      if 127 < E then sb * 2 ^ 31 + 0x7F800000
      else if N < 2 ^ 23 then sb * 2 ^ 31 + N
      else sb * 2 ^ 31 + (E + 127).toNat * 2 ^ 23 + (N - 2 ^ 23) := by
    unfold f32Word
    simp only [ite_eq_right ha.ne', ← hE₀, ← hu_def, ← hR, hNt, hs2]
  have hinf : (sb * 2 ^ 31 + 0x7F800000) >>> 23 % 256 = 255 := by
    rw [Nat.shiftRight_eq_div_pow]; norm_num only; omega
  have key : wordQ (f32Word y) = some ((if (sb == 1) = true then -1 else 1) * (R * u)) := by
    rw [hw] at hf ⊢
    dsimp only at hf ⊢
    by_cases hc : N₀ = 2 ^ 24
    · simp only [hc, ite_true] at hf ⊢
      have h127 : ¬ (127 < E₀ + 1) := by
        intro h; rw [ite_eq_left h] at hf; exact hf hinf
      simp only [h127, ite_false, show ¬ (2 ^ 23 < 2 ^ 23) by omega, Nat.sub_self, add_zero]
      have hef : (E₀ + 1 + 127).toNat < 255 := by omega
      have hv := AtomErr.fp32Value_word sb _ 0 hs1 hef (by norm_num)
      rw [Nat.add_zero, ite_eq_right (by omega)] at hv
      rw [wordQ_of_fin hv]
      congr 1
      unfold Term.value
      have hx : (((E₀ + 1 + 127).toNat : ℕ) : ℤ) - 127 - 23 = (E₀ - 23) + 1 := by omega
      simp only [hx]
      rw [hR', hc, hu_def, zpow_add_one₀ two_ne_zero]
      push_cast; ring
    · simp only [hc, ite_false] at hf ⊢
      have h127 : ¬ (127 < E₀) := by
        intro h; rw [ite_eq_left h] at hf; exact hf hinf
      simp only [h127, ite_false]
      by_cases hsub : N₀ < 2 ^ 23
      · simp only [hsub, ite_true]
        have hbin : binade |y| < -126 := by
          by_contra h; have := hR23 (not_lt.1 h); omega
        have hE : E₀ = -126 := max_eq_right hbin.le
        have hv := AtomErr.fp32Value_word sb 0 N₀ hs1 (by norm_num) hsub
        rw [ite_eq_left rfl] at hv
        rw [show sb * 2 ^ 31 + N₀ = sb * 2 ^ 31 + 0 * 2 ^ 23 + N₀ by ring]
        by_cases h0 : N₀ = 0
        · rw [ite_eq_left h0] at hv
          rw [wordQ_of_fin hv, hR', h0]; simp [Term.value, Term.zero]
        · rw [ite_eq_right h0] at hv
          rw [wordQ_of_fin hv]
          congr 1
          unfold Term.value
          rw [hR', hu_def, hE]; norm_num
      · simp only [hsub, ite_false]
        have hef : (E₀ + 127).toNat < 255 := by omega
        have hm : N₀ - 2 ^ 23 < 2 ^ 23 := by omega
        rw [wordQ_of_fin (by rw [AtomErr.fp32Value_word sb _ _ hs1 hef hm, ite_eq_right (by omega)])]
        congr 1
        unfold Term.value
        rw [hR', show 2 ^ 23 + (N₀ - 2 ^ 23) = N₀ by omega, hu_def,
          show (((E₀ + 127).toNat : ℕ) : ℤ) - 127 - 23 = E₀ - 23 by omega]
        ring
  rw [key]
  congr 1
  have hsg : ((sb == 1) = true) ↔ y < 0 := by rw [hsb]; split <;> simp_all
  have hsgn : (if (sb == 1) = true then (-1 : ℚ) else 1) * |y| = y := by
    by_cases hneg : y < 0
    · rw [ite_eq_left (hsg.2 hneg), abs_of_neg hneg]; ring
    · rw [ite_eq_right (fun h => hneg (hsg.1 h)), abs_of_nonneg (not_lt.1 hneg)]; ring
  rcases le_or_gt (-126) (binade |y|) with hb | hb
  · have hE : E₀ = binade |y| := max_eq_left hb
    unfold rn ulp
    rw [hR, hu_def, hE]
    by_cases hneg : y < 0
    · rw [ite_eq_left (hsg.2 hneg), ite_eq_left hneg]; ring
    · rw [ite_eq_right (fun h => hneg (hsg.1 h)), ite_eq_right hneg]; ring
  · have hE : E₀ = -126 := max_eq_right hb.le
    obtain ⟨m, hm⟩ := hg
    have h149 : (0 : ℚ) < 2 ^ (-149 : ℤ) := two_zpow_pos _
    have hum : u = 2 ^ (-149 : ℤ) := by rw [hu_def, hE]; norm_num
    have hyu : |y| / u = ((|m| : ℤ) : ℚ) := by
      rw [hum, hm, abs_mul, abs_of_pos h149, mul_div_assoc, div_self h149.ne', mul_one]; push_cast; rfl
    have hRm : (R : ℚ) * u = |y| := by
      rw [hR, hyu, roundEven_intCast, ← hyu, div_mul_cancel₀ _ hu.ne']
    rw [hRm, hsgn]
    have hlt : |y| < 2 ^ (-126 : ℤ) := lt_of_lt_of_le hb2 (zpow_le_zpow_right₀ one_le_two (by omega))
    have hm23 : |m| < 2 ^ 24 := by
      have h1 : (|m| : ℚ) * 2 ^ (-149 : ℤ) < 2 ^ 23 * 2 ^ (-149 : ℤ) := by
        rw [show (2 : ℚ) ^ 23 * 2 ^ (-149 : ℤ) = 2 ^ (-126 : ℤ) by norm_num]
        rw [hm, abs_mul, abs_of_pos h149] at hlt
        exact_mod_cast hlt
      have h2 : (|m| : ℚ) < 2 ^ 23 := lt_of_mul_lt_mul_right h1 h149.le
      have h3 : |m| < 2 ^ 23 := by exact_mod_cast h2
      omega
    exact (rn_exact ⟨m, -149, hm23, hm⟩).symm

/-- **A promotion's value**: a finite promotion of two finite words is `rnAdd` of their values. -/
theorem promoteWord_wordQ {t w s : ℕ} {x y : ℚ} (hx : wordQ t = some x) (hy : wordQ w = some y)
    (h : promoteWord? t w = some s) : wordQ s = some (rnAdd x y) := by
  obtain ⟨s', hs', hss⟩ := PeelExactProofs.promoteWord_val h
  rw [hss]
  unfold addWord? at hs'
  rw [hx, hy] at hs'
  dsimp only at hs'
  obtain ⟨rfl, -⟩ := PeelExactProofs.finite?_eq_some hs'
  unfold rnAdd rnWord
  by_cases h0 : x + y = 0
  · rw [ite_eq_left h0, h0, rn_exact ⟨0, 0, by norm_num, by simp⟩]
    split_ifs
    · exact PeelExactProofs.wordQ_negZero
    · exact Freeze.wordQ_zero
  · rw [ite_eq_right h0]
    rw [rnWord, ite_eq_right h0] at hs'
    obtain ⟨m, hm⟩ := wordQ_grid hx
    obtain ⟨n, hn⟩ := wordQ_grid hy
    exact f32Word_value ⟨m + n, by rw [hm, hn]; push_cast; ring⟩ (AccumProofs.finite?_exp hs')

theorem windowWord_group (p : Pipeline) : ∀ (g : List AtomIn) (w0 w : ℕ),
    g.foldlM (fun w x => finite? (p.step w (List.ofFn x.1) (List.ofFn x.2))) w0 = some w →
      g.foldl (fun w x => p.step w (List.ofFn x.1) (List.ofFn x.2)) w0 = w
  | [], w0, w, h => by simpa using h
  | x :: xs, w0, w, h => by
    simp only [List.foldlM_cons, List.foldl_cons] at h ⊢
    obtain ⟨w1, h1, h⟩ := Option.bind_eq_some_iff.1 h
    obtain ⟨rfl, -⟩ := PeelExactProofs.finite?_eq_some h1
    exact windowWord_group p xs _ w h

/-- **Step 5, the chain**: the pinned order's word chain reads back as the value chain, `Accum.chainP`, the ticket of
the record `devAt p G`. -/
theorem chainWordP_value (p : Pipeline) (G : ℕ) (atoms : List AtomIn) {c : ℕ}
    (h : chainWordP? p G atoms = some c) : Accum.chainP p G atoms = wordQ c := by
  unfold chainWordP? at h
  unfold Accum.chainP Accum.totalP
  rw [← Freeze.wordQ_zero]
  suffices H : ∀ (gs : List (List AtomIn)) (t : ℕ),
      gs.foldlM (fun t g => (windowWordP? p g).bind (promoteWord? t)) t = some c →
        (gs.map (Accum.groupWordP p)).foldl promote (wordQ t) = wordQ c from H _ 0 h
  intro gs
  induction gs with
  | nil =>
    intro t ht
    simp only [List.foldlM_nil, pure, Option.some.injEq] at ht
    subst ht; rfl
  | cons g gs ih =>
    intro t ht
    simp only [List.foldlM_cons, List.map_cons, List.foldl_cons] at ht ⊢
    obtain ⟨t', ht', hrest⟩ := Option.bind_eq_some_iff.1 ht
    obtain ⟨w, hw, hp⟩ := Option.bind_eq_some_iff.1 ht'
    have hg : Accum.groupWordP p g = w := windowWord_group p g 0 w hw
    obtain ⟨s', hs', -⟩ := PeelExactProofs.promoteWord_val hp
    unfold addWord? at hs'
    rcases hx : wordQ t with _ | x
    · rw [hx] at hs'; simp at hs'
    rcases hy : wordQ w with _ | y
    · rw [hx, hy] at hs'; simp at hs'
    have e : promote (some x) (Accum.groupWordP p g) = wordQ t' := by
      rw [hg]; simp only [promote, hy]; exact (promoteWord_wordQ hx hy hp).symm
    rw [e]; exact ih t' hrest

end Words

/-! ## U on any accumulator -/

section U

theorem packValue_nonfin : ∀ v : Value, (∀ t, v ≠ .fin t) → (packValue v >>> 23) % 256 = 255
  | .nan, _ => by decide
  | .inf s, _ => by cases s <;> decide
  | .fin t, h => absurd rfl (h t)

/-- A K16 step on a NaN or infinite accumulator writes NaN or an infinity. -/
theorem k16Step_nonfin {acc : ℕ} (x y : Fin 16 → ℕ) (h : (acc >>> 23) % 256 = 255) :
    (k16Step acc x y >>> 23) % 256 = 255 := by
  unfold k16Step
  apply packValue_nonfin
  intro t ht
  unfold groupStepTotal at ht
  rcases hv : fp32Value acc with _ | s | t'
  · rw [hv] at ht; simp [Value.isNaN] at ht
  · rw [hv] at ht
    cases s <;> simp [Value.isNaN, Value.isInf] at ht <;> split_ifs at ht
  · exact fp32Value_nonfin h t' hv

theorem foldl_nonfin (pa pb : List ℕ) : ∀ (l : List ℕ) (c : ℕ), (c >>> 23) % 256 = 255 →
    (l.foldl (fun acc s => k16Step acc (slice16 pa s) (slice16 pb s)) c >>> 23) % 256 = 255
  | [], _, h => h
  | _ :: l, _, h => foldl_nonfin pa pb l _ (k16Step_nonfin _ _ h)

theorem foldl_wordQ (pa pb : List ℕ) : ∀ (l : List ℕ) (c : ℕ),
    wordQ (l.foldl (fun acc s => k16Step acc (slice16 pa s) (slice16 pb s)) c) =
      (l.foldlM (uStep? pa pb) c).bind wordQ
  | [], c => rfl
  | s :: l, c => by
    simp only [List.foldl_cons, List.foldlM_cons]
    rcases hf : uStep? pa pb c s with _ | c'
    · have he : (k16Step c (slice16 pa s) (slice16 pb s) >>> 23) % 256 = 255 := by
        unfold uStep? finite? at hf
        split_ifs at hf with h
        exact h
      rw [wordQ_nonfin (foldl_nonfin pa pb l _ he)]; rfl
    · have e : k16Step c (slice16 pa s) (slice16 pb s) = c' := (PeelExactProofs.finite?_eq_some hf).1.symm
      rw [e]; exact foldl_wordQ pa pb l c'

/-- **U's word on any accumulator** reads back as `uWord?`'s, `none` once a step leaves the finite range. -/
theorem wordQ_uFold (pa pb : List ℕ) (c : ℕ) : wordQ (uFold pa pb c) = (uWord? c pa pb).bind wordQ :=
  foldl_wordQ pa pb _ c

end U

/-! ## The tile unit's values -/

section Tile

variable (p : Pipeline)

/-- **The ticket**: a transcript whose one unit is correct, holding at entry `(ti, tj)`'s inputs row `i` of `A′` and row
`j` of `B̃`, has `C̃`'s word of value `Accum.chainAt p 4` at `(i, j)` wherever the pinned order's chain is finite, whether
or not U's factor rows exist. -/
theorem tile_ticket {nw m n TM TN : ℕ} (hnw : 0 < nw) (X : (tileSpec p (128 * nw) TM TN).toCircuit.Transcript)
    (hX : (tileSpec p (128 * nw) TM TN).part.Correct X 0)
    (Aq : Matrix (Fin m) (Fin (128 * nw)) Code) (Bq : Matrix (Fin n) (Fin (128 * nw)) Code) (i : Fin m) (j : Fin n)
    {ti tj : ℕ} (hti : ti < TM) (htj : tj < TN)
    (hA : ∀ l : Fin (128 * nw), val X (ti * (128 * nw) + l) = (Aq i l).val)
    (hB : ∀ l : Fin (128 * nw), val X (TM * (128 * nw) + tj * (128 * nw) + l) = (Bq j l).val) {c : ℕ}
    (hc : chainWordP? p 4 (rowAtoms (Aq i) (Bq j)) = some c) :
    wordQ (val X (cGate (128 * nw) TM TN ti tj)) = Accum.chainAt p 4 Aq Bq i j := by
  rw [tile_out p TM TN X hX _ (cGate_mem p TM TN _ hti htj), tileVal_chain p TM TN hnw X _ _ hti htj hA hB hc]
  exact (chainWordP_value p 4 _ hc).symm

/-- **The tile unit computes the ticket and the useful value** (L3's steps 4 and 5 for the tile template): a transcript
whose one unit is correct, holding at entry `(ti, tj)`'s inputs row `i` of `A′`, row `j` of `B̃` and the factor rows
`P_A,i`, `P_B,j` that U's pinned order forms from them (`peelRowsP`), has on entry `(ti, tj)`'s outputs `C̃`'s word of
value `Accum.chainAt p 4` and `U`'s of value `peelExactValP p 4` at `(i, j)`: the record `devAt p 4`'s ticket and
useful value (`tile_devAt`). -/
theorem tile_values {nw m n TM TN : ℕ} (hnw : 0 < nw) (X : (tileSpec p (128 * nw) TM TN).toCircuit.Transcript)
    (hX : (tileSpec p (128 * nw) TM TN).part.Correct X 0)
    (Aq : Matrix (Fin m) (Fin (128 * nw)) Code) (Bq : Matrix (Fin n) (Fin (128 * nw)) Code)
    (FA FB : Matrix (Fin (128 * nw)) (Fin 32) Code) (beA : Matrix (Fin m) (Fin 32) ℕ)
    (beB : Matrix (Fin n) (Fin 32) ℕ) (i : Fin m) (j : Fin n) {ti tj : ℕ} (hti : ti < TM) (htj : tj < TN)
    (hA : ∀ l : Fin (128 * nw), val X (ti * (128 * nw) + l) = (Aq i l).val)
    (hB : ∀ l : Fin (128 * nw), val X (TM * (128 * nw) + tj * (128 * nw) + l) = (Bq j l).val)
    {c : ℕ} {pa pb : List ℕ} (hr : PeelExactProofsP.peelRowsP p 4 Aq Bq FA FB beA beB i j = some (c, pa, pb))
    (hpa : ∀ q < 64, val X ((TM + TN) * (128 * nw) + 64 * ti + q) = pa.getD q 0)
    (hpb : ∀ q < 64, val X ((TM + TN) * (128 * nw) + 64 * TM + 64 * tj + q) = pb.getD q 0) :
    wordQ (val X (cGate (128 * nw) TM TN ti tj)) = Accum.chainAt p 4 Aq Bq i j ∧
      wordQ (val X (uGate (128 * nw) TM TN ti tj)) = peelExactValP p 4 Aq Bq FA FB beA beB i j := by
  obtain ⟨-, -, -, hc, -⟩ := PeelExactProofsP.peelRowsP_spec hr
  obtain ⟨-, hU⟩ := tile_correct p TM TN hnw X hX (Aq i) (Bq j) pa pb hti htj hA hB hpa hpb hc
  refine ⟨tile_ticket p hnw X hX Aq Bq i j hti htj hA hB hc, ?_⟩
  rw [hU, wordQ_uFold]
  simp only [peelExactValP, PeelExactProofsP.peelExactP_eq_rows, hr, Option.bind_some]

/-- `tile_values` at a device record `devAt p 4 pr`: its ticket and useful value, `pearl-c-sm120` v1's
(`devSm120v1 pr = devAt sm120E4m3 4 pr`) among them. -/
theorem tile_devAt (pr : Prices) {nw m n TM TN : ℕ} (hnw : 0 < nw)
    (X : (tileSpec p (128 * nw) TM TN).toCircuit.Transcript) (hX : (tileSpec p (128 * nw) TM TN).part.Correct X 0)
    (Aq : Matrix (Fin m) (Fin (128 * nw)) Code) (Bq : Matrix (Fin n) (Fin (128 * nw)) Code)
    (FA FB : Matrix (Fin (128 * nw)) (Fin 32) Code) (beA : Matrix (Fin m) (Fin 32) ℕ)
    (beB : Matrix (Fin n) (Fin 32) ℕ) (i : Fin m) (j : Fin n) {ti tj : ℕ} (hti : ti < TM) (htj : tj < TN)
    (hA : ∀ l : Fin (128 * nw), val X (ti * (128 * nw) + l) = (Aq i l).val)
    (hB : ∀ l : Fin (128 * nw), val X (TM * (128 * nw) + tj * (128 * nw) + l) = (Bq j l).val)
    {c : ℕ} {pa pb : List ℕ} (hr : PeelExactProofsP.peelRowsP p 4 Aq Bq FA FB beA beB i j = some (c, pa, pb))
    (hpa : ∀ q < 64, val X ((TM + TN) * (128 * nw) + 64 * ti + q) = pa.getD q 0)
    (hpb : ∀ q < 64, val X ((TM + TN) * (128 * nw) + 64 * TM + 64 * tj + q) = pb.getD q 0) :
    wordQ (val X (cGate (128 * nw) TM TN ti tj)) = (devAt p 4 pr).ticket Aq Bq i j ∧
      wordQ (val X (uGate (128 * nw) TM TN ti tj)) = (devAt p 4 pr).useful Aq Bq FA FB beA beB i j :=
  tile_values p hnw X hX Aq Bq FA FB beA beB i j hti htj hA hB hr hpa hpb

end Tile

end Pouw.PearlC.TileCircuit
