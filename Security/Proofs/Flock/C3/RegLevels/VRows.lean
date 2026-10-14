import Proofs.Flock.C3.RegLevels.OpensRegs

/-!
# Check 5's `v` rows: `rec-residuals`' rows as bits of `vOf`

`checkResiduals` (check 5, at a `rec-residuals/m<m>-k<k_log>-c<claims>-p<j>` unit) registers, for each rep `r < 2`, part
`j`'s elements `vs` of `base ++ comb` in u16 words, `base` the rep's `verifierRows` v without its comb and `comb` the
checked comb where `vs` reaches it, else zero (`RecCoins.checkResiduals`, `rec_vstage.stage`). Here those rows are read
as `HVCoins`' bits:

* `words_bit`, `padTo_get`: bit `b % 16` of word `b / 16` of `words xs`, padded, is bit `b % 128` of `xs[b / 128]`;
* `rep_ok`: a rep's v is `verifierRows`' base then the zero comb, and `vOf` at any comb is that base then the comb
  (`verifierRows_comb`);
* `full_vOf`, `row_vOf`, `rows_vOf`: the check's rows are bit for bit `wbit` of `sub vs (vOf …)` at the rep's coins,
  extra claims and the checked comb;
* `vbit_of_rows`: `VRowsAre`'s bits (`Family/VCoins.lean`) where the registrant's bits are those rows (`value_tie_coll`
  at `rec-residuals`' rows, off `ValueColl`) and the rep is the family's.

The `mapM` the hypotheses state is `checkResiduals`' rows, and `residualRowsOf` on the value-source lane.
-/

namespace FlockC3.RegLevels

open Flock FlockVBridge
open FlockVBridge.Decode (sub vOf)

theorem u16_testBit (x : UInt64) {k j : ℕ} (hk : k < 4) (hj : j < 16) :
    ((x >>> (16 * k).toUInt64) &&& 0xFFFF).toNat.testBit j = x.toNat.testBit (16 * k + j) := by
  have h64 : (16 * k).toUInt64.toNat % 64 = 16 * k := by
    simp only [Nat.toUInt64_eq, UInt64.toNat_ofNat']
    omega
  rw [UInt64.toNat_and, UInt64.toNat_shiftRight, h64, Nat.testBit_and, Nat.testBit_shiftRight,
    show (0xFFFF : UInt64).toNat = 2 ^ 16 - 1 from rfl, Nat.testBit_two_pow_sub_one]
  simp only [Nat.add_comm, hj, decide_true, Bool.and_true]

private theorem flatMap8 {α : Type} (g : F128 → List α) (hg : ∀ x, (g x).length = 8) :
    ∀ (xs : List F128) (a s : ℕ) (ha : a < xs.length), s < 8 → (xs.flatMap g)[8 * a + s]? = (g xs[a])[s]?
  | [], _, _, h, _ => absurd h (by simp)
  | x :: xs, 0, s, _, hs => by
    rw [List.flatMap_cons, List.getElem?_append_left (by rw [hg]; omega), Nat.mul_zero, Nat.zero_add]; rfl
  | x :: xs, a + 1, s, ha, hs => by
    rw [List.flatMap_cons, List.getElem?_append_right (by rw [hg]; omega), hg,
      show 8 * (a + 1) + s - 8 = 8 * a + s by omega, flatMap8 g hg xs a s (by simp at ha; omega) hs]
    rfl

/-- `RecCoins.words`' four u16 words of a 64-bit word, low first. -/
abbrev u16s (x : UInt64) : Array ℕ := (Array.range 4).map fun k => ((x >>> (16 * k).toUInt64) &&& 0xFFFF).toNat

theorem words_get? (xs : Array F128) {a s : ℕ} (ha : a < xs.size) (hs : s < 8) :
    (RecCoins.words xs)[8 * a + s]? = (u16s xs[a].lo ++ u16s xs[a].hi)[s]? := by
  have := flatMap8 (fun x => (u16s x.lo ++ u16s x.hi).toList) (fun x => by simp) xs.toList a s
    (by simpa only [Array.length_toList] using ha) hs
  unfold RecCoins.words
  rw [← Array.getElem?_toList, Array.toList_flatMap, this, Array.getElem?_toList]
  simp only [Array.getElem_toList]

/-- **A bit of `words xs`** is the element's: bit `b % 16` of word `b / 16` is bit `b % 128` of `xs[b / 128]`. -/
theorem words_bit (xs : Array F128) {b : ℕ} (hb : b < 128 * xs.size) :
    ((RecCoins.words xs)[b / 16]!).testBit (b % 16) = bitOf xs[b / 128]! (b % 128) := by
  have ha : b / 128 < xs.size := by omega
  have h := words_get? xs ha (s := b / 16 % 8) (by omega)
  rw [show 8 * (b / 128) + b / 16 % 8 = b / 16 by omega] at h
  rw [getElem!_def, h, getElem!_pos xs _ ha]
  unfold bitOf FlockLevel3.w128
  rw [Nat.testBit_two_pow_mul_add _ (UInt64.toNat_lt _)]
  by_cases hs : b / 16 % 8 < 4
  · rw [Array.getElem?_append_left (by simpa only [Array.size_map, Array.size_range] using hs)]
    simp only [u16s, Array.getElem?_map, Array.getElem?_range, hs, if_true, Option.map_some]
    rw [u16_testBit _ hs (by omega), if_pos (by omega), show 16 * (b / 16 % 8) + b % 16 = b % 128 by omega]
  · rw [Array.getElem?_append_right (by simp only [Array.size_map, Array.size_range]; omega)]
    simp only [u16s, Array.size_map, Array.size_range, Array.getElem?_map, Array.getElem?_range,
      show b / 16 % 8 - 4 < 4 by omega, if_true, Option.map_some]
    rw [u16_testBit _ (by omega) (by omega), if_neg (by omega),
      show 16 * (b / 16 % 8 - 4) + b % 16 = b % 128 - 64 by omega]

/-- Padding with zero words changes no word (`0` is `ℕ`'s default). -/
theorem padTo_get (ws : Array ℕ) (n k : ℕ) : (RecCoins.padTo ws n)[k]! = ws[k]! := by
  unfold RecCoins.padTo
  rw [getElem!_def, getElem!_def, Array.getElem?_append]
  by_cases hk : k < ws.size
  · rw [if_pos hk]
  · rw [if_neg hk, Array.getElem?_eq_none (by omega : ws.size ≤ k), Array.getElem?_replicate]
    split
    · rename_i heq; split at heq <;> cases heq; rfl
    · rfl

/-- **A rep's v is `verifierRows`' at its coins**: `Inner.rep`'s rounds are the inner session's, its v is `verifierRows`'
base then the zero comb, and `vOf` at any comb `C` is that base, then `C`. -/
theorem rep_ok {inn : RecCoins.Inner} {sh : Shape} {extras : Array Claim} {r : ℕ} {rep : RecCoins.Rep}
    (h : inn.rep sh extras r = .ok rep) : inn.rounds r = .ok rep.rounds ∧
      ∃ V : Array F128, rep.v = V ++ RecCoins.ZERO_COMB ∧ ∀ C, vOf sh rep.rounds extras C = fun k => (V ++ C).getD k 0 := by
  unfold RecCoins.Inner.rep at h
  obtain ⟨rounds, hr, h⟩ := FlockVBridge.PartS.bind_ok h
  split at h
  · rename_i v info kappa hv
    cases h
    obtain ⟨V, rfl, hV⟩ := RecAlgebra.verifierRows_comb hv
    exact ⟨hr, V, rfl, fun C => Decode.vOf_eq (hV C)⟩
  · cases h

/-- **Check 5's elements of a rep's v are `vOf`'s**: at `vs`, `base ++ cb` (`checkResiduals`' `full`) is `vOf` at any
comb `C` that is `cb` where `vs` reaches the comb. -/
theorem full_vOf {inn : RecCoins.Inner} {sh : Shape} {extras : Array Claim} {r : ℕ} {rep : RecCoins.Rep}
    (h : inn.rep sh extras r = .ok rep) {vs : Array ℕ} {cb C : Array F128}
    (hCcb : vs.any (· ≥ (rep.v.extract 0 (rep.v.size - 64)).size) = true → cb = C) :
    vs.map ((rep.v.extract 0 (rep.v.size - 64) ++ cb)[·]!) = vs.map (vOf sh rep.rounds extras C) := by
  obtain ⟨-, V, hv, hV⟩ := rep_ok h
  have hb : rep.v.extract 0 (rep.v.size - 64) = V := by
    rw [hv]
    simp only [RecCoins.ZERO_COMB, Array.size_append, Array.size_replicate, Nat.add_sub_cancel,
      Array.extract_append, Array.extract_size, Nat.zero_le, Nat.sub_eq_zero_of_le, Nat.sub_self, Array.extract_zero,
      Array.append_empty]
  rw [hV C, hb]
  rw [hb] at hCcb
  apply Array.map_congr_left
  intro i hi
  rw [getElem!_def]
  by_cases ha : vs.any (· ≥ V.size) = true
  · rw [hCcb ha]; simp only [Array.getD_eq_getD_getElem?]; rfl
  · have hiV : i < V.size := by
      have := Array.any_eq_false.mp (Bool.eq_false_iff.mpr ha)
      simp only [Array.mem_iff_getElem] at hi
      obtain ⟨k, hk, rfl⟩ := hi
      simpa only [gt_iff_lt, ge_iff_le, decide_eq_true_eq, Nat.not_le] using this k hk
    simp only [Array.getElem?_append_left hiV, hiV, getElem?_pos, Array.getD_eq_getD_getElem?, Option.getD_some]

/-- **Check 5's `v` row of a rep is `vOf`'s bits**: the row `checkResiduals` registers for rep `r` is, bit `b % 16` of
word `b / 16`, `wbit` of part `vs`'s elements of `vOf` at the rep's coins and extra claims and any comb `C` that is the
checked comb where `vs` reaches the comb. -/
theorem row_vOf {inn : RecCoins.Inner} {sh : Shape} {extras : Array Claim} {vs : Array ℕ} {r : ℕ}
    {row : Array ℕ}
    (h : (do
      let rep ← inn.rep sh extras r
      let base := rep.v.extract 0 (rep.v.size - 64)
      let comb ← if vs.any (· ≥ base.size) then inn.comb r rep else pure RecCoins.ZERO_COMB
      let full := base ++ comb
      pure (RecCoins.padTo (RecCoins.words (vs.map (full[·]!))) (RecResiduals.portWordsOf vs.size)) :
        V (Array ℕ)) = .ok row) :
    ∃ rep, inn.rep sh extras r = .ok rep ∧ ∀ C, (vs.any (· ≥ rep.v.size - 64) = true → inn.comb r rep = .ok C) →
      ∀ b < 128 * vs.size, (row[b / 16]!).testBit (b % 16) = wbit (sub vs (vOf sh rep.rounds extras C)) b := by
  obtain ⟨rep, hrep, h⟩ := FlockVBridge.PartS.bind_ok h
  have hsz : (rep.v.extract 0 (rep.v.size - 64)).size = rep.v.size - 64 := by
    simp only [Array.size_extract, Nat.sub_le, Nat.min_eq_left, Nat.sub_zero]
  obtain ⟨cb, hcb, hrow⟩ : ∃ cb, (vs.any (· ≥ rep.v.size - 64) = true → inn.comb r rep = .ok cb) ∧
      row = RecCoins.padTo (RecCoins.words (vs.map ((rep.v.extract 0 (rep.v.size - 64) ++ cb)[·]!)))
        (RecResiduals.portWordsOf vs.size) := by
    dsimp only at h
    split at h
    · obtain ⟨cb, hcb, h⟩ := FlockVBridge.PartS.bind_ok h
      simp only [pure, Except.pure, Except.ok.injEq] at h
      exact ⟨cb, fun _ => hcb, h.symm⟩
    · rename_i ha
      rw [pure_bind] at h
      simp only [pure, Except.pure, Except.ok.injEq] at h
      exact ⟨RecCoins.ZERO_COMB, fun ha' => absurd (by rw [hsz]; exact ha') ha, h.symm⟩
  subst hrow
  refine ⟨rep, hrep, fun C hC b hb => ?_⟩
  have hCcb : vs.any (· ≥ (rep.v.extract 0 (rep.v.size - 64)).size) = true → cb = C := fun ha => by
    rw [hsz] at ha; have := (hcb ha).symm.trans (hC ha); cases this; rfl
  rw [padTo_get, full_vOf hrep hCcb, words_bit _ (by simpa only [Array.size_map] using hb), wbit, sub,
    dif_pos (by omega)]
  rw [getElem!_pos _ _ (by simp only [Array.size_map]; omega), Array.getElem_map]

/-- **Check 5's `v` rows are `vOf`'s bits** (`row_vOf` at both reps): the two rows `checkResiduals` registers. -/
theorem rows_vOf {inn : RecCoins.Inner} {sh : Shape} {extras : Array Claim} {vs : Array ℕ} {rows : Array (Array ℕ)}
    (h : (#[0, 1].mapM fun r => do
      let rep ← inn.rep sh extras r
      let base := rep.v.extract 0 (rep.v.size - 64)
      let comb ← if vs.any (· ≥ base.size) then inn.comb r rep else pure RecCoins.ZERO_COMB
      let full := base ++ comb
      pure (RecCoins.padTo (RecCoins.words (vs.map (full[·]!))) (RecResiduals.portWordsOf vs.size)) :
        V (Array (Array ℕ))) = .ok rows) :
    rows.size = 2 ∧ ∀ r < 2, ∃ rep, inn.rep sh extras r = .ok rep ∧
      ∀ C, (vs.any (· ≥ rep.v.size - 64) = true → inn.comb r rep = .ok C) →
        ∀ b < 128 * vs.size, ((rows[r]!)[b / 16]!).testBit (b % 16) = wbit (sub vs (vOf sh rep.rounds extras C)) b := by
  obtain ⟨a, b, ha, hb, rfl⟩ := FlockC3.Opening.mapM_two h
  refine ⟨rfl, fun r hr => ?_⟩
  match r, hr with
  | 0, _ => exact row_vOf ha
  | 1, _ => exact row_vOf hb

/-- **`VRowsAre`'s bits from check 5's rows**: where a registrant's bits `f` are rep `r`'s row of `checkResiduals`
(`value_tie_coll` at `rec-residuals`' rows, off `ValueColl`) and the rep the check reads has the family's rounds and,
where `vs` reaches the comb, its comb (the inner session's coins are the rep's: the lane that proves `HFiles`), `f` is
`wbit` of `vOf` at them. -/
theorem vbit_of_rows {inn : RecCoins.Inner} {sh : Shape} {extras : Array Claim} {vs : Array ℕ}
    {rows : Array (Array ℕ)}
    (h : (#[0, 1].mapM fun r => do
      let rep ← inn.rep sh extras r
      let base := rep.v.extract 0 (rep.v.size - 64)
      let comb ← if vs.any (· ≥ base.size) then inn.comb r rep else pure RecCoins.ZERO_COMB
      let full := base ++ comb
      pure (RecCoins.padTo (RecCoins.words (vs.map (full[·]!))) (RecResiduals.portWordsOf vs.size)) :
        V (Array (Array ℕ))) = .ok rows)
    {r : ℕ} (hr : r < 2) {rounds : Array (Array F128)} {comb : Array F128}
    (hrep : ∀ rep, inn.rep sh extras r = .ok rep →
      rep.rounds = rounds ∧ (vs.any (· ≥ rep.v.size - 64) = true → inn.comb r rep = .ok comb))
    {f : ℕ → Bool} (htie : ∀ b < 128 * vs.size, f b = ((rows[r]!)[b / 16]!).testBit (b % 16)) :
    ∀ b < 128 * vs.size, f b = wbit (sub vs (vOf sh rounds extras comb)) b := by
  obtain ⟨-, hall⟩ := rows_vOf h
  obtain ⟨rep, hrp, hbits⟩ := hall r hr
  obtain ⟨rfl, hcomb⟩ := hrep rep hrp
  intro b hb
  rw [htie b hb, hbits comb hcomb b hb]

end FlockC3.RegLevels
