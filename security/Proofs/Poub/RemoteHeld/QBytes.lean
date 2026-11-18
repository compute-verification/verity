module

public import Proofs.Poub.RemoteHeld.Subprotocol
public import Proofs.Properties.Poub.Remote.SharedLink

/-!
# What the bandwidth component publishes of `q_bytes`, proved (`Properties.Poub.Subprotocol.PublishedQBytes`)

`certify` subtracts the pooled bound's W, rounded up, from the receipts of `windowsOf`, each sample's receipts and
draws. On an accepted record those are the record's: each sample's receipts are `Record.sampleReceiptsAt`'s
(`windowsOf_fst`, `record_receipts_eq`), so the count it subtracts from is `Record.receipts`', and W is at least the
sample's pooled bound of the same windows (`Properties.Poub.pooledCovers`, `certify_qBytes_le`).
-/

@[expose] public section

namespace Properties.Poub.Subprotocol.SecurityProofs

open _root_.Poub.Line Properties.Poub.Line Properties.Poub.Line.SecurityProofs

theorem foldl_add_map {α : Type} (f : α → ℕ) :
    ∀ (l : List α) (a : ℕ), l.foldl (fun x w => x + f w) a = a + (l.map f).sum
  | [], a => by simp
  | w :: l, a => by rw [List.foldl_cons, foldl_add_map f l, List.map_cons, List.sum_cons]; omega

/-- The sum of `f` over the slice `[a, b)` of `A`, index by index. -/
theorem foldl_extract {α : Type} [Inhabited α] (A : Array α) (f : α → ℕ) (a b : ℕ) :
    (A.extract a b).foldl (fun x w => x + f w) 0 =
      ((List.range (min b A.size - a)).map fun k => f A[a + k]!).sum := by
  rw [← Array.foldl_toList, foldl_add_map, Nat.zero_add]
  congr 1
  apply List.ext_getElem
  · simp only [List.length_map, Array.length_toList, Array.size_extract, List.length_range]
  · intro k h1 h2
    have hk : a + k < A.size := by simp at h1; omega
    simp only [List.getElem_map, Array.getElem_toList, Array.getElem_extract, List.getElem_range]
    rw [getElem!_pos A (a + k) hk]

/-- `certify`'s rounding: `⌈q⌉` as `(max(num, 0) + den − 1) / den` is at least `q`. -/
theorem rat_le_ceil (q : ℚ) : (q : ℝ) ≤ (((q.num.toNat + q.den - 1) / q.den : ℕ) : ℝ) := by
  have hd : 0 < q.den := q.den_pos
  have h1 := Nat.div_add_mod (q.num.toNat + q.den - 1) q.den
  have h2 := Nat.mod_lt (q.num.toNat + q.den - 1) hd
  generalize (q.num.toNat + q.den - 1) / q.den = k at h1 ⊢
  generalize (q.num.toNat + q.den - 1) % q.den = r at h1 h2
  have hk : q.num.toNat ≤ q.den * k := by
    generalize q.den * k = X at h1 ⊢
    omega
  have hnum : (q.num : ℝ) ≤ (q.num.toNat : ℝ) := by exact_mod_cast Int.self_le_toNat q.num
  have hdR : (0 : ℝ) < q.den := by exact_mod_cast hd
  rw [Rat.cast_def, div_le_iff₀ hdR]
  calc (q.num : ℝ) ≤ q.num.toNat := hnum
    _ ≤ ((q.den * k : ℕ) : ℝ) := by exact_mod_cast hk
    _ = k * q.den := by push_cast; ring

/-- At one window a sample or more, `windowsOf`'s receipts are `sampleReceipts`': each window's at g = 1, and above it
`samplesOf`'s, whose form `windowsOf` has. -/
theorem windowsOf_fst (sm : Poub.Session.Summary) {g : ℕ} (hg : 1 ≤ g) :
    (Poub.Protocol.windowsOf sm g).map (·.1) = sampleReceipts g sm.windows := by
  rcases Nat.lt_or_ge g 2 with h2 | h2
  · obtain rfl : g = 1 := by omega
    rw [sampleReceipts, if_pos le_rfl]
    apply Array.ext
    · simp [Poub.Protocol.windowsOf]
    · intro i h1 _
      have hi : i < sm.windows.size := by simpa [Poub.Protocol.windowsOf] using h1
      simp only [Poub.Protocol.windowsOf, Array.getElem_map, Array.getElem_range]
      rw [foldl_extract]
      have hm : min (i * 1 + 1) sm.windows.size - i * 1 = 1 := by omega
      rw [hm]
      simp only [List.range_one, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, Nat.mul_one,
        Nat.add_zero]
      rw [getElem!_pos sm.windows i hi]
  · rw [sampleReceipts, if_neg (by omega)]
    rfl

/-- On an accepted record, each sample's receipts in the Lean verifier's record are those its windows have in the state
the record's session reaches (`receipts_of_line` without the line's limit). -/
theorem record_receipts_eq {node : Node} {r : Record} {sm : Poub.Session.Summary} (hv : Verified node r)
    (hsm : r.summary node = some sm) :
    sampleReceipts node.groupsPerCoin sm.windows = Array.ofFn fun c : Fin node.samples => r.sampleReceiptsAt node c := by
  obtain ⟨rs, s, hrun, hout⟩ := hv
  have hs : s.summary = some sm := by
    unfold Record.summary at hsm
    rw [hrun] at hsm
    exact hsm
  have hreach := reached_of_run hrun
  have hW := (windows_of_verified hrun hout).1
  have hS := summary_windows hs
  have hsz := (summary_of hs).1
  by_cases hg : node.groupsPerCoin ≤ 1
  · have hD : node.samples = node.windows := by unfold Node.samples; rw [ite_eq_left hg]
    rw [sampleReceipts, ite_eq_left hg, hS]
    apply Array.ext
    · simp [hD, hW]
    · intro i h1 h2
      have hi : i < s.windows.size := by simpa using h1
      have hat : r.sampleReceiptsAt node i = s.windows[i].receipts := by
        rw [Record.sampleReceiptsAt, ite_eq_left hg, Record.receiptsAt, hreach, Array.getElem?_eq_getElem hi]
        all_goals rfl
      simp only [Array.getElem_map, Array.getElem_ofFn]
      exact hat.symm
  · have hD : node.samples = (node.windows + node.groupsPerCoin - 1) / node.groupsPerCoin := by
      unfold Node.samples; rw [ite_eq_right hg]
    rw [sampleReceipts, ite_eq_right hg]
    apply Array.ext
    · simp [samplesOf, hD, hsz, hW]
    · intro i h1 h2
      have hat : r.sampleReceiptsAt node i = ((samplesOf node.groupsPerCoin sm.windows).map (·.1))[i] := by
        rw [Record.sampleReceiptsAt, ite_eq_right hg, hreach]
        simp only [samplesOf, Array.getElem_map, Array.getElem_range]
        exact (foldl_extract_congr _root_.Poub.Line.receipts Poub.Session.Win.receipts hS _ _).symm
      simp only [Array.getElem_ofFn]
      exact hat.symm

open ResourceAccounting.Remote.Sample in
/-- **`certify`'s `q_bytes` against a pooled bound**: where the windows `certify` pools, whenever it pools them, are
`(U_c, d_c)`, its `q_bytes` and `BLOCK_BYTES` times any `P` below their pooled bound, itself below `Σ U`, are together
at most `BLOCK_BYTES · Σ U`: its W covers the pooled bound (`pooledCovers`) and its rounding only raises W
(`rat_le_ceil`); where it pools none it publishes 0. -/
theorem certify_qBytes_le (params : Poub.Session.Params) (g : ℕ) (sm : Poub.Session.Summary) {n : ℕ}
    (U d : Fin n → ℕ) (P : ℝ)
    (hws : ∀ b, Poub.Pooled.pooled (Poub.Protocol.windowsOf sm g) Poub.Protocol.DELTA = .ok b →
      Poub.Protocol.windowsOf sm g = Array.ofFn fun c => (U c, d c))
    (hP : P ≤ pooled n U d (Poub.Protocol.DELTA : ℝ))
    (hS : pooled n U d (Poub.Protocol.DELTA : ℝ) ≤ ∑ c, (U c : ℝ)) :
    ((Poub.Protocol.certify params g sm).1.qBytes : ℝ) + (Poub.Codec.BLOCK_BYTES : ℝ) * P ≤
      (Poub.Codec.BLOCK_BYTES : ℝ) * ∑ c, (U c : ℝ) := by
  have hB : (0 : ℝ) ≤ Poub.Codec.BLOCK_BYTES := Nat.cast_nonneg _
  unfold Poub.Protocol.certify
  dsimp only
  split
  · rename_i b hb
    have hw := hws b hb
    have hc := Properties.Poub.pooledCovers n U d Poub.Protocol.DELTA b (hw ▸ hb)
    have hsum : ((((Poub.Protocol.windowsOf sm g).map (·.1)).toList.sum : ℕ) : ℝ) = ∑ c, (U c : ℝ) := by
      rw [hw]
      simp [Array.map_ofFn, List.sum_ofFn, Function.comp_def]
    have hceil := rat_le_ceil b.wrong
    generalize (b.wrong.num.toNat + b.wrong.den - 1) / b.wrong.den = w at hceil ⊢
    generalize ((Poub.Protocol.windowsOf sm g).map (·.1)).toList.sum = R at hsum ⊢
    rw [← hsum]
    have hPw : P ≤ w := hP.trans (hc.trans hceil)
    rcases le_or_gt w R with h | h
    · rw [Nat.cast_mul, Nat.cast_sub h]
      have := mul_le_mul_of_nonneg_left hPw hB
      nlinarith
    · rw [Nat.sub_eq_zero_of_le h.le, Nat.zero_mul, Nat.cast_zero, zero_add]
      exact mul_le_mul_of_nonneg_left (hP.trans (hS.trans hsum.symm.le)) hB
  · simp only [Nat.cast_zero, zero_add]
    exact mul_le_mul_of_nonneg_left (hP.trans hS) hB

end Properties.Poub.Subprotocol.SecurityProofs
