import Proofs.Pouw.Lead.Network.Defs
import Proofs.Pouw.Lead.Network.Rot
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.FinCases
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring
import Mathlib.Algebra.BigOperators.Field
import Mathlib.Data.Nat.Cast.Order.Field

/-!
# The network term at the three points and over a forward

The network term of a call that exchanges `E` bits is `networkGap L C E = β E / 32` registers, `β = n_max · W_mac`
(`betaLead`). With words chunks of 65,536 words (`l1w`), `W_mac` is at most the words' work a MAC, their compressions'
and A's (`betaLead_mul_le`), the padded call's rows bounding A's:

* `betaLead_le_range`: at `n, K ≥ 4,096` (`GapAt`'s range), with `stepsFor`'s steps or fewer, `W_mac ≤ 128.06`. At the
  three points, `networkAt : NetworkAt` (cap 32) and `networkAt320 : NetworkAt320` (cap 320): a call whose egress is
  at most its stock share of the cap (`egressShare`) has a network term of at most 0.036% · cap / 32 of its stock.
* `llama_betaLead_le`: at Llama-3.2-1B's own four matmuls, `W_mac ≤ 128.69` (K 2,048 × N 2,048 the largest).
* `forward_networkGap_le` and `llama_forward_network`: a forward's calls, egress summed within `cap · T` bits, have
  network terms summing to at most 0.0362% · cap / 32 of the forward's stock: `networkForward32 : NetworkForward 32`
  0.0362% and `networkForward320 : NetworkForward 320` 0.362%.
-/

namespace Pouw.SecurityProofs.Lead.Network

open Finset Pouw.Lead Pouw.Guarantees.Lead

/-! ## The lead's padded shape -/

theorem le_padRows (m : ℕ) : m ≤ padRows m := by
  unfold padRows
  omega

theorem le_padCols (n : ℕ) : n ≤ padCols n := by
  unfold padCols
  omega

theorem lead_rows_le (T m n K : ℕ) : (lead T m n K).rows ≤ (lead T m n K).m := le_padRows m

theorem one_le_lead_m {T m n K : ℕ} (hm : 1 ≤ m) : 1 ≤ (lead T m n K).m := hm.trans (le_padRows m)

theorem stepsFor_le (K : ℕ) : Pouw.Lead.stepsFor K ≤ stepsFor K := by
  unfold Pouw.Lead.stepsFor stepsFor
  omega

/-! ## `W_mac` -/

theorem wordChunks_le (L : Pouw.Lead.Layout) (hcw : L.chunkWords = 65536) :
    (wordChunks L : ℚ) ≤ (L.m * L.n * L.S + 65535) / 65536 := by
  unfold wordChunks
  rw [hcw, show L.m * L.n * L.S + 65536 - 1 = L.m * L.n * L.S + 65535 by omega]
  calc (((L.m * L.n * L.S + 65535) / 65536 : ℕ) : ℚ) ≤ ((L.m * L.n * L.S + 65535 : ℕ) : ℚ) / (65536 : ℕ) :=
        Nat.cast_div_le
    _ = _ := by push_cast; ring

/-- **The honest work a row**: with words chunks of 65,536 words, `m ≥ 1` and A's rows at most the padded rows, the
honest work is at most `m` times `n S (1 + c / 32 + c / 65,536) + c + a K / 128`, `c` and `a` ShaWrites' registers a
words and an A compression. -/
theorem honestRegs_le (L : Pouw.Lead.Layout) (C : Pouw.Lead.Commit L) (hC : ChunksOf L C) (hcw : L.chunkWords = 65536)
    (hm : 1 ≤ L.m) (hr : L.rows ≤ L.m) :
    honestRegs L C ≤ L.m * (L.n * L.S * (1 + shaWordsRegs / 32 + shaWordsRegs / 65536) + shaWordsRegs +
      shaARegs * L.K / 128) := by
  have hch := wordChunks_le L hcw
  have hm' : (1 : ℚ) ≤ L.m := by exact_mod_cast hm
  have hr' : (L.rows : ℚ) ≤ L.m := by exact_mod_cast hr
  have hc : (0 : ℚ) ≤ shaWordsRegs := by unfold shaWordsRegs; norm_num
  have ha : (0 : ℚ) ≤ shaARegs * L.K := by unfold shaARegs; positivity
  rw [← sub_nonneg]
  have e : (L.m : ℚ) * (L.n * L.S * (1 + shaWordsRegs / 32 + shaWordsRegs / 65536) + shaWordsRegs +
      shaARegs * L.K / 128) - honestRegs L C =
      shaWordsRegs * (L.m * L.n * L.S / 65536 + L.m - wordChunks L) + shaARegs * L.K * (L.m - L.rows) / 128 := by
    unfold honestRegs wordsBlocks aBlocks wordsRegs
    rw [hC]
    ring
  rw [e]
  have h1 : (0 : ℚ) ≤ shaWordsRegs * (L.m * L.n * L.S / 65536 + L.m - wordChunks L) := mul_nonneg hc (by linarith)
  have h2 : (0 : ℚ) ≤ shaARegs * L.K * (L.m - L.rows) / 128 :=
    div_nonneg (mul_nonneg ha (by linarith)) (by norm_num)
  linarith

/-- **`β` times the call's `n K`** is at most `n_max · 32` times the work a row of `honestRegs_le`. -/
theorem betaLead_mul_le (L : Pouw.Lead.Layout) (C : Pouw.Lead.Commit L) (hC : ChunksOf L C) (hcw : L.chunkWords = 65536)
    (hm : 1 ≤ L.m) (hr : L.rows ≤ L.m) (hn : 0 < L.n) (hK : 0 < L.K) :
    betaLead L C * (L.n * L.K) ≤ 256 / 3 * 32 *
      (L.n * L.S * (1 + shaWordsRegs / 32 + shaWordsRegs / 65536) + shaWordsRegs + shaARegs * L.K / 128) := by
  have hR := honestRegs_le L C hC hcw hm hr
  have hm' : (0 : ℚ) < L.m := by exact_mod_cast hm
  have hn' : (0 : ℚ) < L.n := by exact_mod_cast hn
  have hK' : (0 : ℚ) < L.K := by exact_mod_cast hK
  have e : betaLead L C * (L.n * L.K) = 256 / 3 * 32 * (honestRegs L C / L.m) := by
    unfold betaLead
    field_simp
  rw [e]
  refine mul_le_mul_of_nonneg_left ?_ (by norm_num)
  rw [div_le_iff₀ hm']
  exact hR.trans_eq (mul_comm _ _)

/-- `β` at a call of `n` columns, inner dimension `K` and `S` steps, from a bound on the work a row. -/
theorem betaLead_le_of_shape (L : Pouw.Lead.Layout) (C : Pouw.Lead.Commit L) (hC : ChunksOf L C)
    (hcw : L.chunkWords = 65536) (hm : 1 ≤ L.m) (hr : L.rows ≤ L.m) {n K S : ℕ} (hn : L.n = n) (hK : L.K = K)
    (hS : L.S = S) (hn0 : 0 < n) (hK0 : 0 < K) (B : ℚ)
    (hB : 256 / 3 * 32 * (n * S * (1 + shaWordsRegs / 32 + shaWordsRegs / 65536) + shaWordsRegs +
      shaARegs * K / 128) ≤ B * (n * K)) : betaLead L C ≤ B := by
  have h := betaLead_mul_le L C hC hcw hm hr (hn ▸ hn0) (hK ▸ hK0)
  rw [hn, hK, hS] at h
  have hnK : (0 : ℚ) < n * K := by positivity
  exact le_of_mul_le_mul_right (h.trans hB) hnK

/-- **`W_mac ≤ 128.06` at `n, K ≥ 4,096`**: `β ≤ 256 / 3 · 128.06` for every call of at least one row, A's rows at
most the padded rows, words chunks of 65,536 words and at most `stepsFor K` steps. -/
theorem betaLead_le_range (L : Pouw.Lead.Layout) (C : Pouw.Lead.Commit L) (hC : ChunksOf L C)
    (hcw : L.chunkWords = 65536) (hm : 1 ≤ L.m) (hr : L.rows ≤ L.m) (hn : 4096 ≤ L.n) (hK : 4096 ≤ L.K)
    (hS : L.S ≤ stepsFor L.K) :
    betaLead L C ≤ 256 / 3 * (12806 / 100) := by
  have h := betaLead_mul_le L C hC hcw hm hr (by omega) (by omega)
  have hS28 : 28 * L.S ≤ L.K + 35 := by
    unfold stepsFor at hS
    omega
  have hS' : (28 : ℚ) * L.S ≤ L.K + 35 := by exact_mod_cast hS28
  have hn' : (4096 : ℚ) ≤ L.n := by exact_mod_cast hn
  have hK' : (4096 : ℚ) ≤ L.K := by exact_mod_cast hK
  have hnS : (L.n : ℚ) * (28 * L.S) ≤ L.n * (L.K + 35) := mul_le_mul_of_nonneg_left hS' (by linarith)
  have hprod : (0 : ℚ) ≤ (L.n - 4096) * (L.K - 4096) := mul_nonneg (by linarith) (by linarith)
  have hX : 256 / 3 * 32 * (L.n * L.S * (1 + shaWordsRegs / 32 + shaWordsRegs / 65536) + shaWordsRegs +
      shaARegs * L.K / 128) ≤ 256 / 3 * (12806 / 100) * (L.n * L.K) := by
    unfold shaWordsRegs shaARegs
    nlinarith [hnS, hprod]
  have hnK : (0 : ℚ) < L.n * L.K := by positivity
  exact le_of_mul_le_mul_right (h.trans hX) hnK

theorem betaLead_nonneg (L : Pouw.Lead.Layout) (C : Pouw.Lead.Commit L) : 0 ≤ betaLead L C := by
  unfold betaLead honestRegs wordsBlocks aBlocks wordsRegs shaWordsRegs shaARegs
  positivity

/-! ## The term at the three points -/

theorem leadMacs_q : (leadMacs : ℚ) = 973078528 := by exact_mod_cast leadMacs_eq

/-- **The network term at a call's stock share of the cap**: a call of `n, K ≥ 4,096` whose egress `E` is at most its
stock share of `L.cap` bits a token (`egressShare`) has a network term of at most `0.036% · cap / 32` of its stock. -/
theorem networkGap_le_share (L : Pouw.Lead.Layout) (C : Pouw.Lead.Commit L) (hC : ChunksOf L C)
    (hcw : L.chunkWords = 65536) (hm : 1 ≤ L.m) (hr : L.rows ≤ L.m) (hn : 4096 ≤ L.n) (hK : 4096 ≤ L.K)
    (hS : L.S ≤ stepsFor L.K) (E : ℚ) (hE : E ≤ egressShare L) :
    networkGap L C E ≤ 36 / 100000 * (L.cap / 32 : ℚ) * stockRegs L := by
  have hβ := betaLead_le_range L C hC hcw hm hr hn hK hS
  have hβ0 := betaLead_nonneg L C
  have hE' : E * 973078528 ≤ L.cap * (L.m * L.n * L.K) := by
    unfold egressShare at hE
    rw [leadMacs_q] at hE
    exact (le_div_iff₀ (by norm_num)).1 hE
  have hcap : (0 : ℚ) ≤ L.cap * (L.m * L.n * L.K) := by positivity
  have h1 : betaLead L C * (E * 973078528) ≤ betaLead L C * (L.cap * (L.m * L.n * L.K)) :=
    mul_le_mul_of_nonneg_left hE' hβ0
  have h2 : betaLead L C * (L.cap * (L.m * L.n * L.K)) ≤ 256 / 3 * (12806 / 100) * (L.cap * (L.m * L.n * L.K)) :=
    mul_le_mul_of_nonneg_right hβ hcap
  unfold networkGap stockRegs
  nlinarith [h1, h2, hcap]

/-- The network term at a call of the lead at its stock share of the cap. -/
theorem networkGap_lead {T m n K : ℕ} (hm : 1 ≤ m) (hn : 4096 ≤ n) (hK : 4096 ≤ K) (L : Pouw.Lead.Layout)
    (hL : L = lead T m n K ∨ L = lead320 T m n K) (C : Pouw.Lead.Commit L) (hC : ChunksOf L C) (E : ℚ)
    (hE : E ≤ egressShare L) : networkGap L C E ≤ 36 / 100000 * (L.cap / 32 : ℚ) * stockRegs L := by
  have hm' : 1 ≤ (lead T m n K).m := one_le_lead_m hm
  have hr := lead_rows_le T m n K
  have hn' : 4096 ≤ (lead T m n K).n := hn.trans (le_padCols n)
  rcases hL with rfl | rfl
  · exact networkGap_le_share _ C hC rfl hm' hr hn' hK (stepsFor_le K) E hE
  · exact networkGap_le_share _ C hC rfl hm' hr hn' hK (stepsFor_le K) E hE

theorem networkGap_lead32 {T m n K : ℕ} (hm : 1 ≤ m) (hn : 4096 ≤ n) (hK : 4096 ≤ K)
    (C : Pouw.Lead.Commit (lead T m n K)) (hC : ChunksOf _ C) (E : ℚ) (hE : E ≤ egressShare (lead T m n K)) :
    networkGap _ C E ≤ 36 / 100000 * stockRegs (lead T m n K) := by
  have h := networkGap_lead hm hn hK _ (Or.inl rfl) C hC E hE
  have e : ((lead T m n K).cap / 32 : ℚ) = 1 := by
    show ((32 : ℕ) : ℚ) / 32 = 1
    norm_num
  rwa [e, mul_one] at h

theorem networkGap_lead320 {T m n K : ℕ} (hm : 1 ≤ m) (hn : 4096 ≤ n) (hK : 4096 ≤ K)
    (C : Pouw.Lead.Commit (lead320 T m n K)) (hC : ChunksOf _ C) (E : ℚ) (hE : E ≤ egressShare (lead320 T m n K)) :
    networkGap _ C E ≤ 36 / 10000 * stockRegs (lead320 T m n K) := by
  have h := networkGap_lead hm hn hK _ (Or.inr rfl) C hC E hE
  have e : ((lead320 T m n K).cap / 32 : ℚ) = 10 := by
    show ((320 : ℕ) : ℚ) / 32 = 10
    norm_num
  rw [e] at h
  linarith

/-- **The network term at the three points, cap 32** (`NetworkAt`): at most 0.036% of stock. -/
theorem networkAt : NetworkAt := fun n K hn hK =>
  ⟨networkGap_lead32 (T := 512) (m := 64) (by norm_num) hn hK,
    networkGap_lead32 (T := 512) (m := 512) (by norm_num) hn hK,
    networkGap_lead32 (T := 8192) (m := 8192) (by norm_num) hn hK⟩

/-- **The network term at the three points, cap 320** (`NetworkAt320`): at most 0.36% of stock. -/
theorem networkAt320 : NetworkAt320 := fun n K hn hK =>
  ⟨networkGap_lead320 (T := 512) (m := 64) (by norm_num) hn hK,
    networkGap_lead320 (T := 512) (m := 512) (by norm_num) hn hK,
    networkGap_lead320 (T := 8192) (m := 8192) (by norm_num) hn hK⟩

/-! ## Over a forward -/

/-- **A forward's network terms**: calls whose `β` is at most `B` and whose egress sums to `ΣE` have network terms
summing to at most `B · ΣE / 32`. -/
theorem forward_networkGap_le {ι : Type*} (s : Finset ι) (L : ι → Pouw.Lead.Layout) (C : ∀ c, Pouw.Lead.Commit (L c))
    (E : ι → ℚ) (B : ℚ) (hE : ∀ c ∈ s, 0 ≤ E c) (hB : ∀ c ∈ s, betaLead (L c) (C c) ≤ B) :
    ∑ c ∈ s, networkGap (L c) (C c) (E c) ≤ B * (∑ c ∈ s, E c) / 32 := by
  rw [mul_sum, sum_div]
  exact sum_le_sum fun c hc =>
    div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_right (hB c hc) (hE c hc)) (by norm_num)

theorem llamaShapes_padCols (q : Fin 4) : padCols (llamaShapes q).1 = (llamaShapes q).1 := by
  fin_cases q <;> rfl

theorem llamaCall_n (cap T m : ℕ) (c : Fin (T / m) × Fin 16 × Fin 4) :
    (llamaCall cap T m c).n = (llamaShapes c.2.2).1 := llamaShapes_padCols c.2.2

theorem llamaCall_S (cap T m : ℕ) (c : Fin (T / m) × Fin 16 × Fin 4) :
    (llamaCall cap T m c).S = Pouw.Lead.stepsFor (llamaShapes c.2.2).2 := rfl

theorem stepsFor_2048 : Pouw.Lead.stepsFor 2048 = 74 := by norm_num [Pouw.Lead.stepsFor]

theorem stepsFor_8192 : Pouw.Lead.stepsFor 8192 = 293 := by norm_num [Pouw.Lead.stepsFor]

/-- **`W_mac ≤ 128.69` at Llama-3.2-1B's own calls**, at any number of rows. -/
theorem llama_betaLead_le (cap T m : ℕ) (hm : 1 ≤ m) (c : Fin (T / m) × Fin 16 × Fin 4)
    (C : Pouw.Lead.Commit (llamaCall cap T m c)) (hC : ChunksOf _ C) :
    betaLead _ C ≤ 256 / 3 * (12869 / 100) := by
  have hm' : 1 ≤ (llamaCall cap T m c).m := one_le_lead_m (T := T) (n := (llamaShapes c.2.2).1)
    (K := (llamaShapes c.2.2).2) hm
  have hr : (llamaCall cap T m c).rows ≤ (llamaCall cap T m c).m := le_padRows m
  obtain ⟨r, l, q⟩ := c
  fin_cases q
  · exact betaLead_le_of_shape _ C hC rfl hm' hr (n := 3072) (K := 2048) (S := 74)
      ((llamaCall_n cap T m _).trans rfl) rfl ((llamaCall_S cap T m _).trans stepsFor_2048) (by norm_num)
      (by norm_num) _ (by unfold shaWordsRegs shaARegs; norm_num)
  · exact betaLead_le_of_shape _ C hC rfl hm' hr (n := 2048) (K := 2048) (S := 74)
      ((llamaCall_n cap T m _).trans rfl) rfl ((llamaCall_S cap T m _).trans stepsFor_2048) (by norm_num)
      (by norm_num) _ (by unfold shaWordsRegs shaARegs; norm_num)
  · exact betaLead_le_of_shape _ C hC rfl hm' hr (n := 16384) (K := 2048) (S := 74)
      ((llamaCall_n cap T m _).trans rfl) rfl ((llamaCall_S cap T m _).trans stepsFor_2048) (by norm_num)
      (by norm_num) _ (by unfold shaWordsRegs shaARegs; norm_num)
  · exact betaLead_le_of_shape _ C hC rfl hm' hr (n := 2048) (K := 8192) (S := 293)
      ((llamaCall_n cap T m _).trans rfl) rfl ((llamaCall_S cap T m _).trans stepsFor_8192) (by norm_num)
      (by norm_num) _ (by unfold shaWordsRegs shaARegs; norm_num)

/-- A forward's stock: `(T / m) · padRows m · leadMacs / 32` registers, its calls' padded rows over its layers'
matmuls. -/
theorem llama_stock (cap T m : ℕ) :
    ∑ c, stockRegs (llamaCall cap T m c) = ((T / m : ℕ) : ℚ) * padRows m * leadMacs / 32 := by
  have e : ∀ c : Fin (T / m) × Fin 16 × Fin 4, stockRegs (llamaCall cap T m c) =
      (padRows m : ℚ) / 32 * (((llamaShapes c.2.2).1 * (llamaShapes c.2.2).2 : ℕ) : ℚ) := fun c => by
    simp only [stockRegs, llamaCall, lead, llamaShapes_padCols]
    push_cast
    ring
  simp only [e, Fintype.sum_prod_type, sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul, ← mul_sum]
  unfold leadMacs
  push_cast
  ring

/-- A forward of `m ∣ T` rows a call has at least the unpadded stock, `T · leadMacs / 32`. -/
theorem llama_stock_ge (cap T m : ℕ) (hmT : m ∣ T) :
    (T : ℚ) * 973078528 / 32 ≤ ∑ c, stockRegs (llamaCall cap T m c) := by
  have hT : T ≤ T / m * padRows m :=
    calc T = T / m * m := (Nat.div_mul_cancel hmT).symm
      _ ≤ T / m * padRows m := Nat.mul_le_mul_left _ (le_padRows m)
  have hT' : (T : ℚ) ≤ ((T / m : ℕ) : ℚ) * padRows m := by exact_mod_cast hT
  rw [llama_stock, leadMacs_q]
  linarith

/-- **The network term over a forward of Llama-3.2-1B**: at `T` tokens in calls of `m ∣ T` rows, egress summed within
`cap · T` bits, the calls' network terms sum to at most `0.0362% · cap / 32` of the forward's stock. -/
theorem llama_forward_network (cap T m : ℕ) (hm : 1 ≤ m) (hmT : m ∣ T) (C : ∀ c, Pouw.Lead.Commit (llamaCall cap T m c))
    (hC : ∀ c, ChunksOf _ (C c)) (E : Fin (T / m) × Fin 16 × Fin 4 → ℚ) (hE : ∀ c, 0 ≤ E c)
    (hsum : ∑ c, E c ≤ cap * T) :
    ∑ c, networkGap (llamaCall cap T m c) (C c) (E c) ≤
      362 / 1000000 * (cap / 32 : ℚ) * ∑ c, stockRegs (llamaCall cap T m c) := by
  have h1 := forward_networkGap_le univ (llamaCall cap T m) C E (256 / 3 * (12869 / 100)) (fun c _ => hE c)
    fun c _ => llama_betaLead_le cap T m hm c (C c) (hC c)
  have hcT : (0 : ℚ) ≤ cap * T := by positivity
  have hstock := llama_stock_ge cap T m hmT
  calc ∑ c, networkGap (llamaCall cap T m c) (C c) (E c) ≤ 256 / 3 * (12869 / 100) * (∑ c, E c) / 32 := h1
    _ ≤ 256 / 3 * (12869 / 100) * (cap * T) / 32 := by gcongr
    _ ≤ 362 / 1000000 * (cap / 32 : ℚ) * (T * 973078528 / 32) := by nlinarith [hcT]
    _ ≤ 362 / 1000000 * (cap / 32 : ℚ) * ∑ c, stockRegs (llamaCall cap T m c) := by gcongr

/-- **0.0362% of the forward's stock at 32 bits a token** (`NetworkForward 32`). -/
theorem networkForward32 : NetworkForward 32 (362 / 1000000) := by
  intro T m hm hmT C hC E hE hsum
  have h := llama_forward_network 32 T m hm hmT C hC E hE hsum
  have e : ((32 : ℕ) : ℚ) / 32 = 1 := by norm_num
  rwa [e, mul_one] at h

/-- **0.362% of the forward's stock at 320 bits a token** (`NetworkForward 320`). -/
theorem networkForward320 : NetworkForward 320 (362 / 100000) := by
  intro T m hm hmT C hC E hE hsum
  have h := llama_forward_network 320 T m hm hmT C hC E hE hsum
  have e : ((320 : ℕ) : ℚ) / 32 = 10 := by norm_num
  rw [e] at h
  linarith

end Pouw.SecurityProofs.Lead.Network
