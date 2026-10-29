import Definitions.Pouw.Guarantees.Lead
import Mathlib.Algebra.Order.BigOperators.Group.List
import Mathlib.Data.Nat.Cast.Order.Field
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring

/-!
# The lead's network term against the statement

What `Rot`'s theorems prove of the statement's definitions:
* `lead_cells_rot`: every step of the lead, step 0 included, carries the three coin lanes of its 64-row tile at
  `-rot`'s weights, `foldEven` on even rows and `foldOdd` on odd rows, of odd coins `2u − 127`; the rest of the step's
  coin term reads no lane; and the data fit 21 lanes on step 0 and 28 on each later step. So the lead's cells meet
  `Cell.Rot`, and `n_max ≤ nMax` (`Cells.lead_cells_macs_le`).
* `networkGap_le`, `networkGap_le320`: a call's network term, `nMax · wMac · callBits / 32`, is at most 0.0361% of
  its stock at cap 32 and 0.361% at cap 320 for every call of `n, K ≥ 4,096` (`networkGap_le_of` at the call's
  `wMac ≤ 128.59`, `betaLead_le_range`, and `rows ≤ m`).
* `forward_sum`: a forward's calls' bits sum to its cap, `cap · T`, so no call takes another's share.
* `llama_forward`: over Llama-3.2-1B's own calls (`llamaCalls`), whose `K = 2,048` is below `GapAt`'s range, a
  forward's network term is at most 0.0363% of its stock at cap 32 and 0.363% at cap 320 (`llama_betaLead_le`: the
  calls' `wMac ≤ 129.25`, `2,048 × 2,048` the largest).

With words chunks of 65,536 words (`l1w`), `wMac` is at most the words' work a MAC, their compressions' and A's
(`betaLead_le_of_row`), the padded call's rows bounding A's; a call of no rows has `wMac = 0`.
-/

namespace Pouw.SecurityProofs.Lead.Network

open Pouw.Lead

/-- **Every step carries the three coin lanes under `-rot`**: the lead's coin tile is 64 rows; at
every step, row `i`'s lanes are `127 s₀ + s₁ + s₂` on even rows and `s₀ + 127 s₁ + s₂` on odd rows, of its tile's coins
at the step and column; what the step's coin term adds besides is the same at any two coin values that agree off the
lanes; each lane's coin is `2u − 127` of its draw `u < 128`; and the data fit 21 lanes on step 0 and 28 on each later
step. -/
theorem lead_cells_rot (T m n K : ℕ) :
    (lead T m n K).tile = 64 ∧ K ≤ 21 + 28 * ((lead T m n K).S - 1) ∧
    (∀ (ω : Coins (lead T m n K)) (i : Fin (lead T m n K).m) (t : Fin (lead T m n K).S)
        (j : Fin (lead T m n K).n), ∃ h : (i : ℕ) / 64 < (lead T m n K).tiles,
      laneTerm (lead T m n K) ω i t j =
        if (i : ℕ) % 2 = 0 then
          127 * ω.s t ⟨(i : ℕ) / 64, h⟩ 0 j + ω.s t ⟨(i : ℕ) / 64, h⟩ 1 j + ω.s t ⟨(i : ℕ) / 64, h⟩ 2 j
        else
          ω.s t ⟨(i : ℕ) / 64, h⟩ 0 j + 127 * ω.s t ⟨(i : ℕ) / 64, h⟩ 1 j + ω.s t ⟨(i : ℕ) / 64, h⟩ 2 j) ∧
    (∀ ω ω' : Coins (lead T m n K), ω.1 = ω'.1 → ω.2.1 = ω'.2.1 → ω.2.2.1 = ω'.2.2.1 → ∀ i t j,
      coinTerm (lead T m n K) ω i t j - laneTerm (lead T m n K) ω i t j =
        coinTerm (lead T m n K) ω' i t j - laneTerm (lead T m n K) ω' i t j) ∧
    ∀ (ω : Coins (lead T m n K)) t k ℓ j, ω.s t k ℓ j = 2 * ((ω.2.2.2 t k ℓ j : ℕ) : ℤ) - 127 := by
  refine ⟨rfl, ?_, ?_, ?_, ?_⟩
  · show K ≤ 21 + 28 * (stepsFor K - 1)
    unfold stepsFor
    omega
  · intro ω i t j
    have hi : (i : ℕ) < padRows m := i.isLt
    have h : (i : ℕ) / 64 < (lead T m n K).tiles := by
      show (i : ℕ) / 64 < (padRows m + 64 - 1) / 64
      omega
    refine ⟨h, ?_⟩
    show ∑ ℓ : Fin 3, rotWeight i ℓ * (if h' : (i : ℕ) / 64 < (lead T m n K).tiles then
      ω.s t ⟨(i : ℕ) / 64, h'⟩ ℓ j else 0) = _
    rcases Nat.mod_two_eq_zero_or_one (i : ℕ) with hp | hp
    · simp [Fin.sum_univ_three, rotWeight, hp, h]
    · simp [Fin.sum_univ_three, rotWeight, hp, h]
  · intro ω ω' h1 h2 h3 i t j
    unfold coinTerm Coins.G
    rw [h1, h2, h3]
    ring
  · intro ω t k ℓ j
    show 2 * ((ω.2.2.2 t k ℓ j : ℕ) : ℤ) - (2 ^ 7 - 1) = _
    norm_num

/-! ## `wMac` -/

theorem le_padRows (m : ℕ) : m ≤ padRows m := by
  unfold padRows
  omega

theorem le_padCols (n : ℕ) : n ≤ padCols n := by
  unfold padCols
  omega

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
theorem honestRegs_le (L : Pouw.Lead.Layout) (C : Pouw.Lead.Commit L) (hC : Pouw.Guarantees.Lead.ChunksOf L C)
    (hcw : L.chunkWords = 65536) (hm : 1 ≤ L.m) (hr : L.rows ≤ L.m) :
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

theorem betaLead_nonneg (L : Pouw.Lead.Layout) (C : Pouw.Lead.Commit L) : 0 ≤ betaLead L C := by
  unfold betaLead nMax wMac honestRegs wordsBlocks aBlocks wordsRegs shaWordsRegs shaARegs
  positivity

/-- **`β` from the honest work a row**: `β ≤ B` when `n_max · 32` times the work a row of `honestRegs_le` is at most
`B · n K`. -/
theorem betaLead_le_of_row (L : Pouw.Lead.Layout) (C : Pouw.Lead.Commit L) (hC : Pouw.Guarantees.Lead.ChunksOf L C)
    (hcw : L.chunkWords = 65536) (hr : L.rows ≤ L.m) (hn : 0 < L.n) (hK : 0 < L.K) (B : ℚ) (hB0 : 0 ≤ B)
    (hB : 256 / 3 * 32 * (L.n * L.S * (1 + shaWordsRegs / 32 + shaWordsRegs / 65536) + shaWordsRegs +
      shaARegs * L.K / 128) ≤ B * (L.n * L.K)) : betaLead L C ≤ B := by
  rcases Nat.eq_zero_or_pos L.m with hm | hm
  · have e : betaLead L C = 0 := by
      unfold betaLead wMac
      rw [hm]
      simp
    rw [e]
    exact hB0
  have hR := honestRegs_le L C hC hcw hm hr
  have hm' : (0 : ℚ) < L.m := by exact_mod_cast hm
  have hn' : (0 : ℚ) < L.n := by exact_mod_cast hn
  have hK' : (0 : ℚ) < L.K := by exact_mod_cast hK
  have e : betaLead L C * (L.n * L.K) = 256 / 3 * 32 * (honestRegs L C / L.m) := by
    unfold betaLead nMax wMac
    field_simp
  have h : betaLead L C * (L.n * L.K) ≤ B * (L.n * L.K) := by
    rw [e]
    refine le_trans (mul_le_mul_of_nonneg_left ?_ (by norm_num)) hB
    rw [div_le_iff₀ hm']
    exact hR.trans_eq (mul_comm _ _)
  exact le_of_mul_le_mul_right h (by positivity)

/-- **`W_mac ≤ 128.59` at `n, K ≥ 4,096`**: `β ≤ 256 / 3 · 128.59` for every call with A's rows at most the padded
rows, words chunks of 65,536 words and `stepsFor K` steps. -/
theorem betaLead_le_range (L : Pouw.Lead.Layout) (C : Pouw.Lead.Commit L) (hC : Pouw.Guarantees.Lead.ChunksOf L C)
    (hcw : L.chunkWords = 65536) (hr : L.rows ≤ L.m) (hn : 4096 ≤ L.n) (hK : 4096 ≤ L.K) (hS : L.S = stepsFor L.K) :
    betaLead L C ≤ 256 / 3 * (12859 / 100) := by
  refine betaLead_le_of_row L C hC hcw hr (by omega) (by omega) _ (by norm_num) ?_
  have hS28 : 28 * L.S ≤ L.K + 34 := by
    rw [hS]
    unfold stepsFor
    omega
  have hS' : (28 : ℚ) * L.S ≤ L.K + 34 := by exact_mod_cast hS28
  have hn' : (4096 : ℚ) ≤ L.n := by exact_mod_cast hn
  have hK' : (4096 : ℚ) ≤ L.K := by exact_mod_cast hK
  have hnS : (L.n : ℚ) * (28 * L.S) ≤ L.n * (L.K + 34) := mul_le_mul_of_nonneg_left hS' (by linarith)
  have hprod : (0 : ℚ) ≤ (L.n - 4096) * (L.K - 4096) := mul_nonneg (by linarith) (by linarith)
  unfold shaWordsRegs shaARegs
  nlinarith [hnS, hprod]

/-- **The network term from `β`**: a call whose `β` is at most `B`, with `B · cap ≤ s · fwdMacs` and A's rows at most
its padded rows, has a network term of at most `s` of its stock. -/
theorem networkGap_le_of (L : Pouw.Lead.Layout) (C : Pouw.Lead.Commit L) {B s : ℚ} (hβ : betaLead L C ≤ B)
    (hr : L.rows ≤ L.m) (hF : 0 < L.fwdMacs) (hs : B * L.cap ≤ s * L.fwdMacs) :
    networkGap L C ≤ s * stockRegs L := by
  have hβ0 := betaLead_nonneg L C
  have hr' : (L.rows : ℚ) ≤ L.m := by exact_mod_cast hr
  have hF' : (0 : ℚ) < L.fwdMacs := by exact_mod_cast hF
  have hY : (0 : ℚ) ≤ L.m * L.n * L.K := by positivity
  have hX : (L.cap : ℚ) * (L.rows * L.n * L.K) ≤ L.cap * (L.m * L.n * L.K) := by
    refine mul_le_mul_of_nonneg_left ?_ (by positivity)
    exact mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_right hr' (by positivity)) (by positivity)
  have h1 : betaLead L C * (L.cap * (L.rows * L.n * L.K)) ≤ B * (L.cap * (L.m * L.n * L.K)) :=
    mul_le_mul hβ hX (by positivity) (hβ0.trans hβ)
  have h2 : B * (L.cap * (L.m * L.n * L.K)) ≤ s * L.fwdMacs * (L.m * L.n * L.K) :=
    calc B * (L.cap * (L.m * L.n * L.K)) = B * L.cap * (L.m * L.n * L.K) := by ring
      _ ≤ s * L.fwdMacs * (L.m * L.n * L.K) := mul_le_mul_of_nonneg_right hs hY
  unfold networkGap Pouw.Lead.Layout.callBits stockRegs
  calc betaLead L C * (L.cap * (L.rows * L.n * L.K) / L.fwdMacs) / 32
      = betaLead L C * (L.cap * (L.rows * L.n * L.K)) / (32 * L.fwdMacs) := by ring
    _ ≤ s * L.fwdMacs * (L.m * L.n * L.K) / (32 * L.fwdMacs) :=
        div_le_div_of_nonneg_right (h1.trans h2) (by positivity)
    _ = s * (L.m * L.n * L.K / 32) := by field_simp

/-- **A call's network term at cap 32**: at most 0.0361% of its stock for every call of
`n, K ≥ 4,096`. -/
theorem networkGap_le (T m n K : ℕ) (hn : 4096 ≤ n) (hK : 4096 ≤ K) (C : Pouw.Lead.Commit (lead T m n K))
    (hC : Pouw.Guarantees.Lead.ChunksOf _ C) :
    networkGap (lead T m n K) C ≤ 361 / 1000000 * stockRegs (lead T m n K) := by
  refine networkGap_le_of _ C (betaLead_le_range _ C hC rfl (le_padRows m) (hn.trans (le_padCols n)) hK rfl)
    (le_padRows m) (by show 0 < 973078528; norm_num) ?_
  show 256 / 3 * (12859 / 100) * ((32 : ℕ) : ℚ) ≤ 361 / 1000000 * ((973078528 : ℕ) : ℚ)
  norm_num

/-- **A call's network term at cap 320**: at most 0.361% of its stock for every call of
`n, K ≥ 4,096`. -/
theorem networkGap_le320 (T m n K : ℕ) (hn : 4096 ≤ n) (hK : 4096 ≤ K) (C : Pouw.Lead.Commit (lead320 T m n K))
    (hC : Pouw.Guarantees.Lead.ChunksOf _ C) :
    networkGap (lead320 T m n K) C ≤ 361 / 100000 * stockRegs (lead320 T m n K) := by
  refine networkGap_le_of _ C (betaLead_le_range _ C hC rfl (le_padRows m) (hn.trans (le_padCols n)) hK rfl)
    (le_padRows m) (by show 0 < 973078528; norm_num) ?_
  show 256 / 3 * (12859 / 100) * ((320 : ℕ) : ℚ) ≤ 361 / 100000 * ((973078528 : ℕ) : ℚ)
  norm_num

theorem sum_map_mul_cast {α : Type*} (l : List α) (a : ℚ) (f : α → ℕ) :
    (l.map fun x => a * (f x : ℚ)).sum = a * ((l.map f).sum : ℚ) := by
  induction l with
  | nil => simp
  | cons x l ih => simp [ih, mul_add]

/-- **A forward's calls' bits sum to its cap**: calls `(rows, n, K)` whose MACs, over padded columns,
are the forward's `T` tokens' have bits summing to `32 T` at cap 32 and `320 T` at cap 320. -/
theorem forward_sum (T : ℕ) (calls : List (ℕ × ℕ × ℕ))
    (h : (calls.map fun c => c.1 * padCols c.2.1 * c.2.2).sum = T * 973078528) :
    (calls.map fun c => (lead T c.1 c.2.1 c.2.2).callBits).sum = 32 * T ∧
      (calls.map fun c => (lead320 T c.1 c.2.1 c.2.2).callBits).sum = 320 * T := by
  have h' : ((calls.map fun c => c.1 * padCols c.2.1 * c.2.2).sum : ℚ) = T * 973078528 := by exact_mod_cast h
  have e32 : ∀ c : ℕ × ℕ × ℕ, (lead T c.1 c.2.1 c.2.2).callBits =
      32 / 973078528 * ((c.1 * padCols c.2.1 * c.2.2 : ℕ) : ℚ) := fun c => by
    simp only [Pouw.Lead.Layout.callBits, lead]
    push_cast
    ring
  have e320 : ∀ c : ℕ × ℕ × ℕ, (lead320 T c.1 c.2.1 c.2.2).callBits =
      320 / 973078528 * ((c.1 * padCols c.2.1 * c.2.2 : ℕ) : ℚ) := fun c => by
    simp only [Pouw.Lead.Layout.callBits, lead320, lead]
    push_cast
    ring
  constructor
  · simp only [e32]
    rw [sum_map_mul_cast, h']
    ring
  · simp only [e320]
    rw [sum_map_mul_cast, h']
    ring

/-- Llama-3.2-1B's linears a layer, as `(n, K)`: `2048 × 3072`, `2048 × 2048`, `2048 × 16384` and `8192 × 2048`;
16 layers make its 973,078,528 MACs a token. -/
def llamaCalls : List (ℕ × ℕ) := [(3072, 2048), (2048, 2048), (16384, 2048), (2048, 8192)]

/-- **`W_mac ≤ 129.25` at Llama-3.2-1B's own calls**, at any number of rows. -/
theorem llama_betaLead_le (L : Pouw.Lead.Layout) (C : Pouw.Lead.Commit L) (hC : Pouw.Guarantees.Lead.ChunksOf L C)
    (hcw : L.chunkWords = 65536) (hr : L.rows ≤ L.m) (hc : (L.n, L.K) ∈ llamaCalls) (hS : L.S = stepsFor L.K) :
    betaLead L C ≤ 256 / 3 * (12925 / 100) := by
  simp only [llamaCalls, List.mem_cons, Prod.mk.injEq, List.not_mem_nil, or_false] at hc
  rcases hc with ⟨hn, hK⟩ | ⟨hn, hK⟩ | ⟨hn, hK⟩ | ⟨hn, hK⟩ <;>
  · refine betaLead_le_of_row L C hC hcw hr (by omega) (by omega) _ (by norm_num) ?_
    rw [hS, hK, hn]
    norm_num [stepsFor, shaWordsRegs, shaARegs]

/-- A call of Llama-3.2-1B has its own columns: they are whole MMA tiles. -/
theorem llama_padCols {c : ℕ × ℕ} (hc : c ∈ llamaCalls) : padCols c.1 = c.1 := by
  simp only [llamaCalls, List.mem_cons, List.not_mem_nil, or_false] at hc
  rcases hc with rfl | rfl | rfl | rfl <;> rfl

/-- A call of Llama-3.2-1B has a network term of at most `0.0363% · cap / 32` of its stock, at cap 32 or 320. -/
theorem llama_networkGap_le (T m : ℕ) {c : ℕ × ℕ} (hc : c ∈ llamaCalls) (L : Pouw.Lead.Layout)
    (hL : L = lead T m c.1 c.2 ∨ L = lead320 T m c.1 c.2) (C : Pouw.Lead.Commit L)
    (hC : Pouw.Guarantees.Lead.ChunksOf L C) : networkGap L C ≤ 363 / 1000000 * (L.cap / 32 : ℚ) * stockRegs L := by
  have hmem : ((lead T m c.1 c.2).n, (lead T m c.1 c.2).K) ∈ llamaCalls := by
    show (padCols c.1, c.2) ∈ llamaCalls
    rw [llama_padCols hc]
    exact hc
  rcases hL with rfl | rfl
  · refine networkGap_le_of _ C (llama_betaLead_le _ C hC rfl (le_padRows m) hmem rfl) (le_padRows m)
      (by show 0 < 973078528; norm_num) ?_
    show 256 / 3 * (12925 / 100) * ((32 : ℕ) : ℚ) ≤ 363 / 1000000 * (((32 : ℕ) : ℚ) / 32) * ((973078528 : ℕ) : ℚ)
    norm_num
  · refine networkGap_le_of _ C (llama_betaLead_le _ C hC rfl (le_padRows m) hmem rfl) (le_padRows m)
      (by show 0 < 973078528; norm_num) ?_
    show 256 / 3 * (12925 / 100) * ((320 : ℕ) : ℚ) ≤
      363 / 1000000 * (((320 : ℕ) : ℚ) / 32) * ((973078528 : ℕ) : ℚ)
    norm_num

theorem attach_sum {f : ℕ × ℕ → ℚ} :
    (llamaCalls.attach.map fun c => f c.1).sum = (llamaCalls.map f).sum := by
  rw [List.attach_map_val]

/-- **Llama-3.2-1B's forward**: at `m` rows a call, with each call's commitment at its layout, the
network terms of its calls are at most 0.0363% of their stock at cap 32 and 0.363% at cap 320. -/
theorem llama_forward (T m : ℕ) (C : ∀ c ∈ llamaCalls, Pouw.Lead.Commit (lead T m c.1 c.2))
    (C320 : ∀ c ∈ llamaCalls, Pouw.Lead.Commit (lead320 T m c.1 c.2))
    (hC : ∀ c hc, Pouw.Guarantees.Lead.ChunksOf _ (C c hc))
    (hC320 : ∀ c hc, Pouw.Guarantees.Lead.ChunksOf _ (C320 c hc)) :
    (llamaCalls.attach.map fun c => networkGap _ (C c.1 c.2)).sum ≤
        363 / 1000000 * (llamaCalls.map fun c => stockRegs (lead T m c.1 c.2)).sum ∧
      (llamaCalls.attach.map fun c => networkGap _ (C320 c.1 c.2)).sum ≤
        363 / 100000 * (llamaCalls.map fun c => stockRegs (lead320 T m c.1 c.2)).sum := by
  constructor
  · calc (llamaCalls.attach.map fun c => networkGap _ (C c.1 c.2)).sum
        ≤ (llamaCalls.attach.map fun c => 363 / 1000000 * stockRegs (lead T m c.1.1 c.1.2)).sum :=
          List.sum_le_sum fun c _ => by
            have h := llama_networkGap_le T m c.2 _ (Or.inl rfl) (C c.1 c.2) (hC c.1 c.2)
            have e : ((lead T m c.1.1 c.1.2).cap / 32 : ℚ) = 1 := by
              show ((32 : ℕ) : ℚ) / 32 = 1
              norm_num
            rwa [e, mul_one] at h
      _ = 363 / 1000000 * (llamaCalls.map fun c => stockRegs (lead T m c.1 c.2)).sum := by
          rw [attach_sum (f := fun c => 363 / 1000000 * stockRegs (lead T m c.1 c.2)), List.sum_map_mul_left]
  · calc (llamaCalls.attach.map fun c => networkGap _ (C320 c.1 c.2)).sum
        ≤ (llamaCalls.attach.map fun c => 363 / 100000 * stockRegs (lead320 T m c.1.1 c.1.2)).sum :=
          List.sum_le_sum fun c _ => by
            have h := llama_networkGap_le T m c.2 _ (Or.inr rfl) (C320 c.1 c.2) (hC320 c.1 c.2)
            have e : ((lead320 T m c.1.1 c.1.2).cap / 32 : ℚ) = 10 := by
              show ((320 : ℕ) : ℚ) / 32 = 10
              norm_num
            rw [e] at h
            linarith
      _ = 363 / 100000 * (llamaCalls.map fun c => stockRegs (lead320 T m c.1 c.2)).sum := by
          rw [attach_sum (f := fun c => 363 / 100000 * stockRegs (lead320 T m c.1 c.2)), List.sum_map_mul_left]

end Pouw.SecurityProofs.Lead.Network
