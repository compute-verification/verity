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
* `networkFwd` (`Pouw.Guarantees.Lead.NetworkFwd`): under `SerialForwards`, an accepted forward's network term is at
  most the largest `β` of its calls times its bits over 32: its calls' exchanged bits are at most what the certifier
  counted, which is at most its bits.
* `forward_sum`: a forward's calls' bits sum to its cap, `cap · T`.
* `networkLlama` (`Pouw.Guarantees.Lead.NetworkLlama`): over Llama-3.2-1B's own calls (`llamaCalls`), a forward's
  network term is at most `0.0363% · cap / 32` of its calls' served stock (`llama_betaLead_le`: the calls'
  `wMac ≤ 129.25`, `2,048 × 2,048` the largest; `llama_call`: a call's bits are `32 · cap / fwdMacs` of its served
  stock, its columns whole MMA tiles).

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

/-! ## A forward's network term -/

/-- **A forward's network term** (`Pouw.Guarantees.Lead.NetworkFwd`): each call's `β` is at most `B`, the calls'
exchanged bits are at most the certifier's count and release (`SerialForwards`), and those are at most the forward's
bits (`Forward.Accepted`). -/
theorem networkFwd : Pouw.Guarantees.Lead.NetworkFwd := by
  intro F B hser hacc hB hβ
  have h1 : (F.calls.map fun c => betaLead c.L c.C * c.egress).sum ≤
      (F.calls.map fun c => B * (c.egress : ℚ)).sum :=
    List.sum_le_sum fun c hc => mul_le_mul_of_nonneg_right (hβ c hc) (Nat.cast_nonneg _)
  have h2 : (F.calls.map fun c => B * (c.egress : ℚ)).sum = B * ((F.calls.map FwdCall.egress).sum : ℚ) :=
    sum_map_mul_cast _ _ _
  have h3 : ((F.calls.map FwdCall.egress).sum : ℚ) ≤ F.bits := by
    have : ((F.calls.map FwdCall.egress).sum : ℚ) ≤ ((F.count + F.release : ℕ) : ℚ) := Nat.cast_le.2 hser
    exact this.trans hacc
  unfold Forward.networkRegs
  rw [h2] at h1
  exact div_le_div_of_nonneg_right (h1.trans (mul_le_mul_of_nonneg_left h3 hB)) (by norm_num)

/-- **A call of Llama-3.2-1B**, at the lead or at cap 320, at any number of rows: `β ≤ 256 / 3 · 129.25`, and its bits
are `1,024 / fwdMacs · cap / 32` of its served stock. -/
theorem llama_call {L : Pouw.Lead.Layout} (C : Pouw.Lead.Commit L)
    (hL : ∃ T m : ℕ, ∃ x ∈ llamaCalls, L = lead T m x.1 x.2 ∨ L = lead320 T m x.1 x.2)
    (hC : Pouw.Guarantees.Lead.ChunksOf L C) :
    betaLead L C ≤ 256 / 3 * (12925 / 100) ∧
      L.callBits = 1024 / 973078528 * ((L.cap / 32 : ℚ) * servedRegs L) := by
  obtain ⟨T, m, x, hx, hL⟩ := hL
  have hcols : padCols x.1 = x.1 := llama_padCols hx
  have hmem : ((lead T m x.1 x.2).n, (lead T m x.1 x.2).K) ∈ llamaCalls := by
    show (padCols x.1, x.2) ∈ llamaCalls
    rw [hcols]
    exact hx
  rcases hL with rfl | rfl
  · refine ⟨llama_betaLead_le _ C hC rfl (le_padRows m) hmem rfl, ?_⟩
    simp only [Pouw.Lead.Layout.callBits, servedRegs, lead, hcols]
    push_cast
    ring
  · refine ⟨llama_betaLead_le _ C hC rfl (le_padRows m) hmem rfl, ?_⟩
    simp only [Pouw.Lead.Layout.callBits, servedRegs, lead320, lead, hcols]
    push_cast
    ring

/-- **Llama-3.2-1B's forward** (`Pouw.Guarantees.Lead.NetworkLlama`): `networkFwd` at `B = 256 / 3 · 129.25`
(`llama_call`), and `B · 32 / fwdMacs ≤ 0.0363%`. -/
theorem networkLlama : Pouw.Guarantees.Lead.NetworkLlama := by
  intro F hser hacc hcalls
  have h := networkFwd F (256 / 3 * (12925 / 100)) hser hacc (by norm_num) fun c hc =>
    (llama_call c.C (hcalls c hc).1 (hcalls c hc).2).1
  have hbits : F.bits = 1024 / 973078528 * (F.calls.map fun c => (c.L.cap / 32 : ℚ) * servedRegs c.L).sum := by
    unfold Forward.bits
    rw [List.map_congr_left fun c hc => (llama_call c.C (hcalls c hc).1 (hcalls c hc).2).2,
      List.sum_map_mul_left]
  have hs : 0 ≤ (F.calls.map fun c => (c.L.cap / 32 : ℚ) * servedRegs c.L).sum :=
    List.sum_nonneg fun y hy => by
      obtain ⟨c, -, rfl⟩ := List.mem_map.1 hy
      unfold servedRegs
      positivity
  rw [hbits] at h
  linarith

end Pouw.SecurityProofs.Lead.Network
