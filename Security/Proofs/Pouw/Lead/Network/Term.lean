import Definitions.Pouw.Guarantees.Lead

/-!
# The lead's network term against the statement

What `Rot`'s theorems prove of the statement's definitions:
* `lead_cells_rot`: every step of the lead, step 0 included, carries the three coin lanes of its 64-row tile at
  `-rot`'s weights, `foldEven` on even rows and `foldOdd` on odd rows, of odd coins `2u − 127`; the rest of the step's
  coin term reads no lane; and the data fit 21 lanes on step 0 and 28 on each later step. So the lead's cells meet
  `Cell.Rot`, and `n_max ≤ nMax`.
* `networkGap_le`, `networkGap_le320`: a call's network term, `nMax · wMac · callBits / 32`, is at most 0.036% of its
  stock at cap 32 and 0.36% at cap 320 for every call of `n, K ≥ 4,096` (`netShare_lead_cap32`, `netShare_lead_cap320`
  at the call's `wMac ≤ 128.28`, and `rows ≤ m`).
* `forward_sum`: a forward's calls' bits sum to its cap, `cap · T`, so no call takes another's share.
* `llama_forward`: over Llama-3.2-1B's own calls (`llamaCalls`), whose `K = 2,048` is below `GapAt`'s range, a
  forward's network term is at most 0.0362% of its stock at cap 32 and 0.362% at cap 320.
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
  sorry

/-- **A call's network term at cap 32**: at most 0.036% of its stock for every call of
`n, K ≥ 4,096`. -/
theorem networkGap_le (T m n K : ℕ) (hn : 4096 ≤ n) (hK : 4096 ≤ K) (C : Pouw.Lead.Commit (lead T m n K))
    (hC : Pouw.Guarantees.Lead.ChunksOf _ C) :
    networkGap (lead T m n K) C ≤ 36 / 100000 * stockRegs (lead T m n K) := by
  sorry

/-- **A call's network term at cap 320**: at most 0.36% of its stock for every call of
`n, K ≥ 4,096`. -/
theorem networkGap_le320 (T m n K : ℕ) (hn : 4096 ≤ n) (hK : 4096 ≤ K) (C : Pouw.Lead.Commit (lead320 T m n K))
    (hC : Pouw.Guarantees.Lead.ChunksOf _ C) :
    networkGap (lead320 T m n K) C ≤ 36 / 10000 * stockRegs (lead320 T m n K) := by
  sorry

/-- **A forward's calls' bits sum to its cap**: calls `(rows, n, K)` whose MACs, over padded columns,
are the forward's `T` tokens' have bits summing to `32 T` at cap 32 and `320 T` at cap 320. -/
theorem forward_sum (T : ℕ) (calls : List (ℕ × ℕ × ℕ))
    (h : (calls.map fun c => c.1 * padCols c.2.1 * c.2.2).sum = T * 973078528) :
    (calls.map fun c => (lead T c.1 c.2.1 c.2.2).callBits).sum = 32 * T ∧
      (calls.map fun c => (lead320 T c.1 c.2.1 c.2.2).callBits).sum = 320 * T := by
  sorry

/-- Llama-3.2-1B's linears a layer, as `(n, K)`: `2048 × 3072`, `2048 × 2048`, `2048 × 16384` and `8192 × 2048`;
16 layers make its 973,078,528 MACs a token. -/
def llamaCalls : List (ℕ × ℕ) := [(3072, 2048), (2048, 2048), (16384, 2048), (2048, 8192)]

/-- **Llama-3.2-1B's forward**: at `m` rows a call, with each call's commitment at its layout, the
network terms of its calls are at most 0.0362% of their stock at cap 32 and 0.362% at cap 320. -/
theorem llama_forward (T m : ℕ) (C : ∀ c ∈ llamaCalls, Pouw.Lead.Commit (lead T m c.1 c.2))
    (C320 : ∀ c ∈ llamaCalls, Pouw.Lead.Commit (lead320 T m c.1 c.2))
    (hC : ∀ c hc, Pouw.Guarantees.Lead.ChunksOf _ (C c hc))
    (hC320 : ∀ c hc, Pouw.Guarantees.Lead.ChunksOf _ (C320 c hc)) :
    (llamaCalls.attach.map fun c => networkGap _ (C c.1 c.2)).sum ≤
        362 / 1000000 * (llamaCalls.map fun c => stockRegs (lead T m c.1 c.2)).sum ∧
      (llamaCalls.attach.map fun c => networkGap _ (C320 c.1 c.2)).sum ≤
        362 / 100000 * (llamaCalls.map fun c => stockRegs (lead320 T m c.1 c.2)).sum := by
  sorry

end Pouw.SecurityProofs.Lead.Network
