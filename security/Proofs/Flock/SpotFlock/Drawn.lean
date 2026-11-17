module

public import Proofs.Flock.SpotFlock.RevealsOpen
public import Proofs.Flock.Claims.Sampling.Select.SecondCount

/-! How many proof blocks SPOT[Flock]'s two stages prove: at most `K · Σ_t k_t` besides the always-checked ones. -/

@[expose] public section

namespace SamplingDeploy

open SpotFlock (firstK select_subset_size)

theorem RegTwo.sample_size_le (g : RegTwo) {K : ℕ} (hK : firstK g.first = some K) (src : ByteArray) :
    (g.sample src).size ≤ K := by
  unfold RegTwo.sample
  cases hs : Spot.Select.select g.first.rule g.first.program g.first.query g.first.table g.first.closure src with
  | error e => exact Nat.zero_le _
  | ok r =>
    obtain ⟨draw, proved, read⟩ := r
    exact (select_subset_size hK hs).le

theorem RegTwo.proved_size_le (g : RegTwo) (src₁ src₂ : ByteArray) {draw : Lean.Json} {n : ℕ}
    (hsel : Spot.Select.selectSecond (g.sample src₁) g.ns g.ks src₂ = .ok (draw, n)) {K : ℕ}
    (hK : firstK g.first = some K) : (Spot.Draw.proved draw).size ≤ K * g.ks.toList.sum := by
  rw [Properties.Spot.SecurityProofs.size_proved_selectSecond hsel]
  exact Nat.mul_le_mul_right _ (g.sample_size_le hK src₁)

theorem RegTwo.card_biUnion_checked_le_count (g : RegTwo) (N : ℕ) (src₁ src₂ : ByteArray) {draw : Lean.Json} {n : ℕ}
    (hsel : Spot.Select.selectSecond (g.sample src₁) g.ns g.ks src₂ = .ok (draw, n)) {K : ℕ}
    (hK : firstK g.first = some K) (needs : ℕ → Finset ℕ) (M : ℕ) (hM : ∀ u, (needs u).card ≤ M) :
    ((g.checked N src₁ src₂).toFinset.biUnion needs).card ≤
      ∑ v ∈ (Finset.range N).filter (fun v => g.always v = true), (needs v).card + K * g.ks.toList.sum * M := by
  refine (g.card_biUnion_checked_le N src₁ src₂ hsel needs).trans (Nat.add_le_add_left ?_ _)
  have hs := List.sum_le_length_nsmul
    ((Spot.Draw.proved draw).toList.map fun j => (needs (g.blockAt (g.sample src₁) j.toNat)).card) M
    (by
      intro y hy
      obtain ⟨j, _, rfl⟩ := List.mem_map.1 hy
      exact hM _)
  rw [List.length_map, Array.length_toList, smul_eq_mul] at hs
  exact hs.trans (Nat.mul_le_mul_right _ (g.proved_size_le src₁ src₂ hsel hK))

end SamplingDeploy

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy

variable (h : Held)

theorem code_card_biUnion_le_count (x : Msg) (r : Fin ((code h).V.k x) → FlockDeploy.Coin) (needs : ℕ → Finset ℕ)
    {draw : Lean.Json} {n : ℕ}
    (hsel : Spot.Select.selectSecond (h.reg.sample (src (r (order h x).first))) h.reg.ns h.reg.ks
      (stageSrc (stageCoins h x r)) = .ok (draw, n)) {K : ℕ} (hK : firstK h.reg.first = some K)
    (M : ℕ) (hM : ∀ u, (needs u).card ≤ M) :
    (((code h).checked x r).toFinset.biUnion needs).card ≤
      ∑ v ∈ (Finset.range (SamplingDeploy.units ((base h).atOrder (order h)) x)).filter
          (fun v => h.reg.always v = true), (needs v).card + K * h.reg.ks.toList.sum * M :=
  (Finset.card_le_card (Finset.biUnion_subset_biUnion_of_subset_left needs fun u hu =>
    List.mem_toFinset.2 (code_checked_sub h x r u (List.mem_toFinset.1 hu)))).trans
    (h.reg.card_biUnion_checked_le_count _ _ _ hsel hK needs M hM)

end SpotFlock
