import Mathlib
import Proofs.Accounting.Device
import Proofs.Accounting.RateAudit.Partial

/-!
# Proof that the rate audit with partial completion is a joint run's bandwidth guarantee, and its rebuild term in passes

At a value of the other audits' coins, the run's bandwidth audit is a partial device of `RateAuditPartial`'s class at
`S = S_chip`, `L = ∑ cap_i · T_bw` and `q = rate_oracle · T_bw`. On a bandwidth coin where the compute account did at
least `W`, W (`ComputeAdditive`) puts the game's compute within `F·T − W`, and the game's footprint is at most its
ingress, which the bandwidth account's reads cover: the joint event is inside `RateAuditPartial`'s at `F = F·T − W`
(`RateOfAuditPartial`). The passes form follows from the floor by `x₁ · b · F ≤ c_root · BW` (`passes_floor`).

In the device model W holds whenever the game's compute is charged to the accounts other than the compute account
(`computeAdditive_of_othersWork`, `RealizesRate`'s `compute_le`), and `WorkHardness` holds for a game whose every
completion costs a whole word (`workHardness_whole`), `RateAudit`'s all-or-nothing rebuild.
-/

namespace Accounting.SecurityProofs

open Accounting Accounting.Rate

/-- W holds for a joint run whose bandwidth game's compute the work of the accounts other than the compute account
covers: every op is charged once (`Step.Fits`). -/
theorem computeAdditive_of_othersWork {N b h ρ s : ℕ} {ι : Type} [Fintype ι] {d : Device ι} {α ΩW ΩS : Type}
    [Fintype α] [DecidableEq α] {r : Run d α ΩW (RateCoins N b h ρ s) ΩS} (real : RealizesRatePartial r) (p : Price)
    (cTag : ℕ) (cost : ℕ → ℕ)
    (hle : ∀ x y s', (real.dev (x, s')).compute cTag cost y.1 y.2.1 y.2.2.1 ≤ r.othersWork p (x, y, s')) :
    Assumptions.ComputeAdditive p cTag cost real := fun x y s' => by
  have h1 := hle x y s'
  have h2 := r.work_add_othersWork p (x, y, s')
  omega

/-- `WorkHardness` holds when every completion costs a whole word, as `RateAudit`'s rebuild does. -/
theorem workHardness_whole (b rounds cRoot : ℕ) :
    Assumptions.WorkHardness b rounds cRoot (fun _ => rounds * cRoot) := fun _ hx => Nat.mul_le_mul_right _ hx

theorem RateOfAuditPartial : Accounting.Guarantees.RateOfAuditPartial := by
  intro N b h ρ s k cTag rounds cRoot lam W cost p ι α ΩW ΩS _ _ d r real hw hadd x s'
  refine (Bandwidth.pr_mono (fun y hy => ?_)).trans
    (RateAuditPartial N b h ρ d.chip (∑ i, d.cap i * (r.win .bandwidth).len) (d.rate .oracle * (r.win .bandwidth).len)
      s k (d.priced p * (r.win .compute).len - W) cTag rounds cRoot lam cost (real.dev (x, s')) hw)
  obtain ⟨hacc, hW, hread⟩ := hy
  refine ⟨(real.accepts_iff x y s').1 hacc, ?_, ?_⟩
  · have h1 := hadd x y s'
    omega
  · have h1 := PChip.card_addrs_le ((real.dev (x, s')).world y.1 y.2.1 y.2.2.1)
      ((real.dev (x, s')).chip ((real.dev (x, s')).state y.1))
    have h3 : (real.dev (x, s')).footprint y.1 y.2.1 y.2.2.1 * (rounds * cRoot) ≤
        r.read (x, y, s') * (rounds * cRoot) :=
      Nat.mul_le_mul_right _ (h1.trans (real.ingress_le x y s'))
    exact lt_of_le_of_lt h3 hread

/-- The passes form's event is inside the floor's: with a word's one-round encode at least `x₁` times its read
(`x₁ · b · F ≤ c_root · BW`), the completions' `(F·T − W − k·c_tag)·b/(rounds · c_root)` bits are at most
`BW·(F·T − W − k·c_tag)/(rounds · x₁ · F)`. -/
theorem passes_floor {b ρ S L P T W cTag rounds cRoot x₁ k lam bw read : ℕ} (hx : x₁ * b * P ≤ cRoot * bw)
    (h : read * (rounds * x₁ * P) <
      (k * b - (S + (L - ρ) + lam)) * (rounds * x₁ * P) - bw * (P * T - W - k * cTag)) :
    read * (rounds * cRoot) < floor b ρ S L (P * T - W) cTag (rounds * cRoot) k lam := by
  unfold floor
  generalize hS : S + (L - ρ) + lam = S' at h ⊢
  generalize hD : P * T - W - k * cTag = D at h ⊢
  generalize hM : rounds * x₁ * P = M at h
  have hMpos : 0 < M := by
    rcases Nat.eq_zero_or_pos M with h0 | h0
    · rw [h0] at h; simp at h
    · exact h0
  have hx₁ : 0 < x₁ := Nat.pos_of_ne_zero (fun h0 => by subst h0; simp at hM; omega)
  have hP : 0 < P := Nat.pos_of_ne_zero (fun h0 => by subst h0; simp at hM; omega)
  have h1 := Nat.lt_sub_iff_add_lt.1 h
  have hkS : S' ≤ k * b := by
    by_contra hc
    rw [Nat.sub_eq_zero_of_le (by omega), zero_mul] at h1
    omega
  have h2 : (read + S') * M + bw * D < k * b * M := by
    have e : (k * b - S') * M + S' * M = k * b * M := by rw [← add_mul, Nat.sub_add_cancel hkS]
    nlinarith
  rcases Nat.eq_zero_or_pos cRoot with hc0 | hc0
  · exfalso
    subst hc0
    have hb : b = 0 := by
      have : x₁ * b * P = 0 := Nat.le_zero.1 (by simpa using hx)
      rcases Nat.mul_eq_zero.1 this with h3 | h3
      · rcases Nat.mul_eq_zero.1 h3 with h4 | h4
        · omega
        · exact h4
      · omega
    subst hb
    simp at h2
  have h3 : ((read + S') * M + bw * D) * cRoot < k * b * M * cRoot := Nat.mul_lt_mul_of_pos_right h2 hc0
  have h4 : D * (x₁ * b * P) ≤ D * (cRoot * bw) := Nat.mul_le_mul_left D hx
  have key : (read * (rounds * cRoot) + (S' * (rounds * cRoot) + D * b)) * (x₁ * P) <
      k * b * (rounds * cRoot) * (x₁ * P) := by
    rw [← hM] at h3
    nlinarith
  have := Nat.lt_of_mul_lt_mul_right key
  omega

theorem RateOfAuditPartialPasses : Accounting.Guarantees.RateOfAuditPartialPasses := by
  intro N b h ρ s k cTag rounds cRoot x₁ lam W cost p ι α ΩW ΩS _ _ d r real hw hadd hx x s'
  refine (Bandwidth.pr_mono (fun y hy => ?_)).trans
    (RateOfAuditPartial N b h ρ s k cTag rounds cRoot lam W cost p ι α ΩW ΩS d r real hw hadd x s')
  obtain ⟨hacc, hW, hread⟩ := hy
  exact ⟨hacc, hW, passes_floor hx hread⟩

end Accounting.SecurityProofs
