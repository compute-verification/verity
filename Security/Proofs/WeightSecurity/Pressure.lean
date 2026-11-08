module

public import Proofs.Flock.Claims.Sampling.Count
public import Proofs.WeightSecurity.Deployed

/-!
# The deployed laws' detection pressure

`TheftServedDeployed` takes `pa B ≤ x^|B|` for each window's law. It holds at every strata law (all, uniform subset,
stratified and work) whose strata each sample at most their size and miss one more incorrect proof block with
probability at most `x`, that is `fac n k 0 = (n − k)/n ≤ x`.

* `gr_le_pow`: a stratum misses `m` incorrect proof blocks with probability at most `x^m`.
* `strataPa_le_pow`: a strata sample misses `B` with probability at most `x^|B|`.
* `pa_le_pow`: so does a window's law, at its strata.
-/

@[expose] public section

namespace WeightSecurity.SecurityProofs

open Finset Flock.Guarantees Claims.Sampling Claims.Sampling.Count

theorem gr_le_pow {n k : ℕ} {x : ℝ} (hf : fac n k 0 ≤ x) (m : ℕ) : gr n k m ≤ x ^ m :=
  calc gr n k m = ∏ j ∈ range m, fac n k j := rfl
    _ ≤ ∏ _j ∈ range m, x := prod_le_prod₀ (fun j _ => fac_nonneg n k j) fun j _ => (fac_anti (Nat.zero_le j)).trans hf
    _ = x ^ m := by rw [prod_const, card_range]

theorem strataPa_le_pow (ss : List (ℕ × ℕ)) (hk : ∀ p ∈ ss, p.2 ≤ p.1) {x : ℝ} (hf : ∀ p ∈ ss, fac p.1 p.2 0 ≤ x)
    (B : Finset ℕ) (hB : ∀ v ∈ B, v < (ss.map Prod.fst).sum) : strataPa ss 0 B ≤ x ^ B.card := by
  have hsum := sum_card_window ss 0 B
  have hfil : (B.filter fun v => 0 ≤ v ∧ v < 0 + (ss.map Prod.fst).sum) = B :=
    filter_true_of_mem fun v hv => ⟨Nat.zero_le v, by rw [zero_add]; exact hB v hv⟩
  rw [hfil] at hsum
  rw [strataPa_eq_V ss hk 0 B, V, ← hsum, ← prod_pow_eq_pow_sum]
  refine prod_le_prod₀ (fun i _ => gr_nonneg _ _ _) fun i hi => gr_le_pow (hf _ ?_) _
  rw [List.getD_eq_getElem _ _ (mem_range.1 hi)]
  exact List.getElem_mem _

/-- **A window's law misses `B` with probability at most `x^|B|`** when its strata `ss` cover its population, each samples
at most its size, and each misses one more incorrect proof block with probability at most `x`. -/
theorem pa_le_pow (w : JointAuditV5) {ss : List (ℕ × ℕ)}
    (hss : (Properties.SampledProofs.oneStageOf w.law).strata w.ns = some ss) (hk : ∀ p ∈ ss, p.2 ≤ p.1)
    (hcov : (ss.map Prod.fst).sum = w.ns.sum) {x : ℝ} (hf : ∀ p ∈ ss, fac p.1 p.2 0 ≤ x) (B : Finset (Fin w.nJ)) :
    w.pa B ≤ x ^ B.card := by
  have hB : B.map Fin.valEmbedding ⊆ range w.ns.sum := fun v hv => by
    obtain ⟨u, _, rfl⟩ := mem_map.1 hv
    rw [mem_range, w.hns]
    exact u.2
  rw [JointAuditV5.pa, OneStage.Pa_strata hss hB, ← card_map Fin.valEmbedding]
  exact strataPa_le_pow ss hk hf _ fun v hv => by
    rw [hcov]
    exact mem_range.1 (hB hv)

end WeightSecurity.SecurityProofs
