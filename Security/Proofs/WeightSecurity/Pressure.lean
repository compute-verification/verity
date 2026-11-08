module

public import Proofs.Flock.Claims.Sampling.Count
public import Proofs.WeightSecurity.Deployed

/-!
# The deployed sampling designs' detection pressure

`TheftServedDeployed` takes `pa B ≤ x^|B|` for each window's sampling design. It holds at every strata design (all,
uniform subset, stratified and work) whose strata each sample at most their size and miss one more incorrect proof block
with probability at most `x`, that is `fac n k 0 = (n − k)/n ≤ x`.

* `gr_le_pow`: a stratum misses `m` incorrect proof blocks with probability at most `x^m`.
* `strataPa_le_pow`: a strata sample misses `B` with probability at most `x^|B|`.
* `pa_le_pow`: so does a window's sampling design, at its strata.
* `registered_strata`: a registered sampling design's strata cover its population and each sample at most their size.
* `pa_le_pow_of_fac`: so a registered window's sampling design misses `B` with probability at most `x^|B|` once each
  of its strata has `fac n k 0 ≤ x`.
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

/-- **A window's sampling design misses `B` with probability at most `x^|B|`** when its strata `ss` cover its
population, each samples at most its size and misses one more incorrect proof block with probability at most `x`. -/
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

/-- **A registered sampling design's strata** cover its population and each sample at most their size. -/
theorem registered_strata {law : Flock.Draw.Law} {ns : List ℕ} {kd : ℕ} (h : Flock.Bridge.LawOn law ns kd) :
    ∃ ss, (Properties.SampledProofs.oneStageOf law).strata ns = some ss ∧ (∀ p ∈ ss, p.2 ≤ p.1) ∧
      (ss.map Prod.fst).sum = ns.sum := by
  cases law with
  | subset k =>
    obtain ⟨rfl, hk⟩ := h
    exact ⟨[(ns.sum, k)], rfl, fun p hp => by rw [List.mem_singleton.1 hp]; exact hk, by simp⟩
  | bernoulli a b => exact h.elim
  | stratified K St =>
    obtain ⟨-, hK, hn, -, -⟩ := h
    have hfst : ∀ g : ℕ → ℕ, (ns.map fun n => (n, g n)).map Prod.fst = ns :=
      fun _ => List.map_map.trans (List.map_id' ns)
    refine ⟨_, rfl, fun p hp => ?_, congrArg List.sum (hfst _)⟩
    obtain ⟨n, hm, rfl⟩ := List.mem_map.1 hp
    exact Flock.Bridge.stratumK_le hK (hn _ hm) (List.le_sum_of_mem hm)
  | work K ws =>
    obtain ⟨⟨-, hsz⟩, -, -⟩ := h
    have hfst : ∀ (l : List (ℕ × ℕ × ℕ)) (g : ℕ × ℕ × ℕ → ℕ),
        (l.map fun p => (p.1, g p)).map Prod.fst = l.map Prod.fst :=
      fun _ _ => List.map_map.trans rfl
    have hlen : ns.length ≤ ((ws.toList.map (·.work)).zip (ws.toList.map (·.floor))).length := by
      rw [← hsz]
      simp
    refine ⟨_, rfl, fun p hp => ?_, ?_⟩
    · obtain ⟨q, -, rfl⟩ := List.mem_map.1 hp
      exact min_le_left _ _
    · rw [hfst, List.map_fst_zip hlen]

/-- **A registered window's sampling design misses `B` with probability at most `x^|B|`** when each of its strata misses
one more incorrect proof block with probability at most `x`. -/
theorem pa_le_pow_of_fac (w : JointAuditV5) {x : ℝ}
    (hf : ∀ ss, (Properties.SampledProofs.oneStageOf w.law).strata w.ns = some ss → ∀ p ∈ ss, fac p.1 p.2 0 ≤ x)
    (B : Finset (Fin w.nJ)) : w.pa B ≤ x ^ B.card := by
  obtain ⟨ss, hss, hk, hcov⟩ := registered_strata w.hlaw
  exact pa_le_pow w hss hk hcov (hf ss hss) B

end WeightSecurity.SecurityProofs
