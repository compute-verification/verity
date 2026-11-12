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
* `fac_le_of_rate`, `pa_le_pow_of_rate`: a sampling design that samples each stratum at the rate `num/den` or more, as
  a rate rule's does, has `x = 1 − num/den`, whatever sizes the prover registers.
* `TheftServedDeployedRate`: `TheftServedDeployed` at such designs, with `(1 − num/den)^M` in its bound.
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
    (hss : (Properties.Spot.oneStageOf w.law).strata w.ns = some ss) (hk : ∀ p ∈ ss, p.2 ≤ p.1)
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
    ∃ ss, (Properties.Spot.oneStageOf law).strata ns = some ss ∧ (∀ p ∈ ss, p.2 ≤ p.1) ∧
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
    (hf : ∀ ss, (Properties.Spot.oneStageOf w.law).strata w.ns = some ss → ∀ p ∈ ss, fac p.1 p.2 0 ≤ x)
    (B : Finset (Fin w.nJ)) : w.pa B ≤ x ^ B.card := by
  obtain ⟨ss, hss, hk, hcov⟩ := registered_strata w.hlaw
  exact pa_le_pow w hss hk hcov (hf ss hss) B

/-- **A stratum sampled at the rate `num/den` or more** (`num · n ≤ den · k`) misses one more incorrect proof block
with probability at most `1 − num/den`. -/
theorem fac_le_of_rate {num den n k : ℕ} (hd : 0 < den) (hq : num ≤ den) (h : num * n ≤ den * k) :
    fac n k 0 ≤ 1 - (num : ℝ) / den := by
  have hden : (0 : ℝ) < den := by exact_mod_cast hd
  have hq1 : 0 ≤ 1 - (num : ℝ) / den := sub_nonneg.2 ((div_le_one₀ hden).2 (by exact_mod_cast hq))
  unfold fac
  simp only [Nat.sub_zero]
  rcases Nat.eq_zero_or_pos n with rfl | hn
  · rw [Nat.cast_zero, div_zero]
    exact hq1
  rcases le_or_gt n k with hk | hk
  · rw [Nat.sub_eq_zero_of_le hk, Nat.cast_zero, zero_div]
    exact hq1
  have hn' : (0 : ℝ) < n := by exact_mod_cast hn
  have h' : (num : ℝ) * n ≤ k * den := by exact_mod_cast h.trans_eq (Nat.mul_comm den k)
  rw [Nat.cast_sub hk.le, sub_div, div_self hn'.ne']
  exact sub_le_sub_left ((div_le_div_iff₀ hden hn').2 h') 1

/-- **A registered window's sampling design at a rate** misses `B` with probability at most `(1 − num/den)^|B|` when
each of its strata samples at the rate `num/den` or more, however the prover sizes them. -/
theorem pa_le_pow_of_rate (w : JointAuditV5) {num den : ℕ} (hd : 0 < den) (hq : num ≤ den)
    (hr : ∀ ss, (Properties.Spot.oneStageOf w.law).strata w.ns = some ss → ∀ p ∈ ss, num * p.1 ≤ den * p.2)
    (B : Finset (Fin w.nJ)) : w.pa B ≤ (1 - (num : ℝ) / den) ^ B.card :=
  pa_le_pow_of_fac w (fun ss hss p hp => fac_le_of_rate hd hq (hr ss hss p hp)) B

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden WeightSecurity.Assumptions in
/-- **Model weight security at the deployed audit, at a rate.** `TheftServedDeployed` when each window's sampling
design samples every stratum at the rate `num/den` or more: the audit leaves `M` incorrect proof blocks undetected with
probability at most the windows' knowledge and binding terms plus `(1 − num/den)^M`, whatever populations the prover
registers. -/
theorem TheftServedDeployedRate {Ω R Θ Q Adv O V Tick Θ' : Type} [Fintype Ω] [Fintype R] [Fintype Adv]
    [Inhabited Adv] [Fintype O] [DecidableEq O] [Fintype V] [Fintype Tick] [Inhabited Tick]
    (W : ℕ → JointAuditV5) (T L E M num den : ℕ) (hd : 0 < den) (hq : num ≤ den)
    (C : Θ → Q × Adv → O → V) (gen : Ω → Θ) (Stolen : Θ → Θ' → Prop) (A : Set (Outsider R (Q × Adv) (O → V) Θ'))
    (p : ℝ) (Ou : Outsider R (Q × Adv) (Option ((O → V) × Tick)) Θ')
    (s : Θ → R → Strategy (periodG (fun t => servedGame (W t) (Release O V Tick Adv)) 0 T))
    (hhon : HonestUseHard C gen Stolen T A p)
    (hsim : Simulable A Ou (servedChannel R Q Adv O V Tick T (E * M + (L + 1) * Fintype.card O)))
    (hA3 : ∀ t < T, Assumptions.UniformRandomBytes (W t).Lb (W t).src)
    (hr : ∀ t < T, ∀ ss, (Properties.Spot.oneStageOf (W t).law).strata (W t).ns = some ss →
      ∀ p ∈ ss, num * p.1 ≤ den * p.2)
    (hT : ∀ θ r t (τ : Strategy (servedGame (W t) (Release O V Tick Adv))),
      PeriodPlays (fun t => servedGame (W t) (Release O V Tick Adv)) 0 T (s θ r) t τ → ∀ k ω j,
      Assumptions.TableCRZAt execArith hs512
        (planZHJP ((W t).ms.get k).dj ((W t).ms.get k).y₀ (((W t).memberLaw (servedAudit (W t) τ) k).draw ω)
          (reg ((W t).memberSim (servedAudit (W t) τ) k))) j
        (zcStrat execArith hs512 (planZHJP ((W t).ms.get k).dj ((W t).ms.get k).y₀)
          (cont ((W t).memberSim (servedAudit (W t) τ) k) ω))
        ((W t).ms.get k).qF ((W t).ms.get k).qS)
    (hL : ∀ θ r t (τ : Strategy (servedGame (W t) (Release O V Tick Adv))),
      PeriodPlays (fun t => servedGame (W t) (Release O V Tick Adv)) 0 T (s θ r) t τ → ∀ k,
      Assumptions.LinkCRZCAt execArith hs512 ((W t).ms.get k).P
        (planZHJP ((W t).ms.get k).dj ((W t).ms.get k).y₀) (tabZHJP ((W t).ms.get k).dj ((W t).ms.get k).y₀)
        ((W t).ms.get k).vb (reg ((W t).memberSim (servedAudit (W t) τ) k))
        (cont ((W t).memberSim (servedAudit (W t) τ) k)) ((W t).ms.get k).k ((W t).ms.get k).Rw ((W t).ms.get k).M
        ((W t).ms.get k).t') :
    Verity.Game.avg (fun z : Ω × R => Verity.Game.value
        (fun l => Stolen (gen z.1) (Ou.out z.2 (Served.view Ou L z.2 l T))) (servedRun W T (s (gen z.1) z.2))) ≤
      ENNReal.ofReal (Fintype.card (servedChannel R Q Adv O V Tick T (E * M + (L + 1) * Fintype.card O)).S * p) +
        (∑ t ∈ Finset.range T, (W t).eps + ENNReal.ofReal ((1 - (num : ℝ) / den) ^ M)) +
        Verity.Game.avg (fun z : Ω × R => Verity.Game.value
          (fun l => ¬ Served.Local C E Ou L (gen z.1) z.2 l) (servedRun W T (s (gen z.1) z.2))) := by
  have hden : (0 : ℝ) < den := by exact_mod_cast hd
  have h1 : (num : ℝ) / den ≤ 1 := (div_le_one₀ hden).2 (by exact_mod_cast hq)
  have h0 : 0 ≤ (num : ℝ) / den := div_nonneg (Nat.cast_nonneg _) hden.le
  exact TheftServedDeployed W T L E M _ (sub_nonneg.2 h1) (sub_le_self _ h0) C gen Stolen A p Ou s hhon hsim hA3
    (fun t ht B => pa_le_pow_of_rate (W t) hd hq (hr t ht) B) hT hL

end WeightSecurity.SecurityProofs
