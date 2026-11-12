module

public import Proofs.Flock.EndToEnd.Joint.V5.Member
public import Proofs.Flock.EndToEnd.Joint.Prod
public import Proofs.Flock.Soundness.Game

/-!
# Proof of the bound over a window's members, per window and pooled

The proofs of `JointSampledV5`, `JointShareV5` and `JointHorizonV5` (`Joint/V5/Statement.lean`):

* `JointAuditV5.accepted_drawnWrong_le`: the window's audit accepts with a sampled proof block incorrect with
  probability at most `Σ_m (ε_ks,m + ε_bind,m)`: an incorrect sampled proof block is member `m`'s, in its part of the
  sample, and an accepted run accepted the session of member `m`, which has that proof block in its part
  (`forall₂_getElem?`), so a union over the members (`ZkSession.prob_exists_le`) of
  `JointAuditV5.member_accepted_drawnWrong_le` bounds it;
* `JointAuditV5.drawMiss_le`: the sample misses a set `B` with probability at most the sampling design's own miss `P_a(B)`, under
  A3 (`Bridge.execSel_escape_le`);
* `Flock.SecurityProofs.JointSampledV5`: an accepted run with every proof block of `W` incorrect either has a sampled
  proof block incorrect or its sample misses `W`;
* `Flock.SecurityProofs.JointShareV5`: `JointSampledV5` at the window's incorrect proof blocks, whose `P_a` is at most
  the most of any set of as many;
* `Flock.SecurityProofs.JointHorizonV5`: a run in which every window is accepted and the incorrect proof blocks are not
  in the period's confidence set either has a window accepted with a sampled proof block incorrect, at most
  `Σ_t Σ_m (ε_ks,t,m + ε_bind,t,m)` by `period_union`, or every sample misses with the product of the `P_a`s below
  `δ`, at most `δ` by `period_prod`.
-/

@[expose] public section

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden
open scoped ENNReal

/-- **Every pair of `List.Forall₂`**: the `k`-th of the first list has its partner at the second's `k`-th place. -/
theorem forall₂_getElem? {α β : Type} {R : α → β → Prop} :
    ∀ {l₁ : List α} {l₂ : List β}, List.Forall₂ R l₁ l₂ → ∀ k : Fin l₁.length, ∃ t, l₂[k.1]? = some t ∧ R (l₁.get k) t
  | [], [], .nil, k => k.elim0
  | _ :: _, b :: _, .cons h _, ⟨0, _⟩ => ⟨b, rfl, h⟩
  | _ :: _, _ :: _, .cons _ hs, ⟨k + 1, hk⟩ => forall₂_getElem? hs ⟨k, Nat.lt_of_succ_lt_succ hk⟩

namespace JointAuditV5

open Classical

variable (w : JointAuditV5)

/-- **The window's audit accepts with a sampled proof block incorrect** at the strategy `τ` that plays it: at most
`Σ_m (ε_ks,m + ε_bind,m)`, under `cr/sha-512` and `ecr/sha-512` at each member's session played alone. -/
theorem accepted_drawnWrong_le (τ : Strategy w.game)
    (hCR : ∀ k ω j, TableCRZ execArith hs512
      (planZHJP (w.ms.get k).dj (w.ms.get k).y₀ ((w.memberLaw τ k).draw ω) (reg (w.memberSim τ k))) j
      (zcStrat execArith hs512 (planZHJP (w.ms.get k).dj (w.ms.get k).y₀) (cont (w.memberSim τ k) ω))
      (w.ms.get k).qF (w.ms.get k).qS)
    (hA2 : ∀ k, LinkCRZC execArith hs512 (w.ms.get k).P (planZHJP (w.ms.get k).dj (w.ms.get k).y₀)
      (tabZHJP (w.ms.get k).dj (w.ms.get k).y₀) (w.ms.get k).vb (reg (w.memberSim τ k)) (cont (w.memberSim τ k))
      (w.ms.get k).k (w.ms.get k).Rw (w.ms.get k).M (w.ms.get k).t') :
    prob (fun o => w.Accepts o ∧ ∃ u ∈ o.2.1, u ∈ w.wrongSet τ) w.game τ ≤ w.eps := by
  refine (prob_mono (q := fun o : w.Reg × Finset (Fin w.nJ) × List Discharge.Exec.Transcript =>
      ∃ k ∈ Finset.univ, (o.2.2[k.1]?).elim False (fun t => AcceptsZKP (w.ms.get k).I (w.ms.get k).own t ∧
        ∃ u ∈ (w.ms.get k).part o.2.1, u ∈ w.memberWrong τ k)) (fun o ho => ?_) _ _).trans
    ((FlockSoundness.Discharge.ZkSession.prob_exists_le _ _ _ _).trans
      (Finset.sum_le_sum fun k _ => w.member_accepted_drawnWrong_le τ k (hCR k) (hA2 k)))
  obtain ⟨hacc, u, hu, hw⟩ := ho
  obtain ⟨k, -, hk⟩ := Finset.mem_biUnion.1 hw
  obtain ⟨v, hv, rfl⟩ := Finset.mem_image.1 hk
  obtain ⟨t, ht, hacc'⟩ := forall₂_getElem? hacc k
  have hv' : v ∈ (w.ms.get k).part o.2.1 := Finset.mem_filter.2 ⟨Finset.mem_univ _, hu⟩
  refine ⟨k, Finset.mem_univ _, ?_⟩
  rw [ht]
  exact And.intro (hacc' ⟨v, hv'⟩) ⟨v, hv', hv⟩

private theorem prob_const {α : Type} (Q : Prop) [Decidable Q] (g : Game α) (s : Strategy g) :
    prob (fun _ => Q) g s = if Q then 1 else 0 := by
  by_cases hQ : Q
  · simp only [hQ, ite_true]
    exact prob_true g s
  · simp only [hQ, ite_false]
    exact prob_false g s

/-- **The window's sample misses a set `B`** with the acceptance probability of its live sampling design. -/
theorem prob_drawMiss (B : Finset (Fin w.nJ)) (τ : Strategy w.game) :
    prob (fun o => ¬ ∃ u ∈ o.2.1, u ∈ B) w.game τ = w.L.escape B := by
  show avg (fun ω => prob _ ((w.sessions (w.L.draw ω) τ.1).map fun ts => (τ.1, w.L.draw ω, ts)) (τ.2 ω)) =
    prCoin fun ω => Disjoint (w.L.draw ω) B
  unfold prCoin
  refine congrArg avg (funext fun ω => ?_)
  rw [prob_map, prob_const (¬ ∃ u ∈ w.L.draw ω, u ∈ B)]
  beta_reduce
  have hiff : (¬ ∃ u ∈ w.L.draw ω, u ∈ B) ↔ Disjoint (w.L.draw ω) B := by simp [Finset.disjoint_left]
  by_cases hd : Disjoint (w.L.draw ω) B
  · rw [ite_eq_left (hiff.2 hd), ite_eq_left hd]
  · rw [ite_eq_right (mt hiff.1 hd), ite_eq_right hd]

/-- **The window's sample misses a set `B`** with probability at most the sampling design's own miss `P_a(B)`, under A3. -/
theorem drawMiss_le (hA3 : Assumptions.UniformRandomBytes w.Lb w.src) (B : Finset (Fin w.nJ)) (τ : Strategy w.game) :
    prob (fun o => ¬ ∃ u ∈ o.2.1, u ∈ B) w.game τ ≤ ENNReal.ofReal (w.pa B) := by
  rw [prob_drawMiss]
  exact Bridge.execSel_escape_le w.src hA3 w.hlaw w.hns B

end JointAuditV5

namespace JointPeriodV5

variable {W : ℕ → JointAuditV5}

/-- A run whose `P_a`s multiply to at least `δ` holds at most the period's `maxHarm` of any weight. -/
theorem harm_le_maxHarm (h : (t : ℕ) → Fin (W t).nJ → ℕ) {δ : ℝ} {l : List (AuditRecV5 W)}
    (hl : ENNReal.ofReal δ ≤ (l.map AuditRecV5.pa).prod) :
    (((l.map (AuditRecV5.harm h)).sum : ℕ) : ℝ) ≤ maxHarm W h δ l := by
  have hB' : ENNReal.ofReal δ ≤ ∏ i : Fin l.length, AuditRecV5.pa l[i.1] := by
    rw [Fin.prod_univ_fun_getElem]; exact hl
  have hB : ENNReal.ofReal δ ≤ ∏ i : Fin l.length, ENNReal.ofReal ((W l[i].idx).pa l[i].wrong) :=
    hB'.trans_eq (Finset.prod_congr rfl fun _ _ => rfl)
  have e : (l.map (AuditRecV5.harm h)).sum = ∑ i : Fin l.length, ∑ u ∈ l[i].wrong, h l[i].idx u := by
    rw [← Fin.sum_univ_fun_getElem]; exact Finset.sum_congr rfl fun _ _ => rfl
  have hb : BddAbove (Set.range fun B : {B : (i : Fin l.length) → Finset (Fin (W l[i].idx).nJ) //
      ENNReal.ofReal δ ≤ ∏ i, ENNReal.ofReal ((W l[i].idx).pa (B i))} =>
        ((∑ i, ∑ u ∈ B.1 i, h l[i].idx u : ℕ) : ℝ)) :=
    (Set.finite_range _).bddAbove
  rw [e]
  exact le_ciSup hb ⟨fun i => l[i].wrong, hB⟩

/-- **A run whose `P_a`s multiply to at least `δ` is in the period's confidence set.** -/
theorem within_of_le {δ : ℝ} (ands bits : (t : ℕ) → Fin (W t).nJ → ℕ) {l : List (AuditRecV5 W)}
    (hl : ENNReal.ofReal δ ≤ (l.map AuditRecV5.pa).prod) : Within W δ ands bits l :=
  ⟨hl, harm_le_maxHarm ands hl, harm_le_maxHarm bits hl⟩

end JointPeriodV5

end Flock.Guarantees

namespace Flock.SecurityProofs

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden Flock.Guarantees
open scoped ENNReal

/-- **A window's sample over several members, `EndToEndHiddenV5`'s form.** -/
theorem JointSampledV5 : Flock.Guarantees.JointSampledV5 := by
  intro w hA3 τ hCR hA2 W
  classical
  calc prob (fun o => w.Accepts o ∧ W ⊆ w.wrongSet τ) w.game τ
      ≤ prob (fun o => (w.Accepts o ∧ ∃ u ∈ o.2.1, u ∈ w.wrongSet τ) ∨ ¬ ∃ u ∈ o.2.1, u ∈ W) w.game τ := by
        refine prob_mono (fun o ho => ?_) _ _
        by_cases hd : ∃ u ∈ o.2.1, u ∈ W
        · obtain ⟨u, hu, hw⟩ := hd
          exact Or.inl ⟨ho.1, u, hu, ho.2 hw⟩
        · exact Or.inr hd
    _ ≤ prob (fun o => w.Accepts o ∧ ∃ u ∈ o.2.1, u ∈ w.wrongSet τ) w.game τ +
          prob (fun o => ¬ ∃ u ∈ o.2.1, u ∈ W) w.game τ := prob_or_le _ _ _ _
    _ ≤ w.eps + ENNReal.ofReal (w.pa W) := add_le_add (w.accepted_drawnWrong_le τ hCR hA2) (w.drawMiss_le hA3 W τ)
    _ = ENNReal.ofReal (w.pa W) + w.eps := add_comm _ _

/-- **A window's sample over several members, the share form.** -/
theorem JointShareV5 : Flock.Guarantees.JointShareV5 := by
  intro w hA3 τ hCR hA2 ε
  classical
  by_cases hc : ε * w.nJ < (w.wrongSet τ).card
  · have hle : w.pa (w.wrongSet τ) ≤ ⨆ B : {B : Finset (Fin w.nJ) // ε * w.nJ < B.card}, w.pa B.1 :=
      le_ciSup (f := fun B : {B : Finset (Fin w.nJ) // ε * w.nJ < B.card} => w.pa B.1)
        (Set.finite_range _).bddAbove ⟨w.wrongSet τ, hc⟩
    calc prob (fun o => w.Accepts o ∧ ε * w.nJ < (w.wrongSet τ).card) w.game τ
        ≤ prob (fun o => w.Accepts o ∧ w.wrongSet τ ⊆ w.wrongSet τ) w.game τ :=
          prob_mono (fun _ ho => ⟨ho.1, subset_rfl⟩) _ _
      _ ≤ ENNReal.ofReal (w.pa (w.wrongSet τ)) + w.eps := JointSampledV5 w hA3 τ hCR hA2 _
      _ ≤ _ := add_le_add (ENNReal.ofReal_le_ofReal hle) le_rfl
  · calc prob (fun o => w.Accepts o ∧ ε * w.nJ < (w.wrongSet τ).card) w.game τ
        ≤ prob (fun _ => False) w.game τ := prob_mono (fun _ ho => hc ho.2) _ _
      _ = 0 := prob_false _ _
      _ ≤ _ := zero_le

/-- **A monitoring period of windows of several members, pooled.** -/
theorem JointHorizonV5 : Flock.Guarantees.JointHorizonV5 := by
  intro W T δ ands bits hA3 s hT hL
  classical
  let cr : (t : ℕ) → Strategy (W t).game → Prop := fun t τ =>
    (∀ k ω j, Assumptions.TableCRZAt execArith hs512
      (planZHJP ((W t).ms.get k).dj ((W t).ms.get k).y₀ (((W t).memberLaw τ k).draw ω) (reg ((W t).memberSim τ k))) j
      (zcStrat execArith hs512 (planZHJP ((W t).ms.get k).dj ((W t).ms.get k).y₀) (cont ((W t).memberSim τ k) ω))
      ((W t).ms.get k).qF ((W t).ms.get k).qS) ∧
    ∀ k, Assumptions.LinkCRZCAt execArith hs512 ((W t).ms.get k).P (planZHJP ((W t).ms.get k).dj ((W t).ms.get k).y₀)
      (tabZHJP ((W t).ms.get k).dj ((W t).ms.get k).y₀) ((W t).ms.get k).vb (reg ((W t).memberSim τ k))
      (cont ((W t).memberSim τ k)) ((W t).ms.get k).k ((W t).ms.get k).Rw ((W t).ms.get k).M ((W t).ms.get k).t'
  have hs : PeriodGood (fun t => (W t).game) (fun t τ => t < T ∧ cr t τ) 0 T s :=
    periodGood_mono _ T 0 s (fun j τ _ hj hcr => ⟨by omega, hcr⟩)
      (periodGood_of_plays (fun t => (W t).game) cr T 0 s fun t τ hp => And.intro (hT t τ hp) (hL t τ hp))
  have hA := period_union (fun t => (W t).game) (JointAuditV5.record W)
    (fun r => r.accepted ∧ r.drawnWrong) (fun t τ => t < T ∧ cr t τ) (fun t => (W t).eps)
    (fun t τ hτ => (W t).accepted_drawnWrong_le τ hτ.2.1 hτ.2.2) T 0 s hs
  have hB := period_prod (fun t => (W t).game) (JointAuditV5.record W) (fun r => ¬ r.drawnWrong)
    AuditRecV5.pa (fun t τ => t < T ∧ cr t τ)
    (fun t τ hτ => ⟨ENNReal.ofReal ((W t).pa ((W t).wrongSet τ)), ENNReal.ofReal_ne_top, fun _ => rfl,
      (W t).drawMiss_le (hA3 t hτ.1) ((W t).wrongSet τ) τ⟩) T 0 s (ENNReal.ofReal δ) hs
  calc value (fun l => (∀ r ∈ l, r.accepted) ∧ ¬ JointPeriodV5.Within W δ ands bits l) (JointPeriodV5.run W T s)
      ≤ value (fun l => (∃ r ∈ l, r.accepted ∧ r.drawnWrong) ∨
          ((∀ r ∈ l, ¬ r.drawnWrong) ∧ (l.map AuditRecV5.pa).prod < ENNReal.ofReal δ))
          (JointPeriodV5.run W T s) := by
        refine value_mono (fun l hl => ?_) _
        by_cases hd : ∃ r ∈ l, r.drawnWrong
        · obtain ⟨r, hr, hdr⟩ := hd
          exact Or.inl ⟨r, hr, hl.1 r hr, hdr⟩
        · refine Or.inr ⟨fun r hr hdr => hd ⟨r, hr, hdr⟩, ?_⟩
          by_contra hlt
          exact hl.2 (JointPeriodV5.within_of_le ands bits (not_lt.1 hlt))
    _ ≤ value (fun l => ∃ r ∈ l, r.accepted ∧ r.drawnWrong) (JointPeriodV5.run W T s) +
          value (fun l => (∀ r ∈ l, ¬ r.drawnWrong) ∧ (l.map AuditRecV5.pa).prod < ENNReal.ofReal δ)
            (JointPeriodV5.run W T s) :=
        value_or_le _ _ _
    _ ≤ (∑ t ∈ Finset.range T, (W t).eps) + ENNReal.ofReal δ := by
        gcongr
        · exact hA.trans_eq (Finset.sum_congr rfl fun t _ => congrArg (fun t => (W t).eps) (Nat.zero_add t))
        · exact hB
    _ = ENNReal.ofReal δ + ∑ t ∈ Finset.range T, (W t).eps := add_comm _ _

end Flock.SecurityProofs
