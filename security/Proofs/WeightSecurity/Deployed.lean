module

public import Proofs.Flock.EndToEnd.Joint.V5.Bound
public import Proofs.WeightSecurity.Served
public import Proofs.WeightSecurity.Stop

/-!
# Weight security at the deployed audit

`TheftServed` at the deployed one-stage audit. Each window, the datacenter releases what leaves on the egress slots, at
its timing bucket and on its advice, and then the window's audit runs (`JointAuditV5`: every member's `SampledKSLiveV5`
session on one live sample). The probability that it leaves `M` incorrect proof blocks undetected is at most
$$\sum_t \sum_m (\varepsilon_{ks,t,m} + \varepsilon_{bind,t,m}) + x^M$$
by `period_undetected`. A window is accepted with a sampled incorrect proof block with probability at most its members'
knowledge and binding terms (`JointAuditV5.accepted_drawnWrong_le`). Its sample misses its `w` incorrect proof blocks
with probability at most `P_a ≤ x^w` (`JointAuditV5.drawMiss_le`, with `x` the sampling design's per-block miss: a
premise on its `P_a`, which `pa_le_pow_of_fac` discharges at every registered sampling design whose strata each have
`(n − k)/n ≤ x`). The named assumptions are `JointHorizonV5`'s, at every window, and weight security's own.
-/

@[expose] public section

namespace WeightSecurity.SecurityProofs

open Flock.Guarantees FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec
  FlockSoundness.Refine FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden WeightSecurity.Assumptions
open scoped ENNReal

variable {O V Tick Adv : Type}

/-- A window's release: what leaves on the egress slots, its timing bucket and the advice it ran on. -/
abbrev Release (O V Tick Adv : Type) := (O → V) × Tick × Adv

/-- **A window served at the deployed audit**: the datacenter commits the window's release, then the window's audit
runs. -/
noncomputable def servedGame (w : JointAuditV5) (Rel : Type) :
    Game (Rel × (w.Reg × Finset (Fin w.nJ) × List Discharge.Exec.Transcript)) :=
  .send Rel fun m => w.game.map fun o => (m, o)

/-- The audit's strategy that a served window's strategy plays. -/
noncomputable def servedAudit (w : JointAuditV5) {Rel : Type} (τ : Strategy (servedGame w Rel)) : Strategy w.game :=
  Strategy.ofMap (fun o => (τ.1, o)) w.game τ.2

/-- A served window's record: its verdict, whether its sample holds an incorrect proof block, how many it has, and its
release. -/
noncomputable def servedRec (w : JointAuditV5) (τ : Strategy (servedGame w (Release O V Tick Adv)))
    (b : Release O V Tick Adv × (w.Reg × Finset (Fin w.nJ) × List Discharge.Exec.Transcript)) : Served O V Tick Adv :=
  ⟨w.Accepts b.2, ∃ u ∈ b.2.2.1, u ∈ w.wrongSet (servedAudit w τ), (w.wrongSet (servedAudit w τ)).card,
    b.1.1, b.1.2.1, b.1.2.2⟩

/-- The deployment's run over `T` windows at the datacenter's strategy `s`. -/
noncomputable def servedRun (W : ℕ → JointAuditV5) (T : ℕ)
    (s : Strategy (periodG (fun t => servedGame (W t) (Release O V Tick Adv)) 0 T)) :
    Game (List (Served O V Tick Adv)) :=
  periodRun (fun t => servedGame (W t) (Release O V Tick Adv)) (fun t => servedRec (W t)) 0 T s

theorem servedRec_accepted_le (w : JointAuditV5) (τ : Strategy (servedGame w (Release O V Tick Adv)))
    (hCR : ∀ k ω j, Assumptions.TableCRZAt execArith hs512
      (planZHJP (w.ms.get k).dj (w.ms.get k).y₀ ((w.memberLaw (servedAudit w τ) k).draw ω)
        (reg (w.memberSim (servedAudit w τ) k))) j
      (zcStrat execArith hs512 (planZHJP (w.ms.get k).dj (w.ms.get k).y₀) (cont (w.memberSim (servedAudit w τ) k) ω))
      (w.ms.get k).qF (w.ms.get k).qS)
    (hA2 : ∀ k, Assumptions.LinkCRZCAt execArith hs512 (w.ms.get k).P (planZHJP (w.ms.get k).dj (w.ms.get k).y₀)
      (tabZHJP (w.ms.get k).dj (w.ms.get k).y₀) (w.ms.get k).vb (reg (w.memberSim (servedAudit w τ) k))
      (cont (w.memberSim (servedAudit w τ) k)) (w.ms.get k).k (w.ms.get k).Rw (w.ms.get k).M (w.ms.get k).t') :
    prob (fun b => (servedRec w τ b).accepted ∧ (servedRec w τ b).sampledIncorrect) (servedGame w _) τ ≤ w.eps := by
  show prob (fun b => w.Accepts b.2 ∧ ∃ u ∈ b.2.2.1, u ∈ w.wrongSet (servedAudit w τ))
    (w.game.map fun o => (τ.1, o)) τ.2 ≤ _
  rw [prob_map]
  exact w.accepted_drawnWrong_le (servedAudit w τ) hCR hA2

theorem servedRec_miss_le (w : JointAuditV5) (hA3 : Assumptions.UniformRandomBytes w.Lb w.src) (x : ℝ) (hx : 0 ≤ x)
    (hpa : ∀ B, w.pa B ≤ x ^ B.card) (τ : Strategy (servedGame w (Release O V Tick Adv))) :
    prob (fun b => ¬ (servedRec w τ b).sampledIncorrect) (servedGame w _) τ ≤
      ENNReal.ofReal x ^ (w.wrongSet (servedAudit w τ)).card := by
  show prob (fun b => ¬ ∃ u ∈ b.2.2.1, u ∈ w.wrongSet (servedAudit w τ)) (w.game.map fun o => (τ.1, o)) τ.2 ≤ _
  rw [prob_map, ← ENNReal.ofReal_pow hx]
  exact (w.drawMiss_le hA3 _ (servedAudit w τ)).trans (ENNReal.ofReal_le_ofReal (hpa _))

/-- **Model weight security at the deployed audit.** `TheftServed` with the probability of leaving `M` incorrect proof
blocks undetected bounded by the windows' knowledge and binding terms plus `x^M`, at every datacenter strategy `s θ r`,
under `JointHorizonV5`'s named assumptions at every window and weight security's `HonestUseHard` and `Simulable`. -/
theorem TheftServedDeployed {Ω R Θ Q Adv O V Tick Θ' : Type} [Fintype Ω] [Fintype R] [Fintype Adv] [Inhabited Adv]
    [Fintype O] [DecidableEq O] [Fintype V] [Fintype Tick] [Inhabited Tick]
    (W : ℕ → JointAuditV5) (T L E M : ℕ) (x : ℝ) (hx0 : 0 ≤ x) (hx1 : x ≤ 1)
    (C : Θ → Q × Adv → O → V) (gen : Ω → Θ) (Stolen : Θ → Θ' → Prop) (A : Set (Outsider R (Q × Adv) (O → V) Θ'))
    (p : ℝ) (Ou : Outsider R (Q × Adv) (Option ((O → V) × Tick)) Θ')
    (s : Θ → R → Strategy (periodG (fun t => servedGame (W t) (Release O V Tick Adv)) 0 T))
    (hhon : HonestUseHard C gen Stolen T A p)
    (hsim : Simulable A Ou (servedChannel R Q Adv O V Tick T (E * M + (L + 1) * Fintype.card O)))
    (hA3 : ∀ t < T, Assumptions.UniformRandomBytes (W t).Lb (W t).src)
    (hpa : ∀ t < T, ∀ B, (W t).pa B ≤ x ^ B.card)
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
        (∑ t ∈ Finset.range T, (W t).eps + ENNReal.ofReal (x ^ M)) +
        Verity.Game.avg (fun z : Ω × R => Verity.Game.value
          (fun l => ¬ Served.Local C E Ou L (gen z.1) z.2 l) (servedRun W T (s (gen z.1) z.2))) := by
  classical
  refine TheftServed C gen Stolen A p Ou T L E M _ (fun θ r => servedRun W T (s θ r)) hhon hsim fun θ r => ?_
  let cr : (t : ℕ) → Strategy (servedGame (W t) (Release O V Tick Adv)) → Prop := fun t τ =>
    (∀ k ω j, Assumptions.TableCRZAt execArith hs512
      (planZHJP ((W t).ms.get k).dj ((W t).ms.get k).y₀ (((W t).memberLaw (servedAudit (W t) τ) k).draw ω)
        (reg ((W t).memberSim (servedAudit (W t) τ) k))) j
      (zcStrat execArith hs512 (planZHJP ((W t).ms.get k).dj ((W t).ms.get k).y₀)
        (cont ((W t).memberSim (servedAudit (W t) τ) k) ω))
      ((W t).ms.get k).qF ((W t).ms.get k).qS) ∧
    ∀ k, Assumptions.LinkCRZCAt execArith hs512 ((W t).ms.get k).P
      (planZHJP ((W t).ms.get k).dj ((W t).ms.get k).y₀) (tabZHJP ((W t).ms.get k).dj ((W t).ms.get k).y₀)
      ((W t).ms.get k).vb (reg ((W t).memberSim (servedAudit (W t) τ) k))
      (cont ((W t).memberSim (servedAudit (W t) τ) k)) ((W t).ms.get k).k ((W t).ms.get k).Rw ((W t).ms.get k).M
      ((W t).ms.get k).t'
  have hs : PeriodGood (fun t => servedGame (W t) (Release O V Tick Adv)) (fun t τ => t < T ∧ cr t τ) 0 T (s θ r) :=
    periodGood_mono _ T 0 (s θ r) (fun j τ _ hj hcr => ⟨by omega, hcr⟩)
      (periodGood_of_plays (fun t => servedGame (W t) (Release O V Tick Adv)) cr T 0 (s θ r)
        fun t τ hp => ⟨hT θ r t τ hp, hL θ r t τ hp⟩)
  have h := period_undetected (fun t => servedGame (W t) (Release O V Tick Adv)) (fun t => servedRec (W t))
    (fun t τ => t < T ∧ cr t τ) (fun t => (W t).eps) (ENNReal.ofReal x) (ENNReal.ofReal_le_one.2 hx1)
    (fun t τ hτ => servedRec_accepted_le (W t) τ hτ.2.1 hτ.2.2)
    (fun t τ hτ => ⟨((W t).wrongSet (servedAudit (W t) τ)).card, fun _ => rfl,
      servedRec_miss_le (W t) (hA3 t hτ.1) x hx0 (hpa t hτ.1) τ⟩)
    T 0 (s θ r) M hs
  rw [← ENNReal.ofReal_pow hx0] at h
  simp only [Nat.zero_add] at h
  exact h

end WeightSecurity.SecurityProofs
