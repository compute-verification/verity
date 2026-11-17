module

public import Definitions.Pouw.Assumptions.Deadline
public import Definitions.Pouw.PearlC.Deadline
public import Proofs.Pouw.PearlC.Records
public import Definitions.Pouw.PearlC.Deadline
public import Definitions.Pouw.Game.Deadline
public import Definitions.Pouw.PearlC.Game

/-! The deadline guarantees' premises are satisfiable; the run is constructed, so the witnesses show the premises
consistent and not that a physical run's record meets `HonestClock`, a named assumption. -/

@[expose] public section

namespace Pouw.PearlC

variable {Q R S : Type}

def cubeWorkload (sh : Shape) : Workload := ⟨⟨1, fun _ => sh, fun _ => 0⟩, fun _ _ _ => 0⟩

theorem cube_inDomain (P : Protocol Q R S) (sh : Shape) (hm : 0 < sh.m) (hn : 64 ≤ sh.n ∧ 64 ∣ sh.n)
    (hk : 0 < sh.k ∧ 128 ∣ sh.k ∧ sh.k ≤ 2 ^ 16) :
    (cubeWorkload sh).InDomain P (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh) := by
  refine ⟨by norm_num [cubeWorkload], fun u _ => ?_, fun _ _ => rfl⟩
  have hG : (devSm120v1 Prices.sm120Loop).G = 4 := rfl
  show 0 < sh.m ∧ (64 ≤ sh.n ∧ 64 ∣ sh.n) ∧ 0 < sh.k ∧ 128 ∣ sh.k ∧
    ((devSm120v1 Prices.sm120Loop).G = 0 ∨ 32 * (devSm120v1 Prices.sm120Loop).G ∣ sh.k) ∧ sh.k ≤ 2 ^ 16
  rw [hG]
  exact ⟨hm, hn, hk.1, hk.2.1, Or.inr (by simpa using hk.2.1), hk.2.2⟩

theorem cube8192_inDomain (P : Protocol Q R S) :
    (cubeWorkload sh8192).InDomain P (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192) :=
  cube_inDomain P sh8192 (by decide) ⟨by decide, by decide⟩ ⟨by decide, by decide, by decide⟩

theorem cube16384_inDomain (P : Protocol Q R S) :
    (cubeWorkload sh16384).InDomain P (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh16384) :=
  cube_inDomain P sh16384 (by decide) ⟨by decide, by decide⟩ ⟨by decide, by decide, by decide⟩

def ownRun (CM : CostModel Q R S) (U : Workload) (A : Adversary CM) : TimedRun Q R S where
  ts := 0
  tr := fun _ _ => 1
  spent := fun H s t => t * CM.cost A.online H U A.act (A.pre.eval H) s

theorem ownRun_spentIn (CM : CostModel Q R S) (U : Workload) (A : Adversary CM) (H : Q → R) (s : S) :
    (ownRun CM U A).spentIn H s (ownRun CM U A).ts 1 = CM.cost A.online H U A.act (A.pre.eval H) s := by
  simp [ownRun, TimedRun.spentIn]

theorem ownRun_onTime (CM : CostModel Q R S) (U : Workload) (A : Adversary CM) (H : Q → R) (s : S) :
    (ownRun CM U A).tr H s ≤ 1 :=
  le_refl _

theorem honestClock_ownRun (CM : CostModel Q R S) (U : Workload) (A : Adversary CM) :
    Assumptions.HonestClock (CM.adversaries U) A (ownRun CM U A) 1 :=
  ⟨A, rfl, fun H s => ⟨fun _ => ⟨rfl, fun _ hm => hm⟩, (ownRun_spentIn CM U A H s).ge, le_refl _⟩⟩

theorem boundedBy_ownRun (CM : CostModel Q R S) (U : Workload) (A : Adversary CM) {T : ℝ} {q : ℕ}
    (h : A.Bounded U T q) : A.BoundedBy U (ownRun CM U A) 1 T q :=
  fun H s => ⟨(ownRun_spentIn CM U A H s).le.trans (h H s).1, (h H s).2⟩

theorem byDeadline_premises_8192 (P : Protocol Q R S) (CM : CostModel Q R S) (A : Adversary CM) {T : ℝ} {q : ℕ}
    (h : A.Bounded (cubeWorkload sh8192) T q) :
    (cubeWorkload sh8192).InDomain P (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192) ∧
      ∃ (run : TimedRun Q R S) (td : ℝ), Assumptions.HonestClock (CM.adversaries (cubeWorkload sh8192)) A run td ∧
        A.BoundedBy (cubeWorkload sh8192) run td T q ∧ ∀ H s, run.tr H s ≤ td :=
  ⟨cube8192_inDomain P, ownRun CM _ A, 1, honestClock_ownRun CM _ A, boundedBy_ownRun CM _ A h,
    ownRun_onTime CM _ A⟩

end Pouw.PearlC
