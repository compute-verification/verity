import Definitions.Pouw.Assumptions.PearlC.Deadline
import Definitions.Pouw.PearlC.Device
import Definitions.Pouw.PearlC.DevicePricesLoop

/-!
# The deadline guarantees' premises are satisfiable (non-vacuity witnesses)

`Definitions/Pouw/Guarantees/Deadline.lean`'s sm_120 guarantees (`PearlCSm120Rev1Cap1000ByDeadline_8192` and the others)
take, besides TT_OUT, a workload in `pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh`, the honest clock
`HonestClock CM U A run td` and the budget `A.BoundedBy U run td T q`. These witnesses meet all three, together with
delivery by the deadline:

* `cube_inDomain`: the one-unit workload of shape `sh` is in the sm_120 v1 domain at `sh` (`cube8192_inDomain`,
  `cube16384_inDomain`);
* `ownRun`, `honestClock_ownRun`, `boundedBy_ownRun`: every adversary of every cost model, bounded by `T` and `q`, has a
  timed run that delivers at the deadline, meets the honest clock with `A' = A`, and spends at most `T` by it.

They are generic in the cost model, so they cover the honest W1 program at `W1.w1Priced (W1Device.sm120Loop sha fetch)
salt`, the guarantees' cost model at W1. The run is constructed: these show the premises consistent, not that a
physical run's record meets `HonestClock`, which stays the named assumption.
-/

namespace Pouw.PearlC

variable {Q R S : Type}

/-- One unit of shape `sh`, weight id `0`, every weight word `0`. -/
def cubeWorkload (sh : Shape) : Workload := ⟨⟨1, fun _ => sh, fun _ => 0⟩, fun _ _ _ => 0⟩

/-- A one-unit workload is in the sm_120 v1 domain at its shape when the shape is (`devSm120v1` promotes every 4 atoms,
so its group condition is `128 ∣ k`). -/
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

/-- The adversary's own timed run: the salt at `0`, the transcript delivered at `1`, and the online program's cost spent
at a constant rate over `(0, 1]`. -/
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

/-- The honest clock holds on the adversary's own run at the deadline `1`, with `A' = A`. -/
theorem honestClock_ownRun (CM : CostModel Q R S) (U : Workload) (A : Adversary CM) :
    Assumptions.HonestClock CM U A (ownRun CM U A) 1 :=
  ⟨A, rfl, fun H s => ⟨fun _ => rfl, (ownRun_spentIn CM U A H s).ge, le_refl _⟩⟩

/-- An adversary bounded by `T` and `q` is bounded by them on its own run by the deadline `1`. -/
theorem boundedBy_ownRun (CM : CostModel Q R S) (U : Workload) (A : Adversary CM) {T : ℝ} {q : ℕ}
    (h : A.Bounded U T q) : A.BoundedBy U (ownRun CM U A) 1 T q :=
  fun H s => ⟨(ownRun_spentIn CM U A H s).le.trans (h H s).1, (h H s).2⟩

/-- **The deadline guarantees' premises apart from TT_OUT, met together** at 8192³: the one-unit workload is in the
domain, and every bounded adversary has a run, on time on every coin, that meets the honest clock and the budget. -/
theorem byDeadline_premises_8192 (P : Protocol Q R S) (CM : CostModel Q R S) (A : Adversary CM) {T : ℝ} {q : ℕ}
    (h : A.Bounded (cubeWorkload sh8192) T q) :
    (cubeWorkload sh8192).InDomain P (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192) ∧
      ∃ (run : TimedRun Q R S) (td : ℝ), Assumptions.HonestClock CM (cubeWorkload sh8192) A run td ∧
        A.BoundedBy (cubeWorkload sh8192) run td T q ∧ ∀ H s, run.tr H s ≤ td :=
  ⟨cube8192_inDomain P, ownRun CM _ A, 1, honestClock_ownRun CM _ A, boundedBy_ownRun CM _ A h,
    ownRun_onTime CM _ A⟩

end Pouw.PearlC
