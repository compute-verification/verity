import Proofs.Accounting.Joint
import Proofs.Accounting.Bandwidth.Honest
import Definitions.Accounting.BandwidthAudit

/-!
# The joint guarantee's hypotheses hold

An ideal audit, which accepts exactly when its account's charge meets its certificate, has its guarantee with error 0
at every certificate. So `Run.ComputeSound`, `Run.BandwidthSound` and `Run.CapacitySound` are satisfiable together on any
run, and the joint guarantee and its covert residual are not vacuous.

The bandwidth audit's honest device realizes its game on a device with no on-chip state, no side channel, `N` oracle
queries and `N·b` bits of device memory bandwidth per tick, over a one-tick window (`honestRun_realizesBandwidth`), so
`Accounting.Guarantees.BandwidthOfAudit` is not vacuous either.
-/

namespace Accounting.Run

open Pouw

variable {ι : Type} {d : Device ι} {α ΩW ΩQ ΩS : Type} [Fintype α] (r : Run d α ΩW ΩQ ΩS)

theorem computeSound_ideal [Fintype ΩW] {p : Price} {W : ℝ} (h : ∀ ω, r.accepts .compute ω → W ≤ r.work p ω) :
    r.ComputeSound p W 0 := fun _ _ _ hT =>
  (pr_eq_zero fun x hx => absurd ((h _ hx.1).trans (hT x)) (not_le.2 hx.2)).le

theorem bandwidthSound_ideal [Fintype ΩQ] {Q : ℝ} (h : ∀ ω, r.accepts .bandwidth ω → Q ≤ r.read ω) :
    r.BandwidthSound Q 0 := fun _ _ _ _ hB =>
  (pr_eq_zero fun q hq => absurd ((h _ hq.1).trans (hB q)) (not_le.2 hq.2)).le

theorem capacitySound_ideal [Fintype ΩS] {S : ℝ} (h : ∀ ω, r.accepts .capacity ω → S * (r.win .capacity).len ≤ r.held ω) :
    r.CapacitySound S 0 := fun _ _ _ _ hB =>
  (pr_eq_zero fun s hs => absurd ((h _ hs.1).trans (hB s)) (not_le.2 hs.2)).le

end Accounting.Run

namespace Accounting.Sanity

open Bandwidth.Completeness (honest honest_accepts)

/-- A device with no on-chip state, no device memory and no side channel, `N` ops of each class and `N·b` bits of
device memory bandwidth per tick. -/
def honestDevice (N b : ℕ) : Device Empty := ⟨0, 0, fun _ => N, N * b, Empty.elim⟩

/-- One account reading `N·b` bits of device memory every tick, over one-tick windows, with the bandwidth audit's verdict
the honest device's. -/
def honestRun (N b h s : ℕ) : Run (honestDevice N b) Unit Unit (BandwidthCoins N b h s) Unit where
  acct _ := ()
  win _ := ⟨0, 1⟩
  trace _ _ := ⟨fun _ _ => 0, fun _ => N * b, fun _ i => i.elim, fun _ => 0, fun _ => 0⟩
  fits _ _ := ⟨fun _ => by simp [honestDevice], by simp [honestDevice], fun i => i.elim, by simp [honestDevice],
    by simp [honestDevice]⟩
  accepts _ ω := Bandwidth.Accepts (honest N b h) ω.2.1.1 ω.2.1.2.1 ω.2.1.2.2

theorem honestRun_realizesBandwidth (N b h s : ℕ) : Nonempty (RealizesBandwidth 0 (honestRun N b h s)) :=
  ⟨{ dev := fun _ => { honest N b h with
        bounded := fun st w => by
          obtain ⟨h1, h2, h3⟩ := (honest N b h).bounded st w
          exact ⟨by simpa using h1, h2, by simpa [honestRun, honestDevice] using h3⟩ }
     accepts_iff := fun _ _ _ => Iff.rfl
     ingress_le := fun _ y _ => by
      refine le_of_eq ((honest_accepts (s := s) y.1 y.2.1 y.2.2).2.trans ?_)
      simp [Run.read, Trace.read, Window.ticks, honestRun] }⟩

end Accounting.Sanity
