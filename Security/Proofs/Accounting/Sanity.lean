import Proofs.Accounting.Joint

/-!
# The joint guarantee's hypotheses hold

An ideal audit, which accepts exactly when its account's charge meets its certificate, has its guarantee with error 0
at every certificate. So `Run.ComputeSound`, `Run.BandwidthSound` and `Run.CapacitySound` are satisfiable together on any
run, and the joint guarantee and its covert residual are not vacuous.
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
