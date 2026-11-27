module

public import Properties.InferenceOnly.Threads
public import Proofs.Properties.InferenceOnly
public import Proofs.Slowdown.Threads

/-!
# A verdict of compliance carries the thread bound, proved

Each statement of `Properties.InferenceOnly.Threads`, from the verdict's compliance at its rate
(`InferenceOnlyCheck.Properties.SecurityProofs.Witnesses`) and the thread bound's proof at a rate
(`Slowdown.SecurityProofs.ThreadedThrottledSlots_rate` and `ThreadedThrottledDarkSlots_rate`, in
`Proofs.Slowdown.Threads`), at `c = gamma` and `α = num / den`, so `k = α c`.
-/

@[expose] public section

namespace InferenceOnlyVerifier.Properties.SecurityProofs

/-- `num / den` of naturals is nonnegative. -/
theorem num_div_den_nonneg (i : Instance) : (0 : ℝ) ≤ (i.num : ℝ) / i.den :=
  div_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _)

/-- **A verdict of compliance throttles every threaded reference training algorithm the declaration contains, if its
weights are incompressible per parameter slot and none of its fresh input registered** (`WeightsIncompressibleSlots`,
`ThetaRegistered`). -/
theorem ThrottlesSlots : Properties.ThrottlesSlots :=
  fun j out hev i P d hpar hdec hL hθ hc b_w b_a r T σ dd e M hbw hba hdd hB hT hW =>
    Slowdown.SecurityProofs.ThreadedThrottledSlots_rate i.gamma ((i.num : ℝ) / i.den) b_w b_a r d T σ dd e M
      (num_div_den_nonneg i) hbw hba hdd (Witnesses j out hev i P d hpar hdec hL hθ hc).1 hB hT hW

/-- **A verdict of compliance puts counted multiplications off the books in dark gates**, for every threaded reference
training algorithm the declaration contains partly, if its weights are incompressible per parameter slot and none of its
fresh input registered (`WeightsIncompressibleSlots`, `ThetaRegistered`). -/
theorem ThrottlesDarkSlots : Properties.ThrottlesDarkSlots :=
  fun j out hev i P d hpar hdec hL hθ hc b_w b_a r T σ dd e M hbw hba hdd hB hT hW =>
    Slowdown.SecurityProofs.ThreadedThrottledDarkSlots_rate i.gamma ((i.num : ℝ) / i.den) b_w b_a r d T σ dd e M
      (num_div_den_nonneg i) hbw hba hdd (Witnesses j out hev i P d hpar hdec hL hθ hc).1 hB hT hW

end InferenceOnlyVerifier.Properties.SecurityProofs
