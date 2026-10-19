import Properties.Nci.Threads
import Properties.Proofs.Nci
import Proofs.Nci.Threads

/-!
# A verdict of compliance carries the thread bound, proved

Each statement of `Properties.Nci.Threads`, from the verdict's compliance (`NciCheck.Properties.SecurityProofs.Witnesses`)
and the thread bound's proof (`Nci.SecurityProofs.ThreadedThrottled`, `ThreadedThrottledDark` and their `Slots` forms, in
`Proofs.Nci.Threads`), at `α = num / den`.
-/

namespace NciCheck.Properties.SecurityProofs

/-- `num / den` of naturals is nonnegative. -/
theorem num_div_den_nonneg (i : Instance) : (0 : ℝ) ≤ (i.num : ℝ) / i.den :=
  div_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _)

/-- **A verdict of compliance throttles every threaded reference circuit the declaration contains, if its weights are
incompressible** (`WeightsIncompressible`). -/
theorem Throttles : Properties.Throttles :=
  fun j out hev i P d hpar hdec hL hθ hc b_w b_a c T dd e hbw hba hdd hcon hT hW =>
    Nci.SecurityProofs.ThreadedThrottled i.gamma ((i.num : ℝ) / i.den) b_w b_a c d T dd e (num_div_den_nonneg i)
      hbw hba hdd hcon (Witnesses j out hev i P d hpar hdec hL hθ hc) hT hW

/-- **A verdict of compliance puts counted multiplications off the books in dark gates**, for every threaded reference
circuit the declaration contains partly, if its weights are incompressible (`WeightsIncompressible`). -/
theorem ThrottlesDark : Properties.ThrottlesDark :=
  fun j out hev i P d hpar hdec hL hθ hc b_w b_a c T dd e M hbw hba hdd hT hW =>
    Nci.SecurityProofs.ThreadedThrottledDark i.gamma ((i.num : ℝ) / i.den) b_w b_a c d T dd e M
      (num_div_den_nonneg i) hbw hba hdd (Witnesses j out hev i P d hpar hdec hL hθ hc) hT hW

/-- **`Throttles` per parameter slot**, for threaded reference circuits under `WeightsIncompressibleSlots`. -/
theorem ThrottlesSlots : Properties.ThrottlesSlots :=
  fun j out hev i P d hpar hdec hL hθ hc b_w b_a c T σ dd e hbw hba hdd hcon hT hW =>
    Nci.SecurityProofs.ThreadedThrottledSlots i.gamma ((i.num : ℝ) / i.den) b_w b_a c d T σ dd e
      (num_div_den_nonneg i) hbw hba hdd hcon (Witnesses j out hev i P d hpar hdec hL hθ hc) hT hW

/-- **`ThrottlesDark` per parameter slot**, for threaded reference circuits under `WeightsIncompressibleSlots`. -/
theorem ThrottlesDarkSlots : Properties.ThrottlesDarkSlots :=
  fun j out hev i P d hpar hdec hL hθ hc b_w b_a c T σ dd e M hbw hba hdd hT hW =>
    Nci.SecurityProofs.ThreadedThrottledDarkSlots i.gamma ((i.num : ℝ) / i.den) b_w b_a c d T σ dd e M
      (num_div_den_nonneg i) hbw hba hdd (Witnesses j out hev i P d hpar hdec hL hθ hc) hT hW

end NciCheck.Properties.SecurityProofs
