import Proofs.Properties.Nci
import Proofs.Properties.Nci.Agrees
import Properties.NciDesc.Verdict

/-!
# What a verdict of the description-sized check carries, from `DescAgrees`

Each statement of `Properties.NciDesc.Verdict` is its `Properties.Nci` counterpart carried to every expansion by
`DescAgrees` (`onExpansions_of`): `descReports_of` and `descWitnesses_of` from `Reports` and `Witnesses`, and the four
thread bounds from `Throttles` and its forms, which `Proofs.Nci.DescVerdict` (package `security_proofs`) supplies.
`DescReports` and `DescWitnesses` are the guarantees, from `SecurityProofs.DescAgrees` (`Proofs.Properties.Nci.Agrees`).
-/

namespace NciCheck.Properties.SecurityProofs

open Lean (Json)

/-- **A property of every verdict of `NciCheck.evaluate` holds of every verdict of `NciCheck.Desc.evaluate`, on each
expansion**, given `DescAgrees`. -/
theorem onExpansions_of (h : Properties.DescAgrees) {Q : Json → Json → Prop}
    (hQ : ∀ j out, evaluate j = .ok out → Q j out) {j out : Json} (hev : Desc.evaluate j = .ok out) :
    OnExpansions Q j out :=
  ⟨(h j out hev).1, fun j' hj' => hQ j' out ((h j out hev).2 j' hj')⟩

/-- `Desc.check` returns the verdict `Desc.evaluate` gives. -/
theorem desc_check_ok {j out : Json} (hev : Desc.evaluate j = .ok out) : Desc.check j = out := by
  simp only [Desc.check, hev]

theorem descReports_of (h : Properties.DescAgrees) : Properties.DescReports :=
  fun _ _ hev => onExpansions_of h Reports hev

theorem descWitnesses_of (h : Properties.DescAgrees) : Properties.DescWitnesses :=
  fun _ _ hev => onExpansions_of h Witnesses hev

/-- **The description-sized check's verdict is the definition's.** -/
theorem DescReports : Properties.DescReports :=
  descReports_of DescAgrees

/-- **A verdict of compliance of the description-sized check witnesses the policy.** -/
theorem DescWitnesses : Properties.DescWitnesses :=
  descWitnesses_of DescAgrees

theorem descThrottles_of (h : Properties.DescAgrees) (hT : Properties.Throttles) : Properties.DescThrottles :=
  fun _ _ hev => onExpansions_of h hT hev

theorem descThrottlesDark_of (h : Properties.DescAgrees) (hT : Properties.ThrottlesDark) :
    Properties.DescThrottlesDark :=
  fun _ _ hev => onExpansions_of h hT hev

theorem descThrottlesSlots_of (h : Properties.DescAgrees) (hT : Properties.ThrottlesSlots) :
    Properties.DescThrottlesSlots :=
  fun _ _ hev => onExpansions_of h hT hev

theorem descThrottlesDarkSlots_of (h : Properties.DescAgrees) (hT : Properties.ThrottlesDarkSlots) :
    Properties.DescThrottlesDarkSlots :=
  fun _ _ hev => onExpansions_of h hT hev

end NciCheck.Properties.SecurityProofs
