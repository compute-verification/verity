import Properties.Nci
import Definitions.Nci.Assumptions.Baseline
import Definitions.Nci.Assumptions.Threads

/-!
# A verdict of compliance carries the thread bound

`Witnesses` (`Properties.Nci`) says that a verdict of `NciCheck.evaluate` whose `complies` is `true` means every
declaration the Program lays out, with the baseline `theta` names, complies with `Nci.Complies Γ (num / den)`. The thread
bounds, `Nci.Guarantees.ThreadedThrottled`, `ThreadedThrottledDark` and their `Slots` forms
(`Definitions/Nci/Guarantees/Threads.lean`), bound a complying declaration that contains a reference circuit. The statements
here join the two, so that the check's verdict carries the bound: on every instance with a verdict whose `complies` is
`true`, every such declaration `d` meets the bound's conclusion at `Γ = gamma` and `α = num / den`, for every reference
circuit `c` it contains, under the bound's premises.

* `Throttles`: `ThreadedThrottled`'s conclusion, for every `c` that `d` contains, `(dd, e)`-threaded at read cap `αΓ`
  with its weights incompressible (`WeightsIncompressible`), with no novel input of `c` in the baseline `theta` names
  (`BaselineRegistered`), which the check does not see.
* `ThrottlesDark`: `ThreadedThrottledDark`'s, for every `c` that `d` contains partly (`PartialContainment`), with no
  dark value or novel input of `c` in the baseline `theta` names (`BaselineRegistered`), which the check does not see.
* `ThrottlesSlots`, `ThrottlesDarkSlots`: the same with the weights charged per parameter slot
  (`WeightsIncompressibleSlots`).

The bounds' `0 ≤ α` is no premise here: `num / den` of naturals is nonnegative. Each statement is proved as
`NciCheck.Properties.SecurityProofs.X : NciCheck.Properties.X` in `Proofs.Nci.Verdict` (package `security_proofs`), from
`Witnesses` and the bound's proof: a property whose proof needs `security_proofs` is stated here and proved there.
-/

namespace NciCheck.Properties

open Lean (Json)
open Flock.Program (Prog)

/-- When the verdict `out` on `j` has `complies` `true`, every declaration `d` the Program lays out with the baseline
`theta` names has at least `b_a / (αΛ)` gates per counted multiplication of each reference circuit `c` it contains that is
`(dd, e)`-threaded with incompressible weights, with its baseline registered (`BaselineRegistered`), `Λ` being Theorem 1's bound at read cap `αΓ`, at `Γ = gamma` and
`α = num / den` (`Nci.Guarantees.ThreadedThrottled`'s conclusion). -/
def Throttled (j out : Json) : Prop :=
  ∀ (i : Instance) (P : Prog) (d : Verity.Declaration),
    Parses j i → Flock.Program.decode i.program.toUTF8 = .ok P → LaysOut P d.D → Names i.theta d.θ →
    out.getObjVal? "complies" = .ok (.bool true) →
    let α : ℝ := (i.num : ℝ) / i.den
    ∀ (b_w b_a : ℝ) (c : Verity.Declaration) (T : Nci.Threading c) (dd e : ℕ) (M : Verity.Containment c.D d.D),
      0 < b_w → 0 < b_a → 0 < dd → Nci.Assumptions.BaselineRegistered c d (Nci.PartialContainment.ofContainment M) →
      T.Threaded i.gamma (α * i.gamma) dd e →
      Nci.Assumptions.WeightsIncompressible c T i.gamma (α * i.gamma) b_w b_a →
        b_a * T.M.card ≤ α * T.bound (α * i.gamma) b_w dd e * d.D.size d.D.computed

/-- **A verdict of compliance throttles every threaded reference circuit the declaration contains, if its weights are
incompressible and its baseline registered** (the named assumptions `WeightsIncompressible` and `BaselineRegistered`).
On every instance with a verdict, the verdict is `Throttled`: the bound holds for each reference circuit `c` the
declaration contains that is `(dd, e)`-threaded at read cap `αΓ`, has incompressible weights and has no novel input in
the baseline `theta` names, and says nothing of any other. -/
def Throttles : Prop :=
  ∀ (j out : Json), evaluate j = .ok out → Throttled j out

/-- When the verdict `out` on `j` has `complies` `true`, every declaration `d` the Program lays out with the baseline
`theta` names, containing a `(dd, e)`-threaded reference circuit `c` with incompressible weights partly, with its
baseline registered (`BaselineRegistered`), has `c`'s counted multiplications at most its dark gates plus
`αΛ size(d) / b_a`, at `Γ = gamma` and `α = num / den` (`Nci.Guarantees.ThreadedThrottledDark`'s conclusion). -/
def ThrottledDark (j out : Json) : Prop :=
  ∀ (i : Instance) (P : Prog) (d : Verity.Declaration),
    Parses j i → Flock.Program.decode i.program.toUTF8 = .ok P → LaysOut P d.D → Names i.theta d.θ →
    out.getObjVal? "complies" = .ok (.bool true) →
    let α : ℝ := (i.num : ℝ) / i.den
    ∀ (b_w b_a : ℝ) (c : Verity.Declaration) (T : Nci.Threading c) (dd e : ℕ) (M : Nci.PartialContainment c d),
      0 < b_w → 0 < b_a → 0 < dd → Nci.Assumptions.BaselineRegistered c d M → T.Threaded i.gamma (α * i.gamma) dd e →
      Nci.Assumptions.WeightsIncompressible c T i.gamma (α * i.gamma) b_w b_a →
        (T.M.card : ℝ) ≤ M.dark.card + α * T.bound (α * i.gamma) b_w dd e * d.D.size d.D.computed / b_a

/-- **A verdict of compliance puts counted multiplications off the books in dark gates**, for every threaded reference
circuit the declaration contains partly, if its weights are incompressible and its baseline registered (the named
assumptions `WeightsIncompressible` and `BaselineRegistered`). On every instance with a verdict, the verdict is
`ThrottledDark`: the bound holds for each reference circuit `c` the declaration contains partly that is
`(dd, e)`-threaded at read cap `αΓ`, has incompressible weights and has no dark value or novel input in the baseline
`theta` names, and says nothing of any other. -/
def ThrottlesDark : Prop :=
  ∀ (j out : Json), evaluate j = .ok out → ThrottledDark j out

/-- `Throttled` with the weights charged per parameter slot of a slot map `σ` (`WeightsIncompressibleSlots`):
`Nci.Guarantees.ThreadedThrottledSlots`'s conclusion. -/
def ThrottledSlots (j out : Json) : Prop :=
  ∀ (i : Instance) (P : Prog) (d : Verity.Declaration),
    Parses j i → Flock.Program.decode i.program.toUTF8 = .ok P → LaysOut P d.D → Names i.theta d.θ →
    out.getObjVal? "complies" = .ok (.bool true) →
    let α : ℝ := (i.num : ℝ) / i.den
    ∀ (b_w b_a : ℝ) (c : Verity.Declaration) (T : Nci.Threading c) (σ : T.Slots) (dd e : ℕ)
      (M : Verity.Containment c.D d.D),
      0 < b_w → 0 < b_a → 0 < dd → Nci.Assumptions.BaselineRegistered c d (Nci.PartialContainment.ofContainment M) →
      T.Threaded i.gamma (α * i.gamma) dd e →
      Nci.Assumptions.WeightsIncompressibleSlots c T σ i.gamma (α * i.gamma) b_w b_a →
        b_a * T.M.card ≤ α * T.bound (α * i.gamma) b_w dd e * d.D.size d.D.computed

/-- **`Throttles` per parameter slot**: a verdict of compliance throttles every threaded reference circuit the
declaration contains, if its weights are incompressible per parameter slot of a slot map `σ` and its baseline
registered (the named assumptions `WeightsIncompressibleSlots` and `BaselineRegistered`). On every instance with a
verdict, the verdict is `ThrottledSlots`. -/
def ThrottlesSlots : Prop :=
  ∀ (j out : Json), evaluate j = .ok out → ThrottledSlots j out

/-- `ThrottledDark` with the weights charged per parameter slot of a slot map `σ` (`WeightsIncompressibleSlots`):
`Nci.Guarantees.ThreadedThrottledDarkSlots`'s conclusion. -/
def ThrottledDarkSlots (j out : Json) : Prop :=
  ∀ (i : Instance) (P : Prog) (d : Verity.Declaration),
    Parses j i → Flock.Program.decode i.program.toUTF8 = .ok P → LaysOut P d.D → Names i.theta d.θ →
    out.getObjVal? "complies" = .ok (.bool true) →
    let α : ℝ := (i.num : ℝ) / i.den
    ∀ (b_w b_a : ℝ) (c : Verity.Declaration) (T : Nci.Threading c) (σ : T.Slots) (dd e : ℕ)
      (M : Nci.PartialContainment c d),
      0 < b_w → 0 < b_a → 0 < dd → Nci.Assumptions.BaselineRegistered c d M → T.Threaded i.gamma (α * i.gamma) dd e →
      Nci.Assumptions.WeightsIncompressibleSlots c T σ i.gamma (α * i.gamma) b_w b_a →
        (T.M.card : ℝ) ≤ M.dark.card + α * T.bound (α * i.gamma) b_w dd e * d.D.size d.D.computed / b_a

/-- **`ThrottlesDark` per parameter slot**: for every threaded reference circuit the declaration contains partly, if its
weights are incompressible per parameter slot of a slot map `σ` and its baseline registered (the named assumptions
`WeightsIncompressibleSlots` and `BaselineRegistered`). On every instance with a verdict, the verdict is
`ThrottledDarkSlots`. -/
def ThrottlesDarkSlots : Prop :=
  ∀ (j out : Json), evaluate j = .ok out → ThrottledDarkSlots j out

end NciCheck.Properties
