module

public import Properties.InferenceOnly
public import Definitions.InferenceOnly.Assumptions.Theta
public import Definitions.Slowdown.Assumptions.Threads

/-! A verdict of compliance carries the thread bound, under premises the check does not see. -/

@[expose] public section

namespace InferenceOnlyCheck.Properties

open Lean (Json)
open Flock.Program (Prog)

/-- When the verdict `out` on `j` has `complies` `true`, every declaration `d` the Program lays out with the registered
information `theta` names has at least `c b_a / (k Λ)` gates per counted multiplication of each reference training
circuit `r` it contains that is `(dd, e)`-threaded with incompressible weights, with no fresh input of `r` in `d`'s
registered information (`ThetaRegistered`), `Λ` being Theorem 1's bound at fresh-input budget `k`, at `c = gamma`,
`α = num / den` and `k = α c` (`Slowdown.Guarantees.ThreadedThrottled`'s conclusion). -/
def Throttled (j out : Json) : Prop :=
  ∀ (i : Instance) (P : Prog) (d : Verity.RegisteredCircuit),
    Parses j i → Flock.Program.decode i.program.toUTF8 = .ok P → LaysOut P d.D → Names i.theta d.θ →
    out.getObjVal? "complies" = .ok (.bool true) →
    let c : ℕ := i.gamma
    let α : ℝ := (i.num : ℝ) / i.den
    let k : ℝ := α * c
    ∀ (b_w b_a : ℝ) (r : Verity.RegisteredCircuit) (T : Slowdown.Threading r) (dd e : ℕ)
      (M : Verity.Containment r.D d.D),
      0 < b_w → 0 < b_a → 0 < dd →
      InferenceOnly.Assumptions.ThetaRegistered r d (Slowdown.PartialContainment.ofContainment M) →
      T.Threaded c k dd e → Slowdown.Assumptions.WeightsIncompressible r T c k b_w b_a →
        b_a * T.M.card ≤ α * T.bound k b_w dd e * d.D.size d.D.computed

/-- **A verdict of compliance throttles every threaded reference training circuit the declaration contains, if its
weights are incompressible and none of its fresh input registered** (the named assumptions `WeightsIncompressible` and
`ThetaRegistered`). On every instance with a verdict, the verdict is `Throttled`: the bound holds for each reference
training circuit `r` the declaration contains that is `(dd, e)`-threaded at fresh-input budget `k`, has incompressible
weights and has no fresh input in the registered information `theta` names, and says nothing of any other. -/
def Throttles : Prop :=
  ∀ (j out : Json), evaluate j = .ok out → Throttled j out

/-- When the verdict `out` on `j` has `complies` `true`, every declaration `d` the Program lays out with the registered
information `theta` names, containing a `(dd, e)`-threaded reference training circuit `r` with incompressible weights
partly, with no fresh input of `r` in `d`'s registered information (`ThetaRegistered`), has `r`'s counted
multiplications at most its dark gates plus `αΛ size(d) / b_a`, at `c = gamma`, `α = num / den` and `k = α c`
(`Slowdown.Guarantees.ThreadedThrottledDark`'s conclusion). -/
def ThrottledDark (j out : Json) : Prop :=
  ∀ (i : Instance) (P : Prog) (d : Verity.RegisteredCircuit),
    Parses j i → Flock.Program.decode i.program.toUTF8 = .ok P → LaysOut P d.D → Names i.theta d.θ →
    out.getObjVal? "complies" = .ok (.bool true) →
    let c : ℕ := i.gamma
    let α : ℝ := (i.num : ℝ) / i.den
    let k : ℝ := α * c
    ∀ (b_w b_a : ℝ) (r : Verity.RegisteredCircuit) (T : Slowdown.Threading r) (dd e : ℕ)
      (M : Slowdown.PartialContainment r d),
      0 < b_w → 0 < b_a → 0 < dd → InferenceOnly.Assumptions.ThetaRegistered r d M → T.Threaded c k dd e →
      Slowdown.Assumptions.WeightsIncompressible r T c k b_w b_a →
        (T.M.card : ℝ) ≤ M.dark.card + α * T.bound k b_w dd e * d.D.size d.D.computed / b_a

/-- **A verdict of compliance puts counted multiplications off the books in dark gates**, for every threaded reference
training circuit the declaration contains partly, if its weights are incompressible and none of its fresh input
registered (the named assumptions `WeightsIncompressible` and `ThetaRegistered`). On every instance with a verdict, the
verdict is `ThrottledDark`: the bound holds for each reference training circuit `r` the declaration contains partly that
is `(dd, e)`-threaded at fresh-input budget `k`, has incompressible weights and has no dark value or fresh input in the
registered information `theta` names, and says nothing of any other. -/
def ThrottlesDark : Prop :=
  ∀ (j out : Json), evaluate j = .ok out → ThrottledDark j out

/-- `Throttled` with the weights charged per parameter slot of a slot map `σ` (`WeightsIncompressibleSlots`):
`Slowdown.Guarantees.ThreadedThrottledSlots`'s conclusion. -/
def ThrottledSlots (j out : Json) : Prop :=
  ∀ (i : Instance) (P : Prog) (d : Verity.RegisteredCircuit),
    Parses j i → Flock.Program.decode i.program.toUTF8 = .ok P → LaysOut P d.D → Names i.theta d.θ →
    out.getObjVal? "complies" = .ok (.bool true) →
    let c : ℕ := i.gamma
    let α : ℝ := (i.num : ℝ) / i.den
    let k : ℝ := α * c
    ∀ (b_w b_a : ℝ) (r : Verity.RegisteredCircuit) (T : Slowdown.Threading r) (σ : T.Slots) (dd e : ℕ)
      (M : Verity.Containment r.D d.D),
      0 < b_w → 0 < b_a → 0 < dd →
      InferenceOnly.Assumptions.ThetaRegistered r d (Slowdown.PartialContainment.ofContainment M) →
      T.Threaded c k dd e → Slowdown.Assumptions.WeightsIncompressibleSlots r T σ c k b_w b_a →
        b_a * T.M.card ≤ α * T.bound k b_w dd e * d.D.size d.D.computed

/-- **`Throttles` per parameter slot**: a verdict of compliance throttles every threaded reference training circuit the
declaration contains, if its weights are incompressible per parameter slot of a slot map `σ` and none of its fresh input
registered (the named assumptions `WeightsIncompressibleSlots` and `ThetaRegistered`). On every instance with a
verdict, the verdict is `ThrottledSlots`. -/
def ThrottlesSlots : Prop :=
  ∀ (j out : Json), evaluate j = .ok out → ThrottledSlots j out

/-- `ThrottledDark` with the weights charged per parameter slot of a slot map `σ` (`WeightsIncompressibleSlots`):
`Slowdown.Guarantees.ThreadedThrottledDarkSlots`'s conclusion. -/
def ThrottledDarkSlots (j out : Json) : Prop :=
  ∀ (i : Instance) (P : Prog) (d : Verity.RegisteredCircuit),
    Parses j i → Flock.Program.decode i.program.toUTF8 = .ok P → LaysOut P d.D → Names i.theta d.θ →
    out.getObjVal? "complies" = .ok (.bool true) →
    let c : ℕ := i.gamma
    let α : ℝ := (i.num : ℝ) / i.den
    let k : ℝ := α * c
    ∀ (b_w b_a : ℝ) (r : Verity.RegisteredCircuit) (T : Slowdown.Threading r) (σ : T.Slots) (dd e : ℕ)
      (M : Slowdown.PartialContainment r d),
      0 < b_w → 0 < b_a → 0 < dd → InferenceOnly.Assumptions.ThetaRegistered r d M → T.Threaded c k dd e →
      Slowdown.Assumptions.WeightsIncompressibleSlots r T σ c k b_w b_a →
        (T.M.card : ℝ) ≤ M.dark.card + α * T.bound k b_w dd e * d.D.size d.D.computed / b_a

/-- **`ThrottlesDark` per parameter slot**: for every threaded reference training circuit the declaration contains
partly, if its weights are incompressible per parameter slot of a slot map `σ` and none of its fresh input registered
(the named assumptions `WeightsIncompressibleSlots` and `ThetaRegistered`). On every instance with a verdict, the
verdict is `ThrottledDarkSlots`. -/
def ThrottlesDarkSlots : Prop :=
  ∀ (j out : Json), evaluate j = .ok out → ThrottledDarkSlots j out

end InferenceOnlyCheck.Properties
