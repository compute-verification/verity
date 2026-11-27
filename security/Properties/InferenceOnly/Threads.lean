module

public import Properties.InferenceOnly
public import Definitions.InferenceOnly.Assumptions.Theta
public import Definitions.Slowdown.Assumptions.Threads

/-!
# A verdict of compliance carries the thread bound

`Witnesses` (`Properties.InferenceOnly`) says that a verdict of `InferenceOnlyCheck.evaluate` whose `complies` is `true`
means every declaration the Program lays out, with the registered information `theta` names, complies with the
inference-only policy at its rate, `InferenceOnly.CompliesRate c α`, at the gate budget `c = gamma` and the rate `α =
num / den`. The thread bounds, `Slowdown.Guarantees.ThreadedThrottledSlots` and `ThreadedThrottledDarkSlots`
(`Definitions/Slowdown/Guarantees/Threads.lean`), bound a complying declaration that contains a reference training
circuit. The statements here join the two, so that the check's verdict carries the bound: on every instance with a
verdict whose `complies` is `true`, every such declaration `d` meets the bound's conclusion at `c` and the fresh-input
budget `k = α c`, for every reference training circuit `r` it contains, under the bound's premises.

* `ThrottlesSlots`: `ThreadedThrottledSlots`' conclusion, for every `r` that `d` contains, `(dd, e)`-threaded at
  fresh-input budget `k` with its weights incompressible per parameter slot (`WeightsIncompressibleSlots`), with no
  fresh input of `r` in the registered information `theta` names (`ThetaRegistered`), which the check does not see.
* `ThrottlesDarkSlots`: `ThreadedThrottledDarkSlots`', for every `r` that `d` contains partly (`PartialContainment`),
  with no dark value or fresh input of `r` in the registered information `theta` names (`ThetaRegistered`), which the
  check does not see.

`k` need not be a natural here, since the check's rate is any `num / den`: the bounds are stated at the rate (proved as
`Slowdown.SecurityProofs.ThreadedThrottledSlots_rate` and `ThreadedThrottledDarkSlots_rate`), and their `0 ≤ α` is no
premise: `num / den` of naturals is nonnegative. Each statement is proved as
`InferenceOnlyCheck.Properties.SecurityProofs.X : InferenceOnlyCheck.Properties.X` in `Proofs.InferenceOnly.Verdict`
(package `security_proofs`), from `Witnesses` and the bound's proof: a property whose proof needs `security_proofs` is
stated here and proved there.
-/

@[expose] public section

namespace InferenceOnlyVerifier.Properties

open Lean (Json)
open Flock.Program (Prog)

/-- When the verdict `out` on `j` has `complies` `true`, every declaration `d` the Program lays out with the registered
information `theta` names has at least `c b_a / (k Λ)` gates per counted multiplication of each reference training
circuit `r` it contains that is `(dd, e)`-threaded with weights incompressible per parameter slot of a slot map `σ`
(`WeightsIncompressibleSlots`), with no fresh input of `r` in `d`'s registered information (`ThetaRegistered`), `Λ`
being Theorem 1's bound at fresh-input budget `k`, at `c = gamma`, `α = num / den` and `k = α c`
(`Slowdown.Guarantees.ThreadedThrottledSlots`' conclusion). -/
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

/-- **A verdict of compliance throttles every threaded reference training circuit the declaration contains, if its
weights are incompressible per parameter slot and none of its fresh input registered** (the named assumptions
`WeightsIncompressibleSlots` and `ThetaRegistered`). On every instance with a verdict, the verdict is `ThrottledSlots`:
the bound holds for each reference training circuit `r` the declaration contains that is `(dd, e)`-threaded at
fresh-input budget `k`, has incompressible weights and has no fresh input in the registered information `theta` names,
and says nothing of any other. -/
def ThrottlesSlots : Prop :=
  ∀ (j out : Json), evaluate j = .ok out → ThrottledSlots j out

/-- When the verdict `out` on `j` has `complies` `true`, every declaration `d` the Program lays out with the registered
information `theta` names, containing a `(dd, e)`-threaded reference training circuit `r` with weights incompressible
per parameter slot of a slot map `σ` (`WeightsIncompressibleSlots`) partly, with no fresh input of `r` in `d`'s
registered information (`ThetaRegistered`), has `r`'s counted multiplications at most its dark gates plus `αΛ size(d) /
b_a`, at `c = gamma`, `α = num / den` and `k = α c` (`Slowdown.Guarantees.ThreadedThrottledDarkSlots`' conclusion). -/
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

/-- **A verdict of compliance puts counted multiplications off the books in dark gates**, for every threaded reference
training circuit the declaration contains partly, if its weights are incompressible per parameter slot and none of its
fresh input registered (the named assumptions `WeightsIncompressibleSlots` and `ThetaRegistered`). On every instance
with a verdict, the verdict is `ThrottledDarkSlots`: the bound holds for each reference training circuit `r` the
declaration contains partly that is `(dd, e)`-threaded at fresh-input budget `k`, has incompressible weights and has no
dark value or fresh input in the registered information `theta` names, and says nothing of any other. -/
def ThrottlesDarkSlots : Prop :=
  ∀ (j out : Json), evaluate j = .ok out → ThrottledDarkSlots j out

end InferenceOnlyVerifier.Properties
