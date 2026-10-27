import Definitions.Accounting.Actual
import Definitions.Accounting.Additive

/-!
# Compute additivity's named assumptions (trusted)

What compute additivity on the actual circuit (`Accounting.Guarantees.GatesAdditive`) assumes beyond the machine model
𝓜, each a named `Prop` over the actual circuit `A` of one run, taken as a hypothesis by the guarantees that use it:

* `PaidBy` is the link 𝓜 lacks between its runs and `A`'s gates. 𝓜 counts a device's instructions and the units of
  each resource they use; `A` counts gates; nothing relates the two. `PaidBy A X π R κ iss w` says the work `A` places
  at the device's leaves `X` during `w`, at the pricing `π`, is paid for by the units of the resources `R` that the
  device's instructions `iss` use then, at `κ_r` work a unit of resource `r`. It holds when each instruction `i`
  computes at most `c(i)` work of `A`, at the tick it issues (𝓜 has no latency), and `c(i) ≤ Σ_{r ∈ R} κ_r u_r(i)`:
  `κ` covers every instruction's gates by its resources, as a tensor-core instruction's gates by its tensor units.
* `ShareNoGate` is what remains of compute additivity once capacity is a theorem: the works subprotocols certify are
  done by gates of their own, disjoint parts of `A`.
-/

namespace Accounting.Assumptions

open Finset

/-- **The device's instructions pay for its gates**: the work `A` places at the leaves `X` during `w`, at the pricing
`π`, is at most what the units of the resources `R` its instructions `iss` use during `w` pay for, at `κ_r` work a
unit of resource `r`. -/
def PaidBy {L : Type*} (A : Placed L) (X : Finset L) (π : Fin A.N → ℕ) {ι : Type} {m : Verity.Accel.Machine ι}
    (R : Finset m.Res) (κ : m.Res → ℚ≥0) (iss : List (List m.Instr)) (w : Window) : Prop :=
  (A.work π (A.region X ↑w.ticks) : ℚ≥0) ≤ ∑ t ∈ w.ticks, ∑ r ∈ R, κ r * Verity.Accel.Machine.usage iss t r

/-- **The works `W` share no gate of `A[X, I]`**: pairwise disjoint sets of computed gates of `A` placed in `X` during
`I`, the `j`th of work at least `W j` at the pricing `π`.

It is compute additivity's conjecture W (`note:console/20261012T0645Z-draft-actual-circuit-model` §5, "compute adds by
a direct-sum conjecture") without its capacity, which `Accounting.Guarantees.GatesAdditive` proves from 𝓜: each `W j`
is a lower bound a subprotocol's own guarantee or conjecture gives on the work its certificate needs (PoUW's work,
`WorkHardness` for a completion), and `ShareNoGate` adds that those works are done by different gates, so no gate counts
for two of them. It reads no hardware parameter. At a pricing of `0` or `1` a gate (a gate's AND count) it is
equivalent to `Σ_j W j ≤ π(A[X, I])`: cut any order of the region's priced gates into consecutive parts. It is a
direct-sum statement, and for circuit size these fail in general (computing several functions together can take fewer
gates than computing each apart), so it is a conjecture about the subprotocols' functions, not a fact of circuits. -/
def ShareNoGate {L : Type*} (A : Placed L) (X : Finset L) (I : Set ℕ) (π : Fin A.N → ℕ) {k : ℕ} (W : Fin k → ℕ) :
    Prop :=
  ∃ S : Fin k → Finset (Fin A.N), (∀ j, S j ⊆ A.region X I) ∧ (∀ i j, i ≠ j → Disjoint (S i) (S j)) ∧
    ∀ j, W j ≤ A.work π (S j)

end Accounting.Assumptions
