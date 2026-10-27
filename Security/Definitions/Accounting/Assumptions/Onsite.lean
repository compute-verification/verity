import Definitions.Accounting.Onsite

/-!
# The onsite bridges no result gives: the wipe's state and the integrity claim's containment (trusted)

Two of the bridges from Layer 2's claims to the onsite class (`Accounting.OnsiteClass`,
note:training-20261027T0134Z-draft-e2e-overhead-proof Step 3), stated as named premises on a period and the map `S`
from the game's runs to their actual circuits. Each is the bridge itself: on every run where its Layer-2 claim holds,
the run's actual circuit meets its hypothesis of `Accounting.Guarantees.Actual.DampenerWindowsCap`, verbatim. Neither
follows from any result on `main`, and each docstring says what is missing. A guarantee that takes one is conditional
on it until it is proved, and proving it removes it.
-/

namespace Accounting.Assumptions

open Verity Finset

/-- **The wipes leave at most `R` bits of state** (bridge `WipeState`). On every run `o` where every wipe `w k` meets
its survivor bound at fraction `f` (`WipeLive.Survives`, `JointHorizonBlocksLive`'s wipe claim, whose error is
`Pous.Guarantees.StreamedWipe`'s), each location `X i`'s state at the start of each window `[kω, (k + 1)ω)` before `t`
is at most `R` bits (`Placed.state` of `(S o).A`). The wipes are those that open the locations' windows, one for each
location and window start.

What is missing:

* `Survives` bounds `kept`, a field of the run: the bits of foreign content beyond the committed carried state that
  the block can read back from what the wipe leaves. That is information: `StreamedWipe`'s event is the prover
  recovering a uniform prior content `z` from what it holds and what left (`Pr[accept ∧ recover z] ≤ …`). No field or
  premise ties `kept` to the actual circuit.
* `Placed.state` is structural: the width of the gates placed in `X i` before `kω` that `X i` reads at `kω` or later.
  A prover that passes the wipe and reads its labels, or any value it held through the wipe, after `kω` carries them as
  state, up to the whole fill (2.6 TB at the deployed wipe), while the information that survives about a uniform prior
  is about `1.075 GB` plus the egress (`Pous.Guarantees.StreamedWipeMeets2600GB`). So the width bound does not follow
  from the information bound for every strategy: as a premise it restricts the strategies the guarantee speaks of,
  which nothing checks (`Accounting.Guarantees.Actual.ThreadedRuntime`'s module docstring: composing the cap with a
  wipe needs its read bound in its determination form).
* What closes it: Layer 3's read bound in its determination form, the state counted by what it determines about the
  novel inputs, against the wipe's mutual-information form (memory-accounting's MEM-98); a per-server `R` is
  #4828's `StreamedWipeMeetsServer5GB` (open, not on `main`). Then the bridge has an error, the wipe's, in
  `δ_bridge`. -/
def WipeState {ι : Type} {d : Device ι} {O L : Type} (S : O → Actual.Run L) {m : ℕ} (w : Fin m → WipeLive d O)
    (f : ℝ) {n : ℕ} (X : Fin n → Finset L) (t ω : ℕ) (R : ℝ) : Prop :=
  ∀ o, (∀ k, (w k).Survives f o) → ∀ i (k : ℕ), k * ω < t → ((S o).A.state (X i) (k * ω) : ℝ) ≤ R

/-- **The integrity and work claims give the reference circuit's containment** (bridge `IntegrityContains`). On every
run `o` where the sampled-proofs claim holds (`PeriodLive.Holds .sampled`: every transcript the openings of `o`'s
commitment open to has at most an `ε` share of the period circuit `C`'s units unbound) and PoUW's does
(`Holds .pouw`: at least `W` of work on the schedule), the developer's declaration over `[0, t)` at the locations
`X i`, with the baseline `θ o`, contains the reference circuit `ref` (`Placed.declaration` of `(S o).A`).

What is missing:

* Neither claim reads the actual circuit's gates. Integrity is about the values of `C`'s transcript under the
  commitment, work about the schedule's priced ops; containment (`Circuit.Containment`) is structural, each gate of
  `ref` a gate of the actual region reading the images of its reads with its operation. A prover that computes `C`'s
  transcript by another circuit (each product through two squarings, say) meets both claims, and its region contains
  no gate of `ref`'s products.
* At `ε > 0` an `ε` share of the units may be wrong, which gives at most a partial containment
  (`Nci.PartialContainment`, its dark gates charged by `Accounting.Guarantees.Actual.ThreadedRuntimeDark`), not
  `Contains`.
* `ref` is the reference circuit and `C` the period's: their link is NCI's check of the declaration
  (`NciCheck.Properties.DescThrottles`, in `security_properties`), and nothing builds a `Declared` from a compiled
  Program (`Accounting.Guarantees.Compose.CircuitCredit`'s docstring: it speaks of the declared circuit, not of a served
  workload).
* What closes it: a device-model premise that the hardware runs the period's circuit gate by gate (its region over
  `[0, t)` contains `C`), with NCI's containment of `ref` in `C`, at `ε = 0`, or the dark form at `ε > 0`; then
  integrity and work are not what gives it. -/
def IntegrityContains {ι Acc : Type} [Fintype Acc] {d : Device ι} {C : Verity.Circuit} {Cm O : Type}
    (r : PeriodLive d Acc C Cm O) (G : Verity.Game O) (s : Verity.Game.Strategy G) (c : Certificates) (p : Price)
    {L : Type} [DecidableEq L] (S : O → Actual.Run L) {n : ℕ} (X : Fin n → Finset L) (t : ℕ)
    (θ : ∀ o, Finset (Fin (S o).A.N)) (hθ : ∀ o, ∀ a ∈ θ o, (S o).A.isInput a = true) (ref : Declaration) : Prop :=
  ∀ o, r.Holds c p G s .sampled o → r.Holds c p G s .pouw o →
    ref.Contains ((S o).A.declaration (univ.biUnion X) t (θ o) (hθ o))

end Accounting.Assumptions
