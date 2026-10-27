import Definitions.Accounting.Assumptions.Onsite
import Definitions.Accounting.Guarantees.JointLive

/-!
# The onsite bridges: Layer 2's claims imply the onsite class (trusted)

Step 3 of the end-to-end overhead guarantee (note:training-20261027T0134Z-draft-e2e-overhead-proof): the onsite class
(`Accounting.OnsiteClass`, the hypotheses of `Accounting.Guarantees.Actual.DampenerWindowsCap` that read the actual
circuit) holds on every run where the claims hold, each hypothesis from a claim or from the device model:

| hypothesis | from | status |
| --- | --- | --- |
| `Causal ℓ`, `ComputeRate` | the device model (Layer 1) | premise, per run |
| `Links`, `linkCapacity ≤ αF` | the device model: `DeclaredLinks` and the boundary links' capacity | proved, `CertifierLinksDeclared` |
| `state (X i) (kω) ≤ R` | the wipes' survivor bound | premise, `Accounting.Assumptions.WipeState` |
| `ref.Contains (declaration …)` | the integrity and work claims | premise, `Accounting.Assumptions.IntegrityContains` |

* `CertifierLinksDeclared`: `Accounting.CertifierLinks` from the links as declared (`DeclaredLinks`) and each
  location's boundary capacity at most `αF`. The network certifier's claim is not what gives it: `commWithin` counts
  the bits the certifiers saw cross over the period, at most `K`, and Layer 3 reads a capacity a tick. Reading the
  count needs Layer 3 with each window's ingress (`Placed.ingress`) at most a budget in place of `β_X ω`, and a
  realization tying `commWithin` to the actual circuit's boundary crossings under certifier coverage (`Covered`);
  neither is on `main`.
* `ClaimsImplyOnsiteClass`: on the runs where every subprotocol's claim holds and every wipe meets its survivor bound,
  the onsite class holds, given the bridges: `Pr[claims ∧ ¬ class] = 0`. No bridge has an error as stated, so
  `δ_bridge = 0`; `WipeState`'s determination form would bring the wipe's.
* `AcceptedOnsiteClass`: with Layer 2 (`JointPeriodLive`'s per-period guarantees and the wipes' survivor bounds),
  `Pr[every subprotocol and every wipe accepted ∧ ¬ class ∧ no break] ≤ ∑_a δ_a + ∑_k ε_wipe,k`: Steps 2 and 3 of the
  spine, which Step 4's `DampenerWindowsCap` reads on the class.

The proofs are in `Security/Proofs/Accounting/Onsite.lean`.
-/

namespace Accounting.Guarantees.Onsite

open Accounting Verity Finset

/-- **The links half of the onsite class, from the links as declared.** For a map `S` from runs to actual circuits whose
links are as declared (`DeclaredLinks`: each link one of at most `2^{β e}` signals a tick each way, and on every run
every wire between two leaves on a declared link joining them, within its capacity), with each location `X i`'s
boundary links carrying at most `α F` bits a tick, every run meets `OnsiteClass`'s links half
(`Accounting.CertifierLinks`). Both halves are the device model: the network certifier's count is not read. -/
def CertifierLinksDeclared : Prop :=
  ∀ {O L E : Type} [Fintype E] (S : O → Actual.Run L) (ends : E → L × L) (β : E → ℕ)
    (link : ∀ o, Fin (S o).A.N → Fin (S o).A.N → E) (n : ℕ) (X : Fin n → Finset L) (α F : ℝ),
    DeclaredLinks S ends β link → (∀ i, (linkCapacity ends β (X i) : ℝ) ≤ α * F) →
      CertifierLinks S ends β link n X α F

/-- **The claims imply the onsite class.** On any device and any period read off the run of a causal game `G` over a
circuit, with wipes `w` read off the same run and a map `S` from its runs to their actual circuits, at any strategy
`s`: when on every run the actual circuit is causal at `ℓ` and computes at most `Φ (X i)` gates a tick at each location
(the device model), its links meet `CertifierLinks`, and the bridges `WipeState` (the wipes' survivor bounds give each
window's starting state at most `R`) and `IntegrityContains` (the integrity and work claims give the reference
circuit `ref`'s containment in the declaration over `[0, t)`) hold, the probability that every subprotocol's claim
holds (`PeriodLive.Holds` at certificates `c` and price `p`) and every wipe meets its survivor bound at `f` while the
run's actual circuit is outside the onsite class (`Accounting.OnsiteClass`) is `0`. -/
def ClaimsImplyOnsiteClass : Prop :=
  ∀ {ι Acc : Type} [Fintype Acc] (d : Device ι) (C : Verity.Circuit) (Cm O : Type) (r : PeriodLive d Acc C Cm O)
    (G : Verity.Game O) (c : Certificates) (p : Price) (m : ℕ) (w : Fin m → WipeLive d O) (f : ℝ)
    {L E : Type} [Fintype E] [DecidableEq L] (S : O → Actual.Run L) (ℓ : L → L → ℕ) (Φ : Finset L → ℕ)
    (ends : E → L × L) (β : E → ℕ) (link : ∀ o, Fin (S o).A.N → Fin (S o).A.N → E) (n : ℕ) (X : Fin n → Finset L)
    (θ : ∀ o, Finset (Fin (S o).A.N)) (hθ : ∀ o, ∀ a ∈ θ o, (S o).A.isInput a = true) (t ω : ℕ) (R F α : ℝ)
    (ref : Declaration) (s : Verity.Game.Strategy G),
    (∀ o, (S o).A.Causal ℓ) → (∀ o, (S o).A.ComputeRate (Set.range X) (fun _ => 1) Φ) →
    CertifierLinks S ends β link n X α F → Assumptions.WipeState S w f X t ω R →
    Assumptions.IntegrityContains r G s c p S X t θ hθ ref →
      Verity.Game.prob (fun o => ((∀ a, r.Holds c p G s a o) ∧ ∀ k, (w k).Survives f o) ∧
          ¬ OnsiteClass (S o).A ℓ Φ ends β (link o) n X (θ o) (hθ o) t ω R F α ref) G s = 0

/-- **The accepted runs are in the onsite class, but for Layer 2's errors.** In `ClaimsImplyOnsiteClass`'s setting,
when the five subprotocols have their per-period guarantees over `G` at certificates `c`, price `p` and errors `δ`
(`PeriodLive.Sound`, as `JointPeriodLive` takes them) and each wipe `w k` is accepted while it fails its survivor bound
at `f` with probability at most `ε_wipe k` at every strategy (as `JointHorizonBlocksLive` takes them): at every
strategy, under the device model and the bridges there, the probability that every subprotocol and every wipe accepts
while the run's actual circuit is outside the onsite class, with no two openings of one position under the run's
commitment (`PeriodLive.Break`), is at most `∑_a δ_a + ∑_k ε_wipe k`. -/
def AcceptedOnsiteClass : Prop :=
  ∀ {ι Acc : Type} [Fintype Acc] (d : Device ι) (C : Verity.Circuit) (Cm O : Type) (r : PeriodLive d Acc C Cm O)
    (G : Verity.Game O) (c : Certificates) (p : Price) (δ : Audit → ℝ) (m : ℕ) (w : Fin m → WipeLive d O) (f : ℝ)
    (εWipe : Fin m → ℝ)
    {L E : Type} [Fintype E] [DecidableEq L] (S : O → Actual.Run L) (ℓ : L → L → ℕ) (Φ : Finset L → ℕ)
    (ends : E → L × L) (β : E → ℕ) (link : ∀ o, Fin (S o).A.N → Fin (S o).A.N → E) (n : ℕ) (X : Fin n → Finset L)
    (θ : ∀ o, Finset (Fin (S o).A.N)) (hθ : ∀ o, ∀ a ∈ θ o, (S o).A.isInput a = true) (t ω : ℕ) (R F α : ℝ)
    (ref : Declaration),
    (∀ a, r.Sound G c p a (δ a)) →
    (∀ k (s : Verity.Game.Strategy G),
      Verity.Game.prob (fun o => (w k).accepts o ∧ ¬ (w k).Survives f o) G s ≤ ENNReal.ofReal (εWipe k)) →
    ∀ s : Verity.Game.Strategy G,
    (∀ o, (S o).A.Causal ℓ) → (∀ o, (S o).A.ComputeRate (Set.range X) (fun _ => 1) Φ) →
    CertifierLinks S ends β link n X α F → Assumptions.WipeState S w f X t ω R →
    Assumptions.IntegrityContains r G s c p S X t θ hθ ref →
      Verity.Game.prob (fun o => ((∀ a, r.accepts a o) ∧ ∀ k, (w k).accepts o) ∧
          ¬ OnsiteClass (S o).A ℓ Φ ends β (link o) n X (θ o) (hθ o) t ω R F α ref ∧ ¬ r.Break G s o) G s ≤
        ∑ a, ENNReal.ofReal (δ a) + ∑ k, ENNReal.ofReal (εWipe k)

end Accounting.Guarantees.Onsite
