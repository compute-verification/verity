import Definitions.Accounting.Assumptions.Onsite
import Definitions.Accounting.Assumptions.Coverage
import Definitions.Accounting.Onsite.Params
import Definitions.Nci.Assumptions.Threads

/-!
# The onsite end-to-end overhead guarantee (trusted)

The end-to-end overhead proof's onsite spine (note:training-20261027T0134Z-draft-e2e-overhead-proof):

* `OnsiteTimeLowerBound` (step 5): on the onsite class (`Accounting.OnsiteClass`) at valid parameters
  (`Onsite.Valid`), under `Nci.Assumptions.WeightsIncompressible`, `Accounting.Guarantees.Actual.DampenerWindowsCap`'s
  `b_a M ≤ n ⌈t/ω⌉₊ K Λ(K)` inverted: `M ≥ N` takes `t ≥ t_min = ω (b_a N/(n K Λ(K)) − 1) + 1` ticks (`Onsite.tMin`),
  the last window partial. In size, the actual circuit's gates on the servers over `[0, t)` number at most `Φ_total t`
  (`Onsite.computeTotal`), so a computation in the class that does `N` counted multiplications holds the servers'
  `Φ_total` gates a tick for at least `t_min` ticks.
* `OnsiteEndToEnd` (steps 3 to 7): at every strategy of the period's causal game, the probability that every
  subprotocol and every wipe accepts and the run succeeds at the task with a horizon below `t_min`, with no two
  openings of one position, is at most `∑_a δ_a + ∑_k ε_wipe,k + η`: Layer 2's errors (`PeriodLive.Sound`, as
  `Accounting.Guarantees.JointPeriodLive` takes them, and the wipes' survivor bounds) and coverage's
  (`Accounting.Assumptions.Coverage`). The bridges from the claims to the class are
  `Accounting.Guarantees.Onsite.ClaimsImplyOnsiteClass`'s: the device model per run, `Accounting.CertifierLinks`,
  `Accounting.Assumptions.WipeState` and `Accounting.Assumptions.IntegrityContains`. None has an error as stated, so the
  spine's `δ_bridge` is `0`.

Each has an honest statement (`_honest`: its hypotheses hold at a constructed instance where its conclusion says
something) and a nontrivial one (`_nontrivial`: without one of its premises it is false at a constructed instance; for
`OnsiteEndToEnd`, at a verifier that accepts everything), the top's rule of 26 Oct (LEAN-66).

The proofs are `Accounting.SecurityProofs.OnsiteTimeLowerBound` and `OnsiteEndToEnd`, and their witnesses'.
-/

namespace Accounting.Guarantees

open Accounting Verity Finset

/-- **The onsite time floor** (step 5). For an actual circuit `A` in the onsite class (`OnsiteClass`) at valid onsite
parameters `P` (`Onsite.Valid`), a wire-to-link map `link`, a baseline `θ` and the horizon `t`, with the reference
circuit's weights incompressible at `(Γ, K = R + α F ω)`: when `M = M_r + M_d` counted multiplications, at most `T`'s
(`M ≤ |T.M|`), number at least `N`, the horizon is at least `t_min = ω (b_a N/(n K Λ(K)) − 1) + 1` ticks
(`Onsite.tMin`), and the gates `A` places on the servers during `[0, t)` number at most `Φ_total t`. -/
def OnsiteTimeLowerBound : Prop :=
  ∀ {L E : Type} [Fintype E] [DecidableEq L] (P : Onsite L E) (A : Placed L) (link : Fin A.N → Fin A.N → E)
    (θ : Finset (Fin A.N)) (hθ : ∀ a ∈ θ, A.isInput a = true) (t M N : ℕ),
    P.Valid → OnsiteClass A P.hw.ℓ P.hw.compute P.hw.ends P.hw.β link P.n P.X θ hθ t P.ω P.R P.F P.α P.c →
    Nci.Assumptions.WeightsIncompressible P.c P.T P.Γ P.K P.b_w P.b_a → M ≤ P.T.M.card → N ≤ M →
      P.tMin N ≤ t ∧ (A.region (univ.biUnion P.X) (Set.Iio t)).card ≤ P.computeTotal * t

/-- **`OnsiteTimeLowerBound`'s hypotheses hold where its conclusion says something**: some actual circuit is in the
onsite class at valid parameters whose weights are incompressible, with `N ≤ M ≤ |T.M|` counted multiplications and a
positive floor `t_min N`. -/
def OnsiteTimeLowerBound_honest : Prop :=
  ∃ (L E : Type) (_ : Fintype E) (_ : DecidableEq L) (P : Onsite L E) (A : Placed L) (link : Fin A.N → Fin A.N → E)
    (θ : Finset (Fin A.N)) (hθ : ∀ a ∈ θ, A.isInput a = true) (t M N : ℕ),
    P.Valid ∧ OnsiteClass A P.hw.ℓ P.hw.compute P.hw.ends P.hw.β link P.n P.X θ hθ t P.ω P.R P.F P.α P.c ∧
    Nci.Assumptions.WeightsIncompressible P.c P.T P.Γ P.K P.b_w P.b_a ∧ M ≤ P.T.M.card ∧ N ≤ M ∧ 0 < P.tMin N

/-- **`OnsiteTimeLowerBound` needs `WeightsIncompressible`**: without it, the time floor is false. -/
def OnsiteTimeLowerBound_nontrivial : Prop :=
  ¬ ∀ {L E : Type} [Fintype E] [DecidableEq L] (P : Onsite L E) (A : Placed L) (link : Fin A.N → Fin A.N → E)
    (θ : Finset (Fin A.N)) (hθ : ∀ a ∈ θ, A.isInput a = true) (t M N : ℕ),
    P.Valid → OnsiteClass A P.hw.ℓ P.hw.compute P.hw.ends P.hw.β link P.n P.X θ hθ t P.ω P.R P.F P.α P.c →
    M ≤ P.T.M.card → N ≤ M →
      P.tMin N ≤ t ∧ (A.region (univ.biUnion P.X) (Set.Iio t)).card ≤ P.computeTotal * t

/-- **The onsite end-to-end guarantee** (steps 3 to 7). On any device and any period read off the run of a causal game
`G`, whose five subprotocols have their per-period guarantees over `G` at certificates `c`, price `p` and errors `δ`
(`PeriodLive.Sound`, as `JointPeriodLive` takes them), with wipes `w` read off the same run, each accepted while it fails
its survivor bound at `f` with probability at most `ε_wipe k` at every strategy, at valid onsite parameters `P`, and a
map `S` from the runs to their actual circuits (with the wire-to-link maps `link` and baselines `θ`) over the horizon
`t`, with the run's counted multiplications `M o ≤ |T.M|` and whether it succeeds at the task, `success o`: when the
reference circuit's weights are incompressible, every run's actual circuit is causal and within the servers' compute
(the device model), the bridges `CertifierLinks`, `WipeState` and `IntegrityContains` hold, and succeeding in the onsite
class takes `N` counted multiplications but with probability `η` (`Assumptions.Coverage`), then at every strategy the
probability that every subprotocol and every wipe accepts and the run succeeds with the horizon below `t_min`
(`Onsite.tMin N`), with no two openings of one position under the run's commitment (`PeriodLive.Break`), is at most
`∑_a δ_a + ∑_k ε_wipe k + η`. -/
def OnsiteEndToEnd : Prop :=
  ∀ {ι Acc : Type} [Fintype Acc] (d : Device ι) (C : Verity.Circuit) (Cm O : Type) (r : PeriodLive d Acc C Cm O)
    (G : Verity.Game O) (c : Certificates) (p : Price) (δ : Audit → ℝ) (m : ℕ) (w : Fin m → WipeLive d O) (f : ℝ)
    (εWipe : Fin m → ℝ) {L E : Type} [Fintype E] [DecidableEq L] (P : Onsite L E) (S : O → Actual.Run L)
    (link : ∀ o, Fin (S o).A.N → Fin (S o).A.N → E) (θ : ∀ o, Finset (Fin (S o).A.N))
    (hθ : ∀ o, ∀ a ∈ θ o, (S o).A.isInput a = true) (t : ℕ) (M : O → ℕ) (success : O → Prop) (N : ℕ) (η : ℝ),
    P.Valid → Nci.Assumptions.WeightsIncompressible P.c P.T P.Γ P.K P.b_w P.b_a → (∀ o, M o ≤ P.T.M.card) →
    (∀ a, r.Sound G c p a (δ a)) →
    (∀ k (s : Verity.Game.Strategy G),
      Verity.Game.prob (fun o => (w k).accepts o ∧ ¬ (w k).Survives f o) G s ≤ ENNReal.ofReal (εWipe k)) →
    (∀ o, (S o).A.Causal P.hw.ℓ) → (∀ o, (S o).A.ComputeRate (Set.range P.X) (fun _ => 1) P.hw.compute) →
    CertifierLinks S P.hw.ends P.hw.β link P.n P.X P.α P.F → Assumptions.WipeState S w f P.X t P.ω P.R →
    Assumptions.Coverage G
      (fun o => OnsiteClass (S o).A P.hw.ℓ P.hw.compute P.hw.ends P.hw.β (link o) P.n P.X (θ o) (hθ o) t P.ω P.R P.F
        P.α P.c) success M N η →
    ∀ s : Verity.Game.Strategy G, Assumptions.IntegrityContains r G s c p S P.X t θ hθ P.c →
      Verity.Game.prob (fun o => ((∀ a, r.accepts a o) ∧ ∀ k, (w k).accepts o) ∧ success o ∧ (t : ℝ) < P.tMin N ∧
          ¬ r.Break G s o) G s ≤
        ∑ a, ENNReal.ofReal (δ a) + ∑ k, ENNReal.ofReal (εWipe k) + ENNReal.ofReal η

/-- **`OnsiteEndToEnd`'s hypotheses hold where its conclusion says something**: at some period, valid parameters and
world, every hypothesis holds at errors `0`, every subprotocol and every wipe accepts and the run succeeds with no break
with probability 1 at every strategy, and the floor `t_min N` is positive. -/
def OnsiteEndToEnd_honest : Prop :=
  ∃ (ι Acc : Type) (_ : Fintype Acc) (d : Device ι) (C : Verity.Circuit) (Cm O : Type) (r : PeriodLive d Acc C Cm O)
    (G : Verity.Game O) (c : Certificates) (p : Price) (m : ℕ) (w : Fin m → WipeLive d O) (f : ℝ)
    (L E : Type) (_ : Fintype E) (_ : DecidableEq L) (P : Onsite L E) (S : O → Actual.Run L)
    (link : ∀ o, Fin (S o).A.N → Fin (S o).A.N → E) (θ : ∀ o, Finset (Fin (S o).A.N))
    (hθ : ∀ o, ∀ a ∈ θ o, (S o).A.isInput a = true) (t : ℕ) (M : O → ℕ) (success : O → Prop) (N : ℕ),
    P.Valid ∧ Nci.Assumptions.WeightsIncompressible P.c P.T P.Γ P.K P.b_w P.b_a ∧ (∀ o, M o ≤ P.T.M.card) ∧
    (∀ a, r.Sound G c p a 0) ∧
    (∀ k (s : Verity.Game.Strategy G), Verity.Game.prob (fun o => (w k).accepts o ∧ ¬ (w k).Survives f o) G s = 0) ∧
    (∀ o, (S o).A.Causal P.hw.ℓ) ∧ (∀ o, (S o).A.ComputeRate (Set.range P.X) (fun _ => 1) P.hw.compute) ∧
    CertifierLinks S P.hw.ends P.hw.β link P.n P.X P.α P.F ∧ Assumptions.WipeState S w f P.X t P.ω P.R ∧
    Assumptions.Coverage G
      (fun o => OnsiteClass (S o).A P.hw.ℓ P.hw.compute P.hw.ends P.hw.β (link o) P.n P.X (θ o) (hθ o) t P.ω P.R P.F
        P.α P.c) success M N 0 ∧
    (∀ s : Verity.Game.Strategy G, Assumptions.IntegrityContains r G s c p S P.X t θ hθ P.c) ∧
    (∀ s : Verity.Game.Strategy G,
      Verity.Game.prob (fun o => ((∀ a, r.accepts a o) ∧ ∀ k, (w k).accepts o) ∧ success o ∧ ¬ r.Break G s o) G s =
        1) ∧
    0 < P.tMin N

/-- **`OnsiteEndToEnd` needs `IntegrityContains`, coverage and `WeightsIncompressible`, at a verifier that accepts
everything**: with any one of the three dropped, the statement is false even for periods whose every subprotocol and
every wipe accepts every run. -/
def OnsiteEndToEnd_nontrivial : Prop :=
  (¬ ∀ {ι Acc : Type} [Fintype Acc] (d : Device ι) (C : Verity.Circuit) (Cm O : Type) (r : PeriodLive d Acc C Cm O)
      (G : Verity.Game O) (c : Certificates) (p : Price) (δ : Audit → ℝ) (m : ℕ) (w : Fin m → WipeLive d O) (f : ℝ)
      (εWipe : Fin m → ℝ) {L E : Type} [Fintype E] [DecidableEq L] (P : Onsite L E) (S : O → Actual.Run L)
      (link : ∀ o, Fin (S o).A.N → Fin (S o).A.N → E) (θ : ∀ o, Finset (Fin (S o).A.N))
      (hθ : ∀ o, ∀ a ∈ θ o, (S o).A.isInput a = true) (t : ℕ) (M : O → ℕ) (success : O → Prop) (N : ℕ) (η : ℝ),
      (∀ a o, r.accepts a o) → (∀ k o, (w k).accepts o) →
      P.Valid → Nci.Assumptions.WeightsIncompressible P.c P.T P.Γ P.K P.b_w P.b_a → (∀ o, M o ≤ P.T.M.card) →
      (∀ a, r.Sound G c p a (δ a)) →
      (∀ k (s : Verity.Game.Strategy G),
        Verity.Game.prob (fun o => (w k).accepts o ∧ ¬ (w k).Survives f o) G s ≤ ENNReal.ofReal (εWipe k)) →
      (∀ o, (S o).A.Causal P.hw.ℓ) → (∀ o, (S o).A.ComputeRate (Set.range P.X) (fun _ => 1) P.hw.compute) →
      CertifierLinks S P.hw.ends P.hw.β link P.n P.X P.α P.F → Assumptions.WipeState S w f P.X t P.ω P.R →
      Assumptions.Coverage G
        (fun o => OnsiteClass (S o).A P.hw.ℓ P.hw.compute P.hw.ends P.hw.β (link o) P.n P.X (θ o) (hθ o) t P.ω P.R
          P.F P.α P.c) success M N η →
      ∀ s : Verity.Game.Strategy G,
        Verity.Game.prob (fun o => ((∀ a, r.accepts a o) ∧ ∀ k, (w k).accepts o) ∧ success o ∧
            (t : ℝ) < P.tMin N ∧ ¬ r.Break G s o) G s ≤
          ∑ a, ENNReal.ofReal (δ a) + ∑ k, ENNReal.ofReal (εWipe k) + ENNReal.ofReal η) ∧
  (¬ ∀ {ι Acc : Type} [Fintype Acc] (d : Device ι) (C : Verity.Circuit) (Cm O : Type) (r : PeriodLive d Acc C Cm O)
      (G : Verity.Game O) (c : Certificates) (p : Price) (δ : Audit → ℝ) (m : ℕ) (w : Fin m → WipeLive d O) (f : ℝ)
      (εWipe : Fin m → ℝ) {L E : Type} [Fintype E] [DecidableEq L] (P : Onsite L E) (S : O → Actual.Run L)
      (link : ∀ o, Fin (S o).A.N → Fin (S o).A.N → E) (θ : ∀ o, Finset (Fin (S o).A.N))
      (hθ : ∀ o, ∀ a ∈ θ o, (S o).A.isInput a = true) (t : ℕ) (M : O → ℕ) (success : O → Prop) (N : ℕ) (η : ℝ),
      (∀ a o, r.accepts a o) → (∀ k o, (w k).accepts o) →
      P.Valid → Nci.Assumptions.WeightsIncompressible P.c P.T P.Γ P.K P.b_w P.b_a → (∀ o, M o ≤ P.T.M.card) →
      (∀ a, r.Sound G c p a (δ a)) →
      (∀ k (s : Verity.Game.Strategy G),
        Verity.Game.prob (fun o => (w k).accepts o ∧ ¬ (w k).Survives f o) G s ≤ ENNReal.ofReal (εWipe k)) →
      (∀ o, (S o).A.Causal P.hw.ℓ) → (∀ o, (S o).A.ComputeRate (Set.range P.X) (fun _ => 1) P.hw.compute) →
      CertifierLinks S P.hw.ends P.hw.β link P.n P.X P.α P.F → Assumptions.WipeState S w f P.X t P.ω P.R →
      ∀ s : Verity.Game.Strategy G, Assumptions.IntegrityContains r G s c p S P.X t θ hθ P.c →
        Verity.Game.prob (fun o => ((∀ a, r.accepts a o) ∧ ∀ k, (w k).accepts o) ∧ success o ∧
            (t : ℝ) < P.tMin N ∧ ¬ r.Break G s o) G s ≤
          ∑ a, ENNReal.ofReal (δ a) + ∑ k, ENNReal.ofReal (εWipe k) + ENNReal.ofReal η) ∧
  (¬ ∀ {ι Acc : Type} [Fintype Acc] (d : Device ι) (C : Verity.Circuit) (Cm O : Type) (r : PeriodLive d Acc C Cm O)
      (G : Verity.Game O) (c : Certificates) (p : Price) (δ : Audit → ℝ) (m : ℕ) (w : Fin m → WipeLive d O) (f : ℝ)
      (εWipe : Fin m → ℝ) {L E : Type} [Fintype E] [DecidableEq L] (P : Onsite L E) (S : O → Actual.Run L)
      (link : ∀ o, Fin (S o).A.N → Fin (S o).A.N → E) (θ : ∀ o, Finset (Fin (S o).A.N))
      (hθ : ∀ o, ∀ a ∈ θ o, (S o).A.isInput a = true) (t : ℕ) (M : O → ℕ) (success : O → Prop) (N : ℕ) (η : ℝ),
      (∀ a o, r.accepts a o) → (∀ k o, (w k).accepts o) →
      P.Valid → (∀ o, M o ≤ P.T.M.card) →
      (∀ a, r.Sound G c p a (δ a)) →
      (∀ k (s : Verity.Game.Strategy G),
        Verity.Game.prob (fun o => (w k).accepts o ∧ ¬ (w k).Survives f o) G s ≤ ENNReal.ofReal (εWipe k)) →
      (∀ o, (S o).A.Causal P.hw.ℓ) → (∀ o, (S o).A.ComputeRate (Set.range P.X) (fun _ => 1) P.hw.compute) →
      CertifierLinks S P.hw.ends P.hw.β link P.n P.X P.α P.F → Assumptions.WipeState S w f P.X t P.ω P.R →
      Assumptions.Coverage G
        (fun o => OnsiteClass (S o).A P.hw.ℓ P.hw.compute P.hw.ends P.hw.β (link o) P.n P.X (θ o) (hθ o) t P.ω P.R
          P.F P.α P.c) success M N η →
      ∀ s : Verity.Game.Strategy G, Assumptions.IntegrityContains r G s c p S P.X t θ hθ P.c →
        Verity.Game.prob (fun o => ((∀ a, r.accepts a o) ∧ ∀ k, (w k).accepts o) ∧ success o ∧
            (t : ℝ) < P.tMin N ∧ ¬ r.Break G s o) G s ≤
          ∑ a, ENNReal.ofReal (δ a) + ∑ k, ENNReal.ofReal (εWipe k) + ENNReal.ofReal η)

end Accounting.Guarantees
