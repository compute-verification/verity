module

public import Definitions.Pouw.Guarantees.Confidence
public import Definitions.Draft.ActualCircuit.Registration
public import Definitions.Draft.ActualCircuit.ComputeAccounting
public import Definitions.Draft.ActualCircuit.Policy

/-!
# The four protocols together (draft)

* `FourFromParts`: registration, forced matrix multiplication (a), compute accounting and the policy in one epoch, as
  `ResourceAccounting.Guarantees.Composition` over the three that draw or read challenges (registration, forced matrix
  multiplication, the policy) with `ResourceAccounting.Guarantees.Fit` read on each run: all accept and the actual
  circuit is outside the egress set or compute accounting's bound exceeds the limit with probability at most forced
  matrix multiplication's error. Compute accounting draws no challenge of its own and accepts when forced matrix
  multiplication does, so it enters as `Fit` applied where (a)'s event holds, at error `0`.
* `epochsGame`, `epochsAccept`, `epochsSum`, `LateChallenges`, `ReadsBefore`, `RunsMatchGame`, `PooledEpochs`: the
  monitoring period's composition of (b)'s epochs. `Composition` adds errors, so it can't carry (b)'s pooling;
  `PooledEpochs` is Theorem S's tail (`ResourceAccounting.SecurityProofs.TS.budget_tail`, Lemma C's step 1) applied to
  the round game of the actual runs (`epochsGame`), with the skip events' union beside it. That the game's acceptance
  and deficiency are the runs' is a consequence of causality and of each epoch's reading ending before the next
  epoch's challenges are placed (`RunsMatchGame`), not a premise on the joint strategy.
-/

@[expose] public section

namespace Draft.ActualCircuit

open ResourceAccounting ResourceAccounting.PouwWork ResourceAccounting.TS Pouw.RelativeSecurity Verity Finset

open Classical in
/-- **The four in one epoch, from `Composition` and `Fit`.** `Composition`, `Fit`, registration's egress property,
forced matrix multiplication (a) and the policy give: in the order registration, forced matrix multiplication, policy,
with verifiers at `V₀`, `P.V`, `P.V`, acceptances the registration valid, `acceptsOn`, and compute accounting's number
`n = Φ · (td − ts) + ρ · h − Wb · Nmin` at most `lim`, under the four's premises and for every joint strategy `S` that
`Composition` reads, all accept and either the actual circuit is outside the egress set (some rated subtree sends more than its
registered boundary capacity), or at a
view holding forced matrix multiplication's challenge with the node's help at most `h` a run implementing `impl` has
`req_μ(impl) > lim`, for at most a `δdet + δskip` share of the challenges. The credits are read on the joint run through
forced matrix multiplication's slice. The device is fixed at `x` and the credit at `Wb · Nmin`, (a)'s; compute
accounting's statement reads both from the view (`ComputeAccountingEpoch`).

Not vacuous because the conclusion of its consequent fails for verifiers that accept every view against the joint
strategy that holds the challenge and runs a program with `req_μ > lim` and no forced work, where the premise that fails
is `PouwDetectionAt`. -/
def FourFromParts : Prop :=
  Guarantees.Composition.{0} → Guarantees.Fit.{0} → RegistrationEgress → ForcedMatmulEpoch → PolicyEpoch.{0} →
  ∀ (D : Registration.Digests) (Aud : Registration.Auditor) (c : Registration.Composite) (reg : Registration)
    (β : Fin (Site.ofLog reg).links.length → ℕ)
    (Ω : Fin 3 → Type) [∀ j, Fintype (Ω j)] [∀ j, DecidableEq (Ω j)]
    (P : Verdict (Ω 1) (Site.ofLog reg).Place) (𝓕 : Set (Actual.Run (Site.ofLog reg).Place))
    (V₀ : Finset (Site.ofLog reg).Place) (rd : Readout (Ω 1) (Site.ofLog reg).Place)
    (X : Finset (Site.ofLog reg).Place) (π : (A : Placed (Site.ofLog reg).Place) → Fin A.N → ℕ)
    (m : (A : Placed (Site.ofLog reg).Place) → Finset (Fin A.N)) (Wb Nmin ρ h : ℕ) (δdet δskip : ℝ)
    (digest : Site.Sheet → String) (x : Site.Loc) (cuda oracle : ℕ) (p : Price) (lim : ℕ)
    (impl : Actual.Run (Site.ofLog reg).Place → Prop)
    (S : ((j : Fin 3) → Ω j) → Actual.Run (Site.ofLog reg).Place),
    let Φ := registeredRate digest Aud (Site.ofLog reg) x cuda oracle p
    let n := Φ * (P.td - P.ts) + ρ * h - Wb * Nmin
    let V : Fin 3 → Finset (Site.ofLog reg).Place := ![V₀, P.V, P.V]
    let accs : Fin 3 → VerifierView (Site.ofLog reg).Place → Prop :=
      ![fun _ => Registration.Valid D Aud c reg, acceptsOn P rd, fun _ => n ≤ lim]
    RunsOnRegisteredLinks reg β 𝓕 →
      PouwDetectionAt 𝓕 P rd Nmin δdet → PouwFloorAt 𝓕 P rd X π m Wb ρ δskip →
      AtRegisteredRate 𝓕 P.V X π (fun _ => Φ) →
      Additive (creditedRuns (Ω := Ω 1) 𝓕 P X π m (fun _ => x) x (Wb * Nmin) ρ h) (epochUse P X π ρ) (Wb * Nmin)
        (req (epochUse P X π ρ) 𝓕 impl) impl →
      (∀ r, S r ∈ 𝓕) → CausalFrom S →
      (∀ j r r', (∀ k, k ≠ j → r k = r' k) → (S r).inputs (↑(V j))ᶜ = (S r').inputs (↑(V j))ᶜ) →
      ((univ.filter fun r => (∀ j, accs j ((S r).view (V j))) ∧
          ¬((S r).A ∈ egressSet reg β ∧
            (rd.chal ((S r).view P.V) = r 1 → (S r).A.ingress (X ∪ P.V) (Set.Ico P.ts P.td) ≤ h → impl (S r) →
              req (epochUse P X π ρ) 𝓕 impl ≤ (lim : ℕ∞)))).card : ℝ) ≤
        (δdet + δskip) * Fintype.card ((j : Fin 3) → Ω j)

/-- A path that is `ω₀` in every round. -/
def fillPaths {Ω : Type} (ω₀ : Ω) : (n : ℕ) → Paths Ω n
  | 0 => ()
  | n + 1 => (ω₀, fillPaths ω₀ n)

/-- **The round game of a monitoring period's actual runs**: round `j`'s share and check are epoch `j`'s `d j` and
`pass j` read on the strategy over epoch `j`'s challenge, the earlier epochs' challenges fixed and the later ones at
`ω₀`. Its rounds depend only on the challenges so far, as `RoundGame`'s do. -/
def epochsGame {Ω L : Type} (ω₀ : Ω) : (n : ℕ) → (Paths Ω n → Actual.Run L) →
    (Fin n → (Ω → Actual.Run L) → Ω → ℝ) → (Fin n → (Ω → Actual.Run L) → Ω → Prop) → RoundGame Ω n
  | 0, _, _, _ => .done
  | n + 1, S, d, pass =>
    .round (d 0 fun ω => S (ω, fillPaths ω₀ n)) (pass 0 fun ω => S (ω, fillPaths ω₀ n))
      fun ω => epochsGame ω₀ n (fun y => S (ω, y)) (fun j => d j.succ) (fun j => pass j.succ)

/-- **Every epoch passes on the actual runs**: epoch `j`'s `pass j` read on the strategy over epoch `j`'s challenge, the
other epochs' challenges at their values in `x`. -/
def epochsAccept {Ω L : Type} : (n : ℕ) → (Paths Ω n → Actual.Run L) → (Fin n → (Ω → Actual.Run L) → Ω → Prop) →
    Paths Ω n → Prop
  | 0, _, _, _ => True
  | n + 1, S, pass, x =>
    pass 0 (fun ω => S (ω, x.2)) x.1 ∧ epochsAccept n (fun y => S (x.1, y)) (fun j => pass j.succ) x.2

/-- **The epochs' shortfalls summed on the actual runs**, each read as `epochsAccept` reads its check. -/
def epochsSum {Ω L : Type} : (n : ℕ) → (Paths Ω n → Actual.Run L) → (Fin n → (Ω → Actual.Run L) → Ω → ℝ) →
    Paths Ω n → ℝ
  | 0, _, _, _ => 0
  | n + 1, S, d, x => d 0 (fun ω => S (ω, x.2)) x.1 + epochsSum n (fun y => S (x.1, y)) (fun j => d j.succ) x.2

/-- **Each epoch's challenges come late**: the verifier `𝒱` places epoch `k + j + 1`'s challenges at or after tick
`c (k + j + 1)`, so what `𝒱` holds before it and what crosses from `𝒱` to the rest before it are the same whatever the
later epochs' challenges. Protocol structure: the verifier draws an epoch's challenge after the earlier epochs' ticks. -/
def LateChallenges {Ω L : Type} [LinearOrder L] (𝒱 : Finset L) (c : ℕ → ℕ) :
    (n k : ℕ) → (Paths Ω n → Actual.Run L) → Prop
  | 0, _, _ => True
  | n + 1, k, S =>
    (∀ ω (y y' : Paths Ω n), (S (ω, y)).before 𝒱 (c (k + 1)) = (S (ω, y')).before 𝒱 (c (k + 1)) ∧
      (S (ω, y)).crossing ↑𝒱 (↑𝒱)ᶜ (Set.Iio (c (k + 1))) = (S (ω, y')).crossing ↑𝒱 (↑𝒱)ᶜ (Set.Iio (c (k + 1)))) ∧
    ∀ ω, LateChallenges 𝒱 c n (k + 1) fun y => S (ω, y)

/-- **`f` reads only ticks before `t`**: it is the same at any two runs whose gates placed before `t` agree. -/
def ReadsBefore {Ω L α : Type} [Fintype L] (f : (Ω → Actual.Run L) → Ω → α) (t : ℕ) : Prop :=
  ∀ S S' r r', (S r).before Finset.univ t = (S' r').before Finset.univ t → f S r = f S' r'

/-- **The game's acceptance and deficiency are the runs'** (to be proved): for a strategy over the period's challenges
that is causal, whose input gates off the verifier `𝒱` don't depend on the challenges, whose challenges come late
(`LateChallenges`), and whose epochs' checks and shortfalls read only ticks before the next epoch's challenges
(`ReadsBefore`), `epochsGame` accepts and has deficiency exactly where the runs do (`epochsAccept`, `epochsSum`): the
later epochs' challenges, set to `ω₀` in the game, change nothing an earlier epoch reads. New. Not vacuous because:
I cannot name one; it is to be a theorem, and its content is that the period's epochs are decided in order. -/
def RunsMatchGame : Prop :=
  ∀ {Ω L : Type} [LinearOrder L] [Fintype L] (n : ℕ) (ω₀ : Ω) (S : Paths Ω n → Actual.Run L) (𝒱 : Finset L)
    (c : ℕ → ℕ) (d : Fin n → (Ω → Actual.Run L) → Ω → ℝ) (pass : Fin n → (Ω → Actual.Run L) → Ω → Prop),
    CausalFrom S → (∀ x x', (S x).inputs (↑𝒱)ᶜ = (S x').inputs (↑𝒱)ᶜ) → LateChallenges 𝒱 c n 0 S →
    (∀ j : Fin n, ReadsBefore (d j) (c (j + 1))) → (∀ j : Fin n, ReadsBefore (pass j) (c (j + 1))) →
    ∀ x, ((epochsGame ω₀ n S d pass).accepts x ↔ epochsAccept n S pass x) ∧
      (epochsGame ω₀ n S d pass).deficiency x = epochsSum n S d x

/-- **A monitoring period's epochs, pooled**: Theorem S's tail (`budget_tail`) at `epochsGame`, read on the actual runs
by `RunsMatchGame`, with the skip events' union beside it. For `n` epochs at the verifier `𝒱`, each with an
exponential-moment bound at `Λ` on its shortfall `d j` where its check `acc j` and its keep event `keep j` hold ((b)'s
conclusion, `ForcedMatmulEpochPooled`), and `keep j` failing with probability at most `δ j` (the new A3,
`PouwFloorAt`), every strategy over the period's challenges that is causal, has its runs in `𝓕` and its inputs off `𝒱`
fixed, whose challenges come late and whose epochs read only ticks before the next epoch's challenges, accepts in every
epoch with the shortfalls summing to at least `Λ · t` with probability at most `exp(−t) + ∑ δ j`: the moments multiply
and the period's tolerance is counted once.

Not vacuous because its conclusion fails at `t = 1`, `∑ δ j = 0`, for the verifier that accepts every view against
the strategy that holds every challenge and computes nothing in any epoch (every shortfall positive, so the sum is at
least `Λ` at a small enough `Λ`), where the premise that fails is the per-epoch moment bound. -/
def PooledEpochs : Prop :=
  ∀ {Ω L : Type} [Fintype Ω] [LinearOrder L] [Fintype L] (n : ℕ) (ω₀ : Ω) (𝓕 : Set (Actual.Run L))
    (𝒱 : Finset L) (c : ℕ → ℕ) (S : Paths Ω n → Actual.Run L)
    (acc keep : Fin n → (Ω → Actual.Run L) → Ω → Prop) (d : Fin n → (Ω → Actual.Run L) → Ω → ℝ) (Λ t : ℝ)
    (δ : Fin n → ℝ),
    0 < Λ →
    (∀ j, Sound Ω 𝓕 𝒱 (fun _ => True)
      (fun S' _ => RoundGame.BudgetAt Λ (.round (d j S') (fun r => acc j S' r ∧ keep j S' r) fun _ => .done)) 0) →
    (∀ j, Sound Ω 𝓕 𝒱 (fun _ => True) (keep j) (δ j)) →
    (∀ x, S x ∈ 𝓕) → CausalFrom S → (∀ x x', (S x).inputs (↑𝒱)ᶜ = (S x').inputs (↑𝒱)ᶜ) →
    LateChallenges 𝒱 c n 0 S →
    (∀ j : Fin n, ReadsBefore (acc j) (c (j + 1))) → (∀ j : Fin n, ReadsBefore (keep j) (c (j + 1))) →
    (∀ j : Fin n, ReadsBefore (d j) (c (j + 1))) →
    prob (fun x => epochsAccept n S acc x ∧ Λ * t ≤ epochsSum n S d x) ≤ Real.exp (-t) + ∑ j, δ j

end Draft.ActualCircuit
