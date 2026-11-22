module

public import Definitions.Pouw.Guarantees.Confidence
public import Definitions.Draft.ActualCircuit.Registration
public import Definitions.Draft.ActualCircuit.ComputeAccounting
public import Definitions.Draft.ActualCircuit.Policy

/-!
# The four protocols together (draft)

* `FourTogether`: registration, forced matrix multiplication (a), compute accounting and the policy in one epoch, as
  `ResourceAccounting.Guarantees.Composition` at `k = 4`: all four accept and the actual circuit is outside the
  intersection of their sets with probability at most the sum of their errors, which is forced matrix multiplication's
  alone; and its reading, that the actual circuit then lies in the egress set and the free set at the limit.
* `FourFromParts`: that `FourTogether` is that instance: `Composition` and the four statements give it.
* `DecidedInOrder`, `PooledComposition`: the monitoring period's composition of (b)'s epochs, which `Composition` can't
  state: the epochs' exponential-moment bounds multiply and their skip events add, so the period's tolerance
  `Λ · ln(1/δ₀)` is counted once.

Where the shape bends. Registration and the policy draw no challenge and by themselves constrain no actual circuit:
registration contributes its egress set through a physical premise at error `0`, and the policy's set is all circuits
on any transcript it accepts. Compute accounting draws no challenge of its own: `Composition` asks independent
challenges for each protocol, so it enters as what it adds to forced matrix multiplication, an inclusion at error `0`
(`ComputeAccountingGiven`). `Composition` adds errors, so it can't carry (b)'s pooling: an exponential moment
composes by multiplying, which `PooledComposition` states beside it.
-/

@[expose] public section

namespace Draft.ActualCircuit

open ResourceAccounting ResourceAccounting.PouwWork Verity Finset

open Classical in
/-- **The four in one epoch, as `Composition` at `k = 4`.** In the order registration, forced matrix multiplication,
compute accounting, policy: verifiers at `V₀`, `P.V`, `P.V`, `P.V`; acceptances, the registration valid, PoUW's `acc`
twice, and the free number `n` (the registered rate of `x` times the epoch's length less `Wb · Nmin`) at most `lim`;
sets, the egress set of the subtree at `i`, the forced set at `Wb · Nmin`, the forced set's complement with the free set
at `n`, and all circuits when `n ≤ lim`; errors `0`, `δdet + δluck + δskip`, `0`, `0`. Under the four's premises, for
every joint strategy `S` that `Composition` reads:

* all four accept and some set misses the actual circuit, read on each protocol's slice of the challenges, for at most a
  `δdet + δluck + δskip` share of the challenges (`Composition`'s conclusion);
* all four accept and the actual circuit is outside the egress set or outside the free set at `lim` for at most that
  share: had the node taken in no help during the epoch, its work forced matrix multiplication does not credit is at
  most `lim`.

The intersection of the four sets on a transcript the policy accepts lies in the egress set and the free set at
`n ≤ lim`, which gives the second from the first. -/
def FourTogether : Prop :=
  ∀ (D : Registration.Digests) (Aud : Registration.Auditor) (c : Registration.Composite) (reg : Registration)
    (i : String) (β : Fin (Site.ofLog reg).links.length → ℕ)
    (Ω : Fin 4 → Type) [∀ j, Fintype (Ω j)] [∀ j, DecidableEq (Ω j)]
    (P : Verdict (Ω 1) (Site.ofLog reg).Place) (𝓕 : Set (Actual.Run (Site.ofLog reg).Place))
    (V₀ : Finset (Site.ofLog reg).Place) (acc : Transcript (Site.ofLog reg).Place → Prop)
    (X : Finset (Site.ofLog reg).Place) (π : (A : Placed (Site.ofLog reg).Place) → Fin A.N → ℕ)
    (m : (A : Placed (Site.ofLog reg).Place) → Finset (Fin A.N)) (Wb Nmin ρ : ℕ) (δdet δluck δskip : ℝ)
    (digest : Site.Sheet → String) (x : Site.Loc) (cuda oracle : ℕ) (p : Price) (lim : ℕ)
    (S : ((j : Fin 4) → Ω j) → Actual.Run (Site.ofLog reg).Place),
    let I := Set.Ico P.ts P.td
    let n := freeNumber (registeredRate digest Aud (Site.ofLog reg) x cuda oracle p) (P.td - P.ts) (Wb * Nmin)
    let V : Fin 4 → Finset (Site.ofLog reg).Place := ![V₀, P.V, P.V, P.V]
    let accs : Fin 4 → Transcript (Site.ofLog reg).Place → Prop :=
      ![fun _ => Registration.Valid D Aud c reg, acc, acc, fun _ => n ≤ lim]
    let 𝒞 : Fin 4 → Transcript (Site.ofLog reg).Place → Set (Placed (Site.ofLog reg).Place) :=
      ![fun _ => egressSet reg i β, fun _ => forcedSet P.V X I π m (Wb * Nmin),
        fun _ => (forcedSet P.V X I π m (Wb * Nmin))ᶜ ∪ freeSet P.V X I π m n,
        fun _ => if n ≤ lim then Set.univ else ∅]
    let δ : Fin 4 → ℝ := ![0, δdet + δluck + δskip, 0, 0]
    (Site.ofLog reg).RatedOn β
        ↑(boundary (Site.ofLog reg).placeEnds ((Site.ofLog reg).placesIn ((Site.ofLog reg).under i))) →
      (Site.ofLog reg).HostsWhole i → RunsOnRegisteredLinks reg β 𝓕 →
      AcceptIsVerdict 𝓕 P acc →
      ForStrategies (Ω 1) 𝓕 P.V (fun S => Assumptions.PouwDetection S P Nmin δdet ∧
        Assumptions.PouwLuck S P Nmin δluck ∧
        Assumptions.PouwFloorOff S P X (fun r => π (S r).A) (fun r => m (S r).A) Wb Nmin ρ δskip) →
      AtRegisteredRate 𝓕 X π (registeredRate digest Aud (Site.ofLog reg) x cuda oracle p) →
      (∀ r, S r ∈ 𝓕) → CausalFrom S →
      (∀ j r r', (∀ k, k ≠ j → r k = r' k) → (S r).inputs (↑(V j))ᶜ = (S r').inputs (↑(V j))ᶜ) →
      ((univ.filter fun r => (∀ j, accs j ((S r).view (V j))) ∧
          ¬∀ j, Lies (V j) (𝒞 j) (fun y => S (Function.update r j y)) (r j)).card : ℝ) ≤
        (∑ j, δ j) * Fintype.card ((j : Fin 4) → Ω j) ∧
      ((univ.filter fun r => (∀ j, accs j ((S r).view (V j))) ∧
          (S r).A ∉ egressSet reg i β ∩ freeSet P.V X I π m lim).card : ℝ) ≤
        (δdet + δluck + δskip) * Fintype.card ((j : Fin 4) → Ω j)

/-- **The four together are an instance of `Composition`**: `Composition`, registration's egress property, forced
matrix multiplication (a), what compute accounting adds and the policy give `FourTogether`. -/
def FourFromParts : Prop :=
  Guarantees.Composition.{0} → RegistrationEgress → ForcedMatmulEpoch → ComputeAccountingGiven → PolicyEpoch.{0} →
    FourTogether

/-- **Each epoch is decided before the next one's challenges**: epoch `j`'s acceptance and shortfall, read on the
joint strategy `S`, are the same at any two challenges that agree on epochs `0` to `j`. Protocol structure: an epoch's
verdict is read before a later epoch's challenges are drawn. New. -/
def DecidedInOrder {L : Type} {n : ℕ} {Ω : Fin n → Type} (S : ((j : Fin n) → Ω j) → Actual.Run L)
    (V : Fin n → Finset L) (acc : Fin n → Transcript L → Prop) (d : Fin n → Transcript L → Placed L → ℝ) : Prop :=
  ∀ j r r', (∀ k, k ≤ j → r k = r' k) →
    (acc j ((S r).view (V j)) ↔ acc j ((S r').view (V j))) ∧
      d j ((S r).view (V j)) (S r).A = d j ((S r').view (V j)) (S r').A

open Classical in
/-- **A monitoring period's epochs, pooled.** For `n` epochs, each with its own challenges `Ω j`, verifier at `V j`
accepting by `acc j`, and an exponential-moment bound at `Λ` off a skip event of probability `δ j` on the actual
circuit's shortfall `d j ≥ 0` (`BudgetSound`, (b)'s per-epoch statement), and every joint strategy `S` that
`Composition` reads whose epochs are decided in order (`DecidedInOrder`): every epoch accepts and the shortfalls sum to
at least `Λ · ln(1/δ₀)` for at most a `δ₀ + ∑ δ j` share of the challenges. The skip events add, the moments multiply,
and the tolerance is counted once for the period. Not an instance of `Composition`, which adds each protocol's error at
a set fixed per protocol; in a monitoring period the epochs are its execution blocks. -/
def PooledComposition : Prop :=
  ∀ {L : Type} [LinearOrder L] {n : ℕ} (Ω : Fin n → Type) [∀ j, Fintype (Ω j)] [∀ j, DecidableEq (Ω j)]
    (𝓕 : Set (Actual.Run L)) (V : Fin n → Finset L) (acc : Fin n → Transcript L → Prop)
    (d : Fin n → Transcript L → Placed L → ℝ) (Λ δ₀ : ℝ) (δ : Fin n → ℝ) (S : ((j : Fin n) → Ω j) → Actual.Run L),
    0 < Λ → 0 < δ₀ → (∀ j τ A, 0 ≤ d j τ A) →
    (∀ j, BudgetSound (Ω j) 𝓕 (V j) (acc j) (d j) Λ (δ j)) →
    (∀ r, S r ∈ 𝓕) → CausalFrom S →
    (∀ j r r', (∀ k, k ≠ j → r k = r' k) → (S r).inputs (↑(V j))ᶜ = (S r').inputs (↑(V j))ᶜ) →
    DecidedInOrder S V acc d →
    ((univ.filter fun r => (∀ j, acc j ((S r).view (V j))) ∧
        Λ * Real.log (1 / δ₀) ≤ ∑ j, d j ((S r).view (V j)) (S r).A).card : ℝ) ≤
      (δ₀ + ∑ j, δ j) * Fintype.card ((j : Fin n) → Ω j)

end Draft.ActualCircuit
