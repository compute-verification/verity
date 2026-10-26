import Definitions.Accounting.Bandwidth.Causal

/-!
# B3 and Mbw on a strategy's runs (trusted)

The rows B3 and Mbw of `note:console/20261012T0645Z-draft-actual-circuit-model` (v2.8), stated over the runs of a
strategy causal from every tick (`CausalFrom`) rather than over an abstract device of the bandwidth audit's game. The
strategy maps the store (for Mbw, the store and `z`) and the oracle to a run; the chip is the location `X`, the window is
`[t₀, t₀ + T)`, and the chip commits to `τ` of what it did in the window, the tags the verifier checks at `s` blocks
drawn uniformly. The hypotheses are the model's constraints on such a strategy:
- the chip's state at `t₀` is at most `M` bits (memory) and fixed by the store, or by the store and `z`, before the
  seed (`hpre`: the oracle doesn't reach it);
- every wire into the chip runs on a port of HBM, of the rest or of the oracle, and the links are alphabets
  (`Alphabet`), so the rest's ports carry at most `βs` bits a tick, their capacities summed: `c_side = βs·T`;
- HBM's ports are passive memory of the widths `wd` the chip's requests fix (`PassiveMemory`), or in the capacity forms
  HBM hears only the chip (`HearsChip`) and every tick's HBM signal counts at `βm` bits, HBM's ports' capacities summed;
- the oracle is a location the chip asks at most `qt` queries a tick (`OracleLocation`).

The chip's state is charged `M + 1` bits, its length included, so at `e = 0` the row's floor is
`k·b − M_chip − 1 − c_side − λ`. The proofs are in `Security/Proofs/Accounting/Bandwidth/Causal.lean` and
`CausalJoint.lean`: the chip of such a run is the tick transducer `Ticked.Chip`, whose bounds are the game's
`AuditInteractive`, `AuditInteractiveRO` and `AuditJoint`.
-/

namespace Bandwidth.Guarantees

open Pous (Bits pr)
open Accounting
open Bandwidth.Causal
open scoped ENNReal

/-- **B3, the bandwidth audit on a strategy's runs, from the links.** For a strategy `S` causal from every tick, the chip
`X` with its state at `t₀` at most `M` bits and fixed by the store, every wire into it on a port of HBM, the rest or the
oracle, the links alphabets, HBM's ports passive memory of widths `wd` and the oracle a location asked at most `qt`
queries a tick, and the chip committing to `τ` of what it did in the window: for every `m`, the probability over the
store, the oracle and the sample that the tags pass while

  `∑ wd + (M + 1) + βs·T + m·b + λ < k·b`

is at most `((k − 1)/N)^s + C(N, m)·2^{βs·T}/2^{h·m} + C(T·qt, k − m)/2^λ`, `βs` the rest's ports' capacities summed
and `∑ wd` the bits HBM delivered in the window. This is `AuditInteractive`'s bound at `S = M + 1`, `c = βs·T`,
`q = T·qt`: the console's B3 at `e > 0`, `m·b` less for relayed tags, with no `2^e`. Proof:
`Bandwidth.SecurityProofs.B3`.

Frozen (no-function, world-unlisted): stated over arbitrary strategies causal from every tick, with the chip's ports'
constraints as hypotheses that are not on the premises list, rather than a function's acceptance. -/
def B3 : Prop :=
  ∀ {L K : Type*} [LinearOrder L] {N b h : ℕ} (S : Store N b × Oracle N b h → Actual.Run L), CausalFrom S →
    ∀ (β : K → ℕ) (link : ∀ r, Fin (S r).A.N → Fin (S r).A.N → K), Alphabet S β link →
    ∀ (X : Finset L) (t₀ T M qt : ℕ), (∀ r, (S r).mem X t₀ ≤ M) →
    (∀ E O O', (S (E, O)).content X t₀ = (S (E, O')).content X t₀) →
    ∀ hbmPorts sidePorts orPorts : List (K × L),
    (∀ r g, ∀ a ∈ (S r).A.reads g, (S r).A.x a ∉ X → (S r).A.x g ∈ X →
      (link r g a, (S r).A.x g) ∈ hbmPorts ++ sidePorts ++ orPorts) →
    ∀ wd : ℕ → Acts L → ℕ, PassiveMemory S link X t₀ T hbmPorts wd →
    ∀ Q : ℕ → Acts L → List (Fin N × Bits b), (∀ u a, (Q u a).length ≤ qt) →
    OracleLocation S link X t₀ T orPorts Q →
    ∀ (τ : Acts L → Fin N → Bits h) (s k m lam : ℕ),
    pr (fun x : Store N b × Oracle N b h × (Fin s → Fin N) =>
        (∀ i, τ ((S (x.1, x.2.1)).acts ↑X (Set.Ico t₀ (t₀ + T))) (x.2.2 i) = x.2.1 (x.2.2 i, x.1 (x.2.2 i))) ∧
          (∑ i ∈ Finset.range T, wd (t₀ + i) ((S (x.1, x.2.1)).acts ↑X (Set.Ico t₀ (t₀ + i)))) + (M + 1) +
            (sidePorts.map fun p => β p.1).sum * T + m * b + lam < k * b)
      ≤ (((k - 1 : ℕ) : ℝ≥0∞) / N) ^ s +
          (N.choose m : ℝ≥0∞) * 2 ^ (T * (sidePorts.map fun p => β p.1).sum) / 2 ^ (h * m)
        + (((T * qt).choose (k - m) : ℕ) : ℝ≥0∞) / 2 ^ lam

/-- **B3 at `e = 0`.** `B3` when the rest's ports' signals don't depend on the oracle: the probability that the tags
pass while `∑ wd + (M + 1) + βs·T + λ < k·b` is at most `AuditInteractiveRO`'s bound
`((k − 1)/N)^s + s/2^h + C(T·qt, k)/2^λ`, with no relayed-tag term. So an accept gives the console's B3 at `e = 0`:
the chip read at least `k·b − M_chip − 1 − c_side − λ` bits from HBM in the window, `c_side = βs·T`. Proof:
`Bandwidth.SecurityProofs.B3RO`.

Frozen (no-function, world-unlisted): as `B3`. -/
def B3RO : Prop :=
  ∀ {L K : Type*} [LinearOrder L] {N b h : ℕ} (S : Store N b × Oracle N b h → Actual.Run L), CausalFrom S →
    ∀ (β : K → ℕ) (link : ∀ r, Fin (S r).A.N → Fin (S r).A.N → K), Alphabet S β link →
    ∀ (X : Finset L) (t₀ T M qt : ℕ), (∀ r, (S r).mem X t₀ ≤ M) →
    (∀ E O O', (S (E, O)).content X t₀ = (S (E, O')).content X t₀) →
    ∀ hbmPorts sidePorts orPorts : List (K × L),
    (∀ r g, ∀ a ∈ (S r).A.reads g, (S r).A.x a ∉ X → (S r).A.x g ∈ X →
      (link r g a, (S r).A.x g) ∈ hbmPorts ++ sidePorts ++ orPorts) →
    ∀ wd : ℕ → Acts L → ℕ, PassiveMemory S link X t₀ T hbmPorts wd →
    ∀ Q : ℕ → Acts L → List (Fin N × Bits b), (∀ u a, (Q u a).length ≤ qt) →
    OracleLocation S link X t₀ T orPorts Q →
    (∀ E O O' u, portSignals S link sidePorts (E, O) u = portSignals S link sidePorts (E, O') u) →
    ∀ (τ : Acts L → Fin N → Bits h) (s k lam : ℕ),
    pr (fun x : Store N b × Oracle N b h × (Fin s → Fin N) =>
        (∀ i, τ ((S (x.1, x.2.1)).acts ↑X (Set.Ico t₀ (t₀ + T))) (x.2.2 i) = x.2.1 (x.2.2 i, x.1 (x.2.2 i))) ∧
          (∑ i ∈ Finset.range T, wd (t₀ + i) ((S (x.1, x.2.1)).acts ↑X (Set.Ico t₀ (t₀ + i)))) + (M + 1) +
            (sidePorts.map fun p => β p.1).sum * T + lam < k * b)
      ≤ (((k - 1 : ℕ) : ℝ≥0∞) / N) ^ s + (s : ℝ≥0∞) / 2 ^ h + (((T * qt).choose k : ℕ) : ℝ≥0∞) / 2 ^ lam

/-- **B3 in the capacity form.** `B3` with HBM hearing only the chip in place of passive memory, and every tick's HBM
signal counted at `βm` bits, HBM's ports' capacities summed: the probability that the tags pass while
`βm·T + (M + 1) + βs·T + m·b + λ < k·b` is at most `B3`'s bound. It needs of HBM only that it computes no tag (what it
delivers is a function of the store and of what the chip did), so it holds without passive memory. Proof:
`Bandwidth.SecurityProofs.B3Capacity`.

Frozen (no-function, world-unlisted): as `B3`. -/
def B3Capacity : Prop :=
  ∀ {L K : Type*} [LinearOrder L] {N b h : ℕ} (S : Store N b × Oracle N b h → Actual.Run L), CausalFrom S →
    ∀ (β : K → ℕ) (link : ∀ r, Fin (S r).A.N → Fin (S r).A.N → K), Alphabet S β link →
    ∀ (X : Finset L) (t₀ T M qt : ℕ), (∀ r, (S r).mem X t₀ ≤ M) →
    (∀ E O O', (S (E, O)).content X t₀ = (S (E, O')).content X t₀) →
    ∀ hbmPorts sidePorts orPorts : List (K × L),
    (∀ r g, ∀ a ∈ (S r).A.reads g, (S r).A.x a ∉ X → (S r).A.x g ∈ X →
      (link r g a, (S r).A.x g) ∈ hbmPorts ++ sidePorts ++ orPorts) →
    HearsChip S link X t₀ T hbmPorts →
    ∀ Q : ℕ → Acts L → List (Fin N × Bits b), (∀ u a, (Q u a).length ≤ qt) →
    OracleLocation S link X t₀ T orPorts Q →
    ∀ (τ : Acts L → Fin N → Bits h) (s k m lam : ℕ),
    pr (fun x : Store N b × Oracle N b h × (Fin s → Fin N) =>
        (∀ i, τ ((S (x.1, x.2.1)).acts ↑X (Set.Ico t₀ (t₀ + T))) (x.2.2 i) = x.2.1 (x.2.2 i, x.1 (x.2.2 i))) ∧
          (hbmPorts.map fun p => β p.1).sum * T + (M + 1) + (sidePorts.map fun p => β p.1).sum * T + m * b + lam <
            k * b)
      ≤ (((k - 1 : ℕ) : ℝ≥0∞) / N) ^ s +
          (N.choose m : ℝ≥0∞) * 2 ^ (T * (sidePorts.map fun p => β p.1).sum) / 2 ^ (h * m)
        + (((T * qt).choose (k - m) : ℕ) : ℝ≥0∞) / 2 ^ lam

/-- **B3 in the capacity form at `e = 0`.** `B3RO` with HBM hearing only the chip in place of passive memory, every
tick's HBM signal counted at `βm` bits: the probability that the tags pass while `βm·T + (M + 1) + βs·T + λ < k·b` is
at most `B3RO`'s bound. Proof: `Bandwidth.SecurityProofs.B3CapacityRO`.

Frozen (no-function, world-unlisted): as `B3`. -/
def B3CapacityRO : Prop :=
  ∀ {L K : Type*} [LinearOrder L] {N b h : ℕ} (S : Store N b × Oracle N b h → Actual.Run L), CausalFrom S →
    ∀ (β : K → ℕ) (link : ∀ r, Fin (S r).A.N → Fin (S r).A.N → K), Alphabet S β link →
    ∀ (X : Finset L) (t₀ T M qt : ℕ), (∀ r, (S r).mem X t₀ ≤ M) →
    (∀ E O O', (S (E, O)).content X t₀ = (S (E, O')).content X t₀) →
    ∀ hbmPorts sidePorts orPorts : List (K × L),
    (∀ r g, ∀ a ∈ (S r).A.reads g, (S r).A.x a ∉ X → (S r).A.x g ∈ X →
      (link r g a, (S r).A.x g) ∈ hbmPorts ++ sidePorts ++ orPorts) →
    HearsChip S link X t₀ T hbmPorts →
    ∀ Q : ℕ → Acts L → List (Fin N × Bits b), (∀ u a, (Q u a).length ≤ qt) →
    OracleLocation S link X t₀ T orPorts Q →
    (∀ E O O' u, portSignals S link sidePorts (E, O) u = portSignals S link sidePorts (E, O') u) →
    ∀ (τ : Acts L → Fin N → Bits h) (s k lam : ℕ),
    pr (fun x : Store N b × Oracle N b h × (Fin s → Fin N) =>
        (∀ i, τ ((S (x.1, x.2.1)).acts ↑X (Set.Ico t₀ (t₀ + T))) (x.2.2 i) = x.2.1 (x.2.2 i, x.1 (x.2.2 i))) ∧
          (hbmPorts.map fun p => β p.1).sum * T + (M + 1) + (sidePorts.map fun p => β p.1).sum * T + lam < k * b)
      ≤ (((k - 1 : ℕ) : ℝ≥0∞) / N) ^ s + (s : ℝ≥0∞) / 2 ^ h + (((T * qt).choose k : ℕ) : ℝ≥0∞) / 2 ^ lam

/-- **Mbw, bandwidth adds by counting, on a strategy's runs, from the links** (`e = 0`). For a strategy `S` over the
store, `z` (uniform over `n` bits, independent of the store, the seed and the oracle) and the oracle, causal from every
tick, with the chip as in `B3RO` (its state at `t₀` fixed by the store and `z`, HBM's ports passive memory over the pair
(store, `z`)), and the chip's view determining `z` (`ζ` of its content at `t₀` and of what crossed into it in the window
is `z`): the probability over the store, `z`, the oracle and the sample that the tags pass while

  `∑ wd + (M + 1) + βs·T + λ < k·b + n`

is at most `AuditRO`'s error `((k − 1)/N)^s + s/2^h + C(T·qt, k)/2^λ`. Read with `cert_bw = k·b − (M + 1) − βs·T − λ`,
what `B3RO` certifies: an accept gives `n ≤ ∑ wd − cert_bw` at that error and no extra slack. This is the console's Mbw
for the chip of a strategy's run. Proof: `Bandwidth.SecurityProofs.Mbw`.

Frozen (no-function, world-unlisted): as `B3`. -/
def Mbw : Prop :=
  ∀ {L K : Type*} [LinearOrder L] {N b h n : ℕ} (S : (Store N b × Bits n) × Oracle N b h → Actual.Run L),
    CausalFrom S →
    ∀ (β : K → ℕ) (link : ∀ r, Fin (S r).A.N → Fin (S r).A.N → K), Alphabet S β link →
    ∀ (X : Finset L) (t₀ T M qt : ℕ), (∀ r, (S r).mem X t₀ ≤ M) →
    (∀ e O O', (S (e, O)).content X t₀ = (S (e, O')).content X t₀) →
    ∀ hbmPorts sidePorts orPorts : List (K × L),
    (∀ r g, ∀ a ∈ (S r).A.reads g, (S r).A.x a ∉ X → (S r).A.x g ∈ X →
      (link r g a, (S r).A.x g) ∈ hbmPorts ++ sidePorts ++ orPorts) →
    ∀ wd : ℕ → Acts L → ℕ, PassiveMemory S link X t₀ T hbmPorts wd →
    ∀ Q : ℕ → Acts L → List (Fin N × Bits b), (∀ u a, (Q u a).length ≤ qt) →
    OracleLocation S link X t₀ T orPorts Q →
    (∀ e O O' u, portSignals S link sidePorts (e, O) u = portSignals S link sidePorts (e, O') u) →
    ∀ ζ : List Bool → List (Cross L) → Bits n,
    (∀ E z O, ζ ((S ((E, z), O)).content X t₀)
      ((List.range T).map fun i => (S ((E, z), O)).crossing (↑X)ᶜ ↑X {t₀ + i}) = z) →
    ∀ (τ : Acts L → Fin N → Bits h) (s k lam : ℕ),
    pr (fun x : (Store N b × Bits n) × Oracle N b h × (Fin s → Fin N) =>
        (∀ i, τ ((S (x.1, x.2.1)).acts ↑X (Set.Ico t₀ (t₀ + T))) (x.2.2 i) = x.2.1 (x.2.2 i, x.1.1 (x.2.2 i))) ∧
          (∑ i ∈ Finset.range T, wd (t₀ + i) ((S (x.1, x.2.1)).acts ↑X (Set.Ico t₀ (t₀ + i)))) + (M + 1) +
            (sidePorts.map fun p => β p.1).sum * T + lam < k * b + n)
      ≤ (((k - 1 : ℕ) : ℝ≥0∞) / N) ^ s + (s : ℝ≥0∞) / 2 ^ h + (((T * qt).choose k : ℕ) : ℝ≥0∞) / 2 ^ lam

/-- **Mbw in the capacity form** (`e = 0`). `Mbw` with HBM hearing only the chip in place of passive memory, every
tick's HBM signal counted at `βm` bits: the probability that the tags pass while `βm·T + (M + 1) + βs·T + λ < k·b + n`
is at most `AuditRO`'s error. So an accept gives `n ≤ βm·T − cert_bw`, the console's `q ≤ β_HBM·T − cert_bw =
res_bw`, which needs of HBM only that it computes no tag. Proof: `Bandwidth.SecurityProofs.MbwCapacity`.

Frozen (no-function, world-unlisted): as `B3`. -/
def MbwCapacity : Prop :=
  ∀ {L K : Type*} [LinearOrder L] {N b h n : ℕ} (S : (Store N b × Bits n) × Oracle N b h → Actual.Run L),
    CausalFrom S →
    ∀ (β : K → ℕ) (link : ∀ r, Fin (S r).A.N → Fin (S r).A.N → K), Alphabet S β link →
    ∀ (X : Finset L) (t₀ T M qt : ℕ), (∀ r, (S r).mem X t₀ ≤ M) →
    (∀ e O O', (S (e, O)).content X t₀ = (S (e, O')).content X t₀) →
    ∀ hbmPorts sidePorts orPorts : List (K × L),
    (∀ r g, ∀ a ∈ (S r).A.reads g, (S r).A.x a ∉ X → (S r).A.x g ∈ X →
      (link r g a, (S r).A.x g) ∈ hbmPorts ++ sidePorts ++ orPorts) →
    HearsChip S link X t₀ T hbmPorts →
    ∀ Q : ℕ → Acts L → List (Fin N × Bits b), (∀ u a, (Q u a).length ≤ qt) →
    OracleLocation S link X t₀ T orPorts Q →
    (∀ e O O' u, portSignals S link sidePorts (e, O) u = portSignals S link sidePorts (e, O') u) →
    ∀ ζ : List Bool → List (Cross L) → Bits n,
    (∀ E z O, ζ ((S ((E, z), O)).content X t₀)
      ((List.range T).map fun i => (S ((E, z), O)).crossing (↑X)ᶜ ↑X {t₀ + i}) = z) →
    ∀ (τ : Acts L → Fin N → Bits h) (s k lam : ℕ),
    pr (fun x : (Store N b × Bits n) × Oracle N b h × (Fin s → Fin N) =>
        (∀ i, τ ((S (x.1, x.2.1)).acts ↑X (Set.Ico t₀ (t₀ + T))) (x.2.2 i) = x.2.1 (x.2.2 i, x.1.1 (x.2.2 i))) ∧
          (hbmPorts.map fun p => β p.1).sum * T + (M + 1) + (sidePorts.map fun p => β p.1).sum * T + lam <
            k * b + n)
      ≤ (((k - 1 : ℕ) : ℝ≥0∞) / N) ^ s + (s : ℝ≥0∞) / 2 ^ h + (((T * qt).choose k : ℕ) : ℝ≥0∞) / 2 ^ lam

end Bandwidth.Guarantees
