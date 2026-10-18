import Models.Accounting.Assumptions.TS
import Definitions.Accounting.TS.Steer

/-!
# Steering under one σ (trusted)

STEER-PROOF's route B (note:20261018T0303Z-draft-steer-proof), each `def G : Prop` proved as
`Accounting.SecurityProofs.TS.G` in `Proofs.Accounting.TS.Steer`:

* `LinSteerSegments`: LIN-STEER holds for a segmented consumer exactly when its noise-active segments all carry the
  same number of noise-active coordinates. A consumer row whose noise-active coordinates fall unevenly across its
  words (a dither's active set, or any row with a short last segment) refutes it: zeroing a segment of `a` coordinates
  saves `1/G_act` of the work for `a/|act|` of the coordinates.
* `LinSteerFalse`: so `LinSteer` is false as a premise over consumers: three coordinates in two segments refute it.
* `SegmentSteer`: the bound that does hold: when every noise-active segment carries at least `a` noise-active
  coordinates, zeroing `Z` saves at most `|Z|/a` segments' work.
* `AdaptSteerFail`: under E1(b)'s accounting, where a segment touched by `Z` counts as saved, LIN-STEER fails for every
  row with fewer active segments than active coordinates, by a factor of the mean active count.
* `SteerUnion`: the union bound for steering of any depth under one σ. A cheater picks its wrong values `κ` after σ,
  from a set `K` fixed before it; a segment is zeroed under `κ` when its own fresh noise equals the value `κ` forces on
  it (a function of `κ` and every other coin). If each segment's noise has point mass at most `μ` (HMU's
  `CoinPointMass` gives `μ` for the dither forms, `2^{-32}` for a uniform word), the chance that some `κ` zeroes some
  `n` segments is at most `|K|·C(G, n)·μ^n`.
-/

namespace Accounting.Guarantees.TS

open Accounting.TS Accounting.Assumptions.TS Pouw.RelativeSecurity

/-- **LIN-STEER for a segmented consumer.** For `c > 0`, the segmented consumer satisfies LIN-STEER exactly when every
two segments holding a noise-active coordinate hold equally many. -/
def LinSteerSegments : Prop :=
  ∀ {m G : ℕ} (seg : Fin m → Fin G) (act : Finset (Fin m)) (c : ℝ), 0 < c →
    (LinSteer (segConsumer seg act c) ↔
      ∀ g g', 0 < segCount seg act g → 0 < segCount seg act g' → segCount seg act g = segCount seg act g')

/-- **LIN-STEER is false.** Some segmented consumer violates it. -/
def LinSteerFalse : Prop :=
  ∃ (m G : ℕ) (seg : Fin m → Fin G) (act : Finset (Fin m)), ¬ LinSteer (segConsumer seg act 1)

/-- **The segment bound.** If every segment holding a noise-active coordinate holds at least `a > 0` of them, zeroing
any `Z ⊆ act` saves at most `|Z|/a` segments' work `c`. -/
def SegmentSteer : Prop :=
  ∀ {m G : ℕ} (seg : Fin m → Fin G) (act : Finset (Fin m)) (c : ℝ) (a : ℕ), 0 ≤ c → 0 < a →
    (∀ g, 0 < segCount seg act g → a ≤ segCount seg act g) →
    ∀ Z ⊆ act, (segConsumer seg act c).cost ∅ - (segConsumer seg act c).cost Z ≤ (Z.card : ℝ) / a * c

/-- **LIN-STEER fails under E1(b)'s accounting.** If a segment touched by `Z` counts as saved, LIN-STEER fails for
every row with `c > 0` and fewer noise-active segments than noise-active coordinates. -/
def AdaptSteerFail : Prop :=
  ∀ {m G : ℕ} (seg : Fin m → Fin G) (act : Finset (Fin m)) (c : ℝ), 0 < c →
    (segActive seg act ∅).card < act.card → ¬ LinSteer (adaptConsumer seg act c)

/-- **The steering union bound.** On coins `Ω₀ × (Fin G → B)`, uniform, with segment `g`'s noise `η g` a function of
its own fresh coins and of point mass at most `μ`, and `v κ ω₀ g` the noise that wrong values `κ ∈ K` would need at
segment `g` to zero it (any function of `κ` and of the other coins `ω₀`): the probability that some `κ` zeroes some
`n` segments is at most `|K|·C(G, n)·μ^n`. -/
def SteerUnion : Prop :=
  ∀ {Ω₀ B N K : Type} [Fintype Ω₀] [Fintype B] [Fintype K] {G : ℕ} (η : Fin G → B → N)
    (v : K → Ω₀ → Fin G → N) (μ : ℝ) (n : ℕ), 0 ≤ μ → (∀ g t, prob (fun b : B => η g b = t) ≤ μ) →
    prob (fun ω : Ω₀ × (Fin G → B) => ∃ κ, ∃ S : Finset (Fin G), S.card = n ∧ ∀ g ∈ S, η g (ω.2 g) = v κ ω.1 g) ≤
      Fintype.card K * G.choose n * μ ^ n

end Accounting.Guarantees.TS
