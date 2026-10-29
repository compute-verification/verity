import Definitions.Pouw.Dimension.Adaptive

/-!
# adaptive-dw: the guarantees (trusted; draft model, pending review)

Each statement is over the chunk model (`Pouw.Dimension.Pointwise`, draft) with `Holds` and `AdaptiveDW`
(`Pouw.Dimension.Pointwise`, `Adaptive.lean`). Its proof is the theorem
`Pouw.SecurityProofs.Adaptive.G : Pouw.Guarantees.Adaptive.G`.

* `RunChunks`: a run that holds the target chunk values writes at least the number of them no initial chunk holds.
* `AdaptiveChunks`: the pointwise lift. Over any finite set of coin draws, adaptive-dw holds at the sum of each draw's
  new target chunk values.
* `AdaptiveBall`: the balls' lift. Over any finite set of coin draws, adaptive-dw holds at `Σ_g Σ_y (r y + 1)` when
  each draw's targets are further than `r y` from every initial chunk and pairwise further than the sum of radii.
* `AdaptiveBallTable`: the same without the hypothesis on the initial memory, less `R + 1` a slot a draw: a table
  fixed before the coins buys at most one target's ball a slot.
* `LevelsObstruction`: when each draw's targets are row-step chunks `base i + c i t · 1` (a column-free increment),
  adaptive-dw fails above `w · Σ_g Σ_i |{c i t}|`, whatever the initial memory.
* `OnesObstruction`: when each row's chunk repeats at every step whose coin bit is `false`, adaptive-dw fails above
  `w · Σ_g Σ_i (1 + #true bits)`, whatever the initial memory.
-/

namespace Pouw.Guarantees.Adaptive

open Finset Pouw.Dimension.Pointwise

open Classical in
/-- **A run writes at least its targets' new chunk values** (chunk model; draft model, pending review). -/
def RunChunks : Prop :=
  ∀ (A V : Type) [DecidableEq A] [DecidableEq V] [Inhabited V] (w : ℕ) (m₀ : A → Fin w → V)
    (P : List (Step A V w)) (Y : Finset (Fin w → V)), Holds m₀ P Y →
    (Y.filter fun y => ∀ a, m₀ a ≠ y).card ≤ writes P

open Classical in
/-- **The pointwise lift** (chunk model; draft model, pending review): for every finite set of coin draws `Ω` and
targets `Y g`, every family of runs, the run at `g` holding `Y g`, writes in total at least the draws' new target chunk
values. -/
def AdaptiveChunks : Prop :=
  ∀ (A V : Type) [DecidableEq A] [DecidableEq V] [Inhabited V] (w : ℕ) (Ω : Type) [Fintype Ω]
    (m₀ : A → Fin w → V) (Y : Ω → Finset (Fin w → V)),
    AdaptiveDW Ω m₀ Y (∑ g, ((Y g).filter fun y => ∀ a, m₀ a ≠ y).card)

/-- **The balls' lift** (chunk model; draft model, pending review): when draw `g`'s targets each sit further than
`r g y` in Hamming distance from every initial chunk, and any two further apart than the sum of their radii, every
family of runs, the run at `g` holding `Y g`, writes in total at least `Σ_g Σ_y (r g y + 1)`. -/
def AdaptiveBall : Prop :=
  ∀ (A V : Type) [DecidableEq A] [DecidableEq V] [Inhabited V] (w : ℕ) (Ω : Type) [Fintype Ω]
    (m₀ : A → Fin w → V) (Y : Ω → Finset (Fin w → V)) (r : Ω → (Fin w → V) → ℕ),
    (∀ g, ∀ y ∈ Y g, ∀ a, r g y < hammingDist (m₀ a) y) →
    (∀ g, ∀ y ∈ Y g, ∀ y' ∈ Y g, y ≠ y' → r g y + r g y' < hammingDist y y') →
    AdaptiveDW Ω m₀ Y (∑ g, ∑ y ∈ Y g, (r g y + 1))

/-- **The balls' lift against any initial memory** (chunk model; draft model, pending review): when draw `g`'s
targets are pairwise further apart in Hamming distance than the sum of their radii, each radius at most `R`, every
family of runs, the run at `g` holding `Y g`, writes in total at least `Σ_g Σ_y (r g y + 1)` less `R + 1` for each
slot of memory at each draw, whatever the initial memory holds. -/
def AdaptiveBallTable : Prop :=
  ∀ (A V : Type) [DecidableEq A] [Fintype A] [DecidableEq V] [Inhabited V] (w : ℕ) (Ω : Type) [Fintype Ω]
    (m₀ : A → Fin w → V) (Y : Ω → Finset (Fin w → V)) (r : Ω → (Fin w → V) → ℕ) (R : ℕ),
    (∀ g, ∀ y ∈ Y g, r g y ≤ R) →
    (∀ g, ∀ y ∈ Y g, ∀ y' ∈ Y g, y ≠ y' → r g y + r g y' < hammingDist y y') →
    AdaptiveDW Ω m₀ Y (∑ g, ∑ y ∈ Y g, (r g y + 1) - Fintype.card Ω * (Fintype.card A * (R + 1)))

/-- **Column-free increments refute adaptive-dw** (chunk model; draft model, pending review): when draw `g`'s targets
are the row-step chunks `base g i + c g i t · 1` over rows and steps, adaptive-dw fails at every bound above
`w · Σ_g Σ_i |{c g i t : t}|`, from any initial memory with a slot. -/
def LevelsObstruction : Prop :=
  ∀ (A V : Type) [DecidableEq A] [DecidableEq V] [Add V] (w : ℕ) (ι T : Type) (Ω : Type) [Fintype Ω] [Nonempty A]
    (m₀ : A → Fin w → V) (rows : Finset ι) (steps : Finset T) (base : Ω → ι → Fin w → V) (c : Ω → ι → T → V)
    (B : ℕ), w * ∑ g, ∑ i ∈ rows, (steps.image (c g i)).card < B →
    ¬ AdaptiveDW Ω m₀ (fun g => (rows ×ˢ steps).image fun it => base g it.1 + fun _ => c g it.1 it.2) B

/-- **Chunks that repeat on a `false` bit refute adaptive-dw** (chunk model; draft model, pending review): when draw
`g`'s targets are row-step chunks `F g i t`, `t ≤ k`, and each row's chunk at step `t + 1` equals its chunk at `t`
whenever the bit `b g i t` is `false`, adaptive-dw fails at every bound above `w · Σ_g Σ_i (1 + #{t < k : b g i t})`,
from any initial memory with a slot. -/
def OnesObstruction : Prop :=
  ∀ (A V : Type) [DecidableEq A] [DecidableEq V] (w : ℕ) (ι : Type) (Ω : Type) [Fintype Ω] [Nonempty A]
    (m₀ : A → Fin w → V) (rows : Finset ι) (k : ℕ) (F : Ω → ι → ℕ → Fin w → V) (b : Ω → ι → ℕ → Bool),
    (∀ g i t, t < k → b g i t = false → F g i (t + 1) = F g i t) → ∀ B : ℕ,
    w * ∑ g, ∑ i ∈ rows, (1 + ((range k).filter fun t => b g i t).card) < B →
    ¬ AdaptiveDW Ω m₀ (fun g => (rows ×ˢ range (k + 1)).image fun it => F g it.1 it.2) B

end Pouw.Guarantees.Adaptive
