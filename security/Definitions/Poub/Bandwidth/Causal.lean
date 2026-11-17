module

public import Definitions.Poub.Bandwidth.Game
public import Definitions.Pouw.Confidence
import Mathlib.Basic.ENNReal.Real

/-! The bandwidth audit on a strategy's runs: the chip's ports and the hypotheses on them (`PassiveMemory`, `HearsChip`,
`OracleLocation`). -/

@[expose] public section

namespace Bandwidth.Causal

open Pous (Bits)
open ResourceAccounting

/-- What a location does during an interval, as `Actual.Run.acts` gives it. -/
abbrev Acts (L : Type*) :=
  Set (ℕ × L × ℕ × ℕ × ℕ) × Set (ℕ × L × ℕ × L × ℕ × ℕ × ℕ) × Set (ℕ × ℕ × L × ℕ × ℕ × ℕ)

/-- What crosses into a location at a tick, as `Actual.Run.crossing` gives it. -/
abbrev Cross (L : Type*) := Set (ℕ × L × ℕ × L × ℕ × ℕ × ℕ)

/-- **Passive memory** at HBM's ports over the window: what they deliver to the chip at each tick `u` is a function `G`
of the strategy's input other than the oracle (the store, or the store and `z` for Mbw) and of what the chip did since
`t₀`, and given what the chip did, `a`, takes at most `2^{wd u a}` values: a reply carries only its values, at the
width the chip's requests fix. -/
def PassiveMemory {L K ε : Type*} {N b h : ℕ} (S : ε × Oracle N b h → Actual.Run L)
    (link : ∀ r, Fin (S r).A.N → Fin (S r).A.N → K) (X : Finset L) (t₀ T : ℕ) (hbmPorts : List (K × L))
    (wd : ℕ → Acts L → ℕ) : Prop :=
  ∃ G : ε → ℕ → Acts L → List (Set (ℕ × L × ℕ × ℕ × ℕ)),
    (∀ E O, ∀ i < T, portSignals S link hbmPorts (E, O) (t₀ + i) =
      G E (t₀ + i) ((S (E, O)).acts ↑X (Set.Ico t₀ (t₀ + i)))) ∧
    ∀ u a, (Set.range fun E => G E u a).encard ≤ ((2 ^ wd u a : ℕ) : ℕ∞)

/-- **HBM hears only the chip** at its ports over the window: what they deliver to the chip at each tick `u` is a
function `G` of the strategy's input other than the oracle and of what the chip did since `t₀` (the capacity form's
passive memory: no width). -/
def HearsChip {L K ε : Type*} {N b h : ℕ} (S : ε × Oracle N b h → Actual.Run L)
    (link : ∀ r, Fin (S r).A.N → Fin (S r).A.N → K) (X : Finset L) (t₀ T : ℕ) (hbmPorts : List (K × L)) : Prop :=
  ∃ G : ε → ℕ → Acts L → List (Set (ℕ × L × ℕ × ℕ × ℕ)),
    ∀ E O, ∀ i < T, portSignals S link hbmPorts (E, O) (t₀ + i) =
      G E (t₀ + i) ((S (E, O)).acts ↑X (Set.Ico t₀ (t₀ + i)))

/-- **The oracle as a location** at its ports over the window: what they deliver to the chip at each tick `u` is a
function of what the chip did since `t₀`, `a`, and of the oracle's answers to the queries `Q u a`. -/
def OracleLocation {L K ε : Type*} {N b h : ℕ} (S : ε × Oracle N b h → Actual.Run L)
    (link : ∀ r, Fin (S r).A.N → Fin (S r).A.N → K) (X : Finset L) (t₀ T : ℕ) (orPorts : List (K × L))
    (Q : ℕ → Acts L → List (Fin N × Bits b)) : Prop :=
  ∃ R : ℕ → Acts L → List (Bits h) → List (Set (ℕ × L × ℕ × ℕ × ℕ)),
    ∀ E O, ∀ i < T, portSignals S link orPorts (E, O) (t₀ + i) =
      R (t₀ + i) ((S (E, O)).acts ↑X (Set.Ico t₀ (t₀ + i)))
        ((Q (t₀ + i) ((S (E, O)).acts ↑X (Set.Ico t₀ (t₀ + i)))).map O)

end Bandwidth.Causal
