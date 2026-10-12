import Definitions.Accounting.Bandwidth.Game
import Definitions.Accounting.Confidence

/-!
# The bandwidth audit on a strategy's runs: the chip's ports (trusted)

The vocabulary of B3 and Mbw on the runs of a strategy (`note:console/20261012T0645Z-draft-actual-circuit-model`, v2.8:
section 5's Mbw and the rows B3 and Mbw). A strategy here maps the store (for Mbw, the store and `z`) and the oracle to a
run, `S : ε × Oracle N b h → Actual.Run L`; the chip is the location `X` and the window is `[t₀, t₀ + T)`. What the chip
did since `t₀` is `Actual.Run.acts` over `X` (`Acts`), and what reaches it at a tick is what its ports carry
(`portSignals`: each port a link and the leaf of `X` it delivers to).

Three hypotheses on the chip's ports, each over the window, each a constraint of the model's table that the guarantees
take as a hypothesis:
- `PassiveMemory` (the model's passive memory): HBM's ports deliver a function of the strategy's input other than the
  oracle and of what the chip did, and given what the chip did, one of at most `2^{wd}` signals: a read carries only its
  values, at the width `wd` the chip's requests fix. HBM computes no tag and signals nothing through when it delivers.
- `HearsChip` (the capacity form's): HBM's ports deliver a function of the strategy's input other than the oracle and of
  what the chip did, with no bound on how many signals; with HBM's links as alphabets that is passive memory at their
  capacities summed.
- `OracleLocation`: the oracle's ports deliver a function of what the chip did and of the oracle's answers to the
  queries `Q` that fixes, the oracle being a location the chip queries and nothing else reads.
-/

namespace Bandwidth.Causal

open Pous (Bits)
open Accounting

/-- What a location does during an interval, as `Actual.Run.acts` gives it. -/
abbrev Acts (L : Type*) := Set (ℕ × L × ℕ × ℕ) × Set (ℕ × L × ℕ × L × ℕ × ℕ) × Set (ℕ × ℕ × L × ℕ × ℕ)

/-- What crosses into a location at a tick, as `Actual.Run.crossing` gives it. -/
abbrev Cross (L : Type*) := Set (ℕ × L × ℕ × L × ℕ × ℕ)

/-- **Passive memory** at HBM's ports over the window: what they deliver to the chip at each tick `u` is a function `G`
of the strategy's input other than the oracle (the store, or the store and `z` for Mbw) and of what the chip did since
`t₀`, and given what the chip did, `a`, takes at most `2^{wd u a}` values: a reply carries only its values, at the
width the chip's requests fix. -/
def PassiveMemory {L K ε : Type*} {N b h : ℕ} (S : ε × Oracle N b h → Actual.Run L)
    (link : ∀ r, Fin (S r).A.N → Fin (S r).A.N → K) (X : Finset L) (t₀ T : ℕ) (hbmPorts : List (K × L))
    (wd : ℕ → Acts L → ℕ) : Prop :=
  ∃ G : ε → ℕ → Acts L → List (Set (ℕ × L × ℕ × ℕ)),
    (∀ E O, ∀ i < T, portSignals S link hbmPorts (E, O) (t₀ + i) =
      G E (t₀ + i) ((S (E, O)).acts ↑X (Set.Ico t₀ (t₀ + i)))) ∧
    ∀ u a, (Set.range fun E => G E u a).encard ≤ ((2 ^ wd u a : ℕ) : ℕ∞)

/-- **HBM hears only the chip** at its ports over the window: what they deliver to the chip at each tick `u` is a
function `G` of the strategy's input other than the oracle and of what the chip did since `t₀` (the capacity form's
passive memory: no width). -/
def HearsChip {L K ε : Type*} {N b h : ℕ} (S : ε × Oracle N b h → Actual.Run L)
    (link : ∀ r, Fin (S r).A.N → Fin (S r).A.N → K) (X : Finset L) (t₀ T : ℕ) (hbmPorts : List (K × L)) : Prop :=
  ∃ G : ε → ℕ → Acts L → List (Set (ℕ × L × ℕ × ℕ)),
    ∀ E O, ∀ i < T, portSignals S link hbmPorts (E, O) (t₀ + i) =
      G E (t₀ + i) ((S (E, O)).acts ↑X (Set.Ico t₀ (t₀ + i)))

/-- **The oracle as a location** at its ports over the window: what they deliver to the chip at each tick `u` is a
function of what the chip did since `t₀`, `a`, and of the oracle's answers to the queries `Q u a`. -/
def OracleLocation {L K ε : Type*} {N b h : ℕ} (S : ε × Oracle N b h → Actual.Run L)
    (link : ∀ r, Fin (S r).A.N → Fin (S r).A.N → K) (X : Finset L) (t₀ T : ℕ) (orPorts : List (K × L))
    (Q : ℕ → Acts L → List (Fin N × Bits b)) : Prop :=
  ∃ R : ℕ → Acts L → List (Bits h) → List (Set (ℕ × L × ℕ × ℕ)),
    ∀ E O, ∀ i < T, portSignals S link orPorts (E, O) (t₀ + i) =
      R (t₀ + i) ((S (E, O)).acts ↑X (Set.Ico t₀ (t₀ + i)))
        ((Q (t₀ + i) ((S (E, O)).acts ↑X (Set.Ico t₀ (t₀ + i)))).map O)

end Bandwidth.Causal
