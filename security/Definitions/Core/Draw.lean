module

public import Mathlib.Data.Fintype.Pi
public import Definitions.Core.Game
import Mathlib.Algebra.Order.Module.Field
import Mathlib.Basic.ENNReal.Real

/-!
# Samples: which units the verifier's coins pick

Every area reads these, and defines none of its own:

* `Law U`: a sampling design over the units `U`, the verifier's uniform coins `Ω` and the unit set each coin samples. What a sampling design
  gives a guarantee is its escape: `escape L B`, the probability that a sample misses every unit of `B`; `miss L K`, the
  largest escape over sets of at least `K` units, the sampling term of a count curve `δ(K) = miss K + ε`; and
  `incl L u`, the probability that `u` is sampled;
* `Draw s N`: `s` indices into `N` units, each uniform and independent of the others (`Draw.law`, its sampling design);
* `Draw.Subset N k`: `k` of `N` units, without replacement, each `k`-subset one coin (the `subset:K` sampling design);
* `Coins Pub s N`: a window's coins, public randomness `Pub` and an index sample.

The probabilities are the uniform ones over one coin (`Game.prCoin`).
-/

@[expose] public section

namespace Verity

open scoped ENNReal

/-- A sampling design over the units `U`: the verifier's uniform coins, and the unit set each samples. -/
structure Law (U : Type) where
  Ω : Type
  [fintype : Fintype Ω]
  [nonempty : Nonempty Ω]
  draw : Ω → Finset U

attribute [instance] Law.fintype Law.nonempty

namespace Law

variable {U : Type}

/-- The probability that a sample misses every unit of `B`. -/
noncomputable def escape (L : Law U) (B : Finset U) : ℝ≥0∞ :=
  Game.prCoin fun ω => Disjoint (L.draw ω) B

/-- The count curve's sampling term: the largest escape over sets of at least `K` units. -/
noncomputable def miss (L : Law U) (K : ℕ) : ℝ≥0∞ :=
  ⨆ (B : Finset U) (_ : K ≤ B.card), L.escape B

/-- The inclusion probability of a unit. -/
noncomputable def incl (L : Law U) (u : U) : ℝ≥0∞ :=
  Game.prCoin fun ω => u ∈ L.draw ω

end Law

/-- **An index sample**: `s` indices into `N` units, each uniform and independent of the others. -/
abbrev Draw (s N : ℕ) : Type := Fin s → Fin N

/-- The sampling design of an index sample over `N > 0` units: the units its indices hit. -/
def Draw.law (s N : ℕ) [NeZero N] : Law (Fin N) where
  Ω := Draw s N
  nonempty := ⟨fun _ => ⟨0, Nat.pos_of_ne_zero (NeZero.ne N)⟩⟩
  draw ω := Finset.univ.image ω

/-- **A subset sample**: `k` of `N` units, without replacement, each `k`-subset one coin (the `subset:K` sampling
design). -/
abbrev Draw.Subset (N k : ℕ) : Type := {s : Finset (Fin N) // s.card = k}

/-- **A window's coins**: the public randomness `Pub` the window's checks read, and an index sample of `s` from `N`. -/
abbrev Coins (Pub : Type) (s N : ℕ) : Type := Pub × Draw s N

end Verity
