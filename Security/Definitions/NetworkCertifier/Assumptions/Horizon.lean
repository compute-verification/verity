import Definitions.NetworkCertifier.SampledAudit

/-!
# A horizon's named assumptions: each window's draw, and each window's soundness

`Guarantees.HorizonWrongUnitsLe` bounds the wrong units of a horizon of windows, each audited on its own live coins. It
takes two named assumptions per window. One is the draw's law: a uniform `k`-subset of the window's `n` units
(`DrawUniform`), the law A3 gives the verifier's draw (`Flock.Draw`). The other is the window's soundness: its audit
accepts while its draw meets a unit wrong at its registration with probability at most `ε` (`WindowSound`), the bound
the audit's knowledge soundness gives at its deployment.
-/

namespace NetTiming

open Claims Finset

/-- **The window's draw is a uniform `k`-subset of its `n` units** (`uniform-draw`): every coin draws `k` units, and
each `k`-subset is drawn by the same share `1 / C(n, k)` of the coins. -/
def DrawUniform {Ω : Type} [Fintype Ω] {n : ℕ} (draw : Ω → Finset (Fin n)) (k : ℕ) : Prop :=
  (∀ x, (draw x).card = k) ∧
    ∀ S : Finset (Fin n), S.card = k → (univ.filter fun x => draw x = S).card * n.choose k = Fintype.card Ω

/-- **The window's accepted audit missed its wrong units** (`window-sound`): over the horizon's coins `ω`, window `i`'s
audit accepts (`acc`) while its draw at its own coins `ω i` meets the units wrong at its registration (`W ω`) with
probability at most `ε`. -/
def WindowSound {m : ℕ} {Ω : Type} [Fintype Ω] {n : ℕ} (acc : (Fin m → Ω) → Prop) (draw : Ω → Finset (Fin n))
    (W : (Fin m → Ω) → Finset (Fin n)) (i : Fin m) (ε : ℝ) : Prop :=
  Pr (fun ω => acc ω ∧ ¬Disjoint (draw (ω i)) (W ω)) ≤ ε

end NetTiming
