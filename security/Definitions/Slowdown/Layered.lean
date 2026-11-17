module

public import Definitions.Slowdown.Threads

/-! Keyed reference training algorithms, a class whose `(d, e)`-threading follows from their wiring and widths alone. -/

@[expose] public section

namespace Slowdown

open Verity Finset

/-- Following `key` from `a`, through gates outside `stop`, reaches `u`. -/
inductive KeyPath {N : ℕ} (key : Fin N → Option (Fin N)) (stop : Finset (Fin N)) : Fin N → Fin N → Prop
  | refl (a : Fin N) : KeyPath key stop a a
  | step {a k u : Fin N} : a ∉ stop → key a = some k → KeyPath key stop k u → KeyPath key stop a u

/-- **A keyed reference training algorithm**: every computed thread value outside the roots and lookups has a key in its thread;
each root has at least `d` fan-in reads whose key paths are disjoint; at most `e` counted multiplications have an
activation whose key path reaches any one value, and at most `eLk` one that reaches any one lookup; each lookup reads an
input of its thread and a column of fresh inputs at least `b` bits wide, disjoint from the columns of its thread's other
lookups; and no thread value is baseline. -/
structure Layered {c : RegisteredCircuit} (T : Threading c) (d e eLk b : ℕ) where
  /-- The key of a value: one of its reads, in its thread. -/
  key : Fin c.D.N → Option (Fin c.D.N)
  key_reads : ∀ g k, key g = some k → k ∈ c.D.reads g
  key_thread : ∀ g k, key g = some k → T.thread k = T.thread g
  /-- The roots: wide gates whose fan-in reads' key paths are disjoint. -/
  roots : Finset (Fin c.D.N)
  /-- The lookups: gates that read an input of their thread and a column of fresh inputs. -/
  lookups : Finset (Fin c.D.N)
  keyed : ∀ g t, c.D.isInput g = false → T.thread g = some t → g ∉ roots ∪ lookups → (key g).isSome
  thread_θ : ∀ g t, T.thread g = some t → g ∉ c.θ
  /-- A root's fan-in: reads of its thread. -/
  fanin : Fin c.D.N → Finset (Fin c.D.N)
  fanin_reads : ∀ r ∈ roots, fanin r ⊆ c.D.reads r
  fanin_thread : ∀ r ∈ roots, ∀ p ∈ fanin r, T.thread p = T.thread r
  fanin_card : ∀ r ∈ roots, d ≤ (fanin r).card
  fanin_disjoint : ∀ r ∈ roots, ∀ p ∈ fanin r, ∀ q ∈ fanin r, p ≠ q → ∀ u,
    KeyPath key (roots ∪ lookups) p u → ¬ KeyPath key (roots ∪ lookups) q u
  reach : ∀ (u : Fin c.D.N) (X : Finset (Fin c.D.N)), X ⊆ T.M →
    (∀ g ∈ X, KeyPath key (roots ∪ lookups) (T.act g) u) → X.card ≤ e
  reach_lookups : ∀ v ∈ lookups, ∀ X : Finset (Fin c.D.N), X ⊆ T.M →
    (∀ g ∈ X, KeyPath key (roots ∪ lookups) (T.act g) v) → X.card ≤ eLk
  lookup_input : ∀ v ∈ lookups, ∃ k ∈ c.D.reads v, c.D.isInput k = true ∧ T.thread k = T.thread v
  /-- A lookup's column: the fresh inputs it selects from. -/
  col : Fin c.D.N → Finset (Fin c.D.N)
  col_reads : ∀ v ∈ lookups, col v ⊆ c.D.reads v
  col_input : ∀ v ∈ lookups, ∀ g ∈ col v, c.D.isInput g = true ∧ g ∉ c.θ
  col_bits : ∀ v ∈ lookups, b ≤ c.D.width (col v)
  col_disjoint : ∀ v ∈ lookups, ∀ v' ∈ lookups, v ≠ v' → T.thread v = T.thread v' → Disjoint (col v) (col v')

end Slowdown
