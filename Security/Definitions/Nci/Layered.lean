import Definitions.Nci.Threads

/-!
# NCI over threads: keyed reference circuits

A structural class of reference circuits whose `(d, e)`-threading (`Threading.Threaded`) follows from their wiring and
widths alone (`Nci.Guarantees.ThreadedOfLayered`).

Each computed value of a thread has a *key*, one of its reads in the same thread, except at the *stops*:

* *roots*, each reading at least `d` values of its thread (its *legs*) whose key paths never meet: the steps of a
  matrix-product entry's `Dot16` chain after its first, whose legs are the previous step's accumulator and its 16
  activations;
* *lookups*, each reading an input of its thread and a *column* of novel input gates at least `b` bits wide, the
  columns of one thread's lookups disjoint: an embedding row, selected by the thread's token.

Following keys from a value walks back through its thread until it reaches a stop or leaves a subcircuit. A small
subcircuit that holds a root reads `d` values of the root's thread or holds lookups, and one that holds more than `E`
lookups of a thread reads more than `E` columns; so a subcircuit that reads fewer than `d − E` values of a thread holds
none of its roots, and each of that thread's counted multiplications it holds is charged to the value where its
activation's key path leaves the subcircuit, or to one of at most `E` lookups.

* `KeyPath key stop a u`: following `key` from `a`, through gates outside `stop`, reaches `u`.
* `Layered T d e eLk b`: the class, where `e` bounds the counted multiplications whose activations' key paths reach any
  one value and `eLk` those that reach any one lookup.
-/

namespace Nci

open Verity Finset

/-- Following `key` from `a`, through gates outside `stop`, reaches `u`. -/
inductive KeyPath {N : ℕ} (key : Fin N → Option (Fin N)) (stop : Finset (Fin N)) : Fin N → Fin N → Prop
  | refl (a : Fin N) : KeyPath key stop a a
  | step {a k u : Fin N} : a ∉ stop → key a = some k → KeyPath key stop k u → KeyPath key stop a u

/-- **A keyed reference circuit**: every computed thread value outside the roots and lookups has a key in its thread;
each root has at least `d` legs whose key paths are disjoint; at most `e` counted multiplications have an activation
whose key path reaches any one value, and at most `eLk` one that reaches any one lookup; each lookup reads an input of
its thread and a column of novel inputs at least `b` bits wide, disjoint from the columns of its thread's other lookups;
and no thread value is baseline. -/
structure Layered {c : RegisteredCircuit} (T : Threading c) (d e eLk b : ℕ) where
  /-- The key of a value: one of its reads, in its thread. -/
  key : Fin c.D.N → Option (Fin c.D.N)
  key_reads : ∀ g k, key g = some k → k ∈ c.D.reads g
  key_thread : ∀ g k, key g = some k → T.thread k = T.thread g
  /-- The roots: wide gates whose legs' key paths are disjoint. -/
  roots : Finset (Fin c.D.N)
  /-- The lookups: gates that read an input of their thread and a column of novel inputs. -/
  lookups : Finset (Fin c.D.N)
  keyed : ∀ g t, c.D.isInput g = false → T.thread g = some t → g ∉ roots ∪ lookups → (key g).isSome
  thread_θ : ∀ g t, T.thread g = some t → g ∉ c.θ
  /-- A root's legs: reads of its thread. -/
  legs : Fin c.D.N → Finset (Fin c.D.N)
  legs_reads : ∀ r ∈ roots, legs r ⊆ c.D.reads r
  legs_thread : ∀ r ∈ roots, ∀ p ∈ legs r, T.thread p = T.thread r
  legs_card : ∀ r ∈ roots, d ≤ (legs r).card
  legs_disjoint : ∀ r ∈ roots, ∀ p ∈ legs r, ∀ q ∈ legs r, p ≠ q → ∀ u,
    KeyPath key (roots ∪ lookups) p u → ¬ KeyPath key (roots ∪ lookups) q u
  reach : ∀ (u : Fin c.D.N) (X : Finset (Fin c.D.N)), X ⊆ T.M →
    (∀ g ∈ X, KeyPath key (roots ∪ lookups) (T.act g) u) → X.card ≤ e
  reach_lookups : ∀ v ∈ lookups, ∀ X : Finset (Fin c.D.N), X ⊆ T.M →
    (∀ g ∈ X, KeyPath key (roots ∪ lookups) (T.act g) v) → X.card ≤ eLk
  lookup_input : ∀ v ∈ lookups, ∃ k ∈ c.D.reads v, c.D.isInput k = true ∧ T.thread k = T.thread v
  /-- A lookup's column: the novel inputs it selects from. -/
  col : Fin c.D.N → Finset (Fin c.D.N)
  col_reads : ∀ v ∈ lookups, col v ⊆ c.D.reads v
  col_input : ∀ v ∈ lookups, ∀ g ∈ col v, c.D.isInput g = true ∧ g ∉ c.θ
  col_bits : ∀ v ∈ lookups, b ≤ c.D.width (col v)
  col_disjoint : ∀ v ∈ lookups, ∀ v' ∈ lookups, v ≠ v' → T.thread v = T.thread v' → Disjoint (col v) (col v')

end Nci
