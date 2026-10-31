import Definitions.Nci

/-!
# The inference-only policy over threads: the vocabulary

The thread draft ("Distinguishing inference from frontier training via novel communication intensity") bounds
the fresh-input rate of every small subcircuit of a reference training circuit `C` from two facts about `C`'s structure:
each weight is used a bounded number of times per thread, and a subcircuit pays an entry fee before it can do much of a
thread's work. It states them over `C` itself, not over the declared circuit, which reaches them through containment.

* `Threading C`: `C`'s counted multiplications (gates that multiply a weight by an activation, where an activation is
  any non-weight value, gradients included) and a thread partition, putting each non-weight value in at most one thread,
  with each weight used at most `reuse` times per thread. A counted multiplication belongs to the thread of its
  activation. In a standard transformer a thread is one token position at one training step, and `reuse = 2`.
* `Small C c k R`: the draft's small subcircuit, at most `c` computed gates reading at most `k` fresh input
  bits (its `F*` and `S*`). Under the policy at `(c, k)` every inference block is small.
* `Threading.Threaded c k d e`: the draft's `(d, e)`-threaded: every small subcircuit that reads fewer than `d` fresh
  input values of a thread computes at most `e` of that thread's counted multiplications per value it reads from the
  thread.
-/

namespace Nci

open Verity Finset

/-- **Counted multiplications and a thread partition** of the reference circuit `C`. -/
structure Threading (C : RegisteredCircuit) where
  /-- The number of threads. -/
  m : ℕ
  /-- The weight gates. -/
  W : Finset (Fin C.D.N)
  /-- The thread of each non-weight value, if it has one. -/
  thread : Fin C.D.N → Option (Fin m)
  thread_W : ∀ g ∈ W, thread g = none
  /-- The counted multiplications. -/
  M : Finset (Fin C.D.N)
  M_computed : M ⊆ C.D.computed
  /-- The weight a counted multiplication reads. -/
  weight : Fin C.D.N → Fin C.D.N
  weight_W : ∀ g ∈ M, weight g ∈ W
  weight_reads : ∀ g ∈ M, weight g ∈ C.D.reads g
  /-- The activation a counted multiplication reads; it has a thread. -/
  act : Fin C.D.N → Fin C.D.N
  act_reads : ∀ g ∈ M, act g ∈ C.D.reads g
  act_thread : ∀ g ∈ M, (thread (act g)).isSome
  /-- The most times a weight is used per thread. -/
  reuse : ℕ
  reuse_le : ∀ w t, (M.filter fun g => weight g = w ∧ thread (act g) = some t).card ≤ reuse

/-- **A small subcircuit** (the draft's `F < F*`, `S < S*`, here as `≤`): at most `c` computed gates, reading at most
`k` fresh input bits. -/
def Small (C : RegisteredCircuit) (c : ℕ) (k : ℝ) (R : Finset (Fin C.D.N)) : Prop :=
  R ⊆ C.D.computed ∧ C.D.size R ≤ c ∧ (novelReads C R : ℝ) ≤ k

namespace Threading

variable {C : RegisteredCircuit} (T : Threading C)

/-- The counted multiplications in `R`. -/
def mults (R : Finset (Fin C.D.N)) : Finset (Fin C.D.N) := T.M ∩ R

/-- The counted multiplications in `R` that belong to thread `t`. -/
def threadMults (R : Finset (Fin C.D.N)) (t : Fin T.m) : Finset (Fin C.D.N) :=
  (T.mults R).filter fun g => T.thread (T.act g) = some t

/-- The weights `R` uses: those of which it holds a counted multiplication. -/
def weightsUsed (R : Finset (Fin C.D.N)) : Finset (Fin C.D.N) := (T.mults R).image T.weight

/-- The fresh input values of thread `t` that `R` reads: the gates of `in(R) \ θ` in thread `t`. -/
def threadReads (R : Finset (Fin C.D.N)) (t : Fin T.m) : Finset (Fin C.D.N) :=
  (C.D.into R \ C.θ).filter fun g => T.thread g = some t

/-- **`(d, e)`-threaded** at `(c, k)`: every small subcircuit that reads fewer than `d` fresh input values of a thread
computes at most `e` of that thread's counted multiplications per value it reads from the thread. At `d = 1, e = 0`: a
small subcircuit does none of a thread's multiplications without reading one of its values. -/
def Threaded (c : ℕ) (k : ℝ) (d e : ℕ) : Prop :=
  ∀ R, Small C c k R → ∀ t, (T.threadReads R t).card < d →
    (T.threadMults R t).card ≤ e * (T.threadReads R t).card

/-- **The bound of the draft's Theorem 1**, as the most counted multiplications per `b_a` bits read by a subcircuit
reading at most `k` bits, at `b_w` bits charged per weight used: `Λ = max(e, reuse · k / (4 d b_w))`. The fresh-input
floor is `b_a / (2Λ) = (b_a/2) · min(1/e, 4 d b_w/(reuse · k))`, the draft's at `b_w = b_a = b`. -/
noncomputable def bound (k : ℝ) (b_w : ℝ) (d e : ℕ) : ℝ :=
  max (e : ℝ) (T.reuse * k / (4 * d * b_w))

end Threading

end Nci
