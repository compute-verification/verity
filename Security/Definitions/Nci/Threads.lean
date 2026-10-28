import Definitions.Nci

/-!
# NCI over threads: the vocabulary

The thread draft ("Distinguishing inference from frontier training via novel communication intensity") bounds
the NCI of every small subcircuit of a reference training circuit `C` from two facts about `C`'s structure: each weight
is used a bounded number of times per thread, and a subcircuit pays an entry fee before it can do much of a thread's work.
It states them over `C` itself, not over the declared circuit, which reaches them through containment.

* `Threading c`: `c`'s counted multiplications (gates that multiply a weight by an activation, where an activation is any
  non-weight value, gradients included) and a thread partition, putting each non-weight value in at most one thread,
  with each weight used at most `reuse` times per thread. A counted multiplication belongs to the thread of its
  activation. In a standard transformer a thread is one token position at one training step, and `reuse = 2`.
* `Small c Γ S R`: the draft's small subcircuit, at most `Γ` computed gates reading at most `S` bits of novel
  information (its `F*` and `S*`).
* `Threading.Threaded Γ S d e`: the draft's `(d, e)`-threaded: every small subcircuit that reads fewer than `d` novel values
  of a thread computes at most `e` of that thread's counted multiplications per value it reads from the thread.
-/

namespace Nci

open Verity Finset

/-- **Counted multiplications and a thread partition** of the reference circuit `c`. -/
structure Threading (c : RegisteredCircuit) where
  /-- The number of threads. -/
  m : ℕ
  /-- The weight gates. -/
  W : Finset (Fin c.D.N)
  /-- The thread of each non-weight value, if it has one. -/
  thread : Fin c.D.N → Option (Fin m)
  thread_W : ∀ g ∈ W, thread g = none
  /-- The counted multiplications. -/
  M : Finset (Fin c.D.N)
  M_computed : M ⊆ c.D.computed
  /-- The weight a counted multiplication reads. -/
  weight : Fin c.D.N → Fin c.D.N
  weight_W : ∀ g ∈ M, weight g ∈ W
  weight_reads : ∀ g ∈ M, weight g ∈ c.D.reads g
  /-- The activation a counted multiplication reads; it has a thread. -/
  act : Fin c.D.N → Fin c.D.N
  act_reads : ∀ g ∈ M, act g ∈ c.D.reads g
  act_thread : ∀ g ∈ M, (thread (act g)).isSome
  /-- The draft's `c`: the most times a weight is used per thread. -/
  reuse : ℕ
  reuse_le : ∀ w t, (M.filter fun g => weight g = w ∧ thread (act g) = some t).card ≤ reuse

/-- **A small subcircuit** (the draft's `F < F*`, `S < S*`, here as `≤`): at most `Γ` computed gates, reading at most
`S` bits of novel information. -/
def Small (c : RegisteredCircuit) (Γ : ℕ) (S : ℝ) (R : Finset (Fin c.D.N)) : Prop :=
  R ⊆ c.D.computed ∧ c.D.size R ≤ Γ ∧ (novelReads c R : ℝ) ≤ S

namespace Threading

variable {c : RegisteredCircuit} (T : Threading c)

/-- The counted multiplications in `R`. -/
def mults (R : Finset (Fin c.D.N)) : Finset (Fin c.D.N) := T.M ∩ R

/-- The counted multiplications in `R` that belong to thread `t`. -/
def threadMults (R : Finset (Fin c.D.N)) (t : Fin T.m) : Finset (Fin c.D.N) :=
  (T.mults R).filter fun g => T.thread (T.act g) = some t

/-- The weights `R` uses: those of which it holds a counted multiplication. -/
def weightsUsed (R : Finset (Fin c.D.N)) : Finset (Fin c.D.N) := (T.mults R).image T.weight

/-- The novel values of thread `t` that `R` reads: the gates of `in(R) \ θ` in thread `t`. -/
def threadReads (R : Finset (Fin c.D.N)) (t : Fin T.m) : Finset (Fin c.D.N) :=
  (c.D.into R \ c.θ).filter fun g => T.thread g = some t

/-- **`(d, e)`-threaded** at `(Γ, S)`: every small subcircuit that reads fewer than `d` novel values of a thread computes
at most `e` of that thread's counted multiplications per value it reads from the thread. At `d = 1, e = 0`: a small
subcircuit does none of a thread's multiplications without reading one of its values. -/
def Threaded (Γ : ℕ) (S : ℝ) (d e : ℕ) : Prop :=
  ∀ R, Small c Γ S R → ∀ t, (T.threadReads R t).card < d →
    (T.threadMults R t).card ≤ e * (T.threadReads R t).card

/-- **The NCI bound of the draft's Theorem 1**, as the most counted multiplications per `b_a` bits read, at `b_w`
bits charged per weight used: `Λ = max(e, c · S / (4 d b_w))`. The NCI floor is `b_a / (2Λ) =
(b_a/2) · min(1/e, 4 d b_w/(cS))`, the draft's at `b_w = b_a = b`. -/
noncomputable def bound (S : ℝ) (b_w : ℝ) (d e : ℕ) : ℝ :=
  max (e : ℝ) (T.reuse * S / (4 * d * b_w))

end Threading

end Nci
