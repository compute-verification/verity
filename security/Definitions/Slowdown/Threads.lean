module

public import Definitions.InferenceOnly

/-! The vocabulary of the inference-only policy over threads, stated over the reference training circuit itself, which the declared circuit reaches through containment. -/

@[expose] public section

open InferenceOnly

namespace Slowdown

open Verity Finset

/-- **Counted multiplications and a thread partition** of the reference training algorithm `C`. -/
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
  R ⊆ C.D.computed ∧ C.D.size R ≤ c ∧ (freshInput C R : ℝ) ≤ k

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

end Slowdown
