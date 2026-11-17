module

public import Definitions.Pouw.Basic.Prob
public import Mathlib.Algebra.Order.Archimedean.Real.Basic
public import Mathlib.Order.Interval.Finset.Nat
public import Mathlib.Algebra.BigOperators.Group.Finset.Defs

/-! A PoUB session's view of the node's memory, as a trace of where each registered block is, tick by tick. -/

@[expose] public section

namespace ResourceAccounting.Remote.Held

open Finset

/-- A PoUB session's registered store as a trace of the node's memory, over coins `Ω`. -/
structure Trace (Ω : Type) where
  blocks : ℕ
  passes : ℕ
  accepts : Ω → Prop
  /-- `t_p`: pass `p` is `[t_p, t_(p+1))` -/
  tick : Ω → ℕ → ℕ
  plan : Ω → ℕ → Finset (Fin blocks)
  correct : Ω → ℕ → Finset (Fin blocks)
  mem : Ω → ℕ → Finset (Fin blocks)
  /-- the registered blocks on the node's chips at tick `t` -/
  chip : Ω → ℕ → Finset (Fin blocks)
  /-- the blocks the node's compute re-encodes from their payloads at tick `t` -/
  rebuilt : Ω → ℕ → ℕ
  disk : Ω → ℕ → ℕ
  /-- the blocks carried in over the node's boundary links at tick `t` -/
  link : Ω → ℕ → ℕ

/-- A paced session's groups of passes on the verifier's clock: `C` credited groups and the closing group `C`. -/
structure Pace (Ω : Type) where
  credited : ℕ
  first : Ω → ℕ → ℕ
  /-- `s_g`: group `g`'s coin's release -/
  released : Ω → ℕ → ℕ
  /-- `e_g = max(group g's last root's arrival, s_(g+1))` -/
  ended : Ω → ℕ → ℕ

/-- What a route (`Trace.rebuilt`, `Trace.disk`, `Trace.link`) brings in over `[u, v)`. -/
def over {Ω : Type} (f : Ω → ℕ → ℕ) (ω : Ω) (u v : ℕ) : ℕ := ∑ t ∈ Ico u v, f ω t

namespace Trace

variable {Ω : Type} (T : Trace Ω)

/-- The registered blocks in node memory or on a chip at tick `t`. -/
def present (ω : Ω) (t : ℕ) : Finset (Fin T.blocks) := T.mem ω t ∪ T.chip ω t

/-- The blocks that came into node memory or onto a chip at tick `t`, from neither. -/
def entered (ω : Ω) (t : ℕ) : ℕ := (T.present ω (t + 1) \ T.present ω t).card

/-- The credited passes' registered blocks, summed over the passes. -/
def planned (ω : Ω) : ℕ := ∑ p ∈ range T.passes, (T.plan ω p).card

/-- The credited passes' wrong receipts of registered blocks. -/
def incorrect (ω : Ω) : ℕ := ∑ p ∈ range T.passes, (T.plan ω p \ T.correct ω p).card

/-- The registered blocks pass `p` read on the node: those of its plan whose receipts are correct, each in node memory
or on a chip at a tick of the pass. -/
def read (ω : Ω) (p : ℕ) : Finset (Fin T.blocks) :=
  T.plan ω p ∩ T.correct ω p ∩ (Ico (T.tick ω p) (T.tick ω (p + 1))).biUnion (T.present ω)

/-- The registered blocks the first `P` passes read on the node, summed over the passes. -/
def reads (ω : Ω) (P : ℕ) : ℕ := ∑ p ∈ range P, (T.read ω p).card

/-- The snapshot: the registered blocks in node memory at each credited pass's start, summed over the passes. -/
def held (ω : Ω) : ℕ := ∑ p ∈ range T.passes, (T.mem ω (T.tick ω p)).card

/-- Held over time: the registered blocks in node memory at each tick of `[u, v)`, summed over the ticks. -/
def heldOver (ω : Ω) (u v : ℕ) : ℕ := ∑ t ∈ Ico u v, (T.mem ω t).card

variable [Fintype Ω]

/-- The trace meets `P` except with probability at most `δ` over the coins on which the audit accepts. -/
def Holds (δ : ℝ) (P : Ω → Prop) : Prop := Pouw.pr (fun ω => T.accepts ω ∧ ¬ P ω) ≤ δ

end Trace

namespace Pace

variable {Ω : Type} (K : Pace Ω)

/-- `n_g`, group `g`'s passes. -/
def size (ω : Ω) (g : ℕ) : ℕ := K.first ω (g + 1) - K.first ω g

/-- `L`, the longest `e_g − s_g` over the credited groups and the closing one. -/
def longest (ω : Ω) : ℕ := (range (K.credited + 1)).sup fun g => K.ended ω g - K.released ω g

/-- The credited span `[e_0, s_C)`: from the first group's end (before its reads a block may never have been there) to
the closing group's coin, whose reads back it. -/
def start (ω : Ω) : ℕ := K.ended ω 0

/-- The credited span's end, `s_C`. -/
def stop (ω : Ω) : ℕ := K.released ω K.credited

/-- `v_g`, the ticks of absence a wrong receipt of group `g` can account for: `(s_(g+1) − s_g)/n_g` in a credited group,
`L/n_C` in the closing one. -/
noncomputable def rate (ω : Ω) (g : ℕ) : ℝ :=
  if g < K.credited then ((K.released ω (g + 1) - K.released ω g : ℕ) : ℝ) / K.size ω g
  else (K.longest ω : ℝ) / K.size ω g

/-- The most absence `W` wrong receipts can account for, spread over the groups `g ≤ C` at their rates `v_g`, at most
`n_g · N` in group `g` (a pass has at most `N` registered blocks): the fractional knapsack's value, which taking the
groups highest rate first attains. -/
noncomputable def fill (ω : Ω) (N : ℕ) (W : ℝ) : ℝ :=
  sSup ((fun x : ℕ → ℝ => ∑ g ∈ range (K.credited + 1), K.rate ω g * x g) ''
    {x | (∀ g, 0 ≤ x g ∧ x g ≤ (K.size ω g * N : ℕ)) ∧ ∑ g ∈ range (K.credited + 1), x g ≤ W})

end Pace

end ResourceAccounting.Remote.Held
