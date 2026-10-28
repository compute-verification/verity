import Definitions.Pouw.Basic.Prob
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.Order.Archimedean.Real.Basic
import Mathlib.Order.Interval.Finset.Nat

/-!
# What a PoUB session shows held: the node's memory as a trace (trusted)

A PoUB session over a registered store of `N` blocks, at each outcome `ω` of the audit's coins `Ω` (uniform,
`Pouw.pr`), seen as a trace of where each registered block is, tick by tick on the verifier's clock:

* `mem ω t`: the registered blocks in node memory (every GPU's HBM and the host's DRAM) at tick `t`;
* `chip ω t`: those on the node's chips (every GPU's on-chip storage and the host's caches), not in node memory;
* a block in neither is absent: on the node's disks, outside the node, or nowhere, and it comes back only as one of
  `rebuilt ω t` (re-encoded from its payload by the node's compute), `disk ω t` (read from the node's disks) or
  `link ω t` (carried in over the node's boundary links), the blocks each route brings in at tick `t`.

The session's credited passes are `p < P`. Pass `p` is `[tick ω p, tick ω (p + 1))`: `tick ω p` is the tick its seed
could first be computed, from its window's challenge and the root before it, and `tick ω (p + 1)` the tick its own root
left, so by the chain its correct receipts are computed inside it. `plan ω p` is the registered blocks its plan holds and
`correct ω p` those whose receipts are correct.

A paced session (`Pace`) groups the passes: group `g`'s passes are `[first ω g, first ω (g + 1))`, its coin left at
`released ω g` (`s_g`) and `ended ω g` is `e_g = max(its last root's arrival, s_(g + 1))`. Groups `g < C` are credited
and group `C` closes them.

What the session shows held, at an outcome:

* `held ω`, the snapshot: `Σ_p |mem ω (tick ω p)|`, the registered blocks in node memory at each credited pass's start;
* `heldOver ω u v`, held over time: `Σ_{t ∈ [u, v)} |mem ω t|`, the registered blocks in node memory at each tick.
-/

namespace Accounting.Poub.Held

open Finset

/-- A PoUB session's registered store as a trace of the node's memory, over coins `Ω`. -/
structure Trace (Ω : Type) where
  /-- `N`, the registered blocks -/
  blocks : ℕ
  /-- `P`, the credited passes -/
  passes : ℕ
  /-- the verifier's verdict -/
  accepts : Ω → Prop
  /-- `t_p`: pass `p` is `[t_p, t_(p+1))` -/
  tick : Ω → ℕ → ℕ
  /-- the registered blocks pass `p`'s plan holds -/
  plan : Ω → ℕ → Finset (Fin blocks)
  /-- the registered blocks of pass `p` whose receipts are correct -/
  correct : Ω → ℕ → Finset (Fin blocks)
  /-- the registered blocks in node memory at tick `t` -/
  mem : Ω → ℕ → Finset (Fin blocks)
  /-- the registered blocks on the node's chips at tick `t` -/
  chip : Ω → ℕ → Finset (Fin blocks)
  /-- the blocks the node's compute re-encodes from their payloads at tick `t` -/
  rebuilt : Ω → ℕ → ℕ
  /-- the blocks read from the node's disks at tick `t` -/
  disk : Ω → ℕ → ℕ
  /-- the blocks carried in over the node's boundary links at tick `t` -/
  link : Ω → ℕ → ℕ

/-- A paced session's groups of passes on the verifier's clock: `C` credited groups and the closing group `C`. -/
structure Pace (Ω : Type) where
  /-- `C`, the credited groups -/
  credited : ℕ
  /-- group `g`'s first pass -/
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
def wrong (ω : Ω) : ℕ := ∑ p ∈ range T.passes, (T.plan ω p \ T.correct ω p).card

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

end Accounting.Poub.Held
