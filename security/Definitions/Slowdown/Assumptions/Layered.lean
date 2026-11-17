module

public import Definitions.Slowdown.Layered

/-! The keyed form of the reference training algorithm, a named conjecture: that it is keyed is checked on paper, not in Lean, because no Lean reference training algorithm of a transformer exists. -/

@[expose] public section

namespace Slowdown.Assumptions

open Verity

/-- **The reference training algorithm is keyed, at Kimi K2's shapes** (conjectured): the reference training circuit `c` is written
in the declarations' primitives, which contain it gate for gate, so each matrix-product entry is a chain of `Dot16`
steps, each reading the previous step's accumulator, 16 activations and 16 weights. `T` counts the `Dot16` steps of the
forward and activation-gradient products of updated weights, each with one weight and one activation designated
round-robin (the backward's offset by 8, so a designated weight is used once per thread). The threading `T` has a keyed
form (`Layered`) whose roots each have fan-in at least `d₀ = 17` (every step of a chain after its first, its fan-in the
previous accumulator and its 16 activations; no step of a weight's product reads more values of its thread), whose
values each reach at most `e = 2328` counted steps (an MoE input value's: the router's 384 products and gate and up of
8 routed experts and the shared one, `9 · 2 · 2048`, 37248 products in 16-wide steps), whose lookups reach none
(`e_lk = 0`), and whose lookups' columns are bf16 embedding columns, `b = 163840 · 16` bits.

`e_lk = 0` takes `T`'s counted steps to leave out layer 0's attention-input projections, the 132 steps (`W_dq`'s 1536
and `W_dkv`'s 576 products, 16 to a step) designating each embedding entry, about 0.05% of the count, besides the LM
head's forward; with them counted a lookup reaches 132, and `ThreadedOfLayered` gives `e + 7 · 132 = 3252` at
`S = 2·10⁷`. Within those primitives the reference training algorithm is keyed by how it is written: a chain's first step keyed to
one of its activations, a residual keyed to its branch, a norm keyed past its shared scale, attention keyed through its
diagonal term, and MoE as one body per token and slot, or as static capacity slots at capacity factor 1. It fails for
a reference training algorithm written otherwise (a residual keyed to its skip path makes a value reach every later layer's
fan-out), at a larger capacity factor (`e = 280 + 256 · ⌈8 · cf⌉`), with FP8 embeddings, whose columns are
`163840 · 8` bits, and wherever a gather reads a computed embedding table, as every training step after a run's first
does when it trains the table: that gather is no lookup, so its thread's first chains key back to the token, where the
fan-in reads of each chain's first root meet (`d₀ = 1`). -/
def ReferenceLayered (c : RegisteredCircuit) (T : Threading c) : Prop :=
  Nonempty (Layered T 17 2328 0 (163840 * 16))

/-- **The read cap is below `E + 1` lookup columns**: `S < (E + 1) b` for lookup columns `b` bits wide, so that an
inference block reading at most `S` bits holds at most `E` of a thread's lookups (`ThreadedOfLayered`). -/
def CapBelowColumns (S : ℝ) (E b : ℕ) : Prop :=
  S < ((E + 1) * b : ℕ)

end Slowdown.Assumptions
