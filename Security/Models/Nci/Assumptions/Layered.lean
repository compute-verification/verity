import Definitions.Nci.Layered

/-!
# NCI over threads: the reference circuit is keyed

`Nci.Guarantees.ThreadedOfLayered` derives the thread draft's threading premise from a keyed form of the reference
circuit (`Layered`). That the reference training circuit is keyed, and at which parameters, is checked against a
transformer block at Kimi K2's shapes on paper (`note:training-20261019T0446Z-threading-layered`), not in Lean: no Lean
reference circuit of a transformer exists to build a `Layered` on. Until one does, it is a named conjecture, stated over
a reference circuit `c` and its threading `T` as `WeightsIncompressible` is.
-/

namespace Nci.Assumptions

open Verity

/-- **The reference circuit is keyed, at Kimi K2's shapes** (conjectured): the threading `T` of the reference training
circuit `c` has a keyed form (`Layered`) whose roots each have at least `d₀ = 512` legs (the narrowest root is a `W_ukv`
output, over the `kv_lora = 512` latent; every other root is at least 1536 wide), whose values each reach at most
`e = 37248` counted multiplications (an MoE input value's: the router's 384, and gate and up of 8 routed experts and
the shared one, `9 · 2 · 2048`), whose lookups reach none (`e_lk = 0`), and whose lookups' columns are bf16 embedding
columns, `b = 163840 · 16` bits.

`e_lk = 0` takes `T`'s counted multiplications to leave out layer 0's attention-input projections, the 2112 products
(`W_dq`'s 1536 and `W_dkv`'s 576) of each embedding entry, about 0.05% of the count, besides the LM head's forward;
with them counted a lookup reaches 2112, and `ThreadedOfLayered` gives `e + 7 · 2112 = 52032` at `S = 2·10⁷`. The
reference circuit is ours to write, and it is keyed by how it is written: each matrix-product entry one wide sum of its
products, a residual keyed to its branch, a norm keyed past its shared scale, attention keyed through its diagonal term,
and MoE as static capacity slots at capacity factor 1. It fails for a reference circuit written otherwise (a residual
keyed to its skip path makes a value reach every later layer's fan-out), at a larger capacity factor
(`e = 4480 + 4096 · ⌈8 · cf⌉`), and with FP8 embeddings, whose columns are `163840 · 8` bits. -/
def ReferenceLayered (c : Declaration) (T : Threading c) : Prop :=
  Nonempty (Layered T 512 37248 0 (163840 * 16))

end Nci.Assumptions
