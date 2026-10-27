import Definitions.Accounting.Remote.Held

/-!
# What devices outside the node compute on, over a remote audit session (trusted)

A remote audit session's trace (`Accounting.Remote.Held.Trace`) says where each registered block is on the node, tick
by tick on the verifier's clock. `Outside T` is the same session's record of the computation off the node, at each
outcome `ω` and tick `t`:

* `blocks ω t`: the registered blocks some device outside the node computes on at tick `t`, on any of their bytes;
* `receipts ω t`: the registered blocks whose receipts some device outside the node computes on at tick `t`: hashes
  them, roots them, or derives a seed from a root over them.

A device outside the node is any hardware the node's declaration doesn't list: another host, a switch, a GPU across a
link. Holding a block and sending it in over a boundary link is not computing on it: the block then comes in by the
trace's `link` route.
-/

namespace Accounting.Remote.Held

/-- What devices outside the node compute on during a remote audit session whose node memory is the trace `T`. -/
structure Outside {Ω : Type} (T : Trace Ω) where
  /-- the registered blocks a device outside the node computes on at tick `t` -/
  blocks : Ω → ℕ → Finset (Fin T.blocks)
  /-- the registered blocks whose receipts a device outside the node computes on at tick `t` -/
  receipts : Ω → ℕ → Finset (Fin T.blocks)

end Accounting.Remote.Held
