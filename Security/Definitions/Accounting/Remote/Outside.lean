import Definitions.Accounting.Remote.Held

/-!
# What devices outside the node compute for it, and what enters it, over a remote audit session (trusted)

A remote audit session's trace (`Accounting.Remote.Held.Trace`) says where each registered block is on the node, tick
by tick on the verifier's clock. `Outside T` is the same session's record of the computation off the node for it, at
each outcome `ω` and tick `t`:

* `blocks ω t`: the registered blocks some device outside the node computes on at tick `t`, on any of their bytes,
  for the node;
* `receipts ω t`: the registered blocks whose receipts some device outside the node computes on at tick `t` for the
  node: hashes them, roots them, or derives a seed from a root over them.

A device outside the node is any hardware the node's declaration doesn't list: another host, a switch, a GPU across a
link. It computes for the node when what it computes, or anything computed from it, comes into the node while the
session's credited passes run: the receipts the verifier takes are the node's, over the session's channel, so a
computation off the node reaches them only by coming in. A device computing on a block whose result comes in after the
passes, or never, computes nothing the passes' receipts read. Holding a block and sending it in over a boundary link
is not computing on it: the block then comes in by the trace's `link` route.

`Ingress Ω` is what enters the node over its uplinks, the links between it and any other host, at each outcome and
tick: `entered ω t`, the bits other than the verifier's messages that enter it at tick `t`, and `counted ω t`, those
the network certifiers on its uplinks sign as entering it then. `Quiet T n ω` says `n` is 0 at every tick of every
credited pass of `T` at `ω`: at `n = counted` it is the event the certifiers observe (`auditor_line.py`'s boundary
check: each uplink's certified ingress over the span is the verifier's own frames).
-/

namespace Accounting.Remote.Held

/-- What devices outside the node compute for it during a remote audit session whose node memory is the trace `T`. -/
structure Outside {Ω : Type} (T : Trace Ω) where
  /-- the registered blocks a device outside the node computes on for it at tick `t` -/
  blocks : Ω → ℕ → Finset (Fin T.blocks)
  /-- the registered blocks whose receipts a device outside the node computes on for it at tick `t` -/
  receipts : Ω → ℕ → Finset (Fin T.blocks)

/-- What enters the node over its uplinks during a remote audit session, over coins `Ω`. -/
structure Ingress (Ω : Type) where
  /-- the bits other than the verifier's messages that enter the node over its uplinks at tick `t` -/
  entered : Ω → ℕ → ℕ
  /-- the bits other than the verifier's messages that the network certifiers on its uplinks sign as entering at
  tick `t` -/
  counted : Ω → ℕ → ℕ

/-- `n` is 0 at every tick of every credited pass of `T` at `ω`. -/
def Quiet {Ω : Type} (T : Trace Ω) (n : Ω → ℕ → ℕ) (ω : Ω) : Prop :=
  ∀ p < T.passes, ∀ t, T.tick ω p ≤ t → t < T.tick ω (p + 1) → n ω t = 0

end Accounting.Remote.Held
