module

public import Definitions.Poub.Remote.Held

/-! What devices outside the node compute for it over a PoUB session, and what enters it over its uplinks. -/

@[expose] public section

namespace ResourceAccounting.Remote.Held

/-- What devices outside the node compute for it during a PoUB session whose node memory is the trace `T`. -/
structure Outside {Ω : Type} (T : Trace Ω) where
  /-- the registered blocks a device outside the node computes on for it at tick `t` -/
  blocks : Ω → ℕ → Finset (Fin T.blocks)
  /-- the registered blocks whose receipts a device outside the node computes on for it at tick `t` -/
  receipts : Ω → ℕ → Finset (Fin T.blocks)

/-- What enters the node over its uplinks during a PoUB session, over coins `Ω`. -/
structure Ingress (Ω : Type) where
  /-- the bits other than the verifier's messages that enter the node over its uplinks at tick `t` -/
  entered : Ω → ℕ → ℕ
  /-- the bits other than the verifier's messages that the network attesters on its uplinks sign as entering at
  tick `t` -/
  counted : Ω → ℕ → ℕ

/-- `n` is 0 at every tick of every credited pass of `T` at `ω`. -/
def Quiet {Ω : Type} (T : Trace Ω) (n : Ω → ℕ → ℕ) (ω : Ω) : Prop :=
  ∀ p < T.passes, ∀ t, T.tick ω p ≤ t → t < T.tick ω (p + 1) → n ω t = 0

end ResourceAccounting.Remote.Held
