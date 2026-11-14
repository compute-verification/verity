module

public import Definitions.Accounting.Remote.Held
public import Definitions.Accounting.Remote.Sketch

/-!
# The receipts' sketch: toy instances

The session and the traces at which `Accounting.Guarantees.Remote.SketchBoundDeployed`'s witnesses are stated.

* `honest`: one pass of one candidate over a store of one 4096-byte block, at the deployed receipts (`m = 128`,
  `h = 512`, 64 sign rows). Its device holds the block's 128 sectors on chip (`S = 2^15` bits) and tags the block with
  one query, the receipt hash of its index and its sketch. SHA-512's keys at the seed are the outcome's own coins (`Hs`
  the keys, `keys` the identity).
* `honestTrace`: `honest`'s trace, which accepts where the block's receipt is right, counts the block correct and holds
  it in node memory at every tick.
* `acceptAll`: one pass over one block, counted correct and never present, on an audit that accepts every outcome.
-/

@[expose] public section

namespace Accounting.Remote.Sketch.Toy

open Finset Accounting.RateSketch Accounting.Remote.Held

/-- One candidate seed a pass. -/
abbrev G₁ : Fin 1 → ℕ := fun _ => 1

/-- What SHA-512's outputs are read over: the session's keys themselves. -/
abbrev Hs₁ : Type := Keys 1 1 128 512 64 G₁

/-- The honest device: it holds the block's 128 sectors on chip and tags the block, its index and its sketch, with one
query. -/
def device : RateSketch.Prover 1 128 0 512 32768 0 1 256 Sector (SketchSigns 64 128) (Fin 64 → ZMod (2 ^ 32)) where
  held := univ
  held_le := by simp
  chip σ v := .tag (Fin.elim0, 0, sketch σ fun u => v ⟨(0, u), mem_univ _⟩) fun y => .done fun _ => y
  bounded σ v O D := by simp [Chip.linkSteps, Chip.queries]

/-- **The honest session**: one pass of one candidate, running `device`, its key the outcome's own. -/
def honest : Remote.Sketch.Run Unit Hs₁ 1 1 128 512 32768 0 1 64 G₁ where
  device _ _ _ _ := device
  keys _ _ x := x
  kept _ _ := 0

/-- The honest session's trace: it accepts where the block's receipt is right, counts the block correct and holds it
in node memory at every tick. -/
def honestTrace : Trace (Outcome Unit Hs₁ 1 128) where
  blocks := 1
  passes := 1
  accepts ω := (0 : Fin 1) ∈ honest.correctAt ω 0 0
  tick _ p := p
  plan _ _ := univ
  correct _ _ := univ
  mem _ _ := univ
  chip _ _ := ∅
  rebuilt _ _ := 0
  disk _ _ := 0
  link _ _ := 0

/-- The toy: one pass over one block, counted correct and never in node memory or on a chip, on an audit that accepts
every outcome. -/
def acceptAll : Trace Unit where
  blocks := 1
  passes := 1
  accepts _ := True
  tick _ _ := 0
  plan _ _ := univ
  correct _ _ := univ
  mem _ _ := ∅
  chip _ _ := ∅
  rebuilt _ _ := 0
  disk _ _ := 0
  link _ _ := 0

end Accounting.Remote.Sketch.Toy
