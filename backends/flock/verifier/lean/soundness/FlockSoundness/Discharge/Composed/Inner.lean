import FlockSoundness.Discharge.ZkLink
import FlockSoundness.Discharge.ZkExec
import FlockSoundness.Discharge.ZkView
import FlockSoundness.Discharge.ZkBind

/-!
# The inner proof's fields of a draw's `--zk` table

`coinedOf`'s tables take the fields the inner proof's constraint `⟨λ, h⟩ = θ` reads (`Table.InnerHolds`) from `innerOf`:
the level-0 lanes, the clear values, Ligerito's value lists and the verifier's replay `λ`, `θ`. `innerOf` takes the
hidden-value and value-list types, the sizes and the coin type from `bodyOf`; `innerOf_holds`
(`Composed/Inner/Holds.lean`) is its obligation.
-/

namespace FlockSoundness.Discharge.Composed

open Flock Model ZkSession ZkView ZK

local notation "𝔽" => FlockLevel3.GF128

/-- **The inner proof's fields** of a `Coined` table over the witnesses `W` at the sizes `z` (`Coined`'s `lanes`, `vals`,
`lig`, `lam`, `theta`). -/
structure InnerFields (W Hi VL : Type) (z : Sizes) where
  lanes : W → (Fin z.nMask → Fin 128 → ZMod 2) → Fin z.nL → Fin z.L → 𝔽
  vals : Fin 2 → W → (Fin z.nMask → Fin 128 → ZMod 2) → (Fin 2 → Fin 2 → Fin z.L → 𝔽) → Fin z.nE → 𝔽
  lig : Fin 2 → (Fin z.L → FlockLevel3.GF256) → Hi → VL
  lam : Fin 2 → (Fin z.nC → Fin 128 → 𝔽) × (Fin 2 → Fin z.nE → 𝔽) × (Fin 2 → Fin z.L → FlockLevel3.GF256) ×
    (Fin 2 → Fin z.tpad → FlockLevel3.GF256) × (Fin 2 → 𝔽 × 𝔽) × (Fin 2 → VL) → PadIdx z.nE → 𝔽
  theta : Fin 2 → (Fin z.nC → Fin 128 → 𝔽) × (Fin 2 → Fin z.nE → 𝔽) × (Fin 2 → Fin z.L → FlockLevel3.GF256) ×
    (Fin 2 → Fin z.tpad → FlockLevel3.GF256) × (Fin 2 → 𝔽 × 𝔽) × (Fin 2 → VL) → 𝔽

/-- **The draw's inner-proof fields** (open: the inner lane's to write, from `verify --zk`'s inner proof at the draw:
its statement `Tz.S`, the executable statement `st`), at `bodyOf`'s types `Hi`, `VL`, sizes `z` and coin vector `c`. -/
noncomputable def innerOf {mPts : ℕ} (Tz : TabZK 𝔽 mPts) (st : Stmt) (Hi VL : Type) [AddCommGroup Hi] [Fintype Hi]
    [DecidableEq Hi] (z : Sizes) (Coins : Type) (c : Coins) : InnerFields (Witness Tz.S.m) Hi VL z := sorry

end FlockSoundness.Discharge.Composed
