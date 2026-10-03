import FlockSoundness.Discharge.Composed.Coins
import FlockSoundness.Discharge.Composed.Geometry
import FlockSoundness.Discharge.Composed.Pads
import FlockSoundness.Discharge.Composed.Inner.Holds

/-!
# `coinedOf_setup`, from the refinement's three facts

`coinedOf` fixes the pinned fields of `CoinedOfSetup` (the statement, `RANK_POINTS`, the mask words) and `CoinedOK`'s
coin ties and padding columns by definition, and `coin_facts` gives the rank check, `W^r` and `β ≠ 0` from `Admits`.
What is left reads `bodyOf`, the refinement, and is stated here as three named facts, the third in its own file at every draw:

* `bodyOf_geometry`: M1's level-0 padding geometry, the opened positions at each coin vector distinct and in range
  (`geometry_of_padOK` from the table's `PadOK` once `bodyOf` takes `cols` and `t_pad` from its level 0 and pads the
  queries to distinct positions).
* `bodyOf_padsOnto`: each rep's pads code reaches every value at its opened positions (`padsOnto_pow` once `bodyOf`'s
  `Cp` is the inner code's padding monomials at distinct nonzero opened points).
* `innerOf_holds` (`Composed/Inner/Holds.lean`, the inner lane's): the honest prover's pads satisfy the inner proof's constraint at every valid witness whose registered
  values are the public ones (`Table.InnerHolds`; `innerOf`'s `lam`, `theta`, `vals`, `lanes`, `lig`).
-/

namespace FlockSoundness.Discharge.Composed

open Game Audit Exec Refine Flock Model ZkSession ZkLink ZkExec DecodeZL FlockLevel3 ZkView

local notation "𝔽" => FlockLevel3.GF128

variable {mPts n : ℕ} {Reg : Type}

/-- **M1's level-0 padding geometry at the draw** (open, refinement). -/
theorem bodyOf_geometry (I : Inputs) (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    (y₀ : DrawSetupZK I mPts S₀ (dj S₀)) (S : Finset (Fin n)) (R : Reg) (y : DrawSetupZK I mPts S (dj S)) :
    let C := coinedOf (tablesAtZK (Reg := Reg) dj y₀ S R) y.x.st
    C.cols ≤ 40 ∧ C.z.tpad ≤ 2 ^ C.cols ∧ C.z.nS ≤ C.z.tpad ∧
      ∀ c, C.Admits c → Function.Injective (C.pos c) ∧ ∀ q, C.pos c q < 2 ^ (C.cols + 1) := by
  sorry

/-- **The pads onto at the draw** (open, refinement). -/
theorem bodyOf_padsOnto (I : Inputs) (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    (y₀ : DrawSetupZK I mPts S₀ (dj S₀)) (S : Finset (Fin n)) (R : Reg) (y : DrawSetupZK I mPts S (dj S)) :
    let C := coinedOf (tablesAtZK (Reg := Reg) dj y₀ S R) y.x.st
    ∀ c, C.Admits c → (C.T c).tab.PadsOnto := by
  sorry

theorem coinedOf_setup (I : Inputs) (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    (y₀ : DrawSetupZK I mPts S₀ (dj S₀)) : CoinedOfSetup (Reg := Reg) I dj y₀ := by
  intro S R y
  obtain ⟨g1, g2, g3, g4⟩ := bodyOf_geometry (Reg := Reg) I dj y₀ S R y
  refine ⟨rfl, rfl, rfl, fun _ => rfl, ⟨(bodyOf _ _).rest0⟩, g1, g2, g3, fun c hc => ?_⟩
  have hβ : ∀ r, ((coinedOf (tablesAtZK (Reg := Reg) dj y₀ S R) y.x.st).T c).β r = fOf (c.beta r) := fun _ => rfl
  have hρ : ∀ r, ((coinedOf (tablesAtZK (Reg := Reg) dj y₀ S R) y.x.st).T c).ρ r 0 = GF256.of (c.rho r).1 ∧
      ((coinedOf (tablesAtZK (Reg := Reg) dj y₀ S R) y.x.st).T c).ρ r 1 = GF256.of (c.rho r).2 :=
    fun _ => ⟨rfl, rfl⟩
  have hw : ∀ k x, ((coinedOf (tablesAtZK (Reg := Reg) dj y₀ S R) y.x.st).T c).eqw k x =
      fOf (eqWord (c.points k) ((coinedOf (tablesAtZK (Reg := Reg) dj y₀ S R) y.x.st).maskWords x)) :=
    fun _ _ => rfl
  have he : ∀ a : F128, ((coinedOf (tablesAtZK (Reg := Reg) dj y₀ S R) y.x.st).T c).embed (fOf a) =
      GF256.of ⟨a, 0⟩ := fun _ => rfl
  obtain ⟨hrank, hW, hb⟩ := coin_facts _ c hc hβ hρ hw he
  exact ⟨g4 c hc, ⟨hβ, hρ, hw, he⟩, hrank, fun _ _ => rfl, hW, bodyOf_padsOnto I dj y₀ S R y c hc, hb,
    innerOf_holds I dj y₀ S R y c hc⟩

end FlockSoundness.Discharge.Composed
