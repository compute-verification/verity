import Mathlib.LinearAlgebra.Vandermonde
import FlockSoundness.Assumptions.Zk
import FlockSoundness.Protocol.Discharge.ZkHidden.DefsJ

/-!
# Zero knowledge of production's `--zk` session: hidden outputs, `J` tables

`zk_session_view` at the session production runs. Its tables are `setupHidden`'s (hidden outputs: the output rows, their
Δ pairs `HmOut.delta`, `HmIn`'s public-input filter, and every row's `b ‖ c` region) at `--session-tables J`, each table
with its own coin vector.

* `zk_session_viewH`: at one table, the view of `tablesAtZKH`'s table at every draw `verify --zk` sets up.
* `ZkViewAtJ`, `zk_session_viewHJ`: at `J` tables, the joint view of `planZHJ`'s tables (`zk_session_soundHJ`'s plan),
  at any coin vector per table, within `Σ_j 2·|Hid_j|·2^-193` of the per-table simulators' joint view, both ways.
* `zk_session_composedHJ`: `zk_session_soundHJ` and `zk_session_viewHJ` on the same plan.

`avoidsMask` is a check on the statement (`st.da`, `st.db`, `st.regions`), so at a `setupHidden` statement it covers the
output rows' Δ pairs and regions as at a `setupH` one. Registered reads (`checkRegistered`) add no prover message: they
hold the statement to the verifier's own files and only shrink the accepting event.
-/

namespace FlockSoundness.Discharge.ZkHidden

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind HiddenExec Composed
  Placed StrictCR DecodeZL
open FlockSoundness.Discharge.Tables (LiveOutJ)
open scoped ENNReal

variable {mPts n : ℕ} {Reg : Type} (P : LigProver)

/-- **Zero knowledge at `J` `CoinedSetup`s, at `δ₁` per hidden leaf** (`ZkViewAt` at `J` tables): at any coin vector of
each table that the honest prover doesn't refuse (`CoinedSetup.Admits`), for any encoding `enc` of the joint view and any
event `E`, the real joint view at valid witnesses `w j` whose registered values are the public ones and the simulators' at
any `w0 j`, each table's prover randomness independent, are within `Σ_j 2·|Hid_j|·δ₁`, both ways. -/
def ZkViewAtJ {J : ℕ} (C : Fin J → CoinedSetup) (δ₁ : ℝ≥0∞) : Prop :=
  ∀ (c : ∀ j, CoinsZK (C j).Rest), (∀ j, (C j).Admits (c j)) → ∀ {Out : Type}
    (enc : (∀ j, ((C j).T (c j)).tab.View) → Out) (w w0 : ∀ j, Witness (C j).S.m), (∀ j, (C j).Valid (w j)) →
    ∀ (_hreg : ∀ j, ((C j).T (c j)).tab.sReg (w j) = ((C j).T (c j)).tab.pubReg) (E : Out → Prop),
    Game.prCoin (fun p : ∀ j, ((C j).T (c j)).tab.Rand × (((C j).T (c j)).tab.Hid → ZkView.Salt) =>
        E (enc fun j => ((C j).T (c j)).viewExe (w j) (p j).1 (p j).2))
        ≤ Game.prCoin (fun p : ∀ j, ((C j).T (c j)).tab.SimRand × (((C j).T (c j)).tab.Hid → ZkView.Salt) =>
            E (enc fun j => ((C j).T (c j)).simExe (w0 j) (p j).1 (p j).2))
          + ∑ j, 2 * Fintype.card ((C j).T (c j)).tab.Hid * δ₁
      ∧ Game.prCoin (fun p : ∀ j, ((C j).T (c j)).tab.SimRand × (((C j).T (c j)).tab.Hid → ZkView.Salt) =>
          E (enc fun j => ((C j).T (c j)).simExe (w0 j) (p j).1 (p j).2))
        ≤ Game.prCoin (fun p : ∀ j, ((C j).T (c j)).tab.Rand × (((C j).T (c j)).tab.Hid → ZkView.Salt) =>
            E (enc fun j => ((C j).T (c j)).viewExe (w j) (p j).1 (p j).2))
          + ∑ j, 2 * Fintype.card ((C j).T (c j)).tab.Hid * δ₁

section Nonempty

variable {F K W Salt B D Hi VL : Type} [Field F] [Field K] [AddCommGroup B] [AddCommGroup Salt] [AddCommGroup Hi]

end Nonempty

end FlockSoundness.Discharge.ZkHidden
