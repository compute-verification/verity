module

public import Proofs.Flock.Bridge.Canon

/-!
# A sampled proof block's openings are canonical (`Bridge.OpeningsCanonTarget`, proved)

`OpeningsCanonTarget` asks `canonAt` at every committed wire of a sampled proof block: at a constant (`Bridge.constsOf`), the
opening's bit is the constant; at any other wire, the opening is canonical (`canonOpen`), which asks the wire to have a
row. A sampled proof block's committed constants are the constant `oneG` and the zero `zeroG`: an input no unit reads is in no
unit's `io` (`Bridge.owner_ne_none_of_io`).

* `Bridge.canonOpenH rows g v y`: `g` has no row, or the opening is canonical there (`canonOpen`);
* `Bridge.OpeningsCanonTargetH`: `OpeningsCanonTarget`'s second clause at `canonOpenH`;
* `Bridge.openingsCanonH : Bridge.OpeningsCanonTargetH`. At a wire with a row `(u', q, k)`, its commit string is
  `.inl (u', q)` (`Layout.ofRows_pos_some`), whose value is `(widthOf W (portRowW q), padBits b)` for the bits `b` the
  witness holds there (`Readers.ofBits`'s `rowOf`), canonical by `rowShaped_rowOf` at `portRowW q = 16 · words ≤ W`,
  and whose salt is the witness's bytes at `saltAt`, `List.ofFn` over `Fin 192`;
* `Bridge.openingsCanon_row`: the same, as `canonOpen`, at every wire that has a row;
* `HiddenExec.rowOfH_ne_none_of_io`: a committed wire of a sampled proof block other than the constant and the zero has a row
  (`Layout.single_gate`; a computed one is a returned output's, `OutsOut`, so an output row's bit);
* `Bridge.openingsCanon_const`: at the constant and the zero, the opening's bit is the decoder's (`opening_spec`), which
  reads the unit's constant column, 1, or a zero column, 0, at the public place (`UnitPlace.decode_one`,
  `constCols_keyProgH`, `zeroCols_keyProgH`); a constant input no unit reads isn't the unit's;
* `Bridge.openingsCanon : Bridge.OpeningsCanonTarget`.
-/

@[expose] public section

namespace Flock.Bridge

/-- A witness's bytes at `N` addresses are `N` bytes. -/
theorem bytesAt_length {m N : ℕ} (z) (a : Fin N → Fin 8 → Fin (2 ^ m)) :
    (FlockSoundness.Binding.bytesAt (m := m) z a).length = N := by
  simp [FlockSoundness.Binding.bytesAt]

end Flock.Bridge

namespace FlockSoundness.Discharge.HiddenExec

open Flock FlockLevel3 Model Spot.Sampling Binding Placed
open FlockSoundness.Discharge.Layout (keyN keyRowOf portRowW OutColAt outDecode OutsOut KeysRow)

variable {c : Flock.Circuit} {hU : Built c.unit} {n : ℕ} {Reg : Type} {mPts : ℕ}
  {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length} {si : SitesH plan tab}

end FlockSoundness.Discharge.HiddenExec

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Game Spot.Sampling FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden FlockSoundness.Discharge.ZkSession
open FlockSoundness.Discharge.Tables (LiveOutJ)

end Flock.Guarantees
