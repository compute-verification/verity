/-
A row net's slot in the accepted template statement. At a row of a `sha512x3` or `hm96` slot past the net's input rows,
the block matrices' row is the net's row placed in the slot plus Δ's first-loop pairs (`rowNet_entry`): Δ's other flat
pairs are at the row nets' input bits (`TemplateLayout.deltaIn`), and the typed tail's at unit columns, which are in the
root's and the parts' slots. So in every satisfying witness the slot holds its net's rows (`laidRows_of_sat`): the
computed rows hold, and the constant column is 1 (its row is `pin · pin`, or the pin itself). With `HmNets.check`, VU
`g`'s `sha512x3` and `hm96` slots hold the verifier's layouts past their 8192 and 4096 input rows (`shaSlot_of_sat`,
`hmSlot_of_sat`), `port_commit_rows`'s `hsha` and `hhm`.
-/
import FlockSoundness.Protocol.Discharge.Hm.Nets
import FlockSoundness.Protocol.ExecTemplatePlace

open Flock FlockLevel3

namespace FlockSoundness.Discharge.Hm

open Compose Flock.DeriveAll

/-- The block's bit at column `y` of block `o` (0 past the block). -/
def blockBit (S : Model.Statement) (z : Model.Witness S.m) (o : Fin (2 ^ S.nbl)) (y : ℕ) : Bool :=
  FlockSoundness.bit (if h : y < 2 ^ S.kLog then S.block z o ⟨y, h⟩ else 0)

/-- A row of `Sparse.ofRows`'s rows, its columns below the width. -/
theorem ofRows_cols_lt {rows : Array (Array ℕ)} {width : ℕ} {m : Sparse} (h : Sparse.ofRows rows width = .ok m)
    (i : ℕ) : ∀ y ∈ (rows.getD i #[]).toList, y < width := by
  intro y hy
  by_cases hi : i < rows.size
  · have hri : rows.getD i #[] = rows[i] := by simp [Array.getD_eq_getD_getElem?, hi]
    rw [hri] at hy
    exact (ofRows_spec h).2.2.2 _ (Array.getElem_mem_toList hi) y hy
  · have hri : rows.getD i #[] = #[] := by simp [Array.getD_eq_getD_getElem?, hi]
    rw [hri] at hy
    cases hy

theorem shaNet_inRows {c : Circuit} (h : Flock.HmNets.check c = .ok ()) :
    (c.nets.getD c.shaNet default).2.inWords * Flock.WORD = 8192 := by
  obtain ⟨-, -, -, -, -, hi, -, -, -, -, -, -, -, hw, -⟩ := Net.ofRows_ok (shaNet_ofRows h)
  rw [hw, ← hi, (shaNet_groups h).1]
  decide

theorem hmNet_inRows {c : Circuit} (h : Flock.HmNets.check c = .ok ()) :
    (c.nets.getD c.hmNet default).2.inWords * Flock.WORD = 4096 := by
  obtain ⟨-, -, -, -, -, hi, -, -, -, -, -, -, -, hw, -⟩ := Net.ofRows_ok (hmNet_ofRows h)
  rw [hw, ← hi, (hmNet_groups h).1]
  decide

section Slots

variable {types : String → Option CircuitType} {ls : Array Flock.Layout} {info : String → Option TableInfo}
  {words : String → Option (Array ℕ)} {unit : ℕ} {done : Array D} {st : Stmt} {tb : TypedBlock}

end Slots

end FlockSoundness.Discharge.Hm
