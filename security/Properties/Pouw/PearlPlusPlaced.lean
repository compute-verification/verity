module

public import Properties.Remote.ValueTie

/-!
# Pearl+'s cells in the tied circuit (`Verity.ValueTie.Tying.place`, `Charges.untied`)

What a run held to its registration record W gives the tied circuit (`Verity.ValueTie.Tying`): where the circuit's cells
are the run's real cells at W's entries (`TiesTheRun`), every credited cell is placed from below, and each block's
given-up cells are a count of the records alone. That the cells are those is a hypothesis of each statement, which the
tie discharges where it builds the tied circuit from the run (`Verity.ValueTie.Tying.AtW` holds one side of it); it is
no named assumption. `TieMeans`, what `tieOf`'s answer establishes, is a premise of each, stated and unproved in
`Properties.Pouw.PearlPlusTie`.

* `CreditedArePlaced`: every credited cell (`creditedCells`) of a run held to W is placed at its home block
  (`Tying.place`), at 32 y gates of that block of the tied circuit's blocks (`Tying.blocks`). The placement is from
  below (`Tying.FromBelow`) at every blocks, homes and untied cells, with no premise; what Pearl+ adds is that the
  cells the run credits are the cells the circuit ties.
* `GivenUpByBlock`: where each cell's home reads its pass, call and A row alone (`hr`), the given-up cells
  (`givenUpCells`) the tied circuit holds at block `u` are `givenUpAt`, the records' given-up cells whose A row `hr`
  homes at `u`; and any window's untied cells at `u` (`Charges.untied`, the set `Verity.ValueTie.PeriodOutsideG`
  counts) number at most that give-up plus its untied cells outside the give-up, which `UntiedCellsAtMost` bounds by
  the wrong share. The give-up is fixed before the coin, so it is public.
-/

@[expose] public section

namespace Properties.Pouw.PearlPlus

open PouwCheck.PearlPlus

/-- **The tied circuit's cells are the run's real cells at W's entries**: a cell is one it ties exactly when it is a
real cell of the run `R` whose call's A rows, y and weight's B rows W registers (`Verity.ValueTie.CellAt`). -/
def TiesTheRun {C : Verity.Circuit} {θ : Finset (Fin C.N)} {L : Verity.ValueTie.Cut C}
    (T : Verity.ValueTie.Tying C θ L) (R : Run) : Prop :=
  ∀ c : Cell, c ∈ T.cells ↔ ∃ (p : Pass) (r : UnitRecord) (da db dy : PouwCommit.RowsDomain)
    (rootA rootB yRoot : ByteArray), Verity.ValueTie.CellAt R c p r da db dy rootA rootB yRoot

/-- **Every credited cell is placed.** Under `TieMeans`, for a run `R` held to W and a tied circuit whose cells are the
run's real cells at W's entries (`TiesTheRun`), at any registered file's blocks `Q` and homes `home`: every credited
cell of the run is placed at its home block, at gates of that block of the tied circuit's blocks, 32 bits of them. -/
def CreditedArePlaced : Prop :=
  TieMeans → ∀ (R : Run) {C : Verity.Circuit} {θ : Finset (Fin C.N)} {L : Verity.ValueTie.Cut C}
    (T : Verity.ValueTie.Tying C θ L) {m : ℕ} (Q : Verity.Partition C m) (home : Cell → Fin m),
    R.HeldToW → TiesTheRun T R → ∀ c ∈ creditedCells R.passes,
      ∃ O, T.place home c = some (home c, O) ∧ O ⊆ (T.blocks Q home).gates (home c) ∧ T.tied.D.width O = 32

/-- **The give-up of block `u`**: the given-up cells of the records `passes` whose (pass, call, A row) the homes `hr`
put at `u`. It reads the records and the homes alone. -/
def givenUpAt {m : ℕ} (passes : List Pass) (hr : ℕ × ℕ × ℕ → Fin m) (u : Fin m) : ℕ :=
  ((givenUpCells passes).filter fun c => hr (c.1, c.2.1, c.2.2.1) = u).card

/-- **The give-up, block by block.** Under `TieMeans`, for a run `R` held to W, a tied circuit whose cells are the run's
real cells at W's entries (`TiesTheRun`) and homes that read each cell's pass, call and A row alone (`hr`): at each
block `u`, the given-up cells the tied circuit ties there are the records' give-up of `u` (`givenUpAt`), and any set
of cells `U`, a window's untied cells among them, holds at most that give-up at `u` and its own cells outside the
give-up there. -/
def GivenUpByBlock : Prop :=
  TieMeans → ∀ (R : Run) {C : Verity.Circuit} {θ : Finset (Fin C.N)} {L : Verity.ValueTie.Cut C}
    (T : Verity.ValueTie.Tying C θ L) {m : ℕ} (home : Cell → Fin m) (hr : ℕ × ℕ × ℕ → Fin m),
    R.HeldToW → TiesTheRun T R → (∀ c : Cell, home c = hr (c.1, c.2.1, c.2.2.1)) → ∀ u : Fin m,
      ((T.cells ∩ givenUpCells R.passes).filter fun c => home c = u).card = givenUpAt R.passes hr u ∧
      ∀ U : Finset Cell, (U.filter fun c => home c = u).card ≤
        givenUpAt R.passes hr u + ((U \ givenUpCells R.passes).filter fun c => home c = u).card

end Properties.Pouw.PearlPlus
