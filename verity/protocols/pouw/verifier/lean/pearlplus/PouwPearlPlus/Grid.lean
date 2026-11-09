module

public import PouwPearlPlus.Check
meta import PouwPearlPlus.Check

/-!
# The grid form of an accepted run's work (`verify_run.grid_work`)

`check`'s work is the run's exact credit at Pearl's assumption (`creditAtAssumption`), which the rows the passes exclude
and give up decide: the retained values', so neither protocol structure nor a coin. A run whose passes are held to a
window class's fixed load (`tieAt`, `Tied.load`) publishes its work in the grid form instead:

* **The structural credit** (`structuralCredit`): `creditAtAssumption` over the passes with every record whole
  (`UnitRecord.whole`: no row excluded or given up), at the same draws, X_O and δ. A whole record's real rows are the
  class's (`heldToLoad`) and its shape and kept width its weight's (`checkRecords`), so the structural credit, MACs and
  credited tiles are the class's and the model's.
* **The give-up** is the structural credit less the exact one, padded up to H1's window grid (`gridPad`, `window.pad` at
  `window.GRID_BITS`: the positive integers with at most `GRID_BITS + 1` significant bits; 0 where nothing is given up).
* **The published work** (`gridWork`) is the structural one with the padded give-up taken off its credit. It reads the
  retained values through one grid level alone, and its credit is at most the exact one, since the padded give-up is at
  least the give-up: a bound that holds at the exact credit holds at it.

`published` is what `pouw-pearlplus` prints for a verdict: the grid form with the exact credit beside it where a class
is held, and `check`'s work as it is where none is (the structural credit is defined by the class). A test instance's
credits nothing in either form.
-/

@[expose] public section

namespace PouwCheck.PearlPlus

/-- H1's window grid (`window.GRID_BITS`): its points are the positive integers with at most `GRID_BITS + 1` significant
bits. -/
def GRID_BITS : Nat := 3

/-- `window.pad`: the least grid point at least `n`; 0 at 0, where there is nothing to pad. -/
def gridPad (n : Nat) (bits : Nat := GRID_BITS) : Nat :=
  if n == 0 then 0
  else
    let s := n.log2 + 1 - (bits + 1)
    (n + 2 ^ s - 1) / 2 ^ s * 2 ^ s

#guard (List.range 17).map (gridPad ·) == List.range 17
#guard [17, 18, 31, 32, 33, 36, 37, 255, 256, 257].map (gridPad ·) == [18, 18, 32, 32, 36, 36, 40, 256, 256, 288]
#guard gridPad 331347979 == 335544320 && gridPad 335544320 == 335544320 && gridPad 335544321 == 369098752

/-- `verify_run.whole`: the record with no row excluded or given up. -/
def UnitRecord.whole (r : UnitRecord) : UnitRecord :=
  { r with excludedA := [], excludedB := [], voluntaryA := [], voluntaryB := [] }

/-- `verify_run.whole_pass`: the pass with each record whole. -/
def Pass.whole (p : Pass) : Pass := { p with records := p.records.map UnitRecord.whole }

/-- `verify_run.structural_credit`: the credit at Pearl's assumption of the run that excludes and gives up no row. -/
def structuralCredit (passes : List Pass) (t over : Nat) (delta : Rat) : Credit :=
  creditAtAssumption (passes.map Pass.whole) t over delta

/-- A run's work in the grid form, and the grid point taken off its structural credit. -/
structure GridWork where
  work : Credit
  giveUp : Nat

/-- `verify_run.grid_work`: the structural work, its credit less the give-up (`structural` less `exact`) padded up to
the grid. -/
def gridWork (structural exact : Credit) : GridWork :=
  let giveUp := gridPad (structural.credited - exact.credited)
  { work := { structural with credited := structural.credited - giveUp }, giveUp }

/-- What a verdict prints of an accepted run's work: `work`, with `giveUp` and the exact credit `exact` where it is the
grid form. -/
structure Published where
  work : Credit
  giveUp : Option Nat := none
  exact : Option Credit := none

/-- An accepted run's work as printed: the grid form (`gridWork`) with `check`'s exact work beside it where the plan held
the passes to a class's fixed load, `check`'s work alone where it held none; a test instance's credits nothing. The
give-up is the exact credit's before a test instance's is set to nothing. -/
def published (i : Input) (ans : RunAnswer) : Option Published :=
  ans.work.map fun w =>
    match ans.plan, ans.tie.bind (·.load) with
    | some pl, some _ =>
      let exact := creditAtAssumption pl.passes i.draws w.overCapDraws DELTA
      let g := gridWork (structuralCredit pl.passes i.draws w.overCapDraws DELTA) exact
      { work := testCredit i.testInstance g.work, giveUp := some g.giveUp, exact := some w }
    | _, _ => { work := w }

/-! ## Class (4, 32, 4) on Llama 3.2 1B: the load of the run of record `r20261106-165116-614b`

Sixteen layers of four linears (k, n): q, k and v as one (2048, 3072), o (2048, 2048), gate and up as one (2048, 16384)
and down (8192, 2048); round 0's calls have 128 rows and each later round's 4, committed in 16. The run of record's
exact credit, 134,194,399,455 MACs of the 135,895,449,600 it declared, is this load's with 19 rows excluded: 4 of an o,
4 of a gate-up and 11 of a down. Its grid form is 4,196,341 MACs less. -/

private def classShapes : List (Nat × Nat) := [(2048, 3072), (2048, 2048), (2048, 16384), (8192, 2048)]

/-- A round's pass of the class's load, `m` committed rows of `real`, its first call of each shape excluding
`excluded`'s rows of that shape. -/
private def classPass (m real : Nat) (excluded : List Nat) : Pass :=
  { (default : Pass) with
    scheme := {}
    records := (List.range 64).map fun c =>
      let (k, n) := classShapes[c % 4]!
      { matmul := ⟨c, m, k, n, c⟩, realRows := real, kKept := k,
        excludedA := if c < 4 then List.range (excluded.getD c 0) else [] } }

private def classRun (excluded : List Nat) : List Pass :=
  classPass 128 128 excluded :: List.replicate 3 (classPass 16 4 [])

#guard
  let exact := creditAtAssumption (classRun [0, 4, 4, 11]) 459 0 DELTA
  let structural := structuralCredit (classRun [0, 4, 4, 11]) 459 0 DELTA
  let g := gridWork structural exact
  (exact.credited, exact.macs) == (134194399455, 135895449600) &&
  (structural.credited, structural.macs) == (134525747434, 136230993920) &&
  structural.credited == (creditAtAssumption (classRun []) 459 0 DELTA).credited &&
  g.giveUp == 335544320 && g.work.credited == 134190203114 && exact.credited - g.work.credited == 4196341 &&
  g.work.macs == structural.macs && g.work.creditedTiles == structural.creditedTiles

end PouwCheck.PearlPlus
