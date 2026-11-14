module

public import Definitions.Accounting.Records

/-!
# The record limit's toy instance

The runs at which `Accounting.Guarantees.LeakPartsRegistered`'s two witnesses are stated. Each run is a one-gate
circuit at a leaf `y`: the gate is in cell `0` at tick `0`, `w` bits wide, and holds a value `v`.

* The honest lemma's strategy is over two secret bits, its gate one bit wide and holding the first. The record's
  format is that one cell of one bit, so over the four runs the record takes two values: the count is the format's,
  not the number of runs.
* The nontrivial lemma's strategy is over the values of a `w`-bit gate, which the gate holds: `2 ^ w` runs that differ
  at the gate's own leaf.
-/

@[expose] public section

namespace Accounting.Records.Toy

/-- The one-gate actual circuit at the leaf `y`: an input of width `w` at cell `0`, tick `0`. -/
def oneGate {L : Type*} (y : L) (w : ℕ) : Placed L where
  N := 1
  w := fun _ => w
  isInput := fun _ => true
  reads := fun _ => ∅
  reads_lt := fun _ _ h => absurd h (Finset.notMem_empty _)
  input_reads := fun _ _ => rfl
  op := fun _ _ => 0
  op_congr := fun _ _ _ _ => rfl
  outputs := ∅
  x := fun _ => y
  t := fun _ => 0
  cell := fun _ => 0
  cell_inj := fun g g' _ => Subsingleton.elim g g'

/-- The run whose gate holds `v`: the one-gate circuit at `y`, the leaf `y` holding its gate. -/
def run {L : Type*} [DecidableEq L] (y : L) {w : ℕ} (v : BitVec w) : Actual.Run L where
  A := oneGate y w
  val := fun _ => v
  faithful := fun _ h => by cases h
  held := fun z _ => Finset.univ.filter fun a => (oneGate y w).x a = z
  holding := ⟨fun _ _ _ ha => ⟨(Finset.mem_filter.1 ha).2, Or.inl rfl⟩,
    fun a _ => Finset.mem_filter.2 ⟨Finset.mem_univ a, rfl⟩,
    fun _ _ h => absurd h (Finset.notMem_empty _)⟩

/-- The strategy over two secret bits: the run whose one-bit gate holds the first. -/
def strategy {L : Type*} [DecidableEq L] (y : L) : Bool × Bool → Actual.Run L := fun r => run y (BitVec.ofBool r.1)

/-- The record's cells at every leaf: cell `0`. -/
def cells {L : Type*} : L → Set ℕ := fun _ => {0}

end Accounting.Records.Toy
