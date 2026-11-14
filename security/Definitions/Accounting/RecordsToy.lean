module

public import Definitions.Accounting.Records

/-!
# The record limit's toy instance

The runs at which `Accounting.Guarantees.LeakPartsRegistered`'s honest lemma is stated. Over two secret bits, each run
is a one-gate circuit at a leaf `y`: the gate is in cell `0` at tick `0`, one bit wide, and holds the first secret
bit. The record's format is that one cell of one bit, so over the four runs the record takes two values: the count is
the format's, not the number of runs.
-/

@[expose] public section

namespace Accounting.Records.Toy

/-- The one-gate actual circuit at the leaf `y`: an input of width one at cell `0`, tick `0`. -/
def oneGate {L : Type*} (y : L) : Placed L where
  N := 1
  w := fun _ => 1
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

/-- The run whose gate holds `b`: the one-gate circuit at `y`, the leaf `y` holding its gate. -/
def run {L : Type*} [DecidableEq L] (y : L) (b : Bool) : Actual.Run L where
  A := oneGate y
  val := fun _ => BitVec.ofBool b
  faithful := fun _ h => by cases h
  held := fun z _ => Finset.univ.filter fun a => (oneGate y).x a = z
  holding := ⟨fun _ _ _ ha => ⟨(Finset.mem_filter.1 ha).2, Or.inl rfl⟩,
    fun a _ => Finset.mem_filter.2 ⟨Finset.mem_univ a, rfl⟩,
    fun _ _ h => absurd h (Finset.notMem_empty _)⟩

/-- The strategy over two secret bits: the run whose gate holds the first. -/
def strategy {L : Type*} [DecidableEq L] (y : L) : Bool × Bool → Actual.Run L := fun r => run y r.1

/-- The record's cells at every leaf: cell `0`. -/
def cells {L : Type*} : L → Set ℕ := fun _ => {0}

end Accounting.Records.Toy
