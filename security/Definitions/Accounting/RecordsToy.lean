module

public import Definitions.Accounting.Records

/-!
# The record limit's toy instance

The runs at which `Accounting.Guarantees.LeakPartsRegistered`'s two witnesses are stated. Each run is a circuit of
input gates, gate `g` at the leaf `x g`, in cell `g` at tick `0`, each `w` bits wide and holding one value `v`.

* The honest lemma's strategy is over two secret bits, its gates one bit wide and holding the first. A record's
  format is cell `0`, one bit, so over the four runs a record takes two values: the count is the format's, not the
  number of runs.
* The nontrivial lemma's strategy is over the values of one `w`-bit gate, which the gate holds: `2 ^ w` runs that
  differ at the gate's own leaf.
-/

@[expose] public section

namespace Accounting.Records.Toy

/-- The actual circuit of `n` input gates of width `w`: gate `g` at the leaf `x g`, in cell `g` at tick `0`. -/
def gates {L : Type*} {n : ℕ} (x : Fin n → L) (w : ℕ) : Placed L where
  N := n
  w := fun _ => w
  isInput := fun _ => true
  reads := fun _ => ∅
  reads_lt := fun _ _ h => absurd h (Finset.notMem_empty _)
  input_reads := fun _ _ => rfl
  op := fun _ _ => 0
  op_congr := fun _ _ _ _ => rfl
  outputs := ∅
  x := x
  t := fun _ => 0
  cell := fun g => g.val
  cell_inj := fun _ _ e => Fin.ext (congrArg (fun p : L × ℕ × ℕ => p.2.1) e)

/-- The run whose gates each hold `v`: the circuit of gates at the leaves `x`, each leaf holding its gates. -/
def run {L : Type*} [DecidableEq L] {n : ℕ} (x : Fin n → L) {w : ℕ} (v : BitVec w) : Actual.Run L where
  A := gates x w
  val := fun _ => v
  faithful := fun _ h => by cases h
  held := fun z _ => Finset.univ.filter fun a => (gates x w).x a = z
  holding := ⟨fun _ _ _ ha => ⟨(Finset.mem_filter.1 ha).2, Or.inl rfl⟩,
    fun a _ => Finset.mem_filter.2 ⟨Finset.mem_univ a, rfl⟩,
    fun _ _ h => absurd h (Finset.notMem_empty _)⟩

/-- The strategy over two secret bits: the run whose one-bit gates hold the first. -/
def strategy {L : Type*} [DecidableEq L] {n : ℕ} (x : Fin n → L) : Bool × Bool → Actual.Run L :=
  fun r => run x (BitVec.ofBool r.1)

/-- The record's cells at every leaf: cell `0`. -/
def cells {L : Type*} : L → Set ℕ := fun _ => {0}

end Accounting.Records.Toy
