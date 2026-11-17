module

public import Definitions.ResourceAccounting.Records

/-! The toy runs at which `ResourceAccounting.Guarantees.LeakPartsRegistered`'s two witnesses are stated. -/

@[expose] public section

namespace ResourceAccounting.Records.Toy

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

def run {L : Type*} [DecidableEq L] {n : ℕ} (x : Fin n → L) {w : ℕ} (v : BitVec w) : Actual.Run L where
  A := gates x w
  val := fun _ => v
  faithful := fun _ h => by cases h
  held := fun z _ => Finset.univ.filter fun a => (gates x w).x a = z
  holding := ⟨fun _ _ _ ha => ⟨(Finset.mem_filter.1 ha).2, Or.inl rfl⟩,
    fun a _ => Finset.mem_filter.2 ⟨Finset.mem_univ a, rfl⟩,
    fun _ _ h => absurd h (Finset.notMem_empty _)⟩

def strategy {L : Type*} [DecidableEq L] {n : ℕ} (x : Fin n → L) : Bool × Bool → Actual.Run L :=
  fun r => run x (BitVec.ofBool r.1)

def cells {L : Type*} : L → Set ℕ := fun _ => {0}

end ResourceAccounting.Records.Toy
