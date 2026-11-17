module

public import Proofs.Flock.Soundness.Discharge.Exec.Event
public import Proofs.Flock.Soundness.Discharge.Recursion.Multi
public import Proofs.Flock.Claims.Steps.VStar
import Proofs.Flock.Soundness.Game
import Proofs.Flock.Soundness.Registered.Binding
public import Proofs.Flock.Soundness.Registered.ReadsX
import Mathlib.Combinatorics.Matroid.Init

/-! # What V*'s bridge obligation reads of `Recursive.ZkReg.Off`, without its recursion imports. -/

@[expose] public section

namespace FlockSoundness.Discharge.Recursion

/-- **A commit string that is a row of a unit of `S`.** -/
def rowIn {n : ℕ} {Q G : Type} (S : Finset (Fin n)) : (Fin n × Q) ⊕ G → Prop
  | .inl (u, _) => u ∈ S
  | .inr _ => False

end FlockSoundness.Discharge.Recursion

namespace FlockSoundness.Registered

open Game Refine Binding Audit

open scoped ENNReal

variable {C : Verity.Boolean.Circuit} {n : ℕ} {P : Audit.Partition C n} {Reg : Type} {mPts : ℕ}
  {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}
  {decode : Audit.Partition.DecoderB (C := C) (Reg := Reg) plan tab}

namespace ReadsX

variable {lay : Layout P plan tab decode} {rs : HmRowsX lay}

/-- **The registered reads at the positions `A` keeps.** -/
def restrict (rd : ReadsX lay rs) (A : lay.Pos → Prop) : ReadsX lay rs where
  isRead p := rd.isRead p ∧ A p
  read_row p h := rd.read_row p h.1
  port := rd.port
  readAt := rd.readAt
  wv := rd.wv
  ws := rd.ws
  wpath := rd.wpath
  committed p h := rd.committed p h.1

end ReadsX

end FlockSoundness.Registered
