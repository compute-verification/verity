module

public import Definitions.Registration.Device
public import Definitions.Registration.Physical

/-! The device's schedule induced by the actual circuit, each gate charged once, to its own account. -/

@[expose] public section

namespace ResourceAccounting

open Finset

namespace Placed

variable {L : Type*} (A : Placed L)

def classPrice (cls : Fin A.N → Option OpClass) (o : OpClass) : Fin A.N → ℕ := fun g => if cls g = some o then 1 else 0

end Placed

namespace Actual.Run

variable {L : Type*} (R : Actual.Run L)

open Classical in
noncomputable def schedule {E ι α : Type} (chip mem : Finset L) (cls : Fin R.A.N → Option OpClass) (acct : Fin R.A.N → α)
    (link : Fin R.A.N → Fin R.A.N → E) (hbm : E × L) (side : ι → E × L) : Schedule ι α := fun t =>
  { ops := fun a o => (((R.A.region (chip ∪ mem) (Set.Ico t (t + 1))).filter fun g => cls g = some o).filter
      fun g => acct g = a).card
    read := fun a => R.A.width ((R.A.readOn link hbm.1 hbm.2 (Set.Ico t (t + 1))).filter fun s => acct s = a)
    side := fun a i =>
      R.A.width ((R.A.readOn link (side i).1 (side i).2 (Set.Ico t (t + 1))).filter fun s => acct s = a)
    chip := fun a => R.A.width ((R.state chip (t + 1)).filter fun s => acct s = a)
    mem := fun a => R.A.width ((R.state mem (t + 1)).filter fun s => acct s = a) }

end Actual.Run

end ResourceAccounting
