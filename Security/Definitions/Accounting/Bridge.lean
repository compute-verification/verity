import Definitions.Accounting.Device
import Definitions.Accounting.Physical

/-!
# The device's schedule, induced by the actual circuit (trusted)

The device model (`Accounting.Device`, `Schedule`) read off the actual circuit of
`note:console/20261012T0645Z-draft-actual-circuit-model` (v2.2). The device is the location `chip ∪ mem`: its on-chip
state is what the location `chip` carries and its device memory what `mem` carries, so nothing it holds goes uncounted.
Bits reach the chip from device memory on the link `hbm.1` toward its end `hbm.2`, and reach the device over side
channel `i` on the link `(side i).1` toward `(side i).2`. Each gate is charged to an account (`acct g`), and a gate of
an op class (`cls g = some o`) is one op of that class; a gate of no class (a send, a move) is charged no op. The
schedule `A` induces (`Placed.schedule`) charges, at tick `t`, to account `a`:

* `ops a o`: the computed gates of class `o` placed in the device at `t` whose account is `a`;
* `read a`, `side a i`: the gates of `a` whose wire the link carries toward its device end, read at `t` (`readOn`);
* `chip a`, `mem a`: the gates of `a` that `chip` (`mem`) carries across `t + 1`, what it holds at the end of tick `t`
  (`Placed.state`).

A gate is charged once, to its own account, so at each tick the accounts' charges partition the device's: the induced
schedule fits the device under memory, links and a compute rate for each op class
(`Accounting.SecurityProofs.Actual.fits`, and `fits_of_feasible` from the feasible set). Charging each gate to one
account is where additivity enters: what a guarantee says of one account's charge, it says of that account's gates
alone. The schedule also covers what the device's gates use when every wire into `chip` from outside runs on the
memory link or a side channel, and the work's pricing is `p` on each class (`covers_ingress`, `covers_work`).
-/

namespace Accounting

open Finset

namespace Placed

variable {L : Type*} (A : Placed L)

/-- The pricing that counts the gates of op class `o`: one for each, nothing for the others. -/
def classPrice (cls : Fin A.N → Option Op) (o : Op) : Fin A.N → ℕ := fun g => if cls g = some o then 1 else 0

open Classical in
/-- **The schedule `A` induces** on the device at `chip ∪ mem`, with its chip at `chip`, its memory at `mem`, its memory
link `hbm` and its side channels `side` (each a link and its end at the device), each gate charged to `acct g` and
counted as an op of `cls g`. -/
noncomputable def schedule {E ι α : Type} (chip mem : Finset L) (cls : Fin A.N → Option Op) (acct : Fin A.N → α)
    (link : Fin A.N → Fin A.N → E) (hbm : E × L) (side : ι → E × L) : Schedule ι α := fun t =>
  { ops := fun a o => (((A.region (chip ∪ mem) (Set.Ico t (t + 1))).filter fun g => cls g = some o).filter
      fun g => acct g = a).card
    read := fun a => A.width ((A.readOn link hbm.1 hbm.2 (Set.Ico t (t + 1))).filter fun s => acct s = a)
    side := fun a i => A.width ((A.readOn link (side i).1 (side i).2 (Set.Ico t (t + 1))).filter fun s => acct s = a)
    chip := fun a => A.width (((A.into (A.region chip (Set.Ici (t + 1)))).filter fun s => A.x s ∈ chip).filter
      fun s => acct s = a)
    mem := fun a => A.width (((A.into (A.region mem (Set.Ici (t + 1)))).filter fun s => A.x s ∈ mem).filter
      fun s => acct s = a) }

end Placed

end Accounting
