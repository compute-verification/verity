import Definitions.Accounting.Joint
import Proofs.Core.Device

/-!
# A joint run's accounts within the device's limits

An account's charge plus every other account's is within the device's limit over the audit's window
(`Run.work_add_othersWork` and the two beside it), from core's bounds on a schedule that fits the device
(`Verity.Schedule.sum_work_le` and the two beside it).
-/

namespace Accounting

namespace Run

variable {ι : Type} {d : Device ι} {α ΩW ΩQ ΩS : Type} [Fintype α] [DecidableEq α] (r : Run d α ΩW ΩQ ΩS)

theorem work_add_othersWork (p : Price) (ω : ΩW × ΩQ × ΩS) :
    r.work p ω + r.othersWork p ω ≤ d.priced p * (r.win .compute).len := by
  calc _ = ∑ a, (r.schedule ω).work (r.win .compute) a p := Finset.add_sum_erase _ (fun a => _) (Finset.mem_univ _)
    _ ≤ _ := Verity.Schedule.sum_work_le (r.fits ω) _ p

theorem read_add_othersRead (ω : ΩW × ΩQ × ΩS) : r.read ω + r.othersRead ω ≤ d.bw * (r.win .bandwidth).len := by
  calc _ = ∑ a, (r.schedule ω).read (r.win .bandwidth) a := Finset.add_sum_erase _ (fun a => _) (Finset.mem_univ _)
    _ ≤ _ := Verity.Schedule.sum_read_le (r.fits ω) _

theorem held_add_othersHeld (ω : ΩW × ΩQ × ΩS) : r.held ω + r.othersHeld ω ≤ d.mem * (r.win .capacity).len := by
  calc _ = ∑ a, (r.schedule ω).held (r.win .capacity) a := Finset.add_sum_erase _ (fun a => _) (Finset.mem_univ _)
    _ ≤ _ := Verity.Schedule.sum_held_le (r.fits ω) _

end Run

end Accounting
