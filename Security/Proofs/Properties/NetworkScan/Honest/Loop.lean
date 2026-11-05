import Properties.NetworkScan.Honest
import Proofs.Properties.NetworkScan.Check
import Proofs.Properties.NetworkScan.Passes

/-!
# `lastLe`'s loop, one step at a time

`loopId`: `lastLe`'s `while` loop one step at a time, through which `cbv` reads it at `ScanFits`' honest instance
(`Properties.NetworkScan.Honest`), since the loop's own recursion is opaque. The honest witness's modules build on
this one.
-/

namespace Properties.NetworkScan.Honest

/-- `lastLe`'s `while` loop, one step at a time: `cbv` reads it through this, since the loop's own recursion is
opaque. -/
theorem loopId {β : Type} (b : β) (f : Unit → β → Id (ForInStep β)) :
    Lean.Loop.forIn (m := Id) ⟨⟩ b f = match (f () b).run with
      | .done v => pure v
      | .yield v => Lean.Loop.forIn (m := Id) ⟨⟩ v f := by
  rw [Lean.Loop.forIn_eq_of_monadTail]
  rfl

end Properties.NetworkScan.Honest
