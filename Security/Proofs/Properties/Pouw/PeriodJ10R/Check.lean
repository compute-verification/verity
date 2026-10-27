import Properties.Pouw.PeriodJ10R.Check
import Proofs.Properties.Pouw.PeriodCheck

/-!
# Proof of `Properties.Pouw.PeriodJ10R.Check`

From `Properties.Pouw.PeriodCheck.acceptMeans`: an accept credits each window by `windowAt`, which is
`Credit.windowLayout` at `registeredOf` and `answeredOf` of the window, and `Credit.window` is its first component.
-/

namespace Properties.Pouw.PeriodJ10R.Check

open Lean (Json toJson)
open PouwCheck PouwCheck.Credit Properties.Pouw.PeriodCheck

theorem acceptedWindows : AcceptedWindows := by
  intro p c e pc prof h
  obtain ⟨_, hwn, ws, hws, _, hK, T, hT, hlen, hall⟩ := acceptMeans p c e pc prof h
  refine ⟨ws, hws, by rw [hwn, hlen], hK, T, hT, fun i => ?_⟩
  have hi : i.val < (ws.zip pc.windows).length := by
    have := i.isLt
    simp only [List.length_zip]
    omega
  have hmem : (ws[i], pc.windows[i.val]'(hlen ▸ i.isLt)) ∈ ws.zip pc.windows :=
    List.mem_iff_getElem.mpr ⟨i.val, hi, by simp [List.getElem_zip]⟩
  obtain ⟨layout, rows, hw⟩ := hall _ hmem
  refine ⟨pc.windows[i.val]'(hlen ▸ i.isLt), ?_⟩
  show (·.1) <$> windowAt p.key c.ledger e.pins p.main pc.k T ws[i] = _
  rw [hw]
  rfl

end Properties.Pouw.PeriodJ10R.Check
