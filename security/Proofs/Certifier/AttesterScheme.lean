module

public import Definitions.Certifier.AttesterScheme
public import Proofs.Certifier.Cells

/-!
# The client's views, read off the network attester's code

`clientViewReads` proves `NetTiming.Attester.ClientViewReads`: the trusted attester runs its code, whose cells are
`clientView` of the schedule, the data and the fail cells (`certifierCode_eq`), and a fail cell lies below its
connection's cells (`failCell_lt`), so below its cap.
-/

@[expose] public section

namespace NetTiming.Attester

/-- A connection's fail cell lies below its number of cells. -/
theorem failCell_lt {κ Cell : Type} (s : CertifierSchedule κ) (arr : κ → Arrivals Cell) (c : κ) {j : ℕ}
    (h : failCell s arr c = some j) : j < s.cells c := by
  unfold failCell at h
  exact List.mem_range.1 (List.mem_of_find?_eq_some h)

theorem clientViewReads : ClientViewReads := by
  intro Conn Cell Srv Dev Ob Z _ f arr tap run htrust cap Ss hcap Zs schedOf srvOf hs z hz
  let s := schedOf z
  let a := arr (srvOf z)
  refine ⟨fun c => match h : failCell s a c with
    | none => none
    | some j => some ⟨j, lt_of_lt_of_le (failCell_lt s a c h) (hcap s (hs z hz) c)⟩, ?_⟩
  rw [htrust]
  show certifierCode s a = _
  rw [certifierCode_eq]
  congr 1
  funext c
  beta_reduce
  split <;> simp_all

end NetTiming.Attester
