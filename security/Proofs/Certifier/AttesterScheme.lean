module

public import Definitions.Certifier.AttesterScheme
public import Proofs.Certifier.Cells
public import Proofs.Flock.Claims.Finder

/-!
# The network attester's scheme: its break term and the client's views

`clientViewReads` proves `NetTiming.Attester.ClientViewReads`: the trusted attester runs its code, whose cells are
`clientView` of the schedule, the data and the fail cells (`certifierCode_eq`), and a fail cell lies below its
connection's cells (`failCell_lt`), so below its cap. `breaksWithinTerm` proves `NetTiming.Attester.BreaksWithinTerm`: a
break is a collision `findColl` finds among evaluations of `H`, which the runs' finder outputs, and `ecrBound` bounds
the finder's probability by definition.
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

theorem breaksWithinTerm : BreaksWithinTerm := by
  intro ph Chan _ ρ op hop Root Coin Complete fixed hcharge H S _ _ play q hlog
  show ENNReal.ofReal (Claims.Pr fun s => (Claims.findColl (play s).calls).isSome = true) ≤
    Claims.ecrBound H (callsFinder play) q
  have h1 : Claims.Pr (fun s => (Claims.findColl (play s).calls).isSome = true) ≤ (callsFinder play).Pr H := by
    unfold Claims.Finder.Pr
    refine Claims.Pr_mono fun s hs => ?_
    obtain ⟨⟨u, v⟩, huv⟩ := Option.isSome_iff_exists.1 hs
    exact ⟨u, v, huv, Claims.findColl_spec H _ (hlog s) u v huv⟩
  refine (ENNReal.ofReal_le_ofReal h1).trans ?_
  unfold Claims.ecrBound
  split_ifs with h
  · exact ENNReal.ofReal_le_ofReal (h.1.trans (div_le_div_of_nonneg_right h.2 (by positivity)))
  · exact le_top

end NetTiming.Attester
