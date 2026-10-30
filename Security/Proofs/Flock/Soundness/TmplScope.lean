import Proofs.Flock.Soundness.ExecRows
import Proofs.Flock.Soundness.Discharge.FailClosed.Scope
import Proofs.Flock.Soundness.Discharge.Aliased.Rows
import Flock.TmplOrder
import Flock.NetAt

/-!
# A templated unit's port reads, from its description

`ProvedScope.portReadsB` walks every computed row of the unit's net. On a templated unit's net (`Tmpl.unrollAt`) it holds
whenever the template is ordered (`portReadsB_unrollAt`): `Tmpl.ordered` makes every logical row read port bits among
the input columns (`TmplOrder.rows_good`), and the position map keeps the input rows in place and moves nothing else
onto them (`Net.posOk`), so a committed row past the input rows reads input columns only at port bits.
-/

open Flock Flock.Tmpl

namespace FlockSoundness.TmplFold

/-- An input column `portSpan` names is a port bit. -/
theorem portSpan_bits {t : Tmpl} {c : Nat} (h : t.portSpan c (c + 1) = true) : c ∈ portBits t.inGroups := by
  unfold Tmpl.portSpan at h
  rw [List.any_eq_true] at h
  obtain ⟨g, hg, hq⟩ := h
  simp only [Bool.and_eq_true, decide_eq_true_eq] at hq
  rw [Discharge.Aliased.portBits_eq]
  refine List.mem_flatMap.2 ⟨g, hg, List.mem_range'_1.2 ⟨hq.1, ?_⟩⟩
  have := hq.2
  rw [← Array.foldl_toList, foldl_add_eq_sum, Nat.zero_add] at this
  omega

/-- **An ordered template's committed net reads, among its input columns, only port bits**: `portReadsB` holds on
`Tmpl.unrollAt`'s net, checked on the description (`Tmpl.ordered`). -/
theorem portReadsB_unrollAt {t : Tmpl} {lays : Array Lay} {net : Net} (ho : t.ordered = true)
    (h : Tmpl.unrollAt t lays = .ok net) (hlog : net.unitLog ≤ 32) : ProvedScope.portReadsB net = true := by
  unfold Tmpl.unrollAt at h
  obtain ⟨acc, hr, h2⟩ := TmplOrder.bind_ok h
  obtain ⟨n0, hn0, -, hcp, -, ha, hb, -, hig, hiw, -, -, hpos, -, -⟩ := Net.ofRowsAt_ok h2
  obtain ⟨-, -, -, -, -, hig0, -, -, -, -, -, -, -, hw0, -⟩ := Net.ofRows_ok hn0
  have hg := TmplOrder.rows_good ho hr
  have hinc : n0.inWords * WORD = t.inCols := by rw [hw0]; rfl
  have hw : 2 ^ net.unitLog ≤ 2 ^ 32 := Nat.pow_le_pow_right (by decide) hlog
  generalize hP : Tmpl.posMap t lays = pos at hcp ha hb hpos
  generalize hI : Net.invOf pos (pos.getD t.constPos 0 + 1) = inv at ha hb hpos
  generalize hC : (t.defs.getD t.root default).cols = cols at hpos
  rw [hinc] at hpos
  have hK : t.constPos = cols - 1 := by unfold Tmpl.constPos; rw [hC]
  unfold ProvedScope.portReadsB
  simp only [List.all_eq_true, List.mem_range, List.mem_append, decide_eq_true_eq]
  intro i hi x hx hxn
  rw [hiw, hinc, hcp] at hi hxn hx ⊢
  rw [Discharge.FailClosed.portMask_getD _ hxn, decide_eq_true_eq, hig, hig0]
  generalize hN : min t.inCols (pos.getD t.constPos 0) = n at hi hx hxn
  have hn1 : n ≤ t.inCols := hN ▸ Nat.min_le_left _ _
  have hn2 : n ≤ pos.getD t.constPos 0 := hN ▸ Nat.min_le_right _ _
  have hn3 : n = t.inCols ∨ n = pos.getD t.constPos 0 := by omega
  rw [ofRows_row ha hw, ofRows_row hb hw] at hx
  by_cases hy : ∃ y < cols, pos.getD y 0 = n + i
  · obtain ⟨y, hyc, hyp⟩ := hy
    have hcols : 0 < cols := by omega
    have hKc : t.constPos < cols := by omega
    have hyK : y ≠ t.constPos := fun e => by subst e; omega
    have hyi : t.inCols ≤ y := by
      refine Nat.le_of_not_lt fun hlt => ?_
      have := Net.posOk_inputs hpos hlt
      omega
    rw [← hyp, Net.placed_at hpos _ hyc, Net.placed_at hpos _ hyc] at hx
    simp only [Array.toList_map, List.mem_map] at hx
    have hreads : ∀ c, TmplOrder.Reads t.inCols t.constPos (fun c => t.portSpan c (c + 1) = true) y c →
        pos.getD c 0 < n → pos.getD c 0 ∈ portBits t.inGroups := by
      intro c hc hlt
      rcases hc with e | ⟨hcy, hcp'⟩
      · subst e; omega
      · by_cases hci : c < t.inCols
        · rw [Net.posOk_inputs hpos hci] at hlt ⊢
          exact portSpan_bits (hcp' hci)
        · exfalso
          have hpc : pos.getD c 0 < t.inCols := by omega
          have e1 := Net.posOk_inputs hpos hpc
          have := Net.posOk_inj hpos (by omega) (by omega) e1
          omega
    rcases hx with ⟨c, hc, rfl⟩ | ⟨c, hc, rfl⟩
    · exact hreads c ((hg y hyi (by omega)).1 c hc) hxn
    · exact hreads c ((hg y hyi (by omega)).2 c hc) hxn
  · have hn : ∀ y < cols, pos.getD y 0 ≠ n + i := fun y hyc e => hy ⟨y, hyc, e⟩
    rw [Net.placed_gap hpos _ hn, Net.placed_gap hpos _ hn] at hx
    simp at hx

end FlockSoundness.TmplFold
