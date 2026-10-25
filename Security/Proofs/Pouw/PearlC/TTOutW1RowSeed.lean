import Definitions.Pouw.PearlC.W1
import Models.Pouw.Assumptions.PearlC.TTOutRowSeed
import Proofs.Pouw.PearlC.Rows

/-!
# The row-seed reductions' cost-model conditions at W1

The row-seed readers of TT_OUT (`RowSeedGamma.lean`, `RowSeedKGamma.lean`, `RowSeedCode.lean`, `RowCapAtGamma.lean`)
take two conditions on the cost model: (C3) `CostModel.SplitClosed`, and for the code-form seed (C5)
`CostModel.RelabelClosed`. Both hold at `W1.w1Priced dv salt`, the cost model TT_OUT is stated at (`TTOutW1`), for
every device and salt encoding:

- (C5) `w1Priced_relabelClosed`: W1 runs no oracle, so relabelling the oracle changes nothing.
- (C3) `w1Priced_splitClosed`: W1 keys a unit's activations, ticket words and useful words by the layout's global row
  (`W1.actAt`, `W1.ticketAt`, `W1.usefulAt`), and on the split layout split unit `v`'s one row is global row `v`
  (`rowOffset_splitLayout`). So the original program's text at the original layout (`splitProg`), run on the split
  workload with each split unit's row as its activations (`splitAct`), starts from the same memory (`init_split`) and
  runs to the same final state (`final_split`): the same transcript words, at the same cost and no calls. Its readout
  groups those words row by row (`readout_split`), which is `splitTranscript`.
-/

namespace Pouw.PearlC.TTOutW1RowSeed

open Pouw.PearlC

variable {Q R S : Type}

/-- W1 runs no oracle, so it meets (C5): the program itself, against `H`, commits and costs what it does against
`H ∘ π`. -/
theorem w1Priced_relabelClosed (dv : W1.W1Device) (salt : S → List ℕ) :
    (W1.w1Priced dv salt : CostModel Q R S).RelabelClosed :=
  fun _ prog _ _ => ⟨prog, fun _ _ _ => ⟨rfl, le_rfl, le_rfl⟩⟩

/-! ## The split layout's global rows -/

/-- Every split unit has one row, so split unit `v` starts at global row `v`. -/
theorem rowOffset_splitLayout (L : Layout) (v : ℕ) : rowOffset (splitLayout L) v = v := by
  simp [rowOffset, splitLayout]

theorem rowsOf_splitLayout (L : Layout) : rowsOf (splitLayout L) = rowsOf L :=
  rowOffset_splitLayout L (rowsOf L)

theorem unitOfRow_splitLayout {L : Layout} {v : ℕ} (hv : v < rowsOf L) : unitOfRow (splitLayout L) v = v := by
  have h := unitOfRow_rowIdx (L := splitLayout L) (u := v) (i := 0) hv Nat.one_pos
  rw [rowIdx, rowOffset_splitLayout, Nat.add_zero] at h
  exact h

theorem rowOfRow_splitLayout {L : Layout} {v : ℕ} (hv : v < rowsOf L) : rowOfRow (splitLayout L) v = 0 := by
  have h := rowOfRow_rowIdx (L := splitLayout L) (u := v) (i := 0) hv Nat.one_pos
  rw [rowIdx, rowOffset_splitLayout, Nat.add_zero] at h
  exact h

theorem rowIdx_splitLayout (L : Layout) (v i : ℕ) : rowIdx (splitLayout L) v i = v + i := by
  rw [rowIdx, rowOffset_splitLayout]

theorem splitWorkload_layout (U : Workload) : (splitWorkload U).layout = splitLayout U.layout := rfl

theorem splitLayout_N (L : Layout) : (splitLayout L).N = rowsOf L := rfl

theorem splitLayout_m (L : Layout) (v : ℕ) : ((splitLayout L).shape v).m = 1 := rfl

theorem splitLayout_k (L : Layout) (v : ℕ) : ((splitLayout L).shape v).k = (L.shape (unitOfRow L v)).k := rfl

/-- Split unit `v`'s row 0 is the global row `v` of its unit's row. -/
theorem rowIdx_split_eq (L : Layout) (v : ℕ) :
    rowIdx (splitLayout L) v 0 = rowIdx L (unitOfRow L v) (rowOfRow L v) := by
  rw [rowIdx_splitLayout, Nat.add_zero, rowIdx_unitOfRow]

theorem actAt_split (L : Layout) (v l : ℕ) :
    W1.actAt (splitLayout L) v 0 l = W1.actAt L (unitOfRow L v) (rowOfRow L v) l := by
  unfold W1.actAt; rw [rowIdx_split_eq]

theorem ticketAt_split (L : Layout) (v j : ℕ) :
    W1.ticketAt (splitLayout L) v 0 j = W1.ticketAt L (unitOfRow L v) (rowOfRow L v) j := by
  unfold W1.ticketAt; rw [rowIdx_split_eq]

theorem usefulAt_split (L : Layout) (v j : ℕ) :
    W1.usefulAt (splitLayout L) v 0 j = W1.usefulAt L (unitOfRow L v) (rowOfRow L v) j := by
  unfold W1.usefulAt; rw [rowIdx_split_eq]

theorem rowCodes_zero (M : Codes) (i l : ℕ) : rowCodes M i 0 l = M i l := rfl

theorem rowCodes_ne (M : Codes) (i : ℕ) {i' : ℕ} (h : i' ≠ 0) (l : ℕ) : rowCodes M i i' l = 0 := by
  unfold rowCodes
  exact ite_eq_right h

/-! ## The split run is the original run -/

/-- The split program: the original's constants and code at the layout `L`, at every layout. -/
def splitProg (pr : W1.Prog) (L : Layout) : W1.Prog :=
  ⟨fun _ => pr.consts L, fun _ => pr.code L, fun _ => pr.bounded L⟩

/-- The split activations: split unit `v`'s row 0 is row `rowOfRow L v` of unit `unitOfRow L v`, and its other rows
are 0. -/
def splitAct (L : Layout) (act : ℕ → Codes) : ℕ → Codes :=
  fun v i' l => if i' = 0 then act (unitOfRow L v) (rowOfRow L v) l else 0

theorem splitAct_zero (L : Layout) (act : ℕ → Codes) (v l : ℕ) :
    splitAct L act v 0 l = act (unitOfRow L v) (rowOfRow L v) l := by
  show (if (0 : ℕ) = 0 then act (unitOfRow L v) (rowOfRow L v) l else 0) = _
  exact ite_eq_left rfl

theorem splitAct_ne (L : Layout) (act : ℕ → Codes) (v : ℕ) {i' : ℕ} (h : i' ≠ 0) (l : ℕ) :
    splitAct L act v i' l = 0 := by
  show (if i' = 0 then act (unitOfRow L v) (rowOfRow L v) l else 0) = 0
  exact ite_eq_right h

/-- **The split run starts from the original's memory.** A declared global row holds the same activation row on both
layouts, and the rest of memory doesn't read the layout's units. -/
theorem init_split (salt : List ℕ) (pr : W1.Prog) (U : Workload) (act : ℕ → Codes) (σ : State) :
    W1.init salt (splitProg pr U.layout) (splitWorkload U) (splitAct U.layout act) σ = W1.init salt pr U act σ := by
  have h4 : ∀ g l : ℕ, (if g < rowsOf (splitWorkload U).layout then
        W1.w32 (splitAct U.layout act (unitOfRow (splitWorkload U).layout g) (rowOfRow (splitWorkload U).layout g) l)
      else 0) = (if g < rowsOf U.layout then W1.w32 (act (unitOfRow U.layout g) (rowOfRow U.layout g) l) else 0) := by
    intro g l
    rw [splitWorkload_layout, rowsOf_splitLayout]
    by_cases hg : g < rowsOf U.layout
    · rw [ite_eq_left hg, ite_eq_left hg, unitOfRow_splitLayout hg, rowOfRow_splitLayout hg, splitAct_zero]
    · rw [ite_eq_right hg, ite_eq_right hg]
  funext a
  simp only [W1.init]
  rw [h4]
  exact rfl

/-- **The split run is the original run:** the same code from the same memory. -/
theorem final_split (dv : W1.W1Device) (salt : List ℕ) (pr : W1.Prog) (U : Workload) (act : ℕ → Codes) (σ : State) :
    W1.final dv salt (splitProg pr U.layout) (splitWorkload U) (splitAct U.layout act) σ =
      W1.final dv salt pr U act σ := by
  show W1.exec dv false (pr.code U.layout)
      ⟨W1.init salt (splitProg pr U.layout) (splitWorkload U) (splitAct U.layout act) σ, [], 0⟩ =
    W1.exec dv false (pr.code U.layout) ⟨W1.init salt pr U act σ, [], 0⟩
  rw [init_split]

theorem w1Read_split (dv : W1.W1Device) (salt : List ℕ) (pr : W1.Prog) (U : Workload) (act : ℕ → Codes)
    (σ : State) :
    W1.w1Read dv salt (splitProg pr U.layout) (splitWorkload U) (splitAct U.layout act) σ =
      W1.w1Read dv salt pr U act σ := by
  unfold W1.w1Read
  rw [final_split]

/-- **The split readout is the original's, row by row**, on every memory: split unit `v` commits global row `v`'s
words, which are row `rowOfRow L v` of unit `unitOfRow L v`, and 0 (or the split activations, also 0) at its other
rows. -/
theorem readout_split (U : Workload) (act : ℕ → Codes) (M : W1.Mem) :
    W1.readout (splitWorkload U) (splitAct U.layout act) M = splitTranscript U.layout (W1.readout U act M) := by
  funext v
  simp only [W1.readout, splitTranscript]
  rw [splitWorkload_layout, splitLayout_N, splitLayout_m, splitLayout_k]
  by_cases hv : v < rowsOf U.layout
  · have hi := rowOfRow_lt hv
    have hu := unitOfRow_lt hv
    rw [ite_eq_left hv, ite_eq_left hv, ite_eq_left hu, Option.map_some]
    refine congrArg some ?_
    dsimp only [rowCommit]
    congr 1
    · funext i' l
      by_cases h0 : i' = 0
      · subst h0
        rw [rowCodes_zero, actAt_split]
        by_cases hl : l < (U.layout.shape (unitOfRow U.layout v)).k
        · rw [ite_eq_left (And.intro Nat.one_pos hl), ite_eq_left (And.intro hi hl)]
        · rw [ite_eq_right (not_and_of_not_right _ hl), ite_eq_right (not_and_of_not_right _ hl), splitAct_zero]
      · have h1 : ¬ i' < 1 := by omega
        rw [rowCodes_ne _ _ h0, ite_eq_right (not_and_of_not_left _ h1), splitAct_ne _ _ _ h0]
    · funext i' j
      by_cases h0 : i' = 0
      · subst h0
        rw [rowCodes_zero, ticketAt_split, ite_eq_left Nat.one_pos, ite_eq_left hi]
      · have h1 : ¬ i' < 1 := by omega
        rw [rowCodes_ne _ _ h0, ite_eq_right h1]
    · funext i' j
      by_cases h0 : i' = 0
      · subst h0
        rw [rowCodes_zero, usefulAt_split, ite_eq_left Nat.one_pos, ite_eq_left hi]
      · have h1 : ¬ i' < 1 := by omega
        rw [rowCodes_ne _ _ h0, ite_eq_right h1]
  · rw [ite_eq_right hv, ite_eq_right hv]

/-- **W1 priced is closed under splitting rows into units (C3)**, for every device and salt encoding: the split
program is the original's text at the original layout, its activations each split unit's row, and its run is the
original run, so it commits the split transcript at the same cost and no calls. -/
theorem w1Priced_splitClosed (dv : W1.W1Device) (salt : S → List ℕ) :
    (W1.w1Priced dv salt : CostModel Q R S).SplitClosed := by
  intro prog U act
  refine ⟨splitProg prog U.layout, splitAct U.layout act, fun H σ s => ⟨?_, ?_, le_rfl⟩⟩
  · show W1.readout (splitWorkload U) (splitAct U.layout act)
        (W1.final dv (salt s) (splitProg prog U.layout) (splitWorkload U) (splitAct U.layout act) σ).mem =
      splitTranscript U.layout (W1.readout U act (W1.final dv (salt s) prog U act σ).mem)
    rw [final_split dv (salt s) prog U act σ, readout_split U act]
  · refine le_of_eq ?_
    show (((W1.final dv (salt s) (splitProg prog U.layout) (splitWorkload U) (splitAct U.layout act) σ).cost +
        dv.prices.fetch * W1.w1Read dv (salt s) (splitProg prog U.layout) (splitWorkload U) (splitAct U.layout act) σ :
          ℚ) : ℝ) =
      (((W1.final dv (salt s) prog U act σ).cost + dv.prices.fetch * W1.w1Read dv (salt s) prog U act σ : ℚ) : ℝ)
    rw [final_split dv (salt s) prog U act σ, w1Read_split dv (salt s) prog U act σ]

end Pouw.PearlC.TTOutW1RowSeed
