import Proofs.Flock.Claims.Recursion.OpenSlots

/-!
# The row a stream round opens is its first holder's

`finalR` cuts the last message at `layR elem` back into each round's openings, slot by slot (`opensSlots_layR`'s
first half), so each round's opened rows are its slots' rows read off their parts' witnesses (`os_rows`). A stream
round's slots are its tops' instances, then its row's slot (`roundSlots`), so the row past the tops' rows is the
round's `rowSlot`'s (`opened_row`).
-/

namespace RecDeploy.OpenSlots

open Flock
open Claims (Msg)
open Claims.Recursion.VStarB
open RecDeploy.DeployedR
open RecDeploy.RestRead

/-- **The opened rows are the slots' rows**: wherever `finalR` reads the last message at `layR elem` as `(os, last)`,
round `j`'s opened rows are round `j`'s slots' rows, each off its part's witness. -/
theorem os_rows (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) {x : Msg}
    (hx : DenseX x) {ws : List Claims.Flock.Wit} (hl : ws.length = nS x) (d : Claims.Flock.Wit) {os last}
    (hf : finalR (((DeployedR.layR elem).msgs x ws).getD (kI x - 1) []) = some (os, last)) :
    ∃ r, readX x = some r ∧ os.map rowsR =
      (slotRounds r).map fun l => l.map fun σ => (⟨(readForms (ws.getD σ.s d) σ.row).toArray⟩ : ByteArray) := by
  obtain ⟨r, hr, hlast, -⟩ := lastAt_parts elem hx hl
  have hn : nS x = r.w.parts.length := by simp [nS, hr]
  obtain ⟨L, hL⟩ : ∃ L : List (List (Msg × Msg)),
      L = (slotRounds r).map fun l => l.map fun σ => σ.opening (ws.getD σ.s d) := ⟨_, rfl⟩
  have hpl : ∀ p ∈ openPieces r, p.1 < ws.length := fun p hp => by
    rw [hl, hn]
    exact openPieces_lt hr p hp
  have hlens : openLens r = L.map fun l => l.map (·.1.length) := by
    rw [openLens_eq, hL, List.map_map]
    refine List.map_congr_left fun l _ => ?_
    simp only [Function.comp_apply, List.map_map]
    refine List.map_congr_left fun σ _ => ?_
    simp [Slot.opening, readForms_length, pieceLen]
  have hbytes : readPieces (openPieces r) ws = L.flatMap (·.flatMap fun p => p.1 ++ p.2) := by
    rw [openBytes r ws d hpl, hL, List.flatMap_map]
    refine flatMap_congr_mem _ fun l _ => ?_
    simp only [List.flatMap_map]
    exact flatMap_congr_mem _ fun σ _ => readSegs_segs _ σ
  have hsalt : ∀ l ∈ L, ∀ p ∈ l, p.2.length = 192 := by
    intro l hl p hp
    rw [hL] at hl
    obtain ⟨l', -, rfl⟩ := List.mem_map.1 hl
    obtain ⟨σ, -, rfl⟩ := List.mem_map.1 hp
    exact HmOpen.length_salt _ _ _ _
  have hf' : finalR (lastAt (restR elem) x ws) = some (os, last) := hf
  rw [hlast, finalR_trailer, hlens, hbytes, CutOpens.cutRounds_groups _ L hsalt] at hf'
  obtain ⟨rfl, -⟩ := Prod.mk.inj (Option.some.inj hf')
  refine ⟨r, hr, ?_⟩
  rw [hL]
  simp [rowsR, Slot.opening, List.map_map, Function.comp_def]

/-- A stream round's tops' slots number its tops' instances. -/
theorem capSlots_flat_length (r : XR) (rs : Array RecVStar.SchedRound) (j : ℕ) :
    ((List.range ((rs.getD (j % rs.size) default).2.1).size).flatMap (capSlotsAt r rs j)).length =
      ((rs.getD (j % rs.size) default).2.1).toList.sum := by
  rw [List.length_flatMap]
  have e : (List.range ((rs.getD (j % rs.size) default).2.1).size).map (fun e => (capSlotsAt r rs j e).length) =
      ((rs.getD (j % rs.size) default).2.1).toList := by
    apply List.ext_getElem (by simp)
    intro i h1 h2
    have hi : i < ((rs.getD (j % rs.size) default).2.1).size := by simpa using h1
    have hi' : i < (rs[j % rs.size]?.getD default).2.1.size := by simpa [Array.getD_eq_getD_getElem?] using hi
    simp [capSlotsAt, capSlots, Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hi']
  rw [e]

/-- **A stream round's row slot sits past its tops' slots.** -/
theorem roundSlots_row {r : XR} {rs : Array RecVStar.SchedRound} {j : ℕ} {σ : Slot} (h : rowSlot r j = some σ) :
    (roundSlots r rs j)[((rs.getD (j % rs.size) default).2.1).toList.sum]? = some σ := by
  unfold roundSlots
  rw [h, Option.toList_some, List.getElem?_append_right (le_of_eq (capSlots_flat_length r rs j)),
    capSlots_flat_length, Nat.sub_self]
  rfl

/-- **Stream round `j`'s opened row is its `rowSlot`'s**: wherever `finalR` reads the last message at `layR elem` as
`(os, last)`, round `j + 1`'s opened row past its tops' rows is the row of stream round `j`'s first holder's slot, off
that holder's witness. -/
theorem opened_row (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) {x : Msg}
    (hx : DenseX x) {ws : List Claims.Flock.Wit} (hl : ws.length = nS x) (d : Claims.Flock.Wit) {os last}
    (hf : finalR (((DeployedR.layR elem).msgs x ws).getD (kI x - 1) []) = some (os, last)) {r : XR}
    (hr : readX x = some r) {rs : Array RecVStar.SchedRound} (hrs : repSched r.shape = some rs) {j : ℕ}
    (hj : j < 2 * rs.size) {σ : Slot} (hσ : rowSlot r j = some σ) :
    (rowsR (os.getD (j + 1) [])).getD ((rs.getD (j % rs.size) default).2.1).toList.sum default =
      (⟨(readForms (ws.getD σ.s d) σ.row).toArray⟩ : ByteArray) := by
  obtain ⟨r', hr', hos⟩ := os_rows elem hx hl d hf
  rw [hr] at hr'
  obtain rfl := Option.some.inj hr'
  obtain ⟨N, hc, -⟩ := capAt_readX hr
  have hSR : slotRounds r =
      capSlots r (nS x - 1) (List.range N) :: (List.range (2 * rs.size)).map (roundSlots r rs) := by
    simp [slotRounds, hc, hrs]
  have e : rowsR (os.getD (j + 1) []) = (os.map rowsR).getD (j + 1) [] := by
    rw [List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD, List.getElem?_map]
    cases os[j + 1]? <;> rfl
  rw [e, hos, hSR]
  simp only [List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_cons_succ, List.getElem?_range hj,
    Option.map_some, Option.getD_some, roundSlots_row hσ]

end RecDeploy.OpenSlots
