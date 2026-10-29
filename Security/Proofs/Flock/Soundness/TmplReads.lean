import Proofs.Flock.Soundness.TmplUnrollAt
import Proofs.Flock.Soundness.SparseRows
import Proofs.Flock.C3.Reps.ReadWhole
import Flock.TmplReads
import Flock.KnownRead

/-!
# A templated unit's surely read columns are read by its committed net

`Tmpl.readsSure` marks, from the template's description, input columns that a row of `Tmpl.rows` from the input rows to
the constant reads (`Flock.TmplReads.rows_sure`). On the committed net (`Tmpl.unitNet`, `Tmpl.unrollAt`'s under
`unrollOk`) each is a column `RecResiduals.readIns` marks (`readsSure_readIns`): the input rows stay at their columns, and
a row past them is placed past them and before the constant. So where `KnownRead.readsOk` checks a templated unit's known
bits at the columns `KnownRead.sureReads` gives, it accepts only what it would at `readIns` (`readsAt_of_readsOk`).
-/

open Flock Flock.Tmpl

namespace FlockSoundness.TmplFold

/-- **Every column a template's description surely reads is one its committed net's computed rows read.** -/
theorem readsSure_readIns {t : Tmpl} {mode : Mode} {lays : Array Lay} (hL : layoutM t mode = .ok lays)
    (h : unrollOk t lays = true) (hlog : (unitNet t lays).unitLog ≤ 32) {y : Nat}
    (hy : t.readsSure.getD y false = true) : (RecResiduals.readIns (unitNet t lays)).getD y false = true := by
  have hu := unitNet_eq hL h
  generalize unitNet t lays = net at hu hlog ⊢
  unfold Tmpl.unrollAt at hu
  obtain ⟨acc, hr, h2⟩ := TmplOrder.bind_ok hu
  obtain ⟨n0, hn0, -, hcp, -, ha, hb, -, -, hiw, -, -, hpos, -, -⟩ := Net.ofRowsAt_ok h2
  obtain ⟨-, -, -, -, -, -, -, -, -, -, -, -, -, hw0, -⟩ := Net.ofRows_ok hn0
  have hinc : n0.inWords * WORD = t.inCols := by rw [hw0]; rfl
  have hw : 2 ^ net.unitLog ≤ 2 ^ 32 := Nat.pow_le_pow_right (by decide) hlog
  obtain ⟨hy1, hy2, r, hr1, hr2, hyr⟩ := TmplReads.rows_sure hr hy
  generalize Tmpl.posMap t lays = pos at hcp ha hb hpos
  generalize Net.invOf pos (pos.getD t.constPos 0 + 1) = inv at ha hb hpos
  generalize hC : (t.defs.getD t.root default).cols = cols at hpos
  rw [hinc] at hpos
  have hK : t.constPos = cols - 1 := by unfold Tmpl.constPos; rw [hC]
  have hrc : r < cols := by omega
  have hKc : t.constPos < cols := by omega
  have hpast : ∀ x < cols, t.inCols ≤ x → t.inCols ≤ pos.getD x 0 := fun x hx hix => Nat.le_of_not_lt fun hlt => by
    have e1 := Net.posOk_inputs hpos hlt
    have := Net.posOk_inj hpos (by omega) hx e1
    omega
  have hrK : pos.getD r 0 < pos.getD t.constPos 0 := by
    have h1 := Net.posOk_lt hpos hrc
    have h2 : pos.getD r 0 ≠ pos.getD t.constPos 0 := fun e => by
      have := Net.posOk_inj hpos hrc hKc e
      omega
    omega
  have hK1 := hpast t.constPos hKc (by omega)
  have hr3 := hpast r hrc hr1
  rw [FlockC3.Reps.ReadWhole.getD_readIns, hiw, hcp, hinc]
  refine ⟨Nat.lt_min.2 ⟨hy1, by omega⟩, pos.getD r 0, (Nat.min_le_left _ _).trans hr3, hrK, ?_⟩
  rw [FlockSoundness.ofRows_row ha hw, FlockSoundness.ofRows_row hb hw, Net.placed_at hpos _ hrc,
    Net.placed_at hpos _ hrc, Array.toList_map, Array.toList_map]
  have hyy : pos.getD y 0 = y := Net.posOk_inputs hpos hy1
  rcases hyr with hyr | hyr
  · exact List.mem_append_left _ (List.mem_map.2 ⟨y, hyr, hyy⟩)
  · exact List.mem_append_right _ (List.mem_map.2 ⟨y, hyr, hyy⟩)

/-- **The columns `KnownRead.sureReads` gives are read by the unit's computed rows.** -/
theorem sureReads_sound {c : Circuit} {rd : Array Bool} (h : KnownRead.sureReads c = some rd) {y : Nat}
    (hy : rd.getD y false = true) : (RecResiduals.readIns c.unit).getD y false = true := by
  unfold KnownRead.sureReads at h
  split at h
  next tu _ =>
    split at h
    next hc =>
      cases h
      simp only [Bool.and_eq_true, decide_eq_true_eq] at hc
      obtain ⟨hlog, hp⟩ := hc
      have he : c.unit = tu.net := by
        unfold withPtrEq at hp
        simpa using hp
      rw [he, tu.isNet] at hlog ⊢
      exact readsSure_readIns tu.laid tu.ok hlog hy
    next => cases h
  next => cases h

/-- **What `KnownRead.readsOk` accepted, at columns the unit's computed rows read**: `readsOkAt` at some `rd` every
column of which `readIns` marks. -/
theorem readsAt_of_readsOk {c : Circuit} {known : Array (String × Array (Array Nat × ByteArray))}
    (h : KnownRead.readsOk c known = .ok ()) :
    ∃ rd, KnownRead.readsOkAt c known rd = .ok () ∧
      ∀ y, rd.getD y false = true → (RecResiduals.readIns c.unit).getD y false = true := by
  unfold KnownRead.readsOk at h
  cases hs : KnownRead.sureReads c with
  | none =>
    simp only [hs] at h
    exact ⟨_, h, fun _ hy => hy⟩
  | some rd =>
    simp only [hs] at h
    cases ha : KnownRead.readsOkAt c known rd with
    | ok v => exact ⟨rd, ha, fun _ hy => sureReads_sound hs hy⟩
    | error e =>
      simp only [ha] at h
      exact ⟨_, h, fun _ hy => hy⟩

end FlockSoundness.TmplFold
