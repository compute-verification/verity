import Proofs.Flock.Soundness.Discharge.Hidden.Wide.Defs
import Proofs.Flock.Soundness.Discharge.Hidden.Binding
import Proofs.Flock.Soundness.Discharge.Hidden.Committed

/-!
# A tail that returns the outputs (`OutputsCommittedTail`, `OutputsBindingTail`)

`outputs_committed'` reads `outNet = unitNet` only for a unit per template instance (`upv_pos`), which the forced-zero
row needs (`Hm.zero_bit`); everything else it uses holds at any `outNet` (the outputs are read at `HmOut.outCol`). So
`outputs_committed_tail'` is its argument with `0 < upv` taken as given, and `outputs_binding_tail_of` is
`outputs_binding_of`'s. `outputs_committed_of_tail`: the Tail statement gives the old one.
-/

set_option Elab.async false

open Flock FlockLevel3

namespace FlockSoundness.Discharge.Hidden

open Hm

theorem outputs_committed_tail' : OutputsCommittedTail := by
  intro tags cf pf coins tables d partition program st ht hi h pop hpop hpt D hD hlt hupv hv1 L regs hregs z hz o g j hg
    hg2 hj hr
  have s := setupHidden_spec' h
  obtain ⟨hp, hcr⟩ := parse_untyped ht (s.parseF hi)
  have P := parse_facts hp
  have L0 : FlockSoundness.Layout (flatTwin st) := layout_of (st := flatTwin st) P s.pin rfl rfl s.m s.kLog
  have F0 : Aliased.UnitFacts tags (flatTwin st) := unitFacts_of_parse (st := flatTwin st) hp rfl
  have R : RowRanges st.c := RowRanges.ofL (st := flatTwin st) L0 F0.parse F0.check
  have hda : st.da = (HmRow.delta st.c st.pin).1 ++ (outPairs st.c).toArray := by
    rw [s.da, hidden_delta_unpinned (unpinned_of_v1 hcr fun p hp => (hv1 p hp).1)]
  have hrows : OutRows st.c := out_rows_v1 hp hcr fun p hp => (hv1 p hp).1
  have hz0 := sat_flat L L0 F0 hv1 hda hrows hz
  have hzero : blockBit (L.model regs) z o (zeroCol st.c) = false :=
    zero_bit L0 F0 hupv (show 0 < st.c.g by omega) hz0 o
  obtain ⟨hpl, -⟩ := hrows.2.2 j hj
  have hbc : bytesOf (blockBit (L.model regs) z o) 128 (bcCol st.c g (HmOut.nIn st.c + j)) =
      Binding.commitString
        ((HmRow.rowPrefix (st.c.ports.getD (HmOut.nIn st.c + j) default).words).data.toList ++
          bytesOf (blockBit (L.model regs) z o) (2 * (st.c.ports.getD (HmOut.nIn st.c + j) default).words)
            (Typed.msgCol st.c g (HmOut.nIn st.c + j)))
        (bytesOf (blockBit (L.model regs) z o) 192 (saltCol st.c g (HmOut.nIn st.c + j))) := by
    have := row_bc_flat L0 F0 rfl hupv hv1 hz0 o hg hpl
    rw [getElem!_eq_getD'] at this
    exact this
  rw [← msg_bytes L L0 F0 hv1 hda hrows hz o hzero hg hj, ← hbc, ← bytesAt_bc L R regs z o hg hpl]
  exact (bc_drawnH h hpop hpt hD hlt L regs hregs hz o hg hg2 hpl hr).symm

/-- **`OutputsBindingTail` from `OutputsCommittedTail`**. -/
theorem outputs_binding_tail_of (hC : OutputsCommittedTail) : OutputsBindingTail := by
  intro tags cf pf coins coins' tables d d' partition partition' program program' st st' ht hi hs hs' pop hpop hpt D D' hD
    hDp hD' hDp' hupv hv1 L regs hregs L' regs' hregs' z hz z' hz' o o' g g' j hg hg2 hg' hg2' hj hr hr' hidx hne
  have hcc : st.c = st'.c := setupHidden_c hs hs'
  have hpop' : HmOut.loadPublic tags st'.c pf = .ok pop := by rw [← hcc]; exact hpop
  have hupv' : 0 < st'.c.upv := by rw [← hcc]; exact hupv
  have hv1' : ∀ p ∈ st'.c.ports.toList, p.v2 = false ∧ p.bits = 16 * p.words ∧ 0 < p.chunks := by
    rw [← hcc]; exact hv1
  have hj' : j < st'.c.outPorts.size := by rw [← hcc]; exact hj
  have e1 := hC ht hi hs hpop hpt hD hDp hupv hv1 L regs hregs hz o hg hg2 hj hr
  have e2 := hC ht hi hs' hpop' hpt hD' hDp' hupv' hv1' L' regs' hregs' hz' o' hg' hg2' hj' hr'
  have hp : HmRow.rowPrefix (st.c.ports.getD (HmOut.nIn st.c + j) default).words =
      HmRow.rowPrefix (st'.c.ports.getD (HmOut.nIn st'.c + j) default).words := by rw [hcc]
  refine commitPair_spec' _ _ _ _ (fun e => ?_) (e1.symm.trans (Eq.trans ?_ e2))
  · rw [hp] at e
    exact hne (List.append_cancel_left e)
  · rw [hidx, hcc]

/-- `OutputsCommittedTail` gives `OutputsCommitted`: a unit that returns the outputs has a unit per template instance at
least. -/
theorem outputs_committed_of_tail (hC : OutputsCommittedTail) : OutputsCommitted := by
  intro tags cf pf coins tables d partition program st ht hi h pop hpop hpt D hD hlt ho hv1 L regs hregs z hz o g j hg
    hg2 hj hr
  have s := setupHidden_spec' h
  obtain ⟨hp, -⟩ := parse_untyped ht (s.parseF hi)
  have F0 : Aliased.UnitFacts tags (flatTwin st) := unitFacts_of_parse (st := flatTwin st) hp rfl
  exact hC ht hi h hpop hpt hD hlt (upv_pos F0.check ho) hv1 L regs hregs hz o hg hg2 hj hr

end FlockSoundness.Discharge.Hidden
