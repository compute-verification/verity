import Proofs.Flock.Soundness.Discharge.Hidden.Facts

/-!
# The row commitment binds an output row (`cr/sha-512`)

`commitPair_spec`: two different rows' bytes with one commit string `b ‖ c` give a SHA-512 collision, in the row hash when
the salts agree (the masks then cancel) and in the salt digest when they differ. `outputs_binding_of`: two sessions that
`OutputsCommitted` covers, on one circuit file, public file and tables, read one registered `b ‖ c` for one unit's output
row, so when their template instances' output words differ, `commitPair` of the two rows is a collision.
-/

open Flock FlockLevel3

namespace FlockSoundness.Discharge.Hidden

open Refine Merkle

theorem commitPair_spec' : CommitPairSpec := by
  intro r y r' y' hr h
  unfold Binding.Collides commitPair
  split_ifs with hy
  · have hb := (Binding.commitString_split h).1
    rw [hy] at hb
    have hx := xorBytes_inj (by rw [hL512, K512.mask_len]) (by rw [hL512, K512.mask_len]) hb
    exact ⟨hr, enc512.dig_inj hx⟩
  · exact ⟨fun e => hy (List.append_cancel_left e), enc512.dig_inj (Binding.commitString_split h).2⟩

/-- Two statements `Stmt.setupHidden` accepts on one circuit file and tables have one circuit. -/
theorem setupHidden_c {tags : Tags} {cf pf pf' : ByteArray} {coins coins' : Coins} {tables : List (String × ByteArray)}
    {draw draw' partition partition' : Option Lean.Json} {program program' : Option ByteArray} {st st' : Stmt}
    (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st)
    (h' : Stmt.setupHidden tags cf pf' coins' tables draw' partition' program' = .ok st') : st.c = st'.c :=
  Except.ok.inj ((setupHidden_spec' h).parse.symm.trans (setupHidden_spec' h').parse)

/-- **`OutputsBinding` from `OutputsCommitted`**. -/
theorem outputs_binding_of (hC : OutputsCommitted) : OutputsBinding := by
  intro tags cf pf coins coins' tables d d' partition partition' program program' st st' ht hi hs hs' pop hpop hpt D D' hD
    hDp hD' hDp' ho hv1 L regs hregs L' regs' hregs' z hz z' hz' o o' g g' j hg hg2 hg' hg2' hj hr hr' hidx hne
  have hcc : st.c = st'.c := setupHidden_c hs hs'
  have hpop' : HmOut.loadPublic tags st'.c pf = .ok pop := by rw [← hcc]; exact hpop
  have ho' : st'.c.outNet = st'.c.unitNet := by rw [← hcc]; exact ho
  have hv1' : ∀ p ∈ st'.c.ports.toList, p.v2 = false ∧ p.bits = 16 * p.words ∧ 0 < p.chunks := by
    rw [← hcc]; exact hv1
  have hj' : j < st'.c.outPorts.size := by rw [← hcc]; exact hj
  have e1 := hC ht hi hs hpop hpt hD hDp ho hv1 L regs hregs hz o hg hg2 hj hr
  have e2 := hC ht hi hs' hpop' hpt hD' hDp' ho' hv1' L' regs' hregs' hz' o' hg' hg2' hj' hr'
  have hp : HmRow.rowPrefix (st.c.ports.getD (HmOut.nIn st.c + j) default).words =
      HmRow.rowPrefix (st'.c.ports.getD (HmOut.nIn st'.c + j) default).words := by rw [hcc]
  refine commitPair_spec' _ _ _ _ (fun e => ?_) (e1.symm.trans (Eq.trans ?_ e2))
  · rw [hp] at e
    exact hne (List.append_cancel_left e)
  · rw [hidx, hcc]

end FlockSoundness.Discharge.Hidden
