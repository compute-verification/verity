import Proofs.Flock.Bridge.PublicToyDesc
import Proofs.Flock.Soundness.Discharge.HiddenExec.KeyH
import Proofs.Flock.Soundness.Discharge.HiddenExec.HeadlineExec

/-!
# Under hidden outputs, a returned gate is an output row's bit

A circuit `HmOut.parse` accepts untyped and in scope (`Layout.scopeOkH`) returns each output word as a bit of an output
row, a port (`ports_pos_of_outsOf`): `HmRow.check` puts every word a copy returns below `outWords`
(`Layout.check_out`), and `HmOut.checkRows` gives each such word its output row (`HiddenExec.outBitRow_ne_none`).

So no circuit file the verifier parses this way has no port and returns a gate (`outputs_noPort`). The structural
premises of `sampledKSLiveV2Z_unopened` (no port, a computed returned gate) are
never met together with their parse `hD`, `hscope` and an untyped statement, and `toyDesc` is no descriptor
`Desc.parse` reads (`toyDesc_unparsed`).
-/

namespace Flock.Bridge

open FlockSoundness FlockSoundness.Discharge

/-- **A circuit that `HmOut.parse` accepts untyped and in scope, and that returns a gate, has a port**: the gate is
bit `t` of output `j` of copy `u'` (`OutColAt`), that output's word is below `outWords` (`Layout.check_out`), and
the word is a bit of an output row (`HiddenExec.outBitRow_ne_none`). -/
theorem ports_pos_of_outsOf {tags : Tags} {cf : ByteArray} {tables : List (String × ByteArray)} {c : Flock.Circuit}
    (ht : tags.typed = false) (hc : HmOut.parse tags cf tables = .ok c) (hU : Built c.unit)
    (hscope : Layout.scopeOkH c = true) {n : ℕ} {g : Fin (Layout.keyN c hU n)} (hg : g ∈ Layout.outsOf c hU n) :
    0 < c.ports.size := by
  obtain ⟨-, u, -, -, u', j, t, hu', hj, ht16, -⟩ := (@Finset.mem_filter _ _ (Classical.decPred _) _ _).1 hg
  have hp := HiddenExec.parse_untyped ht hc
  have P := parse_facts hp.1
  obtain ⟨ls, hls⟩ := P.check
  obtain ⟨-, -, hlo⟩ := Layout.check_out P.typed (Layout.scopeOkH_spec hU hscope).2.1 hls
  have hget : c.leavesOut.getD u' #[] = c.leavesOut[u']! := by
    rw [Hidden.getElem!_eq_getD']; rfl
  have hwlt : HiddenExec.outWordOf c (u', j, t) < c.outWords := by
    rw [show HiddenExec.outWordOf c (u', j, t) = (c.leavesOut.getD u' #[])[j] from getElem!_pos _ j hj]
    refine (hlo u' hu').2 _ ?_
    rw [← hget]
    exact Array.getElem_mem_toList _
  have hne := HiddenExec.outBitRow_ne_none hp.2 (HiddenExec.rowsFit_of_scopeH ht hc hscope) u hwlt ht16
  unfold HiddenExec.outBitRow at hne
  by_contra h0
  rw [dif_neg (by omega)] at hne
  exact hne rfl

/-- **Under hidden outputs, a circuit file with no port returns no gate**: at a descriptor `Desc.parse` reads from an
untyped statement's circuit file, in scope, whose circuit has no port, the key program's circuit has no output. -/
theorem outputs_noPort {I : Discharge.Exec.Inputs} (ht : I.tags.typed = false) {D : Flock.Desc}
    (hD : Flock.Desc.parse (Discharge.ZkExec.zkTags I) I.circuitFile I.tables = .ok D) (hU : Built D.unroll.unit)
    (hscope : Layout.scopeOkH D.unroll = true) (n : ℕ) (hports : D.unroll.ports.size = 0)
    (g : Fin (Layout.keyN D.unroll hU n)) :
    g ∉ ((Placed.keyProg D.unroll hU n).circuit (Layout.outsOf D.unroll hU n)).outputs := fun hg =>
  absurd hports (Nat.pos_iff_ne_zero.1 (ports_pos_of_outsOf ((Discharge.ZkExec.zkTags_typed I).trans ht)
    (Flock.Desc.parse_flat hD) hU hscope (g := g) hg))

/-- **`toyDesc` is no descriptor the verifier parses** from an untyped statement's circuit file, in scope: it has no
port and returns a gate (`toyDesc_structural`). -/
theorem toyDesc_unparsed {I : Discharge.Exec.Inputs} (ht : I.tags.typed = false) (hU : Built toyDesc.unroll.unit)
    (hscope : Layout.scopeOkH toyDesc.unroll = true) :
    Flock.Desc.parse (Discharge.ZkExec.zkTags I) I.circuitFile I.tables ≠ .ok toyDesc := fun hD => by
  obtain ⟨hp, ⟨g, hg, -⟩, -⟩ := toyDesc_structural hU
  exact outputs_noPort ht hD hU hscope 1 hp g hg

end Flock.Bridge
