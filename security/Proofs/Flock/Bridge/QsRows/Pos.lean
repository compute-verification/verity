module

public import Proofs.Flock.Bridge.Transports
public import Proofs.Flock.Soundness.ExecCircuit
public import Proofs.Flock.Soundness.Discharge.Const.Inst
public import Proofs.Flock.Soundness.Discharge.Hidden.Delta
public import Proofs.Flock.Soundness.Discharge.Hidden.Rows
public import Proofs.Flock.Soundness.Discharge.HiddenExec.HeadlineExec
public import Proofs.Flock.Soundness.Discharge.HiddenExec.KeyH
public import Proofs.Flock.Soundness.Discharge.HiddenExec.Setup
public import Proofs.Flock.Soundness.Discharge.Layout.KeyProg
public import Proofs.Flock.Soundness.Discharge.Layout.KeysRow
public import Proofs.Flock.Soundness.Discharge.Layout.OfRows
public import Proofs.Flock.Soundness.Discharge.Layout.OutPin
public import Proofs.Flock.Soundness.Discharge.Layout.OutShape
public import Proofs.Flock.Soundness.Discharge.Layout.Outs
public import Proofs.Flock.Soundness.Discharge.Layout.Pins
public import Proofs.Flock.Soundness.Discharge.Layout.Readers
public import Proofs.Flock.Soundness.Discharge.Layout.Scope
public import Proofs.Flock.Soundness.Discharge.ZkBind.BindX
public import Proofs.Flock.Soundness.Discharge.ZkExec.Event
public import Proofs.Flock.Soundness.Discharge.ZkHidden.DefsJ

/-!
# The commit strings a proof block's wires sit at, in the session's value binding

The session's value binding (`vbOf`) puts a committed wire at its row's commit string when it is a row's bit
(`vbOf_pos_row`) and at a commit string of its own otherwise (`vbOf_pos_none`). At the circuit the session's parse
accepts, the only committed wires of a proof block that are no row's bit are the zero's gate and the constant's:

* `vbOf_pos_none`: a wire that is no row's bit sits at its own commit string.
* `vbOf_pos_io`: a committed wire of proof block `u` sits at one of `u`'s rows, at the zero's gate or at the
  constant's. So a proof block's wires sit at no more than its `c.ports.size` rows and those two gates, the same two for
  every proof block.

The circuit's facts come from the parse and the scope check, with nothing read from a site's acceptance: the output
rows' checks (`Hidden.rowsOut_of_checkRows`), that each returned output is an output word (`Layout.check_out`), the
rows' fit (`rowsFit_of_scopeH`) and the keys' rows (`Layout.keysRow_of_parse`).
-/

@[expose] public section

namespace Flock.Bridge

open Claims Claims.Sampling
open FlockSoundness.Refine FlockSoundness.Binding

section QsPos

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Discharge.ZkLink
  FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed FlockSoundness.Discharge.ZkHidden
  FlockSoundness.Discharge.ZkBind FlockSoundness.Discharge FlockSoundness.Discharge.Composed

variable {n mPts : ℕ} {Reg : Type} {I : Inputs} {dj : Finset (Fin n) → Lean.Json} {S₀ : Finset (Fin n)}
  {d₀ : Lean.Json} {y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀} {ht : I.tags.typed = false} {c : Flock.Circuit}
  {hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c} {hscope : Discharge.Layout.scopeOkH c = true}
  {hU : Built c.unit} {pub : Flock.Public} {hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub}
  {hpt : pub.tables = none}
  {g : Fin ((keyProg c hU n).circuit (Discharge.Layout.outsOf c hU n)).N}

/-- **A wire that is no row's bit sits at its own commit string.** -/
theorem vbOf_pos_none (h : HiddenExec.rowOfH c hU n (Discharge.Layout.outsOf c hU n) g = none) :
    (vbOf dj y₀ ht hc hscope hU hpub hpt (Reg := Reg)).pos g = (Sum.inr g : PosH c hU n) := by
  dsimp only [vbOf, HmRowsZX.bindingZ, LayoutZ.bindingZ, ZkHidden.rsZHJ, ZkHidden.layZHJ, LayoutZ.ofShadow,
    ZkHidden.layShHJ, HiddenExec.keyLayoutH, Discharge.Layout.Readers.layout, Discharge.Layout.ofBits,
    HiddenExec.keyReadersH]
  exact Discharge.Layout.ofRows_pos_none _ h

/-- **A committed wire of proof block `u` sits at one of `u`'s rows, at the zero's gate or at the constant's.** A
committed wire is on a column of the block's instance (`Layout.exists_col_of_io`); one that is no row's bit is the zero,
the constant or a gate the block computes (`Layout.single_gate`), and a computed one is a returned output's bit
(`Layout.outsOut_outsOf`), so an output row's (`HiddenExec.outBitRow_ne_none_of_rowsOut`). -/
theorem vbOf_pos_io (u : Fin n)
    (hg : g ∈ ((keyProg c hU n).partition (Discharge.Layout.outsOf c hU n)).io u) :
    (∃ q : Fin c.ports.size,
        (vbOf dj y₀ ht hc hscope hU hpub hpt (Reg := Reg)).pos g = (Sum.inl (u, q) : PosH c hU n)) ∨
      (vbOf dj y₀ ht hc hscope hU hpub hpt (Reg := Reg)).pos g =
        (Sum.inr (Discharge.Layout.zeroG (c := c) (hU := hU) (n := n)) : PosH c hU n) ∨
      (vbOf dj y₀ ht hc hscope hU hpub hpt (Reg := Reg)).pos g =
        (Sum.inr (Discharge.Layout.oneG (c := c) (hU := hU) (n := n)) : PosH c hU n) := by
  have ht' : (zkTags I).typed = false := (zkTags_typed I).trans ht
  have hrow := FlockSoundness.Discharge.HiddenExec.parse_untyped ht' hc
  have hsc := Discharge.Layout.scopeOkH_spec hU hscope
  have hrf : FlockSoundness.Discharge.Hidden.Any.RowsFitS c :=
    FlockSoundness.Discharge.HiddenExec.rowsFit_of_scopeH ht' hc hscope
  have hkeys : Discharge.Layout.KeysRow c hU (Discharge.Layout.portRowW c) :=
    Discharge.Layout.keysRow_of_parse hrow.1 hsc.1 hsc.2.2
  have hcr : FlockSoundness.Discharge.Hidden.RowsOut c := FlockSoundness.Discharge.Hidden.rowsOut_of_checkRows hrow.2
  have Pf := FlockSoundness.parse_facts hrow.1
  obtain ⟨ls, hls⟩ := Pf.check
  have hlo : ∀ u' < c.upv, ∀ o ∈ (c.leavesOut[u']!).toList, o < c.outWords :=
    fun u' hu' => ((Discharge.Layout.check_out Pf.typed hsc.2.1 hls).2.2 u' hu').2
  have houts := Discharge.Layout.outsOut_outsOf c hU n
  obtain ⟨x, hx⟩ := Discharge.Layout.exists_col_of_io
    (Discharge.Const.Prog.instC (keyProg c hU n) (Discharge.Layout.outsOf c hU n) u) hg
  rcases hr : HiddenExec.rowOfH c hU n (Discharge.Layout.outsOf c hU n) g with _ | ⟨u', q, k⟩
  · right
    rw [vbOf_pos_none hr]
    obtain ⟨hk, hno⟩ := HiddenExec.rowOfH_none (Discharge.Layout.outsOf c hU n) hr
    rcases Discharge.Layout.single_gate (Discharge.Layout.outsOf c hU n) hkeys ⟨x, hx⟩ hg hk with
      h0 | h1 | ⟨hm, -, -⟩
    · exact Or.inl (by rw [h0])
    · exact Or.inr (by rw [h1])
    · exfalso
      subst hx
      obtain ⟨u'', j, t, hoc⟩ := houts u _ hg hm x rfl
      have hno' : HiddenExec.outRowOf c hU n (Discharge.Layout.outsOf c hU n)
          ((Discharge.Const.Prog.instC (keyProg c hU n) (Discharge.Layout.outsOf c hU n) u).wire x) = none := hno
      rw [HiddenExec.outRowOf_wire (Discharge.Layout.outsOf c hU n) x hm
        (HiddenExec.unit_wire (Discharge.Layout.outsOf c hU n) x hm).2, Discharge.Layout.outDecode_of hoc] at hno'
      obtain ⟨hu', hj, htb, -⟩ := hoc
      have hget : c.leavesOut.getD u'' #[] = c.leavesOut[u'']! := by
        rw [FlockSoundness.Discharge.Hidden.getElem!_eq_getD']; rfl
      have hwlt : HiddenExec.outWordOf c (u'', j, t) < c.outWords := by
        rw [show HiddenExec.outWordOf c (u'', j, t) = (c.leavesOut.getD u'' #[])[j] from getElem!_pos _ j hj]
        refine hlo u'' hu' _ ?_
        rw [← hget]
        exact Array.getElem_mem_toList _
      exact HiddenExec.outBitRow_ne_none_of_rowsOut hcr hrf u hwlt htb hno'
  · left
    subst hx
    have hu := HiddenExec.rowOfH_wire_unit (Discharge.Layout.outsOf c hU n) x hr
    rw [hu] at hr
    exact ⟨q, vbOf_pos_row hr⟩

end QsPos

end Flock.Bridge
