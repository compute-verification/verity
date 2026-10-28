import Proofs.Flock.Claims.Recursion.ResFrameElems
import Proofs.Flock.Claims.Recursion.ResCopyRows
import Proofs.Flock.Claims.Recursion.ResLowerDD
import Proofs.Flock.Claims.Recursion.ResHeldRows
import Proofs.Flock.Claims.Recursion.ResOpenedRow
import Proofs.Flock.Claims.Recursion.VVTieOf
import Proofs.Flock.Claims.Recursion.VerdictLoop
import Proofs.Flock.Claims.Recursion.VerdictRowDTie
import Proofs.Flock.VBridge.Family.PartKeys

/-!
# P2 at the deployed constants (`framePorts_D`)

`FramePorts` at the deployed layout (`DeployedR.layR elem`) and its final reader (`DeployedR.finalR`), at the dense
form, `recRead` and `cmtTr recRead`: in a piece's port context, at a decision `D` of the opened messages, a `rec-residuals`
copy's row element `e` at a frame value's `w` index `j` is what `Rows.elemReadC acc` reads there, at every `acc`.

The element is group `p`'s port `i` at offset `o` (`sum_decomp`), port `i` being `m<k>`, `rounds`' `k`-th round:
* the copy reads it off the part's own witness at the port's row (`copy_rowElems`);
* that row is the holder's `rowSlot` row of round `rep · R + t`, `t` the `k`-th committing round (`mem_heldRows_of`,
  `held_rowSlot`), which is the opened round `rep · R + t + 1`'s row at the round's tops (`opened_row`);
* that opened row holds the values chunk `t` of the decided proof absorbs (`DecAt.rows_getD`, `rowsAt_elem`), which
  `framesRead` reads at `j` (`DecAt.frame_elem`), and `elemReadC` reads `framesRead` below the accumulators
  (`Rows.elemReadC_frame`).
-/

namespace RecDeploy.ResLower

open Flock
open Claims (Msg)
open FlockDeploy (Coin FinalP setupOf)
open Claims.Recursion.VStarB
open Flock.RecAlgebra (rounds)

/-- **P2 at the deployed constants**: `FramePorts` at the deployed layout and final reader, the dense form, `recRead`
and `cmtTr recRead`, at every reader `rd`, wire weights, brings and element reader. -/
theorem framePorts_D (rd : Msg → Option FinalP) (wE w : Msg → ℝ) (b : Brings)
    (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) :
    FramePorts rd wE DeployedR.finalR w b elem Claims.Recursion.Form.Dense recRead (cmtTr recRead) := by
  intro x st hx hf c ws hJ s wt hw r hr ss hss j nm pt hsv xO p hxO hck hsh hAg hT hc q hq ho ht m0 k0 cl0 jj0 sh
    holo S ns nv hpar hshp hholo hpart os last hfin hos hD acc e jw hj hjl
  -- the inner schedule
  have hDx : RestRead.DenseX x := ⟨st, hx, hf⟩
  obtain ⟨r5, rs, s5, spec5, st5, hr5, hrs5, hsu5, -, hsched5, hK⟩ := RestRead.schedBridge hDx
  have e5 : r5 = r := Option.some.inj (hr5.symm.trans hr)
  rw [e5] at hrs5
  have hos' : os.length + 1 = 2 + 2 * rs.size := by
    rw [← hK]
    exact hos
  obtain _ | ⟨o₀, ot⟩ := os
  · rw [List.length_nil] at hos'
    omega
  -- the staged statement: one table, `p` its parse
  obtain ⟨-, i0, spec, sts, -, hsu, -⟩ := hsh.2
  have hpe : p = { i0.base.read with sts } := Option.some.inj (hxO.symm.trans (parse_of_setupO hsu))
  subst hpe
  -- the piece's part is part `s`'s statement
  obtain ⟨r1, ss1, nm1, pt1, hr1, hss1, hsv1⟩ := DecompBR.ownRD_some ho
  have e1 : r1 = r := Option.some.inj (hr1.symm.trans hr)
  rw [e1, hss] at hss1
  have e2 : ss1 = ss := (Except.ok.inj hss1).symm
  rw [e2, hsv] at hsv1
  simp only [Option.some.injEq, VStar.Stmt.inner.injEq] at hsv1
  obtain ⟨hjq, -, -⟩ := hsv1
  subst hjq
  -- the unit name is the read shape's, at group `q.key.1`
  obtain ⟨p', r'', sv, hx', hty, hpi, hr'', ⟨ss'', hss'', hsv''⟩, -, htab⟩ := stagedTables hsh.2
  obtain rfl : p' = { i0.base.read with sts } := Option.some.inj (hx'.symm.trans hxO)
  have hsv3 : sv = .inner q.key.1 nm pt := by
    have e1 : r'' = r := Option.some.inj (hr''.symm.trans hr)
    rw [e1, hss] at hss''
    have e2 : ss'' = ss := (Except.ok.inj hss'').symm
    rw [e2, hsv] at hsv''
    exact (Option.some.inj hsv'').symm
  have hname : sts[0].c.unit.name = nm := by
    have h := (htab 0 ht).1
    rw [hsv3] at h
    exact h
  have hn : RecResiduals.unitName m0 k0 cl0 jj0 =
      RecResiduals.unitName r.shape.m r.shape.kLog r.shape.claims q.key.1 :=
    (FlockVBridge.Family.resUnitName_params hpar).trans (hname.trans (DecompBR.inner_named hss hsv))
  obtain ⟨rfl, rfl, rfl, rfl⟩ := FlockVBridge.Family.resUnitName_inj hn
  -- table 0's part is `shI`'s, plain
  obtain ⟨v, hv, p0, r0, S0, ns0, nv0, hr0, hpart0, hk, hqS⟩ := DecompBR.mem_resPD' hq
  obtain ⟨r3, ss3, sh0, hr3, -, hh, hsh0, hI, -⟩ := DecompBR.shI_tie (x := x) hDx hv
  have e3 : r3 = r := Option.some.inj (hr3.symm.trans hr)
  rw [e3] at hh hsh0 hI
  have hshe : sh0 = sh := Option.some.inj (hsh0.symm.trans hshp)
  rw [hshe] at hI
  have hpl := staged_plain (x := x) (st0 := st) hx hf hsh hsu 0 ht
  have hho : holo = false := Except.ok.inj (hholo.symm.trans hpl)
  subst hho
  have hI' : DecompBR.shI v.2.1 v.2.2 = { sh with holo := false } := by
    rw [hI, hh]
  have hpart' : RecResiduals.part (DecompBR.shI v.2.1 v.2.2) q.key.1 = .ok (S, ns, nv) := by
    rw [hI']
    exact hpart
  -- the key: rep `rf`, unit 0
  have hu0 : q.key.2.2.2 = 0 := by simp [hk]
  obtain ⟨rf, hrf⟩ : ∃ rf : Fin 2, (rf : ℕ) = r0 := ⟨⟨r0, hr0⟩, rfl⟩
  subst hrf
  have hn1 : q.key.2.1 + q.key.2.2.1 = (rf : ℕ) := by simp [hk]
  have hn0 : q.key.2.1 + q.key.2.2.1 = 2 * 0 + (rf : ℕ) := by omega
  -- the part's circuit is table 0's
  obtain ⟨cC, hpc⟩ := staged_circuit hsh.2 hr
  have hc0 := partCircuit_staged hsh.2 hxO hty hpi hr hpc ht
  have hpar' : RecResiduals.params cC.unit.name = some (r.shape.m, r.shape.kLog, r.shape.claims, q.key.1) := by
    rw [← hc0]
    exact hpar
  have hholo' : RecResiduals.holoOf cC = .ok false := by
    rw [← hc0]
    exact hholo
  -- the decision
  obtain ⟨D⟩ := hD
  have hv5 : v = (s5, spec5, #[st5]) := Option.some.inj (hv.symm.trans hsu5)
  obtain ⟨hT1, -⟩ := D.tables_of_dense hx hf rfl
  have h0 : 0 < D.sts.size := by omega
  have hG := D.rowG Top.cls₀ (fun _ => #[]) hx hf rfl h0
  obtain ⟨hsp, hsts⟩ := D.setup_eq
  obtain ⟨v', c₁, r₀, rss, fl, link, pf, hv', -, -, -, -, -, -, hd⟩ := Rows.decR_last D.dec
  have hvv : v' = v := Option.some.inj (hv'.symm.trans hv)
  have hds0 : D.d.spec = v'.2.1 := by rw [hd]
  have hdt0 : D.d.sts = v'.2.2 := by rw [hd]
  have hds : D.d.spec = spec5 := by rw [hds0, hvv, hv5]
  have hdt : D.d.sts = #[st5] := by rw [hdt0, hvv, hv5]
  have hshX : D.sh Top.cls₀ 0 = Rows.shapeNoCoin v.2.1 v.2.2 := by
    rw [D.sh_eq Top.cls₀ (fun _ => #[]) hG h0, hsp, hsts, hds0, hdt0, hvv]
  have hshI := shI_alg (xo := x) (sto := st) D hx hf rfl hv
  have hRI : rounds (DecompBR.shI v.2.1 v.2.2) = rounds (Rows.shapeNoCoin v.2.1 v.2.2) := by
    have h : rounds (Rows.algShape (Rows.shapeNoCoin v.2.1 v.2.2)) = rounds (Rows.shapeNoCoin v.2.1 v.2.2) :=
      OpenLower.rounds_congr rfl
    rw [hshI]
    all_goals exact h
  have hsz : RecResiduals.portSizes (DecompBR.shI v.2.1 v.2.2) =
      (rounds (Rows.shapeNoCoin v.2.1 v.2.2)).map (·.size) ++ #[2, 2] := by
    unfold RecResiduals.portSizes
    rw [hRI]
  -- group `q.key.1`'s ports
  obtain ⟨S1, gs, Rg, P, Vg, hS1, hgs, hgj, -⟩ := part_partS hpart'
  obtain ⟨hgp, hgP⟩ := Array.getElem?_eq_some_iff.1 hgj
  have hns := (FlockVBridge.OfVStar.part_eq hS1 hgs hgp hpart').2.1
  rw [hgP] at hns
  dsimp only at hns
  subst hns
  have hPP : RecVStar.partPorts { sh with holo := false } q.key.1 = .ok P := by
    rw [← hI', FlockVBridge.OfVStar.partPorts_eq hS1 hgs hgp, hgP]
  -- element `e` is port `i`'s offset `o`
  have hje : (FlockVBridge.PartS.wIdx (RecResiduals.portSizes (DecompBR.shI v.2.1 v.2.2)) P)[e]? = some jw :=
    (elemPort_eq hv hshI hS1 hgs hgj e).symm.trans hj
  obtain ⟨hes, hje'⟩ := Array.getElem?_eq_some_iff.1 hje
  obtain ⟨i, hi, o, hon, rfl⟩ := ResFrames.sum_decomp
    (P.toList.map ((RecResiduals.portSizes (DecompBR.shI v.2.1 v.2.2))[·]!)) e
    (by rw [FlockVBridge.OfVStar.wIdx_size] at hes; exact hes)
  have hiP : i < P.size := by simpa using hi
  have hnsi : (P.toList.map ((RecResiduals.portSizes (DecompBR.shI v.2.1 v.2.2))[·]!))[i] =
      (RecResiduals.portSizes (DecompBR.shI v.2.1 v.2.2))[P[i]]! := by
    simp
  have hon' : o < (RecResiduals.portSizes (DecompBR.shI v.2.1 v.2.2))[P[i]]! := by
    rw [← hnsi]
    exact hon
  have hwg := FlockVBridge.OfVStar.wIdx_get _ P hiP hon' hes rfl
  have hjl' : jw < (rounds (Rows.shapeNoCoin v.2.1 v.2.2)).flatten.size := by
    have h := hjl
    unfold accBase at h
    rw [hv] at h
    exact h
  -- port `i` is a round's: `m<P[i]>`
  have hkk : P[i] < (rounds (Rows.shapeNoCoin v.2.1 v.2.2)).size := by
    refine Nat.lt_of_not_le fun hge => ?_
    by_cases hle : P[i] ≤ (RecResiduals.portSizes (DecompBR.shI v.2.1 v.2.2)).size
    · have h1 : jw = (RecResiduals.starts (RecResiduals.portSizes (DecompBR.shI v.2.1 v.2.2)))[P[i]]! + o :=
        hje'.symm.trans hwg
      have h2 := FlockVBridge.Family.starts_get (RecResiduals.portSizes (DecompBR.shI v.2.1 v.2.2)) hle
      have h3 := FlockVBridge.OfVStar.take_sum_le (RecResiduals.portSizes (DecompBR.shI v.2.1 v.2.2)).toList hge
      have h4 := ResFrames.take_sizes_full (rounds (Rows.shapeNoCoin v.2.1 v.2.2)) #[2, 2]
      rw [← hsz] at h4
      omega
    · have h4 : (RecResiduals.portSizes (DecompBR.shI v.2.1 v.2.2))[P[i]]! = 0 :=
        getElem!_neg _ P[i] (by omega)
      rw [h4] at hon'
      exact absurd hon' (Nat.not_lt_zero _)
  have hokk : o < (rounds (Rows.shapeNoCoin v.2.1 v.2.2))[P[i]].size := by
    have hlt1 : P[i] < ((rounds (Rows.shapeNoCoin v.2.1 v.2.2)).map (fun a => Array.size a)).size := by
      rw [Array.size_map]
      exact hkk
    have hlt2 : P[i] < ((rounds (Rows.shapeNoCoin v.2.1 v.2.2)).map (fun a => Array.size a) ++ #[2, 2]).size := by
      rw [Array.size_append]
      exact Nat.lt_of_lt_of_le hlt1 (Nat.le_add_right _ _)
    have h := hon'
    rw [hsz, getElem!_pos _ P[i] hlt2, Array.getElem_append_left hlt1, Array.getElem_map] at h
    exact h
  have hle' : P[i] ≤ (RecResiduals.portSizes (DecompBR.shI v.2.1 v.2.2)).size := by
    rw [hsz, Array.size_append, Array.size_map]
    exact Nat.le_of_lt (Nat.lt_of_lt_of_le hkk (Nat.le_add_right _ _))
  have hwj : jw = (((rounds (Rows.shapeNoCoin v.2.1 v.2.2)).toList.map Array.size).take P[i]).sum + o := by
    rw [← hje', hwg, FlockVBridge.Family.starts_get _ hle', hsz, ResFrames.take_sizes_sum _ _ (Nat.le_of_lt hkk)]
  -- the copy reads port `i`'s row off the part's witness
  obtain ⟨hi', hfind, hnames, hrow⟩ := copy_rowElems (w := wt) hsh.2 hr hss hsv hpc hxO ht hpar' hshp hholo'
    hpart hi hon (q.key.2.1 + q.key.2.2.1)
  obtain ⟨hP', hnm⟩ := hnames P hPP
  have hRIf : rounds { sh with holo := false } = rounds (Rows.shapeNoCoin v.2.1 v.2.2) := by
    rw [← hI']
    exact hRI
  have hname_i : pt.S.ports[i].1 = s!"m{P[i]}" := by
    rw [hnm]
    exact ResFrames.portNames_m (by rw [hRIf]; exact hkk)
  have hnout : pt.S.ports[i].1 ≠ "out" := by
    intro h
    have h' := congrArg RecVStar.mIndex (hname_i.symm.trans h)
    rw [FlockC3.Reps.mIndex_m, RecDeploy.mIndex_out] at h'
    cases h'
  -- the frame element: chunk `t`'s value `o`
  have hrsD : (D.ctx (2 * 0 + (rf : ℕ))).sched = rs.toList := by
    show RecDeploy.repSchedAt D.d.spec D.d.sts[(2 * 0 + (rf : ℕ)) / 2]! = rs.toList
    rw [Rows.DecAt.rep_div_two 0 rf, hds, hdt, DecompBR.getElem!_one]
    exact hsched5
  obtain ⟨t, vs, hct, htl, hvs2, hvs1, hRA⟩ := D.frame_elem Top.cls₀ hx hf rfl h0 rf hshX hrsD hkk hokk
  -- the row the part holds is the opened round's
  have hmem := mem_heldRows_of hss hsv hpc hrs5 hi' hname_i hfind hct rf
  have hl0 : ws.length = RecDeploy.nS x := hJ.1
  have hl : ws.length = r.w.parts.length := by
    rw [hl0]
    unfold RecDeploy.nS
    rw [hr]
    all_goals rfl
  obtain ⟨σ, hσ, wσ, hwσ, hheld⟩ := ResHeld.held_rowSlot elem hJ.2.1 hJ.2.2.1 hr hl
    (fun s hs => by
      obtain ⟨-, xO', hck', -, -, hsh', -, -, hcm, -⟩ := hJ.2.2.2 s hs
      exact ⟨_, xO', hsh'.2, hck', hcm⟩) hrs5 hK hw hpc hmem
  have hheld' : readForms wt (((rowHidden cC s!"m{P[i]}" (rf : ℕ)).getD []).map HmNets.Lin.var) =
      readForms wσ σ.row := hheld
  have hσ' : OpenSlots.rowSlot r ((rf : ℕ) * rs.size + t) = some σ := hσ
  have hjR : (rf : ℕ) * rs.size + t < 2 * rs.size := by
    rcases (show (rf : ℕ) = 0 ∨ (rf : ℕ) = 1 by omega) with h | h <;> rw [h] <;> omega
  have hop := OpenSlots.opened_row elem hDx hl0 wt hfin hr hrs5 hjR hσ'
  have hmod : ((rf : ℕ) * rs.size + t) % rs.size = t := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt htl]
  rw [hmod] at hop
  have hgσ : ws.getD σ.s wt = wσ := by
    rw [List.getD_eq_getElem?_getD, hwσ, Option.getD_some]
  rw [hgσ] at hop
  -- the decision's rows are the opened rounds'
  have hlens : FlockDeploy.repLens D.d.spec D.d.sts = [rs.size, rs.size] := by
    unfold FlockDeploy.repLens
    rw [hds, hdt]
    refine (List.map_congr_left (g := fun _ => rs.size) fun k hk => ?_).trans rfl
    have hk2 : k < 2 := by simpa using hk
    show (RecDeploy.repSchedAt spec5 (#[st5] : Array Stmt)[k / 2]!).length = rs.size
    rw [show k / 2 = 0 by omega, DecompBR.getElem!_one, hsched5, Array.length_toList]
  have hrows := Rows.DecAt.rows_getD D (o₀ := o₀) (ot := ot) (last := last) rfl hlens rf htl
  have hgd := ResFrames.getD_toList_eq rs t
  have hra := ResFrames.rowsAt_elem hRA hvs2 hon
  rw [hrows, hgd] at hra
  first
    | erw [hop] at hra
    | replace hra := (congrArg (fun y : ByteArray => elemsOf y.data.toList
        ((P.toList.map ((RecResiduals.portSizes (DecompBR.shI v.2.1 v.2.2))[·]!))[i]'hi) o) hop).symm.trans hra
  have hcp := hrow hnout
  rw [hn1, hname_i, hheld'] at hcp
  have hpa : Rows.portAt (Rows.shapeNoCoin v.2.1 v.2.2) q.key.1
      (((P.toList.map ((RecResiduals.portSizes (DecompBR.shI v.2.1 v.2.2))[·]!)).take i).sum + o) = some jw := by
    have h := hj
    unfold elemPort at h
    rw [hv] at h
    exact h
  unfold keyCopy
  rw [hn1, hu0, hc0, hcp, Rows.elemReadC_frame (key := q.key) acc hv hpa, hn0, hwj, hvs1,
    List.getElem?_eq_getElem hvs2, Option.getD_some]
  · first
      | exact hra
      | simpa using hra
  · rw [hn0, D.framesRead_length Top.cls₀ hx hf rfl h0 rf, hshX]
    exact hjl'

end RecDeploy.ResLower
