import Proofs.Flock.Claims.Recursion.ResSiteD

/-!
# `ResLoweredPD` at the deployed constants (`resLowered_D`)

The site facts need part `s`'s own decision at its staged statement (`PartDec`: table 0 labels its instances `0..n-1`
only at a full draw, which only the decision checks), and a joint acceptance at arbitrary witnesses carries none. So the
decision travels beside the staged statement, in what each part shows (`showsDec`: `showsPart` and `PartDec` wherever
the statement sets up). The chain's commit ties between two parts read both parts' setups and a collision with the
verifier's own evaluations there (`accCommit_eq`), so `ResLoweredPD` is `ResLoweredP` at `JointV` over `showsDec` (the
parts' staged statements `xs`, staged at one prefix of messages: `XsStaged`, which the ties read too), and
`DecompBR.ResLowJPD` is `ResLowJP` there. KS holds `JointV` where it reads `hres`, and at an accepted run each part's
staged statement shows it (`staged_deployRec_part`, `partDec_run`) and the run's V\* statements are staged at one prefix
of its messages (`okRuns_deployRec`).

`resLowered_D`: `ResLoweredPD` at the dense form, `recRead`, `cmtTr recRead`, the concrete pieces and owners, `elemAt
:= Rows.elemReadC (Top.accRC rd key)` and `vvAt := Rows.resVRead`, from the frame ports (P2) at `JointP` over
`showsPart` (`jointP_of_jointV`, `jointP_shows`), the accumulator ports (P3) at that and `JointV`'s statements,
decisions and collision (`PortsAtV`), `ReadsAt` and the base checks; the site facts are `resSite_of_dec`'s. Its steps
are `ResLowerDA`'s.
-/

namespace RecDeploy.OpenLower

open Claims (Msg)
open FlockDeploy (Coin)
open Claims.Recursion.VStarB

/-- **`JointP` is monotone in what each part shows.** -/
theorem jointP_shows {Col T : Type} {H : Msg → Msg} {D : Claims.Recursion.Deploy Coin}
    {DO : Claims.Flock.Deploy Coin Col T} {L : LayoutB} {shape : Claims.Recursion.Form → Prop}
    {shows shows' : ℕ → Msg → Msg → (ℕ → Coin) → Prop} {cmt : Msg → ℕ → Msg → Msg} {x : Msg} {c : ℕ → Coin}
    {ws : List Claims.Flock.Wit} (hw : ∀ {s x xO c}, shows s x xO c → shows' s x xO c)
    (h : JointP H D L (PartAcceptsP H D DO L shape shows cmt) x c ws) :
    JointP H D L (PartAcceptsP H D DO L shape shows' cmt) x c ws := by
  refine ⟨h.1, h.2.1, h.2.2.1, fun s hs => ?_⟩
  obtain ⟨hst, xO, h1, h2, h3, h4, h5⟩ := h.2.2.2 s hs
  exact ⟨hst, xO, h1, h2, h3, hw h4, h5⟩

end RecDeploy.OpenLower

namespace RecDeploy.ResLower

open Flock
open Claims (Msg)
open FlockDeploy (Coin FinalP setupOf)
open Claims.Recursion.VStarB
open RecDeploy.OpenLower (jointP_shows)

noncomputable section

/-- **`ResLoweredP` with each part's decision, at the parts' staged statements** (`JointV` over `showsDec`: the
collision conjunct widened to the verifier's own evaluations at `xs`, which the chain's commit ties read, with `xs`
staged at one prefix of messages, `XsStaged`). -/
def ResLoweredPD (rd : Msg → Option FinalP) (wE : Msg → ℝ) (final : Msg → Option (List (List (Msg × Salt)) × Msg))
    (w : Msg → ℝ) (b : Brings) (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    (shape : Claims.Recursion.Form → Prop) (untr : TrReader) (cmt : Msg → ℕ → Msg → Msg)
    (pieces : Msg → List ResPiece) (ownQ : Msg → ResPiece → Option ℕ)
    (elemAt : Msg → List Msg → List Coin → ℕ × ℕ × ℕ × ℕ → ℕ → _root_.Flock.F128)
    (vvAt : Msg → List Coin → ℕ × ℕ × ℕ × ℕ → ℕ → _root_.Flock.F128) : Prop :=
  ∀ (x : Msg) (st : Claims.Recursion.Stmt), (DRL rd wE final w b elem).parse x = some st → shape st.form →
    ∀ (c : ℕ → Coin) (xs : List Msg) (ws : List Claims.Flock.Wit),
    RecDeploy.OpenLower.JointV FlockDeploy.sha512 (DRL rd wE final w b elem) (DeployedR.layR elem)
      (AccV rd wE final w b elem shape untr cmt) x c xs ws → XsStaged b.st x xs ws.length →
    ∀ s < (DRL rd wE final w b elem).nS x,
      PiecesComputes (_root_.RecDeploy.bdR rd wE final w) (DRL rd wE final w b elem) pieces (resHolds elemAt vvAt) ownQ
        s x c ((DeployedR.layR elem).msgs x ws)

/-- **`ResLoweredPD` at the deployed constants**: at the dense form, the deployed reader and its commitments, the
concrete pieces and owners, `Rows.elemReadC (Top.accRC rd key)` and `Rows.resVRead`, from the frame ports (P2), the
accumulator ports up to one offset (P3), `ReadsAt` and the base checks. -/
theorem resLowered_D (rd : Msg → Option FinalP) (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ) (wE : Msg → ℝ)
    (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) (b : Brings)
    (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    (hBs : BasesHold rd)
    (hRd : RestRead.ReadsAt rd final (DeployedR.restR elem))
    (hset : ∀ xO, ∀ u ∈ resSetup xO, u ∈ b.setup xO)
    (hF : FramePorts rd wE final w b elem Claims.Recursion.Form.Dense recRead (cmtTr recRead))
    (hA : AccPorts rd key wE final w b elem Claims.Recursion.Form.Dense recRead (cmtTr recRead)) :
    ResLoweredPD rd wE final w b elem Claims.Recursion.Form.Dense recRead (cmtTr recRead) DecompBR.resPD
      DecompBR.ownRD (Rows.elemReadC (Top.accRC rd key)) Rows.resVRead := by
  intro x st hx hf c xs ws hJV hms s hs
  have hJ : RecDeploy.OpenLower.JointP FlockDeploy.sha512 (DRL rd wE final w b elem) (DeployedR.layR elem)
      (AccP rd wE final w b elem Claims.Recursion.Form.Dense recRead (cmtTr recRead)) x c ws :=
    jointP_shows (shows := showsDec recRead b) (fun h => h.1) (RecDeploy.OpenLower.jointP_of_jointV hJV)
  have hsw : s < ws.length := lt_of_lt_of_eq hs hJ.1.symm
  have hw : ws[s]? = some ws[s] := List.getElem?_eq_getElem hsw
  intro q hq ho
  obtain ⟨r, ss, nm, pt, hr, hss, hsv⟩ := DecompBR.ownRD_some ho
  refine resPieces_computesV (shows := showsDec recRead b) (fun h => h.1) (hJV.2.2.2 s hsw) hr hss hsv
    (_root_.RecDeploy.bdR rd wE final w) DecompBR.resPD DecompBR.ownRD _ _
    (fun p hxO hck hshD hAg hT hc q' hq' ho' => ?_) q hq ho
  generalize hxs : xs.getD s [] = xO at hxO hck hshD hAg hc
  -- the staged statement: one table, `p` its parse
  have hsh := hshD.1
  obtain ⟨-, i, spec, sts, -, hsu, -⟩ := hsh.2
  have hpe : p = { i.base.read with sts } := Option.some.inj (hxO.symm.trans (parse_of_setupO hsu))
  subst hpe
  have h1 := (oneFull_staged hsh.2 hsu).1
  have ht : 0 < sts.size := by omega
  -- the piece's part
  obtain ⟨r', ss', nm', pt', hr', hss', hsv'⟩ := DecompBR.ownRD_some ho'
  obtain ⟨⟨hG, hbd, hu⟩, hSt⟩ := resSite_of_dec rd wE final w b elem x st hx hf c ws hJ s ws[s] hw r' hr' ss'
    hss' _ nm' pt' hsv' xO _ hxO hck hsh hAg hT hc q' hq' ho' i spec hsu (hshD.2 i spec sts hsu) ht
  refine ⟨ht, hG, hbd, hu, fun m k cl jj sh holo S' ns nv hpar hshp hholo hpart => ?_⟩
  obtain ⟨cC, hpc, hidx, hnp, qv, qo, ptv, pto, hqv, hqo, hvin, hoout, hptv, hpto, hrows, hwo⟩ :=
    hSt m k cl jj sh holo S' ns nv hpar hshp hholo hpart
  have hFr := hF x st hx hf c ws hJ s ws[s] hw r' hr' ss' hss' _ nm' pt' hsv' xO _ hxO hck hsh hAg hT hc q' hq' ho'
    ht m k cl jj sh holo S' ns nv hpar hshp hholo hpart
  have hAc := hA x st hx hf c ws hJ xs hJV.2.2.2 hJV.2.2.1 hms s ws[s] hw r' hr' ss' hss' _ nm' pt' hsv' xO _ hxs hxO
    hck hsh hAg hT hc q' hq' ho' ht m k cl jj sh holo S' ns nv hpar hshp hholo hpart
  -- the unit name is the read shape's, at group `q'.key.1`
  obtain ⟨p', r'', sv, hx', -, -, hr'', ⟨ss'', hss'', hsv''⟩, -, htab⟩ := stagedTables hsh.2
  obtain rfl : p' = { i.base.read with sts } := Option.some.inj (hx'.symm.trans hxO)
  have hsv3 : sv = .inner q'.key.1 nm' pt' := by
    have e1 : r'' = r' := Option.some.inj (hr''.symm.trans hr')
    rw [e1, hss'] at hss''
    have e2 : ss'' = ss' := (Except.ok.inj hss'').symm
    rw [e2, hsv'] at hsv''
    exact (Option.some.inj hsv'').symm
  have hname : sts[0].c.unit.name = nm' := by
    have h := (htab 0 ht).1
    rw [hsv3] at h
    exact h
  have hn : _root_.Flock.RecResiduals.unitName m k cl jj =
      _root_.Flock.RecResiduals.unitName r'.shape.m r'.shape.kLog r'.shape.claims q'.key.1 :=
    (FlockVBridge.Family.resUnitName_params hpar).trans (hname.trans (DecompBR.inner_named hss' hsv'))
  obtain ⟨rfl, rfl, rfl, rfl⟩ := FlockVBridge.Family.resUnitName_inj hn
  -- table 0's part is `shI`'s, plain
  obtain ⟨v, hv, p0, r0, S0, ns0, nv0, hr0, hpart0, hk, hqS⟩ := DecompBR.mem_resPD' hq'
  obtain ⟨r3, ss3, sh0, hr3, -, hh, hsh0, hI, -⟩ := DecompBR.shI_tie (x := x) ⟨st, hx, hf⟩ hv
  have e3 : r3 = r' := Option.some.inj (hr3.symm.trans hr')
  rw [e3] at hh hsh0 hI
  have hshe : sh0 = sh := Option.some.inj (hsh0.symm.trans hshp)
  rw [hshe] at hI
  have hpl := staged_plain (x := x) (st0 := st) hx hf hsh hsu 0 ht
  have hho : holo = false := Except.ok.inj (hholo.symm.trans hpl)
  subst hho
  have hpart' : _root_.Flock.RecResiduals.part (DecompBR.shI v.2.1 v.2.2) q'.key.1 = .ok (S', ns, nv) := by
    rw [hI, hh]
    exact hpart
  have hk1 : q'.key.1 = p0 := by simp only [hk]
  have hSS : S' = S0 := by
    have hp0 : _root_.Flock.RecResiduals.part (DecompBR.shI v.2.1 v.2.2) p0 = .ok (S', ns, nv) := by
      rw [← hk1]; exact hpart'
    have h := hp0.symm.trans hpart0
    simp only [Except.ok.injEq, Prod.mk.injEq] at h
    exact h.1
  refine ⟨hSS.trans hqS.symm, rfl, ?_, ?_⟩
  · -- check 5's tie
    have hk' : q'.key = (q'.key.1, q'.key.2.1, q'.key.2.2.1, 0) := by simp only [hk]
    have hn2 : q'.key.2.1 + q'.key.2.2.1 < 2 := by simp only [hk]; omega
    rw [hk']
    exact resTie_resVRead (x := x) (Bk := fun xi => (_root_.RecDeploy.bdR rd wE final w).DE.V.k xi)
      (o := q'.key.2.1) (g := q'.key.2.2.1) (n := q'.key.2.1 + q'.key.2.2.1) hx hf le_rfl hT hc hsh hck hr' hss'
      hsv' hpc hxO ht hsu ht hidx hpar (hset xO) hshp hholo hpart hG rfl hn2 hnp hqv hqo hvin hoout hptv hpto
      hrows hwo
  · -- the residuals, at the decision `ReadsAt` gives
    intro os last hfin hos hz
    have hRk := hRd x ⟨st, hx, hf⟩ c ws hJ.1 (fun e he => hJ.2.1 e (DecompBR.allCalls_layR elem x ws he)) os last
      hfin hos
    have hos' : os.length + 1 = RestRead.kI x := hos
    have hlen : (os.map (fun o => FlockDeploy.joinR (o.map Prod.fst)) ++ [last]).length =
        ((List.range (RestRead.kI x)).map c).length := by
      simp only [List.length_append, List.length_map, List.length_singleton, List.length_range]
      exact hos'
    obtain ⟨D⟩ := decAt_of (xo := x) (sto := st) hBs hx hf hRk hlen
    have hshI := shI_alg (xo := x) (sto := st) D hx hf rfl hv
    obtain ⟨δ, hδ⟩ := hAc os last hfin hos ⟨D⟩
    have hFr' := hFr os last hfin hos ⟨D⟩
    obtain ⟨S1, gs, R, P, V, hS1, hgs, hgj, hpS⟩ := part_partS hpart'
    refine FlockVBridge.PartS.resZero_off hpS ?_
      (shift_of (fun e => elemPort_eq hv hshI hS1 hgs hgj e) (fun e j hj hl => hFr' (Top.accRC rd key) e j hj hl)
        (fun e j hj hl => hδ e j hj hl)) hz
    rw [accBase_eq hv hshI]
    exact _root_.Flock.RecAlgebra.structureOf_accLow (by rw [hI]; exact hh) hS1

/-- **`ResLoweredPD` at the deployed brings** (`bB (deployedB b) …`), whose setup lists the residual reads
(`resSetup_deployedB`). -/
theorem resLowered_DB (rd : Msg → Option FinalP) (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ) (wE : Msg → ℝ)
    (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) (b : Brings)
    (aux : Msg → ℕ → _root_.Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ) (hd : Msg → _root_.Flock.RecRecord.Record)
    (rest : Msg → List (Msg × Claims.Out) → List UInt8)
    (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    (hBs : BasesHold rd)
    (hRd : RestRead.ReadsAt rd final (DeployedR.restR elem))
    (hF : FramePorts rd wE final w (DecompBR.bB (deployedB b) aux nOf hd rest) elem Claims.Recursion.Form.Dense
      recRead (cmtTr recRead))
    (hA : AccPorts rd key wE final w (DecompBR.bB (deployedB b) aux nOf hd rest) elem Claims.Recursion.Form.Dense
      recRead (cmtTr recRead)) :
    ResLoweredPD rd wE final w (DecompBR.bB (deployedB b) aux nOf hd rest) elem Claims.Recursion.Form.Dense recRead
      (cmtTr recRead) DecompBR.resPD DecompBR.ownRD (Rows.elemReadC (Top.accRC rd key)) Rows.resVRead :=
  resLowered_D rd key wE final w _ elem hBs hRd (resSetup_deployedB b) hF hA

end

end RecDeploy.ResLower

namespace RecDeploy.DecompBR

open Flock
open Claims (Msg Out)
open FlockDeploy (Coin FinalP)
open Claims.Recursion.VStarB
open RecDeploy.OpenLower (JointP)

noncomputable section

variable (rd : Msg → Option FinalP) (b : Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
  (hd : Msg → Flock.RecRecord.Record) (rest : Msg → List (Msg × Out) → List UInt8) (wE : Msg → ℝ)
  (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) (L : LayoutB)
  (resP : Msg → List ResPiece)
  (elemAt : Msg → List Msg → List Coin → ℕ × ℕ × ℕ × ℕ → ℕ → _root_.Flock.F128)
  (vvAt : Msg → List Coin → ℕ × ℕ × ℕ × ℕ → ℕ → _root_.Flock.F128)
  (ownR : Msg → ResPiece → Option ℕ)

/-- What each part's circuit accepts at layout `L`, at its V\* statement and witness, showing its decision. -/
abbrev acceptsLVD :=
  PartAcceptsV FlockDeploy.sha512 (recL rd b aux nOf hd rest wE final w L) (deployZkRec _root_.Flock.Readable.relation)
    L Claims.Recursion.Form.Dense (ResLower.showsDec recRead (bB b aux nOf hd rest)) (cmtTr recRead)

/-- **`ResLowJP` with each part's decision, at the parts' V\* statements**: at `JointV` over `showsDec`, which an
accepted run gives at its staged statements (`staged_deployRec_part`, `partDec_run`). -/
def ResLowJPD : Prop :=
  ∀ (x : Msg) (st : Claims.Recursion.Stmt), parse x = some st → Claims.Recursion.Form.Dense st.form →
    ∀ (c : ℕ → Coin) (xs : List Msg) (ws : List Claims.Flock.Wit),
      OpenLower.JointV FlockDeploy.sha512 (recL rd b aux nOf hd rest wE final w L) L
        (acceptsLVD rd b aux nOf hd rest wE final w L) x c xs ws →
        ResLower.XsStaged (bB b aux nOf hd rest).st x xs ws.length →
      ∀ s < nS x, PiecesComputes (bdR rd wE final w) (recL rd b aux nOf hd rest wE final w L) resP
        (resHolds elemAt vvAt) ownR s x c (L.msgs x ws)

end

/-- **`ResLowJPD` at the deployed constants**: `resLowered_DB` at the concrete pieces and owners,
`Rows.elemReadC (Top.accRC rd key)` and `Rows.resVRead`, from P2, P3, `ReadsAt` and the base checks. -/
theorem resLowJP_D (rd : Msg → Option FinalP) (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ) (b : Brings)
    (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ) (hd : Msg → Flock.RecRecord.Record)
    (rest : Msg → List (Msg × Out) → List UInt8) (wE : Msg → ℝ)
    (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    (hBs : BasesHold rd)
    (hRd : RestRead.ReadsAt rd final (DeployedR.restR elem))
    (hF : ResLower.FramePorts rd wE final w (bB (deployedB b) aux nOf hd rest) elem Claims.Recursion.Form.Dense
      recRead (cmtTr recRead))
    (hA : ResLower.AccPorts rd key wE final w (bB (deployedB b) aux nOf hd rest) elem Claims.Recursion.Form.Dense
      recRead (cmtTr recRead)) :
    ResLowJPD rd (deployedB b) aux nOf hd rest wE final w (DeployedR.layR elem) resPD
      (Rows.elemReadC (Top.accRC rd key)) Rows.resVRead ownRD :=
  fun x st hx hf c xs ws hJ hms s hs =>
    ResLower.resLowered_DB rd key wE final w b aux nOf hd rest elem hBs hRd hF hA x st hx hf c xs ws hJ hms s hs

end RecDeploy.DecompBR
