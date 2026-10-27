import Proofs.Flock.Claims.Recursion.ResLowerD
import Proofs.Flock.Claims.Recursion.DecodeAt
import Proofs.Flock.Claims.Recursion.RestVerdictBases
import Proofs.Flock.Claims.Recursion.SlotsCheck
import Proofs.Flock.Claims.Recursion.DeployedBrings
import Proofs.Flock.C3.Reps.AtList

/-!
# `ResLoweredP` at the deployed constants (`resLowered_D`)

`resLowered_D`: at the dense form, the deployed reader `recRead`, its commitments `cmtTr recRead`, the concrete pieces
and owners (`resPD`, `ownRD`), `elemAt := Rows.elemReadC (Top.accRC rd key)` and `vvAt := Rows.resVRead`, each part's
`rec-residuals` pieces hold at every joint acceptance (`JointP`), from the frame ports (`FramePorts`, P2), the
accumulator ports up to one offset (`AccPorts`, P3) and the part's site facts (`ResSite`), with `ReadsAt` and the base
checks (`BasesHold`) for the decision `AccPorts` reads at. The share is `resPieces_computesP`'s, at each piece:

* **Which part.** The piece's owner `s` is V\*'s statement `.inner q.key.1 nm _` (`ownRD_some`), named for its group
  (`inner_named`: V\*'s list names each part `unitName … j`, `FlockC3.Reps.getElem?_list`); the staged table's
  circuit is named as it (`stagedTables`), so the unit name's parameters are the read shape's and its group is
  `q.key.1` (`resUnitName_inj`). Table 0's part is then `shI`'s (`shI_tie`), plain (`staged_plain`), the piece's `S`.
* **The tie.** `resTie_resVRead` at `Bk := (bdR …).DE.V.k`, from the site facts.
* **The residuals.** `ReadsAt` at the joint acceptance and the base checks give a decision (`decAt_of`); there table
  0's decode shape is `shapeNoCoin` (`DecAt.sh_eq`) and `shI` in every field the algebra reads (`algSh_tie`), so
  `shI = algShape (shapeNoCoin …)` (`shI_alg`): `elemPort` reads the part's `w` index (`elemPort_eq`) and `accBase` is
  where its running sums start (`accBase_eq`). The copy's rows are `elemReadC (accRC …)`'s plus `offOf` (`shift_of`,
  from P2 below `accBase` and P3 above), and the residuals vanish at one where they vanish at the other
  (`resZero_off`, at `structureOf_accLow`).
-/

namespace RecDeploy.DecompBR

open Flock
open Claims (Msg)
open FlockDeploy (Coin FinalP setupOf)

/-- **V\*'s parts are named for their groups**: a statement `.inner j nm _` of the verifier's list is named
`rec-residuals/m<m>-k<k>-c<c>-p<j>` at the inner shape (V\*'s levels, then its parts so named, then candidate 3's and
the caps, none of them `.inner`). -/
theorem inner_named {i : RecVStar.Inner} {ss : Array VStar.Stmt} (h : RecVStar.statements i = .ok ss) {s j : ℕ}
    {nm : String} {pt : VStar.Part} (hs : ss[s]? = some (.inner j nm pt)) :
    nm = RecResiduals.unitName i.m i.kLog i.claims j := by
  obtain ⟨sc, ps, C, Ks, -, hC, hKs, hss⟩ := FlockC3.Reps.getElem?_list h
  rw [hss s] at hs
  split at hs
  · cases hs
  split at hs
  · simp only [Option.some.injEq, VStar.Stmt.inner.injEq] at hs
    obtain ⟨rfl, rfl, -⟩ := hs
    rfl
  · rcases Array.mem_append.1 (Array.mem_of_getElem? hs) with hm | hm
    · rcases FlockC3.Reps.candidate3_kinds hC _ hm with ⟨_, _, he⟩ | ⟨_, _, _, _, he⟩ |
        ⟨_, _, _, _, _, _, he⟩ <;> cases he
    · obtain ⟨_, _, he⟩ := hKs _ hm
      cases he

/-- **A piece's owner is its group's part**: where `ownRD` gives `s`, V\*'s statement `s` is `.inner q.key.1 …`. -/
theorem ownRD_some {x : Msg} {q : ResPiece} {s : ℕ} (ho : ownRD x q = some s) :
    ∃ (r : XR) (ss : Array VStar.Stmt) (nm : String) (pt : VStar.Part), readX x = some r ∧
      RecVStar.statements r.shape = .ok ss ∧ ss[s]? = some (.inner q.key.1 nm pt) := by
  unfold ownRD at ho
  cases hr : readX x with
  | none => simp [hr] at ho
  | some r =>
    rw [hr, Option.bind_some] at ho
    cases hss : RecVStar.statements r.shape with
    | error e => rw [hss] at ho; simp [Except.toOption] at ho
    | ok ss =>
      rw [hss] at ho
      simp only [Except.toOption, Option.bind_some] at ho
      obtain ⟨hsz, hp, -⟩ := Array.findIdx?_eq_some_iff_getElem.1 ho
      have key : ∀ t : VStar.Stmt, ss[s]? = some t → isResOf q.key.1 t = true →
          ∃ nm pt, ss[s]? = some (.inner q.key.1 nm pt) := by
        intro t ht hp
        cases t with
        | inner j nm pt =>
          have hj : j = q.key.1 := by simpa [isResOf] using hp
          subst hj
          exact ⟨nm, pt, ht⟩
        | _ => simp [isResOf] at hp
      obtain ⟨nm, pt, hget⟩ := key ss[s] (Array.getElem?_eq_getElem hsz) hp
      exact ⟨r, ss, nm, pt, by first | rfl | exact hr, by first | rfl | exact hss, hget⟩

/-- **The concrete pieces, opened**: a piece of `resPD x` is group `p`'s at rep `r < 2`, keyed `(p, r, 0, 0)` at the
`S` of `shI`'s part `p`. -/
theorem mem_resPD' {x : Msg} {q : ResPiece} (hq : q ∈ resPD x) :
    ∃ v, setupOf false x = some v ∧ ∃ (p r : ℕ) (S : RecOpen.GfS) (ns : List ℕ) (nv : ℕ), r < 2 ∧
      RecResiduals.part (shI v.2.1 v.2.2) p = .ok (S, ns, nv) ∧ q.key = (p, r, 0, 0) ∧ q.S = S := by
  unfold resPD at hq
  cases hv : setupOf false x with
  | none => rw [hv] at hq; cases hq
  | some v =>
    rw [hv] at hq
    obtain ⟨p, -, hq⟩ := List.mem_flatMap.1 hq
    refine ⟨v, rfl, ?_⟩
    unfold resGroup at hq
    cases hp : RecResiduals.part (shI v.2.1 v.2.2) p with
    | error e => rw [hp] at hq; simp at hq
    | ok t =>
      obtain ⟨S, ns, nv⟩ := t
      rw [hp] at hq
      try dsimp only at hq
      rcases List.mem_cons.1 hq with rfl | hq
      · exact ⟨p, 0, S, ns, nv, by omega, hp, rfl, rfl⟩
      · rw [List.mem_singleton] at hq
        subst hq
        exact ⟨p, 1, S, ns, nv, by omega, hp, rfl, rfl⟩

end RecDeploy.DecompBR

namespace RecDeploy.ResLower

open Flock
open Claims (Msg)
open FlockDeploy (Coin FinalP setupOf)
open Claims.Recursion.VStarB

noncomputable section

/-- **A part is `partS` at its group**: where `part sh j` builds, S's structure and groups return at `sh`, and the part
is `partS` of group `j`'s rows, ports and `v` elements. -/
theorem part_partS {sh : _root_.Flock.Shape} {j : ℕ} {S : _root_.Flock.RecOpen.GfS} {ns : List ℕ} {nv : ℕ}
    (h : _root_.Flock.RecResiduals.part sh j = .ok (S, ns, nv)) :
    ∃ (S0 : _root_.Flock.RecAlgebra.GfS) (gs : Array (Array ℕ × Array ℕ × Array ℕ)) (R P V : Array ℕ),
      _root_.Flock.RecAlgebra.structureOf sh = .ok S0 ∧
      _root_.Flock.RecResiduals.groups S0 (_root_.Flock.RecResiduals.portSizes sh) = .ok gs ∧
      gs[j]? = some (R, P, V) ∧
      _root_.Flock.RecResiduals.partS S0 (_root_.Flock.RecResiduals.portSizes sh) R P V = .ok S := by
  unfold _root_.Flock.RecResiduals.part at h
  obtain ⟨S0, hS0, h⟩ := FlockVBridge.PartS.bind_ok h
  dsimp only at h
  split at h
  · cases h
  · obtain ⟨gs, hgs, h⟩ := FlockVBridge.PartS.bind_ok h
    try dsimp only at h
    cases hj : gs[j]? with
    | none => rw [hj] at h; cases h
    | some t =>
      obtain ⟨R, P, V⟩ := t
      rw [hj] at h
      try dsimp only at h
      obtain ⟨S2, hS2, h⟩ := FlockVBridge.PartS.bind_ok h
      simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, -, -⟩ := h
      exact ⟨S0, gs, R, P, V, hS0, hgs, hj, hS2⟩

/-- **A decision of the opened messages**: at a dense compiled statement whose last message reads (`ReadOkR`, which
`ReadsAt` gives at a joint acceptance), where the base checks hold, the record decodes (`decode_at`), so a `DecAt`
exists. -/
theorem decAt_of (S : Claims.JsonSpec FlockComplete.printJ _root_.Flock.canon) {rd : Msg → Option FinalP}
    (hBs : BasesHold rd) {xo : Msg} {sto : Claims.Recursion.Stmt} (hst : parse xo = some sto)
    (hf : Claims.Recursion.Form.Dense sto.form) {ms : List Msg} {cs : List Coin}
    (hr : FlockDeploy.ReadOkR rd (xIOf xo) ms cs) (hlen : ms.length = cs.length) :
    Nonempty (Rows.DecAt rd (xIOf xo) ms cs) := by
  have hr' := hr
  unfold FlockDeploy.ReadOkR at hr'
  cases h : FlockDeploy.actR rd (xIOf xo) ms cs with
  | rej => rw [h] at hr'; exact False.elim hr'
  | coin _ => rw [h] at hr'; exact False.elim hr'
  | decide s spec sts link c₁ mids f =>
    have hd := FlockDeploy.actR_decide h
    have hs := setupWith_sha512_of (actR_decide_setup h).1
    obtain ⟨sess, hsess⟩ := DecodeAt.decode_at rd (fun _ _ _ => True) (xIOf xo) ms cs
      ⟨⟨xo, sto, hst, hf, rfl⟩, hr, fun b hb => hBs xo sto hst b hb ms cs hlen, trivial⟩ _ hd spec sts hs
    exact ⟨⟨_, hd, spec, sts, hs, sess, hsess⟩⟩

/-- **The shape tie**: at a dense decision, `shI` at the setup is table 0's coin-free shape (`shapeNoCoin`) in every
field the algebra reads: the decode shape is `shapeNoCoin` (`DecAt.sh_eq`) and `shI` there (`algSh_tie`). -/
theorem shI_alg {rd : Msg → Option FinalP} {xi : Msg} {ms : List Msg} {cs : List Coin} (D : Rows.DecAt rd xi ms cs)
    {xo : Msg} {sto : Claims.Recursion.Stmt} (hst : parse xo = some sto)
    (hf : Claims.Recursion.Form.Dense sto.form) (hx : xi = xIOf xo)
    {v : FlockDeploy.Ins × _root_.Flock.SessionSpec × Array _root_.Flock.Stmt} (hv : setupOf false xi = some v) :
    DecompBR.shI v.2.1 v.2.2 = Rows.algShape (Rows.shapeNoCoin v.2.1 v.2.2) := by
  obtain ⟨h1, -⟩ := D.tables_of_dense hst hf hx
  have h0 : 0 < D.sts.size := by omega
  have he := RecDeploy.DecompBR.DecAt.algSh_tie D RecDeploy.Top.cls₀ hst hf hx h0 hv
  have hG := D.rowG RecDeploy.Top.cls₀ (fun _ => #[]) hst hf hx h0
  have hsh := D.sh_eq RecDeploy.Top.cls₀ (fun _ => #[]) hG h0
  obtain ⟨hsp, hsts⟩ := D.setup_eq
  obtain ⟨v', c₁, r₀, rss, fl, link, pf, hv', -, -, -, -, -, -, hd⟩ := Rows.decR_last D.dec
  have hvv : v' = v := Option.some.inj (hv'.symm.trans hv)
  have hds : D.d.spec = v.2.1 := by rw [hd, hvv]
  have hdt : D.d.sts = v.2.2 := by rw [hd, hvv]
  rw [← he, hsh, hsp, hsts, hds, hdt]

/-- `elemPort` reads group `jj`'s `w` index at `shI`, where `shI` is `shapeNoCoin`'s algebra shape. -/
theorem elemPort_eq {xi : Msg} {v : FlockDeploy.Ins × _root_.Flock.SessionSpec × Array _root_.Flock.Stmt}
    (hv : setupOf false xi = some v)
    (hI : DecompBR.shI v.2.1 v.2.2 = Rows.algShape (Rows.shapeNoCoin v.2.1 v.2.2))
    {S0 : _root_.Flock.RecAlgebra.GfS} {gs : Array (Array ℕ × Array ℕ × Array ℕ)} {R P V : Array ℕ}
    (hS : _root_.Flock.RecAlgebra.structureOf (DecompBR.shI v.2.1 v.2.2) = .ok S0)
    (hg : _root_.Flock.RecResiduals.groups S0 (_root_.Flock.RecResiduals.portSizes (DecompBR.shI v.2.1 v.2.2)) =
      .ok gs)
    {jj : ℕ} (hj : gs[jj]? = some (R, P, V)) (e : ℕ) :
    elemPort xi jj e =
      (FlockVBridge.PartS.wIdx (_root_.Flock.RecResiduals.portSizes (DecompBR.shI v.2.1 v.2.2)) P)[e]? := by
  have hps : _root_.Flock.RecResiduals.portSizes (Rows.shapeNoCoin v.2.1 v.2.2) =
      _root_.Flock.RecResiduals.portSizes (DecompBR.shI v.2.1 v.2.2) := by
    rw [hI, Rows.portSizes_alg]
  have hS' : _root_.Flock.RecAlgebra.structureOf (Rows.shapeNoCoin v.2.1 v.2.2) = .ok S0 := by
    rw [← Rows.structureOf_alg, ← hI]
    exact hS
  rw [← hps] at hg ⊢
  unfold elemPort Rows.portAt
  simp [hv, hS', hg, hj]

/-- `accBase` is where `shI`'s running sums start in `w` (`nR`), where `shI` is `shapeNoCoin`'s algebra shape. -/
theorem accBase_eq {xi : Msg} {v : FlockDeploy.Ins × _root_.Flock.SessionSpec × Array _root_.Flock.Stmt}
    (hv : setupOf false xi = some v)
    (hI : DecompBR.shI v.2.1 v.2.2 = Rows.algShape (Rows.shapeNoCoin v.2.1 v.2.2)) :
    accBase xi = _root_.Flock.RecAlgebra.nR (DecompBR.shI v.2.1 v.2.2) := by
  have hr : _root_.Flock.RecAlgebra.rounds (Rows.algShape (Rows.shapeNoCoin v.2.1 v.2.2)) =
      _root_.Flock.RecAlgebra.rounds (Rows.shapeNoCoin v.2.1 v.2.2) :=
    OpenLower.rounds_congr rfl
  unfold accBase
  rw [hv, hI]
  all_goals exact congrArg (fun a => Array.size (Array.flatten a)) hr.symm

/-- **The copy's elements are `E` shifted by `offOf`**: where each element's `w` index is `wI`'s, `E'` is `E` below
`A` and `E` plus `δ` of the running sum above it. -/
theorem shift_of {E E' : ℕ → _root_.Flock.F128} {wI : Array ℕ} {A : ℕ} {δ : ℕ → _root_.Flock.F128}
    {port : ℕ → Option ℕ} (hp : ∀ e, port e = wI[e]?)
    (hlo : ∀ e j, port e = some j → j < A → E' e = E e)
    (hhi : ∀ e j, port e = some j → A ≤ j → E' e = E e + δ ((j - A) % 2)) :
    ∀ e < wI.size, E' e = E e + FlockVBridge.PartS.offOf A wI δ e := by
  intro e he
  have hje : wI[e]? = some wI[e] := Array.getElem?_eq_getElem he
  have hpe : port e = some wI[e] := (hp e).trans hje
  by_cases hlt : wI[e] < A
  · rw [hlo e _ hpe hlt, FlockVBridge.PartS.offOf_low hje hlt, FlockVBridge.PartS.f128_add_zero']
  · have hle : A ≤ wI[e] := Nat.le_of_not_lt hlt
    rw [hhi e _ hpe hle, FlockVBridge.PartS.offOf_acc (c := wI[e] - A)
      (by rw [Nat.add_sub_of_le hle]; exact hje)]

/-- **`ResLoweredP` at the deployed constants**: at the dense form, the deployed reader and its commitments, the
concrete pieces and owners, `Rows.elemReadC (Top.accRC rd key)` and `Rows.resVRead`, from the frame ports (P2), the
accumulator ports up to one offset (P3), the site facts, `ReadsAt` and the base checks. -/
theorem resLowered_D (rd : Msg → Option FinalP) (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ) (wE : Msg → ℝ)
    (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) (b : Brings)
    (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    (S : Claims.JsonSpec FlockComplete.printJ _root_.Flock.canon) (hBs : BasesHold rd)
    (hRd : RestRead.ReadsAt rd final (DeployedR.restR elem))
    (hset : ∀ xO, ∀ u ∈ resSetup xO, u ∈ b.setup xO)
    (hF : FramePorts rd wE final w b elem Claims.Recursion.Form.Dense recRead (cmtTr recRead))
    (hA : AccPorts rd key wE final w b elem Claims.Recursion.Form.Dense recRead (cmtTr recRead))
    (hSite : ResSite rd wE final w b elem) :
    ResLoweredP rd wE final w b elem Claims.Recursion.Form.Dense recRead (cmtTr recRead) DecompBR.resPD
      DecompBR.ownRD (Rows.elemReadC (Top.accRC rd key)) Rows.resVRead := by
  intro x st hx hf c ws hJ s hs
  have hsw : s < ws.length := lt_of_lt_of_eq hs hJ.1.symm
  have hw : ws[s]? = some ws[s] := List.getElem?_eq_getElem hsw
  intro q hq ho
  obtain ⟨r, ss, nm, pt, hr, hss, hsv⟩ := DecompBR.ownRD_some ho
  refine resPieces_computesP (hJ.2.2.2 s hsw) hr hss hsv (_root_.RecDeploy.bdR rd wE final w) DecompBR.resPD
    DecompBR.ownRD _ _ (fun xO p hxO hck hsh hAg hT hc q' hq' ho' => ?_) q hq ho
  -- the staged statement: one table, `p` its parse
  obtain ⟨-, i, spec, sts, -, hsu, -⟩ := hsh.2
  have hpe : p = { i.base.read with sts } := Option.some.inj (hxO.symm.trans (parse_of_setupO hsu))
  subst hpe
  have h1 := (oneFull_staged hsh.2 hsu).1
  have ht : 0 < sts.size := by omega
  -- the piece's part
  obtain ⟨r', ss', nm', pt', hr', hss', hsv'⟩ := DecompBR.ownRD_some ho'
  obtain ⟨⟨hG, hbd, hu⟩, hSt⟩ := hSite x st hx hf c ws hJ s ws[s] hw r' hr' ss' hss' _ nm' pt' hsv' xO _ hxO hck hsh
    hAg hT hc q' hq' ho' i spec hsu ht
  refine ⟨ht, hG, hbd, hu, fun m k cl jj sh holo S' ns nv hpar hshp hholo hpart => ?_⟩
  obtain ⟨cC, hpc, hidx, hnp, qv, qo, ptv, pto, hqv, hqo, hvin, hoout, hptv, hpto, hrows, hwo⟩ :=
    hSt m k cl jj sh holo S' ns nv hpar hshp hholo hpart
  have hFr := hF x st hx hf c ws hJ s ws[s] hw r' hr' ss' hss' _ nm' pt' hsv' xO _ hxO hck hsh hAg hT hc q' hq' ho'
    ht m k cl jj sh holo S' ns nv hpar hshp hholo hpart
  have hAc := hA x st hx hf c ws hJ s ws[s] hw r' hr' ss' hss' _ nm' pt' hsv' xO _ hxO hck hsh hAg hT hc q' hq' ho'
    ht m k cl jj sh holo S' ns nv hpar hshp hholo hpart
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
    obtain ⟨D⟩ := decAt_of (xo := x) (sto := st) S hBs hx hf hRk hlen
    have hshI := shI_alg (xo := x) (sto := st) D hx hf rfl hv
    obtain ⟨δ, hδ⟩ := hAc os last hfin hos ⟨D⟩
    have hFr' := hFr os last hfin hos ⟨D⟩
    obtain ⟨S1, gs, R, P, V, hS1, hgs, hgj, hpS⟩ := part_partS hpart'
    refine FlockVBridge.PartS.resZero_off hpS ?_
      (shift_of (fun e => elemPort_eq hv hshI hS1 hgs hgj e) (fun e j hj hl => hFr' (Top.accRC rd key) e j hj hl)
        (fun e j hj hl => hδ e j hj hl)) hz
    rw [accBase_eq hv hshI]
    exact _root_.Flock.RecAlgebra.structureOf_accLow (by rw [hI]; exact hh) hS1

/-- **`ResLoweredP` at the deployed brings** (`bB (deployedB b) …`), whose setup lists the residual reads
(`resSetup_deployedB`). -/
theorem resLowered_DB (rd : Msg → Option FinalP) (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ) (wE : Msg → ℝ)
    (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) (b : Brings)
    (aux : Msg → ℕ → _root_.Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ) (hd : Msg → _root_.Flock.RecRecord.Record)
    (rest : Msg → List (Msg × Claims.Out) → List UInt8)
    (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    (S : Claims.JsonSpec FlockComplete.printJ _root_.Flock.canon) (hBs : BasesHold rd)
    (hRd : RestRead.ReadsAt rd final (DeployedR.restR elem))
    (hF : FramePorts rd wE final w (DecompBR.bB (deployedB b) aux nOf hd rest) elem Claims.Recursion.Form.Dense
      recRead (cmtTr recRead))
    (hA : AccPorts rd key wE final w (DecompBR.bB (deployedB b) aux nOf hd rest) elem Claims.Recursion.Form.Dense
      recRead (cmtTr recRead))
    (hSite : ResSite rd wE final w (DecompBR.bB (deployedB b) aux nOf hd rest) elem) :
    ResLoweredP rd wE final w (DecompBR.bB (deployedB b) aux nOf hd rest) elem Claims.Recursion.Form.Dense recRead
      (cmtTr recRead) DecompBR.resPD DecompBR.ownRD (Rows.elemReadC (Top.accRC rd key)) Rows.resVRead :=
  resLowered_D rd key wE final w _ elem S hBs hRd (resSetup_deployedB b) hF hA hSite

end

end RecDeploy.ResLower

namespace RecDeploy.DecompBR

open Flock
open Claims (Msg Out)
open FlockDeploy (Coin FinalP)
open Claims.Recursion.VStarB

/-- **`ResLowJP` at the deployed constants**: `resLowered_DB` through `resLowJP_of_resLoweredP`, at the concrete pieces
and owners, `Rows.elemReadC (Top.accRC rd key)` and `Rows.resVRead`. -/
theorem resLowJP_D (rd : Msg → Option FinalP) (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ) (b : Brings)
    (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ) (hd : Msg → Flock.RecRecord.Record)
    (rest : Msg → List (Msg × Out) → List UInt8) (wE : Msg → ℝ)
    (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    (S : Claims.JsonSpec FlockComplete.printJ Flock.canon) (hBs : BasesHold rd)
    (hRd : RestRead.ReadsAt rd final (DeployedR.restR elem))
    (hF : ResLower.FramePorts rd wE final w (bB (deployedB b) aux nOf hd rest) elem Claims.Recursion.Form.Dense
      recRead (cmtTr recRead))
    (hA : ResLower.AccPorts rd key wE final w (bB (deployedB b) aux nOf hd rest) elem Claims.Recursion.Form.Dense
      recRead (cmtTr recRead))
    (hSite : ResLower.ResSite rd wE final w (bB (deployedB b) aux nOf hd rest) elem) :
    ResLowJP rd (deployedB b) aux nOf hd rest wE final w (DeployedR.layR elem) resPD
      (Rows.elemReadC (Top.accRC rd key)) Rows.resVRead ownRD :=
  resLowJP_of_resLoweredP rd (deployedB b) aux nOf hd rest wE final w elem resPD ownRD _ _
    (ResLower.resLowered_DB rd key wE final w b aux nOf hd rest elem S hBs hRd hF hA hSite)

end RecDeploy.DecompBR
