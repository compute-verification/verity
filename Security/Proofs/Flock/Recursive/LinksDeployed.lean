import Proofs.Flock.Recursive.LinksRef
import Proofs.Flock.Claims.Recursion.InnerAtCoins
import Proofs.Flock.Recursive.KSLinked

/-!
# The links bind at the deployment, from the link part's reference step

`LinksBindDeployed` asks, on an accepted run whose extracted witness is in `R` and fails the link clause (`LinksHold`
at the rows `tableRows` reads off `E_O`'s witnesses at the judged draw), for two colliding inputs of the link finder.
This file names those inputs and proves the binding from `LinkRefStaged`, the link part's reference step:

* `linkMsgs`: at the link part's circuit and witness, at each instance `n < N`, the reference step's strings
  (`refMsgs`) and the inner half's (`innerMsgs`) at the instance's port words; `linkIns` reads them off the judged
  draw's witnesses, the first view `drawsOf` serves.
* `linksHold_or_coll`: where every part is staged and its witness checks against its V\* statement, a witness of the
  inner relation satisfies the link clause at `tableRows`, or two of `linkMsgs` collide. Row `n` opens the table
  (`rowOpens_staged`); the setup's `rin` and `rout` are check 5's rows of the roots of the inner statement's rows
  (`linkSetup_staged`, the files the session sets up from being the statement's, `InnerAt.src_staged`); the reference
  step gives the instance's unit those roots (`LinkRefStaged`), and the inner half the row (`inner_link`).
* `linkBinds_rec`, `linksBindDeployed_of`: the accepted run's V\* statements are the staged ones (`runVstar_staged`,
  `stagedAll_run`) and `E_O`'s witnesses check against them (`eoGives_outerHolds`), at every compiled record
  `deployRec` whose inner relation is the deployed one's; `recursiveKSLink_links` is `RecursiveKSLinkAt` there.
-/

namespace RecDeploy.Links

open Flock
open Claims (Msg)
open Claims.Recursion.VStarB

/-! ## What the link finder reads -/

/-- The link part's bounds `UB` and `PB`, at V\*'s statement last but one (`linkAt`'s). -/
def linkBounds (r : XR) : Option (ℕ × ℕ) :=
  (RecVStar.statements r.shape).toOption.bind fun ss =>
    match ss[ss.size - 2]? with
    | some (VStar.Stmt.link UB PB _) => some (UB, PB)
    | _ => none

/-- **What the link finder reads at instance `n`** of the link part (circuit `c`, witness `wO`), at the inner statement
`p` and its files' statement `st'`, the inner witness `w` and the roots `R`, `W`: the reference step's strings
(`refMsgs`) and the inner half's (`innerMsgs`), at the instance's port words (`inWordsAt`, `xyr` and `xyw` split as
`linkRefAt` splits them). -/
def instMsgs (p : Readable.Parsed) (w : Claims.Flock.Wit) (UB PB : ℕ) (st' : Stmt) (c : Circuit)
    (wO : Claims.Flock.Wit) (R W : ByteArray) (n : ℕ) : List Msg :=
  refMsgs c wO n R W ++
    (innerMsgs p w UB PB (RecLinkPart.innerRows st') (inWordsAt c wO "trow" n) ((inWordsAt c wO "xyr" n).take 32)
      ((inWordsAt c wO "xyr" n).drop 32) (inWordsAt c wO "raux" n) ((inWordsAt c wO "xyw" n).take 32)
      ((inWordsAt c wO "xyw" n).drop 32) (inWordsAt c wO "waux" n)).map (·.data.toList)

/-- **What the link finder reads at a statement**: where it names the link part (`linkCircuit`, `linkBounds`), its inner
statement parses (`innerParsed`) and its files set it up (`RecScope.innerStmt`), `instMsgs` at each of the part's `N`
instances, at the part's witness among `ws` and the roots of the files' statement's rows; none elsewhere. -/
def linkMsgs (x : Msg) (ws : List Claims.Flock.Wit) (w : Claims.Flock.Wit) : List Msg :=
  match readX x with
  | none => []
  | some r =>
    match linkCircuit r, linkBounds r, innerParsed r, RecScope.innerStmt (filesOf r) with
    | some (s, N, c), some (UB, PB), some p, some st' =>
      let R := (RecLinkTree.rowsRoot UB PB (RecLinkPart.innerRows st').1).toOption.getD default
      let W := (RecLinkTree.rowsRoot UB PB (RecLinkPart.innerRows st').2).toOption.getD default
      (List.range N).flatMap (instMsgs p w UB PB st' c (ws.getD s ⟨[]⟩) R W)
    | _, _, _, _ => []

/-- **The link finder's inputs**: `linkMsgs` at the outer witnesses of the first view, the judged draw's (`drawsOf`). -/
def linkIns (_H : Msg → Msg) (x : Msg)
    (views : List ((ℕ → FlockDeploy.Coin) × List (Msg × Claims.Out) × List Claims.Flock.Wit)) (w : Claims.Flock.Wit)
    (_rs : List (List UInt8)) : List Msg :=
  match views with
  | (_, _, ws) :: _ => linkMsgs x ws w
  | [] => []

theorem linkMsgs_eq {x : Msg} {r : XR} (hr : readX x = some r) {s N : ℕ} {c : Circuit}
    (hc : linkCircuit r = some (s, N, c)) {UB PB : ℕ} (hb : linkBounds r = some (UB, PB))
    {p : Readable.Parsed} (hp : innerParsed r = some p) {st' : Stmt} (hst : RecScope.innerStmt (filesOf r) = some st')
    {R W : ByteArray} (hR : RecLinkTree.rowsRoot UB PB (RecLinkPart.innerRows st').1 = .ok R)
    (hW : RecLinkTree.rowsRoot UB PB (RecLinkPart.innerRows st').2 = .ok W) (ws : List Claims.Flock.Wit)
    (w : Claims.Flock.Wit) :
    linkMsgs x ws w = (List.range N).flatMap (instMsgs p w UB PB st' c (ws.getD s ⟨[]⟩) R W) := by
  simp only [linkMsgs, hr, hc, hb, hp, hst, hR, hW, Except.toOption, Option.getD_some]

/-! ## The `trow` port's words are its row's bytes -/

/-- **Instance `n`'s `trow` words are the bytes of its row**: the `trow` port is 1024 bits, so its row is 128 bytes
and reads back from its u16 words (`bytesOf_wordsOf`). -/
theorem trow_words {c : Circuit} {q : ℕ} (hq : c.ports.findIdx? (·.name == "trow") = some q)
    (hqs : q < c.ports.size) (hb : c.ports[q].bits = 1024) (w : Claims.Flock.Wit) (n : ℕ) :
    RecLinkRef.bytesOf (inWordsAt c w "trow" n) = ⟨(trowAt c w n).toArray⟩ := by
  have hev : (trowAt c w n).length % 2 = 0 := by
    unfold trowAt
    rw [readForms_length, List.length_map]
    cases h : rowHidden c "trow" n with
    | none => simp
    | some ps =>
      unfold rowHidden at h
      obtain ⟨q', hq', h⟩ := Option.bind_eq_some_iff.1 h
      obtain ⟨pt, hpt, h⟩ := Option.bind_eq_some_iff.1 h
      rw [hq] at hq'
      obtain rfl : q = q' := Option.some.inj hq'
      rw [Array.getElem?_eq_getElem hqs] at hpt
      obtain rfl : c.ports[q] = pt := Option.some.inj hpt
      have hl : ps.length = 1024 := by rw [(FlockDeploy.mapM_some_get h).1, List.length_range, hb]
      simp [hl]
  show RecLinkRef.bytesOf (RecLinkRef.wordsOf ⟨(trowAt c w n).toArray⟩) = _
  refine FlockVBridge.RecLink.bytesOf_wordsOf _ ?_
  show (trowAt c w n).toArray.size % 2 = 0
  rw [List.size_toArray]
  exact hev

/-! ## The link clause, or a collision -/

/-- **The link clause holds at the staged parts' witnesses, or two of `linkMsgs` collide**: where each of the
statement's parts is staged and its witness among `ws` checks against its V\* statement, a witness `w` of the inner
relation at the statement's inner statement satisfies `LinksHold` at the rows `tableRows` reads off `ws`, or two of
`linkMsgs x ws w` are distinct with one SHA-512. -/
theorem linksHold_or_coll (hA : LinkRefStaged) {st : Stage} {x : Msg} {ws : List Claims.Flock.Wit}
    (hS : ∀ s < nS x, ∃ xO, StagedPart st s x xO ∧
      Claims.Flock.check (deployZkRec Readable.relation).parse xO (ws.getD s ⟨[]⟩) = true)
    {cs : Claims.Recursion.CompiledStatement} (hx : parse x = some cs) {w : Claims.Flock.Wit}
    (hR : Claims.Flock.R (FlockDeploy.parse false Readable.relation) cs.inner w) :
    LinksHold x w (tableRows x ws) ∨
      ∃ a ∈ linkMsgs x ws w, ∃ b ∈ linkMsgs x ws w, Claims.Collision FlockDeploy.sha512 a b := by
  by_cases hcol : ∃ a ∈ linkMsgs x ws w, ∃ b ∈ linkMsgs x ws w, Claims.Collision FlockDeploy.sha512 a b
  · exact .inr hcol
  refine .inl ?_
  intro r hr T N hl
  obtain ⟨r', p, f, spec', st', st0, hr', hp, hP, hpc, hh, hRS, hst0, hfs, hstm, e1, e2, e3⟩ := inner_of_R hx hR
  obtain rfl : r = r' := Option.some.inj (hr.symm.trans hr')
  -- the link part, V*'s statement last but one, is staged
  obtain ⟨_, _, hpos⟩ := capAt_readX hr
  obtain ⟨xO0, hS0, -⟩ := hS 0 hpos
  obtain ⟨ss, UB, PB, hss, hat, hla⟩ := linkAt_staged hS0 hr hl
  have hsl : ss.size - 2 < nS x := by
    obtain ⟨vs, hvs, hlen⟩ := readX_statements hr
    rw [hss] at hvs
    obtain rfl : ss = vs := Except.ok.inj hvs
    obtain ⟨hlt, -⟩ := Array.getElem?_eq_some_iff.1 hat
    have hn : nS x = r.w.parts.length := by simp [nS, hr]
    omega
  obtain ⟨xO, hSl, hck⟩ := hS _ hsl
  obtain ⟨c, hcc⟩ := staged_circuit hSl hr
  have hlc : linkCircuit r = some (ss.size - 2, N, c) := by
    simp only [linkCircuit, hla, hcc, Option.bind_eq_bind, Option.bind_some, Option.pure_def]
  have hlb : linkBounds r = some (UB, PB) := by
    simp only [linkBounds, hss, Except.toOption, Option.bind_some, hat]
  obtain ⟨u, hu, hrw⟩ := CapLower.rows_staged hSl hr hss hat hcc
  have hu : VStar.linkUnit UB PB = .ok u := hu
  obtain ⟨q, hq, hqs, -, hbits⟩ := trow_port hu hrw
  obtain ⟨ms, i, spec, sts, rfl, hsu, -⟩ := id hSl
  obtain ⟨h0, own, f₁, sf, R, W, hown, hsrc, hsf, hRr, hWr, -, ⟨wi, hrin⟩, ⟨wo, hrout⟩, hidx, hNn, hnn⟩ :=
    linkSetup_staged hSl hr hss hat hcc hl hsu
  -- the files the session sets up from are the statement's, so `R` and `W` are over the inner statement's rows
  obtain ⟨hins, -⟩ := setupO_some hsu
  obtain ⟨r2, rec, hr2, pj, hpj, hsrc'⟩ := InnerAt.src_staged hins
  obtain rfl : r = r2 := Option.some.inj (hr.symm.trans hr2)
  have hf₁ : f₁ = ⟨r.innerRaw.circuit, r.innerRaw.publicFile, pj, rec, none, some (innerFormOf r.innerRaw),
      r.innerRaw.program⟩ :=
    RecCoins.Source.files.inj (Option.some.inj (hsrc.symm.trans hsrc'))
  have hf : f = ⟨r.innerRaw.circuit, r.innerRaw.publicFile, pj, "", none, some (innerFormOf r.innerRaw),
      r.innerRaw.program⟩ := by
    have hfs' := hfs
    rw [filesOf, hpj] at hfs'
    simp only [Option.map_some, Option.some.injEq, RecCoins.Source.files.injEq] at hfs'
    exact hfs'.symm
  have hsf' : f.stmt "the inner session" = .ok sf := by
    have h := InnerAt.stmt_unit (u' := "the inner session") hsf
    rw [hf₁] at h
    rw [hf]
    exact h
  rw [hstm] at hsf'
  obtain rfl : (spec', st') = sf := Except.ok.inj hsf'
  have hinS : RecScope.innerStmt (filesOf r) = some st' := by
    rw [hfs]
    simp [RecScope.innerStmt, hstm, Except.toOption]
  have hlm := linkMsgs_eq hr hlc hlb hp hinS hRr hWr ws w
  have hrows : tableRows x ws = (List.range N).map (trowAt c (ws.getD (ss.size - 2) ⟨[]⟩)) := by
    simp only [tableRows, hr, hlc]
  -- each instance's row holds at the inner witness
  have key : ∀ n < N, LinkHolds p w (trowAt c (ws.getD (ss.size - 2) ⟨[]⟩) n) := by
    intro n hn
    have hmem : ∀ m ∈ instMsgs p w UB PB st' c (ws.getD (ss.size - 2) ⟨[]⟩) R W n, m ∈ linkMsgs x ws w :=
      fun m hm => by rw [hlm]; exact List.mem_flatMap.2 ⟨n, List.mem_range.2 hn, hm⟩
    rcases hA hSl hck hr hss hat hcc hsu h0 hown hidx hNn hnn hRr hWr hrin hrout hn with hok | ⟨a, ha, b, hb, hab⟩
    swap
    · exact (hcol ⟨a, hmem a (List.mem_append_left _ ha), b, hmem b (List.mem_append_left _ hb), hab⟩).elim
    have hok' : RecLinkRef.linkRef UB PB (inWordsAt c (ws.getD (ss.size - 2) ⟨[]⟩) "trow" n)
        ((inWordsAt c (ws.getD (ss.size - 2) ⟨[]⟩) "xyr" n).take 32)
        ((inWordsAt c (ws.getD (ss.size - 2) ⟨[]⟩) "xyr" n).drop 32) (inWordsAt c (ws.getD (ss.size - 2) ⟨[]⟩) "raux" n)
        ((inWordsAt c (ws.getD (ss.size - 2) ⟨[]⟩) "xyw" n).take 32)
        ((inWordsAt c (ws.getD (ss.size - 2) ⟨[]⟩) "xyw" n).drop 32) (inWordsAt c (ws.getD (ss.size - 2) ⟨[]⟩) "waux" n) =
        .ok (RecLinkRef.expected R, RecLinkRef.expected W) := hok
    rcases inner_link hP hpc hh hRS hst0 e1 e2 e3 hRr hWr hok' (trow_words hq hqs hbits _ n) with h | ⟨a, ha, b, hb, hab⟩
    · exact h
    · exact (hcol ⟨a.data.toList, hmem _ (List.mem_append_right _ (List.mem_map.2 ⟨a, ha, rfl⟩)),
        b.data.toList, hmem _ (List.mem_append_right _ (List.mem_map.2 ⟨b, hb, rfl⟩)), collision_of_collides hab⟩).elim
  refine ⟨by rw [hrows, List.length_map, List.length_range], ?_, ?_⟩
  · intro s' N' c' dom hlc' hdom n hn
    rw [hlc] at hlc'
    simp only [Option.some.injEq, Prod.mk.injEq] at hlc'
    obtain ⟨-, -, rfl⟩ := hlc'
    have hn' : n < N := by
      have := hn
      rw [hrows, List.length_map, List.length_range] at this
      exact this
    have e : (tableRows x ws)[n] = trowAt c (ws.getD (ss.size - 2) ⟨[]⟩) n := by
      rw [List.getElem_of_eq hrows hn]
      simp
    rw [e]
    exact rowOpens_staged hSl hck hr hcc hss hat hl hdom hn'
  · intro p' hp' row hrow
    obtain rfl : p = p' := Option.some.inj (hp.symm.trans hp')
    rw [hrows] at hrow
    obtain ⟨n, hn, rfl⟩ := List.mem_map.1 hrow
    exact key n (List.mem_range.1 hn)

/-! ## At the compiled record -/

/-- **The links bind at every compiled record `deployRec`** whose inner relation is the deployed one's: on an accepted
run where the linked extractor recovers a witness in `R` whose link clause fails, `E_O` gave witnesses at the judged
draw (`EO_judged_of_E`), each checking against the V\* statement staged there (`eoGives_outerHolds`,
`runVstar_staged`, `stagedAll_run`), so two of the link finder's inputs collide (`linksHold_or_coll`). -/
theorem linkBinds_rec (hA : LinkRefStaged) {I : Claims.Recursion.Inner FlockDeploy.Coin Claims.Flock.Wit}
    (hIR : ∀ y w, I.R y w → Claims.Flock.R (FlockDeploy.parse false Readable.relation) y w) (ivScope : Msg → Bool)
    (b : Brings) (decO : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.ZkTrace) (x : Msg) :
    Claims.Recursion.LinkBindsAt FlockDeploy.sha512 (deployRec I.V ivScope b) I (deployZkRec Readable.relation) decO
      LinksHold tableRows linkIns x := by
  intro Ω _ _ P ω r₀ c w rs hacc hEL hRx hQ
  rw [Claims.Recursion.EL_eval_fst] at hEL
  obtain ⟨w', hw', hpr⟩ := Option.map_eq_some_iff.1 hEL
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj hpr
  obtain ⟨ws, hws⟩ := Claims.Recursion.EO_judged_of_E _ _ _ _ _ P x ω r₀ c hw'
  have hg : Claims.Recursion.EOGives FlockDeploy.sha512 (deployRec I.V ivScope b) I (deployZkRec Readable.relation)
      decO P x ω r₀ (Claims.Recursion.judgedC c) ws := And.intro hacc hws
  have ho := Claims.Recursion.eoGives_outerHolds hg
  rw [runVstar_staged _ I ivScope b P ω r₀ hacc] at ho
  obtain ⟨hlen, hchk⟩ := ho
  have hlen : ws.length = nS x := hlen
  have hsp := stagedAll_run FlockDeploy.sha512 I.V b x P ω r₀ hacc
  have hS : ∀ s < nS x, ∃ xO, StagedPart b.st s x xO ∧
      Claims.Flock.check (deployZkRec Readable.relation).parse xO (ws.getD s ⟨[]⟩) = true := by
    intro s hs
    have hs' : s < ws.length := by rw [hlen]; exact hs
    refine ⟨_, hsp s hs, ?_⟩
    have e : ws.getD s ⟨[]⟩ = ws[s] := by simp [List.getD_eq_getElem?_getD, hs']
    rw [e]
    exact hchk s hs'
  obtain ⟨cs, hcs, -, hRi⟩ := hRx
  rw [hws, Option.getD_some] at hQ
  rcases linksHold_or_coll hA hS hcs (hIR _ _ hRi) with hL | ⟨a, ha, a', ha', hc⟩
  · exact (hQ hL).elim
  · refine ⟨a, ?_, a', ?_, hc⟩ <;>
      simp only [Claims.Recursion.linkInputs, Claims.Recursion.drawsOf, List.map_cons, Claims.Recursion.drawView,
        hws, Option.getD_some, linkIns] <;>
      assumption

/-- **`LinksBindDeployed` from the link part's reference step**: `linkBinds_rec` at the deployed compiled record
(`recBound`, `deployRec` at `IVStar`) and bound record, whose inner relation is the deployed plain one's. -/
theorem linksBindDeployed_of (hA : LinkRefStaged) :
    _root_.Flock.Guarantees.LinksBindDeployed LinksHold tableRows linkIns := by
  intro b aux nOf hd recRest decI _ decO _ x _
  exact linkBinds_rec hA (I := Claims.Recursion.Inner.ofFlock (bdDeployed Readable.relation
    (FlockDeploy.nextR DeployedR.rdR) (FlockDeploy.deployPlainTrimR Readable.relation DeployedR.rdR).cols
    DeployedR.wER FlockDeploy.joinR rowInDeployed DeployedR.finalR DeployedR.wR).deploy decI)
    (fun _ _ h => h) _ _ decO x

/-- **Knowledge soundness at the linked relation at the deployment, from the link part's reference step**:
`recursiveKSLink_of` at the link clause `LinksHold`, the rows `tableRows` and the finder's inputs `linkIns`. -/
theorem recursiveKSLink_links (hA : LinkRefStaged) (b : Brings) (aux : Msg → ℕ → _root_.Flock.RecRecord.Round)
    (nOf : Msg → ℕ → ℕ) (hd : Msg → _root_.Flock.RecRecord.Record)
    (recRest : Msg → List (Msg × Claims.Out) → List UInt8) :
    _root_.Flock.Guarantees.RecursiveKSLinkAt DeployedR.rdR DeployedR.finalR OpenRows.elemR DeployedR.wER DeployedR.wR
      (deployedB b) aux nOf hd recRest LinksHold tableRows linkIns :=
  _root_.Flock.Guarantees.recursiveKSLink_of (linksBindDeployed_of hA) b aux nOf hd recRest

end RecDeploy.Links
