import Proofs.Flock.Claims.Recursion.ResChainW
import Proofs.Flock.Claims.Recursion.ResStepsW
import Proofs.Flock.Claims.Recursion.AccTies
import Proofs.Flock.Claims.Recursion.ResCopyRows
import Proofs.Flock.Claims.Recursion.TopsOpens
import Proofs.Flock.VBridge.AlgHolds
import Proofs.Flock.VBridge.Family.PartKeys

/-!
# P3's ends half: the copy's accumulator ports read the outer witnesses' chain at the rep's ends

`EndsW` (`ResChainW`) asks, in a piece's port context, that every element of the copy whose `w` index is past the frame
(`accBase`) be one of `w`'s last four, and that the copy read `chainW` there at its rep's ends (`D.accE`). The pieces:

* `acc_elem`: an element of a part's `w` index at or past the frame values is element `o < 2` of port `B.size + b`
  (`b < 2`, `acc0` or `acc1`), at `w`'s element `B.sum + 2b + o`.
* `copy_rowElems` (CAPPIECES' `ResCopyRows`): the copy's element is element `o` of that port's committed row at the
  piece's instance `n`.
* `acc0_held`, `acc1_held`: an algebra part holds row `n · R` at `("acc0", n)` and row `n · R + R` at `("acc1", n)`
  (`accRead`), so the row has an owner (`accCanon_some`), whose bytes `chainW` reads (`chainW_get`).
* `InnerAccTieV` (AccTies): the holder's and the owner's slots carry one commit string, wherever the brings list the
  registered reads (`innerAccTieV`), so they hold one row (`held_rows_eq`).
* `repOpens_eq`: the rep's run length `R` is `D.accE`'s `nOpen` at table 0's schedule.
-/

namespace RecDeploy.ResLower

open Flock
open Claims (Msg)
open FlockDeploy (Coin FinalP setupOf)
open Claims.Recursion.VStarB

/-! ## Arithmetic -/

theorem split_sum : ∀ (ns : List ℕ) (e : ℕ), e < ns.sum →
    ∃ i, ∃ hi : i < ns.length, ∃ k < ns[i], e = (ns.take i).sum + k
  | [], e, h => by simp at h
  | n :: ns, e, h => by
    by_cases he : e < n
    · exact ⟨0, by simp, e, he, by simp⟩
    · have h' : e - n < ns.sum := by simp only [List.sum_cons] at h; omega
      obtain ⟨i, hi, k, hk, hek⟩ := split_sum ns (e - n) h'
      refine ⟨i + 1, by simp only [List.length_cons]; omega, k, hk, ?_⟩
      simp only [List.take_succ_cons, List.sum_cons]
      omega

theorem sizes_sum {α : Type} (A : Array (Array α)) : (A.map (·.size)).toList.sum = A.flatten.size := by
  rw [← Array.length_toList, Array.toList_flatten, List.length_flatten, Array.toList_map, List.map_map]
  rfl

theorem take_sum_append (l : List ℕ) (a b : ℕ) : ((l ++ [a, b]).take (l.length + 1)).sum = l.sum + a := by
  induction l with
  | nil => simp
  | cons x l ih =>
    rw [List.cons_append, List.length_cons, List.take_succ_cons, List.sum_cons, ih, List.sum_cons]
    omega

theorem lit4_get (a b c d : _root_.Flock.F128) :
    #[a, b, c, d][0]! = a ∧ #[a, b, c, d][1]! = b ∧ #[a, b, c, d][2]! = c ∧ #[a, b, c, d][3]! = d :=
  ⟨rfl, rfl, rfl, rfl⟩

theorem sizes_acc (B : Array ℕ) {b : ℕ} (hb : b < 2) : (B ++ #[2, 2])[B.size + b]! = 2 := by
  rw [getElem!_pos (B ++ #[2, 2]) (B.size + b) (by simp; omega), Array.getElem_append_right (by omega)]
  rcases (show b = 0 ∨ b = 1 by omega) with rfl | rfl <;> simp

theorem names_acc (n : ℕ) (f : ℕ → String) {b : ℕ} (hb : b < 2) :
    ((Array.range n).map f ++ #["acc0", "acc1"])[n + b]! = if b = 0 then "acc0" else "acc1" := by
  rw [getElem!_pos ((Array.range n).map f ++ #["acc0", "acc1"]) (n + b) (by simp; omega),
    Array.getElem_append_right (by simp)]
  rcases (show b = 0 ∨ b = 1 by omega) with rfl | rfl <;> simp

/-- **An element past the frame is an accumulator's**: where element `e` of a part's `w` index (`wIdx`, over ports of
sizes `B ++ [2, 2]`) is `w`'s element `j ≥ B.sum`, it is element `o < 2` of the part's row port `i`, which is port
`B.size + b` (`b < 2`), and `j = B.sum + 2b + o`. -/
theorem acc_elem {B P : Array ℕ} {e j : ℕ} (he : (FlockVBridge.PartS.wIdx (B ++ #[2, 2]) P)[e]? = some j)
    (hj : B.toList.sum ≤ j) :
    ∃ i, ∃ hi : i < P.size, ∃ b < 2, ∃ o < 2, P[i] = B.size + b ∧ j = B.toList.sum + 2 * b + o ∧
      e = ((P.toList.map ((B ++ #[2, 2])[·]!)).take i).sum + o := by
  obtain ⟨hea, hej⟩ := Array.getElem?_eq_some_iff.1 he
  have hea' := hea
  rw [FlockVBridge.OfVStar.wIdx_size] at hea'
  obtain ⟨i, hi, o, ho, hae⟩ := split_sum _ e hea'
  have hi' : i < P.size := by simpa using hi
  have ho' : o < (B ++ #[2, 2])[P[i]]! := by
    simp only [List.getElem_map, Array.getElem_toList] at ho
    exact ho
  have hw := FlockVBridge.OfVStar.wIdx_get (B ++ #[2, 2]) P hi' ho' hea hae
  rw [hej] at hw
  have hT : (B ++ #[2, 2]).toList = B.toList ++ [2, 2] := by simp
  have hst : ∀ t ≤ (B ++ #[2, 2]).size,
      (RecResiduals.starts (B ++ #[2, 2]))[t]! = ((B.toList ++ [2, 2]).take t).sum := fun t ht => by
    rw [FlockVBridge.Family.starts_get _ ht, hT]
  have hsz : (B ++ #[2, 2]).size = B.size + 2 := by simp
  rcases Nat.lt_or_ge P[i] B.size with hlt | hge
  · exfalso
    have hgt : (B ++ #[2, 2])[P[i]]! = (B.toList ++ [2, 2])[P[i]]'(by simp; omega) := by
      rw [getElem!_pos _ _ (by rw [hsz]; omega)]
      simp only [← hT, Array.getElem_toList]
    rw [hgt] at ho'
    have hlt' := FlockVBridge.OfVStar.take_sum_lt (B.toList ++ [2, 2]) (by simp; omega) ho' hlt
    have htk : ((B.toList ++ [2, 2]).take B.size).sum = B.toList.sum := by
      rw [List.take_append_of_le_length (by simp), List.take_of_length_le (by simp)]
    rw [hst _ (by rw [hsz]; omega)] at hw
    omega
  · rcases Nat.lt_or_ge P[i] (B.size + 2) with hlt2 | hge2
    · refine ⟨i, hi', P[i] - B.size, by omega, o, ?_, by omega, ?_, ?_⟩
      · have h2 := sizes_acc B (b := P[i] - B.size) (by omega)
        rw [show B.size + (P[i] - B.size) = P[i] by omega] at h2
        rw [h2] at ho'
        exact ho'
      · rw [hw, hst _ (by rw [hsz]; omega)]
        rcases (show P[i] = B.size ∨ P[i] = B.size + 1 by omega) with h | h <;> rw [h]
        · rw [List.take_append_of_le_length (by simp), List.take_of_length_le (by simp)]
          omega
        · have e := take_sum_append B.toList 2 2
          rw [Array.length_toList] at e
          rw [e]
          omega
      · exact hae.trans (by simp)
    · exfalso
      rw [getElem!_neg (B ++ #[2, 2]) P[i] (by rw [hsz]; omega)] at ho'
      exact Nat.not_lt_zero _ ho'

/-! ## The chain's owners -/

/-- **`chainW` at an owned row**: its two elements are the owner's bytes' (`rowW`). -/
theorem chainW_get {x : Msg} {ws : List Claims.Flock.Wit} {r : XR} (hr : readX x = some r) {p o : ℕ} {co : Circuit}
    {ho : String × ℕ} (hc : accCanon r p = some (o, co, ho)) {wo : Claims.Flock.Wit} (hw : ws[o]? = some wo) (t : ℕ)
    {k : ℕ} (hk : k < 2) : (chainW x ws t p)[k]! = elemsOf (readSegs wo (accSegs co ho)) 2 k := by
  have hW : rowW x ws p = some (readSegs wo (accSegs co ho)) := by
    simp only [rowW, hr, Option.bind_some, hc, hw, Option.map_some]
  unfold chainW
  rw [hW]
  rcases (show k = 0 ∨ k = 1 by omega) with rfl | rfl <;> rfl

/-- **A held row has an owner**: where part `s`, of a circuit, holds row `p`, `accCanon` names one. -/
theorem accCanon_some {r : XR} {p s : ℕ} (hs : s < r.w.parts.length) {c : Circuit}
    (hc : r.w.parts[s]?.bind partCircuit = some c) {h : String × ℕ} (hh : h ∈ accHoldings r s p) :
    ∃ e, accCanon r p = some e := by
  obtain ⟨h0, hd⟩ : ∃ h0, (accHoldings r s p).head? = some h0 := by
    cases hl : accHoldings r s p with
    | nil => rw [hl] at hh; cases hh
    | cons a l => exact ⟨a, rfl⟩
  have hH : accHolder r p s = some (s, c, h0) := by
    simp only [accHolder, hc, Option.bind_some, hd, Option.map_some]
  unfold accCanon
  cases h1 : (List.range r.w.parts.length).findSome? (fun s =>
      if (accHoldings r s p).length = 1 then accHolder r p s else none) with
  | some e => exact ⟨e, by simp⟩
  | none =>
    cases h2 : (List.range r.w.parts.length).findSome? (accHolder r p) with
    | some e => exact ⟨e, by simp [h2]⟩
    | none =>
      exfalso
      have := List.findSome?_eq_none_iff.1 h2 s (List.mem_range.2 hs)
      rw [hH] at this
      cases this

theorem acc0_mem {j : ℕ} {nm : String} {pt : VStar.Part} (ha : "acc0" ∈ pt.S.ports.map (·.1)) :
    "acc0" ∈ RecVStar.accPorts (.inner j nm pt) :=
  List.mem_filter.2 ⟨ha, by decide⟩

theorem acc1_mem {j : ℕ} {nm : String} {pt : VStar.Part} (ha : "acc1" ∈ pt.S.ports.map (·.1)) :
    "acc1" ∈ RecVStar.accPorts (.inner j nm pt) :=
  List.mem_filter.2 ⟨ha, by decide⟩

/-- **An algebra part holds its rep's first row at `acc0`**: row `n · R` at `("acc0", n)`, `n < 2`. -/
theorem acc0_held {r : XR} {s j : ℕ} {nm : String} {pt : VStar.Part}
    (hsv : (RecVStar.statements r.shape).toOption.bind (·[s]?) = some (.inner j nm pt)) {n : ℕ} (hn : n < 2)
    (ha : "acc0" ∈ pt.S.ports.map (·.1)) : ("acc0", n) ∈ accHoldings r s (n * repOpens r) := by
  have hR : accRead r (.inner j nm pt) "acc0" n = some (n * repOpens r) := by
    simp only [accRead]
    rw [if_pos ⟨hn, acc0_mem ha⟩, if_pos trivial]
  unfold accHoldings
  rw [hsv]
  refine List.mem_filter.2 ⟨?_, by simp only [hR, beq_self_eq_true]⟩
  unfold accCands
  rcases (show n = 0 ∨ n = 1 by omega) with rfl | rfl <;> simp

/-- **An algebra part holds its rep's last row at `acc1`**: row `n · R + R` at `("acc1", n)`, `n < 2`. -/
theorem acc1_held {r : XR} {s j : ℕ} {nm : String} {pt : VStar.Part}
    (hsv : (RecVStar.statements r.shape).toOption.bind (·[s]?) = some (.inner j nm pt)) {n : ℕ} (hn : n < 2)
    (ha : "acc1" ∈ pt.S.ports.map (·.1)) : ("acc1", n) ∈ accHoldings r s (n * repOpens r + repOpens r) := by
  have hR : accRead r (.inner j nm pt) "acc1" n = some (n * repOpens r + repOpens r) := by
    simp only [accRead]
    rw [if_pos ⟨hn, acc1_mem ha⟩, if_neg (by decide), if_pos trivial]
  unfold accHoldings
  rw [hsv]
  refine List.mem_filter.2 ⟨?_, by simp only [hR, beq_self_eq_true]⟩
  unfold accCands
  rcases (show n = 0 ∨ n = 1 by omega) with rfl | rfl <;> simp

/-! ## The run length -/

theorem nOpen_opensBelow (sch : Schedule) : Rows.nOpen sch = RecVStar.opensBelow sch sch.levels.size := by
  simp only [Rows.nOpen, RecVStar.opensBelow, Array.getElem!_eq_getD]

/-- **The rep's run is `D.accE`'s**: at a decision whose setup is one table at the inner shape's `m`, the rows a rep's
`acc0` and `acc1` read are `nOpen` apart at table 0's schedule. -/
theorem repOpens_eq {rd : Msg → Option FinalP} {xi : Msg} {ms : List Msg} {cs : List Coin}
    (D : Rows.DecAt rd xi ms cs) {v : FlockDeploy.Ins × SessionSpec × Array Stmt} (hv : setupOf false xi = some v)
    {r : XR} {st : Stmt} (hst : v.2.2 = #[st]) (hm : st.m = r.shape.m) :
    repOpens r = Rows.nOpen ((fast100 (FlockDeploy.tabsOf D.sts)[0]!.m).getD default) := by
  obtain ⟨-, hsts⟩ := D.setup_eq
  obtain ⟨v', c₁, r₀, rss, fl, link, pf, hv', -, -, -, -, -, -, hd⟩ := Rows.decR_last D.dec
  have hvv : v' = v := Option.some.inj (hv'.symm.trans hv)
  have hds : D.sts = #[st] := by rw [hsts, hd, hvv, hst]
  have hm0 : (FlockDeploy.tabsOf D.sts)[0]!.m = r.shape.m := by
    have htabs : FlockDeploy.tabsOf #[st] = #[_root_.Flock.Setup.ofCircuit st] := by simp [FlockDeploy.tabsOf]
    rw [hds, htabs, ← hm]
    rfl
  rw [hm0]
  unfold repOpens
  cases fast100 r.shape.m with
  | none => rfl
  | some sch => exact (nOpen_opensBelow sch).symm

/-! ## The ends -/

/-- **A held row is its owner's**: where the holder's slot and the owner's carry one commit string (`hC`), part `s`'s
row at a holding `h` of row `p` is what `chainW` reads at `p`, element by element. -/
theorem held_chainW (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) {x : Msg}
    {ws : List Claims.Flock.Wit} (hT : ∀ e ∈ (DeployedR.layR elem).allCalls x ws, FlockDeploy.sha512 e.1 = e.2)
    (hnc : (Claims.findColl ((DeployedR.layR elem).allCalls x ws)).isSome = false) {r : XR} (hr : readX x = some r)
    (hl : ws.length = r.w.parts.length) {s : ℕ} {cS : Circuit} (hpc : r.w.parts[s]?.bind partCircuit = some cS)
    {p : ℕ} {h : String × ℕ} (hh : h ∈ accHoldings r s p) {wt : Claims.Flock.Wit} (hw : ws[s]? = some wt) {q : ℕ}
    (hq : cS.ports.findIdx? (·.name == h.1) = some q)
    (hC : ∀ o co ho, accCanon r p = some (o, co, ho) → ∀ wo, ws[o]? = some wo →
      ∃ q', co.ports.findIdx? (·.name == ho.1) = some q' ∧
        readForms wt (hmCommit cS q h.2) = readForms wo (hmCommit co q' ho.2))
    (t : ℕ) {k : ℕ} (hk : k < 2) :
    elemsOf (readSegs wt (accSegs cS h)) 2 k = (chainW x ws t p)[k]! := by
  have hsl : s < r.w.parts.length := by
    rw [← hl]
    exact (List.getElem?_eq_some_iff.1 hw).1
  obtain ⟨⟨o, co, ho⟩, hcan⟩ := accCanon_some hsl hpc hh
  obtain ⟨hol, hpco, hhd⟩ := accCanon_spec hcan
  dsimp only at hol hpco hhd
  have hol' : o < ws.length := by rw [hl]; exact hol
  have hwo : ws[o]? = some ws[o] := List.getElem?_eq_getElem hol'
  obtain ⟨q', hq', hcm⟩ := hC o co ho hcan ws[o] hwo
  have hho : ho ∈ accHoldings r o p := List.mem_of_mem_head? hhd
  rw [chainW_get hr hcan hwo t hk]
  congr 1
  exact held_rows_eq elem hT hnc hr hpc hpco hw hwo (mem_accHeld hh) (mem_accHeld hho) hq hq' hcm

/-- **`part`'s row sizes are its ports'**: where `part sh j` builds, S's structure and groups return at `sh`, group `j`
is `(R, P, V)`, and the part's row sizes are `P`'s ports' (`portSizes`). -/
theorem part_ns {sh : _root_.Flock.Shape} {jj : ℕ} {S : _root_.Flock.RecOpen.GfS} {ns : List ℕ} {nv : ℕ}
    (h : _root_.Flock.RecResiduals.part sh jj = .ok (S, ns, nv)) :
    ∃ (S0 : _root_.Flock.RecAlgebra.GfS) (gs : Array (Array ℕ × Array ℕ × Array ℕ)) (R P V : Array ℕ),
      _root_.Flock.RecAlgebra.structureOf sh = .ok S0 ∧
      _root_.Flock.RecResiduals.groups S0 (_root_.Flock.RecResiduals.portSizes sh) = .ok gs ∧
      gs[jj]? = some (R, P, V) ∧ ns = P.toList.map ((_root_.Flock.RecResiduals.portSizes sh)[·]!) := by
  unfold _root_.Flock.RecResiduals.part at h
  obtain ⟨S0, hS0, h⟩ := FlockVBridge.PartS.bind_ok h
  dsimp only at h
  split at h
  · cases h
  · obtain ⟨gs, hgs, h⟩ := FlockVBridge.PartS.bind_ok h
    try dsimp only at h
    cases hj : gs[jj]? with
    | none => rw [hj] at h; cases h
    | some t =>
      obtain ⟨R, P, V⟩ := t
      rw [hj] at h
      try dsimp only at h
      obtain ⟨S2, hS2, h⟩ := FlockVBridge.PartS.bind_ok h
      simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
      obtain ⟨-, rfl, -⟩ := h
      exact ⟨S0, gs, R, P, V, hS0, hgs, hj, rfl⟩

theorem partPorts_of {sh : _root_.Flock.Shape} {jj : ℕ} {S0 : _root_.Flock.RecAlgebra.GfS}
    {gs : Array (Array ℕ × Array ℕ × Array ℕ)} {R P V : Array ℕ}
    (hS : _root_.Flock.RecAlgebra.structureOf sh = .ok S0)
    (hg : _root_.Flock.RecResiduals.groups S0 (_root_.Flock.RecResiduals.portSizes sh) = .ok gs)
    (hj : gs[jj]? = some (R, P, V)) : RecVStar.partPorts sh jj = .ok P := by
  unfold RecVStar.partPorts
  simp only [hS, hg, hj, bind, Except.bind, pure, Except.pure]

set_option maxHeartbeats 1000000 in
/-- **P3's ends half**, under the commit tie (`InnerAccTieV`): in a piece's port context, at a decision, each element
of the copy past the frame is one of `w`'s last four, `acc0`'s or `acc1`'s element `o` (`acc_elem`), and the copy holds
there its committed row's element at the piece's instance `n` (`copy_rowElems`), which is the row its port reads
(`acc0_held`, `acc1_held`: `n · R`, `n · R + R`), so its owner's (`held_chainW`), which `D.accE` reads at the rep's ends
(`repOpens_eq`). -/
theorem endsW_of (rd : Msg → Option FinalP) (wE : Msg → ℝ) (final : Msg → Option (List (List (Msg × Salt)) × Msg))
    (w : Msg → ℝ) (b : Brings) (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    (untr : TrReader) (cmt : Msg → ℕ → Msg → Msg)
    (hC : InnerAccTieV rd wE final w b elem Claims.Recursion.Form.Dense untr cmt) :
    EndsW rd wE final w b elem Claims.Recursion.Form.Dense untr cmt := by
  intro x st hx hf c ws hJ xs hxs hcol hms s wt hw r hr ss hss j nm pt hsv xO p hxs0 hxO hck hsh hAg hT hc q hq ho ht
    m k cl jj sh holo S ns nv hpar hshp hholo hpart os last hfin hos D e j' he hj
  -- the piece's part is group `q.key.1`'s, at the read shape
  obtain ⟨r', ss', nm', pt', hr', hss', hsv'⟩ := DecompBR.ownRD_some ho
  obtain rfl : r = r' := Option.some.inj (hr.symm.trans hr')
  obtain rfl : ss = ss' := Except.ok.inj (hss.symm.trans hss')
  have hsv3 : VStar.Stmt.inner q.key.1 nm' pt' = .inner j nm pt := Option.some.inj (hsv'.symm.trans hsv)
  simp only [VStar.Stmt.inner.injEq] at hsv3
  obtain ⟨hj1, -, -⟩ := hsv3
  subst hj1
  have hsvO : (RecVStar.statements r.shape).toOption.bind (·[s]?) = some (.inner q.key.1 nm pt) := by
    rw [hss]
    exact hsv
  obtain ⟨p', r'', sv, hx', htg, hpi, hr'', ⟨ss'', hss'', hsv''⟩, -, htab⟩ := stagedTables hsh.2
  obtain rfl : p = p' := Option.some.inj (hxO.symm.trans hx')
  obtain rfl : r = r'' := Option.some.inj (hr.symm.trans hr'')
  obtain rfl : ss = ss'' := Except.ok.inj (hss.symm.trans hss'')
  obtain rfl : VStar.Stmt.inner q.key.1 nm pt = sv := Option.some.inj (hsv.symm.trans hsv'')
  have hname : p.sts[0].c.unit.name = nm := (htab 0 ht).1
  have hn : RecResiduals.unitName m k cl jj =
      RecResiduals.unitName r.shape.m r.shape.kLog r.shape.claims q.key.1 :=
    (FlockVBridge.Family.resUnitName_params hpar).trans (hname.trans (DecompBR.inner_named hss hsv))
  obtain ⟨rfl, rfl, rfl, rfl⟩ := FlockVBridge.Family.resUnitName_inj hn
  -- the statement's one table is plain, at the inner shape
  obtain ⟨-, i0, spec, sts, -, hsu, -⟩ := hsh.2
  have hholo0 : RecResiduals.holoOf p.sts[0].c = .ok false := by
    have hpe : p = { i0.base.read with sts } := Option.some.inj (hxO.symm.trans (parse_of_setupO hsu))
    subst hpe
    exact staged_plain (x := x) (st0 := st) hx hf hsh hsu 0 ht
  obtain rfl : holo = false := Except.ok.inj (hholo.symm.trans hholo0)
  obtain ⟨v, hv, p0, r0, S0, ns0, nv0, hr0, -, hk, -⟩ := DecompBR.mem_resPD' hq
  obtain ⟨r3, ss3, sh0, hr3, -, hh, hsh0, hI, st1, hst1, hm1⟩ := DecompBR.shI_tie (x := x) ⟨st, hx, hf⟩ hv
  obtain rfl : r = r3 := Option.some.inj (hr.symm.trans hr3)
  obtain rfl : sh = sh0 := Option.some.inj (hshp.symm.trans hsh0)
  have hshI : DecompBR.shI v.2.1 v.2.2 = { sh with holo := false } := by rw [hI, hh]
  have hpart' : _root_.Flock.RecResiduals.part (DecompBR.shI v.2.1 v.2.2) q.key.1 = .ok (S, ns, nv) := by
    rw [hshI]
    exact hpart
  have hshA := shI_alg (xo := x) (sto := st) D hx hf rfl hv
  -- the element is `acc0`'s or `acc1`'s
  obtain ⟨S1, gs, R, P, V, hS1, hgs, hgj, hns⟩ := part_ns hpart'
  rw [elemPort_eq hv hshA hS1 hgs hgj e] at he
  rw [accBase_eq hv hshA] at hj ⊢
  have hN : _root_.Flock.RecAlgebra.nR (DecompBR.shI v.2.1 v.2.2) =
      ((_root_.Flock.RecAlgebra.rounds (DecompBR.shI v.2.1 v.2.2)).map (·.size)).toList.sum :=
    (sizes_sum _).symm
  have hps : _root_.Flock.RecResiduals.portSizes (DecompBR.shI v.2.1 v.2.2) =
      (_root_.Flock.RecAlgebra.rounds (DecompBR.shI v.2.1 v.2.2)).map (·.size) ++ #[2, 2] := rfl
  rw [hps] at he
  obtain ⟨i, hi, bb, hb, o, ho2, hPi, hj'', he'⟩ := acc_elem he (by rw [← hN]; exact hj)
  refine ⟨by omega, ?_⟩
  have hjo : j' - _root_.Flock.RecAlgebra.nR (DecompBR.shI v.2.1 v.2.2) = 2 * bb + o := by omega
  rw [hjo]
  subst hns
  have hiN : i < (P.toList.map ((_root_.Flock.RecResiduals.portSizes (DecompBR.shI v.2.1 v.2.2))[·]!)).length := by
    simpa using hi
  have hnsi : (P.toList.map ((_root_.Flock.RecResiduals.portSizes (DecompBR.shI v.2.1 v.2.2))[·]!))[i] = 2 := by
    simp only [List.getElem_map, Array.getElem_toList]
    rw [hPi, hps]
    exact sizes_acc ((_root_.Flock.RecAlgebra.rounds (DecompBR.shI v.2.1 v.2.2)).map (·.size)) hb
  have hoN : o < (P.toList.map ((_root_.Flock.RecResiduals.portSizes (DecompBR.shI v.2.1 v.2.2))[·]!))[i] := by
    rw [hnsi]
    exact ho2
  have he2 : e = ((P.toList.map ((_root_.Flock.RecResiduals.portSizes (DecompBR.shI v.2.1 v.2.2))[·]!)).take i).sum +
      o := by rw [hps]; exact he'
  -- the copy's element is its committed row's, at the piece's instance `r0`
  obtain ⟨cS, hpc⟩ := staged_circuit hsh.2 hr
  have hc0 : p.sts[0].c = cS := partCircuit_staged hsh.2 hxO htg hpi hr hpc ht
  rw [hc0] at hpar hholo
  have hkn : q.key.2.1 + q.key.2.2.1 = r0 := by simp only [hk, Nat.add_zero]
  have hkc : keyCopy p ht wt q.key = _root_.Flock.Readable.copyOf p 0 ht wt (r0 / cS.g) (r0 % cS.g) 0 := by
    simp only [keyCopy, hk, hc0, Nat.add_zero]
  rw [hkc, he2]
  obtain ⟨hi', -, hnames, hrow⟩ := copy_rowElems hsh.2 hr hss hsv hpc hxO ht hpar hshp hholo hpart hiN hoN r0
  have hPP : RecVStar.partPorts (DecompBR.shI v.2.1 v.2.2) q.key.1 = .ok P := partPorts_of hS1 hgs hgj
  rw [hshI] at hPP
  obtain ⟨_hPs, hnm⟩ := hnames P hPP
  rw [← hshI] at hnm
  have hA : P[i] = (_root_.Flock.RecAlgebra.rounds (DecompBR.shI v.2.1 v.2.2)).size + bb := by
    rw [hPi]
    simp
  have hnm2 : pt.S.ports[i].1 = if bb = 0 then "acc0" else "acc1" := by
    rw [hnm]
    unfold RecVStar.portNames
    rw [hA]
    exact names_acc _ _ hb
  have hmem : pt.S.ports[i].1 ∈ pt.S.ports.map (·.1) := List.mem_map.2 ⟨pt.S.ports[i], List.getElem_mem .., rfl⟩
  have hnout : pt.S.ports[i].1 ≠ "out" := by
    rw [hnm2]
    split_ifs <;> decide
  rw [hrow hnout, hnsi]
  -- the row it reads is its rep's end, its owner's
  obtain ⟨-, -, hpa⟩ := htab 0 ht
  obtain ⟨u, hu, -, -, -, hrows⟩ := FlockSoundness.Discharge.Recursion.checkParsed_ok hpa
  rw [hc0] at hrows
  obtain ⟨-, hfind, -, -⟩ := port_at hu hrows hi'
  have hR := repOpens_eq D hv hst1 hm1
  have hl : ws.length = r.w.parts.length := by
    have h1 : ws.length = nS x := hJ.1
    rw [h1]
    unfold nS
    rw [hr]
  have hW : ∀ (nmb : String) (row : ℕ), (nmb, r0) ∈ accHoldings r s row → pt.S.ports[i].1 = nmb →
      elemsOf (readForms wt (((rowHidden cS pt.S.ports[i].1 r0).getD []).map HmNets.Lin.var)) 2 o =
        (chainW x ws 0 row)[o]! := fun nmb row hh' hnb => by
    rw [hnb] at hfind ⊢
    have e1 := readSegs_accSegs wt cS (nmb, r0)
    dsimp only at e1
    rw [← e1]
    exact held_chainW elem hJ.2.1 hJ.2.2.1 hr hl hpc hh' hw hfind
      (fun o' co ho' hcan wo hwo => hC x st hx hf c ws hJ xs hxs hcol hms r hr s cS hpc _ nm pt hsvO row (nmb, r0) hh'
        o' co ho' hcan wt wo hw hwo i hfind) 0 ho2
  rw [hkn, ← _root_.Flock.RecAlgebra.getElem!_el]
  have hn22 : r0 / 2 = 0 := by omega
  have hn2m : r0 % 2 = r0 := Nat.mod_eq_of_lt hr0
  simp only [Rows.DecAt.accE, hn22, hn2m, ← hR]
  rcases (show bb = 0 ∨ bb = 1 by omega) with rfl | rfl
  · rw [hW "acc0" (r0 * repOpens r) (acc0_held hsvO hr0 (by rw [hnm2] at hmem; simpa using hmem))
      (by simpa using hnm2)]
    rcases (show o = 0 ∨ o = 1 by omega) with rfl | rfl
    · exact (lit4_get _ _ _ _).1.symm
    · exact (lit4_get _ _ _ _).2.1.symm
  · rw [hW "acc1" (r0 * repOpens r + repOpens r) (acc1_held hsvO hr0 (by rw [hnm2] at hmem; simpa using hmem))
      (by simpa using hnm2)]
    rcases (show o = 0 ∨ o = 1 by omega) with rfl | rfl
    · exact (lit4_get _ _ _ _).2.2.1.symm
    · exact (lit4_get _ _ _ _).2.2.2.symm

/-- **P3's ends half** wherever the brings list the registered reads (`RegSetup`, which holds at KS's brings). -/
theorem endsW_of_reg (rd : Msg → Option FinalP) (wE : Msg → ℝ)
    (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) (b : Brings)
    (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (untr : TrReader)
    (cmt : Msg → ℕ → Msg → Msg) (hRS : RegSetup b) :
    EndsW rd wE final w b elem Claims.Recursion.Form.Dense untr cmt :=
  endsW_of rd wE final w b elem untr cmt (innerAccTieV rd wE final w b elem untr cmt hRS)

/-- **P3 from its steps half**, wherever the brings list the registered reads. -/
theorem accPorts_of_steps (rd : Msg → Option FinalP) (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ) (wE : Msg → ℝ)
    (final : Msg → Option (List (List (Msg × Salt)) × Msg))
    (w : Msg → ℝ) (b : Brings) (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    (untr : TrReader) (cmt : Msg → ℕ → Msg → Msg)
    (hS : StepsWX rd key wE final w b elem Claims.Recursion.Form.Dense untr cmt) (hRS : RegSetup b) :
    AccPorts rd key wE final w b elem Claims.Recursion.Form.Dense untr cmt :=
  accPorts_of_WX rd key wE final w b elem untr cmt hS (endsW_of_reg rd wE final w b elem untr cmt hRS)

/-- `StepsWX` is `StepsWG` at the side condition `XsW … AccV`, curried. -/
theorem stepsWX_of_G (rd : Msg → Option FinalP) (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ) (wE : Msg → ℝ)
    (final : Msg → Option (List (List (Msg × Salt)) × Msg))
    (w : Msg → ℝ) (b : Brings) (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    (shape : Claims.Recursion.Form → Prop) (untr : TrReader) (cmt : Msg → ℕ → Msg → Msg)
    (h : StepsWG rd key wE final w b elem shape untr cmt
      (XsW rd wE final w b elem (AccV rd wE final w b elem shape untr cmt))) :
    StepsWX rd key wE final w b elem shape untr cmt :=
  fun x st hx hf c ws hJ xs hxs hcol hms => h x st hx hf c ws hJ ⟨xs, hxs, hcol, hms⟩

/-- **P3 from `StepsWG` at `XsW`** (STEPSW's form), wherever the brings list the registered reads. -/
theorem accPorts_of_G (rd : Msg → Option FinalP) (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ) (wE : Msg → ℝ)
    (final : Msg → Option (List (List (Msg × Salt)) × Msg))
    (w : Msg → ℝ) (b : Brings) (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    (untr : TrReader) (cmt : Msg → ℕ → Msg → Msg)
    (hS : StepsWG rd key wE final w b elem Claims.Recursion.Form.Dense untr cmt
      (XsW rd wE final w b elem (AccV rd wE final w b elem Claims.Recursion.Form.Dense untr cmt)))
    (hRS : RegSetup b) :
    AccPorts rd key wE final w b elem Claims.Recursion.Form.Dense untr cmt :=
  accPorts_of_steps rd key wE final w b elem untr cmt
    (stepsWX_of_G rd key wE final w b elem _ untr cmt hS) hRS

end RecDeploy.ResLower
