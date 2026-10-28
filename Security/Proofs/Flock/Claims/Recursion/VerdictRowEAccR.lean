import Proofs.Flock.Claims.Recursion.VerdictRowEAcc
import Proofs.Flock.Claims.Recursion.CapLower
import Proofs.Flock.Claims.Recursion.PartsVStar

/-!
# The `rec-acc` chain at the layout of record (`DeployedR.layR`)

`DeployedR.layR elem` records the chain's hm96 calls (`accCalls`, in `moreCalls`), so the chain's facts at
`layoutBAcc` that read only its calls carry over:

* `layoutBAcc_sub_layR`, `layoutBAcc_allCalls_sub_layR`: every call `layoutBAcc rest` records, at any `rest`, `layR elem`
  records, at any `elem`.
* `accPort_deployed`: every holding of the chain names a port of its part's circuit, where `readX` reads the statement
  (`partsVStar`'s parse check, `checkParsed_ok`: the circuit's row ports are V\*'s unit's, and a holding's port, `acc`
  at a level or `acc0`/`acc1` at an algebra part, is among the unit's inputs). It is `accCommit_layoutBAcc`'s `hq`.
* `accHolds_recorded_layR`: `accHolds_layoutBAcc_recorded` with its calls among `layR`'s and `hq` discharged.
* `held_rows_eq_layR`, `out_rows_eq_layR`: two holdings of the chain, or a unit's `acc_out` row and a holding, whose
  slots carry one commit string `b ‖ c` hold one row, wherever `layR`'s calls are true with no collision among them
  (`CapLower.rows_eq_of_calls`: equal `c` halves give equal salts `y`, so equal masks `M·y`, so equal digests
  `d = b ⊕ M·y`). This is the commits' step under hm96's salted digest: a climb to the registered `rec-acc` root starts
  at the commit string, not at `d`.
-/

namespace RecDeploy

open Flock
open Claims (Msg)
open Claims.Recursion.VStarB

/-! ## `layoutBAcc`'s calls are `layR`'s -/

section Sub

variable (rest : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
  (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))

theorem layoutBAcc_sub_layR (x : Msg) (s : ℕ) (w : Claims.Flock.Wit) :
    ∀ e ∈ (layoutBAcc rest).calls x s w, e ∈ (DeployedR.layR elem).calls x s w := fun e he => by
  rw [layoutBAcc_calls] at he
  rcases List.mem_append.1 he with h | h
  · exact DeployedR.layoutBTops_sub_layR elem (finAcc rest) x s w e h
  · obtain ⟨p, hp, rfl⟩ := List.mem_map.1 h
    cases hr : readX x with
    | none => rw [hr] at hp; simp at hp
    | some r =>
      rw [hr] at hp
      exact DeployedR.accCalls_sub_layR elem hr s w p hp

theorem layoutBAcc_allCalls_sub_layR (x : Msg) (ws : List Claims.Flock.Wit) {e : Msg × Msg}
    (h : e ∈ (layoutBAcc rest).allCalls x ws) : e ∈ (DeployedR.layR elem).allCalls x ws := by
  unfold LayoutB.allCalls at h ⊢
  obtain ⟨l, hl, he⟩ := List.mem_flatten.1 h
  obtain ⟨i, hi, rfl⟩ := List.mem_iff_getElem.1 hl
  have hi' : i < ws.length := by simpa using hi
  refine List.mem_flatten.2 ⟨(DeployedR.layR elem).calls x i ws[i], List.mem_iff_getElem.2 ⟨i, by simpa using hi',
    by simp⟩, ?_⟩
  exact layoutBAcc_sub_layR rest elem x i ws[i] e (by simpa using he)

end Sub

/-! ## Every holding's port is its circuit's -/

/-- **V\*'s unit's inputs are ports of a parsed circuit**: where the circuit's row ports are V\*'s (`checkParsed_ok`),
each input the unit has a word count for names one of them. -/
theorem findIdx_isSome_of_rows {c : Circuit} {sub : VStar.Sub} {u : VStar.UnitC}
    (hr : c.ports.toList.map (fun p => (p.name, p.words, p.bits)) = RecVStar.vstarRows sub u) {nm : String}
    (hnm : nm ∈ sub.inputs) (hlen : sub.inputs.length ≤ u.inPorts.length) :
    (c.ports.findIdx? (·.name == nm)).isSome = true := by
  have hz : nm ∈ (sub.inputs.zip u.inPorts).map Prod.fst := by
    rw [List.map_fst_zip] <;> assumption
  obtain ⟨⟨a, k⟩, hm, rfl⟩ := List.mem_map.1 hz
  show (c.ports.findIdx? (·.name == a)).isSome = true
  have hv : (a, k, 16 * k) ∈ RecVStar.vstarRows sub u := by
    unfold RecVStar.vstarRows
    exact List.mem_append_left _ (List.mem_map.2 ⟨(a, k), hm, rfl⟩)
  rw [← hr] at hv
  obtain ⟨pt, hpt, he⟩ := List.mem_map.1 hv
  have hname : pt.name = a := congrArg Prod.fst he
  cases hf : c.ports.findIdx? (·.name == a) with
  | some _ => rfl
  | none =>
    have hn := Array.findIdx?_eq_none_iff.1 hf pt (Array.mem_toList_iff.1 hpt)
    simp [hname] at hn

/-- **Every holding of the chain names a port of its part's circuit** (`accCommit_layoutBAcc`'s `hq`), where `readX`
reads the statement: a level's `acc` is the second of `recOpenSub`'s inputs past `row` and `aux`, an algebra part's
`acc0`/`acc1` one of its `RepS`'s ports, `innerSub`'s inputs, and the parse check holds the circuit's row ports to the
unit's (`checkParsed_ok`), whose inputs have a word count each (`recOpenUnit_ok`, `innerUnit_ok`). -/
theorem accPort_deployed {x : Msg} {r : XR} (hr : readX x = some r) {s : ℕ} {c : Circuit}
    (hc : r.w.parts[s]?.bind partCircuit = some c) {h : String × ℕ} (hh : h ∈ accHeld r s) :
    (c.ports.findIdx? (·.name == h.1)).isSome = true := by
  obtain ⟨vs, hvs, -, hall⟩ := readX_partVStar hr
  have hst0 : (RecVStar.statements r.shape).toOption.bind (·[s]?) = vs[s]? := by rw [hvs]; try rfl
  unfold accHeld at hh
  rw [hst0] at hh
  cases hst : vs[s]? with
  | none => rw [hst] at hh; simp at hh
  | some st =>
    rw [hst] at hh
    dsimp only at hh
    obtain ⟨-, hread⟩ := List.mem_filter.1 hh
    obtain ⟨hs, hst'⟩ := Array.getElem?_eq_some_iff.1 hst
    obtain ⟨-, p, c', -, hp, hpc, -, hpar, -⟩ := hall s hs
    have hcc : c' = c := by
      rw [hp, Option.bind_some, hpc] at hc
      exact Option.some_inj.1 hc
    subst hcc
    obtain ⟨u, hu, -, -, -, hports⟩ := FlockSoundness.Discharge.Recursion.checkParsed_ok hpar
    rw [hst'] at hu hports
    cases st with
    | recOpen lv L H n =>
      have hacc : h.1 = "acc" := by
        simp only [accRead] at hread
        split_ifs at hread with h1
        all_goals first | exact h1 | exact absurd hread Bool.false_ne_true
      obtain ⟨-, -, hin, -⟩ := FlockVBridge.OfVStar.recOpenUnit_ok hu
      rw [hacc]
      refine findIdx_isSome_of_rows hports ?_ ?_
      · show "acc" ∈ ["row", "aux", "acc", "coef", "dirs"]
        exact List.mem_cons_of_mem _ (List.mem_cons_of_mem _ List.mem_cons_self)
      · rw [hin]
        show ["row", "aux", "acc", "coef", "dirs"].length ≤ (VStar.recOpenPorts L H).length
        simp [VStar.recOpenPorts]
    | recOpenStacked lv L H n =>
      have hacc : h.1 = "acc" := by
        simp only [accRead] at hread
        split_ifs at hread with h1
        all_goals first | exact h1 | exact absurd hread Bool.false_ne_true
      have hu' : (VStar.recOpenUnit L H >>= fun u0 =>
          pure { u0 with name := VStar.recOpenStackedName L H }) = Except.ok (ε := String) u := hu
      obtain ⟨u0, hu0, he⟩ := FlockC3.VStarList.eok_bind hu'
      obtain ⟨-, -, hin, -⟩ := FlockVBridge.OfVStar.recOpenUnit_ok hu0
      have hin' : u.inPorts = VStar.recOpenPorts L H := by
        cases he
        exact hin
      rw [hacc]
      refine findIdx_isSome_of_rows hports ?_ ?_
      · show "acc" ∈ ["row", "aux", "acc", "coef", "dirs"]
        exact List.mem_cons_of_mem _ (List.mem_cons_of_mem _ List.mem_cons_self)
      · rw [hin']
        show ["row", "aux", "acc", "coef", "dirs"].length ≤ (VStar.recOpenPorts L H).length
        simp [VStar.recOpenPorts]
    | inner j name pt =>
      have hap : h.1 ∈ RecVStar.accPorts (.inner j name pt) := by
        simp only [accRead] at hread
        split_ifs at hread with h1
        all_goals first | exact h1.2 | exact absurd hread Bool.false_ne_true
      have hap' : h.1 ∈ (pt.S.ports.map (·.1)).filter fun nm => nm == "acc0" || nm == "acc1" := hap
      have hmem : h.1 ∈ pt.S.ports.map (·.1) := (List.mem_filter.1 hap').1
      obtain ⟨-, -, hin, -⟩ := FlockVBridge.OfVStar.innerUnit_ok hu
      refine findIdx_isSome_of_rows hports (List.mem_append_left ["v"] hmem) ?_
      rw [hin]
      show (pt.S.ports.map (·.1) ++ ["v"]).length ≤ (FlockVBridge.OfVStar.repPorts pt.S).length
      simp [FlockVBridge.OfVStar.repPorts]
    | sparse sh hR =>
      simp only [accRead] at hread
      split_ifs at hread
      all_goals exact absurd hread Bool.false_ne_true
    | opening m c0 j S ports nv =>
      simp only [accRead] at hread
      split_ifs at hread
      all_goals exact absurd hread Bool.false_ne_true
    | cap l N =>
      simp only [accRead] at hread
      split_ifs at hread
      all_goals exact absurd hread Bool.false_ne_true

/-! ## At `layR`'s calls -/

section AtLayR

variable (rest : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
  (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))

/-- **A piece holds its running sum at `layoutBAcc`'s chain, off `layR`'s calls** (`accHolds_layoutBAcc_recorded`, with
`l` holding `layR`'s calls and every holding's port its circuit's, `accPort_deployed`). -/
theorem accHolds_recorded_layR {H : Msg → Msg} {l : List (Msg × Msg)} (hT : ∀ e ∈ l, H e.1 = e.2)
    (hnc : (Claims.findColl l).isSome = false) {x : Msg} {ws : List Claims.Flock.Wit}
    (hl : ∀ e ∈ (DeployedR.layR elem).allCalls x ws, e ∈ l)
    (sideAt : Msg → List Msg → OpenPiece → OpenSide) (coefAt : Msg → List FlockDeploy.Coin → OpenPiece → Array F128)
    (q : OpenPiece) (cs : List FlockDeploy.Coin) {v : ℕ → Bool} {LANES HH : ℕ}
    (hO : Flock.Readable.LevelOuts v LANES HH)
    (hLA : LANES = 2 ^ ((sideAt x ((layoutBAcc rest).msgs x ws) q).sch.levels[q.level]!).lanes)
    (hrow : Flock.Readable.openRow v LANES =
      (FlockSoundness.Refine.openAt (sideAt x ((layoutBAcc rest).msgs x ws) q).lig
        ((sideAt x ((layoutBAcc rest).msgs x ws) q).sch.levels.size - 1) q.level).1[q.query]!)
    (hcf : Flock.Readable.openCoefs v LANES HH = coefAt x cs q)
    {r : XR} (hr : readX x = some r) {s : ℕ} {c : Circuit} (hc : r.w.parts[s]?.bind partCircuit = some c)
    {w : Claims.Flock.Wit} (hw : ws[s]? = some w) {u : ℕ} (hh : ("acc", u) ∈ accHeld r s)
    {qa qo : ℕ} (hqa : c.ports.findIdx? (·.name == "acc") = some qa)
    (hqo : c.ports.findIdx? (·.name == "acc_out") = some qo)
    {bi bo : ByteArray} (hbi : FlockVBridge.ColsHold v (128 * LANES + 1024 * max 1 ((HH + 1) / 2)) bi)
    (hszi : 32 ≤ bi.size) (hbo : FlockVBridge.ColsHold v (FlockVBridge.Family.outCol LANES HH + 1024) bo)
    (hszo : 32 ≤ bo.size)
    (hbi' : bi.data.toList = readSegs w (accSegs c ("acc", u)))
    (hbo' : bo.data.toList = readForms w (accOutForms c qo u))
    {pin : ℕ} (hpin : pin = Rows.chainPos (sideAt x ((layoutBAcc rest).msgs x ws) q).sch q.rep q.level q.query)
    {i₀ off₀ o₀ : ℕ} {w₀ : Claims.Flock.Wit} {v₀ : Msg}
    (hin₀ : accIn rest x (Nat.pair pin 0) = some (i₀, off₀))
    (ho₀ : (layoutBAcc rest).rowOwner x (Nat.pair pin 0) = some o₀) (hw₀ : ws[o₀]? = some w₀)
    (hv₀ : (layoutBAcc rest).rowHeld x o₀ w₀ (Nat.pair pin 0) = some v₀) {ri : Msg}
    {fsi gs₀ : List (Msg → Msg)} (hapi : Claims.Recursion.VStar.StepsApart fsi gs₀)
    (hcli : Claims.Recursion.VStar.chainH H fsi (readForms w (hmDigest c qa u)) = ri)
    (hcl₀ : Claims.Recursion.VStar.chainH H gs₀ (accDig x ws (Nat.pair pin 0) o₀) = ri)
    (hini : ∀ u' ∈ Claims.Recursion.VStar.chainIns H fsi (readForms w (hmDigest c qa u)), u' ∈ l.map Prod.fst)
    (hin0 : ∀ u' ∈ Claims.Recursion.VStar.chainIns H gs₀ (accDig x ws (Nat.pair pin 0) o₀), u' ∈ l.map Prod.fst)
    {i₁ off₁ o₁ : ℕ} {w₁ : Claims.Flock.Wit} {v₁ : Msg}
    (hin₁ : accIn rest x (Nat.pair (pin + 1) 0) = some (i₁, off₁))
    (ho₁ : (layoutBAcc rest).rowOwner x (Nat.pair (pin + 1) 0) = some o₁) (hw₁ : ws[o₁]? = some w₁)
    (hv₁ : (layoutBAcc rest).rowHeld x o₁ w₁ (Nat.pair (pin + 1) 0) = some v₁) {ro : Msg}
    {fso gs₁ : List (Msg → Msg)} (hapo : Claims.Recursion.VStar.StepsApart fso gs₁)
    (hclo : Claims.Recursion.VStar.chainH H fso (readForms w (hmDigest c qo u)) = ro)
    (hcl₁ : Claims.Recursion.VStar.chainH H gs₁ (accDig x ws (Nat.pair (pin + 1) 0) o₁) = ro)
    (hino : ∀ u' ∈ Claims.Recursion.VStar.chainIns H fso (readForms w (hmDigest c qo u)), u' ∈ l.map Prod.fst)
    (hin1 : ∀ u' ∈ Claims.Recursion.VStar.chainIns H gs₁ (accDig x ws (Nat.pair (pin + 1) 0) o₁),
      u' ∈ l.map Prod.fst) :
    Rows.accHolds sideAt coefAt (chainAt rest) q x ((layoutBAcc rest).msgs x ws) cs :=
  accHolds_layoutBAcc_recorded rest hT hnc (fun e he => hl e (layoutBAcc_allCalls_sub_layR rest elem x ws he))
    (fun _ _ _ _ hr' hc' hh' => accPort_deployed hr' hc' hh') sideAt coefAt q cs hO hLA hrow hcf hr hc hw hh hqa hqo
    hbi hszi hbo hszo hbi' hbo' hpin hin₀ ho₀ hw₀ hv₀ hapi hcli hcl₀ hini hin0 hin₁ ho₁ hw₁ hv₁ hapo hclo hcl₁ hino
    hin1

/-- A holding's port's calls are among its part's calls at `layR`. -/
theorem accCall_layR {x : Msg} {r : XR} (hr : readX x = some r) {s : ℕ} {c : Circuit}
    (hc : r.w.parts[s]?.bind partCircuit = some c) {ws : List Claims.Flock.Wit} {w : Claims.Flock.Wit}
    (hw : ws[s]? = some w) {h : String × ℕ} (hh : h ∈ accHeld r s) {p : Msg × List HmNets.Lin × List HmNets.Lin}
    (hp : p ∈ accPortCalls c h) :
    (p.1 ++ readForms w p.2.1, readForms w p.2.2) ∈ (DeployedR.layR elem).allCalls x ws :=
  CapLower.mem_allCalls hw (DeployedR.accCalls_sub_layR elem hr s w p (mem_accCalls hc hh hp))

theorem hmCalls_sub_accPortCalls {c : Circuit} {h : String × ℕ} {q : ℕ}
    (hq : c.ports.findIdx? (·.name == h.1) = some q) {p : Msg × List HmNets.Lin × List HmNets.Lin}
    (hp : p ∈ hmCalls c h.1 q h.2) : p ∈ accPortCalls c h := by
  unfold accPortCalls
  rw [hq]
  exact List.mem_append_left _ hp

theorem accOutCalls_sub_accPortCalls {c : Circuit} {u qo : ℕ} (hqo : c.ports.findIdx? (·.name == "acc_out") = some qo)
    {p : Msg × List HmNets.Lin × List HmNets.Lin} (hp : p ∈ accOutCalls c qo u) : p ∈ accPortCalls c ("acc", u) := by
  unfold accPortCalls
  rw [if_pos (rfl : ("acc", u).1 = "acc"), hqo]
  exact List.mem_append_right _ hp

/-- A holding's slot's salt and row calls, at `layR`. -/
theorem heldCalls_layR {x : Msg} {r : XR} (hr : readX x = some r) {s : ℕ} {c : Circuit}
    (hc : r.w.parts[s]?.bind partCircuit = some c) {ws : List Claims.Flock.Wit} {w : Claims.Flock.Wit}
    (hw : ws[s]? = some w) {h : String × ℕ} (hh : h ∈ accHeld r s) {q : ℕ}
    (hq : c.ports.findIdx? (·.name == h.1) = some q) :
    (Hm96.Default512.saltPrefix.data.toList ++ readForms w (hmSalt c q h.2),
        readForms w ((hmCommit c q h.2).drop 512)) ∈ (DeployedR.layR elem).allCalls x ws ∧
      (rowPfx c q ++ readSegs w (accSegs c h), readForms w (hmDigest c q h.2)) ∈
        (DeployedR.layR elem).allCalls x ws := by
  have hL : ∀ p ∈ hmCalls c h.1 q h.2,
      (p.1 ++ readForms w p.2.1, readForms w p.2.2) ∈ (DeployedR.layR elem).allCalls x ws := fun p hp =>
    accCall_layR elem hr hc hw hh (hmCalls_sub_accPortCalls hq hp)
  refine ⟨?_, ?_⟩
  · have e := hL (Hm96.Default512.saltPrefix.data.toList, hmSalt c q h.2, (hmCommit c q h.2).drop 512)
      (by unfold hmCalls; exact List.mem_cons_self)
    dsimp only at e ⊢
    exact e
  · have e := hL ((c.ports[q]?.map fun p => (HmRow.rowPrefix p.words).data.toList).getD [],
      ((rowHidden c h.1 h.2).getD []).map HmNets.Lin.var, hmDigest c q h.2)
      (by unfold hmCalls; exact List.mem_cons_of_mem _ List.mem_cons_self)
    rw [readSegs_accSegs]
    unfold rowPfx
    dsimp only at e ⊢
    exact e

/-- A level unit's `acc_out` slot's salt and row calls, at `layR`. -/
theorem outCalls_layR {x : Msg} {r : XR} (hr : readX x = some r) {s : ℕ} {c : Circuit}
    (hc : r.w.parts[s]?.bind partCircuit = some c) {ws : List Claims.Flock.Wit} {w : Claims.Flock.Wit}
    (hw : ws[s]? = some w) {u : ℕ} (hh : ("acc", u) ∈ accHeld r s) {qo : ℕ}
    (hqo : c.ports.findIdx? (·.name == "acc_out") = some qo) :
    (Hm96.Default512.saltPrefix.data.toList ++ readForms w (hmSalt c qo u),
        readForms w ((hmCommit c qo u).drop 512)) ∈ (DeployedR.layR elem).allCalls x ws ∧
      (rowPfx c qo ++ readForms w (accOutForms c qo u), readForms w (hmDigest c qo u)) ∈
        (DeployedR.layR elem).allCalls x ws := by
  have hL : ∀ p ∈ accOutCalls c qo u,
      (p.1 ++ readForms w p.2.1, readForms w p.2.2) ∈ (DeployedR.layR elem).allCalls x ws := fun p hp =>
    accCall_layR elem hr hc hw hh (accOutCalls_sub_accPortCalls hqo hp)
  refine ⟨?_, ?_⟩
  · have e := hL (Hm96.Default512.saltPrefix.data.toList, hmSalt c qo u, (hmCommit c qo u).drop 512)
      (by unfold accOutCalls; exact List.mem_cons_self)
    dsimp only at e ⊢
    exact e
  · have e := hL ((c.ports[qo]?.map fun p => (HmRow.rowPrefix p.words).data.toList).getD [], accOutForms c qo u,
      hmDigest c qo u) (by unfold accOutCalls; exact List.mem_cons_of_mem _ List.mem_cons_self)
    unfold rowPfx
    dsimp only at e ⊢
    exact e

/-- **Two holdings of the chain with one commit string hold one row**, at `layR` and the joint witnesses: where every
part's recorded calls are true and free of collisions, part `s`'s holding `h` (port `q`) and part `s'`'s holding `h'`
(port `q'`), of ports of one word count, whose slots carry one commit string `b ‖ c`, hold one row. -/
theorem held_rows_eq_layR {x : Msg} {ws : List Claims.Flock.Wit}
    (hT : ∀ e ∈ (DeployedR.layR elem).allCalls x ws, FlockDeploy.sha512 e.1 = e.2)
    (hnc : (Claims.findColl ((DeployedR.layR elem).allCalls x ws)).isSome = false) {r : XR} (hr : readX x = some r)
    {s s' : ℕ} {c c' : Circuit} (hc : r.w.parts[s]?.bind partCircuit = some c)
    (hc' : r.w.parts[s']?.bind partCircuit = some c') {w w' : Claims.Flock.Wit} (hw : ws[s]? = some w)
    (hw' : ws[s']? = some w') {h h' : String × ℕ} (hh : h ∈ accHeld r s) (hh' : h' ∈ accHeld r s') {q q' : ℕ}
    (hq : c.ports.findIdx? (·.name == h.1) = some q) (hq' : c'.ports.findIdx? (·.name == h'.1) = some q')
    (hpfx : rowPfx c q = rowPfx c' q')
    (hcm : readForms w (hmCommit c q h.2) = readForms w' (hmCommit c' q' h'.2)) :
    readSegs w (accSegs c h) = readSegs w' (accSegs c' h') := by
  obtain ⟨h1, h2⟩ := heldCalls_layR elem hr hc hw hh hq
  obtain ⟨h3, h4⟩ := heldCalls_layR elem hr hc' hw' hh' hq'
  rw [readSegs_accSegs] at h2 ⊢
  rw [readSegs_accSegs] at h4 ⊢
  exact CapLower.rows_eq_of_calls hT hnc h1 h2 h3 h4 hpfx hcm

/-- **A unit's `acc_out` row is the chain's next row's holding**, at `layR` and the joint witnesses: where every part's
recorded calls are true and free of collisions, part `s`'s unit `u` (`acc` held, `acc_out` port `qo`) and part `s'`'s
holding `h'` (port `q'`), of ports of one word count, whose slots carry one commit string, hold one row. -/
theorem out_rows_eq_layR {x : Msg} {ws : List Claims.Flock.Wit}
    (hT : ∀ e ∈ (DeployedR.layR elem).allCalls x ws, FlockDeploy.sha512 e.1 = e.2)
    (hnc : (Claims.findColl ((DeployedR.layR elem).allCalls x ws)).isSome = false) {r : XR} (hr : readX x = some r)
    {s s' : ℕ} {c c' : Circuit} (hc : r.w.parts[s]?.bind partCircuit = some c)
    (hc' : r.w.parts[s']?.bind partCircuit = some c') {w w' : Claims.Flock.Wit} (hw : ws[s]? = some w)
    (hw' : ws[s']? = some w') {u : ℕ} (hh : ("acc", u) ∈ accHeld r s) {h' : String × ℕ} (hh' : h' ∈ accHeld r s')
    {qo q' : ℕ} (hqo : c.ports.findIdx? (·.name == "acc_out") = some qo)
    (hq' : c'.ports.findIdx? (·.name == h'.1) = some q') (hpfx : rowPfx c qo = rowPfx c' q')
    (hcm : readForms w (hmCommit c qo u) = readForms w' (hmCommit c' q' h'.2)) :
    readForms w (accOutForms c qo u) = readSegs w' (accSegs c' h') := by
  obtain ⟨h1, h2⟩ := outCalls_layR elem hr hc hw hh hqo
  obtain ⟨h3, h4⟩ := heldCalls_layR elem hr hc' hw' hh' hq'
  rw [readSegs_accSegs] at h4 ⊢
  exact CapLower.rows_eq_of_calls hT hnc h1 h2 h3 h4 hpfx hcm

end AtLayR

end RecDeploy
