module

public import Proofs.Flock.Claims.Recursion.FinOpen
public import Proofs.Flock.Claims.Recursion.FinalPiece
public import Proofs.Flock.Claims.Recursion.BoundAccepts
public import Proofs.Flock.Claims.Recursion.RowsCover
public import Proofs.Flock.Claims.Recursion.SchedDense

/-! V*'s bound layout with the `rec-acc` chain's rows as slots (`layoutBAcc`), at which `RowsLaid` holds with no hypothesis. -/

@[expose] public section

namespace RecDeploy

open Flock
open Claims (Msg)
open FlockDeploy (Coin)
open Claims.Recursion.VStarB

/-- A rep's openings at the inner shape's schedule (`RecVStar.opensBelow` at every level), `n`. -/
def repOpens (r : XR) : ℕ :=
  ((fast100 r.shape.m).map fun sch => RecVStar.opensBelow sch sch.levels.size).getD 0

def stackedOpens (r : XR) : ℕ :=
  ((r.shape.stacked.bind fun i => fast100 i.m).map fun sch => RecVStar.opensBelow sch sch.levels.size).getD 0

def chainTop (r : XR) : ℕ := 2 * repOpens r + stackedOpens r

/-- **The chain's row statement `st` reads at port `nm`, instance `u`**: a level's (R's included) `acc` at the row
`RecVStar.accReads` names; an algebra part's `acc0` at rep `u`'s start `u · n` and `acc1` at its end `u · n + n`, for
the two reps `u < 2`. None elsewhere. -/
def accRead (r : XR) (st : VStar.Stmt) (nm : String) (u : ℕ) : Option ℕ :=
  match st with
  | .inner .. =>
    if u < 2 ∧ nm ∈ RecVStar.accPorts st then
      if nm = "acc0" then some (u * repOpens r)
      else if nm = "acc1" then some (u * repOpens r + repOpens r) else none
    else none
  | _ =>
    if nm = "acc" then (((RecVStar.accReads r.shape st).toOption.bind id).bind fun A => A[u]?).bind id else none

/-- The ports and instances at which a statement may read the chain: `acc0` and `acc1` at both reps, and `acc` at each
of a level's units. -/
def accCands (r : XR) (st : VStar.Stmt) : List (String × ℕ) :=
  [("acc0", 0), ("acc0", 1), ("acc1", 0), ("acc1", 1)] ++
    (List.range ((((RecVStar.accReads r.shape st).toOption.bind id).map Array.size).getD 0)).map
      fun u => ("acc", u)

def accHoldings (r : XR) (s p : ℕ) : List (String × ℕ) :=
  match (RecVStar.statements r.shape).toOption.bind (·[s]?) with
  | some st => (accCands r st).filter fun h => accRead r st h.1 h.2 == some p
  | none => []

/-- A holding as segments: the port's row at the instance (`rowHidden`). -/
def accSegs (c : Circuit) (h : String × ℕ) : List (Msg × List HmNets.Lin) :=
  [(([] : Msg), ((rowHidden c h.1 h.2).getD []).map HmNets.Lin.var)]

def accHolder (r : XR) (p s : ℕ) : Option (ℕ × Circuit × (String × ℕ)) :=
  (r.w.parts[s]?.bind partCircuit).bind fun c => (accHoldings r s p).head?.map fun h => (s, c, h)

def accCanon (r : XR) (p : ℕ) : Option (ℕ × Circuit × (String × ℕ)) :=
  ((List.range r.w.parts.length).findSome? fun s =>
    if (accHoldings r s p).length = 1 then accHolder r p s else none).or
  ((List.range r.w.parts.length).findSome? (accHolder r p))

def accRowAt (r : XR) (s σ : ℕ) : Option (List (Msg × List HmNets.Lin)) :=
  match accCanon r σ.unpair.1 with
  | some (o, c, h) =>
    if s = o then some (accSegs c h)
    else (r.w.parts[s]?.bind partCircuit).bind fun c' => ((accHoldings r s σ.unpair.1)[σ.unpair.2]?).map (accSegs c')
  | none => none

def accHoldAt (r : XR) (s σ : ℕ) : Option (Circuit × (String × ℕ)) :=
  match accCanon r σ.unpair.1 with
  | some (o, c, h) =>
    if s = o then some (c, h)
    else (r.w.parts[s]?.bind partCircuit).bind fun c' => ((accHoldings r s σ.unpair.1)[σ.unpair.2]?).map (c', ·)
  | none => none

def accBlock (r : XR) : List (ℕ × ℕ × List (Msg × List HmNets.Lin)) :=
  (List.range (chainTop r + 1)).filterMap fun p => (accCanon r p).map fun e => (p, e.1, accSegs e.2.1 e.2.2)

/-- **Part `s`'s holdings of the chain, at any row**: each port and instance at which its statement reads a row of it. -/
def accHeld (r : XR) (s : ℕ) : List (String × ℕ) :=
  match (RecVStar.statements r.shape).toOption.bind (·[s]?) with
  | some st => (accCands r st).filter fun h => (accRead r st h.1 h.2).isSome
  | none => []

/-- **Output port `q`'s row at instance `n`**, at forms: word `k` bit `t` (row bit `16 k + t`) the output bit
`HmOut.outCol` places it at in the instance's block and VU, where `k` is within the port's run of output words
(`HmOut.outGroup`), and zero past it (`HmOut.delta`'s forced zero). -/
def accOutForms (c : Circuit) (q n : ℕ) : List HmNets.Lin :=
  (List.range (16 * (c.ports.getD q default).words)).map fun i =>
    if i / 16 < (HmOut.outGroup c (q - HmOut.nIn c)).2.2.2 then
      HmNets.Lin.var (n / c.g * 2 ^ c.kLog +
        (HmOut.outCol c (n % c.g) ((HmOut.outGroup c (q - HmOut.nIn c)).2.2.1 + i / 16) + i % 16))
    else ⟨[], false⟩

/-- **Output port `q`'s hm96 calls at instance `n`**: the salt's, `SHA-512(p_salt ‖ y) = c`, then the row's at its
output bits (`accOutForms`), `SHA-512(rowIn row) = d`. -/
def accOutCalls (c : Circuit) (q n : ℕ) : List (Msg × List HmNets.Lin × List HmNets.Lin) :=
  [(Hm96.Default512.saltPrefix.data.toList, hmSalt c q n, (hmCommit c q n).drop 512),
    ((c.ports[q]?.map fun p => (HmRow.rowPrefix p.words).data.toList).getD [], accOutForms c q n, hmDigest c q n)]

/-- **A holding's hm96 calls**: its port's at its instance (`hmCalls`), and at a level's `acc` its unit's `acc_out`
output row's (`accOutCalls`), the chain's next row (`RecVStar.accReads`). None at a port the circuit lacks. -/
def accPortCalls (c : Circuit) (h : String × ℕ) : List (Msg × List HmNets.Lin × List HmNets.Lin) :=
  (match c.ports.findIdx? (·.name == h.1) with
    | some q => hmCalls c h.1 q h.2
    | none => []) ++
  (if h.1 = "acc" then
    match c.ports.findIdx? (·.name == "acc_out") with
    | some q => accOutCalls c q h.2
    | none => []
  else [])

/-- **Part `s`'s calls on the chain's rows**: each holding's (`accPortCalls`). -/
def accCalls (r : XR) (s : ℕ) : List (Msg × List HmNets.Lin × List HmNets.Lin) :=
  ((r.w.parts[s]?.bind partCircuit).map fun c => (accHeld r s).flatMap (accPortCalls c)).getD []

section Layout

variable (rest : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))

def finAcc (x : Msg) : List (ℕ × List (Msg × List HmNets.Lin)) :=
  finOpen rest x ++ ((readX x).map fun r => (accBlock r).map fun e => e.2).getD []

def accOff (x : Msg) (r : XR) (p : ℕ) : Option ℕ :=
  ((accBlock r).findIdx? (·.1 == p)).map fun k =>
    ((finOpen rest x).map fun q => pieceLen q.2).sum + (((accBlock r).take k).map fun e => pieceLen e.2.2).sum

def accIn (x : Msg) (σ : ℕ) : Option (ℕ × ℕ) :=
  if (layoutBTops (finAcc rest)).fixedAt x (layoutOf.rounds x - 1) = none ∧ 0 < layoutOf.rounds x then
    (readX x).bind fun r => (accOff rest x r σ.unpair.1).map fun o => (layoutOf.rounds x - 1, o)
  else none

def layoutBAcc : LayoutB :=
  { layoutBTops (finAcc rest) with
    shaAt := fun x s => (layoutBTops (finAcc rest)).shaAt x s ++ ((readX x).map fun r => accCalls r s).getD []
    rowAt := fun x s σ => (readX x).bind fun r => accRowAt r s σ
    rowOwner := fun x σ => (readX x).bind fun r => (accCanon r σ.unpair.1).map (·.1)
    rowIn := accIn rest }

/-- V\* commits one chain, so the table index is unread. -/
def chainAt (x : Msg) (ms : List Msg) (_i p : ℕ) : Array _root_.Flock.F128 :=
  match accIn rest x (Nat.pair p 0) with
  | some (i, o) => #[elemsOf ((ms.getD i []).drop o) 2 0, elemsOf ((ms.getD i []).drop o) 2 1]
  | none => #[0, 0]

end Layout

theorem accHolder_spec {r : XR} {p s : ℕ} {e : ℕ × Circuit × (String × ℕ)} (h : accHolder r p s = some e) :
    e.1 = s ∧ r.w.parts[s]?.bind partCircuit = some e.2.1 ∧ (accHoldings r s p).head? = some e.2.2 := by
  unfold accHolder at h
  rw [Option.bind_eq_some_iff] at h
  obtain ⟨c, hc, h⟩ := h
  rw [Option.map_eq_some_iff] at h
  obtain ⟨hh, hh', rfl⟩ := h
  exact ⟨rfl, hc, hh'⟩

theorem accCanon_spec {r : XR} {p : ℕ} {e : ℕ × Circuit × (String × ℕ)} (h : accCanon r p = some e) :
    e.1 < r.w.parts.length ∧ r.w.parts[e.1]?.bind partCircuit = some e.2.1 ∧
      (accHoldings r e.1 p).head? = some e.2.2 := by
  have key : ∀ s ∈ List.range r.w.parts.length, accHolder r p s = some e →
      e.1 < r.w.parts.length ∧ r.w.parts[e.1]?.bind partCircuit = some e.2.1 ∧
        (accHoldings r e.1 p).head? = some e.2.2 := fun s hs he => by
    obtain ⟨h1, hc, hh⟩ := accHolder_spec he
    subst h1
    exact ⟨List.mem_range.1 hs, hc, hh⟩
  unfold accCanon at h
  rcases h1 : (List.range r.w.parts.length).findSome? (fun s =>
      if (accHoldings r s p).length = 1 then accHolder r p s else none) with _ | e'
  · simp only [h1, Option.or] at h
    obtain ⟨s, hs, hf⟩ := List.exists_of_findSome?_eq_some h
    exact key s hs hf
  · simp only [h1, Option.or, Option.some.injEq] at h
    subst h
    obtain ⟨s, hs, hf⟩ := List.exists_of_findSome?_eq_some h1
    split at hf
    · exact key s hs hf
    · cases hf

theorem accCanon_isSome {r : XR} {p s : ℕ} {e : ℕ × Circuit × (String × ℕ)} (hs : s < r.w.parts.length)
    (h : accHolder r p s = some e) : (accCanon r p).isSome := by
  unfold accCanon
  rcases h1 : (List.range r.w.parts.length).findSome? (fun s =>
      if (accHoldings r s p).length = 1 then accHolder r p s else none) with _ | e'
  · simp only [Option.or]
    rcases h2 : (List.range r.w.parts.length).findSome? (accHolder r p) with _ | e''
    · exact absurd h (by rw [List.findSome?_eq_none_iff.1 h2 s (List.mem_range.2 hs)]; simp)
    · rfl
  · rfl

theorem mem_accBlock {r : XR} {e : ℕ × ℕ × List (Msg × List HmNets.Lin)} (he : e ∈ accBlock r) :
    ∃ c h, accCanon r e.1 = some (e.2.1, c, h) ∧ e.2.2 = accSegs c h := by
  unfold accBlock at he
  obtain ⟨p, -, hp⟩ := List.mem_filterMap.1 he
  rw [Option.map_eq_some_iff] at hp
  obtain ⟨a, ha, rfl⟩ := hp
  exact ⟨a.2.1, a.2.2, ha, rfl⟩

def pieceRead (ws : List Claims.Flock.Wit) (q : ℕ × List (Msg × List HmNets.Lin)) : Msg :=
  (ws[q.1]?.map fun w => readSegs w q.2).getD (List.replicate (pieceLen q.2) 0)

theorem pieceRead_length (ws : List Claims.Flock.Wit) (q : ℕ × List (Msg × List HmNets.Lin)) :
    (pieceRead ws q).length = pieceLen q.2 := by
  unfold pieceRead
  cases ws[q.1]? <;> simp [readSegs_length]

theorem flatMap_pieceRead_length (ws : List Claims.Flock.Wit) (L : List (ℕ × List (Msg × List HmNets.Lin))) :
    (L.flatMap (pieceRead ws)).length = (L.map fun q => pieceLen q.2).sum := by
  induction L with
  | nil => rfl
  | cons q L ih => rw [List.flatMap_cons, List.length_append, pieceRead_length, ih, List.map_cons, List.sum_cons]

theorem flatMap_split_at {β γ : Type} (B : List β) (f : β → List γ) {k : ℕ} (hk : k < B.length) :
    B.flatMap f = (B.take k).flatMap f ++ (f B[k] ++ (B.drop (k + 1)).flatMap f) := by
  have hB : B = B.take k ++ B[k] :: B.drop (k + 1) := by
    rw [← List.drop_eq_getElem_cons hk, List.take_append_drop]
  have e := congrArg (List.flatMap f) hB
  rw [List.flatMap_append, List.flatMap_cons] at e
  exact e

section Facts

variable (rest : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))

theorem finAcc_of_readX {x : Msg} {r : XR} (hr : readX x = some r) :
    finAcc rest x = finOpen rest x ++ (accBlock r).map fun e => e.2 := by
  simp only [finAcc, hr, Option.map_some, Option.getD_some]

theorem finPieces_finAcc (shape : Claims.Recursion.Form → Prop) : FinPieces (finAcc rest) shape := by
  intro x st hst hsh
  obtain ⟨hne, hlt⟩ := finPieces_finOpen rest shape x st hst hsh
  obtain ⟨r, hr, -⟩ := readX_of_parse hst
  have hnS : nS x = r.w.parts.length := by simp [nS, hr]
  rw [finAcc_of_readX rest hr]
  refine ⟨fun h => hne (List.append_eq_nil_iff.1 h).1, fun p hp => ?_⟩
  rcases List.mem_append.1 hp with hp | hp
  · exact hlt p hp
  · obtain ⟨e, he, rfl⟩ := List.mem_map.1 hp
    obtain ⟨c, h, hcan, -⟩ := mem_accBlock he
    rw [hnS]
    exact (accCanon_spec hcan).1

theorem heldB_layoutBAcc {W : Type} (IV : Claims.Verifier Coin) (ivScope : Msg → Bool) (b : Brings)
    (I : Claims.Recursion.Inner Coin W) (hk : ∀ x, I.V.k x = (FlockDeploy.V false).k x)
    (shape : Claims.Recursion.Form → Prop) (hM : SchedMatches shape) (hP : CapPorts shape) (hR : RowsCover shape) :
    Claims.Recursion.VStarB.HeldB (deployRec IV ivScope b) I (layoutBAcc rest) shape :=
  heldB_layoutBTops IV ivScope b I hk (finAcc rest) shape hM hP hR (finPieces_finAcc rest shape)

theorem heldB_layoutBAcc_dense {W : Type} (IV : Claims.Verifier Coin) (ivScope : Msg → Bool) (b : Brings)
    (I : Claims.Recursion.Inner Coin W) (hk : ∀ x, I.V.k x = (FlockDeploy.V false).k x) :
    Claims.Recursion.VStarB.HeldB (deployRec IV ivScope b) I (layoutBAcc rest) Claims.Recursion.Form.Dense :=
  heldB_layoutBAcc rest IV ivScope b I hk _ schedMatches_dense (capPorts _) (rowsCover _)

theorem msg_last {x : Msg} {ws : List Claims.Flock.Wit}
    (hfx : (layoutBTops (finAcc rest)).fixedAt x (layoutOf.rounds x - 1) = none) (hpos : 0 < layoutOf.rounds x)
    (hne : finAcc rest x ≠ []) :
    ((layoutBAcc rest).msgs x ws).getD (layoutOf.rounds x - 1) [] = (finAcc rest x).flatMap (pieceRead ws) := by
  have hround : (layoutBAcc rest).rounds x = layoutOf.rounds x := rfl
  have hfx' : (layoutBAcc rest).fixedAt x (layoutOf.rounds x - 1) = none := hfx
  have hpc : (layoutBAcc rest).pieceAt x (layoutOf.rounds x - 1) = finAcc rest x := by
    show stepPieces (finAcc rest) x (layoutOf.rounds x - 1) = _
    unfold stepPieces
    rw [ite_eq_left (by omega)]
  unfold LayoutB.msgs
  rw [hround, List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range (by omega), Option.map_some,
    Option.getD_some]
  unfold LayoutB.msg LayoutB.assembled
  rw [hfx', hpc]
  dsimp only
  rw [ite_eq_right (by rw [List.isEmpty_iff]; exact hne)]
  rfl

theorem row_layoutBAcc {x : Msg} {r : XR} (hr : readX x = some r) {σ o : ℕ}
    {c : Circuit} {h : String × ℕ} (hcan : accCanon r σ.unpair.1 = some (o, c, h)) :
    (layoutBAcc rest).rowOwner x σ = some o ∧
      ∀ w, (layoutBAcc rest).rowHeld x o w σ = some (readSegs w (accSegs c h)) := by
  refine ⟨by simp [layoutBAcc, hr, hcan], fun w => ?_⟩
  simp [LayoutB.rowHeld, layoutBAcc, hr, accRowAt, hcan]

theorem rowsLaid_layoutBAcc : (layoutBAcc rest).RowsLaid := by
  intro x ws σ i off hin
  change accIn rest x σ = some (i, off) at hin
  unfold accIn at hin
  by_cases hc : (layoutBTops (finAcc rest)).fixedAt x (layoutOf.rounds x - 1) = none ∧ 0 < layoutOf.rounds x
  swap
  · rw [ite_eq_right hc] at hin
    cases hin
  rw [ite_eq_left hc] at hin
  obtain ⟨hfx, hpos⟩ := hc
  rw [Option.bind_eq_some_iff] at hin
  obtain ⟨r, hr, hin⟩ := hin
  rw [Option.map_eq_some_iff] at hin
  obtain ⟨o', ho', hio⟩ := hin
  simp only [Prod.mk.injEq] at hio
  obtain ⟨rfl, rfl⟩ := hio
  unfold accOff at ho'
  rw [Option.map_eq_some_iff] at ho'
  obtain ⟨k, hk, rfl⟩ := ho'
  obtain ⟨hkl, hkp, -⟩ := List.findIdx?_eq_some_iff_getElem.1 hk
  have hfin := finAcc_of_readX rest hr
  set B := accBlock r with hB
  have hp : B[k].1 = σ.unpair.1 := by simpa using hkp
  obtain ⟨c, h, hcan, hseg⟩ := mem_accBlock (List.getElem_mem hkl)
  rw [hp] at hcan
  have hne : finAcc rest x ≠ [] := by
    rw [hfin]
    intro h0
    rw [List.append_eq_nil_iff, List.map_eq_nil_iff] at h0
    rw [h0.2] at hkl
    simp at hkl
  have hk' : k < (B.map fun e => e.2).length := by simpa using hkl
  rw [msg_last rest hfx hpos hne, hfin, List.flatMap_append, flatMap_split_at (B.map fun e => e.2) (pieceRead ws) hk', ← List.append_assoc]
  have hlen : ((finOpen rest x).flatMap (pieceRead ws) ++ ((B.map fun e => e.2).take k).flatMap (pieceRead ws)).length =
      ((finOpen rest x).map fun q => pieceLen q.2).sum + ((B.take k).map fun e => pieceLen e.2.2).sum := by
    rw [List.length_append, flatMap_pieceRead_length, flatMap_pieceRead_length, ← List.map_take, List.map_map]
    rfl
  rw [List.drop_left' hlen]
  simp only [List.getElem_map]
  obtain ⟨hown, hheld⟩ := row_layoutBAcc rest hr hcan
  cases hw : ws[B[k].2.1]? with
  | none =>
    have h0 : (layoutBAcc rest).row x ws σ = [] := by simp [LayoutB.row, hown, hw]
    rw [h0]
    simp
  | some w =>
    rw [LayoutB.row_of_owner hown hw (hheld w)]
    have hpr : pieceRead ws B[k].2 = readSegs w (accSegs c h) := by
      simp [pieceRead, hw, hseg]
    rw [hpr, List.take_left' rfl]

theorem readsRows_layoutBAcc {H : Msg → Msg}
    (ha : ∀ x ws, (∀ e ∈ (layoutBAcc rest).allCalls x ws, H e.1 = e.2) → (layoutBAcc rest).RowsAgree x ws) :
    (layoutBAcc rest).ReadsRows H :=
  LayoutB.readsRows_of_agree (rowsLaid_layoutBAcc rest) ha

theorem layoutBAcc_owner {x : Msg} {ws : List Claims.Flock.Wit} (hws : ws.length = nS x) :
    ∀ σ s w v, ws[s]? = some w → (layoutBAcc rest).rowHeld x s w σ = some v →
      ∃ o w' v', (layoutBAcc rest).rowOwner x σ = some o ∧ ws[o]? = some w' ∧
        (layoutBAcc rest).rowHeld x o w' σ = some v' := by
  intro σ s w v _ hv
  simp only [LayoutB.rowHeld, Option.map_eq_some_iff] at hv
  obtain ⟨segs, hseg, -⟩ := hv
  change ((readX x).bind fun r => accRowAt r s σ) = some segs at hseg
  rw [Option.bind_eq_some_iff] at hseg
  obtain ⟨r, hr, hseg⟩ := hseg
  unfold accRowAt at hseg
  rcases hcan : accCanon r σ.unpair.1 with _ | ⟨o, c, h⟩
  · simp [hcan] at hseg
  have hlt : o < r.w.parts.length := (accCanon_spec hcan).1
  have hnS : nS x = r.w.parts.length := by simp [nS, hr]
  obtain ⟨hown, hheld⟩ := row_layoutBAcc rest hr hcan
  exact ⟨o, ws[o]'(by omega), _, hown, List.getElem?_eq_getElem (by omega), hheld _⟩

theorem chainAt_of_accIn {x : Msg} {ms : List Msg} {p i off : ℕ} (hin : accIn rest x (Nat.pair p 0) = some (i, off))
    (t : ℕ) : chainAt rest x ms t p = #[elemsOf ((ms.getD i []).drop off) 2 0, elemsOf ((ms.getD i []).drop off) 2 1] := by
  unfold chainAt
  rw [hin]

theorem elems_of_take {L v : Msg} (h : L.take v.length = v) (hl : 32 ≤ v.length) :
    #[elemsOf L 2 0, elemsOf L 2 1] = #[elemsOf v 2 0, elemsOf v 2 1] := by
  have hl' : L.take (([] : Msg) ++ v).length = [] ++ v := by
    rw [List.nil_append]
    exact h
  have e0 := elemsOf_drop_of_take hl' (n := 2) (by omega) 0
  have e1 := elemsOf_drop_of_take hl' (n := 2) (by omega) 1
  simp only [List.length_nil, List.drop_zero] at e0 e1
  rw [e0, e1]

theorem chainAt_held {x : Msg} {ws : List Claims.Flock.Wit} {p i off o : ℕ} {w : Claims.Flock.Wit} {v : Msg}
    (hin : accIn rest x (Nat.pair p 0) = some (i, off)) (ho : (layoutBAcc rest).rowOwner x (Nat.pair p 0) = some o)
    (hw : ws[o]? = some w) (hv : (layoutBAcc rest).rowHeld x o w (Nat.pair p 0) = some v) (hl : 32 ≤ v.length)
    (t : ℕ) : chainAt rest x ((layoutBAcc rest).msgs x ws) t p = #[elemsOf v 2 0, elemsOf v 2 1] := by
  have hlaid := rowsLaid_layoutBAcc rest x ws (Nat.pair p 0) i off hin
  rw [LayoutB.row_of_owner ho hw hv] at hlaid
  rw [chainAt_of_accIn rest hin t]
  exact elems_of_take hlaid hl

theorem layoutBAcc_calls (x : Msg) (s : ℕ) (w : Claims.Flock.Wit) :
    (layoutBAcc rest).calls x s w = (layoutBTops (finAcc rest)).calls x s w ++
      (((readX x).map fun r => accCalls r s).getD []).map fun p => (p.1 ++ readForms w p.2.1, readForms w p.2.2) := by
  unfold LayoutB.calls layoutBAcc
  exact List.map_append ..

theorem mem_calls_of_accCalls {x : Msg} {r : XR} (hr : readX x = some r) {s : ℕ} (w : Claims.Flock.Wit)
    {p : Msg × List HmNets.Lin × List HmNets.Lin} (hp : p ∈ accCalls r s) :
    (p.1 ++ readForms w p.2.1, readForms w p.2.2) ∈ (layoutBAcc rest).calls x s w := by
  rw [layoutBAcc_calls, hr]
  simp only [Option.map_some, Option.getD_some]
  exact List.mem_append_right _ (List.mem_map.2 ⟨p, hp, rfl⟩)

theorem mem_accCalls {r : XR} {s : ℕ} {c : Circuit} (hc : r.w.parts[s]?.bind partCircuit = some c)
    {h : String × ℕ} (hh : h ∈ accHeld r s) {p : Msg × List HmNets.Lin × List HmNets.Lin}
    (hp : p ∈ accPortCalls c h) : p ∈ accCalls r s := by
  unfold accCalls
  rw [hc]
  simp only [Option.map_some, Option.getD_some]
  exact List.mem_flatMap.2 ⟨h, hh, hp⟩

theorem mem_accHeld {r : XR} {s p : ℕ} {h : String × ℕ} (hh : h ∈ accHoldings r s p) : h ∈ accHeld r s := by
  unfold accHoldings at hh
  unfold accHeld
  rcases hst : (RecVStar.statements r.shape).toOption.bind (·[s]?) with _ | st
  · simp [hst] at hh
  · simp only [hst] at hh ⊢
    obtain ⟨hm, he⟩ := List.mem_filter.1 hh
    exact List.mem_filter.2 ⟨hm, by rw [beq_iff_eq.1 he]; rfl⟩

theorem accRowAt_eq (r : XR) (s σ : ℕ) : accRowAt r s σ = (accHoldAt r s σ).map fun e => accSegs e.1 e.2 := by
  unfold accRowAt accHoldAt
  cases accCanon r σ.unpair.1 with
  | none => rfl
  | some e =>
    obtain ⟨o, c, h⟩ := e
    dsimp only
    split_ifs
    · rfl
    · cases r.w.parts[s]?.bind partCircuit with
      | none => rfl
      | some c' => cases (accHoldings r s σ.unpair.1)[σ.unpair.2]? <;> rfl

theorem accHoldAt_spec {r : XR} {s σ : ℕ} {c : Circuit} {h : String × ℕ} (he : accHoldAt r s σ = some (c, h)) :
    r.w.parts[s]?.bind partCircuit = some c ∧ h ∈ accHoldings r s σ.unpair.1 := by
  unfold accHoldAt at he
  cases hcan : accCanon r σ.unpair.1 with
  | none => simp [hcan] at he
  | some e =>
    obtain ⟨o, c', h'⟩ := e
    rw [hcan] at he
    dsimp only at he
    split_ifs at he with hso
    · cases he
      subst hso
      obtain ⟨-, hc, hh⟩ := accCanon_spec hcan
      exact ⟨hc, List.mem_of_mem_head? (Option.mem_def.2 hh)⟩
    · obtain ⟨c'', hc'', he⟩ := Option.bind_eq_some_iff.1 he
      obtain ⟨h'', hj, he⟩ := Option.map_eq_some_iff.1 he
      cases he
      exact ⟨hc'', List.mem_of_getElem? hj⟩

theorem readSegs_accSegs (w : Claims.Flock.Wit) (c : Circuit) (h : String × ℕ) :
    readSegs w (accSegs c h) = readForms w (((rowHidden c h.1 h.2).getD []).map HmNets.Lin.var) := by
  simp [readSegs, accSegs]

theorem rowCall_held {x : Msg} {r : XR} (hr : readX x = some r) {s : ℕ} {c : Circuit}
    (hc : r.w.parts[s]?.bind partCircuit = some c) {h : String × ℕ} (hh : h ∈ accHeld r s) {q : ℕ}
    (hq : c.ports.findIdx? (·.name == h.1) = some q) {pt : Port} (hpt : c.ports[q]? = some pt)
    (w : Claims.Flock.Wit) :
    ((HmRow.rowPrefix pt.words).data.toList ++ readSegs w (accSegs c h), readForms w (hmDigest c q h.2)) ∈
      (layoutBAcc rest).calls x s w := by
  have hm : ((c.ports[q]?.map fun p => (HmRow.rowPrefix p.words).data.toList).getD [],
      ((rowHidden c h.1 h.2).getD []).map HmNets.Lin.var, hmDigest c q h.2) ∈ accPortCalls c h := by
    unfold accPortCalls
    rw [hq]
    exact List.mem_append_left _ (show _ ∈ hmCalls c h.1 q h.2 by simp [hmCalls])
  have e := mem_calls_of_accCalls rest hr w (mem_accCalls hc hh hm)
  simp only [hpt, Option.map_some, Option.getD_some] at e
  rw [readSegs_accSegs]
  exact e

theorem outCall_held {x : Msg} {r : XR} (hr : readX x = some r) {s : ℕ} {c : Circuit}
    (hc : r.w.parts[s]?.bind partCircuit = some c) {u : ℕ} (hh : ("acc", u) ∈ accHeld r s) {q : ℕ}
    (hq : c.ports.findIdx? (·.name == "acc_out") = some q) {pt : Port} (hpt : c.ports[q]? = some pt)
    (w : Claims.Flock.Wit) :
    ((HmRow.rowPrefix pt.words).data.toList ++ readForms w (accOutForms c q u), readForms w (hmDigest c q u)) ∈
      (layoutBAcc rest).calls x s w := by
  have hm : ((c.ports[q]?.map fun p => (HmRow.rowPrefix p.words).data.toList).getD [], accOutForms c q u,
      hmDigest c q u) ∈ accPortCalls c ("acc", u) := by
    unfold accPortCalls
    rw [if_pos (rfl : ("acc", u).1 = "acc"), hq]
    exact List.mem_append_right _ (show _ ∈ accOutCalls c q u by simp [accOutCalls])
  have e := mem_calls_of_accCalls rest hr w (mem_accCalls hc hh hm)
  simp only [hpt, Option.map_some, Option.getD_some] at e
  exact e

end Facts

end RecDeploy
