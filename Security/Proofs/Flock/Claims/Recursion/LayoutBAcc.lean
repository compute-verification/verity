import Proofs.Flock.Claims.Recursion.FinOpen
import Proofs.Flock.Claims.Recursion.FinalPiece
import Proofs.Flock.Claims.Recursion.BoundAccepts
import Proofs.Flock.Claims.Recursion.RowsCover
import Proofs.Flock.Claims.Recursion.SchedDense

/-!
# V*'s bound layout with the `rec-acc` chain's rows as slots (`layoutBAcc`), and where the messages carry them

V\* commits one `rec-acc` chain (`rec_outer.Chain`): rows `0 … 2n` for both reps' `n` openings each, rep-major, then by
level and query, and R's openings after them. A level's unit reads its opening's row as `acc` (`RecVStar.accReads`,
which `RecVStar.checkAccReads` holds the reads file to), and V\*'s algebra parts read the reps' ends as `acc0` (rows `0`
and `n`) and `acc1` (rows `n` and `2n`) at instances `0` and `1` (`rec_vstage.stage_stacked`). `layoutBTops` places no
row of it, so nothing reads the chain off the messages. `layoutBAcc rest` is `layoutBTops (finAcc rest)` with the chain's
rows as `LayoutB`'s shared-row slots:

* **The holdings** (`accHoldings`): at part `s`, each port and instance at which its statement reads row `p` of the
  chain (`accRead`), each as the port's row at the instance (`rowHidden`, `accSegs`). A level's `acc_out` is an output
  and no holding here: its row `p + 1` is the next unit's `acc`.
* **The slots**: slot `Nat.pair p j` is row `p`'s `j`-th holding at a part. Row `p`'s owner (`accCanon`) is the first
  part holding it once, else the first holding it at all; the owner holds every slot of row `p` at its first holding.
* **Where the messages carry them** (`finAcc`, `accIn`): the bound last message, after `finOpen rest`'s pieces, carries
  each held row once, as its owner holds it (`accBlock`), and every slot of row `p` sits there.
* **The rows' hm96 calls** (`accCalls`): each holding's port's two calls at its instance (`hmCalls`), and at a level's
  `acc` its unit's `acc_out` output row's (`accOutCalls`, the row at its output bits, `accOutForms`), beside
  `layoutBTops`' calls (`layoutBAcc_calls`). `rowCall_held` and `outCall_held` place a holding's row call and its
  unit's `acc_out` row call among its part's calls.

`rowsLaid_layoutBAcc` proves `RowsLaid` with no hypothesis: the readers' messages carry each slot's owner's value where
`accIn` places it. `chainAt rest` reads row `p` there as two elements, and `chainAt_held` proves it reads the row's
owner's value. `heldB_layoutBAcc_dense` is `HeldB` (the extra pieces are parts', `finPieces_finAcc`), and
`readsRows_layoutBAcc` is `ReadsRows` from `RowsAgree` (`readsRows_of_agree`), which `rowsAgree_of_commits` gives off the
rows' commits and the verifier's evaluations (`layoutBAcc_owner` is its owner premise).

The last message is the bound record's, read by `final` and the reader `rd`; the chain block follows `finOpen`'s pieces,
so a `final` that reads those leaves it as a suffix of the element-level message.
-/

namespace RecDeploy

open Flock
open Claims (Msg)
open FlockDeploy (Coin)
open Claims.Recursion.VStarB

/-! ## The chain's holdings -/

/-- A rep's openings at the inner shape's schedule (`RecVStar.opensBelow` at every level), `n`. -/
def repOpens (r : XR) : ℕ :=
  ((fast100 r.shape.m).map fun sch => RecVStar.opensBelow sch sch.levels.size).getD 0

/-- R's openings, which follow both reps' in the chain. -/
def stackedOpens (r : XR) : ℕ :=
  ((r.shape.stacked.bind fun i => fast100 i.m).map fun sch => RecVStar.opensBelow sch sch.levels.size).getD 0

/-- The chain's last row: `2n`, then R's openings. -/
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

/-- **Part `s`'s holdings of row `p`**: each port and instance at which its statement reads it, in order. -/
def accHoldings (r : XR) (s p : ℕ) : List (String × ℕ) :=
  match (RecVStar.statements r.shape).toOption.bind (·[s]?) with
  | some st => (accCands r st).filter fun h => accRead r st h.1 h.2 == some p
  | none => []

/-- A holding as segments: the port's row at the instance (`rowHidden`). -/
def accSegs (c : Circuit) (h : String × ℕ) : List (Msg × List HmNets.Lin) :=
  [(([] : Msg), ((rowHidden c h.1 h.2).getD []).map HmNets.Lin.var)]

/-- Part `s` with its circuit and first holding of row `p`, where it has both. -/
def accHolder (r : XR) (p s : ℕ) : Option (ℕ × Circuit × (String × ℕ)) :=
  (r.w.parts[s]?.bind partCircuit).bind fun c => (accHoldings r s p).head?.map fun h => (s, c, h)

/-- **Row `p`'s owner**: the first part holding it once, else the first holding it at all, with its circuit and first
holding. -/
def accCanon (r : XR) (p : ℕ) : Option (ℕ × Circuit × (String × ℕ)) :=
  ((List.range r.w.parts.length).findSome? fun s =>
    if (accHoldings r s p).length = 1 then accHolder r p s else none).or
  ((List.range r.w.parts.length).findSome? (accHolder r p))

/-- **Slot `σ` at part `s`**: slot `Nat.pair p j` is row `p`'s `j`-th holding there; the owner holds every slot of row
`p` at its first holding. -/
def accRowAt (r : XR) (s σ : ℕ) : Option (List (Msg × List HmNets.Lin)) :=
  match accCanon r σ.unpair.1 with
  | some (o, c, h) =>
    if s = o then some (accSegs c h)
    else (r.w.parts[s]?.bind partCircuit).bind fun c' => ((accHoldings r s σ.unpair.1)[σ.unpair.2]?).map (accSegs c')
  | none => none

/-- The circuit and holding at which part `s` holds slot `σ` (`accRowAt`'s). -/
def accHoldAt (r : XR) (s σ : ℕ) : Option (Circuit × (String × ℕ)) :=
  match accCanon r σ.unpair.1 with
  | some (o, c, h) =>
    if s = o then some (c, h)
    else (r.w.parts[s]?.bind partCircuit).bind fun c' => ((accHoldings r s σ.unpair.1)[σ.unpair.2]?).map (c', ·)
  | none => none

/-- **The chain block**: each held row up to the chain's last, as its owner holds it, with the owner's part. -/
def accBlock (r : XR) : List (ℕ × ℕ × List (Msg × List HmNets.Lin)) :=
  (List.range (chainTop r + 1)).filterMap fun p => (accCanon r p).map fun e => (p, e.1, accSegs e.2.1 e.2.2)

/-! ## The chain's rows' hm96 calls -/

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

/-- **The last round's pieces with the chain**: `finOpen rest`'s, then the chain block's. -/
def finAcc (x : Msg) : List (ℕ × List (Msg × List HmNets.Lin)) :=
  finOpen rest x ++ ((readX x).map fun r => (accBlock r).map fun e => e.2).getD []

/-- Row `p`'s byte offset in the last message: past `finOpen rest`'s pieces and the chain block's rows before it. -/
def accOff (x : Msg) (r : XR) (p : ℕ) : Option ℕ :=
  ((accBlock r).findIdx? (·.1 == p)).map fun k =>
    ((finOpen rest x).map fun q => pieceLen q.2).sum + (((accBlock r).take k).map fun e => pieceLen e.2.2).sum

/-- **Where the messages carry slot `σ`**: the last round, at its row's offset (`accOff`), where the statement doesn't
fix the last round. -/
def accIn (x : Msg) (σ : ℕ) : Option (ℕ × ℕ) :=
  if (layoutBTops (finAcc rest)).fixedAt x (layoutOf.rounds x - 1) = none ∧ 0 < layoutOf.rounds x then
    (readX x).bind fun r => (accOff rest x r σ.unpair.1).map fun o => (layoutOf.rounds x - 1, o)
  else none

/-- **V\*'s bound layout with the `rec-acc` chain's rows as slots**: `layoutBTops (finAcc rest)`, with the chain's
holdings as slots (`accRowAt`), each row's owner (`accCanon`), where the last message carries them (`accIn`), and the
hm96 calls on the chain's rows (`accCalls`) beside `layoutBTops`' calls. -/
def layoutBAcc : LayoutB :=
  { layoutBTops (finAcc rest) with
    shaAt := fun x s => (layoutBTops (finAcc rest)).shaAt x s ++ ((readX x).map fun r => accCalls r s).getD []
    rowAt := fun x s σ => (readX x).bind fun r => accRowAt r s σ
    rowOwner := fun x σ => (readX x).bind fun r => (accCanon r σ.unpair.1).map (·.1)
    rowIn := accIn rest }

/-- **Row `p` of the chain as the messages `ms` carry it**: its two elements at slot `Nat.pair p 0`'s place (`accIn`),
zero where it has none. V\* commits one chain, so the table index is unread. -/
def chainAt (x : Msg) (ms : List Msg) (_i p : ℕ) : Array _root_.Flock.F128 :=
  match accIn rest x (Nat.pair p 0) with
  | some (i, o) => #[elemsOf ((ms.getD i []).drop o) 2 0, elemsOf ((ms.getD i []).drop o) 2 1]
  | none => #[0, 0]

end Layout

/-! ## The owner and the block -/

theorem accHolder_spec {r : XR} {p s : ℕ} {e : ℕ × Circuit × (String × ℕ)} (h : accHolder r p s = some e) :
    e.1 = s ∧ r.w.parts[s]?.bind partCircuit = some e.2.1 ∧ (accHoldings r s p).head? = some e.2.2 := by
  unfold accHolder at h
  rw [Option.bind_eq_some_iff] at h
  obtain ⟨c, hc, h⟩ := h
  rw [Option.map_eq_some_iff] at h
  obtain ⟨hh, hh', rfl⟩ := h
  exact ⟨rfl, hc, hh'⟩

/-- **Row `p`'s owner is a part holding it**, with its circuit and first holding. -/
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

/-- **A part holding row `p` gives it an owner.** -/
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

/-- **Every held row up to the chain's last is in the block.** -/
theorem accBlock_findIdx {r : XR} {p : ℕ} (hp : p ≤ chainTop r) (hc : (accCanon r p).isSome) :
    ((accBlock r).findIdx? (·.1 == p)).isSome := by
  obtain ⟨e, he⟩ := Option.isSome_iff_exists.1 hc
  rw [List.findIdx?_isSome, List.any_eq_true]
  refine ⟨(p, e.1, accSegs e.2.1 e.2.2), ?_, by simp⟩
  unfold accBlock
  exact List.mem_filterMap.2 ⟨p, List.mem_range.2 (by omega), by rw [he]; rfl⟩

/-! ## The last message's bytes -/

/-- A piece of an assembled round off the witnesses, as `LayoutB.assembled` reads it. -/
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

/-- **`FinPieces` at `finAcc`**, at every shape: `finOpen`'s (`finPieces_finOpen`), and each chain row at its owner,
one of the list's parts (`accCanon_spec`). -/
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

/-- **`HeldB` at `layoutBAcc`**: `heldB_layoutBTops` at `finAcc rest`, whose pieces are parts' (`finPieces_finAcc`).
The slots change no round. -/
theorem heldB_layoutBAcc {W : Type} (IV : Claims.Verifier Coin) (ivScope : Msg → Bool) (b : Brings)
    (I : Claims.Recursion.Inner Coin W) (hk : ∀ x, I.V.k x = (FlockDeploy.V false).k x)
    (shape : Claims.Recursion.Form → Prop) (hM : SchedMatches shape) (hP : CapPorts shape) (hR : RowsCover shape) :
    Claims.Recursion.VStarB.HeldB (deployRec IV ivScope b) I (layoutBAcc rest) shape :=
  heldB_layoutBTops IV ivScope b I hk (finAcc rest) shape hM hP hR (finPieces_finAcc rest shape)

/-- **`HeldB` at `layoutBAcc` at the dense statements**, with no premise on the layout: the schedule matches
(`schedMatches_dense`), RecCap's port and the rows' ports are named (`capPorts`, `rowsCover`). -/
theorem heldB_layoutBAcc_dense {W : Type} (IV : Claims.Verifier Coin) (ivScope : Msg → Bool) (b : Brings)
    (I : Claims.Recursion.Inner Coin W) (hk : ∀ x, I.V.k x = (FlockDeploy.V false).k x) :
    Claims.Recursion.VStarB.HeldB (deployRec IV ivScope b) I (layoutBAcc rest) Claims.Recursion.Form.Dense :=
  heldB_layoutBAcc rest IV ivScope b I hk _ schedMatches_dense (capPorts _) (rowsCover _)

/-- The last round's message off the witnesses is `finAcc rest`'s pieces, read off their parts. -/
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

/-- **Slot `σ`'s owner's value**, where the owner is part `o` with circuit `c` and first holding `h`: what `o`'s witness
reads at `accSegs c h`, empty where `o` has no witness. -/
theorem row_layoutBAcc {x : Msg} {r : XR} (hr : readX x = some r) {σ o : ℕ}
    {c : Circuit} {h : String × ℕ} (hcan : accCanon r σ.unpair.1 = some (o, c, h)) :
    (layoutBAcc rest).rowOwner x σ = some o ∧
      ∀ w, (layoutBAcc rest).rowHeld x o w σ = some (readSegs w (accSegs c h)) := by
  refine ⟨by simp [layoutBAcc, hr, hcan], fun w => ?_⟩
  simp [LayoutB.rowHeld, layoutBAcc, hr, accRowAt, hcan]

/-- **The readers' messages carry every slot's owner's value where `accIn` places it** (`RowsLaid`), with no hypothesis:
the last round is `finAcc rest`'s pieces read off their parts (`msg_last`), each piece the length its segments fix
(`pieceRead_length`), so row `p`'s place in the chain block is its owner's reading of its first holding. -/
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

/-- **The parts read the slots where the readers' messages carry them** (`ReadsRows`), wherever the parts agree on the
slots: `readsRows_of_agree` at `rowsLaid_layoutBAcc`. -/
theorem readsRows_layoutBAcc {H : Msg → Msg}
    (ha : ∀ x ws, (∀ e ∈ (layoutBAcc rest).allCalls x ws, H e.1 = e.2) → (layoutBAcc rest).RowsAgree x ws) :
    (layoutBAcc rest).ReadsRows H :=
  LayoutB.readsRows_of_agree (rowsLaid_layoutBAcc rest) ha

/-- **Every held slot's owner holds it** (`rowsAgree_of_commits`' owner premise), where every part has a witness. -/
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

/-- **Part `s` holds its `j`-th holding of row `p` at slot `Nat.pair p j`**, unless it is the row's owner and `j > 0`:
the owner holds every slot of its row at its first holding. -/
theorem rowAt_layoutBAcc {x : Msg} {r : XR} (hr : readX x = some r) {s p j : ℕ} {c : Circuit} {h : String × ℕ}
    (hs : s < r.w.parts.length) (hc : r.w.parts[s]?.bind partCircuit = some c)
    (hj : (accHoldings r s p)[j]? = some h) (hj0 : (accCanon r p).map (·.1) = some s → j = 0) :
    (layoutBAcc rest).rowAt x s (Nat.pair p j) = some (accSegs c h) := by
  have hh : accHolder r p s = some (s, c, (accHoldings r s p).head!) := by
    have hne : accHoldings r s p ≠ [] := fun h0 => by simp [h0] at hj
    obtain ⟨a, l, hal⟩ := List.exists_cons_of_ne_nil hne
    simp [accHolder, hc, hal]
  obtain ⟨e, he⟩ := Option.isSome_iff_exists.1 (accCanon_isSome hs hh)
  show ((readX x).bind fun r => accRowAt r s (Nat.pair p j)) = _
  rw [hr, Option.bind_some]
  unfold accRowAt
  simp only [Nat.unpair_pair]
  obtain ⟨o, c', h'⟩ := e
  rw [he]
  dsimp only
  split_ifs with hso
  · subst hso
    have h0 := hj0 (by rw [he]; rfl)
    subst h0
    have hc' : r.w.parts[s]?.bind partCircuit = some c' := (accCanon_spec he).2.1
    have hh' : (accHoldings r s p).head? = some h' := (accCanon_spec he).2.2
    rw [hc] at hc'
    cases Option.some.inj hc'
    rw [List.head?_eq_getElem?, hj] at hh'
    cases Option.some.inj hh'
    rfl
  · simp only [hc, Option.bind_some, hj, Option.map_some]

/-- **Slot `σ` is placed wherever its row has an owner, up to the chain's last**, and the statement doesn't fix the last
round. -/
theorem accIn_isSome {x : Msg} {r : XR} (hr : readX x = some r) {σ : ℕ}
    (hfx : (layoutBTops (finAcc rest)).fixedAt x (layoutOf.rounds x - 1) = none) (hpos : 0 < layoutOf.rounds x)
    (hp : σ.unpair.1 ≤ chainTop r) (hc : (accCanon r σ.unpair.1).isSome) : (accIn rest x σ).isSome := by
  obtain ⟨k, hk⟩ := Option.isSome_iff_exists.1 (accBlock_findIdx hp hc)
  have hc' : (layoutBTops (finAcc rest)).fixedAt x (layoutOf.rounds x - 1) = none ∧ 0 < layoutOf.rounds x :=
    ⟨hfx, hpos⟩
  unfold accIn accOff
  rw [ite_eq_left hc', hr, Option.bind_some, hk]
  rfl

/-- `chainAt` at a placed slot: the two elements of round `i`'s bytes from offset `off`. -/
theorem chainAt_of_accIn {x : Msg} {ms : List Msg} {p i off : ℕ} (hin : accIn rest x (Nat.pair p 0) = some (i, off))
    (t : ℕ) : chainAt rest x ms t p = #[elemsOf ((ms.getD i []).drop off) 2 0, elemsOf ((ms.getD i []).drop off) 2 1] := by
  unfold chainAt
  rw [hin]

/-- Bytes starting with `v`, of at least two elements, carry `v`'s first two. -/
theorem elems_of_take {L v : Msg} (h : L.take v.length = v) (hl : 32 ≤ v.length) :
    #[elemsOf L 2 0, elemsOf L 2 1] = #[elemsOf v 2 0, elemsOf v 2 1] := by
  have hl' : L.take (([] : Msg) ++ v).length = [] ++ v := by
    rw [List.nil_append]
    exact h
  have e0 := elemsOf_drop_of_take hl' (n := 2) (by omega) 0
  have e1 := elemsOf_drop_of_take hl' (n := 2) (by omega) 1
  simp only [List.length_nil, List.drop_zero] at e0 e1
  rw [e0, e1]

/-- **`chainAt` reads row `p`'s owner's value**: at the readers' messages, where slot `Nat.pair p 0` is placed and its
owner `o` holds `v`, of at least two elements, `chainAt` reads `v`'s first two elements (`rowsLaid_layoutBAcc`). -/
theorem chainAt_held {x : Msg} {ws : List Claims.Flock.Wit} {p i off o : ℕ} {w : Claims.Flock.Wit} {v : Msg}
    (hin : accIn rest x (Nat.pair p 0) = some (i, off)) (ho : (layoutBAcc rest).rowOwner x (Nat.pair p 0) = some o)
    (hw : ws[o]? = some w) (hv : (layoutBAcc rest).rowHeld x o w (Nat.pair p 0) = some v) (hl : 32 ≤ v.length)
    (t : ℕ) : chainAt rest x ((layoutBAcc rest).msgs x ws) t p = #[elemsOf v 2 0, elemsOf v 2 1] := by
  have hlaid := rowsLaid_layoutBAcc rest x ws (Nat.pair p 0) i off hin
  rw [LayoutB.row_of_owner ho hw hv] at hlaid
  rw [chainAt_of_accIn rest hin t]
  exact elems_of_take hlaid hl

/-! ## The chain's calls -/

/-- **`layoutBAcc`'s calls**: `layoutBTops`', then the chain's rows' (`accCalls`). -/
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

/-- A holding of a row is a holding of the chain. -/
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

/-- **What a part holds at a slot is one of its holdings of the slot's row**, at its circuit. -/
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

/-- A held row's bytes: its port's row read at its instance (`rowHidden`). -/
theorem readSegs_accSegs (w : Claims.Flock.Wit) (c : Circuit) (h : String × ℕ) :
    readSegs w (accSegs c h) = readForms w (((rowHidden c h.1 h.2).getD []).map HmNets.Lin.var) := by
  simp [readSegs, accSegs]

/-- **A holding's row call**: where part `s`, of circuit `c`, holds the chain at `h`, at its port `q` (`pt`), it records
the row's evaluation `SHA-512(rowPrefix words ‖ row) = d`, at the row it holds and its slot's digest (`hmDigest`). -/
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

/-- **A level unit's `acc_out` row call**: where part `s`, of circuit `c`, holds the chain at its unit `u`'s `acc`,
and the circuit has the output port `acc_out` at `q` (`pt`), it records that row's evaluation at the output bits
(`accOutForms`) and its slot's digest (`hmDigest`). -/
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
