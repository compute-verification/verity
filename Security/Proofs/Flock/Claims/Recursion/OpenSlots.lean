import Proofs.Flock.Claims.Recursion.SaltRowsSlots
import Proofs.Flock.Claims.Recursion.CutOpens
import Proofs.Flock.Claims.Recursion.RestFinal

/-!
# The openings and commit strings of the rounds before the last are hm96 slots' (`OpensSlots` at `layR`)

At `DeployedR.layR elem` every commit string of a round before the last is some part's hm96 slot output, and the last
message opens each at that same slot: RecCap's `N` instances at round 0 (`capHold`, `capOpen`); at a stream round, each
top's instances at its cap's part (`topPiece`, `capOpenAt`), then the row the round absorbs at its first holder
(`rowPiece`, `rowOpen`). This file lays both sides out as one list of slots a round (`slotRounds`):

* `openLens_eq`, `openBytes`: the trailer's lengths and the openings' bytes are the slots' rows and salts, round by
  round, so `finalR` cuts each round's openings back (`CutOpens.cutRounds_groups`).
* `msgs_slots`: round `j`'s message is its slots' commit strings end to end.
* `opensSlots_layR`: `OpensSlots (layR elem)` from `SlotsRecorded elem`, the one fact left at the slots' forms: each
  slot's two hm96 calls are among its part's recorded calls, and its port's row prefix is the one `rowInDeployed` lays
  at the row's length.
-/

namespace RecDeploy.OpenSlots

open Flock
open Claims (Msg)
open Claims.Recursion.VStarB
open RecDeploy.DeployedR
open RecDeploy.RestRead

/-! ## Lists -/

theorem findSome?_map_out {α β γ : Type} (f : α → Option β) (g : β → γ) :
    ∀ l : List α, (l.findSome? f).map g = l.findSome? fun a => (f a).map g
  | [] => rfl
  | a :: l => by
    rw [List.findSome?_cons, List.findSome?_cons]
    cases f a with
    | some b => rfl
    | none => exact findSome?_map_out f g l

theorem findSome?_congr_mem {α β : Type} {f g : α → Option β} :
    ∀ l : List α, (∀ a ∈ l, f a = g a) → l.findSome? f = l.findSome? g
  | [], _ => rfl
  | a :: l, h => by
    rw [List.findSome?_cons, List.findSome?_cons, h a List.mem_cons_self,
      findSome?_congr_mem l fun b hb => h b (List.mem_cons_of_mem _ hb)]

theorem findSome?_eq_of_unique {α β : Type} {f : α → Option β} {k : α} {v : β} :
    ∀ l : List α, k ∈ l → f k = some v → (∀ a ∈ l, a ≠ k → f a = none) → l.findSome? f = some v
  | [], hk, _, _ => by simp at hk
  | a :: l, hk, hv, hn => by
    rw [List.findSome?_cons]
    by_cases hak : a = k
    · subst hak
      simp [hv]
    · rw [hn a List.mem_cons_self hak]
      exact findSome?_eq_of_unique l ((List.mem_cons.1 hk).resolve_left (Ne.symm hak)) hv
        fun b hb => hn b (List.mem_cons_of_mem _ hb)

theorem flatMap_congr_mem {α β : Type} {f g : α → List β} :
    ∀ l : List α, (∀ a ∈ l, f a = g a) → l.flatMap f = l.flatMap g
  | [], _ => rfl
  | a :: l, h => by
    rw [List.flatMap_cons, List.flatMap_cons, h a List.mem_cons_self,
      flatMap_congr_mem l fun b hb => h b (List.mem_cons_of_mem _ hb)]

/-! ## Slots -/

/-- **An hm96 slot**: at part `s`, of circuit `c`, port `q`'s slot at instance `n`, its row read at port `nm`. -/
structure Slot where
  s : ℕ
  c : Circuit
  nm : String
  q : ℕ
  n : ℕ

/-- The slot's row, at port `nm`'s unit input columns. -/
def Slot.row (σ : Slot) : List HmNets.Lin := ((rowHidden σ.c σ.nm σ.n).getD []).map HmNets.Lin.var

/-- The slot's opening segments: its row, then its salt. -/
def Slot.segs (σ : Slot) : List (Msg × List HmNets.Lin) := openSegs σ.c σ.nm σ.q σ.n

/-- The slot's commit string segment. -/
def Slot.cmt (σ : Slot) : Msg × List HmNets.Lin := (([] : Msg), hmCommit σ.c σ.q σ.n)

/-- The slot's opening off the witness `w`: its row's bytes, then its salt's. -/
def Slot.opening (w : Claims.Flock.Wit) (σ : Slot) : Msg × Msg :=
  (readForms w σ.row, readForms w (hmSalt σ.c σ.q σ.n))

/-- The slot's commit string off the witness `w`. -/
def Slot.commit (w : Claims.Flock.Wit) (σ : Slot) : Msg := readForms w (hmCommit σ.c σ.q σ.n)

/-- Part `s`'s circuit, the default where it has none (as `capOpen` and `topPiece` read it). -/
def cOf (r : XR) (s : ℕ) : Circuit := (r.w.parts[s]?.bind partCircuit).getD default

/-- A cap's slots at part `s`: port `top`'s at instances `ns`. -/
def capSlots (r : XR) (s : ℕ) (ns : List ℕ) : List Slot :=
  ns.map fun n => ⟨s, cOf r s, "top", (capPort (cOf r s)).getD 0, n⟩

/-- The slot of the row stream round `j` absorbs, where part `s` holds one. -/
def rowSlotAt (r : XR) (j s : ℕ) : Option Slot :=
  ((heldRows r s).find? (·.1 == j)).bind fun h =>
    (r.w.parts[s]?.bind partCircuit).map fun c => ⟨s, c, h.2.1, h.2.2.1, h.2.2.2⟩

/-- The slot of the row stream round `j` absorbs, at its first holder (`rowPiece`'s and `rowOpen`'s part). -/
def rowSlot (r : XR) (j : ℕ) : Option Slot := (List.range r.w.parts.length).findSome? (rowSlotAt r j)

/-- The `e`-th top of stream round `j`'s slots, at its cap's part (`capPieceAt`'s, `capOpenAt`'s). -/
def capSlotsAt (r : XR) (rs : Array RecVStar.SchedRound) (j e : ℕ) : List Slot :=
  capSlots r (capsFrom r rs + topsBefore rs (j % rs.size) + e)
    ((List.range (((rs.getD (j % rs.size) default).2.1).getD e 0)).map fun t =>
      j / rs.size * ((rs.getD (j % rs.size) default).2.1).getD e 0 + t)

/-- **Stream round `j`'s slots**, in commit order: each top's, then the row's. -/
def roundSlots (r : XR) (rs : Array RecVStar.SchedRound) (j : ℕ) : List Slot :=
  (List.range ((rs.getD (j % rs.size) default).2.1).size).flatMap (capSlotsAt r rs j) ++ (rowSlot r j).toList

/-- **Every round before the last's slots**, round by round: RecCap's at round 0, then each stream round's. -/
def slotRounds (r : XR) : List (List Slot) :=
  (capAt r).toList.map (fun p => capSlots r p.1 (List.range p.2)) ++
    ((repSched r.shape).map fun rs => (List.range (2 * rs.size)).map (roundSlots r rs)).getD []

/-- **Each slot's calls are recorded at its part, and its row prefix fits**: at a dense compiled statement, each slot's
two hm96 calls (`hmCalls`) are among its part's recorded calls at `layR elem`, and its port's row prefix (`rowPfx`) is
the one `rowInDeployed` lays at its row's byte length. -/
def SlotsRecorded (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) : Prop :=
  ∀ x, DenseX x → ∀ r, readX x = some r → ∀ l ∈ slotRounds r, ∀ σ ∈ l,
    (∀ p ∈ hmCalls σ.c σ.nm σ.q σ.n, p ∈ (DeployedR.layR elem).shaAt x σ.s) ∧
      rowPfx σ.c σ.q = (HmRow.rowPrefix (((σ.row.length + 7) / 8) / 2)).data.toList

/-! ## The pieces as slots -/

theorem capSlots_s {r : XR} {s : ℕ} {ns : List ℕ} {σ : Slot} (h : σ ∈ capSlots r s ns) : σ.s = s := by
  obtain ⟨n, -, rfl⟩ := List.mem_map.1 h
  rfl

theorem rowSlotAt_s {r : XR} {j s : ℕ} {σ : Slot} (h : rowSlotAt r j s = some σ) : σ.s = s := by
  unfold rowSlotAt at h
  obtain ⟨hh, -, h⟩ := Option.bind_eq_some_iff.1 h
  obtain ⟨c, -, rfl⟩ := Option.map_eq_some_iff.1 h
  rfl

theorem capOpen_eq (r : XR) (s : ℕ) (ns : List ℕ) : capOpen r s ns = (capSlots r s ns).flatMap Slot.segs := by
  unfold capOpen capSlots
  rw [List.flatMap_map]
  rfl

theorem topPiece_eq (r : XR) (s n rep : ℕ) :
    topPiece r s n rep = (capSlots r s ((List.range n).map fun t => rep * n + t)).map Slot.cmt := by
  unfold topPiece capSlots
  rw [List.map_map, List.map_map]
  rfl

theorem rowOpen_eq (r : XR) (j : ℕ) : rowOpen r j = (rowSlot r j).toList.map fun σ => (σ.s, σ.segs) := by
  unfold rowOpen rowSlot
  rw [← Option.toList_map, findSome?_map_out]
  refine congrArg Option.toList (congrArg (fun F => List.findSome? F (List.range r.w.parts.length))
    (funext fun s => ?_))
  unfold rowSlotAt
  cases ((heldRows r s).find? (·.1 == j)) <;> cases (r.w.parts[s]?.bind partCircuit) <;> rfl

theorem rowPiece_eq (r : XR) (j : ℕ) : rowPiece r j = (rowSlot r j).toList.map fun σ => (σ.s, [σ.cmt]) := by
  unfold rowPiece rowSlot
  rw [← Option.toList_map, findSome?_map_out]
  refine congrArg Option.toList (congrArg (fun F => List.findSome? F (List.range r.w.parts.length))
    (funext fun s => ?_))
  unfold rowSlotAt
  cases ((heldRows r s).find? (·.1 == j)) <;> cases (r.w.parts[s]?.bind partCircuit) <;> rfl

/-! ## The trailer's lengths -/

theorem rowLens_slots : ∀ σs : List Slot,
    rowLens (σs.flatMap Slot.segs) = σs.map fun σ => pieceLen [(([] : Msg), σ.row)]
  | [] => rfl
  | σ :: σs => by
    rw [List.flatMap_cons, List.map_cons, ← rowLens_slots σs]
    rfl

theorem roundLens_slots (r : XR) (rs : Array RecVStar.SchedRound) (j : ℕ) :
    (roundOpen r rs j).flatMap (fun p => rowLens p.2) =
      (roundSlots r rs j).map fun σ => pieceLen [(([] : Msg), σ.row)] := by
  unfold roundOpen roundSlots
  rw [List.flatMap_append, List.map_append, List.flatMap_map, List.map_flatMap, rowOpen_eq]
  congr 1
  · congr 1
    funext e
    show rowLens (capOpen r _ _) = _
    rw [capOpen_eq, rowLens_slots]
    try rfl
  · cases rowSlot r j <;> rfl

/-- **The trailer's lengths are the slots' rows' byte lengths**, round by round. -/
theorem openLens_eq (r : XR) :
    openLens r = (slotRounds r).map fun l => l.map fun σ => pieceLen [(([] : Msg), σ.row)] := by
  unfold openLens slotRounds
  rw [List.map_append, List.map_map]
  congr 1
  · apply List.map_congr_left
    intro p _
    simp only [Function.comp_apply]
    rw [capOpen_eq, rowLens_slots]
  · cases repSched r.shape with
    | none => rfl
    | some rs =>
      simp only [Option.map_some, Option.getD_some, List.map_map]
      apply List.map_congr_left
      intro j _
      simp only [Function.comp_apply]
      exact roundLens_slots r rs j

/-! ## The openings' bytes -/

theorem readSegs_flatMap {α : Type} (w : Claims.Flock.Wit) (l : List α) (f : α → List (Msg × List HmNets.Lin)) :
    readSegs w (l.flatMap f) = l.flatMap fun a => readSegs w (f a) := by
  unfold readSegs
  rw [List.flatMap_assoc]

theorem readSegs_segs (w : Claims.Flock.Wit) (σ : Slot) :
    readSegs w σ.segs = (σ.opening w).1 ++ (σ.opening w).2 := by
  simp only [readSegs, Slot.segs, openSegs, List.flatMap_cons, List.flatMap_nil, List.nil_append, List.append_nil,
    Slot.opening, Slot.row]

theorem readSegs_cmts (w : Claims.Flock.Wit) (σs : List Slot) :
    readSegs w (σs.map Slot.cmt) = σs.flatMap fun σ => σ.commit w := by
  unfold readSegs
  rw [List.flatMap_map]
  refine flatMap_congr_mem _ fun σ _ => ?_
  simp only [Slot.cmt, Slot.commit, List.nil_append]

theorem readSegs_cmt (w : Claims.Flock.Wit) (σ : Slot) : readSegs w [σ.cmt] = σ.commit w := by
  simp only [readSegs, Slot.cmt, Slot.commit, List.flatMap_cons, List.flatMap_nil, List.nil_append, List.append_nil]

theorem readPieces_eq (ps : List (ℕ × List (Msg × List HmNets.Lin))) (ws : List Claims.Flock.Wit)
    (d : Claims.Flock.Wit) (h : ∀ p ∈ ps, p.1 < ws.length) :
    readPieces ps ws = ps.flatMap fun p => readSegs (ws.getD p.1 d) p.2 := by
  unfold readPieces
  refine flatMap_congr_mem _ fun p hp => ?_
  simp only [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem (h p hp), Option.map_some, Option.getD_some]

theorem piece_slots (ws : List Claims.Flock.Wit) (d : Claims.Flock.Wit) (s : ℕ) (σs : List Slot)
    (h : ∀ σ ∈ σs, σ.s = s) :
    readSegs (ws.getD s d) (σs.flatMap Slot.segs) = σs.flatMap fun σ => readSegs (ws.getD σ.s d) σ.segs := by
  rw [readSegs_flatMap]
  exact flatMap_congr_mem _ fun σ hσ => by simp only [h σ hσ]

theorem roundBytes (r : XR) (rs : Array RecVStar.SchedRound) (j : ℕ) (ws : List Claims.Flock.Wit)
    (d : Claims.Flock.Wit) :
    (roundOpen r rs j).flatMap (fun p => readSegs (ws.getD p.1 d) p.2) =
      (roundSlots r rs j).flatMap fun σ => readSegs (ws.getD σ.s d) σ.segs := by
  unfold roundOpen roundSlots
  rw [List.flatMap_append, List.flatMap_append, List.flatMap_map, List.flatMap_assoc, rowOpen_eq]
  congr 1
  · congr 1
    funext e
    show readSegs (ws.getD _ d) (capOpen r _ _) = _
    rw [capOpen_eq]
    exact piece_slots ws d _ _ fun σ hσ => capSlots_s hσ
  · cases rowSlot r j <;> rfl

/-- **The openings' bytes are the slots' rows and salts**, round by round, where every piece is a witness's part. -/
theorem openBytes (r : XR) (ws : List Claims.Flock.Wit) (d : Claims.Flock.Wit)
    (h : ∀ p ∈ openPieces r, p.1 < ws.length) :
    readPieces (openPieces r) ws =
      (slotRounds r).flatMap (·.flatMap fun σ => readSegs (ws.getD σ.s d) σ.segs) := by
  rw [readPieces_eq _ ws d h]
  unfold openPieces slotRounds
  rw [List.flatMap_append, List.flatMap_append]
  congr 1
  · cases capAt r with
    | none => rfl
    | some p =>
      show readSegs (ws.getD p.1 d) (capOpen r p.1 (List.range p.2)) ++ [] =
        (capSlots r p.1 (List.range p.2)).flatMap (fun σ => readSegs (ws.getD σ.s d) σ.segs) ++ []
      rw [capOpen_eq, piece_slots ws d _ _ fun σ hσ => capSlots_s hσ]
  · cases repSched r.shape with
    | none => rfl
    | some rs =>
      show ((List.range (2 * rs.size)).flatMap (roundOpen r rs)).flatMap _ =
        ((List.range (2 * rs.size)).map (roundSlots r rs)).flatMap _
      rw [List.flatMap_assoc, List.flatMap_map]
      congr 1
      funext j
      exact roundBytes r rs j ws d

/-! ## Every slot and piece is a part's -/

theorem rowSlot_lt {r : XR} {j : ℕ} {σ : Slot} (h : rowSlot r j = some σ) : σ.s < r.w.parts.length := by
  obtain ⟨a, ha, hfa⟩ := List.exists_of_findSome?_eq_some h
  rw [rowSlotAt_s hfa]
  exact List.mem_range.1 ha

theorem capSlotsAt_lt {r : XR} {rs : Array RecVStar.SchedRound} {j e : ℕ} (hpos : 0 < rs.size)
    (hlen : (RecVStar.topSizes rs).length ≤ r.w.parts.length - 1)
    (he : e < ((rs.getD (j % rs.size) default).2.1).size) {σ : Slot} (h : σ ∈ capSlotsAt r rs j e) :
    σ.s < r.w.parts.length := by
  rw [capSlots_s h]
  have hlt := topsBefore_lt (Nat.mod_lt j hpos) he
  unfold capsFrom
  omega

theorem slotRounds_lt {x : Msg} {r : XR} (hr : readX x = some r) :
    ∀ l ∈ slotRounds r, ∀ σ ∈ l, σ.s < r.w.parts.length := by
  intro l hl σ hσ
  obtain ⟨N, hc, hpos⟩ := capAt_readX hr
  have hn : nS x = r.w.parts.length := by simp [nS, hr]
  unfold slotRounds at hl
  rw [hc] at hl
  rcases List.mem_append.1 hl with hl | hl
  · simp only [Option.toList_some, List.map_cons, List.map_nil, List.mem_singleton] at hl
    subst hl
    rw [capSlots_s hσ]
    omega
  · cases hrs : repSched r.shape with
    | none => simp [hrs] at hl
    | some rs =>
      simp only [hrs, Option.map_some, Option.getD_some, List.mem_map, List.mem_range] at hl
      obtain ⟨j, hj, rfl⟩ := hl
      have hpos' : 0 < rs.size := by omega
      unfold roundSlots at hσ
      rcases List.mem_append.1 hσ with hσ | hσ
      · obtain ⟨e, he, hσ⟩ := List.mem_flatMap.1 hσ
        exact capSlotsAt_lt hpos' (topsLen_readX hr hrs) (List.mem_range.1 he) hσ
      · exact rowSlot_lt (Option.mem_toList.1 hσ)

theorem openPieces_lt {x : Msg} {r : XR} (hr : readX x = some r) :
    ∀ p ∈ openPieces r, p.1 < r.w.parts.length := by
  intro p hp
  obtain ⟨N, hc, hpos⟩ := capAt_readX hr
  have hn : nS x = r.w.parts.length := by simp [nS, hr]
  unfold openPieces at hp
  rw [hc] at hp
  rcases List.mem_append.1 hp with hp | hp
  · simp only [Option.toList_some, List.map_cons, List.map_nil, List.mem_singleton] at hp
    subst hp
    show nS x - 1 < _
    omega
  · cases hrs : repSched r.shape with
    | none => simp [hrs] at hp
    | some rs =>
      simp only [hrs, Option.map_some, Option.getD_some] at hp
      obtain ⟨j, hj, hp⟩ := List.mem_flatMap.1 hp
      have hpos' : 0 < rs.size := by
        have := List.mem_range.1 hj
        omega
      exact roundOpen_lt hpos' (topsLen_readX hr hrs) hp

/-! ## The messages as slots -/

theorem held_of_unfixed {L : LayoutB} {x : Msg} {s : ℕ} {w : Claims.Flock.Wit} {i : ℕ} (hf : L.fixedAt x i = none) :
    L.held x s w i = (L.msgAt x s i).map (readSegs w) := by
  simp [LayoutB.held, hf]

/-- **Round `j`'s message is its slots' commit strings end to end**, at every round before the last. -/
theorem msgs_slots (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) {x : Msg}
    (hx : DenseX x) {ws : List Claims.Flock.Wit} (hl : ws.length = nS x) (d : Claims.Flock.Wit) {r : XR}
    (hr : readX x = some r) {j : ℕ} {l : List Slot} (hj : (slotRounds r)[j]? = some l) :
    ((DeployedR.layR elem).msgs x ws).getD j [] = l.flatMap fun σ => σ.commit (ws.getD σ.s d) := by
  obtain ⟨r', rs, s0, spec, st, hr', hrs, hset, hm, hsc, hk⟩ := schedBridge hx
  rw [hr] at hr'
  obtain rfl := Option.some.inj hr'
  have hn : nS x = r.w.parts.length := by simp [nS, hr]
  have hR : layoutOf.rounds x = 2 + 2 * rs.size := hk
  obtain ⟨N, hc, hpos⟩ := capAt_readX hr
  have hSR : slotRounds r =
      capSlots r (nS x - 1) (List.range N) :: (List.range (2 * rs.size)).map (roundSlots r rs) := by
    simp [slotRounds, hc, hrs]
  rw [hSR] at hj
  have hjlen : j < 1 + 2 * rs.size := by
    have := (List.getElem?_eq_some_iff.1 hj).1
    simp at this
    omega
  have hjr : j < (layAt (restR elem)).rounds x := by
    show j < layoutOf.rounds x
    omega
  have hgw : ∀ s (hs : s < ws.length), ws.getD s d = ws[s] := fun s hs => by
    rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hs, Option.getD_some]
  show ((layAt (restR elem)).msgs x ws).getD j [] = _
  rw [(layAt (restR elem)).msgs_getD x ws hjr]
  cases j with
  | zero =>
    simp only [List.getElem?_cons_zero, Option.some.injEq] at hj
    subst hj
    have hp0 : (layAt (restR elem)).pieceAt x 0 = [] := by
      show stepPieces (finOpen (restR elem)) x 0 = []
      unfold stepPieces
      rw [hR, if_neg (by omega), if_pos rfl]
    have hf0 : (layAt (restR elem)).fixedAt x 0 = none := rfl
    have hm0 : ∀ s, (layAt (restR elem)).msgAt x s 0 = capHold r s := fun s => by
      show (readX x).bind (fun r => capHold r s) = _
      rw [hr, Option.bind_some]
    obtain ⟨st', hst, hd⟩ := hx
    obtain ⟨c, hc', hq⟩ := capPorts Claims.Recursion.Form.Dense x st' hst hd r hr
    obtain ⟨q, hq⟩ := Option.isSome_iff_exists.1 hq
    have hcap : ∀ s, s ≠ nS x - 1 → capHold r s = none := fun s hs => by
      simp [capHold, hc, hs]
    have hcapk : capHold r (nS x - 1) = some ((List.range N).map fun n => (([] : Msg), hmCommit c q n)) := by
      simp [capHold, hc, hc', hq]
    have hk' : nS x - 1 < ws.length := by omega
    have hfind : (List.range ws.length).findSome? (fun s => ws[s]?.bind fun w =>
        (layAt (restR elem)).held x s w 0) =
        some (readSegs ws[nS x - 1] ((List.range N).map fun n => (([] : Msg), hmCommit c q n))) := by
      refine findSome?_eq_of_unique _ (List.mem_range.2 hk') ?_ ?_
      · simp only [List.getElem?_eq_getElem hk', Option.bind_some, held_of_unfixed hf0, hm0, hcapk,
          Option.map_some]
      · intro s _ hs
        cases ws[s]? with
        | none => rfl
        | some w => simp only [Option.bind_some, held_of_unfixed hf0, hm0, hcap s hs, Option.map_none]
    simp only [LayoutB.msg, hf0, hp0, List.isEmpty_nil, ↓reduceIte, hfind, Option.getD_some]
    have hcO : cOf r (nS x - 1) = c := by simp [cOf, hc']
    unfold capSlots
    rw [hcO, hq, Option.getD_some, List.flatMap_map]
    unfold readSegs
    rw [List.flatMap_map]
    refine flatMap_congr_mem _ fun n _ => ?_
    simp only [Slot.commit, List.nil_append, hgw _ hk']
  | succ i =>
    simp only [List.getElem?_cons_succ, List.getElem?_map] at hj
    have hi2 : i < 2 * rs.size := by omega
    rw [List.getElem?_range hi2, Option.map_some, Option.some.injEq] at hj
    subst hj
    have hpos' : 0 < rs.size := by omega
    have hsr : streamRound x i = some (rs.getD (i % rs.size) default) := by
      unfold streamRound
      rw [hr, Option.bind_some, hrs, Option.bind_some, if_pos hi2]
    have hpi : (layAt (restR elem)).pieceAt x (i + 1) = topsPiecesAt r rs i := by
      show stepPieces (finOpen (restR elem)) x (i + 1) = _
      unfold stepPieces
      rw [hR, if_neg (by omega), if_neg (by omega), Nat.add_sub_cancel]
      simp [topsPieces, hr, hrs]
    have hfi : (layAt (restR elem)).fixedAt x (i + 1) = if (rs.getD (i % rs.size) default).2.1.isEmpty &&
        !(rs.getD (i % rs.size) default).2.2.2 then some [] else none := by
      show (streamRound x i).bind _ = _
      rw [hsr, Option.bind_some]
    by_cases he : (rs.getD (i % rs.size) default).2.1.isEmpty = true
    · have hpe : topsPiecesAt r rs i = [] := by
        unfold topsPiecesAt
        rw [if_pos he]
      have hsz : (rs.getD (i % rs.size) default).2.1.size = 0 := Array.isEmpty_iff_size_eq_zero.1 he
      have hRS : roundSlots r rs i = (rowSlot r i).toList := by
        unfold roundSlots
        rw [hsz]
        rfl
      rw [hRS]
      by_cases hA : (rs.getD (i % rs.size) default).2.2.2 = true
      · have hf : (layAt (restR elem)).fixedAt x (i + 1) = none := by rw [hfi, he, hA] <;> rfl
        simp only [LayoutB.msg, hf, hpi, hpe, List.isEmpty_nil, ↓reduceIte]
        have hms : ∀ s, (layAt (restR elem)).msgAt x s (i + 1) = (rowSlotAt r i s).map fun σ => [σ.cmt] := by
          intro s
          show (streamRound x i).bind _ = _
          rw [hsr, Option.bind_some]
          simp only [he, hA, ↓reduceIte, hr, Option.bind_some]
          unfold rowSlotAt
          cases ((heldRows r s).find? (·.1 == i)) <;> cases (r.w.parts[s]?.bind partCircuit) <;> rfl
        have hF : ∀ s ∈ List.range ws.length,
            (ws[s]?.bind fun w => (layAt (restR elem)).held x s w (i + 1)) =
              (rowSlotAt r i s).map fun σ => σ.commit (ws.getD σ.s d) := by
          intro s hs
          have hs' := List.mem_range.1 hs
          rw [List.getElem?_eq_getElem hs', Option.bind_some, held_of_unfixed hf, hms]
          cases hσ : rowSlotAt r i s with
          | none => rfl
          | some σ =>
            simp only [Option.map_some, rowSlotAt_s hσ, hgw s hs', readSegs_cmt]
        rw [findSome?_congr_mem _ hF, ← findSome?_map_out, hl, hn]
        show ((rowSlot r i).map _).getD [] = _
        cases rowSlot r i with
        | none => rfl
        | some σ => rw [Option.map_some, Option.getD_some, Option.toList_some, List.flatMap_singleton]
      · have hA' : (rs.getD (i % rs.size) default).2.2.2 = false := by simpa using hA
        have hf : (layAt (restR elem)).fixedAt x (i + 1) = some [] := by rw [hfi, he, hA'] <;> rfl
        have hnone : rowSlot r i = none := by
          unfold rowSlot
          refine List.findSome?_eq_none_iff.2 fun s _ => ?_
          unfold rowSlotAt
          have hfind : (heldRows r s).find? (·.1 == i) = none := by
            refine List.find?_eq_none.2 fun h hh hhi => hA ?_
            have := heldRows_absorbs hh hrs
            rw [beq_iff_eq.1 hhi] at this
            exact this
          rw [hfind]
          rfl
        simp only [LayoutB.msg, hf, hnone]
        rfl
    · have he' : (rs.getD (i % rs.size) default).2.1.isEmpty = false := by simpa using he
      have hf : (layAt (restR elem)).fixedAt x (i + 1) = none := by rw [hfi, he']; rfl
      have hpt : topsPiecesAt r rs i = (List.range (rs.getD (i % rs.size) default).2.1.size).map
          (capPieceAt r rs i) ++ rowPiece r i := by
        unfold topsPiecesAt
        rw [if_neg he]
      have hsz : 0 < (rs.getD (i % rs.size) default).2.1.size := by
        rcases Nat.eq_zero_or_pos (rs.getD (i % rs.size) default).2.1.size with h0 | h0
        · exact absurd (Array.isEmpty_iff_size_eq_zero.2 h0) he
        · exact h0
      have hne : (layAt (restR elem)).pieceAt x (i + 1) ≠ [] := by
        rw [hpi, hpt]
        exact List.ne_nil_of_mem (List.mem_append_left _ (List.mem_map.2 ⟨0, List.mem_range.2 hsz, rfl⟩))
      have hlt : ∀ p ∈ topsPiecesAt r rs i, p.1 < ws.length := by
        intro p hp
        rw [hpt] at hp
        rw [hl, hn]
        rcases List.mem_append.1 hp with hp | hp
        · obtain ⟨e, he', rfl⟩ := List.mem_map.1 hp
          have hlt := topsBefore_lt (Nat.mod_lt i hpos') (List.mem_range.1 he')
          have hlen := topsLen_readX hr hrs
          show capsFrom r rs + topsBefore rs (i % rs.size) + e < r.w.parts.length
          unfold capsFrom
          omega
        · exact rowPiece_lt hp
      rw [LayoutB.msg_of_pieces hf hne]
      unfold LayoutB.assembled
      rw [hpi]
      show readPieces (topsPiecesAt r rs i) ws = _
      rw [readPieces_eq _ ws d hlt, hpt]
      unfold roundSlots
      rw [List.flatMap_append, List.flatMap_append, List.flatMap_map, List.flatMap_assoc, rowPiece_eq]
      congr 1
      · congr 1
        funext e
        show readSegs (ws.getD _ d) (topPiece r _ _ _) = _
        rw [topPiece_eq, readSegs_cmts]
        exact flatMap_congr_mem _ fun σ hσ => by simp only [capSlots_s hσ, capPieceAt]
      · cases rowSlot r i with
        | none => rfl
        | some σ =>
          show readSegs (ws.getD σ.s d) [σ.cmt] ++ [] = σ.commit (ws.getD σ.s d) ++ []
          rw [readSegs_cmt]

/-! ## `OpensSlots` at the layout of record -/

/-- **`OpensSlots` at `layR elem`**, from `SlotsRecorded elem`: `finalR` cuts each round's openings at its slots
(`openLens_eq`, `openBytes`, `cutRounds_groups`), and each round's message is its slots' commit strings
(`msgs_slots`, `commitAt_flatten`). -/
theorem opensSlots_layR (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    (hS : SlotsRecorded elem) : RestLow.OpensSlots (DeployedR.layR elem) := by
  intro x hx ws hl os last hf j i o ho
  obtain ⟨r, hr, hlast, -⟩ := lastAt_parts elem hx hl
  have hn : nS x = r.w.parts.length := by simp [nS, hr]
  obtain ⟨N, -, hpos⟩ := capAt_readX hr
  have h0 : 0 < ws.length := by omega
  obtain ⟨d⟩ : Nonempty Claims.Flock.Wit := ⟨ws[0]⟩
  have hgw : ∀ s (hs : s < ws.length), ws.getD s d = ws[s] := fun s hs => by
    rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hs, Option.getD_some]
  obtain ⟨L, hL⟩ : ∃ L : List (List (Msg × Msg)),
      L = (slotRounds r).map fun l => l.map fun σ => σ.opening (ws.getD σ.s d) := ⟨_, rfl⟩
  have hpl : ∀ p ∈ openPieces r, p.1 < ws.length := fun p hp => by
    rw [hl, hn]
    exact openPieces_lt hr p hp
  have hlens : openLens r = L.map fun l => l.map (·.1.length) := by
    rw [openLens_eq, hL, List.map_map]
    refine List.map_congr_left fun l _ => ?_
    simp only [Function.comp_apply, List.map_map]
    refine List.map_congr_left fun σ _ => ?_
    simp [Slot.opening, readForms_length, pieceLen]
  have hbytes : readPieces (openPieces r) ws = L.flatMap (·.flatMap fun p => p.1 ++ p.2) := by
    rw [openBytes r ws d hpl, hL, List.flatMap_map]
    refine flatMap_congr_mem _ fun l _ => ?_
    simp only [List.flatMap_map]
    exact flatMap_congr_mem _ fun σ _ => readSegs_segs _ σ
  have hsalt : ∀ l ∈ L, ∀ p ∈ l, p.2.length = 192 := by
    intro l hl p hp
    rw [hL] at hl
    obtain ⟨l', -, rfl⟩ := List.mem_map.1 hl
    obtain ⟨σ, -, rfl⟩ := List.mem_map.1 hp
    exact HmOpen.length_salt _ _ _ _
  have hf' : finalR (lastAt (restR elem) x ws) = some (os, last) := hf
  rw [hlast, finalR_trailer, hlens, hbytes, CutOpens.cutRounds_groups _ L hsalt] at hf'
  obtain ⟨rfl, -⟩ := Prod.mk.inj (Option.some.inj hf')
  -- the opening `o` is slot `σ`'s
  rw [hL] at ho
  simp only [List.getD_eq_getElem?_getD, List.getElem?_map] at ho
  cases hj : (slotRounds r)[j]? with
  | none => simp [hj] at ho
  | some l =>
    simp only [hj, Option.map_some, Option.getD_some, List.map_map, List.getElem?_map] at ho
    cases hi : l[i]? with
    | none => simp [hi] at ho
    | some σ =>
      simp only [hi, Option.map_some, Option.some.injEq, Function.comp_apply] at ho
      subst ho
      have hlm : l ∈ slotRounds r := List.mem_of_getElem? hj
      have hσm : σ ∈ l := List.mem_of_getElem? hi
      have hs : σ.s < ws.length := by
        rw [hl, hn]
        exact slotRounds_lt hr l hlm σ hσm
      obtain ⟨hC, hpfx⟩ := hS x hx r hr l hlm σ hσm
      refine ⟨σ.s, hs, σ.c, σ.nm, σ.q, σ.n, fun p hp => ?_, ?_, ?_, ?_⟩
      · unfold LayoutB.calls
        exact List.mem_map.2 ⟨p, hC p hp, rfl⟩
      · rw [hpfx, readForms_length]
        try rfl
      · simp only [Slot.opening, Slot.row, List.getElem?_eq_getElem hs, Option.getD_some]
      · rw [msgs_slots elem hx hl d hr hj, List.flatMap_def]
        have hi' : i < l.length := (List.getElem?_eq_some_iff.1 hi).1
        rw [CutOpens.commitAt_flatten _ (fun m hm => by
            obtain ⟨τ, -, rfl⟩ := List.mem_map.1 hm
            exact HmOpen.length_commit _ _ _ _) i (by simpa using hi'),
          List.getElem_map]
        have hσi : l[i] = σ := (List.getElem?_eq_some_iff.1 hi).2
        rw [hσi, Slot.commit, hgw _ hs]

end RecDeploy.OpenSlots
