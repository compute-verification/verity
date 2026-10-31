import Proofs.Flock.Claims.Recursion.SideCapRead

/-!
# The reader's caps above level 0 (`SideCapUp`), at the deployed `finalR` and `DeployedR.layR`

Above level 0 a piece's cap is the side's `recursiveCaps[ℓ - 1]` (`FlockSoundness.Refine.capAt`), which `ligR` reads
off the opened messages as `topGroups`: rep `k`'s stream rounds' rows, each round cut at its tops' node counts. At the
deployed last message those rows are the rounds' openings (`openPieces`), so the groups are RecCap parts' `top` rows:

* `finalR` cuts every stream round `j` at its openings' row lengths (`openLens`), and reads back the rows the
  openings read off the witnesses (`cutRound_segs`, `cutRound_pieces`, `cutRounds_pieces`, `opened_rows`);
* round `j = k R + i`'s rows start with its tops' openings, the `e`-th at the later cap's part
  `capsFrom + topsBefore rs i + e` over rep `k`'s instances `k n + t` (`capOpenAt`), so cutting at the round's top
  sizes gives exactly those blocks (`round_cut`), and rep `k`'s groups are the blocks of its rounds in order
  (`topGroups_rep`);
* group `g` is round `i`'s top `e` with `g = topsBefore rs i + e`, the schedule's `g`-th top of `n` nodes
  (`topSizes_at`), at V\*'s later cap `.cap (g + 1) (2n)` (`capStmt_at`), and `n = 2^cap` at level `g + 1` of
  `fast100`'s schedule (`topSize_cap`), which `capFirst` reads.

`capEntry_up` is `CapEntryR` above level 0 at every reading of the deployed last message where the last step reads it.
`sideCapUp_of` gives `SideCapUp` at SHA-512, `layR elem` and the bound record at `finalR` from `ReadsAt` and
the opened position's range (`PosCapUp`), and `sideCapR_deployed` puts it with level 0 (`sideCapR_of`).
-/

namespace RecDeploy.CapLower

open Flock
open Claims (Msg Out)
open FlockDeploy (Coin FinalP nodeOf)
open Claims.Recursion.VStarB
open RecDeploy.OpenLower (AtPieces AtOpened sideAtR capR ligR ownQR)
open RecDeploy.RestRead (DenseX)

/-! ## Lists -/

theorem flatMap_congr_mem {α β : Type} {l : List α} {f g : α → List β} (h : ∀ a ∈ l, f a = g a) :
    l.flatMap f = l.flatMap g := by
  rw [List.flatMap_def, List.flatMap_def, List.map_congr_left h]

theorem flatMap_range_at {α : Type} (F : ℕ → List α) {R i e : ℕ} (hi : i < R) (he : e < (F i).length) :
    ((List.range R).flatMap F)[((List.range i).flatMap F).length + e]? = (F i)[e]? := by
  obtain ⟨m, rfl⟩ : ∃ m, R = i + (m + 1) := ⟨R - i - 1, by omega⟩
  rw [List.range_add, List.flatMap_append, List.getElem?_append_right (by omega), Nat.add_sub_cancel_left,
    List.range_succ_eq_map, List.map_cons, List.flatMap_cons, Nat.add_zero, List.getElem?_append_left he]

theorem flatMap_range_split {α : Type} (F : ℕ → List α) : ∀ (R g : ℕ), g < ((List.range R).flatMap F).length →
    ∃ i e, i < R ∧ e < (F i).length ∧ g = ((List.range i).flatMap F).length + e
  | 0, g, h => by simp at h
  | R + 1, g, h => by
    rw [List.range_succ, List.flatMap_append, List.length_append] at h
    by_cases hg : g < ((List.range R).flatMap F).length
    · obtain ⟨i, e, hi, he, rfl⟩ := flatMap_range_split F R g hg
      exact ⟨i, e, by omega, he, rfl⟩
    · refine ⟨R, g - ((List.range R).flatMap F).length, by omega, ?_, by omega⟩
      simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil] at h
      omega

theorem flatMap_range_len {α : Type} (n : ℕ → ℕ) (F : ℕ → ℕ → α) (i : ℕ) :
    ((List.range i).flatMap fun k => (List.range (n k)).map (F k)).length = ((List.range i).map n).sum := by
  rw [List.length_flatMap]
  simp

theorem map_flatMap_flatten {α β γ : Type} (l : List α) (f : α → List β) (g : β → γ) :
    (l.flatMap f).map g = (l.map fun a => (f a).map g).flatten := by
  rw [List.map_flatMap, List.flatMap_def]

theorem range_zip_map {α β : Type} (f : ℕ → α) (g : ℕ → β) (R : ℕ) :
    ((List.range R).map f).zip ((List.range R).map g) = (List.range R).map fun i => (f i, g i) := by
  rw [List.zip_map']

theorem take_toList_range {α : Type} [Inhabited α] (a : Array α) {i : ℕ} (hi : i ≤ a.size) :
    a.toList.take i = (List.range i).map (a.getD · default) := by
  apply List.ext_getElem (by simp; omega)
  intro n h1 h2
  have hn : n < a.size := by simp at h2; omega
  simp [Array.getD, hn]

theorem toList_range_getD {α : Type} [Inhabited α] (a : Array α) :
    a.toList = (List.range a.size).map (a.getD · default) := by
  rw [← take_toList_range a (Nat.le_refl _), List.take_of_length_le (by simp)]

/-- Cutting blocks laid end to end at their own lengths gives the blocks back, whatever follows. -/
theorem cutLens_blocks {α : Type} : ∀ (bs : List (List α)) (rest : List α),
    FlockDeploy.cutLens (bs.map List.length) (bs.flatten ++ rest) = bs
  | [], _ => rfl
  | b :: bs, rest => by
    simp only [List.map_cons, List.flatten_cons, FlockDeploy.cutLens, List.append_assoc, List.take_left',
      List.drop_left', cutLens_blocks bs rest]

/-- **Two reps' rounds cut at their counts**: rep `k`'s are the `k`-th `R`. -/
theorem cutLens_reps {α : Type} (f : ℕ → α) (R : ℕ) {k : ℕ} (hk : k < 2) :
    (FlockDeploy.cutLens [R, R] ((List.range (2 * R)).map f)).getD k [] = (List.range R).map fun i => f (k * R + i) := by
  have e : 2 * R = R + R := by omega
  rw [e, List.range_add, List.map_append, List.map_map]
  rcases (show k = 0 ∨ k = 1 by omega) with rfl | rfl
  · simp [FlockDeploy.cutLens]
  · simp [FlockDeploy.cutLens, Function.comp_def]

theorem caps_get! {α : Type} [Inhabited α] {L : List (List α)} {g : ℕ} {b : List α} (h : L[g]? = some b) :
    (L.map List.toArray).toArray[g]! = b.toArray := by
  obtain ⟨hg, rfl⟩ := List.getElem?_eq_some_iff.1 h
  rw [getElem!_pos _ g (by simpa using hg)]
  simp

/-- The tops before round `i` are its earlier rounds' top counts. -/
theorem topsBefore_sum (rs : Array RecVStar.SchedRound) {i : ℕ} (hi : i ≤ rs.size) :
    topsBefore rs i = ((List.range i).map fun k => ((rs.getD k default).2.1).size).sum := by
  unfold topsBefore
  rw [take_toList_range rs hi, List.length_flatMap, List.map_map]
  rfl

theorem topSizes_sum (rs : Array RecVStar.SchedRound) :
    (RecVStar.topSizes rs).length = ((List.range rs.size).map fun k => ((rs.getD k default).2.1).size).sum := by
  rw [← topsBefore_sum rs (Nat.le_refl _)]
  unfold topsBefore RecVStar.topSizes
  rw [List.take_of_length_le (by simp)]

/-! ## Cutting the openings back -/

/-- Openings that are rows, each followed by a salt of `saltBytes` bytes. -/
def SaltSegs : List (Msg × List HmNets.Lin) → Prop
  | [] => True
  | [_] => False
  | _ :: b :: t => pieceLen [b] = Hm96.sha512.saltBytes ∧ SaltSegs t

theorem saltSegs_cons (a b : Msg × List HmNets.Lin) (t : List (Msg × List HmNets.Lin)) :
    SaltSegs (a :: b :: t) ↔ pieceLen [b] = Hm96.sha512.saltBytes ∧ SaltSegs t := Iff.rfl

/-- The rows of such openings, read off `w`. -/
def segRows (w : Claims.Flock.Wit) : List (Msg × List HmNets.Lin) → List Msg
  | a :: _ :: t => readSegs w [a] :: segRows w t
  | _ => []

theorem segRows_cons (w : Claims.Flock.Wit) (a b : Msg × List HmNets.Lin) (t : List (Msg × List HmNets.Lin)) :
    segRows w (a :: b :: t) = readSegs w [a] :: segRows w t := rfl

/-- **`cutRound` reads openings back**: at their row lengths, the bytes read off `w` are the rows, each then its salt,
and the cut goes on into what follows. -/
theorem cutRound_segs (w : Claims.Flock.Wit) : ∀ (segs : List (Msg × List HmNets.Lin)), SaltSegs segs →
    ∀ {L : List ℕ} {more : Msg} {qs : List (Msg × Salt)} {rest : Msg},
      DeployedR.cutRound L more = some (qs, rest) →
      ∃ ps : List (Msg × Salt), DeployedR.cutRound (DeployedR.rowLens segs ++ L) (readSegs w segs ++ more) =
        some (ps ++ qs, rest) ∧ ps.map Prod.fst = segRows w segs
  | [], _, L, more, qs, rest, h => ⟨[], by simpa [DeployedR.rowLens, readSegs] using h, rfl⟩
  | [_], hs, _, _, _, _, _ => (hs : False).elim
  | a :: b :: t, hs, L, more, qs, rest, h => by
    obtain ⟨hb, ht⟩ := (saltSegs_cons a b t).1 hs
    obtain ⟨ps, hc, hp⟩ := cutRound_segs w t ht h
    have hA : (readSegs w [a]).length = pieceLen [a] := readSegs_length w [a]
    have hB : (readSegs w [b]).length = Hm96.sha512.saltBytes := by rw [readSegs_length, hb]
    have em : readSegs w (a :: b :: t) ++ more = readSegs w [a] ++ (readSegs w [b] ++ (readSegs w t ++ more)) := by
      simp only [readSegs, List.flatMap_cons, List.flatMap_nil, List.append_nil, List.append_assoc]
    have er : DeployedR.rowLens (a :: b :: t) ++ L = pieceLen [a] :: (DeployedR.rowLens t ++ L) := rfl
    have hle : pieceLen [a] + Hm96.sha512.saltBytes ≤
        (readSegs w [a] ++ (readSegs w [b] ++ (readSegs w t ++ more))).length := by
      simp only [List.length_append]
      omega
    refine ⟨(readSegs w [a], DeployedR.saltOf (readSegs w [b])) :: ps, ?_, ?_⟩
    · rw [er, em]
      simp only [DeployedR.cutRound, if_pos hle, List.drop_left' hA, List.drop_left' hB, hc, List.take_left' hA,
        List.take_left' hB, Option.map_some, List.cons_append]
    · rw [List.map_cons, hp, segRows_cons]

theorem saltSegs_openSegs (c : Circuit) (nm : String) (q : ℕ) : ∀ ns : List ℕ, SaltSegs (ns.flatMap (openSegs c nm q))
  | [] => trivial
  | n :: ns => by
    have e : (n :: ns).flatMap (openSegs c nm q) = (([] : Msg), ((rowHidden c nm n).getD []).map HmNets.Lin.var) ::
        (([] : Msg), hmSalt c q n) :: ns.flatMap (openSegs c nm q) := rfl
    rw [e]
    exact (saltSegs_cons _ _ _).2 ⟨RestRead.pieceLen_salt c q n, saltSegs_openSegs c nm q ns⟩

theorem segRows_openSegs (w : Claims.Flock.Wit) (c : Circuit) (nm : String) (q : ℕ) : ∀ ns : List ℕ,
    segRows w (ns.flatMap (openSegs c nm q)) =
      ns.map fun n => readForms w (((rowHidden c nm n).getD []).map HmNets.Lin.var)
  | [] => rfl
  | n :: ns => by
    have e : (n :: ns).flatMap (openSegs c nm q) = (([] : Msg), ((rowHidden c nm n).getD []).map HmNets.Lin.var) ::
        (([] : Msg), hmSalt c q n) :: ns.flatMap (openSegs c nm q) := rfl
    rw [e, segRows_cons, segRows_openSegs w c nm q ns, List.map_cons]
    simp [readSegs]

/-- The rows a piece's openings read off its part's witness (none where the part has none). -/
def pieceRows (ws : List Claims.Flock.Wit) (p : ℕ × List (Msg × List HmNets.Lin)) : List Msg :=
  (ws[p.1]?.map fun w => segRows w p.2).getD []

/-- **`cutRound` reads pieces back**: pieces of parts, each rows and salts, read off the witnesses. -/
theorem cutRound_pieces (ws : List Claims.Flock.Wit) : ∀ (ps : List (ℕ × List (Msg × List HmNets.Lin))),
    (∀ p ∈ ps, p.1 < ws.length ∧ SaltSegs p.2) →
    ∀ {L : List ℕ} {more : Msg} {qs : List (Msg × Salt)} {rest : Msg},
      DeployedR.cutRound L more = some (qs, rest) →
      ∃ os : List (Msg × Salt), DeployedR.cutRound ((ps.flatMap fun p => DeployedR.rowLens p.2) ++ L)
          (DeployedR.readPieces ps ws ++ more) = some (os ++ qs, rest) ∧
        os.map Prod.fst = ps.flatMap (pieceRows ws)
  | [], _, L, more, qs, rest, h => ⟨[], by simpa [DeployedR.readPieces] using h, rfl⟩
  | p :: ps, hp, L, more, qs, rest, h => by
    obtain ⟨os, hc, ho⟩ := cutRound_pieces ws ps (fun q hq => hp q (List.mem_cons_of_mem p hq)) h
    obtain ⟨hlt, hs⟩ := hp p List.mem_cons_self
    have hw : ws[p.1]? = some ws[p.1] := List.getElem?_eq_getElem hlt
    obtain ⟨os', hc', ho'⟩ := cutRound_segs ws[p.1] p.2 hs hc
    refine ⟨os' ++ os, ?_, ?_⟩
    · have e1 : ((p :: ps).flatMap fun p => DeployedR.rowLens p.2) ++ L =
          DeployedR.rowLens p.2 ++ ((ps.flatMap fun p => DeployedR.rowLens p.2) ++ L) := by
        simp only [List.flatMap_cons, List.append_assoc]
      have e2 : DeployedR.readPieces (p :: ps) ws ++ more =
          readSegs ws[p.1] p.2 ++ (DeployedR.readPieces ps ws ++ more) := by
        simp only [DeployedR.readPieces, List.flatMap_cons, hw, Option.map_some, Option.getD_some, List.append_assoc]
      rw [e1, e2, hc', List.append_assoc]
    · rw [List.map_append, ho', ho, List.flatMap_cons]
      simp only [pieceRows, hw, Option.map_some, Option.getD_some]

/-- **`cutRounds` reads rounds of pieces back**, round by round, and leaves what follows. -/
theorem cutRounds_pieces (ws : List Claims.Flock.Wit) (f : ℕ → List (ℕ × List (Msg × List HmNets.Lin))) :
    ∀ (js : List ℕ), (∀ j ∈ js, ∀ p ∈ f j, p.1 < ws.length ∧ SaltSegs p.2) → ∀ (more : Msg),
      ∃ os, DeployedR.cutRounds (js.map fun j => (f j).flatMap fun p => DeployedR.rowLens p.2)
          (DeployedR.readPieces (js.flatMap f) ws ++ more) = some (os, more) ∧
        os.map (List.map Prod.fst) = js.map fun j => (f j).flatMap (pieceRows ws)
  | [], _, more => ⟨[], by simp [DeployedR.readPieces, DeployedR.cutRounds], rfl⟩
  | j :: js, h, more => by
    obtain ⟨os, hc, ho⟩ := cutRounds_pieces ws f js (fun j' hj => h j' (List.mem_cons_of_mem j hj)) more
    have hb : DeployedR.cutRound [] (DeployedR.readPieces (js.flatMap f) ws ++ more) =
        some ([], DeployedR.readPieces (js.flatMap f) ws ++ more) := rfl
    obtain ⟨ps, hp, hpr⟩ := cutRound_pieces ws (f j) (h j List.mem_cons_self) hb
    rw [List.append_nil, List.append_nil] at hp
    refine ⟨ps :: os, ?_, ?_⟩
    · have e : DeployedR.readPieces ((j :: js).flatMap f) ws ++ more =
          DeployedR.readPieces (f j) ws ++ (DeployedR.readPieces (js.flatMap f) ws ++ more) := by
        simp only [List.flatMap_cons, DeployedR.readPieces, List.flatMap_append, List.append_assoc]
      rw [List.map_cons, e]
      simp only [DeployedR.cutRounds, hp, Option.bind_some, hc, Option.map_some]
    · rw [List.map_cons, hpr, ho, List.map_cons]

/-! ## The opened messages' rows -/

/-- **The stream rounds' opened rows**: at a dense compiled statement and witnesses one a part, wherever `finalR`
reads the deployed last message, its first round is round 0's and each later round `j < 2R` is the rows stream round
`j`'s openings read off the witnesses (`roundOpen`, `pieceRows`). -/
theorem opened_rows (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) {x : Msg}
    (hx : DenseX x) {ws : List Claims.Flock.Wit} (hl : ws.length = nS x) {os : List (List (Msg × Salt))} {last : Msg}
    (hf : DeployedR.finalR (RestRead.lastAt (DeployedR.restR elem) x ws) = some (os, last)) :
    ∃ (r : XR) (rs : Array RecVStar.SchedRound) (s0 : FlockDeploy.Ins) (spec : SessionSpec) (st : _root_.Flock.Stmt),
      readX x = some r ∧ repSched r.shape = some rs ∧ FlockDeploy.setupOf false (xIOf x) = some (s0, spec, #[st]) ∧
      st.m = r.shape.m ∧ repSchedAt spec st = rs.toList ∧
      ∃ o0 os', os = o0 :: os' ∧
        os'.map (List.map Prod.fst) = (List.range (2 * rs.size)).map fun j => (roundOpen r rs j).flatMap (pieceRows ws) := by
  obtain ⟨r, rs, s0, spec, st, hr, hrs, hset, hm, hsc, -⟩ := RestRead.schedBridge hx
  obtain ⟨r', hr', hlast, -⟩ := RestRead.lastAt_parts elem hx hl
  rw [hr] at hr'
  have e : r = r' := Option.some.inj hr'
  subst e
  have hn : nS x = r.w.parts.length := by simp [nS, hr]
  have hwl : ws.length = r.w.parts.length := by omega
  have hc0 := RestRead.capAt_level0 hr hm
  obtain ⟨-, -, hp0⟩ := capAt_readX hr
  obtain ⟨st0, hst0, hd0⟩ := hx
  obtain ⟨c', hc', -⟩ := capPorts Claims.Recursion.Form.Dense x st0 hst0 hd0 r hr
  have hlt : nS x - 1 < ws.length := by omega
  obtain ⟨w', hw⟩ : ∃ w', ws[nS x - 1]? = some w' := ⟨_, List.getElem?_eq_getElem hlt⟩
  have hcc : (r.w.parts[nS x - 1]?.bind partCircuit).getD default = c' := by rw [hc', Option.getD_some]
  have hop : openPieces r = (nS x - 1, capOpen r (nS x - 1) (List.range (level0Nodes st))) ::
      (List.range (2 * rs.size)).flatMap (roundOpen r rs) := by
    simp [openPieces, hc0, hrs]
  have hol : DeployedR.openLens r = DeployedR.rowLens (capOpen r (nS x - 1) (List.range (level0Nodes st))) ::
      (List.range (2 * rs.size)).map fun j => (roundOpen r rs j).flatMap fun p => DeployedR.rowLens p.2 := by
    simp [DeployedR.openLens, hc0, hrs]
  have hco : capOpen r (nS x - 1) (List.range (level0Nodes st)) =
      (List.range (level0Nodes st)).flatMap (openSegs c' "top" ((capPort c').getD 0)) := by
    simp only [capOpen, hcc]
  have hrp : DeployedR.readPieces (openPieces r) ws =
      readSegs w' (capOpen r (nS x - 1) (List.range (level0Nodes st))) ++
        DeployedR.readPieces ((List.range (2 * rs.size)).flatMap (roundOpen r rs)) ws := by
    rw [hop]
    simp only [DeployedR.readPieces, List.flatMap_cons, hw, Option.map_some, Option.getD_some]
  rw [hlast, DeployedR.finalR_trailer, hol, hrp, hco, List.append_assoc] at hf
  obtain ⟨ps, hps, -⟩ := cutRound_openSegs w' c' "top" ((capPort c').getD 0) (List.range (level0Nodes st))
    (DeployedR.readPieces ((List.range (2 * rs.size)).flatMap (roundOpen r rs)) ws ++
      DeployedR.readPieces ((elem r).map fun p => (p.1.val, p.2)) ws)
  have hgood : ∀ j ∈ List.range (2 * rs.size), ∀ p ∈ roundOpen r rs j, p.1 < ws.length ∧ SaltSegs p.2 := by
    intro j hj p hp
    have hpos : 0 < rs.size := by have := List.mem_range.1 hj; omega
    refine ⟨by rw [hwl]; exact roundOpen_lt hpos (topsLen_readX hr hrs) hp, ?_⟩
    unfold roundOpen at hp
    rcases List.mem_append.1 hp with hp | hp
    · obtain ⟨e, -, rfl⟩ := List.mem_map.1 hp
      simp only [capOpenAt, capOpen]
      exact saltSegs_openSegs _ _ _ _
    · obtain ⟨c, nm, q, n, he⟩ := RestRead.rowOpen_segs hp
      rw [he]
      exact (saltSegs_cons _ _ _).2 ⟨RestRead.pieceLen_salt c q n, trivial⟩
  obtain ⟨os', hc'', ho'⟩ := cutRounds_pieces ws (roundOpen r rs) (List.range (2 * rs.size)) hgood
    (DeployedR.readPieces ((elem r).map fun p => (p.1.val, p.2)) ws)
  simp only [DeployedR.cutRounds, hps, Option.bind_some, hc'', Option.map_some, Option.some.injEq,
    Prod.mk.injEq] at hf
  exact ⟨r, rs, s0, spec, st, hr, hrs, hset, hm, hsc, ps, os', hf.1.symm, ho'⟩

/-! ## A stream round's tops -/

/-- **A top's openings read back its cap part's rows**: the `e`-th top of round `j = k R + i` is rep `k`'s `n` rows
`k n + t` of `top` at the later cap's part, read off its witness. -/
theorem capOpenAt_rows (r : XR) (rs : Array RecVStar.SchedRound) (ws : List Claims.Flock.Wit) {j e i k : ℕ}
    (hjm : j % rs.size = i) (hjd : j / rs.size = k) {w : Claims.Flock.Wit}
    (hw : ws[capsFrom r rs + topsBefore rs i + e]? = some w) :
    (pieceRows ws (capOpenAt r rs j e)).map (fun m => (⟨m.toArray⟩ : ByteArray)) =
      (List.range (((rs.getD i default).2.1).getD e 0)).map fun t =>
        capRow ((r.w.parts[capsFrom r rs + topsBefore rs i + e]?.bind partCircuit).getD default) w
          (k * ((rs.getD i default).2.1).getD e 0 + t) := by
  unfold capOpenAt
  rw [hjm, hjd]
  unfold pieceRows capOpen
  simp only [hw, Option.map_some, Option.getD_some, segRows_openSegs, List.map_map]
  all_goals rfl

/-- **Round `j`'s rows cut at its tops' node counts are its tops' blocks**: round `j`'s openings are its tops' then
its row's, each top's `n` rows at its cap part (`capOpenAt`), so the cut at the tops' sizes stops at the row. -/
theorem round_cut {x : Msg} {r : XR} (hr : readX x = some r) {rs : Array RecVStar.SchedRound}
    (hrs : repSched r.shape = some rs) {ws : List Claims.Flock.Wit} (hl : ws.length = r.w.parts.length) {j i : ℕ}
    (hi : i < rs.size) (hji : j % rs.size = i) :
    FlockDeploy.cutLens ((rs.getD i default).2.1).toList
        (((roundOpen r rs j).flatMap (pieceRows ws)).map fun m => (⟨m.toArray⟩ : ByteArray)) =
      (List.range ((rs.getD i default).2.1).size).map fun e =>
        (pieceRows ws (capOpenAt r rs j e)).map fun m => (⟨m.toArray⟩ : ByteArray) := by
  have hlen := topsLen_readX hr hrs
  have hblk : ∀ e < ((rs.getD i default).2.1).size,
      ((pieceRows ws (capOpenAt r rs j e)).map fun m => (⟨m.toArray⟩ : ByteArray)).length =
        ((rs.getD i default).2.1).getD e 0 := by
    intro e he
    have hS : capsFrom r rs + topsBefore rs i + e < ws.length := by
      have := topsBefore_lt hi he
      rw [hl]
      unfold capsFrom
      omega
    rw [capOpenAt_rows r rs ws hji rfl (List.getElem?_eq_getElem hS)]
    simp
  have hL : ((List.range ((rs.getD i default).2.1).size).map fun e =>
      (pieceRows ws (capOpenAt r rs j e)).map fun m => (⟨m.toArray⟩ : ByteArray)).map List.length =
        ((rs.getD i default).2.1).toList := by
    rw [List.map_map, ← RestRead.range_map_getD ((rs.getD i default).2.1)]
    apply List.map_congr_left
    intro e he
    exact hblk e (List.mem_range.1 he)
  have hsplit : (((roundOpen r rs j).flatMap (pieceRows ws)).map fun m => (⟨m.toArray⟩ : ByteArray)) =
      ((List.range ((rs.getD i default).2.1).size).map fun e =>
        (pieceRows ws (capOpenAt r rs j e)).map fun m => (⟨m.toArray⟩ : ByteArray)).flatten ++
        (((rowOpen r j).flatMap (pieceRows ws)).map fun m => (⟨m.toArray⟩ : ByteArray)) := by
    unfold roundOpen
    rw [hji, List.flatMap_append, List.map_append, map_flatMap_flatten, List.map_map]
    all_goals rfl
  rw [hsplit, ← hL, cutLens_blocks]

/-- The nodes of the `e`-th top of stream round `k R + i`, as `ligR` reads them off the opened rows. -/
def topNodes (r : XR) (rs : Array RecVStar.SchedRound) (ws : List Claims.Flock.Wit) (k i e : ℕ) : List ByteArray :=
  ((pieceRows ws (capOpenAt r rs (k * rs.size + i) e)).map fun m => (⟨m.toArray⟩ : ByteArray)).map nodeOf

/-- **Rep `k`'s groups are its rounds' tops**, round by round, each top's nodes (`topNodes`). -/
theorem topGroups_rep {x : Msg} {r : XR} (hr : readX x = some r) {rs : Array RecVStar.SchedRound}
    (hrs : repSched r.shape = some rs) {ws : List Claims.Flock.Wit} (hl : ws.length = r.w.parts.length) (k : ℕ) :
    FlockDeploy.topGroups rs.toList ((List.range rs.size).map fun i =>
        ((roundOpen r rs (k * rs.size + i)).flatMap (pieceRows ws)).map fun m => (⟨m.toArray⟩ : ByteArray)) =
      (List.range rs.size).flatMap fun i =>
        (List.range ((rs.getD i default).2.1).size).map (topNodes r rs ws k i) := by
  unfold FlockDeploy.topGroups
  rw [toList_range_getD rs, range_zip_map, List.flatMap_map]
  apply flatMap_congr_mem
  intro i hi
  have hi' := List.mem_range.1 hi
  dsimp only
  rw [round_cut hr hrs hl (j := k * rs.size + i) hi'
    (by rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hi']), List.map_map]
  all_goals rfl

/-! ## The later caps' sizes -/

/-- **The schedule's `g`-th top has `2^cap` nodes at level `g + 1`** of `fast100`'s schedule at the inner `m`: V\*'s
`g`-th later cap holds `2 · 2^cap` instances (`levelCapStatements`), and both reps' `n` nodes (`checkTopsListed`). -/
theorem topSize_cap {x : Msg} {r : XR} (hr : readX x = some r) {rs : Array RecVStar.SchedRound}
    (hrs : repSched r.shape = some rs) {g n : ℕ} (hn : (RecVStar.topSizes rs)[g]? = some n) :
    ∃ s : _root_.Flock.Schedule, _root_.Flock.fast100 r.shape.m = some s ∧ n = 2 ^ (s.levels[g + 1]!).cap := by
  obtain ⟨vs, hvs, -⟩ := readX_statements hr
  obtain ⟨_, _, _, _, _, _, A, C, _, _, _, _, _, _, _, K, N, hK, hT, rfl⟩ := FlockC3.Reps.list_ok hvs
  have hsz := topsListed_size hT hrs
  obtain ⟨hg', -⟩ := List.getElem?_eq_some_iff.1 hn
  have hg : g < K.size := by rw [hsz]; exact hg'
  have hi := topsListed_at hT hrs hn hg
  unfold VStar.levelCapStatements at hK
  cases hs : (FlockC3.Reps.vshOf r.shape).sched with
  | error e => rw [hs] at hK; cases hK
  | ok s =>
    rw [hs] at hK
    cases hK
    refine ⟨s, RestRead.sched_fast100' hs, ?_⟩
    rw [Array.getElem_map, Array.getElem_range] at hi
    simp only [VStar.Stmt.instances] at hi
    omega

/-! ## The reader's caps above level 0 -/

theorem decR_rd_last {rd : Msg → Option FinalP} {x : Msg} {as : List Msg} {cs : List Coin} {d : FlockDeploy.DecR}
    (h : FlockDeploy.decR rd x as cs = some d) : ∃ fl, as.getLast?.bind rd = some fl := by
  simp only [FlockDeploy.decR, Option.bind_eq_bind, Option.pure_def] at h
  obtain ⟨v, -, h⟩ := Option.bind_eq_some_iff.1 h
  obtain ⟨a, ha, h⟩ := Option.bind_eq_some_iff.1 h
  obtain ⟨a₁, -, h⟩ := Option.bind_eq_some_iff.1 h
  obtain ⟨c₁, -, h⟩ := Option.bind_eq_some_iff.1 h
  obtain ⟨fl, hfl, -⟩ := Option.bind_eq_some_iff.1 h
  exact ⟨fl, by rw [ha, Option.bind_some]; exact (Option.filter_eq_some_iff.1 hfl).1⟩

/-- **Where the last step reads the opened messages, `rd` reads the last one.** -/
theorem readOk_last {rd : Msg → Option FinalP} {x : Msg} {ms : List Msg} {cs : List Coin}
    (h : FlockDeploy.ReadOkR rd x ms cs) : ∃ fl, ms.getLast?.bind rd = some fl := by
  unfold FlockDeploy.ReadOkR at h
  revert h
  cases ha : FlockDeploy.actR rd x ms cs with
  | rej => intro h; exact (h : False).elim
  | coin c => intro h; exact (h : False).elim
  | decide s spec sts link c₁ mids f =>
    intro _
    exact decR_rd_last (FlockDeploy.actR_decide ha)

theorem mapM_rowsOf_joinR : ∀ os : List (List (Msg × Salt)),
    (os.map fun o => FlockDeploy.joinR (o.map Prod.fst)).mapM FlockDeploy.rowsOf =
      some (os.map fun o => (o.map Prod.fst).map fun m => (⟨m.toArray⟩ : ByteArray))
  | [] => rfl
  | o :: os => by
    simp only [List.map_cons]
    rw [List.mapM_cons, FlockDeploy.rowsOf_joinR', mapM_rowsOf_joinR os]
    all_goals rfl

/-- **The side's cap entry above level 0 is a later cap's `top` row's node**, at every reading of the deployed last
message where `rd` reads its last message: at a level-`ℓ ≥ 1` piece of rep `k < 2` of table 0, the side's
`recursiveCaps[ℓ - 1]` is rep `k`'s `(ℓ - 1)`-th group (`topGroups_rep`), round `i`'s top `e` with
`ℓ - 1 = topsBefore rs i + e`, the rows `k n + t` of `top` at part `capsFrom + ℓ - 1`, V\*'s `.cap ℓ (2n)`
(`capStmt_at`), with `n = 2^cap` at level `ℓ` (`topSize_cap`), so its entry at the opened node `t < n` is row
`capIdx = k n + t`'s node. Each part has a circuit (`hcirc`, which the parts' acceptance gives: `staged_circuit`). -/
theorem capEntry_up (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    (rd : Msg → Option FinalP) (posAt : Msg → List Coin → OpenPiece → ℕ) {x : Msg} (hx : DenseX x)
    {ws : List Claims.Flock.Wit} (hl : ws.length = nS x) {cs : List Coin} {HH : ℕ} {q : OpenPiece}
    (hl0 : q.level ≠ 0) (ht : q.table = 0) (hrep : q.rep < 2)
    (hpos : ∀ (r : XR) (rs : Array RecVStar.SchedRound), readX x = some r → repSched r.shape = some rs →
      ∃ n, (RecVStar.topSizes rs)[q.level - 1]? = some n ∧ posAt (xIOf x) cs q >>> HH < n)
    (hcirc : ∀ r : XR, readX x = some r → ∀ s' < nS x, ∃ c', r.w.parts[s']?.bind partCircuit = some c')
    {os : List (List (Msg × Salt))} {last : Msg}
    (hf : DeployedR.finalR (RestRead.lastAt (DeployedR.restR elem) x ws) = some (os, last))
    (hrd : ∃ fl, rd last = some fl) :
    CapEntryR rd posAt x (xIOf x) cs ws HH q (os.map (fun o => FlockDeploy.joinR (o.map Prod.fst)) ++ [last]) := by
  obtain ⟨r, rs, s0, spec, st, hr, hrs, hset, hm, hsc, o0, os', rfl, hos'⟩ := opened_rows elem hx hl hf
  obtain ⟨fl, hfl⟩ := hrd
  have hn : nS x = r.w.parts.length := by simp [nS, hr]
  have hwl : ws.length = r.w.parts.length := by omega
  obtain ⟨vs, hvs, -⟩ := readX_statements hr
  obtain ⟨n, hgn, hlt⟩ := hpos r rs hr hrs
  have hl1 : q.level - 1 + 1 = q.level := by omega
  -- the side's groups are rep `q.rep`'s rounds' tops
  have hrow0 : FlockDeploy.rowsOf
      (((o0 :: os').map (fun o => FlockDeploy.joinR (o.map Prod.fst)) ++ [last]).headD []) =
        some ((o0.map Prod.fst).map fun m => (⟨m.toArray⟩ : ByteArray)) :=
    FlockDeploy.rowsOf_joinR' _
  have htail : ((o0 :: os').map (fun o => FlockDeploy.joinR (o.map Prod.fst)) ++ [last]).tail.dropLast =
      os'.map fun o => FlockDeploy.joinR (o.map Prod.fst) := by
    simp
  have hrss : ((o0 :: os').map (fun o => FlockDeploy.joinR (o.map Prod.fst)) ++ [last]).tail.dropLast.mapM
      FlockDeploy.rowsOf = some (os'.map fun o => (o.map Prod.fst).map fun m => (⟨m.toArray⟩ : ByteArray)) := by
    rw [htail]
    exact mapM_rowsOf_joinR os'
  have hlastb : ((o0 :: os').map (fun o => FlockDeploy.joinR (o.map Prod.fst)) ++ [last]).getLast?.bind rd =
      some fl := by
    rw [List.getLast?_concat]
    exact hfl
  have hsched : ∀ r0 cs', (FlockDeploy.repCtx spec #[st] default r0 cs' q.rep).sched = rs.toList := by
    intro r0 cs'
    show repSchedAt spec (#[st])[q.rep / 2]! = _
    rw [Nat.div_eq_of_lt hrep]
    exact hsc
  have hlens : FlockDeploy.repLens spec #[st] = [rs.size, rs.size] := by
    show [(repSchedAt spec (#[st])[0 / 2]!).length, (repSchedAt spec (#[st])[1 / 2]!).length] = _
    have e0 : (#[st])[0 / 2]! = st := rfl
    have e1 : (#[st])[1 / 2]! = st := rfl
    rw [e0]
    try rw [e1]
    rw [hsc, Array.length_toList]
  have hrssE : (os'.map fun o => (o.map Prod.fst).map fun m => (⟨m.toArray⟩ : ByteArray)) =
      (List.range (2 * rs.size)).map fun j =>
        ((roundOpen r rs j).flatMap (pieceRows ws)).map fun m => (⟨m.toArray⟩ : ByteArray) := by
    calc (os'.map fun o => (o.map Prod.fst).map fun m => (⟨m.toArray⟩ : ByteArray))
        = (os'.map (List.map Prod.fst)).map (List.map fun m => (⟨m.toArray⟩ : ByteArray)) := by
          rw [List.map_map]; all_goals rfl
      _ = _ := by rw [hos', List.map_map]; all_goals rfl
  have hk : 2 * q.table + q.rep = q.rep := by rw [ht]; omega
  have hrec : (ligR rd (xIOf x) ((o0 :: os').map (fun o => FlockDeploy.joinR (o.map Prod.fst)) ++ [last]) q).recursiveCaps =
      (((List.range rs.size).flatMap fun i =>
        (List.range ((rs.getD i default).2.1).size).map (topNodes r rs ws q.rep i)).map List.toArray).toArray := by
    have h1 : (ligR rd (xIOf x) ((o0 :: os').map (fun o => FlockDeploy.joinR (o.map Prod.fst)) ++ [last]) q).recursiveCaps =
        ((FlockDeploy.topGroups (FlockDeploy.repCtx spec #[st] default
          ((o0.map Prod.fst).map fun m => (⟨m.toArray⟩ : ByteArray)) [] (2 * q.table + q.rep)).sched
          ((FlockDeploy.cutLens (FlockDeploy.repLens spec #[st])
            (os'.map fun o => (o.map Prod.fst).map fun m => (⟨m.toArray⟩ : ByteArray))).getD
              (2 * q.table + q.rep) [])).map List.toArray).toArray := by
      unfold ligR
      rw [hset, hlastb, hrow0, hrss] <;> rfl
    rw [h1, hk, hsched, hlens, hrssE, cutLens_reps _ _ hrep, topGroups_rep hr hrs hwl q.rep]
  -- the group at `ℓ - 1` is round `i`'s top `e`
  have hlenG : ((List.range rs.size).flatMap fun i =>
      (List.range ((rs.getD i default).2.1).size).map (topNodes r rs ws q.rep i)).length =
        (RecVStar.topSizes rs).length := by
    rw [flatMap_range_len (fun i => ((rs.getD i default).2.1).size) (topNodes r rs ws q.rep), topSizes_sum]
  obtain ⟨i, e, hi, he, hge⟩ := flatMap_range_split
    (fun i => (List.range ((rs.getD i default).2.1).size).map (topNodes r rs ws q.rep i)) rs.size (q.level - 1)
    (by rw [hlenG]; exact (List.getElem?_eq_some_iff.1 hgn).1)
  have he' : e < ((rs.getD i default).2.1).size := by simpa using he
  have hbef : ((List.range i).flatMap fun i =>
      (List.range ((rs.getD i default).2.1).size).map (topNodes r rs ws q.rep i)).length = topsBefore rs i := by
    rw [flatMap_range_len (fun i => ((rs.getD i default).2.1).size) (topNodes r rs ws q.rep),
      topsBefore_sum rs hi.le]
  have hat := flatMap_range_at
    (fun i => (List.range ((rs.getD i default).2.1).size).map (topNodes r rs ws q.rep i)) hi he
  rw [hbef] at hge hat
  rw [← hge] at hat
  try dsimp only at hat
  rw [List.getElem?_map, List.getElem?_range he', Option.map_some] at hat
  have hT := topSizes_at hi he'
  rw [← hge, hgn] at hT
  have hnT : ((rs.getD i default).2.1).getD e 0 = n := by
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem he', Option.getD_some]
    exact (Option.some.inj hT).symm
  -- its part, statement, circuit and witness
  have hS : capsFrom r rs + topsBefore rs i + e = capsFrom r rs + (q.level - 1) := by omega
  have hSlt : capsFrom r rs + (q.level - 1) < ws.length := by
    have := topsBefore_lt hi he'
    have := topsLen_readX hr hrs
    rw [hwl, ← hS]
    unfold capsFrom
    omega
  obtain ⟨c', hc'⟩ := hcirc r hr (capsFrom r rs + (q.level - 1)) (by omega)
  have hw' : ws[capsFrom r rs + (q.level - 1)]? = some ws[capsFrom r rs + (q.level - 1)] :=
    List.getElem?_eq_getElem hSlt
  have hss : vs[capsFrom r rs + (q.level - 1)]? = some (.cap q.level (2 * n)) := by
    have h := capStmt_at hr hrs hgn
    rw [hvs, hl1] at h
    exact h
  have hnodes : topNodes r rs ws q.rep i e =
      ((List.range n).map fun t => capRow c' ws[capsFrom r rs + (q.level - 1)] (q.rep * n + t)).map nodeOf := by
    unfold topNodes
    rw [capOpenAt_rows r rs ws (j := q.rep * rs.size + i) (i := i) (k := q.rep)
      (w := ws[capsFrom r rs + (q.level - 1)])
      (by rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hi])
      (by rw [Nat.add_comm, Nat.add_mul_div_right _ _ (by omega), Nat.div_eq_of_lt hi, Nat.zero_add])
      (by rw [hS]; exact hw'), hS, hc', Option.getD_some, hnT]
  -- `capFirst` at the schedule's level `ℓ`
  obtain ⟨sc, hsc', hpow⟩ := topSize_cap hr hrs hgn
  rw [hl1] at hpow
  have hschAt : OpenLower.schAt (xIOf x) 0 = sc := by
    unfold OpenLower.schAt
    rw [hset]
    show (_root_.Flock.fast100 (FlockDeploy.tabsOf #[st])[0]!.m).getD default = sc
    have htab : (FlockDeploy.tabsOf #[st])[0]! = _root_.Flock.Setup.ofCircuit st := by
      unfold FlockDeploy.tabsOf
      rw [getElem!_pos (#[st].map _root_.Flock.Setup.ofCircuit) 0 (by simp), Array.getElem_map] <;> rfl
    rw [htab]
    show (_root_.Flock.fast100 st.m).getD default = sc
    rw [hm, hsc', Option.getD_some]
  have hfirst : capFirst (xIOf x) q = q.rep * n := by
    rw [capFirst, if_neg hl0, hschAt, ← hpow]
  refine ⟨r, vs, hr, hvs, capsFrom r rs + (q.level - 1), 2 * n, c', ws[capsFrom r rs + (q.level - 1)], hss, hc', hw',
    ?_, ?_⟩
  · rw [capIdx, hfirst]
    rcases (show q.rep = 0 ∨ q.rep = 1 by omega) with h | h <;> rw [h] <;> omega
  · rw [capIdx, hfirst]
    show (FlockSoundness.Refine.capAt (capR (xIOf x) _ q) (ligR rd (xIOf x) _ q) q.level).getD _ ByteArray.empty = _
    rw [FlockSoundness.Refine.capAt, if_neg hl0, hrec, caps_get! hat, hnodes, Array.getD_eq_getD_getElem?,
      List.getElem?_toArray, List.getElem?_map, List.getElem?_map, List.getElem?_range hlt, Option.map_some,
      Option.map_some, Option.getD_some]

/-- The layout of record records every call `layAt` does. -/
theorem allCalls_layAt_layR (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (x : Msg)
    (ws : List Claims.Flock.Wit) {e : Msg × Msg}
    (h : e ∈ (RestRead.layAt (DeployedR.restR elem)).allCalls x ws) : e ∈ (DeployedR.layR elem).allCalls x ws := by
  unfold LayoutB.allCalls at h ⊢
  obtain ⟨l, hl, he⟩ := List.mem_flatten.1 h
  obtain ⟨i, hi, rfl⟩ := List.mem_iff_getElem.1 hl
  have hi' : i < ws.length := by simpa using hi
  refine List.mem_flatten.2 ⟨(DeployedR.layR elem).calls x i ws[i], List.mem_iff_getElem.2 ⟨i, by simpa using hi',
    by simp⟩, ?_⟩
  exact DeployedR.layoutBTops_sub_layR elem _ x i ws[i] e (by simpa using he)

section Deployed

variable (D : Claims.Recursion.Deploy Coin) (shape : Claims.Recursion.Form → Prop)
  (untr : TrReader) (b : Brings) (cmt : Msg → ℕ → Msg → Msg) (L : LayoutB)
  (Bd : Claims.Recursion.Bound Coin (List FlockLevel3.GF128) Claims.Flock.PlainTrace) (rd : Msg → Option FinalP)

/-- **The opened position above level 0 is one of its top's nodes**: at a piece of a level `ℓ ≥ 1`, one of its
table's two reps, the opening's node (`posAt`, below `HH` siblings) is below the node count `n` of the schedule's
`(ℓ - 1)`-th top. -/
def PosCapUp (H : Msg → Msg) (pieces : Msg → List OpenPiece) (posAt : Msg → List Coin → OpenPiece → ℕ) : Prop :=
  AtPieces H D shape untr b cmt L pieces fun x c _ _ HH _ _ _ _ _ q => q.level ≠ 0 →
    q.rep < 2 ∧ ∀ (r : XR) (rs : Array RecVStar.SchedRound), readX x = some r → repSched r.shape = some rs →
      ∃ n, (RecVStar.topSizes rs)[q.level - 1]? = some n ∧
        posAt (Claims.Recursion.xI D x) ((List.range (Bd.DE.V.k (Claims.Recursion.xI D x))).map c) q >>> HH < n

/-- **The reader above level 0 at the deployed layout and last message**: at SHA-512, `layR elem` and the bound
record at `finalR` with rows joined by `joinR`, a piece of level `ℓ ≥ 1` has its cap entry at a later cap's `top` row
(`capEntry_up`): the parts' acceptance gives each part a circuit, and `ReadsAt` that the last step reads
the opened messages, at the calls `layR` records (`allCalls_layAt_layR`). -/
theorem sideCapUp_of (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (nextE : Msg → List Msg → List Coin → Claims.OC Out) (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128))
    (wE : Msg → ℝ) (rowIn : Msg → Msg) (w : Msg → ℝ)
    (pieces : Msg → List OpenPiece) (posAt : Msg → List Coin → OpenPiece → ℕ)
    (hP : ∀ x, D.parse x = parse x) (hN : ∀ x, D.nS x = nS x) (hD : ∀ f, shape f → Claims.Recursion.Form.Dense f)
    (hRd : RestRead.ReadsAt rd DeployedR.finalR (DeployedR.restR elem))
    (hup : PosCapUp D shape untr b cmt (DeployedR.layR elem)
      (bdDeployed rel nextE colsE wE FlockDeploy.joinR rowIn DeployedR.finalR w) FlockDeploy.sha512 pieces posAt) :
    SideCapUp FlockDeploy.sha512 D shape untr b cmt (DeployedR.layR elem)
      (bdDeployed rel nextE colsE wE FlockDeploy.joinR rowIn DeployedR.finalR w) rd pieces posAt := by
  intro x st hx hf c ws hJ r ss s lv LANES HH n hr hss hsv xO w' p hst q hq ho ht hl0
  obtain ⟨hrep, hpos⟩ := hup x st hx hf c ws hJ r ss s lv LANES HH n hr hss hsv xO w' p hst q hq ho ht hl0
  intro os last hfin hos
  have hxi : Claims.Recursion.xI D x = xIOf x := by
    unfold Claims.Recursion.xI xIOf
    rw [hP]
  have hDx : DenseX x := by
    unfold DenseX
    exact ⟨st, by rw [← hP]; exact hx, hD _ hf⟩
  have hl : ws.length = nS x := by rw [← hN]; exact hJ.1
  have htab : q.table = 0 := by
    unfold ownQR at ho
    by_contra hne
    rw [if_neg hne] at ho
    cases ho
  have hcirc : ∀ r' : XR, readX x = some r' → ∀ s' < nS x, ∃ c', r'.w.parts[s']?.bind partCircuit = some c' := by
    intro r' hr' s' hs'
    obtain ⟨-, xO', w'', -, -, -, hsh, -⟩ := hJ.2.2.2 s' (by rw [hN]; exact hs')
    exact staged_circuit hsh.2 hr'
  rw [hxi] at hpos hfin hos ⊢
  rw [msgs_layR] at hfin
  have hT : ∀ e ∈ (RestRead.layAt (DeployedR.restR elem)).allCalls x ws, FlockDeploy.sha512 e.1 = e.2 :=
    fun e he => hJ.2.1 e (allCalls_layAt_layR elem x ws he)
  obtain ⟨fl, hfl⟩ := readOk_last (hRd x hDx c ws hl hT os last hfin hos)
  exact capEntry_up elem rd posAt hDx hl hl0 htab hrep hpos hcirc hfin ⟨fl, by simpa using hfl⟩

/-- **The reader at the deployed layout and last message** (`SideCapR`), at SHA-512: level 0 from its position range
(`sideCapR_of`), the levels above from theirs and `ReadsAt` (`sideCapUp_of`). -/
theorem sideCapR_deployed (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (nextE : Msg → List Msg → List Coin → Claims.OC Out) (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128))
    (wE : Msg → ℝ) (rowIn : Msg → Msg) (w : Msg → ℝ)
    (pieces : Msg → List OpenPiece) (posAt : Msg → List Coin → OpenPiece → ℕ)
    (hP : ∀ x, D.parse x = parse x) (hN : ∀ x, D.nS x = nS x) (hD : ∀ f, shape f → Claims.Recursion.Form.Dense f)
    (hRd : RestRead.ReadsAt rd DeployedR.finalR (DeployedR.restR elem))
    (h0 : PosCap0 FlockDeploy.sha512 D shape untr b cmt (DeployedR.layR elem)
      (bdDeployed rel nextE colsE wE FlockDeploy.joinR rowIn DeployedR.finalR w) pieces posAt)
    (hup : PosCapUp D shape untr b cmt (DeployedR.layR elem)
      (bdDeployed rel nextE colsE wE FlockDeploy.joinR rowIn DeployedR.finalR w) FlockDeploy.sha512 pieces posAt) :
    SideCapR FlockDeploy.sha512 D shape untr b cmt (DeployedR.layR elem)
      (bdDeployed rel nextE colsE wE FlockDeploy.joinR rowIn DeployedR.finalR w) rd pieces posAt :=
  sideCapR_of FlockDeploy.sha512 D shape untr b cmt rd elem rel nextE colsE wE rowIn w pieces posAt hP hN hD h0
    (sideCapUp_of D shape untr b cmt rd elem rel nextE colsE wE rowIn w pieces posAt hP hN hD hRd hup)

end Deployed

end RecDeploy.CapLower
