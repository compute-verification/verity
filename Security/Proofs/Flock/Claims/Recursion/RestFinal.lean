import Proofs.Flock.Claims.Recursion.RestLens
import Proofs.Flock.Claims.Recursion.DeployedR

/-!
# `final` reads the last message at every witnesses (`FinalReads`), at the deployed `finalR` and `restR elem`

`hrest_shape` and `hrest_len` take `FinalReads final fin`: at a dense compiled statement and witnesses one a part,
`final` reads the last message off the witnesses as `k - 1` rounds' rows and salts, round `j`'s as many as the
schedule counts (`nAtSched`). It holds at `finalR` and `restR elem` for every `elem` (`finalReads`), with no
hypothesis:

* **The last message** is assembled from `finOpen (restR elem)`'s pieces: every earlier round's openings
  (`openPieces`), `elem`'s pieces, then the trailer at the last part, whose bytes are constant (`lastAt_parts`).
* **`finalR` reads the trailer off the end** (`finalR_trailer`), so it cuts the rest at the trailer's lengths
  (`openLens`). The openings' bytes are what those lengths sum to (`lastAt_parts`), since every piece is rows each
  followed by its 192-byte salt (`rowLens_openSegs`); so it reads `openLens`' rounds, each as many rows
  as its lengths (`cutRounds_ok`), whatever `elem` reads, and what follows them is `elem`'s pieces (`finalR_last`).
* **The counts are the schedule's**: round 0 opens RecCap's `N = level0Nodes` instances (`capAt_level0`), and stream
  round `i` each top's `n` instances (`capsOpen_count`) and the row it absorbs where it absorbs one (`rowOpen_count`,
  `rowsCoverAll`), which `nAtSched_succ` counts.
-/

namespace RecDeploy.RestRead

open Flock
open Claims (Msg)
open Claims.Recursion.VStarB
open RecDeploy.DeployedR

/-! ## Cutting at lengths -/

/-- A round's bytes at its rows' lengths: each row, then its salt. -/
def sumS (l : List ℕ) : ℕ := (l.map (· + Hm96.sha512.saltBytes)).sum

/-- Every round's bytes at their rows' lengths. -/
def cutLen (lens : List (List ℕ)) : ℕ := (lens.map sumS).sum

theorem sumS_append (a b : List ℕ) : sumS (a ++ b) = sumS a + sumS b := by
  simp only [sumS, List.map_append, List.sum_append]

theorem cutRound_ok : ∀ (l : List ℕ) (m : Msg), sumS l ≤ m.length →
    ∃ ps rest, cutRound l m = some (ps, rest) ∧ ps.length = l.length ∧ rest.length = m.length - sumS l
  | [], m, _ => ⟨[], m, rfl, rfl, by simp [sumS]⟩
  | ℓ :: ls, m, h => by
    have hs : sumS (ℓ :: ls) = ℓ + Hm96.sha512.saltBytes + sumS ls := by
      simp only [sumS, List.map_cons, List.sum_cons]
    rw [hs] at h ⊢
    obtain ⟨ps, rest, hc, hl, hr⟩ := cutRound_ok ls ((m.drop ℓ).drop Hm96.sha512.saltBytes)
      (by simp only [List.length_drop]; omega)
    refine ⟨(m.take ℓ, saltOf ((m.drop ℓ).take Hm96.sha512.saltBytes)) :: ps, rest, ?_, by simp [hl], ?_⟩
    · simp only [cutRound, if_pos (show ℓ + Hm96.sha512.saltBytes ≤ m.length by omega), hc, Option.map_some]
    · rw [hr, List.length_drop, List.length_drop]
      omega

/-- **`cutRounds` reads every round where the bytes suffice**, each as many rows as its lengths. -/
theorem cutRounds_ok : ∀ (lens : List (List ℕ)) (m : Msg), cutLen lens ≤ m.length →
    ∃ os last, cutRounds lens m = some (os, last) ∧ os.length = lens.length ∧
      ∀ j, (os.getD j []).length = (lens.getD j []).length
  | [], m, _ => ⟨[], m, rfl, rfl, fun j => by simp⟩
  | l :: ls, m, h => by
    have hs : cutLen (l :: ls) = sumS l + cutLen ls := by simp only [cutLen, List.map_cons, List.sum_cons]
    rw [hs] at h
    obtain ⟨ps, rest, hc, hl, hr⟩ := cutRound_ok l m (by omega)
    obtain ⟨os, last, hcs, hol, hoj⟩ := cutRounds_ok ls rest (by omega)
    refine ⟨ps :: os, last, ?_, by simp [hol], fun j => ?_⟩
    · simp only [cutRounds, hc, Option.bind_some, hcs, Option.map_some]
    · cases j with
      | zero => simpa using hl
      | succ j => simpa using hoj j

theorem drop_drop3 (m : Msg) (a b c : ℕ) : ((m.drop a).drop b).drop c = m.drop (a + b + c) := by
  simp only [List.drop_drop, Nat.add_assoc]

theorem cutRound_rest : ∀ (l : List ℕ) (m : Msg) {ps : List (Msg × Salt)} {rest : Msg},
    cutRound l m = some (ps, rest) → rest = m.drop (sumS l)
  | [], m, ps, rest, h => by
    have h' : (([] : List (Msg × Salt)), m) = (ps, rest) := Option.some.inj h
    rw [← (Prod.mk.inj h').2]
    simp [sumS]
  | ℓ :: ls, m, ps, rest, h => by
    have hs : sumS (ℓ :: ls) = ℓ + Hm96.sha512.saltBytes + sumS ls := by
      simp only [sumS, List.map_cons, List.sum_cons]
    unfold cutRound at h
    split_ifs at h with hle
    obtain ⟨⟨ps', rest'⟩, hp, he⟩ := Option.map_eq_some_iff.1 h
    simp only [Prod.mk.injEq] at he
    rw [← he.2, cutRound_rest ls _ hp, hs, drop_drop3]

/-- **What follows the rounds** is the message past their bytes. -/
theorem cutRounds_rest : ∀ (lens : List (List ℕ)) (m : Msg) {os : List (List (Msg × Salt))} {last : Msg},
    cutRounds lens m = some (os, last) → last = m.drop (cutLen lens)
  | [], m, os, last, h => by
    have h' : (([] : List (List (Msg × Salt))), m) = (os, last) := Option.some.inj h
    rw [← (Prod.mk.inj h').2]
    simp [cutLen]
  | l :: ls, m, os, last, h => by
    have hs : cutLen (l :: ls) = sumS l + cutLen ls := by simp only [cutLen, List.map_cons, List.sum_cons]
    simp only [cutRounds] at h
    obtain ⟨⟨ps, rest⟩, h1, h⟩ := Option.bind_eq_some_iff.1 h
    obtain ⟨⟨os', last'⟩, h2, he⟩ := Option.map_eq_some_iff.1 h
    rw [← (Prod.mk.inj he).2, cutRounds_rest ls _ h2, cutRound_rest l m h1, List.drop_drop, hs]

/-! ## The openings' bytes and rows -/

theorem pieceLen_append (a b : List (Msg × List HmNets.Lin)) : pieceLen (a ++ b) = pieceLen a + pieceLen b := by
  simp only [pieceLen, List.map_append, List.sum_append]

theorem pieceLen_salt (c : Circuit) (q n : ℕ) :
    pieceLen [(([] : Msg), hmSalt c q n)] = Hm96.sha512.saltBytes := by
  simp only [pieceLen, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, List.length_nil, hmSalt,
    List.length_map, List.length_range]
  rfl

/-- **Openings are rows each followed by its salt**: one row an instance, and the bytes are the rows' and salts'. -/
theorem rowLens_openSegs (c : Circuit) (nm : String) (q : ℕ) : ∀ ns : List ℕ,
    (rowLens (ns.flatMap (openSegs c nm q))).length = ns.length ∧
      pieceLen (ns.flatMap (openSegs c nm q)) = sumS (rowLens (ns.flatMap (openSegs c nm q)))
  | [] => ⟨rfl, rfl⟩
  | n :: ns => by
    obtain ⟨h1, h2⟩ := rowLens_openSegs c nm q ns
    have e : (n :: ns).flatMap (openSegs c nm q) = (([] : Msg), ((rowHidden c nm n).getD []).map HmNets.Lin.var) ::
        (([] : Msg), hmSalt c q n) :: ns.flatMap (openSegs c nm q) := rfl
    have er : ∀ (a b : Msg × List HmNets.Lin) t, rowLens (a :: b :: t) = pieceLen [a] :: rowLens t :=
      fun _ _ _ => rfl
    have ep : ∀ (a b : Msg × List HmNets.Lin) t, pieceLen (a :: b :: t) = pieceLen [a] + pieceLen [b] + pieceLen t :=
      fun a b t => by
        simp only [pieceLen, List.map_cons, List.sum_cons, List.map_nil, List.sum_nil]
        omega
    rw [e, er, ep, pieceLen_salt, h2]
    refine ⟨by simp [h1], ?_⟩
    simp only [sumS, List.map_cons, List.sum_cons]

theorem openSegs_good (c : Circuit) (nm : String) (q n : ℕ) :
    pieceLen (openSegs c nm q n) = sumS (rowLens (openSegs c nm q n)) ∧ (rowLens (openSegs c nm q n)).length = 1 := by
  obtain ⟨h1, h2⟩ := rowLens_openSegs c nm q [n]
  have e : [n].flatMap (openSegs c nm q) = openSegs c nm q n := by simp
  rw [e] at h1 h2
  exact ⟨h2, h1⟩

theorem capOpen_good (r : XR) (s : ℕ) (ns : List ℕ) :
    pieceLen (capOpen r s ns) = sumS (rowLens (capOpen r s ns)) := by
  simp only [capOpen]
  exact (rowLens_openSegs _ _ _ ns).2

theorem capOpen_count (r : XR) (s : ℕ) (ns : List ℕ) : (rowLens (capOpen r s ns)).length = ns.length := by
  simp only [capOpen]
  exact (rowLens_openSegs _ _ _ ns).1

theorem rowOpen_segs {r : XR} {j : ℕ} {p : ℕ × List (Msg × List HmNets.Lin)} (hp : p ∈ rowOpen r j) :
    ∃ c nm q n, p.2 = openSegs c nm q n := by
  unfold rowOpen at hp
  simp only [Option.mem_toList] at hp
  obtain ⟨a, -, hfa⟩ := List.exists_of_findSome?_eq_some hp
  rw [Option.bind_eq_some_iff] at hfa
  obtain ⟨h, -, hfa⟩ := hfa
  rw [Option.map_eq_some_iff] at hfa
  obtain ⟨c, -, rfl⟩ := hfa
  exact ⟨c, _, _, _, rfl⟩

theorem findSome_isSome_congr {α β γ : Type} {f : α → Option β} {g : α → Option γ}
    (h : ∀ a, (f a).isSome = (g a).isSome) : ∀ l : List α, (l.findSome? f).isSome = (l.findSome? g).isSome
  | [] => rfl
  | a :: l => by
    have ha := h a
    rw [List.findSome?_cons, List.findSome?_cons]
    cases hf : f a with
    | some b =>
      cases hg : g a with
      | some c => rfl
      | none => rw [hf, hg] at ha; cases ha
    | none =>
      cases hg : g a with
      | some c => rw [hf, hg] at ha; cases ha
      | none => exact findSome_isSome_congr h l

theorem toList_length_congr {α β : Type} {o : Option α} {o' : Option β} (h : o.isSome = o'.isSome) :
    o.toList.length = o'.toList.length := by
  cases o <;> cases o' <;> simp_all

/-- The row stream round `j` absorbs is opened where `rowPiece` holds it: at the same first part. -/
theorem rowOpen_length (r : XR) (j : ℕ) : (rowOpen r j).length = (rowPiece r j).length := by
  unfold rowOpen rowPiece
  apply toList_length_congr
  apply findSome_isSome_congr
  intro s
  simp [Option.isSome_bind, Option.isSome_map]

theorem flatMap_rowLens_one : ∀ l : List (ℕ × List (Msg × List HmNets.Lin)), (∀ p ∈ l, (rowLens p.2).length = 1) →
    (l.flatMap fun p => rowLens p.2).length = l.length
  | [], _ => rfl
  | p :: l, h => by
    rw [List.flatMap_cons, List.length_append, h p List.mem_cons_self,
      flatMap_rowLens_one l fun q hq => h q (List.mem_cons_of_mem p hq), List.length_cons]
    omega

theorem rowOpen_count (r : XR) (i : ℕ) :
    ((rowOpen r i).flatMap fun p => rowLens p.2).length = (rowPiece r i).length := by
  have h1 : ∀ p ∈ rowOpen r i, (rowLens p.2).length = 1 := by
    intro p hp
    obtain ⟨c, nm, q, n, he⟩ := rowOpen_segs hp
    rw [he]
    exact (openSegs_good c nm q n).2
  rw [flatMap_rowLens_one _ h1, rowOpen_length]

theorem capsOpen_count (r : XR) (rs : Array RecVStar.SchedRound) (i : ℕ) :
    (((List.range ((rs.getD (i % rs.size) default).2.1).size).map (capOpenAt r rs i)).flatMap
      fun p => rowLens p.2).length = ((rs.getD (i % rs.size) default).2.1).toList.sum := by
  rw [List.length_flatMap, List.map_map, ← range_map_getD]
  congr 1
  apply List.map_congr_left
  intro e _
  simp only [Function.comp_apply, capOpenAt, capOpen_count, List.length_map, List.length_range]

theorem pieces_len : ∀ ps : List (ℕ × List (Msg × List HmNets.Lin)),
    (∀ p ∈ ps, pieceLen p.2 = sumS (rowLens p.2)) →
    (ps.map fun p => pieceLen p.2).sum = sumS (ps.flatMap fun p => rowLens p.2)
  | [], _ => rfl
  | p :: ps, h => by
    rw [List.map_cons, List.sum_cons, List.flatMap_cons, sumS_append, h p List.mem_cons_self,
      pieces_len ps fun q hq => h q (List.mem_cons_of_mem p hq)]

theorem rounds_len (f : ℕ → List (ℕ × List (Msg × List HmNets.Lin))) : ∀ js : List ℕ,
    (∀ j ∈ js, ∀ p ∈ f j, pieceLen p.2 = sumS (rowLens p.2)) →
    ((js.flatMap f).map fun p => pieceLen p.2).sum = cutLen (js.map fun j => (f j).flatMap fun p => rowLens p.2)
  | [], _ => rfl
  | j :: js, h => by
    rw [List.flatMap_cons, List.map_append, List.sum_append, pieces_len (f j) (h j List.mem_cons_self),
      rounds_len f js fun j' hj => h j' (List.mem_cons_of_mem j hj)]
    simp only [cutLen, List.map_cons, List.sum_cons]

theorem roundOpen_good (r : XR) (rs : Array RecVStar.SchedRound) (j : ℕ) :
    ∀ p ∈ roundOpen r rs j, pieceLen p.2 = sumS (rowLens p.2) := by
  intro p hp
  unfold roundOpen at hp
  rcases List.mem_append.1 hp with hp | hp
  · obtain ⟨e, -, rfl⟩ := List.mem_map.1 hp
    simp only [capOpenAt]
    exact capOpen_good r _ _
  · obtain ⟨c, nm, q, n, he⟩ := rowOpen_segs hp
    rw [he]
    exact (openSegs_good c nm q n).1

theorem readPieces_length (ps : List (ℕ × List (Msg × List HmNets.Lin))) (ws : List Claims.Flock.Wit) :
    (readPieces ps ws).length = (ps.map fun p => pieceLen p.2).sum := by
  unfold readPieces
  rw [List.length_flatMap]
  congr 1
  apply List.map_congr_left
  intro q _
  cases ws[q.1]? with
  | none => simp
  | some w => simp [readSegs_length]

/-! ## `FinalReads` -/

/-- **The last message off the witnesses**: the openings, `elem`'s pieces, then the trailer at `openLens`, whose
lengths sum to the openings' bytes. -/
theorem lastAt_parts (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) {x : Msg}
    (hx : DenseX x) {ws : List Claims.Flock.Wit} (hl : ws.length = nS x) :
    ∃ r, readX x = some r ∧ lastAt (restR elem) x ws = (readPieces (openPieces r) ws ++
      readPieces ((elem r).map fun p => (p.1.val, p.2)) ws) ++ trailer (openLens r) ∧
      cutLen (openLens r) = (readPieces (openPieces r) ws).length := by
  obtain ⟨r, rs, s0, spec, st, hr, hrs, hset, hm, hsc, hk⟩ := schedBridge hx
  have hn : nS x = r.w.parts.length := by simp [nS, hr]
  have hR : layoutOf.rounds x = 2 + 2 * rs.size := hk
  obtain ⟨N, hc, hpos⟩ := capAt_readX hr
  have hN : N = level0Nodes st := by
    rw [capAt_level0 hr hm] at hc
    exact (Prod.mk.inj (Option.some.inj hc)).2.symm
  have hk1 : kI x - 1 = 2 * rs.size + 1 := by omega
  have hsr : streamRound x (2 * rs.size) = none := by
    unfold streamRound
    rw [hr, Option.bind_some, hrs, Option.bind_some, if_neg (by omega)]
  have hf : (layAt (restR elem)).fixedAt x (2 * rs.size + 1) = none := by
    show (streamRound x (2 * rs.size)).bind _ = none
    rw [hsr]
    rfl
  have hp : (layAt (restR elem)).pieceAt x (2 * rs.size + 1) = finOpen (restR elem) x := by
    show stepPieces (finOpen (restR elem)) x (2 * rs.size + 1) = _
    unfold stepPieces
    rw [hR, if_pos (by omega)]
  have hfo : finOpen (restR elem) x = openPieces r ++ (restR elem r).map fun p => (p.1.val, p.2) := by
    simp [finOpen, hr]
  have htp : ((trailerPiece r).map fun p => (p.1.val, p.2)) =
      [(r.w.parts.length - 1, [(trailer (openLens r), ([] : List HmNets.Lin))])] := by
    rw [trailerPiece, dif_pos (show 0 < r.w.parts.length by omega)]
    all_goals rfl
  -- the last message: the openings, `elem`'s pieces, then the trailer
  have hlast : lastAt (restR elem) x ws = (readPieces (openPieces r) ws ++
      readPieces ((elem r).map fun p => (p.1.val, p.2)) ws) ++ trailer (openLens r) := by
    have hw : ws[r.w.parts.length - 1]? = some ws[r.w.parts.length - 1] := List.getElem?_eq_getElem (by omega)
    have h0 : readForms ws[r.w.parts.length - 1] [] = [] :=
      List.eq_nil_of_length_eq_zero (by rw [readForms_length]; rfl)
    have hT : readSegs ws[r.w.parts.length - 1] [(trailer (openLens r), ([] : List HmNets.Lin))] =
        trailer (openLens r) := by
      simp [readSegs, h0]
    have hne : (layAt (restR elem)).pieceAt x (2 * rs.size + 1) ≠ [] := by
      rw [hp, hfo, restR, List.map_append, htp]
      simp
    unfold lastAt
    rw [(layAt (restR elem)).msgs_getD x ws (i := kI x - 1) (by show kI x - 1 < kI x; omega), hk1,
      LayoutB.msg_of_pieces hf hne, LayoutB.assembled, hp, hfo, restR, List.map_append, htp]
    simp only [readPieces, List.flatMap_append, List.flatMap_cons, List.flatMap_nil, List.append_nil, hw,
      Option.map_some, Option.getD_some, hT, List.append_assoc]
  -- the trailer's lengths, and the openings' bytes
  have hop : openPieces r = (nS x - 1, capOpen r (nS x - 1) (List.range N)) ::
      (List.range (2 * rs.size)).flatMap (roundOpen r rs) := by
    simp [openPieces, hc, hrs]
  have hol : openLens r = rowLens (capOpen r (nS x - 1) (List.range N)) ::
      (List.range (2 * rs.size)).map fun j => (roundOpen r rs j).flatMap fun p => rowLens p.2 := by
    simp [openLens, hc, hrs]
  have hlen : cutLen (openLens r) = ((openPieces r).map fun p => pieceLen p.2).sum := by
    rw [hop, hol, List.map_cons, List.sum_cons,
      rounds_len _ _ fun j _ => roundOpen_good r rs j]
    simp only [cutLen, List.map_cons, List.sum_cons, capOpen_good]
  exact ⟨r, hr, hlast, by rw [readPieces_length, hlen]⟩

/-- **`finalR` reads `elem`'s pieces past the openings.** -/
theorem finalR_last (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) {x : Msg}
    (hx : DenseX x) {ws : List Claims.Flock.Wit} (hl : ws.length = nS x) {os : List (List (Msg × Salt))} {last : Msg}
    (hf : finalR (lastAt (restR elem) x ws) = some (os, last)) :
    ∃ r, readX x = some r ∧ last = readPieces ((elem r).map fun p => (p.1.val, p.2)) ws := by
  obtain ⟨r, hr, hlast, hlen⟩ := lastAt_parts elem hx hl
  refine ⟨r, hr, ?_⟩
  rw [hlast, finalR_trailer] at hf
  rw [cutRounds_rest _ _ hf, hlen, List.drop_left]

/-- **`final` reads the last message at every witnesses**, at the deployed `finalR` and the last round's pieces
`restR elem`, for every `elem`. -/
theorem finalReads (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) :
    FinalReads finalR (restR elem) := by
  intro x hx ws hl
  obtain ⟨r, rs, s0, spec, st, hr, hrs, hset, hm, hsc, hk⟩ := schedBridge hx
  obtain ⟨r', hr', hlast, hlen⟩ := lastAt_parts elem hx hl
  rw [hr] at hr'
  have e : r = r' := Option.some.inj hr'
  subst e
  have hn : nS x = r.w.parts.length := by simp [nS, hr]
  obtain ⟨N, hc, hpos⟩ := capAt_readX hr
  have hN : N = level0Nodes st := by
    rw [capAt_level0 hr hm] at hc
    exact (Prod.mk.inj (Option.some.inj hc)).2.symm
  have hk1 : kI x - 1 = 2 * rs.size + 1 := by omega
  have hol : openLens r = rowLens (capOpen r (nS x - 1) (List.range N)) ::
      (List.range (2 * rs.size)).map fun j => (roundOpen r rs j).flatMap fun p => rowLens p.2 := by
    simp [openLens, hc, hrs]
  obtain ⟨os, last, hcut, hos, hoj⟩ := cutRounds_ok (openLens r)
    (readPieces (openPieces r) ws ++ readPieces ((elem r).map fun p => (p.1.val, p.2)) ws)
    (by rw [List.length_append, ← hlen]; omega)
  have hosl : os.length = 2 * rs.size + 1 := by
    rw [hos, hol, List.length_cons, List.length_map, List.length_range]
  refine ⟨os, last, ?_, ?_, ?_⟩
  · rw [hlast, finalR_trailer, hcut]
  · show os.length + 1 = kI x
    omega
  · intro j hj
    rw [hoj j, hol]
    cases j with
    | zero =>
      rw [List.getD_cons_zero, capOpen_count, List.length_range, nAtSched_zero hset, hN]
    | succ i =>
      have hi2 : i < 2 * rs.size := by omega
      have hsi : streamRound x i = some (rs.getD (i % rs.size) default) := by
        unfold streamRound
        rw [hr, Option.bind_some, hrs, Option.bind_some, if_pos hi2]
      rw [List.getD_cons_succ, List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range hi2,
        Option.map_some, Option.getD_some, roundOpen, List.flatMap_append, List.length_append, capsOpen_count,
        rowOpen_count, nAtSched_succ hset hsc hi2]
      by_cases hA : (rs.getD (i % rs.size) default).2.2.2 = true
      · obtain ⟨s, hs, h, hh, hi⟩ := rowsCoverAll hr hsi hA
        rw [rowPiece_length_one (by omega) hh hi, if_pos hA]
      · have hnr : ∀ s, ∀ h ∈ heldRows r s, h.1 ≠ i := by
          intro s h hh hi
          have := heldRows_absorbs hh hrs
          rw [hi] at this
          exact hA this
        rw [rowPiece_nil hnr, if_neg hA, List.length_nil]

end RecDeploy.RestRead
