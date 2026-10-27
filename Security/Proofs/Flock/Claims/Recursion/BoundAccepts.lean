import Proofs.Flock.Claims.Recursion.BoundRun

/-!
# The inner view accepts, at V*'s bound layout (option (d))

At the compiled record whose readers are V*'s bound layout's (`LayoutB.readers`), an accepted run's inner view is
accepted, or two of the parts' recorded calls collide (`RecDeploy.vstarAccepts_bound`, `VStarAccepts` at each shape).
Each part's inner check holds on the readers' messages, at the run's coins:

* the staging off the run (`VStar.RunPairs`, `VStarB.VstarShowsB`) makes each part's statement show the run's coins and
  the run's commit strings;
* at a round a part holds whole, it holds that commit string (`VStarB.PartOpensB`), and the readers' message there is the
  run's (`RecDeploy.ReadersRunPlayed`); the last round is never held whole (`VStarB.HeldB`);
* at an assembled round, each of the part's pieces sits in the readers' message at its offset
  (`VStarB.LayoutB.assembled_piece`).

So `VStarB.PartChecksB` gives each part's check on the readers' messages, and `VStarB.DecompB` gives the inner verifier's
acceptance. With `Refuses` and `ReadersTrue`, the three shapes give `InnerAcceptsColl` (`RecDeploy.innerAcceptsColl_bound`).
-/

namespace Claims.Recursion.VStarB

open Claims
open Claims.Recursion.VStar (readAt)

theorem readAt_length (w : Flock.Wit) (ps : List ℕ) : (readAt w ps).length = (ps.length + 7) / 8 := by
  simp [readAt, VStar.bytesOf]

theorem readForms_length (w : Flock.Wit) (fs : List _root_.Flock.HmNets.Lin) :
    (readForms w fs).length = (fs.length + 7) / 8 := by
  simp [readForms, VStar.bytesOf]

/-- A piece's bytes have the length its segments fix. -/
theorem readSegs_length (w : Flock.Wit) (segs : List (Msg × List _root_.Flock.HmNets.Lin)) :
    (readSegs w segs).length = pieceLen segs := by
  simp [readSegs, pieceLen, List.length_flatMap, readForms_length]

theorem mem_foldl_pieces (w : Flock.Wit) (s : ℕ) :
    ∀ (l : List (ℕ × List (Msg × List _root_.Flock.HmNets.Lin))) (o₀ : ℕ) (acc : List (ℕ × Msg))
      (e : ℕ × Msg),
      e ∈ (l.foldl (fun (acc : ℕ × List (ℕ × Msg)) p =>
        (acc.1 + pieceLen p.2, if p.1 = s then acc.2 ++ [(acc.1, readSegs w p.2)] else acc.2)) (o₀, acc)).2 →
      e ∈ acc ∨ ∃ pre p post, l = pre ++ p :: post ∧ p.1 = s ∧
        e = (o₀ + (pre.map fun q => pieceLen q.2).sum, readSegs w p.2)
  | [], _, _, _, h => Or.inl h
  | q :: l, o₀, acc, e, h => by
    rw [List.foldl_cons] at h
    rcases mem_foldl_pieces w s l (o₀ + pieceLen q.2)
        (if q.1 = s then acc ++ [(o₀, readSegs w q.2)] else acc) e h with h | ⟨pre, p, post, hl, hp, he⟩
    · split_ifs at h with hq
      · rcases List.mem_append.1 h with h | h
        · exact Or.inl h
        · refine Or.inr ⟨[], q, l, rfl, hq, ?_⟩
          rw [List.mem_singleton] at h
          simp [h]
      · exact Or.inl h
    · refine Or.inr ⟨q :: pre, p, post, by rw [hl]; rfl, hp, ?_⟩
      rw [he]
      simp [Nat.add_assoc]

/-- **Each of part `s`'s pieces of a round is one of the round's pieces**, its offset the pieces' lengths before it. -/
theorem LayoutB.mem_piecesOf {L : LayoutB} {x : Msg} {s : ℕ} {w : Flock.Wit} {i : ℕ} {e : ℕ × Msg}
    (h : e ∈ L.piecesOf x s w i) :
    ∃ pre p post, L.pieceAt x i = pre ++ p :: post ∧ p.1 = s ∧
      e = ((pre.map fun q => pieceLen q.2).sum, readSegs w p.2) := by
  rcases mem_foldl_pieces w s (L.pieceAt x i) 0 [] e h with h | ⟨pre, p, post, hl, hp, he⟩
  · simp at h
  · exact ⟨pre, p, post, hl, hp, by rw [he, Nat.zero_add]⟩

/-- **An assembled round holds each piece at its offset**, read off its part's witness. -/
theorem LayoutB.assembled_piece {L : LayoutB} {x : Msg} {ws : List Flock.Wit} {i s : ℕ} (hs : s < ws.length)
    {pre post : List (ℕ × List (Msg × List _root_.Flock.HmNets.Lin))}
    {p : ℕ × List (Msg × List _root_.Flock.HmNets.Lin)}
    (hl : L.pieceAt x i = pre ++ p :: post) (hp : p.1 = s) :
    ((L.assembled x ws i).drop (pre.map fun q => pieceLen q.2).sum).take (readSegs ws[s] p.2).length =
      readSegs ws[s] p.2 := by
  have hg : ∀ q : ℕ × List (Msg × List _root_.Flock.HmNets.Lin),
      ((ws[q.1]?.map fun w => readSegs w q.2).getD (List.replicate (pieceLen q.2) 0)).length = pieceLen q.2 := by
    intro q
    cases ws[q.1]? with
    | none => simp
    | some w => simp [readSegs_length]
  have hpre : (pre.flatMap fun q => (ws[q.1]?.map fun w => readSegs w q.2).getD
      (List.replicate (pieceLen q.2) 0)).length = (pre.map fun q => pieceLen q.2).sum := by
    rw [List.length_flatMap]
    simp only [hg]
  have hp' : (ws[p.1]?.map fun w => readSegs w p.2).getD (List.replicate (pieceLen p.2) 0) = readSegs ws[s] p.2 := by
    subst hp
    simp [List.getElem?_eq_getElem hs]
  unfold LayoutB.assembled
  rw [hl, List.flatMap_append, List.flatMap_cons, hp', ← hpre, List.drop_left, List.take_left]

theorem LayoutB.msgs_getD (L : LayoutB) (x : Msg) (ws : List Flock.Wit) {i : ℕ} (hi : i < L.rounds x) :
    (L.msgs x ws).getD i [] = L.msg x ws i := by
  simp [LayoutB.msgs, List.getD_eq_getElem?_getD, hi]

/-- A round with pieces that the statement doesn't fix is read assembled. -/
theorem LayoutB.msg_of_pieces {L : LayoutB} {x : Msg} {ws : List Flock.Wit} {i : ℕ} (hf : L.fixedAt x i = none)
    (hp : L.pieceAt x i ≠ []) : L.msg x ws i = L.assembled x ws i := by
  simp only [LayoutB.msg, hf, List.isEmpty_iff, hp, ↓reduceIte]

end Claims.Recursion.VStarB

namespace RecDeploy

open Claims (Msg Out)
open FlockDeploy (Coin)

/-- **`VStarAccepts` at V\*'s bound layout** (option (d)), at one shape: from the staging off the run (`RunPairs`,
`VstarShowsB`), the layout's rounds (`HeldB`), each part's check and holdings (`PartChecksB`, `PartOpensB`), the
decomposition (`DecompB`) and the readers' rounds before the last (`ReadersRunPlayed`), at an inner verifier that plays
all its rounds on replayed messages (`hplay`, as `IVStar` does). -/
theorem vstarAccepts_bound {W Col T : Type} (H : Msg → Msg) (I : Claims.Recursion.Inner Coin W)
    (ivScope : Msg → Bool) (b : Brings) (L : Claims.Recursion.VStarB.LayoutB) (DO : Claims.Flock.Deploy Coin Col T)
    (shape : Claims.Recursion.Form → Prop) (decC : Msg → Msg → Coin)
    (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop)
    (cmt : Msg → ℕ → Msg → Msg)
    (hplay : ∀ x ms (c : Fin (I.V.k (xIOf x)) → Coin),
      (Claims.run H (Claims.replay ms) I.V (xIOf x) () c).msgs.length = I.V.k (xIOf x))
    (hRP : Claims.Recursion.VStar.RunPairs H (deployRec I.V ivScope { b with rd := L.readers }) I decC)
    (hHeld : Claims.Recursion.VStarB.HeldB (deployRec I.V ivScope { b with rd := L.readers }) I L shape)
    (hShows : Claims.Recursion.VStarB.VstarShowsB (deployRec I.V ivScope { b with rd := L.readers }) I L shape decC
      shows cmt)
    (hChk : ∀ s, Claims.Recursion.VStarB.PartChecksB H (deployRec I.V ivScope { b with rd := L.readers }) DO L shape
      shows cmt chk s)
    (hOpen : ∀ s, Claims.Recursion.VStarB.PartOpensB H (deployRec I.V ivScope { b with rd := L.readers }) DO L shape
      shows cmt s)
    (hDec : Claims.Recursion.VStarB.DecompB H (deployRec I.V ivScope { b with rd := L.readers }) I.V L shape chk)
    (hRows : L.ReadsRows H)
    (hRRP : ReadersRunPlayed H I.V { b with rd := L.readers } DO) :
    Claims.Recursion.VStarAccepts H (deployRec I.V ivScope { b with rd := L.readers }) I DO shape := by
  intro x st hst hf Ω P ω r hacc ws ho htrue
  obtain ⟨hlenR, hpr⟩ := hRP x Ω P ω r hacc
  obtain ⟨hK, hcl, hlast⟩ := hHeld x st hst hf
  change I.V.k (xIOf x) + kS x ≤ _ at hlenR
  change I.V.k (xIOf x) = L.rounds x at hK
  have hpl : ((Claims.run H P (deployRec I.V ivScope { b with rd := L.readers }).V x ω r).msgs.take
      (I.V.k (xIOf x) + kS x)).length = I.V.k (xIOf x) + kS x := by
    simp only [List.length_take]
    omega
  have hc : ∀ i < ((Claims.run H P (deployRec I.V ivScope { b with rd := L.readers }).V x ω r).msgs.take
      (I.V.k (xIOf x) + kS x)).length, ∃ m,
      (((Claims.run H P (deployRec I.V ivScope { b with rd := L.readers }).V x ω r).msgs.take
        (I.V.k (xIOf x) + kS x)).getD i ([], .rej)).2 = .msg m ∧ decC x m = Claims.coinSeq r i := by
    intro i hi
    rw [hpl] at hi
    obtain ⟨m, hm, hd⟩ := hpr i hi
    refine ⟨m, ?_, hd⟩
    rwa [List.getD_eq_getElem?_getD, List.getElem?_take_of_lt hi, ← List.getD_eq_getElem?_getD]
  have hsh := hShows x st hst hf _ hpl (Claims.coinSeq r) hc
  -- The readers' rounds before the last are the run's.
  have hlA := runAs_length H I.V { b with rd := L.readers } x P ω r hacc
  have hcA : (runCs (I.V.k (xIOf x) + kS x) r).length = _ :=
    (runCs_length I.V { b with rd := L.readers } x r).trans hlA.symm
  have hoA := ho
  rw [runVstar_staged H I ivScope { b with rd := L.readers } P ω r hacc] at hoA
  rcases hRRP x _ _ ws hlA hcA (stagedAll_run H I.V { b with rd := L.readers } x P ω r hacc) hoA htrue with
    hag | hcoll
  swap
  · exact Or.inr hcoll
  have hrd : ∀ j, j + 1 < I.V.k (xIOf x) → (L.msgs x ws).getD j [] =
      (((Claims.run H P (deployRec I.V ivScope { b with rd := L.readers }).V x ω r).msgs.take
        (I.V.k (xIOf x) + kS x)).map Prod.fst).getD j [] := by
    intro j hj
    have hj' : j < (Claims.run H (Claims.replay (L.msgs x ws)) I.V (xIOf x) ()
        (fun i => (runCs (I.V.k (xIOf x) + kS x) r).getD i.val default)).msgs.length := by
      rw [hplay]
      omega
    have h1 : ((Claims.run H (Claims.replay (L.msgs x ws)) I.V (xIOf x) ()
        (fun i => (runCs (I.V.k (xIOf x) + kS x) r).getD i.val default)).msgs.map Prod.fst).getD j [] = _ :=
      hag j hj' hj
    rw [Claims.Recursion.VStar.run_replay_fst H I.V (xIOf x) (L.msgs x ws) _ hj'] at h1
    exact h1
  -- A part whose own calls collide gives the collision.
  by_cases hpc : ∃ s, ∃ hs : s < ws.length, (Claims.findColl (L.calls x s ws[s])).isSome
  · obtain ⟨s, hs, h⟩ := hpc
    exact Or.inr (Claims.Recursion.VStar.coll_of_part (calls := L.calls) hs htrue h)
  have hpc' : ∀ s (hs : s < ws.length), ¬ (Claims.findColl (L.calls x s ws[s])).isSome :=
    fun s hs h => hpc ⟨s, hs, h⟩
  -- Every part's check holds on the readers' messages, at the run's coins.
  have hchk : ∀ s < (deployRec I.V ivScope { b with rd := L.readers }).nS x,
      chk s x (Claims.coinSeq r) (L.msgs x ws) := by
    intro s hs
    have hsw : s < ws.length := ho.1 ▸ hs
    have hts : ∀ e ∈ L.calls x s ws[s], H e.1 = e.2 :=
      fun e he => htrue e (Claims.Recursion.VStar.mem_calls_of_part hsw he)
    have hshs := hsh s hs
    rcases hOpen s x st hst hf _ ws[s] (ho.2 s hsw) hts with hop | hop
    swap
    · exact absurd hop (hpc' s hsw)
    rcases hChk s x st hst hf _ ws[s] (ho.2 s hsw) hts with hck | hck
    swap
    · exact absurd hck (hpc' s hsw)
    refine hck _ hshs.1 _ (fun i hi m hm => ?_) (fun i hi o m hm => ?_) ?_ ?_
    · have hi1 : i + 1 < L.rounds x := by
        by_contra hne
        have hil : i = I.V.k (xIOf x) - 1 := by omega
        have hn := hlast s
        change L.msgAt x s (I.V.k (xIOf x) - 1) = none at hn
        rw [← hil] at hn
        simp [Claims.Recursion.VStarB.LayoutB.held, hn] at hm
      exact (hrd i (by omega)).trans ((hshs.2 i hi1).symm.trans ((hop _ hshs.1 i hi1).1 m hm).symm)
    · obtain ⟨pre, p, post, hl, hp, he⟩ := Claims.Recursion.VStarB.LayoutB.mem_piecesOf hm
      have hne : L.pieceAt x i ≠ [] := by
        rw [hl]
        simp
      have hfx : L.fixedAt x i = none := by
        rcases hfi : L.fixedAt x i with _ | m'
        · rfl
        · exact absurd ((hcl i (by change i < I.V.k (xIOf x); omega)).2 (by simp [hfi])) hne
      obtain ⟨rfl, rfl⟩ := Prod.mk.inj he
      rw [Claims.Recursion.VStarB.LayoutB.msgs_getD L x ws hi,
        Claims.Recursion.VStarB.LayoutB.msg_of_pieces hfx hne]
      exact Claims.Recursion.VStarB.LayoutB.assembled_piece hsw hl hp
    · exact Claims.Recursion.VStarB.LayoutB.cmtAgrees_of_eq fun i hi1 _ =>
        (hrd i (by omega)).trans (hshs.2 i hi1).symm
    · exact hRows x ws htrue s ws[s] (List.getElem?_eq_getElem hsw)
  rcases hDec x st hst hf (Claims.coinSeq r) ws ho.1 htrue hchk with hd | hd
  · exact Or.inl hd
  · exact Or.inr (Claims.Recursion.VStar.coll_of_sub htrue (fun _ h => h) hd)

/-- **`InnerAcceptsColl` at V\*'s bound layout**: `VStarAccepts` at the three shapes, with `Refuses` and `ReadersTrue`. -/
theorem innerAcceptsColl_bound {W Col T : Type} (H : Msg → Msg) (I : Claims.Recursion.Inner Coin W)
    (ivScope : Msg → Bool) (b : Brings) (L : Claims.Recursion.VStarB.LayoutB) (DO : Claims.Flock.Deploy Coin Col T)
    (hT : ReadersTrue H I.V { b with rd := L.readers } DO)
    (hD : Claims.Recursion.VStarAccepts H (deployRec I.V ivScope { b with rd := L.readers }) I DO
      Claims.Recursion.Form.Dense)
    (hH : Claims.Recursion.VStarAccepts H (deployRec I.V ivScope { b with rd := L.readers }) I DO
      Claims.Recursion.Form.Holo)
    (hS : Claims.Recursion.VStarAccepts H (deployRec I.V ivScope { b with rd := L.readers }) I DO
      Claims.Recursion.Form.Sparse) :
    Claims.Recursion.InnerAcceptsColl H (deployRec I.V ivScope { b with rd := L.readers }) I DO :=
  Claims.Recursion.innerAcceptsColl_of_vstar (refuses I.V ivScope _ H) (callsTrue H I ivScope _ DO hT) hD hH hS

end RecDeploy
