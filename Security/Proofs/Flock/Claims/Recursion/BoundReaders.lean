import Proofs.Flock.Claims.Recursion.BoundBridge
import Proofs.Flock.Claims.Recursion.VCallsAt

/-!
# `KSBoundRoute` without `ReadersRunPlayed` (option (d))

`KSBoundRoute` takes three facts about V*'s readers at the bound layout: `ReadersTrue`, `ReadersLogColl` and
`ReadersRunPlayed`. The route reads `ReadersRunPlayed` only on accepted runs whose `E_O` gives witnesses, at the rounds
before the last, and there it follows from the part facts the route already takes, at one fact about the layout
(`LayoutB.HeldWhole`: a round some part holds whole has no pieces):

* a round the statement fixes is the statement's message (`LayoutB.pick`), the same at every run;
* a round held whole by some part (`VStarB.HeldB`) is read as the first holder holds it, the commit string its statement
  shows (`VStarB.PartOpensB`), the run's (`StagedRuns`);
* a round assembled from pieces is read as the pieces' bytes, each sitting at its offset in the commit string its
  part's statement shows (`PartOpensB` again), so the readers' message is the run's first bytes the pieces span;
* or two of one part's recorded calls collide (`RecDeploy.readers_run`).

`ReadersRunPlayed` itself asks more: at every staged message list in the outer relation, run or not, and at fixed rounds
too, where the readers' message is the statement's and not the run's. So it is not what the route needs, and the route
version here drops it (`RecDeploy.ksBoundRoute_heldWhole`), with `ksBoundRoute`'s statement unchanged. The forks
bind as before (`RecDeploy.forksBind_nf_pick`): two forks agree on the run's messages before the fork, so on `pick` of
them.

The staging is read only at accepted runs (`StagedRuns`), not at every staging as `VStarB.VstarShowsB` asks, which with
`HeldB` leaves the deployed compiled verifier no accepted run at a statement with a part; so `VStar.RunPairs` isn't
needed either.

`ReadersTrue` and `ReadersLogColl` stay premises: each is a fact about V*'s circuits (the SHA-512 evaluations the parts
record are true ones, and they are `IVStar`'s calls), which no layout here pins.
-/

namespace Claims.Recursion.VStarB

open Claims

/-- **Only the last round is assembled from pieces**: every round before the last has none (a fact about the layout,
by construction where the layout gives pieces only at the last round). -/
def LayoutB.WholeBeforeLast (L : LayoutB) : Prop :=
  ∀ x i, i + 1 < L.rounds x → L.pieceAt x i = []

/-- **A round some part holds whole has no pieces**, before the last (a fact about the layout). -/
def LayoutB.HeldWhole (L : LayoutB) : Prop :=
  ∀ x s i, i + 1 < L.rounds x → (L.msgAt x s i).isSome → L.pieceAt x i = []

theorem LayoutB.WholeBeforeLast.heldWhole {L : LayoutB} (h : L.WholeBeforeLast) : L.HeldWhole :=
  fun x _ i hi _ => h x i hi

/-- **The readers' message at a round, given the run's**: the statement's where it fixes the round; where the round is
assembled from pieces, the run's first bytes the pieces span; else the run's. -/
def LayoutB.pick (L : LayoutB) (x : Msg) (i : ℕ) (m : Msg) : Msg :=
  (L.fixedAt x i).getD (if (L.pieceAt x i).isEmpty then m else m.take (L.piecesLen x i))

/-- `CmtAgrees` where the readers' message at each round it asks about is `pick` of the commit string. -/
theorem LayoutB.cmtAgrees_of_pick {L : LayoutB} {cmt : Msg → ℕ → Msg → Msg} {x xO : Msg} {ms : List Msg}
    (h : ∀ i, i + 1 < L.rounds x → L.fixedAt x i = none → ms.getD i [] = L.pick x i (cmt x i xO)) :
    L.CmtAgrees cmt x xO ms := by
  intro i hi hf
  rw [h i hi hf]
  unfold LayoutB.pick
  rw [hf, Option.getD_none]
  refine ⟨fun hp => by simp [hp], ?_⟩
  by_cases hp : L.pieceAt x i = []
  · simp [hp]
  · have hne : (L.pieceAt x i).isEmpty = false := by simpa using hp
    simp only [hne, Bool.false_eq_true, ↓reduceIte, List.take_take, Nat.min_self]

theorem flatMap_take_of_slices {α β : Type} (len : α → ℕ) (g : α → List β) (M : List β) :
    ∀ (l : List α) (o : ℕ), (∀ a ∈ l, (g a).length = len a) →
      (∀ pre a post, l = pre ++ a :: post → ((M.drop (o + (pre.map len).sum)).take (len a)) = g a) →
        l.flatMap g = (M.drop o).take (l.map len).sum
  | [], _, _, _ => by simp
  | a :: l, o, hg, hM => by
    have h0 := hM [] a l rfl
    have ih := flatMap_take_of_slices len g M l (o + len a) (fun b hb => hg b (List.mem_cons_of_mem _ hb))
      fun pre b post h => by
        have := hM (a :: pre) b post (by rw [h]; rfl)
        simpa [Nat.add_assoc] using this
    simp only [List.map_nil, List.sum_nil, Nat.add_zero] at h0
    rw [List.flatMap_cons, List.map_cons, List.sum_cons, List.take_add, ih, h0, List.drop_drop]

/-- **A round assembled from pieces that each sit in `M` at their offsets is `M`'s first bytes they span**, where every
piece's part has a witness. -/
theorem LayoutB.assembled_eq_take {L : LayoutB} {x : Msg} {ws : List Flock.Wit} {i : ℕ} {M : Msg}
    (hp : ∀ p ∈ L.pieceAt x i, p.1 < ws.length)
    (hM : ∀ pre p post, L.pieceAt x i = pre ++ p :: post → ∀ hs : p.1 < ws.length,
      (M.drop (pre.map fun q => pieceLen q.2).sum).take (pieceLen p.2) = readSegs ws[p.1] p.2) :
    L.assembled x ws i = M.take (L.piecesLen x i) := by
  unfold LayoutB.assembled LayoutB.piecesLen
  rw [flatMap_take_of_slices (fun q => pieceLen q.2) _ M (L.pieceAt x i) 0 ?_ ?_, List.drop_zero]
  · intro q _
    cases ws[q.1]? with
    | none => simp
    | some w => simp [readSegs_length]
  · intro pre q post hl
    have hs := hp q (by rw [hl]; simp)
    rw [Nat.zero_add, hM pre q post hl hs, List.getElem?_eq_getElem hs]
    rfl

/-- **A round held whole is read as its holders hold it**: at a round the statement doesn't fix, with no pieces and some
holder, where every holder holds `M`, the readers' message is `M`. -/
theorem LayoutB.msg_held {L : LayoutB} {x : Msg} {ws : List Flock.Wit} {i : ℕ} {M : Msg}
    (hf : L.fixedAt x i = none) (hp : L.pieceAt x i = [])
    (hex : ∃ s < ws.length, (L.msgAt x s i).isSome)
    (hall : ∀ s (hs : s < ws.length) m, L.held x s ws[s] i = some m → m = M) : L.msg x ws i = M := by
  have hfs : ((List.range ws.length).findSome? fun s => ws[s]?.bind fun w => L.held x s w i) = some M := by
    rcases h : (List.range ws.length).findSome? (fun s => ws[s]?.bind fun w => L.held x s w i) with _ | m
    · exfalso
      obtain ⟨s, hs, hm⟩ := hex
      have hn := List.findSome?_eq_none_iff.1 h s (List.mem_range.2 hs)
      simp [List.getElem?_eq_getElem hs, LayoutB.held, hf] at hn
      simp [hn] at hm
    · obtain ⟨s, hs, hm⟩ := List.exists_of_findSome?_eq_some h
      rw [List.mem_range] at hs
      simp only [List.getElem?_eq_getElem hs, Option.bind_some] at hm
      rw [hall s hs m hm]
  simp only [LayoutB.msg, hf, hp, List.isEmpty_nil, ↓reduceIte, hfs, Option.getD_some]

end Claims.Recursion.VStarB

namespace RecDeploy

open Claims (Msg Out Prover run coinSeq replay findColl Collision)
open FlockDeploy (Coin)
open Claims.Recursion (Bound Inner)

/-- **The staged V\* statements show the accepted run**: on an accepted run of a compiled statement of the shape, each
V* statement shows the run's coins (`shows`) and, as the commit string of each inner round before the last (`cmt`), the
run's message at that round. -/
def StagedRuns {W : Type} (H : Msg → Msg) (D : Claims.Recursion.Deploy Coin) (I : Claims.Recursion.Inner Coin W)
    (L : Claims.Recursion.VStarB.LayoutB) (shape : Claims.Recursion.Form → Prop)
    (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) (cmt : Msg → ℕ → Msg → Msg) : Prop :=
  ∀ (x : Msg) (st : Claims.Recursion.CompiledStatement), D.parse x = some st → shape st.form →
    ∀ (Ω : Type) (P : Prover Ω) (ω : Ω) (r : Fin (D.V.k x) → Coin), (run H P D.V x ω r).out = .acc →
      ∀ s < D.nS x, shows s x ((Claims.Recursion.runVstar H D I P x ω r).getD s []) (coinSeq r) ∧
        ∀ i, i + 1 < L.rounds x →
          cmt x i ((Claims.Recursion.runVstar H D I P x ω r).getD s []) =
            ((run H P D.V x ω r).msgs.map Prod.fst).getD i []

/-- **The readers' rounds before the last are the run's, read through `pick`**, at one shape: at a statement of the
shape whose V* statements show the run (`hsh`, from `StagedRuns` at an accepted run), at outer witnesses in the outer
relation there whose recorded calls are true evaluations, each round before the last is the statement's where it fixes
it, the run's message's first bytes its pieces span where it has pieces, else the run's message; or two of one part's
recorded calls collide. From the layout's rounds (`HeldB`, `HeldWhole`) and the parts' holdings and pieces
(`PartOpensB`). -/
theorem readers_run {W Col T : Type} (H : Msg → Msg) (I : Claims.Recursion.Inner Coin W)
    (ivScope : Msg → Bool) (b : Brings) (L : Claims.Recursion.VStarB.LayoutB) (DO : Claims.Flock.Deploy Coin Col T)
    (shape : Claims.Recursion.Form → Prop)
    (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) (cmt : Msg → ℕ → Msg → Msg) (hHW : L.HeldWhole)
    (hHeld : Claims.Recursion.VStarB.HeldB (deployRec I.V ivScope { b with rd := L.readers }) I L shape)
    (hOpen : ∀ s, Claims.Recursion.VStarB.PartOpensB H (deployRec I.V ivScope { b with rd := L.readers }) DO L shape
      shows cmt s)
    {x : Msg} {st : Claims.Recursion.CompiledStatement}
    (hst : (deployRec I.V ivScope { b with rd := L.readers }).parse x = some st) (hf : shape st.form)
    {Ω : Type} {P : Prover Ω} {ω : Ω} {r : Fin ((deployRec I.V ivScope { b with rd := L.readers }).V.k x) → Coin}
    (hsh : ∀ s < (deployRec I.V ivScope { b with rd := L.readers }).nS x,
      shows s x ((Claims.Recursion.runVstar H (deployRec I.V ivScope { b with rd := L.readers }) I P x ω r).getD s [])
        (coinSeq r) ∧
      ∀ i, i + 1 < L.rounds x →
        cmt x i ((Claims.Recursion.runVstar H (deployRec I.V ivScope { b with rd := L.readers }) I P x ω r).getD s []) =
          ((run H P (deployRec I.V ivScope { b with rd := L.readers }).V x ω r).msgs.map Prod.fst).getD i [])
    {ws : List Claims.Flock.Wit}
    (ho : Claims.Recursion.OuterHolds DO
      (Claims.Recursion.runVstar H (deployRec I.V ivScope { b with rd := L.readers }) I P x ω r)
      ((deployRec I.V ivScope { b with rd := L.readers }).nS x) ws)
    (htrue : ∀ e ∈ (deployRec I.V ivScope { b with rd := L.readers }).calls x ws, H e.1 = e.2) :
    (∀ j, j + 1 < I.V.k (xIOf x) → (L.msgs x ws).getD j [] =
        L.pick x j (((run H P (deployRec I.V ivScope { b with rd := L.readers }).V x ω r).msgs.map Prod.fst).getD
          j [])) ∨
      ∃ u v, findColl ((deployRec I.V ivScope { b with rd := L.readers }).calls x ws) = some (u, v) ∧
        Collision H u v := by
  obtain ⟨hK, hcl, -⟩ := hHeld x st hst hf
  change I.V.k (xIOf x) = L.rounds x at hK
  by_cases hpc : ∃ s, ∃ hs : s < ws.length, (Claims.findColl (L.calls x s ws[s])).isSome
  · obtain ⟨s, hs, h⟩ := hpc
    exact Or.inr (Claims.Recursion.VStar.coll_of_part (calls := L.calls) hs htrue h)
  left
  intro j hj
  have hjr : j < L.rounds x := by omega
  have hj1 : j + 1 < L.rounds x := by omega
  rw [Claims.Recursion.VStarB.LayoutB.msgs_getD L x ws hjr]
  have hopen : ∀ s (hs : s < ws.length), ∀ (c : ℕ → Coin),
      shows s x ((Claims.Recursion.runVstar H (deployRec I.V ivScope { b with rd := L.readers }) I P x ω r).getD s [])
        c → ∀ i, i + 1 < L.rounds x →
      (∀ m, L.held x s ws[s] i = some m →
        m = cmt x i ((Claims.Recursion.runVstar H (deployRec I.V ivScope { b with rd := L.readers }) I P x ω r).getD
          s [])) ∧
      ∀ pre p post, L.pieceAt x i = pre ++ p :: post → p.1 = s →
        ((cmt x i ((Claims.Recursion.runVstar H (deployRec I.V ivScope { b with rd := L.readers }) I P x ω r).getD
          s [])).drop (pre.map fun q => Claims.Recursion.VStarB.pieceLen q.2).sum).take
            (Claims.Recursion.VStarB.pieceLen p.2) = Claims.Recursion.VStarB.readSegs ws[s] p.2 := by
    intro s hs
    have hts : ∀ e ∈ L.calls x s ws[s], H e.1 = e.2 :=
      fun e he => htrue e (Claims.Recursion.VStar.mem_calls_of_part hs he)
    rcases hOpen s x st hst hf _ ws[s] (ho.2 s hs) hts with hop | hop
    · exact hop
    · exact absurd ⟨s, hs, hop⟩ hpc
  rcases hfx : L.fixedAt x j with _ | m
  · simp only [Claims.Recursion.VStarB.LayoutB.pick, hfx, Option.getD_none]
    by_cases hpe : L.pieceAt x j = []
    · simp only [hpe, List.isEmpty_nil, ↓reduceIte]
      refine Claims.Recursion.VStarB.LayoutB.msg_held hfx hpe ?_ ?_
      · rcases (hcl j (by change j < I.V.k (xIOf x); omega)).1 with h | ⟨s, hs, h⟩ | ⟨h, -⟩
        · simp [hfx] at h
        · exact ⟨s, lt_of_lt_of_eq hs ho.1.symm, h⟩
        · exact absurd hpe h
      · intro s hs m hm
        have hshs := hsh s (lt_of_lt_of_eq hs ho.1)
        exact ((hopen s hs _ hshs.1 j hj1).1 m hm).trans (hshs.2 j hj1)
    · simp only [List.isEmpty_iff, hpe, ↓reduceIte]
      rw [Claims.Recursion.VStarB.LayoutB.msg_of_pieces hfx hpe]
      have hps : ∀ p ∈ L.pieceAt x j, p.1 < ws.length := by
        rcases (hcl j (by change j < I.V.k (xIOf x); omega)).1 with h | ⟨s, -, h⟩ | ⟨-, h⟩
        · simp [hfx] at h
        · exact absurd (hHW x s j hj1 h) hpe
        · exact fun p hp => lt_of_lt_of_eq (h p hp) ho.1.symm
      refine Claims.Recursion.VStarB.LayoutB.assembled_eq_take hps fun pre p post hl hs => ?_
      have hshs := hsh p.1 (lt_of_lt_of_eq hs ho.1)
      rw [← hshs.2 j hj1]
      exact (hopen p.1 hs _ hshs.1 j hj1).2 pre p post hl rfl
  · simp [Claims.Recursion.VStarB.LayoutB.pick, Claims.Recursion.VStarB.LayoutB.msg, hfx]

/-- **`VStarAccepts` at V\*'s bound layout, without `ReadersRunPlayed`**: `vstarAccepts_bound` with the staging read at
the accepted run (`StagedRuns`) and the readers' rounds before the last off the parts (`readers_run`), at a layout whose
held rounds have no pieces (`HeldWhole`). A round a part holds whole is one the statement doesn't fix and that has no
pieces, so there the readers' message is the run's. -/
theorem vstarAccepts_boundW {W Col T : Type} (H : Msg → Msg) (I : Claims.Recursion.Inner Coin W)
    (ivScope : Msg → Bool) (b : Brings) (L : Claims.Recursion.VStarB.LayoutB) (DO : Claims.Flock.Deploy Coin Col T)
    (shape : Claims.Recursion.Form → Prop)
    (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop)
    (cmt : Msg → ℕ → Msg → Msg) (hHW : L.HeldWhole)
    (hHeld : Claims.Recursion.VStarB.HeldB (deployRec I.V ivScope { b with rd := L.readers }) I L shape)
    (hSt : StagedRuns H (deployRec I.V ivScope { b with rd := L.readers }) I L shape shows cmt)
    (hChk : ∀ s, Claims.Recursion.VStarB.PartChecksB H (deployRec I.V ivScope { b with rd := L.readers }) DO L shape
      shows cmt chk s)
    (hOpen : ∀ s, Claims.Recursion.VStarB.PartOpensB H (deployRec I.V ivScope { b with rd := L.readers }) DO L shape
      shows cmt s)
    (hDec : Claims.Recursion.VStarB.DecompB H (deployRec I.V ivScope { b with rd := L.readers }) I.V L shape chk)
    (hRows : L.ReadsRows H) :
    Claims.Recursion.VStarAccepts H (deployRec I.V ivScope { b with rd := L.readers }) I DO shape := by
  intro x st hst hf Ω P ω r hacc ws ho htrue
  obtain ⟨hK, hcl, hlast⟩ := hHeld x st hst hf
  change I.V.k (xIOf x) = L.rounds x at hK
  have hsh := hSt x st hst hf Ω P ω r hacc
  rcases readers_run H I ivScope b L DO shape shows cmt hHW hHeld hOpen hst hf hsh ho htrue with hrd | hcoll
  swap
  · exact Or.inr hcoll
  by_cases hpc : ∃ s, ∃ hs : s < ws.length, (Claims.findColl (L.calls x s ws[s])).isSome
  · obtain ⟨s, hs, h⟩ := hpc
    exact Or.inr (Claims.Recursion.VStar.coll_of_part (calls := L.calls) hs htrue h)
  have hpc' : ∀ s (hs : s < ws.length), ¬ (Claims.findColl (L.calls x s ws[s])).isSome :=
    fun s hs h => hpc ⟨s, hs, h⟩
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
      have hfx : L.fixedAt x i = none := by
        rcases hfi : L.fixedAt x i with _ | m'
        · rfl
        · simp [Claims.Recursion.VStarB.LayoutB.held, hfi] at hm
      have hpe : L.pieceAt x i = [] := by
        refine hHW x s i hi1 ?_
        rcases h : L.msgAt x s i with _ | a
        · simp [Claims.Recursion.VStarB.LayoutB.held, hfx, h] at hm
        · rfl
      rw [hrd i (by omega)]
      simp only [Claims.Recursion.VStarB.LayoutB.pick, hfx, Option.getD_none, hpe, List.isEmpty_nil, ↓reduceIte]
      exact (hshs.2 i hi1).symm.trans ((hop _ hshs.1 i hi1).1 m hm).symm
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
    · refine Claims.Recursion.VStarB.LayoutB.cmtAgrees_of_pick fun i hi1 _ => ?_
      rw [hrd i (by omega), hshs.2 i hi1]
    · exact hRows x ws htrue s ws[s] (List.getElem?_eq_getElem hsw)
  rcases hDec x st hst hf (Claims.coinSeq r) ws ho.1 htrue hchk with hd | hd
  · exact Or.inl hd
  · exact Or.inr (Claims.Recursion.VStar.coll_of_sub htrue (fun _ h => h) hd)

/-- **`VStarAcceptsV` at V\*'s bound layout, without `ReadersRunPlayed`**: `vstarAccepts_boundW` with each part's
check holding off a collision among its recorded calls and the compiled verifier's own at its V\* statement
(`PartChecksBV`), which are among the run's evaluations (`vcallsAt_sub`). -/
theorem vstarAcceptsV_boundW {W Col T : Type} (H : Msg → Msg) (I : Claims.Recursion.Inner Coin W)
    (ivScope : Msg → Bool) (b : Brings) (L : Claims.Recursion.VStarB.LayoutB) (DO : Claims.Flock.Deploy Coin Col T)
    (shape : Claims.Recursion.Form → Prop)
    (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop)
    (cmt : Msg → ℕ → Msg → Msg) (hHW : L.HeldWhole)
    (hHeld : Claims.Recursion.VStarB.HeldB (deployRec I.V ivScope { b with rd := L.readers }) I L shape)
    (hSt : StagedRuns H (deployRec I.V ivScope { b with rd := L.readers }) I L shape shows cmt)
    (hChk : ∀ s, Claims.Recursion.VStarB.PartChecksBV H (deployRec I.V ivScope { b with rd := L.readers }) DO L shape
      shows cmt chk s)
    (hOpen : ∀ s, Claims.Recursion.VStarB.PartOpensB H (deployRec I.V ivScope { b with rd := L.readers }) DO L shape
      shows cmt s)
    (hDec : Claims.Recursion.VStarB.DecompB H (deployRec I.V ivScope { b with rd := L.readers }) I.V L shape chk)
    (hRows : L.ReadsRows H) :
    Claims.Recursion.VStarAcceptsV H (deployRec I.V ivScope { b with rd := L.readers }) I DO shape := by
  intro x st hst hf Ω P ω r hacc ws ho htrue
  have hS := Claims.Recursion.setupCalls_true H (deployRec I.V ivScope { b with rd := L.readers })
    (Claims.Recursion.runVstar H (deployRec I.V ivScope { b with rd := L.readers }) I P x ω r)
  obtain ⟨hK, hcl, hlast⟩ := hHeld x st hst hf
  change I.V.k (xIOf x) = L.rounds x at hK
  have hsh := hSt x st hst hf Ω P ω r hacc
  rcases readers_run H I ivScope b L DO shape shows cmt hHW hHeld hOpen hst hf hsh ho htrue with hrd | hcoll
  swap
  · exact Or.inr (Claims.Recursion.coll_append htrue hS hcoll)
  by_cases hpc : ∃ s, ∃ hs : s < ws.length, (Claims.findColl (L.calls x s ws[s] ++
      Claims.Recursion.VStarB.vcallsAt H (deployRec I.V ivScope { b with rd := L.readers })
        ((Claims.Recursion.runVstar H (deployRec I.V ivScope { b with rd := L.readers }) I P x ω r).getD s []))).isSome
  · obtain ⟨s, hs, h⟩ := hpc
    have hsv : s < (Claims.Recursion.runVstar H (deployRec I.V ivScope { b with rd := L.readers }) I P x ω r).length :=
      lt_of_lt_of_eq (lt_of_lt_of_eq hs ho.1) (vstar_length _ _ _).symm
    refine Or.inr (Claims.Recursion.VStar.coll_of_sub (fun e he => (List.mem_append.1 he).elim (htrue e) (hS e))
      (fun p hp => ?_) h)
    rcases List.mem_append.1 hp with hp | hp
    · exact List.mem_append_left _ (Claims.Recursion.VStar.mem_calls_of_part hs hp)
    · exact List.mem_append_right _ (Claims.Recursion.VStarB.vcallsAt_sub hsv p hp)
  have hpc' : ∀ s (hs : s < ws.length), ¬ (Claims.findColl (L.calls x s ws[s] ++
      Claims.Recursion.VStarB.vcallsAt H (deployRec I.V ivScope { b with rd := L.readers })
        ((Claims.Recursion.runVstar H (deployRec I.V ivScope { b with rd := L.readers }) I P x ω r).getD s []))).isSome :=
    fun s hs h => hpc ⟨s, hs, h⟩
  have hchk : ∀ s < (deployRec I.V ivScope { b with rd := L.readers }).nS x,
      chk s x (Claims.coinSeq r) (L.msgs x ws) := by
    intro s hs
    have hsw : s < ws.length := ho.1 ▸ hs
    have hts : ∀ e ∈ L.calls x s ws[s], H e.1 = e.2 :=
      fun e he => htrue e (Claims.Recursion.VStar.mem_calls_of_part hsw he)
    have hshs := hsh s hs
    rcases hOpen s x st hst hf _ ws[s] (ho.2 s hsw) hts with hop | hop
    swap
    · exact absurd (Claims.Recursion.findColl_isSome_of_sub H (Claims.Recursion.VStarB.vcallsAt_append_true hts)
        (fun _ he => List.mem_append_left _ he) hop) (hpc' s hsw)
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
      have hfx : L.fixedAt x i = none := by
        rcases hfi : L.fixedAt x i with _ | m'
        · rfl
        · simp [Claims.Recursion.VStarB.LayoutB.held, hfi] at hm
      have hpe : L.pieceAt x i = [] := by
        refine hHW x s i hi1 ?_
        rcases h : L.msgAt x s i with _ | a
        · simp [Claims.Recursion.VStarB.LayoutB.held, hfx, h] at hm
        · rfl
      rw [hrd i (by omega)]
      simp only [Claims.Recursion.VStarB.LayoutB.pick, hfx, Option.getD_none, hpe, List.isEmpty_nil, ↓reduceIte]
      exact (hshs.2 i hi1).symm.trans ((hop _ hshs.1 i hi1).1 m hm).symm
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
    · refine Claims.Recursion.VStarB.LayoutB.cmtAgrees_of_pick fun i hi1 _ => ?_
      rw [hrd i (by omega), hshs.2 i hi1]
    · exact hRows x ws htrue s ws[s] (List.getElem?_eq_getElem hsw)
  rcases hDec x st hst hf (Claims.coinSeq r) ws ho.1 htrue hchk with hd | hd
  · exact Or.inl hd
  · exact Or.inr (Claims.Recursion.VStar.coll_of_sub (fun e he => (List.mem_append.1 he).elim (htrue e) (hS e))
      (fun _ h => List.mem_append_left _ h) hd)

/-- **The readers' rounds before the last are the run's, read through `pick`, off a collision**, where `E_O` gives
witnesses on an accepted run: the run's statement parses (`refuses`), its form is dense, holo or sparse, `E_O`'s
witnesses are in the outer relation (`eoGives_outerHolds`) with true recorded calls (`callsTrue`), and `readers_run`
at that shape. -/
theorem readersRun_pick {W Col T : Type} (H : Msg → Msg) (I : Claims.Recursion.Inner Coin W) (ivScope : Msg → Bool)
    (b : Brings) (L : Claims.Recursion.VStarB.LayoutB) (DO : Claims.Flock.Deploy Coin Col T)
    (hHW : L.HeldWhole)
    (hsh : ∀ shape ∈ [Claims.Recursion.Form.Dense, Claims.Recursion.Form.Holo, Claims.Recursion.Form.Sparse],
      ∃ (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) (cmt : Msg → ℕ → Msg → Msg),
        Claims.Recursion.VStarB.HeldB (deployRec I.V ivScope { b with rd := L.readers }) I L shape ∧
        StagedRuns H (deployRec I.V ivScope { b with rd := L.readers }) I L shape shows cmt ∧
        ∀ s, Claims.Recursion.VStarB.PartOpensB H (deployRec I.V ivScope { b with rd := L.readers }) DO L shape
          shows cmt s)
    (hT : ReadersTrue H I.V { b with rd := L.readers } DO) {decO : List (ℕ × Col) → List T} {x : Msg}
    {Ω : Type} {P : Prover Ω} {ω : Ω} {r : Fin ((deployRec I.V ivScope { b with rd := L.readers }).V.k x) → Coin}
    {c : (Claims.Recursion.EO (deployRec I.V ivScope { b with rd := L.readers }) I DO decO).C x}
    {ws : List Claims.Flock.Wit}
    (hg : Claims.Recursion.EOGives H (deployRec I.V ivScope { b with rd := L.readers }) I DO decO P x ω r c ws) :
    (∀ j, j + 1 < I.V.k (xIOf x) → (L.msgs x ws).getD j [] =
        L.pick x j (((run H P (deployRec I.V ivScope { b with rd := L.readers }).V x ω r).msgs.map Prod.fst).getD
          j [])) ∨
      ∃ u v, findColl ((deployRec I.V ivScope { b with rd := L.readers }).calls x ws) = some (u, v) ∧
        Collision H u v := by
  obtain ⟨st, hst, -⟩ := refuses I.V ivScope { b with rd := L.readers } H x Ω P ω r hg.1
  have ho := Claims.Recursion.eoGives_outerHolds hg
  have ht := callsTrue H I ivScope { b with rd := L.readers } DO hT decO x Ω P ω r c ws hg
  have key : ∀ shape ∈ [Claims.Recursion.Form.Dense, Claims.Recursion.Form.Holo, Claims.Recursion.Form.Sparse],
      shape st.form →
      (∀ j, j + 1 < I.V.k (xIOf x) → (L.msgs x ws).getD j [] =
        L.pick x j (((run H P (deployRec I.V ivScope { b with rd := L.readers }).V x ω r).msgs.map Prod.fst).getD
          j [])) ∨
      ∃ u v, findColl ((deployRec I.V ivScope { b with rd := L.readers }).calls x ws) = some (u, v) ∧
        Collision H u v := by
    intro shape hs hf
    obtain ⟨shows, cmt, hHe, hSt, hOp⟩ := hsh shape hs
    exact readers_run H I ivScope b L DO shape shows cmt hHW hHe hOp hst hf (hSt x st hst hf Ω P ω r hg.1) ho ht
  rcases Claims.Recursion.Form.shape_cases st.form with hf | hf | hf
  · exact key _ (by simp) hf
  · exact key _ (by simp) hf
  · exact key _ (by simp) hf

section Forks

variable {Col T Col' T' : Type} [DecidableEq Col] [DecidableEq Col']

omit [DecidableEq Col'] in
/-- **The inner rounds' forks bind at the normalized record, with the readers' rounds before the last read through a
choice `pk` of the run's** (`forksBind_nf` at `pk x j m = m`): two forks agree on the run's messages before the fork, so
on `pk` of them. The rest is `forksBind_nf`'s. -/
theorem forksBind_nf_pick (H : Msg → Msg) (Bd : Bound Coin Col T) (decI : List (ℕ × Col) → List T)
    (ivScope : Msg → Bool) (b : Brings) (DO : Claims.Flock.Deploy Coin Col' T') (pk : Msg → ℕ → Msg → Msg)
    (hLB : Bd.LastBinds H)
    (hT : Claims.Recursion.CallsTrue H (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) DO)
    (hL : Claims.Recursion.CallsLogColl H (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) DO)
    (hA : Claims.Recursion.InnerAcceptsColl H (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) DO)
    (hP : ∀ (decO : List (ℕ × Col') → List T') (x : Msg) (Ω : Type) (P : Prover Ω) (ω : Ω)
      (r : Fin ((deployRec Bd.IVStar ivScope b).V.k x) → Coin)
      (c : (Claims.Recursion.EO (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) DO decO).C x)
      (ws : List Claims.Flock.Wit),
      Claims.Recursion.EOGives H (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) DO decO P x ω r c
        ws →
      (∀ j, j + 1 < Bd.DE.V.k (xIOf x) → (b.rd.msgs x ws).getD j [] =
          pk x j (((run H P (deployRec Bd.IVStar ivScope b).V x ω r).msgs.map Prod.fst).getD j [])) ∨
        ∃ u v, findColl ((deployRec Bd.IVStar ivScope b).calls x ws) = some (u, v) ∧ Collision H u v) :
    Claims.Recursion.ForksBind H (deployRec Bd.IVStar ivScope (nfBrings Bd b)) (Inner.ofFlock Bd.deploy decI) DO := by
  intro decO x Ω P j t hod
  obtain ⟨ws, ws', hg₁, hg₂, hne⟩ := hod
  have hg₁' : Claims.Recursion.EOGives H (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) DO decO P x
      t.1 t.2.1 t.2.2.2.1 ws := hg₁
  have hg₂' : Claims.Recursion.EOGives H (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) DO decO P x
      t.1 (Claims.Recursion.fork j t.2.1 t.2.2.1) t.2.2.2.2 ws' := hg₂
  have htrue : ∀ p ∈ (deployRec Bd.IVStar ivScope b).calls x ws ++ (deployRec Bd.IVStar ivScope b).calls x ws',
      H p.1 = p.2 := fun p hp =>
    (List.mem_append.1 hp).elim (hT decO x Ω P _ _ _ ws hg₁' p) (hT decO x Ω P _ _ _ ws' hg₂' p)
  have hlift : ∀ l : List (Msg × Msg),
      (∀ p ∈ l, p ∈ (deployRec Bd.IVStar ivScope b).calls x ws ++ (deployRec Bd.IVStar ivScope b).calls x ws') →
      (∃ u v, findColl l = some (u, v) ∧ Collision H u v) →
      Claims.Recursion.forkColl H (deployRec Bd.IVStar ivScope (nfBrings Bd b)) (Inner.ofFlock Bd.deploy decI) DO decO
        P x j t := by
    rintro l hsub ⟨u, v, huv, -⟩
    unfold Claims.Recursion.forkColl
    rw [hg₁.2, hg₂.2]
    exact Claims.Recursion.VStar.coll_of_sub htrue hsub (by simp [huv])
  have hleft := hlift ((deployRec Bd.IVStar ivScope b).calls x ws) fun p hp => List.mem_append_left _ hp
  have hright := hlift ((deployRec Bd.IVStar ivScope b).calls x ws') fun p hp => List.mem_append_right _ hp
  have hv : ∀ (r₀ : Fin ((deployRec Bd.IVStar ivScope (nfBrings Bd b)).V.k x) → Coin) (ws₀ : List Claims.Flock.Wit),
      (((deployRec Bd.IVStar ivScope (nfBrings Bd b)).innerView H x r₀ ws₀).msgs.map Prod.fst).getD j [] =
        if j < Bd.DE.V.k (xIOf x) then (Bd.nfMsgs (xIOf x) (b.rd.msgs x ws₀)).getD j [] else [] :=
    fun r₀ ws₀ => Bd.run_ivstar_fst H (xIOf x) (Bd.nfMsgs (xIOf x) (b.rd.msgs x ws₀)) (fun i => coinSeq r₀ i.val) j
  rw [hv, hv] at hne
  by_cases hjK : j < Bd.DE.V.k (xIOf x)
  swap
  · exact absurd (by simp [hjK]) hne
  simp only [hjK, ↓reduceIte] at hne
  rcases hP decO x Ω P t.1 t.2.1 t.2.2.2.1 ws hg₁' with h₁ | h₁
  swap
  · exact hleft h₁
  rcases hP decO x Ω P t.1 (Claims.Recursion.fork j t.2.1 t.2.2.1) t.2.2.2.2 ws' hg₂' with h₂ | h₂
  swap
  · exact hright h₂
  have hag : ∀ i, i + 1 < Bd.DE.V.k (xIOf x) → i ≤ j → (b.rd.msgs x ws).getD i [] = (b.rd.msgs x ws').getD i [] :=
    fun i hi hij => (h₁ i hi).trans ((congrArg (pk x i) (Claims.Recursion.run_fst_getD H P _ x t.1 i fun i' hi' =>
      Claims.Recursion.fork_agree j t.2.1 t.2.2.1 i' (by omega))).trans (h₂ i hi).symm)
  by_cases hj1 : j + 1 < Bd.DE.V.k (xIOf x)
  · rw [Bd.nfMsgs_getD_lt _ _ hj1, Bd.nfMsgs_getD_lt _ _ hj1] at hne
    exact absurd (hag j hj1 le_rfl) hne
  -- The last round.
  have hk0 : 0 < Bd.DE.V.k (xIOf x) := by omega
  have hjK1 : j = Bd.DE.V.k (xIOf x) - 1 := by omega
  rw [hjK1, Bd.nfMsgs_getD_last _ _ hk0, Bd.nfMsgs_getD_last _ _ hk0] at hne
  rcases hA decO x Ω P t.1 t.2.1 t.2.2.2.1 ws hg₁' with ha₁ | ha₁
  swap
  · exact hleft ha₁
  rcases hA decO x Ω P t.1 (Claims.Recursion.fork j t.2.1 t.2.2.1) t.2.2.2.2 ws' hg₂' with ha₂ | ha₂
  swap
  · exact hright ha₂
  rcases hL decO x Ω P t.1 t.2.1 t.2.2.2.1 ws hg₁' with hl₁ | hl₁
  swap
  · exact hleft hl₁
  rcases hL decO x Ω P t.1 (Claims.Recursion.fork j t.2.1 t.2.2.1) t.2.2.2.2 ws' hg₂' with hl₂ | hl₂
  swap
  · exact hright hl₂
  have hacc : ∀ (r₀ : Fin ((deployRec Bd.IVStar ivScope b).V.k x) → Coin) (ws₀ : List Claims.Flock.Wit),
      ((deployRec Bd.IVStar ivScope b).innerView H x r₀ ws₀).out = .acc →
      (Bd.decAt H (xIOf x) (b.rd.msgs x ws₀) fun i => coinSeq r₀ i.val).1 = .acc := by
    intro r₀ ws₀ h
    change (run H (replay (b.rd.msgs x ws₀)) Bd.IVStar (xIOf x) () fun i => coinSeq r₀ i.val).out = .acc at h
    rw [Bd.run_ivstar H (xIOf x) _ _ hk0] at h
    exact h
  have hlog : ∀ (r₀ : Fin ((deployRec Bd.IVStar ivScope b).V.k x) → Coin) (ws₀ : List Claims.Flock.Wit),
      (∀ e ∈ (Claims.Recursion.innerRun H (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) x r₀
        ws₀).vLog, e ∈ (deployRec Bd.IVStar ivScope b).calls x ws₀) →
      ∀ e ∈ (Bd.decAt H (xIOf x) (b.rd.msgs x ws₀) fun i => coinSeq r₀ i.val).2,
        e ∈ (deployRec Bd.IVStar ivScope b).calls x ws₀ := by
    intro r₀ ws₀ h e he
    apply h
    rw [innerRun_bound, Bd.run_ivstar H (xIOf x) _ _ hk0]
    exact he
  have hacc₁ := hacc _ _ ha₁
  have hacc₂ := hacc _ _ ha₂
  have hlog₁ := hlog _ _ hl₁
  have hlog₂ := hlog _ _ hl₂
  by_cases hf : Bd.final ((b.rd.msgs x ws).getD (Bd.DE.V.k (xIOf x) - 1) []) =
      Bd.final ((b.rd.msgs x ws').getD (Bd.DE.V.k (xIOf x) - 1) [])
  · exfalso
    rcases hs : Bd.final ((b.rd.msgs x ws).getD (Bd.DE.V.k (xIOf x) - 1) []) with _ | p
    · rw [Bd.decAt_none H (xIOf x) _ _ hk0 hs] at hacc₁
      cases hacc₁
    · exact hne (Bd.rep_eq hf (by rw [hs]; rfl))
  obtain ⟨n, hn⟩ : ∃ n, Bd.DE.V.k (xIOf x) = n + 1 := ⟨_, (Nat.succ_pred_eq_of_pos hk0).symm⟩
  have hdl : ∀ {α : Type} (f : ℕ → α), ((List.range (Bd.DE.V.k (xIOf x))).map f).dropLast = (List.range n).map f := by
    intro α f
    rw [hn, List.range_succ, List.map_append, List.map_singleton, List.dropLast_concat]
  have hgl : ∀ ms : List Msg, ((List.range (Bd.DE.V.k (xIOf x))).map fun i => ms.getD i []).getLastD [] =
      ms.getD (Bd.DE.V.k (xIOf x) - 1) [] := by
    intro ms
    rw [hn, List.range_succ, List.map_append, List.map_singleton, List.getLastD_concat, Nat.add_sub_cancel]
  have hcs : ∀ i, i < j → coinSeq (fun i : Fin (Bd.IVStar.k (xIOf x)) => coinSeq t.2.1 i.val) i =
      coinSeq (fun i : Fin (Bd.IVStar.k (xIOf x)) => coinSeq (Claims.Recursion.fork j t.2.1 t.2.2.1) i.val) i :=
    fun i hi => Claims.Recursion.coinSeq_congr fun _ =>
      Claims.Recursion.coinSeq_congr fun h => Claims.Recursion.fork_agree j t.2.1 t.2.2.1 ⟨i, h⟩ hi
  obtain ⟨q, hq, q', hq', hc⟩ := hLB (xIOf x)
    ((List.range (Bd.DE.V.k (xIOf x))).map fun i => (b.rd.msgs x ws).getD i [])
    ((List.range (Bd.DE.V.k (xIOf x))).map fun i => (b.rd.msgs x ws').getD i [])
    ((List.range (Bd.DE.V.k (xIOf x))).map (coinSeq fun i : Fin (Bd.IVStar.k (xIOf x)) => coinSeq t.2.1 i.val))
    ((List.range (Bd.DE.V.k (xIOf x))).map (coinSeq fun i : Fin (Bd.IVStar.k (xIOf x)) =>
      coinSeq (Claims.Recursion.fork j t.2.1 t.2.2.1) i.val))
    (by simp) (by simp) (by simp) (by simp)
    (by
      rw [hdl, hdl]
      exact List.map_congr_left fun i hi => by
        rw [List.mem_range] at hi
        exact hag i (by omega) (by omega))
    (by
      rw [hdl, hdl]
      exact List.map_congr_left fun i hi => by
        rw [List.mem_range] at hi
        exact hcs i (by omega))
    (by rw [hgl, hgl]; exact hf) hacc₁ hacc₂
  have hsub : ∀ e ∈ (Bd.decAt H (xIOf x) (b.rd.msgs x ws) fun i => coinSeq t.2.1 i.val).2 ++
      (Bd.decAt H (xIOf x) (b.rd.msgs x ws') fun i => coinSeq (Claims.Recursion.fork j t.2.1 t.2.2.1) i.val).2,
      e ∈ (deployRec Bd.IVStar ivScope b).calls x ws ++ (deployRec Bd.IVStar ivScope b).calls x ws' := fun e he =>
    (List.mem_append.1 he).elim (fun h => List.mem_append_left _ (hlog₁ e h))
      (fun h => List.mem_append_right _ (hlog₂ e h))
  have hmem : ∀ y ∈ ((Bd.decAt H (xIOf x) (b.rd.msgs x ws) fun i => coinSeq t.2.1 i.val).2 ++
      (Bd.decAt H (xIOf x) (b.rd.msgs x ws') fun i =>
        coinSeq (Claims.Recursion.fork j t.2.1 t.2.2.1) i.val).2).map Prod.fst,
      y ∈ ((deployRec Bd.IVStar ivScope b).calls x ws ++ (deployRec Bd.IVStar ivScope b).calls x ws').map
        Prod.fst := fun y hy => by
    obtain ⟨e, he, rfl⟩ := List.mem_map.1 hy
    exact List.mem_map.2 ⟨e, hsub e he, rfl⟩
  exact hlift _ (fun p hp => hp) (Claims.Recursion.findColl_of_coll H htrue (hmem q hq) (hmem q' hq') hc)

end Forks

/-! ## The route without `ReadersRunPlayed` -/

/-- **`KSBoundDeployed` without `ReadersRunPlayed`**: `KSBoundRoute`'s premises, with `ReadersRunPlayed` dropped at a
layout whose held rounds have no pieces (`HeldWhole`), and the staging read at accepted runs (`StagedRuns`)
where `KSBoundRoute` has `VstarShowsB` and `RunPairs`. The readers' rounds before the last come off V*'s parts
(`readersRun_pick`, `vstarAccepts_boundW`), and the forks bind through `pick` (`forksBind_nf_pick`); the rest is
`ksBoundRoute_of_bridge`'s, at the bridge (`bridge_bdDeployed`). -/
theorem ksBoundRoute_heldWhole (rel relO : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (b : Brings) (L : Claims.Recursion.VStarB.LayoutB) (hRows : L.ReadsRows FlockDeploy.sha512) (hHW : L.HeldWhole) :
    ∀ (nextE : Msg → List Msg → List Coin → Claims.OC Out)
      (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
      (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ),
      let Bd := bdDeployed rel nextE colsE wE join rowIn final w
      let D := recBound rel b L Bd.IVStar
      let bL : Brings := { b with rd := L.readers }
      Bd.PlainKSAbort FlockDeploy.sha512 → Bd.Binds → Bd.WeightCovers → Bd.LastBinds FlockDeploy.sha512 →
      Claims.Flock.KS FlockDeploy.sha512 (deployZkRec relO) →
      (∀ shape ∈ [Claims.Recursion.Form.Dense, Claims.Recursion.Form.Holo, Claims.Recursion.Form.Sparse],
        ∃ (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop)
          (cmt : Msg → ℕ → Msg → Msg),
          (∀ decI, Claims.Recursion.VStarB.HeldB D (Claims.Recursion.Inner.ofFlock Bd.deploy decI) L shape) ∧
          (∀ decI, StagedRuns FlockDeploy.sha512 D (Claims.Recursion.Inner.ofFlock Bd.deploy decI) L shape shows
            cmt) ∧
          (∀ s, Claims.Recursion.VStarB.PartChecksB FlockDeploy.sha512 D (deployZkRec relO) L shape shows cmt chk s) ∧
          (∀ s, Claims.Recursion.VStarB.PartOpensB FlockDeploy.sha512 D (deployZkRec relO) L shape shows cmt s) ∧
          Claims.Recursion.VStarB.DecompB FlockDeploy.sha512 D Bd.IVStar L shape chk) →
      ReadersTrue FlockDeploy.sha512 Bd.IVStar bL (deployZkRec relO) →
      ReadersLogColl FlockDeploy.sha512 Bd.IVStar bL (deployZkRec relO) →
      KSBoundDeployed rel relO b L nextE colsE wE join rowIn final w := by
  intro nextE colsE wE join rowIn final w Bd D bL hP1 hBinds hWC hLB hF hshapes hT hL decI hdec
  let I := Claims.Recursion.Inner.ofFlock Bd.deploy decI
  let ivScope := (FlockDeploy.deployPlain rel).scopeCheck
  have hacc : ∀ shape ∈ [Claims.Recursion.Form.Dense, Claims.Recursion.Form.Holo, Claims.Recursion.Form.Sparse],
      Claims.Recursion.VStarAccepts FlockDeploy.sha512 (deployRec I.V ivScope bL) I (deployZkRec relO) shape := by
    intro shape hs
    obtain ⟨chk, shows, cmt, hHe, hSt, hCk, hOp, hDc⟩ := hshapes shape hs
    exact vstarAccepts_boundW FlockDeploy.sha512 I ivScope b L (deployZkRec relO) shape chk shows cmt hHW
      (hHe decI) (hSt decI) hCk hOp hDc hRows
  have hsh : ∀ shape ∈ [Claims.Recursion.Form.Dense, Claims.Recursion.Form.Holo, Claims.Recursion.Form.Sparse],
      ∃ (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) (cmt : Msg → ℕ → Msg → Msg),
        Claims.Recursion.VStarB.HeldB (deployRec I.V ivScope bL) I L shape ∧
        StagedRuns FlockDeploy.sha512 (deployRec I.V ivScope bL) I L shape shows cmt ∧
        ∀ s, Claims.Recursion.VStarB.PartOpensB FlockDeploy.sha512 (deployRec I.V ivScope bL) (deployZkRec relO) L
          shape shows cmt s := by
    intro shape hs
    obtain ⟨-, shows, cmt, hHe, hSt, -, hOp, -⟩ := hshapes shape hs
    exact ⟨shows, cmt, hHe decI, hSt decI, hOp⟩
  have hA := innerAcceptsColl_bound FlockDeploy.sha512 I ivScope b L (deployZkRec relO) hT
    (hacc _ (by simp)) (hacc _ (by simp)) (hacc _ (by simp))
  have hTc := callsTrue FlockDeploy.sha512 I ivScope bL (deployZkRec relO) hT
  have hLc := callsLogColl FlockDeploy.sha512 I ivScope bL (deployZkRec relO) hT hL
  have hFk := forksBind_nf_pick FlockDeploy.sha512 Bd decI ivScope bL (deployZkRec relO) L.pick hLB hTc hLc hA
    fun decO x Ω P ω r c ws hg =>
      readersRun_pick FlockDeploy.sha512 I ivScope b L (deployZkRec relO) hHW hsh hT hg
  let bN := nfBrings Bd bL
  have hKS : I.KSAbort FlockDeploy.sha512 :=
    bridge_bdDeployed rel nextE colsE wE join rowIn final w decI hdec hBinds hWC (hP1 decI hdec)
  have hdecC : ∀ x c, FlockDeploy.decCoin x (Bd.ans c) = c := fun _ c =>
    (congrArg (FlockDeploy.decCoin _) (bdDeployed_ans rel nextE colsE wE join rowIn final w c)).trans
      (Function.leftInverse_invFun Claims.Bridge.coin_bytes_injective c)
  have hpc := Bd.publicCoin FlockDeploy.sha512 decI FlockDeploy.decCoin hdecC
  have hM := roundsMargin_deployRec FlockDeploy.sha512 I ivScope bN fun x => V_k_pos_flock false x
  have hLn := callsLogColl_nf FlockDeploy.sha512 Bd decI ivScope bL (deployZkRec relO) hLc
  have hAn := innerAcceptsColl_nf FlockDeploy.sha512 Bd decI ivScope bL (deployZkRec relO) hA
  have hKSn := Claims.Recursion.KS_of_flockKS_binds FlockDeploy.sha512 (deployRec I.V ivScope bN) I (deployZkRec relO)
    FlockDeploy.decCoin hF hKS (refuses I.V ivScope bN FlockDeploy.sha512) (innerRounds I ivScope bN)
    (sessionNums I.V ivScope bN relO) (tableErrNonneg I.V ivScope bN) (sessionRounds I ivScope bN relO)
    (sessionAccepts I.V ivScope bN) (sessionRuns_sha512 I ivScope bN relO)
    (innerViewRun FlockDeploy.sha512 I ivScope bN (deployZkRec relO)) hTc hLn hFk hAn hM hpc
    (callsNil I.V ivScope bN FlockDeploy.sha512)
  exact ksA_of_nf FlockDeploy.sha512 Bd decI ivScope bL (deployZkRec relO) hKSn

end RecDeploy
