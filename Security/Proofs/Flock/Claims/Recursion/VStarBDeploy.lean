import Proofs.Flock.Claims.Recursion.BoundBridge
import Proofs.Flock.Claims.Recursion.VStarLinks
import Proofs.Flock.Claims.Recursion.DeployChecked

/-!
# V*'s bound-layout premises at the deployment

`RecDeploy.KSBoundRoute` takes five facts about V*'s parts at the bound layout `L` (`VStarBound.lean`), at each of the
dense, holo and sparse forms: `HeldB`, `VstarShowsB`, `PartChecksB`, `PartOpensB` and `DecompB`, at the compiled
record `deployRec IVStar scope {b with rd := L.readers}`. What holds of them there:

* **`VstarShowsB` is false at `deployRec`** (`vstarShowsB_deployRec_forces`, `vstarShowsB_deployed_rejects`). It asks
  every staging, accepted or not, to show the pairs' commit strings. A staging message whose inner-session record isn't
  UTF-8 (`badStage`) stages a V* statement that doesn't set up (`setupO_staged_none`: `sourceOf` reads the record as
  text), so its numbers aren't the part's and `vstarAt` takes the statement staged at `pick`, which no round-0 message
  moves. So with `HeldB` at the deployed bound record, `VstarShowsB` leaves the compiled verifier no accepted run at any
  statement of the shape with a part.
* **What the route reads of it holds** (`StagedB`, `stagedB_deployRec`): `vstarAccepts_bound` reads `VstarShowsB` only
  at an accepted run's first `k_I + kS` pairs. `StagedB` is that, the bound layout's `VStar.Staged`, and at `deployRec`
  it holds as `staged_deployRec` does, at the coin server's transcript read by a reader (`showsTr`, `cmtTr`) wherever
  that reader reads it back before the last round at an accepted run (`ReadsBackB`, which the coin server's record
  meets: it records no last round) and `HeldB` pins the layout's rounds. `KSBoundRouteStaged` is
  `KSBoundRoute` with `StagedB` for `VstarShowsB` and `RunPairs` proved (`runPairs_deployRec`); it is proved
  (`ksBoundRouteStaged`), and `ksBoundRoute_deployed_parts` is it with the staging discharged.
* **`HeldB` is false at V\*'s registered ports** (`layoutBOf`, `heldB_layoutBOf_iff`): `layoutOf`'s holdings as a bound
  layout leave round 0, the link message, neither fixed, held nor assembled, so `HeldB` holds there only at a shape no
  compiled statement parses at.

`PartChecksB`, `PartOpensB` and `DecompB` stay premises of `ksBoundRoute_deployed_parts`, the first two at
`showsPart`: the V\* statements staged as V\* stages them (`StagedPart`), whose setup's refusals (`RecIns.setup`) hold.
The outer parse (`readOfO`) runs none of them, so off `StagedPart` a statement whose public rows disagree with its
transcript still parses. `StagedB` holds at `showsPart` (`stagedB_deployRec_part`): an accepted run stages every part.
-/

namespace Claims.Recursion.VStarB

open Claims

section Staged

variable {Coin W : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- **The staged V\* statements show the run** (option (d)'s `VStar.Staged`): on an accepted run of a compiled statement
of the shape, each V* statement shows the run's coins and, as the commit string of each inner round before the last, the
run's message at that round. -/
def StagedB (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (L : LayoutB) (shape : Form → Prop)
    (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) (cmt : Msg → ℕ → Msg → Msg) : Prop :=
  ∀ (x : Msg) (st : Stmt), D.parse x = some st → shape st.form →
    ∀ (Ω : Type) (P : Prover Ω) (ω : Ω) (r : Fin (D.V.k x) → Coin), (run H P D.V x ω r).out = .acc →
      ∀ s < D.nS x, shows s x ((runVstar H D I P x ω r).getD s []) (coinSeq r) ∧
        ∀ i, i + 1 < L.rounds x →
          cmt x i ((runVstar H D I P x ω r).getD s []) = ((run H P D.V x ω r).msgs.map Prod.fst).getD i []

end Staged

end Claims.Recursion.VStarB

namespace RecDeploy

open Claims (Msg Out)
open FlockDeploy (Coin)

/-! ## A staging that doesn't set up -/

/-- One byte that no UTF-8 text holds. -/
def badRecord : ByteArray := ⟨#[255]⟩

theorem badRecord_validate : ByteArray.validateUTF8 badRecord = false := by
  unfold ByteArray.validateUTF8
  rw [ByteArray.validateUTF8.go]
  decide +kernel

theorem badRecord_utf8 : String.fromUTF8? badRecord = none := by
  have h : ¬ badRecord.IsValidUTF8 := ByteArray.validateUTF8_eq_false_iff.1 badRecord_validate
  unfold String.fromUTF8?
  rw [dif_neg h]

/-- **A staging message with no public file, the record `badRecord` and no shared rows.** -/
def badStage : Msg := encStage [] badRecord ByteArray.empty

theorem decodeStage_badStage : decodeStage badStage = some ([], badRecord, ByteArray.empty) := by
  rfl

/-- **A V\* statement staged at a record that isn't text doesn't set up**: `sourceOf` reads the inner session's record
as UTF-8 text (`textOf`) whatever the partition, so the statement's readers refuse it. -/
theorem setupO_staged_none {st : Stage} {x : Msg} {ms : List (Msg × Out)} {s : ℕ}
    (hrec : String.fromUTF8? (st.record x ms) = none) : setupO (staged st x ms s) = none := by
  rcases hso : setupO (staged st x ms s) with _ | ⟨i, spec, sts⟩
  · rfl
  exfalso
  have h := (setupO_some hso).1
  have hne : staged st x ms s ≠ [] := fun he => by rw [he, insO_nil] at h; cases h
  obtain ⟨r, p, -, -, hd⟩ := staged_decode hne
  unfold insO at h
  rw [hd] at h
  simp only [Option.bind_eq_bind, Option.bind_some] at h
  split_ifs at h
  · simp at h
  have h' : (partRaw r st x ms s p).ins.toOption = some i := by simpa using h
  obtain ⟨-, hsrc⟩ := RecRaw.ins_inner_src (toOption_eq_some h')
  unfold sourceOf at hsrc
  rw [formOf_partRaw] at hsrc
  dsimp only [partRaw] at hsrc
  cases hq : r.innerRaw.partition with
  | none => rw [hq] at hsrc; simp at hsrc
  | some q =>
    rw [hq] at hsrc
    simp only [bind, Except.bind] at hsrc
    cases hj : FlockDeploy.jsonOf "--rec-inner-partition" q with
    | error e => rw [hj] at hsrc; cases hsrc
    | ok pj =>
      rw [hj] at hsrc
      cases ht : textOf "rec inner session" (st.record x ms) with
      | error e => rw [ht] at hsrc; cases hsrc
      | ok rec => simp [textOf, hrec] at ht

/-! ## `VstarShowsB` at `deployRec` -/

/-- **`VstarShowsB` forces one round-0 message at `deployRec`.** Let part `s` of a compiled statement of the shape have
a staging that sets up (an accepted run stages one: `stagedPart_run`), and the inner verifier and the layout span at
least one and two rounds. Then `VstarShowsB` fails: at `badStage` the part's statement doesn't set up
(`setupO_staged_none`), so its numbers (one round) aren't the part's (`pick`'s, at least two), `vstarAt` takes the
statement staged at `pick`, and that statement would have to show both `[]` and `[0]` as round 0's commit string. -/
theorem vstarShowsB_deployRec_forces {W : Type} {IV : Claims.Verifier Coin} {ivScope : Msg → Bool} {b : Brings}
    {I : Claims.Recursion.Inner Coin W} {L : Claims.Recursion.VStarB.LayoutB} {shape : Claims.Recursion.Form → Prop}
    {decC : Msg → Msg → Coin} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop} {cmt : Msg → ℕ → Msg → Msg}
    (hV : Claims.Recursion.VStarB.VstarShowsB (deployRec IV ivScope b) I L shape decC shows cmt)
    {x : Msg} {st : Claims.Recursion.Stmt} (hst : parse x = some st) (hf : shape st.form) {s : ℕ} (hs : s < nS x)
    (hset : ∃ ms, (setupO (staged b.st x ms s)).isSome) (hk : 0 < I.V.k (xIOf x)) (hr : 1 < L.rounds x) :
    False := by
  let ps : Msg → List (Msg × Out) := fun a =>
    ((a, Out.msg []) :: List.replicate (I.V.k (xIOf x) - 1) (([] : Msg), Out.msg [])) ++ [(badStage, Out.msg [])]
  have hlen : ∀ a, (ps a).length =
      I.V.k (Claims.Recursion.xI (deployRec IV ivScope b) x) + (deployRec IV ivScope b).kS x := by
    intro a
    show _ = I.V.k (xIOf x) + 1
    simp only [ps, List.length_append, List.length_cons, List.length_replicate, List.length_nil]
    omega
  have hmsg : ∀ a, ∀ p ∈ ps a, p.2 = Out.msg [] := by
    intro a p hp
    simp only [ps, List.mem_append, List.mem_cons, List.mem_replicate] at hp
    rcases hp with (rfl | ⟨-, rfl⟩) | (rfl | h)
    · rfl
    · rfl
    · rfl
    · simp at h
  have hc : ∀ a, ∀ i < (ps a).length, ∃ m, ((ps a).getD i ([], .rej)).2 = .msg m ∧ decC x m = decC x [] := by
    intro a i hi
    refine ⟨[], ?_, rfl⟩
    rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hi, Option.getD_some]
    exact hmsg a _ (List.getElem_mem hi)
  have hpick : (setupO (staged b.st x (pick b.st x s) s)).isSome := by
    unfold pick
    rw [dif_pos hset]
    exact Classical.choose_spec hset
  obtain ⟨⟨i, spec, sts⟩, ht⟩ := Option.isSome_iff_exists.1 hpick
  have hv : ∀ a, ((deployRec IV ivScope b).vstar x (ps a)).getD s [] = staged b.st x (pick b.st x s) s := by
    intro a
    show (vstar b.st x (ps a)).getD s [] = _
    rw [vstar_getD b.st x _ hs]
    unfold vstarAt
    rw [if_neg]
    intro heq
    have hk' := congrArg Claims.Recursion.Nums.rounds heq
    change VO.k (staged b.st x (ps a) s) = VO.k (staged b.st x (pick b.st x s) s) at hk'
    have hl : stageOf (ps a) = some ([], badRecord, ByteArray.empty) := by
      show decodeStage (((ps a).getLast?.map Prod.fst).getD []) = _
      rw [show (ps a).getLast? = some (badStage, Out.msg []) from List.getLast?_concat]
      exact decodeStage_badStage
    have hbad : setupO (staged b.st x (ps a) s) = none := by
      apply setupO_staged_none
      show String.fromUTF8? (((stageOf (ps a)).map (·.2.1)).getD ByteArray.empty) = none
      rw [hl]
      exact badRecord_utf8
    simp only [VO, hbad, ht] at hk' <;> omega
  have h1 := (hV x st hst hf (ps []) (hlen _) (fun _ => decC x []) (hc []) s hs).2 0 hr
  have h2 := (hV x st hst hf (ps [0]) (hlen _) (fun _ => decC x []) (hc [0]) s hs).2 0 hr
  rw [hv] at h1 h2
  have e : ∀ a, ((ps a).map Prod.fst).getD 0 [] = a := fun a => rfl
  rw [e] at h1 h2
  exact absurd (h1.symm.trans h2) (by simp)

/-- **`VstarShowsB` with `HeldB` at the deployed bound record leaves no accepted run**: at every choice the code side
makes, wherever V*'s parts at `L` meet `HeldB` and `VstarShowsB` at a shape, the compiled verifier accepts no run of a
compiled statement of the shape with a part. An accepted run stages every part at a statement that sets up
(`stagedPart_run`), the deployed plain verifier takes at least two rounds at the inner statement (`parse_innerOk`), and
`HeldB` makes them the layout's, so `vstarShowsB_deployRec_forces` applies. Since an honest run is accepted, `VstarShowsB`
is false at the deployment wherever an honest run reaches a statement of the shape. -/
theorem vstarShowsB_deployed_rejects (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (b : Brings) (L : Claims.Recursion.VStarB.LayoutB) (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (decI : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace) (shape : Claims.Recursion.Form → Prop)
    (decC : Msg → Msg → Coin) (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) (cmt : Msg → ℕ → Msg → Msg) :
    let Bd := bdDeployed rel nextE colsE wE join rowIn final w
    let D := recBound rel b L Bd.IVStar
    Claims.Recursion.VStarB.HeldB D (Claims.Recursion.Inner.ofFlock Bd.deploy decI) L shape →
    Claims.Recursion.VStarB.VstarShowsB D (Claims.Recursion.Inner.ofFlock Bd.deploy decI) L shape decC shows cmt →
    ∀ (x : Msg) (st : Claims.Recursion.Stmt), parse x = some st → shape st.form → 0 < nS x →
      ∀ (H : Msg → Msg) (Ω : Type) (P : Claims.Prover Ω) (ω : Ω) (r : Fin (D.V.k x) → Coin),
        (Claims.run H P D.V x ω r).out ≠ .acc := by
  intro Bd D hH hV x st hst hf hn H Ω P ω r hacc
  let I := Claims.Recursion.Inner.ofFlock Bd.deploy decI
  let bL : Brings := { b with rd := L.readers }
  obtain ⟨ms, i, spec, sts, hxO, hso, -⟩ := stagedPart_run H Bd.IVStar bL.st x P ω r hacc (s := 0) hn
  have hset : ∃ ms, (setupO (staged bL.st x ms 0)).isSome := ⟨ms, by rw [← hxO, hso]; rfl⟩
  have hk2 : 2 ≤ (FlockDeploy.V false).k (xIOf x) := by
    obtain ⟨s', spec', sts', hs', -⟩ := parse_innerOk hst
    simp only [FlockDeploy.V, hs'] <;> omega
  obtain ⟨hK, -, -⟩ := hH x st hst hf
  have hK' : (FlockDeploy.V false).k (xIOf x) = L.rounds x := hK
  have hk : 0 < I.V.k (xIOf x) := by
    show 0 < (FlockDeploy.V false).k (xIOf x)
    omega
  exact vstarShowsB_deployRec_forces (IV := Bd.IVStar) (ivScope := (FlockDeploy.deployPlain rel).scopeCheck)
    (b := bL) (I := I) hV hst hf hn hset hk (by omega)

/-! ## `HeldB` at V\*'s registered ports -/

/-- **V\*'s registered ports as a bound layout**: `layoutOf`'s rounds, holdings and fixed rounds, each position its
one-wire form, with no pieces. -/
def layoutBOf : Claims.Recursion.VStarB.LayoutB := Claims.Recursion.VStarB.LayoutB.ofLayout layoutOf

/-- **`HeldB` at the registered ports holds only where nothing of the shape parses**: `layoutOf` spans at least one
round (`layoutOf_rounds_pos`), and round 0, the link message, is neither fixed nor held by any part
(`layoutOf_msgAt_zero`), and it has no pieces. -/
theorem heldB_layoutBOf_iff {W : Type} (D : Claims.Recursion.Deploy Coin) (I : Claims.Recursion.Inner Coin W)
    (shape : Claims.Recursion.Form → Prop) :
    Claims.Recursion.VStarB.HeldB D I layoutBOf shape ↔
      ∀ (x : Msg) (st : Claims.Recursion.Stmt), D.parse x = some st → ¬ shape st.form := by
  constructor
  · intro h x st hst hf
    obtain ⟨hK, hcl, -⟩ := h x st hst hf
    have hpos : 0 < I.V.k (Claims.Recursion.xI D x) := by
      rw [hK]
      exact layoutOf_rounds_pos x
    rcases (hcl 0 hpos).1 with h0 | ⟨s, -, h0⟩ | ⟨h0, -⟩
    · have h' : layoutBOf.fixedAt x 0 = none := rfl
      rw [h'] at h0
      simp at h0
    · have h' : layoutBOf.msgAt x s 0 = none := rfl
      rw [h'] at h0
      simp at h0
    · exact h0 rfl
  · intro h x st hst hf
    exact absurd hf (h x st hst)

/-! ## The route with the staging read off the run -/

/-- **`VStarAccepts` at V\*'s bound layout, with the staging off the run** (`vstarAccepts_bound` with `StagedB` for
`VstarShowsB`): the route reads the staging only at an accepted run's pairs. -/
theorem vstarAccepts_bound_staged {W Col T : Type} (H : Msg → Msg) (I : Claims.Recursion.Inner Coin W)
    (ivScope : Msg → Bool) (b : Brings) (L : Claims.Recursion.VStarB.LayoutB) (DO : Claims.Flock.Deploy Coin Col T)
    (shape : Claims.Recursion.Form → Prop) (decC : Msg → Msg → Coin)
    (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop)
    (cmt : Msg → ℕ → Msg → Msg)
    (hplay : ∀ x ms (c : Fin (I.V.k (xIOf x)) → Coin),
      (Claims.run H (Claims.replay ms) I.V (xIOf x) () c).msgs.length = I.V.k (xIOf x))
    (hRP : Claims.Recursion.VStar.RunPairs H (deployRec I.V ivScope { b with rd := L.readers }) I decC)
    (hHeld : Claims.Recursion.VStarB.HeldB (deployRec I.V ivScope { b with rd := L.readers }) I L shape)
    (hSt : Claims.Recursion.VStarB.StagedB H (deployRec I.V ivScope { b with rd := L.readers }) I L shape shows cmt)
    (hChk : ∀ s, Claims.Recursion.VStarB.PartChecksB H (deployRec I.V ivScope { b with rd := L.readers }) DO L shape
      shows cmt chk s)
    (hOpen : ∀ s, Claims.Recursion.VStarB.PartOpensB H (deployRec I.V ivScope { b with rd := L.readers }) DO L shape
      shows cmt s)
    (hDec : Claims.Recursion.VStarB.DecompB H (deployRec I.V ivScope { b with rd := L.readers }) I.V L shape chk)
    (hRows : L.ReadsRows H)
    (hRRP : ReadersRunPlayed H I.V { b with rd := L.readers } DO) :
    Claims.Recursion.VStarAccepts H (deployRec I.V ivScope { b with rd := L.readers }) I DO shape := by
  intro x st hst hf Ω P ω r hacc ws ho htrue
  obtain ⟨hlenR, -⟩ := hRP x Ω P ω r hacc
  obtain ⟨hK, hcl, hlast⟩ := hHeld x st hst hf
  change I.V.k (xIOf x) + kS x ≤ _ at hlenR
  change I.V.k (xIOf x) = L.rounds x at hK
  have hsh : ∀ s < (deployRec I.V ivScope { b with rd := L.readers }).nS x,
      shows s x (((deployRec I.V ivScope { b with rd := L.readers }).vstar x
        ((Claims.run H P (deployRec I.V ivScope { b with rd := L.readers }).V x ω r).msgs.take
          (I.V.k (xIOf x) + kS x))).getD s []) (Claims.coinSeq r) ∧
      ∀ i, i + 1 < L.rounds x →
        cmt x i (((deployRec I.V ivScope { b with rd := L.readers }).vstar x
          ((Claims.run H P (deployRec I.V ivScope { b with rd := L.readers }).V x ω r).msgs.take
            (I.V.k (xIOf x) + kS x))).getD s []) =
        (((Claims.run H P (deployRec I.V ivScope { b with rd := L.readers }).V x ω r).msgs.take
          (I.V.k (xIOf x) + kS x)).map Prod.fst).getD i [] := by
    intro s hs
    obtain ⟨h1, h2⟩ := hSt x st hst hf Ω P ω r hacc s hs
    refine ⟨h1, fun i hi => (h2 i hi).trans ?_⟩
    have hin : i < I.V.k (xIOf x) + kS x := by omega
    rw [List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD, List.map_take, List.getElem?_take_of_lt hin]
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

/-- **`StagedB` at `deployRec`**: as `staged_deployRec`, at the coin server's transcript read by `untr` (`showsTr`,
`cmtTr`), wherever `untr` reads it back before the last round at an accepted run (`ReadsBackB`) at the compiled
statements of the shape, at a layout whose rounds `HeldB` pins to the inner verifier's. -/
theorem stagedB_deployRec {W : Type} (H : Msg → Msg) (I : Claims.Recursion.Inner Coin W) (ivScope : Msg → Bool)
    (b : Brings) (untr : TrReader) (L : Claims.Recursion.VStarB.LayoutB) (shape : Claims.Recursion.Form → Prop)
    (hrb : ReadsBackB H I.V b shape untr)
    (hHeld : Claims.Recursion.VStarB.HeldB (deployRec I.V ivScope b) I L shape) :
    Claims.Recursion.VStarB.StagedB H (deployRec I.V ivScope b) I L shape (showsTr untr) (cmtTr untr) := by
  intro x st hst hf Ω P ω r hacc s hs
  have hL : L.rounds x ≤ I.V.k (xIOf x) := ((hHeld x st hst hf).1).symm.le
  change (Claims.run H P (V I.V b.st) x ω r).out = .acc at hacc
  obtain ⟨ps, hps, hfst, hdrawn⟩ := hrb x st hst hf Ω P ω r hacc
  change s < nS x at hs
  have hne := staged_ne_nil H I.V b.st x P ω r hacc hs
  obtain ⟨hnum, -⟩ := run_round H I.V b.st x P ω r hacc hs rfl rfl
  obtain ⟨xr, p, -, -, hd⟩ := staged_decode hne
  have hK := run_acc_length H I.V b.st x P ω r hacc
  have hm := roundsMargin_V H I.V b.st x P ω r hacc
  have hv : (Claims.Recursion.runVstar H (deployRec I.V ivScope b) I P x ω r).getD s [] =
      staged b.st x ((Claims.run H P (V I.V b.st) x ω r).msgs.take (I.V.k (xIOf x) + kS x)) s := by
    show (vstar b.st x ((Claims.run H P (V I.V b.st) x ω r).msgs.take (I.V.k (xIOf x) + kS x))).getD s [] = _
    rw [vstar_getD b.st x _ hs]
    unfold vstarAt
    rw [ite_eq_left hnum]
  have hdl : ((Claims.run H P (V I.V b.st) x ω r).msgs.take (I.V.k (xIOf x) + kS x)).dropLast =
      (Claims.run H P (V I.V b.st) x ω r).msgs.take (I.V.k (xIOf x)) := by
    apply List.ext_getElem
    · simp only [List.length_dropLast, List.length_take, kS]
      omega
    · intro i h₁ h₂
      simp only [List.getElem_dropLast, List.getElem_take]
  have hshown : shownOf untr x
      (staged b.st x ((Claims.run H P (V I.V b.st) x ω r).msgs.take (I.V.k (xIOf x) + kS x)) s) = some ps := by
    unfold shownOf
    rw [hd, Option.bind_some]
    show Option.bind
      (some (b.tr x ((Claims.run H P (V I.V b.st) x ω r).msgs.take (I.V.k (xIOf x) + kS x)).dropLast)) (untr x) = _
    rw [Option.bind_some, hdl]
    exact hps
  rw [hv]
  refine ⟨⟨_, hshown, hdrawn⟩, fun i hi => ?_⟩
  unfold cmtTr
  rw [hshown]
  exact hfst i (by omega)

/-- **`StagedB` at `showsPart`**: `stagedB_deployRec`, with each part's V\* statement on the run staged as V\* stages
it (`stagedPart_run`). So the parts' premises may be stated at the staged statements alone, where the setup's
refusals hold. -/
theorem stagedB_deployRec_part {W : Type} (H : Msg → Msg) (I : Claims.Recursion.Inner Coin W) (ivScope : Msg → Bool)
    (b : Brings) (untr : TrReader) (L : Claims.Recursion.VStarB.LayoutB) (shape : Claims.Recursion.Form → Prop)
    (hrb : ReadsBackB H I.V b shape untr)
    (hHeld : Claims.Recursion.VStarB.HeldB (deployRec I.V ivScope b) I L shape) :
    Claims.Recursion.VStarB.StagedB H (deployRec I.V ivScope b) I L shape (showsPart untr b) (cmtTr untr) := by
  intro x st hst hf Ω P ω r hacc s hs
  obtain ⟨h1, h2⟩ := stagedB_deployRec H I ivScope b untr L shape hrb hHeld x st hst hf Ω P ω r hacc s hs
  refine ⟨⟨h1, ?_⟩, h2⟩
  change (Claims.run H P (V I.V b.st) x ω r).out = .acc at hacc
  change s < nS x at hs
  obtain ⟨hnum, -⟩ := run_round H I.V b.st x P ω r hacc hs rfl rfl
  have hv : (Claims.Recursion.runVstar H (deployRec I.V ivScope b) I P x ω r).getD s [] =
      staged b.st x ((Claims.run H P (V I.V b.st) x ω r).msgs.take (I.V.k (xIOf x) + kS x)) s := by
    show (vstar b.st x ((Claims.run H P (V I.V b.st) x ω r).msgs.take (I.V.k (xIOf x) + kS x))).getD s [] = _
    rw [vstar_getD b.st x _ hs]
    unfold vstarAt
    rw [ite_eq_left hnum]
  rw [hv]
  exact stagedPart_run H I.V b.st x P ω r hacc hs

/-- **`KSBoundRoute` with the staging read off the run**: `KSBoundRoute`'s premises, with `StagedB` (at an accepted
run) where `KSBoundRoute` has `VstarShowsB` (at every staging, false at the deployment: `vstarShowsB_deployed_rejects`)
and `RunPairs` dropped (it holds at `decCoin`: `runPairs_deployRec`), give `KSBoundDeployed`. -/
def KSBoundRouteStaged (rel relO : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings)
    (L : Claims.Recursion.VStarB.LayoutB) : Prop :=
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
        (∀ decI, Claims.Recursion.VStarB.StagedB FlockDeploy.sha512 D (Claims.Recursion.Inner.ofFlock Bd.deploy decI)
          L shape shows cmt) ∧
        (∀ s, Claims.Recursion.VStarB.PartChecksB FlockDeploy.sha512 D (deployZkRec relO) L shape shows cmt chk s) ∧
        (∀ s, Claims.Recursion.VStarB.PartOpensB FlockDeploy.sha512 D (deployZkRec relO) L shape shows cmt s) ∧
        Claims.Recursion.VStarB.DecompB FlockDeploy.sha512 D Bd.IVStar L shape chk) →
    ReadersTrue FlockDeploy.sha512 Bd.IVStar bL (deployZkRec relO) →
    ReadersLogColl FlockDeploy.sha512 Bd.IVStar bL (deployZkRec relO) →
    ReadersRunPlayed FlockDeploy.sha512 Bd.IVStar bL (deployZkRec relO) →
    KSBoundDeployed rel relO b L nextE colsE wE join rowIn final w

/-- **`KSBoundRouteStaged`**, proved as `ksBoundRoute_of_bridge` is, at the bridge (`bridge_bdDeployed`), with
`vstarAccepts_bound_staged` for `vstarAccepts_bound` and `RunPairs` at `decCoin` (`runPairs_deployRec`). -/
theorem ksBoundRouteStaged (rel relO : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings)
    (L : Claims.Recursion.VStarB.LayoutB)
    (hRows : L.ReadsRows FlockDeploy.sha512) : KSBoundRouteStaged rel relO b L := by
  intro nextE colsE wE join rowIn final w Bd D bL hP1 hBinds hWC hLB hF hshapes hT hL hRRP decI hdec
  let I := Claims.Recursion.Inner.ofFlock Bd.deploy decI
  let ivScope := (FlockDeploy.deployPlain rel).scopeCheck
  have hplay : ∀ x ms (c : Fin (I.V.k (xIOf x)) → Coin),
      (Claims.run FlockDeploy.sha512 (Claims.replay ms) I.V (xIOf x) () c).msgs.length = I.V.k (xIOf x) :=
    fun x ms c => Bd.run_ivstar_length FlockDeploy.sha512 (xIOf x) ms c
  have hacc : ∀ shape ∈ [Claims.Recursion.Form.Dense, Claims.Recursion.Form.Holo, Claims.Recursion.Form.Sparse],
      Claims.Recursion.VStarAccepts FlockDeploy.sha512 (deployRec I.V ivScope bL) I (deployZkRec relO) shape := by
    intro shape hs
    obtain ⟨chk, shows, cmt, hHe, hSt, hCk, hOp, hDc⟩ := hshapes shape hs
    exact vstarAccepts_bound_staged FlockDeploy.sha512 I ivScope b L (deployZkRec relO) shape FlockDeploy.decCoin chk
      shows cmt hplay (runPairs_deployRec FlockDeploy.sha512 I ivScope bL) (hHe decI) (hSt decI) hCk hOp hDc hRows hRRP
  have hA := innerAcceptsColl_bound FlockDeploy.sha512 I ivScope b L (deployZkRec relO) hT
    (hacc _ (by simp)) (hacc _ (by simp)) (hacc _ (by simp))
  have hTc := callsTrue FlockDeploy.sha512 I ivScope bL (deployZkRec relO) hT
  have hLc := callsLogColl FlockDeploy.sha512 I ivScope bL (deployZkRec relO) hT hL
  have hFk := forksBind_nf FlockDeploy.sha512 Bd decI ivScope bL (deployZkRec relO) hLB hTc hLc hA
    fun decO x Ω P ω r c ws hg => readersRun_played FlockDeploy.sha512 I ivScope bL (deployZkRec relO) hplay hT hRRP hg
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

/-- **The deployed route with V\*'s staging discharged**: `KSBoundRouteStaged` at the coin server's transcript read by
`untr`, at the V\* statements staged as V\* stages them (`showsPart`, `cmtTr`), where `StagedB` holds wherever `untr`
reads it back at an accepted run (`ReadsBackB`, `stagedB_deployRec_part`). What V*'s parts still owe at each form is
`HeldB`, `PartChecksB`, `PartOpensB` and `DecompB`, for some inner check `chk`, the two parts' facts only at staged
statements, where the setup's refusals hold. -/
theorem ksBoundRoute_deployed_parts (rel relO : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (b : Brings) (L : Claims.Recursion.VStarB.LayoutB) (hRows : L.ReadsRows FlockDeploy.sha512) (untr : TrReader) :
    ∀ (nextE : Msg → List Msg → List Coin → Claims.OC Out)
      (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
      (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ),
      let Bd := bdDeployed rel nextE colsE wE join rowIn final w
      let D := recBound rel b L Bd.IVStar
      let bL : Brings := { b with rd := L.readers }
      ReadsBackB FlockDeploy.sha512 Bd.IVStar bL (fun _ => True) untr →
      Bd.PlainKSAbort FlockDeploy.sha512 → Bd.Binds → Bd.WeightCovers → Bd.LastBinds FlockDeploy.sha512 →
      Claims.Flock.KS FlockDeploy.sha512 (deployZkRec relO) →
      (∀ shape ∈ [Claims.Recursion.Form.Dense, Claims.Recursion.Form.Holo, Claims.Recursion.Form.Sparse],
        ∃ chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop,
          (∀ decI, Claims.Recursion.VStarB.HeldB D (Claims.Recursion.Inner.ofFlock Bd.deploy decI) L shape) ∧
          (∀ s, Claims.Recursion.VStarB.PartChecksB FlockDeploy.sha512 D (deployZkRec relO) L shape
            (showsPart untr b) (cmtTr untr) chk s) ∧
          (∀ s, Claims.Recursion.VStarB.PartOpensB FlockDeploy.sha512 D (deployZkRec relO) L shape
            (showsPart untr b) (cmtTr untr) s) ∧
          Claims.Recursion.VStarB.DecompB FlockDeploy.sha512 D Bd.IVStar L shape chk) →
      ReadersTrue FlockDeploy.sha512 Bd.IVStar bL (deployZkRec relO) →
      ReadersLogColl FlockDeploy.sha512 Bd.IVStar bL (deployZkRec relO) →
      ReadersRunPlayed FlockDeploy.sha512 Bd.IVStar bL (deployZkRec relO) →
      KSBoundDeployed rel relO b L nextE colsE wE join rowIn final w := by
  intro nextE colsE wE join rowIn final w Bd D bL hrb hP1 hBinds hWC hLB hF hparts hT hL hRRP
  refine ksBoundRouteStaged rel relO b L hRows nextE colsE wE join rowIn final w hP1 hBinds hWC hLB hF ?_ hT hL hRRP
  intro shape hs
  obtain ⟨chk, hHe, hCk, hOp, hDc⟩ := hparts shape hs
  exact ⟨chk, showsPart untr b, cmtTr untr, hHe,
    fun decI => stagedB_deployRec_part FlockDeploy.sha512 (Claims.Recursion.Inner.ofFlock Bd.deploy decI)
      (FlockDeploy.deployPlain rel).scopeCheck bL untr L shape (fun x st hst _ => hrb x st hst trivial) (hHe decI),
    hCk, hOp, hDc⟩

end RecDeploy
