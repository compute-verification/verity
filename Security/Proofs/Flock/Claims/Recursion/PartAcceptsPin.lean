import Proofs.Flock.Claims.Recursion.PartAccepts
import Proofs.Flock.Claims.Recursion.KSLin2Staged
import Proofs.Flock.Claims.Recursion.OpenLowerDeployed

/-!
# `DecompB` with each part's witness pinned to the parts' own (`DecompBP`)

`PartAccepts` names some outer witness `w` at which part `s`'s circuit accepts, and `DecompB` reads it at the parts'
own messages `L.msgs x ws`, so `w` need not be `ws[s]`: `DecompB`'s collision outcome and `Joint`'s no-collision
conjunct, both over `L.allCalls x ws`, say nothing of `w`'s calls. Where a part's check must bind across parts
through the recorded calls (a level part's `out` rows to RecCap's `top` rows), the witness has to be `ws[s]`.

* `PartAcceptsP … s x c w ms`: `PartAccepts` at the witness `w` given.
* `DecompBP`: `DecompB` with part `s`'s check read at `ws[s]`. It follows from `DecompB` at `PartAccepts`
  (`decompBP_of_decompB`), so every route proved at `PartAccepts` gives it.
* `JointP`: `OpenLower.Joint` with part `s`'s check at `ws[s]`, so a fact asked under it (`AtPieces`' `StagedAt`) may
  read `ws[s]`'s calls inside `allCalls x ws`, which `JointP` has true and collision-free. It gives `Joint` at
  `PartAccepts` (`joint_of_jointP`), and `decompBP_of_jointP` is `decompB_of_joint` with the pinned checks: a
  composition proved under `JointP` lands on `DecompBP`.
* `vstarAcceptsV_boundP` and `ksLin2_route_stagedP`: `vstarAcceptsV_boundW` and `ksLin2_route_staged` at `DecompBP`.
  The route reads each part's check at the witness `E_O` extracted for that part, so it never needed more: `PartAcceptsP`
  at `ws[s]` is built there from the same facts `PartChecksBV` was applied to, and no `PartChecksBV` is asked.
-/

namespace Claims.Recursion.VStarB

open Claims

section Facts

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- **What part `s`'s circuit accepts at the outer witness `w`**, read at the inner messages `ms`: `PartAccepts` with
its witness given. -/
def PartAcceptsP (H : Msg → Msg) (D : Deploy Coin) (DO : Flock.Deploy Coin Col T) (L : LayoutB)
    (shape : Form → Prop) (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) (cmt : Msg → ℕ → Msg → Msg) (s : ℕ) (x : Msg)
    (c : ℕ → Coin) (w : Flock.Wit) (ms : List Msg) : Prop :=
  (∃ st : CompiledStatement, D.parse x = some st ∧ shape st.form) ∧
    ∃ xO : Msg, Flock.check DO.parse xO w = true ∧ (∀ e ∈ L.calls x s w, H e.1 = e.2) ∧
      (findColl (L.calls x s w ++ vcallsAt H D xO)).isSome = false ∧ shows s x xO c ∧
      (∀ i < L.rounds x, ∀ m, L.held x s w i = some m → ms.getD i [] = m) ∧
      (∀ i < L.rounds x, ∀ o m, (o, m) ∈ L.piecesOf x s w i → ((ms.getD i []).drop o).take m.length = m) ∧
      L.CmtAgrees cmt x xO ms ∧ L.RowsRead x s w ms

/-- **`DecompB` with each part's check at its own witness**: at a compiled statement of the shape, wherever part `s`'s
check holds at `ws[s]` and the parts' own messages, for every part, and the recorded calls are true, `IV` accepts the
parts' messages or two recorded calls collide. -/
def DecompBP (H : Msg → Msg) (D : Deploy Coin) (IV : Verifier Coin) (L : LayoutB) (shape : Form → Prop)
    (chk : ℕ → Msg → (ℕ → Coin) → Flock.Wit → List Msg → Prop) : Prop :=
  ∀ (x : Msg) (st : CompiledStatement), D.parse x = some st → shape st.form → ∀ (c : ℕ → Coin) (ws : List Flock.Wit),
    ws.length = D.nS x → (∀ e ∈ L.allCalls x ws, H e.1 = e.2) →
    (∀ s (hs : s < ws.length), chk s x c ws[s] (L.msgs x ws)) →
      (run H (replay (L.msgs x ws)) IV (xI D x) () fun i => c i.val).out = .acc ∨
        (findColl (L.allCalls x ws)).isSome

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem partAccepts_of_pin {H : Msg → Msg} {D : Deploy Coin} {DO : Flock.Deploy Coin Col T} {L : LayoutB}
    {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop} {cmt : Msg → ℕ → Msg → Msg} {s : ℕ} {x : Msg}
    {c : ℕ → Coin} {w : Flock.Wit} {ms : List Msg} (h : PartAcceptsP H D DO L shape shows cmt s x c w ms) :
    PartAccepts H D DO L shape shows cmt s x c ms := by
  obtain ⟨hst, xO, h1, h2, h3, h4, h5, h6, h7, h8⟩ := h
  exact ⟨hst, xO, w, h1, h2, h3, h4, h5, h6, h7, h8⟩

omit [Fintype Coin] [Nonempty Coin] in
/-- **`DecompB` at `PartAccepts` pins**: a check at the given witness is one at some witness. -/
theorem decompBP_of_decompB {H : Msg → Msg} {D : Deploy Coin} {DO : Flock.Deploy Coin Col T} {IV : Verifier Coin}
    {L : LayoutB} {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop} {cmt : Msg → ℕ → Msg → Msg}
    (hD : DecompB H D IV L shape (PartAccepts H D DO L shape shows cmt)) :
    DecompBP H D IV L shape (PartAcceptsP H D DO L shape shows cmt) :=
  fun x st hx hf c ws hl hT hC => hD x st hx hf c ws hl hT fun s hs => partAccepts_of_pin (hC s (by omega))

end Facts

end Claims.Recursion.VStarB

namespace RecDeploy.OpenLower

open Claims (Msg)
open FlockDeploy (Coin)
open Claims.Recursion.VStarB

/-- **`Joint` with each part's check at its own witness**: `nS` witnesses, their recorded calls true and free of
collisions, and part `s`'s check at `ws[s]` and the parts' own messages. -/
def JointP (H : Msg → Msg) (D : Claims.Recursion.Deploy Coin) (L : LayoutB)
    (chk : ℕ → Msg → (ℕ → Coin) → Claims.Flock.Wit → List Msg → Prop) (x : Msg) (c : ℕ → Coin)
    (ws : List Claims.Flock.Wit) : Prop :=
  ws.length = D.nS x ∧ (∀ e ∈ L.allCalls x ws, H e.1 = e.2) ∧ (Claims.findColl (L.allCalls x ws)).isSome = false ∧
    ∀ s (hs : s < ws.length), chk s x c ws[s] (L.msgs x ws)

theorem joint_of_jointP {Col T : Type} {H : Msg → Msg} {D : Claims.Recursion.Deploy Coin}
    {DO : Claims.Flock.Deploy Coin Col T} {L : LayoutB} {shape : Claims.Recursion.Form → Prop}
    {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop} {cmt : Msg → ℕ → Msg → Msg} {x : Msg} {c : ℕ → Coin}
    {ws : List Claims.Flock.Wit} (h : JointP H D L (PartAcceptsP H D DO L shape shows cmt) x c ws) :
    Joint H D L (PartAccepts H D DO L shape shows cmt) x c ws :=
  ⟨h.1, h.2.1, h.2.2.1, fun s hs => partAccepts_of_pin (h.2.2.2 s (lt_of_lt_of_eq hs h.1.symm))⟩

/-- **`decompB_of_joint` with the pinned checks**: `DecompBP` at `chkP` wherever `DecompB` holds at `chk` and every
part's `chkP` at its own witness gives every part's `chk` at the parts' own messages. -/
theorem decompBP_of_jointP {H : Msg → Msg} {D : Claims.Recursion.Deploy Coin} {IV : Claims.Verifier Coin}
    {L : LayoutB} {shape : Claims.Recursion.Form → Prop} {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop}
    {chkP : ℕ → Msg → (ℕ → Coin) → Claims.Flock.Wit → List Msg → Prop}
    (hle : ∀ (x : Msg) (st : Claims.Recursion.CompiledStatement), D.parse x = some st → shape st.form →
      ∀ (c : ℕ → Coin) (ws : List Claims.Flock.Wit), JointP H D L chkP x c ws →
        ∀ s < D.nS x, chk s x c (L.msgs x ws))
    (hD : DecompB H D IV L shape chk) : DecompBP H D IV L shape chkP := by
  intro x st hx hf c ws hl hT hC
  by_cases hcol : (Claims.findColl (L.allCalls x ws)).isSome = true
  · exact Or.inr hcol
  · exact hD x st hx hf c ws hl hT (hle x st hx hf c ws ⟨hl, hT, by simpa using hcol, hC⟩)

end RecDeploy.OpenLower

namespace RecDeploy

open Claims (Msg Out Prover run coinSeq replay findColl Collision)
open FlockDeploy (Coin)
open Claims.Recursion (Bound Inner)

/-- **`VStarAcceptsV` at V\*'s bound layout, each part's check at its own witness**: `vstarAcceptsV_boundW` at
`DecompBP` at `PartAcceptsP`, with no `PartChecksBV`: at each part, the facts `PartChecksBV` was applied to at
`E_O`'s witness `ws[s]` are `PartAcceptsP` there. -/
theorem vstarAcceptsV_boundP {W Col T : Type} (H : Msg → Msg) (I : Claims.Recursion.Inner Coin W)
    (ivScope : Msg → Bool) (b : Brings) (L : Claims.Recursion.VStarB.LayoutB) (DO : Claims.Flock.Deploy Coin Col T)
    (shape : Claims.Recursion.Form → Prop) (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop)
    (cmt : Msg → ℕ → Msg → Msg) (hHW : L.HeldWhole)
    (hHeld : Claims.Recursion.VStarB.HeldB (deployRec I.V ivScope { b with rd := L.readers }) I L shape)
    (hSt : StagedRuns H (deployRec I.V ivScope { b with rd := L.readers }) I L shape shows cmt)
    (hOpen : ∀ s, Claims.Recursion.VStarB.PartOpensB H (deployRec I.V ivScope { b with rd := L.readers }) DO L shape
      shows cmt s)
    (hDec : Claims.Recursion.VStarB.DecompBP H (deployRec I.V ivScope { b with rd := L.readers }) I.V L shape
      (Claims.Recursion.VStarB.PartAcceptsP H (deployRec I.V ivScope { b with rd := L.readers }) DO L shape shows cmt))
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
  have hchk : ∀ s (hsw : s < ws.length),
      Claims.Recursion.VStarB.PartAcceptsP H (deployRec I.V ivScope { b with rd := L.readers }) DO L shape shows cmt s x
        (Claims.coinSeq r) ws[s] (L.msgs x ws) := by
    intro s hsw
    have hs : s < (deployRec I.V ivScope { b with rd := L.readers }).nS x := ho.1 ▸ hsw
    have hts : ∀ e ∈ L.calls x s ws[s], H e.1 = e.2 :=
      fun e he => htrue e (Claims.Recursion.VStar.mem_calls_of_part hsw he)
    have hshs := hsh s hs
    rcases hOpen s x st hst hf _ ws[s] (ho.2 s hsw) hts with hop | hop
    swap
    · exact absurd (Claims.Recursion.findColl_isSome_of_sub H (Claims.Recursion.VStarB.vcallsAt_append_true hts)
        (fun _ he => List.mem_append_left _ he) hop) (hpc' s hsw)
    refine ⟨⟨st, hst, hf⟩, _, ho.2 s hsw, hts, Bool.eq_false_iff.2 (hpc' s hsw), hshs.1, fun i hi m hm => ?_,
      fun i hi o m hm => ?_, ?_, ?_⟩
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

/-- **`KSLin2` at the deployed records' dense statements, each part's check at its own witness**:
`ksLin2_route_staged` with `PartChecksBV` and `DecompB` replaced by `DecompBP` at `PartAcceptsP`
(`vstarAcceptsV_boundP`). -/
theorem ksLin2_route_stagedP (rel relO : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings)
    (L : Claims.Recursion.VStarB.LayoutB)
    (hRows : L.ReadsRows FlockDeploy.sha512) (hHW : L.HeldWhole) (Cp : Claims.Recursion.Caps Coin) :
    ∀ (nextE : Msg → List Msg → List Coin → Claims.OC Out)
      (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
      (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ),
      let Bd := bdDeployed rel nextE colsE wE join rowIn final w
      let D := recBound rel b L Bd.IVStar
      let bL : Brings := { b with rd := L.readers }
      Claims.Recursion.KSFlockLin FlockDeploy.sha512 Bd.DE Cp → Bd.Binds → Bd.LastBindsEC FlockDeploy.sha512 →
      Bd.LastCoinFree FlockDeploy.sha512 →
      Claims.Flock.KS FlockDeploy.sha512 (deployZkRec relO) →
      (∀ shape ∈ [Claims.Recursion.Form.Dense],
        ∃ (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) (cmt : Msg → ℕ → Msg → Msg),
          (∀ decI, Claims.Recursion.VStarB.HeldB D (Claims.Recursion.Inner.ofFlock Bd.deploy decI) L shape) ∧
          (∀ decI, Claims.Recursion.VStarB.StagedB FlockDeploy.sha512 D
            (Claims.Recursion.Inner.ofFlock Bd.deploy decI) L shape shows cmt) ∧
          (∀ s, Claims.Recursion.VStarB.PartOpensB FlockDeploy.sha512 D (deployZkRec relO) L shape shows cmt s) ∧
          Claims.Recursion.VStarB.DecompBP FlockDeploy.sha512 D Bd.IVStar L shape
            (Claims.Recursion.VStarB.PartAcceptsP FlockDeploy.sha512 D (deployZkRec relO) L shape shows cmt)) →
      ReadersTrue FlockDeploy.sha512 Bd.IVStar bL (deployZkRec relO) →
      Claims.Recursion.KSLin2On FlockDeploy.sha512 (recBoundAt rel b L) Bd (deployZkRec relO) Cp DenseAt := by
  intro nextE colsE wE join rowIn final w Bd D bL hKS hBinds hEC hCF hF hshapes hT decI hdI decO hdO x hx Ω _ _ P M R
  let I := Claims.Recursion.Inner.ofFlock Bd.deploy decI
  let ivScope := (FlockDeploy.deployPlain rel).scopeCheck
  have hacc : ∀ shape ∈ [Claims.Recursion.Form.Dense],
      Claims.Recursion.VStarAcceptsV FlockDeploy.sha512 (deployRec I.V ivScope bL) I (deployZkRec relO) shape := by
    intro shape hs
    obtain ⟨shows, cmt, hHe, hSt, hOp, hDc⟩ := hshapes shape hs
    exact vstarAcceptsV_boundP FlockDeploy.sha512 I ivScope b L (deployZkRec relO) shape shows cmt hHW (hHe decI)
      (hSt decI) hOp hDc hRows
  have hTc := callsTrue FlockDeploy.sha512 I ivScope bL (deployZkRec relO) hT
  have hA := Claims.Recursion.innerAcceptsCollAtV_of_dense (refuses I.V ivScope bL FlockDeploy.sha512) hTc
    (hacc _ (by simp)) hx
  have hLc := callsLogCollAt_deployRec FlockDeploy.sha512 I ivScope bL (deployZkRec relO)
  obtain ⟨shows, cmt, hHe, hSt, hOp, -⟩ := hshapes _ (List.mem_singleton_self _)
  have hFk := forksBindAt_C_pick_setup FlockDeploy.sha512 Bd decI ivScope bL (deployZkRec relO) L.pick hEC hCF hTc hLc x
    (fun decO Ω P ω r c ws hg => by
      obtain ⟨st, hst, -⟩ := refuses I.V ivScope bL FlockDeploy.sha512 x Ω P ω r hg.1
      exact readers_run FlockDeploy.sha512 I ivScope b L (deployZkRec relO) _ shows cmt hHW (hHe decI) hOp hst
        (hx st hst) (hSt decI x st hst (hx st hst) Ω P ω r hg.1) (Claims.Recursion.eoGives_outerHolds hg)
        (hTc decO x Ω P ω r c ws hg)) hA
  have hdecC : ∀ x c, FlockDeploy.decCoin x (Bd.ans c) = c := fun _ c =>
    (congrArg (FlockDeploy.decCoin _) (bdDeployed_ans rel nextE colsE wE join rowIn final w c)).trans
      (Function.leftInverse_invFun Claims.Bridge.coin_bytes_injective c)
  have hM := roundsMargin_deployRec FlockDeploy.sha512 I ivScope bL fun x => V_k_pos_flock false x
  have hLn := callsLogCollAt_C FlockDeploy.sha512 Bd decI ivScope bL (deployZkRec relO) hCF hLc
  have hAn := innerAcceptsCollAtV_C FlockDeploy.sha512 Bd decI ivScope bL (deployZkRec relO) hCF x hA
  have hbody := Claims.Recursion.Bound.ksLin2_of_flockLin_binds (Bd := Bd) (decI := decI)
    (H := FlockDeploy.sha512) (D := deployRecC Bd ivScope bL) (DO := deployZkRec relO) (decO := decO) (P := P)
    (x := x) FlockDeploy.decCoin Cp hF hKS hdI hdO hdecC hBinds (refuses I.V ivScope bL FlockDeploy.sha512)
    (innerRounds I ivScope bL) (sessionNums I.V ivScope bL relO) (tableErrNonneg I.V ivScope bL)
    (sessionRounds I ivScope bL relO) (sessionAccepts I.V ivScope bL) (sessionRuns_sha512 I ivScope bL relO)
    (innerViewRun_C FlockDeploy.sha512 Bd decI ivScope bL (deployZkRec relO)) hTc hLn hFk hAn hM
    (callsNil I.V ivScope bL FlockDeploy.sha512) M R
  exact ksLin2_of_C_setup FlockDeploy.sha512 Bd decI ivScope bL (deployZkRec relO) decO hCF Cp x P M R hbody

end RecDeploy
