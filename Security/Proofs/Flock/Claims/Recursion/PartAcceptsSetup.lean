import Proofs.Flock.Claims.Recursion.KSLin2LayR

/-!
# `DecompB` and `Joint` with the verifier's own evaluations in the escape (`DecompBV`, `JointV`)

`DecompBP`'s collision outcome and `JointP`'s no-collision conjunct search the parts' recorded calls only,
`L.allCalls x ws`. The compiled verifier makes SHA-512 evaluations of its own at each V\* statement (`Deploy.setup`):
check 5's row commitments, and each registered read's leaf and its climb to the registered root
(`Flock.Registered.opens`, listed by `RecDeploy.RegSetup`). Two parts' commit strings are tied only through those climbs,
so a lowering that equates them (a level's `out` row and RecCap's `top` row, row E's chain slots) needs the escape to
search them too. Route 2C already charges them (`VStarAcceptsV`'s escape is `D.calls x ws ++ setupCalls H D xs` at the
run's V\* statements, and `lamLin2`'s `v` bounds `setupCalls`), so nothing downstream changes.

* `PartAcceptsV … s x c xO w ms`: `PartAcceptsP` with the V\* statement `xO` given too.
* `DecompBV`: `DecompBP` at a list of V\* statements `xs`, part `s`'s check read at `xs.getD s []` and `ws[s]`, and the
  escape `findColl (L.allCalls x ws ++ setupCalls H D xs)`. `DecompBP` gives it (`decompBV_of_decompBP`).
* `JointV`: `JointP` with the same statements and the same widened no-collision conjunct. It gives `JointP`
  (`jointP_of_jointV`), and `decompBV_of_jointV` / `decompBV_of_jointVP` are `decompB_of_joint` /
  `decompBP_of_jointP` landing on `DecompBV`: a lowering asked under `JointV` may read a tie through the verifier's
  climbs.
* `vstarAcceptsV_boundV`, `ksLin2_route_stagedV`, `ksLin2_stepsC_atV`, `ksLin2_stepsC_layRV` and `ksLin2_deployedR_layRV`: the `P` routes at
  `DecompBV`, at the run's V\* statements (`runVstar`).
-/

namespace Claims.Recursion.VStarB

open Claims

section Facts

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- **What part `s`'s circuit accepts at the V\* statement `xO` and the outer witness `w`**, read at the inner messages
`ms`: `PartAcceptsP` with its statement given. -/
def PartAcceptsV (H : Msg → Msg) (D : Deploy Coin) (DO : Flock.Deploy Coin Col T) (L : LayoutB)
    (shape : Form → Prop) (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) (cmt : Msg → ℕ → Msg → Msg) (s : ℕ) (x : Msg)
    (c : ℕ → Coin) (xO : Msg) (w : Flock.Wit) (ms : List Msg) : Prop :=
  (∃ st : Stmt, D.parse x = some st ∧ shape st.form) ∧
    Flock.check DO.parse xO w = true ∧ (∀ e ∈ L.calls x s w, H e.1 = e.2) ∧
      (findColl (L.calls x s w ++ vcallsAt H D xO)).isSome = false ∧ shows s x xO c ∧
      (∀ i < L.rounds x, ∀ m, L.held x s w i = some m → ms.getD i [] = m) ∧
      (∀ i < L.rounds x, ∀ o m, (o, m) ∈ L.piecesOf x s w i → ((ms.getD i []).drop o).take m.length = m) ∧
      L.CmtAgrees cmt x xO ms ∧ L.RowsRead x s w ms

/-- **`DecompBP` at the V\* statements `xs`, with the verifier's evaluations there in the escape**: at a compiled
statement of the shape, wherever part `s`'s check holds at `xs.getD s []`, `ws[s]` and the parts' own messages, for
every part, and the recorded calls are true, `IV` accepts the parts' messages, or two of the recorded calls and the
compiled verifier's own evaluations at `xs` (`setupCalls`) collide. -/
def DecompBV (H : Msg → Msg) (D : Deploy Coin) (IV : Verifier Coin) (L : LayoutB) (shape : Form → Prop)
    (chk : ℕ → Msg → (ℕ → Coin) → Msg → Flock.Wit → List Msg → Prop) : Prop :=
  ∀ (x : Msg) (st : Stmt), D.parse x = some st → shape st.form → ∀ (c : ℕ → Coin) (xs : List Msg)
    (ws : List Flock.Wit), ws.length = D.nS x → (∀ e ∈ L.allCalls x ws, H e.1 = e.2) →
    (∀ s (hs : s < ws.length), chk s x c (xs.getD s []) ws[s] (L.msgs x ws)) →
      (run H (replay (L.msgs x ws)) IV (xI D x) () fun i => c i.val).out = .acc ∨
        (findColl (L.allCalls x ws ++ setupCalls H D xs)).isSome

/-- **`DecompBV` under a side condition `ok` on the statements `xs`**: asked only where `ok x xs ws` holds. KS reads it
at an accepted run's V\* statements, staged at one prefix of the run's messages (`OkRuns`). -/
def DecompBVS (H : Msg → Msg) (D : Deploy Coin) (IV : Verifier Coin) (L : LayoutB) (shape : Form → Prop)
    (chk : ℕ → Msg → (ℕ → Coin) → Msg → Flock.Wit → List Msg → Prop)
    (ok : Msg → List Msg → List Flock.Wit → Prop) : Prop :=
  ∀ (x : Msg) (st : Stmt), D.parse x = some st → shape st.form → ∀ (c : ℕ → Coin) (xs : List Msg)
    (ws : List Flock.Wit), ws.length = D.nS x → (∀ e ∈ L.allCalls x ws, H e.1 = e.2) →
    (∀ s (hs : s < ws.length), chk s x c (xs.getD s []) ws[s] (L.msgs x ws)) → ok x xs ws →
      (run H (replay (L.msgs x ws)) IV (xI D x) () fun i => c i.val).out = .acc ∨
        (findColl (L.allCalls x ws ++ setupCalls H D xs)).isSome

omit [Fintype Coin] [Nonempty Coin] in
theorem decompBVS_of_decompBV {H : Msg → Msg} {D : Deploy Coin} {IV : Verifier Coin} {L : LayoutB}
    {shape : Form → Prop} {chk : ℕ → Msg → (ℕ → Coin) → Msg → Flock.Wit → List Msg → Prop}
    {ok : Msg → List Msg → List Flock.Wit → Prop} (h : DecompBV H D IV L shape chk) :
    DecompBVS H D IV L shape chk ok :=
  fun x st hx hf c xs ws hl hT hC _ => h x st hx hf c xs ws hl hT hC

omit [Fintype Coin] [Nonempty Coin] in
theorem decompBV_of_decompBVS {H : Msg → Msg} {D : Deploy Coin} {IV : Verifier Coin} {L : LayoutB}
    {shape : Form → Prop} {chk : ℕ → Msg → (ℕ → Coin) → Msg → Flock.Wit → List Msg → Prop}
    (h : DecompBVS H D IV L shape chk fun _ _ _ => True) : DecompBV H D IV L shape chk :=
  fun x st hx hf c xs ws hl hT hC => h x st hx hf c xs ws hl hT hC trivial

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem partAcceptsP_of_V {H : Msg → Msg} {D : Deploy Coin} {DO : Flock.Deploy Coin Col T} {L : LayoutB}
    {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop} {cmt : Msg → ℕ → Msg → Msg} {s : ℕ} {x : Msg}
    {c : ℕ → Coin} {xO : Msg} {w : Flock.Wit} {ms : List Msg}
    (h : PartAcceptsV H D DO L shape shows cmt s x c xO w ms) : PartAcceptsP H D DO L shape shows cmt s x c w ms := by
  obtain ⟨hst, h1, h2, h3, h4, h5, h6, h7, h8⟩ := h
  exact ⟨hst, xO, h1, h2, h3, h4, h5, h6, h7, h8⟩

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
/-- The recorded calls and the verifier's evaluations are true together. -/
theorem allCalls_setup_true {H : Msg → Msg} {D : Deploy Coin} {L : LayoutB} {x : Msg} {ws : List Flock.Wit}
    (hT : ∀ e ∈ L.allCalls x ws, H e.1 = e.2) (xs : List Msg) :
    ∀ e ∈ L.allCalls x ws ++ setupCalls H D xs, H e.1 = e.2 :=
  fun e he => (List.mem_append.1 he).elim (hT e) (setupCalls_true H D xs e)

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
/-- A collision among the recorded calls is one among them and the verifier's evaluations. -/
theorem coll_setup_of_coll {H : Msg → Msg} {D : Deploy Coin} {L : LayoutB} {x : Msg} {ws : List Flock.Wit}
    (hT : ∀ e ∈ L.allCalls x ws, H e.1 = e.2) (xs : List Msg) (h : (findColl (L.allCalls x ws)).isSome) :
    (findColl (L.allCalls x ws ++ setupCalls H D xs)).isSome :=
  findColl_isSome_of_sub H (allCalls_setup_true hT xs) (fun _ he => List.mem_append_left _ he) h

omit [Fintype Coin] [Nonempty Coin] in
/-- **`DecompBP` gives `DecompBV`**: a check at a given statement is one at some statement, and a collision among the
recorded calls is one among them and the verifier's evaluations. -/
theorem decompBV_of_decompBP {H : Msg → Msg} {D : Deploy Coin} {DO : Flock.Deploy Coin Col T} {IV : Verifier Coin}
    {L : LayoutB} {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop} {cmt : Msg → ℕ → Msg → Msg}
    (hD : DecompBP H D IV L shape (PartAcceptsP H D DO L shape shows cmt)) :
    DecompBV H D IV L shape (PartAcceptsV H D DO L shape shows cmt) :=
  fun x st hx hf c xs ws hl hT hC =>
    (hD x st hx hf c ws hl hT fun s hs => partAcceptsP_of_V (hC s hs)).imp_right (coll_setup_of_coll hT xs)

end Facts

end Claims.Recursion.VStarB

namespace RecDeploy.OpenLower

open Claims (Msg)
open FlockDeploy (Coin)
open Claims.Recursion.VStarB

/-- **`JointP` at the V\* statements `xs`, free of collisions with the verifier's evaluations there**: `nS`
witnesses, their recorded calls true, no collision among them and the compiled verifier's own evaluations at `xs`
(`setupCalls`), and part `s`'s check at `xs.getD s []`, `ws[s]` and the parts' own messages. -/
def JointV (H : Msg → Msg) (D : Claims.Recursion.Deploy Coin) (L : LayoutB)
    (chk : ℕ → Msg → (ℕ → Coin) → Msg → Claims.Flock.Wit → List Msg → Prop) (x : Msg) (c : ℕ → Coin)
    (xs : List Msg) (ws : List Claims.Flock.Wit) : Prop :=
  ws.length = D.nS x ∧ (∀ e ∈ L.allCalls x ws, H e.1 = e.2) ∧
    (Claims.findColl (L.allCalls x ws ++ Claims.Recursion.setupCalls H D xs)).isSome = false ∧
    ∀ s (hs : s < ws.length), chk s x c (xs.getD s []) ws[s] (L.msgs x ws)

theorem jointP_of_jointV {Col T : Type} {H : Msg → Msg} {D : Claims.Recursion.Deploy Coin}
    {DO : Claims.Flock.Deploy Coin Col T} {L : LayoutB} {shape : Claims.Recursion.Form → Prop}
    {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop} {cmt : Msg → ℕ → Msg → Msg} {x : Msg} {c : ℕ → Coin}
    {xs : List Msg} {ws : List Claims.Flock.Wit} (h : JointV H D L (PartAcceptsV H D DO L shape shows cmt) x c xs ws) :
    JointP H D L (PartAcceptsP H D DO L shape shows cmt) x c ws :=
  ⟨h.1, h.2.1, findColl_false_of_append (allCalls_setup_true h.2.1 xs) h.2.2.1,
    fun s hs => partAcceptsP_of_V (h.2.2.2 s hs)⟩

/-- **`JointV` is monotone in what each part shows.** -/
theorem jointV_shows {Col T : Type} {H : Msg → Msg} {D : Claims.Recursion.Deploy Coin}
    {DO : Claims.Flock.Deploy Coin Col T} {L : LayoutB} {shape : Claims.Recursion.Form → Prop}
    {shows shows' : ℕ → Msg → Msg → (ℕ → Coin) → Prop} {cmt : Msg → ℕ → Msg → Msg} {x : Msg} {c : ℕ → Coin}
    {xs : List Msg} {ws : List Claims.Flock.Wit} (hw : ∀ {s x xO c}, shows s x xO c → shows' s x xO c)
    (h : JointV H D L (PartAcceptsV H D DO L shape shows cmt) x c xs ws) :
    JointV H D L (PartAcceptsV H D DO L shape shows' cmt) x c xs ws := by
  refine ⟨h.1, h.2.1, h.2.2.1, fun s hs => ?_⟩
  obtain ⟨hst, h1, h2, h3, h4, h5⟩ := h.2.2.2 s hs
  exact ⟨hst, h1, h2, h3, hw h4, h5⟩

/-- **`decompB_of_joint` landing on `DecompBV`**: `DecompBV` at `chkV` wherever `DecompB` holds at `chk` and, under
`JointV` at `chkV`, every part's `chk` holds at the parts' own messages. A collision among the recorded calls and the
verifier's evaluations is `DecompBV`'s own second outcome, so `JointV` may assume none. -/
theorem decompBV_of_jointV {H : Msg → Msg} {D : Claims.Recursion.Deploy Coin} {IV : Claims.Verifier Coin}
    {L : LayoutB} {shape : Claims.Recursion.Form → Prop} {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop}
    {chkV : ℕ → Msg → (ℕ → Coin) → Msg → Claims.Flock.Wit → List Msg → Prop}
    (hle : ∀ (x : Msg) (st : Claims.Recursion.Stmt), D.parse x = some st → shape st.form →
      ∀ (c : ℕ → Coin) (xs : List Msg) (ws : List Claims.Flock.Wit), JointV H D L chkV x c xs ws →
        ∀ s < D.nS x, chk s x c (L.msgs x ws))
    (hD : DecompB H D IV L shape chk) : DecompBV H D IV L shape chkV := by
  intro x st hx hf c xs ws hl hT hC
  by_cases hcol : (Claims.findColl (L.allCalls x ws ++ Claims.Recursion.setupCalls H D xs)).isSome = true
  · exact Or.inr hcol
  · exact (hD x st hx hf c ws hl hT (hle x st hx hf c xs ws ⟨hl, hT, by simpa using hcol, hC⟩)).imp_right
      (coll_setup_of_coll hT xs)

/-- **`decompBV_of_jointV` under a side condition**: `DecompBVS` at `ok` wherever every part's `chk` holds under
`JointV` and `ok`. -/
theorem decompBVS_of_jointV {H : Msg → Msg} {D : Claims.Recursion.Deploy Coin} {IV : Claims.Verifier Coin}
    {L : LayoutB} {shape : Claims.Recursion.Form → Prop} {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop}
    {chkV : ℕ → Msg → (ℕ → Coin) → Msg → Claims.Flock.Wit → List Msg → Prop}
    {ok : Msg → List Msg → List Claims.Flock.Wit → Prop}
    (hle : ∀ (x : Msg) (st : Claims.Recursion.Stmt), D.parse x = some st → shape st.form →
      ∀ (c : ℕ → Coin) (xs : List Msg) (ws : List Claims.Flock.Wit), JointV H D L chkV x c xs ws → ok x xs ws →
        ∀ s < D.nS x, chk s x c (L.msgs x ws))
    (hD : DecompB H D IV L shape chk) : DecompBVS H D IV L shape chkV ok := by
  intro x st hx hf c xs ws hl hT hC hok
  by_cases hcol : (Claims.findColl (L.allCalls x ws ++ Claims.Recursion.setupCalls H D xs)).isSome = true
  · exact Or.inr hcol
  · exact (hD x st hx hf c ws hl hT (hle x st hx hf c xs ws ⟨hl, hT, by simpa using hcol, hC⟩ hok)).imp_right
      (coll_setup_of_coll hT xs)

/-- **`decompBP_of_jointP` landing on `DecompBV`**: the same from `DecompBP` at `chkP`, every part's `chkP` read at
its own witness. -/
theorem decompBV_of_jointVP {H : Msg → Msg} {D : Claims.Recursion.Deploy Coin} {IV : Claims.Verifier Coin}
    {L : LayoutB} {shape : Claims.Recursion.Form → Prop}
    {chkP : ℕ → Msg → (ℕ → Coin) → Claims.Flock.Wit → List Msg → Prop}
    {chkV : ℕ → Msg → (ℕ → Coin) → Msg → Claims.Flock.Wit → List Msg → Prop}
    (hle : ∀ (x : Msg) (st : Claims.Recursion.Stmt), D.parse x = some st → shape st.form →
      ∀ (c : ℕ → Coin) (xs : List Msg) (ws : List Claims.Flock.Wit), JointV H D L chkV x c xs ws →
        ∀ s (hs : s < ws.length), chkP s x c ws[s] (L.msgs x ws))
    (hD : DecompBP H D IV L shape chkP) : DecompBV H D IV L shape chkV := by
  intro x st hx hf c xs ws hl hT hC
  by_cases hcol : (Claims.findColl (L.allCalls x ws ++ Claims.Recursion.setupCalls H D xs)).isSome = true
  · exact Or.inr hcol
  · exact (hD x st hx hf c ws hl hT (hle x st hx hf c xs ws ⟨hl, hT, by simpa using hcol, hC⟩)).imp_right
      (coll_setup_of_coll hT xs)

end RecDeploy.OpenLower

namespace RecDeploy

open Claims (Msg Out Prover run coinSeq replay findColl Collision)
open FlockDeploy (Coin FinalP nextR)
open Claims.Recursion (Bound Inner)

/-- **`ok` at every accepted run's V\* statements**, at any outer witnesses of the parts' count. -/
def OkRuns {W : Type} (H : Msg → Msg) (D : Claims.Recursion.Deploy Coin) (I : Claims.Recursion.Inner Coin W)
    (shape : Claims.Recursion.Form → Prop) (ok : Msg → List Msg → List Claims.Flock.Wit → Prop) : Prop :=
  ∀ (x : Msg) (st : Claims.Recursion.Stmt), D.parse x = some st → shape st.form →
    ∀ (Ω : Type) (P : Prover Ω) (ω : Ω) (r : Fin (D.V.k x) → Coin), (run H P D.V x ω r).out = .acc →
      ∀ ws : List Claims.Flock.Wit, ws.length = D.nS x → ok x (Claims.Recursion.runVstar H D I P x ω r) ws

/-- **`VStarAcceptsV` at V\*'s bound layout from `DecompBVS`**: `vstarAcceptsV_boundP` with each part's check at the
run's V\* statement for it, `ok` there (`OkRuns`), and the escape over the recorded calls and the verifier's
evaluations at the run's V\* statements, which is `VStarAcceptsV`'s own. -/
theorem vstarAcceptsV_boundVS {W Col T : Type} (H : Msg → Msg) (I : Claims.Recursion.Inner Coin W)
    (ivScope : Msg → Bool) (b : Brings) (L : Claims.Recursion.VStarB.LayoutB) (DO : Claims.Flock.Deploy Coin Col T)
    (shape : Claims.Recursion.Form → Prop) (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop)
    (cmt : Msg → ℕ → Msg → Msg) (ok : Msg → List Msg → List Claims.Flock.Wit → Prop) (hHW : L.HeldWhole)
    (hHeld : Claims.Recursion.VStarB.HeldB (deployRec I.V ivScope { b with rd := L.readers }) I L shape)
    (hSt : StagedRuns H (deployRec I.V ivScope { b with rd := L.readers }) I L shape shows cmt)
    (hOpen : ∀ s, Claims.Recursion.VStarB.PartOpensB H (deployRec I.V ivScope { b with rd := L.readers }) DO L shape
      shows cmt s)
    (hDec : Claims.Recursion.VStarB.DecompBVS H (deployRec I.V ivScope { b with rd := L.readers }) I.V L shape
      (Claims.Recursion.VStarB.PartAcceptsV H (deployRec I.V ivScope { b with rd := L.readers }) DO L shape shows cmt)
      ok)
    (hok : OkRuns H (deployRec I.V ivScope { b with rd := L.readers }) I shape ok)
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
      Claims.Recursion.VStarB.PartAcceptsV H (deployRec I.V ivScope { b with rd := L.readers }) DO L shape shows cmt s x
        (Claims.coinSeq r)
        ((Claims.Recursion.runVstar H (deployRec I.V ivScope { b with rd := L.readers }) I P x ω r).getD s [])
        ws[s] (L.msgs x ws) := by
    intro s hsw
    have hs : s < (deployRec I.V ivScope { b with rd := L.readers }).nS x := ho.1 ▸ hsw
    have hts : ∀ e ∈ L.calls x s ws[s], H e.1 = e.2 :=
      fun e he => htrue e (Claims.Recursion.VStar.mem_calls_of_part hsw he)
    have hshs := hsh s hs
    rcases hOpen s x st hst hf _ ws[s] (ho.2 s hsw) hts with hop | hop
    swap
    · exact absurd (Claims.Recursion.findColl_isSome_of_sub H (Claims.Recursion.VStarB.vcallsAt_append_true hts)
        (fun _ he => List.mem_append_left _ he) hop) (hpc' s hsw)
    refine ⟨⟨st, hst, hf⟩, ho.2 s hsw, hts, Bool.eq_false_iff.2 (hpc' s hsw), hshs.1, fun i hi m hm => ?_,
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
  rcases hDec x st hst hf (Claims.coinSeq r)
      (Claims.Recursion.runVstar H (deployRec I.V ivScope { b with rd := L.readers }) I P x ω r) ws ho.1 htrue
      hchk (hok x st hst hf Ω P ω r hacc ws ho.1) with hd | hd
  · exact Or.inl hd
  · exact Or.inr (Claims.Recursion.VStar.coll_of_sub (fun e he => (List.mem_append.1 he).elim (htrue e) (hS e))
      (fun _ h => h) hd)

/-- **`VStarAcceptsV` at V\*'s bound layout from `DecompBV`**: `vstarAcceptsV_boundVS` with no side condition. -/
theorem vstarAcceptsV_boundV {W Col T : Type} (H : Msg → Msg) (I : Claims.Recursion.Inner Coin W)
    (ivScope : Msg → Bool) (b : Brings) (L : Claims.Recursion.VStarB.LayoutB) (DO : Claims.Flock.Deploy Coin Col T)
    (shape : Claims.Recursion.Form → Prop) (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop)
    (cmt : Msg → ℕ → Msg → Msg) (hHW : L.HeldWhole)
    (hHeld : Claims.Recursion.VStarB.HeldB (deployRec I.V ivScope { b with rd := L.readers }) I L shape)
    (hSt : StagedRuns H (deployRec I.V ivScope { b with rd := L.readers }) I L shape shows cmt)
    (hOpen : ∀ s, Claims.Recursion.VStarB.PartOpensB H (deployRec I.V ivScope { b with rd := L.readers }) DO L shape
      shows cmt s)
    (hDec : Claims.Recursion.VStarB.DecompBV H (deployRec I.V ivScope { b with rd := L.readers }) I.V L shape
      (Claims.Recursion.VStarB.PartAcceptsV H (deployRec I.V ivScope { b with rd := L.readers }) DO L shape shows cmt))
    (hRows : L.ReadsRows H) :
    Claims.Recursion.VStarAcceptsV H (deployRec I.V ivScope { b with rd := L.readers }) I DO shape :=
  vstarAcceptsV_boundVS H I ivScope b L DO shape shows cmt (fun _ _ _ => True) hHW hHeld hSt hOpen
    (Claims.Recursion.VStarB.decompBVS_of_decompBV hDec) (fun _ _ _ _ _ _ _ _ _ _ _ => trivial) hRows

/-- **`KSLin2` at the deployed records' dense statements from `DecompBVS`**: `ksLin2_route_stagedP` with `DecompBP`
replaced by `DecompBVS` at `PartAcceptsV` and a side condition `ok` every accepted run's V\* statements meet
(`vstarAcceptsV_boundVS`). -/
theorem ksLin2_route_stagedVS (rel relO : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings)
    (L : Claims.Recursion.VStarB.LayoutB)
    (hRows : L.ReadsRows FlockDeploy.sha512) (hHW : L.HeldWhole) (Cp : Claims.Recursion.Caps Coin)
    (ok : Msg → List Msg → List Claims.Flock.Wit → Prop) :
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
          (∀ decI, OkRuns FlockDeploy.sha512 D (Claims.Recursion.Inner.ofFlock Bd.deploy decI) shape ok) ∧
          Claims.Recursion.VStarB.DecompBVS FlockDeploy.sha512 D Bd.IVStar L shape
            (Claims.Recursion.VStarB.PartAcceptsV FlockDeploy.sha512 D (deployZkRec relO) L shape shows cmt) ok) →
      ReadersTrue FlockDeploy.sha512 Bd.IVStar bL (deployZkRec relO) →
      Claims.Recursion.KSLin2On FlockDeploy.sha512 (recBoundAt rel b L) Bd (deployZkRec relO) Cp DenseAt := by
  intro nextE colsE wE join rowIn final w Bd D bL hKS hBinds hEC hCF hF hshapes hT decI hdI decO hdO x hx Ω _ _ P M R
  let I := Claims.Recursion.Inner.ofFlock Bd.deploy decI
  let ivScope := (FlockDeploy.deployPlain rel).scopeCheck
  have hacc : ∀ shape ∈ [Claims.Recursion.Form.Dense],
      Claims.Recursion.VStarAcceptsV FlockDeploy.sha512 (deployRec I.V ivScope bL) I (deployZkRec relO) shape := by
    intro shape hs
    obtain ⟨shows, cmt, hHe, hSt, hOp, hOk, hDc⟩ := hshapes shape hs
    exact vstarAcceptsV_boundVS FlockDeploy.sha512 I ivScope b L (deployZkRec relO) shape shows cmt ok hHW (hHe decI)
      (hSt decI) hOp hDc (hOk decI) hRows
  have hTc := callsTrue FlockDeploy.sha512 I ivScope bL (deployZkRec relO) hT
  have hA := Claims.Recursion.innerAcceptsCollAtV_of_dense (refuses I.V ivScope bL FlockDeploy.sha512) hTc
    (hacc _ (by simp)) hx
  have hLc := callsLogCollAt_deployRec FlockDeploy.sha512 I ivScope bL (deployZkRec relO)
  obtain ⟨shows, cmt, hHe, hSt, hOp, -, -⟩ := hshapes _ (List.mem_singleton_self _)
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

/-- **`KSLin2` at the deployed records' dense statements from `DecompBV`**: `ksLin2_route_stagedVS` with no side
condition. -/
theorem ksLin2_route_stagedV (rel relO : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings)
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
          Claims.Recursion.VStarB.DecompBV FlockDeploy.sha512 D Bd.IVStar L shape
            (Claims.Recursion.VStarB.PartAcceptsV FlockDeploy.sha512 D (deployZkRec relO) L shape shows cmt)) →
      ReadersTrue FlockDeploy.sha512 Bd.IVStar bL (deployZkRec relO) →
      Claims.Recursion.KSLin2On FlockDeploy.sha512 (recBoundAt rel b L) Bd (deployZkRec relO) Cp DenseAt := by
  intro nextE colsE wE join rowIn final w Bd D bL hKS hBinds hEC hCF hF hshapes hT
  refine ksLin2_route_stagedVS rel relO b L hRows hHW Cp (fun _ _ _ => True) nextE colsE wE join rowIn final w hKS
    hBinds hEC hCF hF ?_ hT
  intro shape hs
  obtain ⟨shows, cmt, hHe, hSt, hOp, hDc⟩ := hshapes shape hs
  exact ⟨shows, cmt, hHe, hSt, hOp, fun _ _ _ _ _ _ _ _ _ _ _ _ => trivial,
    Claims.Recursion.VStarB.decompBVS_of_decompBV hDc⟩

section LayR

open Flock

/-- **`ksLin2_stepsC_atP` from `DecompBV` at `PartAcceptsV`**, through `ksLin2_route_stagedV`. The layout stays a
variable, so no step unfolds it. -/
theorem ksLin2_stepsC_atV (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (b : Brings)
    (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8)
    (L : Claims.Recursion.VStarB.LayoutB) (hRows : L.ReadsRows FlockDeploy.sha512) (hHW : L.HeldWhole)
    (wE : Msg → ℝ) (join : List Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (hB : StepsBaseC nextE colsE wE join final w) :
    let bR : Brings := { b with tr := recTr aux nOf hd recRest }
    let Bd := bdDeployed _root_.Flock.Readable.relation nextE colsE wE join rowInDeployed final w
    let D := recBound _root_.Flock.Readable.relation bR L Bd.IVStar
    ReadersTrue FlockDeploy.sha512 Bd.IVStar { bR with rd := L.readers } (deployZkRec _root_.Flock.Readable.relation) →
    (∀ decI, Claims.Recursion.VStarB.HeldB D (Claims.Recursion.Inner.ofFlock Bd.deploy decI) L
      Claims.Recursion.Form.Dense) →
    (∀ s, Claims.Recursion.VStarB.PartOpensB FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation) L
      Claims.Recursion.Form.Dense (showsPart recRead bR) (cmtTr recRead) s) →
    (∀ shape ∈ [Claims.Recursion.Form.Dense],
      Claims.Recursion.VStarB.DecompBV FlockDeploy.sha512 D Bd.IVStar L shape
        (Claims.Recursion.VStarB.PartAcceptsV FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation) L
          shape (showsPart recRead bR) (cmtTr recRead))) →
    Claims.Recursion.KSLin2On FlockDeploy.sha512 (recBoundAt _root_.Flock.Readable.relation bR L) Bd
      (deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep DenseAt := by
  intro bR Bd D hRT hHe hPO hD
  obtain ⟨hKS, hEC, hCF⟩ := hB
  have hrb : ReadsBackB FlockDeploy.sha512 Bd.IVStar { bR with rd := L.readers } (fun _ => True) recRead :=
    readsBack_recTr _ _ { bR with rd := L.readers } _ aux nOf hd recRest rfl
  refine ksLin2_route_stagedV _root_.Flock.Readable.relation _root_.Flock.Readable.relation bR L hRows hHW
    FlockDeploy.KS.CpDep nextE colsE wE join rowInDeployed final w hKS (binds_deployed _ _ _ wE join final w) hEC hCF
    FlockDeploy.KS.zk_KS_outer ?_ hRT
  intro shape hsh
  obtain rfl := List.mem_singleton.1 hsh
  exact ⟨showsPart recRead bR, cmtTr recRead, hHe,
    fun decI => stagedB_deployRec_part FlockDeploy.sha512 (Claims.Recursion.Inner.ofFlock Bd.deploy decI)
      (FlockDeploy.deployPlain _root_.Flock.Readable.relation).scopeCheck { bR with rd := L.readers } recRead L _
      (fun x st hst _ => hrb x st hst trivial) (hHe decI),
    hPO, hD _ hsh⟩

/-- **`ksLin2_stepsC_layRP` from `DecompBV` at `PartAcceptsV`**: `ksLin2_stepsC_atV` at `layR`. -/
theorem ksLin2_stepsC_layRV (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (b : Brings)
    (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8)
    (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE : Msg → ℝ)
    (join : List Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (hB : StepsBaseC nextE colsE wE join final w)
    (hM : MoreSlotsTrue (deployZkRec _root_.Flock.Readable.relation)
      ({ b with tr := recTr aux nOf hd recRest } : Brings).st) :
    let bR : Brings := { b with tr := recTr aux nOf hd recRest }
    let L := DeployedR.layR elem
    let Bd := bdDeployed _root_.Flock.Readable.relation nextE colsE wE join rowInDeployed final w
    let D := recBound _root_.Flock.Readable.relation bR L Bd.IVStar
    (∀ shape ∈ [Claims.Recursion.Form.Dense],
      Claims.Recursion.VStarB.DecompBV FlockDeploy.sha512 D Bd.IVStar L shape
        (Claims.Recursion.VStarB.PartAcceptsV FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation) L
          shape (showsPart recRead bR) (cmtTr recRead))) →
    Claims.Recursion.KSLin2On FlockDeploy.sha512 (recBoundAt _root_.Flock.Readable.relation bR L) Bd
      (deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep DenseAt :=
  fun hD => ksLin2_stepsC_atV nextE colsE b aux nOf hd recRest (DeployedR.layR elem) (readsRows_layR elem)
    (heldWhole_layR elem) wE join final w hB (readersTrue_layR elem _ _ _ hmSlotsCommit_of_full hM)
    (heldB_dense_layR elem nextE colsE _ wE join final w) (partOpensB_dense_layR elem _ _) hD

/-- **`KSLin2` at the deployed reader's steps, at the layout of record, from `DecompBV`**: `ksLin2_stepsC_layRV` at
`stepsBase_R`. `ksLin2_deployedR_layRP` is it at `decompBV_of_decompBP`. -/
theorem ksLin2_deployedR_layRV (rd : Msg → Option FinalP) (b : Brings)
    (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8)
    (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE : Msg → ℝ)
    (join : List Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (hIn : ∀ x, ¬ Claims.Flock.NoAcc FlockDeploy.sha512 (FlockDeploy.VR rd) x → FlockDeploy.KS.PlainRdInR rd x)
    (hM : MoreSlotsTrue (deployZkRec _root_.Flock.Readable.relation)
      ({ b with tr := recTr aux nOf hd recRest } : Brings).st) :
    let bR : Brings := { b with tr := recTr aux nOf hd recRest }
    let L := DeployedR.layR elem
    let Bd := bdDeployed _root_.Flock.Readable.relation (nextR rd)
      (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation rd).cols wE join rowInDeployed final w
    let D := recBound _root_.Flock.Readable.relation bR L Bd.IVStar
    (∀ shape ∈ [Claims.Recursion.Form.Dense],
      Claims.Recursion.VStarB.DecompBV FlockDeploy.sha512 D Bd.IVStar L shape
        (Claims.Recursion.VStarB.PartAcceptsV FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation) L
          shape (showsPart recRead bR) (cmtTr recRead))) →
    Claims.Recursion.KSLin2On FlockDeploy.sha512 (recBoundAt _root_.Flock.Readable.relation bR L) Bd
      (deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep DenseAt :=
  ksLin2_stepsC_layRV _ _ b aux nOf hd recRest elem wE join final w (stepsBase_R rd wE join final w hIn) hM

end LayR

end RecDeploy
