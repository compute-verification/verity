import Proofs.Flock.Claims.Recursion.PartAcceptsPin

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
