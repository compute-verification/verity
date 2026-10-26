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
  (∃ st : Stmt, D.parse x = some st ∧ shape st.form) ∧
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
  ∀ (x : Msg) (st : Stmt), D.parse x = some st → shape st.form → ∀ (c : ℕ → Coin) (ws : List Flock.Wit),
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
    (hle : ∀ (x : Msg) (st : Claims.Recursion.Stmt), D.parse x = some st → shape st.form →
      ∀ (c : ℕ → Coin) (ws : List Claims.Flock.Wit), JointP H D L chkP x c ws →
        ∀ s < D.nS x, chk s x c (L.msgs x ws))
    (hD : DecompB H D IV L shape chk) : DecompBP H D IV L shape chkP := by
  intro x st hx hf c ws hl hT hC
  by_cases hcol : (Claims.findColl (L.allCalls x ws)).isSome = true
  · exact Or.inr hcol
  · exact hD x st hx hf c ws hl hT (hle x st hx hf c ws ⟨hl, hT, by simpa using hcol, hC⟩)

end RecDeploy.OpenLower
