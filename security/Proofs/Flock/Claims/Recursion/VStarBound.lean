module

public import Proofs.Flock.Claims.Recursion.VStar
public import Flock.HmNets
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Topology.Sheaves.Init
import Proofs.Flock.Claims.Recursion.Transport
import Std.Tactic.BVDecide.Normalize.Prop

/-! V\*'s layout and part facts restated at the representation V\* binds (`LayoutB`), where every inner round, round 0 and the last included, is read off V\*'s parts. -/

@[expose] public section

namespace Claims.Recursion.VStarB

open Claims
open Claims.Recursion.VStar (readAt)

/-- An affine form's bit off a witness: the XOR of the hidden values at its wires, and its constant (`Hm.ev` at the
witness's hidden values). -/
def formBit (w : Flock.Wit) (f : _root_.Flock.HmNets.Lin) : Bool := f.s.foldr (fun q b => w.hid.getD q false ^^ b) f.c

/-- The bytes affine forms over the hidden values read off a witness, eight bits a byte, the low bit first. -/
def readForms (w : Flock.Wit) (fs : List _root_.Flock.HmNets.Lin) : Msg := VStar.bytesOf (fs.map (formBit w))

theorem readForms_pos (w : Flock.Wit) (ps : List ℕ) :
    readForms w (ps.map _root_.Flock.HmNets.Lin.var) = readAt w ps := by
  unfold readForms readAt
  congr 1
  simp [List.map_map, Function.comp_def, formBit, _root_.Flock.HmNets.Lin.var]

/-- A message off a witness from its segments: each segment's constant bytes, then the bytes its forms read. -/
def readSegs (w : Flock.Wit) (segs : List (Msg × List _root_.Flock.HmNets.Lin)) : Msg :=
  segs.flatMap fun p => p.1 ++ readForms w p.2

/-- A piece's byte length: each segment's constant bytes, then its forms' bytes, eight bits a byte. -/
def pieceLen (segs : List (Msg × List _root_.Flock.HmNets.Lin)) : ℕ :=
  (segs.map fun p => p.1.length + (p.2.length + 7) / 8).sum

def segsAt (segs : List (Msg × List ℕ)) : List (Msg × List _root_.Flock.HmNets.Lin) :=
  segs.map fun p => (p.1, p.2.map _root_.Flock.HmNets.Lin.var)

theorem readSegs_segsAt (w : Flock.Wit) (segs : List (Msg × List ℕ)) :
    readSegs w (segsAt segs) = VStar.readSegs w segs := by
  unfold readSegs segsAt VStar.readSegs
  induction segs with
  | nil => rfl
  | cons p segs ih => simp only [List.map_cons, List.flatMap_cons, ih, readForms_pos]

/-- **Where V\*'s parts hold the bound view** (option (d)), fixed by the compiled statement `x`: the inner rounds' count
(`rounds x`); where part `s` holds inner round `i`'s message whole (`msgAt x s i`, `none` where it holds none), as
segments, each constant bytes the statement fixes then affine forms over the hidden values for the next bits, in
order; each SHA-512 evaluation part `s` computes (`shaAt x s`): its input's constant prefix, then the forms of the
rest of its input's bits, and those of its output's; the rounds whose message the statement fixes (`fixedAt x i`),
which no part holds; and each round that no part holds whole, assembled from pieces (`pieceAt x i`, empty at a round
held whole or fixed), each a part and its segments, in order. A piece's bytes have the length its segments fix
(`pieceLen`), so each piece sits at an offset the statement fixes. A bit at a position is the one-wire form
(`segsAt`). V*'s shared registered rows, each read by more than one part (the running sums `acc0` and `acc1`, which
rec-open units write and rec-residuals units read), are slots: where part `s` holds shared row `r` (`rowAt x s r`, as
segments, `none` where it holds none), the one part whose witness the row's value is read off (`rowOwner x r`), and
where the inner messages carry the row, a round and a byte offset (`rowIn x r`, `none` where no message carries it). -/
structure LayoutB where
  rounds : Msg → ℕ
  msgAt : Msg → ℕ → ℕ → Option (List (Msg × List _root_.Flock.HmNets.Lin))
  shaAt : Msg → ℕ → List (Msg × List _root_.Flock.HmNets.Lin × List _root_.Flock.HmNets.Lin)
  fixedAt : Msg → ℕ → Option Msg := fun _ _ => none
  pieceAt : Msg → ℕ → List (ℕ × List (Msg × List _root_.Flock.HmNets.Lin)) := fun _ _ => []
  rowAt : Msg → ℕ → ℕ → Option (List (Msg × List _root_.Flock.HmNets.Lin)) := fun _ _ _ => none
  rowOwner : Msg → ℕ → Option ℕ := fun _ _ => none
  rowIn : Msg → ℕ → Option (ℕ × ℕ) := fun _ _ => none

/-- Inner round `i`'s message as part `s` holds it whole, off its witness `w`: none at a round the statement fixes. -/
def LayoutB.held (L : LayoutB) (x : Msg) (s : ℕ) (w : Flock.Wit) (i : ℕ) : Option Msg :=
  if (L.fixedAt x i).isSome then none else (L.msgAt x s i).map (readSegs w)

/-- Round `i` assembled off the witnesses: each piece read off its part's witness, in order. Zero bytes fill a piece
whose part has no witness, so the later pieces keep their offsets. -/
def LayoutB.assembled (L : LayoutB) (x : Msg) (ws : List Flock.Wit) (i : ℕ) : Msg :=
  (L.pieceAt x i).flatMap fun p => (ws[p.1]?.map fun w => readSegs w p.2).getD (List.replicate (pieceLen p.2) 0)

/-- Inner round `i`'s message off the witnesses: the statement's where it fixes it, its assembled message where it has
pieces, else as the first part that holds it whole holds it, empty where none does. -/
def LayoutB.msg (L : LayoutB) (x : Msg) (ws : List Flock.Wit) (i : ℕ) : Msg :=
  match L.fixedAt x i with
  | some m => m
  | none =>
    if (L.pieceAt x i).isEmpty then
      ((List.range ws.length).findSome? fun s => ws[s]?.bind fun w => L.held x s w i).getD []
    else L.assembled x ws i

/-- **The bound readers' messages**: the inner session's `rounds x` messages, each off the witnesses (`LayoutB.msg`). -/
def LayoutB.msgs (L : LayoutB) (x : Msg) (ws : List Flock.Wit) : List Msg :=
  (List.range (L.rounds x)).map (L.msg x ws)

/-- **The bound readers' calls**: part `s`'s SHA-512 evaluations off its witness `w`, each its input (its constant
prefix, then the bytes its forms read) and its output, read at their forms. -/
def LayoutB.calls (L : LayoutB) (x : Msg) (s : ℕ) (w : Flock.Wit) : List (Msg × Msg) :=
  (L.shaAt x s).map fun p => (p.1 ++ readForms w p.2.1, readForms w p.2.2)

def LayoutB.allCalls (L : LayoutB) (x : Msg) (ws : List Flock.Wit) : List (Msg × Msg) :=
  (ws.mapIdx fun s w => L.calls x s w).flatten

/-- Shared registered row `r` as part `s` holds it, off its witness `w`. -/
def LayoutB.rowHeld (L : LayoutB) (x : Msg) (s : ℕ) (w : Flock.Wit) (r : ℕ) : Option Msg :=
  (L.rowAt x s r).map (readSegs w)

def LayoutB.row (L : LayoutB) (x : Msg) (ws : List Flock.Wit) (r : ℕ) : Msg :=
  ((L.rowOwner x r).bind fun o => ws[o]?.bind fun w => L.rowHeld x o w r).getD []

def LayoutB.RowsAgree (L : LayoutB) (x : Msg) (ws : List Flock.Wit) : Prop :=
  ∀ (r s : ℕ) (w : Flock.Wit) (v : Msg), ws[s]? = some w → L.rowHeld x s w r = some v → v = L.row x ws r

theorem LayoutB.row_of_owner {L : LayoutB} {x : Msg} {ws : List Flock.Wit} {r o : ℕ} {w : Flock.Wit} {v : Msg}
    (ho : L.rowOwner x r = some o) (hw : ws[o]? = some w) (hv : L.rowHeld x o w r = some v) : L.row x ws r = v := by
  simp [LayoutB.row, ho, hw, hv]

theorem LayoutB.RowsAgree.held_eq {L : LayoutB} {x : Msg} {ws : List Flock.Wit} (h : L.RowsAgree x ws) {r s : ℕ}
    {w : Flock.Wit} {v : Msg} (hw : ws[s]? = some w) (hv : L.rowHeld x s w r = some v) : v = L.row x ws r :=
  h r s w v hw hv

theorem LayoutB.RowsAgree.held_eq_held {L : LayoutB} {x : Msg} {ws : List Flock.Wit} (h : L.RowsAgree x ws)
    {r s t : ℕ} {w u : Flock.Wit} {v v' : Msg} (hw : ws[s]? = some w) (hu : ws[t]? = some u)
    (hv : L.rowHeld x s w r = some v) (hv' : L.rowHeld x t u r = some v') : v = v' :=
  (h.held_eq hw hv).trans (h.held_eq hu hv').symm

theorem LayoutB.rowsAgree_of_none {L : LayoutB} {x : Msg} {ws : List Flock.Wit}
    (h : ∀ s r, L.rowAt x s r = none) : L.RowsAgree x ws := by
  intro r s w v _ hv
  simp [LayoutB.rowHeld, h s r] at hv

/-- **Part `s` reads the shared rows where the messages carry them**: each shared row it holds off its witness `w` sits
in the messages `ms` at the round and offset the layout places it (`rowIn`). -/
def LayoutB.RowsRead (L : LayoutB) (x : Msg) (s : ℕ) (w : Flock.Wit) (ms : List Msg) : Prop :=
  ∀ (r : ℕ) (v : Msg) (i o : ℕ), L.rowHeld x s w r = some v → L.rowIn x r = some (i, o) →
    ((ms.getD i []).drop o).take v.length = v

def LayoutB.RowsLaid (L : LayoutB) : Prop :=
  ∀ (x : Msg) (ws : List Flock.Wit) (r i o : ℕ), L.rowIn x r = some (i, o) →
    (((L.msgs x ws).getD i []).drop o).take (L.row x ws r).length = L.row x ws r

def LayoutB.ReadsRows (H : Msg → Msg) (L : LayoutB) : Prop :=
  ∀ (x : Msg) (ws : List Flock.Wit), (∀ e ∈ L.allCalls x ws, H e.1 = e.2) →
    ∀ (s : ℕ) (w : Flock.Wit), ws[s]? = some w → L.RowsRead x s w (L.msgs x ws)

theorem LayoutB.readsRows_of_rowIn {H : Msg → Msg} {L : LayoutB} (h : ∀ x r, L.rowIn x r = none) :
    L.ReadsRows H := by
  intro x ws _ s w _ r v i o _ hi
  simp [h] at hi

theorem LayoutB.readsRows_of_agree {H : Msg → Msg} {L : LayoutB} (hl : L.RowsLaid)
    (ha : ∀ x ws, (∀ e ∈ L.allCalls x ws, H e.1 = e.2) → L.RowsAgree x ws) : L.ReadsRows H := by
  intro x ws hT s w hw r v i o hv hi
  rw [ha x ws hT r s w v hw hv]
  exact hl x ws r i o hi

def LayoutB.ofLayout (L : VStar.Layout) : LayoutB where
  rounds := L.rounds
  msgAt x s i := (L.msgAt x s i).map segsAt
  shaAt x s := (L.shaAt x s).map fun p =>
    (p.1, p.2.1.map _root_.Flock.HmNets.Lin.var, p.2.2.map _root_.Flock.HmNets.Lin.var)
  fixedAt := L.fixedAt

theorem LayoutB.held_ofLayout (L : VStar.Layout) (x : Msg) (s : ℕ) (w : Flock.Wit) (i : ℕ) :
    (LayoutB.ofLayout L).held x s w i = L.held x s w i := by
  have hm : (LayoutB.ofLayout L).msgAt x s i = (L.msgAt x s i).map segsAt := rfl
  have hf : (LayoutB.ofLayout L).fixedAt x i = L.fixedAt x i := rfl
  have he : (readSegs w ∘ segsAt) = VStar.readSegs w := funext fun segs => readSegs_segsAt w segs
  unfold LayoutB.held VStar.Layout.held
  rw [hm, hf, Option.map_map, he]

theorem LayoutB.calls_ofLayout (L : VStar.Layout) (x : Msg) (s : ℕ) (w : Flock.Wit) :
    (LayoutB.ofLayout L).calls x s w = L.calls x s w := by
  simp [LayoutB.calls, VStar.Layout.calls, LayoutB.ofLayout, readForms_pos]

theorem LayoutB.rowsAgree_ofLayout (L : VStar.Layout) (x : Msg) (ws : List Flock.Wit) :
    (LayoutB.ofLayout L).RowsAgree x ws :=
  LayoutB.rowsAgree_of_none fun _ _ => rfl

/-- **Part `s`'s pieces of round `i`**, off its witness: each piece it holds, with its byte offset in the round. -/
def LayoutB.piecesOf (L : LayoutB) (x : Msg) (s : ℕ) (w : Flock.Wit) (i : ℕ) : List (ℕ × Msg) :=
  ((L.pieceAt x i).foldl (fun (acc : ℕ × List (ℕ × Msg)) p =>
    (acc.1 + pieceLen p.2, if p.1 = s then acc.2 ++ [(acc.1, readSegs w p.2)] else acc.2)) (0, [])).2

/-- **The compiled verifier's SHA-512 evaluations at a V\* statement `xO`**, each `(u, H u)`: its setup's there
(`Deploy.setup`), among them check 5's row commitments (`RecCoins.checkRows`: at each row port the statement registers,
the commitment the verifier recomputes from its own rows, the verifier's `v` at its coins or zero rows, under the
salt the statement carries). They are `setupCalls` at `xO` alone, so route 2C's finders search them (`sampleCalls`). -/
def vcallsAt {Coin : Type} (H : Msg → Msg) (D : Deploy Coin) (xO : Msg) : List (Msg × Msg) :=
  (D.setup xO).map fun u => (u, H u)

theorem vcallsAt_true {Coin : Type} (H : Msg → Msg) (D : Deploy Coin) (xO : Msg) :
    ∀ e ∈ vcallsAt H D xO, H e.1 = e.2 := by
  intro e he
  obtain ⟨u, -, rfl⟩ := List.mem_map.1 he
  rfl

theorem vcallsAt_append_true {Coin : Type} {H : Msg → Msg} {D : Deploy Coin} {xO : Msg} {l : List (Msg × Msg)}
    (hT : ∀ e ∈ l, H e.1 = e.2) : ∀ e ∈ l ++ vcallsAt H D xO, H e.1 = e.2 :=
  fun e he => (List.mem_append.1 he).elim (hT e) (vcallsAt_true H D xO e)

theorem vcallsAt_sub {Coin : Type} {H : Msg → Msg} {D : Deploy Coin} {xs : List Msg} {s : ℕ} (hs : s < xs.length) :
    ∀ e ∈ vcallsAt H D (xs.getD s []), e ∈ setupCalls H D xs := by
  intro e he
  have hx : xs.getD s [] ∈ xs := by
    rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hs, Option.getD_some]
    exact List.getElem_mem hs
  exact List.mem_flatMap.2 ⟨_, hx, he⟩

theorem findColl_false_of_append {H : Msg → Msg} {l l' : List (Msg × Msg)} (ht : ∀ e ∈ l ++ l', H e.1 = e.2)
    (h : (findColl (l ++ l')).isSome = false) : (findColl l).isSome = false :=
  Bool.eq_false_iff.2 fun hl => by
    rw [findColl_isSome_of_sub H ht (fun _ he => List.mem_append_left _ he) hl] at h
    exact absurd h (by decide)

theorem row_tie {Coin : Type} {H : Msg → Msg} {D : Deploy Coin} {l : List (Msg × Msg)} {xO : Msg}
    {frame : Msg → Msg → Msg} (hinj : ∀ v σ v' σ', frame v σ = frame v' σ' → v = v') {v σ v₀ σ₀ d : Msg}
    (hc : (frame v σ, d) ∈ l) (hT : ∀ e ∈ l, H e.1 = e.2) (hv : frame v₀ σ₀ ∈ D.setup xO)
    (hd : H (frame v₀ σ₀) = d) (hnc : (findColl (l ++ vcallsAt H D xO)).isSome = false) : v = v₀ := by
  refine hinj v σ v₀ σ₀ (Classical.byContradiction fun hne => ?_)
  have hm : (frame v₀ σ₀, H (frame v₀ σ₀)) ∈ vcallsAt H D xO := List.mem_map.2 ⟨_, hv, rfl⟩
  have hs := findColl_isSome_mem H (l ++ vcallsAt H D xO) (vcallsAt_append_true hT) (frame v σ) (frame v₀ σ₀)
    (List.mem_map.2 ⟨_, List.mem_append_left _ hc, rfl⟩) (List.mem_map.2 ⟨_, List.mem_append_right _ hm, rfl⟩)
    ⟨hne, (hT _ hc).trans hd.symm⟩
  rw [hs] at hnc
  exact absurd hnc (by decide)

theorem zero_tie {Coin : Type} {H : Msg → Msg} {D : Deploy Coin} {l : List (Msg × Msg)} {xO : Msg}
    {frame : Msg → Msg → Msg} (hinj : ∀ v σ v' σ', frame v σ = frame v' σ' → v = v') {v σ z σ₀ d : Msg}
    (hc : (frame v σ, d) ∈ l) (hT : ∀ e ∈ l, H e.1 = e.2) (hz : frame z σ₀ ∈ D.setup xO)
    (hd : H (frame z σ₀) = d) (hnc : (findColl (l ++ vcallsAt H D xO)).isSome = false) : v = z :=
  row_tie hinj hc hT hz hd hnc

section Facts

variable {Coin W Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- No part holds the last round whole and a fixed round has no pieces: either would let a part's check hold only off the readers' messages, which leaves `DecompB` empty. -/
def HeldB (D : Deploy Coin) (I : Inner Coin W) (L : LayoutB) (shape : Form → Prop) : Prop :=
  ∀ (x : Msg) (st : CompiledStatement), D.parse x = some st → shape st.form →
    I.V.k (xI D x) = L.rounds x ∧ (∀ i < I.V.k (xI D x),
      ((L.fixedAt x i).isSome ∨ (∃ s < D.nS x, (L.msgAt x s i).isSome) ∨
        (L.pieceAt x i ≠ [] ∧ ∀ p ∈ L.pieceAt x i, p.1 < D.nS x)) ∧
      ((L.fixedAt x i).isSome → L.pieceAt x i = [])) ∧
    ∀ s, L.msgAt x s (I.V.k (xI D x) - 1) = none

/-- The bytes round `i`'s pieces span. -/
def LayoutB.piecesLen (L : LayoutB) (x : Msg) (i : ℕ) : ℕ := ((L.pieceAt x i).map fun p => pieceLen p.2).sum

/-- **The readers' messages are the commit strings `xO` shows**, at each round before the last that the statement
doesn't fix, where `VstarShowsB` makes `cmt` the run's commit string: the whole string at a round with no pieces, and
the bytes the pieces span at an assembled round, which is all of it the readers read there (`LayoutB.pick`). A part
reads such a round through its own statement (a row it opens against the commit string `xO` shows) whether or not it
holds the round or a piece of it, so its check speaks of the readers' message only through this; and every part's
statement shows one string there, so the parts' statements agree on every assembled round's pieces. -/
def LayoutB.CmtAgrees (L : LayoutB) (cmt : Msg → ℕ → Msg → Msg) (x xO : Msg) (ms : List Msg) : Prop :=
  ∀ i, i + 1 < L.rounds x → L.fixedAt x i = none →
    (L.pieceAt x i = [] → ms.getD i [] = cmt x i xO) ∧
      (ms.getD i []).take (L.piecesLen x i) = (cmt x i xO).take (L.piecesLen x i)

theorem LayoutB.cmtAgrees_of_eq {L : LayoutB} {cmt : Msg → ℕ → Msg → Msg} {x xO : Msg} {ms : List Msg}
    (h : ∀ i, i + 1 < L.rounds x → L.fixedAt x i = none → ms.getD i [] = cmt x i xO) : L.CmtAgrees cmt x xO ms :=
  fun i hi hf => ⟨fun _ => h i hi hf, by rw [h i hi hf]⟩

def PartChecksB (H : Msg → Msg) (D : Deploy Coin) (DO : Flock.Deploy Coin Col T) (L : LayoutB)
    (shape : Form → Prop) (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) (cmt : Msg → ℕ → Msg → Msg)
    (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) (s : ℕ) : Prop :=
  ∀ (x : Msg) (st : CompiledStatement), D.parse x = some st → shape st.form → ∀ (xO : Msg) (w : Flock.Wit),
    Flock.check DO.parse xO w = true → (∀ e ∈ L.calls x s w, H e.1 = e.2) →
    (∀ (c : ℕ → Coin), shows s x xO c → ∀ ms : List Msg,
      (∀ i < L.rounds x, ∀ m, L.held x s w i = some m → ms.getD i [] = m) →
      (∀ i < L.rounds x, ∀ o m, (o, m) ∈ L.piecesOf x s w i → ((ms.getD i []).drop o).take m.length = m) →
      L.CmtAgrees cmt x xO ms → L.RowsRead x s w ms →
      chk s x c ms) ∨
      (findColl (L.calls x s w)).isSome

def PartChecksBV (H : Msg → Msg) (D : Deploy Coin) (DO : Flock.Deploy Coin Col T) (L : LayoutB)
    (shape : Form → Prop) (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) (cmt : Msg → ℕ → Msg → Msg)
    (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) (s : ℕ) : Prop :=
  ∀ (x : Msg) (st : CompiledStatement), D.parse x = some st → shape st.form → ∀ (xO : Msg) (w : Flock.Wit),
    Flock.check DO.parse xO w = true → (∀ e ∈ L.calls x s w, H e.1 = e.2) →
    (∀ (c : ℕ → Coin), shows s x xO c → ∀ ms : List Msg,
      (∀ i < L.rounds x, ∀ m, L.held x s w i = some m → ms.getD i [] = m) →
      (∀ i < L.rounds x, ∀ o m, (o, m) ∈ L.piecesOf x s w i → ((ms.getD i []).drop o).take m.length = m) →
      L.CmtAgrees cmt x xO ms → L.RowsRead x s w ms →
      chk s x c ms) ∨
      (findColl (L.calls x s w ++ vcallsAt H D xO)).isSome

omit [Nonempty Coin] [Inhabited Coin] [Fintype Coin] in
theorem partChecksB_toV {H : Msg → Msg} {D : Deploy Coin} {DO : Flock.Deploy Coin Col T} {L : LayoutB}
    {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop} {cmt : Msg → ℕ → Msg → Msg}
    {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop} {s : ℕ} (h : PartChecksB H D DO L shape shows cmt chk s) :
    PartChecksBV H D DO L shape shows cmt chk s := by
  intro x st hx hf xO w hck hT
  rcases h x st hx hf xO w hck hT with h | h
  · exact Or.inl h
  · exact Or.inr (findColl_isSome_of_sub H (vcallsAt_append_true hT) (fun _ he => List.mem_append_left _ he) h)

def PartOpensB (H : Msg → Msg) (D : Deploy Coin) (DO : Flock.Deploy Coin Col T) (L : LayoutB)
    (shape : Form → Prop) (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) (cmt : Msg → ℕ → Msg → Msg) (s : ℕ) : Prop :=
  ∀ (x : Msg) (st : CompiledStatement), D.parse x = some st → shape st.form → ∀ (xO : Msg) (w : Flock.Wit),
    Flock.check DO.parse xO w = true → (∀ e ∈ L.calls x s w, H e.1 = e.2) →
    (∀ (c : ℕ → Coin), shows s x xO c → ∀ i, i + 1 < L.rounds x →
      (∀ m, L.held x s w i = some m → m = cmt x i xO) ∧
        ∀ pre p post, L.pieceAt x i = pre ++ p :: post → p.1 = s →
          ((cmt x i xO).drop (pre.map fun q => pieceLen q.2).sum).take (pieceLen p.2) = readSegs w p.2) ∨
      (findColl (L.calls x s w)).isSome

/-- The last round's message is never sent in the run (`ReadersRunPlayed`), so no commit string is asked of it. -/
def VstarShowsB (D : Deploy Coin) (I : Inner Coin W) (L : LayoutB) (shape : Form → Prop) (decC : Msg → Msg → Coin)
    (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) (cmt : Msg → ℕ → Msg → Msg) : Prop :=
  ∀ (x : Msg) (st : CompiledStatement), D.parse x = some st → shape st.form → ∀ ps : List (Msg × Out),
    ps.length = I.V.k (xI D x) + D.kS x → ∀ c : ℕ → Coin,
      (∀ i < ps.length, ∃ m, (ps.getD i ([], .rej)).2 = .msg m ∧ decC x m = c i) →
        ∀ s < D.nS x, shows s x ((D.vstar x ps).getD s []) c ∧
          ∀ i, i + 1 < L.rounds x → cmt x i ((D.vstar x ps).getD s []) = (ps.map Prod.fst).getD i []

/-- The escape is for a commit string several parts hold: the readers take the first holder's, and a holder that opens it otherwise shows a collision among the calls they record. -/
def DecompB (H : Msg → Msg) (D : Deploy Coin) (IV : Verifier Coin) (L : LayoutB) (shape : Form → Prop)
    (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) : Prop :=
  ∀ (x : Msg) (st : CompiledStatement), D.parse x = some st → shape st.form → ∀ (c : ℕ → Coin) (ws : List Flock.Wit),
    ws.length = D.nS x → (∀ e ∈ L.allCalls x ws, H e.1 = e.2) → (∀ s < D.nS x, chk s x c (L.msgs x ws)) →
      (run H (replay (L.msgs x ws)) IV (xI D x) () fun i => c i.val).out = .acc ∨
        (findColl (L.allCalls x ws)).isSome

end Facts

end Claims.Recursion.VStarB
