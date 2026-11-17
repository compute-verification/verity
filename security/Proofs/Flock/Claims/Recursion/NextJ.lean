module

public import Proofs.Flock.Claims.Recursion.NextCs
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

/-! `IVStar`'s last step at the record's parsed value (`nextJ`), which has no parse; proof-file decoding stays among its verdict's checks. -/

@[expose] public section

namespace FlockDeploy

open Flock
open Lean (Json)

/-- **The prover's last message at the parsed value**: the statement, the record's parsed value and the proof files. -/
structure FinalJ where
  stmt : Claims.Msg
  j : Json
  proofs : Array ByteArray

/-- The record's sample at its parsed value, rebuilt (`recordDraw` after the parse). -/
def sampleOfJ (j : Json) : Option Json :=
  match j.getObjVal? "unit_draw" with
  | .ok Json.null => none
  | .ok d => some (Flock.norm d)
  | .error _ => none

theorem recordSample_of_parse {record : String} {j : Json} (h : Flock.JsonParse.parse record = .ok j) :
    FlockSoundness.Discharge.Exec.recordDraw record = sampleOfJ j := by
  unfold FlockSoundness.Discharge.Exec.recordDraw
  rw [h]
  rfl

/-- The checks after the verdict at the parsed value: the statement, the sample at `f.j`, custody at `j'`. -/
def checksAtJ (s : Ins) (spec : SessionSpec) (x : Claims.Msg) (link : List ByteArray × List ByteArray) (c₁ : Coin)
    (mids : List (Claims.Msg × Coin)) (f : FinalJ) (j' : Json) : Flock.V Unit := do
  Record.check (f.stmt == x) "live: the last message's statement is not the statement"
  Record.check (sampleEq (sampleOfJ f.j) s.read.draw) "live: the record's draw is not the statement's"
  custody spec (← liveOfJ spec j') link c₁ mids
  s.read.check
  if s.read.draw.isNone then ProvedScope.sampleDesign false s.read.tags false none

/-- **The decision at a session at the parsed value**: the plain verdict and custody's reading at the coin-substituted
value. -/
def decisionAtJ (s : Ins) (spec : SessionSpec) (sts : Array Stmt) (x : Claims.Msg)
    (link : List ByteArray × List ByteArray) (c₁ : Coin) (mids : List (Claims.Msg × Coin)) (f : FinalJ) :
    Claims.OCV Unit := do
  verifyJWithM Claims.Hashes.oc spec (tabsOf sts) (coinSubJ spec c₁ mids f.j) f.proofs
  monadLift (checksAtJ s spec x link c₁ mids f (coinSubJ spec c₁ mids f.j))

def decisionJF (s : Ins) (x : Claims.Msg) (link : List ByteArray × List ByteArray) (c₁ : Coin)
    (mids : List (Claims.Msg × Coin)) (f : FinalJ) : Claims.OCV Unit := do
  let (spec, sts) ← setupM Claims.Hashes.oc s.read
  decisionAtJ s spec sts x link c₁ mids f

inductive ActJ where
  | rej
  | coin (c : Coin)
  | decide (s : Ins) (spec : SessionSpec) (sts : Array Stmt) (link : List ByteArray × List ByteArray) (c₁ : Coin)
      (mids : List (Claims.Msg × Coin)) (f : FinalJ)

/-- **`nextJ`'s step**: a coin where the deployed plain verifier answers one; else, at a statement that sets up, decide
where `rd` reads the last message and the link message reads; anything else, refuse. -/
def actJ (rd : Claims.Msg → Option FinalJ) (x : Claims.Msg) (as : List Claims.Msg) (cs : List Coin) : ActJ :=
  match act false x as cs with
  | .coin c => .coin c
  | _ =>
    match setupOf false x with
    | none => .rej
    | some (s, spec, sts) =>
      match as.getLast?, as, cs with
      | some a, a₁ :: rest, c₁ :: crest =>
        match rd a, linkOf a₁ with
        | some f, some link => .decide s spec sts link c₁ ((rest.dropLast.map List.tail).zip crest.dropLast) f
        | _, _ => .rej
      | _, _, _ => .rej

def nextJ (rd : Claims.Msg → Option FinalJ) (x : Claims.Msg) (as : List Claims.Msg) (cs : List Coin) :
    Claims.OC Claims.Out :=
  match actJ rd x as cs with
  | .rej => .ret .rej
  | .coin c => .ret (.msg c.bytes)
  | .decide s _ _ link c₁ mids f => outOf (decisionJF s x link c₁ mids f)

theorem nextJ_coin (rd : Claims.Msg → Option FinalJ) (H : Claims.Msg → Claims.Msg) : ∀ (x : Claims.Msg)
    (as : List Claims.Msg) (cs : List Coin) (c : Coin), act false x as cs = .coin c →
    ((nextJ rd x as cs).eval H).1 = .msg c.bytes := by
  intro x as cs c h
  simp only [nextJ, actJ, h, Claims.OC.eval]

theorem nextJ_decide {rd : Claims.Msg → Option FinalJ} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    {s : Ins} {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f : FinalJ} (h : actJ rd x as cs = .decide s spec sts link c₁ mids f) :
    nextJ rd x as cs = outOf (decisionJF s x link c₁ mids f) := by
  simp only [nextJ, h]

theorem decisionAtJ_eq (s : Ins) (spec : SessionSpec) (sts : Array Stmt) (x : Claims.Msg)
    (link : List ByteArray × List ByteArray) (c₁ : Coin) (mids : List (Claims.Msg × Coin)) {f : Final} {j : Json}
    (hj : Flock.JsonParse.parse f.record = .ok j) (hc : RecordText.printT j = f.record) :
    decisionAtJ s spec sts x link c₁ mids ⟨f.stmt, j, f.proofs⟩ = decisionCsAt s spec sts x link c₁ mids f := by
  have hs : RecordText.check f.record = pure () := RecordText.check_of hj hc
  rw [decisionCsAt_parse s spec sts x link c₁ mids hj]
  simp only [decisionAtJ, decisionJ, verdictJM, checksAtJ, checksJ, recordSample_of_parse hj, hs, bind_pure_unit,
    Bool.false_eq_true, ↓reduceIte]

theorem decisionJF_eq (s : Ins) (x : Claims.Msg) (link : List ByteArray × List ByteArray) (c₁ : Coin)
    (mids : List (Claims.Msg × Coin)) {f : Final} {j : Json} (hj : Flock.JsonParse.parse f.record = .ok j)
    (hc : RecordText.printT j = f.record) :
    decisionJF s x link c₁ mids ⟨f.stmt, j, f.proofs⟩ = decisionCs s x link c₁ mids f := by
  unfold decisionJF decisionCs
  congr 1
  funext p
  obtain ⟨spec, sts⟩ := p
  exact decisionAtJ_eq s spec sts x link c₁ mids hj hc

def ReadOk (rd : Claims.Msg → Option FinalJ) (x : Claims.Msg) (ms : List Claims.Msg) (cs : List Coin) : Prop :=
  match actJ rd x ms cs with
  | .decide _ spec _ _ c₁ mids f => ∃ lv, liveOfJ spec (coinSubJ spec c₁ mids f.j) = .ok lv
  | _ => False

end FlockDeploy

namespace RecDeploy

open Flock
open Claims (Msg Out)
open FlockDeploy (Coin FinalJ nextJ nextJ_coin ReadOk)
open Claims.Recursion.VStarB

structure LastSplitJ (H : Msg → Msg) (rd : Msg → Option FinalJ) (R : Type) where
  checks : Msg → List R
  holds : R → Msg → List Msg → List Coin → Prop
  sound : ∀ (x : Msg) (ms : List Msg) (cs : List Coin), ms.length = (FlockDeploy.V false).k x →
    cs.length = ms.length → ReadOk rd x ms cs → (∀ r ∈ checks x, holds r x ms cs) → ((nextJ rd x ms cs).eval H).1 = .acc

inductive ReadCheck (R : Type) where
  | read
  | verdict (r : R)

def LastSplitJ.toLast {H : Msg → Msg} {rd : Msg → Option FinalJ} {R : Type} (LJ : LastSplitJ H rd R) :
    LastSplit H (nextJ rd) (ReadCheck R) where
  checks x := .read :: (LJ.checks x).map .verdict
  holds q x ms cs := match q with
    | .read => ReadOk rd x ms cs
    | .verdict r => LJ.holds r x ms cs
  sound x ms cs hk hc h := LJ.sound x ms cs hk hc (h .read (List.mem_cons_self ..)) fun r hr =>
    h (.verdict r) (List.mem_cons_of_mem _ (List.mem_map.2 ⟨r, hr, rfl⟩))

def ownJ {R : Type} (D : Claims.Recursion.Deploy Coin) (L : LayoutB) (ownR : Msg → R → Option ℕ) (x : Msg) :
    ReadCheck R → Option ℕ
  | .read => firstHolder D L x ((FlockDeploy.V false).k (Claims.Recursion.xI D x) - 1)
  | .verdict r => ownR x r

theorem ownJ_lt {H : Msg → Msg} {rd : Msg → Option FinalJ} {R : Type} (LJ : LastSplitJ H rd R)
    (ownR : Msg → R → Option ℕ) (D : Claims.Recursion.Deploy Coin) (L : LayoutB) (shape : Claims.Recursion.Form → Prop)
    (hR : ∀ (x : Msg) (st : Claims.Recursion.CompiledStatement), D.parse x = some st → shape st.form →
      ∀ r ∈ LJ.checks (Claims.Recursion.xI D x), ∀ s, ownR x r = some s → s < D.nS x) :
    ∀ (x : Msg) (st : Claims.Recursion.CompiledStatement), D.parse x = some st → shape st.form →
      ∀ r ∈ LJ.toLast.checks (Claims.Recursion.xI D x), ∀ s, ownJ D L ownR x r = some s → s < D.nS x := by
  intro x st hx hf r hr s ho
  cases r with
  | read => exact firstHolder_lt ho
  | verdict r =>
    refine hR x st hx hf r ?_ s ho
    rcases List.mem_cons.1 hr with h | h
    · cases h
    · obtain ⟨r', hr', h⟩ := List.mem_map.1 h
      cases h
      exact hr'

/-- Proof-file decoding is among the verdict's checks and owned by no reader part, so `hlow` is unsatisfiable where
no part decodes the proofs; `RecDeploy.decompB_nextS` is the composition without it. -/
theorem decompB_nextJ (H : Msg → Msg) (rd : Msg → Option FinalJ)
    (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) {join : List Msg → Msg}
    (hj : ∀ rs, ∃ a, join rs = 0 :: a)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    {R : Type} (LJ : LastSplitJ H rd R) (ownR : Msg → R → Option ℕ)
    (rel' : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings) (L : LayoutB)
    (IV : Claims.Verifier Coin) (shape : Claims.Recursion.Form → Prop) {Col' T' : Type}
    (DO : Claims.Flock.Deploy Coin Col' T') (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) (cmt : Msg → ℕ → Msg → Msg)
    (hR : ∀ (x : Msg) (st : Claims.Recursion.CompiledStatement), (recBound rel' b L IV).parse x = some st → shape st.form →
      ∀ r ∈ LJ.checks (Claims.Recursion.xI (recBound rel' b L IV) x), ∀ s, ownR x r = some s →
        s < (recBound rel' b L IV).nS x)
    (hrest : ∀ (x : Msg) (st : Claims.Recursion.CompiledStatement), (recBound rel' b L IV).parse x = some st → shape st.form →
      ∀ q ∈ ivChecks (bdDeployed rel (nextJ rd) colsE wE join rowIn final w)
          (deploySplit H rel (nextJ rd) colsE wE join rowIn final w (nextJ_coin rd H) LJ.toLast)
          (Claims.Recursion.xI (recBound rel' b L IV) x),
        ownHolders (recBound rel' b L IV) L (ownDeploy (ownJ (recBound rel' b L IV) L ownR)) x q = none →
        (∀ r, q = .elem r → ∃ r', r = .last r') →
        ∀ (c : ℕ → Coin) (ws : List Claims.Flock.Wit), ws.length = (recBound rel' b L IV).nS x →
          (∀ e ∈ L.allCalls x ws, H e.1 = e.2) →
          q.Holds H (bdDeployed rel (nextJ rd) colsE wE join rowIn final w)
            (deploySplit H rel (nextJ rd) colsE wE join rowIn final w (nextJ_coin rd H) LJ.toLast)
            (Claims.Recursion.xI (recBound rel' b L IV) x) c (L.msgs x ws))
    (hlow : ∀ s x c ms, PartAccepts H (recBound rel' b L IV) DO L shape shows cmt s x c ms →
      PartComputes H (recBound rel' b L IV) (bdDeployed rel (nextJ rd) colsE wE join rowIn final w)
        (deploySplit H rel (nextJ rd) colsE wE join rowIn final w (nextJ_coin rd H) LJ.toLast)
        (ownHolders (recBound rel' b L IV) L (ownDeploy (ownJ (recBound rel' b L IV) L ownR))) s x c ms) :
    DecompB H (recBound rel' b L IV) (bdDeployed rel (nextJ rd) colsE wE join rowIn final w).IVStar L shape
      (PartAccepts H (recBound rel' b L IV) DO L shape shows cmt) :=
  decompB_partAccepts_of_parts H (recBound rel' b L IV) DO _ L shape shows cmt
    (deploySplit H rel (nextJ rd) colsE wE join rowIn final w (nextJ_coin rd H) LJ.toLast) _ _ hlow
    (ownedBy_deploy H rel (nextJ rd) colsE wE join rowIn final w (nextJ_coin rd H) LJ.toLast
      (ownJ (recBound rel' b L IV) L ownR) _ L shape (ownJ_lt LJ ownR _ L shape hR))
    (stmtFixes_deploy H rel (nextJ rd) colsE wE hj rowIn final w (nextJ_coin rd H) LJ.toLast
      (ownJ (recBound rel' b L IV) L ownR) rel' b L IV shape hrest)

end RecDeploy
