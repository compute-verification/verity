module

public import Proofs.Flock.Claims.Recursion.VStarBound
public import Proofs.Flock.Claims.Recursion.DeployPremises
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
import Mathlib.Tactic.ReduceModChar
import Mathlib.Topology.Sheaves.Init
import Proofs.Flock.Claims.Recursion.FlockKSRecord

/-! `KSBound` pinned to the deployment (option (d)), `KSBoundDeployed`, at the records `recBoundAt`, `deE` and
`bdDeployed`. -/

@[expose] public section

namespace RecDeploy

open Claims (Msg Out)
open FlockDeploy (Coin)

/-- **V\*'s bound layout as the compiled verifier's readers**: the inner messages and each part's calls, off the parts'
witnesses. -/
def _root_.Claims.Recursion.VStarB.LayoutB.readers (L : Claims.Recursion.VStarB.LayoutB) : RecDeploy.Readers :=
  ⟨L.msgs, L.calls⟩

/-- A rep's schedule at a table of the session (`rec_live.rep_rounds`, `Flock.RecVStar.repRounds`): at the table's `m`
and `k_log` and the session's claims, `2 + points · regions` as `FlockDeploy.repRounds` counts them; empty off
`fast100`. -/
def repSchedAt (spec : _root_.Flock.SessionSpec) (st : _root_.Flock.Stmt) : List _root_.Flock.RecVStar.SchedRound :=
  ((_root_.Flock.RecVStar.repRounds st.m st.c.kLog (2 + spec.points * st.regions.size)).map Array.toList).getD []

/-- A table's level-0 top's nodes, `2^{cap₀}` at `fast100`'s first level (`2^7 = 128` at its 218 queries). -/
def level0Nodes (st : _root_.Flock.Stmt) : ℕ :=
  2 ^ ((_root_.Flock.fast100 st.m).map fun s => (s.levels.getD 0 default).cap).getD 0

/-- **The schedule's commit strings at each inner round before the last** (`Bound.nAt` at the deployment), at an inner
statement the deployed plain verifier sets up. Round 0, the link, carries each table's level-0 top, one string a cap
node. Round `1 + i`, the `i`-th recorded round (each table's two reps in turn, each rep's rounds in order), carries the
salted tops it commits, one a cap node, then one row where it absorbs a message. A rep's copies of level 0's top are
round 0's strings again, not new ones. None at a statement that doesn't set up. -/
def nAtSched (x : Msg) (j : ℕ) : ℕ :=
  match FlockDeploy.setupOf false x with
  | none => 0
  | some (_, spec, sts) =>
    if j = 0 then (sts.toList.map level0Nodes).sum
    else
      match (sts.toList.flatMap fun st => repSchedAt spec st ++ repSchedAt spec st)[j - 1]? with
      | some r => r.2.1.toList.sum + if r.2.2.2 then 1 else 0
      | none => 0

/-- **The element-level inner Flock, pinned to the deployed plain record the claims read**: `deployPlainTrim rel` (the
trimmed record `plain_KSAbort` and `ksA_deployed` take) with only its verifier, its columns and its weight replaced.
The verifier keeps the deployed plain verifier's rounds (`FlockDeploy.V false`'s `k`); its steps (`nextE`) read
element-level messages. At the deployed plain verifier's own steps, columns and weight it is `deployPlainTrim rel`
(`deE_plainTrim`). -/
noncomputable def deE (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) :
    Claims.Flock.Deploy Coin (List FlockLevel3.GF128) Claims.Flock.PlainTrace :=
  { FlockDeploy.deployPlainTrim rel with V := ⟨(FlockDeploy.V false).k, nextE⟩, cols := colsE, weight := wE }

/-- **The deployed bound record**: `DE` pinned (`deE`), each round before the last answered by its coin's bytes, the
schedule's commit strings (`nAtSched`) and the pinned hm96 key's salts, salt prefix and salt mask. The rows' digest input
(`rowIn`), how the last message reads (`final`), how a round's rows join (`join`) and `γ`'s weight at `IVStar` (`w`)
are the code side's. -/
noncomputable def bdDeployed (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg))
    (w : Msg → ℝ) : Claims.Recursion.Bound Coin (List FlockLevel3.GF128) Claims.Flock.PlainTrace where
  DE := deE rel nextE colsE wE
  ans c := c.bytes
  join := join
  nAt := nAtSched
  saltLen := _root_.Flock.Hm96.sha512.saltBytes
  saltPrefix := Claims.toMsg _root_.Flock.Hm96.Default512.saltPrefix
  mask y i := (_root_.Flock.Hm96.Default512.mask ⟨Array.ofFn y⟩).get! i.val
  rowIn := rowIn
  final := final
  weight := w

/-- **The compiled record at an inner verifier, at the deployment**: `deployRec` with the deployed plain parser's
scope check for the inner statement and V*'s bound layout's readers. -/
noncomputable def recBound (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings)
    (L : Claims.Recursion.VStarB.LayoutB) (IV : Claims.Verifier Coin) : Claims.Recursion.Deploy Coin :=
  deployRec IV (FlockDeploy.deployPlain rel).scopeCheck { b with rd := L.readers }

/-- **The compiled record at each inner verifier, at the deployment** (`KSBound`'s `RecAt`): `recBound`, whose inner
view runs the verifier it is given on V*'s bound layout's messages at the inner statement. -/
noncomputable def recBoundAt (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings)
    (L : Claims.Recursion.VStarB.LayoutB) : Claims.Recursion.RecAt Coin where
  record := recBound rel b L
  msgs := L.msgs
  xI := xIOf
  runs _ _ _ _ _ := rfl

/-- No theorem evaluates its level (`Claims.Recursion.lam`) at the deployment. Holo statements are candidate 3's own
model. -/
def KSBoundDeployed (rel relO : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings)
    (L : Claims.Recursion.VStarB.LayoutB) (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg))
    (w : Msg → ℝ) : Prop :=
  Claims.Recursion.KSBound FlockDeploy.sha512 (recBoundAt rel b L) (bdDeployed rel nextE colsE wE join rowIn final w)
    (deployZkRec relO)

def KSPrivBoundDeployed (rel relO : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings)
    (L : Claims.Recursion.VStarB.LayoutB) (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg))
    (w : Msg → ℝ) : Prop :=
  Claims.Recursion.KSPrivBound FlockDeploy.sha512 (recBoundAt rel b L)
    (bdDeployed rel nextE colsE wE join rowIn final w) (deployZkRec relO)

theorem innerView_ksBound (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings)
    (L : Claims.Recursion.VStarB.LayoutB) (Bd : Claims.Recursion.Bound Coin (List FlockLevel3.GF128)
      Claims.Flock.PlainTrace) (H : Msg → Msg) (x : Msg) (r : Fin ((recBound rel b L Bd.IVStar).V.k x) → Coin)
    (ws : List Claims.Flock.Wit) :
    (recBound rel b L Bd.IVStar).innerView H x r ws =
      (Claims.run H (Claims.replay (L.msgs x ws)) Bd.IVStar (xIOf x) () fun i => Claims.coinSeq r i.val).seen :=
  rfl

theorem deE_k (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) :
    (deE rel nextE colsE wE).V.k = (FlockDeploy.V false).k :=
  rfl

theorem decOK_deE (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ)
    (decI : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace) :
    Claims.Recursion.decOK (deE rel nextE colsE wE) decI ↔
      Claims.Recursion.decOK (FlockDeploy.deployPlainTrim rel) decI :=
  Iff.rfl

theorem deE_plainTrim (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) :
    deE rel (FlockDeploy.V false).next (FlockDeploy.deployPlainTrim rel).cols
      (FlockDeploy.deployPlainTrim rel).weight = FlockDeploy.deployPlainTrim rel :=
  rfl

theorem plainKSAbort_bd (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (htab : Claims.Recursion.TableBoundθOnAcc FlockDeploy.sha512 (FlockDeploy.deployPlain rel)
      (FlockDeploy.KS.wordLen false) (FlockDeploy.KS.wide false))
    (join : List Msg → Msg) (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg))
    (w : Msg → ℝ) :
    (bdDeployed rel (FlockDeploy.V false).next (FlockDeploy.deployPlainTrim rel).cols
      (FlockDeploy.deployPlainTrim rel).weight join rowIn final w).PlainKSAbort FlockDeploy.sha512 :=
  fun decI hdec => FlockDeploy.KS.plain_KSAbort rel htab decI hdec

/-- Only the rounds before `IV`'s last: the last message, the openings, is never sent in the run, and `IVStar` checks
the openings itself. -/
def ReadersRunPlayed {Col T : Type} (H : Msg → Msg) (IV : Claims.Verifier Coin) (b : Brings)
    (DO : Claims.Flock.Deploy Coin Col T) : Prop :=
  ∀ (x : Msg) (as : List Msg) (cs : List Coin) (ws : List Claims.Flock.Wit), as.length = IV.k (xIOf x) + kS x →
    cs.length = as.length → (∀ s < nS x, StagedPart b.st s x ((stagedAll b.st x as cs).getD s [])) →
      Claims.Recursion.OuterHolds DO (stagedAll b.st x as cs) (nS x) ws →
      (∀ e ∈ callsOf b.rd x ws, H e.1 = e.2) →
        (∀ j < (innerRunAt H IV b.rd x ws cs).msgs.length, j + 1 < IV.k (xIOf x) →
          ((innerRunAt H IV b.rd x ws cs).msgs.map Prod.fst).getD j [] = as.getD j []) ∨
        ∃ u v, Claims.findColl (callsOf b.rd x ws) = some (u, v) ∧ Claims.Collision H u v

end RecDeploy

namespace Claims.Recursion

theorem Bound.ivstar_congr_final {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]
    [DecidableEq Col] (Bd : Bound Coin Col T) (x : Claims.Msg) (pre : List Claims.Msg) (a a' : Claims.Msg)
    (cs : List Coin) (hfin : Bd.final a = Bd.final a') :
    Bd.IVStar.next x (pre ++ [a]) cs = Bd.IVStar.next x (pre ++ [a']) cs := by
  simp [Bound.IVStar, hfin]

theorem Bound.openView_congr_final {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]
    [DecidableEq Col] (Bd : Bound Coin Col T) (ms : List (Claims.Msg × Claims.Out)) (a a' : Claims.Msg)
    (o out : Claims.Out) (p : List (List (Claims.Msg × (Fin Bd.saltLen → UInt8))) × Claims.Msg)
    (hfin : Bd.final a = some p) (hfin' : Bd.final a' = some p) :
    Bd.openView ⟨ms ++ [(a, o)], out⟩ = Bd.openView ⟨ms ++ [(a', o)], out⟩ := by
  simp [Bound.openView, hfin, hfin']

theorem RecAt.not_const {Coin : Type} [Inhabited Coin] (D : RecAt Coin) (D₀ : Claims.Recursion.Deploy Coin) :
    ¬ ∀ IV, D.record IV = D₀ := by
  intro h
  have key : ∀ (IV : Claims.Verifier Coin) (r : Fin (D₀.V.k []) → Coin),
      D₀.innerView id [] r [] =
        (Claims.run id (Claims.replay (D.msgs [] [])) IV (D.xI []) () fun i => Claims.coinSeq r i.val).seen := by
    intro IV
    have hr := D.runs IV
    rw [h IV] at hr
    exact fun r => hr id [] r []
  let r : Fin (D₀.V.k []) → Coin := fun _ => default
  have h1 := congrArg Claims.Seen.out (key ⟨fun _ => 1, fun _ _ _ => .ret .acc⟩ r)
  have h2 := congrArg Claims.Seen.out (key ⟨fun _ => 1, fun _ _ _ => .ret .rej⟩ r)
  have e1 : (Claims.run id (Claims.replay (D.msgs [] [])) ⟨fun _ => 1, fun _ _ _ => .ret .acc⟩ (D.xI []) ()
      fun i => Claims.coinSeq r i.val).seen.out = Claims.Out.acc := rfl
  have e2 : (Claims.run id (Claims.replay (D.msgs [] [])) ⟨fun _ => 1, fun _ _ _ => .ret .rej⟩ (D.xI []) ()
      fun i => Claims.coinSeq r i.val).seen.out = Claims.Out.rej := rfl
  rw [e1] at h1
  rw [e2] at h2
  exact absurd (h1.symm.trans h2) (by decide)

end Claims.Recursion
