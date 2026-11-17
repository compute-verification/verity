module

public import Proofs.Flock.Claims.Recursion.VerdictParts
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
import Proofs.Flock.Claims.JsonNorm
import Proofs.Flock.Claims.Recursion.DeployChecked

/-! `hrest`'s base checks hold off any witnesses, given what the reading fixes (`DecStmt`, `DecShaped`), which no statement fact gives. -/

@[expose] public section

namespace RecDeploy

open Flock
open Lean (Json)
open Claims (Msg)
open FlockDeploy (Coin FinalP nextR nextR_coin VerdictR holdsR decR DecR actR jAt checksAtJ sampleOfJ sampleEq custody
  liveOfJ setupOf setupWith tabsOf pointsOf shapeAt NamesDistinct)
open Claims.Recursion.VStarB

noncomputable section

def DecStmt (rd : Msg → Option FinalP) : Prop :=
  ∀ (x : Msg) (as : List Msg) (cs : List Coin) (d : DecR), decR rd x as cs = some d → d.f.stmt = x

def DecShaped (rd : Msg → Option FinalP) : Prop :=
  ∀ (x : Msg) (as : List Msg) (cs : List Coin) (d : DecR), decR rd x as cs = some d →
    ∀ (k : ℕ) (hk : k < d.f.ps.size),
      FlockVBridge.Family.ProofShaped (shapeAt (tabsOf d.sts) (pointsOf d.spec d.c₁) k) d.f.ps[k]

def SampleTied : Prop :=
  ∀ (x : Msg) (s : FlockDeploy.Ins) (spec : SessionSpec) (sts : Array Stmt), setupOf false x = some (s, spec, sts) →
    s.read.check = .ok () → sampleEq (sampleOfU spec.unitSample) s.read.draw = true

def CustodyRecJ (rd : Msg → Option FinalP) : Prop :=
  ∀ (x : Msg) (as : List Msg) (cs : List Coin) (d : DecR), as.length = cs.length → decR rd x as cs = some d →
    d.s.read.check = .ok () →
    ∃ lv, liveOfJ d.spec (jAt FlockDeploy.sha512 d.spec d.sts d.c₁ d.mids d.f) = .ok lv ∧
      custody d.spec lv d.link d.c₁ d.mids = .ok ()

theorem actR_decide_setup {rd : Msg → Option FinalP} {x : Msg} {as : List Msg} {cs : List Coin}
    {s : FlockDeploy.Ins} {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Msg × Coin)} {f : FinalP} (h : actR rd x as cs = .decide s spec sts link c₁ mids f) :
    setupOf false x = some (s, spec, sts) ∧ decR rd x as cs = some ⟨s, spec, sts, link, c₁, mids, f⟩ := by
  have hd := FlockDeploy.actR_decide h
  obtain ⟨v, c₁', r₀, rss, fl, link', pf, hv, -, -, -, -, -, he⟩ := FlockDeploy.decR_some hd
  obtain ⟨hs, hspec, hsts, -⟩ := DecR.mk.inj he
  subst hs hspec hsts
  exact ⟨hv, hd⟩

theorem holdsR_setup (rd : Msg → Option FinalP) (x : Msg) (as : List Msg) (cs : List Coin) :
    holdsR FlockDeploy.sha512 rd .setup x as cs := by
  unfold holdsR
  split
  · next s spec sts link c₁ mids f h =>
    exact ⟨_, setupWith_sha512_of (actR_decide_setup h).1⟩
  · trivial

theorem holdsR_names (rd : Msg → Option FinalP) (x : Msg) (as : List Msg) (cs : List Coin)
    (hck : ∀ s spec sts, setupOf false x = some (s, spec, sts) → s.read.check = .ok ()) :
    holdsR FlockDeploy.sha512 rd .names x as cs := by
  unfold holdsR
  split
  · next s spec sts link c₁ mids f h =>
    have hx := (actR_decide_setup h).1
    intro spec' sts' hs'
    rw [setupWith_sha512_of hx] at hs'
    obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Except.ok.inj hs')
    exact FlockDeploy.namesDistinct_of_setup (FlockDeploy.setupOf_some hx).2
      (Claims.Bridge.Custody.publicInputs_none_of_check (hck _ _ _ hx))
  · trivial

theorem holdsR_shaped (rd : Msg → Option FinalP) (x : Msg) (as : List Msg) (cs : List Coin)
    (hsh : ∀ d, decR rd x as cs = some d → ∀ (k : ℕ) (hk : k < d.f.ps.size),
      FlockVBridge.Family.ProofShaped (shapeAt (tabsOf d.sts) (pointsOf d.spec d.c₁) k) d.f.ps[k]) :
    holdsR FlockDeploy.sha512 rd .shaped x as cs := by
  unfold holdsR
  split
  · next s spec sts link c₁ mids f h =>
    obtain ⟨hx, hd⟩ := actR_decide_setup h
    intro spec' sts' hs'
    rw [setupWith_sha512_of hx] at hs'
    obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Except.ok.inj hs')
    exact hsh _ hd
  · trivial

theorem checksAtJ_ok_of {s : FlockDeploy.Ins} {spec : SessionSpec} {x : Msg} {link : List ByteArray × List ByteArray}
    {c₁ : Coin} {mids : List (Msg × Coin)} {f : FlockDeploy.FinalJ} {j' : Json} {lv : FlockDeploy.Live}
    (hS : f.stmt = x) (hD : sampleEq (sampleOfJ f.j) s.read.draw = true) (hL : liveOfJ spec j' = .ok lv)
    (hC : custody spec lv link c₁ mids = .ok ()) (hQ : s.read.check = .ok ()) (hN : s.read.draw.isSome = true) :
    checksAtJ s spec x link c₁ mids f j' = .ok () := by
  have hN' : s.read.draw.isNone = false := by
    cases h : s.read.draw with
    | none => rw [h] at hN; cases hN
    | some _ => rfl
  unfold checksAtJ
  simp only [hS, beq_self_eq_true, hD, hL, hC, hQ, hN', Record.check, bind, Except.bind, pure, Except.pure,
    Bool.false_eq_true, ↓reduceIte]
  try rfl

theorem sampleOfJ_recordObj (spec : SessionSpec) (h : ByteArray → ByteArray) (rootB : Array ByteArray)
    (points : Array (Array F128)) (streams : Array (Array Flock.Round)) (proofs : Array ByteArray) :
    sampleOfJ (FlockComplete.JsonRT.recordObj spec h rootB points streams proofs) =
      (sampleOfU spec.unitSample).map Flock.norm := by
  unfold sampleOfJ FlockComplete.JsonRT.recordObj
  rw [FlockComplete.JsonRT.getObjVal?_mkObj (FlockComplete.JsonRT.recordList_distinct spec h rootB points streams
    proofs) (k := "unit_draw") (v := spec.unitSample.getD Json.null) (by simp [FlockComplete.JsonRT.recordList])]
  cases spec.unitSample with
  | none => rfl
  | some d => cases d <;> first | rfl | exact congrArg some (FlockDeploy.norm_norm _).symm

theorem sampleEq_map_norm {a b : Option Json} (hb : ∀ e, b = some e → Flock.norm e = e) (h : sampleEq a b = true) :
    sampleEq (a.map Flock.norm) b = true := by
  rw [FlockDeploy.sampleEq_iff] at h ⊢
  cases h with
  | none => exact .none
  | @some a' b' h' =>
    refine .some ?_
    show FlockDeploy.JsonEquiv (Flock.norm a') b'
    rw [FlockDeploy.norm_congr a' b' h', hb b' rfl]
    exact FlockDeploy.JsonEquiv.refl _

theorem holdsR_after (rd : Msg → Option FinalP) (x : Msg) (as : List Msg) (cs : List Coin)
    (hlen : as.length = cs.length)
    (hck : ∀ s spec sts, setupOf false x = some (s, spec, sts) → s.read.check = .ok () ∧ s.read.draw.isSome = true)
    (hstmt : DecStmt rd) (hud : SampleTied) (hcu : CustodyRecJ rd) :
    holdsR FlockDeploy.sha512 rd .after x as cs := by
  unfold holdsR
  split
  · next s spec sts link c₁ mids f h =>
    obtain ⟨hx, hd⟩ := actR_decide_setup h
    intro spec' sts' hs'
    rw [setupWith_sha512_of hx] at hs'
    obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Except.ok.inj hs')
    obtain ⟨hq, hn⟩ := hck _ _ _ hx
    obtain ⟨lv, hL, hC⟩ := hcu x as cs _ hlen hd hq
    have hD : sampleEq (sampleOfJ f.j) s.read.draw = true := by
      obtain ⟨v, c₁', r₀, rss, fl, link', pf, -, -, -, -, -, -, he⟩ := FlockDeploy.decR_some hd
      obtain ⟨-, hspec, -, -, -, -, hf⟩ := DecR.mk.inj he
      rw [hf]
      dsimp only
      rw [FlockDeploy.recJ, sampleOfJ_recordObj, ← hspec]
      exact sampleEq_map_norm (fun e he => (FlockDeploy.insOf_sample_norm (FlockDeploy.setupOf_some hx).1 he).1)
        (hud _ _ _ _ hx hq)
    exact checksAtJ_ok_of (f := ⟨f.stmt, f.j, #[]⟩) (hstmt x as cs _ hd) hD hL hC hq hn
  · trivial

def BasesHold (rd : Msg → Option FinalP) : Prop :=
  ∀ (x : Msg) (st : Claims.Recursion.CompiledStatement), parse x = some st → ∀ b ∈ VerdictP.bases, ∀ (as : List Msg)
    (cs : List Coin), as.length = cs.length → holdsR FlockDeploy.sha512 rd b (xIOf x) as cs

theorem basesHold (rd : Msg → Option FinalP) (hstmt : DecStmt rd) (hsh : DecShaped rd) (hud : SampleTied)
    (hcu : CustodyRecJ rd) : BasesHold rd := by
  intro x st hx b hb as cs hlen
  have hck : ∀ s spec sts, setupOf false (xIOf x) = some (s, spec, sts) →
      s.read.check = .ok () ∧ s.read.draw.isSome = true := by
    intro s spec sts h
    obtain ⟨s', spec', sts', h', hq, hn⟩ := parse_innerOk hx
    rw [h] at h'
    obtain ⟨rfl, -⟩ := Prod.mk.inj (Option.some.inj h')
    exact ⟨hq, hn⟩
  simp only [VerdictP.bases, List.mem_cons, List.not_mem_nil, or_false] at hb
  rcases hb with rfl | rfl | rfl | rfl
  · exact holdsR_setup rd _ as cs
  · exact holdsR_shaped rd _ as cs (hsh _ as cs)
  · exact holdsR_names rd _ as cs fun s spec sts h => (hck s spec sts h).1
  · exact holdsR_after rd _ as cs hlen hck hstmt hud hcu

theorem restBase_holds (rd : Msg → Option FinalP) (hB : BasesHold rd) {Q : Type} (pieces : Msg → List Q)
    (pholds : Q → Msg → List Msg → List Coin → Prop)
    (hpieces : ∀ (x : Msg) (ms : List Msg) (cs : List Coin), FlockDeploy.ReadOkR rd x ms cs →
      (∀ b ∈ VerdictP.bases, holdsR FlockDeploy.sha512 rd b x ms cs) → (∀ q ∈ pieces x, pholds q x ms cs) →
        holdsR FlockDeploy.sha512 rd .verdict x ms cs)
    (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (rel' : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (bB : Brings) (L : LayoutB)
    (IV : Claims.Verifier Coin) {x : Msg} {st : Claims.Recursion.CompiledStatement} (hx : (recBound rel' bB L IV).parse x = some st)
    {b : VerdictR} (hb : b ∈ VerdictP.bases) (c : ℕ → Coin) (ms : List Msg) :
    (IVCheck.elem (DeployCheck.last (ReadCheck.verdict (VerdictP.base b))) :
        IVCheck (DeployCheck (ReadCheck (VerdictP Q)))).Holds FlockDeploy.sha512
      (bdDeployed rel (nextR rd) colsE wE join rowIn final w)
      (deploySplit FlockDeploy.sha512 rel (nextR rd) colsE wE join rowIn final w (nextR_coin rd FlockDeploy.sha512)
        (lastSplitP FlockDeploy.sha512 rd pieces pholds hpieces).toLast)
      (Claims.Recursion.xI (recBound rel' bB L IV) x) c ms := by
  intro os last _ hos
  refine hB x st hx b hb _ _ ?_
  rw [List.length_map, List.length_range, ← hos]
  simp [Claims.Recursion.Bound.opened]

end

end RecDeploy
