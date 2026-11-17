module

public import Proofs.Flock.Claims.Recursion.VerdictRowDCoin
public import Proofs.Flock.Claims.Recursion.OpenLowerDeployed
public import Proofs.Flock.Claims.Recursion.VerdictRowDSide
public import Proofs.Flock.Recursive.ValueSrc
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
import Proofs.Flock.VBridge.Algebra.RunSkel

/-! Row D's ties at the deployed layout: the side's schedule (`sideSchR_sideAt`) and check 5's positions (`posTieR_of`, from `InnerCoinsR`). -/

@[expose] public section

namespace RecDeploy.OpenLower

open Flock
open Claims (Msg)
open FlockDeploy (Coin FinalP setupOf tabsOf pointsOf)
open Claims.Recursion.VStarB
open FlockSoundness.Refine (shapeOf)
open FlockSoundness.Discharge.Recursion (openRows openDirs innerLevels)
open RecDeploy.Rows (canonRounds posAt sideAt schOf verifierRows_comb)

theorem schOf_eq (x : Msg) (t : ℕ) : schOf x t = schAt x t := by
  unfold schOf schAt
  cases setupOf false x <;> rfl

section Side

variable {H : Msg → Msg} {D : Claims.Recursion.Deploy Coin} {shape : Claims.Recursion.Form → Prop}
  {untr : TrReader} {b : Brings} {cmt : Msg → ℕ → Msg → Msg} {L : LayoutB}
  {Bd : Claims.Recursion.Bound Coin (List FlockLevel3.GF128) Claims.Flock.PlainTrace}

theorem sideSchR_sideAt (rd : Msg → Option FinalP) (pieces : Msg → List OpenPiece) :
    SideSchR H D shape untr b cmt L Bd (sideAt rd) pieces := by
  unfold SideSchR AtPieces AtOpened
  intros
  intros
  exact schOf_eq _ _

end Side

section Congr

variable {sh sh' : Shape}

abbrev algFields (sh : Shape) : ℕ × ℕ × ℕ × Schedule × Bool := (sh.m, sh.kLog, sh.claims, sh.sched, sh.holo)

theorem algFields_eq (h : algFields sh = algFields sh') :
    sh.m = sh'.m ∧ sh.kLog = sh'.kLog ∧ sh.claims = sh'.claims ∧ sh.sched = sh'.sched ∧ sh.holo = sh'.holo := by
  simpa only [algFields, Prod.mk.injEq] using h

theorem rounds_congr (h : algFields sh = algFields sh') : RecAlgebra.rounds sh = RecAlgebra.rounds sh' := by
  obtain ⟨hm, hk, hc, hs, hh⟩ := algFields_eq h
  simp only [RecAlgebra.rounds, RecAlgebra.hRound, hm, hk, hc, hs, hh]

theorem wi_congr (h : algFields sh = algFields sh') : RecAlgebra.wi sh = RecAlgebra.wi sh' := by
  funext k
  simp only [RecAlgebra.wi, RecAlgebra.wNames, rounds_congr h]

theorem order_congr (h : algFields sh = algFields sh') : RecAlgebra.order sh = RecAlgebra.order sh' := by
  obtain ⟨-, -, hc, -, -⟩ := algFields_eq h
  funext k
  simp only [RecAlgebra.order, RecAlgebra.wNames, rounds_congr h, wi_congr h, hc]

theorem symbolic_congr (h : algFields sh = algFields sh') : RecAlgebra.symbolic sh = RecAlgebra.symbolic sh' := by
  obtain ⟨hm, hk, hc, hs, -⟩ := algFields_eq h
  simp only [RecAlgebra.symbolic, hm, hk, hc, hs]

theorem atomTerm_congr (h : algFields sh = algFields sh') (c : ℕ) (k : RecAlgebra.Key) :
    RecAlgebra.atomTerm sh c k = RecAlgebra.atomTerm sh' c k := by
  unfold RecAlgebra.atomTerm
  rw [wi_congr h]

theorem atomOp_congr (h : algFields sh = algFields sh') (c : ℕ) (k : RecAlgebra.Key) :
    RecAlgebra.atomOp sh c k = RecAlgebra.atomOp sh' c k := by
  obtain ⟨-, -, -, -, hh⟩ := algFields_eq h
  unfold RecAlgebra.atomOp
  rw [hh, wi_congr h, atomTerm_congr h]

theorem varOp_congr (h : algFields sh = algFields sh') (c : ℕ) (k : RecAlgebra.Key) :
    RecAlgebra.varOp sh c k = RecAlgebra.varOp sh' c k := by
  unfold RecAlgebra.varOp
  simp only [atomOp_congr h]

theorem orderOf_congr (h : algFields sh = algFields sh') : RecAlgebra.orderOf sh = RecAlgebra.orderOf sh' := by
  funext k
  unfold RecAlgebra.orderOf
  rw [order_congr h]

theorem coefOps_congr (h : algFields sh = algFields sh') (c ri : ℕ) (l : List (RecAlgebra.Key × F128)) :
    RecAlgebra.coefOps sh c ri l = RecAlgebra.coefOps sh' c ri l := by
  induction l with
  | nil => simp only [RecAlgebra.coefOps]
  | cons p rest ih =>
    obtain ⟨k, x⟩ := p
    simp only [RecAlgebra.coefOps, varOp_congr h, ih]

theorem prodOps_congr (h : algFields sh = algFields sh') (c ri : ℕ)
    (l : List ((RecAlgebra.Key × RecAlgebra.Key) × F128)) :
    RecAlgebra.prodOps sh c ri l = RecAlgebra.prodOps sh' c ri l := by
  induction l with
  | nil => simp only [RecAlgebra.prodOps]
  | cons p rest ih =>
    obtain ⟨⟨a, b⟩, x⟩ := p
    simp only [RecAlgebra.prodOps, varOp_congr h, ih]

theorem accTerms_congr (h : algFields sh = algFields sh') (fin ri : ℕ) (cons : List (RecAlgebra.Key × F128)) :
    RecAlgebra.accTerms sh fin ri cons = RecAlgebra.accTerms sh' fin ri cons := by
  unfold RecAlgebra.accTerms
  rw [wi_congr h]

theorem buildRes_congr (h : algFields sh = algFields sh') (fin c ri : ℕ) (f : RecAlgebra.Lin) :
    RecAlgebra.buildRes sh fin c ri f = RecAlgebra.buildRes sh' fin c ri f := by
  unfold RecAlgebra.buildRes
  simp only [accTerms_congr h, coefOps_congr h, prodOps_congr h, orderOf_congr h]

theorem buildAll_congr (h : algFields sh = algFields sh') (fin c : ℕ) (l : List RecAlgebra.Lin) (ri : ℕ) :
    RecAlgebra.buildAll sh fin c l ri = RecAlgebra.buildAll sh' fin c l ri := by
  induction l generalizing ri with
  | nil => simp only [RecAlgebra.buildAll]
  | cons f rest ih => simp only [RecAlgebra.buildAll, buildRes_congr h, ih]

theorem build_congr (h : algFields sh = algFields sh') (lins : Array RecAlgebra.Lin) :
    RecAlgebra.build sh lins = RecAlgebra.build sh' lins := by
  obtain ⟨-, -, hc, hs, -⟩ := algFields_eq h
  unfold RecAlgebra.build
  simp only [hc, hs, buildAll_congr h, RecAlgebra.wNames, rounds_congr h]

theorem residualLins_congr (h : algFields sh = algFields sh') (coins : RecAlgebra.RoundCoins) (extras : Array Claim) :
    RecAlgebra.residualLins sh coins extras = RecAlgebra.residualLins sh' coins extras := by
  obtain ⟨hm, hk, hc, hs, -⟩ := algFields_eq h
  unfold RecAlgebra.residualLins RecAlgebra.run
  simp only [hm, hk, hc, hs, symbolic_congr h]

theorem structureOf_congr (h : algFields sh = algFields sh') :
    RecAlgebra.structureOf sh = RecAlgebra.structureOf sh' := by
  obtain ⟨hm, -, hc, -, -⟩ := algFields_eq h
  unfold RecAlgebra.structureOf RecAlgebra.extrasLike
  simp only [hm, hc, residualLins_congr h, build_congr h]

theorem verifierRows_congr (h : algFields sh = algFields sh') (rounds : Array (Array F128)) (extras : Array Claim)
    (comb : Array F128) :
    RecAlgebra.verifierRows sh rounds extras comb = RecAlgebra.verifierRows sh' rounds extras comb := by
  unfold RecAlgebra.verifierRows
  simp only [residualLins_congr h, build_congr h, structureOf_congr h]

end Congr

theorem verifierRows_filter {sh : Shape} {a b : Array (Array F128)} {extras : Array Claim} {comb : Array F128}
    (h : a.filter (·.size ≠ 0) = b.filter (·.size ≠ 0)) :
    RecAlgebra.verifierRows sh a extras comb = RecAlgebra.verifierRows sh b extras comb := by
  have e : RecAlgebra.RoundCoins.ofRounds a = RecAlgebra.RoundCoins.ofRounds b := by
    unfold RecAlgebra.RoundCoins.ofRounds
    rw [h]
  unfold RecAlgebra.verifierRows
  rw [e]

theorem verifierRows_pos {sh : Shape} {rounds : Array (Array F128)} {extras : Array Claim} {comb : Array F128}
    {w : Array F128 × RecAlgebra.Info RecAlgebra.Lin × Array (Array (Array F128))}
    (h : RecAlgebra.verifierRows sh rounds extras comb = .ok w) : 0 < rounds.size := by
  rcases Nat.eq_zero_or_pos rounds.size with h0 | h0
  · exfalso
    unfold RecAlgebra.verifierRows at h
    obtain ⟨⟨lins, info⟩, hl, -⟩ := FlockVBridge.PartS.bind_ok h
    obtain ⟨j, hr⟩ := RecAlgebra.residualLins_run hl
    obtain ⟨-, r0, -, -, -, k1, -, -, hz, -⟩ := RecAlgebra.run_stages hr
    unfold RecAlgebra.zerocheck at hz
    obtain ⟨v, k₁, hv, -⟩ := RecAlgebra.cm_bind hz
    rcases RecAlgebra.coinVec_ok hv with ⟨h6, -⟩ | ⟨-, hr6, -⟩
    · exact absurd h6 (by decide)
    · rw [Array.eq_empty_of_size_eq_zero h0] at hr6
      simp [RecAlgebra.RoundCoins.ofRounds] at hr6
  · exact h0

theorem openRows_dirs {inn : RecCoins.Inner} {LANES H : ℕ} {cr dr : Array (Array ℕ)}
    (h : openRows inn LANES H = some (cr, dr)) :
    ∃ sh extras l r0 r1, inn.algebra = .ok (sh, extras) ∧ innerLevels sh LANES H = [l] ∧
      inn.rep sh extras 0 = .ok r0 ∧ inn.rep sh extras 1 = .ok r1 ∧ dr = openDirs sh l H r0 r1 := by
  unfold openRows at h
  cases ha : inn.algebra with
  | error e => rw [ha] at h; cases h
  | ok p =>
    obtain ⟨sh, extras⟩ := p
    rw [ha] at h
    dsimp only at h
    cases hl : innerLevels sh LANES H with
    | nil => rw [hl] at h; simp at h
    | cons l t =>
      cases t with
      | cons _ _ => rw [hl] at h; simp at h
      | nil =>
        rw [hl] at h
        dsimp only at h
        cases h0 : inn.rep sh extras 0 <;> cases h1 : inn.rep sh extras 1 <;> rw [h0, h1] at h <;>
          simp only [reduceCtorEq] at h
        rename_i r0 r1
        simp only [Option.some.injEq, Prod.mk.injEq] at h
        exact ⟨sh, extras, l, r0, r1, rfl, hl, h0, h1, h.2.symm⟩

theorem openDirs_get (sh : Shape) (l H : ℕ) (r0 r1 : RecCoins.Rep) {r j : ℕ} (hr : r < 2)
    (hj : j < (sh.sched.levels[l]!).q) :
    (openDirs sh l H r0 r1)[r * (sh.sched.levels[l]!).q + j]! =
      RecCoins.dirsRow (((if r = 0 then r0 else r1).info.lg.conss[l]!).pos[j]!) H := by
  have hlt : r * (sh.sched.levels[l]!).q + j < (openDirs sh l H r0 r1).size := by
    unfold openDirs
    rw [Array.size_map, Array.size_range]
    have : r * (sh.sched.levels[l]!).q ≤ 1 * (sh.sched.levels[l]!).q := Nat.mul_le_mul_right _ (by omega)
    omega
  rw [getElem!_pos (openDirs sh l H r0 r1) _ hlt]
  unfold openDirs
  obtain rfl | rfl : r = 0 ∨ r = 1 := by omega
  · simp [hj, Nat.mod_eq_of_lt hj]
  · simp only [Nat.one_mul]
    have h1 : ¬ (sh.sched.levels[l]!).q + j < (sh.sched.levels[l]!).q := by omega
    have h2 : ((sh.sched.levels[l]!).q + j) % (sh.sched.levels[l]!).q = j := by
      rw [Nat.add_mod_left, Nat.mod_eq_of_lt hj]
    simp [h1, h2]

theorem innerLevels_unique {sh : Shape} {LANES H l l' : ℕ} (h : innerLevels sh LANES H = [l])
    (hl' : l' < sh.sched.levels.size) (hL : 2 ^ (sh.sched.levels[l']!).lanes = LANES)
    (hH : (sh.sched.levels[l']!).d - (sh.sched.levels[l']!).cap = H) : l' = l := by
  rw [getElem!_pos sh.sched.levels l' hl'] at hL hH
  have hm : l' ∈ innerLevels sh LANES H := by
    unfold innerLevels
    simp [List.mem_filter, hl', hL, hH]
  rw [h] at hm
  simpa using hm

theorem algebra_ok {inn : RecCoins.Inner} {sh : Shape} {extras : Array Claim}
    (h : inn.algebra = .ok (sh, extras)) :
    inn.extras = .ok extras ∧ RecResiduals.shape? inn.m inn.kLog (2 + extras.size) = some sh := by
  unfold RecCoins.Inner.algebra at h
  cases he : inn.extras with
  | error e => simp [he, bind, Except.bind] at h
  | ok ex =>
    cases hs : RecResiduals.shape? inn.m inn.kLog (2 + ex.size) with
    | none => simp [he, hs, bind, Except.bind, throw, throwThe, MonadExceptOf.throw] at h
    | some s =>
      simp [he, hs, bind, Except.bind, pure, Except.pure] at h
      obtain ⟨rfl, rfl⟩ := h
      exact ⟨rfl, hs⟩

theorem shape?_ok {m k c : ℕ} {sh : Shape} (h : RecResiduals.shape? m k c = some sh) :
    ∃ sch, fast100 m = some sch ∧ sh.m = m ∧ sh.kLog = k ∧ sh.claims = c ∧ sh.sched = sch ∧ sh.holo = false := by
  unfold RecResiduals.shape? at h
  cases hf : fast100 m with
  | none => simp [hf] at h
  | some s =>
    refine ⟨s, rfl, ?_⟩
    simp [hf] at h
    obtain ⟨-, rfl⟩ := h
    exact ⟨rfl, rfl, rfl, rfl, rfl⟩

def InnerCoinsR (untr : TrReader) (b : Brings) (Bk : Msg → ℕ) (s : ℕ) (x : Msg) (c : ℕ → Coin) : Prop :=
  ∀ (xO : Msg), showsPart untr b s x xO c → ∀ (i : RecIns) (spec : SessionSpec) (sts : Array Stmt),
    setupO xO = some (i, spec, sts) → ∀ (_ : 0 < sts.size) (f : RecCoins.Source) (inn : RecCoins.Inner),
      RecCoins.Source.atC3 i.src i.inner i.commits = some f → f.load sts[0].c.unit.name = .ok inn →
      ∃ v, setupOf false (xIOf x) = some v ∧ inn.m = (tabsOf v.2.2)[0]!.m ∧ inn.kLog = (tabsOf v.2.2)[0]!.kLog ∧
        inn.extras = (tabsOf v.2.2)[0]!.extra (pointsOf v.2.1 (((List.range (Bk (xIOf x))).map c).headD default)) ∧
        ∀ r < 2, ∀ rounds, inn.rounds r = .ok rounds → 0 < rounds.size →
          rounds.filter (·.size ≠ 0) =
            (canonRounds v.2.1 v.2.2 ((List.range (Bk (xIOf x))).map c).tail.dropLast r).filter (·.size ≠ 0)

def PosTieR (untr : TrReader) (b : Brings) (pieces : Msg → List OpenPiece)
    (posAt : Msg → List Coin → OpenPiece → ℕ) (Bk : Msg → ℕ) (s : ℕ) (x : Msg) (c : ℕ → Coin) : Prop :=
  ∀ (xO : Msg), showsPart untr b s x xO c → ∀ (i : RecIns) (spec : SessionSpec) (sts : Array Stmt),
    setupO xO = some (i, spec, sts) → ∀ (_ : 0 < sts.size) (f : RecCoins.Source) (inn : RecCoins.Inner),
      RecCoins.Source.atC3 i.src i.inner i.commits = some f → f.load sts[0].c.unit.name = .ok inn →
      ∀ (r : XR) (ss : Array VStar.Stmt) (lv LANES HH n : ℕ), readX x = some r →
        _root_.Flock.RecVStar.statements r.shape = .ok ss → ss[s]? = some (.recOpen lv LANES HH n) →
      ∀ (cr dr : Array (Array ℕ)), openRows inn LANES HH = some (cr, dr) →
      ∀ q ∈ pieces (xIOf x), ownQR x q = some s →
        dr[q.rep * qAt (xIOf x) q.level + q.query]! =
          RecCoins.dirsRow (posAt (xIOf x) ((List.range (Bk (xIOf x))).map c) q) HH

theorem posTieR_of {untr : TrReader} {b : Brings} {pieces : Msg → List OpenPiece} {Bk : Msg → ℕ} {s : ℕ}
    {x : Msg} {c : ℕ → Coin} {st : Claims.Recursion.CompiledStatement} (hst : parse x = some st)
    (hd : Claims.Recursion.Form.Dense st.form) (hR : PiecesInRangeR pieces)
    (hI : InnerCoinsR untr b Bk s x c) : PosTieR untr b pieces posAt Bk s x c := by
  intro xO hsh i spec sts hsu ht f inn hf hload r ss lv LANES HH n hr hss hsv cr dr hop q hq ho
  obtain ⟨v, hv, hm, hk, hex, hrd⟩ := hI xO hsh i spec sts hsu ht f inn hf hload
  obtain ⟨sh, extras, l, r0, r1, halg, hl, h0, h1, rfl⟩ := openRows_dirs hop
  obtain ⟨hext, hsh'⟩ := algebra_ok halg
  obtain ⟨sch, hsch, hshm, hshk, hshc, hshs, hshh⟩ := shape?_ok hsh'
  obtain ⟨hq0, r', ss', hr', hss', L', H', n', hsv'⟩ := ownQR_some ho
  rw [hr] at hr'
  cases hr'
  rw [hss] at hss'
  cases hss'
  rw [hsv] at hsv'
  simp only [Option.some.injEq, VStar.Stmt.recOpen.injEq] at hsv'
  obtain ⟨hlvq, -, -, -⟩ := hsv'
  obtain ⟨hlvs, hLANES, hHH, -⟩ := levelMatch hst hd hr hss hsv
  have hS : schAt (xIOf x) 0 = sch := by
    unfold schAt
    rw [hv]
    dsimp only
    rw [← hm, hsch, Option.getD_some]
  obtain ⟨hrep, hquery⟩ := hR (xIOf x) q hq
  rw [hq0, hS] at hquery
  have hlev : q.level < sch.levels.size := by
    by_contra hge
    rw [getElem!_neg sch.levels q.level hge] at hquery
    exact Nat.not_lt_zero _ hquery
  rw [hS, ← hlvs, hlvq] at hLANES hHH
  have hql : q.level = l := innerLevels_unique hl (by rw [hshs]; exact hlev) (by rw [hshs]; exact hLANES.symm)
    (by rw [hshs]; exact hHH.symm)
  have hqa : qAt (xIOf x) q.level = (sh.sched.levels[l]!).q := by
    unfold qAt
    rw [hS, hshs, hql]
  rw [hqa, openDirs_get sh l HH r0 r1 hrep (by rw [hshs, ← hql]; exact hquery)]
  refine congrArg (fun p => RecCoins.dirsRow p HH) ?_
  have hrq : inn.rep sh extras q.rep = .ok (if q.rep = 0 then r0 else r1) := by
    obtain h | h : q.rep = 0 ∨ q.rep = 1 := by omega
    · rw [h]; simpa using h0
    · rw [h]; simpa using h1
  obtain ⟨hrounds, hvr⟩ := rep_ok hrq
  rw [verifierRows_filter (hrd q.rep hrep _ hrounds (verifierRows_pos hvr)), verifierRows_comb] at hvr
  have hexs : ((tabsOf v.2.2)[0]!.extra (pointsOf v.2.1 (((List.range (Bk (xIOf x))).map c).headD default))) =
      .ok extras := by rw [← hex, hext]
  unfold posAt
  rw [hv]
  dsimp only
  rw [hq0, hexs]
  simp only [Except.toOption, Option.getD_some, Nat.mul_zero, Nat.zero_add]
  rw [verifierRows_congr (sh := shapeOf (tabsOf v.2.2)[0]! ((fast100 (tabsOf v.2.2)[0]!.m).getD default) extras.size)
    (sh' := sh) (by
      simp only [algFields, Prod.mk.injEq]
      refine ⟨by rw [hshm, hm]; rfl, by rw [hshk, hk]; rfl, by rw [hshc]; rfl, by rw [hshs, ← hm, hsch]; rfl,
        by rw [hshh]; rfl⟩)]
  cases h : RecAlgebra.verifierRows sh (canonRounds v.2.1 v.2.2 ((List.range (Bk (xIOf x))).map c).tail.dropLast q.rep)
      extras #[] with
  | error e => rw [h] at hvr; cases hvr
  | ok w =>
    rw [h] at hvr
    obtain ⟨w₁, w₂, w₃⟩ := w
    simp only [Functor.map, Except.map, Except.ok.injEq, Prod.mk.injEq] at hvr
    obtain ⟨-, hi, -⟩ := hvr
    rw [← hi, hql]

end RecDeploy.OpenLower
