module

public import Proofs.Flock.Claims.Recursion.Deploy
public import Proofs.Flock.Claims.Recursion.DeployMargin
public import Proofs.Flock.Claims.Recursion.DeployScopeRuns
public import Proofs.Flock.Claims.Recursion.FlockKSAcc
public import Proofs.Flock.Claims.Recursion.FlockKSRecord
public import Proofs.Flock.Claims.Recursion.InnerAccepts
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
import Proofs.Flock.Claims.Recursion.DeployFacts
import Proofs.Flock.Claims.Recursion.DeploySessions
import Proofs.Flock.Claims.Recursion.KSStated

/-! The recursion's premises at `deployRec` beyond its code facts, reduced to facts about V\*'s circuits and the
readers, and its KS over the deployed plain inner record (`ksA_deployed`). -/

@[expose] public section

namespace RecDeploy

open Claims (Msg Out Prover run coinSeq replay)
open FlockDeploy (Coin)

variable {W : Type}

theorem scopeMatches (I : Claims.Recursion.Inner Coin W) (ivScope : Msg → Bool) (b : Brings)
    (hI : Claims.ScopeMatches I.Scope ivScope) : Claims.Recursion.ScopeMatches (deployRec I.V ivScope b) I :=
  Flock.RecScope.scopeMatches _ I decodeR ivScope rfl rfl hI

theorem scopeMatches_deployed (rel : Flock.Readable.Parsed → Flock.Readable.Relation)
    (decI : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace) (b : Brings) :
    Claims.Recursion.ScopeMatches (deployRec (FlockDeploy.V false) (FlockDeploy.deployPlain rel).scopeCheck b)
      (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlain rel) decI) :=
  Flock.RecScope.scopeMatches_deployed rel _ decI decodeR rfl rfl

def roundsOf (as : List Msg) (cs : List Coin) : List (Msg × Out) := as.zip (cs.map fun c => .msg c.bytes)

noncomputable def stagedAll (st : Stage) (x : Msg) (as : List Msg) (cs : List Coin) : List Msg :=
  (List.range (nS x)).map (staged st x (roundsOf as cs))

def callsOf (rd : Readers) (x : Msg) (ws : List Claims.Flock.Wit) : List (Msg × Msg) :=
  (ws.mapIdx fun s w => rd.calls x s w).flatten

def innerRunAt (H : Msg → Msg) (IV : Claims.Verifier Coin) (rd : Readers) (x : Msg) (ws : List Claims.Flock.Wit)
    (cs : List Coin) : Claims.Transcript :=
  run H (replay (rd.msgs x ws)) IV (xIOf x) () fun i => cs.getD i.val default

def runAs (n : ℕ) (t : Claims.Transcript) : List Msg := (t.msgs.take n).map Prod.fst

def runCs {k : ℕ} (n : ℕ) (r : Fin k → Coin) : List Coin := (List.range n).map (coinSeq r)

theorem coins_getD {k : ℕ} (r : Fin k → Coin) {m n : ℕ} (h : m ≤ n) :
    (fun i : Fin m => coinSeq r i.val) = fun i => (runCs n r).getD i.val default := by
  funext i
  have := i.isLt
  simp [runCs, List.getD_eq_getElem?_getD, show i.val < n by omega]

theorem innerView_eq (H : Msg → Msg) (IV : Claims.Verifier Coin) (rd : Readers) (x : Msg) {k : ℕ}
    (r : Fin k → Coin) (ws : List Claims.Flock.Wit) {n : ℕ} (h : IV.k (xIOf x) ≤ n) :
    innerView H IV rd x r ws = (innerRunAt H IV rd x ws (runCs n r)).seen := by
  unfold innerView innerRunAt
  rw [coins_getD r h]

section Run

variable (H : Msg → Msg) (IV : Claims.Verifier Coin) (b : Brings) (x : Msg) {Ω : Type} (P : Prover Ω) (ω : Ω)
  (r : Fin ((V IV b.st).k x) → Coin)

theorem runAs_length (hacc : (run H P (V IV b.st) x ω r).out = .acc) :
    (runAs (IV.k (xIOf x) + kS x) (run H P (V IV b.st) x ω r)).length = IV.k (xIOf x) + kS x := by
  have hK := run_acc_length H IV b.st x P ω r hacc
  have hM := roundsMargin_V H IV b.st x P ω r hacc
  simp only [runAs, List.length_map, List.length_take]
  omega

theorem runCs_length : (runCs (IV.k (xIOf x) + kS x) r).length = IV.k (xIOf x) + kS x := by
  simp [runCs]

theorem take_roundsOf (hacc : (run H P (V IV b.st) x ω r).out = .acc) :
    (run H P (V IV b.st) x ω r).msgs.take (IV.k (xIOf x) + kS x) =
      roundsOf (runAs (IV.k (xIOf x) + kS x) (run H P (V IV b.st) x ω r)) (runCs (IV.k (xIOf x) + kS x) r) := by
  have hl := runAs_length H IV b x P ω r hacc
  have hM := roundsMargin_V H IV b.st x P ω r hacc
  have h := innerPairs_run H IV b.st x P ω r hacc (off := IV.k (xIOf x) + kS x) le_rfl (by omega) []
    (cs := runCs (IV.k (xIOf x) + kS x) r) (List.take_of_length_le (by simp [runCs]))
  rw [List.append_nil] at h
  rw [← h]
  show innerPairs _ (runAs _ _) (runCs _ r) = _
  unfold innerPairs roundsOf
  rw [List.take_of_length_le hl.le, List.take_of_length_le (runCs_length IV b x r).le]

end Run

theorem runVstar_staged (H : Msg → Msg) (I : Claims.Recursion.Inner Coin W) (ivScope : Msg → Bool) (b : Brings)
    {x : Msg} {Ω : Type} (P : Prover Ω) (ω : Ω) (r : Fin ((deployRec I.V ivScope b).V.k x) → Coin)
    (hacc : (run H P (deployRec I.V ivScope b).V x ω r).out = .acc) :
    Claims.Recursion.runVstar H (deployRec I.V ivScope b) I P x ω r =
      stagedAll b.st x (runAs (I.V.k (xIOf x) + kS x) (run H P (V I.V b.st) x ω r))
        (runCs (I.V.k (xIOf x) + kS x) r) := by
  show vstar b.st x ((run H P (V I.V b.st) x ω r).msgs.take (I.V.k (xIOf x) + kS x)) = _
  have ht := take_roundsOf H I.V b x P ω r hacc
  rw [ht]
  unfold vstar stagedAll
  refine List.map_congr_left fun s hs => ?_
  rw [List.mem_range] at hs
  have h := (run_round H I.V b.st x P ω r hacc hs rfl rfl).1
  rw [ht] at h
  unfold vstarAt
  split_ifs
  rfl

/-- **Part `s`'s V\* statement as V\* stages it**: `xO` is `staged st x ms s` at some list of messages, its setup
decodes, and its read's check passes. -/
def StagedPart (st : Stage) (s : ℕ) (x xO : Msg) : Prop :=
  ∃ (ms : List (Msg × Out)) (i : RecIns) (spec : Flock.SessionSpec) (sts : Array Flock.Stmt),
    xO = staged st x ms s ∧ setupO xO = some (i, spec, sts) ∧ i.base.read.check = .ok ()

section StagedRun

variable (H : Msg → Msg) (IV : Claims.Verifier Coin) (st : Stage) (x : Msg) {Ω : Type} (P : Prover Ω) (ω : Ω)
  (r : Fin ((V IV st).k x) → Coin)

theorem stagedPart_run (hacc : (run H P (V IV st) x ω r).out = .acc) {s : ℕ} (hs : s < nS x) :
    StagedPart st s x (staged st x ((run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x) + kS x)) s) := by
  obtain ⟨i, f, hi, ⟨j, xO, as', cs', ins, spec, sts, link, c₁, mids, hr, hO⟩, hn⟩ :=
    run_decides H IV st x P ω r hacc hs
  rw [next_sess H hr] at hn
  have hacc' : ((nextO xO as' cs').eval H).1 = .acc := by
    rcases nextO_decide_out H hO with ho | ho
    · exact ho
    · rw [ho] at hn
      exact absurd rfl hn
  obtain ⟨⟨spec', sts', -, -, hc⟩, -⟩ := (nextO_decide H hO).1 hacc'
  obtain ⟨i', hi', hib⟩ := actO_decide hO
  obtain ⟨hk, -, -⟩ := ract_sess_locate hr
  obtain ⟨-, -, hxO⟩ := ract_sess hr
  have hK := run_acc_length H IV st x P ω r hacc
  have hlen : (((run H P (V IV st) x ω r).msgs.take (i + 1)).map Prod.fst).length = i + 1 := by
    simp only [List.length_map, List.length_take]
    omega
  rw [hlen] at hk
  have hip := innerPairs_run H IV st x P ω r hacc (off := i + 1) hk.le (by omega) []
    (range_map_take (coinSeq r) hk.le)
  rw [List.append_nil] at hip
  rw [hip] at hxO
  subst hxO
  refine ⟨_, i', spec, sts, rfl, hi', ?_⟩
  rw [hib]
  exact scope_of_checks hc

end StagedRun

theorem stagedAll_run (H : Msg → Msg) (IV : Claims.Verifier Coin) (b : Brings) (x : Msg) {Ω : Type} (P : Prover Ω)
    (ω : Ω) (r : Fin ((V IV b.st).k x) → Coin) (hacc : (run H P (V IV b.st) x ω r).out = .acc) :
    ∀ s < nS x, StagedPart b.st s x ((stagedAll b.st x (runAs (IV.k (xIOf x) + kS x) (run H P (V IV b.st) x ω r))
      (runCs (IV.k (xIOf x) + kS x) r)).getD s []) := by
  intro s hs
  rw [stagedAll, List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range hs, Option.map_some,
    Option.getD_some, ← take_roundsOf H IV b x P ω r hacc]
  exact stagedPart_run H IV b.st x P ω r hacc hs

section Readers

variable {Col T : Type}

/-- A code fact about V\*'s circuits and the readers. -/
def ReadersTrue (H : Msg → Msg) (IV : Claims.Verifier Coin) (b : Brings) (DO : Claims.Flock.Deploy Coin Col T) :
    Prop :=
  ∀ (x : Msg) (as : List Msg) (cs : List Coin) (ws : List Claims.Flock.Wit), as.length = IV.k (xIOf x) + kS x →
    cs.length = as.length → (∀ s < nS x, StagedPart b.st s x ((stagedAll b.st x as cs).getD s [])) →
      Claims.Recursion.OuterHolds DO (stagedAll b.st x as cs) (nS x) ws → ∀ e ∈ callsOf b.rd x ws, H e.1 = e.2

/-- A code fact about V\*'s circuits and the readers, given only off a collision: where two parts hold different
messages at one round, the run's calls on the message the readers take aren't the other part's. -/
def ReadersLogColl (H : Msg → Msg) (IV : Claims.Verifier Coin) (b : Brings) (DO : Claims.Flock.Deploy Coin Col T) :
    Prop :=
  ∀ (x : Msg) (as : List Msg) (cs : List Coin) (ws : List Claims.Flock.Wit), as.length = IV.k (xIOf x) + kS x →
    cs.length = as.length → (∀ s < nS x, StagedPart b.st s x ((stagedAll b.st x as cs).getD s [])) →
      Claims.Recursion.OuterHolds DO (stagedAll b.st x as cs) (nS x) ws →
      (∀ e ∈ callsOf b.rd x ws, H e.1 = e.2) →
        (∀ e ∈ (innerRunAt H IV b.rd x ws cs).vLog, e ∈ callsOf b.rd x ws) ∨
          ∃ u v, Claims.findColl (callsOf b.rd x ws) = some (u, v) ∧ Claims.Collision H u v

/-- A code fact about V\*'s circuits and the readers. -/
def ReadersOpenPlayed (H : Msg → Msg) (IV : Claims.Verifier Coin) (b : Brings) (DO : Claims.Flock.Deploy Coin Col T)
    (opening : Claims.Recursion.Opening) : Prop :=
  ∀ (x : Msg) (as : List Msg) (cs : List Coin) (ws : List Claims.Flock.Wit), as.length = IV.k (xIOf x) + kS x →
    cs.length = as.length → (∀ s < nS x, StagedPart b.st s x ((stagedAll b.st x as cs).getD s [])) →
      Claims.Recursion.OuterHolds DO (stagedAll b.st x as cs) (nS x) ws →
      ∀ j < (innerRunAt H IV b.rd x ws cs).msgs.length, ∃ es, es ⊆ callsOf b.rd x ws ∧
        opening.opens (as.getD j []) (((innerRunAt H IV b.rd x ws cs).msgs.map Prod.fst).getD j []) es

/-- A code fact about V\*'s circuits and the readers. -/
def ReadersAccept (H : Msg → Msg) (IV : Claims.Verifier Coin) (b : Brings) (DO : Claims.Flock.Deploy Coin Col T)
    (shape : Claims.Recursion.Form → Prop) : Prop :=
  ∀ (x : Msg) (s : Claims.Recursion.CompiledStatement), parse x = some s → shape s.form →
    ∀ (as : List Msg) (cs : List Coin) (ws : List Claims.Flock.Wit), as.length = IV.k (xIOf x) + kS x →
      cs.length = as.length → (∀ s < nS x, StagedPart b.st s x ((stagedAll b.st x as cs).getD s [])) →
        Claims.Recursion.OuterHolds DO (stagedAll b.st x as cs) (nS x) ws →
        (∀ e ∈ callsOf b.rd x ws, H e.1 = e.2) →
          (innerRunAt H IV b.rd x ws cs).out = .acc ∨
            ∃ u v, Claims.findColl (callsOf b.rd x ws) = some (u, v) ∧ Claims.Collision H u v

variable (H : Msg → Msg) (I : Claims.Recursion.Inner Coin W) (ivScope : Msg → Bool) (b : Brings)
  (DO : Claims.Flock.Deploy Coin Col T)

theorem callsTrue (hT : ReadersTrue H I.V b DO) : Claims.Recursion.CallsTrue H (deployRec I.V ivScope b) I DO := by
  intro decO x Ω P ω r c ws hg
  have ho := Claims.Recursion.eoGives_outerHolds hg
  rw [runVstar_staged H I ivScope b P ω r hg.1] at ho
  have hl := runAs_length H I.V b x P ω r hg.1
  have hc : (runCs (I.V.k (xIOf x) + kS x) r).length = _ := (runCs_length I.V b x r).trans hl.symm
  exact hT x _ _ ws hl hc (stagedAll_run H I.V b x P ω r hg.1) ho

theorem callsLogColl (hT : ReadersTrue H I.V b DO) (hL : ReadersLogColl H I.V b DO) :
    Claims.Recursion.CallsLogColl H (deployRec I.V ivScope b) I DO := by
  intro decO x Ω P ω r c ws hg
  have ho := Claims.Recursion.eoGives_outerHolds hg
  rw [runVstar_staged H I ivScope b P ω r hg.1] at ho
  have hl := runAs_length H I.V b x P ω r hg.1
  have hc : (runCs (I.V.k (xIOf x) + kS x) r).length = _ := (runCs_length I.V b x r).trans hl.symm
  have hrun : Claims.Recursion.innerRun H (deployRec I.V ivScope b) I x r ws =
      innerRunAt H I.V b.rd x ws (runCs (I.V.k (xIOf x) + kS x) r) := by
    show run H (replay ((innerView H I.V b.rd x r ws).msgs.map Prod.fst)) I.V (xIOf x) ()
      (fun i => coinSeq r i.val) = _
    unfold innerView
    rw [run_replay_seen, coins_getD r (show I.V.k (xIOf x) ≤ I.V.k (xIOf x) + kS x by omega)]
    rfl
  rw [hrun]
  have hsp := stagedAll_run H I.V b x P ω r hg.1
  exact hL x _ _ ws hl hc hsp ho (hT x _ _ ws hl hc hsp ho)

theorem callsOpenPlayed (opening : Claims.Recursion.Opening) (hO : ReadersOpenPlayed H I.V b DO opening) :
    Claims.Recursion.CallsOpenPlayed H (deployRec I.V ivScope b) I DO opening := by
  intro decO x Ω P ω r c ws hg j hj
  have ho := Claims.Recursion.eoGives_outerHolds hg
  rw [runVstar_staged H I ivScope b P ω r hg.1] at ho
  have hl := runAs_length H I.V b x P ω r hg.1
  have hc : (runCs (I.V.k (xIOf x) + kS x) r).length = _ := (runCs_length I.V b x r).trans hl.symm
  have hv : (deployRec I.V ivScope b).innerView H x r ws =
      (innerRunAt H I.V b.rd x ws (runCs (I.V.k (xIOf x) + kS x) r)).seen :=
    innerView_eq H I.V b.rd x r ws (Nat.le_add_right _ _)
  rw [hv] at hj ⊢
  obtain ⟨es, hes, hop⟩ := hO x _ _ ws hl hc (stagedAll_run H I.V b x P ω r hg.1) ho j hj
  refine ⟨es, hes, ?_⟩
  have hjk : j < I.V.k (xIOf x) :=
    Nat.lt_of_lt_of_le hj (Claims.Recursion.run_length_le H _ I.V (xIOf x) () _)
  have hjn : j < I.V.k (xIOf x) + kS x := Nat.lt_of_lt_of_le hjk (Nat.le_add_right _ _)
  have ha : ((run H P (deployRec I.V ivScope b).V x ω r).msgs.map Prod.fst).getD j [] =
      (runAs (I.V.k (xIOf x) + kS x) (run H P (V I.V b.st) x ω r)).getD j [] := by
    show ((run H P (V I.V b.st) x ω r).msgs.map Prod.fst).getD j [] = _
    simp only [runAs, List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_take_of_lt hjn]
  rw [ha]
  exact hop

theorem vstarAccepts (shape : Claims.Recursion.Form → Prop) (hA : ReadersAccept H I.V b DO shape) :
    Claims.Recursion.VStarAccepts H (deployRec I.V ivScope b) I DO shape := by
  intro x s hs hf Ω P ω r hacc ws ho ht
  rw [runVstar_staged H I ivScope b P ω r hacc] at ho
  have hl := runAs_length H I.V b x P ω r hacc
  have hc : (runCs (I.V.k (xIOf x) + kS x) r).length = _ := (runCs_length I.V b x r).trans hl.symm
  have hv : (deployRec I.V ivScope b).innerView H x r ws =
      (innerRunAt H I.V b.rd x ws (runCs (I.V.k (xIOf x) + kS x) r)).seen :=
    innerView_eq H I.V b.rd x r ws (Nat.le_add_right _ _)
  rw [hv]
  exact hA x s hs hf _ _ ws hl hc (stagedAll_run H I.V b x P ω r hacc) ho ht

theorem innerAcceptsColl (hT : ReadersTrue H I.V b DO)
    (hD : ReadersAccept H I.V b DO Claims.Recursion.Form.Dense)
    (hH : ReadersAccept H I.V b DO Claims.Recursion.Form.Holo)
    (hS : ReadersAccept H I.V b DO Claims.Recursion.Form.Sparse) :
    Claims.Recursion.InnerAcceptsColl H (deployRec I.V ivScope b) I DO :=
  Claims.Recursion.innerAcceptsColl_of_vstar (refuses I.V ivScope b H) (callsTrue H I ivScope b DO hT)
    (vstarAccepts H I ivScope b DO _ hD) (vstarAccepts H I ivScope b DO _ hH) (vstarAccepts H I ivScope b DO _ hS)

end Readers

theorem ksA_deployed (rel relO : Flock.Readable.Parsed → Flock.Readable.Relation) (ivScope : Msg → Bool)
    (b : Brings) (opening : Claims.Recursion.Opening)
    (hF : Claims.Flock.KS FlockDeploy.sha512 (deployZkRec relO))
    (htab : Claims.Recursion.TableBoundθOnAcc FlockDeploy.sha512 (FlockDeploy.deployPlain rel)
      (FlockDeploy.KS.wordLen false) (FlockDeploy.KS.wide false))
    (hT : ReadersTrue FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec relO))
    (hL : ReadersLogColl FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec relO))
    (hO : ReadersOpenPlayed FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec relO) opening)
    (hD : ReadersAccept FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec relO) Claims.Recursion.Form.Dense)
    (hH : ReadersAccept FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec relO) Claims.Recursion.Form.Holo)
    (hS : ReadersAccept FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec relO) Claims.Recursion.Form.Sparse)
    (decI : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace)
    (hdec : Claims.Recursion.decOK (FlockDeploy.deployPlainTrim rel) decI) :
    Claims.Recursion.KSA FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b)
      (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlainTrim rel) decI) (deployZkRec relO) :=
  let I := Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlainTrim rel) decI
  Claims.Recursion.KS_of_flockKS_coll FlockDeploy.sha512 (deployRec I.V ivScope b) I (deployZkRec relO) opening
    FlockDeploy.decCoin hF (FlockDeploy.KS.plain_KSAbort rel htab decI hdec)
    (refuses I.V ivScope b FlockDeploy.sha512) (innerRounds I ivScope b) (sessionNums I.V ivScope b relO)
    (tableErrNonneg I.V ivScope b) (sessionRounds I ivScope b relO) (sessionAccepts I.V ivScope b)
    (sessionRuns_sha512 I ivScope b relO) (innerViewRun FlockDeploy.sha512 I ivScope b (deployZkRec relO))
    (callsTrue FlockDeploy.sha512 I ivScope b (deployZkRec relO) hT)
    (callsLogColl FlockDeploy.sha512 I ivScope b (deployZkRec relO) hT hL)
    (callsOpenPlayed FlockDeploy.sha512 I ivScope b (deployZkRec relO) opening hO)
    (innerAcceptsColl FlockDeploy.sha512 I ivScope b (deployZkRec relO) hT hD hH hS)
    (roundsMargin_deployRec_V false FlockDeploy.sha512 I rfl ivScope b)
    (FlockDeploy.coinsShown_V FlockDeploy.sha512 false) (callsNil I.V ivScope b FlockDeploy.sha512)

/-- No theorem evaluates its level (`Claims.Recursion.lam`) at the deployment. -/
theorem ksFlock_deployed (rel relO : Flock.Readable.Parsed → Flock.Readable.Relation) (ivScope : Msg → Bool)
    (b : Brings) (opening : Claims.Recursion.Opening)
    (hF : Claims.Flock.KS FlockDeploy.sha512 (deployZkRec relO))
    (htab : Claims.Recursion.TableBoundθOnAcc FlockDeploy.sha512 (FlockDeploy.deployPlain rel)
      (FlockDeploy.KS.wordLen false) (FlockDeploy.KS.wide false))
    (hT : ReadersTrue FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec relO))
    (hL : ReadersLogColl FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec relO))
    (hO : ReadersOpenPlayed FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec relO) opening)
    (hD : ReadersAccept FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec relO) Claims.Recursion.Form.Dense)
    (hH : ReadersAccept FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec relO) Claims.Recursion.Form.Holo)
    (hS : ReadersAccept FlockDeploy.sha512 (FlockDeploy.V false) b (deployZkRec relO) Claims.Recursion.Form.Sparse) :
    Claims.Recursion.KSFlock FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b)
      (FlockDeploy.deployPlainTrim rel) (deployZkRec relO) :=
  (Claims.Recursion.KSFlockA_iff_KSFlock _ _ _ _).1 fun decI hdec =>
    ksA_deployed rel relO ivScope b opening hF htab hT hL hO hD hH hS decI hdec

end RecDeploy
