module

public import Proofs.Flock.Claims.KS.ZkDecJ
public import Proofs.Flock.Claims.Recursion.DeployPremises
public import Proofs.Flock.Claims.Recursion.DeploySessions
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
import Proofs.Flock.Claims.Recursion.DeployMargin
import Proofs.Flock.Claims.Recursion.DeployScope

/-! Recursion completeness at `deployRec` (`vstar_complete`) from the named Props `NumsFixed`, `StagedSetsUp` and
`WitSatisfies` about V\*'s builder; `InnerAccepts` is a hypothesis about the coins, not a consequence of the
relation, and holds only at an inner statement the deployed plain verifier admits. -/

@[expose] public section

namespace RecDeploy

open Claims (Msg Out Prover run coinSeq replay)
open Flock
open FlockDeploy (Coin)

structure HonestPre (H : Msg → Msg) (IV : Claims.Verifier Coin) (b : Brings) (w : Claims.Flock.Wit) (x : Msg)
    (ω : Tape b) (cs : List Coin) (v : List (Msg × Msg)) : Prop where
  length : v.length = IV.k (xIOf x) + kS x
  coins : cs.length = v.length
  msg : ∀ i (h : i < v.length), v[i].1 = (((refP IV b w).next x (v.take i) ω).eval H).1
  ans : ∀ i (h : i < v.length), v[i].2 = (cs.getD i default).bytes

def InnerAccepts (H : Msg → Msg) (IV : Claims.Verifier Coin) (x : Msg) (cs : List Coin) (v : List (Msg × Msg)) :
    Prop :=
  (run H (replay ((v.take (IV.k (xIOf x))).map Prod.fst)) IV (xIOf x) () fun i => cs.getD i.val default).out = .acc

/-- A fact about the stager's layout (`vstar stage`). -/
def NumsFixed (st : Stage) : Prop :=
  ∀ (x : Msg) (ms ms' : List (Msg × Out)) (s : ℕ), (setupO (staged st x ms s)).isSome →
    (setupO (staged st x ms' s)).isSome → numsOf (staged st x ms s) = numsOf (staged st x ms' s)

/-- A fact about V\*'s builder. -/
def StagedSetsUp (H : Msg → Msg) (IV : Claims.Verifier Coin) (b : Brings) : Prop :=
  ∀ (x : Msg) (w : Claims.Flock.Wit) (ω : Tape b) (cs : List Coin) (v : List (Msg × Msg)),
    HonestPre H IV b w x ω cs v → InnerAccepts H IV x cs v → ∀ s < nS x,
      (setupO (staged b.st x (asRounds v) s)).isSome

/-- **A witness satisfies tables `sts`, table by table**: each fits the first table's blocks, and its share of the
witness's hidden values (`zOf`) satisfies its model statement. -/
def TablesSat (sts : Array Stmt) (w : Claims.Flock.Wit) : Prop :=
  ∀ c < sts.size, ∃ h : (Readable.Tab.ofStmt sts[c]!).Fits ((sts[0]?.map (·.m)).getD 0)
      ((sts[0]?.map (·.c.kLog)).getD 0),
    ((Readable.Tab.ofStmt sts[c]!).toModel h.2.2).Satisfies (Readable.zOf sts.size w c sts[c]!.m)

/-- A fact about V\*'s builder and its lowering. -/
def WitSatisfies (H : Msg → Msg) (IV : Claims.Verifier Coin) (b : Brings) : Prop :=
  ∀ (x : Msg) (w : Claims.Flock.Wit) (ω : Tape b) (cs : List Coin) (v : List (Msg × Msg)),
    HonestPre H IV b w x ω cs v → InnerAccepts H IV x cs v → ∀ s < nS x, ∀ i spec sts,
      setupO (staged b.st x (asRounds v) s) = some (i, spec, sts) →
        TablesSat sts (b.ref.wit w ω.1 (v.take (IV.k (xIOf x))) s)

def LinksHonest (linkOk : RecIns → ByteArray → ByteArray → Array Stmt → Bool) (H : Msg → Msg)
    (IV : Claims.Verifier Coin) (b : Brings) : Prop :=
  ∀ (x : Msg) (w : Claims.Flock.Wit) (ω : Tape b) (cs : List Coin) (v : List (Msg × Msg)),
    HonestPre H IV b w x ω cs v → ∀ s < nS x, ∀ i spec sts,
      setupO (staged b.st x (asRounds v) s) = some (i, spec, sts) →
        linkOk i (b.tr x (asRounds (v.take (IV.k (xIOf x))))) (b.st.record x (asRounds v)) sts = true

/-- Reads the link input `RecRaw.ins` reads (`linkInputOf`), not the transcript and record passed in. -/
def linkOkSetup (i : RecIns) (_ _ : ByteArray) (sts : Array Stmt) : Bool :=
  (match i.inner, i.link with
    | some n, some lk => Except.toBool (RecVStar.checkLinkRounds n lk)
    | _, _ => true) &&
    Except.toBool (sts.forM (RecVStar.checkTopRows · i.base.registered i.link)) &&
    Except.toBool (sts.forM (RecCoins.checkLink · i.base.registered (RecCoins.Source.atC3 i.src i.inner i.commits)
      i.link))

theorem RecIns.setup_linkRounds {s : RecIns} {spec : SessionSpec} {sts : Array Stmt} (h : s.setup = .ok (spec, sts))
    {n : RecVStar.Inner} {lk : RecVStar.InnerLink} (hn : s.inner = some n) (hl : s.link = some lk) :
    RecVStar.checkLinkRounds n lk = .ok () := by
  unfold RecIns.setup at h
  simp only [bind, Except.bind, pure, Except.pure, hn, hl] at h
  repeat' split at h
  all_goals first | (cases h; done) | skip
  all_goals
    obtain ⟨-, h2⟩ := Prod.mk.inj (Except.ok.inj h)
    subst h2
    assumption

theorem RecIns.setup_topRows {s : RecIns} {spec : SessionSpec} {sts : Array Stmt} (h : s.setup = .ok (spec, sts)) :
    sts.forM (RecVStar.checkTopRows · s.base.registered s.link) = .ok () := by
  unfold RecIns.setup at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals first | (cases h; done) | skip
  all_goals
    obtain ⟨-, h2⟩ := Prod.mk.inj (Except.ok.inj h)
    subst h2
    assumption

theorem RecIns.setup_checkLink {s : RecIns} {spec : SessionSpec} {sts : Array Stmt} (h : s.setup = .ok (spec, sts)) :
    sts.forM (RecCoins.checkLink · s.base.registered (RecCoins.Source.atC3 s.src s.inner s.commits) s.link) =
      .ok () := by
  unfold RecIns.setup at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals first | (cases h; done) | skip
  all_goals
    obtain ⟨-, h2⟩ := Prod.mk.inj (Except.ok.inj h)
    subst h2
    assumption

theorem linkOkSetup_of_setup {i : RecIns} {spec : SessionSpec} {sts : Array Stmt} (h : i.setup = .ok (spec, sts))
    (t r : ByteArray) : linkOkSetup i t r sts = true := by
  unfold linkOkSetup
  rw [RecIns.setup_topRows h, RecIns.setup_checkLink h]
  rcases hn : i.inner with _ | n <;> rcases hl : i.link with _ | lk
  · simp [Except.toBool]
  · simp [Except.toBool]
  · simp [Except.toBool]
  · simp [RecIns.setup_linkRounds h hn hl, Except.toBool]

theorem linksHonest_setup (H : Msg → Msg) (IV : Claims.Verifier Coin) (b : Brings) :
    LinksHonest linkOkSetup H IV b := by
  intro x w ω cs v hv s hs i spec sts hsu
  exact linkOkSetup_of_setup (setupO_some hsu).2 _ _

theorem RecRaw.ins_link {w : RecRaw} {s : RecIns} (h : w.ins = .ok s) : linkInputOf w s.src = .ok s.link := by
  unfold RecRaw.ins at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals first | (cases h; done) | (cases h; assumption)

theorem insO_staged_link {st : Stage} {x : Msg} {ms : List (Msg × Out)} {s : ℕ} {i : RecIns}
    (h : insO (staged st x ms s) = some i) :
    ∃ wr, decodeRec (staged st x ms s) = some wr ∧ wr.transcript = some (st.transcript x ms) ∧
      wr.innerSession = some (st.record x ms) ∧ linkInputOf wr i.src = .ok i.link := by
  have hne : staged st x ms s ≠ [] := fun he => by rw [he, insO_nil] at h; cases h
  obtain ⟨r, p, -, -, hd⟩ := staged_decode hne
  refine ⟨partRaw r st x ms s p, hd, rfl, rfl, ?_⟩
  unfold insO at h
  rw [hd] at h
  simp only [Option.bind_eq_bind, Option.bind_some] at h
  split_ifs at h
  · simp at h
  have h' : (partRaw r st x ms s p).ins.toOption = some i := by simpa using h
  exact RecRaw.ins_link (toOption_eq_some h')

def InnerAdmitted (x : Msg) : Prop :=
  ∃ ins spec sts, FlockDeploy.setupOf false (xIOf x) = some (ins, spec, sts) ∧ ins.read.check = .ok () ∧
    ∃ d, ins.read.draw = some d

theorem innerAdmitted_of_accepts {x : Msg} {cs : List Coin} {v : List (Msg × Msg)}
    (h : InnerAccepts FlockDeploy.sha512 (FlockDeploy.V false) x cs v) : InnerAdmitted x := by
  have hacc : ¬ Claims.Flock.NoAcc FlockDeploy.sha512 (FlockDeploy.V false) (xIOf x) := fun hn => hn _ _ _ _ h
  obtain ⟨ins, spec, sts, -, -, hs, -, hc⟩ := FlockDeploy.KS.acc_check hacc
  exact ⟨ins, spec, sts, hs, hc, FlockDeploy.KS.acc_sample_of_setup hacc hs⟩

theorem parseBytes_of_setupO {xO : Msg} {i : RecIns} {spec : SessionSpec} {sts : Array Stmt}
    (h : setupO xO = some (i, spec, sts)) :
    Readable.parseBytes readOfO xO = some ({ i.base.read with sts } : Readable.Parsed) := by
  obtain ⟨hins, hsu⟩ := setupO_some h
  have hr : readOfO xO = some i.base.read := by rw [readOfO, hins, Option.map_some]
  have hz := readOfO_zk hr
  unfold Readable.parseBytes Readable.Read.parse Readable.Read.tablesOf
  rw [hr, Option.bind_some, RecIns.setup_read hsu]
  simp only [hz, ↓reduceIte, RecIns.setup_shapeOk hsu hz, Option.map_some]

theorem R_of_setupO {xO : Msg} {i : RecIns} {spec : SessionSpec} {sts : Array Stmt}
    (h : setupO xO = some (i, spec, sts)) {w : Claims.Flock.Wit} (ht : TablesSat sts w) :
    Claims.Flock.R (deployZkRec Readable.relation).parse xO w := by
  refine ⟨Readable.toReadable Readable.relation ({ i.base.read with sts } : Readable.Parsed), ?_, ?_⟩
  · show Readable.parseOf Readable.relation readOfO xO = _
    unfold Readable.parseOf
    rw [parseBytes_of_setupO h, Option.map_some]
  · exact (Readable.relS_relation _ w).2 ht

theorem check_of_R {parse : Msg → Option Claims.Flock.Stmt} {x : Msg} {w : Claims.Flock.Wit}
    (h : Claims.Flock.R parse x w) : Claims.Flock.check parse x w = true := by
  obtain ⟨s, hs, hr⟩ := h
  unfold Claims.Flock.check
  rw [hs]
  exact decide_eq_true hr

theorem vstarAt_staged {st : Stage} (hN : NumsFixed st) {x : Msg} {ms : List (Msg × Out)} {s : ℕ}
    (hs : (setupO (staged st x ms s)).isSome) : vstarAt st x ms s = staged st x ms s := by
  have he : ∃ ms, (setupO (staged st x ms s)).isSome := ⟨ms, hs⟩
  have hp : (setupO (staged st x (pick st x s) s)).isSome := by
    unfold pick
    simp only [he, ↓reduceDIte]
    exact Classical.choose_spec he
  unfold vstarAt
  simp only [show numsOf (staged st x ms s) = nums st x s from hN x ms (pick st x s) s hs hp, ↓reduceIte]

theorem transcript_honest (b : Brings) (x : Msg) {v : List (Msg × Msg)} {n : ℕ} (hv : v.length = n + kS x) :
    b.st.transcript x (asRounds v) = b.tr x (asRounds (v.take n)) := by
  show b.tr x (asRounds v).dropLast = _
  unfold asRounds
  rw [List.dropLast_eq_take, List.length_map, hv, List.map_take]
  rfl

theorem vstar_staged_relation (H : Msg → Msg) (IV : Claims.Verifier Coin) (b : Brings) (hN : NumsFixed b.st)
    (hS : StagedSetsUp H IV b) (hW : WitSatisfies H IV b) {x : Msg} {w : Claims.Flock.Wit} {ω : Tape b}
    {cs : List Coin} {v : List (Msg × Msg)} (hv : HonestPre H IV b w x ω cs v) (hI : InnerAccepts H IV x cs v)
    {s : ℕ} (hs : s < nS x) :
    (vstar b.st x (asRounds v)).getD s [] = staged b.st x (asRounds v) s ∧
      ∃ i spec sts, setupO (staged b.st x (asRounds v) s) = some (i, spec, sts) ∧
        Claims.Flock.R (deployZkRec Readable.relation).parse (staged b.st x (asRounds v) s)
          (b.ref.wit w ω.1 (v.take (IV.k (xIOf x))) s) := by
  have hsu := hS x w ω cs v hv hI s hs
  obtain ⟨⟨i, spec, sts⟩, he⟩ := Option.isSome_iff_exists.1 hsu
  refine ⟨?_, i, spec, sts, he, R_of_setupO he (hW x w ω cs v hv hI s hs i spec sts he)⟩
  rw [vstar_getD _ _ _ hs]
  exact vstarAt_staged hN hsu

theorem vstar_complete (H : Msg → Msg) (IV : Claims.Verifier Coin) (b : Brings)
    (linkOk : RecIns → ByteArray → ByteArray → Array Stmt → Bool) (hN : NumsFixed b.st)
    (hS : StagedSetsUp H IV b) (hW : WitSatisfies H IV b) (hL : LinksHonest linkOk H IV b) {x : Msg}
    {w : Claims.Flock.Wit} {ω : Tape b} {cs : List Coin} {v : List (Msg × Msg)} (hv : HonestPre H IV b w x ω cs v)
    (hI : InnerAccepts H IV x cs v) {s : ℕ} (hs : s < nS x) :
    (vstar b.st x (asRounds v)).getD s [] = staged b.st x (asRounds v) s ∧
      ∃ i spec sts, setupO (staged b.st x (asRounds v) s) = some (i, spec, sts) ∧
        Claims.Flock.R (deployZkRec Readable.relation).parse (staged b.st x (asRounds v) s)
          (b.ref.wit w ω.1 (v.take (IV.k (xIOf x))) s) ∧
        linkOk i (b.st.transcript x (asRounds v)) (b.st.record x (asRounds v)) sts = true := by
  obtain ⟨he, i, spec, sts, hsu, hR⟩ := vstar_staged_relation H IV b hN hS hW hv hI hs
  refine ⟨he, i, spec, sts, hsu, hR, ?_⟩
  rw [transcript_honest b x hv.length]
  exact hL x w ω cs v hv s hs i spec sts hsu

theorem vstar_complete_setup (H : Msg → Msg) (IV : Claims.Verifier Coin) (b : Brings) (hN : NumsFixed b.st)
    (hS : StagedSetsUp H IV b) (hW : WitSatisfies H IV b) {x : Msg} {w : Claims.Flock.Wit} {ω : Tape b}
    {cs : List Coin} {v : List (Msg × Msg)} (hv : HonestPre H IV b w x ω cs v) (hI : InnerAccepts H IV x cs v)
    {s : ℕ} (hs : s < nS x) :
    (vstar b.st x (asRounds v)).getD s [] = staged b.st x (asRounds v) s ∧
      ∃ i spec sts, setupO (staged b.st x (asRounds v) s) = some (i, spec, sts) ∧
        Claims.Flock.R (deployZkRec Readable.relation).parse (staged b.st x (asRounds v) s)
          (b.ref.wit w ω.1 (v.take (IV.k (xIOf x))) s) ∧
        linkOkSetup i (b.st.transcript x (asRounds v)) (b.st.record x (asRounds v)) sts = true ∧
        ∃ wr, decodeRec (staged b.st x (asRounds v) s) = some wr ∧
          wr.transcript = some (b.tr x (asRounds (v.take (IV.k (xIOf x))))) ∧
          wr.innerSession = some (b.st.record x (asRounds v)) ∧ linkInputOf wr i.src = .ok i.link := by
  obtain ⟨he, i, spec, sts, hsu, hR, hl⟩ :=
    vstar_complete H IV b linkOkSetup hN hS hW (linksHonest_setup H IV b) hv hI hs
  obtain ⟨wr, hd, ht, hr, hk⟩ := insO_staged_link (setupO_some hsu).1
  refine ⟨he, i, spec, sts, hsu, hR, hl, wr, hd, ?_, hr, hk⟩
  rw [ht, transcript_honest b x hv.length]

theorem vstar_relation (H : Msg → Msg) (IV : Claims.Verifier Coin) (b : Brings) (hN : NumsFixed b.st)
    (hS : StagedSetsUp H IV b) (hW : WitSatisfies H IV b) {x : Msg} {w : Claims.Flock.Wit} {ω : Tape b}
    {cs : List Coin} {v : List (Msg × Msg)} (hv : HonestPre H IV b w x ω cs v) (hI : InnerAccepts H IV x cs v) :
    ∀ s < nS x, Claims.Flock.R (deployZkRec Readable.relation).parse ((vstar b.st x (asRounds v)).getD s [])
      (b.ref.wit w ω.1 (v.take (IV.k (xIOf x))) s) := by
  intro s hs
  obtain ⟨he, -, -, -, -, hR⟩ := vstar_staged_relation H IV b hN hS hW hv hI hs
  rw [he]
  exact hR

theorem outerHolds_honest (H : Msg → Msg) (IV : Claims.Verifier Coin) (b : Brings) (hN : NumsFixed b.st)
    (hS : StagedSetsUp H IV b) (hW : WitSatisfies H IV b) {x : Msg} {w : Claims.Flock.Wit} {ω : Tape b}
    {cs : List Coin} {v : List (Msg × Msg)} (hv : HonestPre H IV b w x ω cs v) (hI : InnerAccepts H IV x cs v) :
    Claims.Recursion.OuterHolds (deployZkRec Readable.relation) (vstar b.st x (asRounds v)) (nS x)
      ((List.range (nS x)).map (b.ref.wit w ω.1 (v.take (IV.k (xIOf x))))) := by
  refine ⟨by simp, fun s h => ?_⟩
  have hs : s < nS x := by simpa using h
  rw [List.getElem_map, List.getElem_range]
  exact check_of_R (vstar_relation H IV b hN hS hW hv hI s hs)

theorem stepWith_length_ge {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (x : Msg) (ω : Ω)
    (vn : List Msg → Claims.OC Out) (K : ℕ)
    (hvn : ∀ (as : List Msg) (a : Msg), as.length < K → ∃ m lo, (vn (as ++ [a])).eval H = (.msg m, lo)) :
    ∀ (f : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t : Claims.Transcript), as.length = t.msgs.length →
      K ≤ t.msgs.length + f → K ≤ (Claims.stepWith H P x ω vn f hist as t).msgs.length
  | 0, _, _, t, _, hK => by simpa [Claims.stepWith] using hK
  | f + 1, hist, as, t, ha, hK => by
    by_cases hk : K ≤ t.msgs.length
    · exact hk.trans (stepWith_mono H P x ω vn (f + 1) hist as t).1.length_le
    obtain ⟨m, lo, hm⟩ := hvn as ((P.next x hist ω).eval H).1 (by omega)
    rw [stepWith_succ_msg H P x ω vn f hist as t hm]
    exact stepWith_length_ge H P x ω vn K hvn f _ _ _ (by simp [ha]) (by simp; omega)

theorem run_length_ge (H : Msg → Msg) (IV : Claims.Verifier Coin) (st : Stage) {x : Msg} {xr : XR}
    (hx : readX x = some xr) {Ω : Type} (P : Prover Ω) (ω : Ω) (r : Fin ((V IV st).k x) → Coin) :
    IV.k (xIOf x) + kS x ≤ (run H P (V IV st) x ω r).msgs.length := by
  rw [run_eq]
  refine stepWith_length_ge H P x ω _ _ (fun as a hl => ?_) _ [] [] _ rfl
    (by simp only [List.length_nil, Nat.zero_add, V]; omega)
  refine ⟨(coinSeq r as.length).bytes, [], ?_⟩
  have hc := ract_inner (IV := IV) (st := st) hx (as := as ++ [a])
    (cs := (List.range (as ++ [a]).length).map (coinSeq r)) (c := coinSeq r as.length)
    (by simp only [List.length_append, List.length_singleton]; omega)
    (by simp [List.range_succ])
  exact next_coin H hc

theorem run_take_asRounds (H : Msg → Msg) (IV : Claims.Verifier Coin) (b : Brings) {x : Msg} {xr : XR}
    (hx : readX x = some xr) {Ω : Type} (P : Prover Ω) (ω : Ω) (r : Fin ((V IV b.st).k x) → Coin) :
    asRounds (((run H P (V IV b.st) x ω r).msgs.take (IV.k (xIOf x) + kS x)).map histOf) =
      (run H P (V IV b.st) x ω r).msgs.take (IV.k (xIOf x) + kS x) := by
  have hlen := run_length_ge H IV b.st hx P ω r
  unfold asRounds
  rw [List.map_map]
  apply List.ext_getElem (by simp)
  intro i h₁ h₂
  simp only [List.length_map, List.length_take] at h₁
  obtain ⟨a, ha⟩ := run_inner H IV b.st x P ω r hx (i := i) (by omega) (by omega)
  rw [List.getElem?_eq_getElem (by omega)] at ha
  simp only [List.getElem_map, List.getElem_take, Function.comp, histOf, Option.some.inj ha, Claims.Recursion.outMsg]

theorem honestPre_run (H : Msg → Msg) (IV : Claims.Verifier Coin) (b : Brings) (w : Claims.Flock.Wit) {x : Msg}
    {xr : XR} (hx : readX x = some xr) (ω : Tape b) (r : Fin ((V IV b.st).k x) → Coin) :
    HonestPre H IV b w x ω (runCs (IV.k (xIOf x) + kS x) r)
      (((run H (refP IV b w) (V IV b.st) x ω r).msgs.take (IV.k (xIOf x) + kS x)).map histOf) where
  length := by
    have := run_length_ge H IV b.st hx (refP IV b w) ω r
    simp only [List.length_map, List.length_take]
    omega
  coins := by
    have := run_length_ge H IV b.st hx (refP IV b w) ω r
    simp only [runCs, List.length_map, List.length_range, List.length_take]
    omega
  msg i h := by
    have hlen := run_length_ge H IV b.st hx (refP IV b w) ω r
    simp only [List.length_map, List.length_take] at h
    obtain ⟨new, hnew, hget⟩ := run_get H (refP IV b w) (V IV b.st) x ω r i (by omega)
    rw [List.getElem?_eq_getElem (by omega)] at hget
    have hv : (((run H (refP IV b w) (V IV b.st) x ω r).msgs.take (IV.k (xIOf x) + kS x)).map histOf).take i =
        new.map histOf := by
      rw [hnew, ← List.map_take, List.take_take, Nat.min_eq_left (by omega)]
    rw [hv]
    simp only [List.getElem_map, List.getElem_take, histOf, Option.some.inj hget]
  ans i h := by
    have hlen := run_length_ge H IV b.st hx (refP IV b w) ω r
    simp only [List.length_map, List.length_take] at h
    obtain ⟨a, ha⟩ := run_inner H IV b.st x (refP IV b w) ω r hx (i := i) (by omega) (by omega)
    rw [List.getElem?_eq_getElem (by omega)] at ha
    have hc : (runCs (IV.k (xIOf x) + kS x) r).getD i default = coinSeq r i := by
      simp [runCs, List.getD_eq_getElem?_getD, show i < IV.k (xIOf x) + kS x by omega]
    simp only [List.getElem_map, List.getElem_take, histOf, Option.some.inj ha, Claims.Recursion.outMsg, hc]

theorem readX_some_of_parse {x : Msg} {s : Claims.Recursion.CompiledStatement} (h : parse x = some s) : ∃ xr, readX x = some xr := by
  cases hx : readX x with
  | some xr => exact ⟨xr, rfl⟩
  | none =>
    simp [parse, RecScope.parseOf, RecScope.parseBytes, decodeR, hx] at h

theorem vstar_complete_run (H : Msg → Msg) (I : Claims.Recursion.Inner Coin Claims.Flock.Wit) (ivScope : Msg → Bool)
    (b : Brings) (linkOk : RecIns → ByteArray → ByteArray → Array Stmt → Bool) (hN : NumsFixed b.st)
    (hS : StagedSetsUp H I.V b) (hW : WitSatisfies H I.V b) (hL : LinksHonest linkOk H I.V b) {x : Msg}
    {w : Claims.Flock.Wit} (hR : Claims.Recursion.R (deployRec I.V ivScope b) I x w) (ω : Tape b)
    (r : Fin ((V I.V b.st).k x) → Coin)
    (hI : InnerAccepts H I.V x (runCs (I.V.k (xIOf x) + kS x) r)
      (((run H (refP I.V b w) (V I.V b.st) x ω r).msgs.take (I.V.k (xIOf x) + kS x)).map histOf))
    {s : ℕ} (hs : s < nS x) :
    ∃ i spec sts, setupO ((Claims.Recursion.runVstar H (deployRec I.V ivScope b) I (refP I.V b w) x ω
        r).getD s []) = some (i, spec, sts) ∧
      Claims.Flock.R (deployZkRec Readable.relation).parse
        ((Claims.Recursion.runVstar H (deployRec I.V ivScope b) I (refP I.V b w) x ω r).getD s [])
        (b.ref.wit w ω.1 (((run H (refP I.V b w) (V I.V b.st) x ω r).msgs.take (I.V.k (xIOf x))).map histOf) s) ∧
      linkOk i (b.st.transcript x ((run H (refP I.V b w) (V I.V b.st) x ω r).msgs.take (I.V.k (xIOf x) + kS x)))
        (b.st.record x ((run H (refP I.V b w) (V I.V b.st) x ω r).msgs.take (I.V.k (xIOf x) + kS x))) sts = true := by
  obtain ⟨s', hp, -, -⟩ := hR
  obtain ⟨xr, hx⟩ := readX_some_of_parse hp
  have hv := honestPre_run H I.V b w hx ω r
  obtain ⟨he, i, spec, sts, hsu, hrel, hl⟩ := vstar_complete H I.V b linkOk hN hS hW hL hv hI hs
  have hrv : Claims.Recursion.runVstar H (deployRec I.V ivScope b) I (refP I.V b w) x ω r =
      vstar b.st x (asRounds (((run H (refP I.V b w) (V I.V b.st) x ω r).msgs.take
        (I.V.k (xIOf x) + kS x)).map histOf)) := by
    rw [run_take_asRounds H I.V b hx]
    rfl
  have hwit : (((run H (refP I.V b w) (V I.V b.st) x ω r).msgs.take (I.V.k (xIOf x) + kS x)).map histOf).take
      (I.V.k (xIOf x)) = ((run H (refP I.V b w) (V I.V b.st) x ω r).msgs.take (I.V.k (xIOf x))).map histOf := by
    rw [← List.map_take, List.take_take, Nat.min_eq_left (Nat.le_add_right _ _)]
  rw [hwit] at hrel
  rw [run_take_asRounds H I.V b hx] at he hsu hrel hl
  refine ⟨i, spec, sts, ?_, ?_, hl⟩
  · rw [hrv, run_take_asRounds H I.V b hx, he]
    exact hsu
  · rw [hrv, run_take_asRounds H I.V b hx, he]
    exact hrel

theorem vstar_complete_run_setup (H : Msg → Msg) (I : Claims.Recursion.Inner Coin Claims.Flock.Wit)
    (ivScope : Msg → Bool) (b : Brings) (hN : NumsFixed b.st) (hS : StagedSetsUp H I.V b) (hW : WitSatisfies H I.V b)
    {x : Msg} {w : Claims.Flock.Wit} (hR : Claims.Recursion.R (deployRec I.V ivScope b) I x w) (ω : Tape b)
    (r : Fin ((V I.V b.st).k x) → Coin)
    (hI : InnerAccepts H I.V x (runCs (I.V.k (xIOf x) + kS x) r)
      (((run H (refP I.V b w) (V I.V b.st) x ω r).msgs.take (I.V.k (xIOf x) + kS x)).map histOf))
    {s : ℕ} (hs : s < nS x) :
    ∃ i spec sts, setupO ((Claims.Recursion.runVstar H (deployRec I.V ivScope b) I (refP I.V b w) x ω
        r).getD s []) = some (i, spec, sts) ∧
      Claims.Flock.R (deployZkRec Readable.relation).parse
        ((Claims.Recursion.runVstar H (deployRec I.V ivScope b) I (refP I.V b w) x ω r).getD s [])
        (b.ref.wit w ω.1 (((run H (refP I.V b w) (V I.V b.st) x ω r).msgs.take (I.V.k (xIOf x))).map histOf) s) ∧
      linkOkSetup i (b.st.transcript x ((run H (refP I.V b w) (V I.V b.st) x ω r).msgs.take (I.V.k (xIOf x) + kS x)))
        (b.st.record x ((run H (refP I.V b w) (V I.V b.st) x ω r).msgs.take (I.V.k (xIOf x) + kS x))) sts = true :=
  vstar_complete_run H I ivScope b linkOkSetup hN hS hW (linksHonest_setup H I.V b) hR ω r hI hs

end RecDeploy
