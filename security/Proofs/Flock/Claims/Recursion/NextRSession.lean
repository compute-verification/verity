module

public import Proofs.Flock.Claims.Complete.Json.Goals
public import Proofs.Flock.Claims.Complete.Json.Session
public import Proofs.Flock.Claims.Recursion.NextRRep
public import Proofs.Flock.Claims.Recursion.NextRSetup
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
import Proofs.Flock.Claims.Complete.Walk
import Proofs.Flock.Claims.Complete.Zk.Walk
import Proofs.Flock.Claims.DeployK
import Proofs.Flock.Claims.Recursion.SchedCount
import Proofs.Flock.Soundness.Discharge.DecodeZL.Sizes

/-! The honest session's proofs are its reps', so the honest run decides at its own proofs. -/

@[expose] public section

namespace FlockComplete.JsonRT

open Flock Transcript FlockComplete FlockDeploy

theorem range_map_getElem! {β : Type} [Inhabited β] (g : ℕ → β) {n i : ℕ} (hi : i < n) :
    ((Array.range n).map g)[i]! = g i := by
  rw [getElem!_pos ((Array.range n).map g) i (by simpa using hi)]
  simp

theorem repP_cap (resp : ℕ → Claims.Msg) (hsP : Hashes PS) (st : Setup) (sch : Schedule) (tr : Trace)
    (tbl : Array F128) (cm0 : Commit) (extras : Array Claim) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64)
    (t : Transcript) :
    Ret ((repP (asksS resp) hsP st sch tr tbl cm0 extras salt nonce).run t) fun pt => pt.1.cap = cm0.cap := by
  unfold repP
  refine FlockComplete.Replay.Ret.run_bind (FlockComplete.Ret.true _) fun ⟨zp, zc⟩ _ _ => ?_
  refine FlockComplete.Replay.Ret.run_bind (FlockComplete.Ret.true _) fun ⟨lp, lc⟩ _ _ => ?_
  refine FlockComplete.Replay.Ret.run_bind (FlockComplete.Ret.true _) fun ⟨ss, T, sws⟩ _ _ => ?_
  refine FlockComplete.Replay.Ret.run_bind (FlockComplete.Ret.true _) fun lg _ _ => ?_
  rw [StateT.run_pure]
  exact FlockComplete.Ret.pure rfl

def sendsP (resp : ℕ → Claims.Msg) (spec : SessionSpec) (tabs : Array Setup) (k : ℕ) : ℕ :=
  (3 + (tabs[k / 2]!.m - 6)) + (3 + (tabs[k / 2]!.fold.kLog - 6)) +
    (2 + (match tabs[k / 2]!.extra (ptsOf spec (coinOf (resp 0))) with
      | .ok e => e.size
      | .error _ => 0) + 1) +
    (((fast100 tabs[k / 2]!.m).getD default).k0 + 5 + 11 * (((fast100 tabs[k / 2]!.m).getD default).levels.size - 1))

def RepRunP (resp : ℕ → Claims.Msg) (h : ByteArray → ByteArray) (spec : SessionSpec) (tabs : Array Setup)
    (schs : Array Schedule) (trs : Array Trace) (cms : Array Commit) (tape : Tape) (k : ℕ) (f : ByteArray) : Prop :=
  ∃ e pt l l', tabs[k / 2]!.extra (ptsOf spec (coinOf (resp 0))) = .ok e ∧
    l.size = 1 + ((List.range k).map (sendsP resp spec tabs)).sum ∧
    ((repP (asksS resp) (Hashes.ofSha512 h) tabs[k / 2]! schs[k / 2]! trs[k / 2]!
      (packed trs[k / 2]! tabs[k / 2]!.m) cms[k / 2]! e (tape.salt (k / 2) (k % 2)) (tape.nonce (k / 2) (k % 2))).run
        (Transcript.init #[] (spec.domain spec.tables[k / 2]! (k % 2)).toUTF8)).go l = (.ok pt, l') ∧
    f = FlockVBridge.Family.encodeProof (FlockComplete.shapeOf tabs[k / 2]! schs[k / 2]! e.size) pt.1

theorem sessionP_repRuns (H : Claims.Msg → Claims.Msg) (resp : ℕ → Claims.Msg) (spec : SessionSpec)
    (tabs : Array Setup) (trs : Array Trace) (tape : Tape) (hts : tabs.size = spec.tables.size)
    {r : String × Array ByteArray} {bodies : Array Claims.Msg}
    (hrun : (sessionP (asksS resp) (Hashes.ofSha512 (Claims.hOf H)) spec tabs trs tape).go #[] = (.ok r, bodies)) :
    ∃ schs cms, (∀ i < tabs.size, fast100 tabs[i]!.m = some schs[i]! ∧
      commitP (Hashes.ofSha512 (Claims.hOf H) : Hashes V) tabs[i]!.merkle (schs[i]!.levels.getD 0 default)
        (packed trs[i]! tabs[i]!.m) (tape.salt i 0 0) = .ok cms[i]!) ∧
      r.2.size = 2 * tabs.size ∧
      ∀ k < 2 * tabs.size, RepRunP resp (Claims.hOf H) spec tabs schs trs cms tape k r.2[k]! := by
  unfold sessionP at hrun
  obtain ⟨schs, l1, hA, hrun⟩ := go_bind_ok hrun
  have hschs := Ret.go (Ret.mapM_array _ (fun st' sch => fast100 st'.m = some sch) _ fun a _ => by
    cases hf : fast100 a.m with
    | none => exact Ret.throw' _
    | some sch => exact Ret.pure rfl) hA
  have hl1 := Still.go (Still.mapM_array (fun a => by
    split
    · exact Still.pure _
    · exact Still.throw _) _) hA
  subst hl1
  obtain ⟨hsz₁, hf₁⟩ := hschs
  dsimp only at hrun
  obtain ⟨cms, l2, hB, hrun⟩ := go_bind_ok hrun
  have hcms := Ret.go (Ret.mapM_array _ (fun i cm => commitP (Hashes.ofSha512 (Claims.hOf H) : Hashes V)
      tabs[i]!.merkle (schs[i]!.levels.getD 0 default) (packed trs[i]! tabs[i]!.m) (tape.salt i 0 0) = .ok cm) _
    fun i hi => by
      simp only [range_map_getElem! _ (Array.mem_range.1 hi)]
      exact Ret.of_pureS (commitP_pureS _ _ _ _ _) fun cm e => e) hB
  have hl2 := Still.go (Still.mapM_array (fun i => Still.of_pureS (commitP_pureS _ _ _ _ _)) _) hB
  subst hl2
  obtain ⟨hsz₂, hf₂⟩ := hcms
  obtain ⟨c₁, l₃, hL, hrun⟩ := go_bind_ok hrun
  rw [link_go] at hL
  simp only [Prod.mk.injEq, Except.ok.injEq, Array.size_empty] at hL
  obtain ⟨rfl, rfl⟩ := hL
  obtain ⟨⟨ss, ps⟩, l₄, hF, hrun⟩ := go_bind_ok hrun
  have hsch : ∀ i < tabs.size, fast100 tabs[i]!.m = some schs[i]! := fun i hi => by
    rw [getElem!_pos tabs i hi, getElem!_pos schs i (by omega)]
    exact hf₁ i hi (by omega)
  have hinv := forIn_array_idx _ _ (fun k (b : Array (Array Round) × Array ByteArray) (l : Array Claims.Msg) =>
      l.size = 1 + ((List.range k).map (sendsP resp spec tabs)).sum ∧
      b.2.size = k ∧ ∀ j < k, RepRunP resp (Claims.hOf H) spec tabs schs trs cms tape j b.2[j]!) ?_
    (by exact ⟨by simp, rfl, by simp⟩) hF
  · obtain ⟨-, h₁, h₂⟩ := hinv
    rw [repStreams_size, ← hts] at h₁ h₂
    dsimp only at hrun
    obtain ⟨rec, l₅, -, hrun⟩ := go_bind_ok hrun
    simp only [go_pure, Prod.mk.injEq, Except.ok.injEq] at hrun
    obtain ⟨rfl, rfl⟩ := hrun
    refine ⟨schs, cms, fun i hi => ⟨hsch i hi, ?_⟩, h₁, h₂⟩
    · have := hf₂ i (by simpa using hi) (by simp at hsz₂; omega)
      simp only [Array.getElem_range] at this
      rw [getElem!_pos cms i (by simp at hsz₂; omega)]
      exact this
  · rintro k ⟨i, rp, name⟩ ⟨ss, ps⟩ l res l' hk ⟨h₀, h₂, h₅⟩ e
    dsimp only at h₂ h₅
    obtain ⟨hk', hkv⟩ := Array.getElem?_eq_some_iff.1 hk
    rw [repStreams_getElem spec hk'] at hkv
    simp only [Prod.mk.injEq] at hkv
    obtain ⟨rfl, rfl, -⟩ := hkv
    have hk2 : k / 2 < tabs.size := by
      rw [repStreams_size] at hk'
      omega
    obtain ⟨extras, l₁, e1, e⟩ := go_bind_ok e
    rw [monadLift_go] at e1
    simp only [Prod.mk.injEq] at e1
    obtain ⟨hex, rfl⟩ := e1
    obtain ⟨⟨pf, tr'⟩, l₂, e2, e⟩ := go_bind_ok e
    simp only [go_pure, Prod.mk.injEq, Except.ok.injEq] at e
    obtain ⟨rfl, rfl⟩ := e
    rw [range_map_getElem! _ hk2] at e2
    have hex' : tabs[k / 2]!.extra (ptsOf spec (coinOf (resp 0))) = .ok extras := by simpa [ptsOf] using hex
    refine ⟨_, rfl, ?_, by simp [h₂], fun j hj => ?_⟩
    · have hm := (RecDeploy.fast100_sizes (hsch _ hk2)).1
      rw [FlockComplete.Replay.repP_count resp _ _ _ _ _ _ _ _ _ (by omega) (hsch _ hk2) _ l e2, h₀,
        List.range_succ, List.map_append, List.sum_append]
      simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, sendsP, hsch _ hk2, Option.getD_some,
        hex']
      omega
    rcases Nat.lt_succ_iff_lt_or_eq.1 hj with hj | rfl
    · rw [push_getElem!_lt _ _ (by omega)]
      exact h₅ j hj
    · refine ⟨extras, (pf, tr'), _, l₂, by simpa [ptsOf] using hex, h₀, e2, ?_⟩
      rw [← h₂, push_getElem!_size]

end FlockComplete.JsonRT

namespace FlockDeploy

open Flock FlockComplete

def filePs (spec : SessionSpec) (sts : Array Stmt) (c₁ : Coin) (pfs : Array ByteArray) : Array Flock.Proof :=
  pfs.mapIdx fun k b => match decodeProof (shapeAt (tabsOf sts) (pointsOf spec c₁) k) b with
    | .ok p => p
    | .error _ => default

def fileCaps (spec : SessionSpec) (sts : Array Stmt) (c₁ : Coin) (pfs : Array ByteArray) :
    Array (Array ByteArray) :=
  (Array.range sts.size).map fun i => (filePs spec sts c₁ pfs)[2 * i]!.cap

theorem filePs_size (spec : SessionSpec) (sts : Array Stmt) (c₁ : Coin) (pfs : Array ByteArray) :
    (filePs spec sts c₁ pfs).size = pfs.size := by
  simp only [filePs, Array.size_mapIdx]

theorem filePs_get {spec : SessionSpec} {sts : Array Stmt} {c₁ : Coin} {pfs : Array ByteArray} {k : ℕ}
    {p : Flock.Proof} (hk : k < pfs.size)
    (h : decodeProof (shapeAt (tabsOf sts) (pointsOf spec c₁) k) pfs[k]! = .ok p) :
    (filePs spec sts c₁ pfs)[k]! = p := by
  rw [getElem!_pos (filePs spec sts c₁ pfs) k (by rw [filePs_size]; exact hk)]
  simp only [filePs, Array.getElem_mapIdx]
  rw [← getElem!_pos pfs k hk, h]

theorem fileCaps_size (spec : SessionSpec) (sts : Array Stmt) (c₁ : Coin) (pfs : Array ByteArray) :
    (fileCaps spec sts c₁ pfs).size = sts.size := by
  simp only [fileCaps, Array.size_map, Array.size_range]

def zAtP (resp : ℕ → Claims.Msg) (spec : SessionSpec) (sts : Array Stmt) (k : ℕ) : ℕ :=
  1 + ((List.range k).map (JsonRT.sendsP resp spec (tabsOf sts))).sum + 2

theorem sessionS_reps {H : Claims.Msg → Claims.Msg} (h64 : Claims.Out64 H) {resp : ℕ → Claims.Msg}
    {spec : SessionSpec} {sts : Array Stmt} {w : Claims.Flock.Wit} {ω : Tape} {c₁ : Coin} (hc : resp 0 = c₁.bytes)
    (hts : sts.size = spec.tables.size)
    (htab : ∀ (i : ℕ) (hi : i < sts.size), JsonRT.TableOk sts[i] ∧
      ∀ e, sts[i].extraClaims (pointsOf spec c₁) = .ok e → e.size ≤ 2048)
    {rec : String} {pfs : Array ByteArray} {bs : Array Claims.Msg}
    (hS : FlockComplete.sessionS H resp spec sts w ω = (.ok (rec, pfs), bs)) (r₀ : List ByteArray) (cs : List Coin) :
    pfs.size = 2 * sts.size ∧ ∀ k < 2 * sts.size,
      decodeProof (shapeAt (tabsOf sts) (pointsOf spec c₁) k) pfs[k]! = .ok (filePs spec sts c₁ pfs)[k]! ∧
      SchedOf (repCtx spec sts c₁ r₀ cs k) ∧
      ((filePs spec sts c₁ pfs)[k]!.zerocheck.finalC =
          dot (lagrange nodesL den6Inv (repCtx spec sts c₁ r₀ cs k).z) (filePs spec sts c₁ pfs)[k]!.zerocheck.round1c →
        HonestRep (repCtx spec sts c₁ r₀ cs k) (filePs spec sts c₁ pfs)[k]!) ∧
      (filePs spec sts c₁ pfs)[k]!.cap = (filePs spec sts c₁ pfs)[2 * (k / 2)]!.cap ∧
      (filePs spec sts c₁ pfs)[k]!.cap.size = RecDeploy.level0Nodes sts[k / 2]! ∧
      (filePs spec sts c₁ pfs)[k]!.zerocheck.finalC = dot (lagrange nodesL den6Inv
        (FlockComplete.coinOf (resp (zAtP resp spec sts k)) 0)) (filePs spec sts c₁ pfs)[k]!.zerocheck.round1c ∧
      FlockVBridge.Family.ProofShaped (shapeAt (tabsOf sts) (pointsOf spec c₁) k) (filePs spec sts c₁ pfs)[k]! := by
  have htabs : (tabsOf sts).size = sts.size := by
    unfold tabsOf
    rw [Array.size_map]
  obtain ⟨schs, cms, hsc, hsz, hk⟩ := JsonRT.sessionP_repRuns H resp spec (tabsOf sts) (tracesOf sts w) ω
    (by rw [htabs, hts]) hS
  rw [htabs] at hsc hsz hk
  replace hsz : pfs.size = 2 * sts.size := hsz
  have hone : ∀ k < 2 * sts.size,
      decodeProof (shapeAt (tabsOf sts) (pointsOf spec c₁) k) pfs[k]! = .ok (filePs spec sts c₁ pfs)[k]! ∧
      SchedOf (repCtx spec sts c₁ r₀ cs k) ∧
      ((filePs spec sts c₁ pfs)[k]!.zerocheck.finalC =
          dot (lagrange nodesL den6Inv (repCtx spec sts c₁ r₀ cs k).z) (filePs spec sts c₁ pfs)[k]!.zerocheck.round1c →
        HonestRep (repCtx spec sts c₁ r₀ cs k) (filePs spec sts c₁ pfs)[k]!) ∧
      (filePs spec sts c₁ pfs)[k]!.cap = cms[k / 2]!.cap ∧
      (filePs spec sts c₁ pfs)[k]!.cap.size = RecDeploy.level0Nodes sts[k / 2]! ∧
      (filePs spec sts c₁ pfs)[k]!.zerocheck.finalC = dot (lagrange nodesL den6Inv
        (FlockComplete.coinOf (resp (zAtP resp spec sts k)) 0)) (filePs spec sts c₁ pfs)[k]!.zerocheck.round1c ∧
      FlockVBridge.Family.ProofShaped (shapeAt (tabsOf sts) (pointsOf spec c₁) k) (filePs spec sts c₁ pfs)[k]! := by
    intro k hk'
    have hk2 : k / 2 < sts.size := by omega
    obtain ⟨e, pt, l, l', he, hls, hrun, hf⟩ := hk k hk'
    obtain ⟨hs, hcm⟩ := hsc (k / 2) hk2
    rw [hc, JsonRT.ptsOf_coin] at he
    have hsh : shapeAt (tabsOf sts) (pointsOf spec c₁) k =
        FlockComplete.shapeOf (tabsOf sts)[k / 2]! schs[k / 2]! e.size := by
      simp only [shapeAt, hs, he, Option.getD_some]
      rfl
    have hcap := JsonRT.Ret.go (JsonRT.repP_cap _ _ _ _ _ _ _ _ _ _ _) hrun
    have ht := tabsOf_get! hk2
    rw [ht] at hs hcm he hrun hf hsh
    have hs' : fast100 sts[k / 2].m = some schs[k / 2]! := hs
    have hon := honestRep_of_repP (spec := spec) (c₁ := c₁) (r₀ := r₀) (cs := cs) (k := k) h64 hk2 (htab _ hk2).1 hs'
      he ((htab _ hk2).2 e he) resp _ hcm _ _ _ hrun
    have hp := JsonRT.repShapedOk h64 (htab _ hk2).1 hs resp _ hcm ((htab _ hk2).2 e he) _ _ _ l pt l' hrun
    have hdec : decodeProof (shapeAt (tabsOf sts) (pointsOf spec c₁) k) pfs[k]! = .ok pt.1 := by
      rw [hsh, show pfs[k]! = (rec, pfs).2[k]! from rfl, hf]
      exact FlockVBridge.Family.decodeProof_encodeProof _ _ hp
    have hget := filePs_get (by rw [hsz]; exact hk') hdec
    have hfz := JsonRT.repP_finalC (p := pt.1) (t' := pt.2) resp _ _ _ _ _ _ _ _
      (by exact le_trans (by decide) (htab _ hk2).1.m22) _ hrun
    rw [hls] at hfz
    rw [hget]
    refine ⟨hdec, schedOf_repCtx hk2 (by simp [hs']) he, hon, hcap, ?_, hfz, by rw [hsh]; exact hp⟩
    have hsz' := FlockSoundness.Discharge.DecodeZL.decodeProof_sizes _ _ _
      (FlockVBridge.Family.decodeProof_encodeProof _ _ hp)
    rw [hsz'.cap]
    show 2 ^ (schs[k / 2]!.levels.getD 0 default).cap =
      2 ^ ((fast100 sts[k / 2]!.m).map fun s => (s.levels.getD 0 default).cap).getD 0
    rw [getElem!_pos sts (k / 2) hk2, hs', Option.map_some, Option.getD_some]
  refine ⟨hsz, fun k hk' => ?_⟩
  obtain ⟨h1, h2, h3, h4, h5, h6⟩ := hone k hk'
  obtain ⟨-, -, -, h4', -, -⟩ := hone (2 * (k / 2)) (by omega)
  refine ⟨h1, h2, h3, ?_, h5, h6⟩
  rw [h4, h4', show 2 * (k / 2) / 2 = k / 2 by omega]

theorem honestRun_files {H : Claims.Msg → Claims.Msg} (h64 : Claims.Out64 H) {x : Claims.Msg}
    {w : Claims.Flock.Wit} {ω : Tape} {k : ℕ} (r : Fin k → Coin) {s : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hx : setupOf false x = some (s, spec, sts)) (hts : sts.size = spec.tables.size)
    (htab : ∀ (i : ℕ) (hi : i < sts.size), JsonRT.TableOk sts[i] ∧
      ∀ e, sts[i].extraClaims (pointsOf spec (Claims.coinSeq r 0)) = .ok e → e.size ≤ 2048)
    {rec : String} {pfs : Array ByteArray} {bs : Array Claims.Msg}
    (hS : FlockComplete.sessionS H (fun j => (Claims.coinSeq r j).bytes) spec sts w ω = (.ok (rec, pfs), bs)) :
    FlockComplete.honestRun H x w ω r = (bs.toList.map (List.cons 0), Final.encode ⟨x, rec,
      filesAt (tabsOf sts) (pointsOf spec (Claims.coinSeq r 0)) (filePs spec sts (Claims.coinSeq r 0) pfs)⟩) := by
  obtain ⟨hsz, hall⟩ := sessionS_reps (c₁ := Claims.coinSeq r 0) h64 (by rfl) hts htab hS [] []
  have hf : filesAt (tabsOf sts) (pointsOf spec (Claims.coinSeq r 0)) (filePs spec sts (Claims.coinSeq r 0) pfs) =
      pfs := filesAt_eq (by rw [filePs_size]) fun k hk => by
    have hk' : k < 2 * sts.size := by simpa [filePs_size, hsz] using hk
    have h := (hall k hk').1
    rw [getElem!_pos pfs k (by omega),
      getElem!_pos (filePs spec sts (Claims.coinSeq r 0) pfs) k (by rw [filePs_size]; omega)] at h
    exact h
  rw [hf]
  exact FlockComplete.honestRun_eq H x w ω r hx hS

/-- Plain zerocheck completeness at the reader's coin, which `filesFinalC` proves where the replies are the coins'
bytes. -/
def FilesFinalC (spec : SessionSpec) (sts : Array Stmt) (c₁ : Coin) (cs : List Coin) (pfs : Array ByteArray) :
    Prop :=
  ∀ k < pfs.size, (filePs spec sts c₁ pfs)[k]!.zerocheck.finalC =
    dot (lagrange nodesL den6Inv (repCtx spec sts c₁ [] cs k).z) (filePs spec sts c₁ pfs)[k]!.zerocheck.round1c

theorem sessionRun_decides {H : Claims.Msg → Claims.Msg} (h64 : Claims.Out64 H) {rd : Claims.Msg → Option FinalP}
    {x last : Claims.Msg} {cs : List Coin} {s : Ins} {spec : SessionSpec} {sts : Array Stmt} {c₁ : Coin}
    {resp : ℕ → Claims.Msg} {w : Claims.Flock.Wit} {ω : Tape} {Ts : ℕ → F128} {fl : FinalP}
    (hx : setupOf false x = some (s, spec, sts)) (hcs : cs.head? = some c₁) (hc : resp 0 = c₁.bytes)
    (hts : sts.size = spec.tables.size)
    (htab : ∀ (i : ℕ) (hi : i < sts.size), JsonRT.TableOk sts[i] ∧
      ∀ e, sts[i].extraClaims (pointsOf spec c₁) = .ok e → e.size ≤ 2048)
    {rec : String} {pfs : Array ByteArray} {bs : Array Claims.Msg}
    (hS : FlockComplete.sessionS H resp spec sts w ω = (.ok (rec, pfs), bs))
    (hfc : FilesFinalC spec sts c₁ cs.tail.dropLast pfs)
    (hfit0 : RowsFit ((capRows (fileCaps spec sts c₁ pfs)).map nodeRow))
    (hfit : ∀ k < (filePs spec sts c₁ pfs).size,
      ∀ rs ∈ repRows (repCtx spec sts c₁ (capRows (fileCaps spec sts c₁ pfs)) cs.tail.dropLast k)
        (filePs spec sts c₁ pfs)[k]! (Ts k), RowsFit rs)
    (hrd : rd last = some fl) (hps : fl.ps = filePs spec sts c₁ pfs) (hst : fl.stmt = x) :
    ∃ d, decR rd x (honestMsgs (fileCaps spec sts c₁ pfs) ((List.range (filePs spec sts c₁ pfs).size).map fun k =>
      repRows (repCtx spec sts c₁ (capRows (fileCaps spec sts c₁ pfs)) cs.tail.dropLast k) (filePs spec sts c₁ pfs)[k]!
        (Ts k)) last) cs = some d := by
  obtain ⟨hsz, hall⟩ := sessionS_reps h64 hc hts htab hS (capRows (fileCaps spec sts c₁ pfs)) cs.tail.dropLast
  have hsh : ShapedAt spec sts c₁ (filePs spec sts c₁ pfs) := fun k hk => by
    have hk' : k < 2 * sts.size := by
      have h := hk
      rw [filePs_size, hsz] at h
      exact h
    have h := (hall k hk').2.2.2.2.2.2
    rwa [getElem!_pos (filePs spec sts c₁ pfs) k hk] at h
  have hcaps : ∀ i < (fileCaps spec sts c₁ pfs).size,
      (fileCaps spec sts c₁ pfs)[i]!.size = RecDeploy.level0Nodes sts[i]! := by
    intro i hi
    rw [fileCaps_size] at hi
    have h := (hall (2 * i) (by omega)).2.2.2.2.1
    rw [show 2 * i / 2 = i by omega] at h
    simp only [fileCaps, JsonRT.range_map_getElem! _ hi]
    exact h
  have hrep : ∀ k < (filePs spec sts c₁ pfs).size,
      SchedOf (repCtx spec sts c₁ (capRows (fileCaps spec sts c₁ pfs)) cs.tail.dropLast k) ∧
        HonestRep (repCtx spec sts c₁ (capRows (fileCaps spec sts c₁ pfs)) cs.tail.dropLast k)
          (filePs spec sts c₁ pfs)[k]! ∧
          (filePs spec sts c₁ pfs)[k]!.cap = (fileCaps spec sts c₁ pfs)[k / 2]! := by
    intro k hk
    rw [filePs_size] at hk
    have hf := hfc k hk
    rw [hsz] at hk
    obtain ⟨-, h2, h3, h4, -, -⟩ := hall k hk
    refine ⟨h2, h3 hf, ?_⟩
    simp only [fileCaps, JsonRT.range_map_getElem! _ (show k / 2 < sts.size by omega)]
    exact h4
  exact ⟨_, readsBack rd x last cs s spec sts c₁ (fileCaps spec sts c₁ pfs) (filePs spec sts c₁ pfs) Ts fl hx hcs
    (fileCaps_size ..) hcaps (by rw [filePs_size, hsz]) hrep hfit0 hfit hrd hps hst hsh⟩

end FlockDeploy
