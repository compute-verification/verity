module

public import Proofs.Flock.Claims.Complete.Replay.LigeritoMerkle
public import Proofs.Flock.Claims.Complete.Replay.LigeritoGoal
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
import Proofs.Flock.Claims.Complete.Json.Session
import Proofs.Flock.Claims.Complete.Replay.LigeritoRep
import Proofs.Flock.Soundness.Refine.Seam
import Std.Tactic.BVDecide.Normalize.Prop

/-! Ligerito's goals on the session walk's two named premises, `RepWalk` and `SchemeWidths`, and the final identity
`LgFinal`. -/

@[expose] public section

namespace FlockComplete.Replay

open Flock Transcript

theorem repP_opens (resp : ℕ → Claims.Msg) (h : ByteArray → ByteArray) (st : Setup) (sch : Schedule) (tr : Trace)
    (tbl : Array F128) (cm0 : Commit) (extras : Array Claim) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64)
    (hm : 14 ≤ st.m) (hn : sch.n ≠ 0) (hL : 2 ≤ sch.levels.size)
    (hq : ∀ j < sch.levels.size, (sch.levels.getD j default).q ≠ 0) (t : Transcript) (ht : t.k = 0)
    (l : Array Claims.Msg) {full : Array F128}
    (hfold : st.fold.fold (coinOf (resp (l.size + 3 + (st.m - 6))) 0)
      (skipWeights (Zc.out st.m (Zc.coinsAt resp l.size) tr).z)
      ((Zc.out st.m (Zc.coinsAt resp l.size) tr).rho.extract 0 (st.fold.kLog - 6)) = .ok full)
    {p : Proof} {t' : Transcript} {l' : Array Claims.Msg}
    (hrun : ((repP (asksS resp) (Hashes.ofSha512 h) st sch tr tbl cm0 extras salt nonce).run t).go l =
      (.ok (p, t'), l')) :
    p.cap = cm0.cap ∧ Lg.Opens (Hashes.ofSha512 h : Hashes V) st.merkle sch cm0 p.ligerito (Lg.cfOf resp l.size)
      (Rs.tEnd tr (abcClaims st.m st.kLog (Zc.out st.m (Zc.coinsAt resp l.size) tr)
          (Lc.out st.fold st.m tr (Zc.out st.m (Zc.coinsAt resp l.size) tr)
            (Lc.coinsAt resp (l.size + 3 + (st.m - 6)) (st.fold.kLog - 6))) ++ extras)
        (Rs.coinsAt resp (l.size + 3 + (st.m - 6) + 3 + (st.fold.kLog - 6))
          (abcClaims st.m st.kLog (Zc.out st.m (Zc.coinsAt resp l.size) tr)
            (Lc.out st.fold st.m tr (Zc.out st.m (Zc.coinsAt resp l.size) tr)
              (Lc.coinsAt resp (l.size + 3 + (st.m - 6)) (st.fold.kLog - 6))) ++ extras).size)
        (Lc.tEnd st.fold st.m tr (Zc.out st.m (Zc.coinsAt resp l.size) tr) full
          (Lc.coinsAt resp (l.size + 3 + (st.m - 6)) (st.fold.kLog - 6))
          (Zc.tEnd st.m st.digest (flat cm0.cap) (Zc.coinsAt resp l.size) tr t))).k := by
  obtain ⟨l₁, hl₁, hz⟩ := zerocheckP_go resp hm st.digest (flat cm0.cap) tr t l
  obtain ⟨l₂, hl₂, hl⟩ := lincheckP_go resp st.fold st.m tr (Zc.out st.m (Zc.coinsAt resp l.size) tr) full
    (Zc.tEnd st.m st.digest (flat cm0.cap) (Zc.coinsAt resp l.size) tr t) l₁ (by rw [hl₁]; exact hfold)
  rw [hl₁] at hl₂ hl
  obtain ⟨l₃, hl₃, hrs⟩ := ringSwitchP_go resp tr (abcClaims st.m st.kLog (Zc.out st.m (Zc.coinsAt resp l.size) tr)
    (Lc.out st.fold st.m tr (Zc.out st.m (Zc.coinsAt resp l.size) tr)
      (Lc.coinsAt resp (l.size + 3 + (st.m - 6)) (st.fold.kLog - 6))) ++ extras) (by simp [abcClaims])
    (Lc.tEnd st.fold st.m tr (Zc.out st.m (Zc.coinsAt resp l.size) tr) full
      (Lc.coinsAt resp (l.size + 3 + (st.m - 6)) (st.fold.kLog - 6))
      (Zc.tEnd st.m st.digest (flat cm0.cap) (Zc.coinsAt resp l.size) tr t)) l₂
  rw [hl₂] at hl₃ hrs
  unfold repP at hrun
  rw [bind_go hz] at hrun
  dsimp only at hrun
  rw [bind_go hl] at hrun
  dsimp only at hrun
  rw [bind_go hrs, StateT.run_bind] at hrun
  obtain ⟨⟨lg, t₄⟩, l₄, h1, h2⟩ := JsonRT.go_bind_ok hrun
  rw [pure_go] at h2
  simp only [Prod.mk.injEq, Except.ok.injEq] at h2
  obtain ⟨⟨rfl, rfl⟩, rfl⟩ := h2
  exact ⟨rfl, Lg.ligeritoP_opens resp h st.merkle sch _ cm0 tbl _ salt nonce l.size hn hL hq
    (by rw [hl₃, Rs.tEnd_k, Lc.tEnd_k, Zc.tEnd_k, ht]; omega) h1⟩

theorem _root_.FlockComplete.Run.lgMerkle_of {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}
    (o : Run H rel) {i : ℕ} (hi : i < o.tabs.size) (r : ℕ)
    (hn : ∀ a b v, (o.tabs[i]!.merkle.opsWith o.hs).node a b = .ok v → v.size = o.tabs[i]!.merkle.digestLen)
    (hlf : ∀ row s v, (o.tabs[i]!.merkle.opsWith o.hs).leaf row s = .ok v → v.size = o.tabs[i]!.merkle.digestLen)
    {tbl : Array F128} {cm0 : Commit} {s0 : ℕ → ByteArray}
    (h0 : commitP o.hs o.tabs[i]!.merkle ((o.sch i).levels.getD 0 default) tbl s0 = .ok cm0)
    (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64) {l : Array Claims.Msg}
    (hl : l.size = repStart o.spec o.sts (2 * i + r)) {t' : Transcript} {l' : Array Claims.Msg}
    (hrun : ((repP (asksS o.resp) (Hashes.ofSha512 (Claims.hOf H)) o.tabs[i]! (o.sch i) (o.trace i) tbl cm0
      (o.extras i) salt nonce).run (Transcript.init #[] (o.spec.domain o.spec.tables[i]! r).toUTF8)).go l =
        (.ok (o.proof i r, t'), l')) :
    Lg.MerkleOk o.hs o.tabs[i]!.merkle (o.sch i) (o.proof i r).cap (o.proof i r).ligerito (o.lgCf i r)
      (o.rsEnd i r).k := by
  have hm : 14 ≤ o.tabs[i]!.m := by have := (o.tab_m hi).2.1; omega
  obtain ⟨hn', hL, -, hq, -⟩ := fast100_ligerito (o.sch_fast100 hi)
  have hseam := FlockSoundness.Refine.fast100_level _ _ (o.sch_fast100 hi)
  have hcp := Lg.fast100_cap (o.sch_fast100 hi)
  obtain ⟨full, hfold⟩ := repP_fold o.resp _ _ _ _ _ _ _ _ _ hm _ l hrun
  obtain ⟨hcap, hop⟩ := repP_opens o.resp (Claims.hOf H) _ _ _ tbl cm0 _ salt nonce hm hn' hL hq _ rfl l hfold hrun
  rw [hl] at hfold hop
  have hfull : o.full i r = full := congrArg val hfold
  rw [← hfull, ← hcap] at hop
  have hM := Lg.merkleOk_of_opens hn hlf (fun j hj => hcp _ (Lg.getD_mem _ hj))
    (fun j hj coins => Lg.positions_lt_d (hseam _ (Lg.getD_mem _ hj)) coins) h0 hop
  rw [hcap]
  exact hM

def RepWalk {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation} (o : Run H rel) (i r : ℕ) :
    Prop :=
  ∃ (tbl : Array F128) (cm0 : Commit) (s0 : ℕ → ByteArray) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64)
    (l : Array Claims.Msg) (t' : Transcript) (l' : Array Claims.Msg),
    commitP o.hs o.tabs[i]!.merkle ((o.sch i).levels.getD 0 default) tbl s0 = .ok cm0 ∧
    l.size = repStart o.spec o.sts (2 * i + r) ∧
    ((repP (asksS o.resp) (Hashes.ofSha512 (Claims.hOf H)) o.tabs[i]! (o.sch i) (o.trace i) tbl cm0 (o.extras i) salt
      nonce).run (Transcript.init #[] (o.spec.domain o.spec.tables[i]! r).toUTF8)).go l = (.ok (o.proof i r, t'), l') ∧
    Agrees (o.stream i r).rounds t'.rounds 0 ∧ (o.stream i r).rounds.size = t'.rounds.size

def SchemeWidths (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) : Prop :=
  ∀ o : Run H rel, ∀ i < o.tabs.size,
    (∀ a b v, (o.tabs[i]!.merkle.opsWith o.hs).node a b = .ok v → v.size = o.tabs[i]!.merkle.digestLen) ∧
    ∀ row s v, (o.tabs[i]!.merkle.opsWith o.hs).leaf row s = .ok v → v.size = o.tabs[i]!.merkle.digestLen

theorem lgAt_of_walk {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation} (o : Run H rel)
    {i : ℕ} (hi : i < o.tabs.size) (r : ℕ) (hw : RepWalk o i r) : LgAt o i r := by
  obtain ⟨tbl, cm0, -, salt, nonce, l, t', l', -, hl, hrun, hag, -⟩ := hw
  exact o.lgAt_of hi r _ tbl cm0 salt nonce hl hrun hag

theorem lgCount_of_walk {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation} (o : Run H rel)
    {i : ℕ} (hi : i < o.tabs.size) (r : ℕ) (hw : RepWalk o i r) : LgCount o i r := by
  obtain ⟨tbl, cm0, -, salt, nonce, l, t', l', -, hl, hrun, -, hsz⟩ := hw
  exact o.lgCount_of hi r _ tbl cm0 salt nonce hl hrun hsz

theorem lgMerkle_of_walk (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation)
    (hW : SchemeWidths H rel) (hw : ∀ o : Run H rel, ∀ i < o.tabs.size, ∀ r < 2, RepWalk o i r) : LgMerkle H rel := by
  intro o i hi r hr
  obtain ⟨tbl, cm0, s0, salt, nonce, l, t', l', h0, hl, hrun, -⟩ := hw o i hi r hr
  exact o.lgMerkle_of hi r (hW o i hi).1 (hW o i hi).2 h0 salt nonce hl hrun

theorem ligerito_goal_walk (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation)
    (hR1 : TraceR1CS H rel) (hzc : ∀ o : Run H rel, ∀ i < o.tabs.size, ∀ r < 2, ZcAt o i r)
    (hlc : ∀ o : Run H rel, ∀ i < o.tabs.size, ∀ r < 2, LcAt o i r) (hal : LcAlgebra H rel)
    (hrs : ∀ o : Run H rel, ∀ i < o.tabs.size, ∀ r < 2, RsAt o i r) (hra : RsAlgebra H rel) (hD : RepDecodes H rel)
    (hw : ∀ o : Run H rel, ∀ i < o.tabs.size, ∀ r < 2, RepWalk o i r) (hW : SchemeWidths H rel)
    (hfin : LgFinal H rel) : Ligerito H rel :=
  ligerito_goal H rel hR1 hzc hlc hal hrs hra hD (fun o i hi r hr => lgAt_of_walk o hi r (hw o i hi r hr))
    (lgMerkle_of_walk H rel hW hw) hfin

theorem consumed_goal_walk (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation)
    (hR1 : TraceR1CS H rel) (hzc : ∀ o : Run H rel, ∀ i < o.tabs.size, ∀ r < 2, ZcAt o i r)
    (hlc : ∀ o : Run H rel, ∀ i < o.tabs.size, ∀ r < 2, LcAt o i r) (hal : LcAlgebra H rel)
    (hrs : ∀ o : Run H rel, ∀ i < o.tabs.size, ∀ r < 2, RsAt o i r) (hra : RsAlgebra H rel) (hD : RepDecodes H rel)
    (hw : ∀ o : Run H rel, ∀ i < o.tabs.size, ∀ r < 2, RepWalk o i r) (hW : SchemeWidths H rel)
    (hfin : LgFinal H rel) : Consumed H rel :=
  consumed_goal H rel hR1 hzc hlc hal hrs hra hD (fun o i hi r hr => lgAt_of_walk o hi r (hw o i hi r hr))
    (lgMerkle_of_walk H rel hW hw) hfin (fun o i hi r hr => lgCount_of_walk o hi r (hw o i hi r hr))

end FlockComplete.Replay
