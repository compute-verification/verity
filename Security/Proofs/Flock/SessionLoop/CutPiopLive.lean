module

public import Proofs.Flock.Claims.ZK.RepRet
public import Proofs.Flock.SessionLoop.FinHmkCut
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
import Proofs.Flock.Claims.Complete.ZkTables.Scope
import Proofs.Flock.Claims.ZK.FinishSwPts
import Proofs.Flock.Claims.ZK.FrameBeta
import Proofs.Flock.Claims.ZK.FrameOuter
import Proofs.Flock.Claims.ZK.FrameRankFault
import Proofs.Flock.Claims.ZK.FrameRho
import Proofs.Flock.SessionLoop.FinPoints
import Proofs.Flock.Soundness.Refine.Arith

/-!
# The cut session in a rep's PIOP, on the opened coins alone

`runCut_cutPiop_of_ret` reads the cut session at a body of rep `2j + r`'s PIOP where every earlier rep's extra claims
return (its `hexts`) and the rep returns (its `hprev`). At a live statement the extra claims return at any points, and
each earlier rep's `β` round is before the next rep's first, so where the coins show no refusal by a round at or past rep
`2j + r`'s first, each earlier rep returns (`repCut_ret`):

* `Masked.betaSlot_lt_repBase`: rep `q`'s `β` round is below `repBase x (q + 1)`; `Masked.slots_order`: a rep's first,
  rank check's, `ρ₀`'s and `β`'s rounds and the next rep's first are in that order.
* `Masked.extra_ok`: each table's extra claims at the opened coins' points return.
* **`Masked.runCut_cutPiop_live`**: `runCut_cutPiop_of_ret` with `¬ Refused x c N`, `repBase x (2j + r) ≤ N`, for
  `hexts` and `hprev`.
* **`LeanRelease.unrefused_prefix`**: on a live history whose last coins the frame takes, no refusal shows by a round the
  history holds from a rep's first round up to its rank check's: `N` can be rep `2j + r`'s first round, and in its PIOP
  the history's length.
-/

@[expose] public section

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace Claims.ZKMal
open Claims FlockSoundness.ZK
open FlockSoundness.Discharge
open CoinLayer

namespace Masked

/-- Rep `q`'s `β` round is before rep `q + 1`'s first. -/
theorem betaSlot_lt_repBase {x : Msg} (hx : Live x) {s spec}
    (hsx : FlockDeploy.setupOf true x = some (s, spec, stsOf x)) {q : ℕ} (hq : q < 2 * (stsOf x).size) :
    betaSlot x (q / 2) (q % 2) < repBase x (q + 1) := by
  have hj : q / 2 < (stsOf x).size := by omega
  obtain ⟨y⟩ := hx.2 (q / 2) hj
  have hz : _root_.Flock.Zk.schedule (stsOf x)[q / 2]!.m = .ok (y.sch, y.tPad) := by
    rw [getElem!_pos (stsOf x) _ hj]
    exact y.hz
  have hk6 : 6 ≤ (stsOf x)[q / 2]!.c.kLog := by
    rw [getElem!_pos (stsOf x) _ hj]
    exact y.wf.1.hk6
  have hrr : FlockDeploy.repRounds true spec (stsOf x)[q / 2]! =
      FlockComplete.piopRoundsZk spec (stsOf x)[q / 2]! + (y.sch.k0 + 7 + 11 * (y.sch.levels.size - 1)) + 4 := by
    unfold FlockDeploy.repRounds FlockComplete.piopRoundsZk
    simp only [↓reduceIte, hz]
    omega
  unfold betaSlot
  rw [show 2 * (q / 2) + q % 2 = q from Nat.div_add_mod q 2, repSlot_eq hsx hz, repBase_eq hsx,
    FlockComplete.ZkTables.repStart_succ, hrr]
  omega

/-- **A rep's rounds in order**: its first, its rank check's, `ρ₀`'s, `β`'s, then the next rep's first. -/
theorem slots_order {x : Msg} (hx : Live x) {s spec}
    (hsx : FlockDeploy.setupOf true x = some (s, spec, stsOf x)) {j r : ℕ} (hj : j < (stsOf x).size) (hr : r < 2) :
    repBase x (2 * j + r) < rankSlot x j r ∧ rankSlot x j r < rhoSlot x j r ∧ rhoSlot x j r < betaSlot x j r ∧
      betaSlot x j r < repBase x (2 * j + r + 1) := by
  obtain ⟨y⟩ := hx.2 j hj
  obtain ⟨-, -, -, hall⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hsx
  have hm : 25 ≤ (stsOf x)[j].m := (hall j hj).1
  have hst : (stsOf x)[(2 * j + r) / 2]! = (stsOf x)[j] := by
    rw [show (2 * j + r) / 2 = j by omega]
    exact getElem!_pos (stsOf x) j hj
  have hz : _root_.Flock.Zk.schedule (stsOf x)[(2 * j + r) / 2]!.m = .ok (y.sch, y.tPad) := by
    rw [hst]
    exact y.hz
  have hβ := betaSlot_lt_repBase hx hsx (q := 2 * j + r) (by omega)
  rw [show (2 * j + r) / 2 = j by omega, show (2 * j + r) % 2 = r by omega] at hβ
  have eR : rankSlot x j r = repBase x (2 * j + r) + ((stsOf x)[j].m + (stsOf x)[j].c.kLog - 6) := by
    unfold rankSlot
    rw [getElem!_pos (stsOf x) j hj]
  have eρ : rhoSlot x j r =
      repBase x (2 * j + r) + FlockComplete.piopRoundsZk spec (stsOf x)[j] + (y.sch.k0 + 2) := by
    unfold rhoSlot
    rw [repSlot_eq hsx hz, ← repBase_eq hsx, hst]
  have eβ : betaSlot x j r = repBase x (2 * j + r) + FlockComplete.piopRoundsZk spec (stsOf x)[j] +
      (y.sch.k0 + 7 + 11 * (y.sch.levels.size - 1) + 2) := by
    unfold betaSlot
    rw [repSlot_eq hsx hz, ← repBase_eq hsx, hst]
  have eP : FlockComplete.piopRoundsZk spec (stsOf x)[j] =
      ((stsOf x)[j].m + (stsOf x)[j].c.kLog - 6) + (2 + spec.points * (stsOf x)[j].regions.size + 1) := rfl
  refine ⟨?_, ?_, ?_, hβ⟩ <;> omega

/-- Each table's extra claims at the opened coins' points return at a live statement: its regions carry their bytes
(`regionBytes_of_setup`). -/
theorem extra_ok {x : Msg} (hx : Live x) {s0 spec} (hsx : FlockDeploy.setupOf true x = some (s0, spec, stsOf x))
    (c : Fin (refK x) → FlockDeploy.Coin) {i : ℕ} (hi : i < (stsOf x).size) :
    ∃ extras, (FlockDeploy.tabsOf (stsOf x))[i]!.extra (ptsE c spec) = .ok extras := by
  have hi' : i < (FlockDeploy.tabsOf (stsOf x)).size := by simpa [FlockDeploy.tabsOf] using hi
  rw [getElem!_pos (FlockDeploy.tabsOf (stsOf x)) i hi']
  simp only [FlockDeploy.tabsOf, Array.getElem_map]
  exact FlockVBridge.Family.extraClaims_exists _ _ (regionBytes_of_setup hx hsx i hi)

set_option maxHeartbeats 1000000 in
/-- **The cut session in rep `2j + r`'s PIOP, on the opened coins alone**: `runCut_cutPiop_of_ret` where the coins show
no refusal by a round `N` at or past rep `2j + r`'s first. Each earlier rep's extra claims return (`extra_ok`) and its
`β` round is below `N` (`betaSlot_lt_repBase`), so it returns (`repCut_ret`). -/
theorem runCut_cutPiop_live {x : Msg} (hx : Live x) (c : Fin (refK x) → FlockDeploy.Coin)
    (src : ℕ → FlockComplete.TabSrc) {s0 : _}
    {spec : _root_.Flock.SessionSpec} (hsx : FlockDeploy.setupOf true x = some (s0, spec, stsOf x))
    {schs : Array (_root_.Flock.Schedule × ℕ)}
    (hs : (FlockComplete.schedsPZk (FlockDeploy.tabsOf (stsOf x)) : _root_.Flock.V _) = .ok schs)
    {cms : Array FlockComplete.Commit}
    (hcms : ∀ l, ((Array.range (FlockDeploy.tabsOf (stsOf x)).size).mapM fun i =>
      FlockComplete.commit0Cut (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash :
          _root_.Flock.Hashes FlockComplete.PS)
        (FlockDeploy.tabsOf (stsOf x))[i]!.merkle (schs[i]!.1.levels.getD 0 default) (src i)).go l = (.ok cms, l))
    (j : Fin (jOf x)) (r : Fin 2) {N : ℕ} (hN : ¬ Refused x c N) (hbN : repBase x (2 * j.val + r.val) ≤ N)
    {extras : Array _root_.Flock.Claim}
    (hext : (FlockDeploy.tabsOf (stsOf x))[j.val]!.extra (ptsE c spec) = .ok extras)
    {ic : FlockComplete.InnerCommit}
    (hic : ∀ l, (FlockComplete.padsTreeCut hsE
      (((_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (stsOf x))[j.val]!.m
        (FlockDeploy.tabsOf (stsOf x))[j.val]!.kLog (zkTab x j).sch.k0).len +
        _root_.Flock.Zk.K_PAD).nextPowerOfTwo * 2 ^ _root_.Flock.Zk.INNER_LOG_INV_RATE)
      ((src j.val).rep r.val)).go l = (.ok ic, l))
    {l : Array Msg} (hl : l.size = repBase x (2 * j.val + r.val))
    {v} {l₁ : Array Msg}
    (hpiop : ((FlockComplete.piopCut (FlockComplete.asksCut (ligCoin c) (refK x))
      (FlockDeploy.tabsOf (stsOf x))[j.val]!
      (_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (stsOf x))[j.val]!.m
        (FlockDeploy.tabsOf (stsOf x))[j.val]!.kLog (zkTab x j).sch.k0)
      ((src j.val).rep r.val) (FlockComplete.flat cms[j.val]!.cap) ic.root extras (priorE c j.val r.val)
      (FlockComplete.MaskRank.masksOf (stsOf x))[j.val]!).run
      (_root_.Flock.Transcript.init #[] (spec.domain spec.tables[j.val]! r.val).toUTF8)).go l =
      (v, l₁))
    {n : ℕ} (hn0 : repBase x (2 * j.val + r.val) ≤ n) (hn1 : n < l₁.size) (hnN : n ≤ refK x) :
    n < (runCut x c n spec src).2.size ∧ (runCut x c n spec src).2[n]! = l₁[n]! := by
  have hjs : j.val < (stsOf x).size := lt_of_jOf j
  have hr2 := r.isLt
  refine runCut_cutPiop_of_ret hx c src hsx hs hcms j r (fun k hk => extra_ok hx hsx c (by omega))
    (fun k hk extras' hxe ic' _ l' hl' => ?_) hext hic hl hpiop hn0 hn1 hnN
  have hkb : betaSlot x (k / 2) (k % 2) < N := by
    have h1 := betaSlot_lt_repBase hx hsx (q := k) (by omega)
    have h2 : repBase x (k + 1) ≤ repBase x (2 * j.val + r.val) := by
      rw [repBase_eq hsx, repBase_eq hsx]
      exact FlockComplete.ZkTables.repStart_mono _ _ (by omega)
    omega
  exact repCut_ret hx c hN hsx hs (by omega) hkb _ _ hxe ic' _ hl'

set_option maxHeartbeats 1000000 in
/-- **The cut session at rep `2j + r`'s PIOP round `jj`, on the opened coins alone**: where the coins show no refusal by
rep `2j + r`'s first round, the cut session stopped at the PIOP's round `jj` reached it and sent there the PIOP's body
`jj` (`piopBody`): its frames at the source's masked values, the table's digest, level 0's cap and the pads root, after
the stream's domain. The PIOP from the source sent its bodies before the cut's end (`piopCut_bodies_go`), and the session
reaches it (`runCut_cutPiop_live`). -/
theorem runCut_piopBody {x : Msg} (hx : Live x) (c : Fin (refK x) → FlockDeploy.Coin)
    (src : ℕ → FlockComplete.TabSrc) {s0 : _}
    {spec : _root_.Flock.SessionSpec} (hsx : FlockDeploy.setupOf true x = some (s0, spec, stsOf x))
    {schs : Array (_root_.Flock.Schedule × ℕ)}
    (hs : (FlockComplete.schedsPZk (FlockDeploy.tabsOf (stsOf x)) : _root_.Flock.V _) = .ok schs)
    {cms : Array FlockComplete.Commit}
    (hcms : ∀ l, ((Array.range (FlockDeploy.tabsOf (stsOf x)).size).mapM fun i =>
      FlockComplete.commit0Cut (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash :
          _root_.Flock.Hashes FlockComplete.PS)
        (FlockDeploy.tabsOf (stsOf x))[i]!.merkle (schs[i]!.1.levels.getD 0 default) (src i)).go l = (.ok cms, l))
    (j : Fin (jOf x)) (r : Fin 2) (hN : ¬ Refused x c (repBase x (2 * j.val + r.val)))
    {extras : Array _root_.Flock.Claim}
    (hext : (FlockDeploy.tabsOf (stsOf x))[j.val]!.extra (ptsE c spec) = .ok extras)
    {ic : FlockComplete.InnerCommit}
    (hic : ∀ l, (FlockComplete.padsTreeCut hsE
      (((_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (stsOf x))[j.val]!.m
        (FlockDeploy.tabsOf (stsOf x))[j.val]!.kLog (zkTab x j).sch.k0).len +
        _root_.Flock.Zk.K_PAD).nextPowerOfTwo * 2 ^ _root_.Flock.Zk.INNER_LOG_INV_RATE)
      ((src j.val).rep r.val)).go l = (.ok ic, l))
    {st : _root_.Flock.Setup} (htab : (FlockDeploy.tabsOf (stsOf x))[j.val]! = st)
    {jj : ℕ} (hjj : jj < Claims.ZKServer.piopNR st.m st.fold.kLog) :
    repBase x (2 * j.val + r.val) + jj < (runCut x c (repBase x (2 * j.val + r.val) + jj) spec src).2.size ∧
    (runCut x c (repBase x (2 * j.val + r.val) + jj) spec src).2[repBase x (2 * j.val + r.val) + jj]! =
      Claims.ZKServer.piopBody
        (_root_.Flock.Transcript.init #[] (spec.domain spec.tables[j.val]! r.val).toUTF8).pending.data.toList
        st.m st.fold.kLog st.digest (FlockComplete.flat cms[j.val]!.cap) ic.root
        (_root_.Flock.Zk.PadLayout.new st.m st.kLog (zkTab x j).sch.k0) ((src j.val).rep r.val).masked jj := by
  subst htab
  have hj : j.val < (stsOf x).size := lt_of_jOf j
  have hr2 := r.isLt
  obtain ⟨-, -, -, hall⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hsx
  have htab' : (FlockDeploy.tabsOf (stsOf x))[j.val]! = _root_.Flock.Setup.ofCircuit (stsOf x)[j.val]! := by
    rw [getElem!_pos (FlockDeploy.tabsOf (stsOf x)) j.val (by simpa [FlockDeploy.tabsOf] using hj),
      getElem!_pos (stsOf x) j.val hj]
    simp [FlockDeploy.tabsOf]
  have hmS : (FlockDeploy.tabsOf (stsOf x))[j.val]!.m = (stsOf x)[j.val]!.m := by rw [htab']; rfl
  have hkS : (FlockDeploy.tabsOf (stsOf x))[j.val]!.fold.kLog = (stsOf x)[j.val]!.c.kLog := by rw [htab']; rfl
  have hm25 : 25 ≤ (stsOf x)[j.val]!.m := by
    rw [getElem!_pos (stsOf x) j.val hj]
    exact (hall _ hj).1
  have hT : FlockComplete.ZkTables.StmtT (stsOf x)[j.val]! := by
    rw [getElem!_pos (stsOf x) j.val hj]
    exact _root_.Flock.Readable.stmtT_of_scope hx.1 hsx hj
  obtain ⟨⟨hwf, -⟩, hslot⟩ := hT
  have hk6 := hwf.hk6
  have hkm := hwf.hk
  have S := slots_order hx hsx hj hr2
  have hK : repBase x (2 * j.val + r.val + 1) ≤ refK x := by
    rw [repBase_eq hsx, refK_eq hsx, ← FlockComplete.ZkTables.repStart_last]
    exact FlockComplete.ZkTables.repStart_mono _ _ (by omega)
  have hR : rankSlot x j.val r.val = repBase x (2 * j.val + r.val) + ((stsOf x)[j.val]!.m + (stsOf x)[j.val]!.c.kLog - 6) :=
    rfl
  have hNR : Claims.ZKServer.piopNR (FlockDeploy.tabsOf (stsOf x))[j.val]!.m
      (FlockDeploy.tabsOf (stsOf x))[j.val]!.fold.kLog =
      6 + ((FlockDeploy.tabsOf (stsOf x))[j.val]!.m - 6) + ((FlockDeploy.tabsOf (stsOf x))[j.val]!.fold.kLog - 6) := rfl
  have hm14 : 14 ≤ (FlockDeploy.tabsOf (stsOf x))[j.val]!.m := by rw [hmS]; omega
  obtain ⟨l, hl⟩ : ∃ l : Array Msg, l.size = repBase x (2 * j.val + r.val) :=
    ⟨Array.replicate _ [], Array.size_replicate⟩
  rcases hP : ((FlockComplete.piopCut (FlockComplete.asksCut (ligCoin c) (refK x))
      (FlockDeploy.tabsOf (stsOf x))[j.val]!
      (_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (stsOf x))[j.val]!.m
        (FlockDeploy.tabsOf (stsOf x))[j.val]!.kLog (zkTab x j).sch.k0)
      ((src j.val).rep r.val) (FlockComplete.flat cms[j.val]!.cap) ic.root extras (priorE c j.val r.val)
      (FlockComplete.MaskRank.masksOf (stsOf x))[j.val]!).run
      (_root_.Flock.Transcript.init #[] (spec.domain spec.tables[j.val]! r.val).toUTF8)).go l with ⟨v, l₁⟩
  obtain ⟨hsz, hbody⟩ := Claims.ZKServer.piopCut_bodies_go (jj := jj) hm14 (by omega)
    (fun zp zc t₁ l₁' e => by
      obtain ⟨hz, -⟩ := FlockComplete.zerocheckCut_at (ligCoin c) (refK x) hm14 _ _ _ _ _ e
      have hρ : (zc.rho.extract 0 ((FlockDeploy.tabsOf (stsOf x))[j.val]!.fold.kLog - 6)).size =
          (FlockDeploy.tabsOf (stsOf x))[j.val]!.fold.kLog - 6 := by
        rw [Array.size_extract, hz.2.1, FlockComplete.Replay.pushes_size]
        omega
      rw [htab'] at hρ ⊢
      exact FlockComplete.JsonRT.fold_ok _ hk6 hslot _ (FlockSoundness.Refine.skipWeights_size _) hρ)
    hP hjj
  rw [hl] at hsz hbody
  obtain ⟨hlt, he⟩ := runCut_cutPiop_live hx c src hsx hs hcms j r hN le_rfl hext hic hl hP
    (n := repBase x (2 * j.val + r.val) + jj) (Nat.le_add_right _ _) (by omega) (by omega)
  exact ⟨hlt, he.trans hbody⟩

end Masked

end Claims.ZKMal

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V)

namespace LeanRelease

open Loop

/-- **The loop's frames refuse every fault shown before a rep's rank check, on a prefix**: on a live history of at most
`refK x` openings whose last coins the frame takes, no refusal shows by a round `N` the history holds from rep `q`'s
first round up to its rank check's. An outer coin or `β` shown there is answered within the history; a rank check or
`ρ` pair shown there is an earlier rep's (`slots_order`), all of whose rounds the history holds; so the frame refused
each (`outer_unrefused`, `rank_unrefused`, `rho_unrefused`, `beta_unrefused`). -/
theorem unrefused_prefix (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Flock.Wit
      ((ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Flock.Wit →
      (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand →
      ℕ → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    (wk : (ofLoop T L₀ real).toSrv.Worker) (x : Msg) (w : Flock.Wit) (hx : Masked.Live x) (hSv : Serves x)
    (ρ : (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand)
    (κ : (ofLoop T L₀ real).toSrv.Chk) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)) (hl : ops.length ≤ refK x)
    (hlive : (ofLoop T L₀ real).toSrv.live wk x w ρ κ ((m₀, ρ₀) :: ops) = true)
    (hr : (ofLoop T L₀ real).refusesCoins x ρ ((m₀, ρ₀) :: ops) = false)
    {N : ℕ} (hN : N ≤ ops.length) {q : ℕ} (hq : q < 2 * (Masked.stsOf x).size)
    (hq0 : Masked.repBase x q ≤ N) (hq1 : N < Masked.rankSlot x (q / 2) (q % 2)) :
    ¬ Masked.Refused x ((ZKMal.CoinLayer.session loopNr T).pad (K := refK x)
      (ops.map fun p => (ZKMal.CoinLayer.session loopNr T).decOf p.2)) N := by
  obtain ⟨s0, spec, hsx⟩ := Masked.setupOf_of_pos (show 0 < (Masked.stsOf x).size by omega)
  have Sq := Masked.slots_order hx hsx (j := q / 2) (r := q % 2) (by omega) (Nat.mod_lt _ (by norm_num))
  rw [Nat.div_add_mod] at Sq
  have hearly : ∀ {j r : ℕ}, j < (Masked.stsOf x).size → r < 2 → Masked.rankSlot x j r ≤ N →
      Masked.betaSlot x j r < ops.length := by
    intro j r hj hr2 hrs
    have Sp := Masked.slots_order hx hsx hj hr2
    have hpq : 2 * j + r + 1 ≤ q := by
      by_contra hc
      rcases Nat.lt_or_ge (2 * j + r) q with h | h
      · omega
      rcases Nat.eq_or_lt_of_le h with he | hlt
      · rw [he, show (2 * j + r) / 2 = j by omega, show (2 * j + r) % 2 = r by omega] at hq1
        omega
      · have hm := FlockComplete.ZkTables.repStart_mono spec (Masked.stsOf x) (show q + 1 ≤ 2 * j + r by omega)
        rw [← Masked.repBase_eq hsx, ← Masked.repBase_eq hsx] at hm
        omega
    have hm := FlockComplete.ZkTables.repStart_mono spec (Masked.stsOf x) hpq
    rw [← Masked.repBase_eq hsx, ← Masked.repBase_eq hsx] at hm
    omega
  rintro (⟨j, hj, r, hr2, ho, e, he, h1⟩ | ⟨j, hj, r, hr2, hth⟩)
  · exact outer_unrefused T L₀ real wk x w hx hSv ρ κ m₀ ρ₀ ops hl hlive hr hj hr2 (by omega) he h1
  · rcases hth with ⟨-, -, -, hrs, hbad⟩ | ⟨hρ, hd⟩ | ⟨hβ, hz⟩
    · exact rank_unrefused T L₀ real wk x w hx hSv ρ κ m₀ ρ₀ ops hl hlive hr hj hr2 (hearly hj hr2 hrs) hbad
    · have Sp := Masked.slots_order hx hsx hj hr2
      rw [rho_unrefused T L₀ real wk x w hx hSv ρ κ m₀ ρ₀ ops hl hlive hr hj hr2 (hearly hj hr2 (by omega))] at hd
      exact Bool.noConfusion hd
    · rw [beta_unrefused T L₀ real wk x w hx hSv ρ κ m₀ ρ₀ ops hl hlive hr hj hr2 (by omega)] at hz
      exact Bool.noConfusion hz

end LeanRelease

end Claims.ZKServer
