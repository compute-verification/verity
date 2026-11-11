module

public import Proofs.Flock.Claims.ZK.CausalExe
public import Proofs.Flock.Claims.ZK.FinishFc
public import Proofs.Flock.Claims.ZK.SimLig
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
import Proofs.Flock.Claims.Complete.Replay.LigeritoRun
import Proofs.Flock.Claims.Complete.Replay.LincheckAlgebra
import Proofs.Flock.Claims.Complete.Shaped
import Proofs.Flock.Claims.Complete.Zk.Lanes
import Proofs.Flock.Claims.Complete.ZkTables.Scope
import Proofs.Flock.Claims.ZK.FinishLigEnd
import Proofs.Flock.Claims.ZK.FinishSwInner
import Proofs.Flock.Claims.ZK.FinishTau
import Proofs.Flock.Claims.ZK.L0PinCut
import Proofs.Flock.Claims.ZK.PiopFree
import Proofs.Flock.Claims.ZK.RepReplay
import Proofs.Flock.Claims.ZK.RepRet
import Proofs.Flock.SessionLoop.CutPiopLive
import Proofs.Flock.SessionLoop.FinHmkFc
import Proofs.Flock.Soundness.Discharge.ZkExec.PlaysRep
import Proofs.Flock.Soundness.Discharge.ZkExec.RunsReplay
import Proofs.Flock.Soundness.Refine.Arith

/-!
# The cut session at a rep's inner proof, on the opened coins alone

The inner proof (`innerCut`) sends four bodies, each the frames it absorbed since its last coin and the squeeze's header:
round 0 the inner proof's label and `final_c`, round 1 the triple's openings `ρ`, `σ`, round 2 `τ`'s commitment and
round 3 `Y` (`innerCutBody`). Cut at its round `i`, it stops there and sends round `i`'s body; where the rep from the
source runs its PIOP and Ligerito opening below the cut and ends that opening on a squeeze, so does the rep, at the
source's masked `final_c` (`piopCut_fc_go`) and the inner proof's `3 + 2 + 2` constraints and its triple
(`INNER_CONS`). At a live statement, where the coins below the cut show no refusal, the earlier reps return
(`Masked.repCut_ret`) and this rep's rank check, pair and, at round 3, `β` pass, so the cut session sends that body.

* `CutFree.innerCut_stop`: the inner proof cut at its round `i < 4` stops there and sends `innerCutBody … i`.
* `CutFree.repCut_inner_stop`: the rep from the source, cut at its inner proof's round `i`, stops there and sends that
  body, wherever its rank check passes, its pair's coins are independent and, at `i = 3`, its `β` is nonzero.
* `Masked.repCut_inner_stop`: that stop at a `Live` statement's table `j`'s rep `r`, answered by the opened coins, where
  the coins below the cut show no refusal.
* **`Masked.runCut_innerBody`**: the cut session stopped at rep `2j + r`'s inner round `i` sends there `innerCutBody`
  at the rep's source.
-/

@[expose] public section

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace FlockComplete.CutFree

open Flock Transcript Replay
open FlockSoundness.Refine (Frame encs)

/-- **The inner proof's body at its round `i`**, after a squeeze: round 0 its label and `final_c`, round 1 the
triple's openings, round 2 `τ`'s commitment, round 3 `Y`, each with the squeeze's header (`N` batching coins at round
1, `Q` query coins at round 3). -/
def innerCutBody (finalC : F128) (src : RepSrc) (N : ℕ) : ℕ → Claims.Msg
  | 0 => encs [.label _root_.Flock.Zk.LABEL_INNER.toUTF8, .one finalC] ++ (Frame.sq 1 1).enc
  | 1 => encs [.vec [src.rhoIn, src.sigmaIn]] ++ (Frame.sq 2 N).enc
  | 2 => encs [.bytes src.tauCom] ++ (Frame.sq 1 1).enc
  | _ => encs [.vec src.y.toList] ++ (Frame.sq 2 _root_.Flock.Zk.Q).enc

theorem push_at {L : Array Claims.Msg} (b : Claims.Msg) {k : ℕ} (hk : k = L.size) : (L.push b)[k]! = b := by
  subst hk
  exact Claims.ZKServer.Loop.getElem!_push_size b

theorem sampleVecP_cut_err (coin : ℕ → ℕ → F128) {n k : ℕ} (hk : k ≠ 0) (t : Transcript) {l : Array Claims.Msg}
    (hl : n ≤ l.size) :
    ((sampleVecP (asksCut coin n) k).run t).go l =
      (.error "past the openings", l.push (Claims.toMsg (_root_.Flock.Transcript.framed t 2 k))) := by
  rw [FlockComplete.SimInner.sampleVecCut_go coin n hk t l, if_pos hl]

/-- A squeeze after a slice absorbed from no pending bytes sends the slice's frame and its header. -/
theorem toMsg_framed_slice {T : Transcript} (vs : Array F128)
    (hT : T.pending = vs.foldl F128.pushBytes (ByteArray.empty ++ _root_.Flock.Transcript.header 3 2 vs.size))
    (kind : UInt8) (n : ℕ) :
    Claims.toMsg (_root_.Flock.Transcript.framed T kind n) = encs [.vec vs.toList] ++ (Frame.sq kind n).enc := by
  unfold Claims.toMsg
  rw [FlockSoundness.Refine.framed_enc, hT, FlockSoundness.Refine.absorbSlice_enc]
  simp [encs]

/-- A squeeze after bytes absorbed from no pending bytes sends their frame and its header. -/
theorem toMsg_framed_bytes {T : Transcript} (b : ByteArray)
    (hT : T.pending = ByteArray.empty ++ _root_.Flock.Transcript.header 5 0 b.size ++ _root_.Flock.Transcript.padded b)
    (kind : UInt8) (n : ℕ) :
    Claims.toMsg (_root_.Flock.Transcript.framed T kind n) = encs [.bytes b] ++ (Frame.sq kind n).enc := by
  unfold Claims.toMsg
  rw [FlockSoundness.Refine.framed_enc, hT, FlockSoundness.Refine.absorbBytes_enc]
  simp [encs]

set_option maxHeartbeats 2000000 in
/-- **The inner proof cut at its round `i`**: answered by `asksCut` with the cut `i < 4` bodies past its first, from no
pending bytes and with `β ≠ 0` where round 3 is reached, it stops at round `i` and sends there `innerCutBody … i`,
with the constraints and the triple's `N` batching coins. -/
theorem innerCut_stop (coin : ℕ → ℕ → F128) (lay : Zk.PadLayout) (ic : InnerCommit) (finalC : F128) (src : RepSrc)
    (cons : Array (String × Zk.Lin)) {t : Transcript} (ht : t.pending = ByteArray.empty) (l : Array Claims.Msg)
    {i : ℕ} (hi : i < 4) {N : ℕ} (hN : cons.size + 3 = N) {n : ℕ} (hn : n = l.size + i)
    (hb : i = 3 → (coin (l.size + 2) 0).isZero = false) :
    ∃ L, ((innerCut (asksCut coin n) lay ic finalC src cons).run t).go l = (.error "past the openings", L) ∧
      L[n]! = innerCutBody finalC src N i := by
  subst hn hN
  unfold innerCut
  try dsimp only
  rw [FlockComplete.Replay.bind_go (FlockComplete.Replay.liftTM_ok
      (x := _root_.Flock.Transcript.label _root_.Flock.Zk.LABEL_INNER) (t := t)
      (t' := ⟨t.rounds, t.k, FlockComplete.Replay.pLabel _root_.Flock.Zk.LABEL_INNER t.pending⟩) (a := ()) rfl l)]
  rw [FlockComplete.Replay.bind_go (FlockComplete.Replay.liftTM_ok (x := _root_.Flock.Transcript.absorbF finalC)
      (t := ⟨t.rounds, t.k, FlockComplete.Replay.pLabel _root_.Flock.Zk.LABEL_INNER t.pending⟩)
      (t' := FlockComplete.Replay.InZ.t₁ finalC t) (a := ()) rfl l)]
  obtain rfl | rfl | rfl | rfl : i = 0 ∨ i = 1 ∨ i = 2 ∨ i = 3 := by omega
  · -- round 0: `ε` is past the cut
    have hs := FlockComplete.SimInner.sampleCut_go coin (l.size + 0) (FlockComplete.Replay.InZ.t₁ finalC t) l
    rw [if_pos (show l.size + 0 ≤ l.size by omega)] at hs
    refine ⟨_, FlockComplete.bind_go_err hs, ?_⟩
    refine (push_at _ (by simp)).trans ?_
    exact Claims.ZKServer.innerBody_enc finalC ht
  · -- round 1: the batching coins are past the cut
    rw [FlockComplete.Replay.bind_go (FlockComplete.sampleP_cut_ok coin _ (by omega : l.size < l.size + 1))]
    try dsimp only
    rw [FlockComplete.Replay.bind_go
      (FlockComplete.Replay.liftTM_ok (_root_.Flock.RecAlgebra.absorbSlice_eq _ _) _)]
    refine ⟨_, FlockComplete.bind_go_err (sampleVecP_cut_err coin (by simp [Zk.triple]) _ ?hl), ?_⟩
    case hl => simp only [Array.size_push]; try omega
    refine (push_at _ (by simp)).trans ?_
    refine (toMsg_framed_slice #[src.rhoIn, src.sigmaIn] (by rfl) _ _).trans ?_
    first
      | (rw [Array.size_append]; rfl)
      | simp [innerCutBody, Zk.triple, encs]
  · -- round 2: `β` is past the cut
    rw [FlockComplete.Replay.bind_go (FlockComplete.sampleP_cut_ok coin _ (by omega : l.size < l.size + 2))]
    try dsimp only
    rw [FlockComplete.Replay.bind_go
      (FlockComplete.Replay.liftTM_ok (_root_.Flock.RecAlgebra.absorbSlice_eq _ _) _)]
    rw [FlockComplete.Replay.bind_go (Claims.ZKServer.sampleVecP_cut_ok coin ?hk _ ?hl)]
    case hk => simp [Zk.triple]
    case hl => simp only [Array.size_push]; try omega
    try dsimp only
    rw [FlockComplete.Replay.bind_go
      (FlockComplete.Replay.liftTM_ok (_root_.Flock.RecAlgebra.absorbBytes_eq _ _) _)]
    refine ⟨_, FlockComplete.bind_go_err (FlockComplete.sampleP_cut_err coin _ ?hl2), ?_⟩
    case hl2 => simp only [Array.size_push]; try omega
    refine (push_at _ (by simp)).trans ?_
    exact toMsg_framed_bytes src.tauCom (by rfl) _ _
  · -- round 3: the query coins are past the cut
    have hb3 := hb rfl
    rw [FlockComplete.Replay.bind_go (FlockComplete.sampleP_cut_ok coin _ (by omega : l.size < l.size + 3))]
    try dsimp only
    rw [FlockComplete.Replay.bind_go
      (FlockComplete.Replay.liftTM_ok (_root_.Flock.RecAlgebra.absorbSlice_eq _ _) _)]
    rw [FlockComplete.Replay.bind_go (Claims.ZKServer.sampleVecP_cut_ok coin ?hk _ ?hl)]
    case hk => simp [Zk.triple]
    case hl => simp only [Array.size_push]; try omega
    try dsimp only
    rw [FlockComplete.Replay.bind_go
      (FlockComplete.Replay.liftTM_ok (_root_.Flock.RecAlgebra.absorbBytes_eq _ _) _)]
    rw [FlockComplete.Replay.bind_go (FlockComplete.sampleP_cut_ok coin _ ?hl3)]
    case hl3 => simp only [Array.size_push]; try omega
    try dsimp only
    simp only [Array.size_push, Nat.add_assoc, Nat.reduceAdd, hb3, Bool.false_eq_true, ↓reduceIte]
    rw [FlockComplete.Replay.bind_go
      (FlockComplete.Replay.liftTM_ok (_root_.Flock.RecAlgebra.absorbSlice_eq _ _) _)]
    refine ⟨_, FlockComplete.bind_go_err (sampleVecP_cut_err coin (by simp [Zk.Q]) _ ?hl4), ?_⟩
    case hl4 => simp only [Array.size_push]; try omega
    refine (push_at _ (by simp)).trans ?_
    exact toMsg_framed_slice src.y (by rfl) _ _

set_option maxHeartbeats 4000000 in
/-- **The cut's rep at its inner proof's round `i`**: answered by `asksCut` cut at the inner proof's round `i < 4`, at
a `--zk` schedule of `m ∈ [25, 35]` and a fold defined at every point of the lincheck's shape, from any source,
level-0 commitment, pads commitment and extra claims, the rep stops there and sends `innerCutBody … i` at the source's
masked `final_c`, wherever its rank check passes, its pair's coins are independent and, at `i = 3`, its `β` is
nonzero. -/
theorem repCut_inner_stop (coin : ℕ → ℕ → F128) {resp : ℕ → Claims.Msg} (hr : ∀ k, coinOf (resp k) = coin k)
    {st : Setup} (h25 : 25 ≤ st.m) (h35 : st.m ≤ 35) (hkm : st.fold.kLog ≤ st.m) (hkk : st.kLog = st.fold.kLog)
    (hfold : ∀ (a b : F128) (lw ρ ts : Array F128), lw.size = 64 → ρ.size = st.fold.kLog - 6 →
      ts.size = st.fold.kLog - 6 → ∃ w, st.fold.folded a b lw ρ ts = .ok w ∧ w.size = 64)
    (hf : ∀ (a : F128) (lw ρ : Array F128), lw.size = 64 → ρ.size = st.fold.kLog - 6 →
      ∃ v, st.fold.fold a lw ρ = .ok v)
    {sch : Schedule} {tPad : ℕ} (hz : Zk.schedule st.m = .ok (sch, tPad)) (rep : ℕ) (dom : ByteArray) (cm0 : Commit)
    (extras : Array Claim) (ic : InnerCommit) (src : RepSrc) (prior : Array (List F128)) (words : List ℕ)
    (l : Array Claims.Msg) {i : ℕ} (hi : i < 4) {n : ℕ} (hn : n = CutEq.innerAt l.size st sch extras + i)
    (hrk : ∀ zp zc t₁ l₁ lp lc t₂ l₂,
      ((zerocheckCut (asksCut coin n) st.m st.digest (flat cm0.cap) ic.root (Zk.PadLayout.new st.m st.kLog sch.k0)
          src.masked).run (Transcript.init #[] dom)).go l = (.ok ((zp, zc), t₁), l₁) →
        ((lincheckCut (asksCut coin n) st.fold st.m (Zk.PadLayout.new st.m st.kLog sch.k0) src.masked zc).run t₁).go
          l₁ = (.ok ((lp, lc), t₂), l₂) →
        MaskRank.maskRankOk (prior ++ MaskRank.roundPoints st.m st.kLog t₂.rounds) words = true)
    (hρ : Zk.rhoIndependent
      ((⟨coin (CutEq.ligAt l.size st extras + 2 + sch.k0) 0, coin (CutEq.ligAt l.size st extras + 2 + sch.k0) 1⟩ :
          F256),
        (⟨coin (CutEq.ligAt l.size st extras + 3 + sch.k0) 0, coin (CutEq.ligAt l.size st extras + 3 + sch.k0) 1⟩ :
          F256)) = true)
    (hβ : i = 3 → (coin (CutEq.innerAt l.size st sch extras + 2) 0).isZero = false) :
    ∃ L, ((repCut (asksCut coin n) (Hashes.ofSha512 Sha512.hash) st sch rep dom cm0 extras ic src prior
      words).run (Transcript.init #[] dom)).go l = (.error "past the openings", L) ∧
      L[n]! = innerCutBody (src.masked (Zk.PadLayout.new st.m st.kLog sch.k0).fc) src
        _root_.Flock.Firewall.Session.INNER_CONS i := by
  have hm : 14 ≤ st.m := by omega
  have hIA : CutEq.innerAt l.size st sch extras = l.size + 3 + (st.m - 6) + (3 + (st.fold.kLog - 6)) +
      (2 + extras.size) + 1 + (sch.k0 + 7 + 11 * (sch.levels.size - 1)) := rfl
  have hLA : CutEq.ligAt l.size st extras = l.size + 3 + (st.m - 6) + (3 + (st.fold.kLog - 6)) +
      (2 + extras.size) + 1 := rfl
  obtain ⟨⟨zp, lp, ss, T, sws⟩, t₃, l₃, hP, hsz₃⟩ := piopCut_ret coin n hm hkm hf
    (Zk.PadLayout.new st.m st.kLog sch.k0) src (flat cm0.cap) ic.root extras prior words (Transcript.init #[] dom) l
    (by omega) hrk
  obtain ⟨zc, t₁, l₁, lc, t₂, l₂, h₁, h₂, -, h₃⟩ := piopCut_split_rank hP
  obtain ⟨lg, zl, rec, t₄, l₄, h₄, hsz₄, hood, hch, hmsg⟩ := ligeritoCut_ret coin n hr h25 h35 hz st.merkle T cm0
    (basisVals sws (st.m - 7)) (Zk.PadLayout.new st.m st.kLog sch.k0) src t₃ l₃ (by omega)
    (by rw [hsz₃]; exact hρ)
  have hR : Replay.Prefix t₂.rounds t₄.rounds :=
    (RankAt.prefix_of_grows coin 0 (RankAt.ringSwitchCut_grows coin n 0 _ _) h₃).trans
      (RankAt.prefix_of_grows coin 0 (RankAt.ligeritoCut_grows coin n 0 _ _ _ _ _ _ _ _) h₄)
  obtain ⟨⟨rp, tt⟩, hq⟩ := replay_cut_ok coin n hr hm hkk (lay := Zk.PadLayout.new st.m st.kLog sch.k0)
    (Nat.le_refl _) (by show st.fold.kLog - 6 ≤ st.kLog - 6; rw [hkk]) src.masked
    { cap := cm0.cap, params := pcsParams st.merkle sch, zerocheck := zp, lincheck := lp, ringSwitches := ss,
      ligerito := lg } ic.root dom h₁ h₂ hR
    (by
      refine hfold _ _ _ _ _ (FlockSoundness.Refine.skipWeights_size _) ?_ (Replay.pushes_size _ _)
      rw [Array.size_extract, Replay.pushes_size]
      omega)
  obtain ⟨lgc, hlgc⟩ := ligeritoConstraints_ok (lay := Zk.PadLayout.new st.m st.kLog sch.k0) (msgs := zl) hood hch hmsg
  have hPF := _root_.Flock.Firewall.Finish.piopCut_fc_go _ st _ src src.masked _ _ extras prior words
    (by exact Nat.le_refl _) (fun _ _ => rfl) hP
  have hfc : zp.finalC = src.masked (Zk.PadLayout.new st.m st.kLog sch.k0).fc := by
    have e := hP.symm.trans hPF
    injection e with e1
    injection e1 with e2
    injection e2 with e3
    injection e3 with e4
    exact congrArg ZerocheckProof.finalC e4
  have ht₄ := _root_.Flock.Firewall.Finish.ligeritoCut_end coin n st.merkle sch T cm0 (basisVals sws (st.m - 7))
    (Zk.PadLayout.new st.m st.kLog sch.k0) src t₃ l₃ _ t₄ l₄ h₄
  have hcs3 : rp.cons.size = 3 := FlockSoundness.Discharge.ZkExec.replay_cons_size _ _ _ _ hq
  obtain ⟨L, hL, hLn⟩ := innerCut_stop coin (Zk.PadLayout.new st.m st.kLog sch.k0) ic zp.finalC src
    (rp.cons ++ #[("ring switch ab", rp.abValue.addConst (Zk.claimCheck rp.claims[0]! ss[0]!)),
      ("ring switch c", rp.cValue.addConst (Zk.claimCheck rp.claims[1]! ss[1]!))] ++ lgc) ht₄ l₄ hi
    (N := _root_.Flock.Firewall.Session.INNER_CONS)
    (by
      rw [Array.size_append, Array.size_append, hcs3, FlockSoundness.Discharge.ZkExec.ligeritoConstraints_size hlgc]
      rfl)
    (n := n) (by omega)
    (fun h3 => by
      rw [show l₄.size + 2 = CutEq.innerAt l.size st sch extras + 2 by omega]
      exact hβ h3)
  refine ⟨L, ?_, by rw [hLn, hfc]⟩
  unfold repCut
  try dsimp only
  rw [bind_go hP]
  try dsimp only
  rw [bind_go h₄]
  try dsimp only
  rw [bind_go (_root_.Flock.Firewall.Finish.get_run_go _ _)]
  try dsimp only
  rw [bind_go (_root_.Flock.Firewall.Finish.liftV_run_go hq _ _)]
  try dsimp only
  rw [bind_go (_root_.Flock.Firewall.Finish.liftV_run_go hlgc _ _)]
  try dsimp only
  exact FlockComplete.bind_go_err hL

end FlockComplete.CutFree

namespace Claims.ZKMal
open Claims FlockSoundness.ZK
open FlockSoundness.Discharge
open CoinLayer

namespace Masked

set_option maxHeartbeats 1000000 in
/-- **A rep stopped at its inner proof's round `i`, on the opened coins alone**: at a `Live` statement whose coins
below table `j`'s rep `r`'s inner round `i < 4` show no refusal, the cut rep answered by the opened coins from its
first send `repBase x (2j + r)`, from any source, level-0 commitment and pads commitment, stops there and sends
`innerCutBody … i` at the source's masked `final_c`. Its rank check passes (`rank_at_unrefused`), its pair's coins are
independent and, at round 3, its `β` is nonzero, all below the cut. -/
theorem repCut_inner_stop {x : Msg} (hx : Live x) (c : Fin (refK x) → FlockDeploy.Coin) {s0}
    {spec : _root_.Flock.SessionSpec} (hsx : FlockDeploy.setupOf true x = some (s0, spec, stsOf x))
    (j : Fin (jOf x)) (r : Fin 2) {i : ℕ} (hi : i < 4)
    (hn : ¬ Refused x c (repSlot x (2 * j.val + r.val) (innerOff i)))
    (dom : ByteArray) (cm0 : FlockComplete.Commit) {extras : Array _root_.Flock.Claim}
    (hext : (FlockDeploy.tabsOf (stsOf x))[j.val]!.extra (ptsE c spec) = .ok extras)
    (ic : FlockComplete.InnerCommit) (src : FlockComplete.RepSrc) {l : Array Msg}
    (hl : l.size = repBase x (2 * j.val + r.val)) :
    ∃ L, ((FlockComplete.repCut (FlockComplete.asksCut (ligCoin c) (repSlot x (2 * j.val + r.val) (innerOff i))) hsE
        (FlockDeploy.tabsOf (stsOf x))[j.val]! (zkTab x j).sch r.val dom cm0 extras ic src
        (priorE c j.val r.val) (FlockComplete.MaskRank.masksOf (stsOf x))[j.val]!).run
        (_root_.Flock.Transcript.init #[] dom)).go l = (.error "past the openings", L) ∧
      L[repSlot x (2 * j.val + r.val) (innerOff i)]! = FlockComplete.CutFree.innerCutBody
        (src.masked (_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (stsOf x))[j.val]!.m
          (FlockDeploy.tabsOf (stsOf x))[j.val]!.kLog (zkTab x j).sch.k0).fc) src
        _root_.Flock.Firewall.Session.INNER_CONS i := by
  have hj : j.val < (stsOf x).size := lt_of_jOf j
  have hr := r.isLt
  obtain ⟨-, -, -, hall⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hsx
  have hst : (stsOf x)[j.val]! = stAt x j := getElem!_pos _ _ _
  have hz₀ : _root_.Flock.Zk.schedule (stsOf x)[j.val]!.m = .ok ((zkTab x j).sch, (zkTab x j).tPad) := by
    rw [hst]
    exact (zkTab x j).hz
  have hz₁ : _root_.Flock.Zk.schedule (stsOf x)[(2 * j.val + r.val) / 2]!.m =
      .ok ((zkTab x j).sch, (zkTab x j).tPad) := by
    rw [show (2 * j.val + r.val) / 2 = j.val by omega]
    exact hz₀
  have htab : (FlockDeploy.tabsOf (stsOf x))[j.val]! = _root_.Flock.Setup.ofCircuit (stsOf x)[j.val]! := by
    rw [getElem!_pos (FlockDeploy.tabsOf (stsOf x)) j.val (by simpa [FlockDeploy.tabsOf] using hj),
      getElem!_pos (stsOf x) j.val hj]
    simp [FlockDeploy.tabsOf]
  rw [htab] at hext ⊢
  have h25 : 25 ≤ (stsOf x)[j.val]!.m := by
    rw [getElem!_pos (stsOf x) j.val hj]
    exact (hall _ hj).1
  have h35 : (stsOf x)[j.val]!.m ≤ 35 := by
    rw [getElem!_pos (stsOf x) j.val hj]
    exact (hall _ hj).2.1
  have hT : FlockComplete.ZkTables.StmtT (stsOf x)[j.val]! := by
    rw [getElem!_pos (stsOf x) j.val hj]
    exact _root_.Flock.Readable.stmtT_of_scope hx.1 hsx hj
  obtain ⟨⟨hwf, -⟩, hslot⟩ := hT
  have hk6 := hwf.hk6
  have hkm := hwf.hk
  have hes : extras.size = spec.points * (stsOf x)[j.val]!.regions.size := by
    rw [FlockComplete.JsonRT.extraClaims_size hext]
    simp [Nat.mul_comm, ptsE]
  have hbase : l.size = FlockComplete.repStartZk spec (stsOf x) (2 * j.val + r.val) := by rw [hl, repBase_eq hsx]
  have hLig : FlockComplete.CutEq.ligAt l.size (_root_.Flock.Setup.ofCircuit (stsOf x)[j.val]!) extras =
      l.size + 3 + ((stsOf x)[j.val]!.m - 6) + (3 + ((stsOf x)[j.val]!.c.kLog - 6)) + (2 + extras.size) + 1 := rfl
  have hIA : FlockComplete.CutEq.innerAt l.size (_root_.Flock.Setup.ofCircuit (stsOf x)[j.val]!) (zkTab x j).sch
      extras = FlockComplete.CutEq.ligAt l.size (_root_.Flock.Setup.ofCircuit (stsOf x)[j.val]!) extras +
        ((zkTab x j).sch.k0 + 7 + 11 * ((zkTab x j).sch.levels.size - 1)) := rfl
  have eN : repSlot x (2 * j.val + r.val) (innerOff i) =
      FlockComplete.CutEq.innerAt l.size (_root_.Flock.Setup.ofCircuit (stsOf x)[j.val]!) (zkTab x j).sch extras +
        i := by
    rw [repSlot_eq hsx hz₁, ← hbase, show (2 * j.val + r.val) / 2 = j.val by omega, hIA, hLig]
    unfold FlockComplete.piopRoundsZk innerOff
    omega
  have eρ : rhoSlot x j.val r.val =
      FlockComplete.CutEq.ligAt l.size (_root_.Flock.Setup.ofCircuit (stsOf x)[j.val]!) extras + 2 +
        (zkTab x j).sch.k0 := by
    unfold rhoSlot
    rw [repSlot_eq hsx hz₁, ← hbase, show (2 * j.val + r.val) / 2 = j.val by omega, hLig]
    unfold FlockComplete.piopRoundsZk
    omega
  have eβ : betaSlot x j.val r.val =
      FlockComplete.CutEq.innerAt l.size (_root_.Flock.Setup.ofCircuit (stsOf x)[j.val]!) (zkTab x j).sch extras +
        2 := by
    unfold betaSlot
    rw [repSlot_eq hsx hz₁, ← hbase, show (2 * j.val + r.val) / 2 = j.val by omega, hIA, hLig]
    unfold FlockComplete.piopRoundsZk
    omega
  have hrank : rankSlot x j.val r.val = l.size + ((stsOf x)[j.val]!.m + (stsOf x)[j.val]!.c.kLog - 6) := by
    unfold rankSlot
    rw [← hl]
  have hrank0 : rankSlot x j.val 0 ≤ rankSlot x j.val r.val := by
    unfold rankSlot
    rw [repBase_eq hsx, repBase_eq hsx]
    have := FlockComplete.ZkTables.repStart_mono spec (stsOf x) (show 2 * j.val + 0 ≤ 2 * j.val + r.val by omega)
    omega
  have hρ' : _root_.Flock.Zk.rhoIndependent (rhoAt c j.val r.val) = true := by
    cases hb : _root_.Flock.Zk.rhoIndependent (rhoAt c j.val r.val)
    · exact (hn (.inr ⟨j.val, hj, r.val, hr, .inr (.inl ⟨by omega, hb⟩)⟩)).elim
    · rfl
  obtain ⟨L, hrun, hL⟩ := FlockComplete.CutFree.repCut_inner_stop (ligCoin c) (coinOf_respE c)
    (st := _root_.Flock.Setup.ofCircuit (stsOf x)[j.val]!) h25 h35 hkm rfl
    (fun a b lw ρ ts (hlw : lw.size = 64) (hρ : ρ.size = (stsOf x)[j.val]!.c.kLog - 6)
        (hts : ts.size = (stsOf x)[j.val]!.c.kLog - 6) => by
      obtain ⟨v, hv⟩ := FlockComplete.JsonRT.fold_ok _ hk6 hslot a hlw hρ
      exact ⟨_, FlockLevel3.folded_ok _ a b lw ρ ts v hv,
        _root_.Flock.RecAlgebra.folded_size (st := (stsOf x)[j.val]!) (FlockLevel3.folded_ok _ a b lw ρ ts v hv)
          (by omega)⟩)
    (fun a lw ρ hlw hρ => FlockComplete.JsonRT.fold_ok _ hk6 hslot a hlw hρ)
    hz₀ r.val dom cm0 extras ic src (priorE c j.val r.val) (FlockComplete.MaskRank.masksOf (stsOf x))[j.val]! l hi
    (n := repSlot x (2 * j.val + r.val) (innerOff i)) eN
    (fun _ _ _ _ _ _ _ _ h₁ h₂ => by
      rw [FlockComplete.MaskRank.masksOf_getElem! hj]
      exact rank_at_unrefused hx hn hj hr (by omega) (by omega) (ligCoin c) _ (fun _ _ _ => rfl)
        (_root_.Flock.Setup.ofCircuit (stsOf x)[j.val]!) rfl rfl rfl _ _ _ _
        (t₀ := _root_.Flock.Transcript.init #[] dom) rfl hl h₁ h₂)
    (by
      rw [show FlockComplete.CutEq.ligAt l.size (_root_.Flock.Setup.ofCircuit (stsOf x)[j.val]!) extras + 3 +
          (zkTab x j).sch.k0 = rhoSlot x j.val r.val + 1 by omega, ← eρ]
      exact hρ')
    (fun h3 => by
      have hβ' : (betaAt c j.val r.val).isZero = false := by
        cases hb : (betaAt c j.val r.val).isZero
        · rfl
        · exact (hn (.inr ⟨j.val, hj, r.val, hr, .inr (.inr ⟨by omega, hb⟩)⟩)).elim
      rw [← eβ]
      exact hβ')
  exact ⟨L, hrun, hL⟩

set_option maxHeartbeats 1000000 in
/-- **The cut session at rep `2j + r`'s inner round `i`, on the opened coins alone**: where the coins below that round
show no refusal, the cut session stopped there sends `innerCutBody … i` at the rep's source: the earlier reps return
(`repCut_ret`, `runCut_cutRep_of_ret`) and the rep stops there (`repCut_inner_stop`). -/
theorem runCut_innerBody {x : Msg} (hx : Live x) (c : Fin (refK x) → FlockDeploy.Coin)
    (src : ℕ → FlockComplete.TabSrc) {s0 : _}
    {spec : _root_.Flock.SessionSpec} (hsx : FlockDeploy.setupOf true x = some (s0, spec, stsOf x))
    {schs : Array (_root_.Flock.Schedule × ℕ)}
    (hs : (FlockComplete.schedsPZk (FlockDeploy.tabsOf (stsOf x)) : _root_.Flock.V _) = .ok schs)
    {cms : Array FlockComplete.Commit}
    (hcms : ∀ l, ((Array.range (FlockDeploy.tabsOf (stsOf x)).size).mapM fun i =>
      FlockComplete.commit0Cut (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash :
          _root_.Flock.Hashes FlockComplete.PS)
        (FlockDeploy.tabsOf (stsOf x))[i]!.merkle (schs[i]!.1.levels.getD 0 default) (src i)).go l = (.ok cms, l))
    (j : Fin (jOf x)) (r : Fin 2) {i : ℕ} (hi : i < 4)
    (hN : ¬ Refused x c (repSlot x (2 * j.val + r.val) (innerOff i)))
    {ic : FlockComplete.InnerCommit}
    (hic : ∀ l, (FlockComplete.padsTreeCut hsE
      (((_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (stsOf x))[j.val]!.m
        (FlockDeploy.tabsOf (stsOf x))[j.val]!.kLog (zkTab x j).sch.k0).len +
        _root_.Flock.Zk.K_PAD).nextPowerOfTwo * 2 ^ _root_.Flock.Zk.INNER_LOG_INV_RATE)
      ((src j.val).rep r.val)).go l = (.ok ic, l)) :
    (runCut x c (repSlot x (2 * j.val + r.val) (innerOff i)) spec src).2[repSlot x (2 * j.val + r.val) (innerOff i)]! =
      FlockComplete.CutFree.innerCutBody
        (((src j.val).rep r.val).masked (_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (stsOf x))[j.val]!.m
          (FlockDeploy.tabsOf (stsOf x))[j.val]!.kLog (zkTab x j).sch.k0).fc) ((src j.val).rep r.val)
        _root_.Flock.Firewall.Session.INNER_CONS i := by
  have hjs : j.val < (stsOf x).size := lt_of_jOf j
  have hr2 := r.isLt
  have hz₁ : _root_.Flock.Zk.schedule (stsOf x)[(2 * j.val + r.val) / 2]!.m =
      .ok ((zkTab x j).sch, (zkTab x j).tPad) := by
    rw [show (2 * j.val + r.val) / 2 = j.val by omega, getElem!_pos (stsOf x) j.val hjs]
    exact (zkTab x j).hz
  have hsl : ∀ e, repSlot x (2 * j.val + r.val) (innerOff e) = repBase x (2 * j.val + r.val) +
      FlockComplete.piopRoundsZk spec (stsOf x)[(2 * j.val + r.val) / 2]! + innerOff e (zkTab x j).sch := by
    intro e
    rw [repSlot_eq hsx hz₁, repBase_eq hsx]
  have hio : innerOff i (zkTab x j).sch = innerOff 0 (zkTab x j).sch + i := by
    unfold innerOff
    omega
  have hβs := betaSlot_inner hx hsx j r
  have hb := (slots_order hx hsx hjs hr2).2.2.2
  have hK : repBase x (2 * j.val + r.val + 1) ≤ refK x := by
    rw [repBase_eq hsx, refK_eq hsx, ← FlockComplete.ZkTables.repStart_last]
    exact FlockComplete.ZkTables.repStart_mono _ _ (by omega)
  have hn0 : repBase x (2 * j.val + r.val) ≤ repSlot x (2 * j.val + r.val) (innerOff i) := by
    have := hsl i
    omega
  have hnN : repSlot x (2 * j.val + r.val) (innerOff i) ≤ refK x := by
    have := hsl i
    have := hsl 0
    omega
  obtain ⟨extras, hext⟩ := extra_ok hx hsx c hjs
  obtain ⟨l, hl⟩ : ∃ l : Array Msg, l.size = repBase x (2 * j.val + r.val) :=
    ⟨Array.replicate _ [], Array.size_replicate⟩
  obtain ⟨L, hR, hL⟩ := repCut_inner_stop hx c hsx j r hi hN (spec.domain spec.tables[j.val]! r.val).toUTF8
    cms[j.val]! hext ic ((src j.val).rep r.val) hl
  refine (runCut_cutRep_of_ret hx c src hsx hs hcms j r (fun k hk => extra_ok hx hsx c (by omega))
    (fun k hk extras' hxe ic' _ l' hl' => ?_) hext hic hl hR hn0 hnN).trans hL
  have hkb : betaSlot x (k / 2) (k % 2) < repSlot x (2 * j.val + r.val) (innerOff i) := by
    have h1 := betaSlot_lt_repBase hx hsx (q := k) (by omega)
    have h2 : repBase x (k + 1) ≤ repBase x (2 * j.val + r.val) := by
      rw [repBase_eq hsx, repBase_eq hsx]
      exact FlockComplete.ZkTables.repStart_mono _ _ (by omega)
    omega
  exact repCut_ret hx c hN hsx hs (by omega) hkb _ _ hxe ic' _ hl'

end Masked

end Claims.ZKMal
