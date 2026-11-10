module

public import Proofs.Flock.Claims.Complete.Zk.Level0Run
public import Proofs.Flock.Claims.Complete.ZkTables.Table
public import Proofs.Flock.Claims.ZK.CausalExe
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
import Proofs.Flock.Claims.Complete.Replay.LincheckAlgebra
import Proofs.Flock.Claims.Complete.Shaped
import Proofs.Flock.Claims.Complete.Zk.Count
import Proofs.Flock.Claims.Complete.Zk.Lanes
import Proofs.Flock.Claims.Complete.Zk.Level0Lanes
import Proofs.Flock.Claims.Complete.Zk.Level0Premises
import Proofs.Flock.Claims.Complete.Zk.Stops
import Proofs.Flock.Claims.Complete.ZkTables.RepRuns
import Proofs.Flock.Claims.Complete.ZkTables.Scope
import Proofs.Flock.Claims.Scope.Fit
import Proofs.Flock.Claims.Scope.InScope
import Proofs.Flock.Claims.Scope.Sizes
import Proofs.Flock.Claims.ZK.FinishSwPts
import Proofs.Flock.Claims.ZK.FinishTau
import Proofs.Flock.Claims.ZK.FinishZc
import Proofs.Flock.Claims.ZK.L0PinCut
import Proofs.Flock.Claims.ZK.L0PinSwitch
import Proofs.Flock.Claims.ZK.PiopFree
import Proofs.Flock.Claims.ZK.PrefixLive
import Proofs.Flock.Claims.ZK.RepFree
import Proofs.Flock.Claims.ZK.RepReplay
import Proofs.Flock.Soundness.Refine.Arith

/-!
# A rep returns on its coins alone

The cut's rep (`repCut`), answered by `asksCut` from its first send, returns wherever the coins it samples pass its
checks: each phase returns from any source with as many sends as it has. No step reads the source's values, and no
step needs a run of the session's rep.

* `Readable.stmtT_of_scope`: each table of the masked scope is a parsed one (`StmtT`), with no witness.
* `CutFree.piopCut_ret`: the PIOP returns where its rank check passes, at `3 + (m − 6) + 3 + (k_log − 6) + 2 + |e| + 1`
  sends.
* `CutFree.ligeritoCut_ret`: the Ligerito opening returns where its pair's coins are independent, at `k₀ + 7 + 11R`
  sends, with a record of the layout's shape: `ligeritoPZk` returns at the replies (`ligeritoPZk_ret`), so at the cut's
  coins (the coupling `ligeritoPZk_agree`, whose throw at send `n + 1` its send count rules out), so the cut opening
  does (`ligeritoPZk_cut_fw`).
* `CutFree.innerCut_ret`: the inner proof returns where `β ≠ 0`, at `4` sends.
* `CutFree.repCut_ret`: the rep returns at `repSends` sends where its rank check passes, its pair's coins are
  independent and its `β` is nonzero.
* **`Masked.repCut_ret`**: at a `Live` statement, table `k / 2`'s rep `k % 2`, answered by the opened coins from its
  first send `repBase x k`, returns at `repBase x (k + 1)` where the coins below round `n`, past the rep's `β` round,
  show no refusal.
-/

@[expose] public section

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace Flock.Readable

open FlockSoundness FlockSoundness.Discharge Exec

/-- **Each table of the masked scope is a parsed one** (`StmtT`): the scope's parse is untyped, has no public input
ports and has the setup's tables. -/
theorem stmtT_of_scope {rel : Parsed → Relation} {x : Claims.Msg} {s : FlockDeploy.Ins} {spec : SessionSpec}
    {sts : Array Stmt} (hsc : Claims.Flock.Scope (FlockDeploy.deployZk rel) x)
    (hset : FlockDeploy.setupOf true x = some (s, spec, sts)) {j : ℕ} (hj : j < sts.size) :
    FlockComplete.ZkTables.StmtT sts[j] := by
  obtain ⟨p, hp, hsc'⟩ := (scope_iff (FlockDeploy.deployZk rel) rel (FlockDeploy.readOf true) rfl
    (fun _ _ h => FlockDeploy.readOf_zk h) x).1 hsc
  have hi := (FlockDeploy.setupOf_some hset).1
  unfold parseBytes FlockDeploy.readOf at hp
  rw [hi] at hp
  have ht := toRead_of_parse (r := s.read) hp
  have hps := parses_of_parse hp
  obtain ⟨hc, -⟩ := (scope_ok_iff p).1 hsc'
  obtain ⟨hRu, -⟩ := (check_ok_iff _ _ _ _ _ _ _ _).1 hc
  have hpi : p.publicInputs = none := Option.not_isSome_iff_eq_none.1 (by simp [hRu.2.2.2.2.2])
  obtain ⟨spec', hset'⟩ := setup_of_tablesOf (tablesOf_of_parses hps)
  rw [ht] at hset'
  have hpe : p.sts = sts := by
    have := hset'.symm.trans (FlockDeploy.Ins.setup_read (FlockDeploy.setupOf_some hset).2)
    simp only [Except.ok.injEq, Prod.mk.injEq] at this
    exact this.2
  have htab := tablesOf_of_parses hps
  rw [hpe] at htab
  exact FlockComplete.ZkTables.stmtT_of htab hRu.2.1 hpi hj

end Flock.Readable

namespace FlockComplete.CutFree

open Flock Transcript Replay

/-- **The cut's PIOP returns where its rank check passes**: answered by `asksCut` with room for its sends, with the
circuit's fold defined at every point of the lincheck's shape, from any source, at `3 + (m − 6) + 3 + (k_log − 6) +
2 + |e| + 1` sends. -/
theorem piopCut_ret (coin : ℕ → ℕ → F128) (n : ℕ) {st : Setup} (hm : 14 ≤ st.m) (hkm : st.fold.kLog ≤ st.m)
    (hf : ∀ (a : F128) (lw ρ : Array F128), lw.size = 64 → ρ.size = st.fold.kLog - 6 →
      ∃ v, st.fold.fold a lw ρ = .ok v)
    (lay : Zk.PadLayout) (src : RepSrc) (cap root : ByteArray) (extras : Array Claim) (prior : Array (List F128))
    (words : List ℕ) (t₀ : Transcript) (l : Array Claims.Msg)
    (hN : l.size + 3 + (st.m - 6) + (3 + (st.fold.kLog - 6)) + (2 + extras.size) + 1 ≤ n)
    (hrk : ∀ zp zc t₁ l₁ lp lc t₂ l₂,
      ((zerocheckCut (asksCut coin n) st.m st.digest cap root lay src.masked).run t₀).go l = (.ok ((zp, zc), t₁), l₁) →
        ((lincheckCut (asksCut coin n) st.fold st.m lay src.masked zc).run t₁).go l₁ = (.ok ((lp, lc), t₂), l₂) →
        MaskRank.maskRankOk (prior ++ MaskRank.roundPoints st.m st.kLog t₂.rounds) words = true) :
    ∃ res t₃ l₃, ((piopCut (asksCut coin n) st lay src cap root extras prior words).run t₀).go l = (.ok (res, t₃), l₃) ∧
      l₃.size = l.size + 3 + (st.m - 6) + (3 + (st.fold.kLog - 6)) + (2 + extras.size) + 1 := by
  obtain ⟨zc, t₁, l₁, h₁⟩ := Claims.ZKServer.zerocheckCut_proof coin n hm st.digest cap root lay src.masked t₀ l
    (by omega)
  obtain ⟨hz, hl₁⟩ := zerocheckCut_at coin n hm _ _ _ _ _ h₁
  have hρ : (zc.rho.extract 0 (st.fold.kLog - 6)).size = st.fold.kLog - 6 := by
    rw [Array.size_extract, hz.2.1, Replay.pushes_size]
    omega
  obtain ⟨lc, t₂, l₂, h₂⟩ := Claims.ZKServer.lincheckCut_proof coin n st.fold st.m lay src.masked zc t₁ l₁ (by omega)
    (hf _ _ _ (FlockSoundness.Refine.skipWeights_size _) hρ)
  obtain ⟨-, hl₂⟩ := lincheckCut_at coin n st.fold st.m lay src.masked zc h₂
  have hrk' := hrk _ _ _ _ _ _ _ _ h₁ h₂
  have hsz : (abcClaims st.m st.kLog zc lc ++ extras).size = 2 + extras.size := by
    rw [Array.size_append, abcClaims_size]
  obtain ⟨⟨ss, T, sws⟩, t₃, l₃, h₃, -, -⟩ := Claims.ZKServer.Switch.ringSwitchCut_bodies coin n src.sv
    (abcClaims st.m st.kLog zc lc ++ extras) (by rw [hsz]; omega) t₂ l₂ (by rw [hsz]; omega)
  obtain ⟨-, hl₃, -⟩ := ringSwitchCut_sws coin n src.sv _ (by rw [hsz]; omega) h₃
  rw [hsz] at hl₃
  unfold piopCut
  rw [bind_go h₁]
  try dsimp only
  rw [bind_go h₂]
  try dsimp only
  rw [bind_go (_root_.Flock.Firewall.Finish.rankCheck_pass_go l₂ hrk')]
  try dsimp only
  rw [bind_go h₃]
  try dsimp only
  exact ⟨_, t₃, l₃, pure_go _ _ _, by omega⟩

/-- **The cut's inner proof returns where `β ≠ 0`**: answered by `asksCut` with room for its `4` sends, from any
source and constraints, at `4` sends. -/
theorem innerCut_ret (coin : ℕ → ℕ → F128) (n : ℕ) (lay : Zk.PadLayout) (ic : InnerCommit) (finalC : F128)
    (src : RepSrc) (cons : Array (String × Zk.Lin)) (t : Transcript) (l : Array Claims.Msg) (hl : l.size + 4 ≤ n)
    (hb : (coin (l.size + 2) 0).isZero = false) :
    ∃ p t' l', ((innerCut (asksCut coin n) lay ic finalC src cons).run t).go l = (.ok (p, t'), l') ∧
      l'.size = l.size + 4 := by
  unfold innerCut
  try dsimp only
  rw [bind_go (liftTM_ok (_root_.Flock.RecAlgebra.label_eq _ _) _)]
  rw [bind_go (liftTM_ok (_root_.Flock.RecAlgebra.absorbF_eq _ _) _)]
  rw [bind_go (sampleP_cut_ok coin _ ?h1)]
  case h1 => omega
  try dsimp only
  rw [bind_go (liftTM_ok (_root_.Flock.RecAlgebra.absorbSlice_eq _ _) _)]
  rw [bind_go (Claims.ZKServer.sampleVecP_cut_ok coin ?h2 _ ?h3)]
  case h2 => simp [Zk.triple]
  case h3 => simp only [Array.size_push]; omega
  try dsimp only
  rw [bind_go (liftTM_ok (_root_.Flock.RecAlgebra.absorbBytes_eq _ _) _)]
  rw [bind_go (sampleP_cut_ok coin _ ?h4)]
  case h4 => simp only [Array.size_push]; omega
  try dsimp only
  simp only [Array.size_push, Nat.add_assoc, Nat.reduceAdd, hb, Bool.false_eq_true, ↓reduceIte]
  rw [bind_go (liftTM_ok (_root_.Flock.RecAlgebra.absorbSlice_eq _ _) _)]
  rw [bind_go (Claims.ZKServer.sampleVecP_cut_ok coin ?h5 _ ?h6)]
  case h5 => simp [Zk.Q]
  case h6 => simp only [Array.size_push]; omega
  try dsimp only
  exact ⟨_, _, _, pure_go _ _ _, by simp only [Array.size_push]; try omega⟩

/-- **The cut's Ligerito opening returns where its pair's coins are independent**: answered by `asksCut` with room for
its `k₀ + 7 + 11R` sends at a `--zk` schedule of `m ∈ [25, 35]`, from any source and basis, with a record of the
layout's shape. `ligeritoPZk` returns at the replies (`ligeritoPZk_ret`), so at the cut's coins, where the coupling
(`ligeritoPZk_agree`) throws only at send `n + 1`, past its send count; the cut opening returns with it
(`ligeritoPZk_cut_fw`), at its exact send count (`ligeritoCut_eqAt`). -/
theorem ligeritoCut_ret (coin : ℕ → ℕ → F128) (n : ℕ) {resp : ℕ → Claims.Msg} (hr : ∀ k, coinOf (resp k) = coin k)
    {m : ℕ} (h25 : 25 ≤ m) (h35 : m ≤ 35) {sch : Schedule} {tPad : ℕ} (hz : Zk.schedule m = .ok (sch, tPad))
    (ms : MerkleScheme) (T : F128) (cm0 : Commit) (b : Array F128) (lay : Zk.PadLayout) (src : RepSrc)
    (t : Transcript) (l : Array Claims.Msg) (hN : l.size + (sch.k0 + 7 + 11 * (sch.levels.size - 1)) ≤ n)
    (hρ : Zk.rhoIndependent ((⟨coin (l.size + 2 + sch.k0) 0, coin (l.size + 2 + sch.k0) 1⟩ : F256),
      (⟨coin (l.size + 3 + sch.k0) 0, coin (l.size + 3 + sch.k0) 1⟩ : F256)) = true) :
    ∃ lg zl rec t' l', ((ligeritoCut (asksCut coin n) (Hashes.ofSha512 Sha512.hash) ms sch T cm0 b lay src).run t).go l =
        (.ok ((lg, zl, rec), t'), l') ∧ l'.size = l.size + (sch.k0 + 7 + 11 * (sch.levels.size - 1)) ∧
      rec.ood.size = 1 ∧ rec.challenges.size = sch.k0 ∧ sch.k0 ≤ lg.messages.size := by
  obtain ⟨hk0, hR⟩ : 1 ≤ sch.k0 ∧ 2 ≤ sch.levels.size := zkSchedule_shape hz
  have hn := L0.zkSchedule_n hz
  have hq := Replay.zkSchedule_q hz
  obtain ⟨hn', hq'⟩ : sch.n + 1 ≤ 65536 ∧ ∀ j, (sch.levels.getD j default).q + 1 ≤ 65536 :=
    JsonRT.zkSchedule_small h25 h35 hz
  obtain ⟨⟨⟨P, zl, rc⟩, t'⟩, l', hret⟩ := L0.ligeritoPZk_ret resp Sha512.hash ms sch T cm0 #[] #[] lay #[] 0 tPad
    (fun _ _ => 0) (fun _ _ => 0) (fun _ _ => ByteArray.empty) (fun _ _ => 0) hn hk0 t l
    (by simp only [coinKOf, hr]; exact hρ)
  have hle := sends_le (ligeritoPZk_sends resp Sha512.hash ms sch T cm0 #[] #[] lay #[] 0 tPad _ _ _ _) hret
  have hLF := L0.ligeritoPZk_lanes resp _ ms sch T cm0 #[] #[] lay #[] 0 tPad _ _ _ _ hn hk0 hret
  obtain ⟨-, hC⟩ : PrefixAgree.Agree n _ _ := _root_.Flock.With.Sim.run _
    (PrefixAgree.ligeritoPZk_agree resp coin (n := n) (fun k _ i _ => congrFun (hr k) i) Sha512.hash ms sch T cm0
      #[] #[] lay #[] 0 tPad (fun _ _ => 0) (fun _ _ => 0) (fun _ _ => ByteArray.empty) (fun _ _ => 0) hn' hq') t
  have hpzk : ((ligeritoPZk (asksCut coin n) (Hashes.ofSha512 Sha512.hash) ms sch T cm0 #[] #[] lay #[] 0 tPad
      (fun _ _ => 0) (fun _ _ => 0) (fun _ _ => ByteArray.empty) fun _ _ => 0).run t).go l =
        (.ok ((P, zl, rc), t'), l') := by
    have hc := hC l (by omega)
    rw [hret] at hc
    rcases hc with ⟨he, -⟩ | ⟨-, hs, bs, hbs⟩
    · exact he.symm
    · have := congrArg Array.size hbs
      simp only [Array.size_append] at this
      omega
  obtain ⟨⟨lg, zl', rec⟩, t₄, l₄, hcut, -, hm1, hc1, ho1⟩ := ligeritoPZk_cut_fw coin n Sha512.hash Sha512.hash ms ms
    sch T T cm0 cm0 #[] #[] b lay lay #[] 0 tPad (fun _ _ => 0) (fun _ _ => 0) (fun _ _ => ByteArray.empty) src t t l l
    rfl _ _ _ hpzk
  try dsimp only at hm1
  try dsimp only at hc1
  try dsimp only at ho1
  have hsz := ((CutEq.ligeritoCut_eqAt coin n ms sch T cm0 b lay src src hn hk0 hR hq t (l := l) rfl (by omega)
    (fun _ => rfl) (fun _ _ _ _ _ => rfl) (fun _ _ _ _ _ => rfl) (fun _ => ⟨rfl, rfl, rfl⟩)).2 _ _ hcut).1
  refine ⟨lg, zl', rec, t₄, l₄, hcut, hsz, ?_, ?_, ?_⟩
  · rw [← ho1, hLF.record]
    try rfl
  · rw [← hc1, hLF.record]
    simp only [ZkTables.Level0.laneCh, List.size_toArray, List.length_map, List.length_range]
  · have := hLF.msgs_size
    omega

set_option maxHeartbeats 4000000 in
/-- **The cut's rep returns on its coins alone**: answered by `asksCut` with room for its `repSends` sends, at a `--zk`
schedule of `m ∈ [25, 35]` and a fold defined at every point of the lincheck's shape, from any source, level-0
commitment, pads commitment and extra claims, the rep returns at `repSends` sends wherever its rank check passes, its
pair's coins are independent and its `β` is nonzero. -/
theorem repCut_ret (coin : ℕ → ℕ → F128) (n : ℕ) {resp : ℕ → Claims.Msg} (hr : ∀ k, coinOf (resp k) = coin k)
    {st : Setup} (h25 : 25 ≤ st.m) (h35 : st.m ≤ 35) (hkm : st.fold.kLog ≤ st.m) (hkk : st.kLog = st.fold.kLog)
    (hfold : ∀ (a b : F128) (lw ρ ts : Array F128), lw.size = 64 → ρ.size = st.fold.kLog - 6 →
      ts.size = st.fold.kLog - 6 → ∃ w, st.fold.folded a b lw ρ ts = .ok w ∧ w.size = 64)
    (hf : ∀ (a : F128) (lw ρ : Array F128), lw.size = 64 → ρ.size = st.fold.kLog - 6 →
      ∃ v, st.fold.fold a lw ρ = .ok v)
    {sch : Schedule} {tPad : ℕ} (hz : Zk.schedule st.m = .ok (sch, tPad)) (rep : ℕ) (dom : ByteArray) (cm0 : Commit)
    (extras : Array Claim) (ic : InnerCommit) (src : RepSrc) (prior : Array (List F128)) (words : List ℕ)
    (l : Array Claims.Msg) (hN : l.size + CutEq.repSends st sch extras ≤ n)
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
    (hβ : (coin (CutEq.innerAt l.size st sch extras + 2) 0).isZero = false) :
    ∃ p tr l', ((repCut (asksCut coin n) (Hashes.ofSha512 Sha512.hash) st sch rep dom cm0 extras ic src prior
      words).run (Transcript.init #[] dom)).go l = (.ok (p, tr), l') ∧ l'.size = l.size + CutEq.repSends st sch extras := by
  have hm : 14 ≤ st.m := by omega
  have hN' := hN
  simp only [CutEq.repSends, CutEq.innerAt, CutEq.ligAt] at hN'
  obtain ⟨⟨zp, lp, ss, T, sws⟩, t₃, l₃, hP, hsz₃⟩ := piopCut_ret coin n hm hkm hf
    (Zk.PadLayout.new st.m st.kLog sch.k0) src (flat cm0.cap) ic.root extras prior words (Transcript.init #[] dom) l
    (by omega) hrk
  obtain ⟨zc, t₁, l₁, lc, t₂, l₂, h₁, h₂, -, h₃⟩ := piopCut_split_rank hP
  obtain ⟨lg, zl, rec, t₄, l₄, h₄, hsz₄, hood, hch, hmsg⟩ := ligeritoCut_ret coin n hr h25 h35 hz st.merkle T cm0
    (basisVals sws (st.m - 7)) (Zk.PadLayout.new st.m st.kLog sch.k0) src t₃ l₃ (by omega) (by rw [hsz₃]; exact hρ)
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
  obtain ⟨inner, t₅, l₅, h₅, hsz₅⟩ := innerCut_ret coin n (Zk.PadLayout.new st.m st.kLog sch.k0) ic zp.finalC src
    (rp.cons ++ #[("ring switch ab", rp.abValue.addConst (Zk.claimCheck rp.claims[0]! ss[0]!)),
      ("ring switch c", rp.cValue.addConst (Zk.claimCheck rp.claims[1]! ss[1]!))] ++ lgc) t₄ l₄ (by omega)
    (by rw [hsz₄, hsz₃]; exact hβ)
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
  rw [bind_go h₅]
  exact ⟨_, _, _, pure_go _ _ _, by simp only [CutEq.repSends, CutEq.innerAt, CutEq.ligAt]; omega⟩

end FlockComplete.CutFree

namespace Claims.ZKMal

open Claims FlockSoundness.ZK
open FlockSoundness.Discharge
open CoinLayer

namespace Masked

set_option maxHeartbeats 1000000 in
/-- **A rep returns on the opened coins alone** (`runCut_cutPiop_of_ret`'s `hprev`): at a `Live` statement whose coins
below round `n`, past table `k / 2`'s rep `k % 2`'s `β` round, show no refusal, the cut rep answered by the opened coins
from its first send `repBase x k`, from any source, level-0 commitment and pads commitment, returns at
`repBase x (k + 1)`. Its rank check passes (`rank_at_unrefused`), its pair's coins are independent and its `β` is
nonzero, all below round `n`. -/
theorem repCut_ret {x : Msg} (hx : Live x) (c : Fin (refK x) → FlockDeploy.Coin) {n : ℕ}
    (hn : ¬ Refused x c n) {s0} {spec : _root_.Flock.SessionSpec}
    (hsx : FlockDeploy.setupOf true x = some (s0, spec, stsOf x)) {schs : Array (_root_.Flock.Schedule × ℕ)}
    (hs : (FlockComplete.schedsPZk (FlockDeploy.tabsOf (stsOf x)) : _root_.Flock.V _) = .ok schs)
    {k : ℕ} (hk : k < 2 * (stsOf x).size) (hkn : betaSlot x (k / 2) (k % 2) < n)
    (dom : ByteArray) (cm0 : FlockComplete.Commit) {extras : Array _root_.Flock.Claim}
    (hext : (FlockDeploy.tabsOf (stsOf x))[k / 2]!.extra (ptsE c spec) = .ok extras)
    (ic : FlockComplete.InnerCommit) (src : FlockComplete.RepSrc) {l : Array Msg} (hl : l.size = repBase x k) :
    ∃ p tr l', ((FlockComplete.repCut (FlockComplete.asksCut (ligCoin c) (refK x)) hsE
        (FlockDeploy.tabsOf (stsOf x))[k / 2]! schs[k / 2]!.1 (k % 2) dom cm0 extras ic src
        (priorE c (k / 2) (k % 2)) (FlockComplete.MaskRank.masksOf (stsOf x))[k / 2]!).run
        (_root_.Flock.Transcript.init #[] dom)).go l = (.ok (p, tr), l') ∧ l'.size = repBase x (k + 1) := by
  have hkk : 2 * (k / 2) + k % 2 = k := Nat.div_add_mod k 2
  have hj : k / 2 < (stsOf x).size := by omega
  have hr : k % 2 < 2 := Nat.mod_lt _ (by norm_num)
  have hJ : jOf x = (stsOf x).size := by
    unfold jOf
    exact ite_eq_left_of_eq_true _ _ (eq_true hx)
  have hi : k / 2 < jOf x := by omega
  obtain ⟨-, -, -, hall⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hsx
  obtain ⟨schs', hs', hschs⟩ := schedsPZk_live hx
  rw [hs] at hs'
  cases hs'
  have hst : (stsOf x)[k / 2]! = stAt x ⟨k / 2, hi⟩ := getElem!_pos _ _ _
  have hz₀ : _root_.Flock.Zk.schedule (stsOf x)[k / 2]!.m = .ok schs[k / 2]! := by
    rw [hschs ⟨k / 2, hi⟩, hst]
    exact (zkTab x ⟨k / 2, hi⟩).hz
  rcases hsp : schs[k / 2]! with ⟨sch, tPad⟩
  rw [hsp] at hz₀
  have htab : (FlockDeploy.tabsOf (stsOf x))[k / 2]! = _root_.Flock.Setup.ofCircuit (stsOf x)[k / 2]! := by
    rw [getElem!_pos (FlockDeploy.tabsOf (stsOf x)) (k / 2) (by simpa [FlockDeploy.tabsOf] using hj),
      getElem!_pos (stsOf x) (k / 2) hj]
    simp [FlockDeploy.tabsOf]
  rw [htab] at hext ⊢
  have h25 : 25 ≤ (stsOf x)[k / 2]!.m := by
    rw [getElem!_pos (stsOf x) (k / 2) hj]
    exact (hall _ hj).1
  have h35 : (stsOf x)[k / 2]!.m ≤ 35 := by
    rw [getElem!_pos (stsOf x) (k / 2) hj]
    exact (hall _ hj).2.1
  have hT : FlockComplete.ZkTables.StmtT (stsOf x)[k / 2]! := by
    rw [getElem!_pos (stsOf x) (k / 2) hj]
    exact _root_.Flock.Readable.stmtT_of_scope hx.1 hsx hj
  obtain ⟨⟨hwf, -⟩, hslot⟩ := hT
  have hk6 := hwf.hk6
  have hkm := hwf.hk
  have hes : extras.size = spec.points * (stsOf x)[k / 2]!.regions.size := by
    rw [FlockComplete.JsonRT.extraClaims_size hext]
    simp [Nat.mul_comm, ptsE]
  have e6 : (_root_.Flock.Setup.ofCircuit (stsOf x)[k / 2]!).fold.kLog = (stsOf x)[k / 2]!.c.kLog := rfl
  have e7 : (_root_.Flock.Setup.ofCircuit (stsOf x)[k / 2]!).m = (stsOf x)[k / 2]!.m := rfl
  have hbase : l.size = FlockComplete.repStartZk spec (stsOf x) k := by rw [hl, repBase_eq hsx]
  have hLig : FlockComplete.CutEq.ligAt l.size (_root_.Flock.Setup.ofCircuit (stsOf x)[k / 2]!) extras =
      l.size + 3 + ((stsOf x)[k / 2]!.m - 6) + (3 + ((stsOf x)[k / 2]!.c.kLog - 6)) + (2 + extras.size) + 1 := rfl
  have hIL : FlockComplete.CutEq.innerAt l.size (_root_.Flock.Setup.ofCircuit (stsOf x)[k / 2]!) sch extras =
      FlockComplete.CutEq.ligAt l.size (_root_.Flock.Setup.ofCircuit (stsOf x)[k / 2]!) extras +
        (sch.k0 + 7 + 11 * (sch.levels.size - 1)) := rfl
  have eρ : rhoSlot x (k / 2) (k % 2) =
      FlockComplete.CutEq.ligAt l.size (_root_.Flock.Setup.ofCircuit (stsOf x)[k / 2]!) extras + 2 + sch.k0 := by
    unfold rhoSlot
    rw [hkk, repSlot_eq hsx hz₀, ← hbase, hLig]
    unfold FlockComplete.piopRoundsZk
    omega
  have eβ : betaSlot x (k / 2) (k % 2) =
      FlockComplete.CutEq.innerAt l.size (_root_.Flock.Setup.ofCircuit (stsOf x)[k / 2]!) sch extras + 2 := by
    unfold betaSlot
    rw [hkk, repSlot_eq hsx hz₀, ← hbase, hIL, hLig]
    unfold FlockComplete.piopRoundsZk
    omega
  have hrank : rankSlot x (k / 2) (k % 2) = l.size + ((stsOf x)[k / 2]!.m + (stsOf x)[k / 2]!.c.kLog - 6) := by
    unfold rankSlot
    rw [hkk, ← hl]
  have hrank0 : rankSlot x (k / 2) 0 ≤ rankSlot x (k / 2) (k % 2) := by
    unfold rankSlot
    rw [repBase_eq hsx, repBase_eq hsx]
    have := FlockComplete.ZkTables.repStart_mono spec (stsOf x)
      (show 2 * (k / 2) + 0 ≤ 2 * (k / 2) + k % 2 by omega)
    omega
  have hρ' : _root_.Flock.Zk.rhoIndependent (rhoAt c (k / 2) (k % 2)) = true := by
    cases hb : _root_.Flock.Zk.rhoIndependent (rhoAt c (k / 2) (k % 2))
    · exact (hn (.inr ⟨k / 2, hj, k % 2, hr, .inr (.inl ⟨by omega, hb⟩)⟩)).elim
    · rfl
  have hβ' : (betaAt c (k / 2) (k % 2)).isZero = false := by
    cases hb : (betaAt c (k / 2) (k % 2)).isZero
    · rfl
    · exact (hn (.inr ⟨k / 2, hj, k % 2, hr, .inr (.inr ⟨hkn, hb⟩)⟩)).elim
  have hroom : FlockComplete.repStartZk spec (stsOf x) (k + 1) ≤ refK x := by
    rw [refK_eq hsx, ← FlockComplete.ZkTables.repStart_last]
    exact FlockComplete.ZkTables.repStart_mono _ _ (by omega)
  have hnext : FlockComplete.repStartZk spec (stsOf x) (k + 1) =
      l.size + FlockComplete.CutEq.repSends (_root_.Flock.Setup.ofCircuit (stsOf x)[k / 2]!) sch extras := by
    rw [FlockComplete.ZkTables.repStart_succ, ← FlockComplete.ZkTables.repRoundsT_eq hz₀ hes, ← hbase]
    simp only [FlockComplete.ZkTables.repRoundsT, FlockComplete.CutEq.repSends, FlockComplete.CutEq.innerAt,
      FlockComplete.CutEq.ligAt, e6, e7]
    omega
  have hlk : l.size = repBase x (2 * (k / 2) + k % 2) := by rw [hkk]; exact hl
  obtain ⟨p, tr, l', hrun, hsz⟩ := FlockComplete.CutFree.repCut_ret (ligCoin c) (refK x) (coinOf_respE c)
    (st := _root_.Flock.Setup.ofCircuit (stsOf x)[k / 2]!) h25 h35 hkm rfl
    (fun a b lw ρ ts (hlw : lw.size = 64) (hρ : ρ.size = (stsOf x)[k / 2]!.c.kLog - 6)
        (hts : ts.size = (stsOf x)[k / 2]!.c.kLog - 6) => by
      obtain ⟨v, hv⟩ := FlockComplete.JsonRT.fold_ok _ hk6 hslot a hlw hρ
      exact ⟨_, FlockLevel3.folded_ok _ a b lw ρ ts v hv,
        _root_.Flock.RecAlgebra.folded_size (st := (stsOf x)[k / 2]!) (FlockLevel3.folded_ok _ a b lw ρ ts v hv)
          (by omega)⟩)
    (fun a lw ρ hlw hρ => FlockComplete.JsonRT.fold_ok _ hk6 hslot a hlw hρ)
    hz₀ (k % 2) dom cm0 extras ic src (priorE c (k / 2) (k % 2)) (FlockComplete.MaskRank.masksOf (stsOf x))[k / 2]! l
    (by rw [← hnext]; exact hroom)
    (fun _ _ _ _ _ _ _ _ h₁ h₂ => by
      rw [FlockComplete.MaskRank.masksOf_getElem! hj]
      exact rank_at_unrefused hx hn hj hr (by omega) (by omega) (ligCoin c) (refK x) (fun _ _ _ => rfl)
        (_root_.Flock.Setup.ofCircuit (stsOf x)[k / 2]!) rfl rfl rfl _ _ _ _
        (t₀ := _root_.Flock.Transcript.init #[] dom) rfl hlk h₁ h₂)
    (by
      rw [show FlockComplete.CutEq.ligAt l.size (_root_.Flock.Setup.ofCircuit (stsOf x)[k / 2]!) extras + 3 + sch.k0 =
        rhoSlot x (k / 2) (k % 2) + 1 by omega, ← eρ]
      exact hρ')
    (by rw [← eβ]; exact hβ')
  refine ⟨p, tr, l', hrun, ?_⟩
  rw [hsz, repBase_eq hsx, hnext]

end Masked

end Claims.ZKMal
