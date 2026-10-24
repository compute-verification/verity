import Proofs.Flock.SessionLoop.FinRepCm
import Proofs.Flock.SessionLoop.RepSrc
import Proofs.Flock.Claims.ZK.SimCommit
import Proofs.Flock.Claims.ZK.FinishInner
import Proofs.Flock.Claims.ZK.FinishSwInner
import Proofs.Flock.Claims.ZK.Blind

/-!
# The pads commitment from the views

At (c.3)'s context, rep stream `k`'s pads commitment from the views' leaves (`Loop.padsView`) has the size, root and tree
of the firewall's (`Finish.innerCommitP` of the request's pads under its leaf salts), wherever table `k / 2`'s rows hold
at its base:

* **`Hm96.viewR_leavesP`**: a real view's pads leaves read the randomness alone, whatever the witness, so where the rows
  hold the switched view's pads leaves are the honest view's (`Coined.swExe_leavesP`, `Hm96.swViewR_eq`) and the views'
  source has the honest views' pads leaves (`Loop.finSrc_padLeaf`, `Masked.tabSrcView_padLeaf`);
* the honest views' pads tree is the masked prover's (`Masked.padsTreeCut_srcV`, `Masked.simCommit`) on the tape's pads
  and leaf salts, which are the request's (`Masked.pads_frame_eq`, `Loop.RepSrc.innerSalt`), and the pads tree reads
  its leaf salts only as `fitBytes 192` below its leaf count (`innerCommitP_salt_congr`).

**`Loop.padsView_ic`** states it.
-/

namespace FlockSoundness.ZK.Table

theorem toBase_leavesP {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] (T : Table F K W Salt Seed D Hi VL)
    (v : T.onVals.View) : (T.toBase v).leavesP = v.leavesP := rfl

/-- **A real view's pads leaves read the randomness alone**, whatever the witness: the opened ones are leaves of the
pads columns, the others hm96 leaves of the pads columns' digests (`Hm96.dig` at `.inr`). -/
theorem Hm96.viewR_leavesP {F K W Salt B D Hi VL C : Type} [Field F] [Field K] [AddCommGroup B] [CharP F 2]
    {T : Table F K W Salt (B × Salt) D Hi VL} (M : T.Hm96 C) (p q : W) (ω : T.Rand) (ys : T.Hid → Salt) :
    (M.viewR p ω ys).leavesP = (M.viewR q ω ys).leavesP := by
  unfold FlockSoundness.ZK.Table.Hm96.viewR FlockSoundness.ZK.Table.viewWith FlockSoundness.ZK.Table.view
  first
  | rfl
  | (funext r
     refine Prod.ext rfl ?_
     funext k
     rfl)

end FlockSoundness.ZK.Table

/-- **Where its rows hold at its base, a switched view's pads leaves are the honest view's**: it is the view at the
workers' values (`Hm96.swViewR_eq`), and the pads leaves read the randomness alone (`Hm96.viewR_leavesP`). -/
theorem FlockSoundness.Discharge.ZkView.Coined.swExe_leavesP {K W Hi VL : Type} [Field K]
    {z : FlockSoundness.Discharge.ZkView.Sizes} (T : FlockSoundness.Discharge.ZkView.Coined K W Hi VL z) (w : W)
    (v : T.tab.Vals) (ω : T.tab.Rand) (ys : T.tab.Hid → FlockSoundness.Discharge.ZkView.Salt)
    (hr : T.tab.onVals.RowsAt (w, v) ω.1 ω.2.1.2.1) :
    (T.swExe w v ω ys).leavesP = (T.viewExe w ω ys).leavesP := by
  have e1 : T.swExe w v ω ys = T.tab.toBase (T.hm96.onVals.viewR (w, v) ω ys) :=
    congrArg T.tab.toBase
      ((FlockSoundness.ZK.Table.Hm96.swViewR_eq T.hm96.onVals (w, v) (w, T.tab.vals) ω ys).trans (ite_eq_left hr))
  have e2 : T.viewExe w ω ys = T.tab.toBase (T.hm96.onVals.viewR (w, T.tab.vals) ω ys) :=
    ((T.hm96.toBase_viewR w ω ys).trans (FlockSoundness.Discharge.ZkView.Coined.viewExe_eq T w ω ys).symm).symm
  rw [e1, e2, FlockSoundness.ZK.Table.toBase_leavesP, FlockSoundness.ZK.Table.toBase_leavesP]
  exact FlockSoundness.ZK.Table.Hm96.viewR_leavesP T.hm96.onVals (w, v) (w, T.tab.vals) ω ys

namespace FlockComplete

variable {M : Type → Type} [Monad M] [MonadExceptOf String M] [MonadLiftT _root_.Flock.V M]

/-- **The pads commitment reads its leaf salts fitted to 192 bytes, below its leaf count.** -/
theorem innerCommitP_salt_congr (hs : _root_.Flock.Hashes M) (p : PadsZk) {salt salt' : ℕ → ByteArray}
    (h : ∀ j < (p.h ++ p.padH).size.nextPowerOfTwo * 2 ^ _root_.Flock.Zk.INNER_LOG_INV_RATE,
      fitBytes 192 (salt j) = fitBytes 192 (salt' j)) :
    innerCommitP hs p salt = innerCommitP hs p salt' := by
  have e : (Array.range ((p.h ++ p.padH).size.nextPowerOfTwo * 2 ^ _root_.Flock.Zk.INNER_LOG_INV_RATE)).map
      (fun j => fitBytes 192 (salt j)) =
      (Array.range ((p.h ++ p.padH).size.nextPowerOfTwo * 2 ^ _root_.Flock.Zk.INNER_LOG_INV_RATE)).map
      (fun j => fitBytes 192 (salt' j)) :=
    Array.map_congr_left fun j hj => h j (Array.mem_range.1 hj)
  first
  | (unfold innerCommitP; dsimp only; rw [e])
  | simp only [innerCommitP, e]

end FlockComplete

namespace Claims.ZKMal.Masked

open FlockSoundness.Discharge

/-- **A source reads its view's pads leaves**: views with the same pads leaves at table `i` give sources with the same
pads leaves at each of its reps. -/
theorem tabSrcView_padLeaf (P : Composed.LigProver) (x : Msg) (cz : ∀ j, Composed.CoinsZK (tabC P x j).Rest)
    (v v' : ∀ j, ((tabC P x j).T (cz j)).tab.View) {i r : ℕ} (hi : i < jOf x) (hr : r < 2)
    (h : (v ⟨i, hi⟩).leavesP = (v' ⟨i, hi⟩).leavesP) :
    ((tabSrcView P x cz v i).rep r).padLeaf = ((tabSrcView P x cz v' i).rep r).padLeaf := by
  have e : ∀ u : ∀ j, ((tabC P x j).T (cz j)).tab.View, ((tabSrcView P x cz u i).rep r).padLeaf =
      padLeafCut P x cz (fun j => viewCut (u j)) ⟨i, hi⟩ ⟨r, hr⟩ := fun u => by
    simp only [tabSrcView, tabSrcCut, repSrcCut, hi, hr, ↓reduceDIte]
  rw [e v, e v']
  unfold padLeafCut
  first
  | (simp only [viewCut, h])
  | (dsimp only [viewCut]; rw [h])

end Claims.ZKMal.Masked

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128 Stmt SessionSpec)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- `ofLoop`'s session types, as in `FrameZK`

namespace Loop

open FlockSoundness.Discharge in
/-- **Where table `i`'s rows hold at its base, the views' pads leaves are the honest views'** (`Coined.swExe_leavesP`,
`Masked.tabSrcView_padLeaf`). -/
theorem finSrc_padLeaf (vs : finS.SwVals) (x : Msg) (w : Flock.Wit) (ν : finS.Nonce) (t : finS.Tape)
    (ops : List (Msg × Msg)) {i r : ℕ} (hi : i < Masked.jOf x) (hr : r < 2)
    (hrows : ((finS.C x ⟨i, hi⟩).T (finS.cz x (finCoins x ops) ⟨i, hi⟩)).tab.onVals.RowsAt
      (finS.wt x w ⟨i, hi⟩, vs x w ν (finCoins x ops) ⟨i, hi⟩)
      (finS.randAt x (finCoins x ops) t ⟨i, hi⟩).1.1 (finS.randAt x (finCoins x ops) t ⟨i, hi⟩).1.2.1.2.1) :
    ((finSrc vs x w ν t ops i).rep r).padLeaf = ((Masked.srcV x (finCoins x ops) w t i).rep r).padLeaf :=
  Masked.tabSrcView_padLeaf Masked.PE x (Masked.czExe Masked.PE x (finCoins x ops))
    (finS.swViewsAt x (finCoins x ops) (finS.wt x w) (vs x w ν (finCoins x ops)) t)
    (Masked.viewsE x (finCoins x ops) w t) hi hr
    (ZkView.Coined.swExe_leavesP ((finS.C x ⟨i, hi⟩).T (finS.cz x (finCoins x ops) ⟨i, hi⟩))
      (finS.wt x w ⟨i, hi⟩) (vs x w ν (finCoins x ops) ⟨i, hi⟩) (finS.randAt x (finCoins x ops) t ⟨i, hi⟩).1
      (finS.randAt x (finCoins x ops) t ⟨i, hi⟩).2 hrows)

variable (vs : finS.SwVals) (L₀ : LeanRelease Flock.Wit Tp)
  (src : Msg → Flock.Wit → Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
  (x : Msg) (w : Flock.Wit)

/-- **The pads commitment from the views is the firewall's** up to its columns and salts: in the masked scope, at a
witness in the relation, where the frame draws the tape's pads (`Loop.RepSrc`) and every table's rows hold at its base
(as `Loop.FinRows` states them), rep stream `k`'s pads commitment from the views' leaves has the size, root and tree of
`Finish.innerCommitP` on its request's pads and leaf salts. -/
theorem padsView_ic (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w)
    (hS : Loop.RepSrc L₀ src x w) {ν : finS.Nonce} {t : finS.Tape} {ops : List (Msg × Msg)}
    (hrows : ∀ j : Fin (Masked.jOf x), ((finS.C x j).T (finS.cz x (finCoins x ops) j)).tab.onVals.RowsAt
      (finS.wt x w j, vs x w ν (finCoins x ops) j)
      (finS.randAt x (finCoins x ops) t j).1.1 (finS.randAt x (finCoins x ops) t j).1.2.1.2.1)
    {s0 : _} {spec : SessionSpec} (hsx : FlockDeploy.setupOf true x = some (s0, spec, Masked.stsOf x))
    {schs : Array (_root_.Flock.Schedule × ℕ)}
    (hschs : (FlockComplete.schedsPZk (FlockDeploy.tabsOf (Masked.stsOf x)) : V _) = .ok schs)
    {k : ℕ} (hk : k < 2 * spec.tables.size) {ic : FlockComplete.InnerCommit} {l : Array Msg}
    (hic : (padsView vs x w ν t ops schs k).go l = (.ok ic, l))
    {h : List (Msg × Msg)} {tb : _root_.Flock.Firewall.Finish.Tab} {hi : Array ByteArray}
    {q : _root_.Flock.Firewall.Finish.RepIn} {f : ByteArray} (hRep : RepAt L₀.draws src x w (ν, t) h spec k tb hi q f)
    {ic' : _root_.Flock.Firewall.Finish.InnerCommit}
    (hic' : (_root_.Flock.Firewall.Finish.innerCommitP (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash)
        (_root_.Flock.Firewall.Finish.padsOf q.draws) (_root_.Flock.Firewall.Finish.saltAt q.draws.salts) :
        V _root_.Flock.Firewall.Finish.InnerCommit) = .ok ic') :
    ic.ch.size = ic'.toC.ch.size ∧ ic.root = ic'.toC.root ∧ ic.layers = ic'.toC.layers := by
  have bok : ∀ {α β : Type} {y : V α} {f : α → V β} {b : β}, y >>= f = .ok b → ∃ a, y = .ok a ∧ f a = .ok b := by
    intro α β y f b hb
    cases y with
    | error e => cases hb
    | ok a => exact ⟨a, rfl, hb⟩
  have hx := Masked.live_of hsc hR
  obtain ⟨-, -, hT, -⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hsx
  have hJ : Masked.jOf x = (Masked.stsOf x).size := by
    unfold Masked.jOf
    rw [ite_eq_left hx]
  have hj : k / 2 < Masked.jOf x := by omega
  have hr : k % 2 < 2 := Nat.mod_lt _ (by decide)
  have hk2 : Loop.frepOf (k / 2) (k % 2) = k := by
    unfold Loop.frepOf
    omega
  have hi' : k / 2 < (Masked.stsOf x).size := Masked.lt_of_jOf ⟨k / 2, hj⟩
  have htab : (FlockDeploy.tabsOf (Masked.stsOf x))[k / 2]! =
      _root_.Flock.Setup.ofCircuit (Masked.stAt x ⟨k / 2, hj⟩) := by
    rw [getElem!_pos (FlockDeploy.tabsOf (Masked.stsOf x)) (k / 2) (by simpa [FlockDeploy.tabsOf] using hi')]
    simp [FlockDeploy.tabsOf, Masked.stAt]
  obtain ⟨schs', hs', hall⟩ := Masked.schedsPZk_live hx
  have hss : schs' = schs := Except.ok.inj (hs'.symm.trans hschs)
  have hsj : schs[k / 2]! = ((Masked.zkTab x ⟨k / 2, hj⟩).sch, (Masked.zkTab x ⟨k / 2, hj⟩).tPad) := by
    rw [← hss]
    exact hall ⟨k / 2, hj⟩
  have hlayE : _root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (Masked.stsOf x))[k / 2]!.m
      (FlockDeploy.tabsOf (Masked.stsOf x))[k / 2]!.kLog (Masked.zkTab x ⟨k / 2, hj⟩).sch.k0 =
      Masked.layE x ⟨k / 2, hj⟩ := by
    rw [htab]
    rfl
  obtain ⟨N, hN⟩ : ∃ N, N = ((Masked.layE x ⟨k / 2, hj⟩).len + _root_.Flock.Zk.K_PAD).nextPowerOfTwo *
      2 ^ _root_.Flock.Zk.INNER_LOG_INV_RATE := ⟨_, rfl⟩
  -- the request's pads and leaf salts are the tape's
  obtain ⟨-, hq, -⟩ := hRep
  unfold repInOf at hq
  obtain ⟨⟨lay, ms⟩, hst, hq⟩ := bok hq
  have hqd : q.draws = _root_.Flock.Firewall.Frame.Draws.ofBytes lay (L₀.draws (ν, t) k) := by
    cases hq
    rfl
  have hLk : (LeanRelease.ofLoopLink L₀ src (realOf src)).lay x k = lay := by
    have e : Loop.startOf x k = _root_.Flock.Firewall.Drive.start (Masked.stsOf x)[k / 2]! := by
      unfold Loop.startOf
      rw [hsx]
      try dsimp only
    show ((Loop.startOf x k).toOption.map (·.1)).getD default = lay
    rw [e, hst]
    all_goals rfl
  have hlay := LeanRelease.lay_ofLoopLink (L₀ := L₀) src (realOf src) ⟨k / 2, hj⟩ ⟨k % 2, hr⟩ hst
  have hd := LeanRelease.draws_ofLoopLink (L₀ := L₀) src (realOf src)
  have hz := hS.size (LeanRelease.ofLoopLink L₀ src (realOf src)) hd ν t ⟨k / 2, hj⟩ ⟨k % 2, hr⟩ hlay
  have hpads := hS.pads (LeanRelease.ofLoopLink L₀ src (realOf src)) hd ν t ⟨k / 2, hj⟩ ⟨k % 2, hr⟩ hlay
    (Masked.UE.tapeZk t)
  have hq' : q.draws = (LeanRelease.ofLoopLink L₀ src (realOf src)).drawsAt x (ν, t) (Loop.frepOf (k / 2) (k % 2)) := by
    rw [hk2, hqd]
    show _ = _root_.Flock.Firewall.Frame.Draws.ofBytes ((LeanRelease.ofLoopLink L₀ src (realOf src)).lay x k)
      (L₀.draws (ν, t) k)
    rw [hLk]
  have hP : (_root_.Flock.Firewall.Finish.padsOf q.draws).toC =
      FlockComplete.padsOf (Masked.UE.tapeZk t) (k / 2) (k % 2) (Masked.layE x ⟨k / 2, hj⟩) :=
    Masked.pads_frame_eq (LeanRelease.ofLoopLink L₀ src (realOf src)) (ν, t) (Masked.UE.tapeZk t) Loop.frepOf x t
      ⟨k / 2, hj⟩ ⟨k % 2, hr⟩ q hq' hlay hz hpads
  have hicC : (FlockComplete.innerCommitP (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash)
      (_root_.Flock.Firewall.Finish.padsOf q.draws).toC (_root_.Flock.Firewall.Finish.saltAt q.draws.salts) :
      V FlockComplete.InnerCommit) = .ok ic'.toC := by
    have e := _root_.Flock.Firewall.Finish.innerCommitP_eq (M := V)
      (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash) (_root_.Flock.Firewall.Finish.padsOf q.draws)
      (_root_.Flock.Firewall.Finish.saltAt q.draws.salts)
    rw [hic'] at e
    exact e.symm
  have hpdH : ((FlockComplete.padsOf (Masked.UE.tapeZk t) (k / 2) (k % 2) (Masked.layE x ⟨k / 2, hj⟩)).h ++
      (FlockComplete.padsOf (Masked.UE.tapeZk t) (k / 2) (k % 2) (Masked.layE x ⟨k / 2, hj⟩)).padH).size =
        (Masked.layE x ⟨k / 2, hj⟩).len + _root_.Flock.Zk.K_PAD := by
    rw [Array.size_append, FlockComplete.padsOf_size]
    simp [FlockComplete.padsOf]
  have hsalt : ∀ j, j < N → FlockComplete.fitBytes 192 ((Masked.UE.tapeZk t).innerSalt (k / 2) (k % 2) j) =
      FlockComplete.fitBytes 192 (_root_.Flock.Firewall.Finish.saltAt q.draws.salts j) := by
    intro j hjN
    have hjc : j < _root_.Flock.Firewall.Frame.codeLen (Masked.layE x ⟨k / 2, hj⟩) := by
      rw [hN] at hjN
      exact hjN
    have hE := (hS.innerSalt (LeanRelease.ofLoopLink L₀ src (realOf src)) hd ν t ⟨k / 2, hj⟩ ⟨k % 2, hr⟩ hlay
      (Masked.UE.tapeZk t) j hjc).trans
      (innerSalt_fit (Masked.UE.tapeZk t) ((LeanRelease.ofLoopLink L₀ src (realOf src)).lay x)
        ((LeanRelease.ofLoopLink L₀ src (realOf src)).draws (ν, t)) Loop.frepOf (k / 2) (k % 2) j hz
        (by rw [hlay]; exact hjc))
    have hF : _root_.Flock.Firewall.Finish.saltAt q.draws.salts j =
        (_root_.Flock.Firewall.Frame.Draws.ofBytes ((LeanRelease.ofLoopLink L₀ src (realOf src)).lay x (Loop.frepOf (k / 2) (k % 2)))
          ((LeanRelease.ofLoopLink L₀ src (realOf src)).draws (ν, t) (Loop.frepOf (k / 2) (k % 2)))).salts.extract
          (_root_.Flock.Firewall.Frame.SALT * j) (_root_.Flock.Firewall.Frame.SALT * j + _root_.Flock.Firewall.Frame.SALT) := by
      rw [hq']
      all_goals rfl
    have e := hE.trans hF.symm
    have hsz : (_root_.Flock.Firewall.Finish.saltAt q.draws.salts j).size = 192 := by
      rw [← e]
      exact FlockComplete.JsonRT.fitBytes_size 192 _
    rw [Claims.ZKServer.fitBytes_of_size _ _ hsz]
    exact e
  have hTape : (FlockComplete.innerCommitP (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash)
      (FlockComplete.padsOf (Masked.UE.tapeZk t) (k / 2) (k % 2) (Masked.layE x ⟨k / 2, hj⟩))
      ((Masked.UE.tapeZk t).innerSalt (k / 2) (k % 2)) : V FlockComplete.InnerCommit) = .ok ic'.toC := by
    have hcg := FlockComplete.innerCommitP_salt_congr (M := V) (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash)
      (FlockComplete.padsOf (Masked.UE.tapeZk t) (k / 2) (k % 2) (Masked.layE x ⟨k / 2, hj⟩))
      (salt := (Masked.UE.tapeZk t).innerSalt (k / 2) (k % 2))
      (salt' := _root_.Flock.Firewall.Finish.saltAt q.draws.salts)
      (fun j hj' => hsalt j (by rw [hN, ← hpdH]; exact hj'))
    rw [hcg, ← hP]
    exact hicC
  -- the views' commitment is the honest views', which is the masked prover's on the tape's pads
  unfold padsView at hic
  rw [hsj] at hic
  dsimp only at hic
  rw [FlockComplete.padsTreeCut_leaf Masked.hsE _ (finSrc_padLeaf vs x w ν t ops hj hr (hrows ⟨k / 2, hj⟩)),
    Masked.padsTreeCut_srcV Masked.simCommit hx hR t (finCoins x ops) hj hr, hlayE, Functor.map_map,
    FlockComplete.go_map', FlockComplete.innerCommitP_quiet _root_.Flock.Sha512.hash _ _ l, hTape] at hic
  have hic1 := congrArg Prod.fst hic
  simp only [Except.map, Except.ok.injEq] at hic1
  subst hic1
  obtain ⟨hch, -⟩ := FlockComplete.Replay.innerCommitP_rows hicC
  rw [hP, hpdH, ← hN] at hch
  refine ⟨?_, rfl, rfl⟩
  simp only [hch, Array.size_map, Array.size_range, hN]

end Loop

end Claims.ZKServer
