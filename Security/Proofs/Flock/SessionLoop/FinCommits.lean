import Proofs.Flock.SessionLoop.FinishParts
import Proofs.Flock.SessionLoop.Link
import Proofs.Flock.Claims.ZK.CausalExe
import Proofs.Flock.Claims.ZK.LinkRecheck

/-!
# (a) `FinCommits`: the commitments the session reads from the views

At a statement in the masked scope with a witness (so `Live`, `Masked.live_of`), where the loop draws the tape's level 0
(`Loop.Level0Src`, `Loop.Src`'s level-0 half):

* `FlockComplete.commit0Cut_cap`: through SHA-512 under `hm96-sha512/v1`, `commit0Cut` returns, its cap layer `d − cap`
  of the tree over the source's level-0 leaves (`layerOf`, as `commitRowsP_cap_eq` has `commitRowsP`'s).
* `FlockComplete.padsTreeCut_ok`: through SHA-512, `padsTreeCut` returns.
* `Loop.finSrc_leaf0`: the views' level-0 leaves are the switched cut's (`cutV_swViewsAt`), so table `i`'s cap is the
  recheck's root of `Masked.linkIn` (`Masked.root_linkIn`), which is the root of the level 0 the loop drew
  (`Loop.level0Of_root`).
* **`Loop.finCommits_of`**: so the schedules return (`schedsPZk_live`), level 0's commitments from the views send
  nothing (`commit0Cut_quiet`) and carry the loop's roots (`rootsOf`), and each pads commitment sends nothing
  (`padsTreeCut_quiet`).
-/

namespace FlockComplete

open _root_.Flock

/-- **Level 0's commitment from a source** through SHA-512 under `hm96-sha512/v1` returns, and its cap is layer
`d − cap` of the tree over the source's level-0 leaves. -/
theorem commit0Cut_cap (lv : Level) (src : TabSrc) :
    ∃ cm, (commit0Cut (Hashes.ofSha512 Sha512.hash : Hashes V) MerkleScheme.hm96Sha512 lv src : V Commit) = .ok cm ∧
      cm.cap = layerOf (fun l r => Sha512.hash (l ++ r)) lv.d src.leaf0 (lv.d - lv.cap) := by
  unfold commit0Cut
  simp only [bind, Except.bind]
  generalize hF : forIn (m := V) (β := Array (Array ByteArray) × Array ByteArray) [0:lv.d - lv.cap]
    (#[(Array.range (2 ^ lv.d)).map src.leaf0], (Array.range (2 ^ lv.d)).map src.leaf0) _ = e
  obtain ⟨r, hr, hI⟩ := forIn_ok_inv' (lv.d - lv.cap)
    (#[(Array.range (2 ^ lv.d)).map src.leaf0], (Array.range (2 ^ lv.d)).map src.leaf0)
    (fun i (p : Array (Array ByteArray) × Array ByteArray) =>
      p.2 = layerOf (fun l r => Sha512.hash (l ++ r)) lv.d src.leaf0 i) rfl (by
      intro i _ p hp
      refine ⟨(p.1.push (layerOf (fun l r => Sha512.hash (l ++ r)) lv.d src.leaf0 (i + 1)),
        layerOf (fun l r => Sha512.hash (l ++ r)) lv.d src.leaf0 (i + 1)), ?_, rfl⟩
      rw [hp, mapM_ok _ fun i => hm96_node_std _ _]
      rfl) hF
  subst hr
  exact ⟨_, rfl, hI⟩

/-- **A rep's pads commitment from a source** through SHA-512 returns. -/
theorem padsTreeCut_ok (n : ℕ) (src : RepSrc) :
    ∃ ic, (padsTreeCut (Hashes.ofSha512 Sha512.hash : Hashes V) n src : V InnerCommit) = .ok ic := by
  unfold padsTreeCut
  simp only [bind, Except.bind]
  generalize hF : forIn (m := V) (β := Array (Array ByteArray) × Array ByteArray) [0:Nat.log2 n]
    (#[(Array.range n).map src.padLeaf], (Array.range n).map src.padLeaf) _ = e
  obtain ⟨r, hr, -⟩ := forIn_ok_inv' (Nat.log2 n)
    (#[(Array.range n).map src.padLeaf], (Array.range n).map src.padLeaf)
    (fun _ (_ : Array (Array ByteArray) × Array ByteArray) => True) trivial (by
      intro i _ p _
      refine ⟨(p.1.push ((Array.range (p.2.size / 2)).map fun i => Sha512.hash (p.2[2 * i]! ++ p.2[2 * i + 1]!)),
        (Array.range (p.2.size / 2)).map fun i => Sha512.hash (p.2[2 * i]! ++ p.2[2 * i + 1]!)), ?_, trivial⟩
      rw [mapM_ok _ fun i => (rfl : (MerkleOps.hm96With (Hashes.ofSha512 Sha512.hash : Hashes V)).node
        p.2[2 * i]! p.2[2 * i + 1]! = .ok (Sha512.hash (p.2[2 * i]! ++ p.2[2 * i + 1]!)))]
      rfl) hF
  subst hr
  exact ⟨_, rfl⟩

end FlockComplete

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128 Stmt SessionSpec)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- `ofLoop`'s session types, as in `FrameZK`

namespace Loop

/-- **The views' level-0 leaves are the switched cut's**: table `i`'s leaf in the views' source is `leafAt` of the
switched prover's cut at the opened coins, the workers' values and the tape (`cutV_swViewsAt`). -/
theorem finSrc_leaf0 (vs : finS.SwVals) (x : Msg) (w : Flock.Wit) (ν : finS.Nonce) (t : finS.Tape)
    (ops : List (Msg × Msg)) {i : ℕ} (hi : i < Masked.jOf x) :
    (finSrc vs x w ν t ops i).leaf0 =
      Masked.leafAt Masked.PE x (Masked.czExe Masked.PE x (finCoins x ops))
        (swCutAt (Masked.tabC Masked.PE x) (Masked.czExe Masked.PE x (finCoins x ops)) (Masked.wtOf Masked.PE x w)
          (vs x w ν (finCoins x ops)) (Masked.UE.tape x t).1
          (swExeOf (Masked.tabC Masked.PE x) (Masked.czExe Masked.PE x (finCoins x ops)) (Masked.wtOf Masked.PE x w)
            (vs x w ν (finCoins x ops)) (Masked.UE.tape x t).1 (Masked.UE.tape x t).2.1)) ⟨i, hi⟩ := by
  have e1 : (finSrc vs x w ν t ops i).leaf0 =
      Masked.leafAt Masked.PE x (Masked.czExe Masked.PE x (finCoins x ops))
        (fun j => Masked.viewCut (finS.swViewsAt x (finCoins x ops) (finS.wt x w) (vs x w ν (finCoins x ops)) t j))
        ⟨i, hi⟩ := by
    simp only [finSrc, Masked.tabSrcView, Masked.tabSrcCut, hi, ↓reduceDIte]
  rw [e1]
  have h0 := congrArg FlockSoundness.ZK.Table.Cut.leaves0
    (congrFun (finS.cutV_swViewsAt x (finCoins x ops) (finS.wt x w) (vs x w ν (finCoins x ops)) t) ⟨i, hi⟩)
  have h : (Masked.viewCut (finS.swViewsAt x (finCoins x ops) (finS.wt x w) (vs x w ν (finCoins x ops)) t
      ⟨i, hi⟩)).leaves0 =
      (swCutAt (Masked.tabC Masked.PE x) (Masked.czExe Masked.PE x (finCoins x ops)) (Masked.wtOf Masked.PE x w)
        (vs x w ν (finCoins x ops)) (Masked.UE.tape x t).1
        (swExeOf (Masked.tabC Masked.PE x) (Masked.czExe Masked.PE x (finCoins x ops)) (Masked.wtOf Masked.PE x w)
          (vs x w ν (finCoins x ops)) (Masked.UE.tape x t).1 (Masked.UE.tape x t).2.1) ⟨i, hi⟩).leaves0 := h0
  have hc : ∀ c₁ c₂ : ∀ j, ((Masked.tabC Masked.PE x j).T (Masked.czExe Masked.PE x (finCoins x ops) j)).tab.Cut,
      (c₁ ⟨i, hi⟩).leaves0 = (c₂ ⟨i, hi⟩).leaves0 →
        Masked.leafAt Masked.PE x (Masked.czExe Masked.PE x (finCoins x ops)) c₁ ⟨i, hi⟩ =
          Masked.leafAt Masked.PE x (Masked.czExe Masked.PE x (finCoins x ops)) c₂ ⟨i, hi⟩ := by
    intro c₁ c₂ e
    unfold Masked.leafAt
    rw [e]
  exact hc _ _ h

variable (vs : finS.SwVals) (L₀ : LeanRelease Flock.Wit Tp)
  (src : Msg → Flock.Wit → Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
  (real : Msg → Flock.Wit → Tp → ℕ → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
  (wk : (LeanRelease.ofLoopLink L₀ src real).toSrv.Worker) (x : Msg) (w : Flock.Wit)

/-- **(a) at the deployed scope**: at a statement in the masked scope with a witness, where the loop draws the tape's
level 0 (`Level0Src`), the schedules return, level 0's commitments from the views send nothing and carry the loop's
roots, and each rep's pads commitment sends nothing. -/
theorem finCommits_of (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w)
    (hS : Level0Src src x w) : FinCommits vs L₀ src real wk x w := by
  intro ν t κ m₀ ρ₀ ops _ s0 spec hsx
  have hx : Masked.Live x := Masked.live_of hsc hR
  obtain ⟨-, -, -, hall⟩ := FlockComplete.ZkTables.tables_of_setup hsc hsx
  have hn := Masked.tabs_size hx
  have hJ : Masked.jOf x = (Masked.stsOf x).size := by
    unfold Masked.jOf
    exact ite_eq_left_of_eq_true _ _ (eq_true hx)
  obtain ⟨schs, hs, hschs⟩ := Masked.schedsPZk_live hx
  refine ⟨schs, hs, ?_⟩
  have hms : ∀ i < (FlockDeploy.tabsOf (Masked.stsOf x)).size,
      (FlockDeploy.tabsOf (Masked.stsOf x))[i]!.merkle = _root_.Flock.MerkleScheme.hm96Sha512 := by
    intro i hi
    have hi' : i < (Masked.stsOf x).size := by simpa [FlockDeploy.tabsOf] using hi
    rw [getElem!_pos (FlockDeploy.tabsOf (Masked.stsOf x)) i hi]
    simp only [FlockDeploy.tabsOf, Array.getElem_map]
    exact (hall i hi').2.2
  let lvAt : ℕ → _root_.Flock.Level := fun i => schs[i]!.1.levels.getD 0 default
  let cmOf : ℕ → FlockComplete.Commit := fun i =>
    Classical.choose (FlockComplete.commit0Cut_cap (lvAt i) (finSrc vs x w ν t ops i))
  have hcm : ∀ i, (FlockComplete.commit0Cut (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash : _root_.Flock.Hashes V)
      _root_.Flock.MerkleScheme.hm96Sha512 (lvAt i) (finSrc vs x w ν t ops i) : V _) = .ok (cmOf i) ∧
      (cmOf i).cap = FlockComplete.layerOf (fun l r => _root_.Flock.Sha512.hash (l ++ r)) (lvAt i).d
        (finSrc vs x w ν t ops i).leaf0 ((lvAt i).d - (lvAt i).cap) := fun i =>
    Classical.choose_spec (FlockComplete.commit0Cut_cap (lvAt i) (finSrc vs x w ν t ops i))
  refine ⟨(Array.range (FlockDeploy.tabsOf (Masked.stsOf x)).size).map cmOf, fun l => ?_, ?_, fun k _ l => ?_⟩
  · have hq : FlockComplete.quietS.R
        ((Array.range (FlockDeploy.tabsOf (Masked.stsOf x)).size).mapM fun i =>
          FlockComplete.commit0Cut (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash : _root_.Flock.Hashes V)
            (FlockDeploy.tabsOf (Masked.stsOf x))[i]!.merkle (lvAt i) (finSrc vs x w ν t ops i))
        (commitsView vs x w ν t ops schs) :=
      _root_.Flock.With.Sim.mapM_array FlockComplete.quietS _ fun i =>
        FlockComplete.CutEq.commit0Cut_quiet _ _ _ _
    rw [hq l, FlockComplete.mapM_ok_mem cmOf _ fun i hi => ?_]
    rw [hms i (by simpa using hi)]
    exact (hcm i).1
  · rw [Array.map_map]
    unfold rootsOf
    have hsz : (FlockDeploy.tabsOf (Masked.stsOf x)).size = (Masked.stsOf x).size := by
      simp [FlockDeploy.tabsOf]
    rw [hsz]
    refine Array.map_congr_left fun i hi => ?_
    have hi' : i < Masked.jOf x := by rw [hJ]; simpa using hi
    obtain ⟨l0, hl, hin⟩ := Loop.Level0Src.linkLevel0 hS ν t ⟨i, hi'⟩
    have hl' : level0Of src x w (ν, t) (Masked.stsOf x) i = .ok l0 := hl
    show FlockComplete.flat (cmOf i).cap = _
    rw [hl']
    show FlockComplete.flat (cmOf i).cap = l0.root
    rw [Loop.level0Of_root hl', hin, Masked.root_linkIn x w t (Masked.czExe Masked.PE x (finCoins x ops))
      (vs x w ν (finCoins x ops)) ⟨i, hi'⟩, (hcm i).2, finSrc_leaf0 vs x w ν t ops hi']
    have e : lvAt i = Masked.lv0 x ⟨i, hi'⟩ := by
      show schs[i]!.1.levels.getD 0 default = _
      rw [hschs ⟨i, hi'⟩]
      rfl
    rw [e]
  · obtain ⟨ic, hic⟩ := FlockComplete.padsTreeCut_ok
      (((_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (Masked.stsOf x))[k / 2]!.m
        (FlockDeploy.tabsOf (Masked.stsOf x))[k / 2]!.kLog schs[k / 2]!.1.k0).len + _root_.Flock.Zk.K_PAD).nextPowerOfTwo *
        2 ^ _root_.Flock.Zk.INNER_LOG_INV_RATE) ((finSrc vs x w ν t ops (k / 2)).rep (k % 2))
    refine ⟨ic, ?_⟩
    unfold padsView
    rw [FlockComplete.CutEq.padsTreeCut_quiet _ _ _ l, hic]

end Loop

end Claims.ZKServer
