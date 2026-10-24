import Proofs.Flock.SessionLoop.FinRep
import Proofs.Flock.SessionLoop.FinCommits
import Proofs.Flock.SessionLoop.Tables
import Proofs.Flock.Claims.ZK.FinishLevel0

/-!
# Level 0's commitment from the views is the loop's table's

At (c.3)'s context, where the loop draws the tape's level 0 (`Loop.Level0Src`), rep stream `k`'s level-0 commitment from
the views (`commitsView`'s `k / 2`-th) has the cap and tree of the one the loop's table keeps (`tb.cm0`, read off the
loop's `Level0` by `Finish.commitOf`), as `Loop.repCut_eraseZ` asks of the two commitments:

* `FlockComplete.commit0Cut_tree`: through SHA-512 under `hm96-sha512/v1`, `commit0Cut` returns, its layers are the
  tree's layers `0, …, d − cap` over the source's level-0 leaves and its cap the last (`commit0Cut_cap`, with the
  layers).
* `Masked.leafAt_linkIn`: below `2^d`, the switched cut's level-0 leaf is the recheck's leaf of `Masked.linkIn`
  (`root_linkIn`'s, leaf by leaf).
* `Finish.tabOfLevel0_cm0`, `Loop.Level0Src.realOf_cm0`: the loop's table at rep stream `2j + r` keeps the recheck's tree
  of `Masked.linkIn` at level 0 of table `j`'s `--zk` schedule.
* **`Loop.cms_cm0`**: so the views' commitment and the table's have one cap and one tree.
-/

namespace FlockComplete

open _root_.Flock

/-- **Level 0's commitment from a source, with its tree**: through SHA-512 under `hm96-sha512/v1`, `commit0Cut`
returns, its layers are layers `0, …, d − cap` of the tree over the source's level-0 leaves and its cap the last. -/
theorem commit0Cut_tree (lv : Level) (src : TabSrc) :
    ∃ cm, (commit0Cut (Hashes.ofSha512 Sha512.hash : Hashes V) MerkleScheme.hm96Sha512 lv src : V Commit) = .ok cm ∧
      cm.layers = (Array.range (lv.d - lv.cap + 1)).map (layerOf (fun l r => Sha512.hash (l ++ r)) lv.d src.leaf0) ∧
      cm.cap = layerOf (fun l r => Sha512.hash (l ++ r)) lv.d src.leaf0 (lv.d - lv.cap) := by
  set L := layerOf (fun l r => Sha512.hash (l ++ r)) lv.d src.leaf0 with hL
  unfold commit0Cut
  simp only [bind, Except.bind]
  generalize hF : forIn (m := V) (β := Array (Array ByteArray) × Array ByteArray) [0:lv.d - lv.cap]
    (#[(Array.range (2 ^ lv.d)).map src.leaf0], (Array.range (2 ^ lv.d)).map src.leaf0) _ = e
  obtain ⟨r, hr, hI⟩ := forIn_ok_inv' (lv.d - lv.cap)
    (#[(Array.range (2 ^ lv.d)).map src.leaf0], (Array.range (2 ^ lv.d)).map src.leaf0)
    (fun i (p : Array (Array ByteArray) × Array ByteArray) => p.1 = (Array.range (i + 1)).map L ∧ p.2 = L i)
    ⟨show #[L 0] = (Array.range 1).map L by
      rw [show Array.range 1 = #[0] by first | rfl | simp [Array.range_succ_eq_map], Array.map_singleton], rfl⟩ (by
      intro i _ p hp
      refine ⟨(p.1.push (L (i + 1)), L (i + 1)), ?_, ?_, rfl⟩
      · rw [hp.2, mapM_ok _ fun i => hm96_node_std _ _]
        rfl
      · show p.1.push (L (i + 1)) = (Array.range (i + 1 + 1)).map L
        rw [hp.1, Array.range_succ (n := i + 1), Array.append_singleton, Array.map_push]) hF
  subst hr
  exact ⟨_, rfl, hI.1, hI.2⟩

end FlockComplete

namespace Flock.Firewall.Finish

/-- **A table from a level 0 keeps its commitment** as `commitOf` reads it off. -/
theorem tabOfLevel0_cm0 {st : _root_.Flock.Setup} {l : _root_.Flock.Firewall.Level0.Level0}
    {t : _root_.Flock.Firewall.Finish.Tab} (h : _root_.Flock.Firewall.Finish.tabOfLevel0 st l = .ok t) :
    t.cm0 = commitOf l := by
  unfold _root_.Flock.Firewall.Finish.tabOfLevel0 at h
  simp only [bind, Except.bind] at h
  split at h
  · cases h
  split at h
  · cases h
  rcases hs : _root_.Flock.Zk.schedule st.m with e | ⟨sch, tPad⟩
  · rw [hs] at h; cases h
  · rw [hs] at h
    simp only at h
    split at h
    · cases h
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    rfl

end Flock.Firewall.Finish

namespace Claims.ZKMal

open Claims
open FlockSoundness.Discharge
open CoinLayer

namespace Masked

/-- **Below `2^d`, the switched cut's level-0 leaf is the recheck's leaf of `linkIn`**, at every coin vector and any
values the workers supply (`root_linkIn`'s, leaf by leaf). -/
theorem leafAt_linkIn (x : Msg) (w : Claims.Flock.Wit) (t : UE.Tape) (cz : ∀ j, Composed.CoinsZK (tabC PE x j).Rest)
    (v : ∀ j, ((tabC PE x j).T (cz j)).tab.Vals) (j : Fin (jOf x)) {q : ℕ} (hq : q < 2 ^ (lv0 x j).d) :
    leafAt PE x cz (swCutAt (tabC PE x) cz (wtOf PE x w) v (UE.tape x t).1
        (swExeOf (tabC PE x) cz (wtOf PE x w) v (UE.tape x t).1 (UE.tape x t).2.1)) j q =
      (_root_.Flock.Firewall.Link.leaves (lv0 x j) (zkTab x j).tPad (linkIn x w t j))[q]! := by
  obtain ⟨-, -, -, -, hd, -⟩ := tabC_z PE x j
  have hq' : q < (tabC PE x j).z.nS + (tabC PE x j).z.nU := lt_of_lt_of_eq hq hd.symm
  rw [_root_.Flock.Firewall.Link.leaves_getElem! _ _ _ hq]
  refine (leafAt_swCutAt PE x cz (wtOf PE x w) v (UE.tape x t).1
    (swExeOf (tabC PE x) cz (wtOf PE x w) v (UE.tape x t).1 (UE.tape x t).2.1) j hq').trans
    ((swExeOf_inl (tabC PE x) cz (wtOf PE x w) v (UE.tape x t).1 (UE.tape x t).2.1 j ⟨q, hq'⟩).trans
    ((exeOf_inl PE (TapeLayout.exe PE) exeTape (agrees0_exe PE) w t cz j ⟨q, hq'⟩).trans ?_))
  rw [_root_.Flock.Firewall.Link.rows_eq] <;> rfl

end Masked

end Claims.ZKMal

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128 Stmt SessionSpec)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- `ofLoop`'s session types, as in `FrameZK`

/-- **The loop's table keeps the recheck's tree of `linkIn`** where `src` draws the tape's level 0: at rep stream
`2j + r`, the level-0 commitment of the loop's table (`Loop.realOf`) has the layers `0, …, d − cap` of `Link.layer` over
`Masked.linkIn`'s leaves at level 0 of table `j`'s `--zk` schedule, and the last as its cap. -/
theorem Loop.Level0Src.realOf_cm0 {src : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray)} {x : Msg} {w : Flock.Wit}
    (hS : Loop.Level0Src src x w) (ν : (CoinLayer.session Loop.loopNr Masked.dataE).Nonce)
    (t : (CoinLayer.session Loop.loopNr Masked.dataE).Tape) (j : Fin (Masked.jOf x)) (r : Fin 2)
    {tb : _root_.Flock.Firewall.Finish.Tab} {hi : Array ByteArray}
    (h : Loop.realOf src x w (ν, t) (2 * j.val + r.val) = .ok (tb, hi)) :
    tb.cm0.layers = (Array.range ((Masked.lv0 x j).d - (Masked.lv0 x j).cap + 1)).map
        (_root_.Flock.Firewall.Link.layer (Masked.lv0 x j).d fun q =>
          (_root_.Flock.Firewall.Link.leaves (Masked.lv0 x j) (Masked.zkTab x j).tPad (Masked.linkIn x w t j))[q]!) ∧
      tb.cm0.cap = _root_.Flock.Firewall.Link.layer (Masked.lv0 x j).d (fun q =>
          (_root_.Flock.Firewall.Link.leaves (Masked.lv0 x j) (Masked.zkTab x j).tPad (Masked.linkIn x w t j))[q]!)
        ((Masked.lv0 x j).d - (Masked.lv0 x j).cap) := by
  obtain ⟨l, hl, hin⟩ := hS ν t j r
  obtain ⟨s0, spec, hsx⟩ := Masked.setupOf_of_pos (lt_of_le_of_lt (Nat.zero_le _) (Masked.lt_of_jOf j))
  unfold Loop.realOf at h
  rw [hsx] at h
  dsimp only at h
  unfold Loop.level0At at hl
  rcases hs : src x w (ν, t) (2 * j.val + r.val) with e | ⟨g, d, hi'⟩
  · rw [hs] at hl
    cases hl
  · rw [hs] at h hl
    simp only [bind, Except.bind] at h hl
    rw [hl] at h
    simp only at h
    split at h
    · cases h
    · rename_i tb' htb
      simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, -⟩ := h
      rw [_root_.Flock.Firewall.Finish.tabOfLevel0_cm0 htb]
      obtain ⟨hm, -, ⟨s, hz, hlv⟩, -, hlay, -⟩ := _root_.Flock.Firewall.Level0.ofDraws_ok hl
      have hm' : l.m = (Masked.stAt x j).m := by
        rw [show l.m = l.toIn.m from rfl, hin]
        rfl
      rw [hm] at hm'
      rw [hm'] at hz
      obtain ⟨hs', ht'⟩ := Prod.mk.inj (Except.ok.inj (hz.symm.trans (Masked.zkTab x j).hz))
      have hlv' : l.lv = Masked.lv0 x j := by rw [hlv, hs']; rfl
      rw [hlv', ht', hin] at hlay
      unfold _root_.Flock.Firewall.Finish.commitOf
      refine ⟨hlay, ?_⟩
      show l.layers.back?.getD #[] = _
      rw [hlay, Array.range_succ (n := (Masked.lv0 x j).d - (Masked.lv0 x j).cap), Array.append_singleton,
        Array.map_push, Array.back?_push, Option.getD_some]

namespace Loop

variable (vs : finS.SwVals) (src : Msg → Flock.Wit → Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
  (x : Msg) (w : Flock.Wit)

/-- **Level 0's commitment from the views is the loop's table's**: at a statement in the masked scope with a witness,
where the loop draws the tape's level 0 (`Level0Src`), at the session's schedules, wherever the views' commitments
return `cms` and rep stream `k`'s table is `tb` (`RepAt`), `cms[k / 2]` has `tb`'s level-0 cap and tree. -/
theorem cms_cm0 (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w)
    (hS : Level0Src src x w) {ν : finS.Nonce} {t : finS.Tape} {ops : List (Msg × Msg)} {s0 spec}
    (hsx : FlockDeploy.setupOf true x = some (s0, spec, Masked.stsOf x)) {schs : Array (_root_.Flock.Schedule × ℕ)}
    (hschs : (FlockComplete.schedsPZk (FlockDeploy.tabsOf (Masked.stsOf x)) : V _) = .ok schs)
    {cms : Array FlockComplete.Commit} (hcms : ∀ l, (commitsView vs x w ν t ops schs).go l = (.ok cms, l))
    {k : ℕ} (hk : k < 2 * spec.tables.size) {draws : Tp → ℕ → ByteArray} {h : List (Msg × Msg)} {tb hi q f}
    (hRep : RepAt draws src x w (ν, t) h spec k tb hi q f) :
    cms[k / 2]!.cap = tb.cm0.toC.cap ∧ cms[k / 2]!.layers = tb.cm0.toC.layers := by
  have hx : Masked.Live x := Masked.live_of hsc hR
  obtain ⟨-, -, hT, hall⟩ := FlockComplete.ZkTables.tables_of_setup hsc hsx
  have hJ : Masked.jOf x = (Masked.stsOf x).size := by
    unfold Masked.jOf
    exact ite_eq_left_of_eq_true _ _ (eq_true hx)
  have hj : k / 2 < Masked.jOf x := by omega
  obtain ⟨schs', hs', hschs'⟩ := Masked.schedsPZk_live hx
  have e : schs' = schs := Except.ok.inj (hs'.symm.trans hschs)
  subst e
  have hms : ∀ i < (FlockDeploy.tabsOf (Masked.stsOf x)).size,
      (FlockDeploy.tabsOf (Masked.stsOf x))[i]!.merkle = _root_.Flock.MerkleScheme.hm96Sha512 := by
    intro i hi
    have hi' : i < (Masked.stsOf x).size := by simpa [FlockDeploy.tabsOf] using hi
    rw [getElem!_pos (FlockDeploy.tabsOf (Masked.stsOf x)) i hi]
    simp only [FlockDeploy.tabsOf, Array.getElem_map]
    exact (hall i hi').2.2
  let lvAt : ℕ → _root_.Flock.Level := fun i => schs'[i]!.1.levels.getD 0 default
  let cmOf : ℕ → FlockComplete.Commit := fun i =>
    Classical.choose (FlockComplete.commit0Cut_tree (lvAt i) (finSrc vs x w ν t ops i))
  have hcm : ∀ i, (FlockComplete.commit0Cut (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash :
      _root_.Flock.Hashes V) _root_.Flock.MerkleScheme.hm96Sha512 (lvAt i) (finSrc vs x w ν t ops i) : V _) =
        .ok (cmOf i) ∧
      (cmOf i).layers = (Array.range ((lvAt i).d - (lvAt i).cap + 1)).map
        (FlockComplete.layerOf (fun l r => _root_.Flock.Sha512.hash (l ++ r)) (lvAt i).d (finSrc vs x w ν t ops i).leaf0) ∧
      (cmOf i).cap = FlockComplete.layerOf (fun l r => _root_.Flock.Sha512.hash (l ++ r)) (lvAt i).d
        (finSrc vs x w ν t ops i).leaf0 ((lvAt i).d - (lvAt i).cap) := fun i =>
    Classical.choose_spec (FlockComplete.commit0Cut_tree (lvAt i) (finSrc vs x w ν t ops i))
  have hq : FlockComplete.quietS.R
      ((Array.range (FlockDeploy.tabsOf (Masked.stsOf x)).size).mapM fun i =>
        FlockComplete.commit0Cut (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash : _root_.Flock.Hashes V)
          (FlockDeploy.tabsOf (Masked.stsOf x))[i]!.merkle (lvAt i) (finSrc vs x w ν t ops i))
      (commitsView vs x w ν t ops schs') :=
    _root_.Flock.With.Sim.mapM_array FlockComplete.quietS _ fun i =>
      FlockComplete.CutEq.commit0Cut_quiet _ _ _ _
  have hV : ((Array.range (FlockDeploy.tabsOf (Masked.stsOf x)).size).mapM fun i =>
      FlockComplete.commit0Cut (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash : _root_.Flock.Hashes V)
        (FlockDeploy.tabsOf (Masked.stsOf x))[i]!.merkle (lvAt i) (finSrc vs x w ν t ops i)) =
      .ok ((Array.range (FlockDeploy.tabsOf (Masked.stsOf x)).size).map cmOf) :=
    FlockComplete.mapM_ok_mem cmOf _ fun i hi => by
      rw [hms i (by simpa using hi)]
      exact (hcm i).1
  have hv := hcms #[]
  rw [hq #[], hV] at hv
  have hc : cms = (Array.range (FlockDeploy.tabsOf (Masked.stsOf x)).size).map cmOf :=
    (Except.ok.inj (Prod.mk.inj hv).1).symm
  have hj' : k / 2 < (FlockDeploy.tabsOf (Masked.stsOf x)).size := by
    simpa [FlockDeploy.tabsOf, ← hJ] using hj
  have hci : cms[k / 2]! = cmOf (k / 2) := by
    rw [hc, getElem!_pos _ (k / 2) (by simpa using hj'), Array.getElem_map, Array.getElem_range]
  have hlv : lvAt (k / 2) = Masked.lv0 x ⟨k / 2, hj⟩ := by
    show schs'[k / 2]!.1.levels.getD 0 default = _
    rw [hschs' ⟨k / 2, hj⟩]
    rfl
  have hk2 : k = 2 * (⟨k / 2, hj⟩ : Fin (Masked.jOf x)).val + (⟨k % 2, Nat.mod_lt _ (by omega)⟩ : Fin 2).val := by
    show k = 2 * (k / 2) + k % 2
    omega
  obtain ⟨hr, -, -⟩ := hRep
  rw [hk2] at hr
  obtain ⟨hly, hcp⟩ := Level0Src.realOf_cm0 hS ν t ⟨k / 2, hj⟩ ⟨k % 2, Nat.mod_lt _ (by omega)⟩ hr
  have hL : FlockComplete.layerOf (fun l r => _root_.Flock.Sha512.hash (l ++ r)) (Masked.lv0 x ⟨k / 2, hj⟩).d
      (finSrc vs x w ν t ops (k / 2)).leaf0 =
      _root_.Flock.Firewall.Link.layer (Masked.lv0 x ⟨k / 2, hj⟩).d (fun q =>
        (_root_.Flock.Firewall.Link.leaves (Masked.lv0 x ⟨k / 2, hj⟩) (Masked.zkTab x ⟨k / 2, hj⟩).tPad
          (Masked.linkIn x w t ⟨k / 2, hj⟩))[q]!) := by
    funext h'
    rw [_root_.Flock.Firewall.Link.layer_eq]
    refine _root_.Flock.Firewall.Link.layerOf_congr _ _ (fun q hq => ?_) h'
    rw [finSrc_leaf0 vs x w ν t ops hj]
    exact Masked.leafAt_linkIn x w t _ _ ⟨k / 2, hj⟩ hq
  rw [hci, (hcm (k / 2)).2.1, (hcm (k / 2)).2.2, hlv, hL]
  exact ⟨hcp.symm, hly.symm⟩

end Loop

end Claims.ZKServer
