import Proofs.Flock.Claims.ZK.FinishCommit
import Proofs.Flock.Claims.Complete.ZkTables.Round0

/-!
# Level 0 as the session loop keeps it is `commit0`'s

The finish on the session loop's frames port reads table `i`'s level-0 commitment off the loop's `Level0`
(`Finish.commitOf`) instead of committing it again (`Finish.commit0`, `flock-firewall finish`'s `table` path). Here the
two are one, wherever the loop drew that `Level0` (`Level0.ofDraws`):

* `commitRowsP_tree`: through `Hashes` whose node and leaf are functions, `FlockComplete.commitRowsP` returns, its
  layers are the tree's layers from the leaves to the cap and its cap the top one (`commitRowsP_cap_eq`'s, with the
  layers).
* `commitRows_tree`: so the port's `commitRows` keeps `Link.layer`'s layers over the rows' `hm96-sha512/v1` leaves
  (`commitRowsP_eq`).
* **`commitOf_eq`**: a `Level0` that `ofDraws` returns commits as `commit0` at its own level, `t_pad` and table
  (`Level0.ofDraws_ok`: its rows and layers are `Link`'s of `toIn`).
-/

namespace Flock.Firewall.Finish


/-- **A level's tree from its leaves**: through `Hashes` whose node is `nd` and whose leaf is `lf`, `commitRowsP`
returns, its layers are layers `0, …, d − cap` of the tree over the rows' leaves under their fitted salts, and its cap
the last of them. -/
theorem commitRowsP_tree {hs : Hashes V} {ms : MerkleScheme} {nd : ByteArray → ByteArray → ByteArray}
    {lf : Array F128 → ByteArray → ByteArray} (hn : ∀ l r, (ms.opsWith hs).node l r = .ok (nd l r))
    (hl : ∀ row s, (ms.opsWith hs).leaf row s = .ok (lf row s)) (lv : Level) (rows : Array (Array F128))
    (salt : ℕ → ByteArray) :
    ∃ cm, (FlockComplete.commitRowsP hs ms lv rows salt : V FlockComplete.Commit) = .ok cm ∧
      cm.layers = (Array.range (lv.d - lv.cap + 1)).map
        (FlockComplete.layerOf nd lv.d fun q => lf rows[q]! (FlockComplete.fitBytes ms.saltBytes (salt q))) ∧
      cm.cap = FlockComplete.layerOf nd lv.d (fun q => lf rows[q]! (FlockComplete.fitBytes ms.saltBytes (salt q))) (lv.d - lv.cap) := by
  set L := FlockComplete.layerOf nd lv.d (fun q => lf rows[q]! (FlockComplete.fitBytes ms.saltBytes (salt q))) with hL
  have h0 : ((Array.range (2 ^ lv.d)).mapM fun q => (ms.opsWith hs).leaf rows[q]!
      ((Array.range (2 ^ lv.d)).map fun q => FlockComplete.fitBytes ms.saltBytes (salt q))[q]! : V _) = .ok (L 0) := by
    rw [FlockComplete.mapM_ok _ fun q => hl _ _]
    congr 1
    refine Array.ext (by simp [hL, FlockComplete.layerOf]) fun q h1 _ => ?_
    have hq : q < 2 ^ lv.d := by simpa using h1
    simp only [hL, FlockComplete.layerOf, Array.getElem_map, Array.getElem_range]
    rw [getElem!_pos ((Array.range (2 ^ lv.d)).map fun q => FlockComplete.fitBytes ms.saltBytes (salt q)) q (by simpa using hq)]
    simp
  unfold FlockComplete.commitRowsP
  simp only [bind, Except.bind, h0]
  generalize hF : forIn (m := V) (β := Array (Array ByteArray) × Array ByteArray) [0:lv.d - lv.cap] (#[L 0], L 0) _ = e
  obtain ⟨r, hr, hI⟩ := FlockComplete.forIn_ok_inv' (lv.d - lv.cap) (#[L 0], L 0)
    (fun i (p : Array (Array ByteArray) × Array ByteArray) => p.1 = (Array.range (i + 1)).map L ∧ p.2 = L i)
    ⟨show #[L 0] = (Array.range 1).map L by
      rw [show Array.range 1 = #[0] by first | rfl | simp [Array.range_succ_eq_map], Array.map_singleton], rfl⟩ (by
      intro i _ p hp
      refine ⟨(p.1.push (L (i + 1)), L (i + 1)), ?_, ?_, rfl⟩
      · rw [hp.2, FlockComplete.mapM_ok _ fun i => hn _ _]
        rfl
      · show p.1.push (L (i + 1)) = (Array.range (i + 1 + 1)).map L
        rw [hp.1, Array.range_succ (n := i + 1), Array.append_singleton, Array.map_push]) hF
  subst hr
  exact ⟨_, rfl, hI.1, hI.2⟩

/-- **The port's tree** of given rows: `commitRows` keeps layers `0, …, d − cap` of `Link.layer` over the rows'
`hm96-sha512/v1` leaves under their salts fitted to 192 bytes, and the last as its cap. -/
theorem commitRows_tree (lv : Level) (rows : Array (Array F128)) (salt : ℕ → ByteArray) :
    (commitRows lv rows salt).layers = (Array.range (lv.d - lv.cap + 1)).map
        (Link.layer lv.d fun q => MerkleScheme.hm96Sha512.leaf rows[q]! (fitBytes 192 (salt q))) ∧
      (commitRows lv rows salt).cap =
        Link.layer lv.d (fun q => MerkleScheme.hm96Sha512.leaf rows[q]! (fitBytes 192 (salt q))) (lv.d - lv.cap) := by
  obtain ⟨cm, hcm, hly, hcp⟩ := commitRowsP_tree (hs := Hashes.ofSha512 Sha512.hash) (ms := MerkleScheme.hm96Sha512)
    FlockComplete.hm96_node_std FlockComplete.hm96_leaf_std lv rows salt
  rw [commitRowsP_eq (M := V)] at hcm
  have e : (commitRows lv rows salt).toC = cm := Except.ok.inj hcm
  subst e
  have hlay : ∀ lf : ℕ → ByteArray, FlockComplete.layerOf (fun l r => Sha512.hash (l ++ r)) lv.d lf = Link.layer lv.d lf :=
    fun lf => funext fun h => (Link.layer_eq lv.d lf h).symm
  rw [hlay] at hly hcp
  exact ⟨hly, hcp⟩

/-- **The session loop's level 0 is `commit0`'s**: where `ofDraws` returns `l`, the commitment read off it
(`commitOf`) is `commit0`'s at its level, `t_pad` and table, so the loop's finish proves at the commitment the `table`
path commits. -/
theorem commitOf_eq {m : ℕ} {g : Array F128} {d : ByteArray} {l : Level0.Level0} (h : Level0.ofDraws m g d = .ok l) :
    commitOf l = commit0 l.lv l.tPad l.toIn := by
  obtain ⟨-, -, -, hrows, hlayers, -⟩ := Level0.ofDraws_ok h
  obtain ⟨hly, hcp⟩ := commitRows_tree l.lv (Link.rows l.lv l.tPad l.toIn.g l.toIn.extra l.toIn.lanePad) l.toIn.salt
  have hlf : (Link.layer l.lv.d fun q => (Link.leaves l.lv l.tPad l.toIn)[q]!) =
      Link.layer l.lv.d fun q => MerkleScheme.hm96Sha512.leaf
        (Link.rows l.lv l.tPad l.toIn.g l.toIn.extra l.toIn.lanePad)[q]! (fitBytes 192 (l.toIn.salt q)) :=
    funext fun k => by
      rw [Link.layer_eq, Link.layer_eq]
      exact Link.layerOf_congr _ _ (fun q hq => Link.leaves_getElem! _ _ _ hq) k
  rw [hlf] at hlayers
  have hc : ∀ c : Commit, c = ⟨c.rows, c.salts, c.layers, c.cap⟩ := fun _ => rfl
  unfold commit0
  rw [hc (commitRows _ _ _), hly, hcp]
  unfold commitOf
  rw [hlayers, Array.range_succ (n := l.lv.d - l.lv.cap), Array.append_singleton, Array.map_push, Array.back?_push, Option.getD_some, hrows]
  rfl

end Flock.Firewall.Finish
