module

public import Proofs.Flock.Claims.Sampling.Deployed
public import Proofs.Flock.Claims.Sampling.Leaf
public import Proofs.Flock.Claims.Sampling.Path

/-!
# The deployed commitment at ports that read the roots

`SamplingDeploy.commit c` checks an opening's path at the port of `c.own` whose root is the one the opening is checked
under (`regPath`), so an opening passes only under a root that one of `c.own`'s ports holds (`opens_own`). A code fixes
`c.own` before its run, so roots that arrive with the run (a statement's roots, an interior registration's) open
nothing under it.

`commitAt c own` is `c`'s commitment with the ports a function of the roots and the position: an opening at `o.pos`
checked under `roots` is checked at the port of `own roots o.pos` whose root it is. The position lets one code check a
statement's roots and an interior registration's each at its own ports' shapes, since in a fixed partition a position's
gate is a boundary gate or an interior one. `commit c` is `commitAt c` at the constant ports `c.own`
(`commit_eq_commitAt`). `commitAt` binds at every hash (`commitAt_binds`): `Binds` compares two openings at one
position under one root list, where the ports are fixed.

`rooted shapes roots` is `shapes` with their roots read off the list, by index. It changes no port's leaves, so the
depth is the shapes' (`depth_rooted`, `commitAt_rooted_calls_le`). An opening at port `k` passes at `rooted` ports where
its leaf climbs to the list's `k`th root at port `k`'s shape and no earlier entry is that root (`opens_rooted`); a list
without repeats has no such entry (`before_of_nodup`).
-/

@[expose] public section

namespace SamplingDeploy

open Claims Claims.Sampling
open Properties.NetworkCertifier.Bytes (bs)

/-- **An opening passes the deployed commitment only under a root of the code's own ports**: `regPath` finds its port
by the root. -/
theorem opens_own {c : Code} {H : Msg → Msg} {roots : List Msg} {o : Opening Bool}
    (h : (commit c).Opens H roots o = true) :
    ∃ ri, c.at_ roots o.pos = some ri ∧ ∃ p ∈ c.own, bs p.root = ri.1 := by
  unfold CommitmentScheme.Opens SamplingDeploy.commit at h
  simp only [pathCommit] at h
  cases ha : c.at_ roots o.pos with
  | none => simp [ha, OC.eval] at h
  | some ri =>
    refine ⟨ri, rfl, ?_⟩
    simp only [ha, ocBind_eval] at h
    cases hl : ((hm96Leaf (row c.place c.bits) c.salt o).eval H).1 with
    | none => simp [hl, OC.eval] at h
    | some l =>
      simp only [hl] at h
      unfold regPath at h
      cases hf : c.own.find? fun p => decide (bs p.root = ri.1) with
      | none => simp [hf, OC.eval] at h
      | some p => exact ⟨p, Array.mem_of_find?_eq_some hf, by simpa using Array.find?_some hf⟩

/-- **The deployed commitment at ports that read the roots**: `c`'s leaves and paths, with an opening at `o.pos` checked
under `roots` at the port of `own roots o.pos` whose root it is. -/
def commitAt (c : Code) (own : List Msg → ℕ → Array _root_.Flock.Registered.Port) : CommitmentScheme Bool where
  opens roots o :=
    (pathCommit c.at_ (hm96Leaf (row c.place c.bits) c.salt) c.π (regPath (own roots o.pos)) c.layer).opens roots o
  layer := c.layer

/-- **The deployed commitment is `commitAt` at its own ports**, whatever the roots and the position. -/
theorem commit_eq_commitAt (c : Code) : commit c = commitAt c fun _ _ => c.own := rfl

/-- **`commitAt` at a root list and a position is `commit` at the ports there.** -/
theorem commitAt_opens (c : Code) (own : List Msg → ℕ → Array _root_.Flock.Registered.Port) (roots : List Msg)
    (o : Opening Bool) :
    (commitAt c own).opens roots o = (commit { c with own := own roots o.pos }).opens roots o := rfl

/-- **`commitAt` binds, at every hash**: two openings at one position under one root list are checked at the same
ports, where hm96 leaves of rows that place the value (`row_places`) under the registered path check bind
(`pathCommit_binds`). -/
theorem commitAt_binds (H : Msg → Msg) (c : Code) (own : List Msg → ℕ → Array _root_.Flock.Registered.Port) :
    (commitAt c own).Binds H := by
  intro roots o o' hp
  have e : ∀ o'' : Opening Bool, o''.pos = o.pos →
      (commitAt c own).opens roots o'' = (commit { c with own := own roots o.pos }).opens roots o'' := by
    intro o'' h
    rw [commitAt_opens, h]
  unfold CommitmentScheme.Opens
  rw [e o rfl, e o' hp.symm]
  exact pathCommit_binds (at_ := c.at_) (π := c.π) (layer := c.layer)
    (hm96Leaf_binds H (row_places c.place c.bits) c.salt) (regPath_binds H (own roots o.pos)) roots o o' hp

/-- **One opening's calls under `commitAt`**: `opens_calls_le` at the ports of its root list and position. -/
theorem commitAt_calls_le (H : Msg → Msg) (c : Code) (own : List Msg → ℕ → Array _root_.Flock.Registered.Port)
    (roots : List Msg) (o : Opening Bool) :
    (((commitAt c own).opens roots o).eval H).2.length ≤ 4 + depth (own roots o.pos) := by
  rw [commitAt_opens]
  exact opens_calls_le H { c with own := own roots o.pos } roots o

/-- **An opening passes `commitAt`** where `at_` checks it under root `r`, the port `regPath` finds for `r` among the
ports of its root list and position is `p`, and its leaf passes `p`'s path check there. -/
theorem opens_commitAt {c : Code} {own : List Msg → ℕ → Array _root_.Flock.Registered.Port} {H : Msg → Msg}
    {roots : List Msg} {o : Opening Bool} {idx : ℕ} {r l : Msg} {p : _root_.Flock.Registered.Port}
    (hat : c.at_ roots o.pos = some (r, idx))
    (hfind : (own roots o.pos).find? (fun q => decide (bs q.root = r)) = some p)
    (hleaf : ((hm96Leaf (row c.place c.bits) c.salt o).eval H).1 = some l)
    (hpath : ((regPathAt p r idx l (c.π o)).eval H).1 = true) :
    (commitAt c own).Opens H roots o = true := by
  unfold CommitmentScheme.Opens commitAt
  simp only [pathCommit, hat, ocBind_eval, hleaf]
  unfold regPath
  rw [hfind]
  exact hpath

/-- **Ports at a root list**: each port of `shapes` with its root the list's entry at its index, and a port past the
list as it is. -/
def rooted (shapes : Array _root_.Flock.Registered.Port) (roots : List Msg) : Array _root_.Flock.Registered.Port :=
  shapes.mapIdx fun k p => match roots[k]? with
    | some r => { p with root := bytes r }
    | none => p

/-- **`rooted` keeps each port's leaves.** -/
theorem leaves_rooted (shapes : Array _root_.Flock.Registered.Port) (roots : List Msg) :
    (rooted shapes roots).toList.map (·.leaves) = shapes.toList.map (·.leaves) := by
  apply List.ext_getElem
  · simp [rooted]
  · intro i h₁ h₂
    simp only [List.getElem_map, Array.getElem_toList, rooted, Array.getElem_mapIdx]
    cases roots[i]? <;> rfl

/-- **`rooted` keeps the depth**: it sets roots only. -/
theorem depth_rooted (shapes : Array _root_.Flock.Registered.Port) (roots : List Msg) :
    depth (rooted shapes roots) = depth shapes := by
  have h : ∀ own : Array _root_.Flock.Registered.Port,
      depth own = (own.toList.map (·.leaves)).foldr (fun n m => max (_root_.Flock.log2ceil n) m) 0 := by
    intro own
    rw [List.foldr_map]
    rfl
  rw [h, h, leaves_rooted]

/-- **The port `regPath` finds at `rooted` ports** for the list's `k`th root, where no earlier entry of the list is
that root: port `k`, with that root. -/
theorem find?_rooted {shapes : Array _root_.Flock.Registered.Port} {roots : List Msg} {k : ℕ} {r : Msg}
    (hk : k < shapes.size) (hr : roots[k]? = some r) (hfirst : ∀ j < k, roots[j]? ≠ some r) :
    (rooted shapes roots).find? (fun q => decide (bs q.root = r)) = some { shapes[k] with root := bytes r } := by
  have hkl : k < roots.length := (List.getElem?_eq_some_iff.1 hr).1
  rw [Array.find?_eq_some_iff_getElem]
  refine ⟨decide_eq_true (bs_bytes r), k, by simpa [rooted] using hk, by simp [rooted, hr], fun j hj => ?_⟩
  have hjl : j < roots.length := hj.trans hkl
  have hne : roots[j] ≠ r := fun h => hfirst j hj (List.getElem?_eq_some_iff.2 ⟨hjl, h⟩)
  simp [rooted, List.getElem?_eq_some_iff.2 ⟨hjl, rfl⟩, bs_bytes, hne]

/-- **A list without repeats has no earlier entry equal to its `k`th**, `opens_rooted`'s order premise. -/
theorem before_of_nodup {roots : List Msg} {k : ℕ} {r : Msg} (hnd : roots.Nodup) (hr : roots[k]? = some r) :
    ∀ j < k, roots[j]? ≠ some r := fun _ hj h =>
  Nat.ne_of_lt hj ((hnd.getElem?_inj (hj.trans (List.getElem?_eq_some_iff.1 hr).1)).1 (h.trans hr.symm))

/-- **An opening at port `k` passes at `rooted` ports** where `at_` checks it under the list's `k`th root, no earlier
entry of the list is that root, and its leaf passes the path check at port `k` of its position's shapes. -/
theorem opens_rooted {c : Code} {shapes : ℕ → Array _root_.Flock.Registered.Port} {H : Msg → Msg}
    {roots : List Msg} {o : Opening Bool} {k idx : ℕ} {r l : Msg}
    (hat : c.at_ roots o.pos = some (r, idx)) (hk : k < (shapes o.pos).size) (hr : roots[k]? = some r)
    (hfirst : ∀ j < k, roots[j]? ≠ some r)
    (hleaf : ((hm96Leaf (row c.place c.bits) c.salt o).eval H).1 = some l)
    (hpath : ((regPathAt (shapes o.pos)[k] r idx l (c.π o)).eval H).1 = true) :
    (commitAt c fun roots g => rooted (shapes g) roots).Opens H roots o = true :=
  opens_commitAt hat (find?_rooted hk hr hfirst) hleaf hpath

/-- **One opening's calls at `rooted` ports** are bounded by the depth of its position's shapes. -/
theorem commitAt_rooted_calls_le (H : Msg → Msg) (c : Code) (shapes : ℕ → Array _root_.Flock.Registered.Port)
    (roots : List Msg) (o : Opening Bool) :
    (((commitAt c fun roots g => rooted (shapes g) roots).opens roots o).eval H).2.length ≤
      4 + depth (shapes o.pos) := by
  simpa only [depth_rooted] using commitAt_calls_le H c (fun roots g => rooted (shapes g) roots) roots o

end SamplingDeploy
