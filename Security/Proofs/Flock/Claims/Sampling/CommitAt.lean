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

`commitAt c own` is `c`'s commitment with the ports a function of the roots: an opening checked under `roots` is checked
at the port of `own roots` whose root it is. `commit c` is `commitAt c` at the constant ports `c.own`
(`commit_eq_commitAt`), and `commitAt` binds at every hash, one root list at a time, by `commit_binds`'s argument
(`commitAt_binds`).

`rooted c.own roots` is the code's ports with their roots read off the list, by index. It changes no port's leaves, so
the depth is the code's (`depth_rooted`) and one opening's calls are bounded as `commit`'s are
(`commitAt_rooted_calls_le`).
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

/-- **The deployed commitment at ports that read the roots**: `c`'s leaves and paths, with an opening checked under
`roots` at the port of `own roots` whose root it is. -/
def commitAt (c : Code) (own : List Msg → Array _root_.Flock.Registered.Port) : CommitmentScheme Bool where
  opens roots :=
    (pathCommit c.at_ (hm96Leaf (row c.place c.bits) c.salt) c.π (regPath (own roots)) c.layer).opens roots
  layer := c.layer

/-- **The deployed commitment is `commitAt` at its own ports**, whatever the roots. -/
theorem commit_eq_commitAt (c : Code) : commit c = commitAt c fun _ => c.own := rfl

/-- **`commitAt` binds, at every hash**: under each root list, hm96 leaves of rows that place the value
(`row_places`) under the registered path check at that list's ports (`pathCommit_binds`). -/
theorem commitAt_binds (H : Msg → Msg) (c : Code) (own : List Msg → Array _root_.Flock.Registered.Port) :
    (commitAt c own).Binds H := by
  intro roots
  exact pathCommit_binds (at_ := c.at_) (π := c.π) (layer := c.layer)
    (hm96Leaf_binds H (row_places c.place c.bits) c.salt) (regPath_binds H (own roots)) roots

/-- **`commitAt` under a root list is `commit` at that list's ports.** -/
theorem commitAt_opens (c : Code) (own : List Msg → Array _root_.Flock.Registered.Port) (roots : List Msg) :
    (commitAt c own).opens roots = (commit { c with own := own roots }).opens roots := rfl

/-- **One opening's calls under `commitAt`**: `opens_calls_le` at the list's ports. -/
theorem commitAt_calls_le (H : Msg → Msg) (c : Code) (own : List Msg → Array _root_.Flock.Registered.Port)
    (roots : List Msg) (o : Opening Bool) :
    (((commitAt c own).opens roots o).eval H).2.length ≤ 4 + depth (own roots) := by
  rw [commitAt_opens]
  exact opens_calls_le H { c with own := own roots } roots o

/-- **Ports at a root list**: each port of `shapes` with its root the list's entry at its index, and a port past the
list as it is. -/
def rooted (shapes : Array _root_.Flock.Registered.Port) (roots : List Msg) : Array _root_.Flock.Registered.Port :=
  shapes.mapIdx fun k p => match roots[k]? with
    | some r => { p with root := bytes r }
    | none => p

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

/-- **One opening's calls at the code's ports with the list's roots** are bounded as `commit`'s are, by the code's
deepest port. -/
theorem commitAt_rooted_calls_le (H : Msg → Msg) (c : Code) (roots : List Msg) (o : Opening Bool) :
    (((commitAt c (rooted c.own)).opens roots o).eval H).2.length ≤ 4 + depth c.own := by
  simpa only [depth_rooted] using commitAt_calls_le H c (rooted c.own) roots o

end SamplingDeploy
