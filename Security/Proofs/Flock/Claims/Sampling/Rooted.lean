module

public import Proofs.Flock.Claims.Sampling.Deployed
public import Proofs.Flock.Claims.Sampling.Leaf
public import Proofs.Flock.Claims.Sampling.Path

/-!
# Ports that read the run's roots

`SamplingDeploy.commit c` checks an opening at `o.pos` under `roots` at the port of `c.ownAt roots o.pos` whose root it
is (`regPath`), so an opening passes only under a root that one of those ports holds (`opens_ownAt`). At the default
ports, `c.own` at every position, roots that arrive with the run (a statement's roots, an interior registration's) open
nothing.

`rooted shapes roots` is `shapes` with their roots read off the list, by index. It changes no port's leaves, so the
depth is the shapes' (`depth_rooted`), and a code reading `rooted` ports meets `Code.ownAt_depth` where its shapes do
(`ownAt_depth_rooted`). At a code whose positions read `rooted` ports (`OwnRooted`), an opening at port `k` passes where
its leaf climbs to the list's `k`th root at port `k`'s shape and no earlier entry is that root (`opens_rooted`); a list
without repeats has no such entry (`before_of_nodup`). The position lets one code check a statement's roots and an
interior registration's each at its own ports' shapes, since in a fixed partition a position's gate is a boundary gate
or an interior one.
-/

@[expose] public section

namespace SamplingDeploy

open Claims Claims.Sampling
open Properties.NetworkCertifier.Bytes (bs)

/-- **An opening passes the deployed commitment only under a root of the ports its position reads**: `regPath` finds
its port by the root. -/
theorem opens_ownAt {c : Code} {H : Msg → Msg} {roots : List Msg} {o : Opening Bool}
    (h : (commit c).Opens H roots o = true) :
    ∃ ri, c.at_ roots o.pos = some ri ∧ ∃ p ∈ c.ownAt roots o.pos, bs p.root = ri.1 := by
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
      cases hf : (c.ownAt roots o.pos).find? fun p => decide (bs p.root = ri.1) with
      | none => simp [hf, OC.eval] at h
      | some p => exact ⟨p, Array.mem_of_find?_eq_some hf, by simpa using Array.find?_some hf⟩

/-- **An opening passes the deployed commitment** where `at_` checks it under root `r`, the port `regPath` finds for
`r` among the ports its position reads under its roots is `p`, and its leaf passes `p`'s path check there. -/
theorem opens_commit {c : Code} {H : Msg → Msg} {roots : List Msg} {o : Opening Bool} {idx : ℕ} {r l : Msg}
    {p : _root_.Flock.Registered.Port} (hat : c.at_ roots o.pos = some (r, idx))
    (hfind : (c.ownAt roots o.pos).find? (fun q => decide (bs q.root = r)) = some p)
    (hleaf : ((hm96Leaf (row c.place c.bits) c.salt o).eval H).1 = some l)
    (hpath : ((regPathAt p r idx l (c.π o)).eval H).1 = true) :
    (commit c).Opens H roots o = true := by
  unfold CommitmentScheme.Opens commit
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

/-- **`rooted` ports are no deeper than the registered ports** where no position's shapes are: `Code.ownAt_depth` at
`rooted` ports. -/
theorem ownAt_depth_rooted {shapes : ℕ → Array _root_.Flock.Registered.Port} {own : Array _root_.Flock.Registered.Port}
    (h : ∀ g, depth (shapes g) ≤ depth own) (roots : List Msg) (g : ℕ) :
    depth (rooted (shapes g) roots) ≤ depth own := by
  rw [depth_rooted]
  exact h g

/-- **A code whose positions read `rooted` ports**: at each root list and position, the shapes at that position with
their roots read off the list. -/
def OwnRooted (c : Code) (shapes : ℕ → Array _root_.Flock.Registered.Port) : Prop :=
  ∀ roots g, c.ownAt roots g = rooted (shapes g) roots

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
    {roots : List Msg} {o : Opening Bool} {k idx : ℕ} {r l : Msg} (hown : OwnRooted c shapes)
    (hat : c.at_ roots o.pos = some (r, idx)) (hk : k < (shapes o.pos).size) (hr : roots[k]? = some r)
    (hfirst : ∀ j < k, roots[j]? ≠ some r)
    (hleaf : ((hm96Leaf (row c.place c.bits) c.salt o).eval H).1 = some l)
    (hpath : ((regPathAt (shapes o.pos)[k] r idx l (c.π o)).eval H).1 = true) :
    (commit c).Opens H roots o = true :=
  opens_commit (p := { (shapes o.pos)[k] with root := bytes r }) hat
    (by rw [hown]; exact find?_rooted hk hr hfirst) hleaf hpath

end SamplingDeploy
