import Proofs.Flock.VBridge.PartS
import Proofs.Flock.VBridge.Algebra.ToVStar
import Proofs.Flock.Soundness.ExecDraw

/-!
# The parts' residuals cover S's

The verifier cuts V*'s algebra S into parts (`RecResiduals.groups`) and pins each part's unit (`RecResiduals.check`).
When every part's residuals are zero, so is every residual of S, and `GfResiduals`' semantics on S (`residuals`, which
E's stage theorems read) returns them:

* `groups_cover`: `groups` refuses unless its parts hold each of S's residuals, so every residual is some part's.
* `residuals_zero_of_parts`: a part's residual `k` is S's residual `R[k]` (`PartS.res_partS`), at part ports reading S's
  ports where the part's index says; with `groups_cover`, every residual of S is zero.
-/

namespace FlockVBridge.Parts

open Flock FlockVBridge.OfVStar FlockVBridge.PartS
open Flock.RecResiduals (groups partS)
open Flock.RecAlgebra (residuals resValue el resValue_toV)

/-! ## Every residual is some part's -/

theorem mem_foldl_append {α : Type} (x : ℕ) :
    ∀ (gs : List (Array ℕ × α)) (acc : Array ℕ),
      x ∈ gs.foldl (fun a g => a ++ g.1) acc ↔ x ∈ acc ∨ ∃ g ∈ gs, x ∈ g.1
  | [], acc => by simp
  | g :: gs, acc => by
    rw [List.foldl_cons, mem_foldl_append x gs (acc ++ g.1)]
    simp only [Array.mem_append, List.mem_cons, exists_eq_or_imp]
    tauto

/-- **Every residual of S is in a part**: `groups` refuses unless the parts' residuals, sorted, are `0 … n - 1`. -/
theorem groups_cover {S : RecAlgebra.GfS} {sizes : Array ℕ} {gs : Array (Array ℕ × Array ℕ × Array ℕ)}
    (h : groups S sizes = .ok gs) {i : ℕ} (hi : i < S.res.size) :
    ∃ j, ∃ hj : j < gs.size, ∃ k, ∃ hk : k < gs[j].1.size, gs[j].1[k] = i := by
  unfold groups at h
  obtain ⟨gs', -, h⟩ := bind_ok h
  dsimp only at h
  split at h
  · cases h
  · rename_i hall
    cases h
    have hr : i ∈ ((gs.foldl (fun a g => a ++ g.1) #[]).qsort (· < ·)).toList := by
      rw [Bool.not_eq_true, bne_eq_false_iff_eq] at hall
      rw [hall]
      exact List.mem_range.2 hi
    rw [Array.mem_toList_iff, FlockSoundness.Audit.DrawExec.mem_qsort, ← Array.foldl_toList,
      mem_foldl_append] at hr
    rcases hr with hr | ⟨g, hg, hx⟩
    · simp at hr
    · obtain ⟨j, hj, rfl⟩ := Array.getElem_of_mem (Array.mem_toList_iff.1 hg)
      obtain ⟨k, hk, hk'⟩ := Array.getElem_of_mem hx
      exact ⟨j, hj, k, hk, hk'⟩

/-! ## S's residuals from its parts' -/

theorem vS_eq (S : RecAlgebra.GfS) : vS S = S.toV := rfl

/-- **Every residual of S is zero when every part's is**: at S's parts (`groups`), when each part's S (`partS`) has
every residual zero on the part's ports, which read S's ports `w`, `v` where the part's index says (its `w` element `k`
is `w`'s element `wIdx[k]`, its `v` element `k` is `v`'s `V[k]`), `GfResiduals` on S returns residuals that are all
zero. -/
theorem residuals_zero_of_parts {S : RecAlgebra.GfS} (hS : S.BitsOk) {sizes : Array ℕ}
    {gs : Array (Array ℕ × Array ℕ × Array ℕ)} (hg : groups S sizes = .ok gs) {w v : Array F128}
    (hw : w.size = S.nw) (hv : v.size = S.nv)
    (hparts : ∀ j (hj : j < gs.size), ∃ S', partS S sizes gs[j].1 gs[j].2.1 gs[j].2.2 = .ok S' ∧
      ∀ k (hk : k < S'.res.length),
        resVal (gfsOf S') (termVal (fun k => el w ((wIdx sizes gs[j].2.1).getD k 0))
          (fun k => el v (gs[j].2.2.getD k 0))) (resOf S'.res[k]) = 0) :
    ∃ rs, residuals S w v = .ok rs ∧ ∀ x ∈ rs, x = 0 := by
  refine ⟨S.res.map (resValue S w v), ?_, ?_⟩
  · unfold residuals
    simp [hw, hv]
  · intro x hx
    obtain ⟨i, hi, rfl⟩ := Array.getElem_of_mem hx
    rw [Array.size_map] at hi
    rw [Array.getElem_map, resValue_toV S w v hS.1 _ (hS.2 _ (Array.getElem_mem hi))]
    obtain ⟨j, hj, k, hk, hki⟩ := groups_cover hg hi
    obtain ⟨S', hS', hz⟩ := hparts j hj
    obtain ⟨hn, hres⟩ := res_partS hS' (w := el w) (v := el v)
      (w' := fun k => el w ((wIdx sizes gs[j].2.1).getD k 0)) (v' := fun k => el v (gs[j].2.2.getD k 0))
      (fun k hk => by simp only [Array.getD, hk, ↓reduceDIte]; rfl) (fun k hk => by simp only [Array.getD, hk, ↓reduceDIte]; rfl)
    have hk' : k < S'.res.length := hn ▸ hk
    have e := (hres k hk').symm.trans (hz k hk')
    rw [vS_eq] at e
    simp only [hki] at e
    rw [← e]
    congr 2
    simp [Flock.RecAlgebra.GfS.toV, List.getD_eq_getElem?_getD, hi]

end FlockVBridge.Parts
