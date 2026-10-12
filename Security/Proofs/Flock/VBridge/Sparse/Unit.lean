import Flock.RecSparse

/-!
# The verifier's `rec-sparse/…` unit

`RecSparse.check` accepts a `rec-sparse/n<NU>-e<ELL>-s<S_LOG>-t<T_LOG>` unit only if it is the net the verifier builds
itself at the shape its name gives (`RecSparse.unitNet`, `rec_sparse.lowering` as `Flock.RecSparse` states it), and
`HmRow.parse` calls it on every circuit (`ParseFacts.recSparse`). `check_sparse` states what it accepted.
-/

namespace FlockVBridge.Sparse

open Flock

/-- `params` gives `sh` only for the name `unitName sh`. -/
theorem unitName_params {name : String} {sh : RecSparse.Shape} (h : RecSparse.params name = some sh) :
    RecSparse.unitName sh = name := by
  unfold RecSparse.params at h
  repeat' (first | split at h | dsimp only at h)
  all_goals first
    | (simp only [Option.some.injEq] at h; subst h; assumption)
    | cases h

/-- **What `RecSparse.check` accepted**: a unit whose name gives the shape `sh` is at a valid shape, has the shape's
input and output bits, and is the net the verifier builds at `sh`. -/
theorem check_sparse {c : Circuit} (h : RecSparse.check c = .ok ()) {sh : RecSparse.Shape}
    (hp : RecSparse.params c.unit.name = some sh) :
    sh.valid = true ∧ RecOpen.inBits c.unit = 16 * RecSparse.inWords sh ∧
      RecOpen.outBits c.unit = 16 * RecSparse.outWords sh ∧ RecSparse.unitNet sh = .ok c.unit := by
  unfold RecSparse.check at h
  rw [hp] at h
  dsimp only at h
  split at h
  · cases h
  · rename_i hv
    split at h
    · cases h
    · split at h
      · cases h
      · rename_i hb
        split at h
        · cases h
        · rename_i g hg
          split at h
          · rename_i hcg
            simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, not_or, Decidable.not_not] at hb
            exact ⟨by simpa using hv, hb.1, hb.2, by rw [hcg]; exact hg⟩
          · cases h

end FlockVBridge.Sparse
