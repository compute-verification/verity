import Proofs.Flock.Claims.Recursion.VStarAccept
import Proofs.Flock.Claims.Recursion.VStarFitHeld
import Proofs.Flock.Claims.Recursion.SchedDense
import Proofs.Flock.C3.Reps.AtList

/-!
# V\*'s parts at the dense form are levels, algebra parts and caps (`partAt_dense`)

At a statement the reader reads at the dense form (`Form.Dense`: no sparse phase), the verifier's own inner shape names
no RecSparse shape, so candidate 3's list is empty (`candidate3_ok`: none without a `sparse` entry, and R without one is
refused), and V\*'s list is its levels (`.recOpen`), its algebra parts (`.inner`) and its caps (`.cap`)
(`getElem?_list`). No part is `.sparse`, `.recOpenStacked` or `.opening`, the statements `partIn` has no builder for.

`hok_dense` is `hok_of_fits` with its premise split by those three kinds: a level's checks (`LevelInFits`), an algebra
part's (`AlgInFits`) and a cap's (`capIn` accepting).
-/

namespace RecDeploy.VStarBuild

open _root_.Flock
open Claims (Msg)

/-- **A dense statement's inner shape has no sparse phase**: the form's `sparse` is the verifier's own inner shape's. -/
theorem sparse_none_of_dense {x : Msg} {st : Claims.Recursion.Stmt} (hx : parse x = some st)
    (hf : Claims.Recursion.Form.Dense st.form) : ∃ r, readX x = some r ∧ r.shape.sparse = none := by
  obtain ⟨r, hr, -, rfl⟩ := parse_read hx
  refine ⟨r, hr, ?_⟩
  have h2 : (r.shape.sparse.map fun sp => (sp.nu, sp.ell, sp.sLog, sp.tLog)) = none := hf.2
  cases h : r.shape.sparse with
  | none => rfl
  | some sp => rw [h] at h2; cases h2

/-- **Every part of a dense statement is a level, an algebra part or a cap**: candidate 3's list is empty without a
sparse phase, so V\*'s list is its levels, its algebra parts and its caps. -/
theorem partAt_dense {x : Msg} {st : Claims.Recursion.Stmt} (hx : parse x = some st)
    (hf : Claims.Recursion.Form.Dense st.form) {s : ℕ} {sv : _root_.VStar.Stmt} {c : Circuit}
    (hp : partAt x s = some (sv, c)) :
    (∃ l L H n, sv = .recOpen l L H n) ∨ (∃ j nm p, sv = .inner j nm p) ∨ ∃ l N, sv = .cap l N := by
  obtain ⟨r, hr, hsp⟩ := sparse_none_of_dense hx hf
  obtain ⟨ss, hss, -⟩ := readX_parts hr
  have hsv : ss[s]? = some sv := by
    have h := (partAt_some hr hp).1
    rw [hss] at h
    exact h
  obtain ⟨sch, ps, C, Ks, -, hC, hK, hget⟩ := FlockC3.Reps.getElem?_list hss
  rw [hget s, (FlockC3.VStarList.candidate3_ok hC).1 hsp] at hsv
  split_ifs at hsv
  · exact .inl ⟨_, _, _, _, (Option.some.inj hsv).symm⟩
  · exact .inr (.inl ⟨_, _, _, (Option.some.inj hsv).symm⟩)
  · rw [Array.empty_append] at hsv
    exact .inr (.inr (hK _ (Array.mem_iff_getElem?.2 ⟨_, hsv⟩)))

/-- **The stager accepts at every part of a dense statement** where it accepts at each of its kinds: a level's checks
hold (`LevelInFits`), an algebra part's (`AlgInFits`), and `capIn` accepts at a cap. -/
theorem hok_dense {I : InnerRec} {x : Msg} {w : Claims.Flock.Wit} {ρ : I.Tape} {hist : List (Msg × Msg)}
    {st : Claims.Recursion.Stmt} {s : ℕ} (hx : parse x = some st) (hf : Claims.Recursion.Form.Dense st.form)
    (hL : ∀ l L H n c src, partAt x s = some (.recOpen l L H n, c) → srcOf I x w ρ hist = some src →
      LevelInFits src l L H n)
    (hA : ∀ j nm p c src, partAt x s = some (.inner j nm p, c) → srcOf I x w ρ hist = some src →
      AlgInFits src p.S.ports)
    (hC : ∀ l N c src, partAt x s = some (.cap l N, c) → srcOf I x w ρ hist = some src →
      ∃ pin, _root_.VStar.In.capIn src l N = .ok pin) :
    ∀ sv c src, partAt x s = some (sv, c) → srcOf I x w ρ hist = some src →
      ∃ pin, _root_.VStar.In.partIn src (progOf c) sv = .ok pin := by
  refine hok_of_fits fun sv c src hp hs => ?_
  rcases partAt_dense hx hf hp with ⟨l, L, H, n, rfl⟩ | ⟨j, nm, p, rfl⟩ | ⟨l, N, rfl⟩
  · exact hL l L H n c src hp hs
  · exact hA j nm p c src hp hs
  · exact hC l N c src hp hs

end RecDeploy.VStarBuild
