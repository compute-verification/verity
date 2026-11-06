import Proofs.Flock.Claims.Recursion.CallsTop
import Proofs.Flock.Claims.Recursion.LastCallsR

/-!
# A decision's SHA-512 calls on every run

`decisionAtS` digests the proofs' files, then runs the plain verdict at the reader's record `jAt` (`decisionAtS_log`),
whose oracle log is its logged run's (`verifyJ_log`). So on every run, accepting or not, it calls SHA-512 at most once a
file (`filesDg_calls_le`) and then `cl_verifyJ_rec`'s count at `jAt`'s streams (`decisionAtS_calls_le`).
-/

namespace FlockDeploy.KS

open Flock FlockSoundness.Refine

/-- **The files' digests over `OC` call at most once a file.** -/
theorem filesDg_calls_le (H : Claims.Msg → Claims.Msg) (spec : SessionSpec) (xs : Array ByteArray) :
    ((xs.mapM (spec.dgWith Claims.Hashes.oc)).run.eval H).2.length ≤ xs.size := by
  have hs := ((Claims.ocLog H).mapM_array xs
    fun b => With.dgWith_sim (Claims.ocLog H) (Claims.ocLog_hashes H) spec b) #[]
  have hc := calcCL.mapM_const (c := 1) (fun b => cl_dg (Claims.hOf H) spec b) xs #[]
  have e := congrArg List.length hs.2
  rw [List.length_append, List.length_map, Array.length_toList, Array.length_toList] at e
  have h0 : (#[] : Array ByteArray).size = 0 := rfl
  omega

/-- **A decision's SHA-512 calls on every run**: one per proof file, then the verdict's at the reader's record, whose
streams are `sj`, each of at most `R` rounds. -/
theorem decisionAtS_calls_le (H : Claims.Msg → Claims.Msg) (s : Ins) (spec : SessionSpec) (sts : Array Stmt)
    (x : Claims.Msg) (link : List ByteArray × List ByteArray) (c₁ : Coin) (mids : List (Claims.Msg × Coin))
    (f : FinalP) (sj : Array Lean.Json) (R : ℕ) (hsj : Record.arr (jAt H spec sts c₁ mids f) "streams" = .ok sj)
    (hR : ∀ s ∈ sj, ∀ rj, Record.arr s "rounds" = .ok rj → rj.size ≤ R) :
    ((decisionAtS s spec sts x link c₁ mids f).run.eval H).2.length ≤
      f.ps.size + (2 * spec.tables.size + 2 + sj.size * (2 * R) + spec.repStreams.size +
        (spec.repStreams.size + (tabsOf sts).size * (2 * (2 ^ 16 + R)))) := by
  rw [decisionAtS_log, List.length_append]
  have h1 := filesDg_calls_le H spec (filesAt (tabsOf sts) (pointsOf spec c₁) f.ps)
  have hsz : (filesAt (tabsOf sts) (pointsOf spec c₁) f.ps).size = f.ps.size := by
    simp only [filesAt, Array.size_mapIdx]
  have h2 := congrArg List.length (verifyJ_log H spec (tabsOf sts) (jAt H spec sts c₁ mids f)
    (filesAt (tabsOf sts) (pointsOf spec c₁) f.ps)).1
  rw [List.length_map, Array.length_toList] at h2
  have h3 := cl_verifyJ_rec (Claims.hOf H) id spec (tabsOf sts) (jAt H spec sts c₁ mids f)
    (filesAt (tabsOf sts) (pointsOf spec c₁) f.ps) sj R hsj hR #[]
  have h0 : (#[] : Array ByteArray).size = 0 := rfl
  omega

end FlockDeploy.KS
