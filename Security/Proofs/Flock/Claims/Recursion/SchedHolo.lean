import Proofs.Flock.Claims.Recursion.DeployLayout
import Proofs.Flock.Claims.Recursion.InnerAccepts

/-!
# `SchedMatches` fails at the holo form

`SchedMatches` (`DeployLayout.lean`) asks that the deployed plain verifier take `2 + 2R` rounds at the inner statement,
`R` the rounds of the verifier's own inner shape's rep schedule (`repSched`). At a holo shape that schedule is
`RecVStar.holoRepRounds`, which is `RecVStar.repRounds` with z_partial's round split in two, one round longer
(`holoRepRounds_size`). The plain verifier's count, `FlockDeploy.V false`'s `2 + 2 · repRounds`, has no holo term: where
it is `2 + 2R` at the plain schedule, as at a dense shape, it is two short of the holo schedule's, so `SchedMatches`
fails at every such holo statement (`not_schedMatches_holo`).
-/

namespace RecDeploy

open Flock
open Claims (Msg)
open FlockDeploy (Coin)

/-- **The holo schedule is one round longer than the plain one**: `RecVStar.holoRepRounds` is
`RecVStar.repRounds` with z_partial's round `m + k_log − 7` replaced by two. -/
theorem holoRepRounds_size {m kLog claims : ℕ} {rs : Array RecVStar.SchedRound}
    (h : RecVStar.holoRepRounds m kLog claims = some rs) :
    ∃ rs₀, RecVStar.repRounds m kLog claims = some rs₀ ∧ rs.size = rs₀.size + 1 := by
  unfold RecVStar.holoRepRounds at h
  cases h0 : RecVStar.repRounds m kLog claims with
  | none =>
    rw [h0] at h
    first
    | cases h
    | simp at h
  | some out =>
    rw [h0] at h
    refine ⟨out, rfl, ?_⟩
    cases hk : out[m + kLog - 7]? with
    | none => simp [hk] at h
    | some v =>
      by_cases hv : v = ((1, #[], 0, true) : RecVStar.SchedRound)
      · subst hv
        have hlt : m + kLog - 7 < out.size := by
          first
          | exact (Array.getElem?_eq_some_iff.1 hk).1
          | (obtain ⟨h1, -⟩ := Array.getElem?_eq_some_iff.1 hk; exact h1)
          | (by_contra hc; rw [Array.getElem?_eq_none (by omega)] at hk; cases hk)
        simp [hk] at h
        subst h
        first
        | (simp only [Array.size_append, Array.size_extract, List.size_toArray, List.length_cons, List.length_nil]
           omega)
        | (simp; omega)
        | simp
      · first
        | simp [hk, hv] at h
        | (have hne : (some v != some ((1, #[], 0, true) : RecVStar.SchedRound)) = true := by
             simpa [bne_iff_ne, ne_eq] using hv
           simp [hk, hne] at h)

/-- **`SchedMatches` fails at the holo form** at any holo statement the reader reads where the deployed plain
verifier's rounds are `2 + 2R` at the plain schedule `R` (`hk`): the holo schedule is one round longer
(`holoRepRounds_size`), so `2 + 2R` is not its count. -/
theorem not_schedMatches_holo {x : Msg} {st : Claims.Recursion.CompiledStatement} {r : XR} {rs₀ : Array RecVStar.SchedRound}
    (hst : parse x = some st) (hf : Claims.Recursion.Form.Holo st.form) (hr : readX x = some r)
    (hho : r.shape.holo = true) (h₀ : RecVStar.repRounds r.shape.m r.shape.kLog r.shape.claims = some rs₀)
    (hk : (FlockDeploy.V false).k (xIOf x) = 2 + 2 * rs₀.size) :
    ¬ SchedMatches Claims.Recursion.Form.Holo := by
  intro hM
  obtain ⟨r', rs, hr', hrs, hk'⟩ := hM x st hst hf
  rw [hr] at hr'
  cases hr'
  unfold repSched at hrs
  first
  | simp only [hho, ↓reduceIte] at hrs
  | rw [hho, if_pos rfl] at hrs
  obtain ⟨rs₁, h₁, hsz⟩ := holoRepRounds_size hrs
  rw [h₀] at h₁
  cases h₁
  omega

end RecDeploy
