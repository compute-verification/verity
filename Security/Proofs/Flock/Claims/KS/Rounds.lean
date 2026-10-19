import Proofs.Flock.Claims.KS.Refit
import Proofs.Flock.Claims.DeployK

/-!
# The deployed plain verifier's fork rounds

`oldEsc_fitsFork_plain` takes the old fork terms' rounds as a map `j` from a rep and a level to the deployed verifier's
rounds, no two sharing one. Here `j r ℓ` is the round whose message carries rep `r`'s level-`ℓ + 1` cap at table 0, the
commitment after which `advRSL`'s collision finder continues the prover twice. After the link message (round 0), rep
`r`'s stream starts `r · R` rounds in (`R = repRounds`; the streams in `spec.repStreams` order, table 0's two first),
its Ligerito after zerocheck's, lincheck's and the ring switch's rounds (`preLig`, `verifyRepWith_rounds`), and level
`n`'s cap is in Ligerito's round `capRound k0 n` (`ligeritoWith_rounds`): level 1's after level 0's out-of-domain point,
its coin and the `k0` lane challenges, level `n ≥ 2`'s after the first four of level `n − 1`'s eleven rounds. The last
index, `n = L`, whose weight `2^{d}·Q` is `0`, lands in the last level's seven rounds.

* `forkRound`, `forkRound_lt`: the round, before the last message of `V false`.
* `plainJ`, `plainJ_inj`: the map, no two of whose rounds are one; `oldEsc_fitsFork_plainJ`, the old escapes at these
  rounds fit the plain weight.
-/

namespace FlockDeploy.KS

open Flock

/-- Ligerito's round whose message carries level `n`'s cap, `1 ≤ n`, counted from Ligerito's first round. -/
def capRound (k0 n : ℕ) : ℕ := if n ≤ 1 then k0 + 2 else k0 + 13 + 11 * (n - 2)

theorem capRound_lt {k0 L n : ℕ} (hL : 2 ≤ L) (hn : n ≤ L) : capRound k0 n < k0 + 5 + 11 * (L - 1) := by
  unfold capRound
  split <;> omega

theorem capRound_inj {k0 n n' : ℕ} (h1 : 1 ≤ n) (h1' : 1 ≤ n') (h : capRound k0 n = capRound k0 n') : n = n' := by
  unfold capRound at h
  split at h <;> split at h <;> omega

/-- A plain rep's rounds before Ligerito: zerocheck's, lincheck's and the ring switch's (`repRounds`). -/
def preLig (spec : SessionSpec) (st : Stmt) : ℕ :=
  let claims := 2 + spec.points * st.regions.size
  (3 + (st.m - 6)) + (3 + (st.c.kLog - 6)) + (claims + 1)

/-- **The plain fork round** of rep `r` and level `ℓ + 1` at table 0, a round of `V false`: after the link message,
rep `r`'s `r · R` earlier rounds, its rounds before Ligerito and Ligerito's `capRound`. -/
def forkRound (spec : SessionSpec) (st : Stmt) (k0 r ℓ : ℕ) : ℕ :=
  1 + r * repRounds false spec st + preLig spec st + capRound k0 (ℓ + 1)

theorem repRounds_plain {spec : SessionSpec} {st : Stmt} {s : Flock.Schedule} (hs : fast100 st.m = some s) :
    repRounds false spec st = preLig spec st + (s.k0 + 5 + 11 * (s.levels.size - 1)) := by
  unfold repRounds preLig
  simp only [Bool.false_eq_true, ↓reduceIte, hs]

/-- Table 0 of the statement's tables is the first of its setup's. -/
theorem setupOf_head {zk : Bool} {x : Claims.Msg} {spec : SessionSpec} {st : Stmt}
    (h0 : (tables zk x).head? = some (spec, st)) :
    ∃ s sts, setupOf zk x = some (s, spec, sts) ∧ sts[0]? = some st := by
  unfold tables at h0
  split at h0
  · rename_i s spec' sts hs
    rcases sts with ⟨_ | ⟨a, l⟩⟩
    · simp at h0
    · simp only [List.map_cons, List.head?_cons, Option.some.injEq, Prod.mk.injEq] at h0
      obtain ⟨rfl, rfl⟩ := h0
      exact ⟨s, _, hs, rfl⟩
  · simp at h0

/-- Table 0's two reps are among the session's recorded rounds. -/
theorem rep_le_rounds {zk : Bool} {spec : SessionSpec} {sts : Array Stmt} {st : Stmt} (h : sts[0]? = some st) :
    2 * repRounds zk spec st ≤ rounds zk spec sts := by
  rw [rounds_sum]
  rcases sts with ⟨_ | ⟨a, l⟩⟩
  · simp at h
  · simp only [List.getElem?_toArray, List.getElem?_cons_zero, Option.some.injEq] at h
    subst h
    rw [List.size_toArray, List.length_cons, List.range_succ_eq_map, List.map_cons, List.sum_cons]
    simp

/-- The Accounting schedule at an `m` is the executable's, read as the model's (`fast100_schedOf`). -/
theorem exec_sched {m : ℕ} {sch : FlockSoundness.Accounting.Schedule}
    (hsch : FlockSoundness.Accounting.fast100 m = some sch) :
    ∃ s, fast100 m = some s ∧ sch = FlockSoundness.Refine.schedOf s := by
  have h1 := FlockSoundness.Refine.fast100_schedOf m
  rw [hsch] at h1
  obtain ⟨s, hs, e⟩ := Option.map_eq_some_iff.1 h1
  exact ⟨s, hs, e.symm⟩

/-- **The plain fork rounds are rounds of `V false`**: at a statement whose table 0 has `fast100` schedule `sch`, rep
`r < 2`'s level-`ℓ + 1` cap round, `ℓ < L`, is before the last message. -/
theorem forkRound_lt {x : Claims.Msg} {spec : SessionSpec} {st : Stmt} (h0 : (tables false x).head? = some (spec, st))
    {sch : FlockSoundness.Accounting.Schedule} (hsch : FlockSoundness.Accounting.fast100 st.m = some sch)
    {r ℓ : ℕ} (hr : r < 2) (hℓ : ℓ < sch.levels.length) : forkRound spec st sch.k0 r ℓ < (V false).k x := by
  obtain ⟨s, sts, hs, hst⟩ := setupOf_head h0
  obtain ⟨sF, hF, rfl⟩ := exec_sched hsch
  have h2 := (FlockSoundness.Refine.fast100_dims _ _ hF).1.2.1
  simp only [FlockSoundness.Refine.schedOf, List.length_map, Array.length_toList] at hℓ ⊢
  have hR := repRounds_plain (spec := spec) hF
  have hle := rep_le_rounds (zk := false) (spec := spec) hst
  have hlt := capRound_lt (k0 := sF.k0) h2 (show ℓ + 1 ≤ sF.levels.size by omega)
  have hk : (V false).k x = 2 + rounds false spec sts := by simp only [V, hs]
  unfold forkRound
  rw [hk]
  interval_cases r <;> omega

/-- **The plain fork rounds** as `oldEsc_fitsFork_plain`'s map: rep `r`'s level-`ℓ + 1` cap round at table 0. -/
def plainJ {x : Claims.Msg} {spec : SessionSpec} {st : Stmt} (h0 : (tables false x).head? = some (spec, st))
    {sch : FlockSoundness.Accounting.Schedule} (hsch : FlockSoundness.Accounting.fast100 st.m = some sch)
    (r : Fin 2) (ℓ : Fin sch.levels.length) : Fin ((V false).k x) :=
  ⟨forkRound spec st sch.k0 r.val ℓ.val, forkRound_lt h0 hsch r.isLt ℓ.isLt⟩

/-- **No two plain fork rounds are one.** -/
theorem plainJ_inj {x : Claims.Msg} {spec : SessionSpec} {st : Stmt} (h0 : (tables false x).head? = some (spec, st))
    {sch : FlockSoundness.Accounting.Schedule} (hsch : FlockSoundness.Accounting.fast100 st.m = some sch)
    (r : Fin 2) (ℓ : Fin sch.levels.length) (r' : Fin 2) (ℓ' : Fin sch.levels.length)
    (h : plainJ h0 hsch r ℓ = plainJ h0 hsch r' ℓ') : r = r' ∧ ℓ = ℓ' := by
  obtain ⟨sF, hF, rfl⟩ := exec_sched hsch
  have h2 := (FlockSoundness.Refine.fast100_dims _ _ hF).1.2.1
  have hℓ := ℓ.isLt
  have hℓ' := ℓ'.isLt
  simp only [FlockSoundness.Refine.schedOf, List.length_map, Array.length_toList] at hℓ hℓ'
  have hlt := capRound_lt (k0 := sF.k0) h2 (show ℓ.val + 1 ≤ sF.levels.size by omega)
  have hlt' := capRound_lt (k0 := sF.k0) h2 (show ℓ'.val + 1 ≤ sF.levels.size by omega)
  have hR := repRounds_plain (spec := spec) hF
  have hv := congrArg Fin.val h
  simp only [plainJ, forkRound, FlockSoundness.Refine.schedOf] at hv
  have hrr : r = r' := by
    apply Fin.ext
    have hr := r.isLt
    have hr' := r'.isLt
    rcases (show r.val = 0 ∨ r.val = 1 by omega) with e | e <;>
      rcases (show r'.val = 0 ∨ r'.val = 1 by omega) with e' | e' <;> rw [e, e'] at hv <;> omega
  subst hrr
  refine ⟨rfl, Fin.ext ?_⟩
  have := capRound_inj (k0 := sF.k0) (by omega) (by omega) (by omega : capRound sF.k0 (ℓ.val + 1) =
    capRound sF.k0 (ℓ'.val + 1))
  omega

/-- **The old escapes fit the plain weight at the deployed fork rounds**: `oldEsc_fitsFork_plain` at `plainJ`, rep
`r`'s level-`ℓ + 1` cap round at table 0. -/
theorem oldEsc_fitsFork_plainJ (rel : Readable.Parsed → Readable.Relation) {Ω : Type} (P : Claims.Prover Ω)
    (x : Claims.Msg) {spec : SessionSpec} {st : Stmt} (h0 : (tables false x).head? = some (spec, st))
    {sch : FlockSoundness.Accounting.Schedule} (hsch : FlockSoundness.Accounting.fast100 st.m = some sch)
    (ev : Fin 2 → Fin sch.levels.length → Ω → (Fin ((deployPlain rel).V.k x) → Coin) →
      (Fin ((deployPlain rel).V.k x) → Coin) → Prop)
    (ev₁ : Ω → (Fin ((deployPlain rel).V.k x) → Coin) → Prop)
    (tied : ∀ r ℓ ω s s', ev r ℓ ω s s' → Claims.TiedColl sha512
      ((Claims.run sha512 P (deployPlain rel).V x ω s).vLog ++
        (Claims.run sha512 P (deployPlain rel).V x ω (Claims.forkCoins (plainJ h0 hsch r ℓ) s s')).vLog))
    (tied₁ : ∀ ω s, ev₁ ω s → Claims.TiedColl sha512 (Claims.run sha512 P (deployPlain rel).V x ω s).vLog) :
    (Claims.Flock.oldEsc sha512 (deployPlain rel).V P x sch (plainJ h0 hsch) ev ev₁ tied tied₁).FitsFork
      ((deployPlain rel).weight x) ((deployPlain rel).k x) :=
  oldEsc_fitsFork_plain rel P x h0 hsch (plainJ h0 hsch) (plainJ_inj h0 hsch) ev ev₁ tied tied₁

end FlockDeploy.KS
