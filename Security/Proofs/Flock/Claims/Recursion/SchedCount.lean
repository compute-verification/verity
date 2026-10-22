import Proofs.Flock.Claims.Recursion.DeployLayout

/-!
# The plain schedule's size, and the plain verifier's count at it

`SchedMatches` (`DeployLayout.lean`) asks that the deployed plain verifier, `FlockDeploy.V false`, take `2 + 2R` rounds
at the inner statement, `R` the size of the verifier's own inner shape's rep schedule. At a dense shape that schedule
is `RecVStar.repRounds`. This file computes its size, `(m − 6) + (k_log − 6) + claims + k₀ + 11 · (levels − 1) + 12`
(`repRounds_closed`), which is `FlockDeploy.repRounds false`'s count of one rep's rounds at the same `m`, `k_log` and
claims (`flockRepRounds_eq`). So at an inner statement that sets up as one table whose `m`, `k_log` and claims are the
verifier's shape's, the plain verifier's count is `2 + 2R` (`k_plain`).

`SchedDense.lean` links the two at the dense form: the verifier's setup holds its own inner shape to the inner
statement's files (`RecSetup.checkShape`), and the plain verifier's one table is the files' under a named premise
(`InnerSetupIsFiles`).
-/

namespace RecDeploy

open Flock

theorem forIn_opt_size {α : Type} (g : ℕ → ℕ) (f : ℕ → Array α × ℕ → Option (ForInStep (Array α × ℕ)))
    (hf : ∀ i s t, f i s = some t → ∃ u, t = .yield u ∧ u.1.size = s.1.size + g i) :
    ∀ (l : List ℕ) (s0 s : Array α × ℕ), forIn l s0 f = some s → s.1.size = s0.1.size + (l.map g).sum := by
  intro l
  induction l with
  | nil => intro s0 s h; simp at h; subst h; simp
  | cons a l ih =>
    intro s0 s h
    simp only [List.forIn_cons] at h
    cases hfa : f a s0 with
    | none => rw [hfa] at h; cases h
    | some t =>
      rw [hfa] at h
      obtain ⟨u, rfl, hu⟩ := hf a s0 t hfa
      have := ih u s h
      simp only [List.map_cons, List.sum_cons]
      omega

theorem sum_map_one (l : List ℕ) : (l.map fun _ => 1).sum = l.length := by
  induction l with
  | nil => rfl
  | cons a l ih => simp only [List.map_cons, List.sum_cons, ih, List.length_cons]; omega

/-- **`fast100`'s numbers**: it has a schedule only at `m ∈ [22, 35]`, with `n = m − 7`, `k₀` and its levels' count as
`fast100` sets them. -/
theorem fast100_sizes {m : ℕ} {sch : Schedule} (h : fast100 m = some sch) :
    22 ≤ m ∧ m ≤ 35 ∧ sch.n = m - 7 ∧ sch.k0 = (if m = 28 then 4 else if m = 29 then 5 else 6) ∧
      sch.levels.size = (if m ≤ 24 then 3 else if m ≤ 27 then 4 else if m ≤ 30 then 5 else if m ≤ 33 then 6 else 7) := by
  unfold fast100 at h
  by_cases hm : (m < 22 || m > 35) = true
  · simp [hm] at h
  · simp only [hm] at h
    simp at hm
    simp only [Std.Legacy.Range.forIn_eq_forIn_range'] at h
    simp only [Option.bind_eq_bind, Option.pure_def] at h
    simp at h
    simp only [Option.bind_eq_some_iff] at h
    obtain ⟨s, hs, he⟩ := h
    cases he
    have := forIn_opt_size (fun _ => 1) _ (fun i s t ht => by
      simp only [Option.some.injEq] at ht
      exact ⟨_, ht.symm, by simp⟩) _ _ _ hs
    rw [sum_map_one, List.length_range'] at this
    exact ⟨hm.1, hm.2, rfl, rfl, by simpa using this⟩

/-- The rounds pass `i` of `RecVStar.repRounds`' loop over a schedule's `r` later levels appends: four, then a level's
seven, or the last level's three. -/
def loopG (r i : ℕ) : ℕ := 4 + (if i + 1 < r then 7 else 3)

/-- **`RecVStar.repRounds`' size**: the zerocheck's and lincheck's rounds, each claim's, level 0's, and each later
level's (`loopG`). -/
theorem repRounds_size {m kLog claims : ℕ} {rs : Array RecVStar.SchedRound}
    (h : RecVStar.repRounds m kLog claims = some rs) :
    ∃ sch, fast100 m = some sch ∧ rs.size = (3 + (m - 5) + 1 + (kLog - 6) + 1 + claims + 1) + (2 + sch.k0 + 7) +
      ((List.range' 0 (sch.levels.size - 1)).map (loopG (sch.levels.size - 1))).sum := by
  unfold RecVStar.repRounds at h
  cases hs : fast100 m with
  | none => simp [hs] at h
  | some sch =>
    refine ⟨sch, rfl, ?_⟩
    simp only [hs, Std.Legacy.Range.forIn_eq_forIn_range'] at h
    simp only [Option.bind_eq_bind, Option.pure_def, Option.bind_some] at h
    simp only [Option.bind_eq_some_iff] at h
    obtain ⟨s, hl, he⟩ := h
    cases he
    have := forIn_opt_size (loopG (sch.levels.size - 1)) _ (fun i s t ht => by
      by_cases h4 : s.2 < 4
      · simp [h4] at ht
      · by_cases hi : i + 1 < sch.levels.size - 1
        · simp [h4, hi] at ht
          refine ⟨_, ht.symm, ?_⟩
          simp [loopG, hi]
        · simp [h4, hi] at ht
          refine ⟨_, ht.symm, ?_⟩
          simp [loopG, hi]) _ _ _ hl
    simp at this
    rw [this]
    simp
    omega

/-- **The plain schedule's size in closed form**, at `k_log ≥ 6`: `FlockDeploy.repRounds false`'s count. -/
theorem repRounds_closed {m kLog claims : ℕ} {rs : Array RecVStar.SchedRound}
    (h : RecVStar.repRounds m kLog claims = some rs) (hk : 6 ≤ kLog) : ∃ sch, fast100 m = some sch ∧
      rs.size = (3 + (m - 6)) + (3 + (kLog - 6)) + (claims + 1) + (sch.k0 + 5 + 11 * (sch.levels.size - 1)) := by
  obtain ⟨sch, hs, hsz⟩ := repRounds_size h
  obtain ⟨h1, -, -, -, hL⟩ := fast100_sizes hs
  have hsum : ((List.range' 0 (sch.levels.size - 1)).map (loopG (sch.levels.size - 1))).sum =
      11 * (sch.levels.size - 1) - 4 ∧ 1 ≤ sch.levels.size - 1 := by
    rw [hL]; repeat' (first | exact ⟨rfl, by decide⟩ | split)
  refine ⟨sch, hs, ?_⟩
  omega

/-- **The plain verifier counts one rep's rounds as the plain schedule has them**: at a table whose `m`, `k_log` and
claims (`2` and one per region and link point) have the plain schedule `rs`, at `k_log ≥ 6`. -/
theorem flockRepRounds_eq {spec : SessionSpec} {st : Stmt} {rs : Array RecVStar.SchedRound}
    (h : RecVStar.repRounds st.m st.c.kLog (2 + spec.points * st.regions.size) = some rs) (hk : 6 ≤ st.c.kLog) :
    FlockDeploy.repRounds false spec st = rs.size := by
  obtain ⟨sch, hs, hsz⟩ := repRounds_closed h hk
  rw [hsz]
  unfold FlockDeploy.repRounds
  simp only [Bool.false_eq_true, ↓reduceIte, hs]

/-- **The plain verifier's count at a one-table inner statement of the verifier's shape**: where the plain verifier
sets the inner statement up as the one table `st`, whose `m`, `k_log` and claims are the shape's, at `k_log ≥ 6`, it
takes `2 + 2R` rounds, `R` the plain schedule's. -/
theorem k_plain {xI : Claims.Msg} {s : FlockDeploy.Ins} {spec : SessionSpec} {st : Stmt} {sh : RecVStar.Inner}
    {rs : Array RecVStar.SchedRound} (hset : FlockDeploy.setupOf false xI = some (s, spec, #[st]))
    (hm : sh.m = st.m) (hk : sh.kLog = st.c.kLog) (hc : sh.claims = 2 + spec.points * st.regions.size)
    (hk6 : 6 ≤ st.c.kLog) (hrs : RecVStar.repRounds sh.m sh.kLog sh.claims = some rs) :
    (FlockDeploy.V false).k xI = 2 + 2 * rs.size := by
  rw [hm, hk, hc] at hrs
  have he := flockRepRounds_eq hrs hk6
  simp only [FlockDeploy.V, hset]
  simp [FlockDeploy.rounds, he]

end RecDeploy
