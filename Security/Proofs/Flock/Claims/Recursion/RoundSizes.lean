import Proofs.Flock.Claims.Recursion.VerdictTargetL
import Proofs.Flock.VBridge.Family.Setup
import Proofs.Flock.C3.Compose.HoloStreamSizes

/-!
# The schedule's coin counts are the rounds `verifierRows` takes

`roundSizes sh` is the one list of round sizes at which `verifierRows sh` returns (`verifierRows_schedule`). It is the
list `run` draws: the zerocheck's, the lincheck's and ring switching's (`headSz`, from `zerocheck_coins`,
`lincheck_coins`, `ringSwitch_spec`), then Ligerito's up to its final check (`ligSz`, `ligeritoRun_sz`, step by step
as `ligeritoRun_k` counts them), at a schedule of the seams (`roundSizes_eq`). The deployed schedule `repRounds` lists
the same counts (`repRounds_sz`): its head's at every `k_log` and claim count (`repHead_sz`), and its tail, which
neither reaches (`repRounds_tail`), by evaluating it at each `m` below 36.

So at a decision's table `i`, the schedule's coin counts are `roundSizes` at the table's shape (`DecAt.hsched`), and
row R's `hsz` holds wherever `structureOf` returns at that shape (`Top.hsz`).
-/

namespace FlockVBridge.Family

open Flock FlockSoundness.Refine

/-! ## The counts -/

/-- A level's queries and batching coin (`query`, RecAlgebra.lean:400–406): `q` positions, `log₂(q − 1) + 1`
consistency coins, one coin. -/
def qSz (sch : Flock.Schedule) (i : ℕ) : List ℕ :=
  [(sch.levels.getD i default).q, Nat.log2 ((sch.levels.getD i default).q - 1) + 1, 1]

/-- The coin counts `levelsLoop` draws, `fuel` levels from level `i` at the split dimension `split`
(RecAlgebra.lean:419–436). -/
def loopSz (sch : Flock.Schedule) (R : ℕ) : ℕ → ℕ → ℕ → List ℕ
  | 0, _, _ => []
  | fuel + 1, i, split =>
    List.replicate 4 2 ++
      if i + 1 < R then
        [split - 4 + 1, 1] ++ ([split - 4 + 1, 1] ++ (qSz sch (i + 1) ++ loopSz sch R fuel (i + 1) (split - 4 + 1)))
      else qSz sch R ++ loopSz sch R fuel (i + 1) split

/-- **The coin counts Ligerito draws up to its final check** (`ligeritoRun`, RecAlgebra.lean:440–454). -/
def ligSz (sch : Flock.Schedule) : List ℕ :=
  [sch.n, 1] ++ (List.replicate sch.k0 2 ++ ([sch.n - sch.k0 + 1, 1] ++ ([sch.n - sch.k0 + 1, 1] ++
    (qSz sch 0 ++ loopSz sch (sch.levels.size - 1) (sch.levels.size - 1) 0 (sch.n - sch.k0 + 1)))))

/-- **The coin counts before Ligerito**: the zerocheck's (RecAlgebra.lean:247–255), the lincheck's (288–295) and ring
switching's at `cl` claims (328–333). -/
def headSz (m kLog cl : ℕ) : List ℕ :=
  [6, m - 13, 1] ++ (List.replicate (m - 6) 1 ++ ([1, 1] ++ (List.replicate (kLog - 6) 1 ++ ([1] ++
    (List.replicate cl 7 ++ [cl])))))

/-- The rounds of `w` from `k` have the sizes `l`. -/
def SzAt (w : Array (Array F128)) (k : ℕ) (l : List ℕ) : Prop :=
  ∀ t (h : t < l.length), (w[k + t]!).size = l[t]

namespace SzAt

variable {w : Array (Array F128)}

theorem nil {k : ℕ} : SzAt w k [] := fun t h => absurd h (Nat.not_lt_zero t)

theorem one {k a : ℕ} (h : (w[k]!).size = a) : SzAt w k [a] := by
  intro t ht
  have : t = 0 := by simp at ht; omega
  subst this
  simpa using h

theorem cons {k k' a : ℕ} {l : List ℕ} (h0 : (w[k]!).size = a) (h : SzAt w k' l) (hk : k' = k + 1) :
    SzAt w k (a :: l) := by
  subst hk
  intro t ht
  cases t with
  | zero => simpa using h0
  | succ t =>
    have := h t (by simp at ht; omega)
    rw [show k + (t + 1) = k + 1 + t by omega]
    simpa using this

theorem app {k k' : ℕ} {l₁ l₂ : List ℕ} (h₁ : SzAt w k l₁) (h₂ : SzAt w k' l₂) (hk : k' = k + l₁.length) :
    SzAt w k (l₁ ++ l₂) := by
  subst hk
  intro t ht
  rw [List.getElem_append]
  split
  · rename_i h; exact h₁ t h
  · rename_i h
    have := h₂ (t - l₁.length) (by simp at ht; omega)
    rwa [show k + l₁.length + (t - l₁.length) = k + t by omega] at this

theorem rep {k n a : ℕ} (h : ∀ t < n, (w[k + t]!).size = a) : SzAt w k (List.replicate n a) := by
  intro t ht
  rw [List.getElem_replicate]
  exact h t (by simpa using ht)

theorem sizes {l : List ℕ} (h : SzAt w 0 l) (hs : w.size = l.length) : w.toList.map Array.size = l := by
  apply List.ext_getElem (by simp [hs])
  intro t h1 h2
  have := h t h2
  rw [Nat.zero_add, getElem!_pos w t (by simp at h1; omega)] at this
  simpa using this

end SzAt

/-! ## Ligerito's draws -/

section Lig

variable {w : Array (Array F128)} [Fact (Flock.RecAlgebra.NoEmpty w)] {M : Flock.RecAlgebra.Messages F128}

theorem challenge_sz {st st' : Flock.RecAlgebra.LSt F128} {k k' : ℕ} {rk : F256}
    (h : Flock.RecAlgebra.challenge M st k (Flock.RecAlgebra.Coins.ofRounds w) = .ok ((rk, st'), k')) :
    k' = k + 1 ∧ (w[k]!).size = 2 := by
  obtain ⟨s, ⟨a0, a1, a2⟩, msg, ood, oods, conss⟩ := st
  unfold Flock.RecAlgebra.challenge at h
  obtain ⟨hc, h⟩ := Flock.RecAlgebra.vcoinK2_bind h
  have h := Flock.RecAlgebra.vget_bind h
  dsimp only at h
  have h := Flock.RecAlgebra.vset_bind h
  obtain ⟨mm, -, h⟩ := Flock.RecAlgebra.vnextMsg_bind h
  have h := Flock.RecAlgebra.vmodify_bind h
  obtain ⟨-, -, rfl⟩ := Flock.RecAlgebra.vpure h
  exact ⟨rfl, hc⟩

theorem challenges_sz : ∀ (n : ℕ) (st st' : Flock.RecAlgebra.LSt F128) (k k' : ℕ) (chs : Array F256),
    Flock.RecAlgebra.challenges M n st k (Flock.RecAlgebra.Coins.ofRounds w) = .ok ((chs, st'), k') →
    k' = k + n ∧ SzAt w k (List.replicate n 2)
  | 0, st, st', k, k', chs, h => by
    simp only [Flock.RecAlgebra.challenges] at h
    obtain ⟨-, -, rfl⟩ := Flock.RecAlgebra.vpure h
    exact ⟨rfl, SzAt.nil⟩
  | n + 1, st, st', k, k', chs, h => by
    simp only [Flock.RecAlgebra.challenges] at h
    obtain ⟨rk, st₁, k₁, h₁, h⟩ := Flock.RecAlgebra.vbind_ok h
    obtain ⟨rest, st₂, k₂, h₂, h⟩ := Flock.RecAlgebra.vbind_ok h
    obtain ⟨-, -, rfl⟩ := Flock.RecAlgebra.vpure h
    obtain ⟨rfl, hc⟩ := challenge_sz h₁
    obtain ⟨rfl, hs⟩ := challenges_sz n _ _ _ _ _ h₂
    exact ⟨by omega, SzAt.cons hc hs rfl⟩

theorem oodIntro_sz {dim split : ℕ} {st st' : Flock.RecAlgebra.LSt F128} {k k' : ℕ} {u : PUnit} (hd : dim ≠ 0)
    (h : Flock.RecAlgebra.oodIntro M dim split st k (Flock.RecAlgebra.Coins.ofRounds w) = .ok ((u, st'), k')) :
    k' = k + 2 ∧ SzAt w k [dim, 1] := by
  unfold Flock.RecAlgebra.oodIntro at h
  obtain ⟨hz, h⟩ := Flock.RecAlgebra.vcoinVec_bind hd h
  obtain ⟨y, -, h⟩ := Flock.RecAlgebra.vnextOod_bind h
  obtain ⟨mm, -, h⟩ := Flock.RecAlgebra.vnextMsg_bind h
  obtain ⟨hb, h⟩ := Flock.RecAlgebra.vcoinOne_bind h
  unfold Flock.RecAlgebra.addScaled at h
  obtain ⟨-, rfl⟩ := Flock.RecAlgebra.vmodify_ok (Flock.RecAlgebra.vmodify_bind h)
  exact ⟨by omega, SzAt.cons hz (SzAt.one hb) rfl⟩

theorem queryFold_sz {sch : Flock.Schedule} {i cols : ℕ} {fold : Array F256} {st st' : Flock.RecAlgebra.LSt F128}
    {k k' : ℕ} {u : PUnit} (hq : (sch.levels.getD i default).q ≠ 0)
    (h : Flock.RecAlgebra.queryFold sch M i cols fold st k (Flock.RecAlgebra.Coins.ofRounds w) = .ok ((u, st'), k')) :
    k' = k + 3 ∧ SzAt w k (qSz sch i) := by
  obtain ⟨e, h0, h1, h2, -⟩ := Flock.RecAlgebra.vqueryFold hq h
  exact ⟨by omega, SzAt.cons h0 (SzAt.cons h1 (SzAt.one h2) rfl) rfl⟩

theorem levelsLoop_sz {sch : Flock.Schedule} {R : ℕ} (hq : ∀ i ≤ R, (sch.levels.getD i default).q ≠ 0) :
    ∀ (fuel i split : ℕ) (levels : Array (Array F256)) (e : ℕ) (st : Flock.RecAlgebra.LSt F128) (k : ℕ)
      (out : Array (Array F256) × ℕ) (st' : Flock.RecAlgebra.LSt F128) (k' : ℕ), i + fuel ≤ R →
      Flock.RecAlgebra.levelsLoop sch M R fuel i split levels e st k (Flock.RecAlgebra.Coins.ofRounds w) =
        .ok ((out, st'), k') →
      k' = k + (loopSz sch R fuel i split).length ∧ SzAt w k (loopSz sch R fuel i split)
  | 0, _, _, _, _, st, k, out, st', k', _, h => by
    simp only [Flock.RecAlgebra.levelsLoop] at h
    obtain ⟨-, -, rfl⟩ := Flock.RecAlgebra.vpure h
    exact ⟨by simp [loopSz], SzAt.nil⟩
  | fuel + 1, i, split, levels, e, st, k, out, st', k', hi, h => by
    simp only [Flock.RecAlgebra.levelsLoop] at h
    by_cases hs : split < 4
    · rw [ite_eq_left hs] at h
      exact (Flock.RecAlgebra.vthrow h).elim
    rw [ite_eq_right hs] at h
    obtain ⟨chs, st₁, k₁, h₁, h⟩ := Flock.RecAlgebra.vbind_ok h
    obtain ⟨e₁, s₁⟩ := challenges_sz 4 st st₁ k k₁ chs h₁
    try dsimp only at h
    by_cases hi' : i + 1 < R
    · rw [ite_eq_left hi'] at h
      obtain ⟨u₂, st₂, k₂, h₂, h⟩ := Flock.RecAlgebra.vbind_ok h
      obtain ⟨u₃, st₃, k₃, h₃, h⟩ := Flock.RecAlgebra.vbind_ok h
      obtain ⟨u₄, st₄, k₄, h₄, h⟩ := Flock.RecAlgebra.vbind_ok h
      obtain ⟨e₂, s₂⟩ := oodIntro_sz (by omega) h₂
      obtain ⟨e₃, s₃⟩ := oodIntro_sz (by omega) h₃
      obtain ⟨e₄, s₄⟩ := queryFold_sz (hq (i + 1) (by omega)) h₄
      obtain ⟨e₅, s₅⟩ := levelsLoop_sz hq fuel (i + 1) _ _ _ st₄ k₄ out st' k' (by omega) h
      simp only [loopSz, hi', ↓reduceIte]
      refine ⟨?_, SzAt.app s₁ (SzAt.app s₂ (SzAt.app s₃ (SzAt.app s₄ s₅ ?_) ?_) ?_) ?_⟩
      all_goals simp [qSz] <;> omega
    · rw [ite_eq_right hi'] at h
      obtain ⟨r, st₂, k₂, h₂, h⟩ := Flock.RecAlgebra.vbind_ok h
      obtain ⟨rfl, rfl, hq0, hq1, rfl⟩ := Flock.RecAlgebra.vquery (hq R le_rfl) h₂
      try dsimp only at h
      obtain ⟨hb, h⟩ := Flock.RecAlgebra.vcoinOne_bind h
      obtain ⟨⟨⟩, st₃, k₃, h₃, h⟩ := Flock.RecAlgebra.vbind_ok h
      obtain ⟨-, rfl⟩ := Flock.RecAlgebra.vmodify_ok h₃
      obtain ⟨e₅, s₅⟩ := levelsLoop_sz hq fuel (i + 1) _ _ _ st₃ _ out st' k' (by omega) h
      have s₄ : SzAt w k₁ (qSz sch R) := SzAt.cons hq0 (SzAt.cons hq1 (SzAt.one hb) rfl) rfl
      simp only [loopSz, hi', ↓reduceIte]
      refine ⟨?_, SzAt.app s₁ (SzAt.app s₄ s₅ ?_) ?_⟩
      all_goals simp [qSz] <;> omega

/-- **Ligerito up to its final check draws `ligSz`'s rounds**, at a schedule of the seams. -/
theorem ligeritoRun_sz {sch : Flock.Schedule} {st₀ : Flock.RecAlgebra.LSt F128} {k₀ k' : ℕ}
    {out : Array F256 × Array (Array F256) × ℕ} {stV : Flock.RecAlgebra.LSt F128} (hdim : DimSeam sch)
    (hlv : ∀ i < sch.levels.size, LevelSeam (sch.levels.getD i default))
    (hrun : (Flock.RecAlgebra.ligeritoRun sch M).run st₀ k₀ (Flock.RecAlgebra.Coins.ofRounds w) =
      .ok ((out, stV), k')) :
    k' = k₀ + (ligSz sch).length ∧ SzAt w k₀ (ligSz sch) := by
  obtain ⟨hn7, h2, hl0, hc0, hl4, hcols, hext⟩ := hdim
  have hc01 := hcols 0 (by omega)
  have hq : ∀ i ≤ sch.levels.size - 1, (sch.levels.getD i default).q ≠ 0 :=
    fun i hi => (hlv i (by omega)).q_ne_zero
  rw [StateT.run] at hrun
  unfold Flock.RecAlgebra.ligeritoRun at hrun
  obtain ⟨hz0, h⟩ := Flock.RecAlgebra.vcoinVec_bind (by omega) hrun
  obtain ⟨y0, -, h⟩ := Flock.RecAlgebra.vnextOod_bind h
  obtain ⟨hb0, h⟩ := Flock.RecAlgebra.vcoinOne_bind h
  obtain ⟨⟨⟩, stZ, kZ, hZ, h⟩ := Flock.RecAlgebra.vbind_ok h
  obtain ⟨-, rfl⟩ := Flock.RecAlgebra.vmodify_ok hZ
  obtain ⟨m0, -, h⟩ := Flock.RecAlgebra.vnextMsg_bind h
  obtain ⟨⟨⟩, stA, kA, hA, h⟩ := Flock.RecAlgebra.vbind_ok h
  obtain ⟨-, rfl⟩ := Flock.RecAlgebra.vmodify_ok hA
  obtain ⟨ic, st₁, k₁, h₁, h⟩ := Flock.RecAlgebra.vbind_ok h
  try dsimp only at h
  obtain ⟨⟨⟩, st₂, k₂, h₂, h⟩ := Flock.RecAlgebra.vbind_ok h
  obtain ⟨⟨⟩, st₃, k₃, h₃, h⟩ := Flock.RecAlgebra.vbind_ok h
  obtain ⟨⟨⟩, st₄, k₄, h₄, h⟩ := Flock.RecAlgebra.vbind_ok h
  obtain ⟨⟨lv', ex'⟩, st₅, k₅, h₅, h⟩ := Flock.RecAlgebra.vbind_ok h
  try dsimp only at h
  obtain ⟨-, -, rfl⟩ := Flock.RecAlgebra.vpure h
  obtain ⟨e₁, s₁⟩ := challenges_sz sch.k0 _ _ _ _ _ h₁
  obtain ⟨e₂, s₂⟩ := oodIntro_sz (by omega) h₂
  obtain ⟨e₃, s₃⟩ := oodIntro_sz (by omega) h₃
  obtain ⟨e₄, s₄⟩ := queryFold_sz (hlv 0 (by omega)).q_ne_zero h₄
  obtain ⟨e₅, s₅⟩ := levelsLoop_sz hq _ _ _ _ _ st₄ _ _ _ _ (by omega) h₅
  unfold ligSz
  refine ⟨?_, SzAt.app (SzAt.cons hz0 (SzAt.one hb0) rfl)
    (SzAt.app s₁ (SzAt.app s₂ (SzAt.app s₃ (SzAt.app s₄ s₅ ?_) ?_) ?_) ?_) ?_⟩
  all_goals simp [qSz] <;> omega

end Lig

/-! ## The rounds `verifierRows` takes -/

/-- **`roundSizes` is the schedule's counts**: at a shape where `structureOf` returns, with `2 + n` claims, `m ≥ 14`
and a schedule of the seams, the rounds `verifierRows` takes are the zerocheck's, the lincheck's and ring switching's
(`headSz`), then Ligerito's (`ligSz`). -/
theorem roundSizes_eq {sh : Shape} {S : Flock.RecAlgebra.GfS} {sch : Flock.Schedule} {n : ℕ}
    (hS : Flock.RecAlgebra.structureOf sh = .ok S) (hcl : sh.claims = 2 + n) (hm : 14 ≤ sh.m)
    (hsch : sh.sched = sch) (hdim : DimSeam sch)
    (hlv : ∀ i < sch.levels.size, LevelSeam (sch.levels.getD i default)) :
    roundSizes sh = headSz sh.m sh.kLog (2 + n) ++ ligSz sch := by
  have hneZ := Flock.RecAlgebra.noEmpty_zeroRounds (roundSizes_ne hS)
  obtain ⟨x, hx⟩ := (roundSizes_spec hS hneZ (Flock.RecAlgebra.extrasLike sh) #[]).2
    ⟨zeroRounds_sizes _, by simp [Flock.RecAlgebra.extrasLike]; omega⟩
  obtain ⟨v, info, kappa⟩ := x
  obtain ⟨lins, hl⟩ := residualLins_of_verifierRows hx
  obtain ⟨j, hj⟩ := Flock.RecAlgebra.residualLins_run hl
  obtain ⟨-, r0, r1, rsr, lgr, k1, k2, k3, h0, h1, h2, h3, -⟩ := Flock.RecAlgebra.run_stages hj
  obtain ⟨c₀, c₁, c₂, e0, e1, e2, -, -, -, hk1, ezc⟩ := Flock.RecAlgebra.zerocheck_coins hm h0
  obtain ⟨cα, cβ, cr, eα, eβ, er, -, -, -, -, -, -, -, -, hk2, elc⟩ := Flock.RecAlgebra.lincheck_coins h1
  have hcs : (Flock.RecAlgebra.claimsOf sh.m sh.kLog info.zc info.lc (Flock.RecAlgebra.extrasLike sh)).size =
      2 + n := by
    simp [Flock.RecAlgebra.claimsOf, Flock.RecAlgebra.extrasLike]; omega
  obtain ⟨_, -, -, eg, hk3, -, -, ers⟩ := Flock.RecAlgebra.ringSwitch_spec (by omega) h2
  have hzc : (Flock.RecAlgebra.symbolic sh).zc.size = sh.m - 6 := by simp [Flock.RecAlgebra.symbolic]
  have hlc : (Flock.RecAlgebra.symbolic sh).lc.size = sh.kLog - 6 := by simp [Flock.RecAlgebra.symbolic]
  rw [hzc] at hk1 ezc
  rw [hlc] at hk2 elc er
  rw [hcs] at eg hk3
  have h3' := Flock.RecAlgebra.ligerito_sim (Flock.RecAlgebra.Lin.eval_hom fun _ => 0) _ _ _ _ _ _ _ _ h3
  obtain ⟨st₀, out, stV, k₄, hrun, hdr⟩ := ligerito_drawn h3'
  rw [hsch] at hrun
  have : Fact (Flock.RecAlgebra.NoEmpty (Flock.RecAlgebra.zeroRounds (roundSizes sh))) := ⟨hneZ⟩
  obtain ⟨hk, pL⟩ := ligeritoRun_sz hdim hlv hrun
  have hd : k₄ = (Flock.RecAlgebra.zeroRounds (roundSizes sh)).size := by
    have e : k₄ = ((Flock.RecAlgebra.zeroRounds (roundSizes sh)).filter (·.size ≠ 0)).size := beq_iff_eq.mp hdr
    rwa [Flock.RecAlgebra.filter_noEmpty hneZ] at e
  generalize hc : Flock.RecAlgebra.zeroRounds (roundSizes sh) = w at hneZ e0 e1 e2 ezc eα eβ er elc eg ers pL hd
  have p0 : SzAt w 0 [6, sh.m - 13, 1] :=
    SzAt.cons (size_at (by omega) (round_size hneZ e0).2)
      (SzAt.cons (size_at (by omega) (round_size hneZ e1).2) (SzAt.one (size_at (by omega) (round_size hneZ e2).2))
        rfl) rfl
  have p1 : SzAt w 3 (List.replicate (sh.m - 6) 1) := SzAt.rep fun t ht => by
    obtain ⟨c', hc', -⟩ := ezc t ht
    exact size_at (by omega) (round_size hneZ hc').2
  have p2 : SzAt w k1 [1, 1] :=
    SzAt.cons (size_at (by omega) (round_size hneZ eα).2) (SzAt.one (size_at (by omega) (round_size hneZ eβ).2)) rfl
  have p3 : SzAt w (k1 + 2) (List.replicate (sh.kLog - 6) 1) := SzAt.rep fun t ht => by
    obtain ⟨c', hc', -⟩ := elc t ht
    exact size_at (by omega) (round_size hneZ hc').2
  have p4 : SzAt w (k1 + 2 + (sh.kLog - 6)) [1] := SzAt.one (size_at (by omega) (round_size hneZ er).2)
  have p5 : SzAt w k2 (List.replicate (2 + n) 7) := SzAt.rep fun t ht => by
    obtain ⟨c', hc', -⟩ := ers t (by omega)
    exact size_at (by omega) (round_size hneZ hc').2
  have p6 : SzAt w (k2 + (2 + n)) [2 + n] := SzAt.one (size_at (by omega) (round_size hneZ eg).2)
  have pH : SzAt w 0 (headSz sh.m sh.kLog (2 + n)) := by
    unfold headSz
    exact SzAt.app p0 (SzAt.app p1 (SzAt.app p2 (SzAt.app p3 (SzAt.app p4 (SzAt.app p5 p6 (by simp)) (by simp <;> omega))
      (by simp <;> omega)) (by simp <;> omega)) (by simp <;> omega)) (by simp <;> omega)
  have hall := SzAt.sizes (SzAt.app pH pL (by simp [headSz]; omega)) (by simp [headSz]; omega)
  rw [← hall, ← hc]
  exact (zeroRounds_sizes _).symm

/-! ## The schedule's counts -/

theorem repHead_sz {m : ℕ} (hm : 6 ≤ m) (kLog claims : ℕ) :
    (FlockC3.Compose.repHead m kLog claims).toList.map (·.1) = headSz m kLog claims := by
  obtain ⟨m, rfl⟩ : ∃ k, m = k + 6 := ⟨m - 6, by omega⟩
  simp only [FlockC3.Compose.repHead, headSz, show m + 6 - 5 = m + 1 by omega, show m + 6 - 6 = m by omega]
  simp [List.map_replicate, List.replicate_succ']

/-- **`repRounds` lists the counts `run` draws**: its head's (`repHead_sz`) at every `k_log` and claim count, and
Ligerito's after it, which neither reaches (`repRounds_tail`), by evaluating `repRounds` at each `m` below 36 at
`k_log = 6` and one claim. -/
theorem repRounds_sz {m kLog claims : ℕ} {out : Array RecVStar.SchedRound} {sch : Flock.Schedule}
    (h : RecVStar.repRounds m kLog claims = some out) (hs : fast100 m = some sch) :
    out.toList.map (·.1) = headSz m kLog claims ++ ligSz sch := by
  obtain ⟨hm22, hm⟩ := FlockC3.Compose.repRounds_m h
  obtain ⟨T, rfl, hall⟩ := FlockC3.Compose.repRounds_tail h
  have key : ∀ m' < 36, (fast100 m').all fun s => (RecVStar.repRounds m' 6 1).all fun o =>
      o.toList.map (·.1) == headSz m' 6 1 ++ ligSz s := by
    decide +kernel
  have h1 := key m hm
  rw [hs, Option.all_some, hall 6 1, Option.all_some, beq_iff_eq, Array.toList_append, List.map_append,
    repHead_sz (by omega)] at h1
  rw [Array.toList_append, List.map_append, repHead_sz (by omega), List.append_cancel_left h1]

end FlockVBridge.Family

namespace RecDeploy.Rows

open Flock FlockSoundness.Refine
open FlockDeploy (Coin FinalP tabsOf)
open FlockVBridge.Family (roundSizes)
open FlockSoundness.Discharge.Recursion (HClass)

namespace DecAt

variable {rd : Claims.Msg → Option FinalP} {x : Claims.Msg} {ms : List Claims.Msg} {cs : List Coin}
  (D : DecAt rd x ms cs) (cls : HClass)

/-- **Row R's `hsched`**: at row G's table `i`, where `structureOf` returns at its shape, the schedule's coin counts
are the rounds `verifierRows` takes there (`roundSizes_eq`, `repRounds_sz`): the shape's `m`, `k_log` and schedule are
the statement's and its `fast100` one, and its claims the statement's `2 + points · |regions|` (`points_eq`). -/
theorem hsched {acc : ℕ → Array F128} {i : ℕ} (hG : D.RowG cls acc i) (hi : i < D.sts.size)
    {S : Flock.RecAlgebra.GfS} (hS : Flock.RecAlgebra.structureOf (D.sh cls i) = .ok S) :
    (RecDeploy.repSchedAt D.spec D.sts[i]!).map (·.1) = roundSizes (D.sh cls i) := by
  have htab : (D.files cls i).st = Setup.ofCircuit D.sts[i] := D.tab_eq hi
  have hfast : fast100 D.sts[i].m = some (D.files cls i).sch := by
    have := hG.fast
    rw [htab] at this
    exact this
  obtain ⟨h22, h35⟩ := FlockVBridge.Family.fast100_range hfast
  have hsm : (D.sh cls i).m = D.sts[i].m := by
    show (D.files cls i).st.m = _
    rw [htab]
    rfl
  have hsk : (D.sh cls i).kLog = D.sts[i].c.kLog := by
    show (D.files cls i).st.kLog = _
    rw [htab]
    rfl
  have hex : D.sts[i].extraClaims D.sess.points = .ok (D.files cls i).extras := by
    have := hG.extra
    rw [htab] at this
    exact this
  have hn : (D.files cls i).extras.size = D.spec.points * D.sts[i].regions.size := by
    rw [FlockDeploy.extraClaims_size hex, D.points_eq, Nat.mul_comm]
    unfold FlockDeploy.pointsOf
    rw [Array.size_map, Array.size_range]
  have hcl : (D.sh cls i).claims = 2 + D.spec.points * D.sts[i].regions.size := by
    show 2 + (D.files cls i).extras.size = _
    rw [hn]
  have hdim := (fast100_dims _ _ hfast).1
  have hlv : ∀ j < (D.files cls i).sch.levels.size, LevelSeam ((D.files cls i).sch.levels.getD j default) :=
    fun j hj => fast100_level _ _ hfast _ (by unfold Array.getD; rw [dite_eq_left hj]; exact Array.getElem_mem hj)
  have hv := FlockVBridge.Family.roundSizes_eq hS hcl (by omega) rfl hdim hlv
  have key : ∀ m < 36, (fast100 m).isSome → (RecVStar.repRounds m 6 1).isSome := by decide +kernel
  obtain ⟨out, hout⟩ := Option.isSome_iff_exists.1 (key D.sts[i].m (by omega) (by simp [hfast]))
  obtain ⟨T, -, hall⟩ := FlockC3.Compose.repRounds_tail hout
  have h2 := hall D.sts[i].c.kLog (2 + D.spec.points * D.sts[i].regions.size)
  rw [getElem!_pos D.sts i hi]
  simp only [RecDeploy.repSchedAt, h2, Option.map_some, Option.getD_some]
  rw [FlockVBridge.Family.repRounds_sz h2 hfast, hv, hsm, hsk]
  rfl

end DecAt

end RecDeploy.Rows

namespace RecDeploy.Top

open Flock
open Claims (Msg)
open FlockDeploy (Coin FinalP)
open FlockSoundness.Discharge.Recursion (HClass)
open RecDeploy.Rows (DecAt)

variable {rd : Msg → Option FinalP} {P : Msg → List Msg → List Coin → Prop}

/-- **Row R's `hsz` where the rows are asked**, in `TargetR`'s form, wherever `structureOf` returns at table `i`'s shape
(row R's `hS`): the schedule's coin counts are `roundSizes` there (`DecAt.hsched`, at row G). -/
theorem hsz (acc : Msg → List Msg → List Coin → ℕ → Array F128) {x : Msg} {ms : List Msg} {cs : List Coin}
    (hA : At rd P x ms cs) (D : DecAt rd x ms cs) (cls : HClass) {i : ℕ} (hi : i < D.sts.size)
    {S : Flock.RecAlgebra.GfS} (hS : Flock.RecAlgebra.structureOf (D.sh cls i) = .ok S) :
    ∀ r : Fin 2, (D.rep cls (acc x ms cs) i r).rounds.toList.map Array.size =
      FlockVBridge.Family.roundSizes (D.sh cls i) := by
  obtain ⟨⟨xo, sto, hst, hf, hx⟩, -, -, -⟩ := id hA
  exact hsz_of_sched acc hA D cls hi (D.hsched cls (D.rowG cls (acc x ms cs) hst hf hx hi) hi hS)

end RecDeploy.Top
