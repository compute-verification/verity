import Proofs.Flock.Soundness.Discharge.ZkExec.PlaysStatements

/-!
# The inner proof's runs

`zk_inner_plays`: `Zk.verifyInner` on seven rows refines the model's compiled inner proof. After `final_c'` (absorbed,
no coin), the executable draws `ε`, absorbs `[ρ, σ]`, draws the ten batching coins (seven rows and the triple's
three), absorbs `τ`'s commitment, draws `β`, absorbs `Y` and draws the `Q` query words; the model draws and receives the
same, in order. What follows the last coin is checks only (`forIn_keeps`).
-/

namespace FlockSoundness.Discharge.ZkExec

open Flock Refine Exec Game Model DecodeZL Transcript

local notation "𝔽" => FlockLevel3.GF128

/-- A loop whose every step yields and leaves the transcript as it was leaves the transcript as it was. -/
theorem forIn_keeps {σ : Type} (body : ℕ → σ → TM (ForInStep σ))
    (hbody : ∀ i s t b t', body i s t = .ok (b, t') → ∃ s', b = ForInStep.yield s' ∧ t' = t) :
    ∀ (n a : ℕ) (s : σ) (t t' : Flock.Transcript) (s' : σ),
      (forIn (List.range' a n) s body : TM σ) t = .ok (s', t') → t' = t
  | 0, a, s, t, t', s', h => by
      simp only [List.range'_zero, List.forIn_nil] at h
      exact (pure_ok h).2
  | n + 1, a, s, t, t', s', h => by
      rw [forIn_range'_cons] at h
      obtain ⟨r, t₁, h₁, h₂⟩ := bind_ok h
      obtain ⟨s₁, rfl, ht⟩ := hbody a s t r t₁ h₁
      exact (forIn_keeps body hbody n (a + 1) s₁ t₁ t' s' h₂).trans ht

theorem zk_inner_plays : ZkInnerPlays := by
  intro lay pr finalC cons t t' hcons hwf h
  unfold Zk.verifyInner at h
  dsimp only at h
  obtain ⟨-, h⟩ := check_bind h
  obtain ⟨t₁, k₁, h⟩ := label_bind h
  obtain ⟨t₂, k₂, h⟩ := absorbF_bind h
  obtain ⟨lt₂, n₂, t₃, r₃, kk₃, h⟩ := sample_bind h
  obtain ⟨t₄, k₄, h⟩ := absorbSlice_bind h
  obtain ⟨lt₄, n₄, t₅, r₅, kk₅, h⟩ := sampleVec_bind (by simp [Zk.triple]) h
  obtain ⟨t₆, k₆, h⟩ := absorbBytes_bind h
  obtain ⟨lt₆, n₆, t₇, r₇, kk₇, h⟩ := sample_bind h
  obtain ⟨-, h⟩ := check_bind h
  obtain ⟨t₈, k₈, h⟩ := absorbSlice_bind h
  obtain ⟨lt₈, n₈, t₉, r₉, kk₉, h⟩ := sampleVec_bind (by decide) h
  obtain ⟨-, h⟩ := check_bind h
  obtain ⟨-, h⟩ := check_bind h
  rw [forIn_range_eq] at h
  obtain ⟨_, tL, hloop, hE⟩ := bind_ok h
  have hL := forIn_keeps _ (by
    intro i s tt b tt' hb
    obtain ⟨-, hb⟩ := check_bind hb
    obtain ⟨-, hb⟩ := check_bind hb
    obtain ⟨-, hb⟩ := check_bind hb
    obtain ⟨rfl, rfl⟩ := pure_ok hb
    exact ⟨_, rfl, rfl⟩) Zk.Q 0 _ _ _ _ hloop
  have ht' : t' = t₉ := (pure_ok hE).2.trans hL
  rw [ht']
  have hr₂ : t₂.rounds = t.rounds := by rw [k₂.2, k₁.2]
  have hk₂ : t₂.k = t.k := by rw [k₂.1, k₁.1]
  have hr₄ : t₄.rounds = t.rounds := by rw [k₄.2, r₃, hr₂]
  have hk₄ : t₄.k = t.k + 1 := by rw [k₄.1, kk₃, hk₂]
  have hr₆ : t₆.rounds = t.rounds := by rw [k₆.2, r₅, hr₄]
  have hk₆ : t₆.k = t.k + 2 := by rw [k₆.1, kk₅, hk₄]
  have hr₈ : t₈.rounds = t.rounds := by rw [k₈.2, r₇, hr₆]
  have hk₈ : t₈.k = t.k + 3 := by rw [k₈.1, kk₇, hk₆]
  have hr₉ : t₉.rounds = t.rounds := by rw [r₉, hr₈]
  have hk₉ : t₉.k = t.k + 4 := by rw [kk₉, hk₈]
  set c₀ := (t₂.rounds[t₂.k]!).coins with hc₀
  set c₁ := (t₄.rounds[t₄.k]!).coins with hc₁
  set c₂ := (t₆.rounds[t₆.k]!).coins with hc₂
  set c₃ := (t₈.rounds[t₈.k]!).coins with hc₃
  have hsz₀ : c₀.size = 1 := by rw [hc₀, RoundsWF.get (hr₂ ▸ hwf), n₂]
  have hsz₁ : c₁.size = 7 + 3 := by rw [hc₁, RoundsWF.get (hr₄ ▸ hwf), n₄]; simp [Zk.triple, hcons]
  have hsz₂ : c₂.size = 1 := by rw [hc₂, RoundsWF.get (hr₆ ▸ hwf), n₆]
  have hsz₃ : c₃.size = Zk.Q := by rw [hc₃, RoundsWF.get (hr₈ ▸ hwf), n₈]
  have hcoins : consumed t.rounds t.k t₉.k = [c₀, c₁, c₂, c₃] := by
    rw [hk₉, consumed_append t.rounds (show t.k ≤ t.k + 1 by omega) (by omega),
      consumed_append t.rounds (show t.k + 1 ≤ t.k + 2 by omega) (by omega),
      consumed_append t.rounds (show t.k + 2 ≤ t.k + 3 by omega) (by omega), consumed_succ, consumed_succ,
      consumed_succ, consumed_succ, hc₀, hc₁, hc₂, hc₃, hr₂, hk₂, hr₄, hk₄, hr₆, hk₆, hr₈, hk₈]
    rfl
  refine ⟨hr₉, by omega, fun Hs N dom np nP pos rows tr root => ?_⟩
  rw [hcoins]
  unfold ZkLink.innerGameC
  refine ⟨?w, ?_⟩
  swap
  refine Plays.bind' (ms₂ := [⟨𝔽 × 𝔽, (fOf pr.rho, fOf pr.sigma)⟩, ⟨Dig, toDig pr.tauCommit⟩,
      ⟨Fin (np + nP) → 𝔽, vecOf (np + nP) pr.y⟩]) (cs₂ := [c₁, c₂, c₃]) (Plays.drawOne c₀ hsz₀) ?_ rfl rfl
  refine Plays.bind' (ms₂ := [⟨Dig, toDig pr.tauCommit⟩, ⟨Fin (np + nP) → 𝔽, vecOf (np + nP) pr.y⟩])
    (cs₂ := [c₁, c₂, c₃]) (Plays.recv _ (fOf pr.rho, fOf pr.sigma)) ?_ rfl rfl
  refine Plays.bind' (ms₂ := [⟨Dig, toDig pr.tauCommit⟩, ⟨Fin (np + nP) → 𝔽, vecOf (np + nP) pr.y⟩])
    (cs₂ := [c₂, c₃]) (Plays.drawVec c₁ hsz₁ (by decide)) ?_ rfl rfl
  refine Plays.bind' (ms₂ := [⟨Fin (np + nP) → 𝔽, vecOf (np + nP) pr.y⟩]) (cs₂ := [c₂, c₃])
    (Plays.recv _ (toDig pr.tauCommit)) ?_ rfl rfl
  refine Plays.bind' (ms₂ := [⟨Fin (np + nP) → 𝔽, vecOf (np + nP) pr.y⟩]) (cs₂ := [c₃])
    (Plays.drawOne c₂ hsz₂) ?_ rfl rfl
  refine Plays.bind' (ms₂ := []) (cs₂ := [c₃]) (Plays.recv _ (vecOf (np + nP) pr.y)) ?_ rfl rfl
  refine Plays.bind' (ms₂ := []) (cs₂ := []) (Plays.drawVec c₃ hsz₃ (by decide)) ?_ rfl rfl
  exact Plays.pure (ι := fOf) _

end FlockSoundness.Discharge.ZkExec
