import Proofs.Flock.Claims.Complete.Zk.Lanes
import Proofs.Flock.Claims.Complete.Zk.ReplayPos

/-!
# What the masked Ligerito returns, by coin position

`zkLigeritoWith_pos`: wherever the shipped masked Ligerito (`Zk.ligeritoWith hs`) returns from `t`, its record is the
coins it read there, whatever the hashes and the proof: the target `T`, level 0's out-of-domain coin (round `t.k + 1`'s)
with the proof's first out-of-domain value, the lane rounds' `K` coins and the pair `ρ`; and it ends at round
`t.k + k0 + 7 + 11 R` of the same rounds with nothing pending. `zk_ligerito_rec` is the compiled verifier's.

* The steps, with their coins and the out-of-domain list: `lsampleVecWith_eq`, `lsampleWith_eq`, `lsampleKWith_eq`,
  `challengeWith_oods`, `oodIntroWith_oods`, `queryLevelWith_oods`, `queryLevel0With_oods`.
* `RunZk.lg_pos`: the run's masked Ligerito, at table 0's schedule.
-/

namespace FlockComplete

open Flock Transcript FlockSoundness.Refine FlockSoundness.Discharge.ZkExec FlockDeploy

variable {β : Type}

/-! ## The steps -/

theorem lsampleVecWith_eq (hs : Hashes V) {n : ℕ} (hn : n ≠ 0) {f : Array F128 → LM β} {st st' : LState}
    {t t' : Transcript} {b : β} (h : (liftW (M := V) (sampleVecWith hs n) >>= f) st t = .ok ((b, st'), t')) :
    (t.rounds[t.k]!).n = n ∧ ∃ c t₁, c = (t.rounds[t.k]!).coins ∧ t₁.rounds = t.rounds ∧ t₁.k = t.k + 1 ∧
      t₁.pending = ByteArray.empty ∧ f c st t₁ = .ok ((b, st'), t') := by
  rw [liftW_V] at h
  obtain ⟨c, t₁, h₁, h₂⟩ := lift_bind h
  obtain ⟨hn', -, hr, hk⟩ := FlockDeploy.sampleVecWith_ok hs hn h₁
  obtain ⟨ec, ep⟩ := RetV.squeeze hs 2 hn (P := fun a t₁ => a = (t.rounds[t.k]!).coins ∧
    t₁.pending = ByteArray.empty) ⟨rfl, rfl⟩ c t₁ h₁
  exact ⟨hn', c, t₁, ec, hr, hk, ep, h₂⟩

theorem lsampleWith_eq (hs : Hashes V) {f : F128 → LM β} {st st' : LState} {t t' : Transcript} {b : β}
    (h : (liftW (M := V) (sampleWith hs) >>= f) st t = .ok ((b, st'), t')) :
    (t.rounds[t.k]!).n = 1 ∧ ∃ c t₁, c = (t.rounds[t.k]!).coins.getD 0 0 ∧ t₁.rounds = t.rounds ∧
      t₁.k = t.k + 1 ∧ t₁.pending = ByteArray.empty ∧ f c st t₁ = .ok ((b, st'), t') := by
  rw [liftW_V] at h
  obtain ⟨c, t₁, h₁, h₂⟩ := lift_bind h
  obtain ⟨hn', hr, hk⟩ := FlockDeploy.sampleWith_ok hs h₁
  obtain ⟨ec, ep⟩ := RetV.sample hs (P := fun a t₁ => a = (t.rounds[t.k]!).coins.getD 0 0 ∧
    t₁.pending = ByteArray.empty) ⟨rfl, rfl⟩ c t₁ h₁
  exact ⟨hn', c, t₁, ec, hr, hk, ep, h₂⟩

theorem lsampleKWith_eq (hs : Hashes V) {f : F256 → LM β} {st st' : LState} {t t' : Transcript} {b : β}
    (h : (liftW (M := V) (sampleKWith hs) >>= f) st t = .ok ((b, st'), t')) :
    (t.rounds[t.k]!).n = 2 ∧ ∃ c t₁, c = coinK (t.rounds[t.k]!).coins ∧ t₁.rounds = t.rounds ∧
      t₁.k = t.k + 1 ∧ t₁.pending = ByteArray.empty ∧ f c st t₁ = .ok ((b, st'), t') := by
  rw [liftW_V] at h
  obtain ⟨c, t₁, h₁, h₂⟩ := lift_bind h
  obtain ⟨hn', hr, hk⟩ := FlockDeploy.sampleKWith_ok hs h₁
  have e : RetV (sampleKWith hs) t fun a t₁ => a = coinK (t.rounds[t.k]!).coins ∧ t₁.pending = ByteArray.empty := by
    unfold sampleKWith
    exact RetV.bind (RetV.squeeze hs 2 (by decide) (RetV.pure ⟨rfl, rfl⟩))
  obtain ⟨ec, ep⟩ := e c t₁ h₁
  exact ⟨hn', c, t₁, ec, hr, hk, ep, h₂⟩

theorem challengeWith_oods (hs : Hashes V) {p : LigeritoProof} {f : F256 → LM β} {st st' : LState}
    {t t' : Transcript} {b : β} (h : (challengeWith hs p >>= f) st t = .ok ((b, st'), t')) :
    ∃ c st₁ t₁, st₁.oods = st.oods ∧ t₁.rounds = t.rounds ∧ t₁.k = t.k + 1 ∧ f c st₁ t₁ = .ok ((b, st'), t') := by
  obtain ⟨r, st₁, t₁, h₁, h₂⟩ := lbind_ok h
  unfold challengeWith at h₁
  simp only [liftLM_V] at h₁
  obtain ⟨-, c, t₃, hr₃, hk₃, h₁⟩ := lsampleKWith_bind hs h₁
  have h₁ := lmodify_bind h₁
  obtain ⟨m, t₄, -, k₄, h₁⟩ := nextMsg_bind h₁
  have h₁ := lmodify_bind h₁
  obtain ⟨rfl, rfl, rfl⟩ := lpure_ok h₁
  refine ⟨_, _, _, ?_, k₄.2.trans hr₃, k₄.1.trans hk₃, h₂⟩
  rfl

theorem oodIntroWith_oods (hs : Hashes V) {p : LigeritoProof} {dim split : ℕ} (hdim : dim ≠ 0) {f : PUnit → LM β}
    {st st' : LState} {t t' : Transcript} {b : β} (h : (oodIntroWith hs p dim split >>= f) st t = .ok ((b, st'), t')) :
    ∃ st₁ t₁, (∃ x, st₁.oods = st.oods.push x) ∧ t₁.rounds = t.rounds ∧ t₁.k = t.k + 2 ∧
      f () st₁ t₁ = .ok ((b, st'), t') := by
  obtain ⟨u, st₁, t₁, h₁, h₂⟩ := lbind_ok h
  unfold oodIntroWith at h₁
  simp only [liftLM_V] at h₁
  obtain ⟨-, t₃, hr₃, hk₃, h₁⟩ := lsampleVecWith_bind hs hdim h₁
  obtain ⟨y, -, h₁⟩ := nextOod_bind h₁
  obtain ⟨_, t₄, h₄, h₁⟩ := lift_bind h₁
  have k₄ := absorbF_ok h₄
  obtain ⟨m, t₅, -, k₅, h₁⟩ := nextMsg_bind h₁
  obtain ⟨t₆, k₆, h₁⟩ := claimPow_bind h₁
  obtain ⟨-, bc, t₇, hr₇, hk₇, h₁⟩ := lsampleWith_bind hs h₁
  obtain ⟨rfl, rfl⟩ := lmodify_ok h₁
  refine ⟨_, _, ?_, by rw [hr₇, k₆.2, k₅.2, k₄.2, hr₃], by rw [hk₇, k₆.1, k₅.1, k₄.1, hk₃], h₂⟩
  exact ⟨_, rfl⟩

theorem queryLevelWith_oods (hs : Hashes V) {ms : MerkleScheme} {p : LigeritoProof} {lv : Flock.Level}
    {cap : Array ByteArray} {rows : Array (Array F128)} {sibs salts : Array ByteArray} {fold : Array F256}
    (hq : lv.q ≠ 0) {f : Array ℕ × Array F128 × F256 → LM β} {st st' : LState} {t t' : Transcript} {b : β}
    (h : (queryLevelWith hs ms p lv cap rows sibs salts fold >>= f) st t = .ok ((b, st'), t')) :
    ∃ r st₁ t₁, st₁.oods = st.oods ∧ t₁.rounds = t.rounds ∧ t₁.k = t.k + 2 ∧ f r st₁ t₁ = .ok ((b, st'), t') := by
  obtain ⟨r, st₁, t₁, h₁, h₂⟩ := lbind_ok h
  unfold queryLevelWith at h₁
  simp only [liftLM_V] at h₁
  have h₁ := lget_bind h₁
  obtain ⟨qn, _, h₁⟩ := nonceFrom_bind h₁
  obtain ⟨cn, _, h₁⟩ := nonceFrom_bind h₁
  have h₁ := lset_bind h₁
  obtain ⟨_, t₃, h₃, h₁⟩ := lift_bind h₁
  obtain ⟨t₃', k₃, h₃'⟩ := pow_bind (f := fun _ => pure ()) (by rw [bind_pure]; exact h₃)
  obtain ⟨rfl, rfl⟩ := pure_ok h₃'
  obtain ⟨-, t₄, hr₄, hk₄, h₁⟩ := lsampleVecWith_bind hs hq h₁
  obtain ⟨_, t₅, h₅, h₁⟩ := lift_bind h₁
  obtain ⟨t₅', k₅, h₅'⟩ := pow_bind (f := fun _ => pure ()) (by rw [bind_pure]; exact h₅)
  obtain ⟨rfl, rfl⟩ := pure_ok h₅'
  obtain ⟨-, t₆, hr₆, hk₆, h₁⟩ := lsampleVecWith_bind hs (by omega) h₁
  cases hmc : merkleCheckWith (ms.opsWith hs) ms cap lv.cap lv.d (positions lv (t₃.rounds[t₃.k]!).coins) rows sibs
      salts (2 ^ lv.lanes) with
  | error e => simp only [hmc] at h₁; obtain ⟨_, _, _, h₀, _⟩ := lbind_ok h₁; cases h₀
  | ok u =>
    simp only [hmc] at h₁
    simp only [List.forIn_pure_yield_eq_foldl, Std.Legacy.Range.forIn_eq_forIn_range', pure_bind] at h₁
    obtain ⟨rfl, rfl, rfl⟩ := lpure_ok h₁
    refine ⟨_, _, _, ?_, ?_, ?_, h₂⟩
    · rfl
    · rw [hr₆, k₅.2, hr₄, k₃.2]
    · rw [hk₆, k₅.1, hk₄, k₃.1]

theorem queryLevel0With_oods (hs : Hashes V) {ms : MerkleScheme} {p : LigeritoProof} {lv : Flock.Level}
    {cap : Array ByteArray} {fold : Array F256} (hq : lv.q ≠ 0) {f : Array ℕ × Array F128 × F256 → LM β}
    {st st' : LState} {t t' : Transcript} {b : β}
    (h : (Zk.queryLevel0With hs ms p lv cap fold >>= f) st t = .ok ((b, st'), t')) :
    ∃ r st₁ t₁, st₁.oods = st.oods ∧ t₁.rounds = t.rounds ∧ t₁.k = t.k + 2 ∧ f r st₁ t₁ = .ok ((b, st'), t') := by
  obtain ⟨r, st₁, t₁, h₁, h₂⟩ := lbind_ok h
  unfold Zk.queryLevel0With at h₁
  simp only [liftLM_V] at h₁
  have h₁ := lget_bind h₁
  obtain ⟨qn, _, h₁⟩ := nonceFrom_bind h₁
  obtain ⟨cn, _, h₁⟩ := nonceFrom_bind h₁
  have h₁ := lset_bind h₁
  obtain ⟨_, t₃, h₃, h₁⟩ := lift_bind h₁
  obtain ⟨t₃', k₃, h₃'⟩ := pow_bind (f := fun _ => pure ()) (by rw [bind_pure]; exact h₃)
  obtain ⟨rfl, rfl⟩ := pure_ok h₃'
  obtain ⟨-, t₄, hr₄, hk₄, h₁⟩ := lsampleVecWith_bind hs hq h₁
  obtain ⟨_, t₅, h₅, h₁⟩ := lift_bind h₁
  obtain ⟨t₅', k₅, h₅'⟩ := pow_bind (f := fun _ => pure ()) (by rw [bind_pure]; exact h₅)
  obtain ⟨rfl, rfl⟩ := pure_ok h₅'
  obtain ⟨-, t₆, hr₆, hk₆, h₁⟩ := lsampleVecWith_bind hs (by omega) h₁
  cases hmc : merkleCheckWith (ms.opsWith hs) ms cap lv.cap lv.d (positions lv (t₃.rounds[t₃.k]!).coins)
      p.initialProof.rows p.initialProof.siblings p.initialProof.salts (2 ^ lv.lanes + Zk.EXTRA_LANES) with
  | error e => simp only [hmc] at h₁; obtain ⟨_, _, _, h₀, _⟩ := lbind_ok h₁; cases h₀
  | ok u =>
    simp only [hmc] at h₁
    simp only [List.forIn_pure_yield_eq_foldl, Std.Legacy.Range.forIn_eq_forIn_range', pure_bind] at h₁
    obtain ⟨rfl, rfl, rfl⟩ := lpure_ok h₁
    refine ⟨_, _, _, ?_, ?_, ?_, h₂⟩
    · rfl
    · rw [hr₆, k₅.2, hr₄, k₃.2]
    · rw [hk₆, k₅.1, hk₄, k₃.1]

theorem transcript_eq {t : Transcript} {R : Array Round} {k : ℕ} {p : ByteArray} (hR : t.rounds = R) (hk : t.k = k)
    (hp : t.pending = p) : t = ⟨R, k, p⟩ := by
  subst hR hk hp
  rfl

/-! ## The record -/

set_option maxHeartbeats 4000000 in
/-- **What the masked Ligerito returns, by coin position**: the target, level 0's out-of-domain coin (round
`t.k + 1`'s) with the proof's first out-of-domain value, the lane rounds' `K` coins and the pair's, and it ends at round
`t.k + k0 + 7 + 11 R` of the same rounds with nothing pending, whatever the hashes and the proof; those rounds issued
one coin, then two each. -/
theorem zkLigeritoWith_pos (hs : Hashes V) (ms : MerkleScheme) (sch : Schedule) (T : F128)
    (cap : Array ByteArray) (p : LigeritoProof) (basisAt : Array F256 → ℕ → Array F256) (zl : Zk.Level0)
    (hn : sch.n ≠ 0) (hk0 : 1 ≤ sch.k0) (h2 : 2 ≤ sch.levels.size)
    (hq : ∀ i < sch.levels.size, (sch.levels.getD i default).q ≠ 0) {t t' : Transcript} {rc : Zk.L0Rec}
    (h : Zk.ligeritoWith hs ms sch T cap p basisAt zl t = .ok (rc, t')) :
    rc.target = T ∧ rc.ood = #[((t.rounds[t.k + 1]!).coins.getD 0 0, p.oodValues[0]!)] ∧
      rc.challenges.size = sch.k0 ∧ (∀ j < sch.k0, rc.challenges[j]! = coinK (t.rounds[t.k + 2 + j]!).coins) ∧
      rc.rho = (coinK (t.rounds[t.k + 2 + sch.k0]!).coins, coinK (t.rounds[t.k + 3 + sch.k0]!).coins) ∧
      t' = ⟨t.rounds, t.k + (sch.k0 + 7 + 11 * (sch.levels.size - 1)), ByteArray.empty⟩ ∧
      (t.rounds[t.k + 1]!).n = 1 ∧ (∀ j < sch.k0, (t.rounds[t.k + 2 + j]!).n = 2) ∧
      (t.rounds[t.k + 2 + sch.k0]!).n = 2 ∧ (t.rounds[t.k + 3 + sch.k0]!).n = 2 := by
  obtain ⟨hrE, hkE⟩ := zkLigeritoWith_rounds hs ms sch T cap p basisAt zl hn hk0 h2 hq h
  unfold Zk.ligeritoWith at h
  simp only [liftTM_V, liftLM_V] at h
  obtain ⟨-, h⟩ := check_bind h
  obtain ⟨-, h⟩ := check_bind h
  obtain ⟨t₁, k₁, h⟩ := label_bind h
  obtain ⟨t₂, k₂, h⟩ := absorbF_bind h
  obtain ⟨t₃, k₃, h⟩ := absorbBytes_bind h
  obtain ⟨t₃', k₃', h⟩ := label_bind h
  obtain ⟨⟨rc', stF⟩, tF, h₄, h⟩ := bind_ok h
  obtain ⟨rfl, rfl⟩ := pure_ok h
  rw [StateT.run] at h₄
  obtain ⟨-, z, t₅, -, hr₅, hk₅, -, h₄⟩ := lsampleVecWith_eq hs hn h₄
  obtain ⟨y₀, -, h₄⟩ := nextOod_bind h₄
  obtain ⟨_, t₆, h₆, h₄⟩ := lift_bind h₄
  have k₆ := absorbF_ok h₆
  obtain ⟨t₇, k₇, h₄⟩ := claimPow_bind h₄
  obtain ⟨hn₈, bc, t₈, ebc, hr₈, hk₈, -, h₄⟩ := lsampleWith_eq hs h₄
  have h₄ := lmodify_bind h₄
  obtain ⟨m₀, t₉, -, k₉, h₄⟩ := nextMsg_bind h₄
  have h₄ := lmodify_bind h₄
  obtain ⟨lane, st₁₀, t₁₀, h₁₀, h₄⟩ := lbind_ok h₄
  rw [lforIn_range_eq] at h₁₀
  obtain ⟨hr₁₀, hk₁₀, ho₁₀, hsz₁₀, -, hch₁₀, hρ₁₀, hnl₁₀, hnρ₀, hnρ₁⟩ := lane_rec _ sch.k0 (by
      intro i v st tE bv st' tE' hiB hb
      obtain ⟨hnR, r, tR, rfl, hrR, hkR, -, hb⟩ := lsampleKWith_eq hs hb
      have hb := lmodify_bind hb
      try dsimp only at hb
      by_cases hj : i + 1 = sch.k0
      · rw [ite_eq_left (show (i + 1 == sch.k0) = true by simp [hj])] at hb
        obtain ⟨_, t₂, h₂, hb⟩ := lift_bind hb
        obtain ⟨f₂k, f₂r⟩ := absorbK_ok h₂
        obtain ⟨_, t₃, h₃, hb⟩ := lift_bind hb
        obtain ⟨f₃k, f₃r⟩ := absorbK_ok h₃
        obtain ⟨hn₄, ρ₀, t₄, eρ₀, hr₄, hk₄, -, hb⟩ := lsampleKWith_eq hs hb
        obtain ⟨hn₅, ρ₁, t₅, eρ₁, hr₅, hk₅, -, hb⟩ := lsampleKWith_eq hs hb
        try dsimp only at hb
        by_cases hρ : (!Zk.rhoIndependent (ρ₀, ρ₁)) = true
        · rw [ite_eq_left hρ] at hb
          obtain ⟨_, _, _, h0, _⟩ := lbind_ok hb
          cases h0
        rw [ite_eq_right hρ] at hb
        try simp only [pure_bind] at hb
        obtain ⟨_, t₆, h₆, hb⟩ := lift_bind hb
        obtain ⟨f₆k, f₆r⟩ := label_ok h₆
        obtain ⟨_, t₇, h₇, hb⟩ := lift_bind hb
        obtain ⟨f₇k, f₇r⟩ := absorbK_ok h₇
        obtain ⟨_, t₈, h₈, hb⟩ := lift_bind hb
        obtain ⟨f₈k, f₈r⟩ := absorbSlice_ok h₈
        have hb := lmodify_bind hb
        obtain ⟨m, t₉, -, ⟨f₉k, f₉r⟩, hb⟩ := nextMsg_bind hb
        have hb := lmodify_bind hb
        obtain ⟨rfl, rfl, rfl⟩ := lpure_ok hb
        refine ⟨?_, by rw [f₉r, f₈r, f₇r, f₆r, hr₅, hr₄, f₃r, f₂r, hrR], by rw [ite_eq_left hj]; omega, rfl, hnR,
          fun _ => ⟨?_, ?_⟩⟩
        · subst eρ₀ eρ₁
          rw [ite_eq_left hj, show tE.k + 1 = t₃.k by omega, show tE.k + 2 = t₄.k by omega,
            show tE.rounds = t₃.rounds by rw [f₃r, f₂r, hrR], show t₄.rounds = t₃.rounds from hr₄]
        · rw [show tE.k + 1 = t₃.k by omega, show tE.rounds = t₃.rounds by rw [f₃r, f₂r, hrR]]
          exact hn₄
        · rw [show tE.k + 2 = t₄.k by omega, show tE.rounds = t₄.rounds by rw [hr₄, f₃r, f₂r, hrR]]
          exact hn₅
      · rw [ite_eq_right (show ¬ (i + 1 == sch.k0) = true by simp [hj])] at hb
        try simp only [pure_bind] at hb
        obtain ⟨m, t₉, -, ⟨f₉k, f₉r⟩, hb⟩ := nextMsg_bind hb
        have hb := lmodify_bind hb
        obtain ⟨rfl, rfl, rfl⟩ := lpure_ok hb
        refine ⟨?_, by rw [f₉r, hrR], by rw [ite_eq_right hj]; omega, rfl, hnR, fun h => absurd h hj⟩
        rw [ite_eq_right hj])
    sch.k0 0 _ _ t₉ _ _ t₁₀ (by omega) (by omega) h₁₀
  have hI₁₀ : st₁₀.oods[0]? = some ⟨z, bc, none⟩ := by
    rw [ho₁₀]; rfl
  try dsimp only at h₄
  obtain ⟨_, t₁₁, h₁₁, h₄⟩ := lift_bind h₄
  have k₁₁ := absorbBytes_ok h₁₁
  obtain ⟨st₁₂, t₁₂, ⟨x₁₂, ho₁₂⟩, -, -, h₄⟩ := oodIntroWith_oods hs (show sch.n - sch.k0 + 1 ≠ 0 by omega) h₄
  obtain ⟨st₁₃, t₁₃, ⟨x₁₃, ho₁₃⟩, -, -, h₄⟩ := oodIntroWith_oods hs (show sch.n - sch.k0 + 1 ≠ 0 by omega) h₄
  obtain ⟨⟨pos0, al0, e0⟩, st₁₄, t₁₄, ho₁₄, -, -, h₄⟩ := queryLevel0With_oods hs (hq 0 (by omega)) h₄
  have hI₁₄ : st₁₄.oods[0]? = some ⟨z, bc, none⟩ := by
    rw [ho₁₄, ho₁₃, ho₁₂]
    exact push_get0 (push_get0 hI₁₀)
  dsimp only at h₄
  obtain ⟨extra, stX, tX, hX, h₄'⟩ := lbind_ok h₄
  clear h₄
  rw [lforIn_range_eq, List.forIn_pure_yield_eq_foldl] at hX
  obtain ⟨rfl, rfl, rfl⟩ := lpure_ok hX
  obtain ⟨mq, t₁₅, -, -, h₄⟩ := nextMsg_bind h₄'
  obtain ⟨t₁₆, -, h₄⟩ := claimPow_bind h₄
  obtain ⟨-, bq, t₁₇, -, -, -, hp₁₇, h₄⟩ := lsampleWith_eq hs h₄
  have h₄ := lmodify_bind h₄
  obtain ⟨out, st₁₈, t₁₈, hloop, hcont⟩ := lbind_ok h₄
  rw [lforIn_range_eq] at hloop
  obtain ⟨hI₁₈, hp₁₈⟩ := lforIn_inv' _ (fun st tE => st.oods[0]? = some ⟨z, bc, none⟩ ∧
      tE.pending = ByteArray.empty) (sch.levels.size - 1) (by
      intro i v stE tE bv stE' tE' hi hI hb
      obtain ⟨hI, -⟩ := hI
      try dsimp only at hb
      by_cases c4 : v.1 < 4
      · rw [ite_eq_left c4] at hb
        cases hb
      rw [ite_eq_right c4] at hb
      obtain ⟨chs, st₁, t₁, hf, hb⟩ := lbind_ok hb
      rw [lforIn_range_eq] at hf
      have ho₁ : st₁.oods = stE.oods := lforIn_inv _ (fun st => st.oods = stE.oods) (by
          intro j s st tt b st' tt' hI hb'
          obtain ⟨c, st₂, tt₂, ho₂, -, -, hb'⟩ := challengeWith_oods hs hb'
          obtain ⟨-, rfl, -⟩ := lpure_ok hb'
          exact ho₂.trans hI) _ _ _ _ _ _ _ rfl hf
      by_cases hi' : i + 1 < sch.levels.size - 1
      · rw [ite_eq_left hi'] at hb
        obtain ⟨_, t₂, -, hb⟩ := lift_bind hb
        obtain ⟨st₃, t₃, ⟨x₃, ho₃⟩, -, -, hb⟩ := oodIntroWith_oods hs (show v.1 - 4 + 1 ≠ 0 by omega) hb
        obtain ⟨st₄, t₄, ⟨x₄, ho₄⟩, -, -, hb⟩ := oodIntroWith_oods hs (show v.1 - 4 + 1 ≠ 0 by omega) hb
        obtain ⟨_, st₅, t₅, ho₅, -, -, hb⟩ := queryLevelWith_oods hs (hq (i + 1) (by omega)) hb
        obtain ⟨mq, t₆, -, -, hb⟩ := nextMsg_bind hb
        obtain ⟨t₇, -, hb⟩ := claimPow_bind hb
        obtain ⟨-, bq, t₈, -, -, -, hp₈, hb⟩ := lsampleWith_eq hs hb
        have hb := lmodify_bind hb
        by_cases c5 : v.1 - 4 + 1 < 4
        · rw [ite_eq_left c5] at hb
          cases hb
        rw [ite_eq_right c5] at hb
        obtain ⟨-, rfl, rfl⟩ := lpure_ok hb
        refine ⟨?_, hp₈⟩
        show st₅.oods[0]? = _
        rw [ho₅, ho₄, ho₃, ho₁]
        exact push_get0 (push_get0 hI)
      · rw [ite_eq_right hi'] at hb
        have hb := lget_bind hb
        by_cases hmc₀ : (st₁.msg != p.messages.size) = true
        · rw [ite_eq_left hmc₀] at hb
          obtain ⟨_, _, _, h0, _⟩ := lbind_ok hb
          cases h0
        rw [ite_eq_right hmc₀] at hb
        by_cases hyr : (p.finalProof.yr.size != 2 ^ (v.1 - 4 + 1)) = true
        · rw [ite_eq_left hyr] at hb
          obtain ⟨_, _, _, h0, _⟩ := lbind_ok hb
          cases h0
        rw [ite_eq_right hyr] at hb
        obtain ⟨_, st₂, t₂, h₂, hb₂⟩ := lbind_ok hb
        clear hb
        rw [← Array.forIn_toList] at h₂
        obtain ⟨rfl, -⟩ := forIn_absorbF_keeps _ h₂
        obtain ⟨_, st₃, t₃, ho₃, -, -, hb⟩ :=
          queryLevelWith_oods hs (hq (sch.levels.size - 1) (by omega)) hb₂
        obtain ⟨t₄, -, hb⟩ := claimPow_bind hb
        obtain ⟨-, bq, t₅, -, -, -, hp₅, hb⟩ := lsampleWith_eq hs hb
        have hb := lmodify_bind hb
        obtain ⟨-, rfl, rfl⟩ := lpure_ok hb
        refine ⟨?_, hp₅⟩
        show st₃.oods[0]? = _
        rw [ho₃, ho₁]
        exact hI)
    (sch.levels.size - 1) 0 _ _ t₁₇ _ _ t₁₈ (by omega) ⟨hI₁₄, hp₁₇⟩ hloop
  obtain rfl := finalProgZ_ok hcont
  have hrec := finalProgZ_rec hcont
  subst hrec
  have er₇ : t₇.rounds = t.rounds := by rw [k₇.2, k₆.2, hr₅, k₃'.2, k₃.2, k₂.2, k₁.2]
  have ek₇ : t₇.k = t.k + 1 := by rw [k₇.1, k₆.1, hk₅, k₃'.1, k₃.1, k₂.1, k₁.1]
  have er₉ : t₉.rounds = t.rounds := by rw [k₉.2, hr₈, er₇]
  have ek₉ : t₉.k = t.k + 2 := by rw [k₉.1, hk₈, ek₇]
  have eb : (st₁₈.oods[0]!).beta = (t.rounds[t.k + 1]!).coins.getD 0 0 := by
    rw [getElem!_of_getElem? hI₁₈, ← er₇, ← ek₇]
    exact ebc
  refine ⟨rfl, ?_, by simpa using hsz₁₀, fun j hj => ?_, ?_, transcript_eq hrE hkE hp₁₈, ?_, fun j hj => ?_, ?_, ?_⟩
  · show #[((st₁₈.oods[0]!).beta, p.oodValues[0]!)] = _
    rw [eb]
  · have := hch₁₀ j hj
    simp only [Array.size_empty, Nat.zero_add] at this
    rw [this, er₉, ek₉]
  · rw [hρ₁₀, er₉, ek₉, show t.k + 2 + sch.k0 + 1 = t.k + 3 + sch.k0 by omega]
  · rw [← er₇, ← ek₇]
    exact hn₈
  · rw [← er₉, ← ek₉]
    exact hnl₁₀ j hj
  · rw [← er₉, ← ek₉]
    exact hnρ₀
  · rw [show t.k + 3 + sch.k0 = t.k + 2 + sch.k0 + 1 by omega, ← er₉, ← ek₉]
    exact hnρ₁

variable {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}

/-- **The run's masked Ligerito, by coin position**, at table 0's `--zk` schedule: its record is the coins of the rounds
the ring switch left it, and it ends `k0 + 7 + 11 R` rounds on with nothing pending; those rounds issued one coin, then
two each. -/
theorem RunZk.lg_pos (hLg : LigeritoZk H rel) (o : RunZk H rel) {st : Flock.Stmt}
    (htabs : o.tabs = #[Setup.ofCircuit st]) {v : Schedule × ℕ} (hv : Zk.schedule st.m = .ok v) (hsch : o.sch 0 = v.1)
    {r : ℕ} (hr : r < 2) :
    (o.lg 0 r).1.target = (o.sw 0 r).1.1 ∧
      (o.lg 0 r).1.ood = #[(((o.sw 0 r).2.rounds[(o.sw 0 r).2.k + 1]!).coins.getD 0 0,
        (o.zp 0 r).proof.ligerito.oodValues[0]!)] ∧
      (o.lg 0 r).1.challenges.size = v.1.k0 ∧
      (∀ j < v.1.k0, (o.lg 0 r).1.challenges[j]! = coinK ((o.sw 0 r).2.rounds[(o.sw 0 r).2.k + 2 + j]!).coins) ∧
      (o.lg 0 r).1.rho = (coinK ((o.sw 0 r).2.rounds[(o.sw 0 r).2.k + 2 + v.1.k0]!).coins,
        coinK ((o.sw 0 r).2.rounds[(o.sw 0 r).2.k + 3 + v.1.k0]!).coins) ∧
      (o.lg 0 r).2 = ⟨(o.sw 0 r).2.rounds, (o.sw 0 r).2.k + (v.1.k0 + 7 + 11 * (v.1.levels.size - 1)),
        ByteArray.empty⟩ ∧
      ((o.sw 0 r).2.rounds[(o.sw 0 r).2.k + 1]!).n = 1 ∧
      (∀ j < v.1.k0, ((o.sw 0 r).2.rounds[(o.sw 0 r).2.k + 2 + j]!).n = 2) ∧
      ((o.sw 0 r).2.rounds[(o.sw 0 r).2.k + 2 + v.1.k0]!).n = 2 ∧
      ((o.sw 0 r).2.rounds[(o.sw 0 r).2.k + 3 + v.1.k0]!).n = 2 := by
  have h0 : 0 < o.tabs.size := by rw [htabs]; simp
  obtain ⟨h25, -, hsm, hk0, -, hdim, hlv, -, -⟩ := zk_schedule_seams st.m v.1 v.2 hv
  obtain ⟨hn7, h2, -, -, -, -, -⟩ := hdim
  have e := hLg o 0 h0 r hr
  rw [hsch] at e
  exact zkLigeritoWith_pos o.hs _ v.1 _ _ _ _ _ (by omega) hk0 h2 (fun i hi => (hlv i hi).q_ne_zero)
    (t' := (o.lg 0 r).2) (rc := (o.lg 0 r).1) e

end FlockComplete
