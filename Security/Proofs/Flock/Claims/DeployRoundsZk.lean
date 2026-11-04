import Proofs.Flock.Claims.DeployRounds
import Proofs.Flock.Soundness.Discharge.ZkExec.PlaysLig
import Proofs.Flock.Soundness.Discharge.ZkExec.RunsSeams
import Flock.With.Zk

/-!
# The rounds the shipped masked verifier reads

`Zk.verifyWith h`'s phases, as `DeployRounds` has the plain verifier's: which round each squeeze reads does not depend
on the hash. The masked replay reads `m + (kLog − 6)` rounds, Ligerito `k0 + 7 + 11 R` (the plain count and the lane
phase's `ρ` pair), and the inner proof four; a masked rep reads all of its stream's rounds (`zkVerifyRepWith_rounds`).
-/

namespace FlockDeploy

open Flock Transcript FlockSoundness.Refine

variable {α β : Type}

/-! ## The masked replay -/

/-- **The masked replay reads `m + (kLog − 6)` rounds**: zerocheck's skip and outer points, its first coin and one per
residual round, then lincheck's `α`, `β`, one per sumcheck round, and `r'`. -/
theorem replayWith_rounds (hs : Hashes V) {st : Setup} {lay : Zk.PadLayout} {p : Proof} {root : ByteArray}
    (hm : 14 ≤ st.m) {t t' : Transcript} {rp : Zk.Replayed} (h : Zk.replayWith hs st lay p root t = .ok (rp, t')) :
    t'.rounds = t.rounds ∧ t'.k = t.k + (st.m + (st.kLog - 6)) := by
  unfold Zk.replayWith at h
  simp only [liftTM_V] at h
  obtain ⟨t₁, k₁, h⟩ := label_bind h
  obtain ⟨t₂, k₂, h⟩ := absorbBytes_bind h
  obtain ⟨t₃, k₃, h⟩ := absorbBytes_bind h
  obtain ⟨t₄, k₄, h⟩ := label_bind h
  obtain ⟨t₅, k₅, h⟩ := label_bind h
  obtain ⟨t₆, k₆, h⟩ := absorbBytes_bind h
  obtain ⟨-, -, t₇, r₇, kk₇, h⟩ := sampleVecWith_bind hs (by omega) h
  obtain ⟨-, -, t₈, r₈, kk₈, h⟩ := sampleVecWith_bind hs (by omega) h
  obtain ⟨t₉, k₉, h⟩ := absorbSlice_bind h
  obtain ⟨t₁₀, k₁₀, h⟩ := absorbSlice_bind h
  obtain ⟨-, -, t₁₁, r₁₁, kk₁₁, h⟩ := sampleWith_bind hs h
  obtain ⟨s, t₁₂, hloop, h⟩ := bind_ok h
  rw [forIn_range_eq] at hloop
  obtain ⟨hr₁₂, hk₁₂⟩ := forIn_moves _ 1 (by
      intro i st' tt b tt' hb
      obtain ⟨tt₁, kk₁, hb⟩ := absorbF_bind hb
      obtain ⟨tt₂, kk₂, hb⟩ := absorbF_bind hb
      obtain ⟨-, -, tt₃, rr₃, kk₃, hb⟩ := sampleWith_bind hs hb
      obtain ⟨rfl, rfl⟩ := pure_ok hb
      exact ⟨by rw [rr₃, kk₂.2, kk₁.2], by rw [kk₃, kk₂.1, kk₁.1], _, rfl⟩) (st.m - 6) 0 _ t₁₁ t₁₂ s hloop
  obtain ⟨t₁₃, k₁₃, h⟩ := absorbF_bind h
  obtain ⟨t₁₄, k₁₄, h⟩ := absorbF_bind h
  obtain ⟨t₁₅, k₁₅, h⟩ := label_bind h
  obtain ⟨-, -, t₁₆, r₁₆, kk₁₆, h⟩ := sampleWith_bind hs h
  obtain ⟨-, -, t₁₇, r₁₇, kk₁₇, h⟩ := sampleWith_bind hs h
  obtain ⟨s', t₁₈, hloop', h⟩ := bind_ok h
  rw [forIn_range_eq] at hloop'
  obtain ⟨hr₁₈, hk₁₈⟩ := forIn_moves _ 1 (by
      intro i st' tt b tt' hb
      obtain ⟨tt₁, kk₁, hb⟩ := absorbF_bind hb
      obtain ⟨tt₂, kk₂, hb⟩ := absorbF_bind hb
      obtain ⟨-, -, tt₃, rr₃, kk₃, hb⟩ := sampleWith_bind hs hb
      obtain ⟨rfl, rfl⟩ := pure_ok hb
      exact ⟨by rw [rr₃, kk₂.2, kk₁.2], by rw [kk₃, kk₂.1, kk₁.1], _, rfl⟩) (st.kLog - 6) 0 _ t₁₇ t₁₈ s' hloop'
  try dsimp only at h
  cases hf : st.fold.folded ((t₁₅.rounds[t₁₅.k]!).coins.getD 0 0) ((t₁₆.rounds[t₁₆.k]!).coins.getD 0 0)
      (skipWeights ((t₁₀.rounds[t₁₀.k]!).coins.getD 0 0)) (s.2.extract 0 (st.kLog - 6)) s'.2 with
  | error err => rw [hf] at h; obtain ⟨_, _, h0, _⟩ := bind_ok h; cases h0
  | ok v =>
  rw [hf] at h
  simp only [pure_bind] at h
  obtain ⟨-, h⟩ := check_bind h
  obtain ⟨t₁₉, k₁₉, h⟩ := absorbSlice_bind h
  obtain ⟨-, -, t₂₀, r₂₀, kk₂₀, h⟩ := sampleWith_bind hs h
  obtain ⟨rfl, rfl⟩ := pure_ok h
  refine ⟨by rw [r₂₀, k₁₉.2, hr₁₈, r₁₇, r₁₆, k₁₅.2, k₁₄.2, k₁₃.2, hr₁₂, r₁₁, k₁₀.2, k₉.2, r₈, r₇, k₆.2, k₅.2,
    k₄.2, k₃.2, k₂.2, k₁.2], ?_⟩
  rw [kk₂₀, k₁₉.1, hk₁₈, kk₁₇, kk₁₆, k₁₅.1, k₁₄.1, k₁₃.1, hk₁₂, kk₁₁, k₁₀.1, k₉.1, kk₈, kk₇, k₆.1, k₅.1, k₄.1,
    k₃.1, k₂.1, k₁.1]
  omega

/-! ## The inner proof -/

/-- A step of `V` lifted into the transcript monad keeps the transcript. -/
theorem vlift_bind {x : V α} {f : α → TM β} {t t' : Transcript} {b : β}
    (h : ((monadLift x : TM α) >>= f) t = .ok (b, t')) : ∃ a, x = .ok a ∧ f a t = .ok (b, t') := by
  obtain ⟨a, t₁, h₁, h₂⟩ := bind_ok h
  change (StateT.lift x) t = _ at h₁
  unfold StateT.lift at h₁
  cases hx : x with
  | error e => rw [hx] at h₁; cases h₁
  | ok v => rw [hx] at h₁; cases h₁; exact ⟨_, rfl, h₂⟩

/-- **The inner proof reads four rounds**: `ε`, the constraints' `γ`s, `β` and the `Q` column queries. -/
theorem verifyInnerWith_rounds (hs : Hashes V) {lay : Zk.PadLayout} {pr : Zk.InnerProof} {finalC : F128}
    {cons : Array (String × Zk.Lin)} {t t' : Transcript} {u : Unit}
    (h : Zk.verifyInnerWith hs lay pr finalC cons t = .ok (u, t')) :
    t'.rounds = t.rounds ∧ t'.k = t.k + 4 := by
  unfold Zk.verifyInnerWith at h
  simp only [liftTM_V] at h
  obtain ⟨-, h⟩ := check_bind h
  obtain ⟨t₁, k₁, h⟩ := label_bind h
  obtain ⟨t₂, k₂, h⟩ := absorbF_bind h
  obtain ⟨-, -, t₃, r₃, kk₃, h⟩ := sampleWith_bind hs h
  obtain ⟨t₄, k₄, h⟩ := absorbSlice_bind h
  obtain ⟨-, -, t₅, r₅, kk₅, h⟩ := sampleVecWith_bind hs (by simp [Zk.triple]) h
  obtain ⟨t₆, k₆, h⟩ := absorbBytes_bind h
  obtain ⟨-, -, t₇, r₇, kk₇, h⟩ := sampleWith_bind hs h
  obtain ⟨-, h⟩ := check_bind h
  obtain ⟨t₈, k₈, h⟩ := absorbSlice_bind h
  obtain ⟨-, -, t₉, r₉, kk₉, h⟩ := sampleVecWith_bind hs (by decide) h
  obtain ⟨tc, -, h⟩ := vlift_bind h
  obtain ⟨-, h⟩ := check_bind h
  obtain ⟨-, h⟩ := check_bind h
  obtain ⟨s, t₁₀, hloop, h⟩ := bind_ok h
  obtain ⟨rfl, rfl⟩ := pure_ok h
  rw [forIn_range_eq] at hloop
  obtain ⟨hr₁₀, hk₁₀⟩ := forIn_keeps _ (by
      intro i s tt b tt' hb
      obtain ⟨-, hb⟩ := check_bind hb
      obtain ⟨-, hb⟩ := check_bind hb
      obtain ⟨_, -, hb⟩ := vlift_bind hb
      obtain ⟨_, -, hb⟩ := vlift_bind hb
      obtain ⟨_, -, hb⟩ := vlift_bind hb
      obtain ⟨-, hb⟩ := check_bind hb
      obtain ⟨rfl, rfl⟩ := pure_ok hb
      exact ⟨rfl, rfl⟩) _ _ _ _ _ hloop
  refine ⟨by rw [hr₁₀, r₉, k₈.2, r₇, k₆.2, r₅, k₄.2, r₃, k₂.2, k₁.2], ?_⟩
  rw [hk₁₀, kk₉, k₈.1, kk₇, k₆.1, kk₅, k₄.1, kk₃, k₂.1, k₁.1]

/-! ## Masked Ligerito -/

open FlockSoundness.Discharge.ZkExec in
/-- `queryLevel0With`: level 0's query coins are round `t.k`'s, `q` of them, and the Merkle check of its rows (with
the extra lanes) passes at their positions; the consistency coins are round `t.k + 1`'s. -/
theorem queryLevel0With_bind (hs : Hashes V) {ms : MerkleScheme} {p : LigeritoProof} {lv : Flock.Level}
    {cap : Array ByteArray} {fold : Array F256} (hq : lv.q ≠ 0) {f : Array ℕ × Array F128 × F256 → LM β}
    {st st' : LState} {t t' : Transcript} {b : β}
    (h : (Zk.queryLevel0With hs ms p lv cap fold >>= f) st t = .ok ((b, st'), t')) :
    (t.rounds[t.k]!).n = lv.q ∧
      merkleCheckWith (ms.opsWith hs) ms cap lv.cap lv.d (positions lv (t.rounds[t.k]!).coins) p.initialProof.rows
        p.initialProof.siblings p.initialProof.salts (2 ^ lv.lanes + Zk.EXTRA_LANES) = .ok () ∧
      ∃ r st₁ t₁, t₁.rounds = t.rounds ∧ t₁.k = t.k + 2 ∧ f r st₁ t₁ = .ok ((b, st'), t') := by
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
  obtain ⟨hn₄, t₄, hr₄, hk₄, h₁⟩ := lsampleVecWith_bind hs hq h₁
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
    refine ⟨by rw [← k₃.2, ← k₃.1]; exact hn₄, by rw [← k₃.2, ← k₃.1]; exact hmc, _, _, _, ?_, ?_, h₂⟩
    · rw [hr₆, k₅.2, hr₄, k₃.2]
    · rw [hk₆, k₅.1, hk₄, k₃.1]

/-- The lane loop: each of its `k0` steps moves one round, and the last two more (the lane phase's `ρ` pair). -/
theorem lforIn_lanes {σ : Type} (body : ℕ → σ → LM (ForInStep σ)) (k0 : ℕ)
    (hbody : ∀ i s st t b st' t', i < k0 → body i s st t = .ok ((b, st'), t') →
      t'.rounds = t.rounds ∧ t'.k = t.k + (if i + 1 = k0 then 3 else 1) ∧ ∃ s', b = .yield s') :
    ∀ (n a : ℕ) (s : σ) (st : LState) (t : Transcript) (out : σ) (st' : LState) (t' : Transcript),
      a + n = k0 → forIn (List.range' a n) s body st t = .ok ((out, st'), t') →
      t'.rounds = t.rounds ∧ t'.k = t.k + n + (if n = 0 then 0 else 2) := by
  intro n
  induction n with
  | zero =>
    intro a s st t out st' t' _ h
    simp only [List.range'_zero, List.forIn_nil] at h
    obtain ⟨rfl, rfl, rfl⟩ := lpure_ok h
    simp
  | succ n ih =>
    intro a s st t out st' t' ha h
    rw [List.range'_succ, List.forIn_cons] at h
    obtain ⟨b, st₁, t₁, h₁, h₂⟩ := lbind_ok h
    obtain ⟨hr₁, hk₁, s', rfl⟩ := hbody a s st t b st₁ t₁ (by omega) h₁
    obtain ⟨hr, hk⟩ := ih (a + 1) s' st₁ t₁ out st' t' (by omega) h₂
    refine ⟨hr.trans hr₁, ?_⟩
    rw [hk, hk₁]
    by_cases hn : n = 0
    · rw [ite_eq_left (show a + 1 = k0 by omega), ite_eq_left hn, ite_eq_right (by omega)]
      omega
    · rw [ite_eq_right (show ¬ a + 1 = k0 by omega), ite_eq_right hn, ite_eq_right (by omega)]
      omega

open FlockSoundness.Discharge.ZkExec in
/-- **Masked Ligerito reads `k0 + 7 + 11 R` rounds** on `R + 1 ≥ 2` levels: the plain count's rounds, and the lane
phase's `ρ` pair after the last lane challenge. -/
theorem zkLigeritoWith_rounds (hs : Hashes V) (ms : MerkleScheme) (sch : Schedule) (T : F128)
    (cap : Array ByteArray) (p : LigeritoProof) (basisAt : Array F256 → ℕ → Array F256) (zl : Zk.Level0)
    (hn : sch.n ≠ 0) (hk0 : 1 ≤ sch.k0) (h2 : 2 ≤ sch.levels.size)
    (hq : ∀ i < sch.levels.size, (sch.levels.getD i default).q ≠ 0) {t t' : Transcript} {rc : Zk.L0Rec}
    (h : Zk.ligeritoWith hs ms sch T cap p basisAt zl t = .ok (rc, t')) :
    t'.rounds = t.rounds ∧ t'.k = t.k + (sch.k0 + 7 + 11 * (sch.levels.size - 1)) := by
  unfold Zk.ligeritoWith at h
  simp only [liftTM_V, liftLM_V] at h
  obtain ⟨-, h⟩ := check_bind h
  obtain ⟨-, h⟩ := check_bind h
  obtain ⟨t₁, k₁, h⟩ := label_bind h
  obtain ⟨t₂, k₂, h⟩ := absorbF_bind h
  obtain ⟨t₃, k₃, h⟩ := absorbBytes_bind h
  obtain ⟨t₃', k₃', h⟩ := label_bind h
  obtain ⟨⟨rc', stF⟩, t₄, h₄, h⟩ := bind_ok h
  obtain ⟨-, rfl⟩ := pure_ok h
  rw [StateT.run] at h₄
  obtain ⟨-, t₅, hr₅, hk₅, h₄⟩ := lsampleVecWith_bind hs hn h₄
  obtain ⟨y₀, -, h₄⟩ := nextOod_bind h₄
  obtain ⟨_, t₆, h₆, h₄⟩ := lift_bind h₄
  have k₆ := absorbF_ok h₆
  obtain ⟨t₇, k₇, h₄⟩ := claimPow_bind h₄
  obtain ⟨-, bc, t₈, hr₈, hk₈, h₄⟩ := lsampleWith_bind hs h₄
  have h₄ := lmodify_bind h₄
  obtain ⟨m₀, t₉, -, k₉, h₄⟩ := nextMsg_bind h₄
  have h₄ := lmodify_bind h₄
  obtain ⟨lane, st₁₀, t₁₀, h₁₀, h₄⟩ := lbind_ok h₄
  rw [lforIn_range_eq] at h₁₀
  obtain ⟨hr₁₀, hk₁₀⟩ := lforIn_lanes _ sch.k0 (by
      intro i v st tt b st' tt' _ hb
      obtain ⟨-, r, tt₁, hr₁, hk₁, hb⟩ := lsampleKWith_bind hs hb
      have hb := lmodify_bind hb
      try dsimp only at hb
      by_cases hj : i + 1 = sch.k0
      · rw [ite_eq_left (show (i + 1 == sch.k0) = true by simp [hj])] at hb
        obtain ⟨_, tt₂, h₂, hb⟩ := lift_bind hb
        have k₂ := absorbK_ok h₂
        obtain ⟨_, tt₃, h₃, hb⟩ := lift_bind hb
        have k₃ := absorbK_ok h₃
        obtain ⟨-, ρ₀, tt₄, hr₄, hk₄, hb⟩ := lsampleKWith_bind hs hb
        obtain ⟨-, ρ₁, tt₅, hr₅, hk₅, hb⟩ := lsampleKWith_bind hs hb
        try dsimp only at hb
        by_cases hρ : (!Zk.rhoIndependent (ρ₀, ρ₁)) = true
        · rw [ite_eq_left hρ] at hb
          obtain ⟨_, _, _, h0, _⟩ := lbind_ok hb
          cases h0
        rw [ite_eq_right hρ] at hb
        try simp only [pure_bind] at hb
        obtain ⟨_, tt₆, h₆, hb⟩ := lift_bind hb
        have k₆ := label_ok h₆
        obtain ⟨_, tt₇, h₇, hb⟩ := lift_bind hb
        have k₇ := absorbK_ok h₇
        obtain ⟨_, tt₈, h₈, hb⟩ := lift_bind hb
        have k₈ := absorbSlice_ok h₈
        have hb := lmodify_bind hb
        obtain ⟨m, tt₉, -, k₉, hb⟩ := nextMsg_bind hb
        have hb := lmodify_bind hb
        obtain ⟨rfl, rfl, rfl⟩ := lpure_ok hb
        refine ⟨by rw [k₉.2, k₈.2, k₇.2, k₆.2, hr₅, hr₄, k₃.2, k₂.2, hr₁], ?_, _, rfl⟩
        rw [ite_eq_left hj, k₉.1, k₈.1, k₇.1, k₆.1, hk₅, hk₄, k₃.1, k₂.1, hk₁]
      · rw [ite_eq_right (show ¬ (i + 1 == sch.k0) = true by simp [hj])] at hb
        try simp only [pure_bind] at hb
        obtain ⟨m, tt₉, -, k₉, hb⟩ := nextMsg_bind hb
        have hb := lmodify_bind hb
        obtain ⟨rfl, rfl, rfl⟩ := lpure_ok hb
        refine ⟨by rw [k₉.2, hr₁], ?_, _, rfl⟩
        rw [ite_eq_right hj, k₉.1, hk₁]) sch.k0 0 _ _ t₉ lane st₁₀ t₁₀ (by omega) h₁₀
  try dsimp only at h₄
  obtain ⟨_, t₁₁, h₁₁, h₄⟩ := lift_bind h₄
  have k₁₁ := absorbBytes_ok h₁₁
  obtain ⟨st₁₂, t₁₂, hr₁₂, hk₁₂, h₄⟩ := oodIntroWith_bind hs (by omega) h₄
  obtain ⟨st₁₃, t₁₃, hr₁₃, hk₁₃, h₄⟩ := oodIntroWith_bind hs (by omega) h₄
  obtain ⟨-, -, ⟨pos0, al0, e0⟩, st₁₄, t₁₄, hr₁₄, hk₁₄, h₄⟩ := queryLevel0With_bind hs (hq 0 (by omega)) h₄
  dsimp only at h₄
  obtain ⟨extra, stX, tX, hX, h₄'⟩ := lbind_ok h₄
  clear h₄
  rw [lforIn_range_eq, List.forIn_pure_yield_eq_foldl] at hX
  obtain ⟨rfl, rfl, rfl⟩ := lpure_ok hX
  obtain ⟨mq, t₁₅, -, k₁₅, h₄⟩ := nextMsg_bind h₄'
  obtain ⟨t₁₆, k₁₆, h₄⟩ := claimPow_bind h₄
  obtain ⟨-, bq, t₁₇, hr₁₇, hk₁₇, h₄⟩ := lsampleWith_bind hs h₄
  have h₄ := lmodify_bind h₄
  obtain ⟨out, st₁₈, t₁₈, hloop, hcont⟩ := lbind_ok h₄
  rw [lforIn_range_eq] at hloop
  have hL := lforIn_levels _ (sch.levels.size - 1) ?hstep ?hlast (sch.levels.size - 1) 0 _ _ _ out st₁₈
    t₁₈ (by omega) (by omega) hloop
  case hstep =>
    intro i s st tt b st' tt' hi hb
    by_cases c4 : s.1 < 4
    · rw [ite_eq_left c4] at hb; obtain ⟨_, _, _, h0, _⟩ := lbind_ok hb; cases h0
    rw [ite_eq_right c4] at hb
    obtain ⟨chs, st₁, tt₁, hf, hb⟩ := lbind_ok hb
    rw [lforIn_range_eq] at hf
    obtain ⟨hr₁, hk₁⟩ := lforIn_moves _ 1 (by
        intro j s' st' tt' b' st'' tt'' hb'
        obtain ⟨c, st₁, tt₁, hr₁, hk₁, hb'⟩ := challengeWith_bind hs hb'
        obtain ⟨rfl, rfl, rfl⟩ := lpure_ok hb'
        exact ⟨hr₁, hk₁, _, rfl⟩) 4 0 _ _ _ chs st₁ tt₁ hf
    rw [ite_eq_left hi] at hb
    obtain ⟨_, tt₂, h₂, hb⟩ := lift_bind hb
    have k₂ := absorbBytes_ok h₂
    obtain ⟨st₃, tt₃, hr₃, hk₃, hb⟩ := oodIntroWith_bind hs (by omega) hb
    obtain ⟨st₄, tt₄, hr₄, hk₄, hb⟩ := oodIntroWith_bind hs (by omega) hb
    obtain ⟨-, -, r₅, st₅, tt₅, hr₅, hk₅, hb⟩ := queryLevelWith_bind hs (hq (i + 1) (by omega)) hb
    obtain ⟨mm, tt₆, -, k₆, hb⟩ := nextMsg_bind hb
    obtain ⟨tt₇, k₇, hb⟩ := claimPow_bind hb
    obtain ⟨-, bb, tt₈, hr₈, hk₈, hb⟩ := lsampleWith_bind hs hb
    have hb := lmodify_bind hb
    by_cases c4' : s.1 - 4 + 1 < 4
    · rw [ite_eq_left c4'] at hb; obtain ⟨_, _, _, h0, _⟩ := lbind_ok hb; cases h0
    rw [ite_eq_right c4'] at hb
    obtain ⟨rfl, rfl, rfl⟩ := lpure_ok hb
    exact ⟨by rw [hr₈, k₇.2, k₆.2, hr₅, hr₄, hr₃, k₂.2, hr₁],
      by rw [hk₈, k₇.1, k₆.1, hk₅, hk₄, hk₃, k₂.1, hk₁], _, rfl⟩
  case hlast =>
    intro i s st tt b st' tt' hi hb
    by_cases c4 : s.1 < 4
    · rw [ite_eq_left c4] at hb; obtain ⟨_, _, _, h0, _⟩ := lbind_ok hb; cases h0
    rw [ite_eq_right c4] at hb
    obtain ⟨chs, st₁, tt₁, hf, hb⟩ := lbind_ok hb
    rw [lforIn_range_eq] at hf
    obtain ⟨hr₁, hk₁⟩ := lforIn_moves _ 1 (by
        intro j s' st' tt' b' st'' tt'' hb'
        obtain ⟨c, st₁, tt₁, hr₁, hk₁, hb'⟩ := challengeWith_bind hs hb'
        obtain ⟨rfl, rfl, rfl⟩ := lpure_ok hb'
        exact ⟨hr₁, hk₁, _, rfl⟩) 4 0 _ _ _ chs st₁ tt₁ hf
    rw [ite_eq_right (show ¬ i + 1 < sch.levels.size - 1 by omega)] at hb
    have hb := lget_bind hb
    by_cases hmc₀ : (st₁.msg != p.messages.size) = true
    · rw [ite_eq_left hmc₀] at hb; obtain ⟨_, _, _, h0, _⟩ := lbind_ok hb; cases h0
    rw [ite_eq_right hmc₀] at hb
    by_cases hyr : (p.finalProof.yr.size != 2 ^ (s.1 - 4 + 1)) = true
    · rw [ite_eq_left hyr] at hb; obtain ⟨_, _, _, h0, _⟩ := lbind_ok hb; cases h0
    rw [ite_eq_right hyr] at hb
    obtain ⟨_, st₂, tt₂, h₂, hb⟩ := lbind_ok hb
    rw [← Array.forIn_toList] at h₂
    obtain ⟨-, k₂⟩ := forIn_absorbF_keeps _ h₂
    obtain ⟨-, -, r₃, st₃, tt₃, hr₃, hk₃, hb⟩ :=
      queryLevelWith_bind hs (hq (sch.levels.size - 1) (by omega)) hb
    obtain ⟨tt₄, k₄, hb⟩ := claimPow_bind hb
    obtain ⟨-, bb, tt₅, hr₅, hk₅, hb⟩ := lsampleWith_bind hs hb
    have hb := lmodify_bind hb
    obtain ⟨rfl, rfl, rfl⟩ := lpure_ok hb
    exact ⟨by rw [hr₅, k₄.2, hr₃, k₂.2, hr₁], by rw [hk₅, k₄.1, hk₃, k₂.1, hk₁], _, rfl⟩
  obtain ⟨hrL, hkL⟩ := hL
  obtain rfl := finalProgZ_ok hcont
  refine ⟨by rw [hrL, hr₁₇, k₁₆.2, k₁₅.2, hr₁₄, hr₁₃, hr₁₂, k₁₁.2, hr₁₀, k₉.2, hr₈, k₇.2, k₆.2, hr₅, k₃'.2, k₃.2,
    k₂.2, k₁.2], ?_⟩
  rw [hkL, hk₁₇, k₁₆.1, k₁₅.1, hk₁₄, hk₁₃, hk₁₂, k₁₁.1, hk₁₀, k₉.1, hk₈, k₇.1, k₆.1, hk₅, k₃'.1, k₃.1, k₂.1, k₁.1]
  rw [ite_eq_right (show ¬ sch.k0 = 0 by omega)]
  omega

/-! ## One rep -/

open FlockSoundness.Discharge.ZkExec in
/-- **A masked rep reads every recorded round of its stream**: when `Zk.verifyRepWith hs` accepts rep `r` of table `i`,
at the table's `--zk` schedule, its stream has `(m + (kLog − 6)) + (claims + 1) + (k0 + 7 + 11 R) + 4` rounds (the
replay, the ring switch over `claims = 2 + |extras|`, Ligerito over `R + 1` levels, the inner proof), whatever the hashes
`hs` are. -/
theorem zkVerifyRepWith_rounds (hs : Hashes V) {spec : SessionSpec} {st : Setup} {sch : Schedule} {tPad : ℕ}
    {sess : RunRecord} {i r : ℕ} {bytes c : ByteArray} (hsch : Zk.schedule st.m = .ok (sch, tPad))
    (h : Zk.verifyRepWith hs spec st sch tPad sess i r bytes = .ok c) :
    ∃ (stream : StreamRec) (extras : Array Claim),
      sess.streams.find? (fun x => x.name == toString spec.tables[i]! ++ toString "/rep" ++ toString r) = some stream ∧
      st.extra sess.points = .ok extras ∧
      stream.rounds.size =
        (st.m + (st.kLog - 6)) + ((2 + extras.size) + 1) + (sch.k0 + 7 + 11 * (sch.levels.size - 1)) + 4 := by
  obtain ⟨h25, -, hsm, hk0, -, hdim, hlv, -, -⟩ := zk_schedule_seams _ _ _ hsch
  obtain ⟨hn7, h2, -, -, -, -, -⟩ := hdim
  unfold Zk.verifyRepWith at h
  dsimp only at h
  cases hs' : Array.find? (fun x => x.name == toString spec.tables[i]! ++ toString "/rep" ++ toString r)
      sess.streams with
  | none => rw [hs'] at h; cases h
  | some stream =>
  rw [hs'] at h
  obtain ⟨extras, hex, h⟩ := ebind_ok h
  obtain ⟨zp, -, h⟩ := ebind_ok h
  by_cases c1 : (zp.proof.params != pcsParams st.merkle sch) = true
  · rw [ite_eq_left c1] at h; obtain ⟨_, h0, _⟩ := ebind_ok h; cases h0
  rw [ite_eq_right c1] at h
  by_cases c1' : (zp.inner.root.size != 64) = true
  · rw [ite_eq_left c1'] at h; obtain ⟨_, h0, _⟩ := ebind_ok h; cases h0
  rw [ite_eq_right c1'] at h
  obtain ⟨⟨u, tr⟩, hrun, h⟩ := ebind_ok h
  change (if (tr.k != stream.rounds.size) = true then _ else _) = _ at h
  by_cases c2 : (tr.k != stream.rounds.size) = true
  · rw [ite_eq_left c2] at h; obtain ⟨_, h0, _⟩ := ebind_ok h; cases h0
  simp only [bne_iff_ne, ne_eq, not_not] at c2
  refine ⟨stream, extras, rfl, hex, ?_⟩
  rw [StateT.run] at hrun
  obtain ⟨rp, t₁, h₁, hrun⟩ := bind_ok hrun
  obtain ⟨-, hk₁⟩ := replayWith_rounds hs (by omega) h₁
  try dsimp only at hrun
  obtain ⟨⟨T, sws⟩, t₂, h₂, hrun⟩ := bind_ok hrun
  obtain ⟨-, hk₂⟩ := ringSwitchWith_rounds hs (by simp) h₂
  obtain ⟨rc, t₃, h₃, hrun⟩ := bind_ok hrun
  obtain ⟨-, hk₃⟩ := zkLigeritoWith_rounds hs _ sch _ _ _ _ _ (by omega) hk0 h2
    (fun i hi => (hlv i hi).q_ne_zero) h₃
  cases hv : Zk.ligeritoConstraints (Zk.PadLayout.new st.m st.kLog sch.k0) rc zp.proof.ligerito zp.level0 with
  | error e =>
    rw [hv] at hrun
    try dsimp only at hrun
    obtain ⟨_, _, h₄, _⟩ := bind_ok hrun
    cases h₄
  | ok v =>
  rw [hv] at hrun
  try dsimp only at hrun
  obtain ⟨lg, t₄, h₄, hrun'⟩ := bind_ok hrun
  clear hrun
  obtain ⟨rfl, rfl⟩ := pure_ok h₄
  obtain ⟨-, hk₅⟩ := verifyInnerWith_rounds hs hrun'
  rw [← c2, hk₅, hk₃, hk₂, hk₁]
  simp only [Transcript.init, Array.size_append, List.size_toArray, List.length_cons, List.length_nil]
  omega

end FlockDeploy
