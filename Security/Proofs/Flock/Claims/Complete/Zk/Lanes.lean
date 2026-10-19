import Proofs.Flock.Claims.Complete.Zk.Skeleton
import Proofs.Flock.Claims.DeployRoundsZk
import Proofs.Flock.Soundness.Discharge.ZkExec.VerdictLane
import Proofs.Flock.Soundness.Discharge.ZkExec.RunsSizes

/-!
# Level 0's lane phase gives its constraints wherever the masked Ligerito returns (`LaneConstraints`)

`laneConstraints_goal`: `Zk.ligeritoConstraints` throws only when the record is not the layout's (one out-of-domain
value, `k0` lane challenges, at least `k0` sumcheck messages), and otherwise folds without failing.

* `lforIn_count`: a loop whose every step yields, adding one to a count, adds its length.
* `zkLigeritoWith_rec`: the record the masked Ligerito returns has one out-of-domain value and `k0` lane challenges, one
  pushed per lane round, whatever the hashes are.
* `ligeritoConstraints_ok`: on such a record, and `k0` messages, the lane phase returns.
* The messages: a decoded masked proof has `k0 + 8 + 7 (R − 1)` (`zk_decode_sizes`).
-/

namespace FlockComplete

open Flock Transcript FlockSoundness.Refine FlockSoundness.Discharge.ZkExec FlockDeploy

/-- **A loop whose every step yields, adding one to `sz`**, adds its length to `sz`. -/
theorem lforIn_count {σ : Type} (body : ℕ → σ → LM (ForInStep σ)) (sz : σ → ℕ)
    (hb : ∀ i s st t b st' t', body i s st t = .ok ((b, st'), t') → ∃ s', b = .yield s' ∧ sz s' = sz s + 1) :
    ∀ (n a : ℕ) (s : σ) (st : LState) (t : Transcript) (s' : σ) (st' : LState) (t' : Transcript),
      (forIn (List.range' a n) s body : LM σ) st t = .ok ((s', st'), t') → sz s' = sz s + n
  | 0, a, s, st, t, s', st', t', h => by
      simp only [List.range'_zero, List.forIn_nil] at h
      obtain ⟨rfl, -, -⟩ := lpure_ok h
      rfl
  | n + 1, a, s, st, t, s', st', t', h => by
      rw [List.range'_succ, List.forIn_cons] at h
      obtain ⟨r, st₁, t₁, h₁, h₂⟩ := lbind_ok h
      obtain ⟨s₁, rfl, hs₁⟩ := hb a s st t r st₁ t₁ h₁
      have := lforIn_count body sz hb n (a + 1) s₁ st₁ t₁ s' st' t' h₂
      omega

set_option maxHeartbeats 1000000 in
/-- **The masked Ligerito's record**: one out-of-domain value and `k0` lane challenges, whatever the hashes are. -/
theorem zkLigeritoWith_rec (hs : Hashes V) (ms : MerkleScheme) (sch : Schedule) (T : F128)
    (cap : Array ByteArray) (p : LigeritoProof) (basisAt : Array F256 → ℕ → Array F256) (zl : Zk.Level0)
    {t t' : Transcript} {rc : Zk.L0Rec} (h : Zk.ligeritoWith hs ms sch T cap p basisAt zl t = .ok (rc, t')) :
    rc.ood.size = 1 ∧ rc.challenges.size = sch.k0 := by
  unfold Zk.ligeritoWith at h
  simp only [liftTM_V, liftLM_V] at h
  obtain ⟨-, h⟩ := check_bind h
  obtain ⟨-, h⟩ := check_bind h
  obtain ⟨t₁, -, h⟩ := label_bind h
  obtain ⟨t₂, -, h⟩ := absorbF_bind h
  obtain ⟨t₃, -, h⟩ := absorbBytes_bind h
  obtain ⟨t₃', -, h⟩ := label_bind h
  obtain ⟨⟨rc', stF⟩, t₄, h₄, h⟩ := bind_ok h
  obtain ⟨rfl, rfl⟩ := pure_ok h
  rw [StateT.run] at h₄
  obtain ⟨_, _, _, -, h₄⟩ := lbind_ok h₄
  obtain ⟨_, _, _, -, h₄⟩ := lbind_ok h₄
  obtain ⟨_, _, _, -, h₄⟩ := lbind_ok h₄
  obtain ⟨_, _, _, -, h₄⟩ := lbind_ok h₄
  obtain ⟨_, _, _, -, h₄⟩ := lbind_ok h₄
  obtain ⟨_, _, _, -, h₄⟩ := lbind_ok h₄
  obtain ⟨_, _, _, -, h₄⟩ := lbind_ok h₄
  obtain ⟨_, _, _, -, h₄⟩ := lbind_ok h₄
  obtain ⟨lane, st₁₀, t₁₀, h₁₀, h₄⟩ := lbind_ok h₄
  rw [lforIn_range_eq] at h₁₀
  have hsz := lforIn_count _ (fun s => s.1.size) (by
      intro i v st tt b st' tt' hb
      obtain ⟨-, r, tt₁, -, -, hb⟩ := lsampleKWith_bind hs hb
      have hb := lmodify_bind hb
      try dsimp only at hb
      by_cases hj : i + 1 = sch.k0
      · rw [ite_eq_left (show (i + 1 == sch.k0) = true by simp [hj])] at hb
        obtain ⟨_, tt₂, -, hb⟩ := lift_bind hb
        obtain ⟨_, tt₃, -, hb⟩ := lift_bind hb
        obtain ⟨-, ρ₀, tt₄, -, -, hb⟩ := lsampleKWith_bind hs hb
        obtain ⟨-, ρ₁, tt₅, -, -, hb⟩ := lsampleKWith_bind hs hb
        try dsimp only at hb
        by_cases hρ : (!Zk.rhoIndependent (ρ₀, ρ₁)) = true
        · rw [ite_eq_left hρ] at hb
          obtain ⟨_, _, _, h0, _⟩ := lbind_ok hb
          cases h0
        rw [ite_eq_right hρ] at hb
        try simp only [pure_bind] at hb
        obtain ⟨_, tt₆, -, hb⟩ := lift_bind hb
        obtain ⟨_, tt₇, -, hb⟩ := lift_bind hb
        obtain ⟨_, tt₈, -, hb⟩ := lift_bind hb
        have hb := lmodify_bind hb
        obtain ⟨m, tt₉, -, -, hb⟩ := nextMsg_bind hb
        have hb := lmodify_bind hb
        obtain ⟨rfl, rfl, rfl⟩ := lpure_ok hb
        exact ⟨_, rfl, by simp⟩
      · rw [ite_eq_right (show ¬ (i + 1 == sch.k0) = true by simp [hj])] at hb
        try simp only [pure_bind] at hb
        obtain ⟨m, tt₉, -, -, hb⟩ := nextMsg_bind hb
        have hb := lmodify_bind hb
        obtain ⟨rfl, rfl, rfl⟩ := lpure_ok hb
        exact ⟨_, rfl, by simp⟩) sch.k0 0 _ _ _ lane st₁₀ t₁₀ h₁₀
  try dsimp only at h₄
  obtain ⟨_, _, _, -, h₄⟩ := lbind_ok h₄
  obtain ⟨_, _, _, -, h₄⟩ := lbind_ok h₄
  obtain ⟨_, _, _, -, h₄⟩ := lbind_ok h₄
  obtain ⟨⟨pos0, al0, e0⟩, _, _, -, h₄⟩ := lbind_ok h₄
  dsimp only at h₄
  obtain ⟨extra, _, _, -, h₄⟩ := lbind_ok h₄
  obtain ⟨_, _, _, -, h₄⟩ := lbind_ok h₄
  obtain ⟨_, _, _, -, h₄⟩ := lbind_ok h₄
  obtain ⟨_, _, _, -, h₄⟩ := lbind_ok h₄
  obtain ⟨_, _, _, -, h₄⟩ := lbind_ok h₄
  obtain ⟨out, st₁₈, t₁₈, -, hcont⟩ := lbind_ok h₄
  have hrec := finalProgZ_rec hcont
  subst hrec
  exact ⟨rfl, by simpa using hsz⟩

/-- **The lane phase returns** on a record of the layout's shape and at least `k0` sumcheck messages. -/
theorem ligeritoConstraints_ok {lay : Zk.PadLayout} {rec : Zk.L0Rec} {lg : LigeritoProof} {msgs : Zk.Level0Msgs}
    (ho : rec.ood.size = lay.nOod0) (hc : rec.challenges.size = lay.k0) (hm : lay.k0 ≤ lg.messages.size) :
    ∃ v, Zk.ligeritoConstraints lay rec lg msgs = .ok v := by
  unfold Zk.ligeritoConstraints
  dsimp only
  split
  · rename_i h
    simp [ho, hc] at h
    omega
  · simp only [Std.Legacy.Range.forIn_eq_forIn_range', range_size, List.forIn_pure_yield_eq_foldl]
    exact ⟨_, rfl⟩

variable {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}

/-- **Level 0's lane phase gives its constraints** wherever the masked Ligerito returns on a decoded proof: the goal
`LaneConstraints`. -/
theorem laneConstraints_goal (hLg : LigeritoZk H rel) (hD : RepDecodesZk H rel) : LaneConstraints H rel := by
  intro o i hi r hr
  obtain ⟨ho, hc⟩ := zkLigeritoWith_rec _ _ _ _ _ _ _ _ (t' := (o.lg i r).2) (rc := (o.lg i r).1) (hLg o i hi r hr)
  have hm := (zk_decode_sizes _ _ _ (hD o i hi r hr)).1.messages
  obtain ⟨v, hv⟩ := ligeritoConstraints_ok (lay := o.lay i) (lg := (o.zp i r).proof.ligerito)
    (msgs := (o.zp i r).level0) ho hc (by simp only [RunZk.lay, Zk.PadLayout.new, shapeZk] at hm ⊢; omega)
  unfold RunZk.lanes
  rw [hv]
  rfl

end FlockComplete
