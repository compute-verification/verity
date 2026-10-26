import Proofs.Flock.Claims.KS.PlainLevels
import Proofs.Flock.Claims.KS.ZkEvent
import Proofs.Flock.Claims.KS.ZkTables
import Proofs.Flock.Claims.DeployRoundsZk
import Proofs.Flock.Claims.KS.ConflictRun

/-!
# Every level's queries in the masked verifier over the hash oracle

`PlainLevels` for the masked verifier (`Zk.verifyRepWith hs` in `V`, any hash oracle `hs`): masked Ligerito draws level
`ℓ`'s query coins `2 + levRound k0 R ℓ` rounds in, the plain round and the lane phase's `ρ` pair
(`zkLigeritoWith_levels`), and level `ℓ ≥ 1`'s check passes against the proof's `ℓ − 1`-th recursive cap on its
level-`ℓ` rows (`openAt`), `2^lanes` wide as in the plain verifier.

So two rows one rep of an accepted masked run opens at one later-level position, if they differ, are two `Opening`s of
that position against one root in the run's own log, a collision there (`zk_levelClash_tiedJ`).

* `zkLigeritoWith_levels`: level `ℓ ≥ 1`'s Merkle check in an accepted masked Ligerito run.
* `zkLevelRound`, `zkLevelRowsOf`: the round of level `ℓ`'s query coins in a masked rep's stream, and the rep's
  level-`ℓ` rows at the positions of its coins.
* `zkLevels_verifyRepWith`, `zkLevels_verifyWithM`: an accepted masked rep's, and an accepted masked session's, later
  levels' Merkle checks at those rounds, on those rows.
* `zkRawLevelRowsJ`, `zk_levelRowJ_raw`, `zk_levelClash_tiedJ`: the level-`ℓ` rows table `j`'s rep `k` opens in a masked
  record, each an `Opening` in an accepted run's log, and two of them at one position that differ collide there.
-/

namespace FlockDeploy.KS

open Flock Transcript FlockSoundness FlockSoundness.Refine

local notation "𝔽" => FlockLevel3.GF128

open FlockSoundness.Discharge.ZkExec in
/-- **Every later level's queries over the oracle**, masked: in an accepted masked Ligerito run on at least two levels,
level `ℓ ≥ 1`'s Merkle check passes against the proof's `ℓ − 1`-th recursive cap, on its level-`ℓ` rows, siblings and
salts, at the positions of round `t.k + 2 + levRound k0 R ℓ`'s coins. -/
theorem zkLigeritoWith_levels (hs : Hashes Flock.V) (ms : MerkleScheme) (sch : Schedule) (T : F128)
    (cap : Array ByteArray) (p : LigeritoProof) (basisAt : Array F256 → ℕ → Array F256) (zl : Zk.Level0)
    (hn : sch.n ≠ 0) (hk0 : 1 ≤ sch.k0) (h2 : 2 ≤ sch.levels.size)
    (hq : ∀ i < sch.levels.size, (sch.levels.getD i default).q ≠ 0) {t t' : Transcript} {rc : Zk.L0Rec}
    (h : Zk.ligeritoWith hs ms sch T cap p basisAt zl t = .ok (rc, t')) (ℓ : ℕ) (h1 : 1 ≤ ℓ)
    (hℓ : ℓ < sch.levels.size) :
    merkleCheckWith (ms.opsWith hs) ms p.recursiveCaps[ℓ - 1]! (sch.levels.getD ℓ default).cap
      (sch.levels.getD ℓ default).d
      (positions (sch.levels.getD ℓ default) (t.rounds[t.k + 2 + levRound sch.k0 (sch.levels.size - 1) ℓ]!).coins)
      (openAt p (sch.levels.size - 1) ℓ).1 (openAt p (sch.levels.size - 1) ℓ).2.1
      (openAt p (sch.levels.size - 1) ℓ).2.2 (2 ^ (sch.levels.getD ℓ default).lanes) = .ok () := by
  obtain ⟨i, rfl⟩ : ∃ i, ℓ = i + 1 := ⟨ℓ - 1, by omega⟩
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
  obtain ⟨out, st₁₈, t₁₈, hloop, -⟩ := lbind_ok h₄
  rw [lforIn_range_eq] at hloop
  have er₁₇ : t₁₇.rounds = t.rounds := by
    rw [hr₁₇, k₁₆.2, k₁₅.2, hr₁₄, hr₁₃, hr₁₂, k₁₁.2, hr₁₀, k₉.2, hr₈, k₇.2, k₆.2, hr₅, k₃'.2, k₃.2, k₂.2, k₁.2]
  have ek₁₇ : t₁₇.k = t.k + (sch.k0 + 11) := by
    rw [hk₁₇, k₁₆.1, k₁₅.1, hk₁₄, hk₁₃, hk₁₂, k₁₁.1, hk₁₀, k₉.1, hk₈, k₇.1, k₆.1, hk₅, k₃'.1, k₃.1, k₂.1, k₁.1,
      ite_eq_right (show ¬ sch.k0 = 0 by omega)]
    omega
  refine lforIn_inv _ (fun i s tt => i < sch.levels.size - 1 → tt.rounds = t.rounds ∧
      tt.k = t.k + (sch.k0 + 11) + 11 * i ∧ s.2.1 = p.recursiveCaps[i]!)
    (fun i => merkleCheckWith (ms.opsWith hs) ms p.recursiveCaps[i + 1 - 1]! (sch.levels.getD (i + 1) default).cap
      (sch.levels.getD (i + 1) default).d
      (positions (sch.levels.getD (i + 1) default)
        (t.rounds[t.k + 2 + levRound sch.k0 (sch.levels.size - 1) (i + 1)]!).coins)
      (openAt p (sch.levels.size - 1) (i + 1)).1 (openAt p (sch.levels.size - 1) (i + 1)).2.1
      (openAt p (sch.levels.size - 1) (i + 1)).2.2 (2 ^ (sch.levels.getD (i + 1) default).lanes) = .ok ())
    (sch.levels.size - 1) ?_ (sch.levels.size - 1) 0 _ _ t₁₇ out st₁₈ t₁₈ (by omega)
    (fun _ => ⟨er₁₇, by omega, rfl⟩) hloop i (by omega) (by omega)
  intro i s st tt b st' tt' hiN hI hb
  obtain ⟨hrt, hkt, hcap⟩ := hI hiN
  by_cases c4 : s.1 < 4
  · rw [ite_eq_left c4] at hb; obtain ⟨_, _, _, h0, _⟩ := lbind_ok hb; cases h0
  rw [ite_eq_right c4] at hb
  obtain ⟨chs, st₁, tt₁, hf, hb⟩ := lbind_ok hb
  rw [lforIn_range_eq] at hf
  obtain ⟨hr₁, hk₁⟩ := FlockDeploy.lforIn_moves _ 1 (by
      intro j s' st' tt' b' st'' tt'' hb'
      obtain ⟨c, st₁, tt₁, hr₁, hk₁, hb'⟩ := challengeWith_bind hs hb'
      obtain ⟨rfl, rfl, rfl⟩ := lpure_ok hb'
      exact ⟨hr₁, hk₁, _, rfl⟩) 4 0 _ _ _ chs st₁ tt₁ hf
  by_cases hi : i + 1 < sch.levels.size - 1
  · rw [ite_eq_left hi] at hb
    obtain ⟨_, tt₂, h₂, hb⟩ := lift_bind hb
    have k₂ := absorbBytes_ok h₂
    obtain ⟨st₃, tt₃, hr₃, hk₃, hb⟩ := oodIntroWith_bind hs (by omega) hb
    obtain ⟨st₄, tt₄, hr₄, hk₄, hb⟩ := oodIntroWith_bind hs (by omega) hb
    obtain ⟨-, hmc, r₅, st₅, tt₅, hr₅, hk₅, hb⟩ := queryLevelWith_bind hs (hq (i + 1) (by omega)) hb
    obtain ⟨mm, tt₆, -, k₆, hb⟩ := nextMsg_bind hb
    obtain ⟨tt₇, k₇, hb⟩ := claimPow_bind hb
    obtain ⟨-, bb, tt₈, hr₈, hk₈, hb⟩ := lsampleWith_bind hs hb
    have hb := lmodify_bind hb
    by_cases c4' : s.1 - 4 + 1 < 4
    · rw [ite_eq_left c4'] at hb; obtain ⟨_, _, _, h0, _⟩ := lbind_ok hb; cases h0
    rw [ite_eq_right c4'] at hb
    obtain ⟨rfl, rfl, rfl⟩ := lpure_ok hb
    refine ⟨_, rfl, fun _ => ⟨by rw [hr₈, k₇.2, k₆.2, hr₅, hr₄, hr₃, k₂.2, hr₁, hrt], ?_, rfl⟩, ?_⟩
    · rw [hk₈, k₇.1, k₆.1, hk₅, hk₄, hk₃, k₂.1, hk₁, hkt]; omega
    · have er : tt₄.rounds = t.rounds := by rw [hr₄, hr₃, k₂.2, hr₁, hrt]
      have ek : tt₄.k = t.k + 2 + levRound sch.k0 (sch.levels.size - 1) (i + 1) := by
        rw [hk₄, hk₃, k₂.1, hk₁, hkt, levRound, ite_eq_left hi]; omega
      have ho : openAt p (sch.levels.size - 1) (i + 1) =
          (p.recursiveProofs[i]!.rows, p.recursiveProofs[i]!.siblings, p.recursiveProofs[i]!.salts) := by
        simp [openAt, hi]
      rw [ho, Nat.add_sub_cancel, ← hcap, ← er, ← ek]
      exact hmc
  · rw [ite_eq_right hi] at hb
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
    obtain ⟨-, hmc, r₃, st₃, tt₃, hr₃, hk₃, hb⟩ :=
      queryLevelWith_bind hs (hq (sch.levels.size - 1) (by omega)) hb
    obtain ⟨tt₄, k₄, hb⟩ := claimPow_bind hb
    obtain ⟨-, bb, tt₅, hr₅, hk₅, hb⟩ := lsampleWith_bind hs hb
    have hb := lmodify_bind hb
    obtain ⟨rfl, rfl, rfl⟩ := lpure_ok hb
    refine ⟨_, rfl, fun h' => absurd h' hi, ?_⟩
    have hR : sch.levels.size - 1 = i + 1 := by omega
    have er : tt₂.rounds = t.rounds := by rw [k₂.2, hr₁, hrt]
    have ek : tt₂.k = t.k + 2 + levRound sch.k0 (sch.levels.size - 1) (i + 1) := by
      rw [k₂.1, hk₁, hkt, levRound, ite_eq_right hi]; omega
    have ho : openAt p (sch.levels.size - 1) (i + 1) =
        (p.finalProof.rows, p.finalProof.siblings, p.finalProof.salts) := by
      simp [openAt, hi]
    rw [ho, Nat.add_sub_cancel, ← hcap, ← er, ← ek]
    rw [hR] at hmc
    exact hmc

/-! ## An accepted rep, and an accepted session -/

/-- **The round of level `ℓ`'s query coins** in a masked rep's stream: the replay's `m + (kLog − 6)` rounds, the ring
switch's `claims + 1`, then masked Ligerito's `2 + levRound k0 R ℓ`. -/
def zkLevelRound (m kLog k0 claims R ℓ : ℕ) : ℕ :=
  (m + (kLog - 6)) + (claims + 1) + 2 + levRound k0 R ℓ

/-- **A masked rep's opened level-`ℓ` rows** from its recorded rounds and proof bytes, at the positions of round
`zkLevelRound`'s coins, for a level `1 ≤ ℓ` of its schedule: none otherwise, or if the proof does not decode. -/
def zkLevelRowsOf (sh : Shape) (rounds : Array Round) (bytes : ByteArray) (ℓ : ℕ) : List (ℕ × List 𝔽) :=
  if 1 ≤ ℓ ∧ ℓ < sh.sched.levels.size then
    match Zk.decode sh bytes with
    | .ok zp =>
      Claims.Flock.zkRepCols (sh.sched.levels.getD ℓ default)
        ((rounds[zkLevelRound sh.m sh.kLog sh.sched.k0 sh.claims (sh.sched.levels.size - 1) ℓ]?.map
          (·.coins)).getD #[]) (openAt zp.proof.ligerito (sh.sched.levels.size - 1) ℓ).1
    | .error _ => []
  else []

open FlockSoundness.Discharge.ZkExec (shapeZK zk_schedule_seams) in
/-- **An accepted masked rep's later levels' queries sit at rounds `zkLevelRound` of its stream.** At the `--zk`
schedule of its table, a rep that `Zk.verifyRepWith hs` accepts (in `V`, any oracle) passed level `ℓ`'s Merkle check,
for every level `1 ≤ ℓ`, against the proof's `ℓ − 1`-th recursive cap on its level-`ℓ` rows at the positions of round
`zkLevelRound`'s coins, and the reader's level-`ℓ` rows (`zkLevelRowsOf`) are those positions with those rows. -/
theorem zkLevels_verifyRepWith (hs : Hashes Flock.V) (spec : SessionSpec) (st : Setup) (sch : Schedule) (tPad : ℕ)
    (sess : Session) (i r : ℕ) (bytes cap : ByteArray) (hsch : Zk.schedule st.m = .ok (sch, tPad))
    (h : Zk.verifyRepWith hs spec st sch tPad sess i r bytes = .ok cap) :
    ∃ (stream : StreamRec) (extras : Array Claim) (zp : Zk.ZkProof),
      sess.streams.find? (fun x => x.name == s!"{spec.tables[i]!}/rep{r}") = some stream ∧
      st.extra sess.points = .ok extras ∧ Zk.decode (shapeZK st sch extras) bytes = .ok zp ∧
      ∀ ℓ, 1 ≤ ℓ → ℓ < sch.levels.size →
        zkLevelRowsOf (shapeZK st sch extras) stream.rounds bytes ℓ =
          Claims.Flock.zkRepCols (sch.levels.getD ℓ default)
            (stream.rounds[zkLevelRound st.m st.kLog sch.k0 (2 + extras.size) (sch.levels.size - 1) ℓ]!).coins
            (openAt zp.proof.ligerito (sch.levels.size - 1) ℓ).1 ∧
        merkleCheckWith (st.merkle.opsWith hs) st.merkle zp.proof.ligerito.recursiveCaps[ℓ - 1]!
          (sch.levels.getD ℓ default).cap (sch.levels.getD ℓ default).d
          (positions (sch.levels.getD ℓ default)
            (stream.rounds[zkLevelRound st.m st.kLog sch.k0 (2 + extras.size) (sch.levels.size - 1) ℓ]!).coins)
          (openAt zp.proof.ligerito (sch.levels.size - 1) ℓ).1 (openAt zp.proof.ligerito (sch.levels.size - 1) ℓ).2.1
          (openAt zp.proof.ligerito (sch.levels.size - 1) ℓ).2.2 (2 ^ (sch.levels.getD ℓ default).lanes) = .ok () := by
  obtain ⟨h25, -, hsm, hk0, -, hdim, hL, -, -⟩ := zk_schedule_seams _ _ _ hsch
  have hq : ∀ j < sch.levels.size, (sch.levels.getD j default).q ≠ 0 := fun j hj => (hL j hj).q_ne_zero
  have hn : sch.n ≠ 0 := by rw [hdim.1, hsm]; omega
  have h2 : 2 ≤ sch.levels.size := hdim.2.1
  unfold Zk.verifyRepWith at h
  simp only [monadLift_self] at h
  split at h
  swap
  · cases h
  rename_i stream hstream
  obtain ⟨extras, hext, h⟩ := ebind_ok h
  obtain ⟨zp, hzp, h⟩ := ebind_ok h
  by_cases c0 : (zp.proof.params != pcsParams st.merkle sch) = true
  · rw [ite_eq_left c0] at h; obtain ⟨_, h0, _⟩ := ebind_ok h; cases h0
  rw [ite_eq_right c0] at h
  by_cases c1 : (zp.inner.root.size != 64) = true
  · rw [ite_eq_left c1] at h; obtain ⟨_, h0, _⟩ := ebind_ok h; cases h0
  rw [ite_eq_right c1] at h
  obtain ⟨⟨⟨⟩, tr⟩, hrun, -⟩ := ebind_ok h
  rw [StateT.run] at hrun
  obtain ⟨rp, t₁, h₁, hrun⟩ := bind_ok hrun
  obtain ⟨hr₁, hk₁⟩ := FlockSoundness.ZkQuery.replayWith_moves hs st _ _ _ (by omega) h₁
  obtain ⟨⟨T, sws⟩, t₂, h₂, hrun⟩ := bind_ok hrun
  obtain ⟨hr₂, hk₂⟩ := FlockSoundness.ZkQuery.ringSwitchWith_moves hs _ _ _ (by simp) h₂
  obtain ⟨rec, t₃, h₃, -⟩ := bind_ok hrun
  have er : t₂.rounds = stream.rounds := by rw [hr₂, hr₁]; rfl
  refine ⟨stream, extras, zp, hstream, hext, hzp, fun ℓ h1 hℓ => ⟨?_, ?_⟩⟩
  · have hzp' : Zk.decode (shapeZK st sch extras) bytes = .ok zp := hzp
    rw [zkLevelRowsOf, ite_eq_left ⟨h1, by simpa [shapeZK] using hℓ⟩, hzp']
    simp only [shapeZK]
    rw [Claims.Flock.coins_getElem!]
  · have hmc := zkLigeritoWith_levels hs _ _ _ _ _ _ _ hn hk0 h2 hq h₃ ℓ h1 hℓ
    have ek : t₂.k + 2 + levRound sch.k0 (sch.levels.size - 1) ℓ =
        zkLevelRound st.m st.kLog sch.k0 (2 + extras.size) (sch.levels.size - 1) ℓ := by
      rw [hk₂, hk₁]; simp [zkLevelRound, Transcript.init] <;> omega
    rw [er, ek] at hmc
    exact hmc

open FlockSoundness.Discharge.ZkExec (shapeZK) in
/-- **Every later level's query round in an accepted masked session** (`Zk.verifyWithM` in `V`, any oracle):
`zkLevels_verifyRepWith` at every table `i` and rep `r < 2` (`verifyWithM_reps`). -/
theorem zkLevels_verifyWithM (hs : Hashes Flock.V) (spec : SessionSpec) (tabs : Array Setup) (record : String)
    (proofs : Array ByteArray) (h : Zk.verifyWithM hs spec tabs record proofs = .ok ()) :
    ∃ sess : Session, Record.decodeWith hs spec record = .ok sess ∧ tabs.size = spec.tables.size ∧
      ∀ (i : ℕ) (hi : i < tabs.size), ∃ (sch : Schedule) (tPad : ℕ), Zk.schedule tabs[i].m = .ok (sch, tPad) ∧
        ∀ r < 2, ∃ (stream : StreamRec) (extras : Array Claim) (zp : Zk.ZkProof),
          sess.streams.find? (fun x => x.name == s!"{spec.tables[i]!}/rep{r}") = some stream ∧
          tabs[i].extra sess.points = .ok extras ∧ Zk.decode (shapeZK tabs[i] sch extras) proofs[2 * i + r]! = .ok zp ∧
          ∀ ℓ, 1 ≤ ℓ → ℓ < sch.levels.size →
            zkLevelRowsOf (shapeZK tabs[i] sch extras) stream.rounds proofs[2 * i + r]! ℓ =
              Claims.Flock.zkRepCols (sch.levels.getD ℓ default)
                (stream.rounds[zkLevelRound tabs[i].m tabs[i].kLog sch.k0 (2 + extras.size)
                  (sch.levels.size - 1) ℓ]!).coins (openAt zp.proof.ligerito (sch.levels.size - 1) ℓ).1 ∧
            merkleCheckWith (tabs[i].merkle.opsWith hs) tabs[i].merkle zp.proof.ligerito.recursiveCaps[ℓ - 1]!
              (sch.levels.getD ℓ default).cap (sch.levels.getD ℓ default).d
              (positions (sch.levels.getD ℓ default)
                (stream.rounds[zkLevelRound tabs[i].m tabs[i].kLog sch.k0 (2 + extras.size)
                  (sch.levels.size - 1) ℓ]!).coins)
              (openAt zp.proof.ligerito (sch.levels.size - 1) ℓ).1
              (openAt zp.proof.ligerito (sch.levels.size - 1) ℓ).2.1
              (openAt zp.proof.ligerito (sch.levels.size - 1) ℓ).2.2 (2 ^ (sch.levels.getD ℓ default).lanes) =
                .ok () := by
  obtain ⟨sess, hsess, hsz, hreps⟩ := FlockSoundness.ZkQuery.verifyWithM_reps hs spec tabs record proofs h
  refine ⟨sess, hsess, hsz, fun i hi => ?_⟩
  obtain ⟨sch, tPad, c, hsch, hr⟩ := hreps i hi
  exact ⟨sch, tPad, hsch, fun r hr2 => zkLevels_verifyRepWith hs spec tabs[i] sch tPad sess i r _ c hsch (hr r hr2)⟩

/-! ## An accepted masked run's later openings -/

theorem zkLevelRowsOf_congr (sh : Shape) {a b : Array Round} (bytes : ByteArray) (ℓ : ℕ)
    (h : ∀ j : ℕ, a[j]?.map (fun rd : Round => rd.coins) = b[j]?.map (fun rd : Round => rd.coins)) :
    zkLevelRowsOf sh a bytes ℓ = zkLevelRowsOf sh b bytes ℓ := by
  unfold zkLevelRowsOf
  rw [h]

/-- **The level-`ℓ` rows table `j`'s rep `k` opens in a masked record**: `zkLevelRowsOf` at the reader's table, on rep
`k`'s recorded rounds and proof file. -/
def zkRawLevelRowsJ (s : Claims.Seen) (j k ℓ : ℕ) : List (ℕ × List 𝔽) :=
  match runOf true s j with
  | some r =>
    match r.reps[k]? with
    | some y => zkLevelRowsOf r.shape y.1 y.2 ℓ
    | none => []
  | none => []

namespace Gen

variable [S : ZkSetup]

/-- **The level-`ℓ` rows table `j`'s rep `k` opens in a masked record**: `zkLevelRowsOf` at the reader's table, on rep
`k`'s recorded rounds and proof file. -/
def zkRawLevelRowsJ (s : Claims.Seen) (j k ℓ : ℕ) : List (ℕ × List 𝔽) :=
  match runOf true s j with
  | some r =>
    match r.reps[k]? with
    | some y => zkLevelRowsOf r.shape y.1 y.2 ℓ
    | none => []
  | none => []

/-- **Two rows one rep of an accepted masked run opens at one later-level position collide in its own log**: two of
table `j`'s rep `k`'s level-`ℓ` rows at one position that differ are two `Opening`s of that position against the
flattened `ℓ − 1`-th recursive cap of the rep's proof, each SHA-512 input of them in the run's verifier log: its verdict
accepts with SHA-512 answered within that log (`acc_within`), so level `ℓ`'s Merkle check passed on those rows at those
positions (`zkLevels_verifyWithM`), and the check, carried back to `OC`, logged them (`merkleCheck_logs`). -/
theorem zk_levelClash_tiedJ {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω}
    {r : Fin ((V true).k x) → Coin} {j k ℓ q : ℕ} {a b : List 𝔽}
    (h : (Claims.run sha512 P (V true) x ω r).out = .acc)
    (ha : (q, a) ∈ zkRawLevelRowsJ (Claims.run sha512 P (V true) x ω r).seen j k ℓ)
    (hb : (q, b) ∈ zkRawLevelRowsJ (Claims.run sha512 P (V true) x ω r).seen j k ℓ) (hab : a ≠ b) :
    Claims.TiedColl sha512 (Claims.run sha512 P (V true) x ω r).vLog := by
  obtain ⟨s, spec, sts, link, c₁, mids, f, hs, hf, hlink, hc, hv⟩ := acc_within h
  generalize hL : (Claims.run sha512 P (V true) x ω r).vLog = L at hv ⊢
  obtain ⟨hx, lv, hlv, -⟩ := Claims.Bridge.Custody.checks_ok hc
  unfold zkRawLevelRowsJ at ha hb
  split at ha
  swap
  · simp at ha
  rename_i rr hrr
  rw [hrr] at hb
  obtain ⟨st, extras, hst, hext, rfl⟩ := runOf_eqJ hf hx hs hlv hrr
  obtain ⟨hk, ha, hb⟩ : k < 2 ∧
      (q, a) ∈ zkLevelRowsOf (Claims.Flock.zkProofShape (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog
        (2 + extras.size) (Setup.ofCircuit st).merkle.digestLen (saltLenOf true (Setup.ofCircuit st)))
        (lv.streams.getD (2 * j + k) #[]) (f.proofs.getD (2 * j + k) ByteArray.empty) ℓ ∧
      (q, b) ∈ zkLevelRowsOf (Claims.Flock.zkProofShape (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog
        (2 + extras.size) (Setup.ofCircuit st).merkle.digestLen (saltLenOf true (Setup.ofCircuit st)))
        (lv.streams.getD (2 * j + k) #[]) (f.proofs.getD (2 * j + k) ByteArray.empty) ℓ := by
    rcases k with _ | _ | k
    · exact ⟨by omega, ha, hb⟩
    · exact ⟨by omega, ha, hb⟩
    · simp at ha
  have hstsj : j < sts.size := by
    rcases Array.getElem?_eq_some_iff.1 hst with ⟨h0, -⟩
    exact h0
  have hstj : sts[j] = st := by
    rcases Array.getElem?_eq_some_iff.1 hst with ⟨h0, e⟩
    exact e
  have hj : j < (tabsOf sts).size := by simpa [tabsOf] using hstsj
  have htj : (tabsOf sts)[j] = Setup.ofCircuit st := by rw [tabsOf_getElem hj]; simp only [hstj]
  replace hv : Zk.verifyWithM (hashesIn (L.map Prod.fst)) spec (tabsOf sts) f.record f.proofs = .ok () := hv
  obtain ⟨sess, hdec, hsz, hreps⟩ :=
    zkLevels_verifyWithM (hashesIn (L.map Prod.fst)) spec (tabsOf sts) f.record f.proofs hv
  obtain ⟨sch, tPad, hsch, hrep⟩ := hreps j hj
  obtain ⟨stream, extras', zp, hstream, hext', hzp, hlev⟩ := hrep k hk
  simp only [htj] at hsch hext' hzp hlev
  have hdec' : Record.decode spec f.record = .ok sess := by
    rw [← With.decodeWith_std]
    exact (With.decodeWith_sim sub (sub_hashes (L.map Prod.fst)) spec f.record) sess hdec
  obtain ⟨-, hpts, -, hstreams⟩ := Claims.Bridge.Custody.decode_live hdec' hlv
  rw [hpts, hext] at hext'
  cases hext'
  have htab : j < spec.tables.size := by rw [← hsz]; exact hj
  obtain ⟨hk', hrs⟩ := Claims.Bridge.Custody.repStreams_getElem spec htab hk
  obtain ⟨y, hy, hfa⟩ := hstreams (2 * j + k) hk'
  rw [hrs] at hy
  have hys : y = stream := Option.some.inj (hy.symm.trans hstream)
  subst hys
  have hcoins := forall₂_coins (fun _ _ hab => hab.2.1) hfa
  have hzs : Claims.Flock.zkSched (Setup.ofCircuit st).m = (sch, tPad) := by
    simp only [Claims.Flock.zkSched]
    rw [hsch]
  have hsh : Claims.Flock.zkProofShape (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog (2 + extras.size)
      (Setup.ofCircuit st).merkle.digestLen (saltLenOf true (Setup.ofCircuit st)) =
      FlockSoundness.Discharge.ZkExec.shapeZK (Setup.ofCircuit st) sch extras := by
    simp only [Claims.Flock.zkProofShape, hzs, FlockSoundness.Discharge.ZkExec.shapeZK, saltLenOf, ite_true]
  have hsched : (Claims.Flock.zkProofShape (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog (2 + extras.size)
      (Setup.ofCircuit st).merkle.digestLen (saltLenOf true (Setup.ofCircuit st))).sched = sch := by
    rw [hsh]
    rfl
  have hℓ : 1 ≤ ℓ ∧ ℓ < sch.levels.size := by
    by_contra hn
    rw [zkLevelRowsOf, ite_eq_right (by rw [hsched]; exact hn)] at ha
    simp at ha
  obtain ⟨hent, hmc⟩ := hlev ℓ hℓ.1 hℓ.2
  have hrowsEq : zkLevelRowsOf (Claims.Flock.zkProofShape (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog
      (2 + extras.size) (Setup.ofCircuit st).merkle.digestLen (saltLenOf true (Setup.ofCircuit st)))
      (lv.streams.getD (2 * j + k) #[]) (f.proofs.getD (2 * j + k) ByteArray.empty) ℓ =
      Claims.Flock.zkRepCols (sch.levels.getD ℓ default)
        (y.rounds[zkLevelRound (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog sch.k0 (2 + extras.size)
          (sch.levels.size - 1) ℓ]!).coins (openAt zp.proof.ligerito (sch.levels.size - 1) ℓ).1 := by
    rw [← hent, hsh]
    have e1 : lv.streams.getD (2 * j + k) #[] = lv.streams[2 * j + k]! := by
      rw [show (#[] : Array Flock.Round) = default from rfl, getD_eq_getElem!]
    have e2 : f.proofs.getD (2 * j + k) ByteArray.empty = f.proofs[2 * j + k]! := by
      rw [show ByteArray.empty = default from rfl, getD_eq_getElem!]
    rw [e1, e2]
    refine zkLevelRowsOf_congr _ _ _ fun i => ?_
    rw [← Array.getElem?_toList, ← Array.getElem?_toList]
    exact (hcoins i).symm
  rw [hrowsEq] at ha hb
  obtain ⟨i, hi, -, hiq, hia⟩ := mem_zkRepCols ha
  obtain ⟨i', hi', -, hiq', hib⟩ := mem_zkRepCols hb
  have hms : (Setup.ofCircuit st).merkle = MerkleScheme.hm96Sha512 := by
    have := checks_hm96 hc (setupOf_some hs).2 (setupOf_zk hs) j hstsj
    rw [hstj] at this
    exact this
  rw [hms] at hmc
  have hruns := With.merkleCheckWith_sim (back (L.map Prod.fst))
    (With.opsWith_sim (back (L.map Prod.fst)) (back_hashes (L.map Prod.fst)) MerkleScheme.hm96Sha512)
    MerkleScheme.hm96Sha512 _ _ _ _ _ _ _ _ () hmc
  rw [opsWith_hm96] at hruns
  obtain ⟨hpure, hlog⟩ := merkleCheck_logs hruns
  obtain ⟨hcs, hcw⟩ := merkleCheck_cap hpure
  have hop : ∀ i < (positions (sch.levels.getD ℓ default) (y.rounds[zkLevelRound (Setup.ofCircuit st).m
      (Setup.ofCircuit st).kLog sch.k0 (2 + extras.size) (sch.levels.size - 1) ℓ]!).coins).size,
      Opening MerkleScheme.hm96Sha512 (sch.levels.getD ℓ default) (2 ^ (sch.levels.getD ℓ default).lanes)
        (zp.proof.ligerito.recursiveCaps[ℓ - 1]!.foldl (· ++ ·) ByteArray.empty) L
        (positions (sch.levels.getD ℓ default) (y.rounds[zkLevelRound (Setup.ofCircuit st).m
          (Setup.ofCircuit st).kLog sch.k0 (2 + extras.size) (sch.levels.size - 1) ℓ]!).coins)[i]!
        (openAt zp.proof.ligerito (sch.levels.size - 1) ℓ).1[i]! := fun i hi =>
    ⟨Or.inl rfl, zp.proof.ligerito.recursiveCaps[ℓ - 1]!,
      (openAt zp.proof.ligerito (sch.levels.size - 1) ℓ).2.2.getD i ByteArray.empty,
      sibsOf (openAt zp.proof.ligerito (sch.levels.size - 1) ℓ).2.1
        ((sch.levels.getD ℓ default).d - (sch.levels.getD ℓ default).cap) i,
      hcs, hcw, rfl, merkleCheck_opens _ _ _ _ _ _ _ _ _ hpure i hi, merkleCheck_sibs _ _ _ _ _ _ _ _ _ hpure i,
      by rw [opensInsOf_hm96]; exact hlog i hi⟩
  have ho₁ := hop i hi
  have ho₂ := hop i' hi'
  rw [hiq] at ho₁
  rw [hiq'] at ho₂
  exact tiedColl_of_append_self (tiedColl_of_openings ho₁ ho₂ fun e => hab (by rw [hia, hib, e]))

end Gen

/-- **Two rows one rep of an accepted masked run opens at one later-level position collide in its own log**: two of
table `j`'s rep `k`'s level-`ℓ` rows at one position that differ are two `Opening`s of that position against the
flattened `ℓ − 1`-th recursive cap of the rep's proof, each SHA-512 input of them in the run's verifier log: its verdict
accepts with SHA-512 answered within that log (`acc_within`), so level `ℓ`'s Merkle check passed on those rows at those
positions (`zkLevels_verifyWithM`), and the check, carried back to `OC`, logged them (`merkleCheck_logs`). -/
theorem zk_levelClash_tiedJ {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω}
    {r : Fin ((V true).k x) → Coin} {j k ℓ q : ℕ} {a b : List 𝔽}
    (h : (Claims.run sha512 P (V true) x ω r).out = .acc)
    (ha : (q, a) ∈ zkRawLevelRowsJ (Claims.run sha512 P (V true) x ω r).seen j k ℓ)
    (hb : (q, b) ∈ zkRawLevelRowsJ (Claims.run sha512 P (V true) x ω r).seen j k ℓ) (hab : a ≠ b) :
    Claims.TiedColl sha512 (Claims.run sha512 P (V true) x ω r).vLog :=
  Gen.zk_levelClash_tiedJ (S := S₀) (Ω := Ω) (P := P) (x := x) (ω := ω) (r := r) (j := j) (k := k) (ℓ := ℓ) (q := q)
    (a := a) (b := b) (h := h) (ha := ha) (hb := hb) (hab := hab)

end FlockDeploy.KS
