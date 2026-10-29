import Proofs.Flock.Claims.ZkQueryRound
import Proofs.Flock.Claims.DeployCode
import Proofs.Flock.Soundness.Refine.Seam

/-!
# Level 0's query round in the plain verifier over the hash oracle

`plainQueryRound m kLog k0 claims` (`Claims.Flock.plainQueryRound`) is the round of a plain rep's stream whose coins are
level 0's query positions. As `Proofs.Flock.Claims.ZkQueryRound` does for the masked verifier, it is proved here from
the plain verifier's own transcript, `verifyRepWith hs` run in `V` with any hash oracle `hs`:

* zerocheck consumes `m − 3` rounds (`zerocheckWith_moves`), and lincheck `kLog − 3` on a circuit fold that refuses a
  point of the wrong length (`lincheckWith_moves`; the statement's fold, `Stmt.folded_size`), the ring switch
  `claims + 1` (`ZkQuery.ringSwitchWith_moves`);
* Ligerito draws level 0's query coins after `k0 + 6` rounds of its own: the OOD point and its coin, the `k0` lane
  challenges, and level 1's two OOD intros (`plainLigeritoWith_query`);
* so an accepted rep passed level 0's Merkle check on its proof's rows at the positions of round `plainQueryRound` of
  its stream, against the committed `root_B` (`plainQueryRound_verifyRepWith`), and the reader's columns
  (`plainRepColsOf`) are those positions with those rows, for every rep of every table of an accepted session
  (`plainQueryRound_verifyWithM`).
-/

namespace FlockDeploy.KS

open Flock Transcript FlockSoundness FlockSoundness.Refine FlockSoundness.Discharge.Exec FlockSoundness.ZkQuery

/-! ## Phases -/

/-- **Zerocheck's rounds**: `m − 3` (the skip and outer coins, `z`, and `m − 6` sumcheck rounds), at `m ≥ 14`. -/
theorem zerocheckWith_moves (hs : Hashes Flock.V) (m : ℕ) (digest cap : ByteArray) (p : ZerocheckProof) (hm : 14 ≤ m)
    {t t' : Flock.Transcript} {zc : ZcOut} (h : bindAndZerocheckWith hs m digest cap p t = .ok (zc, t')) :
    t'.rounds = t.rounds ∧ t'.k = t.k + (m - 3) := by
  unfold bindAndZerocheckWith at h
  simp only [With.liftTM_V] at h
  obtain ⟨t₁, k₁, h⟩ := label_bind h
  obtain ⟨t₂, k₂, h⟩ := absorbBytes_bind h
  obtain ⟨t₃, k₃, h⟩ := absorbBytes_bind h
  obtain ⟨t₄, k₄, h⟩ := label_bind h
  obtain ⟨_, t₅, h₅, h⟩ := bind_ok h
  obtain ⟨-, -, hr₅, hk₅⟩ := sampleVecWith_ok hs (by decide) h₅
  obtain ⟨_, t₆, h₆, h⟩ := bind_ok h
  obtain ⟨-, -, hr₆, hk₆⟩ := sampleVecWith_ok hs (by omega) h₆
  obtain ⟨t₇, k₇, h⟩ := absorbSlice_bind h
  obtain ⟨t₈, k₈, h⟩ := absorbSlice_bind h
  obtain ⟨_, t₉, h₉, h⟩ := bind_ok h
  obtain ⟨-, -, hr₉, hk₉⟩ := sampleWith_ok hs h₉
  obtain ⟨-, h⟩ := check_bind h
  obtain ⟨s, t₁₀, hloop, h⟩ := bind_ok h
  rw [forIn_range_eq] at hloop
  obtain ⟨hr₁₀, hk₁₀⟩ := forIn_moves _ 1 (by
      intro i s tt b tt' hb
      obtain ⟨u₁, f₁, hb⟩ := absorbF_bind hb
      obtain ⟨u₂, f₂, hb⟩ := absorbF_bind hb
      obtain ⟨_, u₃, g₃, hb⟩ := bind_ok hb
      obtain ⟨-, -, hr₃, hk₃⟩ := sampleWith_ok hs g₃
      obtain ⟨rfl, rfl⟩ := pure_ok hb
      exact ⟨⟨_, rfl⟩, by rw [hr₃, f₂.2, f₁.2], by rw [hk₃, f₂.1, f₁.1]⟩) (m - 6) 0 _ _ _ _ hloop
  obtain ⟨-, h⟩ := check_bind h
  obtain ⟨t₁₁, k₁₁, h⟩ := absorbF_bind h
  obtain ⟨t₁₂, k₁₂, h⟩ := absorbF_bind h
  obtain ⟨-, rfl⟩ := pure_ok h
  refine ⟨by rw [k₁₂.2, k₁₁.2, hr₁₀, hr₉, k₈.2, k₇.2, hr₆, hr₅, k₄.2, k₃.2, k₂.2, k₁.2], ?_⟩
  rw [k₁₂.1, k₁₁.1, hk₁₀, hk₉, k₈.1, k₇.1, hk₆, hk₅, k₄.1, k₃.1, k₂.1, k₁.1]
  omega

/-- **Lincheck's rounds**: `kLog − 3` (`α`, `β`, `kLog − 6` sumcheck rounds and `r′`), on a circuit fold that reads a
point of `kLog − 6` coordinates; so `kLog ≥ 6`. -/
theorem lincheckWith_moves (hs : Hashes Flock.V) (c : CircuitFold) (zc : ZcOut) (p : LincheckProof)
    (hf : ∀ a b lw rho ts v, c.folded a b lw rho ts = .ok v → rho.size + 6 = c.kLog)
    {t t' : Flock.Transcript} {lc : LcOut} (h : lincheckWith hs c zc p t = .ok (lc, t')) :
    6 ≤ c.kLog ∧ t'.rounds = t.rounds ∧ t'.k = t.k + (c.kLog - 3) := by
  unfold lincheckWith at h
  simp only [With.liftTM_V] at h
  obtain ⟨t₁, k₁, h⟩ := label_bind h
  obtain ⟨α₀, t₂, h₂, h⟩ := bind_ok h
  obtain ⟨-, -, hr₂, hk₂⟩ := sampleWith_ok hs h₂
  obtain ⟨β₀, t₃, h₃, h⟩ := bind_ok h
  obtain ⟨-, -, hr₃, hk₃⟩ := sampleWith_ok hs h₃
  obtain ⟨s₂, t₄, hloop, h⟩ := bind_ok h
  rw [forIn_range_eq] at hloop
  obtain ⟨hr₄, hk₄⟩ := forIn_moves _ 1 (by
      intro j s tt b tt' hb
      obtain ⟨u₁, f₁, hb⟩ := absorbF_bind hb
      obtain ⟨u₂, f₂, hb⟩ := absorbF_bind hb
      obtain ⟨_, u₃, g₃, hb⟩ := bind_ok hb
      obtain ⟨-, -, hr₃, hk₃⟩ := sampleWith_ok hs g₃
      obtain ⟨rfl, rfl⟩ := pure_ok hb
      exact ⟨⟨_, rfl⟩, by rw [hr₃, f₂.2, f₁.2], by rw [hk₃, f₂.1, f₁.1]⟩) (c.kLog - 6) 0 _ _ _ _ hloop
  cases hv : c.folded α₀ β₀ (skipWeights zc.z) (zc.rho.extract 0 (c.kLog - 6)) s₂.2 with
  | error e =>
    simp only [hv] at h
    obtain ⟨_, _, h₀, _⟩ := bind_ok h
    cases h₀
  | ok v =>
    have h6 : 6 ≤ c.kLog := by
      have := hf _ _ _ _ _ _ hv
      simp only [Array.size_extract] at this
      omega
    simp only [hv, pure_bind] at h
    obtain ⟨-, h⟩ := check_bind h
    obtain ⟨t₆, k₆, h⟩ := absorbSlice_bind h
    obtain ⟨-, h⟩ := check_bind h
    obtain ⟨_, t₇, h₇, h⟩ := bind_ok h
    obtain ⟨-, -, hr₇, hk₇⟩ := sampleWith_ok hs h₇
    obtain ⟨-, rfl⟩ := pure_ok h
    refine ⟨h6, by rw [hr₇, k₆.2, hr₄, hr₃, hr₂, k₁.2], ?_⟩
    rw [hk₇, k₆.1, hk₄, hk₃, hk₂, k₁.1]
    omega

/-- The statement's circuit fold refuses a point whose length is not `kLog − 6`. -/
theorem Stmt.folded_size (st : Stmt) {a b : F128} {lw rho ts v : Array F128} (h : st.folded a b lw rho ts = .ok v) :
    rho.size + 6 = st.c.kLog := by
  unfold Stmt.folded at h
  split at h
  · unfold Stmt.foldedPartial at h
    by_cases c : (rho.size + 6 != st.c.kLog || lw.size != 64) = true
    · rw [ite_eq_left c] at h; obtain ⟨_, h0, _⟩ := ebind_ok h; cases h0
    · simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, not_or, not_not] at c
      exact c.1
  · obtain ⟨w, hw, -⟩ := ebind_ok h
    unfold Stmt.fold at hw
    by_cases c : (rho.size + 6 != st.c.kLog || lw.size != 64) = true
    · rw [ite_eq_left c] at hw; obtain ⟨_, h0, _⟩ := ebind_ok hw; cases h0
    · simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, not_or, not_not] at c
      exact c.1

/-! ## Ligerito up to level 0's queries -/

/-- A Ligerito loop whose every step yields and consumes `c` rounds consumes `n · c` rounds over `n` steps. -/
theorem lforIn_moves {σ : Type} (body : ℕ → σ → LM (ForInStep σ)) (c : ℕ)
    (hbody : ∀ i s st t b st' t', body i s st t = .ok ((b, st'), t') →
      (∃ s', b = .yield s') ∧ t'.rounds = t.rounds ∧ t'.k = t.k + c) :
    ∀ (n a : ℕ) (s : σ) (st : LState) (t : Flock.Transcript) (out : σ) (st' : LState) (t' : Flock.Transcript),
      (forIn (List.range' a n) s body : LM σ) st t = .ok ((out, st'), t') → t'.rounds = t.rounds ∧ t'.k = t.k + n * c
  | 0, a, s, st, t, out, st', t', h => by
      simp only [List.range'_zero, List.forIn_nil] at h
      obtain ⟨rfl, rfl, rfl⟩ := lpure_ok h
      simp
  | n + 1, a, s, st, t, out, st', t', h => by
      rw [List.range'_succ, List.forIn_cons] at h
      obtain ⟨b, st₁, t₁, h₁, h₂⟩ := lbind_ok h
      obtain ⟨⟨s₁, rfl⟩, hr₁, hk₁⟩ := hbody a s st t b st₁ t₁ h₁
      obtain ⟨hr, hk⟩ := lforIn_moves body c hbody n (a + 1) s₁ st₁ t₁ out st' t' h₂
      exact ⟨hr.trans hr₁, by rw [hk, hk₁, Nat.succ_mul]; omega⟩

/-- `challengeWith`: a lane challenge, one round. -/
theorem challengeWith_moves (hs : Hashes Flock.V) (p : LigeritoProof) {st st' : LState} {t t' : Flock.Transcript}
    {r : F256} (h : challengeWith hs p st t = .ok ((r, st'), t')) : t'.rounds = t.rounds ∧ t'.k = t.k + 1 := by
  unfold challengeWith at h
  simp only [With.liftW_V, With.liftLM_V] at h
  obtain ⟨r, tR, hR, h⟩ := lift_bind h
  obtain ⟨-, hrR, hkR⟩ := sampleKWith_ok hs hR
  have h := lmodify_bind h
  obtain ⟨m, t₉, -, k₉, h⟩ := nextMsg_bind h
  have h := lmodify_bind h
  obtain ⟨rfl, rfl, rfl⟩ := lpure_ok h
  exact ⟨k₉.2.trans hrR, by rw [k₉.1, hkR]⟩

/-- **Level 0's queries over the oracle**, plain: the Merkle check passes on the proof's level-0 rows at the positions
of the query coins, round `t.k`'s. -/
theorem queryLevelWith_mc (hs : Hashes Flock.V) {ms : MerkleScheme} {p : LigeritoProof} {lv : Flock.Level}
    {cap : Array ByteArray} {rows : Array (Array F128)} {sibs salts : Array ByteArray} {fold : Array F256}
    (hq : lv.q ≠ 0) {β : Type} {f : Array ℕ × Array F128 × F256 → LM β} {st st' : LState} {t t' : Flock.Transcript}
    {b : β} (h : (queryLevelWith hs ms p lv cap rows sibs salts fold >>= f) st t = .ok ((b, st'), t')) :
    t.k < t.rounds.size ∧
      merkleCheckWith (ms.opsWith hs) ms cap lv.cap lv.d (positions lv (t.rounds[t.k]!).coins) rows sibs salts
        (2 ^ lv.lanes) = .ok () := by
  obtain ⟨r, st₁, t₁, h₁, h₂⟩ := lbind_ok h
  unfold queryLevelWith at h₁
  simp only [With.liftW_V, With.liftLM_V] at h₁
  have h₁ := lget_bind h₁
  obtain ⟨qn, _, h₁⟩ := nonceFrom_bind h₁
  obtain ⟨cn, _, h₁⟩ := nonceFrom_bind h₁
  have h₁ := lset_bind h₁
  obtain ⟨_, t₃, h₃, h₁⟩ := lift_bind h₁
  obtain ⟨t₃', k₃, h₃'⟩ := pow_bind (f := fun _ => pure ()) (by rw [bind_pure]; exact h₃)
  obtain ⟨rfl, rfl⟩ := pure_ok h₃'
  obtain ⟨coins, t₄, h₄, h₁⟩ := lift_bind h₁
  have hlt := squeezeWith_lt hs hq h₄
  obtain ⟨-, hc₄, -, -⟩ := sampleVecWith_ok hs hq h₄
  subst hc₄
  obtain ⟨_, t₅, h₅, h₁⟩ := lift_bind h₁
  obtain ⟨alpha, t₆, h₆, h₁⟩ := lift_bind h₁
  cases hmc : merkleCheckWith (ms.opsWith hs) ms cap lv.cap lv.d (positions lv (t₃.rounds[t₃.k]!).coins) rows sibs
      salts (2 ^ lv.lanes) with
  | error e => simp only [hmc] at h₁; obtain ⟨_, _, _, h₀, _⟩ := lbind_ok h₁; cases h₀
  | ok u =>
    rw [← k₃.2, ← k₃.1]
    exact ⟨hlt, hmc⟩

/-- **Ligerito's rounds before level 0's queries**, plain: `k0 + 6`. The OOD point and its coin, the `k0` lane
challenges, and level 1's two OOD intros, two rounds each; so level 0's Merkle check passes at the positions of round
`t.k + k0 + 6`'s coins. -/
theorem plainLigeritoWith_query (hs : Hashes Flock.V) (ms : MerkleScheme) (sch : Flock.Schedule) (T : F128)
    (cap : Array ByteArray) (p : LigeritoProof) (basisAt : Array F256 → ℕ → Array F256) {t t' : Flock.Transcript}
    {u : Unit} (hn : sch.n ≠ 0) (hq : (sch.levels.getD 0 default).q ≠ 0)
    (h : ligeritoWith hs ms sch T cap p basisAt t = .ok (u, t')) :
    t.k + sch.k0 + 6 < t.rounds.size ∧
      merkleCheckWith (ms.opsWith hs) ms cap (sch.levels.getD 0 default).cap (sch.levels.getD 0 default).d
        (positions (sch.levels.getD 0 default) (t.rounds[t.k + sch.k0 + 6]!).coins) p.initialProof.rows
        p.initialProof.siblings p.initialProof.salts (2 ^ (sch.levels.getD 0 default).lanes) = .ok () := by
  unfold ligeritoWith at h
  simp only [With.liftTM_V, With.liftW_V, With.liftLM_V] at h
  obtain ⟨-, h⟩ := check_bind h
  obtain ⟨t₁, k₁, h⟩ := label_bind h
  obtain ⟨t₂, k₂, h⟩ := absorbF_bind h
  obtain ⟨t₃, k₃, h⟩ := absorbBytes_bind h
  obtain ⟨⟨rc', stF⟩, t₄, h₄, -⟩ := bind_ok h
  rw [StateT.run] at h₄
  obtain ⟨z, t₅, h₅, h₄⟩ := lift_bind h₄
  obtain ⟨-, -, hr₅, hk₅⟩ := sampleVecWith_ok hs hn h₅
  obtain ⟨y₀, -, h₄⟩ := nextOod_bind h₄
  obtain ⟨_, t₆, h₆, h₄⟩ := lift_bind h₄
  have k₆ := absorbF_ok h₆
  obtain ⟨t₇, k₇, h₄⟩ := claimPow_bind h₄
  obtain ⟨bc, t₈, h₈, h₄⟩ := lift_bind h₄
  obtain ⟨-, -, hr₈, hk₈⟩ := sampleWith_ok hs h₈
  have h₄ := lmodify_bind h₄
  obtain ⟨m₀, t₉, -, k₉, h₄⟩ := nextMsg_bind h₄
  have h₄ := lmodify_bind h₄
  obtain ⟨lane, st₁₀, t₁₀, h₁₀, h₄⟩ := lbind_ok h₄
  rw [lforIn_range_eq] at h₁₀
  obtain ⟨hr₁₀, hk₁₀⟩ := lforIn_moves _ 1 (by
      intro i v st tE bv st' tE' hb
      obtain ⟨r, st₁, t₁, h₁, hb⟩ := lbind_ok hb
      obtain ⟨hr₁, hk₁⟩ := challengeWith_moves hs p h₁
      obtain ⟨rfl, rfl, rfl⟩ := lpure_ok hb
      exact ⟨⟨_, rfl⟩, hr₁, hk₁⟩) sch.k0 0 _ _ t₉ _ _ t₁₀ h₁₀
  try dsimp only at h₄
  obtain ⟨_, t₁₁, h₁₁, h₄⟩ := lift_bind h₄
  have k₁₁ := absorbBytes_ok h₁₁
  obtain ⟨st₁₂, t₁₂, hr₁₂, hk₁₂, h₄⟩ := oodIntroWith_bind hs (show sch.n - sch.k0 + 1 ≠ 0 by omega) h₄
  obtain ⟨st₁₃, t₁₃, hr₁₃, hk₁₃, h₄⟩ := oodIntroWith_bind hs (show sch.n - sch.k0 + 1 ≠ 0 by omega) h₄
  have hmc := queryLevelWith_mc hs hq h₄
  have er : t₁₃.rounds = t.rounds := by
    rw [hr₁₃, hr₁₂, k₁₁.2, hr₁₀, k₉.2, hr₈, k₇.2, k₆.2, hr₅, k₃.2, k₂.2, k₁.2]
  have ek : t₁₃.k = t.k + sch.k0 + 6 := by
    rw [hk₁₃, hk₁₂, k₁₁.1, hk₁₀, k₉.1, hk₈, k₇.1, k₆.1, hk₅, k₃.1, k₂.1, k₁.1]
    omega
  rw [er, ek] at hmc
  exact hmc

/-! ## An accepted rep, and an accepted session -/

theorem getD_zero_mem {α : Type} [Inhabited α] {xs : Array α} (h : 0 < xs.size) : xs.getD 0 default ∈ xs := by
  unfold Array.getD
  rw [dite_eq_left h]
  exact Array.getElem_mem h

/-- **An accepted plain rep's level-0 queries sit at round `plainQueryRound` of its stream.** At the table's `fast100`
schedule, on a circuit fold that refuses a point of the wrong length, a rep that `verifyRepWith hs` accepts (in `V`, any
oracle) has a stream with that round, and level 0's Merkle check passed against the committed `root_B` on the proof's
opened rows at the positions of that round's coins. -/
theorem plainQueryRound_verifyRepWith (hs : Hashes Flock.V) (spec : SessionSpec) (st : Setup) (sch : Flock.Schedule)
    (sess : RunRecord) (i r : ℕ) (bytes cap : ByteArray) (hsch : fast100 st.m = some sch)
    (hf : ∀ a b lw rho ts v, st.fold.folded a b lw rho ts = .ok v → rho.size + 6 = st.fold.kLog)
    (hkl : st.fold.kLog = st.kLog) (h : verifyRepWith hs spec st sch sess i r bytes = .ok cap) :
    ∃ (stream : StreamRec) (extras : Array Flock.Claim) (p : Proof),
      sess.streams.find? (fun x => x.name == s!"{spec.tables[i]!}/rep{r}") = some stream ∧
      st.extra sess.points = .ok extras ∧ decodeProof (shapeOf st sch extras.size) bytes = .ok p ∧
      Claims.Flock.plainQueryRound st.m st.kLog sch.k0 (2 + extras.size) < stream.rounds.size ∧
      merkleCheckWith (st.merkle.opsWith hs) st.merkle p.cap (sch.levels.getD 0 default).cap
        (sch.levels.getD 0 default).d
        (positions (sch.levels.getD 0 default)
          (stream.rounds[Claims.Flock.plainQueryRound st.m st.kLog sch.k0 (2 + extras.size)]!).coins)
        p.ligerito.initialProof.rows p.ligerito.initialProof.siblings p.ligerito.initialProof.salts
        (2 ^ (sch.levels.getD 0 default).lanes) = .ok () ∧
      cap = p.cap.foldl (· ++ ·) ByteArray.empty ∧ cap = sess.rootB[i]! := by
  have h22 := exec_fast100_ge hsch
  obtain ⟨hdim, hsm⟩ := fast100_dims _ _ hsch
  have hq : (sch.levels.getD 0 default).q ≠ 0 :=
    (fast100_level _ _ hsch _ (getD_zero_mem (by have := hdim.2.1; omega))).q_ne_zero
  have hn : sch.n ≠ 0 := by rw [hdim.1, hsm]; omega
  unfold verifyRepWith at h
  simp only [monadLift_self] at h
  split at h
  swap
  · cases h
  rename_i stream hstream
  obtain ⟨extras, hext, h⟩ := ebind_ok h
  obtain ⟨p, hp, h⟩ := ebind_ok h
  by_cases c0 : (p.params != pcsParams st.merkle sch) = true
  · rw [ite_eq_left c0] at h; obtain ⟨_, h0, _⟩ := ebind_ok h; cases h0
  rw [ite_eq_right c0] at h
  obtain ⟨⟨⟨⟩, tr⟩, hrun, h⟩ := ebind_ok h
  dsimp only at h
  by_cases c2 : (tr.k != stream.rounds.size) = true
  · rw [ite_eq_left c2] at h; obtain ⟨_, h0, _⟩ := ebind_ok h; cases h0
  rw [ite_eq_right c2] at h
  by_cases c3 : (p.cap.foldl (· ++ ·) ByteArray.empty != sess.rootB[i]!) = true
  · rw [ite_eq_left c3] at h; obtain ⟨_, h0, _⟩ := ebind_ok h; cases h0
  rw [ite_eq_right c3] at h
  simp only [pure, Except.pure, Except.ok.injEq] at h
  subst h
  have hb : (p.cap.foldl (· ++ ·) ByteArray.empty == sess.rootB[i]!) = true := by
    cases hb : (p.cap.foldl (· ++ ·) ByteArray.empty == sess.rootB[i]!)
    · exact absurd (by simp [bne, hb]) c3
    · rfl
  rw [StateT.run] at hrun
  obtain ⟨zc, t₁, h₁, hrun⟩ := bind_ok hrun
  obtain ⟨hr₁, hk₁⟩ := zerocheckWith_moves hs _ _ _ _ (by omega) h₁
  obtain ⟨lc, t₂, h₂, hrun⟩ := bind_ok hrun
  obtain ⟨h6, hr₂, hk₂⟩ := lincheckWith_moves hs _ _ _ hf h₂
  rw [hkl] at h6
  obtain ⟨⟨T, sws⟩, t₃, h₃, hrun⟩ := bind_ok hrun
  have hsz2 : (abcClaims st.m st.kLog zc lc).size = 2 := rfl
  obtain ⟨hr₃, hk₃⟩ := ringSwitchWith_moves hs _ _ _ (by rw [Array.size_append, hsz2]; omega) h₃
  obtain ⟨hlt, hmc⟩ := plainLigeritoWith_query hs _ _ _ _ _ _ hn hq hrun
  have er : t₃.rounds = stream.rounds := by rw [hr₃, hr₂, hr₁]; rfl
  have ek : t₃.k + sch.k0 + 6 = Claims.Flock.plainQueryRound st.m st.kLog sch.k0 (2 + extras.size) := by
    rw [hk₃, hk₂, hk₁, hkl, Array.size_append, hsz2]
    simp [Claims.Flock.plainQueryRound, Transcript.init]; omega
  rw [er, ek] at hlt hmc
  exact ⟨stream, extras, p, hstream, hext, hp, hlt, hmc, rfl, bytes_eq_of_beq hb⟩

/-- `plainProofShape` at a table's `fast100` schedule is the shape `verifyRep` decodes its proofs at. -/
theorem plainProofShape_eq (st : Setup) (sch : Flock.Schedule) (extras : Array Flock.Claim)
    (hsch : fast100 st.m = some sch) :
    Claims.Flock.plainProofShape st.m st.kLog (2 + extras.size) st.merkle.digestLen
      (if st.merkle.saltBytes = 0 && !st.openedSalts then none else some st.merkle.saltBytes) =
      shapeOf st sch extras.size := by
  simp only [Claims.Flock.plainProofShape, Claims.Flock.plainSched, hsch, Option.getD_some, shapeOf]

/-- **The reader's columns of an accepted plain rep are the verifier's openings**: `plainRepColsOf` on the rep's stream
and proof file is the proof's level-0 rows at the positions of round `plainQueryRound`'s coins, the positions and rows
on which level 0's Merkle check passed against the committed `root_B` (`plainQueryRound_verifyRepWith`). -/
theorem plainRepColsOf_verifyRepWith (hs : Hashes Flock.V) (spec : SessionSpec) (st : Setup) (sch : Flock.Schedule)
    (sess : RunRecord) (i r : ℕ) (bytes cap : ByteArray) (hsch : fast100 st.m = some sch)
    (hf : ∀ a b lw rho ts v, st.fold.folded a b lw rho ts = .ok v → rho.size + 6 = st.fold.kLog)
    (hkl : st.fold.kLog = st.kLog) (h : verifyRepWith hs spec st sch sess i r bytes = .ok cap) :
    ∃ (stream : StreamRec) (extras : Array Flock.Claim) (p : Proof),
      sess.streams.find? (fun x => x.name == s!"{spec.tables[i]!}/rep{r}") = some stream ∧
      st.extra sess.points = .ok extras ∧ decodeProof (shapeOf st sch extras.size) bytes = .ok p ∧
      Claims.Flock.plainQueryRound st.m st.kLog sch.k0 (2 + extras.size) < stream.rounds.size ∧
      Claims.Flock.plainRepColsOf (Claims.Flock.plainProofShape st.m st.kLog (2 + extras.size) st.merkle.digestLen
          (if st.merkle.saltBytes = 0 && !st.openedSalts then none else some st.merkle.saltBytes)) stream.rounds bytes =
        Claims.Flock.zkRepCols (sch.levels.getD 0 default)
          (stream.rounds[Claims.Flock.plainQueryRound st.m st.kLog sch.k0 (2 + extras.size)]!).coins
          p.ligerito.initialProof.rows ∧
      merkleCheckWith (st.merkle.opsWith hs) st.merkle p.cap (sch.levels.getD 0 default).cap
        (sch.levels.getD 0 default).d
        (positions (sch.levels.getD 0 default)
          (stream.rounds[Claims.Flock.plainQueryRound st.m st.kLog sch.k0 (2 + extras.size)]!).coins)
        p.ligerito.initialProof.rows p.ligerito.initialProof.siblings p.ligerito.initialProof.salts
        (2 ^ (sch.levels.getD 0 default).lanes) = .ok () ∧
      p.cap.foldl (· ++ ·) ByteArray.empty = sess.rootB[i]! := by
  obtain ⟨stream, extras, p, hstream, hext, hp, hlt, hmc, hcap, hroot⟩ :=
    plainQueryRound_verifyRepWith hs spec st sch sess i r bytes cap hsch hf hkl h
  refine ⟨stream, extras, p, hstream, hext, hp, hlt, ?_, hmc, hcap ▸ hroot⟩
  rw [plainProofShape_eq st sch extras hsch, Claims.Flock.plainRepColsOf, hp]
  simp only [shapeOf, Array.getElem?_eq_getElem hlt, Option.map_some, Option.getD_some,
    getElem!_pos stream.rounds (Claims.Flock.plainQueryRound st.m st.kLog sch.k0 (2 + extras.size)) hlt]

/-- **`verifyWithM`'s accepted session, table by table and rep by rep**: in `V`, under any oracle, an accepted
session decodes, has one statement per table, and each table `i`'s rep `r` (`r < 2`, proof file `2i + r`) is accepted
by `verifyRepWith` at the table's `fast100` schedule, both with one cap. -/
theorem plainVerifyWithM_reps (hs : Hashes Flock.V) (spec : SessionSpec) (tabs : Array Setup) (record : String)
    (proofs : Array ByteArray) (h : verifyWithM hs spec tabs record proofs = .ok ()) :
    ∃ sess : RunRecord, Record.decodeWith hs spec record = .ok sess ∧ tabs.size = spec.tables.size ∧
      ∀ (i : ℕ) (hi : i < tabs.size), ∃ (sch : Flock.Schedule) (c : ByteArray),
        fast100 tabs[i].m = some sch ∧ ∀ r < 2,
          verifyRepWith hs spec tabs[i] sch sess i r proofs[2 * i + r]! = .ok c := by
  unfold verifyWithM at h
  dsimp only at h
  by_cases c0 : (tabs.size != spec.tables.size || tabs.size == 0) = true
  · rw [ite_eq_left c0] at h; obtain ⟨_, h0, _⟩ := ebind_ok h; cases h0
  rw [ite_eq_right c0] at h
  obtain ⟨sess, hsess, h⟩ := ebind_ok h
  obtain ⟨_, _, h⟩ := ebind_ok h
  obtain ⟨schs, hschs, h⟩ := ebind_ok h
  split at h
  · obtain ⟨_, h0, _⟩ := ebind_ok h; cases h0
  split at h
  · obtain ⟨_, h0, _⟩ := ebind_ok h; cases h0
  obtain ⟨_, _, h⟩ := ebind_ok h
  obtain ⟨_, hloop, _⟩ := ebind_ok h
  rw [Std.Legacy.Range.forIn_eq_forIn_range'] at hloop
  simp only [Std.Legacy.Range.size, Nat.sub_zero, Nat.add_one_sub_one, Nat.div_one] at hloop
  have c0' : tabs.size = spec.tables.size := by
    simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, beq_iff_eq, not_or, not_not] at c0
    exact c0.1
  obtain ⟨hsz, hsch⟩ := array_mapM_ok_getElem hschs
  have hP := range'_forIn_each tabs.size _
    (fun i => ∃ c, verifyRepWith hs spec tabs[i]! schs[i]! sess i 0 proofs[2 * i]! = .ok c ∧
      verifyRepWith hs spec tabs[i]! schs[i]! sess i 1 proofs[2 * i + 1]! = .ok c)
    (fun a s y hy => by
      obtain ⟨c0, h0, hy⟩ := ebind_ok hy
      obtain ⟨c1, h1, hy⟩ := ebind_ok hy
      by_cases c3 : (c0 != c1) = true
      · rw [ite_eq_left c3] at hy; obtain ⟨_, h0, _⟩ := ebind_ok hy; cases h0
      rw [ite_eq_right c3] at hy
      simp only [pure, Except.pure, Except.ok.injEq] at hy
      have hb : (c0 == c1) = true := by
        cases hb : (c0 == c1)
        · exact absurd (by simp [bne, hb]) c3
        · rfl
      rw [bytes_eq_of_beq hb] at h0
      exact ⟨⟨c1, h0, h1⟩, _, hy.symm⟩) _ _ hloop
  refine ⟨sess, hsess, c0', fun i hi => ?_⟩
  have hs' : i < schs.size := hsz ▸ hi
  obtain ⟨c, h0, h1⟩ := hP i hi
  rw [getElem!_pos tabs i hi, getElem!_pos schs i hs'] at h0 h1
  refine ⟨schs[i], c, ?_, fun r hr => ?_⟩
  · have hf := hsch i hi hs'
    cases hm : fast100 tabs[i].m with
    | none => rw [hm] at hf; cases hf
    | some s => rw [hm] at hf; cases hf; rfl
  interval_cases r
  · simpa using h0
  · exact h1

/-- **Level 0's query round in an accepted plain session** (`verifyWithM` in `V`, any oracle), on tables whose circuit
folds refuse a point of the wrong length: for every table `i` and rep `r < 2`, the rep's stream has round
`plainQueryRound m kLog k0 claims` at the table's `m`, `kLog`, schedule and claim count, the reader's columns of the rep
(`plainRepColsOf` at `plainProofShape`) are the proof's level-0 rows at the positions of that round's coins, and level
0's Merkle check passed on exactly those rows at exactly those positions against the table's committed `root_B`. -/
theorem plainQueryRound_verifyWithM (hs : Hashes Flock.V) (spec : SessionSpec) (tabs : Array Setup) (record : String)
    (proofs : Array ByteArray)
    (hf : ∀ st ∈ tabs, (∀ a b lw rho ts v, st.fold.folded a b lw rho ts = .ok v → rho.size + 6 = st.fold.kLog) ∧
      st.fold.kLog = st.kLog)
    (h : verifyWithM hs spec tabs record proofs = .ok ()) :
    ∃ sess : RunRecord, Record.decodeWith hs spec record = .ok sess ∧ tabs.size = spec.tables.size ∧
      ∀ (i : ℕ) (hi : i < tabs.size), ∃ sch : Flock.Schedule, fast100 tabs[i].m = some sch ∧ ∀ r < 2,
        ∃ (stream : StreamRec) (extras : Array Flock.Claim) (p : Proof),
          sess.streams.find? (fun x => x.name == s!"{spec.tables[i]!}/rep{r}") = some stream ∧
          tabs[i].extra sess.points = .ok extras ∧
          decodeProof (shapeOf tabs[i] sch extras.size) proofs[2 * i + r]! = .ok p ∧
          Claims.Flock.plainQueryRound tabs[i].m tabs[i].kLog sch.k0 (2 + extras.size) < stream.rounds.size ∧
          Claims.Flock.plainRepColsOf (Claims.Flock.plainProofShape tabs[i].m tabs[i].kLog (2 + extras.size)
              tabs[i].merkle.digestLen
              (if tabs[i].merkle.saltBytes = 0 && !tabs[i].openedSalts then none else some tabs[i].merkle.saltBytes))
              stream.rounds proofs[2 * i + r]! =
            Claims.Flock.zkRepCols (sch.levels.getD 0 default)
              (stream.rounds[Claims.Flock.plainQueryRound tabs[i].m tabs[i].kLog sch.k0 (2 + extras.size)]!).coins
              p.ligerito.initialProof.rows ∧
          merkleCheckWith (tabs[i].merkle.opsWith hs) tabs[i].merkle p.cap (sch.levels.getD 0 default).cap
            (sch.levels.getD 0 default).d
            (positions (sch.levels.getD 0 default)
              (stream.rounds[Claims.Flock.plainQueryRound tabs[i].m tabs[i].kLog sch.k0 (2 + extras.size)]!).coins)
            p.ligerito.initialProof.rows p.ligerito.initialProof.siblings p.ligerito.initialProof.salts
            (2 ^ (sch.levels.getD 0 default).lanes) = .ok () ∧
          p.cap.foldl (· ++ ·) ByteArray.empty = sess.rootB[i]! := by
  obtain ⟨sess, hsess, hsz, hreps⟩ := plainVerifyWithM_reps hs spec tabs record proofs h
  refine ⟨sess, hsess, hsz, fun i hi => ?_⟩
  obtain ⟨sch, c, hsch, hr⟩ := hreps i hi
  obtain ⟨hfi, hkl⟩ := hf tabs[i] (Array.getElem_mem hi)
  exact ⟨sch, hsch, fun r hr2 => plainRepColsOf_verifyRepWith hs spec tabs[i] sch sess i r _ c hsch hfi hkl (hr r hr2)⟩

end FlockDeploy.KS
