import Proofs.Flock.Claims.KS.PlainClashL
import Proofs.Flock.Claims.KS.Entries
import Proofs.Flock.Claims.DeployRounds
import Proofs.Flock.Claims.Recursion.PlainK

/-!
# Every level's queries in the plain verifier over the hash oracle

`plainQueryRound_verifyWithM` places level 0's Merkle check of an accepted plain rep at round `plainQueryRound` of its
stream. Here the same holds for every later level, from the plain verifier's own transcript (`verifyRepWith hs` in `V`,
any hash oracle `hs`): Ligerito draws level `ℓ`'s query coins `levRound k0 R ℓ` rounds in, `k0 + 6` for level 0, `11`
later for each inner level, and `7` into the last level's iteration (`ligeritoWith_levels`), and level `ℓ ≥ 1`'s check
passes against the proof's `ℓ − 1`-th recursive cap on its level-`ℓ` rows (`openAt`).

So two rows one rep of an accepted plain run opens at one later-level position, if they differ, are two `Opening`s
of that position against one root in the run's own log, a `PlainLevelClash` (`plain_levelClash`).

* `levRound`: the round of level `ℓ`'s query coins in a Ligerito run.
* `lforIn_inv`: a Ligerito loop whose steps keep an invariant gives each step's conclusion.
* `ligeritoWith_levels`: level `ℓ ≥ 1`'s Merkle check in an accepted Ligerito run.
* `plainLevelRound`, `plainLevelColsOf`: the round of level `ℓ`'s query coins in a plain rep's stream, and the rep's
  level-`ℓ` rows at the positions of its coins.
* `plainLevels_verifyRepWith`, `plainLevels_verifyWithM`: an accepted rep's, and an accepted session's, later levels'
  Merkle checks at those rounds, on those rows.
* `plainRawLevelCols`, `plain_levelOpening`, `plain_levelClash`: the level-`ℓ` rows table 0's rep `k` opens in a
  plain record, each an `Opening` in an accepted run's log against the rep's proof's cap, and two of them at one
  position that differ clash there.
-/

namespace FlockDeploy.KS

open Flock Transcript FlockSoundness FlockSoundness.Refine

/-- **The round of level `ℓ`'s query coins** in a Ligerito run on `R + 1` levels, from its first round: level 0's after
`k0 + 6` rounds, each inner level's `11` after the one before, and the last level's `7` into its iteration. -/
def levRound (k0 R ℓ : ℕ) : ℕ := if ℓ < R then k0 + 6 + 11 * ℓ else k0 + 2 + 11 * ℓ

/-- **A Ligerito loop keeping an invariant**: over `a, …, a + n − 1 < N`, where every step below `N` from a state with
the invariant yields one with it and establishes `Q i`, every step's `Q i` holds. -/
theorem lforIn_inv {σ : Type} (body : ℕ → σ → LM (ForInStep σ)) (Inv : ℕ → σ → Transcript → Prop) (Q : ℕ → Prop)
    (N : ℕ) (hstep : ∀ i s st t b st' t', i < N → Inv i s t → body i s st t = .ok ((b, st'), t') →
      ∃ s', b = .yield s' ∧ Inv (i + 1) s' t' ∧ Q i) :
    ∀ (n a : ℕ) (s : σ) (st : LState) (t : Transcript) (out : σ) (st' : LState) (t' : Transcript), a + n ≤ N →
      Inv a s t → forIn (List.range' a n) s body st t = .ok ((out, st'), t') → ∀ i, a ≤ i → i < a + n → Q i := by
  intro n
  induction n with
  | zero => intro a s st t out st' t' _ _ _ i h1 h2; omega
  | succ n ih =>
    intro a s st t out st' t' hN hI h i h1 h2
    rw [List.range'_succ, List.forIn_cons] at h
    obtain ⟨b, st₁, t₁, h₁, h₂⟩ := lbind_ok h
    obtain ⟨s', rfl, hI', hQ⟩ := hstep a s st t b st₁ t₁ (by omega) hI h₁
    rcases Nat.eq_or_lt_of_le h1 with rfl | h1
    · exact hQ
    · exact ih (a + 1) s' st₁ t₁ out st' t' (by omega) hI' h₂ i h1 (by omega)

/-- **Every later level's queries over the oracle**, plain: in an accepted Ligerito run on at least two levels, level
`ℓ ≥ 1`'s Merkle check passes against the proof's `ℓ − 1`-th recursive cap, on its level-`ℓ` rows, siblings and salts,
at the positions of round `t.k + levRound k0 R ℓ`'s coins. -/
theorem ligeritoWith_levels (hs : Hashes Flock.V) (ms : MerkleScheme) (sch : Schedule) (T : F128)
    (cap : Array ByteArray) (p : LigeritoProof) (basisAt : Array F256 → ℕ → Array F256) (hn : sch.n ≠ 0)
    (h2 : 2 ≤ sch.levels.size) (hq : ∀ i < sch.levels.size, (sch.levels.getD i default).q ≠ 0) {t t' : Transcript}
    {u : Unit} (h : ligeritoWith hs ms sch T cap p basisAt t = .ok (u, t')) (ℓ : ℕ) (h1 : 1 ≤ ℓ)
    (hℓ : ℓ < sch.levels.size) :
    merkleCheckWith (ms.opsWith hs) ms p.recursiveCaps[ℓ - 1]! (sch.levels.getD ℓ default).cap
      (sch.levels.getD ℓ default).d
      (positions (sch.levels.getD ℓ default) (t.rounds[t.k + levRound sch.k0 (sch.levels.size - 1) ℓ]!).coins)
      (openAt p (sch.levels.size - 1) ℓ).1 (openAt p (sch.levels.size - 1) ℓ).2.1
      (openAt p (sch.levels.size - 1) ℓ).2.2 (2 ^ (sch.levels.getD ℓ default).lanes) = .ok () := by
  obtain ⟨i, rfl⟩ : ∃ i, ℓ = i + 1 := ⟨ℓ - 1, by omega⟩
  unfold ligeritoWith at h
  simp only [liftTM_V, liftLM_V] at h
  obtain ⟨-, h⟩ := check_bind h
  obtain ⟨t₁, k₁, h⟩ := label_bind h
  obtain ⟨t₂, k₂, h⟩ := absorbF_bind h
  obtain ⟨t₃, k₃, h⟩ := absorbBytes_bind h
  obtain ⟨⟨u', stF⟩, t₄, h₄, h⟩ := bind_ok h
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
  obtain ⟨initCh, st₁₀, t₁₀, h₁₀, h₄⟩ := lbind_ok h₄
  rw [lforIn_range_eq] at h₁₀
  obtain ⟨hr₁₀, hk₁₀⟩ := FlockDeploy.lforIn_moves _ 1 (by
      intro i s st tt b st' tt' hb
      obtain ⟨c, st₁, tt₁, hr₁, hk₁, hb⟩ := challengeWith_bind hs hb
      obtain ⟨rfl, rfl, rfl⟩ := lpure_ok hb
      exact ⟨hr₁, hk₁, _, rfl⟩) sch.k0 0 _ _ t₉ initCh st₁₀ t₁₀ h₁₀
  obtain ⟨_, t₁₁, h₁₁, h₄⟩ := lift_bind h₄
  have k₁₁ := absorbBytes_ok h₁₁
  obtain ⟨st₁₂, t₁₂, hr₁₂, hk₁₂, h₄⟩ := oodIntroWith_bind hs (by omega) h₄
  obtain ⟨st₁₃, t₁₃, hr₁₃, hk₁₃, h₄⟩ := oodIntroWith_bind hs (by omega) h₄
  obtain ⟨-, -, ⟨pos0, al0, e0⟩, st₁₄, t₁₄, hr₁₄, hk₁₄, h₄⟩ := queryLevelWith_bind hs (hq 0 (by omega)) h₄
  dsimp only at h₄
  obtain ⟨mq, t₁₅, -, k₁₅, h₄⟩ := nextMsg_bind h₄
  obtain ⟨t₁₆, k₁₆, h₄⟩ := claimPow_bind h₄
  obtain ⟨-, bq, t₁₇, hr₁₇, hk₁₇, h₄⟩ := lsampleWith_bind hs h₄
  have h₄ := lmodify_bind h₄
  obtain ⟨out, st₁₈, t₁₈, hloop, -⟩ := lbind_ok h₄
  rw [lforIn_range_eq] at hloop
  have er₁₇ : t₁₇.rounds = t.rounds := by
    rw [hr₁₇, k₁₆.2, k₁₅.2, hr₁₄, hr₁₃, hr₁₂, k₁₁.2, hr₁₀, k₉.2, hr₈, k₇.2, k₆.2, hr₅, k₃.2, k₂.2, k₁.2]
  have ek₁₇ : t₁₇.k = t.k + (sch.k0 + 9) := by
    rw [hk₁₇, k₁₆.1, k₁₅.1, hk₁₄, hk₁₃, hk₁₂, k₁₁.1, hk₁₀, k₉.1, hk₈, k₇.1, k₆.1, hk₅, k₃.1, k₂.1, k₁.1]; omega
  refine lforIn_inv _ (fun i s tt => i < sch.levels.size - 1 → tt.rounds = t.rounds ∧
      tt.k = t.k + (sch.k0 + 9) + 11 * i ∧ s.2.1 = p.recursiveCaps[i]!)
    (fun i => merkleCheckWith (ms.opsWith hs) ms p.recursiveCaps[i + 1 - 1]! (sch.levels.getD (i + 1) default).cap
      (sch.levels.getD (i + 1) default).d
      (positions (sch.levels.getD (i + 1) default)
        (t.rounds[t.k + levRound sch.k0 (sch.levels.size - 1) (i + 1)]!).coins)
      (openAt p (sch.levels.size - 1) (i + 1)).1 (openAt p (sch.levels.size - 1) (i + 1)).2.1
      (openAt p (sch.levels.size - 1) (i + 1)).2.2 (2 ^ (sch.levels.getD (i + 1) default).lanes) = .ok ())
    (sch.levels.size - 1) ?_ (sch.levels.size - 1) 0 _ _ t₁₇ out st₁₈ t₁₈ (by omega)
    (fun _ => ⟨er₁₇, by rw [ek₁₇]; omega, rfl⟩) hloop i (by omega) (by omega)
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
      have ek : tt₄.k = t.k + levRound sch.k0 (sch.levels.size - 1) (i + 1) := by
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
    have ek : tt₂.k = t.k + levRound sch.k0 (sch.levels.size - 1) (i + 1) := by
      rw [k₂.1, hk₁, hkt, levRound, ite_eq_right hi]; omega
    have ho : openAt p (sch.levels.size - 1) (i + 1) =
        (p.finalProof.rows, p.finalProof.siblings, p.finalProof.salts) := by
      simp [openAt, hi]
    rw [ho, Nat.add_sub_cancel, ← hcap, ← er, ← ek]
    rw [hR] at hmc
    exact hmc

/-! ## An accepted rep, and an accepted session -/

/-- **The round of level `ℓ`'s query coins** in a plain rep's stream: zerocheck's `3 + (m − 6)` rounds, lincheck's
`3 + (kLog − 6)`, the ring switch's `claims + 1`, then Ligerito's `levRound k0 R ℓ`. -/
def plainLevelRound (m kLog k0 claims R ℓ : ℕ) : ℕ :=
  (3 + (m - 6)) + (3 + (kLog - 6)) + (claims + 1) + levRound k0 R ℓ

/-- **A plain rep's opened level-`ℓ` rows** from its recorded rounds and proof bytes, at the positions of round
`plainLevelRound`'s coins, for a level `1 ≤ ℓ` of its schedule: none otherwise, or if the proof does not decode. -/
def plainLevelColsOf (sh : Shape) (rounds : Array Round) (bytes : ByteArray) (ℓ : ℕ) :
    List (ℕ × List FlockLevel3.GF128) :=
  if 1 ≤ ℓ ∧ ℓ < sh.sched.levels.size then
    match decodeProof sh bytes with
    | .ok p =>
      Claims.Flock.zkRepCols (sh.sched.levels.getD ℓ default)
        ((rounds[plainLevelRound sh.m sh.kLog sh.sched.k0 sh.claims (sh.sched.levels.size - 1) ℓ]?.map
          (·.coins)).getD #[]) (openAt p.ligerito (sh.sched.levels.size - 1) ℓ).1
    | .error _ => []
  else []

/-- **An accepted plain rep's later levels' queries sit at rounds `plainLevelRound` of its stream.** At the table's
`fast100` schedule, on a circuit fold that refuses a point of the wrong length, a rep that `verifyRepWith hs` accepts
(in `V`, any oracle) passed level `ℓ`'s Merkle check, for every level `1 ≤ ℓ`, against the proof's `ℓ − 1`-th recursive
cap on its level-`ℓ` rows at the positions of round `plainLevelRound`'s coins, and the reader's level-`ℓ` rows
(`plainLevelColsOf`) are those positions with those rows. -/
theorem plainLevels_verifyRepWith (hs : Hashes Flock.V) (spec : SessionSpec) (st : Setup) (sch : Flock.Schedule)
    (sess : Session) (i r : ℕ) (bytes cap : ByteArray) (hsch : fast100 st.m = some sch)
    (hf : ∀ a b lw rho ts v, st.fold.folded a b lw rho ts = .ok v → rho.size + 6 = st.fold.kLog)
    (hkl : st.fold.kLog = st.kLog) (h : verifyRepWith hs spec st sch sess i r bytes = .ok cap) :
    ∃ (stream : StreamRec) (extras : Array Flock.Claim) (p : Proof),
      sess.streams.find? (fun x => x.name == s!"{spec.tables[i]!}/rep{r}") = some stream ∧
      st.extra sess.points = .ok extras ∧ decodeProof (shapeOf st sch extras.size) bytes = .ok p ∧
      ∀ ℓ, 1 ≤ ℓ → ℓ < sch.levels.size →
        plainLevelColsOf (Claims.Flock.plainProofShape st.m st.kLog (2 + extras.size) st.merkle.digestLen
            (if st.merkle.saltBytes = 0 && !st.openedSalts then none else some st.merkle.saltBytes))
            stream.rounds bytes ℓ =
          Claims.Flock.zkRepCols (sch.levels.getD ℓ default)
            (stream.rounds[plainLevelRound st.m st.kLog sch.k0 (2 + extras.size) (sch.levels.size - 1) ℓ]!).coins
            (openAt p.ligerito (sch.levels.size - 1) ℓ).1 ∧
        merkleCheckWith (st.merkle.opsWith hs) st.merkle p.ligerito.recursiveCaps[ℓ - 1]!
          (sch.levels.getD ℓ default).cap (sch.levels.getD ℓ default).d
          (positions (sch.levels.getD ℓ default)
            (stream.rounds[plainLevelRound st.m st.kLog sch.k0 (2 + extras.size) (sch.levels.size - 1) ℓ]!).coins)
          (openAt p.ligerito (sch.levels.size - 1) ℓ).1 (openAt p.ligerito (sch.levels.size - 1) ℓ).2.1
          (openAt p.ligerito (sch.levels.size - 1) ℓ).2.2 (2 ^ (sch.levels.getD ℓ default).lanes) = .ok () := by
  have h22 := exec_fast100_ge hsch
  obtain ⟨hdim, hsm⟩ := fast100_dims _ _ hsch
  have hq : ∀ j < sch.levels.size, (sch.levels.getD j default).q ≠ 0 := fun j hj =>
    (fast100_level _ _ hsch _ (by unfold Array.getD; rw [dite_eq_left hj]; exact Array.getElem_mem hj)).q_ne_zero
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
  rw [StateT.run] at hrun
  obtain ⟨zc, t₁, h₁, hrun⟩ := bind_ok hrun
  obtain ⟨hr₁, hk₁⟩ := zerocheckWith_moves hs _ _ _ _ (by omega) h₁
  obtain ⟨lc, t₂, h₂, hrun⟩ := bind_ok hrun
  obtain ⟨h6, hr₂, hk₂⟩ := lincheckWith_moves hs _ _ _ hf h₂
  rw [hkl] at h6
  obtain ⟨⟨T, sws⟩, t₃, h₃, hrun⟩ := bind_ok hrun
  have hsz2 : (abcClaims st.m st.kLog zc lc).size = 2 := rfl
  obtain ⟨hr₃, hk₃⟩ := ZkQuery.ringSwitchWith_moves hs _ _ _ (by rw [Array.size_append, hsz2]; omega) h₃
  have er : t₃.rounds = stream.rounds := by rw [hr₃, hr₂, hr₁]; rfl
  refine ⟨stream, extras, p, hstream, hext, hp, fun ℓ h1 hℓ => ⟨?_, ?_⟩⟩
  · have hp' : decodeProof (shapeOf st sch extras.size) bytes = .ok p := hp
    rw [plainProofShape_eq st sch extras hsch, plainLevelColsOf, ite_eq_left ⟨h1, by simpa [shapeOf] using hℓ⟩, hp']
    simp only [shapeOf]
    rw [Claims.Flock.coins_getElem!]
  · have hmc := ligeritoWith_levels hs _ _ _ _ _ _ hn (by have := hdim.2.1; omega) hq hrun ℓ h1 hℓ
    have ek : t₃.k + levRound sch.k0 (sch.levels.size - 1) ℓ =
        plainLevelRound st.m st.kLog sch.k0 (2 + extras.size) (sch.levels.size - 1) ℓ := by
      rw [hk₃, hk₂, hk₁, hkl, Array.size_append, hsz2]
      simp [plainLevelRound, Transcript.init]; omega
    rw [er, ek] at hmc
    exact hmc

/-- **Every later level's query round in an accepted plain session** (`verifyWithM` in `V`, any oracle), on tables
whose circuit folds refuse a point of the wrong length: for every table `i`, rep `r < 2` and level `1 ≤ ℓ` of the
table's schedule, level `ℓ`'s Merkle check passed against the proof's `ℓ − 1`-th recursive cap on its level-`ℓ` rows at
the positions of round `plainLevelRound`'s coins of the rep's stream, and the reader's level-`ℓ` rows
(`plainLevelColsOf` at `plainProofShape`) are those positions with those rows. -/
theorem plainLevels_verifyWithM (hs : Hashes Flock.V) (spec : SessionSpec) (tabs : Array Setup) (record : String)
    (proofs : Array ByteArray)
    (hf : ∀ st ∈ tabs, (∀ a b lw rho ts v, st.fold.folded a b lw rho ts = .ok v → rho.size + 6 = st.fold.kLog) ∧
      st.fold.kLog = st.kLog)
    (h : verifyWithM hs spec tabs record proofs = .ok ()) :
    ∃ sess : Session, Record.decodeWith hs spec record = .ok sess ∧ tabs.size = spec.tables.size ∧
      ∀ (i : ℕ) (hi : i < tabs.size), ∃ sch : Flock.Schedule, fast100 tabs[i].m = some sch ∧ ∀ r < 2,
        ∃ (stream : StreamRec) (extras : Array Flock.Claim) (p : Proof),
          sess.streams.find? (fun x => x.name == s!"{spec.tables[i]!}/rep{r}") = some stream ∧
          tabs[i].extra sess.points = .ok extras ∧
          decodeProof (shapeOf tabs[i] sch extras.size) proofs[2 * i + r]! = .ok p ∧
          ∀ ℓ, 1 ≤ ℓ → ℓ < sch.levels.size →
            plainLevelColsOf (Claims.Flock.plainProofShape tabs[i].m tabs[i].kLog (2 + extras.size)
                tabs[i].merkle.digestLen
                (if tabs[i].merkle.saltBytes = 0 && !tabs[i].openedSalts then none else some tabs[i].merkle.saltBytes))
                stream.rounds proofs[2 * i + r]! ℓ =
              Claims.Flock.zkRepCols (sch.levels.getD ℓ default)
                (stream.rounds[plainLevelRound tabs[i].m tabs[i].kLog sch.k0 (2 + extras.size)
                  (sch.levels.size - 1) ℓ]!).coins (openAt p.ligerito (sch.levels.size - 1) ℓ).1 ∧
            merkleCheckWith (tabs[i].merkle.opsWith hs) tabs[i].merkle p.ligerito.recursiveCaps[ℓ - 1]!
              (sch.levels.getD ℓ default).cap (sch.levels.getD ℓ default).d
              (positions (sch.levels.getD ℓ default)
                (stream.rounds[plainLevelRound tabs[i].m tabs[i].kLog sch.k0 (2 + extras.size)
                  (sch.levels.size - 1) ℓ]!).coins)
              (openAt p.ligerito (sch.levels.size - 1) ℓ).1 (openAt p.ligerito (sch.levels.size - 1) ℓ).2.1
              (openAt p.ligerito (sch.levels.size - 1) ℓ).2.2 (2 ^ (sch.levels.getD ℓ default).lanes) = .ok () := by
  obtain ⟨sess, hsess, hsz, hreps⟩ := plainVerifyWithM_reps hs spec tabs record proofs h
  refine ⟨sess, hsess, hsz, fun i hi => ?_⟩
  obtain ⟨sch, c, hsch, hr⟩ := hreps i hi
  obtain ⟨hfi, hkl⟩ := hf tabs[i] (Array.getElem_mem hi)
  exact ⟨sch, hsch, fun r hr2 => plainLevels_verifyRepWith hs spec tabs[i] sch sess i r _ c hsch hfi hkl (hr r hr2)⟩

/-- `plainLevels_verifyWithM` at `verifyWithMK keep`, on the session the element-level decoder reads. -/
theorem plainLevels_verifyWithMK (keep : Bool → Bool) (hs : Hashes Flock.V) (spec : SessionSpec) (tabs : Array Setup)
    (record : String) (proofs : Array ByteArray)
    (hf : ∀ st ∈ tabs, (∀ a b lw rho ts v, st.fold.folded a b lw rho ts = .ok v → rho.size + 6 = st.fold.kLog) ∧
      st.fold.kLog = st.kLog)
    (h : verifyWithMK keep hs spec tabs record proofs = .ok ()) :
    ∃ sess : Session, Record.decodeWithK keep hs spec record = .ok sess ∧ tabs.size = spec.tables.size ∧
      ∀ (i : ℕ) (hi : i < tabs.size), ∃ sch : Flock.Schedule, fast100 tabs[i].m = some sch ∧ ∀ r < 2,
        ∃ (stream : StreamRec) (extras : Array Flock.Claim) (p : Proof),
          sess.streams.find? (fun x => x.name == s!"{spec.tables[i]!}/rep{r}") = some stream ∧
          tabs[i].extra sess.points = .ok extras ∧
          decodeProof (shapeOf tabs[i] sch extras.size) proofs[2 * i + r]! = .ok p ∧
          ∀ ℓ, 1 ≤ ℓ → ℓ < sch.levels.size →
            plainLevelColsOf (Claims.Flock.plainProofShape tabs[i].m tabs[i].kLog (2 + extras.size)
                tabs[i].merkle.digestLen
                (if tabs[i].merkle.saltBytes = 0 && !tabs[i].openedSalts then none else some tabs[i].merkle.saltBytes))
                stream.rounds proofs[2 * i + r]! ℓ =
              Claims.Flock.zkRepCols (sch.levels.getD ℓ default)
                (stream.rounds[plainLevelRound tabs[i].m tabs[i].kLog sch.k0 (2 + extras.size)
                  (sch.levels.size - 1) ℓ]!).coins (openAt p.ligerito (sch.levels.size - 1) ℓ).1 ∧
            merkleCheckWith (tabs[i].merkle.opsWith hs) tabs[i].merkle p.ligerito.recursiveCaps[ℓ - 1]!
              (sch.levels.getD ℓ default).cap (sch.levels.getD ℓ default).d
              (positions (sch.levels.getD ℓ default)
                (stream.rounds[plainLevelRound tabs[i].m tabs[i].kLog sch.k0 (2 + extras.size)
                  (sch.levels.size - 1) ℓ]!).coins)
              (openAt p.ligerito (sch.levels.size - 1) ℓ).1 (openAt p.ligerito (sch.levels.size - 1) ℓ).2.1
              (openAt p.ligerito (sch.levels.size - 1) ℓ).2.2 (2 ^ (sch.levels.getD ℓ default).lanes) = .ok () := by
  obtain ⟨sess, schs, hsess, hsz, -, hall⟩ := verifyWithMK_tables keep hs h
  refine ⟨sess, hsess, hsz, fun i hi => ?_⟩
  obtain ⟨hi', c₀, c₁, hsch, h₀, h₁⟩ := hall i hi
  obtain ⟨hfi, hkl⟩ := hf tabs[i] (Array.getElem_mem hi)
  refine ⟨schs[i], hsch, fun r hr2 => ?_⟩
  rcases (show r = 0 ∨ r = 1 by omega) with rfl | rfl
  · rw [Nat.add_zero]
    exact plainLevels_verifyRepWith hs spec tabs[i] schs[i] sess i 0 _ c₀ hsch hfi hkl h₀
  · exact plainLevels_verifyRepWith hs spec tabs[i] schs[i] sess i 1 _ c₁ hsch hfi hkl h₁

/-! ## An accepted plain run's later openings -/

theorem plainLevelColsOf_congr (sh : Shape) {a b : Array Round} (bytes : ByteArray) (ℓ : ℕ)
    (h : ∀ j : ℕ, a[j]?.map (fun rd : Round => rd.coins) = b[j]?.map (fun rd : Round => rd.coins)) :
    plainLevelColsOf sh a bytes ℓ = plainLevelColsOf sh b bytes ℓ := by
  unfold plainLevelColsOf
  rw [h]

/-- **The level-`ℓ` rows table 0's rep `k` opens in a plain record**: `plainLevelColsOf` at the reader's table, on rep
`k`'s recorded rounds and proof file. -/
def plainRawLevelCols (s : Claims.Seen) (k ℓ : ℕ) : List (ℕ × List FlockLevel3.GF128) :=
  match runOf false s 0 with
  | some r =>
    match r.reps[k]? with
    | some y => plainLevelColsOf r.plainShape y.1 y.2 ℓ
    | none => []
  | none => []

/-- **A row one rep of an accepted plain run opens at a later level is an `Opening` in its log**: a level-`ℓ` row of
table 0's rep `k` is an `Opening` of its position against the flattened `ℓ − 1`-th recursive cap of the rep's proof
(the proof file decoded at the table's `fast100` schedule and `|regions| · points` extra claims), a cap of `2^cap`
64-byte digests, each SHA-512 input of it in the run's verifier log: its verdict accepts with SHA-512 answered within
that log (`acc_within`), so level `ℓ`'s Merkle check passed on the rep's rows at those positions
(`plainLevels_verifyWithM`), and the check at the table's scheme, carried back to `OC`, logged them
(`merkleCheck_logsOf`). -/
theorem plain_levelOpeningK (keep : Bool → Bool) {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω}
    {r : Fin ((V false).k x) → Coin} {k ℓ q : ℕ} {a : List FlockLevel3.GF128}
    (h : (Claims.run sha512 P (VK keep) x ω r).out = .acc)
    (ha : (q, a) ∈ plainRawLevelCols (Claims.run sha512 P (VK keep) x ω r).seen k ℓ) :
    ∃ (s : Ins) (spec : SessionSpec) (sts : Array Stmt) (f : Final) (st : Stmt) (sch : Flock.Schedule) (p : Proof)
      (row : Array F128), setupOf false x = some (s, spec, sts) ∧
      lastFinal (Claims.run sha512 P (VK keep) x ω r).seen = some f ∧ sts[0]? = some st ∧
      fast100 st.m = some sch ∧
      decodeProof (shapeOf (Setup.ofCircuit st) sch (st.regions.size * spec.points)) f.proofs[k]! = .ok p ∧
      a = row.toList.map FlockLevel3.GF128.of ∧
      p.ligerito.recursiveCaps[ℓ - 1]!.size = 2 ^ (sch.levels.getD ℓ default).cap ∧
      (∀ b ∈ p.ligerito.recursiveCaps[ℓ - 1]!.toList, b.size = 64) ∧
      Opening (Setup.ofCircuit st).merkle (sch.levels.getD ℓ default) (2 ^ (sch.levels.getD ℓ default).lanes)
        (p.ligerito.recursiveCaps[ℓ - 1]!.foldl (· ++ ·) ByteArray.empty) (Claims.run sha512 P (VK keep) x ω r).vLog
        q row := by
  obtain ⟨s, spec, sts, link, c₁, mids, f, hs, hf, hlink, hc, hv⟩ := accK_within keep h
  generalize hL : (Claims.run sha512 P (VK keep) x ω r).vLog = L at hv ⊢
  obtain ⟨hx, lv, hlv, -⟩ := Claims.Bridge.Custody.checks_ok hc
  unfold plainRawLevelCols at ha
  split at ha
  swap
  · simp at ha
  rename_i rr hrr
  obtain ⟨st, extras, hst, hext, rfl⟩ := runOf_eq hf hx hs hlv hrr
  obtain ⟨hk, ha⟩ : k < 2 ∧
      (q, a) ∈ plainLevelColsOf (Claims.Flock.plainProofShape (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog
        (2 + extras.size) (Setup.ofCircuit st).merkle.digestLen (saltLenOf false (Setup.ofCircuit st)))
        (lv.streams.getD k #[]) (f.proofs.getD k ByteArray.empty) ℓ := by
    rcases k with _ | _ | k
    · exact ⟨by omega, ha⟩
    · exact ⟨by omega, ha⟩
    · simp at ha
  have hsts0 : 0 < sts.size := by
    rcases Array.getElem?_eq_some_iff.1 hst with ⟨h0, -⟩
    exact h0
  have hst0 : sts[0] = st := by
    rcases Array.getElem?_eq_some_iff.1 hst with ⟨h0, e⟩
    exact e
  have h0 : 0 < (tabsOf sts).size := by simpa [tabsOf] using hsts0
  have ht0 : (tabsOf sts)[0] = Setup.ofCircuit st := by rw [tabsOf_getElem h0]; simp only [hst0]
  obtain ⟨hszL, hall⟩ := rep_plainK keep _ hv hlv
  have hkL : k < lv.streams.size := by rw [hszL]; omega
  obtain ⟨sess, sch, y, c, hpts, hpn, hschK, hrep, hy, hfa⟩ := hall k hkL
  have hk2 : k / 2 = 0 := by omega
  have hkm : k % 2 = k := by omega
  have hst0' : sts[k / 2]! = st := by rw [hk2, getElem!_pos sts 0 hsts0, hst0]
  rw [hst0'] at hschK
  rw [hst0', hk2, hkm] at hrep
  rw [hk2, hkm] at hy
  have hsch : fast100 (Setup.ofCircuit st).m = some sch := hschK
  obtain ⟨hfi, hkl⟩ := tabsOf_fold sts (Setup.ofCircuit st) (ht0 ▸ Array.getElem_mem h0)
  obtain ⟨stream, extras', p, hstream, hext', hp', hlev⟩ :=
    plainLevels_verifyRepWith _ spec (Setup.ofCircuit st) sch sess 0 k _ c hsch hfi hkl hrep
  rw [hpts, hext] at hext'
  cases hext'
  have hys : y = stream := Option.some.inj (hy.symm.trans hstream)
  subst hys
  have hcoins := forall₂_coins (fun _ _ hab => hab.2) hfa
  have hsched : (Claims.Flock.plainProofShape (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog (2 + extras.size)
      (Setup.ofCircuit st).merkle.digestLen (saltLenOf false (Setup.ofCircuit st))).sched = sch := by
    simp only [Claims.Flock.plainProofShape, Claims.Flock.plainSched, hsch]
    rfl
  have hℓ : 1 ≤ ℓ ∧ ℓ < sch.levels.size := by
    by_contra hn
    rw [plainLevelColsOf, ite_eq_right (by rw [hsched]; exact hn)] at ha
    simp at ha
  obtain ⟨hent, hmc⟩ := hlev ℓ hℓ.1 hℓ.2
  have hcolsEq : plainLevelColsOf (Claims.Flock.plainProofShape (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog
      (2 + extras.size) (Setup.ofCircuit st).merkle.digestLen (saltLenOf false (Setup.ofCircuit st)))
      (lv.streams.getD k #[]) (f.proofs.getD k ByteArray.empty) ℓ =
      Claims.Flock.zkRepCols (sch.levels.getD ℓ default)
        (y.rounds[plainLevelRound (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog sch.k0 (2 + extras.size)
          (sch.levels.size - 1) ℓ]!).coins (openAt p.ligerito (sch.levels.size - 1) ℓ).1 := by
    rw [← hent]
    have e1 : lv.streams.getD k #[] = lv.streams[k] := by
      rw [show (#[] : Array Flock.Round) = default from rfl, getD_eq_getElem!, getElem!_pos lv.streams k hkL]
    have e2 : f.proofs.getD k ByteArray.empty = f.proofs[k]! := by
      rw [show ByteArray.empty = default from rfl, getD_eq_getElem!]
    rw [e1, e2]
    refine plainLevelColsOf_congr _ _ _ fun j => ?_
    rw [← Array.getElem?_toList, ← Array.getElem?_toList]
    exact hcoins j
  rw [hcolsEq] at ha
  obtain ⟨j, hj, -, hjq, hja⟩ := mem_zkRepCols ha
  have hms : (Setup.ofCircuit st).merkle = MerkleScheme.hm96Sha512 ∨
      (Setup.ofCircuit st).merkle = MerkleScheme.sha512 "flock-leaf/sha512-unsalted" := by
    rcases checks_leaf hc (setupOf_some hs).2 0 hsts0 with h | ⟨-, h⟩
    · rw [hst0] at h
      exact Or.inl h
    · rw [hst0] at h
      exact Or.inr h
  have hdl : (Setup.ofCircuit st).merkle.digestLen = 64 := by
    rcases hms with e | e <;> rw [e] <;> rfl
  have hruns := With.merkleCheckWith_sim (back (L.map Prod.fst))
    (With.opsWith_sim (back (L.map Prod.fst)) (back_hashes (L.map Prod.fst)) (Setup.ofCircuit st).merkle)
    (Setup.ofCircuit st).merkle _ _ _ _ _ _ _ _ () hmc
  obtain ⟨hpure, hlog⟩ := merkleCheck_logsOf hms hruns
  obtain ⟨hcs, hcw⟩ := merkleCheck_cap hpure
  have hop : ∀ i < (positions (sch.levels.getD ℓ default) (y.rounds[plainLevelRound (Setup.ofCircuit st).m
      (Setup.ofCircuit st).kLog sch.k0 (2 + extras.size) (sch.levels.size - 1) ℓ]!).coins).size,
      Opening (Setup.ofCircuit st).merkle (sch.levels.getD ℓ default) (2 ^ (sch.levels.getD ℓ default).lanes)
        (p.ligerito.recursiveCaps[ℓ - 1]!.foldl (· ++ ·) ByteArray.empty) L
        (positions (sch.levels.getD ℓ default) (y.rounds[plainLevelRound (Setup.ofCircuit st).m
          (Setup.ofCircuit st).kLog sch.k0 (2 + extras.size) (sch.levels.size - 1) ℓ]!).coins)[i]!
        (openAt p.ligerito (sch.levels.size - 1) ℓ).1[i]! := fun i hi =>
    ⟨hms, p.ligerito.recursiveCaps[ℓ - 1]!,
      (openAt p.ligerito (sch.levels.size - 1) ℓ).2.2.getD i ByteArray.empty,
      sibsOf (openAt p.ligerito (sch.levels.size - 1) ℓ).2.1
        ((sch.levels.getD ℓ default).d - (sch.levels.getD ℓ default).cap) i,
      hcs, fun b hb => (hcw b hb).trans hdl, rfl, merkleCheck_opens _ _ _ _ _ _ _ _ _ hpure i hi,
      fun b hb => (merkleCheck_sibs _ _ _ _ _ _ _ _ _ hpure i b hb).trans hdl, hlog i hi⟩
  have hn : extras.size = st.regions.size * spec.points := by
    rw [extraClaims_size (st := st) hext, ← hpts, hpn]
  rw [hn] at hp'
  exact ⟨s, spec, sts, f, st, sch, p, _, hs, hf, hst, hsch, hp', hja, hcs, fun b hb => (hcw b hb).trans hdl,
    hjq ▸ hop j hj⟩

/-- `plain_levelOpeningK` at the deployed plain verifier (`keep := id`). -/
theorem plain_levelOpening {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω}
    {r : Fin ((V false).k x) → Coin} {k ℓ q : ℕ} {a : List FlockLevel3.GF128}
    (h : (Claims.run sha512 P (V false) x ω r).out = .acc)
    (ha : (q, a) ∈ plainRawLevelCols (Claims.run sha512 P (V false) x ω r).seen k ℓ) :
    ∃ (s : Ins) (spec : SessionSpec) (sts : Array Stmt) (f : Final) (st : Stmt) (sch : Flock.Schedule) (p : Proof)
      (row : Array F128), setupOf false x = some (s, spec, sts) ∧
      lastFinal (Claims.run sha512 P (V false) x ω r).seen = some f ∧ sts[0]? = some st ∧
      fast100 st.m = some sch ∧
      decodeProof (shapeOf (Setup.ofCircuit st) sch (st.regions.size * spec.points)) f.proofs[k]! = .ok p ∧
      a = row.toList.map FlockLevel3.GF128.of ∧
      p.ligerito.recursiveCaps[ℓ - 1]!.size = 2 ^ (sch.levels.getD ℓ default).cap ∧
      (∀ b ∈ p.ligerito.recursiveCaps[ℓ - 1]!.toList, b.size = 64) ∧
      Opening (Setup.ofCircuit st).merkle (sch.levels.getD ℓ default) (2 ^ (sch.levels.getD ℓ default).lanes)
        (p.ligerito.recursiveCaps[ℓ - 1]!.foldl (· ++ ·) ByteArray.empty) (Claims.run sha512 P (V false) x ω r).vLog
        q row :=
  plain_levelOpeningK id (Ω := Ω) (P := P) (x := x) (ω := ω) (r := r) (k := k) (ℓ := ℓ) (q := q) (a := a) (h := h) (ha := ha)

/-- **Two rows one rep of an accepted plain run opens at one later-level position clash in its log**: two of table 0's
rep `k`'s level-`ℓ` rows at one position that differ are two `Opening`s of that position against one root, the
flattened `ℓ − 1`-th recursive cap of the rep's proof (`plain_levelOpening`). -/
theorem plain_levelClashK (keep : Bool → Bool) {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω}
    {r : Fin ((V false).k x) → Coin} {k ℓ q : ℕ} {a b : List FlockLevel3.GF128}
    (h : (Claims.run sha512 P (VK keep) x ω r).out = .acc)
    (ha : (q, a) ∈ plainRawLevelCols (Claims.run sha512 P (VK keep) x ω r).seen k ℓ)
    (hb : (q, b) ∈ plainRawLevelCols (Claims.run sha512 P (VK keep) x ω r).seen k ℓ) (hab : a ≠ b) :
    ∃ (ms : MerkleScheme) (lv : Flock.Level) (lanes : ℕ),
      PlainLevelClash ms lv lanes (Claims.run sha512 P (VK keep) x ω r).vLog := by
  obtain ⟨s, spec, sts, f, st, sch, p, row, hs, hf, hst, hsch, hp, hra, -, -, hop⟩ := plain_levelOpeningK keep h ha
  obtain ⟨s', spec', sts', f', st', sch', p', row', hs', hf', hst', hsch', hp', hrb, -, -, hop'⟩ :=
    plain_levelOpeningK keep h hb
  obtain ⟨rfl, rfl, rfl⟩ : s = s' ∧ spec = spec' ∧ sts = sts' := by simpa using hs.symm.trans hs'
  obtain rfl := Option.some.inj (hf.symm.trans hf')
  obtain rfl := Option.some.inj (hst.symm.trans hst')
  obtain rfl := Option.some.inj (hsch.symm.trans hsch')
  obtain rfl := Except.ok.inj (hp.symm.trans hp')
  exact ⟨_, _, _, _, q, row, row', fun he => hab (by rw [hra, hrb, he]), hop, hop'⟩

/-- `plain_levelClashK` at the deployed plain verifier (`keep := id`). -/
theorem plain_levelClash {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω}
    {r : Fin ((V false).k x) → Coin} {k ℓ q : ℕ} {a b : List FlockLevel3.GF128}
    (h : (Claims.run sha512 P (V false) x ω r).out = .acc)
    (ha : (q, a) ∈ plainRawLevelCols (Claims.run sha512 P (V false) x ω r).seen k ℓ)
    (hb : (q, b) ∈ plainRawLevelCols (Claims.run sha512 P (V false) x ω r).seen k ℓ) (hab : a ≠ b) :
    ∃ (ms : MerkleScheme) (lv : Flock.Level) (lanes : ℕ),
      PlainLevelClash ms lv lanes (Claims.run sha512 P (V false) x ω r).vLog :=
  plain_levelClashK id (Ω := Ω) (P := P) (x := x) (ω := ω) (r := r) (k := k) (ℓ := ℓ) (q := q) (a := a) (b := b) (h := h) (ha := ha) (hb := hb) (hab := hab)

end FlockDeploy.KS
