import Proofs.Flock.Claims.ZkCols
import Proofs.Flock.Soundness.Discharge.ZkExec.PlaysLig
import Proofs.Flock.Soundness.Discharge.ZkExec.RunsSeams
import Proofs.Flock.Soundness.Discharge.ZkExec.Verify
import Flock.With.ZkEq

/-!
# Level 0's query round in the masked verifier over the hash oracle

`zkQueryRound m kLog k0 claims` (`Claims.Flock.ZkCols`) is the round of rep `r`'s stream whose coins are level 0's
query positions. Here it is proved from the masked verifier's own transcript, `Zk.verifyRepWith hs` run in `V` with any
hash oracle `hs` (so `Zk.verifyWith h = Zk.verifyWithM (Hashes.ofSha512 h)` for every `h`, and `Zk.verify` itself
when every table's Merkle scheme is the registry's, `Zk.verify_eq_verifyWith`):

* a squeeze of `n ≠ 0` coins through the oracle consumes the next round and returns its coins (`squeezeWith_ok`): the
  oracle only answers a round's digest, so the round accounting is the compiled verifier's under every oracle;
* the masked replay consumes `m + (kLog − 6)` rounds (`replayWith_moves`), the ring switch `claims + 1`
  (`ringSwitchWith_moves`);
* Ligerito draws level 0's query coins after `k0 + 8` rounds of its own (`ligeritoWith_query`): the non-ZK
  verifier's `2 + k0 + 4` (the OOD point and its coin, the `k0` lane challenges, level 1's two OOD intros; the
  accounting `Refine.ligerito_refines` proves) and the pair's two `ρ` coins after the last lane challenge;
* so an accepted rep passed level 0's Merkle check on its proof's rows at the positions of round `zkQueryRound` of its
  stream (`zkQueryRound_verifyRepWith`), and the reader's columns (`zkRepColsOf`) are those positions with those rows'
  witness lanes, their first `2^k0` symbols (`zkRepColsOf_verifyRepWith`), for every rep of every table of an
  accepted session (`zkQueryRound_verifyWithM`; for `Zk.verify` on the registry's Merkle schemes,
  `zkQueryRound_verify`).

Each table is checked on its own: its reps' streams, its schedule's level 0 and its own `root_B[i]`. A multi-table
session commits one level-0 codeword per table, so `zkRunCols` reads one table's.
-/

namespace FlockSoundness.ZkQuery

open Flock Transcript Refine Discharge.Exec Discharge.ZkExec

/-! ## The oracle's squeeze, in `V` -/

/-- A squeeze of `n ≠ 0` coins through any oracle consumes round `t.k`, which issued `n` coins, and returns its
coins. -/
theorem squeezeWith_ok (hs : Hashes V) {kind : UInt8} {n : ℕ} (hn : n ≠ 0) {t t₁ : Flock.Transcript}
    {cs : Array F128} (h : squeezeWith hs kind n t = .ok (cs, t₁)) :
    (t.rounds[t.k]!).n = n ∧ cs = (t.rounds[t.k]!).coins ∧ t₁.rounds = t.rounds ∧ t₁.k = t.k + 1 := by
  unfold squeezeWith at h
  rw [ite_eq_right hn] at h
  split at h
  · cases h
  · rename_i r hr
    have hlt : t.k < t.rounds.size := by
      by_contra hge; rw [Array.getElem?_eq_none (by omega)] at hr; cases hr
    have hget : t.rounds[t.k]! = r := by
      rw [getElem!_pos t.rounds t.k hlt]; exact Option.some.inj ((Array.getElem?_eq_getElem hlt).symm.trans hr)
    rw [hget]
    by_cases h1 : r.n = n
    · rw [ite_eq_right (not_not.2 h1)] at h
      obtain ⟨ok, -, h⟩ := ebind_ok h
      split at h
      · cases h
      · cases h
        exact ⟨h1, rfl, rfl, rfl⟩
    · rw [ite_eq_left h1] at h
      cases h

theorem sampleWith_ok (hs : Hashes V) {t t₁ : Flock.Transcript} {c : F128} (h : sampleWith hs t = .ok (c, t₁)) :
    (t.rounds[t.k]!).n = 1 ∧ c = (t.rounds[t.k]!).coins.getD 0 0 ∧ t₁.rounds = t.rounds ∧ t₁.k = t.k + 1 := by
  unfold sampleWith at h
  obtain ⟨cs, t₂, h₁, h₂⟩ := bind_ok h
  obtain ⟨hn, hcs, hr, hk⟩ := squeezeWith_ok hs one_ne_zero h₁
  obtain ⟨rfl, rfl⟩ := pure_ok h₂
  exact ⟨hn, hcs ▸ rfl, hr, hk⟩

theorem sampleVecWith_ok (hs : Hashes V) {n : ℕ} (hn : n ≠ 0) {t t₁ : Flock.Transcript} {cs : Array F128}
    (h : sampleVecWith hs n t = .ok (cs, t₁)) :
    (t.rounds[t.k]!).n = n ∧ cs = (t.rounds[t.k]!).coins ∧ t₁.rounds = t.rounds ∧ t₁.k = t.k + 1 :=
  squeezeWith_ok hs hn h

theorem sampleKWith_ok (hs : Hashes V) {t t₁ : Flock.Transcript} {c : F256} (h : sampleKWith hs t = .ok (c, t₁)) :
    (t.rounds[t.k]!).n = 2 ∧ t₁.rounds = t.rounds ∧ t₁.k = t.k + 1 := by
  unfold sampleKWith at h
  obtain ⟨cs, t₂, h₁, h₂⟩ := bind_ok h
  obtain ⟨hn, -, hr, hk⟩ := squeezeWith_ok hs (by decide) h₁
  obtain ⟨rfl, rfl⟩ := pure_ok h₂
  exact ⟨hn, hr, hk⟩

theorem squeezeWith_lt (hs : Hashes V) {kind : UInt8} {n : ℕ} (hn : n ≠ 0) {t t₁ : Flock.Transcript}
    {cs : Array F128} (h : squeezeWith hs kind n t = .ok (cs, t₁)) : t.k < t.rounds.size := by
  unfold squeezeWith at h
  rw [ite_eq_right hn] at h
  split at h
  · cases h
  · rename_i r hr
    by_contra hge; rw [Array.getElem?_eq_none (by omega)] at hr; cases hr

/-! ## Loops and phases -/

/-- A loop whose every step yields and consumes `c` rounds consumes `n · c` rounds over `n` steps. -/
theorem forIn_moves {σ : Type} (body : ℕ → σ → TM (ForInStep σ)) (c : ℕ)
    (hbody : ∀ i s t b t', body i s t = .ok (b, t') →
      (∃ s', b = .yield s') ∧ t'.rounds = t.rounds ∧ t'.k = t.k + c) :
    ∀ (n a : ℕ) (s : σ) (t t' : Flock.Transcript) (out : σ),
      (forIn (List.range' a n) s body : TM σ) t = .ok (out, t') → t'.rounds = t.rounds ∧ t'.k = t.k + n * c
  | 0, a, s, t, t', out, h => by
      simp only [List.range'_zero, List.forIn_nil] at h
      obtain ⟨rfl, rfl⟩ := pure_ok h
      simp
  | n + 1, a, s, t, t', out, h => by
      rw [forIn_range'_cons] at h
      obtain ⟨b, t₁, h₁, h₂⟩ := bind_ok h
      obtain ⟨⟨s₁, rfl⟩, hr₁, hk₁⟩ := hbody a s t b t₁ h₁
      obtain ⟨hr, hk⟩ := forIn_moves body c hbody n (a + 1) s₁ t₁ t' out h₂
      exact ⟨hr.trans hr₁, by rw [hk, hk₁, Nat.succ_mul]; omega⟩

/-- **The masked replay's rounds**: `m + (kLog − 6)` (two outer coin rounds, `z`, `m − 6` zerocheck rounds, `α`, `β`,
`kLog − 6` lincheck rounds and `r′`), at `m ≥ 14`. -/
theorem replayWith_moves (hs : Hashes V) (st : Setup) (lay : Zk.PadLayout) (p : Proof) (root : ByteArray)
    (hm : 14 ≤ st.m) {t t' : Flock.Transcript} {rp : Zk.Replayed}
    (h : Zk.replayWith hs st lay p root t = .ok (rp, t')) :
    t'.rounds = t.rounds ∧ t'.k = t.k + (st.m + (st.kLog - 6)) := by
  unfold Zk.replayWith at h
  simp only [With.liftTM_V] at h
  obtain ⟨t₁, k₁, h⟩ := label_bind h
  obtain ⟨t₂, k₂, h⟩ := absorbBytes_bind h
  obtain ⟨t₃, k₃, h⟩ := absorbBytes_bind h
  obtain ⟨t₄, k₄, h⟩ := label_bind h
  obtain ⟨t₅, k₅, h⟩ := label_bind h
  obtain ⟨t₆, k₆, h⟩ := absorbBytes_bind h
  obtain ⟨_, t₇, h₇, h⟩ := bind_ok h
  obtain ⟨-, -, hr₇, hk₇⟩ := sampleVecWith_ok hs (by decide) h₇
  obtain ⟨_, t₈, h₈, h⟩ := bind_ok h
  obtain ⟨-, -, hr₈, hk₈⟩ := sampleVecWith_ok hs (by omega) h₈
  obtain ⟨t₉, k₉, h⟩ := absorbSlice_bind h
  obtain ⟨t₁₀, k₁₀, h⟩ := absorbSlice_bind h
  obtain ⟨zz, t₁₁, h₁₁, h⟩ := bind_ok h
  obtain ⟨-, -, hr₁₁, hk₁₁⟩ := sampleWith_ok hs h₁₁
  obtain ⟨s, t₁₂, hloop, h⟩ := bind_ok h
  rw [forIn_range_eq] at hloop
  obtain ⟨hr₁₂, hk₁₂⟩ := forIn_moves _ 1 (by
      intro i s tt b tt' hb
      obtain ⟨u₁, f₁, hb⟩ := absorbF_bind hb
      obtain ⟨u₂, f₂, hb⟩ := absorbF_bind hb
      obtain ⟨_, u₃, g₃, hb⟩ := bind_ok hb
      obtain ⟨-, -, hr₃, hk₃⟩ := sampleWith_ok hs g₃
      obtain ⟨rfl, rfl⟩ := pure_ok hb
      exact ⟨⟨_, rfl⟩, by rw [hr₃, f₂.2, f₁.2], by rw [hk₃, f₂.1, f₁.1]⟩) (st.m - 6) 0 _ _ _ _ hloop
  obtain ⟨t₁₃, k₁₃, h⟩ := absorbF_bind h
  obtain ⟨t₁₄, k₁₄, h⟩ := absorbF_bind h
  obtain ⟨t₁₅, k₁₅, h⟩ := label_bind h
  obtain ⟨α₀, t₁₆, h₁₆, h⟩ := bind_ok h
  obtain ⟨-, -, hr₁₆, hk₁₆⟩ := sampleWith_ok hs h₁₆
  obtain ⟨β₀, t₁₇, h₁₇, h⟩ := bind_ok h
  obtain ⟨-, -, hr₁₇, hk₁₇⟩ := sampleWith_ok hs h₁₇
  obtain ⟨s₂, t₁₈, hloop₂, h⟩ := bind_ok h
  rw [forIn_range_eq] at hloop₂
  obtain ⟨hr₁₈, hk₁₈⟩ := forIn_moves _ 1 (by
      intro j s tt b tt' hb
      obtain ⟨u₁, f₁, hb⟩ := absorbF_bind hb
      obtain ⟨u₂, f₂, hb⟩ := absorbF_bind hb
      obtain ⟨_, u₃, g₃, hb⟩ := bind_ok hb
      obtain ⟨-, -, hr₃, hk₃⟩ := sampleWith_ok hs g₃
      obtain ⟨rfl, rfl⟩ := pure_ok hb
      exact ⟨⟨_, rfl⟩, by rw [hr₃, f₂.2, f₁.2], by rw [hk₃, f₂.1, f₁.1]⟩) (st.kLog - 6) 0 _ _ _ _ hloop₂
  cases hf : st.fold.folded α₀ β₀ (skipWeights zz) (s.2.extract 0 (st.kLog - 6)) s₂.2 with
  | error e =>
    simp only [hf] at h
    obtain ⟨_, _, h₁₉, _⟩ := bind_ok h
    cases h₁₉
  | ok combv =>
    simp only [hf, pure_bind] at h
    obtain ⟨-, h⟩ := check_bind h
    obtain ⟨t₂₀, k₂₀, h⟩ := absorbSlice_bind h
    obtain ⟨_, t₂₁, h₂₁, h⟩ := bind_ok h
    obtain ⟨-, -, hr₂₁, hk₂₁⟩ := sampleWith_ok hs h₂₁
    obtain ⟨-, rfl⟩ := pure_ok h
    refine ⟨by rw [hr₂₁, k₂₀.2, hr₁₈, hr₁₇, hr₁₆, k₁₅.2, k₁₄.2, k₁₃.2, hr₁₂, hr₁₁, k₁₀.2, k₉.2, hr₈, hr₇, k₆.2,
      k₅.2, k₄.2, k₃.2, k₂.2, k₁.2], ?_⟩
    rw [hk₂₁, k₂₀.1, hk₁₈, hk₁₇, hk₁₆, k₁₅.1, k₁₄.1, k₁₃.1, hk₁₂, hk₁₁, k₁₀.1, k₉.1, hk₈, hk₇, k₆.1, k₅.1, k₄.1,
      k₃.1, k₂.1, k₁.1]
    omega

/-- **The ring switch's rounds**: one per claim and the batching coins, `claims + 1`. -/
theorem ringSwitchWith_moves (hs : Hashes V) (m : ℕ) (claims : Array Claim) (rs : Array (Array F128))
    (hn : claims.size ≠ 0) {t t' : Flock.Transcript} {res : F128 × Array Switched}
    (h : ringSwitchWith hs m claims rs t = .ok (res, t')) :
    t'.rounds = t.rounds ∧ t'.k = t.k + (claims.size + 1) := by
  unfold ringSwitchWith at h
  simp only [With.liftTM_V] at h
  obtain ⟨t₁, k₁, h⟩ := label_bind h
  obtain ⟨s, t₂, hloop, h⟩ := bind_ok h
  rw [forIn_range_eq] at hloop
  obtain ⟨hr₂, hk₂⟩ := forIn_moves _ 1 (by
      intro j s tt b tt' hb
      obtain ⟨u₁, f₁, hb⟩ := label_bind hb
      obtain ⟨u₂, f₂, hb⟩ := absorbSlice_bind hb
      obtain ⟨-, hb⟩ := check_bind hb
      obtain ⟨_, u₃, g₃, hb⟩ := bind_ok hb
      obtain ⟨-, -, hr₃, hk₃⟩ := sampleVecWith_ok hs (by decide) g₃
      obtain ⟨rfl, rfl⟩ := pure_ok hb
      exact ⟨⟨_, rfl⟩, by rw [hr₃, f₂.2, f₁.2], by rw [hk₃, f₂.1, f₁.1]⟩) claims.size 0 _ _ _ _ hloop
  obtain ⟨_, t₃, h₃, h⟩ := bind_ok h
  obtain ⟨-, -, hr₃, hk₃⟩ := sampleVecWith_ok hs hn h₃
  obtain ⟨-, rfl⟩ := pure_ok h
  exact ⟨by rw [hr₃, hr₂, k₁.2], by rw [hk₃, hk₂, k₁.1]; omega⟩

/-! ## Ligerito up to level 0's queries -/

/-- `oodIntroWith`: an out-of-domain point and its batching coin, two rounds. -/
theorem oodIntroWith_bind (hs : Hashes V) {p : LigeritoProof} {dim split : ℕ} (hdim : dim ≠ 0) {β : Type}
    {f : PUnit → LM β} {st st' : LState} {t t' : Flock.Transcript} {b : β}
    (h : (oodIntroWith hs p dim split >>= f) st t = .ok ((b, st'), t')) :
    ∃ st₁ t₁, t₁.rounds = t.rounds ∧ t₁.k = t.k + 2 ∧ f () st₁ t₁ = .ok ((b, st'), t') := by
  obtain ⟨⟨⟩, st₁, t₁, h₁, h₂⟩ := lbind_ok h
  suffices hk : t₁.rounds = t.rounds ∧ t₁.k = t.k + 2 from ⟨st₁, t₁, hk.1, hk.2, h₂⟩
  unfold oodIntroWith at h₁
  simp only [With.liftW_V, With.liftLM_V] at h₁
  obtain ⟨z, t₃, h₃, h₁⟩ := lift_bind h₁
  obtain ⟨-, -, hr₃, hk₃⟩ := sampleVecWith_ok hs hdim h₃
  obtain ⟨y, -, h₁⟩ := nextOod_bind h₁
  obtain ⟨_, t₄, h₄, h₁⟩ := lift_bind h₁
  have k₄ := absorbF_ok h₄
  obtain ⟨m, t₅, -, k₅, h₁⟩ := nextMsg_bind h₁
  obtain ⟨t₆, k₆, h₁⟩ := claimPow_bind h₁
  obtain ⟨bc, t₇, h₇, h₁⟩ := lift_bind h₁
  obtain ⟨-, -, hr₇, hk₇⟩ := sampleWith_ok hs h₇
  obtain ⟨-, rfl⟩ := lmodify_ok h₁
  exact ⟨by rw [hr₇, k₆.2, k₅.2, k₄.2, hr₃], by rw [hk₇, k₆.1, k₅.1, k₄.1, hk₃]⟩

/-- **Level 0's queries over the oracle**: the Merkle check passes on the proof's level-0 rows at the positions of the
query coins, round `t.k`'s. -/
theorem queryLevel0With_mc (hs : Hashes V) {ms : MerkleScheme} {p : LigeritoProof} {lv : Flock.Level}
    {cap : Array ByteArray} {fold : Array F256} (hq : lv.q ≠ 0) {β : Type}
    {f : Array ℕ × Array F128 × F256 → LM β} {st st' : LState} {t t' : Flock.Transcript} {b : β}
    (h : (Zk.queryLevel0With hs ms p lv cap fold >>= f) st t = .ok ((b, st'), t')) :
    t.k < t.rounds.size ∧
      merkleCheckWith (ms.opsWith hs) ms cap lv.cap lv.d (positions lv (t.rounds[t.k]!).coins) p.initialProof.rows
        p.initialProof.siblings p.initialProof.salts (2 ^ lv.lanes + Zk.EXTRA_LANES) = .ok () := by
  obtain ⟨r, st₁, t₁, h₁, h₂⟩ := lbind_ok h
  unfold Zk.queryLevel0With at h₁
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
  cases hmc : merkleCheckWith (ms.opsWith hs) ms cap lv.cap lv.d (positions lv (t₃.rounds[t₃.k]!).coins)
      p.initialProof.rows p.initialProof.siblings p.initialProof.salts (2 ^ lv.lanes + Zk.EXTRA_LANES) with
  | error e => simp only [hmc] at h₁; obtain ⟨_, _, _, h₀, _⟩ := lbind_ok h₁; cases h₀
  | ok u =>
    rw [← k₃.2, ← k₃.1]
    exact ⟨hlt, hmc⟩

/-- **Ligerito's rounds before level 0's queries**: `k0 + 8`. The OOD point and its coin, the `k0` lane challenges
with the pair's two `ρ` coins after the last, and level 1's two OOD intros, two rounds each; so level 0's Merkle check
passes at the positions of round `t.k + k0 + 8`'s coins. -/
theorem ligeritoWith_query (hs : Hashes V) (ms : MerkleScheme) (sch : Flock.Schedule) (T : F128)
    (cap : Array ByteArray) (p : LigeritoProof) (basisAt : Array F256 → ℕ → Array F256) (zl : Zk.Level0)
    {t t' : Flock.Transcript} {rec : Zk.L0Rec} (hn : sch.n ≠ 0) (hk0 : 1 ≤ sch.k0)
    (hq : (sch.levels.getD 0 default).q ≠ 0)
    (h : Zk.ligeritoWith hs ms sch T cap p basisAt zl t = .ok (rec, t')) :
    t.k + sch.k0 + 8 < t.rounds.size ∧ merkleCheckWith (ms.opsWith hs) ms cap (sch.levels.getD 0 default).cap (sch.levels.getD 0 default).d
      (positions (sch.levels.getD 0 default) (t.rounds[t.k + sch.k0 + 8]!).coins) p.initialProof.rows
      p.initialProof.siblings p.initialProof.salts (2 ^ (sch.levels.getD 0 default).lanes + Zk.EXTRA_LANES) =
      .ok () := by
  unfold Zk.ligeritoWith at h
  simp only [With.liftTM_V, With.liftW_V, With.liftLM_V] at h
  obtain ⟨-, h⟩ := check_bind h
  obtain ⟨-, h⟩ := check_bind h
  obtain ⟨t₁, k₁, h⟩ := label_bind h
  obtain ⟨t₂, k₂, h⟩ := absorbF_bind h
  obtain ⟨t₃, k₃, h⟩ := absorbBytes_bind h
  obtain ⟨t₃', k₃', h⟩ := label_bind h
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
  obtain ⟨hr₁₀, hk₁₀, -⟩ := lane_coins _ sch.k0 (by
      intro i v st tE bv st' tE' hiB hb
      obtain ⟨r, tR, hR', hb⟩ := lift_bind hb
      obtain ⟨hnR, hrR, hkR⟩ := sampleKWith_ok hs hR'
      have hb := lmodify_bind hb
      try dsimp only at hb
      by_cases hj : i + 1 = sch.k0
      · rw [ite_eq_left (show (i + 1 == sch.k0) = true by simp [hj])] at hb
        obtain ⟨_, t₂, h₂, hb⟩ := lift_bind hb
        obtain ⟨f₂k, f₂r⟩ := absorbK_ok h₂
        obtain ⟨_, t₃, h₃, hb⟩ := lift_bind hb
        obtain ⟨f₃k, f₃r⟩ := absorbK_ok h₃
        obtain ⟨ρ₀, t₄, h₄, hb⟩ := lift_bind hb
        obtain ⟨hn₄, hr₄, hk₄⟩ := sampleKWith_ok hs h₄
        obtain ⟨ρ₁, t₅, h₅, hb⟩ := lift_bind hb
        obtain ⟨hn₅, hr₅, hk₅⟩ := sampleKWith_ok hs h₅
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
        refine ⟨_, rfl, by rw [f₉r, f₈r, f₇r, f₆r, hr₅, hr₄, f₃r, f₂r, hrR], by rw [ite_eq_left hj]; omega, hnR,
          fun _ => ⟨?_, ?_⟩⟩
        · rw [show tE.k + 1 = t₃.k by omega, show tE.rounds = t₃.rounds by rw [f₃r, f₂r, hrR]]
          exact hn₄
        · rw [show tE.k + 2 = t₄.k by omega, show tE.rounds = t₄.rounds by rw [hr₄, f₃r, f₂r, hrR]]
          exact hn₅
      · rw [ite_eq_right (show ¬ (i + 1 == sch.k0) = true by simp [hj])] at hb
        try simp only [pure_bind] at hb
        obtain ⟨m, t₉, -, ⟨f₉k, f₉r⟩, hb⟩ := nextMsg_bind hb
        have hb := lmodify_bind hb
        obtain ⟨rfl, rfl, rfl⟩ := lpure_ok hb
        exact ⟨_, rfl, by rw [f₉r, hrR], by rw [ite_eq_right hj]; omega, hnR, fun h => absurd h hj⟩)
    sch.k0 0 _ _ t₉ _ _ t₁₀ (by omega) (by omega) h₁₀
  try dsimp only at h₄
  obtain ⟨_, t₁₁, h₁₁, h₄⟩ := lift_bind h₄
  have k₁₁ := absorbBytes_ok h₁₁
  obtain ⟨st₁₂, t₁₂, hr₁₂, hk₁₂, h₄⟩ := oodIntroWith_bind hs (show sch.n - sch.k0 + 1 ≠ 0 by omega) h₄
  obtain ⟨st₁₃, t₁₃, hr₁₃, hk₁₃, h₄⟩ := oodIntroWith_bind hs (show sch.n - sch.k0 + 1 ≠ 0 by omega) h₄
  have hmc := queryLevel0With_mc hs hq h₄
  have er : t₁₃.rounds = t.rounds := by
    rw [hr₁₃, hr₁₂, k₁₁.2, hr₁₀, k₉.2, hr₈, k₇.2, k₆.2, hr₅, k₃'.2, k₃.2, k₂.2, k₁.2]
  have ek : t₁₃.k = t.k + sch.k0 + 8 := by
    rw [hk₁₃, hk₁₂, k₁₁.1, hk₁₀, k₉.1, hk₈, k₇.1, k₆.1, hk₅, k₃'.1, k₃.1, k₂.1, k₁.1]
    omega
  rw [er, ek] at hmc
  exact hmc

/-! ## An accepted rep, and an accepted session -/

/-- **An accepted rep's level-0 queries sit at round `zkQueryRound` of its stream.** At the `--zk` schedule of its
table, a rep that `Zk.verifyRepWith hs` accepts (in `V`, any oracle) has a stream with that round, and level 0's
Merkle check passed against the committed `root_B` on the proof's opened rows at the positions of that round's coins. -/
theorem zkQueryRound_verifyRepWith (hs : Hashes V) (spec : SessionSpec) (st : Setup) (sch : Flock.Schedule)
    (tPad : ℕ) (sess : RunRecord) (i r : ℕ) (bytes cap : ByteArray) (hsch : Flock.Zk.schedule st.m = .ok (sch, tPad))
    (h : Zk.verifyRepWith hs spec st sch tPad sess i r bytes = .ok cap) :
    ∃ (stream : StreamRec) (extras : Array Flock.Claim) (zp : Flock.Zk.ZkProof),
      sess.streams.find? (fun x => x.name == s!"{spec.tables[i]!}/rep{r}") = some stream ∧
      st.extra sess.points = .ok extras ∧ Flock.Zk.decode (shapeZK st sch extras) bytes = .ok zp ∧
      Claims.Flock.zkQueryRound st.m st.kLog sch.k0 (2 + extras.size) < stream.rounds.size ∧
      merkleCheckWith (st.merkle.opsWith hs) st.merkle zp.proof.cap (sch.levels.getD 0 default).cap
        (sch.levels.getD 0 default).d
        (positions (sch.levels.getD 0 default)
          (stream.rounds[Claims.Flock.zkQueryRound st.m st.kLog sch.k0 (2 + extras.size)]!).coins)
        zp.proof.ligerito.initialProof.rows zp.proof.ligerito.initialProof.siblings
        zp.proof.ligerito.initialProof.salts (2 ^ (sch.levels.getD 0 default).lanes + Zk.EXTRA_LANES) = .ok () ∧
      cap = zp.proof.cap.foldl (· ++ ·) ByteArray.empty ∧ cap = sess.rootB[i]! := by
  obtain ⟨h25, -, hsm, hk0, -, hdim, hL, -, -⟩ := zk_schedule_seams _ _ _ hsch
  have hq : (sch.levels.getD 0 default).q ≠ 0 := (hL 0 (by have := hdim.2.1; omega)).q_ne_zero
  have hn : sch.n ≠ 0 := by rw [hdim.1, hsm]; omega
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
  obtain ⟨⟨⟨⟩, tr⟩, hrun, h⟩ := ebind_ok h
  dsimp only at h
  by_cases c2 : (tr.k != stream.rounds.size) = true
  · rw [ite_eq_left c2] at h; obtain ⟨_, h0, _⟩ := ebind_ok h; cases h0
  rw [ite_eq_right c2] at h
  by_cases c3 : (zp.proof.cap.foldl (· ++ ·) ByteArray.empty != sess.rootB[i]!) = true
  · rw [ite_eq_left c3] at h; obtain ⟨_, h0, _⟩ := ebind_ok h; cases h0
  rw [ite_eq_right c3] at h
  simp only [pure, Except.pure, Except.ok.injEq] at h
  subst h
  have hb : (zp.proof.cap.foldl (· ++ ·) ByteArray.empty == sess.rootB[i]!) = true := by
    cases hb : (zp.proof.cap.foldl (· ++ ·) ByteArray.empty == sess.rootB[i]!)
    · exact absurd (by simp [bne, hb]) c3
    · rfl
  rw [StateT.run] at hrun
  obtain ⟨rp, t₁, h₁, hrun⟩ := bind_ok hrun
  obtain ⟨hr₁, hk₁⟩ := replayWith_moves hs st _ _ _ (by omega) h₁
  obtain ⟨⟨T, sws⟩, t₂, h₂, hrun⟩ := bind_ok hrun
  obtain ⟨hr₂, hk₂⟩ := ringSwitchWith_moves hs _ _ _ (by simp) h₂
  obtain ⟨rec, t₃, h₃, -⟩ := bind_ok hrun
  obtain ⟨hlt, hmc⟩ := ligeritoWith_query hs _ _ _ _ _ _ _ hn hk0 hq h₃
  have er : t₂.rounds = stream.rounds := by rw [hr₂, hr₁]; rfl
  have ek : t₂.k + sch.k0 + 8 = Claims.Flock.zkQueryRound st.m st.kLog sch.k0 (2 + extras.size) := by
    rw [hk₂, hk₁]; simp [Claims.Flock.zkQueryRound, Transcript.init]; omega
  rw [er, ek] at hlt hmc
  exact ⟨stream, extras, zp, hstream, hext, hzp, hlt, hmc, rfl, bytes_eq_of_beq hb⟩

/-- `zkProofShape` at a table's `--zk` schedule is the shape `Zk.verifyRep` decodes its proofs at. -/
theorem zkProofShape_eq (st : Setup) (sch : Flock.Schedule) (tPad : ℕ) (extras : Array Flock.Claim)
    (hsch : Flock.Zk.schedule st.m = .ok (sch, tPad)) :
    Claims.Flock.zkProofShape st.m st.kLog (2 + extras.size) st.merkle.digestLen
      (if st.merkle.saltBytes = 0 then none else some st.merkle.saltBytes) = shapeZK st sch extras := by
  simp only [Claims.Flock.zkProofShape, Claims.Flock.zkSched, hsch, shapeZK]

/-- **The reader's columns of an accepted rep are the verifier's openings**: `zkRepColsOf` on the rep's stream and
proof file is the witness lanes (the first `2^k0` symbols) of the proof's level-0 rows at the positions of round
`zkQueryRound`'s coins, the positions and rows on which level 0's Merkle check passed against the committed `root_B`
(`zkQueryRound_verifyRepWith`). -/
theorem zkRepColsOf_verifyRepWith (hs : Hashes V) (spec : SessionSpec) (st : Setup) (sch : Flock.Schedule)
    (tPad : ℕ) (sess : RunRecord) (i r : ℕ) (bytes cap : ByteArray) (hsch : Flock.Zk.schedule st.m = .ok (sch, tPad))
    (h : Zk.verifyRepWith hs spec st sch tPad sess i r bytes = .ok cap) :
    ∃ (stream : StreamRec) (extras : Array Flock.Claim) (zp : Flock.Zk.ZkProof),
      sess.streams.find? (fun x => x.name == s!"{spec.tables[i]!}/rep{r}") = some stream ∧
      st.extra sess.points = .ok extras ∧ Flock.Zk.decode (shapeZK st sch extras) bytes = .ok zp ∧
      Claims.Flock.zkQueryRound st.m st.kLog sch.k0 (2 + extras.size) < stream.rounds.size ∧
      Claims.Flock.zkRepColsOf (Claims.Flock.zkProofShape st.m st.kLog (2 + extras.size) st.merkle.digestLen
          (if st.merkle.saltBytes = 0 then none else some st.merkle.saltBytes)) stream.rounds bytes =
        (Claims.Flock.zkRepCols (sch.levels.getD 0 default)
          (stream.rounds[Claims.Flock.zkQueryRound st.m st.kLog sch.k0 (2 + extras.size)]!).coins
          zp.proof.ligerito.initialProof.rows).map (fun p => (p.1, p.2.take (2 ^ (sch.levels.getD 0 default).lanes))) ∧
      merkleCheckWith (st.merkle.opsWith hs) st.merkle zp.proof.cap (sch.levels.getD 0 default).cap
        (sch.levels.getD 0 default).d
        (positions (sch.levels.getD 0 default)
          (stream.rounds[Claims.Flock.zkQueryRound st.m st.kLog sch.k0 (2 + extras.size)]!).coins)
        zp.proof.ligerito.initialProof.rows zp.proof.ligerito.initialProof.siblings
        zp.proof.ligerito.initialProof.salts (2 ^ (sch.levels.getD 0 default).lanes + Zk.EXTRA_LANES) = .ok () ∧
      zp.proof.cap.foldl (· ++ ·) ByteArray.empty = sess.rootB[i]! := by
  obtain ⟨stream, extras, zp, hstream, hext, hzp, hlt, hmc, hcap, hroot⟩ :=
    zkQueryRound_verifyRepWith hs spec st sch tPad sess i r bytes cap hsch h
  refine ⟨stream, extras, zp, hstream, hext, hzp, hlt, ?_, hmc, hcap ▸ hroot⟩
  rw [zkProofShape_eq st sch tPad extras hsch, Claims.Flock.zkRepColsOf, hzp]
  simp only [shapeZK, Array.getElem?_eq_getElem hlt, Option.map_some, Option.getD_some,
    getElem!_pos stream.rounds (Claims.Flock.zkQueryRound st.m st.kLog sch.k0 (2 + extras.size)) hlt]

/-- **`Zk.verifyWithM`'s accepted session, table by table and rep by rep**: in `V`, under any oracle, an accepted
session decodes, has one statement per table, and each table `i`'s rep `r` (`r < 2`, proof file `2i + r`) is accepted
by `Zk.verifyRepWith` at the table's `--zk` schedule. -/
theorem verifyWithM_reps (hs : Hashes V) (spec : SessionSpec) (tabs : Array Setup) (record : String)
    (proofs : Array ByteArray) (h : Zk.verifyWithM hs spec tabs record proofs = .ok ()) :
    ∃ sess : RunRecord, Record.decodeWith hs spec record = .ok sess ∧ tabs.size = spec.tables.size ∧
      ∀ (i : ℕ) (hi : i < tabs.size), ∃ (sch : Flock.Schedule) (tPad : ℕ) (c : ByteArray),
        Flock.Zk.schedule tabs[i].m = .ok (sch, tPad) ∧ ∀ r < 2,
          Zk.verifyRepWith hs spec tabs[i] sch tPad sess i r proofs[2 * i + r]! = .ok c := by
  unfold Zk.verifyWithM at h
  dsimp only at h
  by_cases c0 : (tabs.size != spec.tables.size || tabs.size == 0) = true
  · rw [ite_eq_left c0] at h; obtain ⟨_, h0, _⟩ := ebind_ok h; cases h0
  rw [ite_eq_right c0] at h
  by_cases c1 : spec.coinTree.isNone = true
  · rw [ite_eq_left c1] at h; obtain ⟨_, h0, _⟩ := ebind_ok h; cases h0
  rw [ite_eq_right c1] at h
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
    (fun i => ∃ c, Zk.verifyRepWith hs spec tabs[i]! (schs[i]!).1 (schs[i]!).2 sess i 0 proofs[2 * i]! = .ok c ∧
      Zk.verifyRepWith hs spec tabs[i]! (schs[i]!).1 (schs[i]!).2 sess i 1 proofs[2 * i + 1]! = .ok c)
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
  refine ⟨schs[i].1, schs[i].2, c, hsch i hi hs', fun r hr => ?_⟩
  interval_cases r
  · simpa using h0
  · exact h1

/-- **Level 0's query round in an accepted `--zk` session** (`Zk.verifyWithM` in `V`, any oracle): for every table `i`
and rep `r < 2`, the rep's stream has round `zkQueryRound m kLog k0 claims` at the table's `m`, `kLog`, schedule and
claim count (`2 +` its extra claims at the session's link points), the reader's columns of the rep (`zkRepColsOf` at
`zkProofShape`) are the witness lanes of the proof's
level-0 rows at the positions of that round's coins, and level 0's Merkle check passed on exactly those rows at exactly
those positions against the table's committed `root_B`. -/
theorem zkQueryRound_verifyWithM (hs : Hashes V) (spec : SessionSpec) (tabs : Array Setup) (record : String)
    (proofs : Array ByteArray) (h : Zk.verifyWithM hs spec tabs record proofs = .ok ()) :
    ∃ sess : RunRecord, Record.decodeWith hs spec record = .ok sess ∧ tabs.size = spec.tables.size ∧
      ∀ (i : ℕ) (hi : i < tabs.size), ∃ (sch : Flock.Schedule) (tPad : ℕ),
        Flock.Zk.schedule tabs[i].m = .ok (sch, tPad) ∧ ∀ r < 2,
        ∃ (stream : StreamRec) (extras : Array Flock.Claim) (zp : Flock.Zk.ZkProof),
          sess.streams.find? (fun x => x.name == s!"{spec.tables[i]!}/rep{r}") = some stream ∧
          tabs[i].extra sess.points = .ok extras ∧
          Flock.Zk.decode (shapeZK tabs[i] sch extras) proofs[2 * i + r]! = .ok zp ∧
          Claims.Flock.zkQueryRound tabs[i].m tabs[i].kLog sch.k0 (2 + extras.size) < stream.rounds.size ∧
          Claims.Flock.zkRepColsOf (Claims.Flock.zkProofShape tabs[i].m tabs[i].kLog (2 + extras.size)
              tabs[i].merkle.digestLen (if tabs[i].merkle.saltBytes = 0 then none else some tabs[i].merkle.saltBytes))
              stream.rounds proofs[2 * i + r]! =
            (Claims.Flock.zkRepCols (sch.levels.getD 0 default)
              (stream.rounds[Claims.Flock.zkQueryRound tabs[i].m tabs[i].kLog sch.k0 (2 + extras.size)]!).coins
              zp.proof.ligerito.initialProof.rows).map
                (fun p => (p.1, p.2.take (2 ^ (sch.levels.getD 0 default).lanes))) ∧
          merkleCheckWith (tabs[i].merkle.opsWith hs) tabs[i].merkle zp.proof.cap (sch.levels.getD 0 default).cap
            (sch.levels.getD 0 default).d
            (positions (sch.levels.getD 0 default)
              (stream.rounds[Claims.Flock.zkQueryRound tabs[i].m tabs[i].kLog sch.k0 (2 + extras.size)]!).coins)
            zp.proof.ligerito.initialProof.rows zp.proof.ligerito.initialProof.siblings
            zp.proof.ligerito.initialProof.salts (2 ^ (sch.levels.getD 0 default).lanes + Zk.EXTRA_LANES) = .ok () ∧
          zp.proof.cap.foldl (· ++ ·) ByteArray.empty = sess.rootB[i]! := by
  obtain ⟨sess, hsess, hsz, hreps⟩ := verifyWithM_reps hs spec tabs record proofs h
  refine ⟨sess, hsess, hsz, fun i hi => ?_⟩
  obtain ⟨sch, tPad, c, hsch, hr⟩ := hreps i hi
  refine ⟨sch, tPad, hsch, fun r hr2 => ?_⟩
  exact zkRepColsOf_verifyRepWith hs spec tabs[i] sch tPad sess i r _ c hsch (hr r hr2)

/-- **The same for the deployed `--zk` verifier** (`Zk.verify`) on tables whose Merkle schemes are the registry's
(`Zk.verify_eq_verifyWith`; every table `Stmt.setupTables` sets up, `setupTables_named`). -/
theorem zkQueryRound_verify (spec : SessionSpec) (tabs : Array Setup) (record : String) (proofs : Array ByteArray)
    (hN : ∀ st ∈ tabs, st.merkle.Named) (h : Flock.Zk.verify spec tabs record proofs = .ok ()) :
    ∃ sess : RunRecord, Record.decodeWith (Hashes.ofSha512 Sha512.hash : Hashes V) spec record = .ok sess ∧
      tabs.size = spec.tables.size ∧
      ∀ (i : ℕ) (hi : i < tabs.size), ∃ (sch : Flock.Schedule) (tPad : ℕ),
        Flock.Zk.schedule tabs[i].m = .ok (sch, tPad) ∧ ∀ r < 2,
        ∃ (stream : StreamRec) (extras : Array Flock.Claim) (zp : Flock.Zk.ZkProof),
          sess.streams.find? (fun x => x.name == s!"{spec.tables[i]!}/rep{r}") = some stream ∧
          tabs[i].extra sess.points = .ok extras ∧
          Flock.Zk.decode (shapeZK tabs[i] sch extras) proofs[2 * i + r]! = .ok zp ∧
          Claims.Flock.zkQueryRound tabs[i].m tabs[i].kLog sch.k0 (2 + extras.size) < stream.rounds.size ∧
          Claims.Flock.zkRepColsOf (Claims.Flock.zkProofShape tabs[i].m tabs[i].kLog (2 + extras.size)
              tabs[i].merkle.digestLen (if tabs[i].merkle.saltBytes = 0 then none else some tabs[i].merkle.saltBytes))
              stream.rounds proofs[2 * i + r]! =
            (Claims.Flock.zkRepCols (sch.levels.getD 0 default)
              (stream.rounds[Claims.Flock.zkQueryRound tabs[i].m tabs[i].kLog sch.k0 (2 + extras.size)]!).coins
              zp.proof.ligerito.initialProof.rows).map
                (fun p => (p.1, p.2.take (2 ^ (sch.levels.getD 0 default).lanes))) ∧
          merkleCheckWith (tabs[i].merkle.opsWith (Hashes.ofSha512 Sha512.hash : Hashes V)) tabs[i].merkle zp.proof.cap
            (sch.levels.getD 0 default).cap (sch.levels.getD 0 default).d
            (positions (sch.levels.getD 0 default)
              (stream.rounds[Claims.Flock.zkQueryRound tabs[i].m tabs[i].kLog sch.k0 (2 + extras.size)]!).coins)
            zp.proof.ligerito.initialProof.rows zp.proof.ligerito.initialProof.siblings
            zp.proof.ligerito.initialProof.salts (2 ^ (sch.levels.getD 0 default).lanes + Zk.EXTRA_LANES) = .ok () ∧
          zp.proof.cap.foldl (· ++ ·) ByteArray.empty = sess.rootB[i]! :=
  zkQueryRound_verifyWithM (Hashes.ofSha512 Sha512.hash : Hashes V) spec tabs record proofs
    (by rw [Flock.Zk.verify_eq_verifyWith spec tabs record proofs hN] at h; exact h)

end FlockSoundness.ZkQuery
