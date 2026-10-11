import Proofs.Flock.Soundness.Discharge.ZkExec.VerdictForms

/-!
# Level 0's lane rows are the executable's

`zk_lane_rows`: the two coordinate rows of the `K`-row the compiled masked level 0 hands the inner proof are the two
constraints `Zk.ligeritoConstraints` builds, at the record `Zk.ligerito` returns.

* `ligeritoConstraints_eq`: the two constraints, as the lane phase's claim folds (`laneLhs`).
* `zk_ligerito_rec`: `Zk.ligerito`'s record holds the batched value, level 0's out-of-domain coin and value, the lane
  rounds' coins and the pair's coins `ρ`.
* `foldZK_rel`, `ligeritoPadZC_row`: the model's lane rounds fold the claim as the executable does, so every run of
  `ligeritoPadZC` returns the row `laneLhs` is.
-/

namespace FlockSoundness.Discharge.ZkExec

open Flock Refine Exec Game Model DecodeZL Transcript

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

/-! ## The executable's lane rows -/

/-- A lane round's step on the claim: `u₀ + r·(claim + u₂) + r²·u₂` on the masked message `j`. -/
def laneStepE (lay : Zk.PadLayout) (lg : LigeritoProof) (ch : Array F256) (claim : Zk.Lin2) (j : ℕ) : Zk.Lin2 :=
  Zk.l2add (Zk.l2add (Zk.l2masked lg.messages[j]!.1 (lay.lgMsg j 0) (lay.lgMsg j 1) lay.len)
      (Zk.l2scale (Zk.l2add claim (Zk.l2masked lg.messages[j]!.2 (lay.lgMsg j 2) (lay.lgMsg j 3) lay.len)) ch[j]!))
    (Zk.l2scale (Zk.l2masked lg.messages[j]!.2 (lay.lgMsg j 2) (lay.lgMsg j 3) lay.len) (ch[j]! * ch[j]!))

/-- The lane phase's claim, `target + β·y₀` through the lane rounds, then `+ ρ₀·e₀ + ρ₁·e₁ + T'`. -/
def laneLhs (lay : Zk.PadLayout) (rec : Zk.L0Rec) (lg : LigeritoProof) (msgs : Zk.Level0Msgs) : Zk.Lin2 :=
  let claim0 := (Zk.Lin.konst rec.target lay.len).add
    ((Zk.Lin.masked lg.oodValues[0]! (lay.lgOod 0) lay.len).scale rec.ood[0]!.1)
  let claim := (List.range' 0 lay.k0).foldl (laneStepE lay lg rec.challenges) (claim0, Zk.Lin.konst 0 lay.len)
  Zk.l2add (Zk.l2add claim (Zk.l2scale (Zk.l2masked msgs.e0 (lay.lgE 0 0) (lay.lgE 0 1) lay.len) rec.rho.1))
    (Zk.l2add (Zk.l2scale (Zk.l2masked msgs.e1 (lay.lgE 1 0) (lay.lgE 1 1) lay.len) rec.rho.2)
      (Zk.l2konst msgs.tPrime lay.len))

theorem ligeritoConstraints_eq {lay : Zk.PadLayout} {rec : Zk.L0Rec} {lg : LigeritoProof} {msgs : Zk.Level0Msgs}
    {v : Array (String × Zk.Lin)} (h : Zk.ligeritoConstraints lay rec lg msgs = .ok v) (h1 : lay.nOod0 = 1) :
    rec.ood.size = 1 ∧ v[0]!.2 = (laneLhs lay rec lg msgs).1 ∧ v[1]!.2 = (laneLhs lay rec lg msgs).2 := by
  unfold Zk.ligeritoConstraints at h
  dsimp only at h
  split at h
  · cases h
  · rename_i hc
    simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, not_or, not_not, decide_eq_true_eq, not_lt] at hc
    obtain ⟨⟨ho, -⟩, -⟩ := hc
    rw [h1] at ho
    simp only [Std.Legacy.Range.forIn_eq_forIn_range', range_size, List.forIn_pure_yield_eq_foldl, ho] at h
    cases h
    refine ⟨ho, ?_, ?_⟩ <;> simp [laneLhs, List.range'_one] <;> rfl

/-! ## The executable's record -/

/-- A `K` coin as `sampleK` reads it. -/
def coinK (c : Array F128) : F256 := ⟨c.getD 0 0, c.getD 1 0⟩

theorem lforIn_inv {σ : Type} (body : ℕ → σ → LM (ForInStep σ)) (I : LState → Prop)
    (hb : ∀ i s st t b st' t', I st → body i s st t = .ok ((b, st'), t') → I st') :
    ∀ (l : List ℕ) (s : σ) (st : LState) (t : Flock.Transcript) (s' : σ) (st' : LState) (t' : Flock.Transcript),
      I st → (forIn l s body : LM σ) st t = .ok ((s', st'), t') → I st'
  | [], s, st, t, s', st', t', hI, h => by
      simp only [List.forIn_nil] at h
      obtain ⟨-, rfl, -⟩ := lpure_ok h
      exact hI
  | a :: l, s, st, t, s', st', t', hI, h => by
      rw [List.forIn_cons] at h
      obtain ⟨r, st₁, t₁, h₁, h₂⟩ := lbind_ok h
      have hI₁ := hb a s st t r st₁ t₁ hI h₁
      cases r with
      | done b => obtain ⟨-, rfl, -⟩ := lpure_ok h₂; exact hI₁
      | yield b => exact lforIn_inv body I hb l b st₁ t₁ s' st' t' hI₁ h₂

theorem chs_oods (p : LigeritoProof) :
    ∀ (l : List ℕ) (chs chs' : Array F256) (stE stE' : LState) (t t' : Flock.Transcript),
      forIn l chs (fun _ chs => challenge p >>= fun x => pure (ForInStep.yield (chs.push x))) stE t =
        .ok ((chs', stE'), t') → stE'.oods = stE.oods
  | [], chs, chs', stE, stE', t, t', h => by
      simp only [List.forIn_nil] at h
      obtain ⟨-, rfl, -⟩ := lpure_ok h
      rfl
  | a :: l, chs, chs', stE, stE', t, t', h => by
      rw [List.forIn_cons] at h
      obtain ⟨r, st₁, t₁, h₁, h₂⟩ := lbind_ok h
      obtain ⟨-, -, m, t₂, -, -, -, h₁⟩ := challenge_bind h₁
      obtain ⟨rfl, rfl, rfl⟩ := lpure_ok h₁
      simp only at h₂
      have e := chs_oods p l _ chs' _ stE' t₁ t' h₂
      exact e

theorem push_get_lt {α : Type} [Inhabited α] (a : Array α) (x : α) {j : ℕ} (hj : j < a.size) :
    (a.push x)[j]! = a[j]! := by
  rw [getElem!_pos (a.push x) j (by simp; omega), getElem!_pos a j hj, Array.getElem_push_lt hj]

theorem push_get_size {α : Type} [Inhabited α] (a : Array α) (x : α) : (a.push x)[a.size]! = x := by
  rw [getElem!_pos (a.push x) a.size (by simp), Array.getElem_push_eq]

/-- The lane rounds' record: one `K` coin pushed per round, the last round's two `ρ` coins after it. -/
theorem lane_rec (body : ℕ → Array F256 × F256 × F256 → LM (ForInStep (Array F256 × F256 × F256))) (K : ℕ)
    (hbody : ∀ i s st t b st' t', i < K → body i s st t = .ok ((b, st'), t') →
      b = .yield (s.1.push (coinK (t.rounds[t.k]!).coins),
        if i + 1 = K then (coinK (t.rounds[t.k + 1]!).coins, coinK (t.rounds[t.k + 2]!).coins) else s.2) ∧
      t'.rounds = t.rounds ∧ t'.k = t.k + 1 + (if i + 1 = K then 2 else 0) ∧ st'.oods = st.oods ∧
      (t.rounds[t.k]!).n = 2 ∧ (i + 1 = K → (t.rounds[t.k + 1]!).n = 2 ∧ (t.rounds[t.k + 2]!).n = 2)) :
    ∀ n a s st t s' st' t', a + n = K → 0 < n → forIn (List.range' a n) s body st t = .ok ((s', st'), t') →
      t'.rounds = t.rounds ∧ t'.k = t.k + n + 2 ∧ st'.oods = st.oods ∧ s'.1.size = s.1.size + n ∧
      (∀ j < s.1.size, s'.1[j]! = s.1[j]!) ∧ (∀ j < n, s'.1[s.1.size + j]! = coinK (t.rounds[t.k + j]!).coins) ∧
      s'.2 = (coinK (t.rounds[t.k + n]!).coins, coinK (t.rounds[t.k + n + 1]!).coins) ∧
      (∀ j < n, (t.rounds[t.k + j]!).n = 2) ∧ (t.rounds[t.k + n]!).n = 2 ∧ (t.rounds[t.k + n + 1]!).n = 2
  | 0, _, _, _, _, _, _, _, _, h0, _ => absurd h0 (lt_irrefl 0)
  | 1, a, s, st, t, s', st', t', hK, _, h => by
      rw [List.range'_succ, List.forIn_cons] at h
      obtain ⟨b, st₁, t₁, h₁, h₂⟩ := lbind_ok h
      obtain ⟨rfl, hr, hk, ho, hn0, hnρ⟩ := hbody a s st t b st₁ t₁ (by omega) h₁
      obtain ⟨hnρ₀, hnρ₁⟩ := hnρ (by omega)
      simp only [List.range'_zero, List.forIn_nil] at h₂
      obtain ⟨rfl, rfl, rfl⟩ := lpure_ok h₂
      rw [ite_eq_left (show a + 1 = K by omega)] at hk ⊢
      refine ⟨hr, by omega, ho, by simp, fun j hj => ?_, fun j hj => ?_, rfl, fun j hj => ?_, hnρ₀, hnρ₁⟩
      · exact push_get_lt _ _ hj
      · obtain rfl : j = 0 := by omega
        exact push_get_size _ _
      · obtain rfl : j = 0 := by omega
        exact hn0
  | n + 2, a, s, st, t, s', st', t', hK, _, h => by
      rw [List.range'_succ, List.forIn_cons] at h
      obtain ⟨b, st₁, t₁, h₁, h₂⟩ := lbind_ok h
      obtain ⟨rfl, hr, hk, ho, hn0, -⟩ := hbody a s st t b st₁ t₁ (by omega) h₁
      rw [ite_eq_right (show ¬ a + 1 = K by omega)] at hk
      simp only at h₂
      rw [ite_eq_right (show ¬ a + 1 = K by omega)] at h₂
      obtain ⟨hr', hk', ho', hsz, hpre, hnew, hρ, hnl, hnρ₀, hnρ₁⟩ :=
        lane_rec body K hbody (n + 1) (a + 1) _ st₁ t₁ s' st' t' (by omega) (by omega) h₂
      simp only [Array.size_push] at hsz hpre hnew
      refine ⟨hr'.trans hr, by omega, ho'.trans ho, by omega, fun j hj => ?_, fun j hj => ?_, ?_, fun j hj => ?_,
        ?_, ?_⟩
      · rw [hpre j (by omega)]
        exact push_get_lt _ _ hj
      · rcases j with _ | j
        · rw [hpre _ (by omega)]
          exact push_get_size _ _
        · rw [show s.1.size + (j + 1) = s.1.size + 1 + j by omega, hnew j (by omega), hr,
            show t₁.k + j = t.k + (j + 1) by omega]
      · rw [hρ, hr, show t₁.k + (n + 1) = t.k + (n + 2) by omega]
      · rcases j with _ | j
        · exact hn0
        · rw [← hr, show t.k + (j + 1) = t₁.k + j by omega]
          exact hnl j (by omega)
      · rw [← hr, show t.k + (n + 2) = t₁.k + (n + 1) by omega]
        exact hnρ₀
      · rw [← hr, show t.k + (n + 2) + 1 = t₁.k + (n + 1) + 1 by omega]
        exact hnρ₁

/-- `finalProgZ`'s record: the target, level 0's out-of-domain coin and value, the lane rounds' and pair's coins. -/
theorem finalProgZ_rec {p : LigeritoProof} {basisAt : Array F256 → ℕ → Array F256} {n : ℕ} {T : F128}
    {initCh : Array F256} {rho : F256 × F256} {levels : Array (Array F256)} {ext : ℕ} {st st' : LState}
    {tr tr' : Flock.Transcript} {rec : Zk.L0Rec}
    (h : finalProgZ p basisAt n T initCh rho levels ext st tr = .ok ((rec, st'), tr')) :
    rec = { target := T, ood := #[((st.oods[0]!).beta, p.oodValues[0]!)], challenges := initCh, rho } := by
  unfold finalProgZ at h
  have h1 := lget_bind h
  clear h
  dsimp only at h1
  change ((if ((origE initCh levels).size + ext != n) = true then _ else _ : LM Zk.L0Rec) st tr) = _ at h1
  by_cases c1 : ((origE initCh levels).size + ext != n) = true
  · rw [ite_eq_left c1] at h1; obtain ⟨_, _, _, h0, _⟩ := lbind_ok h1; cases h0
  rw [ite_eq_right c1] at h1
  change ((if ((basisAt (origE initCh levels) ext).size != 2 ^ ext) = true then _ else _ : LM Zk.L0Rec) st tr) = _
    at h1
  by_cases c2 : ((basisAt (origE initCh levels) ext).size != 2 ^ ext) = true
  · rw [ite_eq_left c2] at h1; obtain ⟨_, _, _, h0, _⟩ := lbind_ok h1; cases h0
  rw [ite_eq_right c2] at h1
  obtain ⟨res₁, st₁, tr₁, hl₁, h2⟩ := lbind_ok h1
  clear h1
  rw [← Array.forIn_toList] at hl₁
  obtain ⟨-, rfl, rfl, -⟩ := forIn_yield_fold _
    (fun res o => oodStep ext (Refine.oodFixed levels (origE initCh levels) o) o res)
    (fun o => o.z.size = (Refine.oodFixed levels (origE initCh levels) o).1.size + ext) (by
      intro o s st tr b st' tr' hb
      change ((if (o.z.size != (Refine.oodFixed levels (origE initCh levels) o).1.size + ext) = true then _ else _ :
        LM (ForInStep (Array F256))) st tr) = _ at hb
      by_cases cz : (o.z.size != (Refine.oodFixed levels (origE initCh levels) o).1.size + ext) = true
      · rw [ite_eq_left cz] at hb; obtain ⟨_, _, _, h0, _⟩ := lbind_ok hb; cases h0
      rw [ite_eq_right cz] at hb
      obtain ⟨rfl, rfl, rfl⟩ := lpure_ok hb
      simp only [bne_iff_ne, ne_eq, not_not] at cz
      exact ⟨rfl, rfl, rfl, cz⟩) _ _ _ _ _ _ _ hl₁
  obtain ⟨res₂, st₂, tr₂, hl₂, h3⟩ := lbind_ok h2
  clear h2
  rw [← Array.forIn_toList] at hl₂
  obtain ⟨-, rfl, rfl, -⟩ := forIn_yield_fold _ (consStep (svTable (n + 1)) ext levels)
    (fun cs => (Refine.consFixed levels cs.start).size + ext = cs.cols) (by
      intro cs s st tr b st' tr' hb
      change ((if ((Refine.consFixed levels cs.start).size + ext != cs.cols) = true then _ else _ :
        LM (ForInStep (Array F256))) st tr) = _ at hb
      by_cases cz : ((Refine.consFixed levels cs.start).size + ext != cs.cols) = true
      · rw [ite_eq_left cz] at hb; obtain ⟨_, _, _, h0, _⟩ := lbind_ok hb; cases h0
      rw [ite_eq_right cz] at hb
      obtain ⟨add, st₃, tr₃, hadd, hb'⟩ := lbind_ok hb
      clear hb
      rw [lforIn_range_eq, List.forIn_pure_yield_eq_foldl] at hadd
      obtain ⟨rfl, rfl, rfl⟩ := lpure_ok hadd
      obtain ⟨rfl, rfl, rfl⟩ := lpure_ok hb'
      simp only [bne_iff_ne, ne_eq, not_not] at cz
      exact ⟨rfl, rfl, rfl, cz⟩) _ _ _ _ _ _ _ hl₂
  change ((if (lhsE p.finalProof.yr ext res₂ != st₂.s) = true then _ else _ : LM Zk.L0Rec) st₂ tr₂) = _ at h3
  by_cases c3 : (lhsE p.finalProof.yr ext res₂ != st₂.s) = true
  · rw [ite_eq_left c3] at h3; obtain ⟨_, _, _, h0, _⟩ := lbind_ok h3; cases h0
  rw [ite_eq_right c3] at h3
  by_cases c4 : (st₂.ood != p.oodValues.size || st₂.qn != p.queryNonces.size ||
      st₂.cn != p.consistencyNonces.size || st₂.cbn != p.claimNonces.size) = true
  · rw [ite_eq_left c4] at h3; obtain ⟨_, _, _, h0, _⟩ := lbind_ok h3; cases h0
  rw [ite_eq_right c4] at h3
  obtain ⟨rfl, -, -⟩ := lpure_ok h3
  rfl

theorem push_get0 {a : Array OodCtx} {o x : OodCtx} (h : a[0]? = some o) : (a.push x)[0]? = some o := by
  have hs : 0 < a.size := by
    rcases Nat.eq_zero_or_pos a.size with h0 | h0
    · rw [Array.getElem?_eq_none (by omega)] at h; cases h
    · exact h0
  rw [Array.getElem?_push, ite_eq_right (by omega), h]

theorem lforIn_inv' {σ : Type} (body : ℕ → σ → LM (ForInStep σ)) (I : LState → Flock.Transcript → Prop) (B : ℕ)
    (hb : ∀ i s st t b st' t', i < B → I st t → body i s st t = .ok ((b, st'), t') → I st' t') :
    ∀ (n a : ℕ) (s : σ) (st : LState) (t : Flock.Transcript) (s' : σ) (st' : LState) (t' : Flock.Transcript),
      a + n ≤ B → I st t → (forIn (List.range' a n) s body : LM σ) st t = .ok ((s', st'), t') → I st' t'
  | 0, a, s, st, t, s', st', t', _, hI, h => by
      simp only [List.range'_zero, List.forIn_nil] at h
      obtain ⟨-, rfl, rfl⟩ := lpure_ok h
      exact hI
  | n + 1, a, s, st, t, s', st', t', hB, hI, h => by
      rw [List.range'_succ, List.forIn_cons] at h
      obtain ⟨r, st₁, t₁, h₁, h₂⟩ := lbind_ok h
      have hI₁ := hb a s st t r st₁ t₁ (by omega) hI h₁
      cases r with
      | done b => obtain ⟨-, rfl, rfl⟩ := lpure_ok h₂; exact hI₁
      | yield b => exact lforIn_inv' body I B hb n (a + 1) b st₁ t₁ s' st' t' (by omega) hI₁ h₂

set_option maxHeartbeats 4000000 in
/-- **`Zk.ligerito`'s record**: the target, level 0's out-of-domain coin, the lane rounds' coins and the pair's coins,
at the rounds the run consumed, and those rounds' coin counts. -/
theorem zk_ligerito_rec (ms : MerkleScheme) (sch : Flock.Schedule) (T : F128) (cap : Array ByteArray)
    (p : LigeritoProof) (basisAt : Array F256 → ℕ → Array F256) (zl : Zk.Level0) (t t' : Flock.Transcript)
    (rec : Zk.L0Rec) (hdim : DimSeam sch) (hlv : ∀ i < sch.levels.size, LevelSeam (sch.levels.getD i default))
    (hk0 : 1 ≤ sch.k0) (h : Zk.ligerito ms sch T cap p basisAt zl t = .ok (rec, t')) :
    rec.target = T ∧ rec.ood.size = 1 ∧ rec.ood[0]!.1 = (t.rounds[t.k + 1]!).coins.getD 0 0 ∧
      rec.challenges.size = sch.k0 ∧
      (∀ j < sch.k0, rec.challenges[j]! = coinK (t.rounds[t.k + 2 + j]!).coins) ∧
      rec.rho = (coinK (t.rounds[t.k + 2 + sch.k0]!).coins, coinK (t.rounds[t.k + 3 + sch.k0]!).coins) ∧
      t'.rounds = t.rounds ∧ t.k + 4 + sch.k0 ≤ t'.k ∧
      (t.rounds[t.k]!).n = sch.n ∧ (t.rounds[t.k + 1]!).n = 1 ∧ (∀ j < sch.k0, (t.rounds[t.k + 2 + j]!).n = 2) ∧
      (t.rounds[t.k + 2 + sch.k0]!).n = 2 ∧ (t.rounds[t.k + 3 + sch.k0]!).n = 2 := by
  obtain ⟨hn7, h2, hl0, hc0, hl4, hcols, hext⟩ := hdim
  have hc01 := hcols 0 (by omega)
  have hn3 : 3 ≤ sch.n := by omega
  unfold Zk.ligerito at h
  obtain ⟨hcap, h⟩ := check_bind h
  obtain ⟨hzl, h⟩ := check_bind h
  obtain ⟨t₁, k₁, h⟩ := label_bind h
  obtain ⟨t₂, k₂, h⟩ := absorbF_bind h
  obtain ⟨t₃, k₃, h⟩ := absorbBytes_bind h
  obtain ⟨t₃', k₃', h⟩ := label_bind h
  obtain ⟨⟨rc', stF⟩, t₄, h₄, h⟩ := bind_ok h
  obtain ⟨rfl, rfl⟩ := pure_ok h
  rw [StateT.run] at h₄
  obtain ⟨z, t₅, h₅, h₄⟩ := lift_bind h₄
  obtain ⟨hn₅, rfl, hr₅, hk₅⟩ := sampleVec_ok (by omega) h₅
  obtain ⟨y₀, -, h₄⟩ := nextOod_bind h₄
  obtain ⟨_, t₆, h₆, h₄⟩ := lift_bind h₄
  have k₆ := absorbF_ok h₆
  obtain ⟨t₇, k₇, h₄⟩ := claimPow_bind h₄
  obtain ⟨bc, t₈, h₈, h₄⟩ := lift_bind h₄
  obtain ⟨-, hn₈, rfl, hr₈, hk₈⟩ := sample_ok h₈
  have h₄ := lmodify_bind h₄
  obtain ⟨m₀, t₉, -, k₉, h₄⟩ := nextMsg_bind h₄
  have h₄ := lmodify_bind h₄
  obtain ⟨lane, st₁₀, t₁₀, h₁₀, h₄⟩ := lbind_ok h₄
  rw [lforIn_range_eq] at h₁₀
  obtain ⟨hr₁₀, hk₁₀, ho₁₀, hsz₁₀, -, hch₁₀, hρ₁₀, hnl₁₀, hnρ₀, hnρ₁⟩ := lane_rec _ sch.k0 (by
      intro i v st tE bv st' tE' hiB hb
      obtain ⟨r, tR, hR', hb⟩ := lift_bind hb
      obtain ⟨-, hnR, rfl, hrR, hkR⟩ := sampleK_ok hR'
      have hb := lmodify_bind hb
      try dsimp only at hb
      by_cases hj : i + 1 = sch.k0
      · rw [ite_eq_left (show (i + 1 == sch.k0) = true by simp [hj])] at hb
        obtain ⟨_, t₂, h₂, hb⟩ := lift_bind hb
        obtain ⟨f₂k, f₂r⟩ := absorbK_ok h₂
        obtain ⟨_, t₃, h₃, hb⟩ := lift_bind hb
        obtain ⟨f₃k, f₃r⟩ := absorbK_ok h₃
        obtain ⟨ρ₀, t₄, h₄, hb⟩ := lift_bind hb
        obtain ⟨-, hn₄, eρ₀, hr₄, hk₄⟩ := sampleK_ok h₄
        obtain ⟨ρ₁, t₅, h₅, hb⟩ := lift_bind hb
        obtain ⟨-, hn₅, eρ₁, hr₅, hk₅⟩ := sampleK_ok h₅
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
          rfl
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
        rw [ite_eq_right hj]
        rfl)
    sch.k0 0 _ _ t₉ _ _ t₁₀ (by omega) (by omega) h₁₀
  have hI₁₀ : st₁₀.oods[0]? = some ⟨(t₃'.rounds[t₃'.k]!).coins, (t₇.rounds[t₇.k]!).coins.getD 0 0, none⟩ := by
    rw [ho₁₀]; rfl
  try dsimp only at h₄
  obtain ⟨_, t₁₁, h₁₁, h₄⟩ := lift_bind h₄
  have k₁₁ := absorbBytes_ok h₁₁
  obtain ⟨-, -, yA, mA, t₁₂, -, -, hr₁₂, hk₁₂, h₄⟩ :=
    oodIntro_bind (show sch.n - sch.k0 + 1 ≠ 0 by omega) h₄
  obtain ⟨-, -, yB, mB, t₁₃, -, -, hr₁₃, hk₁₃, h₄⟩ :=
    oodIntro_bind (show sch.n - sch.k0 + 1 ≠ 0 by omega) h₄
  have hlv0 := hlv 0 (by omega)
  obtain ⟨-, -, ⟨pos0, al0, e0⟩, t₁₄, hr₁₄, hk₁₄, h₄⟩ := queryLevel0_bind hlv0.q_ne_zero h₄
  dsimp only at h₄
  obtain ⟨extra, stX, tX, hX, h₄'⟩ := lbind_ok h₄
  clear h₄
  rw [lforIn_range_eq, List.forIn_pure_yield_eq_foldl] at hX
  obtain ⟨rfl, rfl, rfl⟩ := lpure_ok hX
  obtain ⟨mq, t₁₅, -, k₁₅, h₄⟩ := nextMsg_bind h₄'
  obtain ⟨t₁₆, k₁₆, h₄⟩ := claimPow_bind h₄
  obtain ⟨bq, t₁₇, h₁₇, h₄⟩ := lift_bind h₄
  obtain ⟨-, -, -, hr₁₇, hk₁₇⟩ := sample_ok h₁₇
  have h₄ := lmodify_bind h₄
  obtain ⟨out, st₁₈, t₁₈, hloop, hcont⟩ := lbind_ok h₄
  rw [lforIn_range_eq] at hloop
  obtain ⟨hI₁₈, hr₁₈, hk₁₈⟩ := lforIn_inv' _ (fun st tE => st.oods[0]? =
      some ⟨(t₃'.rounds[t₃'.k]!).coins, (t₇.rounds[t₇.k]!).coins.getD 0 0, none⟩ ∧ tE.rounds = t₁₇.rounds ∧
      t₁₇.k ≤ tE.k) (sch.levels.size - 1) (by
      intro i v stE tE bv stE' tE' hi hI hb
      obtain ⟨hI, hrE, hkE⟩ := hI
      try dsimp only at hb
      by_cases c4 : v.1 < 4
      · rw [ite_eq_left c4] at hb
        cases hb
      rw [ite_eq_right c4] at hb
      obtain ⟨chs, st₁, t₁, hf, hb⟩ := lbind_ok hb
      rw [lforIn_range_eq] at hf
      have ho₁ := chs_oods p _ _ _ _ _ _ _ hf
      obtain ⟨hr₁, hk₁, -⟩ := fold_coins p 4 0 #[] chs stE st₁ tE t₁ hf
      by_cases hi' : i + 1 < sch.levels.size - 1
      · rw [ite_eq_left hi'] at hb
        obtain ⟨_, t₂, h₂, hb⟩ := lift_bind hb
        have k₂ := absorbBytes_ok h₂
        obtain ⟨-, -, yA, mA, t₃, -, -, hr₃, hk₃, hb⟩ := oodIntro_bind (show v.1 - 4 + 1 ≠ 0 by omega) hb
        obtain ⟨-, -, yB, mB, t₄, -, -, hr₄, hk₄, hb⟩ := oodIntro_bind (show v.1 - 4 + 1 ≠ 0 by omega) hb
        obtain ⟨-, -, -, t₅, hr₅, hk₅, hb⟩ := queryLevel_bind (hlv (i + 1) (by omega)).q_ne_zero hb
        obtain ⟨mq, t₆, -, k₆, hb⟩ := nextMsg_bind hb
        obtain ⟨t₇, k₇, hb⟩ := claimPow_bind hb
        obtain ⟨bq, t₈, h₈, hb⟩ := lift_bind hb
        obtain ⟨-, -, -, hr₈, hk₈⟩ := sample_ok h₈
        have hb := lmodify_bind hb
        by_cases c5 : v.1 - 4 + 1 < 4
        · rw [ite_eq_left c5] at hb
          cases hb
        rw [ite_eq_right c5] at hb
        obtain ⟨-, rfl, rfl⟩ := lpure_ok hb
        refine ⟨push_get0 (push_get0 (ho₁ ▸ hI)), ?_, ?_⟩
        · rw [hr₈, k₇.2, k₆.2, hr₅, hr₄, hr₃, k₂.2, hr₁, hrE]
        · rw [hk₈, k₇.1, k₆.1, hk₅, hk₄, hk₃, k₂.1, hk₁]; omega
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
        obtain ⟨rfl, k₂⟩ := forIn_absorbF_keeps _ h₂
        obtain ⟨-, -, -, t₃, hr₃, hk₃, hb⟩ :=
          queryLevel_bind (hlv (sch.levels.size - 1) (by omega)).q_ne_zero hb₂
        obtain ⟨t₄, k₄, hb⟩ := claimPow_bind hb
        obtain ⟨bq, t₅, h₅, hb⟩ := lift_bind hb
        obtain ⟨-, -, -, hr₅, hk₅⟩ := sample_ok h₅
        have hb := lmodify_bind hb
        obtain ⟨-, rfl, rfl⟩ := lpure_ok hb
        refine ⟨ho₁ ▸ hI, ?_, ?_⟩
        · rw [hr₅, k₄.2, hr₃, k₂.2, hr₁, hrE]
        · rw [hk₅, k₄.1, hk₃, k₂.1, hk₁]; omega)
    (sch.levels.size - 1) 0 _ _ t₁₇ _ _ t₁₈ (by omega) ⟨push_get0 (push_get0 hI₁₀), rfl, le_rfl⟩ hloop
  have hfin := finalProgZ_ok hcont
  have hrec := finalProgZ_rec hcont
  subst hrec
  have er₇ : t₇.rounds = t.rounds := by rw [k₇.2, k₆.2, hr₅, k₃'.2, k₃.2, k₂.2, k₁.2]
  have ek₇ : t₇.k = t.k + 1 := by rw [k₇.1, k₆.1, hk₅, k₃'.1, k₃.1, k₂.1, k₁.1]
  have er₉ : t₉.rounds = t.rounds := by rw [k₉.2, hr₈, er₇]
  have ek₉ : t₉.k = t.k + 2 := by rw [k₉.1, hk₈, ek₇]
  have er₁₇ : t₁₇.rounds = t.rounds := by
    rw [hr₁₇, k₁₆.2, k₁₅.2, hr₁₄, hr₁₃, hr₁₂, k₁₁.2, hr₁₀, er₉]
  have ek₁₇ : t₁₇.k = t.k + 2 + sch.k0 + 9 := by
    rw [hk₁₇, k₁₆.1, k₁₅.1, hk₁₄, hk₁₃, hk₁₂, k₁₁.1, hk₁₀, ek₉]
  refine ⟨rfl, rfl, ?_, by simpa using hsz₁₀, fun j hj => ?_, ?_, by rw [hfin, hr₁₈, er₁₇], by
    rw [hfin]; omega, ?_, ?_, fun j hj => ?_, ?_, ?_⟩
  · show (st₁₈.oods[0]!).beta = _
    rw [getElem!_of_getElem? hI₁₈, er₇, ek₇]
  · have := hch₁₀ j hj
    simp only [Array.size_empty, Nat.zero_add] at this
    rw [this, er₉, ek₉]
  · rw [hρ₁₀, er₉, ek₉, show t.k + 2 + sch.k0 + 1 = t.k + 3 + sch.k0 by omega]
  · rw [show t.rounds = t₃'.rounds by rw [k₃'.2, k₃.2, k₂.2, k₁.2], show t.k = t₃'.k by rw [k₃'.1, k₃.1, k₂.1, k₁.1]]
    exact hn₅
  · rw [← er₇, ← ek₇]
    exact hn₈
  · rw [← er₉, ← ek₉]
    exact hnl₁₀ j hj
  · rw [← er₉, ← ek₉]
    exact hnρ₀
  · rw [← er₉, show t.k + 3 + sch.k0 = t₉.k + sch.k0 + 1 by omega]
    exact hnρ₁

/-! ## The model's lane rounds -/

theorem coinK_limbs (c : Array F128) : execArith.ofLimbs (fOf c[0]!) (fOf c[1]!) = gOf (coinK c) :=
  ofLimbs_coin c

/-- One lane round on forms is `laneStepE`. -/
theorem laneStep_rel {e : ℕ} (lay : Zk.PadLayout) (P : ZkSession.Pads0 lay.len e) (lg : LigeritoProof)
    (ch : Array F256) (j : ℕ) (hP : ∀ t : Fin 4, (P.msg j t).val = lay.lgMsg j t.val)
    {s : ZkSession.LinK 𝕂 lay.len} {x : Zk.Lin2} (hs : Rel2 s x) (c : Array F128) (hc : coinK c = ch[j]!) :
    Rel2 ((ZkSession.fromMsgL (ZkSession.unmask execArith P j (gOf lg.messages[j]!.1, gOf lg.messages[j]!.2))
        s).eval (execArith.ofLimbs (fOf c[0]!) (fOf c[1]!))) (laneStepE lay lg ch x j) := by
  have h1 := rel2_masked lg.messages[j]!.1 (P.msg j 0) (P.msg j 1) (hP 0) (hP 1)
    (by simp [Zk.PadLayout.lgMsg])
  have h2 := rel2_masked lg.messages[j]!.2 (P.msg j 2) (P.msg j 3) (hP 2) (hP 3)
    (by simp [Zk.PadLayout.lgMsg])
  have h := rel2_add (rel2_add h1 (rel2_scale (rel2_add hs h2) ch[j]!)) (rel2_scale h2 (ch[j]! * ch[j]!))
  rw [coinK_limbs, hc]
  convert h using 1
  · simp only [ZkSession.QuadL.eval, ZkSession.fromMsgL, ZkSession.unmask, gOf_mul, pow_two]
  · rfl

/-- **The model's lane rounds fold the claim as the executable does.** -/
theorem foldZK_rel {e : ℕ} (lay : Zk.PadLayout) (P : ZkSession.Pads0 lay.len e) (lg : LigeritoProof)
    (ch : Array F256) (K : ℕ) (hP : ∀ j < K, ∀ t : Fin 4, (P.msg j t).val = lay.lgMsg j t.val) :
    ∀ (k j : ℕ) (s : ZkSession.LinK 𝕂 lay.len) (x : Zk.Lin2) (cs : List (Array F128)), j + k ≤ K → 0 < k →
      Rel2 s x → cs.length = k → (∀ i < k, (cs[i]!).size = 2 ∧ coinK cs[i]! = ch[j + i]!) →
      ∃ out, Plays fOf (ZkSession.foldRoundsZK execArith P k j s
          (ZkSession.fromMsgL (ZkSession.unmask execArith P j (gOf lg.messages[j]!.1, gOf lg.messages[j]!.2)) s))
        (kMsgs lg (j + 1) (k - 1)) cs out ∧ Rel2 out.2 ((List.range' j k).foldl (laneStepE lay lg ch) x)
  | 0, _, _, _, _, _, h0, _, _, _ => absurd h0 (lt_irrefl 0)
  | 1, j, s, x, cs, hK, _, hs, hl, hc => by
      obtain ⟨c, rfl⟩ : ∃ c, cs = [c] := by
        match cs, hl with
        | [c], _ => exact ⟨c, rfl⟩
      obtain ⟨hc2, hck⟩ := hc 0 (by omega)
      simp only [List.getElem!_cons_zero, Nat.add_zero] at hc2 hck
      rw [ZkSession.foldRoundsZK, show kMsgs lg (j + 1) (1 - 1) = [] by simp [kMsgs]]
      have hk : Plays fOf (Model.drawK execArith) [] [c] (execArith.ofLimbs (fOf c[0]!) (fOf c[1]!)) := by
        unfold Model.drawK
        exact Plays.bind' (ms₂ := []) (cs₂ := []) (Plays.drawVec c hc2 (by omega)) (Plays.pure _) (by simp) (by simp)
      refine ⟨_, Plays.bind' (ms₂ := []) (cs₂ := []) hk (Plays.pure _) rfl rfl, ?_⟩
      · simp only [List.range'_one, List.foldl_cons, List.foldl_nil]
        exact laneStep_rel lay P lg ch j (hP j (by omega)) hs c hck
  | k + 2, j, s, x, cs, hK, _, hs, hl, hc => by
      obtain ⟨c, cs, rfl⟩ : ∃ c cs', cs = c :: cs' := by
        match cs, hl with
        | c :: cs', _ => exact ⟨c, cs', rfl⟩
      obtain ⟨hc2, hck⟩ := hc 0 (by omega)
      simp only [List.getElem!_cons_zero, Nat.add_zero] at hc2 hck
      have hstep := laneStep_rel lay P lg ch j (hP j (by omega)) hs c hck
      obtain ⟨out, hpl, hrel⟩ := foldZK_rel lay P lg ch K hP (k + 1) (j + 1) _ _ cs (by omega) (by omega) hstep
        (by simpa using hl) (fun i hi => by
          have := hc (i + 1) (by omega)
          simpa [show j + (i + 1) = j + 1 + i by omega] using this)
      rw [ZkSession.foldRoundsZK, show kMsgs lg (j + 1) (k + 2 - 1) =
          kMsg lg.messages[j + 1]! :: kMsgs lg (j + 1 + 1) (k + 1 - 1) by simp [kMsgs, List.range'_succ]]
      have hk : Plays fOf (Model.drawK execArith) [] [c] (execArith.ofLimbs (fOf c[0]!) (fOf c[1]!)) := by
        unfold Model.drawK
        exact Plays.bind' (ms₂ := []) (cs₂ := []) (Plays.drawVec c hc2 (by omega)) (Plays.pure _) (by simp) (by simp)
      refine ⟨_, Plays.bind' hk (Plays.bind' (Plays.recv _ _) (Plays.bind' hpl (Plays.pure _) (List.append_nil _).symm
        (List.append_nil _).symm) rfl rfl) rfl rfl, ?_⟩
      rw [List.range'_succ, List.foldl_cons]
      exact hrel

theorem plays_drawK (c : Array F128) (hc : c.size = 2) :
    Plays fOf (Model.drawK execArith) [] [c] (execArith.ofLimbs (fOf c[0]!) (fOf c[1]!)) := by
  unfold Model.drawK
  exact Plays.bind' (ms₂ := []) (cs₂ := []) (Plays.drawVec c hc (by omega)) (Plays.pure _) (by simp) (by simp)

theorem rel2_neg {np : ℕ} {a : ZkSession.LinK 𝕂 np} {x : Zk.Lin2} (ha : Rel2 a x) : Rel2 (-a) x := by
  have e : -a = a := Prod.ext (funext fun i => CharTwo.neg_eq (a.1 i)) (CharTwo.neg_eq a.2)
  rw [e]; exact ha

/-- Level 0's out-of-domain claim `T + β·(y + h_ood)` on forms. -/
theorem claim0_rel (lay : Zk.PadLayout) (i : Fin lay.len) (hi : i.val = lay.lgOod 0) (T y β : F128) :
    Rel2 (ZkSession.cK (FlockLevel3.embed (fOf T)) + FlockLevel3.embed (fOf β) •
        (ZkSession.cK (FlockLevel3.embed (fOf y)) + ZkSession.padF i))
      ((Zk.Lin.konst T lay.len).add ((Zk.Lin.masked y (lay.lgOod 0) lay.len).scale β), Zk.Lin.konst 0 lay.len) := by
  have h := rel2_lift (rel_add (rel_konst (np := lay.len) T) (rel_scale (rel_masked y i hi) β))
  convert h using 1
  refine Prod.ext (funext fun j => ?_) ?_
  · simp only [liftK, ZkSession.cK, ZkSession.padF, ZkSession.cF, ZkSession.mskF, ZkSession.pF, Prod.fst_add,
      Prod.smul_fst, Pi.add_apply, Pi.smul_apply, Pi.zero_apply, smul_eq_mul, zero_add, map_mul]
    by_cases hj : j = i
    · subst hj; simp
    · simp [Pi.single_eq_of_ne hj]
  · simp only [liftK, ZkSession.cK, ZkSession.padF, ZkSession.cF, ZkSession.mskF, ZkSession.pF, Prod.snd_add,
      Prod.smul_snd, smul_eq_mul, add_zero, map_add, map_mul]

set_option maxHeartbeats 1000000 in
/-- **Every run of the compiled masked level 0 returns the lane phase's row** (`laneLhs`), at the record whose
target, out-of-domain coin, lane coins and pair coins the run's coins are. -/
theorem ligeritoPadZC_row (lay : Zk.PadLayout) (P : ZkSession.Pads0 lay.len 2) (S : Accounting.Schedule)
    (sch : Flock.Schedule) (tp : ℕ) (c₀ : 𝔽) (r : ℕ) (Tm : 𝔽) (b : ℕ → 𝔽) (lg : LigeritoProof)
    (zl : Zk.Level0Msgs) (rec : Zk.L0Rec) (lv₀ : Accounting.Level) (rest : List Accounting.Level)
    (hlv : S.levels = lv₀ :: rest) (hk : S.k0 = lay.k0) (hk' : sch.k0 = lay.k0) (hk1 : 1 ≤ lay.k0)
    (hood : P.ood.val = lay.lgOod 0) (hmsg : ∀ j < lay.k0, ∀ t : Fin 4, (P.msg j t).val = lay.lgMsg j t.val)
    (hev : ∀ (i : Fin 2) (c : Fin 2), (P.ev i c).val = lay.lgE i.val c.val) (hT : Tm = fOf rec.target)
    (zc bc c₁ c₂ : Array F128) (lanes rc : List (Array F128)) (hzc : zc.size = S.m - 7) (hm : S.m - 7 ≠ 1)
    (hbc : bc.size = 1) (hβ : rec.ood[0]!.1 = bc.getD 0 0) (hl : lanes.length = lay.k0)
    (hlanes : ∀ i < lay.k0, (lanes[i]!).size = 2 ∧ coinK lanes[i]! = rec.challenges[i]!)
    (hc₁ : c₁.size = 2) (hc₂ : c₂.size = 2) (hρ : rec.rho = (coinK c₁, coinK c₂))
    {out : ZkSession.LinK 𝕂 lay.len × PendL 𝔽 Dig}
    (h : Plays fOf (ZkLink.ligeritoPadZC (D := Dig) execArith P S tp c₀ r Tm b) (ligMsgsZ lg capDig sch tp zl)
      (zc :: bc :: (lanes ++ (c₁ :: c₂ :: rc))) out) :
    Rel2 out.1 (laneLhs lay rec lg zl) := by
  obtain ⟨R, hR⟩ : ∃ R, ligMsgsZ lg capDig sch tp zl = fMsg lg.oodValues[0]! :: kMsg lg.messages[0]! ::
      (kMsgs lg 1 (lay.k0 - 1) ++ (evMsg zl :: tMsg zl :: ybarMsg tp zl :: R)) :=
    ⟨_, by simp only [ligMsgsZ, hk', List.append_assoc, List.cons_append, List.nil_append]; rfl⟩
  rw [hR] at h
  unfold ZkLink.ligeritoPadZC at h
  rw [hlv] at h
  dsimp only at h
  have h := plays_bind_split (ms₁ := []) (cs₁ := [zc]) h (Plays.drawVec zc hzc hm)
  have h := plays_bind_split (ms₁ := [fMsg lg.oodValues[0]!]) (cs₁ := []) h (Plays.recv _ _)
  have h := plays_bind_split (ms₁ := []) (cs₁ := [bc]) h (Plays.drawOne bc hbc)
  have h := plays_bind_split (ms₁ := [kMsg lg.messages[0]!]) (cs₁ := []) h (Plays.recv _ _)
  obtain ⟨⟨ini, sF⟩, hfold, hrel⟩ := foldZK_rel lay P lg rec.challenges lay.k0 hmsg lay.k0 0
    (ZkSession.cK (FlockLevel3.embed Tm) + FlockLevel3.embed (fOf bc[0]!) •
      (ZkSession.cK (FlockLevel3.embed (fOf lg.oodValues[0]!)) + ZkSession.padF P.ood))
    ((Zk.Lin.konst rec.target lay.len).add ((Zk.Lin.masked lg.oodValues[0]! (lay.lgOod 0) lay.len).scale rec.ood[0]!.1),
      Zk.Lin.konst 0 lay.len) lanes (by omega) (by omega)
    (by rw [hT, getElem!_eq_getD, ← hβ]; exact claim0_rel lay P.ood hood rec.target lg.oodValues[0]! rec.ood[0]!.1)
    hl (fun i hi => by simpa using hlanes i hi)
  rw [hk] at h
  have h := plays_bind_split (ms₁ := kMsgs lg (0 + 1) (lay.k0 - 1)) (cs₁ := lanes) h hfold
  dsimp only at h
  have h := plays_bind_split (ms₁ := [evMsg zl]) (cs₁ := []) h (Plays.recv _ _)
  have hρs : Plays fOf (Model.drawKs execArith 2) [] [c₁, c₂]
      (Fin.cons (execArith.ofLimbs (fOf c₁[0]!) (fOf c₁[1]!))
        (Fin.cons (execArith.ofLimbs (fOf c₂[0]!) (fOf c₂[1]!)) Fin.elim0)) := by
    simp only [Model.drawKs]
    exact Plays.bind' (plays_drawK c₁ hc₁) (Plays.bind' (Plays.bind' (plays_drawK c₂ hc₂)
      (Plays.bind' (Plays.pure _) (Plays.pure _) rfl rfl) rfl rfl) (Plays.pure _) rfl rfl) rfl rfl
  have h := plays_bind_split (ms₁ := []) (cs₁ := [c₁, c₂]) h hρs
  have h := plays_bind_split (ms₁ := [tMsg zl]) (cs₁ := []) h (Plays.recv _ _)
  have h := plays_bind_split (ms₁ := [ybarMsg tp zl]) (cs₁ := []) h (Plays.recv _ _)
  obtain ⟨ms₁, ms₂, cs₁, cs₂, pend, hms, hcs, htail, hret⟩ := plays_bind_inv _ h
  obtain ⟨-, -, rfl⟩ := plays_ret_inv hret
  clear htail hret hms hcs h hR
  have hE0 := rel2_masked zl.e0 (P.ev 0 0) (P.ev 0 1) (hev 0 0) (hev 0 1) (by simp [Zk.PadLayout.lgE])
  have hE1 := rel2_masked zl.e1 (P.ev 1 0) (P.ev 1 1) (hev 1 0) (hev 1 1) (by simp [Zk.PadLayout.lgE])
  have hfin := rel2_add (rel2_add hrel (rel2_scale hE0 rec.rho.1))
    (rel2_add (rel2_scale hE1 rec.rho.2) (rel2_neg (rel2_konst (np := lay.len) zl.tPrime)))
  convert hfin using 1
  · rw [Fin.sum_univ_two, hρ, coinK_limbs, coinK_limbs]
    simp only [Fin.cons_zero, Fin.cons_one, Matrix.cons_val_zero, Matrix.cons_val_one]
    rw [sub_eq_add_neg]
    abel
  · rfl

end FlockSoundness.Discharge.ZkExec
