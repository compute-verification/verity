import Proofs.Flock.Soundness.Refine.Final
import Flock.With.Verify

/-!
# The rounds the shipped verifier reads

`verifyWith h` replays each rep's recorded rounds through its transcript, and refuses a rep whose rounds it did not all
consume (S15). Its squeezes are the oracle's (`squeezeWith`), but which round each one reads does not depend on the
hash: a squeeze of `n ≠ 0` coins reads round `t.k` and moves to `t.k + 1`, whatever `hs` is.
-/

namespace FlockDeploy

open Flock Transcript FlockSoundness.Refine

variable {α β : Type}

/-! ## The oracle's squeezes, at `V` -/

theorem liftTM_V (x : TM α) : liftTM (M := V) x = x := rfl

theorem liftLM_V (x : LM α) : liftLM (M := V) x = x := rfl

theorem squeezeWith_ok (hs : Hashes V) {kind : UInt8} {n : ℕ} (hn : n ≠ 0) {t t₁ : Transcript} {cs : Array F128}
    (h : squeezeWith hs kind n t = .ok (cs, t₁)) :
    t.k < t.rounds.size ∧ (t.rounds[t.k]!).n = n ∧ cs = (t.rounds[t.k]!).coins ∧ t₁.rounds = t.rounds ∧
      t₁.k = t.k + 1 := by
  unfold squeezeWith at h
  simp only [hn, ↓reduceIte] at h
  cases hr : t.rounds[t.k]? with
  | none => rw [hr] at h; cases h
  | some r =>
    rw [hr] at h
    simp only at h
    split at h
    · cases h
    · rename_i hrn
      cases ha : r.answersWith hs (framed t kind n) with
      | error e => rw [ha] at h; cases h
      | ok b =>
        rw [ha] at h
        cases b
        · cases h
        · cases h
          have hlt : t.k < t.rounds.size := by
            by_contra hge; rw [Array.getElem?_eq_none (by omega)] at hr; cases hr
          have hget : t.rounds[t.k]! = r := by
            rw [getElem!_pos t.rounds t.k hlt]
            exact Option.some.inj ((Array.getElem?_eq_getElem hlt).symm.trans hr)
          simp only [not_not] at hrn
          exact ⟨hlt, hget ▸ hrn, hget ▸ rfl, rfl, rfl⟩

/-- `sampleVecWith n` then `f`: round `t.k` issued `n` coins, and `f` continues with them from round `t.k + 1`. -/
theorem sampleVecWith_bind (hs : Hashes V) {n : ℕ} (hn : n ≠ 0) {f : Array F128 → TM β} {t t' : Transcript}
    {b : β} (h : (sampleVecWith hs n >>= f) t = .ok (b, t')) :
    t.k < t.rounds.size ∧ (t.rounds[t.k]!).n = n ∧
      ∃ t₁, t₁.rounds = t.rounds ∧ t₁.k = t.k + 1 ∧ f (t.rounds[t.k]!).coins t₁ = .ok (b, t') := by
  obtain ⟨cs, t₁, h₁, h₂⟩ := bind_ok h
  obtain ⟨hlt, hn', hcs, hr, hk⟩ := squeezeWith_ok hs hn h₁
  exact ⟨hlt, hn', t₁, hr, hk, hcs ▸ h₂⟩

theorem sampleVecWith_ok (hs : Hashes V) {n : ℕ} (hn : n ≠ 0) {t t₁ : Transcript} {cs : Array F128}
    (h : sampleVecWith hs n t = .ok (cs, t₁)) :
    (t.rounds[t.k]!).n = n ∧ cs = (t.rounds[t.k]!).coins ∧ t₁.rounds = t.rounds ∧ t₁.k = t.k + 1 := by
  obtain ⟨-, hn', hcs, hr, hk⟩ := squeezeWith_ok hs hn h
  exact ⟨hn', hcs, hr, hk⟩

/-- `sampleWith` then `f`: round `t.k` issued one coin. -/
theorem sampleWith_bind (hs : Hashes V) {f : F128 → TM β} {t t' : Transcript} {b : β}
    (h : (sampleWith hs >>= f) t = .ok (b, t')) :
    t.k < t.rounds.size ∧ (t.rounds[t.k]!).n = 1 ∧
      ∃ t₁, t₁.rounds = t.rounds ∧ t₁.k = t.k + 1 ∧ f ((t.rounds[t.k]!).coins.getD 0 0) t₁ = .ok (b, t') := by
  obtain ⟨c, t₁, h₁, h₂⟩ := bind_ok h
  obtain ⟨cs, t₂, h₃, h₄⟩ := bind_ok h₁
  obtain ⟨hlt, hn', hcs, hr, hk⟩ := squeezeWith_ok hs Nat.one_ne_zero h₃
  obtain ⟨rfl, rfl⟩ := pure_ok h₄
  exact ⟨hlt, hn', _, hr, hk, hcs ▸ h₂⟩

theorem sampleWith_ok (hs : Hashes V) {t t₁ : Transcript} {c : F128} (h : sampleWith hs t = .ok (c, t₁)) :
    (t.rounds[t.k]!).n = 1 ∧ t₁.rounds = t.rounds ∧ t₁.k = t.k + 1 := by
  obtain ⟨cs, t₂, h₃, h₄⟩ := bind_ok h
  obtain ⟨-, hn', -, hr, hk⟩ := squeezeWith_ok hs Nat.one_ne_zero h₃
  obtain ⟨rfl, rfl⟩ := pure_ok h₄
  exact ⟨hn', hr, hk⟩

theorem sampleKWith_ok (hs : Hashes V) {t t₁ : Transcript} {c : F256} (h : sampleKWith hs t = .ok (c, t₁)) :
    (t.rounds[t.k]!).n = 2 ∧ t₁.rounds = t.rounds ∧ t₁.k = t.k + 1 := by
  obtain ⟨cs, t₂, h₃, h₄⟩ := bind_ok h
  obtain ⟨-, hn', -, hr, hk⟩ := squeezeWith_ok hs (by decide) h₃
  obtain ⟨rfl, rfl⟩ := pure_ok h₄
  exact ⟨hn', hr, hk⟩

/-! ## Loops -/

/-- A loop over `a, …, a + n − 1` whose every step yields after moving `d` rounds moves `n · d` rounds. -/
theorem forIn_moves {σ : Type} (body : ℕ → σ → TM (ForInStep σ)) (d : ℕ)
    (hbody : ∀ i s t b t', body i s t = .ok (b, t') →
      t'.rounds = t.rounds ∧ t'.k = t.k + d ∧ ∃ s', b = .yield s') :
    ∀ (n a : ℕ) (s : σ) (t t' : Transcript) (out : σ), forIn (List.range' a n) s body t = .ok (out, t') →
      t'.rounds = t.rounds ∧ t'.k = t.k + n * d := by
  intro n
  induction n with
  | zero =>
    intro a s t t' out h
    simp only [List.range'_zero, List.forIn_nil] at h
    obtain ⟨rfl, rfl⟩ := pure_ok h
    simp
  | succ n ih =>
    intro a s t t' out h
    rw [forIn_range'_cons] at h
    obtain ⟨b, t₁, h₁, h₂⟩ := bind_ok h
    obtain ⟨hr₁, hk₁, s', rfl⟩ := hbody a s t b t₁ h₁
    obtain ⟨hr, hk⟩ := ih (a + 1) s' t₁ t' out h₂
    exact ⟨hr.trans hr₁, by rw [hk, hk₁, Nat.succ_mul]; omega⟩

/-- A loop whose every step keeps the rounds and the counter keeps them. -/
theorem forIn_keeps {γ σ : Type} (body : γ → σ → TM (ForInStep σ))
    (hbody : ∀ x s t b t', body x s t = .ok (b, t') → t'.rounds = t.rounds ∧ t'.k = t.k) :
    ∀ (l : List γ) (s : σ) (t t' : Transcript) (out : σ), forIn l s body t = .ok (out, t') →
      t'.rounds = t.rounds ∧ t'.k = t.k := by
  intro l
  induction l with
  | nil =>
    intro s t t' out h
    simp only [List.forIn_nil] at h
    obtain ⟨rfl, rfl⟩ := pure_ok h
    exact ⟨rfl, rfl⟩
  | cons x l ih =>
    intro s t t' out h
    rw [List.forIn_cons] at h
    obtain ⟨b, t₁, h₁, h₂⟩ := bind_ok h
    obtain ⟨hr₁, hk₁⟩ := hbody x s t b t₁ h₁
    cases b with
    | done s' => obtain ⟨rfl, rfl⟩ := pure_ok h₂; exact ⟨hr₁, hk₁⟩
    | yield s' =>
      obtain ⟨hr, hk⟩ := ih s' t₁ t' out h₂
      exact ⟨hr.trans hr₁, hk.trans hk₁⟩

/-! ## Zerocheck, lincheck and the ring switch -/

/-- **Zerocheck reads `3 + (m − 6)` rounds**: the skip and outer points, the first coin, one per residual round. -/
theorem zerocheckWith_rounds (hs : Hashes V) {m : ℕ} (hm : 14 ≤ m) {digest cap : ByteArray} {p : ZerocheckProof}
    {t t' : Transcript} {zc : ZcOut} (h : bindAndZerocheckWith hs m digest cap p t = .ok (zc, t')) :
    t'.rounds = t.rounds ∧ t'.k = t.k + 3 + (m - 6) := by
  unfold bindAndZerocheckWith at h
  simp only [liftTM_V] at h
  obtain ⟨t₁, k₁, h⟩ := label_bind h
  obtain ⟨t₂, k₂, h⟩ := absorbBytes_bind h
  obtain ⟨t₃, k₃, h⟩ := absorbBytes_bind h
  obtain ⟨t₄, k₄, h⟩ := label_bind h
  obtain ⟨-, -, t₅, r₅, kk₅, h⟩ := sampleVecWith_bind hs (by omega) h
  obtain ⟨-, -, t₆, r₆, kk₆, h⟩ := sampleVecWith_bind hs (by omega) h
  obtain ⟨t₇, k₇, h⟩ := absorbSlice_bind h
  obtain ⟨t₈, k₈, h⟩ := absorbSlice_bind h
  obtain ⟨-, -, t₉, r₉, kk₉, h⟩ := sampleWith_bind hs h
  obtain ⟨-, h⟩ := check_bind h
  obtain ⟨s, t₁₀, hloop, h⟩ := bind_ok h
  obtain ⟨-, h⟩ := check_bind h
  obtain ⟨t₁₁, k₁₁, h⟩ := absorbF_bind h
  obtain ⟨t₁₂, k₁₂, h⟩ := absorbF_bind h
  obtain ⟨rfl, rfl⟩ := pure_ok h
  rw [forIn_range_eq] at hloop
  obtain ⟨hr₁₀, hk₁₀⟩ := forIn_moves _ 1 (by
      intro i st tt b tt' hb
      rcases hg : p.rounds[i]! with ⟨g1, ginf⟩
      simp only [hg] at hb
      obtain ⟨tt₁, kk₁, hb⟩ := absorbF_bind hb
      obtain ⟨tt₂, kk₂, hb⟩ := absorbF_bind hb
      obtain ⟨-, -, tt₃, rr₃, kk₃, hb⟩ := sampleWith_bind hs hb
      obtain ⟨rfl, rfl⟩ := pure_ok hb
      exact ⟨by rw [rr₃, kk₂.2, kk₁.2], by rw [kk₃, kk₂.1, kk₁.1], _, rfl⟩) (m - 6) 0 _ t₉ t₁₀ s hloop
  refine ⟨by rw [k₁₂.2, k₁₁.2, hr₁₀, r₉, k₈.2, k₇.2, r₆, r₅, k₄.2, k₃.2, k₂.2, k₁.2], ?_⟩
  rw [k₁₂.1, k₁₁.1, hk₁₀, kk₉, k₈.1, k₇.1, kk₆, kk₅, k₄.1, k₃.1, k₂.1, k₁.1]
  omega

/-- **Lincheck reads `3 + (kLog − 6)` rounds**: `α`, `β`, one per sumcheck round, and `r'`. -/
theorem lincheckWith_rounds (hs : Hashes V) {c : CircuitFold} {zc : ZcOut} {p : LincheckProof} {t t' : Transcript}
    {lc : LcOut} (h : lincheckWith hs c zc p t = .ok (lc, t')) :
    t'.rounds = t.rounds ∧ t'.k = t.k + 3 + (c.kLog - 6) := by
  unfold lincheckWith at h
  simp only [liftTM_V] at h
  obtain ⟨t₁, k₁, h⟩ := label_bind h
  obtain ⟨-, -, t₂, r₂, kk₂, h⟩ := sampleWith_bind hs h
  obtain ⟨-, -, t₃, r₃, kk₃, h⟩ := sampleWith_bind hs h
  obtain ⟨s, t₄, hloop, h⟩ := bind_ok h
  cases hf : c.folded ((t₁.rounds[t₁.k]!).coins.getD 0 0) ((t₂.rounds[t₂.k]!).coins.getD 0 0) (skipWeights zc.z)
      (zc.rho.extract 0 (c.kLog - 6)) s.2 with
  | error err => rw [hf] at h; obtain ⟨_, _, h0, _⟩ := bind_ok h; cases h0
  | ok v =>
  rw [hf] at h
  simp only [pure_bind] at h
  obtain ⟨-, h⟩ := check_bind h
  obtain ⟨t₆, k₆, h⟩ := absorbSlice_bind h
  obtain ⟨-, h⟩ := check_bind h
  obtain ⟨-, -, t₇, r₇, kk₇, h⟩ := sampleWith_bind hs h
  obtain ⟨rfl, rfl⟩ := pure_ok h
  rw [forIn_range_eq] at hloop
  obtain ⟨hr₄, hk₄⟩ := forIn_moves _ 1 (by
      intro i st tt b tt' hb
      rcases hg : p.rounds[i]! with ⟨q1, qinf⟩
      simp only [hg] at hb
      obtain ⟨tt₁, kk₁, hb⟩ := absorbF_bind hb
      obtain ⟨tt₂, kk₂, hb⟩ := absorbF_bind hb
      obtain ⟨-, -, tt₃, rr₃, kk₃, hb⟩ := sampleWith_bind hs hb
      obtain ⟨rfl, rfl⟩ := pure_ok hb
      exact ⟨by rw [rr₃, kk₂.2, kk₁.2], by rw [kk₃, kk₂.1, kk₁.1], _, rfl⟩) (c.kLog - 6) 0 _ t₃ t₄ s hloop
  refine ⟨by rw [r₇, k₆.2, hr₄, r₃, r₂, k₁.2], ?_⟩
  rw [kk₇, k₆.1, hk₄, kk₃, kk₂, k₁.1]
  omega

/-- **The ring switch reads `claims + 1` rounds**: one per claim, then the batching coins. -/
theorem ringSwitchWith_rounds (hs : Hashes V) {m : ℕ} {claims : Array Claim} {rs : Array (Array F128)}
    (hc : claims.size ≠ 0) {t t' : Transcript} {out : F128 × Array Switched}
    (h : ringSwitchWith hs m claims rs t = .ok (out, t')) :
    t'.rounds = t.rounds ∧ t'.k = t.k + (claims.size + 1) := by
  unfold ringSwitchWith at h
  simp only [liftTM_V] at h
  obtain ⟨t₁, k₁, h⟩ := label_bind h
  obtain ⟨s, t₂, hloop, h⟩ := bind_ok h
  obtain ⟨-, -, t₃, r₃, kk₃, h⟩ := sampleVecWith_bind hs hc h
  obtain ⟨rfl, rfl⟩ := pure_ok h
  rw [forIn_range_eq] at hloop
  obtain ⟨hr₂, hk₂⟩ := forIn_moves _ 1 (by
      intro i st tt b tt' hb
      obtain ⟨tt₁, kk₁, hb⟩ := label_bind hb
      obtain ⟨tt₂, kk₂, hb⟩ := absorbSlice_bind hb
      obtain ⟨-, hb⟩ := check_bind hb
      obtain ⟨-, -, tt₃, rr₃, kk₃, hb⟩ := sampleVecWith_bind hs (by decide) hb
      obtain ⟨rfl, rfl⟩ := pure_ok hb
      exact ⟨by rw [rr₃, kk₂.2, kk₁.2], by rw [kk₃, kk₂.1, kk₁.1], _, rfl⟩) claims.size 0 _ t₁ t₂ s hloop
  refine ⟨by rw [r₃, hr₂, k₁.2], ?_⟩
  rw [kk₃, hk₂, k₁.1]
  omega

/-! ## Ligerito's steps -/

theorem liftW_V (x : TM α) : liftW (M := V) x = lift x := rfl

theorem lsampleVecWith_bind (hs : Hashes V) {n : ℕ} (hn : n ≠ 0) {f : Array F128 → LM β} {st st' : LState}
    {t t' : Transcript} {b : β} (h : (liftW (M := V) (sampleVecWith hs n) >>= f) st t = .ok ((b, st'), t')) :
    (t.rounds[t.k]!).n = n ∧
      ∃ t₁, t₁.rounds = t.rounds ∧ t₁.k = t.k + 1 ∧ f (t.rounds[t.k]!).coins st t₁ = .ok ((b, st'), t') := by
  rw [liftW_V] at h
  obtain ⟨cs, t₁, h₁, h₂⟩ := lift_bind h
  obtain ⟨hn', hcs, hr, hk⟩ := sampleVecWith_ok hs hn h₁
  exact ⟨hn', t₁, hr, hk, hcs ▸ h₂⟩

theorem lsampleWith_bind (hs : Hashes V) {f : F128 → LM β} {st st' : LState} {t t' : Transcript} {b : β}
    (h : (liftW (M := V) (sampleWith hs) >>= f) st t = .ok ((b, st'), t')) :
    (t.rounds[t.k]!).n = 1 ∧ ∃ c t₁, t₁.rounds = t.rounds ∧ t₁.k = t.k + 1 ∧ f c st t₁ = .ok ((b, st'), t') := by
  rw [liftW_V] at h
  obtain ⟨c, t₁, h₁, h₂⟩ := lift_bind h
  obtain ⟨hn', hr, hk⟩ := sampleWith_ok hs h₁
  exact ⟨hn', c, t₁, hr, hk, h₂⟩

theorem lsampleKWith_bind (hs : Hashes V) {f : F256 → LM β} {st st' : LState} {t t' : Transcript} {b : β}
    (h : (liftW (M := V) (sampleKWith hs) >>= f) st t = .ok ((b, st'), t')) :
    (t.rounds[t.k]!).n = 2 ∧ ∃ c t₁, t₁.rounds = t.rounds ∧ t₁.k = t.k + 1 ∧ f c st t₁ = .ok ((b, st'), t') := by
  rw [liftW_V] at h
  obtain ⟨c, t₁, h₁, h₂⟩ := lift_bind h
  obtain ⟨hn', hr, hk⟩ := sampleKWith_ok hs h₁
  exact ⟨hn', c, t₁, hr, hk, h₂⟩

/-- `challengeWith`: one `K` coin. -/
theorem challengeWith_bind (hs : Hashes V) {p : LigeritoProof} {f : F256 → LM β} {st st' : LState}
    {t t' : Transcript} {b : β} (h : (challengeWith hs p >>= f) st t = .ok ((b, st'), t')) :
    ∃ c st₁ t₁, t₁.rounds = t.rounds ∧ t₁.k = t.k + 1 ∧ f c st₁ t₁ = .ok ((b, st'), t') := by
  obtain ⟨r, st₁, t₁, h₁, h₂⟩ := lbind_ok h
  unfold challengeWith at h₁
  simp only [liftLM_V] at h₁
  obtain ⟨-, c, t₃, hr₃, hk₃, h₁⟩ := lsampleKWith_bind hs h₁
  have h₁ := lmodify_bind h₁
  obtain ⟨m, t₄, -, k₄, h₁⟩ := nextMsg_bind h₁
  have h₁ := lmodify_bind h₁
  obtain ⟨rfl, rfl, rfl⟩ := lpure_ok h₁
  exact ⟨_, _, _, k₄.2.trans hr₃, k₄.1.trans hk₃, h₂⟩

/-- `oodIntroWith`: an out-of-domain point and its batching coin, two rounds. -/
theorem oodIntroWith_bind (hs : Hashes V) {p : LigeritoProof} {dim split : ℕ} (hdim : dim ≠ 0) {f : PUnit → LM β}
    {st st' : LState} {t t' : Transcript} {b : β} (h : (oodIntroWith hs p dim split >>= f) st t = .ok ((b, st'), t')) :
    ∃ st₁ t₁, t₁.rounds = t.rounds ∧ t₁.k = t.k + 2 ∧ f () st₁ t₁ = .ok ((b, st'), t') := by
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
  exact ⟨_, _, by rw [hr₇, k₆.2, k₅.2, k₄.2, hr₃], by rw [hk₇, k₆.1, k₅.1, k₄.1, hk₃], h₂⟩

/-- `queryLevelWith`: the query coins are round `t.k`'s, `q` of them, and the Merkle check passes at their positions;
the consistency coins are round `t.k + 1`'s. -/
theorem queryLevelWith_bind (hs : Hashes V) {ms : MerkleScheme} {p : LigeritoProof} {lv : Flock.Level}
    {cap : Array ByteArray} {rows : Array (Array F128)} {sibs salts : Array ByteArray} {fold : Array F256}
    (hq : lv.q ≠ 0) {f : Array ℕ × Array F128 × F256 → LM β} {st st' : LState} {t t' : Transcript} {b : β}
    (h : (queryLevelWith hs ms p lv cap rows sibs salts fold >>= f) st t = .ok ((b, st'), t')) :
    (t.rounds[t.k]!).n = lv.q ∧
      merkleCheckWith (ms.opsWith hs) ms cap lv.cap lv.d (positions lv (t.rounds[t.k]!).coins) rows sibs salts
        (2 ^ lv.lanes) = .ok () ∧
      ∃ r st₁ t₁, t₁.rounds = t.rounds ∧ t₁.k = t.k + 2 ∧ f r st₁ t₁ = .ok ((b, st'), t') := by
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
  obtain ⟨hn₄, t₄, hr₄, hk₄, h₁⟩ := lsampleVecWith_bind hs hq h₁
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
    refine ⟨by rw [← k₃.2, ← k₃.1]; exact hn₄, by rw [← k₃.2, ← k₃.1]; exact hmc, _, _, _, ?_, ?_, h₂⟩
    · rw [hr₆, k₅.2, hr₄, k₃.2]
    · rw [hk₆, k₅.1, hk₄, k₃.1]

/-- An LM loop over `a, …, a + n − 1` whose every step yields after moving `d` rounds moves `n · d` rounds. -/
theorem lforIn_moves {σ : Type} (body : ℕ → σ → LM (ForInStep σ)) (d : ℕ)
    (hbody : ∀ i s st t b st' t', body i s st t = .ok ((b, st'), t') →
      t'.rounds = t.rounds ∧ t'.k = t.k + d ∧ ∃ s', b = .yield s') :
    ∀ (n a : ℕ) (s : σ) (st : LState) (t : Transcript) (out : σ) (st' : LState) (t' : Transcript),
      forIn (List.range' a n) s body st t = .ok ((out, st'), t') → t'.rounds = t.rounds ∧ t'.k = t.k + n * d := by
  intro n
  induction n with
  | zero =>
    intro a s st t out st' t' h
    simp only [List.range'_zero, List.forIn_nil] at h
    obtain ⟨rfl, rfl, rfl⟩ := lpure_ok h
    simp
  | succ n ih =>
    intro a s st t out st' t' h
    rw [List.range'_succ, List.forIn_cons] at h
    obtain ⟨b, st₁, t₁, h₁, h₂⟩ := lbind_ok h
    obtain ⟨hr₁, hk₁, s', rfl⟩ := hbody a s st t b st₁ t₁ h₁
    obtain ⟨hr, hk⟩ := ih (a + 1) s' st₁ t₁ out st' t' h₂
    exact ⟨hr.trans hr₁, by rw [hk, hk₁, Nat.succ_mul]; omega⟩

/-- Ligerito's level loop: each of the first `R − 1` iterations moves `11` rounds, the last `7`. -/
theorem lforIn_levels {σ : Type} (body : ℕ → σ → LM (ForInStep σ)) (R : ℕ)
    (hstep : ∀ i s st t b st' t', i + 1 < R → body i s st t = .ok ((b, st'), t') →
      t'.rounds = t.rounds ∧ t'.k = t.k + 11 ∧ ∃ s', b = .yield s')
    (hlast : ∀ i s st t b st' t', i + 1 = R → body i s st t = .ok ((b, st'), t') →
      t'.rounds = t.rounds ∧ t'.k = t.k + 7 ∧ ∃ s', b = .yield s') :
    ∀ (left i : ℕ) (s : σ) (st : LState) (t : Transcript) (out : σ) (st' : LState) (t' : Transcript),
      0 < left → i + left = R → forIn (List.range' i left) s body st t = .ok ((out, st'), t') →
      t'.rounds = t.rounds ∧ t'.k = t.k + (11 * (left - 1) + 7)
  | 0, _, _, _, _, _, _, _, h0, _, _ => absurd h0 (lt_irrefl 0)
  | 1, i, s, st, t, out, st', t', _, hR, h => by
      rw [List.range'_succ, List.forIn_cons] at h
      obtain ⟨b, st₁, t₁, h₁, h₂⟩ := lbind_ok h
      obtain ⟨hr₁, hk₁, s', rfl⟩ := hlast i s st t b st₁ t₁ (by omega) h₁
      simp only [List.range'_zero, List.forIn_nil] at h₂
      obtain ⟨rfl, rfl, rfl⟩ := lpure_ok h₂
      exact ⟨hr₁, by rw [hk₁]⟩
  | left + 2, i, s, st, t, out, st', t', _, hR, h => by
      rw [List.range'_succ, List.forIn_cons] at h
      obtain ⟨b, st₁, t₁, h₁, h₂⟩ := lbind_ok h
      obtain ⟨hr₁, hk₁, s', rfl⟩ := hstep i s st t b st₁ t₁ (by omega) h₁
      obtain ⟨hr, hk⟩ := lforIn_levels body R hstep hlast (left + 1) (i + 1) s' st₁ t₁ out st' t' (by omega)
        (by omega) h₂
      exact ⟨hr.trans hr₁, by rw [hk, hk₁]; omega⟩

/-- **Ligerito reads `k0 + 5 + 11 (R)` rounds** on `R + 1 ≥ 2` levels: level 0's out-of-domain point and coin, the `k0`
lane challenges, level 1's two out-of-domain intros, level 0's query and consistency coins and its batching coin, then
`11` rounds per inner level and `7` for the last. Level 0's query coins are round `t.k + k0 + 6`, and the Merkle check
of level 0's opened rows passes at their positions. -/
theorem ligeritoWith_rounds (hs : Hashes V) (ms : MerkleScheme) (sch : Schedule) (T : F128) (cap : Array ByteArray)
    (p : LigeritoProof) (basisAt : Array F256 → ℕ → Array F256) (hn : sch.n ≠ 0) (h2 : 2 ≤ sch.levels.size)
    (hq : ∀ i < sch.levels.size, (sch.levels.getD i default).q ≠ 0) {t t' : Transcript} {u : Unit}
    (h : ligeritoWith hs ms sch T cap p basisAt t = .ok (u, t')) :
    t'.rounds = t.rounds ∧ t'.k = t.k + (sch.k0 + 5 + 11 * (sch.levels.size - 1)) ∧
      (t.rounds[t.k + (sch.k0 + 6)]!).n = (sch.levels.getD 0 default).q ∧
      merkleCheckWith (ms.opsWith hs) ms cap (sch.levels.getD 0 default).cap (sch.levels.getD 0 default).d
        (positions (sch.levels.getD 0 default) (t.rounds[t.k + (sch.k0 + 6)]!).coins) p.initialProof.rows
        p.initialProof.siblings p.initialProof.salts (2 ^ (sch.levels.getD 0 default).lanes) = .ok () := by
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
  obtain ⟨hr₁₀, hk₁₀⟩ := lforIn_moves _ 1 (by
      intro i s st tt b st' tt' hb
      obtain ⟨c, st₁, tt₁, hr₁, hk₁, hb⟩ := challengeWith_bind hs hb
      obtain ⟨rfl, rfl, rfl⟩ := lpure_ok hb
      exact ⟨hr₁, hk₁, _, rfl⟩) sch.k0 0 _ _ t₉ initCh st₁₀ t₁₀ h₁₀
  obtain ⟨_, t₁₁, h₁₁, h₄⟩ := lift_bind h₄
  have k₁₁ := absorbBytes_ok h₁₁
  obtain ⟨st₁₂, t₁₂, hr₁₂, hk₁₂, h₄⟩ := oodIntroWith_bind hs (by omega) h₄
  obtain ⟨st₁₃, t₁₃, hr₁₃, hk₁₃, h₄⟩ := oodIntroWith_bind hs (by omega) h₄
  obtain ⟨hnQ, hmc, ⟨pos0, al0, e0⟩, st₁₄, t₁₄, hr₁₄, hk₁₄, h₄⟩ := queryLevelWith_bind hs (hq 0 (by omega)) h₄
  dsimp only at h₄
  obtain ⟨mq, t₁₅, -, k₁₅, h₄⟩ := nextMsg_bind h₄
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
  have hfin : finalProg p basisAt sch.n initCh out.2.2.1 out.2.2.2 st₁₈ t₁₈ = .ok ((u', stF), t') := hcont
  obtain ⟨-, rfl, -⟩ := finalProg_ok hfin
  have er₁₃ : t₁₃.rounds = t.rounds := by
    rw [hr₁₃, hr₁₂, k₁₁.2, hr₁₀, k₉.2, hr₈, k₇.2, k₆.2, hr₅, k₃.2, k₂.2, k₁.2]
  have ek₁₃ : t₁₃.k = t.k + (sch.k0 + 6) := by
    rw [hk₁₃, hk₁₂, k₁₁.1, hk₁₀, k₉.1, hk₈, k₇.1, k₆.1, hk₅, k₃.1, k₂.1, k₁.1]; omega
  refine ⟨by rw [hrL, hr₁₇, k₁₆.2, k₁₅.2, hr₁₄, er₁₃], ?_, by rw [← er₁₃, ← ek₁₃]; exact hnQ,
    by rw [← er₁₃, ← ek₁₃]; exact hmc⟩
  rw [hkL, hk₁₇, k₁₆.1, k₁₅.1, hk₁₄, ek₁₃]
  omega

end FlockDeploy
