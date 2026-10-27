import Proofs.Flock.Claims.DeployK
import Proofs.Flock.Claims.Bridge.Custody

/-!
# The coins each recorded round issues

A squeeze reads its round only where that round issued the coins it asks for (`squeezeWith_ok`), and `verifyWith h`
refuses a rep whose rounds it did not all consume, so where a rep accepts, every round of its stream issued the number
of coins the statement fixes, whatever the hashes (`ReadsN`): zerocheck's at `m` (`zcNs`), lincheck's at `kLog`
(`lcNs`), the ring switch's at its claims (`rsNs`) and Ligerito's at the schedule (`ligNs`, its levels' `levelNs`), in
that order (`repNs`, `verifyRepWith_readsN`). So each rep stream of an accepted plain record issued its table's
`plainRepNs` (`streams_ns_plain`), its rounds in `spec.repStreams` order the session's `plainNs`
(`rounds_ns_plain`), and custody reads each round message's coin at that count (`custody_vals_plain`).
-/

namespace FlockDeploy

open Flock Transcript FlockSoundness.Refine

/-! ## Rounds and their coin counts -/

/-- **Rounds `k, k + 1, …` of `rs` issued `ns` coins.** -/
def ReadsN (rs : Array Round) (k : ℕ) (ns : List ℕ) : Prop :=
  ∀ i (h : i < ns.length), (rs[k + i]!).n = ns[i]

theorem readsN_nil (rs : Array Round) (k : ℕ) : ReadsN rs k [] :=
  fun _ h => absurd h (Nat.not_lt_zero _)

theorem readsN_cons {rs : Array Round} {k k' a : ℕ} {ns : List ℕ} (h : (rs[k]!).n = a) (hk : k' = k + 1)
    (h' : ReadsN rs k' ns) : ReadsN rs k (a :: ns) := by
  subst hk
  intro i hi
  cases i with
  | zero => simpa using h
  | succ i =>
    have e : k + (i + 1) = k + 1 + i := by omega
    rw [e]
    exact h' i (by simp only [List.length_cons] at hi; omega)

theorem readsN_append {rs : Array Round} {k k' : ℕ} {ns₁ ns₂ : List ℕ} (h₁ : ReadsN rs k ns₁)
    (hk : k' = k + ns₁.length) (h₂ : ReadsN rs k' ns₂) : ReadsN rs k (ns₁ ++ ns₂) := by
  subst hk
  intro i hi
  by_cases hlt : i < ns₁.length
  · rw [List.getElem_append_left hlt]
    exact h₁ i hlt
  · have e : k + i = k + ns₁.length + (i - ns₁.length) := by omega
    rw [List.getElem_append_right (by omega), e]
    exact h₂ _ (by simp only [List.length_append] at hi; omega)

/-- **The coin counts of all of `rs`**, where its rounds from `0` issued `ns` and it has as many rounds. -/
theorem map_n_of_readsN {rs : Array Round} {ns : List ℕ} (h : ReadsN rs 0 ns) (hs : rs.size = ns.length) :
    rs.toList.map Round.n = ns := by
  refine List.ext_getElem (by simp [hs]) fun i h₁ h₂ => ?_
  have hi : i < rs.size := by simpa using h₁
  have := h i h₂
  rw [Nat.zero_add, getElem!_pos rs i hi] at this
  simpa using this

/-! ## Loops -/

/-- A loop whose every step reads one round of `c` coins reads `n` such rounds. -/
theorem forIn_readsN {σ : Type} (body : ℕ → σ → TM (ForInStep σ)) (c : ℕ)
    (hbody : ∀ i s t b t', body i s t = .ok (b, t') →
      (t.rounds[t.k]!).n = c ∧ t'.rounds = t.rounds ∧ t'.k = t.k + 1 ∧ ∃ s', b = .yield s') :
    ∀ (n a : ℕ) (s : σ) (t t' : Transcript) (out : σ), forIn (List.range' a n) s body t = .ok (out, t') →
      t'.rounds = t.rounds ∧ t'.k = t.k + n ∧ ReadsN t.rounds t.k (List.replicate n c) := by
  intro n
  induction n with
  | zero =>
    intro a s t t' out h
    simp only [List.range'_zero, List.forIn_nil] at h
    obtain ⟨rfl, rfl⟩ := pure_ok h
    exact ⟨rfl, rfl, readsN_nil _ _⟩
  | succ n ih =>
    intro a s t t' out h
    rw [forIn_range'_cons] at h
    obtain ⟨b, t₁, h₁, h₂⟩ := bind_ok h
    obtain ⟨hn, hr₁, hk₁, s', rfl⟩ := hbody a s t b t₁ h₁
    obtain ⟨hr, hk, hR⟩ := ih (a + 1) s' t₁ t' out h₂
    rw [hr₁] at hr hR
    refine ⟨hr, by rw [hk, hk₁]; omega, ?_⟩
    rw [List.replicate_succ]
    exact readsN_cons hn hk₁ hR

/-- An LM loop whose every step reads one round of `c` coins reads `n` such rounds. -/
theorem lforIn_readsN {σ : Type} (body : ℕ → σ → LM (ForInStep σ)) (c : ℕ)
    (hbody : ∀ i s st t b st' t', body i s st t = .ok ((b, st'), t') →
      (t.rounds[t.k]!).n = c ∧ t'.rounds = t.rounds ∧ t'.k = t.k + 1 ∧ ∃ s', b = .yield s') :
    ∀ (n a : ℕ) (s : σ) (st : LState) (t : Transcript) (out : σ) (st' : LState) (t' : Transcript),
      forIn (List.range' a n) s body st t = .ok ((out, st'), t') →
      t'.rounds = t.rounds ∧ t'.k = t.k + n ∧ ReadsN t.rounds t.k (List.replicate n c) := by
  intro n
  induction n with
  | zero =>
    intro a s st t out st' t' h
    simp only [List.range'_zero, List.forIn_nil] at h
    obtain ⟨rfl, rfl, rfl⟩ := lpure_ok h
    exact ⟨rfl, rfl, readsN_nil _ _⟩
  | succ n ih =>
    intro a s st t out st' t' h
    rw [List.range'_succ, List.forIn_cons] at h
    obtain ⟨b, st₁, t₁, h₁, h₂⟩ := lbind_ok h
    obtain ⟨hn, hr₁, hk₁, s', rfl⟩ := hbody a s st t b st₁ t₁ h₁
    obtain ⟨hr, hk, hR⟩ := ih (a + 1) s' st₁ t₁ out st' t' h₂
    rw [hr₁] at hr hR
    refine ⟨hr, by rw [hk, hk₁]; omega, ?_⟩
    rw [List.replicate_succ]
    exact readsN_cons hn hk₁ hR

/-- The coins of a loop over `a, …, a + n − 1` whose step `i` reads `F i x` at a counter `x` it moves to `D i x`. -/
def loopNs (F : ℕ → ℕ → List ℕ) (D : ℕ → ℕ → ℕ) : ℕ → ℕ → ℕ → List ℕ
  | _, 0, _ => []
  | a, n + 1, x => F a x ++ loopNs F D (a + 1) n (D a x)

/-- An LM loop whose step `i` reads `F i (d s)` at state `s` and moves `d` to `D i (d s)` reads `loopNs F D`. -/
theorem lforIn_readsN_loop {σ : Type} (body : ℕ → σ → LM (ForInStep σ)) (d : σ → ℕ) (F : ℕ → ℕ → List ℕ)
    (D : ℕ → ℕ → ℕ)
    (hbody : ∀ i s st t b st' t', body i s st t = .ok ((b, st'), t') →
      t'.rounds = t.rounds ∧ t'.k = t.k + (F i (d s)).length ∧ ReadsN t.rounds t.k (F i (d s)) ∧
        ∃ s', b = .yield s' ∧ d s' = D i (d s)) :
    ∀ (n a : ℕ) (s : σ) (st : LState) (t : Transcript) (out : σ) (st' : LState) (t' : Transcript),
      forIn (List.range' a n) s body st t = .ok ((out, st'), t') →
      t'.rounds = t.rounds ∧ t'.k = t.k + (loopNs F D a n (d s)).length ∧
        ReadsN t.rounds t.k (loopNs F D a n (d s)) := by
  intro n
  induction n with
  | zero =>
    intro a s st t out st' t' h
    simp only [List.range'_zero, List.forIn_nil] at h
    obtain ⟨rfl, rfl, rfl⟩ := lpure_ok h
    exact ⟨rfl, rfl, readsN_nil _ _⟩
  | succ n ih =>
    intro a s st t out st' t' h
    rw [List.range'_succ, List.forIn_cons] at h
    obtain ⟨b, st₁, t₁, h₁, h₂⟩ := lbind_ok h
    obtain ⟨hr₁, hk₁, hR₁, s', rfl, hd⟩ := hbody a s st t b st₁ t₁ h₁
    obtain ⟨hr, hk, hR⟩ := ih (a + 1) s' st₁ t₁ out st' t' h₂
    rw [hr₁, hd] at hR
    rw [hd] at hk
    refine ⟨hr.trans hr₁, ?_, readsN_append hR₁ hk₁ hR⟩
    rw [hk, hk₁]
    simp only [loopNs, List.length_append]
    omega

/-! ## Zerocheck, lincheck and the ring switch -/

/-- Zerocheck's coins at `m`: the skip point's `6`, the outer point's `m − 13`, the first coin, one per residual
round. -/
def zcNs (m : ℕ) : List ℕ := [6, m - 13, 1] ++ List.replicate (m - 6) 1

/-- Lincheck's coins at `kLog`: `α`, `β`, one per sumcheck round, and `r'`. -/
def lcNs (kLog : ℕ) : List ℕ := [1, 1] ++ List.replicate (kLog - 6) 1 ++ [1]

/-- The ring switch's coins at `n` claims: `7` per claim, then the `n` batching coins. -/
def rsNs (n : ℕ) : List ℕ := List.replicate n 7 ++ [n]

theorem zerocheckWith_readsN (hs : Hashes Flock.V) {m : ℕ} (hm : 14 ≤ m) {digest cap : ByteArray} {p : ZerocheckProof}
    {t t' : Transcript} {zc : ZcOut} (h : bindAndZerocheckWith hs m digest cap p t = .ok (zc, t')) :
    t'.rounds = t.rounds ∧ t'.k = t.k + (zcNs m).length ∧ ReadsN t.rounds t.k (zcNs m) := by
  unfold bindAndZerocheckWith at h
  simp only [liftTM_V] at h
  obtain ⟨t₁, k₁, h⟩ := label_bind h
  obtain ⟨t₂, k₂, h⟩ := absorbBytes_bind h
  obtain ⟨t₃, k₃, h⟩ := absorbBytes_bind h
  obtain ⟨t₄, k₄, h⟩ := label_bind h
  obtain ⟨-, n₅, t₅, r₅, kk₅, h⟩ := sampleVecWith_bind hs (by omega) h
  obtain ⟨-, n₆, t₆, r₆, kk₆, h⟩ := sampleVecWith_bind hs (by omega) h
  obtain ⟨t₇, k₇, h⟩ := absorbSlice_bind h
  obtain ⟨t₈, k₈, h⟩ := absorbSlice_bind h
  obtain ⟨-, n₉, t₉, r₉, kk₉, h⟩ := sampleWith_bind hs h
  obtain ⟨-, h⟩ := check_bind h
  obtain ⟨s, t₁₀, hloop, h⟩ := bind_ok h
  obtain ⟨-, h⟩ := check_bind h
  obtain ⟨t₁₁, k₁₁, h⟩ := absorbF_bind h
  obtain ⟨t₁₂, k₁₂, h⟩ := absorbF_bind h
  obtain ⟨rfl, rfl⟩ := pure_ok h
  rw [forIn_range_eq] at hloop
  obtain ⟨hr₁₀, hk₁₀, hR⟩ := forIn_readsN _ 1 (by
      intro i st tt b tt' hb
      rcases hg : p.rounds[i]! with ⟨g1, ginf⟩
      simp only [hg] at hb
      obtain ⟨tt₁, kk₁, hb⟩ := absorbF_bind hb
      obtain ⟨tt₂, kk₂, hb⟩ := absorbF_bind hb
      obtain ⟨-, nn₃, tt₃, rr₃, kk₃, hb⟩ := sampleWith_bind hs hb
      obtain ⟨rfl, rfl⟩ := pure_ok hb
      rw [kk₂.2, kk₁.2, kk₂.1, kk₁.1] at nn₃
      exact ⟨nn₃, by rw [rr₃, kk₂.2, kk₁.2], by rw [kk₃, kk₂.1, kk₁.1], _, rfl⟩) (m - 6) 0 _ t₉ t₁₀ s hloop
  have hr₄ : t₄.rounds = t.rounds := by rw [k₄.2, k₃.2, k₂.2, k₁.2]
  have hk₄ : t₄.k = t.k := by rw [k₄.1, k₃.1, k₂.1, k₁.1]
  have hr₈ : t₈.rounds = t.rounds := by rw [k₈.2, k₇.2, r₆, r₅, hr₄]
  have hk₈ : t₈.k = t.k + 2 := by rw [k₈.1, k₇.1, kk₆, kk₅, hk₄]
  rw [hr₄, hk₄] at n₅
  rw [r₅, hr₄, kk₅, hk₄] at n₆
  rw [hr₈, hk₈] at n₉
  rw [r₉, hr₈, kk₉, hk₈] at hR
  refine ⟨by rw [k₁₂.2, k₁₁.2, hr₁₀, r₉, hr₈], ?_, ?_⟩
  · rw [k₁₂.1, k₁₁.1, hk₁₀, kk₉, hk₈]
    simp only [zcNs, List.length_append, List.length_cons, List.length_nil, List.length_replicate]
    omega
  · exact readsN_cons n₅ rfl (readsN_cons n₆ rfl (readsN_cons n₉ (by omega) hR))

theorem lincheckWith_readsN (hs : Hashes Flock.V) {c : CircuitFold} {zc : ZcOut} {p : LincheckProof} {t t' : Transcript}
    {lc : LcOut} (h : lincheckWith hs c zc p t = .ok (lc, t')) :
    t'.rounds = t.rounds ∧ t'.k = t.k + (lcNs c.kLog).length ∧ ReadsN t.rounds t.k (lcNs c.kLog) := by
  unfold lincheckWith at h
  simp only [liftTM_V] at h
  obtain ⟨t₁, k₁, h⟩ := label_bind h
  obtain ⟨-, n₂, t₂, r₂, kk₂, h⟩ := sampleWith_bind hs h
  obtain ⟨-, n₃, t₃, r₃, kk₃, h⟩ := sampleWith_bind hs h
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
  obtain ⟨-, n₇, t₇, r₇, kk₇, h⟩ := sampleWith_bind hs h
  obtain ⟨rfl, rfl⟩ := pure_ok h
  rw [forIn_range_eq] at hloop
  obtain ⟨hr₄, hk₄, hR⟩ := forIn_readsN _ 1 (by
      intro i st tt b tt' hb
      rcases hg : p.rounds[i]! with ⟨q1, qinf⟩
      simp only [hg] at hb
      obtain ⟨tt₁, kk₁, hb⟩ := absorbF_bind hb
      obtain ⟨tt₂, kk₂, hb⟩ := absorbF_bind hb
      obtain ⟨-, nn₃, tt₃, rr₃, kk₃, hb⟩ := sampleWith_bind hs hb
      obtain ⟨rfl, rfl⟩ := pure_ok hb
      rw [kk₂.2, kk₁.2, kk₂.1, kk₁.1] at nn₃
      exact ⟨nn₃, by rw [rr₃, kk₂.2, kk₁.2], by rw [kk₃, kk₂.1, kk₁.1], _, rfl⟩) (c.kLog - 6) 0 _ t₃ t₄ s hloop
  rw [k₁.2, k₁.1] at n₂
  rw [r₂, k₁.2, kk₂, k₁.1] at n₃
  rw [r₃, r₂, k₁.2, kk₃, kk₂, k₁.1] at hR
  have hr₆ : t₆.rounds = t.rounds := by rw [k₆.2, hr₄, r₃, r₂, k₁.2]
  have hk₆ : t₆.k = t.k + 2 + (c.kLog - 6) := by rw [k₆.1, hk₄, kk₃, kk₂, k₁.1]
  rw [hr₆, hk₆] at n₇
  refine ⟨by rw [r₇, hr₆], ?_, ?_⟩
  · rw [kk₇, hk₆]
    simp only [lcNs, List.length_append, List.length_cons, List.length_nil, List.length_replicate]
    omega
  · refine readsN_append (readsN_append (readsN_cons n₂ rfl (readsN_cons n₃ rfl (readsN_nil _ _))) rfl hR) ?_
      (readsN_cons n₇ rfl (readsN_nil _ _))
    simp only [List.length_append, List.length_cons, List.length_nil, List.length_replicate]
    omega

theorem ringSwitchWith_readsN (hs : Hashes Flock.V) {m : ℕ} {claims : Array Claim} {rs : Array (Array F128)}
    (hc : claims.size ≠ 0) {t t' : Transcript} {out : F128 × Array Switched}
    (h : ringSwitchWith hs m claims rs t = .ok (out, t')) :
    t'.rounds = t.rounds ∧ t'.k = t.k + (rsNs claims.size).length ∧ ReadsN t.rounds t.k (rsNs claims.size) := by
  unfold ringSwitchWith at h
  simp only [liftTM_V] at h
  obtain ⟨t₁, k₁, h⟩ := label_bind h
  obtain ⟨s, t₂, hloop, h⟩ := bind_ok h
  obtain ⟨-, n₃, t₃, r₃, kk₃, h⟩ := sampleVecWith_bind hs hc h
  obtain ⟨rfl, rfl⟩ := pure_ok h
  rw [forIn_range_eq] at hloop
  obtain ⟨hr₂, hk₂, hR⟩ := forIn_readsN _ 7 (by
      intro i st tt b tt' hb
      obtain ⟨tt₁, kk₁, hb⟩ := label_bind hb
      obtain ⟨tt₂, kk₂, hb⟩ := absorbSlice_bind hb
      obtain ⟨-, hb⟩ := check_bind hb
      obtain ⟨-, nn₃, tt₃, rr₃, kk₃, hb⟩ := sampleVecWith_bind hs (by decide) hb
      obtain ⟨rfl, rfl⟩ := pure_ok hb
      rw [kk₂.2, kk₁.2, kk₂.1, kk₁.1] at nn₃
      exact ⟨nn₃, by rw [rr₃, kk₂.2, kk₁.2], by rw [kk₃, kk₂.1, kk₁.1], _, rfl⟩) claims.size 0 _ t₁ t₂ s hloop
  rw [k₁.2, k₁.1] at hR
  rw [hr₂, k₁.2, hk₂, k₁.1] at n₃
  refine ⟨by rw [r₃, hr₂, k₁.2], ?_, readsN_append hR (by simp) (readsN_cons n₃ rfl (readsN_nil _ _))⟩
  rw [kk₃, hk₂, k₁.1]
  simp only [rsNs, List.length_append, List.length_replicate, List.length_cons, List.length_nil]
  omega

/-! ## Ligerito's steps -/

/-- `challengeWith`: round `t.k` issued the two coins of one `K` coin. -/
theorem challengeWith_readsN {β : Type} (hs : Hashes Flock.V) {p : LigeritoProof} {f : F256 → LM β}
    {st st' : LState} {t t' : Transcript} {b : β} (h : (challengeWith hs p >>= f) st t = .ok ((b, st'), t')) :
    (t.rounds[t.k]!).n = 2 ∧
      ∃ c st₁ t₁, t₁.rounds = t.rounds ∧ t₁.k = t.k + 1 ∧ f c st₁ t₁ = .ok ((b, st'), t') := by
  obtain ⟨r, st₁, t₁, h₁, h₂⟩ := lbind_ok h
  unfold challengeWith at h₁
  simp only [liftLM_V] at h₁
  obtain ⟨hn, c, t₃, hr₃, hk₃, h₁⟩ := lsampleKWith_bind hs h₁
  have h₁ := lmodify_bind h₁
  obtain ⟨m, t₄, -, k₄, h₁⟩ := nextMsg_bind h₁
  have h₁ := lmodify_bind h₁
  obtain ⟨rfl, rfl, rfl⟩ := lpure_ok h₁
  exact ⟨hn, _, _, _, k₄.2.trans hr₃, k₄.1.trans hk₃, h₂⟩

/-- `oodIntroWith dim`: an out-of-domain point of `dim` coins, then its batching coin. -/
theorem oodIntroWith_readsN {β : Type} (hs : Hashes Flock.V) {p : LigeritoProof} {dim split : ℕ} (hdim : dim ≠ 0)
    {f : PUnit → LM β} {st st' : LState} {t t' : Transcript} {b : β}
    (h : (oodIntroWith hs p dim split >>= f) st t = .ok ((b, st'), t')) :
    ReadsN t.rounds t.k [dim, 1] ∧
      ∃ st₁ t₁, t₁.rounds = t.rounds ∧ t₁.k = t.k + 2 ∧ f () st₁ t₁ = .ok ((b, st'), t') := by
  obtain ⟨u, st₁, t₁, h₁, h₂⟩ := lbind_ok h
  unfold oodIntroWith at h₁
  simp only [liftLM_V] at h₁
  obtain ⟨n₃, t₃, hr₃, hk₃, h₁⟩ := lsampleVecWith_bind hs hdim h₁
  obtain ⟨y, -, h₁⟩ := nextOod_bind h₁
  obtain ⟨_, t₄, h₄, h₁⟩ := lift_bind h₁
  have k₄ := absorbF_ok h₄
  obtain ⟨m, t₅, -, k₅, h₁⟩ := nextMsg_bind h₁
  obtain ⟨t₆, k₆, h₁⟩ := claimPow_bind h₁
  obtain ⟨n₇, bc, t₇, hr₇, hk₇, h₁⟩ := lsampleWith_bind hs h₁
  obtain ⟨rfl, rfl⟩ := lmodify_ok h₁
  have e₆r : t₆.rounds = t.rounds := by rw [k₆.2, k₅.2, k₄.2, hr₃]
  have e₆k : t₆.k = t.k + 1 := by rw [k₆.1, k₅.1, k₄.1, hk₃]
  rw [e₆r, e₆k] at n₇
  exact ⟨readsN_cons n₃ rfl (readsN_cons n₇ rfl (readsN_nil _ _)), _, _, by rw [hr₇, e₆r], by rw [hk₇, e₆k], h₂⟩

/-- A level's query coins, `q`, then its consistency coins, `⌊log₂ (q − 1)⌋ + 1`. -/
def qNs (lv : Flock.Level) : List ℕ := [lv.q, Nat.log2 (lv.q - 1) + 1]

/-- `queryLevelWith lv`: the level's query coins, then its consistency coins (`qNs`). -/
theorem queryLevelWith_readsN {β : Type} (hs : Hashes Flock.V) {ms : MerkleScheme} {p : LigeritoProof}
    {lv : Flock.Level} {cap : Array ByteArray} {rows : Array (Array F128)} {sibs salts : Array ByteArray}
    {fold : Array F256} (hq : lv.q ≠ 0) {f : Array ℕ × Array F128 × F256 → LM β} {st st' : LState}
    {t t' : Transcript} {b : β}
    (h : (queryLevelWith hs ms p lv cap rows sibs salts fold >>= f) st t = .ok ((b, st'), t')) :
    ReadsN t.rounds t.k (qNs lv) ∧
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
  obtain ⟨hn₆, t₆, hr₆, hk₆, h₁⟩ := lsampleVecWith_bind hs (by omega) h₁
  cases hmc : merkleCheckWith (ms.opsWith hs) ms cap lv.cap lv.d (positions lv (t₃.rounds[t₃.k]!).coins) rows sibs
      salts (2 ^ lv.lanes) with
  | error e => simp only [hmc] at h₁; obtain ⟨_, _, _, h₀, _⟩ := lbind_ok h₁; cases h₀
  | ok u =>
    simp only [hmc] at h₁
    simp only [List.forIn_pure_yield_eq_foldl, Std.Legacy.Range.forIn_eq_forIn_range', pure_bind] at h₁
    obtain ⟨rfl, rfl, rfl⟩ := lpure_ok h₁
    have e₅r : t₅.rounds = t.rounds := by rw [k₅.2, hr₄, k₃.2]
    have e₅k : t₅.k = t.k + 1 := by rw [k₅.1, hk₄, k₃.1]
    rw [e₅r, e₅k] at hn₆
    rw [k₃.2, k₃.1] at hn₄
    exact ⟨readsN_cons hn₄ rfl (readsN_cons hn₆ rfl (readsN_nil _ _)), _, _, _, by rw [hr₆, e₅r],
      by rw [hk₆, e₅k], h₂⟩

/-! ## Ligerito -/

/-- Ligerito's level loop at step `i`, at split dimension `x`: the four lane challenges, then at an inner level
(`i + 1 < R`) level `i + 1`'s two out-of-domain intros at `x − 4 + 1`, its query and consistency coins and its batching
coin, and at the last its query and consistency coins and its batching coin. -/
def levelNs (sch : Schedule) (i x : ℕ) : List ℕ :=
  if i + 1 < sch.levels.size - 1 then
    List.replicate 4 2 ++ [x - 4 + 1, 1] ++ [x - 4 + 1, 1] ++ qNs (sch.levels.getD (i + 1) default) ++ [1]
  else List.replicate 4 2 ++ qNs (sch.levels.getD (sch.levels.size - 1) default) ++ [1]

/-- The split dimension after step `i` of the level loop, at `x`. -/
def levelDim (sch : Schedule) (i x : ℕ) : ℕ := if i + 1 < sch.levels.size - 1 then x - 4 + 1 else x

/-- Ligerito's coins at `sch`: level 0's out-of-domain point and coin, the `k0` lane challenges, level 1's two
out-of-domain intros at `n − k0 + 1`, level 0's query and consistency coins and its batching coin, then the level
loop's. -/
def ligNs (sch : Schedule) : List ℕ :=
  [sch.n, 1] ++ List.replicate sch.k0 2 ++ [sch.n - sch.k0 + 1, 1] ++ [sch.n - sch.k0 + 1, 1] ++
    qNs (sch.levels.getD 0 default) ++ [1] ++
    loopNs (levelNs sch) (levelDim sch) 0 (sch.levels.size - 1) (sch.n - sch.k0 + 1)

theorem ligeritoWith_readsN (hs : Hashes Flock.V) (ms : MerkleScheme) (sch : Schedule) (T : F128)
    (cap : Array ByteArray) (p : LigeritoProof) (basisAt : Array F256 → ℕ → Array F256) (hn : sch.n ≠ 0)
    (h2 : 2 ≤ sch.levels.size) (hq : ∀ i < sch.levels.size, (sch.levels.getD i default).q ≠ 0) {t t' : Transcript}
    {u : Unit} (h : ligeritoWith hs ms sch T cap p basisAt t = .ok (u, t')) :
    t'.rounds = t.rounds ∧ t'.k = t.k + (ligNs sch).length ∧ ReadsN t.rounds t.k (ligNs sch) := by
  unfold ligeritoWith at h
  simp only [liftTM_V, liftLM_V] at h
  obtain ⟨-, h⟩ := check_bind h
  obtain ⟨t₁, k₁, h⟩ := label_bind h
  obtain ⟨t₂, k₂, h⟩ := absorbF_bind h
  obtain ⟨t₃, k₃, h⟩ := absorbBytes_bind h
  obtain ⟨⟨u', stF⟩, t₄, h₄, h⟩ := bind_ok h
  obtain ⟨-, rfl⟩ := pure_ok h
  rw [StateT.run] at h₄
  obtain ⟨n₅, t₅, hr₅, hk₅, h₄⟩ := lsampleVecWith_bind hs hn h₄
  obtain ⟨y₀, -, h₄⟩ := nextOod_bind h₄
  obtain ⟨_, t₆, h₆, h₄⟩ := lift_bind h₄
  have k₆ := absorbF_ok h₆
  obtain ⟨t₇, k₇, h₄⟩ := claimPow_bind h₄
  obtain ⟨n₈, bc, t₈, hr₈, hk₈, h₄⟩ := lsampleWith_bind hs h₄
  have h₄ := lmodify_bind h₄
  obtain ⟨m₀, t₉, -, k₉, h₄⟩ := nextMsg_bind h₄
  have h₄ := lmodify_bind h₄
  obtain ⟨initCh, st₁₀, t₁₀, h₁₀, h₄⟩ := lbind_ok h₄
  rw [lforIn_range_eq] at h₁₀
  obtain ⟨hr₁₀, hk₁₀, R₁₀⟩ := lforIn_readsN _ 2 (by
      intro i s st tt b st' tt' hb
      obtain ⟨hc, c, st₁, tt₁, hr₁, hk₁, hb⟩ := challengeWith_readsN hs hb
      obtain ⟨rfl, rfl, rfl⟩ := lpure_ok hb
      exact ⟨hc, hr₁, hk₁, _, rfl⟩) sch.k0 0 _ _ t₉ initCh st₁₀ t₁₀ h₁₀
  obtain ⟨_, t₁₁, h₁₁, h₄⟩ := lift_bind h₄
  have k₁₁ := absorbBytes_ok h₁₁
  obtain ⟨R₁₂, st₁₂, t₁₂, hr₁₂, hk₁₂, h₄⟩ := oodIntroWith_readsN hs (by omega) h₄
  obtain ⟨R₁₃, st₁₃, t₁₃, hr₁₃, hk₁₃, h₄⟩ := oodIntroWith_readsN hs (by omega) h₄
  obtain ⟨R₁₄, ⟨pos0, al0, e0⟩, st₁₄, t₁₄, hr₁₄, hk₁₄, h₄⟩ := queryLevelWith_readsN hs (hq 0 (by omega)) h₄
  dsimp only at h₄
  obtain ⟨mq, t₁₅, -, k₁₅, h₄⟩ := nextMsg_bind h₄
  obtain ⟨t₁₆, k₁₆, h₄⟩ := claimPow_bind h₄
  obtain ⟨n₁₇, bq, t₁₇, hr₁₇, hk₁₇, h₄⟩ := lsampleWith_bind hs h₄
  have h₄ := lmodify_bind h₄
  obtain ⟨out, st₁₈, t₁₈, hloop, hcont⟩ := lbind_ok h₄
  rw [lforIn_range_eq] at hloop
  have hL := lforIn_readsN_loop _ Prod.fst (levelNs sch) (levelDim sch) ?hbody (sch.levels.size - 1) 0 _ _ _ out
    st₁₈ t₁₈ hloop
  case hbody =>
    intro i s st tt b st' tt' hb
    by_cases c4 : s.1 < 4
    · rw [ite_eq_left c4] at hb; obtain ⟨_, _, _, h0, _⟩ := lbind_ok hb; cases h0
    rw [ite_eq_right c4] at hb
    obtain ⟨chs, st₁, tt₁, hf, hb⟩ := lbind_ok hb
    rw [lforIn_range_eq] at hf
    obtain ⟨hr₁, hk₁, R₁⟩ := lforIn_readsN _ 2 (by
        intro j s' st' tt' b' st'' tt'' hb'
        obtain ⟨hc, c, st₁, tt₁, hr₁, hk₁, hb'⟩ := challengeWith_readsN hs hb'
        obtain ⟨rfl, rfl, rfl⟩ := lpure_ok hb'
        exact ⟨hc, hr₁, hk₁, _, rfl⟩) 4 0 _ _ _ chs st₁ tt₁ hf
    by_cases hi : i + 1 < sch.levels.size - 1
    · rw [ite_eq_left hi] at hb
      obtain ⟨_, tt₂, h₂, hb⟩ := lift_bind hb
      have k₂ := absorbBytes_ok h₂
      obtain ⟨R₃, st₃, tt₃, hr₃, hk₃, hb⟩ := oodIntroWith_readsN hs (by omega) hb
      obtain ⟨R₄, st₄, tt₄, hr₄, hk₄, hb⟩ := oodIntroWith_readsN hs (by omega) hb
      obtain ⟨R₅, r₅, st₅, tt₅, hr₅, hk₅, hb⟩ := queryLevelWith_readsN hs (hq (i + 1) (by omega)) hb
      obtain ⟨mm, tt₆, -, k₆, hb⟩ := nextMsg_bind hb
      obtain ⟨tt₇, k₇, hb⟩ := claimPow_bind hb
      obtain ⟨n₈, bb, tt₈, hr₈, hk₈, hb⟩ := lsampleWith_bind hs hb
      have hb := lmodify_bind hb
      by_cases c4' : s.1 - 4 + 1 < 4
      · rw [ite_eq_left c4'] at hb; obtain ⟨_, _, _, h0, _⟩ := lbind_ok hb; cases h0
      rw [ite_eq_right c4'] at hb
      obtain ⟨rfl, rfl, rfl⟩ := lpure_ok hb
      have e₂r : tt₂.rounds = tt.rounds := by rw [k₂.2, hr₁]
      have e₂k : tt₂.k = tt.k + 4 := by rw [k₂.1, hk₁]
      rw [e₂r, e₂k] at R₃
      rw [hr₃, e₂r, hk₃, e₂k] at R₄
      rw [hr₄, hr₃, e₂r, hk₄, hk₃, e₂k] at R₅
      have e₇r : tt₇.rounds = tt.rounds := by rw [k₇.2, k₆.2, hr₅, hr₄, hr₃, e₂r]
      have e₇k : tt₇.k = tt.k + 4 + 2 + 2 + 2 := by rw [k₇.1, k₆.1, hk₅, hk₄, hk₃, e₂k]
      rw [e₇r, e₇k] at n₈
      rw [levelNs, ite_eq_left hi]
      refine ⟨by rw [hr₈, e₇r], ?_, ?_, _, rfl, by rw [levelDim, ite_eq_left hi]⟩
      · rw [hk₈, e₇k]
        simp only [qNs, List.length_append, List.length_replicate, List.length_cons, List.length_nil]
      · refine readsN_append (readsN_append (readsN_append (readsN_append R₁ ?_ R₃) ?_ R₄) ?_ R₅) ?_
          (readsN_cons n₈ rfl (readsN_nil _ _)) <;>
        simp only [qNs, List.length_append, List.length_replicate, List.length_cons, List.length_nil]
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
      obtain ⟨R₃, r₃, st₃, tt₃, hr₃, hk₃, hb⟩ :=
        queryLevelWith_readsN hs (hq (sch.levels.size - 1) (by omega)) hb
      obtain ⟨tt₄, k₄, hb⟩ := claimPow_bind hb
      obtain ⟨n₅, bb, tt₅, hr₅, hk₅, hb⟩ := lsampleWith_bind hs hb
      have hb := lmodify_bind hb
      obtain ⟨rfl, rfl, rfl⟩ := lpure_ok hb
      rw [k₂.2, hr₁, k₂.1, hk₁] at R₃
      have e₄r : tt₄.rounds = tt.rounds := by rw [k₄.2, hr₃, k₂.2, hr₁]
      have e₄k : tt₄.k = tt.k + 4 + 2 := by rw [k₄.1, hk₃, k₂.1, hk₁]
      rw [e₄r, e₄k] at n₅
      rw [levelNs, ite_eq_right hi]
      refine ⟨by rw [hr₅, e₄r], ?_, ?_, _, rfl, by rw [levelDim, ite_eq_right hi]⟩
      · rw [hk₅, e₄k]
        simp only [qNs, List.length_append, List.length_replicate, List.length_cons, List.length_nil]
      · refine readsN_append (readsN_append R₁ ?_ R₃) ?_ (readsN_cons n₅ rfl (readsN_nil _ _)) <;>
        simp only [qNs, List.length_append, List.length_replicate, List.length_cons, List.length_nil]
  obtain ⟨hrL, hkL, RL⟩ := hL
  have hfin : finalProg p basisAt sch.n initCh out.2.2.1 out.2.2.2 st₁₈ t₁₈ = .ok ((u', stF), t') := hcont
  obtain ⟨-, rfl, -⟩ := finalProg_ok hfin
  have e₃r : t₃.rounds = t.rounds := by rw [k₃.2, k₂.2, k₁.2]
  have e₃k : t₃.k = t.k := by rw [k₃.1, k₂.1, k₁.1]
  have e₇r : t₇.rounds = t.rounds := by rw [k₇.2, k₆.2, hr₅, e₃r]
  have e₇k : t₇.k = t.k + 1 := by rw [k₇.1, k₆.1, hk₅, e₃k]
  have e₉r : t₉.rounds = t.rounds := by rw [k₉.2, hr₈, e₇r]
  have e₉k : t₉.k = t.k + 2 := by rw [k₉.1, hk₈, e₇k]
  have e₁₁r : t₁₁.rounds = t.rounds := by rw [k₁₁.2, hr₁₀, e₉r]
  have e₁₁k : t₁₁.k = t.k + 2 + sch.k0 := by rw [k₁₁.1, hk₁₀, e₉k]
  have e₁₆r : t₁₆.rounds = t.rounds := by rw [k₁₆.2, k₁₅.2, hr₁₄, hr₁₃, hr₁₂, e₁₁r]
  have e₁₆k : t₁₆.k = t.k + 2 + sch.k0 + 2 + 2 + 2 := by rw [k₁₆.1, k₁₅.1, hk₁₄, hk₁₃, hk₁₂, e₁₁k]
  rw [e₃r, e₃k] at n₅
  rw [e₇r, e₇k] at n₈
  rw [e₉r, e₉k] at R₁₀
  rw [e₁₁r, e₁₁k] at R₁₂
  rw [hr₁₂, e₁₁r, hk₁₂, e₁₁k] at R₁₃
  rw [hr₁₃, hr₁₂, e₁₁r, hk₁₃, hk₁₂, e₁₁k] at R₁₄
  rw [e₁₆r, e₁₆k] at n₁₇
  rw [hr₁₇, e₁₆r, hk₁₇, e₁₆k] at RL
  refine ⟨by rw [hrL, hr₁₇, e₁₆r], ?_, ?_⟩
  · rw [hkL, hk₁₇, e₁₆k]
    simp only [ligNs, qNs, List.length_append, List.length_replicate, List.length_cons, List.length_nil]
    omega
  · refine readsN_append (readsN_append (readsN_append (readsN_append (readsN_append (readsN_append
      (readsN_cons n₅ rfl (readsN_cons n₈ rfl (readsN_nil _ _))) ?_ R₁₀) ?_ R₁₂) ?_ R₁₃) ?_ R₁₄) ?_
        (readsN_cons n₁₇ rfl (readsN_nil _ _))) ?_ RL <;>
      simp only [qNs, List.length_append, List.length_replicate, List.length_cons, List.length_nil] <;> omega

/-! ## One rep -/

/-- A plain rep's coins at `m`, `kLog`, `claims` claims and the schedule `sch`. -/
def repNs (m kLog claims : ℕ) (sch : Schedule) : List ℕ := zcNs m ++ lcNs kLog ++ rsNs claims ++ ligNs sch

/-- **A plain rep's rounds issued its statement's coins**: when `verifyRepWith hs` accepts rep `r` of table `i` at the
table's `fast100` schedule, whatever the hashes `hs` are, its stream's rounds issued `repNs` at `m`, `kLog` and
`2 + |extras|` claims, one count per round. -/
theorem verifyRepWith_readsN (hs : Hashes Flock.V) {spec : SessionSpec} {st : Setup} {sch : Schedule}
    {sess : Session} {i r : ℕ} {bytes c : ByteArray} (hk : st.fold.kLog = st.kLog) (hf : fast100 st.m = some sch)
    (h : verifyRepWith hs spec st sch sess i r bytes = .ok c) :
    ∃ (stream : StreamRec) (extras : Array Claim),
      sess.streams.find? (fun x => x.name == toString spec.tables[i]! ++ toString "/rep" ++ toString r) = some stream ∧
      st.extra sess.points = .ok extras ∧
      stream.rounds.toList.map Round.n = repNs st.m st.kLog (2 + extras.size) sch := by
  obtain ⟨h22, hn, h2, hq⟩ := fast100_rounds_ok hf
  unfold verifyRepWith at h
  dsimp only at h
  cases hs' : Array.find? (fun x => x.name == toString spec.tables[i]! ++ toString "/rep" ++ toString r)
      sess.streams with
  | none => rw [hs'] at h; cases h
  | some stream =>
  rw [hs'] at h
  obtain ⟨extras, hex, h⟩ := ebind_ok h
  obtain ⟨p, hp, h⟩ := ebind_ok h
  by_cases c1 : (p.params != pcsParams st.merkle sch) = true
  · rw [ite_eq_left c1] at h; obtain ⟨_, h0, _⟩ := ebind_ok h; cases h0
  rw [ite_eq_right c1] at h
  obtain ⟨⟨u, tr⟩, hrun, h⟩ := ebind_ok h
  change (if (tr.k != stream.rounds.size) = true then _ else _) = _ at h
  by_cases c2 : (tr.k != stream.rounds.size) = true
  · rw [ite_eq_left c2] at h; obtain ⟨_, h0, _⟩ := ebind_ok h; cases h0
  simp only [bne_iff_ne, ne_eq, not_not] at c2
  refine ⟨stream, extras, rfl, hex, ?_⟩
  rw [StateT.run] at hrun
  obtain ⟨zc, t₁, h₁, hrun⟩ := bind_ok hrun
  obtain ⟨hr₁, hk₁, R₁⟩ := zerocheckWith_readsN hs (by omega) h₁
  obtain ⟨lc, t₂, h₂, hrun⟩ := bind_ok hrun
  obtain ⟨hr₂, hk₂, R₂⟩ := lincheckWith_readsN hs h₂
  obtain ⟨⟨T, sws⟩, t₃, h₃, hrun⟩ := bind_ok hrun
  obtain ⟨hr₃, hk₃, R₃⟩ := ringSwitchWith_readsN hs (by simp [abcClaims]) h₃
  obtain ⟨hr₄, hk₄, R₄⟩ := ligeritoWith_readsN hs st.merkle sch T p.cap p.ligerito (basis sws) hn h2 hq hrun
  have hsz : (abcClaims st.m st.kLog zc lc ++ extras).size = 2 + extras.size := by simp [abcClaims]
  rw [hsz] at R₃ hk₃
  rw [hk] at R₂ hk₂
  rw [hr₁] at R₂
  rw [hr₂, hr₁] at R₃
  rw [hr₃, hr₂, hr₁] at R₄
  have R := readsN_append (readsN_append (readsN_append R₁ hk₁ R₂) (by rw [hk₂, hk₁, List.length_append]; omega) R₃)
    (by rw [hk₃, hk₂, hk₁]; simp only [List.length_append]; omega) R₄
  refine map_n_of_readsN R ?_
  rw [← c2, hk₄, hk₃, hk₂, hk₁]
  simp only [Transcript.init, repNs, List.length_append]
  omega

/-! ## The plain session -/

/-- A plain rep's coins at its table's statement: `repNs` at the table's `m`, `kLog`, claims and `fast100` schedule. -/
def plainRepNs (spec : SessionSpec) (st : Stmt) : List ℕ :=
  repNs st.m st.c.kLog (2 + spec.points * st.regions.size) (Claims.Flock.plainSched st.m)

theorem map_n_of_forall₂ {l₁ l₂ : List Round} (h : List.Forall₂ (fun a b => a.n = b.n ∧ a.coins = b.coins) l₁ l₂) :
    l₁.map Round.n = l₂.map Round.n := by
  induction h with
  | nil => rfl
  | cons h _ ih => simp [h.1, ih]

/-- **Each plain rep stream's rounds issued its table's coins**: when the plain verdict accepts the record at any
hashes and `liveOf` reads it, there are two rep streams per table, and the rounds of stream `k` (table `k / 2`'s rep
`k % 2`) issued `plainRepNs spec sts[k / 2]`, one count per round, so the statement fixes every round's coin count.
At any `keep` (`verifyWithMK`). -/
theorem streams_ns_plainK (keep : Bool → Bool) (hs : Hashes Flock.V) {spec : SessionSpec} {sts : Array Stmt}
    {record : String} {proofs : Array ByteArray} {lv : Live}
    (hv : verifyWithMK keep hs spec (tabsOf sts) record proofs = .ok ()) (hl : liveOf spec record = .ok lv) :
    lv.streams.size = 2 * sts.size ∧
      ∀ (k : ℕ) (hk : k < lv.streams.size), lv.streams[k].toList.map Round.n = plainRepNs spec sts[k / 2]! := by
  obtain ⟨hsz, hall⟩ := rep_plainK keep hs hv hl
  refine ⟨hsz, fun k hk => ?_⟩
  obtain ⟨sess, sch, stream, c, -, hpn, hf, hrep, hfind, hfa⟩ := hall k hk
  obtain ⟨stream', extras, hfind', hex, hns⟩ := verifyRepWith_readsN hs rfl hf hrep
  rw [hfind] at hfind'
  cases hfind'
  have hx : extras.size = sts[k / 2]!.regions.size * sess.points.size := extraClaims_size hex
  have hsch : Claims.Flock.plainSched sts[k / 2]!.m = sch := by
    simp only [Claims.Flock.plainSched, hf, Option.getD_some]
  rw [map_n_of_forall₂ hfa, hns, hx, hpn, plainRepNs, hsch, Nat.mul_comm]
  rfl

theorem streams_ns_plain (hs : Hashes Flock.V) {spec : SessionSpec} {sts : Array Stmt} {record : String}
    {proofs : Array ByteArray} {lv : Live} (hv : verifyWithM hs spec (tabsOf sts) record proofs = .ok ())
    (hl : liveOf spec record = .ok lv) :
    lv.streams.size = 2 * sts.size ∧
      ∀ (k : ℕ) (hk : k < lv.streams.size), lv.streams[k].toList.map Round.n = plainRepNs spec sts[k / 2]! :=
  streams_ns_plainK id hs hv hl

/-- The plain session's coins, round by round in `spec.repStreams` order: each table's two reps' `plainRepNs`. -/
def plainNs (spec : SessionSpec) (sts : Array Stmt) : List ℕ :=
  (List.range (2 * sts.size)).flatMap fun k => plainRepNs spec sts[k / 2]!

/-- **An accepted plain record's rounds issued the session's coins**: when the plain verdict accepts the record at any
hashes and `liveOf` reads it, its rounds, in `spec.repStreams` order, issued `plainNs spec sts`, one count per round. -/
theorem rounds_ns_plain (hs : Hashes Flock.V) {spec : SessionSpec} {sts : Array Stmt} {record : String}
    {proofs : Array ByteArray} {lv : Live} (hv : verifyWithM hs spec (tabsOf sts) record proofs = .ok ())
    (hl : liveOf spec record = .ok lv) :
    (lv.streams.toList.flatMap (·.toList)).map Round.n = plainNs spec sts := by
  obtain ⟨hsz, hall⟩ := streams_ns_plain hs hv hl
  have e : lv.streams.toList = (List.range lv.streams.size).map fun k => lv.streams[k]! := by
    refine List.ext_getElem (by simp) fun i h₁ h₂ => ?_
    simp only [Array.length_toList] at h₁
    simp [getElem!_pos lv.streams i h₁]
  rw [List.map_flatMap, e, List.flatMap_map, plainNs, ← hsz, List.flatMap_def, List.flatMap_def]
  refine congrArg List.flatten (List.map_congr_left fun k hk => ?_)
  have hk' : k < lv.streams.size := List.mem_range.1 hk
  rw [getElem!_pos lv.streams k hk']
  exact hall k hk'

/-- **Custody reads each round's coin at the statement's count**: where the plain verdict accepts the record at any
hashes and its custody passes on the round messages `mids`, each message's coin is read at its round's count in
`plainNs spec sts`, by a recorded round of that count carrying the message. -/
theorem custody_vals_plain (hs : Hashes Flock.V) {spec : SessionSpec} {sts : Array Stmt} {record : String}
    {proofs : Array ByteArray} {lv : Live} (hv : verifyWithM hs spec (tabsOf sts) record proofs = .ok ())
    (hl : liveOf spec record = .ok lv) {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} (hc : custody spec lv link c₁ mids = .ok ()) :
    List.Forall₂ (fun (n : ℕ) (am : Claims.Msg × Coin) =>
      ∃ rd : Round, rd.n = n ∧ rd.bytes = some ⟨am.1.toArray⟩ ∧ rd.coins = am.2.vals n) (plainNs spec sts) mids := by
  rw [← rounds_ns_plain hs hv hl, List.forall₂_map_left_iff]
  exact (Claims.Bridge.Custody.custody_ok hc).2.2.2.imp fun rd am h => ⟨rd, rfl, h⟩

end FlockDeploy
