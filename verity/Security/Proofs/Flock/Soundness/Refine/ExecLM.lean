import Proofs.Flock.Soundness.Refine.Exec
-- `simp` over the range loops below needs `Std.Legacy.Range.mk.congr_simp`; LookupRows declares it, and the kernel
-- replay rejects a second declaration.
import Proofs.Flock.Level3.LookupRows

/-!
# Runs of the executable's Ligerito programs

The executable's Ligerito verifier (PROTOCOL.md §13) runs in `LM = StateT LState TM`: the running claim `s`, the round
polynomial `q`, the cursors into the proof's lists, and the out-of-domain and consistency contexts, over the replay
transcript. A run that succeeds, `x st t = .ok ((a, st'), t')`, is taken apart one step at a time, as `Exec` does for
`TM`: `lbind_ok` peels a `bind`, and `lift_bind`, `lget_bind`, `lset_bind`, `lmodify_bind` peel the steps that read or
write one of the two states.
-/

namespace FlockSoundness.Refine

open Flock Transcript

variable {α β : Type}

theorem lbind_ok {x : LM α} {f : α → LM β} {st st' : LState} {t t' : Transcript} {b : β}
    (h : (x >>= f) st t = .ok ((b, st'), t')) :
    ∃ a st₁ t₁, x st t = .ok ((a, st₁), t₁) ∧ f a st₁ t₁ = .ok ((b, st'), t') := by
  change (StateT.bind x f) st t = _ at h
  unfold StateT.bind at h
  obtain ⟨⟨a, st₁⟩, t₁, h₁, h₂⟩ := bind_ok h
  exact ⟨a, st₁, t₁, h₁, h₂⟩

theorem lpure_ok {a a' : α} {st st' : LState} {t t' : Transcript}
    (h : (pure a : LM α) st t = .ok ((a', st'), t')) : a' = a ∧ st' = st ∧ t' = t := by
  cases h; exact ⟨rfl, rfl, rfl⟩

theorem lift_ok {x : TM α} {st st' : LState} {t t' : Transcript} {a : α}
    (h : (lift x : LM α) st t = .ok ((a, st'), t')) : st' = st ∧ x t = .ok (a, t') := by
  change (StateT.lift x) st t = _ at h
  unfold StateT.lift at h
  obtain ⟨a', t₁, h₁, h₂⟩ := bind_ok h
  cases h₂
  exact ⟨rfl, h₁⟩

theorem lift_bind {x : TM α} {f : α → LM β} {st st' : LState} {t t' : Transcript} {b : β}
    (h : (lift x >>= f) st t = .ok ((b, st'), t')) :
    ∃ a t₁, x t = .ok (a, t₁) ∧ f a st t₁ = .ok ((b, st'), t') := by
  obtain ⟨a, st₁, t₁, h₁, h₂⟩ := lbind_ok h
  obtain ⟨hst, hx⟩ := lift_ok h₁
  subst hst
  exact ⟨a, t₁, hx, h₂⟩

theorem lget_bind {f : LState → LM β} {st st' : LState} {t t' : Transcript} {b : β}
    (h : (get >>= f) st t = .ok ((b, st'), t')) : f st st t = .ok ((b, st'), t') := by
  obtain ⟨a, st₁, t₁, h₁, h₂⟩ := lbind_ok h
  cases h₁
  exact h₂

theorem lset_bind {s : LState} {f : PUnit → LM β} {st st' : LState} {t t' : Transcript} {b : β}
    (h : (set s >>= f) st t = .ok ((b, st'), t')) : f () s t = .ok ((b, st'), t') := by
  obtain ⟨a, st₁, t₁, h₁, h₂⟩ := lbind_ok h
  cases h₁
  exact h₂

theorem lmodify_bind {g : LState → LState} {f : PUnit → LM β} {st st' : LState} {t t' : Transcript} {b : β}
    (h : (modify g >>= f) st t = .ok ((b, st'), t')) : f () (g st) t = .ok ((b, st'), t') := by
  obtain ⟨a, st₁, t₁, h₁, h₂⟩ := lbind_ok h
  cases h₁
  exact h₂

theorem lmodify_ok {g : LState → LState} {st st' : LState} {t t' : Transcript} {a : PUnit}
    (h : (modify g : LM PUnit) st t = .ok ((a, st'), t')) : st' = g st ∧ t' = t := by
  cases h; exact ⟨rfl, rfl⟩

/-! ## The Ligerito helpers -/

/-- `nextMsg`: the next sumcheck message of the proof, absorbed; the message cursor moves on. -/
theorem nextMsg_ok {p : LigeritoProof} {st st' : LState} {t t' : Transcript} {m : F256 × F256}
    (h : nextMsg p st t = .ok ((m, st'), t')) :
    p.messages[st.msg]? = some m ∧ st' = { st with msg := st.msg + 1 } ∧ Keeps t t' := by
  unfold nextMsg at h
  have h := lget_bind h
  cases hm : p.messages[st.msg]? with
  | none => simp only [hm] at h; cases h
  | some m' =>
    simp only [hm] at h
    have h := lset_bind h
    obtain ⟨_, t₁, h₁, h⟩ := lift_bind h
    obtain ⟨_, t₂, h₂, h⟩ := lift_bind h
    obtain ⟨rfl, rfl, rfl⟩ := lpure_ok h
    obtain ⟨t₁', k₁, h₁'⟩ := absorbK_bind (f := fun _ => pure ()) (by rw [bind_pure]; exact h₁)
    obtain ⟨rfl, rfl⟩ := pure_ok h₁'
    obtain ⟨t₂', k₂, h₂'⟩ := absorbK_bind (f := fun _ => pure ()) (by rw [bind_pure]; exact h₂)
    obtain ⟨rfl, rfl⟩ := pure_ok h₂'
    exact ⟨rfl, rfl, k₂.1.trans k₁.1, k₂.2.trans k₁.2⟩

theorem nextMsg_bind {p : LigeritoProof} {f : F256 × F256 → LM β} {st st' : LState} {t t' : Transcript} {b : β}
    (h : (nextMsg p >>= f) st t = .ok ((b, st'), t')) :
    ∃ m t₁, p.messages[st.msg]? = some m ∧ Keeps t t₁ ∧ f m { st with msg := st.msg + 1 } t₁ = .ok ((b, st'), t') := by
  obtain ⟨m, st₁, t₁, h₁, h₂⟩ := lbind_ok h
  obtain ⟨hm, rfl, hk⟩ := nextMsg_ok h₁
  exact ⟨m, t₁, hm, hk, h₂⟩

/-- `nextOod`: the next out-of-domain value of the proof; the cursor moves on. -/
theorem nextOod_bind {p : LigeritoProof} {f : F128 → LM β} {st st' : LState} {t t' : Transcript} {b : β}
    (h : (nextOod p >>= f) st t = .ok ((b, st'), t')) :
    ∃ v, p.oodValues[st.ood]? = some v ∧ f v { st with ood := st.ood + 1 } t = .ok ((b, st'), t') := by
  obtain ⟨v, st₁, t₁, h₁, h₂⟩ := lbind_ok h
  unfold nextOod at h₁
  have h₁ := lget_bind h₁
  cases hv : p.oodValues[st.ood]? with
  | none => simp only [hv] at h₁; cases h₁
  | some v' =>
    simp only [hv] at h₁
    have h₁ := lset_bind h₁
    obtain ⟨rfl, rfl, rfl⟩ := lpure_ok h₁
    exact ⟨v, rfl, h₂⟩

/-- `nonceFrom`: a present nonce, the state unchanged. -/
theorem nonceFrom_bind {xs : Array UInt64} {i : ℕ} {what : String} {f : UInt64 → LM β} {st st' : LState}
    {t t' : Transcript} {b : β} (h : (nonceFrom xs i what >>= f) st t = .ok ((b, st'), t')) :
    ∃ v, xs[i]? = some v ∧ f v st t = .ok ((b, st'), t') := by
  obtain ⟨v, st₁, t₁, h₁, h₂⟩ := lbind_ok h
  unfold nonceFrom at h₁
  cases hv : xs[i]? with
  | none => simp only [hv] at h₁; cases h₁
  | some v' =>
    simp only [hv] at h₁
    obtain ⟨rfl, rfl, rfl⟩ := lpure_ok h₁
    exact ⟨v, rfl, h₂⟩

/-- `claimPow`: a claim-batching nonce, absorbed; the cursor moves on. -/
theorem claimPow_bind {p : LigeritoProof} {f : PUnit → LM β} {st st' : LState} {t t' : Transcript} {b : β}
    (h : (claimPow p >>= f) st t = .ok ((b, st'), t')) :
    ∃ t₁, Keeps t t₁ ∧ f () { st with cbn := st.cbn + 1 } t₁ = .ok ((b, st'), t') := by
  obtain ⟨u, st₁, t₁, h₁, h₂⟩ := lbind_ok h
  unfold claimPow at h₁
  have h₁ := lget_bind h₁
  obtain ⟨n, _, h₁⟩ := nonceFrom_bind h₁
  have h₁ := lset_bind h₁
  obtain ⟨hst, hx⟩ := lift_ok h₁
  subst hst
  obtain ⟨t₂, k₂, h₃⟩ := pow_bind (f := fun _ => pure ()) (by rw [bind_pure]; exact hx)
  obtain ⟨rfl, rfl⟩ := pure_ok h₃
  exact ⟨_, k₂, h₂⟩

/-- `challenge`: a `K` coin (the next round's two coins), the claim `s ← q(r)`, and the next message's polynomial. -/
theorem challenge_bind {p : LigeritoProof} {f : F256 → LM β} {st st' : LState} {t t' : Transcript} {b : β}
    (h : (challenge p >>= f) st t = .ok ((b, st'), t')) :
    t.k < t.rounds.size ∧ (t.rounds[t.k]!).n = 2 ∧ ∃ m t₁, p.messages[st.msg]? = some m ∧
      t₁.rounds = t.rounds ∧ t₁.k = t.k + 1 ∧
      f ⟨(t.rounds[t.k]!).coins.getD 0 0, (t.rounds[t.k]!).coins.getD 1 0⟩
        { st with s := st.q.eval ⟨(t.rounds[t.k]!).coins.getD 0 0, (t.rounds[t.k]!).coins.getD 1 0⟩,
                  q := Poly.ofMsg m (st.q.eval ⟨(t.rounds[t.k]!).coins.getD 0 0, (t.rounds[t.k]!).coins.getD 1 0⟩),
                  msg := st.msg + 1 } t₁ = .ok ((b, st'), t') := by
  obtain ⟨r, st₁, t₁, h₁, h₂⟩ := lbind_ok h
  unfold challenge at h₁
  obtain ⟨r', t₃, h₃, h₁⟩ := lift_bind h₁
  obtain ⟨hlt, hn, hr', hr₃, hk₃⟩ := sampleK_ok h₃
  have h₁ := lmodify_bind h₁
  obtain ⟨m, t₄, hm, k₄, h₁⟩ := nextMsg_bind h₁
  have h₁ := lmodify_bind h₁
  obtain ⟨rfl, rfl, rfl⟩ := lpure_ok h₁
  subst hr'
  exact ⟨hlt, hn, m, _, hm, k₄.2.trans hr₃, k₄.1.trans hk₃, h₂⟩

/-- `oodIntro`: an out-of-domain point (round `t.k`'s `dim` coins), its value and intro message from the proof, and the
batching coin (round `t.k + 1`); the claim and the polynomial gain the intro scaled by the coin. -/
theorem oodIntro_bind {p : LigeritoProof} {dim split : ℕ} (hdim : dim ≠ 0) {f : PUnit → LM β} {st st' : LState}
    {t t' : Transcript} {b : β} (h : (oodIntro p dim split >>= f) st t = .ok ((b, st'), t')) :
    (t.rounds[t.k]!).n = dim ∧ (t.rounds[t.k + 1]!).n = 1 ∧ ∃ y m t₁, p.oodValues[st.ood]? = some y ∧
      p.messages[st.msg]? = some m ∧ t₁.rounds = t.rounds ∧ t₁.k = t.k + 2 ∧
      f () { st with s := st.s + (F256.ofF y).smul ((t.rounds[t.k + 1]!).coins.getD 0 0),
                     q := st.q.addScaled (Poly.ofMsg m (F256.ofF y)) ((t.rounds[t.k + 1]!).coins.getD 0 0),
                     msg := st.msg + 1, ood := st.ood + 1, cbn := st.cbn + 1,
                     oods := st.oods.push ⟨(t.rounds[t.k]!).coins, (t.rounds[t.k + 1]!).coins.getD 0 0, some split⟩ }
        t₁ = .ok ((b, st'), t') := by
  obtain ⟨u, st₁, t₁, h₁, h₂⟩ := lbind_ok h
  unfold oodIntro at h₁
  obtain ⟨z, t₃, h₃, h₁⟩ := lift_bind h₁
  obtain ⟨cs, t₃', h₃', h₃''⟩ := bind_ok (show (sampleVec dim >>= pure) t = _ by rw [bind_pure]; exact h₃)
  obtain ⟨-, hn₃, hz, hr₃, hk₃⟩ := squeeze_ok hdim h₃'
  obtain ⟨rfl, rfl⟩ := pure_ok h₃''
  obtain ⟨y, hy, h₁⟩ := nextOod_bind h₁
  obtain ⟨_, t₄, h₄, h₁⟩ := lift_bind h₁
  obtain ⟨t₄', k₄, h₄'⟩ := absorbF_bind (f := fun _ => pure ()) (by rw [bind_pure]; exact h₄)
  obtain ⟨rfl, rfl⟩ := pure_ok h₄'
  obtain ⟨m, t₅, hm, k₅, h₁⟩ := nextMsg_bind h₁
  obtain ⟨t₆, k₆, h₁⟩ := claimPow_bind h₁
  obtain ⟨bc, t₇, h₇, h₁⟩ := lift_bind h₁
  obtain ⟨-, hn₇, hbc, hr₇, hk₇⟩ := sample_ok h₇
  obtain ⟨rfl, rfl⟩ := lmodify_ok h₁
  have er : t₆.rounds = t.rounds := by rw [k₆.2, k₅.2, k₄.2, hr₃]
  have ek : t₆.k = t.k + 1 := by rw [k₆.1, k₅.1, k₄.1, hk₃]
  subst hz hbc
  refine ⟨hn₃, by rw [← er, ← ek]; exact hn₇, y, m, _, hy, hm, hr₇.trans er, by rw [hk₇, ek], ?_⟩
  rw [er, ek] at h₂
  exact h₂

theorem range_size (n : ℕ) : [:n].size = n := by simp [Std.Legacy.Range.size]

/-- The consistency value `e = Σ_j eq(α)[j]·Σ_l eq_K(fold)[l]·row_j[l]` (§13.3 step 10), as the executable folds it. -/
def consE (rows : Array (Array F128)) (q : ℕ) (alpha : Array F128) (fold : Array F256) : F256 :=
  (List.range q).foldl (fun e j =>
    e + F256.smul ((List.range rows[j]!.size).foldl
      (fun acc l => acc + F256.smul (eqTableK fold)[l]! rows[j]![l]!) 0) (eqTable alpha)[j]!) 0

/-- `queryLevel`: the query coins (round `t.k`, `q` coins) give the positions, the consistency coins (round `t.k + 1`)
the batching vector; the Merkle check passes, and the consistency value is `consE`. -/
theorem queryLevel_bind {ms : MerkleScheme} {p : LigeritoProof} {lv : Flock.Level} {cap : Array ByteArray}
    {rows : Array (Array F128)} {sibs salts : Array ByteArray} {fold : Array F256} (hq : lv.q ≠ 0)
    {f : Array ℕ × Array F128 × F256 → LM β} {st st' : LState} {t t' : Transcript} {b : β}
    (h : (queryLevel ms p lv cap rows sibs salts fold >>= f) st t = .ok ((b, st'), t')) :
    (t.rounds[t.k]!).n = lv.q ∧ (t.rounds[t.k + 1]!).n = (lv.q - 1).log2 + 1 ∧
      merkleCheck ms cap lv.cap lv.d (positions lv (t.rounds[t.k]!).coins) rows sibs salts (2 ^ lv.lanes) = .ok () ∧
      ∃ t₁, t₁.rounds = t.rounds ∧ t₁.k = t.k + 2 ∧
        f (positions lv (t.rounds[t.k]!).coins, (t.rounds[t.k + 1]!).coins,
            consE rows lv.q (t.rounds[t.k + 1]!).coins fold)
          { st with qn := st.qn + 1, cn := st.cn + 1 } t₁ = .ok ((b, st'), t') := by
  obtain ⟨r, st₁, t₁, h₁, h₂⟩ := lbind_ok h
  unfold queryLevel at h₁
  have h₁ := lget_bind h₁
  obtain ⟨qn, _, h₁⟩ := nonceFrom_bind h₁
  obtain ⟨cn, _, h₁⟩ := nonceFrom_bind h₁
  have h₁ := lset_bind h₁
  obtain ⟨_, t₃, h₃, h₁⟩ := lift_bind h₁
  obtain ⟨t₃', k₃, h₃'⟩ := pow_bind (f := fun _ => pure ()) (by rw [bind_pure]; exact h₃)
  obtain ⟨rfl, rfl⟩ := pure_ok h₃'
  obtain ⟨coins, t₄, h₄, h₁⟩ := lift_bind h₁
  obtain ⟨cs₄, t₄', h₄', h₄''⟩ := bind_ok (show (sampleVec lv.q >>= pure) t₃ = _ by rw [bind_pure]; exact h₄)
  obtain ⟨-, hn₄, hc₄, hr₄, hk₄⟩ := squeeze_ok hq h₄'
  obtain ⟨rfl, rfl⟩ := pure_ok h₄''
  obtain ⟨_, t₅, h₅, h₁⟩ := lift_bind h₁
  obtain ⟨t₅', k₅, h₅'⟩ := pow_bind (f := fun _ => pure ()) (by rw [bind_pure]; exact h₅)
  obtain ⟨rfl, rfl⟩ := pure_ok h₅'
  obtain ⟨alpha, t₆, h₆, h₁⟩ := lift_bind h₁
  obtain ⟨cs₆, t₆', h₆', h₆''⟩ := bind_ok (show (sampleVec ((lv.q - 1).log2 + 1) >>= pure) t₅ = _ by
    rw [bind_pure]; exact h₆)
  obtain ⟨-, hn₆, hc₆, hr₆, hk₆⟩ := squeeze_ok (by omega) h₆'
  obtain ⟨rfl, rfl⟩ := pure_ok h₆''
  have er₄ : t₅.rounds = t.rounds := by rw [k₅.2, hr₄, k₃.2]
  have ek₄ : t₅.k = t.k + 1 := by rw [k₅.1, hk₄, k₃.1]
  have er₃ : t₃.rounds = t.rounds := k₃.2
  have ek₃ : t₃.k = t.k := k₃.1
  subst hc₄ hc₆
  cases hmc : merkleCheck ms cap lv.cap lv.d (positions lv (t₃.rounds[t₃.k]!).coins) rows sibs salts (2 ^ lv.lanes) with
  | error e => simp only [hmc] at h₁; obtain ⟨_, _, _, h₀, _⟩ := lbind_ok h₁; cases h₀
  | ok u =>
    simp only [hmc] at h₁
    simp only [List.forIn_pure_yield_eq_foldl, Std.Legacy.Range.forIn_eq_forIn_range', pure_bind] at h₁
    obtain ⟨rfl, rfl, rfl⟩ := lpure_ok h₁
    simp only [range_size, ← List.range_eq_range'] at h₂
    refine ⟨by rw [← er₃, ← ek₃]; exact hn₄, by rw [← er₄, ← ek₄]; exact hn₆,
      by rw [← er₃, ← ek₃]; exact hmc, _, hr₆.trans er₄, by rw [hk₆, ek₄], ?_⟩
    rw [er₃, ek₃, er₄, ek₄] at h₂
    exact h₂

end FlockSoundness.Refine
