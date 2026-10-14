import Proofs.Flock.Soundness.Discharge.ZkExec.PlaysStatements

/-!
# The masked replay's runs

`zk_replay_plays`: `Zk.replay` refines the model's masked zerocheck and lincheck, with the two masked claims' points.
The executable draws exactly the clear verifier's coins (§1, §4); its messages are the masked proof's, and its checks
on them are constraints, so nothing it does after a coin depends on whether a check holds.

* `forIn_push`: a loop that consumes one one-coin round per step and pushes that coin keeps the coins, in order.
* `residualZK_plays`, `lcRoundsZK_plays`: the model's masked residual and lincheck rounds, fed round `i`'s messages and
  coins, end with those coins and the claim `resOut`, `lcRoundsOut` computes.

The replay's claims are `abcClaims` on the coins: `ab`'s point is the lincheck's (its rounds' coins reversed, then the
zerocheck's residual coins), `c`'s is `(L_s(z), pinned ++ r_rest)`, which `zcVector_eq` reads as the model's `r`.
-/

namespace FlockSoundness.Discharge.ZkExec

open Flock Refine Exec Game Model DecodeZL Transcript

local notation "𝔽" => FlockLevel3.GF128

/-- **A loop that pushes its rounds' coins.** If every step consumes the next round, which issued one coin, and pushes
that coin onto the state's array, the loop over `n` steps consumes `n` rounds and pushes their coins, in order. -/
theorem forIn_push {σ : Type} (body : ℕ → σ × Array F128 → TM (ForInStep (σ × Array F128)))
    (hbody : ∀ i s t b t', body i s t = .ok (b, t') →
      (t.rounds[t.k]!).n = 1 ∧ t'.rounds = t.rounds ∧ t'.k = t.k + 1 ∧
        ∃ x, b = .yield (x, s.2.push ((t.rounds[t.k]!).coins.getD 0 0))) :
    ∀ (n a : ℕ) (s : σ × Array F128) (t t' : Flock.Transcript) (out : σ × Array F128),
      (forIn (List.range' a n) s body : TM (σ × Array F128)) t = .ok (out, t') →
      t'.rounds = t.rounds ∧ t'.k = t.k + n ∧ (∀ i < n, (t.rounds[t.k + i]!).n = 1) ∧
        out.2 = s.2 ++ ((List.range n).map fun i => (t.rounds[t.k + i]!).coins.getD 0 0).toArray
  | 0, a, s, t, t', out, h => by
      simp only [List.range'_zero, List.forIn_nil] at h
      obtain ⟨rfl, rfl⟩ := pure_ok h
      simp
  | n + 1, a, s, t, t', out, h => by
      rw [forIn_range'_cons] at h
      obtain ⟨b, t₁, h₁, h₂⟩ := bind_ok h
      obtain ⟨hn, hr, hk, x, rfl⟩ := hbody a s t b t₁ h₁
      obtain ⟨hr', hk', hn', ho⟩ := forIn_push body hbody n (a + 1) _ t₁ t' out h₂
      rw [hr, hk] at hn' ho
      refine ⟨hr'.trans hr, by omega, fun i hi => ?_, ?_⟩
      · rcases i with _ | i
        · simpa using hn
        · rw [show t.k + (i + 1) = t.k + 1 + i by omega]; exact hn' i (by omega)
      · rw [ho, List.range_succ_eq_map]
        simp [Nat.add_assoc, Nat.add_comm 1, Function.comp_def]

/-- The pushed coins, read back: the array has one per step, step `i`'s at `i`. -/
theorem push_get (a : Array F128) (n : ℕ) (f : ℕ → F128) (h : a = #[] ++ ((List.range n).map f).toArray) :
    a.size = n ∧ ∀ i < n, a[i]! = f i := by
  subst h
  refine ⟨by simp, fun i hi => ?_⟩
  rw [getElem!_pos _ _ (by simp; omega)]
  simp

/-- The `c` claim (stated so `simp` reaches it without unfolding `skipWeights`). -/
theorem abcClaims_one (m kLog : ℕ) (zc : Flock.ZcOut) (lc : Flock.LcOut) :
    (abcClaims m kLog zc lc)[1]! = { value := zc.vC, skip := skipWeights zc.z, x := zc.rRest } := rfl

section Rounds

variable {np : ℕ} (P : ZkSession.PadsA np) (rounds : Array (F128 × F128)) (C : ℕ → Array F128)

/-- The claim the masked residual rounds end with, fed round `a + i`'s messages and coins at step `i`. -/
noncomputable def resOut : (n : ℕ) → ℕ → ℕ → (Fin n → 𝔽) → ZkSession.LinF 𝔽 np → ZkSession.LinF 𝔽 np
  | 0, _, _, _, cl => cl
  | n + 1, a, j, req, cl => resOut n (a + 1) (j + 1) (Fin.tail req)
      (ZkSession.residualStepL (req 0) (fOf (C a)[0]!) cl (ZkSession.mskF (P.ml j 0) (pairOf rounds[a]!).1)
        (ZkSession.mskF (P.ml j 1) (pairOf rounds[a]!).2))

/-- **The masked residual rounds.** Fed round `a + i`'s messages and coins at step `i`, they end with those coins and
the claim `resOut`. -/
theorem residualZK_plays : ∀ (n a j : ℕ) (req : Fin n → 𝔽) (cl : ZkSession.LinF 𝔽 np),
    (∀ i ∈ List.range' a n, (C i).size = 1) →
    Plays fOf (ZkSession.residualRoundsZK P n j req cl)
      ((List.range' a n).map fun i => ⟨𝔽 × 𝔽, pairOf rounds[i]!⟩) ((List.range' a n).map C)
      (fun k => fOf (C (a + k.val))[0]!, resOut P rounds C n a j req cl)
  | 0, a, j, req, cl, _ => by
      simp only [List.range'_zero, List.map_nil, resOut]
      rw [ZkSession.residualRoundsZK]
      have e : (Fin.elim0 : Fin 0 → 𝔽) = fun k => fOf (C (a + k.val))[0]! := funext fun k => k.elim0
      rw [← e]
      exact Plays.pure _
  | n + 1, a, j, req, cl, hC => by
      rw [ZkSession.residualRoundsZK]
      have hCa : (C a).size = 1 := hC a (by rw [List.range'_succ]; exact List.mem_cons_self)
      have hih := residualZK_plays n (a + 1) (j + 1) (Fin.tail req)
        (ZkSession.residualStepL (req 0) (fOf (C a)[0]!) cl (ZkSession.mskF (P.ml j 0) (pairOf rounds[a]!).1)
          (ZkSession.mskF (P.ml j 1) (pairOf rounds[a]!).2))
        (fun i hi => hC i (by rw [List.range'_succ]; exact List.mem_cons_of_mem _ hi))
      refine Plays.bind' (ms₂ := (List.range' (a + 1) n).map fun i => ⟨𝔽 × 𝔽, pairOf rounds[i]!⟩)
        (cs₂ := (List.range' a (n + 1)).map C) (Plays.recv _ (pairOf rounds[a]!)) ?_ (by simp [List.range'_succ])
        (by simp)
      refine Plays.bind' (ms₂ := (List.range' (a + 1) n).map fun i => ⟨𝔽 × 𝔽, pairOf rounds[i]!⟩)
        (cs₂ := (List.range' (a + 1) n).map C) (Plays.drawOne (C a) hCa) ?_ (by simp) (by simp [List.range'_succ])
      refine Plays.bind' (ms₂ := []) (cs₂ := []) hih ?_ (by simp) (by simp)
      convert Plays.pure (ι := fOf) _ using 2
      · funext k
        refine Fin.cases ?_ (fun k => ?_) k
        · simp
        · simp [Nat.add_assoc, Nat.add_comm 1]
      · simp [resOut]

/-- The claim the masked lincheck rounds end with, fed round `a + i`'s messages and coins at step `i`. -/
noncomputable def lcRoundsOut : ℕ → ℕ → ℕ → ZkSession.LinF 𝔽 np → ZkSession.LinF 𝔽 np
  | 0, _, _, cl => cl
  | n + 1, a, j, cl =>
      let q₁ := ZkSession.mskF (P.lc j 0) (pairOf rounds[a]!).1
      let q₂ := ZkSession.mskF (P.lc j 1) (pairOf rounds[a]!).2
      let t := fOf (C a)[0]!
      let e₀ := cl + q₁
      lcRoundsOut n (a + 1) (j + 1) ((t ^ 2) • q₂ + t • (e₀ + q₁ + q₂) + e₀)

/-- **The masked lincheck rounds.** Fed round `a + i`'s messages and coins at step `i`, they end with those coins and
the claim `lcRoundsOut`. -/
theorem lcRoundsZK_plays : ∀ (n a j : ℕ) (cl : ZkSession.LinF 𝔽 np),
    (∀ i ∈ List.range' a n, (C i).size = 1) →
    Plays fOf (ZkSession.lincheckRoundsZK P n j cl)
      ((List.range' a n).map fun i => ⟨𝔽 × 𝔽, pairOf rounds[i]!⟩) ((List.range' a n).map C)
      (fun k => fOf (C (a + k.val))[0]!, lcRoundsOut P rounds C n a j cl)
  | 0, a, j, cl, _ => by
      simp only [List.range'_zero, List.map_nil, lcRoundsOut]
      rw [ZkSession.lincheckRoundsZK]
      have e : (Fin.elim0 : Fin 0 → 𝔽) = fun k => fOf (C (a + k.val))[0]! := funext fun k => k.elim0
      rw [← e]
      exact Plays.pure _
  | n + 1, a, j, cl, hC => by
      rw [ZkSession.lincheckRoundsZK]
      have hCa : (C a).size = 1 := hC a (by rw [List.range'_succ]; exact List.mem_cons_self)
      have hih := lcRoundsZK_plays n (a + 1) (j + 1)
        ((fOf (C a)[0]! ^ 2) • ZkSession.mskF (P.lc j 1) (pairOf rounds[a]!).2 +
          fOf (C a)[0]! • (cl + ZkSession.mskF (P.lc j 0) (pairOf rounds[a]!).1 +
            ZkSession.mskF (P.lc j 0) (pairOf rounds[a]!).1 + ZkSession.mskF (P.lc j 1) (pairOf rounds[a]!).2) +
          (cl + ZkSession.mskF (P.lc j 0) (pairOf rounds[a]!).1))
        (fun i hi => hC i (by rw [List.range'_succ]; exact List.mem_cons_of_mem _ hi))
      refine Plays.bind' (ms₂ := (List.range' (a + 1) n).map fun i => ⟨𝔽 × 𝔽, pairOf rounds[i]!⟩)
        (cs₂ := (List.range' a (n + 1)).map C) (Plays.recv _ (pairOf rounds[a]!)) ?_ (by simp [List.range'_succ])
        (by simp)
      refine Plays.bind' (ms₂ := (List.range' (a + 1) n).map fun i => ⟨𝔽 × 𝔽, pairOf rounds[i]!⟩)
        (cs₂ := (List.range' (a + 1) n).map C) (Plays.drawOne (C a) hCa) ?_ (by simp) (by simp [List.range'_succ])
      refine Plays.bind' (ms₂ := []) (cs₂ := []) hih ?_ (by simp) (by simp)
      convert Plays.pure (ι := fOf) _ using 2
      · funext k
        refine Fin.cases ?_ (fun k => ?_) k
        · simp
        · simp [Nat.add_assoc, Nat.add_comm 1]
      · simp [lcRoundsOut]

end Rounds

theorem zk_replay_plays : ZkReplayPlays := by
  intro st lay p root t t' rp S hm hk h15 hwf h
  have hk6 : 6 ≤ S.kLog := S.hk6
  have hkm : S.kLog ≤ S.m := S.hk
  unfold Zk.replay at h
  dsimp only at h
  rw [hm, hk] at h
  obtain ⟨t₁, k₁, h⟩ := label_bind h
  obtain ⟨t₂, k₂, h⟩ := absorbBytes_bind h
  obtain ⟨t₃, k₃, h⟩ := absorbBytes_bind h
  obtain ⟨t₄, k₄, h⟩ := label_bind h
  obtain ⟨t₅, k₅, h⟩ := label_bind h
  obtain ⟨t₆, k₆, h⟩ := absorbBytes_bind h
  obtain ⟨lt₆, n₆, t₇, r₇, kk₇, h⟩ := sampleVec_bind (by decide) h
  obtain ⟨lt₇, n₇, t₈, r₈, kk₈, h⟩ := sampleVec_bind (by omega) h
  obtain ⟨t₉, k₉, h⟩ := absorbSlice_bind h
  obtain ⟨t₁₀, k₁₀, h⟩ := absorbSlice_bind h
  obtain ⟨lt₁₀, n₁₀, t₁₁, r₁₁, kk₁₁, h⟩ := sample_bind h
  obtain ⟨s, t₁₂, hloop, h⟩ := bind_ok h
  obtain ⟨t₁₃, k₁₃, h⟩ := absorbF_bind h
  obtain ⟨t₁₄, k₁₄, h⟩ := absorbF_bind h
  obtain ⟨t₁₅, k₁₅, h⟩ := label_bind h
  obtain ⟨lt₁₅, n₁₅, t₁₆, r₁₆, kk₁₆, h⟩ := sample_bind h
  obtain ⟨lt₁₆, n₁₆, t₁₇, r₁₇, kk₁₇, h⟩ := sample_bind h
  obtain ⟨s', t₁₈, hloop', h⟩ := bind_ok h
  rcases hf : st.fold.folded ((t₁₅.rounds[t₁₅.k]!).coins.getD 0 0) ((t₁₆.rounds[t₁₆.k]!).coins.getD 0 0)
      (skipWeights ((t₁₀.rounds[t₁₀.k]!).coins.getD 0 0)) (s.2.extract 0 (S.kLog - 6)) s'.2 with err | v
  · rw [hf] at h
    obtain ⟨_, _, h₀, _⟩ := bind_ok h
    cases h₀
  rw [hf] at h
  simp only [pure_bind] at h
  obtain ⟨-, h⟩ := check_bind h
  obtain ⟨t₁₉, k₁₉, h⟩ := absorbSlice_bind h
  obtain ⟨lt₁₉, n₁₉, t₂₀, r₂₀, kk₂₀, h⟩ := sample_bind h
  obtain ⟨hrp, ht'⟩ := pure_ok h
  subst hrp
  -- the loops: one one-coin round per step, its coin pushed
  rw [forIn_range_eq] at hloop hloop'
  obtain ⟨hr₁₂, hk₁₂, hn₁₂, hs⟩ := forIn_push _ (fun i st' tt b tt' hb => by
      obtain ⟨tt₁, kk₁, hb⟩ := absorbF_bind hb
      obtain ⟨tt₂, kk₂, hb⟩ := absorbF_bind hb
      obtain ⟨-, nn, tt₃, rr₃, kk₃, hb⟩ := sample_bind hb
      obtain ⟨rfl, rfl⟩ := pure_ok hb
      have e1 : tt₂.rounds = tt.rounds := by rw [kk₂.2, kk₁.2]
      have e2 : tt₂.k = tt.k := by rw [kk₂.1, kk₁.1]
      refine ⟨by rw [← e1, ← e2]; exact nn, by rw [rr₃, e1], by rw [kk₃, e2], ?_⟩
      rw [← e1, ← e2]
      exact ⟨_, rfl⟩) (S.m - 6) 0 _ t₁₁ t₁₂ s hloop
  obtain ⟨hr₁₈, hk₁₈, hn₁₈, hs'⟩ := forIn_push _ (fun i st' tt b tt' hb => by
      obtain ⟨tt₁, kk₁, hb⟩ := absorbF_bind hb
      obtain ⟨tt₂, kk₂, hb⟩ := absorbF_bind hb
      obtain ⟨-, nn, tt₃, rr₃, kk₃, hb⟩ := sample_bind hb
      obtain ⟨rfl, rfl⟩ := pure_ok hb
      have e1 : tt₂.rounds = tt.rounds := by rw [kk₂.2, kk₁.2]
      have e2 : tt₂.k = tt.k := by rw [kk₂.1, kk₁.1]
      refine ⟨by rw [← e1, ← e2]; exact nn, by rw [rr₃, e1], by rw [kk₃, e2], ?_⟩
      rw [← e1, ← e2]
      exact ⟨_, rfl⟩) (S.kLog - 6) 0 _ t₁₇ t₁₈ s' hloop'
  -- the rounds and the counter, in terms of `t`
  have hr₆ : t₆.rounds = t.rounds := by rw [k₆.2, k₅.2, k₄.2, k₃.2, k₂.2, k₁.2]
  have hk₆' : t₆.k = t.k := by rw [k₆.1, k₅.1, k₄.1, k₃.1, k₂.1, k₁.1]
  have hr₇ : t₇.rounds = t.rounds := by rw [r₇, hr₆]
  have hk₇ : t₇.k = t.k + 1 := by rw [kk₇, hk₆']
  have hr₁₀ : t₁₀.rounds = t.rounds := by rw [k₁₀.2, k₉.2, r₈, hr₇]
  have hk₁₀ : t₁₀.k = t.k + 2 := by rw [k₁₀.1, k₉.1, kk₈, hk₇]
  have hr₁₁ : t₁₁.rounds = t.rounds := by rw [r₁₁, hr₁₀]
  have hk₁₁ : t₁₁.k = t.k + 3 := by rw [kk₁₁, hk₁₀]
  have hr₁₅ : t₁₅.rounds = t.rounds := by rw [k₁₅.2, k₁₄.2, k₁₃.2, hr₁₂, hr₁₁]
  have hk₁₅ : t₁₅.k = t.k + 3 + (S.m - 6) := by rw [k₁₅.1, k₁₄.1, k₁₃.1, hk₁₂, hk₁₁]
  have hr₁₆ : t₁₆.rounds = t.rounds := by rw [r₁₆, hr₁₅]
  have hk₁₆ : t₁₆.k = t.k + 3 + (S.m - 6) + 1 := by rw [kk₁₆, hk₁₅]
  have hr₁₇ : t₁₇.rounds = t.rounds := by rw [r₁₇, hr₁₆]
  have hk₁₇ : t₁₇.k = t.k + 3 + (S.m - 6) + 2 := by rw [kk₁₇, hk₁₆]
  have hr₁₉ : t₁₉.rounds = t.rounds := by rw [k₁₉.2, hr₁₈, hr₁₇]
  have hk₁₉ : t₁₉.k = t.k + 3 + (S.m - 6) + 2 + (S.kLog - 6) := by rw [k₁₉.1, hk₁₈, hk₁₇]
  have hrt : t'.rounds = t.rounds := by rw [ht', r₂₀, hr₁₉]
  have hkt : t'.k = t.k + 3 + (S.m - 6) + 2 + (S.kLog - 6) + 1 := by rw [ht', kk₂₀, hk₁₉]
  set c₀ := (t₆.rounds[t₆.k]!).coins with hc₀
  set c₁ := (t₇.rounds[t₇.k]!).coins with hc₁
  set c₂ := (t₁₀.rounds[t₁₀.k]!).coins with hc₂
  set C : ℕ → Array F128 := fun i => (t₁₁.rounds[t₁₁.k + i]!).coins with hC
  set cα := (t₁₅.rounds[t₁₅.k]!).coins with hcα
  set cβ := (t₁₆.rounds[t₁₆.k]!).coins with hcβ
  set C' : ℕ → Array F128 := fun i => (t₁₇.rounds[t₁₇.k + i]!).coins with hC'
  set cr := (t₁₉.rounds[t₁₉.k]!).coins with hcr
  have hsz₀ : c₀.size = 6 := by rw [hc₀, RoundsWF.get (hr₆ ▸ hwf), n₆]
  have hsz₁ : c₁.size = S.m - 13 := by rw [hc₁, RoundsWF.get (hr₇ ▸ hwf), n₇]
  have hsz₂ : c₂.size = 1 := by rw [hc₂, RoundsWF.get (hr₁₀ ▸ hwf), n₁₀]
  have hszα : cα.size = 1 := by rw [hcα, RoundsWF.get (hr₁₅ ▸ hwf), n₁₅]
  have hszβ : cβ.size = 1 := by rw [hcβ, RoundsWF.get (hr₁₆ ▸ hwf), n₁₆]
  have hszr : cr.size = 1 := by rw [hcr, RoundsWF.get (hr₁₉ ▸ hwf), n₁₉]
  have hCsz : ∀ i ∈ List.range' 0 (S.m - 6), (C i).size = 1 := fun i hi => by
    rw [hC]; simp only; rw [RoundsWF.get (hr₁₁ ▸ hwf), hn₁₂ i (by simpa using hi)]
  have hC'sz : ∀ i ∈ List.range' 0 (S.kLog - 6), (C' i).size = 1 := fun i hi => by
    rw [hC']; simp only; rw [RoundsWF.get (hr₁₇ ▸ hwf), hn₁₈ i (by simpa using hi)]
  -- the coins consumed, split at the lincheck's first
  have hcoins₁ : consumed t.rounds t.k (t.k + 3 + (S.m - 6)) = c₀ :: c₁ :: c₂ :: (List.range' 0 (S.m - 6)).map C := by
    rw [consumed_append t.rounds (show t.k ≤ t.k + 1 by omega) (by omega),
      consumed_append t.rounds (show t.k + 1 ≤ t.k + 2 by omega) (by omega),
      consumed_append t.rounds (show t.k + 2 ≤ t.k + 3 by omega) (by omega), consumed_succ, consumed_succ,
      consumed_succ, consumed_eq_map, hc₀, hc₁, hc₂, hC, hr₆, hk₆', hr₇, hk₇, hr₁₀, hk₁₀, hr₁₁, hk₁₁]
    rfl
  have hcoins₂ : consumed t.rounds (t.k + 3 + (S.m - 6)) t'.k =
      cα :: cβ :: ((List.range' 0 (S.kLog - 6)).map C' ++ [cr]) := by
    rw [hkt, consumed_append t.rounds (show t.k + 3 + (S.m - 6) ≤ t.k + 3 + (S.m - 6) + 1 by omega) (by omega),
      consumed_append t.rounds (show t.k + 3 + (S.m - 6) + 1 ≤ t.k + 3 + (S.m - 6) + 2 by omega) (by omega),
      consumed_append t.rounds (show t.k + 3 + (S.m - 6) + 2 ≤ t.k + 3 + (S.m - 6) + 2 + (S.kLog - 6) by omega)
        (by omega), consumed_succ, consumed_succ, consumed_succ, consumed_eq_map, hcα, hcβ, hcr, hC', hr₁₅, hk₁₅,
      hr₁₆, hk₁₆, hr₁₇, hk₁₇, hr₁₉, hk₁₉]
    simp
  -- the coins the loops pushed
  obtain ⟨hssz, hsget⟩ := push_get s.2 (S.m - 6) _ hs
  obtain ⟨hs'sz, hs'get⟩ := push_get s'.2 (S.kLog - 6) _ hs'
  have hzM : fOf c₂[0]! = fOf (c₂.getD 0 0) := by rw [getElem!_eq_getD]
  have hrM : fOf cr[0]! = fOf (cr.getD 0 0) := by rw [getElem!_eq_getD]
  refine ⟨hrt, by omega, ?_⟩
  intro np P tr
  refine ⟨t.k + 3 + (S.m - 6), ?zc, ?row, ?ab, ?row', by omega, by omega, ?pz, ?pl, ?c0, ?c1⟩
  case pz =>
    rw [hcoins₁]
    unfold ZkSession.zerocheckZK
    refine Plays.bind' (ms₂ := zcMsgs S.m p.zerocheck) (cs₂ := c₁ :: c₂ :: (List.range' 0 (S.m - 6)).map C)
      (Plays.drawVec c₀ hsz₀ (by decide)) ?_ (by simp) (by simp)
    refine Plays.bind' (ms₂ := zcMsgs S.m p.zerocheck) (cs₂ := c₂ :: (List.range' 0 (S.m - 6)).map C)
      (Plays.drawVec c₁ hsz₁ (by omega)) ?_ (by simp) (by simp)
    refine Plays.bind' (ms₂ := ((List.range (S.m - 6)).map fun i => ⟨𝔽 × 𝔽, pairOf p.zerocheck.rounds[i]!⟩) ++
      [⟨𝔽 × 𝔽, (fOf p.zerocheck.finalA, fOf p.zerocheck.finalB)⟩]) (cs₂ := c₂ :: (List.range' 0 (S.m - 6)).map C)
      (Plays.recv _ (vecOf 64 p.zerocheck.round1ab, vecOf 64 p.zerocheck.round1c)) ?_ (by simp [zcMsgs]) (by simp)
    refine Plays.bind' (ms₂ := ((List.range (S.m - 6)).map fun i => ⟨𝔽 × 𝔽, pairOf p.zerocheck.rounds[i]!⟩) ++
      [⟨𝔽 × 𝔽, (fOf p.zerocheck.finalA, fOf p.zerocheck.finalB)⟩]) (cs₂ := (List.range' 0 (S.m - 6)).map C)
      (Plays.drawOne c₂ hsz₂) ?_ (by simp) (by simp)
    refine Plays.bind' (ms₂ := [⟨𝔽 × 𝔽, (fOf p.zerocheck.finalA, fOf p.zerocheck.finalB)⟩]) (cs₂ := [])
      (residualZK_plays P p.zerocheck.rounds C (S.m - 6) 0 0 _ _ hCsz) ?_ (by simp [List.range_eq_range']) (by simp)
    refine Plays.bind' (ms₂ := []) (cs₂ := []) (Plays.recv _ (fOf p.zerocheck.finalA, fOf p.zerocheck.finalB)) ?_
      (by simp) (by simp)
    exact Plays.pure (ι := fOf) _
  case pl =>
    rw [hcoins₂]
    unfold ZkSession.lincheckZK
    refine Plays.bind' (ms₂ := lcMsgs S.kLog p.lincheck)
      (cs₂ := cβ :: ((List.range' 0 (S.kLog - 6)).map C' ++ [cr])) (Plays.drawOne cα hszα) ?_ (by simp) (by simp)
    refine Plays.bind' (ms₂ := lcMsgs S.kLog p.lincheck) (cs₂ := (List.range' 0 (S.kLog - 6)).map C' ++ [cr])
      (Plays.drawOne cβ hszβ) ?_ (by simp) (by simp)
    refine Plays.bind' (ms₂ := [⟨Fin 64 → 𝔽, vecOf 64 p.lincheck.zPartial⟩]) (cs₂ := [cr])
      (lcRoundsZK_plays P p.lincheck.rounds C' (S.kLog - 6) 0 0 _ hC'sz) ?_
      (by simp [lcMsgs, List.range_eq_range']) (by simp)
    refine Plays.bind' (ms₂ := []) (cs₂ := [cr]) (Plays.recv _ (vecOf 64 p.lincheck.zPartial)) ?_ (by simp) (by simp)
    refine Plays.bind' (ms₂ := []) (cs₂ := []) (Plays.drawOne cr hszr) ?_ (by simp) (by simp)
    exact Plays.pure (ι := fOf) _
  case c0 =>
    intro w
    show ClaimRel S.m (⟨w, skipWeights (cr.getD 0 0), s'.2.reverse ++ s.2.extract (S.kLog - 6) (S.m - 6)⟩ :
      Flock.Claim) _
    have hxs : (s'.2.reverse ++ s.2.extract (S.kLog - 6) (S.m - 6)).size = S.m - 6 := by
      simp [hssz, hs'sz]; omega
    refine ⟨rfl, skipWeights_size _, fun _ => by simp only [hrM]; rfl, hxs, fun i => ?_⟩
    simp only
    rw [getElem!_pos _ i.val (by rw [hxs]; exact i.isLt), Array.getElem_append]
    by_cases hi : i.val < S.kLog - 6
    · have hi' : i.val < s'.2.reverse.size := by simp [hs'sz, hi]
      rw [dite_eq_left hi, dite_eq_left hi', Array.getElem_reverse, ← getElem!_pos s'.2 _ (by omega),
        hs'get _ (by omega), Nat.zero_add, getElem!_eq_getD]
      congr 3
      simp [hs'sz]
    · have hi' : ¬ i.val < s'.2.reverse.size := by simp [hs'sz, hi]
      have hidx : S.kLog - 6 + (i.val - s'.2.reverse.size) = i.val := by simp [hs'sz]; omega
      rw [dite_eq_right hi, dite_eq_right hi', Array.getElem_extract,
        ← getElem!_pos s.2 _ (by simp [hs'sz] at *; omega), hidx, hsget _ i.isLt, Nat.zero_add, getElem!_eq_getD]
  case c1 =>
    intro w
    simp only [abcClaims_one]
    exact ⟨rfl, skipWeights_size _, fun _ => by rw [hzM]; rfl,
      by rw [Array.size_append, FlockLevel3.pinned_size, hsz₁]; omega, fun i => zcVector_eq c₁ hsz₁ (by omega) i⟩

end FlockSoundness.Discharge.ZkExec
