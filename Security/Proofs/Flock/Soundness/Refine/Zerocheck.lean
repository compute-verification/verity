import Proofs.Flock.Soundness.Refine.Arith

/-!
# The zerocheck refines the model's (PROTOCOL.md §9.1–§9.2)

`zerocheck_refines`: if the executable's `bindAndZerocheck` succeeds, the model's `zerocheck`, played with the
proof's zerocheck messages (`zcMsgs`) and the coins of the rounds the executable consumed, returns `some zc'`, and `zc'`
is the executable's output read in the model's field (`ZcRel`).

The residual rounds (Z6) are a loop in the executable and a recursion in the model. `zStep` is one round as the
executable computes it; `residual_plays` runs the model's rounds on the fold of `zStep`, and `forIn_rounds` shows the
executable's loop is that fold.
-/

namespace FlockSoundness.Refine

open Flock FlockLevel3 Game Transcript

local notation "𝔽" => FlockLevel3.GF128

/-- One residual round (Z6) as the executable computes it, on round `i`'s messages and coins `c`. -/
def zStep (r : Array F128) (rounds : Array (F128 × F128)) (i : ℕ) (s : F128 × Array F128) (c : Array F128) :
    F128 × Array F128 :=
  let g := rounds[i]!
  let req := r[i]!
  let g0 := (s.1 + req * g.1) * (1 + req).inv
  let rr := c.getD 0 0
  (g0 * (1 + rr) + g.1 * rr + g.2 * rr * (1 + rr), s.2.push rr)

theorem zStep_fold_size (r : Array F128) (rounds : Array (F128 × F128)) (C : ℕ → Array F128) :
    ∀ (l : List ℕ) (s : F128 × Array F128),
      ((l.foldl (fun s i => zStep r rounds i s (C i)) s).2.size = s.2.size + l.length) ∧
      ∀ j < s.2.size, (l.foldl (fun s i => zStep r rounds i s (C i)) s).2[j]! = s.2[j]! := by
  intro l
  induction l with
  | nil => intro s; simp
  | cons i l ih =>
      intro s
      obtain ⟨h1, h2⟩ := ih (zStep r rounds i s (C i))
      refine ⟨by rw [List.foldl_cons, h1]; simp [zStep]; omega, fun j hj => ?_⟩
      rw [List.foldl_cons, h2 j (by simp [zStep]; omega)]
      simp only [zStep]
      rw [getElem!_pos _ j (by simp; omega), getElem!_pos _ j hj, Array.getElem_push_lt]

/-- The model's `residualStep` on the words of `zStep`. -/
theorem residualStep_zStep (r : Array F128) (rounds : Array (F128 × F128)) (i : ℕ) (s : F128 × Array F128)
    (c : Array F128) :
    Model.residualStep (fOf r[i]!) (fOf s.1) (fOf rounds[i]!.1) (fOf rounds[i]!.2) (fOf c[0]!) =
      fOf (zStep r rounds i s c).1 := by
  simp only [Model.residualStep, zStep, getElem!_eq_getD c 0, fOf_add, fOf_mul, fOf_inv, fOf_one]

/-- **The residual rounds.** The model's `residualRounds`, fed round `i`'s messages `rounds[i]` and coins `C i`, ends
with the challenges and the claim of the fold of `zStep`. -/
theorem residual_plays (r : Array F128) (rounds : Array (F128 × F128)) (C : ℕ → Array F128) :
    ∀ (n a : ℕ) (s : F128 × Array F128), (∀ i ∈ List.range' a n, (C i).size = 1) →
      Plays fOf (Model.residualRounds n (fun j : Fin n => fOf r[a + j.val]!) (fOf s.1))
        ((List.range' a n).map fun i => ⟨𝔽 × 𝔽, pairOf rounds[i]!⟩) ((List.range' a n).map C)
        (fun j => fOf ((List.range' a n).foldl (fun s i => zStep r rounds i s (C i)) s).2[s.2.size + j.val]!,
          fOf ((List.range' a n).foldl (fun s i => zStep r rounds i s (C i)) s).1) := by
  intro n
  induction n with
  | zero =>
      intro a s _
      simp only [List.range'_zero, List.map_nil, List.foldl_nil]
      rw [Model.residualRounds]
      have e : (Fin.elim0 : Fin 0 → 𝔽) = fun j => fOf s.2[s.2.size + j.val]! := funext fun j => j.elim0
      rw [← e]
      exact Plays.pure _
  | succ n ih =>
      intro a s hC
      rw [Model.residualRounds]
      have hih := ih (a + 1) (zStep r rounds a s (C a))
        (fun i hi => hC i (by rw [List.range'_succ]; exact List.mem_cons_of_mem _ hi))
      have hCa : (C a).size = 1 := hC a (by rw [List.range'_succ]; exact List.mem_cons_self)
      have htail : (Fin.tail fun j : Fin (n + 1) => fOf r[a + j.val]!) = fun j : Fin n => fOf r[a + 1 + j.val]! := by
        funext j; simp only [Fin.tail, Fin.val_succ]; congr 2; omega
      have hstep : Model.residualStep (fOf r[a + (0 : Fin (n + 1)).val]!) (fOf s.1) (pairOf rounds[a]!).1
          (pairOf rounds[a]!).2 (fOf (C a)[0]!) = fOf (zStep r rounds a s (C a)).1 := by
        simpa [pairOf] using residualStep_zStep r rounds a s (C a)
      obtain ⟨hsz, hkeep⟩ := zStep_fold_size r rounds C (List.range' (a + 1) n) (zStep r rounds a s (C a))
      have hs1 : (zStep r rounds a s (C a)).2.size = s.2.size + 1 := by simp [zStep]
      refine Plays.bind' (ms₂ := (List.range' (a + 1) n).map fun i => ⟨𝔽 × 𝔽, pairOf rounds[i]!⟩)
        (cs₂ := (List.range' a (n + 1)).map C) (Plays.recv _ (pairOf rounds[a]!)) ?_ ?_ ?_
      · refine Plays.bind' (ms₂ := (List.range' (a + 1) n).map fun i => ⟨𝔽 × 𝔽, pairOf rounds[i]!⟩)
          (cs₂ := (List.range' (a + 1) n).map C) (Plays.drawOne (C a) hCa) ?_ ?_ ?_
        · rw [htail, hstep]
          refine Plays.bind' (ms₂ := []) (cs₂ := []) hih ?_ ?_ ?_
          · simp only [List.range'_succ, List.foldl_cons]
            convert Plays.pure (ι := fOf) _ using 2
            funext j
            refine Fin.cases ?_ (fun j => ?_) j
            · rw [Fin.cons_zero, Fin.val_zero, Nat.add_zero s.2.size, hkeep s.2.size (by omega)]
              simp only [zStep]
              rw [getElem!_pos _ s.2.size (by simp), Array.getElem_push_eq, getElem!_eq_getD]
            · rw [Fin.cons_succ, hs1, Fin.val_succ, show s.2.size + (j.val + 1) = s.2.size + 1 + j.val by omega]
          all_goals simp
        all_goals simp [List.range'_succ]
      all_goals simp [List.range'_succ]

theorem consumed_eq_map (rs : Array Round) (k n : ℕ) :
    consumed rs k (k + n) = (List.range' 0 n).map fun i => (rs[k + i]!).coins := by
  unfold consumed
  rw [Nat.add_sub_cancel_left, List.range'_eq_map_range, List.range'_eq_map_range, List.map_map, List.map_map]
  congr 1
  funext i
  simp

/-- **The zerocheck refines the model's** (PROTOCOL.md §9.1–§9.2). If the executable's statement binding and zerocheck
succeed from `t` to `t'`, the model's zerocheck, played with the proof's zerocheck messages and the coins of the rounds
consumed from `t` to `t'`, returns the executable's output, read in the model's field. -/
theorem zerocheck_refines (m : ℕ) (hm : 15 ≤ m) (digest cap : ByteArray) (p : ZerocheckProof)
    (t t' : Transcript) (zc : Flock.ZcOut) (hwf : RoundsWF t.rounds)
    (h : bindAndZerocheck m digest cap p t = .ok (zc, t')) :
    t'.rounds = t.rounds ∧ t.k ≤ t'.k ∧
      ∃ zc', Plays fOf (Model.zerocheck execArith m) (zcMsgs m p) (consumed t.rounds t.k t'.k) (some zc') ∧
        ZcRel m zc zc' := by
  unfold bindAndZerocheck at h
  obtain ⟨t₁, k₁, h⟩ := label_bind h
  obtain ⟨t₂, k₂, h⟩ := absorbBytes_bind h
  obtain ⟨t₃, k₃, h⟩ := absorbBytes_bind h
  obtain ⟨t₄, k₄, h⟩ := label_bind h
  obtain ⟨lt₄, n₄, t₅, r₅, kk₅, h⟩ := sampleVec_bind (by omega) h
  obtain ⟨lt₅, n₅, t₆, r₆, kk₆, h⟩ := sampleVec_bind (by omega) h
  obtain ⟨t₇, k₇, h⟩ := absorbSlice_bind h
  obtain ⟨t₈, k₈, h⟩ := absorbSlice_bind h
  obtain ⟨lt₈, n₈, t₉, r₉, kk₉, h⟩ := sample_bind h
  obtain ⟨hpc, h⟩ := check_bind h
  obtain ⟨s, t₁₀, hloop, h⟩ := bind_ok h
  obtain ⟨hfin, h⟩ := check_bind h
  obtain ⟨t₁₁, k₁₁, h⟩ := absorbF_bind h
  obtain ⟨t₁₂, k₁₂, h⟩ := absorbF_bind h
  obtain ⟨rfl, rfl⟩ := pure_ok h
  -- the rounds and the counter, in terms of `t`
  have hr₄ : t₄.rounds = t.rounds := by rw [k₄.2, k₃.2, k₂.2, k₁.2]
  have hk₄ : t₄.k = t.k := by rw [k₄.1, k₃.1, k₂.1, k₁.1]
  have hr₈ : t₈.rounds = t.rounds := by rw [k₈.2, k₇.2, r₆, r₅, hr₄]
  have hk₈ : t₈.k = t.k + 2 := by rw [k₈.1, k₇.1, kk₆, kk₅, hk₄]
  have hr₉ : t₉.rounds = t.rounds := by rw [r₉, hr₈]
  have hk₉ : t₉.k = t.k + 3 := by rw [kk₉, hk₈]
  have hr₅ : t₅.rounds = t.rounds := by rw [r₅, hr₄]
  have hk₅ : t₅.k = t.k + 1 := by rw [kk₅, hk₄]
  set c₀ := (t₄.rounds[t₄.k]!).coins with hc₀
  set c₁ := (t₅.rounds[t₅.k]!).coins with hc₁
  set c₂ := (t₈.rounds[t₈.k]!).coins with hc₂
  have hwf₄ : RoundsWF t₄.rounds := hr₄ ▸ hwf
  have hwf₅ : RoundsWF t₅.rounds := hr₅ ▸ hwf
  have hwf₈ : RoundsWF t₈.rounds := hr₈ ▸ hwf
  have hsz₀ : c₀.size = 6 := by rw [hc₀, hwf₄ t₄.k lt₄, n₄]
  have hsz₁ : c₁.size = m - 13 := by rw [hc₁, hwf₅ t₅.k lt₅, n₅]
  have hsz₂ : c₂.size = 1 := by rw [hc₂, hwf₈ t₈.k lt₈, n₈]
  -- the residual rounds: one round consumed per step
  rw [forIn_range_eq] at hloop
  obtain ⟨hr₁₀, hk₁₀, hn₁₀, hs⟩ := forIn_rounds _ (zStep (pinned ++ c₁) p.rounds) 1 (by
      intro i st tt b tt' hb
      rcases hg : p.rounds[i]! with ⟨g1, ginf⟩
      simp only [hg] at hb
      obtain ⟨tt₁, kk₁, hb⟩ := absorbF_bind hb
      obtain ⟨tt₂, kk₂, hb⟩ := absorbF_bind hb
      obtain ⟨-, nn, tt₃, rr₃, kk₃, hb⟩ := sample_bind hb
      obtain ⟨rfl, rfl⟩ := pure_ok hb
      have e1 : tt₂.rounds = tt.rounds := by rw [kk₂.2, kk₁.2]
      have e2 : tt₂.k = tt.k := by rw [kk₂.1, kk₁.1]
      refine ⟨by rw [← e1, ← e2]; exact nn, by rw [rr₃, e1], by rw [kk₃, e2], ?_⟩
      rw [← e1, ← e2]
      simp [zStep, hg]) (m - 6) 0 t₉.k _ t₉ t₁₀ s (by omega) hloop
  have hkt : t'.k = t.k + 3 + (m - 6) := by rw [k₁₂.1, k₁₁.1, hk₁₀, hk₉]
  have hrt : t'.rounds = t.rounds := by rw [k₁₂.2, k₁₁.2, hr₁₀, hr₉]
  set C : ℕ → Array F128 := fun i => (t₉.rounds[t₉.k + i]!).coins with hC
  have hCsz : ∀ i ∈ List.range' 0 (m - 6), (C i).size = 1 := fun i hi => by
    have hwf₉ : RoundsWF t₉.rounds := hr₉ ▸ hwf
    rw [hC]; simp only; rw [RoundsWF.get hwf₉, hn₁₀ i hi]
  -- the coins consumed
  have hcoins : consumed t.rounds t.k t'.k = c₀ :: c₁ :: c₂ :: (List.range' 0 (m - 6)).map C := by
    rw [hkt, consumed_append t.rounds (show t.k ≤ t.k + 1 by omega) (by omega),
      consumed_append t.rounds (show t.k + 1 ≤ t.k + 2 by omega) (by omega),
      consumed_append t.rounds (show t.k + 2 ≤ t.k + 3 by omega) (by omega), consumed_succ, consumed_succ,
      consumed_succ, consumed_eq_map, hc₀, hc₁, hc₂, hr₄, hk₄, hr₅, hk₅, hr₈, hk₈, hC, hr₉, hk₉]
    rfl
  -- the model's arithmetic on the executable's words
  have hz : fOf c₂[0]! = fOf (c₂.getD 0 0) := by rw [getElem!_eq_getD]
  have hpcM : (∑ i : Fin 64, execArith.lagΛ (fOf (c₂.getD 0 0)) i * vecOf 64 p.round1c i) =
      fOf (dot (lagrange nodesL den6Inv (c₂.getD 0 0)) p.round1c) := by
    rw [fOf_dot _ _ 64 (by rw [lagrange_size, nodesL_size])]; rfl
  have hclaim : (∑ i : Fin 64, execArith.combW (fOf (c₂.getD 0 0)) i * (vecOf 64 p.round1ab i + vecOf 64 p.round1c i)) +
      (∑ i : Fin 64, execArith.lagΛ (fOf (c₂.getD 0 0)) i * vecOf 64 p.round1c i) =
      fOf (dot (combWeights (c₂.getD 0 0)) (Array.map (fun i => p.round1ab[i]! + p.round1c[i]!) (Array.range 64)) +
        dot (lagrange nodesL den6Inv (c₂.getD 0 0)) p.round1c) := by
    rw [hpcM, fOf_add, fOf_dot _ _ 64 (combWeights_size _)]
    refine congrArg (· + fOf (dot (lagrange nodesL den6Inv (c₂.getD 0 0)) p.round1c)) ?_
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [getElem!_range_map _ 64 i i.isLt, fOf_add]
    rfl
  have hr : Model.zcVector execArith (vecOf (m - 13) c₁) = fun j : Fin (m - 6) => fOf (pinned ++ c₁)[0 + j.val]! := by
    funext j; rw [zcVector_eq c₁ hsz₁ (by omega), Nat.zero_add]
  have hres := residual_plays (pinned ++ c₁) p.rounds C (m - 6) 0
    (dot (combWeights (c₂.getD 0 0)) (Array.map (fun i => p.round1ab[i]! + p.round1c[i]!) (Array.range 64)) +
      dot (lagrange nodesL den6Inv (c₂.getD 0 0)) p.round1c, #[]) hCsz
  rw [← hs, ← hclaim, ← hr] at hres
  have hpcE : dot (lagrange nodesL den6Inv (c₂.getD 0 0)) p.round1c = p.finalC := eq_of_beq hpc
  have hfinE : s.1 = p.finalA * p.finalB := eq_of_beq hfin
  obtain ⟨hsz, -⟩ := zStep_fold_size (pinned ++ c₁) p.rounds C (List.range' 0 (m - 6))
    (dot (combWeights (c₂.getD 0 0)) (Array.map (fun i => p.round1ab[i]! + p.round1c[i]!) (Array.range 64)) +
      dot (lagrange nodesL den6Inv (c₂.getD 0 0)) p.round1c, #[])
  rw [← hs] at hsz
  refine ⟨hrt, by omega, ⟨fOf (c₂.getD 0 0), fun j => fOf s.2[0 + j.val]!,
    Model.zcVector execArith (vecOf (m - 13) c₁), fOf p.finalA, fOf p.finalB, fOf p.finalC⟩, ?_, ?_⟩
  · rw [hcoins]
    unfold Model.zerocheck
    refine Plays.bind' (ms₂ := zcMsgs m p) (cs₂ := c₁ :: c₂ :: (List.range' 0 (m - 6)).map C)
      (Plays.drawVec c₀ hsz₀ (by omega)) ?_ (by simp) (by simp)
    refine Plays.bind' (ms₂ := zcMsgs m p) (cs₂ := c₂ :: (List.range' 0 (m - 6)).map C)
      (Plays.drawVec c₁ hsz₁ (by omega)) ?_ (by simp) (by simp)
    refine Plays.bind' (ms₂ := ((List.range (m - 6)).map fun i => ⟨𝔽 × 𝔽, pairOf p.rounds[i]!⟩) ++
      [⟨𝔽 × 𝔽, (fOf p.finalA, fOf p.finalB)⟩]) (cs₂ := c₂ :: (List.range' 0 (m - 6)).map C)
      (Plays.recv _ (vecOf 64 p.round1ab, vecOf 64 p.round1c)) ?_ (by simp [zcMsgs]) (by simp)
    refine Plays.bind' (ms₂ := ((List.range (m - 6)).map fun i => ⟨𝔽 × 𝔽, pairOf p.rounds[i]!⟩) ++
      [⟨𝔽 × 𝔽, (fOf p.finalA, fOf p.finalB)⟩]) (cs₂ := (List.range' 0 (m - 6)).map C)
      (Plays.drawOne c₂ hsz₂) ?_ (by simp) (by simp)
    simp only [hz]
    refine Plays.bind' (ms₂ := [⟨𝔽 × 𝔽, (fOf p.finalA, fOf p.finalB)⟩]) (cs₂ := []) hres ?_
      (by simp [List.range_eq_range']) (by simp)
    refine Plays.bind' (ms₂ := []) (cs₂ := []) (Plays.recv _ (fOf p.finalA, fOf p.finalB)) ?_ (by simp) (by simp)
    have hif : fOf s.1 = fOf p.finalA * fOf p.finalB := by rw [hfinE, fOf_mul]
    simp only [hif, ite_true]
    convert Plays.pure (ι := fOf) _ using 3
    all_goals first | rfl | rw [hpcM, hpcE]
  · refine ⟨rfl, by simpa using hsz, fun i => by simp, by rw [Array.size_append, pinned_size, hsz₁]; omega,
      fun i => zcVector_eq c₁ hsz₁ (by omega) i, rfl, rfl, rfl⟩

end FlockSoundness.Refine
