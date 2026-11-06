import Proofs.Flock.Claims.Sampling.KS

/-!
# Flock's KS at a checked unit's session

`SessionKS` (`Proofs.Flock.Claims.Sampling.KS`) is Flock's KS at the session a selection picks among the checked units.
Here it is proved from `Claims.Flock.KS` and three facts about the deployed sampled verifier, each a hypothesis to
discharge there:

* `DrawFixed`: a checked unit's session is in the second stage, and the checked units don't read its coins.
* `SessionRuns`: at a run that checks `u`, `u`'s session is a Flock run, on its own rounds (an order embedding of
  Flock's), of a Flock prover fixed by the run's coins off them; the session decides the run; and the sampled verifier
  makes the Flock verifier's `H` calls. The session's view needs only the coins before its first round.
* `InnerNumbers`: the inner numbers (`innerWaits`, `innerCap`, `innerN₀`) are Flock's at every session's statement, and
  its `ε_table` and weight at most the inner ones.

At first-stage coins `(ω, r)`, the selected unit's rounds are a set that resampling keeps (`Ex_resample`), so the second
stage's coins average over the session's coins with the coins off it fixed, and there the sampled run is the Flock
prover's (`firstWait_map` for the inner `E^rep`, `Ex_comp_inj` for its coins). `β(ω, r)` is `B`'s collision
probability at `(ω, r)`: a collision of the Flock verifier's calls in its `B` is one of the sampled verifier's at the
fork on the same round, and Flock's `γ` at its `k_F ≤ k` rounds, times the `k/k_F` that guessing among the sampled
rounds costs, is at most `γ_F` (`γFlock_mul_le`).

* `SessionKS_of`: **`SessionKS` from Flock's KS**, `DrawFixed`, `SessionRuns`, `InnerNumbers` and `ε_table ≥ 0`.

A session may cover several checked units, as the one-stage audit's one session proves every drawn unit:
`SessionRunsSlots` asks `SessionRuns` only of the units at the slots whose sessions cover the rest (`SlotsCover`,
`Sampling.KS`), and gives it (`sessionRuns_of_slots`); one session per unit is `SessionRuns` itself
(`sessionRunsSlots_units_iff`). `InnerNumbers` at one session over every unit is one statement about it
(`innerNumbers_one_iff`).
-/

namespace Claims.Sampling

open Claims Finset

/-! ## Coins restricted along an injective map, and resampled on a set they determine -/

section Coins

variable {Coin : Type} [Fintype Coin] [Nonempty Coin]

/-- **Uniform coins restricted along an injective map are uniform.** -/
theorem Ex_comp_inj {ι ι' : Type} [Fintype ι] [Fintype ι'] [DecidableEq ι] [DecidableEq ι'] (e : ι' → ι)
    (he : Function.Injective e) (h : (ι' → Coin) → ℝ) :
    Ex (fun c : ι → Coin => h fun i => c (e i)) = Ex h := by
  classical
  let eq : (ι → Coin) ≃ (ι' → Coin) × ({t // t ∉ Set.range e} → Coin) :=
    (Equiv.piEquivPiSubtypeProd (fun t => t ∈ Set.range e) fun _ => Coin).trans
      (Equiv.prodCongr (Equiv.arrowCongr (Equiv.ofInjective e he).symm (Equiv.refl Coin)) (Equiv.refl _))
  calc Ex (fun c : ι → Coin => h fun i => c (e i)) = Ex (fun c => h (eq c).1) := rfl
    _ = Ex (fun s : (ι' → Coin) × ({t // t ∉ Set.range e} → Coin) => h s.1) := Ex_equiv eq (fun s => h s.1)
    _ = Ex h := Ex_fst h

theorem Pr_comp_inj {ι ι' : Type} [Fintype ι] [Fintype ι'] [DecidableEq ι] [DecidableEq ι'] (e : ι' → ι)
    (he : Function.Injective e) (q : (ι' → Coin) → Prop) :
    Pr (fun c : ι → Coin => q fun i => c (e i)) = Pr q := by
  classical
  rw [← Ex_ind (fun c : ι → Coin => q fun i => c (e i)), ← Ex_ind q]
  exact Ex_comp_inj e he fun c => if q c then 1 else 0

omit [Fintype Coin] [Nonempty Coin] in
theorem forkOn_empty {k : ℕ} (r d : Fin k → Coin) : forkOn ∅ r d = r :=
  funext fun t => by simp [forkOn]

/-- **Resampling on a set the coins determine**: when resampling `d` on `S d` keeps `S d`, averaging over `d` and then
over fresh coins on `S d` is averaging over `d`: swapping `d` and the fresh coins on `S d` is an involution. -/
theorem Ex_resample {k : ℕ} (S : (Fin k → Coin) → Finset (Fin k)) (hS : ∀ d σ, S (forkOn (S d) d σ) = S d)
    (F : (Fin k → Coin) → ℝ) :
    Ex (fun d => Ex fun σ => F (forkOn (S d) d σ)) = Ex F := by
  have inv : ∀ p : (Fin k → Coin) × (Fin k → Coin),
      (forkOn (S (forkOn (S p.1) p.1 p.2)) (forkOn (S p.1) p.1 p.2) (forkOn (S p.1) p.2 p.1),
        forkOn (S (forkOn (S p.1) p.1 p.2)) (forkOn (S p.1) p.2 p.1) (forkOn (S p.1) p.1 p.2)) = p := by
    intro p
    rw [hS]
    refine Prod.ext (funext fun t => ?_) (funext fun t => ?_) <;> by_cases h : t ∈ S p.1 <;> simp [forkOn, h]
  let sw : (Fin k → Coin) × (Fin k → Coin) ≃ (Fin k → Coin) × (Fin k → Coin) :=
    { toFun := fun p => (forkOn (S p.1) p.1 p.2, forkOn (S p.1) p.2 p.1)
      invFun := fun p => (forkOn (S p.1) p.1 p.2, forkOn (S p.1) p.2 p.1)
      left_inv := inv
      right_inv := inv }
  have h1 := Ex_prod (fun p : (Fin k → Coin) × (Fin k → Coin) => F (forkOn (S p.1) p.1 p.2))
  have h2 : Ex (fun p : (Fin k → Coin) × (Fin k → Coin) => F (forkOn (S p.1) p.1 p.2)) =
      Ex (fun p : (Fin k → Coin) × (Fin k → Coin) => F p.1) := Ex_equiv sw (fun p => F p.1)
  rw [← Ex_fst (β := Fin k → Coin) F, ← h2, h1]

theorem Pr_pi_eval {ι E : Type} [Fintype ι] [DecidableEq ι] [Fintype E] [Nonempty E] (i : ι) (q : E → Prop) :
    Pr (fun c : ι → E => q (c i)) = Pr q := by
  classical
  rw [← Ex_ind (fun c : ι → E => q (c i)), ← Ex_ind q]
  exact Ex_pi_eval i fun e => if q e then 1 else 0

omit [Fintype Coin] [Nonempty Coin] in
/-- Fresh coins on `J ⊆ I`, then `r` off `I`, are fresh on `J` of the coins fresh on `I`. -/
theorem forkOn_sub {k : ℕ} {I J : Finset (Fin k)} (h : J ⊆ I) (r d σ : Fin k → Coin) :
    forkOn I r (forkOn J d σ) = forkOn J (forkOn I r d) σ :=
  funext fun t => by
    by_cases hJ : t ∈ J
    · simp [forkOn, hJ, h hJ]
    · simp [forkOn, hJ]

end Coins

theorem finRange_map_cast {α : Type} {n m : ℕ} (h : n = m) (f : Fin m → α) :
    (List.finRange n).map (fun i => f (Fin.cast h i)) = (List.finRange m).map f := by
  subst h
  rfl

/-- Three-argument currying. -/
def curry3 {α β γ E : Type} : (α × β × γ → E) ≃ (α → β → γ → E) where
  toFun f a b c := f (a, b, c)
  invFun g p := g p.1 p.2.1 p.2.2
  left_inv _ := rfl
  right_inv _ := rfl

/-! ## Flock's waits under a map of the coins -/

/-- `waitAcc` on coins mapped by `g`, against runs that accept together and open alike at the accepted ones, keeps
what opens alike. -/
theorem waitAcc_map {R R' Op : Type} (H : Msg → Msg) (run : R → Trace) (run' : R' → Trace) (g : R → R')
    (opens opens' : Seen → List Op) (k : ℕ) :
    ∀ (rs : List R) (acc acc' : List Seen),
      (∀ r ∈ rs, ((run r).seen.out = .acc ↔ (run' (g r)).seen.out = .acc) ∧
        ((run r).seen.out = .acc → opens (run r).seen = opens' (run' (g r)).seen)) →
      acc.length = acc'.length → acc.flatMap opens = acc'.flatMap opens' →
      ((waitAcc k rs acc).eval H run).1.flatMap opens = ((waitAcc k (rs.map g) acc').eval H run').1.flatMap opens'
  | [], acc, acc', _, _, h => h
  | r :: rs, acc, acc', hr, hl, h => by
    simp only [waitAcc, List.map_cons]
    rw [← hl]
    split_ifs with hk
    · exact h
    · simp only [RP.eval]
      obtain ⟨hiff, hop⟩ := hr r (List.mem_cons_self ..)
      have hrs := fun r' hr' => hr r' (List.mem_cons_of_mem _ hr')
      by_cases ha : (run r).seen.out = .acc
      · rw [ite_eq_left ha, ite_eq_left (hiff.1 ha)]
        exact waitAcc_map H run run' g opens opens' k rs _ _ hrs (by simp [hl])
          (by simp [List.flatMap_append, h, hop ha])
      · rw [ite_eq_right ha, ite_eq_right (fun h' => ha (hiff.2 h'))]
        exact waitAcc_map H run run' g opens opens' k rs _ _ hrs hl h

/-- **Flock's `E^rep` waits on coins mapped by `g`**: against runs that accept together and open alike at the accepted
ones, `firstWait` finds the same witness. -/
theorem firstWait_map {R R' W Op : Type} (H : Msg → Msg) (run : R → Trace) (run' : R' → Trace) (g : R → R')
    (opens opens' : Seen → List Op) (k : ℕ) (assemble : List Op → Option W) (check : W → Bool) :
    ∀ rss : List (List R),
      (∀ rs ∈ rss, ∀ r ∈ rs, ((run r).seen.out = .acc ↔ (run' (g r)).seen.out = .acc) ∧
        ((run r).seen.out = .acc → opens (run r).seen = opens' (run' (g r)).seen)) →
      ((firstWait k opens assemble check rss).eval H run).1 =
        ((firstWait k opens' assemble check (rss.map (List.map g))).eval H run').1
  | [], _ => rfl
  | rs :: rss, hr => by
    have ih := firstWait_map H run run' g opens opens' k assemble check rss
      fun rs' h => hr rs' (List.mem_cons_of_mem _ h)
    have hw := waitAcc_map H run run' g opens opens' k rs [] [] (hr rs (List.mem_cons_self ..)) rfl rfl
    simp only [firstWait, List.map_cons]
    rw [(RP.eval_bind _ _ _ _).1, (RP.eval_bind _ _ _ _).1, hw]
    cases assemble (((waitAcc k (rs.map g) []).eval H run').1.flatMap opens') with
    | none => exact ih
    | some w =>
      simp only
      split_ifs
      · rfl
      · exact ih

/-! ## `B`'s collisions, as logged collisions -/

theorem B_Pr_logColl {Ω Coin : Type} [Fintype Ω] [Nonempty Ω] [Fintype Coin] [Nonempty Coin] [Inhabited Coin]
    (H : Msg → Msg) (V : Verifier Coin) (P : Prover Ω) (x : Msg) (hk : 0 < V.k x) :
    (B H V P x hk).Pr H = Pr fun s : BCoins Ω Coin (V.k x) =>
      LogColl ((run H P V x s.2.1 s.2.2.1).vLog ++ (run H P V x s.2.1 (forkCoins s.1 s.2.2.1 s.2.2.2)).vLog) := by
  rw [B_Pr_eq]
  refine Pr_congr fun s => ⟨fun h => ?_, fun h => findColl_isSome _ h⟩
  obtain ⟨⟨u, v⟩, huv⟩ := Option.isSome_iff_exists.1 h
  obtain ⟨hc, hu, hv⟩ := B_out_spec H V P x hk s u v huv
  obtain ⟨p, hp, rfl⟩ := List.mem_map.1 hu
  obtain ⟨p', hp', rfl⟩ := List.mem_map.1 hv
  have hg : ∀ q ∈ (run H P V x s.2.1 s.2.2.1).vLog ++ (run H P V x s.2.1 (forkCoins s.1 s.2.2.1 s.2.2.2)).vLog,
      H q.1 = q.2 := fun q hq => (List.mem_append.1 hq).elim (fun hq => (run_logs H P V x _ _).2 q hq)
        fun hq => (run_logs H P V x _ _).2 q hq
  exact ⟨p, hp, p', hp', hc.1, by rw [← hg p hp, ← hg p' hp', hc.2]⟩

theorem PrB_nonneg {Ω Coin : Type} [Fintype Ω] [Nonempty Ω] [Fintype Coin] [Nonempty Coin] [Inhabited Coin]
    (H : Msg → Msg) (V : Verifier Coin) (P : Prover Ω) (x : Msg) : 0 ≤ PrB H V P x := by
  unfold PrB
  split_ifs
  · exact Pr_nonneg _
  · exact le_rfl

theorem Pr_unit {X : Type} [Fintype X] (p : X → Prop) : Pr (fun s : Unit × X => p s.2) = Pr p :=
  Pr_equiv (Equiv.punitProd X) p

section Restrict

variable {Coin : Type} [Fintype Coin] [Nonempty Coin]

/-- Two independent coin vectors restricted along an injective map. -/
theorem Pr_restrict_pair {kS kF : ℕ} (emb : Fin kF → Fin kS) (hemb : Function.Injective emb)
    (Q : (Fin kF → Coin) → (Fin kF → Coin) → Prop) :
    Pr (fun s : (Fin kS → Coin) × (Fin kS → Coin) => Q (fun i => s.1 (emb i)) fun i => s.2 (emb i)) =
      Pr (fun s : (Fin kF → Coin) × (Fin kF → Coin) => Q s.1 s.2) := by
  rw [Pr_prod_eq_Ex, Pr_prod_eq_Ex]
  have h : ∀ a : Fin kF → Coin, Pr (fun R2 : Fin kS → Coin => Q a fun i => R2 (emb i)) = Pr (Q a) :=
    fun a => Pr_comp_inj emb hemb (Q a)
  simp only [h]
  exact Ex_comp_inj emb hemb fun a => Pr (Q a)

/-- A coin vector and an `E^rep`'s coins, restricted along an injective map and reindexed. -/
theorem Pr_restrict_waits {kS kF W C W' C' : ℕ} (emb : Fin kF → Fin kS) (hemb : Function.Injective emb)
    (hw : W' = W) (hc : C' = C) (Q : (Fin kF → Coin) → (Fin W' → Fin C' → Fin kF → Coin) → Prop) :
    Pr (fun s : (Fin kS → Coin) × (Fin W → Fin C → Fin kS → Coin) =>
        Q (fun i => s.1 (emb i)) fun i j t => s.2 (Fin.cast hw i) (Fin.cast hc j) (emb t)) =
      Pr (fun s : (Fin kF → Coin) × (Fin W' → Fin C' → Fin kF → Coin) => Q s.1 s.2) := by
  rw [Pr_prod_eq_Ex, Pr_prod_eq_Ex]
  let e : Fin W' × Fin C' × Fin kF → Fin W × Fin C × Fin kS := fun p => (Fin.cast hw p.1, Fin.cast hc p.2.1, emb p.2.2)
  have he : Function.Injective e := by
    intro p q h
    simp only [e, Prod.mk.injEq] at h
    obtain ⟨h1, h2, h3⟩ := h
    exact Prod.ext (Fin.ext (by simpa [Fin.ext_iff] using h1))
      (Prod.ext (Fin.ext (by simpa [Fin.ext_iff] using h2)) (hemb h3))
  have h : ∀ a : Fin kF → Coin,
      Pr (fun ρ0 : Fin W → Fin C → Fin kS → Coin => Q a fun i j t => ρ0 (Fin.cast hw i) (Fin.cast hc j) (emb t)) =
        Pr (Q a) := fun a =>
    (Pr_equiv curry3 _).symm.trans ((Pr_comp_inj e he fun f => Q a (curry3 f)).trans (Pr_equiv curry3 (Q a)))
  simp only [h]
  exact Ex_comp_inj emb hemb fun a => Pr (Q a)

end Restrict

/-- Flock's `γ` at `k_F ≤ k` rounds and weight `w_F ≤ w`, times `k`, is at most `γ_F` at `k` and `w` times `k_F`: with
`a = √k_F`, `b = √k`, `s = √(2w)`, it is `16a²b²(s + a)² ≤ 16a²b²(s + b)²`. -/
theorem γFlock_mul_le {kF kS : ℕ} {wF w : ℝ} (hk : kF ≤ kS) (hw : wF ≤ w) :
    16 * (Real.sqrt (2 * kF * wF) + kF) ^ 2 * kS ≤ 16 * (Real.sqrt (2 * kS * w) + kS) ^ 2 * kF := by
  have hkF : (0 : ℝ) ≤ kF := Nat.cast_nonneg _
  have hkS : (0 : ℝ) ≤ kS := Nat.cast_nonneg _
  have hkk : (kF : ℝ) ≤ kS := by exact_mod_cast hk
  have ha : Real.sqrt kF ^ 2 = kF := Real.sq_sqrt hkF
  have hb : Real.sqrt kS ^ 2 = kS := Real.sq_sqrt hkS
  have hab : Real.sqrt kF ≤ Real.sqrt kS := Real.sqrt_le_sqrt hkk
  have ha0 : 0 ≤ Real.sqrt kF := Real.sqrt_nonneg _
  have hs0 : 0 ≤ Real.sqrt (2 * w) := Real.sqrt_nonneg _
  have e1 : Real.sqrt (2 * kF * wF) ≤ Real.sqrt kF * Real.sqrt (2 * w) := by
    rw [← Real.sqrt_mul hkF]
    exact Real.sqrt_le_sqrt (by nlinarith)
  have e2 : Real.sqrt (2 * kS * w) = Real.sqrt kS * Real.sqrt (2 * w) := by
    rw [← Real.sqrt_mul hkS]
    ring_nf
  have h1 : Real.sqrt (2 * kF * wF) + kF ≤ Real.sqrt kF * (Real.sqrt (2 * w) + Real.sqrt kF) := by nlinarith
  rw [e2]
  calc 16 * (Real.sqrt (2 * kF * wF) + kF) ^ 2 * kS
      ≤ 16 * (Real.sqrt kF * (Real.sqrt (2 * w) + Real.sqrt kF)) ^ 2 * kS := by gcongr
    _ = 16 * Real.sqrt kF ^ 2 * Real.sqrt kS ^ 2 * (Real.sqrt (2 * w) + Real.sqrt kF) ^ 2 := by rw [hb]; ring
    _ ≤ 16 * Real.sqrt kF ^ 2 * Real.sqrt kS ^ 2 * (Real.sqrt (2 * w) + Real.sqrt kS) ^ 2 := by gcongr
    _ = 16 * (Real.sqrt kS * Real.sqrt (2 * w) + Real.sqrt kS ^ 2) ^ 2 * Real.sqrt kF ^ 2 := by ring
    _ = 16 * (Real.sqrt kS * Real.sqrt (2 * w) + kS) ^ 2 * kF := by rw [ha, hb]

/-! ## What the deployed sampled verifier is taken to do at a session -/

section Session

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]

/-- **The draw precedes the sessions**: a unit checked at `r` has its session's rounds in the second stage, and fresh
coins on them keep the checked units. -/
def DrawFixed (D : Deploy Coin Col T) : Prop :=
  ∀ x r u, (∃ p : Fin (nPU D x), (D.checked x r)[p.val]? = some u) →
    D.rounds x u ⊆ D.stage2 x ∧ ∀ d, D.checked x (forkOn (D.rounds x u) r d) = D.checked x r

/-- **A checked unit's session is a Flock run**: at coins `r` that check `u`, the session's statement `xS`, an order
embedding `emb` of Flock's rounds onto `u`'s, and a Flock prover `P'` (the sampled prover with `r`'s coins off `u`'s
rounds), such that at coins `r'` agreeing with `r` off `u`'s rounds the sampled run accepts exactly when `P'`'s run on
`r'`'s session coins does (the session decides the run), and at coins agreeing with `r` on the rounds outside `u`'s
before its first, `u`'s session is `P'`'s run and the sampled verifier makes its `H` calls. -/
def SessionRuns (H : Msg → Msg) (D : Deploy Coin Col T) : Prop :=
  ∀ x (Ω : Type) (P : Prover Ω) (ω : Ω) (r : Fin (D.V.k x) → Coin) (u : ℕ),
    (∃ p : Fin (nPU D x), (D.checked x r)[p.val]? = some u) →
    ∃ (xS : Msg) (emb : Fin (D.flock.V.k xS) → Fin (D.V.k x)) (P' : Prover Unit),
      StrictMono emb ∧ (∀ t, t ∈ D.rounds x u ↔ ∃ i, emb i = t) ∧
      (∀ r' : Fin (D.V.k x) → Coin, (∀ t ∉ D.rounds x u, r' t = r t) →
        ((run H P D.V x ω r').out = .acc ↔ (run H P' D.flock.V xS () fun i => r' (emb i)).out = .acc)) ∧
      ∀ r' : Fin (D.V.k x) → Coin, (∀ t ∉ D.rounds x u, (∀ i, t < emb i) → r' t = r t) →
        D.session x u (run H P D.V x ω r').seen = some (xS, (run H P' D.flock.V xS () fun i => r' (emb i)).seen) ∧
          ∀ q ∈ (run H P' D.flock.V xS () fun i => r' (emb i)).vLog, q ∈ (run H P D.V x ω r').vLog

/-- **The inner numbers are Flock's** at every session's statement: its `E^rep`'s waits, truncation and `N₀`, and its
`ε_table` and weight at most the inner ones. -/
def InnerNumbers (D : Deploy Coin Col T) : Prop :=
  ∀ x u v xS vS, D.session x u v = some (xS, vS) →
    D.flock.waits xS = D.innerWaits x ∧ D.flock.cap xS = D.innerCap x ∧ D.flock.N₀ xS = D.innerN₀ x ∧
      D.flock.εtable xS ≤ D.εtable x ∧ D.flock.weight xS ≤ D.weight x

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col] in
/-- **`DrawFixed` from the draw's place in the run**: where each checked unit's session is in the second stage and the
checked units read only the coins off the second stage, fresh coins on a session keep them. -/
theorem drawFixed_of_first (D : Deploy Coin Col T)
    (hrounds : ∀ x r u, (∃ p : Fin (nPU D x), (D.checked x r)[p.val]? = some u) → D.rounds x u ⊆ D.stage2 x)
    (hfirst : ∀ x r r', (∀ t ∉ D.stage2 x, r t = r' t) → D.checked x r = D.checked x r') : DrawFixed D := by
  intro x r u hu
  refine ⟨hrounds x r u hu, fun d => hfirst x _ _ fun t ht => ?_⟩
  have hn : t ∉ D.rounds x u := fun hm => ht (hrounds x r u hu hm)
  simp [forkOn, hn]

/-- **`SessionRuns` a session at a time**: at coins `r`, the unit checked at a slot below `S x` has its session a Flock
run on its rounds, as `SessionRuns` says of every checked unit. Where the slots cover the checked units
(`SlotsCover`), that is `SessionRuns` (`sessionRuns_of_slots`): one session over every checked unit, as the one-stage
audit runs, is one obligation per run (`S = 1`), and one session per unit is `SessionRuns` itself (`S = nPU D`,
`sessionRunsSlots_units_iff`). -/
def SessionRunsSlots (H : Msg → Msg) (D : Deploy Coin Col T) (S : Msg → ℕ) : Prop :=
  ∀ x (Ω : Type) (P : Prover Ω) (ω : Ω) (r : Fin (D.V.k x) → Coin) (p : ℕ) (u : ℕ),
    p < S x → (D.checked x r)[p]? = some u →
    ∃ (xS : Msg) (emb : Fin (D.flock.V.k xS) → Fin (D.V.k x)) (P' : Prover Unit),
      StrictMono emb ∧ (∀ t, t ∈ D.rounds x u ↔ ∃ i, emb i = t) ∧
      (∀ r' : Fin (D.V.k x) → Coin, (∀ t ∉ D.rounds x u, r' t = r t) →
        ((run H P D.V x ω r').out = .acc ↔ (run H P' D.flock.V xS () fun i => r' (emb i)).out = .acc)) ∧
      ∀ r' : Fin (D.V.k x) → Coin, (∀ t ∉ D.rounds x u, (∀ i, t < emb i) → r' t = r t) →
        D.session x u (run H P D.V x ω r').seen = some (xS, (run H P' D.flock.V xS () fun i => r' (emb i)).seen) ∧
          ∀ q ∈ (run H P' D.flock.V xS () fun i => r' (emb i)).vLog, q ∈ (run H P D.V x ω r').vLog

omit [Fintype Coin] [Nonempty Coin] [DecidableEq Col] [Zero Col] in
/-- **`SessionRuns` from its slot form**: each checked unit has a slot's rounds and session. -/
theorem sessionRuns_of_slots {H : Msg → Msg} {D : Deploy Coin Col T} {S : Msg → ℕ} (hc : SlotsCover D S)
    (h : SessionRunsSlots H D S) : SessionRuns H D := by
  intro x Ω P ω r u hu
  obtain ⟨p, hp, u', hu', hr, hs⟩ := hc x r u hu
  rw [hr, hs]
  exact h x Ω P ω r p u' hp hu'

omit [Fintype Coin] [Nonempty Coin] [DecidableEq Col] [Zero Col] in
/-- **At one session per unit the slot form is `SessionRuns`.** -/
theorem sessionRunsSlots_units_iff (H : Msg → Msg) (D : Deploy Coin Col T) :
    SessionRunsSlots H D (nPU D) ↔ SessionRuns H D :=
  ⟨sessionRuns_of_slots (slotsCover_units D), fun h x Ω P ω r p u hp hu => h x Ω P ω r u ⟨⟨p, hp⟩, hu⟩⟩

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col] in
/-- **`InnerNumbers` at one session over every unit**: one statement about the run's one session. `InnerNumbers` reads
every session a unit has, whichever units share it, so several sessions at `x` must have Flock's numbers alike: the
same tables' shapes give the same `N₀`. -/
theorem innerNumbers_one_iff (D : Deploy Coin Col T) (hs : ∀ x u u', D.session x u = D.session x u') :
    InnerNumbers D ↔ ∀ x v xS vS, D.session x 0 v = some (xS, vS) →
      D.flock.waits xS = D.innerWaits x ∧ D.flock.cap xS = D.innerCap x ∧ D.flock.N₀ xS = D.innerN₀ x ∧
        D.flock.εtable xS ≤ D.εtable x ∧ D.flock.weight xS ≤ D.weight x :=
  ⟨fun h x v xS vS hv => h x 0 v xS vS hv, fun h x u v xS vS hv => h x v xS vS (hs x u 0 ▸ hv)⟩

/-- The unit a selection picks among those checked at `R`. -/
def selUnit (D : Deploy Coin Col T) (x : Msg) (sel : List ℕ → Option (Fin (nPU D x)))
    (R : Fin (D.V.k x) → Coin) : Option ℕ :=
  (sel (D.checked x R)).bind fun i => (D.checked x R)[i.val]?

/-- Its session's rounds, none when no unit is picked. -/
def selRounds (D : Deploy Coin Col T) (x : Msg) (sel : List ℕ → Option (Fin (nPU D x)))
    (R : Fin (D.V.k x) → Coin) : Finset (Fin (D.V.k x)) :=
  match selUnit D x sel R with
  | some u => D.rounds x u
  | none => ∅

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col] in
theorem selUnit_mem {D : Deploy Coin Col T} {x : Msg} {sel : List ℕ → Option (Fin (nPU D x))}
    {R : Fin (D.V.k x) → Coin} {u : ℕ} (h : selUnit D x sel R = some u) :
    ∃ p : Fin (nPU D x), (D.checked x R)[p.val]? = some u := by
  unfold selUnit at h
  obtain ⟨i, -, hi⟩ := Option.bind_eq_some_iff.1 h
  exact ⟨i, hi⟩

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col] in
theorem selUnit_forkOn {D : Deploy Coin Col T} (hdraw : DrawFixed D) {x : Msg}
    {sel : List ℕ → Option (Fin (nPU D x))} {R : Fin (D.V.k x) → Coin} {u : ℕ}
    (h : selUnit D x sel R = some u) (σ : Fin (D.V.k x) → Coin) :
    selUnit D x sel (forkOn (D.rounds x u) R σ) = some u := by
  have hc := (hdraw x R u (selUnit_mem h)).2 σ
  unfold selUnit at h ⊢
  rw [hc]
  exact h

/-- The sampled `B`'s collision at the tape `ω` and the judged run's coins `R`, forking on round `j` to `R2`. -/
def SColl (H : Msg → Msg) (D : Deploy Coin Col T) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) (j : Fin (D.V.k x))
    (R R2 : Fin (D.V.k x) → Coin) : Prop :=
  LogColl ((run H P D.V x ω R).vLog ++ (run H P D.V x ω (forkCoins j R R2)).vLog)

/-- **`B`'s collision probability at `(ω, R)`**: over its round guess and the fresh coins after it. -/
noncomputable def collAt (H : Msg → Msg) (D : Deploy Coin Col T) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (R : Fin (D.V.k x) → Coin) : ℝ :=
  (∑ j, Pr fun R2 => SColl H D P x ω j R R2) / D.V.k x

omit [Nonempty Coin] [DecidableEq Col] [Zero Col] in
theorem collAt_nonneg (H : Msg → Msg) (D : Deploy Coin Col T) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (R : Fin (D.V.k x) → Coin) : 0 ≤ collAt H D P x ω R :=
  div_nonneg (sum_nonneg fun _ _ => Pr_nonneg _) (Nat.cast_nonneg _)

omit [DecidableEq Col] [Zero Col] in
/-- `B`'s collision probability at `(ω, R)` averages to `Pr[B]`. -/
theorem Ex_collAt (H : Msg → Msg) (D : Deploy Coin Col T) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω)
    (x : Msg) : Ex (fun p : Ω × (Fin (D.V.k x) → Coin) => collAt H D P x p.1 p.2) = PrB H D.V P x := by
  unfold PrB
  split_ifs with hk
  · rw [B_Pr_logColl, Pr_guess, Ex_prod]
    simp only [collAt, Ex_div, Ex_finsum]
    congr 1
    refine Finset.sum_congr rfl fun j _ => ?_
    symm
    rw [Pr_prod_eq_Ex]
    congr 1
    funext ω
    rw [Pr_prod_eq_Ex]
    rfl
  · have h0 : (D.V.k x : ℝ) = 0 := by exact_mod_cast Nat.eq_zero_of_not_pos hk
    simp [collAt, h0, Ex]

set_option synthInstance.maxSize 1024 in
/-- **One session's coins, the others fixed**: at coins `R0` that check `u`, with fresh coins on `u`'s rounds, the run
accepting while `u`'s `E^rep` finds no witness is Flock's bad event of the session's prover (`SessionRuns`), so it has
probability at most Flock's error, `κ_F + √(γ_F·Pr[B])` at its fresh rounds, with that prover's `B` a fork of the
sampled run on the session's rounds. -/
theorem fiber_bound {H : Msg → Msg} {D : Deploy Coin Col T} (hKS : Claims.Flock.KS H D.flock)
    (hruns : SessionRuns H D) (hnum : InnerNumbers D) (decode : List (ℕ × Col) → List T)
    (hdec : Flock.ListWithin D.flock.len D.flock.enc D.flock.θ decode) (x : Msg) {Ω : Type} (P : Prover Ω) (ω : Ω)
    (R0 : Fin (D.V.k x) → Coin) (u : ℕ) (hu : ∃ p : Fin (nPU D x), (D.checked x R0)[p.val]? = some u) :
    Pr (fun s : (Fin (D.V.k x) → Coin) × D.UnitCoins x =>
        (run H P D.V x ω (forkOn (D.rounds x u) R0 s.1)).out = .acc ∧
          unitWit H D decode x (run H P D.V x ω) u (forkOn (D.rounds x u) R0 s.1) s.2 = none) ≤
      κF D x + Real.sqrt (γF D x * Ex fun σ => collAt H D P x ω (forkOn (D.rounds x u) R0 σ)) := by
  obtain ⟨xS, emb, P', hmono, hrange, hacc, hview⟩ := hruns x Ω P ω R0 u hu
  have hS : ∀ i, emb i ∈ D.rounds x u := fun i => (hrange _).2 ⟨i, rfl⟩
  have hinj : Function.Injective emb := hmono.injective
  obtain ⟨hw, hc, hN, hεF, hwt⟩ := hnum x u _ xS _ (hview R0 fun _ _ _ => rfl).1
  have hk : Claims.Flock.Deploy.k D.flock xS = D.innerK x := by
    unfold Claims.Flock.Deploy.k Deploy.innerK
    rw [hN]
  have hkle : D.flock.V.k xS ≤ D.V.k x := by simpa using Fintype.card_le_of_injective emb hinj
  have hag0 : ∀ σ : Fin (D.V.k x) → Coin, ∀ t ∉ D.rounds x u, forkOn (D.rounds x u) R0 σ t = R0 t :=
    fun σ t ht => by simp [forkOn, ht]
  -- the sampled event is Flock's bad event at the session's coins
  have himp : ∀ (σ : Fin (D.V.k x) → Coin) (ρ0 : D.UnitCoins x),
      (run H P D.V x ω (forkOn (D.rounds x u) R0 σ)).out = .acc →
      unitWit H D decode x (run H P D.V x ω) u (forkOn (D.rounds x u) R0 σ) ρ0 = none →
      (run H P' D.flock.V xS () fun i => σ (emb i)).out = .acc ∧
        ¬ (Claims.Flock.R D.flock.parse).ext xS
          ((Claims.Flock.E D.flock decode).eval H P' xS () (fun i => σ (emb i))
            fun i j t => ρ0 (Fin.cast hw i) (Fin.cast hc j) (emb t)).1 := by
    intro σ ρ0 h1 h2
    have he : (fun i => forkOn (D.rounds x u) R0 σ (emb i)) = fun i => σ (emb i) :=
      funext fun i => by simp [forkOn, hS i]
    have ha := (hacc _ (hag0 σ)).1 h1
    rw [he] at ha
    refine ⟨ha, ?_⟩
    have hsess := (hview _ fun t ht _ => hag0 σ t ht).1
    unfold unitWit at h2
    rw [hsess] at h2
    simp only at h2
    let g : (Fin (D.V.k x) → Coin) → Fin (D.flock.V.k xS) → Coin := fun R i => R (emb i)
    let Ls : List (List (Fin (D.V.k x) → Coin)) := (List.finRange (D.innerWaits x)).map fun i =>
      (List.finRange (D.innerCap x)).map fun j => forkOn (D.rounds x u) (forkOn (D.rounds x u) R0 σ) (ρ0 i j)
    have hcorr : ∀ rs ∈ Ls, ∀ r ∈ rs,
        ((run H P D.V x ω r).seen.out = .acc ↔ (run H P' D.flock.V xS () (g r)).seen.out = .acc) ∧
          ((run H P D.V x ω r).seen.out = .acc →
            D.sessionCols x u (run H P D.V x ω r).seen = [D.flock.cols (run H P' D.flock.V xS () (g r)).seen]) := by
      intro rs hrs r hr
      obtain ⟨i, -, rfl⟩ := List.mem_map.1 hrs
      obtain ⟨j, -, rfl⟩ := List.mem_map.1 hr
      have hag : ∀ t ∉ D.rounds x u,
          forkOn (D.rounds x u) (forkOn (D.rounds x u) R0 σ) (ρ0 i j) t = R0 t := fun t ht => by simp [forkOn, ht]
      refine ⟨hacc _ hag, fun _ => ?_⟩
      unfold Deploy.sessionCols
      rw [(hview _ fun t ht _ => hag t ht).1]
    have hL : ((List.finRange (D.flock.waits xS)).map fun i => (List.finRange (D.flock.cap xS)).map
        fun j t => ρ0 (Fin.cast hw i) (Fin.cast hc j) (emb t)) = Ls.map (List.map g) := by
      refine (finRange_map_cast hw (fun i' => (List.finRange (D.flock.cap xS)).map
        fun j t => ρ0 i' (Fin.cast hc j) (emb t))).trans ?_
      rw [List.map_map]
      refine List.map_congr_left fun i _ => ?_
      refine (finRange_map_cast hc (fun j' t => ρ0 i j' (emb t))).trans ?_
      simp only [Function.comp_apply, List.map_map]
      refine List.map_congr_left fun j _ => ?_
      funext t
      simp [g, forkOn, hS t]
    have hfw := firstWait_map H (run H P D.V x ω) (run H P' D.flock.V xS ()) g (D.sessionCols x u)
      (fun v => [D.flock.cols v]) (D.innerK x) (Flock.assemble D.flock.parse D.flock.table decode D.flock.read xS)
      (Flock.check D.flock.parse xS) Ls hcorr
    have ha' : (run H P' D.flock.V xS () fun i => σ (emb i)).seen.out = .acc := ha
    simp only [Extractor.eval, Claims.Flock.E, ERep, RP.eval]
    rw [ite_eq_left ha', hk, hL, ← hfw, h2]
    exact id
  -- `B` of the session's prover is a fork of the sampled run on the session's rounds
  have hB : (D.flock.V.k xS : ℝ) * PrB H D.flock.V P' xS ≤
      Ex fun σ => ∑ j, Pr fun R2 => SColl H D P x ω j (forkOn (D.rounds x u) R0 σ) R2 := by
    by_cases hkF : 0 < D.flock.V.k xS
    · rw [show PrB H D.flock.V P' xS = (B H D.flock.V P' xS hkF).Pr H from dite_eq_left hkF, B_Pr_logColl, Pr_guess,
        mul_div_cancel₀ _ (by positivity : (D.flock.V.k xS : ℝ) ≠ 0), Ex_finsum]
      have hterm : ∀ jF : Fin (D.flock.V.k xS),
          Pr (fun b : Unit × (Fin (D.flock.V.k xS) → Coin) × (Fin (D.flock.V.k xS) → Coin) =>
            LogColl ((run H P' D.flock.V xS b.1 b.2.1).vLog ++
              (run H P' D.flock.V xS b.1 (forkCoins jF b.2.1 b.2.2)).vLog)) ≤
          Ex fun σ => Pr fun R2 => SColl H D P x ω (emb jF) (forkOn (D.rounds x u) R0 σ) R2 := by
        intro jF
        rw [Pr_unit (fun b : (Fin (D.flock.V.k xS) → Coin) × (Fin (D.flock.V.k xS) → Coin) =>
            LogColl ((run H P' D.flock.V xS () b.1).vLog ++ (run H P' D.flock.V xS () (forkCoins jF b.1 b.2)).vLog)),
          ← Pr_restrict_pair emb hinj (fun ρ1 ρ2 =>
            LogColl ((run H P' D.flock.V xS () ρ1).vLog ++ (run H P' D.flock.V xS () (forkCoins jF ρ1 ρ2)).vLog)),
          ← Pr_prod_eq_Ex (fun s : (Fin (D.V.k x) → Coin) × (Fin (D.V.k x) → Coin) =>
            SColl H D P x ω (emb jF) (forkOn (D.rounds x u) R0 s.1) s.2)]
        refine Pr_mono fun s hcol => ?_
        have hlt : ∀ i, (emb i).val < (emb jF).val ↔ i.val < jF.val := fun i =>
          hmono.lt_iff_lt (a := i) (b := jF)
        have e1 : (fun i => s.1 (emb i)) = fun i => forkOn (D.rounds x u) R0 s.1 (emb i) :=
          funext fun i => by simp [forkOn, hS i]
        have e2 : forkCoins jF (fun i => s.1 (emb i)) (fun i => s.2 (emb i)) =
            fun i => forkCoins (emb jF) (forkOn (D.rounds x u) R0 s.1) s.2 (emb i) :=
          funext fun i => by by_cases h : i.val < jF.val <;> simp [forkCoins, forkOn, hS i, h, hlt]
        rw [e2, e1] at hcol
        have v1 := (hview (forkOn (D.rounds x u) R0 s.1) fun t ht _ => hag0 s.1 t ht).2
        have v2 := (hview (forkCoins (emb jF) (forkOn (D.rounds x u) R0 s.1) s.2) fun t ht hb => by
          have : t.val < (emb jF).val := hb jF
          simp [forkCoins, this, hag0 s.1 t ht]).2
        exact hcol.mono fun q hq => (List.mem_append.1 hq).elim (fun h => List.mem_append_left _ (v1 q h))
          fun h => List.mem_append_right _ (v2 q h)
      refine (Finset.sum_le_sum fun jF _ => hterm jF).trans ?_
      rw [← Finset.sum_image (s := Finset.univ) (g := emb)
        (f := fun j => Ex fun σ => Pr fun R2 => SColl H D P x ω j (forkOn (D.rounds x u) R0 σ) R2)
        (fun a _ b _ h => hinj h)]
      exact Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ _) fun j _ _ => Ex_nonneg fun _ => Pr_nonneg _
    · rw [show PrB H D.flock.V P' xS = 0 from dite_eq_right hkF, mul_zero]
      exact Ex_nonneg fun _ => sum_nonneg fun _ _ => Pr_nonneg _
  have hκ : Claims.Flock.κ D.flock xS ≤ κF D x := by
    unfold Claims.Flock.κ κF
    rw [hw, hc, hk]
    linarith
  have hEx : (Ex fun σ => collAt H D P x ω (forkOn (D.rounds x u) R0 σ)) =
      (Ex fun σ => ∑ j, Pr fun R2 => SColl H D P x ω j (forkOn (D.rounds x u) R0 σ) R2) / D.V.k x := by
    simp only [collAt, Ex_div]
  have hγ : Claims.Flock.γ D.flock xS * PrB H D.flock.V P' xS ≤
      γF D x * Ex fun σ => collAt H D P x ω (forkOn (D.rounds x u) R0 σ) := by
    rw [hEx]
    have hY : 0 ≤ Ex fun σ => ∑ j, Pr fun R2 => SColl H D P x ω j (forkOn (D.rounds x u) R0 σ) R2 :=
      Ex_nonneg fun _ => sum_nonneg fun _ _ => Pr_nonneg _
    by_cases hkF : 0 < D.flock.V.k xS
    · have hkS : (0 : ℝ) < D.V.k x := by exact_mod_cast lt_of_lt_of_le hkF hkle
      have hg : Claims.Flock.γ D.flock xS * D.V.k x ≤ γF D x * D.flock.V.k xS := by
        unfold Claims.Flock.γ γF
        exact γFlock_mul_le hkle hwt
      have hP := PrB_nonneg H D.flock.V P' xS
      calc Claims.Flock.γ D.flock xS * PrB H D.flock.V P' xS
          = Claims.Flock.γ D.flock xS * D.V.k x * PrB H D.flock.V P' xS / D.V.k x := by field_simp
        _ ≤ γF D x * D.flock.V.k xS * PrB H D.flock.V P' xS / D.V.k x := by gcongr
        _ = γF D x * (D.flock.V.k xS * PrB H D.flock.V P' xS) / D.V.k x := by ring
        _ ≤ γF D x * (Ex fun σ => ∑ j, Pr fun R2 => SColl H D P x ω j (forkOn (D.rounds x u) R0 σ) R2) /
            D.V.k x := by gcongr; exact γF_nonneg D x
        _ = _ := by ring
    · rw [show PrB H D.flock.V P' xS = 0 from dite_eq_right hkF, mul_zero]
      exact mul_nonneg (γF_nonneg D x) (div_nonneg hY (Nat.cast_nonneg _))
  calc Pr (fun s : (Fin (D.V.k x) → Coin) × D.UnitCoins x =>
        (run H P D.V x ω (forkOn (D.rounds x u) R0 s.1)).out = .acc ∧
          unitWit H D decode x (run H P D.V x ω) u (forkOn (D.rounds x u) R0 s.1) s.2 = none)
      ≤ Pr (fun s : (Fin (D.V.k x) → Coin) × D.UnitCoins x =>
          (run H P' D.flock.V xS () fun i => s.1 (emb i)).out = .acc ∧
            ¬ (Claims.Flock.R D.flock.parse).ext xS
              ((Claims.Flock.E D.flock decode).eval H P' xS () (fun i => s.1 (emb i))
                fun i j t => s.2 (Fin.cast hw i) (Fin.cast hc j) (emb t)).1) :=
        Pr_mono fun s h => himp s.1 s.2 h.1 h.2
    _ = PrBad H D.flock.V (Claims.Flock.R D.flock.parse) (Claims.Flock.E D.flock decode) xS P' := by
        rw [Pr_restrict_waits emb hinj hw hc (fun ρF c => (run H P' D.flock.V xS () ρF).out = .acc ∧
          ¬ (Claims.Flock.R D.flock.parse).ext xS ((Claims.Flock.E D.flock decode).eval H P' xS () ρF c).1)]
        unfold PrBad
        exact (Pr_unit _).symm
    _ ≤ Claims.Flock.κ D.flock xS + Real.sqrt (Claims.Flock.γ D.flock xS * PrB H D.flock.V P' xS) :=
        hKS decode hdec xS Unit P'
    _ ≤ κF D x + Real.sqrt (γF D x * Ex fun σ => collAt H D P x ω (forkOn (D.rounds x u) R0 σ)) :=
        add_le_add hκ (Real.sqrt_le_sqrt hγ)

/-! ## The selected unit's session, resampled -/

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col] in
theorem selRounds_some {D : Deploy Coin Col T} {x : Msg} {sel : List ℕ → Option (Fin (nPU D x))}
    {R : Fin (D.V.k x) → Coin} {u : ℕ} (h : selUnit D x sel R = some u) : selRounds D x sel R = D.rounds x u := by
  unfold selRounds
  rw [h]

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col] in
theorem selRounds_none {D : Deploy Coin Col T} {x : Msg} {sel : List ℕ → Option (Fin (nPU D x))}
    {R : Fin (D.V.k x) → Coin} (h : selUnit D x sel R = none) : selRounds D x sel R = ∅ := by
  unfold selRounds
  rw [h]

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col] in
/-- Resampling the selected unit's rounds keeps them (`DrawFixed`). -/
theorem selRounds_resample {D : Deploy Coin Col T} (hdraw : DrawFixed D) {x : Msg}
    (sel : List ℕ → Option (Fin (nPU D x))) (r d σ : Fin (D.V.k x) → Coin) :
    selRounds D x sel (forkOn (D.stage2 x) r (forkOn (selRounds D x sel (forkOn (D.stage2 x) r d)) d σ)) =
      selRounds D x sel (forkOn (D.stage2 x) r d) := by
  cases h : selUnit D x sel (forkOn (D.stage2 x) r d) with
  | none => rw [selRounds_none h, forkOn_empty, selRounds_none h]
  | some u =>
    rw [selRounds_some h, forkOn_sub (hdraw x _ u (selUnit_mem h)).1]
    exact selRounds_some (selUnit_forkOn hdraw h σ)

/-- The probability over a unit's `E^rep` coins that the run at `R` accepts and the unit the selection picks finds no
witness. -/
noncomputable def selMiss (H : Msg → Msg) (D : Deploy Coin Col T) (decode : List (ℕ × Col) → List T) {Ω : Type}
    (P : Prover Ω) (x : Msg) (ω : Ω) (sel : List ℕ → Option (Fin (nPU D x))) (R : Fin (D.V.k x) → Coin) : ℝ :=
  Pr fun ρ0 : D.UnitCoins x => (run H P D.V x ω R).out = .acc ∧
    ∃ u, selUnit D x sel R = some u ∧ unitWit H D decode x (run H P D.V x ω) u R ρ0 = none

omit [Nonempty Coin] [DecidableEq Col] [Zero Col] in
theorem selMiss_none (H : Msg → Msg) (D : Deploy Coin Col T) (decode : List (ℕ × Col) → List T) {Ω : Type}
    (P : Prover Ω) (x : Msg) (ω : Ω) (sel : List ℕ → Option (Fin (nPU D x))) {R : Fin (D.V.k x) → Coin}
    (h : selUnit D x sel R = none) : selMiss H D decode P x ω sel R = 0 := by
  unfold selMiss
  rw [h]
  exact (Pr_congr fun _ => ⟨fun ⟨_, _, hu, _⟩ => Option.some_ne_none _ hu.symm, False.elim⟩).trans Pr_false

set_option synthInstance.maxSize 1024 in
/-- **At the selected unit's coins**, the others fixed: `fiber_bound`. -/
theorem selMiss_fiber {H : Msg → Msg} {D : Deploy Coin Col T} (hKS : Claims.Flock.KS H D.flock)
    (hdraw : DrawFixed D) (hruns : SessionRuns H D) (hnum : InnerNumbers D) (decode : List (ℕ × Col) → List T)
    (hdec : Flock.ListWithin D.flock.len D.flock.enc D.flock.θ decode) (x : Msg) {Ω : Type} (P : Prover Ω) (ω : Ω)
    (sel : List ℕ → Option (Fin (nPU D x))) {R0 : Fin (D.V.k x) → Coin} {u : ℕ}
    (h : selUnit D x sel R0 = some u) :
    (Ex fun σ => selMiss H D decode P x ω sel (forkOn (D.rounds x u) R0 σ)) ≤
      κF D x + Real.sqrt (γF D x * Ex fun σ => collAt H D P x ω (forkOn (D.rounds x u) R0 σ)) := by
  have e : ∀ σ, selMiss H D decode P x ω sel (forkOn (D.rounds x u) R0 σ) =
      Pr fun ρ0 : D.UnitCoins x => (run H P D.V x ω (forkOn (D.rounds x u) R0 σ)).out = .acc ∧
        unitWit H D decode x (run H P D.V x ω) u (forkOn (D.rounds x u) R0 σ) ρ0 = none := by
    intro σ
    unfold selMiss
    rw [selUnit_forkOn hdraw h σ]
    refine Pr_congr fun ρ0 => ⟨fun ⟨ha, u', hu', hw⟩ => ?_, fun ⟨ha, hw⟩ => ⟨ha, u, rfl, hw⟩⟩
    obtain rfl := Option.some.inj hu'
    exact ⟨ha, hw⟩
  simp only [e]
  rw [← Pr_prod_eq_Ex (fun s : (Fin (D.V.k x) → Coin) × D.UnitCoins x =>
    (run H P D.V x ω (forkOn (D.rounds x u) R0 s.1)).out = .acc ∧
      unitWit H D decode x (run H P D.V x ω) u (forkOn (D.rounds x u) R0 s.1) s.2 = none)]
  exact fiber_bound hKS hruns hnum decode hdec x P ω R0 u (selUnit_mem h)

set_option synthInstance.maxSize 1024 in
/-- At second-stage coins `d`, resampling the selected unit's rounds: `selMiss_fiber`, or `0` when none is picked. -/
theorem resample_bound {H : Msg → Msg} {D : Deploy Coin Col T} (hKS : Claims.Flock.KS H D.flock)
    (hdraw : DrawFixed D) (hruns : SessionRuns H D) (hnum : InnerNumbers D) (hε : ∀ x, 0 ≤ D.εtable x)
    (decode : List (ℕ × Col) → List T) (hdec : Flock.ListWithin D.flock.len D.flock.enc D.flock.θ decode) (x : Msg)
    {Ω : Type} (P : Prover Ω) (ω : Ω) (sel : List ℕ → Option (Fin (nPU D x))) (r d : Fin (D.V.k x) → Coin) :
    (Ex fun σ => selMiss H D decode P x ω sel
        (forkOn (D.stage2 x) r (forkOn (selRounds D x sel (forkOn (D.stage2 x) r d)) d σ))) ≤
      κF D x + Real.sqrt (γF D x * Ex fun σ => collAt H D P x ω
        (forkOn (D.stage2 x) r (forkOn (selRounds D x sel (forkOn (D.stage2 x) r d)) d σ))) := by
  cases h : selUnit D x sel (forkOn (D.stage2 x) r d) with
  | none =>
    rw [selRounds_none h]
    simp only [forkOn_empty]
    rw [selMiss_none H D decode P x ω sel h, Ex_const (α := Fin (D.V.k x) → Coin) (0 : ℝ)]
    exact add_nonneg (κF_nonneg D x (hε x)) (Real.sqrt_nonneg _)
  | some u =>
    rw [selRounds_some h]
    simp only [forkOn_sub (hdraw x _ u (selUnit_mem h)).1]
    exact selMiss_fiber hKS hdraw hruns hnum decode hdec x P ω sel h

set_option synthInstance.maxSize 1024 in
/-- **`SessionKS` from Flock's KS**: `β(ω, r)` is `B`'s collision probability averaged over the second stage, which
averages to `Pr[B]` (`Ex_collAt`); at `(ω, r)`, the selected unit's `E^rep` coins are one coordinate of the units'
(`Pr_pi_eval`), and resampling its rounds (`Ex_resample`, `selRounds_resample`) gives `resample_bound` at each
second-stage `d`, averaged by Jensen's inequality (`Ex_sqrt_le`). -/
theorem SessionKS_of {H : Msg → Msg} {D : Deploy Coin Col T} (hKS : Claims.Flock.KS H D.flock)
    (hdraw : DrawFixed D) (hruns : SessionRuns H D) (hnum : InnerNumbers D) (hε : ∀ x, 0 ≤ D.εtable x) :
    SessionKS H D := by
  intro decode hdec x Ω _ _ P
  refine ⟨fun ω r => Ex fun d => collAt H D P x ω (forkOn (D.stage2 x) r d),
    fun ω r => Ex_nonneg fun d => collAt_nonneg H D P x ω _, le_of_eq ?_, fun ω r sel => ?_⟩
  · rw [Ex_prod, ← Ex_collAt H D P x, Ex_prod]
    congr 1
    funext ω
    exact (Ex_prod (fun p : (Fin (D.V.k x) → Coin) × (Fin (D.V.k x) → Coin) =>
      collAt H D P x ω (forkOn (D.stage2 x) p.1 p.2))).symm.trans (Ex_forkOn _ _)
  have hS := selRounds_resample hdraw sel r
  have hres := Ex_resample (fun d => selRounds D x sel (forkOn (D.stage2 x) r d)) hS
    fun d => collAt H D P x ω (forkOn (D.stage2 x) r d)
  have hres' := Ex_resample (fun d => selRounds D x sel (forkOn (D.stage2 x) r d)) hS
    fun d => selMiss H D decode P x ω sel (forkOn (D.stage2 x) r d)
  beta_reduce at hres hres'
  have step1 : Pr (fun p : (Fin (D.V.k x) → Coin) × (Fin (nPU D x) → D.UnitCoins x) =>
      (run H P D.V x ω (forkOn (D.stage2 x) r p.1)).out = .acc ∧
        ∃ i, sel (D.checked x (forkOn (D.stage2 x) r p.1)) = some i ∧
          ∃ u, (D.checked x (forkOn (D.stage2 x) r p.1))[i.val]? = some u ∧
            unitWit H D decode x (run H P D.V x ω) u (forkOn (D.stage2 x) r p.1) (p.2 i) = none) ≤
      Ex fun d => selMiss H D decode P x ω sel (forkOn (D.stage2 x) r d) := by
    rw [Pr_prod_eq_Ex]
    refine Ex_mono fun d => ?_
    cases hi : sel (D.checked x (forkOn (D.stage2 x) r d)) with
    | none =>
      refine (Pr_mono fun ρs h => ?_).trans (Pr_false.le.trans (Pr_nonneg _))
      obtain ⟨_, i, hi', _⟩ := h
      exact Option.some_ne_none _ (hi.symm.trans hi').symm
    | some i =>
      refine (Pr_mono fun ρs h => ?_).trans_eq (Pr_pi_eval i fun ρ0 : D.UnitCoins x =>
        (run H P D.V x ω (forkOn (D.stage2 x) r d)).out = .acc ∧
          ∃ u, selUnit D x sel (forkOn (D.stage2 x) r d) = some u ∧
            unitWit H D decode x (run H P D.V x ω) u (forkOn (D.stage2 x) r d) ρ0 = none)
      obtain ⟨ha, i', hi', u, hu, hw⟩ := h
      obtain rfl := Option.some.inj (hi.symm.trans hi')
      refine ⟨ha, u, ?_, hw⟩
      unfold selUnit
      rw [hi]
      exact hu
  calc _ ≤ Ex fun d => selMiss H D decode P x ω sel (forkOn (D.stage2 x) r d) := step1
    _ = Ex fun d => Ex fun σ => selMiss H D decode P x ω sel
          (forkOn (D.stage2 x) r (forkOn (selRounds D x sel (forkOn (D.stage2 x) r d)) d σ)) := hres'.symm
    _ ≤ Ex fun d => κF D x + Real.sqrt (γF D x * Ex fun σ => collAt H D P x ω
          (forkOn (D.stage2 x) r (forkOn (selRounds D x sel (forkOn (D.stage2 x) r d)) d σ))) :=
        Ex_mono fun d => resample_bound hKS hdraw hruns hnum hε decode hdec x P ω sel r d
    _ = κF D x + Ex fun d => Real.sqrt (γF D x * Ex fun σ => collAt H D P x ω
          (forkOn (D.stage2 x) r (forkOn (selRounds D x sel (forkOn (D.stage2 x) r d)) d σ))) := by
        rw [Ex_add, Ex_const]
    _ ≤ κF D x + Real.sqrt (Ex fun d => γF D x * Ex fun σ => collAt H D P x ω
          (forkOn (D.stage2 x) r (forkOn (selRounds D x sel (forkOn (D.stage2 x) r d)) d σ))) :=
        add_le_add le_rfl (Ex_sqrt_le _ fun d =>
          mul_nonneg (γF_nonneg D x) (Ex_nonneg fun σ => collAt_nonneg H D P x ω _))
    _ = κF D x + Real.sqrt (γF D x * Ex fun d => collAt H D P x ω (forkOn (D.stage2 x) r d)) := by
        rw [Ex_mul_left, hres]

end Session

end Claims.Sampling
