import Proofs.Flock.Claims.Sampling.Runs
import Proofs.Flock.Claims.Sampling.Session

/-!
# The sampling extractor's expected reruns

`E_runs_exp`: **`ExtractRunsExpLe`** at the claims block's extractor `E`, at every decoder, at
`1 + k + k²·(1 + n_PU·m′·k′)`, when the draw precedes the sessions (`DrawFixed`). Each wait amortizes against the
acceptance probability it waits on (`waitRuns_exp`, `accExp_le`):

* the judged run accepts with probability `ε`, and the outer wait then makes `k/ε` reruns in expectation;
* its accepted runs weigh as runs given acceptance (`waitAccC_sum`); fresh on the second stage, a run `r`'s runs
  accept with probability `ε₂(r)` (`Ex_acc_forkOn`), and the inner wait at `r` makes `k/ε₂(r)`;
* at each accepted inner run and each checked unit's position, the unit's `m′` waits make `m′·k′/ε_u`, where `ε_u` is
  the acceptance probability fresh on the unit's rounds. Resampling those rounds keeps the checked units (`DrawFixed`,
  `Ex_resample`), so acceptance there weighs as `ε_u` (`unit_exp_le`).

`level_ex` is one wait and a body at each kept run, used at the outer and the inner wait.
-/

namespace Claims.Sampling

open Claims

/-! ## Sums along a wait's kept runs -/

theorem sum_zip_finRange {α : Type} : ∀ (n : ℕ) (F : Fin n × α → ℝ) (l : List α),
    (((List.finRange n).zip l).map F).sum = ∑ i : Fin n, (l[i.val]?.map fun a => F (i, a)).getD 0
  | 0, _, _ => by simp
  | _ + 1, _, [] => by simp
  | n + 1, F, a :: l => by
    rw [List.finRange_succ, List.zip_cons_cons, List.map_cons, List.sum_cons, List.zip_map_left, List.map_map,
      sum_zip_finRange n (F ∘ Prod.map Fin.succ id) l, Fin.sum_univ_succ]
    simp

theorem sum_zip_snd_le {α β : Type} (f : α → ℝ) (hf : ∀ a, 0 ≤ f a) :
    ∀ (l₁ : List β) (l₂ : List α), ((l₁.zip l₂).map fun q => f q.2).sum ≤ (l₂.map f).sum
  | [], l₂ => by
    simpa using List.sum_nonneg fun y hy => by
      obtain ⟨a, -, rfl⟩ := List.mem_map.1 hy
      exact hf a
  | _ :: _, [] => by simp
  | _ :: l₁, a :: l₂ => by
    simp only [List.zip_cons_cons, List.map_cons, List.sum_cons]
    exact add_le_add le_rfl (sum_zip_snd_le f hf l₁ l₂)

theorem accSum_map {R R' : Type} (run : R → Trace) (k : ℕ) (g : R → ℝ) (f : R' → R) :
    ∀ (l : List R') (a : ℕ), accSum run k g (l.map f) a = accSum (run ∘ f) k (g ∘ f) l a
  | [], _ => rfl
  | r :: l, a => by simp only [List.map_cons, accSum, accSum_map run k g f l, Function.comp]

theorem accSum_nonneg {R : Type} (run : R → Trace) (k : ℕ) {g : R → ℝ} (hg : ∀ r, 0 ≤ g r) :
    ∀ (l : List R) (a : ℕ), 0 ≤ accSum run k g l a
  | [], _ => le_rfl
  | r :: l, a => by
    simp only [accSum]
    split_ifs
    · exact le_rfl
    · exact add_nonneg (hg r) (accSum_nonneg run k hg l _)
    · exact accSum_nonneg run k hg l _

theorem waitExp_nonneg {D : Type} [Fintype D] (run : D → Trace) (k K a : ℕ) : 0 ≤ waitExp run k K a :=
  Ex_nonneg fun _ => Nat.cast_nonneg _

theorem accExp_nonneg {D : Type} [Fintype D] (run : D → Trace) (k : ℕ) {g : D → ℝ} (hg : ∀ d, 0 ≤ g d) (K a : ℕ) :
    0 ≤ accExp run k g K a :=
  Ex_nonneg fun _ => accSum_nonneg run k hg _ _

/-- A wait and the programs at its kept runs, indexed, rerun as the wait does and then as each program. Stated at
every `k`, so that the kernel never meets the deployed `k`'s `List.finRange`. -/
theorem wait_flat_runs {R γ : Type} (H : Msg → Msg) (run : R → Trace) (k : ℕ) (rs : List R)
    (f : Fin k × (R × Seen) → RP R (List γ)) :
    (((waitAccC k rs []).bind fun l => flatMapRP f ((List.finRange k).zip l)).eval H run).2.2 =
      waitRuns run k rs 0 + (((List.finRange k).zip ((waitAccC k rs []).eval H run).1).map fun q =>
        ((f q).eval H run).2.2).sum := by
  rw [(RP.eval_bind _ _ _ _).2, waitAccC_runs_eq, flatMapRP_runs, List.length_nil]

/-- **The outer wait and what follows it**, as reals: the wait's reruns and each kept run's body bound `B`. Stated at
every `k`, so that the kernel never meets the deployed `k`'s `List.finRange`. -/
theorem outer_runs_le_sum {R γ β : Type} (H : Msg → Msg) (run : R → Trace) (k : ℕ) (rs : List R)
    (body : Fin k × (R × Seen) → RP R (List γ)) (fin : List γ → β) (B : Fin k × (R × Seen) → ℝ)
    (hB : ∀ jo, (((body jo).eval H run).2.2 : ℝ) ≤ B jo) :
    ((((waitAccC k rs []).bind fun outer =>
        (flatMapRP body ((List.finRange k).zip outer)).bind fun found => RP.ret (fin found)).eval H run).2.2 : ℝ) ≤
      waitRuns run k rs 0 + (((List.finRange k).zip ((waitAccC k rs []).eval H run).1).map B).sum := by
  rw [(RP.eval_bind _ _ _ _).2, (RP.eval_bind _ _ _ _).2, waitAccC_runs_eq, flatMapRP_runs]
  simp only [RP.eval, List.length_nil, Nat.add_zero, Nat.cast_add, Nat.cast_list_sum, List.map_map]
  exact add_le_add le_rfl (List.sum_le_sum fun jo _ => hB jo)

/-- **One wait and a body at each kept run, in expectation**: over the wait's `K` draws (each `f` of a uniform `D'`)
and the bodies' coins `X`, the reruns are at most the wait's expected reruns and its kept runs' weights `G`, when each
body averages at most `G` of its run, at every position. -/
theorem level_ex {D' R X : Type} [Fintype D'] [Nonempty D'] [Fintype X] [Nonempty X] (H : Msg → Msg)
    (run : R → Trace) (f : D' → R) (k K : ℕ) (Bf : Fin k → R → X → ℝ) (G : R → ℝ) (hB : ∀ j r, Ex (Bf j r) ≤ G r)
    (hG : ∀ r, 0 ≤ G r) :
    Ex (fun c : (Fin K → D') × X => (waitRuns run k ((List.finRange K).map fun l => f (c.1 l)) 0 : ℝ) +
        (((List.finRange k).zip ((waitAccC k ((List.finRange K).map fun l => f (c.1 l)) []).eval H run).1).map
          fun jo => Bf jo.1 jo.2.1 c.2).sum) ≤
      waitExp (run ∘ f) k K 0 + accExp (run ∘ f) k (G ∘ f) K 0 := by
  have hstep : ∀ σ : Fin K → D', (Ex fun ξ : X =>
      (waitRuns run k ((List.finRange K).map fun l => f (σ l)) 0 : ℝ) +
        (((List.finRange k).zip ((waitAccC k ((List.finRange K).map fun l => f (σ l)) []).eval H run).1).map
          fun jo => Bf jo.1 jo.2.1 ξ).sum) ≤
      (waitRuns (run ∘ f) k ((List.finRange K).map σ) 0 : ℝ) +
        accSum (run ∘ f) k (G ∘ f) ((List.finRange K).map σ) 0 := by
    intro σ
    have hw : waitRuns run k ((List.finRange K).map fun l => f (σ l)) 0 =
        waitRuns (run ∘ f) k ((List.finRange K).map σ) 0 := by
      rw [← waitRuns_map, List.map_map]
      rfl
    rw [Ex_add, Ex_const, Ex_listSum, hw]
    refine add_le_add le_rfl ?_
    calc (((List.finRange k).zip ((waitAccC k ((List.finRange K).map fun l => f (σ l)) []).eval H run).1).map
          fun jo => Ex (Bf jo.1 jo.2.1)).sum
        ≤ (((List.finRange k).zip ((waitAccC k ((List.finRange K).map fun l => f (σ l)) []).eval H run).1).map
          fun jo => G jo.2.1).sum := List.sum_le_sum fun jo _ => hB jo.1 jo.2.1
      _ ≤ (((waitAccC k ((List.finRange K).map fun l => f (σ l)) []).eval H run).1.map fun p => G p.1).sum :=
          sum_zip_snd_le (fun p : R × Seen => G p.1) (fun p => hG p.1) _ _
      _ = accSum run k G ((List.finRange K).map fun l => f (σ l)) 0 := by
          rw [waitAccC_sum]
          simp
      _ = accSum (run ∘ f) k (G ∘ f) ((List.finRange K).map σ) 0 := by
          rw [← accSum_map, List.map_map]
          rfl
  rw [Ex_prod]
  refine (Ex_mono hstep).trans_eq ?_
  rw [Ex_add]
  rfl

/-- `level_ex`'s bound against the wait's acceptance probability: at most `k` reruns for the wait, and `k` times
the kept runs' weights given acceptance. -/
theorem level_amortized {D R : Type} [Fintype D] [Nonempty D] (run : R → Trace) (f : D → R) (k K : ℕ) (G : R → ℝ)
    (hG : ∀ r, 0 ≤ G r) :
    Pr (fun d => (run (f d)).out = .acc) * (waitExp (run ∘ f) k K 0 + accExp (run ∘ f) k (G ∘ f) K 0) ≤
      k + k * Ex (fun d => (if (run (f d)).out = .acc then (1 : ℝ) else 0) * G (f d)) := by
  have h1 := waitRuns_exp (run ∘ f) k K 0 (Nat.zero_le _)
  have h2 := accExp_le (run ∘ f) k (G ∘ f) (fun d => hG (f d)) K 0 (Nat.zero_le _)
  simp only [Function.comp_apply, Nat.cast_zero, sub_zero] at h1 h2
  rw [mul_add]
  exact add_le_add h1 h2

/-! ## The extractor's reruns, slice by slice -/

section RunsExp

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]

/-- A checked unit's waits' reruns at coins `r` and its `E^rep` coins `ρ`: Flock's `m′` waits of `K′` draws, fresh on
the unit's rounds. -/
noncomputable def unitWaits (D : Deploy Coin Col T) (x : Msg) (run' : (Fin (D.V.k x) → Coin) → Trace) (u : ℕ × ℕ)
    (r : Fin (D.V.k x) → Coin) (ρ : D.UnitCoins x) : ℝ :=
  ∑ a, (waitRuns (run' ∘ forkOn (D.rounds x u) r) (D.innerK x) ((List.finRange (D.innerCap x)).map (ρ a)) 0 : ℝ)

/-- Position `i`'s unit's waits, `0` when fewer units are checked at `r`. -/
noncomputable def posWaits (D : Deploy Coin Col T) (x : Msg) (run' : (Fin (D.V.k x) → Coin) → Trace)
    (i : Fin (nPU D x)) (r : Fin (D.V.k x) → Coin) (ρ : D.UnitCoins x) : ℝ :=
  (((D.checked x r)[i.val]?).map fun u => unitWaits D x run' u r ρ).getD 0

/-- Every position's expected unit waits at an accepted inner run `r`. -/
noncomputable def innerW (D : Deploy Coin Col T) (x : Msg) (run' : (Fin (D.V.k x) → Coin) → Trace)
    (r : Fin (D.V.k x) → Coin) : ℝ :=
  ∑ i, Ex fun ρ : D.UnitCoins x => posWaits D x run' i r ρ

/-- The bound on an outer body at outer position `j` and run `r`, on its inner wait's coins `d` and the units' `ξ`. -/
noncomputable def innerB (H : Msg → Msg) (D : Deploy Coin Col T) (x : Msg) (run' : (Fin (D.V.k x) → Coin) → Trace)
    (j : Fin (D.k x)) (r : Fin (D.V.k x) → Coin) (d : Fin (D.cap x) → Fin (D.V.k x) → Coin)
    (ξ : Fin (D.k x) × Fin (D.k x) × Fin (nPU D x) → D.UnitCoins x) : ℝ :=
  (waitRuns run' (D.k x) ((List.finRange (D.cap x)).map fun l => forkOn (D.stage2 x) r (d l)) 0 : ℝ) +
    (((List.finRange (D.k x)).zip ((waitAccC (D.k x) ((List.finRange (D.cap x)).map fun l =>
      forkOn (D.stage2 x) r (d l)) []).eval H run').1).map fun lo =>
        ∑ i, posWaits D x run' i lo.2.1 (ξ (j, lo.1, i))).sum

/-- An outer body's weight given its run `r`: the inner wait's expected reruns fresh on the second stage, and its kept
runs' unit waits. -/
noncomputable def outerG (D : Deploy Coin Col T) (x : Msg) (run' : (Fin (D.V.k x) → Coin) → Trace)
    (r : Fin (D.V.k x) → Coin) : ℝ :=
  waitExp (run' ∘ forkOn (D.stage2 x) r) (D.k x) (D.cap x) 0 +
    accExp (run' ∘ forkOn (D.stage2 x) r) (D.k x) (innerW D x run' ∘ forkOn (D.stage2 x) r) (D.cap x) 0

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col] in
theorem unitWaits_nonneg (D : Deploy Coin Col T) (x : Msg) (run' : (Fin (D.V.k x) → Coin) → Trace) (u : ℕ × ℕ)
    (r : Fin (D.V.k x) → Coin) (ρ : D.UnitCoins x) : 0 ≤ unitWaits D x run' u r ρ :=
  Finset.sum_nonneg fun _ _ => Nat.cast_nonneg _

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col] in
theorem posWaits_nonneg (D : Deploy Coin Col T) (x : Msg) (run' : (Fin (D.V.k x) → Coin) → Trace)
    (i : Fin (nPU D x)) (r : Fin (D.V.k x) → Coin) (ρ : D.UnitCoins x) : 0 ≤ posWaits D x run' i r ρ := by
  unfold posWaits
  cases (D.checked x r)[i.val]? with
  | none => exact le_rfl
  | some u => exact unitWaits_nonneg D x run' u r ρ

omit [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col] in
theorem innerW_nonneg (D : Deploy Coin Col T) (x : Msg) (run' : (Fin (D.V.k x) → Coin) → Trace)
    (r : Fin (D.V.k x) → Coin) : 0 ≤ innerW D x run' r :=
  Finset.sum_nonneg fun i _ => Ex_nonneg fun ρ => posWaits_nonneg D x run' i r ρ

omit [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col] in
theorem outerG_nonneg (D : Deploy Coin Col T) (x : Msg) (run' : (Fin (D.V.k x) → Coin) → Trace)
    (r : Fin (D.V.k x) → Coin) : 0 ≤ outerG D x run' r :=
  add_nonneg (waitExp_nonneg _ _ _ _) (accExp_nonneg _ _ (fun _ => innerW_nonneg D x run' _) _ _)

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col] in
/-- A checked unit's extraction reruns as its waits do: Flock's `m′` waits on its session, at most. -/
theorem unitExt_runs_le_waits (H : Msg → Msg) (D : Deploy Coin Col T) (decode : List (ℕ × Col) → List T) (x : Msg)
    (s : AuditStatement) (u : ℕ × ℕ) (r : Fin (D.V.k x) → Coin) (v : Seen) (ρ : D.UnitCoins x)
    (run' : (Fin (D.V.k x) → Coin) → Trace) :
    (((unitExt D decode x s u r v ρ).eval H run').2.2 : ℝ) ≤ unitWaits D x run' u r ρ := by
  unfold unitExt
  split
  · simp only [RP.eval, Nat.cast_zero]
    exact unitWaits_nonneg D x run' u r ρ
  · rename_i xS _ _
    rw [(RP.eval_bind H run' _ _).2]
    have hw := firstWait_runs_le_sum H run' (D.innerK x) (D.sessionCols x u)
      (Flock.assemble D.flock.parse D.flock.table decode D.flock.read xS) (Flock.check D.flock.parse xS)
      ((List.finRange (D.innerWaits x)).map fun i =>
        (List.finRange (D.innerCap x)).map fun j => forkOn (D.rounds x u) r (ρ i j))
    simp only [List.map_map, Function.comp_def] at hw
    rw [← Fin.sum_univ_def] at hw
    have hw' := (Nat.cast_le (α := ℝ)).2 hw
    rw [Nat.cast_sum] at hw'
    have hu : (∑ a, ((waitRuns run' (D.innerK x)
        ((List.finRange (D.innerCap x)).map fun j => forkOn (D.rounds x u) r (ρ a j)) 0 : ℕ) : ℝ)) =
        unitWaits D x run' u r ρ :=
      Finset.sum_congr rfl fun a _ => by
        rw [← waitRuns_map, List.map_map]
        rfl
    rw [hu] at hw'
    split
    · simpa only [RP.eval, Nat.add_zero] using hw'
    · rw [checkRevealed_runs H run' D.commit _, Nat.add_zero]
      exact hw'

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col] in
/-- At an accepted inner run `li`, every checked unit's extraction reruns at most its position's waits. -/
theorem innerBody_runs_le_waits (H : Msg → Msg) (D : Deploy Coin Col T) (decode : List (ℕ × Col) → List T) (x : Msg)
    (s : AuditStatement) (c : ECoins D x) (j : Fin (D.k x)) (li : Fin (D.k x) × ((Fin (D.V.k x) → Coin) × Seen))
    (run' : (Fin (D.V.k x) → Coin) → Trace) :
    (((innerBody D decode x s c j li).eval H run').2.2 : ℝ) ≤ ∑ i, posWaits D x run' i li.2.1 (c.2.2 (j, li.1, i)) := by
  unfold innerBody
  rw [flatMapRP_runs, Nat.cast_list_sum, List.map_map]
  refine (List.sum_le_sum fun iu _ => unitExt_runs_le_waits H D decode x s iu.2 li.2.1 li.2.2 _ run').trans_eq ?_
  exact sum_zip_finRange (nPU D x) (fun iu => unitWaits D x run' iu.2 li.2.1 (c.2.2 (j, li.1, iu.1))) _

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col] in
theorem outerBody_runs_le_innerB (H : Msg → Msg) (D : Deploy Coin Col T) (decode : List (ℕ × Col) → List T) (x : Msg)
    (s : AuditStatement) (c : ECoins D x) (jo : Fin (D.k x) × ((Fin (D.V.k x) → Coin) × Seen))
    (run' : (Fin (D.V.k x) → Coin) → Trace) :
    (((outerBody D decode x s c jo).eval H run').2.2 : ℝ) ≤ innerB H D x run' jo.1 jo.2.1 (c.2.1 jo.1) c.2.2 := by
  unfold outerBody innerB
  rw [wait_flat_runs, Nat.cast_add, Nat.cast_list_sum, List.map_map]
  exact add_le_add le_rfl (List.sum_le_sum fun li _ => innerBody_runs_le_waits H D decode x s c jo.1 li run')

omit [Inhabited Coin] [DecidableEq Col] [Zero Col] in
/-- An outer body's bound averages at most `outerG` of its run: `level_ex` at the inner wait, fresh on the second
stage. -/
theorem Ex_innerB_le (H : Msg → Msg) (D : Deploy Coin Col T) (x : Msg) (run' : (Fin (D.V.k x) → Coin) → Trace)
    (j : Fin (D.k x)) (r : Fin (D.V.k x) → Coin) :
    Ex (fun c2 : (Fin (D.k x) → Fin (D.cap x) → Fin (D.V.k x) → Coin) ×
        (Fin (D.k x) × Fin (D.k x) × Fin (nPU D x) → D.UnitCoins x) => innerB H D x run' j r (c2.1 j) c2.2) ≤
      outerG D x run' r := by
  have hslice : Ex (fun c2 : (Fin (D.k x) → Fin (D.cap x) → Fin (D.V.k x) → Coin) ×
        (Fin (D.k x) × Fin (D.k x) × Fin (nPU D x) → D.UnitCoins x) => innerB H D x run' j r (c2.1 j) c2.2) =
      Ex (fun c : (Fin (D.cap x) → Fin (D.V.k x) → Coin) ×
        (Fin (D.k x) × Fin (D.k x) × Fin (nPU D x) → D.UnitCoins x) => innerB H D x run' j r c.1 c.2) := by
    rw [Ex_prod, Ex_prod]
    exact Ex_pi_eval j fun d => Ex fun ξ => innerB H D x run' j r d ξ
  rw [hslice]
  unfold innerB outerG
  exact level_ex (D' := Fin (D.V.k x) → Coin) (R := Fin (D.V.k x) → Coin)
    (X := Fin (D.k x) × Fin (D.k x) × Fin (nPU D x) → D.UnitCoins x) H run' (forkOn (D.stage2 x) r) (D.k x)
    (D.cap x) (fun l r' ξ => ∑ i, posWaits D x run' i r' (ξ (j, l, i))) (innerW D x run')
    (fun l r' => by
      rw [Ex_finsum]
      exact le_of_eq (Finset.sum_congr rfl fun i _ =>
        Ex_pi_eval (j, l, i) fun ρ => posWaits D x run' i r' ρ))
    (innerW_nonneg D x run')

omit [Inhabited Coin] [DecidableEq Col] [Zero Col] in
/-- **One position's unit waits, given acceptance**: over the second stage `d` at `r`, acceptance times position
`i`'s expected unit waits is at most `m′·k′`. Resampling the unit's rounds keeps the checked units (`DrawFixed`), so
acceptance weighs as the unit's own acceptance probability, against which its waits amortize. -/
theorem unit_exp_le {D : Deploy Coin Col T} (hdraw : DrawFixed D) (x : Msg) (run' : (Fin (D.V.k x) → Coin) → Trace)
    (i : Fin (nPU D x)) (r : Fin (D.V.k x) → Coin) :
    Ex (fun d : Fin (D.V.k x) → Coin => (if (run' (forkOn (D.stage2 x) r d)).out = .acc then (1 : ℝ) else 0) *
        Ex fun ρ : D.UnitCoins x => posWaits D x run' i (forkOn (D.stage2 x) r d) ρ) ≤
      D.innerWaits x * D.innerK x := by
  let sel : List (ℕ × ℕ) → Option (Fin (nPU D x)) := fun _ => some i
  have hsel : ∀ R, selUnit D x sel R = (D.checked x R)[i.val]? := fun R => rfl
  rw [← Ex_resample (fun d => selRounds D x sel (forkOn (D.stage2 x) r d)) (selRounds_resample hdraw sel r)]
  have hk : (0 : ℝ) ≤ D.innerWaits x * D.innerK x := by positivity
  refine (Ex_mono fun d => ?_).trans_eq (Ex_const (α := Fin (D.V.k x) → Coin) _)
  cases h : selUnit D x sel (forkOn (D.stage2 x) r d) with
  | none =>
    rw [selRounds_none h]
    simp only [forkOn_empty]
    have hp : ∀ ρ, posWaits D x run' i (forkOn (D.stage2 x) r d) ρ = 0 := fun ρ => by
      unfold posWaits
      rw [← hsel, h]
      rfl
    simp only [hp, Ex_const, mul_zero]
    exact hk
  | some u =>
    rw [selRounds_some h]
    have hfix := hdraw x _ u (selUnit_mem h)
    simp only [forkOn_sub hfix.1]
    have hp : ∀ σ ρ, posWaits D x run' i (forkOn (D.rounds x u) (forkOn (D.stage2 x) r d) σ) ρ =
        unitWaits D x run' u (forkOn (D.stage2 x) r d) ρ := fun σ ρ => by
      unfold posWaits unitWaits
      rw [hfix.2 σ, ← hsel, h, Option.map_some, Option.getD_some, forkOn_forkOn']
    simp only [hp]
    have hE : (Ex fun ρ : D.UnitCoins x => unitWaits D x run' u (forkOn (D.stage2 x) r d) ρ) =
        D.innerWaits x * waitExp (run' ∘ forkOn (D.rounds x u) (forkOn (D.stage2 x) r d)) (D.innerK x)
          (D.innerCap x) 0 :=
      Ex_sum_pi (fun σ : Fin (D.innerCap x) → Fin (D.V.k x) → Coin =>
        (waitRuns (run' ∘ forkOn (D.rounds x u) (forkOn (D.stage2 x) r d)) (D.innerK x)
          ((List.finRange (D.innerCap x)).map σ) 0 : ℝ)) (D.innerWaits x)
    rw [hE, Ex_mul_right, Ex_ind]
    have hw := waitRuns_exp (run' ∘ forkOn (D.rounds x u) (forkOn (D.stage2 x) r d)) (D.innerK x) (D.innerCap x) 0
      (Nat.zero_le _)
    simp only [Function.comp_apply, Nat.cast_zero, sub_zero] at hw
    calc Pr (fun σ => (run' (forkOn (D.rounds x u) (forkOn (D.stage2 x) r d) σ)).out = .acc) *
          (D.innerWaits x * waitExp (run' ∘ forkOn (D.rounds x u) (forkOn (D.stage2 x) r d)) (D.innerK x)
            (D.innerCap x) 0)
        = D.innerWaits x * (Pr (fun σ => (run' (forkOn (D.rounds x u) (forkOn (D.stage2 x) r d) σ)).out = .acc) *
          waitExp (run' ∘ forkOn (D.rounds x u) (forkOn (D.stage2 x) r d)) (D.innerK x) (D.innerCap x) 0) := by
          ring
      _ ≤ D.innerWaits x * D.innerK x := mul_le_mul_of_nonneg_left hw (Nat.cast_nonneg _)

omit [Inhabited Coin] [DecidableEq Col] [Zero Col] in
/-- **An outer body's weight against the inner acceptance**: `ε₂(r) · outerG(r) ≤ k + k·n_PU·m′·k′`. -/
theorem outerG_amortized {D : Deploy Coin Col T} (hdraw : DrawFixed D) (x : Msg) (run' : (Fin (D.V.k x) → Coin) → Trace)
    (r : Fin (D.V.k x) → Coin) :
    Pr (fun d => (run' (forkOn (D.stage2 x) r d)).out = .acc) * outerG D x run' r ≤
      D.k x + D.k x * (nPU D x * (D.innerWaits x * D.innerK x)) := by
  refine (level_amortized run' (forkOn (D.stage2 x) r) (D.k x) (D.cap x) (innerW D x run')
    (innerW_nonneg D x run')).trans ?_
  refine add_le_add le_rfl (mul_le_mul_of_nonneg_left ?_ (Nat.cast_nonneg _))
  have hsum : (fun d : Fin (D.V.k x) → Coin =>
      (if (run' (forkOn (D.stage2 x) r d)).out = .acc then (1 : ℝ) else 0) *
        innerW D x run' (forkOn (D.stage2 x) r d)) =
      fun d => ∑ i, (if (run' (forkOn (D.stage2 x) r d)).out = .acc then (1 : ℝ) else 0) *
        Ex fun ρ : D.UnitCoins x => posWaits D x run' i (forkOn (D.stage2 x) r d) ρ := by
    funext d
    rw [innerW, Finset.mul_sum]
  rw [hsum, Ex_finsum]
  calc ∑ i, Ex (fun d : Fin (D.V.k x) → Coin =>
        (if (run' (forkOn (D.stage2 x) r d)).out = .acc then (1 : ℝ) else 0) *
          Ex fun ρ : D.UnitCoins x => posWaits D x run' i (forkOn (D.stage2 x) r d) ρ)
      ≤ ∑ _i : Fin (nPU D x), (D.innerWaits x * D.innerK x : ℝ) :=
        Finset.sum_le_sum fun i _ => unit_exp_le hdraw x run' i r
    _ = nPU D x * (D.innerWaits x * D.innerK x) := by
        rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]

omit [Inhabited Coin] [DecidableEq Col] [Zero Col] in
/-- The extractor's reruns at a statement that parses: the replay, and at an accepted judged run the outer wait and
each kept run's body bound. -/
theorem E_runs_le_outer (H : Msg → Msg) (D : Deploy Coin Col T) (decode : List (ℕ × Col) → List T) {x : Msg}
    {s : AuditStatement} (hs : D.parse x = some s) (run' : (Fin (D.V.k x) → Coin) → Trace) (r₀ : Fin (D.V.k x) → Coin)
    (c : (E D decode).C x) :
    ((((E D decode).ext x r₀ c).eval H run').2.2 : ℝ) ≤
      1 + (if (run' r₀).out = .acc then (1 : ℝ) else 0) *
        ((waitRuns run' (D.k x) ((List.finRange (D.cap x)).map c.1) 0 : ℝ) +
          ((outerList H D x run' c).map fun jo => innerB H D x run' jo.1 jo.2.1 (c.2.1 jo.1) c.2.2).sum) := by
  rw [E_ext_eq]
  simp only [RP.eval]
  split
  · rename_i hn
    rw [hs] at hn
    exact absurd hn (Option.some_ne_none s)
  · rename_i s' hs'
    obtain rfl : s = s' := Option.some.inj (hs.symm.trans hs')
    split
    · rename_i ha
      have ha' : (run' r₀).out = .acc := ha
      have h := outer_runs_le_sum H run' (D.k x) ((List.finRange (D.cap x)).map c.1) (outerBody D decode x s c)
        (fun found => some (assemble s found)) (fun jo => innerB H D x run' jo.1 jo.2.1 (c.2.1 jo.1) c.2.2)
        (fun jo => outerBody_runs_le_innerB H D decode x s c jo run')
      rw [ite_eq_left ha', one_mul, Nat.cast_add, Nat.cast_one, add_comm (1 : ℝ)]
      unfold outerList
      exact add_le_add h le_rfl
    · rename_i ha
      have ha' : ¬ (run' r₀).out = .acc := ha
      rw [ite_eq_right ha', zero_mul, add_zero, Nat.cast_add, Nat.cast_one]
      simp only [RP.eval, Nat.cast_zero, zero_add, le_refl]

omit [DecidableEq Col] [Zero Col] in
/-- **`ExtractRunsExpLe` at the sampling extractor**, at every decoder, when the draw precedes the sessions: at most
`1 + k + k²·(1 + n_PU·m′·k′)` reruns in expectation. -/
theorem E_runs_exp (H : Msg → Msg) (D : Deploy Coin Col T) (hdraw : DrawFixed D) (decode : List (ℕ × Col) → List T) :
    ExtractRunsExpLe H (E D decode)
      (fun x => 1 + D.k x + (D.k x : ℝ) ^ 2 * (1 + nPU D x * D.innerWaits x * D.innerK x)) := by
  intro x Ω P ω
  have hk0 : (0 : ℝ) ≤ D.k x := Nat.cast_nonneg _
  have hX0 : (0 : ℝ) ≤ nPU D x * D.innerWaits x * D.innerK x := by positivity
  cases hp : D.parse x with
  | none =>
    have h1 : ∀ s : (Fin (D.V.k x) → Coin) × (E D decode).C x,
        (((E D decode).eval H P x ω s.1 s.2).2.2 : ℝ) = 1 := by
      intro s
      simp only [Extractor.eval]
      rw [E_ext_eq]
      simp only [RP.eval]
      split
      · simp [RP.eval]
      · rename_i s' hs'
        rw [hp] at hs'
        exact absurd hs'.symm (Option.some_ne_none s')
    simp only [h1, Ex_const]
    nlinarith
  | some s =>
    let run' := run H P D.V x ω
    let Rout : (E D decode).C x → ℝ := fun c =>
      (waitRuns run' (D.k x) ((List.finRange (D.cap x)).map c.1) 0 : ℝ) +
        ((outerList H D x run' c).map fun jo => innerB H D x run' jo.1 jo.2.1 (c.2.1 jo.1) c.2.2).sum
    have hpt : ∀ s' : (Fin (D.V.k x) → Coin) × (E D decode).C x,
        (((E D decode).eval H P x ω s'.1 s'.2).2.2 : ℝ) ≤
          1 + (if (run' s'.1).out = .acc then (1 : ℝ) else 0) * Rout s'.2 :=
      fun s' => E_runs_le_outer H D decode hp run' s'.1 s'.2
    have hsplit : Ex (fun s' : (Fin (D.V.k x) → Coin) × (E D decode).C x =>
        1 + (if (run' s'.1).out = .acc then (1 : ℝ) else 0) * Rout s'.2) =
        1 + Pr (fun r => (run' r).out = .acc) * Ex Rout := by
      rw [Ex_add, Ex_const, Ex_prod]
      congr 1
      have hin : ∀ a : Fin (D.V.k x) → Coin, (Ex fun b : (E D decode).C x =>
          (if (run' a).out = .acc then (1 : ℝ) else 0) * Rout b) =
          (if (run' a).out = .acc then (1 : ℝ) else 0) * Ex Rout := fun a => Ex_mul_left _ Rout
      refine (congrArg Ex (funext hin)).trans ?_
      rw [Ex_mul_right, Ex_ind]
    have hout : Pr (fun r => (run' r).out = .acc) * Ex Rout ≤
        D.k x + D.k x * Ex (fun r => (if (run' r).out = .acc then (1 : ℝ) else 0) * outerG D x run' r) := by
      have hle := level_ex (D' := Fin (D.V.k x) → Coin) (R := Fin (D.V.k x) → Coin)
        (X := (Fin (D.k x) → Fin (D.cap x) → Fin (D.V.k x) → Coin) ×
          (Fin (D.k x) × Fin (D.k x) × Fin (nPU D x) → D.UnitCoins x)) H run' id (D.k x) (D.cap x)
        (fun j r c2 => innerB H D x run' j r (c2.1 j) c2.2) (outerG D x run')
        (fun j r => Ex_innerB_le H D x run' j r) (outerG_nonneg D x run')
      refine (mul_le_mul_of_nonneg_left hle (Pr_nonneg _)).trans ?_
      exact level_amortized run' id (D.k x) (D.cap x) (outerG D x run') (outerG_nonneg D x run')
    have hacc : Ex (fun r => (if (run' r).out = .acc then (1 : ℝ) else 0) * outerG D x run' r) ≤
        D.k x + D.k x * (nPU D x * (D.innerWaits x * D.innerK x)) := by
      rw [Ex_acc_forkOn (D.stage2 x) (fun r => (run' r).out = .acc) (outerG D x run') fun r d => by
        unfold outerG
        rw [forkOn_forkOn']]
      refine (Ex_mono fun r => outerG_amortized hdraw x run' r).trans_eq ?_
      exact Ex_const _
    calc Ex (fun s' : (Fin (D.V.k x) → Coin) × (E D decode).C x => (((E D decode).eval H P x ω s'.1 s'.2).2.2 : ℝ))
        ≤ 1 + Pr (fun r => (run' r).out = .acc) * Ex Rout := (Ex_mono hpt).trans_eq hsplit
      _ ≤ 1 + (D.k x + D.k x * (D.k x + D.k x * (nPU D x * (D.innerWaits x * D.innerK x)))) := by
          exact add_le_add le_rfl (hout.trans (add_le_add le_rfl (mul_le_mul_of_nonneg_left hacc hk0)))
      _ = 1 + D.k x + (D.k x : ℝ) ^ 2 * (1 + nPU D x * D.innerWaits x * D.innerK x) := by ring

end RunsExp

end Claims.Sampling
