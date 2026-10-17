import Proofs.Flock.Claims.Recursion.Cost
import Proofs.Flock.Claims.Recursion.Hybrid

/-!
# The recursion's expected cost: Wald's argument over forks

`E_O`'s waits at V* session `s` rerun forks of the judged run at the session's first round `off_s`: the coins before it
the judged run's, the rest fresh. `off_s` is read off the judged run's first `k_I` messages (`D.vstar`), which read only
its first `k_I ≤ off_s` coins (`run_msgs_take`), so `off_s` is fixed by the coins before it (`PrefixDet`). Swapping two
coin draws' coins from that round on is then an involution (`swapAt_swapAt`), so a fork of a uniform draw is uniform
(`Ex_fork`), and an accepted judged run weighs a function of the prefix by the prefix's acceptance probability
(`Ex_acc_fork`). With one wait's `ε·N ≤ k` (`waitRuns_exp`) at each prefix, session `s`'s waits add at most `m′_s·k_s`
in expectation (`forkWaits_exp`), once an accepted run shows session `s` accepting (`SessionAccepts`, a fact of the
compiled verifier's reader `D.session`). So `E_O` reruns the prover at most `1 + Σ_s m′_s·k_s` times in expectation
(`EO_runs_exp`), and with `EO_runs_le`, `Efficient` holds (`efficient`).
-/

namespace Claims.Recursion

open Claims

/-! ## A run's first messages read only its first coins -/

theorem stepWith_prefix {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (x : Msg) (ω : Ω) (vn : List Msg → OC Out) :
    ∀ (N : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t : Trace),
      t.msgs <+: (stepWith H P x ω vn N hist as t).msgs
  | 0, _, _, _ => List.prefix_refl _
  | N + 1, hist, as, t => by
    simp only [stepWith]
    split
    · refine List.IsPrefix.trans ?_ (stepWith_prefix H P x ω vn N _ _ _)
      exact List.prefix_append _ _
    · exact List.prefix_append _ _
    · exact List.prefix_append _ _

theorem stepWith_take {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (x : Msg) (ω : Ω) (vn vn' : List Msg → OC Out) :
    ∀ (N n : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t : Trace) (L : ℕ), L = t.msgs.length + n →
      (∀ bs : List Msg, bs.length ≤ as.length + n → vn bs = vn' bs) →
      (stepWith H P x ω vn N hist as t).msgs.take L = (stepWith H P x ω vn' N hist as t).msgs.take L
  | 0, _, _, _, _, _, _, _ => rfl
  | N + 1, 0, hist, as, t, L, hL, _ => by
    rw [hL, Nat.add_zero, ← List.prefix_iff_eq_take.1 (stepWith_prefix H P x ω vn (N + 1) hist as t),
      ← List.prefix_iff_eq_take.1 (stepWith_prefix H P x ω vn' (N + 1) hist as t)]
  | N + 1, n + 1, hist, as, t, L, hL, h => by
    have hv : vn (as ++ [((P.next x hist ω).eval H).1]) = vn' (as ++ [((P.next x hist ω).eval H).1]) :=
      h _ (by simp)
    simp only [stepWith, hv]
    split
    · exact stepWith_take H P x ω vn vn' N n _ _ _ L (by simp only [List.length_append, List.length_singleton]; omega)
        (fun bs hb => h bs (by simp only [List.length_append, List.length_singleton] at hb; omega))
    · rfl
    · rfl

/-- **A run's first `n` messages read only its first `n` coins.** -/
theorem run_msgs_take {Ω Coin : Type} [Inhabited Coin] (H : Msg → Msg) (P : Prover Ω) (V : Verifier Coin) (x : Msg)
    (ω : Ω) (n : ℕ) {r r' : Fin (V.k x) → Coin} (h : ∀ i : Fin (V.k x), i.val < n → r i = r' i) :
    (run H P V x ω r).msgs.take n = (run H P V x ω r').msgs.take n := by
  have := stepWith_take H P x ω (fun as => V.next x as ((List.range as.length).map (coinSeq r)))
    (fun as => V.next x as ((List.range as.length).map (coinSeq r'))) (V.k x) n [] [] ⟨[], [], [], .rej⟩ n
    (by simp) (fun bs hb => by
      simp only [List.length_nil, Nat.zero_add] at hb
      congr 1
      refine List.map_congr_left fun i hi => ?_
      rw [List.mem_range] at hi
      unfold coinSeq
      split_ifs with hk
      · exact h ⟨i, hk⟩ (by simp only; omega)
      · rfl)
  exact this

/-! ## Forks at a round the coins before it fix -/

/-- **`off` is read off the coins before it**: two draws that agree before `off r` have one `off`. -/
def PrefixDet {Coin : Type} {k : ℕ} (off : (Fin k → Coin) → ℕ) : Prop :=
  ∀ r r' : Fin k → Coin, (∀ i : Fin k, i.val < off r → r i = r' i) → off r' = off r

theorem fork_agree {Coin : Type} {k : ℕ} (off : ℕ) (r d : Fin k → Coin) :
    ∀ i : Fin k, i.val < off → r i = fork off r d i := fun i hi => by simp [fork, hi]

/-- Two draws with their coins from the first's `off` on swapped. -/
def swapAt {Coin : Type} {k : ℕ} (off : (Fin k → Coin) → ℕ) (p : (Fin k → Coin) × (Fin k → Coin)) :
    (Fin k → Coin) × (Fin k → Coin) :=
  (fork (off p.1) p.1 p.2, fork (off p.1) p.2 p.1)

theorem swapAt_swapAt {Coin : Type} {k : ℕ} {off : (Fin k → Coin) → ℕ} (hoff : PrefixDet off)
    (p : (Fin k → Coin) × (Fin k → Coin)) : swapAt off (swapAt off p) = p := by
  obtain ⟨r, c⟩ := p
  have h : off (fork (off r) r c) = off r := hoff r _ (fork_agree (off r) r c)
  simp only [swapAt, h]
  refine Prod.ext (funext fun i => ?_) (funext fun i => ?_) <;> by_cases hi : i.val < off r <;> simp [fork, hi]

/-- **A fork of a uniform draw is uniform**, at a round its coins before it fix. -/
theorem Ex_fork {Coin : Type} [Fintype Coin] [Nonempty Coin] {k : ℕ} {off : (Fin k → Coin) → ℕ}
    (hoff : PrefixDet off) (F : (Fin k → Coin) → ℝ) :
    Ex (fun p : (Fin k → Coin) × (Fin k → Coin) => F (fork (off p.1) p.1 p.2)) = Ex F := by
  let e : ((Fin k → Coin) × (Fin k → Coin)) ≃ ((Fin k → Coin) × (Fin k → Coin)) :=
    ⟨swapAt off, swapAt off, swapAt_swapAt hoff, swapAt_swapAt hoff⟩
  have h : Ex (fun p : (Fin k → Coin) × (Fin k → Coin) => F (fork (off p.1) p.1 p.2)) =
      Ex (fun p : (Fin k → Coin) × (Fin k → Coin) => F p.1) := by
    unfold Ex
    congr 1
    exact Fintype.sum_equiv e _ _ fun p => rfl
  rw [h, Ex_prod]
  simp only [Ex_const]

/-- **An accepted draw weighs a function of its prefix by the prefix's acceptance probability**: for `g` read off the
coins before `off`, `𝔼[1[A] · g] = 𝔼_r[Pr_d[A(fork r d)] · g(r)]`. -/
theorem Ex_acc_fork {Coin : Type} [Fintype Coin] [Nonempty Coin] {k : ℕ} {off : (Fin k → Coin) → ℕ}
    (hoff : PrefixDet off) (A : (Fin k → Coin) → Prop) [DecidablePred A] (g : (Fin k → Coin) → ℝ)
    (hg : ∀ r r' : Fin k → Coin, (∀ i : Fin k, i.val < off r → r i = r' i) → g r' = g r) :
    Ex (fun r => (if A r then (1 : ℝ) else 0) * g r) =
      Ex (fun r => Pr (fun d => A (fork (off r) r d)) * g r) := by
  rw [← Ex_fork hoff (fun r => (if A r then (1 : ℝ) else 0) * g r), Ex_prod]
  congr 1
  funext r
  have hgr : ∀ d, g (fork (off r) r d) = g r := fun d => hg r _ (fork_agree (off r) r d)
  simp only [hgr]
  rw [Ex_mul_right, Ex_ind]

/-- **Waits over forks**: `m` waits, each over `K` forks of the judged draw at a round `off` its coins before it fix,
stopping at `kk` accepted runs, and started only when the judged draw accepted, add at most `m·kk` reruns in
expectation. At each prefix one wait makes `N` reruns with `ε·N ≤ kk` (`waitRuns_exp`), `ε` the prefix's acceptance
probability, which is the judged draw's weight there (`Ex_acc_fork`). -/
theorem forkWaits_exp {Coin : Type} [Fintype Coin] [Nonempty Coin] {k : ℕ} {off : (Fin k → Coin) → ℕ}
    (hoff : PrefixDet off) (run' : (Fin k → Coin) → Trace) (kk m K : ℕ) :
    Ex (fun s : (Fin k → Coin) × (Fin m → Fin K → Fin k → Coin) =>
      (if (run' s.1).out = .acc then (1 : ℝ) else 0) *
        ∑ i, (waitRuns (fun d => run' (fork (off s.1) s.1 d)) kk ((List.finRange K).map (s.2 i)) 0 : ℝ)) ≤
      m * kk := by
  rw [Ex_prod]
  have hin : ∀ r : Fin k → Coin, (Ex fun ρ : Fin m → Fin K → Fin k → Coin =>
      (if (run' r).out = .acc then (1 : ℝ) else 0) *
        ∑ i, (waitRuns (fun d => run' (fork (off r) r d)) kk ((List.finRange K).map (ρ i)) 0 : ℝ)) =
      (if (run' r).out = .acc then (1 : ℝ) else 0) * (m * waitExp (fun d => run' (fork (off r) r d)) kk K 0) := by
    intro r
    rw [Ex_mul_left, Ex_sum_pi (fun σ : Fin K → Fin k → Coin =>
      (waitRuns (fun d => run' (fork (off r) r d)) kk ((List.finRange K).map σ) 0 : ℝ)) m]
    rfl
  simp only [hin]
  rw [Ex_acc_fork hoff (fun r => (run' r).out = .acc) _ ?_]
  · calc _ ≤ Ex (fun _ : Fin k → Coin => (m : ℝ) * kk) := Ex_mono fun r => ?_
      _ = m * kk := Ex_const _
    have h := waitRuns_exp (fun d => run' (fork (off r) r d)) kk K 0 (Nat.zero_le _)
    simp only [Nat.cast_zero, sub_zero] at h
    rw [mul_left_comm]
    exact mul_le_mul_of_nonneg_left h (Nat.cast_nonneg _)
  · intro r r' h
    have ho := hoff r r' h
    have hf : (fun d => run' (fork (off r') r' d)) = fun d => run' (fork (off r) r d) := by
      funext d
      congr 1
      funext i
      simp only [fork, ho]
      split_ifs with hi
      · exact (h i hi).symm
      · rfl
    simp only [hf]

/-! ## `E_O`'s expected reruns -/

/-- **An accepted run shows each V* session accepting** (a fact of the compiled verifier's reader `D.session`): the
compiled verifier accepts a run only if each V* session's verifier does, and `D.session x s` reads off the run's view
session `s`'s view with that verifier's decision. -/
def SessionAccepts {Coin : Type} (D : Deploy Coin) : Prop :=
  ∀ (x : Msg) (s : ℕ) (v : Seen), s < D.nS x → v.out = .acc → (D.session x s v).out = .acc

/-- Session `s`'s view of the prover's run at coins `r`, as `E_O`'s waits see a rerun. -/
def sessRun {Coin : Type} [Inhabited Coin] (H : Msg → Msg) (D : Deploy Coin) {Ω : Type} (P : Prover Ω) (x : Msg)
    (ω : Ω) (s : ℕ) (r : Fin (D.V.k x) → Coin) : Trace :=
  servedRun H (viewing (D.session x s)) (run H P D.V x ω) r

/-- Session `s`'s first round in the run at coins `r`: the inner rounds, then the earlier sessions' rounds at the V*
statements read off the run's inner messages. -/
def offO {Coin W Col T : Type} [Inhabited Coin] (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W)
    (DO : Flock.Deploy Coin Col T) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) (s : ℕ) (r : Fin (D.V.k x) → Coin) :
    ℕ :=
  I.V.k (xI D x) + D.kS x +
    (((D.vstar x ((run H P D.V x ω r).msgs.take (I.V.k (xI D x) + D.kS x))).take s).map DO.V.k).sum

/-- Session `s`'s first round is read off the coins before it: the V* statements read only the inner and staging
rounds' messages, which read only those rounds' coins. -/
theorem offO_prefixDet {Coin W Col T : Type} [Inhabited Coin] (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W)
    (DO : Flock.Deploy Coin Col T) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) (s : ℕ) :
    PrefixDet (offO H D I DO P x ω s) := by
  intro r r' h
  have hm := run_msgs_take H P D.V x ω (I.V.k (xI D x) + D.kS x) (r := r) (r' := r') fun i hi =>
    h i (by unfold offO; omega)
  simp only [offO, hm]

theorem serve_viewing_runs {R α : Type} (H : Msg → Msg) (run : R → Trace) (f : Seen → Seen) (p : RP R α) :
    ((serve (viewing f) p).eval H run).2.2 = (p.eval H (servedRun H (viewing f) run)).2.2 := by
  rw [serve_eq, (RP.serve_eval H _ run p).2.2, RP.eval_runs_eq]
  simp only [viewing_runs, List.map_const', List.sum_replicate, smul_eq_mul, mul_one]

theorem waitRuns_map {R R' : Type} (run : R' → Trace) (g : R → R') (k : ℕ) :
    ∀ (rs : List R) (a : ℕ), waitRuns run k (rs.map g) a = waitRuns (fun r => run (g r)) k rs a
  | [], _ => rfl
  | r :: rs, a => by simp only [List.map_cons, waitRuns, waitRuns_map run g k rs]

/-- **Served waits rerun the prover once per wait rerun**: each wait at draws `ρ i`, rerunning at `g (ρ i j)` seen
through `f`, reruns as one wait over the served runs. -/
theorem servedWaits_runs_le {R W' Op Dr : Type} (H : Msg → Msg) (run : R → Trace) (f : Seen → Seen) (k : ℕ)
    (opens : Seen → List Op) (assemble : List Op → Option W') (check : W' → Bool) {m K : ℕ} (g : Dr → R)
    (ρ : Fin m → Fin K → Dr) :
    ((serve (viewing f) (firstWait k opens assemble check
        ((List.finRange m).map fun i => (List.finRange K).map fun j => g (ρ i j)))).eval H run).2.2 ≤
      ∑ i, waitRuns (fun d => servedRun H (viewing f) run (g d)) k ((List.finRange K).map (ρ i)) 0 := by
  rw [serve_viewing_runs]
  refine (firstWait_runs_le_sum H _ k opens assemble check _).trans (le_of_eq ?_)
  rw [List.map_map, Fin.sum_univ_def]
  congr 1
  refine List.map_congr_left fun i _ => ?_
  simp only [Function.comp]
  rw [show ((List.finRange K).map fun j => g (ρ i j)) = ((List.finRange K).map (ρ i)).map g by
    rw [List.map_map]; rfl, waitRuns_map]

theorem EO_runs_pt {Coin W Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
    (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (decO : List (ℕ × Col) → List T) (hS : SessionAccepts D) (x : Msg) {Ω : Type} (P : Prover Ω) (ω : Ω)
    (r₀ : Fin (D.V.k x) → Coin) (c : (EO D I DO decO).C x) :
    (((EO D I DO decO).eval H P x ω r₀ c).2.2 : ℝ) ≤ 1 + ∑ s : Fin (D.nS x),
      (if (sessRun H D P x ω s r₀).out = .acc then (1 : ℝ) else 0) *
        ∑ i, (waitRuns (fun d => sessRun H D P x ω s (fork (offO H D I DO P x ω s r₀) r₀ d)) (D.nums x s).k
          ((List.finRange (D.nums x s).cap).map (c s i)) 0 : ℝ) := by
  simp only [Extractor.eval, EO, RP.eval]
  split_ifs with hacc
  · have hs : ∀ s : Fin (D.nS x), (sessRun H D P x ω s r₀).out = .acc := fun s => hS x s _ s.isLt hacc
    simp only [hs, ↓reduceIte, one_mul]
    rw [Nat.cast_add, Nat.cast_one, add_comm (1 : ℝ)]
    refine add_le_add_left ((Nat.cast_le.2 (collect_runs_le H _ _)).trans ?_) 1
    rw [List.map_map, Nat.cast_list_sum, List.map_map, ← Fin.sum_univ_def]
    refine Finset.sum_le_sum fun s _ => ?_
    simp only [Function.comp]
    exact_mod_cast servedWaits_runs_le H (run H P D.V x ω) (D.session x s) _ _ _ _
      (fork (offO H D I DO P x ω s r₀) r₀) (c s)
  · simp only [RP.eval, Nat.zero_add, Nat.cast_one]
    refine le_add_of_nonneg_right (Finset.sum_nonneg fun s _ => mul_nonneg (by split_ifs <;> norm_num)
      (Finset.sum_nonneg fun i _ => Nat.cast_nonneg _))

theorem Ex_pair_coord {α ι : Type} [Fintype α] [Fintype ι] [DecidableEq ι] {C : ι → Type} [∀ i, Fintype (C i)]
    [∀ i, Nonempty (C i)] (i : ι) (f : α → C i → ℝ) :
    Ex (fun p : α × ((j : ι) → C j) => f p.1 (p.2 i)) = Ex (fun p : α × C i => f p.1 p.2) := by
  rw [Ex_prod, Ex_prod]
  congr 1
  funext a
  exact Ex_pi_coord i (f a)

/-- **`E_O` reruns the prover at most `1 + Σ_s m′_s·k_s` times in expectation**, once an accepted run shows each V*
session accepting: the judged run, then session `s`'s waits over forks at its first round (`forkWaits_exp`). -/
theorem EO_runs_exp {Coin W Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
    (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (decO : List (ℕ × Col) → List T) (hS : SessionAccepts D) :
    ExtractRunsExpLe H (EO D I DO decO) fun x =>
      1 + ∑ s : Fin (D.nS x), (((D.nums x s).waits * (D.nums x s).k : ℕ) : ℝ) := by
  intro x Ω P ω
  have hpt := fun p : (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x => EO_runs_pt H D I DO decO hS x P ω p.1 p.2
  refine (Ex_mono hpt).trans ?_
  rw [Ex_add, Ex_const, Ex_sum]
  refine add_le_add_right (Finset.sum_le_sum fun s _ => ?_) 1
  calc _ = Ex (fun p : (Fin (D.V.k x) → Coin) × (Fin (D.nums x s).waits → Fin (D.nums x s).cap → Fin (D.V.k x) → Coin) =>
          (if (sessRun H D P x ω s p.1).out = .acc then (1 : ℝ) else 0) *
            ∑ i, (waitRuns (fun d => sessRun H D P x ω s (fork (offO H D I DO P x ω s p.1) p.1 d)) (D.nums x s).k
              ((List.finRange (D.nums x s).cap).map (p.2 i)) 0 : ℝ)) :=
        Ex_pair_coord (C := fun s : Fin (D.nS x) =>
          Fin (D.nums x s).waits → Fin (D.nums x s).cap → Fin (D.V.k x) → Coin) s
          (fun r cs => (if (sessRun H D P x ω s r).out = .acc then (1 : ℝ) else 0) *
            ∑ i, (waitRuns (fun d => sessRun H D P x ω s (fork (offO H D I DO P x ω s r) r d)) (D.nums x s).k
              ((List.finRange (D.nums x s).cap).map (cs i)) 0 : ℝ))
    _ ≤ (D.nums x s).waits * (D.nums x s).k :=
        forkWaits_exp (offO_prefixDet H D I DO P x ω s) (sessRun H D P x ω s) _ _ _
    _ = (((D.nums x s).waits * (D.nums x s).k : ℕ) : ℝ) := by push_cast; rfl

/-- **The recursion's cost, proved**: `E*` reruns the prover at most `(1 + m′_I·K_I)·(1 + Σ_s m′_s·K_s)` times, and
`N_I·(1 + Σ_s m′_s·k_s)` in expectation, once an accepted run shows each V* session accepting (`SessionAccepts`). -/
theorem efficient {Coin W Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
    (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (decO : List (ℕ × Col) → List T) (hS : SessionAccepts D) : Efficient H D I DO decO :=
  efficient_of_EO H D I DO decO (EO_runs_le H D I DO decO) (EO_runs_exp H D I DO decO hS)

end Claims.Recursion
