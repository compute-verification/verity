module

public import Definitions.Claims.Sampling
public import Proofs.Flock.Claims.ERepCost

/-! Coins fresh on a set of rounds, and the sampled extractor's waits. -/

@[expose] public section

namespace Claims.Sampling

open Claims

theorem forkOn_forkOn {k : ℕ} {Coin : Type} (I : Finset (Fin k)) (r d d' : Fin k → Coin) :
    forkOn I (forkOn I r d) d' = forkOn I r d' :=
  funext fun t => by by_cases h : t ∈ I <;> simp [forkOn, h]

theorem forkOn_forkOn' {k : ℕ} {Coin : Type} (I : Finset (Fin k)) (r d : Fin k → Coin) :
    forkOn I (forkOn I r d) = forkOn I r :=
  funext fun d' => forkOn_forkOn I r d d'

def swapOn {k : ℕ} {Coin : Type} (I : Finset (Fin k)) :
    (Fin k → Coin) × (Fin k → Coin) ≃ (Fin k → Coin) × (Fin k → Coin) where
  toFun p := (forkOn I p.1 p.2, forkOn I p.2 p.1)
  invFun p := (forkOn I p.1 p.2, forkOn I p.2 p.1)
  left_inv p := Prod.ext (funext fun t => by by_cases h : t ∈ I <;> simp [forkOn, h])
    (funext fun t => by by_cases h : t ∈ I <;> simp [forkOn, h])
  right_inv p := Prod.ext (funext fun t => by by_cases h : t ∈ I <;> simp [forkOn, h])
    (funext fun t => by by_cases h : t ∈ I <;> simp [forkOn, h])

theorem Ex_forkOn {k : ℕ} {Coin : Type} [Fintype Coin] [Nonempty Coin] (I : Finset (Fin k))
    (F : (Fin k → Coin) → ℝ) :
    Ex (fun p : (Fin k → Coin) × (Fin k → Coin) => F (forkOn I p.1 p.2)) = Ex F := by
  have h1 : Ex (fun p : (Fin k → Coin) × (Fin k → Coin) => F (forkOn I p.1 p.2)) =
      Ex (fun p : (Fin k → Coin) × (Fin k → Coin) => F p.1) := by
    unfold Ex
    congr 1
    exact Fintype.sum_equiv (swapOn I) _ _ fun p => rfl
  rw [h1, Ex_prod]
  congr 1
  funext a
  exact Ex_const (α := Fin k → Coin) (F a)

theorem Ex_acc_forkOn {k : ℕ} {Coin : Type} [Fintype Coin] [Nonempty Coin] (I : Finset (Fin k))
    (acc : (Fin k → Coin) → Prop) [DecidablePred acc] (g : (Fin k → Coin) → ℝ) (hg : ∀ r d, g (forkOn I r d) = g r) :
    Ex (fun r => (if acc r then (1 : ℝ) else 0) * g r) =
      Ex (fun r => Pr (fun d => acc (forkOn I r d)) * g r) := by
  rw [← Ex_forkOn I (fun r => (if acc r then (1 : ℝ) else 0) * g r), Ex_prod]
  congr 1
  funext r
  simp only [hg]
  rw [Ex_mul_right, Ex_ind]

theorem Pr_acc_forkOn {k : ℕ} {Coin C : Type} [Fintype Coin] [Nonempty Coin] [Fintype C] (I : Finset (Fin k))
    (acc : (Fin k → Coin) → Prop) [DecidablePred acc] (Q : (Fin k → Coin) → C → Prop)
    (hQ : ∀ r d c, Q (forkOn I r d) c ↔ Q r c) :
    Pr (fun s : (Fin k → Coin) × C => acc s.1 ∧ Q s.1 s.2) =
      Ex (fun r => Pr (fun d => acc (forkOn I r d)) * Pr (fun c => Q r c)) := by
  classical
  have hs : Pr (fun s : (Fin k → Coin) × C => acc s.1 ∧ Q s.1 s.2) =
      Ex (fun r => (if acc r then (1 : ℝ) else 0) * Pr (fun c => Q r c)) := by
    rw [← Ex_ind, Ex_prod]
    congr 1
    funext r
    rw [← Ex_ind (fun c => Q r c), ← Ex_mul_left]
    congr 1
    funext c
    by_cases h1 : acc r <;> by_cases h2 : Q r c <;> simp [h1, h2]
  rw [hs, Ex_acc_forkOn I acc _ fun r d => Pr_congr fun c => hQ r d c]

theorem Ex_finsum {α ι : Type} [Fintype α] (s : Finset ι) (f : ι → α → ℝ) :
    Ex (fun a => ∑ i ∈ s, f i a) = ∑ i ∈ s, Ex (f i) := by
  unfold Ex
  rw [Finset.sum_comm, Finset.sum_div]

theorem Ex_pi_eval {ι E : Type} [Fintype ι] [DecidableEq ι] [Fintype E] [Nonempty E] (i : ι) (f : E → ℝ) :
    Ex (fun c : ι → E => f (c i)) = Ex f := by
  have h : Ex (fun c : ι → E => f (c i)) = Ex (fun p : E × ({j // j ≠ i} → E) => f p.1) := by
    unfold Ex
    rw [Fintype.card_congr (Equiv.funSplitAt i E)]
    congr 1
    exact Fintype.sum_equiv (Equiv.funSplitAt i E) _ _ fun c => rfl
  rw [h, Ex_prod]
  congr 1
  funext a
  exact Ex_const (α := {j // j ≠ i} → E) (f a)

theorem sum_map_eq_count {ι : Type} [Fintype ι] [DecidableEq ι] (l : List ι) (f : ι → ℝ) :
    (l.map f).sum = ∑ i, (l.count i : ℝ) * f i := by
  rw [Finset.sum_list_map_count, ← Finset.sum_subset (Finset.subset_univ l.toFinset)]
  · simp only [nsmul_eq_mul]
  · intro i _ hi
    rw [List.count_eq_zero.2 (fun h => hi (List.mem_toFinset.2 h)), Nat.cast_zero, zero_mul]

def waitsProg {D W Op : Type} (m k K : ℕ) (opens : Seen → List Op) (assemble : List Op → Option W)
    (check : W → Bool) (ρ : Fin m → Fin K → D) : RP D (Option W) :=
  firstWait k opens assemble check ((List.finRange m).map fun a => (List.finRange K).map (ρ a))

theorem waits_amortized {D W Op : Type} [Fintype D] [Nonempty D] (H : Msg → Msg) (run' : D → Transcript) (m k K : ℕ)
    (opens : Seen → List Op) (assemble : List Op → Option W) (check : W → Bool) :
    Pr (fun d => (run' d).out = .acc) *
        Ex (fun ρ : Fin m → Fin K → D => (((waitsProg m k K opens assemble check ρ).eval H run').2.2 : ℝ)) ≤
      m * k := by
  have hle : ∀ ρ : Fin m → Fin K → D, (((waitsProg m k K opens assemble check ρ).eval H run').2.2 : ℝ) ≤
      ∑ a, (waitRuns run' k ((List.finRange K).map (ρ a)) 0 : ℝ) := by
    intro ρ
    have hw := firstWait_runs_le_sum H run' k opens assemble check
      ((List.finRange m).map fun a => (List.finRange K).map (ρ a))
    simp only [List.map_map, Function.comp_def] at hw
    rw [← Fin.sum_univ_def] at hw
    exact_mod_cast hw
  have hE : Ex (fun ρ : Fin m → Fin K → D => ∑ a, (waitRuns run' k ((List.finRange K).map (ρ a)) 0 : ℝ)) =
      m * waitExp run' k K 0 :=
    Ex_sum_pi (fun σ : Fin K → D => (waitRuns run' k ((List.finRange K).map σ) 0 : ℝ)) m
  have h0 := waitRuns_exp run' k K 0 (Nat.zero_le _)
  have hε := Pr_nonneg fun d => (run' d).out = .acc
  calc Pr (fun d => (run' d).out = .acc) *
        Ex (fun ρ : Fin m → Fin K → D => (((waitsProg m k K opens assemble check ρ).eval H run').2.2 : ℝ))
      ≤ Pr (fun d => (run' d).out = .acc) * (m * waitExp run' k K 0) :=
        mul_le_mul_of_nonneg_left (hE ▸ Ex_mono hle) hε
    _ = m * (Pr (fun d => (run' d).out = .acc) * waitExp run' k K 0) := by ring
    _ ≤ m * k := by
        simp only [Nat.cast_zero, sub_zero] at h0
        exact mul_le_mul_of_nonneg_left h0 (Nat.cast_nonneg _)

inductive NoHash {R α : Type} : RP R α → Prop
  | ret (a : α) : NoHash (.ret a)
  | rerun (r : R) (k : Seen → RP R α) : (∀ v, NoHash (k v)) → NoHash (.rerun r k)

theorem NoHash.bind {R α β : Type} {p : RP R α} {f : α → RP R β} (hp : NoHash p) (hf : ∀ a, NoHash (f a)) :
    NoHash (p.bind f) := by
  induction hp with
  | ret a => exact hf a
  | rerun r k _ ih => exact NoHash.rerun r _ fun v => ih v

theorem NoHash.waitAcc {R : Type} (k : ℕ) : ∀ (rs : List R) (acc : List Seen), NoHash (Claims.waitAcc k rs acc) := by
  intro rs
  induction rs with
  | nil => intro acc; exact NoHash.ret _
  | cons r rs ih =>
    intro acc
    simp only [Claims.waitAcc]
    split
    · exact NoHash.ret _
    · exact NoHash.rerun r _ fun v => ih _

theorem NoHash.firstWait {R W Op : Type} (k : ℕ) (opens : Seen → List Op) (assemble : List Op → Option W)
    (check : W → Bool) : ∀ rss : List (List R), NoHash (Claims.firstWait k opens assemble check rss) := by
  intro rss
  induction rss with
  | nil => exact NoHash.ret _
  | cons rs rss ih =>
    refine (NoHash.waitAcc k rs []).bind fun vs => ?_
    split
    · split
      · exact NoHash.ret _
      · exact ih
    · exact ih

theorem calls_le_of_noHash {R α : Type} (H : Msg → Msg) (run : R → Transcript) (t : ℕ)
    (ht : ∀ r, (run r).pLog.length + (run r).vLog.length ≤ t) {p : RP R α} (hp : NoHash p) :
    (p.eval H run).2.1 ≤ t * (p.eval H run).2.2 := by
  induction hp with
  | ret a => simp [RP.eval]
  | rerun r k _ ih =>
    simp only [RP.eval]
    have := ih (run r).seen
    have := ht r
    rw [Nat.mul_succ]
    omega

end Claims.Sampling
