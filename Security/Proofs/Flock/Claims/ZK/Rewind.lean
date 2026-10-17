import Proofs.Flock.Claims.ZK.GK
import Proofs.Flock.Claims.Prob

/-!
# The rewinding argument, over abstract runs

* Uniform averages: a bijection keeps `Pr` (`Pr_equiv`), a product's `Pr` is the average of the sections in either order
  (`Pr_prod_left`, `Pr_prod_right`), independent coordinates multiply (`Pr_and_prod`), and averages of pointwise close
  functions are close (`Ex_abs_sub_le`).
* `Pr_hybrid`: `N` independent samples of `A` and of `B`, each within `δ` on every event, are within `N·δ` on every
  event.
* `rewind_close`: the first run's transcript is within `1/(N+1) + Pr[binding failure]` of the output of `N` rewinds that
  rerun the same prover on fresh tapes until one opens the first run's coins again (`gkOut`). The proof is the
  involution swapping the first run with the first successful rewind; what is left is the first run opening some coins
  and no rewind opening any, `Q·(1−Q)^N ≤ 1/(N+1)` (`mul_one_sub_pow_le`).
-/

namespace Claims.ZKMal

open Claims

/-! ## Uniform averages -/

theorem Pr_equiv {α β : Type} [Fintype α] [Fintype β] (e : α ≃ β) (p : β → Prop) :
    Pr (fun a => p (e a)) = Pr p := by
  classical
  rw [Pr_eq_sum (fun a => p (e a)), Pr_eq_sum p, Fintype.card_congr e]
  congr 1
  exact Fintype.sum_equiv e _ _ fun _ => rfl

theorem Pr_prod_left {α β : Type} [Fintype α] [Fintype β] (p : α × β → Prop) :
    Pr p = Ex fun a => Pr fun b => p (a, b) := by
  classical
  unfold Ex
  rw [Pr_eq_sum p, Fintype.sum_prod_type, Fintype.card_prod]
  simp only [Pr_eq_sum (fun b => p (_, b))]
  rw [← Finset.sum_div]
  push_cast
  rw [div_div, mul_comm (Fintype.card α : ℝ)]

theorem Pr_prod_right {α β : Type} [Fintype α] [Fintype β] (p : α × β → Prop) :
    Pr p = Ex fun b => Pr fun a => p (a, b) := by
  rw [← Pr_equiv (Equiv.prodComm β α) p, Pr_prod_left]
  rfl

theorem Pr_const {α : Type} [Fintype α] [Nonempty α] (P : Prop) [Decidable P] :
    Pr (fun _ : α => P) = if P then 1 else 0 := by
  classical
  have : (0 : ℝ) < Fintype.card α := by exact_mod_cast Fintype.card_pos
  rw [Pr_eq_sum]
  split_ifs <;> simp [this.ne']

/-- An event splits on another. -/
theorem Pr_split {α : Type} [Fintype α] (p q : α → Prop) :
    Pr p = Pr (fun a => p a ∧ q a) + Pr (fun a => p a ∧ ¬ q a) := by
  classical
  rw [Pr_eq_sum p, Pr_eq_sum (fun a => p a ∧ q a), Pr_eq_sum (fun a => p a ∧ ¬ q a), ← add_div,
    ← Finset.sum_add_distrib]
  congr 1
  refine Finset.sum_congr rfl fun a _ => ?_
  by_cases hp : p a <;> by_cases hq : q a <;> simp [hp, hq]

theorem Pr_or_le {α : Type} [Fintype α] (p q : α → Prop) : Pr (fun a => p a ∨ q a) ≤ Pr p + Pr q := by
  rw [Pr_split (fun a => p a ∨ q a) p]
  have h1 := Pr_mono (p := fun a => (p a ∨ q a) ∧ p a) (q := p) fun _ h => h.2
  have h2 := Pr_mono (p := fun a => (p a ∨ q a) ∧ ¬ p a) (q := q) fun _ h => h.1.resolve_left h.2
  linarith

theorem Pr_compl {α : Type} [Fintype α] [Nonempty α] (p : α → Prop) : Pr (fun a => ¬ p a) = 1 - Pr p := by
  have h := Pr_split (fun _ : α => True) p
  classical
  rw [Pr_const True] at h
  simp only [true_and, ite_true] at h
  linarith

theorem Ex_mul_const {α : Type} [Fintype α] (f : α → ℝ) (c : ℝ) : Ex (fun a => f a * c) = Ex f * c := by
  unfold Ex; rw [← Finset.sum_mul]; ring

theorem Ex_const_add {α : Type} [Fintype α] [Nonempty α] (c : ℝ) (f : α → ℝ) : Ex (fun a => c + f a) = c + Ex f := by
  rw [Ex_add, Ex_const]

/-- **Independent coordinates multiply.** -/
theorem Pr_and_prod {α β : Type} [Fintype α] [Fintype β] (p : α → Prop) (q : β → Prop) :
    Pr (fun s : α × β => p s.1 ∧ q s.2) = Pr p * Pr q := by
  classical
  rw [Pr_prod_left]
  have e : ∀ a, Pr (fun b => p a ∧ q b) = (if p a then 1 else 0) * Pr q := fun a => by
    by_cases h : p a
    · simp only [h, true_and, ite_true, one_mul]
    · simp only [h, false_and, ite_false, zero_mul]
      exact Pr_false
  simp only [e]
  rw [Ex_mul_const, Pr_eq_sum p]
  rfl

/-- **Averages of pointwise close functions are close.** -/
theorem Ex_abs_sub_le {α : Type} [Fintype α] {f g b : α → ℝ} (h : ∀ a, |f a - g a| ≤ b a) :
    |Ex f - Ex g| ≤ Ex b := by
  have e : Ex f - Ex g = Ex fun a => f a - g a := by
    unfold Ex; rw [← sub_div, ← Finset.sum_sub_distrib]
  have e2 : Ex (fun a => -b a) = -Ex b := by
    unfold Ex; rw [Finset.sum_neg_distrib, neg_div]
  rw [e, abs_le]
  refine ⟨?_, Ex_mono fun a => (abs_le.mp (h a)).2⟩
  have := Ex_mono (f := fun a => -b a) (g := fun a => f a - g a) fun a => (abs_le.mp (h a)).1
  linarith

/-- `Q·(1−Q)^N ≤ 1/(N+1)` on `[0, 1]`. -/
theorem mul_one_sub_pow_le (Q : ℝ) (h0 : 0 ≤ Q) (h1 : Q ≤ 1) (N : ℕ) : Q * (1 - Q) ^ N ≤ 1 / (N + 1) := by
  have hb : 1 + N * Q ≤ (1 + Q) ^ N := one_add_mul_le_pow (by linarith) N
  have hy0 : 0 ≤ (1 - Q) ^ N := pow_nonneg (by linarith) N
  have hp : (1 - Q) ^ N * (1 + Q) ^ N ≤ 1 := by
    rw [← mul_pow]
    exact pow_le_one₀ (by nlinarith) (by nlinarith)
  have hy : (1 - Q) ^ N * (1 + N * Q) ≤ 1 := le_trans (mul_le_mul_of_nonneg_left hb hy0) hp
  have hq : 0 ≤ (1 - Q) ^ N * (1 - Q) := mul_nonneg hy0 (by linarith)
  rw [le_div_iff₀ (by positivity)]
  nlinarith

/-! ## Hybrids over independent samples -/

/-- **`N` independent samples**: if `A` and `B` are within `δ` on every event, `N` independent copies of each are within
`N·δ` on every event. -/
theorem Pr_hybrid {Ω₁ Ω₂ M : Type} [Fintype Ω₁] [Nonempty Ω₁] [Fintype Ω₂] [Nonempty Ω₂] (A : Ω₁ → M) (B : Ω₂ → M)
    (δ : ℝ) (h : ∀ E : M → Prop, |Pr (fun a => E (A a)) - Pr (fun b => E (B b))| ≤ δ) :
    ∀ (N : ℕ) (f : (Fin N → M) → Prop),
      |Pr (fun α : Fin N → Ω₁ => f fun i => A (α i)) - Pr (fun β : Fin N → Ω₂ => f fun i => B (β i))| ≤ N * δ
  | 0, f => by
      classical
      have e1 : ∀ α : Fin 0 → Ω₁, (fun i => A (α i)) = Fin.elim0 := fun _ => funext fun i => i.elim0
      have e2 : ∀ β : Fin 0 → Ω₂, (fun i => B (β i)) = Fin.elim0 := fun _ => funext fun i => i.elim0
      simp only [e1, e2, Pr_const]
      simp
  | N + 1, f => by
      have cons_comp : ∀ {Ω : Type} (F : Ω → M) (a : Ω) (α : Fin N → Ω),
          (fun i => F ((Fin.cons a α : Fin (N + 1) → Ω) i)) = Fin.cons (F a) fun i => F (α i) := by
        intro Ω F a α
        funext i
        cases i using Fin.cases <;> simp
      have split : ∀ {Ω : Type} [Fintype Ω] (F : Ω → M),
          Pr (fun α : Fin (N + 1) → Ω => f fun i => F (α i)) =
            Pr (fun p : Ω × (Fin N → Ω) => f (Fin.cons (F p.1) fun i => F (p.2 i))) := by
        intro Ω _ F
        rw [← Pr_equiv (Fin.consEquiv fun _ : Fin (N + 1) => Ω)]
        refine Pr_congr fun p => ?_
        rw [← cons_comp]
        rfl
      rw [split A, split B]
      have hA : |Pr (fun p : Ω₁ × (Fin N → Ω₁) => f (Fin.cons (A p.1) fun i => A (p.2 i))) -
          Pr (fun p : Ω₁ × (Fin N → Ω₂) => f (Fin.cons (A p.1) fun i => B (p.2 i)))| ≤ N * δ := by
        rw [Pr_prod_left, Pr_prod_left]
        refine le_trans (Ex_abs_sub_le fun a => Pr_hybrid A B δ h N fun g => f (Fin.cons (A a) g)) ?_
        rw [Ex_const]
      have hB : |Pr (fun p : Ω₁ × (Fin N → Ω₂) => f (Fin.cons (A p.1) fun i => B (p.2 i))) -
          Pr (fun p : Ω₂ × (Fin N → Ω₂) => f (Fin.cons (B p.1) fun i => B (p.2 i)))| ≤ δ := by
        rw [Pr_prod_right, Pr_prod_right]
        refine le_trans (Ex_abs_sub_le fun β => h fun m => f (Fin.cons m fun i => B (β i))) ?_
        rw [Ex_const]
      refine le_trans (abs_sub_le _
        (Pr fun p : Ω₁ × (Fin N → Ω₂) => f (Fin.cons (A p.1) fun i => B (p.2 i))) _) ?_
      push_cast
      linarith

/-! ## The rewinds -/

/-- **The first rewind that opened every coin**, its transcript; `d` if none did. -/
def firstSucc {X : Type} (d : X) : (N : ℕ) → (Fin N → St × X) → X
  | 0, _ => d
  | N + 1, ys => if (ys 0).1 = .succ then (ys 0).2 else firstSucc d N fun i => ys i.succ

theorem firstSucc_eq {X : Type} (d : X) : ∀ (N : ℕ) (ys : Fin N → St × X) (i : Fin N), (ys i).1 = .succ →
    (∀ j < i, (ys j).1 ≠ .succ) → firstSucc d N ys = (ys i).2
  | 0, _, i, _, _ => i.elim0
  | N + 1, ys, i, hi, hlt => by
      simp only [firstSucc]
      cases i using Fin.cases with
      | zero => simp [hi]
      | succ i =>
          simp only [hlt 0 (Fin.succ_pos i), ite_false]
          exact firstSucc_eq d N _ i hi fun j hj => hlt j.succ (Fin.succ_lt_succ_iff.mpr hj)

/-- What the first run passes on: its transcript if it opened no coin vector, else the coins. -/
def viewOf {Ω C X : Type} (Rc : Ω → Option C) (τ : Ω → X) (r : Ω) : X ⊕ C :=
  match Rc r with
  | none => .inl (τ r)
  | some c => .inr c

/-- **The Goldreich–Kahan output**: the first run's transcript if it opened no coin vector, else the first of the
rewinds at its coins that opened them all (`d` if none did). -/
def gkOut {C X : Type} {N : ℕ} (d : X) (v : X ⊕ C) (Y : C → Fin N → St × X) : X :=
  match v with
  | .inl t => t
  | .inr c => firstSucc d N (Y c)

/-- **A binding failure** among the rewinds before the first that opened every coin. -/
def gkMM {C X : Type} {N : ℕ} (v : X ⊕ C) (Y : C → Fin N → St × X) : Prop :=
  match v with
  | .inl _ => False
  | .inr c => ∃ i, (Y c i).1 = .mism ∧ ∀ j < i, (Y c j).1 ≠ .succ

section Rewind

variable {Ω C X : Type} [Fintype Ω] [Nonempty Ω]

/-- The swap of the first run with rewind `i`. -/
def swapAt {N : ℕ} (i : Fin N) (p : Ω × (Fin N → Ω)) : Ω × (Fin N → Ω) := (p.2 i, Function.update p.2 i p.1)

omit [Fintype Ω] [Nonempty Ω] in
theorem swapAt_swapAt {N : ℕ} (i : Fin N) (p : Ω × (Fin N → Ω)) : swapAt i (swapAt i p) = p := by
  obtain ⟨a, α⟩ := p
  simp [swapAt]

/-- **The Goldreich–Kahan rewinds are close to the first run**: against rewinds of the same prover on fresh tapes, a run
opens `c` exactly when a rewind at `c` succeeds (`h1`), a successful rewind is the run (`h2`), and a rewind that stops
before a binding failure is a run that opens nothing (`h3`). Then the output is within `1/(N+1)` plus the probability of
a binding failure of the first run's transcript. -/
theorem rewind_close (N : ℕ) (d : X) (Rc : Ω → Option C) (τ : Ω → X) (st : C → Ω → St) (T : C → Ω → X)
    (h1 : ∀ c r, st c r = .succ ↔ Rc r = some c) (h2 : ∀ c r, st c r = .succ → T c r = τ r)
    (h3 : ∀ c r, st c r = .abort → Rc r = none) (Ev : X → Prop) :
    |Pr (fun r => Ev (τ r)) -
        Pr (fun p : Ω × (Fin N → Ω) => Ev (gkOut d (viewOf Rc τ p.1) fun c i => (st c (p.2 i), T c (p.2 i))))| ≤
      1 / (N + 1) + Pr (fun p : Ω × (Fin N → Ω) => gkMM (viewOf Rc τ p.1) fun c i => (st c (p.2 i), T c (p.2 i))) := by
  classical
  set Y : Ω × (Fin N → Ω) → C → Fin N → St × X := fun p c i => (st c (p.2 i), T c (p.2 i)) with hY
  let Bi : Fin N → Ω × (Fin N → Ω) → Prop := fun i p =>
    ∃ c, Rc p.1 = some c ∧ st c (p.2 i) = .succ ∧ ∀ j < i, st c (p.2 j) ≠ .succ
  have huniq : ∀ i i' p, Bi i p → Bi i' p → i = i' := by
    rintro i i' p ⟨c, hc, hs, hlt⟩ ⟨c', hc', hs', hlt'⟩
    rw [hc] at hc'
    cases hc'
    rcases lt_trichotomy i i' with h | h | h
    · exact absurd hs (hlt' i h)
    · exact h
    · exact absurd hs' (hlt i' h)
  have hswap : ∀ i p, Bi i p → Bi i (swapAt i p) := by
    rintro i ⟨a, α⟩ ⟨c, hc, hs, hlt⟩
    refine ⟨c, (h1 c _).mp hs, ?_, fun j hj => ?_⟩
    · simp only [swapAt, Function.update_self]
      exact (h1 c a).mpr hc
    · simp only [swapAt]
      rw [Function.update_of_ne (ne_of_lt hj)]
      exact hlt j hj
  let Φ : Ω × (Fin N → Ω) → Ω × (Fin N → Ω) := fun p =>
    if h : ∃ i, Bi i p then swapAt (Classical.choose h) p else p
  have hΦB : ∀ i p, Bi i p → Φ p = swapAt i p := by
    intro i p hB
    simp only [Φ]
    split
    · next h => rw [huniq _ _ p (Classical.choose_spec h) hB]
    · next h => exact absurd ⟨i, hB⟩ h
  have hΦn : ∀ p, (¬ ∃ i, Bi i p) → Φ p = p := fun p h => by
    simp only [Φ]
    split
    · next h' => exact absurd h' h
    · rfl
  have hinv : Function.Involutive Φ := by
    intro p
    by_cases h : ∃ i, Bi i p
    · obtain ⟨i, hB⟩ := h
      rw [hΦB i p hB, hΦB i _ (hswap i p hB), swapAt_swapAt]
    · rw [hΦn p h, hΦn p h]
  let Fail : Ω × (Fin N → Ω) → Prop := fun p => ∃ c, Rc p.1 = some c ∧ ∀ i, st c (p.2 i) ≠ .succ
  let out : Ω × (Fin N → Ω) → X := fun p => gkOut d (viewOf Rc τ p.1) (Y p)
  have hFail : ∀ p, Fail (Φ p) ↔ Fail p := by
    intro p
    by_cases h : ∃ i, Bi i p
    · obtain ⟨i, hB⟩ := h
      rw [hΦB i p hB]
      have nf : ∀ q, Bi i q → ¬ Fail q := by
        rintro q ⟨c, hc, hs, -⟩ ⟨c', hc', hall⟩
        rw [hc] at hc'
        cases hc'
        exact hall i hs
      exact iff_of_false (nf _ (hswap i p hB)) (nf p hB)
    · rw [hΦn p h]
  have hout : ∀ p, ¬ Fail p → (Ev (out p) ↔ Ev (τ (Φ p).1)) := by
    rintro ⟨a, α⟩ hF
    cases hc : Rc a with
    | none =>
        have : ¬ ∃ i, Bi i (a, α) := by
          rintro ⟨i, c, hc', -⟩
          rw [hc] at hc'
          cases hc'
        rw [hΦn _ this]
        simp [out, viewOf, gkOut, hc]
    | some c =>
        have hex : ∃ i, st c (α i) = .succ := by
          by_contra hn
          exact hF ⟨c, hc, fun i hi => hn ⟨i, hi⟩⟩
        obtain ⟨i, hi, hlt⟩ : ∃ i, st c (α i) = .succ ∧ ∀ j < i, st c (α j) ≠ .succ := by
          have hne : (Finset.univ.filter fun i => st c (α i) = .succ).Nonempty := by
            obtain ⟨i, hi⟩ := hex
            exact ⟨i, Finset.mem_filter.mpr ⟨Finset.mem_univ _, hi⟩⟩
          refine ⟨_, (Finset.mem_filter.mp (Finset.min'_mem _ hne)).2, fun j hj hs => ?_⟩
          exact absurd hj (not_lt.mpr (Finset.min'_le _ j (Finset.mem_filter.mpr ⟨Finset.mem_univ _, hs⟩)))
        have hB : Bi i (a, α) := ⟨c, hc, hi, hlt⟩
        rw [hΦB i _ hB]
        have e : out (a, α) = τ (α i) := by
          simp only [out, viewOf, hc, gkOut]
          rw [firstSucc_eq d N (Y (a, α) c) i hi hlt]
          exact h2 c (α i) hi
        rw [e]
        rfl
  have hmain : |Pr (fun p : Ω × (Fin N → Ω) => Ev (τ p.1)) - Pr (fun p => Ev (out p))| ≤ Pr Fail := by
    have eA := Pr_split (fun p : Ω × (Fin N → Ω) => Ev (τ p.1)) Fail
    have eB := Pr_split (fun p : Ω × (Fin N → Ω) => Ev (out p)) Fail
    have eq : Pr (fun p => Ev (out p) ∧ ¬ Fail p) = Pr (fun p : Ω × (Fin N → Ω) => Ev (τ p.1) ∧ ¬ Fail p) := by
      rw [← Pr_equiv (Function.Involutive.toPerm Φ hinv) (fun p : Ω × (Fin N → Ω) => Ev (τ p.1) ∧ ¬ Fail p)]
      refine Pr_congr fun p => ?_
      simp only [Function.Involutive.coe_toPerm]
      rw [hFail p]
      exact ⟨fun ⟨h, hn⟩ => ⟨(hout p hn).mp h, hn⟩, fun ⟨h, hn⟩ => ⟨(hout p hn).mpr h, hn⟩⟩
    rw [eA, eB, eq]
    have a1 := Pr_mono (p := fun p : Ω × (Fin N → Ω) => Ev (τ p.1) ∧ Fail p) (q := Fail) fun _ h => h.2
    have a2 := Pr_mono (p := fun p : Ω × (Fin N → Ω) => Ev (out p) ∧ Fail p) (q := Fail) fun _ h => h.2
    have a3 := Pr_nonneg (fun p : Ω × (Fin N → Ω) => Ev (τ p.1) ∧ Fail p)
    have a4 := Pr_nonneg (fun p : Ω × (Fin N → Ω) => Ev (out p) ∧ Fail p)
    rw [abs_le]
    constructor <;> linarith
  let Ab : Ω × (Fin N → Ω) → Prop := fun p => (fun r => Rc r ≠ none) p.1 ∧ (fun α : Fin N → Ω => ∀ i, Rc (α i) = none) p.2
  have hFail_le : Pr Fail ≤ 1 / (N + 1) + Pr (fun p => gkMM (viewOf Rc τ p.1) (Y p)) := by
    have hsub : ∀ p, Fail p → Ab p ∨ gkMM (viewOf Rc τ p.1) (Y p) := by
      rintro ⟨a, α⟩ ⟨c, hc, hall⟩
      by_cases hm : ∃ i, st c (α i) = .mism
      · right
        obtain ⟨i, hi⟩ := hm
        simp only [viewOf, hc, gkMM, hY]
        exact ⟨i, hi, fun j _ => hall j⟩
      · left
        refine ⟨by simp [hc], fun i => h3 c (α i) ?_⟩
        have h₁ := hall i
        have h₂ : st c (α i) ≠ .mism := fun h => hm ⟨i, h⟩
        cases hs : st c (α i)
        · exact absurd hs h₁
        · exact absurd hs h₂
        · rfl
    have hAb : Pr Ab ≤ 1 / (N + 1) := by
      have e := Pr_and_prod (fun r => Rc r ≠ none) (fun α : Fin N → Ω => ∀ i, Rc (α i) = none)
      rw [Pr_pi N (fun r => Rc r = none)] at e
      have hc : Pr (fun r => Rc r = none) = 1 - Pr (fun r => Rc r ≠ none) := by
        rw [← Pr_compl]
        exact Pr_congr fun r => by simp
      rw [hc] at e
      have : Pr Ab = Pr (fun r => Rc r ≠ none) * (1 - Pr fun r => Rc r ≠ none) ^ N := e
      rw [this]
      exact mul_one_sub_pow_le _ (Pr_nonneg _) (Pr_le_one _) N
    calc Pr Fail ≤ Pr (fun p => Ab p ∨ gkMM (viewOf Rc τ p.1) (Y p)) := Pr_mono hsub
      _ ≤ Pr Ab + Pr (fun p => gkMM (viewOf Rc τ p.1) (Y p)) := Pr_or_le _ _
      _ ≤ _ := by linarith
  rw [← Pr_fst (β := Fin N → Ω) (fun r => Ev (τ r))]
  exact le_trans hmain hFail_le

end Rewind

end Claims.ZKMal
