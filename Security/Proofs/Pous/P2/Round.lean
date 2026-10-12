import Models.Pous.Guarantees.P2
import Proofs.Pous.P2.RingBasic
import Proofs.Pous.Thm1.Theorem1iiiTimed

/-!
# P2 in the ring model: the whole audit from its rounds

A sequential responder's state at round `t` is a function of the setup and the `t` indices before it, and round `t`'s
index is drawn after it. So if, after every history, the state is short and answers more than a `c` fraction of the
indices with probability at most `ε` over the setup (`RoundSecureOn`), then:

1. at a fixed setup, off the rounds whose state is short and answers more than a `c` fraction (`bad`), the audit passes
   with at least `m` short rounds with probability at most `c^m` (`pr_chain_le`, the chain rule over the rounds);
2. a round is bad with probability at most `ε` over the setup and the indices, so some round is with probability at
   most `k·ε` (`pr_held_le`), whatever the history: the union is over the `k` rounds, not over the histories.

With every state `S` bits (`m = k`) this is the sequential audit (`auditSecureOn_of_round`), and at P2's parameters
`(501,578/2^19)^105 ≤ 1%` and `105 · 2^-136 ≤ 2^-129` (`p2RingAuditSecure_of_round`).
-/

namespace PousP2Round

open Pous PousThm1 PousP2Ring
open scoped ENNReal

/-! ### A responder's rounds over any state -/

section Trace

variable {τ : Type} {B ℓ : ℕ}

/-- The blocks a sequential responder returns for the indices `is` from round `j` in state `σ`, each with the state
its round started from; `nxt j σ i` is round `j`'s block for index `i` and the state it leaves. -/
def trace (nxt : ℕ → τ → Fin B → Bits ℓ × τ) : ℕ → List (Fin B) → τ → List (Bits ℓ × τ)
  | _, [], _ => []
  | j, i :: is, σ => ((nxt j σ i).1, σ) :: trace nxt (j + 1) is (nxt j σ i).2

/-- The state after the indices `is`, from round `j` in state `σ`. -/
def after (nxt : ℕ → τ → Fin B → Bits ℓ × τ) : ℕ → List (Fin B) → τ → τ
  | _, [], σ => σ
  | j, i :: is, σ => after nxt (j + 1) is (nxt j σ i).2

theorem ofFn_cons {α : Type} {k : ℕ} (a : α) (I : Fin k → α) :
    List.ofFn (Fin.cons a I : Fin (k + 1) → α) = a :: List.ofFn I := by
  simp [List.ofFn_succ]

theorem ofFn_cons_map {α β : Type} {k : ℕ} (g : α → β) (a : α) (I : Fin k → α) :
    List.ofFn (fun t => g ((Fin.cons a I : Fin (k + 1) → α) t)) = g a :: List.ofFn fun t => g (I t) := by
  simp [List.ofFn_succ]

theorem pr_false {α : Type} [Fintype α] : pr (fun _ : α => False) = 0 := pr_false'

/-- **The chain rule with short rounds.** At a fixed setup: if after every history `h` that `good` admits, a round
whose state is short answers a uniform index with probability at most `c`, then the audit passes, with at least `m`
of its rounds short and every history `good`, with probability at most `c^m`. -/
theorem pr_chain_le [NeZero B] (nxt : ℕ → τ → Fin B → Bits ℓ × τ) (C : Fin B → Bits ℓ) (short : τ → Bool)
    (c : ℝ≥0∞) :
    ∀ (k j m : ℕ) (σ : τ) (good : List (Fin B) → Prop),
      (∀ h : List (Fin B), h.length < k → good h → short (after nxt j h σ) = true →
        pr (fun i : Fin B => (nxt (j + h.length) (after nxt j h σ) i).1 = C i) ≤ c) →
      pr (fun I : Fin k → Fin B =>
          (trace nxt j (List.ofFn I) σ).map Prod.fst = List.ofFn (fun t => C (I t)) ∧
            m ≤ ((trace nxt j (List.ofFn I) σ).map Prod.snd).countP short ∧
            ∀ t : Fin k, good ((List.ofFn I).take t)) ≤ c ^ m
  | 0, j, m, σ, good, _ => by
      rcases Nat.eq_zero_or_pos m with rfl | hm
      · rw [pow_zero]
        exact pr_le_one _
      · refine le_trans (pr_mono fun I hI => ?_) (pr_false.le.trans bot_le)
        obtain ⟨-, hc, -⟩ := hI
        simp [trace] at hc
        omega
  | k + 1, j, m, σ, good, hyp => by
      have : Nonempty (Fin B) := ⟨⟨0, Nat.pos_of_ne_zero (NeZero.ne B)⟩⟩
      rw [← pr_equiv (Fin.consEquiv fun _ : Fin (k + 1) => Fin B :
        Fin B × (Fin k → Fin B) ≃ (Fin (k + 1) → Fin B))]
      -- the rest of the audit from the state round `j` leaves, with `m'` short rounds
      let R : ℕ → Fin B → (Fin k → Fin B) → Prop := fun m' i₀ I =>
        (trace nxt (j + 1) (List.ofFn I) (nxt j σ i₀).2).map Prod.fst = List.ofFn (fun t => C (I t)) ∧
          m' ≤ ((trace nxt (j + 1) (List.ofFn I) (nxt j σ i₀).2).map Prod.snd).countP short ∧
          ∀ t : Fin k, good (i₀ :: (List.ofFn I).take t)
      have hR : ∀ m' i₀, pr (R m' i₀) ≤ c ^ m' := fun m' i₀ =>
        pr_chain_le nxt C short c k (j + 1) m' (nxt j σ i₀).2 (fun h => good (i₀ :: h)) fun h hh hg hs => by
          have := hyp (i₀ :: h) (by simp only [List.length_cons]; omega) hg hs
          simpa [after, Nat.add_assoc, Nat.add_comm 1] using this
      -- what the event says of round `j` and of the rest
      have hsplit : ∀ i₀ (I : Fin k → Fin B),
          ((trace nxt j (List.ofFn (Fin.cons i₀ I : Fin (k + 1) → Fin B)) σ).map Prod.fst =
              List.ofFn (fun t => C ((Fin.cons i₀ I : Fin (k + 1) → Fin B) t)) ∧
            m ≤ ((trace nxt j (List.ofFn (Fin.cons i₀ I : Fin (k + 1) → Fin B)) σ).map Prod.snd).countP short ∧
            ∀ t : Fin (k + 1), good ((List.ofFn (Fin.cons i₀ I : Fin (k + 1) → Fin B)).take t)) →
          good [] ∧ (nxt j σ i₀).1 = C i₀ ∧ R (m - if short σ then 1 else 0) i₀ I := by
        intro i₀ I ⟨h1, h2, h3⟩
        rw [ofFn_cons, ofFn_cons_map] at h1
        rw [ofFn_cons] at h2 h3
        simp only [trace, List.map_cons, List.cons.injEq, List.countP_cons] at h1 h2
        refine ⟨by simpa using h3 0, h1.1, h1.2, by split_ifs at h2 ⊢ <;> simp_all, fun t => ?_⟩
        simpa using h3 t.succ
      by_cases hs : short σ = true
      · by_cases hg : good []
        · have h0 : pr (fun i : Fin B => (nxt j σ i).1 = C i) ≤ c := by
            simpa [after] using hyp [] (by simp) hg (by simpa [after] using hs)
          rcases Nat.eq_zero_or_pos m with rfl | hm
          · rw [pow_zero]
            exact pr_le_one _
          · obtain ⟨m', rfl⟩ : ∃ m', m = m' + 1 := ⟨m - 1, by omega⟩
            refine le_trans (pr_mono fun x hx => ?_) ((pr_and_le _ (R m') c (c ^ m') h0 (hR m')).trans
              (by rw [pow_succ']))
            obtain ⟨-, h1, h2⟩ := hsplit x.1 x.2 hx
            simp only [hs, ↓reduceIte, Nat.add_sub_cancel] at h2
            exact ⟨h1, h2⟩
        · refine le_trans (pr_mono fun x hx => ?_) (pr_false.le.trans bot_le)
          exact hg (hsplit x.1 x.2 hx).1
      · refine le_trans (pr_mono fun x hx => ?_) ((pr_and_le (fun _ => True) (R m) 1 (c ^ m) (pr_le_one _)
          (hR m)).trans (by rw [one_mul]))
        obtain ⟨-, -, h2⟩ := hsplit x.1 x.2 hx
        simp only [hs, Bool.false_eq_true, ↓reduceIte, Nat.sub_zero] at h2
        exact ⟨trivial, h2⟩

/-- **Averaging over the second coordinate.** -/
theorem pr_swap_le {X J : Type} [Fintype X] [Fintype J] [Nonempty X] [Nonempty J] (P : X → J → Prop) (c : ℝ≥0∞)
    (h : ∀ j, pr (fun x => P x j) ≤ c) : pr (fun y : X × J => P y.1 y.2) ≤ c := by
  rw [← pr_equiv (Equiv.prodComm J X)]
  have := pr_prod_le (fun j x => P x j) (fun _ => False) c fun j _ => h j
  rwa [pr_false, zero_add] at this

/-- **The audit from its rounds.** Over a setup `x` and `k` uniform indices: if after every history `h` of fewer than
`k` indices the state is short and answers more than a `c` fraction of the indices with probability at most `ε` over
the setup, the audit passes with at least `m` short rounds with probability at most `c^m + k·ε`. -/
theorem pr_held_le {X : Type} [Fintype X] [Nonempty X] [NeZero B] (nxt : X → ℕ → τ → Fin B → Bits ℓ × τ)
    (σ₀ : X → τ) (C : X → Fin B → Bits ℓ) (short : τ → Bool) (c ε : ℝ≥0∞) (k m : ℕ)
    (hround : ∀ h : List (Fin B), h.length < k →
      pr (fun x : X => short (after (nxt x) 0 h (σ₀ x)) = true ∧
        c < pr (fun i : Fin B => (nxt x h.length (after (nxt x) 0 h (σ₀ x)) i).1 = C x i)) ≤ ε) :
    pr (fun y : X × (Fin k → Fin B) =>
        (trace (nxt y.1) 0 (List.ofFn y.2) (σ₀ y.1)).map Prod.fst = List.ofFn (fun t => C y.1 (y.2 t)) ∧
          m ≤ ((trace (nxt y.1) 0 (List.ofFn y.2) (σ₀ y.1)).map Prod.snd).countP short) ≤ c ^ m + k * ε := by
  have : Nonempty (Fin B) := ⟨⟨0, Nat.pos_of_ne_zero (NeZero.ne B)⟩⟩
  let bad : X → List (Fin B) → Prop := fun x h => short (after (nxt x) 0 h (σ₀ x)) = true ∧
    c < pr (fun i : Fin B => (nxt x h.length (after (nxt x) 0 h (σ₀ x)) i).1 = C x i)
  let E : X → (Fin k → Fin B) → Prop := fun x I =>
    (trace (nxt x) 0 (List.ofFn I) (σ₀ x)).map Prod.fst = List.ofFn (fun t => C x (I t)) ∧
      m ≤ ((trace (nxt x) 0 (List.ofFn I) (σ₀ x)).map Prod.snd).countP short ∧
      ∀ t : Fin k, ¬ bad x ((List.ofFn I).take t)
  refine le_trans (pr_mono fun y hy => ?_) ((pr_or_le (fun y => E y.1 y.2)
    (fun y => ∃ t ∈ (Finset.univ : Finset (Fin k)), bad y.1 ((List.ofFn y.2).take t))).trans (add_le_add ?_ ?_))
  · by_cases hb : ∃ t : Fin k, bad y.1 ((List.ofFn y.2).take t)
    · obtain ⟨t, ht⟩ := hb
      exact Or.inr ⟨t, Finset.mem_univ _, ht⟩
    · exact Or.inl ⟨hy.1, hy.2, fun t ht => hb ⟨t, ht⟩⟩
  · have := pr_prod_le E (fun _ => False) (c ^ m) fun x _ =>
      pr_chain_le (nxt x) (C x) short c k 0 m (σ₀ x) (fun h => ¬ bad x h) fun h _ hg hs => by
        rw [Nat.zero_add]
        exact not_lt.1 fun hlt => hg ⟨hs, hlt⟩
    rwa [pr_false, zero_add] at this
  · refine (pr_exists_le _ _).trans ?_
    calc ∑ t ∈ (Finset.univ : Finset (Fin k)), pr (fun y : X × (Fin k → Fin B) => bad y.1 ((List.ofFn y.2).take t))
        ≤ ∑ _t ∈ (Finset.univ : Finset (Fin k)), ε := Finset.sum_le_sum fun t _ =>
          pr_swap_le (fun x I => bad x ((List.ofFn I).take t)) ε fun I =>
            hround _ (by simp only [List.length_take, List.length_ofFn]; omega)
      _ = k * ε := by rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]

end Trace

/-! ### The sequential audit over the ring bound -/

/-- A bit list read as `S` bits, `false` past its end. -/
def ofList {S : ℕ} (l : List Bool) : Bits S := fun k => l.getD k false

theorem ofList_ofFn {S : ℕ} (b : Bits S) : ofList (List.ofFn b) = b := by
  funext k
  simp [ofList]

theorem seqAnswers_eq {M : IdealModel} {ℓ S B : ℕ} (f : M.Qry → M.Ans)
    (step : ℕ → Bits S → Fin B → Prog M (Bits ℓ × Bits S)) :
    ∀ (j : ℕ) (l : List (Fin B)) (σ : Bits S),
      seqAnswers f step j l σ = (trace (fun j σ i => (step j σ i).run f) j l σ).map Prod.fst
  | _, [], _ => rfl
  | j, i :: is, σ => by
      simp only [seqAnswers, trace, List.map_cons, seqAnswers_eq f step (j + 1) is]

theorem countP_true {α : Type} (l : List α) : l.countP (fun _ => true) = l.length := by
  induction l <;> simp_all

theorem length_trace {τ : Type} {B ℓ : ℕ} (nxt : ℕ → τ → Fin B → Bits ℓ × τ) :
    ∀ (j : ℕ) (l : List (Fin B)) (σ : τ), (trace nxt j l σ).length = l.length
  | _, [], _ => rfl
  | j, _ :: is, σ => by simp [trace, length_trace nxt (j + 1) is]

/-- **The sequential audit from its rounds.** If one round of the audit over the ring bound is answered from an
`S`-bit state with probability at most `c`, off `ε` (`RoundSecureOn`), then every `S`-bit, `(D, Q, L)`-bounded
sequential responder passes `k` rounds with probability at most `c^k + k·ε`. -/
theorem auditSecureOn_of_round {B w p : ℕ} [NeZero p] (sch : Scheme (ringRO B w p)) (S D Q L k : ℕ) (c ε : ℝ≥0∞)
    (h : RoundSecureOn sch S D Q L c ε) (W : Bits sch.n) (A₁ : Preprocessor sch S)
    (A₂ : SeqResponderOn sch S D Q L) :
    pr (seqAcceptsOn sch k W A₁ A₂) ≤ c ^ k + k * ε := by
  have : NeZero sch.B := ⟨sch.B_pos.ne'⟩
  let nxt : (ringRO B w p).Ω × sch.Coins → ℕ → Bits S → Fin sch.B → Bits sch.ℓ × Bits S :=
    fun x j σ i => (A₂.step W (sch.enc W x.1 x.2).1 j σ i).run ((ringRO B w p).answer x.1)
  let σ₀ : (ringRO B w p).Ω × sch.Coins → Bits S :=
    fun x => A₁ W (sch.enc W x.1 x.2).1 (sch.enc W x.1 x.2).2 x.1
  have hre : pr (seqAcceptsOn sch k W A₁ A₂) = pr (fun y : ((ringRO B w p).Ω × sch.Coins) × (Fin k → Fin sch.B) =>
      seqAnswers ((ringRO B w p).answer y.1.1) (A₂.step W (sch.enc W y.1.1 y.1.2).1) 0 (List.ofFn y.2)
        (A₁ W (sch.enc W y.1.1 y.1.2).1 (sch.enc W y.1.1 y.1.2).2 y.1.1) =
      List.ofFn fun t => (sch.enc W y.1.1 y.1.2).2 (y.2 t)) :=
    pr_equiv (Equiv.prodAssoc _ _ _).symm (fun y : ((ringRO B w p).Ω × sch.Coins) × (Fin k → Fin sch.B) =>
      seqAnswers ((ringRO B w p).answer y.1.1) (A₂.step W (sch.enc W y.1.1 y.1.2).1) 0 (List.ofFn y.2)
        (A₁ W (sch.enc W y.1.1 y.1.2).1 (sch.enc W y.1.1 y.1.2).2 y.1.1) =
      List.ofFn fun t => (sch.enc W y.1.1 y.1.2).2 (y.2 t))
  rw [hre]
  refine le_trans (pr_mono fun y hy => ?_) (pr_held_le nxt σ₀ (fun x => (sch.enc W x.1 x.2).2) (fun _ => true) c ε k k
    fun hist hh => le_trans (pr_mono fun x hx => ?_) (h W (fun x => List.ofFn (after (nxt x) 0 hist (σ₀ x)))
      (fun pp σ i => progMap Prod.fst (A₂.step W pp hist.length (ofList σ) i))
      fun pp σ i => ringBounded_progMap _ (A₂.bounded _ _ _ _ _)))
  · rw [seqAnswers_eq] at hy
    refine ⟨hy, ?_⟩
    rw [countP_true, List.length_map, length_trace, List.length_ofFn]
  · refine ⟨by rw [List.length_ofFn], ?_⟩
    simpa only [ofList_ofFn, run_progMap] using hx.2

/-! ### At P2's parameters -/

theorem c_eq : ((501578 : ℝ≥0∞) / 2 ^ 19) = ENNReal.ofReal ((501578 : ℝ) / 2 ^ 19) := by
  rw [ENNReal.ofReal_div_of_pos (by positivity), ENNReal.ofReal_pow (by norm_num), ENNReal.ofReal_ofNat,
    ENNReal.ofReal_ofNat]

theorem c_lt_one : ((501578 : ℝ≥0∞) / 2 ^ 19) < 1 := by
  rw [c_eq, ENNReal.ofReal_lt_one]
  norm_num

/-- `(501,578/2^19)^105 ≤ 1%`. -/
theorem c_pow_le : ((501578 : ℝ≥0∞) / 2 ^ 19) ^ 105 ≤ ENNReal.ofReal Params.δ := by
  rw [c_eq, ← ENNReal.ofReal_pow (by positivity)]
  exact ENNReal.ofReal_le_ofReal (by norm_num [Params.δ])

theorem eps_lt_one : (2⁻¹ : ℝ≥0∞) ^ 136 < 1 :=
  pow_lt_one₀ bot_le (ENNReal.inv_lt_one.2 ENNReal.one_lt_two) (by norm_num)

/-- `105 · 2^-136 ≤ 2^-129`. -/
theorem eps_le : ((105 : ℕ) : ℝ≥0∞) * (2⁻¹ : ℝ≥0∞) ^ 136 ≤ (2⁻¹ : ℝ≥0∞) ^ 129 := by
  have h7 : ((105 : ℕ) : ℝ≥0∞) * (2⁻¹ : ℝ≥0∞) ^ 7 ≤ 1 := by
    rw [← ENNReal.inv_pow, ← div_eq_mul_inv, ENNReal.div_le_iff (by positivity) (by simp), one_mul]
    exact_mod_cast (by norm_num : (105 : ℕ) ≤ 2 ^ 7)
  calc ((105 : ℕ) : ℝ≥0∞) * (2⁻¹ : ℝ≥0∞) ^ 136 = ((105 : ℕ) : ℝ≥0∞) * (2⁻¹ : ℝ≥0∞) ^ 7 * (2⁻¹ : ℝ≥0∞) ^ 129 := by
        rw [mul_assoc, ← pow_add]
    _ ≤ 1 * (2⁻¹ : ℝ≥0∞) ^ 129 := by gcongr
    _ = _ := one_mul _

/-- **The whole audit from the per-round form, at P2's deployed point**: `P2RingRoundSecure D Q L` gives
`P2RingAuditSecure D Q L`, through `(501,578/2^19)^105 + 105 · 2^-136 ≤ 1% + 2^-129`. -/
theorem p2RingAuditSecure_of_round (D Q L : ℕ) (h : Pous.P2.Assumptions.P2RingRoundSecure D Q L) :
    Pous.P2.Assumptions.P2RingAuditSecure D Q L := by
  intro W A₁ A₂
  refine (auditSecureOn_of_round _ _ D Q L 105 _ _ h W A₁ A₂).trans ?_
  gcongr
  · exact c_pow_le
  · exact eps_le.trans le_self_add

end PousP2Round
