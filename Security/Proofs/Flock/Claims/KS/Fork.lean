import Proofs.Flock.Claims.Aggregate
import Proofs.Flock.Claims.ERepCost

/-!
# The finders' events, covered by `B`

A per-tape knowledge bound escapes on events where the prover breaks a commitment. Each such event the soundness proof
uses is a collision among the verifier's own `H` calls: in two runs at one tape sharing the coins before some round
`j` (a fork), or in one run. `B` guesses `j`, runs that fork and outputs the first collision among both runs' verifier
calls, and `findColl` misses no collision of a list of evaluations (`findColl_none`). So a fork event has probability at
most `B`'s fork collision probability at its round (`Pr_fork_le`), and a one-run event at most `Pr[B]`
(`Pr_one_le_PrB`).

`Escapes` bundles one prover's escape events at `x`: fork events with weights under a square root, and one-run events
with linear weights. When they fit `C` (at most `mult` fork events at one round, `mult · Σ c ≤ 2C`, `Σ β ≤ k(x)`),
their term averages, over the tape, to at most `√((√(2·k(x)·C) + k(x))² · Pr[B])` (`Escapes.le`), so four times it is
at most `√(γ · Pr[B])` with Flock's `γ = 16·(√(2·k(x)·C) + k(x))²`.
-/

namespace Claims

/-- `findColl` finds nothing only if no two pairs of `l` share an output with different inputs. -/
theorem findColl_none : ∀ l : List (Msg × Msg), findColl l = none → ∀ p ∈ l, ∀ q ∈ l, p.2 = q.2 → p.1 = q.1
  | [], _ => by simp
  | (q₀, a) :: tl, h => by
    simp only [findColl] at h
    split at h
    · simp at h
    · rename_i hf
      have ih := findColl_none tl h
      have hno : ∀ p ∈ tl, ¬ (p.1 ≠ q₀ ∧ p.2 = a) := by
        intro p hp hc
        have := List.find?_eq_none.1 hf p hp
        simp [hc.1, hc.2] at this
      intro p hp q hq hpq
      simp only [List.mem_cons] at hp hq
      rcases hp with rfl | hp <;> rcases hq with rfl | hq
      · rfl
      · by_contra hne; exact hno q hq ⟨fun e => hne e.symm, hpq.symm⟩
      · by_contra hne; exact hno p hp ⟨hne, hpq⟩
      · exact ih p hp q hq hpq

/-- The calls `l` collide: two different inputs among them have one `H` value. -/
def TiedColl (H : Msg → Msg) (l : List (Msg × Msg)) : Prop :=
  ∃ u v, Collision H u v ∧ u ∈ l.map Prod.fst ∧ v ∈ l.map Prod.fst

theorem TiedColl.append_right {H : Msg → Msg} {l₁ : List (Msg × Msg)} (h : TiedColl H l₁) (l₂ : List (Msg × Msg)) :
    TiedColl H (l₁ ++ l₂) := by
  obtain ⟨u, v, hc, hu, hv⟩ := h
  exact ⟨u, v, hc, by simp only [List.map_append, List.mem_append]; exact Or.inl hu,
    by simp only [List.map_append, List.mem_append]; exact Or.inl hv⟩

/-- **`findColl` misses no collision** of a list of evaluations of `H`. -/
theorem findColl_isSome (H : Msg → Msg) {l : List (Msg × Msg)} (hl : ∀ p ∈ l, H p.1 = p.2) (h : TiedColl H l) :
    (findColl l).isSome := by
  obtain ⟨u, v, ⟨hne, huv⟩, hu, hv⟩ := h
  obtain ⟨p, hp, rfl⟩ := List.mem_map.1 hu
  obtain ⟨q, hq, rfl⟩ := List.mem_map.1 hv
  cases hf : findColl l with
  | some _ => rfl
  | none => exact absurd (findColl_none l hf p hp q hq (by rw [← hl p hp, ← hl q hq, huv])) hne

/-! ## Averages -/

theorem Ex_sum {α I : Type} [Fintype α] (s : Finset I) (f : I → α → ℝ) :
    Ex (fun a => ∑ i ∈ s, f i a) = ∑ i ∈ s, Ex (f i) := by
  unfold Ex
  rw [Finset.sum_comm, Finset.sum_div]

/-- An average over the first coordinate of an event's probability over the second is the event's probability. -/
theorem Ex_Pr {α β : Type} [Fintype α] [Fintype β] (p : α → β → Prop) :
    Ex (fun a => Pr (p a)) = Pr (fun s : α × β => p s.1 s.2) := by
  classical
  rw [← Ex_ind (fun s : α × β => p s.1 s.2), Ex_prod]
  congr 1; funext a
  exact (Ex_ind (p a)).symm

section B

variable {Ω Coin : Type} [Fintype Ω] [Nonempty Ω] [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

theorem PrB_nonneg (H : Msg → Msg) (V : Verifier Coin) (P : Prover Ω) (x : Msg) : 0 ≤ PrB H V P x := by
  unfold PrB
  split_ifs
  · exact Pr_nonneg _
  · exact le_rfl

theorem PrB_le_one (H : Msg → Msg) (V : Verifier Coin) (P : Prover Ω) (x : Msg) : PrB H V P x ≤ 1 := by
  unfold PrB
  split_ifs
  · exact Pr_le_one _
  · exact zero_le_one

omit [Fintype Ω] [Nonempty Ω] [Fintype Coin] [Nonempty Coin] in
/-- A verifier that takes no rounds at `x` makes no `H` calls there. -/
theorem run_vLog_of_k_zero (H : Msg → Msg) (P : Prover Ω) (V : Verifier Coin) (x : Msg) (h : V.k x = 0) (ω : Ω)
    (r : Fin (V.k x) → Coin) : (run H P V x ω r).vLog = [] := by
  unfold run
  generalize (fun as => V.next x as ((List.range as.length).map (coinSeq r))) = vn
  rw [h]
  rfl

/-- **A fork event is covered by `B`**: an event on two runs at one tape, sharing the coins before round `j`, on which
the verifier's calls collide, has probability at most `B`'s fork collision probability at `j`. -/
theorem Pr_fork_le (H : Msg → Msg) (V : Verifier Coin) (P : Prover Ω) (x : Msg) (hk : 0 < V.k x)
    (j : Fin (V.k x)) (ev : Ω → (Fin (V.k x) → Coin) → (Fin (V.k x) → Coin) → Prop)
    (hev : ∀ ω r r', ev ω r r' →
      TiedColl H ((run H P V x ω r).vLog ++ (run H P V x ω (forkCoins j r r')).vLog)) :
    Pr (fun s : Ω × (Fin (V.k x) → Coin) × (Fin (V.k x) → Coin) => ev s.1 s.2.1 s.2.2) ≤
      forkPr H V P x hk j := by
  unfold forkPr
  refine Pr_mono fun s hs => ?_
  refine findColl_isSome H ?_ (hev _ _ _ hs)
  intro p hp
  rcases List.mem_append.1 hp with hp | hp
  · exact (run_logs H P V x _ _).2 p hp
  · exact (run_logs H P V x _ _).2 p hp

/-- **A one-run event is covered by every fork of `B`**. -/
theorem Pr_one_le_forkPr (H : Msg → Msg) (V : Verifier Coin) (P : Prover Ω) (x : Msg) (hk : 0 < V.k x)
    (j : Fin (V.k x)) (ev₁ : Ω → (Fin (V.k x) → Coin) → Prop)
    (hev : ∀ ω r, ev₁ ω r → TiedColl H (run H P V x ω r).vLog) :
    Pr (fun s : Ω × (Fin (V.k x) → Coin) => ev₁ s.1 s.2) ≤ forkPr H V P x hk j := by
  rw [← Pr_drop (γ := Fin (V.k x) → Coin) fun ω r => ev₁ ω r]
  exact Pr_fork_le H V P x hk j (fun ω r _ => ev₁ ω r) fun ω r r' h => (hev ω r h).append_right _

/-- **A one-run event is covered by `B`**: an event on one run on which the verifier's calls collide has probability at
most `Pr[B]`. -/
theorem Pr_one_le_PrB (H : Msg → Msg) (V : Verifier Coin) (P : Prover Ω) (x : Msg)
    (ev₁ : Ω → (Fin (V.k x) → Coin) → Prop) (hev : ∀ ω r, ev₁ ω r → TiedColl H (run H P V x ω r).vLog) :
    Pr (fun s : Ω × (Fin (V.k x) → Coin) => ev₁ s.1 s.2) ≤ PrB H V P x := by
  by_cases hk : 0 < V.k x
  · have hk' : (0 : ℝ) < V.k x := by exact_mod_cast hk
    rw [PrB_eq_avg H V P x hk, le_div_iff₀ hk']
    calc Pr (fun s : Ω × (Fin (V.k x) → Coin) => ev₁ s.1 s.2) * V.k x
        = ∑ _j : Fin (V.k x), Pr (fun s : Ω × (Fin (V.k x) → Coin) => ev₁ s.1 s.2) := by
          rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, mul_comm]
      _ ≤ ∑ j, forkPr H V P x hk j := Finset.sum_le_sum fun j _ => Pr_one_le_forkPr H V P x hk j ev₁ hev
  · have h0 : V.k x = 0 := by omega
    rw [Pr_congr (q := fun _ => False) fun s => ⟨fun h => ?_, False.elim⟩, Pr_false]
    · exact PrB_nonneg H V P x
    obtain ⟨u, v, _, hu, _⟩ := hev s.1 s.2 h
    rw [run_vLog_of_k_zero H P V x h0] at hu
    simp at hu

end B

/-! ## One prover's escapes -/

/-- **One prover's escapes at `x`**: the events on which a per-tape knowledge bound escapes, each tied to a collision
among the verifier's `H` calls. Fork events `ev i` on two runs at one tape sharing the coins before round `j i`, with
weights `c i`, each under a square root; one-run events `ev₁ l` with linear weights `β l`. -/
structure Escapes {Ω Coin : Type} [Inhabited Coin] (H : Msg → Msg) (V : Verifier Coin) (P : Prover Ω) (x : Msg) where
  I : Type
  [finI : Fintype I]
  c : I → ℝ
  j : I → Fin (V.k x)
  ev : I → Ω → (Fin (V.k x) → Coin) → (Fin (V.k x) → Coin) → Prop
  L : Type
  [finL : Fintype L]
  β : L → ℝ
  ev₁ : L → Ω → (Fin (V.k x) → Coin) → Prop
  c_nonneg : ∀ i, 0 ≤ c i
  β_nonneg : ∀ l, 0 ≤ β l
  tied : ∀ i ω r r', ev i ω r r' →
    TiedColl H ((run H P V x ω r).vLog ++ (run H P V x ω (forkCoins (j i) r r')).vLog)
  tied₁ : ∀ l ω r, ev₁ l ω r → TiedColl H (run H P V x ω r).vLog

attribute [instance] Escapes.finI Escapes.finL

section Escapes

variable {Ω Coin : Type} [Fintype Ω] [Nonempty Ω] [Fintype Coin] [Nonempty Coin] [Inhabited Coin]
  {H : Msg → Msg} {V : Verifier Coin} {P : Prover Ω} {x : Msg}

/-- The escapes' term at tape `ω`: `Σ_i √(c_i · Pr_{r,r'}[ev_i]) + Σ_l β_l · Pr_r[ev₁_l]`. -/
noncomputable def Escapes.term (e : Escapes H V P x) (ω : Ω) : ℝ :=
  (∑ i, Real.sqrt (e.c i * Pr fun s : (Fin (V.k x) → Coin) × (Fin (V.k x) → Coin) => e.ev i ω s.1 s.2)) +
    ∑ l, e.β l * Pr fun r : Fin (V.k x) → Coin => e.ev₁ l ω r

/-- The escapes fit the weight `C`: at most `mult` fork events at one round, `mult · Σ_i c_i ≤ 2C`, and
`Σ_l β_l ≤ k(x)`. -/
def Escapes.Fits (e : Escapes H V P x) (C : ℝ) : Prop :=
  ∃ mult : ℕ, (∀ r, (Finset.univ.filter fun i => e.j i = r).card ≤ mult) ∧ mult * ∑ i, e.c i ≤ 2 * C ∧
    ∑ l, e.β l ≤ V.k x

omit [Fintype Ω] [Nonempty Ω] [Nonempty Coin] in
theorem Escapes.term_nonneg (e : Escapes H V P x) (ω : Ω) : 0 ≤ e.term ω :=
  add_nonneg (Finset.sum_nonneg fun _ _ => Real.sqrt_nonneg _)
    (Finset.sum_nonneg fun l _ => mul_nonneg (e.β_nonneg l) (Pr_nonneg _))

/-- **The escapes, into one root**: escapes that fit `C` average, over the tape, to at most
`√((√(2·k(x)·C) + k(x))² · Pr[B])`; four times that is `√(16·(√(2·k(x)·C) + k(x))² · Pr[B])`. -/
theorem Escapes.le (e : Escapes H V P x) (C : ℝ) (hfit : e.Fits C) :
    4 * Ex e.term ≤ Real.sqrt (16 * (Real.sqrt (2 * V.k x * C) + V.k x) ^ 2 * PrB H V P x) := by
  obtain ⟨mult, hm, hmc, hβ⟩ := hfit
  have hc0 : 0 ≤ ∑ i, e.c i := Finset.sum_nonneg fun i _ => e.c_nonneg i
  have hC : 0 ≤ 2 * C := (mul_nonneg (Nat.cast_nonneg mult) hc0).trans hmc
  have hp0 := PrB_nonneg H V P x
  have hp1 := PrB_le_one H V P x
  have hkx : (0 : ℝ) ≤ V.k x := Nat.cast_nonneg _
  set A : e.I → ℝ := fun i =>
    Pr fun s : Ω × (Fin (V.k x) → Coin) × (Fin (V.k x) → Coin) => e.ev i s.1 s.2.1 s.2.2 with hA
  set A₁ : e.L → ℝ := fun l => Pr fun s : Ω × (Fin (V.k x) → Coin) => e.ev₁ l s.1 s.2 with hA₁
  have hEx : Ex e.term ≤ (∑ i, Real.sqrt (e.c i * A i)) + ∑ l, e.β l * A₁ l := by
    unfold Escapes.term
    rw [Ex_add, Ex_sum, Ex_sum]
    refine add_le_add (Finset.sum_le_sum fun i _ => ?_) (le_of_eq (Finset.sum_congr rfl fun l _ => ?_))
    · refine (Ex_sqrt_le _ fun ω => mul_nonneg (e.c_nonneg i) (Pr_nonneg _)).trans (le_of_eq ?_)
      rw [Ex_mul_left, Ex_Pr]
    · rw [Ex_mul_left, Ex_Pr]
  have hfork : (∑ i, Real.sqrt (e.c i * A i)) ≤ Real.sqrt (2 * V.k x * C * PrB H V P x) := by
    by_cases hk : 0 < V.k x
    · refine (collapse_B Finset.univ H V P x hk e.c A (fun i _ => e.c_nonneg i) (fun i _ => Pr_nonneg _) e.j
        (fun i _ => Pr_fork_le H V P x hk (e.j i) (e.ev i) (e.tied i)) mult fun r => hm r).trans ?_
      refine Real.sqrt_le_sqrt ?_
      have := mul_le_mul_of_nonneg_left hmc (mul_nonneg hkx hp0)
      nlinarith [this]
    · have : IsEmpty e.I := ⟨fun i => absurd (e.j i).pos hk⟩
      simp only [Finset.univ_eq_empty, Finset.sum_empty]
      exact Real.sqrt_nonneg _
  have hlin : ∑ l, e.β l * A₁ l ≤ V.k x * PrB H V P x := by
    refine (Finset.sum_le_sum fun l _ =>
      mul_le_mul_of_nonneg_left (Pr_one_le_PrB H V P x (e.ev₁ l) (e.tied₁ l)) (e.β_nonneg l)).trans ?_
    rw [← Finset.sum_mul]
    exact mul_le_mul_of_nonneg_right hβ hp0
  have hroot := sqrt_add_lin (C := 2 * V.k x * C) (β := V.k x) (by nlinarith) hkx hp0 hp1
  have h16 : Real.sqrt (16 * (Real.sqrt (2 * V.k x * C) + V.k x) ^ 2 * PrB H V P x) =
      4 * Real.sqrt ((Real.sqrt (2 * V.k x * C) + V.k x) ^ 2 * PrB H V P x) := by
    rw [mul_assoc, Real.sqrt_mul (by norm_num : (0 : ℝ) ≤ 16), show (16 : ℝ) = 4 ^ 2 by norm_num,
      Real.sqrt_sq (by norm_num)]
  rw [h16]
  linarith

end Escapes

end Claims
