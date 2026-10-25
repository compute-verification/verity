import Proofs.Flock.Claims.KS.ZkFitsW
import Proofs.Flock.Claims.KS.ForkSlack

/-!
# The masked weight's room for one more event per round, and the slack's escapes

The fork bound's slack (`Claims.Flock.forkSlack`) adds one fork event per round. The plain weight pays for it with
level 0's unused `2·2^{d₀}·Q₀` (`FlockDeploy.KS.plain_fork_room`). The masked weight already spends level 0's on its
fork event (`Claims.Flock.cZW`), but counted per round (`Claims.Escapes.FitsForkR`) only round `0`'s events are charged
twice, and where round `0` holds only level 0's event of each table (`hr0`: the deployed level caps, pads and `τ` are
absorbed after the link message), the weight's room past the joined escapes (`Claims.Flock.joinCW`) and the fork of
weight `k` is every level cap's and both pads roots' weight once more (`Claims.Flock.roomW`, `zk_fork_room`): each
table's weight pays twice for its fork events but `τ`'s `2` and level 0's, and `k`'s ceiling fits in the `2` per table.

The slack's events (`Claims.Flock.slackG`) merge with the joined escapes by round (`Claims.Flock.slackEscW`): round
`t`'s event is the tables' fork events there or the slack's, of weight `(√J_t + √S_t)²` (`Claims.Flock.mergeCW`), `J_t`
the joined weight and `S_t` every table's slack weight (`Claims.Flock.slackTW`, `slackCW`), since
`√(J·p) + √(S·q) ≤ (√J + √S)·√Pr[E ∨ G]` when `p, q ≤ Pr[E ∨ G]`. Each table's fork terms at the slack are at most the
merged escapes' term (`Claims.Flock.slackEscW_term`, from `forkSlack_le` and `sum_sqrt_slack_le`), and the merged
weights exceed the joined ones by at most `4·√(Σ J·Σ S) + 2·Σ S` (`Claims.Flock.mergeCW_le`, Cauchy–Schwarz), so where
that fits the room (`Claims.Flock.SlackFitW`) the merged escapes fit the deployed weight per round
(`slackEscW_fitsForkR_zk`). A slack event is an accepting fork or an accepting run with a collision, so the merged
escapes are on accepted first runs, the fork accepting or the first run alone colliding (`Claims.Escapes.OnAcc₁`,
`Claims.Flock.slackEscW_onAcc₁`).
-/

namespace Claims

/-- **Escapes on accepted first runs**: each fork event holds only where its first run accepts and its fork accepts or
the first run alone has a collision among the verifier's calls, and each one-run event only where its run accepts. -/
def Escapes.OnAcc₁ {Ω Coin : Type} [Inhabited Coin] {H : Msg → Msg} {V : Verifier Coin} {P : Prover Ω} {x : Msg}
    (e : Escapes H V P x) : Prop :=
  (∀ i ω r r', e.ev i ω r r' → (run H P V x ω r).out = .acc ∧
      ((run H P V x ω (forkCoins (e.j i) r r')).out = .acc ∨ TiedColl H (run H P V x ω r).vLog)) ∧
    ∀ l ω r, e.ev₁ l ω r → (run H P V x ω r).out = .acc

theorem Escapes.OnAcc.onAcc₁ {Ω Coin : Type} [Inhabited Coin] {H : Msg → Msg} {V : Verifier Coin} {P : Prover Ω}
    {x : Msg} {e : Escapes H V P x} (h : e.OnAcc) : e.OnAcc₁ :=
  ⟨fun i ω r r' hi => ⟨(h.1 i ω r r' hi).1, Or.inl (h.1 i ω r r' hi).2⟩, h.2⟩

end Claims

namespace Claims.Flock

/-- **The joined masked weight at round `r`**: every table's fork events there (`joinEscW`'s weights). -/
noncomputable def joinCW {K : ℕ} (n : ℕ) (sch : ℕ → FlockSoundness.Accounting.Schedule) (Np : ℕ → ℕ)
    (rd : (j : ℕ) → IZW (sch j).levels.length → Fin K) (r : Fin K) : ℝ :=
  ∑ j ∈ Finset.range n, ∑ i ∈ Finset.univ.filter (fun i => rd j i = r), cZW (sch j) (Np j) i

theorem joinCW_nonneg {K : ℕ} (n : ℕ) (sch : ℕ → FlockSoundness.Accounting.Schedule) (Np : ℕ → ℕ)
    (rd : (j : ℕ) → IZW (sch j).levels.length → Fin K) (r : Fin K) : 0 ≤ joinCW n sch Np rd r :=
  Finset.sum_nonneg fun _ _ => Finset.sum_nonneg fun _ _ => cZW_nonneg _ _ _

/-- **The masked weight's room**: every table's level caps' weights `Σ_{r,ℓ} 2^{d_{ℓ+1}}·Q_{ℓ+1}` and pads roots'
`2·N_p²`. -/
noncomputable def roomW (n : ℕ) (sch : ℕ → FlockSoundness.Accounting.Schedule) (Np : ℕ → ℕ) : ℝ :=
  ∑ j ∈ Finset.range n,
    ((∑ a : Fin 2 × Fin (sch j).levels.length, cOf (sch j) (a.2.val + 1)) + 2 * ((Np j : ℝ) * Np j))

theorem roomW_nonneg (n : ℕ) (sch : ℕ → FlockSoundness.Accounting.Schedule) (Np : ℕ → ℕ) : 0 ≤ roomW n sch Np :=
  Finset.sum_nonneg fun _ _ => add_nonneg (Finset.sum_nonneg fun _ _ => cOf_nonneg _ _) (by positivity)

/-- **The joined weights are every table's.** -/
theorem joinCW_sum {K : ℕ} (n : ℕ) (sch : ℕ → FlockSoundness.Accounting.Schedule) (Np : ℕ → ℕ)
    (rd : (j : ℕ) → IZW (sch j).levels.length → Fin K) :
    ∑ r, joinCW n sch Np rd r = ∑ j ∈ Finset.range n, ∑ i, cZW (sch j) (Np j) i := by
  unfold joinCW
  rw [Finset.sum_comm]
  exact Finset.sum_congr rfl fun j _ => Finset.sum_fiberwise _ _ _

/-- **Round `0`'s joined weight is at most level 0's events'**, where only level 0's event of a table is at round
`0`. -/
theorem joinCW_zero_le {K : ℕ} (n : ℕ) (sch : ℕ → FlockSoundness.Accounting.Schedule) (Np : ℕ → ℕ)
    (rd : (j : ℕ) → IZW (sch j).levels.length → Fin K) (hr0 : ∀ j i, (rd j i).val = 0 → i = .inr ())
    (r : Fin K) (hr : r.val = 0) :
    joinCW n sch Np rd r ≤ ∑ j ∈ Finset.range n, cZW (sch j) (Np j) (.inr ()) := by
  refine Finset.sum_le_sum fun j _ => ?_
  have hsub : Finset.univ.filter (fun i => rd j i = r) ⊆ {.inr ()} := fun i hi => by
    rw [Finset.mem_singleton]
    exact hr0 j i ((congrArg Fin.val (Finset.mem_filter.1 hi).2).trans hr)
  refine (Finset.sum_le_sum_of_subset_of_nonneg hsub fun i _ _ => cZW_nonneg _ _ i).trans ?_
  rw [Finset.sum_singleton]

/-- **The room and level 0's events are every table's weight but `2`.** -/
theorem roomW_add (n : ℕ) (sch : ℕ → FlockSoundness.Accounting.Schedule) (Np : ℕ → ℕ) :
    roomW n sch Np + ∑ j ∈ Finset.range n, cZW (sch j) (Np j) (.inr ()) =
      ∑ j ∈ Finset.range n, ∑ i, cZW (sch j) (Np j) i - 2 * n := by
  have hper : ∀ j, (∑ a : Fin 2 × Fin (sch j).levels.length, cOf (sch j) (a.2.val + 1)) +
      2 * ((Np j : ℝ) * Np j) + cZW (sch j) (Np j) (.inr ()) = ∑ i, cZW (sch j) (Np j) i - 2 := fun j => by
    have hz : cZW (sch j) (Np j) (.inr ()) =
        2 ^ (FlockSoundness.lvOf (sch j) 0).logLen * (2 * (FlockSoundness.lvOf (sch j) 0).queries) := rfl
    rw [cZW_sum, hz]
    ring
  unfold roomW
  rw [← Finset.sum_add_distrib, Finset.sum_congr rfl fun j _ => hper j, Finset.sum_sub_distrib, Finset.sum_const,
    Finset.card_range, nsmul_eq_mul]
  ring

theorem sum_sqrt_mul_le {ι : Type} [Fintype ι] (f g : ι → ℝ) (hf : ∀ i, 0 ≤ f i) (hg : ∀ i, 0 ≤ g i) :
    ∑ i, Real.sqrt (f i * g i) ≤ Real.sqrt ((∑ i, f i) * ∑ i, g i) := by
  rw [Real.sqrt_mul (Finset.sum_nonneg fun i _ => hf i)]
  exact (le_of_eq (Finset.sum_congr rfl fun i _ => Real.sqrt_mul (hf i) (g i))).trans
    (Real.sum_sqrt_mul_sqrt_le Finset.univ hf hg)

/-! ## The slack's weights, merged with the joined ones -/

section Slack

variable {K : ℕ} (n : ℕ) (sch : ℕ → FlockSoundness.Accounting.Schedule) (Np : ℕ → ℕ)
  (rd : (j : ℕ) → IZW (sch j).levels.length → Fin K) (δ : ℕ → ℝ)

/-- **A fork event's weight is at most the joined weight at its round.** -/
theorem cZW_le_joinCW {j : ℕ} (hj : j < n) (i : IZW (sch j).levels.length) :
    cZW (sch j) (Np j) i ≤ joinCW n sch Np rd (rd j i) := by
  have h1 : cZW (sch j) (Np j) i ≤
      ∑ i' ∈ Finset.univ.filter (fun i' => rd j i' = rd j i), cZW (sch j) (Np j) i' :=
    Finset.single_le_sum (f := cZW (sch j) (Np j)) (fun i' _ => cZW_nonneg _ _ i')
      (by simp only [Finset.mem_filter, Finset.mem_univ, true_and])
  exact h1.trans (Finset.single_le_sum (f := fun j' => ∑ i' ∈ Finset.univ.filter (fun i' => rd j' i' = rd j i),
      cZW (sch j') (Np j') i') (fun j' _ => Finset.sum_nonneg fun i' _ => cZW_nonneg _ _ i') (Finset.mem_range.2 hj))

/-- **Table `j`'s slack weight at round `t`**: `(Σ_i √(c_i·w_{i,t}))²`, `w_{i,t}` the slack's weight at round `t` of
the fork at `i`'s round (`slackW`). -/
noncomputable def slackCW (j : ℕ) (t : Fin K) : ℝ :=
  (∑ i, Real.sqrt (cZW (sch j) (Np j) i * slackW K δ (rd j i).val t.val)) ^ 2

/-- **The slack weight at round `t`**: every table's. -/
noncomputable def slackTW (t : Fin K) : ℝ :=
  ∑ j ∈ Finset.range n, slackCW sch Np rd δ j t

theorem slackTW_nonneg (t : Fin K) : 0 ≤ slackTW n sch Np rd δ t :=
  Finset.sum_nonneg fun _ _ => sq_nonneg _

theorem slackCW_le_slackTW {j : ℕ} (hj : j < n) (t : Fin K) : slackCW sch Np rd δ j t ≤ slackTW n sch Np rd δ t :=
  Finset.single_le_sum (f := fun j => slackCW sch Np rd δ j t) (fun _ _ => sq_nonneg _) (Finset.mem_range.2 hj)

/-- **The merged weight at round `t`**: `(√J_t + √S_t)²`, the joined weight's and the slack's. -/
noncomputable def mergeCW (t : Fin K) : ℝ :=
  (Real.sqrt (joinCW n sch Np rd t) + Real.sqrt (slackTW n sch Np rd δ t)) ^ 2

theorem mergeCW_eq (t : Fin K) : mergeCW n sch Np rd δ t = joinCW n sch Np rd t +
    2 * Real.sqrt (joinCW n sch Np rd t * slackTW n sch Np rd δ t) + slackTW n sch Np rd δ t := by
  unfold mergeCW
  rw [Real.sqrt_mul (joinCW_nonneg n sch Np rd t), add_sq, Real.sq_sqrt (joinCW_nonneg n sch Np rd t),
    Real.sq_sqrt (slackTW_nonneg n sch Np rd δ t)]
  ring

/-- **The slack fits the masked weight's room**: `4·√(Σ_t J_t · Σ_t S_t) + 2·Σ_t S_t ≤` the room. -/
def SlackFitW : Prop :=
  4 * Real.sqrt ((∑ t, joinCW n sch Np rd t) * ∑ t, slackTW n sch Np rd δ t) + 2 * ∑ t, slackTW n sch Np rd δ t ≤
    roomW n sch Np

/-- **The merged weights exceed the joined ones by at most `4·√(Σ J·Σ S) + 2·Σ S`**, a round `t₀` counted twice:
`(√J + √S)² = J + 2√(J·S) + S` and Cauchy–Schwarz. -/
theorem mergeCW_le (t₀ : Fin K) :
    mergeCW n sch Np rd δ t₀ + ∑ t, mergeCW n sch Np rd δ t ≤
      joinCW n sch Np rd t₀ + ∑ t, joinCW n sch Np rd t +
        (4 * Real.sqrt ((∑ t, joinCW n sch Np rd t) * ∑ t, slackTW n sch Np rd δ t) +
          2 * ∑ t, slackTW n sch Np rd δ t) := by
  have hJ := joinCW_nonneg n sch Np rd
  have hS := slackTW_nonneg n sch Np rd δ
  have hcs := sum_sqrt_mul_le (joinCW n sch Np rd) (slackTW n sch Np rd δ) hJ hS
  have hJ₀ : joinCW n sch Np rd t₀ ≤ ∑ t, joinCW n sch Np rd t :=
    Finset.single_le_sum (f := joinCW n sch Np rd) (fun t _ => hJ t) (Finset.mem_univ t₀)
  have hS₀ : slackTW n sch Np rd δ t₀ ≤ ∑ t, slackTW n sch Np rd δ t :=
    Finset.single_le_sum (f := slackTW n sch Np rd δ) (fun t _ => hS t) (Finset.mem_univ t₀)
  have h₀ : Real.sqrt (joinCW n sch Np rd t₀ * slackTW n sch Np rd δ t₀) ≤
      Real.sqrt ((∑ t, joinCW n sch Np rd t) * ∑ t, slackTW n sch Np rd δ t) :=
    Real.sqrt_le_sqrt (mul_le_mul hJ₀ hS₀ (hS t₀) (Finset.sum_nonneg fun t _ => hJ t))
  have hsum : ∑ t, mergeCW n sch Np rd δ t = ∑ t, joinCW n sch Np rd t +
      2 * ∑ t, Real.sqrt (joinCW n sch Np rd t * slackTW n sch Np rd δ t) + ∑ t, slackTW n sch Np rd δ t := by
    rw [Finset.mul_sum, ← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl fun t _ => mergeCW_eq n sch Np rd δ t
  rw [hsum, mergeCW_eq n sch Np rd δ t₀]
  linarith

end Slack

/-! ## The joined escapes and the slack's events, merged by round -/

section SlackEsc

variable {Ω Coin : Type} [Inhabited Coin] (H : Msg → Msg) (V : Verifier Coin) (P : Prover Ω) (x : Msg)

/-- **The joined escapes and the slack's events, merged by round**: round `t`'s event is the tables' fork events there
(`joinEscW`'s) or the slack's (`slackG`), of the merged weight (`mergeCW`). -/
noncomputable def slackEscW (n : ℕ) (sch : ℕ → FlockSoundness.Accounting.Schedule) (Np : ℕ → ℕ)
    (rd : (j : ℕ) → IZW (sch j).levels.length → Fin (V.k x))
    (ev : (j : ℕ) → IZW (sch j).levels.length → Ω → (Fin (V.k x) → Coin) → (Fin (V.k x) → Coin) → Prop)
    (ev₁ : ℕ → Ω → (Fin (V.k x) → Coin) → Prop)
    (tied : ∀ j i ω s s', ev j i ω s s' →
      TiedColl H ((run H P V x ω s).vLog ++ (run H P V x ω (forkCoins (rd j i) s s')).vLog))
    (tied₁ : ∀ j ω s, ev₁ j ω s → TiedColl H (run H P V x ω s).vLog) (δ : ℕ → ℝ) : Escapes H V P x where
  I := Fin (V.k x)
  c t := mergeCW n sch Np rd δ t
  j t := t
  ev t ω s s' := (∃ j < n, ∃ i, rd j i = t ∧ ev j i ω s s') ∨ slackG H V P x t ω s s'
  L := Unit
  β _ := 1
  ev₁ _ ω s := ∃ j < n, ev₁ j ω s
  c_nonneg _ := sq_nonneg _
  β_nonneg _ := zero_le_one
  tied t ω s s' := by
    rintro (⟨j, -, i, rfl, h⟩ | h)
    · exact tied j i ω s s' h
    · exact slackG_tied H V P x t ω s s' h
  tied₁ _ ω s := by
    rintro ⟨j, -, h⟩
    exact tied₁ j ω s h

variable {H V P x}

section Args

variable (n : ℕ) (sch : ℕ → FlockSoundness.Accounting.Schedule) (Np : ℕ → ℕ)
  (rd : (j : ℕ) → IZW (sch j).levels.length → Fin (V.k x))
  (ev : (j : ℕ) → IZW (sch j).levels.length → Ω → (Fin (V.k x) → Coin) → (Fin (V.k x) → Coin) → Prop)
  (ev₁ : ℕ → Ω → (Fin (V.k x) → Coin) → Prop)
  (tied : ∀ j i ω s s', ev j i ω s s' →
    TiedColl H ((run H P V x ω s).vLog ++ (run H P V x ω (forkCoins (rd j i) s s')).vLog))
  (tied₁ : ∀ j ω s, ev₁ j ω s → TiedColl H (run H P V x ω s).vLog) (δ : ℕ → ℝ)

/-- **The merged escapes fit `C` with the fork event of weight `c`, counted per round**, when the slack fits the room
and `2·c`, round `0`'s joined weight, every round's and the room are at most `2C` (`mergeCW_le`). -/
theorem slackEscW_fitsForkR (hfit : SlackFitW n sch Np rd δ) (C c : ℝ) (hk : 0 < V.k x)
    (h : 2 * c + joinCW n sch Np rd ⟨0, hk⟩ + ∑ t, joinCW n sch Np rd t + roomW n sch Np ≤ 2 * C) :
    (slackEscW H V P x n sch Np rd ev ev₁ tied tied₁ δ).FitsForkR C c := by
  refine Escapes.fitsForkR_of_round _ C c ⟨fun a b h => h, fun r => ⟨r, rfl⟩⟩ (⟨0, hk⟩ : Fin (V.k x)) rfl
    (le_trans ?_ h) ?_
  · have hm := mergeCW_le n sch Np rd δ ⟨0, hk⟩
    show 2 * c + mergeCW n sch Np rd δ ⟨0, hk⟩ + ∑ t, mergeCW n sch Np rd δ t ≤ _
    unfold SlackFitW at hfit
    linarith
  · show ∑ _l : Unit, (1 : ℝ) ≤ V.k x
    rw [Fintype.sum_unique]
    exact_mod_cast hk

/-- **The merged escapes are on accepted first runs** where every table's events are on accepted runs: a slack event
is an accepting fork with a collision, or an accepting run with one. -/
theorem slackEscW_onAcc₁
    (hacc : ∀ j i ω s s', ev j i ω s s' →
      (run H P V x ω s).out = .acc ∧ (run H P V x ω (forkCoins (rd j i) s s')).out = .acc)
    (hacc₁ : ∀ j ω s, ev₁ j ω s → (run H P V x ω s).out = .acc) :
    (slackEscW H V P x n sch Np rd ev ev₁ tied tied₁ δ).OnAcc₁ := by
  refine ⟨?_, ?_⟩
  · rintro t ω s s' (⟨j, -, i, rfl, h⟩ | (h | ⟨-, h⟩))
    · exact ⟨(hacc j i ω s s' h).1, Or.inl (hacc j i ω s s' h).2⟩
    · exact ⟨h.1, Or.inl h.2.1⟩
    · exact ⟨h.1, Or.inr h.2⟩
  · rintro - ω s ⟨j, -, h⟩
    exact hacc₁ j ω s h

/-- **Each table's fork terms at the slack are at most the merged escapes' term**, at a table whose fork events are
at distinct rounds, where each fork event's bound `b i` is at most its probability and the slack of the fork at its
round: the slack is at most the slack's events at their weights (`forkSlack_le`), folding into one root per round
(`sum_sqrt_slack_le`), and at each round the table's joined root and slack root are at most the merged one. -/
theorem slackEscW_term [Fintype Coin] [Nonempty Coin] (hδ : ∀ t, 0 ≤ δ t) {j : ℕ} (hj : j < n)
    (hinj : Function.Injective (rd j)) (ω : Ω) (b : IZW (sch j).levels.length → ℝ)
    (hb : ∀ i, b i ≤ (Pr fun s : (Fin (V.k x) → Coin) × (Fin (V.k x) → Coin) => ev j i ω s.1 s.2) +
      forkSlack H V P x δ (rd j i).val ω) :
    (∑ i, Real.sqrt (cZW (sch j) (Np j) i * b i)) + (Pr fun s : Fin (V.k x) → Coin => ev₁ j ω s) ≤
      (slackEscW H V P x n sch Np rd ev ev₁ tied tied₁ δ).term ω := by
  set R : Fin (V.k x) → ℝ := fun t => Pr fun s : (Fin (V.k x) → Coin) × (Fin (V.k x) → Coin) =>
    (∃ j < n, ∃ i, rd j i = t ∧ ev j i ω s.1 s.2) ∨ slackG H V P x t ω s.1 s.2
  set p : IZW (sch j).levels.length → ℝ := fun i =>
    Pr fun s : (Fin (V.k x) → Coin) × (Fin (V.k x) → Coin) => ev j i ω s.1 s.2
  set q : Fin (V.k x) → ℝ := fun t =>
    Pr fun s : (Fin (V.k x) → Coin) × (Fin (V.k x) → Coin) => slackG H V P x t ω s.1 s.2
  have h1 : ∀ i, b i ≤ p i + ∑ t, slackW (V.k x) δ (rd j i).val t.val * q t := fun i =>
    (hb i).trans (add_le_add le_rfl (forkSlack_le H V P x hδ (rd j i) ω))
  have h2 : (∑ i, Real.sqrt (cZW (sch j) (Np j) i * b i)) ≤
      (∑ i, Real.sqrt (cZW (sch j) (Np j) i * p i)) + ∑ t, Real.sqrt (slackCW sch Np rd δ j t * q t) := by
    have h3 := sum_sqrt_slack_le (cZW (sch j) (Np j)) p (fun i t => slackW (V.k x) δ (rd j i).val t.val) q
      (cZW_nonneg _ _) (fun _ => Pr_nonneg _) (fun _ _ => slackW_nonneg hδ _ _ _) fun _ => Pr_nonneg _
    unfold slackCW
    exact (Finset.sum_le_sum fun i _ => Real.sqrt_le_sqrt (mul_le_mul_of_nonneg_left (h1 i)
      (cZW_nonneg _ _ i))).trans h3
  have hA : (∑ i, Real.sqrt (cZW (sch j) (Np j) i * p i)) ≤
      ∑ t, Real.sqrt (joinCW n sch Np rd t) * Real.sqrt (R t) := by
    have hle : ∀ i, Real.sqrt (cZW (sch j) (Np j) i * p i) ≤
        Real.sqrt (joinCW n sch Np rd (rd j i)) * Real.sqrt (R (rd j i)) := fun i => by
      have hpi : p i ≤ R (rd j i) := Pr_mono fun s h => Or.inl ⟨j, hj, i, rfl, h⟩
      rw [← Real.sqrt_mul (joinCW_nonneg n sch Np rd _)]
      exact Real.sqrt_le_sqrt (mul_le_mul (cZW_le_joinCW n sch Np rd hj i) hpi (Pr_nonneg _)
        (joinCW_nonneg n sch Np rd _))
    calc (∑ i, Real.sqrt (cZW (sch j) (Np j) i * p i))
        ≤ ∑ i, Real.sqrt (joinCW n sch Np rd (rd j i)) * Real.sqrt (R (rd j i)) :=
          Finset.sum_le_sum fun i _ => hle i
      _ = ∑ t ∈ Finset.univ.image (rd j), Real.sqrt (joinCW n sch Np rd t) * Real.sqrt (R t) :=
          (Finset.sum_image (f := fun t => Real.sqrt (joinCW n sch Np rd t) * Real.sqrt (R t))
            fun a _ b _ h => hinj h).symm
      _ ≤ ∑ t, Real.sqrt (joinCW n sch Np rd t) * Real.sqrt (R t) :=
          Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ _)
            fun _ _ _ => mul_nonneg (Real.sqrt_nonneg _) (Real.sqrt_nonneg _)
  have hB : (∑ t, Real.sqrt (slackCW sch Np rd δ j t * q t)) ≤
      ∑ t, Real.sqrt (slackTW n sch Np rd δ t) * Real.sqrt (R t) := Finset.sum_le_sum fun t _ => by
    have hqt : q t ≤ R t := Pr_mono fun s h => Or.inr h
    rw [← Real.sqrt_mul (slackTW_nonneg n sch Np rd δ t)]
    exact Real.sqrt_le_sqrt (mul_le_mul (slackCW_le_slackTW n sch Np rd δ hj t) hqt (Pr_nonneg _)
      (slackTW_nonneg n sch Np rd δ t))
  have hC : ∀ t, Real.sqrt (mergeCW n sch Np rd δ t * R t) = Real.sqrt (joinCW n sch Np rd t) * Real.sqrt (R t) +
      Real.sqrt (slackTW n sch Np rd δ t) * Real.sqrt (R t) := fun t => by
    have hm : 0 ≤ Real.sqrt (joinCW n sch Np rd t) + Real.sqrt (slackTW n sch Np rd δ t) :=
      add_nonneg (Real.sqrt_nonneg _) (Real.sqrt_nonneg _)
    rw [mergeCW, Real.sqrt_mul (sq_nonneg _) (R t), Real.sqrt_sq hm, add_mul]
  have hsum : ∑ t, Real.sqrt (mergeCW n sch Np rd δ t * R t) =
      (∑ t, Real.sqrt (joinCW n sch Np rd t) * Real.sqrt (R t)) +
        ∑ t, Real.sqrt (slackTW n sch Np rd δ t) * Real.sqrt (R t) := by
    rw [← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl fun t _ => hC t
  show _ ≤ (∑ t : Fin (V.k x), Real.sqrt (mergeCW n sch Np rd δ t * R t)) +
    ∑ _l : Unit, (1 : ℝ) * (Pr fun s : Fin (V.k x) → Coin => ∃ j < n, ev₁ j ω s)
  refine add_le_add ((h2.trans (add_le_add hA hB)).trans hsum.symm.le) ?_
  rw [Fintype.sum_unique, one_mul]
  exact Pr_mono fun s h => ⟨j, hj, h⟩

end Args

end SlackEsc

end Claims.Flock

namespace FlockDeploy.KS

open Flock FlockSoundness

namespace Gen

variable [S : ZkSetup]

/-- **The masked weight's room for one more event per round**: at a statement whose tables `st j` are C-Flock ZK's,
when each table's pads' weight is at most its deployed pads' domain and only level 0's event of a table is at round
`0`, the fork of weight `k` and the joined escapes, round `0`'s charged twice, leave every table's level caps' and pads
roots' weights (`Claims.Flock.roomW`) of twice the deployed weight: each table's weight pays for its events
(`cZW_sum_le_tableWeight`), round `0` holds at most level 0's, the room and level 0's are every table's but `2`
(`Claims.Flock.roomW_add`), and `2·k < 8·N₀/e + 2`. -/
theorem zk_fork_room (rel : Readable.Parsed → Readable.Relation) (x : Claims.Msg) {spec : SessionSpec}
    (st : ℕ → Stmt) (hst : ∀ j < (tables true x).length, (tables true x)[j]? = some (spec, st j))
    (sch : ℕ → Accounting.Schedule) (hsch : ∀ j < (tables true x).length, sch j = Accounting.m1Schedule (st j).m)
    (Np : ℕ → ℕ) (hNp : ∀ j < (tables true x).length,
      Np j ≤ Claims.Flock.Numbers.padDomain (Zk.PadLayout.new (st j).m (st j).c.kLog
        (Claims.Flock.zkSched (st j).m).1.k0))
    (rd : (j : ℕ) → Claims.Flock.IZW (sch j).levels.length → Fin ((deployZk rel).V.k x))
    (hr0 : ∀ j i, (rd j i).val = 0 → i = .inr ()) (r₀ : Fin ((deployZk rel).V.k x)) (hr₀ : r₀.val = 0) :
    2 * ((deployZk rel).k x : ℝ) + Claims.Flock.joinCW (tables true x).length sch Np rd r₀ +
        ∑ r, Claims.Flock.joinCW (tables true x).length sch Np rd r +
        Claims.Flock.roomW (tables true x).length sch Np ≤
      2 * (deployZk rel).weight x := by
  set n := (tables true x).length with hn
  have hz := Claims.Flock.joinCW_zero_le n sch Np rd hr0 r₀ hr₀
  have hroom := Claims.Flock.roomW_add n sch Np
  rw [Claims.Flock.joinCW_sum]
  have hw : weight true x = ∑ j ∈ Finset.range n, tableWeight true (st j) :=
    sum_map_eq_range (tables true x) (fun j => (spec, st j)) hst _
  have hN : N₀ true x = ∑ j ∈ Finset.range n, tableLen true (st j) :=
    sum_map_eq_range (tables true x) (fun j => (spec, st j)) hst _
  have hk : ((deployZk rel).k x : ℝ) = (⌈4 * N₀ true x / Real.exp 1⌉₊ : ℝ) := rfl
  show 2 * ((deployZk rel).k x : ℝ) + _ + _ + _ ≤ 2 * weight true x
  rcases Nat.eq_zero_or_pos n with h0 | hpos
  · have := k_le_weight true x
    have hj0 : Claims.Flock.joinCW n sch Np rd r₀ = 0 := by
      unfold Claims.Flock.joinCW
      rw [h0, Finset.sum_range_zero]
    have hr0' : Claims.Flock.roomW n sch Np = 0 := by
      unfold Claims.Flock.roomW
      rw [h0, Finset.sum_range_zero]
    rw [hj0, hr0', h0, Finset.sum_range_zero, hk]
    linarith
  have hper : ∀ j ∈ Finset.range n, 4 * tableLen true (st j) / Real.exp 1 +
      ∑ i, Claims.Flock.cZW (sch j) (Np j) i ≤ tableWeight true (st j) := fun j hj => by
    have hj := Finset.mem_range.1 hj
    exact cZW_sum_le_tableWeight (hsch j hj) (Np j) (hNp j hj)
  have hs := Finset.sum_le_sum hper
  rw [Finset.sum_add_distrib, ← Finset.sum_div, ← Finset.mul_sum, ← hN, ← hw] at hs
  have he := Real.exp_pos 1
  have hceil := Nat.ceil_lt_add_one (div_nonneg (mul_nonneg (by norm_num : (0 : ℝ) ≤ 4) (N₀_nonneg true x)) he.le)
  have hn1 : (1 : ℝ) ≤ n := by exact_mod_cast hpos
  rw [hk]
  linarith

end Gen

/-- **The masked weight's room for one more event per round**: at a statement whose tables `st j` are C-Flock ZK's,
when each table's pads' weight is at most its deployed pads' domain and only level 0's event of a table is at round
`0`, the fork of weight `k` and the joined escapes, round `0`'s charged twice, leave every table's level caps' and pads
roots' weights (`Claims.Flock.roomW`) of twice the deployed weight: each table's weight pays for its events
(`cZW_sum_le_tableWeight`), round `0` holds at most level 0's, the room and level 0's are every table's but `2`
(`Claims.Flock.roomW_add`), and `2·k < 8·N₀/e + 2`. -/
theorem zk_fork_room (rel : Readable.Parsed → Readable.Relation) (x : Claims.Msg) {spec : SessionSpec}
    (st : ℕ → Stmt) (hst : ∀ j < (tables true x).length, (tables true x)[j]? = some (spec, st j))
    (sch : ℕ → Accounting.Schedule) (hsch : ∀ j < (tables true x).length, sch j = Accounting.m1Schedule (st j).m)
    (Np : ℕ → ℕ) (hNp : ∀ j < (tables true x).length,
      Np j ≤ Claims.Flock.Numbers.padDomain (Zk.PadLayout.new (st j).m (st j).c.kLog
        (Claims.Flock.zkSched (st j).m).1.k0))
    (rd : (j : ℕ) → Claims.Flock.IZW (sch j).levels.length → Fin ((deployZk rel).V.k x))
    (hr0 : ∀ j i, (rd j i).val = 0 → i = .inr ()) (r₀ : Fin ((deployZk rel).V.k x)) (hr₀ : r₀.val = 0) :
    2 * ((deployZk rel).k x : ℝ) + Claims.Flock.joinCW (tables true x).length sch Np rd r₀ +
        ∑ r, Claims.Flock.joinCW (tables true x).length sch Np rd r +
        Claims.Flock.roomW (tables true x).length sch Np ≤
      2 * (deployZk rel).weight x :=
  Gen.zk_fork_room (S := S₀) (rel := rel) (x := x) (spec := spec) (st := st) (hst := hst) (sch := sch) (hsch := hsch)
    (Np := Np) (hNp := hNp) (rd := rd) (hr0 := hr0) (r₀ := r₀) (hr₀ := hr₀)

namespace Gen

variable [S : ZkSetup]

/-- **The merged escapes fit the deployed masked weight with the fork event of weight `k`, counted per round**, at a
statement whose tables `st j` are C-Flock ZK's, when each table's pads' weight is at most its deployed pads' domain,
only level 0's event of a table is at round `0` and the slack fits the room (`zk_fork_room`,
`Claims.Flock.slackEscW_fitsForkR`). -/
theorem slackEscW_fitsForkR_zk (rel : Readable.Parsed → Readable.Relation) {Ω : Type} (P : Claims.Prover Ω)
    (x : Claims.Msg) {spec : SessionSpec} (st : ℕ → Stmt)
    (hst : ∀ j < (tables true x).length, (tables true x)[j]? = some (spec, st j))
    (sch : ℕ → Accounting.Schedule) (hsch : ∀ j < (tables true x).length, sch j = Accounting.m1Schedule (st j).m)
    (Np : ℕ → ℕ) (hNp : ∀ j < (tables true x).length,
      Np j ≤ Claims.Flock.Numbers.padDomain (Zk.PadLayout.new (st j).m (st j).c.kLog
        (Claims.Flock.zkSched (st j).m).1.k0))
    (rd : (j : ℕ) → Claims.Flock.IZW (sch j).levels.length → Fin ((deployZk rel).V.k x))
    (hr0 : ∀ j i, (rd j i).val = 0 → i = .inr ())
    (ev : (j : ℕ) → Claims.Flock.IZW (sch j).levels.length → Ω → (Fin ((deployZk rel).V.k x) → Coin) →
      (Fin ((deployZk rel).V.k x) → Coin) → Prop)
    (ev₁ : ℕ → Ω → (Fin ((deployZk rel).V.k x) → Coin) → Prop)
    (tied : ∀ j i ω s s', ev j i ω s s' → Claims.TiedColl sha512
      ((Claims.run sha512 P (deployZk rel).V x ω s).vLog ++
        (Claims.run sha512 P (deployZk rel).V x ω (Claims.forkCoins (rd j i) s s')).vLog))
    (tied₁ : ∀ j ω s, ev₁ j ω s → Claims.TiedColl sha512 (Claims.run sha512 P (deployZk rel).V x ω s).vLog)
    (δ : ℕ → ℝ) (hfit : Claims.Flock.SlackFitW (tables true x).length sch Np rd δ) :
    (Claims.Flock.slackEscW sha512 (deployZk rel).V P x (tables true x).length sch Np rd ev ev₁ tied tied₁
      δ).FitsForkR ((deployZk rel).weight x) ((deployZk rel).k x) :=
  Claims.Flock.slackEscW_fitsForkR (tables true x).length sch Np rd ev ev₁ tied tied₁ δ hfit _ _ (V_k_pos true x)
    (zk_fork_room rel x st hst sch hsch Np hNp rd hr0 ⟨0, V_k_pos true x⟩ rfl)

end Gen

/-- **The merged escapes fit the deployed masked weight with the fork event of weight `k`, counted per round**, at a
statement whose tables `st j` are C-Flock ZK's, when each table's pads' weight is at most its deployed pads' domain,
only level 0's event of a table is at round `0` and the slack fits the room (`zk_fork_room`,
`Claims.Flock.slackEscW_fitsForkR`). -/
theorem slackEscW_fitsForkR_zk (rel : Readable.Parsed → Readable.Relation) {Ω : Type} (P : Claims.Prover Ω)
    (x : Claims.Msg) {spec : SessionSpec} (st : ℕ → Stmt)
    (hst : ∀ j < (tables true x).length, (tables true x)[j]? = some (spec, st j))
    (sch : ℕ → Accounting.Schedule) (hsch : ∀ j < (tables true x).length, sch j = Accounting.m1Schedule (st j).m)
    (Np : ℕ → ℕ) (hNp : ∀ j < (tables true x).length,
      Np j ≤ Claims.Flock.Numbers.padDomain (Zk.PadLayout.new (st j).m (st j).c.kLog
        (Claims.Flock.zkSched (st j).m).1.k0))
    (rd : (j : ℕ) → Claims.Flock.IZW (sch j).levels.length → Fin ((deployZk rel).V.k x))
    (hr0 : ∀ j i, (rd j i).val = 0 → i = .inr ())
    (ev : (j : ℕ) → Claims.Flock.IZW (sch j).levels.length → Ω → (Fin ((deployZk rel).V.k x) → Coin) →
      (Fin ((deployZk rel).V.k x) → Coin) → Prop)
    (ev₁ : ℕ → Ω → (Fin ((deployZk rel).V.k x) → Coin) → Prop)
    (tied : ∀ j i ω s s', ev j i ω s s' → Claims.TiedColl sha512
      ((Claims.run sha512 P (deployZk rel).V x ω s).vLog ++
        (Claims.run sha512 P (deployZk rel).V x ω (Claims.forkCoins (rd j i) s s')).vLog))
    (tied₁ : ∀ j ω s, ev₁ j ω s → Claims.TiedColl sha512 (Claims.run sha512 P (deployZk rel).V x ω s).vLog)
    (δ : ℕ → ℝ) (hfit : Claims.Flock.SlackFitW (tables true x).length sch Np rd δ) :
    (Claims.Flock.slackEscW sha512 (deployZk rel).V P x (tables true x).length sch Np rd ev ev₁ tied tied₁
      δ).FitsForkR ((deployZk rel).weight x) ((deployZk rel).k x) :=
  Gen.slackEscW_fitsForkR_zk (S := S₀) (rel := rel) (Ω := Ω) (P := P) (x := x) (spec := spec) (st := st) (hst := hst)
    (sch := sch) (hsch := hsch) (Np := Np) (hNp := hNp) (rd := rd) (hr0 := hr0) (ev := ev) (ev₁ := ev₁) (tied := tied)
    (tied₁ := tied₁) (δ := δ) (hfit := hfit)

end FlockDeploy.KS
