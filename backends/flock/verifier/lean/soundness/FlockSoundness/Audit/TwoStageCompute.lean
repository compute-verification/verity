import FlockSoundness.Audit.Harm
import FlockSoundness.Audit.Examples
import FlockSoundness.Audit.Flock

/-!
# Free compute of the two-stage audit, variant (b)

A consumer weights each coarse (replay) unit `u` by its work `w u`, e.g. the VU count `n_v u` of
the replay unit. The free compute of a session is the work of its wrong replay units,
`∑_{u ∈ wrong_c} w u`. `sampled_proofs/PROTOCOL.md` ("Consumers") charges a wrong replay unit its
class's `n_v` VUs, and the escape of a set of wrong replay units is at most `∏ (1 − p·k/n_v)`.

**Why the dilution `k/n_v`.** The interior of a drawn replay unit is committed after the coarse
draw, so the prover knows which wrong replay units were drawn before it fixes their fine units. A
wrong replay unit needs only one wrong fine unit (`Analysis₂.compose`), and the prover's best is
exactly one (`m = 1`): any further wrong fine unit can only raise the chance that the fine draw hits
one, so it concentrates the lie into a single fine unit. A uniform `k`-of-`n_v` fine draw then hits
it with probability `k/n_v`, not `1`, and a Bernoulli(`p`) coarse draw selects the unit with
probability `p`: the unit escapes with `1 − p·k/n_v`, independently across units. `effEscape`
already takes the worst fine unit (`⨆ v`), so `m = 1` is the adversary's optimum built into the
bound, which `NotChains.two_stage_b_tight` shows is attained. The adversary is unbounded; its only
other chances are the named per-unit terms `ε_ks` and `δ_link`.

* `IsHarmBound₂`: `IsHarmBound` with the two-stage escape `effEscape` in place of a law's escape;
* `twoStage_compute`: `Pr[accept ∧ ∑_{u ∈ wrong_c} w u > W] ≤ δ + ε_ks + δ_link`;
* `effEscape_bernoulli_subset_le`: the closed form, `effEscape B ≤ ∏_{u ∈ B} (1 − p·k_u/n_u)` for a
  Bernoulli(`p = num/den`) first stage and a uniform `k_u`-of-`n_u` second stage in each unit;
* `twoStage_wrong_set_le`, `twoStage_compute_closed`: the profile and the free-compute bound in that
  closed form;
* `flock_two_stage_compute`, `flock_two_stage_compute_closed`: on Flock, with `ε_ks + δ_link =
  2^-205` (`flock_two_stage_sound`).
-/

namespace FlockSoundness.Audit

open Game
open scoped ENNReal

variable {nc nf : ℕ}

/-- **`harm_bound`'s specification for the two-stage escape**: every coarse wrong set that escapes
both stages with probability at least `δ` has work at most `W`. -/
def IsHarmBound₂ (L₁ : Law nc) (L₂ : Fin nc → Law nf) (parent : Fin nf → Fin nc) (w : Fin nc → ℝ)
    (δ : ℝ≥0∞) (W : ℝ) : Prop :=
  ∀ B : Finset (Fin nc), δ ≤ effEscape L₁ L₂ parent B → ∑ u ∈ B, w u ≤ W

theorem effEscape_lt_of_harm_gt {L₁ : Law nc} {L₂ : Fin nc → Law nf} {parent : Fin nf → Fin nc}
    {w : Fin nc → ℝ} {δ : ℝ≥0∞} {W : ℝ} (hW : IsHarmBound₂ L₁ L₂ parent w δ W)
    {B : Finset (Fin nc)} (hB : W < ∑ u ∈ B, w u) : effEscape L₁ L₂ parent B < δ :=
  lt_of_not_ge fun h => absurd (hW B h) (not_le.2 hB)

variable {L₁ : Law nc} {L₂ : Fin nc → Law nf} {Reg : Type} {Int : Finset (Fin nc) → Type}
  {session : Finset (Fin nf) → Reg → (S₁ : Finset (Fin nc)) → Int S₁ → Game Bool}

/-- **Free compute of the two-stage audit.** If `W` is a harm bound at `δ` for the work `w` under
the effective escape, the session accepts while its wrong replay units carry more than `W` work with
probability at most `δ + ε_ks + δ_link`. -/
theorem twoStage_compute (A : Analysis₂ L₁ L₂ Reg Int session) {εks δlink : ℝ≥0∞}
    (hks : A.KnowledgeSound εks) (hlink : A.LinkSound δlink) (w : Fin nc → ℝ) {δ : ℝ≥0∞} {W : ℝ}
    (hW : IsHarmBound₂ L₁ L₂ A.parent w δ W) (σ : Strategy (twoStage L₁ L₂ Reg Int session)) :
    prob (fun o => o.1 = true ∧ W < ∑ u ∈ A.wrongC (A.committedCOf σ), w u)
      (twoStage L₁ L₂ Reg Int session) σ ≤ δ + εks + δlink := by
  refine (twoStage_profile A hks hlink (fun B => W < ∑ u ∈ B, w u) σ).trans ?_
  gcongr
  exact iSup₂_le fun B hB => (effEscape_lt_of_harm_gt hW hB).le

/-! ## The closed form: Bernoulli first stage, uniform subsets inside -/

/-- The second stage of a replay unit: a uniform `k`-subset of its `n` fine units, placed by `e`. -/
def subsetIn {n k : ℕ} (hk : k ≤ n) (e : Fin n ↪ Fin nf) : Law nf :=
  (Law.subset n k hk).map e

theorem incl_subsetIn {n k : ℕ} (hk : k ≤ n) (e : Fin n ↪ Fin nf) (i : Fin n) :
    (subsetIn hk e).incl (e i) = (Law.subset n k hk).incl i :=
  congrArg prCoin (funext fun _ => propext (Finset.mem_map' e))

/-- A uniform `k`-of-`n` subset misses a given unit with probability `(n − k)/n`. -/
theorem subset_one_sub_incl {n k : ℕ} (hk : k ≤ n) (i : Fin n) :
    1 - (Law.subset n k hk).incl i = ((n - k : ℕ) : ℝ≥0∞) / (n : ℝ≥0∞) := by
  rw [← Law.escape_singleton, Law.subset_escape, Finset.card_singleton,
    Law.choose_ratio hk (Nat.one_le_iff_ne_zero.2 (Fin.pos i).ne')]

/-- One Bernoulli(`num/den`) factor of a unit escaping the fine draw with `(n − k)/n`:
`(den·n − num·k)/(den·n) = 1 − (num/den)·(k/n)`. -/
theorem bernFactor_subset {num den n k : ℕ} [NeZero den] (hnum : num ≤ den) (hk : k ≤ n)
    (hn : 1 ≤ n) :
    Law.bernFactor num den (((n - k : ℕ) : ℝ≥0∞) / (n : ℝ≥0∞)) =
      ((den * n - num * k : ℕ) : ℝ≥0∞) / ((den * n : ℕ) : ℝ≥0∞) := by
  unfold Law.bernFactor
  have hn0 : (n : ℝ≥0∞) ≠ 0 := by exact_mod_cast (show n ≠ 0 by omega)
  have hd0 : (den : ℝ≥0∞) ≠ 0 := by exact_mod_cast (NeZero.ne den)
  have hnt : (n : ℝ≥0∞) ≠ ∞ := ENNReal.natCast_ne_top _
  have key : (num : ℝ≥0∞) * (((n - k : ℕ) : ℝ≥0∞) / n) + ((den - num : ℕ) : ℝ≥0∞) =
      ((den * n - num * k : ℕ) : ℝ≥0∞) / n := by
    rw [ENNReal.eq_div_iff hn0 hnt, mul_add, mul_comm (n : ℝ≥0∞) (_ * _), mul_assoc,
      ENNReal.div_mul_cancel hn0 hnt]
    have h1 : num * (n - k) + n * (den - num) = den * n - num * k := by
      have : num * k ≤ den * n := Nat.mul_le_mul hnum hk
      zify [hk, hnum, this]
      ring
    exact_mod_cast h1
  rw [key]
  push_cast
  rw [div_eq_mul_inv, div_eq_mul_inv, div_eq_mul_inv, mul_assoc,
    ENNReal.mul_inv (Or.inl hd0) (Or.inr hn0), mul_comm (den : ℝ≥0∞)⁻¹]

/-- **The closed form of the effective escape.** First stage Bernoulli(`p = num/den`); inside each
replay unit `u`, a uniform `k u`-subset of its `n u` fine units, placed by `e u` onto every fine
unit of `u`. Then `effEscape B ≤ ∏_{u ∈ B} (den·n_u − num·k_u)/(den·n_u) = ∏ (1 − p·k_u/n_u)`. -/
theorem effEscape_bernoulli_subset_le {num den : ℕ} [NeZero den] (hnum : num ≤ den)
    (parent : Fin nf → Fin nc) (n k : Fin nc → ℕ) (hk : ∀ u, k u ≤ n u) (hn : ∀ u, 1 ≤ n u)
    (e : ∀ u, Fin (n u) ↪ Fin nf) (hcov : ∀ u v, parent v = u → ∃ i, e u i = v)
    (B : Finset (Fin nc)) :
    effEscape (Law.bernoulli nc num den) (fun u => subsetIn (hk u) (e u)) parent B ≤
      ∏ u ∈ B, ((den * n u - num * k u : ℕ) : ℝ≥0∞) / ((den * n u : ℕ) : ℝ≥0∞) := by
  rw [effEscape_bernoulli_prod hnum]
  refine Finset.prod_le_prod fun u _ => ?_
  rw [← bernFactor_subset hnum (hk u) (hn u)]
  unfold Law.bernFactor
  gcongr
  refine iSup₂_le fun v hv => ?_
  obtain ⟨i, rfl⟩ := hcov u v hv
  rw [incl_subsetIn, subset_one_sub_incl]

/-- The closed-form per-unit escape `(den·n_u − num·k_u)/(den·n_u) = 1 − p·k_u/n_u`. -/
noncomputable def subsetFactor (num den : ℕ) (n k : Fin nc → ℕ) (u : Fin nc) : ℝ≥0∞ :=
  ((den * n u - num * k u : ℕ) : ℝ≥0∞) / ((den * n u : ℕ) : ℝ≥0∞)

theorem subsetFactor_le_one (num den : ℕ) (n k : Fin nc → ℕ) (u : Fin nc) :
    subsetFactor num den n k u ≤ 1 := by
  unfold subsetFactor
  rcases Nat.eq_zero_or_pos (den * n u) with h | h
  · rw [h]; simp
  · refine ENNReal.div_le_of_le_mul ?_
    rw [one_mul]
    exact_mod_cast Nat.sub_le _ _

/-- **The profile in closed form**: the session accepts while every replay unit of `B` is wrong
with probability at most `∏_{u ∈ B} (1 − p·k_u/n_u) + ε_ks + δ_link`. -/
theorem twoStage_wrong_set_le {num den : ℕ} [NeZero den] (hnum : num ≤ den) (n k : Fin nc → ℕ)
    (hk : ∀ u, k u ≤ n u) (hn : ∀ u, 1 ≤ n u) (e : ∀ u, Fin (n u) ↪ Fin nf)
    {Reg : Type} {Int : Finset (Fin nc) → Type}
    {session : Finset (Fin nf) → Reg → (S₁ : Finset (Fin nc)) → Int S₁ → Game Bool}
    (A : Analysis₂ (Law.bernoulli nc num den) (fun u => subsetIn (hk u) (e u)) Reg Int session)
    (hcov : ∀ u v, A.parent v = u → ∃ i, e u i = v) {εks δlink : ℝ≥0∞}
    (hks : A.KnowledgeSound εks) (hlink : A.LinkSound δlink) (B : Finset (Fin nc))
    (σ : Strategy (twoStage (Law.bernoulli nc num den) (fun u => subsetIn (hk u) (e u)) Reg Int
      session)) :
    prob (fun o => o.1 = true ∧ B ⊆ A.wrongC (A.committedCOf σ)) _ σ ≤
      (∏ u ∈ B, subsetFactor num den n k u) + εks + δlink := by
  refine (twoStage_profile A hks hlink (B ⊆ ·) σ).trans ?_
  gcongr
  refine iSup₂_le fun B' hB' => (effEscape_bernoulli_subset_le hnum A.parent n k hk hn e hcov B').trans ?_
  exact Finset.prod_le_prod_of_subset_of_le_one hB' fun u _ _ => subsetFactor_le_one num den n k u

/-- **`harm_bound`'s specification in closed form**: every set whose product
`∏ (1 − p·k_u/n_u)` is at least `δ` has work at most `W`. -/
def IsHarmBoundClosed (num den : ℕ) (n k : Fin nc → ℕ) (w : Fin nc → ℝ) (δ : ℝ≥0∞) (W : ℝ) : Prop :=
  ∀ B : Finset (Fin nc), δ ≤ ∏ u ∈ B, subsetFactor num den n k u → ∑ u ∈ B, w u ≤ W

theorem IsHarmBoundClosed.isHarmBound₂ {num den : ℕ} [NeZero den] (hnum : num ≤ den)
    {parent : Fin nf → Fin nc} {n k : Fin nc → ℕ} (hk : ∀ u, k u ≤ n u) (hn : ∀ u, 1 ≤ n u)
    {e : ∀ u, Fin (n u) ↪ Fin nf} (hcov : ∀ u v, parent v = u → ∃ i, e u i = v)
    {w : Fin nc → ℝ} {δ : ℝ≥0∞} {W : ℝ} (hW : IsHarmBoundClosed num den n k w δ W) :
    IsHarmBound₂ (Law.bernoulli nc num den) (fun u => subsetIn (hk u) (e u)) parent w δ W :=
  fun B hB => hW B (hB.trans (effEscape_bernoulli_subset_le hnum parent n k hk hn e hcov B))

/-- **Free compute in closed form**: Bernoulli(`p`) replay units, uniform `k_u`-of-`n_u` fine units
inside each; `W` a harm bound at `δ` for `∏ (1 − p·k_u/n_u)`. -/
theorem twoStage_compute_closed {num den : ℕ} [NeZero den] (hnum : num ≤ den) (n k : Fin nc → ℕ)
    (hk : ∀ u, k u ≤ n u) (hn : ∀ u, 1 ≤ n u) (e : ∀ u, Fin (n u) ↪ Fin nf)
    {Reg : Type} {Int : Finset (Fin nc) → Type}
    {session : Finset (Fin nf) → Reg → (S₁ : Finset (Fin nc)) → Int S₁ → Game Bool}
    (A : Analysis₂ (Law.bernoulli nc num den) (fun u => subsetIn (hk u) (e u)) Reg Int session)
    (hcov : ∀ u v, A.parent v = u → ∃ i, e u i = v) {εks δlink : ℝ≥0∞}
    (hks : A.KnowledgeSound εks) (hlink : A.LinkSound δlink) (w : Fin nc → ℝ) {δ : ℝ≥0∞} {W : ℝ}
    (hW : IsHarmBoundClosed num den n k w δ W)
    (σ : Strategy (twoStage (Law.bernoulli nc num den) (fun u => subsetIn (hk u) (e u)) Reg Int
      session)) :
    prob (fun o => o.1 = true ∧ W < ∑ u ∈ A.wrongC (A.committedCOf σ), w u) _ σ ≤
      δ + εks + δlink :=
  twoStage_compute A hks hlink w (hW.isHarmBound₂ hnum hk hn hcov) σ

/-! ## At the oracle layer, and on Flock -/

namespace Partition

variable {C : Circuit} {Pf : Partition C nf} {Pc : Partition C nc}

/-- **Free compute at the oracle layer**: `Pr[accept ∧ ∑_{u ∈ wrong_c(R)} w u > W] ≤ δ + ε`. -/
theorem Refines.oracle₂_compute (r : Refines Pf Pc) {L₁ : Law nc} {L₂ : Fin nc → Law nf}
    {Int : Finset (Fin nc) → Type} {ext : ∀ S₁, Int S₁ → (Fin C.N → Bool)}
    {session : Finset (Fin nf) → (Fin C.N → Bool) → (S₁ : Finset (Fin nc)) → Int S₁ → Game Bool}
    {ε : ℝ≥0∞} (hs : SessionSound₂ Pf Pc ext session ε) (w : Fin nc → ℝ) {δ : ℝ≥0∞} {W : ℝ}
    (hW : IsHarmBound₂ L₁ L₂ r.parent w δ W)
    (σ : Strategy (twoStage L₁ L₂ (Fin C.N → Bool) Int session)) :
    prob (fun o => o.1 = true ∧ W < ∑ u ∈ Pc.wrong (reg₂ σ), w u) (twoStage L₁ L₂ _ Int session) σ ≤
      δ + ε := by
  refine (r.oracle₂_profile hs (fun B => W < ∑ u ∈ B, w u) σ).trans ?_
  gcongr
  exact iSup₂_le fun B hB => (effEscape_lt_of_harm_gt hW hB).le

/-- **Free compute of the two-stage audit on Flock**: with the fine partition's lowering, the
session accepts while its wrong replay units carry more than `W` work with probability at most
`δ + 2^-205`. -/
theorem flock_two_stage_compute (r : Refines Pf Pc)
    (plan : Finset (Fin nf) → (Fin C.N → Bool) → List FlockTable) (hlow : Pf.LoweringSound plan)
    {L₁ : Law nc} {L₂ : Fin nc → Law nf}
    {Int : Finset (Fin nc) → Type} (ext : ∀ S₁, Int S₁ → (Fin C.N → Bool))
    (w : Fin nc → ℝ) {δ : ℝ≥0∞} {W : ℝ} (hW : IsHarmBound₂ L₁ L₂ r.parent w δ W)
    (σ : Strategy (twoStage L₁ L₂ (Fin C.N → Bool) Int
      (fun S R S₁ I => flockSession plan S (extend (Pc := Pc) R (ext S₁ I))))) :
    prob (fun o => o.1 = true ∧ W < ∑ u ∈ Pc.wrong (reg₂ σ), w u)
        (twoStage L₁ L₂ _ Int fun S R S₁ I => flockSession plan S (extend (Pc := Pc) R (ext S₁ I))) σ ≤
      δ + ENNReal.ofReal ((2 : ℝ) ^ (-(205 : ℝ))) :=
  r.oracle₂_compute (flock_two_stage_sound plan hlow ext) w hW σ

/-- **Free compute on Flock in closed form**: Bernoulli(`p = num/den`) replay units, a uniform
`k_u`-of-`n_u` subset of each drawn unit's fine units, and `W` a harm bound at `δ` for
`∏ (1 − p·k_u/n_u)`: `Pr[accept ∧ ∑_{u ∈ wrong_c} w u > W] ≤ δ + 2^-205`. -/
theorem flock_two_stage_compute_closed (r : Refines Pf Pc)
    (plan : Finset (Fin nf) → (Fin C.N → Bool) → List FlockTable) (hlow : Pf.LoweringSound plan)
    {num den : ℕ} [NeZero den] (hnum : num ≤ den) (n k : Fin nc → ℕ) (hk : ∀ u, k u ≤ n u)
    (hn : ∀ u, 1 ≤ n u) (e : ∀ u, Fin (n u) ↪ Fin nf) (hcov : ∀ u v, r.parent v = u → ∃ i, e u i = v)
    {Int : Finset (Fin nc) → Type} (ext : ∀ S₁, Int S₁ → (Fin C.N → Bool))
    (w : Fin nc → ℝ) {δ : ℝ≥0∞} {W : ℝ} (hW : IsHarmBoundClosed num den n k w δ W)
    (σ : Strategy (twoStage (Law.bernoulli nc num den) (fun u => subsetIn (hk u) (e u))
      (Fin C.N → Bool) Int (fun S R S₁ I => flockSession plan S (extend (Pc := Pc) R (ext S₁ I))))) :
    prob (fun o => o.1 = true ∧ W < ∑ u ∈ Pc.wrong (reg₂ σ), w u)
        (twoStage (Law.bernoulli nc num den) (fun u => subsetIn (hk u) (e u)) _ Int
          fun S R S₁ I => flockSession plan S (extend (Pc := Pc) R (ext S₁ I))) σ ≤
      δ + ENNReal.ofReal ((2 : ℝ) ^ (-(205 : ℝ))) :=
  flock_two_stage_compute r plan hlow ext w (hW.isHarmBound₂ hnum hk hn hcov) σ

end Partition

end FlockSoundness.Audit
