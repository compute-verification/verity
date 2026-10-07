import Proofs.Flock.Soundness.Audit.Influence
import Proofs.Flock.Soundness.Audit.Flock

/-!
# Exfiltration through an accepted two-stage audit

The two-stage analogue of `audit_exfiltration`, for variant (b) (`twoStage`): the verifier draws coarse replay units by
`L₁`, the prover then commits the drawn units' interiors, and the verifier draws fine proof units by `L₂ u` inside each
drawn coarse unit, fresh. Since the interior is chosen after the coarse draw, a wrong coarse unit is caught only when
the fine draw inside it hits a wrong fine unit the prover placed knowing the coarse draw; so the coarse set `B` escapes
with the effective escape `effEscape L₁ L₂ parent B`, not `L₁.escape B`, and every bound here is stated with it.

* `IsHarmBoundOn e h δ H`: `IsHarmBound` for an arbitrary escape function `e` (`IsHarmBound L = IsHarmBoundOn L.escape`,
  `isHarmBound_iff`).
* `effMiss L₁ L₂ parent K`: the largest effective escape over coarse sets of at least `K` units (as `Law.miss`).
* `card_influenceSet_le_harmOn`, `card_admissible_le_of`: `card_influenceSet_le_harm` and `card_admissible_le` for any
  escape function.
* `Refines.twoStage_exfiltration_influenceSet`, `Refines.twoStage_exfiltration`: a prover that picks its strategy from a
  uniform message `m`, and a receiver that sees only what the committed coarse transcript delivers on `D`, recover `m`
  from an accepted two-stage audit with probability at most `2^H · ∑_{j ≤ k} C(nc, j) / |M| + δ + ε_ks + δ_link + δ_in`.
* `flock_two_stage_exfiltration`: the same on Flock's session at the oracle layer, with `ε_ks = 2^-205` and
  `δ_link = 0`.

The prover is unbounded: the bounds hold for every strategy of the game. Computational content enters only through
`ε_ks` and `δ_link` (and `δ_in`), as named hypotheses or as discharged by Flock's session soundness.
-/

namespace FlockSoundness.Audit

open Game Finset
open scoped ENNReal

variable {n : ℕ}

/-- **`harm_bound`'s specification for an escape function `e`**: every wrong-unit set that escapes with probability at
least `δ` under `e` has harm at most `H`. -/
def IsHarmBoundOn (e : Finset (Fin n) → ℝ≥0∞) (h : Fin n → ℝ) (δ : ℝ≥0∞) (H : ℝ) : Prop :=
  ∀ B : Finset (Fin n), δ ≤ e B → ∑ u ∈ B, h u ≤ H

theorem isHarmBound_iff {L : Law n} {h : Fin n → ℝ} {δ : ℝ≥0∞} {H : ℝ} :
    IsHarmBound L h δ H ↔ IsHarmBoundOn L.escape h δ H := Iff.rfl

variable {nf nc : ℕ}

/-- The largest effective escape over coarse sets of at least `K` units. -/
noncomputable def effMiss (L₁ : Law nc) (L₂ : Fin nc → Law nf) (parent : Fin nf → Fin nc) (K : ℕ) : ℝ≥0∞ :=
  ⨆ (B : Finset (Fin nc)) (_ : K ≤ B.card), effEscape L₁ L₂ parent B

theorem effEscape_le_effMiss (L₁ : Law nc) (L₂ : Fin nc → Law nf) (parent : Fin nf → Fin nc) {K : ℕ}
    {B : Finset (Fin nc)} (h : K ≤ B.card) : effEscape L₁ L₂ parent B ≤ effMiss L₁ L₂ parent K :=
  le_iSup₂_of_le (f := fun B (_ : K ≤ B.card) => effEscape L₁ L₂ parent B) B h le_rfl

namespace Partition

variable {C : Circuit} (P : Partition C n)

/-- `card_influenceSet_le_harm` for any escape function `e`. -/
theorem card_influenceSet_le_harmOn {D : Finset (Fin C.N)} {a : Fin C.N → Bool} (e : Finset (Fin n) → ℝ≥0∞)
    (hD : D ⊆ P.committed) (hb : Fin n → ℕ) (K : Finset (Fin n) → Finset (Fin C.N))
    (hK : ∀ B, K B ⊆ P.committed) (hsep : ∀ B, P.Separates (K B) B D) (hw : ∀ B, (K B).card ≤ ∑ u ∈ B, hb u)
    {δ : ℝ≥0∞} {H : ℕ} (hH : IsHarmBoundOn e (fun u => (hb u : ℝ)) δ H) :
    (P.influenceSet D e a δ).card ≤ 2 ^ H * (univ.filter fun B => δ ≤ e B).card := by
  refine (P.card_influenceSet_le hD e a δ K hK hsep).trans ?_
  calc ∑ B ∈ univ.filter (fun B => δ ≤ e B), 2 ^ (K B).card
      ≤ ∑ B ∈ univ.filter (fun B => δ ≤ e B), 2 ^ H := by
        refine sum_le_sum fun B hB => pow_le_pow_right₀ (by norm_num) ((hw B).trans ?_)
        have h1 := hH B (mem_filter.1 hB).2
        have h2 : ((∑ u ∈ B, hb u : ℕ) : ℝ) ≤ (H : ℝ) := by push_cast; exact h1
        exact_mod_cast h2
    _ = 2 ^ H * (univ.filter fun B => δ ≤ e B).card := by rw [sum_const, smul_eq_mul, mul_comm]

end Partition

/-- `card_admissible_le` for any escape function `e` that is below `δ` on every set of more than `k` units. -/
theorem card_admissible_le_of (e : Finset (Fin n) → ℝ≥0∞) {δ : ℝ≥0∞} {k : ℕ}
    (hk : ∀ B : Finset (Fin n), k + 1 ≤ B.card → e B < δ) :
    (univ.filter fun B : Finset (Fin n) => δ ≤ e B).card ≤ ∑ j ∈ range (k + 1), n.choose j := by
  classical
  have hsub : (univ.filter fun B : Finset (Fin n) => δ ≤ e B) ⊆
      (range (k + 1)).biUnion fun j => (univ : Finset (Fin n)).powersetCard j := by
    intro B hB
    have hle : B.card ≤ k := by
      by_contra hc
      exact absurd (mem_filter.1 hB).2 (not_le.2 (hk B (by omega)))
    exact mem_biUnion.2 ⟨B.card, mem_range.2 (by omega), mem_powersetCard.2 ⟨subset_univ _, rfl⟩⟩
  refine (card_le_card hsub).trans (card_biUnion_le.trans (le_of_eq (sum_congr rfl fun j _ => ?_)))
  rw [card_powersetCard, card_univ, Fintype.card_fin]

/-- **The counting step of exfiltration**, for any game whose outcome starts with the verdict: if, for every message,
accepting while the delivered value `y m` lies outside `Yd` has probability at most `ε`, then a receiver `g` of `y m`
recovers a uniform message from an accepted run with probability at most `|Yd|/|M| + ε`. -/
theorem exfil_of_influence {β O M : Type} [Fintype M] [Nonempty M] [DecidableEq M] [DecidableEq O] {G : Game (Bool × β)}
    (Yd : Finset O) (y : M → O) (g : O → M) (σ : M → Strategy G) {ε : ℝ≥0∞}
    (hY : ∀ m, prob (fun o => o.1 = true ∧ y m ∉ Yd) G (σ m) ≤ ε) :
    avg (fun m => prob (fun o => o.1 = true ∧ g (y m) = m) G (σ m)) ≤
      (Yd.card : ℝ≥0∞) / (Fintype.card M : ℝ≥0∞) + ε := by
  classical
  have step : ∀ m, prob (fun o => o.1 = true ∧ g (y m) = m) G (σ m) ≤
      (if y m ∈ Yd ∧ g (y m) = m then 1 else 0) + prob (fun o => o.1 = true ∧ y m ∉ Yd) G (σ m) := by
    intro m
    refine (prob_mono (q := fun o => (o.1 = true ∧ (y m ∈ Yd ∧ g (y m) = m)) ∨ (o.1 = true ∧ y m ∉ Yd))
      (fun o h => ?_) _ _).trans ((prob_or_le _ _ _ _).trans (add_le_add ?_ le_rfl))
    · by_cases hy : y m ∈ Yd
      · exact Or.inl ⟨h.1, hy, h.2⟩
      · exact Or.inr ⟨h.1, hy⟩
    · rw [prob_and_const]
      split_ifs
      · exact prob_le_one _ _ _
      · exact le_rfl
  calc avg (fun m => prob (fun o => o.1 = true ∧ g (y m) = m) G (σ m))
      ≤ avg (fun m => (if y m ∈ Yd ∧ g (y m) = m then (1 : ℝ≥0∞) else 0) +
          prob (fun o => o.1 = true ∧ y m ∉ Yd) G (σ m)) := avg_mono step
    _ = prCoin (fun m => y m ∈ Yd ∧ g (y m) = m) +
          avg (fun m => prob (fun o => o.1 = true ∧ y m ∉ Yd) G (σ m)) := by
        rw [avg_add, avg_ite_eq_prCoin]
    _ ≤ (Yd.card : ℝ≥0∞) / (Fintype.card M : ℝ≥0∞) + ε := by
        refine add_le_add ?_ ((avg_mono hY).trans (avg_const _).le)
        rw [prCoin_eq_card]
        gcongr
        exact_mod_cast card_guess_le Yd y g

namespace Partition

variable {C : Circuit} {Pf : Partition C nf} {Pc : Partition C nc} {a : Fin C.N → Bool} {εks δlink δin : ℝ≥0∞}
  {D : Finset (Fin C.N)}

/-- The influence-set form of `Refines.twoStage_exfiltration`: a prover that picks its two-stage strategy `σ m` from a
uniform message `m`, and a receiver `g` that sees only what the committed coarse transcript delivers on `D`, succeed
together (the audit accepts and `g` returns `m`) with probability at most
`|influenceSet (effEscape)|/|M| + δ + ε_ks + δ_link + δ_in`. The interior is committed after the coarse draw, which is
why the influence set is taken at `effEscape L₁ L₂ r.parent`. -/
theorem Refines.twoStage_exfiltration_influenceSet (r : Refines Pf Pc) {L₁ : Law nc} {L₂ : Fin nc → Law nf}
    {Reg : Type} {Int : Finset (Fin nc) → Type}
    {session : Finset (Fin nf) → Reg → (S₁ : Finset (Fin nc)) → Int S₁ → Game Bool}
    (Xc : (R : Reg) → Cont₁ L₁ L₂ Reg Int session R → (Fin C.N → Bool))
    (Yint : (R : Reg) → (ω₁ : L₁.Ω) → (I : Int (L₁.draw ω₁)) → Cont₂ L₁ L₂ Reg Int session R ω₁ I →
      (Fin C.N → Bool))
    (ext : (S : Finset (Fin nf)) → (R : Reg) → (S₁ : Finset (Fin nc)) → (I : Int S₁) →
      Strategy (session S R S₁ I) → Fin nf → Option (Fin C.N → Bool))
    (hks : (r.analysis₂ Xc Yint ext).KnowledgeSound εks) (hlink : (r.analysis₂ Xc Yint ext).LinkSound δlink)
    (hin : AnchorsSound₂ Xc a δin) (δ : ℝ≥0∞) {M : Type} [Fintype M] [Nonempty M] [DecidableEq M]
    (σ : M → Strategy (twoStage L₁ L₂ Reg Int session)) (g : (D → Bool) → M) :
    avg (fun m => prob (fun o => o.1 = true ∧ g (out D (Xc (reg₂ (σ m)) (cont₁ (σ m)))) = m)
        (twoStage L₁ L₂ Reg Int session) (σ m)) ≤
      ((Pc.influenceSet D (effEscape L₁ L₂ r.parent) a δ).card : ℝ≥0∞) / (Fintype.card M : ℝ≥0∞) +
        (δ + εks + δlink + δin) := by
  classical
  exact exfil_of_influence _ (fun m => out D (Xc (reg₂ (σ m)) (cont₁ (σ m)))) g σ
    fun m => r.twoStage_influence Xc Yint ext hks hlink hin δ (σ m)

/-- **Exfiltration through an accepted two-stage audit, with the location term** (variant (b)): a prover that picks its
strategy `σ m` from a uniform message `m`, and a receiver `g` that sees only what the committed coarse transcript
delivers on `D`, succeed together with probability at most
`2^H · ∑_{j ≤ k} C(nc, j) / |M| + δ + ε_ks + δ_link + δ_in`. The verifier draws coarse replay units by `L₁`, and the
prover commits their interiors only after that draw, before the fresh fine draw `L₂ u`; so a coarse set escapes with
`effEscape L₁ L₂ r.parent`, and that is the escape of both the harm bound `H` (each replay unit's width `hb` on a
separator, for example its exits, `card_exits_le`) and the location condition `effMiss (k + 1) < δ`. The prover is
unbounded; computational content enters only through `ε_ks`, `δ_link` and `δ_in`. -/
theorem Refines.twoStage_exfiltration (r : Refines Pf Pc) {L₁ : Law nc} {L₂ : Fin nc → Law nf}
    {Reg : Type} {Int : Finset (Fin nc) → Type}
    {session : Finset (Fin nf) → Reg → (S₁ : Finset (Fin nc)) → Int S₁ → Game Bool}
    (Xc : (R : Reg) → Cont₁ L₁ L₂ Reg Int session R → (Fin C.N → Bool))
    (Yint : (R : Reg) → (ω₁ : L₁.Ω) → (I : Int (L₁.draw ω₁)) → Cont₂ L₁ L₂ Reg Int session R ω₁ I →
      (Fin C.N → Bool))
    (ext : (S : Finset (Fin nf)) → (R : Reg) → (S₁ : Finset (Fin nc)) → (I : Int S₁) →
      Strategy (session S R S₁ I) → Fin nf → Option (Fin C.N → Bool))
    (hks : (r.analysis₂ Xc Yint ext).KnowledgeSound εks) (hlink : (r.analysis₂ Xc Yint ext).LinkSound δlink)
    (hin : AnchorsSound₂ Xc a δin) (hD : D ⊆ Pc.committed) (hb : Fin nc → ℕ)
    (K : Finset (Fin nc) → Finset (Fin C.N)) (hK : ∀ B, K B ⊆ Pc.committed) (hsep : ∀ B, Pc.Separates (K B) B D)
    (hw : ∀ B, (K B).card ≤ ∑ u ∈ B, hb u) {δ : ℝ≥0∞} {H k : ℕ}
    (hH : IsHarmBoundOn (effEscape L₁ L₂ r.parent) (fun u => (hb u : ℝ)) δ H)
    (hk : effMiss L₁ L₂ r.parent (k + 1) < δ) {M : Type} [Fintype M] [Nonempty M] [DecidableEq M]
    (σ : M → Strategy (twoStage L₁ L₂ Reg Int session)) (g : (D → Bool) → M) :
    avg (fun m => prob (fun o => o.1 = true ∧ g (out D (Xc (reg₂ (σ m)) (cont₁ (σ m)))) = m)
        (twoStage L₁ L₂ Reg Int session) (σ m)) ≤
      ((2 ^ H * ∑ j ∈ range (k + 1), nc.choose j : ℕ) : ℝ≥0∞) / (Fintype.card M : ℝ≥0∞) +
        (δ + εks + δlink + δin) := by
  refine (r.twoStage_exfiltration_influenceSet Xc Yint ext hks hlink hin δ σ g).trans ?_
  have hcard := (Pc.card_influenceSet_le_harmOn (a := a) _ hD hb K hK hsep hw hH).trans
    (Nat.mul_le_mul_left _ (card_admissible_le_of _ fun B hB =>
      (effEscape_le_effMiss L₁ L₂ r.parent hB).trans_lt hk))
  gcongr

/-- **Exfiltration through an accepted two-stage audit on Flock** (variant (b), at the oracle layer): the prover
registers the coarse transcript, commits the drawn replay units' interiors after the coarse draw, and Flock's batched
session proves the fine units drawn fresh inside them. Given the fine partition's lowering (`LoweringSound`, which with
`flock_two_stage_sound` gives `ε_ks = 2^-205` and `δ_link = 0`) and the input unit `AnchorsSound₂` at `δ_in`, a prover
that picks its strategy from a uniform message and a receiver that sees only `out D` of the registration succeed
together with probability at most `2^H · ∑_{j ≤ k} C(nc, j) / |M| + δ + 2^-205 + 0 + δ_in`, with `H` and `k` taken at
`effEscape` because the interior is chosen after the coarse draw. -/
theorem flock_two_stage_exfiltration (r : Refines Pf Pc)
    (plan : Finset (Fin nf) → (Fin C.N → Bool) → List FlockTable) (hlow : Pf.LoweringSound plan)
    {L₁ : Law nc} {L₂ : Fin nc → Law nf} {Int : Finset (Fin nc) → Type} (ext : ∀ S₁, Int S₁ → (Fin C.N → Bool))
    (hin : AnchorsSound₂ (L₁ := L₁) (L₂ := L₂)
      (session := fun S R S₁ I => flockSession plan S (extend (Pc := Pc) R (ext S₁ I))) (fun R _ => R) a δin)
    (hD : D ⊆ Pc.committed) (hb : Fin nc → ℕ)
    (K : Finset (Fin nc) → Finset (Fin C.N)) (hK : ∀ B, K B ⊆ Pc.committed) (hsep : ∀ B, Pc.Separates (K B) B D)
    (hw : ∀ B, (K B).card ≤ ∑ u ∈ B, hb u) {δ : ℝ≥0∞} {H k : ℕ}
    (hH : IsHarmBoundOn (effEscape L₁ L₂ r.parent) (fun u => (hb u : ℝ)) δ H)
    (hk : effMiss L₁ L₂ r.parent (k + 1) < δ) {M : Type} [Fintype M] [Nonempty M] [DecidableEq M]
    (σ : M → Strategy (twoStage L₁ L₂ (Fin C.N → Bool) Int
      (fun S R S₁ I => flockSession plan S (extend (Pc := Pc) R (ext S₁ I)))))
    (g : (D → Bool) → M) :
    avg (fun m => prob (fun o => o.1 = true ∧ g (out D (reg₂ (σ m))) = m)
        (twoStage L₁ L₂ _ Int fun S R S₁ I => flockSession plan S (extend (Pc := Pc) R (ext S₁ I))) (σ m)) ≤
      ((2 ^ H * ∑ j ∈ range (k + 1), nc.choose j : ℕ) : ℝ≥0∞) / (Fintype.card M : ℝ≥0∞) +
        (δ + ENNReal.ofReal ((2 : ℝ) ^ (-(205 : ℝ))) + 0 + δin) :=
  r.twoStage_exfiltration (fun R _ => R) (fun _ ω₁ I _ => ext (L₁.draw ω₁) I)
    (fun _ R S₁ I _ _ => some (extend (Pc := Pc) R (ext S₁ I)))
    (r.oracle₂_knowledgeSound (flock_two_stage_sound plan hlow ext)) r.oracle₂_linkSound
    hin hD hb K hK hsep hw hH hk σ g

end Partition

end FlockSoundness.Audit
