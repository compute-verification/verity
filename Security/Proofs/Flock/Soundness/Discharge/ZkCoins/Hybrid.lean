import Proofs.Flock.Soundness.Discharge.ZkCoins.Lazy
import Proofs.Flock.Soundness.Discharge.ZkCoins.Leaf
import Proofs.Flock.Soundness.Discharge.ZkCoins.Statements
import Proofs.Flock.Soundness.CROnly.Hiding
import Proofs.Flock.Soundness.CROnly.Hm96Sha512
import Proofs.Flock.Soundness.CROnly.ZKKey

/-!
# Committed coins against live coins: a hybrid over committed rounds

`committed_le` (`CommittedLe`) and its hm96-sha512 form `committed_le_hm96` (`CommittedLeHm96`).

Each committed round is in a mode (`m j : Option D`): real (`none`), with its block and salt uniform and the prover
shown its leaf, or ideal with leaf `v` (`some v`), with its block uniform, its salt drawn from the posterior of `v`
(`kern`) and the prover shown `v`. `Exp m` is the committed prover's success probability in mode `m`; with every round
real it is `probC`.

* `exp_step`: one round from real to ideal costs `δ`, with the ideal leaf averaged over its law. For each block, the
  salt is first redrawn from its posterior given its own leaf (`fiber_avg`, exact); the run is then a `[0,1]`-valued
  test of that leaf, and T1 (`hiding_fun`) swaps it for an ideal leaf; with the leaf fixed, the round's law changes
  (`reweight`) to the ideal mode's.
* `exp_le`: by induction on the number of real rounds; when none is left, the prover's view is a constant and the
  rounds' block marginals are uniform, so `lazy` bounds the run by `value` against live coins.

The bound is `value`, the optimal adaptive prover's against live coins, for every committed prover: the commitment
costs `δ` per committed round, added once, and whatever `value` satisfies (the table bound's square, from
`value_interleave_le`) carries over.
-/

namespace FlockSoundness.Discharge.ZkCoins

open FlockSoundness Game Finset Function
open scoped ENNReal

theorem avg_swap {A B : Type} [Fintype A] [Fintype B] (f : A → B → ℝ≥0∞) :
    avg (fun a => avg fun b => f a b) = avg fun b => avg fun a => f a b := by
  unfold avg
  simp only [Finset.mul_sum]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun b _ => Finset.sum_congr rfl fun a _ => ?_
  ring

theorem sum_mul_avg {Ω Q : Type} [Fintype Ω] [Fintype Q] (w : Ω → ℝ≥0∞) (f : Ω → Q → ℝ≥0∞) :
    ∑ ω, w ω * avg (fun q => f ω q) = avg fun q => ∑ ω, w ω * f ω q := by
  unfold avg
  simp only [Finset.mul_sum]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun q _ => Finset.sum_congr rfl fun ω _ => ?_
  ring

theorem avg_const_mul {A : Type} [Fintype A] (x : ℝ≥0∞) (f : A → ℝ≥0∞) :
    avg (fun a => x * f a) = x * avg f := by
  unfold avg
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun a _ => ?_
  ring

theorem fiber_avg' {Salt D : Type} [Fintype Salt] (g : Salt → D) (Φ : Salt → D → ℝ≥0∞) :
    avg (fun y => Φ y (g y)) = avg fun y => ∑ y', kern g (g y) y' * Φ y' (g y) := by
  unfold avg
  rw [← Finset.mul_sum, ← Finset.mul_sum, fiber_avg]

section Hybrid

variable {I Blk Salt D α : Type} [Fintype I] [DecidableEq I] [Fintype Blk] [Nonempty Blk] [Fintype Salt]
  [Nonempty Salt]

/-- Round `j`'s law in mode `m j`: uniform when real; a uniform block and the posterior of leaf `v` when ideal. -/
noncomputable def wt (ℓ : I → Blk × Salt → D) (m : I → Option D) (j : I) (z : Blk × Salt) : ℝ≥0∞ :=
  (m j).elim (Fintype.card (Blk × Salt) : ℝ≥0∞)⁻¹ fun v =>
    (Fintype.card Blk : ℝ≥0∞)⁻¹ * kern (fun y => ℓ j (z.1, y)) v z.2

/-- What the prover is shown of round `j` in mode `m j`: the leaf when real, `v` when ideal. -/
def view (ℓ : I → Blk × Salt → D) (m : I → Option D) (ω : I → Blk × Salt) (j : I) : D :=
  (m j).getD (ℓ j (ω j))

/-- The committed prover's success probability with the rounds in modes `m`. -/
noncomputable def Exp (p : α → Prop) (g : TGame I Blk α) (ℓ : I → Blk × Salt → D)
    (τ : (I → D) → TGame.CStrat Salt g) (m : I → Option D) : ℝ≥0∞ :=
  ∑ ω, W (wt ℓ m) ω * TGame.pr p g (τ (view ℓ m ω)) ω

/-- The run after round `k` is ideal with leaf `v` and block `b`, with the salt from the posterior of `v`. -/
noncomputable def post (p : α → Prop) (g : TGame I Blk α) (ℓ : I → Blk × Salt → D)
    (τ : (I → D) → TGame.CStrat Salt g) (m : I → Option D) (ω : I → Blk × Salt) (k : I) (b : Blk) (v : D) :
    ℝ≥0∞ :=
  ∑ y, kern (fun y => ℓ k (b, y)) v y * TGame.pr p g (τ (update (view ℓ m ω) k v)) (update ω k (b, y))

omit [Fintype I] [DecidableEq I] [Nonempty Blk] [Nonempty Salt] in
theorem wt_none {ℓ : I → Blk × Salt → D} {m : I → Option D} {j : I} (h : m j = none) (z : Blk × Salt) :
    wt ℓ m j z = (Fintype.card (Blk × Salt) : ℝ≥0∞)⁻¹ := by
  unfold wt
  rw [h]
  rfl

omit [Fintype I] [DecidableEq I] [Nonempty Blk] [Nonempty Salt] in
theorem wt_some {ℓ : I → Blk × Salt → D} {m : I → Option D} {j : I} {v : D} (h : m j = some v)
    (z : Blk × Salt) : wt ℓ m j z = (Fintype.card Blk : ℝ≥0∞)⁻¹ * kern (fun y => ℓ j (z.1, y)) v z.2 := by
  unfold wt
  rw [h]
  rfl

omit [Fintype I] [DecidableEq I] [Nonempty Blk] in
theorem wt_blocks (ℓ : I → Blk × Salt → D) (m : I → Option D) (j : I) (b : Blk) :
    ∑ y, wt ℓ m j (b, y) = (Fintype.card Blk : ℝ≥0∞)⁻¹ := by
  cases h : m j with
  | none =>
    simp only [wt_none h, Finset.sum_const, Finset.card_univ, nsmul_eq_mul, Fintype.card_prod, Nat.cast_mul]
    rw [ENNReal.mul_inv (Or.inr (ENNReal.natCast_ne_top _)) (Or.inl (ENNReal.natCast_ne_top _)), mul_left_comm,
      ENNReal.mul_inv_cancel (Nat.cast_ne_zero.2 Fintype.card_ne_zero) (ENNReal.natCast_ne_top _), mul_one]
  | some v =>
    simp only [wt_some h]
    rw [← Finset.mul_sum, sum_kern, mul_one]

omit [Fintype I] [DecidableEq I] in
theorem sum_wt (ℓ : I → Blk × Salt → D) (m : I → Option D) (j : I) : ∑ z, wt ℓ m j z = 1 := by
  rw [Fintype.sum_prod_type]
  simp only [wt_blocks, Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
  exact ENNReal.mul_inv_cancel (Nat.cast_ne_zero.2 Fintype.card_ne_zero) (ENNReal.natCast_ne_top _)

omit [Fintype I] [Fintype Blk] [Nonempty Blk] [Fintype Salt] [Nonempty Salt] in
theorem view_update_none {ℓ : I → Blk × Salt → D} {m : I → Option D} {k : I} (hk : m k = none)
    (ω : I → Blk × Salt) (z : Blk × Salt) : view ℓ m (update ω k z) = update (view ℓ m ω) k (ℓ k z) := by
  funext j
  by_cases hj : j = k
  · subst hj
    simp [view, hk]
  · simp only [view, update_of_ne hj]

omit [Fintype I] [Fintype Blk] [Nonempty Blk] [Fintype Salt] [Nonempty Salt] in
theorem view_update_some (ℓ : I → Blk × Salt → D) (m : I → Option D) (k : I) (v : D) (ω : I → Blk × Salt)
    (z : Blk × Salt) : view ℓ (update m k (some v)) (update ω k z) = update (view ℓ m ω) k v := by
  funext j
  by_cases hj : j = k
  · subst hj
    simp [view]
  · simp only [view, update_of_ne hj]

omit [Fintype I] [Nonempty Blk] [Nonempty Salt] in
theorem wt_update_ne (ℓ : I → Blk × Salt → D) (m : I → Option D) (k : I) (o : Option D) {j : I} (hj : j ≠ k) :
    wt ℓ (update m k o) j = wt ℓ m j := by
  funext z
  simp only [wt, update_of_ne hj]

omit [Fintype I] [Nonempty Blk] [Nonempty Salt] in
theorem wt_update_self (ℓ : I → Blk × Salt → D) (m : I → Option D) (k : I) (v : D) (z : Blk × Salt) :
    wt ℓ (update m k (some v)) k z = (Fintype.card Blk : ℝ≥0∞)⁻¹ * kern (fun y => ℓ k (z.1, y)) v z.2 :=
  wt_some (update_self k (some v) m) z

/-- **The ideal mode's run**: with round `k` ideal with leaf `v`, the committed prover's success probability is the
real-mode weights' average of `post`'s block average. -/
theorem exp_ideal (p : α → Prop) (g : TGame I Blk α) (ℓ : I → Blk × Salt → D)
    (τ : (I → D) → TGame.CStrat Salt g) (m : I → Option D) (k : I) (v : D) :
    Exp p g ℓ τ (update m k (some v)) = ∑ ω, W (wt ℓ m) ω * avg fun b => post p g ℓ τ m ω k b v := by
  have hG : ∀ (ω : I → Blk × Salt) (z₀ : Blk × Salt),
      (∑ z, wt ℓ (update m k (some v)) k z *
        TGame.pr p g (τ (view ℓ (update m k (some v)) (update (update ω k z₀) k z))) (update (update ω k z₀) k z)) =
      ∑ z, wt ℓ (update m k (some v)) k z *
        TGame.pr p g (τ (view ℓ (update m k (some v)) (update ω k z))) (update ω k z) := by
    intro ω z₀
    simp only [update_idem]
  calc Exp p g ℓ τ (update m k (some v))
      = ∑ ω, W (wt ℓ (update m k (some v))) ω * ∑ z, wt ℓ (update m k (some v)) k z *
          TGame.pr p g (τ (view ℓ (update m k (some v)) (update ω k z))) (update ω k z) :=
        resample (sum_wt ℓ (update m k (some v)) k)
          (fun ω => TGame.pr p g (τ (view ℓ (update m k (some v)) ω)) ω)
    _ = ∑ ω, W (wt ℓ m) ω * ∑ z, wt ℓ (update m k (some v)) k z *
          TGame.pr p g (τ (view ℓ (update m k (some v)) (update ω k z))) (update ω k z) :=
        reweight (fun j hj => wt_update_ne ℓ m k (some v) hj) (sum_wt ℓ _ k) (sum_wt ℓ m k)
          (fun ω => ∑ z, wt ℓ (update m k (some v)) k z *
            TGame.pr p g (τ (view ℓ (update m k (some v)) (update ω k z))) (update ω k z)) hG
    _ = ∑ ω, W (wt ℓ m) ω * avg fun b => post p g ℓ τ m ω k b v := by
        refine Finset.sum_congr rfl fun ω _ => ?_
        congr 1
        simp only [wt_update_self, view_update_some]
        rw [Fintype.sum_prod_type]
        unfold avg post
        refine Finset.sum_congr rfl fun b _ => ?_
        rw [Finset.mul_sum]
        refine Finset.sum_congr rfl fun y _ => ?_
        exact mul_assoc _ _ _

/-- **One round from real to ideal.** If round `k`'s leaf, for every block, is `δ`-hidden against the ideal leaves
`ideal q` (`hide`), then making round `k` ideal, with its leaf averaged over `q`, costs at most `δ`. -/
theorem exp_step {Q : Type} [Fintype Q] (p : α → Prop) (g : TGame I Blk α) (ℓ : I → Blk × Salt → D)
    (τ : (I → D) → TGame.CStrat Salt g) (ideal : Q → D) {δ : ℝ≥0∞} (k : I)
    (hide : ∀ (b : Blk) (Γ : D → ℝ≥0∞), (∀ d, Γ d ≤ 1) →
      avg (fun y => Γ (ℓ k (b, y))) ≤ avg (fun q => Γ (ideal q)) + δ)
    (m : I → Option D) (hk : m k = none) :
    Exp p g ℓ τ m ≤ avg (fun q => Exp p g ℓ τ (update m k (some (ideal q)))) + δ := by
  have hreal : ∀ ω, ∑ z, wt ℓ m k z * TGame.pr p g (τ (view ℓ m (update ω k z))) (update ω k z) ≤
      avg (fun q => avg fun b => post p g ℓ τ m ω k b (ideal q)) + δ := by
    intro ω
    have hb : ∀ b : Blk,
        avg (fun y : Salt => TGame.pr p g (τ (update (view ℓ m ω) k (ℓ k (b, y)))) (update ω k (b, y))) ≤
          avg (fun q => post p g ℓ τ m ω k b (ideal q)) + δ := by
      intro b
      refine (le_of_eq (fiber_avg' (fun y => ℓ k (b, y))
        (fun y L => TGame.pr p g (τ (update (view ℓ m ω) k L)) (update ω k (b, y))))).trans ?_
      exact hide b (post p g ℓ τ m ω k b)
        (fun v => sum_kern_le_one _ _ _ fun y => TGame.pr_le_one p g _ _)
    calc ∑ z, wt ℓ m k z * TGame.pr p g (τ (view ℓ m (update ω k z))) (update ω k z)
        = avg (fun z : Blk × Salt => TGame.pr p g (τ (update (view ℓ m ω) k (ℓ k z))) (update ω k z)) := by
          unfold avg
          simp only [wt_none hk, view_update_none hk]
      _ = avg (fun b : Blk => avg fun y : Salt =>
            TGame.pr p g (τ (update (view ℓ m ω) k (ℓ k (b, y)))) (update ω k (b, y))) :=
          ZK.avg_prod_split _
      _ ≤ avg (fun b : Blk => avg (fun q => post p g ℓ τ m ω k b (ideal q)) + δ) := avg_mono hb
      _ = avg (fun q => avg fun b => post p g ℓ τ m ω k b (ideal q)) + δ := by
          rw [avg_add, avg_const]
          congr 1
          exact avg_swap _
  calc Exp p g ℓ τ m
      = ∑ ω, W (wt ℓ m) ω * ∑ z, wt ℓ m k z * TGame.pr p g (τ (view ℓ m (update ω k z))) (update ω k z) :=
        resample (sum_wt ℓ m k) (fun ω => TGame.pr p g (τ (view ℓ m ω)) ω)
    _ ≤ ∑ ω, W (wt ℓ m) ω * (avg (fun q => avg fun b => post p g ℓ τ m ω k b (ideal q)) + δ) :=
        Finset.sum_le_sum fun ω _ => mul_le_mul_right (hreal ω) _
    _ = avg (fun q => ∑ ω, W (wt ℓ m) ω * avg fun b => post p g ℓ τ m ω k b (ideal q)) + δ := by
        simp only [mul_add, Finset.sum_add_distrib]
        rw [← Finset.sum_mul, sum_W_one (sum_wt ℓ m), one_mul]
        congr 1
        exact sum_mul_avg _ _
    _ = avg (fun q => Exp p g ℓ τ (update m k (some (ideal q)))) + δ := by
        simp only [exp_ideal]

theorem card_none_update (m : I → Option D) (k : I) (v : D) :
    (univ.filter fun j => update m k (some v) j = none) = (univ.filter fun j => m j = none).erase k := by
  ext j
  by_cases hj : j = k
  · subst hj
    simp
  · simp [hj]

/-- **The hybrid**: with `n` real rounds, the committed prover's success probability is at most `value` against
live coins plus `n·δ`. -/
theorem exp_le {Q : Type} [Fintype Q] [Nonempty Q] (p : α → Prop) (g : TGame I Blk α) (hg : g.Distinct)
    (ℓ : I → Blk × Salt → D) (τ : (I → D) → TGame.CStrat Salt g) (ideal : Q → D) {δ : ℝ≥0∞}
    (hide : ∀ (k : I) (b : Blk) (Γ : D → ℝ≥0∞), (∀ d, Γ d ≤ 1) →
      avg (fun y => Γ (ℓ k (b, y))) ≤ avg (fun q => Γ (ideal q)) + δ) :
    ∀ (n : ℕ) (m : I → Option D), (univ.filter fun j => m j = none).card = n →
      Exp p g ℓ τ m ≤ value p g.erase + n * δ := by
  intro n
  induction n with
  | zero =>
    intro m hm
    rw [Nat.cast_zero, zero_mul, add_zero]
    let ω₀ : I → Blk × Salt := fun _ => (Classical.arbitrary Blk, Classical.arbitrary Salt)
    have hview : ∀ ω, view ℓ m ω = view ℓ m ω₀ := by
      intro ω
      funext j
      cases h : m j with
      | none =>
        have hj : j ∈ univ.filter fun j => m j = none := by simp [h]
        exact absurd (Finset.card_pos.2 ⟨j, hj⟩) (by omega)
      | some v => simp only [view, h, Option.getD_some]
    calc Exp p g ℓ τ m = ∑ ω, W (wt ℓ m) ω * TGame.pr p g (τ (view ℓ m ω₀)) ω :=
          Finset.sum_congr rfl fun ω _ => by rw [hview ω]
      _ ≤ value p g.erase := lazy p (wt ℓ m) (wt_blocks ℓ m) g hg _
  | succ n ih =>
    intro m hm
    obtain ⟨k, hk⟩ : ∃ k, m k = none := by
      obtain ⟨k, hk⟩ := Finset.card_pos.1 (show 0 < (univ.filter fun j => m j = none).card by omega)
      exact ⟨k, (mem_filter.1 hk).2⟩
    calc Exp p g ℓ τ m ≤ avg (fun q => Exp p g ℓ τ (update m k (some (ideal q)))) + δ :=
          exp_step p g ℓ τ ideal k (hide k) m hk
      _ ≤ avg (fun _ : Q => value p g.erase + n * δ) + δ := by
          have hkm : k ∈ univ.filter fun j => m j = none := by simp [hk]
          have hq : ∀ q : Q, Exp p g ℓ τ (update m k (some (ideal q))) ≤ value p g.erase + n * δ := fun q => by
            refine ih _ ?_
            rw [card_none_update, Finset.card_erase_of_mem hkm, hm, Nat.add_sub_cancel]
          exact add_le_add (avg_mono hq) le_rfl
      _ = value p g.erase + ((n + 1 : ℕ) : ℝ≥0∞) * δ := by
          rw [avg_const]
          push_cast
          ring

end Hybrid

/-- **Committed coins cost their hiding gap once a round** (`CommittedLe`). -/
theorem committed_le : CommittedLe := by
  intro I Blk Salt B Cs D α _ _ _ _ _ _ _ _ p g hg dig My cs H δ₁ hT τ
  have h := exp_le p g hg (leaf dig My cs H) τ (fun q : B × Salt => H (q.1, cs q.2))
    (fun k b Γ hΓ => hiding_fun My cs hT H (dig k b) Γ hΓ) (Fintype.card I) (fun _ => none) (by simp)
  refine Eq.trans_le ?_ h
  unfold Exp probC avg
  refine Finset.sum_congr rfl fun ω _ => ?_
  congr 1
  have hw : ∀ j, wt (leaf dig My cs H) (fun _ : I => (none : Option D)) j (ω j) =
      (Fintype.card (Blk × Salt) : ℝ≥0∞)⁻¹ := fun j => wt_none rfl _
  unfold W
  simp only [hw, Finset.prod_const, Finset.card_univ, Fintype.card_pi, Nat.cast_pow, ENNReal.inv_pow]

/-- **Committed coins at hm96-sha512 under a fresh key** (`CommittedLeHm96`): `committed_le` at each key's event,
interaction, digests, leaf hash and prover, and at its hiding gap (`CROnly.hm96Hiding_gap`), averaged over the key
(`CROnly.hm96_sha512_sum`). -/
theorem committed_le_hm96 : CommittedLeHm96 := by
  intro E I Blk Cs D α _ _ _ hE _ _ _ _ _ p g hg dig cs hcs H τ
  calc avg (fun κ => probC (p κ) (g κ) (dig κ) (CROnly.hankel (n := 512) (l := 1536) κ) cs (H κ) (τ κ))
      ≤ avg (fun κ : Fin (512 + 1536 - 1) → E => value (p κ) (g κ).erase + Fintype.card I *
          ENNReal.ofReal (CROnly.hidingGap (CROnly.hankel (n := 512) (l := 1536) κ) cs)) :=
        avg_mono fun κ => committed_le (p κ) (g κ) (hg κ) (dig κ) (CROnly.hankel (n := 512) (l := 1536) κ) cs (H κ) _
          (CROnly.hm96Hiding_gap _ _) (τ κ)
    _ = avg (fun κ : Fin (512 + 1536 - 1) → E => value (p κ) (g κ).erase) + Fintype.card I *
          avg (fun κ : Fin (512 + 1536 - 1) → E =>
            ENNReal.ofReal (CROnly.hidingGap (CROnly.hankel (n := 512) (l := 1536) κ) cs)) := by
        rw [avg_add, avg_const_mul]
    _ ≤ avg (fun κ : Fin (512 + 1536 - 1) → E => value (p κ) (g κ).erase) +
          Fintype.card I * ENNReal.ofReal (1 / 2 ^ 257) := by
        gcongr
        exact CROnly.avg_ofReal_le _ (fun κ => CROnly.hidingGap_nonneg _ _) (CROnly.hm96_sha512_sum hE cs hcs)

end FlockSoundness.Discharge.ZkCoins
