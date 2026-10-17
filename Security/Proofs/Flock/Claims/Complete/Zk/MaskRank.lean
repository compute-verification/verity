import Proofs.Flock.Level3.ArithFacts
import Proofs.Flock.Level3.GF256

/-!
# The mask slot's rank

The `--zk` prover refuses coins at which the mask slot's weights are rank-deficient (`flock-circuit.rs`, "degenerate
coins"): with the claims' points `p_1, …, p_k` (both reps' `ab` and `c` claims so far, cumulative), the vectors
`v_x = (eq(p_1, x), …, eq(p_k, x))` over the slot's `N = 2^D` words `x` must have GF(2)-rank `128·k`. In
`A = GF(2^128)^k` with pointwise product, `v_x = ∏_t eq(c_t, x_t)` at `c_t = (p_{1,t}, …, p_{k,t})`, where
`eq(c, 1) = c` and `eq(c, 0) = 1 + c`; the GF(2)-span is the additive subgroup the `v_x` generate (`span`), and the rank
is full when it is `⊤` (`RankFull`).

* `span_cons`: one more coordinate `c₀` takes the span `U` to `U + c₀·U` (the recursion every bound runs on);
* `span_map`, `rankFull_sub`: the span commutes with a ring hom, so full rank at the `k` points gives full rank at any
  subset: rep 1's check passing means rep 0's passed, and the refusal is one event per table;
* `RankBound`: the bound's statement, the fraction of coins at which the rank is deficient, with some coordinates
  fixed (the `c` claims' pinned word coordinates).
-/

namespace FlockComplete.MaskRank

section Ring

variable {A : Type*} [CommRing A]

/-- The eq-vector at the Boolean point `x`: `∏_t eq(c_t, x_t)`, with `eq(c, 1) = c` and `eq(c, 0) = 1 + c`. -/
def eqProd {D : ℕ} (c : Fin D → A) (x : Fin D → Bool) : A := ∏ t, if x t then c t else 1 + c t

/-- **The GF(2)-span of the eq-vectors**: in characteristic 2, the additive subgroup they generate. -/
def span {D : ℕ} (c : Fin D → A) : AddSubgroup A := AddSubgroup.closure (Set.range (eqProd c))

theorem eqProd_cons {D : ℕ} (c₀ : A) (c : Fin D → A) (b : Bool) (x : Fin D → Bool) :
    eqProd (Fin.cons c₀ c : Fin (D + 1) → A) (Fin.cons b x) = (if b then c₀ else 1 + c₀) * eqProd c x := by
  simp only [eqProd, Fin.prod_univ_succ, Fin.cons_zero, Fin.cons_succ]

theorem range_eqProd_cons {D : ℕ} (c₀ : A) (c : Fin D → A) :
    Set.range (eqProd (Fin.cons c₀ c : Fin (D + 1) → A)) =
      (fun v => (1 + c₀) * v) '' Set.range (eqProd c) ∪ (fun v => c₀ * v) '' Set.range (eqProd c) := by
  ext y
  constructor
  · rintro ⟨x, rfl⟩
    rw [← Fin.cons_self_tail x, eqProd_cons]
    cases x 0
    · exact Or.inl ⟨_, ⟨Fin.tail x, rfl⟩, by simp⟩
    · exact Or.inr ⟨_, ⟨Fin.tail x, rfl⟩, by simp⟩
  · rintro (⟨v, ⟨x, rfl⟩, rfl⟩ | ⟨v, ⟨x, rfl⟩, rfl⟩)
    · exact ⟨Fin.cons false x, by rw [eqProd_cons]; simp⟩
    · exact ⟨Fin.cons true x, by rw [eqProd_cons]; simp⟩

/-- With no coordinate the span is GF(2)·1. -/
theorem span_zero (c : Fin 0 → A) : span c = AddSubgroup.closure {1} := by
  have : Set.range (eqProd c) = {1} := by
    ext y
    simp only [Set.mem_range, Set.mem_singleton_iff, eqProd, Finset.univ_eq_empty, Finset.prod_empty]
    exact ⟨fun ⟨_, h⟩ => h.symm, fun h => ⟨finZeroElim, h.symm⟩⟩
  rw [span, this]

/-- **The span recursion**: one more coordinate `c₀` takes the span `U` to `U + c₀·U`, since
`(1 + c₀)·v + c₀·v = v` in characteristic 2. -/
theorem span_cons (h2 : ∀ a : A, a + a = 0) {D : ℕ} (c₀ : A) (c : Fin D → A) :
    span (Fin.cons c₀ c : Fin (D + 1) → A) = span c ⊔ (span c).map (AddMonoidHom.mulLeft c₀) := by
  apply le_antisymm
  · rw [span, AddSubgroup.closure_le, range_eqProd_cons]
    rintro y (⟨v, hv, rfl⟩ | ⟨v, hv, rfl⟩)
    · show (1 + c₀) * v ∈ _
      rw [add_mul, one_mul]
      exact AddSubgroup.add_mem _ (AddSubgroup.mem_sup_left (AddSubgroup.subset_closure hv))
        (AddSubgroup.mem_sup_right ⟨v, AddSubgroup.subset_closure hv, rfl⟩)
    · exact AddSubgroup.mem_sup_right ⟨v, AddSubgroup.subset_closure hv, rfl⟩
  · refine sup_le ?_ ?_
    · rw [span, AddSubgroup.closure_le]
      intro v hv
      have e : v = (1 + c₀) * v + c₀ * v := by rw [add_mul, one_mul, add_assoc, h2, add_zero]
      rw [SetLike.mem_coe, e]
      refine AddSubgroup.add_mem _ (AddSubgroup.subset_closure ?_) (AddSubgroup.subset_closure ?_)
      · rw [range_eqProd_cons]; exact Or.inl ⟨v, hv, rfl⟩
      · rw [range_eqProd_cons]; exact Or.inr ⟨v, hv, rfl⟩
    · rw [span, AddMonoidHom.map_closure, AddSubgroup.closure_le]
      rintro _ ⟨v, hv, rfl⟩
      apply AddSubgroup.subset_closure
      rw [range_eqProd_cons]
      exact Or.inr ⟨v, hv, rfl⟩

/-- **The span commutes with a ring hom**: the image of the span is the span of the images. -/
theorem span_map {B : Type*} [CommRing B] (π : A →+* B) {D : ℕ} (c : Fin D → A) :
    (span c).map π.toAddMonoidHom = span (fun t => π (c t)) := by
  rw [span, AddMonoidHom.map_closure, span, ← Set.range_comp]
  congr 2
  funext x
  simp [eqProd, map_prod, apply_ite]

theorem span_top_of_map {B : Type*} [CommRing B] (π : A →+* B) (hπ : Function.Surjective π) {D : ℕ}
    (c : Fin D → A) (h : span c = ⊤) : span (fun t => π (c t)) = ⊤ := by
  rw [← span_map, h]
  exact AddSubgroup.map_top_of_surjective _ hπ

end Ring

/-! ## At the deployed field -/

open FlockLevel3

/-- **The mask slot's weights have full rank** at the points `p` (point `i`'s word coordinate `t`): the eq-vectors over
the slot's `2^D` words span `GF(2^128)^k` over GF(2), i.e. have GF(2)-rank `128·k`. -/
def RankFull {k D : ℕ} (p : Fin k → Fin D → GF128) : Prop := span (fun t i => p i t) = ⊤

theorem add_self_pi {k : ℕ} (a : Fin k → GF128) : a + a = 0 := funext fun i => CharTwo.add_self_eq_zero (a i)

/-- **Full rank at `k` points gives it at any sub-family**: the projection to the sub-family is a surjective ring hom.
So the prover's cumulative refusal (rep 0's `ab`, `c`; then all four points at rep 1) is the one event "the last
check's points are rank-deficient". -/
theorem rankFull_sub {j k D : ℕ} (e : Fin j ↪ Fin k) (p : Fin k → Fin D → GF128) (h : RankFull p) :
    RankFull (fun i => p (e i)) := by
  let π : (Fin k → GF128) →+* (Fin j → GF128) := RingHom.pi fun i => Pi.evalRingHom _ (e i)
  have hπ : Function.Surjective π := fun b =>
    ⟨Function.extend e b 0, funext fun i' => by
      simp only [π, RingHom.pi_apply, Pi.evalRingHom_apply]
      exact e.injective.extend_apply b 0 i'⟩
  have key : (fun t => π (fun i => p i t)) = fun t i => p (e i) t := by
    funext t i
    simp only [π, RingHom.pi_apply, Pi.evalRingHom_apply]
  unfold RankFull
  rw [← key]
  exact span_top_of_map π hπ _ h

/-- **Nonzero scalars per point keep the rank.** The prover's check evaluates `eq` over the whole word index, so the
slot's fixed in-block bits multiply point `i`'s weights by `γ i`, the product of their `eq` factors. Where every `γ i`
is nonzero that is a GF(2)-automorphism of `GF(2^128)^k`: the scaled vectors have full rank exactly at `RankFull`, and
the refusal's other cause is a coin at the one value that zeroes its `eq` factor. -/
theorem rankFull_scale_iff {k D : ℕ} (γ : Fin k → GF128) (hγ : ∀ i, γ i ≠ 0) (p : Fin k → Fin D → GF128) :
    AddSubgroup.closure (Set.range fun x => γ * eqProd (fun t i => p i t) x) = ⊤ ↔ RankFull p := by
  have hl : γ⁻¹ * γ = 1 := funext fun i => inv_mul_cancel₀ (hγ i)
  have hr : γ * γ⁻¹ = 1 := funext fun i => mul_inv_cancel₀ (hγ i)
  have hinj : Function.Injective (AddMonoidHom.mulLeft γ) := fun x y hxy => by
    have := congrArg (γ⁻¹ * ·) hxy
    simpa only [AddMonoidHom.coe_mulLeft, ← mul_assoc, hl, one_mul] using this
  have hsur : Function.Surjective (AddMonoidHom.mulLeft γ) := fun y =>
    ⟨γ⁻¹ * y, by rw [AddMonoidHom.coe_mulLeft, ← mul_assoc, hr, one_mul]⟩
  have hmap : AddSubgroup.closure (Set.range fun x => γ * eqProd (fun t i => p i t) x) =
      (span fun t i => p i t).map (AddMonoidHom.mulLeft γ) := by
    rw [span, AddMonoidHom.map_closure, ← Set.range_comp]
    rfl
  have htop : (⊤ : AddSubgroup (Fin k → GF128)).map (AddMonoidHom.mulLeft γ) = ⊤ :=
    AddSubgroup.map_top_of_surjective _ hsur
  unfold RankFull
  rw [hmap]
  exact ⟨fun h => AddSubgroup.map_injective hinj (h.trans htop.symm), fun h => h ▸ htop⟩

open Classical in
/-- **The rank bound's statement** at `k` points with `D` word coordinates each: coordinate `t` of point `i` is the
constant `fix i t` where that is `some`, else its own uniform coin; at most a fraction `2^-κ` of the coins leave the
mask slot's weights rank-deficient. -/
def RankBound (k D : ℕ) (fix : Fin k → Fin D → Option GF128) (κ : ℕ) : Prop :=
  (Finset.univ.filter fun r : Fin k → Fin D → GF128 => ¬ RankFull fun i t => (fix i t).getD (r i t)).card * 2 ^ κ ≤
    Fintype.card (Fin k → Fin D → GF128)

/-- The deployed table's four points, in order rep 0's `ab`, `c`, rep 1's `ab`, `c`, at `m = 25`, `mask_log = 14`
(`N = 1024` words, `D = 10`): a `c` claim's first six word coordinates are the pinned `pinned[1..7)`, every other
coordinate a coin of its own. -/
def deployedFix (i : Fin 4) (t : Fin 10) : Option GF128 :=
  if i.val % 2 = 1 ∧ t.val < 6 then some (Flock.pinned[t.val + 1]!) else none

/-- **The deployed rank bound**, the conjunct G11's `CoinsT` needs (as `¬ RankFull` in `BadZk`, one slot per table):
at most `2^-100` of the coins refuse. Open: the note's argument, not yet in Lean. -/
def DeployedRankBound : Prop := RankBound 4 10 deployedFix 100

end FlockComplete.MaskRank
