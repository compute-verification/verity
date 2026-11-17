module

public import Definitions.Core.Game
public import Definitions.Flock.ZK.RealView
public import Definitions.Flock.ZK.SHVZK
public import Proofs.Flock.Soundness.ZK.Dist
public import Proofs.Flock.Soundness.ZK.RealLeaves
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Tactic.Positivity.Finset
import Proofs.Flock.Soundness.Game
import Proofs.Flock.Soundness.ZK.Hiding
import Proofs.Flock.Soundness.ZK.SHVZK

/-! Zero knowledge of one session of `J` masked tables from its tables', by independence (`SameDist.pi`,
`prCoin_pi_close`); every leaf's salt is the operating system's bytes, uniform under
`Assumptions.UniformRandomBytes`. -/

@[expose] public section

namespace FlockSoundness.ZK

open Finset Game
open scoped ENNReal

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- instance search over the ZK view's and coins' product types

theorem SameDist.pi {J : ℕ} {Ω Ω' V : Fin J → Type} [∀ j, Fintype (Ω j)] [∀ j, Fintype (Ω' j)]
    (f : ∀ j, Ω j → V j) (g : ∀ j, Ω' j → V j) (h : ∀ j, SameDist (f j) (g j)) :
    SameDist (fun ω : ∀ j, Ω j => fun j => f j (ω j)) (fun σ : ∀ j, Ω' j => fun j => g j (σ j)) := by
  classical
  have count : ∀ {Ω₀ : Fin J → Type} [∀ j, Fintype (Ω₀ j)] (k : ∀ j, Ω₀ j → V j) (v : ∀ j, V j),
      #{ω : ∀ j, Ω₀ j | (fun j => k j (ω j)) = v} = ∏ j, #{x : Ω₀ j | k j x = v j} := by
    intro Ω₀ _ k v
    rw [← Fintype.card_piFinset]
    congr 1
    ext ω
    simp [funext_iff, Fintype.mem_piFinset]
  intro v
  have key : (∏ j, #{x : Ω j | f j x = v j}) * Fintype.card (∀ j, Ω' j)
      = (∏ j, #{x : Ω' j | g j x = v j}) * Fintype.card (∀ j, Ω j) := by
    rw [Fintype.card_pi, Fintype.card_pi, ← Finset.prod_mul_distrib, ← Finset.prod_mul_distrib]
    exact Finset.prod_congr rfl fun j _ => by convert h j (v j) using 2
  convert key using 2
  · convert count f v
  · convert count g v

theorem prCoin_pi_close_aux : ∀ (J : ℕ) {Ω Ω' V : Fin J → Type} [∀ j, Fintype (Ω j)] [∀ j, Nonempty (Ω j)]
    [∀ j, Fintype (Ω' j)] [∀ j, Nonempty (Ω' j)] (f : ∀ j, Ω j → V j) (g : ∀ j, Ω' j → V j) (ε : Fin J → ℝ≥0∞),
    (∀ j (E : V j → Prop), prCoin (fun ω => E (f j ω)) ≤ prCoin (fun σ => E (g j σ)) + ε j) →
    ∀ E : (∀ j, V j) → Prop,
      prCoin (fun ω : ∀ j, Ω j => E fun j => f j (ω j))
        ≤ prCoin (fun σ : ∀ j, Ω' j => E fun j => g j (σ j)) + ∑ j, ε j := by
  intro J
  induction J with
  | zero =>
    intro Ω Ω' V _ _ _ _ f g ε _ E
    have e : ∀ {Ω₀ : Fin 0 → Type} [∀ j, Fintype (Ω₀ j)] (k : ∀ j, Ω₀ j → V j),
        prCoin (fun ω : ∀ j, Ω₀ j => E fun j => k j (ω j)) = prCoin (fun _ : ∀ j, Ω₀ j => E finZeroElim) := by
      intro Ω₀ _ k
      congr 1
      funext ω
      congr 1
      funext j
      exact j.elim0
    rw [e f, e g, Finset.univ_eq_empty, Finset.sum_empty, add_zero]
    unfold prCoin
    rw [avg_const, avg_const]
  | succ n ih =>
    intro Ω Ω' V _ _ _ _ f g ε h E
    have hA : prCoin (fun ω : ∀ j, Ω j => E fun j => f j (ω j))
        = prCoin (fun p : (∀ j : Fin n, Ω j.succ) × Ω 0 =>
            E (Fin.cons (f 0 p.2) fun j => f j.succ (p.1 j))) := by
      refine Eq.trans ?_ (prCoin_congr_equiv ((Fin.consEquiv Ω).symm.trans (Equiv.prodComm _ _)) _)
      congr 1
      funext ω
      congr 1
      funext j
      refine Fin.cases ?_ (fun j => ?_) j <;> rfl
    have hB : prCoin (fun p : (∀ j : Fin n, Ω j.succ) × Ω' 0 => E (Fin.cons (g 0 p.2) fun j => f j.succ (p.1 j)))
        = prCoin (fun p : Ω' 0 × (∀ j : Fin n, Ω j.succ) => E (Fin.cons (g 0 p.1) fun j => f j.succ (p.2 j))) :=
      prCoin_congr_equiv (Equiv.prodComm _ _) (fun p : Ω' 0 × (∀ j : Fin n, Ω j.succ) =>
        E (Fin.cons (g 0 p.1) fun j => f j.succ (p.2 j)))
    have hC : prCoin (fun p : Ω' 0 × (∀ j : Fin n, Ω' j.succ) => E (Fin.cons (g 0 p.1) fun j => g j.succ (p.2 j)))
        = prCoin (fun σ : ∀ j, Ω' j => E fun j => g j (σ j)) := by
      refine Eq.trans ?_ (prCoin_congr_equiv (Fin.consEquiv Ω') _)
      congr 1
      funext p
      congr 1
      funext j
      refine Fin.cases ?_ (fun j => ?_) j <;> rfl
    have s1 : prCoin (fun p : (∀ j : Fin n, Ω j.succ) × Ω 0 => E (Fin.cons (f 0 p.2) fun j => f j.succ (p.1 j)))
        ≤ prCoin (fun p : (∀ j : Fin n, Ω j.succ) × Ω' 0 => E (Fin.cons (g 0 p.2) fun j => f j.succ (p.1 j))) + ε 0 := by
      rw [prCoin_prod_split (fun (r : ∀ j : Fin n, Ω j.succ) (x : Ω 0) => E (Fin.cons (f 0 x) fun j => f j.succ (r j))),
        prCoin_prod_split (fun (r : ∀ j : Fin n, Ω j.succ) (y : Ω' 0) => E (Fin.cons (g 0 y) fun j => f j.succ (r j)))]
      calc avg (fun r : ∀ j : Fin n, Ω j.succ => prCoin fun x : Ω 0 => E (Fin.cons (f 0 x) fun j => f j.succ (r j)))
          ≤ avg (fun r : ∀ j : Fin n, Ω j.succ =>
              prCoin (fun y : Ω' 0 => E (Fin.cons (g 0 y) fun j => f j.succ (r j))) + ε 0) :=
            avg_mono fun r => h 0 (fun v => E (Fin.cons v fun j => f j.succ (r j)))
        _ = _ := by rw [avg_add, avg_const]
    have s2 : prCoin (fun p : Ω' 0 × (∀ j : Fin n, Ω j.succ) => E (Fin.cons (g 0 p.1) fun j => f j.succ (p.2 j)))
        ≤ prCoin (fun p : Ω' 0 × (∀ j : Fin n, Ω' j.succ) => E (Fin.cons (g 0 p.1) fun j => g j.succ (p.2 j)))
          + ∑ j : Fin n, ε j.succ := by
      rw [prCoin_prod_split (fun (y : Ω' 0) (r : ∀ j : Fin n, Ω j.succ) => E (Fin.cons (g 0 y) fun j => f j.succ (r j))),
        prCoin_prod_split (fun (y : Ω' 0) (r : ∀ j : Fin n, Ω' j.succ) => E (Fin.cons (g 0 y) fun j => g j.succ (r j)))]
      calc avg (fun y : Ω' 0 => prCoin fun r : ∀ j : Fin n, Ω j.succ => E (Fin.cons (g 0 y) fun j => f j.succ (r j)))
          ≤ avg (fun y : Ω' 0 => prCoin (fun r : ∀ j : Fin n, Ω' j.succ => E (Fin.cons (g 0 y) fun j => g j.succ (r j)))
              + ∑ j : Fin n, ε j.succ) :=
            avg_mono fun y => ih (fun j => f j.succ) (fun j => g j.succ) (fun j => ε j.succ) (fun j => h j.succ)
              (fun vs => E (Fin.cons (g 0 y) vs))
        _ = _ := by rw [avg_add, avg_const]
    rw [hA]
    calc _ ≤ _ := s1
      _ = _ := by rw [hB]
      _ ≤ _ := add_le_add s2 le_rfl
      _ = prCoin (fun σ : ∀ j, Ω' j => E fun j => g j (σ j)) + ∑ j, ε j := by
        rw [hC, Fin.sum_univ_succ]
        ring

theorem prCoin_pi_close {J : ℕ} {Ω Ω' V : Fin J → Type} [∀ j, Fintype (Ω j)] [∀ j, Nonempty (Ω j)]
    [∀ j, Fintype (Ω' j)] [∀ j, Nonempty (Ω' j)] (f : ∀ j, Ω j → V j) (g : ∀ j, Ω' j → V j) (ε : Fin J → ℝ≥0∞)
    (h : ∀ j (E : V j → Prop), prCoin (fun ω => E (f j ω)) ≤ prCoin (fun σ => E (g j σ)) + ε j)
    (E : (∀ j, V j) → Prop) :
    prCoin (fun ω : ∀ j, Ω j => E fun j => f j (ω j)) ≤ prCoin (fun σ : ∀ j, Ω' j => E fun j => g j (σ j)) + ∑ j, ε j :=
  prCoin_pi_close_aux J f g ε h E

namespace Session

variable {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] {J : ℕ} (T : Fin J → Table F K W Salt Seed D Hi VL)

variable [CharP F 2]

noncomputable def view (w : Fin J → W) (ω : ∀ j, (T j).Rand) : ∀ j, (T j).View := fun j => (T j).view (w j) (ω j)

noncomputable def sim (w0 : Fin J → W) (σ : ∀ j, (T j).SimRand) : ∀ j, (T j).View := fun j => (T j).sim (w0 j) (σ j)

noncomputable def preView (w : Fin J → W) (ω : ∀ j, (T j).Rand) : ∀ j, (T j).PreView :=
  fun j => (T j).preView (w j) (ω j)

variable [AddCommGroup Salt] [AddCommGroup Seed] [AddCommGroup Hi] [Fintype F] [DecidableEq F] [Fintype K]
  [DecidableEq K] [Fintype Salt] [DecidableEq Salt] [Fintype Seed] [DecidableEq Seed] [Fintype Hi] [DecidableEq Hi]

theorem session_shvzk (hrank : ∀ j, (T j).Rank) (hpad : ∀ j, (T j).PadOnto)
    (hW : ∀ j r, Function.Bijective fun p : F × F => (T j).Wr r p.1 p.2) (hpads : ∀ j, (T j).PadsOnto)
    (hβ : ∀ j r, (T j).β r ≠ 0) (w : Fin J → W) (hreg : ∀ j, (T j).sReg (w j) = (T j).pubReg)
    (hinner : ∀ j, (T j).InnerHolds (w j)) (w0 : Fin J → W) :
    SameDist (view T w) (sim T w0) :=
  SameDist.pi (fun j => (T j).view (w j)) (fun j => (T j).sim (w0 j))
    (fun j => (T j).table_shvzk (hrank j) (hpad j) (hW j) (hpads j) (hβ j) (w j) (hreg j) (hinner j) (w0 j))

omit [Fintype K] [DecidableEq K] in
theorem session_prefinal_indep (hrank : ∀ j, (T j).Rank)
    (hW : ∀ j r, Function.Bijective fun p : F × F => (T j).Wr r p.1 p.2) (hβ : ∀ j r, (T j).β r ≠ 0)
    (w w' : Fin J → W) (hreg : ∀ j, (T j).sReg (w j) = (T j).pubReg) (hreg' : ∀ j, (T j).sReg (w' j) = (T j).pubReg) :
    SameDist (preView T w) (preView T w') :=
  SameDist.pi (fun j => (T j).preView (w j)) (fun j => (T j).preView (w' j))
    (fun j => (T j).table_prefinal_indep (hrank j) (hW j) (hβ j) (w j) (w' j) (hreg j) (hreg' j))

end Session

namespace Session

variable {F K W Salt B D Hi VL : Type} [Field F] [Field K] [AddCommGroup B] {J : ℕ}
  (T : Fin J → Table F K W Salt (B × Salt) D Hi VL) [CharP F 2]
  [AddCommGroup Salt] [AddCommGroup Hi] [Fintype F] [DecidableEq F] [Fintype K] [DecidableEq K]
  [Fintype Salt] [DecidableEq Salt] [Fintype B] [DecidableEq B] [Fintype Hi] [DecidableEq Hi]

theorem session_shvzk_hm96 {C : Type} (M : ∀ j, (T j).Hm96 C) {δ₁ : ℝ≥0∞}
    (hT1 : ∀ j, Claims.Hm96Hiding (M j).My (M j).cs δ₁) (hrank : ∀ j, (T j).Rank) (hpad : ∀ j, (T j).PadOnto)
    (hW : ∀ j r, Function.Bijective fun p : F × F => (T j).Wr r p.1 p.2) (hpads : ∀ j, (T j).PadsOnto)
    (hβ : ∀ j r, (T j).β r ≠ 0) (w : Fin J → W) (hreg : ∀ j, (T j).sReg (w j) = (T j).pubReg)
    (hinner : ∀ j, (T j).InnerHolds (w j)) (w0 : Fin J → W) (E : (∀ j, (T j).View) → Prop) :
    prCoin (fun p : ∀ j, (T j).Rand × ((T j).Hid → Salt) => E fun j => (M j).viewR (w j) (p j).1 (p j).2)
        ≤ prCoin (fun p : ∀ j, (T j).SimRand × ((T j).Hid → Salt) => E fun j => (M j).simR (w0 j) (p j).1 (p j).2)
          + ∑ j, 2 * Fintype.card (T j).Hid * δ₁
      ∧ prCoin (fun p : ∀ j, (T j).SimRand × ((T j).Hid → Salt) => E fun j => (M j).simR (w0 j) (p j).1 (p j).2)
        ≤ prCoin (fun p : ∀ j, (T j).Rand × ((T j).Hid → Salt) => E fun j => (M j).viewR (w j) (p j).1 (p j).2)
          + ∑ j, 2 * Fintype.card (T j).Hid * δ₁ := by
  have : Nonempty Salt := ⟨0⟩
  have : ∀ j, Nonempty ((T j).Rand × ((T j).Hid → Salt)) := fun _ => ⟨0⟩
  have : ∀ j, Nonempty ((T j).SimRand × ((T j).Hid → Salt)) := fun _ => ⟨0⟩
  have h1 : ∀ j (E' : (T j).View → Prop),
      prCoin (fun p : (T j).Rand × ((T j).Hid → Salt) => E' ((M j).viewR (w j) p.1 p.2))
        ≤ prCoin (fun p : (T j).SimRand × ((T j).Hid → Salt) => E' ((M j).simR (w0 j) p.1 p.2))
          + 2 * Fintype.card (T j).Hid * δ₁ := fun j E' => by
    convert ((T j).table_shvzk_hm96 (M j) (hT1 j) (hrank j) (hpad j) (hW j) (hpads j) (hβ j) (w j) (hreg j)
      (hinner j) (w0 j) E').1 using 3
  have h2 : ∀ j (E' : (T j).View → Prop),
      prCoin (fun p : (T j).SimRand × ((T j).Hid → Salt) => E' ((M j).simR (w0 j) p.1 p.2))
        ≤ prCoin (fun p : (T j).Rand × ((T j).Hid → Salt) => E' ((M j).viewR (w j) p.1 p.2))
          + 2 * Fintype.card (T j).Hid * δ₁ := fun j E' => by
    convert ((T j).table_shvzk_hm96 (M j) (hT1 j) (hrank j) (hpad j) (hW j) (hpads j) (hβ j) (w j) (hreg j)
      (hinner j) (w0 j) E').2 using 3
  exact ⟨prCoin_pi_close (Ω := fun j => (T j).Rand × ((T j).Hid → Salt))
      (Ω' := fun j => (T j).SimRand × ((T j).Hid → Salt)) (V := fun j => (T j).View)
      (fun j p => (M j).viewR (w j) p.1 p.2) (fun j p => (M j).simR (w0 j) p.1 p.2)
      (fun j => 2 * Fintype.card (T j).Hid * δ₁) h1 E,
    prCoin_pi_close (Ω := fun j => (T j).SimRand × ((T j).Hid → Salt))
      (Ω' := fun j => (T j).Rand × ((T j).Hid → Salt)) (V := fun j => (T j).View)
      (fun j p => (M j).simR (w0 j) p.1 p.2) (fun j p => (M j).viewR (w j) p.1 p.2)
      (fun j => 2 * Fintype.card (T j).Hid * δ₁) h2 E⟩

end Session

end FlockSoundness.ZK
