module

public import Proofs.Flock.ZkAny.Tables
public import Proofs.Flock.Soundness.ZK.Abort
public import Definitions.Flock.ZK.RealView

/-! The ZK server's release with real hm96 leaves on both branches, at any values, within `2·δ₁` a leaf it leaves unopened (`Session.release_hm96`). -/

@[expose] public section

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace FlockSoundness.ZK

open Verity.Game
open scoped ENNReal

namespace Table

variable {F K W Salt B D Hi VL : Type} [Field F] [Field K] [AddCommGroup B] [CharP F 2]
  (T : Table F K W Salt (B × Salt) D Hi VL)

abbrev PreLeaves := (Fin T.nS ⊕ (Fin 2 × Fin T.nQ) ⊕ Fin 2) ⊕ T.Hid

def preOf (v : T.View) : T.PreView :=
  { sHat := v.sHat, sReg := v.sReg, masked := v.masked, y1 := v.y1, ybar := v.ybar, lig := v.lig, inner := v.inner
    Y1 := v.Y1, Y2 := v.Y2, leaves0 := v.leaves0, tauCom := v.tauCom, leavesP := v.leavesP }

variable {T} {C : Type}

noncomputable def Hm96.preR (M : T.Hm96 C) (w : W) (ω : T.Rand) (ys : T.Hid → Salt) : T.PreView :=
  T.preOf (M.viewR w ω ys)

def Hm96.TauLeaf (M : T.Hm96 C) (digT : F → B) : Prop :=
  ∀ t y, T.leafT t y = M.H (digT t + M.My y, M.cs y)

noncomputable def Hm96.releaseR (M : T.Hm96 C) (go : T.Rand → Bool) (w : W) (ω : T.Rand) (ys : T.Hid → Salt) :
    T.View ⊕ T.PreView :=
  if go ω then .inl (M.viewR w ω ys) else .inr (M.preR w ω ys)

end Table

namespace Session

variable {F K W Salt B D Hi VL : Type} [Field F] [Field K] [AddCommGroup B] [CharP F 2] {J : ℕ}
  (T : Fin J → Table F K W Salt (B × Salt) D Hi VL) {C : Type} (M : ∀ j, (T j).Hm96 C) {St : Type}

noncomputable def releaseR (full : St → Fin J → Bool) (w : Fin J → W) (s : St)
    (p : ∀ j, (T j).Rand × ((T j).Hid → Salt)) : ∀ j, (T j).View ⊕ (T j).PreView :=
  fun j => if full s j then .inl ((M j).viewR (w j) (p j).1 (p j).2) else .inr ((M j).preR (w j) (p j).1 (p j).2)

noncomputable def simR (full : St → Fin J → Bool) (w₀ : Fin J → W) (s : St)
    (σ : ∀ j, ((T j).SimRand × ((T j).Hid → Salt)) × ((T j).Rand × ((T j).Hid → Salt))) :
    ∀ j, (T j).View ⊕ (T j).PreView :=
  fun j => if full s j then .inl ((M j).simR (w₀ j) (σ j).1.1 (σ j).1.2)
    else .inr ((M j).preR (w₀ j) (σ j).2.1 (σ j).2.2)

def unopened (full : St → Fin J → Bool) (s : St) : ℕ :=
  ∑ j, if full s j then Fintype.card (T j).Hid else Fintype.card (T j).PreLeaves

/-- A server that answers `level0` only once every table's witness is in gives it. -/
def WitnessesFirst {Ω : Type} (wit : Ω → Fin J → W) : Prop :=
  ∀ p p', wit p = wit p'

end Session

theorem SameDist.prCoin_prod_eq {Ω Ω' U V : Type} [Fintype Ω] [Fintype Ω'] [Fintype U] [Nonempty Ω] [Nonempty Ω']
    {f : Ω → V} {g : Ω' → V} (h : SameDist f g) (G : V → U → Prop) :
    prCoin (fun p : Ω × U => G (f p.1) p.2) = prCoin (fun p : Ω' × U => G (g p.1) p.2) := by
  calc prCoin (fun p : Ω × U => G (f p.1) p.2)
      = prCoin (fun q : U × Ω => G (f q.2) q.1) :=
        (prCoin_congr_equiv (Equiv.prodComm U Ω) (fun p : Ω × U => G (f p.1) p.2)).symm
    _ = avg fun u => prCoin fun ω => G (f ω) u := prCoin_prod_split fun (u : U) ω => G (f ω) u
    _ = avg fun u => prCoin fun σ => G (g σ) u := congrArg avg (funext fun u => h.prCoin_eq fun v => G v u)
    _ = prCoin (fun q : U × Ω' => G (g q.2) q.1) := (prCoin_prod_split fun (u : U) σ => G (g σ) u).symm
    _ = prCoin (fun p : Ω' × U => G (g p.1) p.2) :=
        prCoin_congr_equiv (Equiv.prodComm U Ω') (fun p : Ω' × U => G (g p.1) p.2)

namespace Table

variable {F K W Salt B D Hi VL : Type} [Field F] [Field K] [CharP F 2] [Fintype F] [DecidableEq F] [Fintype K]
  [DecidableEq K] [Fintype Salt] [DecidableEq Salt] [Fintype B] [DecidableEq B] [Fintype Hi] [DecidableEq Hi]
  [AddCommGroup Salt] [AddCommGroup B] [AddCommGroup Hi] (T : Table F K W Salt (B × Salt) D Hi VL) {C : Type}

def hidWith (v : T.View) (L : T.Hid → D) : T.View :=
  { v with
    leaves0 := (v.leaves0.1, fun q => L (.inl q))
    leavesP := fun r => ((v.leavesP r).1, fun k => L (.inr (r, k))) }

variable {T}

theorem Hm96.derived_close (M : T.Hm96 C) {δ₁ : ℝ≥0∞} (hT1 : Claims.Hm96Hiding M.My M.cs δ₁) (hrank : T.Rank)
    (hpad : T.PadOnto) (hW : ∀ r, Function.Bijective fun p : F × F => T.Wr r p.1 p.2) (hpads : T.PadsOnto)
    (hβ : ∀ r, T.β r ≠ 0) (w w₀ : W) (E : T.View → Prop) :
    prCoin (fun p : T.Rand × (T.Hid → Salt) =>
        E (T.hidWith (T.psi (T.realCore w p.1)) fun i => M.H (M.dig w p.1 i + M.My (p.2 i), M.cs (p.2 i)))) ≤
      prCoin (fun σ : T.SimRand × (T.Hid → Salt) => E (M.simR w₀ σ.1 σ.2)) + 2 * Fintype.card T.Hid * δ₁ := by
  have : Nonempty Salt := ⟨0⟩
  have : Nonempty T.Rand := ⟨0⟩
  have : Nonempty T.SimRand := ⟨0⟩
  have h1 := ideal_leaves_swap_fintype M.My M.cs hT1 M.H (M.dig w)
    fun ω L => E (T.hidWith (T.psi (T.realCore w ω)) L)
  have hm := (T.derived_sim hrank hpad hW hpads hβ w w₀).prCoin_prod_eq
    fun v (u : T.Hid → B × Salt) => E (T.hidWith v fun i => M.H ((u i).1, M.cs (u i).2))
  have h2 := real_leaves_swap_fintype M.My M.cs hT1 M.H (M.digS w₀)
    fun σ L => E (T.hidWith (T.sim w₀ σ) L)
  calc _ ≤ _ := h1
    _ = _ := congrArg (· + _) hm
    _ ≤ (prCoin (fun σ : T.SimRand × (T.Hid → Salt) => E (M.simR w₀ σ.1 σ.2)) + Fintype.card T.Hid * δ₁) +
          Fintype.card T.Hid * δ₁ := add_le_add h2 le_rfl
    _ = _ := by rw [add_assoc, ← two_mul, ← mul_assoc]

omit [Fintype K] [DecidableEq K] in
theorem Hm96.preR_close (M : T.Hm96 C) {digT : F → B} (hT : M.TauLeaf digT) {δ₁ : ℝ≥0∞}
    (hT1 : Claims.Hm96Hiding M.My M.cs δ₁) (hrank : T.Rank)
    (hW : ∀ r, Function.Bijective fun p : F × F => T.Wr r p.1 p.2) (hβ : ∀ r, T.β r ≠ 0) (w w₀ : W)
    (hreg : T.sReg w = T.pubReg) (hreg₀ : T.sReg w₀ = T.pubReg) (E : T.PreView → Prop) :
    prCoin (fun p : T.Rand × (T.Hid → Salt) => E (M.preR w p.1 p.2)) ≤
      prCoin (fun p : T.Rand × (T.Hid → Salt) => E (M.preR w₀ p.1 p.2)) + 2 * Fintype.card T.PreLeaves * δ₁ := by
  have : Nonempty Salt := ⟨0⟩
  have : Nonempty T.Rand := ⟨0⟩
  have hr : ∀ w', prCoin (fun p : T.Rand × (T.Hid → Salt) => E (M.preR w' p.1 p.2)) =
      prCoin (fun p : T.Rand × (T.PreLeaves → Salt) =>
        E (T.preLeaves w' p.1 fun i => M.H (M.digAll digT w' p.1 i + M.My (p.2 i), M.cs (p.2 i)))) := fun w' =>
    (congrArg prCoin (funext fun p : T.Rand × (T.Hid → Salt) =>
      congrArg E (T.toPre_viewR M digT hT w' p.1 p.2))).trans (T.prCoin_salts M digT w' E)
  have hi : ∀ w', prCoin (fun p : T.Rand × (T.PreLeaves → B × Salt) =>
        E (T.preLeaves w' p.1 fun i => M.H ((p.2 i).1, M.cs (p.2 i).2))) =
      prCoin (fun ω => E (T.preView w' ω)) := fun w' =>
    (congrArg prCoin (funext fun p : T.Rand × (T.PreLeaves → B × Salt) =>
      congrArg (fun L => E (T.preLeaves w' p.1 L))
        (funext fun i => (M.ideal_eq (p.2 i).1 (p.2 i).2).symm))).trans (T.prCoin_preLeaves_ideal w' E)
  have h1 := ideal_leaves_swap_fintype (ι := T.PreLeaves) M.My M.cs hT1 M.H (M.digAll digT w)
    fun ω L => E (T.preLeaves w ω L)
  have h2 := real_leaves_swap_fintype (ι := T.PreLeaves) M.My M.cs hT1 M.H (M.digAll digT w₀)
    fun ω L => E (T.preLeaves w₀ ω L)
  have hA := (T.table_prefinal_indep hrank hW hβ w w₀ hreg hreg₀).prCoin_eq E
  calc prCoin (fun p : T.Rand × (T.Hid → Salt) => E (M.preR w p.1 p.2))
      = _ := hr w
    _ ≤ _ := h1
    _ = prCoin (fun ω => E (T.preView w₀ ω)) + Fintype.card T.PreLeaves * δ₁ :=
        congrArg (· + Fintype.card T.PreLeaves * δ₁) ((hi w).trans hA)
    _ = prCoin (fun p : T.Rand × (T.PreLeaves → B × Salt) =>
          E (T.preLeaves w₀ p.1 fun i => M.H ((p.2 i).1, M.cs (p.2 i).2))) + Fintype.card T.PreLeaves * δ₁ :=
        congrArg (· + Fintype.card T.PreLeaves * δ₁) (hi w₀).symm
    _ ≤ (prCoin (fun p : T.Rand × (T.PreLeaves → Salt) =>
          E (T.preLeaves w₀ p.1 fun i => M.H (M.digAll digT w₀ p.1 i + M.My (p.2 i), M.cs (p.2 i)))) +
          Fintype.card T.PreLeaves * δ₁) + Fintype.card T.PreLeaves * δ₁ := add_le_add h2 le_rfl
    _ = _ := by rw [← hr w₀, add_assoc, ← two_mul, ← mul_assoc]

omit [Fintype F] [DecidableEq F] [Fintype K] [DecidableEq K] [Fintype Salt] [DecidableEq Salt] [Fintype B]
  [DecidableEq B] [Fintype Hi] [DecidableEq Hi] in
theorem Hm96.viewR_derived (M : T.Hm96 C) (hW : ∀ r, Function.Bijective fun p : F × F => T.Wr r p.1 p.2)
    (hβ : ∀ r, T.β r ≠ 0) (w : W) (hreg : T.sReg w = T.pubReg) (ω : T.Rand) (hω : T.HoldsAt w ω)
    (ys : T.Hid → Salt) :
    M.viewR w ω ys = T.hidWith (T.psi (T.realCore w ω)) fun i => M.H (M.dig w ω i + M.My (ys i), M.cs (ys i)) :=
  congrArg (fun v => T.hidWith v fun i => M.H (M.dig w ω i + M.My (ys i), M.cs (ys i)))
    (T.psi_realCore_at hW hβ w hreg ω hω).symm

theorem Hm96.release_hm96 (M : T.Hm96 C) {digT : F → B} (hT : M.TauLeaf digT) {δ₁ : ℝ≥0∞}
    (hT1 : Claims.Hm96Hiding M.My M.cs δ₁) (hrank : T.Rank) (hpad : T.PadOnto)
    (hW : ∀ r, Function.Bijective fun p : F × F => T.Wr r p.1 p.2) (hpads : T.PadsOnto) (hβ : ∀ r, T.β r ≠ 0)
    (go : T.Rand → Bool) (w w₀ : W) (hreg : T.sReg w = T.pubReg) (hreg₀ : T.sReg w₀ = T.pubReg)
    (hgo : T.Gated go w) :
    PerOutcome (fun p : T.Rand × (T.Hid → Salt) => go p.1) (fun p => M.releaseR go w p.1 p.2)
      (gatedSim (fun σ : T.SimRand × (T.Hid → Salt) => M.simR w₀ σ.1 σ.2)
        fun p : T.Rand × (T.Hid → Salt) => M.preR w₀ p.1 p.2)
      fun b => if b then 2 * Fintype.card T.Hid * δ₁ else 2 * Fintype.card T.PreLeaves * δ₁ := by
  have : Nonempty Salt := ⟨0⟩
  have : Nonempty T.Rand := ⟨0⟩
  have : Nonempty T.SimRand := ⟨0⟩
  exact gate_perOutcome (view := fun p : T.Rand × (T.Hid → Salt) => M.viewR w p.1 p.2)
    (d := fun p => T.hidWith (T.psi (T.realCore w p.1)) fun i => M.H (M.dig w p.1 i + M.My (p.2 i), M.cs (p.2 i)))
    (cut := fun p => M.preR w p.1 p.2)
    (fun E' => M.preR_close hT hT1 hrank hW hβ w w₀ hreg hreg₀ E')
    (fun E' => M.derived_close hT1 hrank hpad hW hpads hβ w w₀ E')
    (fun p hp => M.viewR_derived hW hβ w hreg p.1 (hgo p.1 hp) p.2)

end Table

namespace Session

variable {F K W Salt B D Hi VL : Type} [Field F] [Field K] [CharP F 2] [Fintype F] [DecidableEq F] [Fintype K]
  [DecidableEq K] [Fintype Salt] [DecidableEq Salt] [Fintype B] [DecidableEq B] [Fintype Hi] [DecidableEq Hi]
  [AddCommGroup Salt] [AddCommGroup B] [AddCommGroup Hi] {J : ℕ}

noncomputable def branchR (T : Fin J → Table F K W Salt (B × Salt) D Hi VL) {C : Type} (M : ∀ j, (T j).Hm96 C)
    {St : Type} (full : St → Fin J → Bool) (w : Fin J → W) (s : St) (j : Fin J)
    (x : (T j).Rand × ((T j).Hid → Salt)) : (T j).View ⊕ (T j).PreView :=
  if full s j then
    .inl ((T j).hidWith ((T j).psi ((T j).realCore (w j) x.1)) fun i =>
      (M j).H ((M j).dig (w j) x.1 i + (M j).My (x.2 i), (M j).cs (x.2 i)))
  else .inr ((M j).preR (w j) x.1 x.2)

theorem branchR_le (T : Fin J → Table F K W Salt (B × Salt) D Hi VL) {C : Type} (M : ∀ j, (T j).Hm96 C)
    {digT : Fin J → F → B} (hT : ∀ j, (M j).TauLeaf (digT j)) {δ₁ : ℝ≥0∞}
    (hT1 : ∀ j, Claims.Hm96Hiding (M j).My (M j).cs δ₁) (hrank : ∀ j, (T j).Rank) (hpad : ∀ j, (T j).PadOnto)
    (hW : ∀ j r, Function.Bijective fun p : F × F => (T j).Wr r p.1 p.2) (hpads : ∀ j, (T j).PadsOnto)
    (hβ : ∀ j r, (T j).β r ≠ 0) {St : Type} (full : St → Fin J → Bool) (w w₀ : Fin J → W)
    (hreg : ∀ j, (T j).sReg (w j) = (T j).pubReg) (hreg₀ : ∀ j, (T j).sReg (w₀ j) = (T j).pubReg) (s : St)
    (j : Fin J) (E : (T j).View ⊕ (T j).PreView → Prop) :
    prCoin (fun x => E (branchR T M full w s j x)) ≤
      prCoin (fun σ : ((T j).SimRand × ((T j).Hid → Salt)) × ((T j).Rand × ((T j).Hid → Salt)) =>
        E (if full s j then .inl ((M j).simR (w₀ j) σ.1.1 σ.1.2) else .inr ((M j).preR (w₀ j) σ.2.1 σ.2.2))) +
      (if full s j then 2 * Fintype.card (T j).Hid * δ₁ else 2 * Fintype.card (T j).PreLeaves * δ₁) := by
  have : Nonempty Salt := ⟨0⟩
  have : Nonempty (T j).Rand := ⟨0⟩
  have : Nonempty (T j).SimRand := ⟨0⟩
  unfold branchR
  cases hf : full s j
  · show prCoin (fun x : (T j).Rand × ((T j).Hid → Salt) => E (.inr ((M j).preR (w j) x.1 x.2))) ≤
      prCoin (fun σ : ((T j).SimRand × ((T j).Hid → Salt)) × ((T j).Rand × ((T j).Hid → Salt)) =>
        E (.inr ((M j).preR (w₀ j) σ.2.1 σ.2.2))) + 2 * Fintype.card (T j).PreLeaves * δ₁
    rw [prCoin_snd (A := (T j).SimRand × ((T j).Hid → Salt))
      (fun x : (T j).Rand × ((T j).Hid → Salt) => E (.inr ((M j).preR (w₀ j) x.1 x.2)))]
    exact (M j).preR_close (hT j) (hT1 j) (hrank j) (hW j) (hβ j) (w j) (w₀ j) (hreg j) (hreg₀ j)
      fun v => E (.inr v)
  · show prCoin (fun x : (T j).Rand × ((T j).Hid → Salt) =>
        E (.inl ((T j).hidWith ((T j).psi ((T j).realCore (w j) x.1)) fun i =>
          (M j).H ((M j).dig (w j) x.1 i + (M j).My (x.2 i), (M j).cs (x.2 i))))) ≤
      prCoin (fun σ : ((T j).SimRand × ((T j).Hid → Salt)) × ((T j).Rand × ((T j).Hid → Salt)) =>
        E (.inl ((M j).simR (w₀ j) σ.1.1 σ.1.2))) + 2 * Fintype.card (T j).Hid * δ₁
    rw [prCoin_fst (E := (T j).Rand × ((T j).Hid → Salt))
      (fun σ : (T j).SimRand × ((T j).Hid → Salt) => E (.inl ((M j).simR (w₀ j) σ.1 σ.2)))]
    exact (M j).derived_close (hT1 j) (hrank j) (hpad j) (hW j) (hpads j) (hβ j) (w j) (w₀ j) fun v => E (.inl v)

theorem release_hm96 (T : Fin J → Table F K W Salt (B × Salt) D Hi VL) {C : Type} (M : ∀ j, (T j).Hm96 C)
    {digT : Fin J → F → B} (hT : ∀ j, (M j).TauLeaf (digT j)) {δ₁ : ℝ≥0∞}
    (hT1 : ∀ j, Claims.Hm96Hiding (M j).My (M j).cs δ₁) (hrank : ∀ j, (T j).Rank) (hpad : ∀ j, (T j).PadOnto)
    (hW : ∀ j r, Function.Bijective fun p : F × F => (T j).Wr r p.1 p.2) (hpads : ∀ j, (T j).PadsOnto)
    (hβ : ∀ j r, (T j).β r ≠ 0) {St : Type} (out : (∀ j, (T j).Rand) → St)
    (full : St → Fin J → Bool) (w w₀ : Fin J → W) (hreg : ∀ j, (T j).sReg (w j) = (T j).pubReg)
    (hreg₀ : ∀ j, (T j).sReg (w₀ j) = (T j).pubReg) (hgo : Release.Gated T out full w) :
    PerOutcome (fun p : ∀ j, (T j).Rand × ((T j).Hid → Salt) => out fun j => (p j).1)
      (fun p => releaseR T M full w (out fun j => (p j).1) p) (simR T M full w₀)
      fun s => 2 * unopened T full s * δ₁ := by
  have : Nonempty Salt := ⟨0⟩
  have : ∀ j, Nonempty ((T j).Rand × ((T j).Hid → Salt)) := fun _ => ⟨(0, fun _ => 0)⟩
  have : ∀ j, Nonempty (((T j).SimRand × ((T j).Hid → Salt)) × ((T j).Rand × ((T j).Hid → Salt))) :=
    fun _ => ⟨((0, fun _ => 0), (0, fun _ => 0))⟩
  let ε : St → Fin J → ℝ≥0∞ := fun s j =>
    if full s j then 2 * Fintype.card (T j).Hid * δ₁ else 2 * Fintype.card (T j).PreLeaves * δ₁
  have hε : ∀ s, ∑ j, ε s j = 2 * unopened T full s * δ₁ := fun s => by
    simp only [ε, unopened, Nat.cast_sum, Finset.mul_sum, Finset.sum_mul]
    refine Finset.sum_congr rfl fun j _ => ?_
    split_ifs <;> rfl
  have hrel : Splits (fun p : ∀ j, (T j).Rand × ((T j).Hid → Salt) => releaseR T M full w (out fun j => (p j).1) p)
      (fun p => out fun j => (p j).1) (fun s p => fun j => branchR T M full w s j (p j)) :=
    fun p => funext fun j => by
      simp only [releaseR, branchR]
      by_cases hf : full (out fun j => (p j).1) j = true
      · rw [ite_eq_left hf, ite_eq_left hf,
          (M j).viewR_derived (hW j) (hβ j) (w j) (hreg j) (p j).1 (hgo (fun j => (p j).1) j hf) (p j).2]
      · rw [ite_eq_right hf, ite_eq_right hf]
  refine perOutcome_of_split hrel fun s E' => ?_
  have h := prCoin_pi_close (Ω := fun j => (T j).Rand × ((T j).Hid → Salt))
    (Ω' := fun j => ((T j).SimRand × ((T j).Hid → Salt)) × ((T j).Rand × ((T j).Hid → Salt)))
    (V := fun j => (T j).View ⊕ (T j).PreView) (fun j => branchR T M full w s j)
    (fun j σ => if full s j then .inl ((M j).simR (w₀ j) σ.1.1 σ.1.2) else .inr ((M j).preR (w₀ j) σ.2.1 σ.2.2))
    (ε s) (fun j E'' => branchR_le T M hT hT1 hrank hpad hW hpads hβ full w w₀ hreg hreg₀ s j E'') E'
  rw [hε s] at h
  exact h

theorem release_hm96_first (T : Fin J → Table F K W Salt (B × Salt) D Hi VL) {C : Type} (M : ∀ j, (T j).Hm96 C)
    {digT : Fin J → F → B} (hT : ∀ j, (M j).TauLeaf (digT j)) {δ₁ : ℝ≥0∞}
    (hT1 : ∀ j, Claims.Hm96Hiding (M j).My (M j).cs δ₁) (hsim : ∀ j, (T j).Simulable) {St : Type}
    (out : (∀ j, (T j).Rand) → St) (full : St → Fin J → Bool)
    (wit : (∀ j, (T j).Rand × ((T j).Hid → Salt)) → Fin J → W) (w₀ : Fin J → W) (hfirst : WitnessesFirst wit)
    (hreg : ∀ p j, (T j).RegPublic (wit p j)) (hreg₀ : ∀ j, (T j).RegPublic (w₀ j))
    (hgo : ∀ p, Release.Gated T out full (wit p)) :
    PerOutcome (fun p : ∀ j, (T j).Rand × ((T j).Hid → Salt) => out fun j => (p j).1)
      (fun p => releaseR T M full (wit p) (out fun j => (p j).1) p) (simR T M full w₀)
      fun s => 2 * unopened T full s * δ₁ := by
  have : Nonempty Salt := ⟨0⟩
  let p₀ : ∀ j, (T j).Rand × ((T j).Hid → Salt) := fun _ => (0, fun _ => 0)
  obtain ⟨w, rfl⟩ : ∃ w, wit = fun _ => w := ⟨wit p₀, funext fun p => hfirst p p₀⟩
  exact release_hm96 T M hT hT1 (fun j => (hsim j).1) (fun j => (hsim j).2.1) (fun j => (hsim j).2.2.1)
    (fun j => (hsim j).2.2.2.1) (fun j => (hsim j).2.2.2.2) out full w w₀ (hreg p₀) hreg₀ (hgo p₀)

end Session

end FlockSoundness.ZK
