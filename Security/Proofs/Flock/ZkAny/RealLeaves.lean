module

public import Proofs.Flock.ZkAny.Session
public import Proofs.Flock.SampledGeneric.FlockHide
public import Definitions.Flock.ZK.RealRelease
public import Definitions.Flock.Guarantees.ZkAny

/-!
# The ZK server's release with real hm96 leaves, at any values

`ZkAny/Table.lean` and `ZkAny/Session.lean` bound the release behind the server's checks with every unopened leaf ideal.
Here every leaf is real, on both branches. Each branch costs `2·δ₁` a leaf it leaves unopened, through
`hm96_commit_ideal_close` (the commitment to any trace against the ideal one, `δ₁` a leaf, both ways), and nothing reads
the values:

* `prCoin_commit_ideal_close`: `hm96_commit_ideal_close` beside other randomness, at the test "the event holds".
* `SameDist.prCoin_prod_eq`: equal distributions stay equal beside an independent coin.
* `Hm96.derived_close`, where the check passes: the view is the derived view with real hidden leaves (`psi_realCore_at`).
  Its hidden leaves go ideal, at each fixed vector of ideal leaves the derived view is distributed as `S_shvzk(x, e)`
  (`derived_sim`), and the leaves go real again on `S_shvzk`'s side: `2·|Hid|·δ₁` at any witness.
* `Hm96.preR_close`, where it fails: the real pre-final messages read the prover's salts at the opened positions only in
  those leaves, so the salts move to a fresh vector over every leaf (`Hm96.prCoin_preR`); every leaf goes ideal, Lemma A
  (`table_prefinal_indep`) holds at each fixed vector of ideal leaves, and the leaves go real again at the dummy:
  `2·|PreLeaves|·δ₁` at any witness carrying the public region data.
* `Hm96.release_hm96`, `Flock.SecurityProofs.TableReleaseHm96`: one table, by `ZkAnyGate`.
* `Session.release_hm96`, `Flock.SecurityProofs.SessionReleaseHm96`: a session, by `prCoin_pi_close` and `ZkAnyOutcome`,
  at witnesses fixed before the server's draws.
-/

@[expose] public section

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace FlockSoundness.ZK

open Verity.Game
open scoped ENNReal

/-- **`hm96_commit_ideal_close` beside other randomness**: on `(ω, y)`, with the trace `X ω` and the event read from `ω`,
the commitment to `X ω` and the ideal commitment give the event probabilities within `N·δ₁` of each other, both ways:
`hm96_commit_ideal_close` at each `ω`, at the test "the event holds", averaged over `ω`. -/
theorem prCoin_commit_ideal_close {Ω ι Y B Cc D : Type} [Fintype Ω] [Nonempty Ω] [Fintype ι] [DecidableEq ι]
    [Fintype Y] [Nonempty Y] [AddCommGroup B] [Fintype B] (My : Y → B) (c : Y → Cc) {δ₁ : ℝ≥0∞}
    (hT1 : Claims.Hm96Hiding My c δ₁) (H : B × Cc → D) (X : Ω → ι → B) (G : Ω → (ι → D) → Prop) :
    prCoin (fun p : Ω × (ι → Y) => G p.1 fun i => H (X p.1 i + My (p.2 i), c (p.2 i))) ≤
        prCoin (fun p : Ω × (ι → B × Y) => G p.1 fun i => H ((p.2 i).1, c (p.2 i).2)) + Fintype.card ι * δ₁ ∧
      prCoin (fun p : Ω × (ι → B × Y) => G p.1 fun i => H ((p.2 i).1, c (p.2 i).2)) ≤
        prCoin (fun p : Ω × (ι → Y) => G p.1 fun i => H (X p.1 i + My (p.2 i), c (p.2 i))) +
          Fintype.card ι * δ₁ := by
  classical
  have h := fun ω => SampledGeneric.hm96_commit_ideal_close My c hT1 H (X ω)
    (fun L => if G ω L then (1 : ℝ≥0∞) else 0) (fun L => by split_ifs <;> simp)
  have h1 : ∀ ω, prCoin (fun y : ι → Y => G ω fun i => H (X ω i + My (y i), c (y i))) ≤
      prCoin (fun u : ι → B × Y => G ω fun i => H ((u i).1, c (u i).2)) + Fintype.card ι * δ₁ := fun ω => (h ω).1
  have h2 : ∀ ω, prCoin (fun u : ι → B × Y => G ω fun i => H ((u i).1, c (u i).2)) ≤
      prCoin (fun y : ι → Y => G ω fun i => H (X ω i + My (y i), c (y i))) + Fintype.card ι * δ₁ := fun ω => (h ω).2
  rw [prCoin_prod_split (fun ω (y : ι → Y) => G ω fun i => H (X ω i + My (y i), c (y i))),
    prCoin_prod_split (fun ω (u : ι → B × Y) => G ω fun i => H ((u i).1, c (u i).2))]
  constructor
  · calc avg (fun ω => prCoin fun y : ι → Y => G ω fun i => H (X ω i + My (y i), c (y i)))
        ≤ avg (fun ω => prCoin (fun u : ι → B × Y => G ω fun i => H ((u i).1, c (u i).2)) +
            Fintype.card ι * δ₁) := avg_mono h1
      _ = _ := by rw [avg_add, avg_const]
  · calc avg (fun ω => prCoin fun u : ι → B × Y => G ω fun i => H ((u i).1, c (u i).2))
        ≤ avg (fun ω => prCoin (fun y : ι → Y => G ω fun i => H (X ω i + My (y i), c (y i))) +
            Fintype.card ι * δ₁) := avg_mono h2
      _ = _ := by rw [avg_add, avg_const]

/-- **Equal distributions beside an independent coin**: an event of `f ω` and an independent uniform `u` has the same
probability as the event of `g σ` and `u`. -/
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

/-- A view with its hidden leaves `L`: `viewWith` and `simWith` are it at the prover's view and at `S_shvzk`'s. -/
def hidWith (v : T.View) (L : T.Hid → D) : T.View :=
  { v with
    leaves0 := (v.leaves0.1, fun q => L (.inl q))
    leavesP := fun r => ((v.leavesP r).1, fun k => L (.inr (r, k))) }

/-- A pre-final view with every leaf `L`. -/
def preWith (v : T.PreView) (L : T.PreLeaves → D) : T.PreView :=
  { v with
    leaves0 := (fun q => L (.inl (.inl q)), fun q => L (.inr (.inl q)))
    tauCom := fun r => L (.inl (.inr (.inl r)))
    leavesP := fun r => (fun k => L (.inl (.inr (.inr (r, k)))), fun k => L (.inr (.inr (r, k)))) }

/-- The opened leaves' salts in the prover's randomness: level 0's, `y_τ`, and the pads trees'. -/
def openOf (s : T.Seeds) : T.Opened → Salt :=
  Sum.elim s.1 (Sum.elim s.2.2.2.2.2.2.1 fun rk => s.2.2.2.1 rk.1 rk.2)

/-- Replace the opened leaves' salts. -/
def setOpen (s : T.Seeds) (v : T.Opened → Salt) : T.Seeds :=
  (fun q => v (.inl q), s.2.1, s.2.2.1, fun r k => v (.inr (.inr (r, k))), s.2.2.2.2.1, s.2.2.2.2.2.1,
    fun r => v (.inr (.inl r)), s.2.2.2.2.2.2.2)

omit [CharP F 2] [Fintype F] [DecidableEq F] [Fintype K] [DecidableEq K] [Fintype Salt] [DecidableEq Salt] [Fintype B]
  [DecidableEq B] [Fintype Hi] [DecidableEq Hi] [AddCommGroup Salt] [AddCommGroup B] [AddCommGroup Hi] in
theorem openOf_setOpen (s : T.Seeds) (v : T.Opened → Salt) : T.openOf (T.setOpen s v) = v := by
  funext i
  rcases i with q | r | ⟨r, k⟩ <;> rfl

/-- The prover's randomness with the opened leaves' salts replaced. -/
def withOpen (ω : T.Rand) (v : T.Opened → Salt) : T.Rand :=
  (ω.1, ω.2.1, ω.2.2.1, ω.2.2.2.1, ω.2.2.2.2.1, ω.2.2.2.2.2.1, T.setOpen ω.2.2.2.2.2.2 v)

variable {T}

/-- The digests the pre-final leaves commit, on witness `w` and randomness `ω`: level 0's columns at every position,
`τ`'s digest, and each rep's pads columns at every position. -/
noncomputable def Hm96.preDig (M : T.Hm96 C) (digT : F → B) (w : W) (ω : T.Rand) : T.PreLeaves → B
  | .inl (.inl q) => M.dig0 ((T.view w ω).rows q, (T.view w ω).xrows q)
  | .inl (.inr (.inl r)) => digT ((T.view w ω).tau r)
  | .inl (.inr (.inr (r, k))) => M.digP ((T.view w ω).ch r k, (T.view w ω).cm r k)
  | .inr i => M.dig w ω i

/-- **The derived view with real hidden leaves is within `2·|Hid|·δ₁` of `S_shvzk(x, e)` with real ones**, at any
witness: its hidden leaves go ideal, the derived view is distributed as `S_shvzk(x, e)` at each fixed vector of ideal
leaves (`derived_sim`), and the leaves go real again on `S_shvzk`'s side. -/
theorem Hm96.derived_close (M : T.Hm96 C) {δ₁ : ℝ≥0∞} (hT1 : Claims.Hm96Hiding M.My M.cs δ₁) (hrank : T.Rank)
    (hpad : T.PadOnto) (hW : ∀ r, Function.Bijective fun p : F × F => T.Wr r p.1 p.2) (hpads : T.PadsOnto)
    (hβ : ∀ r, T.β r ≠ 0) (w w₀ : W) (E : T.View → Prop) :
    prCoin (fun p : T.Rand × (T.Hid → Salt) =>
        E (T.hidWith (T.psi (T.realCore w p.1)) fun i => M.H (M.dig w p.1 i + M.My (p.2 i), M.cs (p.2 i)))) ≤
      prCoin (fun σ : T.SimRand × (T.Hid → Salt) => E (M.simR w₀ σ.1 σ.2)) + 2 * Fintype.card T.Hid * δ₁ := by
  have : Nonempty Salt := ⟨0⟩
  have : Nonempty T.Rand := ⟨0⟩
  have : Nonempty T.SimRand := ⟨0⟩
  have h1 := (prCoin_commit_ideal_close M.My M.cs hT1 M.H (M.dig w)
    fun ω L => E (T.hidWith (T.psi (T.realCore w ω)) L)).1
  have hm := (T.derived_sim hrank hpad hW hpads hβ w w₀).prCoin_prod_eq
    fun v (u : T.Hid → B × Salt) => E (T.hidWith v fun i => M.H ((u i).1, M.cs (u i).2))
  have h2 := (prCoin_commit_ideal_close M.My M.cs hT1 M.H (M.digS w₀)
    fun σ L => E (T.hidWith (T.sim w₀ σ) L)).2
  calc _ ≤ _ := h1
    _ = _ := congrArg (· + _) hm
    _ ≤ (prCoin (fun σ : T.SimRand × (T.Hid → Salt) => E (M.simR w₀ σ.1 σ.2)) + Fintype.card T.Hid * δ₁) +
          Fintype.card T.Hid * δ₁ := add_le_add h2 le_rfl
    _ = _ := by rw [add_assoc, ← two_mul, ← mul_assoc]

omit [Fintype F] [DecidableEq F] [Fintype K] [DecidableEq K] [Fintype Salt] [DecidableEq Salt] [Fintype B]
  [DecidableEq B] [Fintype Hi] [DecidableEq Hi] [AddCommGroup Salt] [AddCommGroup Hi] in
/-- **The real pre-final messages, read on the salts the prover's randomness doesn't hold**: at any replacement `v` of
the opened leaves' salts, they are the pre-final view with every leaf an hm96 leaf of its digest, under the prover's
own opened salts and the fresh ones. -/
theorem Hm96.preR_eq (M : T.Hm96 C) {digT : F → B} (hT : M.TauLeaf digT) (w : W) (ω : T.Rand)
    (v : T.Opened → Salt) (ys : T.Hid → Salt) :
    M.preR w ω ys = T.preWith (T.preView w (T.withOpen ω v)) fun i =>
      M.H (M.preDig digT w (T.withOpen ω v) i + M.My (Sum.elim (T.openOf ω.2.2.2.2.2.2) ys i),
        M.cs (Sum.elim (T.openOf ω.2.2.2.2.2.2) ys i)) := by
  refine PreView.ext rfl rfl rfl rfl rfl rfl rfl rfl rfl ?_ ?_ ?_
  · exact Prod.ext (funext fun q => M.leaf0_eq _ _) rfl
  · exact funext fun r => hT _ _
  · exact funext fun r => Prod.ext (funext fun k => M.leafP_eq _ _) rfl

omit [DecidableEq F] [Fintype K] [DecidableEq K] [DecidableEq Salt] [DecidableEq B] [DecidableEq Hi] [AddCommGroup Hi] in
/-- **The opened leaves' salts move to a fresh vector**: the real pre-final messages read the prover's salts at the
opened positions only in those leaves, so every event has the probability it has with every leaf under its own fresh
salt. -/
theorem Hm96.prCoin_preR (M : T.Hm96 C) {digT : F → B} (hT : M.TauLeaf digT) (w : W) (E : T.PreView → Prop) :
    prCoin (fun p : T.Rand × (T.Hid → Salt) => E (M.preR w p.1 p.2)) =
      prCoin (fun p : T.Rand × (T.PreLeaves → Salt) =>
        E (T.preWith (T.preView w p.1) fun i => M.H (M.preDig digT w p.1 i + M.My (p.2 i), M.cs (p.2 i)))) := by
  have : Nonempty Salt := ⟨0⟩
  let Φ : (T.Rand × (T.Hid → Salt)) × (T.Opened → Salt) ≃ T.Rand × (T.PreLeaves → Salt) :=
    { toFun := fun x => (T.withOpen x.1.1 x.2, Sum.elim (T.openOf x.1.1.2.2.2.2.2.2) x.1.2)
      invFun := fun p => ((T.withOpen p.1 (p.2 ∘ Sum.inl), p.2 ∘ Sum.inr), T.openOf p.1.2.2.2.2.2.2)
      left_inv := fun x => Prod.ext (Prod.ext rfl rfl) (T.openOf_setOpen _ _)
      right_inv := fun p => Prod.ext rfl
        ((congrArg (Sum.elim · (p.2 ∘ Sum.inr)) (T.openOf_setOpen _ _)).trans (Sum.elim_comp_inl_inr p.2)) }
  calc prCoin (fun p : T.Rand × (T.Hid → Salt) => E (M.preR w p.1 p.2))
      = prCoin (fun x : (T.Rand × (T.Hid → Salt)) × (T.Opened → Salt) => E (M.preR w x.1.1 x.1.2)) :=
        (prCoin_fst (E := T.Opened → Salt) fun p : T.Rand × (T.Hid → Salt) => E (M.preR w p.1 p.2)).symm
    _ = prCoin (fun x => (fun p : T.Rand × (T.PreLeaves → Salt) =>
          E (T.preWith (T.preView w p.1) fun i => M.H (M.preDig digT w p.1 i + M.My (p.2 i), M.cs (p.2 i)))) (Φ x)) :=
        congrArg prCoin (funext fun x => congrArg E (M.preR_eq hT w x.1.1 x.2 x.1.2))
    _ = _ := prCoin_congr_equiv Φ fun p : T.Rand × (T.PreLeaves → Salt) =>
          E (T.preWith (T.preView w p.1) fun i => M.H (M.preDig digT w p.1 i + M.My (p.2 i), M.cs (p.2 i)))

omit [Fintype K] [DecidableEq K] in
/-- **The real pre-final messages are within `2·|PreLeaves|·δ₁` of the dummy's**, at any witness carrying the public
region data: the opened leaves' salts move to a fresh vector (`prCoin_preR`), every leaf goes ideal, Lemma A holds at
each fixed vector of ideal leaves (`table_prefinal_indep`), and the leaves go real again at the dummy. -/
theorem Hm96.preR_close (M : T.Hm96 C) {digT : F → B} (hT : M.TauLeaf digT) {δ₁ : ℝ≥0∞}
    (hT1 : Claims.Hm96Hiding M.My M.cs δ₁) (hrank : T.Rank)
    (hW : ∀ r, Function.Bijective fun p : F × F => T.Wr r p.1 p.2) (hβ : ∀ r, T.β r ≠ 0) (w w₀ : W)
    (hreg : T.sReg w = T.pubReg) (hreg₀ : T.sReg w₀ = T.pubReg) (E : T.PreView → Prop) :
    prCoin (fun p : T.Rand × (T.Hid → Salt) => E (M.preR w p.1 p.2)) ≤
      prCoin (fun p : T.Rand × (T.Hid → Salt) => E (M.preR w₀ p.1 p.2)) + 2 * Fintype.card T.PreLeaves * δ₁ := by
  have : Nonempty Salt := ⟨0⟩
  have : Nonempty T.Rand := ⟨0⟩
  have h1 := (prCoin_commit_ideal_close M.My M.cs hT1 M.H (M.preDig digT w)
    fun ω L => E (T.preWith (T.preView w ω) L)).1
  have hm := (T.table_prefinal_indep hrank hW hβ w w₀ hreg hreg₀).prCoin_prod_eq
    fun v (u : T.PreLeaves → B × Salt) => E (T.preWith v fun i => M.H ((u i).1, M.cs (u i).2))
  have h2 := (prCoin_commit_ideal_close M.My M.cs hT1 M.H (M.preDig digT w₀)
    fun ω L => E (T.preWith (T.preView w₀ ω) L)).2
  rw [M.prCoin_preR hT w E, M.prCoin_preR hT w₀ E]
  calc _ ≤ _ := h1
    _ = _ := congrArg (· + _) hm
    _ ≤ _ := add_le_add h2 le_rfl
    _ = _ := by rw [add_assoc, ← two_mul, ← mul_assoc]

omit [Fintype F] [DecidableEq F] [Fintype K] [DecidableEq K] [Fintype Salt] [DecidableEq Salt] [Fintype B]
  [DecidableEq B] [Fintype Hi] [DecidableEq Hi] in
/-- **Where the check holds, the real view is the derived view with real hidden leaves** (`psi_realCore_at`). -/
theorem Hm96.viewR_derived (M : T.Hm96 C) (hW : ∀ r, Function.Bijective fun p : F × F => T.Wr r p.1 p.2)
    (hβ : ∀ r, T.β r ≠ 0) (w : W) (hreg : T.sReg w = T.pubReg) (ω : T.Rand) (hω : T.HoldsAt w ω)
    (ys : T.Hid → Salt) :
    M.viewR w ω ys = T.hidWith (T.psi (T.realCore w ω)) fun i => M.H (M.dig w ω i + M.My (ys i), M.cs (ys i)) :=
  congrArg (fun v => T.hidWith v fun i => M.H (M.dig w ω i + M.My (ys i), M.cs (ys i)))
    (T.psi_realCore_at hW hβ w hreg ω hω).symm

/-- **One table's release behind the check, at any values, with real leaves**: each branch within `2·δ₁` a leaf it
leaves unopened of its simulator with real leaves, and the release within the sum of both. -/
theorem Hm96.release_hm96 (M : T.Hm96 C) {digT : F → B} (hT : M.TauLeaf digT) {δ₁ : ℝ≥0∞}
    (hT1 : Claims.Hm96Hiding M.My M.cs δ₁) (hrank : T.Rank) (hpad : T.PadOnto)
    (hW : ∀ r, Function.Bijective fun p : F × F => T.Wr r p.1 p.2) (hpads : T.PadsOnto) (hβ : ∀ r, T.β r ≠ 0)
    (go : T.Rand → Bool) (w w₀ : W) (hreg : T.sReg w = T.pubReg) (hreg₀ : T.sReg w₀ = T.pubReg)
    (hgo : T.Gated go w) (E : T.View ⊕ T.PreView → Prop) :
    prCoin (fun p : T.Rand × (T.Hid → Salt) => go p.1 = true ∧ E (.inl (M.viewR w p.1 p.2))) ≤
        prCoin (fun σ : T.SimRand × (T.Hid → Salt) => E (.inl (M.simR w₀ σ.1 σ.2))) +
          2 * Fintype.card T.Hid * δ₁ ∧
      prCoin (fun p : T.Rand × (T.Hid → Salt) => go p.1 = false ∧ E (.inr (M.preR w p.1 p.2))) ≤
        prCoin (fun p : T.Rand × (T.Hid → Salt) => E (.inr (M.preR w₀ p.1 p.2))) +
          2 * Fintype.card T.PreLeaves * δ₁ ∧
      prCoin (fun p : T.Rand × (T.Hid → Salt) => E (M.releaseR go w p.1 p.2)) ≤
        prCoin (fun σ : T.SimRand × (T.Hid → Salt) => E (.inl (M.simR w₀ σ.1 σ.2))) +
          prCoin (fun p : T.Rand × (T.Hid → Salt) => E (.inr (M.preR w₀ p.1 p.2))) +
            (2 * Fintype.card T.Hid * δ₁ + 2 * Fintype.card T.PreLeaves * δ₁) :=
  Flock.SecurityProofs.ZkAnyGate (fun p : T.Rand × (T.Hid → Salt) => M.viewR w p.1 p.2)
    (fun p => T.hidWith (T.psi (T.realCore w p.1)) fun i => M.H (M.dig w p.1 i + M.My (p.2 i), M.cs (p.2 i)))
    (fun p => M.preR w p.1 p.2) (fun p => go p.1) (fun σ : T.SimRand × (T.Hid → Salt) => M.simR w₀ σ.1 σ.2)
    (fun p : T.Rand × (T.Hid → Salt) => M.preR w₀ p.1 p.2) _ _
    (fun E' => M.preR_close hT hT1 hrank hW hβ w w₀ hreg hreg₀ E')
    (fun E' => M.derived_close hT1 hrank hpad hW hpads hβ w w₀ E')
    (fun p hp => M.viewR_derived hW hβ w hreg p.1 (hgo p.1 hp) p.2) E

end Table

namespace Session

variable {F K W Salt B D Hi VL : Type} [Field F] [Field K] [CharP F 2] [Fintype F] [DecidableEq F] [Fintype K]
  [DecidableEq K] [Fintype Salt] [DecidableEq Salt] [Fintype B] [DecidableEq B] [Fintype Hi] [DecidableEq Hi]
  [AddCommGroup Salt] [AddCommGroup B] [AddCommGroup Hi] {J : ℕ}

/-- **Table `j`'s branch at the outcome `s`, with real leaves**: the derived view with real hidden leaves where its file
left, the real pre-final messages elsewhere. -/
noncomputable def branchR (T : Fin J → Table F K W Salt (B × Salt) D Hi VL) {C : Type} (M : ∀ j, (T j).Hm96 C)
    {St : Type} (full : St → Fin J → Bool) (w : Fin J → W) (s : St) (j : Fin J)
    (x : (T j).Rand × ((T j).Hid → Salt)) : (T j).View ⊕ (T j).PreView :=
  if full s j then
    .inl ((T j).hidWith ((T j).psi ((T j).realCore (w j) x.1)) fun i =>
      (M j).H ((M j).dig (w j) x.1 i + (M j).My (x.2 i), (M j).cs (x.2 i)))
  else .inr ((M j).preR (w j) x.1 x.2)

/-- **Each table's branch is within `2·δ₁` a leaf it leaves unopened of its simulator**: `Hm96.derived_close` where its
file left, `Hm96.preR_close` elsewhere. -/
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

/-- **A session's release behind the ZK server's checks, at any values, with real leaves**: at each outcome, jointly with
it, within `2·δ₁` a leaf it leaves unopened of the simulator at that outcome; overall within the sum over outcomes, and
within `|St|` times the uniform-outcome simulator plus the outcomes' errors. -/
theorem release_hm96 (T : Fin J → Table F K W Salt (B × Salt) D Hi VL) {C : Type} (M : ∀ j, (T j).Hm96 C)
    {digT : Fin J → F → B} (hT : ∀ j, (M j).TauLeaf (digT j)) {δ₁ : ℝ≥0∞}
    (hT1 : ∀ j, Claims.Hm96Hiding (M j).My (M j).cs δ₁) (hrank : ∀ j, (T j).Rank) (hpad : ∀ j, (T j).PadOnto)
    (hW : ∀ j r, Function.Bijective fun p : F × F => (T j).Wr r p.1 p.2) (hpads : ∀ j, (T j).PadsOnto)
    (hβ : ∀ j r, (T j).β r ≠ 0) {St : Type} [Fintype St] (out : (∀ j, (T j).Rand) → St)
    (full : St → Fin J → Bool) (w w₀ : Fin J → W) (hreg : ∀ j, (T j).sReg (w j) = (T j).pubReg)
    (hreg₀ : ∀ j, (T j).sReg (w₀ j) = (T j).pubReg) (hgo : Gated T out full w)
    (E : (∀ j, (T j).View ⊕ (T j).PreView) → Prop) :
    (∀ s, prCoin (fun p : ∀ j, (T j).Rand × ((T j).Hid → Salt) => out (fun j => (p j).1) = s ∧
        E (releaseR T M full w (out fun j => (p j).1) p)) ≤
      prCoin (fun σ => E (simR T M full w₀ s σ)) + 2 * unopened T full s * δ₁) ∧
    prCoin (fun p : ∀ j, (T j).Rand × ((T j).Hid → Salt) => E (releaseR T M full w (out fun j => (p j).1) p)) ≤
      ∑ s, (prCoin (fun σ => E (simR T M full w₀ s σ)) + 2 * unopened T full s * δ₁) ∧
    prCoin (fun p : ∀ j, (T j).Rand × ((T j).Hid → Salt) => E (releaseR T M full w (out fun j => (p j).1) p)) ≤
      (Fintype.card St : ℝ≥0∞) *
        prCoin (fun q : St × (∀ j, ((T j).SimRand × ((T j).Hid → Salt)) × ((T j).Rand × ((T j).Hid → Salt))) =>
          E (simR T M full w₀ q.1 q.2)) + ∑ s, 2 * unopened T full s * δ₁ := by
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
  have hrel : ∀ p : ∀ j, (T j).Rand × ((T j).Hid → Salt),
      releaseR T M full w (out fun j => (p j).1) p = fun j => branchR T M full w (out fun j => (p j).1) j (p j) :=
    fun p => funext fun j => by
      simp only [releaseR, branchR]
      by_cases hf : full (out fun j => (p j).1) j = true
      · rw [ite_eq_left hf, ite_eq_left hf,
          (M j).viewR_derived (hW j) (hβ j) (w j) (hreg j) (p j).1 (hgo (fun j => (p j).1) j hf) (p j).2]
      · rw [ite_eq_right hf, ite_eq_right hf]
  have hB : ∀ s (E' : (∀ j, (T j).View ⊕ (T j).PreView) → Prop),
      prCoin (fun p : ∀ j, (T j).Rand × ((T j).Hid → Salt) => E' fun j => branchR T M full w s j (p j)) ≤
        prCoin (fun σ => E' (simR T M full w₀ s σ)) + ∑ j, ε s j := fun s E' =>
    prCoin_pi_close (Ω := fun j => (T j).Rand × ((T j).Hid → Salt))
      (Ω' := fun j => ((T j).SimRand × ((T j).Hid → Salt)) × ((T j).Rand × ((T j).Hid → Salt)))
      (V := fun j => (T j).View ⊕ (T j).PreView) (fun j => branchR T M full w s j)
      (fun j σ => if full s j then .inl ((M j).simR (w₀ j) σ.1.1 σ.1.2) else .inr ((M j).preR (w₀ j) σ.2.1 σ.2.2))
      (ε s) (fun j E'' => branchR_le T M hT hT1 hrank hpad hW hpads hβ full w w₀ hreg hreg₀ s j E'') E'
  obtain ⟨hper, hsum, hcard⟩ := Flock.SecurityProofs.ZkAnyOutcome
    (fun p : ∀ j, (T j).Rand × ((T j).Hid → Salt) => releaseR T M full w (out fun j => (p j).1) p)
    (fun p => out fun j => (p j).1)
    (fun s (p : ∀ j, (T j).Rand × ((T j).Hid → Salt)) => fun j => branchR T M full w s j (p j))
    (fun s σ => simR T M full w₀ s σ) (fun s => ∑ j, ε s j) hrel hB E
  simp only [hε] at hper hsum hcard
  exact ⟨hper, hsum, hcard⟩

end Session

end FlockSoundness.ZK

namespace Flock.SecurityProofs

/-- **One table's release behind the ZK server's check, at any values, with real hm96 leaves**
(`Flock.Guarantees.TableReleaseHm96`). -/
theorem TableReleaseHm96 : Flock.Guarantees.TableReleaseHm96 := by
  intro F K W Salt B D Hi VL C _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ T M digT δ₁ go w w₀ hT1 hT hsim hreg hreg₀ hgo E
  obtain ⟨hrank, hpad, hW, hpads, hβ⟩ := hsim
  exact M.release_hm96 hT hT1 hrank hpad hW hpads hβ go w w₀ hreg hreg₀ hgo E

/-- **A session's release behind the ZK server's checks, at any values, with real hm96 leaves**
(`Flock.Guarantees.SessionReleaseHm96`): at witnesses fixed before the server's draws, `Session.release_hm96` at the one
witness they take. -/
theorem SessionReleaseHm96 : Flock.Guarantees.SessionReleaseHm96 := by
  intro F K W Salt B D Hi VL C _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ J T M digT δ₁ St _ out full wit w₀ hfirst hT1 hT hsim
    hreg hreg₀ hgo E
  have : Nonempty Salt := ⟨0⟩
  let p₀ : ∀ j, (T j).Rand × ((T j).Hid → Salt) := fun _ => (0, fun _ => 0)
  obtain ⟨w, rfl⟩ : ∃ w, wit = fun _ => w := ⟨wit p₀, funext fun p => hfirst p p₀⟩
  exact FlockSoundness.ZK.Session.release_hm96 T M hT hT1 (fun j => (hsim j).1) (fun j => (hsim j).2.1)
    (fun j => (hsim j).2.2.1) (fun j => (hsim j).2.2.2.1) (fun j => (hsim j).2.2.2.2) out full w w₀ (hreg p₀) hreg₀
    (hgo p₀) E

end Flock.SecurityProofs
