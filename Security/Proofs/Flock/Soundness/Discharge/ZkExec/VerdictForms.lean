import Proofs.Flock.Soundness.Discharge.ZkExec.VerdictInner
import Proofs.Flock.Soundness.Discharge.ZkExec.VerdictOpens
import Proofs.Flock.Soundness.Refine.Lincheck
import Proofs.Flock.Soundness.Refine.ArithK
import Proofs.Flock.Soundness.LigeritoDecode
import Proofs.Flock.Soundness.Discharge.ZkExec.PlaysDet

/-!
# The model's forms are the executable's

What `zk_inner_of_rep` reads its rows through.

* The model's forms over `F` (`LinF`) are the executable's `Lin` coefficient by coefficient (`rel_add`, `rel_scale`,
  `rel_masked`, `rel_combo`, ...), and its forms over `K` (`LinK`) are the executable's `Lin2`, one `Lin` per
  coordinate (`Rel2`), whose two coordinate rows are `limbRows`'.
* `plays_prefix`, `plays_bind_split`: a run of a game is the only one on a prefix of its messages and coins, so a run
  of `g >>= f` continues, past a known run of `g`, with a run of `f` at its result.
-/

namespace FlockSoundness.Discharge.ZkExec

open Flock Refine Exec Game Model DecodeZL Transcript

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

/-! ## Forms over `F` -/

section Forms

variable {np : ℕ}

theorem linRel_iff {ℓ : ZkSession.LinF 𝔽 np} {x : Zk.Lin} :
    LinRel ℓ.toLin x ↔ x.a.size = np ∧ ℓ.2 = fOf x.c ∧ ∀ i : Fin np, ℓ.1 i = fOf x.a[i.val]! := Iff.rfl

theorem rel_add {a b : ZkSession.LinF 𝔽 np} {x y : Zk.Lin} (ha : LinRel a.toLin x) (hb : LinRel b.toLin y) :
    LinRel (a + b).toLin (x.add y) := by
  obtain ⟨hxs, hxc, hxa⟩ := linRel_iff.1 ha
  obtain ⟨hys, hyc, hya⟩ := linRel_iff.1 hb
  refine ⟨by simp [Zk.Lin.add, hxs, hys], ?_, fun i => ?_⟩
  · show a.2 + b.2 = fOf (x.c + y.c)
    rw [fOf_add, hxc, hyc]
  · show a.1 i + b.1 i = fOf ((Array.range (min x.a.size y.a.size)).map fun i => x.a[i]! + y.a[i]!)[i.val]!
    rw [getElem!_range_map _ (by rw [hxs, hys, Nat.min_self]; exact i.isLt), fOf_add, hxa, hya]

theorem rel_scale {a : ZkSession.LinF 𝔽 np} {x : Zk.Lin} (ha : LinRel a.toLin x) (k : F128) :
    LinRel (fOf k • a).toLin (x.scale k) := by
  obtain ⟨hxs, hxc, hxa⟩ := linRel_iff.1 ha
  refine ⟨by simp [Zk.Lin.scale, hxs], ?_, fun i => ?_⟩
  · show fOf k * a.2 = fOf (x.c * k)
    rw [fOf_mul, ← hxc, mul_comm]
  · show fOf k * a.1 i = fOf (x.a.map (· * k))[i.val]!
    rw [map_get! _ _ (by rw [hxs]; exact i.isLt), fOf_mul, ← hxa, mul_comm]

theorem rel_addConst {a : ZkSession.LinF 𝔽 np} {x : Zk.Lin} (ha : LinRel a.toLin x) (k : F128) :
    LinRel (a + ZkSession.cF (fOf k)).toLin (x.addConst k) := by
  obtain ⟨hxs, hxc, hxa⟩ := linRel_iff.1 ha
  refine ⟨hxs, ?_, fun i => ?_⟩
  · show a.2 + fOf k = fOf (x.c + k)
    rw [fOf_add, hxc]
  · show a.1 i + 0 = fOf x.a[i.val]!
    rw [add_zero, hxa]

theorem rel_konst (c : F128) : LinRel (ZkSession.cF (fOf c) : ZkSession.LinF 𝔽 np).toLin (Zk.Lin.konst c np) := by
  refine ⟨by simp [Zk.Lin.konst], rfl, fun i => ?_⟩
  show (0 : 𝔽) = fOf (Array.replicate np (0 : F128))[i.val]!
  rw [repl_get np i.isLt, fOf_zero]

theorem rel_masked (v : F128) (i : Fin np) {idx : ℕ} (hi : i.val = idx) :
    LinRel (ZkSession.mskF i (fOf v)).toLin (Zk.Lin.masked v idx np) := by
  subst hi
  refine ⟨by simp [Zk.Lin.masked], by simp [ZkSession.mskF, ZkSession.cF, ZkSession.pF, ZkSession.LinF.toLin,
    Zk.Lin.masked], fun j => ?_⟩
  show (ZkSession.cF (fOf v) + ZkSession.pF i : ZkSession.LinF 𝔽 np).1 j =
    fOf ((Array.replicate np (0 : F128)).set! i.val 1)[j.val]!
  rw [set_get _ _ (by simp) (by simp), repl_get np j.isLt]
  simp only [ZkSession.cF, ZkSession.pF, Prod.fst_add, Pi.add_apply, Pi.zero_apply, zero_add, Pi.single_apply,
    Fin.ext_iff]
  split_ifs <;> simp [fOf_one, fOf_zero]

/-- **`Lin.combo` is the model's weighted sum**, of forms on the pads. -/
theorem rel_combo {m : ℕ} (ws : Array F128) (ls : Array Zk.Lin) (f : Fin m → ZkSession.LinF 𝔽 np)
    (hw : ws.size = m) (hl : ls.size = m) (hf : ∀ k : Fin m, LinRel (f k).toLin ls[k.val]!) :
    LinRel (∑ k, fOf ws[k.val]! • f k).toLin (Zk.Lin.combo ws ls np) := by
  refine ⟨combo_size _ _ _, ?_, fun i => ?_⟩
  · show (∑ k, fOf ws[k.val]! • f k).2 = _
    rw [combo_c ws ls np m hw hl, Prod.snd_sum]
    refine Finset.sum_congr rfl fun k _ => ?_
    rw [Prod.smul_snd, smul_eq_mul, (linRel_iff.1 (hf k)).2.1]
  · show (∑ k, fOf ws[k.val]! • f k).1 i = _
    rw [combo_a ws ls np m hw hl (fun k hk => (hf ⟨k, hk⟩).1) i.val i.isLt, Prod.fst_sum, Finset.sum_apply]
    refine Finset.sum_congr rfl fun k _ => ?_
    rw [Prod.smul_fst, Pi.smul_apply, smul_eq_mul, (linRel_iff.1 (hf k)).2.2 i]

theorem add_size' {x y : Zk.Lin} (hx : x.a.size = np) (hy : y.a.size = np) : (x.add y).a.size = np := by
  simp [Zk.Lin.add, hx, hy]

theorem add_get {x y : Zk.Lin} (hx : x.a.size = np) (hy : y.a.size = np) {i : ℕ} (hi : i < np) :
    fOf (x.add y).a[i]! = fOf x.a[i]! + fOf y.a[i]! := by
  show fOf ((Array.range (min x.a.size y.a.size)).map fun i => x.a[i]! + y.a[i]!)[i]! = _
  rw [getElem!_range_map _ (by rw [hx, hy, Nat.min_self]; exact hi), fOf_add]

theorem scale_get {x : Zk.Lin} (hx : x.a.size = np) (k : F128) {i : ℕ} (hi : i < np) :
    fOf (x.scale k).a[i]! = fOf x.a[i]! * fOf k := by
  show fOf (x.a.map (· * k))[i]! = _
  rw [map_get! _ _ (by rw [hx]; exact hi), fOf_mul]

end Forms

/-! ## Forms over `K` -/

section FormsK

variable {np : ℕ}

/-- **A `K`-form is the executable's `Lin2`**, coordinate by coordinate: `x = x₀ + x₁·u` on `np` pads. -/
def Rel2 (ℓ : ZkSession.LinK 𝕂 np) (x : Zk.Lin2) : Prop :=
  x.1.a.size = np ∧ x.2.a.size = np ∧ ℓ.2 = execArith.ofLimbs (fOf x.1.c) (fOf x.2.c) ∧
    ∀ i : Fin np, ℓ.1 i = execArith.ofLimbs (fOf x.1.a[i.val]!) (fOf x.2.a[i.val]!)

theorem rel2_add {a b : ZkSession.LinK 𝕂 np} {x y : Zk.Lin2} (ha : Rel2 a x) (hb : Rel2 b y) :
    Rel2 (a + b) (Zk.l2add x y) := by
  obtain ⟨hx1, hx2, hxc, hxa⟩ := ha
  obtain ⟨hy1, hy2, hyc, hya⟩ := hb
  refine ⟨add_size' hx1 hy1, add_size' hx2 hy2, ?_, fun i => ?_⟩
  · show a.2 + b.2 = execArith.ofLimbs (fOf (x.1.c + y.1.c)) (fOf (x.2.c + y.2.c))
    rw [hxc, hyc, ofLimbs_add, fOf_add, fOf_add]
  · show a.1 i + b.1 i = execArith.ofLimbs (fOf (x.1.add y.1).a[i.val]!) (fOf (x.2.add y.2).a[i.val]!)
    rw [hxa, hya, ofLimbs_add, add_get hx1 hy1 i.isLt, add_get hx2 hy2 i.isLt]

theorem ofLimbs_eq (a b : 𝔽) : execArith.ofLimbs a b = FlockLevel3.embed a + FlockLevel3.embed b * FlockLevel3.u :=
  rfl

theorem ofLimbs_gOf_mul (a b : 𝔽) (k : F256) :
    gOf k * execArith.ofLimbs a b = execArith.ofLimbs (a * fOf k.c0 + b * fOf (k * F256.u).c0)
      (a * fOf k.c1 + b * fOf (k * F256.u).c1) := by
  have h1 := gOf_limbs k
  have h2 := gOf_limbs (k * F256.u)
  rw [gOf_mul, gOf_u] at h2
  rw [ofLimbs_eq, ofLimbs_eq, map_add, map_add, map_mul, map_mul, map_mul, map_mul]
  linear_combination (FlockLevel3.embed a) * h1 + (FlockLevel3.embed b) * h2

theorem rel2_scale {a : ZkSession.LinK 𝕂 np} {x : Zk.Lin2} (ha : Rel2 a x) (k : F256) :
    Rel2 (gOf k • a) (Zk.l2scale x k) := by
  obtain ⟨hx1, hx2, hxc, hxa⟩ := ha
  have hs : ∀ (z : Zk.Lin) (c : F128), z.a.size = np → (z.scale c).a.size = np := fun z c hz => by
    simp [Zk.Lin.scale, hz]
  refine ⟨add_size' (hs _ _ hx1) (hs _ _ hx2), add_size' (hs _ _ hx1) (hs _ _ hx2), ?_, fun i => ?_⟩
  · show gOf k * a.2 = _
    rw [hxc, ofLimbs_gOf_mul]
    simp only [Zk.l2scale, Zk.Lin.add, Zk.Lin.scale, fOf_add, fOf_mul]
  · show gOf k * a.1 i = _
    rw [hxa, ofLimbs_gOf_mul]
    simp only [Zk.l2scale]
    rw [add_get (hs _ _ hx1) (hs _ _ hx2) i.isLt, add_get (hs _ _ hx1) (hs _ _ hx2) i.isLt,
      scale_get hx1 _ i.isLt, scale_get hx2 _ i.isLt, scale_get hx1 _ i.isLt, scale_get hx2 _ i.isLt]

theorem rel2_konst (v : F256) : Rel2 (ZkSession.cK (gOf v) : ZkSession.LinK 𝕂 np) (Zk.l2konst v np) := by
  refine ⟨by simp [Zk.l2konst, Zk.Lin.konst], by simp [Zk.l2konst, Zk.Lin.konst], gOf_limbs v, fun i => ?_⟩
  show (0 : 𝕂) = execArith.ofLimbs (fOf (Array.replicate np (0 : F128))[i.val]!)
    (fOf (Array.replicate np (0 : F128))[i.val]!)
  rw [repl_get np i.isLt, fOf_zero, ofLimbs_zero]

theorem rel2_masked (v : F256) (i₀ i₁ : Fin np) {x₀ x₁ : ℕ} (h₀ : i₀.val = x₀) (h₁ : i₁.val = x₁) (hne : x₀ ≠ x₁) :
    Rel2 (ZkSession.cK (gOf v) + ZkSession.padK execArith i₀ i₁) (Zk.l2masked v x₀ x₁ np) := by
  subst h₀ h₁
  refine ⟨by simp [Zk.l2masked, Zk.Lin.masked], by simp [Zk.l2masked, Zk.Lin.masked], ?_, fun j => ?_⟩
  · show gOf v + 0 = _
    rw [add_zero, gOf_limbs]
    rfl
  · have e : (ZkSession.cK (gOf v) + ZkSession.padK execArith i₀ i₁ : ZkSession.LinK 𝕂 np).1 j =
        (if j.val = i₀.val then 1 else 0) + (if j.val = i₁.val then execArith.u else 0) := by
      simp [ZkSession.cK, ZkSession.padK, Pi.single_apply, ← Fin.val_inj]
    rw [e]
    show _ = execArith.ofLimbs (fOf ((Array.replicate np (0 : F128)).set! i₀.val 1)[j.val]!)
      (fOf ((Array.replicate np (0 : F128)).set! i₁.val 1)[j.val]!)
    rw [set_get _ _ (by simp) (by simp), set_get _ _ (by simp) (by simp), repl_get np j.isLt]
    by_cases ha : j.val = i₀.val
    · rw [ite_eq_left ha, ite_eq_left ha, ite_eq_right (show ¬ j.val = i₁.val by omega),
        ite_eq_right (show ¬ j.val = i₁.val by omega), fOf_one, fOf_zero, ofLimbs_embed, map_one, add_zero]
    · rw [ite_eq_right ha, ite_eq_right ha, fOf_zero]
      by_cases hb : j.val = i₁.val
      · rw [ite_eq_left hb, ite_eq_left hb, fOf_one, ofLimbs_eq, map_zero, map_one, zero_add, one_mul, zero_add]
        rfl
      · rw [ite_eq_right hb, ite_eq_right hb, fOf_zero, ofLimbs_zero, add_zero]

/-- An `F`-form read in `K`. -/
noncomputable def liftK (a : ZkSession.LinF 𝔽 np) : ZkSession.LinK 𝕂 np := (fun i => FlockLevel3.embed (a.1 i), FlockLevel3.embed a.2)

theorem rel2_lift {a : ZkSession.LinF 𝔽 np} {x : Zk.Lin} (ha : LinRel a.toLin x) :
    Rel2 (liftK a) (x, Zk.Lin.konst 0 np) := by
  obtain ⟨hxs, hxc, hxa⟩ := linRel_iff.1 ha
  refine ⟨hxs, by simp [Zk.Lin.konst], ?_, fun i => ?_⟩
  · show FlockLevel3.embed a.2 = execArith.ofLimbs (fOf x.c) (fOf 0)
    rw [hxc, fOf_zero, ofLimbs_embed]
    rfl
  · show FlockLevel3.embed (a.1 i) = execArith.ofLimbs (fOf x.a[i.val]!) (fOf (Array.replicate np (0 : F128))[i.val]!)
    rw [repl_get np i.isLt, hxa, fOf_zero, ofLimbs_embed]
    rfl

theorem coords_ofLimbs (a b : 𝔽) : ZkSession.coords execArith (execArith.ofLimbs a b) = (a, b) :=
  Function.leftInverse_invFun execArith_correct.ofLimbs_bijective.1 (a, b)

/-- **A `K`-row's two coordinate rows are the executable's two `Lin`s.** -/
theorem limbRows_rel {ℓ : ZkSession.LinK 𝕂 np} {x : Zk.Lin2} (h : Rel2 ℓ x) :
    LinRel (ZkSession.limbRows execArith ℓ 0) x.1 ∧ LinRel (ZkSession.limbRows execArith ℓ 1) x.2 := by
  obtain ⟨hx1, hx2, hxc, hxa⟩ := h
  refine ⟨⟨hx1, ?_, fun i => ?_⟩, ⟨hx2, ?_, fun i => ?_⟩⟩
  · show (ZkSession.coords execArith ℓ.2).1 = _
    rw [hxc, coords_ofLimbs]
  · show (ZkSession.coords execArith (ℓ.1 i)).1 = _
    rw [hxa, coords_ofLimbs]
  · show (ZkSession.coords execArith ℓ.2).2 = _
    rw [hxc, coords_ofLimbs]
  · show (ZkSession.coords execArith (ℓ.1 i)).2 = _
    rw [hxa, coords_ofLimbs]

end FormsK

/-! ## Runs on a prefix -/

section Prefix

variable {W F : Type} [Inhabited W] {ι : W → F} [Finite F] [Nontrivial F]

/-- **A run is the only one on a prefix**: two runs of `g` whose messages and coins are prefixes of the same lists are
on the same prefixes, to the same result. -/
theorem plays_prefix {α : Type} {g : Game α} {ms₁ ms₁' ms₂ ms₂' : List Msg} {cs₁ cs₁' cs₂ cs₂' : List (Array W)}
    {a a' : α} (h₁ : Plays ι g ms₁ cs₁ a) (h₂ : Plays ι g ms₁' cs₁' a') (hm : ms₁ ++ ms₂ = ms₁' ++ ms₂')
    (hc : cs₁ ++ cs₂ = cs₁' ++ cs₂') : ms₁ = ms₁' ∧ cs₁ = cs₁' ∧ a = a' := by
  induction h₁ generalizing ms₁' cs₁' a' with
  | ret a =>
      cases h₂
      exact ⟨rfl, rfl, rfl⟩
  | @send M k m ms cs a _ ih =>
      cases h₂ with
      | send m' h' =>
          simp only [List.cons_append, List.cons.injEq, Sigma.mk.injEq, heq_eq_eq, true_and] at hm
          obtain ⟨rfl, hm⟩ := hm
          obtain ⟨rfl, rfl, rfl⟩ := ih h' hm hc
          exact ⟨rfl, rfl, rfl⟩
  | @coinVec n iF iN k c hc' hn ms cs a _ ih =>
      generalize hg : @Game.coin α (Fin n → F) iF iN k = g at h₂
      cases h₂ with
      | ret => cases hg
      | send => cases hg
      | @coinVec n' iF' iN' k' c' _ _ _ _ _ h' =>
          injection hg with hC hiF hk
          obtain rfl := fin_fun_inj hC
          obtain rfl := eq_of_heq hiF
          obtain rfl := eq_of_heq hk
          simp only [List.cons_append, List.cons.injEq] at hc
          obtain ⟨rfl, hc⟩ := hc
          obtain ⟨rfl, rfl, rfl⟩ := ih h' hm hc
          exact ⟨rfl, rfl, rfl⟩
      | coinOne _ _ _ =>
          injection hg with hC
          exact absurd (fin_fun_ne hC.symm) hn
  | @coinOne iF iN k c hc' ms cs a _ ih =>
      generalize hg : @Game.coin α F iF iN k = g at h₂
      cases h₂ with
      | ret => cases hg
      | send => cases hg
      | coinVec _ _ hn' _ =>
          injection hg with hC
          exact absurd (fin_fun_ne hC) hn'
      | @coinOne iF' iN' k' c' _ _ _ _ h' =>
          injection hg with _ hiF hk
          subst hiF hk
          simp only [List.cons_append, List.cons.injEq] at hc
          obtain ⟨rfl, hc⟩ := hc
          obtain ⟨rfl, rfl, rfl⟩ := ih h' hm hc
          exact ⟨rfl, rfl, rfl⟩

/-- **A run of `g >>= f` past a run of `g`** is a run of `f` at its result. -/
theorem plays_bind_split {α β : Type} {g : Game α} {f : α → Game β} {ms₁ ms₂ : List Msg}
    {cs₁ cs₂ : List (Array W)} {a : α} {b : β} (h : Plays ι (g >>= f) (ms₁ ++ ms₂) (cs₁ ++ cs₂) b)
    (hg : Plays ι g ms₁ cs₁ a) : Plays ι (f a) ms₂ cs₂ b := by
  obtain ⟨ms₁', ms₂', cs₁', cs₂', a', hm, hc, h₁, h₂⟩ := plays_bind_inv g h
  obtain ⟨rfl, rfl, rfl⟩ := plays_prefix hg h₁ hm hc
  rw [List.append_cancel_left hm, List.append_cancel_left hc]
  exact h₂

end Prefix

end FlockSoundness.Discharge.ZkExec
