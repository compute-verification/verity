import Properties.Pous.EndToEnd
import Proofs.Pous.Sanity
import Proofs.Pous.Thm1.Theorem1iiSeq

namespace Pous.EndToEnd.SecurityProofs

open MemoryChallenger
open scoped ENNReal

theorem size_bytesOf {n : ℕ} (x : Bytes n) : (bytesOf x).size = n := by
  simp [bytesOf, ByteArray.size]

theorem get!_bytesOf {n : ℕ} (x : Bytes n) (i : ℕ) (hi : i < n) : ((bytesOf x).get! i).toNat = (x ⟨i, hi⟩).val := by
  simp only [bytesOf, ByteArray.get!]
  rw [getElem!_pos (Array.ofFn fun i => (x i).val.toUInt8) i (by simpa using hi)]
  simp [Array.getElem_ofFn]

theorem beNat_succ (b : ByteArray) (off len : ℕ) :
    Flock.Draw.beNat b off (len + 1) = Flock.Draw.beNat b off len * 256 + (b.get! (off + len)).toNat := by
  simp only [Flock.Draw.beNat, List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil]

theorem beNat_zero (b : ByteArray) (off : ℕ) : Flock.Draw.beNat b off 0 = 0 := rfl

theorem beNat_eq (x : Bytes 4096) :
    Flock.Draw.beNat (bytesOf x) 0 11 =
      ((((((((((x 0).val * 256 + (x 1).val) * 256 + (x 2).val) * 256 + (x 3).val) * 256 + (x 4).val) * 256 +
        (x 5).val) * 256 + (x 6).val) * 256 + (x 7).val) * 256 + (x 8).val) * 256 + (x 9).val) * 256 + (x 10).val := by
  simp only [beNat_succ, beNat_zero, Nat.zero_add, zero_mul]
  simp (disch := omega) only [get!_bytesOf]
  rfl

theorem width_b19 : Flock.Draw.width (2 ^ 19) = 11 := by
  unfold Flock.Draw.width; rw [Nat.log2_two_pow]

theorem uniformOn_eq (x : Bytes 4096) :
    FlockSoundness.Randomness.uniformOn (2 ^ 19) (bytesOf x) =
      some ((x 8).val % 8 * 65536 + (x 9).val * 256 + (x 10).val) := by
  have hsz : (bytesOf x).size = 4096 := size_bytesOf x
  unfold FlockSoundness.Randomness.uniformOn
  rw [hsz, Flock.Draw.uniform, width_b19, hsz, beNat_eq]
  have := (x 0).isLt; have := (x 1).isLt; have := (x 2).isLt; have := (x 3).isLt; have := (x 4).isLt
  have := (x 5).isLt; have := (x 6).isLt; have := (x 7).isLt; have := (x 8).isLt; have := (x 9).isLt
  have := (x 10).isLt
  have h88 : (2 : ℕ) ^ (8 * 11) - 2 ^ (8 * 11) % 2 ^ 19 = 309485009821345068724781056 := by norm_num
  simp only [h88]
  split_ifs <;> first | omega | simp only [Option.map_some, Option.some.injEq]; omega

/-- The bytes a draw leaves unread at `2^19`: all but bytes 8, 9 and 10, and the top five bits of byte 8. -/
abbrev Rest : Type :=
  Fin 32 × ({k : {j : {j : Fin 4096 // j ≠ 8} // j ≠ ⟨9, by decide⟩} // k ≠ ⟨⟨10, by decide⟩, by decide⟩} → Fin 256)

/-- Bytes 8, 9 and 10 as the index and the rest. -/
def mix : Fin 256 × Fin 256 × Fin 256 × (({k : {j : {j : Fin 4096 // j ≠ 8} // j ≠ ⟨9, by decide⟩} //
    k ≠ ⟨⟨10, by decide⟩, by decide⟩} → Fin 256)) ≃ Fin (2 ^ 19) × Rest where
  toFun y := (⟨y.1.val % 8 * 65536 + y.2.1.val * 256 + y.2.2.1.val, by omega⟩, ⟨y.1.val / 8, by omega⟩, y.2.2.2)
  invFun z := (⟨z.2.1.val * 8 + z.1.val / 65536, by omega⟩, ⟨z.1.val / 256 % 256, by omega⟩,
    ⟨z.1.val % 256, by omega⟩, z.2.2)
  left_inv y := by
    obtain ⟨a, b, c, r⟩ := y
    simp only [Prod.mk.injEq, Fin.ext_iff, and_true]
    omega
  right_inv z := by
    obtain ⟨i, t, r⟩ := z
    simp only [Prod.mk.injEq, Fin.ext_iff, and_true]
    omega

/-- A stream as the index `uniformOn` reads from it at `2^19` and the bytes it leaves. -/
def split : Bytes 4096 ≃ Fin (2 ^ 19) × Rest :=
  (Equiv.piSplitAt (8 : Fin 4096) fun _ => Fin 256).trans
    ((Equiv.prodCongr (Equiv.refl _) ((Equiv.piSplitAt (⟨9, by decide⟩ : {j : Fin 4096 // j ≠ 8}) fun _ => Fin 256).trans
      (Equiv.prodCongr (Equiv.refl _) (Equiv.piSplitAt
        (⟨⟨10, by decide⟩, by decide⟩ : {j : {j : Fin 4096 // j ≠ 8} // j ≠ ⟨9, by decide⟩}) fun _ => Fin 256)))).trans mix)

theorem split_fst (x : Bytes 4096) : ((split x).1 : ℕ) = (x 8).val % 8 * 65536 + (x 9).val * 256 + (x 10).val := rfl

theorem uniformOn_split4 (x : Bytes 4096) :
    FlockSoundness.Randomness.uniformOn (2 ^ 19) (bytesOf x) = some (split x).1.val := by
  rw [uniformOn_eq, split_fst]

theorem uniformOn_split (x : Bytes DRAW_BYTES) :
    FlockSoundness.Randomness.uniformOn (2 ^ 19) (bytesOf x) = some (split x).1.val :=
  uniformOn_split4 x

theorem drawExact : DrawExact := by
  show ∀ P : Fin (2 ^ 19) → Prop, pr (fun s : Bytes 4096 =>
    ∃ i : Fin (2 ^ 19), FlockSoundness.Randomness.uniformOn (2 ^ 19) (bytesOf s) = some i.val ∧ P i) = pr P
  intro P
  have h : (fun s : Bytes 4096 =>
      ∃ i : Fin (2 ^ 19), FlockSoundness.Randomness.uniformOn (2 ^ 19) (bytesOf s) = some i.val ∧ P i) =
      fun s : Bytes 4096 => P (split s).1 := by
    funext s
    apply propext
    rw [uniformOn_split4]
    constructor
    · rintro ⟨i, hi, hP⟩
      have : (split s).1 = i := Fin.ext (Option.some.inj hi)
      rwa [this]
    · intro hP
      exact ⟨_, rfl, hP⟩
  rw [h, PousThm1.pr_equiv split (fun y => P y.1), PousThm1.pr_fst]


/-! ### Reindexing, over any types -/

theorem pr_reindex {Ω C T I R : Type} [Fintype Ω] [Fintype C] [Fintype T] [Fintype I] [Fintype R] [Nonempty R]
    (e : T ≃ I × R) (F : Ω → C → I → Prop) :
    pr (fun x : Ω × C × T => F x.1 x.2.1 (e x.2.2).1) = pr (fun x : Ω × C × I => F x.1 x.2.1 x.2.2) := by
  let f : Ω × C × T ≃ (Ω × C × I) × R :=
    (Equiv.prodCongr (Equiv.refl _) (Equiv.prodCongr (Equiv.refl _) e)).trans
      ((Equiv.prodCongr (Equiv.refl _) (Equiv.prodAssoc _ _ _).symm).trans (Equiv.prodAssoc _ _ _).symm)
  exact (PousThm1.pr_equiv f (fun y : (Ω × C × I) × R => F y.1.1 y.1.2.1 y.1.2.2)).trans
    (PousThm1.pr_fst (β := R) fun y : Ω × C × I => F y.1 y.2.1 y.2.2)

theorem pr_false {α : Type} [Fintype α] : pr (fun _ : α => False) = 0 := by
  classical
  rw [pr_def]; simp

/-- Slicing on the commitment's coins and indices: when the event needs a commitment and every commitment `v` bounds
it by `c` over the rest, it is bounded by `c`. -/
theorem pr_slice_le {Ω C I V : Type} [Fintype Ω] [Fintype C] [Fintype I] [Nonempty Ω] [Nonempty C] [Nonempty I]
    (E : Ω → Option V → I → Prop) (com : C → I → Option V) (c : ℝ≥0∞) (hnone : ∀ ω I, ¬ E ω none I)
    (hsome : ∀ v, pr (fun o : Ω × Unit × I => E o.1 (some v) o.2.2) ≤ c) :
    pr (fun x : Ω × C × I × I => E x.1 (com x.2.1 x.2.2.2) x.2.2.1) ≤ c := by
  have h : ∀ a : C × I, ¬ False → pr (fun o : Ω × Unit × I => E o.1 (com a.1 a.2) o.2.2) ≤ c := by
    intro a _
    rcases hw : com a.1 a.2 with _ | v
    · have : (fun o : Ω × Unit × I => E o.1 none o.2.2) = fun _ => False :=
        funext fun o => propext ⟨hnone _ _, False.elim⟩
      rw [this, pr_false]
      exact bot_le
    · exact hsome v
  have h₁ := PousThm1.pr_prod_le (α := C × I) (β := Ω × Unit × I) (fun a o => E o.1 (com a.1 a.2) o.2.2)
    (fun _ => False) c h
  rw [pr_false, zero_add] at h₁
  let e : (C × I) × Ω × Unit × I ≃ Ω × C × I × I :=
    { toFun := fun y => (y.2.1, y.1.1, y.2.2.2, y.1.2)
      invFun := fun x => ((x.2.1, x.2.2.2), (x.1, (), x.2.2.1))
      left_inv := fun _ => rfl
      right_inv := fun _ => rfl }
  have h₂ := PousThm1.pr_equiv e fun x : Ω × C × I × I => E x.1 (com x.2.1 x.2.2.2) x.2.2.1
  rw [← h₂]
  exact h₁

/-! ### From the streams to uniform indices -/

/-- The game's draw as the indices it reads and the bytes it leaves. -/
def splitAll : Streams ≃ (Fin 105 → Fin (2 ^ 19)) × (Fin 105 → Rest) :=
  (Equiv.piCongrRight fun _ => split).trans (Equiv.arrowProdEquivProdArrow _ _ _)

/-- The server passes on the indices `I` and the coin view `v`. -/
def Ev {S D Q L : ℕ} (session binding : ByteArray) (W : Bits sch.n) (A : Server S D Q L)
    (ω : (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω) (v : Option (ByteArray × ByteArray))
    (I : Fin 105 → Fin (2 ^ 19)) : Prop :=
  ∃ v', v = some v' ∧
    seqAnswers ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).answer ω)
        (A.step (commitmentOf session binding v') W (sch.enc W ω ()).1) 0 (List.ofFn I)
        (A.pre W (sch.enc W ω ()).1 (sch.enc W ω ()).2 ω) =
      List.ofFn fun j => (sch.enc W ω ()).2 (I j)

theorem passes_iff {S D Q L : ℕ} (session binding : ByteArray) (W : Bits sch.n) (A : Server S D Q L)
    (x : (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams) :
    Passes session binding W A x ↔
      Ev session binding W A x.1 (coinView session x.2.1 (splitAll x.2.2).1) (splitAll x.2.2).1 := by
  have hI : ∀ I : Fin 105 → Fin (2 ^ 19),
      (∀ j, FlockSoundness.Randomness.uniformOn (2 ^ 19) (bytesOf (x.2.2 j)) = some (I j).val) ↔
        I = (splitAll x.2.2).1 := by
    intro I
    simp only [uniformOn_split, Option.some.injEq]
    constructor
    · intro h; funext j; exact Fin.ext (h j).symm
    · rintro rfl j; rfl
  constructor
  · rintro ⟨I, v, hu, hv, hs⟩
    obtain rfl := (hI I).1 hu
    exact ⟨v, hv, hs⟩
  · rintro ⟨v, hv, hs⟩
    exact ⟨_, v, (hI _).2 rfl, hv, hs⟩

/-- The streams' draw is the uniform indices' (`ε_rs = 0`). -/
theorem pr_passes {S D Q L : ℕ} (session binding : ByteArray) (W : Bits sch.n) (A : Server S D Q L) :
    pr (Passes session binding W A) =
      pr fun x : (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × (Fin 105 → Fin (2 ^ 19)) =>
        Ev session binding W A x.1 (coinView session x.2.1 x.2.2) x.2.2 := by
  rw [show Passes session binding W A = fun x => Ev session binding W A x.1
      (coinView session x.2.1 (splitAll x.2.2).1) (splitAll x.2.2).1 from
    funext fun x => propext (passes_iff session binding W A x)]
  exact pr_reindex splitAll fun ω hc I => Ev session binding W A ω (coinView session hc I) I

/-! ### The budget and the ring game -/

theorem budget : (2⁻¹ : ℝ≥0∞) ^ 129 + (2⁻¹ : ℝ≥0∞) ^ 130 + (2⁻¹ : ℝ≥0∞) ^ 131 ≤ Params.εMax := by
  have hk : ∀ n : ℕ, (2⁻¹ : ℝ≥0∞) ^ (n + 1) + (2⁻¹ : ℝ≥0∞) ^ (n + 1) = (2⁻¹ : ℝ≥0∞) ^ n := by
    intro n
    rw [← two_mul, pow_succ, mul_comm ((2⁻¹ : ℝ≥0∞) ^ n), ← mul_assoc,
      ENNReal.mul_inv_cancel (by norm_num) (by norm_num), one_mul]
  have h31 : (2⁻¹ : ℝ≥0∞) ^ 131 ≤ (2⁻¹ : ℝ≥0∞) ^ 130 :=
    pow_le_pow_of_le_one bot_le (ENNReal.inv_le_one.2 one_le_two) (by norm_num)
  calc _ ≤ (2⁻¹ : ℝ≥0∞) ^ 129 + ((2⁻¹ : ℝ≥0∞) ^ 130 + (2⁻¹ : ℝ≥0∞) ^ 130) := by rw [add_assoc]; gcongr
    _ = Params.εMax := by rw [hk 129, hk 128]; rfl

set_option exponentiation.threshold 20000 in
/-- Every session commitment `v` turns the server into a `SeqResponderOn` of `P2RingAuditSecure`'s game. -/
def responder {S D Q L : ℕ} (session binding : ByteArray) (A : Server S D Q L) (v : ByteArray × ByteArray) :
    SeqResponderOn sch S D Q L :=
  ⟨A.step (commitmentOf session binding v), fun W pp j σ i => A.bounded _ W pp j σ i⟩

theorem ev_some_le {S D Q L : ℕ} (hP : Pous.P2.Assumptions.P2RingAuditSecure D Q L) (hS : S = Pous.EndToEnd.S)
    (session binding : ByteArray) (W : Bits sch.n) (A : Server S D Q L) (v : ByteArray × ByteArray) :
    pr (fun o : (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × Unit × (Fin 105 → Fin (2 ^ 19)) =>
        Ev session binding W A o.1 (some v) o.2.2) ≤
      ENNReal.ofReal Params.δ + ((2⁻¹ : ℝ≥0∞) ^ 129 + (2⁻¹ : ℝ≥0∞) ^ 130) := by
  subst hS
  have h := hP W A.pre (responder session binding A v)
  refine le_trans (pr_mono ?_) h
  rintro o ⟨v', hv, hs⟩
  obtain rfl := Option.some.inj hv
  exact hs

set_option exponentiation.threshold 20000 in
theorem endToEndRing : EndToEndRing := by
  intro hH D Q L hP session binding W A
  rw [pr_passes]
  have h₁ := hH session (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω (Ev session binding W A)
  have h₂ := pr_slice_le (Ω := (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω) (C := HmCoins)
    (I := Fin 105 → Fin (2 ^ 19)) (Ev session binding W A) (coinView session)
    (ENNReal.ofReal Params.δ + ((2⁻¹ : ℝ≥0∞) ^ 129 + (2⁻¹ : ℝ≥0∞) ^ 130))
    (fun _ _ ⟨_, h, _⟩ => by cases h) (ev_some_le hP rfl session binding W A)
  refine h₁.trans ((add_le_add h₂ le_rfl).trans ?_)
  rw [add_assoc]
  gcongr
  exact budget

set_option exponentiation.threshold 20000 in
theorem endToEndRingDeployed : EndToEndRingDeployed := fun hH hP => endToEndRing hH _ _ _ hP

end Pous.EndToEnd.SecurityProofs
