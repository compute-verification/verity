import Properties.Pous.EndToEnd
import Proofs.Pous.Sanity
import Proofs.Pous.Thm1.Theorem1iiSeq
import Proofs.Pous.P2.Round

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
/-- `EndToEndRing` from the whole audit's form, `P2RingAuditSecure`. -/
theorem endToEndRing_of_audit (hH : Pous.P2.Assumptions.CoinCommitHides coinView ((2⁻¹ : ℝ≥0∞) ^ 131)) (D Q L : ℕ)
    (hP : Pous.P2.Assumptions.P2RingAuditSecure D Q L) (session binding : ByteArray) (W : Bits sch.n)
    (A : Server S D Q L) : pr (Passes session binding W A) ≤ ENNReal.ofReal Params.δ + Params.εMax := by
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

theorem endToEndRing : EndToEndRing := fun hH D Q L hP =>
  endToEndRing_of_audit hH D Q L (PousP2Round.p2RingAuditSecure_of_round D Q L hP)

set_option exponentiation.threshold 20000 in
theorem endToEndRingDeployed : EndToEndRingDeployed := fun hH hP => endToEndRing hH _ _ _ hP

/-! ### The code always builds the coin tree -/

theorem uint8_clear (v : UInt8) : (v &&& 0x7f) &&& 0x80 = 0 := by
  rw [UInt8.and_assoc, show (0x7f &&& 0x80 : UInt8) = 0 from rfl]
  exact UInt8.toBitVec_inj.1 (by simp)

theorem get!_set!_self (b : ByteArray) (i : ℕ) (v : UInt8) (h : i < b.size) : (b.set! i v).get! i = v := by
  obtain ⟨bs⟩ := b
  exact Array.getElem!_set!_self bs i v h

theorem checkKey_coinKey (k : Bytes 256) : checkKey (coinKey k) = .ok () := by
  have hs : (coinKey k).size = 256 := by rw [coinKey, ByteArray.size_set!, size_bytesOf]
  have hg : (coinKey k).get! 255 = (bytesOf k).get! 255 &&& 0x7f := by
    have h := get!_set!_self (bytesOf k) 255 ((bytesOf k).get! 255 &&& 0x7f) (by rw [size_bytesOf]; omega)
    exact h
  unfold checkKey
  rw [show HM96.keyBytes = 256 from rfl, hs, hg, uint8_clear]
  rfl

theorem list_mapM_ok {α β ε : Type} (f : α → Except ε β) :
    ∀ l : List α, (∀ a ∈ l, ∃ b, f a = .ok b) → ∃ bs, l.mapM f = .ok bs
  | [], _ => ⟨[], rfl⟩
  | a :: l, h => by
    obtain ⟨b, hb⟩ := h a (by simp)
    obtain ⟨bs, hbs⟩ := list_mapM_ok f l fun x hx => h x (by simp [hx])
    exact ⟨b :: bs, by rw [List.mapM_cons, hb, hbs]; rfl⟩

theorem beN_ok (n x : ℕ) (w : String) (h : x < 2 ^ (8 * n)) : ∃ b, beN n x w = .ok b := by
  unfold beN; simp only [h, ↓reduceIte]; exact ⟨_, rfl⟩

theorem coinLeaf_ok (key session : ByteArray) (j index : ℕ) (salt : ByteArray) (hk : checkKey key = .ok ())
    (hj : j < 2 ^ 32) (hi : index < 2 ^ 64) (hs : salt.size = 192) : ∃ b, coinLeaf key session j index salt = .ok b := by
  obtain ⟨bj, hbj⟩ := beN_ok 4 j "j" hj
  obtain ⟨bi, hbi⟩ := beN_ok 8 index "index" hi
  have hsalt : checkSalt salt = .ok () := by
    unfold checkSalt; rw [hs]; rfl
  simp only [coinLeaf, coinDigest, hbj, hbi, hk, hsalt]
  exact ⟨_, rfl⟩

theorem ofFn_getElem! {α : Type} [Inhabited α] {n : ℕ} (f : Fin n → α) (a : ℕ) (h : a < n) :
    (Array.ofFn f)[a]! = f ⟨a, h⟩ := by
  rw [getElem!_pos (Array.ofFn f) a (by simpa using h)]
  simp [Array.getElem_ofFn]

theorem make_ok_of (session key : ByteArray) (indices : Array ℕ) (salts : Array ByteArray) (hs : session.size = 32)
    (hn : indices.size = 105) (hsalts : salts.size = indices.size) (hlt : ∀ j < indices.size, indices[j]! < 2 ^ 19)
    (hsalt : ∀ j < indices.size, salts[j]!.size = 192) (hk : checkKey key = .ok ()) :
    ∃ t, CoinTree.make session (2 ^ 19) indices key salts = .ok t := by
  have hcond : (decide (2 ^ 19 < 1) || indices.isEmpty || salts.size != indices.size ||
      indices.any fun x => decide (x ≥ 2 ^ 19)) = false := by
    have he : indices.isEmpty = false := by
      rw [Array.isEmpty_eq_false_iff]; intro h; rw [h] at hn; simp at hn
    rw [hsalts, he, bne_self_eq_false]
    simp only [Bool.or_false, Bool.or_eq_false_iff]
    refine ⟨by decide, Bool.eq_false_iff.2 fun h => ?_⟩
    rw [Array.any_eq_true] at h
    obtain ⟨i, hi, hp⟩ := h
    have := hlt i hi
    rw [getElem!_pos indices i hi] at this
    simp only [decide_eq_true_eq] at hp
    omega
  obtain ⟨hm, hhm⟩ : ∃ hm, (Array.range indices.size).mapM (fun j =>
      coinLeaf key session j indices[j]! salts[j]!) = .ok hm := by
    rw [Array.mapM_eq_mapM_toList, Array.toList_range]
    obtain ⟨bs, hbs⟩ := list_mapM_ok _ (List.range indices.size) fun a ha => by
      have h := List.mem_range.1 ha
      exact coinLeaf_ok key session a indices[a]! salts[a]! hk (by omega) (by have := hlt a h; omega) (hsalt a h)
    exact ⟨bs.toArray, by rw [hbs]; rfl⟩
  simp only [CoinTree.make, hs, ne_eq, not_true_eq_false, ↓reduceIte, hcond, Bool.false_eq_true]
  rw [hk]
  simp only [bind, Except.bind]
  rw [hhm]
  exact ⟨_, rfl⟩

theorem coinView_some (session : ByteArray) (hs : session.size = 32) (c : HmCoins) (I : Fin 105 → Fin (2 ^ 19)) :
    ∃ v, coinView session c I = some v := by
  obtain ⟨t, ht⟩ := make_ok_of session (coinKey c.1) (Array.ofFn fun j => (I j).val)
    (Array.ofFn fun j => bytesOf (c.2 j)) hs (Array.size_ofFn) (by rw [Array.size_ofFn, Array.size_ofFn])
    (fun j hj => by
      rw [Array.size_ofFn] at hj
      rw [ofFn_getElem! _ j hj]; exact (I _).isLt)
    (fun j hj => by
      rw [Array.size_ofFn] at hj
      rw [ofFn_getElem! _ j hj]; exact size_bytesOf _)
    (checkKey_coinKey c.1)
  exact ⟨_, by rw [coinView, ht]; rfl⟩

/-! ### Non-vacuity: the honest full-state server -/

theorem stateBound_lt : S < FULL := by
  show ⌊Params.ρ * ((2 ^ 19 * 16448 : ℕ) : ℚ)⌋₊ < 2 ^ 19 * 16448
  rw [Nat.floor_lt (by unfold Params.ρ; positivity)]
  unfold Params.ρ
  norm_num

set_option exponentiation.threshold 20000 in
/-- The honest full-state responder: it keeps all of `C` and answers each index from it, with no query. -/
noncomputable def honestOn : SeqResponderOn sch FULL 0 0 0 :=
  ⟨fun _ _ _ σ i => .ret ((codeEquiv sch.B sch.ℓ).symm σ i, σ), fun _ _ _ _ _ _ => ⟨le_rfl, le_rfl, le_rfl⟩⟩

set_option exponentiation.threshold 20000 in
/-- The honest full-state server, whatever the session commitment. -/
noncomputable def honest : Server FULL 0 0 0 :=
  ⟨honestPre sch, fun _ => honestOn.step, fun _ => honestOn.bounded⟩

set_option exponentiation.threshold 20000 in
theorem honest_answers (f : (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Qry → (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ans)
    (W : Bits sch.n) (pp : Bits sch.t) (C : Fin sch.B → Bits sch.ℓ) (k : ℕ) (I : Fin k → Fin sch.B) :
    seqAnswers f (honestOn.step W pp) 0 (List.ofFn I) (codeEquiv _ _ C) = List.ofFn fun j => C (I j) := by
  show seqAnswers f (fun _ σ i => .ret ((codeEquiv sch.B sch.ℓ).symm σ i, σ)) 0 (List.ofFn I) (codeEquiv _ _ C) = _
  rw [seqAnswers_const f (fun σ i => (codeEquiv sch.B sch.ℓ).symm σ i), List.map_ofFn]
  congr 1
  funext j
  simp

set_option exponentiation.threshold 20000 in
theorem honestPassesRing : HonestPassesRing :=
  ⟨stateBound_lt, honestPre sch, honestOn, fun W => pr_eq_one fun x =>
    honest_answers _ W _ (sch.enc W x.1 x.2.1).2 105 x.2.2⟩

set_option exponentiation.threshold 20000 in
theorem honestPassesAccepts : HonestPassesAccepts := by
  refine ⟨honest, fun session hs binding W => pr_eq_one fun x => ?_⟩
  obtain ⟨v, hv⟩ := coinView_some session hs x.2.1 (splitAll x.2.2).1
  exact ⟨(splitAll x.2.2).1, v, fun j => uniformOn_split (x.2.2 j), hv,
    honest_answers _ W (sch.enc W x.1 ()).1 (sch.enc W x.1 ()).2 105 (splitAll x.2.2).1⟩

/-! ### The graded ring game -/

theorem rounds_eq {M : IdealModel} {ℓ B : ℕ} (f : M.Qry → M.Ans)
    (step : ℕ → List Bool → Fin B → Prog M (Bits ℓ × List Bool)) :
    ∀ (j : ℕ) (l : List (Fin B)) (σ : List Bool),
      rounds f step j l σ = PousP2Round.trace (fun j σ i => (step j σ i).run f) j l σ
  | _, [], _ => rfl
  | j, i :: is, σ => by simp only [rounds, PousP2Round.trace, rounds_eq f step (j + 1) is]

/-- The streams' draw is the uniform indices' (`ε_rs = 0`), for any event of the coin view and the indices. -/
theorem pr_draw (session : ByteArray)
    (G : (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω → ByteArray × ByteArray → (Fin 105 → Fin (2 ^ 19)) → Prop) :
    pr (fun x : (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams =>
        ∃ (I : Fin 105 → Fin (2 ^ 19)) (v : ByteArray × ByteArray),
          (∀ j, FlockSoundness.Randomness.uniformOn (2 ^ 19) (bytesOf (x.2.2 j)) = some (I j).val) ∧
            coinView session x.2.1 I = some v ∧ G x.1 v I) =
      pr fun x : (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × (Fin 105 → Fin (2 ^ 19)) =>
        ∃ v, coinView session x.2.1 x.2.2 = some v ∧ G x.1 v x.2.2 := by
  have h : (fun x : (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams =>
      ∃ (I : Fin 105 → Fin (2 ^ 19)) (v : ByteArray × ByteArray),
        (∀ j, FlockSoundness.Randomness.uniformOn (2 ^ 19) (bytesOf (x.2.2 j)) = some (I j).val) ∧
          coinView session x.2.1 I = some v ∧ G x.1 v I) =
      fun x => (fun ω c I => ∃ v, coinView session c I = some v ∧ G ω v I) x.1 x.2.1 (splitAll x.2.2).1 := by
    funext x
    apply propext
    have hI : ∀ I : Fin 105 → Fin (2 ^ 19),
        (∀ j, FlockSoundness.Randomness.uniformOn (2 ^ 19) (bytesOf (x.2.2 j)) = some (I j).val) ↔
          I = (splitAll x.2.2).1 := by
      intro I
      simp only [uniformOn_split, Option.some.injEq]
      constructor
      · intro h; funext j; exact Fin.ext (h j).symm
      · rintro rfl j; rfl
    constructor
    · rintro ⟨I, v, hu, hv, hG⟩
      obtain rfl := (hI I).1 hu
      exact ⟨v, hv, hG⟩
    · rintro ⟨v, hv, hG⟩
      exact ⟨_, v, (hI _).2 rfl, hv, hG⟩
  rw [h]
  exact pr_reindex splitAll fun ω c I => ∃ v, coinView session c I = some v ∧ G ω v I

/-- The server passes on the indices `I` and the coin view `v`, beginning at least `m` rounds short. -/
def EvS {D Q L : ℕ} (m : ℕ) (session binding : ByteArray) (W : Bits sch.n) (A : AnyServer D Q L)
    (ω : (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω) (v : Option (ByteArray × ByteArray))
    (I : Fin 105 → Fin (2 ^ 19)) : Prop :=
  ∃ v', v = some v' ∧
    ((roundsOf session binding W A ω v' I).map Prod.fst = List.ofFn (fun j => (sch.enc W ω ()).2 (I j)) ∧
      m ≤ (roundsOf session binding W A ω v' I).countP fun r => decide (r.2.length ≤ S))

set_option exponentiation.threshold 20000 in
/-- Every session commitment `v` turns the server into a responder of `PousP2Round.graded_of_round`. -/
theorem graded_some_le {D Q L : ℕ} (hP : Pous.P2.Assumptions.P2RingRoundSecure D Q L) (session binding : ByteArray)
    (W : Bits sch.n) (A : AnyServer D Q L) (m : ℕ) (v : ByteArray × ByteArray) :
    pr (fun o : (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × Unit × (Fin 105 → Fin (2 ^ 19)) =>
        EvS m session binding W A o.1 (some v) o.2.2) ≤
      ((501578 : ℝ≥0∞) / 2 ^ 19) ^ m + ((105 : ℕ) : ℝ≥0∞) * (2⁻¹ : ℝ≥0∞) ^ 136 := by
  have h := PousP2Round.graded_of_round sch S D Q L 105 m _ _ hP W
    (fun x => A.pre W (sch.enc W x.1 x.2).1 (sch.enc W x.1 x.2).2 x.1) (A.step (commitmentOf session binding v) W)
    (A.bounded _ W)
  rw [← PousThm1.pr_equiv (Equiv.prodAssoc (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω sch.Coins
    (Fin 105 → Fin sch.B)).symm] at h
  refine le_trans (pr_mono fun o ho => ?_) h
  obtain ⟨v', hv, hs⟩ := ho
  obtain rfl := Option.some.inj hv
  simp only [roundsOf, rounds_eq] at hs
  exact hs

/-- `105 · 2^-136 + 2^-131 ≤ 2^-128`. -/
theorem budget_graded : ((105 : ℕ) : ℝ≥0∞) * (2⁻¹ : ℝ≥0∞) ^ 136 + (2⁻¹ : ℝ≥0∞) ^ 131 ≤ Params.εMax :=
  le_trans (add_le_add PousP2Round.eps_le le_rfl) (le_trans (add_le_add le_self_add le_rfl) budget)

set_option exponentiation.threshold 20000 in
theorem endToEndRingGraded : EndToEndRingGraded := by
  intro hH D Q L hP session binding W A m
  have h₀ := pr_draw session fun ω v I =>
    (roundsOf session binding W A ω v I).map Prod.fst = List.ofFn (fun j => (sch.enc W ω ()).2 (I j)) ∧
      m ≤ (roundsOf session binding W A ω v I).countP fun r => decide (r.2.length ≤ S)
  have h₁ := hH session (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω (EvS m session binding W A)
  have h₂ := pr_slice_le (Ω := (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω) (C := HmCoins)
    (I := Fin 105 → Fin (2 ^ 19)) (EvS m session binding W A) (coinView session)
    (((501578 : ℝ≥0∞) / 2 ^ 19) ^ m + ((105 : ℕ) : ℝ≥0∞) * (2⁻¹ : ℝ≥0∞) ^ 136)
    (fun _ _ ⟨_, h, _⟩ => by cases h) (graded_some_le hP session binding W A m)
  calc pr (PassesShort m session binding W A) = _ := h₀
    _ ≤ _ := h₁
    _ ≤ _ := add_le_add h₂ le_rfl
    _ ≤ _ := by rw [add_assoc]; exact add_le_add le_rfl budget_graded

set_option exponentiation.threshold 20000 in
theorem endToEndRingGradedDeployed : EndToEndRingGradedDeployed := fun hH hP => endToEndRingGraded hH _ _ _ hP

/-! ### Non-vacuity: a server holding 496,693 blocks -/

set_option exponentiation.threshold 20000 in
/-- The first 496,693 blocks of `C`, as bits. -/
@[irreducible] noncomputable def kept (C : sch.Code) : List Bool :=
  List.ofFn fun k : Fin (496693 * 16448) =>
    C ⟨k.val / 16448, show k.val / 16448 < 2 ^ 19 by have := k.isLt; omega⟩
      ⟨k.val % 16448, show k.val % 16448 < 16448 from Nat.mod_lt _ (by norm_num)⟩

set_option exponentiation.threshold 20000 in
/-- A server that keeps the first 496,693 blocks of `C` and answers each index among them from its state, with no
query. -/
noncomputable def keeper : AnyServer 0 0 0 :=
  ⟨fun _ _ C _ => kept C, fun _ _ _ _ σ i => .ret (fun b => σ.getD (i.val * 16448 + b.val) false, σ),
    fun _ _ _ _ _ _ _ => ⟨le_rfl, le_rfl, le_rfl⟩⟩

theorem keeper_run (f : (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Qry → (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ans)
    (sc : SessionCommitment) (W : Bits sch.n) (pp : Bits sch.t) (j : ℕ) (σ : List Bool) (i : Fin sch.B) :
    (keeper.step sc W pp j σ i).run f = (fun b => σ.getD (i.val * 16448 + b.val) false, σ) := rfl

set_option exponentiation.threshold 20000 in
theorem kept_block (C : sch.Code) (i : Fin sch.B) (hi : (i : ℕ) < 496693) (b : Fin sch.ℓ) :
    (kept C).getD (i.val * 16448 + b.val) false = C i b := by
  have hb : b.val < 16448 := b.isLt
  have hk : i.val * 16448 + b.val < 496693 * 16448 := by omega
  unfold kept
  rw [List.getD_eq_getElem?_getD, List.getElem?_ofFn, dite_eq_left hk, Option.getD_some]
  congr 1
  · congr 1
    show (i.val * 16448 + b.val) / 16448 = i.val
    omega
  · exact Fin.ext (show (i.val * 16448 + b.val) % 16448 = b.val by omega)

set_option exponentiation.threshold 20000 in
theorem length_kept (C : sch.Code) : (kept C).length ≤ S := by
  have h : (kept C).length = 496693 * 16448 := by
    unfold kept
    exact List.length_ofFn
  rw [h]
  show 496693 * 16448 ≤ ⌊Params.ρ * ((2 ^ 19 * 16448 : ℕ) : ℚ)⌋₊
  refine Nat.le_floor ?_
  unfold Params.ρ
  norm_num

set_option exponentiation.threshold 20000 in
theorem keeper_rounds (f : (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Qry → (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ans)
    (sc : SessionCommitment) (W : Bits sch.n) (pp : Bits sch.t) (C : sch.Code) :
    ∀ (j : ℕ) (l : List (Fin sch.B)), (∀ i ∈ l, (i : ℕ) < 496693) →
      rounds f (keeper.step sc W pp) j l (kept C) = l.map fun i => (C i, kept C)
  | _, [], _ => rfl
  | j, i :: is, h => by
    simp only [rounds, keeper_run, List.map_cons]
    rw [keeper_rounds f sc W pp C (j + 1) is fun i' hi' => h i' (List.mem_cons_of_mem _ hi')]
    congr 2
    funext b
    exact kept_block C i (h i List.mem_cons_self) b

set_option exponentiation.threshold 20000 in
theorem shortServerPasses : ShortServerPasses := by
  refine ⟨keeper, fun session hs binding W => ?_⟩
  have hlow : pr (fun x : (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams =>
      ∀ j, ((splitAll x.2.2).1 j : ℕ) < 496693) = ((496693 : ℝ≥0∞) / 2 ^ 19) ^ 105 := by
    refine (pr_reindex splitAll fun _ _ I => ∀ j, (I j : ℕ) < 496693).trans ?_
    refine (pr_snd (α := (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω)
      fun y : HmCoins × (Fin 105 → Fin (2 ^ 19)) => ∀ j, (y.2 j : ℕ) < 496693).trans ?_
    refine (pr_snd (α := HmCoins) fun I : Fin 105 → Fin (2 ^ 19) => ∀ j, (I j : ℕ) < 496693).trans ?_
    rw [pr_all_lt 105 (2 ^ 19) 496693 (by norm_num)]
    norm_num
  rw [← hlow]
  refine pr_mono fun x hx => ?_
  obtain ⟨v, hv⟩ := coinView_some session hs x.2.1 (splitAll x.2.2).1
  refine ⟨(splitAll x.2.2).1, v, fun j => uniformOn_split (x.2.2 j), hv, ?_⟩
  have hr := keeper_rounds ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).answer x.1) (commitmentOf session binding v) W
    (sch.enc W x.1 ()).1 (sch.enc W x.1 ()).2 0 (List.ofFn (splitAll x.2.2).1) fun i hi => by
      obtain ⟨j, rfl⟩ := List.mem_ofFn.1 hi
      exact hx j
  have hro : roundsOf session binding W keeper x.1 v (splitAll x.2.2).1 =
      (List.ofFn (splitAll x.2.2).1).map fun i => ((sch.enc W x.1 ()).2 i, kept (sch.enc W x.1 ()).2) := by
    unfold roundsOf
    exact hr
  rw [hro]
  refine ⟨?_, le_of_eq ((List.countP_eq_length.2 fun r hr => ?_).trans ?_).symm⟩
  · exact List.map_map.trans List.map_ofFn
  · obtain ⟨i, -, rfl⟩ := List.mem_map.1 hr
    exact decide_eq_true (length_kept _)
  · exact (List.length_map _).trans List.length_ofFn

end Pous.EndToEnd.SecurityProofs
