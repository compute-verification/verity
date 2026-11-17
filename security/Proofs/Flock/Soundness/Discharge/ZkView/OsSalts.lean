module

public import Proofs.Flock.Soundness.Discharge.ZkView.Leaf
public import Proofs.Flock.Soundness.Discharge.PrivateCircuit.Defs
public import Proofs.Flock.Soundness.CROnly.Hiding
public import Proofs.Flock.Soundness.ZK.RealLeaves
public import Proofs.Flock.Soundness.ExecSample

/-! Salts from the operating system (A3); no computational property of SHA-512 enters their hiding. -/

@[expose] public section

namespace FlockSoundness.Discharge.ZkView

open Game ZK
open scoped ENNReal

/-- **A salt from 192 bytes**, in `bitsOf`'s order: bit `t` is bit `t % 8` of byte `t / 8`. -/
def saltOfBytes (b : Fin 192 → Fin 256) : Salt := fun t =>
  if (b ⟨t.val / 8, by have := t.isLt; omega⟩).val.testBit (t.val % 8) then 1 else 0

theorem byteOf_testBit_fin (v : Fin 256) :
    byteOf (fun j : Fin 8 => if v.val.testBit j.val then (1 : ZMod 2) else 0) = UInt8.ofFin v := by
  refine UInt8.toNat_inj.1 (Nat.eq_of_testBit_eq fun b => ?_)
  rw [UInt8.toNat_ofFin]
  by_cases hb : b < 8
  · rw [show b = (⟨b, hb⟩ : Fin 8).val from rfl, testBit_byteOf]
    by_cases h : v.val.testBit b <;> simp [h]
  · have h8 : 2 ^ 8 ≤ 2 ^ b := Nat.pow_le_pow_right (by norm_num) (by omega)
    have h1 := lt_of_lt_of_le (UInt8.toNat_lt
      (byteOf fun j : Fin 8 => if v.val.testBit j.val then (1 : ZMod 2) else 0)) h8
    have h2 : v.val < 2 ^ b := lt_of_lt_of_le v.isLt h8
    rw [Nat.testBit_eq_false_of_lt h1, Nat.testBit_eq_false_of_lt h2]

theorem saltOfBytes_apply (b : Fin 192 → Fin 256) (i : Fin 192) (j : Fin 8) (h : 8 * i.val + j.val < 1536) :
    saltOfBytes b ⟨8 * i.val + j.val, h⟩ = if (b i).val.testBit j.val then 1 else 0 := by
  have e1 : (⟨(8 * i.val + j.val) / 8, by omega⟩ : Fin 192) = i := Fin.ext (by dsimp only; omega)
  have e2 : (8 * i.val + j.val) % 8 = j.val := by omega
  simp only [saltOfBytes, e1, e2]

/-- Stated for every `k` because at a literal `k` such as 192, `Array.ofFn` unfolds element by element past the
recursion limit. -/
theorem bytesOf_eq_streamOf {k : ℕ} (x : Fin (8 * k) → ZMod 2) (b : Fin k → Fin 256)
    (h : ∀ (i : Fin k) (j : Fin 8) (hij : 8 * i.val + j.val < 8 * k),
      x ⟨8 * i.val + j.val, hij⟩ = if (b i).val.testBit j.val then 1 else 0) :
    bytesOf x = Audit.SamplingDesign.streamOf b := by
  unfold bytesOf Audit.SamplingDesign.streamOf
  congr 1
  refine congrArg Array.ofFn (funext fun i => ?_)
  rw [← byteOf_testBit_fin (b i)]
  exact congrArg byteOf (funext fun j => h i j _)

theorem saltBytes_saltOfBytes (b : Fin 192 → Fin 256) : saltBytes (saltOfBytes b) = Audit.SamplingDesign.streamOf b :=
  bytesOf_eq_streamOf (k := 192) (saltOfBytes b) b fun i j h => saltOfBytes_apply b i j h

theorem leaf_os (x : ByteArray) (b : Fin 192 → Fin 256) (hx : x.size = 64) :
    Flock.Hm96.Default512.leaf x (Audit.SamplingDesign.streamOf b) =
      leafH (bitsOf x + My (saltOfBytes b), saltHash (saltOfBytes b)) := by
  rw [← saltBytes_saltOfBytes]
  exact leafSplit x _ hx

theorem saltOfBytes_injective : Function.Injective saltOfBytes := by
  intro b b' hb
  funext i
  have h := byteOf_testBit_fin (b i)
  have e : (fun j : Fin 8 => if (b i).val.testBit j.val then (1 : ZMod 2) else 0) =
      fun j : Fin 8 => if (b' i).val.testBit j.val then (1 : ZMod 2) else 0 := funext fun j => by
    rw [← saltOfBytes_apply b i j (by omega), ← saltOfBytes_apply b' i j (by omega), hb]
  rw [e, byteOf_testBit_fin (b' i)] at h
  have := congrArg UInt8.toFin h
  rwa [UInt8.toFin_ofFin, UInt8.toFin_ofFin, eq_comm] at this

noncomputable def saltEquiv : (Fin 192 → Fin 256) ≃ Salt :=
  Equiv.ofBijective saltOfBytes ((Fintype.bijective_iff_injective_and_card _).2
    ⟨saltOfBytes_injective, by
      simp only [Fintype.card_fun, Fintype.card_fin, ZMod.card]
      rw [show (256 : ℕ) = 2 ^ 8 by norm_num, ← pow_mul]⟩)

noncomputable def saltsEquiv (n : ℕ) : (Fin n → Fin 192 → Fin 256) ≃ (Fin n → Salt) :=
  Equiv.piCongrRight fun _ => saltEquiv

theorem uniformRandomBytesTest_of {Ω : Type} [Fintype Ω] {m : ℕ} {L : Fin m → ℕ}
    {t : Ω → (i : Fin m) → Fin (L i) → Fin 256} (h : Assumptions.UniformRandomBytes L t)
    (E : ((i : Fin m) → Fin (L i) → Fin 256) → Prop) : Assumptions.UniformRandomBytesTest L t E 0 :=
  ⟨(h E).le.trans le_self_add, (h E).ge.trans le_self_add⟩

private theorem avg_const_le {C : Type} [Fintype C] (x : ℝ≥0∞) : avg (fun _ : C => x) ≤ x := by
  rcases isEmpty_or_nonempty C with hC | hC
  · simp [avg]
  · exact (avg_const x).le

theorem prCoin_os {Ω Y ι : Type} [Fintype Ω] [Fintype Y] [Fintype ι] [DecidableEq ι] {L : ι → ℕ}
    {t : Ω → (i : ι) → Fin (L i) → Fin 256} (e : ((i : ι) → Fin (L i) → Fin 256) ≃ Y) (E : Y → Prop) {η : ℝ≥0∞}
    (hA3 : Assumptions.UniformRandomBytesTest L t (fun b => E (e b)) η) :
    prCoin (fun ω => E (e (t ω))) ≤ prCoin E + η ∧ prCoin E ≤ prCoin (fun ω => E (e (t ω))) + η := by
  rw [← prCoin_congr_equiv e E]
  exact hA3

theorem prCoin_os_prod {A Ω Y ι : Type} [Fintype A] [Fintype Ω] [Fintype Y] [Fintype ι] [DecidableEq ι] {L : ι → ℕ}
    {t : Ω → (i : ι) → Fin (L i) → Fin 256} (e : ((i : ι) → Fin (L i) → Fin 256) ≃ Y) (E : A → Y → Prop) {η : ℝ≥0∞}
    (hA3 : ∀ a, Assumptions.UniformRandomBytesTest L t (fun b => E a (e b)) η) :
    prCoin (fun p : A × Ω => E p.1 (e (t p.2))) ≤ prCoin (fun p : A × Y => E p.1 p.2) + η ∧
      prCoin (fun p : A × Y => E p.1 p.2) ≤ prCoin (fun p : A × Ω => E p.1 (e (t p.2))) + η := by
  rw [prCoin_prod_split (fun a ω => E a (e (t ω))), prCoin_prod_split E]
  constructor
  · calc avg (fun a => prCoin fun ω => E a (e (t ω)))
        ≤ avg (fun a => prCoin (fun y => E a y) + η) := avg_mono fun a => (prCoin_os e (E a) (hA3 a)).1
      _ = avg (fun a => prCoin fun y => E a y) + avg (fun _ : A => η) := avg_add _ _
      _ ≤ _ := add_le_add le_rfl (avg_const_le η)
  · calc avg (fun a => prCoin fun y => E a y)
        ≤ avg (fun a => prCoin (fun ω => E a (e (t ω))) + η) := avg_mono fun a => (prCoin_os e (E a) (hA3 a)).2
      _ = avg (fun a => prCoin fun ω => E a (e (t ω))) + avg (fun _ : A => η) := avg_add _ _
      _ ≤ _ := add_le_add le_rfl (avg_const_le η)

theorem hm96Hiding_os {Ω Y B C ι : Type} [Fintype Ω] [Fintype Y] [Fintype B] [Fintype ι] [DecidableEq ι] {L : ι → ℕ}
    {t : Ω → (i : ι) → Fin (L i) → Fin 256} (e : ((i : ι) → Fin (L i) → Fin 256) ≃ Y) (My : Y → B) (c : Y → C)
    {δ₁ : ℝ≥0∞} (hT1 : Claims.Hm96Hiding My c δ₁) (E : B × C → Prop) {η : ℝ≥0∞}
    (hA3 : Assumptions.UniformRandomBytesTest L t (fun b => E (My (e b), c (e b))) η) :
    prCoin (fun ω => E (My (e (t ω)), c (e (t ω)))) ≤ prCoin (fun p : B × Y => E (p.1, c p.2)) + δ₁ + η :=
  (prCoin_os e (fun y => E (My y, c y)) hA3).1.trans (add_le_add (Assumptions.hm96Hiding_apply hT1 E) le_rfl)

/-- That `hidingGap My saltHash` is at most `2^-193` is the assumption `hash-derived-key`; at all but `2^-64` of the
keys it is (`CROnly.hm96_sha512_bad_keys`). -/
theorem hm96Sha512_os {Ω : Type} [Fintype Ω] {t : Ω → Fin 1 → Fin 192 → Fin 256} (E : Bits × ByteArray → Prop)
    {η : ℝ≥0∞} (hA3 : Assumptions.UniformRandomBytesTest (fun _ : Fin 1 => 192) t
      (fun b => E (My (saltOfBytes (b 0)), saltHash (saltOfBytes (b 0)))) η) :
    prCoin (fun ω => E (My (saltOfBytes (t ω 0)), saltHash (saltOfBytes (t ω 0)))) ≤
      prCoin (fun p : Bits × Salt => E (p.1, saltHash p.2)) + ENNReal.ofReal (CROnly.hidingGap My saltHash) + η :=
  hm96Hiding_os ((Equiv.funUnique (Fin 1) _).trans saltEquiv) My saltHash (CROnly.hm96Hiding_gap My saltHash) E hA3

/-- The digests may read anything in `Ω`; the streams are read by their leaves only. -/
theorem ideal_leaves_swap_os {Ω Ωos Y B C D ι : Type} [Fintype Ω] [Nonempty Ω] [Fintype Ωos] [Fintype Y] [Nonempty Y]
    [AddCommGroup B] [Fintype B] [Fintype ι] [DecidableEq ι] {n : ℕ} {L : ι → ℕ}
    {t : Ωos → (i : ι) → Fin (L i) → Fin 256} (e : ((i : ι) → Fin (L i) → Fin 256) ≃ (Fin n → Y))
    (My : Y → B) (c : Y → C) {δ₁ : ℝ≥0∞} (hT1 : Claims.Hm96Hiding My c δ₁) (H : B × C → D)
    (x : Ω → Fin n → B) (G : Ω → (Fin n → D) → Prop) {η : ℝ≥0∞}
    (hA3 : ∀ ω, Assumptions.UniformRandomBytesTest L t (fun b => G ω fun i => H (x ω i + My (e b i), c (e b i))) η) :
    prCoin (fun p : Ω × Ωos => G p.1 fun i => H (x p.1 i + My (e (t p.2) i), c (e (t p.2) i)))
      ≤ prCoin (fun p : Ω × (Fin n → B × Y) => G p.1 fun i => H ((p.2 i).1, c (p.2 i).2)) + n * δ₁ + η :=
  (prCoin_os_prod e (fun ω y => G ω fun i => H (x ω i + My (y i), c (y i))) hA3).1.trans
    (add_le_add (ideal_leaves_swap My c hT1 H n x G) le_rfl)

theorem hides_os {P V Ω Y ι : Type} [Fintype Ω] [Fintype Y] [Fintype ι] [DecidableEq ι] {L : ι → ℕ}
    {t : Ω → (i : ι) → Fin (L i) → Fin 256} (e : ((i : ι) → Fin (L i) → Fin 256) ≃ Y) {reg : P → Y → V} {ε : ℝ≥0∞}
    (h : PrivateCircuit.Hides reg ε) (p p' : P) (E : V → Prop) {η η' : ℝ≥0∞}
    (hA3 : Assumptions.UniformRandomBytesTest L t (fun b => E (reg p (e b))) η)
    (hA3' : Assumptions.UniformRandomBytesTest L t (fun b => E (reg p' (e b))) η') :
    prCoin (fun ω => E (reg p (e (t ω)))) ≤ prCoin (fun ω => E (reg p' (e (t ω)))) + ε + η + η' :=
  calc prCoin (fun ω => E (reg p (e (t ω))))
      ≤ prCoin (fun y => E (reg p y)) + η := (prCoin_os e (fun y => E (reg p y)) hA3).1
    _ ≤ prCoin (fun y => E (reg p' y)) + ε + η := add_le_add (h p p' E) le_rfl
    _ ≤ prCoin (fun ω => E (reg p' (e (t ω)))) + η' + ε + η :=
        add_le_add (add_le_add (prCoin_os e (fun y => E (reg p' y)) hA3').2 le_rfl) le_rfl
    _ = _ := by ring

theorem prCoin_unit_snd {C : Type} [Fintype C] (P : C → Prop) : prCoin (fun p : Unit × C => P p.2) = prCoin P := by
  rw [prCoin_prod_split fun (_ : Unit) c => P c]
  exact avg_const _

theorem hides_leaves {P Y B C D ι : Type} [Fintype Y] [Nonempty Y] [AddCommGroup B] [Fintype B] [Fintype ι]
    [DecidableEq ι] (My : Y → B) (c : Y → C) {δ₁ : ℝ≥0∞} (hT1 : Claims.Hm96Hiding My c δ₁) (H : B × C → D)
    (x : P → ι → B) :
    PrivateCircuit.Hides (fun p (y : ι → Y) i => H (x p i + My (y i), c (y i))) (2 * Fintype.card ι * δ₁) := by
  intro p p' E
  have h1 := ideal_leaves_swap_fintype My c hT1 H (Ω := Unit) (fun _ => x p) fun _ l => E l
  have h2 := real_leaves_swap_fintype My c hT1 H (Ω := Unit) (fun _ => x p') fun _ l => E l
  rw [prCoin_unit_snd fun y : ι → Y => E fun i => H (x p i + My (y i), c (y i))] at h1
  rw [prCoin_unit_snd fun y : ι → Y => E fun i => H (x p' i + My (y i), c (y i))] at h2
  calc _ ≤ _ := h1
    _ ≤ _ := add_le_add h2 le_rfl
    _ = _ := by ring

theorem hides_leaves_os {P Ω : Type} [Fintype Ω] {n : ℕ} {t : Ω → Fin n → Fin 192 → Fin 256} (x : P → Fin n → Bits)
    (p p' : P) (E : (Fin n → ByteArray) → Prop) {η η' : ℝ≥0∞}
    (hA3 : Assumptions.UniformRandomBytesTest (fun _ : Fin n => 192) t
      (fun b => E fun i => leafH (x p i + My (saltOfBytes (b i)), saltHash (saltOfBytes (b i)))) η)
    (hA3' : Assumptions.UniformRandomBytesTest (fun _ : Fin n => 192) t
      (fun b => E fun i => leafH (x p' i + My (saltOfBytes (b i)), saltHash (saltOfBytes (b i)))) η') :
    prCoin (fun ω => E fun i => leafH (x p i + My (saltOfBytes (t ω i)), saltHash (saltOfBytes (t ω i)))) ≤
      prCoin (fun ω => E fun i => leafH (x p' i + My (saltOfBytes (t ω i)), saltHash (saltOfBytes (t ω i)))) +
        2 * n * ENNReal.ofReal (CROnly.hidingGap My saltHash) + η + η' := by
  have h := hides_leaves My saltHash (CROnly.hm96Hiding_gap My saltHash) leafH x
  rw [Fintype.card_fin] at h
  exact hides_os (saltsEquiv n) h p p' E hA3 hA3'

end FlockSoundness.Discharge.ZkView
