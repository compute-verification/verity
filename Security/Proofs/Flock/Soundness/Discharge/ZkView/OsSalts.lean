import Proofs.Flock.Soundness.Discharge.ZkView.Leaf
import Proofs.Flock.Soundness.Discharge.PrivateCircuit.Defs
import Proofs.Flock.Soundness.CROnly.Hiding
import Proofs.Flock.Soundness.ZK.RealLeaves
import Proofs.Flock.Soundness.ExecDraw

/-!
# Salts from the operating system (A3)

C-Flock reads every hm96 leaf salt from the operating system, 192 bytes a leaf (`getrandom`, the kernel's `getrandom(2)`;
Daniel, 9 Oct, decision 4(a)), as the recursive audit's firewall and a registrant do. The hiding and zero-knowledge
theorems draw each salt uniformly from `Salt`, hm96-sha512's 1536 salt bits. The step between the two is A3
(`Assumptions.UniformRandomBytes`, `uniform/io-getrandombytes`: the operating system's bytes are independent and
uniform), and no premise on a stream generator is left on the salts:

* `saltOfBytes`, `saltEquiv`: 192 bytes as a salt, in `bitsOf`'s order (bit `t` is bit `t % 8` of byte `t / 8`), a
  bijection. `saltBytes_saltOfBytes`: the bytes a leaf hashes for that salt are the OS's bytes themselves
  (`Audit.Law.streamOf`), so the executable's leaf under the OS's salt is the model's leaf at `saltOfBytes`
  (`leaf_os`, from `leafSplit`). `saltsEquiv n`: `n` salts from `n` streams of 192 bytes.
* `prCoin_os`, `prCoin_os_prod`: an event of values decoded bijectively from A3's bytes has the probability it has for
  uniform values, alone or beside any independent randomness.
* `hm96Hiding_os`: hm96 hiding (`Assumptions.Hm96Hiding`, a statement about uniform salts) at the OS's salt;
  `hm96Sha512_os`: at hm96-sha512's pinned key with no premise but A3, at the bound `hidingGap My saltHash`
  (`CROnly.hm96Hiding_gap`), which `hash-derived-key` says is at most `2^-193`.
* `ideal_leaves_swap_os`: T1 once per leaf (`ZK.ideal_leaves_swap`) for `n` leaves salted from `n` OS streams.
* `hides_os`: a registration that hides under uniform salts hides, as a `PrivateCircuit.Hides`, under the OS's.
  `hides_leaves`: a registration of hm96 leaves hides at `2·|ι|·δ₁` under uniform salts; `hides_leaves_os`: an
  hm96-sha512 one at the OS's salts, at `2·n·hidingGap My saltHash`, with no premise but A3.

No computational property of SHA-512 enters. hm96's hiding is the leftover hash lemma over the Hankel family
(`CROnly.sum_hidingGap_le`), which reads of the salt hash `c(y) = SHA-512(sp ‖ y)` only that it takes at most `2^512`
values; SHA-512's collision resistance (`cr/sha-512`) enters binding, not hiding.
-/

namespace FlockSoundness.Discharge.ZkView

open Game ZK
open scoped ENNReal

/-! ## 192 bytes as a salt -/

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

/-- `k` bytes whose bits are `b`'s are `b` as a byte stream. Stated for every `k`: at a literal `k` such as 192,
`Array.ofFn` unfolds element by element past the recursion limit. -/
theorem bytesOf_eq_streamOf {k : ℕ} (x : Fin (8 * k) → ZMod 2) (b : Fin k → Fin 256)
    (h : ∀ (i : Fin k) (j : Fin 8) (hij : 8 * i.val + j.val < 8 * k),
      x ⟨8 * i.val + j.val, hij⟩ = if (b i).val.testBit j.val then 1 else 0) :
    bytesOf x = Audit.Law.streamOf b := by
  unfold bytesOf Audit.Law.streamOf
  congr 1
  refine congrArg Array.ofFn (funext fun i => ?_)
  rw [← byteOf_testBit_fin (b i)]
  exact congrArg byteOf (funext fun j => h i j _)

/-- **The leaf's salt bytes are the OS's bytes**: the 192 bytes the executable's leaf hashes for `saltOfBytes b`
(`saltBytes`) are `b` itself, as a byte stream. -/
theorem saltBytes_saltOfBytes (b : Fin 192 → Fin 256) : saltBytes (saltOfBytes b) = Audit.Law.streamOf b :=
  bytesOf_eq_streamOf (k := 192) (saltOfBytes b) b fun i j h => saltOfBytes_apply b i j h

/-- **The executable's leaf at the OS's salt bytes** is the model's at `saltOfBytes`: for a 64-byte digest `x`,
`Default512.leaf x b = H(x ⊕ M·y, c(y))` with `y = saltOfBytes b` (`leafSplit`). -/
theorem leaf_os (x : ByteArray) (b : Fin 192 → Fin 256) (hx : x.size = 64) :
    Flock.Hm96.Default512.leaf x (Audit.Law.streamOf b) =
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

/-- **192 bytes are a salt**, one for one. -/
noncomputable def saltEquiv : (Fin 192 → Fin 256) ≃ Salt :=
  Equiv.ofBijective saltOfBytes ((Fintype.bijective_iff_injective_and_card _).2
    ⟨saltOfBytes_injective, by
      simp only [Fintype.card_fun, Fintype.card_fin, ZMod.card]
      rw [show (256 : ℕ) = 2 ^ 8 by norm_num, ← pow_mul]⟩)

theorem saltEquiv_apply (b : Fin 192 → Fin 256) : saltEquiv b = saltOfBytes b := rfl

/-- **`n` salts from `n` streams of 192 bytes**, stream `i` salting leaf `i`. -/
noncomputable def saltsEquiv (n : ℕ) : (Fin n → Fin 192 → Fin 256) ≃ (Fin n → Salt) :=
  Equiv.piCongrRight fun _ => saltEquiv

theorem saltsEquiv_apply {n : ℕ} (b : Fin n → Fin 192 → Fin 256) (i : Fin n) :
    saltsEquiv n b i = saltOfBytes (b i) := rfl

/-! ## A3's bytes as uniform values -/

/-- **A3's bytes, decoded one for one, are uniform**: for a byte source that meets A3, every event of the values `e`
decodes from its bytes has the probability it has for a uniform value. -/
theorem prCoin_os {Ω Y : Type} [Fintype Ω] [Fintype Y] {m : ℕ} {L : Fin m → ℕ}
    {t : Ω → (i : Fin m) → Fin (L i) → Fin 256} (hA3 : Assumptions.UniformRandomBytes L t)
    (e : ((i : Fin m) → Fin (L i) → Fin 256) ≃ Y) (E : Y → Prop) :
    prCoin (fun ω => E (e (t ω))) = prCoin E :=
  (hA3 fun b => E (e b)).trans (prCoin_congr_equiv e E)

/-- **The same beside independent randomness**: in an experiment on `A × Ω` whose `Ω` is the byte source, the values
decoded from it may be taken uniform. -/
theorem prCoin_os_prod {A Ω Y : Type} [Fintype A] [Fintype Ω] [Fintype Y] {m : ℕ} {L : Fin m → ℕ}
    {t : Ω → (i : Fin m) → Fin (L i) → Fin 256} (hA3 : Assumptions.UniformRandomBytes L t)
    (e : ((i : Fin m) → Fin (L i) → Fin 256) ≃ Y) (E : A → Y → Prop) :
    prCoin (fun p : A × Ω => E p.1 (e (t p.2))) = prCoin (fun p : A × Y => E p.1 p.2) := by
  rw [prCoin_prod_split (fun a ω => E a (e (t ω))), prCoin_prod_split E]
  exact congrArg avg (funext fun a => prCoin_os hA3 e (E a))

/-! ## hm96 hiding at the OS's salts -/

/-- **hm96 hiding at the OS's salt** (A3): where `Hm96Hiding My c δ₁` holds for a uniform salt, it holds for the salt
`e` decodes from a byte source that meets A3. -/
theorem hm96Hiding_os {Ω Y B C : Type} [Fintype Ω] [Fintype Y] [Fintype B] {m : ℕ} {L : Fin m → ℕ}
    {t : Ω → (i : Fin m) → Fin (L i) → Fin 256} (hA3 : Assumptions.UniformRandomBytes L t)
    (e : ((i : Fin m) → Fin (L i) → Fin 256) ≃ Y) (My : Y → B) (c : Y → C) {δ₁ : ℝ≥0∞}
    (hT1 : Assumptions.Hm96Hiding My c δ₁) (E : B × C → Prop) :
    prCoin (fun ω => E (My (e (t ω)), c (e (t ω)))) ≤ prCoin (fun p : B × Y => E (p.1, c p.2)) + δ₁ := by
  rw [prCoin_os hA3 e fun y => E (My y, c y)]
  exact hT1 E

/-- **hm96-sha512 hides at the OS's salt, with no premise but A3**: for the salt of one 192-byte stream of a byte
source that meets A3 (`uniform/io-getrandombytes`), no test tells `(M(κ₀)·y, SHA-512(sp ‖ y))` from `(U, SHA-512(sp ‖ y))`
with advantage above `hidingGap My saltHash`, the statistical distance at the pinned key `κ₀`. That this is at most
`2^-193` is `hash-derived-key`; at all but `2^-64` of the keys it is (`CROnly.hm96_sha512_bad_keys`). -/
theorem hm96Sha512_os {Ω : Type} [Fintype Ω] {t : Ω → Fin 1 → Fin 192 → Fin 256}
    (hA3 : Assumptions.UniformRandomBytes (fun _ : Fin 1 => 192) t) (E : Bits × ByteArray → Prop) :
    prCoin (fun ω => E (My (saltOfBytes (t ω 0)), saltHash (saltOfBytes (t ω 0)))) ≤
      prCoin (fun p : Bits × Salt => E (p.1, saltHash p.2)) + ENNReal.ofReal (CROnly.hidingGap My saltHash) :=
  hm96Hiding_os hA3 ((Equiv.funUnique (Fin 1) _).trans saltEquiv) My saltHash (CROnly.hm96Hiding_gap My saltHash) E

/-- **T1 for `n` leaves salted from the OS** (A3): in an experiment on `Ω` and a byte source of `n` streams that meets
A3, whose leaf `i` is `H(x_i(ω) + M·y_i, c(y_i))` with `y_i` decoded from stream `i`, replacing every leaf by an ideal
one `H(U_i, c(y_i))` changes the probability of any event by at most `n·δ₁`. The digests may read anything in `Ω`; the
streams are read by their leaves only. -/
theorem ideal_leaves_swap_os {Ω Ωos Y B C D : Type} [Fintype Ω] [Nonempty Ω] [Fintype Ωos] [Fintype Y] [Nonempty Y]
    [AddCommGroup B] [Fintype B] {n : ℕ} {L : Fin n → ℕ} {t : Ωos → (i : Fin n) → Fin (L i) → Fin 256}
    (hA3 : Assumptions.UniformRandomBytes L t) (e : ((i : Fin n) → Fin (L i) → Fin 256) ≃ (Fin n → Y))
    (My : Y → B) (c : Y → C) {δ₁ : ℝ≥0∞} (hT1 : Assumptions.Hm96Hiding My c δ₁) (H : B × C → D)
    (x : Ω → Fin n → B) (G : Ω → (Fin n → D) → Prop) :
    prCoin (fun p : Ω × Ωos => G p.1 fun i => H (x p.1 i + My (e (t p.2) i), c (e (t p.2) i)))
      ≤ prCoin (fun p : Ω × (Fin n → B × Y) => G p.1 fun i => H ((p.2 i).1, c (p.2 i).2)) + n * δ₁ := by
  rw [prCoin_os_prod hA3 e fun ω y => G ω fun i => H (x ω i + My (y i), c (y i))]
  exact ideal_leaves_swap My c hT1 H n x G

/-! ## A registration under the OS's salts -/

/-- **A registration that hides under uniform salts hides under the OS's** (A3): `reg` read at the salt `e` decodes
from a byte source that meets A3 is a registration on the source's outcomes, and it hides up to the same `ε`. -/
theorem hides_os {P V Ω Y : Type} [Fintype Ω] [Fintype Y] {m : ℕ} {L : Fin m → ℕ}
    {t : Ω → (i : Fin m) → Fin (L i) → Fin 256} (hA3 : Assumptions.UniformRandomBytes L t)
    (e : ((i : Fin m) → Fin (L i) → Fin 256) ≃ Y) {reg : P → Y → V} {ε : ℝ≥0∞} (h : PrivateCircuit.Hides reg ε) :
    PrivateCircuit.Hides (fun p (ω : Ω) => reg p (e (t ω))) ε := fun p p' E => by
  rw [prCoin_os hA3 e fun y => E (reg p y), prCoin_os hA3 e fun y => E (reg p' y)]
  exact h p p' E

theorem prCoin_unit_snd {C : Type} [Fintype C] (P : C → Prop) : prCoin (fun p : Unit × C => P p.2) = prCoin P := by
  rw [prCoin_prod_split fun (_ : Unit) c => P c]
  exact avg_const _

/-- **A registration of hm96 leaves hides**: one that commits to the program's `x p i` with the leaves
`H(x p i + M·y_i, c(y_i))` under uniform salts hides the program up to `2·|ι|·δ₁` (`PrivateCircuit.Hides`), by T1 to
ideal leaves at `p` and back at `p'`. -/
theorem hides_leaves {P Y B C D ι : Type} [Fintype Y] [Nonempty Y] [AddCommGroup B] [Fintype B] [Fintype ι]
    [DecidableEq ι] (My : Y → B) (c : Y → C) {δ₁ : ℝ≥0∞} (hT1 : Assumptions.Hm96Hiding My c δ₁) (H : B × C → D)
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

/-- **hm96-sha512 registrations hide at the OS's salts, with no premise but A3**: a registration whose leaf `i` is
`H(x p i ⊕ M(κ₀)·y_i, SHA-512(sp ‖ y_i))`, `y_i` the salt of stream `i` of a byte source that meets A3
(`uniform/io-getrandombytes`), hides the program up to `2·n·hidingGap(My, saltHash)`; under `hash-derived-key` that is
at most `2·n·2^-193`. -/
theorem hides_leaves_os {P Ω : Type} [Fintype Ω] {n : ℕ} {t : Ω → Fin n → Fin 192 → Fin 256}
    (hA3 : Assumptions.UniformRandomBytes (fun _ : Fin n => 192) t) (x : P → Fin n → Bits) :
    PrivateCircuit.Hides
      (fun p (ω : Ω) i => leafH (x p i + My (saltOfBytes (t ω i)), saltHash (saltOfBytes (t ω i))))
      (2 * n * ENNReal.ofReal (CROnly.hidingGap My saltHash)) := by
  have h := hides_leaves My saltHash (CROnly.hm96Hiding_gap My saltHash) leafH x
  rw [Fintype.card_fin] at h
  exact hides_os hA3 (saltsEquiv n) h

end FlockSoundness.Discharge.ZkView
