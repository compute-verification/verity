import Properties.Pous.PeriodGame
import Proofs.Pous.PeriodAt
import Proofs.Accounting.Prob

/-!
# Proof of PoUS's per-period guarantee in the shared coin game

**At uniform streams** (`periodHeldCoins`, `periodHeldCoinsAt`). `PeriodHeld`'s probability is `Pous.pr`'s fraction
(`Accounting.pr_eq_toReal`), which is `Verity.Game.prCoin` (`pr_eq_prCoin`). Where the session, the device record and
the schedule don't read the primitive, the event at `(ω, y)` is the event at `(ω₀, y)` for every `ω`, so over
`(ω, y)` it has the probability it has over `y` at any one `ω₀` (`prCoin_snd`), and the ring model's bound is the coin
game's (`prCoin_le_of_pr`).

**At the deployed challenger's streams** (`periodHeldSha512`, `periodHeldSha512At`). `derive` is SHA-512 keyed by the
64-byte source after `derivePre`, at the one input `deriveTail` (`derive_eq`), and its key is the digest's first 32
bytes (`keyOf_eq'`); block `c` of stream `j` is SHA-512 keyed by that key after `STREAM`, at `streamInputs (j, c)`
(`streamsOf_eq`), 37 bytes each and distinct (`streamInputs_length`, `streamInputs_inj`), so none is a prefix of another.
`Sha512Prf` at the key test replaces the derived key by the first 32 bytes of a uniform digest, uniform bytes; at the
streams' test it replaces the blocks by uniform digests, which lay out as uniform streams (`blocksEquiv`). Each step adds
its `ε` (`prCoin_challenger_le`), on top of the bound at uniform streams.
-/

namespace Pous.Period.SecurityProofs

open Pous.EndToEnd PousCheck PousCheck.Properties MemoryChallenger FlockSoundness.Randomness Verity.Game
open scoped ENNReal

set_option exponentiation.threshold 20000

/-! ## The coin game's probability -/

theorem prCoin_congr_equiv {A B : Type} [Fintype A] [Fintype B] (e : A ≃ B) {p : A → Prop} {q : B → Prop}
    (h : ∀ a, p a ↔ q (e a)) : prCoin p = prCoin q := by
  classical
  unfold prCoin avg
  rw [Fintype.card_congr e]
  exact Fintype.sum_equiv e _ _ fun a => by simp only [h a]

theorem prCoin_fst {Y Z : Type} [Fintype Y] [Fintype Z] [Nonempty Z] (p : Y → Prop) :
    prCoin (fun x : Y × Z => p x.1) = prCoin p := by
  classical
  unfold prCoin avg
  rw [Fintype.sum_prod_type, Fintype.card_prod]
  refine Finset.sum_congr rfl fun y _ => ?_
  change ∑ _z : Z, ((Fintype.card Y * Fintype.card Z : ℕ) : ℝ≥0∞)⁻¹ * (if p y then 1 else 0) = _
  rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
  have hZ : (Fintype.card Z : ℝ≥0∞) ≠ 0 := by exact_mod_cast Fintype.card_ne_zero
  have hZ' : (Fintype.card Z : ℝ≥0∞) ≠ ⊤ := ENNReal.natCast_ne_top _
  rw [Nat.cast_mul, ENNReal.mul_inv (Or.inr hZ') (Or.inl (ENNReal.natCast_ne_top _))]
  calc (Fintype.card Z : ℝ≥0∞) * ((Fintype.card Y : ℝ≥0∞)⁻¹ * (Fintype.card Z : ℝ≥0∞)⁻¹ * (if p y then 1 else 0))
      = (Fintype.card Y : ℝ≥0∞)⁻¹ * (if p y then 1 else 0) * ((Fintype.card Z : ℝ≥0∞) * (Fintype.card Z : ℝ≥0∞)⁻¹) := by
        ring
    _ = _ := by rw [ENNReal.mul_inv_cancel hZ hZ', mul_one]

theorem prCoin_snd {Y Z : Type} [Fintype Y] [Fintype Z] [Nonempty Y] (p : Z → Prop) :
    prCoin (fun x : Y × Z => p x.2) = prCoin p :=
  (prCoin_congr_equiv (Equiv.prodComm Y Z) fun _ => Iff.rfl).trans (prCoin_fst p)

theorem prCoin_ne_top {C : Type} [Fintype C] (p : C → Prop) : prCoin p ≠ ⊤ := by
  classical
  unfold prCoin avg
  refine ENNReal.sum_ne_top.2 fun c _ => ENNReal.mul_ne_top ?_ ?_
  · by_cases h : Fintype.card C = 0
    · exfalso; exact (Fintype.card_eq_zero_iff.1 h).elim c
    · exact ENNReal.inv_ne_top.2 (by exact_mod_cast h)
  · by_cases h : p c <;> simp [h]

theorem pr_eq_prCoin {C : Type} [Fintype C] (p : C → Prop) : Pous.pr p = prCoin p := by
  classical
  unfold Pous.pr prCoin avg
  rw [ENNReal.div_eq_inv_mul, Finset.card_filter, Nat.cast_sum, Finset.mul_sum]
  refine Finset.sum_congr rfl fun c _ => ?_
  by_cases hc : p c <;> simp [hc]

/-- A ring-model bound on an event that doesn't read the primitive is the coin game's bound at any primitive. -/
theorem prCoin_le_of_pr {Ω Y : Type} [Fintype Ω] [Fintype Y] (E : Ω × Y → Prop) {b : ℝ} (h : Pouw.pr E ≤ b)
    (hE : ∀ ω ω' y, E (ω, y) ↔ E (ω', y)) (ω : Ω) : prCoin (fun y => E (ω, y)) ≤ ENNReal.ofReal b := by
  have : Nonempty Ω := ⟨ω⟩
  have h1 : prCoin E = prCoin (fun y => E (ω, y)) := by
    rw [← prCoin_snd (Y := Ω) fun y => E (ω, y)]
    refine prCoin_congr_equiv (Equiv.refl _) fun x => ?_
    exact hE x.1 ω x.2
  rw [Accounting.pr_eq_toReal, pr_eq_prCoin, h1] at h
  rw [← ENNReal.ofReal_toReal (prCoin_ne_top _)]
  exact ENNReal.ofReal_le_ofReal h

theorem ofReal_bound : ENNReal.ofReal ((Params.δ : ℝ) + Params.εMax.toReal) ≤ ENNReal.ofReal Params.δ + Params.εMax :=
  ENNReal.ofReal_add_le.trans (by gcongr; exact ENNReal.ofReal_toReal_le)

/-! ## The memory challenger's derivation, as SHA-512 on messages -/

theorem keyOf_eq (d : Hash.Digest) : keyOf d = ⟨Array.ofFn fun i : Fin 32 => d (Fin.castAdd 32 i)⟩ := by
  unfold keyOf
  ext1
  rw [ByteArray.data_extract]
  apply Array.ext
  · simp
  · intro i h1 h2
    simp [Fin.castAdd]

theorem fold_toList {α : Type} (f : α → ByteArray) (l : List α) (a : ByteArray) :
    (l.foldl (fun acc c => acc ++ f c) a).data.toList = a.data.toList ++ l.flatMap (fun c => (f c).data.toList) := by
  induction l generalizing a with
  | nil => simp
  | cons x l ih => simp [ih, ByteArray.data_append, List.append_assoc]

theorem flatMap_range_getElem? {α : Type} (g : ℕ → List α) (hg : ∀ c, (g c).length = 64) :
    ∀ (n b : ℕ), b < 64 * n → ((List.range n).flatMap g)[b]? = (g (b / 64))[b % 64]? := by
  intro n
  induction n with
  | zero => intro b hb; omega
  | succ n ih =>
    intro b hb
    have hlen : ((List.range n).flatMap g).length = 64 * n := by
      simp [List.length_flatMap, hg, mul_comm]
    rw [List.range_succ, List.flatMap_append]
    by_cases hbn : b < 64 * n
    · rw [List.getElem?_append_left (by omega)]; exact ih b hbn
    · rw [List.getElem?_append_right (by omega), hlen]
      simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil]
      have h1 : b / 64 = n := by omega
      have h2 : b % 64 = b - 64 * n := by omega
      rw [h1, h2]

theorem sha512_size (x : ByteArray) : (Flock.Sha512.hash x).size = 64 := by
  unfold Flock.Sha512.hash; exact Array.size_ofFn

theorem hash_ofFn (x : ByteArray) :
    (⟨Array.ofFn fun i : Fin 64 => (Flock.Sha512.hash x).data.getD i.val 0⟩ : ByteArray) = Flock.Sha512.hash x := by
  have hs := sha512_size x
  ext1
  apply Array.ext
  · simpa using hs.symm
  · intro i h1 h2
    simp only [Array.getElem_ofFn, Array.getD]
    split
    · rfl
    · exact absurd h2 ‹_›

theorem hash_digest (m : Hash.Msg) : (⟨Array.ofFn (sha512Digest m)⟩ : ByteArray) = Flock.Sha512.hash ⟨m.toArray⟩ := by
  unfold sha512Digest; exact hash_ofFn ⟨m.toArray⟩

theorem ofList_append (a : ByteArray) (l : List UInt8) (b : ByteArray) :
    (⟨(a.data.toList ++ l ++ b.data.toList).toArray⟩ : ByteArray) = a ++ ⟨l.toArray⟩ ++ b := by
  apply ByteArray.ext
  simp only [ByteArray.data_append]
  simp

theorem natBytes_lt (j : ℕ) (hj : j < 256) : natBytes j = ⟨#[j.toUInt8]⟩ := by
  have hlen : max 1 (((if j = 0 then 0 else Nat.log2 j + 1) + 7) / 8) = 1 := by
    split
    · rfl
    · have : Nat.log2 j < 8 := (Nat.log2_lt (by assumption)).2 (by simpa using hj)
      omega
  unfold natBytes
  simp only [hlen]
  apply ByteArray.ext
  apply Array.ext
  · simp
  · intro i h1 h2; simp at h2; subst h2; simp

theorem be8_size (x : ℕ) : (be8 x).size = 8 := by simp [be8, ByteArray.size]

theorem be8_last (c : ℕ) : (be8 c).data.toList[7]? = some c.toUInt8 := by
  simp [be8]

/-- What every `streamInputs` starts with: the index's frame header, the frame of `"u"` and the integer frame's header. -/
def streamHeader : ByteArray := ⟨#[98]⟩ ++ be8 20 ++ frame (.str "u") ++ ⟨#[105]⟩ ++ be8 1

theorem streamHeader_size : streamHeader.size = 28 := by decide +kernel

theorem streamHeader_length : streamHeader.data.toList.length = 28 := by decide +kernel

theorem streamInputs_eq (jc : Fin 105 × Fin 64) :
    streamInputs jc = streamHeader.data.toList ++ [jc.1.val.toUInt8] ++ (be8 jc.2.val).data.toList := by
  have hn := natBytes_lt jc.1.val (by omega)
  have hu : (frame (.str "u")).size = 10 := by decide +kernel
  have hi : frame (.int jc.1.val) = ⟨#[105]⟩ ++ be8 1 ++ ⟨#[jc.1.val.toUInt8]⟩ := by
    simp only [frame, hn]; rfl
  have hb : (ByteArray.empty ++ frame (.str "u") ++ frame (.int jc.1.val)).size = 20 := by
    rw [hi]; simp only [ByteArray.size_append, ByteArray.size_empty, hu, be8_size]; rfl
  have hf : frames [.str "u", .int jc.1.val] =
      ⟨#[98]⟩ ++ be8 20 ++ (ByteArray.empty ++ frame (.str "u") ++ frame (.int jc.1.val)) := by
    simp only [frames, List.foldl]
    conv_lhs => rw [frame]
    rw [hb]
  unfold streamInputs
  rw [hf, hi, streamHeader]
  simp only [ByteArray.empty_append, ByteArray.append_assoc, ByteArray.data_append, Array.toList_append,
    List.append_assoc]

theorem streamInputs_length (jc : Fin 105 × Fin 64) : (streamInputs jc).length = 37 := by
  rw [streamInputs_eq]
  simp [streamHeader_size, be8_size]

theorem streamInputs_inj {a b : Fin 105 × Fin 64} (h : streamInputs a = streamInputs b) : a = b := by
  have h28 : ∀ jc : Fin 105 × Fin 64, (streamInputs jc)[28]? = some jc.1.val.toUInt8 := by
    intro jc
    rw [streamInputs_eq, List.append_assoc, List.getElem?_append_right (by rw [streamHeader_length]),
      streamHeader_length]
    rfl
  have h36 : ∀ jc : Fin 105 × Fin 64, (streamInputs jc)[36]? = some jc.2.val.toUInt8 := by
    intro jc
    rw [streamInputs_eq, List.getElem?_append_right (by simp [streamHeader_size])]
    simpa [streamHeader_size] using be8_last jc.2.val
  have e1 := (h28 a).symm.trans (h ▸ h28 b)
  have e2 := (h36 a).symm.trans (h ▸ h36 b)
  simp only [Option.some.injEq] at e1 e2
  have u : ∀ (x y : ℕ), x < 256 → y < 256 → x.toUInt8 = y.toUInt8 → x = y := by
    intro x y hx hy hxy
    have := congrArg UInt8.toNat hxy
    simpa [UInt8.toNat_ofNat, Nat.mod_eq_of_lt hx, Nat.mod_eq_of_lt hy] using this
  exact Prod.ext (Fin.ext (u _ _ (by omega) (by omega) e1)) (Fin.ext (u _ _ (by omega) (by omega) e2))

theorem block_length (key : ByteArray) (idx : List Part) (c : ℕ) : (block key idx c).data.toList.length = 64 := by
  rw [Array.length_toList, ByteArray.size_data, block, sha512_size]

theorem fold_blocks (key : ByteArray) (idx : List Part) :
    ((List.range 64).foldl (fun acc c => acc ++ block key idx c) ByteArray.empty).data.toList =
      (List.range 64).flatMap (fun c => (block key idx c).data.toList) := by
  rw [fold_toList, ByteArray.data_empty, Array.toList_empty, List.nil_append]

theorem fold_blocks_size (key : ByteArray) (idx : List Part) :
    ((List.range 64).foldl (fun acc c => acc ++ block key idx c) ByteArray.empty).data.size = 4096 := by
  rw [← Array.length_toList, fold_blocks, List.length_flatMap]
  simp only [block_length, List.map_const', List.length_range, List.sum_replicate, smul_eq_mul]

theorem stream_eq (key : ByteArray) (idx : List Part) :
    stream key idx 4096 =
      ((List.range 64).foldl (fun acc c => acc ++ block key idx c) ByteArray.empty).extract 0 4096 := by
  unfold stream; rfl

theorem stream_getElem? (key : ByteArray) (idx : List Part) (b : ℕ) (hb : b < 4096) :
    (stream key idx 4096).data[b]? = (block key idx (b / 64)).data[b % 64]? := by
  rw [stream_eq, ByteArray.data_extract, Array.getElem?_extract, fold_blocks_size, ← Array.getElem?_toList,
    fold_blocks, ← Array.getElem?_toList]
  rw [ite_eq_left_of_eq_true _ _ (eq_true (by omega)), Nat.zero_add]
  exact flatMap_range_getElem? _ (block_length key idx) 64 b (by omega)

theorem stream_getD (key : ByteArray) (idx : List Part) (b : ℕ) (hb : b < 4096) :
    (stream key idx 4096).data.getD b 0 = (block key idx (b / 64)).data.getD (b % 64) 0 := by
  rw [Array.getD_eq_getD_getElem?, Array.getD_eq_getD_getElem?, stream_getElem? key idx b hb]

theorem derive_eq (N : ℕ) (sid : ByteArray) (src : Bytes 64) :
    derive (bytesOf src) COIN_DOMAIN (coinContext N sid) =
      keyOf (sha512Digest (derivePre ++ List.ofFn (fun i => (src i).val.toUInt8) ++ deriveTail N sid)) := by
  have hs : (bytesOf src).size = 64 := by simp [bytesOf, ByteArray.size]
  unfold keyOf
  rw [hash_digest]
  unfold derivePre deriveTail derive
  rw [ofList_append, List.toArray_ofFn]
  simp only [frame, hs, ByteArray.append_assoc]
  rfl

/-- What block `jc.2` of stream `jc.1` hashes under the key `K`. -/
def blockMsg (K : Fin 32 → UInt8) (jc : Fin 105 × Fin 64) : Hash.Msg :=
  STREAM.data.toList ++ List.ofFn K ++ streamInputs jc

theorem block_eq (K : Fin 32 → UInt8) (jc : Fin 105 × Fin 64) :
    block ⟨Array.ofFn K⟩ [.str "u", .int jc.1.val] jc.2.val = ⟨Array.ofFn (sha512Digest (blockMsg K jc))⟩ := by
  rw [hash_digest]
  unfold block blockMsg streamInputs
  apply congrArg Flock.Sha512.hash
  rw [ofList_append, List.toArray_ofFn, ByteArray.append_assoc, ByteArray.append_assoc]

theorem streamsOf_eq (K : Fin 32 → UInt8) :
    streamsOf ⟨Array.ofFn K⟩ = streamsOfBlocks (fun jc => sha512Digest (blockMsg K jc)) := by
  funext j b
  have hb : b.val < 4096 := b.isLt
  unfold streamsOf streamsOfBlocks
  change ((stream _ _ 4096).data.getD b.val 0).toFin = _
  rw [stream_getD _ _ _ hb]
  have := block_eq K (j, ⟨b.val / 64, by omega⟩)
  simp only at this
  rw [this, Array.getD_eq_getD_getElem?, Array.getElem?_ofFn]
  simp [Nat.mod_lt]

/-! ## The two tests of `Sha512Prf` -/

/-- A byte as the code holds it. -/
def byteEquiv : Fin 256 ≃ UInt8 where
  toFun x := x.val.toUInt8
  invFun u := u.toFin
  left_inv x := by apply Fin.ext; simp
  right_inv u := by simp

/-- Uniform digests at the 105 · 64 block inputs are uniform streams, laid out in counter mode. -/
def blocksEquiv : (Fin 105 × Fin 64 → Hash.Digest) ≃ Streams where
  toFun := streamsOfBlocks
  invFun s jc i := byteEquiv (s jc.1 ⟨jc.2.val * 64 + i.val, by unfold DRAW_BYTES; omega⟩)
  left_inv U := by
    funext jc i
    simp only [streamsOfBlocks]
    have h1 : (jc.2.val * 64 + i.val) / 64 = jc.2.val := by omega
    have h2 : (jc.2.val * 64 + i.val) % 64 = i.val := by omega
    simp only [h1, h2]
    exact byteEquiv.apply_symm_apply _
  right_inv s := by
    funext j b
    simp only [streamsOfBlocks]
    have : b.val / 64 * 64 + b.val % 64 = b.val := Nat.div_add_mod' b.val 64
    simp only [this]
    exact byteEquiv.symm_apply_apply _

/-- A digest's first 32 bytes and its last 32. -/
def digestSplit : Hash.Digest ≃ (Fin 32 → UInt8) × (Fin 32 → UInt8) :=
  (Equiv.arrowCongr (finSumFinEquiv : Fin 32 ⊕ Fin 32 ≃ Fin 64).symm (Equiv.refl UInt8)).trans
    (Equiv.sumArrowEquivProdArrow _ _ _)

theorem digestSplit_fst (d : Hash.Digest) : (digestSplit d).1 = fun i => d (Fin.castAdd 32 i) := by
  funext i
  simp only [digestSplit, Equiv.trans_apply, Equiv.arrowCongr_apply, Equiv.sumArrowEquivProdArrow_apply_fst,
    Function.comp_apply, Equiv.symm_symm, Equiv.refl_apply]
  exact congrArg d (finSumFinEquiv_apply_left i)

theorem keyOf_eq' (d : Hash.Digest) : keyOf d = ⟨Array.ofFn (digestSplit d).1⟩ := by
  rw [keyOf_eq, digestSplit_fst]

/-- **The bound at the deployed challenger's streams**: a bound `b` at uniform streams, plus `Sha512Prf`'s `ε` at the
key test (one input) and at the streams' test (105 · 64 inputs). -/
theorem prCoin_challenger_le (N : ℕ) (sid : ByteArray) (F : HmCoins → Streams → Prop) {b : ℝ≥0∞}
    (hb : prCoin (fun y : HmCoins × Streams => F y.1 y.2) ≤ b) (ε : ℕ → ℕ → ℝ≥0∞) (t : ℕ)
    (h1 : Hash.Assumptions.Sha512Prf sha512Digest ε derivePre 64 (fun _ : Unit => deriveTail N sid)
      (fun (c : HmCoins) k => F c (streamsOf (keyOf (k ())))) t)
    (h2 : Hash.Assumptions.Sha512Prf sha512Digest ε STREAM.data.toList 32 streamInputs
      (fun (c : HmCoins) U => F c (streamsOfBlocks U)) t) :
    prCoin (fun z : Bytes 64 × HmCoins => F z.2 (challengerStreams N sid z.1)) ≤ b + ε t 1 + ε t (105 * 64) := by
  have hpre : derivePre.length = 37 := by decide +kernel
  have hst : STREAM.data.toList.length = 28 := by decide +kernel
  have r1 : prCoin (fun z : Bytes 64 × HmCoins => F z.2 (challengerStreams N sid z.1)) =
      prCoin (fun z : (Fin 64 → UInt8) × HmCoins =>
        F z.2 (streamsOf (keyOf (sha512Digest (derivePre ++ List.ofFn z.1 ++ deriveTail N sid))))) :=
    prCoin_congr_equiv ((Equiv.arrowCongr (Equiv.refl _) byteEquiv).prodCongr (Equiv.refl _)) fun z => by
      simp only [challengerStreams, derive_eq]; rfl
  have p1 := h1 (by norm_num) (by rw [hpre]; norm_num) (fun _ _ _ => rfl)
  rw [Fintype.card_unit] at p1
  have r2 : prCoin (fun z : (Unit → Hash.Digest) × HmCoins => F z.2 (streamsOf (keyOf (z.1 ())))) =
      prCoin (fun z : (Fin 32 → UInt8) × HmCoins => F z.2 (streamsOf ⟨Array.ofFn z.1⟩)) := by
    refine Eq.trans ?_ (prCoin_fst (Z := Fin 32 → UInt8)
      (fun z : (Fin 32 → UInt8) × HmCoins => F z.2 (streamsOf ⟨Array.ofFn z.1⟩)))
    refine prCoin_congr_equiv (((Equiv.funUnique Unit Hash.Digest).trans digestSplit).prodCongr (Equiv.refl _) |>.trans
      ((Equiv.prodAssoc _ _ _).trans ((Equiv.prodCongr (Equiv.refl _) (Equiv.prodComm _ _)).trans
        (Equiv.prodAssoc _ _ _).symm))) fun z => ?_
    simp only [keyOf_eq']
    rfl
  have r3 : prCoin (fun z : (Fin 32 → UInt8) × HmCoins => F z.2 (streamsOf ⟨Array.ofFn z.1⟩)) =
      prCoin (fun z : (Fin 32 → UInt8) × HmCoins =>
        F z.2 (streamsOfBlocks fun i => sha512Digest (STREAM.data.toList ++ List.ofFn z.1 ++ streamInputs i))) := by
    simp only [streamsOf_eq]; rfl
  have p2 := h2 le_rfl (by rw [hst]; norm_num) fun i j hij =>
    streamInputs_inj (hij.eq_of_length (by rw [streamInputs_length, streamInputs_length]))
  rw [Fintype.card_prod, Fintype.card_fin, Fintype.card_fin] at p2
  have r4 : prCoin (fun z : (Fin 105 × Fin 64 → Hash.Digest) × HmCoins => F z.2 (streamsOfBlocks z.1)) =
      prCoin (fun y : HmCoins × Streams => F y.1 y.2) :=
    prCoin_congr_equiv (blocksEquiv.prodCongr (Equiv.refl HmCoins) |>.trans (Equiv.prodComm Streams HmCoins))
      fun z => by simp only [Equiv.trans_apply, Equiv.prodCongr_apply, Equiv.prodComm_apply, Prod.map, Prod.swap,
        Equiv.refl_apply]; rfl
  have q1 : prCoin (fun z : (Fin 32 → UInt8) × HmCoins =>
        F z.2 (streamsOfBlocks fun i => sha512Digest (STREAM.data.toList ++ List.ofFn z.1 ++ streamInputs i))) ≤
      prCoin (fun z : (Fin 105 × Fin 64 → Hash.Digest) × HmCoins => F z.2 (streamsOfBlocks z.1)) + ε t (105 * 64) :=
    p2
  have q0 : prCoin (fun z : Bytes 64 × HmCoins => F z.2 (challengerStreams N sid z.1)) ≤
      prCoin (fun z : (Unit → Hash.Digest) × HmCoins => F z.2 (streamsOf (keyOf (z.1 ())))) + ε t 1 := by
    rw [r1]; exact p1
  rw [r2, r3, r4] at *
  calc _ ≤ _ := q0
    _ ≤ b + ε t (105 * 64) + ε t 1 := by gcongr; exact q1.trans (by gcongr)
    _ = b + ε t 1 + ε t (105 * 64) := add_right_comm _ _ _

/-! ## The four guarantees -/

theorem periodHeldCoins : PeriodHeldCoins := by
  intro hC D Q L hW ι α ΩW ΩQ _ _ d r P τ round S hdev hinst hS o hind ω
  refine (prCoin_le_of_pr _ (periodHeld hC D Q L hW ι α ΩW ΩQ d r P τ round S hdev hinst hS o)
    (fun ω ω' y => ?_) ω).trans ofReal_bound
  obtain ⟨hs, hd, hsch⟩ := hind ω ω' y
  simp only [Audits.Accepts, Audits.Break, Accounting.Run.held, hs, hd, hsch]

theorem periodHeldSha512 : PeriodHeldSha512 := by
  intro hC D Q L hW ι α ΩW ΩQ _ _ d r P τ round S hdev hinst hS o hind ω ε t h1 h2
  exact prCoin_challenger_le (2 ^ 19) P.sid (fun c s => AcceptedShort r P S (o.1, o.2, (ω, c, s)))
    (periodHeldCoins hC D Q L hW ι α ΩW ΩQ d r P τ round S hdev hinst hS o hind ω) ε t h1 h2

theorem periodHeldCoinsAt : PeriodHeldCoinsAt := by
  intro p ε₀ D Q L hnum hW pt N _ h1 h2 hH ι α ΩW ΩQ _ _ d r P round S hdev hS o hind ω
  refine (prCoin_le_of_pr _ (periodHeldAt p ε₀ D Q L hnum hW pt N h1 h2 hH ι α ΩW ΩQ d r P round S hdev hS o)
    (fun ω ω' y => ?_) ω).trans ofReal_bound
  obtain ⟨hs, hd, hsch⟩ := hind ω ω' y
  simp only [AuditsOf.Accepts, AuditsOf.Break, Accounting.Run.held, hs, hd, hsch]

theorem periodHeldSha512At : PeriodHeldSha512At := by
  intro p ε₀ D Q L hnum hW pt N _ h1 h2 hH ι α ΩW ΩQ _ _ d r P round S hdev hS o hind ω ε t k1 k2
  exact prCoin_challenger_le N P.sid (fun c s => AcceptedShortAt pt N r P S (o.1, o.2, (ω, c, s)))
    (periodHeldCoinsAt p ε₀ D Q L hnum hW pt N h1 h2 hH ι α ΩW ΩQ d r P round S hdev hS o hind ω) ε t k1 k2

end Pous.Period.SecurityProofs
