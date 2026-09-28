/-
`Arith.Correct.pinned_indep` (§9.2): the 128 equality weights of the seven pinned zerocheck coordinates are independent
over GF(2). The certificate is an inverse: for each `k`, the XOR of the weights `pinCols[k]` selects is `x^k` (checked by
the kernel on the executable's `eqTable Flock.pinned`). So the bit matrix `M` of the weights has a right inverse, hence a
left inverse, and `M c = 0` forces `c = 0`.
-/
import Proofs.Flock.Level3.ArithFacts
import Proofs.Flock.Level3.Assembly
import Mathlib.Data.Matrix.Mul
import Mathlib.LinearAlgebra.Matrix.NonsingularInverse

open Polynomial Flock

set_option maxRecDepth 20000

namespace FlockLevel3

/-- Column `k` of the inverse: bit `b` set iff weight `b` enters `x^k` (Gaussian elimination, offline). -/
def pinCols : Array F128 := #[
  ⟨0xffffffffffffffff, 0xffffffffffffffff⟩,
  ⟨0xd09b1bb02d6fc14e, 0xfe123fcdb7d8d470⟩,
  ⟨0xade34a979d08aea7, 0x0f30cecb310a3269⟩,
  ⟨0x643f3d89f3c421a0, 0x8407f6863f620884⟩,
  ⟨0x6f636818ef27bb19, 0xf534eb10618931e2⟩,
  ⟨0xb6a330b8ae672a8e, 0xed880f89e7767e27⟩,
  ⟨0x35749bd9c3ef5895, 0x6a9f50c599b91418⟩,
  ⟨0x993e395b007021ac, 0xab3f2f46ad422547⟩,
  ⟨0xe414fd76cc98af9d, 0xd47b053693e61cac⟩,
  ⟨0xd5acade75dffee43, 0x13dbf936ceb0d600⟩,
  ⟨0x0c113e90a9e27532, 0x2466ec90b6cf71c2⟩,
  ⟨0x787e05d90edff9ad, 0xbb16a2969aaaf24e⟩,
  ⟨0x2f2b84b155a7e18f, 0x2654518d8538df00⟩,
  ⟨0x70e718d40bf9e364, 0xb174223934ede201⟩,
  ⟨0x4c9945ac28f1254f, 0x9c5a21f9e44827cf⟩,
  ⟨0x8a9446e3f73ab102, 0x93105201ca26d52c⟩,
  ⟨0x00e2a328c7c47aa2, 0x6daa9798918457c3⟩,
  ⟨0x40ff92a2cbc63a05, 0xca600dbfdad8faf0⟩,
  ⟨0x2b2936c8eebe6118, 0x7b5a0a0a9af648f6⟩,
  ⟨0x560ccd4a47f73d57, 0xa925a787bdee6822⟩,
  ⟨0xe4c3d0c00b36f354, 0xf62c59ba6642ab0f⟩,
  ⟨0x3da4ca58dfc41eae, 0xd107fb5cd4fc28f6⟩,
  ⟨0x1841e5f8a1685623, 0x0362711cc7b45a37⟩,
  ⟨0x6a5adaa508bd5a20, 0x464c42154c4a8c42⟩,
  ⟨0x16b32009a526f508, 0x45868ff574882af1⟩,
  ⟨0xd4f6aea05e62a428, 0xab76311f4e3e22d7⟩,
  ⟨0x2c830628de1d2b9d, 0x9d655cd5efe18ce1⟩,
  ⟨0x894bdce03b6dbdeb, 0xb08fbd44c274927a⟩,
  ⟨0xf9c16f699cf674a6, 0x7b279f41dcc13500⟩,
  ⟨0xbe556c380fe28357, 0xd4b0ecc18a939ef0⟩,
  ⟨0x943de32427d32443, 0xd8320e2fbb328a5a⟩,
  ⟨0x5078771c8df5cfc3, 0xca24837bd766fa03⟩,
  ⟨0xaccc888edd4ba718, 0x3fd4ce0724f6bbe6⟩,
  ⟨0xcdf83289eaff4f96, 0x16faad1cdaf0c252⟩,
  ⟨0x2554362b09652b52, 0x697524e9e5ebf534⟩,
  ⟨0x922ce995eae89f8c, 0x65765de807b3b20d⟩,
  ⟨0x4639df66fcddbe0c, 0xf6fb3f6f92b451c9⟩,
  ⟨0xc7ab6caf348946ae, 0x06612ea82206ee54⟩,
  ⟨0xfd09eb3c498b5c40, 0x2954149dedce3973⟩,
  ⟨0xd5fd3fdcdc2eaac4, 0x05f80fe6f75bf0fa⟩,
  ⟨0x80e5bad08e8c4f60, 0x9448049a48c02c3b⟩,
  ⟨0x3ee7caac91ae340b, 0x616473eb3da4444b⟩,
  ⟨0xe95c9ee43a096c85, 0x94701d76378ba8bd⟩,
  ⟨0x87c3fe18148dd10b, 0x597d9f947620c222⟩,
  ⟨0x83e1ccc8582017ba, 0x1b30b7f4d59fe178⟩,
  ⟨0x89fce17c6de5263b, 0x2f796ccde6716c3f⟩,
  ⟨0x76923bf072992384, 0x60fae352a32ed13a⟩,
  ⟨0x8531dcf22d473d6c, 0x160267a61ce7b0c3⟩,
  ⟨0xecba4decb117d152, 0x91bf9e2ff299647c⟩,
  ⟨0x1cca909faa168d6b, 0xfab9243c77dcd5c9⟩,
  ⟨0x982e1de9643bde2e, 0xd48e1e2887c2347d⟩,
  ⟨0xef4c324ffe8e5d43, 0xe6c0e722ea98077c⟩,
  ⟨0x3945be64518d46bc, 0xe61be75cf05991b0⟩,
  ⟨0x30c995cb5296b9bc, 0x3d1b994631cf5d66⟩,
  ⟨0xfac64bf777534c5f, 0xe1727195baff90a2⟩,
  ⟨0xfcc55c106c087615, 0x00152ffddd174b27⟩,
  ⟨0xc55c106c08761500, 0x152ffddd174b27fc⟩,
  ⟨0xaac9f1cde0d4c472, 0x9818bbf6a61ee066⟩,
  ⟨0xbe665fd252a4d98b, 0xa12f7c602e28c312⟩,
  ⟨0x46ddb83c432c855d, 0x1ca6fc6fdf60fe4c⟩,
  ⟨0xfa714f9a81e3ed92, 0xedaa883653b75c78⟩,
  ⟨0xe70bb9f647284495, 0x4818ef71589b4b54⟩,
  ⟨0xd939c7a7eb0b1841, 0xb900151831e63463⟩,
  ⟨0x5acac6a5c1f4aa46, 0x083c42ba901720cd⟩,
  ⟨0x024a5528ef54ec09, 0xdc13f8d668a9a8f1⟩,
  ⟨0x4388032d8a42d04a, 0x9b775b8bfe8dee4d⟩,
  ⟨0xd42b9dbca45b98ab, 0xeaf25fea453653a1⟩,
  ⟨0xf220c52e1fc9513c, 0x44574c96934d96f8⟩,
  ⟨0x5e8f97ef919f4ec1, 0x665791b62a7b4205⟩,
  ⟨0xcef88719ebe78a2c, 0xf30a3f90a254e25f⟩,
  ⟨0x7b2975b4ae9bedce, 0x9b2633f5d185ae32⟩,
  ⟨0x755d04987d661cab, 0xbb9a21c54d762c99⟩,
  ⟨0x0c2ac5c2ec42e78f, 0xaad7025a59e6080d⟩,
  ⟨0xe9747a6fde075274, 0x20b4da8c96323f0f⟩,
  ⟨0x797c82576b87ea24, 0x19859f95d52d8073⟩,
  ⟨0x269142be8aaa1ff2, 0xa28e90510df9b5a9⟩,
  ⟨0x9a7ff6e02d01857d, 0x9977939ef606fe8e⟩,
  ⟨0x11fd6ceaa7bc0862, 0xd2e5d77e97cf4214⟩,
  ⟨0x6a587584b995e303, 0xc5730d8d0cce0341⟩,
  ⟨0x0bbbbaae460de1b8, 0xdc93056bb95f98f5⟩,
  ⟨0xb2678584d34f614a, 0x1b8ae65a08bdea44⟩,
  ⟨0x0fb5adf70293d63b, 0x9528c210c47a500e⟩,
  ⟨0x776cf3d5ae6eb1e2, 0x1d403a29262118f2⟩,
  ⟨0x52a6b882e38e8c7b, 0x178e0d81b3a88b7f⟩,
  ⟨0x624223d758ff18bb, 0x01377ace1c5bb19d⟩,
  ⟨0x5bbfc9addf4165e9, 0x2b980d52fa48f454⟩,
  ⟨0x51fc92bb56a5a40d, 0xf13e5d6d287f0530⟩,
  ⟨0x4d1febf8ac076b07, 0x979b53e3a38b13c1⟩,
  ⟨0xef09c08a7a612a2b, 0x960e54d28e8b05dc⟩,
  ⟨0xe0beaca93c79d8c2, 0x1feba6b12f647148⟩,
  ⟨0xb2daf8e1b4550bb2, 0x84cd0814af28e384⟩,
  ⟨0x8aa6005f7e0da919, 0x3fca79802b623134⟩,
  ⟨0xa770e32aacf14e96, 0x084d2a134e7a1074⟩,
  ⟨0xb86fda45eab03c09, 0xad7b51080599110c⟩,
  ⟨0xe392a794cc48f8dd, 0xd87ff721a3f23a39⟩,
  ⟨0xff3cc7f7162951c3, 0x87dd8d6317d69974⟩,
  ⟨0x472634f962bca139, 0x0b7250ea2d087a23⟩,
  ⟨0xc5072b8fc7b4ea29, 0xb63cf6b98fe3fdb6⟩,
  ⟨0xe353449d856c44e6, 0x8016deb3a08e94f8⟩,
  ⟨0x675c0497c73da890, 0x94b9ffac3fdcf2bd⟩,
  ⟨0x87598de52049c40b, 0x909f459c217ac2ac⟩,
  ⟨0xe69687fbd45a0a82, 0xc68054a135836c94⟩,
  ⟨0xeef6e7c7e90f1998, 0x0bf777800c20f623⟩,
  ⟨0x15d41504740c4b29, 0x331b9c98a76ffd1f⟩,
  ⟨0x79af303036928116, 0x495c51fa8ef6e198⟩,
  ⟨0x642c1f23719745a8, 0xe15c5d80fdb591cb⟩,
  ⟨0x16918816a8018115, 0x2e393aba971622b9⟩,
  ⟨0x02674fc0b667d36d, 0x3c4e576d65993e93⟩,
  ⟨0x4d805ee6a66043b6, 0xa85e998f4d650ca6⟩,
  ⟨0x71cc62d4bc11ccbd, 0x91f092044cdf4622⟩,
  ⟨0x6ae5a892ac0b626b, 0xb5b50f8231fe8b54⟩,
  ⟨0x2a6f7bf25a0f7fc6, 0x2dd2bbdf45e8cd0d⟩,
  ⟨0xd72b8936d872794d, 0xf3f26c6d636631f5⟩,
  ⟨0xa8275a873b688cce, 0x6375ce34e356042b⟩,
  ⟨0x1bef89bfbc030e4c, 0xbd125f349c221a47⟩,
  ⟨0xe8c2a60b499df2cf, 0x6ad34f34e6f0bbd7⟩,
  ⟨0x2f03ebd172da7bac, 0xe720de39e4edea9a⟩,
  ⟨0x6f003e1d25f27755, 0x1ac03f1c244d1e46⟩,
  ⟨0x71922af49fdc17d2, 0xc313477295773be5⟩,
  ⟨0x97818e80cf34c2f8, 0xf4a3464deb47515a⟩,
  ⟨0x4dd9b66d9d471567, 0x66c7914d88efafe9⟩,
  ⟨0x98d9051533bc2c2c, 0x630ac43236b90e4c⟩,
  ⟨0xe5b01bb768a3ec4c, 0xc21859e173287d77⟩,
  ⟨0xac2cd38290968211, 0xe35f16e515f8aaf8⟩,
  ⟨0x247e1506e9749fdc, 0x15adc2ce83c4c31d⟩,
  ⟨0x88cc9b2ce25e1872, 0x1a27a86229fa0187⟩,
  ⟨0xbd371b3333b330d2, 0x2484397f2268fa02⟩,
  ⟨0x5e5ba6435f9a19ad, 0x59c34d023d2132ff⟩]

/-- The XOR of the entries of `es` below `n` that `mask` selects. -/
def xorSel (mask : F128) (es : Array F128) : ℕ → F128
  | 0 => 0
  | n + 1 => xorSel mask es n + (if F128.bit mask n then es[n]! else 0)

def pinCertOK : Bool :=
  (List.range 128).all fun k => decide (xorSel pinCols[k]! (eqTable Flock.pinned) 128 = F128.monomial k)

theorem pinCert : pinCertOK = true := by decide +kernel

theorem xorSel_eq (mask : F128) (es : Array F128) (n : ℕ) :
    GF128.of (xorSel mask es n) = ∑ b ∈ Finset.range n, if F128.bit mask b then GF128.of es[b]! else 0 := by
  induction n with
  | zero => simp [xorSel, GF128.zero_eq]
  | succ n ih =>
    rw [xorSel, ← GF128.add_eq, ih, Finset.sum_range_succ]
    congr 1
    split_ifs <;> simp [GF128.zero_eq]

/-- `eq(pinned, b)`, the soundness model's `eqAt A.pinned b`. -/
noncomputable def eqAtPinned (b : ℕ) : GF128 := ∏ j : Fin 7, if b.testBit j then pinnedF j else 1 + pinnedF j

theorem pinned_size : Flock.pinned.size = 7 := by decide +kernel

theorem eqTable_pinned_size : (eqTable Flock.pinned).size = 128 := by rw [eqTable_size, pinned_size]; rfl

theorem eqAtPinned_eq (b : ℕ) (hb : b < 128) : eqAtPinned b = GF128.of (eqTable Flock.pinned)[b]! := by
  have hb' : b < (eqTable Flock.pinned).size := by rw [eqTable_pinned_size]; exact hb
  rw [getElem!_pos (eqTable Flock.pinned) b hb', eqTable_get]
  refine Fintype.prod_equiv (finCongr pinned_size.symm) _ _ (fun j => ?_)
  simp only [eqFactor, pinnedF, finCongr_apply, Fin.val_cast]
  rw [getElem!_pos Flock.pinned j.val (by rw [pinned_size]; exact j.isLt)]
  rfl

theorem unpack_zero : unpack 0 = 0 := by
  funext v
  show (F128.poly (GF128.val 0)).coeff v = 0
  rw [show GF128.val 0 = 0 from GF128.zero_eq, F128.poly_zero]; rfl

/-- `unpack` as an additive map. -/
noncomputable def unpackHom : GF128 →+ (Fin 128 → ZMod 2) := ⟨⟨unpack, unpack_zero⟩, unpack_add⟩

theorem monomial_w128 : ∀ k : Fin 128, w128 (F128.monomial k).lo (F128.monomial k).hi = 1 <<< k.val := by decide +kernel

theorem unpack_monomial (k : Fin 128) (i : Fin 128) : unpack (GF128.of (F128.monomial k)) i = if i = k then 1 else 0 := by
  show (F128.poly (F128.monomial k)).coeff i = _
  rw [F128.poly, monomial_w128, natPoly_shiftLeft, natPoly_one, mul_one, coeff_X_pow]
  simp [Fin.ext_iff, eq_comm]

/-- Row `i`, column `b`: bit `i` of weight `b`. -/
noncomputable def pinM : Matrix (Fin 128) (Fin 128) (ZMod 2) := fun i b => unpack (eqAtPinned b) i

def pinN : Matrix (Fin 128) (Fin 128) (ZMod 2) := fun b k => if F128.bit pinCols[k.val]! b then 1 else 0

theorem pinCert_k (k : Fin 128) : xorSel pinCols[k.val]! (eqTable Flock.pinned) 128 = F128.monomial k := by
  have h := pinCert
  unfold pinCertOK at h
  exact of_decide_eq_true (List.all_eq_true.mp h k.val (List.mem_range.mpr k.isLt))

theorem pinM_apply (i b : Fin 128) : pinM i b = unpack (eqAtPinned b) i := rfl
theorem pinN_apply (b k : Fin 128) : pinN b k = if F128.bit pinCols[k.val]! b then 1 else 0 := rfl

attribute [irreducible] pinM pinN eqAtPinned

theorem unpack_ite (p : Prop) [Decidable p] (a : GF128) (i : Fin 128) :
    unpack (if p then a else 0) i = if p then unpack a i else 0 := by
  split_ifs <;> simp [unpack_zero]

theorem pinM_mul_pinN : pinM * pinN = 1 := by
  ext i k
  rw [Matrix.mul_apply, Matrix.one_apply]
  let f : ℕ → GF128 := fun b => if F128.bit pinCols[k.val]! b then GF128.of (eqTable Flock.pinned)[b]! else 0
  have key : ∀ j : Fin 128, pinM i j * pinN j k = unpackHom (f j) i := by
    intro j
    rw [pinM_apply, pinN_apply, eqAtPinned_eq j.val j.isLt]
    show _ = unpack (f j) i
    simp only [f, unpack_ite]
    split_ifs <;> simp
  calc ∑ j, pinM i j * pinN j k = ∑ j : Fin 128, unpackHom (f j) i := Finset.sum_congr rfl fun j _ => key j
    _ = unpackHom (∑ j : Fin 128, f j) i := by rw [map_sum, Finset.sum_apply]
    _ = unpackHom (∑ b ∈ Finset.range 128, f b) i := by rw [Fin.sum_univ_eq_sum_range f 128]
    _ = unpack (GF128.of (F128.monomial k)) i := by rw [← pinCert_k k, xorSel_eq]; rfl
    _ = if i = k then 1 else 0 := unpack_monomial k i

theorem unpack_cast_mul (z : ZMod 2) (a : GF128) (i : Fin 128) : unpack ((z.val : GF128) * a) i = unpack a i * z := by
  fin_cases z
  · show unpack (((0 : ℕ) : GF128) * a) i = unpack a i * 0
    rw [Nat.cast_zero, zero_mul, unpack_zero, mul_zero]; rfl
  · show unpack (((1 : ℕ) : GF128) * a) i = unpack a i * 1
    rw [Nat.cast_one, one_mul, mul_one]

/-- **`Arith.Correct.pinned_indep`.** -/
theorem pinned_indep : ∀ c : Fin 128 → ZMod 2,
    (∑ b : Fin 128, ((c b).val : GF128) * eqAtPinned b.val) = 0 → c = 0 := by
  intro c h
  have hNM : pinN * pinM = 1 := (mul_eq_one_comm (M := Matrix (Fin 128) (Fin 128) (ZMod 2))).mp pinM_mul_pinN
  have hMc : pinM.mulVec c = 0 := by
    funext i
    have h1 := congrFun (congrArg unpackHom h) i
    rw [map_sum, map_zero, Finset.sum_apply, Pi.zero_apply] at h1
    rw [Pi.zero_apply, ← h1, Matrix.mulVec, dotProduct]
    refine Finset.sum_congr rfl fun b _ => ?_
    rw [pinM_apply]
    exact (unpack_cast_mul (c b) (eqAtPinned b.val) i).symm
  calc c = (pinN * pinM).mulVec c := by rw [hNM, Matrix.one_mulVec]
    _ = pinN.mulVec (pinM.mulVec c) := (Matrix.mulVec_mulVec _ _ _).symm
    _ = 0 := by rw [hMc, Matrix.mulVec_zero]

end FlockLevel3
