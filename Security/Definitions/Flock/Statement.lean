import Definitions.Flock.Arith
import Mathlib.Data.Matrix.Mul
import Mathlib.Basic.Real.Basic
import Mathlib.Data.Fintype.Card

/-!
# Statements, witnesses, claims and the committed values

Every statement proves one block R1CS over GF(2) (PROTOCOL.md §16.1): `A·z ∘ B·z = z` with
`A = I ⊗ A₀`, `B = I ⊗ B₀`, a pinned constant column equal to 1 in every block, and public
regions whose bits the verifier knows (§16.2). This block R1CS is the circuit's lowered layout;
that it means the named circuit is the lowering theorem (level 3), outside this lane.

**The committed values.** The prover commits a level-0 table `C₀` (§13.1: `2^k₀` lanes, one
Reed–Solomon codeword position per column). In Flock's list-decoding regime a table is close to
up to `L₀` codewords, and no out-of-domain sample follows the commitment (§9.1, paper Remark 11),
so the commitment binds to that short list, not to one witness. `CommittedSatisfies` says that
some message whose encoding is within the level-0 proximity radius of `C₀` unpacks to a witness
satisfying the statement. This is the claim soundness certifies.
-/

namespace FlockSoundness.Model

/-- A witness: `2^m` bits; bit `a` is at address `a = s + 64·t + 2^k_log·o` (§9). -/
abbrev Witness (m : ℕ) := Fin (2 ^ m) → ZMod 2

/-- A public region (§16.2): the in-block addresses whose non-free bits equal `fixedBits`
(`fixedBits j` is `mask_j ∧ fixed_j`), with the known value of each region bit in each block
(`bits b i` is bit `i` of `bytes(region, b)`). -/
structure Region (kLog nbl : ℕ) where
  free : List (Fin kLog)
  fixedBits : Fin kLog → Bool
  bits : Fin (2 ^ nbl) → ℕ → ZMod 2

/-- The in-block address of region bit `i`: the free bits from `i`, the others from `fixedBits`. -/
def Region.inBlock {kLog nbl : ℕ} (R : Region kLog nbl) (i : ℕ) : ℕ :=
  (∑ t : Fin R.free.length, if i.testBit t then 2 ^ (R.free.get t).val else 0)
    + ∑ j : Fin kLog, if j ∉ R.free ∧ R.fixedBits j then 2 ^ j.val else 0

/-- One block R1CS statement (§16.1–§16.2). -/
structure Statement where
  m : ℕ
  kLog : ℕ
  hk : kLog ≤ m
  /-- `k_skip = 6 ≤ k_log`: the skip index is address bits 0–5 of every block (§9, §10). -/
  hk6 : 6 ≤ kLog
  pin : Fin (2 ^ kLog)
  A₀ : Matrix (Fin (2 ^ kLog)) (Fin (2 ^ kLog)) (ZMod 2)
  B₀ : Matrix (Fin (2 ^ kLog)) (Fin (2 ^ kLog)) (ZMod 2)
  regions : List (Region kLog (m - kLog))

namespace Statement

variable (S : Statement)

/-- Blocks. -/
abbrev nbl : ℕ := S.m - S.kLog

theorem addr_lt (o : Fin (2 ^ S.nbl)) (i : Fin (2 ^ S.kLog)) :
    o.val * 2 ^ S.kLog + i.val < 2 ^ S.m := by
  have h : 2 ^ S.m = 2 ^ S.nbl * 2 ^ S.kLog := by
    rw [← pow_add, Nat.sub_add_cancel S.hk]
  rw [h]
  calc o.val * 2 ^ S.kLog + i.val < o.val * 2 ^ S.kLog + 2 ^ S.kLog := by omega
    _ = (o.val + 1) * 2 ^ S.kLog := by ring
    _ ≤ 2 ^ S.nbl * 2 ^ S.kLog := Nat.mul_le_mul_right _ o.isLt

/-- The address of in-block bit `i` of block `o`. -/
def addr (o : Fin (2 ^ S.nbl)) (i : Fin (2 ^ S.kLog)) : Fin (2 ^ S.m) :=
  ⟨o.val * 2 ^ S.kLog + i.val, S.addr_lt o i⟩

/-- Block `o` of a witness. -/
def block (z : Witness S.m) (o : Fin (2 ^ S.nbl)) : Fin (2 ^ S.kLog) → ZMod 2 :=
  fun i => z (S.addr o i)

/-- **The statement.** Every block satisfies the R1CS, the pin is 1 in every block, and every
region's bits equal their public values. -/
def Satisfies (z : Witness S.m) : Prop :=
  (∀ o i, (S.A₀.mulVec (S.block z o)) i * (S.B₀.mulVec (S.block z o)) i = S.block z o i)
    ∧ (∀ o, S.block z o S.pin = 1)
    ∧ ∀ R ∈ S.regions, ∀ (o : Fin (2 ^ S.nbl)) (i : ℕ) (_hi : i < 2 ^ R.free.length),
        ∀ h : R.inBlock i < 2 ^ S.kLog, S.block z o ⟨R.inBlock i, h⟩ = R.bits o i

end Statement

/-! ## Claims -/

variable {F : Type} [Field F]

/-- A claim handed to the opening (§11): `(value, skip weights, point)`. -/
structure Claim (F : Type) (m : ℕ) where
  value : F
  skip : Fin 64 → F
  x : Fin (m - 6) → F

/-- The value a claim asserts: `Σ_s skip[s]·Σ_{x'} eq(x, x')·z[s + 64·x']`, the evaluation of
the witness's extension that is multilinear in address bits 6… and has the given weights on the
64 skip positions (the quirky extension for `ab` and `c`, the multilinear one for regions). -/
def Claim.eval {m : ℕ} (c : Claim F m) (z : Witness m) : F :=
  ∑ a : Fin (2 ^ m), c.skip ⟨a.val % 64, Nat.mod_lt _ (by norm_num)⟩ * eqAt c.x (a.val / 64)
    * ((z a).val : F)

/-- A claim holds for a witness. -/
def Claim.Holds {m : ℕ} (c : Claim F m) (z : Witness m) : Prop := c.eval z = c.value

/-! ## The committed values -/

/-- A level-0 table: `2^k₀` lanes, `2^d₀` codeword positions (§13.1). -/
abbrev Table (F : Type) (lanes d : ℕ) := Fin (2 ^ lanes) → Fin (2 ^ d) → F

/-- A level-0 message: `2^k₀` lanes of `2^cols₀` message columns. -/
abbrev Message (F : Type) (lanes cols : ℕ) := Fin (2 ^ lanes) → Fin (2 ^ cols) → F

variable {K : Type} [Field K]

/-- The interleaved encoding of a message (§13.1). -/
def encode (A : Arith F K) {lanes cols : ℕ} (d : ℕ) (M : Message F lanes cols) : Table F lanes d :=
  fun ℓ q => A.encodeAt cols (M ℓ) q

open Classical in
/-- `U` is `γ`-close to `C`: they agree, in every lane, on at least a `1 − γ` fraction of the
positions (paper Appendix C.3). -/
def Close {lanes d : ℕ} (γ : ℝ) (C U : Table F lanes d) : Prop :=
  ((Finset.univ.filter fun q : Fin (2 ^ d) => ∀ ℓ, C ℓ q = U ℓ q).card : ℝ) ≥ (1 - γ) * 2 ^ d

/-- The witness a level-0 message packs (§12.1, §13.1): table entry `t[c·2^k₀ + ℓ] = M[ℓ][c]` is
the packed element `f(y) = Σ_v z[128y + v]·x^v`. -/
def witnessOf (A : Arith F K) (m k₀ cols : ℕ) (M : Message F k₀ cols) : Witness m :=
  fun a =>
    let y := a.val / 128
    let ℓ := y % 2 ^ k₀
    let c := y / 2 ^ k₀
    if h : c < 2 ^ cols then A.unpack (M ⟨ℓ, Nat.mod_lt _ (Nat.two_pow_pos _)⟩ ⟨c, h⟩)
      ⟨a.val % 128, Nat.mod_lt _ (by norm_num)⟩ else 0

/-- **The committed values satisfy the statement:** some message whose encoding is `γ₀`-close
to the committed level-0 table packs a witness that satisfies `S`. -/
def CommittedSatisfies (A : Arith F K) (S : Statement) (k₀ cols₀ d₀ : ℕ) (γ₀ : ℝ)
    (C₀ : Table F k₀ d₀) : Prop :=
  ∃ M : Message F k₀ cols₀, Close γ₀ C₀ (encode A d₀ M) ∧ S.Satisfies (witnessOf A S.m k₀ cols₀ M)

end FlockSoundness.Model
