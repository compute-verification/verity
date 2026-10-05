import FlockSoundness.Model.Session
import ArkLib.Data.CodingTheory.ProximityGap.CapacityBounds
import Mathlib.Algebra.Polynomial.Eval.Defs
import Mathlib.Algebra.CharP.Defs
import FlockSoundness.Protocol.Accounting.Fast100

/-!
# Shared definitions of the soundness statement

The arithmetic facts the proof uses (`Arith.Correct`), the committed values at the level-0
radius (`Committed`), and the doomed start of a rep (`RepDoomed`).
-/

namespace FlockSoundness

open Game Model Accounting

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]

/-- The facts about the arithmetic (`Model.Arith`) that the soundness proof uses. They are
properties of concrete functions, to be proved for the GHASH instance that the executable
verifier defines (`Flock.Field.*`); they are not assumptions. -/
structure Model.Arith.Correct (A : Arith F K) : Prop where
  card_F : Fintype.card F = 2 ^ 128
  card_K : Fintype.card K = 2 ^ 256
  charTwo : CharP F 2
  charTwoK : CharP K 2
  /-- `pack` and `unpack` are inverse, and `unpack` is additive: the basis `x^v` (§12.1). -/
  unpack_pack : ∀ b, A.unpack (A.pack b) = b
  pack_unpack : ∀ a, A.pack (A.unpack a) = a
  unpack_add : ∀ a a', A.unpack (a + a') = A.unpack a + A.unpack a'
  /-- The skip nodes `S` and the output coset `Λ` (§9): 128 distinct points. `lagS` and `lagΛ`
  are the Lagrange weights on each, and `combW` evaluates a polynomial of degree below 128 that
  vanishes on `S` from its values on `Λ` (`Pcomb`, §9.2). -/
  nodes : ∃ nodeS nodeΛ : Fin 64 → F, Function.Injective nodeS ∧ Function.Injective nodeΛ ∧
    (∀ s i, nodeS s ≠ nodeΛ i) ∧
    (∀ p : Polynomial F, p.natDegree < 64 → ∀ ζ, ∑ s, A.lagS ζ s * p.eval (nodeS s) = p.eval ζ) ∧
    (∀ p : Polynomial F, p.natDegree < 64 → ∀ ζ, ∑ i, A.lagΛ ζ i * p.eval (nodeΛ i) = p.eval ζ) ∧
    (∀ p : Polynomial F, p.natDegree < 128 → (∀ s, p.eval (nodeS s) = 0) →
      ∀ ζ, ∑ i, A.combW ζ i * p.eval (nodeΛ i) = p.eval ζ)
  /-- The pinned coordinates' 128 equality weights are independent over GF(2), and none is 1
  (§9.2 and the `inv(0) = 0` convention, A5). -/
  pinned_indep : ∀ c : Fin 128 → ZMod 2,
    (∑ b : Fin 128, ((c b).val : F) * eqAt A.pinned b.val) = 0 → c = 0
  pinned_ne_one : ∀ i, A.pinned i ≠ 1
  /-- The level codes are Reed–Solomon codes (§13.1): distinct domain points (`ω_q` is the low
  word of `q`, so up to `2^64` of them), and the novel basis spans the polynomials of degree below
  `2^L` (for `L ≤ 40`, the range the executable's normalisers are checked on). Every `fast100`
  level fits (`fast100_levels_fit`). -/
  omega_injective : ∀ d ≤ 64, Set.InjOn A.omega {q | q < 2 ^ d}
  xhat_poly : ∀ L ≤ 40, ∃ P : Fin (2 ^ L) → Polynomial F, (∀ c, (P c).natDegree < 2 ^ L) ∧
    LinearIndependent F P ∧ ∀ (c : Fin (2 ^ L)) ω, A.Xhat c.val L ω = (P c).eval ω
  /-- `K = F ⊕ F·u` (§2.4): a `K` coin is two `F` limbs (`drawK`, §13.2). -/
  ofLimbs_bijective : Function.Bijective fun w : F × F => A.ofLimbs w.1 w.2
  /-- Query coins give uniform positions (§13.5): the low `b ≤ 64` bits of `lo` are balanced. -/
  lo_balanced : ∀ b ≤ 64, ∀ t < 2 ^ b,
    (Finset.univ.filter fun w : F => A.lo w % 2 ^ b = t).card = Fintype.card F / 2 ^ b

/-- A level in the range of `Arith.Correct`'s code facts: at most `2^40` message columns and
`2^64` positions. -/
def Accounting.Level.Fits (l : Level) : Prop := l.logCols ≤ 40 ∧ l.logLen ≤ 64

instance (l : Level) : Decidable l.Fits := by unfold Accounting.Level.Fits; infer_instance

/-- Level 0 of a schedule. -/
def Accounting.Schedule.level₀ (sch : Schedule) : Level := sch.levels.headD default

/-- The committed level-0 table as a table. -/
def asTable (sch : Schedule) (C : Oracle F) : Table F sch.k0 sch.level₀.logLen :=
  fun ℓ q => C ℓ q

/-- "The committed values satisfy the statement", at the level-0 radius of `sch`. -/
def Committed (A : Arith F K) (S : Statement) (sch : Schedule) (C : Oracle F) : Prop :=
  CommittedSatisfies A S sch.k0 sch.level₀.logCols sch.level₀.logLen sch.level₀.radius
    (asTable sch C)

/-- The statement's shape, for the accounting. -/
def Model.Statement.shape (S : Statement) (mPts : ℕ) : Stmt := ⟨S.kLog, S.regions.length, mPts⟩

/-! ## The witness's A-side and B-side, and the pin's extension -/

theorem Model.Statement.block_index_lt (S : Statement) (a : Fin (2 ^ S.m)) :
    a.val / 2 ^ S.kLog < 2 ^ S.nbl := by
  have h : 2 ^ S.m = 2 ^ S.nbl * 2 ^ S.kLog := by rw [← pow_add, Nat.sub_add_cancel S.hk]
  rw [Nat.div_lt_iff_lt_mul (Nat.two_pow_pos _), ← h]
  exact a.isLt

/-- The A-side bits: `(A·z)` at address `a = o·2^k_log + i` is `(A₀·z_o)_i`. -/
def aVec (S : Statement) (z : Witness S.m) : Witness S.m := fun a =>
  (S.A₀.mulVec (S.block z ⟨a.val / 2 ^ S.kLog, S.block_index_lt a⟩))
    ⟨a.val % 2 ^ S.kLog, Nat.mod_lt _ (Nat.two_pow_pos _)⟩

/-- The B-side bits. -/
def bVec (S : Statement) (z : Witness S.m) : Witness S.m := fun a =>
  (S.B₀.mulVec (S.block z ⟨a.val / 2 ^ S.kLog, S.block_index_lt a⟩))
    ⟨a.val % 2 ^ S.kLog, Nat.mod_lt _ (Nat.two_pow_pos _)⟩

/-! ## The codewords near the committed table -/

/-- The witness a message packs. -/
def wit (A : Arith F K) (S : Statement) (sch : Schedule) (M : Message F sch.k0 sch.level₀.logCols) :
    Witness S.m :=
  witnessOf A S.m sch.k0 sch.level₀.logCols M

/-! ## Doomed states between the phases -/

/-! ## The doomed states over any list of messages

The phases before Ligerito use the messages near the committed table (`closeMsgs`) only through
membership and the list's size, so each is proved over any set `W` of messages (`…_of_list`). The
doomed states above are these at `W = closeMsgs A sch C` (`repDoomed_iff`, `dZc_eq`, `dLc_eq`,
`dOp_eq`). The padded level 0 takes `W` to be its padded code's witnesses (`SoundnessPad`). -/

end FlockSoundness
