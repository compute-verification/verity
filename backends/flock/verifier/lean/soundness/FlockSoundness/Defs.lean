import FlockSoundness.Model.Session
import FlockSoundness.Accounting.Numbers
import FlockSoundness.Assumptions
import Mathlib.Algebra.Polynomial.Eval.Defs
import Mathlib.Algebra.CharP.Defs

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
  /-- The skip nodes (§9): 64 distinct points on which `lagS` are the Lagrange weights. -/
  lagS_interp : ∃ node : Fin 64 → F, Function.Injective node ∧
    ∀ p : Polynomial F, p.natDegree < 64 → ∀ ζ, ∑ s, A.lagS ζ s * p.eval (node s) = p.eval ζ
  /-- The output coset `Λ`, disjoint from the skip nodes (§9.2). -/
  lagΛ_interp : ∃ node : Fin 64 → F, Function.Injective node ∧
    ∀ p : Polynomial F, p.natDegree < 64 → ∀ ζ, ∑ i, A.lagΛ ζ i * p.eval (node i) = p.eval ζ
  /-- The pinned coordinates' 128 equality weights are independent over GF(2), and none is 1
  (§9.2 and the `inv(0) = 0` convention, A5). -/
  pinned_indep : ∀ c : Fin 128 → ZMod 2,
    (∑ b : Fin 128, ((c b).val : F) * eqAt A.pinned b.val) = 0 → c = 0
  pinned_ne_one : ∀ i, A.pinned i ≠ 1
  /-- The level codes are Reed–Solomon codes (§13.1): distinct domain points, and the novel
  basis spans the polynomials of degree below `2^L`. -/
  omega_injective : ∀ d, Set.InjOn A.omega {q | q < 2 ^ d}
  xhat_poly : ∀ L, ∃ P : Fin (2 ^ L) → Polynomial F, (∀ c, (P c).natDegree < 2 ^ L) ∧
    LinearIndependent F P ∧ ∀ (c : Fin (2 ^ L)) ω, A.Xhat c.val L ω = (P c).eval ω
  /-- Query coins give uniform positions (§13.5): the low `b ≤ 64` bits of `lo` are balanced. -/
  lo_balanced : ∀ b ≤ 64, ∀ t < 2 ^ b,
    (Finset.univ.filter fun w : F => A.lo w % 2 ^ b = t).card = Fintype.card F / 2 ^ b

/-- Level 0 of a schedule. -/
def Accounting.Schedule.level₀ (sch : Schedule) : Level := sch.levels.headD default

/-- The committed level-0 table as a table. -/
def asTable (sch : Schedule) (C : Oracle F) : Table F sch.k0 sch.level₀.logLen :=
  fun ℓ q => C ℓ q

/-- "The committed values satisfy the statement", at the level-0 radius of `sch`. -/
def Committed (A : Arith F K) (S : Statement) (sch : Schedule) (C : Oracle F) : Prop :=
  CommittedSatisfies A S sch.k0 sch.level₀.logCols sch.level₀.logLen sch.level₀.radius
    (asTable sch C)

/-- A rep starts doomed when no codeword near the committed table packs a witness that satisfies
the block R1CS and the pin and makes every opened region claim true. -/
def RepDoomed (A : Arith F K) (S : Statement) (sch : Schedule) (C : Oracle F)
    (extra : List (Claim F S.m)) : Prop :=
  ∀ M : Message F sch.k0 sch.level₀.logCols,
    Close sch.level₀.radius (asTable sch C) (encode A sch.level₀.logLen M) →
      let z := witnessOf A S.m sch.k0 sch.level₀.logCols M
      ¬ ((∀ o i, (S.A₀.mulVec (S.block z o)) i * (S.B₀.mulVec (S.block z o)) i = S.block z o i)
          ∧ (∀ o, S.block z o S.pin = 1) ∧ ∀ c ∈ extra, c.Holds z)

/-- The statement's shape, for the accounting. -/
def Model.Statement.shape (S : Statement) (mPts : ℕ) : Stmt := ⟨S.kLog, S.regions.length, mPts⟩

omit [Fintype F] [DecidableEq F] in
theorem extraClaims_length (S : Statement) (ptLocal mPts : ℕ) (pts : Fin 2 → Fin mPts → F) :
    (extraClaims S ptLocal mPts pts).length = 2 * S.regions.length := by
  unfold extraClaims
  induction S.regions with
  | nil => simp
  | cons R rs ih => simp [List.flatMap_cons, ih]; ring

end FlockSoundness
