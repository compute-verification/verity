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
  /-- The level codes are Reed–Solomon codes (§13.1): distinct domain points, and the novel
  basis spans the polynomials of degree below `2^L`. -/
  omega_injective : ∀ d, Set.InjOn A.omega {q | q < 2 ^ d}
  xhat_poly : ∀ L, ∃ P : Fin (2 ^ L) → Polynomial F, (∀ c, (P c).natDegree < 2 ^ L) ∧
    LinearIndependent F P ∧ ∀ (c : Fin (2 ^ L)) ω, A.Xhat c.val L ω = (P c).eval ω
  /-- `K = F ⊕ F·u` (§2.4): a `K` coin is two `F` limbs (`drawK`, §13.2). -/
  ofLimbs_bijective : Function.Bijective fun w : F × F => A.ofLimbs w.1 w.2
  /-- Query coins give uniform positions (§13.5): the low `b ≤ 64` bits of `lo` are balanced. -/
  lo_balanced : ∀ b ≤ 64, ∀ t < 2 ^ b,
    (Finset.univ.filter fun w : F => A.lo w % 2 ^ b = t).card = Fintype.card F / 2 ^ b

/-- The skip nodes (§9): 64 distinct points on which `lagS` are the Lagrange weights. -/
theorem Model.Arith.Correct.lagS_interp {A : Arith F K} (hA : A.Correct) :
    ∃ node : Fin 64 → F, Function.Injective node ∧
      ∀ p : Polynomial F, p.natDegree < 64 → ∀ ζ, ∑ s, A.lagS ζ s * p.eval (node s) = p.eval ζ := by
  obtain ⟨nS, _, hS, _, _, hlag, _, _⟩ := hA.nodes
  exact ⟨nS, hS, hlag⟩

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

/-- The pin column's multilinear extension over the blocks. The lincheck's constant-wire coin
`β` checks that it is 1 at `ρ_out` (§10, A8). -/
def pinMLE (S : Statement) (z : Witness S.m) (ρOut : Fin S.nbl → F) : F :=
  ∑ o : Fin (2 ^ S.nbl), eqAt ρOut o.val * ((S.block z o S.pin).val : F)

/-- `ρ_out`: the zerocheck's challenges for the block bits (address bits `k_log … m−1`). -/
def Model.ZcOut.ρOut (S : Statement) (zc : ZcOut F S.m) : Fin S.nbl → F := fun j =>
  if h : S.kLog - 6 + j.val < S.m - 6 then zc.ρ ⟨S.kLog - 6 + j.val, h⟩ else 0

/-! ## The codewords near the committed table -/

/-- Messages whose encodings are within the level-0 radius of the committed table. -/
def closeMsgs (A : Arith F K) (sch : Schedule) (C : Oracle F) :
    Set (Message F sch.k0 sch.level₀.logCols) :=
  {M | Close sch.level₀.radius (asTable sch C) (encode A sch.level₀.logLen M)}

/-- The witness a message packs. -/
def wit (A : Arith F K) (S : Statement) (sch : Schedule) (M : Message F sch.k0 sch.level₀.logCols) :
    Witness S.m :=
  witnessOf A S.m sch.k0 sch.level₀.logCols M

/-- The packed table entry `f(y)` of a message (§12.1, §13.1: `t[c·2^k₀ + ℓ] = M[ℓ][c]`). -/
def packedAt (sch : Schedule) (M : Message F sch.k0 sch.level₀.logCols) (y : ℕ) : F :=
  if h : y / 2 ^ sch.k0 < 2 ^ sch.level₀.logCols then
    M ⟨y % 2 ^ sch.k0, Nat.mod_lt _ (Nat.two_pow_pos _)⟩ ⟨y / 2 ^ sch.k0, h⟩
  else 0

/-! ## Doomed states between the phases -/

/-- Doomed after the zerocheck. -/
def DZc (A : Arith F K) (S : Statement) (sch : Schedule) (C : Oracle F)
    (extra : List (Claim F S.m)) : Option (ZcOut F S.m) → Prop
  | none => True
  | some zc => ∀ M ∈ closeMsgs A sch C,
      ¬ ((⟨zc.va, A.lagS zc.z, zc.ρ⟩ : Claim F S.m).Holds (aVec S (wit A S sch M))
          ∧ (⟨zc.vb, A.lagS zc.z, zc.ρ⟩ : Claim F S.m).Holds (bVec S (wit A S sch M))
          ∧ (cClaim A zc).Holds (wit A S sch M)
          ∧ pinMLE S (wit A S sch M) (zc.ρOut S) = 1
          ∧ ∀ e ∈ extra, e.Holds (wit A S sch M))

/-- Doomed after the lincheck. -/
def DLc (A : Arith F K) (S : Statement) (sch : Schedule) (C : Oracle F)
    (extra : List (Claim F S.m)) (zc : ZcOut F S.m) : Option (Claim F S.m) → Prop
  | none => True
  | some ab => ∀ M ∈ closeMsgs A sch C,
      ¬ (ab.Holds (wit A S sch M) ∧ (cClaim A zc).Holds (wit A S sch M)
          ∧ ∀ e ∈ extra, e.Holds (wit A S sch M))

/-- Doomed after the opening: no codeword near the committed table has the batched value. -/
def DOp (A : Arith F K) (S : Statement) (sch : Schedule) (C : Oracle F) :
    Option (OpenOut F) → Prop
  | none => True
  | some op => ∀ M ∈ closeMsgs A sch C,
      (∑ y ∈ Finset.range (2 ^ (S.m - 7)), packedAt sch M y * op.b y) ≠ op.T

end FlockSoundness
