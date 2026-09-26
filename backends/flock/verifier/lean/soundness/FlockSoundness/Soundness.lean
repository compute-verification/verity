import FlockSoundness.Model.Session
import FlockSoundness.Accounting.Numbers
import FlockSoundness.Assumptions
import Mathlib.Algebra.Polynomial.Eval.Defs
import Mathlib.Algebra.CharP.Defs

/-!
# Soundness of Flock's interactive protocol, as PROTOCOL.md specifies it

**`table_sound`.** For every prover strategy, the probability that the verifier accepts a table
(both reps, on one committed root, with live coins) while no codeword within the level-0 radius
of the committed table packs a witness satisfying the statement, is at most
`tableError (fast100 m) shape`.

**`table_sound_fast100`.** With the numbers: at most `2^-195.5` for `22 ≤ m ≤ 33` and `2^-195.4`
for `m = 34, 35`, for every statement shape in `Stmt.InRange`.

This is the interactive-oracle layer: the committed tables are oracles. The Merkle layer
(`Merkle`) adds SHA-256's collision term for the compiled protocol.

**Status.** `table_sound` is proved from two lemmas, `rep_sound` (one rep from a doomed start)
and `link_sound` (the link points), which are still `sorry`; see `ASSUMPTIONS.md` for the plan.
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

/-- **One rep is sound** (the rep's round-by-round analysis, PROTOCOL.md §9–§13): from a doomed
start, whatever the prover does, the rep accepts with probability at most `repError`.
*Status: to prove* (per-phase lemmas; see `ASSUMPTIONS.md`). -/
theorem rep_sound (hMCA : Assumptions.BCHKS25Thm46) (A : Arith F K) (hA : A.Correct)
    (S : Statement) (sch : Schedule) (hsch : fast100 S.m = some sch) (hm : 13 ≤ S.m)
    (mPts : ℕ) (C : Oracle F) (extra : List (Claim F S.m))
    (hextra : extra.length = 2 * S.regions.length) (hdoom : RepDoomed A S sch C extra) :
    value (fun b => b = true) (rep A S sch hm C extra) ≤
      ENNReal.ofReal (repError sch (S.shape mPts)) := by
  sorry

/-- **The link points** (§16.2): if the committed values do not satisfy the statement, the two
link points leave some rep start undoomed with probability at most `linkError`.
*Status: to prove* (Schwartz–Zippel on each region's multilinear difference, twice). -/
theorem link_sound (A : Arith F K) (hA : A.Correct) (S : Statement) (sch : Schedule)
    (hsch : fast100 S.m = some sch) (ptLocal mPts : ℕ) (C : Oracle F)
    (hC : ¬ Committed A S sch C) :
    prCoin (fun pts : Fin 2 → Fin mPts → F =>
        ¬ RepDoomed A S sch C (extraClaims S ptLocal mPts pts)) ≤
      ENNReal.ofReal (linkError sch (S.shape mPts)) := by
  sorry

omit [Fintype F] [DecidableEq F] in
theorem extraClaims_length (S : Statement) (ptLocal mPts : ℕ) (pts : Fin 2 → Fin mPts → F) :
    (extraClaims S ptLocal mPts pts).length = 2 * S.regions.length := by
  unfold extraClaims
  induction S.regions with
  | nil => simp
  | cons R rs ih => simp [List.flatMap_cons, ih]; ring

/-- **Soundness of one table** (interactive-oracle layer). For every prover strategy, the
verifier accepts both reps while the committed values do not satisfy the statement with
probability at most `tableError`. -/
theorem table_sound (hMCA : Assumptions.BCHKS25Thm46) (A : Arith F K) (hA : A.Correct)
    (S : Statement) (sch : Schedule) (hsch : fast100 S.m = some sch) (hm : 13 ≤ S.m)
    (ptLocal mPts : ℕ) (σ : Strategy (table A S sch hm ptLocal mPts)) :
    prob (fun out => out.1 = true ∧ ¬ Committed A S sch out.2) (table A S sch hm ptLocal mPts) σ
      ≤ ENNReal.ofReal (tableError sch (S.shape mPts)) := by
  refine (prob_le_value _ _ σ).trans ?_
  set ε := repError sch (S.shape mPts)
  have hε : 0 ≤ ε := repError_nonneg sch _ (fast100_levels_ok S.m sch hsch)
  have hlink0 : 0 ≤ linkError sch (S.shape mPts) := by
    unfold linkError qF; positivity
  simp only [table, bind_eq, pure_eq, recv, draw, bind_send, bind_ret, bind_coin, value_send]
  refine iSup_le fun C => ?_
  by_cases hC : Committed A S sch C
  · -- every outcome carries `C`, so the event never happens
    have h0 : value (fun out : Bool × Oracle F => out.1 = true ∧ ¬ Committed A S sch out.2)
        (.coin (Fin 2 → Fin mPts → F) fun pts =>
          ((rep A S sch hm C (extraClaims S ptLocal mPts pts)).interleave
              (rep A S sch hm C (extraClaims S ptLocal mPts pts))).bind
            fun accepts => .ret (accepts.1 && accepts.2, C)) = 0 := by
      rw [value_coin]
      refine Finset.sum_eq_zero fun pts _ => ?_
      rw [value_bind_ret, value_congr (q := fun _ => False) (fun acc => by simp [hC]),
        value_false, mul_zero]
    rw [h0]
    exact zero_le
  · refine (value_coin_le _ _ _
      (fun pts => ¬ RepDoomed A S sch C (extraClaims S ptLocal mPts pts))
      (ENNReal.ofReal (linkError sch (S.shape mPts))) (ENNReal.ofReal ε * ENNReal.ofReal ε)
      (link_sound A hA S sch hsch ptLocal mPts C hC) fun pts hpts => ?_).trans ?_
    · rw [not_not] at hpts
      rw [value_bind_ret]
      have h := rep_sound hMCA A hA S sch hsch hm mPts C _
        (extraClaims_length S ptLocal mPts pts) hpts
      calc value (fun acc : Bool × Bool =>
              (acc.1 && acc.2, C).1 = true ∧ ¬ Committed A S sch (acc.1 && acc.2, C).2)
              ((rep A S sch hm C (extraClaims S ptLocal mPts pts)).interleave
                (rep A S sch hm C (extraClaims S ptLocal mPts pts)))
          = value (fun acc : Bool × Bool => acc.1 = true ∧ acc.2 = true)
              ((rep A S sch hm C (extraClaims S ptLocal mPts pts)).interleave
                (rep A S sch hm C (extraClaims S ptLocal mPts pts))) :=
            value_congr (fun acc => by simp [hC]) _
        _ ≤ _ := value_interleave_le (fun b : Bool => b = true) (fun b : Bool => b = true) _ _
        _ ≤ ENNReal.ofReal ε * ENNReal.ofReal ε := mul_le_mul' h h
    · rw [tableError, ENNReal.ofReal_add hlink0 (by positivity), sq, ENNReal.ofReal_mul hε]

/-- **The theorem with the numbers.** `fast100` run twice on one root with live coins: at most
`2^-195.5` per table for `22 ≤ m ≤ 33`, for every statement shape in range. -/
theorem table_sound_fast100 (hMCA : Assumptions.BCHKS25Thm46) (A : Arith F K) (hA : A.Correct)
    (S : Statement) (sch : Schedule) (hsch : fast100 S.m = some sch) (hm1 : 22 ≤ S.m)
    (hm2 : S.m ≤ 33) (ptLocal mPts : ℕ) (hshape : (S.shape mPts).InRange)
    (σ : Strategy (table A S sch (by omega) ptLocal mPts)) :
    prob (fun out => out.1 = true ∧ ¬ Committed A S sch out.2)
      (table A S sch (by omega) ptLocal mPts) σ ≤ ENNReal.ofReal ((2 : ℝ) ^ (-(195.5 : ℝ))) :=
  (table_sound hMCA A hA S sch hsch (by omega) ptLocal mPts σ).trans
    (ENNReal.ofReal_le_ofReal (tableError_le_of_le_33 hm1 hm2 hsch _ hshape))

end FlockSoundness
