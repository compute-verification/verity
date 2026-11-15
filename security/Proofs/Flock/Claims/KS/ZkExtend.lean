module

public import Proofs.Flock.Claims.KS.Extend
public import Proofs.Flock.Claims.KS.ZkTables
public import Proofs.Flock.Claims.KS.ZkUntrim
import all Init.Data.Array.Basic
import all Init.Data.Nat.Power2.Basic
import all Init.Data.Range.Basic
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Topology.Sheaves.Init

/-!
# `ExtendJ` at the untrimmed masked instance

`Claims.Flock.ExtendJ D N Ok` asks, at each table `j` of a statement: a table at table `j`'s word's positions whose
columns are `Ok` or `0` (a position no rerun opened) and which does not extract (`GoodTableAt`) has a table of `Ok`
columns that keeps its nonzero ones and does not extract either. At the untrimmed masked instance
(`FlockDeploy.deployZk`), `Ok` is a column of table `j`'s witness lanes (`wideJ`) and an unopened position reads `[]`,
which no trace's column equals.

The argument is the one `Proofs.Flock.Claims.KS.Extend` gives at any code (`Claims.Flock.extendJ_of_fill`,
`Claims.Flock.fillable_of_avoid`, `Claims.GS.exists_avoid`), here at the masked code: its points are distinct
(`zk_inj`), Johnson's regime holds one agreement short of its radius at every `m` served, since `n ≥ 2^12` and the
rate is at least `1/2` (`zk_reg`, `Claims.GS.reg_of_rate`), and the indices' `n = 2^d` summed stay below
`|GF(2^128)|` (`zk_card`).

* `FlockDeploy.KS.zk_reg`, `FlockDeploy.KS.zk_inj`, `FlockDeploy.KS.zk_card`, `FlockDeploy.KS.zk_avoid`: the masked
  code's regime, points and size, and the first lane to fill with.
* `FlockDeploy.KS.zk_width`, `FlockDeploy.KS.zk_extendJ`: `ExtendJ` at `deployZk rel`, at every relation.
* `FlockDeploy.KS.zk_KS_untrim`: so `Flock.KS` at the untrimmed masked instance at SHA-512 at the deployed relation
  holds with no premise (`zk_KS_untrim_of_extendJ`).
* `FlockDeploy.KS.zk_KS_untrim_at`: and with `E^rep` truncated at any `K₀ > 0` (`FlockDeploy.deployZkAt`), from one
  wait's bound at the instance (`zk_oneWait_untrim`), which reads no truncation.
-/

@[expose] public section

namespace FlockDeploy.KS

open Flock

local notation "𝔽" => FlockLevel3.GF128

/-- Every masked level 0 the verifier serves has at least `2^12` positions (the least is `m = 25`'s `2^13`). -/
theorem zkLen_ge12 : ∀ i : Claims.Flock.ZkIdx, 12 ≤ Claims.Flock.zkLen i := by
  decide +kernel

/-- **The masked code's regime one agreement short of its radius**, at every `m` served. -/
theorem zk_reg (i : Claims.Flock.ZkIdx) (t : ℕ)
    (ht : (1 - (1 - Real.sqrt (((2 : ℝ) ^ Claims.Flock.zkCols i + Claims.Flock.zkPad i) /
      2 ^ Claims.Flock.zkLen i) - 1 / 200)) * ((2 ^ Claims.Flock.zkLen i : ℕ) : ℝ) - 1 ≤ t) :
    ((Claims.Flock.flockZkLevelZero i).k + 1) * 2 ^ Claims.Flock.zkLen i < t * t := by
  show (2 ^ Claims.Flock.zkCols i + Claims.Flock.zkPad i + 1) * 2 ^ Claims.Flock.zkLen i < t * t
  have hn : 1024 ≤ 2 ^ Claims.Flock.zkLen i :=
    le_trans (by norm_num) (Nat.pow_le_pow_right (by norm_num) (zkLen_ge12 i))
  have hk : 2 ^ Claims.Flock.zkLen i ≤ 2 * (2 ^ Claims.Flock.zkCols i + Claims.Flock.zkPad i) := by
    rw [(Claims.Flock.zkShape_table i).2.2.2.1, pow_succ]
    omega
  refine Claims.GS.reg_of_rate hn hk (le_of_eq_of_le ?_ ht)
  push_cast
  ring

/-- **The masked code's points are distinct** at each index's `2^d` positions. -/
theorem zk_inj (i : Claims.Flock.ZkIdx) :
    Set.InjOn (Claims.Flock.flockZkLevelZero i).dom {q | q < 2 ^ Claims.Flock.zkLen i} :=
  FlockSoundness.execArith_correct.omega_injective (Claims.Flock.zkLen i)
    ((Claims.Flock.zkShape_fits i).1.trans (by norm_num))

/-- **The indices' codeword lengths summed stay below `|GF(2^128)|`**: each is at most `2^40`. -/
theorem zk_card : ∑ i : Claims.Flock.ZkIdx, 2 ^ Claims.Flock.zkLen i < Fintype.card 𝔽 := by
  rw [FlockLevel3.card_F]
  calc ∑ i : Claims.Flock.ZkIdx, 2 ^ Claims.Flock.zkLen i ≤ ∑ _i : Claims.Flock.ZkIdx, 2 ^ 40 :=
        Finset.sum_le_sum fun i _ => Nat.pow_le_pow_right (by norm_num) (Claims.Flock.zkShape_fits i).1
    _ < 2 ^ 128 := by simp

namespace Gen

variable [S : ZkSetup]

/-- **A first lane no masked trace one agreement short of its radius has at `p`**, at any word `W`, read at the
untrimmed masked instance (`Claims.GS.exists_avoid` at the deployed masked code). -/
theorem zk_avoid (rel : Readable.Parsed → Readable.Relation) (W : ℕ → List 𝔽) (p : ℕ) :
    ∃ c : 𝔽, ∀ M : Claims.Flock.ZkTrace,
      (1 - (deployZk rel).θ M) * ((deployZk rel).len M : ℝ) ≤
        (((Finset.univ : Finset (Fin ((deployZk rel).len M))).filter fun q =>
          (deployZk rel).enc M q.val = W q.val).card : ℝ) + 1 →
      ((deployZk rel).enc M p)[0]? ≠ some c :=
  Claims.GS.exists_avoid Claims.Flock.flockZkLevelZero (fun _ => pow_pos (by norm_num) _)
    (fun i => 2 ^ Claims.Flock.zkLen i)
    (fun i => 1 - Real.sqrt (((2 : ℝ) ^ Claims.Flock.zkCols i + Claims.Flock.zkPad i) / 2 ^ Claims.Flock.zkLen i) -
      1 / 200) zk_inj zk_reg zk_card W p

end Gen

/-- **A first lane no masked trace one agreement short of its radius has at `p`**, at any word `W`, read at the
untrimmed masked instance (`Claims.GS.exists_avoid` at the deployed masked code). -/
theorem zk_avoid (rel : Readable.Parsed → Readable.Relation) (W : ℕ → List 𝔽) (p : ℕ) :
    ∃ c : 𝔽, ∀ M : Claims.Flock.ZkTrace,
      (1 - (deployZk rel).θ M) * ((deployZk rel).len M : ℝ) ≤
        (((Finset.univ : Finset (Fin ((deployZk rel).len M))).filter fun q =>
          (deployZk rel).enc M q.val = W q.val).card : ℝ) + 1 →
      ((deployZk rel).enc M p)[0]? ≠ some c :=
  Gen.zk_avoid (S := S₀) (rel := rel) (W := W) (p := p)

/-- **Table `j`'s witness lanes are a width**, at every nonempty word of the statement. -/
theorem zk_width (x : Claims.Msg) (j : ℕ) (h : 0 < wordLenJ true x j) :
    ∃ w, 0 < w ∧ ∀ u : List 𝔽, u.length = w → wideJ true x j u := by
  cases hh : (tables true x)[j]? with
  | none => simp [wordLenJ, hh] at h
  | some a =>
    obtain ⟨_, st⟩ := a
    exact ⟨2 ^ (Claims.Flock.zkLevel0 st.m).lanes, by positivity, fun u hu => by simp [wideJ, hh, hu]⟩

/-- **`ExtendJ` at the untrimmed masked instance**, at every relation: `Claims.Flock.extendJ_of_fill` at the masked
code's first lane (`zk_avoid`) and table `j`'s witness lanes (`zk_width`). -/
theorem zk_extendJ (rel : Readable.Parsed → Readable.Relation) :
    Claims.Flock.ExtendJ (deployZk rel) (wordLenJ true) (wideJ true) :=
  Claims.Flock.extendJ_of_fill (Claims.Flock.fillable_of_avoid _ _ _ (zk_avoid rel) zk_width)

namespace Gen

variable [S : ZkSetup]

/-- **Table `j`'s witness lanes are a width**, at every nonempty word of the statement. -/
theorem zk_width (x : Claims.Msg) (j : ℕ) (h : 0 < wordLenJ true x j) :
    ∃ w, 0 < w ∧ ∀ u : List 𝔽, u.length = w → wideJ true x j u := by
  cases hh : (tables true x)[j]? with
  | none => simp [wordLenJ, hh] at h
  | some a =>
    obtain ⟨_, st⟩ := a
    exact ⟨2 ^ (Claims.Flock.zkLevel0 st.m).lanes, by positivity, fun u hu => by simp [wideJ, hh, hu]⟩

/-- **`ExtendJ` at the untrimmed masked instance**, at every relation: `Claims.Flock.extendJ_of_fill` at the masked
code's first lane (`zk_avoid`) and table `j`'s witness lanes (`zk_width`). -/
theorem zk_extendJ (rel : Readable.Parsed → Readable.Relation) :
    Claims.Flock.ExtendJ (deployZk rel) (wordLenJ true) (wideJ true) :=
  Claims.Flock.extendJ_of_fill (Claims.Flock.fillable_of_avoid _ _ _ (zk_avoid rel) zk_width)

end Gen

namespace Gen

variable [S : ZkSetup]

/-- **One wait's bound at the untrimmed masked instance**, at SHA-512 at the deployed relation, with no premise:
`zk_oneWait_untrim_of_extendJ` with `ExtendJ` (`zk_extendJ`). -/
theorem zk_oneWait_untrim : Claims.Flock.OneWait sha512 (deployZk Readable.relation) :=
  zk_oneWait_untrim_of_extendJ (zk_extendJ Readable.relation)

/-- **Masked knowledge soundness at the untrimmed instance**, at SHA-512 at the deployed relation, with no premise:
`zk_KS_untrim_of_extendJ` with `ExtendJ` (`zk_extendJ`). -/
theorem zk_KS_untrim : Claims.Flock.KS sha512 (deployZk Readable.relation) :=
  zk_KS_untrim_of_extendJ (zk_extendJ Readable.relation)

/-- **Masked knowledge soundness at the untrimmed instance with `E^rep` truncated at `K₀ > 0`**, at SHA-512 at the
deployed relation, with no other premise: one wait's bound at the instance (`zk_oneWait_untrim`), which reads no
truncation, at `deployZkAt K₀` (`zk_of_oneWait_sha512_at`). -/
theorem zk_KS_untrim_at {K₀ : ℕ} (hK₀ : 0 < K₀) : Claims.Flock.KS sha512 (deployZkAt K₀ Readable.relation) :=
  zk_of_oneWait_sha512_at Readable.relation hK₀ zk_splits zk_oneWait_untrim

end Gen

/-- **Masked knowledge soundness at the untrimmed instance**, at SHA-512 at the deployed relation, with no premise:
`zk_KS_untrim_of_extendJ` with `ExtendJ` (`zk_extendJ`). -/
theorem zk_KS_untrim : Claims.Flock.KS sha512 (deployZk Readable.relation) :=
  Gen.zk_KS_untrim (S := S₀)

/-- **Masked knowledge soundness at the untrimmed instance with `E^rep` truncated at `K₀ > 0`**, at SHA-512 at the
deployed relation, with no other premise (`Gen.zk_KS_untrim_at`). -/
theorem zk_KS_untrim_at {K₀ : ℕ} (hK₀ : 0 < K₀) : Claims.Flock.KS sha512 (deployZkAt K₀ Readable.relation) :=
  Gen.zk_KS_untrim_at (S := S₀) hK₀

end FlockDeploy.KS
