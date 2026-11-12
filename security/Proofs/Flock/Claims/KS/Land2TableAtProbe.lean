module

public import Proofs.Flock.Claims.KS.ZkTableAt
public import Proofs.Flock.Claims.KS.ZkTables
public import Proofs.Flock.Claims.KS.TableAt
public import Proofs.Flock.Claims.NumbersDeploy
public import Proofs.Flock.Claims.Numbers
public import Proofs.Flock.Claims.DeployTables
public import Proofs.Flock.Claims.DeployCode
public import Definitions.Claims.SamplingTable
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Algebra.Order.Star.Real
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
import Mathlib.Tactic.Polynomial.Basic
import Mathlib.Tactic.ReduceModChar
import Mathlib.Topology.Sheaves.Init
import Std.Tactic.BVDecide.Normalize.Prop

/-!
# A reader's probe of Flock's statements at one table of a session

Not for `main`: a second reader's checks of the statements at one table of a session (`Flock.KSAt`, `Flock.κAt`,
`Flock.RAt`), each a theorem this file proves from what the tree already has.

* **The size at a table.** `κAt` is the session's `κ` read at the table's error and word length (`κAt_eq_κ`), so the
  session's number lemmas give the table's: `κ_j ≤ 2^{−202.98}` and `λ_j ≥ 201.98` from the table's numbers
  (`κAt_le_of`, `lamAt_ge_of`), at a masked table in scope (`κAt_zk_le`) and at every table of a masked session in
  scope (`zkIn_at`), where the session's own are `n·2^{−202.98}` and `201.98 − log₂ n`. A word of `2^23` symbols
  stops each wait at `12,343,986` accepted reruns (`kAt_23`).
* **Past the statement's tables** the word is empty and the error at the table is at least `1` (`one_le_κAt_past`),
  while the deployed `table` still gives a statement there (`table_ne_none`).
* **At a statement of one table** table `0`'s relation is the statement's (`RAt_zero_iff_R`) and its error the
  session's (`κAt_zero_eq_κ`).
-/

@[expose] public section

namespace FlockDeploy.KS.Land2Probe

open Flock

/-! ## The size at a table -/

/-- `κAt` is `κ` at the record whose `εtable` and `N₀` are table `j`'s error and word length. -/
theorem κAt_eq_κ {Coin Col T : Type} (D : Claims.Flock.Deploy Coin Col T) (N : Claims.Msg → ℕ → ℕ)
    (ε : Claims.Msg → ℕ → ℝ) (j : ℕ) (x : Claims.Msg) :
    Claims.Flock.κAt D N ε j x =
      Claims.Flock.κ { D with εtable := fun y => ε y j, N₀ := fun y => (N y j : ℝ) } x := rfl

/-- `κ_j ≤ 2^{−202.98}` from the table's numbers: its error at most `2^{−205}`, `210` waits, the truncation `2^256`
and a word of `2^10` to `2^23` symbols (`Numbers.κ_le` at the table's record). -/
theorem κAt_le_of {Coin Col T : Type} (D : Claims.Flock.Deploy Coin Col T) (N : Claims.Msg → ℕ → ℕ)
    (ε : Claims.Msg → ℕ → ℝ) (j : ℕ) (x : Claims.Msg) (hε : ε x j ≤ 2 ^ (-(205 : ℝ))) (hw : 210 ≤ D.waits x)
    (hK : D.cap x = Claims.Flock.Numbers.cap) (hN : (2 : ℝ) ^ 10 ≤ (N x j : ℝ)) (hN' : (N x j : ℝ) ≤ 2 ^ 23) :
    Claims.Flock.κAt D N ε j x ≤ 2 ^ (-(202.98 : ℝ)) :=
  Claims.Flock.Numbers.κ_le { D with εtable := fun y => ε y j, N₀ := fun y => (N y j : ℝ) } x hε hw hK hN hN'

/-- `λ_j ≥ 201.98` from the table's numbers and the session's `γ ≤ 2^105` (`Numbers.lam_ge` at the table's
record). -/
theorem lamAt_ge_of {Coin Col T : Type} (D : Claims.Flock.Deploy Coin Col T) (N : Claims.Msg → ℕ → ℕ)
    (ε : Claims.Msg → ℕ → ℝ) (j : ℕ) (x : Claims.Msg) (hε0 : 0 ≤ ε x j) (hε : ε x j ≤ 2 ^ (-(205 : ℝ)))
    (hw : 210 ≤ D.waits x) (hK : D.cap x = Claims.Flock.Numbers.cap) (hN : (2 : ℝ) ^ 10 ≤ (N x j : ℝ))
    (hN' : (N x j : ℝ) ≤ 2 ^ 23) (hγ : Claims.Flock.γ D x ≤ 2 ^ (105 : ℝ)) :
    201.98 ≤ Claims.lamOf (Claims.Flock.κAt D N ε j x) (Claims.Flock.γ D x) :=
  Claims.Flock.Numbers.lam_ge { D with εtable := fun y => ε y j, N₀ := fun y => (N y j : ℝ) } x hε0 hε hw hK hN
    hN' hγ

/-- A table's own error at a table of the statement. -/
theorem tableErrAt_eq {x : Claims.Msg} {j : ℕ} {spec : SessionSpec} {st : Stmt}
    (h : (tables true x)[j]? = some (spec, st)) : tableErrAt true x j = tableErr true spec st := by
  unfold tableErrAt
  rw [h]
  rfl

/-- A masked table's word's length is its level-0 codeword length. -/
theorem wordLenJ_cast {x : Claims.Msg} {j : ℕ} {spec : SessionSpec} {st : Stmt}
    (h : (tables true x)[j]? = some (spec, st)) : (wordLenJ true x j : ℝ) = tableLen true st := by
  have e : wordLenJ true x j = 2 ^ (Claims.Flock.zkLevel0 st.m).d := by
    simp [wordLenJ, h]
  rw [e, Nat.cast_pow, Nat.cast_ofNat]
  first
    | done
    | simp only [tableLen, ite_true]
    | simp [tableLen]

/-- **`κ_j ≤ 2^{−202.98}` at a masked table in scope**: a table of the statement at an `m` the masked verifier
serves and a shape in `InRange`. -/
theorem κAt_zk_le (rel : Readable.Parsed → Readable.Relation) {x : Claims.Msg} {j : ℕ} {spec : SessionSpec}
    {st : Stmt} (h : (tables true x)[j]? = some (spec, st)) (h25 : 25 ≤ st.m) (h35 : st.m ≤ 35)
    (hst : (acct spec st).InRange) :
    Claims.Flock.κAt (deployZk rel) (wordLenJ true) (tableErrAt true) j x ≤ 2 ^ (-(202.98 : ℝ)) := by
  have hw := wordLenJ_cast h
  obtain ⟨hlo, hhi⟩ := tableLen_zk_bounds st h25 h35
  have hε : tableErrAt true x j ≤ 2 ^ (-(205 : ℝ)) := by
    rw [tableErrAt_eq h]
    exact tableErr_zk_le spec st h25 h35 hst
  exact κAt_le_of (deployZk rel) (wordLenJ true) (tableErrAt true) j x hε (waits_deploy rel x).2.ge
    (cap_deploy rel x).2 (le_of_le_of_eq hlo hw.symm) (le_of_eq_of_le hw hhi)

/-- **At a masked session in scope, at every table of the statement, `κ_j ≤ 2^{−202.98}` and `λ_j ≥ 201.98`**,
where the session's own are `n·2^{−202.98}` and `201.98 − log₂ n` (`κ_zk_le`, `lam_zk_ge`). -/
theorem zkIn_at (rel : Readable.Parsed → Readable.Relation) {x : Claims.Msg} {s : Ins} {spec : SessionSpec}
    {sts : Array Stmt} (h : ZkIn x s spec sts) {j : ℕ} (hj : j < sts.size) :
    Claims.Flock.κAt (deployZk rel) (wordLenJ true) (tableErrAt true) j x ≤ 2 ^ (-(202.98 : ℝ)) ∧
      201.98 ≤ Claims.lamOf (Claims.Flock.κAt (deployZk rel) (wordLenJ true) (tableErrAt true) j x)
        (Claims.Flock.γ (deployZk rel) x) := by
  have hmem : sts[j] ∈ sts.toList := by
    first
      | exact Array.getElem_mem_toList hj
      | simp
  obtain ⟨h25, h35, hst⟩ := h.tab sts[j] hmem
  have ht : (tables true x)[j]? = some (spec, sts[j]) :=
    tables_getElem? h.setup (Array.getElem?_eq_getElem hj)
  have hw := wordLenJ_cast ht
  obtain ⟨hlo, hhi⟩ := tableLen_zk_bounds sts[j] h25 h35
  have hε0 : 0 ≤ tableErrAt true x j := by
    rw [tableErrAt_eq ht]
    exact tableErr_nonneg true spec sts[j]
  have hε : tableErrAt true x j ≤ 2 ^ (-(205 : ℝ)) := by
    rw [tableErrAt_eq ht]
    exact tableErr_zk_le spec sts[j] h25 h35 hst
  exact ⟨κAt_le_of (deployZk rel) (wordLenJ true) (tableErrAt true) j x hε (waits_deploy rel x).2.ge
      (cap_deploy rel x).2 (le_of_le_of_eq hlo hw.symm) (le_of_eq_of_le hw hhi),
    lamAt_ge_of (deployZk rel) (wordLenJ true) (tableErrAt true) j x hε0 hε (waits_deploy rel x).2.ge
      (cap_deploy rel x).2 (le_of_le_of_eq hlo hw.symm) (le_of_eq_of_le hw hhi)
      (γ_le_105_of (deployZk rel) x h.size_le (Vk_zk_le rel h) (weight_nonneg true x) (weight_zk_le rel h))⟩

/-- `k_j` at a word of `2^23` symbols is `12,343,986`. -/
theorem kAt_23 (N : Claims.Msg → ℕ → ℕ) (j : ℕ) (x : Claims.Msg) (h : N x j = 2 ^ 23) :
    Claims.Flock.kAt N j x = 12343986 := by
  have e : Claims.Flock.Numbers.kRep 23 = Claims.Flock.Numbers.kVal 23 :=
    Claims.Flock.Numbers.kRep_eq (by simp)
  have e' : Claims.Flock.Numbers.kVal 23 = 12343986 := by
    first
      | rfl
      | simp [Claims.Flock.Numbers.kVal]
  rw [e'] at e
  unfold Claims.Flock.Numbers.kRep at e
  unfold Claims.Flock.kAt
  rw [h, Nat.cast_pow, Nat.cast_ofNat]
  exact e

/-! ## Past the statement's tables -/

/-- Past the statement's tables the word is empty, `k_j = 0`, and the error at the table is at least `1`. -/
theorem one_le_κAt_past (rel : Readable.Parsed → Readable.Relation) (x : Claims.Msg) (j : ℕ)
    (hj : (tables true x).length ≤ j) :
    1 ≤ Claims.Flock.κAt (deployZk rel) (wordLenJ true) (tableErrAt true) j x := by
  have h0 : wordLenJ true x j = 0 := by
    simp [wordLenJ, List.getElem?_eq_none hj]
  have hk : Claims.Flock.kAt (wordLenJ true) j x = 0 := by
    unfold Claims.Flock.kAt
    rw [h0]
    simp
  have hε : 0 ≤ tableErrAt true x j := by
    unfold tableErrAt
    cases (tables true x)[j]? with
    | none => exact le_rfl
    | some p => exact tableErr_nonneg true p.1 p.2
  exact Claims.Flock.one_le_κAt (deployZk rel) (wordLenJ true) (tableErrAt true) j x hε hk

/-- At a statement that parses, the deployed `table` gives a statement at every index, past its tables too. -/
theorem table_ne_none {rel : Readable.Parsed → Readable.Relation} {x : Claims.Msg} {s : Claims.Flock.Stmt}
    (hs : parse true rel x = some s) (j : ℕ) : (deployZk rel).table x j ≠ none := by
  obtain ⟨p, -, -, -, -, htab⟩ := parsed_of_parse hs
  have h : (deployZk rel).table x j = some (Readable.toReadable rel (p.only j)) := htab j
  rw [h]
  exact Option.some_ne_none _

/-! ## At a statement of one table -/

/-- At a statement of one table, table `0`'s relation is the statement's. -/
theorem RAt_zero_iff_R {rel : Readable.Parsed → Readable.Relation} {x : Claims.Msg} {s : Claims.Flock.Stmt}
    (hs : parse true rel x = some s) (h1 : s.tables = 1) (w : Claims.Flock.Wit) :
    Claims.Flock.RAt (deployZk rel).table 0 x w ↔ Claims.Flock.R (deployZk rel).parse x w := by
  have ht : (deployZk rel).table x 0 = some s := table_zero_of_tables_eq_one hs h1
  have hp : (deployZk rel).parse x = some s := hs
  constructor
  · rintro ⟨t, h, hR⟩
    exact ⟨t, hp.trans (ht.symm.trans h), hR⟩
  · rintro ⟨t, h, hR⟩
    exact ⟨t, ht.trans (hp.symm.trans h), hR⟩

/-- `κAt` is `κ` where the table's error and word length are the session's. -/
theorem κAt_eq_κ_of {Coin Col T : Type} (D : Claims.Flock.Deploy Coin Col T) (N : Claims.Msg → ℕ → ℕ)
    (ε : Claims.Msg → ℕ → ℝ) (j : ℕ) (x : Claims.Msg) (hε : ε x j = D.εtable x) (hN : (N x j : ℝ) = D.N₀ x) :
    Claims.Flock.κAt D N ε j x = Claims.Flock.κ D x := by
  unfold Claims.Flock.κAt Claims.Flock.κ Claims.Flock.kAt Claims.Flock.Deploy.k
  rw [hε, hN]

/-- At a statement of one table, table `0`'s error is the session's. -/
theorem κAt_zero_eq_κ (rel : Readable.Parsed → Readable.Relation) {x : Claims.Msg} {spec : SessionSpec}
    {st : Stmt} (h : tables true x = [(spec, st)]) :
    Claims.Flock.κAt (deployZk rel) (wordLenJ true) (tableErrAt true) 0 x = Claims.Flock.κ (deployZk rel) x := by
  have h0 : (tables true x)[0]? = some (spec, st) := by simp [h]
  have hε : tableErrAt true x 0 = εtable true x := by
    rw [tableErrAt_eq h0]
    unfold εtable
    rw [h]
    simp
  have hN : (wordLenJ true x 0 : ℝ) = N₀ true x := by
    rw [wordLenJ_cast h0]
    unfold N₀
    rw [h]
    simp
  exact κAt_eq_κ_of (deployZk rel) (wordLenJ true) (tableErrAt true) 0 x hε hN

end FlockDeploy.KS.Land2Probe
