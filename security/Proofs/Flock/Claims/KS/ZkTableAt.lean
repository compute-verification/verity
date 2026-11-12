module

public import Proofs.Flock.Claims.KS.ZkExtend
public import Proofs.Flock.Claims.KS.TableAt
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
import Proofs.Flock.Claims.DeployTables
import Proofs.Flock.Claims.KS.Deploy
import Proofs.Flock.Claims.KS.JointR
import Proofs.Flock.Claims.KS.Range
import Proofs.Flock.Claims.KS.TableBound
import Proofs.Flock.Claims.KS.ZkClashPads
import Proofs.Flock.Claims.KS.ZkLiveW
import Proofs.Flock.Claims.KS.ZkPadsSlot
import Proofs.Flock.Claims.KS.ZkXferW

/-!
# Masked knowledge soundness at each table of a session, at the untrimmed instance

`Flock.KSAt` at the deployed masked record, its tables' words' lengths (`wordLenJ true`) and SHA-512, at the tables'
errors summed, with no premise (`zk_KSAt_untrim`): the extractor at one table of a session, whose waits stop at
`k_j = ⌈4·N_j/e⌉` accepted reruns, is knowledge sound at that table's own relation.

* **Every table's bound counted per round** holds there with no premise (`zk_tableBoundJR_untrim`): what
  `zk_oneWait_untrim` is assembled from, before the tables are joined.
* **A table's word is no longer than `N₀`** (`wordLenJ_le_N₀`).
* **The `k` range at each table** (`kAt_range_zk`): where some run accepts, each of the statement's tables has a word
  of at least `2^10` columns, so `k_j ≥ 8`; past the statement's tables the word is empty and `k_j = 0`.
-/

@[expose] public section

namespace FlockDeploy.KS

open Flock

namespace Gen

variable [S : ZkSetup]

/-- **Every masked table's bound counted per round at the untrimmed instance**, at SHA-512 at the deployed relation,
with no premise: the live sessions' bound at the deployed decoders (`TrimBridge.zkXferWJ_of_kinds` at the cap/`τ` and
pads fork bounds and the view's pads clash) and `ExtendJ` (`zk_extendJ`). -/
theorem zk_tableBoundJR_untrim :
    Claims.Flock.TableBoundJR sha512 (deployZk Readable.relation) (wordLenJ true)
      (Claims.Flock.orZeroJ (wideJ true)) :=
  zk_tableBoundJR_of_extend Readable.relation (zk_extendJ Readable.relation)
    (Claims.Flock.tableBoundJR_of_accAtR fun x => zk_tableBoundJAccAtR
      (fun x hno => zkLiveAccWJ_of_xferWJ hno
        (TrimBridge.zkXferWJ_of_kinds (fun x _ => zkSimForkLeCapTau x) (fun x _ => zkSimForkLePads x)
          (fun x _ => zkViewClashQP x) x hno)) x)

/-- A table's word is no longer than `N₀`, the tables' words' lengths summed. -/
theorem wordLenJ_le_N₀ (zk : Bool) (x : Claims.Msg) (j : ℕ) : (wordLenJ zk x j : ℝ) ≤ N₀ zk x := by
  have h1 : wordLenJ zk x j ≤ ∑ i : Fin (j + 1), wordLenJ zk x i :=
    Finset.single_le_sum (f := fun i : Fin (j + 1) => wordLenJ zk x i) (fun _ _ => Nat.zero_le _)
      (Finset.mem_univ (Fin.last j))
  exact le_trans (Nat.cast_le.2 h1) (sum_wordLenJ_le zk x (j + 1))

/-- **The masked verifier's `k` range at each table**: at every statement and table, the table is one of the
statement's and `k_j ≥ 8`, or its word is empty and `k_j = 0`, or the verifier accepts no run at SHA-512. -/
theorem kAt_range_zk (rel : Readable.Parsed → Readable.Relation) (x : Claims.Msg) (j : ℕ) :
    (j < Claims.Flock.nTab (deployZk rel) x ∧ 8 ≤ Claims.Flock.kAt (wordLenJ true) j x) ∨
      Claims.Flock.kAt (wordLenJ true) j x = 0 ∨ Claims.Flock.NoAcc sha512 (deployZk rel).V x := by
  by_cases hacc : Claims.Flock.NoAcc sha512 (deployZk rel).V x
  · exact .inr (.inr hacc)
  obtain ⟨ins, spec, sts, record, proofs, hs, hv⟩ := acc_verdict (zk := true) hacc
  obtain ⟨-, hm⟩ := served_zk (by simpa using hv)
  obtain ⟨s, hp, -⟩ := acc_inScope rel (zk := true) hacc
  have hlen : (tables true x).length = sts.size := by
    unfold tables
    rw [hs]
    simp
  by_cases hj : j < sts.size
  · have hn : Claims.Flock.nTab (deployZk rel) x = (tables true x).length := by
      rw [Claims.Flock.nTab_of_parse (D := deployZk rel) hp, tables_length hp (by rw [hs]; rfl)]
    refine .inl ⟨by rw [hn, hlen]; exact hj, ?_⟩
    obtain ⟨h25, h35⟩ := hm j hj
    have hd := zkLen_ge ⟨sts[j].m - 25, by omega⟩
    have hi : Claims.Flock.ZkIdx.m ⟨sts[j].m - 25, by omega⟩ = sts[j].m := by
      simp only [Claims.Flock.ZkIdx.m]
      omega
    have hd10 : 10 ≤ (Claims.Flock.zkLevel0 sts[j].m).d := by
      unfold Claims.Flock.zkLen at hd
      rw [hi] at hd
      exact hd
    have hw : wordLenJ true x j = 2 ^ (Claims.Flock.zkLevel0 sts[j].m).d := by
      simp [wordLenJ, tables_getElem? hs (Array.getElem?_eq_getElem hj)]
    have hN : (2 : ℝ) ^ 10 ≤ (wordLenJ true x j : ℝ) := by
      rw [hw, Nat.cast_pow, Nat.cast_ofNat]
      exact two_pow_le hd10
    have := Claims.Flock.Numbers.k_ge hN
    unfold Claims.Flock.kAt
    omega
  · refine .inr (.inl ?_)
    have hge : (tables true x).length ≤ j := by
      rw [hlen]
      omega
    have h0 : wordLenJ true x j = 0 := by
      simp [wordLenJ, List.getElem?_eq_none hge]
    unfold Claims.Flock.kAt
    rw [h0]
    simp

/-- **Masked knowledge soundness at each table of a session, at the untrimmed instance**, at SHA-512 at the deployed
relation, with no premise: `Flock.KSAt` at the tables' words' lengths and the tables' errors summed
(`Claims.Flock.KSAt_of_tableBoundJR` at `zk_tableBoundJR_untrim`). -/
theorem zk_KSAt_untrim :
    Claims.Flock.KSAt sha512 (deployZk Readable.relation) (wordLenJ true)
      (fun x _ => (deployZk Readable.relation).εtable x) :=
  Claims.Flock.KSAt_of_tableBoundJR sha512 (deployZk Readable.relation) (wordLenJ true)
    (fun x j _ => wordLenJ_le_N₀ true x j) (fun x => εtable_nonneg true x) (V_k_pos true) (fun _ => cap_pos)
    (kAt_range_zk Readable.relation) (Claims.Flock.orZeroJ (wideJ true)) (fun _ _ => .inl rfl)
    (Claims.Flock.colsOkJ_orZero _ _ _ _ (zk_colsOkJ sha512 _)) (zk_conflictTiedJ _) zk_tableBoundJR_untrim

end Gen

/-- **Masked knowledge soundness at each table of a session, at the untrimmed instance**, at SHA-512 at the deployed
relation, with no premise (`Gen.zk_KSAt_untrim`). -/
theorem zk_KSAt_untrim :
    Claims.Flock.KSAt sha512 (deployZk Readable.relation) (wordLenJ true)
      (fun x _ => (deployZk Readable.relation).εtable x) :=
  Gen.zk_KSAt_untrim (S := S₀)

end FlockDeploy.KS
