module

public import Proofs.Flock.Claims.KS.ZkExtend
public import Proofs.Flock.Claims.KS.KSTable
public import Proofs.Flock.Claims.KS.TableErrJ
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

`Flock.KSTable` at the deployed masked record, its tables' words' lengths (`wordLenJ true`) and SHA-512, at each table's
own error (`tableErrJ true`), with no premise (`zk_KSTable_untrim`): the extractor at one table of a session, whose
waits stop at `k_j = ⌈4·N_j/e⌉` accepted reruns, is knowledge sound at that table's own relation, and its knowledge
error reads that table's error alone, not `ε_table`, the tables' errors summed.

* **A table's own error** (`tableErrJ`, defined in `TableErrJ`) is at most `ε_table` (`tableErrJ_le_εtable`).
* **Each table's bound at its own error, counted per round** holds there with no premise (`zk_tableBoundEachR_untrim`):
  the live sessions' bound (`zk_tableBoundEachAccAtR_of_sessionsW`) at Flock ZK's padded tables
  (`zk_tableBoundEachAccAtR_live`), where no run accepts with no escapes
  (`Claims.Flock.tableBoundEachAccAtR_of_noAcc`), and `ExtendJ`. It is what `zk_oneWait_untrim` is assembled from,
  before the tables' errors are summed and the tables joined.
* **A table's word is no longer than `N₀`** (`wordLenJ_le_N₀`).
* **The `k` range at each table** (`kTable_range_zk`): where some run accepts, each of the statement's tables has a word
  of at least `2^10` columns, so `k_j ≥ 8`; past the statement's tables the word is empty and `k_j = 0`.
-/

@[expose] public section

namespace Claims.Flock

/-- **Where the verifier accepts no run**, `TableBoundEachAccAtR` holds with no escapes at any nonnegative errors,
when the verifier takes a round and the weight covers `k`. -/
theorem tableBoundEachAccAtR_of_noAcc {Coin Col T : Type} [Fintype Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]
    (H : Msg → Msg) (D : Deploy Coin Col T) (N : Msg → ℕ → ℕ) (Ok : Msg → ℕ → Col → Prop) (ε : ℕ → ℝ) (x : Msg)
    (hno : NoAcc H D.V x) (hk : (D.k x : ℝ) ≤ 2 * D.weight x) (hε : ∀ j < nTab D x, 0 ≤ ε j)
    (hVk : 0 < D.V.k x) : TableBoundEachAccAtR H D N Ok ε x := by
  intro Ω _ _ P
  refine ⟨Escapes.nil H D.V P x, (Escapes.onAcc_nil H D.V P x).onAcc₁,
    (Escapes.fitsFork_nil H D.V P x _ _ hk).fitsForkR (Nat.cast_nonneg _) hVk, fun ω j hj C _ _ => ?_⟩
  have h0 : accPr H D.V P x ω = 0 := by
    unfold accPr
    rw [Pr_congr (q := fun _ => False) fun r => ⟨fun h => hno Ω P ω r h, False.elim⟩, Pr_false]
  rw [h0, Escapes.term_nil]
  have := Pr_nonneg fun r : Fin (D.V.k x) → Coin => (run H P D.V x ω r).out = .acc ∧
    ∃ (p : Fin (N x j)) (u : Col), lookCol (D.cols (run H P D.V x ω r).seen j) p.val = some u ∧ u ≠ C p
  have := hε j hj
  linarith

end Claims.Flock

namespace FlockDeploy.KS

open Flock

namespace Gen

variable [S : ZkSetup]

theorem tableErrJ_eq {zk : Bool} {x : Claims.Msg} {spec : SessionSpec} {st : Stmt} {j : ℕ}
    (h : (tables zk x)[j]? = some (spec, st)) : tableErrJ zk x j = tableErr zk spec st := by
  unfold tableErrJ
  rw [h]
  rfl

theorem tableErrJ_nonneg (zk : Bool) (x : Claims.Msg) (j : ℕ) : 0 ≤ tableErrJ zk x j := by
  unfold tableErrJ
  cases (tables zk x)[j]? with
  | none => exact le_rfl
  | some p => exact tableErr_nonneg zk p.1 p.2

/-- **A table's own error is at most `ε_table`**, the tables' errors summed. -/
theorem tableErrJ_le_εtable (zk : Bool) (x : Claims.Msg) (j : ℕ) : tableErrJ zk x j ≤ εtable zk x := by
  cases h : (tables zk x)[j]? with
  | none =>
    unfold tableErrJ
    rw [h]
    exact εtable_nonneg zk x
  | some p =>
    rw [tableErrJ_eq (spec := p.1) (st := p.2) h]
    exact tableErr_le_εtable (spec := p.1) (st := p.2) h

section Live

open FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkSession

/-- **Every masked table's bound at its own error at a statement in scope, off the table on a witness lane**, the
escapes counted per round: `zk_tableBoundEachAccAtR_of_sessionsW` at Flock ZK's padded tables (`zk_atTabs`) and the
live sessions' bound there (`zkSessWJ_of_liveW`). -/
theorem zk_tableBoundEachAccAtR_live {x : Claims.Msg} {s : Claims.Flock.Stmt}
    (hs : parse true Readable.relation x = some s) (hin : Claims.Flock.InScope true s) {ins : Ins}
    {spec : SessionSpec} {sts : Array Stmt} (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (hl : ZkLiveAccWJ x) :
    Claims.Flock.TableBoundEachAccAtR sha512 (deployZk Readable.relation) (wordLenJ true) (wideJ true)
      (tableErrJ true x) x := by
  obtain ⟨L, hz, hk, hlay, hb⟩ := zk_atTabs hs hin hset h0
    (B := fun T => ZkLiveAccAtWJ execArith Discharge.ZkExec.hs512 (wideJ true) x T →
      Claims.Flock.TableBoundEachAccAtR sha512 (deployZk Readable.relation) (wordLenJ true) (wideJ true)
        (tableErrJ true x) x)
    fun T hn hst hT hnc hlT => (zk_tableBoundEachAccAtR_of_sessionsW execArith execArith_correct
      Discharge.ZkExec.hs512 Readable.relation (wideJ true) hn (stJ h0) hst T hT hnc (zkSessWJ_of_liveW hlT)).mono
        (ε' := tableErrJ true x) fun j hj => (tableErrJ_eq (hst j (lt_of_lt_of_eq hj hn))).ge
  exact hb (hl ins spec sts hset h0 L hz hk hlay)

end Live

/-- **Every masked table's bound at its own error at every statement, off the table on a witness lane**, the escapes
counted per round, at full-width columns, from `ZkLiveAccWJ` at the statements some run accepts; elsewhere no run
accepts. -/
theorem zk_tableBoundEachAccAtR (hl : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkLiveAccWJ x) (x : Claims.Msg) :
    Claims.Flock.TableBoundEachAccAtR sha512 (deployZk Readable.relation) (wordLenJ true) (wideJ true)
      (tableErrJ true x) x := by
  by_cases hno : Claims.Flock.NoAcc sha512 (V true) x
  · exact Claims.Flock.tableBoundEachAccAtR_of_noAcc _ _ _ _ _ x hno (zk_k_le Readable.relation x)
      (fun j _ => tableErrJ_nonneg true x j) (V_k_pos true x)
  · obtain ⟨s, hs, hin⟩ := acc_inScope Readable.relation hno
    obtain ⟨ins, spec, sts, record, proofs, hset, hv, -⟩ := acc_check hno
    exact zk_tableBoundEachAccAtR_live hs hin hset (served_zk (by simpa using hv)).1 (hl x hno)

/-- **Each masked table's bound at its own error, counted per round, at the untrimmed instance**, at SHA-512 at the
deployed relation, with no premise: the live sessions' bound at the deployed decoders (`TrimBridge.zkXferWJ_of_kinds`
at its fork bounds and the view's pads clash) and `ExtendJ` (`zk_extendJ`). -/
theorem zk_tableBoundEachR_untrim :
    Claims.Flock.TableBoundEachR sha512 (deployZk Readable.relation) (wordLenJ true)
      (Claims.Flock.orZeroJ (wideJ true)) (tableErrJ true) :=
  Claims.Flock.tableBoundEachR_of_extend _ _ _ _ _ (wideJ_ne_zero true) (zk_colsOkJ sha512 Readable.relation)
    (zk_extendJ Readable.relation)
    (Claims.Flock.tableBoundEachR_of_eachAccAtR fun x => zk_tableBoundEachAccAtR
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
theorem kTable_range_zk (rel : Readable.Parsed → Readable.Relation) (x : Claims.Msg) (j : ℕ) :
    (j < Claims.Flock.nTab (deployZk rel) x ∧ 8 ≤ Claims.Flock.kTable (wordLenJ true) j x) ∨
      Claims.Flock.kTable (wordLenJ true) j x = 0 ∨ Claims.Flock.NoAcc sha512 (deployZk rel).V x := by
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
    unfold Claims.Flock.kTable
    omega
  · refine .inr (.inl ?_)
    have hge : (tables true x).length ≤ j := by
      rw [hlen]
      omega
    have h0 : wordLenJ true x j = 0 := by
      simp [wordLenJ, List.getElem?_eq_none hge]
    unfold Claims.Flock.kTable
    rw [h0]
    simp

/-- **Masked knowledge soundness at each table of a session, at the untrimmed instance**, at SHA-512 at the deployed
relation, with no premise: `Flock.KSTable` at the tables' words' lengths and each table's own error
(`Claims.Flock.KSTable_of_tableBoundEachR` at `zk_tableBoundEachR_untrim`). -/
theorem zk_KSTable_untrim :
    Claims.Flock.KSTable sha512 (deployZk Readable.relation) (wordLenJ true) (tableErrJ true) :=
  Claims.Flock.KSTable_of_tableBoundEachR sha512 (deployZk Readable.relation) (wordLenJ true) (tableErrJ true)
    (fun x j _ => wordLenJ_le_N₀ true x j) (tableErrJ_nonneg true) (V_k_pos true) (fun _ => cap_pos)
    (kTable_range_zk Readable.relation) (Claims.Flock.orZeroJ (wideJ true)) (fun _ _ => .inl rfl)
    (Claims.Flock.colsOkJ_orZero _ _ _ _ (zk_colsOkJ sha512 _)) (zk_conflictTiedJ _) zk_tableBoundEachR_untrim

end Gen

/-- **A table's own error is at most `ε_table`**, the tables' errors summed. -/
theorem tableErrJ_le_εtable (zk : Bool) (x : Claims.Msg) (j : ℕ) : tableErrJ zk x j ≤ εtable zk x :=
  Gen.tableErrJ_le_εtable (S := S₀) (zk := zk) (x := x) (j := j)

/-- **Masked knowledge soundness at each table of a session, at the untrimmed instance**, at SHA-512 at the deployed
relation, with no premise, at each table's own error (`Gen.zk_KSTable_untrim`). -/
theorem zk_KSTable_untrim :
    Claims.Flock.KSTable sha512 (deployZk Readable.relation) (wordLenJ true) (tableErrJ true) :=
  Gen.zk_KSTable_untrim (S := S₀)

end FlockDeploy.KS
