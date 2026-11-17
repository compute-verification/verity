module

public import Proofs.Flock.Claims.Complete.WalkPacked
public import Proofs.Flock.Claims.Complete.ZkTables.RingSwitch
public import Proofs.Flock.Claims.DeployEfficient
public import Proofs.Flock.Claims.KS.PlainAcc
public import Proofs.Flock.Claims.ZK.Hypotheses
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
import Proofs.Flock.Claims.Bits
import Proofs.Flock.Claims.Complete.ZkTables.StmtOne
import Proofs.Flock.Claims.KS.ZkTauSlot

/-! Flock's claims at the deployed records (`flock_plain_assembled`, `flock_zk_assembled`), with what remains as
hypotheses. -/

@[expose] public section

namespace FlockDeploy.Assembled

open Flock

theorem tableErr_nonneg (zk : Bool) (spec : SessionSpec) (st : Stmt) : 0 ≤ tableErr zk spec st := by
  unfold tableErr
  split
  · unfold FlockSoundness.Accounting.padLinkError
    positivity
  · exact FlockSoundness.tableError_nonneg _ _

theorem εtable_nonneg (zk : Bool) (x : Claims.Msg) : 0 ≤ εtable zk x := by
  unfold εtable
  refine List.sum_nonneg fun a ha => ?_
  obtain ⟨p, -, rfl⟩ := List.mem_map.1 ha
  exact tableErr_nonneg zk p.1 p.2

theorem V_k_pos (zk : Bool) (x : Claims.Msg) : 0 < (V zk).k x := by
  simp only [V]
  split <;> omega

theorem γ_pos {Coin Col T : Type} (D : Claims.Flock.Deploy Coin Col T) (x : Claims.Msg) (hk : 0 < D.V.k x) :
    0 < Claims.Flock.γ D x := by
  unfold Claims.Flock.γ
  have hk' : (0 : ℝ) < D.V.k x := by exact_mod_cast hk
  have hs := Real.sqrt_nonneg (2 * D.V.k x * D.weight x)
  have h : 0 < Real.sqrt (2 * D.V.k x * D.weight x) + D.V.k x := by linarith
  exact mul_pos (by norm_num) (pow_pos h 2)

theorem bits_plain (rel : Readable.Parsed → Readable.Relation) (H : Claims.Msg → Claims.Msg)
    (hKS : Claims.Flock.KS H (deployPlainTrim rel)) : Claims.Flock.Bits H (deployPlainTrim rel) :=
  Claims.Flock.Bits_of_KS hKS (εtable_nonneg false)

theorem bits_zk (rel : Readable.Parsed → Readable.Relation) (H : Claims.Msg → Claims.Msg)
    (hKS : Claims.Flock.KS H (deployZkTrim rel)) : Claims.Flock.Bits H (deployZkTrim rel) :=
  Claims.Flock.Bits_of_KS hKS (εtable_nonneg true)

theorem bits_ge_plain (rel : Readable.Parsed → Readable.Relation) (H : Claims.Msg → Claims.Msg)
    (hKS : Claims.Flock.KS H (deployPlainTrim rel)) (a b : Claims.Msg → ℝ)
    (hκ : ∀ x, Claims.Flock.κ (deployPlainTrim rel) x ≤ 2 ^ (-a x))
    (hγ : ∀ x, Claims.Flock.γ (deployPlainTrim rel) x ≤ 2 ^ b x) :
    Claims.Bits H (deployPlainTrim rel).V (Claims.Flock.R (deployPlainTrim rel).parse)
      (Claims.Flock.E (deployPlainTrim rel) (deployPlainTrim rel).decode) fun x => min (a x) (255.5 - b x / 2) - 1 :=
  Claims.Flock.Bits_ge hKS (εtable_nonneg false) hκ (fun x => γ_pos (deployPlainTrim rel) x (V_k_pos false x)) hγ _
    (decode_plain rel ▸ listWithin_plainTrim rel)

theorem bits_ge_zk (rel : Readable.Parsed → Readable.Relation) (H : Claims.Msg → Claims.Msg)
    (hKS : Claims.Flock.KS H (deployZkTrim rel)) (a b : Claims.Msg → ℝ)
    (hκ : ∀ x, Claims.Flock.κ (deployZkTrim rel) x ≤ 2 ^ (-a x))
    (hγ : ∀ x, Claims.Flock.γ (deployZkTrim rel) x ≤ 2 ^ b x) :
    Claims.Bits H (deployZkTrim rel).V (Claims.Flock.R (deployZkTrim rel).parse)
      (Claims.Flock.E (deployZkTrim rel) (deployZkTrim rel).decode) fun x => min (a x) (255.5 - b x / 2) - 1 :=
  Claims.Flock.Bits_ge hKS (εtable_nonneg true) hκ (fun x => γ_pos (deployZkTrim rel) x (V_k_pos true x)) hγ _
    (decode_zk rel ▸ listWithin_zkTrim rel)

/-- What remains is exactly `hKS` and `hComplete`. -/
theorem flock_plain_assembled (rel : Readable.Parsed → Readable.Relation) (H : Claims.Msg → Claims.Msg)
    (hKS : Claims.Flock.KS H (deployPlainTrim rel))
    (hComplete : Claims.Complete H (deployPlainTrim rel).V FlockComplete.honestP
      (Claims.Flock.R (deployPlainTrim rel).parse) (Claims.Flock.Scope (deployPlainTrim rel)) (FlockComplete.δc H)) :
    Claims.Flock.ScopeMatches (deployPlainTrim rel) ∧
      Claims.Flock.KS H (deployPlainTrim rel) ∧
      Claims.Flock.Bits H (deployPlainTrim rel) ∧
      (∀ a b : Claims.Msg → ℝ, (∀ x, Claims.Flock.κ (deployPlainTrim rel) x ≤ 2 ^ (-a x)) →
        (∀ x, Claims.Flock.γ (deployPlainTrim rel) x ≤ 2 ^ b x) →
        Claims.Bits H (deployPlainTrim rel).V (Claims.Flock.R (deployPlainTrim rel).parse)
          (Claims.Flock.E (deployPlainTrim rel) (deployPlainTrim rel).decode)
          fun x => min (a x) (255.5 - b x / 2) - 1) ∧
      Claims.Flock.Efficient H (deployPlainTrim rel) plainStepPoly ∧
      Claims.Complete H (deployPlainTrim rel).V FlockComplete.honestP (Claims.Flock.R (deployPlainTrim rel).parse)
        (Claims.Flock.Scope (deployPlainTrim rel)) (FlockComplete.δc H) :=
  ⟨scopeMatches_plain rel, hKS, bits_plain rel H hKS, bits_ge_plain rel H hKS, efficient_plain H rel, hComplete⟩

theorem _root_.Claims.Flock.complete_mapCols {Coin Col T : Type} [Fintype Coin] [Inhabited Coin]
    (H : Claims.Msg → Claims.Msg) (D : Claims.Flock.Deploy Coin Col T) (f : Col → Col) :
    Claims.Flock.Complete H (D.mapCols f) ↔ Claims.Flock.Complete H D :=
  Iff.rfl

/-- `hKS` is what remains. -/
theorem flock_plain_assembled_relation (H : Claims.Msg → Claims.Msg)
    (hKS : Claims.Flock.KS H (deployPlainTrim Readable.relation)) :
    Claims.Flock.ScopeMatches (deployPlainTrim Readable.relation) ∧
      Claims.Flock.KS H (deployPlainTrim Readable.relation) ∧
      Claims.Flock.Bits H (deployPlainTrim Readable.relation) ∧
      (∀ a b : Claims.Msg → ℝ, (∀ x, Claims.Flock.κ (deployPlainTrim Readable.relation) x ≤ 2 ^ (-a x)) →
        (∀ x, Claims.Flock.γ (deployPlainTrim Readable.relation) x ≤ 2 ^ b x) →
        Claims.Bits H (deployPlainTrim Readable.relation).V (Claims.Flock.R (deployPlainTrim Readable.relation).parse)
          (Claims.Flock.E (deployPlainTrim Readable.relation) (deployPlainTrim Readable.relation).decode)
          fun x => min (a x) (255.5 - b x / 2) - 1) ∧
      Claims.Flock.Efficient H (deployPlainTrim Readable.relation) plainStepPoly ∧
      Claims.Flock.Complete H (deployPlainTrim Readable.relation) :=
  flock_plain_assembled Readable.relation H hKS
    ((Claims.Flock.complete_mapCols H (deployPlain Readable.relation) trimCol).2
      (FlockComplete.Replay.complete_plain_json H))

/-- The live sessions' bound with escapes on accepted runs is what remains. -/
theorem flock_plain_assembled_trim
    (hl : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V false) x → FlockDeploy.KS.PlainLiveAcc x) :
    Claims.Flock.ScopeMatches (deployPlainTrim Readable.relation) ∧
      Claims.Flock.KS sha512 (deployPlainTrim Readable.relation) ∧
      Claims.Flock.Bits sha512 (deployPlainTrim Readable.relation) ∧
      (∀ a b : Claims.Msg → ℝ, (∀ x, Claims.Flock.κ (deployPlainTrim Readable.relation) x ≤ 2 ^ (-a x)) →
        (∀ x, Claims.Flock.γ (deployPlainTrim Readable.relation) x ≤ 2 ^ b x) →
        Claims.Bits sha512 (deployPlainTrim Readable.relation).V
          (Claims.Flock.R (deployPlainTrim Readable.relation).parse)
          (Claims.Flock.E (deployPlainTrim Readable.relation) (deployPlainTrim Readable.relation).decode)
          fun x => min (a x) (255.5 - b x / 2) - 1) ∧
      Claims.Flock.Efficient sha512 (deployPlainTrim Readable.relation) plainStepPoly ∧
      Claims.Flock.Complete sha512 (deployPlainTrim Readable.relation) :=
  flock_plain_assembled_relation sha512 (FlockDeploy.KS.plain_KS_of_liveAcc hl)

/-- What remains is exactly `hKS`, `hComplete` and `Flock.zk_deployZk_of_models`' hypotheses, `hT1` its named
assumption. -/
theorem flock_zk_assembled (rel : Readable.Parsed → Readable.Relation) (H : Claims.Msg → Claims.Msg)
    (hKS : Claims.Flock.KS H (deployZkTrim rel))
    (hComplete : Claims.Complete H (deployZkTrim rel).V FlockComplete.ZkTables.honestPR
      (Claims.Flock.R (deployZkTrim rel).parse) (Claims.Flock.Scope (deployZkTrim rel)) (FlockComplete.ZkTables.δc H))
    (G : Claims.ZKMal.Proto Claims.Flock.Wit) (t : ℕ) (ht : 0 < t) (hK : ∀ x, G.K x = (V true).k x - 1)
    {Salt B Cc : Type} [AddCommGroup Salt] [AddCommGroup B] [Fintype Salt] [DecidableEq Salt] [Fintype B]
    [DecidableEq B] (My : Salt → B) (cs : Salt → Cc) {δ₁ : ENNReal} (hT1 : Claims.Hm96Hiding My cs δ₁) (hδ₁ : δ₁ ≠ ⊤)
    (Mp : ∀ x, G.Nonce → G.PreModel x Salt B Cc) (Mc : ∀ x, G.Nonce → (Fin (G.K x) → G.Coin) → G.CoinModel x Salt B Cc)
    (nL nB : Claims.Msg → ℕ) (hnL : ∀ x ν, Fintype.card (Mp x ν).L ≤ nL x) (hnB : ∀ x ν c, (Mc x ν c).size ≤ nB x)
    (hL : G.MsgLink H (Claims.Flock.R (deployZkTrim rel).parse) My cs Mp Mc) (hS : G.SimLink H Mc)
    (hN : G.NonDegenerate Mc) (hI : G.InnerHolds (Claims.Flock.R (deployZkTrim rel).parse) Mc) (hC : G.CoinCheck)
    (coinOf : Claims.Msg → Claims.Msg) (tapeOf : G.Nonce × G.Rand → FlockComplete.TapeZk)
    (hG : G.Honest H FlockComplete.ZkTables.honestPR coinOf tapeOf) :
    Claims.Flock.ScopeMatches (deployZkTrim rel) ∧
      Claims.Flock.KS H (deployZkTrim rel) ∧
      Claims.Flock.Bits H (deployZkTrim rel) ∧
      (∀ a b : Claims.Msg → ℝ, (∀ x, Claims.Flock.κ (deployZkTrim rel) x ≤ 2 ^ (-a x)) →
        (∀ x, Claims.Flock.γ (deployZkTrim rel) x ≤ 2 ^ b x) →
        Claims.Bits H (deployZkTrim rel).V (Claims.Flock.R (deployZkTrim rel).parse)
          (Claims.Flock.E (deployZkTrim rel) (deployZkTrim rel).decode) fun x => min (a x) (255.5 - b x / 2) - 1) ∧
      Claims.Flock.ZK H (Claims.ZKMal.zkOf rel G coinOf tapeOf (G.simulatorE t (t * 3 ^ t)))
        (fun x => 8 * nL x * δ₁.toReal + 8 * t * nB x * δ₁.toReal + 2 * (1 / 3) ^ t) (fun _ => 4 * t * 3 ^ t) ∧
      Claims.Flock.Efficient H (deployZkTrim rel) zkStepPoly ∧
      Claims.Complete H (deployZkTrim rel).V FlockComplete.ZkTables.honestPR (Claims.Flock.R (deployZkTrim rel).parse)
        (Claims.Flock.Scope (deployZkTrim rel)) (FlockComplete.ZkTables.δc H) :=
  ⟨scopeMatches_zk rel, hKS, bits_zk rel H hKS, bits_ge_zk rel H hKS,
    Claims.ZKMal.Flock.zk_deployZk_of_models H rel G t ht hK My cs hT1 hδ₁ Mp Mc nL nB hnL hnB hL hS hN hI hC coinOf
      tapeOf hG,
    efficient_zk H rel, hComplete⟩

/-- `hKS`, the honest masked run's premises, the mask slots' rank bound and `Flock.zk_deployZk_of_models`' hypotheses
are what remains. -/
theorem flock_zk_assembled_relation (H : Claims.Msg → Claims.Msg)
    (hKS : Claims.Flock.KS H (deployZkTrim Readable.relation))
    (hPR : ∀ c : FlockComplete.ZkTables.Case H Readable.relation, FlockComplete.ZkTables.ProverRunsAt c)
    (hR : ∀ c : FlockComplete.ZkTables.Case H Readable.relation, FlockComplete.ZkTables.RepShapedAt c)
    (hLI : ∀ o : FlockComplete.ZkTables.Run H Readable.relation,
      FlockComplete.ZkTables.LgInnerAt H Readable.relation o)
    (hMR : FlockComplete.MaskRank.MaskRankBound)
    (G : Claims.ZKMal.Proto Claims.Flock.Wit) (t : ℕ) (ht : 0 < t) (hK : ∀ x, G.K x = (V true).k x - 1)
    {Salt B Cc : Type} [AddCommGroup Salt] [AddCommGroup B] [Fintype Salt] [DecidableEq Salt] [Fintype B]
    [DecidableEq B] (My : Salt → B) (cs : Salt → Cc) {δ₁ : ENNReal} (hT1 : Claims.Hm96Hiding My cs δ₁) (hδ₁ : δ₁ ≠ ⊤)
    (Mp : ∀ x, G.Nonce → G.PreModel x Salt B Cc) (Mc : ∀ x, G.Nonce → (Fin (G.K x) → G.Coin) → G.CoinModel x Salt B Cc)
    (nL nB : Claims.Msg → ℕ) (hnL : ∀ x ν, Fintype.card (Mp x ν).L ≤ nL x) (hnB : ∀ x ν c, (Mc x ν c).size ≤ nB x)
    (hL : G.MsgLink H (Claims.Flock.R (deployZkTrim Readable.relation).parse) My cs Mp Mc) (hS : G.SimLink H Mc)
    (hN : G.NonDegenerate Mc) (hI : G.InnerHolds (Claims.Flock.R (deployZkTrim Readable.relation).parse) Mc)
    (hC : G.CoinCheck) (coinOf : Claims.Msg → Claims.Msg) (tapeOf : G.Nonce × G.Rand → FlockComplete.TapeZk)
    (hG : G.Honest H FlockComplete.ZkTables.honestPR coinOf tapeOf) :
    Claims.Flock.ScopeMatches (deployZkTrim Readable.relation) ∧
      Claims.Flock.KS H (deployZkTrim Readable.relation) ∧
      Claims.Flock.Bits H (deployZkTrim Readable.relation) ∧
      (∀ a b : Claims.Msg → ℝ, (∀ x, Claims.Flock.κ (deployZkTrim Readable.relation) x ≤ 2 ^ (-a x)) →
        (∀ x, Claims.Flock.γ (deployZkTrim Readable.relation) x ≤ 2 ^ b x) →
        Claims.Bits H (deployZkTrim Readable.relation).V (Claims.Flock.R (deployZkTrim Readable.relation).parse)
          (Claims.Flock.E (deployZkTrim Readable.relation) (deployZkTrim Readable.relation).decode)
          fun x => min (a x) (255.5 - b x / 2) - 1) ∧
      Claims.Flock.ZK H (Claims.ZKMal.zkOf Readable.relation G coinOf tapeOf (G.simulatorE t (t * 3 ^ t)))
        (fun x => 8 * nL x * δ₁.toReal + 8 * t * nB x * δ₁.toReal + 2 * (1 / 3) ^ t) (fun _ => 4 * t * 3 ^ t) ∧
      Claims.Flock.Efficient H (deployZkTrim Readable.relation) zkStepPoly ∧
      Claims.Flock.Complete H (deployZkTrim Readable.relation) :=
  flock_zk_assembled Readable.relation H hKS
    ((Claims.Flock.complete_mapCols H (deployZk Readable.relation) trimCol).2
      (FlockComplete.ZkTables.complete_json H hPR hR hLI hMR))
    G t ht hK My cs hT1 hδ₁ Mp Mc nL nB hnL hnB hL hS hN hI hC coinOf tapeOf hG

/-- The honest masked run's premises, the mask slots' rank bound and `Flock.zk_deployZk_of_models`' hypotheses, `hT1`
its named assumption, are what remain. -/
theorem flock_zk_assembled_trim
    (hPR : ∀ c : FlockComplete.ZkTables.Case sha512 Readable.relation, FlockComplete.ZkTables.ProverRunsAt c)
    (hR : ∀ c : FlockComplete.ZkTables.Case sha512 Readable.relation, FlockComplete.ZkTables.RepShapedAt c)
    (hLI : ∀ o : FlockComplete.ZkTables.Run sha512 Readable.relation,
      FlockComplete.ZkTables.LgInnerAt sha512 Readable.relation o)
    (hMR : FlockComplete.MaskRank.MaskRankBound)
    (G : Claims.ZKMal.Proto Claims.Flock.Wit) (t : ℕ) (ht : 0 < t) (hK : ∀ x, G.K x = (V true).k x - 1)
    {Salt B Cc : Type} [AddCommGroup Salt] [AddCommGroup B] [Fintype Salt] [DecidableEq Salt] [Fintype B]
    [DecidableEq B] (My : Salt → B) (cs : Salt → Cc) {δ₁ : ENNReal} (hT1 : Claims.Hm96Hiding My cs δ₁) (hδ₁ : δ₁ ≠ ⊤)
    (Mp : ∀ x, G.Nonce → G.PreModel x Salt B Cc) (Mc : ∀ x, G.Nonce → (Fin (G.K x) → G.Coin) → G.CoinModel x Salt B Cc)
    (nL nB : Claims.Msg → ℕ) (hnL : ∀ x ν, Fintype.card (Mp x ν).L ≤ nL x) (hnB : ∀ x ν c, (Mc x ν c).size ≤ nB x)
    (hL : G.MsgLink sha512 (Claims.Flock.R (deployZkTrim Readable.relation).parse) My cs Mp Mc)
    (hS : G.SimLink sha512 Mc) (hN : G.NonDegenerate Mc)
    (hI : G.InnerHolds (Claims.Flock.R (deployZkTrim Readable.relation).parse) Mc)
    (hC : G.CoinCheck) (coinOf : Claims.Msg → Claims.Msg) (tapeOf : G.Nonce × G.Rand → FlockComplete.TapeZk)
    (hG : G.Honest sha512 FlockComplete.ZkTables.honestPR coinOf tapeOf) :
    Claims.Flock.ScopeMatches (deployZkTrim Readable.relation) ∧
      Claims.Flock.KS sha512 (deployZkTrim Readable.relation) ∧
      Claims.Flock.Bits sha512 (deployZkTrim Readable.relation) ∧
      (∀ a b : Claims.Msg → ℝ, (∀ x, Claims.Flock.κ (deployZkTrim Readable.relation) x ≤ 2 ^ (-a x)) →
        (∀ x, Claims.Flock.γ (deployZkTrim Readable.relation) x ≤ 2 ^ b x) →
        Claims.Bits sha512 (deployZkTrim Readable.relation).V (Claims.Flock.R (deployZkTrim Readable.relation).parse)
          (Claims.Flock.E (deployZkTrim Readable.relation) (deployZkTrim Readable.relation).decode)
          fun x => min (a x) (255.5 - b x / 2) - 1) ∧
      Claims.Flock.ZK sha512 (Claims.ZKMal.zkOf Readable.relation G coinOf tapeOf (G.simulatorE t (t * 3 ^ t)))
        (fun x => 8 * nL x * δ₁.toReal + 8 * t * nB x * δ₁.toReal + 2 * (1 / 3) ^ t) (fun _ => 4 * t * 3 ^ t) ∧
      Claims.Flock.Efficient sha512 (deployZkTrim Readable.relation) zkStepPoly ∧
      Claims.Flock.Complete sha512 (deployZkTrim Readable.relation) :=
  flock_zk_assembled_relation sha512 FlockDeploy.KS.zk_KS hPR hR hLI hMR G t ht hK My cs hT1 hδ₁ Mp Mc nL nB hnL hnB
    hL hS hN hI hC coinOf tapeOf hG

end FlockDeploy.Assembled
