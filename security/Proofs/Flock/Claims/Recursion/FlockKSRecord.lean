module

public import Proofs.Flock.Claims.Bridge.Hm96Tree
public import Proofs.Flock.Claims.KS.Trim
public import Proofs.Flock.Claims.Recursion.FlockKSAcc
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
import Proofs.Flock.Claims.KS.ConflictTied
import Proofs.Flock.Claims.KS.Deploy
import Proofs.Flock.Claims.KS.Range
import Proofs.Flock.Claims.KS.Width
import Proofs.Flock.Claims.Prob
import Proofs.Flock.Claims.Recursion.FlockKSDeploy
import Proofs.Flock.Claims.Recursion.InnerAbort
import Proofs.Flock.Claims.Recursion.KSStated

/-! Flock meets `Inner.KSAbort` at the deployed trimmed records with all but the table bound discharged
(`plain_KSFlock`, `zk_KSFlock`). -/

@[expose] public section

namespace Claims.Recursion

open Claims Claims.Flock

theorem tableBoundθOnAcc_mapCols {Coin Col T : Type} [Fintype Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]
    (H : Msg → Msg) (D : Flock.Deploy Coin Col T) (N : Msg → ℕ) (f : Col → Col) (hf : f 0 = 0)
    (Ok : Msg → Col → Prop) (hok : ColsOk H D N Ok) (hinj : ∀ x u v, Ok x u → Ok x v → f u = f v → u = v)
    (htab : TableBoundθOnAcc H D N Ok) : TableBoundθOnAcc H (D.mapCols f) N (OkThrough f Ok) := by
  intro x Ω _ _ P
  obtain ⟨e, hon, hfit, hb⟩ := htab x Ω P
  refine ⟨e, hon, hfit, fun ω C' hC' hg' => ?_⟩
  choose C hC hfC using hC'
  have hg : ¬ GoodTable D N x C := fun h => hg' (by
    have := goodTable_mapCols D N x f hf C h
    rwa [show (fun p => f (C p)) = C' from funext hfC] at this)
  refine (hb ω C hC hg).trans (add_le_add le_rfl (Pr_mono fun r ⟨ha, p, u, hu, hne⟩ => ⟨ha, p, f u, ?_, ?_⟩))
  · show lookCol ((D.cols (run H P D.V x ω r).seen 0).map fun q => (q.1, f q.2)) p.val = some (f u)
    rw [lookCol_map, hu]
    rfl
  · rw [← hfC p]
    exact fun he => hne (hinj x u (C p) (hok x Ω P ω r p.val u ha p.isLt hu) (hC p) he)

end Claims.Recursion

namespace FlockDeploy

open Claims.Recursion

noncomputable def decCoin (_ : Claims.Msg) : Claims.Msg → Coin := Function.invFun Coin.bytes

theorem coinsShown_V (H : Claims.Msg → Claims.Msg) (zk : Bool) : CoinsShown H (V zk) decCoin :=
  fun x as cs c m _ h => by
    rw [Claims.Bridge.answersCoin_deploy zk H x as cs c m h]
    exact Function.leftInverse_invFun Claims.Bridge.coin_bytes_injective c

end FlockDeploy

namespace FlockDeploy.KS

open Flock
open Claims.Recursion

theorem plain_kSplits (rel : Readable.Parsed → Readable.Relation)
    (hone : ∀ x, (∃ s, (deployPlain rel).parse x = some s ∧ s.tables = 1) ∨
      Claims.Flock.NoAcc sha512 (deployPlain rel).V x) (x : Claims.Msg) :
    (8 ≤ (deployPlainTrim rel).k x ∧ Claims.Flock.Splits (deployPlainTrim rel) x) ∨
      Claims.Flock.NoAcc sha512 (deployPlainTrim rel).V x :=
  (k_range_plain rel x).elim
    (fun hk => ((hone x).imp_left fun ⟨_, hs, h1⟩ => splits_of_one (deployPlainTrim rel) rfl rfl hs h1).imp
      (fun h => ⟨hk, h⟩) id) .inr

theorem zk_kSplits (rel : Readable.Parsed → Readable.Relation)
    (hone : ∀ x, (∃ s, (deployZk rel).parse x = some s ∧ s.tables = 1) ∨
      Claims.Flock.NoAcc sha512 (deployZk rel).V x) (x : Claims.Msg) :
    (8 ≤ (deployZkTrim rel).k x ∧ Claims.Flock.Splits (deployZkTrim rel) x) ∨
      Claims.Flock.NoAcc sha512 (deployZkTrim rel).V x :=
  (k_range_zk rel x).elim
    (fun hk => ((hone x).imp_left fun ⟨_, hs, h1⟩ => splits_of_one (deployZkTrim rel) rfl rfl hs h1).imp
      (fun h => ⟨hk, h⟩) id) .inr

theorem plain_KSAbort (rel : Readable.Parsed → Readable.Relation)
    (htab : TableBoundθOnAcc sha512 (deployPlain rel) (wordLen false) (wide false))
    (decode : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace)
    (hdec : decOK (deployPlainTrim rel) decode) : (Inner.ofFlock (deployPlainTrim rel) decode).KSAbort sha512 :=
  KSAbort_of_tableBoundOnAcc sha512 (deployPlainTrim rel) (wordLen false) (trimWide false) decCoin
    (wordLen_le_N₀ false) (fun x => εtable_nonneg false x) (V_k_pos false) (fun _ => cap_pos)
    (plain_kSplits rel (plain_hone rel)) (plain_hone rel) (trimWide_nil false)
    (Claims.Flock.colsOk_mapCols sha512 (deployPlain rel) (wordLen false) trimCol (wide false)
      (plain_colsOk sha512 rel))
    (Claims.Flock.conflictTied_mapCols sha512 (deployPlain rel) (wordLen false) trimCol (plain_conflictTied rel))
    (tableBoundθOnAcc_mapCols sha512 (deployPlain rel) (wordLen false) trimCol rfl (wide false)
      (plain_colsOk sha512 rel) (wide_trimCol_inj false) htab)
    (coinsShown_V sha512 false) (accLast_V_sha512 false) (decideFree_V sha512 false) (refusable_V sha512 false)
    decode hdec

theorem zk_KSAbort (rel : Readable.Parsed → Readable.Relation)
    (hone : ∀ x, (∃ s, (deployZk rel).parse x = some s ∧ s.tables = 1) ∨
      Claims.Flock.NoAcc sha512 (deployZk rel).V x)
    (htab : TableBoundθOnAcc sha512 (deployZk rel) (wordLen true) (wide true))
    (decode : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.ZkTrace)
    (hdec : decOK (deployZkTrim rel) decode) : (Inner.ofFlock (deployZkTrim rel) decode).KSAbort sha512 :=
  KSAbort_of_tableBoundOnAcc sha512 (deployZkTrim rel) (wordLen true) (trimWide true) decCoin
    (wordLen_le_N₀ true) (fun x => εtable_nonneg true x) (V_k_pos true) (fun _ => cap_pos)
    (zk_kSplits rel hone) hone (trimWide_nil true)
    (Claims.Flock.colsOk_mapCols sha512 (deployZk rel) (wordLen true) trimCol (wide true) (zk_colsOk sha512 rel))
    (Claims.Flock.conflictTied_mapCols sha512 (deployZk rel) (wordLen true) trimCol (zk_conflictTied rel))
    (tableBoundθOnAcc_mapCols sha512 (deployZk rel) (wordLen true) trimCol rfl (wide true)
      (zk_colsOk sha512 rel) (wide_trimCol_inj true) htab)
    (coinsShown_V sha512 true) (accLast_V_sha512 true) (decideFree_V sha512 true) (refusable_V sha512 true)
    decode hdec

theorem plain_KSFlock (rel : Readable.Parsed → Readable.Relation) (D : Deploy Coin)
    (DO : Claims.Flock.Deploy Coin (List FlockLevel3.GF128) Claims.Flock.PlainTrace)
    (opening : Claims.Recursion.Opening) (hF : Claims.Flock.KS sha512 DO)
    (htab : TableBoundθOnAcc sha512 (deployPlain rel) (wordLen false) (wide false)) (hRef : Refuses sha512 D)
    (hRounds : ∀ x, (deployPlainTrim rel).V.k (xI D x) ≤ D.V.k x) (hN : SessionNums D DO) (hε : TableErrNonneg D)
    (hRd : ∀ decI, SessionRounds D (Inner.ofFlock (deployPlainTrim rel) decI) DO) (hS : SessionAccepts D)
    (hR : ∀ decI, SessionRuns sha512 D (Inner.ofFlock (deployPlainTrim rel) decI) DO)
    (hV : ∀ decI, InnerViewRun sha512 D (Inner.ofFlock (deployPlainTrim rel) decI) DO)
    (hT : ∀ decI, CallsTrue sha512 D (Inner.ofFlock (deployPlainTrim rel) decI) DO)
    (hL : ∀ decI, CallsLogColl sha512 D (Inner.ofFlock (deployPlainTrim rel) decI) DO)
    (hO : ∀ decI, CallsOpenPlayed sha512 D (Inner.ofFlock (deployPlainTrim rel) decI) DO opening)
    (hAcc : ∀ decI, InnerAccepts sha512 D (Inner.ofFlock (deployPlainTrim rel) decI) DO) (hN0 : CallsNil sha512 D) :
    KSFlock sha512 D (deployPlainTrim rel) DO :=
  (KSFlockA_iff_KSFlock _ _ _ _).1 fun decI h => KS_of_flockKS sha512 D _ DO opening decCoin hF
    (plain_KSAbort rel htab decI h) hRef hRounds hN hε (hRd decI) hS (hR decI) (hV decI) (hT decI) (hL decI)
    (hO decI) (hAcc decI) (coinsShown_V sha512 false) hN0

theorem zk_KSFlock (rel : Readable.Parsed → Readable.Relation) (D : Deploy Coin)
    (DO : Claims.Flock.Deploy Coin (List FlockLevel3.GF128) Claims.Flock.ZkTrace)
    (opening : Claims.Recursion.Opening) (hF : Claims.Flock.KS sha512 DO)
    (hone : ∀ x, (∃ s, (deployZk rel).parse x = some s ∧ s.tables = 1) ∨
      Claims.Flock.NoAcc sha512 (deployZk rel).V x)
    (htab : TableBoundθOnAcc sha512 (deployZk rel) (wordLen true) (wide true)) (hRef : Refuses sha512 D)
    (hRounds : ∀ x, (deployZkTrim rel).V.k (xI D x) ≤ D.V.k x) (hN : SessionNums D DO) (hε : TableErrNonneg D)
    (hRd : ∀ decI, SessionRounds D (Inner.ofFlock (deployZkTrim rel) decI) DO) (hS : SessionAccepts D)
    (hR : ∀ decI, SessionRuns sha512 D (Inner.ofFlock (deployZkTrim rel) decI) DO)
    (hV : ∀ decI, InnerViewRun sha512 D (Inner.ofFlock (deployZkTrim rel) decI) DO)
    (hT : ∀ decI, CallsTrue sha512 D (Inner.ofFlock (deployZkTrim rel) decI) DO)
    (hL : ∀ decI, CallsLogColl sha512 D (Inner.ofFlock (deployZkTrim rel) decI) DO)
    (hO : ∀ decI, CallsOpenPlayed sha512 D (Inner.ofFlock (deployZkTrim rel) decI) DO opening)
    (hAcc : ∀ decI, InnerAccepts sha512 D (Inner.ofFlock (deployZkTrim rel) decI) DO) (hN0 : CallsNil sha512 D) :
    KSFlock sha512 D (deployZkTrim rel) DO :=
  (KSFlockA_iff_KSFlock _ _ _ _).1 fun decI h => KS_of_flockKS sha512 D _ DO opening decCoin hF
    (zk_KSAbort rel hone htab decI h) hRef hRounds hN hε (hRd decI) hS (hR decI) (hV decI) (hT decI) (hL decI)
    (hO decI) (hAcc decI) (coinsShown_V sha512 true) hN0

end FlockDeploy.KS
