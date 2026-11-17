module

public import Proofs.Flock.Claims.Recursion.DecodeAt
public import Proofs.Flock.Claims.Recursion.VerdictFramed
public import Proofs.Flock.Claims.Recursion.VerdictPublics
public import Proofs.Flock.Claims.Recursion.VerdictRowECoin
public import Proofs.Flock.Claims.Recursion.VerdictRowRDeployed
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
import Proofs.Flock.Claims.Recursion.RoundSizes
import Proofs.Flock.Claims.Recursion.VerdictLoop
import Proofs.Flock.Claims.Recursion.VerdictRowG
import Proofs.Flock.Claims.Recursion.VerdictTargetL

/-! `NonVacuous`'s row targets from what rows R, D and E leave open at the deployed reader. -/

@[expose] public section

namespace RecDeploy

open Flock
open Claims (Msg)
open FlockDeploy (Coin FinalP)
open FlockVBridge.PartS (wIdx)
open Flock.RecAlgebra (verifierRows structureOf el)
open Flock.RecResiduals (portSizes groups)
open FlockSoundness.Discharge.Recursion (HClass)
open RecDeploy.Rows (DecAt)

noncomputable section

namespace Top

section Targets

variable (rd : Msg → Option FinalP) (P : Msg → List Msg → List Coin → Prop)
  (acc : Msg → List Msg → List Coin → ℕ → Array F128)

def TargetRDep : Prop :=
  ∀ (x : Msg) (ms : List Msg) (cs : List Coin), At rd P x ms cs → ∀ (D : DecAt rd x ms cs) (cls : HClass) (i : ℕ),
    i < D.sts.size →
      (∀ r : Fin 2, (acc x ms cs (2 * i + r)).size = 4) ∧
      ∃ (rpieces : List ResPiece) (elemAt : Msg → List Msg → List Coin → ℕ × ℕ × ℕ × ℕ → ℕ → F128)
        (vvAt : Msg → List Coin → ℕ × ℕ × ℕ × ℕ → ℕ → F128) (keyAt : Fin 2 → ℕ → ℕ × ℕ × ℕ × ℕ),
        (∀ q ∈ rpieces, resHolds elemAt vvAt q x ms cs) ∧
        (∀ (S : RecAlgebra.GfS) (gs : Array (Array ℕ × Array ℕ × Array ℕ)),
          structureOf (D.sh cls i) = .ok S → groups S (portSizes (D.sh cls i)) = .ok gs →
          ∀ (r : Fin 2) p (_hp : p < gs.size), ∃ q ∈ rpieces, q.key = keyAt r p ∧
            ∃ ns nv, RecResiduals.part (D.sh cls i) p = .ok (q.S, ns, nv)) ∧
        (∀ (S : RecAlgebra.GfS) (gs : Array (Array ℕ × Array ℕ × Array ℕ)),
          structureOf (D.sh cls i) = .ok S → groups S (portSizes (D.sh cls i)) = .ok gs →
          ∀ (r : Fin 2) p (hp : p < gs.size) k (hk : k < (wIdx (portSizes (D.sh cls i)) gs[p].2.1).size),
            elemAt x ms cs (keyAt r p) k =
              el (D.rep cls (acc x ms cs) i r).w (wIdx (portSizes (D.sh cls i)) gs[p].2.1)[k]) ∧
        ∀ (S : RecAlgebra.GfS) (gs : Array (Array ℕ × Array ℕ × Array ℕ)),
          structureOf (D.sh cls i) = .ok S → groups S (portSizes (D.sh cls i)) = .ok gs →
          ∀ (r : Fin 2) v info kappa, verifierRows (D.sh cls i) (D.rep cls (acc x ms cs) i r).rounds
              (D.rep cls (acc x ms cs) i r).extras (D.rep cls (acc x ms cs) i r).comb = .ok (v, info, kappa) →
            ∀ p (hp : p < gs.size) k (hk : k < gs[p].2.2.size), vvAt x cs (keyAt r p) k = el v gs[p].2.2[k]

def TargetDDep : Prop :=
  ∀ (x : Msg) (ms : List Msg) (cs : List Coin), At rd P x ms cs → ∀ (D : DecAt rd x ms cs) (cls : HClass) (i : ℕ),
    i < D.sts.size →
      ∃ _key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ,
        ∀ r : Fin 2, ∀ v info kappa,
          verifierRows (D.sh cls i) (D.rep cls (acc x ms cs) i r).rounds (D.rep cls (acc x ms cs) i r).extras
            (D.rep cls (acc x ms cs) i r).comb = .ok (v, info, kappa) →
          ∀ ℓ < (D.files cls i).sch.levels.size, ((D.files cls i).sch.levels[ℓ]!).q ≤ (info.lg.conss[ℓ]!).pos.size ∧
            Rows.levelCheck (D.files cls i).sch ((D.files cls i).proof r).cap ((D.files cls i).proof r).ligerito ℓ
              (info.lg.conss[ℓ]!).pos = .ok ()

def TargetDMerkle : Prop :=
  ∀ (x : Msg) (ms : List Msg) (cs : List Coin), At rd P x ms cs → ∀ (D : DecAt rd x ms cs) (cls : HClass) (i : ℕ),
    i < D.sts.size → ∀ r : Fin 2, ∃ (cf : ℕ → ℕ → F128) (k : ℕ),
      FlockComplete.Replay.Lg.MerkleOk Flock.With.std FlockVBridge.Family.plain (D.files cls i).sch
        ((D.files cls i).proof r).cap ((D.files cls i).proof r).ligerito cf k ∧
      ∀ v info kappa,
        verifierRows (D.sh cls i) (D.rep cls (acc x ms cs) i r).rounds (D.rep cls (acc x ms cs) i r).extras
          (D.rep cls (acc x ms cs) i r).comb = .ok (v, info, kappa) →
        ∀ ℓ < (D.files cls i).sch.levels.size, (info.lg.conss[ℓ]!).pos =
          Flock.positions ((D.files cls i).sch.levels[ℓ]!) ((Array.range ((D.files cls i).sch.levels[ℓ]!).q).map
            (cf (k + FlockComplete.Replay.Lg.qOff (D.files cls i).sch ℓ)))

end Targets

variable {rd : Msg → Option FinalP} {P : Msg → List Msg → List Coin → Prop}
  {acc : Msg → List Msg → List Coin → ℕ → Array F128}

theorem targetR_of (h : TargetRDep rd P acc) : TargetR rd P acc := by
  intro x ms cs hA D cls i hi
  obtain ⟨⟨xo, sto, hst, hf, hx⟩, hr, hb, -⟩ := id hA
  obtain ⟨hacc, rpieces, elemAt, vvAt, keyAt, hres, hcoverR, hE, hvv⟩ := h x ms cs hA D cls i hi
  obtain ⟨S, gs, hS, hg⟩ := D.structure_ok cls hst hf hx hi D.points_eq
  have hG := D.rowG cls (acc x ms cs) hst hf hx hi
  exact ⟨S, gs, hS, hg, hsz acc hA D cls hi hS,
    fun r => D.hw_of_acc cls (acc x ms cs) i r hG D.points_eq (idx_lt D hi r)
      (shaped_of_base hr hb D _ (idx_lt D hi r)) hS (hacc r),
    fun r => D.partsZero_of_pieces cls (acc x ms cs) i r rpieces elemAt vvAt hres (keyAt r) (hcoverR S gs hS hg r)
      (hE S gs hS hg r) fun v info kappa hv => hvv S gs hS hg r v info kappa hv⟩

theorem targetD_of (h : TargetDDep rd P acc) : TargetD rd P acc := by
  intro x ms cs hA D cls i hi
  obtain ⟨⟨xo, sto, hst, hf, hx⟩, -, -, -⟩ := id hA
  have hti : i < D.spec.tables.size := by
    obtain ⟨h1, h2⟩ := D.tables_of_dense hst hf hx
    omega
  obtain ⟨S, gs, hS, -⟩ := D.structure_ok cls hst hf hx hi D.points_eq
  obtain ⟨key, hlev⟩ := h x ms cs hA D cls i hi
  exact D.rowD_target cls (acc x ms cs) key hi (Rows.sideAt rd) Rows.posAt
    (D.hside cls i hi (Rows.DecAt.rowDPiecesAt key i x))
    (D.hpos cls (acc x ms cs) i hst hf hx hi hti (Rows.DecAt.rowDPiecesAt key i x))
    (D.rowsRun_of cls (acc x ms cs) hti hS (hsz acc hA D cls hi hS)) hlev

theorem targetDDep_of_merkle (h : TargetDMerkle rd P acc) : TargetDDep rd P acc := by
  intro x ms cs hA D cls i hi
  obtain ⟨⟨xo, sto, hst, hf, hx⟩, -, -, -⟩ := id hA
  have hG := D.rowG cls (acc x ms cs) hst hf hx hi
  refine ⟨fun t r ℓ j => (t, r, ℓ, j), fun r => ?_⟩
  obtain ⟨cf, k, hM, htie⟩ := h x ms cs hA D cls i hi r
  exact D.levels_of_merkleOk cls (acc x ms cs) r Rows.plain_opsWith_std hG.fast hM htie

theorem targets_of_open (rd : Msg → Option FinalP) {Q : Type} (rows : Msg → List Q)
    (holds : Q → Msg → List Msg → List Coin → Prop) (acc : Msg → List Msg → List Coin → ℕ → Array F128)
    (hR : TargetRDep rd (RowsHold rows holds) acc) (hD : TargetDMerkle rd (RowsHold rows holds) acc)
    (hE : TargetE rd (RowsHold rows holds) acc) (hDec : TargetDecode rd (RowsHold rows holds)) :
    Targets rd rows holds acc :=
  ⟨fun x ms cs hA D cls _ hi r =>
      let ⟨_, _, hst, hf, hx⟩ := hA.1
      D.framed_dense cls (acc x ms cs) r hst hf hx hi,
    targetL_holds acc, targetD_of (targetDDep_of_merkle hD), targetR_of hR, hE, hDec,
    fun _ _ _ _ D => D.publics_empty⟩

theorem targets_of_rows (rd : Msg → Option FinalP) {Q : Type} (rows : Msg → List Q)
    (holds : Q → Msg → List Msg → List Coin → Prop) (chainAt : Msg → List Msg → ℕ → ℕ → Array F128)
    (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ)
    (hR : TargetRDep rd (RowsHold rows holds) (accR rd chainAt))
    (hD : TargetDMerkle rd (RowsHold rows holds) (accR rd chainAt))
    (hholds : ∀ (x : Msg) (ms : List Msg) (cs : List Coin), RowsHold rows holds x ms cs →
      ∀ q ∈ Rows.rowDPieces key x, Rows.accHolds (Rows.sideAt rd) Rows.coefRead chainAt q x ms cs)
    (hDec : TargetDecode rd (RowsHold rows holds)) :
    Targets rd rows holds (accR rd chainAt) :=
  targets_of_open rd rows holds _ hR hD (targetE_rowD rd _ chainAt key hholds) hDec

theorem targets_of_rowsJ (rd : Msg → Option FinalP) {Q : Type}
    (rows : Msg → List Q) (holds : Q → Msg → List Msg → List Coin → Prop)
    (chainAt : Msg → List Msg → ℕ → ℕ → Array F128) (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ)
    (hR : TargetRDep rd (RowsHold rows holds) (accR rd chainAt))
    (hD : TargetDMerkle rd (RowsHold rows holds) (accR rd chainAt))
    (hholds : ∀ (x : Msg) (ms : List Msg) (cs : List Coin), RowsHold rows holds x ms cs →
      ∀ q ∈ Rows.rowDPieces key x, Rows.accHolds (Rows.sideAt rd) Rows.coefRead chainAt q x ms cs) :
    Targets rd rows holds (accR rd chainAt) :=
  targets_of_rows rd rows holds chainAt key hR hD hholds (DecodeAt.decode_at rd (RowsHold rows holds))

end Top

end

end RecDeploy
